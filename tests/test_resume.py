"""细粒度接续语义测试。

- 只复用“完整发布 + 校验和核验 + 前驱全部可信”的节点;
- 损坏块 / 半成品节点及其全部后继必须重算;
- 图身份不一致拒绝复用;
- 恢复时改变 worker 数不影响结果与任务身份;
- 失败节点的输出对后继不可见 (已有 test_failure_*, 这里补检查点组合)。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from sciruntime import CheckpointStore, Executor
from sciruntime.checkpoint import CheckpointMismatch
from sciruntime.workflow import build_resume_graph, serial_reference
from examples.resume_workflow_demo import run_once


def _x():
    return np.random.default_rng(2026).standard_normal(64)


def _final(ckpt_dir, x, *, workers, budget=8192):
    rc, info, err = run_once(ckpt_dir, workers=workers,
                             memory_budget=budget, slow=False)
    assert rc == 0, err
    return info


def test_partial_resume_reuses_only_verified(tmp_path):
    """跑 2 个提交后硬退出, 换 worker 数重开: 共享节点复用, 其余重算。"""
    d = str(tmp_path)
    rc, _, err = run_once(d, workers=1, memory_budget=8192,
                          exit_after_commits=2, slow=False)
    assert rc == 43, err
    info = _final(d, _x(), workers=3, budget=8192)
    assert info["ok"]
    assert set(info["from_checkpoint"]) == {"shared0", "shared1"}
    assert "slow_0" in info["recomputed"]
    assert "reduce" in info["recomputed"]
    assert info["result_equal"]


def test_corrupt_block_invalidates_descendants(tmp_path):
    """损坏一个已提交块: 该节点及其全部后继重算, 结果仍逐位一致。"""
    d = str(tmp_path)
    x = _x()
    rc, _, err = run_once(d, workers=1, memory_budget=8192,
                          exit_after_commits=2, slow=False)
    assert rc == 43, err

    files = [os.path.join(d, "blocks", f)
             for f in os.listdir(os.path.join(d, "blocks"))]
    assert files
    with open(files[0], "r+b") as f:
        f.seek(0)
        f.write(b"\x00\x00")

    info = _final(d, x, workers=2)
    assert "reduce" in info["recomputed"]
    assert info["result_equal"]
    # 被损坏的块文件已被丢弃重写, 盘上只留下通过核验的提交
    ck = CheckpointStore(d)
    for nid in ("shared0", "shared1", "reduce"):
        assert ck.is_completed(nid)


def test_half_written_block_is_never_reused(tmp_path):
    """块 rename 之后、meta 更新之前崩溃: 孤儿块不允许任何后继读到。"""
    d = str(tmp_path)
    x = _x()
    rc, _, err = run_once(d, workers=1, memory_budget=8192,
                          exit_after_commits=2, slow=False)
    assert rc == 43, err
    rc, _, err = run_once(d, workers=2, memory_budget=8192,
                          crash_after_block="b0", slow=False)
    assert rc == 42, err

    ck2 = CheckpointStore(d)
    assert "slow_0" not in ck2.completed_nodes()
    info = _final(d, x, workers=4)
    assert "slow_0" in info["recomputed"]
    assert info["result_equal"]


def test_graph_identity_mismatch_rejected(tmp_path):
    rc, _, err = run_once(str(tmp_path), workers=1, memory_budget=8192,
                          exit_after_commits=1, slow=False)
    assert rc == 43, err
    g_other, _ = build_resume_graph(64, n_branches=3)
    ck = CheckpointStore(str(tmp_path))
    with pytest.raises(CheckpointMismatch):
        ck.attach_graph(g_other.identity())


def test_peak_inflight_respects_budget(tmp_path):
    info = _final(str(tmp_path), _x(), workers=2, budget=4096)
    assert info["ok"]
    assert info["peak_inflight_bytes"] <= 4096
