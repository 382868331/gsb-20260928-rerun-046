"""实际输入改变后的恢复语义。

- 图结构不变但输入改变: 不能复用依赖旧输入的节点, 结果必须对应新输入;
- 多个共享输入只改变一部分: 只失效受影响源块的祖先, 其他分支保留;
- 恢复前提供非破坏性影响预览, 用户取消时旧记录原封不动;
- 不同 worker 数 + 块损坏 + 计算中途退出叠加, 最终结果仍与当前输入的
  完整重算逐位一致。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from sciruntime import (
    CheckpointStore,
    Executor,
    ResumeCancelled,
    fingerprint_array,
)
from sciruntime.matrix import (
    assemble_result,
    build_blocked_matmul_graph,
    serial_reference_matmul,
)
from sciruntime.workflow import build_resume_graph


WORKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "matmul_resume_worker.py",
)


def _matmul(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((48, 32))
    b = np.random.default_rng(seed + 1000).standard_normal((32, 40))
    return a, b


def _graph(a, b):
    g, outs = build_blocked_matmul_graph(
        a.shape, b.shape, row_block=16, k_block=32, col_block=20
    )
    return g, outs


def _run(graph, outs, ck, a, b, *, workers=2, decider=None):
    ex = Executor(graph, max_workers=workers, checkpoint=ck,
                  resume_decider=decider)
    stats = ex.run(feed={"__src_A": a, "__src_B": b})
    return stats, assemble_result(ex.store, outs, a.shape[0], n_cols_blocks=2)


def test_same_graph_changed_input_recomputes_all_and_new_result(tmp_path):
    d = str(tmp_path)
    a0, b0 = _matmul(1)
    g0, outs0 = _graph(a0, b0)
    s0, c0 = _run(g0, outs0, CheckpointStore(d), a0, b0)
    assert s0.ok

    # 同构新图、不同实际输入 (仅 A 改变, B 完全相同)
    a1, b1 = _matmul(2)
    b1 = b0.copy()
    g1, outs1 = _graph(a1, b1)
    plans = []
    s1, c1 = _run(g1, outs1, CheckpointStore(d), a1, b1,
                  decider=lambda p: (plans.append(p), True)[1])
    # A 改变: 依赖 A 的节点全部重算; 纯 B 切分分支不依赖 A, 保留复用。
    reused = set(plans[0].reusable_nodes)
    assert "split_B_cols" in reused
    assert "split_A_rows" not in reused
    assert all(nid not in reused for nid in g1.nodes
               if nid.startswith("gemm_") or nid.startswith("reduce_"))
    assert len(plans[0].recompute) + len(plans[0].reusable) == len(g1.nodes)
    assert set(plans[0].changed_inputs) == {"__src_A"}
    assert "split_A_rows" not in set(s1.from_checkpoint)
    assert "split_B_cols" in set(s1.from_checkpoint)
    assert not np.array_equal(c1, c0)
    assert np.array_equal(c1, serial_reference_matmul(a1, b1, k_block=32))


def test_change_one_source_preserves_unaffected_branch(tmp_path):
    d = str(tmp_path)
    a0, b0 = _matmul(7)
    g0, outs0 = _graph(a0, b0)
    s0, _ = _run(g0, outs0, CheckpointStore(d), a0, b0, workers=3)
    assert s0.ok

    # 只改变 B: 只依赖 A 的切分链 (split_A_rows / split_A_*_cols) 保留,
    # 依赖 B 的节点以及每个 gemm/reduce 都必须重算。
    a1, b1 = a0.copy(), _matmul(8)[1]
    g1, outs1 = _graph(a1, b1)
    ck = CheckpointStore(d)
    plan = ck.plan_for_graph(
        g1.topological_order(),
        {
            nid: {g1.producer_of(b) for b in node.inputs} - {None}
            for nid, node in g1.nodes.items()
        },
        node_sources=g1.node_source_dependencies(),
        input_fingerprints={
            "__src_A": fingerprint_array(a1),
            "__src_B": fingerprint_array(b1),
        },
    )
    reusable = set(plan.reusable_nodes)
    assert "split_A_rows" in reusable
    assert all(nid in reusable for nid in g1.nodes
               if nid.startswith("split_A_") and nid != "split_A_rows")
    assert "split_B_cols" not in reusable
    assert all(nid not in reusable for nid in g1.nodes
               if nid.startswith("gemm_") or nid.startswith("reduce_"))
    reason = dict((i.node_id, i.reason) for i in plan.recompute)
    assert "split_B_cols" in reason
    assert "输入" in reason["split_B_cols"]

    s1, c1 = _run(g1, outs1, CheckpointStore(d), a1, b1, workers=4)
    assert "split_A_rows" in s1.from_checkpoint
    assert "split_B_cols" in s1.recomputed
    assert np.array_equal(c1, serial_reference_matmul(a1, b1, k_block=32))


def test_preview_is_non_destructive_and_cancel_keeps_old_record(tmp_path):
    d = str(tmp_path)
    x0 = np.random.default_rng(0).standard_normal(64)
    g0, meta0 = build_resume_graph(64, n_branches=4)
    ex = Executor(g0, max_workers=2, checkpoint=CheckpointStore(d))
    s = ex.run(feed={meta0["source"]: x0})
    assert s.ok
    old_meta = open(os.path.join(d, "meta.json"), encoding="utf-8").read()
    old_blocks = set(os.listdir(os.path.join(d, "blocks")))

    x1 = np.random.default_rng(1).standard_normal(64)
    g1, meta1 = build_resume_graph(64, n_branches=4)
    ck = CheckpointStore(d)
    plan = ck.plan_for_graph(
        g1.topological_order(),
        {
            nid: {g1.producer_of(b) for b in node.inputs} - {None}
            for nid, node in g1.nodes.items()
        },
        node_sources=g1.node_source_dependencies(),
        input_fingerprints={meta1["source"]: fingerprint_array(x1)},
    )
    # 预览后磁盘记录完全不变
    assert open(os.path.join(d, "meta.json"), encoding="utf-8").read() == old_meta
    assert set(os.listdir(os.path.join(d, "blocks"))) == old_blocks
    assert plan.changed_inputs == [meta1["source"]]
    assert not plan.reusable
    assert "输入" in plan.describe()

    # 通过 Executor 取消: 抛 ResumeCancelled, 旧记录依旧原样
    with pytest.raises(ResumeCancelled):
        _cancel_run(g1, meta1, d, x1)
    assert open(os.path.join(d, "meta.json"), encoding="utf-8").read() == old_meta
    assert set(os.listdir(os.path.join(d, "blocks"))) == old_blocks
    # 旧检查点再次用旧输入打开, 仍然可以全部复用并得到旧结果
    g2, meta2 = build_resume_graph(64, n_branches=4)
    ex2 = Executor(g2, max_workers=1, checkpoint=CheckpointStore(d))
    s2 = ex2.run(feed={meta2["source"]: x0})
    assert len(s2.from_checkpoint) == len(g2.nodes)


def _cancel_run(graph, meta, d, x):
    ex = Executor(graph, max_workers=2, checkpoint=CheckpointStore(d),
                  resume_decider=lambda _plan: False)
    return ex.run(feed={meta["source"]: x})


def test_changed_workers_corruption_and_midrun_exit_combined(tmp_path):
    d = str(tmp_path)

    def invoke(cfg):
        cfg = dict(cfg, ckpt=d)
        proc = subprocess.run(
            [sys.executable, WORKER, json.dumps(cfg)],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
        )
        info = None
        for line in proc.stdout.splitlines():
            if line.startswith("WORKER_JSON "):
                info = json.loads(line.split(" ", 1)[1])
        return proc.returncode, info, proc.stderr

    # 旧输入完整跑一部分后硬退出 (43), workers=1
    rc, _, err = invoke({"seed": 11, "workers": 1,
                         "exit_after_commits": 3})
    assert rc == 43, err

    # 换成新输入 (seed=12) 并改 workers=3, 在重算中途再硬退出 (43)
    rc, _, err = invoke({"seed": 12, "workers": 3,
                         "exit_after_commits": 4})
    assert rc == 43, err

    # 篡改一个已提交的 B 切分块 (B_col_*.npy): 该块及其后继失效,
    # A 分支的提交不受影响。
    block_files = [
        f for f in os.listdir(os.path.join(d, "blocks"))
        if f.startswith("B_col_") and f.endswith(".npy")
    ]
    assert block_files
    with open(os.path.join(d, "blocks", block_files[0]), "r+b") as f:
        f.seek(0)
        f.write(b"\x00\x00")

    # workers=2 重开完成, 结果必须与新输入的串行参考逐位一致
    rc, info, err = invoke({"seed": 12, "workers": 2})
    assert rc == 0, err
    assert info["ok"], err
    assert info["result_equal"], err
    reused = set(info["from_checkpoint"])
    # A 切分链不依赖被改的 B, 且未损坏, 必须保留
    assert "split_A_rows" in reused
    # 被损坏的 B 块与 gemm/reduce 全部重算
    assert "split_B_cols" in set(info["recomputed"])
    assert all(nid in set(info["recomputed"])
               for nid in ("reduce_0_0", "reduce_2_1"))

    # 再次同输入重开: 全部节点都应可信复用
    rc, info, err = invoke({"seed": 12, "workers": 5})
    assert rc == 0, err
    a, b = _matmul(12)
    assert len(info["from_checkpoint"]) == len(_graph(a, b)[0].nodes)
    assert not info["recomputed"]
    assert info["result_equal"]
