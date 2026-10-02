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

from sciruntime import CheckpointStore, Executor, Graph
from sciruntime.checkpoint import CheckpointMismatch
from sciruntime.operators import register_operator
from sciruntime.workflow import build_resume_graph, serial_reference
from examples.resume_workflow_demo import run_once


class AlwaysFail:
    def run(self, inputs, *, params, ctx):
        raise RuntimeError("revision branch failed")


def _build_revision_old_graph():
    g = Graph()
    g.add_node("a", "wf_scale", inputs=("x",), outputs=("a",),
               params={"scale": 2.0, "bias": 0.0})
    g.add_node("b", "wf_leaf", inputs=("a",), outputs=("b"),
               params={"scale": 3.0})
    g.add_node("c", "wf_leaf", inputs=("b",), outputs=("result",),
               params={"scale": 5.0})
    g.add_node("d", "wf_scale", inputs=("x",), outputs=("d",),
               params={"scale": 7.0, "bias": 1.0})
    g.add_node("e", "wf_leaf", inputs=("d",), outputs=("independent",),
               params={"scale": -1.0})
    g.freeze()
    return g


def _build_insert_graph():
    g = Graph()
    g.add_node("a", "wf_scale", inputs=("x",), outputs=("a",),
               params={"scale": 2.0, "bias": 0.0})
    g.add_node("correction", "wf_leaf", inputs=("a",),
               outputs=("fixed",), params={"scale": 1.1})
    g.add_node("b", "wf_leaf", inputs=("fixed",), outputs=("b"),
               params={"scale": 3.0})
    g.add_node("c", "wf_leaf", inputs=("b",), outputs=("result",),
               params={"scale": 5.0})
    g.add_node("d", "wf_scale", inputs=("x",), outputs=("d",),
               params={"scale": 7.0, "bias": 1.0})
    g.add_node("e", "wf_leaf", inputs=("d",), outputs=("independent",),
               params={"scale": -1.0})
    g.freeze()
    return g


def _run_full(graph, ckpt_path=None):
    x = np.arange(8.0)
    ck = CheckpointStore(ckpt_path) if ckpt_path else None
    ex = Executor(graph, checkpoint=ck, max_workers=2)
    stats = ex.run({"x": x})
    assert stats.ok, stats.failed
    return ex, x, stats


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


def test_changed_input_same_graph_forces_full_recompute(tmp_path):
    """图结构完全不变但输入数组改变: 旧节点一个都不能复用。"""
    d = str(tmp_path)
    rc, _, err = run_once(d, workers=2, memory_budget=8192,
                          exit_after_commits=2, slow=False)
    assert rc == 43, err
    x_new = np.random.default_rng(99).standard_normal(64)
    rc, info, err = run_once(d, workers=3, memory_budget=8192,
                             seed=99, slow=False)
    assert rc == 0, err
    assert info["ok"]
    assert info["from_checkpoint"] == [], info
    g, _ = build_resume_graph(64)
    assert set(info["recomputed"]) == set(g.nodes), info
    # 最终结果必须对应新输入, 而不是旧输入的缓存
    rc, _, err = run_once(d, workers=1, memory_budget=8192,
                          seed=99, slow=False)
    assert rc == 0, err
    g2, meta = build_resume_graph(64)
    ck = CheckpointStore(d)
    ex = Executor(g2, max_workers=2, checkpoint=ck)
    ex.run(feed={meta["source"]: x_new})
    assert np.array_equal(ex.result(meta["output"]), serial_reference(x_new))
    # 且不能等于旧输入的结果
    assert not np.array_equal(ex.result(meta["output"]), serial_reference(_x()))


def test_preview_is_readonly_and_cancellable(tmp_path):
    """恢复前影响预览不改变旧记录; 用户可以取消接续。"""
    d = str(tmp_path)
    rc, _, err = run_once(d, workers=1, memory_budget=8192,
                          exit_after_commits=2, slow=False)
    assert rc == 43, err
    old = CheckpointStore(d)
    committed_before = set(old.completed_nodes())
    blocks_before = sorted(os.listdir(os.path.join(d, "blocks")))

    g, meta = build_resume_graph(64)
    x_new = np.random.default_rng(7).standard_normal(64)
    plan = old.plan_recovery(g, {meta["source"]: x_new})
    text = plan.describe()
    assert "shared0" in plan.recompute and "shared1" in plan.recompute
    assert plan.reusable == set()
    assert "必须重算" in text and "输入" in text

    # 不调用 apply_recovery / 不运行 Executor: 旧 meta 与块原样保留
    after = CheckpointStore(d)
    assert set(after.completed_nodes()) == committed_before
    assert sorted(os.listdir(os.path.join(d, "blocks"))) == blocks_before


def test_partial_input_change_keeps_unaffected_branch(tmp_path):
    """双源图只改一个源: 未依赖它的分支保留, 依赖它的分支与 fan-in 重算。"""
    d = str(tmp_path)
    rc, info, err = run_once(d, workers=2, memory_budget=8192,
                             exit_after_commits=4, slow=False,
                             two_source=True)
    assert rc == 43, err
    rc, info, err = run_once(d, workers=4, memory_budget=8192,
                             seed=123, slow=False, two_source=True)
    assert rc == 0, err
    assert info["ok"] and info["result_equal"]
    assert set(info["from_checkpoint"]) == {"pb", "lb"}, info
    assert set(info["recomputed"]) == {"pa", "la", "join"}, info


def test_corruption_crash_and_changed_input_combined(tmp_path):
    """换 worker + 结果损坏 + 中途退出 + 输入改变同时出现仍与完整重算一致。"""
    d = str(tmp_path)
    rc, _, err = run_once(d, workers=4, memory_budget=8192,
                          exit_after_commits=3, slow=False,
                          two_source=True)
    assert rc == 43, err
    # 损坏 b 分支的输出块, 同时改变 a 源; worker 数从 4 换成 2。
    rc, info, err = run_once(d, workers=2, memory_budget=8192,
                             seed=555, slow=False, two_source=True,
                             corrupt_block="b1.npy")
    assert rc == 0, err
    assert info["ok"] and info["result_equal"], err
    assert set(info["recomputed"]) == {"pa", "la", "pb", "lb", "join"}, info


def test_revision_preview_separates_input_and_relation_changes(tmp_path):
    old = _build_revision_old_graph()
    old_ckpt = str(tmp_path / "old")
    _, x, _ = _run_full(old, old_ckpt)

    new = _build_insert_graph()
    store = CheckpointStore(old_ckpt)
    plan = store.plan_revision(new, old, {"x": x})
    text = plan.describe()

    assert plan.graph_match is False
    assert plan.reusable == {"a", "d", "e"}
    assert plan.pending == {"correction", "b", "c"}
    assert plan.added_nodes == {"correction"}
    assert "correction" in plan.recompute
    assert plan.recompute["b"].startswith("input_edge_changed:")
    assert plan.recompute["c"].startswith("dependency:")
    assert plan.changed_sources == {}
    assert set(plan.changed_relations) == {"b", "correction"}
    assert "输入变化:" not in text
    assert "计算关系变化:" in text

    reopened = CheckpointStore(old_ckpt)
    assert set(reopened.completed_nodes()) == {"a", "b", "c", "d", "e"}


def test_revision_inserts_correction_and_executes_derived_checkpoint(tmp_path):
    old = _build_revision_old_graph()
    old_ckpt = str(tmp_path / "old")
    old_ex, x, _ = _run_full(old, old_ckpt)

    new = _build_insert_graph()
    source = CheckpointStore(old_ckpt)
    plan = source.plan_revision(new, old, {"x": x})
    new_ckpt = source.materialize_revision(
        str(tmp_path / "insert"), new, old, plan, {"x": x}
    )
    ex = Executor(new, max_workers=3, checkpoint=new_ckpt)
    stats = ex.run({"x": x})

    assert stats.ok, stats.failed
    assert set(stats.from_checkpoint) == {"a", "d", "e"}
    assert set(stats.recomputed) == {"correction", "b", "c"}
    expected = (x * 2.0 * 1.1 * 3.0 * 5.0)
    assert np.array_equal(ex.result("result"), expected)
    assert np.array_equal(ex.result("independent"), old_ex.result("independent"))

    finished = CheckpointStore(str(tmp_path / "insert"))
    assert finished.provenance("a")["graph_id"] == old.identity()
    assert finished.provenance("a")["node_id"] == "a"
    assert finished.provenance("correction")["graph_id"] == new.identity()
    original = CheckpointStore(old_ckpt)
    assert set(original.completed_nodes()) == {"a", "b", "c", "d", "e"}


def test_revision_deletes_middle_node_and_reevaluates_new_edges(tmp_path):
    old = _build_revision_old_graph()
    old_ckpt = str(tmp_path / "old")
    _, x, _ = _run_full(old, old_ckpt)

    g = Graph()
    g.add_node("a", "wf_scale", inputs=("x",), outputs=("a",),
               params={"scale": 2.0, "bias": 0.0})
    g.add_node("c", "wf_leaf", inputs=("a",), outputs=("result",),
               params={"scale": 5.0})
    g.add_node("d", "wf_scale", inputs=("x",), outputs=("d",),
               params={"scale": 7.0, "bias": 1.0})
    g.add_node("e", "wf_leaf", inputs=("d",), outputs=("independent",),
               params={"scale": -1.0})
    g.freeze()

    source = CheckpointStore(old_ckpt)
    plan = source.plan_revision(g, old, {"x": x})
    assert plan.removed_nodes == {"b"}
    assert plan.reusable == {"a", "d", "e"}
    assert plan.recompute["c"].startswith("input_edge_changed:")
    candidate = source.materialize_revision(
        str(tmp_path / "deleted"), g, old, plan, {"x": x}
    )
    ex = Executor(g, max_workers=1, checkpoint=candidate)
    stats = ex.run({"x": x})
    assert stats.ok, stats.failed
    assert set(stats.from_checkpoint) == {"a", "d", "e"}
    assert set(stats.recomputed) == {"c"}
    assert np.array_equal(ex.result("result"), x * 2.0 * 5.0)


def test_failed_revision_does_not_overwrite_other_candidate_or_original(tmp_path):
    register_operator("revision_boom", AlwaysFail())
    old = _build_revision_old_graph()
    old_ckpt = str(tmp_path / "old")
    _, x, _ = _run_full(old, old_ckpt)
    good = _build_insert_graph()

    bad = Graph()
    bad.add_node("a", "wf_scale", inputs=("x",), outputs=("a",),
                 params={"scale": 2.0, "bias": 0.0})
    bad.add_node("bad", "revision_boom", inputs=("a",), outputs=("bad",))
    bad.add_node("result", "wf_leaf", inputs=("bad",),
                 outputs=("result",), params={"scale": 1.0})
    bad.freeze()

    source = CheckpointStore(old_ckpt)
    bad_plan = source.plan_revision(bad, old, {"x": x})
    bad_ckpt = source.materialize_revision(
        str(tmp_path / "bad"), bad, old, bad_plan, {"x": x}
    )
    bad_stats = Executor(bad, max_workers=1, checkpoint=bad_ckpt).run({"x": x})
    assert not bad_stats.ok
    assert "bad" in bad_stats.failed

    good_plan = source.plan_revision(good, old, {"x": x})
    good_ckpt = source.materialize_revision(
        str(tmp_path / "good"), good, old, good_plan, {"x": x}
    )
    good_stats = Executor(good, max_workers=4, checkpoint=good_ckpt).run({"x": x})
    assert good_stats.ok, good_stats.failed
    assert set(good_stats.from_checkpoint) == {"a", "d", "e"}
    bad_after = CheckpointStore(str(tmp_path / "bad"))
    good_after = CheckpointStore(str(tmp_path / "good"))
    old_after = CheckpointStore(old_ckpt)
    assert "bad" not in bad_after.completed_nodes()
    assert "result" not in bad_after.completed_nodes()
    assert set(good_after.completed_nodes()) == set(good.nodes)
    assert set(old_after.completed_nodes()) == set(old.nodes)
