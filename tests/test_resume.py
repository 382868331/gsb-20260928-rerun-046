"""细粒度接续语义测试。

- 只复用“完整发布 + 校验和核验 + 前驱全部可信”的节点;
- 损坏块 / 半成品节点及其全部后继必须重算;
- 图身份不一致拒绝复用;
- 恢复时改变 worker 数不影响结果与任务身份;
- 失败节点的输出对后继不可见 (已有 test_failure_*, 这里补检查点组合)。
"""
from __future__ import annotations

import os
import json
import subprocess
import sys

import numpy as np
import pytest

from sciruntime import CheckpointStore, Executor
from sciruntime.graph import Graph
from sciruntime.operators import Operator, register_operator, operator_registry
from sciruntime.checkpoint import CheckpointMismatch
from sciruntime.workflow import build_resume_graph, serial_reference
from examples.resume_workflow_demo import run_once


class FailOp(Operator):
    def run(self, inputs, *, params, ctx):
        raise RuntimeError("new branch failure")


def _register_failure_op():
    if "test_fail_op" not in operator_registry:
        register_operator("test_fail_op", FailOp())


def _evolution_base_graph():
    g = Graph()
    g.add_node("base", "wf_scale", inputs=("__src_x",), outputs=("a1",),
               params={"scale": 2.0, "bias": 1.0})
    g.add_node("indep", "wf_scale", inputs=("__src_x",), outputs=("u1",),
               params={"scale": 3.0, "bias": -2.0})
    g.add_node("join", "wf_reduce", inputs=("a1", "u1"), outputs=("result",))
    g.freeze()
    return g


def _insert_graph():
    g = Graph()
    g.add_node("base", "wf_scale", inputs=("__src_x",), outputs=("a1",),
               params={"scale": 2.0, "bias": 1.0})
    g.add_node("indep", "wf_scale", inputs=("__src_x",), outputs=("u1",),
               params={"scale": 3.0, "bias": -2.0})
    g.add_node("correct", "wf_scale", inputs=("a1",), outputs=("c1",),
               params={"scale": 1.5, "bias": 0.25})
    g.add_node("join", "wf_reduce", inputs=("c1", "u1"), outputs=("result",))
    g.freeze()
    return g


def _replace_graph():
    g = Graph()
    g.add_node("base", "wf_scale", inputs=("__src_x",), outputs=("a1",),
               params={"scale": -1.0, "bias": 4.0})
    g.add_node("indep", "wf_scale", inputs=("__src_x",), outputs=("u1",),
               params={"scale": 3.0, "bias": -2.0})
    g.add_node("join", "wf_reduce", inputs=("a1", "u1"), outputs=("result",))
    g.freeze()
    return g


def _rewire_graph():
    g = Graph()
    g.add_node("indep", "wf_scale", inputs=("__src_x",), outputs=("u1",),
               params={"scale": 3.0, "bias": -2.0})
    g.add_node("join", "wf_reduce", inputs=("__src_x", "u1"),
               outputs=("result",))
    g.freeze()
    return g


def _failure_graph():
    g = Graph()
    g.add_node("base", "wf_scale", inputs=("__src_x",), outputs=("a1",),
               params={"scale": 2.0, "bias": 1.0})
    g.add_node("indep", "wf_scale", inputs=("__src_x",), outputs=("u1",),
               params={"scale": 3.0, "bias": -2.0})
    g.add_node("bad", "test_fail_op", inputs=("a1",), outputs=("bad1",))
    g.freeze()
    return g


def _prepare_evolution_checkpoint(path, x):
    ck = CheckpointStore(path)
    ex = Executor(_evolution_base_graph(), max_workers=1, checkpoint=ck)
    stats = ex.run(feed={"__src_x": x})
    assert stats.ok
    return ck


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


def test_graph_evolution_previews_separate_input_and_relation_changes(tmp_path):
    d = str(tmp_path)
    x = _x()
    parent = _prepare_evolution_checkpoint(d, x)

    plan_insert = parent.plan_recovery(_insert_graph(), {"__src_x": x})
    assert plan_insert.reusable == {"base", "indep"}
    assert plan_insert.new_nodes == {"correct"}
    assert plan_insert.recompute["join"].startswith("graph_changed:")
    assert plan_insert.changed_sources == {}
    assert set(plan_insert.changed_relations) == {"join"}
    assert plan_insert.to_execute == {"correct", "join"}
    assert "变化的输入" not in plan_insert.describe()
    assert "计算关系变化" in plan_insert.describe()

    plan_replace = parent.plan_recovery(_replace_graph(), {"__src_x": x})
    assert plan_replace.changed_sources == {}
    assert plan_replace.recompute["base"] == "graph_changed:node_recipe"
    assert plan_replace.recompute["join"].startswith("graph_changed:dependency:base")
    assert plan_replace.reusable == {"indep"}

    plan_rewire = parent.plan_recovery(_rewire_graph(), {"__src_x": x})
    assert plan_rewire.removed_nodes == {"base"}
    assert plan_rewire.reusable == {"indep"}
    assert plan_rewire.recompute["join"] == "graph_changed:inputs"

    x_new = np.random.default_rng(101).standard_normal(64)
    plan_input = parent.plan_recovery(_insert_graph(), {"__src_x": x_new})
    assert plan_input.changed_sources == {"__src_x": "input_changed"}
    assert set(plan_input.changed_relations) == {"join"}


def test_multiple_derived_graphs_share_parent_without_mutating_it(tmp_path):
    d = str(tmp_path)
    x = _x()
    parent_path = os.path.join(d, "parent")
    _prepare_evolution_checkpoint(parent_path, x)

    parent_before = CheckpointStore(parent_path)
    before_nodes = set(parent_before.completed_nodes())

    insert_plan = parent_before.plan_recovery(_insert_graph(), {"__src_x": x})
    replace_plan = parent_before.plan_recovery(_replace_graph(), {"__src_x": x})
    assert insert_plan.to_execute == {"correct", "join"}
    assert replace_plan.to_execute == {"base", "join"}

    insert_ck, _ = CheckpointStore.derive(
        parent_path, os.path.join(d, "insert"), _insert_graph(), {"__src_x": x},
        branch_id="insert",
    )
    replace_ck, _ = CheckpointStore.derive(
        parent_path, os.path.join(d, "replace"), _replace_graph(), {"__src_x": x},
        branch_id="replace",
    )

    insert_ex = Executor(_insert_graph(), max_workers=3, checkpoint=insert_ck)
    insert_stats = insert_ex.run(feed={"__src_x": x})
    assert insert_stats.ok
    assert set(insert_stats.from_checkpoint) == {"base", "indep"}
    assert set(insert_stats.recomputed) == {"correct", "join"}
    expected_insert = ((x * 2.0 + 1.0) * 1.5 + 0.25) + (x * 3.0 - 2.0)
    assert np.array_equal(insert_ex.result("result"), expected_insert)

    replace_ex = Executor(_replace_graph(), max_workers=2, checkpoint=replace_ck)
    replace_stats = replace_ex.run(feed={"__src_x": x})
    assert replace_stats.ok
    assert set(replace_stats.from_checkpoint) == {"indep"}
    assert np.array_equal(
        replace_ex.result("result"),
        (-x + 4.0) + (x * 3.0 - 2.0),
    )

    parent_after = CheckpointStore(parent_path)
    assert set(parent_after.completed_nodes()) == before_nodes
    old_ex = Executor(_evolution_base_graph(),
                      checkpoint=CheckpointStore(parent_path), max_workers=4)
    old_stats = old_ex.run(feed={"__src_x": x})
    assert old_stats.ok and old_stats.recomputed == []
    assert np.array_equal(old_ex.result("result"), (x * 2.0 + 1.0) + (x * 3.0 - 2.0))

    origin_base = insert_ck.provenance("base")
    assert origin_base["origin_node_id"] == "base"
    assert origin_base["derived_from_revision_id"] == parent_after.revision_id
    origin_correct = insert_ck.provenance("correct")
    assert origin_correct["origin_revision_id"] == insert_ck.revision_id


def test_derived_branch_failure_does_not_overwrite_parent_or_other_branch(tmp_path):
    _register_failure_op()
    d = str(tmp_path)
    x = _x()
    parent_path = os.path.join(d, "parent")
    _prepare_evolution_checkpoint(parent_path, x)
    parent_revision = CheckpointStore(parent_path).revision_id

    bad_ck, _ = CheckpointStore.derive(
        parent_path, os.path.join(d, "bad"), _failure_graph(), {"__src_x": x},
        branch_id="bad",
    )
    good_ck, _ = CheckpointStore.derive(
        parent_path, os.path.join(d, "rewire"), _rewire_graph(), {"__src_x": x},
        branch_id="rewire",
    )

    bad_ex = Executor(_failure_graph(), max_workers=2, checkpoint=bad_ck)
    bad_stats = bad_ex.run(feed={"__src_x": x})
    assert set(bad_stats.failed) == {"bad"}
    assert set(bad_stats.from_checkpoint) == {"base", "indep"}

    good_ex = Executor(_rewire_graph(), max_workers=1, checkpoint=good_ck)
    good_stats = good_ex.run(feed={"__src_x": x})
    assert good_stats.ok
    assert set(good_stats.from_checkpoint) == {"indep"}
    assert np.array_equal(good_ex.result("result"), x + (x * 3.0 - 2.0))

    parent = CheckpointStore(parent_path)
    assert set(parent.completed_nodes()) == {"base", "indep", "join"}
    assert parent.revision_id == parent_revision
    bad_reopened = CheckpointStore(os.path.join(d, "bad"))
    assert set(bad_reopened.completed_nodes()) == {"base", "indep"}
    assert bad_reopened.provenance("base")["derived_from_revision_id"] == parent_revision


def test_derived_graph_changes_workers_after_midrun_exit(tmp_path):
    d = str(tmp_path)
    x = _x()
    parent_path = os.path.join(d, "parent")
    child_path = os.path.join(d, "child")
    _prepare_evolution_checkpoint(parent_path, x)
    ck, plan = CheckpointStore.derive(
        parent_path, child_path, _insert_graph(), {"__src_x": x},
        branch_id="worker-change",
    )
    assert plan.to_execute == {"correct", "join"}

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        os.getcwd(), os.path.join(os.getcwd(), "tests"),
    ]))
    code = (
        "import json, os; "
        "from sciruntime import CheckpointStore, Executor; "
        "from test_resume import _insert_graph, _x; "
        "x=_x(); ck=CheckpointStore(%r); "
        "ex=Executor(_insert_graph(), max_workers=int(os.environ['WORKERS']), "
        "checkpoint=ck, exit_after_commits=(int(os.environ['EXITS']) if 'EXITS' in os.environ else None)); "
        "st=ex.run(feed={'__src_x': x}); "
        "out=ex.result('result') if st.ok else None; "
        "print('JSON '+json.dumps({'ok': st.ok, 'from': st.from_checkpoint, "
        "'recomputed': st.recomputed, 'equal': None if out is None else bool((out == ((x*2+1)*1.5+0.25)+(x*3-2)).all())}))"
        % child_path
    )
    env_one = dict(env, WORKERS="1", EXITS="1")
    first = subprocess.run([sys.executable, "-c", code], env=env_one)
    assert first.returncode == 43

    partial = CheckpointStore(child_path)
    assert set(partial.completed_nodes()) == {"base", "indep", "correct"}
    assert set(CheckpointStore(parent_path).completed_nodes()) == {
        "base", "indep", "join",
    }

    env_two = dict(env, WORKERS="3")
    second = subprocess.run(
        [sys.executable, "-c", code], env=env_two, capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stderr
    payload = json.loads(second.stdout.strip().split("JSON ", 1)[1])
    assert payload["ok"]
    assert set(payload["from"]) == {"base", "indep", "correct"}
    assert payload["recomputed"] == ["join"]
    assert payload["equal"]
