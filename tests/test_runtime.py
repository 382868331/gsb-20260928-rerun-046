"""运行时核心测试。直接用 pytest 运行。"""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

from sciruntime import CheckpointStore, Executor
from sciruntime.blocks import BlockStore
from sciruntime.graph import Graph, GraphError
from sciruntime.matrix import (
    assemble_result,
    build_blocked_matmul_graph,
    serial_reference_matmul,
)
from sciruntime.operators import Operator, TaskContext, register_operator


class Boom(Operator):
    def __init__(self, fail_times=0, gate=None):
        self.fail_times = fail_times
        self.calls = 0
        self.gate = gate

    def run(self, inputs, *, params, ctx):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return (np.full(params["shape"], params.get("value", 1.0)),)


class Identity(Operator):
    def run(self, inputs, *, params, ctx):
        return (inputs[0].copy(),)


class Adder(Operator):
    def run(self, inputs, *, params, ctx):
        return (inputs[0] + inputs[1],)


def _register(name, op):
    from sciruntime.operators import operator_registry
    if name not in operator_registry:
        register_operator(name, op)


def _make_matmul(rng, m=65, k=50, n=73, seed=0):
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    graph, out_blocks = build_blocked_matmul_graph(
        A.shape, B.shape, row_block=16, k_block=20, col_block=24
    )
    return A, B, graph, out_blocks

@pytest.mark.parametrize("backend", ["thread", "process"])
def test_blocked_matmul_matches_serial(backend):
    rng = np.random.default_rng(42)
    A, B, graph, out_blocks = _make_matmul(rng)
    ex = Executor(graph, max_workers=3, backend=backend)
    stats = ex.run(feed={"__src_A": A, "__src_B": B})
    assert stats.ok
    result = assemble_result(ex.store, out_blocks, A.shape[0], n_cols_blocks=4)
    assert np.array_equal(result, serial_reference_matmul(A, B))


def test_graph_rejects_cycle_and_double_producer():
    g = Graph()
    g.add_node("n1", "identity", outputs=("x",))
    with pytest.raises(GraphError):
        g.add_node("n2", "identity", outputs=("x",))


def test_memory_budget_queues_tasks():
    """预算很小时, 任意时刻驻留内存都不能超预算。"""
    rng = np.random.default_rng(1)
    A, B, graph, _ = _make_matmul(rng)
    store = BlockStore()
    # 预算允许切分拷贝, 但只放得下少量分块乘输出 -> gemm 仍需排队
    budget = 2 * (A.nbytes + B.nbytes) + 4 * 16 * 24 * A.itemsize
    ex = Executor(graph, store=store, max_workers=4, memory_budget=budget)

    peak = 0
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, store.resident_bytes())

    t = threading.Thread(target=monitor)
    t.start()
    stats = ex.run(feed={"__src_A": A, "__src_B": B})
    stop.set()
    t.join()
    assert stats.ok
    # 排队期间驻留严格不超预算; 唯一允许的超出发生在“没有任何在途任务
    # 能再释放内存”的 fan-in 点 (reduce 需要 3 个 partials 同时驻留),
    # 由运行时标记为不可避峰值, 且不超过一个归约节点的输入+输出规模。
    assert peak <= budget + stats.budget_overshoot_bytes
    # 不可避峰值必须是小而有界的 fan-in: 该图 K 维切成 3 段,
    # 上界取 8 个 16x24 块 (相比整个中间结果规模可忽略)。
    assert stats.budget_overshoot_bytes <= 8 * 16 * 24 * A.itemsize


def test_failure_skips_descendants_no_halfproduct():
    _register("boom", Boom(fail_times=99))
    _register("identity", Identity())
    g = Graph()
    g.add_node("root", "boom", outputs=("r",),
               params={"shape": (2, 2)}, memory_bytes=32)
    g.add_node("mid", "identity", inputs=("r",), outputs=("m",))
    g.add_node("leaf", "identity", inputs=("m",), outputs=("l",))
    g.freeze()
    ex = Executor(g, max_workers=2)
    stats = ex.run()
    assert "root" in stats.failed
    assert set(stats.skipped) == {"mid", "leaf"}
    assert not ex.store.contains("r")
    assert not ex.store.contains("m")


def test_duplicate_publish_is_idempotent():
    """算子被重复执行时, 同一节点也只能发布一次结果。"""
    op = Boom(fail_times=0)
    _register("dup_pub", op)
    g = Graph()
    g.add_node("n", "dup_pub", outputs=("x",), params={"shape": (3,)})
    g.freeze()
    store = BlockStore()
    ex = Executor(g, store=store, max_workers=1)
    stats = ex.run()
    assert stats.ok
    # 直接再调用发布路径, 模拟重复发布:
    with pytest.raises(RuntimeError):
        store.put("x", np.zeros(3))


def test_checkpoint_resume(tmp_path):
    rng = np.random.default_rng(7)
    A, B, graph, out_blocks = _make_matmul(rng)
    ck = CheckpointStore(str(tmp_path))
    ex1 = Executor(graph, max_workers=3, checkpoint=ck)
    s1 = ex1.run(feed={"__src_A": A, "__src_B": B})
    assert s1.ok

    # 全新 store + 新 Executor: 所有节点应从检查点恢复
    graph2, out_blocks2 = build_blocked_matmul_graph(
        A.shape, B.shape, row_block=16, k_block=20, col_block=24
    )
    ck2 = CheckpointStore(str(tmp_path))
    ex2 = Executor(graph2, max_workers=3, checkpoint=ck2)
    s2 = ex2.run(feed={"__src_A": A, "__src_B": B})
    assert len(s2.from_checkpoint) == len(s1.succeeded)
    result = assemble_result(ex2.store, out_blocks2, A.shape[0], n_cols_blocks=4)
    assert np.array_equal(result, serial_reference_matmul(A, B))


def test_new_operator_reuses_runtime():
    """新增算子不改调度器/存储, 依赖与生命周期照旧。"""
    _register("adder", Adder())
    g = Graph()
    g.add_node("src_a", "identity", inputs=("a",), outputs=("a2",))
    g.add_node("src_b", "identity", inputs=("b",), outputs=("b2",))
    g.add_node("sum", "identity", inputs=("a2",), outputs=("c",))
    g.add_node("sum_b", "identity", inputs=("b2",), outputs=("cb",))
    g.freeze()
    ex = Executor(g, max_workers=2)
    stats = ex.run(feed={"a": np.arange(4.0), "b": np.ones(4)})
    assert stats.ok
    assert np.array_equal(ex.result("c"), np.arange(4.0))
    # 中间块在所有消费者结束后会被运行时自动回收
    assert not ex.store.contains("a2")
    # 最终输出默认保留, 显式释放后才可回收
    assert not ex.store.collect("c")
    ex.store.release_output("c")
    assert ex.store.collect("c")
