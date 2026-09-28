"""内存预算排队、失败传播、幂等发布等保证。"""

import threading

import numpy as np
import pytest

from sciruntime import Engine, EngineConfig, Graph, Node
from sciruntime.blocks import Block
from sciruntime.operators import OperatorRegistry
from sciruntime.state import NodeStatus


class ConstOp:
    def __init__(self, value=1.0):
        self.value = value

    def run(self, inputs, params, store):
        return Block(np.full((2, 2), self.value, dtype=np.float64))


class AddOp:
    def run(self, inputs, params, store):
        return Block(inputs[0].data + inputs[1].data)


class SlowOp:
    def __init__(self, gate: threading.Event):
        self.gate = gate

    def run(self, inputs, params, store):
        self.gate.wait(2)
        return Block(np.ones((2, 2)))


class FailOp:
    def __init__(self, exc=RuntimeError("boom")):
        self.exc = exc

    def run(self, inputs, params, store):
        raise self.exc


def _registry():
    reg = OperatorRegistry()
    reg.register("const", ConstOp())
    reg.register("add", AddOp())
    return reg


def test_memory_budget_queues_then_runs(tmp_path):
    """预算只允许一个节点驻留时，其余节点排队等待，最终全部完成。"""

    import time

    gate = threading.Event()
    reg = OperatorRegistry()
    reg.register("slow", SlowOp(gate))
    graph = Graph()
    tile = 2 * 2 * 8  # 32B
    concurrency = []
    max_seen = 0

    class TrackedOp:
        def run(self, inputs, params, store):
            nonlocal max_seen
            concurrency.append(1)
            max_seen = max(max_seen, len(concurrency))
            gate.wait(2)
            concurrency.pop()
            return Block(np.ones((2, 2)))

    reg.register("tracked", TrackedOp())
    for i in range(6):
        graph.add(Node(f"n{i}", "slow", estimated_bytes=tile))
    graph.nodes.clear()
    for i in range(6):
        graph.add(Node(f"n{i}", "tracked", estimated_bytes=tile))
    engine = Engine(
        EngineConfig(
            workdir=tmp_path,
            max_workers=4,
            memory_budget=tile + 1,  # 任何时刻只允许 1 个节点
            registry=reg,
        )
    )
    done = []
    thread = threading.Thread(
        target=lambda: (done.append(engine.run(graph)), engine.close())
    )
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive()
    gate.set()
    thread.join(5)
    assert not thread.is_alive()
    assert all(st == NodeStatus.SUCCEEDED for st in done[0].statuses.values())
    # 尽管 max_workers=4，预算串行化使任意时刻最多 1 个节点驻留。
    assert max_seen == 1


def test_node_exceeding_budget_is_error(tmp_path):
    reg = _registry()
    graph = Graph()
    graph.add(Node("n0", "const", estimated_bytes=10**9))
    with Engine(EngineConfig(workdir=tmp_path, registry=reg, memory_budget=1024)) as engine:
        with pytest.raises(RuntimeError, match="exceeds memory budget"):
            engine.run(graph)


def test_failure_skips_all_descendants(tmp_path):
    """失败节点的后继全部 SKIPPED，绝不执行；独立分支照常完成。"""

    reg = _registry()
    reg.register("fail", FailOp())
    graph = Graph()
    graph.add(Node("bad", "fail", estimated_bytes=32))
    graph.add(Node("mid", "add", inputs=["bad"], estimated_bytes=32))
    graph.add(Node("leaf", "add", inputs=["mid"], estimated_bytes=32))
    graph.add(Node("good", "const", keep=True, estimated_bytes=32))
    with Engine(EngineConfig(workdir=tmp_path, max_workers=3, registry=reg)) as engine:
        result = engine.run(graph)
    assert not result.ok
    assert result.statuses["bad"] == NodeStatus.FAILED
    assert result.statuses["mid"] == NodeStatus.SKIPPED
    assert result.statuses["leaf"] == NodeStatus.SKIPPED
    assert result.statuses["good"] == NodeStatus.SUCCEEDED
    # 后继块根本不存在：无半成品可读。
    assert not engine.store.has("mid")
    assert not engine.store.has("leaf")


def test_duplicate_completion_publishes_once(tmp_path):
    """模拟同一节点被两个执行者同时完成：结果只发布一次，状态不变。"""

    reg = _registry()
    graph = Graph()
    graph.add(Node("n0", "const", keep=True, estimated_bytes=32))
    with Engine(EngineConfig(workdir=tmp_path, registry=reg)) as engine:
        result = engine.run(graph)
        # 再次成功到达：CAS 应拒绝重复发布。
        again = engine.state.mark_succeeded("n0", {"nbytes": 999})
    assert result.statuses["n0"] == NodeStatus.SUCCEEDED
    assert again is False
    assert engine.state.result_meta("n0") == {"nbytes": 32}


def test_cycle_and_missing_dependency_rejected():
    graph = Graph()
    graph.add(Node("a", "const", inputs=["b"]))
    graph.add(Node("b", "const", inputs=["a"]))
    with pytest.raises(ValueError, match="cycle"):
        graph.validate()
    graph2 = Graph()
    graph2.add(Node("a", "const", inputs=["ghost"]))
    with pytest.raises(ValueError, match="unknown node"):
        graph2.validate()
