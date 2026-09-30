"""恢复测试用计算图: 共享前置 + 慢分支 + 内存受限 fan-out/fan-in。

图结构 (n_branches 个慢分支)::

    src
     |
    shared0 -> shared1            (两个共享前置节点, 所有分支都要消费)
                  |
        +---------+---------+
        |         |         |
     slow_0     slow_1     slow_2     (可注入 sleep 的慢分支, 单输出块)
        |         |         |
      leaf_0    leaf_1     leaf_2    (每个分支的后继, 证明半成品不可读)
        |         |         |
        +---------+---------+
                  |
                reduce                    (最终输出)

任务身份完全由节点 id / 算子 / 块 id / 参数决定, 与工作者数量无关,
因此恢复时可以改变 max_workers。
"""
from __future__ import annotations

import os

from typing import Any, Dict, Tuple

import numpy as np

from .graph import Graph
from .operators import Operator, register_operator


class ScaleAdd(Operator):
    """确定性变换: out = in * scale + bias。"""

    def run(self, inputs, *, params, ctx):
        a = np.asarray(inputs[0], dtype=np.float64)
        return (a * float(params["scale"]) + float(params["bias"]),)


class SlowScale(Operator):
    """慢分支: 发布前 sleep; 任何崩溃都发生在发布之前。

    写入只发生在算子内部局部数组上, 完成前调用方拿不到结果。
    """

    def run(self, inputs, *, params, ctx):
        import time

        delay = float(os.environ.get(f"SCI_SLOW_SLEEP_{ctx.node_id}",
                                     os.environ.get("SCI_SLOW_SLEEP", "0")))
        if delay:
            time.sleep(delay)
        a = np.asarray(inputs[0], dtype=np.float64)
        return (a * float(params["scale"]) + float(params["bias"]),)


class LeafScale(Operator):
    def run(self, inputs, *, params, ctx):
        a = np.asarray(inputs[0], dtype=np.float64)
        return (a * float(params["scale"]),)


class DeterministicReduce(Operator):
    """按分支编号顺序的确定性归约 (浮点加法顺序固定, 可逐位重现)。"""

    def run(self, inputs, *, params, ctx):
        acc = None
        for part in inputs:
            part = np.asarray(part, dtype=np.float64)
            acc = part.copy() if acc is None else acc + part
        return (acc,)


def register_all() -> None:
    for name, op in [
        ("wf_scale", ScaleAdd()),
        ("wf_slow", SlowScale()),
        ("wf_leaf", LeafScale()),
        ("wf_reduce", DeterministicReduce()),
    ]:
        if name not in _registered:
            register_operator(name, op)
            _registered.add(name)


_registered: set = set()
register_all()


def build_resume_graph(
    n: int = 64,
    *,
    n_branches: int = 4,
) -> Tuple[Graph, Dict[str, Any]]:
    """构造恢复测试图, 返回 (graph, {"source": ..., "output": ..., ...})。"""
    g = Graph()
    nbytes = n * 8

    g.add_node(
        "shared0", "wf_scale",
        inputs=("__src_x",), outputs=("s0",),
        params={"scale": 1.5, "bias": 0.25},
        memory_bytes=nbytes,
    )
    g.add_node(
        "shared1", "wf_scale",
        inputs=("s0",), outputs=("s1",),
        params={"scale": 0.75, "bias": -0.5},
        memory_bytes=nbytes,
    )

    leaf_outputs = []
    branch_blocks = []
    for b in range(n_branches):
        bbid = f"b{b}"
        branch_blocks.append(bbid)
        g.add_node(
            f"slow_{b}", "wf_slow",
            inputs=("s1",), outputs=(bbid,),
            params={"scale": 1.0 + 0.1 * b, "bias": 0.01 * b},
            memory_bytes=nbytes,
        )
        lid = f"l{b}"
        leaf_outputs.append(lid)
        g.add_node(
            f"leaf_{b}", "wf_leaf",
            inputs=(bbid,), outputs=(lid,),
            params={"scale": 2.0},
            memory_bytes=nbytes,
        )

    g.add_node(
        "reduce", "wf_reduce",
        inputs=tuple(leaf_outputs), outputs=("result",),
        params={},
        memory_bytes=nbytes,
    )
    g.freeze()

    branch_nodes = [f"slow_{b}" for b in range(n_branches)]
    leaf_nodes = [f"leaf_{b}" for b in range(n_branches)]
    meta = {
        "source": "__src_x",
        "output": "result",
        "shared_nodes": ["shared0", "shared1"],
        "branch_nodes": branch_nodes,
        "leaf_nodes": leaf_nodes,
        "block_bytes": nbytes,
    }
    return g, meta


def serial_reference(x: np.ndarray, *, n_branches: int = 4) -> np.ndarray:
    """与图完全相同运算/归约顺序的串行参考。"""
    s0 = x * 1.5 + 0.25
    s1 = s0 * 0.75 - 0.5
    acc = None
    for b in range(n_branches):
        bb = s1 * (1.0 + 0.1 * b) + 0.01 * b
        leaf = bb * 2.0
        acc = leaf.copy() if acc is None else acc + leaf
    return acc


def build_two_source_graph(n: int = 64) -> Tuple[Graph, Dict[str, Any]]:
    """两个互相独立的源块各自走一条分支, 最后 fan-in 相加。

    用于验证“共享输入只改变一部分”的恢复: 只改 __src_a 时 pa/la 重算,
    pb/lb 与 __src_b 无关, 必须原样复用; fan-in 节点 join 依赖两侧,
    总是重算。
    """
    g = Graph()
    nbytes = n * 8
    g.add_node(
        "pa", "wf_scale",
        inputs=("__src_a",), outputs=("a1",),
        params={"scale": 2.0, "bias": 0.0},
        memory_bytes=nbytes,
    )
    g.add_node(
        "pb", "wf_scale",
        inputs=("__src_b",), outputs=("b1",),
        params={"scale": 1.5, "bias": 1.0},
        memory_bytes=nbytes,
    )
    g.add_node(
        "la", "wf_leaf",
        inputs=("a1",), outputs=("la",),
        params={"scale": 3.0},
        memory_bytes=nbytes,
    )
    g.add_node(
        "lb", "wf_leaf",
        inputs=("b1",), outputs=("lb",),
        params={"scale": -2.0},
        memory_bytes=nbytes,
    )
    g.add_node(
        "join", "wf_reduce",
        inputs=("la", "lb"), outputs=("result",),
        params={},
        memory_bytes=nbytes,
    )
    g.freeze()
    meta = {
        "sources": ("__src_a", "__src_b"),
        "output": "result",
        "branch_a": ["pa", "la"],
        "branch_b": ["pb", "lb"],
        "block_bytes": nbytes,
    }
    return g, meta


def serial_two_source(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """与 build_two_source_graph 完全相同运算顺序的串行参考。"""
    la = (a * 2.0) * 3.0
    lb = (b * 1.5 + 1.0) * -2.0
    return la + lb
