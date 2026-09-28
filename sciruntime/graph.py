"""计算图：节点声明算子与数据依赖，拓扑校验与遍历。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blocks import BlockHandle


@dataclass
class Node:
    """计算节点。

    - ``inputs``：依赖的上游节点 id 列表（同时是数据块句柄来源）；
    - ``params``：算子参数（分块坐标、形状等）；
    - ``keep``：完成后是否保留输出块（最终结果置 True）；
    - ``estimated_bytes``：对输出块大小的预估，供内存准入使用。
    """

    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    keep: bool = False
    estimated_bytes: int = 0

    def handle(self) -> BlockHandle:
        return BlockHandle(self.id)


class Graph:
    """DAG 容器：负责校验、拓扑序与下游/引用统计。"""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        self.nodes[node.id] = node
        return node

    def validate(self) -> None:
        for node in self.nodes.values():
            seen: set[str] = set()
            for dep in node.inputs:
                if dep not in self.nodes:
                    raise ValueError(f"node {node.id!r} depends on unknown node {dep!r}")
                if dep in seen:
                    raise ValueError(f"node {node.id!r} has duplicate input {dep!r}")
                if dep == node.id:
                    raise ValueError(f"node {node.id!r} depends on itself")
                seen.add(dep)
        # 环检测（DFS 三色法）。
        color: dict[str, int] = {}

        def visit(nid: str, stack: list[str]) -> None:
            if color.get(nid) == 1:
                cycle = " -> ".join(stack[stack.index(nid):] + [nid])
                raise ValueError(f"cycle detected: {cycle}")
            if color.get(nid) == 2:
                return
            color[nid] = 1
            stack.append(nid)
            for dep in self.nodes[nid].inputs:
                visit(dep, stack)
            stack.pop()
            color[nid] = 2

        for nid in self.nodes:
            visit(nid, [])

    def topo_layers(self) -> list[list[str]]:
        """Kahn 分层：同一层内节点彼此无依赖，可并行。"""

        self.validate()
        indegree = {nid: len(node.inputs) for nid, node in self.nodes.items()}
        dependents: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.inputs:
                dependents[dep].append(nid)
        remaining = dict(indegree)
        layers: list[list[str]] = []
        ready = sorted(nid for nid, deg in remaining.items() if deg == 0)
        while ready:
            layers.append(ready)
            next_ready: list[str] = []
            for nid in ready:
                for child in dependents[nid]:
                    remaining[child] -= 1
                    if remaining[child] == 0:
                        next_ready.append(child)
            ready = sorted(next_ready)
        if len(layers) and sum(len(layer) for layer in layers) != len(self.nodes):
            raise ValueError("cycle detected")
        return layers

    def consumer_counts(self) -> dict[str, int]:
        """每个节点输出被多少其他节点直接消费（用于引用计数）。"""

        counts = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.inputs:
                counts[dep] += 1
        return counts
