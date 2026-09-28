"""计算图: 节点 (算子调用) 与数据块依赖。

节点声明:
- inputs:  读取的 block_id 列表
- outputs: 写出的 block_id 列表
- operator: 算子名 (见 operators.registry)
- params:  算子参数
- memory_bytes: 该节点执行时除输入外需要的额外内存预算 (输出块+临时空间)

图构建后做拓扑校验; 运行时只依赖 inputs/outputs, 新算子无需改动调度器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


class GraphError(ValueError):
    pass


@dataclass
class Node:
    node_id: str
    operator: str
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)
    memory_bytes: int = 0


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self._producers: Dict[str, str] = {}      # block_id -> node_id
        self._consumers: Dict[str, List[str]] = {} # block_id -> [node_id]
        self._deps: Dict[str, set] = {}           # node_id -> {upstream node_id}

    # ----- 构建 -----
    def add_node(
        self,
        node_id: str,
        operator: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        params: Optional[Dict[str, Any]] = None,
        memory_bytes: int = 0,
    ) -> Node:
        if node_id in self.nodes:
            raise GraphError(f"节点 id 重复: {node_id!r}")
        node = Node(
            node_id=node_id,
            operator=operator,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            params=dict(params or {}),
            memory_bytes=int(memory_bytes),
        )
        self.nodes[node_id] = node
        self._deps.setdefault(node_id, set())
        for bid in node.outputs:
            if bid in self._producers:
                raise GraphError(
                    f"block {bid!r} 同时被 {self._producers[bid]!r} 和 {node_id!r} 生产"
                )
            self._producers[bid] = node_id
        return node

    def freeze(self) -> "Graph":
        """校验图并建立依赖索引。构建完成后必须调用。"""
        for nid, node in self.nodes.items():
            deps = self._deps[nid]
            for bid in node.inputs:
                producer = self._producers.get(bid)
                if producer is None:
                    continue
                if producer == nid:
                    raise GraphError(f"节点 {nid!r} 不能消费自己的输出 {bid!r}")
                deps.add(producer)
            # 登记消费者关系 (用于数据生命周期), 去重每个 (块,节点) 对
            for bid in node.inputs:
                lst = self._consumers.setdefault(bid, [])
                if nid not in lst:
                    lst.append(nid)
        self._topo_order()  # 仅为探测环
        return self

    # ----- 查询 -----
    def dependencies(self, node_id: str) -> set:
        return set(self._deps[node_id])

    def producer_of(self, block_id: str) -> Optional[str]:
        return self._producers.get(block_id)

    def consumers_of(self, block_id: str) -> List[str]:
        return list(self._consumers.get(block_id, []))

    def source_blocks(self) -> List[str]:
        """没有生产者、由外部喂入的块。"""
        return [bid for bid in self._consumers if bid not in self._producers]

    def all_blocks(self) -> List[str]:
        blocks = set(self._producers) | set(self._consumers)
        return sorted(blocks)

    def consumer_count(self, block_id: str) -> int:
        """一个块被图中多少个节点消费 (用于生命周期回收)。"""
        return len(self._consumers.get(block_id, ()))

    def _topo_order(self) -> List[str]:
        indeg = {nid: 0 for nid in self.nodes}
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, deps in self._deps.items():
            indeg[nid] = len(deps)
            for d in deps:
                children[d].append(nid)
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: List[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for child in children[nid]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(order) != len(self.nodes):
            stuck = [n for n, d in indeg.items() if d > 0]
            raise GraphError(f"计算图存在环, 涉及节点: {stuck}")
        return order

    def topological_order(self) -> List[str]:
        return self._topo_order()

    def output_blocks(self) -> List[str]:
        """图的最终输出: 没有任何消费者的块。"""
        return [
            bid
            for bid in self._producers
            if not self._consumers.get(bid)
        ]
