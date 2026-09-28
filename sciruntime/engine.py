"""本地执行引擎：线程池或子进程池，含内存预算准入与断点恢复。

关键保证：

- 内存预算不足时节点排队等待；单节点需求超过预算直接判错，避免死等；
- 节点失败（含子进程崩溃）后其全部后继标记 SKIPPED，绝不执行，
  因此后继不可能读到半成品；
- 成功发布由 WAL 中的状态 CAS 保护：同一节点重复完成不会重复发布；
- 重新创建 Engine 指向同一工作目录时，已完成节点直接复用，
  只执行未完成节点。
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from pathlib import Path

from .blocks import Block, BlockHandle, BlockStore
from .graph import Graph, Node
from .operators import OperatorRegistry, registry as default_registry
from .state import NodeStatus, RunState


@dataclass
class EngineConfig:
    workdir: str = "./scirun"
    max_workers: int = 2
    memory_budget: int = 256 * 1024 * 1024
    executor: str = "thread"  # "thread" | "process"
    registry: OperatorRegistry = field(default_factory=lambda: default_registry)


@dataclass
class EngineResult:
    statuses: dict[str, NodeStatus]
    outputs: dict[str, Block]
    failed: dict[str, str]
    skipped: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


def _run_operator_thread(op, inputs, params) -> Block:
    return op.run(inputs, params, None)  # type: ignore[arg-type]


def _run_operator_process(
    store_root: str, op_name: str, inputs: list[str], params: dict
) -> dict:
    """子进程任务：经磁盘读取上游块，结果回传主进程统一发布。"""

    from .blocks import BlockStore
    from .operators import registry
    from . import sciops  # noqa: F401  确保内置算子在子进程中完成注册

    store = BlockStore(store_root)
    blocks = [store.get(BlockHandle(dep)) for dep in inputs]
    result = registry.get(op_name).run(blocks, params, store)
    return {"data": result.data, "meta": result.meta or {}}


class Engine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        if self.config.executor not in {"thread", "process"}:
            raise ValueError("executor must be 'thread' or 'process'")
        workdir = Path(self.config.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.store = BlockStore(workdir / "blocks")
        self.state = RunState(workdir / "state.log")
        self._reserved = 0

    def close(self) -> None:
        self.state.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, graph: Graph) -> EngineResult:
        graph.validate()
        nodes = graph.nodes
        dependents = self._dependents(graph)

        # 生命周期：每个块的引用数 = 仍未终结的直接消费者数；
        # keep 块额外受 keep 集合保护，不随引用归零删除。
        # 恢复运行时，已无未终结消费者的非 keep 块立即清理。
        for node in nodes.values():
            if node.keep:
                self.store.keep(node.id)
            if not self.state.is_terminal(node.id):
                for dep in node.inputs:
                    self.store.retain(dep)
        # 重建内存账本；恢复时若非 keep 块已无未终结消费者，立即删除。
        for nid, node in nodes.items():
            if self.store.has(nid):
                self._reserved += self.store.nbytes(nid)
                if not node.keep:
                    self._reserved -= self.store.release(nid)

        pending: deque[str] = deque(
            nid for nid in nodes if not self.state.is_terminal(nid)
        )
        running: dict[Future, str] = {}
        attempts = 0
        pool = self._make_pool()

        try:
            while pending or running:
                progressed = True
                while progressed:
                    progressed = False
                    for _ in range(len(pending)):
                        nid = pending.popleft()
                        node = nodes[nid]
                        decision = self._admit(node)
                        if decision in {"wait", "blocked"}:
                            pending.append(nid)
                            continue
                        if decision == "skip":
                            self._skip_tree(nid, dependents, nodes)
                            progressed = True
                            continue
                        attempts += 1
                        self.state.mark_running(nid, attempts)
                        self._reserve(node.estimated_bytes)
                        running[self._submit(pool, node)] = nid
                        progressed = True

                if not running:
                    if pending:
                        stuck = nodes[pending[0]]
                        raise RuntimeError(
                            f"node {stuck.id!r} requires {stuck.estimated_bytes} bytes, "
                            f"exceeds memory budget {self.config.memory_budget}"
                        )
                    break

                done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    nid = running.pop(future)
                    node = nodes[nid]
                    self._reserve(-node.estimated_bytes)
                    try:
                        payload = future.result()
                    except Exception as exc:
                        self._fail_tree(nid, dependents, nodes, repr(exc))
                        continue
                    block = (
                        Block(payload["data"], payload.get("meta"))
                        if isinstance(payload, dict)
                        else payload
                    )
                    # 幂等发布：仅第一次成功写块、推进状态、释放依赖。
                    if not self.state.is_terminal(nid):
                        real_nbytes = self.store.publish(nid, block)
                        self._reserve(real_nbytes)
                        if self.state.mark_succeeded(nid, {"nbytes": real_nbytes}):
                            self._release_inputs(node)
                            self._release_self_if_orphan(nid, nodes, dependents)
            return self._collect(nodes)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _make_pool(self):
        if self.config.executor == "thread":
            return concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.max_workers,
            mp_context=mp.get_context("spawn"),
        )

    def _submit(self, pool, node: Node) -> Future:
        if self.config.executor == "thread":
            input_blocks = [self.store.get(BlockHandle(dep)) for dep in node.inputs]
            return pool.submit(
                _run_operator_thread,
                self.config.registry.get(node.op),
                input_blocks,
                dict(node.params),
            )
        return pool.submit(
            _run_operator_process,
            str(self.store.root),
            node.op,
            list(node.inputs),
            dict(node.params),
        )

    def _admit(self, node: Node) -> str:
        upstream = [self.state.status(dep) for dep in node.inputs]
        if any(st in {NodeStatus.FAILED, NodeStatus.SKIPPED} for st in upstream):
            return "skip"
        if any(st != NodeStatus.SUCCEEDED for st in upstream):
            return "wait"
        required = node.estimated_bytes
        if required > self.config.memory_budget:
            return "blocked"
        if self._reserved + required > self.config.memory_budget:
            return "wait"
        return "admitted"

    def _fail_tree(
        self, nid: str, dependents: dict[str, list[str]], nodes: dict[str, Node], message: str
    ) -> None:
        if self.state.mark_failed(nid, message):
            self._release_inputs(nodes[nid])
            self._release_self_if_orphan(nid, nodes, dependents)
        for child in dependents[nid]:
            if not self.state.is_terminal(child):
                self._skip_tree(child, dependents, nodes)

    def _skip_tree(
        self, nid: str, dependents: dict[str, list[str]], nodes: dict[str, Node]
    ) -> None:
        queue = deque([nid])
        seen: set[str] = set()
        while queue:
            cur = queue.popleft()
            if cur in seen or self.state.is_terminal(cur):
                continue
            seen.add(cur)
            if self.state.mark_skipped(cur, "upstream failed"):
                self._release_inputs(nodes[cur])
                self._release_self_if_orphan(cur, nodes, dependents)
            queue.extend(dependents[cur])

    def _release_inputs(self, node: Node) -> None:
        for dep in node.inputs:
            self._reserve(-self.store.release(dep))

    def _release_self_if_orphan(
        self, nid: str, nodes: dict[str, Node], dependents: dict[str, list[str]]
    ) -> None:
        """非 keep 块在所有消费者终结后立即回收（归还内存预算）。"""

        if nodes[nid].keep:
            return
        if not all(self.state.is_terminal(child) for child in dependents[nid]):
            return
        self._reserve(-self.store.release(nid))

    def _reserve(self, delta: int) -> None:
        self._reserved = max(0, self._reserved + delta)

    def _collect(self, nodes: dict[str, Node]) -> EngineResult:
        statuses = {nid: self.state.status(nid) for nid in nodes}
        outputs = {
            nid: self.store.get(node.handle())
            for nid, node in nodes.items()
            if node.keep and statuses[nid] == NodeStatus.SUCCEEDED
        }
        failed, skipped = {}, {}
        for nid, st in statuses.items():
            msg = self.state.failed().get(nid, "")
            if st == NodeStatus.FAILED:
                failed[nid] = msg
            elif st == NodeStatus.SKIPPED:
                skipped[nid] = msg
        return EngineResult(statuses, outputs, failed, skipped)

    @staticmethod
    def _dependents(graph: Graph) -> dict[str, list[str]]:
        dependents: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
        for nid, node in graph.nodes.items():
            for dep in node.inputs:
                dependents[dep].append(nid)
        return dependents
