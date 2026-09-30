"""并行执行器。

职责:
1. 按拓扑依赖调度就绪节点, 使用线程池 (进程内) 或进程池 (本机子进程);
2. 内存预算 admission control: 一个节点启动前要求
       当前驻留块内存 + 节点输入块 + 节点额外内存 <= memory_budget,
   不满足则排队等待, 而不是一次性把所有任务丢进池子里;
3. 失败隔离: 节点失败后其所有直接/间接后继标记为 skipped, 绝不执行,
   因而不可能消费到半成品; 失败节点在发布前的写入对任何人不可见;
4. 幂等发布: 每个节点 (在本进程/本次运行内) 只成功发布一次; 即使任务被
   重复执行 (重试或恢复后重放), 已发布块也不会重复 put;
5. 检查点恢复: 已完成节点跳过执行, 块从检查点灌入 BlockStore;
6. 数据生命周期: 节点结束后释放输入 pin, 无人消费的块被回收。
"""
from __future__ import annotations

import os
import threading
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .blocks import BlockStore
from .checkpoint import CheckpointStore, fingerprint_feed
from .graph import Graph, Node
from .operators import TaskContext, get_operator


# 子进程 worker 的入口 (顶层函数才能被 ProcessPoolExecutor pickle)
_WORKER_REGISTRY_READY = False


def _ensure_builtin_operators() -> None:
    global _WORKER_REGISTRY_READY
    if _WORKER_REGISTRY_READY:
        return
    from . import matrix as _matrix  # noqa: F401

    _WORKER_REGISTRY_READY = True


def _remote_run(
    op_name: str,
    node_id: str,
    attempt: int,
    input_arrays: Sequence[np.ndarray],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, ...]:
    _ensure_builtin_operators()
    op = get_operator(op_name)
    return tuple(op.run(list(input_arrays), params=params, ctx=TaskContext(node_id, attempt)))


@dataclass
class NodeResult:
    status: str  # "success" | "failed" | "skipped"
    error: Optional[str] = None


@dataclass
class ExecutionStats:
    succeeded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    from_checkpoint: List[str] = field(default_factory=list)
    executed: List[str] = field(default_factory=list)
    stopped: bool = False
    peak_resident_bytes: int = 0
    peak_inflight_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


class TaskFailure(RuntimeError):
    pass


class Executor:
    """并行图执行器。

    backend:
      "thread"  -> ThreadPoolExecutor, 默认, 算子在本进程内并行;
      "process" -> ProcessPoolExecutor, 算子在本机子进程内并行 (spawn)。
    """

    def __init__(
        self,
        graph: Graph,
        store: Optional[BlockStore] = None,
        *,
        max_workers: Optional[int] = None,
        memory_budget: Optional[int] = None,
        backend: str = "thread",
        checkpoint: Optional[CheckpointStore] = None,
        max_retries: int = 0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.graph = graph
        self.store = store or BlockStore()
        self.backend = backend
        self.max_workers = max_workers or min(8, (os.cpu_count() or 2))
        # 默认给一个宽裕预算; 传入字节数可严格控制排队
        self.memory_budget = memory_budget
        self.checkpoint = checkpoint
        self.max_retries = max_retries
        self.stop_event = stop_event

        self._cond = threading.Condition()
        self._remaining_deps: Dict[str, int] = {}
        self._children: Dict[str, List[str]] = {}
        self._done: Dict[str, NodeResult] = {}
        self._inflight: Dict[str, Future] = {}
        # 发布去重: 一个节点只允许进入发布阶段一次
        self._published: set = set()
        self._pinned_inputs: Dict[str, List[str]] = {}
        self._inflight_bytes = 0
        self._stats = ExecutionStats()

    # ---------- 公共入口 ----------
    def run(self, feed: Optional[Dict[str, np.ndarray]] = None) -> ExecutionStats:
        feed = dict(feed or {})
        self._prepare(feed)

        if self.backend == "thread":
            pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                      thread_name_prefix="sci")
        elif self.backend == "process":
            ctx = __import__("multiprocessing").get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        else:
            raise ValueError(f"未知 backend: {self.backend!r}")

        try:
            inflight: Dict[Future, str] = {}
            while len(self._done) < len(self.graph.nodes):
                if self.stop_event is not None and self.stop_event.is_set():
                    self._stats.stopped = True
                    break
                with self._cond:
                    newly = self._pump_locked(pool)
                    for nid, fut in newly:
                        inflight[fut] = nid
                if not inflight:
                    # 没有在跑任务: 可能全部跳过/失败, 或内存预算下暂时无法启动
                    with self._cond:
                        remaining = [
                            n for n in self.graph.nodes
                            if n not in self._done and n not in self._inflight
                        ]
                    if not remaining:
                        break
                    # 预算下无法启动新任务, 且没有任务在跑: 预算配置无解
                    raise RuntimeError(
                        "内存预算过小: 没有正在执行的任务, 但仍有节点无法启动: "
                        f"{remaining[:3]} ..."
                    )
                done, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                for fut in done:
                    nid = inflight.pop(fut)
                    # 结果由 done callback 处理; 这里只负责唤醒后重新 pump
                    with self._cond:
                        self._resolve_future_locked(nid, fut)
                        self._record_memory_locked()
                        self._cond.notify_all()
        finally:
            pool.shutdown(wait=True)
            self.store.collect_garbage()
        return self._stats

    def result(self, block_id: str) -> np.ndarray:
        return self.store.get(block_id)

    # ---------- 初始化 ----------
    def _prepare(self, feed: Dict[str, np.ndarray]) -> None:
        if self.checkpoint is not None:
            self.checkpoint.bind_run(self.graph.graph_id(), fingerprint_feed(feed))
        checkpoint_done: set = set()
        if self.checkpoint is not None:
            checkpoint_done = {
                nid for nid in self.graph.nodes if self.checkpoint.is_completed(nid)
            }
        # 生命周期登记: 每个块 *仍待执行* 的消费者数量。
        # 最终输出块 (无消费者) 保留, 调用方取走结果后可用 collect 显式回收。
        for bid in self.graph.all_blocks():
            live_consumers = [
                c for c in self.graph.consumers_of(bid) if c not in checkpoint_done
            ]
            self.store.register_block(
                bid,
                len(live_consumers),
                retain=not self.graph.consumers_of(bid),
            )
        # 外部喂入的源块 (登记其图内消费者数量, 避免被提前回收)
        for bid, val in feed.items():
            if bid not in {b for b in self.graph.all_blocks()}:
                live = [
                    c for c in self.graph.consumers_of(bid) if c not in checkpoint_done
                ]
                self.store.register_block(
                    bid,
                    len(live),
                    retain=len(self.graph.consumers_of(bid)) == 0,
                )
            self.store.put(bid, np.asarray(val))
        for nid, node in self.graph.nodes.items():
            deps = {self.graph.producer_of(bid) for bid in node.inputs}
            deps.discard(None)
            self._remaining_deps[nid] = len(deps)
            for d in deps:
                self._children.setdefault(d, []).append(nid)

        # 检查点恢复: 已完成节点直接标成功并灌回块
        if self.checkpoint is not None:
            completed_on_disk = [
                nid for nid in self.graph.nodes if self.checkpoint.is_completed(nid)
            ]
            restored = self.checkpoint.load_completed(completed_on_disk)
            self.store.restore(restored)
            for nid in completed_on_disk:
                self._mark_published_locked(nid)
                self._finish_success_locked(nid)
                self._stats.from_checkpoint.append(nid)
        with self._cond:
            self._record_memory_locked()

    # ---------- 调度循环 ----------
    def _ready_nodes_locked(self) -> List[str]:
        return [
            nid
            for nid, deps in self._remaining_deps.items()
            if deps == 0 and nid not in self._done and nid not in self._inflight
        ]

    def _required_memory_locked(self, node: Node) -> int:
        # 输入块已经计入 resident_bytes; 这里只需节点的新增内存
        # (输出块 + 临时空间), 避免重复计算导致人为饿死。
        return int(node.memory_bytes)

    def _pump_locked(self, pool: Any) -> List[Tuple[str, Future]]:
        # 回收生命周期结束的块, 释放预算
        self.store.collect_garbage()
        newly: List[Tuple[str, Future]] = []

        for nid in self._ready_nodes_locked():
            node = self.graph.nodes[nid]
            budget = self.memory_budget
            if budget is not None:
                resident = self.store.resident_bytes()
                need = self._required_memory_locked(node)
                if resident + self._inflight_bytes + need > budget:
                    continue  # 内存不足: 保持排队

            # 先取消检查: 任一依赖失败/跳过则本节点跳过
            bad_dep = [
                d for d in self.graph.dependencies(nid)
                if d in self._done and self._done[d].status != "success"
            ]
            if bad_dep:
                self._skip_chain_locked(nid, cause=f"上游 {bad_dep[0]} 未成功")
                continue

            # pin 输入, 保证执行期间不被回收
            inputs = [self.store.get(bid) for bid in node.inputs]
            for bid in node.inputs:
                self.store.pin(bid)
            self._pinned_inputs[nid] = list(node.inputs)

            fut = self._submit(pool, node, inputs, attempt=0)
            self._inflight[nid] = fut
            self._inflight_bytes += self._required_memory_locked(node)
            self._record_memory_locked()
            newly.append((nid, fut))
        return newly

    def _resolve_future_locked(self, nid: str, fut: Future) -> None:
        """future 完成后的统一处理 (在调用线程持锁执行, 无回调重入)。"""
        self._stats.executed.append(nid)
        self._inflight_bytes -= max(0, self.graph.nodes[nid].memory_bytes)
        try:
            outputs = fut.result()
        except BaseException as exc:  # 算子失败
            self._release_inputs(nid)
            detail = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            err = exc if isinstance(exc, Exception) else RuntimeError(detail)
            self._finish_failure_locked(nid, err)
            self._stats.failed[nid] = detail
            return
        self._release_inputs(nid)

        if self.stop_event is not None and self.stop_event.is_set():
            self._stats.stopped = True
            self._inflight.pop(nid, None)
            return

        node = self.graph.nodes[nid]
        if len(outputs) != len(node.outputs):
            self._finish_failure_locked(
                nid,
                TaskFailure(
                    f"算子 {node.operator} 返回 {len(outputs)} 个输出, "
                    f"节点声明 {len(node.outputs)} 个"
                ),
            )
            return

        if nid in self._published:
            self._finish_success_locked(nid)
            return
        self._mark_published_locked(nid)
        try:
            for bid, val in zip(node.outputs, outputs):
                self.store.put(bid, val)
        except Exception as exc:
            self._finish_failure_locked(nid, exc)
            self._stats.failed[nid] = repr(exc)
            return
        if self.checkpoint is not None:
            try:
                self.checkpoint.commit_node(nid, node.outputs, outputs)
            except Exception as exc:
                for bid in node.outputs:
                    self.store.discard(bid)
                self._finish_failure_locked(nid, exc)
                self._stats.failed[nid] = repr(exc)
                return
        self._finish_success_locked(nid)

    # ---------- 提交 ----------
    def _submit(self, pool: Any, node: Node, inputs: List[np.ndarray], attempt: int) -> Future:
        if self.backend == "thread":
            return pool.submit(self._run_inline, node, list(inputs), attempt)
        return pool.submit(
            _remote_run, node.operator, node.node_id, attempt,
            tuple(inputs), dict(node.params),
        )

    def _run_inline(
        self, node: Node, inputs: List[np.ndarray], attempt: int
    ) -> Tuple[np.ndarray, ...]:
        op = get_operator(node.operator)
        return tuple(
            op.run(inputs, params=node.params, ctx=TaskContext(node.node_id, attempt))
        )

    def _mark_published_locked(self, nid: str) -> None:
        self._published.add(nid)

    def _finish_success_locked(self, nid: str) -> None:
        if nid not in self._done:
            self._done[nid] = NodeResult("success")
            self._stats.succeeded.append(nid)
        for child in self._children.get(nid, ()):
            self._remaining_deps[child] -= 1
        self._inflight.pop(nid, None)

    def _finish_failure_locked(self, nid: str, exc: Exception) -> None:
        self._done[nid] = NodeResult("failed", error=repr(exc))
        if nid not in self._stats.failed:
            self._stats.failed[nid] = repr(exc)
        self._inflight.pop(nid, None)
        # 后继全部跳过
        for child in list(self._children.get(nid, ())):
            if child not in self._done:
                self._skip_chain_locked(child, cause=f"上游 {nid} 失败")

    def _skip_chain_locked(self, nid: str, *, cause: str) -> None:
        stack = [(nid, cause)]
        while stack:
            cur, why = stack.pop()
            if cur in self._done or cur in self._inflight:
                continue
            self._done[cur] = NodeResult("skipped", error=why)
            self._stats.skipped.append(cur)
            # 该节点不会消费其输入了
            for bid in self.graph.nodes[cur].inputs:
                self.store.consumer_finished(bid)
            for child in self._children.get(cur, ()):
                stack.append((child, f"上游 {cur} 被跳过"))

    def _release_inputs(self, nid: str) -> None:
        bids = self._pinned_inputs.pop(nid, None)
        if not bids:
            return
        for bid in bids:
            # 该消费者已结束: 同时解除 pin 与待消费计数
            try:
                self.store.release(bid)
            except RuntimeError:
                pass
            self.store.consumer_finished(bid)

    def _record_memory_locked(self) -> None:
        resident = self.store.resident_bytes()
        self._stats.peak_resident_bytes = max(
            self._stats.peak_resident_bytes, resident
        )
        self._stats.peak_inflight_bytes = max(
            self._stats.peak_inflight_bytes, self._inflight_bytes
        )
