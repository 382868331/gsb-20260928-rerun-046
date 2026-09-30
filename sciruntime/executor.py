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
from .checkpoint import CheckpointStore, RecoveryPlan
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
    recomputed: List[str] = field(default_factory=list)
    peak_inflight_bytes: int = 0
    # “没有在途任务可释放内存”时, fan-in 节点不可避免的超预算峰值
    budget_overshoot_bytes: int = 0

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
        crash_after_block: Optional[str] = None,
        exit_after_commits: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.store = store or BlockStore()
        self.backend = backend
        self.max_workers = max_workers or min(8, (os.cpu_count() or 2))
        # 默认给一个宽裕预算; 传入字节数可严格控制排队
        self.memory_budget = memory_budget
        self.checkpoint = checkpoint
        self.max_retries = max_retries
        self._crash_after_block = crash_after_block
        self._exit_after_commits = exit_after_commits
        self._committed = 0

        self._cond = threading.Condition()
        self._remaining_deps: Dict[str, int] = {}
        self._children: Dict[str, List[str]] = {}
        self._done: Dict[str, NodeResult] = {}
        self._inflight: Dict[str, Future] = {}
        # 发布去重: 一个节点只允许进入发布阶段一次
        self._published: set = set()
        self._pinned_inputs: Dict[str, List[str]] = {}
        self._stats = ExecutionStats()
        self._resumable: set = set()

    # ---------- 公共入口 ----------
    def preview_resume(
        self, feed: Optional[Dict[str, np.ndarray]] = None
    ) -> RecoveryPlan:
        """恢复前的只读影响预览, 不执行任何节点、不改动旧检查点记录。

        返回 :class:`RecoveryPlan`: 哪些已完成节点可以复用、哪些必须重算
        以及原因 (输入改变 / 输入无法核验 / 上游重算 / 结果块损坏)。
        用户可以仅查看而不调用 :meth:`run`, 旧记录保持不变 (取消接续)。
        没有配置检查点时返回空计划。
        """
        if self.checkpoint is None:
            return RecoveryPlan(set(), {}, {}, {}, graph_match=True)
        return self.checkpoint.plan_recovery(self.graph, feed or {})

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
                with self._cond:
                    newly = self._pump_locked(pool)
                    for nid, fut in newly:
                        inflight[fut] = nid
                    self._record_inflight_peak_locked()
                if not inflight:
                    # 没有在跑任务: 可能全部跳过/失败, 或内存预算下暂时无法启动
                    with self._cond:
                        remaining = [
                            n for n in self.graph.nodes
                            if n not in self._done and n not in self._inflight
                        ]
                    if not remaining:
                        break
                    # 没有在途任务却仍有节点未完成: 内部状态不一致
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
                        self._cond.notify_all()
                with self._cond:
                    self._record_inflight_peak_locked()
        finally:
            pool.shutdown(wait=True)
            self.store.collect_garbage()
        return self._stats

    def result(self, block_id: str) -> np.ndarray:
        return self.store.get(block_id)

    # ---------- 初始化 ----------
    def _prepare(self, feed: Dict[str, np.ndarray]) -> None:
        checkpoint_done: set = set()
        if self.checkpoint is not None:
            # 任务身份 = 图结构 + 实际输入 + 节点依赖。先做只读的影响
            # 预览, 确认后才丢弃失效节点并记录新输入指纹; 只看图结构相同
            # 就复用旧节点会把旧输入的结果当成新输入的结果。
            plan = self.checkpoint.plan_recovery(self.graph, feed)
            checkpoint_done = self.checkpoint.apply_recovery(
                self.graph, plan, feed
            )
            self._resumable = set(checkpoint_done)
        # 生命周期登记: 每个块 *仍待执行* 的消费者数量。
        # 最终输出块 (无消费者) 保留, 调用方取走结果后可用 collect 显式回收。
        for bid in self.graph.all_blocks():
            live_consumers = [
                c for c in self.graph.consumers_of(bid) if c not in checkpoint_done
            ]
            self.store.register_block(
                bid,
                len(live_consumers),
                # 只有图的最终输出块 (本来就没有任何消费者) 才保留。
                # 不能因为“待执行消费者恰好为 0”就 retain——恢复时
                # 所有下游都在 pending, 那会把已恢复的中间块全部钉住。
                retain=not self.graph.consumers_of(bid),
            )
        # 外部喂入的源块 (登记其图内消费者数量, 避免被提前回收)
        for bid, val in feed.items():
            if bid not in {b for b in self.graph.all_blocks()}:
                live = [
                    c for c in self.graph.consumers_of(bid) if c not in checkpoint_done
                ]
                self.store.register_block(
                    bid, len(live), retain=not self.graph.consumers_of(bid)
                )
            self.store.put(bid, np.asarray(val))
        for nid, node in self.graph.nodes.items():
            deps = {self.graph.producer_of(bid) for bid in node.inputs}
            deps.discard(None)
            self._remaining_deps[nid] = len(deps)
            for d in deps:
                children = self._children.setdefault(d, [])
                if nid not in children:
                    children.append(nid)

        # 检查点恢复: 已核验节点直接标成功。块按需从磁盘惰性物化,
        # 而不是一次性全灌入内存 (否则紧预算下恢复即死锁)。
        if self.checkpoint is not None:
            self.store.set_loader(self.checkpoint.load_block)
            completed_on_disk = [
                nid for nid in self.graph.nodes if nid in checkpoint_done
            ]
            for nid in completed_on_disk:
                self._mark_published_locked(nid)
                self._done[nid] = NodeResult("success")
                self._stats.from_checkpoint.append(nid)
                self._stats.succeeded.append(nid)
                # 已恢复节点不会再走实时完成路径, 这里显式把其全部
                # 后继的剩余依赖数减掉, 否则后继永远停在 deps>0。
                for child in self._children.get(nid, ()):
                    self._remaining_deps[child] -= 1

    def _record_inflight_peak_locked(self) -> None:
        """记录峰值在途数据量: 驻留块 + 正在执行节点声明的工作内存。"""
        working = sum(
            self.graph.nodes[nid].memory_bytes for nid in self._inflight
        )
        peak = self.store.resident_bytes() + working
        if peak > self._stats.peak_inflight_bytes:
            self._stats.peak_inflight_bytes = peak

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
            # 先取消检查: 任一依赖失败/跳过则本节点跳过
            bad_dep = [
                d for d in self.graph.dependencies(nid)
                if d in self._done and self._done[d].status != "success"
            ]
            if bad_dep:
                self._skip_chain_locked(nid, cause=f"上游 {bad_dep[0]} 未成功")
                continue

            # 检查点块此时才真正从磁盘读入, 计入驻留内存后再判预算。
            for bid in node.inputs:
                self.store.preload(bid)
            need = self._required_memory_locked(node)
            if budget is not None:
                resident = self.store.resident_bytes()
                # 没有任何任务在途时, 内存已不会再被前序释放;
                # 此时继续排队必然死锁, 必须放行 (预算下的不可避峰值)。
                if self._inflight and resident + need > budget:
                    continue  # 真正的可选占用超预算: 排队等待前序释放
                if resident + need > budget:
                    overshoot = resident + need - budget
                    self._stats.budget_overshoot_bytes = max(
                        self._stats.budget_overshoot_bytes, overshoot
                    )

            # pin 输入, 保证执行期间不被回收
            inputs = [self.store.get(bid) for bid in node.inputs]
            for bid in node.inputs:
                self.store.pin(bid)
            self._pinned_inputs[nid] = list(node.inputs)

            fut = self._submit(pool, node, inputs, attempt=0)
            self._inflight[nid] = fut
            newly.append((nid, fut))
        return newly

    def _resolve_future_locked(self, nid: str, fut: Future) -> None:
        """future 完成后的统一处理 (在调用线程持锁执行, 无回调重入)。"""
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
                self._commit_checkpoint(node, outputs)
            except Exception as exc:
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
            if nid not in self._resumable and nid not in self._stats.recomputed:
                self._stats.recomputed.append(nid)
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

    def _commit_checkpoint(self, node: Node, outputs: Sequence[np.ndarray]) -> None:
        checkpoint = self.checkpoint
        if self._crash_after_block is None:
            checkpoint.commit_node(node.node_id, node.outputs, outputs)
        else:
            # 带硬中断钩子的检查点 (通常由独立子进程使用)。
            from .checkpoint import CheckpointStore as _CS

            crashing = _CS(str(checkpoint.path),
                           crash_after_block=self._crash_after_block)
            crashing.attach_graph(checkpoint.graph_id)
            # 保留已确认的输入指纹, 避免硬退出钩子写 meta 时把它清空。
            crashing._inputs = dict(checkpoint._inputs)
            crashing.commit_node(node.node_id, node.outputs, outputs)
        self._committed += 1
        if (self._exit_after_commits is not None
                and self._committed >= self._exit_after_commits):
            # 已提交若干完整节点后硬退出, 模拟“跑到一半”被杀死。
            os._exit(43)
