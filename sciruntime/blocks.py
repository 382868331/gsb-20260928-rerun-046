"""数据块 (Block) 抽象与 BlockStore。

Block 只是 *元数据* (id / shape / dtype / 由哪个节点产生)。真正的 numpy
数组存放在 BlockStore 中, 并由引用计数管理生命周期:

- completed 节点发布时调用 store.put(block_id, ndarray);
- 消费方 (后继节点) 开始执行时 pin 住输入, 结束后 release;
- 当某块的引用计数归零且它的所有下游都已结束, store 可以回收内存;
- block 也支持溢出到磁盘 (spill), 用于超出内存预算时的排队控制。

BlockStore 本身不假设数据类型, 任何支持 numpy 协议的数组均可。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Block:
    """一个数据块的不可变元数据。"""

    block_id: str
    shape: Tuple[int, ...] = ()
    dtype: str = "float64"
    producer: Optional[str] = None  # 产生该块的节点 id
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        if not self.shape:
            return 0
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


class BlockNotFoundError(KeyError):
    """访问一个尚未产生或已被回收的数据块。"""


class BlockStore:
    """线程安全的数据块仓库, 负责引用计数与内存生命周期。

    每个 block_id 对应一个不可变数组 (放入后不再修改), 因此:
    - 多个消费者可以安全共享同一份内存;
    - 节点重试不会看到“半成品”——失败的写入在成功提交前对任何人不可见。
    """

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._data: Dict[str, np.ndarray] = {}
        self._pins: Dict[str, int] = {}
        # block_id -> 还剩余多少个 (尚未结束的) 节点需要消费它
        self._remaining_consumers: Dict[str, int] = {}
        # 最终输出块: 消费者数为 0, 保留到调用方显式 release_output
        self._retained_outputs: set = set()

    # ----- 调度器登记的生命周期信息 -----
    def register_block(self, block_id: str, consumer_count: int, *, retain: bool = False) -> None:
        """登记一个将要产生的块及其下游消费者数量。"""
        with self._lock:
            if block_id not in self._remaining_consumers:
                self._remaining_consumers[block_id] = consumer_count
                if retain:
                    self._retained_outputs.add(block_id)

    # ----- 发布 (必须原子且一次性) -----
    def put(self, block_id: str, value: np.ndarray) -> None:
        """发布一个块。重复发布同一 block_id 会被拒绝。

        运行时正常流程不会对同一 block_id 调用两次 put (节点发布是幂等的);
        这里仍然防御性拒绝, 避免重复发布污染结果。
        """
        with self._lock:
            if block_id in self._data:
                raise RuntimeError(f"block {block_id!r} 已经发布过, 不允许重复发布")
            self._data[block_id] = np.asarray(value)
            self._pins.setdefault(block_id, 0)
            self._lock.notify_all()

    def contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._data

    # ----- 消费 -----
    def wait_for(self, block_id: str, timeout: Optional[float] = None) -> np.ndarray:
        with self._lock:
            if not self._lock.wait_for(lambda: block_id in self._data, timeout):
                raise BlockNotFoundError(block_id)
            return self._data[block_id]

    def get(self, block_id: str) -> np.ndarray:
        with self._lock:
            try:
                return self._data[block_id]
            except KeyError:
                raise BlockNotFoundError(block_id) from None

    # ----- 引用计数 -----
    def pin(self, block_id: str) -> None:
        with self._lock:
            self._pins[block_id] = self._pins.get(block_id, 0) + 1

    def release(self, block_id: str) -> None:
        """解除一次 pin。消费者生命周期计数由 consumer_finished 单独维护。"""
        with self._lock:
            n = self._pins.get(block_id, 0)
            if n <= 0:
                raise RuntimeError(f"block {block_id!r} 引用计数为负")
            self._pins[block_id] = n - 1
            self._lock.notify_all()

    def consumer_finished(self, block_id: str) -> None:
        """某消费者结束 (无论成功失败), 减少待消费计数, 归零即可回收。"""
        with self._lock:
            if self._remaining_consumers.get(block_id, 0) > 0:
                self._remaining_consumers[block_id] -= 1
            self._lock.notify_all()

    def is_collectable(self, block_id: str) -> bool:
        with self._lock:
            return (
                block_id in self._data
                and self._pins.get(block_id, 0) == 0
                and self._remaining_consumers.get(block_id, 0) <= 0
                and block_id not in self._retained_outputs
            )

    def release_output(self, block_id: str) -> None:
        """调用方取走最终输出后, 允许回收该块。"""
        with self._lock:
            self._retained_outputs.discard(block_id)
            self._lock.notify_all()

    def collect(self, block_id: str) -> bool:
        """回收一个块的内存, 返回是否真的回收。"""
        with self._lock:
            if self.is_collectable(block_id):
                self._data.pop(block_id, None)
                return True
            return False

    def collect_garbage(self) -> int:
        with self._lock:
            dead = [bid for bid in self._data if self.is_collectable(bid)]
            for bid in dead:
                self._data.pop(bid, None)
            return len(dead)

    # ----- 内存统计 -----
    def resident_bytes(self) -> int:
        with self._lock:
            return int(sum(v.nbytes for v in self._data.values()))

    def block_ids(self) -> list[str]:
        with self._lock:
            return list(self._data)

    # ----- 持久化支持 (恢复时直接灌入已完成块) -----
    def snapshot(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return dict(self._data)

    def restore(self, items: Dict[str, np.ndarray]) -> None:
        with self._lock:
            for bid, val in items.items():
                self._data.setdefault(bid, np.asarray(val))
            self._lock.notify_all()
