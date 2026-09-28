"""数据块（data block）抽象与带生命周期的块存储。

块是算子之间流动的唯一数据载体：算子读入若干 :class:`BlockHandle`，
产出一个 :class:`Block`。存储层负责：

- 内容落盘（``.npy``）与同目录原子 sidecar（``.json``），进程崩溃后仍可复用；
- 引用计数：图中节点完成后，不再被任何未完成节点需要的块被及时删除；
- 字节级内存计费：供调度器做内存预算准入。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Block:
    """算子产出的数据块：一个 numpy 数组加可选结构化元数据。"""

    data: np.ndarray
    meta: dict | None = None

    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass(frozen=True)
class BlockHandle:
    """对某个产出节点所发布块的引用句柄（值对象，可安全共享）。"""

    producer: str


class BlockStore:
    """线程安全的块存储：内存缓存 + 磁盘持久化 + 引用计数释放。"""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}
        self._refs: dict[str, int] = {}
        self._kept: set[str] = set()

    # ---- 发布 / 读取 ----

    def publish(self, producer: str, block: Block) -> int:
        """原子发布一个块；重复发布同一生产者时旧内容保留、不重复落盘。

        返回块字节数。写入采用临时文件 + ``os.replace``，保证不会留下
        半成品数据被后继读取。
        """

        data = np.ascontiguousarray(block.data)
        meta = dict(block.meta or {})
        token = uuid.uuid4().hex
        npy_tmp = self.root / f".{producer}.{token}.tmp"
        json_tmp = self.root / f".{producer}.{token}.json.tmp"
        npy_path = self._npy_path(producer)
        json_path = self._json_path(producer)
        np.save(str(npy_tmp) + ".npy", data, allow_pickle=False)
        npy_tmp = self.root / (npy_tmp.name + ".npy")
        with open(json_tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "nbytes": int(data.nbytes)}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(npy_tmp, npy_path)
        os.replace(json_tmp, json_path)
        with self._lock:
            already = producer in self._data
            self._data[producer] = data
            self._meta[producer] = meta
            self._refs.setdefault(producer, 0)
        return int(data.nbytes) if not already else int(data.nbytes)

    def get(self, handle: BlockHandle) -> Block:
        """读取块：优先内存缓存，否则从磁盘加载（崩溃恢复场景）。"""

        producer = handle.producer
        with self._lock:
            if producer in self._data:
                return Block(self._data[producer], dict(self._meta.get(producer, {})))
        npy_path = self._npy_path(producer)
        json_path = self._json_path(producer)
        if not npy_path.exists() or not json_path.exists():
            raise KeyError(f"block not found: {producer}")
        data = np.load(npy_path, allow_pickle=False)
        with open(json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload.get("meta", {})
        with self._lock:
            self._data.setdefault(producer, data)
            self._meta.setdefault(producer, meta)
            self._refs.setdefault(producer, 0)
        return Block(data, dict(meta))

    def has(self, producer: str) -> bool:
        with self._lock:
            if producer in self._data:
                return True
        return self._npy_path(producer).exists() and self._json_path(producer).exists()

    def evict_memory(self, producer: str) -> None:
        """把块从内存缓存中移除（磁盘保留），以释放内存预算。"""

        with self._lock:
            self._data.pop(producer, None)

    # ---- 生命周期：引用计数 ----

    def retain(self, producer: str) -> None:
        with self._lock:
            self._refs[producer] = self._refs.get(producer, 0) + 1

    def keep(self, producer: str) -> None:
        """声明块为最终结果：生命周期不受引用计数影响。"""

        with self._lock:
            self._kept.add(producer)

    def release(self, producer: str) -> int:
        """释放一次引用；返回本次真正释放的字节数（未归零或 keep 时为 0）。"""

        with self._lock:
            count = self._refs.get(producer, 0) - 1
            if count > 0:
                self._refs[producer] = count
                return 0
            self._refs[producer] = 0
            if producer in self._kept:
                return 0
            nbytes = int(self._data[producer].nbytes) if producer in self._data else 0
            self._data.pop(producer, None)
            self._meta.pop(producer, None)
        self._npy_path(producer).unlink(missing_ok=True)
        self._json_path(producer).unlink(missing_ok=True)
        return nbytes

    def nbytes(self, producer: str) -> int:
        """块字节数（内存或磁盘），未知返回 0。"""

        with self._lock:
            if producer in self._data:
                return int(self._data[producer].nbytes)
        json_path = self._json_path(producer)
        if json_path.exists():
            with open(json_path, encoding="utf-8") as fh:
                return int(json.load(fh).get("nbytes", 0))
        return 0

    # ---- 内部 ----

    def _npy_path(self, producer: str) -> Path:
        return self.root / f"{producer}.npy"

    def _json_path(self, producer: str) -> Path:
        return self.root / f"{producer}.json"
