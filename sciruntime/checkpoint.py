"""检查点: 保存已完成节点与其输出块, 支持崩溃后恢复。

设计为一个目录:
  meta.json   已完成节点列表
  blocks/<block_id>.npy  各输出块

提交是原子的: 先写临时文件再 rename; 只有节点 *全部* 输出块都落盘后
才把节点记入 meta.json。因此检查点里不会出现“半成品节点”。
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Iterable, List, Sequence

import numpy as np


class CheckpointStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._blocks_dir = os.path.join(path, "blocks")
        os.makedirs(self._blocks_dir, exist_ok=True)
        self._meta_path = os.path.join(path, "meta.json")
        self._completed = self._load_meta()

    def _load_meta(self) -> Dict[str, List[str]]:
        if not os.path.exists(self._meta_path):
            return {}
        with open(self._meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        # node_id -> [block_id, ...]
        return {nid: list(bids) for nid, bids in raw.items()}

    def completed_nodes(self) -> List[str]:
        return list(self._completed)

    def is_completed(self, node_id: str) -> bool:
        return node_id in self._completed

    def _atomic_write_meta(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix="meta.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._completed, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._meta_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def commit_node(self, node_id: str, block_ids: Sequence[str], values: Sequence[np.ndarray]) -> None:
        """原子提交一个完成节点: 先落全部块, 再更新 meta。重复提交是幂等的。"""
        if node_id in self._completed:
            return
        written: List[str] = []
        for bid, val in zip(block_ids, values):
            dest = os.path.join(self._blocks_dir, f"{_safe(bid)}.npy")
            if not os.path.exists(dest):
                fd, tmp = tempfile.mkstemp(dir=self._blocks_dir, suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as f:
                        np.lib.format.write_array(f, np.asarray(val), allow_pickle=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, dest)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            written.append(bid)
        self._completed[node_id] = list(block_ids)
        self._atomic_write_meta()

    def load_block(self, block_id: str) -> np.ndarray:
        dest = os.path.join(self._blocks_dir, f"{_safe(block_id)}.npy")
        return np.load(dest, allow_pickle=False)

    def load_completed(self, node_ids: Iterable[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for nid in node_ids:
            if nid not in self._completed:
                continue
            for bid in self._completed[nid]:
                out[bid] = self.load_block(bid)
        return out


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
