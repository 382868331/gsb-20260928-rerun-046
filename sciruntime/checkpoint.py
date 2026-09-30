"""崩溃恢复检查点。

只有完整写出并通过校验的输出块, 随后其生产者节点进入 meta, 才能被恢复。
临时文件和未在 meta 中登记的孤立块不会被后继读取。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


class CheckpointError(RuntimeError):
    pass


def fingerprint_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256(arr.tobytes())
    digest.update(repr((arr.shape, arr.dtype.str)).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def fingerprint_feed(feed: Mapping[str, np.ndarray]) -> Dict[str, str]:
    return {name: fingerprint_array(feed[name]) for name in sorted(feed)}


class CheckpointStore:
    def __init__(self, path: str, *, graph_id: Optional[str] = None) -> None:
        self.path = path
        self._blocks_dir = os.path.join(path, "blocks")
        os.makedirs(self._blocks_dir, exist_ok=True)
        self._meta_path = os.path.join(path, "meta.json")
        self._graph_id = graph_id
        self._completed, meta_graph_id, self._feed_fingerprints = self._load_meta()
        if graph_id is not None and meta_graph_id is not None and graph_id != meta_graph_id:
            raise CheckpointError("检查点属于另一个计算图, 不能复用其节点")
        self._cleanup_staging_files()
        self.verify_completed()
        self._cleanup_orphan_blocks()

    def _load_meta(self) -> tuple[
        Dict[str, Dict[str, str]], Optional[str], Optional[Dict[str, str]]
    ]:
        if not os.path.exists(self._meta_path):
            return {}, None, None
        with open(self._meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise CheckpointError("无法识别的检查点 meta.json")
        if raw.get("version") == 2:
            completed = {
                nid: dict(blocks)
                for nid, blocks in raw.get("completed", {}).items()
            }
            return completed, raw.get("graph_id"), raw.get("feed_fingerprints")
        return (
            {nid: {bid: "" for bid in bids} for nid, bids in raw.items()},
            None,
            None,
        )

    def completed_nodes(self) -> list[str]:
        return list(self._completed)

    def is_completed(self, node_id: str) -> bool:
        return node_id in self._completed

    def bind_run(self, graph_id: str, feed_fingerprints: Mapping[str, str]) -> None:
        if self._graph_id is not None and self._graph_id != graph_id:
            raise CheckpointError("检查点属于另一个计算图, 不能复用其节点")
        feed = dict(feed_fingerprints)
        if self._feed_fingerprints is not None and self._feed_fingerprints != feed:
            raise CheckpointError("输入数据与检查点不一致, 不能复用旧结果")
        self._graph_id = graph_id
        self._feed_fingerprints = feed

    def _atomic_write_meta(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix="meta.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 2,
                        "graph_id": self._graph_id,
                        "feed_fingerprints": self._feed_fingerprints,
                        "completed": self._completed,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._meta_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _cleanup_staging_files(self) -> None:
        patterns = (
            os.path.join(self.path, "meta.*.tmp"),
            os.path.join(self._blocks_dir, "*.tmp"),
        )
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def _cleanup_orphan_blocks(self) -> None:
        referenced = {
            os.path.join(self._blocks_dir, f"{_safe(bid)}.npy")
            for blocks in self._completed.values()
            for bid in blocks
        }
        for path in glob.glob(os.path.join(self._blocks_dir, "*.npy")):
            if path not in referenced:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def commit_node(
        self,
        node_id: str,
        block_ids: Sequence[str],
        values: Sequence[np.ndarray],
    ) -> None:
        if node_id in self._completed:
            return
        written: Dict[str, str] = {}
        for bid, val in zip(block_ids, values):
            checksum = fingerprint_array(val)
            dest = os.path.join(self._blocks_dir, f"{_safe(bid)}.npy")
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
            written[bid] = checksum
        self._completed[node_id] = written
        self._atomic_write_meta()

    def load_block(self, block_id: str) -> np.ndarray:
        dest = os.path.join(self._blocks_dir, f"{_safe(block_id)}.npy")
        value = np.load(dest, allow_pickle=False)
        expected = next(
            (blocks[block_id] for blocks in self._completed.values()
             if block_id in blocks),
            "",
        )
        if expected and fingerprint_array(value) != expected:
            raise CheckpointError(f"检查点块校验失败: {block_id}")
        return value

    def load_completed(self, node_ids: Iterable[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for nid in node_ids:
            if nid not in self._completed:
                continue
            for bid in self._completed[nid]:
                out[bid] = self.load_block(bid)
        return out

    def verify_completed(self) -> None:
        """启动时核验磁盘上所有完整发布的块。"""
        for blocks in self._completed.values():
            for bid in blocks:
                if blocks[bid]:
                    self.load_block(bid)


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
