"""检查点: 保存已完成节点与其输出块, 支持硬中断后的细粒度恢复。

目录布局::

    meta.json                 图身份 + 已提交节点及各块 sha256
    blocks/<block_id>.npy     各输出块 (temp + rename 原子发布)

可复用的充要条件: 节点在 meta.json 中、其全部块文件存在且 sha256 与
meta 一致、且所有前驱也通过同样的核验。正在写出的块只存在于 *.tmp 或
成为无 meta 记录的孤儿文件, 恢复时一律清除并重算; 损坏块会导致该节点
及其全部后继失效重算, 后继不可能提前读到不可信数据。
"""
from __future__ import annotations

import json
import os
import hashlib
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


class CheckpointStore:
    META_VERSION = 2

    def __init__(self, path: str, *, crash_after_block: Optional[str] = None) -> None:
        """crash_after_block: 测试钩子——该块文件落盘后、meta 更新前
        立即 os._exit, 模拟进程在“写出一半”时被硬杀掉。"""
        self.path = path
        self._blocks_dir = os.path.join(path, "blocks")
        os.makedirs(self._blocks_dir, exist_ok=True)
        self._meta_path = os.path.join(path, "meta.json")
        self._crash_after_block = crash_after_block
        self._graph_id: Optional[str] = None
        # node_id -> [{"block": ..., "sha256": ...}, ...]
        self._nodes: Dict[str, List[Dict[str, str]]] = {}
        self._completed = self._load_meta()
        self._cleanup_temp_files()

    def _load_meta(self) -> Dict[str, List[str]]:
        """兼容 v1 读取; 返回 node_id -> [block_id, ...] 的视图。"""
        if not os.path.exists(self._meta_path):
            return {}
        with open(self._meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw.get("version") == self.META_VERSION:
            self._graph_id = raw.get("graph_id")
            self._nodes = {
                nid: [dict(item) for item in items]
                for nid, items in raw.get("nodes", {}).items()
            }
            return {nid: [item["block"] for item in items]
                    for nid, items in self._nodes.items()}
        # v1: {node_id: [block_id, ...]}
        if isinstance(raw, dict):
            return {nid: list(bids) for nid, bids in raw.items()}
        return {}

    @property
    def graph_id(self) -> Optional[str]:
        return self._graph_id

    def attach_graph(self, graph_id: str) -> None:
        """绑定图身份。检查点只能被结构完全相同的图复用。"""
        if self._graph_id is None:
            self._graph_id = graph_id
            self._atomic_write_meta()
        elif self._graph_id != graph_id:
            raise CheckpointMismatch(
                f"检查点图身份 {self._graph_id[:12]} 与当前图 {graph_id[:12]} 不一致"
            )

    def _cleanup_temp_files(self) -> None:
        for name in os.listdir(self._blocks_dir):
            if name.endswith(".tmp"):
                try:
                    os.unlink(os.path.join(self._blocks_dir, name))
                except OSError:
                    pass

    def completed_nodes(self) -> List[str]:
        return list(self._completed)

    def is_completed(self, node_id: str) -> bool:
        return node_id in self._completed

    def _meta_payload(self) -> Dict[str, object]:
        return {
            "version": self.META_VERSION,
            "graph_id": self._graph_id,
            "nodes": self._nodes,
        }

    def _atomic_write_meta(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix="meta-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._meta_payload(), f, ensure_ascii=False, indent=2)
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
        items: List[Dict[str, str]] = []
        for bid, val in zip(block_ids, values):
            dest = os.path.join(self._blocks_dir, f"{_safe(bid)}.npy")
            if not os.path.exists(dest):
                fd, tmp = tempfile.mkstemp(dir=self._blocks_dir, suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as f:
                        np.lib.format.write_array(f, np.asarray(val), allow_pickle=False)
                        f.flush()
                        os.fsync(f.fileno())
                    digest = _sha256_file(tmp)
                    os.replace(tmp, dest)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
                if self._crash_after_block == bid:
                    # 块已 rename 但 meta 尚未更新: 模拟硬中断。
                    os._exit(42)
                digest = _sha256_file(dest)
            else:
                digest = _sha256_file(dest)
            items.append({"block": bid, "sha256": digest})
        self._nodes[node_id] = items
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

    # ----- 恢复时核验 -----
    def _block_path(self, block_id: str) -> str:
        return os.path.join(self._blocks_dir, f"{_safe(block_id)}.npy")

    def _node_files_ok(self, node_id: str) -> bool:
        """节点的全部块文件存在且 sha256 与 meta 一致。"""
        items = self._nodes.get(node_id)
        if not items:
            return False
        for item in items:
            path = self._block_path(item["block"])
            if not os.path.exists(path):
                return False
            try:
                if _sha256_file(path) != item.get("sha256"):
                    return False
            except OSError:
                return False
        return True

    def verify_for_graph(
        self,
        topo_order: Sequence[str],
        predecessors: Dict[str, Set[str]],
    ) -> Set[str]:
        """按拓扑顺序核验检查点, 返回可复用节点集合。

        任一前驱不可复用则本节点也不可复用 (闭包失效); 失效节点的
        meta 记录与块文件被删除, 磁盘上只留下通过核验的提交。
        """
        valid: Set[str] = set()
        for nid in topo_order:
            if nid not in self._completed:
                continue
            if predecessors.get(nid, set()) <= valid and self._node_files_ok(nid):
                valid.add(nid)

        bad = [nid for nid in list(self._nodes) if nid not in valid]
        if bad:
            for nid in bad:
                self._discard_node(nid)
            self._atomic_write_meta()
        self._remove_orphan_files()
        return valid

    def _discard_node(self, node_id: str) -> None:
        items = self._nodes.pop(node_id, None)
        self._completed.pop(node_id, None)
        for item in items or ():
            try:
                os.unlink(self._block_path(item["block"]))
            except OSError:
                pass

    def _remove_orphan_files(self) -> None:
        referenced = {
            f"{_safe(item['block'])}.npy"
            for items in self._nodes.values() for item in items
        }
        for name in os.listdir(self._blocks_dir):
            if name.endswith(".npy") and name not in referenced:
                try:
                    os.unlink(os.path.join(self._blocks_dir, name))
                except OSError:
                    pass


class CheckpointMismatch(RuntimeError):
    pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
