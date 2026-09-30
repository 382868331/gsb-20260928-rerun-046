"""检查点: 保存已完成节点与其输出块, 支持硬中断后的细粒度恢复。

目录布局::

    meta.json                 图身份 + 已提交节点及各块 sha256
    blocks/<block_id>.npy     各输出块 (temp + rename 原子发布)

可复用的充要条件: 节点在 meta.json 中、其全部块文件存在且 sha256 与
meta 一致、所有前驱也通过同样的核验、**且该节点传递依赖的每个外部
输入块的内容指纹与当前喂入数据一致**。图结构相同不代表任务相同:
实际输入改变时, 只有依赖了被改输入的节点 (及其后继) 失效重算,
未受影响的分支保留。正在写出的块只存在于 *.tmp 或成为无 meta 记录的
孤儿文件, 恢复时一律清除并重算; 损坏块会导致该节点及其全部后继失效
重算, 后继不可能提前读到不可信数据。

恢复前可用 :meth:`CheckpointStore.plan_for_graph` 得到非破坏性的影响
预览 (哪些节点复用、哪些重算及原因); 用户取消时不会触碰任何已提交
记录。确认后 :meth:`apply_plan` 才会真正丢弃失效节点。
"""
from __future__ import annotations

import io
import json
import os
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np


class CheckpointStore:
    META_VERSION = 3

    def __init__(self, path: str, *, crash_after_block: Optional[str] = None) -> None:
        """crash_after_block: 测试钩子——该块文件落盘后、meta 更新前
        立即 os._exit, 模拟进程在“写出一半”时被硬杀掉。"""
        self.path = path
        self._blocks_dir = os.path.join(path, "blocks")
        os.makedirs(self._blocks_dir, exist_ok=True)
        self._meta_path = os.path.join(path, "meta.json")
        self._crash_after_block = crash_after_block
        self._graph_id: Optional[str] = None
        # 外部输入块 block_id -> 上次运行时的内容 sha256
        self._inputs: Dict[str, str] = {}
        # node_id -> [{"block": ..., "sha256": ...}, ...]
        self._nodes: Dict[str, List[Dict[str, str]]] = {}
        # node_id -> [该节点传递依赖的外部输入块 id]
        self._node_sources: Dict[str, List[str]] = {}
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
            self._inputs = dict(raw.get("inputs", {}))
            self._nodes = {
                nid: [dict(item) for item in _node_blocks(items)]
                for nid, items in raw.get("nodes", {}).items()
            }
            self._node_sources = {
                nid: list(_node_sources(items))
                for nid, items in raw.get("nodes", {}).items()
            }
            return {nid: [item["block"] for item in items]
                    for nid, items in self._nodes.items()}
        if isinstance(raw, dict) and raw.get("version") == 2:
            # v2: 结构身份相同但没有输入内容背书。迁移后所有依赖外部
            # 输入的节点必须重算 (源指纹缺失即视为不可信)。
            self._graph_id = raw.get("graph_id")
            self._nodes = {
                nid: [dict(item) for item in _node_blocks(items)]
                for nid, items in raw.get("nodes", {}).items()
            }
            self._node_sources = {
                nid: list(_node_sources(items))
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

    def input_fingerprints(self) -> Dict[str, str]:
        """检查点记录的各外部输入块内容指纹。"""
        return dict(self._inputs)

    def record_input_fingerprints(self, fingerprints: Mapping[str, str]) -> None:
        """登记当前运行实际喂入的外部输入内容指纹并落盘。

        旧输入的节点若与新指纹冲突, 会在随后的 plan/apply 阶段被判定
        失效; 仍然存活的旧节点只可能依赖未改变的输入。
        """
        self._inputs.update(dict(fingerprints))
        self._atomic_write_meta()

    def _meta_payload(self) -> Dict[str, object]:
        nodes = {}
        for nid, items in self._nodes.items():
            nodes[nid] = {
                "blocks": [dict(item) for item in items],
                "sources": list(self._node_sources.get(nid, ())),
            }
        return {
            "version": self.META_VERSION,
            "graph_id": self._graph_id,
            "inputs": dict(self._inputs),
            "nodes": nodes,
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

    def commit_node(
        self,
        node_id: str,
        block_ids: Sequence[str],
        values: Sequence[np.ndarray],
        sources: Sequence[str] = (),
    ) -> None:
        """原子提交一个完成节点: 先落全部块, 再更新 meta。重复提交是幂等的。"""
        if node_id in self._completed:
            return
        items: List[Dict[str, str]] = []
        for bid, val in zip(block_ids, values):
            dest = os.path.join(self._blocks_dir, f"{_safe(bid)}.npy")
            # 失效重算可能复用相同 block_id (例如同构新图换了输入)。
            # 不能因为文件还在就跳过写入——那会把旧输入的块当成新结果。
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
            items.append({"block": bid, "sha256": digest})
        self._node_sources[node_id] = sorted(set(sources))
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

    def plan_for_graph(
        self,
        topo_order: Sequence[str],
        predecessors: Dict[str, Set[str]],
        node_sources: Optional[Mapping[str, Set[str]]] = None,
        input_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> "ResumePlan":
        """非破坏性的恢复影响预览。

        按拓扑顺序判定每个已提交节点能否复用, 绝不删除/改写任何记录。
        判定依次考虑: 输入内容是否改变、块文件是否完整、前驱是否可信。
        """
        node_sources = node_sources or {}
        input_fingerprints = input_fingerprints or {}
        valid: Set[str] = set()
        reusable: List[NodeImpact] = []
        recompute: List[NodeImpact] = []
        changed_inputs = sorted(
            bid for bid, old in self._inputs.items()
            if input_fingerprints.get(bid) != old
        )
        for nid in topo_order:
            if nid not in self._completed:
                recompute.append(NodeImpact(nid, "该节点尚未提交 (首次执行)"))
                continue
            sources = set(self._node_sources.get(nid, ())) | set(
                node_sources.get(nid, ())
            )
            stale = sorted(
                bid for bid in sources
                if bid not in input_fingerprints
                or self._inputs.get(bid) != input_fingerprints[bid]
            )
            missing_inputs = sorted(
                bid for bid in stale if bid not in input_fingerprints
            )
            changed = sorted(bid for bid in stale if bid not in missing_inputs)
            bad_preds = sorted(
                d for d in predecessors.get(nid, set()) if d not in valid
            )
            files_ok = self._node_files_ok(nid)
            if changed:
                reason = f"外部输入 {', '.join(changed)} 内容已改变, 依赖它的结果必须重算"
            elif missing_inputs:
                reason = f"外部输入 {', '.join(missing_inputs)} 本次未提供或无内容指纹"
            elif not files_ok:
                reason = "结果块缺失或 sha256 校验失败 (损坏/写出一半)"
            elif bad_preds:
                reason = f"前驱节点 {', '.join(bad_preds[:3])} 必须重算, 本节点随之失效"
            else:
                valid.add(nid)
                reusable.append(NodeImpact(
                    nid,
                    "块完整、校验和一致、前驱可信且实际输入未改变",
                ))
                continue
            recompute.append(NodeImpact(nid, reason))
        return ResumePlan(
            graph_id=self._graph_id or "",
            reusable=reusable,
            recompute=recompute,
            changed_inputs=changed_inputs,
        )

    def apply_plan(self, plan: "ResumePlan") -> Set[str]:
        """按预览结果丢弃失效节点的 meta 记录与块文件, 返回复用集合。"""
        valid = {item.node_id for item in plan.reusable}
        bad = [nid for nid in list(self._nodes) if nid not in valid]
        if bad:
            for nid in bad:
                self._discard_node(nid)
            self._atomic_write_meta()
        self._remove_orphan_files()
        return set(valid)

    def verify_for_graph(
        self,
        topo_order: Sequence[str],
        predecessors: Dict[str, Set[str]],
        node_sources: Optional[Mapping[str, Set[str]]] = None,
        input_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> Set[str]:
        """兼容旧调用: 直接生成预览并应用 (破坏性)。"""
        plan = self.plan_for_graph(
            topo_order, predecessors,
            node_sources=node_sources,
            input_fingerprints=input_fingerprints,
        )
        return self.apply_plan(plan)

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


class ResumeCancelled(RuntimeError):
    """用户在查看恢复影响预览后取消接续; 检查点记录保持原样。"""


@dataclass
class NodeImpact:
    node_id: str
    reason: str


@dataclass
class ResumePlan:
    graph_id: str
    reusable: List[NodeImpact] = field(default_factory=list)
    recompute: List[NodeImpact] = field(default_factory=list)
    changed_inputs: List[str] = field(default_factory=list)

    @property
    def reusable_nodes(self) -> List[str]:
        return [item.node_id for item in self.reusable]

    @property
    def recompute_nodes(self) -> List[str]:
        return [item.node_id for item in self.recompute]

    def describe(self) -> str:
        lines = [
            f"图身份: {self.graph_id[:12]}",
            f"改变的外部输入: {', '.join(self.changed_inputs) or '(无)'}",
            f"可复用节点 ({len(self.reusable)}):",
        ]
        lines += [f"  = {item.node_id}: {item.reason}" for item in self.reusable]
        lines.append(f"必须重算节点 ({len(self.recompute)}):")
        lines += [f"  ! {item.node_id}: {item.reason}" for item in self.recompute]
        return "\n".join(lines)


def fingerprint_array(value: np.ndarray) -> str:
    """计算数组实际内容 (形状/dtype/字节序/原始数据) 的 sha256。

    使用与块落盘相同的 .npy 序列化, 形状或 dtype 改变也会改变指纹。
    """
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(value),
                              allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _node_blocks(items: object) -> list:
    """v3 节点记录取 blocks; 兼容直接是列表的旧格式。"""
    if isinstance(items, dict):
        return list(items.get("blocks", []))
    return list(items)  # type: ignore[arg-type]


def _node_sources(items: object) -> list:
    if isinstance(items, dict):
        return list(items.get("sources", []))
    return []


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
