"""检查点: 保存已完成节点与其输出块, 支持硬中断后的细粒度恢复。

目录布局::

    meta.json                 图身份 + 已提交节点及各块 sha256
    blocks/<block_id>.npy     各输出块 (temp + rename 原子发布)

任务身份 = 图结构身份 (节点/算子/块/参数) **加上实际喂入源块的内容**。
图形结构没有变化不等于任务相同: 输入数组改变后, 所有 (传递) 依赖该源块
的旧节点都必须失效重算, 不能把旧完成标记当成新输入已处理; 未依赖被改动
源块的分支则可以原样保留。

可复用的充要条件: 节点在 meta.json 中、其传递依赖的全部源块指纹与当前
输入一致、其全部块文件存在且 sha256 与 meta 一致、且所有直接前驱也通过
同样的核验。正在写出的块只存在于 *.tmp 或成为无 meta 记录的孤儿文件,
恢复时一律清除并重算; 损坏块会导致该节点及其全部后继失效重算, 后继不
可能提前读到不可信数据。

恢复分两步: :meth:`CheckpointStore.plan_recovery` 只做只读的影响预览
(区分输入变化与计算关系变化), 不改动盘上任何旧记录, 调用方可取消;
:meth:`CheckpointStore.apply_recovery` 用于原图同目录接续,
:meth:`CheckpointStore.derive` 用于从旧检查点非破坏性派生多个新图分支。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


class CheckpointStore:
    META_VERSION = 4

    def __init__(self, path: str, *, crash_after_block: Optional[str] = None) -> None:
        """crash_after_block: 测试钩子——该块文件落盘后、meta 更新前
        立即 os._exit, 模拟进程在“写出一半”时被硬杀掉。"""
        self.path = path
        self._blocks_dir = os.path.join(path, "blocks")
        os.makedirs(self._blocks_dir, exist_ok=True)
        self._meta_path = os.path.join(path, "meta.json")
        self._crash_after_block = crash_after_block
        self._graph_id: Optional[str] = None
        self._graph_manifest: Optional[List[Dict[str, Any]]] = None
        self._revision_id: Optional[str] = None
        self._lineage: Dict[str, Any] = {}
        # 本次检查点运行时实际喂入的源块内容指纹 block_id -> sha256
        self._inputs: Dict[str, str] = {}
        # node_id -> [{"block": ..., "sha256": ...}, ...]
        self._nodes: Dict[str, List[Dict[str, str]]] = {}
        # node_id -> 每份结果最初由哪个图/节点产生
        self._provenance: Dict[str, Dict[str, Any]] = {}
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
            self._graph_manifest = raw.get("graph_manifest")
            self._revision_id = raw.get("revision_id")
            self._lineage = dict(raw.get("lineage", {}))
            self._inputs = dict(raw.get("inputs", {}))
            self._nodes = {
                nid: [dict(item) for item in items]
                for nid, items in raw.get("nodes", {}).items()
            }
            self._provenance = {
                nid: dict(value) for nid, value in raw.get("provenance", {}).items()
            }
            return {nid: [item["block"] for item in items]
                    for nid, items in self._nodes.items()}
        # v1/v2: v2 有 graph_id 与 nodes 但没有输入指纹;
        # v1 是 {node_id: [block_id, ...]}。
        # 两者都没有源块内容记录, 输入身份无法核验, 按不可信处理
        # (input_unverified), 旧节点全部重算, 绝不盲目复用。
        if isinstance(raw, dict):
            self._graph_id = raw.get("graph_id")
            self._revision_id = raw.get("revision_id")
            nodes = raw.get("nodes")
            if isinstance(nodes, dict):
                self._nodes = {
                    nid: [dict(item) for item in items]
                    for nid, items in nodes.items()
                }
                return {nid: [item["block"] for item in items]
                        for nid, items in self._nodes.items()}
            return {
                nid: list(bids) for nid, bids in raw.items()
                if isinstance(nid, str) and isinstance(bids, list)
            }
        return {}

    @property
    def graph_id(self) -> Optional[str]:
        return self._graph_id

    @property
    def revision_id(self) -> str:
        if self._revision_id is None:
            self._revision_id = uuid.uuid4().hex
        return self._revision_id

    def provenance(self, node_id: str) -> Optional[Dict[str, Any]]:
        return dict(self._provenance.get(node_id, {})) or None

    def attach_graph(self, graph_id: str, manifest: Optional[List[Dict[str, Any]]] = None) -> None:
        """绑定图身份。检查点只能被结构完全相同的图复用。"""
        if self._graph_id is None:
            self._graph_id = graph_id
            self._graph_manifest = manifest
        elif self._graph_id != graph_id:
            raise CheckpointMismatch(
                f"检查点图身份 {self._graph_id[:12]} 与当前图 {graph_id[:12]} 不一致"
            )
        elif manifest is not None and self._graph_manifest is None:
            self._graph_manifest = manifest

    @classmethod
    def derive(
        cls,
        parent_path: str,
        child_path: str,
        graph,
        feed: Mapping[str, np.ndarray],
        *,
        branch_id: Optional[str] = None,
    ) -> Tuple["CheckpointStore", "RecoveryPlan"]:
        """从旧检查点非破坏性派生一个新图分支, 不修改旧检查点。

        只有当前图能够按结构配方和数据血缘解释的节点会复制到子检查点。
        子分支失败或被删除都不会影响父检查点或其他派生分支。
        """
        parent = cls(parent_path)
        plan = parent.plan_recovery(graph, feed)
        if os.path.abspath(child_path) == os.path.abspath(parent_path):
            raise ValueError("派生检查点不能使用父检查点同一目录")
        if os.path.exists(child_path) and os.listdir(child_path):
            raise ValueError(f"派生目标目录非空: {child_path}")

        os.makedirs(os.path.join(child_path, "blocks"), exist_ok=True)
        child = cls(child_path)
        child._graph_id = graph.identity()
        child._graph_manifest = graph.manifest()
        child._revision_id = branch_id or uuid.uuid4().hex
        child._inputs = dict(cls.fingerprint_feed(feed))
        child._lineage = {
            "parent_path": os.path.abspath(parent.path),
            "parent_revision_id": parent.revision_id,
            "parent_graph_id": parent.graph_id,
        }

        for nid in sorted(plan.reusable):
            child._nodes[nid] = [dict(item) for item in parent._nodes[nid]]
            child._completed[nid] = [item["block"] for item in child._nodes[nid]]
            inherited = parent._provenance.get(nid, {
                "origin_graph_id": parent.graph_id,
                "origin_node_id": nid,
                "origin_revision_id": parent.revision_id,
            })
            child._provenance[nid] = {
                **dict(inherited),
                "derived_from_graph_id": parent.graph_id,
                "derived_from_node_id": nid,
                "derived_from_revision_id": parent.revision_id,
                "derived_from_path": os.path.abspath(parent.path),
            }
            for item in parent._nodes[nid]:
                shutil.copy2(
                    parent._block_path(item["block"]),
                    child._block_path(item["block"]),
                )
        child._atomic_write_meta()
        return child, plan

    # ----- 输入身份 -----
    @staticmethod
    def fingerprint_array(value: np.ndarray) -> str:
        """数组内容指纹: dtype/shape/字节序 + 原始字节的 sha256。

        同内容数组即使是不同对象也得到相同指纹; 任何元素改动都会改变指纹。
        """
        arr = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(repr((str(arr.dtype), arr.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(arr.tobytes())
        return h.hexdigest()

    @classmethod
    def fingerprint_feed(
        cls, feed: Mapping[str, np.ndarray]
    ) -> Dict[str, str]:
        return {bid: cls.fingerprint_array(val) for bid, val in feed.items()}

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
            "revision_id": self.revision_id,
            "lineage": self._lineage,
            "graph_id": self._graph_id,
            "graph_manifest": self._graph_manifest,
            "inputs": self._inputs,
            "nodes": self._nodes,
            "provenance": self._provenance,
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
        self._provenance[node_id] = {
            "origin_graph_id": self._graph_id,
            "origin_node_id": node_id,
            "origin_revision_id": self.revision_id,
        }
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

    def plan_recovery(
        self,
        graph,
        feed: Mapping[str, np.ndarray],
    ) -> "RecoveryPlan":
        """恢复前的只读影响预览: 不修改 meta、不删除任何块文件。

        按拓扑顺序逐个判定已提交节点能否复用, 返回
        :class:`RecoveryPlan`, 说明每个节点复用 / 重算及其原因。
        判定依据包含实际输入 (源块内容指纹) 与节点依赖: 图结构不变但
        输入改变时, 仅 (传递) 依赖被改动源块的分支失效; 结果损坏的节点
        及其全部后继也失效。调用方审阅后可放弃 (直接丢弃 plan), 盘上
        旧记录保持原样; 或调用 :meth:`apply_recovery` 确认接续。
        """
        topo_order = graph.topological_order()
        current = self.fingerprint_feed(feed)
        recorded = self._inputs

        reusable: Set[str] = set()
        recompute: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
        changed_sources: Dict[str, str] = {}
        changed_relations: Dict[str, str] = {}
        missing_sources: Set[str] = set()
        new_nodes: Set[str] = set()
        removed_nodes: Set[str] = set()
        current_nodes = set(graph.nodes)
        old_manifest = {
            item["id"]: item for item in (self._graph_manifest or [])
        }
        current_manifest = {item["id"]: item for item in graph.manifest()}
        graph_match = self._graph_id is not None and self._graph_id == graph.identity()
        graph_unverified = (
            self._graph_id is None
            or (self._graph_manifest is None and not graph_match)
        )
        old_producers = {
            block: item["id"]
            for item in old_manifest.values()
            for block in item.get("outputs", ())
        }

        for nid, item in old_manifest.items():
            if nid not in current_manifest:
                removed_nodes.add(nid)

        for bid in sorted(set(graph.source_blocks()) | set(recorded) | set(current)):
            old, new = recorded.get(bid), current.get(bid)
            if old is None and new is None:
                continue
            if new is None:
                changed_sources[bid] = "source_removed"
            elif old is None:
                # 旧记录没有该输入的指纹 (如 v2 检查点或新出现的源块):
                # 无法证明旧节点处理的就是这份输入。
                missing_sources.add(bid)
                changed_sources[bid] = "input_unverified"
            elif old != new:
                changed_sources[bid] = "input_changed"

        for nid in topo_order:
            if nid not in self._completed:
                if nid not in old_manifest or nid in self._nodes:
                    new_nodes.add(nid)
                continue  # 旧图没有的节点正常执行, 不属于“旧结果失效”
            node = graph.nodes[nid]
            bad_dep = next(
                (d for d in sorted(graph.dependencies(nid)) if d not in reusable),
                None,
            )
            sources = graph.transitive_sources(nid)
            if graph_unverified:
                reason = "graph_unverified"
            elif not graph_match:
                old_node = old_manifest.get(nid)
                if old_node is None:
                    reason = "graph_changed:node_added"
                else:
                    relation_bad = self._relation_change_reason(
                        graph, nid, reusable, old_manifest,
                        old_producers, changed_relations,
                    )
                    if relation_bad is not None:
                        reason = relation_bad
                        changed_relations[nid] = reason
                    elif old_node != current_manifest[nid]:
                        reason = "graph_changed:node_recipe"
                        changed_relations[nid] = reason
                    elif bad_dep is not None:
                        reason = f"dependency:{bad_dep}"
                    elif sources & missing_sources:
                        hit = sorted(sources & missing_sources)
                        reason = f"input_unverified:{hit[0]}"
                    else:
                        hit = sorted(sources & set(changed_sources))
                        if hit:
                            bid = hit[0]
                            reason = f"{changed_sources[bid]}:{bid}"
                        elif not self._node_files_ok(nid):
                            reason = "corrupt_or_missing_block"
                        else:
                            reusable.add(nid)
                            continue
            elif bad_dep is not None:
                reason = f"dependency:{bad_dep}"
            elif sources & missing_sources:
                hit = sorted(sources & missing_sources)
                reason = f"input_unverified:{hit[0]}"
            else:
                hit = sorted(sources & set(changed_sources))
                if hit:
                    bid = hit[0]
                    reason = f"{changed_sources[bid]}:{bid}"
                elif not self._node_files_ok(nid):
                    reason = "corrupt_or_missing_block"
                else:
                    reusable.add(nid)
                    continue
            recompute[nid] = reason
            reasons[nid] = reason
            if reason.startswith("graph_changed:"):
                changed_relations[nid] = reason

        return RecoveryPlan(
            reusable=reusable,
            recompute=recompute,
            reasons=reasons,
            changed_sources=changed_sources,
            graph_match=graph_match,
            changed_relations=changed_relations,
            new_nodes=new_nodes,
            removed_nodes=removed_nodes,
            current_nodes=current_nodes,
        )

    def _relation_change_reason(
        self,
        graph,
        node_id: str,
        reusable: Set[str],
        old_manifest: Mapping[str, Dict[str, Any]],
        old_producers: Mapping[str, str],
        changed_relations: Mapping[str, str],
    ) -> Optional[str]:
        """识别同名节点的输入边/前驱是否已经换成另一套计算关系。"""
        current_deps = graph.dependencies(node_id)
        old_node = old_manifest.get(node_id)
        if old_node is None:
            return "graph_changed:node_added"
        old_inputs = set(old_node.get("inputs", []))
        current_inputs = set(graph.nodes[node_id].inputs)
        if old_inputs != current_inputs:
            return "graph_changed:inputs"
        for bid in graph.nodes[node_id].inputs:
            current_producer = graph.producer_of(bid)
            if (current_producer is not None
                    and old_producers.get(bid) != current_producer):
                return f"graph_changed:producer:{current_producer}"
        for dep in current_deps:
            if dep not in reusable and (
                dep not in old_manifest or dep in changed_relations
            ):
                return f"graph_changed:dependency:{dep}"
        return None

    def apply_recovery(
        self,
        graph,
        plan: "RecoveryPlan",
        feed: Mapping[str, np.ndarray],
    ) -> Set[str]:
        """确认影响预览后真正接续: 丢弃失效节点, 记录当前输入指纹。

        只有调用本方法才会修改盘上记录; 被丢弃节点的 meta 记录与块文件
        被删除, 磁盘上只留下通过核验的提交。返回可复用节点集合。
        """
        self.attach_graph(graph.identity(), graph.manifest())
        # 闭包失效: 计划外的任何已提交节点 (例如本次预览之后又被判定失效
        # 的传递后继) 一并丢弃。
        reusable = set(plan.reusable)
        bad = [nid for nid in list(self._nodes) if nid not in reusable]
        for nid in bad:
            self._discard_node(nid)
        for nid in reusable:
            if nid not in self._provenance:
                self._provenance[nid] = {
                    "origin_graph_id": self._graph_id,
                    "origin_node_id": nid,
                    "origin_revision_id": self.revision_id,
                }
        self._inputs = dict(self.fingerprint_feed(feed))
        self._atomic_write_meta()
        self._remove_orphan_files()
        return set(reusable)

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


@dataclasses.dataclass
class RecoveryPlan:
    """恢复前影响预览 (只读) 与确认接续的依据。

    - reusable:    完整发布、校验通过、前驱可信且输入指纹一致, 可直接复用;
    - recompute:   必须重算的已提交节点 -> 原因码;
    - changed_sources: 内容指纹与旧记录不同 / 缺失 / 无法核验的源块;
    - graph_match: 当前图结构身份是否与检查点一致。
    """

    reusable: Set[str]
    recompute: Dict[str, str]
    reasons: Dict[str, str]
    changed_sources: Dict[str, str]
    graph_match: bool = True
    changed_relations: Dict[str, str] = dataclasses.field(default_factory=dict)
    new_nodes: Set[str] = dataclasses.field(default_factory=set)
    removed_nodes: Set[str] = dataclasses.field(default_factory=set)
    current_nodes: Set[str] = dataclasses.field(default_factory=set)

    _REASON_TEXT = {
        "input_changed": "输入数组内容已改变",
        "input_unverified": "旧检查点没有该输入的内容指纹, 无法证明处理过当前输入",
        "source_removed": "当前运行不再提供该源块",
        "corrupt_or_missing_block": "结果块缺失或校验和失配 (损坏)",
        "graph_unverified": "旧检查点没有图身份与输入指纹",
        "graph_changed": "计算关系已改变, 同名节点也不能沿用旧值",
    }

    def describe(self) -> str:
        """生成人类可读的影响预览文本。"""
        lines = [
            "恢复影响预览:",
            f"  可复用节点 ({len(self.reusable)}): "
            + (", ".join(sorted(self.reusable)) or "无"),
            f"  必须重算节点 ({len(self.recompute)}): "
            + (", ".join(sorted(self.recompute)) or "无"),
        ]
        if self.changed_sources:
            lines.append("  变化的输入:")
            for bid, kind in sorted(self.changed_sources.items()):
                lines.append(
                    f"    - {bid}: {self._REASON_TEXT.get(kind, kind)}"
                )
        if self.changed_relations:
            lines.append("  计算关系变化:")
            for nid in sorted(self.changed_relations):
                lines.append(f"    - {nid}: {self.explain(nid)}")
        if self.new_nodes:
            lines.append("  新增节点: " + ", ".join(sorted(self.new_nodes)))
        if self.removed_nodes:
            lines.append("  删除节点: " + ", ".join(sorted(self.removed_nodes)))
        if self.recompute:
            lines.append("  重算原因:")
            for nid in sorted(self.recompute):
                lines.append(f"    - {nid}: {self.explain(nid)}")
        if not self.changed_sources and not self.recompute:
            lines.append("  输入与依赖均未变化, 全部已完成节点可原样复用。")
        return "\n".join(lines)

    def explain(self, node_id: str) -> str:
        """把原因码翻译成人话。"""
        code = self.reasons.get(node_id, "")
        if code.startswith("dependency:"):
            return f"依赖的上游节点 {code.split(':', 1)[1]} 需要重算"
        if code.startswith("graph_changed:dependency:"):
            return f"计算关系改变: 上游节点 {code.rsplit(':', 1)[1]} 在新图中不可复用"
        if code.startswith("graph_changed:producer:"):
            return f"计算关系改变: 输入块改由节点 {code.rsplit(':', 1)[1]} 生产"
        if code == "graph_changed:inputs":
            return "计算关系改变: 节点输入边集合不同"
        if code == "graph_changed:node_recipe":
            return "计算关系改变: 算子、参数或输出块声明不同"
        if code == "graph_changed:node_added":
            return "计算关系改变: 当前节点在旧图中不存在"
        for prefix, text in self._REASON_TEXT.items():
            if code.startswith(prefix + ":") or code == prefix:
                detail = code.split(":", 1)[1] if ":" in code else ""
                return f"{text} ({detail})" if detail else text
        return code or "节点尚未提交"

    @property
    def to_execute(self) -> Set[str]:
        """当前图需要本次实际执行的全部节点。"""
        return set(self.current_nodes) - self.reusable


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
