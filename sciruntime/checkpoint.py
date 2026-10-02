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
(哪些节点可复用、哪些必须重算及原因), 不改动盘上任何旧记录, 调用方可
取消; :meth:`CheckpointStore.apply_recovery` 才真正丢弃失效节点并记录新
输入指纹。

当前图与旧图身份不同 (插入校正、替换计算或删除/改接分支) 时不能在原目录
直接改名复用; 应使用 :meth:`CheckpointStore.plan_revision` 与
:meth:`CheckpointStore.materialize_revision` 派生独立候选检查点。多个候选
共享同一个只读旧检查点, 可先比较 pending 节点再执行; 候选成功或失败都不
覆盖旧图和其他候选。复用记录保留 provenance 链, 可追溯到来源图和节点。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
        # 本次检查点运行时实际喂入的源块内容指纹 block_id -> sha256
        self._inputs: Dict[str, str] = {}
        # node_id -> {"blocks": [...], "provenance": {...}}
        self._nodes: Dict[str, Dict[str, object]] = {}
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
            self._nodes = self._decode_nodes(raw.get("nodes", {}))
            return self._completed_view()
        # v1/v2: v2 有 graph_id 与 nodes 但没有输入指纹;
        # v1 是 {node_id: [block_id, ...]}。
        # 两者都没有源块内容记录, 输入身份无法核验, 按不可信处理
        # (input_unverified), 旧节点全部重算, 绝不盲目复用。
        if isinstance(raw, dict):
            self._graph_id = raw.get("graph_id")
            nodes = raw.get("nodes")
            if isinstance(nodes, dict):
                self._nodes = self._decode_nodes(nodes)
                return self._completed_view()
            return {
                nid: list(bids) for nid, bids in raw.items()
                if isinstance(nid, str) and isinstance(bids, list)
            }
        return {}

    def _decode_nodes(
        self, raw_nodes: Dict[str, object]
    ) -> Dict[str, Dict[str, object]]:
        decoded: Dict[str, Dict[str, object]] = {}
        for nid, value in raw_nodes.items():
            if isinstance(value, dict) and isinstance(value.get("blocks"), list):
                blocks = [dict(item) for item in value["blocks"]
                          if isinstance(item, dict)]
                provenance = dict(value.get("provenance") or {})
            else:
                blocks = [dict(item) for item in (value or [])
                          if isinstance(item, dict)]
                provenance = {}
            if not provenance:
                provenance = {
                    "graph_id": self._graph_id,
                    "node_id": nid,
                    "origin": "legacy",
                    "chain": [],
                }
            provenance.setdefault("chain", [])
            decoded[nid] = {"blocks": blocks, "provenance": provenance}
        return decoded

    def _completed_view(self) -> Dict[str, List[str]]:
        return {
            nid: [item["block"] for item in record["blocks"]]
            for nid, record in self._nodes.items()
        }

    @property
    def graph_id(self) -> Optional[str]:
        return self._graph_id

    def attach_graph(self, graph_id: str) -> None:
        """绑定图身份。同目录恢复只能用于结构完全相同的图。"""
        if self._graph_id is None:
            self._graph_id = graph_id
        elif self._graph_id != graph_id:
            raise CheckpointMismatch(
                f"检查点图身份 {self._graph_id[:12]} 与当前图 {graph_id[:12]} 不一致"
            )

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
            "graph_id": self._graph_id,
            "inputs": self._inputs,
            "nodes": {
                nid: {
                    "blocks": record["blocks"],
                    "provenance": record["provenance"],
                }
                for nid, record in self._nodes.items()
            },
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
        *,
        provenance: Optional[Mapping[str, object]] = None,
    ) -> None:
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
        origin = dict(provenance or {
            "graph_id": self._graph_id,
            "node_id": node_id,
            "origin": "computed",
        })
        origin.setdefault("graph_id", self._graph_id)
        origin.setdefault("node_id", node_id)
        origin.setdefault("origin", "computed")
        origin.setdefault("chain", [])
        self._nodes[node_id] = {
            "blocks": items,
            "provenance": origin,
        }
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
        record = self._nodes.get(node_id)
        items = record.get("blocks") if record else None
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

    def provenance(self, node_id: str) -> Dict[str, object]:
        record = self._nodes.get(node_id)
        return dict(record["provenance"]) if record else {}

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
        if self._graph_id is not None and self._graph_id != graph.identity():
            raise CheckpointMismatch(
                f"检查点图身份 {self._graph_id[:12]} 与当前图 "
                f"{graph.identity()[:12]} 不一致; 请使用 plan_revision 派生新图"
            )
        return self._build_recovery_plan(graph, graph, feed, allow_revision=False)

    def plan_revision(
        self,
        new_graph,
        old_graph,
        feed: Mapping[str, np.ndarray],
    ) -> "RecoveryPlan":
        """只读规划从旧检查点派生一张新图, 不改动当前检查点。"""
        if self._graph_id is not None and self._graph_id != old_graph.identity():
            raise CheckpointMismatch(
                f"检查点图身份 {self._graph_id[:12]} 与旧图 "
                f"{old_graph.identity()[:12]} 不一致"
            )
        return self._build_recovery_plan(
            new_graph, old_graph, feed, allow_revision=True
        )

    def _build_recovery_plan(
        self,
        graph,
        old_graph,
        feed: Mapping[str, np.ndarray],
        *,
        allow_revision: bool,
    ) -> "RecoveryPlan":
        topo_order = graph.topological_order()
        current = self.fingerprint_feed(feed)
        recorded = self._inputs

        reusable: Set[str] = set()
        recompute: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
        changed_sources: Dict[str, str] = {}
        missing_sources: Set[str] = set()

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
            node = graph.nodes[nid]
            relation_reason = ""
            if not allow_revision and self._graph_id != graph.identity():
                relation_reason = "graph_identity_changed"
            elif allow_revision and nid not in old_graph.nodes:
                relation_reason = "node_added"
            elif allow_revision and nid in old_graph.nodes:
                equal, why = _equivalent_relation(old_graph, graph, nid, nid)
                if not equal:
                    relation_reason = why

            bad_dep = next(
                (d for d in sorted(graph.dependencies(nid)) if d not in reusable),
                None,
            )
            sources = graph.transitive_sources(nid)
            if nid not in self._completed:
                reason = relation_reason or "not_submitted"
            elif relation_reason:
                reason = relation_reason
            elif self._graph_id is None:
                reason = "graph_unverified"
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

        added_nodes = {
            nid for nid in graph.nodes
            if allow_revision and nid not in old_graph.nodes
        }
        removed_nodes = {
            nid for nid in old_graph.nodes
            if allow_revision and nid not in graph.nodes
        }
        changed_relations = {
            nid: reason
            for nid, reason in recompute.items()
            if allow_revision and (
                reason == "node_added"
                or reason.startswith("input_edge_changed:")
                or (
                    nid in old_graph.nodes
                    and reason in {
                        "operator_changed", "params_changed",
                        "input_arity_changed", "output_arity_changed",
                    }
                )
            )
        }
        return RecoveryPlan(
            reusable=reusable,
            recompute=recompute,
            reasons=reasons,
            changed_sources=changed_sources,
            graph_match=old_graph.identity() == graph.identity(),
            pending=set(topo_order) - reusable,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            changed_relations=changed_relations,
        )

    def materialize_revision(
        self,
        destination_path: str,
        new_graph,
        old_graph,
        plan: "RecoveryPlan",
        feed: Mapping[str, np.ndarray],
    ) -> "CheckpointStore":
        """把新图可复用结果复制到独立检查点; 不修改旧检查点。"""
        if self._graph_id is not None and self._graph_id != old_graph.identity():
            raise CheckpointMismatch("派生检查点的旧图与源检查点不一致")
        expected = self.plan_revision(new_graph, old_graph, feed)
        if expected.reusable != plan.reusable or expected.pending != plan.pending:
            raise CheckpointMismatch("恢复计划已过期, 请重新预览后再派生")
        dest = CheckpointStore(destination_path)
        if dest._nodes or os.path.exists(dest._meta_path):
            raise CheckpointMismatch(f"候选检查点已存在: {destination_path}")
        dest.attach_graph(new_graph.identity())
        dest._inputs = dict(self.fingerprint_feed(feed))
        for nid in new_graph.topological_order():
            if nid not in plan.reusable:
                continue
            old_record = self._nodes[nid]
            old_items = old_record["blocks"]
            old_node = old_graph.nodes[nid]
            new_node = new_graph.nodes[nid]
            if len(old_items) != len(new_node.outputs):
                raise CheckpointMismatch(f"节点 {nid} 输出数量不一致, 不能复用")
            new_items = []
            for item, old_bid, new_bid in zip(
                old_items, old_node.outputs, new_node.outputs
            ):
                source_path = self._block_path(old_bid)
                target_path = dest._block_path(new_bid)
                if _sha256_file(source_path) != item.get("sha256"):
                    raise CheckpointMismatch(f"复用块 {old_bid} 校验失败")
                shutil.copy2(source_path, target_path)
                new_items.append({
                    "block": new_bid,
                    "sha256": item["sha256"],
                })
            dest._nodes[nid] = {
                "blocks": new_items,
                "provenance": {
                    "graph_id": old_graph.identity(),
                    "node_id": nid,
                    "source_path": os.path.abspath(self.path),
                    "origin": "reused_from_checkpoint",
                    "chain": [
                        old_record["provenance"],
                    ],
                },
            }
            dest._completed[nid] = list(new_node.outputs)
        dest._atomic_write_meta()
        dest._remove_orphan_files()
        return dest

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
        self.attach_graph(graph.identity())
        # 闭包失效: 计划外的任何已提交节点 (例如本次预览之后又被判定失效
        # 的传递后继) 一并丢弃。
        reusable = set(plan.reusable)
        bad = [nid for nid in list(self._nodes) if nid not in reusable]
        for nid in bad:
            self._discard_node(nid)
        self._inputs = dict(self.fingerprint_feed(feed))
        self._atomic_write_meta()
        self._remove_orphan_files()
        return set(reusable)

    def _discard_node(self, node_id: str) -> None:
        record = self._nodes.pop(node_id, None)
        self._completed.pop(node_id, None)
        items = record.get("blocks") if record else None
        for item in items or ():
            try:
                os.unlink(self._block_path(item["block"]))
            except OSError:
                pass

    def _remove_orphan_files(self) -> None:
        referenced = {
            f"{_safe(item['block'])}.npy"
            for record in self._nodes.values()
            for item in record["blocks"]
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
    - pending: 当前图中不能复用、需要执行的全部节点 (含新增节点);
    - graph_match: 当前图结构身份是否与检查点一致。
    """

    reusable: Set[str]
    recompute: Dict[str, str]
    reasons: Dict[str, str]
    changed_sources: Dict[str, str]
    graph_match: bool = True
    pending: Set[str] = dataclasses.field(default_factory=set)
    added_nodes: Set[str] = dataclasses.field(default_factory=set)
    removed_nodes: Set[str] = dataclasses.field(default_factory=set)
    changed_relations: Dict[str, str] = dataclasses.field(
        default_factory=dict
    )

    _REASON_TEXT = {
        "input_changed": "输入数组内容已改变",
        "input_unverified": "旧检查点没有该输入的内容指纹, 无法证明处理过当前输入",
        "source_removed": "当前运行不再提供该源块",
        "corrupt_or_missing_block": "结果块缺失或校验和失配 (损坏)",
        "graph_unverified": "旧检查点没有图身份与输入指纹",
        "operator_changed": "计算算子已替换",
        "params_changed": "计算参数已改变",
        "input_arity_changed": "输入数量已改变",
        "output_arity_changed": "输出数量已改变",
        "input_edge_changed": "计算关系已改接到其他来源",
        "node_added": "当前图新增节点",
        "not_submitted": "节点尚未提交",
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
            lines.append("  输入变化:")
            for bid, kind in sorted(self.changed_sources.items()):
                lines.append(
                    f"    - {bid}: {self._REASON_TEXT.get(kind, kind)}"
                )
        if self.changed_relations:
            lines.append("  计算关系变化:")
            for nid in sorted(self.changed_relations):
                lines.append(f"    - {nid}: {self.explain(nid)}")
        if self.added_nodes:
            lines.append("  新增节点: " + ", ".join(sorted(self.added_nodes)))
        if self.removed_nodes:
            lines.append("  删除节点: " + ", ".join(sorted(self.removed_nodes)))
        if self.pending:
            lines.append(
                "  需要执行节点 (" + str(len(self.pending)) + "): "
                + ", ".join(sorted(self.pending))
            )
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
        for prefix, text in self._REASON_TEXT.items():
            if code.startswith(prefix + ":") or code == prefix:
                detail = code.split(":", 1)[1] if ":" in code else ""
                return f"{text} ({detail})" if detail else text
        return code or "节点尚未提交"


def _same_params(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _input_origin(
    graph, node_id: str, index: int, memo: Dict[Tuple[str, int], object]
) -> object:
    """返回某个输入位置的直接数据来源; 上游变化由依赖链另行传播。"""
    key = (node_id, index)
    if key in memo:
        return memo[key]
    node = graph.nodes[node_id]
    block_id = node.inputs[index]
    producer = graph.producer_of(block_id)
    if producer is None:
        origin = ("source", block_id)
    else:
        try:
            output_index = graph.nodes[producer].outputs.index(block_id)
        except ValueError:
            output_index = -1
        origin = ("node", producer, output_index)
    memo[key] = origin
    return origin


def _equivalent_relation(
    old_graph, new_graph, old_id: str, new_id: str
) -> Tuple[bool, str]:
    old_node = old_graph.nodes[old_id]
    new_node = new_graph.nodes[new_id]
    if old_node.operator != new_node.operator:
        return False, "operator_changed"
    if not _same_params(old_node.params, new_node.params):
        return False, "params_changed"
    if len(old_node.inputs) != len(new_node.inputs):
        return False, "input_arity_changed"
    if len(old_node.outputs) != len(new_node.outputs):
        return False, "output_arity_changed"
    old_memo: Dict[Tuple[str, int], object] = {}
    new_memo: Dict[Tuple[str, int], object] = {}
    for index in range(len(old_node.inputs)):
        old_origin = _input_origin(old_graph, old_id, index, old_memo)
        new_origin = _input_origin(new_graph, new_id, index, new_memo)
        if old_origin != new_origin:
            return False, f"input_edge_changed:{index}"
    return True, ""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
