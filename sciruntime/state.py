"""持久化执行状态：追加式 WAL + 内存索引，支撑断点恢复。

状态转移：PENDING -> RUNNING -> SUCCEEDED / FAILED / SKIPPED。
发布结果（publish）与状态转移写在同一条日志里并 fsync，保证
“保存完成节点后恢复执行”时不会丢结果，也不会重复发布。
"""

from __future__ import annotations

import enum
import json
import os
import threading
from pathlib import Path


class NodeStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL = {NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.SKIPPED}


class RunState:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._status: dict[str, NodeStatus] = {}
        self._meta: dict[str, dict] = {}
        self._error: dict[str, str] = {}
        self._fh = open(self.path, "a+", encoding="utf-8")
        self._fh.seek(0)
        for line in self._fh:
            line = line.strip()
            if line:
                self._apply(json.loads(line))

    # ---- 查询 ----

    def status(self, node_id: str) -> NodeStatus:
        with self._lock:
            return self._status.get(node_id, NodeStatus.PENDING)

    def is_terminal(self, node_id: str) -> bool:
        return self.status(node_id) in TERMINAL

    def succeeded(self) -> set[str]:
        with self._lock:
            return {nid for nid, st in self._status.items() if st == NodeStatus.SUCCEEDED}

    def failed(self) -> dict[str, str]:
        with self._lock:
            return dict(self._error)

    def result_meta(self, node_id: str) -> dict | None:
        with self._lock:
            return self._meta.get(node_id)

    # ---- 原子条件状态转移 ----

    def mark_running(self, node_id: str, attempt: int) -> bool:
        """RUNNING 仅允许从 PENDING 转入；恢复时旧 RUNNING 视为 PENDING。"""

        return self._transition(
            {"type": "running", "node": node_id, "attempt": attempt},
            node_id,
            NodeStatus.RUNNING,
            allow={NodeStatus.PENDING},
        )

    def mark_succeeded(self, node_id: str, meta: dict) -> bool:
        """SUCCEEDED 是一次性发布：已成功的节点重复完成不会再写日志。"""

        return self._transition(
            {"type": "succeeded", "node": node_id, "meta": meta},
            node_id,
            NodeStatus.SUCCEEDED,
            allow={NodeStatus.RUNNING, NodeStatus.PENDING},
            meta=meta,
            idempotent_already=NodeStatus.SUCCEEDED,
        )

    def mark_failed(self, node_id: str, message: str) -> bool:
        return self._transition(
            {"type": "failed", "node": node_id, "error": message},
            node_id,
            NodeStatus.FAILED,
            allow={NodeStatus.RUNNING, NodeStatus.PENDING},
            error=message,
            idempotent_already=NodeStatus.FAILED,
        )

    def mark_skipped(self, node_id: str, reason: str) -> bool:
        return self._transition(
            {"type": "skipped", "node": node_id, "reason": reason},
            node_id,
            NodeStatus.SKIPPED,
            allow={NodeStatus.PENDING},
            error=reason,
            idempotent_already=NodeStatus.SKIPPED,
        )

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    # ---- 内部 ----

    def _transition(
        self,
        record: dict,
        node_id: str,
        new_status: NodeStatus,
        allow: set[NodeStatus],
        meta: dict | None = None,
        error: str | None = None,
        idempotent_already: NodeStatus | None = None,
    ) -> bool:
        with self._lock:
            current = self._status.get(node_id, NodeStatus.PENDING)
            if idempotent_already is not None and current == idempotent_already:
                return False
            if current not in allow:
                return False
            self._write_locked(record)
            self._status[node_id] = new_status
            if meta is not None:
                self._meta[node_id] = meta
            if error is not None:
                self._error[node_id] = error
            return True

    def _write_locked(self, record: dict) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def _apply(self, record: dict) -> None:
        kind = record["type"]
        node_id = record["node"]
        if kind == "running":
            # 上次进程崩溃在运行中：恢复时回到 PENDING 重新执行。
            self._status[node_id] = NodeStatus.PENDING
        elif kind == "succeeded":
            self._status[node_id] = NodeStatus.SUCCEEDED
            self._meta[node_id] = record.get("meta", {})
        elif kind == "failed":
            self._status[node_id] = NodeStatus.FAILED
            self._error[node_id] = record.get("error", "")
        elif kind == "skipped":
            self._status[node_id] = NodeStatus.SKIPPED
            self._error[node_id] = record.get("reason", "")
