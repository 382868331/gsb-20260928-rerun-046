"""用于验证细粒度崩溃恢复的确定性计算图。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .checkpoint import CheckpointStore
from .executor import Executor
from .graph import Graph
from .operators import Operator, TaskContext, register_operator


class TraceableAdd(Operator):
    def run(
        self,
        inputs: Sequence[np.ndarray],
        *,
        params: Mapping[str, Any],
        ctx: TaskContext,
    ) -> Tuple[np.ndarray, ...]:
        trace_path = os.environ.get("SCIRUNTIME_RECOVERY_TRACE")
        if trace_path:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(ctx.node_id + "\n")
        delay = float(params.get("delay", 0.0))
        stage = os.environ.get("SCIRUNTIME_RECOVERY_STAGE")
        job_dir = os.environ.get("SCIRUNTIME_RECOVERY_JOB_DIR")
        gate = str(Path(job_dir) / f"stop-{stage}-ack") if stage and job_dir else None
        wait_for_gate = bool(
            delay or (stage == "prefix" and ctx.node_id.startswith("slow_"))
        )
        if wait_for_gate:
            stage = os.environ.get("SCIRUNTIME_RECOVERY_STAGE")
            deadline = time.monotonic() + delay
            while delay > 0 and time.monotonic() < deadline:
                if gate and Path(gate).exists():
                    break
                time.sleep(0.01)
            if delay == 0:
                while not (gate and Path(gate).exists()):
                    time.sleep(0.01)
        if len(inputs) == 1:
            value = np.asarray(inputs[0]) + float(params.get("bias", 0.0))
        else:
            value = np.asarray(inputs[0]).copy()
            for other in inputs[1:]:
                value = value + np.asarray(other)
            value = value + float(params.get("bias", 0.0))
        return (value.astype(np.float64, copy=False),)


_REGISTERED = False


def register_recovery_operators() -> None:
    global _REGISTERED
    if not _REGISTERED and "traceable_add" not in _registry():
        register_operator("traceable_add", TraceableAdd())
        _REGISTERED = True


def _registry():
    from .operators import operator_registry
    return operator_registry


def build_recovery_graph(size: int = 64) -> Tuple[Graph, str]:
    register_recovery_operators()
    g = Graph()
    item_bytes = np.dtype(np.float64).itemsize
    block_bytes = size * item_bytes

    g.add_node(
        "prepare_a",
        "traceable_add",
        inputs=("source",),
        outputs=("prepared_a",),
        params={"bias": 1},
        memory_bytes=block_bytes,
    )
    g.add_node(
        "shared_prefix",
        "traceable_add",
        inputs=("prepared_a",),
        outputs=("prefix",),
        params={"bias": 2},
        memory_bytes=block_bytes,
    )
    g.add_node(
        "slow_left",
        "traceable_add",
        inputs=("prefix",),
        outputs=("left",),
        params={"bias": 4, "delay": 2.0},
        memory_bytes=block_bytes,
    )
    g.add_node(
        "slow_right",
        "traceable_add",
        inputs=("prefix",),
        outputs=("right",),
        params={"bias": 8, "delay": 2.0},
        memory_bytes=block_bytes,
    )
    g.add_node(
        "join",
        "traceable_add",
        inputs=("left", "right"),
        outputs=("answer",),
        params={"bias": 16},
        memory_bytes=block_bytes,
    )
    g.freeze()
    return g, "answer"


def execute_stage(job_dir: str, stage: str, workers: int, budget_blocks: int) -> dict:
    register_recovery_operators()
    path = Path(job_dir)
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / f"calls-{stage}.log"
    os.environ["SCIRUNTIME_RECOVERY_TRACE"] = str(trace_path)
    os.environ["SCIRUNTIME_RECOVERY_STAGE"] = stage
    os.environ["SCIRUNTIME_RECOVERY_JOB_DIR"] = str(path)

    graph, output = build_recovery_graph()
    checkpoint = CheckpointStore(str(path / "checkpoint"), graph_id=graph.graph_id())
    stop_path = path / f"stop-{stage}"
    stop_watcher = None
    stop_event = None
    if stage in {"prefix", "branch"}:
        from threading import Event, Thread

        stop_event = Event()

        def watch() -> None:
            while not stop_event.is_set():
                if stop_path.exists():
                    stop_event.set()
                    stop_path.with_name(stop_path.name + "-ack").write_text("ack")
                    break
                time.sleep(0.01)

        stop_watcher = Thread(target=watch, daemon=True)
        stop_watcher.start()

    block_bytes = 64 * np.dtype(np.float64).itemsize
    source = np.arange(64, dtype=np.float64)
    ex = Executor(
        graph,
        max_workers=workers,
        memory_budget=block_bytes * budget_blocks,
        checkpoint=checkpoint,
        stop_event=stop_event,
    )
    stats = ex.run(feed={"source": source})
    result = ex.store.get(output) if not stats.stopped and ex.store.contains(output) else None
    summary = {
        "stage": stage,
        "workers": workers,
        "succeeded": stats.succeeded,
        "executed": stats.executed,
        "from_checkpoint": stats.from_checkpoint,
        "stopped": stats.stopped,
        "failed": stats.failed,
        "skipped": stats.skipped,
        "peak_resident_bytes": stats.peak_resident_bytes,
        "peak_inflight_bytes": stats.peak_inflight_bytes,
        "result": None if result is None else np.asarray(result).tolist(),
    }
    with open(path / f"summary-{stage}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)
    return summary


def main() -> None:
    import sys

    job_dir, stage, workers, budget_blocks = sys.argv[1:5]
    execute_stage(job_dir, stage, int(workers), int(budget_blocks))


if __name__ == "__main__":
    main()
