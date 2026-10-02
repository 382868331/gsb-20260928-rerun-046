"""Engine for task graphs whose shape is decided by computed results.

The full task graph cannot be listed up front: a *discovery* phase runs
step by step, and each finished step may emit the actual partitions and
the scope of follow-up work.  The engine checkpoints after every unit of
work, so a run can be interrupted (e.g. when new partitions appear) and
resumed later with identical results.

Key invariants
--------------
* Partition identity is a content hash of (compute_version, input) --
  never the partition name.  Duplicate notifications of the same
  partition are ignored: no recompute, no double aggregation.
* Aggregation is only *complete* once discovery is done and every
  discovered partition has finished; before that it is at best *partial*
  and its scope says exactly which partitions it covers.
* Revisions derive a branch by copy-on-write.  Old results are reused
  only when input *and* compute meaning (compute_version) still match;
  deleted / split / merged partitions simply get different identities.
* Final results are independent of parallelism, interruption points and
  the number of resumes.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .store import Store


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def partition_identity(compute_version: str, part_input: Any) -> str:
    """Content address of a partition: same input + same compute meaning
    => same identity.  Names are labels and never part of the identity."""
    h = hashlib.sha256()
    h.update(compute_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical(part_input).encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    input: Any


@dataclass
class Job:
    """A job binds data (params) to a registered plugin."""

    name: str
    params: dict = field(default_factory=dict)
    compute_version: str = "v1"


@dataclass
class Plugin:
    """User-supplied semantics.

    discover_step(step, params) -> (specs, done)
        Deterministic, resumable discovery: step is a monotonically
        increasing cursor persisted by the engine.  Returns the partition
        specs revealed by this step and whether discovery is finished.
    compute(part_input, params) -> result
    aggregate(entries, params) -> result
        entries: list of {"name", "input", "result"} sorted by identity.
    """

    discover_step: Callable[[int, dict], tuple]
    compute: Callable[[Any, dict], Any]
    aggregate: Callable[[list, dict], Any]


class Engine:
    def __init__(self, store: Store):
        self.store = store
        self._plugins: dict[str, Plugin] = {}

    # ---------------------------------------------------------- setup
    def register(self, job_name: str, plugin: Plugin) -> None:
        self._plugins[job_name] = plugin

    def submit(self, branch: str, job: Job) -> None:
        if self.store.exists(branch):
            raise ValueError(f"branch {branch!r} already exists")
        if job.name not in self._plugins:
            raise ValueError(f"no plugin registered for job {job.name!r}")
        state = {
            "job": {
                "name": job.name,
                "params": job.params,
                "compute_version": job.compute_version,
            },
            "discovery": {"next_step": 0, "done": False},
            "partitions": {},
            "aggregation": {"status": "pending", "scope": [], "result": None},
            "events": [],
        }
        self._event(state, "submit", [], f"job {job.name} submitted")
        self.store.save(branch, state)

    # ---------------------------------------------------------- events
    def _event(self, state: dict, kind: str, scope: list, detail: str) -> None:
        state["events"].append(
            {"seq": len(state["events"]), "kind": kind, "scope": scope,
             "detail": detail}
        )

    # ---------------------------------------------------------- discovery
    def _discover_once(self, state: dict, plugin: Plugin) -> list[str]:
        disc = state["discovery"]
        if disc["done"]:
            return []
        step = disc["next_step"]
        specs, done = plugin.discover_step(step, state["job"]["params"])
        new_ids: list[str] = []
        for spec in specs:
            ident = partition_identity(
                state["job"]["compute_version"], spec.input
            )
            if ident in state["partitions"]:
                part = state["partitions"][ident]
                if part.get("confirmed", True):
                    # Duplicate notification: never recompute, never re-aggregate.
                    self._event(state, "partition", [ident],
                                f"duplicate notification of {spec.name!r} ignored")
                else:
                    # A result kept by a revision only counts once the new
                    # discovery re-notifies the identical partition.
                    part["confirmed"] = True
                    part["name"] = spec.name
                    self._event(state, "partition", [ident],
                                f"reconfirmed kept result for {spec.name!r}")
                continue
            state["partitions"][ident] = {
                "name": spec.name,
                "input": spec.input,
                "status": "pending",
                "confirmed": True,
                "result": None,
                "error": None,
            }
            new_ids.append(ident)
        disc["next_step"] = step + 1
        disc["done"] = bool(done)
        if disc["done"]:
            # Partitions kept by a revision but never re-notified by the new
            # discovery were deleted / split / merged: drop them so they can
            # never leak into the aggregation scope.
            stale = [pid for pid, p in state["partitions"].items()
                     if not p.get("confirmed", True)]
            for pid in stale:
                name = state["partitions"][pid]["name"]
                del state["partitions"][pid]
                self._event(state, "partition", [pid],
                            f"dropped stale partition {name!r} "
                            f"(not re-notified after revision)")
        self._event(
            state, "discovery", new_ids,
            f"step {step}: {len(new_ids)} new partition(s)"
            + (", discovery done" if disc["done"] else ""),
        )
        return new_ids

    # ---------------------------------------------------------- compute
    def _compute_pending(self, state: dict, plugin: Plugin,
                         max_workers: int) -> list[str]:
        todo = [pid for pid, p in state["partitions"].items()
                if p.get("confirmed", True)
                and p["status"] in ("pending", "failed")]
        if not todo:
            return []
        params = state["job"]["params"]

        def work(pid: str):
            part_input = state["partitions"][pid]["input"]
            try:
                return pid, plugin.compute(part_input, params), None
            except Exception as exc:  # partial failure is recorded, not fatal
                return pid, None, f"{type(exc).__name__}: {exc}"

        done_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for pid, result, error in pool.map(work, sorted(todo)):
                part = state["partitions"][pid]
                if error is None:
                    part["status"] = "done"
                    part["result"] = result
                    part["error"] = None
                    done_ids.append(pid)
                    self._event(state, "partition", [pid],
                                f"computed {part['name']!r}")
                else:
                    part["status"] = "failed"
                    part["error"] = error
                    self._event(state, "partition", [pid],
                                f"FAILED {part['name']!r}: {error}")
        return done_ids

    # ---------------------------------------------------------- aggregation
    def _aggregate(self, state: dict, plugin: Plugin) -> None:
        parts = state["partitions"]
        live = {pid: p for pid, p in parts.items() if p.get("confirmed", True)}
        if any(p["status"] != "done" for p in live.values()):
            return  # blocked: unfinished or failed partitions remain
        disc_done = state["discovery"]["done"]
        scope = sorted(live)
        agg = state["aggregation"]
        if agg["status"] == "complete" and agg["scope"] == scope and disc_done:
            return  # already final for exactly this scope
        entries = [
            {"name": live[pid]["name"], "input": live[pid]["input"],
             "result": parts[pid]["result"]}
            for pid in scope
        ]
        result = plugin.aggregate(entries, state["job"]["params"])
        agg["status"] = "complete" if disc_done else "partial"
        agg["scope"] = scope
        agg["result"] = result
        self._event(
            state, "aggregation", scope,
            f"aggregation {agg['status']} over {len(scope)} partition(s)",
        )

    # ---------------------------------------------------------- run
    def run(self, branch: str, max_workers: int = 1,
            stop_on_new_partition: bool = False,
            max_steps: Optional[int] = None) -> dict:
        """Advance the graph as far as possible, checkpointing each unit.

        stop_on_new_partition simulates an operator halting the run the
        moment discovery reveals more work; resuming continues exactly
        where it stopped.  Returns the final (persisted) state.
        """
        state = self.store.load(branch)
        plugin = self._plugins[state["job"]["name"]]
        steps = 0
        while True:
            progressed = False
            if not state["discovery"]["done"]:
                new_ids = self._discover_once(state, plugin)
                self.store.save(branch, state)
                progressed = True
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    return state
                if new_ids and stop_on_new_partition:
                    self._event(state, "discovery", new_ids,
                                "stopped on new partitions")
                    self.store.save(branch, state)
                    return state
            computed = self._compute_pending(state, plugin, max_workers)
            if computed:
                self.store.save(branch, state)
                progressed = True
            before = (state["aggregation"]["status"],
                      list(state["aggregation"]["scope"]))
            self._aggregate(state, plugin)
            after = (state["aggregation"]["status"],
                     list(state["aggregation"]["scope"]))
            if before != after:
                self.store.save(branch, state)
                progressed = True
            if not progressed:
                return state

    # ---------------------------------------------------------- queries
    def progress(self, branch: str) -> dict:
        """Read-only progress query.  Never triggers discovery or compute."""
        state = self.store.load(branch)
        parts = state["partitions"]
        by_status: dict[str, list] = {}
        for pid, p in sorted(parts.items()):
            by_status.setdefault(p["status"], []).append(pid)
        return {
            "branch": branch,
            "job": dict(state["job"]),
            "discovery": dict(state["discovery"]),
            "partitions": by_status,
            "aggregation": {
                "status": state["aggregation"]["status"],
                "scope": list(state["aggregation"]["scope"]),
            },
            "provenance": {
                pid: {"name": p["name"], "input": p["input"]}
                for pid, p in sorted(parts.items())
            },
        }

    def result(self, branch: str) -> Any:
        state = self.store.load(branch)
        agg = state["aggregation"]
        if agg["status"] != "complete":
            raise RuntimeError(
                f"aggregation is {agg['status']}, not complete "
                f"(scope={agg['scope']})"
            )
        return agg["result"]

    def events(self, branch: str, kind: Optional[str] = None) -> list:
        evts = self.store.load(branch)["events"]
        return [e for e in evts if kind is None or e["kind"] == kind]

    # ---------------------------------------------------------- revisions
    def revise(self, src_branch: str, dst_branch: str,
               new_params: Optional[dict] = None,
               new_compute_version: Optional[str] = None) -> dict:
        """Derive a revised plan from an existing branch.

        The source branch is never modified (copy-on-write), so a failed
        revision leaves the original graph and sibling revisions intact.
        Old partition results are kept only when the partition's identity
        -- input plus compute meaning -- is unchanged; partitions that
        were deleted, split or merged under the revision simply do not
        match and are recomputed by the next run.
        """
        if self.store.exists(dst_branch):
            raise ValueError(f"branch {dst_branch!r} already exists")
        state = self.store.derive(src_branch, dst_branch)
        job = state["job"]
        if new_params is not None:
            job["params"] = new_params
        if new_compute_version is not None:
            job["compute_version"] = new_compute_version

        kept, dropped = {}, []
        for old_id, part in state["partitions"].items():
            new_id = partition_identity(job["compute_version"], part["input"])
            if new_id == old_id and part["status"] == "done":
                part["confirmed"] = False  # must be re-notified by new discovery
                kept[new_id] = part
            else:
                dropped.append((part["name"], old_id))
        state["partitions"] = kept
        state["discovery"] = {"next_step": 0, "done": False}
        state["aggregation"] = {"status": "pending", "scope": [], "result": None}
        self._event(
            state, "revision", sorted(kept),
            f"derived from {src_branch!r}: kept {len(kept)} result(s), "
            f"invalidated {len(dropped)}"
            + (f" ({', '.join(n for n, _ in dropped)})" if dropped else ""),
        )
        self.store.save(dst_branch, state)
        return state
