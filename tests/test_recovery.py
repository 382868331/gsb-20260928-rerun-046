from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sciruntime.checkpoint import CheckpointStore, CheckpointError, fingerprint_array
from sciruntime.recovery_graph import build_recovery_graph


def _wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _run_stage(job_dir: Path, stage: str, workers: int, budget_blocks: int) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sciruntime.recovery_graph",
            str(job_dir),
            stage,
            str(workers),
            str(budget_blocks),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return json.loads((job_dir / f"summary-{stage}.json").read_text())


def _start_stage(job_dir: Path, stage: str, workers: int, budget_blocks: int):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sciruntime.recovery_graph",
            str(job_dir),
            stage,
            str(workers),
            str(budget_blocks),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _calls(job_dir: Path, stage: str) -> list[str]:
    path = job_dir / f"calls-{stage}.log"
    return path.read_text().splitlines() if path.exists() else []


def test_real_interrupted_resume_reuses_only_verified_nodes(tmp_path):
    graph, output = build_recovery_graph()
    ckpt_path = tmp_path / "checkpoint"

    proc = _start_stage(tmp_path, "prefix", 1, 2)
    _wait_for(tmp_path / "calls-prefix.log")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if "slow_left" in _calls(tmp_path, "prefix"):
            break
        time.sleep(0.01)
    (tmp_path / "stop-prefix").write_text("stop")
    proc.wait(timeout=5)
    assert proc.returncode == 0
    first = json.loads((tmp_path / "summary-prefix.json").read_text())

    assert set(first["executed"]) == {"prepare_a", "shared_prefix", "slow_left"}
    assert first["succeeded"] == ["prepare_a", "shared_prefix"]
    assert first["stopped"]
    first_ck = CheckpointStore(str(ckpt_path), graph_id=graph.graph_id())
    assert set(first_ck.completed_nodes()) == {"prepare_a", "shared_prefix"}
    assert not list((ckpt_path / "blocks").glob("*.tmp"))
    assert not list(ckpt_path.glob("meta.*.tmp"))

    proc = _start_stage(tmp_path, "branch", 4, 4)
    _wait_for(tmp_path / "calls-branch.log")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        calls = set(_calls(tmp_path, "branch"))
        if {"slow_left", "slow_right"}.issubset(calls):
            break
        time.sleep(0.01)
    (tmp_path / "stop-branch").write_text("stop")
    proc.wait(timeout=5)
    assert proc.returncode == 0
    second = json.loads((tmp_path / "summary-branch.json").read_text())

    assert second["from_checkpoint"] == ["prepare_a", "shared_prefix"]
    assert set(second["executed"]).issubset({"slow_left", "slow_right"})
    assert "slow_left" in _calls(tmp_path, "branch")
    assert "slow_right" in _calls(tmp_path, "branch")
    assert second["stopped"]
    second_ck = CheckpointStore(str(ckpt_path), graph_id=graph.graph_id())
    assert set(second_ck.completed_nodes()) == {"prepare_a", "shared_prefix"}
    assert not list((ckpt_path / "blocks").glob("*.tmp"))
    assert second["peak_inflight_bytes"] == 2 * 64 * 8

    third = _run_stage(tmp_path, "final", 3, 4)
    final_ck = CheckpointStore(str(ckpt_path), graph_id=graph.graph_id())
    assert set(final_ck.completed_nodes()) == set(graph.nodes)
    assert third["from_checkpoint"] == ["prepare_a", "shared_prefix"]
    assert set(third["executed"]) == {"slow_left", "slow_right", "join"}
    assert "join" in _calls(tmp_path, "final")
    assert third["peak_inflight_bytes"] == 2 * 64 * 8

    expected = np.arange(64, dtype=np.float64)
    expected = ((expected + 3) + 4) + ((expected + 3) + 8) + 16
    assert np.array_equal(np.asarray(third["result"]), expected)


def test_checkpoint_rejects_wrong_identity_and_corrupt_block(tmp_path):
    graph, output = build_recovery_graph()
    ckpt = CheckpointStore(str(tmp_path), graph_id=graph.graph_id())
    value = np.arange(4, dtype=np.float64)
    ckpt.bind_run(graph.graph_id(), {"source": fingerprint_array(np.arange(64, dtype=np.float64))})
    ckpt.commit_node("prepare_a", ["prepared_a"], [value])
    assert np.array_equal(ckpt.load_block("prepared_a"), value)

    block_path = tmp_path / "blocks" / "prepared_a.npy"
    corrupted = value.copy()
    corrupted[0] += 1
    with open(block_path, "wb") as f:
        np.lib.format.write_array(f, corrupted, allow_pickle=False)
    with pytest.raises(CheckpointError, match="校验失败"):
        CheckpointStore(str(tmp_path), graph_id=graph.graph_id())

    wrong_graph, _ = build_recovery_graph()
    wrong_graph.nodes["prepare_a"].params["bias"] = 99
    with pytest.raises(CheckpointError, match="另一个计算图"):
        CheckpointStore(str(tmp_path), graph_id=wrong_graph.graph_id())
