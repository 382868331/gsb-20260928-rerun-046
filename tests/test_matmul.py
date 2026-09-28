"""分块矩阵乘法：与串行 numpy 一致性（线程池/子进程池、断点恢复）。"""

import json
import subprocess
import sys

import numpy as np
import pytest

from sciruntime import Engine, EngineConfig
from sciruntime.matmul_graph import assemble, build_matmul_graph


@pytest.mark.parametrize("executor", ["thread", "process"])
@pytest.mark.parametrize("block_shape", [(8, 8, 8), (16, 12, 8)])
def test_matches_serial_numpy(tmp_path, executor, block_shape):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((32, 24), dtype=np.float64)
    b = rng.standard_normal((24, 16), dtype=np.float64)
    graph, tiles = build_matmul_graph(a, b, block_shape)
    with Engine(
        EngineConfig(
            workdir=tmp_path / "run",
            max_workers=3,
            memory_budget=1024 * 1024,
            executor=executor,
        )
    ) as engine:
        result = engine.run(graph)
    assert result.ok
    got = assemble(result.outputs, tiles, block_shape, (32, 16))
    assert np.allclose(got, a @ b)


def _run_partial_then_crash(workdir, gate_path, data_path):
    """外部进程：部分源块完成后，在部分积开始前硬退出（模拟崩溃）。"""

    import os
    import time

    from pathlib import Path
    from sciruntime.operators import registry

    arrays = np.load(data_path)
    a, b = arrays["a"], arrays["b"]
    graph, _ = build_matmul_graph(a, b, (8, 8, 8))

    class CrashingPartial:
        def run(self, inputs, params, store):
            Path(gate_path).write_text("crashing")
            time.sleep(0.2)
            os._exit(9)

    registry._ops["matmul_partial"] = CrashingPartial()
    engine = Engine(EngineConfig(workdir=workdir, max_workers=2, memory_budget=10**9))
    engine.run(graph)


def test_resume_after_real_crash(tmp_path):
    """硬崩溃后新进程复用已完成源块，重跑部分积，结果与串行一致。"""

    from pathlib import Path

    rng = np.random.default_rng(7)
    a = rng.standard_normal((16, 16))
    b = rng.standard_normal((16, 16))
    graph, tiles = build_matmul_graph(a, b, (8, 8, 8))
    workdir = tmp_path / "run"
    gate = tmp_path / "gate"
    data_path = tmp_path / "matrices.npz"
    np.savez(data_path, a=a, b=b)

    code = (
    "import sys; sys.path.insert(0, %r);"
    "from tests.test_matmul import _run_partial_then_crash;"
    "_run_partial_then_crash(%r, %r, %r)"
    ) % (
        str(Path(__file__).resolve().parent.parent),
        str(workdir),
        str(gate),
        str(data_path),
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent
    )
    assert proc.returncode == 9
    # 确有源块在崩溃前已落盘并记录成功。
    succeeded = [
        json.loads(line)["node"]
        for line in (workdir / "state.log").read_text().splitlines()
        if json.loads(line).get("type") == "succeeded"
    ]
    assert any(nid.startswith("A(") for nid in succeeded)

    # 新进程恢复：已完成节点不重跑，只补跑未完成部分。
    with Engine(EngineConfig(workdir=workdir, max_workers=2, memory_budget=10**9)) as engine:
        result = engine.run(graph)
    assert result.ok, result.failed
    got = assemble(result.outputs, tiles, (8, 8, 8), (16, 16))
    assert np.allclose(got, a @ b)


def test_resume_is_noop_when_complete(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16))
    b = rng.standard_normal((16, 16))
    graph, tiles = build_matmul_graph(a, b, (8, 8, 8))
    workdir = tmp_path / "run"
    with Engine(EngineConfig(workdir=workdir, max_workers=2)) as engine:
        engine.run(graph)
    with Engine(EngineConfig(workdir=workdir, max_workers=2)) as engine:
        result = engine.run(graph)
    got = assemble(result.outputs, tiles, (8, 8, 8), (16, 16))
    assert np.allclose(got, a @ b)
    assert (workdir / "blocks" / "C(0,0).npy").exists()
    assert not (workdir / "blocks" / "P(0,0,0).npy").exists()
