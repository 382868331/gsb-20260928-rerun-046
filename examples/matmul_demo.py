"""首轮演示：带依赖、扇入复用的分块矩阵乘法。

用法::

    python -m examples.matmul_demo                # 线程池
    python -m examples.matmul_demo --process      # 子进程池
    python -m examples.matmul_demo --resume       # 用同一目录演示断点恢复
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from sciruntime import Engine, EngineConfig
from sciruntime.matmul_graph import assemble, build_matmul_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", action="store_true", help="使用子进程池")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budget-mb", type=int, default=64)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--resume", action="store_true", help="复用已有工作目录")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    a = rng.standard_normal((48, 32), dtype=np.float64)
    b = rng.standard_normal((32, 48), dtype=np.float64)
    block_shape = (16, 16, 16)  # 3x2x3 分块

    graph, tiles = build_matmul_graph(a, b, block_shape)
    workdir = args.workdir or (
        "./scirun-demo" if args.resume else str(Path(tempfile.mkdtemp(prefix="scirun-")))
    )
    config = EngineConfig(
        workdir=workdir,
        max_workers=args.workers,
        memory_budget=args.budget_mb * 1024 * 1024,
        executor="process" if args.process else "thread",
    )
    print(f"workdir={workdir} executor={config.executor} nodes={len(graph.nodes)}")
    with Engine(config) as engine:
        result = engine.run(graph)
        if args.resume:
            result = engine.run(graph)  # 第二次：全部节点从 WAL 恢复
    assert result.ok, f"execution failed: {result.failed} {result.skipped}"
    got = assemble(result.outputs, tiles, block_shape, a.shape[:1] + b.shape[1:])
    expected = a @ b
    assert np.allclose(got, expected), "结果与串行 numpy 不一致"
    print(f"OK: max abs diff vs numpy = {np.abs(got - expected).max():.3e}")


if __name__ == "__main__":
    main()
