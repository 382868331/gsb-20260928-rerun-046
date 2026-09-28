"""首轮示例: 带依赖的分块矩阵乘法。

串行参考: numpy A @ B。
并行运行: sciruntime 执行分块计算图, 并拼接结果对比。
同时演示: 内存预算下的排队、检查点保存与恢复。
运行: .venv/bin/python -m examples.blocked_matmul_demo
"""
from __future__ import annotations

import shutil
import tempfile

import numpy as np

from sciruntime import CheckpointStore, Executor
from sciruntime.matrix import (
    assemble_result,
    build_blocked_matmul_graph,
    serial_reference_matmul,
)


def run_graph(A, B, *, backend="thread", memory_budget=None, checkpoint=None):
    m, k = A.shape
    k2, n = B.shape
    graph, out_blocks = build_blocked_matmul_graph(
        (m, k), (k2, n), row_block=32, k_block=24, col_block=40
    )
    ex = Executor(
        graph,
        max_workers=4,
        memory_budget=memory_budget,
        backend=backend,
        checkpoint=checkpoint,
    )
    stats = ex.run(feed={"__src_A": A, "__src_B": B})
    result = assemble_result(ex.store, out_blocks, m,
                             n_cols_blocks=(n + 39) // 40)
    return result, stats


def main() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((96, 72))
    B = rng.standard_normal((72, 120))
    expected = serial_reference_matmul(A, B, k_block=24)

    print("== 1) 线程后端 + 内存预算 (强制排队) ==")
    tight_budget = 2 * (A.nbytes + B.nbytes) + 6 * 32 * 40 * A.itemsize
    result, stats = run_graph(A, B, backend="thread", memory_budget=tight_budget)
    print("成功节点:", len(stats.succeeded), "最大误差:", np.abs(result - expected).max())
    assert np.array_equal(result, expected)

    print("== 2) 子进程后端 ==")
    result, stats = run_graph(A, B, backend="process")
    print("成功节点:", len(stats.succeeded), "最大误差:", np.abs(result - expected).max())
    assert np.array_equal(result, expected)

    print("== 3) 检查点: 保存后用新 Executor 恢复 ==")
    ckpt_dir = tempfile.mkdtemp(prefix="sci-ckpt-")
    try:
        ck1 = CheckpointStore(ckpt_dir)
        _, s1 = run_graph(A, B, checkpoint=ck1)
        ck2 = CheckpointStore(ckpt_dir)
        result2, s2 = run_graph(A, B, checkpoint=ck2)
        print("首轮:", len(s1.succeeded), "恢复命中:", len(s2.from_checkpoint))
        assert len(s2.from_checkpoint) == len(s1.succeeded)
        assert np.array_equal(result2, expected)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print("全部通过: 并行结果与串行 A @ B 完全一致。")


if __name__ == "__main__":
    main()
