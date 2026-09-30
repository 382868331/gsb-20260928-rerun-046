"""测试驱动: 在子进程中对分块矩阵乘法图做带检查点的一次运行。

通过 JSON 参数 (argv[1]) 选择 A/B 输入种子、worker 数、内存预算以及
硬中断钩子, 用于验证“改变实际输入 + 不同 worker 数 + 块损坏 +
中途退出”同时出现时仍能正确接续。
"""
from __future__ import annotations

import json
import sys

import numpy as np

from sciruntime import CheckpointStore, Executor
from sciruntime.matrix import (
    assemble_result,
    build_blocked_matmul_graph,
    serial_reference_matmul,
)


def _matrix(seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((48, 32)),
            np.random.default_rng(seed + 1000).standard_normal((32, 40)))


def main():
    cfg = json.loads(sys.argv[1])
    a, b = _matrix(cfg["seed"])
    graph, outputs = build_blocked_matmul_graph(
        a.shape, b.shape, row_block=16, k_block=32, col_block=20
    )
    ck = CheckpointStore(cfg["ckpt"])
    ex = Executor(
        graph,
        max_workers=cfg["workers"],
        memory_budget=cfg.get("budget"),
        backend="thread",
        checkpoint=ck,
        crash_after_block=cfg.get("crash_after_block"),
        exit_after_commits=cfg.get("exit_after_commits"),
    )
    stats = ex.run(feed={"__src_A": a, "__src_B": b})
    result = assemble_result(ex.store, outputs, a.shape[0], n_cols_blocks=2)
    print("WORKER_JSON " + json.dumps({
        "ok": stats.ok,
        "from_checkpoint": stats.from_checkpoint,
        "recomputed": stats.recomputed,
        "result_equal": bool(
            np.array_equal(result, serial_reference_matmul(a, b, k_block=32))
        ),
    }))


if __name__ == "__main__":
    main()
