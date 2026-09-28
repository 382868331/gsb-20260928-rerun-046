"""分块矩阵乘法计算图构造。

C = A @ B，A 形状 (mb*bm, kb*bk)，B 形状 (kb*bk, nb*bn)：

- (mb*kb + kb*nb) 个源块节点（原始矩阵切块）；
- mb*nb*kb 个 ``matmul_partial`` 部分积节点（展示扇入复用：
  A_ik 被所有 j 复用，B_kj 被所有 i 复用）；
- mb*nb 个 ``tile_sum`` 聚合节点，标记 keep=True 作为最终结果块。
"""

from __future__ import annotations

import numpy as np

from .blocks import Block
from .graph import Graph, Node
from .sciops import register_all


def build_matmul_graph(
    a: np.ndarray,
    b: np.ndarray,
    block_shape: tuple[int, int, int],
) -> tuple[Graph, dict[str, tuple[int, int]]]:
    register_all()
    bm, bk, bn = block_shape
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    mb, kb, nb = m // bm, k // bk, n // bn
    assert (m, k, n) == (mb * bm, kb * bk, nb * bn), "维度需被块大小整除"
    dt = np.dtype(a.dtype)
    item = dt.itemsize

    graph = Graph()
    tiles: dict[str, tuple[int, int]] = {}

    def src(prefix: str, matrix: np.ndarray, rows: int, cols: int, bs0: int, bs1: int) -> None:
        for i in range(rows):
            for j in range(cols):
                nid = f"{prefix}({i},{j})"
                graph.add(
                    Node(
                        id=nid,
                        op="matrix_source",
                        params={"matrix": matrix, "tile": (i, j), "block_shape": (bs0, bs1)},
                        estimated_bytes=bs0 * bs1 * item,
                    )
                )

    src("A", a, mb, kb, bm, bk)
    src("B", b, kb, nb, bk, bn)

    partials: dict[tuple[int, int], list[str]] = {}
    for i in range(mb):
        for j in range(nb):
            for kk in range(kb):
                pid = f"P({i},{j},{kk})"
                graph.add(
                    Node(
                        id=pid,
                        op="matmul_partial",
                        inputs=[f"A({i},{kk})", f"B({kk},{j})"],
                        params={"i": i, "j": j, "k": kk},
                        estimated_bytes=bm * bn * item,
                    )
                )
                partials.setdefault((i, j), []).append(pid)

    for (i, j), pids in partials.items():
        cid = f"C({i},{j})"
        graph.add(
            Node(
                id=cid,
                op="tile_sum",
                inputs=pids,
                params={"tile": (i, j)},
                keep=True,
                estimated_bytes=bm * bn * item,
            )
        )
        tiles[cid] = (i, j)
    return graph, tiles


def assemble(
    outputs: dict[str, Block],
    tiles: dict[str, tuple[int, int]],
    block_shape: tuple[int, int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """把最终输出块拼回完整矩阵。"""

    bm, _bk, bn = block_shape
    m, n = shape
    out = np.empty((m, n), dtype=next(iter(outputs.values())).data.dtype)
    for cid, (i, j) in tiles.items():
        out[i * bm:(i + 1) * bm, j * bn:(j + 1) * bn] = outputs[cid].data
    return out
