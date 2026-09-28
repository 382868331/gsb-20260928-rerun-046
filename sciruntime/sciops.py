"""首轮科学算子：分块矩阵的源块、乘法部分积、加法聚合。

算子只消费/生产 :class:`Block`，不感知调度，后续新增算子沿用同一契约。
"""

from __future__ import annotations

import numpy as np

from .blocks import Block, BlockStore
from .operators import registry


class MatrixSource:
    """把一个预先给出的矩阵按 (tile_i, tile_j) 切块发布。"""

    def run(self, inputs: list[Block], params: dict, store: BlockStore) -> Block:
        matrix: np.ndarray = params["matrix"]
        tile_i, tile_j = params["tile"]
        bm, bn = params["block_shape"]
        block = matrix[tile_i * bm:(tile_i + 1) * bm, tile_j * bn:(tile_j + 1) * bn]
        return Block(np.ascontiguousarray(block), {"kind": "matrix_tile"})


class MatMulPartial:
    """C_ij 的第 k 个部分积：A_ik @ B_kj。"""

    def run(self, inputs: list[Block], params: dict, store: BlockStore) -> Block:
        a_tile, b_tile = inputs[0].data, inputs[1].data
        return Block(
            np.ascontiguousarray(a_tile @ b_tile),
            {"kind": "partial", "i": params["i"], "j": params["j"], "k": params["k"]},
        )


class TileSum:
    """对同一 (i,j) 的若干部分积求和，得到输出矩阵块 C_ij。"""

    def run(self, inputs: list[Block], params: dict, store: BlockStore) -> Block:
        total = np.array(inputs[0].data, copy=True)
        for block in inputs[1:]:
            total = total + block.data
        return Block(np.ascontiguousarray(total), {"kind": "result_tile", "tile": list(params["tile"])})


def register_all() -> None:
    """幂等注册（重复调用安全），供子进程导入。"""

    for name, op in (
        ("matrix_source", MatrixSource()),
        ("matmul_partial", MatMulPartial()),
        ("tile_sum", TileSum()),
    ):
        try:
            registry.register(name, op)
        except ValueError:
            pass


register_all()
