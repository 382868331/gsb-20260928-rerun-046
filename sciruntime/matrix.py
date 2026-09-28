"""首轮科学算子: 分块矩阵运算。

提供算子:
- mat_split_rows / mat_split_cols: 把矩阵按行/列切成数据块
- matmul_partial: C_ij_t = A_i_t @ B_t_j (一个分块乘)
- reduce_sum: 沿 t 维确定性顺序求和  (sum 与顺序无关, 浮点可重现)
- elementwise: 逐元素 ufunc 链 (add/mul/scale), 演示其他算子复用同一运行时

build_blocked_matmul_graph 构造 A@B 的分块计算图, 输出块即为最终结果。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .graph import Graph
from .operators import Operator, TaskContext, register_operator


class SplitRows(Operator):
    def run(self, inputs, *, params, ctx):
        (a,) = inputs
        cuts = _cuts(a.shape[0], params["block"])  # type: ignore[index]
        return tuple(np.array(a[s:e, :], copy=True) for s, e in cuts)


class SplitCols(Operator):
    def run(self, inputs, *, params, ctx):
        (a,) = inputs
        cuts = _cuts(a.shape[1], params["block"])
        return tuple(np.array(a[:, s:e], copy=True) for s, e in cuts)


class MatmulPartial(Operator):
    """单个分块乘法。

    用固定顺序的 outer-product 点积累加, 不依赖并行 BLAS 的归约顺序,
    因此并行/子进程调度与串行参考逐位一致。
    """

    def run(self, inputs, *, params, ctx):
        a_block, b_block = inputs
        a_block = np.asarray(a_block)
        b_block = np.asarray(b_block)
        out = np.zeros((a_block.shape[0], b_block.shape[1]),
                       dtype=np.result_type(a_block, b_block))
        for t in range(a_block.shape[1]):
            out = out + np.multiply.outer(a_block[:, t], b_block[t, :])
        return (out,)


class ReduceSum(Operator):
    """按传入顺序求和。调度器保证输入按 params["order"] 的确定性顺序传入。"""

    def run(self, inputs, *, params, ctx):
        if not inputs:
            raise ValueError("reduce_sum 至少需要一个输入")
        acc = np.zeros_like(inputs[0])
        for part in inputs:
            acc = acc + part
        return (acc,)


class Identity(Operator):
    def run(self, inputs, *, params, ctx):
        return (inputs[0].copy(),)


class Elementwise(Operator):
    """逐元素算子: out = a OP b (b 可以是标量参数)。"""

    def run(self, inputs, *, params, ctx):
        kind = params["kind"]
        if kind == "scale":
            (a,) = inputs
            return (np.asarray(a) * params["scalar"],)
        if kind == "add":
            a, b = inputs
            return (np.asarray(a) + np.asarray(b),)
        if kind == "mul":
            a, b = inputs
            return (np.asarray(a) * np.asarray(b),)
        raise ValueError(f"未知 elementwise kind: {kind}")


def register_all() -> None:
    for name, op in [
        ("mat_split_rows", SplitRows()),
        ("mat_split_cols", SplitCols()),
        ("matmul_partial", MatmulPartial()),
        ("reduce_sum", ReduceSum()),
        ("elementwise", Elementwise()),
        ("identity", Identity()),
    ]:
        if name not in _registered:
            register_operator(name, op)
            _registered.add(name)


_registered: set = set()
register_all()


def _cuts(n: int, block: int) -> List[Tuple[int, int]]:
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def build_blocked_matmul_graph(
    a_shape: Tuple[int, int],
    b_shape: Tuple[int, int],
    *,
    row_block: int,
    k_block: int,
    col_block: int,
    a_name: str = "A",
    b_name: str = "B",
    dtype: str = "float64",
) -> Tuple[Graph, List[str]]:
    """构造 C = A @ B 的分块计算图。

    返回 (graph, output_block_ids)。结果块按行块顺序排列, 拼接即得 C。
    """
    m, k = a_shape
    k2, n = b_shape
    if k != k2:
        raise ValueError("A 的列数必须等于 B 的行数")
    row_cuts = _cuts(m, row_block)
    k_cuts = _cuts(k, k_block)
    col_cuts = _cuts(n, col_block)

    g = Graph()

    # 源块: 把整块 A/B 按行/列切开。为了让 K 维也分块, 这里对行块再切列。
    a_row_blocks: List[str] = []
    for i, (rs, re) in enumerate(row_cuts):
        bid = f"{a_name}_row_{i}"
        a_row_blocks.append(bid)
    g.add_node(
        f"split_{a_name}_rows",
        "mat_split_rows",
        inputs=(f"__src_{a_name}",),
        outputs=tuple(a_row_blocks),
        params={"block": row_block},
        memory_bytes=int(np.prod(a_shape)) * np.dtype(dtype).itemsize,
    )
    b_col_blocks: List[str] = []
    for j, (cs, ce) in enumerate(col_cuts):
        bid = f"{b_name}_col_{j}"
        b_col_blocks.append(bid)
    g.add_node(
        f"split_{b_name}_cols",
        "mat_split_cols",
        inputs=(f"__src_{b_name}",),
        outputs=tuple(b_col_blocks),
        params={"block": col_block},
        memory_bytes=int(np.prod(b_shape)) * np.dtype(dtype).itemsize,
    )

    # A 行块再沿 K 切; B 列块再沿 K 切
    a_tile: Dict[Tuple[int, int], str] = {}
    for i in range(len(row_cuts)):
        outs = []
        for t in range(len(k_cuts)):
            bid = f"A_tile_{i}_{t}"
            a_tile[(i, t)] = bid
            outs.append(bid)
        rsize = row_cuts[i][1] - row_cuts[i][0]
        g.add_node(
            f"split_A_{i}_cols",
            "mat_split_cols",
            inputs=(a_row_blocks[i],),
            outputs=tuple(outs),
            params={"block": k_block},
            memory_bytes=rsize * k * np.dtype(dtype).itemsize,
        )
    b_tile: Dict[Tuple[int, int], str] = {}
    for j in range(len(col_cuts)):
        outs = []
        for t in range(len(k_cuts)):
            bid = f"B_tile_{t}_{j}"
            b_tile[(t, j)] = bid
            outs.append(bid)
        csize = col_cuts[j][1] - col_cuts[j][0]
        g.add_node(
            f"split_B_{j}_rows",
            "mat_split_rows",
            inputs=(b_col_blocks[j],),
            outputs=tuple(outs),
            params={"block": k_block},
            memory_bytes=k * csize * np.dtype(dtype).itemsize,
        )

    # 分块乘 + 沿 K 归约
    output_blocks: List[str] = []
    itemsize = np.dtype(dtype).itemsize
    for i, (rs, re) in enumerate(row_cuts):
        for j, (cs, ce) in enumerate(col_cuts):
            partial_ids = []
            for t, (ks, ke) in enumerate(k_cuts):
                pid = f"C_partial_{i}_{j}_{t}"
                partial_ids.append(pid)
                g.add_node(
                    f"gemm_{i}_{j}_{t}",
                    "matmul_partial",
                    inputs=(a_tile[(i, t)], b_tile[(t, j)]),
                    outputs=(pid,),
                    params={},
                    memory_bytes=(re - rs) * (ce - cs) * itemsize,
                )
            cid = f"C_block_{i}_{j}"
            g.add_node(
                f"reduce_{i}_{j}",
                "reduce_sum",
                inputs=tuple(partial_ids),  # t 从小到大, 归约顺序确定
                outputs=(cid,),
                params={"order": list(range(len(k_cuts)))},
                memory_bytes=(re - rs) * (ce - cs) * itemsize,
            )
            output_blocks.append(cid)

    g.freeze()
    return g, output_blocks


def assemble_result(
    store_or_values: Any,
    output_blocks: Sequence[str],
    n_rows: int,
    n_cols_blocks: int,
) -> np.ndarray:
    """按行块 × 列块网格把输出块拼回完整矩阵。"""
    rows = []
    get = (
        store_or_values.get
        if hasattr(store_or_values, "get")
        else (lambda bid: store_or_values[bid])
    )
    grid = [output_blocks[i:i + n_cols_blocks]
            for i in range(0, len(output_blocks), n_cols_blocks)]
    for row_ids in grid:
        rows.append(np.hstack([np.asarray(get(bid)) for bid in row_ids]))
    return np.vstack(rows)


def serial_reference_matmul(
    a: np.ndarray, b: np.ndarray, k_block: int = 20
) -> np.ndarray:
    """与分块图完全相同归约顺序的串行参考实现 (可逐位对比):
    先做块内 matmul, 再沿 K 按块顺序累加。"""
    a = np.asarray(a)
    b = np.asarray(b)
    k = a.shape[1]
    blocks = []
    for s in range(0, k, k_block):
        e = min(s + k_block, k)
        aa, bb = a[:, s:e], b[s:e, :]
        part = np.zeros((a.shape[0], b.shape[1]), dtype=np.result_type(a, b))
        for t in range(e - s):
            part = part + np.multiply.outer(aa[:, t], bb[t, :])
        blocks.append(part)
    acc = np.zeros_like(blocks[0])
    for part in blocks:
        acc = acc + part
    return acc
