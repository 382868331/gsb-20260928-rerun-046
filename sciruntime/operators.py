"""算子注册表。

扩展新的科学算子时, 只需:

    class MyOp(Operator):
        def run(self, inputs, *, params, ctx):
            ...
            return tuple_of_ndarrays
    register_operator("my_op", MyOp)

然后在图里以 operator="my_op" 建节点即可——依赖调度与数据生命周期完全复用。

run() 的返回值约定: 与节点 outputs 等长的 tuple/list; 每个元素一个 ndarray。
算子内部抛异常即视为任务失败, 由执行器处理重试/取消后继。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class TaskContext:
    node_id: str
    attempt: int


class Operator:
    """算子基类。子类实现 run。"""

    def run(
        self,
        inputs: Sequence[np.ndarray],
        *,
        params: Mapping[str, Any],
        ctx: TaskContext,
    ) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


operator_registry: Dict[str, Operator] = {}


def register_operator(name: str, operator: Operator) -> None:
    if name in operator_registry:
        raise RuntimeError(f"算子 {name!r} 已注册")
    operator_registry[name] = operator


def get_operator(name: str) -> Operator:
    try:
        return operator_registry[name]
    except KeyError:
        raise KeyError(f"未注册的算子: {name!r}") from None
