"""算子注册表：扩展新科学算子时只需实现 Operator 并注册。

所有算子共享相同契约：从 :class:`~sciruntime.blocks.BlockStore` 读取上游块，
返回一个 :class:`~sciruntime.blocks.Block`。依赖解析、发布、生命周期均由
运行时统一处理，算子自身不感知调度与持久化。
"""

from __future__ import annotations

from typing import Protocol

from .blocks import Block, BlockStore


class Operator(Protocol):
    """科学算子协议。"""

    def run(
        self, inputs: list[Block], params: dict, store: BlockStore
    ) -> Block: ...


class OperatorRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, Operator] = {}

    def register(self, name: str, op: Operator) -> None:
        if name in self._ops:
            raise ValueError(f"operator already registered: {name}")
        self._ops[name] = op

    def get(self, name: str) -> Operator:
        try:
            return self._ops[name]
        except KeyError as exc:
            raise KeyError(f"operator not registered: {name}") from exc


registry = OperatorRegistry()
