"""sciruntime: 进程内/子进程并行科学计算运行时。

三件可复用基础：

- :class:`~sciruntime.graph.Graph`：计算图（节点 + 数据依赖）。
- :class:`~sciruntime.blocks.BlockStore`：数据块的可复用存储与生命周期。
- :class:`~sciruntime.state.RunState`：持久化的执行状态（结果/失败/恢复）。

调度由 :class:`~sciruntime.engine.Engine` 负责，算子在
:mod:`sciruntime.operators` 中注册，后续新算子沿用同一套依赖与数据生命周期。
"""

from .blocks import Block, BlockHandle, BlockStore
from .engine import Engine, EngineConfig, EngineResult
from .graph import Graph, Node
from .operators import Operator, OperatorRegistry, registry
from .state import NodeStatus, RunState

__all__ = [
    "Block",
    "BlockHandle",
    "BlockStore",
    "Engine",
    "EngineConfig",
    "EngineResult",
    "Graph",
    "Node",
    "Operator",
    "OperatorRegistry",
    "registry",
    "RunState",
    "NodeStatus",
]
