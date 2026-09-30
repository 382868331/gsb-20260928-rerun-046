"""sciruntime: 进程内 / 本机子进程的并行科学计算运行时。

核心抽象:
- Graph:   计算图 (节点 = 算子调用, 边 = 数据块依赖)
- Block:   数据块的元数据 (真正的 ndarray 由 BlockStore 管理生命周期)
- Executor:执行引擎, 对外发布结果是幂等的 (同一节点重复执行只发布一次)
"""
from .blocks import Block, BlockStore
from .graph import Graph, Node
from .operators import Operator, register_operator, operator_registry
from .executor import Executor, ExecutionStats
from .checkpoint import (
    CheckpointStore,
    ResumeCancelled,
    ResumePlan,
    fingerprint_array,
)

__all__ = [
    "Block",
    "BlockStore",
    "Graph",
    "Node",
    "Operator",
    "register_operator",
    "operator_registry",
    "Executor",
    "ExecutionStats",
    "CheckpointStore",
    "ResumeCancelled",
    "ResumePlan",
    "fingerprint_array",
]
