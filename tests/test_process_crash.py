"""子进程内崩溃（os._exit）时，后继被标记跳过，结果不被污染。"""

import os

import numpy as np
import pytest

from sciruntime import Engine, EngineConfig, Graph, Node
from sciruntime.blocks import Block
from sciruntime.operators import OperatorRegistry, registry
from sciruntime.state import NodeStatus


class CrashOp:
    def run(self, inputs, params, store):
        os._exit(7)


class IdentityOp:
    def run(self, inputs, params, store):
        return Block(inputs[0].data.copy())


def test_subprocess_crash_marks_successors_skipped(tmp_path):
    registry.register("crash", CrashOp())
    registry.register("identity", IdentityOp())
    graph = Graph()
    graph.add(Node("bad", "crash", estimated_bytes=64))
    graph.add(Node("child", "identity", inputs=["bad"], estimated_bytes=64))
    with Engine(
        EngineConfig(workdir=tmp_path, executor="process", max_workers=2)
    ) as engine:
        result = engine.run(graph)
    assert result.statuses["bad"] == NodeStatus.FAILED
    assert result.statuses["child"] == NodeStatus.SKIPPED
    assert not engine.store.has("bad")
    assert not engine.store.has("child")
