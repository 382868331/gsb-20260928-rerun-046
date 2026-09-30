# sciruntime: 并行科学计算运行时

本机进程内 / 本机子进程的轻量并行科学计算运行时。三个可复用基础:

- **计算图 `Graph`**: 节点 = 算子调用, 边 = 数据块依赖; 自动拓扑排序、环检测。
- **数据块 `BlockStore`**: numpy 数组的线程安全仓库, 引用计数 (pin) + 消费者
  计数管理生命周期, 中间块自动回收, 最终输出显式保留。
- **执行状态 `Executor` / `CheckpointStore`**: 并行调度、内存预算排队、失败
  传播、幂等发布, 以及完成节点校验后的细粒度持久化与恢复。

全部在本机运行, 无集群依赖。

## 关键语义

- **内存预算排队**: 启动节点前检查 `驻留内存 + 节点新增内存 <= budget`,
  超预算的节点排队, 等前序任务结束释放内存后再调度 (`executor.py`)。
- **失败隔离**: 节点失败后其全部后继标记为 `skipped` 不会执行, 因此不可
  能读到半成品; 失败节点在成功提交前的输出对任何消费者不可见。
- **幂等发布**: 每节点只允许发布一次, 重试/恢复重放重复执行也不会重复
  `put`; BlockStore 对重复 block_id 的发布直接报错做防御。
- **细粒度检查点恢复**: 所有输出块先原子落盘 (temp + rename) 并记录 SHA-256,
  再更新 `meta.json`; meta 同时绑定稳定图身份和输入指纹。只有完整发布并
  核验过的节点可复用, 正在执行/写出的节点在重启后重算。恢复时可以改变工作者
  数量、后端和预算, 但不能换图或换输入。
- **可扩展算子**: 实现 `Operator.run(inputs, params, ctx)` 并
  `register_operator` 即可, 依赖调度与数据生命周期完全复用, 无需改运行时。

## 首轮算子: 分块矩阵乘法

`sciruntime/matrix.py` 把 `C = A @ B` 构造成 “行/列切分 → K 维切块 →
分块乘 → 沿 K 确定性归约” 的计算图。块内乘法用固定顺序的 outer-product
累加, 串行参考 `serial_reference_matmul` 采用同一归约顺序, 所以并行结果
与串行逐位一致 (`np.array_equal`, 不是 allclose)。

## 使用

```bash
.venv/bin/python -m examples.blocked_matmul_demo   # 线程/子进程/恢复三个场景
.venv/bin/python -m pytest tests/ -q               # 10 个测试
```

`tests/test_recovery.py` 会启动三个独立进程, 在共享前置后、两个慢分支执行中
分别退出, 再用不同工作者数量重开, 并核对节点复用/重算集合与峰值在途数据。

```python
from sciruntime import Executor
from sciruntime.matrix import build_blocked_matmul_graph, assemble_result

graph, outputs = build_blocked_matmul_graph(
    A.shape, B.shape, row_block=32, k_block=24, col_block=40
)
stats = Executor(graph, max_workers=4, backend="process").run(
    feed={"__src_A": A, "__src_B": B}
)
C = assemble_result(ex.store, outputs, A.shape[0], n_cols_blocks=...)
```

新增算子:

```python
from sciruntime.operators import Operator, register_operator

class MyOp(Operator):
    def run(self, inputs, *, params, ctx):
        return (inputs[0] * 2,)

register_operator("my_op", MyOp())  # 然后即可在图节点中用 operator="my_op"
```
