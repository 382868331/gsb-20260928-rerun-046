# sciruntime: 并行科学计算运行时

本机进程内 / 本机子进程的轻量并行科学计算运行时。三个可复用基础:

- **计算图 `Graph`**: 节点 = 算子调用, 边 = 数据块依赖; 自动拓扑排序、环检测。
- **数据块 `BlockStore`**: numpy 数组的线程安全仓库, 引用计数 (pin) + 消费者
  计数管理生命周期, 中间块自动回收, 最终输出显式保留。
- **执行状态 `Executor` / `CheckpointStore`**: 并行调度、内存预算排队、失败
  传播、幂等发布, 以及完成节点的持久化与恢复。

全部在本机运行, 无集群依赖。

## 关键语义

- **内存预算排队**: 启动节点前检查 `驻留内存 + 节点新增内存 <= budget`,
  超预算的节点排队, 等前序任务结束释放内存后再调度 (`executor.py`)。
- **失败隔离**: 节点失败后其全部后继标记为 `skipped` 不会执行, 因此不可
  能读到半成品; 失败节点在成功提交前的输出对任何消费者不可见。
- **幂等发布**: 每节点只允许发布一次, 重试/恢复重放重复执行也不会重复
  `put`; BlockStore 对重复 block_id 的发布直接报错做防御。
- **细粒度检查点接续**: 输出块先原子落盘 (temp + rename), 再连同
  sha256 一起更新 `meta.json`; 只有“完整发布 **且** 校验和核验通过
  **且** 前驱全部可信”的节点能复用。正在写出的块 (*.tmp / 无 meta
  的孤儿文件)、损坏块及其全部后继在重开时一律失效重算, 后继不可能
  提前读到半成品或失败节点的输出。检查点绑定图身份哈希, 恢复允许
  改变 worker 数 / 内存预算 / 后端, 但不允许换图复用。恢复时块按需
  从磁盘惰性物化, 紧内存预算下也不会“一恢复就死锁”。
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
.venv/bin/python -m examples.resume_workflow_demo  # 真实硬中断后变 worker 接续
.venv/bin/python -m pytest tests/ -q               # 13 个测试
```

### 接续演示验证了什么

`examples/resume_workflow_demo.py` 用“共享前置 -> 4 条慢分支 -> fan-in
归约”的内存受限图, 在**独立子进程**里多次真实中断 (SIGKILL /
os._exit, 无 finally、无缓冲刷新), 而不是整图跑完后再加载:

- 共享节点提交后、慢分支执行到一半时 SIGKILL, 换 worker 数重开;
- 某个块文件 rename 之后、meta.json 更新之前 os._exit(42)
  (写出一半): 孤儿块对后继不可见, 下次打开时清除并重算;
- 篡改一个已提交块: 校验和失配, 该节点及其全部后继重算;
- 每次重开核对复用集合 (stats.from_checkpoint)、重算集合
  (stats.recomputed)、峰值在途数据 (stats.peak_inflight_bytes)
  及与串行参考逐位一致的最终结果。

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
