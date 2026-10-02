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
  **且** 前驱全部可信 **且** 实际输入内容指纹一致”的节点能复用。
  任务身份 = 图结构 (节点/算子/块/参数) **加上实际喂入源数组的内容**,
  图形结构没变不等于任务相同: 输入数组改变后, 所有传递依赖该源块的旧
  节点一律重算, 绝不能把旧完成标记当成新输入已处理; 未依赖被改动源块
  的分支原样保留 (共享输入只改一部分时只重算受影响分支)。正在写出的块
  (*.tmp / 无 meta 的孤儿文件)、损坏块及其全部后继在重开时一律失效重算,
  后继不可能提前读到半成品或失败节点的输出。检查点绑定图身份哈希,
  恢复允许改变 worker 数 / 内存预算 / 后端; 同一检查点目录不直接换图,
  图结构调整通过只读预览后派生独立候选检查点。恢复时块按需从磁盘惰性
  物化, 紧内存预算下也不会“一恢复就死锁”。
- **恢复前影响预览 (可取消)**: `CheckpointStore.plan_recovery(graph, feed)`
  是纯只读检查, 返回 `RecoveryPlan`, 列出哪些节点可复用、哪些必须重算
  及原因 (`输入改变` / `输入无法核验` / `上游依赖重算` / `结果块损坏`),
  `RecoveryPlan.describe()` 给出可读文本; 只要不调用 `apply_recovery` /
  不运行 Executor, 盘上旧 meta 和块文件一个字节都不变, 用户可以取消。
- **从旧检查点派生新图**: 插入校正、替换计算或删除分支后使用
  `CheckpointStore.plan_revision(new_graph, old_graph, feed)` 做只读预览。
  预览分别列出输入变化与计算关系变化, 只复用当前图仍能解释的节点;
  下游改接其他来源时按新依赖重新判定, 未受影响的独立分支继续复用。
  `materialize_revision(...)` 会把可复用块复制到新的候选检查点目录,
  多个候选可从同一旧检查点并行比较和执行, 任一候选失败不会覆盖原图或
  其他候选; 每份复用结果都在 `provenance` 中记录来源图、来源节点和链。
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
.venv/bin/python -m pytest tests/ -q               # 21 个测试
```

### 接续演示验证了什么

`examples/resume_workflow_demo.py` 用“共享前置 -> 4 条慢分支 -> fan-in
归约”的内存受限图, 在**独立子进程**里多次真实中断 (SIGKILL /
os._exit, 无 finally、无缓冲刷新), 而不是整图跑完后再加载:

- 共享节点提交后、慢分支执行到一半时 SIGKILL, 换 worker 数重开;
- 某个块文件 rename 之后、meta.json 更新之前 os._exit(42)
  (写出一半): 孤儿块对后继不可见, 下次打开时清除并重算;
- 篡改一个已提交块: 校验和失配, 该节点及其全部后继重算;
- 图结构完全不变但输入数组换成新内容: 影响预览显示全部旧节点必须重算,
  取消预览旧记录不变, 确认接续后结果与新输入的完整重算逐位一致;
- 双源图只改其中一个源: 未受影响分支原样复用, 受影响分支与 fan-in
  重算; 换 worker 数 + 块损坏 + 中途退出 + 输入改变同时出现时,
  最终结果仍与当前输入完整重算一致;
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

## taskgraph: 按计算结果展开的任务图

taskgraph/ 在 sciruntime 之上补充「先检查数据、再决定展开哪些后续任务」的
动态任务图：

- **发现（discovery）**：discover_step(step, params) -> (specs, done)，
  以持久化游标逐步推进，可中断、可恢复；每步可产生若干 PartitionSpec。
- **分区计算（partition）**：compute(input, params)，结果按分区身份存档。
- **汇总（aggregation）**：aggregate(entries, params)，仅对确定范围标记
  complete；发现未结束时最多为 partial，且 scope 明确记录覆盖哪些分区。

关键不变量：

- **分区身份 = hash(compute_version, input)**，与名称无关。相同分区被重复
  通知时忽略：不重复计算、不重复汇总。
- **断点恢复**：每个工作单元完成后原子落盘（Store 每分支一个状态文件）。
  改变并行度、发现新分区即停、汇总前再恢复，最终结果与从头执行一致。
- **修订（revise）**：从旧分支派生新分支（copy-on-write，失败方案不影响
  原图与兄弟方案）。仅当旧分区的输入与计算含义（compute_version）都不变时
  才保留结果，且必须被新发现过程**重新通知**才生效；被删除/拆分/合并的
  旧分区在发现结束时自动剔除，不会混入汇总范围。同名不同含义不复用。
- **观测**：事件流区分 discovery / partition / aggregation 及各自 scope；
  progress() 为只读查询，不触发新的发现或计算。

测试见 tests/test_recovery.py（空分区、迟到分区、重复通知去重、部分失败
重试、分区拆分/合并/删除的修订复用、并行度与中断点不变性、失败方案隔离、
进度查询无副作用）。
