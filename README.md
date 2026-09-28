# sciruntime：Python 并行科学计算运行时

进程内（线程池）或本机子进程（spawn 进程池）执行科学计算图的小型运行时。
三件可复用基础贯穿始终，后续新增科学算子无需改动调度与数据层：

| 基础 | 实现 | 职责 |
| --- | --- | --- |
| 计算图 | `sciruntime/graph.py` | 节点、数据依赖、DAG 校验（缺依赖/环/重复边） |
| 数据块 | `sciruntime/blocks.py` | numpy 块的原子发布、磁盘持久化、引用计数生命周期 |
| 执行状态 | `sciruntime/state.py` | 追加式 WAL + 条件状态转移，支撑幂等发布与恢复 |

调度引擎在 `sciruntime/engine.py`；算子通过 `sciruntime/operators.py` 的注册表
扩展，首轮算子在 `sciruntime/sciops.py`（矩阵源块、部分积、块求和）。

## 首轮工作流：分块矩阵乘法

`C = A @ B` 被切为源块 → 部分积 `A_ik @ B_kj` → 块求和三级 DAG，
天然体现扇入复用（`A_ik` 被所有 `j` 复用，`B_kj` 被所有 `i` 复用）。

```bash
python -m examples.matmul_demo                 # 线程池
python -m examples.matmul_demo --process       # 子进程池（spawn）
python -m examples.matmul_demo --process --workdir ./run --resume  # 断点恢复
```

## 关键语义

- **内存预算排队**：调度器按 `estimated_bytes` 做准入，预算不足时节点排队；
  单节点需求超过预算直接报错而非死等。消费者全部终结后非保留块立即回收归还预算。
- **失败不污染后继**：节点抛错或子进程崩溃（`os._exit`）后，该节点标记
  FAILED，全部后继递归标记 SKIPPED，绝不调度，因此后继读不到半成品块。
- **幂等发布**：成功发布受 WAL 中 `PENDING/RUNNING → SUCCEEDED` 的 CAS
  保护；同一节点重复完成不会重复写块或重复推进状态。块写入用临时文件 +
  `os.replace` 原子替换。
- **保存完成节点后恢复**：成功记录 fsync 落盘后才视为完成。重建 `Engine`
  指向同一 `workdir` 时，已完成节点从磁盘复用，仅补跑未完成节点；
  上次崩溃时停留在 RUNNING 的节点会重新执行。最终结果与串行 `numpy` 一致。
- **统一数据生命周期**：算子只读写 `Block`，由运行时统一发布、引用计数、
  回收；标记 `keep=True` 的最终块在运行结束后保留。

## 扩展新算子

实现 `run(inputs, params, store) -> Block` 并注册即可，依赖与数据生命周期自动复用：

```python
from sciruntime.blocks import Block
from sciruntime.operators import registry

class MyOp:
    def run(self, inputs, params, store):
        return Block(inputs[0].data * 2)

registry.register("my_op", MyOp())
```

## 测试

```bash
python -m pytest tests -q
```

覆盖：与串行 numpy 一致性（线程/进程、多种分块）、内存预算排队、
超预算报错、失败/子进程崩溃的后继跳过、幂等发布、图校验、
真实进程硬崩溃（exit code 9）后的恢复与结果一致性。

注：子进程池使用 `spawn` 启动方式，以避免本机 `forkserver` 在过长
`TMPDIR` 下创建 AF_UNIX socket 失败。
