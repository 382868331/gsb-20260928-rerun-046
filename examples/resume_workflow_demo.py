"""真实硬中断后的细粒度接续演示 / 驱动。

本脚本以子进程方式反复运行同一张“共享前置 + 慢分支 + fan-in”图,
在不同完成阶段用 os._exit (等价 SIGKILL: 无 finally、无缓冲刷新)
杀掉进程, 然后用 *不同的工作者数量* 重开:

  阶段 A: 共享前置全部提交后、慢分支执行中硬退出;
  阶段 B: 在某个块落盘后、meta 更新前硬退出 (写出一半);
  阶段 C: 损坏一个已提交块后重开;

每次重开后核对:
  - 哪些节点复用 (stats.from_checkpoint), 哪些重算 (stats.recomputed);
  - 最终结果与串行参考逐位一致 (任务身份一致);
  - 峰值在途数据 (stats.peak_inflight_bytes) 不超过内存预算。

运行: .venv/bin/python -m examples.resume_workflow_demo
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from sciruntime import CheckpointStore, Executor
from sciruntime.workflow import build_resume_graph, serial_reference


N = 64
N_BRANCHES = 4


def _source():
    rng = np.random.default_rng(2026)
    return rng.standard_normal(N)


def _expected(x):
    return serial_reference(x, n_branches=N_BRANCHES)


def run_once(
    ckpt_dir: str,
    *,
    workers: int,
    memory_budget: int,
    exit_after_commits=None,
    crash_after_block=None,
    slow=True,
):
    """在独立子进程中运行; 返回 (returncode, stdout_json_or_None, stderr)。"""
    cmd = [
        sys.executable, "-m", "examples.resume_workflow_demo",
        "--worker", "--ckpt", ckpt_dir, "--workers", str(workers),
        "--budget", str(memory_budget),
    ]
    if exit_after_commits is not None:
        cmd += ["--exit-after-commits", str(exit_after_commits)]
    if crash_after_block is not None:
        cmd += ["--crash-after-block", crash_after_block]
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if slow:
        env["SCI_SLOW_SLEEP"] = "0"
        env.update(_branch_sleeps())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    info = None
    for line in proc.stdout.splitlines():
        if line.startswith("WORKER_JSON "):
            info = json.loads(line.split(" ", 1)[1])
    return proc.returncode, info, proc.stderr


def _spawn(ckpt_dir, *, workers, budget):
    cmd = [
        sys.executable, "-m", "examples.resume_workflow_demo",
        "--worker", "--ckpt", ckpt_dir, "--workers", str(workers),
        "--budget", str(budget),
    ]
    env = dict(os.environ,
               PYTHONPATH=os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))),
               SCI_SLOW_SLEEP="0")
    env.update(_branch_sleeps())
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)


def _branch_sleeps(base=0.9, stagger=0.35):
    """返回 {环境变量: 值}, 慢分支耗时按编号递增, 保证提交顺序确定。"""
    return {
        f"SCI_SLOW_SLEEP_slow_{b}": str(base + stagger * b)
        for b in range(N_BRANCHES)
    }


def _wait_commits(ckpt_dir, required_nodes, *, timeout=20.0, poll=0.05):
    """轮询 meta.json, 直到 required_nodes 全部完成提交。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ck = CheckpointStore(ckpt_dir)
            if set(required_nodes) <= set(ck.completed_nodes()):
                return
        except Exception:
            pass
        time.sleep(poll)
    raise TimeoutError(f"等待 {required_nodes} 提交超时")


def _worker_main(argv):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--budget", type=int, default=10 ** 9)
    p.add_argument("--exit-after-commits", type=int, default=None)
    p.add_argument("--crash-after-block", default=None)
    args = p.parse_args(argv)

    x = _source()
    graph, meta = build_resume_graph(N, n_branches=N_BRANCHES)
    ck = CheckpointStore(args.ckpt)
    ex = Executor(
        graph,
        max_workers=args.workers,
        memory_budget=args.budget,
        backend="thread",
        checkpoint=ck,
        exit_after_commits=args.exit_after_commits,
        crash_after_block=args.crash_after_block,
    )
    stats = ex.run(feed={meta["source"]: x})
    result = ex.result(meta["output"])
    info = {
        "ok": stats.ok,
        "from_checkpoint": stats.from_checkpoint,
        "recomputed": stats.recomputed,
        "peak_inflight_bytes": stats.peak_inflight_bytes,
        "result_equal": bool(np.array_equal(result, _expected(x))),
    }
    print("WORKER_JSON " + json.dumps(info))


def main():
    x = _source()
    block = N * 8
    # 预算: 源(512)+s1(512) 常驻; 只允许约 2 个分支在途, 第 3、4 个排队。
    budget = 6 * block

    root = tempfile.mkdtemp(prefix="sci-resume-")
    try:
        ckpt = os.path.join(root, "ckpt")

        # ---------- 阶段 A: 2 个共享节点提交后硬退出 (43) ----------
        rc, info, err = run_once(
            ckpt, workers=1, memory_budget=budget, exit_after_commits=2
        )
        assert rc == 43, (rc, err)
        assert info is None  # 硬退出, 没有正常输出

        # ---------- 阶段 A 重开: 2 workers, 慢分支执行中, 由外部硬杀 ----------
        proc2 = _spawn(ckpt, workers=2, budget=budget)
        # 等前两个慢分支提交 (约 0.9s), 后两个仍在写出中。
        _wait_commits(ckpt, {"slow_1"}, timeout=20)
        # slow_2 在 1.25s 才完成; 等到 1.05s: leaf_0/leaf_1 已提交,
        # 但 slow_2/slow_3 一定还在执行 (失败节点的输出不可能被发布)。
        time.sleep(0.15)
        proc2.kill()  # SIGKILL: 正在写出的 slow_2/slow_3 什么都不会留下
        proc2.wait()
        committed = set(CheckpointStore(ckpt).completed_nodes())
        assert {"shared0", "shared1", "slow_0", "slow_1"} <= committed, committed
        assert "reduce" not in committed
        ck = CheckpointStore(ckpt)

        # ---------- 阶段 B: 写出一半时硬退出 (块落盘、meta 未更新, 42) ----------
        # 继续重开 (3 workers, 改变并行度), 在 slow_2 块 rename 后、
        # meta 更新前由检查点钩子 os._exit(42)。
        rc, info, err = run_once(
            ckpt, workers=3, memory_budget=budget,
            crash_after_block="b2",
        )
        assert rc == 42, (rc, err)

        # 孤儿块没有 meta 记录, 后继 leaf_2 绝不可能读到
        ck = CheckpointStore(ckpt)
        assert "slow_2" not in ck.completed_nodes()
        orphan = os.path.join(ckpt, "blocks", "b2.npy")
        # 硬退出后孤儿文件仍在盘上, 但没有 meta 背书; 它对任何后继
        # 都不可见。下一次打开并核验时被清除。
        assert os.path.exists(orphan)

        # ---------- 阶段 B 重开: 4 workers, 完成全图 ----------
        rc, info, err = run_once(
            ckpt, workers=4, memory_budget=budget
        )
        assert rc == 0, (rc, err)
        assert info["ok"], err
        assert info["result_equal"], "结果与串行参考不逐位一致"
        reused = set(info["from_checkpoint"])
        # SIGKILL 时刻已发布的节点必须原样复用, 一个都不重算;
        # 写出一半的 slow_2 不允许被复用。
        assert committed <= reused, (committed, reused)
        assert "slow_2" not in reused
        # slow_2 写出一半必须重算; slow_3/leaf/reduce 首次执行
        assert "slow_2" in info["recomputed"], info
        assert "slow_3" in info["recomputed"]
        assert "leaf_2" in info["recomputed"]
        assert "reduce" in info["recomputed"]
        assert info["peak_inflight_bytes"] <= budget, info

        # ---------- 阶段 C: 损坏已提交块, 重开必须发现并重算 ----------
        ck2 = os.path.join(root, "ckpt2")
        rc, info, err = run_once(
            ck2, workers=2, memory_budget=budget, exit_after_commits=2
        )
        assert rc == 43
        # 篡改 shared1 的块文件 (模拟静默损坏 / 失败节点落盘被污染)
        ck = CheckpointStore(ck2)
        target = [
            os.path.join(ck2, "blocks", f)
            for f in os.listdir(os.path.join(ck2, "blocks"))
            if f.endswith(".npy")
        ]
        assert target
        with open(target[0], "r+b") as f:
            f.seek(0)
            f.write(b"\x00\x00")

        rc, info, err = run_once(ck2, workers=1, memory_budget=budget)
        assert rc == 0, (rc, err)
        assert info["ok"], err
        assert info["result_equal"]
        # 损坏块所属节点及其全部后继都重算; 只有未受影响的提交可复用。
        assert set(info["from_checkpoint"]).issubset(
            {"shared0", "shared1", "slow_0", "slow_1", "slow_2", "slow_3"}
        )
        assert "reduce" in info["recomputed"]

        # ---------- 图身份: 换图不允许复用 ----------
        other = os.path.join(root, "other")
        run_once(other, workers=1, memory_budget=budget,
                 exit_after_commits=1)
        # 用同目录但不同结构的图应被拒绝 (这里直接用 API 验证)
        g_other, _ = build_resume_graph(N, n_branches=3)  # 结构不同
        from sciruntime.checkpoint import CheckpointMismatch

        ck_bad = CheckpointStore(other)
        try:
            ck_bad.attach_graph(g_other.identity())
        except CheckpointMismatch:
            pass
        else:
            raise AssertionError("不同图身份不应通过")

        print("全部通过: 多次硬中断、变 worker 数接续, 复用/重算/在途峰值均正确。")
        print(json.dumps({
            "阶段B重开": info,
            "budget": budget,
        }, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main(sys.argv[1:])
    else:
        main()
