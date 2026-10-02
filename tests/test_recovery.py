import pytest

from taskgraph import Engine, Job, PartitionSpec, Store
from taskgraph.engine import Plugin


# ---------------------------------------------------------------- plugins
def make_plugin(specs_by_step, compute=None, aggregate=None, compute_version="v1"):
    """specs_by_step: list of lists; step i reveals specs_by_step[i];
    discovery is done after the last step."""
    calls = {"discover": 0, "compute": []}

    def discover_step(step, params):
        calls["discover"] += 1
        if step >= len(specs_by_step):
            return [], True
        return specs_by_step[step], step == len(specs_by_step) - 1

    def default_compute(part_input, params):
        calls["compute"].append(part_input)
        return sum(part_input)

    def default_aggregate(entries, params):
        return sum(e["result"] for e in entries)

    plugin = Plugin(
        discover_step=discover_step,
        compute=compute or default_compute,
        aggregate=aggregate or default_aggregate,
    )
    return plugin, calls


@pytest.fixture
def env(tmp_path):
    store = Store(str(tmp_path / "states"))
    engine = Engine(store)
    return engine, store


def submit(engine, branch="main", params=None, specs=None, **kw):
    specs = [[PartitionSpec(f"p{i}", s)] for i, s in enumerate(specs or [[1], [2], [3]])]
    plugin, calls = make_plugin(specs, **kw)
    engine.register("job", plugin)
    engine.submit(branch, Job("job", params or {}))
    return calls


# ---------------------------------------------------------------- tests
def test_basic_run_completes(env):
    engine, _ = env
    submit(engine, specs=[[1, 2], [3, 4], [5]])
    engine.run("main")
    assert engine.result("main") == 15
    prog = engine.progress("main")
    assert prog["discovery"]["done"] is True
    assert prog["aggregation"]["status"] == "complete"
    assert len(prog["aggregation"]["scope"]) == 3


def test_empty_partition_set(env):
    engine, _ = env
    plugin, _ = make_plugin([[]])  # discovery done, zero partitions
    engine.register("job", plugin)
    engine.submit("main", Job("job"))
    engine.run("main")
    assert engine.result("main") == 0
    assert engine.progress("main")["aggregation"]["scope"] == []


def test_duplicate_notification_not_recomputed(env):
    engine, _ = env
    spec = PartitionSpec("p", [1, 2])
    # same partition (same input) notified in two discovery steps
    plugin, calls = make_plugin([[spec], [spec]])
    engine.register("job", plugin)
    engine.submit("main", Job("job"))
    engine.run("main")
    assert calls["compute"].count([1, 2]) == 1
    aggs = engine.events("main", "aggregation")
    assert aggs[-1]["detail"].startswith("aggregation complete")
    assert len(aggs[-1]["scope"]) == 1
    assert engine.result("main") == 3


def test_late_partition_invalidates_partial_aggregation(env):
    engine, _ = env
    # step 0 reveals one partition and is NOT done; step 1 adds a late one
    plugin, _ = make_plugin([
        [PartitionSpec("early", [10])],
        [PartitionSpec("late", [20])],
    ])
    # patch: step 0 must not be "done"
    orig = plugin.discover_step
    plugin.discover_step = lambda s, p: (orig(s, p)[0], s >= 1)
    engine.register("job", plugin)
    engine.submit("main", Job("job"))
    engine.run("main")
    aggs = engine.events("main", "aggregation")
    statuses = [a["detail"] for a in aggs]
    assert any("partial" in s for s in statuses)
    assert aggs[-1]["detail"].startswith("aggregation complete")
    assert len(aggs[-1]["scope"]) == 2
    assert engine.result("main") == 30


def test_aggregation_not_complete_before_discovery_done(env):
    engine, _ = env
    plugin, _ = make_plugin([[PartitionSpec("a", [1])]])
    plugin.discover_step = lambda s, p: ([PartitionSpec("a", [1])], False) if s == 0 else ([], True)
    engine.register("job", plugin)
    engine.submit("main", Job("job"))
    engine.run("main", max_steps=1)
    engine.run("main", max_steps=1)  # compute runs, discovery still open
    prog = engine.progress("main")
    assert prog["aggregation"]["status"] != "complete"
    with pytest.raises(RuntimeError):
        engine.result("main")
    engine.run("main")
    assert engine.result("main") == 1


def test_partial_failure_blocks_aggregation_and_recovers(env):
    engine, _ = env
    broken = {"on": True}

    def flaky(part_input, params):
        if part_input == [2] and broken["on"]:
            raise ValueError("boom")
        return sum(part_input)

    submit(engine, specs=[[1], [2], [3]], compute=flaky)
    engine.run("main")
    prog = engine.progress("main")
    assert len(prog["partitions"]["failed"]) == 1
    assert prog["aggregation"]["status"] != "complete"
    broken["on"] = False
    engine.run("main")  # retry succeeds
    assert engine.result("main") == 6
    assert "failed" not in engine.progress("main")["partitions"]


def test_parallelism_and_interruption_invariant(env):
    engine, store = env
    specs = [[i, i + 1] for i in range(0, 20, 2)]
    # reference: single-threaded straight run
    submit(engine, "ref", specs=specs)
    engine.run("ref")
    expected = engine.result("ref")
    # variant: parallel, stop on every new partition, resume repeatedly
    submit(engine, "var", specs=specs)
    for _ in range(50):
        state = engine.run("var", max_workers=8, stop_on_new_partition=True)
        if state["aggregation"]["status"] == "complete":
            break
    assert engine.result("var") == expected
    # scope identical too
    assert (engine.progress("var")["aggregation"]["scope"]
            == engine.progress("ref")["aggregation"]["scope"])


def test_resume_before_aggregation_matches_fresh(env):
    engine, _ = env
    submit(engine, "fresh", specs=[[1], [2], [3]])
    engine.run("fresh")
    submit(engine, "resumed", specs=[[1], [2], [3]])
    engine.run("resumed", max_steps=1)   # stop mid-discovery
    engine.run("resumed", max_steps=1)
    engine.run("resumed")                # resume and aggregate
    assert engine.result("resumed") == engine.result("fresh")


def test_revision_reuses_only_matching_partitions(env):
    engine, _ = env
    calls = submit(engine, "main", specs=[[1], [2], [3]])
    engine.run("main")
    assert engine.result("main") == 6

    # revision A: params change but partition inputs/meaning unchanged
    calls2 = {"compute": []}
    plugin2, _ = make_plugin(
        [[PartitionSpec("p0", [1])], [PartitionSpec("p1", [2])],
         [PartitionSpec("p2", [3])]],
        compute=lambda i, p: calls2["compute"].append(i) or sum(i),
    )
    engine.register("job", plugin2)
    engine.revise("main", "revA", new_params={"note": "same work"})
    engine.run("revA")
    assert calls2["compute"] == []  # everything reused
    assert engine.result("revA") == 6

    # revision B: partition split ([2] -> [1,1]) and one deleted, one added
    calls3 = {"compute": []}
    plugin3, _ = make_plugin(
        [[PartitionSpec("q0", [1])], [PartitionSpec("q1", [1, 1])],
         [PartitionSpec("q2", [9])]],
        compute=lambda i, p: calls3["compute"].append(i) or sum(i),
    )
    engine.register("job", plugin3)
    engine.revise("main", "revB")
    engine.run("revB")
    # [1] reused (same input+version); [1,1] and [9] computed; [2],[3] gone
    assert sorted(calls3["compute"]) == [[1, 1], [9]]
    assert engine.result("revB") == 12
    dropped = [e for e in engine.events("revB", "partition")
               if "dropped stale" in e["detail"]]
    assert len(dropped) == 2  # [2] and [3] never re-notified
    # original untouched
    assert engine.result("main") == 6


def test_revision_with_new_compute_version_reuses_nothing(env):
    engine, _ = env
    submit(engine, "main", specs=[[1], [2]])
    engine.run("main")
    calls = {"compute": []}
    plugin, _ = make_plugin(
        [[PartitionSpec("p0", [1])], [PartitionSpec("p1", [2])]],
        compute=lambda i, p: calls["compute"].append(i) or sum(i) * 10,
    )
    engine.register("job", plugin)
    engine.revise("main", "rev", new_compute_version="v2")
    engine.run("rev")
    assert sorted(calls["compute"]) == [[1], [2]]  # names equal, meaning differs
    assert engine.result("rev") == 30
    assert engine.result("main") == 3


def test_failed_revision_does_not_touch_source_or_siblings(env):
    engine, _ = env
    submit(engine, "main", specs=[[1], [2]])
    engine.run("main")
    engine.revise("main", "good")
    engine.revise("main", "bad", new_compute_version="v2")

    def boom(part_input, params):
        raise RuntimeError("always fails")

    plugin, _ = make_plugin(
        [[PartitionSpec("p0", [1])], [PartitionSpec("p1", [2])]],
        compute=boom,
    )
    engine.register("job", plugin)
    engine.run("bad")
    assert engine.progress("bad")["aggregation"]["status"] != "complete"
    # source and sibling revision unaffected
    assert engine.result("main") == 3
    engine.register("job", make_plugin(
        [[PartitionSpec("p0", [1])], [PartitionSpec("p1", [2])]])[0])
    engine.run("good")
    assert engine.result("good") == 3


def test_progress_query_has_no_side_effects(env):
    engine, store = env
    calls = submit(engine, "main", specs=[[1], [2]])
    engine.run("main", max_steps=1)
    before_discover = calls["discover"]
    snap = store.load("main")
    engine.progress("main")
    engine.progress("main")
    assert calls["discover"] == before_discover  # no discovery triggered
    assert store.load("main") == snap            # state untouched


def test_events_distinguish_kinds_and_scope(env):
    engine, _ = env
    submit(engine, "main", specs=[[1], [2]])
    engine.run("main")
    kinds = {e["kind"] for e in engine.events("main")}
    assert {"discovery", "partition", "aggregation"} <= kinds
    for e in engine.events("main", "partition"):
        assert len(e["scope"]) >= 1
    agg = engine.events("main", "aggregation")[-1]
    assert len(agg["scope"]) == 2


def test_provenance_of_every_result(env):
    engine, _ = env
    submit(engine, "main", specs=[[7], [8]])
    engine.run("main")
    prov = engine.progress("main")["provenance"]
    assert sorted(p["input"] for p in prov.values()) == [[7], [8]]
