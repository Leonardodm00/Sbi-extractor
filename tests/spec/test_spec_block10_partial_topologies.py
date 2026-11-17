"""Spec-derived tests of Block 10 (partial topologies dropped whole), with its
boundaries to Block 6 (the plan) and Block 7 (export and gate). Tester-written
from specs/SPEC.md section 5, Block 10 (and 2.5, 2.6, 6.2-6.5, 7.1-7.3).

Expected values are computed here from the files on disk (detections read with
numpy, job_args edited here) or by hand; the code under test is only run.
K(t) = 3 x 2 = 6 throughout (a product: the sum would be 5).
"""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest

import _bank_helpers as H

K_WORKERS, K_PER = 3, 2
K = K_WORKERS * K_PER
PLAN = os.path.join(H.EXPORT, "bank_plan.py")
TASK = os.path.join(H.EXPORT, "bank_task.py")
GATE = os.path.join(H.EXPORT, "bank_gate.py")


@pytest.fixture(scope="module", autouse=True)
def _env():
    H.sbi_hpc_dir()


# --------------------------------------------------------------- unit level

def test_dispatch_size_product_and_refusals(tmp_path):
    """P1: K(t) = _resolved_n_workers x n_params_per_worker; each a JSON integer
    >= 1, else LoadError naming the key."""
    import bank_plan_load as LOAD
    d = tmp_path / "task"
    d.mkdir()
    ja = d / "job_args.json"

    def put(doc):
        ja.write_text(json.dumps(doc), encoding="ascii")

    put({"_resolved_n_workers": 3, "n_params_per_worker": 2})
    assert LOAD.dispatch_size(str(d)) == 6                  # product, not 5
    put({"_resolved_n_workers": 192, "n_params_per_worker": 1})
    assert LOAD.dispatch_size(str(d)) == 192
    bad = [("_resolved_n_workers", True), ("_resolved_n_workers", 0),
           ("_resolved_n_workers", -1), ("_resolved_n_workers", 4.0),
           ("_resolved_n_workers", "4"), ("_resolved_n_workers", None),
           ("n_params_per_worker", False), ("n_params_per_worker", 2.5),
           ("n_params_per_worker", float("nan"))]
    for key, value in bad:
        doc = {"_resolved_n_workers": 3, "n_params_per_worker": 2}
        doc[key] = value
        put(doc)
        with pytest.raises(LOAD.LoadError) as ei:
            LOAD.dispatch_size(str(d))
        assert key in str(ei.value), (key, value, str(ei.value))
    for key in ("_resolved_n_workers", "n_params_per_worker"):
        doc = {"_resolved_n_workers": 3, "n_params_per_worker": 2}
        del doc[key]
        put(doc)
        with pytest.raises(LOAD.LoadError) as ei:
            LOAD.dispatch_size(str(d))
        assert key in str(ei.value)


def test_raw_finished_counts_rule(tmp_path):
    """P2: f = #iter_<i>.json files + #non-blank lines of _failures.jsonl (0 without
    it); nothing else counts. Hand-counted per folder."""
    import bank_plan_load as LOAD
    root = tmp_path / "raw"
    root.mkdir()

    def topo(k):
        p = root / ("topo_%05d" % k)
        p.mkdir()
        return p

    t0 = topo(0)                                    # 3 sidecars, no failures file -> 3
    for i in range(3):
        (t0 / ("iter_%05d.json" % i)).write_text("{}")
    t1 = topo(1)                                    # 1 sidecar + 2 lines (blank and
    (t1 / "iter_00007.json").write_text("{}")       # whitespace-only lines and a final
    (t1 / "_failures.jsonl").write_bytes(b'{"a":1}\n\n  \t \n\r\n{"a":2}')   # line without \n)
    t2 = topo(2)                                    # decoys only -> 0
    for name in ("iter_00001.json.tmp", "iter_x.json", "topology_meta.json",
                 "iter_00002.npz", "xiter_00003.json", "iter_00004.JSON"):
        (t2 / name).write_text("{}")
    (t2 / "iter_00005.json").mkdir()                # a folder, not a file
    t3 = topo(3)                                    # CRLF failure lines -> 4
    (t3 / "_failures.jsonl").write_bytes(b'{"a":1}\r\n{"a":2}\r\n{"a":3}\r\n{"a":4}\r\n')
    (root / "not_a_topo").mkdir()                   # ignored
    (root / "topo_00009.txt").write_text("x")       # a file, not a folder: ignored
    got = LOAD.raw_finished_counts(str(root))
    assert got == {0: 3, 1: 3, 2: 0, 3: 4}
    assert list(got) == sorted(got)
    (root / "topo_3").mkdir()                       # two folders giving topology 3
    with pytest.raises(LOAD.LoadError):
        LOAD.raw_finished_counts(str(root))
    with pytest.raises(LOAD.LoadError):
        LOAD.raw_finished_counts(str(tmp_path / "no_such_task"))


def test_plan_false_never_calls_the_block10_readers(tmp_path, monkeypatch, capsys):
    """P8, in process: with false neither dispatch_size nor raw_finished_counts is
    called; with true each is called once per task."""
    import bank_plan
    import bank_plan_load as LOAD
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    calls = []
    real_d, real_f = LOAD.dispatch_size, LOAD.raw_finished_counts
    monkeypatch.setattr(LOAD, "dispatch_size", lambda d: calls.append(("d", d)) or real_d(d))
    monkeypatch.setattr(LOAD, "raw_finished_counts", lambda d: calls.append(("f", d)) or real_f(d))
    for flag, n_calls in ((False, 0), (True, 6)):
        prof = H.write_profile(fx, os.path.join(root, "p_%s.json" % flag),
                               drop_partial_topologies=flag)
        del calls[:]
        rc = bank_plan.main(["--profile", prof, "--out", os.path.join(root, "plan_%s.json" % flag)])
        out = capsys.readouterr().out
        assert rc == 0 and H.last_line(out) == "BANK PLAN OK", out
        assert len(calls) == n_calls, (flag, calls)


def test_model_partial_and_drop_are_pure(tmp_path):
    """P3, P4 at the model level: f < K partial, f = K complete, f > K refused;
    drop_topologies removes from Top(t) and E(t) and recounts, input unchanged."""
    import bank_plan_load as LOAD
    import bank_plan_model as MODEL
    task = LOAD.TaskInputs(name="c_v1/s", campaign_dir="/x", out_dir="/y", contract={},
                           contract_sig="0" * 16, seeds={}, n_topos=4, n_iters=9,
                           topo_iters={0: [0, 1, 2], 1: [0, 1], 4: [3, 5, 7, 9]},
                           empty_topos=(2,))
    rows = MODEL.partial_topologies(task, {0: 6, 1: 5, 2: 0, 4: 6}, 6)
    assert rows == [{"topo": 1, "finished": 5, "expected": 6},
                    {"topo": 2, "finished": 0, "expected": 6}]
    with pytest.raises(MODEL.PlanRefused) as ei:
        MODEL.partial_topologies(task, {0: 7, 1: 5, 2: 0, 4: 6}, 6)
    assert "c_v1/s" in str(ei.value) and "7" in str(ei.value)
    with pytest.raises(MODEL.PlanRefused) as ei:       # raw folders != Top u E
        MODEL.partial_topologies(task, {0: 6, 1: 6, 4: 6, 8: 6}, 6)
    assert "8" in str(ei.value) and "2" in str(ei.value)
    before = json.dumps({str(k): v for k, v in task.topo_iters.items()})
    red = MODEL.drop_topologies(task, [1, 2])
    assert red.topo_iters == {0: [0, 1, 2], 4: [3, 5, 7, 9]}
    assert tuple(red.empty_topos) == ()
    assert (red.n_iters, red.n_topos) == (7, 2)
    assert json.dumps({str(k): v for k, v in task.topo_iters.items()}) == before
    assert (task.n_iters, task.n_topos, tuple(task.empty_topos)) == (9, 4, (2,))
    with pytest.raises(MODEL.PlanRefused):
        MODEL.drop_topologies(task, [3])


# --------------------------------------------------------------- the plan

def _seed_json(campaign_dir):
    """s(t) per SPEC 2.5: job_args keys containing "seed"; seed_master dropped where
    _resolved_seed_master is recorded. Canonical JSON (only the ORDER of two such
    strings is used below)."""
    with open(os.path.join(campaign_dir, "job_args.json"), "r", encoding="utf-8") as fh:
        ja = json.load(fh)
    s = {k: v for k, v in ja.items() if "seed" in k}
    if "_resolved_seed_master" in s:
        s.pop("seed_master", None)
    return H.canonical_json(s)


def build_scenario(root, gate=None):
    """The fixture with K = 6 and, per task, these raw counts f(t, k):

      T0 (v1, twin): topo 0: 5 json + 1 line = 6; topo 1: 4 json = 4 PARTIAL;
                     topo 2: 2 json + 4 lines (a blank and a whitespace-only line
                     between them, no final newline) = 6
      T1 (v2, twin): topo 0: 5 + 1 = 6; topo 1: 3 + 3 = 6; topo 2: 2 json + 3 lines
                     + decoys (iter_00009.json.tmp, iter_y.json, topology_meta.json,
                     a folder iter_00008.json) = 5 PARTIAL (6 if a decoy counted)
      T2 (v5):       topo 0: 6 json = 6; topo 1: 2 json + 4 CRLF lines = 6;
                     topo 2: 3 json + 2 lines = 5 PARTIAL; topo 3 (empty in both
                     trees, E(t)): topology_meta.json only = 0 PARTIAL
    """
    fx = H.write_fixture(root, extra_topos={2: {3: []}}, gate=gate)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.write_sidecars(fx, 0, 0, [0, 1, 2, 3, 4])
    H.write_failures(fx, 0, 0, '{"iter": 99}\n')
    H.write_sidecars(fx, 0, 1, [0, 1, 2, 3])
    H.write_sidecars(fx, 0, 2, [5, 6])
    H.write_failures(fx, 0, 2, '{"a": 1}\n\n   \n{"a": 2}\n{"a": 3}\n{"a": 4}')
    H.write_sidecars(fx, 1, 0, [0, 1, 2, 3, 4])
    H.write_failures(fx, 1, 0, '{"a": 1}\n')
    H.write_sidecars(fx, 1, 1, [0, 1, 2])
    H.write_failures(fx, 1, 1, '{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    H.write_sidecars(fx, 1, 2, [0, 1])
    H.write_failures(fx, 1, 2, '{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    d = H.raw_topo_dir(fx, 1, 2)
    for name in ("iter_00009.json.tmp", "iter_y.json", "topology_meta.json"):
        with open(os.path.join(d, name), "w") as fh:
            fh.write("{}")
    os.makedirs(os.path.join(d, "iter_00008.json"))
    H.write_sidecars(fx, 2, 0, [0, 1, 2, 3, 4, 5])
    H.write_sidecars(fx, 2, 1, [0, 1])
    H.write_failures(fx, 2, 1, '{"a": 1}\r\n{"a": 2}\r\n{"a": 3}\r\n{"a": 4}\r\n')
    H.write_sidecars(fx, 2, 2, [0, 1, 2])
    H.write_failures(fx, 2, 2, '{"a": 1}\n{"a": 2}\n')
    with open(os.path.join(H.raw_topo_dir(fx, 2, 3), "topology_meta.json"), "w") as fh:
        fh.write("{}")
    partial = {0: {1: 4}, 1: {2: 5}, 2: {2: 5, 3: 0}}          # task -> {k: f}
    return fx, partial


def expected_plan(fx, partial):
    """Hand computation of SPEC 2.5 on the kept topologies only."""
    tasks = []
    for n, t in enumerate(fx["tasks"]):
        its = H.detection_iters(fx, n)
        drop = set(partial.get(n, {}))
        top = {k: v for k, v in sorted(its.items()) if v and k not in drop}
        empty = [k for k, v in sorted(its.items()) if not v and k not in drop]
        dropped_sims = sum(len(its[k]) for k in drop)
        tasks.append({"name": "%s/%s" % (t["campaign"], t["sweep"]), "top": top, "empty": empty,
                      "seed": _seed_json(t["campaign_dir"]), "n_iters": sum(map(len, top.values())),
                      "n_dropped": dropped_sims, "full_iters": sum(map(len, its.values()))})
    keys = sorted({(t["seed"], k) for t in tasks for k in t["top"]})
    donor = {key: n for n, key in enumerate(keys)}
    for t in tasks:
        t["donors"] = {k: donor[(t["seed"], k)] for k in t["top"]}
        t["positions_sha256"] = H.sha256_bytes(
            H.canonical_json([[k, list(v)] for k, v in sorted(t["top"].items())]).encode("ascii"))
    return tasks, len(keys)


def run_plan(fx, root, name="plan.json", **overrides):
    prof = H.write_profile(fx, os.path.join(root, name + ".profile.json"), **overrides)
    out = os.path.join(root, name)
    rc, so, se = H.run_py(PLAN, "--profile", prof, "--out", out)
    return rc, so, se, out


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("b10_main"))
    fx, partial = build_scenario(root)
    rc, so, se, out = run_plan(fx, root, "plan_true.json", drop_partial_topologies=True)
    rc2, so2, se2, out2 = run_plan(fx, root, "plan_true_again.json", drop_partial_topologies=True)
    rc3, so3, se3, out3 = run_plan(fx, root, "plan_false.json", drop_partial_topologies=False)
    return {"root": root, "fx": fx, "partial": partial, "true": (rc, so, se, out),
            "again": (rc2, so2, se2, out2), "false": (rc3, so3, se3, out3)}


def test_plan_true_known_answer(scenario):
    """P3-P6: exactly the partial topologies are dropped; per-task fields and the
    plan's donors, counts, positions and layout are those of the kept ones."""
    rc, so, se, out = scenario["true"]
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    plan = H.load_plan_json(out)
    fx, partial = scenario["fx"], scenario["partial"]
    exp, n_donors = expected_plan(fx, partial)
    assert plan["drop_partial_topologies"] is True
    assert plan["profile"]["drop_partial_topologies"] is True
    assert plan["n_partial_topos"] == 4 and plan["n_partial_tasks"] == 3
    assert plan["n_dropped_partial"] == 4 + 2 + 3 == 9
    assert plan["n_donors"] == n_donors == 5
    assert plan["n_rows"] == sum(t["n_iters"] for t in exp) == 23
    assert plan["n_shards"] == 3
    shard_no = 0
    for n, (pt, et) in enumerate(zip(plan["tasks"], exp)):
        assert pt["name"] == et["name"]
        want_partial = [{"topo": k, "finished": f, "expected": K}
                        for k, f in sorted(partial[n].items())]
        assert pt["partial_topos"] == want_partial
        assert pt["n_dropped_partial"] == et["n_dropped"]
        assert pt["n_iters"] == et["n_iters"]
        assert pt["n_topos"] == len(et["top"]) + len(et["empty"])
        assert pt["topo_counts"] == {str(k): len(v) for k, v in et["top"].items()}
        assert pt["empty_topos"] == et["empty"]
        assert pt["positions_sha256"] == et["positions_sha256"]
        assert pt["donors"] == {str(k): v for k, v in et["donors"].items()}
        # layout (2.5): c(t) = max(1, ceil(n / R)) chunks, sizes differing by <= 1
        R = plan["profile"]["rows_per_shard"]
        c = max(1, math.ceil(et["n_iters"] / R))
        assert len(pt["shards"]) == c
        assert [s["shard"] for s in pt["shards"]] == list(range(shard_no, shard_no + c))
        shard_no += c
        assert pt["shards"][0]["start"] == 0 and pt["shards"][-1]["stop"] == et["n_iters"]
    assert plan["estimates"]["n_rows"] == 23
    assert plan["estimates"]["raw_bytes"] == 23 * plan["window"]["W"] * 4


def test_plan_true_sample_and_share_on_kept_topologies(scenario):
    """P4: the sample, the twin check and the kept share see the kept topologies
    only; a donor key whose set of tasks is unchanged keeps its draw."""
    fx, partial = scenario["fx"], scenario["partial"]
    pt = H.load_plan_json(scenario["true"][3])
    pf = H.load_plan_json(scenario["false"][3])
    exp, _ = expected_plan(fx, partial)
    rows_t = {(r[0], r[1]): r for r in pt["sample"]["rows"]}
    rows_f = {(r[0], r[1]): r for r in pf["sample"]["rows"]}
    kept_keys = {(t["name"], k) for t in exp for k in t["top"]}
    assert set(rows_t) == kept_keys                      # no dropped topology sampled
    names = [t["name"] for t in exp]
    # draws unchanged where the group's tasks having k are unchanged (SPEC 2.5 Sample)
    same = [(names[0], 0), (names[1], 0), (names[1], 1), (names[2], 0), (names[2], 1)]
    for key in same:
        assert rows_t[key][2:5] == rows_f[key][2:5], (key, rows_t[key], rows_f[key])
    i_star = rows_t[(names[0], 2)][2]
    assert i_star in (5, 6)
    # kept share, hand-computed: each draw stands for its topology's n(t, k) rows
    num = den = 0
    task_index = {nm: n for n, nm in enumerate(names)}
    for (name, k), r in rows_t.items():
        n_tk = len(exp[task_index[name]]["top"][k])
        kept = H.n_spikes(fx, task_index[name], k, r[2]) >= 18
        assert r[4] == kept
        num += n_tk * kept
        den += n_tk
    share = pt["sample"]["kept_share"]
    assert den == 23
    np.testing.assert_allclose(share["p_hat"], num / den, rtol=1e-12)   # exact ratio of ints
    assert share["n_topologies"] == 6 and share["n_donors"] == 5
    # twin check: T0 keeps {0, 2}, T1 keeps {0, 1}: one shared topology
    tw = pt["sample"]["twin_check"]
    assert len(tw) == 1 and tw[0]["shared_topologies"] == 1


def test_plan_true_printout(scenario):
    """P7 and the donors line of 6.5 (sum_t |Top(t)| after the drop)."""
    so = scenario["true"][1]
    exp, _ = expected_plan(scenario["fx"], scenario["partial"])
    names = [t["name"] for t in exp]
    lines = H.lines_of(so)
    head = ("[plan] partial topologies (drop_partial_topologies): 4 dropped in 3 task(s), "
            "9 simulations; bank_plan.json tasks[].partial_topos")
    assert head in lines
    i = lines.index(head)
    assert lines[i + 1:i + 5] == ["[plan]   %s topo 1: 4 of 6" % names[0],
                                  "[plan]   %s topo 2: 5 of 6" % names[1],
                                  "[plan]   %s topo 2: 5 of 6" % names[2],
                                  "[plan]   %s topo 3: 0 of 6" % names[2]]
    assert any(ln.startswith("[plan] donors") and ": 5 (from 6 topologies" in ln for ln in lines)
    assert not any("not checked" in ln for ln in lines)


def test_plan_true_deterministic(scenario):
    """Two runs with the same profile give the same plan (created, git and
    free_space pinned)."""
    a = open(scenario["true"][3], encoding="ascii").read()
    b = open(scenario["again"][3], encoding="ascii").read()
    assert H.pinned(a) == H.pinned(b)


def test_plan_false_ignores_block10_and_prints_not_checked(scenario):
    """P8: with false the partial topologies are kept and the line is printed."""
    rc, so, se, out = scenario["false"]
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    assert "[plan] partial topologies: not checked (drop_partial_topologies false)" in H.lines_of(so)
    plan = H.load_plan_json(out)
    assert plan["n_rows"] == sum(t["full_iters"] for t in expected_plan(scenario["fx"], {})[0])
    assert "drop_partial_topologies" not in plan and "n_partial_topos" not in plan
    assert all("partial_topos" not in t for t in plan["tasks"])


def test_plan_false_equals_parent_commit_byte_for_byte(scenario, tmp_path):
    """P8 (regression): with false the plan's JSON is the parent commit's (ae71f33,
    block 10's parent) on the same fixture, created / git / free_space pinned; and
    the pre-wave plan (4955e8d) differs only by the two new profile keys."""
    fx = scenario["fx"]
    parent = H.extract_commit(H.PARENT_OF_BLOCK10, str(tmp_path / "parent"))
    base = H.extract_commit(H.BASE, str(tmp_path / "base"))
    if parent is None or base is None:
        pytest.skip("git history not available for git archive")
    prof = H.write_profile(fx, str(tmp_path / "prof_false.json"), drop_partial_topologies=False)
    prof_old = H.write_profile(fx, str(tmp_path / "prof_old.json"), drop_new_keys=True)
    rc, so, se = H.run_py(os.path.join(parent, "export_windows", "bank_plan.py"),
                          "--profile", prof, "--out", str(tmp_path / "parent_plan.json"))
    assert rc == 0, so + se
    rcb, sob, seb = H.run_py(os.path.join(base, "export_windows", "bank_plan.py"),
                             "--profile", prof_old, "--out", str(tmp_path / "base_plan.json"))
    assert rcb == 0, sob + seb
    head = open(scenario["false"][3], encoding="ascii").read()
    par = open(str(tmp_path / "parent_plan.json"), encoding="ascii").read()
    assert H.pinned(head) == H.pinned(par)
    h = json.loads(H.pinned(head))
    for key in ("cohort_map", "drop_partial_topologies"):
        h["profile"].pop(key)
    b = json.loads(H.pinned(open(str(tmp_path / "base_plan.json"), encoding="ascii").read()))
    assert h == b


def test_plan_true_without_partials_differs_only_by_block10_fields(tmp_path):
    """With every topology complete, true and false give the same plan but for the
    Block 10 fields (P4: nothing dropped, nothing else changes)."""
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    rc, so, se, out_t = run_plan(fx, root, "t.json", drop_partial_topologies=True)
    assert rc == 0, so + se
    rc, so2, se2, out_f = run_plan(fx, root, "f.json", drop_partial_topologies=False)
    assert rc == 0, so2 + se2
    t = json.loads(H.pinned(open(out_t, encoding="ascii").read()))
    f = json.loads(H.pinned(open(out_f, encoding="ascii").read()))
    assert t.pop("drop_partial_topologies") is True
    assert (t.pop("n_partial_topos"), t.pop("n_partial_tasks"), t.pop("n_dropped_partial")) == (0, 0, 0)
    for task in t["tasks"]:
        assert task.pop("partial_topos") == [] and task.pop("n_dropped_partial") == 0
    t["profile"]["drop_partial_topologies"] = False
    assert t == f
    assert ("[plan] partial topologies (drop_partial_topologies): 0 dropped in 0 task(s), "
            "0 simulations; bank_plan.json tasks[].partial_topos") in H.lines_of(so)


def test_plan_false_does_not_read_dispatch_keys(tmp_path):
    """P8: with false a job_args.json without the dispatch keys (and no raw sidecar
    at all) does not refuse; with true it refuses naming the task and the key."""
    root = str(tmp_path)
    fx = H.write_fixture(root)
    rc, so, se, _ = run_plan(fx, root, "f.json", drop_partial_topologies=False)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    H.edit_json(os.path.join(fx["tasks"][1]["campaign_dir"], "job_args.json"),
                n_params_per_worker=H.DELETE)
    rc, so, se, out = run_plan(fx, root, "t.json", drop_partial_topologies=True)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert "n_params_per_worker" in so and "campaign_cadex_hhgap_v2/sweep_intel_task0000" in so
    assert not os.path.exists(out)


@pytest.mark.parametrize("value", [True, 6.0, "6", 0])
def test_plan_true_refuses_bad_dispatch_values(tmp_path, value):
    """P1 through the entry point: a bool, a float, a string, 0 refuse (no plan)."""
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    H.edit_json(os.path.join(fx["tasks"][0]["campaign_dir"], "job_args.json"),
                _resolved_n_workers=value)
    rc, so, se, out = run_plan(fx, root, "t.json", drop_partial_topologies=True)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert "_resolved_n_workers" in so
    assert not os.path.exists(out)


def test_plan_true_refusals(tmp_path):
    """P3, P9: f > K refuses naming task and topology; a raw-only topology folder
    refuses naming both sides; two refused tasks are listed as in 6.2; an empty
    bank after the drop refuses. No plan is written."""
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    # f > K in task 0, topology 0 (5 sidecars + 2 lines = 7)
    H.write_failures(fx, 0, 0, '{"a": 1}\n{"a": 2}\n')
    rc, so, se, out = run_plan(fx, root, "over.json", drop_partial_topologies=True)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert "campaign_cadex_hhgap_v1/sweep_intel_task0000" in so and "topology 0 holds 7" in so
    assert not os.path.exists(out)
    # and a raw-only topology folder in task 2: two tasks refused, each named
    os.makedirs(H.raw_topo_dir(fx, 2, 7))
    rc, so, se, out = run_plan(fx, root, "two.json", drop_partial_topologies=True)
    assert rc == 1, so + se
    assert H.last_line(so) == "REFUSED: 2 of 3 tasks refused (each named above)"
    refused = [ln for ln in H.lines_of(so) if ln.startswith("[plan] refused")]
    assert len(refused) == 2 and "[7]" in refused[1]
    # with false neither is read: OK
    rc, so, se, out = run_plan(fx, root, "false.json", drop_partial_topologies=False)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se


def test_plan_true_refuses_an_empty_bank(tmp_path):
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)            # no sidecar, no failure: f = 0 everywhere
    rc, so, se, out = run_plan(fx, root, "empty.json", drop_partial_topologies=True)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert not os.path.exists(out)


def test_plan_prints_the_first_ten_partial_topologies(tmp_path):
    """P7: eleven partial topologies (empty ones, E(t)) print the head line and ten
    lines; dropped from E(t), they count 0 simulations."""
    root = str(tmp_path)
    fx = H.write_fixture(root, extra_topos={0: {k: [] for k in range(10, 21)}})
    H.set_dispatch(fx, K_WORKERS, K_PER)
    for n in range(3):
        for k, its in H.detection_iters(fx, n).items():
            if its:
                H.write_sidecars(fx, n, k, its)
                if K - len(its):
                    H.write_failures(fx, n, k, '{"a": 0}\n' * (K - len(its)))
    rc, so, se, out = run_plan(fx, root, "p.json", drop_partial_topologies=True)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    lines = H.lines_of(so)
    head = [ln for ln in lines if ln.startswith("[plan] partial topologies")]
    assert head == ["[plan] partial topologies (drop_partial_topologies): 11 dropped in 1 "
                    "task(s), 0 simulations; bank_plan.json tasks[].partial_topos"]
    item = [ln for ln in lines if ln.startswith("[plan]   campaign") and " topo " in ln]
    assert len(item) == 10
    plan = H.load_plan_json(out)
    assert plan["tasks"][0]["empty_topos"] == [] and plan["tasks"][0]["n_dropped_partial"] == 0
    assert [p["topo"] for p in plan["tasks"][0]["partial_topos"]] == list(range(10, 21))


# --------------------------------------------------------------- export and gate

def _export_all(plan_path, bank_root, n_tasks):
    res = []
    for n in range(n_tasks):
        rc, so, se = H.run_py(TASK, "--plan", plan_path, "--task-index", n, "--bank-root", bank_root)
        res.append((rc, H.last_line(so), so + se))
    rc, so, se = H.run_py(GATE, "--plan", plan_path, "--bank-root", bank_root)
    return res, (rc, H.last_line(so), so + se)


def test_export_and_gate_after_the_drop(scenario, tmp_path):
    """P10 and boundary B1: every kept row comes from a kept (k, i) with N >= 18;
    every such position is in the bank once; read = the plan's post-drop n_iters;
    the gate passes; a second bank root gives identical arrays."""
    fx, partial = scenario["fx"], scenario["partial"]
    plan_path = scenario["true"][3]
    plan = H.load_plan_json(plan_path)
    exp, _ = expected_plan(fx, partial)
    lb = H.latent_bank()
    banks = []
    for b in ("bank_a", "bank_b"):
        bank = str(tmp_path / b)
        tasks, gate = _export_all(plan_path, bank, 3)
        for n, (rc, last, text) in enumerate(tasks):
            assert rc == 0 and last == "TASK %d OK" % n, text
        assert gate[0] == 0 and gate[1].startswith("wrote ") and "BANK_RECORD.json" in gate[1], gate[2]
        banks.append(bank)
    total_kept = 0
    for n, et in enumerate(exp):
        want = {(k, i) for k, its in et["top"].items() for i in its if H.n_spikes(fx, n, k, i) >= 18}
        with open(os.path.join(banks[0], "tasks", "task_%04d.json" % n), encoding="ascii") as fh:
            marker = json.load(fh)
        assert marker["totals"]["read"] == et["n_iters"] == plan["tasks"][n]["n_iters"]
        assert marker["totals"]["kept"] == len(want)
        assert marker["totals"]["dropped_floor"] == et["n_iters"] - len(want)
        k_of = {v: k for k, v in et["donors"].items()}
        got = set()
        for rec in marker["shards"]:
            if rec["file"] is None:
                continue
            with np.load(os.path.join(banks[0], rec["file"])) as z:
                a = {key: z[key] for key in z.files}
            with np.load(os.path.join(banks[1], rec["file"])) as z2:
                for key in z2.files:                       # determinism across two runs
                    np.testing.assert_array_equal(z2[key], a[key])
            assert set(a) == set(lb.ROW_ARRAYS)            # the simulated shard: no rep_group
            assert np.all(a["well"] == n)
            for d, i, rid, x in zip(a["donor"], a["window_idx"], a["realisation_id"], a["x"]):
                k = k_of[int(d)]
                assert (k, int(i)) in want
                assert (k, int(i)) not in got
                got.add((k, int(i)))
                assert int(rid) == H.seed_run(fx, n, k, int(i))
                # count conservation (2.1): x sums to N (rtol 1e-6, float32 x)
                np.testing.assert_allclose(np.sum(x, dtype=np.float64),
                                           H.n_spikes(fx, n, k, int(i)), rtol=1e-6)
            for k in partial[n]:                           # nothing of a dropped topology
                assert all(kk != k for kk, _ in got)
        assert got == want
        total_kept += len(want)
    assert total_kept == 17
    shards = sorted(os.path.join(banks[0], f) for f in os.listdir(banks[0])
                    if f.startswith("shard_") and f.endswith(".npz"))
    merged, side = lb.concat_shards(shards)
    assert merged["x"].shape == (17, plan["window"]["W"])
    assert np.all((merged["theta"] >= 0) & (merged["theta"] <= 1))
    with open(os.path.join(banks[0], "BANK_RECORD.json"), encoding="ascii") as fh:
        record = json.load(fh)
    # the gate compares with the plan's post-drop counts (7.3 unchanged)
    assert record["counts"]["rows_read"] == 23 and record["counts"]["rows_kept"] == 17
    assert record["counts"]["n_donors"] == 5


def test_export_task_left_without_topologies(tmp_path):
    """A task whose every topology is partial keeps the plan going (only an EMPTY
    BANK refuses): the plan gives it no row and no donor, its array element writes
    a marker with read 0, and the gate passes (layout: c(t) = max(1, ...) = 1)."""
    root = str(tmp_path)
    fx = H.write_fixture(root)
    H.set_dispatch(fx, K_WORKERS, K_PER)
    H.complete_all(fx, K)
    for k, its in H.detection_iters(fx, 2).items():     # task 2: every topology partial
        for i in its:
            os.remove(os.path.join(H.raw_topo_dir(fx, 2, k), "iter_%05d.json" % i))
        fp = os.path.join(H.raw_topo_dir(fx, 2, k), "_failures.jsonl")
        if os.path.exists(fp):
            os.remove(fp)
    rc, so, se, out = run_plan(fx, root, "p.json", drop_partial_topologies=True)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    plan = H.load_plan_json(out)
    t2 = plan["tasks"][2]
    assert t2["n_iters"] == 0 and t2["donors"] == {} and t2["topo_counts"] == {}
    assert t2["n_dropped_partial"] == 11
    assert plan["n_donors"] == 3
    tasks, gate = _export_all(out, str(tmp_path / "bank"), 3)
    for n, (rc, last, text) in enumerate(tasks):
        assert rc == 0 and last == "TASK %d OK" % n, text
    assert gate[0] == 0, gate[2]
    assert not os.path.exists(str(tmp_path / "bank" / ("shard_%04d.npz" % t2["shards"][0]["shard"])))


def test_export_refuses_a_listed_topology_absent_from_its_tree(tmp_path):
    """P10: a partial topology the plan lists, renamed after the plan in both trees
    (counts unchanged), makes the task fail naming it; the gate then refuses."""
    root = str(tmp_path)
    fx, partial = build_scenario(root)
    rc, so, se, out = run_plan(fx, root, "p.json", drop_partial_topologies=True)
    assert rc == 0, so + se
    for tree in ("campaign_dir", "out_dir"):
        base = fx["tasks"][0][tree]
        os.rename(os.path.join(base, "topo_00001"), os.path.join(base, "topo_00009"))
    bank = str(tmp_path / "bank")
    rc, so, se = H.run_py(TASK, "--plan", out, "--task-index", 0, "--bank-root", bank)
    assert rc == 1 and H.last_line(so).startswith("FAILED:"), so + se
    assert "topology 1 " in H.last_line(so)            # the listed topology is named
    for n in (1, 2):
        rc, so, se = H.run_py(TASK, "--plan", out, "--task-index", n, "--bank-root", bank)
        assert rc == 0, so + se
    rc, so, se = H.run_py(GATE, "--plan", out, "--bank-root", bank)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert not os.path.exists(os.path.join(bank, "BANK_RECORD.json"))


def test_drop_composes_with_the_gate_axis(tmp_path):
    """Swappability: the contact-gate fixture (D-107, topology_axes with stoa_gate_p)
    with the drop: the kept rows' last theta column is (g(t, k) - 0.2) / 0.8."""
    root = str(tmp_path)
    fx, partial = build_scenario(root, gate=(0.2, 1.0))
    rc, so, se, out = run_plan(fx, root, "p.json", drop_partial_topologies=True)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    plan = H.load_plan_json(out)
    assert plan["label"]["param_names"][-1] == "stoa_gate_p"
    exp, _ = expected_plan(fx, partial)
    bank = str(tmp_path / "bank")
    tasks, gate = _export_all(out, bank, 3)
    assert all(rc == 0 for rc, _, _ in tasks), tasks
    assert gate[0] == 0, gate[2]
    for n, et in enumerate(exp):
        k_of = {v: k for k, v in et["donors"].items()}
        f = os.path.join(bank, "shard_%04d.npz" % plan["tasks"][n]["shards"][0]["shard"])
        with np.load(f) as z:
            for d, i, th in zip(z["donor"], z["window_idx"], z["theta"]):
                k = k_of[int(d)]
                raw = os.path.join(H.raw_topo_dir(fx, n, k), "iter_%05d.npz" % int(i))
                with np.load(raw) as r:                    # g(t, k) read from the raw file
                    g = float(r["stoa_gate_p"])
                # float32 storage of u: atol 1e-7 (Block 7.1's tolerance)
                np.testing.assert_allclose(th[-1], (g - 0.2) / 0.8, atol=1e-7)
