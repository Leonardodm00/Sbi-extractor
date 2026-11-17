"""End to end through the real entry points (tester step 6), 2026-11-17_1605, commit
5b61d6c: `launch_bank.sh plan | test | array | gate | real` (fake qsub, SPEC sec. 7),
the array element `bank_task.py` and the gate `bank_gate.py` run directly as the job
scripts would, on one fixture that exercises every change in scope at once:

  - Block 10 on: task 0 wholly partial (plan n_iters 0), tasks 1 and 2 complete;
  - Block 7.4 amendment 2: the test on task 0 reads 0 simulations, so `array` with
    WALLTIME=auto refuses naming TEST_INDEX=1; the test on task 1 then sizes it;
  - Block 6.1 amendment 2 / Block 9: an absolute cohort_map that leaves well 1 out;
  - Block 8 amendments 2, 3: well 3 holds only traces shorter than one window.

The whole sequence runs twice on the same inputs (the first run's outputs moved
aside); the outputs must exist, be finite, have the spec's shapes, dtypes and units,
and be identical between the runs (bit for bit for every array; every sidecar's
contract fields and digest; the plan with its clock, checkout and disk fields pinned).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import _bank_helpers as H
from test_spec_block7_launcher import fake_qsub, qsub_calls, ABORT

LAUNCH = os.path.join(H.EXPORT, "launch_bank.sh")
TASK_PY = os.path.join(H.EXPORT, "bank_task.py")
GATE_PY = os.path.join(H.EXPORT, "bank_gate.py")


def _launch(mode, prof, qsub, log, **over):
    env = dict(os.environ, PROFILE=prof, QSUB=qsub, FAKE_QSUB_LOG=log,
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PIPELINE_JOBS_DIR", "WORK_ROOT",
              "CHAIN_NAME", "AVOID_FILE", "AVOID_HOLD_S"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in over.items()})
    r = subprocess.run(["bash", LAUNCH, mode], env=env, capture_output=True, text=True,
                       timeout=600)
    return r.returncode, r.stdout, r.stderr


def _build_inputs(root):
    H.sbi_hpc_dir()
    fx = H.write_fixture(os.path.join(root, "fx"))
    for n in range(3):                                   # Block 10: task 0 wholly partial
        H.set_dispatch(fx, 1, 100 if n == 0 else 1, tasks=[n])
        if n != 0:
            for k, its in H.detection_iters(fx, n).items():
                H.write_sidecars(fx, n, k, its[:1])
    import cohort_manifest as CM                         # Block 8: well 3 too short
    with open(fx["manifest"], "r", encoding="ascii") as fh:
        man = json.load(fh)
    for s in range(9):
        p = os.path.join(man["extract_root"], man["wells"][3]["out_dir"],
                         "trace_subregion_%02d.npz" % s)
        with np.load(p) as z:
            a = {k: np.array(z[k]) for k in z.files}
        a["ifr_trace"], a["T_rec"] = a["ifr_trace"][:10000], np.float64(100.0)
        np.savez_compressed(p, **a)
    man["wells"][3]["T_rec"] = 100.0
    for k in [k for k in man if k.startswith("_")]:
        man.pop(k)
    fx["digests"]["manifest"] = CM.write_manifest(man, fx["manifest"])
    cmap = os.path.join(root, "maps", "cohort_map.csv")  # Block 9: well 1 left out
    os.makedirs(os.path.dirname(cmap))
    with open(cmap, "w", encoding="ascii") as fh:
        fh.write("out_dir,donor,rep_group,include,note\n")
        for wi, w in enumerate(man["wells"]):
            fh.write('%s,cond_%s,grp_%d,%s,"well %d, fixture"\n'
                     % (w["out_dir"], "a" if wi < 2 else "b", wi, "no" if wi == 1 else "yes", wi))
    prof = H.write_profile(fx, os.path.join(root, "profile.json"), drop_partial_topologies=True,
                           cohort_map=os.path.abspath(cmap))
    return fx, prof, man, cmap


def _run_once(root, fx, prof, man, cmap):
    qsub = fake_qsub(root)
    log = os.path.join(root, "qsub.log")
    if os.path.exists(log):
        os.remove(log)
    rc, so, se = _launch("plan", prof, qsub, log)
    assert rc == 0 and "BANK PLAN OK" in so, so + se
    plan_path = os.path.join(fx["root"], "plan", "bank_plan.json")
    with open(plan_path, "r", encoding="ascii") as fh:
        plan = json.load(fh)
    assert [t["n_iters"] for t in plan["tasks"]] == [0, 10, 11]
    out = {"plan": H.pinned(open(plan_path, encoding="ascii").read())}
    # the test on task 0 (empty), its element, then the refused auto walltime
    rc, so, se = _launch("test", prof, qsub, log, TEST_INDEX=0)
    assert rc == 0, so + se
    rc, so, se = H.run_py(TASK_PY, "--plan", plan_path, "--task-index", 0)
    assert rc == 0 and H.last_line(so) == "TASK 0 OK", so + se
    rc, so, se = _launch("array", prof, qsub, log, TEST_INDEX=0, WALLTIME="auto")
    assert rc == 3 and H.last_line(so) == ABORT % (0, 1), so + se
    assert len(qsub_calls(log)) == 1
    # the test on task 1, then the array and its gate
    rc, so, se = _launch("test", prof, qsub, log, TEST_INDEX=1)
    assert rc == 0, so + se
    rc, so, se = H.run_py(TASK_PY, "--plan", plan_path, "--task-index", 1)
    assert rc == 0 and H.last_line(so) == "TASK 1 OK", so + se
    rc, so, se = _launch("array", prof, qsub, log, TEST_INDEX=1, WALLTIME="auto")
    assert rc == 0, so + se
    calls = qsub_calls(log)
    assert len(calls) == 4 and "-J 0-2%20" in calls[2] and "walltime=00:30:00" in calls[2]
    assert "depend=afterok:3000.fake" in calls[3] and calls[3].endswith("run_bank_gate.pbs")
    for i in range(3):                                   # the array's elements
        rc, so, se = H.run_py(TASK_PY, "--plan", plan_path, "--task-index", i)
        assert rc == 0 and H.last_line(so) == "TASK %d OK" % i, so + se
    rc, so, se = H.run_py(GATE_PY, "--plan", plan_path)
    assert rc == 0 and H.last_line(so).startswith("wrote ") and "BANK_RECORD.json" in so, so + se
    rc, so, se = _launch("gate", prof, qsub, log)
    assert rc == 0 and len(qsub_calls(log)) == 5, so + se
    rc, so, se = _launch("real", prof, qsub, log)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    out["real_stdout"] = [ln for ln in H.lines_of(so) if not ln.startswith(("[real] wrote", "[launch]"))]
    bank = os.path.join(fx["root"], "banks", "giulia_sim")
    real = os.path.join(fx["root"], "banks", "giulia_real")
    with open(os.path.join(bank, "BANK_RECORD.json"), "r", encoding="ascii") as fh:
        out["record"] = json.load(fh)
    out["sim"] = {}
    for p in sorted(glob.glob(os.path.join(bank, "shard_*.npz"))):
        with np.load(p) as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        with open(p[:-4] + ".json", "r", encoding="ascii") as fh:
            side = json.load(fh)
        out["sim"][os.path.basename(p)] = (arrays, side)
    with np.load(os.path.join(real, "shard_0000.npz")) as z:
        ra = {k: np.array(z[k]) for k in z.files}
    with open(os.path.join(real, "shard_0000.json"), "r", encoding="ascii") as fh:
        rs = json.load(fh)
    out["real"] = (ra, rs)
    out["markers"] = []
    for i in range(3):
        with open(os.path.join(bank, "tasks", "task_%04d.json" % i), "r", encoding="ascii") as fh:
            out["markers"].append(json.load(fh)["totals"])
    return out


def _check_outputs(out, man, cmap):
    lb = H.latent_bank()
    # simulated bank: shapes, dtypes, finiteness, unit box, the sidecar's units
    n_sim = 0
    for name, (a, side) in out["sim"].items():
        n = a["x"].shape[0]
        n_sim += n
        assert a["x"].shape == (n, 18000) and a["x"].dtype == np.float32
        assert a["theta"].shape == (n, 23) and a["theta"].dtype == np.float32
        assert np.all(np.isfinite(a["x"])) and np.all(a["x"] >= 0)
        assert np.all(np.isfinite(a["theta"])) and np.all((a["theta"] >= 0) & (a["theta"] <= 1))
        assert a["realisation_id"].dtype == np.uint64 and a["contaminated"].dtype == bool
        for k in ("cls", "donor", "well", "batch", "subregion", "window_idx"):
            assert a[k].dtype == np.int64 and a[k].shape == (n,), k
        assert not np.any(a["well"] == 0)              # task 0 dropped whole (Block 10)
        assert side["fs"] == 100.0 and side["w_size"] == 0.01 and side["T_win"] == 180.0
        assert side["W"] == 18000 and side["coord"] == "unit_box" and side["arm"] == "S"
        assert side["contract_digest"] == lb.contract_digest(side)
    rec = out["record"]
    assert n_sim > 0
    assert sum(m["read"] for m in out["markers"]) == 21
    assert sum(m["kept"] for m in out["markers"]) == n_sim
    # real bank: the map and the short well
    a, side = out["real"]
    n = a["x"].shape[0]
    assert a["x"].shape == (n, 18000) and a["x"].dtype == np.float32 and np.all(np.isfinite(a["x"]))
    assert a["theta"].shape == (n, 0) and a["rep_group"].dtype == np.int64
    assert sorted(set(a["well"].tolist())) == [0, 2]   # well 1 excluded, well 3 too short
    assert n == 9 + 27
    assert side["provenance"]["too_short"] == {"traces": 9,
                                               "wells_without_rows": [man["wells"][3]["out_dir"]]}
    assert side["latent_spec"]["cohort_map"]["excluded"] == [man["wells"][1]["out_dir"]]
    assert side["latent_spec"]["cohort_map"]["sha256"] == H.sha256_file(cmap)
    assert side["contract_digest"] == lb.contract_digest(side)
    head = [ln for ln in out["real_stdout"] if ln.startswith("[real] cohort map ")]
    assert head and head[0].startswith("[real] cohort map %s sha256 " % cmap)
    return rec


def _without_volatile(d):
    """Drop the clock, host and environment fields, and plan_sha256: the plan records its
    creation time and the free space (Block 6.5), so two plans of the same inputs have
    different digests by design; the plan's content is compared pinned instead."""
    if isinstance(d, dict):
        return {k: _without_volatile(v) for k, v in d.items()
                if k not in ("host", "built", "created", "started", "finished", "python",
                             "numpy", "time", "environment", "written", "git", "plan_sha256")}
    if isinstance(d, list):
        return [_without_volatile(v) for v in d]
    return d


def test_end_to_end_twice(tmp_path):
    root = str(tmp_path)
    fx, prof, man, cmap = _build_inputs(root)
    run1 = _run_once(root, fx, prof, man, cmap)
    _check_outputs(run1, man, cmap)
    for d in ("banks", "plan"):
        os.rename(os.path.join(fx["root"], d), os.path.join(fx["root"], d + "_run1"))
    run2 = _run_once(root, fx, prof, man, cmap)
    _check_outputs(run2, man, cmap)
    # identical results under the same inputs
    assert run1["plan"] == run2["plan"]
    assert run1["markers"] == run2["markers"]
    assert run1["real_stdout"] == run2["real_stdout"]
    assert sorted(run1["sim"]) == sorted(run2["sim"])
    for name in run1["sim"]:
        (a1, s1), (a2, s2) = run1["sim"][name], run2["sim"][name]
        assert sorted(a1) == sorted(a2)
        for k in a1:
            assert a1[k].dtype == a2[k].dtype
            np.testing.assert_array_equal(a1[k], a2[k])
        assert {k: v for k, v in s1.items() if k != "provenance"} == \
               {k: v for k, v in s2.items() if k != "provenance"}
    (r1, t1), (r2, t2) = run1["real"], run2["real"]
    for k in r1:
        np.testing.assert_array_equal(r1[k], r2[k])
    assert {k: v for k, v in t1.items() if k != "provenance"} == \
           {k: v for k, v in t2.items() if k != "provenance"}
    assert _without_volatile(t1["provenance"]) == _without_volatile(t2["provenance"])
    assert _without_volatile(run1["record"]) == _without_volatile(run2["record"])
