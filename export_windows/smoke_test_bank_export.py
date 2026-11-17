"""Smoke test for the Giulia window banks -- Blocks 7 and 8 in specs/SPEC.md
(B5.2-B5.5).

Covers Block 7.1 (one task's rows and shards), 7.2 (the array element), 7.3
(the gate and the bank record), 7.4 (the launcher and the job scripts, with a
fake qsub and a fake conda), 7.5 (the same under the Simulation-Based-Inference
pipeline's job protocol, D-113: the environment's own folder on a node without
conda, exit 96 and 94; it needs SBI_HPC_DIR at a checkout with
pipeline/jobs/stage_job.sh) and Block 8 (the real bank). Amendment 2 of
2026-11-17: WALLTIME=auto after a test task of 0 simulations (7.4, F3), the
traces shorter than one window and an out_dir without <class>/<root>/ (8, F4,
F5). Every check runs on
the synthetic record and cohort of bank_plan_fixture.py, written to a
temporary folder; nothing here reads the cluster's files or submits a job.
One optional check runs the joint stack's runner in --dry-run mode on the
fixture's banks, when torch and sbi are importable.

Run (from the Sbi-extractor root, in the sbi_export environment):
    export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc   # latent_bank, 9fcbc88 or later
    python3 export_windows/smoke_test_bank_export.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table; exit
code 1 if any check failed, errored or is TODO. The fixture's folders are
removed at exit.

Keep this file pure ASCII and LF only (hpc-python-compat).
"""
from __future__ import annotations

import atexit
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np
from numpy.testing import assert_allclose

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bank_export as EXP             # noqa: E402
import bank_gate as GATE              # noqa: E402
import bank_plan as PLAN              # noqa: E402
import bank_plan_fixture as FX        # noqa: E402
import bank_real as REAL              # noqa: E402
import bank_task as TASK              # noqa: E402
import cohort_manifest as CM          # noqa: E402  (the repo root; the fixture's manifest)
import pipeline_protocol_fixture as PPF  # noqa: E402  (the repo root; D-113)
import smoke_test_bank_plan as SP     # noqa: E402  (its helpers; its checks run in its own suite)

SEED = 0
REPORT_PACKAGES = ("numpy", "scipy")
W = 18000                               # 180 s at 0.01 s (D-041)
SCRIPTS = ("launch_bank.sh", "run_bank_task.pbs", "run_bank_gate.pbs")
SHARD_GLOB = "shard_[0-9][0-9][0-9][0-9].npz"


# --------------------------------------------------------------- helpers ---

class Exported(object):
    """A fresh fixture, its plan, and (by default) every task exported."""

    def __init__(self, export=True, extra_topos=None, gate=None, campaign_names=None,
                 **profile_overrides):
        self.f = SP.Fixture(extra_topos=extra_topos, gate=gate, campaign_names=campaign_names,
                            **profile_overrides)
        self.cfg = self.f.cfg()
        rc, out = SP.run_main(PLAN.main, ["--profile", self.f.profile])
        assert rc == 0, out[-1:]
        self.plan_path = self.cfg.plan_out
        self.plan = EXP.load_plan(self.plan_path)
        self.bank = self.cfg.bank_root
        if export:
            for i in range(len(self.plan["tasks"])):
                EXP.export_task(self.plan, i, self.bank, log=lambda *a: None)

    def marker(self, i):
        with open(EXP.marker_path(self.bank, i)) as fh:
            return json.load(fh)

    def close(self):
        self.f.close()


_SHARED = {}


def shared():
    """One exported fixture that no check mutates (the gate's record aside)."""
    if "e" not in _SHARED:
        e = Exported()
        atexit.register(e.close)
        _SHARED["e"] = e
    return _SHARED["e"]


def u_hand(plan, t, k, i, truth):
    """The label in the unit box, recomputed here from the fixture's truth."""
    tr = truth[(FX.task_name(t), k, i)]
    theta_a = np.concatenate([tr["theta"][15:37], [tr["conn_prob"]]])
    b = np.asarray(plan["label"]["bounds_declared"])
    return (theta_a - b[:, 0]) / (b[:, 1] - b[:, 0])


def kept_positions(t):
    return [(k, i) for k in sorted(t[4]) for i in t[4][k] if FX.spikes_of(t, k, i) >= 18]


def run_bash(args, env, cwd=HERE):
    r = subprocess.run(["bash"] + list(args), cwd=cwd, env=env, capture_output=True,
                       text=True, timeout=600)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()


def fake_tools(root):
    """A fake qsub that records its arguments, and a fake conda whose activate
    puts this interpreter's folder first on PATH (the job scripts' env block)."""
    b = os.path.join(root, "fakebin")
    base = os.path.join(root, "fakebase")
    os.makedirs(b)
    os.makedirs(os.path.join(base, "etc", "profile.d"))
    with open(os.path.join(b, "qsub"), "w") as fh:
        fh.write('#!/bin/bash\nlog="$FAKE_QSUB_LOG"\n'
                 'n=$(( $(cat "$log" 2>/dev/null | wc -l) + 1 ))\n'
                 'printf "%s\\n" "$*" >> "$log"\n'
                 'case " $* " in *" -J "*) echo "${n}000[].pbsserver01" ;; '
                 '*) echo "${n}000.pbsserver01" ;; esac\n')
    with open(os.path.join(b, "conda"), "w") as fh:
        fh.write('#!/bin/bash\nif [ "$1" = info ] && [ "$2" = --base ]; then echo "%s"; '
                 'exit 0; fi\necho "fake conda: $*" >&2\n' % base)
    with open(os.path.join(base, "etc", "profile.d", "conda.sh"), "w") as fh:
        fh.write('conda() {\n  if [ "$1" = activate ]; then\n'
                 '    [ "$2" = sbi_export ] || { echo "fake conda: no env $2" >&2; return 1; }\n'
                 '    export PATH="%s:$PATH"; export CONDA_DEFAULT_ENV="$2"; return 0\n'
                 '  fi\n  command conda "$@"\n}\n' % os.path.dirname(sys.executable))
    for name in ("qsub", "conda"):
        os.chmod(os.path.join(b, name), 0o755)
    return b


def launcher_env(profile, fb, log):
    """launch_bank.sh's environment as test_74_launcher_and_jobs builds it: the
    fixture's profile, the fake qsub logging to log, this interpreter first on PATH."""
    env = dict(os.environ, PROFILE=profile, QSUB=os.path.join(fb, "qsub"), FAKE_QSUB_LOG=log,
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "ENV_NAME"):
        env.pop(k, None)
    return env


def plan_with_empty_tasks(empty):
    """(fixture, plan): the default fixture planned with drop_partial_topologies true
    (Block 10), every topology of the tasks in `empty` partial, so that their plan
    n_iters is 0. Their job_args.json dispatch K = 1 x 100 per topology and none is
    finished; the other tasks dispatch K = 1 x 1 and each of their raw topologies holds
    one iter_<i>.json, so they keep every simulation."""
    f = SP.Fixture(drop_partial_topologies=True)
    try:
        for n, t in enumerate(FX.TASKS):
            SP.rewrite_job_args(f, t, _resolved_n_workers=1,
                                n_params_per_worker=100 if n in empty else 1)
            for k, iters in ([] if n in empty else sorted(t[4].items())):
                with open(os.path.join(f.dir, "sim", t[0], t[1], "topo_%05d" % k,
                                       "iter_%05d.json" % iters[0]), "w") as fh:
                    json.dump({"topo_idx": k, "iter_idx": iters[0]}, fh)
        rc, out = SP.run_main(PLAN.main, ["--profile", f.profile])
        assert rc == 0, out[-3:]
        plan = EXP.load_plan(f.cfg().plan_out)
        n_iters = [t["n_iters"] for t in plan["tasks"]]
        assert [n == 0 for n in n_iters] == [n in empty for n in range(len(FX.TASKS))], n_iters
    except Exception:                    # no folder left behind by a fixture that failed
        f.close()
        raise
    return f, plan


def real_well(wi):
    """(culture, out_dir) of the fixture's real well wi, as its manifest names them."""
    cls_name, root_name, well = FX.REAL_WELLS[wi][:3]
    return "%s__%s" % (root_name, well), "/".join((cls_name, root_name, well))


def resign_manifest(f, change):
    """Apply change(doc) to the fixture's cohort manifest, rewrite it with its .sha256
    (cohort_manifest.write_manifest) and the profile's digest prefix; the digest."""
    with open(f.fx["manifest"]) as fh:
        doc = json.load(fh)
    change(doc)
    digest = CM.write_manifest(doc, f.fx["manifest"])
    FX.write_profile(f.fx, f.profile, cohort_manifest_sha256_prefix=digest[:16])
    return digest


def cut_traces(f, wi, t_rec, subregions=range(9)):
    """The archives of well wi's subregions cut to their first t_rec seconds (Delta_t
    0.01 s), with T_rec t_rec, as an extractor writes a shorter recording."""
    cls_name, root_name, well = FX.REAL_WELLS[wi][:3]
    for s in subregions:
        p = os.path.join(f.dir, "real", cls_name, root_name, well, "trace_subregion_%02d.npz" % s)
        with np.load(p) as z:
            trace = np.asarray(z["ifr_trace"])
        SP.rewrite_npz(p, ifr_trace=trace[:int(round(t_rec / 0.01))], T_rec=np.float64(t_rec))


# ------------------------------------------------------- Block 7.1 checks ---

def test_71_rows_known_answer():
    e = shared()
    lb, _ln = EXP.stage1_modules()
    truth = e.f.fx["truth"]
    for idx, t in enumerate(FX.TASKS):
        pos = [(k, i) for k in sorted(t[4]) for i in t[4][k]]
        kept = kept_positions(t)          # by hand: N >= 18 detections in [0, 180) s
        tot = e.marker(idx)["totals"]
        assert (tot["read"], tot["kept"], tot["dropped_floor"]) == (
            len(pos), len(kept), len(pos) - len(kept)), (idx, tot)
        a, _side = lb.read_shard(os.path.join(e.bank, "shard_%04d.npz" % idx))
        donors = {int(k): d for k, d in e.plan["tasks"][idx]["donors"].items()}
        inv = {d: k for k, d in donors.items()}
        got = [(inv[int(d)], int(w)) for d, w in zip(a["donor"], a["window_idx"])]
        assert got == kept, (idx, got, kept)
        n = len(kept)
        assert a["x"].shape == (n, W) and a["x"].dtype == np.float32
        assert np.all(a["well"] == idx) and np.all(a["batch"] == [1, 2, 5][idx])
        assert a["realisation_id"].dtype == np.uint64
        assert [int(r) for r in a["realisation_id"]] == [FX.seed_run_of(t, k, i) for k, i in kept]
        assert np.all(a["cls"] == -1) and np.all(a["subregion"] == 0)
        assert a["nu"].shape == (n, 5) and not np.any(a["nu"]) and not np.any(a["contaminated"])
        want = np.array([u_hand(e.plan, t, k, i, truth) for k, i in kept])
        assert_allclose(a["theta"], want, rtol=0, atol=1e-7)     # float32 storage of u in [0, 1]
        sums = a["x"].astype(np.float64).sum(axis=1)             # count conservation (sec. 2.1)
        assert_allclose(sums, [FX.spikes_of(t, k, i) for k, i in kept], rtol=1e-6)


def test_71_sidecar_contract():
    e = shared()
    lb, ln = EXP.stage1_modules()
    want = lb.build_sidecar(**EXP.contract_kwargs(e.plan, ln.NU_COMPONENTS))["contract_digest"]
    paths = sorted(glob.glob(os.path.join(e.bank, SHARD_GLOB)))
    assert len(paths) == 3, paths
    for idx, p in enumerate(paths):
        with open(p[:-4] + ".json") as fh:
            side = json.load(fh)
        assert side["contract_digest"] == want == lb.contract_digest(side)
        assert (side["coord"], side["arm"], side["W"], side["fs"], side["T_win"]) == (
            "unit_box", "S", W, 100.0, 180.0), side["coord"]
        assert side["param_names"] == e.plan["label"]["param_names"]
        assert side["bounds_theta"] == e.plan["label"]["bounds_declared"]
        assert side["generator_sha256"] == e.plan["inputs"]["record_sha256"]
        assert side["latent_spec"]["kind"] == "campaign"
        assert side["latent_spec"]["on_declared_box"] == [0.01, 3.0]
        assert side["provenance"]["on_box"] == FX.TASKS[idx][2]
        assert side["provenance"]["plan_sha256"] == e.plan["_sha256"]
    a, side = lb.concat_shards(paths)
    assert a["x"].shape == (24, W) and a["theta"].dtype == np.float32
    assert 0.0 <= float(a["theta"].min()) and float(a["theta"].max()) <= 1.0


def test_71_empty_topology():
    e = Exported(export=False, extra_topos={SP.EMPTY[0]: {SP.EMPTY[1]: []}})
    try:
        t = FX.TASKS[SP.EMPTY[0]]
        folders = [os.path.join(e.f.dir, tree, t[0], t[1], "topo_%05d" % SP.EMPTY[1])
                   for tree in ("sim", "mea")]
        for p in folders:                 # renamed in both trees: the counts are unchanged
            os.rename(p, p[:-5] + "%05d" % (SP.EMPTY[1] + 1))
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index",
                                          str(SP.EMPTY[0])])
        assert rc == 1 and "differ from the plan's" in out[-1], out[-1:]
        assert not os.path.exists(EXP.marker_path(e.bank, SP.EMPTY[0]))
        for p in folders:
            os.rename(p[:-5] + "%05d" % (SP.EMPTY[1] + 1), p)
        for i in range(len(e.plan["tasks"])):
            EXP.export_task(e.plan, i, e.bank, log=lambda *a: None)
        base = shared()
        lb, _ln = EXP.stage1_modules()
        for idx in range(len(FX.TASKS)):
            assert e.marker(idx)["totals"] == base.marker(idx)["totals"], idx
            a, _side = lb.read_shard(os.path.join(e.bank, "shard_%04d.npz" % idx))
            b, _side = lb.read_shard(os.path.join(base.bank, "shard_%04d.npz" % idx))
            assert sorted(a) == sorted(b), (sorted(a), sorted(b))
            for key in a:
                assert np.array_equal(a[key], b[key]), (idx, key)
    finally:
        e.close()


def test_71_gate_rows():
    """D-107: the gate's value is each row's last label column; nothing else moves."""
    e = Exported(gate=FX.GATE_BOX)
    try:
        base = shared()
        lb, _ln = EXP.stage1_modules()
        lo, hi = FX.GATE_BOX
        for idx, t in enumerate(FX.TASKS):
            assert e.marker(idx)["totals"] == base.marker(idx)["totals"], idx
            a, side = lb.read_shard(os.path.join(e.bank, "shard_%04d.npz" % idx))
            b, _side = lb.read_shard(os.path.join(base.bank, "shard_%04d.npz" % idx))
            kept = kept_positions(t)
            assert a["theta"].shape == (len(kept), 24) and a["theta"].dtype == np.float32
            assert np.array_equal(a["theta"][:, :23], b["theta"]), idx      # same computation
            want = np.array([(FX.gate_of(t, k) - lo) / (hi - lo) for k, _i in kept])
            assert_allclose(a["theta"][:, 23], want, rtol=0, atol=1e-7)      # float32 storage
            for key in a:
                if key != "theta":
                    assert np.array_equal(a[key], b[key]), (idx, key)
            assert side["param_names"][-1] == "stoa_gate_p"
            assert side["bounds_theta"][-1] == [lo, hi]
            assert side["latent_spec"]["topology_axes"] == ["conn_prob", "stoa_gate_p"]
        rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "1"])
        assert rc == 0 and "BANK_RECORD.json" in out[-1], out[-3:]
    finally:
        e.close()


def test_71_gate_refusals():
    e = Exported(export=False, gate=FX.GATE_BOX)
    try:
        t = FX.TASKS[0]
        last = SP.raw_path(e.f.dir, t, 1, t[4][1][-1])     # topology 1, its last iteration
        SP.rewrite_npz(last, stoa_gate_p=np.float64(0.5))  # gate_of(t, 1) is 0.35
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "0"])
        assert rc == 1 and out[-1].startswith("FAILED:"), out[-1:]
        assert "topology 1: stoa_gate_p differs" in out[-1] and "0.35 vs 0.5" in out[-1], out[-1:]
        assert not os.path.exists(EXP.marker_path(e.bank, 0))
        SP.rewrite_job_args(e.f, FX.TASKS[2], stoa_gate_hi=0.9)   # after the plan
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "2"])
        assert rc == 1 and "label bounds outside O_N differ from the plan's" in out[-1], out[-1:]
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "1"])
        assert rc == 0 and out[-1] == "TASK 1 OK", out[-2:]
        rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "1"])
        assert rc == 1 and out[-1].startswith("REFUSED:"), out[-3:]
        assert not os.path.exists(os.path.join(e.bank, "BANK_RECORD.json"))
    finally:
        e.close()


def test_71_unversioned_campaign():
    """D-108: a campaign without _v<N> exports with batch 0, and the gate passes."""
    e = Exported(campaign_names={0: "campaign_meama_rho1600_n01"})
    try:
        lb, _ln = EXP.stage1_modules()
        batches = []
        for idx in range(3):
            a, _side = lb.read_shard(os.path.join(e.bank, "shard_%04d.npz" % idx))
            batches.append(sorted(set(int(v) for v in a["batch"])))
        assert batches == [[0], [2], [5]], batches
        rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "1"])
        assert rc == 0 and "BANK_RECORD.json" in out[-1], out[-3:]
    finally:
        e.close()


# ------------------------------------------------------- Block 7.2 checks ---

def test_72_task_cli():
    e = Exported(export=False)
    try:
        for i in range(3):
            rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", str(i)])
            assert rc == 0 and out[-1] == "TASK %d OK" % i, out[-2:]
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "1"])
        assert rc == 0 and "already complete" in out[-2], out[-2:]
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "3"])
        assert rc == 1 and out[-1].startswith("FAILED:") and "not a task" in out[-1], out[-1:]
        other = os.path.join(e.f.dir, "other", "bank_plan.json")       # a second plan, same folder
        p2 = FX.write_profile(e.f.fx, os.path.join(e.f.dir, "p2.json"), sample_seed=99,
                              plan_out=other)
        rc, out = SP.run_main(PLAN.main, ["--profile", p2])
        assert rc == 0, out[-1:]
        rc, out = SP.run_main(TASK.main, ["--plan", other, "--task-index", "0"])
        assert rc == 1 and "another plan" in out[-1], out[-1:]
    finally:
        e.close()


def test_72_detections_changed_after_plan():
    e = Exported(export=False)
    try:                                  # same counts, another index: the plan's chunks would shift
        t = FX.TASKS[0]
        p = SP.mea_path(e.f.dir, t, 1, 3)
        os.rename(p, SP.mea_path(e.f.dir, t, 1, 9))
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "0"])
        assert rc == 1 and "differ from the plan's" in out[-1], out[-1:]
        assert not os.path.exists(EXP.marker_path(e.bank, 0))
    finally:
        e.close()


def test_72_stale_shard_of_an_empty_chunk():
    e = Exported(export=False, floor_hz=50.0)       # every window below 50 Hz: no file at all
    try:
        os.makedirs(e.bank)
        stale = os.path.join(e.bank, "shard_0000.npz")
        shutil.copy(SP.mea_path(e.f.dir, FX.TASKS[0], 0, 0), stale)
        rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", "0"])
        assert rc == 1 and "from an earlier run" in out[-1], out[-1:]
        os.rename(stale, stale + ".aside")
        for i in range(3):
            rc, out = SP.run_main(TASK.main, ["--plan", e.plan_path, "--task-index", str(i)])
            assert rc == 0, out[-1:]
        assert not glob.glob(os.path.join(e.bank, SHARD_GLOB))
        os.rename(stale + ".aside", stale)             # the gate sees it too
        rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "1"])
        assert rc == 1 and any("keeps no row" in line for line in out), out
        assert not os.path.exists(os.path.join(e.bank, "BANK_RECORD.json"))
    finally:
        e.close()


# ------------------------------------------------------- Block 7.3 checks ---

def test_73_gate_pass():
    e = shared()
    lb, ln = EXP.stage1_modules()
    rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "2"])
    assert rc == 0 and out[-1].startswith("wrote ") and "BANK_RECORD.json" in out[-1], out[-3:]
    path = os.path.join(e.bank, "BANK_RECORD.json")
    with open(path + ".sha256") as fh:
        assert fh.read().split()[0] == SP.sha256_of(path)
    with open(path) as fh:
        rec = json.load(fh)
    kept = sum(e.marker(i)["totals"]["kept"] for i in range(3))
    assert (rec["counts"]["rows_read"], rec["counts"]["rows_kept"], kept) == (32, 24, 24)
    assert (rec["counts"]["tasks"], rec["counts"]["shards"]) == (3, 3)
    want = lb.build_sidecar(**EXP.contract_kwargs(e.plan, ln.NU_COMPONENTS))["contract_digest"]
    assert rec["contract_digest"] == want and rec["plan_sha256"] == e.plan["_sha256"]


def test_73_gate_refusals():
    lb, _ln = EXP.stage1_modules()

    def refuses(mutate, fragment):
        e = Exported()
        try:
            mutate(e)
            rc, out = SP.run_main(GATE.main, ["--plan", e.plan_path, "--workers", "1"])
            assert rc == 1 and out[-1].startswith("REFUSED:"), out[-3:]
            assert any(fragment in line for line in out), (fragment, out)
            assert not os.path.exists(os.path.join(e.bank, "BANK_RECORD.json"))
        finally:
            e.close()

    def no_marker(e):
        p = EXP.marker_path(e.bank, 1)
        os.rename(p, p + ".aside")

    def flip_byte(e):
        p = os.path.join(e.bank, "shard_0000.npz")
        with open(p, "r+b") as fh:
            fh.seek(200)
            b = fh.read(1)
            fh.seek(200)
            fh.write(bytes([b[0] ^ 0xFF]))

    def stray(e):
        shutil.copy(os.path.join(e.bank, "shard_0000.npz"), os.path.join(e.bank, "shard_0099.npz"))

    def other_plan(e):
        p = EXP.marker_path(e.bank, 2)
        m = e.marker(2)
        m["plan_sha256"] = "0" * 64
        with open(p, "w") as fh:
            json.dump(m, fh)

    def wrong_well(e):                   # the bytes and the marker agree; the content does not
        p = os.path.join(e.bank, "shard_0001.npz")
        a, side = lb.read_shard(p)
        a["well"] = a["well"] + 1
        lb.write_shard(p, a, side)
        m = e.marker(1)
        m["shards"][0]["sha256"] = EXP.sha256_file(p)
        m["shards"][0]["json_sha256"] = EXP.sha256_file(p[:-4] + ".json")
        with open(EXP.marker_path(e.bank, 1), "w") as fh:
            json.dump(m, fh)

    refuses(no_marker, "no marker")
    refuses(flip_byte, "changed since its marker")
    refuses(stray, "the plan does not name")
    refuses(other_plan, "another plan")
    refuses(wrong_well, "well / batch other than")


# ------------------------------------------------------- Block 7.4 checks ---

def test_74_launcher_and_jobs():
    e = Exported(export=False)
    try:
        fb = fake_tools(e.f.dir)
        log = os.path.join(e.f.dir, "qsub.log")
        env = dict(os.environ, PROFILE=e.f.profile, QSUB=os.path.join(fb, "qsub"),
                   FAKE_QSUB_LOG=log,
                   PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "ENV_NAME"):
            env.pop(k, None)
        rc, out = run_bash(["launch_bank.sh", "plan"], env)
        assert rc == 0 and "BANK PLAN OK" in out, out[-3:]
        rc, out = run_bash(["launch_bank.sh", "array"], env)
        assert rc == 3 and "is required for the array" in out[-1] and not os.path.exists(log), out[-1:]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, WALLTIME="auto"))
        assert rc == 3 and "run the test first" in out[-1] and not os.path.exists(log), out[-1:]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, WALLTIME="HH:MM:SS"))
        assert rc == 3 and "is not hh:mm:ss" in out[-1] and not os.path.exists(log), out[-1:]
        rc, out = run_bash(["launch_bank.sh"], env)
        assert rc == 2 and any("launch_bank.sh -- B5" in line for line in out), out[:3]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, DRYRUN="1", WALLTIME="01:00:00"))
        assert rc == 0 and not os.path.exists(log), out[-3:]
        assert any("-J 0-2%20" in line for line in out), out
        rc, out = run_bash(["launch_bank.sh", "test"], dict(env, TEST_INDEX="7"))
        assert rc == 3 and "not a task of the plan" in out[-1], out[-1:]
        rc, out = run_bash(["launch_bank.sh", "test"], dict(env, TEST_INDEX="1"))
        assert rc == 0, out[-3:]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, WALLTIME="01:00:00"))
        assert rc == 0, out[-3:]
        with open(log) as fh:
            calls = fh.read().splitlines()
        assert len(calls) == 3, calls
        assert "TASK_INDEX=1" in calls[0] and " -J " not in calls[0] and "run_bank_task.pbs" in calls[0]
        assert "-J 0-2%20" in calls[1] and calls[1].endswith("run_bank_task.pbs")
        assert "_^array_index^.log" in calls[1]
        assert "depend=afterok:2000[].pbsserver01" in calls[2] and calls[2].endswith("run_bank_gate.pbs")
        # the job's environment: this one (LD_LIBRARY_PATH from the env's activate.d
        # included, HPC_PATHS.md 7.1), the fake conda first on PATH, no index or plan
        jenv = {k: v for k, v in os.environ.items()
                if k not in ("PBS_ARRAY_INDEX", "TASK_INDEX", "PLAN", "DRYRUN", "ENV_NAME")}
        jenv.update(PATH=fb + os.pathsep + os.environ.get("PATH", ""), PBS_O_WORKDIR=HERE,
                    PLAN=e.plan_path)
        for i in range(3):
            rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, PBS_ARRAY_INDEX=str(i)))
            assert rc == 0 and ("TASK %d OK" % i) in out and "env=sbi_export" in " ".join(out), out[-4:]
        rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, TASK_INDEX="1"))   # a plain job
        assert rc == 0 and "TASK 1 OK" in out and "task=1" in " ".join(out), out[-4:]
        rc, out = run_bash(["run_bank_gate.pbs"], dict(jenv, NCPUS="2"))
        assert rc == 0 and any(line.startswith("wrote ") for line in out), out[-4:]
        # WALLTIME=auto from task 1's marker: seconds, so the 30 min floor
        rc, out = run_bash(["launch_bank.sh", "array"],
                           dict(env, DRYRUN="1", WALLTIME="auto", TEST_INDEX="1"))
        assert rc == 0 and any(line.startswith("[launch] WALLTIME=auto: test task 1 read 10")
                               for line in out), out[-8:]
        assert any("walltime=00:30:00" in line and "-J 0-2%20" in line for line in out), out[-8:]
        mk = EXP.marker_path(e.bank, 1)
        with open(mk) as fh:
            m = json.load(fh)
        m["plan_sha256"] = "0" * 64
        with open(mk, "w") as fh:
            json.dump(m, fh)
        rc, out = run_bash(["launch_bank.sh", "array"],
                           dict(env, DRYRUN="1", WALLTIME="auto", TEST_INDEX="1"))
        assert rc == 3 and "belongs to another plan" in out[-1], out[-1:]
        rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, PBS_ARRAY_INDEX="0", ENV_NAME="nope"))
        assert rc == 2 and any("conda activate nope" in line for line in out), out[-3:]
        rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, PBS_O_WORKDIR=e.f.dir, TASK_INDEX="0"))
        assert rc == 2 and "Sbi-extractor/export_windows" in out[-1], out[-1:]
    finally:
        e.close()


def test_74_auto_walltime_empty_test_task():
    """Amendment 2 (F3): after a test task that read 0 simulations (Block 10 can leave
    a task without one) `array` with WALLTIME=auto refuses, exit 3, with or without
    DRYRUN, naming the plan's first task with simulations, and submits nothing; `test`
    itself still submits such a task, and the element runs it. Task 0 empty: the first
    is task 1 (neither task 0 nor the largest, task 2). Task 1 empty: the first is task 0
    (not the next one, task 2), and a test on task 0 still sizes the array (the floor)."""
    msg = ("ABORT: ERROR WALLTIME=auto: the test task %d read 0 simulations, so it gives no "
           "pace: run the test on a task that holds simulations (TEST_INDEX=%d, the plan's first)")
    for empty, first in ((0, 1), (1, 0)):
        f, plan = plan_with_empty_tasks({empty})
        try:
            fb = fake_tools(f.dir)
            log = os.path.join(f.dir, "qsub.log")
            env = launcher_env(f.profile, fb, log)
            rc, out = run_bash(["launch_bank.sh", "test"], dict(env, TEST_INDEX=str(empty)))
            assert rc == 0, out[-3:]
            with open(log) as fh:
                calls = fh.read().splitlines()
            assert len(calls) == 1 and "TASK_INDEX=%d," % empty in calls[0], calls
            rc, out = SP.run_main(TASK.main, ["--plan", plan["_path"], "--task-index", str(empty)])
            assert rc == 0 and out[-1] == "TASK %d OK" % empty, out[-2:]
            with open(EXP.marker_path(f.cfg().bank_root, empty)) as fh:
                assert json.load(fh)["totals"]["read"] == 0
            for dry in ("0", "1"):
                rc, out = run_bash(["launch_bank.sh", "array"],
                                   dict(env, WALLTIME="auto", TEST_INDEX=str(empty), DRYRUN=dry))
                assert rc == 3 and out[-1] == msg % (empty, first), (dry, out[-2:])
                assert not any(line.startswith("[launch] WALLTIME=auto:") for line in out), out
                with open(log) as fh:
                    assert fh.read().splitlines() == calls, dry       # nothing submitted
            if empty == 1:
                rc, out = SP.run_main(TASK.main, ["--plan", plan["_path"], "--task-index", "0"])
                assert rc == 0 and out[-1] == "TASK 0 OK", out[-2:]
                rc, out = run_bash(["launch_bank.sh", "array"],
                                   dict(env, WALLTIME="auto", TEST_INDEX="0", DRYRUN="1"))
                assert rc == 0 and any(line.startswith("[launch] WALLTIME=auto: test task 0 read 11")
                                       for line in out), out[-8:]
                assert any("walltime=00:30:00" in line and "-J 0-2%20" in line for line in out), out[-8:]
        finally:
            f.close()


def test_75_pipeline_protocol():
    """D-113: launch_bank.sh passes the pipeline's variables to every job, and
    both job scripts start through the pipeline's job protocol: on a node
    without conda the environment's own folder; no environment anywhere ->
    exit 96 and nothing written; a host on the avoid list -> exit 94."""
    hpc = os.environ.get("SBI_HPC_DIR")
    jd = PPF.jobs_dir(hpc)
    assert jd is not None, ("SBI_HPC_DIR=%s has no pipeline/jobs/stage_job.sh (D-113): pull "
                            "Simulation-Based-Inference" % hpc)
    e = Exported(export=False)
    try:
        fb = fake_tools(e.f.dir)
        log = os.path.join(e.f.dir, "qsub.log")
        wr = os.path.join(e.f.dir, "pipe_work")
        envf = PPF.make_env_folder(e.f.dir)
        pv = PPF.protocol_env(wr, env_prefix=envf, chain="pl_bank", hpc=hpc)
        # a node without conda: no folder holding a `conda` on PATH, no CONDA*
        nc_path = PPF.no_conda_path(os.environ.get("PATH", ""))
        base = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "_CONDA", "PBS_"))}
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "ENV_NAME", "TASK_INDEX", "PLAN", "AVOID_FILE",
                  "SKIP_CONDA"):
            base.pop(k, None)
        env = dict(base, PROFILE=e.f.profile, QSUB=os.path.join(fb, "qsub"), FAKE_QSUB_LOG=log,
                   PATH=os.path.dirname(sys.executable) + os.pathsep + nc_path, **pv)
        want = "ENV_NAME=sbi_export,PIPELINE_JOBS_DIR=%s,WORK_ROOT=%s,CHAIN_NAME=pl_bank,AVOID_HOLD_S=0" \
            % (jd, wr)
        rc, out = run_bash(["launch_bank.sh", "plan"], env)
        assert rc == 0 and "BANK PLAN OK" in out, out[-3:]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, DRYRUN="1", WALLTIME="01:00:00"))
        gate = [line for line in out if "run_bank_gate.pbs" in line]
        assert rc == 0 and not os.path.exists(log) and len(gate) == 1 and want in gate[0], out[-4:]
        rc, out = run_bash(["launch_bank.sh", "test"], dict(env, WORK_ROOT=wr + ",x", TEST_INDEX="1"))
        assert rc == 3 and "contains whitespace or a comma" in " ".join(out) and not os.path.exists(log), out[-2:]
        rc, out = run_bash(["launch_bank.sh", "test"], dict(env, PIPELINE_JOBS_DIR=e.f.dir, TEST_INDEX="1"))
        assert rc == 3 and "has no stage_job.sh" in " ".join(out) and not os.path.exists(log), out[-2:]
        rc, out = run_bash(["launch_bank.sh", "test"], dict(env, TEST_INDEX="1"))
        assert rc == 0, out[-3:]
        rc, out = run_bash(["launch_bank.sh", "array"], dict(env, WALLTIME="01:00:00"))
        assert rc == 0, out[-3:]
        with open(log) as fh:
            calls = fh.read().splitlines()
        assert len(calls) == 3 and all(want in c for c in calls), calls
        assert "TASK_INDEX=1," in calls[0] and calls[0].endswith("run_bank_task.pbs")
        assert "-J 0-2%20" in calls[1] and calls[2].endswith("run_bank_gate.pbs")
        # the jobs, run in place as a node would run them
        jenv = dict(base, PATH=nc_path, PBS_O_WORKDIR=HERE, PLAN=e.plan_path, **pv)
        decoy = os.path.join(e.f.dir, "home_decoy")             # no ~/.conda/envs/sbi_export
        os.makedirs(decoy)
        pv0 = PPF.protocol_env(os.path.join(e.f.dir, "pipe_work_noenv"), chain="pl_bank", hpc=hpc)
        rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, HOME=decoy, TASK_INDEX="0", **pv0))
        text = "\n".join(out)
        assert rc == 96 and "RUN_ARGV ENV FAULT: host " in text and "then exiting 96" in text \
            and "[job]" not in text and not os.path.exists(EXP.marker_path(e.bank, 0)), out[-4:]
        avoid = os.path.join(e.f.dir, "avoid_nodes.txt")
        with open(avoid, "w") as fh:
            fh.write("# test\n%s\n" % socket.gethostname())
        rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, TASK_INDEX="0", AVOID_FILE=avoid))
        text = "\n".join(out)
        assert rc == 94 and "RUN_ARGV AVOID: host %s" % socket.gethostname() in text \
            and "[job]" not in text and not os.path.exists(EXP.marker_path(e.bank, 0)), out[-4:]
        folder = "sbi_export activated from its folder %s" % envf
        for i in range(3):
            rc, out = run_bash(["run_bank_task.pbs"], dict(jenv, PBS_ARRAY_INDEX=str(i)))
            text = "\n".join(out)
            assert rc == 0 and ("TASK %d OK" % i) in out and folder in text \
                and "(the pipeline's job protocol, D-113)" in text and "env=sbi_export" in text, out[-6:]
        rc, out = run_bash(["run_bank_gate.pbs"], dict(jenv, NCPUS="2"))
        assert rc == 0 and folder in "\n".join(out) and any(line.startswith("wrote ") for line in out), out[-4:]
    finally:
        e.close()


def test_74_scripts_bytes():
    for name in SCRIPTS:
        p = os.path.join(HERE, name)
        with open(p, "rb") as fh:
            data = fh.read()
        assert data.count(b"\r") == 0 and max(data) < 128, name
        assert data.startswith(b"#!/bin/bash\n"), name
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)


# --------------------------------------------------------- Block 8 checks ---

def test_81_windows_of():
    assert REAL.windows_of(30000, W) == [0]
    assert REAL.windows_of(36000, W) == [0, W]
    assert REAL.windows_of(60000, W) == [0, W, 2 * W]
    assert REAL.windows_of(17999, W) == []


def test_81_real_known_answer():
    f = SP.Fixture()
    try:
        arrays, info = REAL.build_real(f.cfg())
        assert info["per_class"] == {
            "100N0A_wo": {"windows": 18, "below_floor": 9, "wells": 2},
            "50N50A": {"windows": 27, "below_floor": 0, "wells": 1},
            "70N30A": {"windows": 27, "below_floor": 0, "wells": 1}}, info["per_class"]
        assert arrays["x"].shape == (72, W) and arrays["theta"].shape == (72, 0)
        roots = sorted(w[1] for w in FX.REAL_WELLS)
        real = os.path.join(f.dir, "real")
        traces = {}
        for r in range(72):
            w, s, j = int(arrays["well"][r]), int(arrays["subregion"][r]), int(arrays["window_idx"][r])
            cls_name, root_name, well = FX.REAL_WELLS[w][:3]
            key = (w, s)
            if key not in traces:
                with np.load(os.path.join(real, cls_name, root_name, well,
                                          "trace_subregion_%02d.npz" % s)) as z:
                    traces[key] = np.asarray(z["ifr_trace"])
            assert np.array_equal(arrays["x"][r], traces[key][j * W:(j + 1) * W]), (w, s, j)
            assert int(arrays["cls"][r]) == FX.REAL_CLASSES.index(cls_name)
            assert int(arrays["batch"][r]) == roots.index(root_name)
            assert int(arrays["donor"][r]) == w == int(arrays["realisation_id"][r])
        per_well = {w: int(np.sum(arrays["well"] == w)) for w in range(4)}
        assert per_well == {0: 9, 1: 9, 2: 27, 3: 27}, per_well     # 300 s: 1 window; 600 s: 3
    finally:
        f.close()


def test_81_real_cli_and_contract():
    f = SP.Fixture()
    try:
        lb, _ln = EXP.stage1_modules()
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 0 and out[-1] == "REAL BANK OK", out[-3:]
        p = os.path.join(f.cfg().real_bank_root, "shard_0000.npz")
        a, side = lb.read_shard(p)
        assert a["theta"].shape == (72, 0) and side["arm"] == "R" and side["W"] == W
        assert side["param_names"] == [] and side["bounds_theta"] == [] and side["coord"] == "unit_box"
        assert side["generator_sha256"] == SP.sha256_of(f.fx["manifest"])
        assert side["latent_spec"]["kind"] == "campaign_real"
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 1 and "never overwritten" in out[-1], out[-1:]
    finally:
        f.close()


def test_81_real_refusals():
    for overrides, fragment in (({"cohort_manifest_sha256_prefix": "0" * 16}, "the profile expects"),
                                ({"n_e": 2}, "D-043")):
        f = SP.Fixture(**overrides)
        try:
            rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
            assert rc == 1 and fragment in out[-1], out[-1:]
        finally:
            f.close()
    f = SP.Fixture()
    try:
        cls_name, root_name, well = FX.REAL_WELLS[2][:3]
        p = os.path.join(f.dir, "real", cls_name, root_name, well, "trace_subregion_04.npz")
        os.rename(p, p + ".aside")
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 1 and "missing archive" in out[-1], out[-1:]
    finally:
        f.close()


def test_81_real_too_short():
    """Amendment 2 (F4; SPEC sec. 4), with amendment 3: a trace shorter than one window
    gives no row and is counted. The manifest gives no T_rec for wells 0 and 2 (it is
    optional), so their archives can be cut while the manifest, and with it the
    contract digest, stays the same. Well 0 (class 0, 300 s) cut to 100 s (K = 10000
    < W): no row, its class's count line and the well named, the sidecar's
    provenance.too_short {"traces": 9, "wells_without_rows": [its out_dir]} with
    latent_spec and the contract digest the uncut fixture's, per_class counting the
    wells with rows, the other rows bit for bit those of the uncut fixture without that
    well's; on the uncut fixture no line and too_short 0 and []. Then three of well 2's
    nine traces cut to 150 s: counted (12), the well not named, since it keeps rows.
    Last, the manifest giving well 0 the T_rec of its cut archives (100 s): the same
    rows and too_short, latent_spec moved by the manifest's digest alone."""
    f = SP.Fixture()
    try:
        lb, _ln = EXP.stage1_modules()
        culture, out_dir = real_well(0)
        line_9 = ("[real] 100N0A_wo: 9 trace(s) shorter than one window (K < W, W = 18000 "
                  "samples): no row")
        line_well = "[real]   %s (%s): every trace shorter than one window, no row" % (culture, out_dir)

        def no_t_rec(doc):
            for wi in (0, 2):
                doc["wells"][wi]["T_rec"] = None
        resign_manifest(f, no_t_rec)
        base, info = REAL.build_real(f.cfg())
        assert info["too_short"] == {"traces": 0, "wells_without_rows": []}, info["too_short"]
        rc, base_out = SP.run_main(REAL.main, ["--profile", f.profile,
                                               "--out-root", os.path.join(f.dir, "real_0")])
        assert rc == 0 and not any("shorter than one window" in line for line in base_out), base_out
        _a, side0 = lb.read_shard(os.path.join(f.dir, "real_0", "shard_0000.npz"))
        assert side0["provenance"].get("too_short") == {"traces": 0, "wells_without_rows": []}, \
            side0["provenance"]
        assert "too_short" not in side0["latent_spec"], side0["latent_spec"]
        cut_traces(f, 0, 100.0)
        for cut, n_short in (((base["well"] == 0), 9),
                             ((base["well"] == 0) | ((base["well"] == 2) & (base["subregion"] < 3)), 12)):
            if n_short == 12:
                cut_traces(f, 2, 150.0, subregions=(0, 1, 2))
            arrays, info = REAL.build_real(f.cfg())
            assert sorted(arrays) == sorted(base), sorted(arrays)
            for key in base:                                    # the other rows, bit for bit
                assert arrays[key].dtype == base[key].dtype, key
                assert np.array_equal(arrays[key], base[key][~cut]), (n_short, key)
            assert info["too_short"] == {"traces": n_short, "wells_without_rows": [out_dir]}, info
            assert info["per_class"]["100N0A_wo"] == {"windows": 9, "below_floor": 9, "wells": 1}
            assert info["n_wells"] == 4, info["n_wells"]
            rc, out = SP.run_main(REAL.main, ["--profile", f.profile,
                                              "--out-root", os.path.join(f.dir, "real_%d" % n_short)])
            assert rc == 0 and out[-1] == "REAL BANK OK", out[-3:]
            assert out[1].startswith("[real] 100N0A_wo ") and "1 wells,    9 windows" in out[1], out
            assert out[2:4] == [line_9, line_well], out
            if n_short == 9:
                assert out[4:6] == base_out[2:4] and len(out) == len(base_out) + 2, (out, base_out)
            else:
                assert out[4].startswith("[real] 50N50A ") and "1 wells,   18 windows" in out[4], out
                assert out[5] == ("[real] 50N50A: 3 trace(s) shorter than one window (K < W, "
                                  "W = 18000 samples): no row"), out
                assert out[6] == base_out[3] and len(out) == len(base_out) + 3, (out, base_out)
            _a, side = lb.read_shard(os.path.join(f.dir, "real_%d" % n_short, "shard_0000.npz"))
            assert side["provenance"].get("too_short") == {
                "traces": n_short, "wells_without_rows": [out_dir]}, side["provenance"]
            assert side["latent_spec"] == side0["latent_spec"], side["latent_spec"]
            assert side["contract_digest"] == side0["contract_digest"] == lb.contract_digest(side)
        digest = resign_manifest(f, lambda doc: doc["wells"][0].update(T_rec=100.0))
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile,
                                          "--out-root", os.path.join(f.dir, "real_t_rec")])
        assert rc == 0 and out[-1] == "REAL BANK OK", out[-3:]
        a, side = lb.read_shard(os.path.join(f.dir, "real_t_rec", "shard_0000.npz"))
        assert sorted(a) == sorted(arrays), sorted(a)
        for key in arrays:
            assert a[key].dtype == arrays[key].dtype and np.array_equal(a[key], arrays[key]), key
        assert side["provenance"].get("too_short") == {"traces": 12, "wells_without_rows": [out_dir]}
        assert side["latent_spec"] == dict(side0["latent_spec"], cohort_manifest_sha256=digest)
        assert side["generator_sha256"] == digest != side0["generator_sha256"]
    finally:
        f.close()


def test_81_real_out_dir_refusals():
    """Amendment 2 (F5): a manifest well whose out_dir has fewer than three components
    (`control`, then `70N30A/AraC_70N30A`) is refused, `<culture>: its folder <out_dir>
    has no <class>/<root>/ prefix`, exit 1 and no shard, never an uncaught IndexError;
    on every well before any archive is read (well 0's archive 04, moved aside, is
    never reached), and also when a cohort map leaves that well out."""
    culture = real_well(3)[0]
    for bad in ("control", "70N30A/AraC_70N30A"):
        f = SP.Fixture()
        try:
            cls_name, root_name, well = FX.REAL_WELLS[0][:3]
            p = os.path.join(f.dir, "real", cls_name, root_name, well, "trace_subregion_04.npz")
            os.rename(p, p + ".aside")
            digest = resign_manifest(f, lambda doc: doc["wells"][3].update(out_dir=bad))
            cmap = os.path.join(f.dir, "cohort_map.csv")
            with open(cmap, "w") as fh:
                fh.write("out_dir,donor,rep_group,include\n")
                fh.write("".join("%s,d%d,g%d,yes\n" % (real_well(wi)[1], wi, wi) for wi in range(3)))
                fh.write("%s,d3,g3,no\n" % bad)
            want = "REFUSED: %s: its folder %s has no <class>/<root>/ prefix" % (culture, bad)
            for overrides in ({}, {"cohort_map": cmap}):
                FX.write_profile(f.fx, f.profile, cohort_manifest_sha256_prefix=digest[:16],
                                 **overrides)
                assert f.cfg().cohort_map == (cmap if overrides else None)
                rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
                assert rc == 1 and out[-1] == want, (bad, overrides, out[-1:])
                assert not os.path.exists(os.path.join(f.cfg().real_bank_root, "shard_0000.npz"))
        finally:
            f.close()


# -------------------------------------------- the runner, optional (SKIP) ---

def test_90_runner_dry_run():
    runner = os.path.join(os.environ.get("SBI_HPC_DIR", ""), "joint", "stage3", "run_joint_arms.py")
    missing = [m for m in ("torch", "sbi") if importlib.util.find_spec(m) is None]
    if missing or not os.path.isfile(runner):
        raise unittest.SkipTest("needs torch and sbi (the sbi_env environment) and %s; missing: %s"
                                % (runner, missing or "the runner"))
    e = shared()
    f = SP.Fixture()
    try:
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 0, out[-1:]
        real = os.path.join(f.cfg().real_bank_root, SHARD_GLOB)
        r = subprocess.run([sys.executable, runner, "--arm", "A1", "--seed", "0",
                            "--sim-shards", os.path.join(e.bank, SHARD_GLOB), "--real-shards", real,
                            "--out-dir", os.path.join(f.dir, "runs"), "--dry-run"],
                           cwd=os.path.dirname(runner), capture_output=True, text=True, timeout=600)
        text = r.stdout + r.stderr
        assert r.returncode == 0, text[-2000:]
        assert "sim shards       : 3, rows 24, W 18000, d_theta 23" in text, text[-1500:]
        assert "real rows        : 72" in text, text[-1500:]
    finally:
        f.close()


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append("SBI_HPC_DIR=%s" % os.environ.get("SBI_HPC_DIR", "(unset)"))
    parts.append(platform.platform())
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds,
                                           lines[0] if lines else "")).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
