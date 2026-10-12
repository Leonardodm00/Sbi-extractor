"""Smoke test for the Giulia window banks -- Blocks 7 and 8 in specs/SPEC.md
(B5.2-B5.5).

Covers Block 7.1 (one task's rows and shards), 7.2 (the array element), 7.3
(the gate and the bank record), 7.4 (the launcher and the job scripts, with a
fake qsub and a fake conda) and Block 8 (the real bank). Every check runs on
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
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np
from numpy.testing import assert_allclose

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_export as EXP             # noqa: E402
import bank_gate as GATE              # noqa: E402
import bank_plan as PLAN              # noqa: E402
import bank_plan_fixture as FX        # noqa: E402
import bank_real as REAL              # noqa: E402
import bank_task as TASK              # noqa: E402
import smoke_test_bank_plan as SP     # noqa: E402  (its helpers; its checks run in its own suite)

SEED = 0
REPORT_PACKAGES = ("numpy", "scipy")
W = 18000                               # 180 s at 0.01 s (D-041)
SCRIPTS = ("launch_bank.sh", "run_bank_task.pbs", "run_bank_gate.pbs")
SHARD_GLOB = "shard_[0-9][0-9][0-9][0-9].npz"


# --------------------------------------------------------------- helpers ---

class Exported(object):
    """A fresh fixture, its plan, and (by default) every task exported."""

    def __init__(self, export=True, extra_topos=None, **profile_overrides):
        self.f = SP.Fixture(extra_topos=extra_topos, **profile_overrides)
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
