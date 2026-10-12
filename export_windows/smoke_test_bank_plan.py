"""Smoke test for the bank plan -- Block 6 in specs/SPEC.md (B5.1).

Covers Blocks 6.1 (configuration), 6.2 (inputs), 6.3 (contract, label,
twins, donors, layout), 6.4a (one row), 6.4 (sample and estimates) and 6.5
(the plan file). Every check runs on a synthetic record written to a
temporary folder (bank_plan_fixture.py); nothing here reads the cluster's
files. Blocks 7 and 8 have their own suite, smoke_test_bank_export.py.

Run (from the Sbi-extractor root, in the sbi_export environment):
    export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc   # cohort_manifest's DSN tree
    python3 export_windows/smoke_test_bank_plan.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table; exit
code 1 if any check failed, errored or is TODO. A failed check prints its
message; the fixture's folders are removed at exit.

Keep this file pure ASCII and LF only (hpc-python-compat).
"""
from __future__ import annotations

import atexit
import collections
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import unittest
import zlib

import numpy as np
from numpy.testing import assert_allclose

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_plan_config as CFG        # noqa: E402
import bank_plan_fixture as FX        # noqa: E402
import bank_plan_load as LOAD         # noqa: E402
import bank_plan_model as MODEL       # noqa: E402
import bank_plan_sample as SAMPLE     # noqa: E402
import bank_row as ROW                # noqa: E402
import bank_plan as PLAN              # noqa: E402
import sbi_labels as SL               # noqa: E402
import cohort_manifest as SP_CM       # noqa: E402  (bank_plan_load put the repo on sys.path)

SEED = 0
REPORT_PACKAGES = ("numpy", "scipy")
W = 18000                              # 180 s at 0.01 s (D-041)


# --------------------------------------------------------------- helpers ---

class Fixture(object):
    """A fresh fixture and its profile in a temporary folder."""

    def __init__(self, **profile_overrides):
        self.dir = tempfile.mkdtemp(prefix="bank_plan_fx_")
        self.fx = FX.write_fixture(self.dir)
        self.profile = FX.write_profile(self.fx, os.path.join(self.dir, "profile.json"),
                                        **profile_overrides)

    def cfg(self):
        return CFG.load_profile(self.profile, repo_root=self.dir)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def expect_error(exc_type, fragment, func, *args, **kwargs):
    """func(*args) must raise exc_type with fragment in its message."""
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        assert fragment in str(exc), "message %r lacks %r" % (str(exc), fragment)
        return
    raise AssertionError("%s did not raise %s (%s)" % (func.__name__, exc_type.__name__,
                                                       fragment))


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


_SHARED = {}


def shared():
    """One fixture that no check mutates, with its config, inputs and registry."""
    if not _SHARED:
        f = Fixture()
        atexit.register(f.close)
        cfg = f.cfg()
        inp = LOAD.load_inputs(cfg)
        reg = SL.load_registry(cfg.sim_dir)
        lab = MODEL.label_columns(inp.tasks, reg, cfg, inp.freeze)
        _SHARED.update(f=f, cfg=cfg, inp=inp, reg=reg, lab=lab,
                       names=[FX.task_name(t) for t in FX.TASKS])
    return _SHARED


def with_contract(task, **changes):
    c = dict(task.contract)
    c.update(changes)
    return dataclasses.replace(task, contract=c)


def run_main(main, argv):
    """(exit code, printed lines) of a CLI main()."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue().strip().splitlines()


def mea_path(root, t, k, i):
    return os.path.join(root, "mea", t[0], t[1], "topo_%05d" % k, "mea_iter_%05d.npz" % i)


def raw_path(root, t, k, i):
    return os.path.join(root, "sim", t[0], t[1], "topo_%05d" % k, "iter_%05d.npz" % i)


def rewrite_npz(path, **changes):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d.update(changes)
    np.savez_compressed(path, **d)


def all_samples(sh, draws=None):
    """sample_topology for every (task, k) at the plan's draws."""
    inp, cfg, reg, lab = sh["inp"], sh["cfg"], sh["reg"], sh["lab"]
    groups = MODEL.seed_groups(inp.tasks)
    draws = draws or SAMPLE.draw_iterations(inp.tasks, groups, cfg)
    out = []
    for t in inp.tasks:
        spec = MODEL.task_label_spec(t, reg, cfg, inp.freeze)
        for k in sorted(t.topo_iters):
            s = SAMPLE.sample_topology(t, k, draws[(t.name, k)]["i"], inp, cfg, spec,
                                       lab.bounds_declared)
            out.append(s)
    return out, groups, draws


# ------------------------------------------------------- Block 6.1 checks ---

def test_61_giulia_profile_decided_values():
    cfg = CFG.load_profile("giulia")
    root = CFG.REPO_ROOT
    assert cfg.on_declared_box == (0.01, 3.0), cfg.on_declared_box          # D-074
    assert cfg.t_win_s == 180.0 and cfg.n_e == 1, (cfg.t_win_s, cfg.n_e)    # D-041, D-043
    assert cfg.floor_hz == 0.1, cfg.floor_hz                                # D-075
    assert cfg.topology_axes == ("conn_prob",), cfg.topology_axes           # G3
    assert cfg.on_axis == "O_N"
    assert cfg.plan_out == os.path.join(root, "export_windows", "giulia", "bank_plan.json")
    assert cfg.freeze_path == os.path.join(root, "artifacts", "label_axes_giulia_hhgap.json")
    assert (cfg.record_sha256_prefix, cfg.freeze_sha256_prefix,
            cfg.cohort_manifest_sha256_prefix) == ("7af3759a0f3c1350", "29e38c89b08a2dd8",
                                                   "21b7ef03bb7f776b")
    assert "giulia" in cfg.bank_root and "giulia" in cfg.plan_out           # D-046
    assert "giulia" in cfg.real_bank_root                                   # D-046
    assert cfg.expected_p == 23, cfg.expected_p                             # G3


def test_61_refusals():
    f = Fixture()
    try:
        bad = [("record_sha256_prefix", "XYZ", "16 lowercase"),
               ("on_declared_box", [3.0, 0.01], "0 < lo < hi"),
               ("on_declared_box", [0.0, 3.0], "0 < lo < hi"),
               ("floor_hz", 0.0, "floor_hz"), ("floor_rtol", 0.01, "floor_rtol"),
               ("n_e", 0, "n_e"), ("n_e", 1.5, "integer"), ("t_win_s", 0.0, "t_win_s"),
               ("bootstrap_resamples", 50, "bootstrap_resamples"),
               ("sample_seed", -1, "sample_seed"), ("surprise", 1, "unknown keys"),
               ("expected_p", 1, "expected_p"), ("bytes_sample_rows", 0, "bytes_sample_rows"),
               ("rows_per_shard", 50, "rows_per_shard"), ("rows_per_shard", True, "integer")]
        for key, value, fragment in bad:
            p = FX.write_profile(f.fx, os.path.join(f.dir, "bad.json"), **{key: value})
            expect_error(CFG.ConfigError, fragment, CFG.load_profile, p, f.dir)
        with open(f.profile) as fh:
            prof = json.load(fh)
        del prof["floor_hz"]
        p = os.path.join(f.dir, "missing.json")
        with open(p, "w") as fh:
            json.dump(prof, fh)
        expect_error(CFG.ConfigError, "missing keys", CFG.load_profile, p, f.dir)
        rel = FX.write_profile(f.fx, os.path.join(f.dir, "rel.json"), plan_out="x/plan.json")
        assert CFG.load_profile(rel, f.dir).plan_out == os.path.join(f.dir, "x", "plan.json")
    finally:
        f.close()


# ------------------------------------------------------- Block 6.2 checks ---

def test_62_known_answer():
    f = Fixture()
    try:
        inp = LOAD.load_inputs(f.cfg())
        assert [t.name for t in inp.tasks] == [FX.task_name(t) for t in FX.TASKS]
        for got, want in zip(inp.tasks, FX.TASKS):
            assert got.seeds.get("_resolved_seed_master") == want[3], got.seeds
            assert "seed_master" not in got.seeds, got.seeds   # dropped (task_seed)
            assert got.contract["param_bounds"][36] == want[2], got.name
            assert got.topo_iters == FX.topo_iters_of(want), (got.name, got.topo_iters)
            assert got.contract["simtime"] == 180.0
        assert (inp.dt, inp.sigma_sm, inp.electrodes_per_subset) == (0.01, 0.02, 1)
    finally:
        f.close()


def test_62_reference_digests():
    f = Fixture()
    try:
        inp = LOAD.load_inputs(f.cfg())
        assert inp.record_sha256 == sha256_of(f.fx["record"])
        assert inp.freeze_sha256 == sha256_of(f.fx["freeze"])
        assert inp.manifest_sha256 == sha256_of(f.fx["manifest"])
    finally:
        f.close()


def test_62_invariants():
    f = Fixture()
    try:
        inp = LOAD.load_inputs(f.cfg())
        for t, rec in zip(inp.tasks, f.fx["tasks"]):
            assert sum(len(v) for v in t.topo_iters.values()) == t.n_iters == rec["n_iters"]
            assert len(t.topo_iters) == t.n_topos == rec["n_topos"]
            assert t.contract_sig == rec["contract_sig"]
    finally:
        f.close()


def test_62_contract_readers():
    f = Fixture()
    try:
        t = FX.TASKS[1]
        name = FX.task_name(t)
        det = LOAD.read_detection(os.path.join(f.dir, "mea", t[0], t[1], "topo_00000",
                                               "mea_iter_00003.npz"))
        truth = f.fx["truth"][(name, 0, 3)]
        assert det["det_t"].dtype == np.float64 and det["det_ch"].dtype == np.int64
        assert det["det_t"].shape == (truth["n"],) == det["det_ch"].shape
        assert det["n_e"] == 1 and det["iter_idx"] == 3 and det["topo_idx"] == 0
        assert (det["conn_prob"], det["seed_run"]) == (truth["conn_prob"], truth["seed_run"])
        assert det["theta"].dtype == np.float64 and det["theta"].shape == (FX.N_PARAMS,)
        assert np.array_equal(det["theta"], truth["theta"])
        assert np.array_equal(det["params"], truth["params"])
        raw = LOAD.read_raw_ident(os.path.join(f.dir, "sim", t[0], t[1], "topo_00000",
                                               "iter_00003.npz"))
        assert (raw["conn_prob"], raw["seed_run"]) == (truth["conn_prob"], truth["seed_run"])
    finally:
        f.close()


def test_62_refusals():
    def run(mutate, fragment, **overrides):
        f = Fixture(**overrides)
        try:
            mutate(f)
            expect_error(LOAD.LoadError, fragment, LOAD.load_inputs, f.cfg())
        finally:
            f.close()

    def tamper_record(f):
        with open(f.fx["record"], "r+b") as fh:
            blob = fh.read().replace(b"sim_reextraction", b"sim_reextractioN")
            fh.seek(0)
            fh.write(blob)

    def drop_sidecar(f):
        os.remove(f.fx["record"] + ".sha256")

    def two_axes(f):
        doc = {"topology_axes": ["conn_prob", "p0_conn"]}
        digest = FX._write_json(f.fx["freeze"], doc)
        FX.write_profile(f.fx, f.profile, freeze_sha256_prefix=digest[:16])

    def drop_file(f):
        t = FX.TASKS[2]
        os.remove(os.path.join(f.dir, "mea", t[0], t[1], "topo_00000", "mea_iter_00005.npz"))

    def change_contract(f):
        t = FX.TASKS[0]
        path = os.path.join(f.dir, "sim", t[0], t[1], "job_args.json")
        with open(path) as fh:
            doc = json.load(fh)
        doc["simtime"] = 200.0
        with open(path, "w") as fh:
            json.dump(doc, fh)

    def no_tasks(f):
        with open(f.fx["record"]) as fh:
            doc = json.load(fh)
        doc["tasks"] = []
        digest = FX._write_json(f.fx["record"], doc)
        FX.write_profile(f.fx, f.profile, record_sha256_prefix=digest[:16])

    run(tamper_record, "does not match its .sha256")
    run(lambda f: None, "the profile expects", freeze_sha256_prefix="0" * 16)
    run(lambda f: None, "the profile expects", cohort_manifest_sha256_prefix="0" * 16)
    run(drop_sidecar, "no file at")
    run(two_axes, "topology_axes")
    run(drop_file, "the record says")
    run(change_contract, "contract signature")
    run(no_tasks, "lists no tasks")

    f = Fixture()
    try:
        t = FX.TASKS[0]
        topo = os.path.join(f.dir, "mea", t[0], t[1], "topo_00001")
        src = os.path.join(topo, "mea_iter_00001.npz")
        with np.load(src) as z:
            keep = {k: z[k] for k in z.files if k != "conn_prob"}
        np.savez_compressed(os.path.join(topo, "mea_iter_00007.npz"), **keep)
        expect_error(LOAD.LoadError, "has no", LOAD.read_detection,
                     os.path.join(topo, "mea_iter_00007.npz"))
        shutil.copy(src, os.path.join(topo, "mea_iter_00008.npz"))
        expect_error(LOAD.LoadError, "its name says", LOAD.read_detection,
                     os.path.join(topo, "mea_iter_00008.npz"))
        np.savez_compressed(os.path.join(topo, "mea_iter_00001.npz"),
                            **dict(keep, conn_prob=np.float64("nan")))
        expect_error(LOAD.LoadError, "not finite", LOAD.read_detection, src)
        good = dict(keep, conn_prob=np.float64(0.2))
        short = dict(good, det_t=np.array([1.0, 2.0], np.float32),
                     det_ch=np.zeros(3, np.int32))
        np.savez_compressed(src, **short)
        expect_error(LOAD.LoadError, "differ in shape", LOAD.read_detection, src)
        nan_t = dict(good, det_t=np.array([1.0, np.nan], np.float32),
                     det_ch=np.zeros(2, np.int32))
        np.savez_compressed(src, **nan_t)
        expect_error(LOAD.LoadError, "NaN or Inf", LOAD.read_detection, src)
    finally:
        f.close()


# ------------------------------------------------------- Block 6.3 checks ---

def test_63_contract_known_answer():
    sh = shared()
    c = MODEL.check_contract(sh["inp"].tasks, sh["cfg"])
    assert c["on_index"] == 36, c["on_index"]
    boxes = {tuple(b["box"]): b["n_tasks"] for b in c["on_boxes"]}
    assert boxes == {(0.01, 3.0): 2, (0.03, 3.0): 1}, boxes
    assert c["manifest_version"] == {"3": 3}, c["manifest_version"]
    assert c["single"]["simtime"] == 180.0


def test_63_contract_refusals():
    sh = shared()
    tasks, cfg = list(sh["inp"].tasks), sh["cfg"]
    bad = [with_contract(tasks[0], simtime=200.0)] + tasks[1:]
    expect_error(MODEL.PlanRefused, "differs in simtime", MODEL.check_contract, bad, cfg)
    bad = [with_contract(t, simtime=200.0) for t in tasks]
    expect_error(MODEL.PlanRefused, "the window is 180.0 s", MODEL.check_contract, bad, cfg)
    pb = [list(r) for r in tasks[2].contract["param_bounds"]]
    pb[36] = [0.005, 3.0]
    bad = tasks[:2] + [with_contract(tasks[2], param_bounds=pb)]
    expect_error(MODEL.PlanRefused, "O_N box [0.005, 3.0]", MODEL.check_contract, bad, cfg)
    expect_error(MODEL.PlanRefused, "(D-074)", MODEL.check_contract, bad, cfg)
    pb = [list(r) for r in tasks[2].contract["param_bounds"]]
    pb[20] = [0.5, 2.5]
    bad = tasks[:2] + [with_contract(tasks[2], param_bounds=pb)]
    expect_error(MODEL.PlanRefused, "param_bounds outside the O_N row",
                 MODEL.check_contract, bad, cfg)
    ok = tasks[:2] + [with_contract(tasks[2], manifest_version=4)]   # reported only
    assert MODEL.check_contract(ok, cfg)["manifest_version"] == {"3": 2, "4": 1}


def test_63_label_columns():
    sh = shared()
    lab = sh["lab"]
    assert list(lab.param_names) == FX.PARAM_NAMES[15:37] + ["conn_prob"], lab.param_names
    assert lab.on_column == 21 and lab.bounds_declared.shape == (23, 2)
    assert lab.coord[21] == "ln" and all(c == "linear" for j, c in enumerate(lab.coord)
                                          if j != 21), lab.coord
    # the declared O_N row (D-074); rtol: float64 roundoff of a log
    assert_allclose(lab.bounds_declared[21], [math.log(0.01), math.log(3.0)], rtol=1e-12)
    assert np.array_equal(lab.bounds_declared[:21], np.array([[0.5, 2.0]] * 21))
    assert np.array_equal(lab.bounds_declared[22], [0.05, 0.4])


def test_63_label_refusals():
    sh = shared()
    tasks, cfg, reg, frz = list(sh["inp"].tasks), sh["cfg"], sh["reg"], sh["inp"].freeze
    expect_error(MODEL.PlanRefused, "the profile expects 24", MODEL.label_columns, tasks, reg,
                 dataclasses.replace(cfg, expected_p=24), frz)
    linear = []
    for t in tasks:                       # O_N under one decade: linear by rule (1)
        pb = [list(r) for r in t.contract["param_bounds"]]
        pb[36] = [0.5, 3.0]
        linear.append(with_contract(t, param_bounds=pb, log_params=[]))
    expect_error(MODEL.PlanRefused, "the declared box is ln", MODEL.label_columns, linear,
                 reg, cfg, frz)


def test_63_twins_and_donors():
    sh = shared()
    tasks, names = list(sh["inp"].tasks), sh["names"]
    groups = MODEL.seed_groups(tasks)
    assert len(groups) == 2 and sum(g["twin"] for g in groups) == 1, groups
    twin = [g for g in groups if g["twin"]][0]
    assert twin["tasks"] == names[:2], twin
    dm = MODEL.donor_map(tasks)
    hand = {(t[3], k) for t in FX.TASKS for k in t[4]}      # (seed master, topology)
    assert dm["n_donors"] == len(hand) == 6, dm["n_donors"]
    for k in (0, 1, 2):
        assert dm["by_task"][names[0]][k] == dm["by_task"][names[1]][k]
    assert not set(dm["by_task"][names[2]].values()) & set(dm["by_task"][names[0]].values())
    ids = sorted(v for d in dm["by_task"].values() for v in set(d.values()))
    assert sorted(set(ids)) == list(range(6)), ids
    bad = [tasks[0], dataclasses.replace(tasks[1], contract_sig=tasks[0].contract_sig), tasks[2]]
    expect_error(MODEL.PlanRefused, "replay", MODEL.seed_groups, bad)


def test_63_shard_layout():
    Fake = collections.namedtuple("Fake", "name n_iters")
    lay = MODEL.shard_layout([Fake("a", 250), Fake("b", 100), Fake("c", 1)], 100)
    assert lay["n_shards"] == 5, lay
    sizes = {n: [c["stop"] - c["start"] for c in lay["by_task"][n]] for n in "abc"}
    assert sizes == {"a": [84, 83, 83], "b": [100], "c": [1]}, sizes
    assert [c["shard"] for n in "abc" for c in lay["by_task"][n]] == list(range(5))
    for n, total in (("a", 250), ("b", 100), ("c", 1)):       # brute force: each position once
        covered = [j for c in lay["by_task"][n] for j in range(c["start"], c["stop"])]
        assert covered == list(range(total)), (n, covered[:5])
    assert [c["stop"] - c["start"] for c in
            MODEL.shard_layout([Fake("d", 200)], 100)["by_task"]["d"]] == [100, 100]


def test_63_positions_digest():
    ti = {2: [5, 6], 0: [0, 1]}
    want = hashlib.sha256(json.dumps([[0, [0, 1]], [2, [5, 6]]],
                                     separators=(",", ":")).encode("ascii")).hexdigest()
    assert MODEL.positions_sha256(ti) == want
    assert MODEL.positions_sha256({2: [5, 7], 0: [0, 1]}) != want
    t0 = shared()["inp"].tasks[0]
    assert MODEL.sorted_positions(t0) == [(k, i) for k in sorted(FX.TASKS[0][4])
                                          for i in FX.TASKS[0][4][k]]


# ------------------------------------------------------ Block 6.4a checks ---

def test_64a_rate_known_answer():
    rng = np.random.default_rng(SEED)
    for n in (0, 17, 18, 19, 500):
        det_t = np.sort(rng.uniform(0.5, 179.5, n))
        x = ROW.window_from_detections(det_t, np.zeros(n, dtype=np.int64), 1, 180.0, 0.01, 0.02)
        assert x.shape == (W,) and x.dtype == np.float32, (x.shape, x.dtype)
        r = ROW.mean_rate_hz(x, 0.01)
        if n == 0:
            assert r == 0.0, r
        else:   # rtol: float32 storage of x (6e-8 per sample), float64 sum
            assert_allclose(r, n / 180.0, rtol=1e-6)


def test_64a_floor():
    assert ROW.is_kept(0.1 * (1 - 1e-7), 0.1, 1e-6)
    assert not ROW.is_kept(0.1 * (1 - 1e-7), 0.1, 0.0)
    assert not ROW.is_kept(17 / 180.0, 0.1, 1e-6) and ROW.is_kept(18 / 180.0, 0.1, 1e-6)
    sh = shared()
    t, k, i = FX.TASKS[1], 0, 3
    det = LOAD.read_detection(mea_path(sh["f"].dir, t, k, i))
    spec = MODEL.task_label_spec(sh["inp"].tasks[1], sh["reg"], sh["cfg"], sh["inp"].freeze)
    rng = np.random.default_rng(SEED + 1)
    for n, kept in ((17, False), (18, True)):
        d = dict(det, det_t=np.sort(rng.uniform(0.5, 179.5, n)), det_ch=np.zeros(n, np.int64))
        row = ROW.build_row(d, spec, sh["lab"].bounds_declared, 180.0, 0.01, 0.02, 1, W, 0.1, 1e-6)
        assert row["kept"] is kept, (n, row["rate_hz"])


def test_64a_unit_box():
    b = np.array([[0.0, 1.0], [math.log(0.01), math.log(3.0)], [0.05, 0.4]])
    u, sn = ROW.to_unit_box([0.25, math.log(0.03), 0.4], b)
    assert_allclose(u, [0.25, math.log(3.0) / math.log(300.0), 1.0], rtol=1e-12)
    assert sn == 0
    u, sn = ROW.to_unit_box([-1e-13, 0.0, 0.1], b)
    assert u[0] == 0.0 and sn == 1, (u, sn)
    expect_error(ROW.RowError, "column(s) [0]", ROW.to_unit_box, [-1e-6, 0.0, 0.1], b)
    expect_error(ROW.RowError, "outside the declared box", ROW.to_unit_box, [np.nan, 0.0, 0.1], b)


def test_64a_label_matches_hand():
    sh = shared()
    t, k, i = FX.TASKS[1], 0, 3
    lab = sh["lab"]
    det = LOAD.read_detection(mea_path(sh["f"].dir, t, k, i))
    spec = MODEL.task_label_spec(sh["inp"].tasks[1], sh["reg"], sh["cfg"], sh["inp"].freeze)
    row = ROW.build_row(det, spec, lab.bounds_declared, 180.0, 0.01, 0.02, 1, W, 0.1, 1e-6)
    truth = sh["f"].fx["truth"][(FX.task_name(t), k, i)]
    theta_a = np.concatenate([truth["theta"][15:37], [truth["conn_prob"]]])
    lo, hi = lab.bounds_declared[:, 0], lab.bounds_declared[:, 1]
    assert_allclose(row["theta_a"], theta_a, rtol=1e-12)
    assert_allclose(row["u"], (theta_a - lo) / (hi - lo), rtol=1e-12)
    assert row["kept"] == (truth["n"] >= 18) and row["n_det"] == truth["n"]


def test_64a_refusals():
    sh = shared()
    t = FX.TASKS[1]
    det = LOAD.read_detection(mea_path(sh["f"].dir, t, 1, 2))      # 500 spikes
    assert det["det_t"].size > 0
    spec = MODEL.task_label_spec(sh["inp"].tasks[1], sh["reg"], sh["cfg"], sh["inp"].freeze)
    b = sh["lab"].bounds_declared
    expect_error(ROW.RowError, "D-043", ROW.build_row, dict(det, n_e=2), spec, b, 180.0, 0.01,
                 0.02, 1, W, 0.1, 1e-6)
    expect_error(ROW.RowError, "D-041", ROW.build_row, det, spec, b, 170.0, 0.01, 0.02, 1, W,
                 0.1, 1e-6)
    expect_error(ROW.RowError, "det_ch outside", ROW.build_row,
                 dict(det, det_ch=np.ones_like(det["det_ch"])), spec, b, 180.0, 0.01, 0.02, 1,
                 W, 0.1, 1e-6)


# ------------------------------------------------------- Block 6.4 checks ---

def test_64_draws():
    sh = shared()
    tasks, cfg, names = list(sh["inp"].tasks), sh["cfg"], sh["names"]
    groups = MODEL.seed_groups(tasks)
    d1 = SAMPLE.draw_iterations(tasks, groups, cfg)
    assert d1 == SAMPLE.draw_iterations(tasks, groups, cfg)           # determinism
    for k in (0, 1):                     # common indices: one draw for both twins
        assert d1[(names[0], k)]["i"] == d1[(names[1], k)]["i"], k
        assert d1[(names[0], k)]["comparable"] and d1[(names[1], k)]["comparable"]
    assert not d1[(names[0], 2)]["comparable"] and not d1[(names[1], 2)]["comparable"]
    assert d1[(names[0], 2)]["i"] in (5, 6) and d1[(names[1], 2)]["i"] in (0, 1)
    twin = [g for g in groups if g["twin"]][0]
    for k in (0, 1):                     # independent recomputation of the draw
        key = "%s|%d" % (twin["seed_key"], k)
        rng = np.random.default_rng(np.random.SeedSequence(
            [cfg.sample_seed, zlib.crc32(key.encode("ascii"))]))
        common = sorted(set(FX.TASKS[0][4][k]) & set(FX.TASKS[1][4][k]))
        assert d1[(names[0], k)]["i"] == common[int(rng.integers(0, len(common)))]
    key = "%s|%d" % (names[1], 2)        # the disjoint topology: keyed per task
    rng = np.random.default_rng(np.random.SeedSequence(
        [cfg.sample_seed, zlib.crc32(key.encode("ascii"))]))
    assert d1[(names[1], 2)]["i"] == FX.TASKS[1][4][2][int(rng.integers(0, 2))]


def test_64_kept_share_known_answer():
    sh = shared()
    samples, groups, draws = all_samples(sh)
    dm = MODEL.donor_map(sh["inp"].tasks)
    donors_of = {(t.name, k): dm["by_task"][t.name][k] for t in sh["inp"].tasks
                 for k in t.topo_iters}
    num = den = 0.0
    for t in FX.TASKS:                   # by hand: kept iff the file holds >= 18 spikes
        for k, its in t[4].items():
            i = draws[(FX.task_name(t), k)]["i"]
            den += len(its)
            num += len(its) * (FX.spikes_of(t, k, i) >= 18)
    share = SAMPLE.kept_share(samples, donors_of, sh["cfg"])
    assert abs(share["p_hat"] - num / den) < 1e-12, (share["p_hat"], num / den)
    assert share["ci95"][0] <= share["p_hat"] <= share["ci95"][1], share
    assert share["n_donors"] == 6 and share["n_topologies"] == 9, share
    assert [s["kept"] for s in samples] == [
        FX.spikes_of(t, k, draws[(FX.task_name(t), k)]["i"]) >= 18
        for t in FX.TASKS for k in sorted(t[4])]


def test_64_bootstrap_by_donor():
    samples = [{"task": "a", "k": 0, "n": 1, "kept": True, "rate_hz": 1.0},
               {"task": "a", "k": 1, "n": 1, "kept": True, "rate_hz": 1.0},
               {"task": "b", "k": 0, "n": 1, "kept": False, "rate_hz": 0.0}]
    donors_of = {("a", 0): 7, ("a", 1): 7, ("b", 0): 9}
    uniq, num, den = SAMPLE.donor_totals(samples, donors_of)
    assert list(uniq) == [7, 9] and list(num) == [2.0, 0.0] and list(den) == [2.0, 1.0]
    reps = SAMPLE.bootstrap_reps(num, den, 400, 3)
    vals = set(np.round(reps, 12).tolist())
    # {A, A} -> 1, {A, B} -> 2/3, {B, B} -> 0; over topologies 1/3 and 1/2 would occur.
    # Each value has probability >= 1/4 per resample: all three appear in 400.
    assert vals == {0.0, round(2.0 / 3.0, 12), 1.0}, vals


def test_64_twin_check():
    sh = shared()
    samples, groups, _ = all_samples(sh)
    tc = SAMPLE.twin_check(groups, samples)
    assert tc == [{"tasks": sh["names"][:2], "shared_topologies": 3, "conn_prob_equal": 1,
                   "seed_run_comparable": 2, "seed_run_equal": 2}], tc


def test_64_bytes_per_row():
    zeros = [np.zeros(W, np.float32)] * 20
    assert SAMPLE.bytes_per_row(zeros) < 1000.0
    rng = np.random.default_rng(SEED)
    noise = [rng.uniform(0.0, 1.0, W).astype(np.float32) for _ in range(5)]
    assert SAMPLE.bytes_per_row(noise) > 60000.0         # raw: 4 * 18000 = 72,000


def test_64_sample_refusal():
    f = Fixture()
    try:
        cfg = f.cfg()
        inp = LOAD.load_inputs(cfg)
        reg = SL.load_registry(cfg.sim_dir)
        lab = MODEL.label_columns(inp.tasks, reg, cfg, inp.freeze)
        t = FX.TASKS[0]
        rewrite_npz(raw_path(f.dir, t, 0, 0), seed_run=np.int64(FX.seed_run_of(t, 0, 0) + 1))
        spec = MODEL.task_label_spec(inp.tasks[0], reg, cfg, inp.freeze)
        expect_error(LOAD.LoadError, "the raw file", SAMPLE.sample_topology, inp.tasks[0], 0, 0,
                     inp, cfg, spec, lab.bounds_declared)
        space = SAMPLE.free_space(os.path.join(f.dir, "no", "such", "folder"))
        assert space["path_checked"] == f.dir and space["free_bytes"] > 0, space
    finally:
        f.close()


# ------------------------------------------------------- Block 6.5 checks ---

def test_65_plan_end_to_end():
    f = Fixture()
    try:
        rc, out = run_main(PLAN.main, ["--profile", f.profile])
        assert rc == 0 and out[-1] == "BANK PLAN OK", out[-3:]
        cfg = f.cfg()
        with open(cfg.plan_out + ".sha256") as fh:
            assert fh.read().split()[0] == sha256_of(cfg.plan_out)
        with open(cfg.plan_out) as fh:
            plan = json.load(fh)
        assert (plan["n_rows"], plan["n_donors"], plan["n_shards"]) == (32, 6, 3), plan["n_rows"]
        assert [t["batch"] for t in plan["tasks"]] == [1, 2, 5]
        assert [t["on_box"] for t in plan["tasks"]] == [[0.03, 3.0], [0.01, 3.0], [0.01, 3.0]]
        for t, want in zip(plan["tasks"], FX.TASKS):
            assert t["positions_sha256"] == MODEL.positions_sha256(want[4])
        assert plan["inputs"]["record_sha256"] == sha256_of(f.fx["record"])
        assert plan["inputs"]["freeze_sha256"] == sha256_of(f.fx["freeze"])
        assert plan["inputs"]["cohort_manifest_sha256"] == sha256_of(f.fx["manifest"])
        rows = plan["sample"]["rows"]
        assert len(rows) == 9
        by = {FX.task_name(t): t for t in FX.TASKS}
        n_of = {(r[0], r[1]): len(by[r[0]][4][r[1]]) for r in rows}
        p = (sum(n_of[(r[0], r[1])] * (FX.spikes_of(by[r[0]], r[1], r[2]) >= 18) for r in rows)
             / float(sum(n_of.values())))
        assert abs(plan["sample"]["kept_share"]["p_hat"] - p) < 1e-12
        assert_allclose(plan["label"]["bounds_declared"][21], [math.log(0.01), math.log(3.0)],
                        rtol=1e-12)
    finally:
        f.close()


def test_65_plan_refusals():
    cases = [({"expected_p": 24}, "the profile expects 24"),
             ({"n_e": 2}, "electrodes_per_subset is 1, the profile's n_e 2"),
             ({"t_win_s": 175.0}, "the window is 175.0 s")]
    for overrides, fragment in cases:
        f = Fixture(**overrides)
        try:
            rc, out = run_main(PLAN.main, ["--profile", f.profile])
            assert rc == 1 and out[-1].startswith("REFUSED:") and fragment in out[-1], out[-1:]
            assert not os.path.exists(f.cfg().plan_out), overrides
        finally:
            f.close()
    f = Fixture()
    try:                                  # a cohort manifest of nine electrodes per subset (D-043)
        with open(f.fx["manifest"]) as fh:
            man = json.load(fh)
        man["preprocessing"]["electrodes_per_subset"] = 9
        man.pop("_digest", None)
        digest = SP_CM.write_manifest(man, f.fx["manifest"])
        FX.write_profile(f.fx, f.profile, cohort_manifest_sha256_prefix=digest[:16])
        rc, out = run_main(PLAN.main, ["--profile", f.profile])
        assert rc == 1 and "electrodes_per_subset is 9" in out[-1], out[-1:]
        assert not os.path.exists(f.cfg().plan_out)
    finally:
        f.close()
    f = Fixture()
    try:                                  # the paired mode's seed_run -1, on every file of one topology
        t = FX.TASKS[2]
        for i in t[4][1]:
            rewrite_npz(mea_path(f.dir, t, 1, i), seed_run=np.int64(-1))
            rewrite_npz(raw_path(f.dir, t, 1, i), seed_run=np.int64(-1))
        rc, out = run_main(PLAN.main, ["--profile", f.profile])
        assert rc == 1 and "seed_run < 0" in out[-1], out[-1:]
        assert not os.path.exists(f.cfg().plan_out)
        expect_error(MODEL.PlanRefused, "_v<number>", PLAN.campaign_version, "campaign_x")
    finally:
        f.close()


def test_65_determinism():
    f = Fixture()
    try:
        plans = []
        for name in ("a.json", "b.json"):
            out = os.path.join(f.dir, name)
            rc, lines = run_main(PLAN.main, ["--profile", f.profile, "--out", out])
            assert rc == 0, lines[-1:]
            with open(out) as fh:
                plans.append(json.load(fh))
        for key in ("sample", "estimates", "tasks", "label", "contract", "seed_groups"):
            assert plans[0][key] == plans[1][key], key
    finally:
        f.close()


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_determinism():
    raise unittest.SkipTest("checked where the randomness is: test_64_draws (the draws) "
                            "and test_65_determinism (the whole plan)")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
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
