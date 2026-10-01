#!/usr/bin/env python3
"""
smoke_test_raster.py -- the raster tool against a synthetic cohort laid out
exactly like davinci's.

The fixture, built in a temporary folder:
  Deep_bio/DATA_C/Batch3/ptrain_*/ptrain_<k>.mat        raw 3Brain-format rasters
  Deep_bio/DATA_C/Batch4/SubBatch1/ptrain_*/...          (uint8 (n, 1), variable
  Deep_bio/DATA_P/Batch4/ptrain_*/...                     "ptrain", scipy v5 .mat)
  Deep_bio/DATA_P/Batch3/SubBatch1/ptrain_*/...
  Deep_bio/extracted_v2/<class>/<root_name>/<well>/traces_meta.json
  Deep_bio/extracted_v2/cohort_manifest.json (+ .sha256, by write_manifest)
12 wells on a 6 x 6 grid, index_base 1, fs_raw 1000 Hz, 20 s, mfr_threshold
0.1 Hz (an electrode is active iff it has >= 2 spikes; one well has an
electrode with exactly 2, the boundary). Every spike is planted and known,
so every check below compares against an INDEPENDENT computation from the
planted spikes, never against the tool's own output.

Checks (one line each):
  R1  fixture: the manifest passes cohort_manifest.read_manifest (sidecar)
  R2  load_cohort: class, batch, n_active from the records; a manifest with
      another layout, and a record with another fs_raw, are refused
  R3  D-026: lower median of n_active per batch folder, ties by culture id,
      batch folders in raw-path order; --wells names; an unknown name refused
  R4  spike table == the planted spikes, sorted by time; present, counts and
      active set exact; parity with the extractor (load_ptrain_folder,
      mean_firing_rates, partition_subregions' discarded)
  R5  check_against_record refuses a wrong raster length, a wrong n_present
      and a discarded set that differs
  R6  cache round trip bit-identical; cache_is_current tells kept from stale
  R7  windows: [t0, t1) half-open at both edges; t1 = T_rec accepted, the
      last sample included; windows outside the recording refused
  R8  rows (D-027): only active electrodes, ascending linear index, grid-row
      ticks; silent and sub-threshold electrodes never drawn
  R9  figures: comparison + per-well; every panel draws exactly the planted
      spikes of its active electrodes in the window, at the right rows; one
      raster axes per well and nothing else (D-029: no rate panel); PNG,
      PDF, JSON written; the JSON names the window and the rule
  R10 CLI end to end: select / cache (2 workers) / plot / rerun keeps /
      --force rebuilds / a tampered record is refused (exit 1) / bad window,
      missing manifest, stale wells.tsv and mixed-manifest caches exit 2 /
      view without a display exits 2
  R11 viewer logic without a display: clamp_window; slider, boxes, keys and
      toolbar zoom move every panel; Save writes the same PNG bytes as
      `plot --window` for the same span (D-028)
  R12 byte hygiene: every file in raster/ is ASCII and LF-only

Run (login node, sbi_export, from Sbi-extractor/raster; ~1 min):
    python3 smoke_test_raster.py              # all checks
    python3 smoke_test_raster.py --only R4 R7 # some
    python3 smoke_test_raster.py --keep -v    # keep the fixture, print details
PASS CONDITION, a line that must APPEAR:  [smoke] 12/12 checks passed
It needs the DSN tree to import cohort_manifest, as select and cache do
(../env.sh: SBI_HPC_DIR, default artifacts/sbi_hpc).

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                             # headless here
import scipy.io as sio                                            # noqa: E402

import raster_data as RD                                          # noqa: E402

FS = 1000.0
N_SAMPLES = 20000                     # T_rec = 20.0 s exactly
WIDTH = 6
BASE = 1
THETA = 0.1                           # 2 spikes / 20 s = 0.1 Hz -> active
GRID = list(range(BASE, BASE + WIDTH * WIDTH))
CLI = os.path.join(_HERE, "run_raster_plots.py")
VERBOSE = False

# (class folder, raw parts under Deep_bio, root_name, {well: n_active})
LAYOUT = [
    ("control", ("DATA_C", "Batch3"), "DATA_C_Batch3",
     {"ptrain_A1": 10, "ptrain_A2": 14, "ptrain_B1": 12}),
    ("control", ("DATA_C", "Batch4", "SubBatch1"), "Batch4_SubBatch1",
     {"ptrain_A1": 8, "ptrain_A2": 8, "ptrain_A3": 20, "ptrain_B1": 5}),
    ("pathological", ("DATA_P", "Batch4"), "DATA_P_Batch4",
     {"ptrain_A1": 9, "ptrain_A2": 15}),
    ("pathological", ("DATA_P", "Batch3", "SubBatch1"), "Batch3_SubBatch1",
     {"ptrain_A1": 11, "ptrain_B2": 7, "ptrain_C3": 13}),
]
# D-026 by hand: lower median of (n_active, culture) per batch, batches by raw path
EXPECTED_TYPICAL = ["DATA_C_Batch3__ptrain_B1", "Batch4_SubBatch1__ptrain_A1",
                    "Batch3_SubBatch1__ptrain_A1", "DATA_P_Batch4__ptrain_A1"]
EDGE_WELL = "DATA_C_Batch3__ptrain_B1"     # carries the planted edge spikes


def log(msg):
    if VERBOSE:
        print("        " + msg)


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #
class Fixture:
    """The synthetic cohort and the ground truth it was built from."""

    def __init__(self, root):
        self.root = root
        self.deep_bio = os.path.join(root, "Deep_bio")
        self.extract_root = os.path.join(self.deep_bio, "extracted_v2")
        self.manifest = os.path.join(self.extract_root, "cohort_manifest.json")
        self.truth = {}          # culture -> {k: np.ndarray of samples}  (present electrodes)
        self.meta = {}           # culture -> dict(class, batch, folder, out_dir, ...)
        rng = np.random.default_rng(20261001)
        wells_doc = []
        for cls, raw_parts, root_name, wells in LAYOUT:
            for well, n_active in wells.items():
                culture = "%s__%s" % (root_name, well)
                folder = os.path.join(self.deep_bio, *raw_parts, well)
                os.makedirs(folder)
                absent = set(rng.choice(GRID, size=3, replace=False).tolist())
                present = [k for k in GRID if k not in absent]
                order = rng.permutation(present).tolist()
                active = sorted(order[:n_active])
                sub = sorted(order[n_active:n_active + 2])       # 1 spike each
                spikes = {}
                for k in present:
                    if k in active:
                        n = int(rng.integers(2, 40))
                    elif k in sub:
                        n = 1
                    else:
                        n = 0
                    spikes[k] = np.sort(rng.choice(N_SAMPLES, size=n, replace=False)).astype(np.int64)
                if culture == EDGE_WELL:
                    k_edge, k_two = active[0], active[1]
                    spikes[k_edge] = np.array([0, 5000, 7500, 10000, N_SAMPLES - 1], np.int64)
                    spikes[k_two] = np.array([4000, 15000], np.int64)   # exactly 2: active
                    self.k_edge, self.k_two = k_edge, k_two
                for k in present:
                    r = np.zeros((N_SAMPLES, 1), dtype=np.uint8)
                    r[spikes[k], 0] = 1
                    sio.savemat(os.path.join(folder, "ptrain_%d.mat" % k),
                                {"ptrain": r}, do_compression=True)
                with open(os.path.join(folder, "notes.txt"), "w") as fh:
                    fh.write("not a raster; the loader must ignore it\n")
                discarded = sorted(k for k in present if spikes[k].size / (N_SAMPLES / FS) < THETA)
                assert len(present) - len(discarded) == n_active
                out_dir = os.path.join(self.extract_root, cls, root_name, well)
                os.makedirs(out_dir)
                frag = {"fs_raw": FS, "index_base": BASE, "grid_width": WIDTH,
                        "n_subsets": 1, "electrodes_per_subset": 1,
                        "mfr_threshold": THETA, "w_size": 0.01, "gaussian_window": 0.02,
                        "T_rec": N_SAMPLES / FS, "n_samples_raw": N_SAMPLES,
                        "n_present": len(present), "discarded": discarded,
                        "source_folder": folder, "extractor_version": "fixture/3",
                        "manifest_version": 1, "fs_ifr": 100.0}
                with open(os.path.join(out_dir, "traces_meta.json"), "w") as fh:
                    json.dump(frag, fh, indent=2, sort_keys=True)
                self.truth[culture] = spikes
                self.meta[culture] = {"class": cls, "batch": root_name, "folder": folder,
                                      "out_dir": out_dir, "active": active,
                                      "discarded": discarded, "present": present}
                wells_doc.append({"culture": culture,
                                  "out_dir": "%s/%s/%s" % (cls, root_name, well),
                                  "n_archives": 1, "T_rec": N_SAMPLES / FS,
                                  "n_present": len(present), "n_samples_raw": N_SAMPLES})
        self.doc = {"manifest_version": 1, "extract_root": self.extract_root,
                    "preprocessing": {"fs_raw": FS, "index_base": BASE, "grid_width": WIDTH,
                                      "n_subsets": 1, "electrodes_per_subset": 1,
                                      "mfr_threshold": THETA, "w_size": 0.01,
                                      "gaussian_window": 0.02},
                    "classes": ["control", "pathological"], "wells": wells_doc}
        import cohort_manifest as CM
        CM.write_manifest(self.doc, self.manifest)

    def planted(self, culture, t0, t1, active_only=True):
        """(times, electrodes) of the planted spikes in [t0, t1)."""
        m = self.meta[culture]
        keep = set(m["active"]) if active_only else set(m["present"])
        ts, es = [], []
        for k, s in self.truth[culture].items():
            if k not in keep:
                continue
            sel = s[(s >= t0 * FS) & (s < t1 * FS)]
            ts.append(sel / FS)
            es.append(np.full(sel.size, k))
        return np.concatenate(ts), np.concatenate(es)


def run_cli(args, env_extra=None, drop=()):
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    for k in drop:
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([sys.executable, CLI] + list(args), capture_output=True,
                       text=True, env=env, timeout=600)
    out = p.stdout + p.stderr
    log("$ run_raster_plots.py %s -> %d" % (" ".join(args), p.returncode))
    for line in out.splitlines()[-6:]:
        log("  | " + line)
    return p.returncode, out


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def r1(fx, ctx):
    import cohort_manifest as CM
    doc = CM.read_manifest(fx.manifest)
    assert doc["_digest"] == open(fx.manifest + ".sha256").read().split()[0]
    assert len(doc["wells"]) == 12
    ctx["cohort"] = RD.load_cohort(fx.manifest)
    return "manifest digest %s..., 12 wells" % doc["_digest"][:12]


def r2(fx, ctx):
    import cohort_manifest as CM
    c = ctx["cohort"]
    assert c.classes == ("control", "pathological")
    assert (c.fs_raw, c.index_base, c.grid_width, c.mfr_threshold) == (FS, BASE, WIDTH, THETA)
    for w in c.wells:
        m = fx.meta[w.culture]
        assert w.class_name == m["class"] and w.batch == m["batch"], w
        assert w.n_active == len(m["active"]) and w.discarded == tuple(m["discarded"])
        assert w.source_folder == os.path.normpath(m["folder"])
    # another layout: out_dir with four components -> refused
    bad = json.loads(json.dumps(fx.doc))
    bad["wells"][0]["out_dir"] = "control/extra/DATA_C_Batch3/ptrain_A1"
    p = os.path.join(fx.root, "bad_layout", "cohort_manifest.json")
    CM.write_manifest(bad, p)
    try:
        RD.load_cohort(p)
        raise AssertionError("a 4-part out_dir was accepted")
    except RD.RasterDataError as exc:
        assert "not <class>/<batch>/<well>" in str(exc)
    # a record whose fs_raw differs from the manifest's -> refused
    frag_path = os.path.join(fx.meta["DATA_P_Batch4__ptrain_A2"]["out_dir"], "traces_meta.json")
    orig = open(frag_path).read()
    try:
        d = json.loads(orig)
        d["fs_raw"] = 10110.09
        open(frag_path, "w").write(json.dumps(d))
        try:
            RD.load_cohort(fx.manifest)
            raise AssertionError("a record with another fs_raw was accepted")
        except RD.RasterDataError as exc:
            assert "fs_raw" in str(exc)
    finally:
        open(frag_path, "w").write(orig)
    # no sidecar -> refused
    p2 = os.path.join(fx.root, "no_sidecar", "cohort_manifest.json")
    os.makedirs(os.path.dirname(p2))
    shutil.copy(fx.manifest, p2)
    try:
        RD.load_cohort(p2)
        raise AssertionError("a manifest without sidecar was accepted")
    except RD.RasterDataError as exc:
        assert "sidecar" in str(exc)
    return "12 records read; layout, fs_raw and missing-sidecar refusals fire"


def r3(fx, ctx):
    c = ctx["cohort"]
    got = [w.culture for w in RD.select_typical_wells(c.wells, c.classes)]
    assert got == EXPECTED_TYPICAL, got
    named = RD.select_named_wells(c.wells, ["DATA_P_Batch4__ptrain_A2", "DATA_C_Batch3__ptrain_A1"])
    assert [w.culture for w in named] == ["DATA_P_Batch4__ptrain_A2", "DATA_C_Batch3__ptrain_A1"]
    for bad in (["nope__ptrain_Z9"], ["DATA_C_Batch3__ptrain_A1", "DATA_C_Batch3__ptrain_A1"]):
        try:
            RD.select_named_wells(c.wells, bad)
            raise AssertionError("bad --wells accepted: %r" % bad)
        except RD.RasterDataError:
            pass
    return "chosen %s" % ", ".join(got)


def r4(fx, ctx):
    from channel_subset_extraction import (load_ptrain_folder, mean_firing_rates,
                                           partition_subregions, _rowcol)
    c = ctx["cohort"]
    ctx["tables"] = {}
    for w in c.wells:
        t = RD.build_spike_table(w, c)
        truth = fx.truth[w.culture]
        m = fx.meta[w.culture]
        assert t.present.tolist() == m["present"]
        assert t.n_spikes.tolist() == [truth[k].size for k in m["present"]]
        assert t.active.tolist() == m["active"]
        assert np.all(np.diff(t.sample) >= 0), "not sorted by time"
        for k in m["present"]:
            assert np.array_equal(t.sample[t.electrode == k], truth[k]), (w.culture, k)
        assert t.sample.size == sum(s.size for s in truth.values())
        # parity with the extractor's own functions
        inv = load_ptrain_folder(w.source_folder, fs_raw=FS, index_base=BASE)
        mfrs = mean_firing_rates(inv, inv.T_rec)
        idx = np.asarray(inv.indices)
        rows, cols = _rowcol(idx, WIDTH, BASE)
        coords = {int(i): (int(r), int(cc)) for i, r, cc in zip(idx, rows, cols)}
        _, disc = partition_subregions(coords, mfrs, n_subsets=1,
                                       electrodes_per_subset=1, mfr_threshold=THETA)
        assert disc == m["discarded"] == t.present[~t.active_mask].tolist()
        RD.check_against_record(t, w)
        ctx["tables"][w.culture] = t
    t = ctx["tables"][EDGE_WELL]
    assert fx.k_two in t.active.tolist(), "an electrode with exactly 2 spikes in 20 s must be active"
    return "12 wells exact; boundary electrode (2 spikes = 0.1 Hz) active"


def r5(fx, ctx):
    c = ctx["cohort"]
    w = next(x for x in c.wells if x.culture == EDGE_WELL)
    t = ctx["tables"][EDGE_WELL]
    for field_, value, needle in (
            ("n_samples_raw", N_SAMPLES + 1, "n_samples"),
            ("n_present", w.n_present + 1, "present electrodes"),
            ("discarded", tuple(w.discarded[1:]), "sub-threshold set")):
        bad = dataclasses.replace(w, **{field_: value})
        try:
            RD.check_against_record(t, bad)
            raise AssertionError("tampered %s accepted" % field_)
        except RD.RasterDataError as exc:
            assert needle in str(exc) and "REFUSED" in str(exc), str(exc)
    return "three tamperings refused"


def r6(fx, ctx):
    c = ctx["cohort"]
    w = next(x for x in c.wells if x.culture == EDGE_WELL)
    t = ctx["tables"][EDGE_WELL]
    out = os.path.join(fx.root, "r6")
    p = RD.save_cache(t, RD.cache_path(out, w.culture))
    u = RD.load_cache(p)
    for name in ("sample", "electrode", "present", "n_spikes", "active_mask"):
        a, b = getattr(t, name), getattr(u, name)
        assert a.dtype.kind == b.dtype.kind and np.array_equal(a, b), name
    for name in ("culture", "class_name", "batch", "n_samples", "fs_raw", "index_base",
                 "grid_width", "mfr_threshold", "meta"):
        assert getattr(t, name) == getattr(u, name), name
    ok, why = RD.cache_is_current(p, w, c)
    assert ok, why
    stale = dataclasses.replace(c, manifest_digest="0" * 64)
    ok, why = RD.cache_is_current(p, w, stale)
    assert not ok and "manifest_digest" in why
    ok, why = RD.cache_is_current(p + ".absent", w, c)
    assert not ok and why == "absent"
    return "round trip exact; stale digest detected"


def r7(fx, ctx):
    t = ctx["tables"][EDGE_WELL]
    row_k = t.active.tolist().index(fx.k_edge)

    def has(view, sample):
        return bool(np.any((view.row == row_k) & (np.abs(view.t - sample / FS) < 0.25 / FS)))
    v = RD.raster_view(t, 5.0, 10.0)
    assert has(v, 5000) and has(v, 7500) and not has(v, 10000), "half-open [5, 10) wrong"
    v = RD.raster_view(t, 0.0, N_SAMPLES / FS)
    assert has(v, 0) and has(v, N_SAMPLES - 1), "first/last sample missing"
    v = RD.raster_view(t, 0.0, N_SAMPLES / FS + 0.5 / FS)       # within one sample period
    for a, b in ((-1, 5), (5, 5), (6, 5), (10, 20.5), (float("nan"), 3)):
        try:
            RD.raster_view(t, a, b)
            raise AssertionError("window [%r, %r) accepted" % (a, b))
        except RD.RasterDataError:
            pass
    assert RD.fmt_window(0, 60) == "t0000.0-0060.0s"
    return "edges: 5.000 in, 10.000 out, 0 and 19.999 in; 5 bad windows refused"


def r8(fx, ctx):
    for culture, t in ctx["tables"].items():
        m = fx.meta[culture]
        v = RD.raster_view(t, 0.0, 20.0)
        assert v.n_rows == len(m["active"])
        rank = {k: i for i, k in enumerate(m["active"])}            # ascending k
        ts, es = fx.planted(culture, 0.0, 20.0)
        exp = sorted(zip(ts.tolist(), [rank[int(e)] for e in es]))
        got = sorted(zip(v.t.tolist(), v.row.tolist()))
        assert len(exp) == len(got) and np.allclose(np.array(exp), np.array(got)), culture
        silent = [k for k in m["present"] if k not in m["active"]]
        assert all(k not in rank for k in silent)
        grid_rows = [(k - BASE) // WIDTH for k in m["active"]]
        for tick, lab in zip(v.row_ticks, v.row_tick_labels):
            assert grid_rows[tick] == int(lab) and (tick == 0 or grid_rows[tick - 1] < int(lab))
    return "rows = active electrodes in linear order, every spike at its row"


def r9(fx, ctx):
    import raster_plot as RP
    c = ctx["cohort"]
    sel = [ctx["tables"][x] for x in EXPECTED_TYPICAL]
    views = [RD.raster_view(t, 2.0, 12.0) for t in sel]
    vbc = RD.group_by_class(views, list(c.classes))
    fig = RP.figure_compare(vbc, list(c.classes), THETA, dpi=120)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == len(sel), "expected one raster axes per well, got %d" % len(visible)
    for ax in visible:
        assert len(ax.lines) == 1, "a panel holds more than its raster"
    for (cls, vs) in vbc.items():
        for i, v in enumerate(vs):
            line = None
            for ax in visible:
                for txt in ax.texts:
                    if txt.get_text() == v.culture:
                        line = ax.lines[0]
            assert line is not None, v.culture
            x, y = line.get_data()
            m = fx.meta[v.culture]
            rank = {k: j for j, k in enumerate(m["active"])}
            ts, es = fx.planted(v.culture, 2.0, 12.0)
            exp = sorted(zip(ts.tolist(), [rank[int(e)] for e in es]))
            assert sorted(zip(np.asarray(x).tolist(), np.asarray(y).tolist())) == \
                sorted((a, b) for a, b in exp), v.culture
    out = os.path.join(fx.root, "r9")
    side = RD.figure_sidecar(sel, vbc, 2.0, 12.0, [], {"manifest": fx.manifest}, how="test")
    files = RP.save_figure(fig, os.path.join(out, "cmp"), ("png", "pdf"), side)
    for f in files:
        assert os.path.getsize(f) > 0, f
    doc = json.load(open(os.path.join(out, "cmp.json")))
    assert doc["window_s"] == [2.0, 12.0] and doc["decisions"]["rows"] == "D-027"
    assert [w["culture"] for w in doc["wells"]] == EXPECTED_TYPICAL
    one = RP.figure_single(views[0], list(c.classes), THETA, dpi=120)
    assert len([a for a in one.axes if a.get_visible()]) == 1
    # opacity: sparse windows opaque; dense ones one shared alpha = 1 / lambda_max
    fig, lay = RP.figure_compare(vbc, list(c.classes), THETA, dpi=120, return_layout=True)
    assert lay.alpha == 1.0 and all(l.get_alpha() == 1.0 for l in lay.lines.values())
    rng = np.random.default_rng(1)

    def fake(name, cls, n):
        return RD.RasterView(culture=name, class_name=cls, batch="b", t0=0.0, t1=10.0,
                             t=np.sort(rng.uniform(0, 10, n)), row=rng.integers(0, 100, n),
                             n_rows=100, n_present=120, row_ticks=(0,), row_tick_labels=("0",))
    dense = {"control": [fake("a", "control", 50000)],
             "pathological": [fake("b", "pathological", 200000)]}
    fig, lay = RP.figure_compare(dense, list(c.classes), THETA, dpi=100, return_layout=True)
    w_px, h_px = lay.panel_px
    lam_max = 200000 / (min(100, h_px) * w_px)
    assert abs(lay.alpha - 1.0 / lam_max) < 1e-12 and lay.alpha < 1.0
    assert len({l.get_alpha() for l in lay.lines.values()}) == 1, "alpha must be shared"
    return "%d panels exact; PNG/PDF/JSON written; alpha 1 sparse, %.3f shared when dense" % (
        len(sel), lay.alpha)


def r10(fx, ctx):
    out = os.path.join(fx.root, "cli")
    common = ["--out-root", out]
    rc, o = run_cli(["select", "--manifest", fx.manifest] + common)
    assert rc == 0 and "[raster] selected 4 well(s)" in o, o
    rows, header = RD.read_wells_tsv(os.path.join(out, "wells.tsv"))
    assert [r["culture"] for r in rows] == EXPECTED_TYPICAL
    rc, o = run_cli(["cache", "--manifest", fx.manifest, "--workers", "2"] + common)
    assert rc == 0 and "[raster] cache: 4/4 well(s) ok" in o and o.count("cache built") == 4, o
    rc, o = run_cli(["plot", "--window", "0", "10", "--window", "5", "20", "--per-well",
                     "--dpi", "100"] + common)
    assert rc == 0 and "[raster] plot: wrote 30 file(s) for 2 window(s)" in o, o
    per = os.listdir(os.path.join(out, "figures", "per_well"))
    assert len(per) == 24 and all(f.split("_t0")[0] in EXPECTED_TYPICAL for f in per), per
    for stamp in ("t0000.0-0010.0s", "t0005.0-0020.0s"):
        for ext in ("png", "pdf", "json"):
            assert os.path.isfile(os.path.join(out, "figures", "raster_compare_%s.%s" % (stamp, ext)))
    rc, o = run_cli(["cache", "--manifest", fx.manifest] + common)
    assert rc == 0 and o.count("cache kept") == 4, o
    rc, o = run_cli(["cache", "--manifest", fx.manifest, "--force"] + common)
    assert rc == 0 and o.count("cache built") == 4, o
    # bad inputs -> exit 2
    rc, o = run_cli(["plot", "--window", "15", "25"] + common)
    assert rc == 2 and "not inside the recording" in o, o
    rc, o = run_cli(["select", "--manifest", os.path.join(fx.root, "absent.json")] + common)
    assert rc == 2 and "cohort manifest not found" in o, o
    rc, o = run_cli(["view"] + common, drop=("DISPLAY", "WAYLAND_DISPLAY"))
    assert rc == 2 and "needs a display" in o, o
    # a record that disagrees with its raw folder -> that well refused, exit 1
    tam = os.path.join(fx.root, "tampered")
    shutil.copytree(fx.root, tam, ignore=shutil.ignore_patterns("cli", "r6", "r9", "tampered"))
    tdoc = json.loads(json.dumps(fx.doc))
    tdoc["extract_root"] = os.path.join(tam, "Deep_bio", "extracted_v2")
    import cohort_manifest as CM
    tman = os.path.join(tam, "Deep_bio", "extracted_v2", "cohort_manifest.json")
    CM.write_manifest(tdoc, tman)
    fp = os.path.join(tam, "Deep_bio", "extracted_v2", "control", "DATA_C_Batch3",
                      "ptrain_B1", "traces_meta.json")
    d = json.load(open(fp))
    d["source_folder"] = d["source_folder"].replace(fx.root, tam)
    d["discarded"] = d["discarded"][1:]
    d["n_present"] = d["n_present"]                                   # unchanged
    json.dump(d, open(fp, "w"))
    # point every other record at the copied raw folders as well
    for root_, _dirs, files in os.walk(os.path.join(tam, "Deep_bio", "extracted_v2")):
        if "traces_meta.json" in files and root_ != os.path.dirname(fp):
            q = os.path.join(root_, "traces_meta.json")
            e = json.load(open(q))
            e["source_folder"] = e["source_folder"].replace(fx.root, tam)
            json.dump(e, open(q, "w"))
    tout = os.path.join(fx.root, "cli_tampered")
    rc, o = run_cli(["cache", "--manifest", tman, "--out-root", tout])
    assert rc == 1 and "REFUSED" in o and "[raster] cache: 3/4 well(s) ok" in o, o
    # the manifest changes: the old wells.tsv is stale -> exit 2 until select again;
    # then the old caches are from another manifest -> plot exit 2 until cache again
    doc2 = json.loads(json.dumps(fx.doc))
    doc2["note"] = "changed"
    CM.write_manifest(doc2, fx.manifest)
    try:
        rc, o = run_cli(["cache", "--manifest", fx.manifest] + common)
        assert rc == 2 and "written from another manifest" in o, o
        rc, o = run_cli(["select", "--manifest", fx.manifest] + common)
        assert rc == 0, o
        rc, o = run_cli(["plot"] + common)
        assert rc == 2 and "another manifest" in o, o
        rc, o = run_cli(["cache", "--manifest", fx.manifest] + common)
        assert rc == 0 and o.count("cache built") == 4, o
        rc, o = run_cli(["plot", "--dpi", "100"] + common)       # default window
        assert rc == 0 and "wrote 3 file(s) for 1 window(s)" in o, o
        assert os.path.isfile(os.path.join(out, "figures", "raster_compare_t0000.0-0020.0s.png"))
    finally:
        CM.write_manifest(fx.doc, fx.manifest)
    ctx["cli_out"] = out
    return "select/cache/plot/keep/force ok; refusals and exit codes as documented"


def r11(fx, ctx):
    import matplotlib.pyplot as plt                               # Agg here
    from matplotlib.backend_bases import KeyEvent
    import raster_viewer as RV
    assert RV.clamp_window(-3, 2, 20) == (0.0, 5.0)
    assert RV.clamp_window(18, 25, 20) == (13.0, 20.0)
    assert RV.clamp_window(5, 5, 20) == (5.0 - RV.MIN_WIDTH_S / 2, 5.0 + RV.MIN_WIDTH_S / 2)
    assert RV.clamp_window(0, 100, 20) == (0.0, 20.0)
    out = ctx.get("cli_out") or os.path.join(fx.root, "cli")
    rows, header = RD.read_wells_tsv(os.path.join(out, "wells.tsv"))
    tables = [RD.load_cache(RD.cache_path(out, r["culture"])) for r in rows]
    classes = header["classes"].split(",")
    vout = os.path.join(fx.root, "viewer_out")
    v = RV.RasterViewer(tables, classes, 2.0, 6.0, vout, rows, header,
                        formats=("png",), dpi=100.0)
    fig = plt.figure(figsize=v.figure_size())
    v.build(fig)
    assert v.timer is None, "no event loop here: windows must apply at once"

    def check(a, b):
        assert np.allclose((v.t0, v.t1), (a, b)), (v.t0, v.t1, a, b)
        for (cls, i), line in v.layout.lines.items():
            t = next(tb for tb in tables if tb.culture == v.layout.axes[(cls, i)].texts[0].get_text())
            ref = RD.raster_view(t, a, b)
            x, y = line.get_data()
            assert np.array_equal(np.asarray(x), ref.t) and np.array_equal(np.asarray(y), ref.row)
        for ax in v.layout.axes.values():
            assert np.allclose(ax.get_xlim(), (a, b))
        assert np.allclose(v.slider.val, (a, b))
        assert float(v.box0.text) == round(a, 3) and float(v.box1.text) == round(b, 3)

    check(2.0, 6.0)
    v.slider.set_val((4.0, 9.0)); check(4.0, 9.0)
    v.box1.set_val("11"); v.on_text(None, "11"); check(4.0, 11.0)
    v.on_key(KeyEvent("key_press_event", fig.canvas, "right")); check(7.5, 14.5)
    v.on_key(KeyEvent("key_press_event", fig.canvas, "up")); check(9.25, 12.75)
    v.on_key(KeyEvent("key_press_event", fig.canvas, "down")); check(7.5, 14.5)
    v.on_key(KeyEvent("key_press_event", fig.canvas, "right")); check(11.0, 18.0)
    v.on_key(KeyEvent("key_press_event", fig.canvas, "right")); check(13.0, 20.0)
    v.layout.first_axes().set_xlim(1.0, 3.0); check(1.0, 3.0)        # toolbar zoom
    v.on_key(KeyEvent("key_press_event", fig.canvas, "home")); check(2.0, 6.0)
    v.apply(5.0, 15.0)
    saved = v.save()
    png = [p for p in saved if p.endswith(".png")][0]
    plt.close(fig)
    # the same window through `plot`, into another folder: same PNG bytes
    pout = os.path.join(fx.root, "plot_same")
    shutil.copytree(out, pout, ignore=shutil.ignore_patterns("figures"))
    rc, o = run_cli(["plot", "--window", "5", "15", "--formats", "png", "--dpi", "100",
                     "--out-root", pout])
    assert rc == 0, o
    ref = os.path.join(pout, "figures", "raster_compare_t0005.0-0015.0s.png")
    h1 = hashlib.sha256(open(png, "rb").read()).hexdigest()
    h2 = hashlib.sha256(open(ref, "rb").read()).hexdigest()
    assert h1 == h2, "viewer Save and plot differ for the same window"
    return "slider/boxes/keys/toolbar move all panels; Save == plot (sha256 %s...)" % h1[:12]


def r12(fx, ctx):
    bad = []
    for f in sorted(os.listdir(_HERE)):
        if not f.endswith((".py", ".pbs", ".sh", ".md")):
            continue
        b = open(os.path.join(_HERE, f), "rb").read()
        if any(c > 127 for c in b) or b"\r" in b:
            bad.append(f)
    assert not bad, "non-ASCII or CR bytes in %s" % bad
    return "every file ASCII, LF only"


CHECKS = [("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4), ("R5", r5), ("R6", r6),
          ("R7", r7), ("R8", r8), ("R9", r9), ("R10", r10), ("R11", r11), ("R12", r12)]
NEEDS = {"R2": ["R1"], "R3": ["R1"], "R4": ["R1"], "R5": ["R1", "R4"], "R6": ["R1", "R4"],
         "R7": ["R1", "R4"], "R8": ["R1", "R4"], "R9": ["R1", "R4"], "R11": ["R10"]}


def main(argv=None):
    global VERBOSE
    ap = argparse.ArgumentParser(description="raster tool smoke suite")
    ap.add_argument("--only", nargs="+", help="run these checks (and what they need)")
    ap.add_argument("--keep", action="store_true", help="keep the fixture folder")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose
    wanted = [c for c, _ in CHECKS] if not args.only else list(args.only)
    todo = []
    for c in wanted:
        for d in NEEDS.get(c, []) + [c]:
            for dd in NEEDS.get(d, []):
                if dd not in todo:
                    todo.append(dd)
            if d not in todo:
                todo.append(d)
    order = [c for c, _ in CHECKS if c in todo]
    root = tempfile.mkdtemp(prefix="raster_smoke_")
    print("[smoke] python %s, numpy %s, matplotlib %s; fixture %s"
          % (sys.version.split()[0], np.__version__, matplotlib.__version__, root))
    t0 = time.time()
    try:
        fx = Fixture(root)
    except Exception:                                             # noqa: BLE001
        traceback.print_exc()
        print("[smoke] FIXTURE FAILED -- if the error names the DSN tree, "
              "source ../env.sh (SBI_HPC_DIR) and rerun")
        return 1
    ctx, passed, failed = {}, 0, []
    fn = dict(CHECKS)
    for c in order:
        t = time.time()
        try:
            msg = fn[c](fx, ctx)
            passed += 1
            print("[smoke] %-4s PASS  %-70s (%.1f s)" % (c, msg, time.time() - t))
        except Exception as exc:                                  # noqa: BLE001
            failed.append(c)
            print("[smoke] %-4s FAIL  %s: %s" % (c, type(exc).__name__, exc))
            if VERBOSE:
                traceback.print_exc()
    if args.keep:
        print("[smoke] fixture kept: %s" % root)
    else:
        shutil.rmtree(root, ignore_errors=True)
    print("[smoke] %d/%d checks passed in %.0f s%s"
          % (passed, len(order), time.time() - t0,
             "" if not failed else "  -- FAILED: " + " ".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
