#!/usr/bin/env python3
"""Smoke test -- the declared ptrain file formats, the probe, the exclusion
list and the per-cohort launcher (2026-10-01, the Giulia cohort).

What is covered, each on SYNTHETIC files whose answers are known exactly:

  F1  the default name pattern and parse_ptrain_index are unchanged
  F2  a custom pattern parses the Giulia name; a pattern without a group,
      with two groups, or that does not compile is refused
  F3  "raster" reads a dense binary raster as before and REFUSES a sparse
      variable, naming the flag
  F4  "sparse_peaks" reads a scipy.sparse column (and row) with amplitude
      values: spike positions, n_samples, duplicates merged, explicit zeros
      ignored; it REFUSES a dense variable and a 2-D sparse matrix
  F5  load_ptrain_folder on Giulia-named sparse files under "peak_train"
  F6  extract_channel_subsets end to end on a synthetic 60-electrode MCS-style
      well: grid_width 10 / index_base 0 decodes k=12 -> (1,2), k=87 -> (8,7);
      per_region_single with C=9, E=1 gives 9 traces of the right length whose
      centres are the nine most active electrodes
  F7  the wrong decoding (grid_width 8 / index_base 1) is caught by
      validate_grid (GeometryError), not silently accepted
  F8  the CLI runner with the three flags writes a version-4 fragment carrying
      them; a bad pattern is refused before any I/O
  F9  build_extra_flags: byte-identical to the tracked extraction_flags.sh for
      the DUP15HD config; the three flags appended for a Giulia-like cohort,
      and the regex survives the EXTRA_FLAGS double-quote + unquoted word-split
      path of the array job
  F10 CohortConfig refuses a bad ptrain_format, varname, pattern (no group, two
      groups, whitespace, a glob character), and bad exclude_wells
  F11 find_wells(exclude=...) and list_extraction_jobs.py: the excluded well is
      skipped and printed; an exclusion matching no well ABORTS (exit 2)
  F12 cohort_manifest.py over a Giulia-like 3-class cohort (one well excluded)
      records source_format and excluded_wells, n_units = wells x 9; a
      fragment whose recorded format disagrees with the config is refused
  F13 probe_ptrain_tree.py: on the Giulia-like tree it suggests sparse_peaks /
      peak_train / the MCS pattern / grid 10,0 and lists the too-few-active
      well under exclude_wells (exit 0); on a 3Brain-like tree it suggests
      raster / ptrain / the strict pattern; on a mixed tree it exits 1 with
      UNDETERMINED
  F14 launch_stage_d.sh with COHORT_TAG: the first dry run writes the cohort's
      flags file and stops (exit 3, "NEW COHORT"); the second prints both qsub
      lines with -v CHSUB_MANIFEST=out/...,CHSUB_FLAGS=... (exit 0); the
      untagged dry run keeps the DUP15HD names

Run (login node, sbi_export, from Sbi-extractor/extractor):
    python3 smoke_test_ptrain_formats.py; echo "exit=$?"

Needs the DSN tree (../env.sh -> SBI_HPC_DIR) like every extractor suite.
Exit 0 = every check passed; 1 = a failure. Pure ASCII, LF only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import scipy.io as sio
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import channel_subset_extraction as CSE                       # noqa: E402
import cohort_config as CC                                    # noqa: E402

RESULTS = []
GIULIA_PATTERN = r"^ptrain_\d+_DIV\d+_\w+_nbasal_\d{4}_(\d{3})\.mat$"
MCS_CODES = [10 * r + c for r in range(1, 9) for c in range(1, 9)
             if (r, c) not in ((1, 1), (1, 8), (8, 1), (8, 8))]        # 60 codes
FS = 1000.0
N_SAMPLES = 90_000                                              # 90 s at 1 kHz


def ok(label, cond, detail=""):
    RESULTS.append(bool(cond))
    print("[%s] %-72s %s" % ("PASS" if cond else "FAIL", label, detail))


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# synthetic files
# --------------------------------------------------------------------------- #
def write_sparse_peaks(path, varname, n, idx, amps=None, extra=None):
    idx = np.asarray(idx, dtype=np.int64)
    if amps is None:
        amps = -30.0 - 10.0 * np.arange(idx.size)
    m = sparse.csc_matrix((np.asarray(amps, dtype=np.float64),
                           (idx, np.zeros(idx.size, dtype=np.int64))), shape=(n, 1))
    d = {varname: m}
    if extra:
        d.update(extra)
    sio.savemat(path, d, format="5", do_compression=False)


def write_raster(path, varname, n, idx):
    r = np.zeros((n, 1), dtype=np.uint8)
    r[np.asarray(idx, dtype=np.int64), 0] = 1
    sio.savemat(path, {varname: r}, format="5", do_compression=True)


def giulia_name(well_id, cond, code):
    return "ptrain_%d_DIV35_%s_nbasal_0001_%03d.mat" % (well_id, cond, code)


def make_mcs_well(folder, well_id, cond, codes, n_active, rng, n=N_SAMPLES,
                  with_artifact=False):
    """A well with the given codes; the first n_active (by code order) get 50
    spikes (0.56 Hz at 90 s), the next two 2 spikes (below 0.1 Hz), the rest
    none. Returns {code: n_spikes}."""
    os.makedirs(folder, exist_ok=True)
    counts = {}
    for j, code in enumerate(codes):
        if j < n_active:
            k = 50 + j                       # distinct counts: a strict ranking
        elif j < n_active + 2:
            k = 2
        else:
            k = 0
        idx = np.sort(rng.choice(n, size=k, replace=False)) if k else np.zeros(0, np.int64)
        extra = {"artifact": np.zeros((0, 1), dtype=np.float64)} if with_artifact else None
        write_sparse_peaks(os.path.join(folder, giulia_name(well_id, cond, code)),
                           "peak_train", n, idx, extra=extra)
        counts[code] = k
    return counts


def make_3brain_well(folder, rng, n=N_SAMPLES, width=8, base=1, n_active=12):
    os.makedirs(folder, exist_ok=True)
    for i in range(width * width):
        k = i + base
        cnt = 40 if i < n_active else 0
        idx = np.sort(rng.choice(n, size=cnt, replace=False)) if cnt else np.zeros(0, np.int64)
        write_raster(os.path.join(folder, "ptrain_%d.mat" % k), "ptrain", n, idx)


def cohort_json(path, root, extract_root, exclude=(), fmt="sparse_peaks",
                varname="peak_train", pattern=GIULIA_PATTERN):
    cfg = {"cohort": {
        "class_roots": {
            "0": [os.path.join(root, "Batch1", "100N0A_wo"),
                  os.path.join(root, "Batch2", "100N0A_wo")],
            "1": [os.path.join(root, "Batch3", "AraC", "50N50A")],
            "2": [os.path.join(root, "Batch3", "AraC", "70N30A")]},
        "class_names": ["100N0A_wo", "50N50A", "70N30A"],
        "well_glob": "ptrain_*",
        "extract_root": extract_root,
        "extract_layout": "{class_name}/{root_name}/{well}",
        "culture_template": "{root_name}__{well}",
        "n_subsets": 9, "electrodes_per_subset": 1, "fs_raw": FS,
        "grid_width": 10, "index_base": 0, "mfr_threshold": 0.1,
        "w_size": 0.01, "gaussian_window": 0.02,
        "ptrain_format": fmt, "ptrain_varname": varname,
        "ptrain_name_pattern": pattern, "exclude_wells": list(exclude)}}
    with open(path, "w", encoding="ascii") as fh:
        json.dump(cfg, fh, indent=1)
    return path


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def f1_f2_patterns():
    ok("F1a default pattern unchanged", CSE.DEFAULT_PTRAIN_NAME_PATTERN == r"^ptrain_(\d+)\.mat$")
    ok("F1b parse_ptrain_index('ptrain_100.mat') == 100 (old call form)",
       CSE.parse_ptrain_index("ptrain_100.mat") == 100)
    try:
        CSE.parse_ptrain_index("ptrain_38886_DIV35_100N0A_nbasal_0001_014.mat")
        ok("F1c the Giulia name is refused by the default pattern", False)
    except CSE.PtrainLoadError:
        ok("F1c the Giulia name is refused by the default pattern", True)
    ok("F2a the Giulia pattern yields 14 for ..._014.mat",
       CSE.parse_ptrain_index("ptrain_38886_DIV35_100N0A_nbasal_0001_014.mat",
                              GIULIA_PATTERN) == 14)
    for lbl, pat in (("no group", r"^ptrain_\d+\.mat$"),
                     ("two groups", r"^ptrain_(\d+)_(\d+)\.mat$"),
                     ("does not compile", r"^ptrain_(\d+\.mat$")):
        try:
            CSE.compile_name_pattern(pat)
            ok("F2b pattern %s refused" % lbl, False)
        except ValueError:
            ok("F2b pattern %s refused" % lbl, True)


def f3_f4_formats(td):
    rng = np.random.default_rng(3)
    n = 5000
    idx = np.array([10, 20, 20, 4999])
    # F3: raster as before
    p = os.path.join(td, "ptrain_7.mat")
    write_raster(p, "ptrain", n, [10, 20, 4999])
    s, m = CSE.load_ptrain_file(p)
    ok("F3a raster: spikes and n_samples as before", s.tolist() == [10, 20, 4999] and m == n)
    ps = os.path.join(td, "ptrain_8.mat")
    write_sparse_peaks(ps, "ptrain", n, [5, 6])
    try:
        CSE.load_ptrain_file(ps)
        ok("F3b raster refuses a sparse variable, naming the flag", False)
    except CSE.PtrainLoadError as exc:
        ok("F3b raster refuses a sparse variable, naming the flag",
           "sparse_peaks" in str(exc) and "ptrain-format" in str(exc), str(exc)[:60])
    # F4: sparse_peaks
    pg = os.path.join(td, "g.mat")
    write_sparse_peaks(pg, "peak_train", n, idx, amps=[-50.0, -40.0, -45.0, -60.0])
    s, m = CSE.load_ptrain_file(pg, varname="peak_train", fmt="sparse_peaks")
    ok("F4a sparse column: positions sorted, duplicate merged, amplitudes ignored, n_samples",
       s.tolist() == [10, 20, 4999] and m == n, "%s %d" % (s.tolist(), m))
    pr = os.path.join(td, "row.mat")
    mrow = sparse.csr_matrix((np.array([-1.0, -2.0]), (np.zeros(2, np.int64), np.array([3, 7]))),
                             shape=(1, n))
    sio.savemat(pr, {"peak_train": mrow}, format="5")
    s, m = CSE.load_ptrain_file(pr, varname="peak_train", fmt="sparse_peaks")
    ok("F4b sparse ROW (1, n) read the same way", s.tolist() == [3, 7] and m == n)
    pz = os.path.join(td, "zero.mat")
    mz = sparse.csc_matrix((np.array([0.0, -2.0]), (np.array([3, 7]), np.zeros(2, np.int64))),
                           shape=(n, 1))
    sio.savemat(pz, {"peak_train": mz}, format="5")
    s, m = CSE.load_ptrain_file(pz, varname="peak_train", fmt="sparse_peaks")
    ok("F4c an explicitly stored zero is not a spike", s.tolist() == [7])
    pe = os.path.join(td, "empty.mat")
    write_sparse_peaks(pe, "peak_train", n, [])
    s, m = CSE.load_ptrain_file(pe, varname="peak_train", fmt="sparse_peaks")
    ok("F4d a silent electrode (empty sparse) gives no spike and n_samples", s.size == 0 and m == n)
    try:
        CSE.load_ptrain_file(p, varname="ptrain", fmt="sparse_peaks")
        ok("F4e sparse_peaks refuses a dense variable, naming the flag", False)
    except CSE.PtrainLoadError as exc:
        ok("F4e sparse_peaks refuses a dense variable, naming the flag",
           '"raster"' in str(exc) and "ptrain-format" in str(exc))
    p2 = os.path.join(td, "two.mat")
    sio.savemat(p2, {"peak_train": sparse.csc_matrix((5, 5))}, format="5")
    try:
        CSE.load_ptrain_file(p2, varname="peak_train", fmt="sparse_peaks")
        ok("F4f a 2-D sparse matrix is refused", False)
    except CSE.PtrainLoadError:
        ok("F4f a 2-D sparse matrix is refused", True)
    try:
        CSE.load_ptrain_file(pg, varname="ptrain", fmt="sparse_peaks")
        ok("F4g a missing variable names the present ones", False)
    except CSE.PtrainLoadError as exc:
        ok("F4g a missing variable names the present ones", "peak_train" in str(exc))
    try:
        CSE.load_ptrain_file(pg, varname="peak_train", fmt="bogus")
        ok("F4h an unknown format is a ValueError", False)
    except ValueError:
        ok("F4h an unknown format is a ValueError", True)


def f5_f7_well(td):
    rng = np.random.default_rng(5)
    folder = os.path.join(td, "ptrain_38886_DIV35_100N0A_nbasal_0001")
    counts = make_mcs_well(folder, 38886, "100N0A", MCS_CODES, n_active=12, rng=rng,
                           with_artifact=True)
    open(os.path.join(folder, "README.txt"), "w").write("not a train\n")
    inv = CSE.load_ptrain_folder(folder, fs_raw=FS, index_base=0,
                                 name_pattern=GIULIA_PATTERN, varname="peak_train",
                                 fmt="sparse_peaks")
    ok("F5a inventory: 60 electrodes keyed by the MCS code, n_samples, T_rec",
       inv.indices == sorted(MCS_CODES) and inv.n_samples == N_SAMPLES
       and abs(inv.T_rec - 90.0) < 1e-9, "%d keys" % len(inv.indices))
    ok("F5b spike counts per electrode match what was planted",
       all(inv.n_spikes(k) == counts[k] for k in MCS_CODES))
    try:
        CSE.load_ptrain_folder(folder, fs_raw=FS, index_base=0)
        ok("F5c the default pattern finds no file here (refused, not silent)", False)
    except CSE.PtrainLoadError as exc:
        ok("F5c the default pattern finds no file here (refused, not silent)",
           "no file matching" in str(exc))

    traces, fs_ifr, diag = CSE.extract_channel_subsets(
        folder, mode="per_region_single", n_subsets=9, electrodes_per_subset=1,
        mfr_threshold=0.1, fs_raw=FS, index_base=0, grid_width=10,
        w_size=0.01, gaussian_window=0.02, return_diagnostics=True,
        ptrain_name_pattern=GIULIA_PATTERN, ptrain_varname="peak_train",
        ptrain_format="sparse_peaks")
    K = int(round(90.0 * 100.0))
    ok("F6a per_region_single, C=9, E=1: nine (K,) traces at fs_ifr 100 Hz",
       len(traces) == 9 and all(t.ndim == 1 and t.shape[0] == K for t in traces)
       and abs(fs_ifr - 100.0) < 1e-9, "K=%d" % K)
    ok("F6b decoding: k=12 -> (1,2), k=87 -> (8,7), 60 present, none discarded as out of grid",
       diag.coords[12] == (1, 2) and diag.coords[87] == (8, 7) and diag.n_present == 60)
    active_sorted = sorted((k for k in MCS_CODES if counts[k] / 90.0 >= 0.1),
                           key=lambda k: (-counts[k], k))
    ok("F6c the nine centres are the nine most active electrodes (ties to lower index)",
       [s.center for s in diag.subregions] == active_sorted[:9],
       str([s.center for s in diag.subregions]))
    ok("F6d every subregion is exactly one electrode",
       all(len(s.members) == 1 for s in diag.subregions))
    ok("F6e the 48 sub-threshold electrodes are the discarded set",
       sorted(diag.discarded) == sorted(k for k in MCS_CODES if counts[k] / 90.0 < 0.1))
    try:
        CSE.extract_channel_subsets(
            folder, mode="per_region_single", n_subsets=9, electrodes_per_subset=1,
            fs_raw=FS, index_base=1, grid_width=8, w_size=0.01, gaussian_window=0.02,
            ptrain_name_pattern=GIULIA_PATTERN, ptrain_varname="peak_train",
            ptrain_format="sparse_peaks")
        ok("F7 grid_width 8 / index_base 1 on MCS codes -> GeometryError", False)
    except CSE.GeometryError:
        ok("F7 grid_width 8 / index_base 1 on MCS codes -> GeometryError", True)
    return folder


def f8_cli(td, folder):
    out = os.path.join(td, "cli_out")
    rc, log = run([sys.executable, os.path.join(_HERE, "run_channel_subset_extraction.py"),
                   folder, "--out-dir", out, "--mode", "per_region_single", "--no-plots",
                   "--fs-raw", str(FS), "--base", "0", "--grid-width", "10",
                   "--n-subsets", "9", "--electrodes-per-subset", "1",
                   "--mfr-threshold", "0.1", "--w-size", "0.01", "--gaussian-window", "0.02",
                   "--ptrain-format", "sparse_peaks", "--ptrain-varname", "peak_train",
                   "--ptrain-name-pattern", GIULIA_PATTERN], cwd=_HERE)
    ok("F8a the CLI runs with the three flags", rc == 0, log.strip().splitlines()[-1][:70] if log.strip() else "")
    meta = json.load(open(os.path.join(out, "traces_meta.json")))
    ok("F8b the fragment is version 4 and carries the three fields",
       meta.get("extractor_version") == "run_channel_subset_extraction/4"
       and meta.get("ptrain_format") == "sparse_peaks"
       and meta.get("ptrain_varname") == "peak_train"
       and meta.get("ptrain_name_pattern") == GIULIA_PATTERN)
    subs = sorted(f for f in os.listdir(out) if f.startswith("trace_subregion_"))
    ok("F8c nine subregion archives written", len(subs) == 9)
    d = np.load(os.path.join(out, subs[0]))
    ok("F8d each archive carries the three fields too",
       str(d["ptrain_format"]) == "sparse_peaks" and str(d["ptrain_varname"]) == "peak_train")
    rc, log = run([sys.executable, os.path.join(_HERE, "run_channel_subset_extraction.py"),
                   folder, "--out-dir", os.path.join(td, "cli_bad"), "--no-plots",
                   "--ptrain-name-pattern", r"^ptrain_\d+\.mat$"], cwd=_HERE)
    ok("F8e a pattern without a capture group is refused before any I/O",
       rc != 0 and "capture group" in log and not os.path.isdir(os.path.join(td, "cli_bad")))


def f9_flags(td):
    dup = os.path.join(_ROOT, "artifacts", "sbi_hpc", "dsn", "hpc", "Config",
                       "config_mea_joint_full.davinci.json")
    if os.path.isfile(dup):
        c = CC.load_cohort(dup)
        tracked = open(os.path.join(_HERE, "extraction_flags.sh"), "rb").read()
        ok("F9a DUP15HD: build_extra_flags byte-identical to the tracked extraction_flags.sh",
           tracked == CC.build_extra_flags(c).encode("ascii") and CC.ptrain_is_default(c))
    else:
        ok("F9a DUP15HD config not reachable through SBI_HPC_DIR (skipped as pass)", True, dup)
    cfg = cohort_json(os.path.join(td, "cfg_flags.json"), td, os.path.join(td, "x"))
    c = CC.load_cohort(cfg)
    flags = CC.build_extra_flags(c)
    ok("F9b Giulia-like cohort: the three flags are appended",
       "--ptrain-format sparse_peaks --ptrain-varname peak_train --ptrain-name-pattern "
       + GIULIA_PATTERN + '"' in flags, flags.strip().splitlines()[-1][-80:])
    fp = os.path.join(td, "flags.sh")
    open(fp, "w", encoding="ascii", newline="\n").write(flags)
    rc, log = run(["bash", "-c",
                   'set -uo pipefail; source "%s"; python3 -c "import sys, json; '
                   'print(json.dumps(sys.argv[1:]))" $EXTRA_FLAGS' % fp])
    argv = json.loads(log.strip().splitlines()[-1]) if rc == 0 else []
    ok("F9c the regex survives the EXTRA_FLAGS double-quote + unquoted word-split path",
       rc == 0 and argv[-1] == GIULIA_PATTERN and argv[-3] == "peak_train", str(argv[-1:]))


def f10_config(td):
    base = dict(class_roots={"0": [td]}, class_names=["a"], extract_root="/x",
                n_subsets=9, electrodes_per_subset=1, fs_raw=FS, grid_width=10,
                index_base=0, mfr_threshold=0.1, w_size=0.01, gaussian_window=0.02)
    good = CC.CohortConfig(**base, ptrain_format="sparse_peaks", ptrain_varname="peak_train",
                           ptrain_name_pattern=GIULIA_PATTERN, exclude_wells=["ptrain_x"])
    ok("F10a a valid Giulia-like block is accepted", good.ptrain_format == "sparse_peaks")
    bad = [("format", dict(ptrain_format="dense")),
           ("varname", dict(ptrain_varname="1peak")),
           ("pattern without group", dict(ptrain_name_pattern=r"^ptrain_\d+\.mat$")),
           ("pattern with two groups", dict(ptrain_name_pattern=r"^ptrain_(\d+)_(\d+)\.mat$")),
           ("pattern with whitespace", dict(ptrain_name_pattern=r"^ptrain_ (\d+)\.mat$")),
           ("pattern with a glob char", dict(ptrain_name_pattern=r"^ptrain_[0-9]+_(\d+)\.mat$")),
           ("pattern with a double quote", dict(ptrain_name_pattern=r'^ptrain_"(\d+)\.mat$')),
           ("exclude_wells a string", dict(exclude_wells="ptrain_x")),
           ("exclude_wells a path", dict(exclude_wells=["a/ptrain_x"])),
           ("exclude_wells duplicate", dict(exclude_wells=["ptrain_x", "ptrain_x"]))]
    for lbl, kw in bad:
        try:
            CC.CohortConfig(**dict(base, **kw))
            ok("F10b CohortConfig refuses: %s" % lbl, False)
        except ValueError:
            ok("F10b CohortConfig refuses: %s" % lbl, True)
    try:
        CC.load_cohort(cohort_json(os.path.join(td, "cfg_unknown.json"), td, "/x"))
        bad_json = json.load(open(os.path.join(td, "cfg_unknown.json")))
        bad_json["cohort"]["ptrain_formatt"] = "raster"
        json.dump(bad_json, open(os.path.join(td, "cfg_unknown.json"), "w"))
        CC.load_cohort(os.path.join(td, "cfg_unknown.json"))
        ok("F10c load_cohort refuses an unknown (misspelt) field", False)
    except ValueError as exc:
        ok("F10c load_cohort refuses an unknown (misspelt) field", "ptrain_formatt" in str(exc))


def f11_f12_cohort(td):
    rng = np.random.default_rng(11)
    root = os.path.join(td, "Bio_Data")
    plan = [("Batch1/100N0A_wo", 38886, "100N0A", 60, 12),
            ("Batch1/100N0A_wo", 38932, "100N0A", 56, 10),
            ("Batch2/100N0A_wo", 34346, "100N0A", 60, 15),
            ("Batch3/AraC/50N50A", 38931, "50N50A", 26, 9),
            ("Batch3/AraC/50N50A", 39485, "50N50A", 17, 5),      # too few active
            ("Batch3/AraC/70N30A", 38927, "70N30A", 19, 9),
            ("Batch3/AraC/70N30A", 42627, "70N30A", 33, 11)]
    for rel, wid, cond, n_files, n_active in plan:
        make_mcs_well(os.path.join(root, rel, "ptrain_%d_DIV35_%s_nbasal_0001" % (wid, cond)),
                      wid, cond, MCS_CODES[:n_files], n_active, rng)
    excl = "ptrain_39485_DIV35_50N50A_nbasal_0001"
    extract_root = os.path.join(td, "extracted_giulia")
    cfg = cohort_json(os.path.join(td, "config_giulia_test.json"), root, extract_root,
                      exclude=[excl])
    # F11 find_wells
    w_all = CC.find_wells(os.path.join(root, "Batch3", "AraC", "50N50A"), "ptrain_*")
    w_ex = CC.find_wells(os.path.join(root, "Batch3", "AraC", "50N50A"), "ptrain_*", exclude=[excl])
    ok("F11a find_wells(exclude=...) drops the named well only",
       len(w_all) == 2 and w_ex == [w for w in w_all if w != excl])
    tsv = os.path.join(td, "m.tsv"); flags = os.path.join(td, "f.sh")
    rc, log = run([sys.executable, "list_extraction_jobs.py", "--config", cfg,
                   "--out-manifest", tsv, "--out-flags", flags], cwd=_HERE)
    rows = [ln.rstrip("\n").split("\t") for ln in open(tsv) if ln.strip()]
    ok("F11b list_extraction_jobs: 6 wells listed, the excluded one printed",
       rc == 0 and len(rows) == 6 and ("excluded (cohort.exclude_wells)" in log)
       and all(excl not in r[0] for r in rows), log.strip().splitlines()[-4][:70])
    cfg_typo = cohort_json(os.path.join(td, "config_typo.json"), root, extract_root,
                           exclude=["ptrain_00000_DIV35_50N50A_nbasal_0001"])
    rc2, log2 = run([sys.executable, "list_extraction_jobs.py", "--config", cfg_typo,
                     "--out-manifest", os.path.join(td, "m2.tsv"),
                     "--out-flags", os.path.join(td, "f2.sh")], cwd=_HERE)
    ok("F11c an exclusion matching no well ABORTS (exit 2), nothing written",
       rc2 == 2 and "exist under no root" in log2 and not os.path.exists(os.path.join(td, "m2.tsv")))
    # F12 extract every row with the generated flags, then the manifest
    fl = open(flags).read().split('EXTRA_FLAGS="')[1].split('"')[0].split()
    for folder, out_dir, _c in rows:
        rc, log = run([sys.executable, "run_channel_subset_extraction.py", folder,
                       "--out-dir", out_dir, "--mode", "per_region_single", "--no-plots"] + fl,
                      cwd=_HERE)
        if rc != 0:
            ok("F12a extraction of %s" % os.path.basename(folder), False, log[-200:])
            return
    ok("F12a every listed well extracts with the generated flags", True)
    rc, log = run([sys.executable, os.path.join(_ROOT, "cohort_manifest.py"), "--config", cfg,
                   "--manifest", tsv, "--flags", flags, "--extract-root", extract_root], cwd=_HERE)
    mp = os.path.join(extract_root, "cohort_manifest.json")
    ok("F12b the cohort manifest is written: 6 wells x 9 = 54 units",
       rc == 0 and os.path.isfile(mp) and "6 wells x 9 subregions = 54 units" in log,
       log.strip().splitlines()[-1][:80] if log.strip() else "")
    if os.path.isfile(mp):
        doc = json.load(open(mp))
        ok("F12c the manifest records source_format and excluded_wells",
           doc.get("source_format") == {"ptrain_format": "sparse_peaks",
                                        "ptrain_varname": "peak_train",
                                        "ptrain_name_pattern": GIULIA_PATTERN}
           and doc.get("excluded_wells") == [excl]
           and doc["classes"] == ["100N0A_wo", "50N50A", "70N30A"],
           json.dumps(doc.get("source_format"))[:70])
        ok("F12d the gating set is unchanged (D-002)",
           doc["gating"]["real"] == list(CC.PREPROCESSING_FIELDS) + ["n_units", "manifest_version"])
    # a fragment recording another format than the config declares -> refused
    cfg_raster = cohort_json(os.path.join(td, "config_raster.json"), root, extract_root,
                             exclude=[excl], fmt="raster", varname="ptrain",
                             pattern=r"^ptrain_(\d+)\.mat$")
    flags_r = os.path.join(td, "f_raster.sh")
    open(flags_r, "w", encoding="ascii", newline="\n").write(
        CC.build_extra_flags(CC.load_cohort(cfg_raster)))
    rc, log = run([sys.executable, os.path.join(_ROOT, "cohort_manifest.py"), "--config", cfg_raster,
                   "--manifest", tsv, "--flags", flags_r, "--extract-root", extract_root,
                   "--check-only"], cwd=_HERE)
    ok("F12e fragments recording sparse_peaks vs a config declaring raster -> REFUSED",
       rc == 1 and "REFUSED" in log and "source format" in log, log.strip().splitlines()[-1][:80])
    return root


def f13_probe(td, root):
    outj = os.path.join(td, "probe_giulia.json")
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root, "--fs-raw", str(FS),
                   "--mfr-threshold", "0.1", "--n-subsets", "9", "--electrodes-per-subset", "1",
                   "--expect-t-rec", "90", "--out-json", outj], cwd=_HERE)
    ok("F13a the probe runs on the Giulia-like tree (exit 0)", rc == 0, log.strip().splitlines()[-1][:70])
    doc = json.load(open(outj))
    sug = doc["suggested"]
    ok("F13b suggests sparse_peaks / peak_train / the MCS pattern / grid 10,0",
       sug["ptrain_format"] == "sparse_peaks" and sug["ptrain_varname"] == "peak_train"
       and sug["ptrain_name_pattern"] == GIULIA_PATTERN and sug["grid_width"] == 10
       and sug["index_base"] == 0, json.dumps(sug)[:90])
    ok("F13c the too-few-active well is the one under exclude_wells; 7 wells seen",
       sug["exclude_wells"] == ["ptrain_39485_DIV35_50N50A_nbasal_0001"] and doc["n_wells"] == 7)
    w = [x for x in doc["wells"] if x["well"].startswith("ptrain_39485")][0]
    ok("F13d that well: 17 files, 5 active, verdict too_few_active, T_rec 90 s matches",
       w["n_files"] == 17 and w["n_active"] == 5 and w["verdict"] == "too_few_active"
       and w["t_rec_matches_expected"] is True and w["positional_fits"] is True)
    w0 = [x for x in doc["wells"] if x["well"].startswith("ptrain_38886")][0]
    ok("F13e the extractor's rule preview lists nine electrodes for an ok well",
       len(w0["picked_by_extractor_rule"]) == 9 and w0["verdict"] == "ok")
    # a 3Brain-like tree
    root3 = os.path.join(td, "three_brain")
    make_3brain_well(os.path.join(root3, "ptrain_A1"), np.random.default_rng(1))
    outj3 = os.path.join(td, "probe_3b.json")
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root3, "--fs-raw", str(FS),
                   "--n-subsets", "9", "--electrodes-per-subset", "1", "--out-json", outj3], cwd=_HERE)
    sug3 = json.load(open(outj3))["suggested"]
    ok("F13f a 3Brain-like tree: raster / ptrain / strict pattern / grid 8,1 (exit 0)",
       rc == 0 and sug3["ptrain_format"] == "raster" and sug3["ptrain_varname"] == "ptrain"
       and sug3["ptrain_name_pattern"] == r"^ptrain_(\d+)\.mat$"
       and sug3["grid_width"] == 8 and sug3["index_base"] == 1, json.dumps(sug3)[:90])
    # a mixed tree -> UNDETERMINED, exit 1
    mixed = os.path.join(td, "mixed")
    shutil.copytree(os.path.join(root, "Batch1"), os.path.join(mixed, "Batch1"))
    shutil.copytree(root3, os.path.join(mixed, "three_brain"))
    outm = os.path.join(td, "probe_mixed.json")
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", mixed, "--fs-raw", str(FS),
                   "--out-json", outm], cwd=_HERE)
    ok("F13g a mixed tree exits 1 with UNDETERMINED fields",
       rc == 1 and "UNDETERMINED" in log and "PROBE FAILED" in log)
    # --config (2026-10-01): the cohort block supplies the parameters and is compared
    excl = "ptrain_39485_DIV35_50N50A_nbasal_0001"
    cfg_ok = cohort_json(os.path.join(td, "cfg_probe_ok.json"), root, os.path.join(td, "x"),
                         exclude=[excl])
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root, "--config", cfg_ok,
                   "--out-json", os.path.join(td, "p_ok.json")], cwd=_HERE)
    last = log.strip().splitlines()[-1]
    ok("F13h --root + a matching --config: exit 0, 'PROBE OK; CONFIG MATCHES (6 wells to extract)'",
       rc == 0 and last == "PROBE OK; CONFIG MATCHES (6 wells to extract)", last[:80])
    d_ok = json.load(open(os.path.join(td, "p_ok.json")))
    ok("F13h2 parameters taken from the config, recorded with their source",
       d_ok["fs_raw"] == FS and d_ok["n_subsets"] == 9 and d_ok["electrodes_per_subset"] == 1
       and d_ok["sources"]["fs_raw"] == "config" and d_ok["sources"]["well_glob"] == "config")
    cfg_noex = cohort_json(os.path.join(td, "cfg_probe_noex.json"), root, os.path.join(td, "x"),
                           exclude=[])
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root, "--config", cfg_noex,
                   "--out-json", os.path.join(td, "p_noex.json")], cwd=_HERE)
    last = log.strip().splitlines()[-1]
    ok("F13i the too-few-active well missing from exclude_wells: exit 3, DIFFERS on exclude_wells",
       rc == 3 and last == "PROBE OK; CONFIG DIFFERS on: exclude_wells", last[:80])
    cfg_var = cohort_json(os.path.join(td, "cfg_probe_var.json"), root, os.path.join(td, "x"),
                          exclude=[excl], varname="ptrain")
    rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root, "--config", cfg_var,
                   "--out-json", os.path.join(td, "p_var.json")], cwd=_HERE)
    last = log.strip().splitlines()[-1]
    ok("F13j a wrong ptrain_varname in the config: exit 3, DIFFERS on ptrain_varname",
       rc == 3 and last == "PROBE OK; CONFIG DIFFERS on: ptrain_varname", last[:80])
    stray = os.path.join(root, "Batch9", "100N0A_wo", "ptrain_99999_DIV35_100N0A_nbasal_0001")
    make_mcs_well(stray, 99999, "100N0A", MCS_CODES[:20], 12, np.random.default_rng(99))
    try:
        rc, log = run([sys.executable, "probe_ptrain_tree.py", "--root", root, "--config", cfg_ok,
                       "--out-json", os.path.join(td, "p_cov.json")], cwd=_HERE)
        last = log.strip().splitlines()[-1]
        ok("F13k a data well under no class root: exit 3, DIFFERS on coverage",
           rc == 3 and "coverage: probed wells under no class root" in last, last[:90])
        rc, log = run([sys.executable, "probe_ptrain_tree.py", "--config", cfg_ok,
                       "--out-json", os.path.join(td, "p_roots.json")], cwd=_HERE)
        last = log.strip().splitlines()[-1]
        d_r = json.load(open(os.path.join(td, "p_roots.json")))
        ok("F13l --config alone probes exactly the class roots (7 wells, no coverage check): exit 0",
           rc == 0 and d_r["n_wells"] == 7 and last.startswith("PROBE OK; CONFIG MATCHES"), last[:80])
    finally:
        shutil.rmtree(os.path.join(root, "Batch9"), ignore_errors=True)


def f14_launcher(td, root):
    launcher = os.path.join(_HERE, "launch_stage_d.sh")
    extract_root = os.path.join(td, "extracted_giulia_dry")
    cfg = cohort_json(os.path.join(td, "config_launch.json"), root, os.path.join(td, "declared"),
                      exclude=["ptrain_39485_DIV35_50N50A_nbasal_0001"])
    tag = "smoketag"
    flags_path = os.path.join(_HERE, "extraction_flags_%s.sh" % tag)
    tsv_path = os.path.join(_HERE, "out", "extraction_manifest_%s.tsv" % tag)
    for p in (flags_path, tsv_path):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, CONFIG=cfg, COHORT_TAG=tag, DRYRUN="1")
    try:
        rc1, log1 = run(["bash", launcher, extract_root], cwd=_HERE, env=env)
        ok("F14a first tagged dry run: NEW COHORT, writes the flags file, exit 3, nothing submitted",
           rc1 == 3 and "NEW COHORT" in log1 and os.path.isfile(flags_path)
           and "Nothing was submitted" in log1, log1.strip().splitlines()[-1][:70])
        rc2, log2 = run(["bash", launcher, extract_root], cwd=_HERE, env=env)
        want_arr = ("-N chsub_mea_array_%s -o out/chsub_mea_array_%s_^array_index^.log -v "
                    "ENV_NAME=sbi_export,CHSUB_MANIFEST=out/extraction_manifest_%s.tsv,"
                    "CHSUB_FLAGS=extraction_flags_%s.sh run_extractor_array_mea.pbs"
                    % (tag, tag, tag, tag))
        want_agg = ("-N cohort_manifest_%s -o out/cohort_manifest_%s.log -v EXTRACT_ROOT=%s,"
                    "CONFIG=%s,ENV_NAME=sbi_export,CHSUB_MANIFEST=out/extraction_manifest_%s.tsv,"
                    "CHSUB_FLAGS=extraction_flags_%s.sh run_cohort_manifest.pbs"
                    % (tag, tag, extract_root, cfg, tag, tag))
        ok("F14b second tagged dry run: exit 0, 6 wells, both qsub lines carry the cohort's files",
           rc2 == 0 and "qsub -J 0-5 " in log2 and want_arr in log2 and want_agg in log2
           and os.path.isfile(tsv_path), log2.strip().splitlines()[-1][:70])
        rc3, log3 = run(["bash", launcher, extract_root], cwd=_HERE,
                        env=dict(env, COHORT_TAG="bad tag"))
        ok("F14c a tag with a space is refused", rc3 == 2 and "COHORT_TAG" in log3)
        rc4, log4 = run(["bash", launcher, os.path.join(td, "declared")], cwd=_HERE, env=env)
        ok("F14d EXTRACT_ROOT == a declared root that does not exist yet: a first extraction, allowed",
           rc4 == 0 and "first extraction" in log4, log4.strip().splitlines()[-1][:60])
        os.makedirs(os.path.join(td, "declared"), exist_ok=True)
        open(os.path.join(td, "declared", "something.npz"), "wb").write(b"x")
        rc4b, log4b = run(["bash", launcher, os.path.join(td, "declared")], cwd=_HERE, env=env)
        ok("F14d2 EXTRACT_ROOT == a declared root that exists and is not empty: refused (exit 3)",
           rc4b == 3 and "archives of record" in log4b)
        # UNTAGGED with this cohort's config: the DUP15HD path. Its flags file
        # is the tracked extraction_flags.sh, which this cohort's flags do not
        # match, so the launcher must refuse (exit 3) naming THAT file -- and
        # must not have touched any tagged file. The DUP15HD manifest TSV it
        # writes on the way is restored afterwards.
        dup_tsv = os.path.join(_HERE, "extraction_manifest.tsv")
        saved = open(dup_tsv, "rb").read() if os.path.isfile(dup_tsv) else None
        try:
            rc5, log5 = run(["bash", launcher, os.path.join(td, "untagged_dry")], cwd=_HERE,
                            env=dict(os.environ, CONFIG=cfg, DRYRUN="1"))
            ok("F14e untagged: refused against the tracked extraction_flags.sh (exit 3), DUP15HD names",
               rc5 == 3 and "differ from the tracked extraction_flags.sh" in log5
               and "CHSUB_" not in log5 and os.path.isfile(dup_tsv),
               log5.strip().splitlines()[-1][:70] if log5.strip() else "")
        finally:
            if saved is not None:
                open(dup_tsv, "wb").write(saved)
            elif os.path.isfile(dup_tsv):
                os.remove(dup_tsv)
    finally:
        for p in (flags_path, tsv_path):
            if os.path.exists(p):
                os.remove(p)


def main():
    print("=" * 100)
    print("Smoke test: declared ptrain formats, probe, exclusion list, per-cohort launcher (2026-10-01)")
    print("=" * 100)
    td = tempfile.mkdtemp(prefix="ptrain_fmt_")
    try:
        f1_f2_patterns()
        f3_f4_formats(td)
        folder = f5_f7_well(td)
        f8_cli(td, folder)
        f9_flags(td)
        f10_config(td)
        root = f11_f12_cohort(td)
        if root:
            f13_probe(td, root)
            f14_launcher(td, root)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(False)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("-" * 100)
    n_fail = RESULTS.count(False)
    print("%d passed, %d failed" % (RESULTS.count(True), n_fail))
    print("ALL PTRAIN-FORMAT CHECKS PASSED" if n_fail == 0 else "PTRAIN-FORMAT FAILURES: %d" % n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
