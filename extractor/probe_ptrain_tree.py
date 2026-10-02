#!/usr/bin/env python3
"""probe_ptrain_tree.py -- read-only census of a tree of per-electrode spike
train files BEFORE a cohort is configured and extracted.

Why this exists (2026-10-01, the Giulia recordings)
---------------------------------------------------
The extractor reads ONE declared file format (cohort block: ptrain_format,
ptrain_varname, ptrain_name_pattern) and refuses any other, and it discards a
well with fewer than n_subsets x electrodes_per_subset electrodes at or above
mfr_threshold -- which, under the cohort manifest's rules, then refuses the
WHOLE cohort. Both facts must be known before the array job is submitted, not
discovered one failed task at a time. This probe opens every file once and
reports, per well:

  - the filename parse: which pattern the names satisfy, and the integer
    index each name yields;
  - the index decoding: whether the indices are MCS-style row/column codes
    (10*row + col, rows and columns 1..8, the four corners absent), which
    the extractor decodes with grid_width 10 / index_base 0, or plain
    row-major positions;
  - what the .mat files hold: every variable's storage class (scipy.sparse
    or dense ndarray), shape, dtype, number of nonzeros, value range -- and
    the ptrain_format the extractor would need for it;
  - n_samples, its consistency across the well, T_rec at --fs-raw (checked
    against --expect-t-rec when given);
  - spike counts, mean firing rates, the number of electrodes at or above
    --mfr-threshold, and whether that meets n_subsets x electrodes_per_subset
    (the verdict), plus the electrodes the extractor's rule would pick.

It then prints a SUGGESTED cohort block fragment (format, variable, pattern,
grid decoding, exclude_wells) for you to review -- nothing is configured by
this script, and nothing is written except the JSON report (--out-json).

It reads with scipy.io.loadmat only; it does not import the extractor, so it
runs on a login node in the sbi_export env in seconds and cannot be broken by
a loader that does not understand the files yet.

Usage
-----
    cd ~/repos/Sbi-extractor/extractor && source ../env.sh
    python3 probe_ptrain_tree.py \\
        --root /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/Bio_Data \\
        --config "$SBI_HPC_DIR/dsn/hpc/Config/config_giulia_cohort.davinci.json" \\
        --out-json out/probe_giulia.json

With --config (2026-10-01) the cohort block supplies fs_raw, mfr_threshold,
n_subsets, electrodes_per_subset and well_glob unless given as flags, and the
census is COMPARED with it: ptrain_format, ptrain_varname,
ptrain_name_pattern, grid_width and index_base must equal what the files
show; every well the census finds below n_subsets x electrodes_per_subset
active electrodes must be in exclude_wells (extra exclusions are reported,
not refused); every probed well must lie under one of the config's class
roots and every well under a class root must have been probed. The last line
is then "PROBE OK; CONFIG MATCHES ..." or "PROBE OK; CONFIG DIFFERS on: ...".
Without --root the config's class roots are probed (no coverage check).

Exit code: 0 when every well was read, the format is unambiguous (even if
some wells fail the electrode count -- that is a finding, not an error) and,
with --config, the config matches; 1 when a file could not be read or the
format is mixed/undetermined; 3 when the census is fine but the config
differs from it; 2 on a usage error.

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy import sparse

# The patterns tried on every file name, most specific first. The extractor
# itself takes ONE pattern from the cohort block; the probe only reports
# which of these the names satisfy, to suggest that one value.
NAME_PATTERNS = (
    ("strict_3brain", r"^ptrain_(\d+)\.mat$"),
    ("giulia_mcs", r"^ptrain_\d+_DIV\d+_\w+_nbasal_\d{4}_(\d{3})\.mat$"),
    ("trailing_int", r"^ptrain.*?(\d+)\.mat$"),
)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_well_dirs(root, well_glob, file_glob):
    """Every directory under root whose basename matches well_glob and that
    directly holds at least one file matching file_glob; sorted. Also the
    well-like directories that hold none (reported, not probed)."""
    wells, empty = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if not fnmatch.fnmatch(os.path.basename(dirpath), well_glob):
            continue
        mats = sorted(f for f in filenames if fnmatch.fnmatch(f, file_glob))
        if mats:
            wells.append((dirpath, mats))
        else:
            empty.append(dirpath)
    wells.sort()
    return wells, sorted(empty)


# --------------------------------------------------------------------------- #
# one file
# --------------------------------------------------------------------------- #
def describe_variable(v):
    """Storage class, shape, dtype, nonzero count and value range of one
    loaded MATLAB variable, as plain JSON-able values."""
    d = {"storage": None, "class": type(v).__name__, "shape": None,
         "dtype": None, "n_nonzero": None, "vmin": None, "vmax": None,
         "looks_like": "other"}
    if sparse.issparse(v):
        d["storage"] = "sparse"
        d["shape"] = [int(x) for x in v.shape]
        d["dtype"] = str(v.dtype)
        rows, cols = v.nonzero()
        d["n_nonzero"] = int(rows.size)
        data = np.asarray(v.data)
        data = data[data != 0]
        if data.size:
            d["vmin"] = float(np.min(data)); d["vmax"] = float(np.max(data))
        if len(d["shape"]) == 2 and 1 in d["shape"]:
            d["looks_like"] = "sparse_column"
        return d
    a = np.asarray(v)
    d["storage"] = "dense"
    d["shape"] = [int(x) for x in a.shape]
    d["dtype"] = str(a.dtype)
    if a.dtype.kind in "biuf" and a.size:
        d["n_nonzero"] = int(np.count_nonzero(a))
        d["vmin"] = float(np.min(a)); d["vmax"] = float(np.max(a))
        singleton = a.ndim == 1 or (a.ndim == 2 and 1 in a.shape)
        if singleton and d["vmin"] >= 0.0 and d["vmax"] <= 1.0:
            d["looks_like"] = "binary_raster"
        elif singleton:
            d["looks_like"] = "dense_nonbinary"      # amplitudes? an index list?
    elif a.dtype.kind in "US":
        d["looks_like"] = "string"
    return d


def probe_file(path):
    """{variables: {name: describe}, error: str or None}."""
    try:
        md = sio.loadmat(path)
    except Exception as exc:                        # noqa: BLE001
        return {"variables": {}, "error": "loadmat failed: %s" % (exc,)}
    out = {}
    for k, v in md.items():
        if k.startswith("__"):
            continue
        out[k] = describe_variable(v)
    return {"variables": out, "error": None}


def train_variable(variables):
    """Pick the variable that holds the train, by priority: the one sparse
    column; else the one binary raster; else the one dense non-binary vector;
    else the only variable. None when a class has two candidates (ambiguous)
    or nothing fits. A second variable of a lower class (SpyCode-style files
    carry an "artifact" vector beside the train, say) does not get in the way."""
    for cls in ("sparse_column", "binary_raster", "dense_nonbinary"):
        cands = [k for k, d in variables.items() if d["looks_like"] == cls]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            return None
    if len(variables) == 1:
        return next(iter(variables))
    return None


def spikes_of(path, varname):
    """(spike sample indices, n_samples) read the way the extractor would,
    for either storage; None on a shape it cannot read."""
    md = sio.loadmat(path)
    v = md[varname]
    if sparse.issparse(v):
        shape = tuple(int(x) for x in v.shape)
        if len(shape) != 2 or 1 not in shape:
            return None
        rows, cols = v.nonzero()
        pos = rows if shape[0] >= shape[1] else cols
        return np.unique(np.asarray(pos, dtype=np.int64)), int(max(shape))
    a = np.asarray(v)
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        return None
    if a.dtype.kind in "biuf" and a.size and float(np.max(a)) <= 1.0 and float(np.min(a)) >= 0.0:
        return np.nonzero(a)[0].astype(np.int64), int(a.size)
    return None


# --------------------------------------------------------------------------- #
# names and indices
# --------------------------------------------------------------------------- #
def parse_names(names):
    """For each pattern: does every name match, and the index per name."""
    res = {}
    for tag, pat in NAME_PATTERNS:
        rx = re.compile(pat)
        idx = {}
        ok = True
        for n in names:
            m = rx.match(n)
            if m is None:
                ok = False
                break
            idx[n] = int(m.group(1))
        res[tag] = {"pattern": pat, "all_match": ok,
                    "indices": idx if ok else {}}
    return res


def decode_positional(indices):
    """MCS-style code k = 10*row + col with row, col in 1..8: the decoded
    (row, col) per index, and whether every index fits that reading with no
    corner (11, 18, 81, 88) present. This is what grid_width 10 / index_base 0
    computes in the extractor (row = k // 10, col = k % 10)."""
    dec, fits = {}, True
    for k in indices:
        r, c = int(k) // 10, int(k) % 10
        dec[int(k)] = [r, c]
        if not (1 <= r <= 8 and 1 <= c <= 8) or (r, c) in ((1, 1), (1, 8), (8, 1), (8, 8)):
            fits = False
    return dec, fits


def decode_rowmajor(indices, width, base):
    """Plain row-major decoding, as a comparison: inside the grid or not."""
    inside = all(0 <= (int(k) - base) < width * width for k in indices)
    return inside


# --------------------------------------------------------------------------- #
# one well
# --------------------------------------------------------------------------- #
def probe_well(dirpath, mats, fs_raw, mfr_threshold, n_subsets, e_per_subset,
               expect_t_rec, max_files):
    w = {"well": os.path.basename(dirpath), "path": dirpath,
         "n_files": len(mats), "files_probed": 0, "errors": [],
         "names": parse_names(mats), "variables": {}, "train_varname": None,
         "storage": None, "format_needed": None,
         "n_samples": None, "n_samples_consistent": None, "T_rec_s": None,
         "t_rec_matches_expected": None,
         "indices": [], "positional_fits": None, "positional_decoded": {},
         "rowmajor_8_1_inside": None, "duplicate_indices": [],
         "spike_counts": {}, "mfr_hz": {}, "n_active": None,
         "needed": int(n_subsets) * int(e_per_subset), "verdict": None,
         "picked_by_extractor_rule": []}
    files = mats if max_files is None else mats[:max_files]

    # ---- the index per file, from the most specific pattern that fits all
    chosen = None
    for tag, _p in NAME_PATTERNS:
        if w["names"][tag]["all_match"]:
            chosen = tag
            break
    w["name_pattern_chosen"] = chosen
    idx_of = w["names"][chosen]["indices"] if chosen else {}

    # ---- every file: variables, storage, train
    storages, varnames, n_samples_seen = set(), set(), {}
    for f in files:
        path = os.path.join(dirpath, f)
        pf = probe_file(path)
        if pf["error"]:
            w["errors"].append("%s: %s" % (f, pf["error"]))
            continue
        w["files_probed"] += 1
        for k, d in pf["variables"].items():
            agg = w["variables"].setdefault(k, {"storage": set(), "looks_like": set(),
                                                "dtype": set(), "n_files": 0})
            agg["storage"].add(d["storage"]); agg["looks_like"].add(d["looks_like"])
            agg["dtype"].add(d["dtype"]); agg["n_files"] += 1
        tv = train_variable(pf["variables"])
        if tv is None:
            w["errors"].append("%s: no single train variable among %r"
                               % (f, sorted(pf["variables"])))
            continue
        varnames.add(tv)
        storages.add(pf["variables"][tv]["storage"])
        got = spikes_of(path, tv)
        if got is None:
            w["errors"].append("%s: variable %r has a shape/values the extractor "
                               "cannot read as a train (%r)"
                               % (f, tv, pf["variables"][tv]))
            continue
        spikes, n = got
        n_samples_seen[f] = n
        k = idx_of.get(f)
        if k is not None:
            if k in w["spike_counts"]:
                w["duplicate_indices"].append(k)
            w["spike_counts"][k] = int(spikes.size)
    for k, agg in w["variables"].items():
        for kk in ("storage", "looks_like", "dtype"):
            agg[kk] = sorted(x for x in agg[kk] if x is not None)

    # ---- storage -> the format the extractor needs
    w["train_varname"] = sorted(varnames)[0] if len(varnames) == 1 else (sorted(varnames) or None)
    w["storage"] = sorted(storages)[0] if len(storages) == 1 else (sorted(storages) or None)
    if w["storage"] == "sparse":
        w["format_needed"] = "sparse_peaks"
    elif w["storage"] == "dense":
        w["format_needed"] = "raster"
    else:
        w["format_needed"] = "UNDETERMINED"

    # ---- n_samples, T_rec
    ns = sorted(set(n_samples_seen.values()))
    if ns:
        w["n_samples_consistent"] = (len(ns) == 1)
        w["n_samples"] = ns[0] if len(ns) == 1 else ns
        if len(ns) == 1:
            w["T_rec_s"] = ns[0] / float(fs_raw)
            if expect_t_rec is not None:
                w["t_rec_matches_expected"] = abs(w["T_rec_s"] - float(expect_t_rec)) < 1e-6

    # ---- indices and their decoding
    idx = sorted(idx_of.values())
    w["indices"] = idx
    if idx:
        dec, fits = decode_positional(idx)
        w["positional_fits"] = fits
        w["positional_decoded"] = {str(k): v for k, v in dec.items()}
        w["rowmajor_8_1_inside"] = decode_rowmajor(idx, 8, 1)

    # ---- MFR, activity, verdict (the extractor's rule: top-C by MFR, ties
    # to the lower index, among those at or above the threshold)
    if w["T_rec_s"] and w["spike_counts"]:
        t = float(w["T_rec_s"])
        w["mfr_hz"] = {str(k): c / t for k, c in sorted(w["spike_counts"].items())}
        valid = [(k, c / t) for k, c in w["spike_counts"].items() if c / t >= mfr_threshold]
        w["n_active"] = len(valid)
        w["verdict"] = "ok" if len(valid) >= w["needed"] else "too_few_active"
        order = sorted(valid, key=lambda kv: (-kv[1], kv[0]))
        w["picked_by_extractor_rule"] = [k for k, _m in order[:int(n_subsets)]]
    elif w["errors"] and not w["spike_counts"]:
        w["verdict"] = "unreadable"
    else:
        w["verdict"] = "no_spike_counts"
    return w


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def suggest(wells, fs_raw, chosen_patterns):
    """The cohort-block fragment the census supports, or UNDETERMINED."""
    fmts = sorted(set(w["format_needed"] for w in wells))
    vars_ = sorted(set(w["train_varname"] for w in wells if isinstance(w["train_varname"], str)))
    pats = sorted(set(w["name_pattern_chosen"] for w in wells if w["name_pattern_chosen"]))
    pos = all(w["positional_fits"] is True for w in wells)
    rm81 = all(w["rowmajor_8_1_inside"] is True for w in wells)
    sug = {
        "fs_raw": float(fs_raw),
        "ptrain_format": fmts[0] if len(fmts) == 1 and fmts[0] != "UNDETERMINED" else "UNDETERMINED",
        "ptrain_varname": vars_[0] if len(vars_) == 1 else "UNDETERMINED",
        "ptrain_name_pattern": (dict(NAME_PATTERNS)[pats[0]] if len(pats) == 1 else "UNDETERMINED"),
        "grid_width": 10 if pos else ("UNDETERMINED" if not rm81 else 8),
        "index_base": 0 if pos else ("UNDETERMINED" if not rm81 else 1),
        "grid_reading": ("row/column code 10*row+col, rows and columns 1..8, no corner: "
                         "decoded exactly by grid_width 10 / index_base 0" if pos else
                         ("plain row-major positions 1..64 (8 x 8, base 1)" if rm81
                          else "neither reading fits every well: see positional_fits per well")),
        "exclude_wells": sorted(w["well"] for w in wells if w["verdict"] != "ok"),
    }
    return sug


def print_report(doc):
    print("=" * 100)
    print("ptrain tree probe: %s" % ", ".join(doc["root"]))
    print("fs_raw %.10g Hz  mfr_threshold %.10g Hz  needed %d active electrode(s) per well"
          % (doc["fs_raw"], doc["mfr_threshold"], doc["needed"]))
    print("=" * 100)
    print("%-44s %5s %6s %-12s %-11s %8s %5s %6s %-15s" % (
        "well", "files", "n_idx", "storage", "variable", "T_rec_s", "pos?", "active", "verdict"))
    for w in doc["wells"]:
        tv = w["train_varname"] if isinstance(w["train_varname"], str) else "MIXED"
        print("%-44s %5d %6d %-12s %-11s %8s %5s %6s %-15s" % (
            w["well"][:44], w["n_files"], len(w["indices"]),
            str(w["storage"]), str(tv)[:11],
            ("%.1f" % w["T_rec_s"]) if w["T_rec_s"] else "?",
            {True: "yes", False: "no", None: "?"}[w["positional_fits"]],
            str(w["n_active"]) if w["n_active"] is not None else "?",
            w["verdict"]))
        for e in w["errors"][:3]:
            print("      ERROR %s" % e)
        if len(w["errors"]) > 3:
            print("      ... %d more error(s)" % (len(w["errors"]) - 3))
    print("-" * 100)
    if doc["empty_well_dirs"]:
        print("well-like directories with no file matching the file glob (not probed):")
        for d in doc["empty_well_dirs"]:
            print("   %s" % d)
    print("variables seen (name: storage / looks_like / dtype, in how many files):")
    for k, agg in sorted(doc["variables_seen"].items()):
        print("   %-14s %s / %s / %s  (%d file(s))" % (
            k, ",".join(agg["storage"]), ",".join(agg["looks_like"]),
            ",".join(agg["dtype"]), agg["n_files"]))
    print("T_rec by well (s): %s" % json.dumps(
        {w["well"]: (round(w["T_rec_s"], 3) if w["T_rec_s"] else None) for w in doc["wells"]},
        indent=None))
    print("")
    print("SUGGESTED cohort fields (review; nothing has been configured):")
    print(json.dumps(doc["suggested"], indent=2, sort_keys=True))
    print("")
    print("verdicts: %s" % json.dumps(doc["verdict_counts"], sort_keys=True))


def load_cohort_block(config_path):
    """The "cohort" object of a DSN JSON config, read with json only (no DSN
    tree, no torch): the probe must run where nothing else is set up yet."""
    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "cohort" not in data or not isinstance(data["cohort"], dict):
        raise KeyError("%s has no \"cohort\" object" % (config_path,))
    return data["cohort"]


def config_wells(cohort, well_glob):
    """{normalised well path: class index} for every immediate child of every
    class root matching well_glob (exclusions INCLUDED: they are data the
    config knows about), plus the class roots that do not exist."""
    out, missing = {}, []
    for c, roots in sorted(cohort.get("class_roots", {}).items()):
        for r in roots:
            r = os.path.normpath(str(r))
            if not os.path.isdir(r):
                missing.append(r)
                continue
            for e in sorted(os.listdir(r)):
                full = os.path.join(r, e)
                if os.path.isdir(full) and fnmatch.fnmatch(e, well_glob):
                    out[os.path.normpath(full)] = int(c)
    return out, missing


def compare_with_config(cohort, sug, results, probe_roots_given, well_glob, fs_raw):
    """What differs between the census and the cohort block, as a dict.

    diffs   : [(field, config value, census value)] -- each one blocks
    info    : [str] -- reported, not blocking
    """
    diffs, info = [], []
    want = {
        "ptrain_format": cohort.get("ptrain_format", "raster"),
        "ptrain_varname": cohort.get("ptrain_varname", "ptrain"),
        "ptrain_name_pattern": cohort.get("ptrain_name_pattern", r"^ptrain_(\d+)\.mat$"),
        "grid_width": cohort.get("grid_width", 48),
        "index_base": cohort.get("index_base", 0),
    }
    for k, v in want.items():
        if sug.get(k) != v:
            diffs.append((k, v, sug.get(k)))
    if "fs_raw" in cohort and abs(float(cohort["fs_raw"]) - float(fs_raw)) > 0:
        diffs.append(("fs_raw", cohort["fs_raw"], fs_raw))
    excl_cfg = set(cohort.get("exclude_wells", []) or [])
    need_excl = set(sug.get("exclude_wells", []))
    missing_excl = sorted(need_excl - excl_cfg)
    if missing_excl:
        diffs.append(("exclude_wells", sorted(excl_cfg), sorted(need_excl | excl_cfg)))
    extra_excl = sorted(excl_cfg - need_excl)
    if extra_excl:
        info.append("excluded by the config although the census finds them usable: %s"
                    % extra_excl)
    cfg_wells, missing_roots = config_wells(cohort, well_glob)
    if missing_roots:
        diffs.append(("class_roots (missing)", missing_roots, "not a directory"))
    probed = set(os.path.normpath(w["path"]) for w in results)
    if probe_roots_given:
        not_in_cfg = sorted(probed - set(cfg_wells))
        not_probed = sorted(set(cfg_wells) - probed)
        if not_in_cfg:
            diffs.append(("coverage: probed wells under no class root", [], not_in_cfg))
        if not_probed:
            diffs.append(("coverage: class-root wells not probed", not_probed, []))
    per_class = {}
    for path, c in cfg_wells.items():
        name = os.path.basename(path)
        key = str(c)
        d = per_class.setdefault(key, {"wells": 0, "excluded": 0})
        d["wells"] += 1
        if name in excl_cfg:
            d["excluded"] += 1
    return {"diffs": diffs, "info": info, "per_class": per_class,
            "n_config_wells": len(cfg_wells),
            "n_config_wells_kept": sum(1 for p in cfg_wells
                                       if os.path.basename(p) not in excl_cfg)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", default=None,
                    help="a tree to walk (repeatable); omitted with --config: "
                         "the config's class roots")
    ap.add_argument("--config", default=None,
                    help="DSN JSON config whose cohort block supplies the "
                         "defaults below and is compared with the census")
    ap.add_argument("--fs-raw", type=float, default=None,
                    help="raw rate [Hz]; default: the config's, else 10000")
    ap.add_argument("--mfr-threshold", type=float, default=None,
                    help="[Hz]; default: the config's, else 0.1")
    ap.add_argument("--n-subsets", type=int, default=None,
                    help="default: the config's, else 9")
    ap.add_argument("--electrodes-per-subset", type=int, default=None,
                    help="default: the config's, else 1")
    ap.add_argument("--well-glob", default=None,
                    help="basename pattern of a well directory; default: the "
                         "config's well_glob, else ptrain_*")
    ap.add_argument("--file-glob", default="ptrain*.mat",
                    help="basename pattern of the per-electrode files")
    ap.add_argument("--expect-t-rec", type=float, default=None,
                    help="flag wells whose T_rec differs from this many seconds")
    ap.add_argument("--max-files-per-well", type=int, default=None,
                    help="probe only the first N files of each well (a quick look)")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--out-json", default="out/probe_ptrain_tree.json")
    a = ap.parse_args(argv)

    cohort = None
    if a.config is not None:
        if not os.path.isfile(a.config):
            print("ABORT: config not found: %s" % a.config)
            return 2
        try:
            cohort = load_cohort_block(a.config)
        except (KeyError, ValueError) as exc:
            print("ABORT: %s" % exc)
            return 2

    def pick(flag_value, key, default):
        if flag_value is not None:
            return flag_value, "flag"
        if cohort is not None and key in cohort:
            return cohort[key], "config"
        return default, "default"

    fs_raw, src_fs = pick(a.fs_raw, "fs_raw", 10000.0)
    mfr_thr, src_thr = pick(a.mfr_threshold, "mfr_threshold", 0.1)
    n_sub, src_c = pick(a.n_subsets, "n_subsets", 9)
    e_per, src_e = pick(a.electrodes_per_subset, "electrodes_per_subset", 1)
    well_glob, src_g = pick(a.well_glob, "well_glob", "ptrain_*")
    fs_raw, mfr_thr, n_sub, e_per = float(fs_raw), float(mfr_thr), int(n_sub), int(e_per)

    if fs_raw <= 0 or n_sub < 1 or e_per < 1 or mfr_thr < 0:
        print("ABORT: fs_raw > 0, n_subsets >= 1, electrodes_per_subset >= 1, mfr_threshold >= 0")
        return 2

    roots_given = bool(a.root)
    if roots_given:
        roots = list(a.root)
    elif cohort is not None:
        roots = [str(r) for _c, rs in sorted(cohort.get("class_roots", {}).items()) for r in rs]
    else:
        print("ABORT: give --root (or --config, whose class roots are then probed)")
        return 2
    for r in roots:
        if not os.path.isdir(r):
            print("ABORT: not a directory: %s" % r)
            return 2

    wells, empty = [], []
    for r in roots:
        w_r, e_r = find_well_dirs(r, well_glob, a.file_glob)
        wells.extend(w_r); empty.extend(e_r)
    seen_paths, uniq = set(), []
    for dirpath, mats in sorted(wells):
        if dirpath not in seen_paths:
            seen_paths.add(dirpath); uniq.append((dirpath, mats))
    wells = uniq
    if a.max_wells is not None:
        wells = wells[:a.max_wells]
    if not wells:
        print("ABORT: no directory matching %r holding a file matching %r under %s"
              % (well_glob, a.file_glob, roots))
        return 2

    results = []
    for dirpath, mats in wells:
        results.append(probe_well(dirpath, mats, fs_raw, mfr_thr, n_sub, e_per,
                                  a.expect_t_rec, a.max_files_per_well))

    seen = {}
    for w in results:
        for k, agg in w["variables"].items():
            s_ = seen.setdefault(k, {"storage": set(), "looks_like": set(), "dtype": set(), "n_files": 0})
            s_["storage"].update(agg["storage"]); s_["looks_like"].update(agg["looks_like"])
            s_["dtype"].update(agg["dtype"]); s_["n_files"] += agg["n_files"]
    for k, s_ in seen.items():
        for kk in ("storage", "looks_like", "dtype"):
            s_[kk] = sorted(s_[kk])

    counts = {}
    for w in results:
        counts[w["verdict"]] = counts.get(w["verdict"], 0) + 1
    sug = suggest(results, fs_raw, None)
    unreadable = any(w["errors"] for w in results)
    undetermined = any(v == "UNDETERMINED" for v in sug.values())
    status = "FAILED (a file could not be read, or the format/pattern/grid is undetermined)" \
        if (unreadable or undetermined) else "OK"
    check = None
    if cohort is not None:
        check = compare_with_config(cohort, sug, results, roots_given, well_glob, fs_raw)
    doc = {"root": [os.path.abspath(r) for r in roots], "fs_raw": fs_raw,
           "mfr_threshold": mfr_thr, "n_subsets": n_sub,
           "electrodes_per_subset": e_per, "well_glob": well_glob,
           "sources": {"fs_raw": src_fs, "mfr_threshold": src_thr,
                       "n_subsets": src_c, "electrodes_per_subset": src_e,
                       "well_glob": src_g},
           "config": os.path.abspath(a.config) if a.config else None,
           "needed": n_sub * e_per,
           "expect_t_rec": a.expect_t_rec, "n_wells": len(results),
           "empty_well_dirs": empty, "variables_seen": seen,
           "wells": results, "verdict_counts": counts, "suggested": sug,
           "status": status, "config_check": check}
    outp = a.out_json
    parent = os.path.dirname(os.path.abspath(outp))
    os.makedirs(parent, exist_ok=True)
    with open(outp, "w", encoding="ascii") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print_report(doc)
    print("wrote %s" % outp)
    if check is not None:
        print("")
        print("CONFIG CHECK against %s" % a.config)
        print("  parameters from: %s" % json.dumps(doc["sources"], sort_keys=True))
        print("  wells under the class roots: %d, kept after exclude_wells: %d; per class %s"
              % (check["n_config_wells"], check["n_config_wells_kept"],
                 json.dumps(check["per_class"], sort_keys=True)))
        for msg in check["info"]:
            print("  INFO %s" % msg)
        for field, cfg_v, census_v in check["diffs"]:
            print("  DIFF %-45s config %s  census %s" % (field, json.dumps(cfg_v), json.dumps(census_v)))
    tail = "PROBE %s" % ("OK" if status == "OK" else status)
    if check is not None:
        if check["diffs"]:
            tail += "; CONFIG DIFFERS on: %s" % ", ".join(d[0] for d in check["diffs"])
        else:
            tail += "; CONFIG MATCHES (%d wells to extract)" % check["n_config_wells_kept"]
    print(tail)
    if status != "OK":
        return 1
    if check is not None and check["diffs"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
