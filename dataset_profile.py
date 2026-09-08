#!/usr/bin/env python3
"""
dataset_profile.py -- what kind of dataset is this, and can it be pooled with
that one?

Generalises preflight_giulia_export.py.  That script answered one question
about one pair of roots with the reference values (n_e = 9, p = 26) hardcoded.
This module answers the same questions about ANY of the four dataset kinds the
pipeline touches, with no reference values baked in: it builds a uniform
Profile, and parity is decided by DIFFING two Profiles against an explicit
contract rather than by comparing against a constant.

Four kinds, auto-detected:

  sim_campaign    <root>/[<campaign>/]<task>/{job_args.json,manifest.json,
                  topo_*/iter_*.npz}                -- the simulator's own output
  mea_output      <root>/[<campaign>/]<task>/topo_*/mea_iter_*.npz
                                                    -- virtual-MEA detections
  real_extracted  <root>/**/*.npz holding an IFR trace
                                                    -- the real cohort archives
  export_shard    <stem>.parquet + <stem>.json      -- an embedded bank

Public API
----------
  profile_dataset(path, ...)      -> Profile (a plain dict, JSON-serialisable)
  compare_profiles(a, b)          -> parity report (hard breaks / soft diffs)
  check_internal_consistency(p)   -> list of warnings about ONE dataset
  PARITY_CONTRACT                 -> the fields and their class

CLI
---
  python3 dataset_profile.py PATH [PATH ...] \
      [--sim-root DIR] [--registry-src HPC_single_run.py] \
      [--window-s 180.0] [--iters-per-unit 2] [--json out.json]

With two or more paths it prints a pairwise parity verdict for every pair.

Reads only.  numpy is required; pyarrow is optional (used only to count rows
and check ||z|| in an export shard -- the sidecar alone carries the contract).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, OrderedDict

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("FATAL: numpy is required (activate sbi_env first)\n")
    raise SystemExit(2)


MEA_ITER_RE = re.compile(r"^mea_iter_\d+\.npz$")
SIM_ITER_RE = re.compile(r"^iter_\d+\.npz$")

# job_args.json keys worth carrying.  Kernel bounds are PRESENT-but-null when
# not swept, so every read goes through `.get(k) is not None`, never `k in d`.
JOB_ARG_SCALARS = ("simtime", "mode", "sweep_group", "conn_rule",
                   "conn_periodic", "density", "_resolved_seed_master")
PRIOR_BOUND_KEYS = ("conn_prob_lo", "conn_prob_hi",
                    "p0_conn_lo", "p0_conn_hi",
                    "d0_conn_lo", "d0_conn_hi",
                    "beta_conn_lo", "beta_conn_hi")

# Candidate key names for the real archives.  The exact schema of those files
# has not been read in this conversation, so the reader scans for any of these
# and reports the full key list either way rather than assuming one layout.
REAL_TRACE_KEYS = ("ifr_trace", "trace", "ifr", "R_norm")
REAL_META_KEYS = ("fs_ifr", "T_rec", "culture_id", "condition", "n_electrodes",
                  "electrodes_per_subset", "subregion", "dt", "sigma_sm")


# ==========================================================================
# the parity contract -- the only place reference values would ever live, and
# deliberately it holds none: a field is HARD if two datasets must agree on it
# to share one frozen encoder and one prior box, SOFT if a difference is worth
# printing but does not by itself invalidate anything.
# ==========================================================================

PARITY_CONTRACT = (
    ("observable.n_e", "hard",
     "electrodes pooled per observable; the mean is taken over exactly these"),
    ("observable.fs_ifr", "hard",
     "IFR sampling rate; set by the frozen checkpoint, not by the data"),
    ("observable.sigma_sm", "hard", "Gaussian smoothing width of the IFR"),
    ("labels.p", "hard", "number of th_* columns"),
    ("labels.param_names", "hard", "identity and order of the label axes"),
    ("labels.coord", "hard", "ln vs linear per axis"),
    ("labels.prior_box", "hard", "the BoxUniform bounds"),
    ("labels.registry_width", "hard",
     "len(PARAM_NAMES) the params vector was recorded against"),
    ("embedding.E", "hard", "embedding dimension"),
    ("embedding.checkpoint_sha256", "hard", "encoder identity (assertion A8)"),
    ("observable.T", "soft",
     "trace duration; only fatal when T < window_s, which yields zero windows"),
    ("observable.fs_acq", "soft", "raw acquisition rate before IFR binning"),
    ("labels.mode", "soft", "Full vs Neuronal"),
    ("labels.sweep_group", "soft", "which registry group was swept"),
    ("labels.conn_rule", "soft", "flat vs weibull"),
    ("observable.rate_hz_per_electrode", "soft",
     "mean detected rate; informational, drives the MFR floor"),
)


# ==========================================================================
# small helpers
# ==========================================================================

def _to_py(obj):
    """numpy scalars/arrays -> plain python, so a Profile is JSON-safe."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _get(profile, dotted, default=None):
    node = profile
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, "unreadable: %s" % exc


NONCONSTANT = "__multiple__"


def _mode_or_none(values):
    """The single value if the field is constant across the units read, else
    {NONCONSTANT: {value: count}}.  The return type itself carries whether the
    field was constant, so no caller has to guess from the value's shape -- a
    prior box of plain ints and a tally of counts are otherwise
    indistinguishable dicts."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    try:
        uniq = set(vals)
        if len(uniq) == 1:
            return vals[0]
        return {NONCONSTANT: dict(Counter(vals))}
    except TypeError:                      # unhashable (lists, dicts) -> json
        keys = [json.dumps(v, sort_keys=True, default=str) for v in vals]
        if len(set(keys)) == 1:
            return vals[0]
        return {NONCONSTANT: dict(Counter(keys))}


def is_nonconstant(field):
    """True when _mode_or_none reported the field as varying across units."""
    return isinstance(field, dict) and NONCONSTANT in field


# ==========================================================================
# discovery
# ==========================================================================

def find_units(root, iter_re):
    """relpath -> abspath for every dir directly containing topo_*/<iter npz>."""
    units = OrderedDict()
    if not os.path.isdir(root):
        return units
    for dirpath, dirnames, _files in os.walk(root):
        topos = sorted(d for d in dirnames if d.startswith("topo_"))
        if not topos:
            continue
        hit = False
        for t in topos:
            tdir = os.path.join(dirpath, t)
            try:
                if any(iter_re.match(f) for f in os.listdir(tdir)):
                    hit = True
                    break
            except OSError:
                continue
        if hit:
            units[os.path.relpath(dirpath, root)] = dirpath
            dirnames[:] = []
    return units


def list_iters(unit_dir, iter_re, limit):
    out = []
    for t in sorted(os.listdir(unit_dir)):
        tdir = os.path.join(unit_dir, t)
        if not (t.startswith("topo_") and os.path.isdir(tdir)):
            continue
        for f in sorted(os.listdir(tdir)):
            if iter_re.match(f):
                out.append(os.path.join(tdir, f))
                if len(out) >= limit:
                    return out
    return out


def count_iters(unit_dir, iter_re):
    n_topo = n_iter = 0
    for t in sorted(os.listdir(unit_dir)):
        tdir = os.path.join(unit_dir, t)
        if not (t.startswith("topo_") and os.path.isdir(tdir)):
            continue
        n_topo += 1
        n_iter += sum(1 for f in os.listdir(tdir) if iter_re.match(f))
    return n_topo, n_iter


def pair_units(mea_units, sim_units):
    """MEA relpath -> sim relpath on the full trailing <campaign>/<task>.

    A leaf-only match is allowed only when the MEA relpath has one component
    AND that task name is unique across the sim tree: every campaign has a
    sweep_intel_task0000, so an unconditional leaf fallback pairs an orphan
    campaign to an unrelated one and exports the wrong theta against a
    right-looking z.  (That bug was caught by the smoke test, not by review.)
    """
    pair_index, leaf_counts, leaf_index = {}, Counter(), {}
    for rel in sim_units:
        parts = rel.split(os.sep)
        pair_index.setdefault(os.sep.join(parts[-2:]), rel)
        leaf_counts[parts[-1]] += 1
        leaf_index.setdefault(parts[-1], rel)
    paired, mea_only, matched = OrderedDict(), [], set()
    for rel in mea_units:
        parts = rel.split(os.sep)
        hit = None
        if len(parts) >= 2:
            hit = pair_index.get(os.sep.join(parts[-2:]))
        elif leaf_counts[parts[-1]] == 1:
            hit = leaf_index.get(parts[-1])
        if hit is None:
            mea_only.append(rel)
        else:
            paired[rel] = hit
            matched.add(hit)
    return paired, mea_only, [r for r in sim_units if r not in matched]


def detect_kind(path):
    """Which of the four kinds `path` is, by content and never by name."""
    if os.path.isfile(path):
        if path.endswith(".parquet") or path.endswith(".json"):
            stem = path[: path.rfind(".")]
            if os.path.isfile(stem + ".json"):
                obj, _ = read_json(stem + ".json")
                if isinstance(obj, dict) and "param_names" in obj:
                    return "export_shard"
            return "unknown"
        return "unknown"
    if not os.path.isdir(path):
        return "unknown"
    if find_units(path, MEA_ITER_RE):
        return "mea_output"
    if find_units(path, SIM_ITER_RE):
        return "sim_campaign"
    for dirpath, _d, files in os.walk(path):
        for f in files:
            if f.endswith(".json") and os.path.isfile(
                    os.path.join(dirpath, f[:-5] + ".parquet")):
                return "export_shard"
        for f in files:
            if f.endswith(".npz"):
                try:
                    with np.load(os.path.join(dirpath, f),
                                 allow_pickle=False) as d:
                        if any(k in d.files for k in REAL_TRACE_KEYS):
                            return "real_extracted"
                except Exception:
                    pass
                break
    return "unknown"


# ==========================================================================
# per-kind readers -- each returns the same Profile shape
# ==========================================================================

def _blank_profile(path, kind):
    return {
        "path": os.path.abspath(path),
        "kind": kind,
        "n_units": 0,
        "units": [],
        "observable": {"n_e": None, "fs_acq": None, "fs_ifr": None,
                       "sigma_sm": None, "T": None, "T_source": None,
                       "rate_hz_per_electrode": None},
        "labels": {"p": None, "param_names": None, "coord": None,
                   "prior_box": None, "registry_width": None,
                   "mode": None, "sweep_group": None, "conn_rule": None,
                   "swept_axes": None, "consumed_axes": None,
                   "inert_axes": None, "n_active_indices": None},
        "embedding": {"E": None, "checkpoint_sha256": None, "window_s": None},
        "provenance": {"n_seed_masters": None, "n_distinct_seed": None,
                       "duplicate_seeds": []},
        "detail": {},
        "warnings": [],
    }


def _read_registry_width(registry_src):
    """len(PARAM_NAMES) from HPC_single_run.py, by ast -- no import, so the
    simulator's own dependencies (Brian2) are not needed."""
    if not registry_src or not os.path.isfile(registry_src):
        return None
    try:
        tree = ast.parse(open(registry_src, "r", errors="replace").read())
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "PARAM_NAMES"
                   for t in node.targets):
                try:
                    return len(ast.literal_eval(node.value))
                except Exception:
                    return None
    return None


def _read_sim_unit(unit_dir, iters_per_unit):
    rec = {}
    job, err = read_json(os.path.join(unit_dir, "job_args.json"))
    rec["job_args_error"] = err
    if isinstance(job, dict):
        rec.update({k: job.get(k) for k in JOB_ARG_SCALARS})
        rec["prior_box"] = {k: job.get(k) for k in PRIOR_BOUND_KEYS
                            if job.get(k) is not None}
        rec["prior_null"] = [k for k in PRIOR_BOUND_KEYS
                             if k in job and job.get(k) is None]
        axd = job.get("_axis_declaration")
        if isinstance(axd, dict):
            for k in ("swept_axes", "consumed_axes", "inert_axes", "fixed_axes"):
                if isinstance(axd.get(k), list):
                    rec[k] = sorted(axd[k])
    man, err = read_json(os.path.join(unit_dir, "manifest.json"))
    rec["manifest_error"] = err
    if isinstance(man, dict):
        rec["manifest_version"] = man.get("manifest_version")
        rec.setdefault("sweep_group", man.get("sweep_group"))
        ai = man.get("active_indices")
        rec["n_active_indices"] = len(ai) if isinstance(ai, list) else None
        axd = man.get("axis_declaration")
        if isinstance(axd, dict):
            for k in ("swept_axes", "consumed_axes", "inert_axes", "fixed_axes"):
                if rec.get(k) is None and isinstance(axd.get(k), list):
                    rec[k] = sorted(axd[k])
    rec["n_topo"], rec["n_iter"] = count_iters(unit_dir, SIM_ITER_RE)
    lens = []
    for p in list_iters(unit_dir, SIM_ITER_RE, iters_per_unit):
        try:
            with np.load(p, allow_pickle=False) as d:
                lens.append({"len_params": int(np.asarray(d["params"]).size)
                             if "params" in d else None,
                             "len_theta": int(np.asarray(d["theta"]).size)
                             if "theta" in d else None,
                             "topology_keys": [k for k in
                                               ("conn_prob", "p0_conn",
                                                "d0_conn", "beta_conn")
                                               if k in d.files],
                             "keys": sorted(d.files)})
        except Exception as exc:
            lens.append({"error": str(exc)})
    rec["iter_samples"] = lens
    return rec


def _read_mea_unit(unit_dir, iters_per_unit):
    rec = {}
    rec["n_topo"], rec["n_iter"] = count_iters(unit_dir, MEA_ITER_RE)
    man, err = read_json(os.path.join(unit_dir, "mea_manifest.json"))
    rec["mea_manifest_error"] = err
    if isinstance(man, dict):
        rec["mea_manifest"] = {k: man.get(k) for k in
                               ("n_topos", "total_iters", "total_done")
                               if k in man}
    samples = []
    for p in list_iters(unit_dir, MEA_ITER_RE, iters_per_unit):
        s = {"file": os.path.basename(p)}
        try:
            with np.load(p, allow_pickle=False) as d:
                s["keys"] = sorted(d.files)
                if "electrode_centers" in d:
                    s["n_e"] = int(np.asarray(d["electrode_centers"]).shape[0])
                for k in ("fs", "simtime"):
                    if k in d:
                        s[k] = float(np.asarray(d[k]).ravel()[0])
                if "det_t" in d:
                    s["n_det"] = int(np.asarray(d["det_t"]).size)
                if "det_ch" in d and s.get("n_e"):
                    ch = np.asarray(d["det_ch"]).ravel().astype(int)
                    if ch.size:
                        s["det_per_channel"] = np.bincount(
                            ch, minlength=s["n_e"]).tolist()
                if "det_src_neuron" in d:
                    s["n_false_positive"] = int(
                        (np.asarray(d["det_src_neuron"]).ravel() < 0).sum())
                if "params" in d:
                    s["len_params"] = int(np.asarray(d["params"]).size)
                if "meta_json" in d:
                    raw = np.asarray(d["meta_json"]).ravel()[0]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    try:
                        s["probe"] = _pluck(json.loads(str(raw)),
                                            ("n_side", "pitch", "edge", "n_sub",
                                             "target_noise_uv", "snr_ref",
                                             "gamma", "n_dec", "k_thresh"))
                    except Exception as exc:
                        s["meta_json_error"] = str(exc)
        except Exception as exc:
            s["error"] = str(exc)
        samples.append(s)
    rec["iter_samples"] = samples
    return rec


def _pluck(node, wanted, found=None):
    found = {} if found is None else found
    if isinstance(node, dict):
        for k, v in node.items():
            if k in wanted and not isinstance(v, (dict, list)):
                found.setdefault(k, v)
            _pluck(v, wanted, found)
    elif isinstance(node, list):
        for v in node:
            _pluck(v, wanted, found)
    return found


# ==========================================================================
# the four profilers
# ==========================================================================

def _profile_sim(path, iters_per_unit, registry_src, max_units):
    prof = _blank_profile(path, "sim_campaign")
    units = find_units(path, SIM_ITER_RE)
    rels = list(units)[:max_units] if max_units else list(units)
    recs = OrderedDict((r, _read_sim_unit(units[r], iters_per_unit)) for r in rels)
    prof["n_units"] = len(units)
    prof["units"] = rels
    _fill_labels_from_sim(prof, recs)
    prof["observable"]["T"] = _mode_or_none([r.get("simtime") for r in recs.values()])
    prof["observable"]["T_source"] = "job_args.simtime"
    prof["labels"]["registry_width"] = _read_registry_width(registry_src)
    prof["detail"]["sim_units"] = recs
    return prof


def _fill_labels_from_sim(prof, recs):
    L = prof["labels"]
    for field, key in (("mode", "mode"), ("sweep_group", "sweep_group"),
                       ("conn_rule", "conn_rule"),
                       ("n_active_indices", "n_active_indices")):
        L[field] = _mode_or_none([r.get(key) for r in recs.values()])
    for key in ("swept_axes", "consumed_axes", "inert_axes"):
        L[key] = _mode_or_none([r.get(key) for r in recs.values()])
    L["prior_box"] = _mode_or_none([r.get("prior_box") for r in recs.values()])
    lens = Counter()
    for r in recs.values():
        for s in r.get("iter_samples", []):
            if s.get("len_params"):
                lens[s["len_params"]] += 1
    prof["detail"]["len_params_tally"] = dict(lens)
    if len(lens) == 1:
        prof["detail"]["len_params"] = list(lens)[0]
    seeds = [r.get("_resolved_seed_master") for r in recs.values()]
    seeds = [s for s in seeds if s is not None]
    cnt = Counter(seeds)
    prof["provenance"] = {"n_seed_masters": len(seeds),
                          "n_distinct_seed": len(cnt),
                          "duplicate_seeds": sorted(k for k, v in cnt.items()
                                                    if v > 1)}
    if isinstance(L["swept_axes"], list):
        L["p"] = len(L["swept_axes"])
        prof["detail"]["p_basis"] = ("len(swept_axes) from _axis_declaration; "
                                     "the exported p also depends on "
                                     "label_axes.json exclusions")


def _profile_mea(path, iters_per_unit, sim_root, registry_src, max_units):
    prof = _blank_profile(path, "mea_output")
    units = find_units(path, MEA_ITER_RE)
    rels = list(units)[:max_units] if max_units else list(units)
    recs = OrderedDict((r, _read_mea_unit(units[r], iters_per_unit)) for r in rels)
    prof["n_units"] = len(units)
    prof["units"] = rels

    first = [s for r in recs.values() for s in r["iter_samples"] if "n_e" in s]
    prof["observable"]["n_e"] = _mode_or_none([s.get("n_e") for s in first])
    prof["observable"]["fs_acq"] = _mode_or_none([s.get("fs") for s in first])
    prof["detail"]["probe"] = _mode_or_none([json.dumps(s.get("probe"),
                                                        sort_keys=True)
                                             for s in first if s.get("probe")])
    prof["detail"]["simtime_in_npz"] = _mode_or_none(
        [s.get("simtime") for s in first])
    prof["observable"]["T"] = prof["detail"]["simtime_in_npz"]
    prof["observable"]["T_source"] = ("mea_iter.simtime (INFERRED from the last "
                                      "spike; prefer job_args.simtime)")

    if sim_root:
        sim_units = find_units(sim_root, SIM_ITER_RE)
        paired, mea_only, sim_only = pair_units(units, sim_units)
        prof["detail"]["paired"] = paired
        prof["detail"]["mea_without_sim"] = mea_only
        prof["detail"]["sim_without_mea"] = sim_only
        sim_recs = OrderedDict(
            (paired[r], _read_sim_unit(sim_units[paired[r]], iters_per_unit))
            for r in rels if r in paired)
        if sim_recs:
            _fill_labels_from_sim(prof, sim_recs)
            T_args = _mode_or_none([r.get("simtime") for r in sim_recs.values()])
            prof["detail"]["simtime_in_job_args"] = T_args
            if T_args is not None and not is_nonconstant(T_args):
                prof["observable"]["T"] = T_args
                prof["observable"]["T_source"] = "job_args.simtime (paired)"
        # The simtime trap is per unit, not per dataset: one quiet simulation
        # records a duration far below the requested one while its neighbours
        # are fine, so a dataset-level scalar comparison misses it entirely.
        mismatch = []
        for rel in rels:
            if rel not in paired:
                continue
            samples = [s for s in recs[rel]["iter_samples"]
                       if isinstance(s.get("simtime"), float)]
            if not samples:
                continue
            t_npz = samples[0]["simtime"]
            t_arg = sim_recs[paired[rel]].get("simtime")
            if isinstance(t_arg, (int, float)) and abs(t_npz - t_arg) > 1e-6:
                mismatch.append({"unit": rel, "npz": t_npz, "job_args": t_arg})
        prof["detail"]["simtime_mismatch_units"] = mismatch
        prof["labels"]["registry_width"] = _read_registry_width(registry_src)

    T = prof["observable"]["T"]
    n_e = prof["observable"]["n_e"]
    dets = [s.get("n_det") for s in first if s.get("n_det") is not None]
    if dets and isinstance(T, (int, float)) and isinstance(n_e, int) and T > 0:
        prof["observable"]["rate_hz_per_electrode"] = round(
            float(np.mean(dets)) / (float(T) * n_e), 4)
    prof["detail"]["mea_units"] = recs
    return prof


def _profile_real(path, max_units):
    prof = _blank_profile(path, "real_extracted")
    files = []
    for dirpath, _d, names in os.walk(path):
        for f in sorted(names):
            if f.endswith(".npz"):
                files.append(os.path.join(dirpath, f))
    prof["n_units"] = len(files)
    sample = files[:max_units] if max_units else files[:5]
    prof["units"] = [os.path.relpath(f, path) for f in sample]
    metas = []
    for f in sample:
        m = {"file": os.path.relpath(f, path)}
        try:
            with np.load(f, allow_pickle=False) as d:
                m["keys"] = sorted(d.files)
                tk = next((k for k in REAL_TRACE_KEYS if k in d.files), None)
                if tk:
                    arr = np.asarray(d[tk])
                    m["trace_key"] = tk
                    m["trace_shape"] = list(arr.shape)
                    m["n_e_from_shape"] = (int(arr.shape[0])
                                           if arr.ndim == 2 else None)
                for k in REAL_META_KEYS:
                    if k in d.files:
                        v = np.asarray(d[k]).ravel()
                        m[k] = _to_py(v[0]) if v.size == 1 else _to_py(v)
        except Exception as exc:
            m["error"] = str(exc)
        metas.append(m)
    prof["detail"]["files"] = metas
    prof["observable"]["fs_ifr"] = _mode_or_none([m.get("fs_ifr") for m in metas])
    prof["observable"]["T"] = _mode_or_none([m.get("T_rec") for m in metas])
    prof["observable"]["T_source"] = "T_rec in the archive"
    n_e = _mode_or_none([m.get("electrodes_per_subset") or m.get("n_electrodes")
                         or m.get("n_e_from_shape") for m in metas])
    prof["observable"]["n_e"] = n_e
    if n_e is None:
        prof["warnings"].append(
            "n_e is not recorded in these archives; it comes from the "
            "extraction config (electrodes_per_subset) and MUST be supplied "
            "by hand before any parity claim is made")
    return prof


def _profile_export(path, max_units):
    prof = _blank_profile(path, "export_shard")
    if os.path.isfile(path):
        stems = [path[: path.rfind(".")]]
    else:
        stems = []
        for dirpath, _d, names in os.walk(path):
            for f in sorted(names):
                if f.endswith(".json") and os.path.isfile(
                        os.path.join(dirpath, f[:-5] + ".parquet")):
                    stems.append(os.path.join(dirpath, f[:-5]))
    prof["n_units"] = len(stems)
    stems = stems[:max_units] if max_units else stems
    prof["units"] = [os.path.basename(s) for s in stems]

    sidecars = []
    for s in stems:
        obj, err = read_json(s + ".json")
        if obj is None:
            prof["warnings"].append("sidecar %s: %s" % (s, err))
            continue
        sidecars.append(obj)
    if not sidecars:
        return prof

    def field(fn):
        return _mode_or_none([fn(o) for o in sidecars])

    prof["labels"]["param_names"] = field(
        lambda o: o.get("param_names"))
    prof["labels"]["coord"] = field(lambda o: o.get("coord"))
    prof["labels"]["prior_box"] = field(lambda o: o.get("bounds_theta"))
    pn = prof["labels"]["param_names"]
    prof["labels"]["p"] = len(pn) if isinstance(pn, list) else None
    emb = [o.get("embedding", {}) for o in sidecars]
    prof["embedding"]["E"] = _mode_or_none(
        [e.get("embedding_dim") for e in emb])
    prof["embedding"]["checkpoint_sha256"] = _mode_or_none(
        [e.get("dsn_checkpoint_sha256") for e in emb])
    prof["embedding"]["window_s"] = _mode_or_none(
        [e.get("window_s") or e.get("window_length") for e in emb])
    obs = [o.get("observable", {}) for o in sidecars]
    prof["observable"]["n_e"] = _mode_or_none(
        [o.get("n_electrodes") or o.get("n_e") for o in obs])
    prof["observable"]["fs_ifr"] = _mode_or_none(
        [o.get("fs_ifr") for o in obs])
    prof["observable"]["sigma_sm"] = _mode_or_none(
        [o.get("sigma_sm") or o.get("gaussian_window") for o in obs])
    prof["observable"]["T"] = _mode_or_none([o.get("T") for o in obs])
    prof["detail"]["sidecar_keys"] = sorted(sidecars[0].keys())

    try:                                   # optional: rows and ||z||
        import pyarrow.parquet as pq
        rows, znorm = 0, None
        for s in stems:
            tbl = pq.read_table(s + ".parquet")
            rows += tbl.num_rows
            if znorm is None and prof["embedding"]["E"]:
                E = prof["embedding"]["E"]
                cols = ["z_%03d" % j for j in range(E)]
                if all(c in tbl.column_names for c in cols):
                    Z = np.column_stack([tbl.column(c).to_numpy()
                                         for c in cols])
                    znorm = float(np.abs(np.linalg.norm(Z, axis=1) - 1).max())
        prof["detail"]["n_rows"] = rows
        prof["detail"]["max_abs_znorm_minus_1"] = znorm
    except ImportError:
        prof["detail"]["n_rows"] = "pyarrow not installed; sidecar only"
    return prof


# ==========================================================================
# public API
# ==========================================================================

def profile_dataset(path, kind=None, iters_per_unit=1, sim_root=None,
                    registry_src=None, max_units=None, window_s=None):
    """Characterise one dataset.  Returns a Profile dict; never raises on a
    schema surprise -- unknown fields land in profile['warnings'].

    kind          force the detection ('sim_campaign' | 'mea_output' |
                  'real_extracted' | 'export_shard'); None auto-detects.
    sim_root      only for kind='mea_output': the tree holding the paired
                  job_args.json / manifest.json, which is where the label
                  axes and the authoritative simtime live.
    registry_src  HPC_single_run.py to read PARAM_NAMES from.
    window_s      the encoder's window length, if known; enables the
                  T < window_s check.
    """
    kind = kind or detect_kind(path)
    if kind == "sim_campaign":
        prof = _profile_sim(path, iters_per_unit, registry_src, max_units)
    elif kind == "mea_output":
        prof = _profile_mea(path, iters_per_unit, sim_root, registry_src,
                            max_units)
    elif kind == "real_extracted":
        prof = _profile_real(path, max_units)
    elif kind == "export_shard":
        prof = _profile_export(path, max_units)
    else:
        prof = _blank_profile(path, "unknown")
        prof["warnings"].append(
            "could not identify this path as any known dataset kind")
    prof["warnings"].extend(check_internal_consistency(prof, window_s=window_s))
    return prof


def check_internal_consistency(profile, window_s=None):
    """Problems visible inside ONE dataset, before any cross-dataset diff."""
    w = []
    for dotted in ("observable.n_e", "observable.fs_acq", "observable.fs_ifr",
                   "observable.T", "labels.mode", "labels.sweep_group",
                   "labels.conn_rule", "labels.prior_box", "labels.swept_axes",
                   "labels.param_names", "embedding.E",
                   "embedding.checkpoint_sha256"):
        val = _get(profile, dotted)
        if is_nonconstant(val):
            w.append("%s is NOT constant across this dataset: %s"
                     % (dotted, val[NONCONSTANT]))

    reg = _get(profile, "labels.registry_width")
    rec = _get(profile, "detail.len_params")
    if reg is not None and rec is not None and reg != rec:
        w.append("registry PARAM_NAMES has %d entries but the recorded params "
                 "vector has %d -- the deployed registry did not produce these "
                 "files; label indexing would be wrong" % (reg, rec))

    mm = _get(profile, "detail.simtime_mismatch_units")
    if mm:
        e = mm[0]
        w.append("simtime disagrees in %d of %d paired unit(s), e.g. %s: "
                 "%.4g s in the npz (inferred from the last spike) vs %.4g s "
                 "in job_args -- pass T from job_args or those traces are "
                 "silently truncated and dropped"
                 % (len(mm), len(_get(profile, "detail.paired") or {}),
                    e["unit"], e["npz"], e["job_args"]))
    t_npz = _get(profile, "detail.simtime_in_npz")
    t_arg = _get(profile, "detail.simtime_in_job_args")
    if (not mm and isinstance(t_npz, (int, float))
            and isinstance(t_arg, (int, float)) and abs(t_npz - t_arg) > 1e-6):
        w.append("simtime disagrees dataset-wide: %.4g s in the npz vs %.4g s "
                 "in job_args" % (t_npz, t_arg))
    if is_nonconstant(t_npz):
        w.append("the npz simtime is not constant across units: %s -- some "
                 "simulations went quiet early" % (t_npz[NONCONSTANT],))

    T = _get(profile, "observable.T")
    if window_s and isinstance(T, (int, float)) and T < window_s:
        w.append("T = %.4g s < window_s = %.4g s -- every trace yields zero "
                 "windows and is dropped" % (T, window_s))

    mo = _get(profile, "detail.mea_without_sim") or []
    if mo:
        w.append("%d MEA unit(s) have no paired sim unit: %s"
                 % (len(mo), mo[:5]))
    so = _get(profile, "detail.sim_without_mea") or []
    if so:
        w.append("%d sim unit(s) have no MEA output: %s" % (len(so), so[:5]))

    dup = _get(profile, "provenance.duplicate_seeds") or []
    if dup:
        w.append("%d seed_master value(s) appear in more than one unit -- those "
                 "units are byte-identical replays: %s" % (len(dup), dup[:6]))

    znorm = _get(profile, "detail.max_abs_znorm_minus_1")
    if isinstance(znorm, float) and znorm > 1e-5:
        w.append("assertion A7 would fail: max ||z||-1 = %.3g" % znorm)
    return w


def compare_profiles(a, b):
    """Diff two Profiles against PARITY_CONTRACT.

    Returns {'hard_breaks': [...], 'soft_diffs': [...], 'not_comparable': [...],
             'verdict': 'POOLABLE' | 'BLOCKED'}.  A field where either side is
    None goes to not_comparable: unknown is never reported as agreement.
    """
    hard, soft, unknown = [], [], []
    for dotted, cls, why in PARITY_CONTRACT:
        va, vb = _get(a, dotted), _get(b, dotted)
        if va is None or vb is None:
            unknown.append({"field": dotted, "a": va, "b": vb, "why": why})
            continue
        if json.dumps(va, sort_keys=True, default=str) == \
           json.dumps(vb, sort_keys=True, default=str):
            continue
        entry = {"field": dotted, "a": va, "b": vb, "why": why}
        (hard if cls == "hard" else soft).append(entry)
    return {"a": a["path"], "b": b["path"], "kind_a": a["kind"],
            "kind_b": b["kind"], "hard_breaks": hard, "soft_diffs": soft,
            "not_comparable": unknown,
            "verdict": "BLOCKED" if hard else "POOLABLE"}


# ==========================================================================
# reporting / CLI
# ==========================================================================

def print_profile(p):
    print("\n" + "=" * 78)
    print("%s  [%s]" % (p["path"], p["kind"]))
    print("=" * 78)
    print("units                : %d" % p["n_units"])
    o, L, E = p["observable"], p["labels"], p["embedding"]
    print("observable")
    print("  n_e                : %s" % (o["n_e"],))
    print("  fs_acq / fs_ifr    : %s / %s" % (o["fs_acq"], o["fs_ifr"]))
    print("  sigma_sm           : %s" % (o["sigma_sm"],))
    print("  T                  : %s   (%s)" % (o["T"], o["T_source"]))
    print("  rate [Hz/electrode]: %s" % (o["rate_hz_per_electrode"],))
    print("labels")
    print("  p                  : %s" % (L["p"],))
    print("  mode/group/rule    : %s / %s / %s"
          % (L["mode"], L["sweep_group"], L["conn_rule"]))
    print("  n_active_indices   : %s" % (L["n_active_indices"],))
    print("  registry_width     : %s   (recorded len(params): %s)"
          % (L["registry_width"], p["detail"].get("len_params_tally")))
    print("  swept_axes         : %s" % (L["swept_axes"],))
    print("  prior_box          : %s" % (L["prior_box"],))
    if L["param_names"]:
        print("  param_names        : %s" % (L["param_names"],))
    print("embedding")
    print("  E / window_s       : %s / %s" % (E["E"], E["window_s"]))
    print("  checkpoint sha256  : %s" % (E["checkpoint_sha256"],))
    if p["detail"].get("probe"):
        print("probe config         : %s" % (p["detail"]["probe"],))
    if p["provenance"]["n_seed_masters"]:
        print("seeds                : %d unit(s), %d distinct"
              % (p["provenance"]["n_seed_masters"],
                 p["provenance"]["n_distinct_seed"]))
    if p["warnings"]:
        print("\nWARNINGS (%d):" % len(p["warnings"]))
        for msg in p["warnings"]:
            print("  ! %s" % msg)
    else:
        print("\nno internal-consistency warnings")


def print_comparison(c):
    print("\n" + "-" * 78)
    print("PARITY  %s  [%s]" % (c["a"], c["kind_a"]))
    print("    vs  %s  [%s]" % (c["b"], c["kind_b"]))
    print("-" * 78)
    print("verdict: %s" % c["verdict"])
    for entry in c["hard_breaks"]:
        print("  HARD  %-28s %r  vs  %r" % (entry["field"], entry["a"],
                                            entry["b"]))
        print("        %s" % entry["why"])
    for entry in c["soft_diffs"]:
        print("  soft  %-28s %r  vs  %r" % (entry["field"], entry["a"],
                                            entry["b"]))
    if c["not_comparable"]:
        print("  unknown on one side (NOT agreement): %s"
              % [e["field"] for e in c["not_comparable"]])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="datasets to profile")
    ap.add_argument("--kind", default=None,
                    choices=["sim_campaign", "mea_output", "real_extracted",
                             "export_shard"],
                    help="force the kind instead of auto-detecting")
    ap.add_argument("--sim-root", default=None,
                    help="paired simulator tree, for kind=mea_output")
    ap.add_argument("--registry-src", default=None,
                    help="HPC_single_run.py to read PARAM_NAMES from")
    ap.add_argument("--iters-per-unit", type=int, default=1)
    ap.add_argument("--max-units", type=int, default=None,
                    help="profile only the first N units (fast pass)")
    ap.add_argument("--window-s", type=float, default=None,
                    help="encoder window length, enables the T<window check")
    ap.add_argument("--json", default=None, help="write all profiles here")
    args = ap.parse_args(argv)

    profiles = []
    for path in args.paths:
        p = profile_dataset(path, kind=args.kind,
                            iters_per_unit=args.iters_per_unit,
                            sim_root=args.sim_root,
                            registry_src=args.registry_src,
                            max_units=args.max_units,
                            window_s=args.window_s)
        print_profile(p)
        profiles.append(p)

    comparisons = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            c = compare_profiles(profiles[i], profiles[j])
            print_comparison(c)
            comparisons.append(c)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"profiles": profiles, "comparisons": comparisons},
                      fh, indent=2, default=str)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
