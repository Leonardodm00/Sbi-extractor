#!/usr/bin/env python3
"""
preflight_giulia_export.py -- read-only probe run BEFORE any Sbi-extractor
export of the campaign_cadex_hhgap_v{1,2,5} campaigns.

Why this exists
---------------
The export of record (ANN/SBI_export_r2) was built from campaign_cadex_rho1300v*:
sweep_group 'neuron_synapse', 23 active indices, p = 26 label axes, and a
9-electrode pooled-IFR observable (n_side = 3).  The hhgap campaigns are a
different sweep (mode Full, sweep_group synapse_astro) and the new MEA runs use
n_side = 1 and n_side = 2, i.e. 1 and 4 electrodes.  Three things therefore have
to be measured, not assumed, before the extractor is pointed at them:

  A. do the MEA units pair 1:1 with sim units (launch_sweep_exports.sh requires
     both sides to exist before it submits anything);
  B. what the label-axis signature actually is, per campaign, and whether it is
     single-valued across the whole set (if not, the shards carry different
     th_* columns and cannot be pooled);
  C. what n_e actually is in the recorded probe geometry, and by how much the
     observable therefore departs from the 9-electrode real arm.

It imports nothing from Sbi-extractor, needs no torch and no checkpoint, opens
no file for writing inside either tree, and can be run from any directory.

Usage
-----
  python3 preflight_giulia_export.py \
      --mea-root /davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out \
      --mea-root /davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_1electrode \
      --sim-root /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro \
      --extractor /davinci-1/home/ldellamea/repos/Sbi-extractor \
      --json preflight_hhgap.json

Add --iters-per-unit 3 to sample more than one iter file per sweep task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.stderr.write("FATAL: numpy is required (activate sbi_env first)\n")
    raise SystemExit(2)


# Keys we care about in job_args.json / manifest.json.  Anything not in this
# list is still reported once, as a raw key listing, so an unexpected schema
# surfaces instead of being silently ignored.
JOB_ARG_KEYS = (
    "simtime", "mode", "sweep_group", "conn_rule", "conn_periodic",
    "density", "n_topologies", "n_params_per_worker",
    "conn_prob_lo", "conn_prob_hi",
    "p0_conn_lo", "p0_conn_hi", "d0_conn_lo", "d0_conn_hi",
    "beta_conn_lo", "beta_conn_hi",
    "_resolved_seed_master",
)

MEA_ITER_RE = re.compile(r"^mea_iter_\d+\.npz$")
SIM_ITER_RE = re.compile(r"^iter_\d+\.npz$")


# --------------------------------------------------------------------------
# discovery -- knows about the directory layout, nothing else
# --------------------------------------------------------------------------

def find_units(root, iter_re):
    """Return OrderedDict relpath -> abspath for every directory under `root`
    that directly contains at least one topo_*/ holding a matching npz.

    Layout-tolerant on purpose: the array runs write
    <root>/<campaign>/<task>/topo_*/, but the earlier hand-run single task
    wrote <root>/v1/<task>/topo_*/, and one confirmed test wrote straight to
    <root>/<task>/topo_*/.  All three are found; the relpath is reported so a
    mismatch is visible rather than guessed at.
    """
    units = OrderedDict()
    if not os.path.isdir(root):
        return units
    for dirpath, dirnames, _filenames in os.walk(root):
        topos = sorted(d for d in dirnames if d.startswith("topo_"))
        if not topos:
            continue
        has_iter = False
        for t in topos:
            tdir = os.path.join(dirpath, t)
            try:
                entries = os.listdir(tdir)
            except OSError:
                continue
            if any(iter_re.match(f) for f in entries):
                has_iter = True
                break
        if has_iter:
            rel = os.path.relpath(dirpath, root)
            units[rel] = dirpath
            dirnames[:] = []  # do not descend into topo_* dirs
    return units


def list_iters(unit_dir, iter_re, limit):
    """Up to `limit` iter files under unit_dir/topo_*/, in sorted order."""
    out = []
    for t in sorted(os.listdir(unit_dir)):
        if not t.startswith("topo_"):
            continue
        tdir = os.path.join(unit_dir, t)
        if not os.path.isdir(tdir):
            continue
        for f in sorted(os.listdir(tdir)):
            if iter_re.match(f):
                out.append(os.path.join(tdir, f))
                if len(out) >= limit:
                    return out
    return out


def count_iters(unit_dir, iter_re):
    n_topo, n_iter = 0, 0
    for t in sorted(os.listdir(unit_dir)):
        tdir = os.path.join(unit_dir, t)
        if not (t.startswith("topo_") and os.path.isdir(tdir)):
            continue
        n_topo += 1
        n_iter += sum(1 for f in os.listdir(tdir) if iter_re.match(f))
    return n_topo, n_iter


# --------------------------------------------------------------------------
# readers -- pure, return plain dicts, never print
# --------------------------------------------------------------------------

def read_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:  # malformed json is a real, seen failure mode
        return None, "unreadable: %s" % exc


def read_sim_unit(unit_dir, iters_per_unit):
    """job_args.json / manifest.json / one iter_*.npz from a sim sweep task."""
    rec = {"dir": unit_dir}
    job, err = read_json(os.path.join(unit_dir, "job_args.json"))
    rec["job_args_error"] = err
    rec["job_args_keys"] = sorted(job.keys()) if isinstance(job, dict) else None
    if isinstance(job, dict):
        # kernel bounds are PRESENT-but-null when not swept: .get(k) is not None
        rec["job_args"] = {k: job.get(k) for k in JOB_ARG_KEYS if job.get(k) is not None}
        rec["job_args_null"] = [k for k in JOB_ARG_KEYS if k in job and job.get(k) is None]
        axd = job.get("_axis_declaration")
        rec["axis_declaration"] = axd if isinstance(axd, dict) else None
    man, err = read_json(os.path.join(unit_dir, "manifest.json"))
    rec["manifest_error"] = err
    if isinstance(man, dict):
        rec["manifest_version"] = man.get("manifest_version")
        rec["sweep_group"] = man.get("sweep_group")
        ai = man.get("active_indices")
        rec["n_active_indices"] = len(ai) if isinstance(ai, list) else None
        rec["active_indices"] = ai if isinstance(ai, list) else None
        if rec.get("axis_declaration") is None and isinstance(man.get("axis_declaration"), dict):
            rec["axis_declaration"] = man["axis_declaration"]
    rec["n_topo"], rec["n_iter"] = count_iters(unit_dir, SIM_ITER_RE)

    samples = []
    for p in list_iters(unit_dir, SIM_ITER_RE, iters_per_unit):
        try:
            with np.load(p, allow_pickle=False) as d:
                keys = sorted(d.files)
                s = {"file": os.path.basename(p), "keys": keys}
                for k in ("params", "theta"):
                    if k in d:
                        s["len_" + k] = int(np.asarray(d[k]).ravel().size)
                s["topology_keys"] = [k for k in
                                      ("conn_prob", "p0_conn", "d0_conn", "beta_conn")
                                      if k in keys]
                samples.append(s)
        except Exception as exc:
            samples.append({"file": os.path.basename(p), "error": str(exc)})
    rec["iter_samples"] = samples
    return rec


def read_mea_unit(unit_dir, iters_per_unit):
    """One or more mea_iter_*.npz: probe geometry, detection load, simtime."""
    rec = {"dir": unit_dir}
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
                keys = sorted(d.files)
                s["keys"] = keys
                if "electrode_centers" in d:
                    ec = np.asarray(d["electrode_centers"])
                    s["n_e"] = int(ec.shape[0])
                    s["electrode_centers"] = ec.tolist()
                if "sigma" in d:
                    s["sigma_shape"] = list(np.asarray(d["sigma"]).shape)
                    s["sigma"] = [float(v) for v in np.asarray(d["sigma"]).ravel()]
                for k in ("fs", "simtime"):
                    if k in d:
                        s[k] = float(np.asarray(d[k]).ravel()[0])
                for k in ("params", "theta"):
                    if k in d:
                        s["len_" + k] = int(np.asarray(d[k]).ravel().size)
                if "det_t" in d:
                    det_t = np.asarray(d["det_t"]).ravel()
                    s["n_det"] = int(det_t.size)
                    s["det_t_max"] = float(det_t.max()) if det_t.size else None
                if "det_ch" in d:
                    ch = np.asarray(d["det_ch"]).ravel().astype(int)
                    s["det_ch_min"] = int(ch.min()) if ch.size else None
                    s["det_ch_max"] = int(ch.max()) if ch.size else None
                    if ch.size and s.get("n_e"):
                        s["det_per_channel"] = np.bincount(
                            ch, minlength=s["n_e"]).tolist()
                if "det_src_neuron" in d:
                    src = np.asarray(d["det_src_neuron"]).ravel()
                    s["n_false_positive"] = int((src < 0).sum())
                if "meta_json" in d:
                    raw = np.asarray(d["meta_json"]).ravel()[0]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    try:
                        meta = json.loads(str(raw))
                        s["meta_probe"] = _pluck_probe(meta)
                    except Exception as exc:
                        s["meta_json_error"] = str(exc)
        except Exception as exc:
            s["error"] = str(exc)
        samples.append(s)
    rec["iter_samples"] = samples
    return rec


def _pluck_probe(meta):
    """Pull the probe/noise knobs out of a nested PipelineConfig dict."""
    wanted = ("n_side", "pitch", "edge", "n_sub", "target_noise_uv",
              "snr_ref", "gamma", "n_dec", "r_ref", "k_thresh", "fs")
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in wanted and not isinstance(v, (dict, list)):
                    found.setdefault(k, v)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(meta)
    return found


# --------------------------------------------------------------------------
# analysis -- signatures and pairing, no IO
# --------------------------------------------------------------------------

def label_signature(sim_rec):
    """The tuple that must be single-valued across every unit exported into one
    pooled bank.  Two units with different signatures produce shards with
    different th_* columns; the contract check (A2) fails at load time."""
    ja = sim_rec.get("job_args") or {}
    axd = sim_rec.get("axis_declaration") or {}
    def _axes(name):
        v = axd.get(name)
        return tuple(sorted(v)) if isinstance(v, list) else None
    return (
        ja.get("mode"),
        sim_rec.get("sweep_group") or ja.get("sweep_group"),
        ja.get("conn_rule"),
        sim_rec.get("n_active_indices"),
        _axes("swept_axes"),
        _axes("consumed_axes"),
        _axes("inert_axes"),
    )


def prior_signature(sim_rec):
    """Bounds that define the BoxUniform prior.  Same rule as above: pooling
    shards drawn from different boxes silently mixes two priors."""
    ja = sim_rec.get("job_args") or {}
    keys = [k for k in JOB_ARG_KEYS if k.endswith(("_lo", "_hi"))]
    return tuple((k, ja.get(k)) for k in keys)


def pair_units(mea_units, sim_units):
    """MEA relpath -> sim relpath.  Returns (paired, mea_only, sim_only).

    Matching is on the full trailing <campaign>/<task>.  A leaf-only match
    (<task> alone) is allowed ONLY when the MEA relpath has a single component
    AND that task name is unique across the whole sim tree -- sweep task names
    repeat across campaigns (every campaign has a sweep_intel_task0000), so an
    unconditional leaf fallback silently pairs an orphan campaign to an
    unrelated one's task and exports the wrong theta with the right-looking z.
    """
    pair_index, leaf_counts, leaf_index = {}, Counter(), {}
    for rel in sim_units:
        parts = rel.split(os.sep)
        pair_index.setdefault(os.sep.join(parts[-2:]), rel)
        leaf_counts[parts[-1]] += 1
        leaf_index.setdefault(parts[-1], rel)
    paired, mea_only = OrderedDict(), []
    matched_sim = set()
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
            matched_sim.add(hit)
    sim_only = [r for r in sim_units if r not in matched_sim]
    return paired, mea_only, sim_only


def scan_cli_flags(path):
    """Declared --flags of a python entry point, by AST (no import, no env)."""
    import ast
    try:
        tree = ast.parse(open(path, "r", errors="replace").read())
    except Exception as exc:
        return {"error": str(exc)}
    flags = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            for a in n.args:
                if isinstance(a, ast.Constant) and str(a.value).startswith("-"):
                    flags.append(str(a.value))
    return {"flags": sorted(set(flags))}


def scan_shell_vars(path):
    """Environment variables a shell entry point reads with a default."""
    try:
        text = open(path, "r", errors="replace").read()
    except Exception as exc:
        return {"error": str(exc)}
    return {"env_vars": sorted(set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)(?::-|\})", text)))}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report(result, expect_ne):
    hr("A. PAIRING  (launch_sweep_exports.sh needs BOTH sides present)")
    for mroot, blk in result["mea_roots"].items():
        print("\nMEA root : %s" % mroot)
        print("  units found        : %d" % len(blk["units"]))
        print("  paired with sim    : %d" % len(blk["paired"]))
        if blk["mea_only"]:
            print("  MEA WITHOUT SIM    : %d  -> %s"
                  % (len(blk["mea_only"]), blk["mea_only"][:5]))
        tot_iter = sum(u.get("n_iter", 0) for u in blk["mea_details"].values())
        print("  mea_iter_*.npz     : %d across %d topo dirs"
              % (tot_iter, sum(u.get("n_topo", 0) for u in blk["mea_details"].values())))
    if result["sim_only"]:
        print("\nSim units with NO MEA output at any root: %d -> %s"
              % (len(result["sim_only"]), result["sim_only"][:5]))

    hr("B. LABEL-AXIS SIGNATURE  (this is what makes p flexible)")
    sigs = result["label_signatures"]
    print("distinct signatures across all paired sim units: %d" % len(sigs))
    for i, (sig, units) in enumerate(sigs.items()):
        mode, group, rule, n_active, swept, consumed, inert = json.loads(sig)
        print("\n  signature %d  (%d unit(s), e.g. %s)" % (i, len(units), units[0]))
        print("    mode / sweep_group / conn_rule : %s / %s / %s" % (mode, group, rule))
        print("    n_active_indices               : %s" % (n_active,))
        print("    swept axes    (%s) : %s" % (len(swept) if swept else "?", swept))
        print("    consumed axes (%s) : %s" % (len(consumed) if consumed else "?", consumed))
        print("    inert axes    (%s) : %s" % (len(inert) if inert else "?", inert))
    if len(sigs) > 1:
        print("\n  >>> MORE THAN ONE SIGNATURE. Shards from different signatures")
        print("      carry different th_* columns and cannot be pooled.")

    print("\nprior-box signatures: %d distinct" % len(result["prior_signatures"]))
    for sig, units in result["prior_signatures"].items():
        print("  %s   (%d units)" % (json.loads(sig), len(units)))

    hr("C. REGISTRY WIDTH  (len(params) recorded vs registry PARAM_NAMES)")
    print("recorded len(params) values : %s" % dict(result["params_len"]))
    print("recorded len(theta)  values : %s" % dict(result["theta_len"]))
    if result["registry"] is not None:
        print("registry PARAM_NAMES        : %s  (%s)"
              % (result["registry"]["n"], result["registry"]["src"]))
        if result["registry"]["n"] not in result["params_len"]:
            print("  >>> MISMATCH: the deployed registry is not the one that")
            print("      produced these files. sbi_labels would index the wrong axes.")

    hr("D. OBSERVABLE PARITY  (n_e, simtime, detection load)")
    print("real arm reference: n_e = %d electrodes per pooled subregion" % expect_ne)
    for mroot, blk in result["mea_roots"].items():
        print("\nMEA root : %s" % mroot)
        for key, geo in blk["geometry"].items():
            print("  %s" % key)
            print("    n_e                    : %s   %s"
                  % (geo.get("n_e"),
                     "OK" if geo.get("n_e") == expect_ne else "PARITY BREAK"))
            print("    probe config           : %s" % (geo.get("meta_probe"),))
            print("    fs / simtime (in npz)  : %s / %s"
                  % (geo.get("fs"), geo.get("simtime")))
            print("    simtime (job_args)     : %s" % (geo.get("simtime_job_args"),))
            if (geo.get("simtime") is not None
                    and geo.get("simtime_job_args") is not None
                    and abs(geo["simtime"] - geo["simtime_job_args"]) > 1e-6):
                print("      >>> the npz simtime is INFERRED from the last spike;")
                print("          pass T from job_args, not from the npz.")
            print("    detections / iter      : %s" % (geo.get("n_det"),))
            print("    per-channel detections : %s" % (geo.get("det_per_channel"),))
            print("    rate [Hz/electrode]    : %s" % (geo.get("rate_hz_per_electrode"),))

    hr("E. SEED / DUPLICATION")
    sm = result["seed_masters"]
    print("sim units read            : %d" % sm["n_units"])
    print("distinct _resolved_seed_master : %d" % sm["n_distinct"])
    if sm["duplicates"]:
        print("seeds appearing more than once : %d -> %s"
              % (len(sm["duplicates"]), sm["duplicates"][:6]))
        print("  >>> these units are byte-identical replays; dedupe before pooling.")

    hr("F. EXTRACTOR ENTRY POINTS  (declared flags, read from the files)")
    for name, info in result["entry_points"].items():
        print("  %-28s %s" % (name, info))

    hr("DONE")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_result(mea_roots, sim_root, extractor, registry_src, iters_per_unit):
    sim_units = find_units(sim_root, SIM_ITER_RE)
    sim_details = OrderedDict(
        (rel, read_sim_unit(path, iters_per_unit)) for rel, path in sim_units.items())

    result = {
        "sim_root": sim_root,
        "n_sim_units": len(sim_units),
        "mea_roots": OrderedDict(),
        "label_signatures": OrderedDict(),
        "prior_signatures": OrderedDict(),
        "params_len": Counter(),
        "theta_len": Counter(),
        "registry": None,
        "seed_masters": {},
        "entry_points": OrderedDict(),
        "sim_only": [],
    }

    matched_sim_all = set()
    for mroot in mea_roots:
        mea_units = find_units(mroot, MEA_ITER_RE)
        paired, mea_only, _ = pair_units(mea_units, sim_units)
        matched_sim_all |= set(paired.values())
        mea_details = OrderedDict(
            (rel, read_mea_unit(mea_units[rel], iters_per_unit)) for rel in mea_units)

        geometry = OrderedDict()
        for rel, mrec in mea_details.items():
            if not mrec["iter_samples"]:
                continue
            s = mrec["iter_samples"][0]
            geo = {k: s.get(k) for k in
                   ("n_e", "fs", "simtime", "n_det", "det_per_channel",
                    "meta_probe", "n_false_positive", "sigma")}
            sim_rel = paired.get(rel)
            if sim_rel is not None:
                geo["simtime_job_args"] = (
                    sim_details[sim_rel].get("job_args") or {}).get("simtime")
            T = geo.get("simtime_job_args") or geo.get("simtime")
            if T and geo.get("n_e") and geo.get("n_det") is not None:
                geo["rate_hz_per_electrode"] = round(
                    geo["n_det"] / (float(T) * geo["n_e"]), 4)
            geometry[rel] = geo

        result["mea_roots"][mroot] = {
            "units": list(mea_units),
            "paired": paired,
            "mea_only": mea_only,
            "mea_details": mea_details,
            "geometry": geometry,
        }

    result["sim_only"] = [r for r in sim_units if r not in matched_sim_all]

    for rel in sorted(matched_sim_all):
        rec = sim_details[rel]
        lsig = json.dumps(label_signature(rec))
        result["label_signatures"].setdefault(lsig, []).append(rel)
        psig = json.dumps(prior_signature(rec))
        result["prior_signatures"].setdefault(psig, []).append(rel)
        for s in rec["iter_samples"]:
            if "len_params" in s:
                result["params_len"][s["len_params"]] += 1
            if "len_theta" in s:
                result["theta_len"][s["len_theta"]] += 1

    seeds = [(rec.get("job_args") or {}).get("_resolved_seed_master")
             for rel, rec in sim_details.items() if rel in matched_sim_all]
    seeds = [s for s in seeds if s is not None]
    cnt = Counter(seeds)
    result["seed_masters"] = {
        "n_units": len(seeds),
        "n_distinct": len(cnt),
        "duplicates": sorted(k for k, v in cnt.items() if v > 1),
    }

    if registry_src and os.path.isfile(registry_src):
        import ast
        try:
            tree = ast.parse(open(registry_src, "r", errors="replace").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if "PARAM_NAMES" in names:
                        val = ast.literal_eval(node.value)
                        result["registry"] = {"n": len(val), "names": list(val),
                                              "src": registry_src}
                        break
        except Exception as exc:
            result["registry"] = {"error": str(exc), "src": registry_src}

    if extractor:
        for fname in ("preflight_label_axes.py", "example_export.py",
                      "sbi_labels.py", "check_preprocessing_parity.py"):
            p = os.path.join(extractor, fname)
            result["entry_points"][fname] = (
                scan_cli_flags(p) if os.path.isfile(p) else {"error": "not found"})
        for fname in ("env.sh", "submit_sbi_export.sh", "launch_sweep_exports.sh"):
            p = os.path.join(extractor, fname)
            result["entry_points"][fname] = (
                scan_shell_vars(p) if os.path.isfile(p) else {"error": "not found"})
        p = os.path.join(extractor, "artifacts", "label_axes.json")
        obj, err = read_json(p)
        result["entry_points"]["artifacts/label_axes.json"] = (
            {"error": err} if obj is None
            else {"n_axes": len(obj.get("param_names", []))
                  if isinstance(obj, dict) else "unknown",
                  "keys": sorted(obj.keys()) if isinstance(obj, dict) else None})

    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mea-root", action="append", required=True,
                    help="virtual-MEA output root; repeat for several probe configs")
    ap.add_argument("--sim-root", required=True,
                    help="directory holding campaign_*/sweep_*_task*/ (Giulia_Astro)")
    ap.add_argument("--extractor", default=None,
                    help="path to repos/Sbi-extractor (optional; reads declared flags)")
    ap.add_argument("--registry-src", default=None,
                    help="HPC_single_run.py whose PARAM_NAMES to compare against")
    ap.add_argument("--iters-per-unit", type=int, default=1,
                    help="iter files to open per sweep task (default 1)")
    ap.add_argument("--expect-ne", type=int, default=9,
                    help="electrodes per pooled subregion in the real arm (default 9)")
    ap.add_argument("--json", default=None, help="also write the full result here")
    args = ap.parse_args(argv)

    result = build_result(args.mea_root, args.sim_root, args.extractor,
                          args.registry_src, args.iters_per_unit)
    report(result, args.expect_ne)

    if args.json:
        def default(o):
            if isinstance(o, Counter):
                return dict(o)
            return str(o)
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2, default=default)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
