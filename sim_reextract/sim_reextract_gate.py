#!/usr/bin/env python3
"""
sim_reextract_gate.py -- Stage C, C8, step 3: the completion GATE of the
simulated-arm re-extraction and the root-level record (the sim-arm analogue
of Stage D's cohort_manifest.py).

Runs after the PBS array, as run_sim_reextract_gate.pbs or by hand:

    cd ~/repos/Sbi-extractor/sim_reextract && conda activate sbi_export
    python3 sim_reextract_gate.py --plan plan.json --workers 8

WHY A SEPARATE GATE
  depend=afterok on an array is ORDERING, not success-gating, on davinci's
  PBS Pro (measured 2026-09-21, probe_array_depend.sh; decision D-006). A
  subjob can die at the walltime and the dependent job still runs. So this
  is the only success signal: it checks every planned task against the
  plan, names every bad one, and writes nothing when any is (the M15 rule
  of cohort_manifest.py). PASS condition, a line that must appear:
      wrote <out_root>/REEXTRACTION_RECORD.json  sha256 <16 hex>

WHAT IT CHECKS, per task of the plan (status run or done)
  - <out>/mea_manifest.json exists; n_topos and total_iters equal the
    plan's counts from the raw simulations; total_done == total_iters;
    its config carries the plan's n_side, pitch, edge, n_sub, fs, and the
    plan's noise_seed_scheme where the plan names one (D-072; a file
    without the key was seeded 'topo_iter')
  - <out>/mea_env.json exists (written by submit_mea_array.sh before the
    run) and names the plan's library sha256 and the plan's tool files
  - no <out>/topo_*/_failures.log (process_campaign.py appends one entry per
    iteration that raised and does not count it as done); with every
    iteration done the log is from an earlier, partial attempt, recorded
    as a warning in the record rather than a failure
  - every topo_*/ of the raw task has its mea_iter_*.npz counterpart, one
    per iter_*.npz, and nothing more
  - every mea_iter_*.npz: electrode_centers has shape (n_e, 2), its
    meta_json carries the plan's n_side, pitch, edge, n_sub, fs (and
    noise_seed_scheme, as above)
  across tasks
  - one environment: python, numpy, scipy, env prefix, interpreter and the
    tool hashes agree across every mea_env.json (one run, one environment,
    for the record)
  and before anything
  - the cohort manifest re-reads to the plan's digest; the tools folder
    still fingerprints to the plan's label; no record exists yet.

THE RECORD, <out_root>/REEXTRACTION_RECORD.json (+ .sha256): the cohort
manifest's digest, the geometry and its decisions, the campaign set and
every task with its counts, the exclusions (replays among them, D-061)
and the plan's replay report, the tools' label and hashes, the library's
sha256, the environment the array ran in, the plan's and the gate's
versions, timestamps. D-012: every file of the new bank was
produced by the new pipeline and says so.

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cohort_manifest as CM                    # noqa: E402
import sim_reextract_plan as PLAN               # noqa: E402

GATE_VERSION = 1
MEA_ITER_RE = re.compile(r"^mea_iter_(\d+)\.npz$")
ITER_RE = re.compile(r"^iter_(\d+)\.npz$")
ENV_KEYS_ONE_RUN = ("python", "numpy", "scipy", "env_prefix", "executable",
                    "library_sha256", "tools_sha256")


def _load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _close(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def check_task(pack):
    """One task. Returns (name, problems, warnings, env_record, n_files_read)."""
    task, geom, library_sha, tool_hashes = pack
    name = "%s/%s" % (task["campaign"], task["sweep"])
    out = task["out_dir"]
    problems, warnings = [], []
    n_read = 0
    env = None
    complete = False

    man_path = os.path.join(out, "mea_manifest.json")
    if not os.path.isfile(man_path):
        problems.append("no mea_manifest.json (the task did not finish)")
    else:
        try:
            man = _load_json(man_path)
        except (OSError, ValueError) as exc:
            man = None
            problems.append("mea_manifest.json unreadable: %r" % (exc,))
        if man is not None:
            if int(man.get("n_topos", -1)) != int(task["n_topos"]):
                problems.append("n_topos %s, planned %d" % (man.get("n_topos"), task["n_topos"]))
            if int(man.get("total_iters", -1)) != int(task["n_iters"]):
                problems.append("total_iters %s, planned %d" % (man.get("total_iters"), task["n_iters"]))
            if int(man.get("total_done", -1)) != int(man.get("total_iters", -2)):
                problems.append("total_done %s of total_iters %s"
                                % (man.get("total_done"), man.get("total_iters")))
            else:
                complete = True
            cfg = man.get("config") or {}
            for key, want in (("n_side", geom["n_side"]), ("pitch", geom["pitch_um"]),
                              ("edge", geom["edge_um"]), ("n_sub", geom["n_sub"]),
                              ("fs", geom["fs"])):
                if not _close(cfg.get(key), want):
                    problems.append("mea_manifest config %s = %r, planned %r" % (key, cfg.get(key), want))
            # how the noise was seeded (D-072); a plan without the key predates it
            if "noise_seed_scheme" in geom and \
                    cfg.get("noise_seed_scheme", "topo_iter") != geom["noise_seed_scheme"]:
                problems.append("mea_manifest config noise_seed_scheme = %r, planned %r"
                                % (cfg.get("noise_seed_scheme", "topo_iter"), geom["noise_seed_scheme"]))

    env_path = os.path.join(out, "mea_env.json")
    if not os.path.isfile(env_path):
        problems.append("no mea_env.json (the job script that ran did not write one)")
    else:
        try:
            env = _load_json(env_path)
        except (OSError, ValueError) as exc:
            problems.append("mea_env.json unreadable: %r" % (exc,))
            env = None
        if env is not None:
            if env.get("library_sha256") != library_sha:
                problems.append("mea_env library_sha256 %s, planned %s"
                                % (str(env.get("library_sha256"))[:16], library_sha[:16]))
            ts = env.get("tools_sha256") or {}
            bad = [f for f, h in tool_hashes.items() if ts.get(f) != h]
            if bad:
                problems.append("mea_env tools_sha256 differs from the plan's tools on %s" % ", ".join(bad))
            if not env.get("scipy") or not env.get("numpy") or not env.get("python"):
                problems.append("mea_env.json lacks a python / numpy / scipy version")

    # the per-topology files
    for td, n_raw in sorted(task.get("iters_per_topo", {}).items()):
        otd = os.path.join(out, td)
        if not os.path.isdir(otd):
            problems.append("%s: no output directory" % td)
            continue
        if os.path.isfile(os.path.join(otd, "_failures.log")):
            # process_campaign.py appends one entry per iteration that raised
            # and counts that iteration as not done. With every iteration done
            # the log is from an earlier, partial attempt (a resume rewrote
            # every file): recorded as a warning, not a failure.
            if complete:
                warnings.append("%s: _failures.log from an earlier attempt (every iteration is "
                                "done in this one; delete the log if that is right)" % td)
            else:
                problems.append("%s: _failures.log present (an iteration raised)" % td)
        names = sorted(n for n in os.listdir(otd) if MEA_ITER_RE.match(n))
        if len(names) != n_raw:
            problems.append("%s: %d mea_iter_*.npz, raw has %d iter_*.npz" % (td, len(names), n_raw))
        for fn in names:
            path = os.path.join(otd, fn)
            try:
                with np.load(path, allow_pickle=False) as d:
                    ec = d["electrode_centers"]
                    meta = json.loads(str(d["meta_json"]))
                n_read += 1
            except Exception as exc:                        # noqa: BLE001
                problems.append("%s/%s: unreadable: %r" % (td, fn, exc))
                continue
            if tuple(ec.shape) != (geom["n_e"], 2):
                problems.append("%s/%s: electrode_centers shape %r, want (%d, 2)"
                                % (td, fn, tuple(ec.shape), geom["n_e"]))
            for key, want in (("n_side", geom["n_side"]), ("pitch", geom["pitch_um"]),
                              ("edge", geom["edge_um"]), ("n_sub", geom["n_sub"]),
                              ("fs", geom["fs"])):
                if not _close(meta.get(key), want):
                    problems.append("%s/%s: meta_json %s = %r, want %r" % (td, fn, key, meta.get(key), want))
            if "noise_seed_scheme" in geom and \
                    meta.get("noise_seed_scheme", "topo_iter") != geom["noise_seed_scheme"]:
                problems.append("%s/%s: meta_json noise_seed_scheme = %r, want %r"
                                % (td, fn, meta.get("noise_seed_scheme", "topo_iter"), geom["noise_seed_scheme"]))
    return name, problems[:40], warnings, env, n_read


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="plan.json written by sim_reextract_plan.py")
    p.add_argument("--workers", type=int, default=4, help="tasks checked in parallel")
    p.add_argument("--record-out", default=None,
                   help="default <out_root>/%s" % PLAN.RECORD_NAME)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = _load_json(args.plan)
    out_root = plan["out_root"]
    record_path = args.record_out or os.path.join(out_root, PLAN.RECORD_NAME)
    print("[gate] plan      : %s (plan_version %s, created %s)"
          % (os.path.abspath(args.plan), plan.get("plan_version"), plan.get("created")))
    print("[gate] out root  : %s" % out_root)

    fatal = []
    # 0. preconditions
    if os.path.isfile(record_path):
        fatal.append("%s already exists; this root is recorded as finished. Remove it only "
                     "if the whole root is being redone." % record_path)
    cm = plan["cohort_manifest"]
    try:
        manifest = CM.read_manifest(cm["path"])
        if manifest["_digest"] != cm["digest"]:
            fatal.append("cohort manifest %s now reads to %s, the plan recorded %s"
                         % (cm["path"], manifest["_digest"][:16], cm["digest"][:16]))
    except (OSError, CM.ManifestError) as exc:
        fatal.append("cohort manifest: %s" % exc)
    try:
        files, label = PLAN.fingerprint_tools(plan["tools"]["dir"])
        if files != plan["tools"]["files"]:
            fatal.append("the tools in %s changed since the plan (%s)" % (
                plan["tools"]["dir"],
                ", ".join(f for f in files if files[f] != plan["tools"]["files"].get(f))))
    except PLAN.PlanError as exc:
        fatal.append(str(exc))
    lib = plan["library"]
    if not os.path.isfile(lib["path"]) or PLAN.sha256_file(lib["path"]) != lib["sha256"]:
        fatal.append("the template library %s is missing or no longer hashes to %s"
                     % (lib["path"], lib["sha256"][:16]))
    if fatal:
        for f in fatal:
            print("[gate] REFUSED: %s" % f)
        print("[gate] REFUSED -- nothing written")
        return 1

    # 1. every task
    tasks = [t for t in plan["tasks"] if t.get("status") in ("run", "done")]
    geom = plan["geometry"]
    packs = [(t, geom, lib["sha256"], plan["tools"]["files"]) for t in tasks]
    print("[gate] tasks     : %d (%d iterations planned), %d worker(s)"
          % (len(tasks), sum(t["n_iters"] for t in tasks), max(1, args.workers)))
    if args.workers > 1 and len(packs) > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(check_task, packs)
    else:
        results = [check_task(p) for p in packs]

    bad = [(n, p) for n, p, _, _, _ in results if p]
    warned = [(n, w) for n, _, w, _, _ in results if w]
    n_read = sum(r[4] for r in results)
    envs = {n: e for n, _, _, e, _ in results if e is not None}

    # 2. one environment across the run
    env_problems = []
    if envs:
        first = envs[sorted(envs)[0]]
        for key in ENV_KEYS_ONE_RUN:
            vals = {}
            for n, e in envs.items():
                vals.setdefault(json.dumps(e.get(key), sort_keys=True), []).append(n)
            if len(vals) > 1:
                env_problems.append("%s differs across tasks: %s" % (key, "; ".join(
                    "%s x%d (e.g. %s)" % (v[:60], len(ns), ns[0]) for v, ns in vals.items())))
    else:
        first = None

    for n, p in bad:
        print("[gate] FAIL %s" % n)
        for line in p:
            print("[gate]      - %s" % line)
    for line in env_problems:
        print("[gate] FAIL environment: %s" % line)
    for n, w in warned:
        for line in w:
            print("[gate] WARN %s: %s" % (n, line))
    print("[gate] read %d mea_iter_*.npz file(s) across %d task(s); %d task(s) bad"
          % (n_read, len(tasks), len(bad)))
    if bad or env_problems or first is None:
        if first is None and not bad:
            print("[gate] FAIL no mea_env.json read")
        print("[gate] REFUSED -- %d bad task(s), nothing written. Re-run those tasks "
              "(plan --resume, then the array) and run this gate again." % len(bad))
        return 1

    # 3. the record
    environment = {k: first.get(k) for k in ("python", "numpy", "scipy", "env_name",
                                              "env_prefix", "executable", "tools_dir")}
    hosts = sorted(set(str(e.get("host")) for e in envs.values()))
    record = {
        "record": "sim_reextraction", "record_version": GATE_VERSION,
        "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "plan_created": plan.get("created"), "plan_version": plan.get("plan_version"),
        "cohort_manifest": plan["cohort_manifest"],
        "geometry": plan["geometry"],
        "extra_args": plan["extra_args"],
        "sim_main": plan["sim_main"], "campaign_glob": plan["campaign_glob"],
        "campaigns": plan["campaigns"], "out_root": out_root,
        "tools": plan["tools"], "library": plan["library"],
        "environment": environment, "hosts": hosts,
        "contract": plan["contract"],
        "counts": {"tasks": len(tasks), "iterations": sum(t["n_iters"] for t in tasks),
                   "files_read": n_read, "tasks_excluded": len(plan.get("excluded", []))},
        "tasks": [{k: t[k] for k in ("campaign", "sweep", "campaign_dir", "out_dir",
                                      "n_topos", "n_iters", "contract_sig") if k in t}
                  for t in tasks],
        "excluded": plan.get("excluded", []),
        "replays": plan.get("replays"),
        "warnings": [{"task": n, "warning": line} for n, w in warned for line in w],
        "gate": {"script": os.path.basename(__file__), "version": GATE_VERSION},
    }
    blob = json.dumps(record, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()
    tmp = record_path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(blob)
    os.replace(tmp, record_path)
    with open(record_path + ".sha256", "w", encoding="ascii") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(record_path)))
    print("[gate] environment: python %s numpy %s scipy %s, %s"
          % (environment["python"], environment["numpy"], environment["scipy"],
             environment["env_prefix"] or environment["executable"]))
    print("[gate] tools: %s" % plan["tools"]["label"])
    print("wrote %s  sha256 %s" % (record_path, digest[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
