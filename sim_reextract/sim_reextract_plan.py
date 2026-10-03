#!/usr/bin/env python3
"""
sim_reextract_plan.py -- Stage C, C8, step 1: the PLAN of the simulated-arm
re-extraction (the sim-arm analogue of Stage D's list_extraction_jobs.py).

Reads the cohort manifest of record, enumerates the raw simulation tasks of
the DUP15HD campaigns, and writes the task manifest the PBS array reads and
a plan.json the completion gate (sim_reextract_gate.py) checks the run
against. It writes NOTHING under the output root and submits nothing.

    cd ~/repos/Sbi-extractor/sim_reextract && conda activate sbi_export
    python3 sim_reextract_plan.py \
        --cohort-manifest ../artifacts/cohort_manifest/cohort_manifest.json \
        --sim-main /davinci-1/home/ldellamea/ANN/Phenomenological/Main \
        --out-root /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2 \
        --tools    /davinci-1/home/ldellamea/ANN/MEA_analysis

WHAT IT DECIDES
  n_side = isqrt(electrodes_per_subset) from the cohort manifest, refusing
  unless n_side^2 == electrodes_per_subset (decision D-015: the sim arm's
  n_e is CONFIGURED from the real arm's manifest, then checked). The pitch
  (60 um, D-021) and the electrode side (25 um, D-024) are the virtual
  probe's defaults, passed explicitly so the record carries them; the
  sampling rate is the manifest's fs_raw (the real device's), also passed.

WHAT IT REFUSES, and why
  - a cohort manifest without its .sha256 sidecar: read_manifest loads a
    copy without one unchecked (HPC_PATHS.md 3b); the frozen copy must
    carry its digest.
  - electrodes_per_subset that is not a perfect square: the virtual probe
    is an n_side x n_side grid.
  - a tools folder whose files are not a known state of
    Astro-Neuron-Network's hpc/MEA Traces (the record must name the code
    that ran; D-012), or an older known state than the one this plan was
    built for (its job cannot activate sbi_export on davinci; D-023).
  - no template library at --library: launch_mea_array.sh would build a new
    one silently, and two launches could use two libraries without a trace.
  - a campaign set whose tasks disagree on their label contract (simtime,
    prior box, kernel bounds, active axes, sweep group, bounds, log axes):
    one bank needs one contract. --allow-mixed-contract overrides, naming.
  - an output root that already holds detections, unless --resume, which
    keeps every task whose output is complete and re-runs the others; and
    always an output root that holds a REEXTRACTION_RECORD.json (finished).

WHAT IT NAMES INSTEAD OF SKIPPING SILENTLY (decision D-013)
  every sweep task without manifest.json or job_args.json, every sweep
  folder with no topo_*/, every task with topo_*/ but no iter_*.npz.
  They are listed in the plan under "excluded" with their reason and are
  not run: a task without manifest.json cannot be exported (no label
  registry), one without job_args.json would export at a guessed simtime.
  Also every task named with --exclude-task CAMPAIGN/SWEEP (repeatable):
  a task the plan cannot tell is unfinished -- a simulation still being
  written looks like a complete one with fewer iter_*.npz -- is left out by
  name, listed under "excluded" with the reason "left out on the command
  line (--exclude-task)", and so named in the record. A name that matches
  no task under the campaign glob is REFUSED (a typo would otherwise
  exclude nothing, silently).

OUTPUT
  --tasks-out  TSV, one line per task to RUN: campaign_dir <TAB> out_dir,
               the format submit_mea_array.sh reads by line number.
  --plan-out   plan.json: everything above plus per-task topology and
               iteration counts, the tools' fingerprints and label, the
               library's sha256, the contract, and the EXTRA_ARGS string.

HPC note (hpc-python-compat): pure ASCII, LF only; stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cohort_manifest as CM  # noqa: E402

PLAN_VERSION = 1
RECORD_NAME = "REEXTRACTION_RECORD.json"

# The virtual probe's geometry for this re-extraction. The count comes from
# the cohort manifest at run time (D-015); pitch and edge are the decisions.
PITCH_UM = 60.0          # D-021 (2026-09-30): the recording device's fixed pitch
EDGE_UM = 25.0           # D-024 (2026-10-01): the virtual electrode's side stays
N_SUB = 4                # process_campaign.py's default, recorded, not decided

TOOL_FILES = ("process_campaign.py", "mea_probe.py", "mea_detection.py",
              "mea_synthesis.py", "mea_plots.py", "eap_template_library.py",
              "submit_mea_array.sh")

# Known states of Astro-Neuron-Network's hpc/MEA Traces, by sha256 of the
# files that run. The label is what the record carries. The tree id is
# `git rev-parse <commit>:"hpc/MEA Traces"` (file modes as committed, 644).
KNOWN_TOOLS = {
    "ANN main 37bf9f8 (2026-09-30, launcher env fix; tree 0969d9f3)": {
        "process_campaign.py": "0263726ae3e203c194d214f290f79172a146ebd8a5f5a7c56dd0545cccdc6850",
        "mea_probe.py": "537641fe1bcfd37bff28c621f3bbed8b90bb7974f03b27f9a8e74eb8e218d658",
        "mea_detection.py": "9510c2aea223f3abe6edae8c35ffd0704bd5860641f5b28e75db2b04b6312a32",
        "mea_synthesis.py": "6834b7c59b0dbbef4786d64113e22141427c9e5d342c97b441f026f7a9a337bd",
        "mea_plots.py": "48b095056e85c90af4e29c56e90ccc13f1b2e5c40f674ac8b6ebc61f8d2c8c39",
        "eap_template_library.py": "5d32d0e8d1bee7002599223242b49031c3f9efbc345cb38fa7db9568e4232029",
        "submit_mea_array.sh": "b64aec71faf07f9d45cb64ce3a70c2b81561f4ce9376fd3835e59dce0309e7ec",
    },
    "ANN main C8 (2026-10-01, job finds the cluster's conda, writes mea_env.json; tree 6e01860c)": {
        "process_campaign.py": "0263726ae3e203c194d214f290f79172a146ebd8a5f5a7c56dd0545cccdc6850",
        "mea_probe.py": "537641fe1bcfd37bff28c621f3bbed8b90bb7974f03b27f9a8e74eb8e218d658",
        "mea_detection.py": "9510c2aea223f3abe6edae8c35ffd0704bd5860641f5b28e75db2b04b6312a32",
        "mea_synthesis.py": "6834b7c59b0dbbef4786d64113e22141427c9e5d342c97b441f026f7a9a337bd",
        "mea_plots.py": "48b095056e85c90af4e29c56e90ccc13f1b2e5c40f674ac8b6ebc61f8d2c8c39",
        "eap_template_library.py": "5d32d0e8d1bee7002599223242b49031c3f9efbc345cb38fa7db9568e4232029",
        "submit_mea_array.sh": "1575bb80f87c08065190637b496b90fa65091a88579b5f2df6d8c2471e509a3a",
    },
}
# The state this plan is built for: its job activates sbi_export on davinci
# (D-023) and writes the mea_env.json the gate reads. Older known states are
# refused unless --allow-older-tools.
REQUIRED_TOOLS_LABEL = "ANN main C8 (2026-10-01, job finds the cluster's conda, writes mea_env.json; tree 6e01860c)"

SWEEP_RE = re.compile(r"^sweep_[^_]+_task\d+$")
ITER_RE = re.compile(r"^iter_\d+\.npz$")

# The fields of a task's label contract, read from job_args.json and
# manifest.json. One bank needs one value of each across the campaign set.
JOB_ARGS_FIELDS = ("simtime", "conn_prob_lo", "conn_prob_hi",
                   "p0_lo", "p0_hi", "d0_lo", "d0_hi", "beta_lo", "beta_hi")
MANIFEST_FIELDS = ("active_indices", "sweep_group", "param_names",
                   "log_params", "log_transform", "conn_rule", "param_bounds")
REPORTED_ONLY = ("manifest_version",)


class PlanError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm(v):
    """A JSON value as a hashable, rounded signature."""
    if isinstance(v, float):
        return ("f", repr(round(v, 12)))
    if isinstance(v, list):
        return ("l", tuple(_norm(x) for x in v))
    if isinstance(v, dict):
        return ("d", tuple(sorted((k, _norm(x)) for k, x in v.items())))
    return ("s", v)


def _short(v, n=70):
    s = json.dumps(v, sort_keys=True)
    return s if len(s) <= n else s[:n - 3] + "..."


# --------------------------------------------------------------------------- #
# the tools folder
# --------------------------------------------------------------------------- #
def fingerprint_tools(tools_dir):
    missing = [f for f in TOOL_FILES if not os.path.isfile(os.path.join(tools_dir, f))]
    if missing:
        raise PlanError("tools folder %s lacks %s" % (tools_dir, ", ".join(missing)))
    files = {f: sha256_file(os.path.join(tools_dir, f)) for f in TOOL_FILES}
    label = None
    for lab, want in KNOWN_TOOLS.items():
        if all(files.get(f) == h for f, h in want.items()):
            label = lab
            break
    return files, label


def check_tools(tools_dir, allow_older):
    files, label = fingerprint_tools(tools_dir)
    if label is None:
        diffs = []
        want = KNOWN_TOOLS[REQUIRED_TOOLS_LABEL]
        for f in TOOL_FILES:
            if files[f] != want[f]:
                diffs.append("%s (%s, want %s)" % (f, files[f][:16], want[f][:16]))
        raise PlanError(
            "the tools in %s are no known state of Astro-Neuron-Network's "
            "hpc/MEA Traces; against %s these differ: %s. The record must name "
            "the code that ran (D-012); update the folder or add the state to "
            "KNOWN_TOOLS." % (tools_dir, REQUIRED_TOOLS_LABEL.split(" (")[0], "; ".join(diffs)))
    if label != REQUIRED_TOOLS_LABEL and not allow_older:
        raise PlanError(
            "the tools in %s are %s, older than this plan needs (%s): their "
            "submit_mea_array.sh cannot activate sbi_export on davinci and writes "
            "no mea_env.json. Copy the newer submit_mea_array.sh in, or pass "
            "--allow-older-tools." % (tools_dir, label, REQUIRED_TOOLS_LABEL))
    return files, label


# --------------------------------------------------------------------------- #
# the raw simulations
# --------------------------------------------------------------------------- #
def _load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def scan_task(sweep_dir):
    """Everything the plan needs to know about one sweep_<tag>_task<idx>."""
    info = {"sweep": os.path.basename(sweep_dir), "dir": sweep_dir,
            "has_job_args": os.path.isfile(os.path.join(sweep_dir, "job_args.json")),
            "has_manifest": os.path.isfile(os.path.join(sweep_dir, "manifest.json")),
            "topo_dirs": [], "n_iters": 0, "iters_per_topo": {}}
    with os.scandir(sweep_dir) as it:
        for e in it:
            if e.is_dir() and e.name.startswith("topo_"):
                info["topo_dirs"].append(e.name)
    info["topo_dirs"].sort()
    for td in info["topo_dirs"]:
        n = 0
        with os.scandir(os.path.join(sweep_dir, td)) as it:
            for e in it:
                if e.is_file() and ITER_RE.match(e.name):
                    n += 1
        info["iters_per_topo"][td] = n
        info["n_iters"] += n
    return info


def task_contract(sweep_dir):
    """The label-contract fields of one task, each a JSON value or 'absent'."""
    out = {}
    ja = os.path.join(sweep_dir, "job_args.json")
    mf = os.path.join(sweep_dir, "manifest.json")
    j = _load_json(ja) if os.path.isfile(ja) else {}
    m = _load_json(mf) if os.path.isfile(mf) else {}
    for k in JOB_ARGS_FIELDS:
        out[k] = j.get(k, "absent")
    for k in MANIFEST_FIELDS + REPORTED_ONLY:
        out[k] = m.get(k, "absent")
    return out


def enumerate_campaigns(sim_main, campaign_glob):
    pattern = os.path.join(sim_main, campaign_glob)
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not dirs:
        raise PlanError("no campaign directory matches %s" % pattern)
    return dirs


def enumerate_tasks(campaign_dirs, out_root):
    """Returns (tasks, excluded); tasks carry status 'run' at this point."""
    tasks, excluded = [], []
    for cd in campaign_dirs:
        camp = os.path.basename(cd)
        sweeps = sorted(e for e in os.listdir(cd)
                        if SWEEP_RE.match(e) and os.path.isdir(os.path.join(cd, e)))
        for sw in sweeps:
            info = scan_task(os.path.join(cd, sw))
            rec = {"campaign": camp, "sweep": sw, "campaign_dir": info["dir"],
                   "out_dir": os.path.join(out_root, camp, sw),
                   "n_topos": len(info["topo_dirs"]), "n_iters": info["n_iters"],
                   "iters_per_topo": info["iters_per_topo"]}
            reasons = []
            if not info["topo_dirs"] and not info["has_job_args"] and not info["has_manifest"]:
                reasons.append("empty sweep folder (no job_args.json, no manifest.json, no topo_*/)")
            else:
                if not info["has_manifest"]:
                    reasons.append("no manifest.json (no label registry: not exportable)")
                if not info["has_job_args"]:
                    reasons.append("no job_args.json (simtime unknown: the export would guess)")
                if not info["topo_dirs"]:
                    reasons.append("no topo_*/ directories")
                elif info["n_iters"] == 0:
                    reasons.append("topo_*/ present but no iter_*.npz")
            if reasons:
                rec["reason"] = "; ".join(reasons)
                excluded.append(rec)
            else:
                rec["status"] = "run"
                tasks.append(rec)
    return tasks, excluded


EXCLUDE_REASON = "left out on the command line (--exclude-task)"


def exclude_named(tasks, excluded, names):
    """Move the tasks named CAMPAIGN/SWEEP from `tasks` to `excluded`.

    Returns (tasks, excluded, names_sorted). Refuses a malformed name and a
    name that is no task of the enumeration (run or already excluded). A
    task already excluded for its own reason keeps it, with this one added.
    """
    want = []
    for nm in names:
        parts = nm.strip().strip("/").split("/")
        if len(parts) != 2 or not parts[0] or not SWEEP_RE.match(parts[1]):
            raise PlanError("--exclude-task %r is not CAMPAIGN/SWEEP, e.g. "
                            "campaign_cadex_rho1300v12/sweep_cfd_task0003" % nm)
        key = "%s/%s" % (parts[0], parts[1])
        if key not in want:
            want.append(key)
    if not want:
        return tasks, excluded, []
    by_key = {"%s/%s" % (t["campaign"], t["sweep"]): t for t in tasks}
    ex_key = {"%s/%s" % (e["campaign"], e["sweep"]): e for e in excluded}
    unknown = [k for k in want if k not in by_key and k not in ex_key]
    if unknown:
        raise PlanError("--exclude-task names no task under the campaign glob: %s"
                        % ", ".join(unknown))
    for k in want:
        if k in ex_key:
            ex_key[k]["reason"] = "%s; %s" % (ex_key[k]["reason"], EXCLUDE_REASON)
    keep = []
    for t in tasks:
        k = "%s/%s" % (t["campaign"], t["sweep"])
        if k in want:
            rec = {key: t[key] for key in ("campaign", "sweep", "campaign_dir", "out_dir",
                                           "n_topos", "n_iters", "iters_per_topo")}
            rec["reason"] = EXCLUDE_REASON
            excluded.append(rec)
        else:
            keep.append(t)
    excluded.sort(key=lambda e: (e["campaign"], e["sweep"]))
    return keep, excluded, sorted(want)


def check_contract(tasks):
    """One value per field across the tasks, else the mixed fields."""
    values = {}
    for t in tasks:
        c = task_contract(t["campaign_dir"])
        t["contract_sig"] = hashlib.sha256(
            json.dumps({k: c[k] for k in JOB_ARGS_FIELDS + MANIFEST_FIELDS},
                       sort_keys=True).encode()).hexdigest()[:16]
        for k, v in c.items():
            values.setdefault(k, {}).setdefault(_norm(v), {"value": v, "tasks": []})
            values[k][_norm(v)]["tasks"].append("%s/%s" % (t["campaign"], t["sweep"]))
    mixed, single = {}, {}
    for k, by in values.items():
        if len(by) == 1:
            single[k] = next(iter(by.values()))["value"]
        else:
            mixed[k] = [{"value": d["value"], "n_tasks": len(d["tasks"]),
                         "tasks": d["tasks"][:6]} for d in by.values()]
    return single, mixed


# --------------------------------------------------------------------------- #
# the output root
# --------------------------------------------------------------------------- #
def existing_output(out_root):
    """Does out_root already hold detections or a finished record?"""
    if os.path.isfile(os.path.join(out_root, RECORD_NAME)):
        return "record"
    if not os.path.isdir(out_root):
        return None
    for camp in os.listdir(out_root):
        cd = os.path.join(out_root, camp)
        if not os.path.isdir(cd):
            continue
        for sw in os.listdir(cd):
            sd = os.path.join(cd, sw)
            if os.path.isfile(os.path.join(sd, "mea_manifest.json")) or \
               os.path.isfile(os.path.join(sd, "mea_env.json")):
                return "detections"
            if os.path.isdir(sd):
                for td in os.listdir(sd):
                    if td.startswith("topo_") and os.path.isdir(os.path.join(sd, td)):
                        return "detections"
    return None


def task_is_complete(task):
    """A task whose output has a mea_manifest.json reporting every planned
    iteration done, and a mea_env.json. Used by --resume."""
    man = os.path.join(task["out_dir"], "mea_manifest.json")
    env = os.path.join(task["out_dir"], "mea_env.json")
    if not (os.path.isfile(man) and os.path.isfile(env)):
        return False
    try:
        d = _load_json(man)
    except (OSError, ValueError):
        return False
    return (int(d.get("total_done", -1)) == int(d.get("total_iters", -2))
            == int(task["n_iters"]) and int(d.get("n_topos", -1)) == int(task["n_topos"]))


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-manifest", required=True,
                   help="the frozen cohort_manifest.json; its .sha256 must sit beside it")
    p.add_argument("--sim-main", required=True, help="SIM_MAIN_DIR, the raw simulations")
    p.add_argument("--out-root", required=True, help="Outputs_v2 (D-014); must not hold detections unless --resume")
    p.add_argument("--tools", required=True, help="the folder holding process_campaign.py and submit_mea_array.sh")
    p.add_argument("--campaign-glob", default="campaign_cadex_rho1300v*",
                   help="under --sim-main (default: %(default)s; D-025: v1-v5, v7-v12)")
    p.add_argument("--library", default=None,
                   help="the EAP template library (default: <tools>/eap_library.npz)")
    p.add_argument("--plan-out", default="plan.json")
    p.add_argument("--tasks-out", default="tasks.tsv")
    p.add_argument("--resume", action="store_true",
                   help="an output root with detections: keep complete tasks, re-run the rest")
    p.add_argument("--allow-mixed-contract", action="store_true",
                   help="proceed although the tasks disagree on a contract field (named)")
    p.add_argument("--exclude-task", action="append", default=[], metavar="CAMPAIGN/SWEEP",
                   help="leave this task out (e.g. a simulation still being written); named in "
                        "the plan and the record as excluded; repeatable; a name matching no "
                        "task is refused")
    p.add_argument("--allow-older-tools", action="store_true",
                   help="accept an older known state of the tools (its job cannot activate sbi_export on davinci)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (PlanError, CM.ManifestError) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 3


def run(args):
    # 1. the cohort manifest, with its sidecar
    cm_path = os.path.abspath(args.cohort_manifest)
    if not os.path.isfile(cm_path):
        raise PlanError("cohort manifest not found: %s" % cm_path)
    if not os.path.isfile(cm_path + ".sha256"):
        raise PlanError("%s has no .sha256 beside it; a copy without its sidecar would "
                        "load unchecked. Freeze the manifest WITH its sidecar." % cm_path)
    manifest = CM.read_manifest(cm_path)
    dt, sigma_sm, n_e, recorded = CM.sim_preprocessing_from_manifest(manifest)
    n_side = math.isqrt(n_e)
    if n_side * n_side != n_e:
        raise PlanError("electrodes_per_subset = %d is not a perfect square; the virtual "
                        "probe is an n_side x n_side grid" % n_e)

    # 2. the tools and the library
    tools_dir = os.path.abspath(args.tools)
    tool_files, tools_label = check_tools(tools_dir, args.allow_older_tools)
    library = os.path.abspath(args.library or os.path.join(tools_dir, "eap_library.npz"))
    if not os.path.isfile(library):
        raise PlanError("template library not found: %s (launch_mea_array.sh would build a "
                        "new one silently; the record must name the one used)" % library)
    library_sha = sha256_file(library)

    # 3. the raw simulations
    sim_main = os.path.abspath(args.sim_main)
    out_root = os.path.abspath(args.out_root)
    if not os.path.isdir(sim_main):
        raise PlanError("sim main not found: %s" % sim_main)
    campaign_dirs = enumerate_campaigns(sim_main, args.campaign_glob)
    tasks, excluded = enumerate_tasks(campaign_dirs, out_root)
    tasks, excluded, excluded_by_name = exclude_named(tasks, excluded, args.exclude_task)
    if not tasks:
        raise PlanError("no runnable task under %s" % ", ".join(campaign_dirs))

    # 4. one contract
    single, mixed = check_contract(tasks)
    mixed_refusing = {k: v for k, v in mixed.items() if k not in REPORTED_ONLY}
    if mixed_refusing and not args.allow_mixed_contract:
        lines = []
        for k, vals in mixed_refusing.items():
            lines.append("  %s: %s" % (k, " | ".join(
                "%s (%d tasks, e.g. %s)" % (_short(d["value"]), d["n_tasks"], d["tasks"][0])
                for d in vals)))
        raise PlanError("the tasks disagree on their label contract -- one bank needs one "
                        "contract:\n%s\nPass --allow-mixed-contract to proceed anyway." % "\n".join(lines))

    # 5. the output root
    state = existing_output(out_root)
    if state == "record":
        raise PlanError("%s already holds %s: a finished re-extraction. Choose another root."
                        % (out_root, RECORD_NAME))
    n_done = 0
    if state == "detections":
        if not args.resume:
            raise PlanError("%s already holds detections. Pass --resume to keep the complete "
                            "tasks and re-run the rest, or choose another root." % out_root)
        for t in tasks:
            if task_is_complete(t):
                t["status"] = "done"
                n_done += 1
    run_tasks = [t for t in tasks if t["status"] == "run"]
    for i, t in enumerate(run_tasks):
        t["array_index"] = i

    # 6. write
    fs = float(recorded["fs_raw"])
    extra_args = "--n_side %d --pitch %s --edge %s --fs %s" % (
        n_side, repr(PITCH_UM), repr(EDGE_UM), repr(fs))
    plan = {
        "plan_version": PLAN_VERSION,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cohort_manifest": {"path": cm_path, "digest": manifest["_digest"],
                            "extractor_commit": manifest.get("extractor_commit"),
                            "electrodes_per_subset": n_e,
                            "w_size": dt, "gaussian_window": sigma_sm, "recorded": recorded},
        "geometry": {"n_side": n_side, "n_e": n_side * n_side, "pitch_um": PITCH_UM,
                     "edge_um": EDGE_UM, "n_sub": N_SUB, "fs": fs,
                     "decisions": {"n_e": "D-015", "pitch": "D-021", "edge": "D-024",
                                   "fs": "the cohort manifest's fs_raw (recorded field)"}},
        "extra_args": extra_args,
        "sim_main": sim_main, "campaign_glob": args.campaign_glob,
        "campaigns": [os.path.basename(c) for c in campaign_dirs],
        "out_root": out_root,
        "tools": {"dir": tools_dir, "label": tools_label, "files": tool_files},
        "library": {"path": library, "sha256": library_sha},
        "contract": {"fields": single, "mixed": mixed},
        "counts": {"tasks_run": len(run_tasks), "tasks_done": n_done,
                   "tasks_excluded": len(excluded),
                   "iters_run": sum(t["n_iters"] for t in run_tasks),
                   "iters_total": sum(t["n_iters"] for t in tasks),
                   "max_iters_task": max(t["n_iters"] for t in tasks),
                   "max_topos_task": max(t["n_topos"] for t in tasks)},
        "tasks": tasks, "excluded": excluded,
        "excluded_by_name": excluded_by_name,
        "tasks_tsv": os.path.abspath(args.tasks_out),
        "resume": bool(args.resume),
    }
    with open(args.tasks_out, "w") as fh:
        for t in run_tasks:
            fh.write("%s\t%s\n" % (t["campaign_dir"], t["out_dir"]))
    with open(args.plan_out, "w") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # 7. say what was decided
    print("[plan] cohort manifest : %s  sha256 %s" % (cm_path, manifest["_digest"][:16]))
    print("[plan] electrodes      : electrodes_per_subset %d -> n_side %d (D-015); "
          "pitch %g um (D-021), edge %g um (D-024); fs %s Hz (the manifest's fs_raw)"
          % (n_e, n_side, PITCH_UM, EDGE_UM, fs))
    print("[plan] EXTRA_ARGS      : %s" % extra_args)
    print("[plan] tools           : %s" % tools_dir)
    print("[plan]                   %s" % tools_label)
    print("[plan] library         : %s  sha256 %s" % (library, library_sha[:16]))
    print("[plan] campaigns       : %d under %s matching %s"
          % (len(campaign_dirs), sim_main, args.campaign_glob))
    by_camp = {}
    for t in tasks:
        by_camp.setdefault(t["campaign"], [0, 0, 0])
        by_camp[t["campaign"]][0 if t["status"] == "run" else 1] += 1
        by_camp[t["campaign"]][2] += t["n_iters"]
    for e in excluded:
        by_camp.setdefault(e["campaign"], [0, 0, 0])
    for camp in sorted(by_camp):
        r, d, n = by_camp[camp]
        ex = sum(1 for e in excluded if e["campaign"] == camp)
        print("[plan]   %-28s run %3d  done %3d  excluded %3d  iterations %7d"
              % (camp, r, d, ex, n))
    print("[plan] tasks to run    : %d (array 0-%d), %d iterations; largest task %d "
          "iterations, %d topologies"
          % (len(run_tasks), max(len(run_tasks) - 1, 0), plan["counts"]["iters_run"],
             plan["counts"]["max_iters_task"], plan["counts"]["max_topos_task"]))
    if n_done:
        print("[plan] tasks complete  : %d kept as done (--resume)" % n_done)
    if excluded:
        print("[plan] EXCLUDED %d task(s), named, not run:" % len(excluded))
        for e in excluded:
            print("[plan]   %s/%s: %s" % (e["campaign"], e["sweep"], e["reason"]))
    if mixed:
        print("[plan] contract fields with more than one value%s:"
              % ("" if mixed_refusing else " (reported only)"))
        for k, vals in mixed.items():
            print("[plan]   %s: %s" % (k, " | ".join(
                "%s x%d" % (_short(d["value"], 40), d["n_tasks"]) for d in vals)))
    print("[plan] contract        : %s" % ", ".join(
        "%s=%s" % (k, _short(single[k], 24)) for k in JOB_ARGS_FIELDS if k in single))
    print("[plan] wrote %s and %s" % (os.path.abspath(args.plan_out), os.path.abspath(args.tasks_out)))
    if not run_tasks:
        print("[plan] nothing to run: every task is complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
