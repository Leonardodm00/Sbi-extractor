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
  and the electrode side are the cohort's device: by default DUP15HD's
  (60 um, D-021; 25 um, D-024), or --pitch-um / --edge-um with the decisions
  that fix them (--decision-pitch / --decision-edge; the Giulia profile
  passes 200 um, D-042, and 26.59 um, D-049); all passed explicitly so the
  record carries them. The sampling rate is the manifest's fs_raw (the real
  device's), also passed. How the tools seed each iteration's noise is read
  from their known state (TOOLS_NOISE_SCHEME) and recorded, so the gate can
  check every file against it.

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

WHAT IT DROPS AS A REPLAY (decision D-061), unless --keep-replays
  a task that replays another task's seed and simulations is not run. Two
  runnable tasks are compared when they recorded the same seed values in
  job_args.json (every key whose name contains "seed"; the CLI seed_master
  is ignored where _resolved_seed_master is recorded) AND carry the same
  label contract. Of such a group, tasks are taken in the keeper order --
  most iterations first, then the lower campaign version, then the lower
  task index (D-063) -- and each is compared with the tasks kept before it:
  it is a replay of one when the two share iteration files AND theta (and
  params) are byte-identical at up to REPLAY_SAMPLE shared files spread from
  the first to the last (D-063). One task per seed (D-064): a replay is
  dropped whether its iteration files are the same as, a subset of, or only
  overlap the kept task's; the iteration files only it holds are dropped
  with it, counted ("files_lost") and printed. A replay is listed under
  "excluded" with the reason "replay (D-061) of CAMPAIGN/SWEEP: ...", in the
  plan and in the record. Kept, and reported: a same-seed task with
  different theta, one with no shared iteration file, one whose files carry
  neither theta nor params. REFUSED: an unreadable iteration file in a
  compared pair, and a replay whose output folder already holds detections
  (move it out of the output root, nothing is deleted: the record must not
  sit beside detections it does not name). --keep-replays computes and
  reports the same, and drops nothing (for comparison only).

OUTPUT
  --tasks-out  TSV, one line per task to RUN: campaign_dir <TAB> out_dir,
               the format submit_mea_array.sh reads by line number.
  --plan-out   plan.json: everything above plus per-task topology and
               iteration counts, the tools' fingerprints and label, the
               library's sha256, the contract, the EXTRA_ARGS string, and
               "replays": the seed keys seen, every same-seed group with each
               member's role and relation, the tasks with no seed recorded.

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
import zipfile
import zlib

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
KNOWN_TOOLS["ANN main G2 (2026-10-06, one worker per core D-071, noise seeded per simulation D-072; tree 028d786a)"] = {
    "process_campaign.py": "e9f2e0e53ea5a9804fa054416448cc80efa6b113960caecfe463b21cafd0eca0",
    "mea_probe.py": "537641fe1bcfd37bff28c621f3bbed8b90bb7974f03b27f9a8e74eb8e218d658",
    "mea_detection.py": "9510c2aea223f3abe6edae8c35ffd0704bd5860641f5b28e75db2b04b6312a32",
    "mea_synthesis.py": "6834b7c59b0dbbef4786d64113e22141427c9e5d342c97b441f026f7a9a337bd",
    "mea_plots.py": "48b095056e85c90af4e29c56e90ccc13f1b2e5c40f674ac8b6ebc61f8d2c8c39",
    "eap_template_library.py": "5d32d0e8d1bee7002599223242b49031c3f9efbc345cb38fa7db9568e4232029",
    "submit_mea_array.sh": "12b0cdf7c3ce1644693be915584bee7c877a909b40eb12a75e4a794f0b1f6189",
}
# How each known state seeds an iteration's additive noise (process_campaign.py,
# noise_entropy): 'topo_iter' before 2026-10-06 (the C8 record's state), 'sim'
# from D-072. A state not listed here is treated as 'topo_iter'.
TOOLS_NOISE_SCHEME = {
    "ANN main 37bf9f8 (2026-09-30, launcher env fix; tree 0969d9f3)": "topo_iter",
    "ANN main C8 (2026-10-01, job finds the cluster's conda, writes mea_env.json; tree 6e01860c)": "topo_iter",
    "ANN main G2 (2026-10-06, one worker per core D-071, noise seeded per simulation D-072; tree 028d786a)": "sim",
}
# Why each older state is older, for the refusal.
OLDER_TOOLS_REASON = {
    "ANN main 37bf9f8 (2026-09-30, launcher env fix; tree 0969d9f3)":
        "its submit_mea_array.sh cannot activate sbi_export on davinci and writes no mea_env.json",
    "ANN main C8 (2026-10-01, job finds the cluster's conda, writes mea_env.json; tree 6e01860c)":
        "its job runs one worker per task whatever the node gives (D-071) and seeds the noise from the "
        "topology and iteration indices alone (D-072)",
}
# The state this plan is built for: its job activates sbi_export on davinci
# (D-023), writes the mea_env.json the gate reads, runs one worker per core
# (D-071) and seeds each simulation's noise from its own seed_run (D-072).
# Older known states are refused unless --allow-older-tools.
REQUIRED_TOOLS_LABEL = "ANN main G2 (2026-10-06, one worker per core D-071, noise seeded per simulation D-072; tree 028d786a)"

SWEEP_RE = re.compile(r"^sweep_[^_]+_task\d+$")
ITER_RE = re.compile(r"^iter_\d+\.npz$")

# The fields of a task's label contract, read from job_args.json and
# manifest.json. One bank needs one value of each across the campaign set.
JOB_ARGS_FIELDS = ("simtime", "conn_prob_lo", "conn_prob_hi",
                   "p0_lo", "p0_hi", "d0_lo", "d0_hi", "beta_lo", "beta_hi")
MANIFEST_FIELDS = ("active_indices", "sweep_group", "param_names",
                   "log_params", "log_transform", "conn_rule", "param_bounds")
REPORTED_ONLY = ("manifest_version",)

# The replay gate (D-061). theta (inference coordinates) and params (natural
# units) are what a replay shares with the task it replays; the spike trains
# and seed_run are compared too, for the report only.
REPLAY_SAMPLE = 10
IDENTITY_MEMBERS = ("theta.npy", "params.npy")
SPIKE_MEMBERS = ("spk_N_t.npy", "spk_N_i.npy", "spk_A_t.npy", "spk_A_i.npy", "seed_run.npy")
REPLAY_REASON = "replay (D-061) of"


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
            "the tools in %s are %s, older than this plan needs (%s): %s. Copy the "
            "newer files in, or pass --allow-older-tools."
            % (tools_dir, label, REQUIRED_TOOLS_LABEL,
               OLDER_TOOLS_REASON.get(label, "an older state")))
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


def contract_signature(c):
    return hashlib.sha256(json.dumps({k: c[k] for k in JOB_ARGS_FIELDS + MANIFEST_FIELDS},
                                     sort_keys=True).encode()).hexdigest()[:16]


def check_contract(tasks):
    """One value per field across the tasks, else the mixed fields."""
    values = {}
    for t in tasks:
        c = task_contract(t["campaign_dir"])
        t["contract_sig"] = contract_signature(c)
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
# replays (D-061)
# --------------------------------------------------------------------------- #
def task_seed(sweep_dir):
    """The seed values a task recorded in its job_args.json: every key whose
    name contains "seed" (the ANN sweep writes _resolved_seed_master and the
    offsets seed_device, seed_neuron, seed_synapse, seed_astro). The CLI
    seed_master is dropped where _resolved_seed_master is recorded: it is None
    when the master seed was resolved from the job id, so it would split two
    runs of one seed."""
    j = _load_json(os.path.join(sweep_dir, "job_args.json"))
    s = {k: j[k] for k in j if "seed" in k.lower()}
    if "_resolved_seed_master" in s:
        s.pop("seed_master", None)
    return s


def _name(t):
    return "%s/%s" % (t["campaign"], t["sweep"])


def _last_int(pattern, text):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 10 ** 9


def keeper_order(t):
    """Most iterations first; then the lower campaign version; then the lower
    task index; then the names, so the order is total."""
    return (-int(t["n_iters"]), _last_int(r"v(\d+)$", t["campaign"]), t["campaign"],
            _last_int(r"task(\d+)$", t["sweep"]), t["sweep"])


def iter_files(t):
    """{topo_dir: set of iter_*.npz names} of one task, read from disk."""
    out = {}
    for td in sorted(t["iters_per_topo"]):
        d = os.path.join(t["campaign_dir"], td)
        out[td] = set(n for n in os.listdir(d) if ITER_RE.match(n))
    return out


def npz_members(path, names):
    """The raw bytes of the named .npy members of an .npz (stdlib zipfile);
    a member the file does not hold is absent from the result. Byte-equal
    members are equal arrays of one dtype and shape."""
    try:
        with zipfile.ZipFile(path) as z:
            have = set(z.namelist())
            return {n: z.read(n) for n in names if n in have}
    except (OSError, zipfile.BadZipFile, zlib.error, EOFError, KeyError, ValueError) as exc:
        raise PlanError("cannot read %s while comparing two same-seed tasks (D-061): %s: %s. "
                        "A file being written? Leave its task out with --exclude-task, or "
                        "repair the file." % (path, type(exc).__name__, exc))


def _sample(seq, k):
    n = len(seq)
    if n <= k:
        return list(seq)
    idx = sorted(set(int(round(i * (n - 1) / float(k - 1))) for i in range(k)))
    return [seq[i] for i in idx]


def replay_relation(keep, other, files, sample_k):
    """How `other` relates to `keep`, a task of the same seed and contract that
    comes before it in the keeper order. "replay" is True only when other's
    iteration files are the same as or a subset of keep's and theta (and
    params) are byte-identical at every sampled shared file."""
    fk, fo = files[_name(keep)], files[_name(other)]
    nk, no = sum(len(v) for v in fk.values()), sum(len(v) for v in fo.values())
    common = sorted((td, n) for td in fo if td in fk for n in fo[td] & fk[td])
    rel = {"replay": False, "files": no, "files_of_kept": nk, "shared": len(common),
           "checked": 0, "spikes_identical": None, "files_lost": 0}
    if not common:
        rel["detail"] = "same seed, no shared iteration file (%d vs %d files): kept" % (no, nk)
        return rel
    spikes_same = True
    for td, n in _sample(common, sample_k):
        a = npz_members(os.path.join(keep["campaign_dir"], td, n), IDENTITY_MEMBERS + SPIKE_MEMBERS)
        b = npz_members(os.path.join(other["campaign_dir"], td, n), IDENTITY_MEMBERS + SPIKE_MEMBERS)
        ids = [m for m in IDENTITY_MEMBERS if m in a and m in b]
        if not ids:
            rel["detail"] = ("same seed, but %s/%s holds neither theta nor params in both tasks: "
                             "undetermined, kept" % (td, n))
            rel["undetermined"] = True
            return rel
        for m in ids:
            if a[m] != b[m]:
                rel["checked"] += 1
                rel["detail"] = ("same seed, DIFFERENT %s at %s/%s: other simulations, kept"
                                 % (m[:-4], td, n))
                return rel
        rel["checked"] += 1
        for m in SPIKE_MEMBERS:
            if (m in a) != (m in b) or (m in a and a[m] != b[m]):
                spikes_same = False
    rel["spikes_identical"] = spikes_same
    subset = all(td in fk and fo[td] <= fk[td] for td in fo)
    same = subset and all(td in fo and fk[td] <= fo[td] for td in fk)
    head = "same theta at %d of %d shared iteration files" % (rel["checked"], len(common))
    spk = "spikes identical there" if spikes_same else "spikes NOT identical there"
    if same:
        rel["replay"] = True
        rel["detail"] = "%s; the same %d iteration files; %s" % (head, no, spk)
    elif subset:
        rel["replay"] = True
        rel["detail"] = ("%s; its %d iteration files are a subset of the kept task's %d; %s"
                         % (head, no, nk, spk))
    else:
        # one task per seed (D-064): an overlapping replay goes too, with the
        # iteration files only it holds
        lost = sum(len(fo[td] - fk.get(td, set())) for td in fo)
        rel["replay"] = True
        rel["files_lost"] = lost
        rel["detail"] = ("%s; it holds %d iteration files the kept task lacks (%d vs %d), dropped "
                         "with it (one task per seed, D-064); %s" % (head, lost, no, nk, spk))
    return rel


def find_replays(tasks, sample_k=REPLAY_SAMPLE):
    """Group the runnable tasks by (seed values, contract signature) and, in
    each group, mark which tasks replay a task kept before them.

    Returns (report, drops): report is the plan's "replays" block without its
    "enabled" and "dropped" fields; drops is a list of (task, kept_task,
    relation)."""
    by_key, by_seed, no_seed, keys_seen = {}, {}, [], {}
    for t in tasks:
        s = task_seed(t["campaign_dir"])
        for k in s:
            keys_seen[k] = keys_seen.get(k, 0) + 1
        if not s:
            no_seed.append(_name(t))
            continue
        sk = json.dumps(s, sort_keys=True)
        sig = contract_signature(task_contract(t["campaign_dir"]))
        by_key.setdefault((sk, sig), []).append(t)
        by_seed.setdefault(sk, set()).add(sig)
    groups, drops = [], []
    for (sk, sig) in sorted(by_key):
        members = sorted(by_key[(sk, sig)], key=keeper_order)
        if len(members) < 2:
            continue
        files = {_name(t): iter_files(t) for t in members}
        kept = [members[0]]
        entries = [{"task": _name(members[0]), "n_iters": members[0]["n_iters"], "role": "kept",
                    "relation": "first in the keeper order"}]
        for o in members[1:]:
            rels = []
            for k in kept:
                r = replay_relation(k, o, files, sample_k)
                rels.append((k, r))
                if r["replay"]:
                    break
            k, r = rels[-1]
            if r["replay"]:
                drops.append((o, k, r))
                entries.append({"task": _name(o), "n_iters": o["n_iters"], "role": "replay",
                                "of": _name(k), "relation": r["detail"],
                                "checked": r["checked"], "shared": r["shared"],
                                "files_lost": r["files_lost"],
                                "spikes_identical": r["spikes_identical"]})
            else:
                kept.append(o)
                entries.append({"task": _name(o), "n_iters": o["n_iters"], "role": "kept",
                                "relation": "; ".join("vs %s: %s" % (_name(kk), rr["detail"])
                                                      for kk, rr in rels)})
        groups.append({"seed": json.loads(sk), "contract_sig": sig, "members": entries})
    across = [{"seed": json.loads(sk), "contracts": sorted(sigs)}
              for sk, sigs in sorted(by_seed.items()) if len(sigs) > 1]
    report = {"sample_files_per_pair": sample_k, "seed_keys": keys_seen,
              "tasks_compared": len(tasks) - len(no_seed),
              "distinct_seeds": len(by_seed),
              "tasks_without_seed": no_seed,
              "groups": groups,
              "seeds_shared_across_contracts": across}
    return report, drops


def drop_replays(tasks, excluded, drops):
    """Move each replay from `tasks` to `excluded`, with its reason. Refuses
    a replay whose output folder already holds detections."""
    names = set(_name(o) for o, _, _ in drops)
    with_output = []
    for o, k, r in drops:
        od = o["out_dir"]
        if os.path.isfile(os.path.join(od, "mea_manifest.json")) or \
           os.path.isfile(os.path.join(od, "mea_env.json")):
            with_output.append((o, od))
    if with_output:
        o0, od0 = with_output[0]
        root0 = os.path.dirname(os.path.dirname(od0))
        dest0 = os.path.join(root0 + "_replays_moved", o0["campaign"])
        raise PlanError("these replays (D-061) already have detections in the output root, which "
                        "the record would not name: %s. Move each folder OUT of the output root "
                        "(nothing is deleted), e.g. mkdir -p '%s' && mv '%s' '%s/', and plan again."
                        % ("; ".join("%s (%s)" % (_name(o), od) for o, od in with_output),
                           dest0, od0, dest0))
    keep = [t for t in tasks if _name(t) not in names]
    for o, k, r in drops:
        rec = {key: o[key] for key in ("campaign", "sweep", "campaign_dir", "out_dir",
                                       "n_topos", "n_iters", "iters_per_topo")}
        seed = task_seed(o["campaign_dir"])
        rec["reason"] = "%s %s: seed %s; %s" % (
            REPLAY_REASON, _name(k), seed.get("_resolved_seed_master", _short(seed, 60)), r["detail"])
        rec["replay_of"] = _name(k)
        rec["files_lost"] = r["files_lost"]
        excluded.append(rec)
    excluded.sort(key=lambda e: (e["campaign"], e["sweep"]))
    return keep, excluded, sorted(names)


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
    p.add_argument("--keep-replays", action="store_true",
                   help="compare same-seed tasks and report the replays (D-061), but drop none "
                        "(for comparison only)")
    p.add_argument("--allow-older-tools", action="store_true",
                   help="accept an older known state of the tools (see OLDER_TOOLS_REASON)")
    p.add_argument("--pitch-um", type=float, default=PITCH_UM,
                   help="the virtual probe's pitch, um (default %(default)s, DUP15HD's device, D-021)")
    p.add_argument("--edge-um", type=float, default=EDGE_UM,
                   help="the virtual electrode's side, um (default %(default)s, D-024)")
    p.add_argument("--decision-pitch", default="D-021",
                   help="the decision that fixes --pitch-um, for the record (default %(default)s)")
    p.add_argument("--decision-edge", default="D-024",
                   help="the decision that fixes --edge-um, for the record (default %(default)s)")
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
    # the folders beside the campaigns that the glob does not take, named for the
    # record (D-050: the Giulia q/ units are outside it and not used)
    taken = set(os.path.basename(c) for c in campaign_dirs)
    outside_glob = sorted(e for e in os.listdir(sim_main)
                          if os.path.isdir(os.path.join(sim_main, e)) and e not in taken)
    tasks, excluded = enumerate_tasks(campaign_dirs, out_root)
    tasks, excluded, excluded_by_name = exclude_named(tasks, excluded, args.exclude_task)
    if not tasks:
        raise PlanError("no runnable task under %s" % ", ".join(campaign_dirs))

    # 3b. replays (D-061): same seed, same contract, same theta -> not run
    replays, drops = find_replays(tasks)
    replays["enabled"] = not args.keep_replays
    replays["dropped"] = []
    if drops and not args.keep_replays:
        tasks, excluded, replays["dropped"] = drop_replays(tasks, excluded, drops)

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
    pitch, edge = float(args.pitch_um), float(args.edge_um)
    if not (pitch > 0 and edge > 0 and math.isfinite(pitch) and math.isfinite(edge)):
        raise PlanError("--pitch-um %r / --edge-um %r: both must be positive and finite" % (pitch, edge))
    noise_scheme = TOOLS_NOISE_SCHEME.get(tools_label, "topo_iter")
    extra_args = "--n_side %d --pitch %s --edge %s --fs %s" % (
        n_side, repr(pitch), repr(edge), repr(fs))
    plan = {
        "plan_version": PLAN_VERSION,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cohort_manifest": {"path": cm_path, "digest": manifest["_digest"],
                            "extractor_commit": manifest.get("extractor_commit"),
                            "electrodes_per_subset": n_e,
                            "w_size": dt, "gaussian_window": sigma_sm, "recorded": recorded},
        "geometry": {"n_side": n_side, "n_e": n_side * n_side, "pitch_um": pitch,
                     "edge_um": edge, "n_sub": N_SUB, "fs": fs,
                     "noise_seed_scheme": noise_scheme,
                     "decisions": {"n_e": "D-015", "pitch": args.decision_pitch,
                                   "edge": args.decision_edge,
                                   "noise_seed_scheme": "the tools' known state (TOOLS_NOISE_SCHEME; D-072)",
                                   "fs": "the cohort manifest's fs_raw (recorded field)"}},
        "extra_args": extra_args,
        "sim_main": sim_main, "campaign_glob": args.campaign_glob,
        "campaigns": [os.path.basename(c) for c in campaign_dirs],
        "outside_glob": outside_glob,
        "out_root": out_root,
        "tools": {"dir": tools_dir, "label": tools_label, "files": tool_files},
        "library": {"path": library, "sha256": library_sha},
        "contract": {"fields": single, "mixed": mixed},
        "counts": {"tasks_run": len(run_tasks), "tasks_done": n_done,
                   "tasks_excluded": len(excluded),
                   "replays_dropped": len(replays["dropped"]),
                   "replays_files_lost": sum(e.get("files_lost", 0) for e in excluded
                                             if e.get("replay_of")),
                   "iters_run": sum(t["n_iters"] for t in run_tasks),
                   "iters_total": sum(t["n_iters"] for t in tasks),
                   "max_iters_task": max(t["n_iters"] for t in tasks),
                   "max_topos_task": max(t["n_topos"] for t in tasks)},
        "tasks": tasks, "excluded": excluded,
        "excluded_by_name": excluded_by_name,
        "replays": replays,
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
          "pitch %g um (%s), edge %g um (%s); fs %s Hz (the manifest's fs_raw)"
          % (n_e, n_side, pitch, args.decision_pitch, edge, args.decision_edge, fs))
    print("[plan] noise seeds     : %s (the tools' state)" % noise_scheme)
    print("[plan] EXTRA_ARGS      : %s" % extra_args)
    print("[plan] tools           : %s" % tools_dir)
    print("[plan]                   %s" % tools_label)
    print("[plan] library         : %s  sha256 %s" % (library, library_sha[:16]))
    print("[plan] campaigns       : %d under %s matching %s"
          % (len(campaign_dirs), sim_main, args.campaign_glob))
    print("[plan] outside the glob: %d folder(s), not read: %s"
          % (len(outside_glob), ", ".join(outside_glob[:12]) + (" ..." if len(outside_glob) > 12 else "")
             if outside_glob else "none"))
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
    print("[plan] replays (D-061) : seed keys %s; %d tasks compared, %d distinct seeds, "
          "%d same-seed group(s), %d task(s) with no seed recorded"
          % (", ".join("%s x%d" % kv for kv in sorted(replays["seed_keys"].items())) or "none",
             replays["tasks_compared"], replays["distinct_seeds"], len(replays["groups"]),
             len(replays["tasks_without_seed"])))
    for g in replays["groups"]:
        print("[plan]   seed %s (contract %s):"
              % (g["seed"].get("_resolved_seed_master", _short(g["seed"], 50)), g["contract_sig"]))
        for m in g["members"]:
            if m["role"] == "replay":
                print("[plan]     %s %s (%d iterations): replay of %s -- %s"
                      % ("DROPPED" if replays["enabled"] else "would drop", m["task"],
                         m["n_iters"], m["of"], m["relation"]))
            else:
                print("[plan]     kept    %s (%d iterations): %s" % (m["task"], m["n_iters"], m["relation"]))
    for a in replays["seeds_shared_across_contracts"]:
        print("[plan]   seed %s is shared across %d label contracts: not compared, all kept"
              % (a["seed"].get("_resolved_seed_master", _short(a["seed"], 50)), len(a["contracts"])))
    n_rep = sum(1 for g in replays["groups"] for m in g["members"] if m["role"] == "replay")
    n_lost = sum(m.get("files_lost", 0) for g in replays["groups"] for m in g["members"]
                 if m["role"] == "replay")
    if replays["enabled"]:
        print("[plan] replays dropped : %d (named under EXCLUDED); iteration files only a dropped "
              "replay held: %d" % (n_rep, n_lost))
    else:
        print("[plan] replays dropped : 0 -- --keep-replays: %d would be dropped, kept on request "
              "(%d iteration files only they hold)" % (n_rep, n_lost))
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
