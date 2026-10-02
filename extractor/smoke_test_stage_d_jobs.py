#!/usr/bin/env python3
"""Smoke test -- the Stage D JOB LAYER, end to end, on a synthetic SpyCode-like
cohort (2026-10-01, the Giulia recordings).

What no other suite covers: launch_stage_d.sh run for REAL (not DRYRUN), the
two PBS scripts it submits (run_extractor_array_mea.pbs per well, then
run_cohort_manifest.pbs), the -v variables that carry a tagged cohort's
manifest and flags into them, the figures the array job draws, and the
aggregation's PASS / REFUSED lines. Nothing is submitted: a fake `qsub` first
on PATH runs every job inline (array tasks in order, each from a spool copy
of the script with PBS_ARRAY_INDEX / PBS_O_WORKDIR set, as PBS does), and a
stub `conda` makes the scripts' activation block a no-op. Everything runs on
a COPY of this repository in a temporary directory, so the real
extractor/out/ is never written.

  J0  the single-well command of the run sheet: run_channel_subset_extraction.py
      with the tracked extraction_flags_giulia.sh, figures ON -> 9 archives,
      the fragment (version 4, sparse_peaks), both PNGs
  J1  launch_stage_d.sh (COHORT_TAG=giulia, the tracked flags file) exits 0
      and reports both job ids; its aggregation line no longer claims the
      job is "held until every array task exits 0" (D-006), and its last
      advice is the whole log (cat), not a grep (2026-10-02)
  J2  the fake qsub saw: -J 0-4 (5 kept wells), -N chsub_mea_array_giulia,
      -v ...CHSUB_MANIFEST=out/extraction_manifest_giulia.tsv,
      CHSUB_FLAGS=extraction_flags_giulia.sh; the aggregation with
      -W depend=afterok:<array id> and the same two variables
  J3  every array task exited 0 and logged the Giulia flags and "done ->"
  J4  every kept well holds 9 trace_subregion_*.npz, traces.npz,
      traces_meta.json (sparse_peaks), subregion_map.png, subregion_ifrs.png;
      the excluded well has no output
  J5  out/cohort_manifest_giulia.log holds "wrote <root>/cohort_manifest.json
      sha256 ..." and "cohort_manifest_exit=0"
  J6  the manifest: 5 wells x 9 = 45 units, the three classes, source_format,
      excluded_wells, T_rec_range [600, 1200] s
  J7  NEGATIVE: the same cohort with exclude_wells empty -> the too-few-active
      well's task exits non-zero (InsufficientElectrodesError) and the
      aggregation REFUSES naming it; no manifest is written. Since 2026-10-02
      the refusal also names the listing it read
      (out/extraction_manifest_giulia.tsv) and the well's array index, says
      the extraction "did not complete" (not "version 1"), and the log ends
      with cohort_manifest_exit=1
  J8  NEGATIVE then POSITIVE (2026-10-02, the Giulia re-run): launching again
      into J7's refused root, which the config declares, is refused (exit 3,
      "holds no cohort_manifest.json"), nothing submitted; once that root is
      moved aside, the same launch is a first extraction again (DRYRUN)

Run (Sbi-extractor/extractor, the sbi_export env; ~1-2 min):
    python3 smoke_test_stage_d_jobs.py; echo "exit=$?"

Needs the DSN tree (../env.sh -> SBI_HPC_DIR) for the config and the IFR.
Exit 0 = every check passed. Pure ASCII, LF only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
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

import dsn_tree                                                   # noqa: E402

RESULTS = []
FS = 10000.0
GIULIA_CONFIG_REL = os.path.join("dsn", "hpc", "Config", "config_giulia_cohort.davinci.json")
MCS_CODES = [10 * r + c for r in range(1, 9) for c in range(1, 9)
             if (r, c) not in ((1, 1), (1, 8), (8, 1), (8, 8))]
TOO_FEW = "ptrain_39485_DIV35_50N50A_nbasal_0001"


def ok(label, cond, detail=""):
    RESULTS.append(bool(cond))
    print("[%s] %-78s %s" % ("PASS" if cond else "FAIL", label, detail))


# --------------------------------------------------------------------------- #
# the fake scheduler and the stub conda
# --------------------------------------------------------------------------- #
FAKE_QSUB = r'''#!/usr/bin/env python3
# fake qsub for smoke_test_stage_d_jobs.py: runs the job inline, records argv.
import json, os, re, shutil, subprocess, sys
state = os.environ["FAKEPBS_STATE"]
argv = sys.argv[1:]
with open(os.path.join(state, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps({"argv": argv, "cwd": os.getcwd()}) + "\n")
opts, script, i = {}, None, 0
while i < len(argv):
    a = argv[i]
    if a in ("-J", "-N", "-o", "-v", "-W", "-l", "-k", "-j", "-e", "-q"):
        opts.setdefault(a, []).append(argv[i + 1]); i += 2
    else:
        script = a; i += 1
if script is None or not os.path.isfile(script):
    sys.stderr.write("qsub: script not found: %r\n" % (script,)); sys.exit(1)
text = open(script).read()
def directive(flag):
    m = re.findall(r"^#PBS\s+%s\s+(\S+)" % re.escape(flag), text, flags=re.M)
    return m[-1] if m else None
name = (opts.get("-N") or [directive("-N") or os.path.basename(script)])[-1]
out = (opts.get("-o") or [directive("-o") or (name + ".o")])[-1]
env = dict(os.environ)
for spec in opts.get("-v", []):
    for kv in spec.split(","):
        k, _, v = kv.partition("=")
        env[k] = v
counter = os.path.join(state, "counter")
n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1000
open(counter, "w").write(str(n))
spool = os.path.join(state, "spool"); os.makedirs(spool, exist_ok=True)
copy = os.path.join(spool, "%d_%s" % (n, os.path.basename(script)))
shutil.copy(script, copy)
home = os.path.join(state, "home"); os.makedirs(home, exist_ok=True)
env["PBS_O_WORKDIR"] = os.getcwd(); env["HOME"] = home
def run_one(idx):
    e = dict(env)
    path = out
    if idx is not None:
        e["PBS_ARRAY_INDEX"] = str(idx)
        path = path.replace("^array_index^", str(idx))
    e["PBS_JOBID"] = "%d%s.fake" % (n, "[%d]" % idx if idx is not None else "")
    if not os.path.isabs(path):
        path = os.path.join(env["PBS_O_WORKDIR"], path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as log:
        rc = subprocess.call(["bash", copy], cwd=home, env=e, stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(state, "exits.jsonl"), "a") as fh:
        fh.write(json.dumps({"job": n, "name": name, "index": idx, "rc": rc, "log": path}) + "\n")
if "-J" in opts:
    a, b = opts["-J"][-1].split("-")
    for idx in range(int(a), int(b) + 1):
        run_one(idx)
    print("%d[].fake" % n)
else:
    run_one(None)
    print("%d.fake" % n)
'''

STUB_CONDA_BIN = r'''#!/bin/bash
# stub conda: "conda info --base" names the stub base; nothing else is needed
# before the scripts source <base>/etc/profile.d/conda.sh
case "$1" in
  info) echo "$FAKEPBS_CONDA_BASE" ;;
  *)    exit 0 ;;
esac
'''

STUB_CONDA_SH = r'''# stub conda.sh: "conda activate X" succeeds and sets CONDA_DEFAULT_ENV
conda() {
  case "$1" in
    activate) export CONDA_DEFAULT_ENV="$2"; return 0 ;;
    info)     echo "$FAKEPBS_CONDA_BASE"; return 0 ;;
    env)      echo "base  $FAKEPBS_CONDA_BASE"; return 0 ;;
    *)        return 0 ;;
  esac
}
'''


def make_fake_scheduler(td):
    fakebin = os.path.join(td, "fakebin")
    state = os.path.join(td, "pbs_state")
    base = os.path.join(td, "conda_base")
    for d in (fakebin, state, os.path.join(base, "etc", "profile.d")):
        os.makedirs(d, exist_ok=True)
    for name, text in (("qsub", FAKE_QSUB), ("conda", STUB_CONDA_BIN)):
        p = os.path.join(fakebin, name)
        open(p, "w").write(text)
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    open(os.path.join(base, "etc", "profile.d", "conda.sh"), "w").write(STUB_CONDA_SH)
    # the job runs "python3": keep the interpreter of this test first on PATH
    py = os.path.join(fakebin, "python3")
    os.symlink(sys.executable, py)
    return fakebin, state, base


def copy_repo(td, sbi_hpc):
    """A copy of this repository's runnable files; artifacts/sbi_hpc -> the DSN tree."""
    dst = os.path.join(td, "repo")
    shutil.copytree(_ROOT, dst, ignore=shutil.ignore_patterns(
        ".git", "artifacts", "out", "__pycache__", "*.npz", "*.png", "*.log"))
    os.makedirs(os.path.join(dst, "artifacts"), exist_ok=True)
    os.symlink(sbi_hpc, os.path.join(dst, "artifacts", "sbi_hpc"))
    return dst


# --------------------------------------------------------------------------- #
# the synthetic cohort (SpyCode-like files)
# --------------------------------------------------------------------------- #
def write_spycode_file(path, n_samples, n_spikes, rng):
    idx = np.sort(rng.choice(n_samples, size=n_spikes, replace=False)) if n_spikes else \
        np.zeros(0, dtype=np.int64)
    amps = -rng.uniform(20.0, 80.0, size=idx.size)
    m = sparse.csc_matrix((amps, (idx, np.zeros(idx.size, dtype=np.int64))),
                          shape=(n_samples, 1))
    sio.savemat(path, {"peak_train": m, "artifact": np.zeros((0, 1))},
                format="5", do_compression=True)


def make_well(folder, well_id, cond, n_files, n_active, n_samples, rng):
    """n_active electrodes at >= 0.2 Hz, two at ~0.03 Hz, the rest silent."""
    os.makedirs(folder, exist_ok=True)
    t = n_samples / FS
    for j, code in enumerate(MCS_CODES[:n_files]):
        if j < n_active:
            k = int(t * (0.2 + 0.05 * j))
        elif j < n_active + 2:
            k = int(t * 0.03)
        else:
            k = 0
        write_spycode_file(os.path.join(
            folder, "ptrain_%d_DIV35_%s_nbasal_0001_%03d.mat" % (well_id, cond, code)),
            n_samples, k, rng)


def build_cohort(td, rng):
    root = os.path.join(td, "Bio_Data")
    plan = [("Batch1/100N0A_wo", 38886, "100N0A", 54, 14, 6_000_000),
            ("Batch1/100N0A_wo", 38932, "100N0A", 56, 12, 12_000_000),
            ("Batch2/100N0A_wo", 34346, "100N0A", 60, 20, 9_000_000),
            ("Batch3/AraC/50N50A", 38931, "50N50A", 26, 11, 9_000_000),
            ("Batch3/AraC/50N50A", 39485, "50N50A", 17, 5, 9_000_000),     # too few
            ("Batch3/AraC/70N30A", 38927, "70N30A", 19, 10, 9_000_000)]
    for rel, wid, cond, nf, na, ns in plan:
        make_well(os.path.join(root, rel, "ptrain_%d_DIV35_%s_nbasal_0001" % (wid, cond)),
                  wid, cond, nf, na, ns, rng)
    return root


def write_config(td, sbi_hpc, root, extract_root, exclude, name):
    """The Giulia config of record, with class_roots / extract_root / exclude_wells
    pointed at the synthetic cohort: every extraction field (hence the flags) is
    the tracked one."""
    src = os.path.join(sbi_hpc, GIULIA_CONFIG_REL)
    cfg = json.load(open(src))
    cfg["cohort"]["class_roots"] = {
        "0": [os.path.join(root, "Batch1", "100N0A_wo"), os.path.join(root, "Batch2", "100N0A_wo")],
        "1": [os.path.join(root, "Batch3", "AraC", "50N50A")],
        "2": [os.path.join(root, "Batch3", "AraC", "70N30A")]}
    cfg["cohort"]["extract_root"] = extract_root
    cfg["cohort"]["exclude_wells"] = list(exclude)
    p = os.path.join(td, name)
    json.dump(cfg, open(p, "w", encoding="ascii"), indent=1)
    return p


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(ln) for ln in open(path) if ln.strip()]


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def j0_single_well(repo, root, td):
    ext = os.path.join(repo, "extractor")
    well = os.path.join(root, "Batch3", "AraC", "50N50A", "ptrain_38931_DIV35_50N50A_nbasal_0001")
    out = os.path.join(td, "single_well_out")
    cmd = ('source extraction_flags_giulia.sh && python3 run_channel_subset_extraction.py "%s" '
           '--out-dir "%s" --mode per_region_single $EXTRA_FLAGS' % (well, out))
    r = subprocess.run(["bash", "-c", cmd], cwd=ext, capture_output=True, text=True)
    files = sorted(os.listdir(out)) if os.path.isdir(out) else []
    meta = json.load(open(os.path.join(out, "traces_meta.json"))) if "traces_meta.json" in files else {}
    ok("J0 single well with the tracked flags, figures on: 9 archives, version-4 fragment, 2 PNGs",
       r.returncode == 0 and sum(f.startswith("trace_subregion_") for f in files) == 9
       and "subregion_map.png" in files and "subregion_ifrs.png" in files
       and meta.get("ptrain_format") == "sparse_peaks"
       and meta.get("extractor_version") == "run_channel_subset_extraction/4"
       and abs(float(meta.get("T_rec", 0)) - 900.0) < 1e-9,
       (r.stdout + r.stderr).strip().splitlines()[-1][:70] if (r.stdout + r.stderr).strip() else "")


def run_launch(repo, cfg, extract_root, fakebin, state, base, sbi_hpc):
    env = dict(os.environ, PATH=fakebin + os.pathsep + os.environ.get("PATH", ""),
               FAKEPBS_STATE=state, FAKEPBS_CONDA_BASE=base, CONFIG=cfg,
               COHORT_TAG="giulia", SBI_HPC_DIR=sbi_hpc, ENV_NAME="sbi_export")
    env.pop("DRYRUN", None)
    r = subprocess.run(["bash", os.path.join(repo, "extractor", "launch_stage_d.sh"), extract_root],
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def j1_to_j6(repo, root, td, fakebin, state, base, sbi_hpc):
    extract_root = os.path.join(td, "extracted_giulia")
    cfg = write_config(td, sbi_hpc, root, extract_root, [TOO_FEW], "config_jobs.json")
    rc, log = run_launch(repo, cfg, extract_root, fakebin, state, base, sbi_hpc)
    m_arr = re.search(r"array : (\S+)", log)
    m_agg = re.search(r"agg   : (\S+)", log)
    ok("J1 launch_stage_d.sh (tagged, not DRYRUN) exits 0 and reports both job ids",
       rc == 0 and m_arr and m_agg
       and "held until" not in log
       and "runs once the array has ended, whatever its tasks' exit codes" in log
       and "then:    cat out/cohort_manifest_giulia.log" in log
       and "grep -h" not in log,
       log.strip().splitlines()[-1][:70] if log.strip() else "")
    calls = read_jsonl(os.path.join(state, "calls.jsonl"))
    arr = [c for c in calls if "-J" in c["argv"]]
    agg = [c for c in calls if any(a.startswith("depend=afterok:") for a in c["argv"])]
    va = arr[0]["argv"][arr[0]["argv"].index("-v") + 1] if arr else ""
    vg = agg[0]["argv"][agg[0]["argv"].index("-v") + 1] if agg else ""
    ok("J2 qsub: -J 0-4, -N chsub_mea_array_giulia, the cohort's files in -v; agg after the array",
       len(arr) == 1 and arr[0]["argv"][arr[0]["argv"].index("-J") + 1] == "0-4"
       and "chsub_mea_array_giulia" in arr[0]["argv"]
       and "CHSUB_MANIFEST=out/extraction_manifest_giulia.tsv" in va
       and "CHSUB_FLAGS=extraction_flags_giulia.sh" in va
       and len(agg) == 1 and ("depend=afterok:%s" % m_arr.group(1)) in agg[0]["argv"]
       and "CHSUB_MANIFEST=out/extraction_manifest_giulia.tsv" in vg
       and ("CONFIG=%s" % cfg) in vg,
       va[:80])
    exits = read_jsonl(os.path.join(state, "exits.jsonl"))
    tasks = [e for e in exits if e["index"] is not None]
    logs_ok = all("--ptrain-format sparse_peaks" in open(e["log"]).read()
                  and "done ->" in open(e["log"]).read() for e in tasks)
    ok("J3 five array tasks, each exit 0, each log shows the Giulia flags and 'done ->'",
       len(tasks) == 5 and all(e["rc"] == 0 for e in tasks) and logs_ok,
       str([e["rc"] for e in tasks]))
    tsv = os.path.join(repo, "extractor", "out", "extraction_manifest_giulia.tsv")
    rows = [ln.rstrip("\n").split("\t") for ln in open(tsv)] if os.path.isfile(tsv) else []
    good = True
    for _folder, out_dir, _c in rows:
        fs = os.listdir(out_dir) if os.path.isdir(out_dir) else []
        meta = json.load(open(os.path.join(out_dir, "traces_meta.json"))) if "traces_meta.json" in fs else {}
        good = good and sum(f.startswith("trace_subregion_") for f in fs) == 9 \
            and {"traces.npz", "subregion_map.png", "subregion_ifrs.png"} <= set(fs) \
            and meta.get("ptrain_format") == "sparse_peaks"
    excluded_out = [d for d, _s, _f in os.walk(extract_root) if TOO_FEW in d]
    ok("J4 every kept well: 9 archives + traces.npz + both PNGs + sparse_peaks fragment; excluded well absent",
       len(rows) == 5 and good and not excluded_out, "%d rows" % len(rows))
    mlog = os.path.join(repo, "extractor", "out", "cohort_manifest_giulia.log")
    mtext = open(mlog).read() if os.path.isfile(mlog) else ""
    ok("J5 out/cohort_manifest_giulia.log: 'wrote .../cohort_manifest.json  sha256' and exit 0",
       re.search(r"wrote \S+/cohort_manifest\.json  sha256 [0-9a-f]{16}", mtext) is not None
       and "cohort_manifest_exit=0" in mtext,
       (mtext.strip().splitlines() or [""])[-1][:70])
    mp = os.path.join(extract_root, "cohort_manifest.json")
    doc = json.load(open(mp)) if os.path.isfile(mp) else {}
    ok("J6 manifest: 5 wells x 9 = 45 units, 3 classes, source_format, excluded_wells, T_rec 600..1200",
       doc.get("n_wells") == 5 and doc.get("n_units") == 45
       and doc.get("classes") == ["100N0A_wo", "50N50A", "70N30A"]
       and (doc.get("source_format") or {}).get("ptrain_format") == "sparse_peaks"
       and doc.get("excluded_wells") == [TOO_FEW]
       and [round(x, 6) for x in doc.get("T_rec_range", [])] == [600.0, 1200.0],
       json.dumps({k: doc.get(k) for k in ("n_wells", "n_units", "T_rec_range")}))


def j7_refusal(repo, root, td, fakebin, sbi_hpc):
    state = os.path.join(td, "pbs_state_neg")
    os.makedirs(state, exist_ok=True)
    base = os.path.join(td, "conda_base")
    extract_root = os.path.join(td, "extracted_giulia_neg")
    cfg = write_config(td, sbi_hpc, root, extract_root, [], "config_jobs_neg.json")
    rc, log = run_launch(repo, cfg, extract_root, fakebin, state, base, sbi_hpc)
    exits = read_jsonl(os.path.join(state, "exits.jsonl"))
    tasks = [e for e in exits if e["index"] is not None]
    bad = [e for e in tasks if e["rc"] != 0]
    bad_log = open(bad[0]["log"]).read() if bad else ""
    mlog = os.path.join(repo, "extractor", "out", "cohort_manifest_giulia.log")
    mtext = open(mlog).read() if os.path.isfile(mlog) else ""
    ok("J7 exclude_wells empty: the too-few well's task fails, the aggregation REFUSES naming it",
       len(tasks) == 6 and len(bad) == 1 and "InsufficientElectrodesError" in bad_log
       and "REFUSED" in mtext and TOO_FEW in mtext
       and not os.path.isfile(os.path.join(extract_root, "cohort_manifest.json")),
       (mtext.strip().splitlines() or [""])[-1][:70])
    # [2026-10-02] what the refusal says about that well, and the exit line
    tsv = os.path.join(repo, "extractor", "out", "extraction_manifest_giulia.tsv")
    lines = [ln.rstrip("\n").split("\t") for ln in open(tsv)] if os.path.isfile(tsv) else []
    idx = [i for i, r in enumerate(lines) if len(r) == 3 and TOO_FEW in r[2]]
    want_idx = "array index %d)" % idx[0] if idx else "<no row>"
    ok("J7b the refusal names out/extraction_manifest_giulia.tsv, the well's array index, "
       "'did not complete'; exit line 1",
       len(idx) == 1 and bad and bad[0]["index"] == idx[0]
       and "out/extraction_manifest_giulia.tsv" in mtext and want_idx in mtext
       and "did not complete" in mtext and "version 1" not in mtext
       and "cohort_manifest_exit=1" in mtext,
       "%s; task %s failed" % (want_idx, bad[0]["index"] if bad else None))
    return cfg, extract_root


def j8_relaunch_after_refusal(repo, td, fakebin, sbi_hpc, cfg, extract_root):
    state = os.path.join(td, "pbs_state_j8")
    os.makedirs(state, exist_ok=True)
    base = os.path.join(td, "conda_base")
    rc, log = run_launch(repo, cfg, extract_root, fakebin, state, base, sbi_hpc)
    calls = read_jsonl(os.path.join(state, "calls.jsonl"))
    ok("J8 re-launch into the refused, declared root: exit 3, 'holds no cohort_manifest.json', nothing submitted",
       rc == 3 and "holds no cohort_manifest.json" in log and "archives of record" in log
       and "Nothing was deleted or submitted" in log and not calls
       and os.path.isdir(extract_root),
       log.strip().splitlines()[-1][:70] if log.strip() else "")
    aside = extract_root + "_partial_test"
    os.rename(extract_root, aside)
    env = dict(os.environ, PATH=fakebin + os.pathsep + os.environ.get("PATH", ""),
               FAKEPBS_STATE=state, FAKEPBS_CONDA_BASE=base, CONFIG=cfg,
               COHORT_TAG="giulia", SBI_HPC_DIR=sbi_hpc, ENV_NAME="sbi_export", DRYRUN="1")
    r = subprocess.run(["bash", os.path.join(repo, "extractor", "launch_stage_d.sh"), extract_root],
                       capture_output=True, text=True, env=env)
    log2 = r.stdout + r.stderr
    calls = read_jsonl(os.path.join(state, "calls.jsonl"))
    ok("J8b after moving it aside: the same launch (DRYRUN) is a first extraction again, 6 wells, nothing submitted",
       r.returncode == 0 and "a first extraction" in log2 and "array 0-5" in log2
       and "(DRYRUN -- nothing was submitted" in log2 and not calls
       and os.path.isdir(aside),
       log2.strip().splitlines()[-1][:70] if log2.strip() else "")


def main():
    print("=" * 100)
    print("Smoke test: the Stage D job layer end to end (fake qsub, stub conda, figures on)")
    print("=" * 100)
    sbi_hpc = os.path.realpath(dsn_tree.sbi_hpc_dir())       # same resolution as every job
    if not os.path.isfile(os.path.join(sbi_hpc, GIULIA_CONFIG_REL)):
        print("ABORT: %s not found under SBI_HPC_DIR=%s (pull the SBI commit first)"
              % (GIULIA_CONFIG_REL, sbi_hpc))
        return 2
    td = tempfile.mkdtemp(prefix="stage_d_jobs_")
    try:
        fakebin, state, base = make_fake_scheduler(td)
        repo = copy_repo(td, sbi_hpc)
        root = build_cohort(td, np.random.default_rng(20261001))
        j0_single_well(repo, root, td)
        j1_to_j6(repo, root, td, fakebin, state, base, sbi_hpc)
        cfg_neg, root_neg = j7_refusal(repo, root, td, fakebin, sbi_hpc)
        j8_relaunch_after_refusal(repo, td, fakebin, sbi_hpc, cfg_neg, root_neg)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(False)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("-" * 100)
    n_fail = RESULTS.count(False)
    print("%d passed, %d failed" % (RESULTS.count(True), n_fail))
    print("ALL STAGE-D JOB CHECKS PASSED" if n_fail == 0 else "STAGE-D JOB FAILURES: %d" % n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
