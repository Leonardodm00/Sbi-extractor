#!/usr/bin/env python3
"""
smoke_test_sim_reextract.py -- Stage C, C8 end to end on a synthetic
simulation tree, through the REAL virtual-MEA pipeline.

    cd ~/repos/Sbi-extractor/sim_reextract && conda activate sbi_export
    ANN_TOOLS=/davinci-1/home/ldellamea/ANN/MEA_analysis python3 smoke_test_sim_reextract.py

Expect: ALL 23 CHECKS PASSED, in two to three minutes (one template-library
build of about 30 s, then process_campaign.py over a few dozen synthetic
iterations). Needs bash and a python3 with numpy + scipy -- sbi_export -- and
the ANN tools folder (ANN_TOOLS, or --tools-dir): process_campaign.py and
its modules at a state the plan knows. Nothing is submitted to PBS: the
launcher is run with a fake qsub that executes submit_mea_array.sh in place,
one array index after another, the way a node would.

WHAT IT BUILDS
  SIM_MAIN with campaign_cadex_rho1300v1 (2 tasks, one of them with two
  topologies), v7 (task0 ok; task1 has job_args.json and topo_*/ but no
  manifest.json, as v7/sweep_cpu_task0009 on davinci), v9 (task0 ok; three
  empty sweep folders, as v9's 46), a campaign_cadex_hhgap_v1 that the glob
  must not match, and a Giulia_Astro folder; every task's job_args.json and
  manifest.json carry one contract. A cohort manifest with its .sha256
  (electrodes_per_subset 9, fs_raw 10110.09). A TOOLS folder: the tool files
  copied from --tools-dir plus a template library built once, as
  ANN/MEA_analysis is laid out on davinci.

CHECKS
  P1  the plan: 4 tasks to run, 4 excluded and named with their reasons,
      n_side 3 from the manifest, EXTRA_ARGS, the tools' label, the
      library's sha256, one contract, tasks.tsv of 4 lines
  P2  a task whose simtime differs -> REFUSED naming simtime;
      --allow-mixed-contract -> accepted, the field reported as mixed
  P3  refused: no .sha256 sidecar; a manifest that no longer matches it;
      electrodes_per_subset 8 (not a square)
  P4  refused: tools no known state (named file); an older known state
      (refused without --allow-older-tools, accepted with it)
  P5  refused: an output root that holds detections; --resume keeps the
      complete task and lists the others
  L1  launch.sh `plan` with DRYRUN=1 submits nothing and says so
  L2  launch.sh `test` with DRYRUN=1 prints one qsub line: a plain job,
      CONDA_ENV, the plan's EXTRA_ARGS, the frozen manifest beside its sidecar
  L3  launch.sh `array` refuses without WALLTIME; with it and DRYRUN=1 prints
      the array line (-J 0-N%C) and the gate's line
  E1  launch.sh `test` (fake qsub): the task runs through submit_mea_array.sh
      and process_campaign.py; mea_manifest.json, mea_env.json, mea_iter files
  E2  launch.sh `array` with RESUME=1: the plan keeps the test task as done,
      the other 3 run, the gate job is submitted with depend=afterok:<array>
  E3  the gate PASSES: REEXTRACTION_RECORD.json (+ .sha256) with the counts,
      the geometry, the environment of this interpreter, the tools' label
  G1  the gate refuses to run over an existing record
  G2  a deleted mea_iter file -> FAIL naming the task and the counts
  G3  a missing mea_env.json -> FAIL
  G4  a _failures.log with every iteration done -> WARN, named in the
      record; with an iteration not done -> FAIL
  G5  the plan's pitch changed (61) -> FAIL on meta_json pitch of every file
  G6  total_done one short in a mea_manifest.json -> FAIL
  G7  the library's bytes changed -> REFUSED before any task
  G8  one task's mea_env.json names another numpy -> FAIL environment
  G9  the tools folder changed since the plan -> REFUSED
  G10 after every repair, the gate passes again
  R1  one task's mea_manifest.json removed (a walltime kill): plan --resume
      -> 3 done, 1 run; the array (one task: a plain job) re-runs it; the
      gate passes and the record counts all 4
  R2  the plan refuses a root that holds a record (finished)

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import cohort_manifest as CM                    # noqa: E402
import dsn_tree                                 # noqa: E402
import sim_reextract_plan as PLAN               # noqa: E402

SBI_HPC = os.path.dirname(dsn_tree.dsn_dir())   # the DSN tree cohort_config.py imports from

_PASS, _FAIL = [], []

TOOL_COPY = ("process_campaign.py", "mea_probe.py", "mea_detection.py", "mea_synthesis.py",
             "mea_plots.py", "eap_template_library.py", "submit_mea_array.sh",
             "launch_mea_array.sh", "build_mea_manifest.py")

FAKE_QSUB = r'''#!/bin/bash
# fake qsub for smoke_test_sim_reextract.py: runs submit_mea_array.sh in
# place, one array index after another, with the -v variables and the PBS
# variables a node would see; any other script is only recorded. Prints a
# job id. Every call is appended to $FAKE_QSUB_LOG.
set -u
name=job; vars=""; jrange=""; depend=""; script=""
while [ $# -gt 0 ]; do
    case "$1" in
        -N) name="$2"; shift 2 ;;
        -q|-l) shift 2 ;;
        -J) jrange="$2"; shift 2 ;;
        -W) depend="$2"; shift 2 ;;
        -v) vars="$2"; shift 2 ;;
        *) script="$1"; shift ;;
    esac
done
n=$(cat "$FAKE_QSUB_COUNTER" 2>/dev/null || echo 100); n=$((n + 1)); echo "$n" > "$FAKE_QSUB_COUNTER"
if [ -n "$jrange" ]; then jid="${n}[].fake"; else jid="${n}.fake"; fi
echo "$jid|$name|$jrange|$depend|$vars|$script|$PWD" >> "$FAKE_QSUB_LOG"
case "$script" in
    *submit_mea_array.sh)
        lo=0; hi=0
        if [ -n "$jrange" ]; then r="${jrange%%%*}"; lo="${r%-*}"; hi="${r#*-}"; fi
        for i in $(seq "$lo" "$hi"); do
            # -v is comma-separated NAME=VALUE; values here carry no commas
            ( IFS=','; for kv in $vars; do export "$kv"; done
              export PBS_ARRAY_INDEX="$i" PBS_O_WORKDIR="$PWD" PBS_JOBID="$jid" PBS_NCPUS="${FAKE_NCPUS:-2}"
              [ -z "$jrange" ] && unset PBS_ARRAY_INDEX
              bash "$script" > "$FAKE_QSUB_LOGDIR/${name}.o${n}.${i}" 2>&1
              echo "exit=$?" >> "$FAKE_QSUB_LOGDIR/${name}.o${n}.${i}" )
        done ;;
esac
echo "$jid"
'''


def check(name, fn):
    try:
        d = fn()
    except Exception as exc:                                  # noqa: BLE001
        _FAIL.append((name, "%s: %s" % (type(exc).__name__, exc)))
        print("  [FAIL] %-3s %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
        return
    _PASS.append((name, d))
    print("  [PASS] %-3s %s" % (name, d))


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _json(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
class Fixture:
    CONTRACT_JOB_ARGS = {"simtime": 200.0, "conn_prob_lo": 0.1, "conn_prob_hi": 0.6,
                         "p0_lo": None, "p0_hi": None, "d0_lo": None, "d0_hi": None,
                         "beta_lo": None, "beta_hi": None, "c_max": 600.0}
    CONTRACT_MANIFEST = {"manifest_version": 3, "active_indices": list(range(23)),
                         "sweep_group": "neuron_synapse", "log_transform": "natural",
                         "param_names": ["p%d" % i for i in range(36)],
                         "log_params": ["p%d" % i for i in range(17)],
                         "param_bounds": [[0.1 * (i + 1), 1.0 * (i + 1)] for i in range(36)],
                         "conn_rule": "weibull"}

    def __init__(self, tools_src, library=None):
        self.root = tempfile.mkdtemp(prefix="sim_reextract_")
        r = self.root
        self.sim_main = os.path.join(r, "Main")
        self.out_root = os.path.join(r, "Outputs_v2")
        self.tools = os.path.join(r, "MEA_analysis")
        self.artifacts = os.path.join(r, "artifacts")
        self.work = os.path.join(r, "sim_reextract")        # a copy of this folder's scripts
        os.makedirs(self.tools)
        os.makedirs(self.artifacts)
        shutil.copytree(HERE, self.work, ignore=shutil.ignore_patterns("__pycache__", "out", "*.tsv", "plan.json"))
        with open(os.path.join(r, "env.sh"), "w") as fh:        # ../env.sh of the copy
            fh.write(': "${ARTIFACTS_DIR:=%s}"\n: "${ENV_NAME:=sbi_export}"\n: "${SBI_HPC_DIR:=%s}"\n'
                     'export ARTIFACTS_DIR ENV_NAME SBI_HPC_DIR\n' % (self.artifacts, SBI_HPC))
        for n in ("cohort_manifest.py", "cohort_config.py", "dsn_tree.py"):
            shutil.copyfile(os.path.join(HERE, "..", n), os.path.join(r, n))
        for n in TOOL_COPY:
            shutil.copyfile(os.path.join(tools_src, n), os.path.join(self.tools, n))
        self.library = os.path.join(self.tools, "eap_library.npz")
        if library and os.path.isfile(library):
            shutil.copyfile(library, self.library)
        else:
            subprocess.run([sys.executable, os.path.join(self.tools, "eap_template_library.py"),
                            "--out", self.library], check=True, capture_output=True, text=True)
        # the cohort manifest of record, with its sidecar, at a path with a space
        self.cm_dir = os.path.join(r, "Deep Summary Network", "extracted_v2")
        self.cm = os.path.join(self.cm_dir, "cohort_manifest.json")
        self.write_cohort_manifest(self.cm, 9)
        # the raw simulations
        self.rng = np.random.default_rng(0)
        self.tasks = {}
        self.make_task("campaign_cadex_rho1300v1", "sweep_cpu_task0000", n_topos=2, n_iters=2)
        self.make_task("campaign_cadex_rho1300v1", "sweep_intel_task0000", n_topos=1, n_iters=3)
        self.make_task("campaign_cadex_rho1300v7", "sweep_cpu_task0000", n_topos=1, n_iters=2)
        self.make_task("campaign_cadex_rho1300v7", "sweep_cpu_task0001", n_topos=1, n_iters=2, manifest=False)
        self.make_task("campaign_cadex_rho1300v9", "sweep_cpu_task0000", n_topos=1, n_iters=2)
        for k in range(1, 4):
            os.makedirs(os.path.join(self.sim_main, "campaign_cadex_rho1300v9", "sweep_cpu_task%04d" % k))
        self.make_task("campaign_cadex_hhgap_v1", "sweep_cpu_task0000", n_topos=1, n_iters=1)
        os.makedirs(os.path.join(self.sim_main, "Giulia_Astro", "campaign_cadex_hhgap_v2"))
        # the fake qsub
        self.bin = os.path.join(r, "bin")
        os.makedirs(self.bin)
        self.qsub = os.path.join(self.bin, "qsub")
        _exe(self.qsub, FAKE_QSUB)
        self.qsub_log = os.path.join(r, "qsub_calls.log")
        self.qsub_logdir = os.path.join(r, "joblogs")
        os.makedirs(self.qsub_logdir)
        self.qsub_counter = os.path.join(r, "qsub_counter")

    def write_cohort_manifest(self, path, electrodes_per_subset, tamper=False):
        doc = {"manifest_version": CM.MANIFEST_VERSION, "created_utc": "2026-09-21T14:36:26+00:00",
               "extractor_version": "run_channel_subset_extraction/3", "extractor_commit": "442eaf4",
               "preprocessing": {"w_size": 0.01, "gaussian_window": 0.02,
                                 "electrodes_per_subset": electrodes_per_subset, "n_subsets": 9,
                                 "mfr_threshold": 0.1, "fs_raw": 10110.09, "index_base": 1,
                                 "grid_width": 48},
               "derived": {"fs_ifr": 100.0, "sigma_sm_bins": 2.0},
               "n_wells": 35, "n_subsets_per_well": 9, "n_units": 315}
        CM.write_manifest(doc, path)
        if tamper:
            with open(path, "a") as fh:
                fh.write("\n")

    def make_task(self, campaign, sweep, n_topos, n_iters, manifest=True, job_args=None):
        d = os.path.join(self.sim_main, campaign, sweep)
        os.makedirs(d, exist_ok=True)
        ja = dict(self.CONTRACT_JOB_ARGS)
        if job_args:
            ja.update(job_args)
        with open(os.path.join(d, "job_args.json"), "w") as fh:
            json.dump(ja, fh)
        if manifest:
            with open(os.path.join(d, "manifest.json"), "w") as fh:
                json.dump(self.CONTRACT_MANIFEST, fh)
        c_max = 600.0
        for k in range(n_topos):
            td = os.path.join(d, "topo_%05d" % k)
            os.makedirs(td, exist_ok=True)
            Nn = 40
            N_pos = self.rng.uniform(0, c_max, (Nn, 2))
            N_pos[0] = [c_max / 2, c_max / 2]
            np.savez_compressed(os.path.join(td, "topology.npz"), N_pos=N_pos)
            with open(os.path.join(td, "topology_meta.json"), "w") as fh:
                json.dump({"c_max": c_max}, fh)
            for n in range(n_iters):
                spk_t, spk_i = [], []
                for i in range(Nn):
                    times = np.sort(self.rng.uniform(0.2, 1.8, self.rng.integers(3, 12)))
                    spk_t.append(times)
                    spk_i.append(np.full(len(times), i))
                np.savez_compressed(
                    os.path.join(td, "iter_%05d.npz" % n),
                    params=np.arange(36, dtype=float), theta=np.arange(26, dtype=float),
                    conn_prob=np.float64(0.3),
                    spk_N_t=np.concatenate(spk_t).astype(np.float32),
                    spk_N_i=np.concatenate(spk_i).astype(np.int32),
                    spk_A_t=np.array([]), spk_A_i=np.array([]), seed_run=np.int64(123 + n))
        self.tasks[(campaign, sweep)] = (n_topos, n_iters)

    # --- running the scripts -------------------------------------------------
    def plan(self, *extra, cm=None, out_root=None, tools=None, resume=False):
        cmd = [sys.executable, os.path.join(self.work, "sim_reextract_plan.py"),
               "--cohort-manifest", cm or self.cm, "--sim-main", self.sim_main,
               "--out-root", out_root or self.out_root, "--tools", tools or self.tools,
               "--plan-out", os.path.join(self.work, "plan.json"),
               "--tasks-out", os.path.join(self.work, "tasks.tsv")] + list(extra)
        if resume:
            cmd.append("--resume")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=self.pyenv())
        return p.returncode, p.stdout + p.stderr

    @staticmethod
    def pyenv():
        e = dict(os.environ)
        e["SBI_HPC_DIR"] = SBI_HPC
        return e

    def gate(self, *extra):
        cmd = [sys.executable, os.path.join(self.work, "sim_reextract_gate.py"),
               "--plan", os.path.join(self.work, "plan.json"), "--workers", "2"] + list(extra)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=self.pyenv())
        return p.returncode, p.stdout + p.stderr

    def launch(self, mode, dryrun, **env):
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("PBS_", "CONDA", "_CONDA", "FAKE_"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", "")
        e.update(COHORT_MANIFEST=self.cm, SIM_MAIN=self.sim_main, OUT_ROOT=self.out_root,
                 TOOLS=self.tools, ARTIFACTS_DIR=self.artifacts, QSUB=self.qsub,
                 NCPUS="2", CONCURRENCY="4", FAKE_QSUB_LOG=self.qsub_log,
                 FAKE_QSUB_LOGDIR=self.qsub_logdir, FAKE_QSUB_COUNTER=self.qsub_counter,
                 FAKE_NCPUS="2", HOME=self.root, SBI_HPC_DIR=SBI_HPC)
        e["DRYRUN"] = "1" if dryrun else "0"
        e.update({k: str(v) for k, v in env.items()})
        p = subprocess.run(["bash", os.path.join(self.work, "launch_sim_reextract.sh"), mode],
                           env=e, capture_output=True, text=True, timeout=900, cwd=self.work)
        return p.returncode, p.stdout + p.stderr

    def qsub_calls(self):
        if not os.path.isfile(self.qsub_log):
            return []
        return [ln.split("|") for ln in open(self.qsub_log).read().splitlines() if ln]

    def out_dir(self, campaign, sweep):
        return os.path.join(self.out_root, campaign, sweep)

    def plan_doc(self):
        return _json(os.path.join(self.work, "plan.json"))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-dir", default=os.environ.get("ANN_TOOLS") or
                    ("/davinci-1/home/ldellamea/ANN/MEA_analysis"
                     if os.path.isdir("/davinci-1/home/ldellamea/ANN/MEA_analysis") else None),
                    help="the ANN tools (ANN_TOOLS); process_campaign.py and its modules")
    ap.add_argument("--library", default=None, help="a pre-built eap_library.npz, to skip the 30 s build")
    ap.add_argument("--keep", action="store_true", help="keep the fixture directory")
    a = ap.parse_args(argv)
    if not a.tools_dir or not os.path.isfile(os.path.join(a.tools_dir, "process_campaign.py")):
        print("give --tools-dir (or ANN_TOOLS): a folder with process_campaign.py", file=sys.stderr)
        return 2
    print("smoke_test_sim_reextract -- tools from %s" % a.tools_dir)
    F = Fixture(a.tools_dir, a.library)
    print("fixture: %s" % F.root)
    print("-" * 70)
    required_label = PLAN.REQUIRED_TOOLS_LABEL

    def p1():
        rc, out = F.plan()
        if rc != 0:
            raise AssertionError("plan refused (rc %d):\n%s" % (rc, out[-1500:]))
        d = F.plan_doc()
        run = [t for t in d["tasks"] if t["status"] == "run"]
        if len(run) != 4 or d["counts"]["tasks_run"] != 4:
            raise AssertionError("expected 4 run tasks, got %d" % len(run))
        ex = {"%s/%s" % (e["campaign"], e["sweep"]): e["reason"] for e in d["excluded"]}
        if len(ex) != 4 or "campaign_cadex_rho1300v7/sweep_cpu_task0001" not in ex \
                or "no manifest.json" not in ex["campaign_cadex_rho1300v7/sweep_cpu_task0001"] \
                or not all("empty sweep folder" in ex.get("campaign_cadex_rho1300v9/sweep_cpu_task%04d" % k, "")
                           for k in (1, 2, 3)):
            raise AssertionError("exclusions wrong: %r" % ex)
        if d["campaigns"] != ["campaign_cadex_rho1300v1", "campaign_cadex_rho1300v7", "campaign_cadex_rho1300v9"]:
            raise AssertionError("campaigns: %r" % d["campaigns"])
        g = d["geometry"]
        if (g["n_side"], g["n_e"], g["pitch_um"], g["edge_um"], g["fs"]) != (3, 9, 60.0, 25.0, 10110.09):
            raise AssertionError("geometry: %r" % g)
        if d["extra_args"] != "--n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09":
            raise AssertionError("extra_args: %r" % d["extra_args"])
        if d["tools"]["label"] != required_label:
            raise AssertionError("tools label: %r" % d["tools"]["label"])
        if d["library"]["sha256"] != _sha(F.library):
            raise AssertionError("library sha256 wrong")
        if d["contract"]["mixed"] or d["contract"]["fields"].get("simtime") != 200.0 \
                or d["contract"]["fields"].get("sweep_group") != "neuron_synapse":
            raise AssertionError("contract: %r" % d["contract"])
        lines = open(os.path.join(F.work, "tasks.tsv")).read().splitlines()
        if len(lines) != 4 or any(len(ln.split("\t")) != 2 for ln in lines):
            raise AssertionError("tasks.tsv: %r" % lines)
        v1 = [t for t in run if t["campaign"] == "campaign_cadex_rho1300v1" and t["sweep"] == "sweep_cpu_task0000"][0]
        if v1["n_topos"] != 2 or v1["n_iters"] != 4 or d["counts"]["iters_run"] != 4 + 3 + 2 + 2:
            raise AssertionError("counts: %r / %r" % (v1, d["counts"]))
        return "plan: 4 to run, 4 excluded and named, n_side 3, EXTRA_ARGS, tools, library, one contract"

    def p2():
        ja = os.path.join(F.sim_main, "campaign_cadex_rho1300v9", "sweep_cpu_task0000", "job_args.json")
        orig = open(ja).read()
        d = json.loads(orig)
        d["simtime"] = 180.0
        with open(ja, "w") as fh:
            json.dump(d, fh)
        try:
            rc, out = F.plan()
            if rc == 0 or "REFUSED" not in out or "simtime" not in out or "180" not in out:
                raise AssertionError("mixed simtime not refused (rc %d):\n%s" % (rc, out[-800:]))
            rc2, out2 = F.plan("--allow-mixed-contract")
            if rc2 != 0 or "simtime" not in F.plan_doc()["contract"]["mixed"]:
                raise AssertionError("--allow-mixed-contract did not proceed (rc %d):\n%s" % (rc2, out2[-800:]))
        finally:
            with open(ja, "w") as fh:
                fh.write(orig)
        return "a mixed contract is refused naming the field; --allow-mixed-contract proceeds and reports it"

    def p3():
        cm2 = os.path.join(F.root, "cm2", "cohort_manifest.json")
        F.write_cohort_manifest(cm2, 9)
        os.remove(cm2 + ".sha256")
        rc, out = F.plan(cm=cm2)
        if rc == 0 or "sidecar" not in out:
            raise AssertionError("no-sidecar not refused (rc %d):\n%s" % (rc, out[-600:]))
        cm3 = os.path.join(F.root, "cm3", "cohort_manifest.json")
        F.write_cohort_manifest(cm3, 9, tamper=True)
        rc, out = F.plan(cm=cm3)
        if rc == 0 or "does not match its .sha256" not in out:
            raise AssertionError("tampered manifest not refused (rc %d):\n%s" % (rc, out[-600:]))
        cm4 = os.path.join(F.root, "cm4", "cohort_manifest.json")
        F.write_cohort_manifest(cm4, 8)
        rc, out = F.plan(cm=cm4)
        if rc == 0 or "perfect square" not in out:
            raise AssertionError("E = 8 not refused (rc %d):\n%s" % (rc, out[-600:]))
        return "refused: no sidecar, a tampered manifest, electrodes_per_subset 8"

    def p4():
        tools2 = os.path.join(F.root, "tools_unknown")
        shutil.copytree(F.tools, tools2)
        with open(os.path.join(tools2, "submit_mea_array.sh"), "a") as fh:
            fh.write("# edited\n")
        rc, out = F.plan(tools=tools2)
        if rc == 0 or "no known state" not in out or "submit_mea_array.sh" not in out:
            raise AssertionError("unknown tools not refused (rc %d):\n%s" % (rc, out[-600:]))
        # an older known state: register this edited copy as one, in-process
        files, _ = PLAN.fingerprint_tools(tools2)
        PLAN.KNOWN_TOOLS["older (test)"] = dict(files)
        try:
            try:
                PLAN.check_tools(tools2, allow_older=False)
                raise AssertionError("older tools accepted without --allow-older-tools")
            except PLAN.PlanError as exc:
                if "older" not in str(exc):
                    raise
            _, label = PLAN.check_tools(tools2, allow_older=True)
            if label != "older (test)":
                raise AssertionError("label %r" % label)
        finally:
            del PLAN.KNOWN_TOOLS["older (test)"]
        return "refused: tools of no known state (file named); an older state unless allowed"

    def p5():
        od = F.out_dir("campaign_cadex_rho1300v1", "sweep_intel_task0000")
        os.makedirs(os.path.join(od, "topo_00000"))
        with open(os.path.join(od, "mea_manifest.json"), "w") as fh:
            json.dump({"n_topos": 1, "total_iters": 3, "total_done": 3, "config": {}}, fh)
        with open(os.path.join(od, "mea_env.json"), "w") as fh:
            json.dump({"record": "mea_env"}, fh)
        try:
            rc, out = F.plan()
            if rc == 0 or "already holds detections" not in out:
                raise AssertionError("existing output not refused (rc %d):\n%s" % (rc, out[-600:]))
            rc, out = F.plan(resume=True)
            d = F.plan_doc()
            done = [t for t in d["tasks"] if t["status"] == "done"]
            if rc != 0 or len(done) != 1 or done[0]["sweep"] != "sweep_intel_task0000" \
                    or d["counts"]["tasks_run"] != 3 or d["resume"] is not True:
                raise AssertionError("--resume wrong (rc %d): %r\n%s" % (rc, d["counts"], out[-600:]))
        finally:
            shutil.rmtree(F.out_root)
        return "refused: a root with detections; --resume keeps the complete task, runs 3"

    def l1():
        rc, out = F.launch("plan", True)
        if rc != 0 or "plan only" not in out or "nothing submitted" not in out or F.qsub_calls():
            raise AssertionError("plan mode (rc %d):\n%s" % (rc, out[-800:]))
        frozen = os.path.join(F.artifacts, "cohort_manifest", "cohort_manifest.json")
        if not os.path.isfile(frozen) or not os.path.isfile(frozen + ".sha256") \
                or _sha(frozen) != _sha(F.cm) or "froze" not in out:
            raise AssertionError("manifest not frozen with its sidecar:\n%s" % out[-600:])
        return "launch plan: plan.json written, the manifest frozen beside its sidecar, nothing submitted"

    def l2():
        rc, out = F.launch("test", True)
        if rc != 0 or "DRYRUN" not in out or F.qsub_calls():
            raise AssertionError("test dry run (rc %d):\n%s" % (rc, out[-800:]))
        line = [ln for ln in out.splitlines() if "submit_mea_array.sh" in ln and "qsub" in ln]
        if len(line) != 1 or "-J" in line[0] or "CONDA_ENV=sbi_export" not in line[0] \
                or "EXTRA_ARGS=--n_side 3 --pitch 60.0 --edge 25.0 --fs 10110.09" not in line[0] \
                or "tasks_test.tsv" not in line[0] or "-N c8_test" not in line[0]:
            raise AssertionError("qsub line: %r" % line)
        if "frozen manifest" not in out:
            raise AssertionError("second run did not reuse the frozen manifest:\n%s" % out[-600:])
        return "launch test, DRYRUN: one plain-job qsub line with CONDA_ENV and the plan's EXTRA_ARGS"

    def l3():
        rc, out = F.launch("array", True)
        if rc == 0 or "WALLTIME" not in out:
            raise AssertionError("array without WALLTIME not refused (rc %d):\n%s" % (rc, out[-600:]))
        rc, out = F.launch("array", True, WALLTIME="01:30:00")
        lines = [ln for ln in out.splitlines() if "qsub" in ln]
        arr = [ln for ln in lines if "submit_mea_array.sh" in ln]
        gate = [ln for ln in lines if "run_sim_reextract_gate.pbs" in ln]
        if rc != 0 or len(arr) != 1 or "-J 0-3%4" not in arr[0] or "walltime=01:30:00" not in arr[0] \
                or "-N c8_mea" not in arr[0] or len(gate) != 1 or "depend=afterok" not in gate[0] \
                or F.qsub_calls():
            raise AssertionError("array dry run (rc %d): %r %r\n%s" % (rc, arr, gate, out[-600:]))
        return "launch array: WALLTIME required; DRYRUN prints -J 0-3%4 and the dependent gate line"

    def e1():
        rc, out = F.launch("test", False)
        calls = F.qsub_calls()
        if rc != 0 or len(calls) != 1 or calls[0][1] != "c8_test" or calls[0][2] != "":
            raise AssertionError("test launch (rc %d) calls %r:\n%s" % (rc, calls, out[-800:]))
        od = F.out_dir("campaign_cadex_rho1300v1", "sweep_cpu_task0000")
        log = glob.glob(os.path.join(F.qsub_logdir, "c8_test.o*"))
        text = open(log[0]).read() if log else ""
        if "exit=0" not in text or "index 0 done." not in text:
            raise AssertionError("the job did not complete:\n%s" % text[-1200:])
        man = _json(os.path.join(od, "mea_manifest.json"))
        env = _json(os.path.join(od, "mea_env.json"))
        files = sorted(glob.glob(os.path.join(od, "topo_*", "mea_iter_*.npz")))
        if man["total_done"] != 4 or man["n_topos"] != 2 or len(files) != 4:
            raise AssertionError("output: %r, %d files" % (man, len(files)))
        if env["extra_args"] != F.plan_doc()["extra_args"] or env["library_sha256"] != _sha(F.library) \
                or env["numpy"] != np.__version__:
            raise AssertionError("mea_env.json: %r" % env)
        with np.load(files[0], allow_pickle=False) as d:
            meta = json.loads(str(d["meta_json"]))
            if d["electrode_centers"].shape != (9, 2) or meta["pitch"] != 60.0 or meta["edge"] != 25.0 \
                    or meta["n_side"] != 3 or abs(meta["fs"] - 10110.09) > 1e-6:
                raise AssertionError("geometry in the file: %r %r" % (d["electrode_centers"].shape, meta))
        return "launch test: one task ran through submit_mea_array.sh + process_campaign.py; files, manifest, env record"

    def e2():
        rc, out = F.launch("array", False)
        if rc == 0 or "already holds detections" not in out:
            raise AssertionError("array over the test task's output not refused without RESUME (rc %d):\n%s" % (rc, out[-600:]))
        rc, out = F.launch("array", False, WALLTIME="01:00:00", RESUME="1")
        calls = F.qsub_calls()
        if rc != 0 or len(calls) != 3:
            raise AssertionError("array launch (rc %d), %d qsub calls:\n%s" % (rc, len(calls), out[-1200:]))
        arr, gate = calls[1], calls[2]
        if arr[1] != "c8_mea" or arr[2] != "0-2%4" or gate[5].endswith("run_sim_reextract_gate.pbs") is False \
                or gate[3] != "depend=afterok:%s" % arr[0] or "PLAN=" not in gate[4]:
            raise AssertionError("calls: %r" % calls[1:])
        d = F.plan_doc()
        if d["counts"]["tasks_done"] != 1 or d["counts"]["tasks_run"] != 3:
            raise AssertionError("resume counts: %r" % d["counts"])
        for (camp, sw), (nt, ni) in F.tasks.items():
            if camp.startswith("campaign_cadex_hhgap") or not os.path.isfile(
                    os.path.join(F.sim_main, camp, sw, "manifest.json")):
                continue
            man = _json(os.path.join(F.out_dir(camp, sw), "mea_manifest.json"))
            if man["total_done"] != nt * ni:
                raise AssertionError("%s/%s: %r" % (camp, sw, man))
        if os.path.isdir(F.out_dir("campaign_cadex_rho1300v7", "sweep_cpu_task0001")):
            raise AssertionError("the excluded task was run")
        return "launch array, RESUME=1: 3 tasks ran as -J 0-2, the gate submitted with depend=afterok:<array>"

    def e3():
        rc, out = F.gate()
        rec_path = os.path.join(F.out_root, PLAN.RECORD_NAME)
        if rc != 0 or "wrote %s" % rec_path not in out or not os.path.isfile(rec_path + ".sha256"):
            raise AssertionError("gate did not pass (rc %d):\n%s" % (rc, out[-1500:]))
        rec = _json(rec_path)
        side = open(rec_path + ".sha256").read().split()[0]
        if side != hashlib.sha256(open(rec_path, "rb").read()).hexdigest():
            raise AssertionError("record sidecar does not match")
        if rec["counts"]["tasks"] != 4 or rec["counts"]["iterations"] != 11 or rec["counts"]["files_read"] != 11 \
                or rec["counts"]["tasks_excluded"] != 4 or rec["geometry"]["n_e"] != 9 \
                or rec["environment"]["numpy"] != np.__version__ or rec["tools"]["label"] != required_label \
                or rec["library"]["sha256"] != _sha(F.library) or len(rec["tasks"]) != 4 \
                or rec["cohort_manifest"]["digest"] != CM.read_manifest(F.cm)["_digest"]:
            raise AssertionError("record content: %r" % {k: rec[k] for k in ("counts", "geometry", "environment", "tools")})
        return "gate PASS: record + sidecar, 4 tasks / 11 iterations / 11 files read, geometry, env, tools, library"

    def g1():
        rc, out = F.gate()
        if rc == 0 or "already exists" not in out:
            raise AssertionError("gate ran over an existing record (rc %d):\n%s" % (rc, out[-600:]))
        os.remove(os.path.join(F.out_root, PLAN.RECORD_NAME))
        os.remove(os.path.join(F.out_root, PLAN.RECORD_NAME + ".sha256"))
        return "gate refuses an existing record"

    od_v9 = F.out_dir("campaign_cadex_rho1300v9", "sweep_cpu_task0000")

    def _expect_fail(needle, label):
        rc, out = F.gate()
        if rc == 0 or "REFUSED" not in out or needle not in out:
            raise AssertionError("%s not caught (rc %d):\n%s" % (label, rc, out[-1200:]))
        if os.path.isfile(os.path.join(F.out_root, PLAN.RECORD_NAME)):
            raise AssertionError("%s: a record was written" % label)

    def g2():
        f = os.path.join(od_v9, "topo_00000", "mea_iter_00001.npz")
        shutil.move(f, f + ".aside")
        try:
            _expect_fail("1 mea_iter_*.npz, raw has 2", "deleted file")
        finally:
            shutil.move(f + ".aside", f)
        return "a deleted mea_iter file: FAIL naming the task and the counts"

    def g3():
        f = os.path.join(od_v9, "mea_env.json")
        shutil.move(f, f + ".aside")
        try:
            _expect_fail("no mea_env.json", "missing env record")
        finally:
            shutil.move(f + ".aside", f)
        return "a missing mea_env.json: FAIL"

    def g4():
        f = os.path.join(od_v9, "topo_00000", "_failures.log")
        open(f, "w").write("iter_00001.npz: ValueError\n")
        mj = os.path.join(od_v9, "mea_manifest.json")
        orig = open(mj).read()
        try:
            # every iteration done: the log is an earlier attempt's -> WARN, record written
            rc, out = F.gate()
            rec_path = os.path.join(F.out_root, PLAN.RECORD_NAME)
            if rc != 0 or "WARN campaign_cadex_rho1300v9/sweep_cpu_task0000: topo_00000: _failures.log from an earlier attempt" not in out \
                    or not os.path.isfile(rec_path):
                raise AssertionError("stale failures log not a warning (rc %d):\n%s" % (rc, out[-800:]))
            rec = _json(rec_path)
            if len(rec.get("warnings", [])) != 1 or rec["warnings"][0]["task"] != "campaign_cadex_rho1300v9/sweep_cpu_task0000":
                raise AssertionError("record warnings: %r" % rec.get("warnings"))
            os.remove(rec_path)
            os.remove(rec_path + ".sha256")
            # an iteration not done in THIS run: FAIL
            d = json.loads(orig)
            d["total_done"] -= 1
            with open(mj, "w") as fh:
                json.dump(d, fh)
            _expect_fail("_failures.log present (an iteration raised)", "failures log")
        finally:
            os.remove(f)
            open(mj, "w").write(orig)
        return "a _failures.log: WARN and recorded when every iteration is done, FAIL when one is not"

    def g5():
        pj = os.path.join(F.work, "plan.json")
        orig = open(pj).read()
        d = json.loads(orig)
        d["geometry"]["pitch_um"] = 61.0
        with open(pj, "w") as fh:
            json.dump(d, fh)
        try:
            _expect_fail("meta_json pitch = 60.0, want 61.0", "pitch")
        finally:
            open(pj, "w").write(orig)
        return "a plan asking pitch 61: FAIL on every file's meta_json"

    def g6():
        mj = os.path.join(od_v9, "mea_manifest.json")
        orig = open(mj).read()
        d = json.loads(orig)
        d["total_done"] -= 1
        with open(mj, "w") as fh:
            json.dump(d, fh)
        try:
            _expect_fail("total_done 1 of total_iters 2", "short manifest")
        finally:
            open(mj, "w").write(orig)
        return "total_done one short: FAIL"

    def g7():
        orig = open(F.library, "rb").read()
        with open(F.library, "ab") as fh:
            fh.write(b"\0")
        try:
            rc, out = F.gate()
            if rc == 0 or "template library" not in out or "no longer hashes" not in out:
                raise AssertionError("changed library not refused (rc %d):\n%s" % (rc, out[-600:]))
        finally:
            open(F.library, "wb").write(orig)
        return "a changed library: REFUSED before any task"

    def g8():
        f = os.path.join(od_v9, "mea_env.json")
        orig = open(f).read()
        d = json.loads(orig)
        d["numpy"] = "0.0.0"
        with open(f, "w") as fh:
            json.dump(d, fh)
        try:
            _expect_fail("numpy differs across tasks", "env mismatch")
        finally:
            open(f, "w").write(orig)
        return "two numpy versions across tasks: FAIL environment"

    def g9():
        f = os.path.join(F.tools, "mea_probe.py")
        orig = open(f).read()
        open(f, "a").write("# edited\n")
        try:
            rc, out = F.gate()
            if rc == 0 or "changed since the plan" not in out or "mea_probe.py" not in out:
                raise AssertionError("changed tools not refused (rc %d):\n%s" % (rc, out[-600:]))
        finally:
            open(f, "w").write(orig)
        return "tools changed since the plan: REFUSED"

    def g10():
        rc, out = F.gate()
        if rc != 0 or "wrote" not in out:
            raise AssertionError("gate does not pass after the repairs (rc %d):\n%s" % (rc, out[-800:]))
        return "after every repair the gate passes again"

    def r1():
        for n in (PLAN.RECORD_NAME, PLAN.RECORD_NAME + ".sha256"):
            os.remove(os.path.join(F.out_root, n))
        os.remove(os.path.join(od_v9, "mea_manifest.json"))     # a walltime kill leaves no manifest
        rc, out = F.launch("array", False, WALLTIME="00:30:00", RESUME="1")
        calls = F.qsub_calls()
        d = F.plan_doc()
        if rc != 0 or d["counts"]["tasks_done"] != 3 or d["counts"]["tasks_run"] != 1 \
                or len(calls) != 5 or calls[3][1] != "c8_mea" or calls[3][2] != "":
            raise AssertionError("resume launch (rc %d) %r calls %d:\n%s" % (rc, d["counts"], len(calls), out[-800:]))
        if not os.path.isfile(os.path.join(od_v9, "mea_manifest.json")):
            raise AssertionError("the killed task was not re-run")
        rc, out = F.gate()
        rec = _json(os.path.join(F.out_root, PLAN.RECORD_NAME))
        if rc != 0 or rec["counts"]["tasks"] != 4 or rec["counts"]["files_read"] != 11:
            raise AssertionError("gate after resume (rc %d): %r\n%s" % (rc, rec.get("counts"), out[-600:]))
        return "resume: 3 kept, the killed task re-run as a plain job, the gate records all 4"

    def r2():
        rc, out = F.plan(resume=True)
        if rc == 0 or "finished" not in out:
            raise AssertionError("a finished root not refused (rc %d):\n%s" % (rc, out[-600:]))
        return "the plan refuses a root that holds a record"

    for nm, fn in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5),
                   ("L1", l1), ("L2", l2), ("L3", l3), ("E1", e1), ("E2", e2), ("E3", e3),
                   ("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4), ("G5", g5), ("G6", g6),
                   ("G7", g7), ("G8", g8), ("G9", g9), ("G10", g10), ("R1", r1), ("R2", r2)):
        check(nm, fn)
    if not a.keep:
        shutil.rmtree(F.root, ignore_errors=True)
    print("-" * 70)
    if _FAIL:
        print("FAILED %d of %d" % (len(_FAIL), len(_PASS) + len(_FAIL)))
        return 1
    print("ALL %d CHECKS PASSED" % len(_PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
