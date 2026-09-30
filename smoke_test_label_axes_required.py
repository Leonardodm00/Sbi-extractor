#!/usr/bin/env python3
"""
smoke_test_label_axes_required.py

Decision 2026-09-24 (Stage C): LABEL_AXES is passed explicitly for every
campaign set. Before it, all three entry points had a silent fallback:

    launch_sweep_exports.sh   unset LABEL_AXES -> "the worker's default"
    submit_sbi_export.sh      LABEL_AXES:-<artifacts>/label_axes.json, i.e.
                              the r2 WEIBULL freeze, right for one campaign
                              family only (under a flat set it trips the NaN
                              guard in assemble_theta_A on every job)
    example_export.py         no --label_axes -> the legacy 4-axis block

Now each refuses without it, and 'none' still selects the legacy block, but
only by name. This suite pins that, running the REAL code of each entry point:
the launcher end to end in DRYRUN mode on a throwaway fixture, the worker's
own LABEL_AXES block cut out of the script and run under its own shell
options, and the exporter's argument check in a child interpreter (torch
stubbed there when absent, so nothing here needs torch, pyarrow, a DSN tree,
a simulator tree or a cluster).

Run:
    python3 smoke_test_label_axes_required.py
    SMOKE_VERBOSE=1 python3 smoke_test_label_axes_required.py

Expect: ALL 12 CHECKS PASSED

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

VERBOSE = bool(os.environ.get("SMOKE_VERBOSE"))
PASSED, FAILED = [], []

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "launch_sweep_exports.sh")
WORKER = os.path.join(HERE, "submit_sbi_export.sh")

# The exporter's refusal names the flag; the tests key on this phrase.
EXPORTER_REFUSAL = "--label_axes is required for --mode campaign"


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("  PASS  %s" % name)
    except Exception as exc:
        FAILED.append((name, exc))
        print("  FAIL  %s : %s" % (name, exc))
        if VERBOSE:
            traceback.print_exc()


def code_lines(path):
    """The script with comment lines stripped, so that a comment quoting the
    old default verbatim cannot read as the default still shipping."""
    with open(path, "r", encoding="ascii") as fh:
        return "\n".join(l for l in fh.read().splitlines()
                         if not l.lstrip().startswith("#"))


# --------------------------------------------------------------------------- #
# fixture: the smallest tree the launcher enumerates as one exportable task
# --------------------------------------------------------------------------- #
def make_fixture(root):
    f = {
        "ckpt": os.path.join(root, "ckpt.pt"),
        "mea": os.path.join(root, "mea"),
        "sim": os.path.join(root, "sim"),
        "sim_main": os.path.join(root, "sim_main"),
        "out": os.path.join(root, "out"),
        "axes": os.path.join(root, "label_axes_fixture.json"),
    }
    task = os.path.join("campaign_fixture", "sweep_cpu_task0000", "topo_00000")
    os.makedirs(os.path.join(f["mea"], task))
    os.makedirs(os.path.join(f["sim"], task))
    os.makedirs(f["sim_main"])
    open(os.path.join(f["mea"], task, "mea_iter_00000.npz"), "wb").close()
    open(os.path.join(f["sim"], task, "iter_00000.npz"), "wb").close()
    with open(os.path.join(f["sim"], "campaign_fixture", "sweep_cpu_task0000",
                           "job_args.json"), "w") as fh:
        json.dump({"simtime": 200.0}, fh)
    open(f["ckpt"], "wb").close()
    with open(f["axes"], "w") as fh:
        json.dump({"topology_axes": ["p0_conn"], "excluded_axes": {}}, fh)
    return f


def run_launcher(f, label_axes):
    """label_axes: None = unset; anything else is exported as given."""
    env = dict(os.environ)
    for v in ("LABEL_AXES", "SBI_HPC_DIR", "MAX_RECORDS", "SIMTIME",
              "TRIM_HEAD_S", "ENV_NAME", "GLOB", "NOCOUNT"):
        env.pop(v, None)
    env["DRYRUN"] = "1"
    env["SIM_MAIN_DIR"] = f["sim_main"]
    if label_axes is not None:
        env["LABEL_AXES"] = label_axes
    return subprocess.run(["bash", LAUNCHER, f["ckpt"], f["mea"], f["sim"],
                           f["out"]], capture_output=True, text=True, env=env,
                          timeout=120)


def worker_block():
    """The worker's LABEL_AXES handling, exactly as shipped: from the line that
    tests for an unset value to the fi that closes the file branch."""
    with open(WORKER, "r", encoding="ascii") as fh:
        lines = fh.read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith('if [ -z "${LABEL_AXES:-}" ]'))
        add = next(i for i in range(start, len(lines))
                   if '--label_axes ${LABEL_AXES}"' in lines[i])
        end = next(i for i in range(add, len(lines)) if lines[i] == "fi")
    except StopIteration:
        raise AssertionError(
            "submit_sbi_export.sh has no 'if [ -z \"${LABEL_AXES:-}\" ]' block "
            "ending in the --label_axes append: LABEL_AXES is not required")
    return "\n".join(lines[start:end + 1])


def run_worker_block(label_axes):
    block = worker_block()
    snippet = ('set -euo pipefail\nEXTRA=""\nARTIFACTS_DIR=/fixture/artifacts\n'
               + block + '\necho "EXTRA=[${EXTRA}]"\n')
    env = dict(os.environ)
    env.pop("LABEL_AXES", None)
    if label_axes is not None:
        env["LABEL_AXES"] = label_axes
    return subprocess.run(["bash", "-c", snippet], capture_output=True,
                          text=True, env=env, timeout=60)


_EXPORTER_CHILD = r'''
import sys, types
try:
    import torch  # noqa: F401
except ImportError:
    t = types.ModuleType("torch")
    t.Tensor = object
    t.nn = types.ModuleType("torch.nn")
    t.nn.functional = types.ModuleType("torch.nn.functional")
    t.no_grad = lambda *a, **k: (lambda f: f)
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = t.nn
    sys.modules["torch.nn.functional"] = t.nn.functional
import example_export as EX
sys.argv = ["example_export.py"] + sys.argv[1:]
EX.main()
'''


def run_exporter(*argv):
    """main() in a child; --sim_dir names a directory that does not exist, so
    a run that gets PAST the argument checks dies at the registry load --
    which is how the tests tell 'refused by the flag check' from 'went on'."""
    return subprocess.run([sys.executable, "-c", _EXPORTER_CHILD] + list(argv),
                          cwd=HERE, capture_output=True, text=True, timeout=300)


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="label_axes_required_")
    try:
        f = make_fixture(root)
        missing = os.path.join(root, "no_such_label_axes.json")

        # ---- the launcher, end to end (DRYRUN) ---------------------------
        def L1():
            r = run_launcher(f, None)
            assert r.returncode == 9, "unset must exit 9, got %d\n%s" % (
                r.returncode, r.stderr[-400:])
            assert "export LABEL_AXES" in r.stderr, r.stderr[-400:]
            assert "qsub" not in r.stdout and "DRY " not in r.stdout, (
                "the refusal must come before the task loop:\n" + r.stdout[-400:])
        check("L1 launcher: unset LABEL_AXES refuses (exit 9) before any task", L1)

        def L2():
            r = run_launcher(f, "")
            assert r.returncode == 9, "empty must exit 9, got %d" % r.returncode
            assert "qsub" not in r.stdout
        check("L2 launcher: an EMPTY LABEL_AXES is refused like an unset one", L2)

        def L3():
            r = run_launcher(f, f["axes"])
            assert r.returncode == 0, r.stderr[-400:]
            assert ("LABEL_AXES=%s" % f["axes"]) in r.stdout, (
                "the file is not forwarded to the job:\n" + r.stdout[-600:])
            assert "submitted        : 1" in r.stdout, r.stdout[-600:]
            assert ("# label axes : %s" % f["axes"]) in r.stdout
        check("L3 launcher: a supplied file is forwarded on the qsub line", L3)

        def L4():
            r = run_launcher(f, "none")
            assert r.returncode == 0, r.stderr[-400:]
            assert "LABEL_AXES=none" in r.stdout, r.stdout[-600:]
        check("L4 launcher: 'none' still passes, by name", L4)

        def L5():
            r = run_launcher(f, missing)
            assert r.returncode == 7, "a missing file must exit 7, got %d" % (
                r.returncode)
            assert "missing file" in r.stderr, r.stderr[-400:]
        check("L5 launcher: a missing file is still refused (exit 7)", L5)

        # ---- the worker's own block, as shipped --------------------------
        def W1():
            r = run_worker_block(None)
            assert r.returncode == 9, "unset must exit 9, got %d\n%s" % (
                r.returncode, r.stderr[-400:])
            assert "LABEL_AXES=... is required" in r.stderr, r.stderr[-400:]
            assert "EXTRA=[" not in r.stdout
        check("W1 worker: unset LABEL_AXES exits 9 instead of defaulting", W1)

        def W2():
            r = run_worker_block(f["axes"])
            assert r.returncode == 0, r.stderr[-400:]
            assert r.stdout.strip() == "EXTRA=[ --label_axes %s]" % f["axes"], (
                r.stdout)
            r = run_worker_block(missing)
            assert r.returncode == 8, "missing file must exit 8, got %d" % (
                r.returncode)
        check("W2 worker: a file is passed as --label_axes; a missing one exits 8", W2)

        def W3():
            r = run_worker_block("none")
            assert r.returncode == 0, r.stderr[-400:]
            assert r.stdout.strip() == "EXTRA=[ --label_axes none]", r.stdout
            assert "LEGACY 4-axis" in r.stderr
        check("W3 worker: 'none' warns and is passed ON as --label_axes none", W3)

        def W4():
            code = code_lines(WORKER)
            assert "LABEL_AXES:-${ARTIFACTS_DIR" not in code, (
                "the r2-weibull default is still executable")
            assert 'LABEL_AXES="${LABEL_AXES:-' not in code, (
                "LABEL_AXES is still given a default")
            for script in (WORKER, LAUNCHER):
                r = subprocess.run(["bash", "-n", script], capture_output=True,
                                   text=True)
                assert r.returncode == 0, "%s: %s" % (script, r.stderr[:300])
        check("W4 no default survives in the worker; both scripts parse (bash -n)", W4)

        # ---- the exporter's own argument check ---------------------------
        dummy = os.path.join(root, "no_such_sim_tree")

        def X1():
            r = run_exporter("--mode", "campaign", "--out", os.path.join(root, "x"),
                             "--sim_dir", dummy, "--checkpoint", f["ckpt"],
                             "--campaign", f["sim"], "--mea_out", f["mea"])
            assert r.returncode == 2, "argparse refusal must exit 2, got %d\n%s" % (
                r.returncode, r.stderr[-400:])
            assert EXPORTER_REFUSAL in r.stderr, r.stderr[-400:]
        check("X1 exporter: --mode campaign without --label_axes is refused", X1)

        def X2():
            for val in ("none", f["axes"]):
                r = run_exporter("--mode", "campaign", "--out",
                                 os.path.join(root, "x"), "--sim_dir", dummy,
                                 "--label_axes", val)
                assert EXPORTER_REFUSAL not in r.stderr, (
                    "--label_axes %s was refused: %s" % (val, r.stderr[-300:]))
                assert "[1/5] loading the run_args registry" in r.stdout, (
                    "with --label_axes %s the run did not reach the registry "
                    "load:\n%s" % (val, (r.stdout + r.stderr)[-400:]))
        check("X2 exporter: 'none' and a file both get past the check", X2)

        def X3():
            r = run_exporter("--mode", "synthetic", "--out",
                             os.path.join(root, "x"), "--sim_dir", dummy)
            assert EXPORTER_REFUSAL not in r.stderr, r.stderr[-300:]
            assert "[1/5] loading the run_args registry" in r.stdout, (
                (r.stdout + r.stderr)[-400:])
        check("X3 exporter: synthetic mode does not need --label_axes", X3)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
