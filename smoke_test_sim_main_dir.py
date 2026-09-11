#!/usr/bin/env python3
"""
smoke_test_sim_main_dir.py

On 2026-09-10 all 51 export jobs died in 10 s with

    ValueError: manifest.json param_bounds is (37, 2) but the registry is (36, 2)

because submit_sbi_export.sh defaulted

    SIM_MAIN_DIR="${SIM_MAIN_DIR:-$HOME/repos/.../Phenomenological_finalv1}"

and a stale SIM_MAIN_DIR from an earlier session (the rho1300 tree, 36-D) had
propagated from the launching shell into every job's -v payload. The login-node
dry run passed --sim_dir explicitly, so it never exercised the env path.

There is more than one simulator tree on this cluster and they differ in
registry width, so no default can be right for all of them. This test pins the
four properties of the fix.

Run:
    python3 smoke_test_sim_main_dir.py
    SMOKE_VERBOSE=1 python3 smoke_test_sim_main_dir.py

Expect: ALL 8 CHECKS PASSED
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import traceback

VERBOSE = bool(os.environ.get("SMOKE_VERBOSE"))
PASSED, FAILED = [], []

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "submit_sbi_export.sh")


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


def code_lines():
    """The script with comment lines stripped: the fix documents the old
    default verbatim to explain it, and that must not read as still shipping."""
    return "\n".join(l for l in open(SCRIPT).read().splitlines()
                     if not l.lstrip().startswith("#"))


def run(snippet, env=None, cwd=None):
    e = dict(os.environ)
    e.pop("SIM_MAIN_DIR", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True,
                          text=True, env=e, cwd=cwd)


# The two constructs under test, isolated from the rest of the script.
REQUIRED = 'export SIM_MAIN_DIR="${SIM_MAIN_DIR:?SIM_MAIN_DIR not set.}"; echo "OK:$SIM_MAIN_DIR"'

WARNER = r'''
CAMPAIGN="$1"; SIM_MAIN_DIR="$2"
_SIM_EXPECT="$(cd "$(dirname "$(dirname "${CAMPAIGN}")")" 2>/dev/null && pwd -P || true)"
_SIM_ACTUAL="$(cd "${SIM_MAIN_DIR}" 2>/dev/null && pwd -P || true)"
if [ -n "${_SIM_EXPECT}" ] && [ -n "${_SIM_ACTUAL}" ] \
   && [ "${_SIM_EXPECT}" != "${_SIM_ACTUAL}" ]; then
    echo "WARN"
else
    echo "QUIET"
fi
'''


def main():
    root = tempfile.mkdtemp(prefix="sim_main_dir_")
    try:
        # Mirror the real layout: the rho1300 tree, with Giulia_Astro NESTED
        # inside it. That nesting is why a naive "is CAMPAIGN under
        # SIM_MAIN_DIR?" prefix test would NOT have caught the 2026-09-10
        # defect -- the campaign was under both trees.
        main_tree = os.path.join(root, "Phenomenological", "Main")
        giulia = os.path.join(main_tree, "Giulia_Astro")
        task = os.path.join(giulia, "campaign_cadex_hhgap_v1", "sweep_intel_task0000")
        os.makedirs(task)

        def T1():
            r = run(REQUIRED)
            assert r.returncode != 0, "unset SIM_MAIN_DIR must fail, not default"
            assert "SIM_MAIN_DIR not set" in r.stderr, r.stderr[:200]
        check("T1 unset SIM_MAIN_DIR exits non-zero instead of defaulting", T1)

        def T2():
            r = run(REQUIRED, env={"SIM_MAIN_DIR": giulia})
            assert r.returncode == 0, r.stderr[:200]
            assert r.stdout.strip() == "OK:" + giulia, r.stdout
        check("T2 a supplied SIM_MAIN_DIR passes through unchanged", T2)

        def T3():
            r = run(REQUIRED, env={"SIM_MAIN_DIR": ""})
            assert r.returncode != 0, "empty must be treated as unset"
        check("T3 an EMPTY SIM_MAIN_DIR also fails (:? covers unset and empty)", T3)

        def T4():
            r = subprocess.run(["bash", "-c", WARNER, "bash", task, giulia],
                               capture_output=True, text=True)
            assert r.stdout.strip() == "QUIET", r.stdout
        check("T4 no warning when SIM_MAIN_DIR is the campaign's own tree", T4)

        def T5():
            r = subprocess.run(["bash", "-c", WARNER, "bash", task, main_tree],
                               capture_output=True, text=True)
            assert r.stdout.strip() == "WARN", (
                "the exact 2026-09-10 case must warn: campaign under "
                "Giulia_Astro, SIM_MAIN_DIR pointing at its parent Main")
        check("T5 the real defect (SIM_MAIN_DIR = the PARENT tree) warns", T5)

        def T6():
            code = code_lines()
            assert "Phenomenological_finalv1}" not in code, \
                "the silent fallback default is still executable"
            assert "SIM_MAIN_DIR:?" in code, "SIM_MAIN_DIR is not required"
        check("T6 the stale-tree default is gone and the variable is required", T6)

        def T7():
            code = code_lines()
            assert '--sim_dir     "${SIM_MAIN_DIR}"' in code, \
                "--sim_dir is not passed explicitly to example_export.py"
            assert '[sbi] sim main   : ${SIM_MAIN_DIR}' in code, \
                "SIM_MAIN_DIR is not printed in the banner"
        check("T7 --sim_dir is explicit and SIM_MAIN_DIR appears in the banner", T7)

        def T8():
            r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[:300]
        check("T8 the patched script still parses (bash -n)", T8)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
