#!/usr/bin/env python3
"""
smoke_test_mea_presence_check.py

submit_sbi_export.sh's "has process_campaign.py been run?" guard used to be

    ls "${MEA_OUT}"/topo_*/mea_iter_*.npz >/dev/null 2>&1

Bash expands that glob in-process, then execve() on /bin/ls fails with E2BIG
once the argument vector exceeds ARG_MAX. A sweep task with ~73,000
mea_iter_*.npz files therefore reported "no topo_*/mea_iter_*.npz under ..."
and exited 4 -- indistinguishable from an unprocessed campaign, because the
2>&1 swallowed "Argument list too long". Small tasks passed, so the failure
hit only the largest tasks in an array.

This test builds a fixture big enough to reproduce the E2BIG, then checks that
the replacement (find -print -quit) gets it right, without regressing the
ordinary small-directory and genuinely-absent cases.

Run:
    python3 smoke_test_mea_presence_check.py
    SMOKE_VERBOSE=1 python3 smoke_test_mea_presence_check.py

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

# The two constructs, isolated. OLD is what shipped; NEW is the replacement.
OLD = 'ls "$1"/topo_*/mea_iter_*.npz >/dev/null 2>&1 && echo FOUND || echo MISSING'
NEW = ('[ -n "$(find "$1" -mindepth 2 -maxdepth 2 -path \'*/topo_*/mea_iter_*.npz\''
       ' -print -quit 2>/dev/null)" ] && echo FOUND || echo MISSING')


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


def run(snippet, path):
    out = subprocess.run(["bash", "-c", snippet, "bash", path],
                         capture_output=True, text=True)
    return out.stdout.strip()


def make_task(root, n_topo, per_topo, pad=0):
    """A sweep-task directory. `pad` lengthens the name to inflate path length,
    so ARG_MAX can be exceeded with far fewer files than 73,000."""
    name = "sweep_cfd_task0002" + ("_" + "x" * pad if pad else "")
    task = os.path.join(root, name)
    for t in range(n_topo):
        d = os.path.join(task, "topo_%05d" % t)
        os.makedirs(d, exist_ok=True)
        for i in range(per_topo):
            open(os.path.join(d, "mea_iter_%05d.npz" % i), "wb").close()
    return task


def main():
    root = tempfile.mkdtemp(prefix="mea_presence_")
    try:
        arg_max = int(subprocess.run(["getconf", "ARG_MAX"], capture_output=True,
                                     text=True).stdout.strip() or 2097152)
        print("  ARG_MAX = %d" % arg_max)

        small = make_task(os.path.join(root, "small"), n_topo=2, per_topo=3)

        # Size the big fixture from ARG_MAX rather than hardcoding a count, so
        # this reproduces the bug on machines with a different limit.
        pad = 180
        approx_path = len(os.path.join(root, "big")) + len(
            "/sweep_cfd_task0002_" + "x" * pad + "/topo_00000/mea_iter_00000.npz")
        need = int(arg_max / approx_path) + 400
        per_topo = 500
        big = make_task(os.path.join(root, "big"),
                        n_topo=(need // per_topo) + 1, per_topo=per_topo, pad=pad)
        n_big = sum(len(f) for _r, _d, f in os.walk(big))
        print("  big fixture: %d files, ~%d bytes of argv" % (n_big, n_big * approx_path))

        empty = os.path.join(root, "empty", "sweep_cpu_task0000")
        os.makedirs(empty, exist_ok=True)

        notopo = os.path.join(root, "notopo", "sweep_cpu_task0001", "topo_00000")
        os.makedirs(notopo, exist_ok=True)
        open(os.path.join(notopo, "something_else.npz"), "wb").close()

        def T1():
            assert run(OLD, small) == "FOUND", "old construct broken on a small dir"
        check("T1 the OLD construct works on a small task (why this went unnoticed)", T1)

        def T2():
            got = run(OLD, big)
            assert got == "MISSING", (
                "expected the old construct to report MISSING via E2BIG, got %r. "
                "ARG_MAX may not have been exceeded on this machine; the bug is "
                "real regardless, but this fixture did not reproduce it." % got)
        check("T2 the OLD construct reports MISSING on a big task -- the bug", T2)

        def T3():
            direct = subprocess.run(
                ["bash", "-c", 'ls "$1"/topo_*/mea_iter_*.npz', "bash", big],
                capture_output=True, text=True)
            assert direct.returncode != 0, "expected a non-zero exit"
            assert "Argument list too long" in direct.stderr, direct.stderr[:200]
        check("T3 the underlying error really is 'Argument list too long'", T3)

        def T4():
            assert run(NEW, big) == "FOUND", "the fix must see the big task"
        check("T4 the NEW construct finds the big task", T4)

        def T5():
            assert run(NEW, small) == "FOUND"
        check("T5 the NEW construct still finds a small task", T5)

        def T6():
            assert run(NEW, empty) == "MISSING"
            assert run(OLD, empty) == "MISSING"
        check("T6 a genuinely empty task is still reported MISSING", T6)

        def T7():
            assert run(NEW, notopo) == "MISSING", (
                "topo_* present but no mea_iter_*.npz must NOT count as found")
        check("T7 topo_* without mea_iter_*.npz does not count as found", T7)

        def T8():
            # Comment lines are stripped first: the fix's own NOTE quotes the
            # old construct verbatim to explain it, and documenting a defect
            # must not look like still shipping it.
            code = "\n".join(l for l in open(SCRIPT).read().splitlines()
                             if not l.lstrip().startswith("#"))
            assert 'ls "${MEA_OUT}"/topo_*/mea_iter_*.npz' not in code, \
                "the E2BIG construct is still executable in submit_sbi_export.sh"
            assert "-print -quit" in code, "the find-based check is not present"
        check("T8 submit_sbi_export.sh no longer builds an argument vector", T8)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
