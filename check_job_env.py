#!/usr/bin/env python3
"""
check_job_env.py -- which job scripts will die silently on a compute node?

Run:
    python3 check_job_env.py                  # scan the current tree
    python3 check_job_env.py ~/repos          # scan somewhere else
    python3 check_job_env.py --quiet          # only the broken ones

THE FAILURE THIS FINDS

`conda activate` is a SHELL FUNCTION, created by conda's init hook. An
interactive login shell sources that hook from .bashrc; a PBS or SLURM batch
shell does NOT. In a batch job the function therefore does not exist, the
bare `conda` binary handles the call instead, and it refuses:

    CondaError: Run 'conda init' before 'conda activate'

exiting 1. `source activate` (the deprecated form) fails the same way on
recent conda. When the script runs under `set -e` -- as job scripts should --
this kills the job BEFORE its first echo, so the symptom is:

    a job that finishes instantly, Exit_status=1, and EMPTY .o and .e files.

No error message anywhere. The script that fails this way looks identical in
an interactive shell, where it works, which is why it survives review.

THE FIX, AND WHY IT IS NOT SOURCED FROM A SHARED FILE

Either of these defines the function before it is called:

    eval "$(conda shell.bash hook)"
    source "$(conda info --base)/etc/profile.d/conda.sh"

with `set +u` around it, because conda's own scripts read unset variables.
This script treats either as SAFE.

The block has to be INLINE in every job script rather than sourced from a
shared helper: under PBS the scheduler runs a COPY of the script from its
spool directory, so $0 does not point at the repo and a $0-relative `source`
of a helper file fails. Duplication is the price of a script that runs
wherever the scheduler puts it.

Best practice beyond activation: resolve the interpreter explicitly
(PYBIN, then PATH, then $HOME/.conda/envs/<env>/bin/python) and call it by
absolute path, then verify its imports. Activation putting the WRONG python
on PATH is silent; an absolute path plus an import check is not.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

EXTS = (".sh", ".pbs", ".slurm")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}

# Defines the shell function. Either form counts.
HOOK = re.compile(r"conda\s+shell\.bash\s+hook|profile\.d/conda\.sh")
# Calls it. Anchored to a statement start so comments and echoes do not match.
ACT = re.compile(r"^\s*(source\s+activate|conda\s+activate|\.\s+activate)\b",
                 re.M)
# An explicit interpreter path makes activation non-load-bearing.
PYBIN = re.compile(r"envs/[^/\s\"']+/bin/python|\$\{?PYBIN")


def scan(root: str):
    rows = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if not f.endswith(EXTS):
                continue
            p = os.path.join(base, f)
            try:
                s = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if not ACT.search(s):
                continue
            has_hook = bool(HOOK.search(s))
            has_bin = bool(PYBIN.search(s))
            strict = ("set -e" in s) or ("set -euo" in s)
            if has_hook or has_bin:
                status = "SAFE"
            elif strict:
                status = "BROKEN"          # dies silently under set -e
            else:
                status = "RISKY"           # continues with the wrong python
            rows.append((status, os.path.relpath(p, root), has_hook, has_bin,
                         strict))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="find job scripts that activate a conda env unsafely")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--quiet", action="store_true",
                    help="list only BROKEN and RISKY")
    args = ap.parse_args(argv)

    rows = scan(args.root)
    if not rows:
        print("no job script activates a conda environment under %s"
              % os.path.abspath(args.root))
        return 0

    order = {"BROKEN": 0, "RISKY": 1, "SAFE": 2}
    print("%-7s %-4s %-4s %-6s %s"
          % ("status", "hook", "pybin", "set -e", "file"))
    print("-" * 78)
    for st, p, hook, pybin, strict in sorted(rows,
                                             key=lambda r: (order[r[0]], r[1])):
        if args.quiet and st == "SAFE":
            continue
        print("%-7s %-4s %-4s %-6s %s"
              % (st, "yes" if hook else "-", "yes" if pybin else "-",
                 "yes" if strict else "-", p))

    n = {k: sum(1 for r in rows if r[0] == k) for k in order}
    print("")
    print("%d script(s) activate an env: %d SAFE, %d RISKY, %d BROKEN"
          % (len(rows), n["SAFE"], n["RISKY"], n["BROKEN"]))
    if n["BROKEN"]:
        print("")
        print("BROKEN = activates with no hook and no explicit interpreter,")
        print("         under set -e. On a compute node this exits 1 before")
        print("         the first echo, leaving EMPTY .o and .e files.")
    if n["RISKY"]:
        print("")
        print("RISKY  = same activation, but no set -e, so the job CONTINUES")
        print("         with whatever python is on PATH. Worse than BROKEN:")
        print("         it produces output, from the wrong environment.")
    return 1 if n["BROKEN"] else 0


if __name__ == "__main__":
    sys.exit(main())
