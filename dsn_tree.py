"""
dsn_tree.py -- where the Deep Summary Network lives, seen from Sbi-extractor.

Migration step 3 (2026-09-19). The DSN is no longer a separate repository:
it is the directory hpc/dsn of Simulation-Based-Inference (a byte-identical
mirror of the retired repo's Main/, see hpc/dsn/README.md there). This module
is the one place in Sbi-extractor that knows that, so nothing else here has
to spell the path.

Resolution, highest precedence first:

  1. an explicit argument (a test pointing at a stub, a deliberate A/B);
  2. the environment variable SBI_HPC_DIR -- the SBI repo's hpc/ directory;
     env.sh defaults it to artifacts/sbi_hpc, a symlink recreated per
     machine (artifacts/README.md);
  3. artifacts/sbi_hpc next to this file, for a python entry point run
     without sourcing env.sh.

The DSN tree is then <SBI_HPC_DIR>/dsn. DSN_MAIN_DIR is not consulted by
anything that imports this module (dsn_frozen.py and example_export.py still
read it until migration step 4 retires it).

Pure ASCII, LF only.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))              # this repo
DEFAULT_SBI_HPC = os.path.join(ROOT, "artifacts", "sbi_hpc")   # symlink

# What the extractor and the exporter import from the tree.
SENTINELS = ("generate_burst_data.py", "config.py", "make_mea_specs.py",
             "backbone.py", "checkpoint.py")


class DSNTreeMissing(RuntimeError):
    """The DSN tree could not be resolved, or lacks a module we import."""


def sbi_hpc_dir(explicit=None):
    """The SBI repo's hpc/ directory, by the precedence in the module note."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("SBI_HPC_DIR")
    if env:
        return os.path.abspath(env)
    return DEFAULT_SBI_HPC


def explain(d=None):
    """One sentence telling the reader what to set. Used in error messages."""
    return ("the DSN tree is <SBI_HPC_DIR>/dsn; SBI_HPC_DIR is %r here "
            "(env.sh defaults it to artifacts/sbi_hpc -- recreate that symlink "
            "with  ln -s ~/SBI/hpc artifacts/sbi_hpc  on davinci, see "
            "artifacts/README.md)" % (d if d is not None else sbi_hpc_dir(),))


def dsn_dir(explicit_hpc=None, require=True):
    """Absolute path of the DSN tree, <SBI_HPC_DIR>/dsn.

    require=True raises DSNTreeMissing naming the absent sentinel files;
    require=False returns the path regardless (the caller checks).
    """
    hpc = sbi_hpc_dir(explicit_hpc)
    d = os.path.join(hpc, "dsn")
    if require:
        if not os.path.isdir(d):
            raise DSNTreeMissing("DSN tree %r does not exist: %s"
                                 % (d, explain(hpc)))
        missing = [s for s in SENTINELS
                   if not os.path.isfile(os.path.join(d, s))]
        if missing:
            raise DSNTreeMissing(
                "DSN tree %r is missing %s: %s"
                % (d, ", ".join(missing), explain(hpc)))
    return d


def add_dsn_to_path(explicit_hpc=None, require=True):
    """dsn_dir(), then put the tree first on sys.path. Returns the path.

    With require=False an unresolvable tree is NOT an error here: the path
    is simply not added, and the caller's own import raises later. That is
    what lets a module resolve at import time without failing import for
    users who never reach the DSN-dependent branch.
    """
    d = dsn_dir(explicit_hpc, require=require)
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
    return d


def status(explicit_hpc=None):
    """Empty string if the tree resolves, else the reason."""
    try:
        dsn_dir(explicit_hpc, require=True)
    except DSNTreeMissing as exc:
        return str(exc)
    return ""


if __name__ == "__main__":
    s = status()
    print("SBI_HPC_DIR : %s" % sbi_hpc_dir())
    print("DSN tree    : %s" % dsn_dir(require=False))
    print("resolves    : %s" % ("yes" if s == "" else "NO -- " + s))
    sys.exit(0 if s == "" else 1)
