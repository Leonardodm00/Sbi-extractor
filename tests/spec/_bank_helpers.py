"""Shared helpers for the spec-derived tests of Blocks 8, 9 and 10 (tester-written).

Nothing here computes an expected value with the code under test: the fixture
writer (export_windows/bank_plan_fixture.py, a test helper) only builds the
input files; expected values are computed in the tests from those files
(numpy, hashlib, json) or by hand from SPEC.md.

Run with:  export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc && source env.sh
           python3 -m pytest tests/spec -q
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORT = os.path.join(REPO, "export_windows")
for _p in (EXPORT, REPO, os.path.join(REPO, "sim_reextract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PARENT_OF_BLOCK10 = "ae71f33"      # the contract commit: parent of block 10's commit 6fd20a6
BASE = "4955e8d4bc0703e5565e1e27a98a1538aaef2f31"   # before wave W3 (REAL_VERSION 1)


def sbi_hpc_dir():
    d = os.environ.get("SBI_HPC_DIR", "")
    if not os.path.isfile(os.path.join(d, "joint", "stage1", "latent_bank.py")):
        raise AssertionError("SBI_HPC_DIR must name a Simulation-Based-Inference hpc/ "
                             "checkout holding joint/stage1/latent_bank.py; got %r" % d)
    return d


def latent_bank():
    """The stage-1 shard contract (the consumer of the banks; not under test)."""
    d = os.path.join(sbi_hpc_dir(), "joint", "stage1")
    if d not in sys.path:
        sys.path.insert(0, d)
    import latent_bank as lb       # noqa: E402
    return lb


def canonical_json(obj):
    """The project's canonical JSON (the stage-1 contract's definition)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def run_py(script, *args, env=None, cwd=None, timeout=600):
    """Run a script with this interpreter; (returncode, stdout, stderr)."""
    e = os.environ.copy()
    e.update(env or {})
    r = subprocess.run([sys.executable, script] + [str(a) for a in args],
                       capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def lines_of(out):
    return [ln for ln in out.splitlines() if ln.strip()]


def last_line(out):
    ls = lines_of(out)
    return ls[-1] if ls else ""


def extract_commit(commit, dest, repo=REPO, paths=()):
    """`git archive <commit> [paths]` unpacked under dest (read-only for the repo);
    None when the history is not available (shallow clone)."""
    p = subprocess.run(["git", "-C", repo, "archive", "--format=tar", commit] + list(paths),
                       capture_output=True)
    if p.returncode != 0:
        return None
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        tf.extractall(dest, filter="data")
    return dest


# ---------------------------------------------------------------- the fixture

def fixture_module():
    import bank_plan_fixture as FX     # noqa: E402  (test helper: writes inputs only)
    return FX


def write_fixture(root, **kw):
    return fixture_module().write_fixture(root, **kw)


def write_profile(fx, path, drop_new_keys=False, **overrides):
    """The fixture's profile; drop_new_keys removes the two keys of Blocks 9 and 10
    (for the code of commits that predate them)."""
    FX = fixture_module()
    FX.write_profile(fx, path, **overrides)
    if drop_new_keys:
        with open(path, "r", encoding="ascii") as fh:
            prof = json.load(fh)
        prof.pop("cohort_map", None)
        prof.pop("drop_partial_topologies", None)
        with open(path, "w", encoding="ascii") as fh:
            json.dump(prof, fh, indent=2, sort_keys=True)
    return path


def edit_json(path, **changes):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    for k, v in changes.items():
        if v is _DELETE:
            doc.pop(k, None)
        else:
            doc[k] = v
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)


_DELETE = object()
DELETE = _DELETE


def raw_topo_dir(fx, n_task, k):
    return os.path.join(fx["tasks"][n_task]["campaign_dir"], "topo_%05d" % k)


def det_topo_dir(fx, n_task, k):
    return os.path.join(fx["tasks"][n_task]["out_dir"], "topo_%05d" % k)


def detection_iters(fx, n_task):
    """{k: sorted i} read from the detection tree on disk (independent scan)."""
    out = {}
    od = fx["tasks"][n_task]["out_dir"]
    for name in sorted(os.listdir(od)):
        if not name.startswith("topo_"):
            continue
        k = int(name[5:])
        its = sorted(int(f[9:-4]) for f in os.listdir(os.path.join(od, name))
                     if f.startswith("mea_iter_") and f.endswith(".npz"))
        out[k] = its
    return out


def n_spikes(fx, n_task, k, i):
    with np.load(os.path.join(det_topo_dir(fx, n_task, k), "mea_iter_%05d.npz" % i)) as z:
        return int(z["det_t"].shape[0])


def seed_run(fx, n_task, k, i):
    with np.load(os.path.join(det_topo_dir(fx, n_task, k), "mea_iter_%05d.npz" % i)) as z:
        return int(z["seed_run"])


def set_dispatch(fx, workers, per_worker, tasks=None):
    """job_args.json keys of K(t) = _resolved_n_workers x n_params_per_worker."""
    for n, t in enumerate(fx["tasks"]):
        if tasks is not None and n not in tasks:
            continue
        edit_json(os.path.join(t["campaign_dir"], "job_args.json"),
                  _resolved_n_workers=workers, n_params_per_worker=per_worker)


def write_sidecars(fx, n_task, k, iters):
    d = raw_topo_dir(fx, n_task, k)
    os.makedirs(d, exist_ok=True)
    for i in iters:
        with open(os.path.join(d, "iter_%05d.json" % i), "w", encoding="ascii") as fh:
            json.dump({"iter_idx": int(i), "topo_idx": int(k)}, fh)


def write_failures(fx, n_task, k, text):
    d = raw_topo_dir(fx, n_task, k)
    with open(os.path.join(d, "_failures.jsonl"), "w", encoding="ascii", newline="") as fh:
        fh.write(text)


def complete_all(fx, K):
    """Every topology of every task finished: one sidecar per detection, failure
    lines up to K (each topology must hold at most K detections)."""
    for n in range(len(fx["tasks"])):
        for k, its in detection_iters(fx, n).items():
            write_sidecars(fx, n, k, its)
            extra = K - len(its)
            assert extra >= 0
            if extra:
                write_failures(fx, n, k, "".join('{"iter": %d, "error": "x"}\n' % (900 + j)
                                                 for j in range(extra)))


def load_plan_json(path):
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def pinned(plan_text):
    """The plan's JSON with created, git and free_space pinned (they depend on the
    clock, the checkout and the disk), re-serialised as bank_plan writes it."""
    d = json.loads(plan_text)
    d["created"], d["git"], d["free_space"] = "PINNED", "PINNED", "PINNED"
    return json.dumps(d, indent=1, sort_keys=True) + "\n"
