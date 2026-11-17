"""Smoke test for Block 10 in specs/SPEC.md: partial topologies dropped whole
(SBI D-148 (5), D-143).

With the profile's drop_partial_topologies true, the plan reads K(t) =
_resolved_n_workers x n_params_per_worker from each task's job_args.json and
f(t, k) = iter_<i>.json sidecars + non-blank _failures.jsonl lines from each raw
topology folder; every topology with f(t, k) < K(t) is removed from Top(t) and
E(t) before the donors, the sample, the layout and the counts, and the export
cuts the plan's kept topologies only. With false nothing of this is read and the
plan file is the parent commit's.

Every check runs on the bank-plan fixture (bank_plan_fixture.py, written to a
temporary folder) extended here, not in the shared fixture, with the raw records
Block 10 reads: each job_args.json gets _resolved_n_workers 2 and
n_params_per_worker 2 (K = 4 per topology), tasks 0 and 2 get a topology 3
empty in both trees, and every raw topology folder gets topology_meta.json and
the sidecars and failure lines of CASES below. f counts sidecars and failure
lines only, never the detection files, so a topology of 5 or 6 detections with
4 sidecars is complete here.

    case                                   where                 f   K   verdict
    4 sidecars                             tasks 0-2, topology 0  4   4   complete
    4 sidecars                             task 0, topology 1     4   4   complete
    2 sidecars                             task 0, topology 2     2   4   partial (its twin's is complete)
    0 sidecars, 4 failure lines, empty     task 0, topology 3     4   4   complete, stays in E(t)
    3 sidecars                             task 1, topology 1     3   4   partial (its twin's is complete)
    2 sidecars, 2 failure lines            task 1, topology 2     4   4   complete
    2 sidecars                             task 2, topology 1     2   4   partial
    3 sidecars, 1 failure line, a .tmp     task 2, topology 2     4   4   complete
    0 sidecars, topology_meta.json only    task 2, topology 3     0   4   partial, in E(t)

The comparison with the parent commit extracts the contract commit REF_COMMIT
with `git archive` and runs its bank_plan.py and this tree's on the same
fixture, each in its own interpreter with the three values that change between
two runs pinned (created, git, free_space); it is skipped, with the reason,
where git or that commit is not available.

Run (from the Sbi-extractor root, in the sbi_export environment):
    export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc   # cohort_manifest, latent_bank
    python3 export_windows/smoke_test_partial_topologies.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table; exit
code 1 if any check failed, errored or is TODO. The fixture's folders are
removed at exit.

Keep this file pure ASCII and LF only (hpc-python-compat).
"""
from __future__ import annotations

import atexit
import contextlib
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bank_export as EXP             # noqa: E402
import bank_gate as GATE              # noqa: E402
import bank_plan as PLAN              # noqa: E402
import bank_plan_fixture as FX        # noqa: E402
import bank_plan_load as LOAD         # noqa: E402
import bank_plan_model as MODEL       # noqa: E402
import bank_task as TASK              # noqa: E402
import smoke_test_bank_plan as SP     # noqa: E402  (its helpers; its checks run in its own suite)

SEED = 0
REPORT_PACKAGES = ("numpy", "scipy")
REF_COMMIT = "ae71f33cf55a6a16bd4ef3c594a23c5d3447c2db"   # the contract commit, Block 10's parent
WORKERS, PER_WORKER = 2, 2
K = WORKERS * PER_WORKER                # simulations dispatched per topology (the entry's K = 4)
EXTRA = {0: {3: []}, 2: {3: []}}        # topology 3, empty in both trees (the record counts it)
REMOVE = object()                       # set_job_args: remove the key

# (task index, k) -> (iter_<i>.json sidecars, _failures.jsonl lines); see the docstring
CASES = {(0, 0): (4, 0), (0, 1): (4, 0), (0, 2): (2, 0), (0, 3): (0, 4),
         (1, 0): (4, 0), (1, 1): (3, 0), (1, 2): (2, 2),
         (2, 0): (4, 0), (2, 1): (2, 0), (2, 2): (3, 1), (2, 3): (0, 0)}
# by hand from the table: task -> {k: f} of its partial topologies (f < K = 4)
PARTIAL = {0: {2: 2}, 1: {1: 3}, 2: {1: 2, 3: 0}}
TMP_LEFT = (2, 2)                       # this folder also holds an iter_<i>.json.tmp


# --------------------------------------------------------------- helpers ---

def names():
    return [FX.task_name(t) for t in FX.TASKS]


def topologies_of(n, extra=EXTRA):
    """Every topology of task n in the fixture: its own and the added empty ones."""
    return sorted(set(FX.TASKS[n][4]) | set((extra or {}).get(n, {})))


def raw_folder(root, n, k):
    t = FX.TASKS[n]
    return os.path.join(root, "sim", t[0], t[1], "topo_%05d" % k)


def set_job_args(root, n, **changes):
    """Change task n's raw job_args.json in place (REMOVE removes a key)."""
    t = FX.TASKS[n]
    path = os.path.join(root, "sim", t[0], t[1], "job_args.json")
    with open(path) as fh:
        ja = json.load(fh)
    for key, value in changes.items():
        if value is REMOVE:
            ja.pop(key, None)
        else:
            ja[key] = value
    with open(path, "w") as fh:
        json.dump(ja, fh, indent=2, sort_keys=True)


def write_raw_records(root, cases, extra=EXTRA):
    """The files Block 10 reads, in the simulator's layout: job_args.json's two
    dispatch keys; per raw topology folder topology_meta.json, the sidecars
    iter_<i>.json (named after the folder's first iterations, then beyond them)
    and _failures.jsonl (one JSON object per line, then a blank and a
    whitespace-only line, which are not runs)."""
    for n in range(len(FX.TASKS)):
        set_job_args(root, n, _resolved_n_workers=WORKERS, n_params_per_worker=PER_WORKER)
        for k in topologies_of(n, extra):
            folder = raw_folder(root, n, k)
            with open(os.path.join(folder, "topology_meta.json"), "w") as fh:
                json.dump({"topo_idx": k, "c_max": 1.0}, fh)
            n_side, n_fail = cases[(n, k)]
            iters = list(FX.TASKS[n][4].get(k, []))
            beyond = list(range(max(iters + [-1]) + 1, max(iters + [-1]) + 1 + n_side))
            for i in (iters + beyond)[:n_side]:
                with open(os.path.join(folder, "iter_%05d.json" % i), "w") as fh:
                    json.dump({"topo_idx": k, "iter_idx": i}, fh)
            if n_fail:
                with open(os.path.join(folder, "_failures.jsonl"), "w") as fh:
                    for j in range(n_fail):
                        fh.write(json.dumps({"topo_idx": k, "iter_idx": 900 + j,
                                             "error_msg": "synthetic failure"}) + "\n")
                    fh.write("\n   \n")
            if (n, k) == TMP_LEFT:       # a kill during the sidecar's atomic write
                with open(os.path.join(folder, "iter_00099.json.tmp"), "w") as fh:
                    fh.write("{")


class Partial(object):
    """The fixture with the raw records of `cases`, its profile saying `drop`."""

    def __init__(self, drop=True, cases=None, extra=EXTRA, **overrides):
        self.f = SP.Fixture(extra_topos=extra, drop_partial_topologies=drop, **overrides)
        self.dir = self.f.dir
        self.profile = self.f.profile
        try:
            write_raw_records(self.dir, cases or CASES, extra)
        except Exception:                # no folder left behind by a fixture that failed
            self.f.close()
            raise

    def cfg(self):
        return self.f.cfg()

    def run_plan(self, out=None):
        """(exit code, printed lines, the plan or None)."""
        argv = ["--profile", self.profile] + (["--out", out] if out else [])
        rc, lines = SP.run_main(PLAN.main, argv)
        path = out or self.cfg().plan_out
        plan = None
        if os.path.isfile(path):
            with open(path) as fh:
                plan = json.load(fh)
        return rc, lines, plan

    def close(self):
        self.f.close()


_SHARED = {}


def shared():
    """One partial fixture with drop true and its plan (test_10_export_and_gate
    exports it into its bank_root); no check changes its inputs or its plan."""
    if not _SHARED:
        p = Partial()
        atexit.register(p.close)
        rc, out, plan = p.run_plan()
        assert rc == 0 and out[-1] == "BANK PLAN OK", out[-3:]
        _SHARED.update(p=p, out=out, plan=plan)
    return _SHARED


def kept_topo_iters(n):
    """Top(t) of task n after the drop, by hand: its topologies with detections
    but those of PARTIAL."""
    return {k: list(v) for k, v in sorted(FX.TASKS[n][4].items()) if k not in PARTIAL[n]}


def kept_empty(n):
    """E(t) of task n after the drop, by hand."""
    return sorted(k for k in EXTRA.get(n, {}) if k not in PARTIAL[n])


def positions_digest(topo_iters):
    """sha256 of the canonical JSON [[k, [i, ...]], ...] (SPEC 2.5), computed here."""
    pos = [[int(k), [int(i) for i in topo_iters[k]]] for k in sorted(topo_iters)]
    return hashlib.sha256(json.dumps(pos, separators=(",", ":")).encode("ascii")).hexdigest()


def fixture_seeds(n):
    """s(t) of the fixture's task n: the job_args keys holding "seed", seed_master
    dropped beside _resolved_seed_master (sim_reextract_plan.task_seed)."""
    return {"_resolved_seed_master": FX.TASKS[n][3], "seed_device": 0, "seed_neuron": 1,
            "seed_synapse": 2, "seed_astro": 3}


def seed_key(seeds):
    return json.dumps(seeds, sort_keys=True, separators=(",", ":"))


def hand_draws(sample_seed):
    """(task name, k) -> (i, comparable) on the kept topologies, recomputed here from
    SPEC 2.5's rule: per seed group and k, one i from the common iterations of the
    tasks that keep k (key "<seed key>|k"), else per task (key "<task>|k")."""
    groups = {}
    for n in range(len(FX.TASKS)):
        groups.setdefault(seed_key(fixture_seeds(n)), []).append(n)
    out = {}
    for key, members in groups.items():
        for k in sorted({k for n in members for k in kept_topo_iters(n)}):
            have = [n for n in members if k in kept_topo_iters(n)]
            common = sorted(set.intersection(*[set(kept_topo_iters(n)[k]) for n in have]))
            if common:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [sample_seed, zlib.crc32(("%s|%d" % (key, k)).encode("ascii"))]))
                i = common[int(rng.integers(0, len(common)))]
                for n in have:
                    out[(names()[n], k)] = (i, len(have) > 1)
            else:
                for n in have:
                    its = kept_topo_iters(n)[k]
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [sample_seed, zlib.crc32(("%s|%d" % (names()[n], k)).encode("ascii"))]))
                    out[(names()[n], k)] = (its[int(rng.integers(0, len(its)))], False)
    return out


def kept_by_floor(n, k, i):
    """The fixture's spike count of (task n, k, i) passes the 0.1 Hz floor: >= 18."""
    return FX.spikes_of(FX.TASKS[n], k, i) >= 18


def expect_refused(lines, *fragments):
    assert lines[-1].startswith("REFUSED:"), lines[-3:]
    text = "\n".join(lines)
    for fragment in fragments:
        assert fragment in text, (fragment, lines[-4:])


class _Now(object):
    @staticmethod
    def now():
        return datetime.datetime(2026, 11, 17, 12, 0, 0, tzinfo=datetime.timezone.utc)


class _PinnedDatetime(object):
    datetime = _Now


def _pinned_free_space(bank_root):
    return {"path_checked": "pinned", "free_bytes": 0, "total_bytes": 0, "lfs_quota": None}


@contextlib.contextmanager
def pinned():
    """The three plan values that change between two runs of one code on one
    fixture -- created, git, free_space -- pinned, in this interpreter."""
    keep = (PLAN._dt, PLAN.git_commit, PLAN.SAMPLE.free_space)
    PLAN._dt, PLAN.git_commit = _PinnedDatetime, (lambda: "pinned")
    PLAN.SAMPLE.free_space = _pinned_free_space
    try:
        yield
    finally:
        PLAN._dt, PLAN.git_commit, PLAN.SAMPLE.free_space = keep


# The same pins for a bank_plan.py of another tree, run in its own interpreter:
# argv = code folder (an export_windows/), profile, output path.
DRIVER = '''\
import datetime
import sys
code_dir, profile, out = sys.argv[1:4]
sys.path.insert(0, code_dir)
import bank_plan as P


class _Now(object):
    @staticmethod
    def now():
        return datetime.datetime(2026, 11, 17, 12, 0, 0, tzinfo=datetime.timezone.utc)


class _PinnedDatetime(object):
    datetime = _Now


P._dt = _PinnedDatetime
P.git_commit = lambda: "pinned"
P.SAMPLE.free_space = lambda bank_root: {"path_checked": "pinned", "free_bytes": 0,
                                         "total_bytes": 0, "lfs_quota": None}
sys.exit(P.main(["--profile", profile, "--out", out]))
'''


def reference_tree(dest):
    """The contract commit's whole tree under dest (git archive, run from the
    checkout's top level: from a subfolder it archives that subfolder only);
    returns its folder that corresponds to this file's. SkipTest without git or
    without the commit."""
    try:
        top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=60)
        if top.returncode != 0:
            raise unittest.SkipTest("%s is not in a git checkout (%s)"
                                    % (HERE, top.stderr.strip()))
        root = top.stdout.strip()
        r = subprocess.run(["git", "-C", root, "archive", "--format=tar", REF_COMMIT],
                           capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise unittest.SkipTest("git cannot run here (%s)" % exc)
    if r.returncode != 0:
        raise unittest.SkipTest("the parent commit %s is not in this checkout (%s)"
                                % (REF_COMMIT[:7], r.stderr.decode("utf-8", "replace").strip()))
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:                           # python before 3.11.4: no extraction filters
            tf.extractall(dest)
    code = os.path.join(dest, os.path.relpath(os.path.realpath(HERE), os.path.realpath(root)))
    if not os.path.isfile(os.path.join(code, "bank_plan.py")):
        raise AssertionError("the archive of %s has no %s" % (REF_COMMIT[:7], code))
    return code


def run_pinned(driver, code_dir, profile, out):
    r = subprocess.run([sys.executable, driver, code_dir, profile, out], capture_output=True,
                       text=True, timeout=600)
    lines = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode, lines


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


# ------------------------------------------------------ the loaders' checks ---

def test_10_loaders_known_answer():
    p = Partial()
    try:
        for n, t in enumerate(FX.TASKS):
            camp = os.path.join(p.dir, "sim", t[0], t[1])
            assert LOAD.dispatch_size(camp) == 4, n
            want = {k: CASES[(n, k)][0] + CASES[(n, k)][1] for k in topologies_of(n)}
            got = LOAD.raw_finished_counts(camp)
            assert got == want and list(got) == sorted(want), (n, got, want)
            assert {k: f for k, f in got.items() if f < K} == PARTIAL[n], (n, got)
        tmp_left = os.path.join(raw_folder(p.dir, *TMP_LEFT), "iter_00099.json.tmp")
        assert os.path.isfile(tmp_left), tmp_left         # present, and not counted above
    finally:
        p.close()


def test_10_dispatch_size_refusals():
    p = Partial()
    try:
        camp = os.path.join(p.dir, "sim", FX.TASKS[1][0], FX.TASKS[1][1])
        for key in LOAD.DISPATCH_KEYS:
            for bad, shown in ((REMOVE, "'absent'"), (0, "0"), (-2, "-2"), (2.0, "2.0"),
                               (True, "True"), ("2", "'2'"), (None, "None")):
                set_job_args(p.dir, 1, **{key: bad})
                SP.expect_error(LOAD.LoadError, "%s: %s is %s, not a positive integer"
                                % (os.path.join(camp, "job_args.json"), key, shown),
                                LOAD.dispatch_size, camp)
            set_job_args(p.dir, 1, **{key: 2})
        set_job_args(p.dir, 1, _resolved_n_workers=3, n_params_per_worker=5)
        assert LOAD.dispatch_size(camp) == 15            # the product, not one key
        os.remove(os.path.join(camp, "job_args.json"))
        SP.expect_error(LOAD.LoadError, "job_args.json cannot be read", LOAD.dispatch_size, camp)
    finally:
        p.close()


def test_10_raw_counts_refusals():
    p = Partial()
    try:
        camp = os.path.join(p.dir, "sim", FX.TASKS[0][0], FX.TASKS[0][1])
        os.makedirs(os.path.join(camp, "topo_3"))       # beside topo_00003
        SP.expect_error(LOAD.LoadError, "two folders give topology 3", LOAD.raw_finished_counts,
                        camp)
        SP.expect_error(LOAD.LoadError, "no raw task folder", LOAD.raw_finished_counts,
                        os.path.join(p.dir, "sim", "absent"))
    finally:
        p.close()


# -------------------------------------------------------- the model's checks ---

def test_10_model_helpers():
    inp = SP.shared()["inp"]                 # the default fixture: Top {0, 1, 2}, E empty
    t = inp.tasks[2]
    rows = MODEL.partial_topologies(t, {0: 4, 1: 3, 2: 0}, 4)
    assert rows == [{"topo": 1, "finished": 3, "expected": 4},
                    {"topo": 2, "finished": 0, "expected": 4}], rows
    assert MODEL.partial_topologies(t, {0: 4, 1: 4, 2: 4}, 4) == []   # f = K: complete
    SP.expect_error(MODEL.PlanRefused, "topology 1 holds 5", MODEL.partial_topologies, t,
                    {0: 4, 1: 5, 2: 6}, 4)
    SP.expect_error(MODEL.PlanRefused, "topology 2 holds 6", MODEL.partial_topologies, t,
                    {0: 4, 1: 5, 2: 6}, 4)
    SP.expect_error(MODEL.PlanRefused, "raw only [7], detections only [2]",
                    MODEL.partial_topologies, t, {0: 4, 1: 4, 7: 4}, 4)
    before = (dict(t.topo_iters), t.empty_topos, t.n_iters, t.n_topos)
    d = MODEL.drop_topologies(t, [1])
    assert d.topo_iters == {0: FX.TASKS[2][4][0], 2: FX.TASKS[2][4][2]}, d.topo_iters
    assert (d.n_iters, d.n_topos, d.empty_topos) == (9, 2, ()), (d.n_iters, d.n_topos)
    assert (dict(t.topo_iters), t.empty_topos, t.n_iters, t.n_topos) == before   # unchanged
    assert MODEL.drop_topologies(t, []) == t
    SP.expect_error(MODEL.PlanRefused, "partial topology 5 is not in its detections tree",
                    MODEL.drop_topologies, t, [1, 5])
    e = SP.empty_fixture()                   # an empty topology: dropped from E(t)
    try:
        te = LOAD.load_inputs(e.cfg()).tasks[SP.EMPTY[0]]
        de = MODEL.drop_topologies(te, [SP.EMPTY[1]])
        assert de.empty_topos == () and de.topo_iters == te.topo_iters, de.empty_topos
        assert (de.n_iters, de.n_topos) == (te.n_iters, te.n_topos - 1)
    finally:
        e.close()


# --------------------------------------------------------- the plan's checks ---

def test_10_plan_drops_exactly_the_partial():
    sh = shared()
    plan, nm = sh["plan"], names()
    assert plan["drop_partial_topologies"] is True and plan["profile"]["drop_partial_topologies"]
    assert (plan["n_partial_topos"], plan["n_partial_tasks"], plan["n_dropped_partial"]) == (
        4, 3, 7), plan["n_dropped_partial"]
    for n, t in enumerate(plan["tasks"]):
        want = [{"topo": k, "finished": f, "expected": K} for k, f in sorted(PARTIAL[n].items())]
        assert t["partial_topos"] == want, (n, t["partial_topos"])
        dropped = sum(len(FX.TASKS[n][4].get(k, [])) for k in PARTIAL[n])
        assert t["n_dropped_partial"] == dropped, (n, t["n_dropped_partial"])
        kti = kept_topo_iters(n)
        assert t["topo_counts"] == {str(k): len(v) for k, v in kti.items()}, n
        assert t["n_iters"] == sum(len(v) for v in kti.values()), n
        assert t["positions_sha256"] == positions_digest(kti), n
        assert t["empty_topos"] == kept_empty(n), (n, t["empty_topos"])
        assert t["n_topos"] == len(kti) + len(kept_empty(n)), n
        assert t["shards"] == [{"shard": n, "start": 0, "stop": t["n_iters"]}], t["shards"]
        assert t["seeds"] == fixture_seeds(n), t["seeds"]
    assert [t["n_iters"] for t in plan["tasks"]] == [9, 7, 9]
    assert (plan["n_rows"], plan["n_shards"]) == (25, 3)        # 32 - 7 simulations
    # donors by hand: the sorted distinct (seed values, k) keys of the kept topologies
    keys = sorted({(seed_key(fixture_seeds(n)), k) for n in range(3) for k in kept_topo_iters(n)})
    assert plan["n_donors"] == len(keys) == 5, (plan["n_donors"], keys)
    for n, t in enumerate(plan["tasks"]):
        want = {str(k): keys.index((seed_key(fixture_seeds(n)), k)) for k in kept_topo_iters(n)}
        assert t["donors"] == want, (n, t["donors"], want)
        assert not set(t["donors"]) & {str(k) for k in PARTIAL[n]}, n   # dropped: no donor
    assert (seed_key(fixture_seeds(2)), 1) not in keys                  # nor in the numbering
    # the sample: one draw per kept topology, on the kept topologies' iterations
    draws = hand_draws(sh["p"].cfg().sample_seed)
    rows = plan["sample"]["rows"]
    assert {(r[0], r[1]) for r in rows} == set(draws) and len(rows) == 6, rows
    for task, k, i, _rate, kept, comparable in rows:
        n = nm.index(task)
        assert (i, comparable) == draws[(task, k)], (task, k, i, comparable)
        assert kept == kept_by_floor(n, k, i), (task, k, i)
    num = sum(len(kept_topo_iters(nm.index(r[0]))[r[1]]) * kept_by_floor(nm.index(r[0]), r[1], r[2])
              for r in rows)
    den = sum(len(kept_topo_iters(nm.index(r[0]))[r[1]]) for r in rows)
    share = plan["sample"]["kept_share"]
    assert abs(share["p_hat"] - num / float(den)) < 1e-12, (share["p_hat"], num, den)
    assert (share["n_topologies"], share["n_donors"]) == (6, 5), share
    assert plan["sample"]["twin_check"] == [{"tasks": nm[:2], "shared_topologies": 1,
                                             "conn_prob_equal": 1, "seed_run_comparable": 1,
                                             "seed_run_equal": 1}], plan["sample"]["twin_check"]
    assert plan["estimates"]["n_rows"] == 25 and plan["estimates"]["raw_bytes"] == 25 * 18000 * 4


def test_10_plan_printed_lines():
    out, nm = shared()["out"], names()
    i = out.index("[plan] partial topologies (drop_partial_topologies): 4 dropped in 3 task(s), "
                  "7 simulations; bank_plan.json tasks[].partial_topos")
    assert out[i + 1:i + 5] == ["[plan]   %s topo 2: 2 of 4" % nm[0],
                                "[plan]   %s topo 1: 3 of 4" % nm[1],
                                "[plan]   %s topo 1: 2 of 4" % nm[2],
                                "[plan]   %s topo 3: 0 of 4" % nm[2]], out[i:i + 6]
    assert not out[i + 5].startswith("[plan]   "), out[i + 5]
    assert "[plan] donors         : 5 (from 6 topologies with simulations; D-076)" in out, [
        line for line in out if line.startswith("[plan] donors")]
    j = out.index("[plan] topologies     : 1 without simulations (the folder is empty in the "
                  "raw task and in its detections: no row, no donor)")
    assert out[j + 1] == "[plan]   no simulation: %s topology 3" % nm[0], out[j + 1]
    assert not any("not checked" in line for line in out), out
    eleven = {(n, k): (4, 0) for n in range(3) for k in range(3)}    # every topology complete
    eleven.update({(1, k): (0, 0) for k in range(3, 14)})             # but eleven empty ones
    p = Partial(extra={1: {k: [] for k in range(3, 14)}}, cases=eleven)
    try:
        rc, lines, plan = p.run_plan()
        assert rc == 0 and lines[-1] == "BANK PLAN OK", lines[-3:]
        i = lines.index("[plan] partial topologies (drop_partial_topologies): 11 dropped in 1 "
                        "task(s), 0 simulations; bank_plan.json tasks[].partial_topos")
        assert lines[i + 1:i + 11] == ["[plan]   %s topo %d: 0 of 4" % (nm[1], k)
                                       for k in range(3, 13)], lines[i:i + 12]
        assert not lines[i + 11].startswith("[plan]   "), lines[i + 11]   # the first ten only
        assert [len(t["partial_topos"]) for t in plan["tasks"]] == [0, 11, 0]
        assert plan["n_rows"] == 32 and plan["n_donors"] == 6       # nothing with rows dropped
    finally:
        p.close()


def test_10_plan_refusals():
    nm = names()

    def refused(mutate, fragments, **kw):
        p = Partial(**kw)
        try:
            mutate(p)
            rc, lines, plan = p.run_plan()
            assert rc == 1 and plan is None, (rc, lines[-3:])
            expect_refused(lines, *fragments)
            return lines
        finally:
            p.close()

    five = dict(CASES)
    five[(0, 0)] = (5, 0)                        # 5 sidecars beside its 5 detections, K = 4
    refused(lambda p: None, ["REFUSED: %s: more runs than the task dispatched per topology, "
                             "K = 4" % nm[0], "topology 0 holds 5"], cases=five)
    refused(lambda p: set_job_args(p.dir, 1, n_params_per_worker=REMOVE),
            ["REFUSED: %s: " % nm[1], "n_params_per_worker is 'absent', not a positive integer"])
    lines = refused(lambda p: set_job_args(p.dir, 1, n_params_per_worker=REMOVE),
                    ["REFUSED: 2 of 3 tasks refused (each named above)"], cases=five)
    assert [line.split(": ")[0] for line in lines[-3:-1]] == [
        "[plan] refused  %s" % nm[0], "[plan] refused  %s" % nm[1]], lines[-3:]

    def rename_raw(p):                   # raw topology 2 of task 2 renamed 9: f has no match
        os.rename(raw_folder(p.dir, 2, 2), raw_folder(p.dir, 2, 9))
    refused(rename_raw, ["%s: the raw task's topology folders and its detections' differ "
                         "(raw only [9], detections only [2])" % nm[2]])
    refused(lambda p: None, ["no detection file is left once the partial topologies are "
                             "dropped (drop_partial_topologies: 11 topologies, 32 simulations)"],
            cases={key: (0, 0) for key in CASES})

    def remove_dropped_detection(p):     # 6.2's record check, on the full tree before the drop
        t = FX.TASKS[1]
        os.remove(SP.mea_path(p.dir, t, 1, 2))
    refused(remove_dropped_detection, ["%s: 3 topology folders (0 without a detection file) "
                                       "and 9 detection files" % nm[1],
                                       "the record says 3 and 10"])


def test_10_false_reads_nothing_more():
    def boom(*_a, **_k):
        raise AssertionError("read with drop_partial_topologies false")

    p = Partial(drop=False)
    keep = (LOAD.dispatch_size, LOAD.raw_finished_counts)
    LOAD.dispatch_size, LOAD.raw_finished_counts = boom, boom
    try:
        a, b = os.path.join(p.dir, "a", "bank_plan.json"), os.path.join(p.dir, "b", "bank_plan.json")
        with pinned():
            rc, out, plan = p.run_plan(out=a)
            assert rc == 0 and out[-1] == "BANK PLAN OK", out[-3:]
            set_job_args(p.dir, 1, n_params_per_worker=REMOVE, _resolved_n_workers=REMOVE)
            rc, out2, _plan = p.run_plan(out=b)
            assert rc == 0 and out2[-1] == "BANK PLAN OK", out2[-3:]
        assert read_bytes(a) == read_bytes(b)          # the keys are not read with false
        assert "[plan] partial topologies: not checked (drop_partial_topologies false)" in out
        assert not any("partial topologies (drop" in line for line in out), out
        for key in ("drop_partial_topologies", "n_partial_topos", "n_partial_tasks",
                    "n_dropped_partial"):
            assert key not in plan, key
        for t in plan["tasks"]:
            assert "partial_topos" not in t and "n_dropped_partial" not in t, t["name"]
        assert plan["profile"]["drop_partial_topologies"] is False
        assert (plan["n_rows"], plan["n_donors"]) == (32, 6)      # nothing dropped
        assert [t["empty_topos"] for t in plan["tasks"]] == [[3], [], [3]]
    finally:
        LOAD.dispatch_size, LOAD.raw_finished_counts = keep
        p.close()


def test_10_false_plan_equals_parent_commit():
    """With false, the plan file is the parent commit's byte for byte (SPEC Block
    10), on the default fixture and on this suite's, its raw records included."""
    tmp = tempfile.mkdtemp(prefix="partial_ref_")
    try:
        ref_code = reference_tree(os.path.join(tmp, "ref"))
        driver_dir = os.path.join(tmp, "driver")
        os.makedirs(driver_dir)
        driver = os.path.join(driver_dir, "pinned_plan.py")
        with open(driver, "w", encoding="ascii") as fh:
            fh.write(DRIVER)
        for label, fixture in (("default", lambda: SP.Fixture()),
                               ("partial records", lambda: Partial(drop=False))):
            f = fixture()
            try:
                outs = {}
                for side, code in (("ref", ref_code), ("new", HERE)):
                    out = os.path.join(f.dir, side, "bank_plan.json")
                    rc, lines = run_pinned(driver, code, f.profile, out)
                    assert rc == 0 and lines[-1] == "BANK PLAN OK", (label, side, lines[-3:])
                    outs[side] = (read_bytes(out), read_bytes(out + ".sha256"), lines)
                assert outs["new"][0] == outs["ref"][0], label               # the plan file
                assert outs["new"][1] == outs["ref"][1], label               # its .sha256
                note = "[plan] partial topologies: not checked (drop_partial_topologies false)"
                assert note in outs["new"][2] and note not in outs["ref"][2], label
            finally:
                f.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- the export's and gate's checks ---

def test_10_export_and_gate():
    sh = shared()
    p, plan = sh["p"], EXP.load_plan(sh["p"].cfg().plan_out)
    bank = p.cfg().bank_root
    lb, _ln = EXP.stage1_modules()
    for n, want in enumerate((9, 7, 9)):     # the array element, on the kept simulations
        rc, out = SP.run_main(TASK.main, ["--plan", plan["_path"], "--task-index", str(n)])
        assert rc == 0 and out[-1] == "TASK %d OK" % n, out[-3:]
        assert "(%d simulations, 1 shard(s))" % want in out[1], out[1]
    total_read = total_kept = would_be_kept = 0
    for n, t in enumerate(FX.TASKS):
        kti = kept_topo_iters(n)
        pos = [(k, i) for k in sorted(kti) for i in kti[k]]
        kept = [(k, i) for k, i in pos if kept_by_floor(n, k, i)]
        with open(EXP.marker_path(bank, n)) as fh:
            tot = json.load(fh)["totals"]
        assert (tot["read"], tot["kept"], tot["dropped_floor"]) == (
            len(pos), len(kept), len(pos) - len(kept)), (n, tot)
        a, _side = lb.read_shard(os.path.join(bank, "shard_%04d.npz" % n))
        inv = {d: int(k) for k, d in plan["tasks"][n]["donors"].items()}
        got = [(inv[int(d)], int(w)) for d, w in zip(a["donor"], a["window_idx"])]
        assert got == kept, (n, got, kept)
        assert not {k for k, _i in got} & set(PARTIAL[n]), n          # no row of a dropped one
        assert [int(r) for r in a["realisation_id"]] == [
            FX.seed_run_of(t, k, i) for k, i in kept], n
        assert np.all(a["well"] == n), n
        sums = a["x"].astype(np.float64).sum(axis=1)     # count conservation (SPEC 2.1)
        np.testing.assert_allclose(sums, [FX.spikes_of(t, k, i) for k, i in kept], rtol=1e-6,
                                   err_msg="task %d" % n)  # float32 storage of x
        total_read += len(pos)
        total_kept += len(kept)
        would_be_kept += sum(kept_by_floor(n, k, i) for k in PARTIAL[n]
                             for i in t[4].get(k, []))
    assert (total_read, total_kept) == (25, 18), (total_read, total_kept)
    assert would_be_kept == 6       # the dropped topologies' rows the floor alone would keep
    rc, out = SP.run_main(GATE.main, ["--plan", plan["_path"], "--workers", "1"])
    assert rc == 0 and out[-1].startswith("wrote ") and "BANK_RECORD.json" in out[-1], out[-3:]
    with open(os.path.join(bank, "BANK_RECORD.json")) as fh:
        rec = json.load(fh)
    assert (rec["counts"]["rows_read"], rec["counts"]["rows_kept"], rec["counts"]["n_donors"],
            rec["counts"]["shards"]) == (25, 18, 5, 3), rec["counts"]


def test_10_export_refusals():
    nm = names()

    def fails(mutate, index, fragments):
        p = Partial()
        try:
            rc, out, plan = p.run_plan()
            assert rc == 0, out[-1:]
            mutate(p)
            rc, out = SP.run_main(TASK.main, ["--plan", p.cfg().plan_out, "--task-index",
                                              str(index)])
            assert rc == 1 and out[-1].startswith("FAILED:"), out[-2:]
            for fragment in fragments:
                assert fragment in out[-1], (fragment, out[-1])
            assert not os.path.exists(EXP.marker_path(p.cfg().bank_root, index))
        finally:
            p.close()

    def rename_dropped(p):               # both trees: the counts are unchanged, topology 3 gone
        t = FX.TASKS[2]
        for tree in ("sim", "mea"):
            d = os.path.join(p.dir, tree, t[0], t[1])
            os.rename(os.path.join(d, "topo_00003"), os.path.join(d, "topo_00004"))
    fails(rename_dropped, 2, ["%s: partial topology 3 is not in its detections tree" % nm[2],
                              "the detections changed after the plan: re-plan"])

    def remove_dropped_detection(p):     # a dropped topology's file: 6.2's full-tree check
        os.remove(SP.mea_path(p.dir, FX.TASKS[1], 1, 2))
    fails(remove_dropped_detection, 1, ["%s: 3 topology folders (0 without a detection file) "
                                        "and 9 detection files" % nm[1],
                                        "the record says 3 and 10"])

    def remove_kept_detection(p):        # a kept topology's file: the same check
        os.remove(SP.mea_path(p.dir, FX.TASKS[0], 1, 3))
    fails(remove_kept_detection, 0, ["the record says 4 and 11"])


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_determinism():
    raise unittest.SkipTest("no randomness of its own: the draws are Block 6.4's, checked on "
                            "the kept topologies in test_10_plan_drops_exactly_the_partial")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append("SBI_HPC_DIR=%s" % os.environ.get("SBI_HPC_DIR", "(unset)"))
    parts.append(platform.platform())
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds,
                                           lines[0] if lines else "")).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
