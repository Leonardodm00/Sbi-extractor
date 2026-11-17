"""Spec-derived tests of Block 7.4 (export_windows/launch_bank.sh), WALLTIME=auto,
as amended on 2026-11-17 (contract amendment 2, the final check's F3/S3), and of its
boundaries with the plan (6.5, Block 10's post-drop n_iters) and the task marker (7.1).
Tester-written (2026-11-17_1605, commit 5b61d6c) from specs/SPEC.md Block 7.4:

  WALLTIME is required for `array`, as hh:mm:ss or `auto`: twice the largest task's
  simulations times the test task's seconds per simulation (from its marker, which
  must be this plan's), plus 10 min, rounded up to a quarter hour, at least 30 min.
  [amendment 2] a test task that read 0 simulations gives no pace, so `array` with
  WALLTIME=auto then refuses, exit 3, submitting nothing, with `ABORT: ERROR
  WALLTIME=auto: the test task <i> read 0 simulations, so it gives no pace: run the
  test on a task that holds simulations (TEST_INDEX=<j>, the plan's first)`, <j> the
  smallest task index whose plan `n_iters` is above 0; `test` itself is unchanged.

Every expected value is computed here by hand from the spec: <j> from the plan's
n_iters read with json, the walltime from the formula above with integer arithmetic.
A fake qsub records its arguments (SPEC sec. 7: the scheduler is not runnable here).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

import pytest

import _bank_helpers as H

LAUNCH = os.path.join(H.EXPORT, "launch_bank.sh")
PLAN_PY = os.path.join(H.EXPORT, "bank_plan.py")
TASK_PY = os.path.join(H.EXPORT, "bank_task.py")
ABORT = ("ABORT: ERROR WALLTIME=auto: the test task %d read 0 simulations, so it gives no "
         "pace: run the test on a task that holds simulations (TEST_INDEX=%d, the plan's first)")


def fake_qsub(root):
    b = os.path.join(root, "fakebin")
    os.makedirs(b, exist_ok=True)
    q = os.path.join(b, "qsub")
    with open(q, "w", encoding="ascii") as fh:
        fh.write('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_QSUB_LOG"\n'
                 'n=$(wc -l < "$FAKE_QSUB_LOG")\necho "${n}000.fake"\n')
    os.chmod(q, 0o755)
    return q


def run_launcher(mode, prof, qsub, log, **over):
    env = dict(os.environ, PROFILE=prof, QSUB=qsub, FAKE_QSUB_LOG=log,
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PIPELINE_JOBS_DIR", "WORK_ROOT",
              "CHAIN_NAME", "AVOID_FILE", "AVOID_HOLD_S"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in over.items()})
    r = subprocess.run(["bash", LAUNCH, mode], env=env, capture_output=True, text=True,
                       timeout=600)
    return r.returncode, r.stdout, r.stderr


def qsub_calls(log):
    if not os.path.exists(log):
        return []
    with open(log, "r", encoding="ascii") as fh:
        return fh.read().splitlines()


def plan_with_empty_tasks(root, empty):
    """The default fixture planned with drop_partial_topologies true (Block 10): every
    topology of the tasks in `empty` partial (K = 1 x 100, nothing finished), every
    topology of the others complete (K = 1 x 1, one iter_<i>.json each). Returns
    (fx, profile path, plan document)."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(os.path.join(root, "fx"))
    for n in range(len(fx["tasks"])):
        H.set_dispatch(fx, 1, 100 if n in empty else 1, tasks=[n])
        if n not in empty:
            for k, its in H.detection_iters(fx, n).items():
                H.write_sidecars(fx, n, k, its[:1])
    prof = H.write_profile(fx, os.path.join(root, "profile.json"), drop_partial_topologies=True)
    rc, so, se = H.run_py(PLAN_PY, "--profile", prof)
    assert rc == 0 and H.last_line(so) == "BANK PLAN OK", so + se
    plan_path = os.path.join(fx["root"], "plan", "bank_plan.json")
    with open(plan_path, "rb") as fh:
        blob = fh.read()
    plan = json.loads(blob.decode("ascii"))
    plan["_path"], plan["_sha256"] = plan_path, H.sha256_bytes(blob)
    return fx, prof, plan


def run_task(plan, i):
    rc, so, se = H.run_py(TASK_PY, "--plan", plan["_path"], "--task-index", i)
    assert rc == 0 and H.last_line(so) == "TASK %d OK" % i, so + se
    with open(os.path.join(plan["profile"]["bank_root"], "tasks", "task_%04d.json" % i),
              "r", encoding="ascii") as fh:
        return json.load(fh)


def first_with_simulations(plan):
    """<j> of amendment 2, by hand: the smallest index whose plan n_iters is above 0."""
    return [n for n, t in enumerate(plan["tasks"]) if t["n_iters"] > 0][0]


@pytest.fixture(scope="module")
def two_empty(tmp_path_factory):
    """Tasks 0 and 1 (the twin pair) wholly partial, task 2 complete: plan n_iters
    (0, 0, 11). <j> = 2 for both empty test tasks, so a rule "the next task" (i + 1)
    and the true rule differ for i = 0."""
    root = str(tmp_path_factory.mktemp("b74_two_empty"))
    fx, prof, plan = plan_with_empty_tasks(root, {0, 1})
    return {"root": root, "fx": fx, "prof": prof, "plan": plan, "qsub": fake_qsub(root)}


def test_plan_has_the_empty_tasks(two_empty):
    """Boundary 6.5 (Block 10) -> 7.4: the plan's post-drop n_iters are (0, 0, 11)."""
    n_iters = [t["n_iters"] for t in two_empty["plan"]["tasks"]]
    assert n_iters == [0, 0, 11]
    assert first_with_simulations(two_empty["plan"]) == 2


def test_test_mode_still_submits_an_empty_task(two_empty):
    """W6: `test` is unchanged: an empty task still gets its one job (TASK_INDEX=i)."""
    log = os.path.join(two_empty["root"], "qsub_test.log")
    for i in (0, 1):
        rc, so, se = run_launcher("test", two_empty["prof"], two_empty["qsub"], log,
                                  TEST_INDEX=i)
        assert rc == 0, so + se
    calls = qsub_calls(log)
    assert len(calls) == 2, calls
    for i, call in enumerate(calls):
        assert "TASK_INDEX=%d," % i in call and call.endswith("run_bank_task.pbs"), call


@pytest.mark.parametrize("i", [0, 1])
@pytest.mark.parametrize("dry", ["0", "1"])
def test_auto_walltime_refuses_an_empty_test_task(two_empty, i, dry):
    """W1-W4, B2, B3: after the element ran task i (its marker reads 0 simulations),
    `array` with WALLTIME=auto exits 3 with amendment 2's line as its last line, <i> the
    test task and <j> = 2 (by hand from the plan), and submits nothing -- also under
    DRYRUN=1, where no submission line may be printed either."""
    plan = two_empty["plan"]
    marker = run_task(plan, i)
    assert marker["totals"]["read"] == 0 and marker["plan_sha256"] == plan["_sha256"]
    log = os.path.join(two_empty["root"], "qsub_array_%d_%s.log" % (i, dry))
    rc, so, se = run_launcher("array", two_empty["prof"], two_empty["qsub"], log,
                              WALLTIME="auto", TEST_INDEX=i, DRYRUN=dry)
    assert rc == 3, so + se
    assert H.last_line(so) == ABORT % (i, first_with_simulations(plan)), so + se
    assert qsub_calls(log) == []
    assert "run_bank_task.pbs" not in so and "run_bank_gate.pbs" not in so, so
    assert "[launch] WALLTIME=auto:" not in so + se


def test_auto_walltime_from_a_test_task_with_simulations(two_empty):
    """W5 (unchanged path): the test on task 2 (11 simulations, the largest) sizes the
    array: at the fixture's pace (a few seconds) the formula's 30 min floor, 00:30:00;
    one array of the plan's 3 tasks and a gate depending on it."""
    plan = two_empty["plan"]
    marker = run_task(plan, 2)
    assert marker["totals"]["read"] == 11
    log = os.path.join(two_empty["root"], "qsub_array_2.log")
    rc, so, se = run_launcher("array", two_empty["prof"], two_empty["qsub"], log,
                              WALLTIME="auto", TEST_INDEX=2)
    assert rc == 0, so + se
    calls = qsub_calls(log)
    assert len(calls) == 2, calls
    assert "walltime=00:30:00" in calls[0] and "-J 0-2%20" in calls[0], calls[0]
    assert "depend=afterok:1000.fake" in calls[1] and calls[1].endswith("run_bank_gate.pbs")


def hand_walltime(secs, read, biggest):
    """Block 7.4's formula with integer arithmetic (seconds are whole in the marker):
    2 * biggest * secs / read + 600 s, rounded UP to a multiple of 900 s, at least 1800 s."""
    num = 2 * biggest * secs + 600 * read          # total seconds times read
    quarters = -(-num // (900 * read))             # ceil(num / (900 read))
    total = max(1800, 900 * quarters)
    return "%02d:%02d:00" % (total // 3600, (total % 3600) // 60)


@pytest.mark.parametrize("secs,want", [(3600, "02:15:00"), (1050, "00:45:00"),
                                       (1051, "01:00:00"), (40000, "22:30:00")])
def test_auto_walltime_formula(tmp_path, secs, want):
    """W7: with the test task's marker timed at `secs` for its 11 simulations and the
    largest task holding 11, the array's walltime is the spec's formula. Cases: a value
    well above the floor (3600 s -> 7800 s -> 9 quarters, 02:15:00), an exact quarter
    (1050 s -> 2700 s, 00:45:00, not rounded up again), one second above it (01:00:00),
    and a long one (40000 s -> 80600 s -> 90 quarters, 22:30:00)."""
    assert hand_walltime(secs, 11, 11) == want
    fx, prof, plan = plan_with_empty_tasks(str(tmp_path), set())
    biggest = max(t["n_iters"] for t in plan["tasks"])
    assert biggest == 11
    marker = run_task(plan, 0)
    read = marker["totals"]["read"]
    assert read == 11
    path = os.path.join(plan["profile"]["bank_root"], "tasks", "task_0000.json")
    t0 = dt.datetime(2026, 11, 17, 10, 0, 0, tzinfo=dt.timezone.utc)
    marker["started"] = t0.isoformat(timespec="seconds")
    marker["finished"] = (t0 + dt.timedelta(seconds=secs)).isoformat(timespec="seconds")
    with open(path, "w", encoding="ascii") as fh:
        json.dump(marker, fh)
    qsub = fake_qsub(str(tmp_path))
    log = os.path.join(str(tmp_path), "qsub.log")
    rc, so, se = run_launcher("array", prof, qsub, log, WALLTIME="auto", TEST_INDEX=0,
                              DRYRUN=1)
    assert rc == 0, so + se
    assert qsub_calls(log) == []
    lines = [ln for ln in so.splitlines() if "run_bank_task.pbs" in ln]
    assert len(lines) == 1 and ("walltime=%s" % want) in lines[0], so
    assert ("walltime=%s" % hand_walltime(secs, read, biggest)) in lines[0]


def test_walltime_refusals_kept(tmp_path):
    """W8 (regression): `array` without WALLTIME, with WALLTIME=auto before any test
    task of the plan, and with a WALLTIME that is not hh:mm:ss each exit 3 and submit
    nothing."""
    fx, prof, plan = plan_with_empty_tasks(str(tmp_path), set())
    qsub = fake_qsub(str(tmp_path))
    log = os.path.join(str(tmp_path), "qsub.log")
    for over in ({}, {"WALLTIME": "auto"}, {"WALLTIME": "1:00"}, {"WALLTIME": "01:60:00"}):
        rc, so, se = run_launcher("array", prof, qsub, log, **over)
        assert rc == 3 and H.last_line(so).startswith("ABORT:"), (over, so + se)
    assert qsub_calls(log) == []


@pytest.mark.xfail(strict=True, reason="finding F3 (2026-11-17_1605, NOTE): in `array` mode "
                   "TEST_INDEX is not validated; WALLTIME=auto with a non-integer TEST_INDEX "
                   "ends in a Python traceback and an `ABORT: ` line with no reason")
def test_array_auto_with_a_bad_test_index_names_it(tmp_path):
    """`test` refuses a TEST_INDEX that is not a task of the plan, naming it; under
    `array` with WALLTIME=auto (which reads the test task's marker by TEST_INDEX) the
    same value should be named too, not end in a traceback. Exit 3 and nothing
    submitted hold either way."""
    fx, prof, plan = plan_with_empty_tasks(str(tmp_path), set())
    qsub = fake_qsub(str(tmp_path))
    log = os.path.join(str(tmp_path), "qsub.log")
    rc, so, se = run_launcher("array", prof, qsub, log, WALLTIME="auto", TEST_INDEX="x")
    assert rc == 3 and qsub_calls(log) == []
    assert "Traceback" not in so + se and "TEST_INDEX=x" in H.last_line(so), so + se
