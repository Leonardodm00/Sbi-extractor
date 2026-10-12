#!/bin/bash
# =============================================================================
# launch_bank.sh -- B5 of the Giulia project: the window banks of arm A1
# (specs/SPEC.md, Blocks 6-8). The plan, then the simulated bank as one PBS
# array (one element per task of the plan) plus one dependent gate job, then
# the real bank.
#
#     cd ~/repos/Sbi-extractor/export_windows && conda activate sbi_export
#     bash launch_bank.sh plan                      # B5.1, here; read-only over the data
#     DRYRUN=1 bash launch_bank.sh test             # the test task's qsub line, nothing submitted
#     bash launch_bank.sh test                      # ONE task as a job (TEST_INDEX, default 0)
#     WALLTIME=auto bash launch_bank.sh array       # every task of the plan + the gate job
#     bash launch_bank.sh gate                      # the gate job alone (after a resubmission)
#     bash launch_bank.sh real                      # B5.5, here: the real bank, one shard
#
# ORDER. plan -> its last line `BANK PLAN OK`, with the estimates above it
# (kept share, GB, free space) -> test -> the test task's log must end in
# `TASK <i> OK`, with its seconds -> array (the test task's element finds its
# marker complete and does nothing) -> the gate's log must end in
# `wrote <bank_root>/BANK_RECORD.json  sha256 <16 hex>` -> real, whose last
# line must be `REAL BANK OK`. WALLTIME=auto sizes the array from the test
# task's pace and the plan's largest task (printed by `test`).
#
# ONE FOLDER, ONE PLAN. The tasks refuse to write into a bank folder that
# holds another plan's markers or record (SPEC sec. 2.6). If you re-plan after
# a test or a run, move the old bank folder aside first; nothing here deletes.
#
# depend=afterok on an array is ORDERING ONLY on davinci's PBS (D-006): the
# gate job runs whatever the elements' exit codes; the gate is what refuses
# a partial bank, naming every task. Read its log, not the array's.
#
# ENVIRONMENT (defaults are davinci's)
#   PROFILE       the bank profile: a name under profiles/ (default giulia) or a .json
#   ENV_NAME      sbi_export (activated inside the jobs)
#   QUEUE         cpu        NCPUS 1 (a task is serial)       MEM 8gb
#   CONCURRENCY   20         WALLTIME  required for `array`: hh:mm:ss, or auto -- twice
#                            the largest task's time at the test task's pace (its
#                            marker), plus 10 min, in quarter hours, at least 30 min;
#                            02:00:00 for `test`
#   GATE_NCPUS    8          GATE_MEM 16gb    GATE_WALLTIME 04:00:00
#   TEST_INDEX    the plan's task the test runs (default 0)
#   DRYRUN=1      print every command, submit nothing
#   QSUB          the submit command (the smoke test substitutes a fake)
#
# Logs: export_windows/<work>/logs/ (gitignored), named per job below.
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
HERE="$(pwd -P)"

MODE="${1:-}"
case "$MODE" in
    plan|test|array|gate|real) ;;
    *) sed -n '2,47p' "$HERE/launch_bank.sh"; exit 2 ;;
esac
# shellcheck source=/dev/null
source ../env.sh

PROFILE="${PROFILE:-giulia}"
ENV_NAME="${ENV_NAME:-sbi_export}"
QUEUE="${QUEUE:-cpu}"
NCPUS="${NCPUS:-1}"
MEM="${MEM:-8gb}"
CONCURRENCY="${CONCURRENCY:-20}"
WALLTIME="${WALLTIME:-}"
GATE_NCPUS="${GATE_NCPUS:-8}"
GATE_MEM="${GATE_MEM:-16gb}"
GATE_WALLTIME="${GATE_WALLTIME:-04:00:00}"
TEST_INDEX="${TEST_INDEX:-0}"
DRYRUN="${DRYRUN:-0}"
QSUB="${QSUB:-qsub}"

command -v python3 >/dev/null 2>&1 || { echo "ABORT: no python3 on PATH (conda activate $ENV_NAME first)"; exit 2; }
echo "[launch] python3 : $(command -v python3)  (env ${CONDA_DEFAULT_ENV:-none})"

# --- the profile's paths (one source: bank_plan_config) ----------------------
PATHS=$(python3 - "$PROFILE" <<'EOF'
import sys
import bank_plan_config as C
try:
    c = C.load_profile(sys.argv[1])
except C.ConfigError as exc:
    print("ERROR %s" % exc)
    sys.exit(1)
print(c.plan_out)
print(c.bank_root)
print(c.real_bank_root)
EOF
) || { echo "ABORT: the profile $PROFILE does not load: $PATHS"; exit 2; }
PLAN=$(sed -n 1p <<< "$PATHS")
BANK_ROOT=$(sed -n 2p <<< "$PATHS")
REAL_ROOT=$(sed -n 3p <<< "$PATHS")
for v in PLAN BANK_ROOT REAL_ROOT; do
    case "${!v}" in *[[:space:],]*) echo "ABORT: $v contains whitespace or a comma, which qsub -v cannot carry: ${!v}"; exit 3 ;; esac
done
WORK="$(dirname "$PLAN")"
LOGDIR="$WORK/logs"
echo "[launch] profile : $PROFILE"
echo "[launch] plan    : $PLAN"

# --- the modes that run here -------------------------------------------------
if [ "$MODE" = "plan" ]; then
    python3 bank_plan.py --profile "$PROFILE"
    rc=$?
    [ "$rc" = "0" ] && echo "[launch] next: DRYRUN=1 bash launch_bank.sh test, then bash launch_bank.sh test"
    exit "$rc"
fi
if [ "$MODE" = "real" ]; then
    echo "[launch] real bank -> $REAL_ROOT"
    python3 bank_real.py --profile "$PROFILE"
    exit $?
fi

# --- the job modes: the plan must exist and match its .sha256 ----------------
INFO=$(python3 - "$PLAN" <<'EOF'
import hashlib, json, os, sys
p = sys.argv[1]
if not (os.path.isfile(p) and os.path.isfile(p + ".sha256")):
    print("ERROR no plan at %s (run: bash launch_bank.sh plan)" % p)
    sys.exit(1)
blob = open(p, "rb").read()
want = open(p + ".sha256").read().split()[0]
if hashlib.sha256(blob).hexdigest() != want:
    print("ERROR %s does not match its .sha256" % p)
    sys.exit(1)
plan = json.loads(blob.decode("ascii"))
tasks = plan["tasks"]
big = max(tasks, key=lambda t: t["n_iters"])
print(len(tasks))
print(want[:16])
print("%s (%d simulations, %d shard(s))" % (big["name"], big["n_iters"], len(big["shards"])))
print(plan["n_rows"])
EOF
) || { echo "ABORT: $INFO"; exit 3; }
N=$(sed -n 1p <<< "$INFO")
PLAN16=$(sed -n 2p <<< "$INFO")
BIGGEST=$(sed -n 3p <<< "$INFO")
NROWS=$(sed -n 4p <<< "$INFO")
[ "$N" -ge 1 ] || { echo "ABORT: the plan lists no task"; exit 3; }
mkdir -p "$LOGDIR"
SUBMISSIONS="$LOGDIR/submissions.txt"

echo "########################################################################"
echo "# mode        : $MODE  ($([ "$DRYRUN" = "1" ] && echo 'DRY RUN, nothing is submitted' || echo SUBMIT))"
echo "# plan        : $PLAN  sha256 $PLAN16"
echo "# tasks       : $N  ($NROWS simulations); largest: $BIGGEST"
echo "# bank        : $BANK_ROOT"
echo "# env         : $ENV_NAME (activated inside the jobs)"
echo "# logs        : $LOGDIR"
echo "# the array's exit says nothing (D-006): read the gate's log"
echo "########################################################################"

JOB_ID=""
if [ "$MODE" = "test" ] || [ "$MODE" = "array" ]; then
    if [ "$MODE" = "test" ]; then
        [ "$TEST_INDEX" -ge 0 ] 2>/dev/null && [ "$TEST_INDEX" -lt "$N" ] \
            || { echo "ABORT: TEST_INDEX=$TEST_INDEX is not a task of the plan (0-$((N - 1)))"; exit 3; }
        CMD=("$QSUB" -N bank_test -q "$QUEUE" -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
             -l "walltime=${WALLTIME:-02:00:00}" -j oe -o "$LOGDIR/bank_test_${TEST_INDEX}.log"
             -v "PLAN=${PLAN},TASK_INDEX=${TEST_INDEX},ENV_NAME=${ENV_NAME}" run_bank_task.pbs)
        LOG_HINT="$LOGDIR/bank_test_${TEST_INDEX}.log"
    else
        [ -n "$WALLTIME" ] || { echo "ABORT: WALLTIME=hh:mm:ss (or WALLTIME=auto, from the test task) is required for the array"; exit 3; }
        if [ "$WALLTIME" = "auto" ]; then
            WALLTIME=$(python3 - "$PLAN" "$BANK_ROOT" "$TEST_INDEX" <<'EOF'
import datetime as dt, hashlib, json, math, os, sys
plan_path, bank_root, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
blob = open(plan_path, "rb").read()
plan = json.loads(blob.decode("ascii"))
m_path = os.path.join(bank_root, "tasks", "task_%04d.json" % idx)
if not os.path.isfile(m_path):
    print("ERROR WALLTIME=auto needs the test task's marker %s (run the test first)" % m_path)
    sys.exit(1)
m = json.load(open(m_path))
if m.get("plan_sha256") != hashlib.sha256(blob).hexdigest():
    print("ERROR the marker %s belongs to another plan" % m_path)
    sys.exit(1)
secs = (dt.datetime.fromisoformat(m["finished"]) - dt.datetime.fromisoformat(m["started"])).total_seconds()
per = max(secs, 1.0) / max(1, m["totals"]["read"])
big = max(t["n_iters"] for t in plan["tasks"])
mins = max(30, int(math.ceil((2.0 * per * big + 600.0) / 900.0)) * 15)
print("%02d:%02d:00" % (mins // 60, mins % 60))
sys.stderr.write("[launch] WALLTIME=auto: test task %d read %d simulations in %.0f s; the largest "
                 "task holds %d: 2 x %.0f s + 10 min, in quarter hours, at least 30 min -> "
                 "%02d:%02d:00\n" % (idx, m["totals"]["read"], secs, big, per * big,
                                      mins // 60, mins % 60))
EOF
) || { echo "ABORT: $WALLTIME"; exit 3; }
        fi
        [[ "$WALLTIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "ABORT: WALLTIME=$WALLTIME is not hh:mm:ss"; exit 3; }
        if [ "$N" -gt 1 ]; then
            CMD=("$QSUB" -N bank_task -q "$QUEUE" -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
                 -l "walltime=${WALLTIME}" -J "0-$((N - 1))%${CONCURRENCY}" -j oe
                 -o "$LOGDIR/bank_task_^array_index^.log"
                 -v "PLAN=${PLAN},ENV_NAME=${ENV_NAME}" run_bank_task.pbs)
        else
            # one task: PBS Pro rejects a single-element array; a plain job runs task 0
            CMD=("$QSUB" -N bank_task -q "$QUEUE" -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
                 -l "walltime=${WALLTIME}" -j oe -o "$LOGDIR/bank_task_0.log"
                 -v "PLAN=${PLAN},TASK_INDEX=0,ENV_NAME=${ENV_NAME}" run_bank_task.pbs)
        fi
        LOG_HINT="$LOGDIR/bank_task_<index>.log"
    fi
    echo "  (cd $HERE && ${CMD[*]})"
    if [ "$DRYRUN" != "1" ]; then
        JOB_ID=$("${CMD[@]}") || { echo "ABORT: qsub failed"; exit 2; }
        echo "  job   : $JOB_ID"
        echo "$(date -Is) $MODE $JOB_ID $PLAN16 $N" >> "$SUBMISSIONS"
    fi
    if [ "$MODE" = "test" ]; then
        [ "$DRYRUN" = "1" ] && { echo; echo "(DRYRUN -- nothing was submitted.)"; exit 0; }
        echo; echo "watch:  qstat -u \$USER   then:  tail -n 5 $LOG_HINT   (last line: TASK $TEST_INDEX OK)"
        exit 0
    fi
fi

# --- the gate job ------------------------------------------------------------
GATE=("$QSUB" -N bank_gate -q "$QUEUE" -l "select=1:ncpus=${GATE_NCPUS}:mem=${GATE_MEM}"
      -l "walltime=${GATE_WALLTIME}" -j oe -o "$LOGDIR/bank_gate.log")
if [ "$MODE" = "array" ]; then
    if [ "$DRYRUN" = "1" ]; then
        echo "  (cd $HERE && ${GATE[*]} -W depend=afterok:<array id> -v PLAN=${PLAN},ENV_NAME=${ENV_NAME} run_bank_gate.pbs)"
        echo; echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"; exit 0
    fi
    GATE+=(-W "depend=afterok:${JOB_ID}")
fi
GATE+=(-v "PLAN=${PLAN},ENV_NAME=${ENV_NAME}" run_bank_gate.pbs)
echo "  (cd $HERE && ${GATE[*]})"
[ "$DRYRUN" = "1" ] && { echo; echo "(DRYRUN -- nothing was submitted.)"; exit 0; }
gate=$("${GATE[@]}") || { echo "ABORT: the gate's qsub failed"; exit 2; }
echo "  gate  : $gate"
echo "$(date -Is) gate $gate $PLAN16" >> "$SUBMISSIONS"
echo
echo "watch:   qstat -u \$USER"
echo "then:    tail -n 3 $LOGDIR/bank_gate.log"
echo "         PASS = a line 'wrote $BANK_ROOT/BANK_RECORD.json  sha256 <16 hex>'; a REFUSED lists"
echo "         every problem as '[gate] FAIL <task>: ...'; resubmit those tasks (TEST_INDEX=<i> ... test),"
echo "         then: bash launch_bank.sh gate"
