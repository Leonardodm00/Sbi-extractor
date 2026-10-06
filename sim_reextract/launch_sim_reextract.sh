#!/bin/bash
# =============================================================================
# launch_sim_reextract.sh -- Stage C, C8: re-run the virtual-MEA detection
# (Astro-Neuron-Network's process_campaign.py) over the existing raw DUP15HD
# simulations into a NEW root, as one PBS array plus one dependent gate job.
# The sim-arm analogue of extractor/launch_stage_d.sh.
#
#     cd ~/repos/Sbi-extractor/sim_reextract && conda activate sbi_export
#     DRYRUN=1 bash launch_sim_reextract.sh plan    # the plan only; always safe
#     DRYRUN=1 bash launch_sim_reextract.sh test    # + the test task's qsub line
#     bash launch_sim_reextract.sh test             # submit ONE task (TEST_INDEX, default 0)
#     WALLTIME=hh:mm:ss bash launch_sim_reextract.sh array   # the array + the gate job
#     bash launch_sim_reextract.sh gate             # the gate job alone (after a resume)
#
# ORDER. plan -> test -> read the test task's log -> array (RESUME=1, since
# the test task's output is in the root) -> the gate's PASS line. The test
# task's log, ~/c8_test.o<jobid>, must show
#     [mea-array] env activated: .../envs/sbi_export/bin/python3
#     [mea-array] python   : 3.11.x (...)   numpy : ...   scipy : ...
#     [mea] N topos, a/b iters processed in X s
# and NOT the WARNING "putting its bin/ first on PATH" (then scipy cannot
# import on davinci: HPC_PATHS.md 7.1; the job script's conda lookup is
# the 2026-10-01 one, which knows davinci's conda base). From "X s" for
# that task's iterations and the plan's largest task, choose WALLTIME.
#
# WHAT IT DOES
#   1. freezes the cohort manifest of record WITH its .sha256 into
#      $ARTIFACTS_DIR/cohort_manifest/ (a space-free path; HPC_PATHS.md 3b),
#      verifying the digest, once;
#   2. runs sim_reextract_plan.py (refusals: see its header), which writes
#      plan.json and tasks.tsv here;
#   3. submits ANN's submit_mea_array.sh from the TOOLS folder, one array
#      member per line of tasks.tsv, with CONDA_ENV=$ENV_NAME and the plan's
#      EXTRA_ARGS (n_side from the cohort manifest, pitch, edge, fs);
#   4. submits run_sim_reextract_gate.pbs with depend=afterok:<array>, which
#      is ORDERING ONLY on this PBS (measured 2026-09-21): the gate is what
#      refuses a partial run. Read out/sim_reextract_gate.log.
#
# ENVIRONMENT (defaults are davinci's)
#   COHORT_MANIFEST  the manifest of record (default: extracted_v2's)
#   SIM_MAIN         raw simulations      OUT_ROOT  Outputs_v2 (D-014)
#   TOOLS            ANN/MEA_analysis (process_campaign.py, submit_mea_array.sh)
#   CAMPAIGN_GLOB    campaign_cadex_rho1300v*   (D-025)
#   ENV_NAME         sbi_export (D-023)         QUEUE cpu   NCPUS 48
#   CONCURRENCY      20                          WALLTIME  required for `array`;
#                                                06:00:00 for `test`
#   RESUME=1         the root already holds detections: keep complete tasks
#   TEST_INDEX       which line of tasks.tsv the test task runs (default 0)
#   PLAN_ARGS        extra flags for sim_reextract_plan.py (e.g. --allow-mixed-contract)
#   DRYRUN=1         print, submit nothing
#   COHORT_TAG       another cohort (2026-10-06): sources profiles/<tag>.sh, which
#                    sets the defaults above for that cohort (an explicit variable
#                    still wins), the plan's geometry flags, and its OWN plan folder
#                    (<tag>/plan.json, tasks.tsv), frozen-manifest folder, job names,
#                    gate log (out/sim_reextract_gate_<tag>.log) and submissions file,
#                    so that two cohorts' runs share no file. Unset: DUP15HD (C8).
#                    COHORT_TAG=giulia is the Giulia project's G2.
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
HERE="$(pwd -P)"

MODE="${1:-}"
case "$MODE" in
    plan|test|array|gate) ;;
    *) sed -n '2,60p' "$0"; exit 2 ;;
esac
# shellcheck source=/dev/null
source ../env.sh

# --- 0. the cohort profile (COHORT_TAG; unset = DUP15HD, as before) ----------
COHORT_TAG="${COHORT_TAG:-}"
PROFILE_GEOM_ARGS=""; PROFILE_WORK=""; PROFILE_FROZEN="cohort_manifest"; PROFILE_JOB="c8"
if [ -n "$COHORT_TAG" ]; then
    case "$COHORT_TAG" in *[!a-z0-9_]*) echo "ABORT: COHORT_TAG=$COHORT_TAG: lower-case letters, digits and _ only"; exit 2 ;; esac
    PROFILE="$HERE/profiles/${COHORT_TAG}.sh"
    [ -f "$PROFILE" ] || { echo "ABORT: no profile for COHORT_TAG=$COHORT_TAG ($PROFILE)"; exit 2; }
    # shellcheck source=/dev/null
    source "$PROFILE"
    { [ -n "$PROFILE_WORK" ] && [ "$PROFILE_JOB" != "c8" ] && [ "$PROFILE_FROZEN" != "cohort_manifest" ]; } \
        || { echo "ABORT: $PROFILE must set its own PROFILE_WORK, PROFILE_JOB and PROFILE_FROZEN"; exit 2; }
    echo "[launch] cohort profile: $COHORT_TAG ($PROFILE)"
fi
WORK="$HERE${PROFILE_WORK:+/$PROFILE_WORK}"
GATE_LOG="out/sim_reextract_gate${COHORT_TAG:+_$COHORT_TAG}.log"
SUBMISSIONS="out/submissions${COHORT_TAG:+_$COHORT_TAG}.txt"

COHORT_MANIFEST="${COHORT_MANIFEST:-/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted_v2/cohort_manifest.json}"
SIM_MAIN="${SIM_MAIN:-/davinci-1/home/ldellamea/ANN/Phenomenological/Main}"
OUT_ROOT="${OUT_ROOT:-/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2}"
TOOLS="${TOOLS:-/davinci-1/home/ldellamea/ANN/MEA_analysis}"
CAMPAIGN_GLOB="${CAMPAIGN_GLOB:-campaign_cadex_rho1300v*}"
ENV_NAME="${ENV_NAME:-sbi_export}"
QUEUE="${QUEUE:-cpu}"
NCPUS="${NCPUS:-48}"
CONCURRENCY="${CONCURRENCY:-20}"
WALLTIME="${WALLTIME:-}"
RESUME="${RESUME:-0}"
TEST_INDEX="${TEST_INDEX:-0}"
PLAN_ARGS="${PLAN_ARGS:-}"
DRYRUN="${DRYRUN:-0}"
QSUB="${QSUB:-qsub}"            # the smoke test substitutes a fake

for v in OUT_ROOT TOOLS SIM_MAIN; do
    case "${!v}" in *[[:space:]]*) echo "ABORT: $v contains whitespace, which qsub -v cannot carry: ${!v}"; exit 3 ;; esac
done
[ -d "$TOOLS" ] || { echo "ABORT: TOOLS not found: $TOOLS"; exit 2; }
[ -f "$TOOLS/submit_mea_array.sh" ] && [ -f "$TOOLS/process_campaign.py" ] \
    || { echo "ABORT: $TOOLS lacks submit_mea_array.sh / process_campaign.py"; exit 2; }
[ -f "$TOOLS/eap_library.npz" ] || { echo "ABORT: no eap_library.npz in $TOOLS (the launcher would build a new one silently; the record must name the one used)"; exit 3; }

# --- 1. the frozen cohort manifest, with its sidecar ------------------------
FROZEN_DIR="$ARTIFACTS_DIR/$PROFILE_FROZEN"
FROZEN="$FROZEN_DIR/cohort_manifest.json"
if [ ! -f "$FROZEN" ]; then
    [ -f "$COHORT_MANIFEST" ] || { echo "ABORT: cohort manifest not found: $COHORT_MANIFEST"; exit 2; }
    [ -f "$COHORT_MANIFEST.sha256" ] || { echo "ABORT: $COHORT_MANIFEST has no .sha256 sidecar; refusing to freeze an unchecked copy"; exit 3; }
    want=$(cut -d' ' -f1 "$COHORT_MANIFEST.sha256")
    got=$(sha256sum "$COHORT_MANIFEST" | cut -d' ' -f1)
    [ "$want" = "$got" ] || { echo "ABORT: $COHORT_MANIFEST does not match its sidecar ($got vs $want)"; exit 3; }
    mkdir -p "$FROZEN_DIR"
    cp "$COHORT_MANIFEST" "$FROZEN" && cp "$COHORT_MANIFEST.sha256" "$FROZEN.sha256" \
        || { echo "ABORT: could not freeze the manifest into $FROZEN_DIR"; exit 2; }
    echo "[launch] froze $COHORT_MANIFEST -> $FROZEN  (sha256 ${got:0:16})"
else
    got=$(sha256sum "$FROZEN" | cut -d' ' -f1)
    want=$(cut -d' ' -f1 "$FROZEN.sha256" 2>/dev/null || echo none)
    [ "$want" = "$got" ] || { echo "ABORT: the frozen $FROZEN does not match its sidecar"; exit 3; }
    # a frozen copy of ANOTHER manifest (2026-10-06): one cohort planned with the
    # other's geometry, silently, before this check
    if [ -f "$COHORT_MANIFEST" ]; then
        cm=$(sha256sum "$COHORT_MANIFEST" | cut -d' ' -f1)
        [ "$cm" = "$got" ] || { echo "ABORT: the frozen $FROZEN (sha256 ${got:0:16}) is not $COHORT_MANIFEST (sha256 ${cm:0:16}); move $FROZEN_DIR aside only if the manifest of record has changed"; exit 3; }
    fi
    echo "[launch] frozen manifest: $FROZEN  (sha256 ${got:0:16})"
fi

# --- 2. the plan ------------------------------------------------------------
resume_flag=""
[ "$RESUME" = "1" ] && resume_flag="--resume"
mkdir -p "$WORK"
# shellcheck disable=SC2086
python3 sim_reextract_plan.py --cohort-manifest "$FROZEN" --sim-main "$SIM_MAIN" \
    --out-root "$OUT_ROOT" --tools "$TOOLS" --campaign-glob "$CAMPAIGN_GLOB" \
    --plan-out "$WORK/plan.json" --tasks-out "$WORK/tasks.tsv" $resume_flag $PROFILE_GEOM_ARGS $PLAN_ARGS \
    || { echo "ABORT: the plan was refused (above)"; exit 3; }
N=$(wc -l < "$WORK/tasks.tsv")
EXTRA_ARGS=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['extra_args'])" "$WORK/plan.json")
[ "$MODE" = "plan" ] && { echo "[launch] plan only: $N task(s) in tasks.tsv; nothing submitted"; exit 0; }

mkdir -p out
# --- 3. the test task or the array ------------------------------------------
if [ "$MODE" = "test" ] || [ "$MODE" = "array" ]; then
    [ "$N" -ge 1 ] || { echo "ABORT: tasks.tsv is empty (every task complete?); nothing to submit"; exit 3; }
    if [ "$MODE" = "test" ]; then
        [ "$TEST_INDEX" -ge 0 ] && [ "$TEST_INDEX" -lt "$N" ] \
            || { echo "ABORT: TEST_INDEX=$TEST_INDEX is not a line of tasks.tsv (0-$((N - 1)))"; exit 3; }
        sed -n "$((TEST_INDEX + 1))p" "$WORK/tasks.tsv" > "$WORK/tasks_test.tsv"
        MANIFEST="$WORK/tasks_test.tsv"; WT="${WALLTIME:-06:00:00}"; JOBNAME="${PROFILE_JOB}_test"
        echo "[launch] test task: line $TEST_INDEX of tasks.tsv -> $(cut -f1 "$WORK/tasks_test.tsv")"
    else
        [ -n "$WALLTIME" ] || { echo "ABORT: WALLTIME=hh:mm:ss is required for the array (from the test task's time per iteration and the plan's largest task)"; exit 3; }
        MANIFEST="$WORK/tasks.tsv"; WT="$WALLTIME"; JOBNAME="${PROFILE_JOB}_mea"
    fi
    QSUB_V="MANIFEST=${MANIFEST},LIB=${TOOLS}/eap_library.npz,CONDA_ENV=${ENV_NAME},EXTRA_ARGS=${EXTRA_ARGS}"
    if [ "$MODE" = "array" ] && [ "$N" -gt 1 ]; then
        ARR_CMD=("$QSUB" -N "$JOBNAME" -q "$QUEUE" -l "select=1:ncpus=${NCPUS},walltime=${WT}"
                 -J "0-$((N - 1))%${CONCURRENCY}" -v "$QSUB_V" "$TOOLS/submit_mea_array.sh")
    else
        # one task: PBS Pro rejects a single-element array; a plain job reads line 1
        ARR_CMD=("$QSUB" -N "$JOBNAME" -q "$QUEUE" -l "select=1:ncpus=${NCPUS},walltime=${WT}"
                 -v "$QSUB_V" "$TOOLS/submit_mea_array.sh")
    fi
    echo "########################################################################"
    echo "# mode        : $MODE  ($([ "$DRYRUN" = "1" ] && echo 'DRY RUN, nothing is submitted' || echo SUBMIT))"
    echo "# tasks       : $N   out root: $OUT_ROOT"
    echo "# tools       : $TOOLS"
    echo "# env         : $ENV_NAME (CONDA_ENV, activated inside the job)"
    echo "# extra args  : $EXTRA_ARGS"
    echo "# walltime    : $WT   ncpus $NCPUS   queue $QUEUE   concurrency $CONCURRENCY"
    echo "# logs        : ~/${JOBNAME}.o<jobid> (#PBS -k eo)"
    echo "# the array's exit says nothing: read $GATE_LOG"
    echo "########################################################################"
    echo "  (cd $TOOLS && ${ARR_CMD[*]})"
    [ "$DRYRUN" = "1" ] && [ "$MODE" = "test" ] && { echo; echo "(DRYRUN -- nothing was submitted.)"; exit 0; }
    if [ "$DRYRUN" != "1" ]; then
        arr=$(cd "$TOOLS" && "${ARR_CMD[@]}") || { echo "ABORT: qsub failed"; exit 2; }
        echo "  job   : $arr"
        echo "$(date -Is) $MODE $arr $OUT_ROOT $N" >> "$SUBMISSIONS"
        [ "$MODE" = "test" ] && { echo; echo "watch:  qstat -u \$USER   then read ~/${JOBNAME}.o${arr%%.*}* (see the header of this script)"; exit 0; }
    fi
fi

# --- 4. the gate job --------------------------------------------------------
# a tagged cohort's gate gets its own job name and log (-o overrides the
# script's #PBS -o); untagged, the command is the C8 one, unchanged
GATE_CMD=("$QSUB")
[ -n "$COHORT_TAG" ] && GATE_CMD+=(-N "${PROFILE_JOB}_gate" -o "$HERE/$GATE_LOG")
if [ "$MODE" = "array" ]; then
    if [ "$DRYRUN" = "1" ]; then
        echo "  ${GATE_CMD[*]} -W depend=afterok:<array id> -v PLAN=${WORK}/plan.json,ENV_NAME=${ENV_NAME} run_sim_reextract_gate.pbs"
        echo; echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"; exit 0
    fi
    GATE_CMD+=(-W "depend=afterok:${arr}")
fi
GATE_CMD+=(-v "PLAN=${WORK}/plan.json,ENV_NAME=${ENV_NAME}" run_sim_reextract_gate.pbs)
echo "  ${GATE_CMD[*]}"
[ "$DRYRUN" = "1" ] && { echo; echo "(DRYRUN -- nothing was submitted.)"; exit 0; }
gate=$("${GATE_CMD[@]}") || { echo "ABORT: the gate's qsub failed"; exit 2; }
echo "  gate  : $gate"
echo "$(date -Is) gate $gate $OUT_ROOT" >> "$SUBMISSIONS"
echo
echo "watch:   qstat -u \$USER"
echo "then:    grep -h 'wrote\|REFUSED\|FAIL' $GATE_LOG"
