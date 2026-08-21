#!/bin/bash
##########################################################################
# launch_sweep_exports.sh -- submit ONE SBI export job per SWEEP TASK.
#
# WHY THIS EXISTS (the bug it replaces)
#
# launch_all_campaigns.sh resolves the export directory with
#
#     hit=$(find "${camp}" -name 'mea_iter_*.npz' | head -1)
#     mea_dir=$(dirname "$(dirname "${hit}")")
#
# For a tree shaped
#
#     <MEA_ROOT>/campaign_<TAG>/sweep_cpu_task<NNNN>/topo_<KKKKK>/mea_iter_*.npz
#
# that lands on sweep_cpu_task0000 and stops. iter_campaign_records() then
# globs topo_* directly under it, so EVERY OTHER sweep task in the campaign
# is dropped -- silently, with no warning and no error, and with a perfectly
# healthy-looking Parquet shard as the output.
#
# That is not merely a smaller n_sim. If the task index partitions the
# parameter sweep (which is the usual reason to have tasks at all), the
# exported rows are a BIASED draw, and the simulated arm is then no longer a
# sample from the prior predictive p(x | M). The misspecification gate's
# entire null is "resample the prior predictive", so a biased simulated arm
# does not weaken the test -- it invalidates it, in a direction nothing
# downstream can detect.
#
# This launcher enumerates (campaign, sweep task) pairs explicitly and
# submits one job for each, with a CAMPAIGN_ID that is unique across both
# levels so the provenance column can still identify the source.
#
# USAGE
#     ./launch_sweep_exports.sh <CKPT> <MEA_ROOT> <SIM_ROOT> <OUT_ROOT> [GLOB]
#
# EXAMPLE
#     export DSN_MAIN_DIR=$HOME/dsn_main          # see NOTE ON SPACES below
#     export SIM_MAIN_DIR=/davinci-1/home/ldellamea/ANN/Phenomenological/Main
#     DRYRUN=1 ./launch_sweep_exports.sh \
#         $HOME/dsn_main/out/refit_mea_A_best/checkpoints/seed_0/best.pt \
#         /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs \
#         /davinci-1/home/ldellamea/ANN/Phenomenological/Main \
#         /davinci-1/home/ldellamea/ANN/SBI_export \
#         'campaign_cadex_rho1300v*'
#
# ALWAYS run with DRYRUN=1 first. It prints every qsub line and the whole
# inventory without submitting anything.
#
# NOTE ON SPACES. PBS passes -v as a COMMA-SEPARATED list and does not
# survive a path containing whitespace. The DSN lives under
# ".../Deep Summary Network/...", which contains one. This script REFUSES to
# submit when DSN_MAIN_DIR or CKPT contains whitespace, and tells you to make
# a symlink:
#
#     ln -s "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/Main" \
#           "$HOME/dsn_main"
#
# Refusing is deliberate: the alternative failure mode is a job that starts,
# truncates the path at the space, and dies with a confusing "path does not
# exist" after sitting in the queue.
#
# ENVIRONMENT
#     DSN_MAIN_DIR  REQUIRED. <Deep-Summary-Network>/Main (or a symlink to it)
#     SIM_MAIN_DIR  REQUIRED. the simulator repo dir sbi_labels imports from
#     GLOB          campaign name pattern (default 'campaign_*')
#     SELECT        PBS select line (default select=1:ncpus=8:mem=32gb)
#     WALLTIME      default 02:00:00
#     ENV_NAME      conda environment (default sbi_export)
#     MAX_RECORDS   stop after N sims per TASK -- use for a first pass
#     SIMTIME       override T [s] for every task. Normally each task's own
#                   job_args.json is used and merely REPORTED here.
#     LABEL_AXES    frozen label_axes.json from preflight_label_axes.py,
#                   passed through to every job so that ALL shards share one
#                   theta column set. Unset => the worker uses its own default
#                   (<repo>/artifacts/label_axes.json) and FAILS if absent.
#     TRIM_HEAD_S   discard the first N seconds of every simulated trace as
#                   burn-in, before windowing. Requires trim_head.patch.
#                   With T = 200 s, W = 180 s and TRIM_HEAD_S=20 the retained
#                   interval [20, 200) is EXACTLY one window.
#     NOCOUNT=1     skip counting mea_iter files (faster on a busy filesystem)
#     DRYRUN=1      print, do not submit
##########################################################################

set -uo pipefail

if [ "$#" -lt 4 ]; then
    sed -n '2,78p' "$0"
    exit 2
fi

CKPT="$1"
MEA_ROOT="$2"
SIM_ROOT="$3"
OUT_ROOT="$4"
GLOB="${5:-${GLOB:-campaign_*}}"

SELECT="${SELECT:-select=1:ncpus=8:mem=32gb}"
WALLTIME="${WALLTIME:-02:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_sbi_export.sh"

# --- guard 0: the whitespace trap ----------------------------------------
has_space() {
    case "$1" in
        *[[:space:]]*) return 0 ;;
        *)             return 1 ;;
    esac
}

for pair in "CKPT:${CKPT}" "DSN_MAIN_DIR:${DSN_MAIN_DIR:-}" \
            "SIM_MAIN_DIR:${SIM_MAIN_DIR:-}" "OUT_ROOT:${OUT_ROOT}" \
            "LABEL_AXES:${LABEL_AXES:-}"; do
    nm="${pair%%:*}"; val="${pair#*:}"
    if [ -n "${val}" ] && has_space "${val}"; then
        echo "ERROR: ${nm} contains whitespace:" >&2
        echo "         ${val}" >&2
        echo "" >&2
        echo "       qsub -v cannot carry it. Make a symlink and use that:" >&2
        echo "         ln -s \"${val}\" \"\$HOME/$(echo "${nm}" | tr 'A-Z' 'a-z')_link\"" >&2
        exit 5
    fi
done

# --- guard 1: required environment ---------------------------------------
if [ -z "${DSN_MAIN_DIR:-}" ] || [ -z "${SIM_MAIN_DIR:-}" ]; then
    echo "ERROR: export DSN_MAIN_DIR and SIM_MAIN_DIR before running." >&2
    echo "       submit_sbi_export.sh defaults them to \$HOME/repos/..., which" >&2
    echo "       is NOT where they live here. A wrong SIM_MAIN_DIR silently" >&2
    echo "       imports a different parameter registry and therefore a" >&2
    echo "       different prior box." >&2
    exit 6
fi

for p in "${CKPT}" "${MEA_ROOT}" "${SIM_ROOT}" "${SUBMIT}" \
         "${DSN_MAIN_DIR}" "${SIM_MAIN_DIR}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: does not exist: ${p}" >&2
        exit 3
    fi
done

# --- guard: the frozen label axes ----------------------------------------
# Checked HERE as well as in the worker, because this launcher fans out one
# job per task dir: without the check the mistake is discovered once per
# submitted job instead of once, before anything is queued. Every shard in a
# campaign set MUST be built from the SAME axis file, or the resulting theta
# matrices differ in width and column meaning and cannot be concatenated.
if [ -n "${LABEL_AXES:-}" ] && [ "${LABEL_AXES}" != "none" ] \
   && [ ! -f "${LABEL_AXES}" ]; then
    echo "ERROR: LABEL_AXES points at a missing file: ${LABEL_AXES}" >&2
    echo "       Run preflight_label_axes.py ONCE over all campaigns first." >&2
    exit 7
fi
mkdir -p "${OUT_ROOT}"

if [ "${DRYRUN:-0}" = "1" ]; then
    MODE_STR="DRY RUN (nothing is submitted)"
else
    MODE_STR="SUBMIT"
fi

echo "########################################################################"
echo "# checkpoint : ${CKPT}"
echo "# mea root   : ${MEA_ROOT}"
echo "# sim root   : ${SIM_ROOT}"
echo "# out root   : ${OUT_ROOT}"
echo "# glob       : ${GLOB}"
echo "# dsn main   : ${DSN_MAIN_DIR}"
echo "# label axes : ${LABEL_AXES:-(worker default: <repo>/artifacts/label_axes.json)}"
echo "# sim main   : ${SIM_MAIN_DIR}"
echo "# resources  : ${SELECT}  walltime=${WALLTIME}"
echo "# mode       : ${MODE_STR}"
echo "########################################################################"
echo ""

shopt -s nullglob
CAMPS=("${MEA_ROOT}"/${GLOB})
shopt -u nullglob

if [ "${#CAMPS[@]}" -eq 0 ]; then
    echo "ERROR: no directories matched ${MEA_ROOT}/${GLOB}" >&2
    ls -1 "${MEA_ROOT}" | head -20 >&2
    exit 4
fi

n_sub=0; n_skip=0; n_task=0; n_iter_total=0
SIMTIMES=""
declare -a SKIPPED=()

for camp in "${CAMPS[@]}"; do
    [ -d "${camp}" ] || continue
    cname="$(basename "${camp}")"

    # Enumerate the SWEEP TASK level. Two layouts are handled: task dirs
    # under the campaign (the normal case here), or topo_* directly under
    # the campaign (a single-task campaign).
    shopt -s nullglob
    TASKS=("${camp}"/*/)
    shopt -u nullglob

    declare -a TASKDIRS=()
    if [ -d "${camp}/topo_00000" ] || compgen -G "${camp}/topo_*" >/dev/null; then
        TASKDIRS=("${camp}")
    else
        for t in "${TASKS[@]}"; do
            t="${t%/}"
            if compgen -G "${t}/topo_*" >/dev/null; then
                TASKDIRS+=("${t}")
            fi
        done
    fi

    if [ "${#TASKDIRS[@]}" -eq 0 ]; then
        echo "SKIP  ${cname}: no topo_* at the campaign or task level"
        SKIPPED+=("${cname}: no topo_*")
        n_skip=$((n_skip + 1)); continue
    fi

    echo "campaign ${cname}: ${#TASKDIRS[@]} sweep task(s)"

    for tdir in "${TASKDIRS[@]}"; do
        tname="$(basename "${tdir}")"
        if [ "${tdir}" = "${camp}" ]; then
            tag="${cname}"
            sim_task="${SIM_ROOT}/${cname}"
        else
            tag="${cname}__${tname}"
            sim_task="${SIM_ROOT}/${cname}/${tname}"
        fi
        n_task=$((n_task + 1))

        if ! compgen -G "${tdir}/topo_*/mea_iter_*.npz" >/dev/null; then
            echo "  SKIP  ${tname}: no mea_iter_*.npz (process_campaign.py not run)"
            SKIPPED+=("${tag}: no detections")
            n_skip=$((n_skip + 1)); continue
        fi
        if [ ! -d "${sim_task}" ]; then
            echo "  SKIP  ${tname}: no sweep dir at ${sim_task}"
            SKIPPED+=("${tag}: sweep dir absent")
            n_skip=$((n_skip + 1)); continue
        fi
        # iter_*.npz but NOT mea_iter_*.npz: the topology block
        # (p0_conn / d0_conn / beta_conn) exists only in the sweep output.
        ihit=$(find "${sim_task}" -maxdepth 2 -name 'iter_*.npz' \
               -not -name 'mea_iter_*.npz' -type f 2>/dev/null | head -1)
        if [ -z "${ihit}" ]; then
            echo "  SKIP  ${tname}: no iter_*.npz under ${sim_task}"
            SKIPPED+=("${tag}: no sweep output -> no 27-D label")
            n_skip=$((n_skip + 1)); continue
        fi

        stem="${OUT_ROOT}/sbi_${tag}"
        if [ -f "${stem}.parquet" ]; then
            echo "  SKIP  ${tname}: ${stem}.parquet exists (delete to redo)"
            SKIPPED+=("${tag}: already exported")
            n_skip=$((n_skip + 1)); continue
        fi

        # Report the declared duration. NEVER read it from the npz: the MEA
        # stage computes simtime as ceil(last detected spike), which for a
        # quiet run is far below the requested duration.
        st="(no job_args.json)"
        if [ -f "${sim_task}/job_args.json" ]; then
            st=$(python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(d.get('simtime','(absent)'))
except Exception as e:
    print('(unreadable: %s)' % type(e).__name__)
" "${sim_task}/job_args.json" 2>/dev/null || echo "(error)")
        fi
        SIMTIMES="${SIMTIMES}${st}"$'\n'

        n_iters="?"
        if [ "${NOCOUNT:-0}" != "1" ]; then
            n_iters=$(find "${tdir}" -name 'mea_iter_*.npz' -type f 2>/dev/null | wc -l)
            n_iter_total=$((n_iter_total + n_iters))
        fi

        VARS="CKPT=${CKPT},CAMPAIGN=${sim_task},MEA_OUT=${tdir}"
        VARS="${VARS},OUT=${stem},CAMPAIGN_ID=${tag}"
        VARS="${VARS},DSN_MAIN_DIR=${DSN_MAIN_DIR},SIM_MAIN_DIR=${SIM_MAIN_DIR}"
        [ -n "${MAX_RECORDS:-}" ] && VARS="${VARS},MAX_RECORDS=${MAX_RECORDS}"
        [ -n "${ENV_NAME:-}" ]    && VARS="${VARS},ENV_NAME=${ENV_NAME}"
        [ -n "${SIMTIME:-}" ]     && VARS="${VARS},SIMTIME=${SIMTIME}"
        [ -n "${TRIM_HEAD_S:-}" ] && VARS="${VARS},TRIM_HEAD_S=${TRIM_HEAD_S}"
        [ -n "${LABEL_AXES:-}" ]  && VARS="${VARS},LABEL_AXES=${LABEL_AXES}"

        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "  DRY   ${tname}  sims=${n_iters}  job_args.simtime=${st}"
            echo "          CAMPAIGN = ${sim_task}"
            echo "          MEA_OUT  = ${tdir}"
            echo "          OUT      = ${stem}"
            echo "          qsub -l ${SELECT} -l walltime=${WALLTIME} -N x_${tag} -v ${VARS} ${SUBMIT}"
        else
            jid=$(qsub -l "${SELECT}" -l "walltime=${WALLTIME}" \
                       -N "x_${tag}" -v "${VARS}" "${SUBMIT}")
            echo "  SUBMIT ${tname}  sims=${n_iters}  simtime=${st}  -> ${jid}"
        fi
        n_sub=$((n_sub + 1))
    done
done

echo ""
echo "======================================================================"
echo "sweep tasks seen : ${n_task}"
echo "submitted        : ${n_sub}"
echo "skipped          : ${n_skip}"
if [ "${NOCOUNT:-0}" != "1" ]; then
    echo "simulations      : ${n_iter_total}  (mea_iter_*.npz across submitted tasks)"
    echo "                   the gate wants n_sim >= 4 * n_real; with 35 cultures"
    echo "                   x 9 subregions x 6 windows that is n_real = 1890,"
    echo "                   so the bar is 7560 rows."
fi

# A mixed simtime across tasks means a mixed-duration simulated arm, which
# displaces the two embedding clouds by a duration artefact before any
# biology enters. Surface it rather than averaging over it.
echo ""
echo "declared simtime values across submitted tasks:"
printf '%s' "${SIMTIMES}" | sort | uniq -c | sed 's/^/  /'
n_distinct=$(printf '%s' "${SIMTIMES}" | sort -u | grep -c . || true)
if [ "${n_distinct}" -gt 1 ]; then
    echo ""
    echo "  WARNING: more than one distinct simtime. Every simulated row must"
    echo "  cover the SAME wall-clock duration as every other, or the arms"
    echo "  differ by a duration artefact. Decide on one T and re-export the"
    echo "  odd ones, or exclude them."
fi

if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo ""
    echo "skipped detail:"
    for s in "${SKIPPED[@]}"; do echo "  - ${s}"; done
fi

if [ "${DRYRUN:-0}" = "1" ]; then
    echo ""
    echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"
else
    echo ""
    echo "Watch with:  qstat -u \$USER"
    echo ""
    echo "When all are done, verify ONE encoder produced every shard. A mixed"
    echo "encoder dataset is void and nothing downstream will notice:"
    echo "  for f in ${OUT_ROOT}/*.json; do python3 -c \"import json,sys;\\"
    echo "    d=json.load(open(sys.argv[1]));\\"
    echo "    print(d['embedding']['dsn_checkpoint_sha256'][:16], sys.argv[1])\" \"\$f\"; done \\"
    echo "    | sort | uniq -c -w16"
    echo "  (exactly ONE line must be printed)"
fi
