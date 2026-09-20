#!/bin/bash
##########################################################################
# launch_all_campaigns.sh -- submit one SBI export job per campaign.
#
# Covers a whole family of campaigns (e.g. campaign_cadex_rho1300v1 .. v8)
# in one command. Each campaign becomes an independent PBS job writing its
# own shard, because the shards are disjoint and there is nothing to gain
# from serialising them.
#
# LAYOUT AUTO-DETECTION. Two layouts are handled:
#
#   CO-LOCATED  the detections and the sweep output live in the same
#               campaign directory (process_campaign.py was run with --out
#               pointing into the campaign tree).
#
#   SPLIT       the detections are under <MEA_ROOT>/<campaign>/ and the
#               sweep output under <SIM_ROOT>/<campaign>/. This is the
#               normal case when MEA post-processing writes to its own
#               Outputs tree.
#
# The split case MUST be handled explicitly rather than guessed, because
# the topology block (p0_conn / d0_conn / beta_conn) is written ONLY into
# the sweep output's iter_*.npz. process_campaign.py does not copy it into
# mea_iter_*.npz. Exporting without it is impossible, not merely degraded.
#
# USAGE:
#     ./launch_all_campaigns.sh <CKPT> <MEA_ROOT> <OUT_ROOT> [SIM_ROOT] [GLOB]
#
# CO-LOCATED example:
#     ./launch_all_campaigns.sh \
#         /davinci-1/home/ldellamea/runs/mea_joint_full/checkpoints/best.pt \
#         /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs \
#         /davinci-1/home/ldellamea/ANN/SBI_export
#
# SPLIT example (sweep output in a different tree):
#     ./launch_all_campaigns.sh \
#         /davinci-1/home/ldellamea/runs/mea_joint_full/checkpoints/best.pt \
#         /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs \
#         /davinci-1/home/ldellamea/ANN/SBI_export \
#         /davinci-1/home/ldellamea/ANN/Campaigns
#
# ALWAYS DRY RUN FIRST -- prints the qsub lines without submitting:
#     DRYRUN=1 ./launch_all_campaigns.sh ...
#
# Environment overrides:
#     GLOB         campaign name pattern (default 'campaign_*')
#     SELECT       PBS select line (default select=1:ncpus=8:mem=32gb)
#     WALLTIME     default 02:00:00
#     ENV_NAME     conda env (default sbi_export)
#     MAX_RECORDS  stop after N sims per campaign -- use for a first pass
#     SIMTIME      override T [s]; normally read from job_args.json
##########################################################################

set -uo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,48p' "$0"
    exit 2
fi

CKPT="$1"
MEA_ROOT="$2"
OUT_ROOT="$3"
SIM_ROOT="${4:-}"                       # empty => co-located
GLOB="${5:-${GLOB:-campaign_*}}"

SELECT="${SELECT:-select=1:ncpus=8:mem=32gb}"
WALLTIME="${WALLTIME:-02:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_sbi_export.sh"

for p in "${CKPT}" "${MEA_ROOT}" "${SUBMIT}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: does not exist: ${p}" >&2
        exit 3
    fi
done
if [ -n "${SIM_ROOT}" ] && [ ! -d "${SIM_ROOT}" ]; then
    echo "ERROR: SIM_ROOT does not exist: ${SIM_ROOT}" >&2
    exit 3
fi
mkdir -p "${OUT_ROOT}"

echo "checkpoint : ${CKPT}"
echo "mea root   : ${MEA_ROOT}"
echo "sim root   : ${SIM_ROOT:-(co-located with mea root)}"
echo "out root   : ${OUT_ROOT}"
echo "glob       : ${GLOB}"
echo "resources  : ${SELECT}  walltime=${WALLTIME}"
echo ""

shopt -s nullglob
CAMPS=("${MEA_ROOT}"/${GLOB})
shopt -u nullglob

if [ "${#CAMPS[@]}" -eq 0 ]; then
    echo "ERROR: no directories matched ${MEA_ROOT}/${GLOB}" >&2
    echo "Contents:" >&2
    ls -1 "${MEA_ROOT}" | head -20 >&2
    exit 4
fi

n_sub=0; n_skip=0
declare -a SKIPPED=()

for camp in "${CAMPS[@]}"; do
    [ -d "${camp}" ] || continue
    name="$(basename "${camp}")"

    # --- locate the detections -------------------------------------------
    # The directory holding topo_*/mea_iter_*.npz is what --mea_out wants,
    # i.e. two levels above the npz itself.
    hit=$(find "${camp}" -name 'mea_iter_*.npz' -type f 2>/dev/null | head -1)
    if [ -z "${hit}" ]; then
        echo "SKIP  ${name}: no mea_iter_*.npz (process_campaign.py not run?)"
        SKIPPED+=("${name}: no detections")
        n_skip=$((n_skip + 1)); continue
    fi
    mea_dir=$(dirname "$(dirname "${hit}")")

    # --- locate the sweep output (the topology join source) --------------
    if [ -n "${SIM_ROOT}" ]; then
        search_root="${SIM_ROOT}/${name}"
        if [ ! -d "${search_root}" ]; then
            echo "SKIP  ${name}: no matching sweep dir at ${search_root}"
            SKIPPED+=("${name}: sweep dir absent under SIM_ROOT")
            n_skip=$((n_skip + 1)); continue
        fi
    else
        search_root="${camp}"
    fi

    ihit=$(find "${search_root}" -name 'iter_*.npz' -not -name 'mea_iter_*.npz' \
           -type f 2>/dev/null | head -1)
    if [ -z "${ihit}" ]; then
        echo "SKIP  ${name}: no iter_*.npz under ${search_root}"
        echo "        The topology block (p0_conn/d0_conn/beta_conn) lives ONLY"
        echo "        in the sweep output. Pass SIM_ROOT as the 4th argument."
        SKIPPED+=("${name}: no sweep output -> cannot build the 27-D label")
        n_skip=$((n_skip + 1)); continue
    fi
    sim_dir=$(dirname "$(dirname "${ihit}")")

    stem="${OUT_ROOT}/sbi_${name}"
    if [ -f "${stem}.parquet" ]; then
        echo "SKIP  ${name}: ${stem}.parquet exists (delete to redo)"
        SKIPPED+=("${name}: already exported")
        n_skip=$((n_skip + 1)); continue
    fi

    VARS="CKPT=${CKPT},CAMPAIGN=${sim_dir},MEA_OUT=${mea_dir}"
    VARS="${VARS},OUT=${stem},CAMPAIGN_ID=${name}"
    [ -n "${MAX_RECORDS:-}" ] && VARS="${VARS},MAX_RECORDS=${MAX_RECORDS}"
    [ -n "${ENV_NAME:-}" ]    && VARS="${VARS},ENV_NAME=${ENV_NAME}"
    [ -n "${SIMTIME:-}" ]     && VARS="${VARS},SIMTIME=${SIMTIME}"

    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "DRY   ${name}"
        echo "        CAMPAIGN = ${sim_dir}"
        echo "        MEA_OUT  = ${mea_dir}"
        echo "        OUT      = ${stem}"
        echo "        qsub -l ${SELECT} -l walltime=${WALLTIME} -N sbi_${name} -v ${VARS} ${SUBMIT}"
    else
        jid=$(qsub -l "${SELECT}" -l "walltime=${WALLTIME}" -N "sbi_${name}" \
                   -v "${VARS}" "${SUBMIT}")
        echo "SUBMIT ${name} -> ${jid}"
    fi
    n_sub=$((n_sub + 1))
done

echo ""
echo "======================================================================"
echo "submitted: ${n_sub}   skipped: ${n_skip}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "skipped detail:"
    for s in "${SKIPPED[@]}"; do echo "  - ${s}"; done
fi
if [ "${DRYRUN:-0}" = "1" ]; then
    echo ""
    echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"
else
    echo ""
    echo "Watch with:  qstat -u \$USER"
    echo "When all are done, verify the checkpoint digest is IDENTICAL across"
    echo "shards -- a mixed-encoder dataset is void and nothing downstream"
    echo "will notice:"
    echo "  for f in ${OUT_ROOT}/*.json; do python3 -c \"import json,sys;\\"
    echo "    d=json.load(open(sys.argv[1]));\\"
    echo "    print(d['embedding']['dsn_checkpoint_sha256'][:16], sys.argv[1])\" \"\$f\"; done \\"
    echo "    | sort | uniq -c -w16"
fi
