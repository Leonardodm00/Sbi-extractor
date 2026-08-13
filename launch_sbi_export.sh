#!/bin/bash
##########################################################################
# Submit one SBI export job per sweep task directory.
#
# A campaign is usually split across several sweep_<NODETAG>_task<IDX>
# directories, each with its own MEA output. Exports are independent -- each
# reads a disjoint set of simulations and writes its own shard -- so they are
# submitted as separate jobs rather than looped inside one.
#
# This is a LOGIN-NODE script. It only calls qsub; it runs no Python.
#
# USAGE:
#     ./launch_sbi_export.sh <CKPT> <CAMPAIGN_ROOT> <MEA_ROOT> <OUT_ROOT> [TAG]
#
# EXAMPLE:
#     ./launch_sbi_export.sh \
#         /scratch/$USER/runs/mea_joint_full/checkpoints/best.pt \
#         /scratch/$USER/campaign_cadex001 \
#         /scratch/$USER/mea_out/campaign_cadex001 \
#         /scratch/$USER/export/cadex001 \
#         cadex001
#
# DRY RUN first -- prints the qsub lines without submitting:
#     DRYRUN=1 ./launch_sbi_export.sh ...
#
# Layout assumed (matching submit_mea.sh):
#     <CAMPAIGN_ROOT>/sweep_*_task*/topo_*/iter_*.npz
#     <MEA_ROOT>/sweep_*_task*/topo_*/mea_iter_*.npz
#
# If your campaign has topo_* directly under the root (no sweep_* level),
# the loop falls back to submitting a single job for the root itself.
##########################################################################

set -euo pipefail

if [ "$#" -lt 4 ]; then
    sed -n '2,30p' "$0"
    exit 2
fi

CKPT="$1"
CAMPAIGN_ROOT="$2"
MEA_ROOT="$3"
OUT_ROOT="$4"
TAG="${5:-$(basename "${CAMPAIGN_ROOT}")}"

SELECT="${SELECT:-select=1:ncpus=8:mem=32gb}"
WALLTIME="${WALLTIME:-02:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for p in "${CKPT}" "${CAMPAIGN_ROOT}" "${MEA_ROOT}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: does not exist: ${p}" >&2
        exit 3
    fi
done
mkdir -p "${OUT_ROOT}"

# Collect the sweep task dirs; fall back to the root if there are none.
mapfile -t SWEEPS < <(find "${CAMPAIGN_ROOT}" -maxdepth 1 -type d -name 'sweep_*' | sort)
if [ "${#SWEEPS[@]}" -eq 0 ]; then
    SWEEPS=("${CAMPAIGN_ROOT}")
fi

n_sub=0
n_skip=0
for sw in "${SWEEPS[@]}"; do
    name="$(basename "${sw}")"
    if [ "${sw}" = "${CAMPAIGN_ROOT}" ]; then
        mea_dir="${MEA_ROOT}"
        cid="${TAG}"
        stem="${OUT_ROOT}/sbi_${TAG}"
    else
        mea_dir="${MEA_ROOT}/${name}"
        cid="${TAG}_${name}"
        stem="${OUT_ROOT}/sbi_${TAG}_${name}"
    fi

    if ! ls "${mea_dir}"/topo_*/mea_iter_*.npz >/dev/null 2>&1; then
        echo "SKIP  ${name}: no mea_iter_*.npz under ${mea_dir}"
        n_skip=$((n_skip + 1))
        continue
    fi
    if [ -f "${stem}.parquet" ]; then
        echo "SKIP  ${name}: ${stem}.parquet already exists (delete to redo)"
        n_skip=$((n_skip + 1))
        continue
    fi

    VARS="CKPT=${CKPT},CAMPAIGN=${sw},MEA_OUT=${mea_dir},OUT=${stem},CAMPAIGN_ID=${cid}"
    [ -n "${MAX_RECORDS:-}" ] && VARS="${VARS},MAX_RECORDS=${MAX_RECORDS}"
    [ -n "${ENV_NAME:-}" ]    && VARS="${VARS},ENV_NAME=${ENV_NAME}"
    [ -n "${SIMTIME:-}" ]     && VARS="${VARS},SIMTIME=${SIMTIME}"

    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "qsub -l ${SELECT} -l walltime=${WALLTIME} -N sbi_${cid} -v ${VARS} ${SCRIPT_DIR}/submit_sbi_export.sh"
    else
        jid=$(qsub -l "${SELECT}" -l "walltime=${WALLTIME}" -N "sbi_${cid}" \
                   -v "${VARS}" "${SCRIPT_DIR}/submit_sbi_export.sh")
        echo "SUBMIT ${name} -> ${jid}"
    fi
    n_sub=$((n_sub + 1))
done

echo ""
echo "submitted: ${n_sub}   skipped: ${n_skip}"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "(DRYRUN -- nothing was actually submitted)"
fi
