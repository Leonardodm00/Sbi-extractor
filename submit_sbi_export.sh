#!/bin/bash
#PBS -S /bin/bash
#PBS -N sbi_export
#PBS -k eo
#PBS -l walltime=02:00:00
##########################################################################
# SBI export wrapper: DSN embeddings paired with 27-D theta labels.
#
# This is POST-post-processing. It reads an existing MEA-processed campaign
# (detected electrode spikes) plus the original sweep output (parameter
# vectors), runs the frozen DSN forward pass, and writes one Parquet shard
# plus a JSON sidecar. It needs numpy + scipy + pyarrow + torch (NO Brian2,
# NO C++ compilation), and the forward pass is cheap: a few thousand 180 s
# windows through a small 1D CNN is minutes, not hours, on CPU.
#
# Walltime is therefore set low. Raise it only if a campaign has >> 10^4
# simulations.
#
# SUBMIT (one campaign):
#     qsub -l select=1:ncpus=8:mem=32gb \
#          -v CKPT=/scratch/USER/runs/mea_joint_full/checkpoints/best.pt,\
#CAMPAIGN=/scratch/USER/campaign_TAG/sweep_c48_task0000,\
#MEA_OUT=/scratch/USER/mea_out/sweep_c48_task0000,\
#OUT=/scratch/USER/export/sbi_TAG_task0000,\
#CAMPAIGN_ID=TAG_task0000 \
#          submit_sbi_export.sh
#
# NOTE ON PATHS: --campaign must point at the directory that DIRECTLY
#       contains the topo_* folders (a sweep_<NODETAG>_task<IDX> dir, not
#       the campaign_<TAG> root), matching the convention in submit_mea.sh.
#       MEA_OUT is the corresponding process_campaign.py output directory.
#       The two must describe the SAME simulations: the export joins them on
#       (topo_idx, iter_idx) and will raise rather than guess if they do not
#       line up.
#
# REQUIRED -v variables:
#     CKPT        frozen DSN checkpoint (.pt)
#     CAMPAIGN    sweep output dir containing topo_*/iter_*.npz
#     MEA_OUT     process_campaign.py output containing topo_*/mea_iter_*.npz
#     OUT         output path STEM, without extension
# OPTIONAL -v variables:
#     CAMPAIGN_ID provenance tag written into every row (default: basename OUT)
#     ENV_NAME    conda environment (default: sbi_export)
#     DSN_MAIN_DIR  default: $HOME/repos/Deep-Summary-Network/Main
#     SIM_MAIN_DIR  default: $HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1
#     MAX_RECORDS if set, stop after N simulations (dry run)
#     SIMTIME     override T [s]. Normally read from job_args.json; set this
#                 ONLY if that file is missing or wrong. Never take it from
#                 the npz -- process_campaign.py infers simtime from the last
#                 spike, so quiet runs record a duration far below the truth.
#     TRIM_HEAD_S discard the first N seconds of every simulated trace as
#                 burn-in, BEFORE windowing. A simulation settles from its
#                 initial conditions; a real recording is already at steady
#                 state, so an untrimmed head puts a transient in every
#                 simulated row and none of the real ones. Usable duration
#                 becomes T - TRIM_HEAD_S and must stay >= the DSN window.
#     DEVICE      cpu (default) or cuda
#     BATCH_SIZE  forward-pass chunk (default 256; does not change results)
##########################################################################

set -euo pipefail

# --- required arguments --------------------------------------------------
missing=0
for v in CKPT CAMPAIGN MEA_OUT OUT; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: -v ${v}=... is required" >&2
        missing=1
    fi
done
if [ "${missing}" -ne 0 ]; then
    echo "" >&2
    echo "usage: qsub -l select=1:ncpus=8:mem=32gb \\" >&2
    echo "         -v CKPT=...,CAMPAIGN=...,MEA_OUT=...,OUT=... \\" >&2
    echo "         submit_sbi_export.sh" >&2
    exit 2
fi

cd "${PBS_O_WORKDIR:-.}"

# --- environment ---------------------------------------------------------
ENV_NAME="${ENV_NAME:-sbi_export}"
module load anaconda3 2>/dev/null || true
# 'conda activate' needs the shell hook under a non-interactive PBS shell;
# 'source activate' is the fallback that works without it.
source activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"

export DSN_MAIN_DIR="${DSN_MAIN_DIR:-$HOME/repos/Deep-Summary-Network/Main}"
export SIM_MAIN_DIR="${SIM_MAIN_DIR:-$HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1}"

# Torch spawns one thread per core by default and then contends with itself on
# a small CNN. Pin it to the cores PBS actually gave us.
NCPUS="${PBS_NCPUS:-1}"
export OMP_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"

CAMPAIGN_ID="${CAMPAIGN_ID:-$(basename "${OUT}")}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# --- fail fast, before burning the allocation ----------------------------
for p in "${CKPT}" "${DSN_MAIN_DIR}" "${SIM_MAIN_DIR}" "${CAMPAIGN}" "${MEA_OUT}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: path does not exist: ${p}" >&2
        exit 3
    fi
done

if ! ls "${MEA_OUT}"/topo_*/mea_iter_*.npz >/dev/null 2>&1; then
    echo "ERROR: no topo_*/mea_iter_*.npz under ${MEA_OUT}" >&2
    echo "       Has process_campaign.py been run over this campaign? The" >&2
    echo "       export is built from DETECTED spikes; it cannot fall back" >&2
    echo "       on the simulator's ground truth without breaking parity" >&2
    echo "       with the real recordings." >&2
    exit 4
fi

mkdir -p "$(dirname "${OUT}")"

EXTRA=""
if [ -n "${MAX_RECORDS:-}" ]; then
    EXTRA="${EXTRA} --max_records ${MAX_RECORDS}"
fi
if [ -n "${SIMTIME:-}" ]; then
    EXTRA="${EXTRA} --simtime ${SIMTIME}"
fi
if [ -n "${TRIM_HEAD_S:-}" ]; then
    EXTRA="${EXTRA} --trim_head_s ${TRIM_HEAD_S}"
fi

echo "[sbi] host       : $(hostname)"
echo "[sbi] started    : $(date -Is)"
echo "[sbi] env        : ${ENV_NAME}   (python $(python -c 'import sys;print(sys.version.split()[0])'))"
echo "[sbi] checkpoint : ${CKPT}"
echo "[sbi] campaign   : ${CAMPAIGN}"
echo "[sbi] mea_out    : ${MEA_OUT}"
echo "[sbi] out stem   : ${OUT}"
echo "[sbi] id         : ${CAMPAIGN_ID}"
echo "[sbi] device     : ${DEVICE}   threads: ${NCPUS}"
echo "[sbi] simtime    : ${SIMTIME:-(from job_args.json)}"
echo "[sbi] trim head  : ${TRIM_HEAD_S:-0} s"
echo ""

python example_export.py --mode campaign \
    --checkpoint  "${CKPT}" \
    --campaign    "${CAMPAIGN}" \
    --mea_out     "${MEA_OUT}" \
    --campaign_id "${CAMPAIGN_ID}" \
    --out         "${OUT}" \
    --device      "${DEVICE}" \
    --batch_size  "${BATCH_SIZE}" \
    ${EXTRA}

echo ""
echo "[sbi] finished   : $(date -Is)"
echo "[sbi] parquet    : ${OUT}.parquet"
echo "[sbi] sidecar    : ${OUT}.json"

# --- post-run summary ----------------------------------------------------
# Surfaces the two numbers worth reading in the .o file without opening the
# sidecar by hand. A nonzero 'too short' count means the simtime trap fired.
python - "${OUT}.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
print("[sbi] rows written      : %d" % d["n_rows"])
print("[sbi] traces used       : %d" % d["n_traces_used"])
print("[sbi] traces TOO SHORT  : %d" % d["n_traces_skipped_too_short"])
print("[sbi] assertions passed : %s" % ", ".join(d["assertions_passed"]))
print("[sbi] ckpt sha256       : %s" % d["embedding"]["dsn_checkpoint_sha256"])
if d["n_traces_skipped_too_short"]:
    print("[sbi] WARNING: traces were dropped for being shorter than the DSN")
    print("[sbi]          window. Check --simtime against what the campaign")
    print("[sbi]          actually ran.")
PYEOF
