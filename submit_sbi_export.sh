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
#     REPO_DIR    override for self-location (default: auto-detected from
#                 this script's own path, or PBS_O_WORKDIR under qsub)
#     CAMPAIGN_ID provenance tag written into every row (default: basename OUT)
#     ENV_NAME    conda environment (default: sbi_export, or from env.sh)
#     DSN_MAIN_DIR  default: from env.sh in this repo (artifacts/dsn_main);
#                   no $HOME guess -- errors loudly if neither is set
#     SIM_MAIN_DIR  default: $HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1
#     MAX_RECORDS if set, stop after N simulations (dry run)
#     SIMTIME     override T [s]. Normally read from job_args.json; set this
#                 ONLY if that file is missing or wrong. Never take it from
#                 the npz -- process_campaign.py infers simtime from the last
#                 spike, so quiet runs record a duration far below the truth.
#     LABEL_AXES  frozen label_axes.json fixing WHICH topology axes enter
#                 theta (default: <artifacts>/label_axes.json; a missing file
#                 is FATAL, since falling back would change p silently).
#                 Set to 'none' to force the legacy 4-axis block.
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

# --- locate the repo, independent of the invocation CWD and of $HOME -----
# Run directly (`bash submit_sbi_export.sh`), BASH_SOURCE[0] is the real
# script path, so its directory IS the repo -- the script then works from
# any CWD, and is immune to a stray PBS_O_WORKDIR left over in the calling
# shell. Run under qsub, PBS copies the script into its spool dir, so
# BASH_SOURCE points there and is useless; PBS_O_WORKDIR (the qsub
# invocation dir) is the fallback for that case, as before. Order of trust:
#     REPO_DIR (explicit -v override) > script's own dir > PBS_O_WORKDIR
# Each candidate is accepted only if it actually contains example_export.py,
# so a wrong guess fails loudly HERE with the candidates printed, not as a
# bare python "No such file or directory" after the banner.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P || true)"
if [ -z "${REPO_DIR:-}" ]; then
    for cand in "${SCRIPT_DIR}" "${PBS_O_WORKDIR:-}"; do
        if [ -n "${cand}" ] && [ -f "${cand}/example_export.py" ]; then
            REPO_DIR="${cand}"
            break
        fi
    done
fi
if [ -z "${REPO_DIR:-}" ] || [ ! -f "${REPO_DIR}/example_export.py" ]; then
    echo "ERROR: cannot locate example_export.py." >&2
    echo "       script dir    : ${SCRIPT_DIR:-unset}" >&2
    echo "       PBS_O_WORKDIR : ${PBS_O_WORKDIR:-unset}" >&2
    echo "       Run/submit from the Sbi-extractor repo root, or pass" >&2
    echo "       -v REPO_DIR=/abs/path/to/Sbi-extractor" >&2
    exit 5
fi
cd "${REPO_DIR}"

# --- repo-local defaults (env.sh), never $HOME ----------------------------
# If present, artifacts/../env.sh sets defaults for DSN_MAIN_DIR (and any
# future repo-scoped path) using `: "${VAR:=...}"`, so it NEVER overrides a
# value already provided via `-v` / the calling shell. Priority is:
#     -v override  >  env.sh (this repo)  >  hard failure (no $HOME guess)
if [ -f "${REPO_DIR}/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/env.sh"
fi

# --- environment ---------------------------------------------------------
ENV_NAME="${ENV_NAME:-sbi_export}"
module load anaconda3 2>/dev/null || true

# WHY THIS IS NOT JUST `conda activate`.
#
# `conda activate` is a SHELL FUNCTION, defined by conda's init hook. A
# non-interactive PBS shell does not source that hook, so the function does
# not exist and the bare `conda` binary refuses:
#
#     CondaError: Run 'conda init' before 'conda activate'
#
# exiting 1 with NOTHING on stdout or stderr. Under `set -euo pipefail` that
# kills the job before the first echo, so the symptom is a job that finishes
# instantly with EMPTY .o and .e files and Exit_status=1 -- no error message
# anywhere. `source activate` (the deprecated form) fails the same way on
# recent conda, and its message was being swallowed by 2>/dev/null.
#
# `eval "$(conda shell.bash hook)"` is what defines the function. This is the
# same pattern Deep-Summary-Network/Main/hpc/run_refit.pbs uses, which is
# known to work on this cluster -- it is what trained the checkpoint this
# job embeds with.
#
# set +u around it because conda's own scripts read unset variables and this
# script runs under `set -u`.
if [ -n "${PYBIN:-}" ]; then
    :                                   # explicit interpreter wins
elif command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${ENV_NAME}" 2>/dev/null || true
    set -u
fi

# Resolve the interpreter explicitly rather than trusting PATH. Order:
# PYBIN, then whatever activation put on PATH, then the env prefix.
PY="${PYBIN:-}"
[ -n "${PY}" ] || PY="$(command -v python3 2>/dev/null || true)"
[ -n "${PY}" ] || PY="$(command -v python 2>/dev/null || true)"
[ -n "${PY}" ] && [ -x "${PY}" ] || PY="${HOME}/.conda/envs/${ENV_NAME}/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: no usable interpreter found." >&2
    echo "       ENV_NAME=${ENV_NAME}" >&2
    echo "       CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none}" >&2
    echo "       Resubmit with -v PYBIN=/abs/path/to/python" >&2
    exit 6
fi

# Fail LOUDLY on the wrong interpreter rather than quietly on the wrong
# library versions. A base-env python may import torch at a DIFFERENT
# version, and torch 2.6 changed the torch.load weights_only default, which
# decides whether the checkpoint config can be read back at all. A job that
# runs to completion against the wrong torch is worse than one that dies.
if ! "${PY}" -c "import torch, numpy, scipy, pyarrow" >/dev/null 2>&1; then
    echo "ERROR: ${PY} cannot import the required packages:" >&2
    "${PY}" -c "import torch, numpy, scipy, pyarrow" >&2 || true
    echo "       CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none}" >&2
    exit 7
fi

export DSN_MAIN_DIR="${DSN_MAIN_DIR:?DSN_MAIN_DIR not set. Expected a default from ${REPO_DIR}/env.sh -- is artifacts/dsn_main present (run relocate_artifacts.sh)? Or pass -v DSN_MAIN_DIR=/abs/path explicitly.}"
# SIM_MAIN_DIR decides which PARAM_NAMES / PARAM_BOUNDS the labels are built
# against, so a wrong value mislabels every axis rather than failing. This used
# to default to $HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1;
# that turned "the variable did not reach the node" into "the labels came from
# some other campaign family", which is strictly worse than not running. There
# is more than one simulator tree on this cluster and they differ in width
# (36-D vs 37-D), so no default can be correct for all of them. Required now,
# matching DSN_MAIN_DIR immediately above.
export SIM_MAIN_DIR="${SIM_MAIN_DIR:?SIM_MAIN_DIR not set. It must name the simulator tree whose registry produced THIS campaign -- e.g. ANN/Phenomenological/Main/Giulia_Astro for campaign_cadex_hhgap_*, ANN/Phenomenological/Main for campaign_cadex_rho1300*. Pass it with -v SIM_MAIN_DIR=/abs/path, or export it before launch_sweep_exports.sh, which forwards it.}"

# Torch spawns one thread per core by default and then contends with itself on
# a small CNN. Pin it to the cores PBS actually gave us.
NCPUS="${PBS_NCPUS:-1}"
export OMP_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"

# The campaign is normally <SIM_MAIN_DIR>/<campaign>/<task>, so the grandparent
# of CAMPAIGN should BE SIM_MAIN_DIR. They may legitimately differ -- the
# registry need not sit with the campaigns -- so this warns rather than exits.
# It would have named the defect immediately on 2026-09-10, when a stale
# SIM_MAIN_DIR from an earlier session propagated to all 51 jobs and the labels
# were being built from a 36-D registry against a 37-D manifest.
_SIM_EXPECT="$(cd "$(dirname "$(dirname "${CAMPAIGN}")")" 2>/dev/null && pwd -P || true)"
_SIM_ACTUAL="$(cd "${SIM_MAIN_DIR}" 2>/dev/null && pwd -P || true)"
if [ -n "${_SIM_EXPECT}" ] && [ -n "${_SIM_ACTUAL}" ] \
   && [ "${_SIM_EXPECT}" != "${_SIM_ACTUAL}" ]; then
    echo "[sbi] WARNING: SIM_MAIN_DIR is not the tree this campaign sits in." >&2
    echo "[sbi]          SIM_MAIN_DIR      : ${_SIM_ACTUAL}" >&2
    echo "[sbi]          campaign implies  : ${_SIM_EXPECT}" >&2
    echo "[sbi]          The labels will be built from the FORMER. If those two" >&2
    echo "[sbi]          trees carry different registries, every axis is" >&2
    echo "[sbi]          mislabelled. registry_from_manifest() will refuse on a" >&2
    echo "[sbi]          width mismatch, but NOT on an equal-width difference." >&2
fi

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

# NOTE: this used to be `ls "${MEA_OUT}"/topo_*/mea_iter_*.npz >/dev/null 2>&1`,
# and it failed on exactly the tasks that matter most. Bash expands the glob
# in-process (fine), then execve() on /bin/ls fails with E2BIG as soon as the
# argument vector exceeds ARG_MAX -- 2097152 bytes on Linux, i.e. roughly 20k
# paths at this path length. A 73k-file sweep task therefore looked EMPTY, and
# the `2>&1` swallowed the "Argument list too long" that would have said so.
# `find -print -quit` stops at the first match and never builds an argument
# vector, so it is O(1) in the number of files rather than O(n).
# The launcher already uses find for the same reason (lines 9, 244, 277).
if [ -z "$(find "${MEA_OUT}" -mindepth 2 -maxdepth 2 \
                -path '*/topo_*/mea_iter_*.npz' -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no topo_*/mea_iter_*.npz under ${MEA_OUT}" >&2
    echo "       Has process_campaign.py been run over this campaign? The" >&2
    echo "       export is built from DETECTED spikes; it cannot fall back" >&2
    echo "       on the simulator's ground truth without breaking parity" >&2
    echo "       with the real recordings." >&2
    exit 4
fi

mkdir -p "$(dirname "${OUT}")"

EXTRA=""
# --- frozen label axes ---------------------------------------------------
# Which topology-level axes enter theta is a decision that MUST be identical
# for every shard: a shard built without this file falls back to the legacy
# 4-axis block (p = 27, including the causally inert conn_prob), and the two
# kinds of shard cannot be concatenated -- different width, different column
# meaning. Silently defaulting would therefore corrupt a 206-job launch in a
# way that only shows up much later, so a missing file is fatal here.
# Set LABEL_AXES=none to deliberately reproduce the legacy behaviour.
LABEL_AXES="${LABEL_AXES:-${ARTIFACTS_DIR:-${REPO_DIR}/artifacts}/label_axes.json}"
if [ "${LABEL_AXES}" = "none" ]; then
    echo "[sbi] WARNING: LABEL_AXES=none -- using the LEGACY 4-axis topology" >&2
    echo "[sbi]          block. Shards built this way are NOT poolable with" >&2
    echo "[sbi]          shards built from a frozen label_axes.json." >&2
else
    if [ ! -f "${LABEL_AXES}" ]; then
        echo "ERROR: label axes file not found: ${LABEL_AXES}" >&2
        echo "       Generate it ONCE over all campaigns, then submit:" >&2
        echo "         python3 preflight_label_axes.py --sim_main <SIM_MAIN_DIR> \\" >&2
        echo "             --campaigns 'campaign_*' --require-conn-rule weibull \\" >&2
        echo "             --exclude 'conn_prob=<reason>' \\" >&2
        echo "             --out ${LABEL_AXES}" >&2
        echo "       Or pass -v LABEL_AXES=none to accept the legacy 4-axis" >&2
        echo "       block deliberately (NOT poolable with frozen-axis shards)." >&2
        exit 8
    fi
    EXTRA="${EXTRA} --label_axes ${LABEL_AXES}"
fi

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
echo "[sbi] env        : ${ENV_NAME}   (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none})"
echo "[sbi] python     : ${PY}"
echo "[sbi] versions   : $("${PY}" -c 'import sys,torch,numpy;print("py",sys.version.split()[0],"torch",torch.__version__,"numpy",numpy.__version__)')"
echo "[sbi] repo       : ${REPO_DIR}"
echo "[sbi] artifacts  : ${ARTIFACTS_DIR:-(env.sh not found -- no repo-local defaults)}"
echo "[sbi] checkpoint : ${CKPT}"
echo "[sbi] sim main   : ${SIM_MAIN_DIR}"
echo "[sbi] campaign   : ${CAMPAIGN}"
echo "[sbi] mea_out    : ${MEA_OUT}"
echo "[sbi] out stem   : ${OUT}"
echo "[sbi] id         : ${CAMPAIGN_ID}"
echo "[sbi] device     : ${DEVICE}   threads: ${NCPUS}"
echo "[sbi] simtime    : ${SIMTIME:-(from job_args.json)}"
echo "[sbi] trim head  : ${TRIM_HEAD_S:-0} s"
echo "[sbi] label axes : ${LABEL_AXES}"
echo ""

"${PY}" "${REPO_DIR}/example_export.py" --mode campaign \
    --checkpoint  "${CKPT}" \
    --sim_dir     "${SIM_MAIN_DIR}" \
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
"${PY}" - "${OUT}.json" <<'PYEOF'
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
