#!/bin/bash
# env.sh -- repo-local defaults for Sbi-extractor scripts.
#
# Sourced automatically by submit_sbi_export.sh, and safe to source by
# hand before running anything else in this repo:
#     source env.sh
#
# The point: cluster paths that used to live loose in $HOME, or get
# retyped into every -v line, now live here -- in the repo, versioned,
# discoverable by anyone who clones it.
#
# Uses `: "${VAR:=default}"` throughout, which sets VAR only if it is
# currently UNSET or empty. A value already provided by the calling shell
# (including an explicit `-v VAR=...` under qsub) is left untouched.
# Priority, highest to lowest:
#     explicit -v / pre-exported value  >  this file  >  no default (error)
#
# ARTIFACTS_DIR is resolved relative to THIS file's own location (the same
# BASH_SOURCE trick submit_sbi_export.sh uses), so it is correct regardless
# of where the repo is cloned or what directory the caller's shell is in.
ENV_SH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
: "${ARTIFACTS_DIR:=${ENV_SH_DIR}/artifacts}"

: "${ENV_NAME:=sbi_export}"

# [migration step 3, 2026-09-19] The DSN is no longer a separate repo: it is
# Simulation-Based-Inference/hpc/dsn. SBI_HPC_DIR names that repo's hpc/
# directory; artifacts/sbi_hpc is a symlink to it, recreated per machine
# (artifacts/README.md). dsn_tree.py applies the same default from python.
: "${SBI_HPC_DIR:=${ARTIFACTS_DIR}/sbi_hpc}"

# DSN_MAIN_DIR is retired (migration step 4, 2026-09-19): nothing in this
# repo reads it any more. If your shell still exports it, unset it.

# Frozen real-cohort specs (see artifacts/README.md for provenance:
# resolved from a checkpoint's own config.data.npz_specs pointer).
: "${SPECS_REAL:=${ARTIFACTS_DIR}/specs_real.json}"

export ARTIFACTS_DIR ENV_NAME SBI_HPC_DIR SPECS_REAL

# CKPT is deliberately NOT defaulted here. Which frozen checkpoint is in
# use is a scientifically consequential choice (HANDOFF.md section 3/7:
# "verify encoder consistency by checksum, not by assumption"), so it
# stays required and explicit at every invocation rather than silently
# auto-selecting "latest". Derive it the same way as always, just from the
# new location:
#     CKPT="$(ls -t "${ARTIFACTS_DIR}/frozen_dsn"/*.pt | head -1)"

# The pipeline's job protocol (D-113). When Simulation-Based-Inference's
# pipeline runs a launcher of this repository it exports PIPELINE_JOBS_DIR
# (its hpc/pipeline/jobs), WORK_ROOT, CHAIN_NAME, AVOID_FILE and AVOID_HOLD_S;
# the launchers append them to their qsub -v lists with sbix_pipeline_v, and
# the job scripts then start and end through $PIPELINE_JOBS_DIR/stage_job.sh
# (the avoid list; conda, else the environment's own folder; exit 94 / 96
# when the job cannot start; the chain's watchdog at the end). By hand none
# is exported: the suffix is empty and the jobs activate as they always did.
# Status 3, with the reason on stderr, for a value qsub -v cannot carry or a
# PIPELINE_JOBS_DIR without stage_job.sh.
sbix_pipeline_v() {
    local v out=""
    [ -n "${PIPELINE_JOBS_DIR:-}" ] || return 0
    if [ ! -f "${PIPELINE_JOBS_DIR}/stage_job.sh" ]; then
        echo "ABORT: PIPELINE_JOBS_DIR=${PIPELINE_JOBS_DIR} has no stage_job.sh (Simulation-Based-Inference before D-113: pull it)" >&2
        return 3
    fi
    for v in PIPELINE_JOBS_DIR WORK_ROOT CHAIN_NAME AVOID_FILE AVOID_HOLD_S; do
        [ -n "${!v:-}" ] || continue
        case "${!v}" in
            *[[:space:],]*)
                echo "ABORT: $v contains whitespace or a comma, which qsub -v cannot carry: ${!v}" >&2
                return 3 ;;
        esac
        out="${out},${v}=${!v}"
    done
    printf '%s' "$out"
}
