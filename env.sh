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

# DSN encoder repo location. A symlink, never a copy: Deep-Summary-Network
# is a separate, actively developed repo -- this only needs to know where
# to find it. See artifacts/README.md for how to recreate the symlink if
# it is ever missing (e.g. on a fresh clone).
: "${DSN_MAIN_DIR:=${ARTIFACTS_DIR}/dsn_main}"

# Frozen real-cohort specs (see artifacts/README.md for provenance:
# resolved from a checkpoint's own config.data.npz_specs pointer).
: "${SPECS_REAL:=${ARTIFACTS_DIR}/specs_real.json}"

export ARTIFACTS_DIR ENV_NAME DSN_MAIN_DIR SPECS_REAL

# CKPT is deliberately NOT defaulted here. Which frozen checkpoint is in
# use is a scientifically consequential choice (HANDOFF.md section 3/7:
# "verify encoder consistency by checksum, not by assumption"), so it
# stays required and explicit at every invocation rather than silently
# auto-selecting "latest". Derive it the same way as always, just from the
# new location:
#     CKPT="$(ls -t "${ARTIFACTS_DIR}/frozen_dsn"/*.pt | head -1)"
