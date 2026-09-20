#!/bin/bash
# relocate_artifacts.sh -- one-time migration: move the frozen DSN
# checkpoint and the real-cohort specs out of $HOME into this repo's
# artifacts/ directory, and create the sbi_hpc pointer (migration step 4:
# the DSN is Simulation-Based-Inference/hpc/dsn; the old dsn_main pointer
# is retired and removed if present).
#
# Safe to re-run: every step is skipped, with a message, if its
# destination already exists. Nothing is deleted until the corresponding
# new copy is verified in place.
#
# Usage:
#   bash relocate_artifacts.sh              # do it
#   bash relocate_artifacts.sh --dry-run     # print what would happen, do nothing
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
REPO_DIR="${SCRIPT_DIR}"
if [ ! -f "${REPO_DIR}/example_export.py" ]; then
    echo "ERROR: expected to find example_export.py next to this script (repo root)." >&2
    echo "       script dir: ${SCRIPT_DIR}" >&2
    exit 1
fi
ARTIFACTS_DIR="${REPO_DIR}/artifacts"

OLD_FROZEN_DSN="${HOME}/frozen_dsn"
OLD_SPECS_REAL="${HOME}/specs_real.json"
OLD_DSN_MAIN="${HOME}/dsn_main"
SBI_HPC_TARGET="${SBI_HPC_TARGET:-${HOME}/SBI/hpc}"

NEW_FROZEN_DSN="${ARTIFACTS_DIR}/frozen_dsn"
NEW_SPECS_REAL="${ARTIFACTS_DIR}/specs_real.json"
NEW_DSN_MAIN="${ARTIFACTS_DIR}/dsn_main"
NEW_SBI_HPC="${ARTIFACTS_DIR}/sbi_hpc"

sha256_of() {
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
}

say() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[dry-run] $*"
    else
        echo "[relocate] $*"
    fi
}

echo "== 1. frozen_dsn checkpoint(s) =="
if [ -d "${OLD_FROZEN_DSN}" ]; then
    if [ -e "${NEW_FROZEN_DSN}" ]; then
        echo "SKIP: ${NEW_FROZEN_DSN} already exists -- not overwriting."
        echo "      If this is a stale partial migration, resolve by hand and re-run."
    elif [ "${DRY_RUN}" -eq 1 ]; then
        say "mkdir -p ${ARTIFACTS_DIR}"
        say "mv ${OLD_FROZEN_DSN} -> ${NEW_FROZEN_DSN}  (checksummed before/after)"
    else
        mkdir -p "${ARTIFACTS_DIR}"
        # Checksum BEFORE the move, in case a cross-filesystem `mv` (e.g.
        # $HOME vs. a davinci-1 scratch mount are not guaranteed to be the
        # same filesystem) degrades to copy-then-delete under the hood.
        before_names=()
        before_sums=()
        for f in "${OLD_FROZEN_DSN}"/*; do
            before_names+=("$(basename "$f")")
            before_sums+=("$(sha256_of "$f")")
        done
        say "mv ${OLD_FROZEN_DSN} -> ${NEW_FROZEN_DSN}"
        mv "${OLD_FROZEN_DSN}" "${NEW_FROZEN_DSN}"
        fail=0
        for i in "${!before_names[@]}"; do
            name="${before_names[$i]}"
            before="${before_sums[$i]}"
            after="$(sha256_of "${NEW_FROZEN_DSN}/${name}")"
            if [ -n "${before}" ] && [ "${before}" != "${after}" ]; then
                echo "ERROR: checksum mismatch after move for ${name}" >&2
                echo "       before: ${before}" >&2
                echo "       after : ${after}" >&2
                fail=1
            fi
        done
        if [ "${fail}" -eq 0 ]; then
            echo "OK: all checksums match after move."
        else
            exit 1
        fi
    fi
else
    echo "SKIP: ${OLD_FROZEN_DSN} does not exist (already migrated, or never existed here)."
fi

echo ""
echo "== 2. specs_real.json =="
if [ -f "${OLD_SPECS_REAL}" ]; then
    if [ -e "${NEW_SPECS_REAL}" ]; then
        echo "SKIP: ${NEW_SPECS_REAL} already exists -- not overwriting."
    elif [ "${DRY_RUN}" -eq 1 ]; then
        say "mkdir -p ${ARTIFACTS_DIR}"
        say "mv ${OLD_SPECS_REAL} -> ${NEW_SPECS_REAL}  (checksummed before/after)"
    else
        mkdir -p "${ARTIFACTS_DIR}"
        before="$(sha256_of "${OLD_SPECS_REAL}")"
        say "mv ${OLD_SPECS_REAL} -> ${NEW_SPECS_REAL}"
        mv "${OLD_SPECS_REAL}" "${NEW_SPECS_REAL}"
        after="$(sha256_of "${NEW_SPECS_REAL}")"
        if [ "${before}" != "${after}" ]; then
            echo "ERROR: checksum mismatch after move for specs_real.json" >&2
            exit 1
        fi
        echo "${after}  specs_real.json" > "${NEW_SPECS_REAL}.sha256"
        echo "OK: checksum matches after move (recorded in specs_real.json.sha256; none existed before)."
    fi
else
    echo "SKIP: ${OLD_SPECS_REAL} does not exist (already migrated, or never existed here)."
fi

echo ""
echo "== 3. sbi_hpc pointer (the DSN is \${SBI_HPC_DIR}/dsn since migration step 4) =="
if [ -L "${NEW_SBI_HPC}" ] && [ -f "${NEW_SBI_HPC}/dsn/backbone.py" ]; then
    echo "SKIP: ${NEW_SBI_HPC} already resolves to a DSN tree."
elif [ -e "${NEW_SBI_HPC}" ]; then
    echo "ERROR: ${NEW_SBI_HPC} exists but does not resolve to <hpc>/dsn/backbone.py -- fix it by hand." >&2
    exit 1
elif [ ! -f "${SBI_HPC_TARGET}/dsn/backbone.py" ]; then
    echo "ERROR: ${SBI_HPC_TARGET}/dsn/backbone.py not found. Set SBI_HPC_TARGET to the" >&2
    echo "       Simulation-Based-Inference clone's hpc/ directory and re-run." >&2
    exit 1
elif [ "${DRY_RUN}" -eq 1 ]; then
    say "mkdir -p ${ARTIFACTS_DIR}"
    say "ln -s '${SBI_HPC_TARGET}' ${NEW_SBI_HPC}"
else
    mkdir -p "${ARTIFACTS_DIR}"
    say "ln -s '${SBI_HPC_TARGET}' ${NEW_SBI_HPC}"
    ln -s "${SBI_HPC_TARGET}" "${NEW_SBI_HPC}"
    echo "OK: artifacts/sbi_hpc -> ${SBI_HPC_TARGET}"
fi

echo ""
echo "== 4. retired dsn_main pointers (nothing reads them any more) =="
for L in "${NEW_DSN_MAIN}" "${OLD_DSN_MAIN}"; do
    if [ -L "${L}" ]; then
        say "rm ${L}   # symlink -> $(readlink "${L}")"
        [ "${DRY_RUN}" -eq 1 ] || rm "${L}"
    elif [ -e "${L}" ]; then
        echo "WARNING: ${L} exists and is not a symlink -- left untouched." >&2
    else
        echo "SKIP: ${L} does not exist."
    fi
done

echo ""
echo "== done =="
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "(dry run -- nothing was actually changed)"
fi
echo "New locations (repo-relative, same from any clone of Sbi-extractor):"
echo "  artifacts/frozen_dsn/*.pt"
echo "  artifacts/specs_real.json"
echo "  artifacts/sbi_hpc -> ${SBI_HPC_TARGET}"
echo ""
echo "env.sh in this repo now supplies SBI_HPC_DIR / SPECS_REAL defaults"
echo "pointing here; nothing under \$HOME is required by these scripts any more."
