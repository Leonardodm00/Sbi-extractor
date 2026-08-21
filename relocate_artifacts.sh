#!/bin/bash
# relocate_artifacts.sh -- one-time migration: move the frozen DSN
# checkpoint, the real-cohort specs, and the dsn_main pointer out of
# $HOME and into this repo's artifacts/ directory.
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

NEW_FROZEN_DSN="${ARTIFACTS_DIR}/frozen_dsn"
NEW_SPECS_REAL="${ARTIFACTS_DIR}/specs_real.json"
NEW_DSN_MAIN="${ARTIFACTS_DIR}/dsn_main"

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
echo "== 3. dsn_main pointer =="
if [ -L "${OLD_DSN_MAIN}" ]; then
    TARGET="$(readlink "${OLD_DSN_MAIN}")"
    if [ -e "${NEW_DSN_MAIN}" ]; then
        echo "SKIP: ${NEW_DSN_MAIN} already exists -- not overwriting."
    elif [ "${DRY_RUN}" -eq 1 ]; then
        say "mkdir -p ${ARTIFACTS_DIR}"
        say "ln -s '${TARGET}' ${NEW_DSN_MAIN}"
        say "verify it resolves, THEN rm ${OLD_DSN_MAIN}"
    else
        mkdir -p "${ARTIFACTS_DIR}"
        say "ln -s '${TARGET}' ${NEW_DSN_MAIN}"
        ln -s "${TARGET}" "${NEW_DSN_MAIN}"
        if [ -d "${NEW_DSN_MAIN}" ]; then
            echo "OK: new symlink resolves (artifacts/dsn_main -> ${TARGET})"
            say "rm ${OLD_DSN_MAIN}"
            rm "${OLD_DSN_MAIN}"
        else
            echo "ERROR: new symlink does not resolve to a directory -- NOT removing the old one." >&2
            echo "       target was: ${TARGET}" >&2
            exit 1
        fi
    fi
elif [ -e "${OLD_DSN_MAIN}" ]; then
    echo "ERROR: ${OLD_DSN_MAIN} exists but is not a symlink -- refusing to touch it." >&2
    echo "       Expected the symlink documented in HANDOFF.md section 2." >&2
    exit 1
else
    echo "SKIP: ${OLD_DSN_MAIN} does not exist (already migrated, or never existed here)."
fi

echo ""
echo "== done =="
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "(dry run -- nothing was actually changed)"
fi
echo "New locations (repo-relative, same from any clone of Sbi-extractor):"
echo "  artifacts/frozen_dsn/*.pt"
echo "  artifacts/specs_real.json"
echo "  artifacts/dsn_main -> (unchanged real target)"
echo ""
echo "env.sh in this repo now supplies DSN_MAIN_DIR / SPECS_REAL defaults"
echo "pointing here; nothing under \$HOME is required by these scripts any more."
