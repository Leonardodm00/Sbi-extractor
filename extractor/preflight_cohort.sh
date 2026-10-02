#!/bin/bash
# =============================================================================
# preflight_cohort.sh -- login-node checks before a cohort's Stage D launch.
#
#     cd ~/repos/Sbi-extractor/extractor && source ../env.sh && conda activate sbi_export
#     bash preflight_cohort.sh "$SBI_HPC_DIR/dsn/hpc/Config/config_giulia_cohort.davinci.json" giulia
#
# Arguments: CONFIG (a DSN JSON config with a cohort block) and the COHORT_TAG
# its launch will use (omit or "" for the DUP15HD files). Read-only: nothing
# outside a temporary directory is written, nothing is submitted.
#
# Steps, each printing PASS or FAIL:
#   1 line endings  -- no CR byte in any .sh/.pbs of this repository
#   2 encoding      -- every .py of this repository is pure ASCII
#   3 compile       -- the extractor modules and the cohort/manifest modules
#   4 DSN tree      -- SBI_HPC_DIR resolves to a tree holding cohort.py with
#                      the source-format fields (the SBI commit is pulled)
#   5 config        -- the cohort block loads (CohortConfig validation) and the
#                      tracked flags file for the tag is byte-identical to what
#                      the config generates
#   6 DUP15HD flags -- extraction_flags.sh is still byte-identical to what
#                      config_mea_joint_full.davinci.json generates
#   7 suites        -- smoke_test_ptrain_formats, smoke_test_stage_d_jobs,
#                      smoke_test_extraction_metadata, smoke_test_cohort_manifest,
#                      the DSN tree's smoke_test_cohort_fields (~3 min in all)
#
# PASS CONDITION, the last line: "PREFLIGHT PASS (7/7)". Anything else names
# the failed steps; paste the whole output.
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
EXT="$PWD"
REPO="$(cd .. && pwd)"

CONFIG="${1:-}"
TAG="${2:-}"
if [ -z "$CONFIG" ]; then
    sed -n '2,30p' "$0"; exit 2
fi
[ -f "$CONFIG" ] || { echo "ABORT: config not found: $CONFIG"; exit 2; }
case "$TAG" in
    "") FLAGS_SH="extraction_flags.sh" ;;
    *[!A-Za-z0-9_]*) echo "ABORT: tag must be [A-Za-z0-9_]+, got '$TAG'"; exit 2 ;;
    *)  FLAGS_SH="extraction_flags_${TAG}.sh" ;;
esac
# shellcheck source=/dev/null
source "$REPO/env.sh"

echo "########################################################################"
echo "# preflight  : $(date -Is)  host $(hostname)"
echo "# repo       : $REPO  ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git clone'))"
echo "# config     : $CONFIG"
echo "# tag        : ${TAG:-(none)}  -> $FLAGS_SH"
echo "# python3    : $(command -v python3)  $(python3 --version 2>&1)"
echo "# conda env  : ${CONDA_DEFAULT_ENV:-none}   (expect sbi_export)"
echo "# SBI_HPC_DIR: $SBI_HPC_DIR"
echo "########################################################################"

PASSED=0
FAILED=()
step() {   # step <n> <label> <command...>
    local n="$1" label="$2"; shift 2
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$n] PASS  $label"
        [ -n "$out" ] && printf '%s\n' "$out" | tail -6 | sed 's/^/        /'
        PASSED=$((PASSED + 1))
    else
        echo "[$n] FAIL  $label (exit $rc)"
        printf '%s\n' "$out" | tail -25 | sed 's/^/        /'
        FAILED+=("$n")
    fi
}

check_cr() {
    python3 - "$REPO" <<'PY'
import os, sys
bad = []
for root, dirs, files in os.walk(sys.argv[1]):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "artifacts", "out")]
    for f in files:
        if f.endswith((".sh", ".pbs", ".slurm")):
            p = os.path.join(root, f)
            n = open(p, "rb").read().count(b"\r")
            if n:
                bad.append("%s (%d CR)" % (p, n))
print("\n".join(bad) if bad else "every .sh/.pbs is LF-only")
sys.exit(1 if bad else 0)
PY
}
check_ascii() {
    python3 - "$REPO" <<'PY'
import os, sys
bad = []
for root, dirs, files in os.walk(sys.argv[1]):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "artifacts", "out")]
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            if any(c > 127 for c in open(p, "rb").read()):
                bad.append(p)
print("\n".join(bad) if bad else "every .py is pure ASCII")
sys.exit(1 if bad else 0)
PY
}
check_compile() {
    python3 -m py_compile "$REPO/cohort_config.py" "$REPO/cohort_manifest.py" \
        "$EXT/channel_subset_extraction.py" "$EXT/run_channel_subset_extraction.py" \
        "$EXT/list_extraction_jobs.py" "$EXT/probe_ptrain_tree.py" \
        "$EXT/channel_subset_viz.py" && echo "compiles"
}
check_dsn() {
    python3 "$REPO/dsn_tree.py" | tail -1 || return 1
    grep -q "validate_ptrain_fields" "$SBI_HPC_DIR/dsn/cohort.py" \
        || { echo "$SBI_HPC_DIR/dsn/cohort.py has no source-format fields: pull the SBI commit"; return 1; }
    echo "cohort.py carries the source-format fields"
}
check_config() {
    python3 - "$REPO" "$CONFIG" "$EXT/$FLAGS_SH" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
import cohort_config as CC
c = CC.load_cohort(sys.argv[2])
want = CC.build_extra_flags(c).encode("ascii")
p = sys.argv[3]
if not os.path.isfile(p):
    print("tracked flags file missing: %s" % p); sys.exit(1)
if open(p, "rb").read() != want:
    print("%s differs from what the config generates:" % p)
    print(want.decode().strip().splitlines()[-1]); sys.exit(1)
print("%d class(es) %r; %s == build_extra_flags(config)"
      % (c.n_classes(), [c.name_of_class(i) for i in range(c.n_classes())], os.path.basename(p)))
PY
}
check_dup15hd() {
    python3 - "$REPO" "$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json" "$EXT/extraction_flags.sh" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import cohort_config as CC
ok = open(sys.argv[3], "rb").read() == CC.build_extra_flags(CC.load_cohort(sys.argv[2])).encode("ascii")
print("extraction_flags.sh byte-identical to the DUP15HD config's flags" if ok
      else "extraction_flags.sh DIFFERS from the DUP15HD config's flags")
sys.exit(0 if ok else 1)
PY
}
check_suites() {
    local rc=0
    ( cd "$EXT" && python3 smoke_test_ptrain_formats.py | tail -1 ) || rc=1
    ( cd "$EXT" && python3 smoke_test_stage_d_jobs.py | tail -1 ) || rc=1
    ( cd "$EXT" && python3 smoke_test_extraction_metadata.py | tail -1 ) || rc=1
    ( cd "$REPO" && python3 smoke_test_cohort_manifest.py | tail -1 ) || rc=1
    ( cd "$SBI_HPC_DIR/dsn" && PYTHONPATH="$PWD" python3 Smoke_Tests/smoke_test_cohort_fields.py | tail -1 ) || rc=1
    return $rc
}

step 1 "line endings"   check_cr
step 2 "encoding"       check_ascii
step 3 "compile"        check_compile
step 4 "DSN tree"       check_dsn
step 5 "config + flags" check_config
step 6 "DUP15HD flags"  check_dup15hd
step 7 "suites"         check_suites

echo "------------------------------------------------------------------------"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "PREFLIGHT PASS ($PASSED/7)"
    exit 0
fi
echo "PREFLIGHT FAIL: step(s) ${FAILED[*]}  ($PASSED/7 passed)"
exit 1
