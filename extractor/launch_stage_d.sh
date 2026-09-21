#!/bin/bash
# =============================================================================
# launch_stage_d.sh -- re-extract the real cohort into a NEW root and build
# its cohort manifest, as one array job plus one dependent aggregation job.
#
#     cd ~/repos/Sbi-extractor/extractor
#     DRYRUN=1 bash launch_stage_d.sh /davinci-1/home/ldellamea/Deep\ Summary\ Network/Deep_bio/extracted_v2
#     bash launch_stage_d.sh          /davinci-1/home/ldellamea/Deep\ Summary\ Network/Deep_bio/extracted_v2
#
# ALWAYS DRYRUN=1 first. It lists the wells, regenerates the flags, checks
# them against the tracked file, and prints both qsub lines without submitting.
#
# WHAT IT REFUSES
#   - an EXTRACT_ROOT equal to the config's own extract_root (the archives of
#     record). Stage D writes to a NEW directory, never over extracted/.
#   - an EXTRACT_ROOT that already holds a cohort_manifest.json.
#   - a regenerated extraction_flags.sh that differs from the tracked one:
#     the config's cohort.* changed and the flags were not regenerated and
#     committed. Fix that first; the array must source committed flags.
#   - a manifest with fewer than 1 well.
#
# ENVIRONMENT
#   CONFIG    default $SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json
#   ENV_NAME  conda env for both jobs (default sbi_export)
#   DRYRUN=1  print, do not submit
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXTRACT_ROOT="${1:-}"
if [ -z "$EXTRACT_ROOT" ]; then
    sed -n '2,29p' "$0"; exit 2
fi
# shellcheck source=/dev/null
source ../env.sh
CONFIG="${CONFIG:-$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json}"
[ -f "$CONFIG" ] || { echo "ABORT: config not found: $CONFIG"; exit 2; }

# --- refuse to write over the archives of record ----------------------------
declared=$(python3 -c "import sys; sys.path.insert(0,'..'); import cohort_config as CC; print(CC.load_cohort(sys.argv[1]).extract_root)" "$CONFIG") || { echo "ABORT: could not read the config's cohort block"; exit 2; }
norm() { python3 -c "import os,sys; print(os.path.normpath(os.path.abspath(sys.argv[1])))" "$1"; }
if [ "$(norm "$EXTRACT_ROOT")" = "$(norm "$declared")" ]; then
    echo "ABORT: EXTRACT_ROOT is the config's own extract_root:"
    echo "         $declared"
    echo "       That is the archives of record. Stage D writes to a NEW root (extracted_v2/)."
    exit 3
fi
if [ -f "$EXTRACT_ROOT/cohort_manifest.json" ]; then
    echo "ABORT: $EXTRACT_ROOT already holds a cohort_manifest.json; pick a fresh root."
    exit 3
fi

# --- list the wells into a NEW manifest; the flags must match the committed file
python3 list_extraction_jobs.py --config "$CONFIG" --extract-root "$EXTRACT_ROOT" \
    --out-manifest extraction_manifest.tsv --out-flags /tmp/stage_d_flags.$$ || { echo "ABORT: list_extraction_jobs.py failed"; exit 2; }
if ! cmp -s /tmp/stage_d_flags.$$ extraction_flags.sh; then
    echo "ABORT: the flags the config generates now differ from the tracked extraction_flags.sh:"
    diff /tmp/stage_d_flags.$$ extraction_flags.sh | sed 's/^/         /'
    echo "       Regenerate and COMMIT extraction_flags.sh before extracting; the array sources the tracked file."
    rm -f /tmp/stage_d_flags.$$; exit 3
fi
rm -f /tmp/stage_d_flags.$$
n=$(wc -l < extraction_manifest.tsv)
[ "$n" -ge 1 ] || { echo "ABORT: empty manifest"; exit 2; }
mkdir -p out

echo "########################################################################"
echo "# config       : $CONFIG"
echo "# extract root : $EXTRACT_ROOT   (config declares: $declared)"
echo "# wells        : $n   -> array 0-$((n - 1))"
echo "# flags        : $(grep EXTRA_FLAGS extraction_flags.sh)"
echo "# env          : ${ENV_NAME:-sbi_export}"
echo "# mode         : $([ "${DRYRUN:-0}" = "1" ] && echo 'DRY RUN (nothing is submitted)' || echo SUBMIT)"
echo "########################################################################"
ARR_CMD=(qsub -J "0-$((n - 1))" -v "ENV_NAME=${ENV_NAME:-sbi_export}" run_extractor_array_mea.pbs)
echo "  ${ARR_CMD[*]}"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "  qsub -W depend=afterok:<array id> -v EXTRACT_ROOT=$EXTRACT_ROOT,CONFIG=$CONFIG,ENV_NAME=${ENV_NAME:-sbi_export} run_cohort_manifest.pbs"
    echo; echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"; exit 0
fi
arr=$("${ARR_CMD[@]}") || { echo "ABORT: array qsub failed"; exit 2; }
echo "  array : $arr"
agg=$(qsub -W "depend=afterok:${arr}" -v "EXTRACT_ROOT=${EXTRACT_ROOT},CONFIG=${CONFIG},ENV_NAME=${ENV_NAME:-sbi_export}" run_cohort_manifest.pbs) || { echo "ABORT: aggregation qsub failed"; exit 2; }
echo "  agg   : $agg   (held until every array task exits 0)"
echo "$(date -Is) $EXTRACT_ROOT $arr $agg" >> out/stage_d_submissions.txt
echo
echo "watch:   qstat -u \$USER"
echo "then:    grep -h 'wrote\|REFUSED\|cohort_manifest_exit' out/cohort_manifest.log"
