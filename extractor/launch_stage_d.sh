#!/bin/bash
# =============================================================================
# launch_stage_d.sh -- re-extract the real cohort into a NEW root and build
# its cohort manifest, as one array job plus one dependent aggregation job.
#
#     cd ~/repos/Sbi-extractor/extractor
#     DRYRUN=1 bash launch_stage_d.sh /davinci-1/home/ldellamea/Deep\ Summary\ Network/Deep_bio/extracted_v2
#     bash launch_stage_d.sh          /davinci-1/home/ldellamea/Deep\ Summary\ Network/Deep_bio/extracted_v2
#
# A SECOND COHORT (2026-10-01, the Giulia recordings) names its config and a
# tag; the tag selects the cohort's own manifest and flags files, log names
# and job names, so two cohorts never share a file:
#
#     CONFIG=$SBI_HPC_DIR/dsn/hpc/Config/config_giulia_cohort.davinci.json \
#     COHORT_TAG=giulia DRYRUN=1 bash launch_stage_d.sh /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia
#
#     tag          manifest                              flags                        logs
#     (unset)      extraction_manifest.tsv               extraction_flags.sh          chsub_mea_array_<i>.log, out/cohort_manifest.log
#     giulia       out/extraction_manifest_giulia.tsv    extraction_flags_giulia.sh   out/chsub_mea_array_giulia_<i>.log, out/cohort_manifest_giulia.log
#
# ALWAYS DRYRUN=1 first. It lists the wells, regenerates the flags, checks
# them against the tracked file, and prints both qsub lines without submitting.
#
# WHAT IT REFUSES
#   - an EXTRACT_ROOT equal to the config's own extract_root when that root
#     exists and is not empty (the archives of record). Stage D writes to a
#     NEW directory, never over extracted/. A declared root that does not
#     exist yet is a FIRST extraction and is allowed (2026-10-01). A first
#     extraction whose aggregation REFUSED leaves a non-empty declared root
#     without a cohort_manifest.json, and is refused the same way: move it
#     aside yourself before re-running (nothing is ever deleted here;
#     2026-10-02, the Giulia cohort's first run).
#   - an EXTRACT_ROOT that already holds a cohort_manifest.json.
#   - a regenerated flags file that differs from the tracked one: the
#     config's cohort.* changed and the flags were not regenerated and
#     committed. Fix that first; the array must source committed flags.
#   - a cohort whose flags file does not exist yet (a NEW cohort): the file is
#     written from the config and the run stops so you can COMMIT it; the
#     second run then passes this check. Nothing is submitted on that run.
#   - a manifest with fewer than 1 well.
#
# ENVIRONMENT
#   CONFIG      default $SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json
#   COHORT_TAG  default empty (the DUP15HD files); "giulia" for the Giulia cohort
#   ENV_NAME    conda env for both jobs (default sbi_export)
#   DRYRUN=1    print, do not submit
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXTRACT_ROOT="${1:-}"
if [ -z "$EXTRACT_ROOT" ]; then
    sed -n '2,47p' "$0"; exit 2
fi
# shellcheck source=/dev/null
source ../env.sh
CONFIG="${CONFIG:-$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json}"
[ -f "$CONFIG" ] || { echo "ABORT: config not found: $CONFIG"; exit 2; }

# --- the cohort's own files (2026-10-01): a tag, or the DUP15HD names --------
COHORT_TAG="${COHORT_TAG:-}"
case "$COHORT_TAG" in
    "") SUFFIX="" ;;
    *[!A-Za-z0-9_]*) echo "ABORT: COHORT_TAG must be [A-Za-z0-9_]+, got '$COHORT_TAG'"; exit 2 ;;
    *)  SUFFIX="_$COHORT_TAG" ;;
esac
# the DUP15HD manifest keeps its tracked-ignored name; a tagged cohort's
# manifest goes under out/ (gitignored as a whole), its flags file stays
# here because it must be COMMITTED (the array sources the tracked file)
if [ -n "$SUFFIX" ]; then MANIFEST_TSV="out/extraction_manifest${SUFFIX}.tsv"; else MANIFEST_TSV="extraction_manifest.tsv"; fi
FLAGS_SH="extraction_flags${SUFFIX}.sh"
mkdir -p out
case "$EXTRACT_ROOT" in
    *[[:space:]]*) echo "ABORT: EXTRACT_ROOT contains whitespace, which qsub -v cannot carry: '$EXTRACT_ROOT'"; exit 2 ;;
esac

# --- refuse to write over the archives of record ----------------------------
declared=$(python3 -c "import sys; sys.path.insert(0,'..'); import cohort_config as CC; print(CC.load_cohort(sys.argv[1]).extract_root)" "$CONFIG") || { echo "ABORT: could not read the config's cohort block"; exit 2; }
norm() { python3 -c "import os,sys; print(os.path.normpath(os.path.abspath(sys.argv[1])))" "$1"; }
if [ "$(norm "$EXTRACT_ROOT")" = "$(norm "$declared")" ]; then
    # [2026-10-01] a cohort extracted for the FIRST time declares the root it
    # is about to fill (there are no archives of record yet): allowed while
    # that directory is absent or empty. An existing, non-empty declared root
    # is the archives of record and is never written over.
    if [ -d "$declared" ] && [ -n "$(ls -A "$declared" 2>/dev/null)" ]; then
        echo "ABORT: EXTRACT_ROOT is the config's own extract_root:"
        echo "         $declared"
        if [ -f "$declared/cohort_manifest.json" ]; then
            echo "       It holds a cohort_manifest.json: it is the archives of record."
            echo "       Stage D writes to a NEW root (e.g. extracted_v2/)."
        else
            # [2026-10-02] no manifest: either an older extraction's archives
            # of record (DUP15HD's extracted/) or an earlier run of this
            # launcher whose aggregation REFUSED. Only the user can tell.
            echo "       It is not empty and holds no cohort_manifest.json: either the archives of record"
            echo "       of an older extraction, or the output of an earlier run whose aggregation REFUSED."
            echo "       Stage D never writes over it. Pass a NEW root; or, ONLY if it is a refused run's"
            echo "       output, move it aside first (mv <root> <root>_partial_<date>) and re-run."
            echo "       Nothing was deleted or submitted."
        fi
        exit 3
    fi
    echo "NOTE: EXTRACT_ROOT is the config's declared extract_root, which does not exist yet (or is empty): a first extraction."
fi
if [ -f "$EXTRACT_ROOT/cohort_manifest.json" ]; then
    echo "ABORT: $EXTRACT_ROOT already holds a cohort_manifest.json; pick a fresh root."
    exit 3
fi

# --- list the wells into a NEW manifest; the flags must match the committed file
python3 list_extraction_jobs.py --config "$CONFIG" --extract-root "$EXTRACT_ROOT" \
    --out-manifest "$MANIFEST_TSV" --out-flags /tmp/stage_d_flags.$$ || { echo "ABORT: list_extraction_jobs.py failed"; exit 2; }
if [ ! -f "$FLAGS_SH" ]; then
    # a NEW cohort: its flags file does not exist yet. Write it from the
    # config, show it, and stop: the array must source a COMMITTED file.
    cp /tmp/stage_d_flags.$$ "$FLAGS_SH"; rm -f /tmp/stage_d_flags.$$
    echo "NEW COHORT: wrote $FLAGS_SH from the config's cohort block:"
    sed 's/^/         /' "$FLAGS_SH"
    echo "       Review it, COMMIT it (git add $FLAGS_SH && git commit), then re-run this launcher."
    echo "       Nothing was submitted."
    exit 3
fi
if ! cmp -s /tmp/stage_d_flags.$$ "$FLAGS_SH"; then
    echo "ABORT: the flags the config generates now differ from the tracked $FLAGS_SH:"
    diff /tmp/stage_d_flags.$$ "$FLAGS_SH" | sed 's/^/         /'
    echo "       Regenerate and COMMIT $FLAGS_SH before extracting; the array sources the tracked file."
    rm -f /tmp/stage_d_flags.$$; exit 3
fi
rm -f /tmp/stage_d_flags.$$
n=$(wc -l < "$MANIFEST_TSV")
[ "$n" -ge 1 ] || { echo "ABORT: empty manifest"; exit 2; }

echo "########################################################################"
echo "# config       : $CONFIG"
echo "# cohort tag   : ${COHORT_TAG:-(none: the DUP15HD files)}"
echo "# extract root : $EXTRACT_ROOT   (config declares: $declared)"
echo "# wells        : $n   -> array 0-$((n - 1))   ($MANIFEST_TSV)"
echo "# flags        : $(grep EXTRA_FLAGS "$FLAGS_SH")   ($FLAGS_SH)"
echo "# env          : ${ENV_NAME:-sbi_export}"
echo "# mode         : $([ "${DRYRUN:-0}" = "1" ] && echo 'DRY RUN (nothing is submitted)' || echo SUBMIT)"
echo "#"
echo "# depend=afterok on an array is ORDERING ONLY on this PBS, measured"
echo "# 2026-09-21 (probe_array_depend.sh): a subjob may exit 1 and the"
echo "# aggregation still runs. It is cohort_manifest.py that refuses a"
echo "# partial cohort, naming every well to re-extract. Read the PASS line"
echo "# in out/cohort_manifest${SUFFIX}.log; do not infer success from the array."
echo "########################################################################"
VARS_ARR="ENV_NAME=${ENV_NAME:-sbi_export}"
VARS_AGG="EXTRACT_ROOT=${EXTRACT_ROOT},CONFIG=${CONFIG},ENV_NAME=${ENV_NAME:-sbi_export}"
ARR_CMD=(qsub -J "0-$((n - 1))")
AGG_OPTS=()
if [ -n "$SUFFIX" ]; then
    # a tagged cohort gets its own job names and log files; the files it reads
    # travel with -v (the .pbs files default to the DUP15HD names when unset)
    VARS_ARR="${VARS_ARR},CHSUB_MANIFEST=${MANIFEST_TSV},CHSUB_FLAGS=${FLAGS_SH}"
    VARS_AGG="${VARS_AGG},CHSUB_MANIFEST=${MANIFEST_TSV},CHSUB_FLAGS=${FLAGS_SH}"
    ARR_CMD+=(-N "chsub_mea_array${SUFFIX}" -o "out/chsub_mea_array${SUFFIX}_^array_index^.log")
    AGG_OPTS=(-N "cohort_manifest${SUFFIX}" -o "out/cohort_manifest${SUFFIX}.log")
fi
ARR_CMD+=(-v "$VARS_ARR" run_extractor_array_mea.pbs)
echo "  ${ARR_CMD[*]}"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "  qsub -W depend=afterok:<array id> ${AGG_OPTS[*]+"${AGG_OPTS[*]} "}-v $VARS_AGG run_cohort_manifest.pbs"
    echo; echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"; exit 0
fi
arr=$("${ARR_CMD[@]}") || { echo "ABORT: array qsub failed"; exit 2; }
echo "  array : $arr"
# ${AGG_OPTS[@]+...}: an empty array expanded under set -u is an "unbound
# variable" on bash < 4.4; the + form expands to nothing in that case.
agg=$(qsub -W "depend=afterok:${arr}" ${AGG_OPTS[@]+"${AGG_OPTS[@]}"} -v "$VARS_AGG" run_cohort_manifest.pbs) || { echo "ABORT: aggregation qsub failed"; exit 2; }
# [corrected 2026-10-02] this line said "(held until every array task exits
# 0)", which the 2026-09-21 measurement above contradicts (D-006).
echo "  agg   : $agg   (runs once the array has ended, whatever its tasks' exit codes)"
echo "$(date -Is) ${COHORT_TAG:-dup15hd} $EXTRACT_ROOT $arr $agg" >> out/stage_d_submissions.txt
echo
echo "watch:   qstat -u \$USER"
# [2026-10-02] the whole log, not a grep: a grep for the headline hid the
# list of wells under REFUSED (the Giulia cohort's first run).
echo "then:    cat out/cohort_manifest${SUFFIX}.log"
echo "         PASS = a line 'wrote <root>/cohort_manifest.json  sha256 <16 hex>'; a REFUSED names every"
echo "         well it could not use, each with its array index; the last line is cohort_manifest_exit=<rc>."
