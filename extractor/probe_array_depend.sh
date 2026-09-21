#!/bin/bash
# =============================================================================
# probe_array_depend.sh -- does PBS Pro's depend=afterok on an ARRAY job do
# what Stage D needs? Two runs answer it:
#
#   bash probe_array_depend.sh pass     all 3 tasks succeed  -> agg MUST run
#   bash probe_array_depend.sh fail     task 1 exits 1       -> agg MUST NOT run
#
# Run them one after the other (each clears out/probe_* first). Then:
#
#   bash probe_array_depend.sh check
#
# KEYWORD, measured [CLUSTER 2026-09-21]. This file was first written with
# depend=afterokarray, which is a TORQUE dependency type; PBS Pro has no
# *array variants at all and rejects the whole -W value with
# "qsub: illegal -W value". In PBS Pro an array dependency uses the ORDINARY
# keyword with an array job id carrying the brackets. Swept on davinci
# against a held array 1723609[]:
#
#   afterok:1723609[].pbsserver01        ACCEPTED
#   afterany:1723609[].pbsserver01       ACCEPTED
#   afterokarray:1723609[].pbsserver01   illegal -W value
#   afterok:1723609[]                    ACCEPTED   (server suffix optional)
#
# so the form used here, and by launch_stage_d.sh, is afterok:<array id> with
# the id exactly as qsub -J printed it.
#
# prints what happened. The pass run must show probe_agg.ran with n_ok=3; the
# fail run must show NO probe_agg.ran and the agg job in state F with a
# dependency-failure comment in qstat -xf (PBS Pro deletes a dependent job
# whose dependency can no longer be satisfied).
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"

case "$mode" in
  pass|fail)
    mkdir -p out
    rm -f out/probe_task_*.ok out/probe_agg.ran out/probe_aoa_*.log
    extra=""
    [ "$mode" = "fail" ] && extra="-v FAIL_TASK=1"
    # shellcheck disable=SC2086
    arr=$(qsub -J 0-2 $extra probe_array_depend_task.pbs) || { echo "ABORT: array qsub failed"; exit 2; }
    echo "array : $arr"
    agg=$(qsub -W "depend=afterok:${arr}" probe_array_depend_agg.pbs) || { echo "ABORT: agg qsub failed"; exit 2; }
    echo "agg   : $agg   (depend=afterok:${arr})"
    echo "$mode $arr $agg" >> out/probe_aoa_submissions.txt
    echo
    echo "wait ~1 min, then:   bash probe_array_depend.sh check"
    ;;
  check)
    echo "== submissions =="; cat out/probe_aoa_submissions.txt 2>/dev/null || echo "(none)"
    echo; echo "== task .ok files =="; ls -1 out/probe_task_*.ok 2>/dev/null || echo "(none)"
    echo; echo "== agg ran? =="; cat out/probe_agg.ran 2>/dev/null || echo "NO -- out/probe_agg.ran absent"
    echo; echo "== agg job states =="
    while read -r m a g; do
        printf '  %-4s agg %-22s ' "$m" "$g"
        qstat -xf "$g" 2>/dev/null | grep -E "job_state|Exit_status|comment" | tr -s ' ' | tr '\n' ' '
        echo
    done < out/probe_aoa_submissions.txt 2>/dev/null
    ;;
  *)
    echo "usage: bash probe_array_depend.sh {pass|fail|check}"; exit 2 ;;
esac
