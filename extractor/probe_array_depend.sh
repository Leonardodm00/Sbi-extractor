#!/bin/bash
# =============================================================================
# probe_array_depend.sh -- on THIS PBS, does a dependent job run when a subjob
# of the array it depends on has exited 1?
#
#   bash probe_array_depend.sh pass     all 3 subjobs succeed
#   bash probe_array_depend.sh fail     subjob 1 exits 1
#   bash probe_array_depend.sh check    verdict for every run so far
#
# Stage D's aggregation job (run_cohort_manifest.pbs) is submitted with
# depend=afterok:<array id> by launch_stage_d.sh. If that does not gate on
# subjob success, the aggregation runs over a partial extraction, and the only
# thing standing between a partial cohort and a written manifest is
# cohort_manifest.py's own assertions. Which of those two protects us is a
# fact about this scheduler, not something to assume.
#
# THE KEYWORD, measured [CLUSTER 2026-09-21]. afterokarray is a TORQUE
# dependency type; PBS Pro has no *array variants and rejects the whole -W
# value with "qsub: illegal -W value". Swept on davinci against a held array
# 1723609[]:
#
#   afterok:1723609[].pbsserver01        ACCEPTED
#   afterany:1723609[].pbsserver01       ACCEPTED
#   afterokarray:1723609[].pbsserver01   illegal -W value
#   afterok:1723609[]                    ACCEPTED   (server suffix optional)
#
# TWO DESIGN FAULTS THIS VERSION FIXES, both of which corrupted the first
# attempt on 2026-09-21 and between them cost three rounds:
#
#   1. ONE SHARED OUTPUT PATH. All three subjobs had the same #PBS -o, ran
#      concurrently and overwrote each other from byte zero. The surviving
#      106 bytes held task 0's lines plus the tail of task 1's failure
#      message. Each subjob now writes its own log.
#   2. NO RUN ISOLATION. State was cleared at SUBMIT time, so a previous
#      run's array -- still running -- wrote its .ok files afterwards, into
#      the new run's evidence. A leftover task_1.ok then read as "the
#      deliberate failure did not fire". Every run now has its own directory
#      under out/probe/<RUN>, nothing is ever cleared, and a new run REFUSES
#      to start while any probe job of a previous run is still in the queue.
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"
# USER is not guaranteed to be exported -- it is not, inside a PBS job, and a
# safety check that dies on an unset variable is worse than no check at all.
ME="${USER:-$(id -un)}"

in_flight() {
    # job ids of any probe job still known to the scheduler, one per line
    if command -v qselect >/dev/null 2>&1; then
        { qselect -u "$ME" -N probe_dep_task 2>/dev/null
          qselect -u "$ME" -N probe_dep_agg  2>/dev/null; } | sed '/^$/d'
    else
        qstat -u "$ME" 2>/dev/null | awk '/probe_dep/ {print $1}'
    fi
}

field() {   # field <jobid> <attribute>   -> value, or "?" if unknown
    qstat -xf "$1" 2>/dev/null | tr ',' '\n' \
        | sed -n "s/^[[:space:]]*$2 = //p" | head -1 | sed 's/[[:space:]]*$//'
}

case "$mode" in
  pass|fail)
    mkdir -p out/probe
    busy="$(in_flight)"
    if [ -n "$busy" ]; then
        echo "ABORT: a previous probe run is still in the queue:"
        printf '  %s\n' $busy
        echo "       Wait for it (qstat -u $ME) or qdel it. Starting now would"
        echo "       let its jobs write into this run's evidence, which is exactly"
        echo "       how the 2026-09-21 run became unreadable."
        exit 3
    fi
    RUN="$(date +%Y%m%d_%H%M%S)_$$"
    RUNDIR="out/probe/$RUN"
    mkdir -p "$RUNDIR"
    FAIL_TASK="none"; [ "$mode" = "fail" ] && FAIL_TASK="1"
    arr=$(qsub -J 0-2 -v "RUN=${RUN},FAIL_TASK=${FAIL_TASK}" probe_array_depend_task.pbs) \
        || { echo "ABORT: array qsub failed"; exit 2; }
    agg=$(qsub -W "depend=afterok:${arr}" -v "RUN=${RUN}" probe_array_depend_agg.pbs) \
        || { echo "ABORT: agg qsub failed (array $arr is still queued; qdel it)"; exit 2; }
    printf '%s\t%s\t%s\t%s\n' "$mode" "$RUN" "$arr" "$agg" >> out/probe/submissions.tsv
    echo "run   : $RUN   mode=$mode  FAIL_TASK=$FAIL_TASK"
    echo "array : $arr"
    echo "agg   : $agg   (depend=afterok:${arr})"
    echo
    echo "wait until 'qstat -u $ME' shows no probe_dep job, then:"
    echo "    bash probe_array_depend.sh check"
    ;;

  check)
    [ -f out/probe/submissions.tsv ] || { echo "no runs yet."; exit 0; }
    busy="$(in_flight)"
    [ -n "$busy" ] && { echo "NOTE: probe jobs still in the queue; verdicts below may be premature:"; printf '  %s\n' $busy; echo; }
    while IFS=$'\t' read -r m run arr agg; do
        d="out/probe/$run"
        oks=$(find "$d" -maxdepth 1 -name 'task_*.ok' 2>/dev/null | wc -l)
        fails=$(find "$d" -maxdepth 1 -name 'task_*.failed' 2>/dev/null | wc -l)
        ran="no"; [ -f "$d/agg.ran" ] && ran="yes"
        a_state=$(field "$arr" job_state); a_exit=$(field "$arr" Exit_status)
        g_state=$(field "$agg" job_state); g_exit=$(field "$agg" Exit_status)
        echo "--- run $run   mode=$m"
        echo "    array  $arr   job_state=${a_state:-?}  Exit_status=${a_exit:-?}"
        for i in 0 1 2; do
            s=$(field "${arr%%[*}[$i]" job_state); e=$(field "${arr%%[*}[$i]" Exit_status)
            printf '      subjob %s  job_state=%-3s Exit_status=%-3s  %s\n' \
                   "$i" "${s:-?}" "${e:-?}" \
                   "$( [ -f "$d/task_$i.ok" ] && echo .ok; [ -f "$d/task_$i.failed" ] && echo .failed )"
        done
        echo "    agg    $agg   job_state=${g_state:-?}  Exit_status=${g_exit:-?}   ran=${ran}"
        [ -f "$d/agg.ran" ] && echo "           $(cat "$d/agg.ran")"
        if [ "$m" = "fail" ]; then
            if [ "$fails" -eq 0 ]; then
                echo "    VERDICT: INCONCLUSIVE -- the deliberate failure did not fire."
            elif [ "$ran" = "yes" ]; then
                echo "    VERDICT: NOT GATED -- a subjob exited 1 and the dependent RAN anyway."
                echo "             afterok on an array gives ORDERING, not success-gating."
            else
                echo "    VERDICT: GATED -- a subjob exited 1 and the dependent did not run."
            fi
        else
            if [ "$oks" -eq 3 ] && [ "$ran" = "yes" ]; then
                echo "    VERDICT: OK -- all 3 subjobs succeeded and the dependent ran."
            else
                echo "    VERDICT: INCONCLUSIVE -- oks=$oks ran=$ran (still running?)."
            fi
        fi
        echo
    done < out/probe/submissions.tsv
    ;;

  clean)
    echo "removing out/probe/ (all runs)"; rm -rf out/probe; ;;

  *)
    echo "usage: bash probe_array_depend.sh {pass|fail|check|clean}"; exit 2 ;;
esac
