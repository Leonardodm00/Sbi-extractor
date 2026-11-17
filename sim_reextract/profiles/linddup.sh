# shellcheck shell=bash
# shellcheck disable=SC2034  # every variable here is read by the launcher that sources it
# =============================================================================
# profiles/linddup.sh -- COHORT_TAG=linddup: the LindDUP subproject's
# re-detection of the lcdup15 campaign at the DUP15HD probe (S2 of the
# Simulation-Based-Inference pipeline, hpc/pipeline/projects/linddup/;
# decisions D-104..D-107 of claude/SBI_decisions_and_ideas_log.md in project
# knowledge; since 2026-11-17 D-138..D-151 of docs/project/
# SBI_decisions_and_ideas_log.md on SBI main: the campaign is LindaColloDUP15,
# lcdup15_rho1300_neur_n01, which replaced MEAMA, D-138). Sourced by
# launch_sim_reextract.sh after ../env.sh, before its
# defaults: each `: "${VAR:=...}"` sets a default an explicit variable still
# overrides (the pipeline passes all four explicitly, from linddup.json).
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================

# the DUP15HD cohort of record (Deep_bio/extracted_v2, sha256 c925c5106137fe82:
# 35 wells, 9 subregions of 9 electrodes), so n_side 3 and fs 10110.09 come
# from the manifest, never from here (D-015, D-051). The path holds a space:
# the launcher freezes a copy into $ARTIFACTS_DIR/cohort_manifest_linddup/
# (space-free) before the plan reads it.
: "${COHORT_MANIFEST:=/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted_v2/cohort_manifest.json}"
# the lcdup15 campaign (D-138): its launch folder in the ANN checkout, the
# simulator source beside the campaign (the folder's patched copies)
: "${SIM_MAIN:=/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP}"
: "${CAMPAIGN_GLOB:=campaign_lcdup15_rho1300_neur_n01}"
# a new root of its own (D-104: linddup in every folder name; lcdup15 since
# D-138, so that nothing of the MEAMA configuration can be mistaken for it)
: "${OUT_ROOT:=/davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_linddup_lcdup15}"

# the DUP15HD device, as C8 used it: 60 um pitch (D-021), a 25 um square
# electrode (D-024) -- the planner's own defaults, written out so that the
# record names them
PROFILE_GEOM_ARGS="--pitch-um 60 --edge-um 25 --decision-pitch D-021 --decision-edge D-024"

# this cohort's own files, so that no file is shared with C8 or with Giulia
PROFILE_WORK="linddup"                    # sim_reextract/linddup/plan.json, tasks.tsv, tasks_test.tsv
PROFILE_FROZEN="cohort_manifest_linddup"  # $ARTIFACTS_DIR/cohort_manifest_linddup/ (read by the pipeline's S4)
PROFILE_JOB="ld"                          # job names ld_test, ld_mea, ld_gate
