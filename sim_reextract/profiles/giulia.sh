# shellcheck shell=bash
# shellcheck disable=SC2034  # every variable here is read by the launcher that sources it
# =============================================================================
# profiles/giulia.sh -- COHORT_TAG=giulia: the Giulia project's G2, the
# re-detection of the hhgap simulations (claude/PLAN_2026-10-06_giulia_G2.md
# in project knowledge; decisions D-042, D-045, D-048..D-051, D-069..D-072).
# Sourced by launch_sim_reextract.sh after ../env.sh, before its defaults:
# each `: "${VAR:=...}"` sets a default an explicit variable still overrides.
#
# HPC note (hpc-python-compat): pure ASCII, LF only.
# =============================================================================

# the Giulia cohort of record (G1, 2026-10-02: 16 wells, electrodes_per_subset
# 1, fs_raw 10000.0, sha256 21b7ef03bb7f776b), so n_side 1 and fs 10000 come
# from the manifest, never from here (D-051)
: "${COHORT_MANIFEST:=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/extracted_giulia/cohort_manifest.json}"
# the hhgap simulations (D-045); the q/ units sit outside the glob (D-050)
: "${SIM_MAIN:=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro}"
: "${CAMPAIGN_GLOB:=campaign_cadex_hhgap_v*}"
# the new root, beside mea_out/ and mea_out_1electrode/ (D-046)
: "${OUT_ROOT:=/davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_giulia_v2}"

# the Giulia device: 200 um pitch (D-042), a 26.59 um square electrode, the
# area of a 30 um disc (D-049)
PROFILE_GEOM_ARGS="--pitch-um 200 --edge-um 26.59 --decision-pitch D-042 --decision-edge D-049"

# this cohort's own files, so that no file is shared with the C8 run
PROFILE_WORK="giulia"                    # sim_reextract/giulia/plan.json, tasks.tsv, tasks_test.tsv
PROFILE_FROZEN="cohort_manifest_giulia"  # $ARTIFACTS_DIR/cohort_manifest_giulia/
PROFILE_JOB="g2"                         # job names g2_test, g2_mea, g2_gate
