# sim_reextract -- Stage C, C8: re-extract the simulated arm

The virtual-MEA detection (`process_campaign.py`, in Astro-Neuron-Network's
`hpc/MEA Traces/`, on davinci `~/ANN/MEA_analysis/`) re-run over the EXISTING
raw DUP15HD simulations into a NEW root, with the electrode count taken from
the real arm's cohort manifest, behind a completion gate that writes a
record. The sim-arm analogue of Stage D (`extractor/launch_stage_d.sh`,
`run_cohort_manifest.pbs`). Decisions: D-009, D-011..D-015 (SC-D1..D5),
D-021, D-023..D-025 in `claude/SBI_decisions_and_ideas_log.md`.

| file | role |
|---|---|
| `sim_reextract_plan.py` | step 1, login node: reads the frozen cohort manifest (sidecar required), sets `n_side = isqrt(electrodes_per_subset)`, enumerates the tasks of the campaign set, names every excluded task, checks one label contract, fingerprints the tools and the template library, writes `plan.json` + `tasks.tsv`. Refusals in its header |
| `launch_sim_reextract.sh` | freezes the manifest into `../artifacts/cohort_manifest/`, runs the plan, submits ANN's `submit_mea_array.sh` (one array member per task, `CONDA_ENV=sbi_export`, the plan's `EXTRA_ARGS`) and the gate job behind it. Modes `plan`, `test`, `array`, `gate`; `DRYRUN=1` prints and submits nothing |
| `sim_reextract_gate.py` | step 3: every planned task against `plan.json` -- every iteration done, the geometry in every `mea_iter_*.npz`, one environment across the run -- then `<out_root>/REEXTRACTION_RECORD.json` (+ `.sha256`), or REFUSED naming every bad task and writing nothing |
| `run_sim_reextract_gate.pbs` | the gate as a job, `depend=afterok:<array>` (ordering only on this PBS: the gate is the success signal) |
| `smoke_test_sim_reextract.py` | 32 checks, end to end on a synthetic simulation tree through the REAL `process_campaign.py`, with a fake `qsub`; Q1-Q6 run a second cohort through `COHORT_TAG` |
| `profiles/<tag>.sh` | another cohort for the launcher (`COHORT_TAG=<tag>`, below); `giulia.sh` is the Giulia project's G2 |
| `c8_diag.py` | after a REFUSED gate, read-only: per bad task the files with data and EMPTY (zero bytes: data lost after the rename), the manifest, host, workers and times from `mea_env.json`, whether `RESUME=1` would re-run or keep it; the passed tasks' timing; `_failures.log`; `qstat -xft` of the array; the job logs' error lines; storage. Run before any launcher mode (it reads the `plan.json` the gate used) |
| `c8_cleanup.py` | after the record (D-065): lists, then with `--delete CODE` deletes, the raw simulations of the replays the record drops, the old `Outputs/` detections of every task the record covers, the moved replay detections, the Giulia real-arm partial roots and (D-066) the DUP15HD pre-Stage-D archives `Deep_bio/extracted/` once every well is in `extracted_v2/` (a NOTE while C0 is not done). Refuses without a record matching its `.sha256`, with an incomplete record task, or when the list changed |
| `smoke_test_c8_tools.py` | 16 checks of the two above on a synthetic tree (no cluster, no ANN tools) |

## On davinci, in order

```bash
# 0. once: the tools folder must hold the 2026-10-06 state of hpc/MEA Traces (G2:
#    one worker per core, D-071; noise seeded per simulation, D-072); the plan refuses
#    the older known states unless --allow-older-tools (the 2026-09-30 one cannot
#    activate sbi_export there; the 2026-10-01 one, C8's, runs one worker per task)
cd ~/repos/Sbi-extractor && module load proxy && git pull --ff-only && git log --oneline -1
cd sim_reextract && conda activate sbi_export
ANN_TOOLS=~/ANN/MEA_analysis python3 smoke_test_sim_reextract.py        # ALL 32 CHECKS PASSED

# 1. the plan (always safe; writes plan.json and tasks.tsv here, nothing under Outputs_v2)
DRYRUN=1 bash launch_sim_reextract.sh plan

# 2. one task, as a plain job, to see the environment and the time per iteration
DRYRUN=1 bash launch_sim_reextract.sh test
bash launch_sim_reextract.sh test                 # TEST_INDEX=k picks another line of tasks.tsv
#    read ~/c8_test.o<jobid>: it must show
#      [mea-array] env activated: /davinci-1/home/ldellamea/.conda/envs/sbi_export/bin/python3
#      [mea-array] python   : 3.11.15 ...   numpy : 2.4.6   scipy : 1.17.1
#      [mea] N topos, a/b iters processed in X s
#    and NOT "putting its bin/ first on PATH" (then scipy does not import: HPC_PATHS.md 7.1)

# 3. the array, over the remaining tasks (RESUME=1: the test task's output is kept)
#    WALLTIME from X s / b iterations of the test task times the plan's largest task, times two
WALLTIME=hh:mm:ss RESUME=1 bash launch_sim_reextract.sh array
#    logs ~/c8_mea.o<jobid>.<index>; the gate's log out/sim_reextract_gate.log

# 4. the verdict: the gate's PASS line, or the tasks to re-run
grep -h 'wrote\|REFUSED\|FAIL\|WARN' out/sim_reextract_gate.log
#    after a walltime kill: RESUME=1 again (complete tasks are kept, the rest re-run), then
bash launch_sim_reextract.sh gate
#    after a REFUSED gate, before anything else (read-only):
python3 c8_diag.py > out/c8_diag_d1.txt 2>&1; tail -1 out/c8_diag_d1.txt     # == D1 done

# 5. once the record exists: what it makes redundant, listed, then deleted (D-065, D-066)
python3 c8_cleanup.py                 # lists into out/c8_cleanup_list.txt, prints the CODE
python3 c8_cleanup.py --delete CODE   # deletes exactly that list
```

Every mode re-runs the plan, with `PLAN_ARGS` passed to it, so a plan flag goes on
every launcher line of a run (`plan`, `test`, `array`), not on the first only.

The plan drops replays by default (D-061): a task whose `job_args.json` records the
same seed values as another task's (`_resolved_seed_master` and the offsets; the
CLI `seed_master` is ignored beside it), under the same label contract, and whose
`iter_*.npz` it shares hold byte-identical `theta` (and `params`) at up to 10 shared
files from the first to the last (D-063). Of a same-seed group the task with the
most iterations is kept (then the lower campaign version, then the lower task
index; D-063), and one task per seed is run (D-064): a replay is dropped whether
its files are the same as, a subset of, or only overlap the kept task's, and the
iteration files only it held are counted (`files_lost`) and printed. A replay is listed under `excluded` with the
reason `replay (D-061) of CAMPAIGN/SWEEP: seed ...; ...`, in the plan and the
record; the plan prints every same-seed group with each member's role, and the
record carries that report (`replays`). Same seed with different theta, a seed
shared across two contracts: kept, and reported. Refused: a replay whose output
folder already holds detections (move it out of the output root; the refusal
prints the `mkdir`/`mv` line) and an unreadable iteration file in a compared pair. The comparison reads
`.npz` members with the standard library, so the plan stays stdlib-only.

Three flags for the plan:

- `--allow-mixed-contract`: proceed although the tasks disagree on a label-contract
  field; the plan and the record name the field and each task's contract signature.
- `--exclude-task CAMPAIGN/SWEEP` (repeatable): leave a task out by name. The plan
  cannot tell a simulation still being written from a complete one with fewer
  `iter_*.npz` -- both are a task with a `manifest.json` and some iterations -- and
  the gate re-counts the raw iterations only when it runs, so a simulation that
  writes nothing between the array's plan and the gate would be recorded as
  complete. Name it here instead; it is listed under `excluded` with the reason
  `left out on the command line (--exclude-task)`, in the plan and in the record.
  A name that matches no task is refused. Find what is being written with
  `find <SIM_MAIN>/campaign_cadex_rho1300v* -name 'iter_*.npz' -mmin -360` and
  `qstat -u $USER`. Example, two flags in one variable:
  `PLAN_ARGS="--allow-mixed-contract --exclude-task campaign_cadex_rho1300v12/sweep_cfd_task0003"`.
- `--keep-replays`: compare and report the replays as above, drop none (for
  comparison only; the printed groups say `would drop`).

Between the array's submission and the gate's log, run no launcher mode: each one
rewrites `plan.json` here, and the gate job reads `plan.json` when it starts, not
when it was submitted.

## Another cohort: `COHORT_TAG` (2026-10-06; the Giulia project's G2)

`COHORT_TAG=<tag>` makes every launcher mode source `profiles/<tag>.sh` after
`../env.sh`. The profile sets that cohort's defaults (an explicit variable still
wins) and the plan's geometry flags, and gives the cohort its own files, so that
two cohorts' runs share none: the plan in `<tag>/plan.json`, `tasks.tsv` and
`tasks_test.tsv`; the frozen manifest in `../artifacts/<PROFILE_FROZEN>/`; the job
names `<PROFILE_JOB>_test`, `_mea`, `_gate`; the gate's log
`out/sim_reextract_gate_<tag>.log` (the gate job's `-o`); `out/submissions_<tag>.txt`.
Unset, the launcher is the C8 one, unchanged. For every cohort, a frozen copy that
is not the manifest of record given is refused before the plan (move the frozen
folder aside only if the manifest of record itself has changed).

`profiles/giulia.sh`: the hhgap simulations, `Main/Giulia_Astro/campaign_cadex_hhgap_v*`
(D-045; the `q/` units sit outside the glob, D-050, and the plan names them under
`outside_glob`); the Giulia cohort manifest, `extracted_giulia/cohort_manifest.json`,
from which `n_side` 1 and `fs` 10000 come (D-051); pitch 200 um (D-042) and edge
26.59 um (D-049), passed to the plan as `--pitch-um`, `--edge-um`, `--decision-pitch`,
`--decision-edge` (their defaults, 60, 25, D-021, D-024, are C8's); the root
`ANN/MEA_analysis/mea_out_giulia_v2`; job names `g2_*`.

```bash
env | grep -E '^(COHORT_MANIFEST|SIM_MAIN|OUT_ROOT|CAMPAIGN_GLOB|TOOLS)='   # nothing: the profile's defaults apply
COHORT_TAG=giulia PLAN_ARGS=--allow-mixed-contract DRYRUN=1 bash launch_sim_reextract.sh plan
COHORT_TAG=giulia PLAN_ARGS=--allow-mixed-contract bash launch_sim_reextract.sh test     # ~/g2_test.o<jobid>
COHORT_TAG=giulia PLAN_ARGS=--allow-mixed-contract WALLTIME=hh:mm:ss RESUME=1 bash launch_sim_reextract.sh array
grep -h 'wrote\|REFUSED\|FAIL\|WARN' out/sim_reextract_gate_giulia.log
```

`--allow-mixed-contract`: hhgap v1 holds two parameter-bound groups; the plan and
the record name the field and each task's signature. The test task's log names the
worker count and where it came from (`[mea-array] workers  : 48 (from NCPUS)`, D-071).

The plan records how the tools seed the noise (`geometry.noise_seed_scheme`, from
the tools' known state: `sim` for the 2026-10-06 tools, D-072; `topo_iter` before),
and the gate checks it in every `mea_manifest.json` and every `mea_iter_*.npz` (a
file without the key was seeded `topo_iter`; a plan without the key, C8's of
2026-10-06, skips the check).

What the record carries: the cohort manifest's digest and `electrodes_per_subset`,
the geometry (`n_side`, pitch, edge, `n_sub`, fs, the noise seeding) with its decisions,
the campaign set, the folders beside it outside the glob, and every task with its topology and iteration counts, the excluded tasks
with their reasons, the tools' label (which state of `hpc/MEA Traces` ran) and
file hashes, the template library's sha256, the environment every task reported
(python, numpy, scipy, env, interpreter), the hosts, the contract, warnings.

## What is deliberately not done here

- No new simulator sweeps: theta and the raw spike times do not change (D-011).
- `Outputs/`, the root of record of the r2-era results, is never written (D-014).
- The output root is never cleaned by these scripts; a partial task is re-run
  in place (`process_campaign.py` rewrites every file atomically), and a
  `_failures.log` left by an earlier attempt is a warning in the record, not a
  deletion.
- The export over `Outputs_v2` (`launch_sweep_exports.sh` with `MEA_ROOT` set to
  it, `LABEL_AXES` re-made over the campaign set) is the next step, not this one.
