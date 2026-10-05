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
| `smoke_test_sim_reextract.py` | 26 checks, end to end on a synthetic simulation tree through the REAL `process_campaign.py`, with a fake `qsub` |

## On davinci, in order

```bash
# 0. once: the tools folder must hold the 2026-10-01 submit_mea_array.sh
#    (the plan refuses the 2026-09-30 one: it cannot activate sbi_export there)
cd ~/repos/Sbi-extractor && module load proxy && git pull --ff-only && git log --oneline -1
cd sim_reextract && conda activate sbi_export
ANN_TOOLS=~/ANN/MEA_analysis python3 smoke_test_sim_reextract.py        # ALL 26 CHECKS PASSED

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

What the record carries: the cohort manifest's digest and `electrodes_per_subset`,
the geometry (`n_side`, pitch, edge, `n_sub`, fs) with its decisions, the campaign
set and every task with its topology and iteration counts, the excluded tasks
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
