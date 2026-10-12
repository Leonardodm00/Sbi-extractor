# export_windows -- the Giulia window banks (B5)

The two Stage 1 banks of the joint DSN+NPE stack's arm A1 for the Giulia
project, in the Simulation-Based-Inference repository's `latent_bank` format:
the simulated bank (`~/banks/giulia_sim/`, one row per re-detected simulation
of the G2 record whose window passes the 0.1 Hz floor) and the real bank
(`~/banks/giulia_real/`, the cohort's 180 s windows). Specification:
`specs/SPEC.md`, secs. 2.5-2.6 and Blocks 6-8. Operating map:
`EXTRACTOR_USAGE.md`, sec. 16. The decisions it implements are cited by ID
(D-041..D-080) in the project's decision log.

## Run order (davinci, login node, `sbi_export`)

```
cd ~/repos/Sbi-extractor/export_windows && source ../env.sh && conda activate sbi_export
bash launch_bank.sh plan                      # last line: BANK PLAN OK (estimates above it)
DRYRUN=1 bash launch_bank.sh test             # the test task's qsub line; nothing submitted
bash launch_bank.sh test                      # giulia/logs/bank_test_0.log ends: TASK 0 OK
WALLTIME=auto bash launch_bank.sh array       # every task + the gate job
tail -n 3 giulia/logs/bank_gate.log           # wrote <bank_root>/BANK_RECORD.json  sha256 <16 hex>
bash launch_bank.sh real                      # last line: REAL BANK OK
```

The launcher's header lists its variables. One bank folder holds one plan's
bank; nothing here deletes or overwrites a written bank.

## Files

| file | block | role |
|---|---|---|
| `bank_plan_config.py`, `profiles/giulia.json` | 6.1 | the profile: inputs with their digest prefixes, decided and proposed values, outputs |
| `bank_plan_load.py` | 6.2 | the record, freeze and cohort manifest against their digests; every task's contract, seeds and detections |
| `bank_plan_model.py` | 6.3 | the contract rule, the label columns, twins and donors, the shard layout |
| `bank_row.py` | 6.4a | one simulation's row: window, rate, floor, label in the unit box |
| `bank_plan_sample.py` | 6.4 | one simulation per topology, the twin check, the estimates |
| `bank_plan.py` | 6.5 | the plan step: writes `giulia/bank_plan.json` (gitignored) |
| `bank_export.py`, `bank_task.py` | 7.1, 7.2 | one task's rows, shards and marker; the array element |
| `bank_gate.py` | 7.3 | the gate; writes `BANK_RECORD.json` |
| `launch_bank.sh`, `run_bank_task.pbs`, `run_bank_gate.pbs` | 7.4 | launcher and job scripts |
| `bank_real.py` | 8 | the real bank |
| `bank_plan_fixture.py` | tests | the synthetic record and cohort |
| `smoke_test_bank_plan.py`, `smoke_test_bank_export.py` | tests | Block 6; Blocks 7-8 (fake qsub and conda) |

## Tests

```
export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc    # or: source ../env.sh
python3 smoke_test_bank_plan.py      # Block 6
python3 smoke_test_bank_export.py    # Blocks 7 and 8; the runner's dry run only where sbi is installed
```

Pure ASCII, LF only (hpc-python-compat); python 3.11 on the cluster.
