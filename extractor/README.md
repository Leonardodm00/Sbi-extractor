# extractor/ -- the real-data (MEA) extractor, moved here from the DSN repo

Migration step 3, 2026-09-19. These files are `Main/hpc/MultiChannel/` of
`Leonardodm00/Deep-Summary-Network` at `7afe1b885b486b60215aeaf5143b637705c5d69a`
(tag `dsn-final-20260919`). `ORIGIN_MANIFEST.tsv` lists every file with its
sha256, its source path and whether it is `verbatim`, `edited` or `new`.
12 of 17 are verbatim; the three edits and the two new files (this README,
`run_extractor_smoke.pbs`) are described below in full, so nothing here has
to be diffed against the retired repo to be understood.

The extractor turns a folder of 3Brain `ptrain_<k>.mat` rasters into pooled
per-subregion IFR archives (`trace_subregion_XX.npz` + `traces.npz` +
`traces_meta.json`), recording every preprocessing parameter it used
(`dt`, `sigma_sm`, `w_size`, `gaussian_window`, `electrodes_per_subset`,
`mfr_threshold`, ...). Those archives are the real arm that `real_source.py`
and `dataset_profile.py` in this repo consume, which is why the extractor
belongs beside them. Its technical documents stayed with the DSN mirror:
`Simulation-Based-Inference/hpc/dsn/Documentation/MULTICHANNEL_TECHNICAL_DOCUMENT.md`
and `MERGE_REPORT_multichannel.md`; `README_HPC_DSN.md` here is the DSN's own
operating note, verbatim, and its paths are the OLD ones.

## What it needs from the DSN tree, and how it finds it

Two things, both in `Simulation-Based-Inference/hpc/dsn`:

| needed by | module | why it is not copied here |
|---|---|---|
| `channel_subset_extraction.py` (lazily, at IFR time) | `generate_burst_data.py` -- `compute_ifr_trace`, `CONTROL_PARAMS` | it is the IFR primitive both arms share; one function in one place is the parity contract's own requirement (EXTRACTOR_USAGE.md S4/S6). Decision 2026-09-19: it stays in the DSN tree, the sim arm's `sim_observable.py` will call the same one (step 4) |
| `list_extraction_jobs.py` | `config.py` (`ExperimentConfig`, whose `cohort` block holds the numbers) and `make_mea_specs.py` (`find_wells`, `expand`, `root_name_for`) | the whole point of that script is to compute the SAME paths the specs generator computes later; importing the generator is what makes the two unable to disagree |

The tree is found by `../dsn_tree.py`: an explicit argument, else
`SBI_HPC_DIR`, else `../artifacts/sbi_hpc` -- a per-machine symlink to the
SBI repo's `hpc/` (`env.sh` sets the same default; `artifacts/README.md`
says how to create it: `ln -s ~/SBI/hpc artifacts/sbi_hpc`). The DSN tree is
`$SBI_HPC_DIR/dsn`. `DSN_MAIN_DIR` is not read by anything in this
directory.

Note that `config.py` imports `backbone.py`, which imports torch, so
`list_extraction_jobs.py` needs the training environment even though it
trains nothing; the archives and the flags file were produced under
`meacnn_cpu`, and that is what the two job scripts still activate. Lifting
`CohortConfig` out of `config.py` so the extractor is torch-free is deferred
to the manifest work (Stage D), where the cohort block is being touched
anyway.

## The three edits, exactly

`channel_subset_extraction.py` -- after the imports, a block that puts this
directory and the repo root on `sys.path`, imports `dsn_tree` and calls
`dsn_tree.add_dsn_to_path(require=False)`, plus a 7-line `_gbd_import()`
helper; the three lazy `from generate_burst_data import ...` lines become
`... = _gbd_import().<name>` so that a tree that does not resolve fails at
the first IFR call with a message naming `SBI_HPC_DIR`, not with a bare
`ModuleNotFoundError`. `require=False` is deliberate: the geometry-only
stages (1-3) never touch the IFR and must import this module even when the
tree is absent, which the smoke run below checks in both directions.
Nothing in the extraction logic changed: `git diff` against the source
blob touches the import block, the helper, and those three import lines,
nothing else.

`list_extraction_jobs.py` -- the `_MAIN = two levels up` block becomes
`dsn_tree.add_dsn_to_path()` (require=True: this script cannot run without
the tree, so it says so at once); the docstring's usage line names the
config's new location.

`run_extractor_array_mea.pbs` -- the "resolve Main/" block becomes
"resolve the extractor directory": submit from here or from the repo root;
`env.sh` is sourced so `SBI_HPC_DIR` is set and printed; an ABORT if
`$SBI_HPC_DIR/dsn/generate_burst_data.py` is absent, before any well is
touched. `ENV_NAME` becomes overridable (`-v ENV_NAME=...`), default
unchanged. The header's SETUP names the new paths. The extraction call
itself, the manifest/flags handling and the conda block are unchanged.

`run_extractor_smoke.pbs` (new) -- the DSN runbook's "7 extractor suites"
loop plus `smoke_test_extraction_metadata.py`, as a committed batch job.

## Left behind in the retired repo, deliberately

| file | why |
|---|---|
| `generate_burst_data.py` | identical to `Main/generate_burst_data.py`, which is in the DSN tree; a second copy is the drift the parity contract forbids |
| `README.md` | one line ("1DCNN with multiple channels") |
| `run_extractor.pbs`, `run_extractor_array.pbs` | pre-cohort single-well and array jobs, 20 lines each, activating `brian_env`, which this pipeline stopped using on 2026-09-14; the manifest-driven `run_extractor_array_mea.pbs` covers a single well with a one-line manifest |

## Running it

Once per machine:

    cd ~/repos/Sbi-extractor && ln -s ~/SBI/hpc artifacts/sbi_hpc && python3 dsn_tree.py

`python3 dsn_tree.py` must print `resolves    : yes`.

List the jobs (regenerates `extraction_manifest.tsv`, gitignored, and
`extraction_flags.sh`, tracked -- commit the flags file together with any
change to `cohort.*` in the config, never hand-edit it):

    cd ~/repos/Sbi-extractor/extractor && source ../env.sh && python3 list_extraction_jobs.py --config "$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json"

Extract (the array is sized to the manifest; write to a NEW `extract_root`,
never over `extracted/` -- Stage D):

    qsub -J 0-$(($(wc -l < extraction_manifest.tsv) - 1)) run_extractor_array_mea.pbs

## Verification of this step

Sandbox, 2026-09-19: the 8 smoke suites pass 8/8 from the ORIGINAL location
(baseline) and 8/8 from here with `SBI_HPC_DIR` pointing at a checkout of
SBI; with `SBI_HPC_DIR=/nonexistent` the three geometry-only suites still
pass and the five that reach the IFR fail with the `SBI_HPC_DIR` message
(the guard shown failing on the case it is meant to catch).
`list_extraction_jobs.py` could not be run in the sandbox (no torch);
its cluster check is the discriminating one:

    cd ~/repos/Sbi-extractor/extractor && conda activate meacnn_cpu && source ../env.sh && python3 list_extraction_jobs.py --config "$SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json" --out-manifest /tmp/m.tsv --out-flags /tmp/f.sh && cmp /tmp/f.sh extraction_flags.sh && wc -l /tmp/m.tsv

must print `wrote 35 well(s)`, a silent `cmp` (byte-identical flags) and
`35 /tmp/m.tsv`. A `cmp` difference means the committed flags were not
produced by this config, which is a finding about the cohort, not about
the move.

Cluster [CLUSTER 2026-09-19]: `run_extractor_smoke.pbs` under `meacnn_cpu`
-> `[job] 8/8 extractor suites passed`; `list_extraction_jobs.py` regenerated
`extraction_flags.sh` byte-identical to the tracked file, 35 wells. Note that
the check must run under an ACTIVATED `meacnn_cpu`: calling its interpreter
by absolute path skips the env's `activate.d` hook that puts its own
libstdc++ first on `LD_LIBRARY_PATH`, and scipy then fails with
`GLIBCXX_3.4.26 not found` (`hpc/dsn/hpc/setup_env_davinci.sh:151-166`).
