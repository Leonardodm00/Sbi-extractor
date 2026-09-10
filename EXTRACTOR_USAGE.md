# Sbi-extractor: operating reference

| Date | Change |
|---|---|
| 2026-09-08 | Initial version. Covers the fixed run order, the four dataset kinds and the new `dataset_profile.py` typing step, the parity contract that decides whether two datasets may be pooled, the standard invocation chain, the four silent-corruption traps, and a troubleshooting index. Written while preparing the `campaign_cadex_hhgap_v{1,2,5}` export; every number carries its source. |
| 2026-09-08 | v2, after the first real profiler run against the hhgap roots. **Corrects sec. 5.1**: those campaigns are `conn_rule=flat`, so `conn_prob` is causally LIVE and the r2 bank's weibull-exclusion invocation is wrong for them. Adds sec. 5.4 (determining the swept axis set with the existing `campaign_axis_audit.py` -- no new tool needed) and sec. 12 (the measured profile of record, including two data-integrity findings and the decision to target the 1-electrode root first). |

**Confidence markers**, same scheme as `HPC_PATHS.md`.
`[KB]` = stated in the project knowledge base (`HPC_PATHS.md`, `SBI_PIPELINE.md`, `witness_usage.md`, the `sbi-export` README), cited to its section.
`[SANDBOX]` = verified by running it in this conversation's sandbox.
`[TO VERIFY]` = not checked against the cluster; run the given command before depending on it.
`[STATED]` = the user said so; not independently checked.
Do not promote a `[TO VERIFY]` to `[KB]` without running the command.

---

## 1. Run order, which cannot be shortcut

```
artifacts/frozen_dsn/*.pt                     the encoder is chosen HERE and nowhere else
        |
        v
dataset_profile.py                            <-- NEW: type the inputs, diff the arms
        |
        v
preflight_label_axes.py -> artifacts/label_axes.json
        |
        v
export (simulated arm)   export (real arm)
example_export.py        real_source.py
        \                   /
         v                 v
     *.parquet + *.json sidecar
        |
        v
gate_run.py -> <stem>_results.json, <stem>_arrays.npz
        |
        v
witness_run.py
```

`[KB -- HPC_PATHS.md sec. 3a, SBI_PIPELINE.md sec. 5]`. Nothing downstream of the export touches the checkpoint: `gate_run.py` reads parquet, `witness_run.py` reads `gate_run.py`'s `_arrays.npz`. Changing the encoder therefore means re-exporting **both** arms, never patching a later stage.

`dataset_profile.py` is new in this document and sits before `preflight_label_axes.py` because both the label freeze and the export assume things about the inputs that were previously only assumed.

## 2. Modules

`[KB -- HPC_PATHS.md sec. 1]`. Repo `Leonardodm00/Sbi-extractor`, cluster path `/davinci-1/home/ldellamea/repos/Sbi-extractor`, branch `feat/real-arm-parity`.

| file | role |
|---|---|
| `dataset_profile.py` | **new, this document.** Types a dataset, extracts its Profile, diffs two Profiles against the parity contract |
| `preflight_label_axes.py` | freezes which axes enter `theta` across all included campaigns into `artifacts/label_axes.json` |
| `sbi_labels.py` | builds the label spec from `manifest.json` / `job_args.json` / the simulator registry |
| `sim_observable.py` | `pooled_spike_counts`, `build_pooled_ifr`, `window_trace`, `reference_compute_ifr_trace` |
| `dsn_frozen.py` | `load_frozen_dsn(checkpoint)` -> `FrozenDSN` (`E`, `w_size`, `gaussian_window`, `window_s`, `l2_normalize`, `ckpt_sha256`) |
| `export_embeddings.py` | `TraceRecord`, `export_embeddings(...)` -> parquet shard + JSON sidecar; assertions A1-A9 |
| `example_export.py` | `--mode campaign` walks `MEA_ROOT` / `SIM_ROOT` and embeds |
| `real_source.py` | the real-arm trace source; groups on the specs `culture` field, never the npz `culture_id` |
| `check_preprocessing_parity.py` | verifies `$\Delta t$`, `$\sigma_{\rm sm}$` and `$n_e$` agree between the arms |
| `submit_sbi_export.sh` | PBS job body; sources `env.sh`, resolves python by absolute path, requires `artifacts/label_axes.json` |
| `launch_sweep_exports.sh` | enumerates every `(campaign, sweep_task)` pair and submits one job each |
| `smoke_test_*.py` | one per module; run before trusting any of them |

`artifacts/` is gitignored machine state `[KB -- HPC_PATHS.md sec. 2]`: `dsn_main` (symlink), `frozen_dsn/*.pt` (a **copy**, never a symlink), `specs_real.json`, `label_axes.json`.

## 3. `dataset_profile.py` -- what kind of dataset is this

`[SANDBOX -- 16/16 smoke checks pass; pure ASCII, LF-only; py_compile and argparse name-resolution gates pass]`

### 3.1 The four kinds, detected by content and never by directory name

| kind | recognised by | what it carries |
|---|---|---|
| `sim_campaign` | `topo_*/iter_*.npz` under a dir with `job_args.json` | label axes, prior box, registry width, seeds, `simtime` |
| `mea_output` | `topo_*/mea_iter_*.npz` | `n_e`, `fs_acq`, detection load, probe config, inferred `simtime` |
| `real_extracted` | any `.npz` holding an IFR trace key | `fs_ifr`, `T_rec`, `n_e` if recorded |
| `export_shard` | `<stem>.parquet` **and** `<stem>.json` with `param_names` | `p`, `param_names`, `coord`, `bounds_theta`, `E`, checkpoint sha256, observable block |

A `mea_output` profiled **alone** cannot know its label axes or its true `T`; pass `--sim-root` so the paired `job_args.json` is read. Without it the profile reports `T_source` as inferred and leaves `labels.p` null rather than guessing.

### 3.2 API

```python
import dataset_profile as D

p_sim  = D.profile_dataset("/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro",
                           registry_src=".../Giulia_Astro/HPC_single_run.py")
p_mea  = D.profile_dataset(".../MEA_analysis/mea_out",
                           sim_root=".../Giulia_Astro", window_s=180.0)
p_bank = D.profile_dataset("/davinci-1/home/ldellamea/ANN/SBI_export_r2")

verdict = D.compare_profiles(p_mea, p_bank)     # 'POOLABLE' | 'BLOCKED'
warns   = D.check_internal_consistency(p_mea)   # problems inside ONE dataset
D.PARITY_CONTRACT                               # the fields and their class
```

`profile_dataset` never raises on a schema surprise; anything unexpected lands in `profile["warnings"]`. Every field that is read per unit but must be constant across the dataset returns either the single value or `{"__multiple__": {value: count}}` -- the return type itself says whether it was constant, so no caller has to infer that from the value's shape. `D.is_nonconstant(field)` is the predicate.

### 3.3 CLI

```bash
conda activate sbi_env
python3 smoke_test_dataset_profile.py            # expect ALL 16 CHECKS PASSED

python3 dataset_profile.py \
    /davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out \
    /davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_1electrode \
    /davinci-1/home/ldellamea/ANN/SBI_export_r2 \
    --sim-root /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro \
    --registry-src /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/HPC_single_run.py \
    --iters-per-unit 2 --window-s 180 --json profile.json
```

With two or more paths it prints a pairwise parity verdict for every pair. `--max-units N` gives a fast first pass over a large tree. pyarrow is optional: without it an `export_shard` is still fully profiled from its sidecar, only the row count and the `||z||` check are skipped.

The **declared CLI flags of the extractor's own entry points** are a separate question and are not this tool's job; `preflight_giulia_export.py --extractor <repo>` (same delivery) reads them off the files by AST, which is how to recover them without importing anything.

### 3.4 What the profiler warns about, and why each one exists

| warning | why it matters |
|---|---|
| a field is not constant across units | the dataset is internally heterogeneous; shards from it carry different contracts |
| registry `PARAM_NAMES` width != recorded `len(params)` | the deployed registry did not produce these files, so `sbi_labels` would index the wrong axes |
| `simtime` in the npz disagrees with `job_args`, per unit | the npz value is inferred from the last spike; those traces are silently truncated and dropped (sec. 6.2) |
| the npz `simtime` is not constant across units | some simulations went quiet early |
| MEA unit with no paired sim unit, or the reverse | `launch_sweep_exports.sh` requires both sides `[KB -- HPC_PATHS.md sec. 8]` |
| a `seed_master` appears in more than one unit | those units are byte-identical replays |
| `T < window_s` | every trace yields zero windows |
| `max abs(||z|| - 1) > 1e-5` | assertion A7 would fail |
| `n_e` not recorded in a real archive | it comes from the extraction config (`electrodes_per_subset`) and must be supplied by hand before any parity claim |

## 4. The parity contract

Two datasets may be pooled into one bank, or compared through one frozen encoder, only if every **hard** field agrees. A field unknown on either side is reported as `not_comparable` and never as agreement -- that distinction is the point of the tool.

| field | class | why |
|---|---|---|
| `observable.n_e` | hard | the observable is a mean over exactly these electrodes |
| `observable.fs_ifr` | hard | set by the frozen checkpoint, not by the data |
| `observable.sigma_sm` | hard | smoothing width of the IFR |
| `labels.p` | hard | number of `th_*` columns |
| `labels.param_names` | hard | identity and order of the axes |
| `labels.coord` | hard | `ln` vs `linear` per axis |
| `labels.prior_box` | hard | the BoxUniform bounds |
| `labels.registry_width` | hard | the `PARAM_NAMES` width `params` was recorded against |
| `embedding.E` | hard | embedding dimension |
| `embedding.checkpoint_sha256` | hard | encoder identity (assertion A8) |
| `observable.T` | soft | only fatal when `T < window_s` |
| `observable.fs_acq` | soft | raw acquisition rate before IFR binning |
| `labels.mode`, `labels.sweep_group`, `labels.conn_rule` | soft | informational; they explain a hard break rather than being one |
| `observable.rate_hz_per_electrode` | soft | drives the MFR floor |

### 4.1 `n_e` is the one that is easy to get wrong

`[KB -- SBI_PIPELINE.md sec. 3; sbi-export README sec. 2.1; HPC_PATHS.md sec. 3]` The observable is a pooled IFR over a **9-electrode subregion**, on both arms: the real cohort is 315 npz files = 35 cultures x 9 subregions, and the simulated arm of record used `n_side = 3`, i.e. 9 electrodes at 60 um pitch, sampled at the same 10110.09 Hz raw rate as the 3Brain recordings. For a subregion of $n_e$ electrodes and for each fixed bin index $k \in \{0, \dots, K-1\}$, with $S_e$ the detected spike-time set of electrode $e$, $\Delta t$ the bin width and $\sigma_{\rm sm}$ the smoothing width:

$$C[k] \;=\; \sum_{e \in \text{subregion}} \bigl| \, S_e \cap [\,k\Delta t,\ (k+1)\Delta t\,) \, \bigr| \tag{1}$$

$$\tilde R[k] \;=\; \max\Bigl( \bigl(\text{gaussian\_filter1d}(C, \sigma_{\rm sm}/\Delta t)\bigr)[k],\ 0 \Bigr) \tag{2}$$

$$R_{\rm norm}[k] \;=\; \tilde R[k] \,/\, n_e \qquad \text{for each fixed } k \tag{3}$$

Equation (3) is a **per-electrode mean**, not a sum. The original handoff's sidecar recorded `"pooling": "sum_over_electrodes"`, which is wrong by exactly the factor $n_e$ `[KB -- sbi-export README sec. 2.1]`.

Because (3) divides by $n_e$, changing $n_e$ leaves the first moment in the same units (Hz/electrode) but not the second: for approximately independent channels the sampling noise on that mean scales as $n_e^{-1/2}$, so at fixed underlying rate a 1-electrode trace is about $3\times$ noisier than a 9-electrode one and a 4-electrode trace about $1.5\times$ `[my own reasoning, not from a source]`. And it is not the same population being estimated more noisily: detection coverage of the culture is 52.4% at 3x3/pitch-60, 38.7% at 2x2/pitch-200 with zero cross-channel overlap, and 9.8% at a single centred electrode, where 11.5% of the culture is outside reach entirely `[KB -- MEA analysis reference sec. 4, sec. 7.2]`.

For a **frozen** encoder that is an input-distribution shift, i.e. the simulation-gap regime. Schmitt et al. show that approximate posteriors under misspecification can look legitimate while being wrong, so posterior-side and posterior-predictive diagnostics cannot adjudicate it and the check has to live in summary space `[KB -- "Detecting Model Misspecification in Amortized Bayesian Inference", full text in project knowledge]`. The harder case is also on record: model error can produce summary statistics that lie **in**-distribution relative to the simulations, so a passing gate is not sufficient either `[KB -- "Misspecification-robust amortised SBI using variational methods", full text in project knowledge, related-work section]`.

Practical consequence: an export whose `n_e` differs from the real arm's is still a valid **sim-only** bank (prior-predictive study, activity filtering, NPE trained on simulations). It is not comparable to `sbi_real_cohort.parquet` and must not be pooled with `SBI_export_r2`. Matching is possible in either direction -- re-run the virtual MEA at `n_side = 3`, or re-extract the real arm at the new `electrodes_per_subset` -- but it is a re-export of both arms, not a flag on a later stage.

## 5. Per-campaign-set preconditions

### 5.1 `p` is not a constant

`[KB -- HPC_PATHS.md sec. 8]`: *"do not assume a fixed number -- it is now determined by `artifacts/label_axes.json`, which itself depends on which campaigns are included and their `conn_rule`. Regenerate/re-check it if the campaign set changes."*

The bank of record: 36 natural axes with 3 frozen (`DeltaT`, `VT`, `gL`), `conn_prob` excluded as causally inert under `conn_rule=weibull`, the Weibull kernel axes `p0_conn, d0_conn, beta_conn` entering from the topology loop, giving $p = 26$ (17 stored as $\ln$, 9 linear) `[KB -- SBI_PIPELINE.md sec. 3]`. The `rho1300` campaigns are a single signature: 215 `job_args.json`, all declaring `simtime = 200.0`, 23 `active_indices` under group `neuron_synapse`, `conn_prob_lo/hi = 0.1/0.6`, all kernel bounds JSON-null `[KB -- HPC_PATHS.md sec. 4a]`.

**The freeze has two forms and the wrong one is silently plausible.** `conn_rule` decides which:

```bash
# weibull campaigns (rho1300*): conn_prob is drawn but never read, so EXCLUDE it
# [KB -- HPC_PATHS.md sec. 2, recorded invocation]
python3 preflight_label_axes.py \
    --sim_main <SIM_MAIN_DIR> \
    --require-conn-rule weibull \
    --exclude 'conn_prob=...' \
    --out artifacts/label_axes.json

# flat campaigns (hhgap v1/v2/v5): conn_prob IS read, so it must be INCLUDED
# and the kernel axes p0_conn/d0_conn/beta_conn do not exist at all
python3 preflight_label_axes.py \
    --sim_main <SIM_MAIN_DIR> \
    --require-conn-rule flat \
    --out artifacts/label_axes_hhgap.json
```

Under `conn_rule=flat`, `conn_prob` is read by `build_topology`'s flat branch and by `Neuronal_Network`'s `S.connect(p=conn_prob)` `[KB -- HPC_PATHS.md sec. 5]`, so excluding it drops a live axis. Under `weibull` the same parameter is drawn and never read, so including it teaches the density estimator that $p(\theta_j \mid x) = p(\theta_j)$ on a coordinate carrying zero information. The two errors are mirror images, both silent, and neither is visible in a variance scan -- an inert axis varies perfectly well.

Two rules around it:

- **Write a new campaign set's freeze to a new path.** Overwriting `artifacts/label_axes.json` silently changes the contract any later re-export of the existing bank would be built under.
- **Read `labels.conn_rule` off the profiler before choosing the form**, not from the last campaign set you worked on. That is exactly how the wrong invocation was nearly used here (sec. 12).

Independent confirmation without reading `job_args.json`: under `flat` the `mea_iter_*.npz` key list contains `conn_prob` and no `p0_conn`/`d0_conn`/`beta_conn`; under `weibull` it is the reverse. `dataset_profile.py` prints those key lists per sampled file.

### 5.2 Registry width

The `params` vector recorded in `iter_*.npz` is only interpretable against the `PARAM_NAMES` that produced it. The cluster registry was 37-D after the O_N port `[KB -- HPC_PATHS repo-hierarchy doc sec. 3.4]`, while the MEA self-test asserts a 36-length `params` `[KB -- MEA analysis reference sec. 5.1]`. These are different vintages. `--registry-src` makes the profiler compare the two and warn; a mismatch means wrong axis indexing, not a cosmetic difference.

### 5.3 Duplicates

`build_mea_manifest.py` has no seed-collision awareness `[KB -- MEA analysis reference sec. 6]`. Duplicated rows are byte-identical replays and the loader deduplicates on distinct `theta`, keep-first `[KB -- SBI_PIPELINE.md sec. 5]`, so this costs shards and compute rather than correctness. `dataset_profile.py` reports `provenance.duplicate_seeds` so the cost is known before submission rather than after.

### 5.4 Determining the swept / consumed / inert axis set

**Do not write a new tool for this: `campaign_axis_audit.py` already does it** and is `[KB -- HPC_PATHS repo-hierarchy doc sec. 3.1]` confirmed present at `/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/`, alongside its own `smoke_test_campaign_axis_audit.py`. It `ast`-parses the registry and the sweep driver rather than importing them, so Brian2 is not needed, and it discovers campaign directories itself (`discover_campaigns`).

It answers a different question from `preflight_label_axes.py` and runs before it: the audit says which axes the *campaign actually swept and consumed*, the freeze says which of those *enter $\theta$*.

```bash
cd /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro
python3 smoke_test_campaign_axis_audit.py          # run first
python3 campaign_axis_audit.py \
    --registry-src ./HPC_single_run.py \
    --sweep-src    ./HPC_main_sweep.py
```

`[TO VERIFY]` the exact flag set beyond `--registry-src` / `--sweep-src`, and whether a `--json` mode exists (its sibling `count_simulations.py` has one). Run `--help` first.

**Why it is needed even though `dataset_profile.py` reports these fields.** The profiler reads `swept_axes` / `consumed_axes` / `inert_axes` out of `job_args.json`'s `_axis_declaration` and `manifest.json`'s `axis_declaration`. Those blocks are written by `build_axis_declaration()`, added in the parameter-recording fix at `manifest_version` 4 `[KB -- HPC_PATHS.md sec. 5]`. Campaigns launched before that fix have no such block, and the profiler correctly reports `null` rather than inventing one -- which is what happened on the hhgap set (sec. 12). The audit reconstructs the same information from the source instead.

**Already on record for these campaigns** `[KB -- HPC_PATHS repo-hierarchy doc sec. 6, a real audit run against all four hhgap campaigns]`: every one ran `mode=Full`, `sweep_group=synapse_astro`, **22 swept axes**, of which **9 are astrocyte axes** -- `C_Theta, F, G_T, I_bias, O_3K, O_N, O_beta, Omega_5P, U_A` -- and all 9 confirmed **swept+consumed, zero inert**, against 11 `ASTRO_PARAMS` in the 37-wide registry. The documented inert-axis failure mode is specific to `mode=Neuronal`, which none of these use.

That record names the 9 astrocyte axes but **not the other 13 of the 22**. Re-run the audit to get the full ordered list before the freeze; `preflight_label_axes.py` needs all 22, not the count.

## 6. The four traps that silently corrupt an export

`[KB -- sbi-export README sec. 2]`. Each is implemented correctly in the package and each contradicts something in the original data-export handoff, which is why they are worth restating rather than assuming.

### 6.1 Pooling is a mean, not a sum
Section 4.1, equation (3). Wrong by the factor $n_e$ if taken from the handoff's sidecar.

### 6.2 `simtime` in the MEA output is inferred from the last spike
`process_campaign.py` computes `simtime = float(np.ceil(spk_t.max()))`, so a quiet simulation records a duration far below the requested one, and any trace shorter than the window length $W$ is dropped with a silent `continue`. **Always pass `T` from the `--simtime` launch flag in `job_args.json`**; `build_pooled_ifr` makes `T` a required argument for this reason. The profiler compares the two per unit -- a dataset-level scalar comparison misses it, because the mismatch is per simulation.

### 6.3 The topology block is not in `mea_iter_*.npz`
`process_campaign.py` writes `conn_prob` but not `p0_conn`, `d0_conn`, `beta_conn`; those exist only in the original `iter_*.npz`. The export joins on `(topo_idx, iter_idx)` and raises if that key is not unique rather than inventing a surrogate. This is why `SIM_ROOT` is not optional.

### 6.4 The topology axes are linear-uniform, never log
`sample_kernel_vector` and the `conn_prob` draw both call `rng.uniform` on natural bounds. The log-axis rule must not be applied to them: `p0_conn` has bounds `[0.1, 1.0]`, i.e. exactly 1.000000 decades, so the rule would classify it as a log axis and the export would store $\ln(p_0)$ against a linear prior box.

## 7. Standard invocation chain

Steps 1-3 are cheap and read-only; do not skip them to save minutes on a job that takes hours.

```bash
# 0. environment.  sbi_env and sbi_export are two real, separate envs.
conda activate sbi_env
python3 -c "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"

# 1. type the inputs and diff them against the existing bank
python3 dataset_profile.py <MEA_ROOT> <SIM_ROOT> <EXISTING_BANK> \
    --sim-root <SIM_ROOT> --registry-src <SIM_ROOT>/HPC_single_run.py \
    --window-s 180 --json profile.json

# 1b. determine the swept/consumed/inert axis set (sec. 5.4), from the
#     simulator tree, not from this repo
cd <SIM_MAIN_DIR> && python3 campaign_axis_audit.py \
    --registry-src ./HPC_single_run.py --sweep-src ./HPC_main_sweep.py

# 2. freeze the label axes for THIS campaign set, to a NEW path.
#    Pick the branch from labels.conn_rule in step 1 -- see sec. 5.1.
#    flat    -> --require-conn-rule flat      (conn_prob INCLUDED)
#    weibull -> --require-conn-rule weibull --exclude 'conn_prob=...'
python3 preflight_label_axes.py --sim_main <SIM_MAIN_DIR> \
    --require-conn-rule <flat|weibull> \
    --out artifacts/label_axes_<tag>.json

# 3. preprocessing parity against the real arm
python3 check_preprocessing_parity.py     # flags: [TO VERIFY], run --help first

# 4. dry run on a handful of simulations, then read the banner
python3 example_export.py --mode campaign \
    --checkpoint artifacts/frozen_dsn/<ckpt>.pt \
    --campaign <SIM_ROOT>/<campaign>/<task> \
    --mea_out  <MEA_ROOT>/<campaign>/<task> \
    --campaign_id <campaign> --out /tmp/dryrun_0000 --max_records 20

# 5. the array
./launch_sweep_exports.sh                 # flags: [TO VERIFY]
```

**Banner checks on the dry run** `[KB -- sbi-export README sec. 5.3]`: `E`, `W`, `T_win`, `fs_ifr` match the training config; `traces too short : 0` (anything above zero means 6.2 fired); `assertions passed` includes A2, A3, A4, A5, A7, A9; the checkpoint SHA-256 is the one you expect.

**`[TO VERIFY]`** The exact flag names in steps 3-5. `HPC_PATHS.md` sec. 3a marks the export invocation as unverified: `launch_sweep_exports.sh` and `submit_sbi_export.sh` take the checkpoint via `env.sh` / `ARTIFACTS_DIR` rather than a bare flag, and the step-4 flags above are transcribed from the `sbi-export` README in project knowledge, which describes a package whose recorded `E` and `p` do **not** match the deployed one (sec. 10). Recover the real flags before typing any of this:

```bash
cd /davinci-1/home/ldellamea/repos/Sbi-extractor
python3 example_export.py --help
python3 preflight_label_axes.py --help
grep -n '^\s*[A-Z_]*=' env.sh
```

### 7.1 Assertions

`[KB -- sbi-export README sec. 8]` A1 transform round trip; A2 `th_*` column count and contract agreement; A3 non-degenerate prior box; A4 no constant `th_*` column; A5 every `theta_A` inside `B` (reported, never clipped); A6 per-row coordinate spot-check against `params`; A7 `abs(norm(z) - 1) < 1e-5`; A9 no NaN/Inf. Failures raise; nothing is silently clipped, repaired or dropped.

**A8 (checkpoint identity across shards) and A10 (cross-campaign compatibility) are deliberately not checked at export time** -- neither is a property of a single shard. Before training, confirm exactly one distinct digest:

```bash
for f in <EXPORT_DIR>/*.json; do
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); \
    print(d['embedding']['dsn_checkpoint_sha256'][:16], sys.argv[1])" "$f"
done | sort | uniq -c -w16
```

More than one distinct digest means the export is void. `dataset_profile.py` covers the same ground for whole directories: profile two export roots and read the verdict.

## 8. Environment and job submission

`[KB -- HPC_PATHS.md sec. 7]`. `sbi_env` and `sbi_export` are two real, separate conda environments; `sbi_env` is the one used for the NPE/tuning stage. `submit_sbi_export.sh` activates via `eval "$(conda shell.bash hook)"` -- a bare `conda activate` fails silently under PBS's non-interactive shell -- and then resolves the interpreter by absolute path rather than trusting `PATH`. Do not replace that block with `module load python`.

## 9. Troubleshooting index

| symptom | cause | check |
|---|---|---|
| `traces too short` above zero in the export banner | trap 6.2: npz `simtime` inferred from the last spike | `dataset_profile.py --sim-root ...`, read the simtime warning |
| export raises on a non-unique `(topo_idx, iter_idx)` | duplicated or mis-paired units | profiler `detail.paired` / `mea_without_sim` |
| shards refuse to load together | `param_names` / `coord` / `bounds_theta` / checkpoint disagree | `compare_profiles` on the two export roots |
| assertion A4 fires (a constant `th_*` column) | an axis in `label_axes.json` was never actually swept in this campaign set | profiler `labels.swept_axes` vs the frozen axes |
| `launch_sweep_exports.sh` submits nothing for a campaign | one side of the `(campaign, task)` pair is missing | profiler `sim_without_mea` |
| a job dies instantly with an empty log | conda activation, not the code | `check_job_env.py`; sec. 8 |
| gate numbers not comparable to an earlier run | different encoder, or `--min_rate` default changed to `0.1` | sidecar `dsn_checkpoint_sha256`; `[KB -- HPC_PATHS.md sec. 5b]` |

## 10. Known contradictions in the sources

The `sbi-export` README in project knowledge records `E = 16` (from `config_mea_joint_full.json`), a 27-column `theta_A` built from `sweep_groups["neuron_synapse"]`, and `n_e = 9`. `SBI_PIPELINE.md` sec. 4 and `HPC_PATHS.md` sec. 3 record the encoder actually in use as **r2** (`refit_mea_joint_full_r2_l0_t82`), `E = 10`, with $p = 26$ and the export of record at `ANN/SBI_export_r2/`, 54 shards, 86,251 rows.

These describe different vintages of the same pipeline. **Do not resolve this from memory in either direction** -- profile the sidecar and read `embedding.embedding_dim`, `dsn_checkpoint_sha256` and `len(param_names)` off the shard that exists. That is exactly what `_profile_export` does, and it is the reason the parity contract holds no reference constants.

## 11. Open items

- `[TO VERIFY]` The declared flags of `example_export.py`, `launch_sweep_exports.sh`, `check_preprocessing_parity.py` on the deployed repo. Section 7 gives the commands.
- `[TO VERIFY]` The exact key schema of the real extracted archives. `dataset_profile._profile_real` scans a candidate key list (`ifr_trace`, `fs_ifr`, `T_rec`, `culture_id`, `electrodes_per_subset`, ...) and reports the full key list either way, so a surprise surfaces rather than being silently mapped. Confirmed from project knowledge only that these files exist as 315 npz, `fs_ifr = 100.0`, `T_rec = 1200.0`, `K = 120000` `[KB -- HPC_PATHS.md sec. 3]`.
- `[TO VERIFY]` Whether `real_source.py` records `n_e` in its sidecar's observable block. If it does not, the hard field `observable.n_e` will come back `not_comparable` for the real arm and must be supplied from the extraction config by hand.
- **Not implemented:** a dedup-aware manifest filter, so an array run against a full campaign glob processes duplicate tasks `[KB -- MEA analysis reference sec. 6]`.
- **Not implemented:** the real-recording export path and the OOD probe set `[KB -- sbi-export README sec. 10]`.
- **Not inspected:** `hpc/Electrode Traces Extractor/` in the simulator repo is a separate, MEA-adjacent tool with no known cluster deployment; it is not this pipeline and has not been compared against it `[KB -- MEA analysis reference sec. 8]`.
- **Open, unresolved:** whether the collapsed embedding ($r_{\rm eff} = 1.017$ simulated, $1.000$ real, of $E = 10$) should be fixed before any restriction is derived -- O2 in `SBI_PIPELINE.md` sec. 13. Every parity verdict in this document is conditional on the frozen encoder, and none of it addresses that.
- `[CLUSTER RUN, sec. 12.1]` The v1 `sweep_cfd_task*` units under `mea_out` have no `mea_manifest.json` and about half the topologies of their 1-electrode counterparts. Cause not established -- job still running, killed, or a different `--limit_topos`. Re-profile that root before exporting from it.
- `[CLUSTER RUN, sec. 12]` Five sim units under a directory named `q/` (`q/sweep_intel_task0005`-`0009`) have no MEA output and appear in no project document. Unidentified; not v4, not hhgap-named. Find out what they are before assuming they are safe to ignore.
- `[CLUSTER RUN, sec. 12]` Two stray unpaired units sit under `mea_out`: `v1/sweep_intel_task0000` and `v1/sweep_intel_task0000_1electrode`, artefacts of the earlier hand-run single-task test. The second is 1-electrode output under the 4-electrode root and is what makes that root's `n_e` non-constant. Move or delete them rather than relying on the pairing step to skip them.
- `[TO VERIFY, sec. 5.4]` The full ordered list of all 22 swept axes. Only the 9 astrocyte ones are named in project knowledge; re-run `campaign_axis_audit.py` before the freeze.
- `[TO VERIFY, sec. 12]` Whether the `sim_without_mea` entries beyond `q/` are the deduplicated-away redundant tasks. Inferred from a matching count of 51, not checked; `ls dedup_root/`.


## 12. Profile of record -- the hhgap campaigns, 2026-09-08

`[CLUSTER RUN]` `dataset_profile.py` against both MEA roots with `--sim-root .../Giulia_Astro --registry-src .../Giulia_Astro/HPC_single_run.py --iters-per-unit 2 --window-s 180`. Single-valued and clean unless noted.

| field | value |
|---|---|
| mode / sweep_group | `Full` / `synapse_astro` |
| conn_rule | **`flat`** -- see sec. 5.1 |
| n_active_indices | 22 |
| prior box | `conn_prob_lo/hi = 0.05 / 0.4` (the r2 bank's is `0.1 / 0.6`; different box, not poolable) |
| registry width vs recorded `len(params)` | 37 vs 37, **no mismatch** |
| `simtime` (job_args) | 180.0 s, constant |
| `fs_acq` | 10110.09 Hz |
| seeds | 51 units, 51 distinct, **zero duplicates** on both roots |
| `swept/consumed/inert_axes` | **not recorded** -- no `_axis_declaration` in these `job_args.json`; use sec. 5.4 |

The 51 units shared by both roots (25 v1 + 6 v2 + 20 v5) match the credited-task count `[KB -- MEA analysis reference sec. 6]` for `dedup_root/campaign_cadex_hhgap_{v1,v2,v5}`. That is a matching count, not a direct check: `ls dedup_root/` before relying on the other `sim_without_mea` entries being deduplicated-away rather than unprocessed.

### 12.1 Target root: `mea_out_1electrode` first

`mea_out_1electrode` profiles clean at $n_e = 1$: 51 units, all paired, every `mea_manifest.json` present with `total_done == total_iters`, no unpaired MEA units.

`mea_out` does not, and is deferred. Its `n_e` is **not constant** (`{4: 104, 1: 2}`) because two stray hand-run units from the earlier single-task test sit under it -- `v1/sweep_intel_task0000` and `v1/sweep_intel_task0000_1electrode`, both unpaired, the second being 1-electrode output filed under the 4-electrode root. More seriously, all seven v1 `sweep_cfd_task*` units are **missing `mea_manifest.json`** and carry roughly half the topology count of the same task IDs in the 1-electrode root:

| task | `mea_out` (4e) | `mea_out_1electrode` (1e) |
|---|---|---|
| `cfd_task0000` | 37 topo / 35,389 iter, **no manifest** | 70 topo / 66,240 iter, done |
| `cfd_task0003` | 36 topo / 34,344 iter, **no manifest** | 70 topo / 67,136 iter, done |
| `cfd_task0006` | 37 topo / 34,822 iter, **no manifest** | 70 topo / 66,659 iter, done |

The `intel_task` families in both v1 and v2 are complete under `mea_out`. Only `cfd_task*` looks mid-run. Re-profile that root after those finish; exporting against a half-written directory produces a shard that passes every assertion and is simply short.

### 12.2 Expected yield, stated before the run rather than after

32 of 51 units report an npz `simtime` far below 180 s (36 samples at exactly 1.0 s), i.e. many simulations go quiet early. `T` comes from `job_args` throughout, so nothing is silently truncated (sec. 6.2), but the MFR floor will drop a substantial fraction downstream. The r2 bank's comparable figure was 34.3% kept at 0.1 Hz/electrode `[KB -- HPC_PATHS.md sec. 4d]`; that number will **not** transfer, since the prior box here is tighter and $n_e$ differs.

Measured on the 1-electrode root: `rate_hz_per_electrode = 3.2501` averaged over the sampled iterations.
