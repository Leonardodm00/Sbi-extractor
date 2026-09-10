# Sbi-extractor: operating reference

| Date | Change |
|---|---|
| 2026-09-08 | Initial version. Covers the fixed run order, the four dataset kinds and the new `dataset_profile.py` typing step, the parity contract that decides whether two datasets may be pooled, the standard invocation chain, the four silent-corruption traps, and a troubleshooting index. Written while preparing the `campaign_cadex_hhgap_v{1,2,5}` export; every number carries its source. |
| 2026-09-08 | v2, after the first real profiler run against the hhgap roots. **Corrects sec. 5.1**: those campaigns are `conn_rule=flat`, so `conn_prob` is causally LIVE and the r2 bank's weibull-exclusion invocation is wrong for them. Adds sec. 5.4 (determining the swept axis set with the existing `campaign_axis_audit.py` -- no new tool needed) and sec. 12 (the measured profile of record, including two data-integrity findings and the decision to target the 1-electrode root first). |
| 2026-09-10 | v4, written after the freeze, the dry runs and the launcher dry run all completed on the cluster. Most `[TO VERIFY]` flags in sec. 7 are now closed by real `--help` output. Adds two new traps (6.5 all-NaN axes counted as swept; 6.6 the live registry vs the campaign's own bounds), two code fixes that came out of them (`registry_from_manifest`, `record_resolved_n_e`), sec. 5.5 (the freeze of record), sec. 12.3 (the full 22-axis audit) and sec. 13 (the export run of record). |
| 2026-09-08 | v3, caught while preparing the first real freeze/export commands. **Corrects sec. 7-8**: `dataset_profile.py`/`campaign_axis_audit.py` need only numpy (`sbi_env` is fine), but `preflight_label_axes.py`/`example_export.py`/`launch_sweep_exports.sh` must run under `sbi_export` per `[KB -- HPC_PATHS.md sec. 7]` -- v1/v2 of this document said `sbi_env` for the whole chain, which would have run the export scripts in the wrong environment. |

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
| `smoke_test_*.py` | one per module; run before trusting any of them. Added this session: `smoke_test_dataset_profile.py` (19), `smoke_test_registry_from_manifest.py` (9), `smoke_test_resolved_n_e.py` (8) |
| `campaign_axis_audit.py` | **not in this repo** -- it lives in the simulator tree (sec. 5.4) and answers a different question from `preflight_label_axes.py` |

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

**What the frozen file carries, and what it does not.** `[CLUSTER RUN]` `label_axes.json` holds only the **topology** block -- keys `topology_axes`, `excluded_axes`, `axis_stats`, `conn_rule_observed`, `topologies_sampled`, `n_independent_topology_draws`, `campaign_glob`, `sim_main`, `created_utc`. It does **not** hold `param_names` or the full `p`; `example_export.py` combines this file's topology axes with the registry's active axes at export time, so `p` only becomes visible in the export banner.

`axis_stats` records the **observed** min/max and distinct count per axis, deliberately -- it is provenance, not a prior. **The prior box therefore comes from somewhere else**: `--conn_prob_lo/--conn_prob_hi` on `example_export.py`, which fall back to `job_args.json` when the flags are absent, and to `0.1 / 0.6` when `job_args.json` itself is missing. The launcher never passes those flags, so the array path is the `job_args.json` fallback -- correct for the hhgap campaigns, which declare `0.05 / 0.4`, but only because that file is present.

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

### 5.5 The freeze of record -- `artifacts/label_axes_hhgap.json`

`[CLUSTER RUN 2026-09-10]` sha256 prefix `1ef04131175605a0`.

```bash
python3 preflight_label_axes.py \
    --sim_main /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro \
    --campaigns 'campaign_cadex_hhgap_v[125]' \
    --require-conn-rule flat \
    --exclude 'p0_conn=Weibull kernel axis; never drawn under conn_rule=flat, recorded as NaN' \
    --exclude 'd0_conn=Weibull kernel axis; never drawn under conn_rule=flat, recorded as NaN' \
    --exclude 'beta_conn=Weibull kernel axis; never drawn under conn_rule=flat, recorded as NaN' \
    --out artifacts/label_axes_hhgap.json
```

| field | value |
|---|---|
| `topology_axes` | `['conn_prob']`, so `p_topology = 1` |
| `excluded_axes` | the three Weibull kernel axes, with reasons (see trap 6.5) |
| `conn_rule_observed` | `['flat']` |
| `topologies_sampled` | 4467 |
| `conn_prob` observed | 2612 distinct in `[0.0503601, 0.399682]` |

The character-class glob `campaign_cadex_hhgap_v[125]` excludes v4 without relying on brace expansion the script's own globbing may not support.

Note 2612 distinct `conn_prob` values against 4467 topologies. That is the clustered-sampling structure: `N_PARAMS_PER_WORKER = 5` rows share one topology-level draw `[KB -- HPC_Campaign_Reference.md sec. 6]`, so along `conn_prob` the effective sample size is the number of topologies, not the number of rows. `n_independent_topology_draws` counts topology directories and does **not** deduplicate that clustering.

**Omitting `--out` reports without writing.** Always do that pass first.

## 6. The traps that silently corrupt an export

`[KB -- sbi-export README sec. 2]`. Each is implemented correctly in the package and each contradicts something in the original data-export handoff, which is why they are worth restating rather than assuming.

### 6.1 Pooling is a mean, not a sum
Section 4.1, equation (3). Wrong by the factor $n_e$ if taken from the handoff's sidecar.

### 6.2 `simtime` in the MEA output is inferred from the last spike
`process_campaign.py` computes `simtime = float(np.ceil(spk_t.max()))`, so a quiet simulation records a duration far below the requested one, and any trace shorter than the window length $W$ is dropped with a silent `continue`. **Always pass `T` from the `--simtime` launch flag in `job_args.json`**; `build_pooled_ifr` makes `T` a required argument for this reason. The profiler compares the two per unit -- a dataset-level scalar comparison misses it, because the mismatch is per simulation.

### 6.3 The topology block is not in `mea_iter_*.npz`
`process_campaign.py` writes `conn_prob` but not `p0_conn`, `d0_conn`, `beta_conn`; those exist only in the original `iter_*.npz`. The export joins on `(topo_idx, iter_idx)` and raises if that key is not unique rather than inventing a surrogate. This is why `SIM_ROOT` is not optional.

### 6.4 The topology axes are linear-uniform, never log
`sample_kernel_vector` and the `conn_prob` draw both call `rng.uniform` on natural bounds. The log-axis rule must not be applied to them: `p0_conn` has bounds `[0.1, 1.0]`, i.e. exactly 1.000000 decades, so the rule would classify it as a log axis and the export would store $\ln(p_0)$ against a linear prior box.

### 6.5 An all-NaN axis is counted as SWEPT, not as constant

`[CLUSTER RUN 2026-09-10]` Under `conn_rule=flat` the Weibull kernel axes are never drawn and are written as NaN. `preflight_label_axes.py` nonetheless reported:

```
p0_conn            4467  in theta      [nan, nan]
d0_conn            4467  in theta      [nan, nan]
beta_conn          4467  in theta      [nan, nan]
```

`n_distinct = 4467` equals `topologies_sampled` exactly: every value counted as unique, because `nan != nan` and a set-based distinct count sees 4467 mutually distinct non-values. `campaign_axis_audit.py`, reading the same files, correctly reports `n_distinct: 1` for each.

This slips past the script's own degenerate-axis check -- its docstring promises that an axis which never varies is *"DETECTABLE from the data, and handled automatically here"*, and NaN is exactly the case that defeats it. It is the mirror of the `conn_prob` problem the script was written for: that one varies but is inert, this one is constant but looks varied.

**Consequence if not caught:** three all-NaN columns frozen into `theta` with NaN bounds. Assertion A9 (no NaN/Inf) fires at export, so it fails loudly rather than corrupting -- but it fails on every job. **Workaround:** name them in `--exclude`, as sec. 5.5 does. The trap remains armed in the tool.

### 6.6 The live registry is not the registry that wrote the data

`[CLUSTER RUN 2026-09-10]` This one stopped the first real dry run, with:

```
ValueError: assertion A6 failed on axis 0 (Sigma): stored theta=1.8010068260352963
but the coordinate rule applied to params gives 6.055741474273584.
```

`load_registry` derives the log-axis set from the **live** `HPC_main_sweep.PARAM_BOUNDS` via rule (1): an axis is `ln` iff its bounds span at least one decade. `Sigma` was recorded as `[1.0, 10.0]` -- **exactly 1.000000 decades**, therefore `ln` -- and the live source had since been changed to `[2.0, 10.0]`, i.e. 0.698970 decades, therefore linear. The stored `theta` holds `ln(6.0557) = 1.8010`; a registry loaded from live source reads it as a natural value. Confirmed exactly: `exp(1.8010068260352963) == 6.055741474273584`.

This is trap 6.4 in mirror image. There, `p0_conn` at `[0.1, 1.0]` sits at exactly 1.000000 decades and would be *wrongly* classified as log; here `Sigma` sat at exactly 1.000000 decades and was *correctly* log until a bound moved and knocked it off the boundary. **An axis parked precisely on the threshold flips coordinate under any bound change at all.**

`O_N` is the same root cause with a milder symptom: `[0.03, 3.0]` live against `[0.01, 3.0]` recorded, both over a decade, so no coordinate flip -- but the declared box would still come from the live bounds, putting the ~0.0103 observed minima outside it and reporting A5 on most rows.

Two guards did **not** catch it, and it is worth knowing why. `load_registry` cross-checks rule (1) re-derived from `PARAM_BOUNDS` against the simulator's own `LOG_PARAMS` and refuses if they disagree -- but both were updated in lockstep, so it guards the two copies of rule (1) against each other, not either against the data. And `dataset_profile.py` tracks only `conn_prob_lo/hi`, not the 37-row bounds table, so its prior-box field was constant and clean.

**Fixed** by `registry_from_manifest()` in `sbi_labels.py`, wired into `example_export.py` after the manifest loads: `param_bounds`, `param_bounds_theta` and the log set now come from the campaign's own `manifest.json`, which records what was in force when its npz files were written. It applies the same rule-(1)-vs-recorded-`log_params` cross-check to the manifest and refuses rather than guessing. New banner lines: `bounds re-sourced from manifest.json`, `differ from the live simulator source on: ...`, `COORDINATE FLIP vs the live source on: ...`.

Because bounds are now per task, a campaign whose tasks were launched against different code versions produces shards with different boxes. v1 is exactly that case (sec. 12.3). That surfaces at pooling instead of hiding, which is the intended behaviour.

### 6.7 The sidecar recorded a flag, not the value that was used

`[CLUSTER RUN 2026-09-10]` `--n_electrodes` is normally omitted, because `n_e` is read from `electrode_centers`. The sidecar echoed the flag, so it wrote `"n_electrodes": null` for a shard whose amplitude scale depends entirely on `n_e` -- the pooled IFR is a mean over exactly `n_e` electrodes (eq. 3). The computation was correct; only the provenance was missing, which makes the shard uncheckable for parity afterwards: `compare_profiles` returns `not_comparable`, never agreement.

**Fixed** by `record_resolved_n_e()` in `example_export.py`, which writes the resolved value and its source (`electrode_centers` or `--n_electrodes`) and raises if `n_e` changes mid-shard -- one sidecar declares one `n_electrodes` for every row, so a mixed shard is silently mis-scaled.

## 7. Standard invocation chain

Steps 1-3 are cheap and read-only; do not skip them to save minutes on a job that takes hours.

```bash
# 0a. environment for steps 1-1b: numpy is all these need. sbi_env works.
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

# 0b. environment for steps 2-5: the export pipeline needs sbi_export, NOT
#     sbi_env -- `[KB -- HPC_PATHS.md sec. 7]`: "Use sbi_export for
#     everything in the export pipeline: python 3.11.15, torch 2.13.0,
#     numpy 2.4.6. Never use base." sbi_env is for the NPE/tuning stage
#     downstream of export (gate_run.py, witness_run.py), a different stage
#     with its own separate env of the same name pattern. Conflating the two
#     was a bug in an earlier version of this document.
conda deactivate && conda activate sbi_export
python3 -c "import sys, torch; print(sys.version.split()[0], torch.__version__)"

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

### 7.0 Verified entry-point interfaces

`[CLUSTER RUN 2026-09-10, from `--help` and from reading the scripts]`. These close most of v3's `[TO VERIFY]` flags.

**`preflight_label_axes.py`** -- `--sim_main` (required), `--campaigns` (glob, default `campaign_*`), `--exclude NAME=REASON` (repeatable), `--require-conn-rule`, `--out` (omit to report only, writing nothing).

**`example_export.py`** -- `--mode {synthetic,campaign}`, `--out` (stem, no extension), `--dsn_main_dir`, `--sim_dir`, `--checkpoint`, `--campaign`, `--mea_out`, `--campaign_id`, `--n_electrodes` (read from `electrode_centers` when omitted), `--simtime` (read from `job_args.json` when omitted; **never** from the npz), `--trim_head_s`, `--sweep_group`, `--label_axes`, `--conn_prob_lo`, `--conn_prob_hi`, `--batch_size`, `--max_records`, `--n_sims`, `--device`.

Two of those decide correctness rather than convenience. **Omitting `--label_axes` silently falls back to "legacy 4-axis behaviour"** -- a different, much smaller theta; never skip it for a real run. And `--trim_head_s` (burn-in discarded before windowing, `T - trim_head_s` must still be at least the DSN window) exists in neither arm's metadata, so it is an export-time choice that must be recorded deliberately; 0 unless there is a reason.

**`launch_sweep_exports.sh`** -- four positional arguments then an optional glob:

```
./launch_sweep_exports.sh <CKPT> <MEA_ROOT> <SIM_ROOT> <OUT_ROOT> [GLOB]
```

Environment: `SIM_MAIN_DIR` (**required**, the tree `sbi_labels` imports from), `DSN_MAIN_DIR` (required), `LABEL_AXES`, `SELECT` (default `select=1:ncpus=8:mem=32gb`), `WALLTIME` (default `02:00:00`), `GLOB`, `DRYRUN=1`.

Three behaviours worth knowing. It enumerates `(campaign, sweep_task)` pairs explicitly, replacing `launch_all_campaigns.sh`, which only ever processed `sweep_cpu_task0000`. It globs **`MEA_ROOT`**, not `SIM_ROOT`, so campaigns with no MEA output cannot appear at all. And it **refuses to submit when `DSN_MAIN_DIR` contains whitespace**, because `qsub -v` cannot carry it -- use the `artifacts/dsn_main` symlink, never the resolved path (sec. 8).

**`submit_sbi_export.sh`** -- reads `CKPT`, `CAMPAIGN`, `MEA_OUT`, `OUT` (all required), plus `CAMPAIGN_ID`, `DSN_MAIN_DIR`, `SIM_MAIN_DIR`, `ENV_NAME`, `LABEL_AXES`, `MAX_RECORDS`, `SIMTIME`, `TRIM_HEAD_S`. It builds its `example_export.py` command line from a hardcoded `EXTRA=""`, so **only those four optional flags can reach the exporter through the array**; anything else needs a code change.

**`LABEL_AXES` defaults to `${ARTIFACTS_DIR}/label_axes.json`** -- the r2 weibull file. For a flat campaign set that would trip the NaN guard in `assemble_theta_A` on every job. Always export `LABEL_AXES` explicitly.

**`env.sh`** uses `: "${VAR:=default}"` throughout, which sets a variable only when unset or empty, so a stale export from an earlier shell silently wins over the file. `CKPT` is deliberately **not** defaulted: which encoder is in use is a scientific choice and stays explicit at every invocation.

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

`[KB -- HPC_PATHS.md sec. 7]`. `sbi_env` and `sbi_export` are two real, separate conda environments (confirmed distinct paths under `.conda/envs/`, neither a typo for the other). **`sbi_export` is the one the export pipeline itself runs under** -- `preflight_label_axes.py`, `example_export.py`, `launch_sweep_exports.sh`, `check_preprocessing_parity.py` -- python 3.11.15, torch 2.13.0, numpy 2.4.6, never `base` (a different torch major version changes the `torch.load` `weights_only` default, which decides whether a checkpoint's config is even readable). `sbi_env` is a separate environment for the NPE/tuning stage downstream of export (`gate_run.py`, `witness_run.py`) and is also sufficient for `dataset_profile.py`/`campaign_axis_audit.py`, which need only numpy. See the corrected sec. 7 invocation chain. `submit_sbi_export.sh` activates via `eval "$(conda shell.bash hook)"` -- a bare `conda activate` fails silently under PBS's non-interactive shell -- and then resolves the interpreter by absolute path rather than trusting `PATH`. Do not replace that block with `module load python`.

**`DSN_MAIN_DIR` must be the symlink, never the resolved path.** `[CLUSTER RUN 2026-09-10]` The real DSN directory contains a literal space (`.../Deep Summary Network/Deep_bio/Main`) and `qsub -v` cannot carry it; `launch_sweep_exports.sh` refuses to submit rather than letting 51 jobs die on the node. Use `repos/Sbi-extractor/artifacts/dsn_main`. Because `env.sh` only sets unset variables, a `DSN_MAIN_DIR` already exported in the shell -- possibly resolved to the spaced target -- takes priority; check with `echo "[$DSN_MAIN_DIR]"` before launching.

**The launcher may not be executable after a fresh clone.** `[CLUSTER RUN]` `./launch_sweep_exports.sh` gave `Permission denied`. Use `bash ./launch_sweep_exports.sh`, or fix it in git so the next clone does not hit it:

```bash
chmod +x launch_sweep_exports.sh submit_sbi_export.sh
git update-index --chmod=+x launch_sweep_exports.sh submit_sbi_export.sh
```

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
| `assertion A6 failed on axis N` | trap 6.6: the live registry's bounds are not the ones that wrote the data | the banner's `COORDINATE FLIP` line; `manifest.json` `param_bounds` vs `PARAM_BOUNDS` |
| an axis shows `n_distinct == topologies_sampled` with range `[nan, nan]` | trap 6.5: all-NaN axis counted as swept | exclude it by name |
| `ERROR: DSN_MAIN_DIR contains whitespace` | the resolved path was used instead of the symlink | sec. 8 |
| `Permission denied` on the launcher | missing execute bit after clone | `bash ./launch_sweep_exports.sh`; sec. 8 |
| sidecar `n_electrodes: null` | trap 6.7, on a pre-fix shard | re-export, or read `n_e` from the MEA root's probe config |
| every job fails immediately across the whole array | `LABEL_AXES` fell back to the r2 default | sec. 7.0 |

## 10. Known contradictions in the sources

The `sbi-export` README in project knowledge records `E = 16` (from `config_mea_joint_full.json`), a 27-column `theta_A` built from `sweep_groups["neuron_synapse"]`, and `n_e = 9`. `SBI_PIPELINE.md` sec. 4 and `HPC_PATHS.md` sec. 3 record the encoder actually in use as **r2** (`refit_mea_joint_full_r2_l0_t82`), `E = 10`, with $p = 26$ and the export of record at `ANN/SBI_export_r2/`, 54 shards, 86,251 rows.

These describe different vintages of the same pipeline. **Do not resolve this from memory in either direction** -- profile the sidecar and read `embedding.embedding_dim`, `dsn_checkpoint_sha256` and `len(param_names)` off the shard that exists. That is exactly what `_profile_export` does, and it is the reason the parity contract holds no reference constants.

## 11. Open items

- **CLOSED v4** -- the declared flags of `example_export.py`, `preflight_label_axes.py`, `launch_sweep_exports.sh` and `submit_sbi_export.sh` are now recorded in sec. 7.0. `check_preprocessing_parity.py` remains `[TO VERIFY]`; it has still never been run, and with `n_e = 1` against a 9-electrode real arm it is expected to fail by design.
- `[TO VERIFY]` The exact key schema of the real extracted archives. `dataset_profile._profile_real` scans a candidate key list (`ifr_trace`, `fs_ifr`, `T_rec`, `culture_id`, `electrodes_per_subset`, ...) and reports the full key list either way, so a surprise surfaces rather than being silently mapped. Confirmed from project knowledge only that these files exist as 315 npz, `fs_ifr = 100.0`, `T_rec = 1200.0`, `K = 120000` `[KB -- HPC_PATHS.md sec. 3]`.
- `[TO VERIFY]` Whether `real_source.py` records `n_e` in its sidecar's observable block. If it does not, the hard field `observable.n_e` will come back `not_comparable` for the real arm and must be supplied from the extraction config by hand.
- **Not implemented:** a dedup-aware manifest filter, so an array run against a full campaign glob processes duplicate tasks `[KB -- MEA analysis reference sec. 6]`.
- **Not implemented:** the real-recording export path and the OOD probe set `[KB -- sbi-export README sec. 10]`.
- **Not inspected:** `hpc/Electrode Traces Extractor/` in the simulator repo is a separate, MEA-adjacent tool with no known cluster deployment; it is not this pipeline and has not been compared against it `[KB -- MEA analysis reference sec. 8]`.
- **Open, unresolved:** whether the collapsed embedding ($r_{\rm eff} = 1.017$ simulated, $1.000$ real, of $E = 10$) should be fixed before any restriction is derived -- O2 in `SBI_PIPELINE.md` sec. 13. Every parity verdict in this document is conditional on the frozen encoder, and none of it addresses that.
- `[CLUSTER RUN, sec. 12.1]` The v1 `sweep_cfd_task*` units under `mea_out` have no `mea_manifest.json` and about half the topologies of their 1-electrode counterparts. Cause not established -- job still running, killed, or a different `--limit_topos`. Re-profile that root before exporting from it.
- `[CLUSTER RUN, sec. 12]` Five sim units under a directory named `q/` (`q/sweep_intel_task0005`-`0009`) have no MEA output and appear in no project document. Unidentified; not v4, not hhgap-named. Find out what they are before assuming they are safe to ignore.
- `[CLUSTER RUN, sec. 12]` Two stray unpaired units sit under `mea_out`: `v1/sweep_intel_task0000` and `v1/sweep_intel_task0000_1electrode`, artefacts of the earlier hand-run single-task test. The second is 1-electrode output under the 4-electrode root and is what makes that root's `n_e` non-constant. Move or delete them rather than relying on the pairing step to skip them.
- **CLOSED v4** -- the full ordered list of all 22 swept axes is in sec. 12.3.
- **Not fixed, trap 6.5:** `preflight_label_axes.py` still counts an all-NaN axis as 4467-distinct-and-swept. The `--exclude` workaround is correct for this campaign set but leaves the trap armed. A distinct count that treats NaN as one value would close it.
- **Not fixed:** `submit_sbi_export.sh` builds its command line from a hardcoded `EXTRA=""`, so only `LABEL_AXES`, `MAX_RECORDS`, `SIMTIME` and `TRIM_HEAD_S` can reach `example_export.py` through the array. Anything else needs a code change.
- `[TO VERIFY]` Whether `WALLTIME=250:00:00` is accepted by the queue, and whether a very long request lands the jobs in a slower-scheduling class. `qmgr -c "list queue @default" | grep -i walltime` before relying on it (sec. 13.3).
- `[TO VERIFY]` The v1 `O_N` split: which of the 25 v1 tasks used `[0.03, 3.0]` and which `[0.01, 3.0]` (sec. 12.3). Per-task `manifest.json` `param_bounds[36]` answers it. Needed before pooling v1 shards with each other, not before exporting them.
- `[TO VERIFY]` `EXTRACTOR_USAGE.md` in the repo vs the copy in project knowledge -- they drift, and the project-knowledge copy is what future chats read. Re-upload after each version bump.
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

### 12.3 The axis audit of record

`[CLUSTER RUN 2026-09-10]` `campaign_axis_audit.py --registry-src ./HPC_single_run.py --sweep-src ./HPC_main_sweep.py`, run from `Giulia_Astro`; smoke test 54/54 first. Writes `./campaign_axis_audit/campaign_axis_audit.{md,json}`. All four campaigns: `mode=Full`, `sweep_group=synapse_astro`, `conn_rule=flat`, 300 um square, `Nn=Na=108`, `rho_n=1200 /mm^2`, 22 swept axes, 9 astro, **zero inert**.

The full 22, in registry order -- this closes v3's `[TO VERIFY]`:

```
Sigma, U_0_ar, U_max, U_0_sr, Omega_f_sr, Omega_f_ar, Omega_d, alpha_syn,
g_ampa, g_nmda, x0, O_G, Omega_G, O_beta, O_3K, Omega_5P, I_bias, F,
C_Theta, U_A, G_T, O_N
```

Every campaign carries the warning *"predates axis_declaration; swept/consumed/inert ... inferred from mode"*, which is why `dataset_profile.py` reports those fields as null (sec. 5.4). The inference is safe here because `mode=Full` instantiates the astrocyte objects.

**v1 alone carries a second warning:** *"param_bounds DIFFER between tasks of this campaign (2 distinct bound matrices) -- tasks were launched against different code versions."* The differing axis is `O_N`: `[0.03, 3.0]` in one group, `[0.01, 3.0]` in the other, both over a decade so both `ln`, no coordinate flip. v2, v4 and v5 are single-valued. With trap 6.6 fixed this is per-task and surfaces at pooling; take the wider box (`0.01`) when combining, since observed minima are ~0.0103 everywhere.

**Cross-campaign seed collisions**, new information not previously on record:

| pair | overlap |
|---|---|
| v1 vs v2 | 18 |
| v1 vs v5 | 18 |
| v2 vs v5 | 20 |
| v4 vs v5 | 7, **prefix containment** |
| v1/v2 vs v4 | 0 |

These do **not** affect the current export: `dataset_profile.py` checked the 51 units actually in `mea_out_1electrode` and found 51 distinct seeds, zero duplicates. The overlaps live in the `intel_task` swaths and v4, none of which were MEA-processed. The MEA selection is self-consistent with the dedup accounting: 7 v1 `cfd` + 18 v1 `intel` + 6 v2 + 20 v5 = 51.

## 13. The export run of record -- `SBI_export_hhgap_1e_r2`

### 13.1 Single-task dry run

`[CLUSTER RUN 2026-09-10]` against `campaign_cadex_hhgap_v5/sweep_cpu_task0000`, `--max_records 20`.

| field | value |
|---|---|
| encoder | `artifacts/frozen_dsn/dsn_r2_20260824.pt`, sha256 `f286f9b71b9f8a8900fa7952a4a42e0b1056b00802fc0063fbf2d16106feb330` |
| geometry | `E = 10`, `W = 18000`, `T_win = 180 s`, `fs_ifr = 100 Hz`, `Delta_t = 0.01 s`, `sigma_sm = 0.02 s` |
| `p` | **23** = 22 run_args + 1 topology (`conn_prob`) |
| coordinates | 20 `ln`, 3 linear -- `I_bias` (`[0.3, 1.0]`, 0.52 dec) and `U_A` (`[0.1, 0.9]`, 0.95 dec) stay linear, plus `conn_prob` |
| `Sigma` | `ln`, box `[0, 2.30259]` = `[ln 1, ln 10]` -- trap 6.6 fixed, visible in one line |
| `O_N` | `ln`, box `[-4.60517, 1.09861]` = `[ln 0.01, ln 3]`, the v5 vintage, not the live `0.03` |
| `conn_prob` | `linear [0.05, 0.4]` |
| rows / used / skipped | 20 / 20 / **0** |
| assertions | A2, A3, A4, A5, A7, A9 |
| observable | `pooling: mean_over_electrodes`, `n_electrodes: 1`, `electrode_forward_model: true` |

The only warning is the expected cross-file A8 note. A5 passing means every row sits inside its declared box -- worth reading explicitly, since A5 *reports* rather than raises, so a clean `assertions_passed` list alone would not prove it.

### 13.2 Launcher configuration

Six settings differ from the r2-era defaults, each a silent wrong answer if missed:

| setting | value |
|---|---|
| MEA root | `ANN/MEA_analysis/mea_out_1electrode` |
| sim root **and** `SIM_MAIN_DIR` | `ANN/Phenomenological/Main/Giulia_Astro` (the same path here; both were `Main/` in the r2 era) |
| `LABEL_AXES` | `artifacts/label_axes_hhgap.json` |
| checkpoint | `artifacts/frozen_dsn/dsn_r2_20260824.pt` |
| `DSN_MAIN_DIR` | `artifacts/dsn_main` (symlink -- sec. 8) |
| out root | `ANN/SBI_export_hhgap_1e_r2` -- new, encoding `n_e` and the encoder, the two hard fields that decide poolability |

`conn_prob` bounds need no flag: the launcher never passes them and `job_args.json` supplies `0.05 / 0.4`.

```bash
cd /davinci-1/home/ldellamea/repos/Sbi-extractor
source env.sh
export DSN_MAIN_DIR=/davinci-1/home/ldellamea/repos/Sbi-extractor/artifacts/dsn_main
export SIM_MAIN_DIR=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro
export LABEL_AXES=/davinci-1/home/ldellamea/repos/Sbi-extractor/artifacts/label_axes_hhgap.json

DRYRUN=1 bash ./launch_sweep_exports.sh \
    /davinci-1/home/ldellamea/repos/Sbi-extractor/artifacts/frozen_dsn/dsn_r2_20260824.pt \
    /davinci-1/home/ldellamea/ANN/MEA_analysis/mea_out_1electrode \
    /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro \
    /davinci-1/home/ldellamea/ANN/SBI_export_hhgap_1e_r2 \
    'campaign_cadex_hhgap_v[125]'
```

### 13.3 Launcher dry run

`[CLUSTER RUN 2026-09-10]` 51 sweep tasks seen, 51 submitted, 0 skipped, **980,959** simulations, `51 180.0` declared simtime values -- constant across every task.

| family | tasks | sims/task | share of work |
|---|---:|---:|---:|
| v1 `cfd_task*` | 7 | 63,035 - 73,287 | ~49% |
| v1 + v2 `intel_task*` | 24 | ~15,000 - 17,700 | ~40% |
| v5 `cpu_task*` | 20 | 4,361 - 6,479 | ~12% |

**Walltime is uniform at the default `02:00:00` while task size varies about 17x.** `submit_sbi_export.sh`'s own comment says to raise it only above ~10^4 simulations; the `intel` tasks are already there and the `cfd` tasks are 6-7x it. Set `WALLTIME` deliberately, or submit v5 alone first and measure with `qstat -xf <jobid> | grep -i used` before committing the large families.

`CAMPAIGN_ID` is `<campaign>__<task>`, i.e. per shard rather than per campaign -- different from the r2 bank's convention; relevant when grouping rows later.

The footer's gate arithmetic (`n_real = 1890` from 35 x 9 x 6) is r2-era and assumes the real arm at `n_e = 9`. With this bank at `n_e = 1` that bar is not the operative constraint; the parity break is (sec. 4.1).
