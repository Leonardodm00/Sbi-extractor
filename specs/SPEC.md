# SPEC -- Sbi-extractor (sbi-export): core export path

Single source of truth for what the code must do. The code follows this file;
when the two disagree, fix whichever is wrong and record why in the decision
log. The integration tester reads only the repository, so every constraint the
code must satisfy is stated here, including those that come from decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears; quantifiers ("for each fixed k")
are stated explicitly.

For code that already exists, this file is drafted from the code and then
confirmed by the author. Every entry drafted from the code says
`Confirmed: inferred` until the author has checked it against what was
intended; anything else stated outside the block entries that is still
unconfirmed is marked `(inferred)` where it appears.

This draft was written on 2026-10-06 from README.md and the docstrings of the
covered files, at commit a8e9a4399808fa6c8e6508e83bbbfeda0bd1217b
(branch feat/real-arm-parity).

Last reviewed at commit: a8e9a4399808fa6c8e6508e83bbbfeda0bd1217b

## 0. Coverage

Paths this spec covers:

- `sim_observable.py`
- `sbi_labels.py`
- `dsn_frozen.py`
- `dsn_tree.py`
- `export_embeddings.py`
- `example_export.py`

Everything else in the repository is not specified yet and is out of scope for
the tester: the cohort and real-data modules, dataset profiling, the preflight
scripts, the extractor, raster and sim_reextract folders, and the shell
launchers. The existing smoke test smoke_test_sbi_export.py exercises the
covered files.

## 1. Scientific goal

Produce the training pairs (z, theta_A) for the amortized neural posterior
estimator (NPE). For each simulation of the Astro-Neuron-Network simulator
that has been through its virtual-MEA pipeline:

1. build the pooled observable x from the DETECTED spikes (not the simulator's
   ground-truth spikes, so that the simulated and real arms both contain the
   detection stage);
2. embed it with the frozen Deep Summary Network, z = h_psi(x);
3. pair z with the label theta_A of the parameters that generated the
   simulation.

Real recordings go through the same export function with theta_A = None.

Deliverables, per shard: a Parquet table `sbi_<campaign_id>_<shard>.parquet`
with one row per window, and a JSON sidecar `sbi_<campaign_id>_<shard>.json`.
Consumer: NPE training, after shards are pooled (pooling is outside this
repository).

## 2. Mathematical formulation

### 2.1 Observable x (Block 1)

For one simulation or recording, let n_e be the number of member electrodes and,
for each electrode e in {1, ..., n_e}, let S_e be the set of detected spike
times [s]. Given the duration T [s], the bin width Delta_t [s] and the
smoothing width sigma_sm [s], define

    K      = floor(T / Delta_t)
    I_k    = [k Delta_t, (k + 1) Delta_t)                     for each k in {0, ..., K - 1}
    C[k]   = sum_{e=1}^{n_e} | S_e intersect I_k |            for each k in {0, ..., K - 1}
    R~[k]  = max(0, (g_{sigma_sm / Delta_t} * C)[k])          for each k in {0, ..., K - 1}
    x[k]   = R~[k] / n_e                                      for each k in {0, ..., K - 1}

where g_s * C is scipy.ndimage.gaussian_filter1d(C, sigma = s) with scipy's
default boundary mode ("reflect") and truncation (4 sigma), and x is stored as
float32 (R~ is cast to float32 before the division by n_e, then the quotient is
cast to float32 again).

- One implementation for both arms. The export path computes C and R~ by
  calling compute_ifr_trace in <SBI_HPC_DIR>/dsn/generate_burst_data.py, the
  same function the real-arm extractor calls, and then divides by n_e in the
  same order the extractor does. A second copy of the recipe is not allowed
  (parity contract, EXTRACTOR_USAGE.md S4/S6).
- x is a per-electrode MEAN. Pooling as a sum over electrodes is wrong by a
  factor of n_e.
- n_e counts MEMBER electrodes, silent ones included: n_e = 9 for both the
  3 x 3 simulated probe and a real subregion (electrodes_per_subset = 9).
- T is the requested duration (the campaign's --simtime launch flag in
  job_args.json), never a duration inferred from the spikes.
- Units of x: spikes per bin per electrode. Multiply by fs_ifr = 1 / Delta_t
  for Hz.
- A spike at exactly t = K Delta_t, the right end of the grid, is outside the
  contract: counting it in the last bin or discarding it are both acceptable,
  and no test may depend on it. (confirmed by the author, 2026-10-06)
- Count conservation: for each spike train with no spike at exactly
  t = K Delta_t,

      n_e * sum_{k=0}^{K-1} x[k] = sum_{e=1}^{n_e} | S_e intersect [0, K Delta_t) |

  up to float32 rounding (relative tolerance 1e-6). The smoothing
  redistributes counts between bins but neither creates nor loses them.
  (confirmed by the author, 2026-10-06)

Windows. With W the window length in samples, W = round(T_win / Delta_t) (T_win,
Delta_t and sigma_sm are read from the encoder checkpoint), and a stride S
(default S = W, disjoint windows), window i covers samples
x[i S], ..., x[i S + W - 1] for each i in {0, ..., n_win - 1}, where

    n_win = floor((K - W) / S) + 1   if K >= W,      n_win = 0   if K < W.

This is MEAWindowDataset's index rule. Burn-in trim (Block 5): when
trim_head_s > 0, the first n_trim = round(trim_head_s / Delta_t) samples are
removed AFTER x is built on the full [0, T] grid and BEFORE windowing.

### 2.2 Label theta_A and prior box B (Block 2)

The simulator registry has n axes with natural bounds [lo_j, hi_j] for each
j in {0, ..., n - 1}. The set of log axes is

    L = { j : lo_j > 0  and  hi_j > 0  and  log10(hi_j / lo_j) >= 1 }      (rule 1)

derived from the registry bounds at run time, never from a hard-coded list.
The threshold is decided robustly to floating-point rounding: bounds that
span exactly one decade in exact arithmetic are in L even when the computed
log10(hi_j / lo_j) falls just below 1, i.e. the computed test is
log10(hi_j / lo_j) >= 1 - 1e-12. (confirmed by the author, 2026-10-06)
The inference coordinate of axis j, for each j, is

    theta_j = ln(vartheta_j)   if j in L,      theta_j = vartheta_j   otherwise,

with vartheta_j the natural value and ln the natural logarithm (never log10).
The same rule must hold where theta is written (the simulator) and where it
is read back (this repository), since the stored theta is sliced, never
re-transformed.

The label of one simulation is

    theta_A = ( (theta_j) for j in A , (eta_a) for a in Tau )

where A is the ordered list of active registry indices (from the campaign's
manifest.json) and Tau is the ordered list of topology axes, a subset of
(conn_prob, p0_conn, d0_conn, beta_conn) fixed by the frozen label_axes.json.
The topology values eta_a are always LINEAR and in natural units: rule 1 is
never applied to them (p0_conn has bounds [0.1, 1.0], exactly one decade, and
would otherwise be misclassified). So p = |A| + |Tau|.

The prior box B in R^{p x 2}: for an active axis j, [ln lo_j, ln hi_j] if
j in L and [lo_j, hi_j] otherwise; for a topology axis, its natural bounds
looked up BY NAME (conn_prob from the campaign's --conn_prob_lo/--conn_prob_hi,
default (0.1, 0.6); p0_conn, d0_conn, beta_conn from the campaign's kernel
bounds, default the registry's). Every row of B must have hi > lo (assertion
A3); the frozen axes DeltaT, VT and gL have lo = hi and can never be active.

The stored theta in each mea_iter npz is ALREADY in inference coordinates and
is sliced, never re-transformed. When the natural vector params is available,
assertion A6 checks, for each j in A, theta_j = ln(params_j) if j in L and
theta_j = params_j otherwise (relative tolerance 1e-8, absolute 1e-10).

Current registry values (2026-10, read from the simulator, NOT invariants):
n = 36, |L| = 26, |A| = 23 (18 ln, 5 linear), p = 27 with the 4-axis topology
block.

### 2.3 Encoder (Block 3)

For each window x_i, for each i in {0, ..., N - 1}:

    zraw_i = output of model.head.proj (the head's final nn.Linear) for input x_i
    z_i    = zraw_i / ||zraw_i||_2        (when the checkpoint has l2_normalize)

so z_i lies on the unit sphere S^{E-1} in R^E. E, the in-channel count C_in,
l2_normalize, window_s = T_win, w_size = Delta_t and gaussian_window = sigma_sm
are read from the checkpoint's embedded config, never from module defaults;
W = round(window_s / w_size); fs_ifr = 1 / w_size.

### 2.4 Export (Block 4)

Every record (one trace, its label and its provenance) is cut into windows
(section 2.1); every window becomes one Parquet row that inherits the record's
theta_A and provenance plus its own window_idx in {0, ..., n_win - 1}. Rows
keep record order and window order.

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| detection times | `<mea_out>/topo_<k>/mea_iter_<n>.npz`, key det_t | (n_det,) | float64 | s | detected spikes (band-pass + Quiroga threshold) |
| detection channels | same file, key det_ch | (n_det,) | int64 | - | electrode index in {0, ..., n_e - 1} (inferred) |
| electrode_centers | same file | (n_e, ...) | float | um | n_e = number of rows |
| theta | same file, key theta | (n,) | float64 | inference coordinates | registry order |
| params (optional) | same file, key params | (n,) | float64 | natural units | registry order; enables A6 |
| topo_idx, iter_idx, seed_run | same file (or file names) | scalars | int | - | (topo_idx, iter_idx) is a unique key |
| topology values | `<campaign>/topo_<k>/iter_<n>.npz`, keys conn_prob, p0_conn, d0_conn, beta_conn | scalars | float | natural | NOT present in mea_iter files; joined on (topo_idx, iter_idx) |
| manifest | `<campaign>/manifest.json` | - | - | - | active_indices, sweep_group, optionally param_bounds |
| launch flags | `<campaign>/job_args.json` | - | - | s | simtime is T; conn_prob_lo, conn_prob_hi |
| checkpoint | `.pt` written by the DSN tree's checkpoint.save_checkpoint | - | - | - | config embedded; identity = SHA-256 of the file |
| observable x | in memory | (K,) | float32 | spikes / bin / electrode | sample k covers I_k |
| Parquet shard | out_stem + ".parquet" | one row per window | see Block 4 | - | snappy compression |
| sidecar | out_stem + ".json" | - | - | - | see Block 4 |

## 4. Global conventions

- Units used internally: seconds and Hz; natural log for ln axes.
- Floating-point precision: x, z, zraw float32; theta_A and B float64.
- Randomness: the export path is deterministic given its inputs. The
  synthetic demo draws from numpy.random.default_rng(0).
- No silent defaults for the observable: Delta_t, sigma_sm, T_win and E come
  from the checkpoint (or, under Stage C, the cohort manifest). The library
  defaults 0.02 s / 0.04 s are not a cohort's values.
- Failures raise. Nothing is silently clipped, repaired or dropped; a trace too
  short for one window is counted and reported, never silently skipped.
- Binding decisions visible in the code (inferred; the decision log itself is
  not in this repository, see open question Q5):
  - 2026-09-19, migration step 4b: one IFR function for both arms
    (compute_ifr_trace in the DSN tree).
  - 2026-09-24: --label_axes is required in campaign mode ('none' requests the
    legacy 4-axis block by name).
  - 2026-09-24, Stage C item C2a: dt and sigma_sm have no defaults in
    iter_campaign_records.

## 5. Blocks

### Block 1: observable

- Module: `sim_observable.py`
- Public API: `build_pooled_ifr(spike_times_s, n_electrodes, T, dt, sigma_sm, normalise_per_electrode=True) -> ndarray`;
  `window_trace(x, window_length, stride=None) -> (Xw, starts)`;
  `reference_compute_ifr_trace(spike_times_s, T, dt, sigma_sm, dsn_main_dir=None) -> (ifr, fs_ifr)`;
  `pooled_spike_counts(spike_times_s, T, dt) -> ndarray` (readable statement
  of C only, NOT on the export path; it discards a spike at t = K Delta_t)
- Inputs: spike_times_s, a 1-D float array of pooled spike times [s] or a
  sequence of per-electrode 1-D arrays (both give the same x); n_electrodes =
  n_e >= 1; T > 0 finite [s]; dt = Delta_t > 0 finite [s]; sigma_sm >= 0 [s].
  For window_trace: x of shape (K,) or (C_in, K); window_length W >= 1;
  stride S >= 1 or None.
- Outputs: x, shape (K,), float32, non-negative, units spikes per bin per
  electrode. window_trace: Xw of shape (n_win, W) or (n_win, C_in, W),
  float32, cut along the last axis, and the list of start offsets.
- Errors: ValueError for NaN or Inf spike times, T or dt not finite and
  positive, floor(T / dt) < 1, n_e < 1, sigma_sm < 0, W < 1, S < 1.
- Parameters: Delta_t and sigma_sm from the checkpoint; n_e from
  electrode_centers; T from job_args.json.
- Library calls relied on: compute_ifr_trace (DSN tree), which uses
  numpy.histogram, scipy.ndimage.gaussian_filter1d and numpy.clip.
- Custom code: window_trace, which reproduces MEAWindowDataset's index rule
  (that rule, not a library's, is the contract).
- Test oracles:
  - Independent recomputation, on spike trains with no spike at exactly
    t = K Delta_t: numpy.histogram on edges k Delta_t, then gaussian_filter1d
    with sigma = sigma_sm / Delta_t, clip at 0, cast to float32, divide by
    n_e, cast to float32; equal to x bit for bit.
  - Mean, not sum: n_e identical electrodes with spike set S give the same x
    as one electrode with S (float32 rounding).
  - Silent electrodes count: appending an empty electrode with n_e + 1 scales
    x by n_e / (n_e + 1).
  - Count conservation (section 2.1, confirmed): for spike trains with no
    spike at exactly t = K Delta_t, n_e * sum_k x[k] equals the number of
    spikes in [0, K Delta_t), relative tolerance 1e-6 (float32), including
    spikes near both ends of the grid.
  - Declared duration: a train whose last spike is at 3 s, with T = 180 s and
    Delta_t = 0.02 s, gives K = 9000 (not 150).
  - Windows: n_win as in section 2.1; window i equals x[i S : i S + W];
    default stride is W; K < W gives zero windows, not an error.
  - Each invalid input listed under Errors raises ValueError.
- Data flow: feeds Block 4 (TraceRecord.trace); built by Block 5.
- Confirmed: inferred (drafted from the code, not yet checked by the author),
  except the points of section 2.1 marked (confirmed by the author, 2026-10-06)
- Status: existing code; smoke tests T1-T4 in smoke_test_sbi_export.py

### Block 2: label

- Module: `sbi_labels.py`
- Public API: `load_registry(sim_dir=None) -> Registry`;
  `build_label_spec(registry, active_indices, sweep_group="neuron_synapse", conn_prob_bounds=(0.1, 0.6), kernel_bounds=None, topology_axes=None, excluded_axes=None) -> LabelSpec`;
  `assemble_theta_A(spec, theta_registry, topology, params_registry=None, check_coord_tol=1e-8) -> ndarray`;
  diagnostics `coordinate_margins`, `at_risk_axes`, `format_margins`
- Inputs: the registry, read from the simulator repository (sim_dir or the
  SIM_MAIN_DIR environment variable); active_indices, unique, each in
  {0, ..., n - 1}; kernel_bounds of shape (3, 2) for (p0, d0, beta);
  theta_registry of shape (n,) in inference coordinates; topology, a dict with
  a natural value for every axis in spec.topology_axes; params_registry of
  shape (n,) in natural units, optional.
- Outputs: LabelSpec with param_names (p), coord (p entries, "ln" or
  "linear"), bounds_theta (p, 2) float64, units, active_indices,
  topology_axes, excluded_axes (name -> reason, carried to the sidecar);
  theta_A of shape (p,), float64, in param_names order.
- Errors: ValueError for empty, duplicate or out-of-range active indices,
  kernel_bounds not (3, 2), unknown or duplicate topology axes, duplicate
  label names, a degenerate prior row (A3), a theta or params width other
  than n, an A6 mismatch; KeyError for a missing topology value.
- Parameters: sweep_group (provenance only); conn_prob_bounds and
  kernel_bounds from the campaign's launch flags.
- Library calls relied on: numpy.
- Custom code: the coordinate rule and label assembly (domain bookkeeping).
- Test oracles:
  - L recomputed independently from the registry bounds with rule 1 equals
    registry.log_param_indices.
  - Rule 1 at the threshold (section 2.2, confirmed): bounds spanning exactly
    one decade are ln axes even when the floating-point ratio hi / lo comes
    out just below 10 (for example lo = 0.07, hi = 0.7, where hi / lo
    evaluates to 9.999999999999998).
  - Every topology axis is "linear", whatever its bounds.
  - bounds_theta row for active j equals rule 1 applied to [lo_j, hi_j];
    topology rows equal their natural bounds, matched by name even when an
    axis is excluded or reordered.
  - A1: natural -> theta -> natural returns both bound edges of every axis
    (relative tolerance 1e-9).
  - A3: making DeltaT, VT or gL active raises at construction.
  - A6: a theta inconsistent with params on one active axis raises.
  - assemble_theta_A returns the stored theta entries unchanged (no
    re-transform), followed by the topology values in topology_axes order.
- Data flow: feeds Block 4 (label_spec and theta_A); built by Block 5.
- Confirmed: inferred (drafted from the code, not yet checked by the author),
  except the rule 1 threshold of section 2.2 (confirmed by the author, 2026-10-06)
- Status: existing code; smoke tests T9-T11

### Block 3: encoder

- Modules: `dsn_frozen.py`, `dsn_tree.py`
- Public API: `load_frozen_dsn(ckpt_path, device="cpu", dsn_main_dir=None, expect_config_json=None, strict=False) -> FrozenDSN`;
  `FrozenDSN.embed(X, batch_size=256, want_zraw=True) -> (Z, Zraw)`;
  `FrozenDSN.fs_ifr`, `FrozenDSN.sigma_bins`, `FrozenDSN.embedding_block(zraw_available)`, `FrozenDSN.observable_block()`;
  `sha256_of_file(path)`; `dsn_tree.sbi_hpc_dir`, `dsn_tree.dsn_dir`,
  `dsn_tree.add_dsn_to_path`
- Inputs: a checkpoint file; X of shape (N, W) when C_in = 1, else
  (N, C_in, W), float.
- Outputs: Z of shape (N, E), float32, rows on S^{E-1} when l2_normalize;
  Zraw of shape (N, E), float32, or None.
- Guarantees: the architecture is rebuilt from the config embedded in the
  checkpoint; model.eval() during the forward pass and the caller's
  train/eval mode restored afterwards; the SHA-256 of the checkpoint file is
  carried; expect_config_json disagreements are warned (raised when strict).
- Errors: FileNotFoundError for a missing checkpoint; a window length or
  channel count that disagrees with the checkpoint is refused; a head without
  an nn.Linear at model.head.proj raises AttributeError; dsn_tree raises
  DSNTreeMissing (naming the fix) when <SBI_HPC_DIR>/dsn lacks any of
  generate_burst_data.py, config.py, make_mea_specs.py, backbone.py,
  checkpoint.py. Resolution order of SBI_HPC_DIR: explicit argument, the
  SBI_HPC_DIR environment variable, then artifacts/sbi_hpc.
- Parameters: none of its own; everything comes from the checkpoint.
- Library calls relied on: torch, torch.nn.functional.normalize, hashlib.
- Custom code: the forward hook that captures zraw.
- Test oracles:
  - max_i | ||z_i||_2 - 1 | < 1e-5 (assertion A7).
  - z_i = zraw_i / ||zraw_i||_2 (relative tolerance 1e-6, float32).
  - Z does not depend on batch_size beyond float32 rounding: max absolute
    difference <= 1e-6 between batch sizes 1, 5 and 256. (inferred: open
    question Q4)
  - The digest equals hashlib.sha256 of the file bytes.
  - W = round(window_s / w_size) from the checkpoint config.
- Not runnable here: the trained checkpoint lives on the cluster. Use the
  untrained demo backbone (example_export._make_demo_dsn), or a checkpoint
  saved from it with the DSN tree's checkpoint.save_checkpoint.
- Data flow: feeds Block 4.
- Confirmed: inferred (drafted from the code, not yet checked by the author)
- Status: existing code; smoke tests T5-T8c

### Block 4: export

- Module: `export_embeddings.py`
- Public API: `TraceRecord(trace, ident={}, theta_A=None)`;
  `export_embeddings(dsn, records, out_stem, label_spec=None, ident_columns=(), extra_sidecar=None, batch_size=256, window_stride=None, want_zraw=True, strict_in_box=True, allow_constant=()) -> ExportResult`;
  `run_assertion_A1(registry, rtol=1e-9)`
- Inputs: an iterable of TraceRecord, each with trace of shape (K,) or
  (C_in, K) built by Block 1, ident (provenance dict) and theta_A of shape
  (p,) or None (real data); out_stem, a path without extension.
- Outputs: `<out_stem>.parquet` with columns, in this order: the
  ident_columns in the given order; window_idx (int); z_000 ... z_{E-1}
  (float32); zraw_000 ... zraw_{E-1} (float32, when want_zraw); th_<name>
  for each name in label_spec.param_names (float64, when label_spec is
  given). `<out_stem>.json` with schema_version = 1, n_rows, n_traces_used,
  n_traces_skipped_too_short, window_stride_samples, embedding (including
  dsn_checkpoint_sha256), observable (including pooling =
  "mean_over_electrodes" and spike_source), assertions_passed (sorted),
  warnings; with a label spec also param_names, coord, param_units,
  bounds_theta and registry; extra_sidecar merged in. ExportResult with the
  two paths and the counts.
- Assertions (failures raise; nothing is clipped):
  - A9: no NaN or Inf in Z, Zraw or theta.
  - A7: max_i | ||z_i||_2 - 1 | < 1e-5 when l2_normalize (otherwise a warning).
  - A2: param_names, coord and bounds_theta all have length p.
  - A3: every row of bounds_theta has hi > lo.
  - A4: no th_* column with zero variance in the shard, except the axes in
    allow_constant (warned instead).
  - A5: every theta_A entry inside its [lo, hi]; raises when strict_in_box,
    otherwise counted in a warning.
  - A8 (checkpoint identity across files) and A10 (cross-campaign
    compatibility) are NOT checked here; the A8 warning carries the digest.
- Errors: ValueError when no record yields a window, when label_spec is given
  and a record has theta_A = None, or when theta_A has a length other than p.
  A record too short for one window is counted in n_traces_skipped_too_short
  and named in a warning.
- Writes are atomic: after a failure, no partial .parquet or .json exists at
  out_stem.
- Parameters: window_stride (default W, disjoint windows; overlap is warned);
  batch_size (throughput only).
- Library calls relied on: pyarrow, json.
- Custom code: none beyond orchestration.
- Test oracles:
  - n_rows = sum over records of n_win.
  - Column names, order and dtypes as listed; window_idx restarts at 0 for
    each record; every row of a record carries its ident and theta_A.
  - Reading the shard back (README section 7) reproduces Z and theta exactly.
  - Each assertion fires on a constructed violation.
  - With label_spec = None: no th_* columns, and A9 is the only assertion in
    assertions_passed.
- Data flow: consumes Blocks 1, 2 and 3; called by Block 5.
- Confirmed: inferred (drafted from the code, not yet checked by the author)
- Status: existing code; smoke test T12

### Block 5: entry point

- Module: `example_export.py`
- Public API: command line `python example_export.py --mode synthetic|campaign --out STEM [options]`;
  `iter_campaign_records(campaign_dir, mea_out_dir, spec, T_sim, campaign_id, n_electrodes=None, *, dt, sigma_sm, verify_coords=True, max_records=None, trim_head_s=0.0, observable_out=None)`;
  `iter_synthetic_records(spec, T_sim, n_sims, n_electrodes, dt, sigma_sm, seed=0)`
- Inputs: --sim_dir or SIM_MAIN_DIR (required in both modes); for campaign
  mode --checkpoint, --campaign, --mea_out, --label_axes (required; 'none'
  for the legacy 4-axis block), --campaign_id, optional --simtime,
  --n_electrodes, --trim_head_s, --conn_prob_lo/hi, --max_records.
- Outputs: one shard (Block 4) at --out; a banner with E, W, T_win, fs_ifr,
  Delta_t, sigma_sm, the digest, rows written, traces used and too short, and
  the assertions passed.
- Behaviour:
  - Synthetic mode needs no data and no checkpoint: an UNTRAINED demo
    backbone (E = 16, l2_normalize), a digest of 64 zeros and a DEMO MODE
    warning, fabricated detections for --n_sims simulations of T = 180 s.
    The embeddings carry no information.
  - Campaign mode: T from job_args.json (or --simtime), never the simtime
    field of the npz; Delta_t and sigma_sm from the checkpoint; n_e from
    electrode_centers (or --n_electrodes); per-electrode spike sets
    S_e = det_t[det_ch == e] for each e in {0, ..., n_e - 1}; the topology
    block joined from `<campaign>/topo_<k>/iter_<n>.npz` on
    (topo_idx, iter_idx), raising when there is not exactly one match or a
    required axis is missing; A6 checked when params is present.
- Library calls relied on: argparse, glob, json, numpy.
- Test oracles:
  - Synthetic mode with --n_sims 16 writes both files, 16 rows, and
    assertions A2, A3, A4, A5, A7, A9.
  - Campaign mode on a small fabricated campaign (npz keys as in section 3,
    job_args.json, manifest.json, a checkpoint saved from the demo backbone)
    joins the topology values of the right iter file onto each row.
  - Campaign mode without --label_axes is refused.
  - A different simtime field inside the npz does not change K.
- Data flow: wires Blocks 1, 2, 3 and 4.
- Confirmed: inferred (drafted from the code, not yet checked by the author)
- Status: existing code; synthetic mode run in a cloud sandbox on 2026-10-06

## 6. Pipeline entry points

Setup in a fresh environment (cloud sandbox or laptop):

    git clone https://github.com/Leonardodm00/Simulation-Based-Inference.git ../Simulation-Based-Inference
    git clone https://github.com/Leonardodm00/Astro-Neuron-Network.git ../Astro-Neuron-Network
    export SBI_HPC_DIR=$(cd ../Simulation-Based-Inference/hpc && pwd)
    export SIM_MAIN_DIR=$(cd ../Astro-Neuron-Network/hpc/Phenomenological_finalv1 && pwd)
    export MPLBACKEND=Agg
    pip install numpy scipy pyarrow torch matplotlib

Both public repositories are used at the head of their default branch; record
their commit hashes.

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_test_sbi_export.py` | the existing smoke tests | setup above | < 2 min |
| `python example_export.py --mode synthetic --out <dir>/sbi_demo_0000 --n_sims 16` | end-to-end run without data | setup above; `<dir>` outside the repository | < 1 min |

Both commands ran to completion with this setup in a cloud sandbox on
2026-10-06.

## 7. Not executable in the sandbox

- A real campaign export: the trained checkpoint and the campaign directories
  live on the cluster. Substitute: the demo backbone (or a checkpoint saved
  from it) and a small fabricated campaign directory with the npz schema of
  section 3.
- The real-recording arm (extractor/): out of coverage.
- A8 and A10: cross-file properties of a set of shards.
- GPU execution: everything runs on CPU here.

## 8. Open questions

- Q4 (raised 2026-10-06): batch-size invariance of Z: exact, or within float32
  rounding?
- Q5 (raised 2026-10-06): the decision log (D-061, D-063, D-064 are cited in
  commits) is not in this repository. Which decisions constrain the covered
  files and should be stated in section 4?

Resolved by the author on 2026-10-06:

- Q1: a spike at exactly t = K Delta_t is not part of the contract
  (section 2.1).
- Q2: rule 1 is ">= 1", decided robustly to floating-point rounding
  (section 2.2).
- Q3: count conservation is part of the contract (section 2.1).
