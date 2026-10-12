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
- `export_windows/` (added 2026-10-09 and 2026-10-10: the Giulia window bank,
  B5.1-B5.5 -- Block 6 the plan step, Block 7 the simulated bank's rows,
  shards, array and gate, with its launcher and job scripts, Block 8 the real
  bank)

Everything else in the repository is not specified yet and is out of scope for
the tester: the cohort and real-data modules, dataset profiling, the preflight
scripts, the extractor, raster and sim_reextract folders, and the shell
launchers outside `export_windows/`. The existing smoke test
smoke_test_sbi_export.py exercises the covered files of Blocks 1-5;
`export_windows/smoke_test_bank_plan.py` exercises Block 6 and
`export_windows/smoke_test_bank_export.py` Blocks 7 and 8. Blocks 6-8 call
`sim_reextract/sim_reextract_plan.py`, `cohort_manifest.py` and
`real_source.py` (all out of coverage) for their contract, seed, manifest and
real-trace helpers, and the Simulation-Based-Inference repository's
`hpc/joint/stage1/latent_bank.py` for the bank's shard format; they rely on
them as libraries and do not specify them.

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

### 1.1 The Giulia window bank (Block 6, added 2026-10-09)

A second consumer. The joint DSN+NPE stack of the Simulation-Based-Inference
repository (`hpc/joint/`), arm A1, trains on a Stage 1 bank: files
`shard_%04d.npz` with a JSON sidecar in that repository's `latent_bank`
contract, one row per window holding the window x itself (not an embedding z)
and its label theta. The bank is built from the Giulia project's re-detected
simulations (the record `mea_out_giulia_v2/REEXTRACTION_RECORD.json`, written
by `sim_reextract/`) and, in a later real mode, from the Giulia cohort's
archives. The steps are B5.1-B5.5 of the project's work plan. Block 6 is B5.1,
the plan step. It reads everything the bank will be built from, checks it,
estimates what the bank will cost, and writes one file,
`export_windows/<work>/bank_plan.json`, and a printed report. It writes no
shard and changes nothing it reads. Consumers: the user (go / stop), and the
bank's next steps, which read the plan.

Block 7 is B5.2-B5.4: from the plan, one PBS array element per task builds
every simulation's row (the window, its rate and the floor, the label in the
unit box), writes the task's kept rows as shards with their sidecars, and a
completion marker; a gate job then checks every task against the plan and
only then writes `BANK_RECORD.json`, the bank's record. Block 8 is B5.5: the
real bank, the Giulia cohort's 288 windows in one shard. Consumers: arm A1 of
`run_joint_arms.py` (glob of the shards), after B6 gives the runner a prior
for these banks (outside this repository: the runner today reads a
`latent_spec` of the bench's kind only).

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

### 2.5 The bank plan (Block 6)

Objects. Let Tasks be the set of tasks the record lists (each a campaign and
sweep pair). For each task t in Tasks:

- c(t), its label contract: the values of `JOB_ARGS_FIELDS` and
  `MANIFEST_FIELDS` of `sim_reextract_plan.py`, read by
  `sim_reextract_plan.task_contract` from the task's `job_args.json` and
  `manifest.json`; kappa(t) = `sim_reextract_plan.contract_signature(c(t))`.
- s(t), its seed values: `sim_reextract_plan.task_seed` (the `job_args.json`
  keys whose name contains "seed"; `seed_master` dropped where
  `_resolved_seed_master` is recorded).
- b(t) = [lo(t), hi(t)], natural units: the row of `param_bounds` at the index
  of `O_N` in `param_names`.
- Top(t): the topology indices k such that the folder `topo_<k>` of the task's
  output folder (the record's `out_dir`) holds at least one
  `mea_iter_<i>.npz`; for each k in Top(t), Iter(t, k) is the set of their
  iteration indices i and n(t, k) = |Iter(t, k)|.
- E(t): the topology indices k such that the output folder holds a folder
  `topo_<k>` with no `mea_iter_<i>.npz` -- topologies without simulations. For
  each k in E(t), the raw task's folder of the same name must exist and hold
  no `iter_<i>.npz` (`sim_reextract_plan.ITER_RE`): the topology is empty in
  the simulation itself, not at detection. A topology in E(t) gives no row and
  no donor.
- The record's `n_topos` counts every `topo_*` folder of the raw task, empty
  ones included (`sim_reextract_plan.scan_task`; the G2 gate requires an output
  folder for each), so for each t the record's `n_topos` must equal
  |Top(t)| + |E(t)| and its `n_iters` must equal sum_{k in Top(t)} n(t, k).
  [corrected 2026-10-12: this rule read "`n_topos` must equal |Top(t)|". The
  first plan on davinci refused `campaign_cadex_hhgap_v1/sweep_cfd_task0000`
  with "69 topologies and 66240 detection files ..., the record says 70 and
  66240": one topology folder of that task is empty in the raw task and in its
  detections.]

Contract rule (D-074). For every field f other than `param_bounds`, c_f(t)
takes one value over Tasks. For `param_bounds`, every row other than O_N's
takes one value over Tasks, and for each t, b(t) lies inside the declared box
B_ON = [0.01, 3.0] (D-074, D-078): 0.01 <= lo(t) and hi(t) <= 3.0.
`manifest_version` is reported, never compared. Any other difference refuses
the plan, naming the field (and the row) and the tasks.

Label columns. p = 23: the 22 axes of the single `active_indices`, in that
order, then the freeze's topology block `conn_prob` (D-016 as applied by G3).
Coordinates and boxes come from `sbi_labels.build_label_spec` on
`sbi_labels.registry_from_manifest(load_registry(sim_dir), manifest)`, with
`conn_prob_bounds` = (conn_prob_lo, conn_prob_hi) from `job_args.json`. The
declared box bounds_theta in R^{23 x 2} is that spec's box with the O_N row
replaced by [ln 0.01, ln 3.0] = [-4.605170, 1.098612] (D-074).

Twins and donors (D-070, D-076). Tasks are grouped by s(t). A group of two
tasks with different kappa is a twin pair. For each t and each k in Top(t),
the donor of (t, k) is the position, in sorted order, of the key
(canonical JSON of s(t), k) among the distinct keys of the record: for each
fixed k, the tasks of one seed group share the donor of topology k (D-076).
Every iteration of (t, k) carries that donor. D is the number of distinct
keys.

Twin check (reported, never refused; it does not change the donor rule). For
each twin pair (t, t') and each k in Top(t) intersect Top(t'), at the sampled
iteration: is `conn_prob` equal on both sides (float64 equality), and is
`seed_run` equal at the same iteration index.

Sample (D-080). For each donor key (seed group g, topology k), one iteration
index i* is drawn uniformly from the intersection of Iter(t, k) over the tasks
t of g that have k, with `numpy.random.default_rng(numpy.random.SeedSequence(
[sample_seed, crc32(key)]))`; when that intersection is empty, each task draws
from its own Iter(t, k) with the key (task name, k), and its seed_run
comparison is reported as not comparable. [proposed, Q6] For each sampled
(t, k, i*) the block reads `mea_iter_<i*>.npz` and the raw `iter_<i*>.npz` of
the same topology; their `conn_prob` and `seed_run` must agree.

The window and the floor (D-041, D-043, D-044, D-075). With T the single
`simtime` [s] of the contract, Delta_t and sigma_sm the frozen cohort
manifest's `w_size` and `gaussian_window` [s], and n_e the number of rows of
`electrode_centers` (must be 1, D-043):

    x  = build_pooled_ifr([det_t[det_ch == e] for e in 0..n_e-1], n_e, T, Delta_t, sigma_sm)
    K  = floor(T / Delta_t),  which must equal W = round(t_win_s / Delta_t) = 18000
    r  = (1 / K) * sum_{k=0}^{K-1} x[k] / Delta_t          [Hz], float64 sum of the float32 x
    kept  iff  r >= floor_hz * (1 - floor_rtol)            (floor_hz = 0.1 Hz, D-075)

with N the number of detections in [0, K Delta_t). By count conservation
(section 2.1), r = N / (K Delta_t) up to float32 rounding, so the floor reads
"at least 18 detected spikes in [0, 180) s"; floor_rtol = 1e-6 keeps a window
of exactly 18 spikes on the kept side despite that rounding. [proposed, Q7]

Estimates (D-080; printed and recorded, never acted on). Over the sampled
topologies, the kept share

    p_hat = sum_{(t,k)} n(t, k) * kept(t, k, i*) / sum_{(t,k)} n(t, k)

(each draw stands for its topology's n(t, k) rows), with a 95% percentile
bootstrap interval over donors: the sampled topologies are grouped by their
donor (the twins' matched topologies form one group, D-076), a resample draws
D groups with replacement (`numpy.random.default_rng(bootstrap_seed)`,
bootstrap_resamples resamples), and each resample's share is
sum n * kept / sum n over the topologies of the drawn groups, a group drawn
twice counted twice. [proposed, Q8; revised 2026-10-10 from "over
topologies": resampling topologies would treat the two sides of a twin pair
as independent draws] Bytes per kept row b_hat: the size of one
`numpy.savez_compressed` file holding the kept sampled windows as one float32
array of shape (m, W), divided by m; it counts x only (the other row arrays
add a few hundred bytes against 72,000 raw). Projected bank size:
p_hat * N_rows * b_hat, beside the raw N_rows * W * 4 B, with
N_rows = sum_t n_iters(t). Free space: `shutil.disk_usage` of the nearest
existing ancestor of bank_root, and the text printed by
`lfs quota -u <user> <mount>` where that command exists (timeout 30 s);
neither can refuse the plan.

Noise key. Every sampled simulation must carry seed_run >= 0: the bank's
realisation_id is the seed_run (section 2.6; D-072: the noise key from G2
on), and the sweep's paired mode writes -1. A sampled seed_run < 0 refuses
the plan, so that the array never meets it first.

The label in the unit box (D-074, D-078). For each simulation, theta_A in
R^p is assembled by `sbi_labels.assemble_theta_A` from the task's own label
spec (its manifest's boxes and coordinates, with the A6 check against the
file's params), and mapped into the unit box of the declared bounds: for each
column j in {0, ..., p - 1},

    u_j = (theta_A,j - lo_j) / (hi_j - lo_j),   [lo_j, hi_j] = row j of bounds_theta.

A u_j outside [-1e-12, 1 + 1e-12] (UNIT_TOL, float64 roundoff of a bound
written and re-read), or not finite, raises and names the column; within that
tolerance u_j is clipped to [0, 1] and counted as snapped. Since every task's
own O_N box lies inside the declared one, a correct label never raises; a v1
task's ln O_N lies in [ln 0.03, ln 3], i.e. u in [ln 3 / ln 300, 1] =
[0.19261, 1].

Shard layout (B5.3). For each task t, in record order, its simulations are
the pairs (k, i), k in Top(t) ascending and, for each k, i in Iter(t, k)
ascending; positions_sha256(t) is the sha256 of their canonical JSON
[[k, [i, ...]], ...]. With R = rows_per_shard, they are cut into
c(t) = max(1, ceil(n_iters(t) / R)) contiguous chunks whose sizes differ by
at most one (the first n_iters(t) mod c(t) chunks one larger), and the chunks
of all tasks are numbered 0, 1, ... in task order: the shard index. A chunk
becomes the file `shard_%04d.npz` (with `shard_%04d.json`) when the floor
keeps at least one of its rows, and no file otherwise.

### 2.6 The banks' rows (Blocks 7 and 8)

Simulated bank. For each task t (its index n_t in the record), each chunk of
t and each position (k, i) of the chunk, the row of section 2.5 (window x,
rate r, kept, u) is built from `<out_dir>/topo_<k>/mea_iter_<i>.npz`; the
kept rows, in position order, are the chunk's shard:

    x               (W,)  float32  the window, spikes per bin per electrode
    theta           (p,)  float32  u, the label in the unit box
    cls             -1             no class on the simulated arm
    nu              (5,)  zeros    no nuisance model on simulations
    realisation_id  uint64         the file's seed_run (its noise key, D-072)
    donor           int64          the donor of (t, k) (D-076)
    well            int64          n_t
    batch           int64          the campaign version v(t): the integer N of
                                   the campaign name's suffix _vN (1, 2, 5)
    subregion       0              one virtual electrode (D-043)
    window_idx      int64          i, the simulation's iteration index
    contaminated    False

Every id is bank-global, so concatenating shards needs no offsets. Within a
task, (donor, window_idx) is unique; across the two sides of a twin pair it
repeats (same donor, same iteration index: two simulations of one network,
D-070, D-076), and (well, donor, window_idx) is unique bank-wide.

The sidecar (`latent_bank.build_sidecar`), one contract for the whole bank:
param_names = the p label names; bounds_theta = the declared bounds (each
axis in its physical coordinate: ln for O_N, linear for the others);
coord = "unit_box" (theta holds u, not theta_A); fs = 1 / Delta_t;
w_size = Delta_t; T_win = 180 s; W; scale_convention = "per_electrode_mean";
latent_spec = {kind "campaign", the label names, physical coordinates and
units, the map theta = (theta_physical - lo) / (hi - lo), the prior "uniform
on the unit box (D-078)", O_N's axis and declared box, the topology axes,
n_windows_per_trace 1, T_win, fs, n_e, the floor and its tolerance, the
digests of the label freeze and of the cohort manifest};
nuisance_spec = {kind "none", the Stage 1 components}; realisation_spec =
{kind "simulator_seed_run"}; gap_spec = {kind "none"}; generator_sha256 = the
record's sha256; theta_withheld = false; arm = "S". The provenance (outside
the digest): the plan's sha256, the task, its index and its own O_N box
(D-078: a correction can be added later without re-exporting), the shard
index and positions, the counts, the versions, host and time.

One folder, one plan. A task refuses to write into a bank_root whose task
markers or BANK_RECORD.json carry another plan's digest, and refuses when a
shard file is present for a chunk of its own that keeps no row. A task whose
marker is complete for this plan (same digest, every file's sha256
unchanged) is not re-run.

The gate (B5.4) writes `BANK_RECORD.json` only when, for every task of the
plan: its marker is this plan's; its chunks are the plan's; read = n_iters
and kept + dropped = read; every listed file has the sha256 its marker
recorded (re-hashed) and the contract digest the plan implies; inside each
shard (the small arrays) rows = kept, well = n_t, batch = v(t), donors among
the task's, subregion 0, cls -1, window_idx >= 0, (donor, window_idx) unique
within the task; and no shard file exists that the plan does not name, nor
one for a chunk whose marker says it keeps no row.

Real bank (B5.5). For each well w of the frozen cohort manifest, in its order
(index n_w): its class c(w), the index in `classes` of the first component of
its `out_dir`; its root b(w), the index of the second component among the
sorted distinct second components; for each subregion s in
{0, ..., n_subsets_per_well - 1}, the trace y_s in R^K
(`real_source.read_real_trace`, checked against fs = 1 / Delta_t;
K = round(T_rec / Delta_t) when the manifest gives T_rec) is cut into the
windows [j W, (j + 1) W) for j in {0, ..., floor((K - W) / W)} (D-056; none
when K < W). Each window is one row: x = the slice, float32, unchanged;
theta with zero columns, shape (n, 0); cls = c(w); nu zeros;
realisation_id = donor = well = n_w (a culture is the unit never split);
batch = b(w); subregion = s; window_idx = j; contaminated False. Windows
whose mean rate is below the simulated bank's floor are kept and counted per
class (D-075, Implies (3)). The sidecar: param_names [], bounds_theta [],
coord "unit_box", the same window fields, latent_spec {kind
"campaign_real", the classes and roots, the floor of the simulated bank,
the manifest's digest}, realisation_spec {kind "recording"},
generator_sha256 = the cohort manifest's sha256, arm "R". On the Giulia
cohort (16 wells, D-055; 300 s and 600 s traces, D-040) the counts are 72,
108 and 108 windows, 288 in all (D-056). A folder that already holds
`shard_0000.npz` is refused: a written bank is never overwritten.

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
| record (Block 6) | `<out_root>/REEXTRACTION_RECORD.json` + `.sha256` (`<hex>  <name>`), by `sim_reextract_gate.py` | JSON | - | - | `tasks`: campaign, sweep, campaign_dir (the raw sweep folder), out_dir (its detections), n_topos (every `topo_*` folder of the raw task, empty ones included: `sim_reextract_plan.scan_task`), n_iters, contract_sig; `counts`; `replays`; `contract` |
| detections at the G2 tools (Block 6) | `<out_dir>/topo_<k>/mea_iter_<i>.npz`, ANN `d0ced88` `process_campaign.py:274-293` | det_t (n_det,); det_ch (n_det,); electrode_centers (n_e, 2); scalars conn_prob, seed_run, topo_idx, iter_idx | float32; int32; float32; float64, int64, int32, int32 | s; -; um; -, -, -, - | det_ch is 0-based; conn_prob is copied from the raw file (NaN when absent); the npz `simtime` is ceil(last spike), never T |
| raw iteration (Block 6) | `<campaign_dir>/topo_<k>/iter_<i>.npz`, ANN `HPC_main_sweep.py:1158-1173` | scalars conn_prob (flat rule), seed_run, topo_idx | float64, int64, int32 | - | seed_run is -1 in paired noise mode |
| label freeze (Block 6) | `artifacts/label_axes_giulia_hhgap.json` + `.sha256`, by `preflight_label_axes.py --out` | JSON | - | - | topology_axes must be ["conn_prob"] |
| frozen cohort manifest (Block 6) | `artifacts/cohort_manifest_giulia/cohort_manifest.json` + `.sha256`, read by `cohort_manifest.read_manifest` | JSON | - | s | `sim_preprocessing_from_manifest` gives Delta_t, sigma_sm, electrodes_per_subset |
| bank plan (Block 6) | `export_windows/<work>/bank_plan.json` + `.sha256` (gitignored, D-079) | JSON | - | - | see Block 6.5 |
| simulated shard (Block 7) | `<bank_root>/shard_%04d.npz`, `latent_bank.write_shard` | the row arrays of section 2.6, n rows | x float32, theta float32, cls int64, nu float32, realisation_id uint64, the other ids int64, contaminated bool | x spikes / bin / electrode; theta unit box | `latent_bank.ROW_ARRAYS`, a closed set |
| shard sidecar (Blocks 7, 8) | `<bank_root>/shard_%04d.json` | canonical JSON | - | fs Hz; w_size, T_win s | the contract fields of `latent_bank.CONTRACT_FIELDS`, `contract_digest`, `provenance` |
| task marker (Block 7) | `<bank_root>/tasks/task_%04d.json` | JSON | - | - | plan_sha256; per chunk: shard, start, stop, read, kept, dropped_floor, snapped, rate min / max, file and json with their sha256, contract_digest; totals; host, python, numpy |
| bank record (Block 7) | `<bank_root>/BANK_RECORD.json` + `.sha256`, by the gate | JSON | - | bytes | plan and its digest, the inputs' digests, the contract digest, the label, the window, counts, per task its totals and files, environment |
| real archive (Block 8) | `<extract_root>/<out_dir>/trace_subregion_%02d.npz`, read by `real_source.read_real_trace` | ifr_trace (K,) | float32 | spikes / bin / electrode | fs_ifr = 1 / Delta_t; T_rec s |
| real shard (Block 8) | `<real_bank_root>/shard_0000.npz` + `.json` | the row arrays of section 2.6; theta (n, 0) | as the simulated shard | as above | arm "R" |

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
- Binding decisions for Block 6, stated 2026-10-09 from the project's
  decision log (project knowledge, `claude/SBI_decisions_and_ideas_log.md`
  v1.54):
  - D-041: windows of 180 s; one window per simulation, untrimmed (T = 180 s,
    W = 18000 at Delta_t = 0.01 s).
  - D-043: one electrode per window on the simulated arm (n_e = 1).
  - D-044: Delta_t = 0.01 s, sigma_sm = 0.02 s, read from the cohort manifest.
  - D-046: every Giulia path carries `giulia` (`~/banks/giulia_sim/`, ...).
  - D-070: same-seed tasks across two label contracts are different
    simulations and the record keeps both (the 18 v1/v2 twin pairs).
  - D-074: all record tasks go into one bank, with O_N declared in
    [0.01, 3.0] in every shard; the tasks whose manifest carries
    [0.03, 3.0] lie inside it.
  - D-075: a row is kept only if mean_t x(t) >= 0.1 Hz.
  - D-076: donor = one network: one id per (task seed values, topology
    index), the twins' same-index topologies merged.
  - D-078: the prior is the declared box; no mixture correction.
  - D-079: B5.1 is committed code in `export_windows/`, its plan file
    gitignored beside it.
  - D-080: B5.1 also estimates the floor's kept share, the bytes per row and
    the free space, as estimates only.
- Further decisions for Blocks 7 and 8, stated 2026-10-10 from the same log:
  - D-006: scheduler behaviour is measured before it is relied on; on
    davinci `depend=afterok` on an array is ordering only, so the gate job
    is the success signal.
  - D-040, D-055: the Giulia real cohort, 16 wells in three classes, traces
    of 300 s (0:100) and 600 s (the AraC classes).
  - D-056: real windows of 180 s, back to back from t = 0 (72, 108 and 108
    windows).
  - D-072: from G2 on, each simulation's recording noise is keyed on its own
    seed_run.
  - D-075, Implies (3): a real window below the floor is outside the training
    support of x; the real mode counts it (no floor on the real arm).

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

### Block 6: bank plan (B5.1 of the Giulia project), in six sub-blocks

Section 2.5 states the mathematics; this entry states the interfaces. Every
sub-block of Blocks 6-8 is pure ASCII, LF only, and runs in the `sbi_export`
environment (python 3.11 on the cluster): no torch, no matplotlib beyond what
`sim_observable` pulls in through the DSN tree.

#### Block 6.1: configuration

- Module: `export_windows/bank_plan_config.py`; profile
  `export_windows/profiles/giulia.json`
- Public API: `BankPlanConfig` (frozen dataclass);
  `load_profile(name_or_path, repo_root=None) -> BankPlanConfig`;
  `ConfigError`
- Inputs: a profile name (a file `profiles/<name>.json`) or a path to a JSON
  file; keys beginning with `_` are comments and are ignored.
- Outputs: a BankPlanConfig whose relative paths are made absolute against
  the repository root (the parent of `export_windows/`).
- Parameters (the Giulia profile's values): `record_path` and
  `record_sha256_prefix` = 7af3759a0f3c1350 (G2's record); `freeze_path` =
  artifacts/label_axes_giulia_hhgap.json and `freeze_sha256_prefix` =
  29e38c89b08a2dd8 (G3); `cohort_manifest_path` =
  artifacts/cohort_manifest_giulia/cohort_manifest.json and
  `cohort_manifest_sha256_prefix` = 21b7ef03bb7f776b (G1, frozen by G2d);
  `sim_dir`, the simulator source holding the hhgap registry (37 axes);
  `on_axis` = "O_N"; `on_declared_box` = (0.01, 3.0) natural units (D-074);
  `topology_axes` = ("conn_prob",); `expected_p` = 23 (G3: 22 active axes
  and conn_prob); `t_win_s` = 180.0 s (D-041); `n_e` = 1 (D-043);
  `floor_hz` = 0.1 Hz (D-075); `floor_rtol` = 1e-6 (Q7); `sample_seed`,
  `bootstrap_resamples` = 2000, `bootstrap_seed` (Q6, Q8);
  `bytes_sample_rows` = 500 (D-080; Q12); `rows_per_shard` = 10000 (B5.3;
  Q12); `bank_root` = /davinci-1/home/ldellamea/banks/giulia_sim and
  `real_bank_root` = /davinci-1/home/ldellamea/banks/giulia_real (D-046);
  `plan_out` = export_windows/giulia/bank_plan.json (D-079).
- Errors: ConfigError for an unknown or missing key; a digest prefix that is
  not 16 lowercase hexadecimal digits; a declared box without
  0 < lo < hi; an empty topology_axes; expected_p not above the topology
  block's width; floor_hz <= 0; floor_rtol outside [0, 1e-3]; n_e < 1;
  t_win_s <= 0; bootstrap_resamples < 100; bytes_sample_rows < 1;
  rows_per_shard outside [100, 100000]; a seed that is not a non-negative
  integer; an integer key given as a float or a bool. NaN fails every one of
  these comparisons.
- Library calls relied on: json, dataclasses.
- Custom code: validation only.
- Test oracles: the Giulia profile loads, and each decided value equals its
  decision (D-041, D-043, D-046, D-074, D-075, D-079), asserted by value;
  each invalid input above raises ConfigError; a relative path resolves
  against the repository root.
- Data flow: feeds Blocks 6.2-8.
- Confirmed: the decided values, yes (2026-10-09, D-041..D-080); the
  proposed values, no (Q6-Q8, Q12)
- Status: tested (smoke suite, 2026-10-10)

#### Block 6.2: inputs (loading and validation)

- Module: `export_windows/bank_plan_load.py`
- Public API: `read_verified_json(path, expected_prefix, what) -> (doc, sha256_hex)`;
  `load_inputs(cfg) -> Inputs`; `scan_detection_tree(out_dir) -> ({k: [i, ...]},
  {k: folder name})` (Top(t) with Iter(t, k), and E(t)); `scan_detections(out_dir)
  -> {k: [i, ...]}` (its first half);
  `read_detection(path) -> dict`; `read_raw_ident(path) -> dict`;
  `TaskInputs`, `Inputs` (frozen dataclasses); `LoadError`
- Inputs: the configuration (6.1); the files of section 3 marked Block 6.
- Outputs: `Inputs` with the record and its digest; the freeze and its
  digest; Delta_t, sigma_sm and electrodes_per_subset with the manifest's
  digest; and, for each task of the record in record order, a `TaskInputs`:
  its name `<campaign>/<sweep>`, campaign_dir, out_dir, c(t) (whose
  param_names, param_bounds and simtime the later blocks read), kappa(t),
  s(t), Top(t) with Iter(t, k) (sorted), E(t) (`empty_topos`, sorted),
  n_topos = |Top(t)| + |E(t)| and n_iters.
  `read_detection` returns det_t float64 (n_det,) [s], det_ch int64
  (n_det,), n_e int (rows of electrode_centers), conn_prob float, seed_run
  int, topo_idx int, iter_idx int, theta float64 (37,) (inference
  coordinates, registry order) and params float64 (37,) (natural units).
  `read_raw_ident` returns conn_prob float, seed_run int, topo_idx int.
- Errors (LoadError, naming the file): a missing file or a missing
  `.sha256`; a digest that differs from its `.sha256` or does not begin with
  the profile's prefix; a record without `tasks`; a task whose recomputed
  kappa(t) differs from the record's `contract_sig`, whose detections folder
  is missing, or whose |Top(t)| + |E(t)| or sum_k n(t, k) differs from the
  record's n_topos or n_iters; a topology of E(t) whose raw folder is missing
  or holds `iter_<i>.npz` files; two folders giving one topology; a freeze whose
  topology_axes differ from the profile's; a cohort manifest refused by
  `cohort_manifest.read_manifest`; a detection or raw file without a required
  key, with det_t and det_ch of different shapes, with a non-finite det_t, or
  with a non-finite conn_prob (the G2 tools write NaN when the raw file has
  none); electrode_centers that is not (n_e, 2); a file name whose index
  differs from the file's own iter_idx or topo_idx. `load_inputs` checks
  every task of the record, so that one run shows every bad task (amended
  2026-10-12): one refused task raises its own LoadError; two or more raise
  one LoadError whose first line counts them (`<n> of <N> tasks refused`) and
  whose further lines are their errors, one per line, at most 40 of them, then
  `... and <m> more`.
- Library calls relied on: hashlib; json; numpy.load(allow_pickle=False);
  `sim_reextract_plan.task_contract`, `task_seed`, `contract_signature`;
  `cohort_manifest.read_manifest`, `sim_preprocessing_from_manifest`.
- Custom code: the directory scan (two regular expressions), because nothing
  in the repository lists detections per topology with their indices.
- Test oracles:
  - Known answer: on the smoke fixture, Inputs reproduces the fixture's
    tasks, seeds, O_N rows and Iter(t, k) exactly; read_detection returns
    the fixture's theta and params.
  - Reference: each returned digest equals hashlib.sha256 of the file's
    bytes, computed independently in the test.
  - Invariants: for each task, sum_k n(t, k) = n_iters and
    |Top(t)| + |E(t)| = n_topos (E(t) empty on the default fixture).
  - Known answer, a topology without simulations (amended 2026-10-12): with
    one empty `topo_<k>` folder added to a task in the raw tree and in the
    detections (the record counting it), Inputs equals the default fixture's
    but for that task's empty_topos (k) and n_topos (+1); a file other than an
    iteration in either empty folder changes nothing.
  - Contract: read_detection's dtypes and shapes as listed; n_e = 1 on the
    fixture.
  - Refusals, each a LoadError naming the file: one byte changed in the
    record; a freeze or a manifest of another digest; a missing sidecar;
    topology_axes ["conn_prob", "p0_conn"]; a record without tasks; a
    detection file without conn_prob, or with conn_prob NaN, or with det_t
    NaN, or with det_t and det_ch of different lengths; a file name whose
    index differs from its iter_idx; a task whose n_iters disagrees with its
    folders; a task whose job_args.json changed after the record; on the
    fixture with an empty topology, an iteration file added to its raw
    folder, its detections folder removed, and an empty detections folder with
    no raw counterpart (the record counting it), and an empty topology in both
    trees that the record does not count; and two refused tasks, both named in
    one LoadError, the second replaced by `... and 1 more` when the cap is set
    to one (amended 2026-10-12).
- Data flow: feeds Blocks 6.3-7.
- Confirmed: no (drafted 2026-10-09 from the B5 plan v1.1, sec. 3, row B5.1)
- Status: tested

#### Block 6.3: contract, label columns, twins, donors and the shard layout

- Module: `export_windows/bank_plan_model.py`
- Public API: `check_contract(tasks, cfg) -> dict` ("single": one value per
  field, the O_N row of param_bounds removed; "on_index"; "on_boxes":
  [{"box", "n_tasks", "tasks"}]; "manifest_version": {value: count});
  `task_label_spec(task, registry, cfg, freeze) -> sbi_labels.LabelSpec`;
  `label_columns(tasks, registry, cfg, freeze) -> LabelColumns`
  (param_names, coord, units, bounds_declared (p, 2) float64, on_column,
  active_indices); `flipped_axes(tasks, registry) -> {task: [axis, ...]}`;
  `seed_key(seeds) -> str`; `seed_groups(tasks) -> [{"seed_key", "tasks",
  "contract_sigs", "twin"}]`; `donor_map(tasks) -> {"n_donors", "by_task":
  {task: {k: donor}}}`; `shard_layout(tasks, rows_per_shard) -> {"n_shards",
  "by_task": {task: [{"shard", "start", "stop"}]}}`;
  `positions_sha256(topo_iters) -> str`; `sorted_positions(task) -> [(k, i)]`;
  `PlanRefused`
- Behaviour: section 2.5 (contract rule, label columns, twins and donors,
  shard layout). Pure functions of the loaded inputs and the registry: no
  file access. The donors are numbered by the sorted keys (seed key, k). A
  coordinate that a task's manifest flips against the simulator's source is
  reported (flipped_axes), never refused: label_columns refuses only when
  the tasks disagree with one another.
- Errors (PlanRefused, naming the field, row and tasks): a contract
  difference other than O_N's box; an O_N box outside B_ON; simtime !=
  t_win_s; param_names without O_N; label names or coordinates differing
  between tasks; p != expected_p; a topology block other than the profile's;
  O_N not "ln" in the label; label bounds outside O_N's differing between
  tasks; a degenerate declared box; two tasks with the same seed values and
  the same contract signature (a replay the record should have excluded,
  D-061, D-070).
- Library calls relied on: `sbi_labels.registry_from_manifest`,
  `build_label_spec`; `sim_reextract_plan.JOB_ARGS_FIELDS`, `MANIFEST_FIELDS`
  and its value normalisation; hashlib; json.
- Custom code: the contract comparison, the grouping and the layout
  (bookkeeping, no numerics).
- Test oracles: the fixture's contract passes with on_boxes [0.01, 3.0] x 2
  and [0.03, 3.0] x 1; one task with simtime 200.0 refuses naming `simtime`;
  simtime 200.0 in every task refuses as not the window; a box [0.005, 3.0]
  refuses naming O_N and D-074; another param_bounds row differing refuses;
  the label has p = 23 with O_N in column 21, its row
  [ln 0.01, ln 3.0] (rtol 1e-12, float64 roundoff) and every other row the
  fixture's box; expected_p 24 refuses; O_N linear in every task refuses;
  2 seed groups, 1 twin pair; D = 6, the count of distinct (seed values, k)
  keys by hand, the twins sharing each k's donor; a replay refuses; the layout
  of n_iters (250, 100, 1) at R = 100 is chunks (84, 83, 83), (100), (1)
  numbered 0-4 and covering each task's positions once (brute force);
  positions_sha256 equals an independent sha256 and changes when one index
  changes.
- Data flow: consumes 6.2; feeds 6.4, 6.5 and 7.
- Confirmed: the rules of D-074 and D-076, yes; the donor numbering and the
  layout, no (Q12)
- Status: tested

#### Block 6.4a: one simulation's row

- Module: `export_windows/bank_row.py`
- Public API: `window_from_detections(det_t, det_ch, n_e, T, dt, sigma_sm) -> ndarray`
  ((K,) float32); `mean_rate_hz(x, dt) -> float`;
  `is_kept(rate_hz, floor_hz, floor_rtol) -> bool`;
  `to_unit_box(theta_a, bounds) -> (u, snapped)` (u (p,) float64);
  `build_row(det, spec_task, bounds, T, dt, sigma_sm, n_e, W, floor_hz, floor_rtol) -> dict`
  (x, rate_hz, kept, u, theta_a, snapped, n_det); `RowError`;
  `UNIT_TOL` = 1e-12
- Behaviour: section 2.5 (the window and the floor; the label in the unit
  box). Shared by the plan's sample (6.4) and the export (7.1), so that the
  estimate and the bank are computed by one piece of code.
- Errors (RowError): det_ch outside [0, n_e); the file's n_e != the bank's
  (D-043); K != W (D-041); a label sbi_labels refuses (A6 included); a u
  outside the box beyond UNIT_TOL, or not finite.
- Library calls relied on: `sim_observable.build_pooled_ifr`;
  `sbi_labels.assemble_theta_A`; numpy.
- Test oracles: r of a window of N spikes in [0.5, 179.5] s equals N / 180 Hz
  for N in {17, 18, 19, 500} (rtol 1e-6: float32 storage of x, float64 sum)
  and 0 for N = 0; build_row drops 17 spikes and keeps 18; is_kept keeps
  r = 0.1 (1 - 1e-7) at floor_rtol 1e-6 and drops it at 0; to_unit_box gives
  hand-computed values, snaps a value 1e-13 below lo to 0 and counts it, and
  raises 1e-6 below lo and on NaN; on a fixture file, u equals
  (theta_A - lo) / (hi - lo) computed in the test from the file's theta (the
  active axes' entries, then conn_prob) and the declared bounds (rtol 1e-12);
  n_e 2, T 170 s and a det_ch of 1 raise.
- Data flow: consumed by 6.4 and 7.1.
- Confirmed: no (Q7: the tolerance)
- Status: tested

#### Block 6.4: the per-topology sample and the estimates

- Module: `export_windows/bank_plan_sample.py`
- Public API: `draw_iterations(tasks, groups, cfg) -> {(task, k): {"i", "comparable"}}`;
  `sample_topology(task, k, i, inputs, cfg, spec_task, bounds) -> dict`
  (task, k, i, n, conn_prob, seed_run, rate_hz, kept, n_det, u_min, u_max,
  snapped, x); `twin_check(groups, samples) -> [{"tasks", "shared_topologies",
  "conn_prob_equal", "seed_run_comparable", "seed_run_equal"}]`;
  `donor_totals(samples, donors_of) -> (donor ids, num, den)` (per donor,
  sum n * kept and sum n over its sampled topologies);
  `bootstrap_reps(num, den, resamples, seed) -> ndarray` (the resampled
  shares); `kept_share(samples, donors_of, cfg) -> dict` (p_hat, ci95,
  unweighted, n_topologies, n_donors, resamples, rate_hz_quantiles);
  `bytes_per_row(windows) -> float`; `free_space(bank_root) -> dict`
  (path_checked, free_bytes, total_bytes, lfs_quota)
- Behaviour: section 2.5 (sample, window and floor, estimates).
- Errors: LoadError when a sampled detection file and its raw file disagree
  on topo_idx, conn_prob or seed_run; RowError from 6.4a.
- Library calls relied on: numpy.random.default_rng,
  numpy.random.SeedSequence, zlib.crc32, numpy.savez_compressed,
  numpy.percentile, numpy.bincount, shutil.disk_usage, subprocess (lfs).
- Test oracles: the same seed gives the same draws, each equal to the draw
  recomputed in the test from SeedSequence([sample_seed, crc32(key)]); the
  twins get one i on their topologies 0 and 1 (common indices) and their own
  draws on topology 2 (disjoint), comparable False; on the fixture p_hat
  equals the weighted share computed by hand from the fixture's spike counts
  (kept iff N >= 18) and lies in its interval; a two-donor case (one donor
  with two kept topologies, one with one dropped, unit weights) has
  donor_totals num (2, 0), den (2, 1) and resample values in {0, 2/3, 1}
  only, which a bootstrap over topologies would not give; the twin check gives 3 shared topologies, conn_prob equal on 1 and
  seed_run equal on 2 of 2 comparable; bytes_per_row is below 1,000 for
  all-zero windows and above 60,000 for uniform noise (raw 72,000); a raw
  file with another seed_run refuses.
- Data flow: consumes 6.2-6.4a; feeds 6.5.
- Confirmed: no (Q6, Q8)
- Status: tested

#### Block 6.5: entry point and the plan file

- Module: `export_windows/bank_plan.py`; command
  `python3 export_windows/bank_plan.py --profile giulia [--out PATH]`
- Outputs: `bank_plan.json` and its `.sha256`, written atomically and only on
  OK. Keys: plan "giulia_window_bank"; plan_version 1; created; git (the
  checkout's short hash); profile (every configuration value); inputs (the
  three paths with their sha256, sim_dir); window (T, dt, sigma_sm, W, fs,
  n_e, floor_hz, floor_rtol); contract (6.3); label (param_names, coord,
  units, bounds_declared, on_axis, on_column, active_indices, topology_axes,
  excluded_axes, flipped); tasks (per task: index, name, campaign, sweep,
  campaign_dir, out_dir, contract_sig, seeds, on_box, batch, n_topos,
  n_iters, topo_counts {k: n}, positions_sha256, empty_topos [k, ...] (E(t);
  amended 2026-10-12), donors {k: donor}, shards [{shard, start, stop}]);
  seed_groups; n_donors; n_shards; n_rows; sample
  (kept_share, bytes_per_row_from, snapped_total, twin_check, rows
  [task, k, i, rate, kept, comparable]); estimates (n_rows, raw_bytes,
  kept_rows and its interval, bytes_per_kept_row, kept_bytes); free_space.
  A printout whose last line is `BANK PLAN OK` (exit 0) or begins `REFUSED:`
  (exit 1); exit 2 on a usage error. The printout's `donors` line counts the
  topologies with simulations, sum_t |Top(t)|; when some E(t) is not empty, a
  `topologies` line gives sum_t |E(t)| and one line per empty topology (task
  and k) follows, at most 10 of them, then their count. A refusal of two or
  more tasks (6.2) prints one `[plan] refused` line per task, then
  `REFUSED: <n> of <N> tasks refused (each named above)` (amended 2026-10-12).
- Errors (REFUSED): every error of 6.1-6.4a; electrodes_per_subset != n_e
  (D-043); t_win_s not a whole number of Delta_t bins; a campaign name
  without the suffix _vN; a record whose tasks hold no detection file; a
  sampled seed_run < 0 (section 2.5).
- Test oracles: on the fixture, OK with n_rows 32, n_donors 6, n_shards 3
  (R = 100), batches (1, 2, 5), 9 sampled topologies, the inputs' digests
  equal to the files' sha256, and the plan equal to its `.sha256`;
  expected_p 24, n_e 2 (named by the manifest check, not only by the row's),
  t_win_s 175, a cohort manifest with electrodes_per_subset 9, and a sampled
  seed_run of -1 each print REFUSED, exit 1 and write no plan; two runs give
  the same sample and estimates; with one empty topology added to a task, OK
  with the default fixture's n_rows, n_donors, n_shards, positions_sha256 and
  sample, that task's empty_topos [k] and n_topos + 1, and the printout naming
  the topology; eleven empty topologies print ten such lines, then
  `... and 1 more`; two refused tasks print one `[plan] refused` line each and
  the REFUSED line last (amended 2026-10-12).
- Data flow: wires 6.1-6.4; feeds Block 7.
- Confirmed: no
- Status: tested

### Block 7: the simulated bank (B5.2-B5.4), in four sub-blocks

Section 2.6 states the rows, the sidecar and the gate's conditions.

#### Block 7.1: rows and shards of one task

- Module: `export_windows/bank_export.py`
- Public API: `export_task(plan, index, bank_root, log=print) -> dict` (the
  marker); `load_plan(path) -> dict` (the plan after its bytes match its
  `.sha256`, with `_sha256` and `_path` added); `contract_kwargs(plan,
  nu_components) -> dict` (the sidecar's contract fields);
  `marker_path(bank_root, index)`; `marker_is_complete(plan, index,
  bank_root) -> bool`; `other_plans(plan, bank_root) -> [digest, ...]`;
  `task_inputs(plan, index) -> TaskInputs`; `stage1_modules() ->
  (latent_bank, latent_nuisance)`; `sha256_file(path)`; `ExportError`
- Inputs: the plan; the task's files (section 3); the SBI repository's
  `hpc/joint/stage1` (found through SBI_HPC_DIR) for `latent_bank` and
  `latent_nuisance.NU_COMPONENTS`.
- Outputs: the task's shards in bank_root and its marker
  `tasks/task_%04d.json` (sections 2.6 and 3), the marker written last and
  atomically.
- Behaviour: section 2.6. Before writing: no other plan's outputs in
  bank_root; the record, the freeze and the manifest re-hashed against the
  plan; the task's detections re-scanned (6.2's checks) and compared with the
  plan's topo_counts, positions_sha256 and empty_topos (E(t), amended
  2026-10-12); the task's label names compared with the plan's. Only the
  floor drops a row.
- Errors (ExportError, naming the file): any of the above differing; a
  seed_run < 0; a RowError or LoadError of any simulation; a stale shard file
  for a chunk that keeps no row; read != n_iters.
- Library calls relied on: `latent_bank.build_sidecar`, `write_shard`; Blocks
  6.2-6.4a.
- Test oracles (the fixture's plan, R = 100): per task, kept = the number of
  its simulations with N >= 18 spikes and dropped the rest (hand oracle from
  the fixture's counts), read = n_iters; in each shard, well, batch, donor,
  window_idx and realisation_id equal the fixture's task index, campaign
  version, the plan's donor of k, i and seed_run; theta equals u recomputed in
  the test from the fixture's theta, conn_prob and the declared bounds
  (atol 1e-7, float32 storage); each row's x sums to its N spikes (rtol 1e-6);
  every sidecar carries the contract digest recomputed in the test from the
  plan, coord "unit_box", arm "S", and its task's on_box in the provenance;
  `latent_bank.concat_shards` reads the bank back (rows = the kept total,
  theta in [0, 1]); a task with an empty topology exports the rows of the
  default fixture's task, and an empty topology renamed after the plan (the
  counts unchanged) is refused (amended 2026-10-12).
- Data flow: consumes the plan (6.5); feeds 7.2 and 7.3.
- Confirmed: no (Q10, Q11, Q14)
- Status: tested

#### Block 7.2: the array element

- Module: `export_windows/bank_task.py`; command
  `python3 export_windows/bank_task.py --plan PLAN --task-index N [--bank-root DIR]`
- Behaviour: N must be a task of the plan; a task whose marker is complete
  for this plan prints `already complete for this plan: nothing to do`;
  otherwise export_task. bank_root defaults to the plan's profile bank_root.
- Outputs: last line `TASK <N> OK` (exit 0) or `FAILED: <reason>` (exit 1).
- Test oracles: every fixture task OK; a second run of one task says
  "already complete"; an index out of range FAILS; a bank_root holding
  another plan's marker FAILS naming "another plan"; a stale shard of a chunk
  that keeps no row FAILS.
- Confirmed: no
- Status: tested

#### Block 7.3: the gate and the bank record

- Module: `export_windows/bank_gate.py`; command
  `python3 export_windows/bank_gate.py --plan PLAN [--bank-root DIR] [--workers N]`
- Behaviour: section 2.6 (the gate); the shard and sidecar files are hashed
  in a pool of N workers; of each shard only the small arrays are read.
- Outputs: on success `<bank_root>/BANK_RECORD.json` and `.sha256`, last line
  `wrote <bank_root>/BANK_RECORD.json  sha256 <16 hex>` (exit 0); otherwise
  every problem as a `[gate] FAIL` line and the last line
  `REFUSED: <n> problem(s), nothing written` (exit 1).
- Test oracles: on the fixture's bank, the record's counts equal the markers'
  sums (24 of 32 rows kept at the fixture's spike counts) and its contract
  digest the recomputed one; each of these refuses and writes nothing: a
  missing marker, a shard changed by one byte, a stray shard_0099.npz, a
  marker of another plan, a shard whose well array was rewritten with its
  marker's hashes updated, a shard file for a chunk whose marker keeps no row.
- Confirmed: no
- Status: tested

#### Block 7.4: launcher and job scripts

- Files: `export_windows/launch_bank.sh`; `export_windows/run_bank_task.pbs`;
  `export_windows/run_bank_gate.pbs`
- Behaviour: `launch_bank.sh` modes `plan` (runs bank_plan.py here), `test`
  (one task as a job, TEST_INDEX), `array` (one element per task,
  `-J 0-(n-1)%CONCURRENCY`, a plain job when the plan has one task, then the
  gate with `-W depend=afterok:<array>`), `gate` (the gate alone), `real`
  (runs bank_real.py here); DRYRUN=1 prints every command and submits
  nothing; QSUB substitutes the submit command; WALLTIME is required for
  `array`, as hh:mm:ss or `auto`: twice the largest task's simulations times
  the test task's seconds per simulation (from its marker, which must be this
  plan's), plus 10 min, rounded up to a quarter hour, at least 30 min.
  Every job writes its log to an absolute path under
  `export_windows/<work>/logs/`. The job scripts activate ENV_NAME (default
  sbi_export) with the environment block of
  `sim_reextract/run_sim_reextract_gate.pbs` (-u armed only after the
  activation), source `../env.sh`, and print their environment before the
  work.
- Test oracles: with a fake qsub, `array` submits one array of the plan's
  task count and a gate that depends on it, `test` one job with its index,
  and DRYRUN=1 submits nothing; `array` refuses without WALLTIME, with
  WALLTIME=auto before any test task, and with a WALLTIME that is not
  hh:mm:ss; WALLTIME=auto after the fixture's tasks gives 00:30:00 (the
  floor); the job scripts, run directly with a fake conda, end in
  `TASK <N> OK` and `wrote ... BANK_RECORD.json`, refuse another
  environment and another submission folder; every script passes `bash -n`
  and holds no CR byte.
- Confirmed: no
- Status: tested in the sandbox (fake qsub, fake conda)

### Block 8: the real bank (B5.5)

- Module: `export_windows/bank_real.py`; command
  `python3 export_windows/bank_real.py --profile giulia [--out-root DIR]`
- Public API: `build_real(cfg) -> (arrays, info)`;
  `windows_of(K, W) -> [start, ...]`; `RealRefused`
- Inputs: the frozen cohort manifest (digest prefix checked,
  electrodes_per_subset = n_e) and its wells' archives.
- Outputs: `<real_bank_root>/shard_0000.npz` and `.json` (section 2.6); per
  class the wells, windows and windows below the floor; last line
  `REAL BANK OK` (exit 0) or `REFUSED: <reason>` (exit 1).
- Errors (REFUSED): a manifest without `.sha256` or of another digest;
  electrodes_per_subset != n_e; a well whose folder names no class; a missing
  archive; K differing from T_rec / Delta_t; no window at all; an existing
  shard_0000.npz.
- Library calls relied on: `cohort_manifest.read_manifest`,
  `sim_preprocessing_from_manifest`; `real_source.read_real_trace`;
  `latent_bank.build_sidecar`, `write_shard`.
- Test oracles: windows_of(30000, 18000) = [0], (36000) = [0, 18000],
  (60000) = [0, 18000, 36000], (17999) = []; on the fixture (two 300 s wells
  of class 0, one 600 s well of each other class, 9 subregions each) 18, 27
  and 27 windows, 9 below the floor (the 0.05 Hz well); each row's x equals
  the archive's slice [j W, (j + 1) W) exactly; cls, donor, well, batch (the
  sorted roots), subregion and window_idx as section 2.6; the sidecar's arm
  "R", theta (n, 0), generator_sha256 the manifest's digest; an existing
  shard, a missing archive and another manifest digest refuse.
- Data flow: consumes the frozen manifest; feeds arm A1's REAL_SHARDS.
- Confirmed: no (Q13)
- Status: tested

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

Block 6 (added 2026-10-09; the same setup, plus `SBI_HPC_DIR` for
`cohort_manifest.py`'s DSN-tree imports):

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python3 export_windows/smoke_test_bank_plan.py` | Block 6's smoke tests, on a fixture written to a temporary folder (its own stand-in registry) | setup above | < 1 min |

Blocks 7 and 8 (added 2026-10-10; the same setup; `SBI_HPC_DIR` must hold
`joint/stage1/latent_bank.py`, Simulation-Based-Inference at `9fcbc88` or
later):

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python3 export_windows/smoke_test_bank_export.py` | Blocks 7 and 8: the export, the array element, the gate, the launcher (fake qsub and conda) and the real bank, on the fixture | setup above; bash | < 2 min |
| `python3 export_windows/bank_plan.py --profile <json>` | the plan step | a profile from `bank_plan_fixture.write_profile` | < 10 s on the fixture |
| `python3 export_windows/bank_task.py --plan <plan> --task-index N` | one task | after the plan | < 10 s per fixture task |
| `python3 export_windows/bank_gate.py --plan <plan>` | the gate and the record | after every task | < 10 s on the fixture |
| `python3 export_windows/bank_real.py --profile <json>` | the real bank | the fixture's profile | < 10 s |

On davinci the sequence is `launch_bank.sh plan`, `test`, `array`, then the
gate's line, then `real` (the launcher's header).

## 7. Not executable in the sandbox

- A real campaign export: the trained checkpoint and the campaign directories
  live on the cluster. Substitute: the demo backbone (or a checkpoint saved
  from it) and a small fabricated campaign directory with the npz schema of
  section 3.
- The real-recording arm (extractor/): out of coverage.
- A8 and A10: cross-file properties of a set of shards.
- GPU execution: everything runs on CPU here.
- Blocks 6-8 on the real record and cohort: the record, the detections
  (about 1.26 million files), the freeze, the frozen manifest and the real
  archives live on the cluster. Substitute: the smoke fixture (three tasks
  in two O_N boxes, one twin pair with a shared, an unshared and a disjoint
  topology; four real wells in three classes), with a stand-in registry of
  the hhgap registry's shape.
- The PBS submission (Block 7.4): run here with a fake qsub that records its
  arguments and a fake conda; the scheduler itself, the queue's limits and
  the walltime per task are measured on davinci by the launcher's `test`
  mode.
- Training on the banks: `run_joint_arms.py --dry-run` reads the fixture's
  banks (3 shards, 24 rows, W 18000, d_theta 23; 72 real rows); a training
  element stops where the runner builds its prior from the bench's
  latent_spec (`run_joint_arms.py:390` at SBI `9fcbc88`), until B6.

## 8. Open questions

- Q4 (raised 2026-10-06): batch-size invariance of Z: exact, or within float32
  rounding?
- Q5 (raised 2026-10-06): the decision log (D-061, D-063, D-064 are cited in
  commits) is not in this repository. Which decisions constrain the covered
  files and should be stated in section 4?

Raised 2026-10-09 for Block 6, each with Claude's proposal; open until the
user confirms:

- Q6: which simulation per topology is sampled. Proposed: one iteration
  drawn uniformly, seeded per donor key, so that both sides of a twin pair
  are read at the same index (section 2.5). Alternative: the lowest index.
- Q7: the floor at exactly 18 spikes. Proposed: compare r with
  floor_hz * (1 - 1e-6), so that float32 rounding cannot drop a window of
  18 spikes; equivalently, at least 18 detections in [0, 180) s. The same
  rule then binds B5.2's filter.
- Q8: the estimator of the kept share. Proposed: the row-weighted share
  p_hat of section 2.5 with a 95% percentile bootstrap interval over donors
  (2000 resamples; revised 2026-10-10 from "over topologies", so that the two
  sides of a twin pair are not resampled as independent draws). Alternative:
  the unweighted share of topologies.
- Q9 (an observation for the triage of Blocks 1 and 5): section 3's
  detection row predates the G2 tools. At ANN `d0ced88` a `mea_iter` file
  stores det_t as float32 and det_ch as int32, and carries conn_prob,
  iter_idx, noise_entropy and the G2 noise scheme in meta_json
  (`process_campaign.py:274-293`). Block 6 states those types; Blocks 1 and
  5 are unchanged until reviewed.

Raised 2026-10-10 for Blocks 6-8, each with Claude's proposal (implemented
as proposed; open until the user confirms):

- Q10: the simulated rows' id fields. Proposed (section 2.6): well = the
  task's index in the record; batch = the campaign version; realisation_id =
  seed_run; window_idx = the iteration index; cls = -1; nu = zeros;
  subregion 0. Only donor is decided (D-076); the runner reads donor (the
  split) and, on the real arm, cls and donor (the replicate pairs).
- Q11: theta is stored in the unit box, coord "unit_box", with bounds_theta
  the declared box in each axis's physical coordinate, so that a prior on
  the unit box is D-078's box. B6 reads it; the alternative is theta in the
  physical coordinate with coord "physical" (the plan's bounds unchanged).
- Q12: rows_per_shard = 10000 (at most 0.72 GB of float32 windows in one
  task's memory, before compression), bytes_sample_rows = 500, and the donor
  numbering by sorted (seed key, topology) keys.
- Q13: the real bank's id fields. Proposed: donor = well = realisation_id =
  the well's index in the frozen manifest (a culture is never split, and its
  windows form the replicate pairs); batch = the index of its class root;
  cls = the index of its class in the manifest's classes.
- Q14: one bank_root per plan, and no written bank overwritten (sections 2.6
  and Block 8): a second plan, or a re-run of the real mode, needs another
  folder or the old bank moved aside by the user.

Resolved by the author on 2026-10-06:

- Q1: a spike at exactly t = K Delta_t is not part of the contract
  (section 2.1).
- Q2: rule 1 is ">= 1", decided robustly to floating-point rounding
  (section 2.2).
- Q3: count conservation is part of the contract (section 2.1).
