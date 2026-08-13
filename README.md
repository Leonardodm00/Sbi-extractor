# sbi-export

Bridge between **Astro-Neuron-Network** (the CAdEx / tripartite-synapse simulator
and its virtual-MEA pipeline) and **Deep-Summary-Network** (the frozen 1D-ResNet
encoder), producing the `(z, theta_A)` training pairs for the amortized Neural
Posterior Estimator.

This package is **read-only** with respect to both source repositories. It
imports from them at runtime and never modifies them, per the export handoff's
"read-and-export only" contract. That is why it lives in its own repository
rather than inside either one.

---

## 1. What it does

```
   <campaign>/topo_k/iter_n.npz          <mea_out>/topo_k/mea_iter_n.npz
   (theta, params, conn_prob,            (det_t, det_ch, theta,
    p0_conn, d0_conn, beta_conn)          electrode_centers, ...)
            |                                        |
            |  join on (topo_idx, iter_idx)          |
            +--------------------+-------------------+
                                 |
                    sbi_labels   |   sim_observable
                theta_A in R^27  |   pooled IFR x in R_{>=0}^K
                                 |          |
                                 |   window at W = round(T_win * fs_ifr)
                                 |          |
                                 |     dsn_frozen: z = h_psi(x), ||z||=1
                                 |          |
                                 +----------+
                                            |
                                  export_embeddings
                                            |
                         sbi_<campaign>_<shard>.parquet  +  .json sidecar
```

### Modules

| File | Role | Depends on |
|---|---|---|
| `dsn_frozen.py` | Load the frozen DSN, forward pass, `z` and `zraw`, SHA-256 | torch, DSN repo |
| `sim_observable.py` | Detections -> pooled per-electrode-normalised IFR -> windows | numpy, scipy |
| `sbi_labels.py` | The 27-D label `theta_A` and its prior box `B` | numpy, sim repo |
| `export_embeddings.py` | Orchestration, assertions A1-A9, Parquet + sidecar | pyarrow |
| `example_export.py` | Runnable demo + Stage-5 campaign-walker template | all of the above |
| `smoke_test_sbi_export.py` | 12 correctness tests | all of the above |

Separation of concerns is strict: swapping the trace source, the label
definition or the encoder touches exactly one file.

---

## 2. The four things that will silently corrupt an export

Read these before running anything. Each is implemented correctly here, and each
contradicts something in the original data-export handoff.

### 2.1 Pooling is a MEAN, not a sum

`channel_subset_extraction.py` Stage 4 defines, for a subregion of `n_e`
electrodes and for each fixed bin index `k`:

```
C[k]       = sum_e | S_e intersect [k*Dt, (k+1)*Dt) |
R_tilde[k] = clip( gaussian_filter1d(C, sigma = sigma_sm / Dt)[k], 0, None )
R_norm[k]  = R_tilde[k] / n_e                      <-- PER-ELECTRODE MEAN
```

The handoff's sidecar says `"pooling": "sum_over_electrodes"`. That is wrong by
a factor of `n_e`. Smoke test **T3** measures it: peak IFR 2.015 (correct) vs
18.139 (handoff), a 9x amplitude offset into a network that is not
scale-invariant. `channel_subset_extraction.py` carries a "SCALE-PARITY FLAG"
comment recording that this reconciliation was deferred to the training/wiring
stage. This package is that stage.

`n_e = 9` on both sides: the real extractor uses `electrodes_per_subset = 9`,
and the simulated probe is `n_side = 3`, i.e. `M = 9` electrodes at 60 um pitch
sampled at 10110.09 Hz -- the same raw rate as the 3Brain recordings.

### 2.2 `simtime` in the MEA output is INFERRED from the last spike

`process_campaign.py` computes `simtime = float(np.ceil(spk_t.max()))`. A quiet
simulation therefore records a duration far below the requested 180 s. Smoke
test **T4**: a run whose last spike is at 3 s gives `K = 150` instead of
`K = 9000`, and `MEAWindowDataset` drops any trace shorter than `W` with a
silent `continue`.

**Always pass `T` from the `--simtime` launch flag in `job_args.json`.**
`build_pooled_ifr` makes `T` a required argument for exactly this reason.

### 2.3 The topology block is NOT in `mea_iter_*.npz`

`process_campaign.py` writes `conn_prob` but not `p0_conn`, `d0_conn`,
`beta_conn`. Those exist only in the original `iter_*.npz`. The export joins on
`(topo_idx, iter_idx)` and raises if that key is not unique rather than
inventing a surrogate.

### 2.4 The topology axes are LINEAR-uniform, never log

`sample_kernel_vector` and the `conn_prob` draw both call `rng.uniform` on
natural bounds. Rule (1) must not be applied to them: `p0_conn` has bounds
`[0.1, 1.0]`, i.e. **exactly** 1.000000 decades (smoke test **T9** checks this
to floating-point exactness), so rule (1) would classify it as a log axis and
the export would store `ln(p0)` against a linear prior box.

### Resolved handoff open points

| Handoff open point | Status |
|---|---|
| `E` unknown, expected in `[8, 16]` | **E = 16** (`config_mea_joint_full.json`), read from the checkpoint at runtime |
| `zraw_*` recoverability unknown | **Recoverable.** Forward hook on `model.head.proj`; T5 confirms non-unit norms |
| Real-data windowing unresolved | **Resolved.** `window_s = 180.0`, `eval_stride_s = 180.0` -> `W_r = floor(1200/180) = 6` disjoint windows |

Note that `config.py`'s *default* is `window_s = 200.0`, which would exceed a
180 s simulation and yield zero windows. `dsn_frozen` therefore refuses to fall
back on that default and reads `data.window_s` from the checkpoint.

---

## 3. Installation

### 3.1 Layout

Clone all three repositories as siblings:

```
~/repos/
  Astro-Neuron-Network/
  Deep-Summary-Network/
  sbi-export/            <-- this package
```

### 3.2 Environment

```bash
conda env create -f environment.yml
conda activate sbi_export
```

Or into an existing environment:

```bash
pip install "numpy>=1.24" "scipy>=1.10" "pyarrow>=12" "torch>=2.0"
```

Brian2 is **not** required. The registry modules import Brian2 only inside
functions, so `sbi_labels` loads them without it.

### 3.3 Point the package at the two source repositories

```bash
export DSN_MAIN_DIR="$HOME/repos/Deep-Summary-Network/Main"
export SIM_MAIN_DIR="$HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1"
```

Add these to `~/.bashrc` on the cluster. Every entry point also accepts
`--dsn_main_dir` / `--sim_dir` explicitly.

---

## 4. Verify the installation

```bash
cd ~/repos/sbi-export
python3 smoke_test_sbi_export.py
```

Expect `PASSED 12   FAILED 0   SKIPPED 0`. With the two env vars unset it
degrades to `PASSED 3  SKIPPED 9` rather than crashing -- useful for checking
that the pure-numpy layer works before the repos are in place.

Set `SMOKE_VERBOSE=1` for full tracebacks on failure.

Then the end-to-end demo, which needs no data at all:

```bash
python3 example_export.py --mode synthetic --out /tmp/demo/sbi_demo_0000 --n_sims 16
```

This builds an **untrained** backbone, so the embeddings are meaningless; it
exists to prove the plumbing and to show you the output shape. The sidecar
records a placeholder SHA-256 of all zeros so a demo file can never be mistaken
for a real export.

---

## 5. Running a real export

### 5.1 Prerequisite: the MEA pipeline must have run

The export is built from **detected** spikes, because the real `ptrain_*.mat`
rasters are themselves detector output. Using the simulator's ground-truth
`spk_N_t` would remove the detection stage from one arm only and break parity.

If `<mea_out>/topo_*/mea_iter_*.npz` does not exist yet, run first:

```bash
cd ~/repos/Astro-Neuron-Network/hpc/MEA\ Traces
python3 process_campaign.py \
    --campaign ~/campaigns/cadex_ns_001 \
    --out      ~/campaigns/cadex_ns_001_mea \
    --workers  32
```

### 5.2 Verify the npz schema against your own files

```bash
python3 -c "import numpy as np; d=np.load('PATH/mea_iter_00000.npz'); print(sorted(d.files))"
python3 -c "import numpy as np; d=np.load('PATH/iter_00000.npz');     print(sorted(d.files))"
```

Confirm `det_t`, `det_ch`, `theta`, `electrode_centers` in the first, and
`p0_conn`, `d0_conn`, `beta_conn`, `conn_prob` in the second.

### 5.3 Dry run on a handful of simulations

```bash
python3 example_export.py --mode campaign \
    --checkpoint  ~/runs/mea_joint_full/checkpoints/best.pt \
    --campaign    ~/campaigns/cadex_ns_001 \
    --mea_out     ~/campaigns/cadex_ns_001_mea \
    --campaign_id cadex_ns_001 \
    --out         /tmp/dryrun_0000 \
    --max_records 20
```

Check the printed banner:

- `E`, `W`, `T_win`, `fs_ifr` match the training config
- `traces too short : 0` -- anything above zero means the `simtime` trap fired
- `assertions passed` includes A2, A3, A4, A5, A7, A9
- the checkpoint SHA-256 is the one you expect

### 5.4 Full shard

```bash
python3 example_export.py --mode campaign \
    --checkpoint  ~/runs/mea_joint_full/checkpoints/best.pt \
    --campaign    ~/campaigns/cadex_ns_001 \
    --mea_out     ~/campaigns/cadex_ns_001_mea \
    --campaign_id cadex_ns_001 \
    --out         ~/export/sbi_cadex_ns_001_0000
```

---

## 6. Using the modules directly

The export function takes an **iterable of `TraceRecord`**, not a campaign path.
That decoupling is deliberate: the real-recording export reuses it unchanged by
yielding records with `theta_A=None`.

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/repos/sbi-export"))

from dsn_frozen import load_frozen_dsn
from sim_observable import build_pooled_ifr
from sbi_labels import load_registry, build_label_spec, assemble_theta_A
from export_embeddings import TraceRecord, export_embeddings, run_assertion_A1

# 1. the encoder -- E, W, Delta_t, sigma_sm all come FROM the checkpoint
dsn = load_frozen_dsn("~/runs/.../best.pt", device="cpu",
                      expect_config_json="~/repos/Deep-Summary-Network/Main/"
                                         "hpc/Config/config_mea_joint_full.json")
print(dsn.embedding_dim, dsn.window_length, dsn.fs_ifr, dsn.ckpt_sha256)

# 2. the label spec
reg  = load_registry()
run_assertion_A1(reg)
spec = build_label_spec(reg, reg.sweep_groups["neuron_synapse"],
                        conn_prob_bounds=(0.1, 0.6))

# 3. the observable, for one simulation
x = build_pooled_ifr(per_electrode_spike_times, n_electrodes=9,
                     T=180.0, dt=dsn.w_size, sigma_sm=dsn.gaussian_window)

# 4. the label, for the same simulation
theta_A = assemble_theta_A(spec, theta36, topo_dict, params_36=params36)

# 5. export
out = export_embeddings(
    dsn, [TraceRecord(trace=x, theta_A=theta_A, ident={...})],
    "~/export/sbi_shard_0000", label_spec=spec,
    ident_columns=("campaign_id", "topo_idx", "iter_idx", "seed_run"))
```

Passing `expect_config_json` is the cheap guard against embedding an export with
the **wrong checkpoint** -- a mistake no downstream diagnostic can detect.

Passing `params_36` enables assertion A6, the per-row coordinate spot-check that
catches a stale registry.

---

## 7. Output format

`sbi_<campaign_id>_<shard>.parquet`, one row per window:

| Column | Type | Meaning |
|---|---|---|
| `campaign_id`, `topo_idx`, `iter_idx`, `seed_run` | provenance | row identity |
| `window_idx` | int | window within the trace; always 0 for 180 s sims |
| `z_000 ... z_015` | float32 | `z = h_psi(x)`, on `S^{E-1}` |
| `zraw_000 ... zraw_015` | float32 | pre-normalisation activations |
| `th_Sigma ... th_beta_conn` | float64 | `theta_A`, 27 columns in `param_names` order |

The sidecar carries `param_names`, `coord` (`"ln"` / `"linear"`),
`bounds_theta`, the registry block, the embedding block (including
`dsn_checkpoint_sha256`), the observable block and the provenance counts.

Reading it back:

```python
import pyarrow.parquet as pq, json, numpy as np
tbl  = pq.read_table("sbi_cadex_ns_001_0000.parquet")
side = json.load(open("sbi_cadex_ns_001_0000.json"))

E = side["embedding"]["embedding_dim"]
Z = np.column_stack([tbl.column("z_%03d" % j).to_numpy() for j in range(E)])
Theta = np.column_stack([tbl.column("th_" + n).to_numpy()
                         for n in side["param_names"]])
B = np.asarray(side["bounds_theta"])          # the sbi BoxUniform prior
```

Mapping posterior samples back to biophysical units:

```python
coord = side["coord"]
nat = Theta.copy()
for j, c in enumerate(coord):
    if c == "ln":
        nat[:, j] = np.exp(Theta[:, j])       # natural log, never log10
```

---

## 8. Assertions

| ID | Statement | Checked in |
|---|---|---|
| A1 | transform round trip at both bounds edges | `run_assertion_A1` |
| A2 | 27 `th_*` columns; names/coord/bounds agree | `export_embeddings` |
| A3 | non-degenerate prior box | `build_label_spec` **and** export |
| A4 | no constant `th_*` column | `export_embeddings` |
| A5 | every `theta_A` inside `B` (reported, never clipped) | `export_embeddings` |
| A6 | per-row coordinate spot-check vs `params` | `assemble_theta_A` |
| A7 | `abs(norm(z) - 1) < 1e-5` | `export_embeddings` |
| A8 | checkpoint identity across shards | **cross-file, NOT checked here** |
| A9 | no NaN / Inf | `export_embeddings` |
| A10 | cross-campaign compatibility | **pooling stage, NOT checked here** |

A8 and A10 are deliberately absent from `assertions_passed`: neither is a
property of a single shard. Before training, compare digests:

```bash
for f in ~/export/*.json; do
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); \
    print(d['embedding']['dsn_checkpoint_sha256'][:16], sys.argv[1])" "$f"
done | sort | uniq -c -w16
```

More than one distinct digest means the export is void.

Failures **raise**. Nothing is silently clipped, repaired or dropped.

---

## 9. HPC notes

Every `.py` here is pure ASCII and LF-only by construction. After transferring
to the cluster, run the verification block before submitting anything:

```bash
python3 --version; echo "conda env: ${CONDA_DEFAULT_ENV:-none}"

# line endings (fatal for any shebang'd file)
python3 - <<'EOF'
import os, sys
bad = [(os.path.join(r,f), open(os.path.join(r,f),'rb').read().count(b'\r'))
       for r,d,fs in os.walk('.') for f in fs
       if f.endswith(('.sh','.pbs','.slurm'))]
bad = [(p,n) for p,n in bad if n]
for p,n in bad: print("CRLF", p, n)
print("FAIL: %d" % len(bad) if bad else "OK: LF-only")
sys.exit(1 if bad else 0)
EOF

# encoding
python3 - <<'EOF'
import os, sys
bad = []
for r,d,fs in os.walk('.'):
    d[:] = [x for x in d if x not in ('__pycache__','.git')]
    for f in fs:
        if f.endswith('.py'):
            p = os.path.join(r,f)
            b = [hex(c) for c in open(p,'rb').read() if c > 127]
            if b: bad.append((p, b[:6]))
for p,b in bad: print("NON-ASCII", p, b)
print("FAIL: %d" % len(bad) if bad else "OK: pure ASCII")
sys.exit(1 if bad else 0)
EOF

python3 -m py_compile *.py && echo "OK: compiles"
python3 smoke_test_sbi_export.py
```

Fix any CRLF with `sed -i 's/\r$//' <file>`.

---

## 10. Not yet implemented

- **Real-recording export** (handoff 6.2). Same `export_embeddings` call with
  `label_spec=None`; needs a trace source over the extracted `.npz` archives.
- **OOD probe set** (handoff 6.3).
- **The simulation-vs-real gate** (handoff 7): geodesic k-NN distances on
  `S^{E-1}` plus a kernel two-sample test. Note that Schmitt et al. prescribe a
  unit-Gaussian summary space via an MMD penalty during training; this DSN is
  metric-learned onto the sphere instead, so their critical values do not
  transfer and the null must be calibrated with a sim-to-sim permutation
  baseline.
