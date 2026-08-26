# `real_cohort_filter.py` -- trimming atypical real windows, and the medoid

**Date:** 2026-08-26
**Repo:** `Sbi-extractor`
**Status:** delivered, smoke-tested (8/8), outputs verified to load through
`gate_data.load_real`.

Removes real windows that sit far from the bulk of their own condition, and
identifies the single most typical window per condition. Both outputs are
drop-in replacements for `--real` downstream.

---

## 1. Why

The truncated-prior box is an **envelope** over per-window
highest-probability regions,

```
T_k = [ min_j lo_j^(k) ,  max_j hi_j^(k) ]        for each axis k        (1)
```

which is a **max-statistic over windows**. One window whose embedding sits
far from the bulk pushes an edge of (1) outward and never gives it back,
however tight every other window is. Windows carrying accentuated
biological variability therefore inflate the cumulative HPR for reasons
having nothing to do with the simulator, and there is no reason to keep
them when the object being built is (1).

The medoid addresses the same inflation from the other end: reducing the
cohort to **one** representative window removes envelope inflation
entirely, and is the single-observation setting that truncated-proposal
SBI (Deistler, Goncalves & Macke) was designed for.

## 2. What this does NOT do -- read before reporting anything

**This will not change the misspecification verdict.** That was settled in
this project before this script existed: deleting the 5% most extreme real
windows retained **99.0%** of the gate statistic, because the rejection is
a *bulk displacement* between the two arms, not a handful of extreme
recordings. Filtering makes the cohort cleaner; it does not make the
simulator adequate.

Anything computed on a filtered cohort must be **reported as coming from a
filtered cohort**. The provenance block written into every output sidecar
records exactly what was removed and under which settings, so this is
recoverable after the fact -- but it belongs in the text, not only in the
JSON.

## 3. Does it actually help? Check before trusting it

The premise -- that a few windows inflate the envelope -- is directly
testable from any finished `prior_truncate.py` run, because
`truncate_arrays.npz` stores `window_lo` / `window_hi` per window. Run this
in the `Simulation-Based-Inference/hpc` directory:

```bash
python3 -c "
import numpy as np, json
r='results/<your_run>'
d=json.load(open(r+'/truncated_prior.json')); a=np.load(r+'/truncate_arrays.npz')
pr=np.array(d['prior_bounds_theta']); w=pr[:,1]-pr[:,0]
lo,hi=a['window_lo'],a['window_hi']
setters=set(lo.argmin(0).tolist())|set(hi.argmax(0).tolist())
print('windows:',lo.shape[0],' envelope-setting:',len(setters))
print('median single-window width/prior: %.4f'%np.median(((hi-lo)/w).mean(1)))
print('envelope width/prior            : %.4f'%(((hi.max(0)-lo.min(0))/w).mean()))
m=np.ones(lo.shape[0],bool); m[list(setters)]=False
print('envelope after dropping them    : %.4f'%(((hi[m].max(0)-lo[m].min(0))/w).mean()))
"
```

Read it as follows.

* **Envelope shrinks materially when the setters are dropped** -- a few
  windows really were inflating it. Filtering will pay off.
* **Envelope barely moves** -- the envelope is set by the *bulk*, so
  whichever window you delete, the next one stands in the same place.
  Filtering cannot tighten it, and the medoid is the only useful output.
* **The number of distinct envelope-setting windows approaches `2p`**
  (two per axis, `p` axes) -- every extremal slot is filled by a different
  window, which is the signature of exchangeable draws rather than
  outliers.
* **Median single-window width near 1.0** -- each individual posterior is
  already close to the prior, so neither filtering nor the medoid can help;
  the limitation is the posterior, not the window set.

## 4. Method

Per condition, independently.

**Culture-balanced medoid.** Each window `j` carries weight
`w_j = 1 / n(culture of j)`, so every culture contributes total weight 1
and a culture with more windows cannot drag the centre. The medoid is

```
m = argmin_i  sum_j  w_j * d(z_i, z_j)                                   (2)
```

an **actual window**, not a synthetic mean, because the downstream pipeline
needs a real observation.

**Robust threshold** on distance to that medoid, `d_i = d(z_i, z_m)`:

```
tau = median_i(d_i)  +  k * 1.4826 * MAD_i(d_i)                          (3)
```

`1.4826 * MAD` is a consistent estimator of sigma for Gaussian data. Median
and MAD are used rather than mean and standard deviation because the
windows being detected would themselves inflate mean and sd -- the classic
masking failure. If MAD is exactly zero the threshold falls back to the
`1 - max_frac` quantile.

**Cap.** At most `--max_frac` of a condition is ever removed; if (3) flags
more, only the most distant are taken. This keeps the procedure a trim
rather than an open-ended cull.

**Replicate guard.** If any single culture would lose more than
`--max_culture_frac` of its windows, the script **refuses**. Losing most of
one culture is deleting a biological replicate, not trimming variable
windows, and must be deliberate (`--allow_culture_loss`).

**The medoid is then recomputed on the cleaned pool** by (2) again, and
that is the one reported.

**Distance space.** Distances are computed in the embedding the pipeline
consumes (`z` by default). Note this project's real arm is close to rank 1,
so "far from the bulk" is in practice a one-dimensional statement. When
`zraw_*` columns exist the same flags are computed there too and the
**Jaccard agreement** between the two spaces is reported: low agreement
means the selection is an artifact of the space chosen and should not be
trusted without a look.

## 5. Usage

Dry run first -- it prints everything and writes nothing:

```bash
cd ~/repos/Sbi-extractor
python3 real_cohort_filter.py \
    --real /davinci-1/home/ldellamea/ANN/SBI_export_r2/sbi_real_cohort.parquet \
    --out  /davinci-1/home/ldellamea/ANN/SBI_export_r2/sbi_real_cohort_f \
    --dry_run
```

Then for real, by dropping `--dry_run`. The `.json` sidecar must sit beside
the input parquet; without it the contract block would be lost and the
output would not load downstream, so the script refuses rather than
guessing.

### Options

| flag | default | meaning |
|---|---|---|
| `--real` | required | real-cohort parquet; its `.json` sidecar must be adjacent |
| `--out` | required | output **prefix**, matching `real_source.py` convention |
| `--space` | `z` | `z` or `zraw`; distances are computed here |
| `--metric` | `euclidean` | or `cosine` (equivalent up to monotonicity on L2-normalised `z`) |
| `--method` | `medoid` | `medoid` (global) or `knn` (local density; flags a window isolated from its own neighbourhood even when not far from the centre) |
| `--n_neighbors` | `10` | k for `--method knn` |
| `--k_mad` | `3.0` | the `k` in (3); larger removes less |
| `--max_frac` | `0.05` | hard cap per condition |
| `--max_culture_frac` | `0.30` | replicate guard ceiling |
| `--allow_culture_loss` | off | downgrade the guard to a warning |
| `--class_col` / `--group_col` | `condition` / `culture` | identity columns |
| `--dry_run` | off | report only, write nothing |

## 6. Outputs

| file | contents |
|---|---|
| `<out>_clean.parquet` / `.json` | every condition, outliers removed. Drop-in for `--real` **anywhere** downstream. |
| `<out>_medoid.parquet` / `.json` | **one row per condition**: the most typical window of the cleaned pool. |
| `<out>_filter_report.json` | counts, thresholds, per-culture breakdown, medoid identity, cross-space agreement. |

Both parquets preserve the input columns, order and dtypes exactly; both
sidecars carry the original contract block (`param_names`, `coord`,
`bounds_theta`, `dsn_checkpoint_sha256`) plus a `real_cohort_filter`
provenance block. Verified: both load through `gate_data.load_real`.

### The medoid file has one limitation

It has one row per condition, so it **cannot** be used with `gate_run.py`,
whose permutation test needs many windows grouped by culture. It is an
input for `prior_truncate.py`, where a single observation is exactly the
intended setting. **The cleaned pool is what runs the full pipeline.**

## 7. Running the pipeline on the outputs

Both are drop-ins, so nothing else changes. In
`Simulation-Based-Inference/hpc`:

```bash
# gate, on the cleaned pool
python3 gate_run.py --real <out>_clean.parquet ...

# truncation on the cleaned pool
qsub -v REAL_PARQUET=<out>_clean.parquet,OUT_DIR=results/trunc_clean \
     jobs/truncate_run.pbs

# truncation on the single most typical window per condition
qsub -v REAL_PARQUET=<out>_medoid.parquet,OUT_DIR=results/trunc_medoid \
     jobs/truncate_run.pbs
```

`truncate_run.pbs` skips shard auto-discovery whenever `REAL_PARQUET` is
given explicitly, so the filtered file is used rather than the export's
original real cohort. Add `ENSEMBLE_DIR=<a previous run>/ensemble` to reuse
a trained ensemble and skip training.

For the medoid run, the box is a single HPR rather than an envelope, so
`n_windows` in the summary will read 1 per condition. Compare its retained
prior mass against the cleaned-pool run: the difference is exactly the
envelope inflation that the medoid removes.

## 8. Smoke test

```bash
cd ~/repos/Sbi-extractor && python3 smoke_test_real_cohort_filter.py
```

Expect `F1`-`F8 PASS` then `ALL CHECKS PASSED`, in seconds. It builds a
synthetic export with a known set of injected far-from-bulk windows and
asserts they are removed with no collateral, that the schema and contract
survive, that the cap and the replicate guard work, that `--dry_run` writes
nothing, and that the medoid is central and drawn from the cleaned pool.

## 9. Assumptions and caveats

* **A culture is one condition.** Enforced; a straddling culture is fatal,
  because otherwise the per-condition split is not a partition of cultures
  and the replicate guard is meaningless.
* **Windows within a condition are treated as exchangeable** for the
  purpose of (3). Between-culture structure is respected only through the
  weights in (2) and the guard, not modelled.
* **Rank-1 caveat.** The real arm is close to `r_eff = 1`, so distances in
  `z` are effectively one-dimensional. This makes the procedure more
  interpretable, not less valid -- but check the reported cross-space
  Jaccard before concluding the selection is encoder-independent.
* **The filter is unsupervised and uses only the real arm.** It never looks
  at the simulator, so it cannot bias the comparison toward agreement. It
  does change what the gate is testing, which is why the provenance block
  exists.
* **Not a fix for misspecification.** See section 2.
