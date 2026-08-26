#!/usr/bin/env python3
"""
real_cohort_filter.py -- trim atypical real windows and pick the medoid.

WHAT THIS IS FOR
----------------
The truncated-prior box is an ENVELOPE over per-window highest-probability
regions,

    T_k = [ min_j lo_j^(k) , max_j hi_j^(k) ]     for each axis k,        (1)

i.e. a max-statistic over windows. A single window whose embedding sits far
from the bulk drags one edge of (1) outward and never gives it back, no
matter how tight every other window is. Windows carrying accentuated
biological variability therefore inflate the cumulative HPR for a reason
that has nothing to do with the simulator. This script removes them and,
on what remains, identifies the single most typical window per condition.

WHAT THIS IS NOT FOR
--------------------
This will NOT change the misspecification verdict. That question was
already settled in this project: deleting the 5% most extreme real windows
retained 99.0% of the gate statistic, because the rejection is a bulk
displacement between the arms, not a handful of extreme recordings. Report
results from a filtered cohort as a filtered cohort; do not present the
filtering as a fix for the gate.

BEFORE YOU TRUST THE OUTPUT, CHECK THE ENVELOPE ACTUALLY MOVES. The gain
assumed above is testable directly from a finished prior_truncate.py run
(see REAL_COHORT_FILTER.md, "Does it actually help?"). If the envelope is
set by the bulk rather than by a few windows, filtering cannot tighten it
and the medoid is the only useful output here.

METHOD
------
Per condition, independently:

1. CULTURE-BALANCED MEDOID. Each window j gets weight w_j = 1 / n(culture
   of j), so every culture contributes total weight 1 and a culture with
   more windows cannot drag the centre. The medoid is

       m = argmin_i  sum_j w_j * d(z_i, z_j),                             (2)

   an ACTUAL window, not a synthetic mean -- the downstream pipeline needs
   a real observation.

2. ROBUST THRESHOLD on the distance to that medoid, d_i = d(z_i, z_m):

       tau = median_i(d_i) + k * 1.4826 * MAD_i(d_i),                     (3)

   with MAD the median absolute deviation and 1.4826 the constant making
   1.4826*MAD a consistent estimator of sigma for Gaussian data. Median
   and MAD are used rather than mean and standard deviation because the
   quantities being detected would themselves inflate mean and sd, which
   is the classic masking failure.

3. CAP. At most --max_frac of the condition is ever removed; if (3) flags
   more, only the --max_frac most distant are taken. The cap is what keeps
   this a trim rather than an open-ended cull.

4. REPLICATE GUARD. If any one culture would lose more than
   --max_culture_frac of its windows, the script REFUSES. Losing most of
   one culture is not trimming variable windows, it is deleting a
   biological replicate, and it must be a deliberate act (--allow_culture_loss)
   rather than a side effect.

5. The medoid is then RECOMPUTED on the cleaned pool, by (2) again, and
   that is the one reported.

Distances are computed in the embedding the downstream pipeline consumes.
Note that the real arm of this project is close to rank 1, so "far from
the bulk" is in practice a one-dimensional statement; when zraw_* columns
are present the same flags are computed there too and the agreement
between the two spaces is reported, so you can see whether the collapse is
driving the selection.

OUTPUTS (all under --out, which is a PREFIX, matching real_source.py)
--------------------------------------------------------------------
  <out>_clean.parquet / .json    every condition, outliers removed.
                                 Drop-in for --real anywhere downstream.
  <out>_medoid.parquet / .json   one row per condition. Drop-in for --real
                                 in prior_truncate.py. NOT usable with
                                 gate_run.py, whose permutation test needs
                                 many windows grouped by culture.
  <out>_filter_report.json       counts, thresholds, per-culture breakdown,
                                 medoid identity, cross-space agreement.

The sidecar .json is copied from the input sidecar with a provenance block
appended, so the contract block (param_names, coord, bounds_theta,
dsn_checkpoint_sha256) survives intact and npe_contract.load_shard can
still open the result.

Only numpy and pandas/pyarrow are needed -- no torch, no sklearn (which is
not in environment.yml). Runs on a login node in seconds.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

import numpy as np

_Z_RE = re.compile(r"^z_(\d+)$")
_ZRAW_RE = re.compile(r"^zraw_(\d+)$")

SCHEMA_VERSION = 1
MAD_TO_SIGMA = 1.4826


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Trim atypical real windows and pick the medoid, "
                    "per condition.")
    ap.add_argument("--real", required=True,
                    help="the real-cohort parquet (e.g. "
                         "sbi_real_cohort.parquet); its .json sidecar must "
                         "sit next to it")
    ap.add_argument("--out", required=True,
                    help="output PREFIX; _clean/_medoid/_filter_report are "
                         "appended")
    ap.add_argument("--space", choices=("z", "zraw"), default="z",
                    help="embedding the distances are computed in "
                         "(default z, what the gate and NPE consume)")
    ap.add_argument("--metric", choices=("euclidean", "cosine"),
                    default="euclidean")
    ap.add_argument("--method", choices=("medoid", "knn"), default="medoid",
                    help="medoid: distance to the culture-balanced medoid "
                         "(global, default). knn: mean distance to the k "
                         "nearest neighbours (local density), which flags a "
                         "window isolated from its own neighbourhood even "
                         "when it is not far from the centre.")
    ap.add_argument("--n_neighbors", type=int, default=10,
                    help="k for --method knn (default 10)")
    ap.add_argument("--k_mad", type=float, default=3.0,
                    help="threshold at median + k_mad * 1.4826 * MAD "
                         "(default 3.0)")
    ap.add_argument("--max_frac", type=float, default=0.05,
                    help="never remove more than this fraction of a "
                         "condition (default 0.05)")
    ap.add_argument("--max_culture_frac", type=float, default=0.30,
                    help="refuse if any culture would lose more than this "
                         "fraction of its windows (default 0.30)")
    ap.add_argument("--allow_culture_loss", action="store_true",
                    help="downgrade the replicate guard to a warning")
    ap.add_argument("--class_col", default="condition")
    ap.add_argument("--group_col", default="culture")
    ap.add_argument("--dry_run", action="store_true",
                    help="report what would be removed; write nothing")
    args = ap.parse_args(argv)

    if not (0.0 < args.max_frac < 1.0):
        ap.error("--max_frac must be in (0, 1)")
    if not (0.0 < args.max_culture_frac <= 1.0):
        ap.error("--max_culture_frac must be in (0, 1]")
    if args.k_mad <= 0.0:
        ap.error("--k_mad must be positive")
    if args.n_neighbors < 1:
        ap.error("--n_neighbors must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ordered_cols(names, rx):
    hits = []
    for n in names:
        m = rx.match(n)
        if m:
            hits.append((int(m.group(1)), n))
    return [n for _, n in sorted(hits)]


def load_cohort(real_path, class_col, group_col):
    """Returns (df, sidecar, z_cols, zraw_cols)."""
    import pandas as pd

    if not os.path.isfile(real_path):
        raise FileNotFoundError("no such parquet: %s" % real_path)
    side_path = os.path.splitext(real_path)[0] + ".json"
    if not os.path.isfile(side_path):
        raise FileNotFoundError(
            "sidecar %s not found. The real export writes <stem>.parquet and "
            "<stem>.json together; without the sidecar the contract block "
            "would be lost and the output would not load downstream."
            % side_path)

    df = pd.read_parquet(real_path)
    with open(side_path, "r", encoding="utf-8") as fh:
        sidecar = json.load(fh)

    z_cols = _ordered_cols(df.columns, _Z_RE)
    zraw_cols = _ordered_cols(df.columns, _ZRAW_RE)
    if not z_cols:
        raise ValueError("no z_* columns in %s" % real_path)
    for col in (class_col, group_col):
        if col not in df.columns:
            raise KeyError(
                "column %r missing from %s; available: %s"
                % (col, real_path, sorted(df.columns)[:20]))

    # A culture must carry exactly one condition, or the per-condition split
    # is not a partition of cultures and the replicate guard is meaningless.
    for cu, sub in df.groupby(df[group_col].astype(str)):
        labs = sorted(set(sub[class_col].astype(str)))
        if len(labs) != 1:
            raise ValueError(
                "culture %r carries conditions %s; one culture must be one "
                "condition." % (cu, labs))
    return df, sidecar, z_cols, zraw_cols


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def pairwise(Z, metric):
    """Full pairwise distance matrix. n is at most a few thousand here."""
    if metric == "cosine":
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        U = Z / norm
        D = 1.0 - (U @ U.T)
        return np.maximum(D, 0.0)
    sq = np.sum(Z * Z, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.sqrt(np.maximum(D, 0.0))


def culture_weights(cultures):
    """w_j = 1 / n(culture of j): every culture gets total weight 1."""
    w = np.empty(cultures.shape[0], dtype=np.float64)
    for cu in np.unique(cultures):
        m = cultures == cu
        w[m] = 1.0 / float(m.sum())
    return w


def medoid_index(D, cultures):
    """Equation (2): the window minimising culture-balanced total distance."""
    w = culture_weights(cultures)
    return int(np.argmin(D @ w))


def robust_threshold(d, k_mad, max_frac):
    """Equation (3), with a percentile fallback for a degenerate MAD."""
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad > 0.0:
        tau = med + k_mad * MAD_TO_SIGMA * mad
        basis = "median + %.3g * 1.4826 * MAD" % k_mad
    else:
        # Every distance identical to the median: MAD carries no scale, so
        # fall back to a quantile. Without this the threshold would be the
        # median itself and would flag half the cohort.
        tau = float(np.quantile(d, 1.0 - max_frac))
        basis = "quantile(1 - max_frac) [MAD was zero]"
    return tau, med, mad, basis


def knn_score(D, k):
    """Mean distance to the k nearest neighbours, excluding self."""
    n = D.shape[0]
    k = min(k, max(1, n - 1))
    part = np.sort(D, axis=1)[:, 1:k + 1]
    return part.mean(axis=1)


def flag_condition(Z, cultures, args):
    """Outlier flags for one condition. Returns (flags, info)."""
    n = Z.shape[0]
    D = pairwise(Z, args.metric)
    m0 = medoid_index(D, cultures)

    if args.method == "knn":
        score = knn_score(D, args.n_neighbors)
        score_name = "mean distance to %d nearest neighbours" \
                     % min(args.n_neighbors, max(1, n - 1))
    else:
        score = D[:, m0]
        score_name = "distance to the culture-balanced medoid"

    tau, med, mad, basis = robust_threshold(score, args.k_mad, args.max_frac)
    flags = score > tau

    cap = int(np.floor(args.max_frac * n))
    capped = False
    if flags.sum() > cap:
        capped = True
        keep_idx = np.argsort(-score)[:cap]
        flags = np.zeros(n, dtype=bool)
        flags[keep_idx] = True

    info = {
        "n_windows": int(n),
        "score": score_name,
        "threshold": float(tau),
        "score_median": float(med),
        "score_mad": float(mad),
        "threshold_basis": basis,
        "n_flagged_by_threshold": int((score > tau).sum()),
        "cap": cap,
        "cap_applied": bool(capped),
        "n_removed": int(flags.sum()),
        "medoid_index_before_clean": int(m0),
    }
    return flags, info, D


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def culture_breakdown(cultures, flags):
    out = {}
    for cu in np.unique(cultures):
        m = cultures == cu
        n = int(m.sum())
        rem = int(flags[m].sum())
        out[str(cu)] = {"n_windows": n, "n_removed": rem,
                        "frac_removed": (rem / n) if n else 0.0}
    return out


def jaccard(a, b):
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return (inter / union) if union else 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    print("[1/4] loading")
    df, sidecar, z_cols, zraw_cols = load_cohort(
        args.real, args.class_col, args.group_col)
    cols = z_cols if args.space == "z" else zraw_cols
    if not cols:
        raise ValueError("--space zraw requested but the parquet has no "
                         "zraw_* columns")
    print("  %d windows, E=%d, zraw present: %s"
          % (len(df), len(z_cols), bool(zraw_cols)))

    classes = df[args.class_col].astype(str).to_numpy()
    cultures = df[args.group_col].astype(str).to_numpy()
    conditions = sorted(set(classes.tolist()))
    print("  conditions: %s, cultures: %d"
          % (conditions, len(set(cultures.tolist()))))

    print("[2/4] flagging, per condition")
    keep_mask = np.ones(len(df), dtype=bool)
    per_condition = {}
    medoid_rows = []
    violations = []

    for cond in conditions:
        sel = np.where(classes == cond)[0]
        Z = df.iloc[sel][cols].to_numpy(dtype=np.float64)
        cu = cultures[sel]
        flags, info, _ = flag_condition(Z, cu, args)

        # Cross-space agreement, when the other embedding is available. If
        # the two disagree badly, the selection is an artefact of whichever
        # space was chosen and should not be trusted without a look.
        other = zraw_cols if args.space == "z" else z_cols
        if other:
            Zo = df.iloc[sel][other].to_numpy(dtype=np.float64)
            flags_o, _, _ = flag_condition(Zo, cu, args)
            info["cross_space"] = {
                "other_space": "zraw" if args.space == "z" else "z",
                "n_flagged_other": int(flags_o.sum()),
                "jaccard": float(jaccard(flags, flags_o)),
            }

        info["per_culture"] = culture_breakdown(cu, flags)
        bad = [c for c, d in info["per_culture"].items()
               if d["frac_removed"] > args.max_culture_frac]
        if bad:
            violations.append((cond, bad, info["per_culture"]))

        keep_mask[sel[flags]] = False
        per_condition[cond] = info
        print("  condition %-6s n=%-5d removed=%-4d (%.2f%%)  threshold=%.4g%s"
              % (cond, info["n_windows"], info["n_removed"],
                 100.0 * info["n_removed"] / max(1, info["n_windows"]),
                 info["threshold"], "  [cap applied]"
                 if info["cap_applied"] else ""))
        if "cross_space" in info:
            print("    cross-space agreement (Jaccard): %.3f"
                  % info["cross_space"]["jaccard"])

    if violations:
        print()
        for cond, bad, brk in violations:
            print("  condition %s: culture(s) exceeding --max_culture_frac "
                  "(%.2f):" % (cond, args.max_culture_frac))
            for c in bad:
                print("    %-40s %d/%d removed (%.1f%%)"
                      % (c, brk[c]["n_removed"], brk[c]["n_windows"],
                         100.0 * brk[c]["frac_removed"]))
        msg = ("a culture would lose more than --max_culture_frac of its "
               "windows. That is deleting a biological replicate, not "
               "trimming variable windows.")
        if not args.allow_culture_loss:
            raise SystemExit(
                "REFUSING: %s\n  Re-run with --allow_culture_loss if this is "
                "deliberate, or raise --k_mad / lower --max_frac." % msg)
        print("  WARNING (--allow_culture_loss): %s" % msg)

    print("[3/4] medoid of the cleaned pool, per condition")
    for cond in conditions:
        sel = np.where((classes == cond) & keep_mask)[0]
        if sel.shape[0] == 0:
            raise ValueError("condition %r has no window left" % cond)
        Z = df.iloc[sel][cols].to_numpy(dtype=np.float64)
        D = pairwise(Z, args.metric)
        m = medoid_index(D, cultures[sel])
        row = int(sel[m])
        medoid_rows.append(row)
        w = culture_weights(cultures[sel])
        per_condition[cond]["medoid"] = {
            "row_index_in_input": row,
            "culture": str(cultures[row]),
            "n_candidates": int(sel.shape[0]),
            "weighted_mean_distance": float((D[m] @ w) / w.sum()),
            "rank_of_weighted_mean_distance": int(
                np.argsort(D @ w).tolist().index(m)) + 1,
        }
        ident = {c: str(df.iloc[row][c]) for c in ("name", "subregion")
                 if c in df.columns}
        per_condition[cond]["medoid"].update(ident)
        print("  condition %-6s medoid row %d, culture %s%s"
              % (cond, row, cultures[row],
                 (", name %s" % ident["name"]) if "name" in ident else ""))

    n_removed = int((~keep_mask).sum())
    print("  total removed: %d of %d (%.2f%%)"
          % (n_removed, len(df), 100.0 * n_removed / max(1, len(df))))

    if args.dry_run:
        print("[4/4] --dry_run: nothing written")
        return 0

    print("[4/4] writing")
    prov = {
        "schema_version": SCHEMA_VERSION,
        "tool": "real_cohort_filter.py",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "source_parquet": os.path.abspath(args.real),
        "space": args.space, "metric": args.metric, "method": args.method,
        "n_neighbors": args.n_neighbors, "k_mad": args.k_mad,
        "max_frac": args.max_frac,
        "max_culture_frac": args.max_culture_frac,
        "allow_culture_loss": bool(args.allow_culture_loss),
        "class_col": args.class_col, "group_col": args.group_col,
        "n_windows_before": int(len(df)),
        "n_windows_after": int(keep_mask.sum()),
        "n_removed": n_removed,
        "per_condition": per_condition,
        "note": "Filtered real cohort. This filtering does NOT address the "
                "misspecification finding, which is a bulk displacement "
                "rather than an outlier effect; report results from this "
                "cohort as coming from a filtered cohort.",
    }

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    clean = df.loc[keep_mask].reset_index(drop=True)
    clean.to_parquet(args.out + "_clean.parquet", index=False)
    side_clean = dict(sidecar)
    side_clean["real_cohort_filter"] = dict(prov, artifact="clean")
    with open(args.out + "_clean.json", "w", encoding="utf-8") as fh:
        json.dump(side_clean, fh, indent=2)

    med = df.loc[sorted(medoid_rows)].reset_index(drop=True)
    med.to_parquet(args.out + "_medoid.parquet", index=False)
    side_med = dict(sidecar)
    side_med["real_cohort_filter"] = dict(
        prov, artifact="medoid",
        note="One row per condition: the most typical window of the cleaned "
             "pool. Usable as --real for prior_truncate.py. NOT usable with "
             "gate_run.py, whose permutation test needs many windows "
             "grouped by culture.")
    with open(args.out + "_medoid.json", "w", encoding="utf-8") as fh:
        json.dump(side_med, fh, indent=2)

    with open(args.out + "_filter_report.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    for suffix in ("_clean.parquet", "_clean.json", "_medoid.parquet",
                   "_medoid.json", "_filter_report.json"):
        print("  wrote %s" % (args.out + suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
