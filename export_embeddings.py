"""
export_embeddings.py
====================

STAGE 4 of the SBI export chain: the requested function. Given a trained DSN
checkpoint and a source of (trace, label) pairs, run the forward pass, pair each
embedding row z with the parameter vector theta_A that generated the simulation
behind it, and write the Parquet + JSON sidecar pair specified in Section 6.1 of
the export handoff.

Separation of concerns. This module is ORCHESTRATION + IO ONLY:
    encoder           -> dsn_frozen.FrozenDSN
    trace construction-> sim_observable
    label assembly    -> sbi_labels
    here              -> iterate, embed, assert, write
Swapping the trace source (simulated campaign / real recordings / OOD probe set)
means writing a new source iterator, not editing this file.

--------------------------------------------------------------------------
THE TRACE SOURCE PROTOCOL
--------------------------------------------------------------------------
export_embeddings takes an ITERABLE of TraceRecord. That decoupling is the whole
design: the real-recording export (handoff Section 6.2) reuses this function
unchanged, simply yielding records whose theta_A is None.

Each TraceRecord carries:
    trace   : (K,) or (C, K) float array -- the pooled IFR for ONE simulation
              or ONE real recording, ALREADY built by sim_observable
    theta_A : (p,) array or None -- the label; None for real data
    ident   : dict of provenance columns (campaign_id, topo_idx, iter_idx, ...)

One trace may yield SEVERAL rows, because a recording longer than T_win is cut
into windows. Every window of one recording inherits that recording's theta_A
and provenance, plus its own window_idx. For the 180 s simulations at
T_win = 180 s this is exactly one row per simulation; for the 1200 s real
recordings it is W_r = floor(1200 / 180) = 6.

--------------------------------------------------------------------------
ASSERTIONS
--------------------------------------------------------------------------
A2, A3, A4, A5, A7, A8, A9 are checked here. A1 (transform round trip) and A6
(coordinate spot-check) are checked in sbi_labels / by run_assertion_A1. A10
(cross-campaign compatibility) is not a per-shard property and belongs to
whatever pools shards; this module writes the fields A10 needs into the sidecar
and refuses to pool.

Failures RAISE. Nothing is silently repaired, clipped or dropped.

HPC note (hpc-python-compat): pure ASCII, LF-only.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from dsn_frozen import FrozenDSN
from sim_observable import window_trace

__all__ = [
    "TraceRecord",
    "ExportResult",
    "run_assertion_A1",
    "export_embeddings",
]

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# the source protocol
# --------------------------------------------------------------------------- #
@dataclass
class TraceRecord:
    """One trace plus the label and provenance that belong to it."""
    trace: np.ndarray
    ident: Dict[str, object] = field(default_factory=dict)
    theta_A: Optional[np.ndarray] = None


@dataclass
class ExportResult:
    parquet_path: str
    sidecar_path: str
    n_rows: int
    n_traces_used: int
    n_traces_skipped_short: int
    embedding_dim: int
    assertions_passed: List[str]
    warnings: List[str]


# --------------------------------------------------------------------------- #
# A1 -- transform round trip
# --------------------------------------------------------------------------- #
def run_assertion_A1(registry, rtol: float = 1e-9) -> None:
    """A1: theta_to_natural(natural_to_theta(v)) == v at both edges of every
    row of PARAM_BOUNDS, for all registry axes.

    Uses the simulator's own transforms, so this tests the functions that
    actually produced the data rather than a local copy of them.
    """
    import HPC_main_sweep as SW  # already importable: load_registry ran first

    lo = registry.param_bounds[:, 0].astype(np.float64)
    hi = registry.param_bounds[:, 1].astype(np.float64)
    for name, v in (("lo", lo), ("hi", hi)):
        back = SW.theta_to_natural(SW.natural_to_theta(v))
        if not np.allclose(back, v, rtol=rtol, atol=0.0):
            worst = int(np.argmax(np.abs(back - v)))
            raise AssertionError(
                "assertion A1 failed at PARAM_BOUNDS[:, %s], worst axis %d "
                "(%s): %r -> %r" % (name, worst,
                                    registry.param_names[worst],
                                    float(v[worst]), float(back[worst])))


# --------------------------------------------------------------------------- #
# atomic write
# --------------------------------------------------------------------------- #
def _atomic_write_bytes(path, writer_fn) -> None:
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        writer_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# the function
# --------------------------------------------------------------------------- #
def export_embeddings(dsn: FrozenDSN,
                      records: Iterable[TraceRecord],
                      out_stem,
                      label_spec=None,
                      ident_columns: Sequence[str] = (),
                      extra_sidecar: Optional[dict] = None,
                      batch_size: int = 256,
                      window_stride: Optional[int] = None,
                      want_zraw: bool = True,
                      strict_in_box: bool = True) -> ExportResult:
    """Embed every window of every record and write <out_stem>.parquet / .json.

    Parameters
    ----------
    dsn : FrozenDSN
        From dsn_frozen.load_frozen_dsn. Supplies psi, E, W, Delta_t, sigma_sm
        and the checkpoint digest.
    records : iterable of TraceRecord
    out_stem : str
        Output path WITHOUT extension, e.g. ".../sbi_<campaign_id>_<shard>".
    label_spec : sbi_labels.LabelSpec or None
        None for real recordings, which have no theta. When None, no th_*
        columns are written and A3/A4/A5 are skipped as inapplicable.
    ident_columns : sequence of str
        Provenance column names to carry through from record.ident, in order.
    extra_sidecar : dict or None
        Merged into the sidecar (e.g. the 'simulation', 'provenance' and
        electrode-geometry parts of 'observable').
    batch_size : int
        Forward-pass chunk size. Does not change Z.
    window_stride : int or None
        Defaults to W, i.e. disjoint windows. Overlap inflates the apparent N.
    want_zraw : bool
        Export the pre-normalisation activations. Recoverable via a hook on the
        head Linear, so the handoff's "optional, may be unavailable" is resolved
        in favour of available.
    strict_in_box : bool
        A5. When True an out-of-box theta_A raises; when False it is counted and
        reported. Never clipped either way.

    Returns
    -------
    ExportResult
    """
    out_stem = os.path.abspath(str(out_stem))
    warnings_out = list(dsn.warnings)

    W = int(dsn.window_length)
    stride = W if window_stride is None else int(window_stride)
    if stride < W:
        warnings_out.append(
            "window_stride=%d < W=%d: windows OVERLAP, which inflates the "
            "apparent sample size N and biases any two-sample or clustering "
            "statistic computed on the exported rows." % (stride, W))

    # ---- 1. gather windows ------------------------------------------------
    X_parts: List[np.ndarray] = []
    idents: List[Dict[str, object]] = []
    thetas: List[np.ndarray] = []
    n_used = 0
    n_short = 0

    for rec in records:
        Xw, starts = window_trace(rec.trace, window_length=W, stride=stride)
        if Xw.shape[0] == 0:
            n_short += 1
            continue
        n_used += 1
        X_parts.append(Xw)
        for w_idx, _s0 in enumerate(starts):
            row_ident = dict(rec.ident)
            row_ident["window_idx"] = int(w_idx)
            idents.append(row_ident)
            if label_spec is not None:
                if rec.theta_A is None:
                    raise ValueError(
                        "label_spec was given but a record carries theta_A=None "
                        "(ident=%r). Mixing labelled and unlabelled rows in one "
                        "shard would silently produce a training set with "
                        "missing labels." % (rec.ident,))
                th = np.asarray(rec.theta_A, dtype=np.float64).ravel()
                if th.shape[0] != label_spec.p:
                    raise ValueError(
                        "theta_A has %d entries, expected p=%d"
                        % (th.shape[0], label_spec.p))
                thetas.append(th)

    if n_short:
        warnings_out.append(
            "%d trace(s) were SHORTER than W = %d samples (%.4g s) and produced "
            "no windows. MEAWindowDataset drops these with a silent `continue`; "
            "the most likely cause is an IFR built on process_campaign.py's "
            "INFERRED simtime (ceil of the last spike) instead of the "
            "campaign's --simtime launch flag."
            % (n_short, W, dsn.window_s))

    if not X_parts:
        raise ValueError(
            "no windows were produced from any record. With W = %d samples "
            "(%.4g s at fs_ifr = %.4g Hz), every trace was too short."
            % (W, dsn.window_s, dsn.fs_ifr))

    X = np.concatenate(X_parts, axis=0)
    n_rows = X.shape[0]

    # ---- 2. forward pass --------------------------------------------------
    Z, Zraw = dsn.embed(X, batch_size=batch_size, want_zraw=want_zraw)
    E = int(dsn.embedding_dim)

    # ---- 3. assertions ----------------------------------------------------
    passed: List[str] = []

    # A9 -- no NaN / Inf
    if not np.all(np.isfinite(Z)):
        raise AssertionError("assertion A9 failed: z contains NaN or Inf")
    if Zraw is not None and not np.all(np.isfinite(Zraw)):
        raise AssertionError("assertion A9 failed: zraw contains NaN or Inf")

    # A7 -- normalisation
    if dsn.l2_normalize:
        norms = np.linalg.norm(Z.astype(np.float64), axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst >= 1e-5:
            raise AssertionError(
                "assertion A7 failed: max | ||z||_2 - 1 | = %.3e >= 1e-5" % (worst,))
        passed.append("A7")
    else:
        warnings_out.append(
            "A7 skipped: the checkpoint has l2_normalize=False, so rows are not "
            "on S^{E-1}.")

    Theta = None
    if label_spec is not None:
        Theta = np.asarray(thetas, dtype=np.float64)
        if Theta.shape != (n_rows, label_spec.p):
            raise RuntimeError(
                "theta matrix is %r, expected (%d, %d)"
                % (Theta.shape, n_rows, label_spec.p))

        if not np.all(np.isfinite(Theta)):
            raise AssertionError("assertion A9 failed: theta contains NaN or Inf")
        passed.append("A9")

        # A2 -- column count / order agreement
        if not (len(label_spec.param_names) == len(label_spec.coord)
                == label_spec.bounds_theta.shape[0] == label_spec.p):
            raise AssertionError("assertion A2 failed: param_names / coord / "
                                 "bounds_theta lengths disagree")
        passed.append("A2")

        # A3 -- non-degenerate box (also enforced at construction)
        widths = label_spec.bounds_theta[:, 1] - label_spec.bounds_theta[:, 0]
        if not np.all(widths > 0.0):
            bad = [label_spec.param_names[i] for i in np.where(widths <= 0)[0]]
            raise AssertionError("assertion A3 failed for %r" % (bad,))
        passed.append("A3")

        # A4 -- no constant columns
        var = Theta.var(axis=0)
        const = [label_spec.param_names[i] for i in np.where(var <= 0.0)[0]]
        if const:
            raise AssertionError(
                "assertion A4 failed: zero-variance th_* column(s) %r. A "
                "constant label column carries no information and will make "
                "the flow's conditional density degenerate along that axis."
                % (const,))
        passed.append("A4")

        # A5 -- in box
        lo = label_spec.bounds_theta[:, 0][None, :]
        hi = label_spec.bounds_theta[:, 1][None, :]
        out_lo = Theta < lo
        out_hi = Theta > hi
        n_viol = int(np.count_nonzero(out_lo | out_hi))
        if n_viol:
            per_axis = {label_spec.param_names[j]:
                        int(np.count_nonzero(out_lo[:, j] | out_hi[:, j]))
                        for j in range(label_spec.p)
                        if np.any(out_lo[:, j] | out_hi[:, j])}
            msg = ("assertion A5: %d theta entries lie OUTSIDE the prior box, "
                   "by axis: %r. Reporting rather than clipping."
                   % (n_viol, per_axis))
            if strict_in_box:
                raise AssertionError(msg)
            warnings_out.append(msg)
        else:
            passed.append("A5")
    else:
        passed.append("A9")

    # A8 is a CROSS-FILE property: it can only be checked when the simulated and
    # real sidecars are compared. The digest is written so that check is possible.
    warnings_out.append(
        "A8 (checkpoint identity) is cross-file and is NOT verified here. "
        "Compare dsn_checkpoint_sha256 across every simulated and real sidecar "
        "before training. Current digest: %s" % (dsn.ckpt_sha256,))

    # ---- 4. build the table ----------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns: Dict[str, object] = {}
    for col in ident_columns:
        columns[col] = [rec_ident.get(col) for rec_ident in idents]
    if "window_idx" not in columns:
        columns["window_idx"] = [int(d["window_idx"]) for d in idents]

    for j in range(E):
        columns["z_%03d" % j] = Z[:, j].astype(np.float32)
    if Zraw is not None:
        for j in range(E):
            columns["zraw_%03d" % j] = Zraw[:, j].astype(np.float32)
    if label_spec is not None:
        for j, name in enumerate(label_spec.column_names):
            columns[name] = Theta[:, j].astype(np.float64)

    table = pa.table(columns)
    parquet_path = out_stem + ".parquet"
    _atomic_write_bytes(parquet_path,
                        lambda p: pq.write_table(table, p, compression="snappy"))

    # ---- 5. sidecar -------------------------------------------------------
    observable = dict(dsn.observable_block())
    observable["pooling"] = "mean_over_electrodes"
    observable["pooling_note"] = (
        "R_norm = gaussian_filter1d(pooled counts, sigma_sm/Delta_t) / n_e. "
        "This is the MEAN per electrode, matching Stage 4 of "
        "channel_subset_extraction.py. The export handoff states "
        "'sum_over_electrodes', which is wrong by a factor of n_e and would "
        "make every simulated trace n_e times too tall.")
    observable["spike_source"] = "detected (band-pass + Quiroga threshold)"

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(n_rows),
        "n_traces_used": int(n_used),
        "n_traces_skipped_too_short": int(n_short),
        "window_stride_samples": int(stride),
        "embedding": dsn.embedding_block(zraw_available=Zraw is not None),
        "observable": observable,
        "assertions_passed": sorted(set(passed)),
        "warnings": warnings_out,
    }
    if label_spec is not None:
        sidecar["param_names"] = list(label_spec.param_names)
        sidecar["coord"] = list(label_spec.coord)
        sidecar["param_units"] = list(label_spec.units)
        sidecar["bounds_theta"] = label_spec.bounds_theta.tolist()
        sidecar["registry"] = label_spec.sidecar_registry_block()
    if extra_sidecar:
        for k, v in extra_sidecar.items():
            if k in ("embedding", "observable") and isinstance(v, dict):
                sidecar[k].update(v)
            else:
                sidecar[k] = v

    sidecar_path = out_stem + ".json"

    def _write_json(p):
        with open(p, "w") as fh:
            json.dump(sidecar, fh, indent=2, sort_keys=False)
            fh.write("\n")

    _atomic_write_bytes(sidecar_path, _write_json)

    return ExportResult(
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        n_rows=n_rows,
        n_traces_used=n_used,
        n_traces_skipped_short=n_short,
        embedding_dim=E,
        assertions_passed=sorted(set(passed)),
        warnings=warnings_out,
    )
