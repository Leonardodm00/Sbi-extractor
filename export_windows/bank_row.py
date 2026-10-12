"""Block 6.4a -- one simulation's bank row: its window x, its mean rate, the
floor, and its label in the unit box (specs/SPEC.md, Block 6.4a; sec. 2.5).

Shared by the plan's sample (Block 6.4) and the export of every row (Block
7.1), so that the estimate and the bank are computed by one piece of code.

  x      the pooled IFR of the window, sim_observable.build_pooled_ifr -- the
         export path's one implementation for both arms (SPEC sec. 2.1), with
         Delta_t and sigma_sm from the frozen cohort manifest (D-044; D-017,
         no defaults) -- (K,) float32, spikes per bin per electrode,
         K = floor(T / Delta_t), which must equal W (D-041)
  r      mean_t x(t) / Delta_t [Hz] = N / (K Delta_t) up to float32 rounding
         (count conservation, SPEC sec. 2.1)
  kept   r >= floor_hz * (1 - floor_rtol)                       (D-075; Q7)
  theta  theta_A of the task's own label spec (A6 against params, D-004),
         mapped into the unit box of the bank's DECLARED bounds (D-074):
         u = (theta_A - lo) / (hi - lo); a value outside [0, 1] by more than
         UNIT_TOL raises (nothing is clipped beyond roundoff).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import sbi_labels as SL                # noqa: E402
import sim_observable as SO            # noqa: E402

UNIT_TOL = 1e-12                       # float64 roundoff of a bound written and re-read


class RowError(ValueError):
    """One simulation that cannot become a row; the message names it."""


def window_from_detections(det_t, det_ch, n_e, T, dt, sigma_sm) -> np.ndarray:
    """x over [0, T): the per-electrode lists of the detections, pooled and smoothed."""
    det_ch = np.asarray(det_ch, dtype=np.int64)
    if det_ch.size and (det_ch.min() < 0 or det_ch.max() >= n_e):
        raise RowError("det_ch outside [0, %d): min %d, max %d"
                       % (n_e, int(det_ch.min()), int(det_ch.max())))
    det_t = np.asarray(det_t, dtype=np.float64)
    per_e = [det_t[det_ch == e] for e in range(int(n_e))]
    return SO.build_pooled_ifr(per_e, n_electrodes=int(n_e), T=float(T), dt=float(dt),
                               sigma_sm=float(sigma_sm))


def mean_rate_hz(x, dt) -> float:
    """mean_t x(t) in Hz: x is spikes per bin per electrode."""
    x = np.asarray(x)
    return float(np.sum(x, dtype=np.float64)) / (x.size * float(dt))


def is_kept(rate_hz, floor_hz, floor_rtol) -> bool:
    return bool(rate_hz >= float(floor_hz) * (1.0 - float(floor_rtol)))


def to_unit_box(theta_a, bounds) -> Tuple[np.ndarray, int]:
    """u = (theta_A - lo) / (hi - lo); returns (u, number of entries snapped from
    roundoff). Raises RowError beyond UNIT_TOL, naming the columns."""
    theta_a = np.asarray(theta_a, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    u = (theta_a - lo) / (hi - lo)
    bad = np.nonzero((u < -UNIT_TOL) | (u > 1.0 + UNIT_TOL) | ~np.isfinite(u))[0]
    if bad.size:
        raise RowError("label outside the declared box in column(s) %r: u = %r"
                       % (bad.tolist(), u[bad].tolist()))
    snapped = int(np.sum((u < 0.0) | (u > 1.0)))
    return np.clip(u, 0.0, 1.0), snapped


def build_row(det: Dict, spec_task, bounds, T, dt, sigma_sm, n_e, W,
              floor_hz, floor_rtol) -> Dict:
    """Everything one simulation contributes: x, r, kept, u (and its source theta_A)."""
    if det["n_e"] != int(n_e):
        raise RowError("n_e is %d, the bank's is %d (D-043)" % (det["n_e"], int(n_e)))
    x = window_from_detections(det["det_t"], det["det_ch"], n_e, T, dt, sigma_sm)
    if x.shape != (int(W),):
        raise RowError("the window has %d samples, W is %d (D-041)" % (x.shape[0], int(W)))
    rate = mean_rate_hz(x, dt)
    try:
        theta_a = SL.assemble_theta_A(spec_task, det["theta"], {"conn_prob": det["conn_prob"]},
                                      params_registry=det["params"])
    except (ValueError, KeyError, RuntimeError) as exc:
        raise RowError("label: %s" % exc)
    u, snapped = to_unit_box(theta_a, bounds)
    return {"x": x, "rate_hz": rate, "kept": is_kept(rate, floor_hz, floor_rtol),
            "u": u, "theta_a": theta_a, "snapped": snapped,
            "n_det": int(np.asarray(det["det_t"]).size)}
