"""
sim_observable.py
=================

STAGE 2 of the SBI export chain: turn the virtual-MEA DETECTIONS of one
simulation into exactly the observable the frozen DSN was trained to receive.

Separation of concerns: this module produces arrays. It does not load a model,
does not read a manifest, does not know what theta is, and does not write files.

--------------------------------------------------------------------------
THE SCALE-PARITY CONTRACT (read this before changing anything here)
--------------------------------------------------------------------------
The DSN was trained on REAL recordings preprocessed by
Deep-Summary-Network/Main/hpc/MultiChannel/channel_subset_extraction.py.
Its Stage 4 defines, for a subregion of n_e electrodes:

    C[k]       = sum_e | S_e intersect [k*Dt, (k+1)*Dt) |     (integer counts)
    R_tilde[k] = gaussian_filter1d(C, sigma = sigma_sm / Dt)
    R_tilde[k] = clip(R_tilde[k], 0, None)
    R_norm[k]  = R_tilde[k] / n_e                              (PER-ELECTRODE MEAN)

for each fixed k in {0, ..., K-1}, with K = floor(T / Dt).

The division by n_e is NOT optional and it is NOT a sum. The export handoff
states the pooling as "sum_over_electrodes"; that is wrong by a factor of n_e.
channel_subset_extraction.py carries an explicit "SCALE-PARITY FLAG" comment
recording that real traces are on a mean-per-electrode scale while the
synthetic generator emits unnormalised summed counts, and that the two
conventions MUST be reconciled because the CNN is not scale invariant. This
module implements the real-data convention on the simulated side, which is the
reconciliation.

Why n_e = 9 lines up on both sides. The real extractor uses
electrodes_per_subset = 9. The simulated probe (mea_probe.ProbeConfig) is
n_side = 3, i.e. M = n_side^2 = 9 electrodes at 60 um pitch, sampled at
fs = 10110.09 Hz -- the same raw rate as the 3Brain recordings. The two
observables are therefore commensurable by construction, provided both are
divided by their own electrode count.

Note that n_e counts MEMBER electrodes, not FIRING electrodes: subregion_ifr
divides by len(subregion.members) whether or not a given electrode fired. A
silent simulated electrode must therefore still count towards M.

--------------------------------------------------------------------------
DETECTED SPIKES, NOT GROUND TRUTH
--------------------------------------------------------------------------
The real ptrain_<k>.mat rasters are detector output. The simulated counterpart
is therefore process_campaign.py's det_t / det_ch (band-pass + Quiroga
threshold + refractory), NOT the simulator's spk_N_t / spk_N_i. Using ground
truth would remove the detection stage from one arm only and break parity.

--------------------------------------------------------------------------
THE simtime TRAP
--------------------------------------------------------------------------
process_campaign.py computes, per iteration,

    simtime = float(np.ceil(spk_t.max()))

i.e. it INFERS the duration from the last spike. A quiet simulation therefore
records a simtime well below the requested 180 s, K varies row to row, and any
resulting trace shorter than W is SILENTLY DROPPED by MEAWindowDataset
(`if L < self.window_length: continue`). build_pooled_ifr therefore takes T as
a REQUIRED explicit argument: pass the campaign's launch flag --simtime (from
job_args.json), never the npz field.

HPC note (hpc-python-compat): pure ASCII, LF-only. numpy + scipy only.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "DEFAULT_W_SIZE",
    "DEFAULT_GAUSSIAN_WINDOW",
    "pooled_spike_counts",
    "build_pooled_ifr",
    "window_trace",
    "reference_compute_ifr_trace",
]

DEFAULT_W_SIZE = 0.02            # Delta_t [s]   -> fs_ifr = 50 Hz
DEFAULT_GAUSSIAN_WINDOW = 0.04   # sigma_sm [s]


# --------------------------------------------------------------------------- #
# binning
# --------------------------------------------------------------------------- #
def pooled_spike_counts(spike_times_s, T: float, dt: float) -> np.ndarray:
    """Population spike-count histogram C[k], pooled over all electrodes.

    Parameters
    ----------
    spike_times_s : 1-D array of spike times [s], pooled across electrodes,
        OR a sequence of per-electrode 1-D arrays. Both are accepted and give
        the SAME result: summing per-electrode histograms over a shared bin
        grid is identical to histogramming the concatenation, because the bin
        edges do not depend on which electrode a spike came from.
    T : float
        Duration [s] of the bin grid. MUST be the requested simulation /
        recording duration, not one inferred from the data (see module header).
    dt : float
        Delta_t [s], the bin width.

    Returns
    -------
    C : (K,) float64, K = floor(T / dt). Non-negative integer-valued.

    Notes
    -----
    Spikes at or beyond the right edge K*dt are DISCARDED rather than folded
    into the last bin. np.histogram closes its final bin on the right, so a
    spike landing exactly at T would otherwise be counted while a spike at
    T + epsilon would not -- an asymmetry that depends on floating-point luck.
    Clipping the domain first makes the rule explicit and identical for the
    simulated and real arms.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be a finite positive duration [s]; got %r" % (T,))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive bin width [s]; got %r" % (dt,))

    K = int(T / dt)                       # K = floor(T / Delta_t)
    if K < 1:
        raise ValueError(
            "T / dt = %r floors to < 1 bin (T=%r, dt=%r)" % (T / dt, T, dt))
    edges = np.arange(K + 1, dtype=np.float64) * dt

    if isinstance(spike_times_s, np.ndarray) and spike_times_s.ndim == 1:
        groups = [spike_times_s]
    else:
        groups = list(spike_times_s)

    C = np.zeros(K, dtype=np.float64)
    for st in groups:
        st = np.asarray(st, dtype=np.float64).ravel()
        if st.size == 0:
            continue
        if not np.all(np.isfinite(st)):
            raise ValueError("spike times contain NaN or Inf")
        st = st[(st >= 0.0) & (st < edges[-1])]
        if st.size == 0:
            continue
        counts, _ = np.histogram(st, bins=edges)
        C += counts
    return C


# --------------------------------------------------------------------------- #
# the IFR
# --------------------------------------------------------------------------- #
def build_pooled_ifr(spike_times_s,
                     n_electrodes: int,
                     T: float,
                     dt: float = DEFAULT_W_SIZE,
                     sigma_sm: float = DEFAULT_GAUSSIAN_WINDOW,
                     normalise_per_electrode: bool = True) -> np.ndarray:
    """The pooled, smoothed, per-electrode-normalised IFR x in R_{>=0}^{K}.

    Implements exactly the real-data convention quoted in the module header.

    Parameters
    ----------
    spike_times_s : detected spike times [s] (pooled, or per-electrode lists)
    n_electrodes : int
        n_e, the number of MEMBER electrodes pooled -- 9 for both a real
        subregion and the simulated 3x3 probe. Silent electrodes still count.
    T : float
        Duration [s]. Pass the launch flag, not an inferred value.
    dt : float
        Delta_t [s].
    sigma_sm : float
        Gaussian smoothing width [s]. Converted to sigma_sm / dt BINS, which is
        what gaussian_filter1d expects.
    normalise_per_electrode : bool
        Divide by n_e. Leave True. Exposed only so the smoke test can exhibit
        the un-normalised variant and show the factor-of-n_e discrepancy.

    Returns
    -------
    x : (K,) float32, K = floor(T / dt), non-negative.
        Units: spikes per bin per electrode. Multiply by fs_ifr = 1/dt for Hz.
    """
    n_e = int(n_electrodes)
    if n_e < 1:
        raise ValueError("n_electrodes must be >= 1; got %r" % (n_electrodes,))
    if sigma_sm < 0.0:
        raise ValueError("sigma_sm must be >= 0; got %r" % (sigma_sm,))

    C = pooled_spike_counts(spike_times_s, T=T, dt=dt)

    if sigma_sm > 0.0:
        R = gaussian_filter1d(C, sigma=float(sigma_sm) / float(dt))
    else:
        R = C.copy()
    # The Gaussian is non-negative and C >= 0, so R >= 0 mathematically; the
    # clip only removes machine-precision negative noise from the filter tails.
    R = np.clip(R, 0.0, None)

    if normalise_per_electrode:
        R = R / float(n_e)
    return R.astype(np.float32)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def window_trace(x, window_length: int, stride: Optional[int] = None
                 ) -> Tuple[np.ndarray, List[int]]:
    """Cut a trace into windows using MEAWindowDataset's exact rule.

    The dataset builds its index as

        s = 0
        while s + W <= L:
            index.append((trace_idx, s, condition))
            s += stride

    so the number of windows is floor((L - W) / stride) + 1 when L >= W, and
    ZERO when L < W. That zero is returned as an empty array here rather than
    raising, but callers MUST check it: in the dataset the same condition is a
    silent `continue`, which is how a too-short trace disappears without a
    warning.

    Parameters
    ----------
    x : (K,) or (C, K) float array. Windows are cut along the LAST axis, as in
        MEAWindowDataset.__getitem__ (traces[ti][..., s:s + W]).
    window_length : int
        W in samples. Take this from FrozenDSN.window_length.
    stride : int or None
        Defaults to window_length, i.e. DISJOINT windows -- the eval_stride_s
        == window_s setting used by the MEA training config. Overlapping
        windows inflate the apparent sample size N and bias any clustering or
        two-sample statistic computed from the rows.

    Returns
    -------
    Xw     : (n_win, W) or (n_win, C, W) float32
    starts : list of the window start offsets, in samples
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    W = int(window_length)
    if W < 1:
        raise ValueError("window_length must be >= 1; got %r" % (window_length,))
    S = W if stride is None else int(stride)
    if S < 1:
        raise ValueError("stride must be >= 1; got %r" % (stride,))

    L = x.shape[-1]
    lead = tuple(x.shape[:-1])
    starts = []
    s = 0
    while s + W <= L:
        starts.append(s)
        s += S

    if not starts:
        return np.empty((0,) + lead + (W,), dtype=np.float32), []

    Xw = np.empty((len(starts),) + lead + (W,), dtype=np.float32)
    for i, s0 in enumerate(starts):
        Xw[i] = x[..., s0:s0 + W]
    return Xw, starts


# --------------------------------------------------------------------------- #
# parity reference (used by the smoke test)
# --------------------------------------------------------------------------- #
def reference_compute_ifr_trace(spike_times_s, T: float, dt: float,
                                sigma_sm: float,
                                dsn_main_dir: Optional[str] = None):
    """Call the DSN repository's OWN compute_ifr_trace, for bit-parity testing.

    Directive 1 says to reuse tested library code. build_pooled_ifr does not
    simply call compute_ifr_trace because that function lives in
    generate_burst_data.py, which imports matplotlib at module scope (an
    unwanted side effect on headless compute nodes) and takes a BurstParams
    dataclass carrying a dozen irrelevant generative fields. The extractor
    itself works around this with a lazy import; we take the same approach, but
    keep the dependency confined to the TEST path so the export can run in an
    environment where the DSN repo is not importable.

    Returns (ifr, fs_ifr), or raises ImportError if the repo is unavailable.
    """
    cand = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    if cand:
        cand = os.path.abspath(cand)
        if cand not in sys.path:
            sys.path.insert(0, cand)
    os.environ.setdefault("MPLBACKEND", "Agg")
    from dataclasses import replace
    from generate_burst_data import CONTROL_PARAMS, compute_ifr_trace

    params = replace(CONTROL_PARAMS, duration_s=float(T),
                     w_size=float(dt), gaussian_window=float(sigma_sm))
    if isinstance(spike_times_s, np.ndarray) and spike_times_s.ndim == 1:
        groups = [np.asarray(spike_times_s, dtype=np.float64)]
    else:
        groups = [np.asarray(g, dtype=np.float64) for g in spike_times_s]
    return compute_ifr_trace(groups, params)
