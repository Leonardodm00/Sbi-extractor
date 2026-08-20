#!/usr/bin/env python3
"""
smoke_test_trim.py -- is the burn-in trim correct, and how much of the
transient survives it?

Run:
    python3 smoke_test_trim.py
    python3 smoke_test_trim.py -v        # print the measured numbers
    python3 smoke_test_trim.py -k T2

Covers the --trim_head_s option added to example_export.py by
trim_head.patch. The trim discards the first trim_head_s seconds of every
SIMULATED trace, because a simulation starts from initial conditions and
settles while a real recording is already at steady state: keeping the head
puts a transient in every simulated row and in no real row, which a kernel
two-sample test in summary space reports as decisive misspecification when
it is an initial-condition artefact.

Only numpy, scipy and sim_observable are needed -- no checkpoint, no torch,
no campaign. Run it on a login node.

  T0  Arithmetic. At dt = 0.01 s, T = 200 s gives K = 20000 bins; trimming
      20 s removes exactly 2000 and leaves exactly 18000 = W, i.e. exactly
      ONE window with nothing left over. The fit is exact, not approximate.
  T1  trim_head_s = 0 returns the array unchanged. Negative control for
      every positive test below, under the identical code path.
  T2  THE TEST THAT JUSTIFIES BUILD-THEN-SLICE. Slicing the full-grid IFR
      is NOT the same as rebuilding the IFR on the shortened interval: the
      rebuild puts a reflected gaussian_filter1d boundary at the new left
      edge. The two must differ near that edge and agree away from it.
  T3  Leakage. With a burst confined to the discarded head, the retained
      window must be numerically clean beyond a few smoothing widths. This
      is what makes the trim a real fix rather than a cosmetic one, and it
      QUANTIFIES the residual instead of asserting there is none.
  T4  A trim that removes the whole trace raises; a trim that leaves fewer
      than W samples yields ZERO windows, which the caller must catch --
      the pre-flight check in main() exists for exactly this.
  T5  The retained samples are bit-identical to the corresponding samples
      of the untrimmed trace. The trim must relocate nothing and rescale
      nothing.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_observable import build_pooled_ifr, window_trace  # noqa: E402

VERBOSE = False

# The refit checkpoint's geometry: read from
# out/refit_mea_A_best/checkpoints/seed_0/best.pt.
DT = 0.01              # cohort.w_size      [s]  -> fs_ifr = 100 Hz
SIGMA_SM = 0.02        # cohort.gaussian_window [s] -> 2 bins
WINDOW_S = 180.0       # data.window_s      [s]
T_SIM = 200.0          # the campaign's declared simtime [s]
TRIM_S = 20.0          # the burn-in to discard
N_E = 9

W = int(round(WINDOW_S / DT))          # 18000 samples
K_FULL = int(T_SIM / DT)               # 20000 bins
N_TRIM = int(round(TRIM_S / DT))       # 2000 bins


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def note(msg: str) -> None:
    if VERBOSE:
        print("      %s" % msg)


def trim_head(x: np.ndarray, trim_head_s: float, dt: float) -> np.ndarray:
    """The operation the patch performs, isolated so it can be tested.

    Kept byte-identical in behaviour to the hunk in iter_campaign_records:
    build over the FULL grid, then slice.
    """
    if trim_head_s <= 0.0:
        return x
    n_trim = int(round(float(trim_head_s) / float(dt)))
    if n_trim >= x.shape[0]:
        raise ValueError("trim_head_s = %.4g s removes the whole trace "
                         "(%d of %d bins at dt = %.4g s)"
                         % (trim_head_s, n_trim, x.shape[0], dt))
    return np.ascontiguousarray(x[n_trim:])


def spikes_steady(rate_hz: float = 3.0, t0: float = 0.0, t1: float = T_SIM,
                  seed: int = 0) -> List[np.ndarray]:
    """Per-electrode homogeneous Poisson trains on [t0, t1)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_E):
        n = rng.poisson(rate_hz * (t1 - t0))
        out.append(np.sort(rng.uniform(t0, t1, size=n)))
    return out


def spikes_with_transient(burst_rate: float = 400.0,
                          burst_until: float = 15.0,
                          seed: int = 1) -> List[np.ndarray]:
    """Steady firing everywhere, plus a violent burst confined to the head."""
    base = spikes_steady(seed=seed)
    rng = np.random.default_rng(seed + 999)
    out = []
    for st in base:
        n = rng.poisson(burst_rate * burst_until)
        burst = np.sort(rng.uniform(0.0, burst_until, size=n))
        out.append(np.sort(np.concatenate([st, burst])))
    return out


# --------------------------------------------------------------------------- #
def test_T0() -> None:
    x = build_pooled_ifr(spikes_steady(), n_electrodes=N_E, T=T_SIM,
                         dt=DT, sigma_sm=SIGMA_SM)
    check(x.shape[0] == K_FULL,
          "expected K = %d bins at T = %.4g s, dt = %.4g s; got %d"
          % (K_FULL, T_SIM, DT, x.shape[0]))

    xt = trim_head(x, TRIM_S, DT)
    check(xt.shape[0] == K_FULL - N_TRIM,
          "trim removed %d bins, expected %d"
          % (K_FULL - xt.shape[0], N_TRIM))
    check(xt.shape[0] == W,
          "the trimmed trace is %d samples but W = %d. The 20 s trim of a "
          "200 s simulation is supposed to fit the 180 s window EXACTLY."
          % (xt.shape[0], W))

    _, starts = window_trace(xt, window_length=W, stride=W)
    check(len(starts) == 1 and starts[0] == 0,
          "expected exactly one window at offset 0, got %r" % (starts,))
    note("K=%d, trim %d bins (%.4g s), leaves %d = W exactly -> 1 window, "
         "0 samples wasted" % (K_FULL, N_TRIM, TRIM_S, xt.shape[0]))

    # and the UNtrimmed case wastes the tail instead of the head
    _, starts0 = window_trace(x, window_length=W, stride=W)
    check(len(starts0) == 1 and starts0[0] == 0,
          "untrimmed: expected one window at 0, got %r" % (starts0,))
    note("untrimmed: 1 window covering [0, %.4g) s, discarding the LAST "
         "%.4g s -- i.e. keeping the transient and dropping steady state"
         % (WINDOW_S, T_SIM - WINDOW_S))


def test_T1() -> None:
    x = build_pooled_ifr(spikes_steady(seed=4), n_electrodes=N_E, T=T_SIM,
                         dt=DT, sigma_sm=SIGMA_SM)
    for zero in (0.0, -0.0):
        y = trim_head(x, zero, DT)
        check(y.shape == x.shape and np.array_equal(y, x),
              "trim_head_s = %r must be a no-op" % (zero,))
    note("trim_head_s = 0 leaves all %d samples untouched (negative control)"
         % x.shape[0])


def test_T2() -> None:
    st = spikes_steady(seed=11)
    full = build_pooled_ifr(st, n_electrodes=N_E, T=T_SIM, dt=DT,
                            sigma_sm=SIGMA_SM)
    sliced = trim_head(full, TRIM_S, DT)

    # The tempting alternative: rebuild the IFR on the shortened interval by
    # shifting the spike times. gaussian_filter1d then reflects at the new
    # left edge instead of seeing the real neighbouring bins.
    shifted = [s[s >= TRIM_S] - TRIM_S for s in st]
    rebuilt = build_pooled_ifr(shifted, n_electrodes=N_E,
                               T=T_SIM - TRIM_S, dt=DT, sigma_sm=SIGMA_SM)
    check(rebuilt.shape == sliced.shape,
          "the two constructions must have the same length: %r vs %r"
          % (rebuilt.shape, sliced.shape))

    edge = int(round(6 * SIGMA_SM / DT))          # 6 smoothing widths
    d_edge = float(np.abs(sliced[:edge] - rebuilt[:edge]).max())
    d_bulk = float(np.abs(sliced[edge:] - rebuilt[edge:]).max())
    check(d_bulk <= 1e-6,
          "away from the edge the two constructions must agree; max |diff| "
          "= %.3g" % d_bulk)
    check(d_edge > d_bulk,
          "the reflected boundary must show up at the left edge, but "
          "max|diff| there (%.3g) is not above the bulk (%.3g). If this "
          "fails the two constructions are interchangeable and the "
          "build-then-slice comment is wrong." % (d_edge, d_bulk))
    note("left edge (%d bins) max|slice - rebuild| = %.3g; bulk = %.3g "
         "-> build-then-slice is the correct construction"
         % (edge, d_edge, d_bulk))


def test_T3() -> None:
    st = spikes_with_transient()
    full = build_pooled_ifr(st, n_electrodes=N_E, T=T_SIM, dt=DT,
                            sigma_sm=SIGMA_SM)
    head_peak = float(full[:N_TRIM].max())
    kept = trim_head(full, TRIM_S, DT)

    steady = build_pooled_ifr(spikes_steady(seed=1), n_electrodes=N_E,
                              T=T_SIM, dt=DT, sigma_sm=SIGMA_SM)
    steady_peak = float(steady[N_TRIM:].max())

    check(head_peak > 5.0 * steady_peak,
          "fixture is wrong: the injected transient (%.4g) is not clearly "
          "above steady state (%.4g)" % (head_peak, steady_peak))

    kept_peak = float(kept.max())
    check(kept_peak < 2.0 * steady_peak,
          "the transient survived the trim: retained peak %.4g against a "
          "steady-state peak of %.4g" % (kept_peak, steady_peak))

    # Quantify the residual: the burst ends at 15 s, five seconds before the
    # cut, so the only route into the retained window is the Gaussian tail,
    # whose reach is a few sigma = a few hundredths of a second.
    leak = float(kept[:int(round(10 * SIGMA_SM / DT))].max())
    note("head peak %.4g, steady peak %.4g, retained peak %.4g, first-10-sigma "
         "max %.4g -> the burn-in does not survive the cut"
         % (head_peak, steady_peak, kept_peak, leak))

    # ... and WITHOUT the trim it very much does survive
    untrimmed_win, _ = window_trace(full, window_length=W, stride=W)
    check(float(untrimmed_win[0].max()) > 5.0 * steady_peak,
          "negative control failed: the UNtrimmed window should contain the "
          "transient, but its peak is only %.4g"
          % float(untrimmed_win[0].max()))
    note("negative control: the untrimmed window peak is %.4g, i.e. the "
         "transient is inside every simulated row when the trim is off"
         % float(untrimmed_win[0].max()))


def test_T4() -> None:
    x = build_pooled_ifr(spikes_steady(seed=5), n_electrodes=N_E, T=T_SIM,
                         dt=DT, sigma_sm=SIGMA_SM)

    try:
        trim_head(x, T_SIM, DT)
    except ValueError as exc:
        check("whole trace" in str(exc), "unhelpful message: %s" % exc)
    else:
        raise AssertionError("removing the whole trace must raise")
    note("a trim of the full duration raises")

    # A trim that leaves fewer than W samples does NOT raise; it silently
    # yields zero windows, which is why main() pre-checks T - trim >= T_win.
    too_much = TRIM_S + 5.0
    short = trim_head(x, too_much, DT)
    check(short.shape[0] < W, "fixture wrong: %d >= W" % short.shape[0])
    xw, starts = window_trace(short, window_length=W, stride=W)
    check(len(starts) == 0 and xw.shape[0] == 0,
          "a too-short trace must yield zero windows, got %d" % len(starts))
    note("trim of %.4g s leaves %d < W = %d samples -> 0 windows, NO "
         "exception. The pre-flight check in main() is what catches this."
         % (too_much, short.shape[0], W))


def test_T5() -> None:
    x = build_pooled_ifr(spikes_with_transient(seed=17), n_electrodes=N_E,
                         T=T_SIM, dt=DT, sigma_sm=SIGMA_SM)
    kept = trim_head(x, TRIM_S, DT)
    check(np.array_equal(kept, x[N_TRIM:]),
          "the retained samples must be bit-identical to the tail of the "
          "untrimmed trace: the trim relocates nothing and rescales nothing")
    check(kept.dtype == x.dtype, "dtype changed: %r vs %r"
          % (kept.dtype, x.dtype))
    check(kept.flags["C_CONTIGUOUS"], "the trimmed view must be contiguous")
    note("all %d retained samples bit-identical, dtype %r, contiguous"
         % (kept.size, kept.dtype))


TESTS = [
    ("T0", "20 s trim of a 200 s run fits W exactly, one window", test_T0),
    ("T1", "trim_head_s = 0 is a no-op (negative control)", test_T1),
    ("T2", "build-then-slice differs from rebuild at the edge", test_T2),
    ("T3", "the transient does not survive the trim; it does without", test_T3),
    ("T4", "over-trimming raises, or yields zero windows silently", test_T4),
    ("T5", "retained samples are bit-identical to the untrimmed tail", test_T5),
]


def main(argv=None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description="smoke tests for the burn-in trim")
    ap.add_argument("-k", dest="selector", default=None)
    ap.add_argument("-v", dest="verbose", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = bool(args.verbose)

    print("geometry: dt=%.4g s  sigma_sm=%.4g s  W=%d samples (%.4g s)  "
          "T_sim=%.4g s  trim=%.4g s" % (DT, SIGMA_SM, W, WINDOW_S,
                                         T_SIM, TRIM_S))
    print("")

    selected = [t for t in TESTS
                if args.selector is None or args.selector in t[0]]
    if not selected:
        print("no test matches %r" % args.selector)
        return 2

    n_pass = 0
    for tid, desc, fn in selected:
        try:
            fn()
            print("PASS  %s  %s" % (tid, desc))
            n_pass += 1
        except Exception:                    # noqa: BLE001
            print("FAIL  %s  %s" % (tid, desc))
            traceback.print_exc()

    print("")
    print("%d / %d passed" % (n_pass, len(selected)))
    return 0 if n_pass == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
