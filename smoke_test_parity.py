#!/usr/bin/env python3
"""
smoke_test_parity.py -- does check_preprocessing_parity.py actually catch a
preprocessing mismatch, and stay quiet when there is none?

Run:
    python3 smoke_test_parity.py
    python3 smoke_test_parity.py -v
    python3 smoke_test_parity.py -k P2

A checker that never fires is worse than no checker: it looks like evidence
while providing none. Every positive test below therefore carries a negative
control under the IDENTICAL criterion -- the same fixtures, the same
thresholds, only the injected fault removed.

Fixtures are synthetic .npz archives in a temporary directory. Nothing here
needs the cluster data, a checkpoint, or torch.

  P0  n_windows implements floor((L - W) / S) + 1, and ZERO below W. The
      arithmetic every other check depends on.
  P1  summarise reports the statistics it claims to, on an array whose
      answers are known in closed form.
  P2  THE TEST THAT JUSTIFIES THE SCRIPT. A simulated arm scaled by n_e = 9
      -- the pooling-convention bug that has already occurred once in this
      project -- is FLAGGED; the same fixtures at the correct scale are NOT.
  P3  An fs_ifr in the real archives that disagrees with the declared
      Delta_t is FLAGGED, and the matching case is not. This is the fault
      that silently changes how many seconds of biology a window spans.
  P4  A real cohort extracted at two different fs_ifr values is flagged as
      internally inconsistent before it is ever compared to anything.
  P5  A simulated trace too short for the window (an over-large trim) is
      flagged rather than silently dropped.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_preprocessing_parity as C  # noqa: E402

VERBOSE = False
DT, SM, NE = 0.01, 0.02, 9
W = int(round(180.0 / DT))          # 18000
GEOM = ["--w_size", str(DT), "--gaussian_window", str(SM),
        "--window_s", "180"]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def note(msg: str) -> None:
    if VERBOSE:
        print("      %s" % msg)


def quiet(fn, *a, **kw):
    """Run fn with stdout suppressed; return (result, captured_text)."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_real(root: str, n_cultures: int = 2, n_sub: int = 2,
              fs: float = 100.0, t_rec: float = 1200.0,
              scale: float = 1.0, odd_one_fs: float = None) -> str:
    """A miniature real cohort; returns the specs path."""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    k = int(t_rec * fs)
    specs = []
    n = 0
    for c in range(n_cultures):
        for s in range(n_sub):
            x = np.abs(rng.normal(0.04, 0.02, size=k)).astype(np.float32) * scale
            p = os.path.join(root, "c%d_s%d.npz" % (c, s))
            this_fs = fs
            if odd_one_fs is not None and n == 0:
                this_fs = odd_one_fs
            np.savez_compressed(p, X=x, ifr_trace=x, fs_ifr=float(this_fs),
                                T_rec=float(t_rec), subregion_index=s,
                                culture_id="ptrain_A%d" % c)
            specs.append({"path": p, "name": "c%d_s%02d" % (c, s),
                          "condition": c % 2, "culture": "B__ptrain_A%d" % c})
            n += 1
    sp = os.path.join(root, "specs.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(specs, fh)
    return sp


def make_sim(root: str, n_files: int = 4, t_sim: float = 200.0,
             rate_hz: float = 4.0, seed: int = 1) -> Tuple[str, str]:
    """Detections whose pooled IFR sits at the same scale as make_real.

    Returns (mea_out_dir, campaign_dir).
    """
    mea = os.path.join(root, "mea", "topo_00000")
    camp = os.path.join(root, "camp")
    os.makedirs(mea, exist_ok=True)
    os.makedirs(camp, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        t, ch = [], []
        for e in range(NE):
            st = np.sort(rng.uniform(0.0, t_sim, size=rng.poisson(rate_hz * t_sim)))
            t.append(st)
            ch.append(np.full(st.size, e))
        np.savez_compressed(os.path.join(mea, "mea_iter_%05d.npz" % i),
                            det_t=np.concatenate(t),
                            det_ch=np.concatenate(ch).astype(np.int64))
    with open(os.path.join(camp, "job_args.json"), "w", encoding="utf-8") as fh:
        json.dump({"simtime": float(t_sim)}, fh)
    return os.path.join(root, "mea"), camp


def run_checker(specs: str, mea: str, camp: str, trim: float = 20.0,
                extra: List[str] = None) -> Tuple[int, List[str], str]:
    argv = (["--mode", "all"] + GEOM +
            ["--specs", specs, "--mea_out", mea, "--campaign", camp,
             "--trim_head_s", str(trim), "--n_sample", "8"] + (extra or []))
    rc, out = quiet(C.main, argv)
    return rc, list(C.FAILURES), out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_P0(root: str) -> None:
    check(C.n_windows(120000, 18000, 18000) == 6, "real windowing wrong")
    check(C.n_windows(18000, 18000, 18000) == 1, "exact fit must give 1")
    check(C.n_windows(17999, 18000, 18000) == 0, "below W must give 0")
    check(C.n_windows(20000, 18000, 18000) == 1, "200 s untrimmed gives 1")
    check(C.n_windows(0, 18000, 18000) == 0, "empty trace must give 0")
    note("120000->6, 18000->1, 17999->0, 20000->1, 0->0")


def test_P1(root: str) -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s = C.summarise(x)
    check(s["n"] == 5.0, "n wrong: %r" % s["n"])
    check(abs(s["mean"] - 2.0) < 1e-12, "mean wrong: %r" % s["mean"])
    check(abs(s["p50"] - 2.0) < 1e-12, "p50 wrong: %r" % s["p50"])
    check(abs(s["max"] - 4.0) < 1e-12, "max wrong: %r" % s["max"])
    check(abs(s["zero_frac"] - 0.2) < 1e-12,
          "zero fraction wrong: %r" % s["zero_frac"])
    check(abs(s["std"] - np.std(x)) < 1e-12, "std wrong")
    note("mean 2.0, p50 2.0, max 4.0, zeros 20 percent on a known array")


def test_P2(root: str) -> None:
    # NEGATIVE CONTROL: real and sim at the same scale.
    specs = make_real(os.path.join(root, "ok"))
    mea, camp = make_sim(os.path.join(root, "ok"))
    rc, fails, out = run_checker(specs, mea, camp)
    ratio_fails = [f for f in fails if "ratio" in f]
    check(not ratio_fails,
          "the correctly scaled arm was flagged: %r" % ratio_fails)
    note("correct scale: no ratio failure (negative control)")

    # POSITIVE: the real arm divided by n_e, i.e. the simulated arm is n_e
    # times too tall. This is the pooling-convention bug.
    specs9 = make_real(os.path.join(root, "bug"), scale=1.0 / NE)
    mea9, camp9 = make_sim(os.path.join(root, "bug"))
    rc9, fails9, out9 = run_checker(specs9, mea9, camp9)
    ratio_fails9 = [f for f in fails9 if "ratio" in f]
    check(ratio_fails9,
          "a factor-of-%d scale error was NOT flagged. Failures seen: %r"
          % (NE, fails9))
    check(rc9 != 0, "the checker must exit non-zero when a check fails")
    check(any("n_e" in f for f in ratio_fails9),
          "the n_e-specific message did not fire: %r" % ratio_fails9)
    note("scale x%d: flagged, %d ratio failure(s), exit code %d"
         % (NE, len(ratio_fails9), rc9))


def test_P3(root: str) -> None:
    # POSITIVE: archives say 50 Hz, the declared Delta_t says 100 Hz.
    specs = make_real(os.path.join(root, "fsbad"), fs=50.0, t_rec=1200.0)
    mea, camp = make_sim(os.path.join(root, "fsbad"))
    rc, fails, out = run_checker(specs, mea, camp)
    check(any("fs_ifr matches" in f for f in fails),
          "an fs_ifr mismatch was not flagged: %r" % fails)
    check("spans" in out,
          "the message must explain that W is a count of SAMPLES, so a "
          "wrong fs silently changes the seconds of biology per window")
    note("fs 50 vs declared 100: flagged, with the seconds-per-window "
         "explanation")

    # NEGATIVE CONTROL: identical fixtures at the right rate.
    specs_ok = make_real(os.path.join(root, "fsok"), fs=100.0)
    mea2, camp2 = make_sim(os.path.join(root, "fsok"))
    rc2, fails2, _ = run_checker(specs_ok, mea2, camp2)
    check(not any("fs_ifr" in f for f in fails2),
          "the matching case was flagged: %r" % fails2)
    note("fs 100 vs declared 100: quiet (negative control)")


def test_P4(root: str) -> None:
    specs = make_real(os.path.join(root, "mixed"), fs=100.0, odd_one_fs=50.0)
    mea, camp = make_sim(os.path.join(root, "mixed"))
    rc, fails, _ = run_checker(specs, mea, camp)
    check(any("one fs_ifr across" in f for f in fails),
          "a cohort with two fs_ifr values was not flagged: %r" % fails)
    note("mixed 50/100 Hz cohort: flagged as internally inconsistent")


def test_P5(root: str) -> None:
    specs = make_real(os.path.join(root, "short"))
    mea, camp = make_sim(os.path.join(root, "short"))
    # 200 s - 25 s = 175 s < 180 s window: every simulated trace is too short.
    rc, fails, out = run_checker(specs, mea, camp, trim=25.0)
    check(any("usable duration" in f or "exactly one window" in f
              for f in fails),
          "an over-large trim was not flagged: %r" % fails)
    check(rc != 0, "the checker must exit non-zero here")
    note("trim 25 s of a 200 s run against a 180 s window: flagged")

    rc2, fails2, _ = run_checker(specs, mea, camp, trim=20.0)
    check(not any("usable duration" in f for f in fails2),
          "the 20 s trim was flagged: %r" % fails2)
    note("trim 20 s: quiet (negative control)")


TESTS = [
    ("P0", "n_windows arithmetic", test_P0),
    ("P1", "summarise reports what it claims", test_P1),
    ("P2", "a factor-of-n_e scale error is caught; correct scale is not",
     test_P2),
    ("P3", "an fs_ifr mismatch is caught; a match is not", test_P3),
    ("P4", "a cohort with two fs_ifr values is caught", test_P4),
    ("P5", "an over-large trim is caught; the right trim is not", test_P5),
]


def main(argv=None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="smoke tests for check_preprocessing_parity.py")
    ap.add_argument("-k", dest="selector", default=None)
    ap.add_argument("-v", dest="verbose", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = bool(args.verbose)

    selected = [t for t in TESTS
                if args.selector is None or args.selector in t[0]]
    if not selected:
        print("no test matches %r" % args.selector)
        return 2

    n_pass = 0
    for tid, desc, fn in selected:
        root = tempfile.mkdtemp(prefix="pp_%s_" % tid)
        try:
            fn(root)
            print("PASS  %s  %s" % (tid, desc))
            n_pass += 1
        except Exception:                    # noqa: BLE001
            print("FAIL  %s  %s" % (tid, desc))
            traceback.print_exc()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print("")
    print("%d / %d passed" % (n_pass, len(selected)))
    return 0 if n_pass == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
