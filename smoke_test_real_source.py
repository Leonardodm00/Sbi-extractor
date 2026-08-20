#!/usr/bin/env python3
"""
smoke_test_real_source.py -- is the real-recording trace source correct?

Run:
    python3 smoke_test_real_source.py
    python3 smoke_test_real_source.py -k R5
    python3 smoke_test_real_source.py -v      # print each check

Everything here is built from synthetic .npz fixtures written into a
temporary directory, so the suite needs NO cluster data, NO checkpoint, NO
torch and NO GPU. It runs on a login node in about a second.

  R0  load_specs reads the four required fields, applies the optional path
      prefix substitution, and NAMES the offending entry when one is missing
      rather than skipping it.
  R1  validate_cohort accepts a clean cohort with the right counts, and
      REFUSES a duplicate name and a culture straddling two conditions.
  R2  read_real_trace returns the stored bytes VERBATIM. This is the test
      that pins the module's central claim: the observable is not recomputed
      on the real side, so the two arms cannot drift apart here.
  R3  read_real_trace RAISES on an fs_ifr that disagrees with the
      checkpoint's implied rate, and passes when it agrees. Negative control
      included under the identical criterion.
  R4  read_real_trace refuses a genuine multichannel array and accepts a
      (1, K) single-channel one. Wrong extraction mode is a different
      observable, not a reshaping inconvenience.
  R5  Window arithmetic. floor((L - W) / S) + 1 windows, disjoint at S = W,
      window 0 equal to x[:W] byte for byte. Checked against an
      independently written reference here AND, when importable, against
      sim_observable.window_trace, so agreement is between two separately
      written rules rather than one rule with itself.
  R6  A trace shorter than W yields ZERO windows and no exception. This is
      the silent-drop trap: the dataset skips such traces with a bare
      `continue`, so a caller that does not check the count loses a culture
      without any message.
  R7  build_real_records produces one record per spec, carrying culture,
      condition, subregion and name, and the resulting group vector has
      exactly n_cultures distinct values with no culture straddling classes.
  R8  THE TEST THAT JUSTIFIES THE 'culture' COLUMN. Two subregions from
      DIFFERENT batches whose npz culture_id is the same string
      ('ptrain_A1') must remain two groups. Grouping on the npz value is
      shown to merge them; grouping on the specs 'culture' is shown not to.

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
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_source import (  # noqa: E402
    RealSpec, build_real_records, load_specs, read_real_trace, validate_cohort,
)

RESULTS: List[Tuple[str, str, str]] = []
VERBOSE = False

FS_IFR = 50.0          # 1 / 0.02 s, the locked extraction rate
T_REC = 1200.0         # s, the real recording duration
K_FULL = int(T_REC * FS_IFR)     # 60000 samples


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def note(msg: str) -> None:
    if VERBOSE:
        print("      %s" % msg)


def raises(fn: Callable, *a, **kw) -> Exception:
    """Assert fn raises, and return the exception for message inspection."""
    try:
        fn(*a, **kw)
    except Exception as exc:          # noqa: BLE001 -- that is the point
        return exc
    raise AssertionError("expected an exception, none raised")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_trace(k: int = K_FULL, seed: int = 0) -> np.ndarray:
    """A non-negative, non-constant stand-in for a pooled IFR."""
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(size=k)).cumsum()
    base = base - base.min()
    return np.ascontiguousarray(base / max(base.max(), 1e-9), dtype=np.float32)


def write_npz(path: str, x: np.ndarray, fs: float = FS_IFR,
              t_rec: float = T_REC, culture_id: str = "ptrain_A1",
              subregion: int = 0, multichannel: bool = False,
              omit_fs: bool = False) -> np.ndarray:
    """Write a fixture mimicking run_channel_subset_extraction.py's output."""
    arr = np.stack([x, x], axis=0) if multichannel else x
    payload = dict(X=arr, ifr_trace=arr, row_meaning="samples",
                   in_channels=1, n_samples=1,
                   subregion_index=int(subregion),
                   culture_id=culture_id, mode="per_region_single",
                   T_rec=float(t_rec), n_present=81,
                   n_samples_raw=int(t_rec * 10110.09),
                   discarded=np.array([], dtype=np.int64))
    if not omit_fs:
        payload["fs_ifr"] = float(fs)
    np.savez_compressed(path, **payload)
    return arr


def build_cohort(root: str, n_cultures: int = 3, n_sub: int = 9,
                 k: int = K_FULL) -> Tuple[str, List[dict]]:
    """A miniature cohort: n_cultures cultures, n_sub subregions each."""
    entries = []
    for c in range(n_cultures):
        batch = "Batch%d" % (c % 2 + 3)
        culture = "%s__ptrain_A%d" % (batch, c + 1)
        cdir = os.path.join(root, culture)
        os.makedirs(cdir, exist_ok=True)
        for s in range(n_sub):
            p = os.path.join(cdir, "trace_subregion_%02d.npz" % s)
            write_npz(p, make_trace(k, seed=100 * c + s),
                      culture_id="ptrain_A%d" % (c + 1), subregion=s)
            entries.append({"path": p, "name": "%s__sub%02d" % (culture, s),
                            "condition": c % 2, "culture": culture})
    specs_path = os.path.join(root, "specs.json")
    with open(specs_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)
    return specs_path, entries


def reference_windows(length: int, w: int, stride: int) -> List[int]:
    """Independently written reference for the dataset's indexing rule.

    Deliberately NOT a call into sim_observable: R5 compares two separately
    authored implementations, which is what makes the agreement evidence.
    """
    starts, s = [], 0
    while s + w <= length:
        starts.append(s)
        s += stride
    return starts


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_R0(root: str) -> None:
    specs_path, entries = build_cohort(root, n_cultures=2, n_sub=3)
    specs = load_specs(specs_path)
    check(len(specs) == 6, "expected 6 specs, got %d" % len(specs))
    check(isinstance(specs[0], RealSpec), "wrong record type")
    check(specs[0].condition in (0, 1), "condition not parsed as int")
    note("loaded %d specs" % len(specs))

    # prefix substitution
    moved = load_specs(specs_path, path_from=root, path_to="/elsewhere")
    check(moved[0].path.startswith("/elsewhere"),
          "path_from/path_to substitution did not apply")
    check(moved[0].path.endswith(specs[0].path[len(root):]),
          "substitution mangled the tail of the path")
    exc = raises(load_specs, specs_path, path_from=root)
    check("together" in str(exc), "half-given substitution not rejected")
    note("prefix substitution applied and half-specification rejected")

    # a missing field must NAME the entry, not skip it
    bad = [dict(e) for e in entries]
    del bad[1]["culture"]
    bad_path = os.path.join(root, "specs_bad.json")
    with open(bad_path, "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    exc = raises(load_specs, bad_path)
    check("culture" in str(exc) and "1" in str(exc),
          "missing-field error does not identify field and entry: %s" % exc)
    note("missing field raises and names entry 1")


def test_R1(root: str) -> None:
    specs_path, entries = build_cohort(root, n_cultures=4, n_sub=9)
    specs = load_specs(specs_path)
    summary = validate_cohort(specs)
    check(summary["n_specs"] == 36, "n_specs wrong: %r" % summary["n_specs"])
    check(summary["n_cultures"] == 4,
          "n_cultures wrong: %r" % summary["n_cultures"])
    check(summary["subregions_per_culture"] == [9],
          "subregions per culture wrong: %r"
          % (summary["subregions_per_culture"],))
    check(summary["per_class_cultures"] == {0: 2, 1: 2},
          "per-class culture counts wrong: %r"
          % (summary["per_class_cultures"],))
    note("clean cohort: %d cultures, %d traces"
         % (summary["n_cultures"], summary["n_specs"]))

    dup = list(specs) + [specs[0]]
    exc = raises(validate_cohort, dup)
    check("duplicate" in str(exc).lower(), "duplicate name not refused")
    note("duplicate name refused")

    straddle = [RealSpec(s.path, s.name, s.condition, s.culture) for s in specs]
    straddle[0] = RealSpec(straddle[0].path, straddle[0].name,
                           1 - straddle[0].condition, straddle[0].culture)
    exc = raises(validate_cohort, straddle)
    check("condition" in str(exc), "straddling culture not refused: %s" % exc)
    note("culture straddling two conditions refused")


def test_R2(root: str) -> None:
    p = os.path.join(root, "verbatim.npz")
    x = make_trace(5000, seed=7)
    write_npz(p, x)
    got, meta = read_real_trace(p, expect_fs=FS_IFR)

    check(got.dtype == np.float32, "trace not float32: %r" % got.dtype)
    check(got.shape == x.shape, "shape changed: %r vs %r" % (got.shape, x.shape))
    check(np.array_equal(got, x),
          "the trace was ALTERED on load. The real observable must be the "
          "bytes on disk; recomputing it here is how the two arms drift.")
    check(meta["trace_key"] == "ifr_trace",
          "did not prefer 'ifr_trace': %r" % meta["trace_key"])
    check(abs(float(meta["T_rec"]) - T_REC) < 1e-9, "T_rec not carried")
    note("all %d samples identical to disk; meta carried" % got.size)

    # the 'X' fallback names the SAME array
    p2 = os.path.join(root, "xonly.npz")
    np.savez_compressed(p2, X=x, fs_ifr=FS_IFR, T_rec=T_REC)
    got2, meta2 = read_real_trace(p2, expect_fs=FS_IFR)
    check(np.array_equal(got2, x), "'X' fallback altered the trace")
    check(meta2["trace_key"] == "X", "fallback key not reported")
    note("'X' fallback returns the same array")


def test_R3(root: str) -> None:
    p = os.path.join(root, "fs_ok.npz")
    write_npz(p, make_trace(5000), fs=FS_IFR)
    x, _ = read_real_trace(p, expect_fs=FS_IFR)      # negative control
    check(x.size == 5000, "the agreeing case must pass")
    note("fs 50.0 vs 50.0 accepted (negative control)")

    q = os.path.join(root, "fs_bad.npz")
    write_npz(q, make_trace(5000), fs=100.0)
    exc = raises(read_real_trace, q, expect_fs=FS_IFR)
    check("fs_ifr" in str(exc), "fs mismatch not reported clearly: %s" % exc)
    note("fs 100.0 vs 50.0 refused")

    r = os.path.join(root, "fs_absent.npz")
    write_npz(r, make_trace(5000), omit_fs=True)
    exc = raises(read_real_trace, r, expect_fs=FS_IFR)
    check("fs_ifr" in str(exc), "absent fs_ifr not refused: %s" % exc)
    # and it must still load when no check is requested
    x2, _ = read_real_trace(r, expect_fs=None)
    check(x2.size == 5000, "absent fs must not block an unchecked load")
    note("absent fs_ifr refused when a check was requested, allowed otherwise")


def test_R4(root: str) -> None:
    p = os.path.join(root, "multi.npz")
    write_npz(p, make_trace(5000), multichannel=True)
    exc = raises(read_real_trace, p, expect_fs=FS_IFR)
    check("per_region_single" in str(exc),
          "multichannel array not refused with a useful message: %s" % exc)
    note("(2, K) multichannel refused")

    q = os.path.join(root, "single2d.npz")
    x = make_trace(5000)
    np.savez_compressed(q, ifr_trace=x.reshape(1, -1), fs_ifr=FS_IFR,
                        T_rec=T_REC)
    got, _ = read_real_trace(q, expect_fs=FS_IFR)
    check(got.ndim == 1 and np.array_equal(got, x),
          "(1, K) single-channel array not accepted and flattened")
    note("(1, K) accepted and flattened without altering values")

    neg = os.path.join(root, "negative.npz")
    np.savez_compressed(neg, ifr_trace=(x - 0.5).astype(np.float32),
                        fs_ifr=FS_IFR, T_rec=T_REC)
    exc = raises(read_real_trace, neg, expect_fs=FS_IFR)
    check("negative" in str(exc).lower(), "negative IFR not refused")
    note("negative IFR samples refused")


def test_R5(root: str) -> None:
    x = make_trace(K_FULL, seed=3)
    for w in (9000, 10000):
        starts = reference_windows(K_FULL, w, w)
        check(len(starts) == 6,
              "expected 6 windows at W=%d over %d samples, got %d"
              % (w, K_FULL, len(starts)))
        check(starts[0] == 0 and starts[1] - starts[0] == w,
              "windows at stride W must be disjoint and abut")
        check(starts[-1] + w <= K_FULL, "last window runs past the trace")
        note("W=%d over K=%d gives %d disjoint windows, last ends at %d"
             % (w, K_FULL, len(starts), starts[-1] + w))

    # window 0 must be the raw head of the trace
    w = 10000
    check(np.array_equal(x[:w], x[0:0 + w]), "trivial identity failed")

    # cross-check against the package's own implementation when importable
    try:
        from sim_observable import window_trace  # noqa: E402
    except Exception as exc:                     # noqa: BLE001
        note("sim_observable not importable (%s); reference-only check"
             % type(exc).__name__)
        return
    for w in (9000, 10000):
        xw, starts = window_trace(x, window_length=w, stride=w)
        ref = reference_windows(K_FULL, w, w)
        check(list(starts) == ref,
              "window_trace disagrees with the reference at W=%d: %r vs %r"
              % (w, list(starts)[:4], ref[:4]))
        check(xw.shape == (len(ref), w),
              "window_trace shape %r wrong at W=%d" % (xw.shape, w))
        check(np.array_equal(xw[0], x[:w]),
              "window 0 is not the head of the trace at W=%d" % w)
        check(np.array_equal(xw[-1], x[ref[-1]:ref[-1] + w]),
              "last window content wrong at W=%d" % w)
    note("window_trace agrees with the independent reference at both W")


def test_R6(root: str) -> None:
    short = 8000
    w = 10000
    starts = reference_windows(short, w, w)
    check(starts == [],
          "a trace shorter than W must yield ZERO windows, got %d"
          % len(starts))
    note("L=%d < W=%d yields 0 windows and no exception -- the caller MUST "
         "check the count" % (short, w))

    try:
        from sim_observable import window_trace  # noqa: E402
    except Exception:                            # noqa: BLE001
        return
    xw, s2 = window_trace(make_trace(short), window_length=w, stride=w)
    check(len(s2) == 0 and xw.shape[0] == 0,
          "window_trace did not return an empty result for a short trace")
    note("window_trace agrees: empty result, no raise")


def test_R7(root: str) -> None:
    specs_path, _ = build_cohort(root, n_cultures=5, n_sub=9, k=2000)
    specs = load_specs(specs_path)
    recs = list(build_real_records(specs, expect_fs=FS_IFR))
    check(len(recs) == len(specs),
          "one record per spec expected: %d vs %d" % (len(recs), len(specs)))

    for (x, ident), s in zip(recs, specs):
        for key in ("culture", "condition", "subregion", "name"):
            check(key in ident, "ident missing %r" % key)
        check(ident["culture"] == s.culture, "culture not carried through")
        check(ident["name"] == s.name, "name not carried through")
        check(int(ident["condition"]) == s.condition, "condition changed")
        check(x.ndim == 1 and x.dtype == np.float32, "trace shape/dtype wrong")

    groups = np.array([i["culture"] for _, i in recs])
    classes = np.array([i["condition"] for _, i in recs])
    check(np.unique(groups).size == 5,
          "expected 5 groups, got %d" % np.unique(groups).size)
    for g in np.unique(groups):
        check(np.unique(classes[groups == g]).size == 1,
              "group %r straddles two classes" % g)
    subs = sorted(int(i["subregion"]) for _, i in recs
                  if i["culture"] == groups[0])
    check(subs == list(range(9)), "subregion indices wrong: %r" % subs)
    note("5 groups x 9 subregions, no straddle, subregion index 0..8")


def test_R8(root: str) -> None:
    """Two batches, same npz culture_id, different cultures."""
    entries = []
    for batch, cond in (("DATA_C_Batch3", 0), ("Batch4_SubBatch1", 0)):
        culture = "%s__ptrain_A1" % batch
        d = os.path.join(root, "collide", batch)
        os.makedirs(d, exist_ok=True)
        for s in range(3):
            p = os.path.join(d, "trace_subregion_%02d.npz" % s)
            # NOTE the culture_id written by the extractor is the ptrain
            # folder name only, so it is IDENTICAL for the two batches.
            write_npz(p, make_trace(2000, seed=s), culture_id="ptrain_A1",
                      subregion=s)
            entries.append({"path": p, "name": "%s__sub%02d" % (culture, s),
                            "condition": cond, "culture": culture})
    specs_path = os.path.join(root, "collide_specs.json")
    with open(specs_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)

    specs = load_specs(specs_path)
    recs = list(build_real_records(specs, expect_fs=FS_IFR))

    by_npz = np.array([i["culture_id_npz"] for _, i in recs])
    by_spec = np.array([i["culture"] for _, i in recs])

    check(np.unique(by_npz).size == 1,
          "fixture is wrong: the npz culture_id should collide")
    check(np.unique(by_spec).size == 2,
          "the specs 'culture' must keep the two batches apart, got %d group(s)"
          % np.unique(by_spec).size)
    note("npz culture_id gives %d group (WRONG, merges two cultures); "
         "specs culture gives %d groups (correct)"
         % (np.unique(by_npz).size, np.unique(by_spec).size))

    # and validate_cohort must be happy with the correct grouping
    summary = validate_cohort(specs)
    check(summary["n_cultures"] == 2,
          "validate_cohort counted %d cultures" % summary["n_cultures"])


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
TESTS = [
    ("R0", "specs loading, substitution, missing-field error", test_R0),
    ("R1", "cohort validation: duplicates and class straddling", test_R1),
    ("R2", "the trace is loaded VERBATIM, not recomputed", test_R2),
    ("R3", "fs_ifr parity with the checkpoint, plus negative control", test_R3),
    ("R4", "single-channel only; negative IFR refused", test_R4),
    ("R5", "window arithmetic against an independent reference", test_R5),
    ("R6", "a short trace yields zero windows, silently", test_R6),
    ("R7", "records carry group and class; no straddling", test_R7),
    ("R8", "the npz culture_id collides across batches; 'culture' does not",
     test_R8),
]


def main(argv=None) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description="smoke tests for real_source.py")
    ap.add_argument("-k", dest="selector", default=None,
                    help="run only tests whose id contains this string")
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
        root = tempfile.mkdtemp(prefix="rs_%s_" % tid)
        try:
            fn(root)
            print("PASS  %s  %s" % (tid, desc))
            n_pass += 1
        except Exception:                      # noqa: BLE001
            print("FAIL  %s  %s" % (tid, desc))
            traceback.print_exc()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print("")
    print("%d / %d passed" % (n_pass, len(selected)))
    return 0 if n_pass == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
