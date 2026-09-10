#!/usr/bin/env python3
"""
smoke_test_registry_from_manifest.py

Reproduces the failure that stopped the campaign_cadex_hhgap dry run and proves
the fix. No cluster, no checkpoint, no campaign data.

The failure, in one line: Sigma's declared bounds were [1.0, 10.0] when the data
was generated (exactly 1.000000 decades -> ln under rule (1)), and the live
simulator source now says [2.0, 10.0] (0.699 decades -> linear). The stored
theta holds ln(Sigma); a registry loaded from the live source reads it as a
natural value.

Run:
    python3 smoke_test_registry_from_manifest.py
    SMOKE_VERBOSE=1 python3 smoke_test_registry_from_manifest.py

Expect: ALL 9 CHECKS PASSED
"""

from __future__ import annotations

import os
import traceback

import numpy as np

from sbi_labels import (Registry, registry_from_manifest, build_label_spec,
                        assemble_theta_A)

VERBOSE = bool(os.environ.get("SMOKE_VERBOSE"))
PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("  PASS  %s" % name)
    except Exception as exc:
        FAILED.append((name, exc))
        print("  FAIL  %s : %s" % (name, exc))
        if VERBOSE:
            traceback.print_exc()


# --------------------------------------------------------------------------
# fixture: a 5-axis registry standing in for the real 37-D one, with Sigma
# parked exactly on rule (1)'s threshold the way the real one is.
# --------------------------------------------------------------------------
NAMES = ["Sigma", "g_ampa", "O_N", "DeltaT", "conn_dummy"]
UNITS = ["mV", "nS", "1/(uM*s)", "mV", "(dimensionless)"]

LIVE_BOUNDS = np.array([[2.0, 10.0],      # Sigma: 0.699 decades -> LINEAR
                        [0.05, 5.0],      # 2.0 decades -> ln
                        [0.03, 3.0],      # 2.0 decades -> ln
                        [2.0, 2.0],       # frozen
                        [0.1, 0.9]])      # 0.954 decades -> linear

DATA_BOUNDS = np.array([[1.0, 10.0],      # Sigma: exactly 1.0 decades -> ln
                        [0.05, 5.0],
                        [0.01, 3.0],      # wider, still ln
                        [2.0, 2.0],
                        [0.1, 0.9]])


def rule1(pb):
    return sorted(k for k in range(pb.shape[0])
                  if pb[k, 0] > 0 and pb[k, 1] > 0
                  and np.log10(pb[k, 1] / pb[k, 0]) >= 1.0)


def bounds_theta(pb, log_idx):
    bt = pb.copy()
    m = np.zeros(pb.shape[0], dtype=bool)
    m[log_idx] = True
    bt[m] = np.log(pb[m])
    return bt


def live_registry():
    L = rule1(LIVE_BOUNDS)
    return Registry(param_names=list(NAMES), param_units=list(UNITS),
                    param_bounds=LIVE_BOUNDS.copy(),
                    param_bounds_theta=bounds_theta(LIVE_BOUNDS, L),
                    log_param_indices=L, log_base="natural",
                    kernel_bounds=np.array([[0.1, 1.0], [60.0, 300.0], [1.0, 2.0]]),
                    sweep_groups={"synapse_astro": [0, 1, 2]})


def manifest(bounds=None, **over):
    pb = DATA_BOUNDS if bounds is None else bounds
    m = {"manifest_version": 3, "param_names": list(NAMES),
         "param_bounds": pb.tolist(),
         "log_params": [NAMES[k] for k in rule1(pb)],
         "log_transform": "natural",
         "active_indices": [0, 1, 2]}
    m.update(over)
    return m


def main():
    def T1():
        assert rule1(LIVE_BOUNDS) == [1, 2], rule1(LIVE_BOUNDS)
        assert rule1(DATA_BOUNDS) == [0, 1, 2], rule1(DATA_BOUNDS)
        assert abs(np.log10(10.0 / 1.0) - 1.0) < 1e-12   # exactly on threshold
    check("T1 Sigma is ln under the data bounds and linear under the live ones", T1)

    def T2():
        reg, changed, flipped = registry_from_manifest(live_registry(), manifest())
        assert reg.log_param_indices == [0, 1, 2], reg.log_param_indices
        assert changed == ["Sigma", "O_N"], changed
        assert flipped == ["Sigma"], flipped
    check("T2 re-sourcing reports the changed axes and the coordinate flip", T2)

    def T3():
        reg, _, _ = registry_from_manifest(live_registry(), manifest())
        assert np.allclose(reg.param_bounds[0], [1.0, 10.0])
        assert np.allclose(reg.param_bounds_theta[0], [np.log(1.0), np.log(10.0)])
        assert np.allclose(reg.param_bounds_theta[4], [0.1, 0.9])   # linear stays
    check("T3 bounds_theta is ln on log axes and natural on linear ones", T3)

    # The end-to-end reproduction: one row whose stored theta holds ln(Sigma).
    params = np.array([6.055741474273584, 1.5, 0.5, 2.0, 0.3])
    theta_data = params.copy()
    for k in rule1(DATA_BOUNDS):
        theta_data[k] = np.log(params[k])

    def T4():
        spec = build_label_spec(live_registry(), [0, 1, 2], "synapse_astro",
                                conn_prob_bounds=(0.05, 0.4),
                                topology_axes=["conn_prob"])
        try:
            assemble_theta_A(spec, theta_data, {"conn_prob": 0.2},
                             params_36=params)
        except ValueError as exc:
            assert "A6 failed" in str(exc) and "Sigma" in str(exc), str(exc)
            return
        raise AssertionError("expected A6 to fire against the live registry")
    check("T4 the live registry reproduces the exact A6 failure on Sigma", T4)

    def T5():
        reg, _, _ = registry_from_manifest(live_registry(), manifest())
        spec = build_label_spec(reg, [0, 1, 2], "synapse_astro",
                                conn_prob_bounds=(0.05, 0.4),
                                topology_axes=["conn_prob"])
        row = assemble_theta_A(spec, theta_data, {"conn_prob": 0.2},
                               params_36=params)
        assert row.shape == (4,), row.shape
        assert np.isclose(row[0], np.log(params[0])), row[0]
    check("T5 the manifest-sourced registry passes A6 and keeps the stored ln", T5)

    def T6():
        reg, _, _ = registry_from_manifest(live_registry(), manifest())
        spec = build_label_spec(reg, [0, 1, 2], "synapse_astro",
                                conn_prob_bounds=(0.05, 0.4),
                                topology_axes=["conn_prob"])
        assert spec.coord[0] == "ln", spec.coord
        lo, hi = spec.bounds_theta[0]
        v = np.log(params[0])
        assert lo <= v <= hi, (lo, v, hi)     # A5 would pass
    check("T6 the row lands inside the declared box (A5 would pass)", T6)

    def T7():
        spec = build_label_spec(live_registry(), [0, 1, 2], "synapse_astro",
                                conn_prob_bounds=(0.05, 0.4),
                                topology_axes=["conn_prob"])
        lo, hi = spec.bounds_theta[0]
        assert not (lo <= np.log(params[0]) <= hi), (lo, hi)
    check("T7 without the fix the same value falls OUTSIDE the live box", T7)

    def T8():
        bad = manifest()
        bad["log_params"] = ["g_ampa", "O_N"]        # omits Sigma
        try:
            registry_from_manifest(live_registry(), bad)
        except ValueError as exc:
            assert "internally inconsistent" in str(exc), str(exc)
            return
        raise AssertionError("expected the manifest cross-check to fire")
    check("T8 a manifest whose log_params contradict its own bounds is refused", T8)

    def T9():
        reg = live_registry()
        same, changed, flipped = registry_from_manifest(reg, {"active_indices": [0]})
        assert same is reg and changed == [] and flipped == []
        try:
            registry_from_manifest(reg, manifest(bounds=DATA_BOUNDS[:3]))
        except ValueError as exc:
            assert "width mismatch" in str(exc) or "registry is" in str(exc), str(exc)
            return
        raise AssertionError("expected a width mismatch to raise")
    check("T9 no param_bounds is a no-op; a wrong-width one raises", T9)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
