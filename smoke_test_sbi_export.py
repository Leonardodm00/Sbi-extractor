#!/usr/bin/env python3
"""
smoke_test_sbi_export.py
========================

Debugging + correctness harness for the SBI export chain. Everything needed to
run it is generated here: no campaign directory, no checkpoint and no real
recordings are required. Where a repository IS available the test uses it and
checks BIT-PARITY against it; where it is not, that test is skipped loudly
rather than silently passing.

RUN
---
    # minimal (no repositories needed; parity + registry tests skip)
    python3 smoke_test_sbi_export.py

    # full (recommended: exercises the real code paths). The DSN tree is
    # <SBI_HPC_DIR>/dsn via dsn_tree.py (env.sh; artifacts/sbi_hpc);
    # --dsn_main_dir overrides it, e.g. to point at a stub.
    python3 smoke_test_sbi_export.py \
        --sim_dir      /path/to/Astro-Neuron-Network/hpc/Phenomenological_finalv1

    # equivalently, via the environment
    export SIM_MAIN_DIR=/path/to/Astro-Neuron-Network/hpc/Phenomenological_finalv1
    python3 smoke_test_sbi_export.py

Exit code 0 = every non-skipped test passed. Non-zero = at least one failed.

WHAT EACH TEST ESTABLISHES
--------------------------
T1  windowing matches MEAWindowDataset's index rule exactly, including the
    silent-drop case L < W.
T2  build_pooled_ifr reproduces the DSN tree's own compute_ifr_trace bit for
    bit, up to the per-electrode division. This is the test that settles the
    pooling-convention question; everything else about scale parity is opinion
    without it.                                          [needs DSN tree]
T2b the NORMALISED sim-arm observable equals the real-arm extractor's recipe
    (compute_ifr_trace, then (ifr / n_e).astype(float32)) bit for bit. Before
    migration step 4b this FAILED on every random trial (last-bit differences
    from dividing float64 vs float32).                    [needs DSN tree]
T2c a spike at exactly t = T is treated as the extractor treats it (counted in
    the last bin). Before step 4b it was discarded.       [needs DSN tree]
T3  per-electrode normalisation is exactly a factor 1/n_e, and the un-normalised
    variant is n_e times larger -- the amplitude error the handoff would cause.
                                                          [needs DSN tree]
T4  the bin grid is driven by the DECLARED T, not by the data, so two traces
    with different last-spike times still give the same K. This is the
    process_campaign 'inferred simtime' trap.
T5  zraw is genuinely the pre-normalisation activation: normalize(zraw) == z.
T6  Z is invariant to batch_size (GroupNorm, no batch statistics).
T7  embed() refuses a window-length mismatch instead of silently resampling.
T8  the checkpoint round trip: save -> load_frozen_dsn -> identical embeddings,
    and the SHA-256 matches an independent digest.       [needs DSN tree]
T9  structural invariants of the registry and the label spec, asserted as
    relations and never as counts: rule (1) re-derived here reproduces L, no
    point-interval axis is active, p = len(active) + len(topology), and the
    topology block is last and linear (so p0_conn is linear whether or not
    rule (1) would have called it a log axis). The census -- n, |L|, the
    active ln/linear split, p -- is REPORTED on the PASS line. [needs sim repo]
T9b every active axis's prior box is the coordinate rule applied to its
    natural box, the bounds-level form of assertion A6.     [needs sim repo]
T10 assertion A1, the transform round trip at both bounds edges. [needs sim repo]
T11 a degenerate (frozen) axis is rejected by A3 at label construction.
T12 end-to-end export: Parquet + sidecar, correct columns, A5 catches an
    out-of-box row, A4 catches a constant column.

HPC note (hpc-python-compat): pure ASCII, LF-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------- #
# tiny test harness
# --------------------------------------------------------------------------- #
class Results:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def ok(self, name, detail=""):
        self.passed.append((name, detail))
        print("  [PASS] %-8s %s" % (name, detail))

    def fail(self, name, detail):
        self.failed.append((name, detail))
        print("  [FAIL] %-8s %s" % (name, detail))

    def skip(self, name, why):
        self.skipped.append((name, why))
        print("  [SKIP] %-8s %s" % (name, why))

    def report(self):
        print("\n" + "=" * 70)
        print("PASSED %d   FAILED %d   SKIPPED %d"
              % (len(self.passed), len(self.failed), len(self.skipped)))
        if self.skipped:
            print("\nskipped:")
            for n, w in self.skipped:
                print("  %-8s %s" % (n, w))
        if self.failed:
            print("\nFAILURES:")
            for n, d in self.failed:
                print("  %-8s %s" % (n, d))
        print("=" * 70)
        return 1 if self.failed else 0


def _run(res, name, fn):
    try:
        detail = fn()
        res.ok(name, detail if detail else "")
    except _Skip as exc:
        res.skip(name, str(exc))
    except Exception as exc:                                    # noqa: BLE001
        res.fail(name, "%s: %s" % (type(exc).__name__, exc))
        if os.environ.get("SMOKE_VERBOSE"):
            traceback.print_exc()


class _Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def make_bursty_spikes(rng, T=180.0, n_electrodes=9, rate_bursts=0.3):
    """Per-electrode detected-spike-time arrays with a shared burst schedule.

    Not biologically faithful; it only needs to be non-uniform in time so the
    Gaussian smoothing and the burst structure are actually exercised.
    """
    n_b = max(1, int(rng.poisson(rate_bursts * T)))
    centers = np.sort(rng.uniform(0.0, T, n_b))
    per_e = []
    for _e in range(n_electrodes):
        parts = [rng.uniform(0.0, T, int(rng.poisson(2.0 * T)))]  # background
        for c in centers:
            k = int(rng.poisson(12))
            if k:
                parts.append(np.clip(rng.normal(c, 0.05, k), 0.0, T - 1e-9))
        st = np.sort(np.concatenate(parts))
        per_e.append(st.astype(np.float64))
    return per_e


def make_tiny_backbone(dsn_main_dir, E=16, W=9000, in_channels=1):
    """Build a small OneDCNNBackbone with the real class but a cheap depth."""
    sys.path.insert(0, dsn_main_dir)
    from backbone import BackboneConfig, build_backbone
    cfg = BackboneConfig(depth_exponent=2, width_multiplier=2.0, stem_width=8,
                         in_channels=in_channels, embedding_size=E,
                         l2_normalize=True, head_fusion=True,
                         head_pool_ops=("mean",), dropout=0.0)
    return build_backbone(cfg), cfg


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_T1_windowing():
    from sim_observable import window_trace
    W = 9000
    x = np.zeros(9000, dtype=np.float32)
    Xw, starts = window_trace(x, W)
    assert Xw.shape == (1, W) and starts == [0], "exact-length trace"

    x = np.zeros(8999, dtype=np.float32)
    Xw, starts = window_trace(x, W)
    assert Xw.shape[0] == 0 and starts == [], "L < W must yield zero windows"

    x = np.arange(60000, dtype=np.float32)
    Xw, starts = window_trace(x, W)          # stride defaults to W
    assert len(starts) == (60000 - W) // W + 1 == 6, "1200 s -> 6 windows"
    assert np.array_equal(Xw[2], x[2 * W:3 * W]), "window content/offset"

    xc = np.zeros((9, 60000), dtype=np.float32)
    Xc, _ = window_trace(xc, W)
    assert Xc.shape == (6, 9, W), "multichannel windows keep the channel axis"
    return "1200 s -> 6 disjoint windows; L<W -> 0"


def test_T2_ifr_parity(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (needed for compute_ifr_trace)")
    from sim_observable import build_pooled_ifr, reference_compute_ifr_trace

    rng = np.random.default_rng(11)
    T, dt, sig, n_e = 180.0, 0.02, 0.04, 9
    per_e = make_bursty_spikes(rng, T=T, n_electrodes=n_e)

    ref, fs_ref = reference_compute_ifr_trace(per_e, T, dt, sig,
                                              dsn_main_dir=dsn_main_dir)
    ours_unnorm = build_pooled_ifr(per_e, n_electrodes=n_e, T=T, dt=dt,
                                   sigma_sm=sig, normalise_per_electrode=False)
    if ours_unnorm.shape != ref.shape:
        raise AssertionError("K mismatch: ours %r ref %r"
                             % (ours_unnorm.shape, ref.shape))
    if not np.array_equal(ours_unnorm, ref):
        worst = float(np.max(np.abs(ours_unnorm.astype(np.float64)
                                    - ref.astype(np.float64))))
        raise AssertionError("not bit-identical to compute_ifr_trace; max |d| "
                             "= %.3e" % worst)
    if abs(fs_ref - 1.0 / dt) > 1e-9:
        raise AssertionError("fs_ifr mismatch")
    return "bit-identical to compute_ifr_trace, K=%d, fs=%.4g Hz" % (ref.size, fs_ref)


def _extractor_recipe(per_e, n_e, T, dt, sig, dsn_main_dir):
    """The real arm's observable, spelled exactly as
    extractor/channel_subset_extraction.py::subregion_ifr spells it."""
    from dataclasses import replace
    if dsn_main_dir and dsn_main_dir not in sys.path:
        sys.path.insert(0, dsn_main_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    from generate_burst_data import CONTROL_PARAMS, compute_ifr_trace
    params = replace(CONTROL_PARAMS, duration_s=float(T), w_size=float(dt),
                     gaussian_window=float(sig))
    ifr, _fs = compute_ifr_trace([np.asarray(g, dtype=np.float64) for g in per_e],
                                 params)
    return (ifr / float(n_e)).astype(np.float32)


def test_T2b_normalised_parity(dsn_main_dir):
    """Sim arm == real arm on the NORMALISED trace, bit for bit, on random
    data. The guard this is: with the pre-4b re-implementation this failed on
    50/50 random trials (worst |d| = 3e-8), because it divided the float64
    trace and the extractor divides the float32 one."""
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (needed for compute_ifr_trace)")
    from sim_observable import build_pooled_ifr
    rng = np.random.default_rng(22)
    T, dt, sig, n_e = 180.0, 0.01, 0.02, 9
    n_diff = 0
    for _trial in range(20):
        per_e = make_bursty_spikes(rng, T=T, n_electrodes=n_e)
        ours = build_pooled_ifr(per_e, n_electrodes=n_e, T=T, dt=dt, sigma_sm=sig)
        real = _extractor_recipe(per_e, n_e, T, dt, sig, dsn_main_dir)
        if ours.dtype != np.float32 or ours.shape != real.shape:
            raise AssertionError("dtype/shape: %r %r vs %r %r"
                                 % (ours.dtype, ours.shape, real.dtype, real.shape))
        if not np.array_equal(ours, real):
            n_diff += 1
    if n_diff:
        raise AssertionError("normalised sim-arm trace differs from the "
                             "extractor recipe on %d/20 trials" % n_diff)
    return "20/20 random trials bit-identical to the extractor recipe (dt=%g, sigma=%g)" % (dt, sig)


def test_T2c_edge_spike_at_T(dsn_main_dir):
    """A spike at exactly t = T: the extractor's np.histogram closes the last
    bin on the right and counts it; the pre-4b re-implementation clipped the
    domain to [0, T) and discarded it. Both arms must now agree."""
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (needed for compute_ifr_trace)")
    from sim_observable import build_pooled_ifr
    T, dt, sig, n_e = 180.0, 0.01, 0.02, 9
    per_e = [np.array([T], dtype=np.float64)] + [np.zeros(0) for _ in range(n_e - 1)]
    ours = build_pooled_ifr(per_e, n_electrodes=n_e, T=T, dt=dt, sigma_sm=sig)
    real = _extractor_recipe(per_e, n_e, T, dt, sig, dsn_main_dir)
    if not np.array_equal(ours, real):
        raise AssertionError("edge spike at t=T: sim mass %.6f vs extractor mass %.6f"
                             % (float(ours.sum()), float(real.sum())))
    if float(real.sum()) <= 0.0:
        raise AssertionError("the extractor recipe should count the t=T spike; "
                             "it did not (mass %.3e)" % float(real.sum()))
    return "t=T spike counted on both arms (mass %.6f)" % float(ours.sum())


def test_T3_per_electrode_normalisation(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (build_pooled_ifr calls compute_ifr_trace)")
    from sim_observable import build_pooled_ifr
    rng = np.random.default_rng(3)
    T, dt, sig, n_e = 60.0, 0.02, 0.04, 9
    per_e = make_bursty_spikes(rng, T=T, n_electrodes=n_e)

    norm = build_pooled_ifr(per_e, n_e, T, dt, sig, normalise_per_electrode=True)
    raw = build_pooled_ifr(per_e, n_e, T, dt, sig, normalise_per_electrode=False)
    ratio = raw.astype(np.float64).sum() / max(norm.astype(np.float64).sum(), 1e-30)
    if not np.isclose(ratio, n_e, rtol=1e-6):
        raise AssertionError("expected a factor of n_e = %d, got %.6f"
                             % (n_e, ratio))
    if not np.all(norm >= 0.0):
        raise AssertionError("IFR must be non-negative")
    return ("handoff 'sum_over_electrodes' would be %dx too tall "
            "(mean peak %.3f vs %.3f)" % (n_e, float(norm.max()), float(raw.max())))


def test_T4_declared_duration_not_inferred(dsn_main_dir):
    """The simtime trap: two runs whose LAST SPIKE differs must still give the
    same K, because T is declared. This is what process_campaign.py's
    simtime = ceil(spk_t.max()) breaks."""
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (build_pooled_ifr calls compute_ifr_trace)")
    from sim_observable import build_pooled_ifr
    T, dt = 180.0, 0.02
    quiet = [np.array([1.0, 2.0, 3.0])]          # last spike at 3 s
    busy = [np.array([1.0, 179.5])]              # last spike at 179.5 s
    a = build_pooled_ifr(quiet, 9, T, dt, 0.04)
    b = build_pooled_ifr(busy, 9, T, dt, 0.04)
    if a.shape != b.shape:
        raise AssertionError("K depends on the data: %r vs %r" % (a.shape, b.shape))
    if a.shape[0] != int(T / dt):
        raise AssertionError("K = %d, expected floor(T/dt) = %d"
                             % (a.shape[0], int(T / dt)))

    # and the inferred-duration variant does NOT: this is the bug, exhibited.
    inferred = build_pooled_ifr(quiet, 9, float(np.ceil(3.0)), dt, 0.04)
    if inferred.shape[0] == a.shape[0]:
        raise AssertionError("the inferred-duration case should differ")
    return ("declared T -> K=%d for both; inferred simtime would give K=%d "
            "(dropped by the dataset)" % (a.shape[0], inferred.shape[0]))


def test_T5_zraw_is_prenorm(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable (needed for the real backbone)")
    import torch
    import torch.nn.functional as F
    from dsn_frozen import FrozenDSN

    W, E = 512, 16
    model, cfg = make_tiny_backbone(dsn_main_dir, E=E, W=W)
    dsn = FrozenDSN(model=model, device=torch.device("cpu"), ckpt_path=__file__,
                    ckpt_sha256="x" * 64, embedding_dim=E, l2_normalize=True,
                    in_channels=1, window_s=W * 0.02, w_size=0.02,
                    gaussian_window=0.04, window_length=W)
    rng = np.random.default_rng(5)
    X = np.abs(rng.normal(size=(23, W))).astype(np.float32)
    Z, Zraw = dsn.embed(X, batch_size=8, want_zraw=True)

    if Zraw is None:
        raise AssertionError("zraw was not captured")
    recon = F.normalize(torch.from_numpy(Zraw), p=2, dim=1).numpy()
    if not np.allclose(recon, Z, atol=1e-5):
        raise AssertionError("normalize(zraw) != z; max |d| = %.3e"
                             % float(np.max(np.abs(recon - Z))))
    norms = np.linalg.norm(Zraw, axis=1)
    if np.allclose(norms, 1.0, atol=1e-4):
        raise AssertionError("zraw is already unit-norm: the hook captured the "
                             "POST-normalisation tensor")
    return ("zraw recovered; ||zraw|| in [%.3f, %.3f] carries the amplitude "
            "that S^{E-1} discards" % (float(norms.min()), float(norms.max())))


def test_T6_batch_invariance(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable")
    import torch
    from dsn_frozen import FrozenDSN
    W, E = 512, 16
    model, _ = make_tiny_backbone(dsn_main_dir, E=E, W=W)
    dsn = FrozenDSN(model=model, device=torch.device("cpu"), ckpt_path=__file__,
                    ckpt_sha256="x" * 64, embedding_dim=E, l2_normalize=True,
                    in_channels=1, window_s=W * 0.02, w_size=0.02,
                    gaussian_window=0.04, window_length=W)
    rng = np.random.default_rng(6)
    X = np.abs(rng.normal(size=(37, W))).astype(np.float32)
    Z1, _ = dsn.embed(X, batch_size=1, want_zraw=False)
    Z2, _ = dsn.embed(X, batch_size=37, want_zraw=False)
    Z3, _ = dsn.embed(X, batch_size=5, want_zraw=False)
    d = max(float(np.max(np.abs(Z1 - Z2))), float(np.max(np.abs(Z1 - Z3))))
    if d > 1e-6:
        raise AssertionError("batch-size dependence: max |d| = %.3e" % d)

    # embed() must RESTORE the caller's mode, not force one. Calling this in the
    # middle of a training loop must not silently leave the model in eval mode.
    model.train()
    dsn.embed(X[:4], batch_size=2, want_zraw=False)
    if not model.training:
        raise AssertionError("embed() left a training-mode model in eval mode")
    model.eval()
    dsn.embed(X[:4], batch_size=2, want_zraw=False)
    if model.training:
        raise AssertionError("embed() left an eval-mode model in training mode")
    return "max |d| over batch sizes {1, 5, 37} = %.2e; caller mode restored" % d


def test_T7_window_length_guard(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable")
    import torch
    from dsn_frozen import FrozenDSN
    W, E = 512, 16
    model, _ = make_tiny_backbone(dsn_main_dir, E=E, W=W)
    dsn = FrozenDSN(model=model, device=torch.device("cpu"), ckpt_path=__file__,
                    ckpt_sha256="x" * 64, embedding_dim=E, l2_normalize=True,
                    in_channels=1, window_s=W * 0.02, w_size=0.02,
                    gaussian_window=0.04, window_length=W)
    try:
        dsn.embed(np.zeros((4, W - 1), dtype=np.float32))
    except ValueError as exc:
        if "window length" not in str(exc):
            raise AssertionError("wrong error: %s" % exc)
    else:
        raise AssertionError("a wrong window length was silently accepted")
    try:
        dsn.embed(np.zeros((4, 3, W), dtype=np.float32))
    except ValueError as exc:
        if "channel" not in str(exc):
            raise AssertionError("wrong error: %s" % exc)
    else:
        raise AssertionError("a wrong channel count was silently accepted")
    return "length and channel mismatches both refused"


def test_T8_checkpoint_roundtrip(dsn_main_dir):
    if not dsn_main_dir:
        raise _Skip("DSN tree not resolvable")
    import torch
    sys.path.insert(0, dsn_main_dir)
    import checkpoint as ckpt_mod
    from dsn_frozen import load_frozen_dsn, sha256_of_file

    W, E = 512, 16
    model, bcfg = make_tiny_backbone(dsn_main_dir, E=E, W=W)
    cfg_dict = {
        "backbone": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in bcfg.__dict__.items()},
        "data": {"window_s": W * 0.02},
        "cohort": {"w_size": 0.02, "gaussian_window": 0.04},
    }
    tmpd = tempfile.mkdtemp(prefix="smoke_ckpt_")
    try:
        path = os.path.join(tmpd, "best.pt")
        ckpt_mod.save_checkpoint(path, cfg_dict, model, epoch=7,
                                 capture_rng=False)
        dsn = load_frozen_dsn(path, device="cpu", dsn_main_dir=dsn_main_dir)

        if dsn.embedding_dim != E:
            raise AssertionError("E: got %d want %d" % (dsn.embedding_dim, E))
        if dsn.window_length != W:
            raise AssertionError("W: got %d want %d" % (dsn.window_length, W))
        if abs(dsn.fs_ifr - 50.0) > 1e-9:
            raise AssertionError("fs_ifr: got %r" % (dsn.fs_ifr,))
        if dsn.ckpt_sha256 != sha256_of_file(path):
            raise AssertionError("digest mismatch")
        if dsn.model.training:
            raise AssertionError("loaded model is in training mode")

        rng = np.random.default_rng(8)
        X = np.abs(rng.normal(size=(11, W))).astype(np.float32)
        model.eval()
        with torch.no_grad():
            want = model(torch.from_numpy(X)).numpy()
        got, _ = dsn.embed(X, batch_size=4, want_zraw=False)
        if not np.allclose(got, want, atol=1e-6):
            raise AssertionError("reloaded weights give different embeddings")
        return ("E=%d W=%d fs=%.4g Hz epoch=%d sha=%s..."
                % (dsn.embedding_dim, dsn.window_length, dsn.fs_ifr,
                   dsn.epoch, dsn.ckpt_sha256[:12]))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def rule_one(param_bounds):
    """Rule (1) of sbi_labels, re-implemented here as an INDEPENDENT check.

    For each fixed axis k with natural bounds (lo_k, hi_k):

        k in L  <=>  lo_k > 0  and  hi_k > 0  and  log10(hi_k / lo_k) >= 1

    The rule is the contract and is asserted. The COUNTS it yields are a
    property of whatever PARAM_BOUNDS currently says and are never asserted:
    sbi_labels derives L mechanically "so it tracks any future bounds edit",
    and a test that pins |L| to a literal contradicts that by construction.
    """
    b = np.asarray(param_bounds, dtype=np.float64)
    return sorted(k for k in range(b.shape[0])
                  if b[k, 0] > 0.0 and b[k, 1] > 0.0
                  and np.log10(b[k, 1] / b[k, 0]) >= 1.0)


def test_T9_registry_invariants(sim_dir, sweep_group="neuron_synapse"):
    """Structural invariants of the registry and the label spec. No counts.

    Every assertion below is a RELATION between measured quantities, so the
    test survives any edit to PARAM_BOUNDS, to the sweep group's membership,
    or to the width of the registry. What the test reports instead is the
    census: n, |L|, the active split and p, printed on the PASS line so that
    a change shows up in the log rather than as a failure.

    Note the two objects that a single literal used to conflate (they were
    both 27 in the registry of record and are not the same number):

      |L|     = how many registry axes rule (1) classifies as log coordinates
      spec.p  = len(active) + len(topology_axes), the label width
    """
    if not sim_dir:
        raise _Skip("needs --sim_dir (Phenomenological_finalv1)")
    from sbi_labels import load_registry, build_label_spec

    # Reaching the next line already establishes the load-time contract:
    # load_registry re-derives L from PARAM_BOUNDS and RAISES if it disagrees
    # with the simulator's own LOG_PARAMS ("the two copies of rule (1) have
    # drifted and the export must stop"). Nothing here needs to re-check that.
    reg = load_registry(sim_dir)

    n = len(reg.param_names)                        # registry width, MEASURED
    if n < 1:
        raise AssertionError("the registry is empty")
    if len(reg.param_units) != n:
        raise AssertionError("param_units has %d entries but param_names has %d"
                             % (len(reg.param_units), n))
    for nm, arr in (("param_bounds", reg.param_bounds),
                    ("param_bounds_theta", reg.param_bounds_theta)):
        if tuple(arr.shape) != (n, 2):
            raise AssertionError("%s has shape %r, expected (%d, 2)"
                                 % (nm, tuple(arr.shape), n))

    L = list(reg.log_param_indices)
    if sorted(set(L)) != L:
        raise AssertionError("log_param_indices is not sorted-unique: %r" % (L,))
    if L and (min(L) < 0 or max(L) >= n):
        raise AssertionError("log_param_indices %r out of range for width %d"
                             % (L, n))
    derived = rule_one(reg.param_bounds)
    if derived != L:
        raise AssertionError(
            "rule (1) re-derived independently in this test gives %r but the "
            "registry carries %r" % (derived, L))

    if sweep_group not in reg.sweep_groups:
        raise AssertionError("sweep group %r absent; registry has %r"
                             % (sweep_group, sorted(reg.sweep_groups)))
    act = list(reg.sweep_groups[sweep_group])
    if not act:
        raise AssertionError("sweep group %r is empty" % sweep_group)

    # Generalises the old DeltaT/VT/gL check: whichever axes are point
    # intervals, none of them may be active, or A3 divides by a zero width.
    frozen = [k for k in range(n)
              if reg.param_bounds[k, 0] == reg.param_bounds[k, 1]]
    overlap = sorted(set(frozen) & set(act))
    if overlap:
        raise AssertionError(
            "point-interval axes %r are in the active set"
            % ([reg.param_names[k] for k in overlap],))

    spec = build_label_spec(reg, act, sweep_group, conn_prob_bounds=(0.1, 0.6))
    n_topo = len(spec.topology_axes)
    if spec.p != len(act) + n_topo:
        raise AssertionError(
            "p = %d but len(active) + len(topology) = %d + %d; this identity "
            "is the only thing about p that is safe to assert"
            % (spec.p, len(act), n_topo))
    tail = spec.p - n_topo
    if spec.param_names[tail:] != list(spec.topology_axes):
        raise AssertionError("topology block is not last / not in order: %r"
                             % (spec.param_names[tail:],))
    if spec.coord[tail:] != ["linear"] * n_topo:
        raise AssertionError("topology block must be linear: %r"
                             % (spec.coord[tail:],))

    # The p0_conn trap as a RELATION, not as a decade count: the topology
    # block is linear by construction, so the trap is live exactly when rule
    # (1) WOULD have called p0_conn a log axis. Widening or narrowing its
    # bounds changes whether the trap is live; it must not fail the test.
    trap = ""
    if "p0_conn" in spec.topology_axes:
        i = spec.param_names.index("p0_conn")
        if spec.coord[i] != "linear":
            raise AssertionError("p0_conn coord is %r, must be linear"
                                 % spec.coord[i])
        p0_lo, p0_hi = float(reg.kernel_bounds[0, 0]), float(reg.kernel_bounds[0, 1])
        if p0_lo > 0.0 and p0_hi > 0.0:
            dec = float(np.log10(p0_hi / p0_lo))
            trap = ("; p0_conn spans %.3f decade(s), rule (1) %s misclassify it"
                    % (dec, "WOULD" if dec >= 1.0 else "would NOT"))

    log_set = set(L)
    n_log_act = sum(1 for k in act if k in log_set)
    lin_names = sorted(reg.param_names[k] for k in act if k not in log_set)
    shown = ", ".join(lin_names[:8]) or "-"
    if len(lin_names) > 8:
        shown += ", +%d more" % (len(lin_names) - 8)
    return ("n=%d |L|=%d; active=%d (%d ln + %d linear: %s); p=%d=%d+%d%s"
            % (n, len(L), len(act), n_log_act, len(lin_names),
               shown, spec.p, len(act), n_topo, trap))


def test_T9b_coordinate_map(sim_dir, sweep_group="neuron_synapse"):
    """The prior box is the coordinate rule applied to the natural box.

    For each fixed active axis k, with (lo_k, hi_k) its NATURAL bounds from
    PARAM_BOUNDS and (a_k, b_k) = spec.bounds_theta[i] the box the flow is
    trained against:

        coord_i = "ln"      =>  (a_k, b_k) = (ln lo_k, ln hi_k)
        coord_i = "linear"  =>  (a_k, b_k) = (lo_k, hi_k)

    This is the same rule assertion A6 applies per exported row, lifted to
    the bounds. A failure means PARAM_BOUNDS_THETA and PARAM_BOUNDS disagree,
    i.e. assertion A5 is admitting or rejecting rows against a box that is
    not the one the simulator sampled from. Count-free.
    """
    if not sim_dir:
        raise _Skip("needs --sim_dir (Phenomenological_finalv1)")
    from sbi_labels import load_registry, build_label_spec
    reg = load_registry(sim_dir)
    if sweep_group not in reg.sweep_groups:
        raise AssertionError("sweep group %r absent" % sweep_group)
    act = list(reg.sweep_groups[sweep_group])
    spec = build_label_spec(reg, act, sweep_group, conn_prob_bounds=(0.1, 0.6))

    bad = []
    for i, k in enumerate(spec.active_indices):
        lo, hi = float(reg.param_bounds[k, 0]), float(reg.param_bounds[k, 1])
        want = (np.log(lo), np.log(hi)) if spec.coord[i] == "ln" else (lo, hi)
        got = (float(spec.bounds_theta[i, 0]), float(spec.bounds_theta[i, 1]))
        if not np.allclose(got, want, rtol=1e-9, atol=1e-12):
            bad.append((reg.param_names[k], spec.coord[i], got, want))
    if bad:
        raise AssertionError(
            "%d active axis/axes whose prior box is not the coordinate rule "
            "applied to the natural box, e.g. %r" % (len(bad), bad[:3]))
    return ("%d active axes: prior box == coordinate rule applied to the "
            "natural box" % len(spec.active_indices))


def test_T10_assertion_A1(sim_dir):
    if not sim_dir:
        raise _Skip("needs --sim_dir")
    from sbi_labels import load_registry
    from export_embeddings import run_assertion_A1
    reg = load_registry(sim_dir)
    run_assertion_A1(reg)
    return "theta_to_natural(natural_to_theta(v)) == v at both bounds edges"


def test_T11_A3_rejects_frozen(sim_dir, sweep_group="neuron_synapse"):
    """A degenerate axis is refused at label construction.

    The frozen axis is FOUND rather than named: whichever axes currently have
    lo == hi, the first one outside the sweep group is the mistake A3 must
    catch. If a future registry freezes nothing, the degenerate box is built
    on the topology side instead, so the assertion is exercised either way and
    the test never depends on 'gL' still existing.
    """
    if not sim_dir:
        raise _Skip("needs --sim_dir")
    from sbi_labels import load_registry, build_label_spec
    reg = load_registry(sim_dir)
    act = list(reg.sweep_groups[sweep_group])
    frozen = [k for k in range(len(reg.param_names))
              if reg.param_bounds[k, 0] == reg.param_bounds[k, 1]
              and k not in set(act)]
    if frozen:
        k = frozen[0]
        trial, kw, what = sorted(act + [k]), {}, reg.param_names[k]
    else:
        # no frozen registry axis to borrow: make the topology box degenerate.
        trial, kw, what = act, {"conn_prob_bounds": (0.3, 0.3)}, "conn_prob"
    kw.setdefault("conn_prob_bounds", (0.1, 0.6))
    try:
        build_label_spec(reg, trial, sweep_group, **kw)
    except ValueError as exc:
        if "A3" not in str(exc):
            raise AssertionError("wrong error: %s" % exc)
        return "including %s raises A3 at label construction" % what
    raise AssertionError("a point-interval axis was accepted into the prior box")


def test_T12_end_to_end(dsn_main_dir, sim_dir):
    if not (dsn_main_dir and sim_dir):
        raise _Skip("needs a resolvable DSN tree and --sim_dir")
    import torch
    from dsn_frozen import FrozenDSN
    from sbi_labels import load_registry, build_label_spec, assemble_theta_A
    from sim_observable import build_pooled_ifr
    from export_embeddings import TraceRecord, export_embeddings

    reg = load_registry(sim_dir)
    act = reg.sweep_groups["neuron_synapse"]
    spec = build_label_spec(reg, act, "neuron_synapse", conn_prob_bounds=(0.1, 0.6))

    # short window so the test is fast; the geometry logic is identical
    T_win, dt = 10.0, 0.02
    W = int(round(T_win / dt))
    E = 16
    model, _ = make_tiny_backbone(dsn_main_dir, E=E, W=W)
    dsn = FrozenDSN(model=model, device=torch.device("cpu"), ckpt_path=__file__,
                    ckpt_sha256="a" * 64, embedding_dim=E, l2_normalize=True,
                    in_channels=1, window_s=T_win, w_size=dt,
                    gaussian_window=0.04, window_length=W)

    rng = np.random.default_rng(12)
    lo, hi = spec.bounds_theta[:, 0], spec.bounds_theta[:, 1]
    records = []
    for i in range(24):
        per_e = make_bursty_spikes(rng, T=T_win, n_electrodes=9)
        x = build_pooled_ifr(per_e, 9, T_win, dt, 0.04)
        th36 = np.zeros(len(reg.param_names), dtype=np.float64)
        draw = rng.uniform(lo, hi)
        for j, k in enumerate(spec.active_indices):
            th36[k] = draw[j]
        topo = {"conn_prob": float(draw[-4]), "p0_conn": float(draw[-3]),
                "d0_conn": float(draw[-2]), "beta_conn": float(draw[-1])}
        th = assemble_theta_A(spec, th36, topo)
        records.append(TraceRecord(
            trace=x, theta_A=th,
            ident={"campaign_id": "smoke", "topo_idx": i // 8, "iter_idx": i,
                   "seed_run": 1000 + i}))

    tmpd = tempfile.mkdtemp(prefix="smoke_export_")
    try:
        stem = os.path.join(tmpd, "sbi_smoke_0000")
        out = export_embeddings(
            dsn, records, stem, label_spec=spec,
            ident_columns=("campaign_id", "topo_idx", "iter_idx", "seed_run"),
            extra_sidecar={"campaign_id": "smoke",
                           "observable": {"n_electrodes": 9,
                                          "electrode_pitch_um": 60.0}},
            batch_size=8)
        if out.n_rows != 24:
            raise AssertionError("n_rows = %d, expected 24" % out.n_rows)

        import pyarrow.parquet as pq
        tbl = pq.read_table(out.parquet_path)
        cols = set(tbl.column_names)
        for need in ("z_000", "z_015", "zraw_000", "th_Sigma", "th_conn_prob",
                     "th_beta_conn", "campaign_id", "window_idx"):
            if need not in cols:
                raise AssertionError("missing column %r" % need)
        n_th = len([c for c in cols if c.startswith("th_")])
        if n_th != spec.p:
            raise AssertionError("%d th_* columns, spec.p = %d" % (n_th, spec.p))

        Z = np.column_stack([tbl.column("z_%03d" % j).to_numpy() for j in range(E)])
        if np.max(np.abs(np.linalg.norm(Z, axis=1) - 1.0)) >= 1e-5:
            raise AssertionError("A7 violated in the written file")

        with open(out.sidecar_path) as fh:
            side = json.load(fh)
        if side["observable"]["pooling"] != "mean_over_electrodes":
            raise AssertionError("sidecar records the wrong pooling convention")
        if len(side["bounds_theta"]) != spec.p or len(side["coord"]) != spec.p:
            raise AssertionError(
                "sidecar bounds_theta/coord are %d/%d long, spec.p = %d"
                % (len(side["bounds_theta"]), len(side["coord"]), spec.p))
        if side["embedding"]["embedding_dim"] != E:
            raise AssertionError("sidecar E")

        # A5 must fire on an out-of-box row
        bad = list(records)
        bad_th = bad[0].theta_A.copy()
        bad_th[0] = spec.bounds_theta[0, 1] + 1.0
        bad[0] = TraceRecord(trace=bad[0].trace, theta_A=bad_th,
                             ident=bad[0].ident)
        try:
            export_embeddings(dsn, bad, os.path.join(tmpd, "bad"),
                              label_spec=spec, batch_size=8)
        except AssertionError as exc:
            if "A5" not in str(exc):
                raise AssertionError("wrong assertion fired: %s" % exc)
        else:
            raise AssertionError("A5 did not fire on an out-of-box theta")

        # A4 must fire on a constant column
        const = [TraceRecord(trace=r.trace, theta_A=records[0].theta_A.copy(),
                             ident=r.ident) for r in records]
        try:
            export_embeddings(dsn, const, os.path.join(tmpd, "const"),
                              label_spec=spec, batch_size=8)
        except AssertionError as exc:
            if "A4" not in str(exc):
                raise AssertionError("wrong assertion fired: %s" % exc)
        else:
            raise AssertionError("A4 did not fire on constant columns")

        return ("%d rows x (%d z + %d zraw + %d th); A4 and A5 both fire; "
                "sidecar records mean_over_electrodes"
                % (out.n_rows, E, E, spec.p))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn_main_dir", default=None,
                    help="explicit DSN tree; default is <SBI_HPC_DIR>/dsn via "
                         "dsn_tree.py, if it resolves (else the DSN tests skip)")
    ap.add_argument("--sim_dir", default=os.environ.get("SIM_MAIN_DIR"),
                    help="<Astro-Neuron-Network>/hpc/Phenomenological_finalv1")
    args = ap.parse_args()

    if args.dsn_main_dir:
        dsn_dir = os.path.abspath(args.dsn_main_dir)
    else:
        import dsn_tree
        dsn_dir = dsn_tree.dsn_dir(require=False)
        if dsn_tree.status():
            print("NOTE: %s" % dsn_tree.status())
            dsn_dir = None
    sim_dir = os.path.abspath(args.sim_dir) if args.sim_dir else None

    print("=" * 70)
    print("SBI export chain -- smoke tests")
    print("  DSN tree : %s" % (dsn_dir or "(not resolvable -> some tests skip)"))
    print("  sim repo : %s" % (sim_dir or "(not given -> some tests skip)"))
    print("=" * 70)

    res = Results()
    _run(res, "T1", test_T1_windowing)
    _run(res, "T2", lambda: test_T2_ifr_parity(dsn_dir))
    _run(res, "T2b", lambda: test_T2b_normalised_parity(dsn_dir))
    _run(res, "T2c", lambda: test_T2c_edge_spike_at_T(dsn_dir))
    _run(res, "T3", lambda: test_T3_per_electrode_normalisation(dsn_dir))
    _run(res, "T4", lambda: test_T4_declared_duration_not_inferred(dsn_dir))
    _run(res, "T5", lambda: test_T5_zraw_is_prenorm(dsn_dir))
    _run(res, "T6", lambda: test_T6_batch_invariance(dsn_dir))
    _run(res, "T7", lambda: test_T7_window_length_guard(dsn_dir))
    _run(res, "T8", lambda: test_T8_checkpoint_roundtrip(dsn_dir))
    _run(res, "T9", lambda: test_T9_registry_invariants(sim_dir))
    _run(res, "T9b", lambda: test_T9b_coordinate_map(sim_dir))
    _run(res, "T10", lambda: test_T10_assertion_A1(sim_dir))
    _run(res, "T11", lambda: test_T11_A3_rejects_frozen(sim_dir))
    _run(res, "T12", lambda: test_T12_end_to_end(dsn_dir, sim_dir))
    return res.report()


if __name__ == "__main__":
    sys.exit(main())
