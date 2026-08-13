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

    # full (recommended: exercises the real code paths)
    python3 smoke_test_sbi_export.py \
        --dsn_main_dir /path/to/Deep-Summary-Network/Main \
        --sim_dir      /path/to/Astro-Neuron-Network/hpc/Phenomenological_finalv1

    # equivalently, via the environment
    export DSN_MAIN_DIR=/path/to/Deep-Summary-Network/Main
    export SIM_MAIN_DIR=/path/to/Astro-Neuron-Network/hpc/Phenomenological_finalv1
    python3 smoke_test_sbi_export.py

Exit code 0 = every non-skipped test passed. Non-zero = at least one failed.

WHAT EACH TEST ESTABLISHES
--------------------------
T1  windowing matches MEAWindowDataset's index rule exactly, including the
    silent-drop case L < W.
T2  build_pooled_ifr reproduces the DSN repo's own compute_ifr_trace bit for
    bit, up to the per-electrode division. This is the test that settles the
    pooling-convention question; everything else about scale parity is opinion
    without it.                                          [needs DSN repo]
T3  per-electrode normalisation is exactly a factor 1/n_e, and the un-normalised
    variant is n_e times larger -- the amplitude error the handoff would cause.
T4  the bin grid is driven by the DECLARED T, not by the data, so two traces
    with different last-spike times still give the same K. This is the
    process_campaign 'inferred simtime' trap.
T5  zraw is genuinely the pre-normalisation activation: normalize(zraw) == z.
T6  Z is invariant to batch_size (GroupNorm, no batch statistics).
T7  embed() refuses a window-length mismatch instead of silently resampling.
T8  the checkpoint round trip: save -> load_frozen_dsn -> identical embeddings,
    and the SHA-256 matches an independent digest.       [needs DSN repo]
T9  rule (1) re-derived from PARAM_BOUNDS gives |L| = 27, 23 active axes,
    19 log + 4 linear, and p0_conn spans exactly 1 decade -- i.e. it WOULD be
    misclassified if rule (1) were applied to the topology block. [needs sim repo]
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
        raise _Skip("needs --dsn_main_dir (DSN repo) for compute_ifr_trace")
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


def test_T3_per_electrode_normalisation():
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


def test_T4_declared_duration_not_inferred():
    """The simtime trap: two runs whose LAST SPIKE differs must still give the
    same K, because T is declared. This is what process_campaign.py's
    simtime = ceil(spk_t.max()) breaks."""
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
        raise _Skip("needs --dsn_main_dir to build the real backbone")
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
        raise _Skip("needs --dsn_main_dir")
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
        raise _Skip("needs --dsn_main_dir")
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
        raise _Skip("needs --dsn_main_dir")
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


def test_T9_registry_counts(sim_dir):
    if not sim_dir:
        raise _Skip("needs --sim_dir (Phenomenological_finalv1)")
    from sbi_labels import load_registry, build_label_spec
    reg = load_registry(sim_dir)

    if len(reg.param_names) != 36:
        raise AssertionError("registry width %d, expected 36" % len(reg.param_names))
    if len(reg.log_param_indices) != 27:
        raise AssertionError("|L| = %d, expected 27" % len(reg.log_param_indices))

    act = reg.sweep_groups["neuron_synapse"]
    if len(act) != 23:
        raise AssertionError("active axes %d, expected 23" % len(act))
    n_log = sum(1 for k in act if k in set(reg.log_param_indices))
    if n_log != 19:
        raise AssertionError("active log axes %d, expected 19" % n_log)
    linear = [reg.param_names[k] for k in act if k not in set(reg.log_param_indices)]
    if sorted(linear) != sorted(["VA", "VR", "I_inj", "Cm"]):
        raise AssertionError("active linear axes %r" % (linear,))

    for nm in ("DeltaT", "VT", "gL"):
        k = reg.param_names.index(nm)
        if reg.param_bounds[k, 0] != reg.param_bounds[k, 1]:
            raise AssertionError("%s is not a point interval" % nm)
        if k in act:
            raise AssertionError("%s must not be active" % nm)

    # the p0_conn trap, exhibited numerically
    p0_lo, p0_hi = reg.kernel_bounds[0]
    decades = float(np.log10(p0_hi / p0_lo))
    if not np.isclose(decades, 1.0, atol=1e-12):
        raise AssertionError("p0_conn spans %.6f decades, expected exactly 1"
                             % decades)

    spec = build_label_spec(reg, act, "neuron_synapse", conn_prob_bounds=(0.1, 0.6))
    if spec.p != 27:
        raise AssertionError("p = %d, expected 27" % spec.p)
    if spec.coord[-4:] != ["linear"] * 4:
        raise AssertionError("topology block must be linear: %r" % spec.coord[-4:])
    if spec.param_names[-4:] != ["conn_prob", "p0_conn", "d0_conn", "beta_conn"]:
        raise AssertionError("topology order: %r" % spec.param_names[-4:])
    return ("|L|=27, 23 active (19 ln + 4 linear), p=27; p0_conn spans exactly "
            "%.1f decade -> rule (1) WOULD misclassify it" % decades)


def test_T10_assertion_A1(sim_dir):
    if not sim_dir:
        raise _Skip("needs --sim_dir")
    from sbi_labels import load_registry
    from export_embeddings import run_assertion_A1
    reg = load_registry(sim_dir)
    run_assertion_A1(reg)
    return "theta_to_natural(natural_to_theta(v)) == v at both bounds edges"


def test_T11_A3_rejects_frozen(sim_dir):
    if not sim_dir:
        raise _Skip("needs --sim_dir")
    from sbi_labels import load_registry, build_label_spec
    reg = load_registry(sim_dir)
    act = list(reg.sweep_groups["neuron_synapse"])
    act.append(reg.param_names.index("gL"))          # the mistake A3 must catch
    try:
        build_label_spec(reg, sorted(act), "neuron_synapse",
                         conn_prob_bounds=(0.1, 0.6))
    except ValueError as exc:
        if "A3" not in str(exc):
            raise AssertionError("wrong error: %s" % exc)
        return "including gL raises A3 at label construction"
    raise AssertionError("a point-interval axis was accepted into the prior box")


def test_T12_end_to_end(dsn_main_dir, sim_dir):
    if not (dsn_main_dir and sim_dir):
        raise _Skip("needs both --dsn_main_dir and --sim_dir")
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
        th36 = np.zeros(36, dtype=np.float64)
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
        if len([c for c in cols if c.startswith("th_")]) != 27:
            raise AssertionError("expected 27 th_* columns")

        Z = np.column_stack([tbl.column("z_%03d" % j).to_numpy() for j in range(E)])
        if np.max(np.abs(np.linalg.norm(Z, axis=1) - 1.0)) >= 1e-5:
            raise AssertionError("A7 violated in the written file")

        with open(out.sidecar_path) as fh:
            side = json.load(fh)
        if side["observable"]["pooling"] != "mean_over_electrodes":
            raise AssertionError("sidecar records the wrong pooling convention")
        if len(side["bounds_theta"]) != 27 or len(side["coord"]) != 27:
            raise AssertionError("sidecar bounds/coord length")
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

        return ("24 rows x (16 z + 16 zraw + 27 th); A4 and A5 both fire; "
                "sidecar records mean_over_electrodes")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn_main_dir", default=os.environ.get("DSN_MAIN_DIR"),
                    help="<Deep-Summary-Network>/Main")
    ap.add_argument("--sim_dir", default=os.environ.get("SIM_MAIN_DIR"),
                    help="<Astro-Neuron-Network>/hpc/Phenomenological_finalv1")
    args = ap.parse_args()

    dsn_dir = os.path.abspath(args.dsn_main_dir) if args.dsn_main_dir else None
    sim_dir = os.path.abspath(args.sim_dir) if args.sim_dir else None

    print("=" * 70)
    print("SBI export chain -- smoke tests")
    print("  DSN repo : %s" % (dsn_dir or "(not given -> some tests skip)"))
    print("  sim repo : %s" % (sim_dir or "(not given -> some tests skip)"))
    print("=" * 70)

    res = Results()
    _run(res, "T1", test_T1_windowing)
    _run(res, "T2", lambda: test_T2_ifr_parity(dsn_dir))
    _run(res, "T3", test_T3_per_electrode_normalisation)
    _run(res, "T4", test_T4_declared_duration_not_inferred)
    _run(res, "T5", lambda: test_T5_zraw_is_prenorm(dsn_dir))
    _run(res, "T6", lambda: test_T6_batch_invariance(dsn_dir))
    _run(res, "T7", lambda: test_T7_window_length_guard(dsn_dir))
    _run(res, "T8", lambda: test_T8_checkpoint_roundtrip(dsn_dir))
    _run(res, "T9", lambda: test_T9_registry_counts(sim_dir))
    _run(res, "T10", lambda: test_T10_assertion_A1(sim_dir))
    _run(res, "T11", lambda: test_T11_A3_rejects_frozen(sim_dir))
    _run(res, "T12", lambda: test_T12_end_to_end(dsn_dir, sim_dir))
    return res.report()


if __name__ == "__main__":
    sys.exit(main())
