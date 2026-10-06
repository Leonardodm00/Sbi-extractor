"""Block 3 (encoder): oracles from SPEC.md section 2.3 and Block 3.

Uses a checkpoint saved from an untrained demo backbone with the DSN tree's own
checkpoint.save_checkpoint (SPEC Block 3, 'Not runnable here').
"""
import hashlib
import os
import sys

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from conftest import save_demo_checkpoint, make_backbone


def _X(n, W, seed=0, C=None):
    rng = np.random.default_rng(seed)
    shape = (n, W) if C is None else (n, C, W)
    return np.abs(rng.normal(size=shape)).astype(np.float32)


def test_geometry_from_checkpoint(frozen):
    # saved with W=100, dt=0.02, sigma=0.04, E=16
    assert frozen.window_length == int(round(frozen.window_s / frozen.w_size)) == 100
    assert frozen.fs_ifr == pytest.approx(50.0)
    assert frozen.sigma_bins == pytest.approx(2.0)
    assert frozen.embedding_dim == 16 and frozen.in_channels == 1
    assert frozen.l2_normalize is True
    assert frozen.epoch == 3


def test_W_rounding_not_floor(dsn_dir, tmp_path):
    """W = round(window_s / w_size): window_s = 0.6, w_size = 0.2 gives
    0.6/0.2 = 2.9999999999999996, floor 2, round 3."""
    from dsn_frozen import load_frozen_dsn
    assert int(0.6 / 0.2) == 2
    # save_demo_checkpoint writes window_s = W*dt; build config by hand instead
    import checkpoint as ckpt_mod
    model, bcfg = make_backbone(dsn_dir)
    cfg = {"backbone": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in bcfg.__dict__.items()},
           "data": {"window_s": 0.6}, "cohort": {"w_size": 0.2, "gaussian_window": 0.4}}
    p = str(tmp_path / "c.pt")
    ckpt_mod.save_checkpoint(p, cfg, model, capture_rng=False)
    assert load_frozen_dsn(p).window_length == 3


def test_digest(frozen, demo_ckpt):
    from dsn_frozen import sha256_of_file
    with open(demo_ckpt, "rb") as fh:
        want = hashlib.sha256(fh.read()).hexdigest()
    assert frozen.ckpt_sha256 == want == sha256_of_file(demo_ckpt)


def test_unit_sphere_and_zraw(frozen):
    X = _X(23, frozen.window_length)
    Z, Zraw = frozen.embed(X)
    assert Z.shape == Zraw.shape == (23, 16)
    assert Z.dtype == np.float32 and Zraw.dtype == np.float32
    n = np.linalg.norm(Z.astype(np.float64), axis=1)
    assert np.max(np.abs(n - 1.0)) < 1e-5            # A7
    zr = Zraw.astype(np.float64)
    assert_allclose(Z, zr / np.linalg.norm(zr, axis=1, keepdims=True), rtol=1e-6, atol=1e-7)


def test_zraw_is_head_proj_output(frozen):
    """Independent capture of model.head.proj output with our own hook."""
    X = _X(4, frozen.window_length, seed=3)
    got = []
    h = frozen.model.head.proj.register_forward_hook(lambda m, i, o: got.append(o.detach().numpy()))
    with torch.no_grad():
        frozen.model.eval()
        frozen.model(torch.from_numpy(X))
    h.remove()
    _, Zraw = frozen.embed(X)
    assert_array_equal(Zraw, got[0].astype(np.float32))


def test_batch_size_invariance(frozen):
    X = _X(37, frozen.window_length, seed=1)
    Z256, _ = frozen.embed(X, batch_size=256)
    for bs in (1, 5):
        Zb, _ = frozen.embed(X, batch_size=bs)
        # SPEC (inferred, Q4): max abs difference <= 1e-6
        assert np.max(np.abs(Zb - Z256)) <= 1e-6


def test_deterministic(frozen):
    X = _X(9, frozen.window_length, seed=2)
    a, _ = frozen.embed(X)
    b, _ = frozen.embed(X)
    assert_array_equal(a, b)


def test_train_mode_restored(frozen):
    X = _X(3, frozen.window_length)
    frozen.model.train()
    try:
        frozen.embed(X)
        assert frozen.model.training
    finally:
        frozen.model.eval()
    frozen.embed(X)
    assert not frozen.model.training


def test_refusals(frozen, tmp_path):
    from dsn_frozen import load_frozen_dsn
    with pytest.raises(ValueError):
        frozen.embed(_X(2, frozen.window_length + 1))
    with pytest.raises(ValueError):
        frozen.embed(_X(2, frozen.window_length, C=2))
    with pytest.raises(FileNotFoundError):
        load_frozen_dsn(str(tmp_path / "nope.pt"))


def test_head_without_proj(frozen):
    import copy
    f2 = copy.copy(frozen)
    m = copy.deepcopy(frozen.model)
    m.head.proj = torch.nn.Identity()
    f2.model = m
    with pytest.raises(AttributeError):
        f2.embed(_X(2, frozen.window_length))


def test_nonfinite_input_refused(frozen):
    X = _X(2, frozen.window_length)
    X[1, 5] = np.nan
    with pytest.raises(ValueError):
        frozen.embed(X)


def test_dsn_tree_resolution(tmp_path, monkeypatch, sbi_hpc_dir):
    import dsn_tree
    monkeypatch.setenv("SBI_HPC_DIR", sbi_hpc_dir)
    assert dsn_tree.sbi_hpc_dir(str(tmp_path)) == os.path.abspath(str(tmp_path))
    assert dsn_tree.sbi_hpc_dir() == os.path.abspath(sbi_hpc_dir)
    monkeypatch.delenv("SBI_HPC_DIR")
    assert dsn_tree.sbi_hpc_dir() == os.path.join(dsn_tree.ROOT, "artifacts", "sbi_hpc")
    # missing sentinel -> DSNTreeMissing naming the file
    d = tmp_path / "dsn"
    d.mkdir()
    for s in dsn_tree.SENTINELS[:-1]:
        (d / s).write_text("")
    with pytest.raises(dsn_tree.DSNTreeMissing, match="checkpoint.py"):
        dsn_tree.dsn_dir(str(tmp_path))
    with pytest.raises(dsn_tree.DSNTreeMissing):
        dsn_tree.dsn_dir(str(tmp_path / "absent"))


def test_negative_or_nan_sigma_in_checkpoint_refused(dsn_dir, tmp_path):
    """SPEC 4: no silent defaults; Block 1 requires sigma_sm >= 0. A checkpoint
    with a negative cohort.gaussian_window should be refused at load time or
    at the latest when the observable is built (Block 1 raises ValueError)."""
    from dsn_frozen import load_frozen_dsn
    p = save_demo_checkpoint(dsn_dir, str(tmp_path / "neg.pt"), sigma=-0.04)
    try:
        f = load_frozen_dsn(p)
    except (ValueError, KeyError):
        return
    import sim_observable
    with pytest.raises(ValueError):
        sim_observable.build_pooled_ifr([np.array([0.1])], 1, 10.0, f.w_size, f.gaussian_window)
