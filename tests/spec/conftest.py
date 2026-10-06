"""Shared fixtures for the spec-derived tests (written by the independent tester).

Environment (SPEC.md section 6): SBI_HPC_DIR -> <Simulation-Based-Inference>/hpc
and SIM_MAIN_DIR -> <Astro-Neuron-Network>/hpc/Phenomenological_finalv1 must be
set. Tests that need them are skipped with a reason when they are not.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _need_env(name):
    v = os.environ.get(name)
    if not v or not os.path.isdir(v):
        pytest.skip("%s not set to an existing directory (SPEC.md section 6)" % name)
    return v


@pytest.fixture(scope="session")
def sbi_hpc_dir():
    return _need_env("SBI_HPC_DIR")


@pytest.fixture(scope="session")
def dsn_dir(sbi_hpc_dir):
    d = os.path.join(sbi_hpc_dir, "dsn")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


@pytest.fixture(scope="session")
def sim_dir():
    return _need_env("SIM_MAIN_DIR")


@pytest.fixture(scope="session")
def registry(sim_dir):
    import sbi_labels
    return sbi_labels.load_registry(sim_dir)


def make_backbone(dsn_dir, E=16, in_channels=1, l2=True):
    """Small untrained backbone from the DSN tree (SPEC Block 3 'Not runnable here')."""
    import torch
    if dsn_dir not in sys.path:
        sys.path.insert(0, dsn_dir)
    from backbone import BackboneConfig, build_backbone
    torch.manual_seed(0)
    cfg = BackboneConfig(depth_exponent=2, width_multiplier=2.0, stem_width=8,
                         in_channels=in_channels, embedding_size=E,
                         l2_normalize=l2, head_fusion=True,
                         head_pool_ops=("mean",))
    model = build_backbone(cfg)
    model.eval()
    return model, cfg


def save_demo_checkpoint(dsn_dir, path, W=100, dt=0.02, sigma=0.04, E=16,
                         in_channels=1, l2=True, epoch=3):
    """Checkpoint saved with the DSN tree's own checkpoint.save_checkpoint."""
    model, bcfg = make_backbone(dsn_dir, E=E, in_channels=in_channels, l2=l2)
    import checkpoint as ckpt_mod
    cfg_dict = {
        "backbone": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in bcfg.__dict__.items()},
        "data": {"window_s": W * dt},
        "cohort": {"w_size": dt, "gaussian_window": sigma},
    }
    ckpt_mod.save_checkpoint(path, cfg_dict, model, epoch=epoch,
                             capture_rng=False)
    return path


@pytest.fixture(scope="session")
def demo_ckpt(dsn_dir, tmp_path_factory):
    p = tmp_path_factory.mktemp("ckpt") / "best.pt"
    return str(save_demo_checkpoint(dsn_dir, str(p)))


@pytest.fixture(scope="session")
def frozen(dsn_dir, demo_ckpt):
    from dsn_frozen import load_frozen_dsn
    return load_frozen_dsn(demo_ckpt, device="cpu")
