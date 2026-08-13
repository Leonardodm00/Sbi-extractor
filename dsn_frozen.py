"""
dsn_frozen.py
=============

STAGE 1 of the SBI export chain: load a frozen Deep Summary Network (DSN)
checkpoint and run the forward pass that turns windowed IFR traces into
embeddings.

Separation of concerns (scientific coding directive 2): this module does
LOADING + FORWARD PASS ONLY. It never builds an IFR, never reads a campaign
directory, never assembles a parameter label, and never writes a Parquet file.
Those live in sim_observable.py, sbi_labels.py and export_embeddings.py
respectively. Swapping the trace source or the label definition must not touch
this file.

What this module guarantees
---------------------------
    1. The architecture is rebuilt from the config EMBEDDED IN THE CHECKPOINT
       (checkpoint.rebuild_model_from_checkpoint), never from a remembered
       hyper-parameter. This is the DSN repository's own contract.
    2. model.eval() is called before every forward pass and the caller's
       train/eval mode is restored afterwards.
    3. The SHA-256 of the checkpoint FILE is computed and carried, so simulated
       and real exports can be PROVEN to share one psi (handoff assertion A8).
    4. Both z (L2-normalised, on the unit sphere) and zraw (pre-normalisation
       activations) are returned. zraw is captured with a forward hook on the
       head's final Linear layer.

Notation (consistent with the export handoff)
---------------------------------------------
    N      : number of windows presented
    W      : window length in SAMPLES,  W = round(T_win * fs_ifr)
    C      : input channel count (backbone.in_channels; 1 for a pooled IFR)
    E      : embedding dimension (backbone.embedding_size)
    x_i    : one clean window,  x_i in R_{>=0}^{W}   (or R^{C x W})
    h_psi  : the frozen encoder
    z_i    : h_psi(x_i) in S^{E-1} subset R^E, with ||z_i||_2 = 1
    zraw_i : the pre-normalisation activation, z_i = zraw_i / ||zraw_i||_2

Why zraw matters (handoff Section 4.3 item 3, and Section 7)
-------------------------------------------------------------
The head does

    zraw = self.proj(pooled)              # bare nn.Linear
    z    = F.normalize(zraw, p=2, dim=1)  # only when cfg.l2_normalize

so the projection to S^{E-1} DISCARDS ||zraw||_2, which is the only place
amplitude information survives. Amplitude is exactly what the real-vs-simulated
scale-parity question turns on, so zraw is recovered here rather than declared
unavailable. The hook is attached to the head's Linear module, so it captures
the tensor BEFORE F.normalize by construction.

HPC note (hpc-python-compat): pure ASCII, LF-only. torch and numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DSNRepoNotFound",
    "FrozenDSN",
    "sha256_of_file",
    "load_frozen_dsn",
]


# --------------------------------------------------------------------------- #
# repository import
# --------------------------------------------------------------------------- #
class DSNRepoNotFound(ImportError):
    """Raised when the Deep-Summary-Network 'Main' directory is not importable."""


def _import_dsn_modules(dsn_main_dir: Optional[str]):
    """Put the DSN repo's Main/ directory on sys.path and import what we need.

    Directive 1 (leverage the ecosystem / reuse tested code): the model is
    rebuilt with the repository's OWN checkpoint.rebuild_model_from_checkpoint
    rather than a re-implementation here, so this export cannot drift away from
    how train.py and evaluate.py construct the same network.

    Parameters
    ----------
    dsn_main_dir : str or None
        Path to <Deep-Summary-Network>/Main. If None, the environment variable
        DSN_MAIN_DIR is used; if that is unset, the modules are assumed to be
        importable already (e.g. this file was dropped into Main/).

    Returns
    -------
    (checkpoint_module, backbone_module)
    """
    cand = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    if cand:
        cand = os.path.abspath(cand)
        if not os.path.isdir(cand):
            raise DSNRepoNotFound(
                "dsn_main_dir=%r is not a directory. Point it at the 'Main' "
                "folder of the Deep-Summary-Network checkout." % (cand,))
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import checkpoint as _ckpt_mod      # noqa: E402
        import backbone as _bb_mod          # noqa: E402
    except ImportError as exc:
        raise DSNRepoNotFound(
            "could not import the DSN modules 'checkpoint' and 'backbone'. "
            "Pass dsn_main_dir=<repo>/Main, or set the DSN_MAIN_DIR "
            "environment variable. Original error: %r" % (exc,))
    return _ckpt_mod, _bb_mod


# --------------------------------------------------------------------------- #
# checkpoint digest (handoff assertion A8)
# --------------------------------------------------------------------------- #
def sha256_of_file(path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks so a large checkpoint does
    not have to be held in memory twice.

    This digests the checkpoint FILE, not the weight tensors. That is the right
    granularity for assertion A8 ("the simulated and real sidecars carry the
    same digest"): it is what the user actually copies between jobs, and it is
    reproducible with `sha256sum` on the cluster without loading torch.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the frozen encoder
# --------------------------------------------------------------------------- #
@dataclass
class FrozenDSN:
    """A loaded, evaluation-mode DSN encoder plus every constant the rest of the
    export chain must NOT hard-code.

    Every field below is read off the checkpoint's embedded config. Nothing in
    the export pipeline may assume a value for W, E, dt or sigma_sm: the whole
    point of carrying them here is that a checkpoint trained with different
    windowing produces a different, self-consistent export instead of a silently
    misaligned one.
    """

    model: "torch.nn.Module"
    device: "torch.device"
    ckpt_path: str
    ckpt_sha256: str

    # --- embedding geometry ---
    embedding_dim: int          # E
    l2_normalize: bool
    in_channels: int            # C

    # --- observable geometry (what the encoder was trained to receive) ---
    window_s: float             # T_win  [s]
    w_size: float               # Delta_t [s]  (IFR bin width)
    gaussian_window: float      # sigma_sm [s]
    window_length: int          # W = round(window_s / w_size)  [samples]

    # --- provenance ---
    epoch: int = -1
    config_dict: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)

    # ----------------------------------------------------------------- #
    @property
    def fs_ifr(self) -> float:
        """IFR sampling rate implied by the bin width, f_s^IFR = 1 / Delta_t [Hz]."""
        return 1.0 / float(self.w_size)

    @property
    def sigma_bins(self) -> float:
        """Gaussian smoothing width in BINS, sigma_sm / Delta_t (dimensionless)."""
        return float(self.gaussian_window) / float(self.w_size)

    # ----------------------------------------------------------------- #
    def _head_linear(self):
        """The head's final bare nn.Linear, whose OUTPUT is zraw.

        Located by attribute path rather than by scanning, so a future head with
        a different structure fails loudly here instead of silently hooking the
        wrong layer and exporting a zraw block that is not the pre-normalisation
        activation.
        """
        head = getattr(self.model, "head", None)
        proj = getattr(head, "proj", None) if head is not None else None
        if not isinstance(proj, torch.nn.Linear):
            raise AttributeError(
                "expected model.head.proj to be an nn.Linear (the DSN "
                "MultiScaleHead contract); got %r. Refusing to guess which "
                "layer produces the pre-normalisation activation."
                % (type(proj).__name__,))
        return proj

    # ----------------------------------------------------------------- #
    def embed(self, X, batch_size: int = 256, want_zraw: bool = True
              ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Embed a stack of clean windows.

        Parameters
        ----------
        X : float array, (N, W) when in_channels == 1, else (N, C, W)
            Clean (un-augmented) windows. This is deliberately a plain array
            rather than a Dataset: the trace source is decoupled from the
            encoder, so the simulated path and the real path share this code.
        batch_size : int
            Forward-pass chunk size. A pure throughput knob: the backbone uses
            GroupNorm (no cross-sample batch statistics), so Z is identical for
            every batch_size. smoke_test_sbi_export.py pins this.
        want_zraw : bool
            Capture the pre-normalisation activations as well.

        Returns
        -------
        Z    : (N, E) float32, rows on S^{E-1} when l2_normalize is True
        Zraw : (N, E) float32 or None
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 2:
            n_ch_seen, W_seen = 1, X.shape[1]
        elif X.ndim == 3:
            n_ch_seen, W_seen = X.shape[1], X.shape[2]
        else:
            raise ValueError(
                "X must be (N, W) or (N, C, W); got shape %r" % (X.shape,))

        if n_ch_seen != self.in_channels:
            raise ValueError(
                "X has %d channel(s) but the checkpoint's backbone expects "
                "in_channels=%d. The channel axis is the single source of "
                "truth for the (C, T) -> (C, W) -> (M, C, W) shape contract; "
                "reshaping here would silently mislabel per-region traces as "
                "channels." % (n_ch_seen, self.in_channels))
        if W_seen != self.window_length:
            raise ValueError(
                "X has window length %d samples but the checkpoint was trained "
                "with W = %d (= round(window_s=%.6g s * fs_ifr=%.6g Hz)). A "
                "length mismatch alone can separate two embedding clouds "
                "before any biology enters; refusing to embed. Re-window the "
                "traces rather than resampling here."
                % (W_seen, self.window_length, self.window_s, self.fs_ifr))
        if X.shape[0] == 0:
            raise ValueError("X is empty; nothing to embed")
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or Inf. Fix the trace construction upstream; "
                "this module will not sanitise its input.")

        n_rows = X.shape[0]
        bs = int(batch_size)
        if bs < 1:
            raise ValueError("batch_size must be >= 1")

        captured = []
        handle = None
        if want_zraw:
            def _hook(_module, _inp, out):
                captured.append(out.detach().to("cpu", dtype=torch.float32).numpy())
            handle = self._head_linear().register_forward_hook(_hook)

        was_training = self.model.training
        self.model.eval()
        z_chunks = []
        try:
            with torch.no_grad():
                for start in range(0, n_rows, bs):
                    xb = torch.from_numpy(X[start:start + bs]).to(self.device)
                    zb = self.model(xb)
                    z_chunks.append(
                        zb.detach().to("cpu", dtype=torch.float32).numpy())
        finally:
            if handle is not None:
                handle.remove()
            if was_training:
                self.model.train()

        Z = np.ascontiguousarray(np.concatenate(z_chunks, axis=0), dtype=np.float32)
        if Z.shape != (n_rows, self.embedding_dim):
            raise RuntimeError(
                "forward produced %r, expected (%d, %d)"
                % (Z.shape, n_rows, self.embedding_dim))

        Zraw = None
        if want_zraw:
            if not captured:
                raise RuntimeError(
                    "the zraw forward hook never fired; model.head.proj was not "
                    "executed during the forward pass")
            Zraw = np.ascontiguousarray(
                np.concatenate(captured, axis=0), dtype=np.float32)
            if Zraw.shape != Z.shape:
                raise RuntimeError(
                    "zraw shape %r does not match z shape %r"
                    % (Zraw.shape, Z.shape))
        return Z, Zraw

    # ----------------------------------------------------------------- #
    def embedding_block(self, zraw_available: bool) -> dict:
        """The sidecar 'embedding' block (handoff Section 6.1)."""
        return {
            "embedding_dim": int(self.embedding_dim),
            "dsn_checkpoint_sha256": self.ckpt_sha256,
            "dsn_checkpoint_path": os.path.abspath(self.ckpt_path),
            "dsn_checkpoint_epoch": int(self.epoch),
            "l2_normalised": bool(self.l2_normalize),
            "zraw_available": bool(zraw_available),
            "in_channels": int(self.in_channels),
        }

    def observable_block(self) -> dict:
        """The parts of the sidecar 'observable' block that the CHECKPOINT
        determines. The electrode-geometry fields are owned by the simulated
        observable stage and are merged in by export_embeddings.py.
        """
        return {
            "ifr_dt_s": float(self.w_size),
            "ifr_smooth_sigma_s": float(self.gaussian_window),
            "ifr_smooth_sigma_bins": float(self.sigma_bins),
            "fs_ifr_hz": float(self.fs_ifr),
            "window_s": float(self.window_s),
            "window_length_samples": int(self.window_length),
            "ifr_units": "spikes per bin per electrode "
                         "(multiply by fs_ifr for Hz)",
        }


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def _dig(d: dict, *path, default=None):
    """Nested dict lookup that returns `default` on any missing key."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_frozen_dsn(ckpt_path,
                    device="cpu",
                    dsn_main_dir: Optional[str] = None,
                    expect_config_json: Optional[str] = None,
                    strict: bool = False) -> FrozenDSN:
    """Load a DSN checkpoint into an evaluation-mode FrozenDSN.

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt checkpoint written by checkpoint.save_checkpoint
        (typically <out>/checkpoints/best.pt).
    device : str or torch.device
    dsn_main_dir : str or None
        Path to <Deep-Summary-Network>/Main (see _import_dsn_modules).
    expect_config_json : str or None
        Optional path to the training config JSON (e.g.
        hpc/Config/config_mea_joint_full.json). When given, the loaded
        checkpoint's window_s / w_size / embedding_size / in_channels are
        compared against it and any disagreement is reported. This is the cheap
        guard against embedding an export with the WRONG checkpoint, which no
        downstream diagnostic can detect.
    strict : bool
        If True, a disagreement with expect_config_json raises instead of
        recording a warning.

    Returns
    -------
    FrozenDSN
    """
    ckpt_mod, _bb_mod = _import_dsn_modules(dsn_main_dir)

    ckpt_path = os.path.abspath(str(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("no such checkpoint: %s" % (ckpt_path,))

    digest = sha256_of_file(ckpt_path)
    device = torch.device(device)

    model, ckpt = ckpt_mod.rebuild_model_from_checkpoint(
        ckpt_path, map_location=str(device))
    model.to(device)
    model.eval()

    cfg = ckpt.get("config", {}) or {}
    warnings_out = []

    # --- embedding geometry: authoritative, from the rebuilt module ---------
    bb_cfg = getattr(model, "cfg", None)
    if bb_cfg is None:
        raise RuntimeError(
            "the rebuilt model exposes no .cfg; cannot read E / in_channels")
    embedding_dim = int(bb_cfg.embedding_size)
    l2_normalize = bool(bb_cfg.l2_normalize)
    in_channels = int(bb_cfg.in_channels)

    if not l2_normalize:
        warnings_out.append(
            "backbone.l2_normalize is False: embeddings are NOT on S^{E-1}, so "
            "handoff assertion A7 (||z||_2 == 1) does not apply and any "
            "geodesic / cosine analysis downstream must be revisited.")

    # --- observable geometry: from the embedded data + cohort blocks -------
    window_s = _dig(cfg, "data", "window_s")
    w_size = _dig(cfg, "cohort", "w_size")
    gaussian_window = _dig(cfg, "cohort", "gaussian_window")

    if window_s is None:
        raise KeyError(
            "checkpoint config has no data.window_s; cannot determine T_win. "
            "Refusing to fall back on the config.py default (200.0 s), which "
            "would exceed a 180 s simulated trace and silently yield ZERO "
            "windows (MEAWindowDataset skips traces shorter than W).")
    if w_size is None:
        w_size = 0.02
        warnings_out.append(
            "checkpoint config has no cohort.w_size; assuming Delta_t = 0.02 s "
            "(DEFAULT_W_SIZE in channel_subset_extraction.py). VERIFY this "
            "against the extraction flags before trusting the export.")
    if gaussian_window is None:
        gaussian_window = 0.04
        warnings_out.append(
            "checkpoint config has no cohort.gaussian_window; assuming "
            "sigma_sm = 0.04 s (DEFAULT_GAUSSIAN_WINDOW). VERIFY as above.")

    window_s = float(window_s)
    w_size = float(w_size)
    gaussian_window = float(gaussian_window)
    if w_size <= 0.0:
        raise ValueError("cohort.w_size must be > 0; got %r" % (w_size,))

    # W = round(window_s * fs_ifr), matching run_optimization.py:
    #     W = int(round(float(cfg.data.window_s) * fs))
    window_length = int(round(window_s / w_size))
    if window_length < 1:
        raise ValueError(
            "window_s / w_size rounds to < 1 sample (window_s=%r, w_size=%r)"
            % (window_s, w_size))

    sigma_bins = gaussian_window / w_size
    if gaussian_window > 0.0 and sigma_bins < 1.0:
        warnings_out.append(
            "sigma_sm = %.4g s is only %.2f bin(s) at Delta_t = %.4g s, so the "
            "Gaussian smoothing is close to a no-op."
            % (gaussian_window, sigma_bins, w_size))

    # --- optional cross-check against the training config JSON -------------
    if expect_config_json is not None:
        with open(expect_config_json, "r") as fh:
            ref = json.load(fh)
        checks = [
            ("data.window_s", window_s, _dig(ref, "data", "window_s")),
            ("cohort.w_size", w_size, _dig(ref, "cohort", "w_size")),
            ("backbone.embedding_size", embedding_dim,
             _dig(ref, "backbone", "embedding_size")),
            ("data.n_channels", in_channels, _dig(ref, "data", "n_channels")),
        ]
        for name, got, want in checks:
            if want is None:
                continue
            same = (int(got) == int(want) if isinstance(want, int)
                    else abs(float(got) - float(want)) <= 1e-9)
            if not same:
                msg = ("checkpoint disagrees with %s on %s: checkpoint=%r, "
                       "config=%r. This usually means the WRONG checkpoint is "
                       "being used for this export."
                       % (os.path.basename(expect_config_json), name, got, want))
                if strict:
                    raise ValueError(msg)
                warnings_out.append(msg)

    return FrozenDSN(
        model=model,
        device=device,
        ckpt_path=ckpt_path,
        ckpt_sha256=digest,
        embedding_dim=embedding_dim,
        l2_normalize=l2_normalize,
        in_channels=in_channels,
        window_s=window_s,
        w_size=w_size,
        gaussian_window=gaussian_window,
        window_length=window_length,
        epoch=int(ckpt.get("epoch", -1)),
        config_dict=cfg,
        warnings=warnings_out,
    )
