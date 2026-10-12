"""Block 6.1 -- the bank plan's configuration (specs/SPEC.md, Block 6.1).

Parameters only. The Giulia project's paths, digests and decided values live
in profiles/giulia.json; nothing here is read by the logic except through a
BankPlanConfig. Each decided value cites its decision in the project's log
(project knowledge, claude/SBI_decisions_and_ideas_log.md).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields
from typing import Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)            # the Sbi-extractor checkout
PROFILE_DIR = os.path.join(HERE, "profiles")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


class ConfigError(ValueError):
    """A profile that cannot configure a bank plan."""


@dataclass(frozen=True)
class BankPlanConfig:
    # -- inputs, each with the 16-hex digest prefix its tool printed --------
    record_path: str                  # G2's REEXTRACTION_RECORD.json
    record_sha256_prefix: str
    freeze_path: str                  # G3's label freeze
    freeze_sha256_prefix: str
    cohort_manifest_path: str         # the frozen copy of G1's manifest
    cohort_manifest_sha256_prefix: str
    sim_dir: str                      # simulator source with the hhgap registry
    # -- decided values ------------------------------------------------------
    on_axis: str                      # the one axis whose box may vary (D-074)
    on_declared_box: Tuple[float, float]   # natural units (D-074, D-078)
    topology_axes: Tuple[str, ...]    # the freeze's topology block (G3)
    expected_p: int                   # label width: active axes + topology block (G3: 23)
    t_win_s: float                    # window length [s] (D-041)
    n_e: int                          # electrodes per window (D-043)
    floor_hz: float                   # floor on mean_t x(t) [Hz] (D-075)
    # -- proposed values (SPEC Q6-Q8, open) ----------------------------------
    floor_rtol: float                 # relative tolerance of the floor (Q7)
    sample_seed: int                  # per-topology draw (Q6)
    bootstrap_resamples: int          # interval of the kept share (Q8)
    bootstrap_seed: int
    bytes_sample_rows: int            # kept windows compressed for bytes per row (D-080)
    rows_per_shard: int               # simulations per shard, before the floor (B5.3)
    # -- outputs -------------------------------------------------------------
    bank_root: str                    # the planned shards' folder (D-046)
    real_bank_root: str               # the real arm's shard folder (D-046; B5.5)
    plan_out: str                     # bank_plan.json, gitignored (D-079)


_PATH_KEYS = ("record_path", "freeze_path", "cohort_manifest_path", "sim_dir",
              "bank_root", "real_bank_root", "plan_out")


def _absolute(path, repo_root):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(repo_root, path))


def _validate(cfg):
    for key in ("record_sha256_prefix", "freeze_sha256_prefix",
                "cohort_manifest_sha256_prefix"):
        if not _HEX16.match(getattr(cfg, key)):
            raise ConfigError("%s must be 16 lowercase hexadecimal digits, got %r"
                              % (key, getattr(cfg, key)))
    lo, hi = cfg.on_declared_box
    if not (0.0 < lo < hi):
        raise ConfigError("on_declared_box must satisfy 0 < lo < hi, got %r"
                          % (cfg.on_declared_box,))
    if not cfg.topology_axes:
        raise ConfigError("topology_axes is empty")
    if not cfg.t_win_s > 0.0:
        raise ConfigError("t_win_s must be > 0 s, got %r" % (cfg.t_win_s,))
    if cfg.n_e < 1:
        raise ConfigError("n_e must be >= 1, got %r" % (cfg.n_e,))
    if not cfg.floor_hz > 0.0:
        raise ConfigError("floor_hz must be > 0 Hz, got %r" % (cfg.floor_hz,))
    if not 0.0 <= cfg.floor_rtol <= 1e-3:
        raise ConfigError("floor_rtol must lie in [0, 1e-3], got %r" % (cfg.floor_rtol,))
    if cfg.bootstrap_resamples < 100:
        raise ConfigError("bootstrap_resamples must be >= 100, got %r"
                          % (cfg.bootstrap_resamples,))
    if cfg.expected_p < len(cfg.topology_axes) + 1:
        raise ConfigError("expected_p must exceed the topology block, got %r"
                          % (cfg.expected_p,))
    if cfg.bytes_sample_rows < 1:
        raise ConfigError("bytes_sample_rows must be >= 1, got %r" % (cfg.bytes_sample_rows,))
    if not 100 <= cfg.rows_per_shard <= 100000:
        raise ConfigError("rows_per_shard must lie in [100, 100000], got %r"
                          % (cfg.rows_per_shard,))
    for key in ("sample_seed", "bootstrap_seed"):
        if getattr(cfg, key) < 0:
            raise ConfigError("%s must be a non-negative integer" % key)


def load_profile(name_or_path, repo_root: Optional[str] = None) -> BankPlanConfig:
    """Read a profile (a name under profiles/, or a JSON path) into a config.

    Keys beginning with '_' are comments. Every other key must be a field of
    BankPlanConfig, and every field must be given: no silent defaults.
    """
    repo_root = repo_root or REPO_ROOT
    path = name_or_path
    if not path.endswith(".json"):
        path = os.path.join(PROFILE_DIR, name_or_path + ".json")
    if not os.path.isfile(path):
        raise ConfigError("no profile at %s" % path)
    with open(path, "r", encoding="ascii") as fh:
        raw = json.load(fh)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    names = [f.name for f in fields(BankPlanConfig)]
    unknown = sorted(set(raw) - set(names))
    missing = sorted(set(names) - set(raw))
    if unknown or missing:
        raise ConfigError("profile %s: unknown keys %r, missing keys %r"
                          % (path, unknown, missing))
    for key in ("n_e", "sample_seed", "bootstrap_resamples", "bootstrap_seed", "expected_p",
                "bytes_sample_rows", "rows_per_shard"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int):
            raise ConfigError("%s must be an integer, got %r" % (key, raw[key]))
    try:
        box = tuple(float(v) for v in raw["on_declared_box"])
        axes = tuple(str(a) for a in raw["topology_axes"])
        floats = {k: float(raw[k]) for k in ("t_win_s", "floor_hz", "floor_rtol")}
    except (TypeError, ValueError) as exc:
        raise ConfigError("profile %s: %s" % (path, exc))
    if len(box) != 2:
        raise ConfigError("on_declared_box must have two entries, got %r" % (box,))
    values = dict(raw, on_declared_box=box, topology_axes=axes, **floats)
    for key in _PATH_KEYS:
        values[key] = _absolute(str(values[key]), repo_root)
    cfg = BankPlanConfig(**values)
    _validate(cfg)
    return cfg
