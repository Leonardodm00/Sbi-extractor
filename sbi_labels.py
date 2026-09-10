"""
sbi_labels.py
=============

STAGE 3 of the SBI export chain: assemble the 27-dimensional SBI training label
theta_A and its prior box B, in the mixed (ln / linear) inference coordinate.

Separation of concerns: pure numpy bookkeeping. No torch, no IFR, no file
writing, no campaign walking.

--------------------------------------------------------------------------
THE LABEL
--------------------------------------------------------------------------
theta_A = ( theta restricted to the 23 active run_args axes , eta )

with the topology block eta = (conn_prob, p0_conn, d0_conn, beta_conn), giving
p = 23 + 4 = 27.

Coordinate rule on the run_args axes, for each fixed k in {0, ..., 35}:

    theta_k = ln(vartheta_k)  if k in L
    theta_k = vartheta_k      otherwise

where L is derived MECHANICALLY from PARAM_BOUNDS, never from a hard-coded
list, so it tracks any future bounds edit:

    L = { k : lo_k > 0  and  hi_k > 0  and  log10(hi_k / lo_k) >= 1 }        (1)

Applying (1) to the current literals gives |L| = 27 of 36 axes; 19 of those 27
fall inside the 23 active axes, leaving 4 linear active axes (VA, VR, I_inj,
Cm). This module RE-DERIVES those counts at import rather than asserting them,
and exposes them for the sidecar.

--------------------------------------------------------------------------
TRAP: THE TOPOLOGY BLOCK IS LINEAR-UNIFORM
--------------------------------------------------------------------------
sample_kernel_vector and the conn_prob draw both call rng.uniform on NATURAL
bounds, so all four topology axes are linear-uniform. Rule (1) must NOT be run
over them: p0_conn has bounds [0.1, 1.0], i.e. exactly 1.0 decades, so (1)
would classify it as a log axis and the export would silently store ln(p0)
against a linear prior box. This module never applies (1) outside the 36-D
registry; the four topology axes are hard-coded as "linear" by construction.

--------------------------------------------------------------------------
TRAP: THE TOPOLOGY VALUES ARE NOT IN THE MEA OUTPUT
--------------------------------------------------------------------------
process_campaign.py writes conn_prob into mea_iter_<n>.npz but NOT p0_conn,
d0_conn or beta_conn. Those three exist only in the ORIGINAL
<campaign>/topo_<k>/iter_<n>.npz. The export therefore requires a join on
(topo_idx, iter_idx); see export_embeddings.py. Read the stored values, do not
recompute them from the kernel bounds.

--------------------------------------------------------------------------
TRAP: FROZEN AXES HAVE POINT-INTERVAL BOUNDS
--------------------------------------------------------------------------
DeltaT, VT and gL have lo == hi. Including any of them would give a prior
density 1 / (hi - lo) = 1 / 0 and a divergent flow loss. Assertion A3 guards
this; here they simply never enter, because they are not in the
'neuron_synapse' sweep group.

HPC note (hpc-python-compat): pure ASCII, LF-only.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TOPOLOGY_AXES",
    "SimRepoNotFound",
    "LabelSpec",
    "load_registry",
    "build_label_spec",
    "assemble_theta_A",
]

# The topology block, in the order it is appended to the label. Names match the
# npz keys written by HPC_main_sweep.py exactly.
TOPOLOGY_AXES = ("conn_prob", "p0_conn", "d0_conn", "beta_conn")

# Units, for the sidecar. conn_prob / p0_conn / beta_conn are dimensionless.
TOPOLOGY_UNITS = {
    "conn_prob": "(dimensionless)",
    "p0_conn": "(dimensionless)",
    "d0_conn": "um",
    "beta_conn": "(dimensionless)",
}


class SimRepoNotFound(ImportError):
    """Raised when the Phenomenological_finalv1 directory is not importable."""


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass
class Registry:
    """The 36-D parameter registry, as read from the simulator's own modules."""
    param_names: List[str]
    param_units: List[str]
    param_bounds: np.ndarray          # (36, 2) NATURAL units
    param_bounds_theta: np.ndarray    # (36, 2) inference coords
    log_param_indices: List[int]      # L, derived via rule (1)
    log_base: str
    kernel_bounds: np.ndarray         # (3, 2) p0, d0, beta -- NATURAL, linear
    sweep_groups: Dict[str, List[int]]


def load_registry(sim_dir: Optional[str] = None) -> Registry:
    """Import HPC_main_sweep / HPC_single_run and read the registry from them.

    Directive 1: the transforms, the bounds and rule (1) are taken from the
    simulator's own source, not re-implemented, so this export cannot disagree
    with what actually produced the data. Both modules import Brian2 only INSIDE
    functions, so importing them here is cheap and does not require a Brian2
    installation. They do import matplotlib at module scope, so the Agg backend
    is forced first (headless compute nodes).

    Parameters
    ----------
    sim_dir : str or None
        Path to <Astro-Neuron-Network>/hpc/Phenomenological_finalv1. Falls back
        to the SIM_MAIN_DIR environment variable, then to the current sys.path.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    cand = sim_dir or os.environ.get("SIM_MAIN_DIR")
    if cand:
        cand = os.path.abspath(cand)
        if not os.path.isdir(cand):
            raise SimRepoNotFound(
                "sim_dir=%r is not a directory. Point it at "
                "<Astro-Neuron-Network>/hpc/Phenomenological_finalv1" % (cand,))
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import HPC_main_sweep as SW      # noqa: E402
        import HPC_single_run as SR      # noqa: E402
    except ImportError as exc:
        raise SimRepoNotFound(
            "could not import HPC_main_sweep / HPC_single_run. Pass "
            "sim_dir=<repo>/hpc/Phenomenological_finalv1 or set SIM_MAIN_DIR. "
            "Original error: %r" % (exc,))

    param_bounds = np.asarray(SW.PARAM_BOUNDS, dtype=np.float64)
    n_dims = param_bounds.shape[0]

    # Re-derive L here from the bounds, and cross-check against the simulator's
    # own LOG_PARAMS. They must agree; if they ever do not, the two copies of
    # rule (1) have drifted and the export must stop.
    derived = sorted(
        k for k in range(n_dims)
        if param_bounds[k, 0] > 0.0 and param_bounds[k, 1] > 0.0
        and np.log10(param_bounds[k, 1] / param_bounds[k, 0]) >= 1.0)
    source = sorted(int(k) for k in SW.LOG_PARAMS)
    if derived != source:
        raise RuntimeError(
            "rule (1) re-derived from PARAM_BOUNDS gives %r but the simulator's "
            "LOG_PARAMS gives %r. Refusing to guess which is authoritative."
            % (derived, source))

    if str(SW.LOG_BASE) != "natural":
        raise RuntimeError(
            "LOG_BASE is %r, expected 'natural'. Every log coordinate in this "
            "export is ln, matching the sbi package convention; a base change "
            "would silently rescale every log axis." % (SW.LOG_BASE,))

    return Registry(
        param_names=list(SR.PARAM_NAMES),
        param_units=list(SR.PARAM_UNITS),
        param_bounds=param_bounds,
        param_bounds_theta=np.asarray(SW.PARAM_BOUNDS_THETA, dtype=np.float64),
        log_param_indices=derived,
        log_base=str(SW.LOG_BASE),
        kernel_bounds=np.asarray(SW.KERNEL_BOUNDS, dtype=np.float64),
        sweep_groups={k: sorted(int(i) for i in v)
                      for k, v in SR.SWEEP_GROUPS.items()},
    )


def registry_from_manifest(reg: Registry, manifest: Dict) -> Tuple[Registry, List[str], List[str]]:
    """Re-source the bounds AND the coordinate rule from the campaign's own
    manifest.json, overriding whatever the live simulator source says.

    Why this must exist
    -------------------
    load_registry() derives L via rule (1) from the CURRENT PARAM_BOUNDS. That
    makes the interpretation of already-generated data a function of code that
    may have moved on since. A bound edited between campaigns can push an axis
    across rule (1)'s one-decade threshold and silently flip its coordinate:
    Sigma at [1.0, 10.0] is exactly 1.000000 decades and therefore ln; widen the
    floor to [2.0, 10.0] and the same axis becomes linear, so a stored ln value
    is read as a natural one. Every campaign's manifest.json records the
    param_bounds and log_params that were actually in force when its npz files
    were written, so that is the only defensible source.

    Returns
    -------
    (registry, changed, flipped)
        registry : a copy with param_bounds, param_bounds_theta and
                   log_param_indices taken from the manifest.
        changed  : names whose natural bounds differ from the live source.
        flipped  : names whose COORDINATE differs from the live source -- the
                   dangerous subset of `changed`.
    """
    pb_raw = manifest.get("param_bounds")
    if pb_raw is None:
        return reg, [], []

    pb = np.asarray(pb_raw, dtype=np.float64)
    if pb.shape != reg.param_bounds.shape:
        raise ValueError(
            "manifest.json param_bounds is %r but the registry is %r. A width "
            "mismatch means this npz was written against a different registry "
            "version; every active index would be mis-mapped."
            % (pb.shape, reg.param_bounds.shape))

    names = manifest.get("param_names")
    if names is not None and list(names) != list(reg.param_names):
        raise ValueError(
            "manifest.json param_names differ from the loaded registry's. "
            "Bounds are matched positionally, so a reordering would put every "
            "axis on the wrong prior interval.")

    if str(manifest.get("log_transform", "natural")) != "natural":
        raise ValueError(
            "manifest.json log_transform is %r, expected 'natural'."
            % (manifest.get("log_transform"),))

    n = pb.shape[0]
    derived = sorted(k for k in range(n)
                     if pb[k, 0] > 0.0 and pb[k, 1] > 0.0
                     and np.log10(pb[k, 1] / pb[k, 0]) >= 1.0)

    # Same cross-check load_registry applies to the live source, applied here
    # to the manifest: rule (1) re-derived from the recorded bounds must agree
    # with the recorded log_params, or the manifest is internally inconsistent.
    lp = manifest.get("log_params")
    if lp is not None:
        idx = {nm: i for i, nm in enumerate(reg.param_names)}
        unknown = [nm for nm in lp if nm not in idx]
        if unknown:
            raise ValueError(
                "manifest.json log_params names %r absent from the registry"
                % (unknown,))
        recorded = sorted(idx[nm] for nm in lp)
        if recorded != derived:
            only_rec = [reg.param_names[k] for k in sorted(set(recorded) - set(derived))]
            only_der = [reg.param_names[k] for k in sorted(set(derived) - set(recorded))]
            raise ValueError(
                "manifest.json is internally inconsistent: rule (1) on its own "
                "param_bounds gives a log set differing from its log_params. "
                "Recorded-only: %r. Derived-only: %r. Refusing to guess which "
                "is authoritative." % (only_rec, only_der))

    bt = pb.copy()
    is_log = np.zeros(n, dtype=bool)
    is_log[derived] = True
    if np.any(pb[is_log] <= 0.0):
        raise ValueError("a log axis has a non-positive bound in manifest.json")
    bt[is_log] = np.log(pb[is_log])

    changed = [reg.param_names[k] for k in range(n)
               if not np.allclose(pb[k], reg.param_bounds[k],
                                  rtol=0.0, atol=0.0, equal_nan=True)]
    flipped = [reg.param_names[k]
               for k in sorted(set(derived) ^ set(reg.log_param_indices))]

    return (replace(reg, param_bounds=pb, param_bounds_theta=bt,
                    log_param_indices=derived),
            changed, flipped)



# --------------------------------------------------------------------------- #
# label specification
# --------------------------------------------------------------------------- #
@dataclass
class LabelSpec:
    """Everything needed to build one theta_A row and to describe it in a sidecar."""

    param_names: List[str]            # p names, exported th_* column order
    coord: List[str]                  # p entries, each "ln" or "linear"
    bounds_theta: np.ndarray          # (p, 2)
    units: List[str]
    active_indices: List[int]         # the registry indices, in column order
    registry: Registry = field(repr=False, default=None)
    sweep_group: str = ""

    # The topology-level axes actually carried in theta, in column order. This
    # is NOT hardcoded: an axis that the campaign DREW but the simulator never
    # READ (conn_prob under --conn_rule weibull) is causally inert and must be
    # excluded, which no variance scan can discover. Defaults to the legacy
    # 4-axis block so old callers are unaffected.
    topology_axes: List[str] = field(
        default_factory=lambda: list(TOPOLOGY_AXES))
    # name -> human-readable reason, carried into the sidecar for provenance.
    # An excluded axis is never silently dropped: it is recorded here.
    excluded_axes: Dict[str, str] = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.param_names)

    @property
    def column_names(self) -> List[str]:
        return ["th_%s" % n for n in self.param_names]

    def sidecar_registry_block(self) -> dict:
        reg = self.registry
        frozen = [reg.param_names[k] for k in range(len(reg.param_names))
                  if reg.param_bounds[k, 0] == reg.param_bounds[k, 1]]
        return {
            "param_names_36": list(reg.param_names),
            "param_units_36": list(reg.param_units),
            "log_param_indices": list(reg.log_param_indices),
            "n_log_axes": len(reg.log_param_indices),
            "log_transform": reg.log_base,
            "active_indices": list(self.active_indices),
            "frozen_excluded": frozen,
            "sweep_group": self.sweep_group,
            "topology_axes": list(TOPOLOGY_AXES),
            "topology_coord": "linear",
            "kernel_bounds_p0_d0_beta": reg.kernel_bounds.tolist(),
        }


def build_label_spec(registry: Registry,
                     active_indices: Sequence[int],
                     sweep_group: str = "neuron_synapse",
                     conn_prob_bounds: Tuple[float, float] = (0.1, 0.6),
                     kernel_bounds: Optional[np.ndarray] = None,
                     topology_axes: Optional[Sequence[str]] = None,
                     excluded_axes: Optional[Dict[str, str]] = None) -> LabelSpec:
    """Build the p = len(active_indices) + len(topology_axes) label spec.

    Parameters
    ----------
    registry : Registry
    active_indices : sequence of int
        The free run_args axes, taken from manifest.json's 'active_indices'
        (which the sweep itself wrote). Passing them in rather than resolving
        the sweep-group name here means the export follows what the CAMPAIGN
        actually did, not what the current source would do.
    sweep_group : str
        Recorded for provenance only.
    conn_prob_bounds : (lo, hi)
        The conn_prob prior, from the campaign's --conn_prob_lo/--conn_prob_hi
        launch flags in job_args.json. conn_prob is drawn in the outer topology
        loop and has no entry in KERNEL_BOUNDS, so it must be supplied.
    kernel_bounds : (3, 2) array or None
        (p0, d0, beta) bounds ACTUALLY used by the campaign, i.e. after any
        --p0_lo/--d0_hi/--beta_lo overrides. Defaults to the module constant.
    topology_axes : sequence of str or None
        Which topology-level axes enter theta, in column order. None keeps the
        legacy 4-axis block. Normally supplied from a frozen label_axes.json
        (see preflight_label_axes.py), so that EVERY shard agrees on the
        column set: deciding this per shard would give different p per shard
        and silently unpoolable theta matrices.
    excluded_axes : dict or None
        name -> reason, for axes deliberately kept OUT of theta. Recorded in
        the spec (and thus the sidecar) rather than dropped silently.

    Returns
    -------
    LabelSpec
    """
    act = [int(k) for k in active_indices]
    n_reg = len(registry.param_names)
    if not act:
        raise ValueError("active_indices is empty")
    if len(set(act)) != len(act):
        raise ValueError("active_indices contains duplicates: %r" % (act,))
    for k in act:
        if not (0 <= k < n_reg):
            raise ValueError(
                "active index %d out of range for a %d-axis registry" % (k, n_reg))

    kb = registry.kernel_bounds if kernel_bounds is None else \
        np.asarray(kernel_bounds, dtype=np.float64)
    if kb.shape != (3, 2):
        raise ValueError("kernel_bounds must be (3, 2); got %r" % (kb.shape,))

    log_set = set(registry.log_param_indices)

    names = [registry.param_names[k] for k in act]
    units = [registry.param_units[k] for k in act]
    coord = ["ln" if k in log_set else "linear" for k in act]
    bounds = [registry.param_bounds_theta[k].tolist() for k in act]

    # --- topology block: ALWAYS linear, ALWAYS natural units ---------------
    # Bounds are looked up BY NAME, so dropping an axis (or reordering) cannot
    # silently shift a column onto the wrong prior interval -- which is what a
    # positional list would do the moment conn_prob is excluded.
    cp_lo, cp_hi = float(conn_prob_bounds[0]), float(conn_prob_bounds[1])
    topo_bounds_by_name = {
        "conn_prob": [cp_lo, cp_hi],
        "p0_conn":   [float(kb[0, 0]), float(kb[0, 1])],
        "d0_conn":   [float(kb[1, 0]), float(kb[1, 1])],
        "beta_conn": [float(kb[2, 0]), float(kb[2, 1])],
    }
    topo_axes = list(TOPOLOGY_AXES) if topology_axes is None \
        else [str(a) for a in topology_axes]
    unknown = [a for a in topo_axes if a not in topo_bounds_by_name]
    if unknown:
        raise ValueError(
            "unknown topology axis/axes %r; this function knows bounds only "
            "for %r. A new topology-level swept parameter needs its prior "
            "interval plumbed through here (and into the sweep's job_args) "
            "before it can enter theta." % (unknown, sorted(topo_bounds_by_name)))
    if len(set(topo_axes)) != len(topo_axes):
        raise ValueError("duplicate topology axes: %r" % (topo_axes,))

    names += topo_axes
    units += [TOPOLOGY_UNITS[a] for a in topo_axes]
    coord += ["linear"] * len(topo_axes)
    bounds += [topo_bounds_by_name[a] for a in topo_axes]

    B = np.asarray(bounds, dtype=np.float64)

    # A3, enforced at construction: a degenerate box is unusable as a prior.
    width = B[:, 1] - B[:, 0]
    bad = [(names[i], float(B[i, 0]), float(B[i, 1]))
           for i in range(len(names)) if not (width[i] > 0.0)]
    if bad:
        raise ValueError(
            "assertion A3 failed at label construction: point-interval bounds "
            "for %r. A point interval gives prior density 1/0 and a divergent "
            "flow loss. Frozen axes (DeltaT, VT, gL) must not be active." % (bad,))

    if len(set(names)) != len(names):
        raise ValueError("duplicate label names: %r" % (names,))

    return LabelSpec(param_names=names, coord=coord, bounds_theta=B,
                     units=units, active_indices=act, registry=registry,
                     sweep_group=sweep_group, topology_axes=topo_axes,
                     excluded_axes=dict(excluded_axes or {}))


# --------------------------------------------------------------------------- #
# one row
# --------------------------------------------------------------------------- #
def assemble_theta_A(spec: LabelSpec,
                     theta_36: np.ndarray,
                     topology: Dict[str, float],
                     params_36: Optional[np.ndarray] = None,
                     check_coord_tol: float = 1e-8) -> np.ndarray:
    """Build one theta_A row, in spec.param_names order.

    Parameters
    ----------
    spec : LabelSpec
    theta_36 : (36,) array
        The 'theta' array stored in the npz -- ALREADY in inference coordinates.
        It is sliced, never re-transformed: re-deriving it from 'params' would
        reintroduce exactly the coordinate ambiguity the stored array exists to
        remove.
    topology : dict
        Must contain all four TOPOLOGY_AXES keys, in NATURAL units.
    params_36 : (36,) array or None
        The natural-unit vector. When given, a spot-check verifies
        theta_k = ln(params_k) on log axes and theta_k = params_k otherwise
        (handoff assertion A6, which catches a stale registry).
    check_coord_tol : float
        Relative tolerance for that spot-check.

    Returns
    -------
    theta_A : (p,) float64
    """
    theta_36 = np.asarray(theta_36, dtype=np.float64).ravel()
    n_reg = len(spec.registry.param_names)
    if theta_36.shape[0] != n_reg:
        raise ValueError(
            "theta has %d entries, expected %d (the registry width). A width "
            "mismatch means the npz was written by a different manifest "
            "version; every active index would be mis-mapped."
            % (theta_36.shape[0], n_reg))

    if params_36 is not None:
        params_36 = np.asarray(params_36, dtype=np.float64).ravel()
        if params_36.shape[0] != n_reg:
            raise ValueError("params has %d entries, expected %d"
                             % (params_36.shape[0], n_reg))
        log_set = set(spec.registry.log_param_indices)
        for k in spec.active_indices:
            want = np.log(params_36[k]) if k in log_set else params_36[k]
            got = theta_36[k]
            if not np.isclose(got, want, rtol=check_coord_tol, atol=1e-10):
                raise ValueError(
                    "assertion A6 failed on axis %d (%s): stored theta=%r but "
                    "the coordinate rule applied to params gives %r. The "
                    "registry used to write this npz disagrees with the one "
                    "loaded here."
                    % (k, spec.registry.param_names[k], got, want))

    topo_axes = list(spec.topology_axes)
    missing = [a for a in topo_axes if a not in topology]
    if missing:
        raise KeyError(
            "topology block is missing %r. p0_conn / d0_conn / beta_conn are "
            "NOT written into mea_iter_*.npz; they must be joined from the "
            "original topo_*/iter_*.npz on (topo_idx, iter_idx)." % (missing,))

    head = theta_36[list(spec.active_indices)]
    tail = np.array([float(topology[a]) for a in topo_axes], dtype=np.float64)

    if not np.all(np.isfinite(tail)):
        raise ValueError(
            "topology block contains NaN or Inf: %r. p0_conn / d0_conn / "
            "beta_conn are written as NaN under the FLAT connectivity rule; "
            "such a campaign has no Weibull kernel and cannot contribute the "
            "topology block." % (dict(zip(topo_axes, tail)),))

    row = np.concatenate([head, tail])
    if row.shape[0] != spec.p:
        raise RuntimeError("assembled %d entries, expected %d" % (row.shape[0], spec.p))
    return row
