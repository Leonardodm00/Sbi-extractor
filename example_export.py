#!/usr/bin/env python3
"""
example_export.py
=================

A COMPLETE, RUNNABLE wiring example for the SBI export chain, and the template
you adapt into the Stage-5 campaign walker.

It runs in two modes:

  --mode synthetic   (default) Fabricates detected spike trains in memory and
                     runs the whole chain: IFR -> windows -> forward pass ->
                     Parquet + sidecar. Needs NO campaign, NO checkpoint and NO
                     real recordings. Use this to verify your environment and to
                     see the shape of the output before touching real data.

  --mode campaign    Walks a real MEA-processed campaign. This is the path you
                     will actually use. It is deliberately written with glob
                     patterns rather than hard-coded filename formats, so a
                     different zero-padding will not break it -- but you MUST
                     verify the npz key names against your own files before
                     trusting a production export (see VERIFY THIS list below).

--------------------------------------------------------------------------
VERIFY THIS BEFORE A PRODUCTION RUN (--mode campaign)
--------------------------------------------------------------------------
  1. <mea_out>/topo_*/mea_iter_*.npz really contains det_t, det_ch,
     electrode_centers, theta, topo_idx, iter_idx, seed_run.
  2. <campaign>/topo_*/iter_*.npz really contains p0_conn, d0_conn, beta_conn,
     conn_prob. These are NOT in the mea_iter file and must be joined.
  3. <campaign>/manifest.json has active_indices and sweep_group.
  4. <campaign>/job_args.json has simtime, conn_prob_lo, conn_prob_hi.
     The --simtime FLAG is the duration used for the IFR grid. NEVER use the
     'simtime' field inside mea_iter_*.npz: process_campaign.py computes it as
     ceil(last spike time), so it varies per run and silently produces traces
     shorter than the DSN window.

Print the keys of one file with:

    python3 -c "import numpy as np; d=np.load('PATH.npz'); print(sorted(d.files))"

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    export DSN_MAIN_DIR=$HOME/repos/Deep-Summary-Network/Main
    export SIM_MAIN_DIR=$HOME/repos/Astro-Neuron-Network/hpc/Phenomenological_finalv1

    # 1. environment check, no data needed
    python3 example_export.py --mode synthetic --out /tmp/demo

    # 2. real campaign
    python3 example_export.py --mode campaign \\
        --checkpoint   $HOME/runs/mea_joint_full/checkpoints/best.pt \\
        --campaign     $HOME/campaigns/cadex_ns_001 \\
        --mea_out      $HOME/campaigns/cadex_ns_001_mea \\
        --campaign_id  cadex_ns_001 \\
        --out          $HOME/export/sbi_cadex_ns_001_0000

HPC note (hpc-python-compat): pure ASCII, LF-only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dsn_frozen import load_frozen_dsn, FrozenDSN           # noqa: E402
from sim_observable import build_pooled_ifr                  # noqa: E402
from sbi_labels import (load_registry, build_label_spec,     # noqa: E402
                        assemble_theta_A, registry_from_manifest)
from export_embeddings import (TraceRecord, export_embeddings,  # noqa: E402
                               run_assertion_A1)

_IDX_RE = re.compile(r"(\d+)")


def _trailing_index(path) -> int:
    """Last integer in a filename, e.g. mea_iter_00042.npz -> 42."""
    hits = _IDX_RE.findall(os.path.basename(path))
    if not hits:
        raise ValueError("no numeric index in %r" % (path,))
    return int(hits[-1])


def _load_label_axes(path):
    """Return (topology_axes, excluded_axes) from a frozen label_axes.json.

    Returns (None, {}) when no path is given, which makes build_label_spec
    fall back to its legacy 4-axis block. Passing the file is what guarantees
    every shard shares one column set: letting each shard decide from its own
    rows would give a different p per shard and theta matrices that cannot be
    concatenated.
    """
    if not path:
        return None, {}
    with open(path) as fh:
        doc = json.load(fh)
    axes = doc.get("topology_axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError(
            "%s has no non-empty 'topology_axes' list. Regenerate it with "
            "preflight_label_axes.py." % (path,))
    excluded = {}
    for k, v in (doc.get("excluded_axes") or {}).items():
        excluded[k] = str(v.get("reason", v)) if isinstance(v, dict) else str(v)
    print("      label axes from %s: %d topology axis/axes %r"
          % (path, len(axes), axes))
    return [str(a) for a in axes], excluded


# --------------------------------------------------------------------------- #
# campaign walking
# --------------------------------------------------------------------------- #
def record_resolved_n_e(observable_out, n_e, source):
    """Write the n_e that was ACTUALLY used into the sidecar's observable block.

    Why this is not just echoing --n_electrodes: that flag is normally omitted,
    because n_e is read from electrode_centers. Echoing it then writes null into
    the sidecar of a shard whose amplitude scale depends entirely on n_e (the
    pooled IFR is a MEAN over exactly n_e electrodes). A shard that does not
    record its own n_e cannot afterwards be checked for preprocessing parity
    against the real arm, or against another simulated bank -- the comparison
    comes back "unknown", never "agree", which is safe but useless.

    Doubles as a within-shard consistency check: n_e must be the same for every
    trace in one shard, since a single sidecar declares one value for all rows.
    """
    if observable_out is None:
        return
    prev = observable_out.get("n_electrodes")
    if prev is not None and int(prev) != int(n_e):
        raise ValueError(
            "n_e is not constant within this shard: %r then %r. One sidecar "
            "declares a single n_electrodes for every row, and the pooled IFR "
            "is divided by it, so a mixed shard is silently mis-scaled."
            % (int(prev), int(n_e)))
    observable_out["n_electrodes"] = int(n_e)
    observable_out["n_electrodes_source"] = str(source)


def iter_campaign_records(campaign_dir, mea_out_dir, spec, T_sim, campaign_id,
                          n_electrodes=None, dt=0.02, sigma_sm=0.04,
                          verify_coords=True, max_records=None,
                          trim_head_s=0.0, observable_out=None):
    """Yield one TraceRecord per completed simulation.

    The join. mea_iter_*.npz has the detections and theta but NOT the Weibull
    kernel parameters; iter_*.npz has those. They are matched on
    (topo_idx, iter_idx), which is the key the handoff assumes is unique. If it
    is not unique in your campaign, this function will raise rather than invent
    a surrogate key.
    """
    topo_dirs = sorted(glob.glob(os.path.join(mea_out_dir, "topo_*")))
    if not topo_dirs:
        raise FileNotFoundError(
            "no topo_* directories under %r. Has process_campaign.py been run "
            "over this campaign yet? Without it there are no detections, and "
            "the export cannot be built from ground-truth spikes without "
            "breaking parity with the real recordings." % (mea_out_dir,))

    n_yield = 0
    for td in topo_dirs:
        topo_name = os.path.basename(td)
        src_topo = os.path.join(campaign_dir, topo_name)
        for mea_path in sorted(glob.glob(os.path.join(td, "mea_iter_*.npz"))):
            iter_idx = _trailing_index(mea_path)

            with np.load(mea_path, allow_pickle=False) as m:
                keys = set(m.files)
                for need in ("det_t", "det_ch", "theta"):
                    if need not in keys:
                        raise KeyError(
                            "%s has no %r (found %r). Verify the MEA output "
                            "schema before exporting."
                            % (mea_path, need, sorted(keys)))
                det_t = np.asarray(m["det_t"], dtype=np.float64)
                det_ch = np.asarray(m["det_ch"], dtype=np.int64)
                theta36 = np.asarray(m["theta"], dtype=np.float64)
                params36 = (np.asarray(m["params"], dtype=np.float64)
                            if "params" in keys else None)
                topo_idx = int(m["topo_idx"]) if "topo_idx" in keys \
                    else _trailing_index(td)
                seed_run = int(m["seed_run"]) if "seed_run" in keys else -1
                if n_electrodes is None:
                    if "electrode_centers" not in keys:
                        raise KeyError(
                            "%s has no electrode_centers, so n_e cannot be "
                            "determined. Pass --n_electrodes explicitly; do NOT "
                            "guess, because n_e sets the amplitude scale."
                            % (mea_path,))
                    n_e = int(np.asarray(m["electrode_centers"]).shape[0])
                    n_e_source = "electrode_centers"
                else:
                    n_e = int(n_electrodes)
                    n_e_source = "--n_electrodes"
                record_resolved_n_e(observable_out, n_e, n_e_source)

            # --- the join, for the topology block ---------------------------
            cand = sorted(glob.glob(os.path.join(src_topo, "iter_*.npz")))
            match = [p for p in cand if _trailing_index(p) == iter_idx]
            if len(match) != 1:
                raise FileNotFoundError(
                    "expected exactly one %s/iter_*.npz with index %d, found "
                    "%d. (topo_idx, iter_idx) is assumed to be a unique row "
                    "key; it is not here."
                    % (src_topo, iter_idx, len(match)))
            with np.load(match[0], allow_pickle=False) as s:
                # Read exactly the axes the frozen spec asks for -- not a
                # hardcoded tuple. An axis the sweep DREW but the simulator
                # never READ (conn_prob under --conn_rule weibull) is absent
                # from spec.topology_axes and is therefore never joined.
                topo = {}
                for a in spec.topology_axes:
                    if a not in s.files:
                        raise KeyError(
                            "%s has no %r, but the frozen label spec requires "
                            "it. Regenerate label_axes.json against the "
                            "campaigns actually being exported."
                            % (match[0], a))
                    topo[a] = float(s[a])

            theta_A = assemble_theta_A(
                spec, theta36, topo,
                params_36=params36 if verify_coords else None)

            # --- the observable ---------------------------------------------
            # Pooled over ALL member electrodes, divided by n_e, on a FIXED
            # T_sim grid taken from the launch flag.
            per_e = [det_t[det_ch == e] for e in range(n_e)]
            x = build_pooled_ifr(per_e, n_electrodes=n_e, T=T_sim,
                                 dt=dt, sigma_sm=sigma_sm)

            # --- burn-in trim -----------------------------------------------
            # A simulation starts from initial conditions and settles; a real
            # recording is already at steady state when the acquisition
            # begins. Keeping the first seconds of a simulation therefore
            # puts a transient in EVERY simulated row and in NO real row,
            # which a two-sample test in summary space reports as decisive
            # misspecification when it is an initial-condition artefact.
            #
            # The IFR is built over the FULL [0, T_sim) grid and only then
            # sliced. Rebuilding it on [trim_head_s, T_sim) instead would put
            # a reflected gaussian_filter1d boundary at the new left edge,
            # where slicing leaves the real neighbouring bins in place.
            if trim_head_s > 0.0:
                n_trim = int(round(float(trim_head_s) / float(dt)))
                if n_trim >= x.shape[0]:
                    raise ValueError(
                        'trim_head_s = %.4g s removes the whole trace '
                        '(%d of %d bins at dt = %.4g s)'
                        % (trim_head_s, n_trim, x.shape[0], dt))
                x = np.ascontiguousarray(x[n_trim:])

            yield TraceRecord(
                trace=x, theta_A=theta_A,
                ident={"campaign_id": campaign_id, "topo_idx": topo_idx,
                       "iter_idx": iter_idx, "seed_run": seed_run})

            n_yield += 1
            if max_records is not None and n_yield >= max_records:
                return


# --------------------------------------------------------------------------- #
# synthetic demo
# --------------------------------------------------------------------------- #
def iter_synthetic_records(spec, T_sim, n_sims, n_electrodes, dt, sigma_sm,
                           seed=0):
    """Fabricated detections, for an environment check. NOT biology."""
    rng = np.random.default_rng(seed)
    lo, hi = spec.bounds_theta[:, 0], spec.bounds_theta[:, 1]
    for i in range(n_sims):
        centers = np.sort(rng.uniform(0.0, T_sim, max(1, int(0.3 * T_sim))))
        per_e = []
        for _e in range(n_electrodes):
            parts = [rng.uniform(0.0, T_sim, int(rng.poisson(2.0 * T_sim)))]
            for c in centers:
                k = int(rng.poisson(10))
                if k:
                    parts.append(np.clip(rng.normal(c, 0.05, k),
                                         0.0, T_sim - 1e-9))
            per_e.append(np.sort(np.concatenate(parts)))
        x = build_pooled_ifr(per_e, n_electrodes, T_sim, dt, sigma_sm)

        draw = rng.uniform(lo, hi)
        theta36 = np.zeros(len(spec.registry.param_names), dtype=np.float64)
        for j, k in enumerate(spec.active_indices):
            theta36[k] = draw[j]
        topo = dict(zip(("conn_prob", "p0_conn", "d0_conn", "beta_conn"),
                        (float(v) for v in draw[-4:])))
        yield TraceRecord(
            trace=x, theta_A=assemble_theta_A(spec, theta36, topo),
            ident={"campaign_id": "synthetic_demo", "topo_idx": i // 8,
                   "iter_idx": i, "seed_run": 1000 + i})


def _make_demo_dsn(dsn_main_dir, W, dt):
    """A randomly initialised backbone, so the demo needs no checkpoint.

    The embeddings are MEANINGLESS -- untrained weights. This exists only to
    prove the plumbing, and the sidecar records a fake digest so the file can
    never be mistaken for a real export.
    """
    import torch
    sys.path.insert(0, dsn_main_dir)
    from backbone import BackboneConfig, build_backbone
    cfg = BackboneConfig(depth_exponent=2, width_multiplier=2.0, stem_width=8,
                         in_channels=1, embedding_size=16, l2_normalize=True,
                         head_fusion=True, head_pool_ops=("mean",))
    model = build_backbone(cfg)
    model.eval()
    return FrozenDSN(
        model=model, device=torch.device("cpu"),
        ckpt_path="(UNTRAINED DEMO -- NOT A REAL CHECKPOINT)",
        ckpt_sha256="0" * 64, embedding_dim=16, l2_normalize=True,
        in_channels=1, window_s=W * dt, w_size=dt, gaussian_window=0.04,
        window_length=W,
        warnings=["DEMO MODE: the encoder is UNTRAINED and the SHA-256 is a "
                  "placeholder. These embeddings carry no information and must "
                  "never be used for inference."])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("synthetic", "campaign"),
                    default="synthetic")
    ap.add_argument("--out", required=True,
                    help="output path stem, WITHOUT extension")
    ap.add_argument("--dsn_main_dir", default=os.environ.get("DSN_MAIN_DIR"))
    ap.add_argument("--sim_dir", default=os.environ.get("SIM_MAIN_DIR"))
    ap.add_argument("--checkpoint", default=None,
                    help="DSN .pt checkpoint (required for --mode campaign)")
    ap.add_argument("--campaign", default=None, help="the sweep output dir")
    ap.add_argument("--mea_out", default=None,
                    help="process_campaign.py output dir")
    ap.add_argument("--campaign_id", default="campaign")
    ap.add_argument("--n_electrodes", type=int, default=None,
                    help="n_e. Read from electrode_centers when omitted.")
    ap.add_argument("--simtime", type=float, default=None,
                    help="T [s]. Read from job_args.json when omitted. "
                         "NEVER taken from the npz.")
    ap.add_argument("--trim_head_s", type=float, default=0.0,
                    help="Discard the first N seconds of every "
                         "simulated trace as burn-in, BEFORE windowing. "
                         "The usable duration becomes T - trim_head_s and "
                         "must still be >= the DSN window.")
    ap.add_argument("--sweep_group", default="neuron_synapse")
    ap.add_argument("--label_axes", default=None,
                    help="Frozen label_axes.json from preflight_label_axes.py. "
                         "Fixes WHICH topology-level axes enter theta, and in "
                         "what order, IDENTICALLY across every shard. Omit "
                         "only to reproduce the legacy 4-axis behaviour.")
    ap.add_argument("--conn_prob_lo", type=float, default=None)
    ap.add_argument("--conn_prob_hi", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_records", type=int, default=None,
                    help="stop after N simulations (for a quick dry run)")
    ap.add_argument("--n_sims", type=int, default=32,
                    help="synthetic mode only")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not args.sim_dir:
        ap.error("--sim_dir (or SIM_MAIN_DIR) is required: the parameter "
                 "registry and the coordinate transforms are read from it.")
    if not args.dsn_main_dir:
        ap.error("--dsn_main_dir (or DSN_MAIN_DIR) is required.")

    # ---- registry + label spec -------------------------------------------
    print("[1/5] loading the 36-D registry from %s" % args.sim_dir)
    reg = load_registry(args.sim_dir)
    run_assertion_A1(reg)
    print("      A1 passed; |L| = %d log axes of %d"
          % (len(reg.log_param_indices), len(reg.param_names)))

    manifest, job_args = {}, {}
    if args.mode == "campaign":
        for req, name in ((args.checkpoint, "--checkpoint"),
                          (args.campaign, "--campaign"),
                          (args.mea_out, "--mea_out")):
            if not req:
                ap.error("%s is required for --mode campaign" % name)
        mpath = os.path.join(args.campaign, "manifest.json")
        jpath = os.path.join(args.campaign, "job_args.json")
        if os.path.isfile(mpath):
            with open(mpath) as fh:
                manifest = json.load(fh)
        if os.path.isfile(jpath):
            with open(jpath) as fh:
                job_args = json.load(fh)

    # The bounds and the coordinate rule must come from the campaign that
    # WROTE these npz files, not from whatever the simulator source says today.
    # A bound edited between campaigns can move an axis across rule (1)'s
    # one-decade threshold and flip ln <-> linear underneath already-generated
    # data. manifest.json records what was actually in force.
    if manifest.get("param_bounds") is not None:
        reg, changed, flipped = registry_from_manifest(reg, manifest)
        print("      bounds re-sourced from manifest.json (manifest_version %s)"
              % (manifest.get("manifest_version"),))
        if changed:
            print("      differ from the live simulator source on: %s"
                  % ", ".join(changed))
        if flipped:
            print("      COORDINATE FLIP vs the live source on: %s "
                  "-- the live source would have mis-read these axes"
                  % ", ".join(flipped))
        run_assertion_A1(reg)
        print("      A1 re-checked; |L| = %d log axes of %d"
              % (len(reg.log_param_indices), len(reg.param_names)))
    else:
        print("      manifest.json records no param_bounds; falling back to "
              "the live simulator source. Bounds are NOT pinned to this "
              "campaign's vintage.")

    active = manifest.get("active_indices")
    sweep_group = manifest.get("sweep_group", args.sweep_group)
    if active is None:
        active = reg.sweep_groups[args.sweep_group]
        print("      manifest.json gave no active_indices; falling back to "
              "sweep group %r (%d axes)" % (args.sweep_group, len(active)))
    else:
        print("      active_indices from manifest.json: %d axes, group %r"
              % (len(active), sweep_group))

    cp_lo = args.conn_prob_lo if args.conn_prob_lo is not None \
        else float(job_args.get("conn_prob_lo", 0.1))
    cp_hi = args.conn_prob_hi if args.conn_prob_hi is not None \
        else float(job_args.get("conn_prob_hi", 0.6))

    kb = reg.kernel_bounds.copy()
    for row, (klo, khi) in enumerate((("p0_lo", "p0_hi"),
                                      ("d0_lo", "d0_hi"),
                                      ("beta_lo", "beta_hi"))):
        # NB: `klo in job_args` is True even when the value is JSON null, and
        # every campaign encodes "not swept" as present-but-null. float(None)
        # raises. null and absent must both fall through to the registry
        # default; only a real number overrides it.
        if job_args.get(klo) is not None:
            kb[row, 0] = float(job_args[klo])
        if job_args.get(khi) is not None:
            kb[row, 1] = float(job_args[khi])

    topo_axes, excluded = _load_label_axes(args.label_axes)
    spec = build_label_spec(reg, active, sweep_group,
                            conn_prob_bounds=(cp_lo, cp_hi), kernel_bounds=kb,
                            topology_axes=topo_axes, excluded_axes=excluded)
    print("[2/5] label spec built: p = %d (%d run_args + %d topology: %s)"
          % (spec.p, len(spec.active_indices), len(spec.topology_axes),
             ", ".join(spec.topology_axes)))
    for nm, why in sorted(spec.excluded_axes.items()):
        print("      EXCLUDED from theta: %s -- %s" % (nm, why))

    # ---- encoder ----------------------------------------------------------
    T_sim = args.simtime
    if T_sim is None:
        T_sim = float(job_args.get("simtime", 180.0))

    if args.mode == "campaign":
        print("[3/5] loading the frozen DSN from %s" % args.checkpoint)
        dsn = load_frozen_dsn(args.checkpoint, device=args.device,
                              dsn_main_dir=args.dsn_main_dir)
    else:
        dt_demo = 0.02
        W_demo = int(round(T_sim / dt_demo))
        print("[3/5] DEMO MODE: building an UNTRAINED backbone (no checkpoint)")
        dsn = _make_demo_dsn(args.dsn_main_dir, W_demo, dt_demo)

    print("      E = %d, W = %d samples, T_win = %.4g s, fs_ifr = %.4g Hz, "
          "Delta_t = %.4g s, sigma_sm = %.4g s"
          % (dsn.embedding_dim, dsn.window_length, dsn.window_s, dsn.fs_ifr,
             dsn.w_size, dsn.gaussian_window))
    print("      checkpoint sha256 = %s" % dsn.ckpt_sha256)
    for w in dsn.warnings:
        print("      WARNING: %s" % w)

    T_usable = T_sim - float(args.trim_head_s)
    if T_usable < dsn.window_s:
        raise SystemExit(
            "FATAL: the usable duration T - trim_head_s = %.4g - %.4g = %.4g s "
            "is shorter than the DSN window T_win = %.4g s. Every trace would "
            "be silently dropped. Either the wrong checkpoint is being used, "
            "or --simtime / --trim_head_s is wrong."
            % (T_sim, args.trim_head_s, T_usable, dsn.window_s))

    extra = {
        "campaign_id": args.campaign_id,
        "simulation": {
            "simtime_s": float(T_sim),
            "trim_head_s": float(args.trim_head_s),
            "usable_duration_s": float(T_usable),
            "sweep_group": sweep_group,
            "mode": job_args.get("mode"),
            "conn_rule": job_args.get("conn_rule"),
            "Nn": job_args.get("Nn"),
        },
        # Which axes entered theta, which did not, and why. Without this a
        # future reader cannot tell a deliberately excluded inert axis from
        # one that was forgotten.
        "label_axes": {
            "source": args.label_axes or "(legacy default, no frozen file)",
            "topology_axes": list(spec.topology_axes),
            "excluded_axes": dict(spec.excluded_axes),
            "p": int(spec.p),
        },
        # n_electrodes is left unset here on purpose: iter_campaign_records
        # fills it with the value it RESOLVED (from electrode_centers, or from
        # the flag when given) via record_resolved_n_e. This dict is the same
        # object export_embeddings merges into the sidecar, and that merge
        # happens after the record generator has been fully consumed, so the
        # resolved value is present by then. smoke_test_resolved_n_e.py pins
        # that ordering.
        "observable": {
            "electrode_forward_model": True,
        },
        "provenance": {
            "n_attempted": manifest.get("n_total_runs"),
            "n_failed_discarded": manifest.get("n_total_failures_discarded"),
            "n_topologies": manifest.get("n_topologies_completed_or_partial"),
            "manifest_version": manifest.get("manifest_version"),
            "launch_flags": job_args,
        },
    }

    # ---- records ----------------------------------------------------------
    print("[4/5] building observables (T = %.4g s, trim_head = %.4g s, "
          "usable = %.4g s, pooled and divided by n_e)"
          % (T_sim, args.trim_head_s, T_usable))
    if args.mode == "campaign":
        records = iter_campaign_records(
            args.campaign, args.mea_out, spec, T_sim, args.campaign_id,
            n_electrodes=args.n_electrodes, dt=dsn.w_size,
            observable_out=extra["observable"],
            sigma_sm=dsn.gaussian_window, max_records=args.max_records,
            trim_head_s=args.trim_head_s)
    else:
        records = iter_synthetic_records(
            spec, T_sim, args.n_sims, args.n_electrodes or 9,
            dsn.w_size, dsn.gaussian_window)

    print("[5/5] embedding and writing")
    out = export_embeddings(
        dsn, records, args.out, label_spec=spec,
        ident_columns=("campaign_id", "topo_idx", "iter_idx", "seed_run"),
        extra_sidecar=extra, batch_size=args.batch_size,
        allow_constant=spec.topology_axes)

    print("\n  rows written        : %d" % out.n_rows)
    print("  traces used         : %d" % out.n_traces_used)
    print("  traces too short    : %d" % out.n_traces_skipped_short)
    print("  embedding dimension : %d" % out.embedding_dim)
    print("  assertions passed   : %s" % ", ".join(out.assertions_passed))
    print("  parquet             : %s" % out.parquet_path)
    print("  sidecar             : %s" % out.sidecar_path)
    if out.warnings:
        print("\n  warnings:")
        for w in out.warnings:
            print("    - %s" % w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
