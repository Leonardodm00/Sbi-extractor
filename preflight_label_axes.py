#!/usr/bin/env python3
"""Decide, ONCE and for the whole export, which topology-level axes enter theta.

Why this exists
---------------
The sweep records what it DREW, not what the simulator READ. Two different
defects hide behind that, and only one of them is visible to a variance scan:

  (a) an axis that never varies      -> a delta prior; the flow's conditional
                                        density degenerates along it. DETECTABLE
                                        from the data, and handled automatically
                                        here.
  (b) an axis that varies but that
      nothing downstream consumes    -> p(x | theta) does not depend on it, so
                                        the true posterior equals the prior on
                                        that axis. NOT detectable from the data:
                                        the column looks perfectly healthy. It
                                        must be declared with --exclude.

The known case of (b) in this project is conn_prob. HPC_main_sweep.py draws it
per topology from U(conn_prob_lo, conn_prob_hi) unconditionally, but under
--conn_rule weibull the topology is built from the Weibull kernel and the edge
list is passed to Neuronal_Network as an explicit list, which takes the
S.connect(i=Source, j=Target) branch; the flat-probability branch
S.connect(p=params_Syn['conn_prob'], ...) never runs. See the handoff
HANDOFF_param_recording.md for the full trace.

Why it must be frozen
---------------------
Deciding the axis set per shard would be a silent disaster: each export task
covers a subset of topologies, so different shards would drop different columns
and emit theta matrices of different width and column order -- unpoolable, and
fatal for a single NPE contract. This script looks at ALL campaigns at once,
writes one label_axes.json, and every shard is then built from that one file.

Usage
-----
    python3 preflight_label_axes.py \
        --sim_main /path/to/Phenomenological/Main \
        --campaigns 'campaign_cadex_rho1300v*' \
        --exclude 'conn_prob=drawn per topology but never read under
                   conn_rule=weibull (explicit edge list bypasses the flat
                   S.connect(p=...) branch)' \
        --out artifacts/label_axes.json

Add --require-conn-rule weibull to refuse to write the file if any campaign
used a different connectivity rule -- under flat, conn_prob IS causally live
and pooling the two would put two generative structures in one theta column.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# Axes this project knows how to place in the topology block. An axis outside
# this set needs its prior interval plumbed into sbi_labels.build_label_spec
# before it can enter theta, so discovering one is a hard stop, not a warning.
KNOWN_TOPOLOGY_AXES = ("conn_prob", "p0_conn", "d0_conn", "beta_conn")


def parse_exclusions(items):
    """['name=reason', ...] -> {name: reason}. A reason is mandatory."""
    out = {}
    for it in items or ():
        if "=" not in it:
            raise SystemExit(
                "--exclude needs NAME=REASON (got %r). An exclusion without a "
                "written reason is indistinguishable from an oversight six "
                "months from now." % (it,))
        name, reason = it.split("=", 1)
        name, reason = name.strip(), " ".join(reason.split())
        if not reason:
            raise SystemExit("--exclude %s has an empty reason" % (name,))
        out[name] = reason
    return out


def scan(sim_main, campaign_glob):
    """One iter_*.npz per topology -> per-axis stats + observed conn_rule set.

    One file per topo_* dir is enough BY CONSTRUCTION: these axes are drawn in
    the outer topology loop, so they are constant within a topology. Reading
    every iteration would be ~10^5 times more I/O for identical numbers.
    """
    stats = {a: {"values": set(), "n_seen": 0} for a in KNOWN_TOPOLOGY_AXES}
    unknown_scalars = set()
    conn_rules = {}
    n_topo = 0

    pattern = os.path.join(sim_main, campaign_glob, "sweep_*", "topo_*")
    for td in sorted(glob.glob(pattern)):
        cand = sorted(glob.glob(os.path.join(td, "iter_*.npz")))
        if not cand:
            continue
        camp = os.path.relpath(td, sim_main).split(os.sep)[0]
        try:
            with np.load(cand[0], allow_pickle=False) as z:
                keys = set(z.files)
                for a in KNOWN_TOPOLOGY_AXES:
                    if a in keys:
                        v = float(z[a])
                        stats[a]["values"].add(round(v, 12))
                        stats[a]["n_seen"] += 1
                # Surface any OTHER scalar that looks like a swept topology
                # parameter, so a newly added axis cannot be silently ignored.
                for k in keys - set(KNOWN_TOPOLOGY_AXES):
                    arr = z[k]
                    if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
                        unknown_scalars.add(k)
        except Exception as exc:                       # noqa: BLE001
            print("  WARNING: unreadable %s (%s)" % (cand[0], exc),
                  file=sys.stderr)
            continue
        n_topo += 1

    for jp in sorted(glob.glob(os.path.join(sim_main, campaign_glob,
                                            "sweep_*", "job_args.json"))):
        camp = os.path.relpath(jp, sim_main).split(os.sep)[0]
        try:
            with open(jp) as fh:
                ja = json.load(fh)
        except Exception:                              # noqa: BLE001
            continue
        conn_rules.setdefault(camp, set()).add(str(ja.get("conn_rule")))

    return stats, unknown_scalars, conn_rules, n_topo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim_main", required=True,
                    help="dir containing the campaign_* trees")
    ap.add_argument("--campaigns", default="campaign_*",
                    help="glob for campaign dirs (default: campaign_*)")
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="NAME=REASON",
                    help="axis that VARIES but is causally inert; repeatable")
    ap.add_argument("--require-conn-rule", default=None,
                    help="refuse to write unless every campaign used this rule")
    ap.add_argument("--out", default=None,
                    help="path for label_axes.json (omit to only report)")
    args = ap.parse_args()

    excluded_req = parse_exclusions(args.exclude)

    print("scanning %s / %s ..." % (args.sim_main, args.campaigns))
    stats, unknown, conn_rules, n_topo = scan(args.sim_main, args.campaigns)
    if n_topo == 0:
        raise SystemExit("no topo_*/iter_*.npz found -- check --sim_main "
                         "and --campaigns")
    print("topologies sampled : %d" % n_topo)

    # --- connectivity rule uniformity -------------------------------------
    all_rules = sorted({r for s in conn_rules.values() for r in s})
    print("conn_rule observed : %r" % (all_rules,))
    if args.require_conn_rule is not None:
        bad = {c: sorted(s) for c, s in conn_rules.items()
               if s != {args.require_conn_rule}}
        if bad:
            raise SystemExit(
                "REFUSING to write: --require-conn-rule %r but these campaigns "
                "differ: %r. Under a different rule the same axis name can be "
                "causally live in one campaign and inert in another; one theta "
                "column cannot mean both." % (args.require_conn_rule, bad))

    # --- per-axis verdict --------------------------------------------------
    print("")
    print("%-12s %10s  %-12s  %s" % ("axis", "n_distinct", "verdict", "range"))
    varying, constant = [], {}
    for a in KNOWN_TOPOLOGY_AXES:
        vals = stats[a]["values"]
        if not vals:
            print("%-12s %10s  %-12s" % (a, "-", "ABSENT"))
            continue
        n = len(vals)
        rng = "[%.6g, %.6g]" % (min(vals), max(vals))
        if a in excluded_req:
            verdict = "EXCLUDED"
        elif n <= 1:
            verdict = "CONSTANT"
            constant[a] = ("constant across the whole export (single value "
                           "%.6g): a delta prior carries no information and "
                           "makes the flow degenerate along this axis"
                           % (min(vals),))
        else:
            verdict = "in theta"
            varying.append(a)
        print("%-12s %10d  %-12s  %s" % (a, n, verdict, rng))

    if unknown:
        print("")
        print("NOTE: other numeric scalars present in iter_*.npz: %r"
              % (sorted(unknown),))
        print("      If any of these is a swept topology parameter it needs "
              "bounds in sbi_labels.build_label_spec before it can join theta.")

    # An excluded axis that turns out to be constant anyway is fine; an
    # excluded axis that is ABSENT is a typo and should be caught loudly.
    ghosts = [a for a in excluded_req if not stats.get(a, {}).get("values")]
    if ghosts:
        raise SystemExit("--exclude names axis/axes never seen in the data: %r"
                         % (ghosts,))

    excluded_all = dict(constant)
    excluded_all.update(excluded_req)

    if not varying:
        raise SystemExit("no topology axis left in theta -- refusing to write")

    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sim_main": os.path.abspath(args.sim_main),
        "campaign_glob": args.campaigns,
        "topologies_sampled": n_topo,
        "conn_rule_observed": all_rules,
        "topology_axes": varying,
        "excluded_axes": {
            k: {"reason": v,
                "n_distinct": len(stats[k]["values"]),
                "min": min(stats[k]["values"]) if stats[k]["values"] else None,
                "max": max(stats[k]["values"]) if stats[k]["values"] else None}
            for k, v in sorted(excluded_all.items())},
        "axis_stats": {
            a: {"n_distinct": len(stats[a]["values"]),
                "min": min(stats[a]["values"]) if stats[a]["values"] else None,
                "max": max(stats[a]["values"]) if stats[a]["values"] else None}
            for a in KNOWN_TOPOLOGY_AXES if stats[a]["values"]},
        # Rows within a topology share theta_topo exactly, so the INDEPENDENT
        # sample size along these axes is the topology count, not the row
        # count. Recorded because every power calculation downstream (the MMD
        # gate's n_sim floor included) counts rows by default.
        "n_independent_topology_draws": n_topo,
    }

    print("")
    print("theta topology block: %r  (p_topology = %d)" % (varying, len(varying)))
    for k, v in sorted(excluded_all.items()):
        print("excluded: %s -- %s" % (k, v))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        blob = json.dumps(doc, indent=2, sort_keys=True) + "\n"
        with open(args.out, "w") as fh:
            fh.write(blob)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        with open(args.out + ".sha256", "w") as fh:
            fh.write("%s  %s\n" % (digest, os.path.basename(args.out)))
        print("")
        print("wrote %s" % args.out)
        print("sha256 %s" % digest[:16])
    else:
        print("")
        print("(no --out given; nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
