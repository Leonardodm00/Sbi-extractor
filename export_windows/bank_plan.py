#!/usr/bin/env python3
"""Block 6.5 -- B5.1, the Giulia window bank's plan step (specs/SPEC.md, Block 6).

    python3 export_windows/bank_plan.py --profile giulia [--out PATH]

Read-only over the data: reads the G2 record, the G3 label freeze, the frozen
G1 cohort manifest, every task's job_args.json / manifest.json, the list of
detection files, and ONE simulation per topology (its detections and its raw
iteration file). Writes one file, the plan (default: the profile's plan_out,
export_windows/giulia/bank_plan.json, gitignored -- D-079) with its .sha256,
and only when every check passes. The bank's next steps (bank_task.py, the
gate) run from that file.

Printed last line: `BANK PLAN OK` (exit 0) or `REFUSED: <reason>` (exit 1).

What it checks and records (D-041, D-043, D-044, D-070, D-074..D-076, D-078..D-080,
D-107, D-108):
  the inputs against their digests; one label contract but O_N's box, every
  task's O_N box inside the declared box (Giulia: [0.01, 3.0]); the label
  columns (Giulia: 23) and the declared box; each campaign's batch number
  (N of a _v<N> suffix, else 0; two campaigns with one number refuse); the
  seed groups, the twin pairs and the donors; the shard layout; for one
  simulation per topology, the window, its rate, the floor and the label in
  the unit box (a topology value read from the raw files, D-107's
  stoa_gate_p, checked equal at the topology's first, last and sampled
  iterations); the twin check (equal conn_prob, equal seed_run); the
  estimates of the floor's kept share, the bytes per row and the bank's size;
  the free space under the bank's folder.

Partial topologies (Block 10; SBI D-148 (5), D-143), with the profile's
drop_partial_topologies true: after the record checks on the full tree, every
topology whose raw folder holds fewer finished runs (iter_<i>.json sidecars and
_failures.jsonl lines) than its task dispatched (job_args.json's
_resolved_n_workers x n_params_per_worker) is removed from the task before the
donors, the seed groups, the sample, the estimates, the layout and the counts:
it gives no row and no donor. The plan lists them per task (partial_topos,
n_dropped_partial). With false nothing of this is read, and the plan file is
the one written before Block 10.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_plan_config as CFG          # noqa: E402
import bank_plan_load as LOAD           # noqa: E402
import bank_plan_model as MODEL         # noqa: E402
import bank_plan_sample as SAMPLE       # noqa: E402
import bank_row as ROW                  # noqa: E402
import sbi_labels as SL                 # noqa: E402  (bank_plan_load put the repo on sys.path)

PLAN_VERSION = 1
CAMPAIGN_RE = re.compile(r"_v(\d+)$")


def build_parser():
    p = argparse.ArgumentParser(description="B5.1: the Giulia window bank's plan step (read-only).")
    p.add_argument("--profile", required=True, help="profile name under export_windows/profiles/, or a .json path")
    p.add_argument("--out", default=None, help="plan file (default: the profile's plan_out)")
    return p


def git_commit():
    try:
        out = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_atomic(path, blob):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(blob)
    os.replace(tmp, path)
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()
    with open(path + ".sha256.tmp", "w", encoding="ascii") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(path)))
    os.replace(path + ".sha256.tmp", path + ".sha256")
    return digest


def campaign_version(campaign):
    """The batch number of a campaign (SPEC sec. 2.6; D-108): N for a name ending
    in _v<N>, 0 for any other name."""
    m = CAMPAIGN_RE.search(campaign)
    return int(m.group(1)) if m else 0


def campaign_batches(campaigns):
    """{campaign: batch} for the distinct names given (in their first order).

    Refuses two different names with one batch number (D-108): the batch column
    would no longer tell the campaigns apart.
    """
    out = {}
    for c in campaigns:
        out.setdefault(str(c), campaign_version(str(c)))
    by_batch = {}
    for c, b in out.items():
        by_batch.setdefault(b, []).append(c)
    clash = sorted((b, names) for b, names in by_batch.items() if len(names) > 1)
    if clash:
        raise MODEL.PlanRefused(
            "two or more campaigns share one batch number, so the bank's batch column "
            "could not tell them apart (D-108; N of a _v<N> suffix, else 0): %s"
            % "; ".join("batch %d: %s" % (b, ", ".join(names)) for b, names in clash))
    return out


def _refuse_tasks(messages, n_tasks):
    """One refused task: its own message; two or more: one message whose first line
    counts them, then one line each (at most LOAD.MAX_REFUSED_SHOWN), as Block 6.2's."""
    if len(messages) == 1:
        return MODEL.PlanRefused(messages[0])
    shown = list(messages[:LOAD.MAX_REFUSED_SHOWN])
    more = len(messages) - len(shown)
    return MODEL.PlanRefused("%d of %d tasks refused\n%s%s"
                             % (len(messages), n_tasks, "\n".join(shown),
                                ("\n... and %d more" % more) if more else ""))


def drop_partial_topologies(tasks):
    """Block 10: (the tasks without their partial topologies, {task name:
    {"partial_topos", "n_dropped_partial"}}).

    For each task, K(t) from its job_args.json and f(t, k) from its raw
    topology folders; every topology with f(t, k) < K(t) is removed from Top(t)
    and E(t) (bank_plan_model.drop_topologies). Every task is checked before
    one error names every refused task.
    """
    kept, partial, refused = [], {}, []
    for t in tasks:
        try:
            dispatched = LOAD.dispatch_size(t.campaign_dir)
            finished = LOAD.raw_finished_counts(t.campaign_dir)
        except LOAD.LoadError as exc:
            refused.append("%s: %s" % (t.name, exc))
            continue
        try:
            rows = MODEL.partial_topologies(t, finished, dispatched)
            reduced = MODEL.drop_topologies(t, [p["topo"] for p in rows])
        except MODEL.PlanRefused as exc:
            refused.append(str(exc))
            continue
        kept.append(reduced)
        partial[t.name] = {"partial_topos": rows,
                           "n_dropped_partial": int(t.n_iters - reduced.n_iters)}
    if refused:
        raise _refuse_tasks(refused, len(tasks))
    return tuple(kept), partial


def run(args):
    t0 = time.time()
    cfg = CFG.load_profile(args.profile)
    out_path = os.path.abspath(args.out) if args.out else cfg.plan_out
    print("[plan] profile        : %s" % args.profile)
    inp = LOAD.load_inputs(cfg)
    print("[plan] record         : %s  sha256 %s  (%d tasks)"
          % (cfg.record_path, inp.record_sha256[:16], len(inp.tasks)))
    print("[plan] label freeze   : %s  sha256 %s  topology %r"
          % (cfg.freeze_path, inp.freeze_sha256[:16], inp.freeze.get("topology_axes")))
    print("[plan] cohort manifest: %s  sha256 %s  (w_size %r, gaussian_window %r, "
          "electrodes_per_subset %d)" % (cfg.cohort_manifest_path, inp.manifest_sha256[:16],
                                         inp.dt, inp.sigma_sm, inp.electrodes_per_subset))
    if inp.electrodes_per_subset != cfg.n_e:
        raise MODEL.PlanRefused("the cohort manifest's electrodes_per_subset is %d, the "
                                "profile's n_e %d (D-043)" % (inp.electrodes_per_subset, cfg.n_e))
    W = int(round(cfg.t_win_s / inp.dt))
    if abs(W * inp.dt - cfg.t_win_s) > 1e-9 or int(cfg.t_win_s / inp.dt) != W:
        raise MODEL.PlanRefused("t_win_s %r is not a whole number of %r s bins"
                                % (cfg.t_win_s, inp.dt))

    batch_of = campaign_batches(rec["campaign"] for rec in inp.record["tasks"])
    batches = [batch_of[rec["campaign"]] for rec in inp.record["tasks"]]
    print("[plan] batches        : %s" % ", ".join(
        "%s %d%s" % (c, b, "" if CAMPAIGN_RE.search(c) else " (no _v<N>: 0, D-108)")
        for c, b in batch_of.items()))
    if sum(t.n_iters for t in inp.tasks) == 0:
        raise MODEL.PlanRefused("the record's tasks hold no detection file")
    partial, partial_totals = None, None
    if cfg.drop_partial_topologies:     # Block 10: after the record checks, before all else
        names = [t.name for t in inp.tasks]
        kept_tasks, partial = drop_partial_topologies(inp.tasks)
        inp = dataclasses.replace(inp, tasks=kept_tasks)
        listed = [(name, p) for name in names for p in partial[name]["partial_topos"]]
        partial_totals = {
            "n_partial_topos": len(listed),
            "n_partial_tasks": sum(1 for name in names if partial[name]["partial_topos"]),
            "n_dropped_partial": sum(partial[name]["n_dropped_partial"] for name in names)}
        print("[plan] partial topologies (drop_partial_topologies): %d dropped in %d task(s), "
              "%d simulations; bank_plan.json tasks[].partial_topos"
              % (partial_totals["n_partial_topos"], partial_totals["n_partial_tasks"],
                 partial_totals["n_dropped_partial"]))
        for name, p in listed[:10]:
            print("[plan]   %s topo %d: %d of %d" % (name, p["topo"], p["finished"],
                                                     p["expected"]))
        if sum(t.n_iters for t in inp.tasks) == 0:
            raise MODEL.PlanRefused("no detection file is left once the partial topologies are "
                                    "dropped (drop_partial_topologies: %d topologies, %d "
                                    "simulations)" % (partial_totals["n_partial_topos"],
                                                      partial_totals["n_dropped_partial"]))
    else:
        print("[plan] partial topologies: not checked (drop_partial_topologies false)")
    contract = MODEL.check_contract(inp.tasks, cfg)
    print("[plan] contract       : one value per field but %s's box; simtime %r s; "
          "manifest_version %r" % (cfg.on_axis, contract["single"]["simtime"],
                                   contract["manifest_version"]))
    for b in contract["on_boxes"]:
        print("[plan]   %s box %r : %d task(s)" % (cfg.on_axis, b["box"], b["n_tasks"]))

    try:
        registry = SL.load_registry(cfg.sim_dir)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise MODEL.PlanRefused("the simulator registry at %s: %s" % (cfg.sim_dir, exc))
    label = MODEL.label_columns(inp.tasks, registry, cfg, inp.freeze)
    flipped = MODEL.flipped_axes(inp.tasks, registry)
    n_ln = sum(1 for c in label.coord if c == "ln")
    print("[plan] label          : p = %d (%d ln, %d linear); %s in column %d, declared "
          "[%.6f, %.6f] (ln)" % (len(label.param_names), n_ln, len(label.coord) - n_ln,
                                 cfg.on_axis, label.on_column,
                                 label.bounds_declared[label.on_column, 0],
                                 label.bounds_declared[label.on_column, 1]))
    if flipped:
        print("[plan]   NOTE coordinates flipped against the simulator's source: %r" % flipped)

    groups = MODEL.seed_groups(inp.tasks)
    donors = MODEL.donor_map(inp.tasks)
    layout = MODEL.shard_layout(inp.tasks, cfg.rows_per_shard)
    n_twin = sum(1 for g in groups if g["twin"])
    n_live = sum(len(t.topo_iters) for t in inp.tasks)
    empties = [(t.name, k) for t in inp.tasks for k in t.empty_topos]
    n_rows = sum(t.n_iters for t in inp.tasks)
    print("[plan] seeds          : %d group(s), %d twin pair(s)" % (len(groups), n_twin))
    print("[plan] donors         : %d (from %d topologies with simulations; D-076)"
          % (donors["n_donors"], n_live))
    if empties:
        print("[plan] topologies     : %d without simulations (the folder is empty in the raw "
              "task and in its detections: no row, no donor)" % len(empties))
        for name, k in empties[:10]:
            print("[plan]   no simulation: %s topology %d" % (name, k))
        if len(empties) > 10:
            print("[plan]   ... and %d more (bank_plan.json: tasks[].empty_topos)"
                  % (len(empties) - 10))
    print("[plan] shards         : %d of at most %d simulations" % (layout["n_shards"],
                                                                     cfg.rows_per_shard))

    specs = {t.name: MODEL.task_label_spec(t, registry, cfg, inp.freeze) for t in inp.tasks}
    draws = SAMPLE.draw_iterations(inp.tasks, groups, cfg)
    samples, windows = [], []
    for n, t in enumerate(inp.tasks):
        for k in sorted(t.topo_iters):
            d = draws[(t.name, k)]
            s = SAMPLE.sample_topology(t, k, d["i"], inp, cfg, specs[t.name],
                                       label.bounds_declared)
            s["comparable"] = d["comparable"]
            if s["kept"] and len(windows) < cfg.bytes_sample_rows:
                windows.append(s["x"])
            del s["x"]
            samples.append(s)
        if (n + 1) % 10 == 0 or n + 1 == len(inp.tasks):
            print("[plan]   sampled %d/%d task(s), %d topologies, %.0f s"
                  % (n + 1, len(inp.tasks), len(samples), time.time() - t0))
    negative = [s for s in samples if s["seed_run"] < 0]
    if negative:
        s = negative[0]
        raise MODEL.PlanRefused("%d sampled simulation(s) carry seed_run < 0 (the sweep's "
                                "paired mode), e.g. %s topology %d iteration %d: the bank's "
                                "realisation_id is the seed_run (D-072), and the export "
                                "refuses them" % (len(negative), s["task"], s["k"], s["i"]))
    donors_of = {(t.name, k): donors["by_task"][t.name][k] for t in inp.tasks
                 for k in t.topo_iters}
    share = SAMPLE.kept_share(samples, donors_of, cfg)
    bpr = SAMPLE.bytes_per_row(windows) if windows else None
    twins = SAMPLE.twin_check(groups, samples)
    space = SAMPLE.free_space(cfg.bank_root)
    raw_bytes = n_rows * W * 4
    kept_rows = share["p_hat"] * n_rows
    est = {"n_rows": n_rows, "raw_bytes": raw_bytes, "kept_rows": kept_rows,
           "kept_rows_ci95": [c * n_rows for c in share["ci95"]],
           "bytes_per_kept_row": bpr,
           "kept_bytes": (kept_rows * bpr) if bpr else None}

    tasks_out = []
    for n, t in enumerate(inp.tasks):
        rec = inp.record["tasks"][n]
        box = t.contract["param_bounds"][contract["on_index"]]
        entry = {
            "index": n, "name": t.name, "campaign": rec["campaign"], "sweep": rec["sweep"],
            "campaign_dir": t.campaign_dir, "out_dir": t.out_dir,
            "contract_sig": t.contract_sig, "seeds": t.seeds, "on_box": [float(box[0]), float(box[1])],
            "batch": batches[n], "n_topos": t.n_topos, "n_iters": t.n_iters,
            "topo_counts": {str(k): len(v) for k, v in sorted(t.topo_iters.items())},
            "positions_sha256": MODEL.positions_sha256(t.topo_iters),
            "empty_topos": [int(k) for k in t.empty_topos],
            "donors": {str(k): v for k, v in sorted(donors["by_task"][t.name].items())},
            "shards": layout["by_task"][t.name]}
        if partial is not None:         # Block 10: partial_topos, n_dropped_partial
            entry.update(partial[t.name])
        tasks_out.append(entry)
    plan = {
        "plan": "giulia_window_bank", "plan_version": PLAN_VERSION,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_commit(), "profile": dataclasses.asdict(cfg),
        "inputs": {"record": cfg.record_path, "record_sha256": inp.record_sha256,
                   "freeze": cfg.freeze_path, "freeze_sha256": inp.freeze_sha256,
                   "cohort_manifest": cfg.cohort_manifest_path,
                   "cohort_manifest_sha256": inp.manifest_sha256, "sim_dir": cfg.sim_dir},
        "window": {"T": cfg.t_win_s, "dt": inp.dt, "sigma_sm": inp.sigma_sm, "W": W,
                   "fs": 1.0 / inp.dt, "n_e": cfg.n_e, "floor_hz": cfg.floor_hz,
                   "floor_rtol": cfg.floor_rtol},
        "contract": contract,
        "label": {"param_names": list(label.param_names), "coord": list(label.coord),
                  "units": list(label.units), "bounds_declared": label.bounds_declared.tolist(),
                  "on_axis": cfg.on_axis, "on_column": label.on_column,
                  "active_indices": list(label.active_indices),
                  "topology_axes": list(cfg.topology_axes),
                  "excluded_axes": MODEL._excluded_axes(inp.freeze), "flipped": flipped},
        "tasks": tasks_out, "seed_groups": groups, "n_donors": donors["n_donors"],
        "n_shards": layout["n_shards"], "n_rows": n_rows,
        "sample": {"kept_share": share, "bytes_per_row_from": len(windows),
                   "snapped_total": int(sum(s["snapped"] for s in samples)),
                   "twin_check": twins,
                   "rows": [[s["task"], s["k"], s["i"], round(s["rate_hz"], 6), s["kept"],
                             s["comparable"]] for s in samples]},
        "estimates": est, "free_space": space}
    if partial_totals is not None:      # Block 10; absent with false (the plan of before)
        plan["drop_partial_topologies"] = True
        plan.update(partial_totals)
    blob = json.dumps(plan, indent=1, sort_keys=True) + "\n"
    blob.encode("ascii")
    digest = write_atomic(out_path, blob)

    gb = 1e9
    print("[plan] floor (D-075)  : kept share %.4f (95%% CI %.4f-%.4f; %d topologies, %d donors; "
          "unweighted %.4f)" % (share["p_hat"], share["ci95"][0], share["ci95"][1],
                                share["n_topologies"], share["n_donors"], share["unweighted"]))
    print("[plan]   rate quantiles [Hz]: %s" % ", ".join(
        "%s %.3g" % (k, v) for k, v in share["rate_hz_quantiles"].items()))
    print("[plan] size           : %d rows, %.1f GB raw; after the floor ~%.0f rows%s"
          % (n_rows, raw_bytes / gb, kept_rows,
             (", ~%.1f GB compressed (%.0f B/row from %d windows)"
              % (est["kept_bytes"] / gb, bpr, len(windows))) if bpr else ""))
    print("[plan] free space     : %.1f GB under %s" % (space["free_bytes"] / gb,
                                                         space["path_checked"]))
    if space["lfs_quota"]:
        for line in space["lfs_quota"].splitlines()[:6]:
            print("[plan]   lfs: %s" % line)
    for tc in twins:
        print("[plan] twin %s / %s: %d shared topologies, conn_prob equal on %d; seed_run "
              "equal on %d of %d comparable" % (tc["tasks"][0], tc["tasks"][1],
                                                tc["shared_topologies"], tc["conn_prob_equal"],
                                                tc["seed_run_equal"], tc["seed_run_comparable"]))
    print("[plan] wrote %s  sha256 %s  (%.0f s)" % (out_path, digest[:16], time.time() - t0))
    print("BANK PLAN OK")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (CFG.ConfigError, LOAD.LoadError, MODEL.PlanRefused, ROW.RowError) as exc:
        head, *rest = str(exc).split("\n")
        for line in rest:                  # several refused tasks: one line each, the count last
            print("[plan] refused  %s" % line)
        print("REFUSED: %s%s" % (head, " (each named above)" if rest else ""))
        return 1


if __name__ == "__main__":
    sys.exit(main())
