"""Block 6.3 -- the bank plan's model: the contract rule, the label columns,
the seed groups (twins) and the donors (specs/SPEC.md, Block 6.3; sec. 2.5).

Pure functions of what Block 6.2 loaded: no file access here. The registry is
read by the caller (sbi_labels.load_registry, the simulator's own source) and
passed in, so that this module never decides where the simulator lives.

Decisions this module implements: D-074 (one bank, O_N declared in one box,
every task's own O_N box inside it), D-070 (same-seed tasks across two label
contracts are different simulations, both kept), D-076 (donor = one network:
one id per (task seed values, topology index), twin topologies merged), and
(Block 10; SBI D-148 (5), D-143) the partial topologies dropped whole:
partial_topologies decides them from the raw counts, drop_topologies removes
them from Top(t) and E(t) for the plan and, from the plan's list, the export.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "sim_reextract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sbi_labels as SL               # noqa: E402
import sim_reextract_plan as PLAN     # noqa: E402

BOX_RTOL = 1e-12                      # float64 roundoff on a JSON-read bound


class PlanRefused(ValueError):
    """The record cannot become one bank as decided; the message names why."""


@dataclass(frozen=True)
class LabelColumns:
    param_names: Tuple[str, ...]      # p names, the order of theta's columns
    coord: Tuple[str, ...]            # p entries, "ln" or "linear" (physical coordinate)
    units: Tuple[str, ...]
    bounds_declared: np.ndarray       # (p, 2) declared box, in each axis's coordinate
    on_column: int                    # O_N's column in theta
    active_indices: Tuple[int, ...]


def _value_key(v):
    return PLAN._norm(v)


def check_contract(tasks: Sequence, cfg) -> Dict:
    """The contract rule of SPEC sec. 2.5. Returns the bank's single contract.

    Every field of sim_reextract_plan's label contract takes one value across
    the tasks, except the O_N row of param_bounds, which may differ provided
    each task's box lies inside cfg.on_declared_box (D-074). simtime must equal
    the window length (D-041). manifest_version is reported, never compared.
    """
    if not tasks:
        raise PlanRefused("no task to plan")
    fields = tuple(PLAN.JOB_ARGS_FIELDS) + tuple(PLAN.MANIFEST_FIELDS)
    names0 = tasks[0].contract.get("param_names")
    if not isinstance(names0, list) or cfg.on_axis not in names0:
        raise PlanRefused("%s: param_names %r does not name %s"
                          % (tasks[0].name, names0, cfg.on_axis))
    on_idx = names0.index(cfg.on_axis)
    single, on_boxes, versions = {}, {}, {}
    for f in fields:
        groups = {}
        for t in tasks:
            v = t.contract.get(f, "absent")
            if f == "param_bounds":
                if not isinstance(v, list) or len(v) != len(names0):
                    raise PlanRefused("%s: param_bounds is not a list of %d rows"
                                      % (t.name, len(names0)))
                box = [float(v[on_idx][0]), float(v[on_idx][1])]
                on_boxes.setdefault(tuple(box), []).append(t.name)
                v = [row for j, row in enumerate(v) if j != on_idx]
            groups.setdefault(_value_key(v), (v, []))[1].append(t.name)
        if len(groups) != 1:
            desc = "; ".join("%s in %d task(s), e.g. %s" % (PLAN._short(val, 60), len(ns), ns[0])
                             for val, ns in groups.values())
            what = "param_bounds outside the %s row" % cfg.on_axis if f == "param_bounds" else f
            raise PlanRefused("the label contract differs in %s: %s" % (what, desc))
        single[f] = next(iter(groups.values()))[0]
    for t in tasks:
        mv = t.contract.get("manifest_version", "absent")
        versions[str(mv)] = versions.get(str(mv), 0) + 1
    if float(single["simtime"]) != float(cfg.t_win_s):
        raise PlanRefused("simtime is %r s, the window is %r s (D-041: one window per "
                          "simulation)" % (single["simtime"], cfg.t_win_s))
    lo_d, hi_d = cfg.on_declared_box
    for (lo, hi), names in on_boxes.items():
        inside = (lo >= lo_d * (1.0 - BOX_RTOL)) and (hi <= hi_d * (1.0 + BOX_RTOL)) and lo < hi
        if not inside:
            raise PlanRefused("%s box [%r, %r] of %d task(s), e.g. %s, is not inside the "
                              "declared [%r, %r] (D-074)" % (cfg.on_axis, lo, hi, len(names),
                                                             names[0], lo_d, hi_d))
    return {"single": single, "on_index": on_idx,
            "on_boxes": [{"box": list(b), "n_tasks": len(n), "tasks": n}
                         for b, n in sorted(on_boxes.items())],
            "manifest_version": versions}


def _excluded_axes(freeze):
    out = {}
    for k, v in (freeze.get("excluded_axes") or {}).items():
        out[str(k)] = str(v.get("reason", v)) if isinstance(v, dict) else str(v)
    return out


def task_label_spec(task, registry, cfg, freeze):
    """The task's own LabelSpec: its manifest's boxes and coordinates (D-004).

    The manifest is authoritative over the simulator's current source
    (registry_from_manifest); a coordinate that differs from the source is
    reported by flipped_axes(), and label_columns() refuses only when the
    tasks disagree with one another. A topology axis bounded by job_args keys
    outside the contract (D-107: stoa_gate_p) takes the task's own box, read
    by bank_plan_load into task.topology_args.
    """
    c = task.contract
    try:
        reg_t, _changed, _flipped = SL.registry_from_manifest(registry, c)
        topology_bounds = SL.topology_bounds_from_job_args(
            getattr(task, "topology_args", None) or {}, list(cfg.topology_axes))
        return SL.build_label_spec(reg_t, c["active_indices"], sweep_group=str(c["sweep_group"]),
                                   conn_prob_bounds=(float(c["conn_prob_lo"]),
                                                     float(c["conn_prob_hi"])),
                                   topology_axes=list(cfg.topology_axes),
                                   excluded_axes=_excluded_axes(freeze),
                                   topology_bounds=topology_bounds)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise PlanRefused("%s: its label spec cannot be built: %s" % (task.name, exc))


def label_columns(tasks, registry, cfg, freeze) -> LabelColumns:
    """The bank's label columns and its declared box (D-074, D-078)."""
    specs = [task_label_spec(t, registry, cfg, freeze) for t in tasks]
    s0 = specs[0]
    for t, s in zip(tasks, specs):
        if list(s.param_names) != list(s0.param_names) or list(s.coord) != list(s0.coord):
            raise PlanRefused("%s: label names or coordinates differ from %s's"
                              % (t.name, tasks[0].name))
    if s0.p != cfg.expected_p:
        raise PlanRefused("the label has p = %d columns, the profile expects %d"
                          % (s0.p, cfg.expected_p))
    if list(s0.topology_axes) != list(cfg.topology_axes):
        raise PlanRefused("the label's topology block %r is not the profile's %r"
                          % (list(s0.topology_axes), list(cfg.topology_axes)))
    col = list(s0.param_names).index(cfg.on_axis)
    if s0.coord[col] != "ln":
        raise PlanRefused("%s is %r in the label; the declared box is ln (D-074)"
                          % (cfg.on_axis, s0.coord[col]))
    bounds = np.array(s0.bounds_theta, dtype=np.float64)
    bounds[col] = [math.log(cfg.on_declared_box[0]), math.log(cfg.on_declared_box[1])]
    for t, s in zip(tasks, specs):
        b = np.array(s.bounds_theta, dtype=np.float64)
        others = [j for j in range(s0.p) if j != col]
        if not np.array_equal(b[others], bounds[others]):
            raise PlanRefused("%s: label bounds outside %s differ from the bank's"
                              % (t.name, cfg.on_axis))
    if not np.all(bounds[:, 1] > bounds[:, 0]):
        raise PlanRefused("a declared label box is degenerate: %r" % (bounds.tolist(),))
    return LabelColumns(param_names=tuple(s0.param_names), coord=tuple(s0.coord),
                        units=tuple(s0.units), bounds_declared=bounds, on_column=col,
                        active_indices=tuple(int(a) for a in s0.active_indices))


def flipped_axes(tasks, registry) -> Dict[str, List[str]]:
    """Per task, the axes whose coordinate its manifest flips against the source."""
    out = {}
    for t in tasks:
        try:
            _reg, _changed, flipped = SL.registry_from_manifest(registry, t.contract)
        except ValueError as exc:
            raise PlanRefused("%s: its manifest cannot re-source the registry: %s"
                              % (t.name, exc))
        if flipped:
            out[t.name] = list(flipped)
    return out


def seed_key(seeds) -> str:
    return json.dumps(seeds, sort_keys=True, separators=(",", ":"))


def seed_groups(tasks) -> List[Dict]:
    """Tasks grouped by their seed values; a group of two contracts is a twin pair."""
    by = {}
    for t in tasks:
        by.setdefault(seed_key(t.seeds), []).append(t)
    out = []
    for key in sorted(by):
        members = by[key]
        sigs = [m.contract_sig for m in members]
        if len(set(sigs)) != len(sigs):
            raise PlanRefused("tasks %r share their seed values AND their label contract: a "
                              "replay the record should have excluded (D-061, D-070)"
                              % [m.name for m in members])
        out.append({"seed_key": key, "tasks": [m.name for m in members],
                    "contract_sigs": sigs, "twin": len(members) > 1})
    return out


def donor_map(tasks) -> Dict:
    """D-076: one donor per (seed values, topology index) over the whole record."""
    keys = sorted({(seed_key(t.seeds), k) for t in tasks for k in t.topo_iters})
    index = {key: n for n, key in enumerate(keys)}
    by_task = {t.name: {k: index[(seed_key(t.seeds), k)] for k in t.topo_iters} for t in tasks}
    return {"n_donors": len(keys), "by_task": by_task}


def shard_layout(tasks, rows_per_shard) -> Dict:
    """Each task's simulations, sorted by (topology, iteration), cut into near-equal
    chunks of at most rows_per_shard; one global shard index per chunk."""
    by_task, nxt = {}, 0
    for t in tasks:
        n = t.n_iters
        n_chunks = max(1, -(-n // int(rows_per_shard)))
        sizes = [n // n_chunks + (1 if j < n % n_chunks else 0) for j in range(n_chunks)]
        chunks, start = [], 0
        for size in sizes:
            chunks.append({"shard": nxt, "start": start, "stop": start + size})
            start += size
            nxt += 1
        by_task[t.name] = chunks
    return {"n_shards": nxt, "by_task": by_task}


def positions_sha256(topo_iters) -> str:
    """Digest of one task's (topology, iteration) list, in the shard layout's order;
    the export compares it before cutting the plan's chunks (Block 7.1)."""
    pos = [[int(k), [int(i) for i in topo_iters[k]]] for k in sorted(topo_iters)]
    return hashlib.sha256(json.dumps(pos, separators=(",", ":")).encode("ascii")).hexdigest()


def sorted_positions(task) -> List[Tuple[int, int]]:
    """The (topology, iteration) pairs of one task in the shard layout's order."""
    return [(k, i) for k in sorted(task.topo_iters) for i in task.topo_iters[k]]


def partial_topologies(task, finished: Dict[int, int], dispatched: int) -> List[Dict]:
    """[Block 10] The partial topologies of one task: [{"topo": k, "finished":
    f(t, k), "expected": K(t)}, ...] for every k with f(t, k) < K(t), sorted by k.

    finished: {k: f(t, k)} for every raw folder topo_<k> of the task
    (bank_plan_load.raw_finished_counts); dispatched: K(t)
    (bank_plan_load.dispatch_size). PlanRefused, naming the task and the
    topologies, when f(t, k) > K(t) (the task holds more runs than it
    dispatched), or when the raw task's topology folders are not exactly the
    detections' Top(t) and E(t): f(t, k) is read per raw folder and the drop
    acts on the detections, so each needs the other.
    """
    raw = set(int(k) for k in finished)
    det = set(task.topo_iters) | set(task.empty_topos)
    if raw != det:
        raise PlanRefused("%s: the raw task's topology folders and its detections' differ "
                          "(raw only %r, detections only %r), so the finished runs of each "
                          "topology cannot be matched to its detections "
                          "(drop_partial_topologies)"
                          % (task.name, sorted(raw - det), sorted(det - raw)))
    over = [(int(k), int(f)) for k, f in sorted(finished.items()) if f > dispatched]
    if over:
        raise PlanRefused("%s: more runs than the task dispatched per topology, K = %d "
                          "(_resolved_n_workers x n_params_per_worker): %s "
                          "(iter_<i>.json sidecars and _failures.jsonl lines)"
                          % (task.name, dispatched,
                             ", ".join("topology %d holds %d" % kf for kf in over)))
    return [{"topo": int(k), "finished": int(f), "expected": int(dispatched)}
            for k, f in sorted(finished.items()) if f < dispatched]


def drop_topologies(task, topos: Iterable[int]):
    """[Block 10] The task with the topologies `topos` removed from Top(t) and
    E(t), and n_iters and n_topos counted again on what remains (a new
    TaskInputs; the input is not changed). PlanRefused, naming the task and the
    topologies, when one of them is in neither (the export's "a listed topology
    absent from its tree")."""
    drop = sorted(set(int(k) for k in topos))
    absent = [k for k in drop if k not in task.topo_iters and k not in task.empty_topos]
    if absent:
        raise PlanRefused("%s: partial topology %s is not in its detections tree (%s)"
                          % (task.name, ", ".join(str(k) for k in absent), task.out_dir))
    gone = set(drop)
    topo = {k: list(v) for k, v in task.topo_iters.items() if k not in gone}
    empty = tuple(k for k in task.empty_topos if k not in gone)
    return dataclasses.replace(task, topo_iters=topo, empty_topos=empty,
                               n_iters=sum(len(v) for v in topo.values()),
                               n_topos=len(topo) + len(empty))
