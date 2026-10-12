"""Block 7.1 -- B5.2 and B5.3: one task's rows and shards for the Giulia
simulated bank (specs/SPEC.md, Block 7.1; secs. 2.5, 2.6).

export_task(plan, task_index) builds every shard the plan assigns to one task:
for each simulation of the chunk, in the plan's (topology, iteration) order,
the row of bank_row.build_row (window, rate, floor, label in the unit box);
the kept rows go into shard_<NNNN>.npz with its JSON sidecar through the
Simulation-Based-Inference repository's latent_bank.write_shard (the Stage 1
contract), and the task's completion marker tasks/task_<NNNN>.json records
every count and every file's sha256. Only the floor drops a row (D-075); any
other defect raises and fails the task (nothing is skipped silently).

One bank_root holds one plan's bank: a task refuses to write into a folder
whose markers or BANK_RECORD.json belong to another plan, and refuses when a
shard file of a chunk that keeps no row is left from an earlier run (it would
enter every glob of the bank).

Row arrays (the closed set of latent_bank.ROW_ARRAYS):
  x               (n, W) float32  the window, spikes per bin per electrode
  theta           (n, p) float32  the label in the unit box of bounds_theta
  cls             (n,)   int64    -1: no class on the simulated arm
  nu              (n, 5) float32  zeros: no nuisance model on simulations
  realisation_id  (n,)   uint64   the simulation's seed_run (its noise key, D-072)
  donor           (n,)   int64    bank-global network id (D-076)
  well            (n,)   int64    the task's index in the record
  batch           (n,)   int64    the campaign version (1, 2, 5)
  subregion       (n,)   int64    0: one virtual electrode (D-043)
  window_idx      (n,)   int64    the simulation's iteration index in its topology
  contaminated    (n,)   bool     False
Every id is bank-global, so concat_shards needs no offsets (USAGE 3.2, F-at).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_plan_load as LOAD           # noqa: E402  (puts the repo root on sys.path)
import bank_plan_model as MODEL         # noqa: E402
import bank_row as ROW                  # noqa: E402
import dsn_tree                         # noqa: E402
import sbi_labels as SL                 # noqa: E402

SCALE_CONVENTION = "per_electrode_mean"
EXPORT_VERSION = 1


class ExportError(RuntimeError):
    """A task that cannot be exported as planned; the message names why."""


def stage1_modules():
    """latent_bank and latent_nuisance from the SBI repository's joint stack."""
    d = os.path.join(dsn_tree.sbi_hpc_dir(), "joint", "stage1")
    if not os.path.isfile(os.path.join(d, "latent_bank.py")):
        raise ExportError("no latent_bank.py under %s (SBI_HPC_DIR)" % d)
    if d not in sys.path:
        sys.path.insert(0, d)
    import latent_bank            # noqa: E402
    import latent_nuisance        # noqa: E402
    return latent_bank, latent_nuisance


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_plan(path) -> Dict:
    """The plan, after its bytes match its .sha256."""
    with open(path, "rb") as fh:
        blob = fh.read()
    side = path + ".sha256"
    if not os.path.isfile(side):
        raise ExportError("the plan %s has no .sha256" % path)
    with open(side, "r", encoding="ascii") as fh:
        want = fh.read().split()[0]
    got = hashlib.sha256(blob).hexdigest()
    if want != got:
        raise ExportError("the plan %s does not match its .sha256" % path)
    plan = json.loads(blob.decode("ascii"))
    plan["_sha256"] = got
    plan["_path"] = os.path.abspath(path)
    return plan


def contract_kwargs(plan, nu_components) -> Dict:
    """The sidecar's contract fields: one value for every shard of the bank."""
    lab, win, inp = plan["label"], plan["window"], plan["inputs"]
    latent_spec = {
        "kind": "campaign", "bank": plan["plan"],
        "param_names": lab["param_names"], "coord_physical": lab["coord"], "units": lab["units"],
        "theta_map": "theta = (theta_physical - lo) / (hi - lo), lo and hi the rows of bounds_theta",
        "prior": "uniform on the unit box (D-078)",
        "on_axis": lab["on_axis"], "on_declared_box": plan["profile"]["on_declared_box"],
        "topology_axes": lab["topology_axes"], "n_windows_per_trace": 1,
        "T_win": win["T"], "fs": win["fs"], "n_e": win["n_e"],
        "floor_hz": win["floor_hz"], "floor_rtol": win["floor_rtol"],
        "label_freeze_sha256": inp["freeze_sha256"],
        "cohort_manifest_sha256": inp["cohort_manifest_sha256"]}
    return {"param_names": lab["param_names"], "bounds_theta": lab["bounds_declared"],
            "coord": "unit_box", "fs": win["fs"], "w_size": win["dt"], "T_win": win["T"],
            "W": win["W"], "scale_convention": SCALE_CONVENTION, "latent_spec": latent_spec,
            "nuisance_spec": {"kind": "none", "components": list(nu_components)},
            "realisation_spec": {"kind": "simulator_seed_run",
                                 "realisation_id": "the simulation's seed_run (D-072)"},
            "gap_spec": {"kind": "none"},
            "generator_sha256": inp["record_sha256"], "theta_withheld": False, "arm": "S"}


def _check_inputs(plan):
    for key, path_key in (("record_sha256", "record"), ("freeze_sha256", "freeze"),
                          ("cohort_manifest_sha256", "cohort_manifest")):
        path = plan["inputs"][path_key]
        if not os.path.isfile(path) or sha256_file(path) != plan["inputs"][key]:
            raise ExportError("%s %s no longer matches the plan's digest" % (path_key, path))


def task_inputs(plan, index):
    """The task re-read and re-checked against the record (Block 6.2's checks)."""
    t = plan["tasks"][index]
    rec = {k: t[k] for k in ("campaign", "sweep", "campaign_dir", "out_dir", "n_topos",
                             "n_iters", "contract_sig")}
    task = LOAD._load_task(rec)
    got = {str(k): len(v) for k, v in task.topo_iters.items()}
    same = MODEL.positions_sha256(task.topo_iters) == t["positions_sha256"]
    if got != t["topo_counts"] or not same:
        raise ExportError("%s: its detection files differ from the plan's (counts or "
                          "indices); re-plan" % t["name"])
    return task


def marker_path(bank_root, index):
    return os.path.join(bank_root, "tasks", "task_%04d.json" % index)


def other_plans(plan, bank_root) -> List[str]:
    """Plan digests, other than this plan's, found in bank_root's markers and record.

    One bank_root holds one plan's bank: a second plan written into the same
    folder would overwrite a finished bank, or mix two banks under one glob.
    """
    found = set()
    paths = [os.path.join(bank_root, "BANK_RECORD.json")]
    tdir = os.path.join(bank_root, "tasks")
    if os.path.isdir(tdir):
        paths += [os.path.join(tdir, f) for f in sorted(os.listdir(tdir)) if f.endswith(".json")]
    for p in paths:
        if os.path.isfile(p):
            with open(p, "r", encoding="ascii") as fh:
                d = json.load(fh).get("plan_sha256")
            if d != plan["_sha256"]:
                found.add(str(d))
    return sorted(found)


def marker_is_complete(plan, index, bank_root) -> bool:
    """A marker of THIS plan whose every file still has its recorded sha256."""
    path = marker_path(bank_root, index)
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="ascii") as fh:
        m = json.load(fh)
    if m.get("plan_sha256") != plan["_sha256"]:
        return False
    for s in m["shards"]:
        if s["file"] is None:
            continue
        for f, h in ((s["file"], s["sha256"]), (s["json"], s["json_sha256"])):
            p = os.path.join(bank_root, f)
            if not os.path.isfile(p) or sha256_file(p) != h:
                return False
    return True


def export_task(plan, index, bank_root, log=print) -> Dict:
    """Write the task's shards and its marker; return the marker."""
    lb, ln = stage1_modules()
    others = other_plans(plan, bank_root)
    if others:
        raise ExportError("%s holds the outputs of another plan (sha256 %s): one folder per "
                          "plan; give this plan another bank_root, or move that bank aside"
                          % (bank_root, ", ".join(o[:16] for o in others)))
    _check_inputs(plan)
    t = plan["tasks"][index]
    win, lab = plan["window"], plan["label"]
    task = task_inputs(plan, index)
    cfg_like = _CfgView(plan)
    try:
        registry = SL.load_registry(plan["inputs"]["sim_dir"])
    except (ImportError, RuntimeError, ValueError) as exc:
        raise ExportError("the simulator registry at %s: %s" % (plan["inputs"]["sim_dir"], exc))
    with open(plan["inputs"]["freeze"], "r", encoding="ascii") as fh:
        freeze = json.load(fh)
    spec = MODEL.task_label_spec(task, registry, cfg_like, freeze)
    if list(spec.param_names) != lab["param_names"]:
        raise ExportError("%s: label names differ from the plan's" % t["name"])
    bounds = np.asarray(lab["bounds_declared"], dtype=np.float64)
    positions = MODEL.sorted_positions(task)
    kwargs = contract_kwargs(plan, ln.NU_COMPONENTS)
    W, p = int(win["W"]), len(lab["param_names"])
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    shards_out, totals = [], {"read": 0, "kept": 0, "dropped_floor": 0, "snapped": 0}
    for chunk in t["shards"]:
        pos = positions[chunk["start"]:chunk["stop"]]
        n = len(pos)
        X = np.empty((n, W), dtype=np.float32)
        TH = np.empty((n, p), dtype=np.float32)
        RID = np.empty(n, dtype=np.uint64)
        DON = np.empty(n, dtype=np.int64)
        WIN = np.empty(n, dtype=np.int64)
        rates, m, dropped, snapped = [], 0, 0, 0
        for (k, i) in pos:
            path = os.path.join(task.out_dir, "topo_%05d" % k, "mea_iter_%05d.npz" % i)
            det = LOAD.read_detection(path)
            if det["seed_run"] < 0:
                raise ExportError("%s: seed_run %d < 0, no realisation id" % (path, det["seed_run"]))
            try:
                row = ROW.build_row(det, spec, bounds, win["T"], win["dt"], win["sigma_sm"],
                                    win["n_e"], W, win["floor_hz"], win["floor_rtol"])
            except ROW.RowError as exc:
                raise ExportError("%s: %s" % (path, exc))
            rates.append(row["rate_hz"])
            snapped += row["snapped"]
            if not row["kept"]:
                dropped += 1
                continue
            X[m], TH[m] = row["x"], row["u"].astype(np.float32)
            RID[m], DON[m], WIN[m] = det["seed_run"], t["donors"][str(k)], i
            m += 1
        rec = {"shard": chunk["shard"], "start": chunk["start"], "stop": chunk["stop"],
               "read": n, "kept": m, "dropped_floor": dropped, "snapped": snapped,
               "rate_hz_min": float(min(rates)) if rates else None,
               "rate_hz_max": float(max(rates)) if rates else None,
               "file": None, "sha256": None, "json": None, "json_sha256": None}
        if not m:
            stale = os.path.join(bank_root, "shard_%04d.npz" % chunk["shard"])
            if os.path.exists(stale):
                raise ExportError("%s: shard %04d keeps no row, but %s exists from an earlier "
                                  "run; it would enter the bank. Remove it (list, then delete) "
                                  "and resubmit" % (t["name"], chunk["shard"], stale))
        else:
            arrays = {"x": X[:m], "theta": TH[:m], "cls": np.full(m, -1, dtype=np.int64),
                      "nu": np.zeros((m, len(ln.NU_COMPONENTS)), dtype=np.float32),
                      "realisation_id": RID[:m], "donor": DON[:m],
                      "well": np.full(m, int(t["index"]), dtype=np.int64),
                      "batch": np.full(m, int(t["batch"]), dtype=np.int64),
                      "subregion": np.zeros(m, dtype=np.int64), "window_idx": WIN[:m],
                      "contaminated": np.zeros(m, dtype=bool)}
            extra = {"plan_sha256": plan["_sha256"], "task": t["name"], "task_index": index,
                     "on_box": t["on_box"], "shard_index": chunk["shard"],
                     "positions": [chunk["start"], chunk["stop"]], "read": n, "kept": m,
                     "dropped_floor": dropped, "export_version": EXPORT_VERSION,
                     "git": plan.get("git"), "host": socket.gethostname(),
                     "built": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
            sidecar = lb.build_sidecar(extra=extra, **kwargs)
            name = "shard_%04d.npz" % chunk["shard"]
            lb.write_shard(os.path.join(bank_root, name), arrays, sidecar)
            jname = "shard_%04d.json" % chunk["shard"]
            rec.update({"file": name, "sha256": sha256_file(os.path.join(bank_root, name)),
                        "json": jname, "json_sha256": sha256_file(os.path.join(bank_root, jname)),
                        "contract_digest": sidecar["contract_digest"]})
        for key, v in (("read", n), ("kept", m), ("dropped_floor", dropped), ("snapped", snapped)):
            totals[key] += v
        shards_out.append(rec)
        log("[task] %s shard %04d: read %d, kept %d, dropped by the floor %d"
            % (t["name"], chunk["shard"], n, m, dropped))
    if totals["read"] != t["n_iters"]:
        raise ExportError("%s: read %d simulations, the plan has %d"
                          % (t["name"], totals["read"], t["n_iters"]))
    marker = {"plan_sha256": plan["_sha256"], "task": t["name"], "task_index": index,
              "shards": shards_out, "totals": totals, "started": started,
              "finished": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "host": socket.gethostname(), "python": platform.python_version(),
              "numpy": np.__version__, "export_version": EXPORT_VERSION}
    path = marker_path(bank_root, index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="ascii") as fh:
        json.dump(marker, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    return marker


class _CfgView(object):
    """The two profile values task_label_spec reads, taken from the plan."""

    def __init__(self, plan):
        self.topology_axes = tuple(plan["label"]["topology_axes"])
