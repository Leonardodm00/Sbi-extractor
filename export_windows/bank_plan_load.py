"""Block 6.2 -- the bank plan's inputs: loading and validation only
(specs/SPEC.md, Block 6.2; section 3, the rows marked Block 6).

Reads the G2 record, the G3 label freeze and the frozen G1 cohort manifest,
each against its .sha256 and the digest prefix of the profile; then, for each
task of the record, its label contract and seed values (through
sim_reextract_plan, the code that planned G2) and the detections it holds per
topology. Nothing is computed here beyond what checks the files agree with
one another; the contract rule, the twins and the donors are Block 6.3.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "sim_reextract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cohort_manifest as CM          # noqa: E402  (reads SBI_HPC_DIR's DSN tree)
import sim_reextract_plan as PLAN     # noqa: E402

TOPO_RE = re.compile(r"^topo_(\d+)$")
MEA_RE = re.compile(r"^mea_iter_(\d+)\.npz$")
DETECTION_KEYS = ("det_t", "det_ch", "electrode_centers", "conn_prob",
                  "seed_run", "topo_idx", "iter_idx", "theta", "params")
RAW_KEYS = ("conn_prob", "seed_run", "topo_idx")
MAX_REFUSED_SHOWN = 40                 # refused tasks named in one LoadError


class LoadError(RuntimeError):
    """An input the bank plan cannot be built on; the message names the file.

    load_inputs checks every task: one refused task raises its own error; two or
    more raise one error whose first line counts them and whose further lines
    are their errors, one per line (bank_plan prints those above its REFUSED line).
    """


@dataclass(frozen=True)
class TaskInputs:
    name: str                          # <campaign>/<sweep>
    campaign_dir: str                  # the raw sweep folder
    out_dir: str                       # its detections (G2)
    contract: Dict                     # c(t): sim_reextract_plan.task_contract
    contract_sig: str                  # kappa(t), equal to the record's
    seeds: Dict                        # s(t): sim_reextract_plan.task_seed
    topo_iters: Dict[int, List[int]]   # Top(t) -> sorted Iter(t, k)
    n_topos: int                       # |Top(t)| + |E(t)|, the record's count
    n_iters: int
    empty_topos: Tuple[int, ...] = ()  # E(t): topology folders without simulations


@dataclass(frozen=True)
class Inputs:
    record: Dict
    record_sha256: str
    freeze: Dict
    freeze_sha256: str
    manifest_sha256: str
    dt: float                          # Delta_t [s] (w_size)
    sigma_sm: float                    # [s] (gaussian_window)
    electrodes_per_subset: int
    tasks: Tuple[TaskInputs, ...]


def read_verified_json(path, expected_prefix, what) -> Tuple[Dict, str]:
    """The JSON at path, after its bytes match its .sha256 and the prefix."""
    side = path + ".sha256"
    for p in (path, side):
        if not os.path.isfile(p):
            raise LoadError("%s: no file at %s" % (what, p))
    with open(path, "rb") as fh:
        blob = fh.read()
    with open(side, "r", encoding="ascii") as fh:
        tokens = fh.read().split()
    digest = hashlib.sha256(blob).hexdigest()
    if not tokens or tokens[0] != digest:
        raise LoadError("%s: %s does not match its .sha256 (%s against %s)"
                        % (what, path, digest[:16], tokens[0][:16] if tokens else "empty"))
    if not digest.startswith(expected_prefix):
        raise LoadError("%s: %s has sha256 %s, the profile expects %s"
                        % (what, path, digest[:16], expected_prefix))
    return json.loads(blob.decode("utf-8")), digest


def scan_detection_tree(out_dir) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """(Top(t) -> sorted Iter(t, k), E(t) -> folder name), from one scan of out_dir.

    Top(t): the topo_<k> folders holding at least one mea_iter_<i>.npz; E(t): those
    holding none (topologies without simulations; SPEC section 2.5).
    """
    if not os.path.isdir(out_dir):
        raise LoadError("no detections folder at %s" % out_dir)
    topo, empty, seen = {}, {}, set()
    with os.scandir(out_dir) as it:
        folders = [(TOPO_RE.match(e.name), e.name, e.path) for e in it if e.is_dir()]
    for m, name, path in folders:
        if m is None:
            continue
        k = int(m.group(1))
        if k in seen:
            raise LoadError("two folders give topology %d in %s" % (k, out_dir))
        seen.add(k)
        with os.scandir(path) as it:
            idx = sorted(int(m2.group(1)) for m2 in (MEA_RE.match(e.name) for e in it)
                         if m2 is not None)
        if len(set(idx)) != len(idx):
            raise LoadError("repeated iteration indices in %s" % path)
        if idx:
            topo[k] = idx
        else:
            empty[k] = name
    return dict(sorted(topo.items())), dict(sorted(empty.items()))


def scan_detections(out_dir) -> Dict[int, List[int]]:
    """Top(t) -> sorted Iter(t, k): the mea_iter indices per topo_<k> folder."""
    return scan_detection_tree(out_dir)[0]


def _raw_iteration_count(raw_dir) -> int:
    """iter_<i>.npz files in one raw topology folder, matched as G2's plan counted them."""
    with os.scandir(raw_dir) as it:
        return sum(1 for e in it if e.is_file() and PLAN.ITER_RE.match(e.name))


def _check_names(path, topo_idx, iter_idx):
    m = MEA_RE.match(os.path.basename(path)) or re.match(r"^iter_(\d+)\.npz$",
                                                         os.path.basename(path))
    t = TOPO_RE.match(os.path.basename(os.path.dirname(path)))
    if m is None or t is None:
        raise LoadError("%s is not <...>/topo_<k>/<prefix>_<i>.npz" % path)
    if iter_idx is not None and int(m.group(1)) != iter_idx:
        raise LoadError("%s holds iter_idx %d, its name says %d"
                        % (path, iter_idx, int(m.group(1))))
    if int(t.group(1)) != topo_idx:
        raise LoadError("%s holds topo_idx %d, its folder says %d"
                        % (path, topo_idx, int(t.group(1))))


def read_detection(path) -> Dict:
    """One mea_iter file's detections and identity, typed as SPEC section 3."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in DETECTION_KEYS if k not in z.files]
        if missing:
            raise LoadError("%s has no %r" % (path, missing))
        det_t = np.asarray(z["det_t"], dtype=np.float64)
        det_ch = np.asarray(z["det_ch"], dtype=np.int64)
        centers = np.asarray(z["electrode_centers"])
        out = {"det_t": det_t, "det_ch": det_ch,
               "n_e": int(centers.shape[0]) if centers.ndim == 2 else -1,
               "conn_prob": float(z["conn_prob"]), "seed_run": int(z["seed_run"]),
               "topo_idx": int(z["topo_idx"]), "iter_idx": int(z["iter_idx"]),
               "theta": np.asarray(z["theta"], dtype=np.float64),
               "params": np.asarray(z["params"], dtype=np.float64)}
    if det_t.ndim != 1 or det_t.shape != det_ch.shape:
        raise LoadError("%s: det_t %r and det_ch %r differ in shape"
                        % (path, det_t.shape, det_ch.shape))
    if det_t.size and not np.all(np.isfinite(det_t)):
        raise LoadError("%s: det_t holds NaN or Inf" % path)
    if not np.isfinite(out["conn_prob"]):
        raise LoadError("%s: conn_prob is not finite (absent from its raw file)" % path)
    if out["n_e"] < 1:
        raise LoadError("%s: electrode_centers is not (n_e, 2)" % path)
    _check_names(path, out["topo_idx"], out["iter_idx"])
    return out


def read_raw_ident(path) -> Dict:
    """The raw iteration file's topology value and noise key."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in RAW_KEYS if k not in z.files]
        if missing:
            raise LoadError("%s has no %r" % (path, missing))
        out = {"conn_prob": float(z["conn_prob"]), "seed_run": int(z["seed_run"]),
               "topo_idx": int(z["topo_idx"])}
    if not np.isfinite(out["conn_prob"]):
        raise LoadError("%s: conn_prob is not finite" % path)
    _check_names(path, out["topo_idx"], None)
    return out


def _load_task(rec) -> TaskInputs:
    name = "%s/%s" % (rec["campaign"], rec["sweep"])
    try:
        contract = PLAN.task_contract(rec["campaign_dir"])
        seeds = PLAN.task_seed(rec["campaign_dir"])
    except (OSError, ValueError) as exc:
        raise LoadError("%s: its job_args.json / manifest.json cannot be read (%s)"
                        % (name, exc))
    sig = PLAN.contract_signature(contract)
    if sig != rec.get("contract_sig"):
        raise LoadError("%s: contract signature %s, the record says %s (%s)"
                        % (name, sig, rec.get("contract_sig"), rec["campaign_dir"]))
    topo, empty = scan_detection_tree(rec["out_dir"])
    n_iters = sum(len(v) for v in topo.values())
    n_topos = len(topo) + len(empty)
    if n_topos != int(rec["n_topos"]) or n_iters != int(rec["n_iters"]):
        raise LoadError("%s: %d topology folders (%d without a detection file) and %d "
                        "detection files in %s, the record says %s and %s"
                        % (name, n_topos, len(empty), n_iters, rec["out_dir"],
                           rec["n_topos"], rec["n_iters"]))
    for k, folder in empty.items():
        raw_dir = os.path.join(rec["campaign_dir"], folder)
        if not os.path.isdir(raw_dir):
            raise LoadError("%s: %s holds no detection file and the raw task has no "
                            "folder %s" % (name, os.path.join(rec["out_dir"], folder), raw_dir))
        n_raw = _raw_iteration_count(raw_dir)
        if n_raw:
            raise LoadError("%s: %s holds no detection file, but %s holds %d iteration "
                            "file(s): its detections are missing"
                            % (name, os.path.join(rec["out_dir"], folder), raw_dir, n_raw))
    return TaskInputs(name=name, campaign_dir=rec["campaign_dir"], out_dir=rec["out_dir"],
                      contract=contract, contract_sig=sig, seeds=seeds,
                      topo_iters=topo, n_topos=n_topos, n_iters=n_iters,
                      empty_topos=tuple(sorted(empty)))


def load_inputs(cfg) -> Inputs:
    """Everything Block 6 reads, checked against its digests and the record."""
    record, record_sha = read_verified_json(cfg.record_path, cfg.record_sha256_prefix,
                                            "record")
    if not isinstance(record.get("tasks"), list) or not record["tasks"]:
        raise LoadError("record: %s lists no tasks" % cfg.record_path)
    freeze, freeze_sha = read_verified_json(cfg.freeze_path, cfg.freeze_sha256_prefix,
                                            "label freeze")
    if list(freeze.get("topology_axes") or []) != list(cfg.topology_axes):
        raise LoadError("label freeze: %s has topology_axes %r, the profile expects %r"
                        % (cfg.freeze_path, freeze.get("topology_axes"),
                           list(cfg.topology_axes)))
    if not os.path.isfile(cfg.cohort_manifest_path + ".sha256"):
        raise LoadError("cohort manifest: no file at %s.sha256" % cfg.cohort_manifest_path)
    try:
        manifest = CM.read_manifest(cfg.cohort_manifest_path)
        dt, sigma_sm, eps, _recorded = CM.sim_preprocessing_from_manifest(manifest)
    except (OSError, KeyError, ValueError, CM.ManifestError) as exc:
        raise LoadError("cohort manifest: %s (%s)" % (cfg.cohort_manifest_path, exc))
    if not manifest["_digest"].startswith(cfg.cohort_manifest_sha256_prefix):
        raise LoadError("cohort manifest: %s has sha256 %s, the profile expects %s"
                        % (cfg.cohort_manifest_path, manifest["_digest"][:16],
                           cfg.cohort_manifest_sha256_prefix))
    tasks, refused = [], []
    for rec in record["tasks"]:
        try:
            tasks.append(_load_task(rec))
        except LoadError as exc:            # every task is checked; one error names them all
            refused.append(exc)
    if len(refused) == 1:
        raise refused[0]
    if refused:                             # line 1 counts them, one further line per task
        shown = [str(exc) for exc in refused[:MAX_REFUSED_SHOWN]]
        more = len(refused) - len(shown)
        raise LoadError("%d of %d tasks refused\n%s%s"
                        % (len(refused), len(record["tasks"]), "\n".join(shown),
                           ("\n... and %d more" % more) if more else ""))
    tasks = tuple(tasks)
    return Inputs(record=record, record_sha256=record_sha, freeze=freeze,
                  freeze_sha256=freeze_sha, manifest_sha256=manifest["_digest"],
                  dt=dt, sigma_sm=sigma_sm, electrodes_per_subset=eps, tasks=tasks)
