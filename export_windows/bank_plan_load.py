"""Block 6.2 -- the bank plan's inputs: loading and validation only
(specs/SPEC.md, Block 6.2; section 3, the rows marked Block 6).

Reads the G2 record, the G3 label freeze and the frozen G1 cohort manifest,
each against its .sha256 and the digest prefix of the profile; then, for each
task of the record, its label contract and seed values (through
sim_reextract_plan, the code that planned G2), the job_args keys that bound
the topology axes outside the contract (D-107: stoa_gate_lo, stoa_gate_hi)
and the detections it holds per topology. Nothing is computed here beyond
what checks the files agree with one another; the contract rule, the twins
and the donors are Block 6.3.

Topology values that the detections do not carry (D-107: stoa_gate_p) are
read from the raw iteration files: read_raw_topology reads a topology's first
and last iterations and requires one value.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "sim_reextract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cohort_manifest as CM          # noqa: E402  (reads SBI_HPC_DIR's DSN tree)
import sbi_labels as SL               # noqa: E402
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
    # job_args values bounding the axes of SL.JOB_ARGS_TOPOLOGY_BOUNDS, outside
    # the contract (D-107): key -> value, "absent" when the file has none
    topology_args: Dict = field(default_factory=dict)


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


def read_raw_ident(path, axes: Sequence[str] = ()) -> Dict:
    """The raw iteration file's topology value and noise key; with axes, also
    the values of those topology axes ("topology": {axis: float})."""
    axes = [str(a) for a in axes]
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in tuple(RAW_KEYS) + tuple(axes) if k not in z.files]
        if missing:
            raise LoadError("%s has no %r" % (path, missing))
        shaped = {a: z[a].shape for a in axes if z[a].ndim != 0}
        if shaped:
            raise LoadError("%s: %r is not a scalar (shapes %r)" % (path, sorted(shaped), shaped))
        out = {"conn_prob": float(z["conn_prob"]), "seed_run": int(z["seed_run"]),
               "topo_idx": int(z["topo_idx"]),
               "topology": {a: float(z[a]) for a in axes}}
    if not np.isfinite(out["conn_prob"]):
        raise LoadError("%s: conn_prob is not finite" % path)
    bad = {a: v for a, v in out["topology"].items() if not np.isfinite(v)}
    if bad:
        raise LoadError("%s: %r is not finite" % (path, bad))
    _check_names(path, out["topo_idx"], None)
    return out


def raw_topology_axes(topology_axes) -> List[str]:
    """Tau_raw (SPEC sec. 2.5): the axes of topology_axes the bank reads from the
    raw iteration files, with their box from job_args (SL.JOB_ARGS_TOPOLOGY_BOUNDS;
    D-107). The Weibull kernel axes are not among them: a bank with one of them
    is refused at its first row, as before."""
    return [str(a) for a in topology_axes if str(a) in SL.JOB_ARGS_TOPOLOGY_BOUNDS]


def read_topology_args(campaign_dir) -> Dict:
    """The task's job_args.json values bounding the axes of
    SL.JOB_ARGS_TOPOLOGY_BOUNDS (D-107), "absent" for a key the file lacks.
    They are read beside the label contract, not in it: the contract signature
    of sim_reextract_plan, and so every record's, is unchanged."""
    path = os.path.join(campaign_dir, "job_args.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            ja = json.load(fh)
    except (OSError, ValueError) as exc:
        raise LoadError("%s cannot be read (%s)" % (path, exc))
    if not isinstance(ja, dict):
        raise LoadError("%s is not a JSON object" % path)
    keys = [k for pair in SL.JOB_ARGS_TOPOLOGY_BOUNDS.values() for k in pair]
    return {k: ja.get(k, "absent") for k in keys}


def dispatch_size(campaign_dir) -> int:
    """[Block 10, SBI D-148 (5)] K(t): the simulations the task dispatched per
    topology, job_args.json's `_resolved_n_workers` x `n_params_per_worker`
    (both positive integers, else LoadError naming the task and the key)."""
    raise NotImplementedError("Block 10")


def raw_finished_counts(campaign_dir) -> Dict[int, int]:
    """[Block 10] f(t, k) for every raw folder topo_<k> of the task: its
    iter_<i>.json files plus the non-blank lines of its _failures.jsonl (0
    without the file)."""
    raise NotImplementedError("Block 10")


def read_raw_topology(campaign_dir, k, iters, axes) -> Dict[str, float]:
    """eta_a(t, k) for each axis a: the value the raw files of topology k hold at
    the first and the last of its iterations, which must be equal (one draw per
    topology, D-107). {} when axes is empty (no file is opened)."""
    axes = [str(a) for a in axes]
    if not axes:
        return {}
    if not len(iters):
        raise LoadError("%s topology %d: no iteration to read %r from"
                        % (campaign_dir, int(k), axes))
    folder = os.path.join(campaign_dir, "topo_%05d" % int(k))
    ends = sorted({int(min(iters)), int(max(iters))})
    found = []
    for i in ends:
        path = os.path.join(folder, "iter_%05d.npz" % i)
        ident = read_raw_ident(path, axes)
        if ident["topo_idx"] != int(k):
            raise LoadError("%s holds topo_idx %d, expected %d" % (path, ident["topo_idx"], int(k)))
        found.append((i, ident["topology"]))
    first, last = found[0][1], found[-1][1]
    differ = [a for a in axes if first[a] != last[a]]
    if differ:
        raise LoadError("%s topology %d: %s differs between iteration %d and iteration %d "
                        "(%s): one value per topology was expected"
                        % (campaign_dir, int(k), ", ".join(differ), found[0][0], found[-1][0],
                           "; ".join("%s %r vs %r" % (a, first[a], last[a]) for a in differ)))
    return dict(first)


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
    try:
        topology_args = read_topology_args(rec["campaign_dir"])
    except LoadError as exc:
        raise LoadError("%s: %s" % (name, exc))
    return TaskInputs(name=name, campaign_dir=rec["campaign_dir"], out_dir=rec["out_dir"],
                      contract=contract, contract_sig=sig, seeds=seeds,
                      topo_iters=topo, n_topos=n_topos, n_iters=n_iters,
                      empty_topos=tuple(sorted(empty)), topology_args=topology_args)


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
