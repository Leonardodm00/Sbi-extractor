"""Test helper for Blocks 6-8 -- a synthetic Giulia-like record (not shipped data).

write_fixture(root) builds, under root, the files the bank code reads, small
enough to write in a second, with every value known in advance:

  sim/campaign_cadex_hhgap_v1/sweep_intel_task0000   O_N box [0.03, 3.0]
  sim/campaign_cadex_hhgap_v2/sweep_intel_task0000   O_N box [0.01, 3.0]
  sim/campaign_cadex_hhgap_v5/sweep_cpu_task0003     O_N box [0.01, 3.0]

The first two share their seed values: a twin pair across two label
contracts, as D-070 found on the real record. Their topology 0 shares its
conn_prob (a shared network), their topology 1 does not, and their topology 2
holds disjoint iteration indices (no common simulation to compare). Every
parameter vector is drawn from one uniform u per (seed values, topology,
iteration), so the twins differ in O_N alone, mapped log-uniformly into their
two boxes (N1's finding). Each task has raw iteration files (sim/...),
detections (mea/...); there is a G2-style record with its .sha256, a label
freeze, a frozen cohort manifest, a stand-in simulator registry (registry/),
and the real arm's archives (real/) under a cohort manifest that lists them.

The spike counts of each detection file are spikes_of(task, k, i), so a test
knows the mean rate of every window in advance (N / 180 Hz).

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zlib
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "sim_reextract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cohort_manifest as CM          # noqa: E402
import sim_reextract_plan as PLAN     # noqa: E402

T_SIM = 180.0
N_PARAMS = 37                          # the hhgap registry's width
PARAM_NAMES = ["p%02d" % j for j in range(N_PARAMS - 1)] + ["O_N"]
ACTIVE = list(range(15, 37))           # 22 active axes, O_N (index 36) last
LIVE_BOUNDS = [[0.5, 2.0]] * (N_PARAMS - 1) + [[0.03, 3.0]]   # the simulator's source
KERNEL_BOUNDS = [[0.1, 1.0], [20.0, 200.0], [0.5, 3.0]]

# task -> (campaign, sweep, O_N box, seed master, {topology k: [iterations]})
TASKS = [
    ("campaign_cadex_hhgap_v1", "sweep_intel_task0000", [0.03, 3.0], 3000000,
     {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3], 2: [5, 6]}),
    ("campaign_cadex_hhgap_v2", "sweep_intel_task0000", [0.01, 3.0], 3000000,
     {0: [0, 1, 2, 3, 4], 1: [0, 1, 2], 2: [0, 1]}),
    ("campaign_cadex_hhgap_v5", "sweep_cpu_task0003", [0.01, 3.0], 4000003,
     {0: [0, 1, 2, 3, 4, 5], 1: [0, 1], 2: [0, 1, 2]}),
]

# the real arm: (class name, root name, well, T_rec [s], rate per subregion [Hz])
REAL_WELLS = [
    ("100N0A_wo", "Batch1_100N0A_wo", "ptrain_1_DIV35_100N0A_nbasal_0001", 300.0, 2.0),
    ("100N0A_wo", "Batch2_100N0A_wo", "ptrain_2_DIV35_100N0A_nbasal_0001", 300.0, 0.05),
    ("50N50A", "AraC_50N50A", "ptrain_3_DIV35_50N50A_nbasal_0001", 600.0, 1.0),
    ("70N30A", "AraC_70N30A", "ptrain_4_DIV35_70N30A_nbasal_0001", 600.0, 3.0),
]
REAL_CLASSES = ["100N0A_wo", "50N50A", "70N30A"]


def task_name(t):
    return "%s/%s" % (t[0], t[1])


def conn_prob_of(t, k):
    """Topology 0 of the twin pair shares one value; everything else differs."""
    if t[3] == 3000000 and k == 0:
        return 0.123456789012
    return round(0.05 + 0.01 * (k + 1) + 0.1 * TASKS.index(t), 12)


def seed_run_of(t, k, i):
    """The twins share their noise keys (same seed streams)."""
    return int(t[3] + 1000 * k + i)


def spikes_of(t, k, i):
    """Detections in [0, 180) s: 0, 17, 18, 19 or more, known per file."""
    return [0, 17, 18, 19, 500, 40][(k + i + TASKS.index(t)) % 6]


def params_of(t, k, i):
    """Natural parameters: one u per (seed, k, i); O_N log-uniform in the task's box."""
    rng = np.random.default_rng(zlib.crc32(("%d/%d/%d" % (t[3], k, i)).encode("ascii")))
    u = rng.uniform(0.0, 1.0, N_PARAMS)
    params = np.empty(N_PARAMS)
    for j in range(N_PARAMS - 1):
        lo, hi = LIVE_BOUNDS[j]
        params[j] = lo + u[j] * (hi - lo)
    lo, hi = t[2]
    params[-1] = lo * (hi / lo) ** u[-1]
    theta = params.copy()
    theta[-1] = np.log(params[-1])       # O_N is the one log axis (rule 1)
    return params, theta


def _write_json(path, doc):
    blob = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with open(path, "w", encoding="ascii") as fh:
        fh.write(blob)
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()
    with open(path + ".sha256", "w", encoding="ascii") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(path)))
    return digest


def _write_registry(reg_dir):
    """Stand-in simulator modules with the attributes sbi_labels.load_registry reads."""
    os.makedirs(reg_dir, exist_ok=True)
    theta_bounds = [list(b) for b in LIVE_BOUNDS]
    theta_bounds[-1] = [float(np.log(LIVE_BOUNDS[-1][0])), float(np.log(LIVE_BOUNDS[-1][1]))]
    with open(os.path.join(reg_dir, "HPC_main_sweep.py"), "w", encoding="ascii") as fh:
        fh.write("PARAM_BOUNDS = %r\nPARAM_BOUNDS_THETA = %r\nLOG_PARAMS = [%d]\n"
                 "LOG_BASE = 'natural'\nKERNEL_BOUNDS = %r\n"
                 % (LIVE_BOUNDS, theta_bounds, N_PARAMS - 1, KERNEL_BOUNDS))
    with open(os.path.join(reg_dir, "HPC_single_run.py"), "w", encoding="ascii") as fh:
        fh.write("PARAM_NAMES = %r\nPARAM_UNITS = %r\nSWEEP_GROUPS = %r\n"
                 % (PARAM_NAMES, ["a.u."] * N_PARAMS, {"synapse_astro": ACTIVE}))


def _write_real(root, w_size=0.01, sigma=0.02):
    """The real arm's archives and a cohort manifest that lists them."""
    real = os.path.join(root, "real")
    wells = []
    rng = np.random.default_rng(3)
    for cls_name, root_name, well, t_rec, rate in REAL_WELLS:
        out_dir = os.path.join(cls_name, root_name, well)
        os.makedirs(os.path.join(real, out_dir), exist_ok=True)
        K = int(round(t_rec / w_size))
        for s in range(9):
            n = int(round(rate * t_rec))
            idx = rng.integers(0, K, n)
            x = np.bincount(idx, minlength=K).astype(np.float32)   # spikes per bin, n_e = 1
            np.savez_compressed(os.path.join(real, out_dir, "trace_subregion_%02d.npz" % s),
                                ifr_trace=x, fs_ifr=np.float64(1.0 / w_size),
                                T_rec=np.float64(t_rec), subregion_index=np.int64(s),
                                culture_id=np.array(well), mode=np.array("per_region_single"))
        wells.append({"culture": "%s__%s" % (root_name, well), "out_dir": out_dir,
                      "n_archives": 9, "T_rec": t_rec, "n_present": 20, "n_samples_raw": 0})
    return real, wells


def write_fixture(root, extra_topos=None) -> Dict:
    """Write the fixture under root; return its paths, digests and truths.

    extra_topos: {task index: {k: [iterations]}} added to that task's topologies; an
    empty list writes the folder topo_<k> in both the raw task and its detections with
    no file in either -- a topology without simulations, which G2's record counts
    among n_topos (sim_reextract_plan.scan_task counts every topo_* folder).
    """
    sim, mea, art = (os.path.join(root, d) for d in ("sim", "mea", "artifacts"))
    _write_registry(os.path.join(root, "registry"))
    rec_tasks, truth = [], {}
    rng = np.random.default_rng(7)
    for n_task, t in enumerate(TASKS):
        camp, sweep, on_box, seed, topo = t
        topo = dict(topo)
        topo.update((extra_topos or {}).get(n_task, {}))
        sd = os.path.join(sim, camp, sweep)
        od = os.path.join(mea, camp, sweep)
        os.makedirs(sd, exist_ok=True)
        job_args = {"simtime": T_SIM, "conn_prob_lo": 0.05, "conn_prob_hi": 0.4,
                    "p0_lo": 0.1, "p0_hi": 1.0, "d0_lo": 20.0, "d0_hi": 200.0,
                    "beta_lo": 0.5, "beta_hi": 3.0, "_resolved_seed_master": seed,
                    "seed_master": None, "seed_device": 0, "seed_neuron": 1,
                    "seed_synapse": 2, "seed_astro": 3}
        bounds = [list(b) for b in LIVE_BOUNDS]
        bounds[-1] = list(on_box)
        manifest = {"active_indices": ACTIVE, "sweep_group": "synapse_astro",
                    "param_names": PARAM_NAMES, "log_params": ["O_N"],
                    "log_transform": "natural", "conn_rule": "flat",
                    "param_bounds": bounds, "manifest_version": 3}
        for name, doc in (("job_args.json", job_args), ("manifest.json", manifest)):
            with open(os.path.join(sd, name), "w", encoding="ascii") as fh:
                json.dump(doc, fh, indent=2, sort_keys=True)
        for k, iters in topo.items():
            os.makedirs(os.path.join(sd, "topo_%05d" % k), exist_ok=True)
            os.makedirs(os.path.join(od, "topo_%05d" % k), exist_ok=True)
            for i in iters:
                cp, sr, n = conn_prob_of(t, k), seed_run_of(t, k, i), spikes_of(t, k, i)
                params, theta = params_of(t, k, i)
                np.savez_compressed(os.path.join(sd, "topo_%05d" % k, "iter_%05d.npz" % i),
                                    theta=theta, params=params,
                                    conn_rule=np.array("flat"), conn_prob=np.float64(cp),
                                    topo_idx=np.int32(k), seed_run=np.int64(sr))
                det_t = np.sort(rng.uniform(0.5, T_SIM - 0.5, n)).astype(np.float32)
                np.savez_compressed(os.path.join(od, "topo_%05d" % k, "mea_iter_%05d.npz" % i),
                                    det_t=det_t, det_ch=np.zeros(n, dtype=np.int32),
                                    electrode_centers=np.array([[150.0, 150.0]], np.float32),
                                    params=params, theta=theta,
                                    conn_prob=np.float64(cp), topo_idx=np.int32(k),
                                    iter_idx=np.int32(i), seed_run=np.int64(sr),
                                    simtime=np.float64(np.ceil(det_t.max()) if n else 1.0))
                truth[(task_name(t), k, i)] = {"conn_prob": cp, "seed_run": sr, "n": n,
                                               "params": params, "theta": theta}
        sig = PLAN.contract_signature(PLAN.task_contract(sd))
        rec_tasks.append({"campaign": camp, "sweep": sweep, "campaign_dir": sd,
                          "out_dir": od, "n_topos": len(topo),
                          "n_iters": sum(len(v) for v in topo.values()),
                          "contract_sig": sig})
    os.makedirs(art, exist_ok=True)
    record = {"record": "sim_reextraction", "tasks": rec_tasks,
              "counts": {"tasks": len(rec_tasks),
                         "iterations": sum(r["n_iters"] for r in rec_tasks)}}
    rec_path = os.path.join(mea, "REEXTRACTION_RECORD.json")
    frz_path = os.path.join(art, "label_axes_giulia_hhgap.json")
    man_path = os.path.join(art, "cohort_manifest_giulia", "cohort_manifest.json")
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    real_root, real_wells = _write_real(root)
    digests = {"record": _write_json(rec_path, record),
               "freeze": _write_json(frz_path, {
                   "topology_axes": ["conn_prob"], "conn_rule_observed": ["flat"],
                   "excluded_axes": {"p0_conn": "fixture", "d0_conn": "fixture",
                                     "beta_conn": "fixture"}}),
               "manifest": CM.write_manifest({
                   "manifest_version": CM.MANIFEST_VERSION, "n_units": 9 * len(real_wells),
                   "extract_root": real_root, "classes": REAL_CLASSES,
                   "n_wells": len(real_wells), "n_subsets_per_well": 9,
                   "derived": {"fs_ifr": 100.0, "sigma_sm_bins": 2.0},
                   "wells": real_wells,
                   "preprocessing": {"fs_raw": 10000.0, "index_base": 0, "grid_width": 10,
                                     "n_subsets": 9, "electrodes_per_subset": 1,
                                     "mfr_threshold": 0.1, "w_size": 0.01,
                                     "gaussian_window": 0.02}}, man_path)}
    return {"root": root, "record": rec_path, "freeze": frz_path, "manifest": man_path,
            "digests": digests, "truth": truth, "tasks": rec_tasks}


def write_profile(fx, path, **overrides) -> str:
    """A profile JSON pointing at the fixture (absolute paths throughout)."""
    prof = {"record_path": fx["record"], "record_sha256_prefix": fx["digests"]["record"][:16],
            "freeze_path": fx["freeze"], "freeze_sha256_prefix": fx["digests"]["freeze"][:16],
            "cohort_manifest_path": fx["manifest"],
            "cohort_manifest_sha256_prefix": fx["digests"]["manifest"][:16],
            "sim_dir": os.path.join(fx["root"], "registry"), "on_axis": "O_N",
            "on_declared_box": [0.01, 3.0], "topology_axes": ["conn_prob"],
            "expected_p": len(ACTIVE) + 1,
            "t_win_s": 180.0, "n_e": 1, "floor_hz": 0.1, "floor_rtol": 1e-6,
            "sample_seed": 11, "bootstrap_resamples": 200, "bootstrap_seed": 12,
            "bytes_sample_rows": 50, "rows_per_shard": 100,
            "bank_root": os.path.join(fx["root"], "banks", "giulia_sim"),
            "real_bank_root": os.path.join(fx["root"], "banks", "giulia_real"),
            "plan_out": os.path.join(fx["root"], "plan", "bank_plan.json")}
    prof.update(overrides)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(prof, fh, indent=2, sort_keys=True)
    return path


def topo_iters_of(t) -> Dict[int, List[int]]:
    return {k: list(v) for k, v in t[4].items()}
