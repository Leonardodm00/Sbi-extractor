"""Block 6.4 -- the bank plan's sample: one simulation per topology, the twin
check, and the estimates of D-080 (specs/SPEC.md, Block 6.4; sec. 2.5).

  draw_iterations   which simulation of each topology is read (Q6: uniform,
                    seeded per donor key, so both sides of a twin pair are read
                    at the same iteration index wherever they share one)
  sample_topology   reads that simulation's detections and its raw iteration
                    file, checks they agree, and builds its row (bank_row)
  twin_check        per twin pair and shared topology: equal conn_prob? equal
                    seed_run at the same index? (reported, never refused)
  kept_share        the row-weighted share of the floor (Q8) with a 95%
                    percentile bootstrap interval over donors (D-076 groups)
  bytes_per_row     the compressed size of the kept sampled windows, per row
  free_space        the free bytes under the bank's folder, and lfs quota's text

Estimates only: nothing here filters, caps or decides (D-080).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import zlib
from typing import Dict, List

import numpy as np

import bank_plan_load as LOAD
import bank_row as ROW


def draw_iterations(tasks, groups, cfg) -> Dict:
    """(task name, k) -> {"i": iteration index, "comparable": bool}."""
    by_name = {t.name: t for t in tasks}
    out = {}
    for g in groups:
        members = [by_name[n] for n in g["tasks"]]
        topos = sorted({k for m in members for k in m.topo_iters})
        for k in topos:
            have = [m for m in members if k in m.topo_iters]
            common = sorted(set.intersection(*[set(m.topo_iters[k]) for m in have]))
            if common:
                key = "%s|%d" % (g["seed_key"], k)
                rng = np.random.default_rng(np.random.SeedSequence(
                    [int(cfg.sample_seed), zlib.crc32(key.encode("ascii"))]))
                i = int(common[int(rng.integers(0, len(common)))])
                for m in have:
                    out[(m.name, k)] = {"i": i, "comparable": len(have) > 1}
            else:
                for m in have:
                    key = "%s|%d" % (m.name, k)
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [int(cfg.sample_seed), zlib.crc32(key.encode("ascii"))]))
                    its = m.topo_iters[k]
                    out[(m.name, k)] = {"i": int(its[int(rng.integers(0, len(its)))]),
                                        "comparable": False}
    return out


def sample_topology(task, k, i, inputs, cfg, spec_task, bounds) -> Dict:
    """Read one simulation and its raw file; return its row summary and window."""
    mea = os.path.join(task.out_dir, "topo_%05d" % k, "mea_iter_%05d.npz" % i)
    raw = os.path.join(task.campaign_dir, "topo_%05d" % k, "iter_%05d.npz" % i)
    det = LOAD.read_detection(mea)
    ident = LOAD.read_raw_ident(raw)
    if det["topo_idx"] != k or ident["topo_idx"] != k:
        raise LOAD.LoadError("%s: topo_idx %d / %d, expected %d"
                             % (task.name, det["topo_idx"], ident["topo_idx"], k))
    if det["conn_prob"] != ident["conn_prob"] or det["seed_run"] != ident["seed_run"]:
        raise LOAD.LoadError("%s topology %d iteration %d: the detections carry conn_prob "
                             "%r seed_run %r, the raw file %r %r"
                             % (task.name, k, i, det["conn_prob"], det["seed_run"],
                                ident["conn_prob"], ident["seed_run"]))
    row = ROW.build_row(det, spec_task, bounds, cfg.t_win_s, inputs.dt, inputs.sigma_sm,
                        cfg.n_e, int(round(cfg.t_win_s / inputs.dt)),
                        cfg.floor_hz, cfg.floor_rtol)
    return {"task": task.name, "k": int(k), "i": int(i), "n": len(task.topo_iters[k]),
            "conn_prob": det["conn_prob"], "seed_run": det["seed_run"],
            "rate_hz": row["rate_hz"], "kept": row["kept"], "n_det": row["n_det"],
            "u_min": float(row["u"].min()), "u_max": float(row["u"].max()),
            "snapped": row["snapped"], "x": row["x"]}


def twin_check(groups, samples) -> List[Dict]:
    """For each twin pair and topology both hold: equal conn_prob and seed_run?"""
    by = {(s["task"], s["k"]): s for s in samples}
    out = []
    for g in groups:
        if not g["twin"]:
            continue
        a, b = g["tasks"][0], g["tasks"][1:]
        for other in b:
            ks = sorted({k for (n, k) in by if n == a} & {k for (n, k) in by if n == other})
            n_cp = sum(by[(a, k)]["conn_prob"] == by[(other, k)]["conn_prob"] for k in ks)
            comp = [k for k in ks if by[(a, k)]["i"] == by[(other, k)]["i"]]
            n_sr = sum(by[(a, k)]["seed_run"] == by[(other, k)]["seed_run"] for k in comp)
            out.append({"tasks": [a, other], "shared_topologies": len(ks),
                        "conn_prob_equal": int(n_cp), "seed_run_comparable": len(comp),
                        "seed_run_equal": int(n_sr)})
    return out


def donor_totals(samples, donors_of):
    """Per donor (sorted ids): the kept rows sum n * kept and the rows sum n of its
    sampled topologies -- the units the bootstrap resamples (D-076)."""
    w = np.array([s["n"] for s in samples], dtype=np.float64)
    kept = np.array([1.0 if s["kept"] else 0.0 for s in samples])
    donors = np.array([donors_of[(s["task"], s["k"])] for s in samples])
    uniq, inv = np.unique(donors, return_inverse=True)
    num = np.bincount(inv, weights=w * kept, minlength=uniq.size)
    den = np.bincount(inv, weights=w, minlength=uniq.size)
    return uniq, num, den


def bootstrap_reps(num, den, resamples, seed) -> np.ndarray:
    """Shares of `resamples` resamples of the donors, drawn with replacement."""
    rng = np.random.default_rng(int(seed))
    d = int(num.size)
    reps = np.empty(int(resamples))
    for b in range(reps.size):
        pick = rng.integers(0, d, d)
        reps[b] = np.sum(num[pick]) / np.sum(den[pick])
    return reps


def kept_share(samples, donors_of, cfg) -> Dict:
    """Row-weighted share of kept rows, with a donor-level bootstrap interval."""
    w = np.array([s["n"] for s in samples], dtype=np.float64)
    kept = np.array([1.0 if s["kept"] else 0.0 for s in samples])
    p_hat = float(np.sum(w * kept) / np.sum(w))
    uniq, num, den = donor_totals(samples, donors_of)
    reps = bootstrap_reps(num, den, cfg.bootstrap_resamples, cfg.bootstrap_seed)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    rates = np.array([s["rate_hz"] for s in samples])
    return {"p_hat": p_hat, "ci95": [float(lo), float(hi)],
            "unweighted": float(np.mean(kept)), "n_topologies": int(len(samples)),
            "n_donors": int(uniq.size), "resamples": int(reps.size),
            "rate_hz_quantiles": {q: float(v) for q, v in zip(
                ("min", "q05", "q25", "q50", "q75", "q95", "max"),
                np.quantile(rates, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))}}


def bytes_per_row(windows) -> float:
    """Compressed bytes per row of an (m, W) float32 block, as numpy writes it."""
    block = np.stack(windows).astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, x=block)
    return float(len(buf.getvalue())) / block.shape[0]


def free_space(bank_root) -> Dict:
    """Free bytes of the filesystem that will hold the bank, and lfs quota's text."""
    probe = os.path.abspath(bank_root)
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    du = shutil.disk_usage(probe)
    out = {"path_checked": probe, "free_bytes": int(du.free), "total_bytes": int(du.total),
           "lfs_quota": None}
    lfs = shutil.which("lfs")
    if lfs:
        mount = probe
        while not os.path.ismount(mount):
            mount = os.path.dirname(mount)
        user = os.environ.get("USER") or ""
        try:
            res = subprocess.run([lfs, "quota", "-u", user, mount], capture_output=True,
                                 text=True, timeout=30)
            out["lfs_quota"] = (res.stdout + res.stderr).strip()[:2000]
        except (OSError, subprocess.SubprocessError) as exc:
            out["lfs_quota"] = "lfs quota failed: %s" % exc
    return out
