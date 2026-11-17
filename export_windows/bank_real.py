#!/usr/bin/env python3
"""Block 8 -- B5.5, the Giulia real bank (specs/SPEC.md, Block 8).

    python3 export_windows/bank_real.py --profile giulia [--out-root DIR]

Reads the frozen G1 cohort manifest (its digest must begin with the profile's
prefix) and, for every well it lists and every subregion archive of that well,
the stored per-electrode-mean IFR (real_source.read_real_trace: the bytes off
disk, checked against fs_ifr = 1 / w_size). Each trace is cut into windows of
W samples, back to back from t = 0 (D-056): window j covers
[j W, (j + 1) W) for j in {0, ..., floor((K - W) / W)}. All windows go into one
shard, shard_0000.npz, in the profile's real_bank_root (~/banks/giulia_real,
D-046), in the Stage 1 contract:

  x               (n, W) float32  the window, spikes per bin per electrode
  theta           (n, 0) float32  no labels on the real arm: zero columns
  cls             (n,)   int64    the condition: the index of the well's class
                                  in the manifest's classes
  nu              (n, 5) float32  zeros
  realisation_id  (n,)   uint64   the well's index (one recording per well)
  donor           (n,)   int64    the cohort map's donor id of the well; without a
                                  map the well's index: a culture is the unit that
                                  is never split (run_joint_arms grouped_split)
  well            (n,)   int64    the well's index in the manifest
  batch           (n,)   int64    the index of the well's class root (plate folder)
  subregion       (n,)   int64    the archive's subregion, 0-8 (D-043)
  window_idx      (n,)   int64    j
  contaminated    (n,)   bool     False
  rep_group       (n,)   int64    the cohort map's replicate pairing group id of
                                  the well; without a map = donor (= well).
                                  The stage-1 contract's optional row array
                                  (SBI latent_bank.OPTIONAL_ROW_ARRAYS, 88c32e5)

The cohort map (Block 9, SBI D-148 (4); REAL_VERSION 2): the profile's
`cohort_map`, read by cohort_map.read_cohort_map, gives each well its donor
and pairing group and may leave wells out: an excluded well contributes no
row (its archives are not read); the ids are those of the map (sorted
labels of the included wells); well, realisation_id, batch (over every
well of the manifest), cls, subregion, window_idx and x are as before.
Without a map (`cohort_map` null) the rows are those of REAL_VERSION 1 and
rep_group = donor = well. The sidecar's latent_spec carries "cohort_map":
null, or the map's sha256, its labels, the excluded out_dirs and the number
of included wells; its per_class and n_wells count the included wells only.

Windows whose mean rate is below the simulated bank's floor (D-075) are kept
and counted: they lie outside the training support of x. A folder that
already holds shard_0000.npz is refused: a written bank is never overwritten.

Too-short traces (Block 8, amendment 2, F4; SPEC sec. 4). A trace with K < W
gives no row, and is counted: per class, when its count n is above 0, the
line `[real] <class>: <n> trace(s) shorter than one window (K < W, W = <W>
samples): no row`, then one line per well of that class left with no row at
all, `[real]   <culture> (<out_dir>): every trace shorter than one window, no
row`. per_class[c]["wells"] counts the wells with rows. Every well of the
manifest, the ones a cohort map leaves out included, must have an out_dir of
three components or more, <class>/<root>/<well...> (amendment 2, F5): checked
before any archive is read, REFUSED `<culture>: its folder <out_dir> has no
<class>/<root>/ prefix` otherwise.

Printed: the cohort map's line(s) (`[real] cohort map ...`), the per-class
lines (each followed by its too-short lines, if any), the shard; last line
`REAL BANK OK` (exit 0) or `REFUSED: <reason>` (exit 1).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import socket
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_export as EXP               # noqa: E402
import bank_plan_config as CFG          # noqa: E402
import bank_row as ROW                  # noqa: E402
import cohort_manifest as CM            # noqa: E402
import cohort_map as CMAP               # noqa: E402
import real_source as RS                # noqa: E402

REAL_VERSION = 2                        # 2: the cohort map and rep_group (Block 9, 2026-11-17)


class RealRefused(RuntimeError):
    pass


def build_parser():
    p = argparse.ArgumentParser(description="B5.5: the Giulia real bank, one shard.")
    p.add_argument("--profile", required=True)
    p.add_argument("--out-root", default=None, help="default: the profile's real_bank_root")
    return p


def windows_of(K, W):
    """Window starts, back to back from 0 (D-056): 0, W, 2W, ... while they fit."""
    return [j * W for j in range((K - W) // W + 1)] if K >= W else []


def cohort_map_spec(cmap):
    """latent_spec["cohort_map"] (Block 9): None without a map."""
    if cmap is None:
        return None
    return {"sha256": cmap.sha256, "donor_labels": list(cmap.donor_labels),
            "rep_group_labels": list(cmap.rep_group_labels), "excluded": list(cmap.excluded),
            "n_wells_included": len(cmap.donor_of)}


def cohort_map_lines(cmap, n_wells):
    """The printed lines of the cohort map (Block 9); n_wells: the manifest's."""
    if cmap is None:
        return ["[real] cohort map: none (donor = rep_group = well)"]
    return (["[real] cohort map %s sha256 %s: %d of %d wells excluded; %d donors; %d pairing groups"
             % (cmap.path, cmap.sha256[:16], len(cmap.excluded), n_wells,
                len(cmap.donor_labels), len(cmap.rep_group_labels))]
            + ["[real]   excluded %s" % d for d in cmap.excluded])


def build_real(cfg, log=print):
    """The real bank's arrays, sidecar arguments and per-class counts."""
    if not os.path.isfile(cfg.cohort_manifest_path + ".sha256"):
        raise RealRefused("no .sha256 beside %s" % cfg.cohort_manifest_path)
    man = CM.read_manifest(cfg.cohort_manifest_path)
    if not man["_digest"].startswith(cfg.cohort_manifest_sha256_prefix):
        raise RealRefused("cohort manifest sha256 %s, the profile expects %s"
                          % (man["_digest"][:16], cfg.cohort_manifest_sha256_prefix))
    dt, _sigma, eps, _rec = CM.sim_preprocessing_from_manifest(man)
    if eps != cfg.n_e:
        raise RealRefused("electrodes_per_subset %d, the profile's n_e %d (D-043)" % (eps, cfg.n_e))
    cmap = None if cfg.cohort_map is None else CMAP.read_cohort_map(cfg.cohort_map, man)
    W = int(round(cfg.t_win_s / dt))
    classes = list(man["classes"])
    root = man["extract_root"]
    n_sub = int(man["n_subsets_per_well"])
    parts_of = []                       # every well of the manifest, the ones a map leaves out
    for w in man["wells"]:              # included, before any archive is read (amendment 2, F5)
        parts = w["out_dir"].replace(os.sep, "/").split("/")
        if len(parts) < 3:
            raise RealRefused("%s: its folder %s has no <class>/<root>/ prefix"
                              % (w["culture"], w["out_dir"]))
        parts_of.append(parts)
    roots = sorted({parts[1] for parts in parts_of})
    X, CLS, WELL, DON, GRP, BAT, SUB, WIN, rates = [], [], [], [], [], [], [], [], []
    short = dict((c, 0) for c in classes)   # traces with K < W, per class (amendment 2, F4)
    no_row = []                             # (class, culture, out_dir) of a well left with no row
    for wi, w in enumerate(man["wells"]):
        if cmap is not None and wi not in cmap.donor_of:
            continue                    # left out by the cohort map: no row (Block 9)
        donor = wi if cmap is None else cmap.donor_of[wi]
        group = donor if cmap is None else cmap.rep_group_of[wi]
        parts = parts_of[wi]
        if parts[0] not in classes:
            raise RealRefused("%s: its folder %r names no class of %r" % (w["culture"], parts[0], classes))
        cls = classes.index(parts[0])
        n_before = len(X)
        for s in range(n_sub):
            path = os.path.join(root, w["out_dir"], "trace_subregion_%02d.npz" % s)
            if not os.path.isfile(path):
                raise RealRefused("missing archive %s" % path)
            x, meta = RS.read_real_trace(path, expect_fs=1.0 / dt)
            K = x.size
            if w.get("T_rec") is not None and K != int(round(float(w["T_rec"]) / dt)):
                raise RealRefused("%s: %d samples, T_rec %r s implies %d"
                                  % (path, K, w["T_rec"], int(round(float(w["T_rec"]) / dt))))
            starts = windows_of(K, W)
            if not starts:
                short[parts[0]] += 1    # K < W: no row, counted and reported (SPEC sec. 4)
            for j, start in enumerate(starts):
                xw = np.ascontiguousarray(x[start:start + W], dtype=np.float32)
                X.append(xw)
                rates.append(ROW.mean_rate_hz(xw, dt))
                CLS.append(cls), WELL.append(wi), BAT.append(roots.index(parts[1]))
                DON.append(donor), GRP.append(group)
                SUB.append(s), WIN.append(j)
        if len(X) == n_before:
            no_row.append((parts[0], w["culture"], "/".join(parts)))
    if not X:
        raise RealRefused("no window: every trace is shorter than %d samples" % W)
    n = len(X)
    rates = np.asarray(rates)
    below = rates < cfg.floor_hz * (1.0 - cfg.floor_rtol)
    arrays = {"x": np.stack(X).astype(np.float32), "theta": np.zeros((n, 0), dtype=np.float32),
              "cls": np.asarray(CLS, dtype=np.int64), "nu": np.zeros((n, 5), dtype=np.float32),
              "realisation_id": np.asarray(WELL, dtype=np.uint64),
              "donor": np.asarray(DON, dtype=np.int64), "well": np.asarray(WELL, dtype=np.int64),
              "batch": np.asarray(BAT, dtype=np.int64), "subregion": np.asarray(SUB, dtype=np.int64),
              "window_idx": np.asarray(WIN, dtype=np.int64), "contaminated": np.zeros(n, dtype=bool),
              "rep_group": np.asarray(GRP, dtype=np.int64)}
    per_class = {c: {"windows": int(np.sum(arrays["cls"] == i)),
                     "below_floor": int(np.sum(below & (arrays["cls"] == i))),
                     "wells": int(len(set(np.asarray(WELL)[arrays["cls"] == i].tolist())))}
                 for i, c in enumerate(classes)}
    n_wells = len(man["wells"]) if cmap is None else len(cmap.donor_of)
    return arrays, {"W": W, "dt": dt, "classes": classes, "roots": roots, "per_class": per_class,
                    "manifest_sha256": man["_digest"], "n_wells": n_wells,
                    "n_wells_manifest": len(man["wells"]), "cohort_map": cmap,
                    "below_floor": int(np.sum(below)),
                    "too_short": {"traces": int(sum(short.values())),
                                  "wells_without_rows": [d for _c, _w, d in no_row]},
                    "too_short_per_class": short, "too_short_wells": no_row}


def too_short_lines(info, c):
    """The printed lines of class c's traces shorter than one window (amendment 2,
    F4): none when it has none."""
    n = info["too_short_per_class"][c]
    if n == 0:
        return []
    return (["[real] %s: %d trace(s) shorter than one window (K < W, W = %d samples): no row"
             % (c, n, info["W"])]
            + ["[real]   %s (%s): every trace shorter than one window, no row" % (culture, out_dir)
               for cls, culture, out_dir in info["too_short_wells"] if cls == c])


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = CFG.load_profile(args.profile)
        out_root = os.path.abspath(args.out_root or cfg.real_bank_root)
        lb, ln = EXP.stage1_modules()
        if "rep_group" not in getattr(lb, "OPTIONAL_ROW_ARRAYS", {}):
            raise RealRefused("%s predates the shard's optional rep_group array (SBI 88c32e5, "
                              "D-148 (4)): pull Simulation-Based-Inference" % lb.__file__)
        arrays, info = build_real(cfg)
        latent_spec = {"kind": "campaign_real", "classes": info["classes"],
                       "class_roots": info["roots"], "theta": "absent: no labels (zero columns)",
                       "windows": "W samples back to back from t = 0 (D-056)",
                       "n_e": cfg.n_e, "floor_hz_of_the_sim_bank": cfg.floor_hz,
                       "cohort_manifest_sha256": info["manifest_sha256"],
                       "cohort_map": cohort_map_spec(info["cohort_map"])}
        sidecar = lb.build_sidecar(
            param_names=[], bounds_theta=[], coord="unit_box", fs=1.0 / info["dt"],
            w_size=info["dt"], T_win=cfg.t_win_s, W=info["W"], scale_convention=EXP.SCALE_CONVENTION,
            latent_spec=latent_spec,
            nuisance_spec={"kind": "none", "components": list(ln.NU_COMPONENTS)},
            realisation_spec={"kind": "recording", "realisation_id": "the well's index"},
            gap_spec={"kind": "none"}, generator_sha256=info["manifest_sha256"],
            theta_withheld=False, arm="R",
            extra={"per_class": info["per_class"], "below_floor": info["below_floor"],
                   "n_wells": info["n_wells"], "real_version": REAL_VERSION,
                   "host": socket.gethostname(),
                   "built": _dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        path = os.path.join(out_root, "shard_0000.npz")
        if os.path.exists(path):
            raise RealRefused("%s exists: a written bank is never overwritten; give "
                              "--out-root another folder, or move that bank aside" % path)
        n = lb.write_shard(path, arrays, sidecar)
        for line in cohort_map_lines(info["cohort_map"], info["n_wells_manifest"]):
            print(line)
        for c, d in info["per_class"].items():
            print("[real] %-10s %3d wells, %4d windows, %3d below the floor"
                  % (c, d["wells"], d["windows"], d["below_floor"]))
            for line in too_short_lines(info, c):
                print(line)
        print("[real] wrote %s  rows %d  sha256 %s  contract %s"
              % (path, n, EXP.sha256_file(path)[:16], sidecar["contract_digest"][:16]))
        print("REAL BANK OK")
        return 0
    except (RealRefused, CFG.ConfigError, CM.ManifestError, CMAP.CohortMapError, KeyError,
            ValueError, OSError) as exc:
        print("REFUSED: %s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
