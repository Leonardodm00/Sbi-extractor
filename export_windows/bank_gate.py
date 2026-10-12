#!/usr/bin/env python3
"""Block 7.3 -- the Giulia simulated bank's gate and record (B5.4;
specs/SPEC.md, Block 7.3; sec. 2.6).

    python3 export_windows/bank_gate.py --plan export_windows/giulia/bank_plan.json [--workers N]

Run after the array (run_bank_gate.pbs, depend=afterok, which is ordering only
on davinci: D-006). Checks every task of the plan, and only then writes
<bank_root>/BANK_RECORD.json with its .sha256:

  - every task has a marker of THIS plan (its digest), listing exactly the
    plan's shards for that task, with read = the plan's simulations and
    kept + dropped = read;
  - every written shard and sidecar has the sha256 its marker recorded
    (recomputed here), and every sidecar's contract digest is the one the
    plan implies (one contract for the whole bank, D-074);
  - inside each shard: rows = kept; well = the task's index; batch = its
    campaign version; donor among the task's planned donors; subregion 0,
    cls -1; no (donor, window_idx) pair twice within the task (the small
    arrays only; x is not decompressed);
  - no shard file in bank_root that the plan does not name, and none for a
    chunk whose marker says it keeps no row.

Printed last line: `wrote <bank_root>/BANK_RECORD.json  sha256 <16 hex>` (exit 0)
or `REFUSED: <n> problem(s), nothing written` (exit 1), every problem listed.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import zipfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_export as EXP               # noqa: E402

GATE_VERSION = 1
SHARD_RE = re.compile(r"^shard_(\d{4,})\.npz$")
SMALL = ("well", "batch", "donor", "window_idx", "subregion", "cls")


def build_parser():
    p = argparse.ArgumentParser(description="Gate and record of the Giulia simulated bank.")
    p.add_argument("--plan", required=True)
    p.add_argument("--bank-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    return p


def _hash(path):
    return path, EXP.sha256_file(path)


def check_task(plan, index, bank_root, hashes, digest_expected):
    """Problems of one task (a list of strings) and its marker (or None)."""
    t = plan["tasks"][index]
    path = EXP.marker_path(bank_root, index)
    if not os.path.isfile(path):
        return ["%s: no marker %s (the task did not finish)" % (t["name"], path)], None
    try:
        with open(path, "r", encoding="ascii") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as exc:
        return ["%s: its marker %s cannot be read (%s)" % (t["name"], path, exc)], None
    bad = []
    if m.get("plan_sha256") != plan["_sha256"]:
        return ["%s: its marker belongs to another plan" % t["name"]], None
    want = [(c["shard"], c["start"], c["stop"]) for c in t["shards"]]
    got = [(s["shard"], s["start"], s["stop"]) for s in m["shards"]]
    if want != got:
        bad.append("%s: shards %r, the plan's %r" % (t["name"], got, want))
    tot = m["totals"]
    if tot["read"] != t["n_iters"] or tot["kept"] + tot["dropped_floor"] != tot["read"]:
        bad.append("%s: read %d kept %d dropped %d, the plan has %d simulations"
                   % (t["name"], tot["read"], tot["kept"], tot["dropped_floor"], t["n_iters"]))
    donors = set(t["donors"].values())
    seen = set()
    for s in m["shards"]:
        if s["file"] is None:
            if s["kept"] != 0:
                bad.append("%s shard %d: kept %d but no file" % (t["name"], s["shard"], s["kept"]))
            if os.path.exists(os.path.join(bank_root, "shard_%04d.npz" % s["shard"])):
                bad.append("%s: shard_%04d.npz exists, but its marker says the chunk keeps no "
                           "row (a file from an earlier run)" % (t["name"], s["shard"]))
            continue
        intact = True
        for f, h in ((s["file"], s["sha256"]), (s["json"], s["json_sha256"])):
            p = os.path.join(bank_root, f)
            if hashes.get(p) != h:
                bad.append("%s: %s is missing or changed since its marker" % (t["name"], f))
                intact = False
        if s.get("contract_digest") != digest_expected:
            bad.append("%s: %s has contract %s, the plan implies %s"
                       % (t["name"], s["file"], str(s.get("contract_digest"))[:16],
                          digest_expected[:16]))
        if intact:              # the bytes its marker recorded: read the small arrays
            p = os.path.join(bank_root, s["file"])
            try:
                with np.load(p, allow_pickle=False) as z:
                    arr = {k: np.asarray(z[k]) for k in SMALL}
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error) as exc:
                bad.append("%s: %s cannot be read (%s)" % (t["name"], s["file"], exc))
                continue
            n = arr["well"].shape[0]
            if n != s["kept"]:
                bad.append("%s: %s holds %d rows, its marker %d" % (t["name"], s["file"], n, s["kept"]))
            if np.any(arr["well"] != t["index"]) or np.any(arr["batch"] != t["batch"]):
                bad.append("%s: %s has well / batch other than %d / %d"
                           % (t["name"], s["file"], t["index"], t["batch"]))
            if not set(np.unique(arr["donor"]).tolist()) <= donors:
                bad.append("%s: %s has donors outside the task's" % (t["name"], s["file"]))
            if np.any(arr["subregion"] != 0) or np.any(arr["cls"] != -1):
                bad.append("%s: %s has subregion != 0 or cls != -1" % (t["name"], s["file"]))
            if np.any(arr["window_idx"] < 0):
                bad.append("%s: %s has a negative window_idx" % (t["name"], s["file"]))
            keys = set(zip(arr["donor"].tolist(), arr["window_idx"].tolist()))
            if len(keys) != n or keys & seen:
                bad.append("%s: %s repeats a (donor, window_idx) row" % (t["name"], s["file"]))
            seen |= keys
    return bad, m


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = EXP.load_plan(args.plan)
    except (EXP.ExportError, OSError, ValueError) as exc:
        print("REFUSED: %s" % exc)
        return 1
    bank_root = os.path.abspath(args.bank_root or plan["profile"]["bank_root"])
    print("[gate] plan %s  sha256 %s; bank %s" % (plan["_path"], plan["_sha256"][:16], bank_root))
    lb, ln = EXP.stage1_modules()
    kw = EXP.contract_kwargs(plan, ln.NU_COMPONENTS)
    digest_expected = lb.build_sidecar(**kw)["contract_digest"]
    planned = {"shard_%04d.npz" % c["shard"] for t in plan["tasks"] for c in t["shards"]}
    on_disk = sorted(os.path.basename(p) for p in glob.glob(os.path.join(bank_root, "shard_*.npz")))
    files = [os.path.join(bank_root, f) for f in on_disk]
    files += [p[:-4] + ".json" for p in files if os.path.isfile(p[:-4] + ".json")]
    with mp.Pool(max(1, int(args.workers))) as pool:
        hashes = dict(pool.map(_hash, files))
    problems, markers = [], []
    stray = [f for f in on_disk if f not in planned]
    if stray:
        problems.append("shard file(s) the plan does not name: %r" % stray[:10])
    for idx in range(len(plan["tasks"])):
        bad, m = check_task(plan, idx, bank_root, hashes, digest_expected)
        problems += bad
        if m is not None:
            markers.append(m)
    if problems:
        for p in problems:
            print("[gate] FAIL %s" % p)
        print("REFUSED: %d problem(s), nothing written" % len(problems))
        return 1
    read = sum(m["totals"]["read"] for m in markers)
    kept = sum(m["totals"]["kept"] for m in markers)
    shards = [s for m in markers for s in m["shards"] if s["file"]]
    est = plan["sample"]["kept_share"]
    record = {
        "record": "giulia_window_bank", "record_version": GATE_VERSION,
        "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "plan": plan["_path"], "plan_sha256": plan["_sha256"], "inputs": plan["inputs"],
        "contract_digest": digest_expected, "bank_root": bank_root,
        "label": {k: plan["label"][k] for k in ("param_names", "coord", "bounds_declared",
                                                "on_axis", "on_column")},
        "window": plan["window"],
        "counts": {"tasks": len(markers), "shards": len(shards), "rows_read": read,
                   "rows_kept": kept, "rows_dropped_floor": read - kept,
                   "kept_share": kept / float(read) if read else None,
                   "kept_share_estimated": est["p_hat"], "kept_share_estimated_ci95": est["ci95"],
                   "n_donors": plan["n_donors"],
                   "bytes": int(sum(os.path.getsize(os.path.join(bank_root, s["file"]))
                                    for s in shards))},
        "tasks": [{"index": m["task_index"], "task": m["task"], "totals": m["totals"],
                   "host": m["host"], "finished": m["finished"],
                   "shards": [{k: s[k] for k in ("shard", "file", "sha256", "read", "kept")}
                              for s in m["shards"]]} for m in markers],
        "environment": {"python": sorted({m["python"] for m in markers}),
                        "numpy": sorted({m["numpy"] for m in markers}),
                        "gate_python": platform.python_version()},
        "gate": {"script": os.path.basename(__file__), "version": GATE_VERSION}}
    blob = json.dumps(record, indent=1, sort_keys=True) + "\n"
    path = os.path.join(bank_root, "BANK_RECORD.json")
    with open(path + ".tmp", "w", encoding="ascii") as fh:
        fh.write(blob)
    os.replace(path + ".tmp", path)
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()
    with open(path + ".sha256", "w", encoding="ascii") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(path)))
    print("[gate] %d tasks, %d shards, %d rows kept of %d read (%.4f; estimated %.4f, "
          "95%% CI %.4f-%.4f), %.2f GB" % (len(markers), len(shards), kept, read,
                                           record["counts"]["kept_share"] or 0.0, est["p_hat"],
                                           est["ci95"][0], est["ci95"][1],
                                           record["counts"]["bytes"] / 1e9))
    print("[gate] contract digest %s" % digest_expected[:16])
    print("wrote %s  sha256 %s" % (path, digest[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
