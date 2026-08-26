#!/usr/bin/env python3
"""
smoke_test_real_cohort_filter.py -- checks for real_cohort_filter.py.

Builds a synthetic real-cohort export (parquet + sidecar, same schema as
real_source.py writes: z_*, zraw_*, culture, condition, subregion, name,
T_rec_s, fs_ifr) with a KNOWN set of injected far-from-bulk windows, then
runs the real CLI in a subprocess and asserts on the outputs.

Ground truth: 2 conditions x 3 cultures x 40 windows = 240 rows. In each
condition a known handful of windows is displaced far from the bulk. A
correct filter removes those and no others, and the medoid must come from
the bulk.

Checks
------
  F1  the CLI exits 0 and writes all five artifacts
  F2  the injected windows are removed and the schema is preserved
      (identical columns, identical dtypes, fewer rows)
  F3  --max_frac caps the removal even when many windows are flagged
  F4  the medoid file has exactly one row per condition, each row is a real
      row of the CLEANED pool, and its distance to the bulk centre is below
      the median (i.e. it is central, not peripheral)
  F5  the sidecar contract block survives into both outputs, with the
      provenance block appended
  F6  the replicate guard REFUSES when one culture would lose too much,
      and --allow_culture_loss downgrades it to a warning
  F7  --dry_run writes nothing
  F8  negative paths: missing sidecar, bad --space, bad --max_frac,
      a culture straddling two conditions

Only numpy/pandas/pyarrow. Seconds on a login node. ASCII-only by policy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "real_cohort_filter.py")

E = 10
CULTURES_PER_COND = 3
WIN_PER_CULTURE = 40
N_INJECT = 3          # far-from-bulk windows injected per condition
SEED = 7


def build_cohort(root, straddle=False, name="sbi_real_cohort"):
    """Synthetic export. Returns (parquet_path, injected_names)."""
    import pandas as pd

    rng = np.random.default_rng(SEED)
    rows, injected = [], []
    for ci, cond in enumerate(("0", "1")):
        centre = rng.normal(size=E)
        centre /= np.linalg.norm(centre)
        for k in range(CULTURES_PER_COND):
            culture = "DATA_C_Batch%d__ptrain_%s%d" % (ci + 1, "ABC"[k], k)
            for w in range(WIN_PER_CULTURE):
                z = centre + 0.05 * rng.normal(size=E)
                rows.append({
                    "culture": culture, "condition": cond,
                    "subregion": w % 9,
                    "name": "%s_w%03d" % (culture, w),
                    "T_rec_s": 180.0, "fs_ifr": 100.0,
                    "_z": z / np.linalg.norm(z)})
        # inject: displaced windows, spread across cultures so no single
        # culture is wiped out by their removal
        for j in range(N_INJECT):
            culture = "DATA_C_Batch%d__ptrain_%s%d" % (ci + 1, "ABC"[j % 3],
                                                       j % 3)
            far = centre + 3.0 * rng.normal(size=E)
            nm = "%s_INJECT%d" % (culture, j)
            injected.append(nm)
            rows.append({
                "culture": culture, "condition": cond, "subregion": 0,
                "name": nm, "T_rec_s": 180.0, "fs_ifr": 100.0,
                "_z": far / np.linalg.norm(far)})

    if straddle:
        rows[0]["condition"] = "1" if rows[0]["condition"] == "0" else "0"

    Z = np.stack([r.pop("_z") for r in rows]).astype(np.float32)
    df = pd.DataFrame(rows)
    for i in range(E):
        df["z_%03d" % i] = Z[:, i]
    for i in range(E):
        df["zraw_%03d" % i] = Z[:, i] * rng.uniform(0.5, 2.0, size=len(df))
    parquet = os.path.join(root, name + ".parquet")
    df.to_parquet(parquet, index=False)

    sidecar = {
        "embedding": {"dsn_checkpoint_sha256": "deadbeef" * 8,
                      "embedding_dim": E},
        "param_names": ["Sigma", "gbarA"],
        "coord": ["linear", "ln"],
        "bounds_theta": [[3.0, 15.0], [-4.6, 2.3]],
        "note": "synthetic fixture",
    }
    with open(os.path.join(root, name + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(sidecar, fh)
    return parquet, injected


def run_cli(extra, expect_fail=False):
    r = subprocess.run([sys.executable, SCRIPT] + extra,
                       capture_output=True, text=True)
    if expect_fail:
        assert r.returncode != 0, \
            "expected failure, got success:\n%s" % r.stdout[-1500:]
        return r
    assert r.returncode == 0, "CLI failed (%d):\n%s\n%s" \
        % (r.returncode, r.stdout[-2500:], r.stderr[-2500:])
    return r


def main():
    import pandas as pd

    root = tempfile.mkdtemp(prefix="smoke_rcf_")
    try:
        parquet, injected = build_cohort(root)
        src = pd.read_parquet(parquet)
        out = os.path.join(root, "filtered")

        # ---- F1 --------------------------------------------------------
        run_cli(["--real", parquet, "--out", out])
        paths = {s: out + s for s in
                 ("_clean.parquet", "_clean.json", "_medoid.parquet",
                  "_medoid.json", "_filter_report.json")}
        for s, p in paths.items():
            assert os.path.isfile(p), "F1: missing %s" % s
        print("F1 PASS: all five artifacts written")

        # ---- F2 --------------------------------------------------------
        clean = pd.read_parquet(paths["_clean.parquet"])
        assert list(clean.columns) == list(src.columns), \
            "F2: column set/order changed"
        assert clean.dtypes.equals(src.dtypes[clean.columns]), \
            "F2: dtypes changed"
        assert len(clean) < len(src), "F2: nothing was removed"
        kept = set(clean["name"].astype(str))
        still = [n for n in injected if n in kept]
        assert not still, "F2: injected windows survived: %s" % still[:5]
        collateral = (len(src) - len(clean)) - len(injected)
        assert collateral <= 2, \
            "F2: %d non-injected windows also removed" % collateral
        print("F2 PASS: %d injected removed, %d collateral, schema preserved"
              % (len(injected), collateral))

        # ---- F3 --------------------------------------------------------
        out3 = os.path.join(root, "capped")
        run_cli(["--real", parquet, "--out", out3, "--k_mad", "0.01",
                 "--max_frac", "0.02"])
        c3 = pd.read_parquet(out3 + "_clean.parquet")
        rep3 = json.load(open(out3 + "_filter_report.json", encoding="utf-8"))
        for cond, info in rep3["per_condition"].items():
            assert info["n_removed"] <= info["cap"], \
                "F3: condition %s removed %d > cap %d" \
                % (cond, info["n_removed"], info["cap"])
            assert info["cap_applied"], "F3: cap should have applied"
        assert len(src) - len(c3) <= int(0.02 * len(src)) + 2, \
            "F3: global removal exceeded the cap"
        print("F3 PASS: --max_frac caps removal (%d removed)"
              % (len(src) - len(c3)))

        # ---- F4 --------------------------------------------------------
        med = pd.read_parquet(paths["_medoid.parquet"])
        assert len(med) == 2, "F4: expected 2 medoid rows, got %d" % len(med)
        assert sorted(med["condition"].astype(str)) == ["0", "1"], \
            "F4: one medoid per condition expected"
        zc = [c for c in clean.columns if c.startswith("z_")]
        for _, mrow in med.iterrows():
            cond = str(mrow["condition"])
            nm = str(mrow["name"])
            pool = clean[clean["condition"].astype(str) == cond]
            assert nm in set(pool["name"].astype(str)), \
                "F4: medoid %s is not in the cleaned pool" % nm
            P = pool[zc].to_numpy(float)
            mz = mrow[zc].to_numpy(float)
            centre = P.mean(axis=0)
            d_med = float(np.linalg.norm(mz - centre))
            d_all = np.linalg.norm(P - centre[None, :], axis=1)
            assert d_med <= np.median(d_all), \
                "F4: medoid is peripheral (d=%.4f vs median %.4f)" \
                % (d_med, float(np.median(d_all)))
        print("F4 PASS: one central medoid per condition, drawn from the "
              "cleaned pool")

        # ---- F5 --------------------------------------------------------
        src_side = json.load(open(os.path.splitext(parquet)[0] + ".json",
                                  encoding="utf-8"))
        for key in ("_clean.json", "_medoid.json"):
            side = json.load(open(paths[key], encoding="utf-8"))
            for k in ("embedding", "param_names", "coord", "bounds_theta"):
                assert side[k] == src_side[k], \
                    "F5: contract key %r altered in %s" % (k, key)
            assert "real_cohort_filter" in side, \
                "F5: provenance block missing from %s" % key
            assert side["real_cohort_filter"]["n_removed"] > 0
        print("F5 PASS: contract block intact, provenance appended")

        # ---- F6 --------------------------------------------------------
        # k_mad tiny + a generous cap flags a lot; a strict per-culture
        # ceiling must then refuse.
        out6 = os.path.join(root, "guard")
        strict = ["--real", parquet, "--out", out6, "--k_mad", "0.01",
                  "--max_frac", "0.5", "--max_culture_frac", "0.05"]
        r = run_cli(strict, expect_fail=True)
        assert "REFUSING" in (r.stdout + r.stderr), \
            "F6: refused but without the expected message"
        assert not os.path.isfile(out6 + "_clean.parquet"), \
            "F6: wrote output despite refusing"
        run_cli(strict + ["--allow_culture_loss"])
        assert os.path.isfile(out6 + "_clean.parquet"), \
            "F6: --allow_culture_loss did not proceed"
        print("F6 PASS: replicate guard refuses, override proceeds")

        # ---- F7 --------------------------------------------------------
        out7 = os.path.join(root, "dry")
        r = run_cli(["--real", parquet, "--out", out7, "--dry_run"])
        assert not any(os.path.isfile(out7 + s) for s in paths), \
            "F7: --dry_run wrote files"
        assert "dry_run" in r.stdout or "nothing written" in r.stdout
        print("F7 PASS: --dry_run writes nothing")

        # ---- F8 --------------------------------------------------------
        lone = os.path.join(root, "lonely.parquet")
        shutil.copy(parquet, lone)                      # no sidecar beside it
        run_cli(["--real", lone, "--out", os.path.join(root, "n1")],
                expect_fail=True)
        run_cli(["--real", parquet, "--out", os.path.join(root, "n2"),
                 "--max_frac", "1.5"], expect_fail=True)
        run_cli(["--real", parquet, "--out", os.path.join(root, "n3"),
                 "--space", "zraw", "--class_col", "nope"], expect_fail=True)
        bad_root = tempfile.mkdtemp(prefix="smoke_rcf_bad_")
        try:
            bad_parquet, _ = build_cohort(bad_root, straddle=True)
            run_cli(["--real", bad_parquet,
                     "--out", os.path.join(bad_root, "n4")], expect_fail=True)
        finally:
            shutil.rmtree(bad_root, ignore_errors=True)
        print("F8 PASS: missing sidecar, bad max_frac, bad column and a "
              "straddling culture all refuse")

        print("ALL CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
