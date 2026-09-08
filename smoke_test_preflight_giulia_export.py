#!/usr/bin/env python3
"""
smoke_test_preflight_giulia_export.py

Builds a synthetic fixture that mimics the documented on-disk schema
(sim: job_args.json / manifest.json / topo_*/iter_*.npz;
 mea: mea_manifest.json / topo_*/mea_iter_*.npz) and asserts every quantity
preflight_giulia_export.py reports.  Needs no cluster, no torch, no checkpoint.

Run:
    python3 smoke_test_preflight_giulia_export.py
    SMOKE_VERBOSE=1 python3 smoke_test_preflight_giulia_export.py   # tracebacks

Expect: ALL 14 CHECKS PASSED
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

import preflight_giulia_export as P

VERBOSE = bool(os.environ.get("SMOKE_VERBOSE"))
PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("  PASS  %s" % name)
    except Exception as exc:
        FAILED.append((name, exc))
        print("  FAIL  %s : %s" % (name, exc))
        if VERBOSE:
            traceback.print_exc()


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------

def make_sim_unit(path, n_params=37, sweep_group="synapse_astro", mode="Full",
                  conn_rule="weibull", simtime=180.0, seed_master=1000,
                  n_topo=2, n_iter=3):
    os.makedirs(path, exist_ok=True)
    job = {
        "simtime": simtime,
        "mode": mode,
        "sweep_group": sweep_group,
        "conn_rule": conn_rule,
        "conn_prob_lo": 0.1, "conn_prob_hi": 0.6,
        "p0_conn_lo": None, "p0_conn_hi": None,      # present-but-null trap
        "d0_conn_lo": None, "d0_conn_hi": None,
        "beta_conn_lo": None, "beta_conn_hi": None,
        "_resolved_seed_master": seed_master,
        "_axis_declaration": {
            "swept_axes": ["O_N", "U_A", "w_e"],
            "consumed_axes": ["O_N", "U_A", "w_e"],
            "fixed_axes": ["DeltaT", "VT", "gL"],
            "inert_axes": [],
        },
    }
    with open(os.path.join(path, "job_args.json"), "w") as fh:
        json.dump(job, fh)
    with open(os.path.join(path, "manifest.json"), "w") as fh:
        json.dump({"manifest_version": 5, "sweep_group": sweep_group,
                   "active_indices": list(range(22))}, fh)
    for t in range(n_topo):
        tdir = os.path.join(path, "topo_%05d" % t)
        os.makedirs(tdir, exist_ok=True)
        for i in range(n_iter):
            np.savez_compressed(
                os.path.join(tdir, "iter_%05d.npz" % i),
                spk_N_t=np.array([0.1, 0.2], dtype=np.float32),
                spk_N_i=np.array([0, 1], dtype=np.int32),
                params=np.arange(n_params, dtype=np.float64),
                theta=np.arange(22, dtype=np.float64),
                p0_conn=np.float64(0.5), d0_conn=np.float64(80.0),
                beta_conn=np.float64(2.0),
            )


def make_mea_unit(path, n_e=4, n_det=400, simtime_npz=180.0, fs=10110.09,
                  n_topo=2, n_iter=3, n_side=2):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "mea_manifest.json"), "w") as fh:
        json.dump({"n_topos": n_topo, "total_iters": n_topo * n_iter,
                   "total_done": n_topo * n_iter}, fh)
    centers = np.array([[50.0 + 200.0 * (k % 2), 50.0 + 200.0 * (k // 2)]
                        for k in range(n_e)], dtype=np.float32)
    meta = {"probe": {"n_side": n_side, "pitch": 200.0, "edge": 26.59,
                      "n_sub": 4}, "noise": {"target_noise_uv": 5.0,
                                             "snr_ref": 15.0, "gamma": 0.5},
            "fs": fs}
    rng = np.random.default_rng(0)
    for t in range(n_topo):
        tdir = os.path.join(path, "topo_%05d" % t)
        os.makedirs(tdir, exist_ok=True)
        for i in range(n_iter):
            np.savez_compressed(
                os.path.join(tdir, "mea_iter_%05d.npz" % i),
                det_ch=rng.integers(0, n_e, size=n_det).astype(np.int32),
                det_t=np.sort(rng.uniform(0, simtime_npz, size=n_det)).astype(np.float32),
                det_amp=rng.uniform(30, 300, size=n_det).astype(np.float32),
                det_src_neuron=np.where(rng.random(n_det) < 0.1, -1, 3).astype(np.int32),
                sigma=np.full(n_e, 5.0, dtype=np.float32),
                electrode_centers=centers,
                params=np.arange(37, dtype=np.float64),
                theta=np.arange(22, dtype=np.float64),
                fs=np.float64(fs), simtime=np.float64(simtime_npz),
                meta_json=np.array(json.dumps(meta)),
            )


def build_fixture(root):
    sim = os.path.join(root, "sim")
    mea4 = os.path.join(root, "mea_out")
    mea1 = os.path.join(root, "mea_out_1electrode")

    # two campaigns, two tasks each; task0001 of v2 has NO mea output
    for camp, seed in (("campaign_cadex_hhgap_v1", 1000),
                       ("campaign_cadex_hhgap_v2", 2000)):
        for k in range(2):
            make_sim_unit(os.path.join(sim, camp, "sweep_intel_task%04d" % k),
                          seed_master=seed + k)
    # a seed collision: v5 task0000 replays v1 task0000's seed
    make_sim_unit(os.path.join(sim, "campaign_cadex_hhgap_v5",
                               "sweep_intel_task0000"), seed_master=1000)

    for camp in ("campaign_cadex_hhgap_v1", "campaign_cadex_hhgap_v2",
                 "campaign_cadex_hhgap_v5"):
        for k in range(2):
            if camp == "campaign_cadex_hhgap_v2" and k == 1:
                continue                      # deliberately missing MEA output
            if camp == "campaign_cadex_hhgap_v5" and k == 1:
                continue                      # no such sim unit either
            make_mea_unit(os.path.join(mea4, camp, "sweep_intel_task%04d" % k),
                          n_e=4, n_side=2)
            make_mea_unit(os.path.join(mea1, camp, "sweep_intel_task%04d" % k),
                          n_e=1, n_side=1, n_det=100)

    # an orphan MEA unit with no sim counterpart
    make_mea_unit(os.path.join(mea4, "campaign_ghost", "sweep_intel_task0000"),
                  n_e=4, n_side=2)

    reg = os.path.join(root, "HPC_single_run.py")
    with open(reg, "w") as fh:
        fh.write("PARAM_NAMES = [%s]\n"
                 % ", ".join("'p%d'" % i for i in range(37)))

    ext = os.path.join(root, "Sbi-extractor")
    os.makedirs(ext, exist_ok=True)
    with open(os.path.join(ext, "preflight_label_axes.py"), "w") as fh:
        fh.write("import argparse\n"
                 "p = argparse.ArgumentParser()\n"
                 "p.add_argument('--sim_main')\n"
                 "p.add_argument('--require-conn-rule')\n"
                 "p.add_argument('--exclude')\n"
                 "p.add_argument('--out')\n")
    with open(os.path.join(ext, "env.sh"), "w") as fh:
        fh.write('ARTIFACTS_DIR="${ARTIFACTS_DIR:-./artifacts}"\n'
                 'DSN_MAIN_DIR="${DSN_MAIN_DIR:-$HOME/dsn_main}"\n')
    return sim, mea4, mea1, reg, ext


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def main():
    root = tempfile.mkdtemp(prefix="preflight_smoke_")
    try:
        sim, mea4, mea1, reg, ext = build_fixture(root)
        res = P.build_result([mea4, mea1], sim, ext, reg, iters_per_unit=2)

        def T1():
            u = P.find_units(sim, P.SIM_ITER_RE)
            assert len(u) == 5, len(u)
            assert all(os.sep in r for r in u), sorted(u)
        check("T1 sim unit discovery finds 5 campaign/task units", T1)

        def T2():
            u = P.find_units(mea4, P.MEA_ITER_RE)
            assert len(u) == 5, sorted(u)     # 4 real + 1 ghost
        check("T2 mea unit discovery finds 5 units incl. the orphan", T2)

        def T3():
            blk = res["mea_roots"][mea4]
            assert len(blk["paired"]) == 4, blk["paired"]
            assert len(blk["mea_only"]) == 1, blk["mea_only"]
            assert "campaign_ghost" in blk["mea_only"][0]
        check("T3 pairing reports the orphan MEA unit as unpaired", T3)

        def T4():
            missing = [r for r in res["sim_only"]]
            assert any("v2" in r and "task0001" in r for r in missing), missing
        check("T4 sim unit with no MEA output is reported", T4)

        def T5():
            assert len(res["label_signatures"]) == 1, res["label_signatures"]
            sig = json.loads(list(res["label_signatures"])[0])
            mode, group, rule, n_active = sig[0], sig[1], sig[2], sig[3]
            assert (mode, group, rule, n_active) == ("Full", "synapse_astro",
                                                     "weibull", 22), sig
            assert sig[4] == ["O_N", "U_A", "w_e"], sig[4]
        check("T5 label signature is single-valued and carries the swept axes", T5)

        def T6():
            odd = os.path.join(sim, "campaign_cadex_hhgap_v9",
                               "sweep_intel_task0000")
            make_sim_unit(odd, sweep_group="neuron_synapse", mode="Neuronal")
            make_mea_unit(os.path.join(mea4, "campaign_cadex_hhgap_v9",
                                       "sweep_intel_task0000"), n_e=4)
            res2 = P.build_result([mea4], sim, None, None, iters_per_unit=1)
            assert len(res2["label_signatures"]) == 2, res2["label_signatures"]
            shutil.rmtree(os.path.join(sim, "campaign_cadex_hhgap_v9"))
            shutil.rmtree(os.path.join(mea4, "campaign_cadex_hhgap_v9"))
        check("T6 a differing sweep_group splits into two signatures", T6)

        def T7():
            assert dict(res["params_len"]) == {37: 8}, dict(res["params_len"])
            assert res["registry"]["n"] == 37, res["registry"]
        check("T7 recorded len(params) and registry PARAM_NAMES agree at 37", T7)

        def T8():
            geo4 = list(res["mea_roots"][mea4]["geometry"].values())[0]
            geo1 = list(res["mea_roots"][mea1]["geometry"].values())[0]
            assert geo4["n_e"] == 4, geo4["n_e"]
            assert geo1["n_e"] == 1, geo1["n_e"]
        check("T8 n_e read from electrode_centers is 4 and 1", T8)

        def T9():
            geo = list(res["mea_roots"][mea4]["geometry"].values())[0]
            assert geo["meta_probe"]["n_side"] == 2, geo["meta_probe"]
            assert abs(geo["meta_probe"]["edge"] - 26.59) < 1e-9
            assert geo["meta_probe"]["target_noise_uv"] == 5.0
        check("T9 probe config is recovered from nested meta_json", T9)

        def T10():
            geo = list(res["mea_roots"][mea4]["geometry"].values())[0]
            assert sum(geo["det_per_channel"]) == geo["n_det"] == 400, geo
            assert len(geo["det_per_channel"]) == 4
            expect = round(400 / (180.0 * 4), 4)
            assert abs(geo["rate_hz_per_electrode"] - expect) < 1e-9, geo
        check("T10 per-channel counts sum to n_det; rate is per electrode", T10)

        def T11():
            geo = list(res["mea_roots"][mea4]["geometry"].values())[0]
            assert geo["simtime_job_args"] == 180.0, geo
            assert geo["simtime"] == 180.0, geo
            # now make the npz disagree, as a quiet simulation would
            u = os.path.join(mea4, "campaign_cadex_hhgap_v1",
                             "sweep_intel_task0000")
            shutil.rmtree(u)
            make_mea_unit(u, n_e=4, simtime_npz=3.0)
            r = P.build_result([mea4], sim, None, None, iters_per_unit=1)
            g = r["mea_roots"][mea4]["geometry"][
                os.path.join("campaign_cadex_hhgap_v1", "sweep_intel_task0000")]
            assert g["simtime"] == 3.0 and g["simtime_job_args"] == 180.0, g
        check("T11 the inferred-simtime trap is detectable (npz 3 s vs args 180 s)", T11)

        def T12():
            sm = res["seed_masters"]
            assert sm["n_units"] == 4, sm
            assert sm["n_distinct"] == 3, sm
            assert 1000 in sm["duplicates"], sm
        check("T12 the replayed seed_master is flagged as a duplicate", T12)

        def T13():
            ja_null = P.read_sim_unit(
                os.path.join(sim, "campaign_cadex_hhgap_v2",
                             "sweep_intel_task0000"), 1)
            assert "p0_conn_lo" not in ja_null["job_args"], ja_null["job_args"]
            assert "p0_conn_lo" in ja_null["job_args_null"], ja_null
        check("T13 present-but-null kernel bounds are not read as swept", T13)

        def T14():
            ep = res["entry_points"]["preflight_label_axes.py"]
            assert "--require-conn-rule" in ep["flags"], ep
            assert "ARTIFACTS_DIR" in res["entry_points"]["env.sh"]["env_vars"]
            assert "error" in res["entry_points"]["example_export.py"]
        check("T14 entry-point flags/vars are scanned; missing files reported", T14)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES", len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
