#!/usr/bin/env python3
"""
smoke_test_dataset_profile.py

Fixtures for all four dataset kinds, then one check per behaviour that
dataset_profile.py is relied on for: detection, field extraction, every
internal-consistency warning, and the parity diff in both verdicts.

Run:
    python3 smoke_test_dataset_profile.py
    SMOKE_VERBOSE=1 python3 smoke_test_dataset_profile.py

Expect: ALL 19 CHECKS PASSED
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback

import numpy as np

import dataset_profile as D

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

VERBOSE = bool(os.environ.get("SMOKE_VERBOSE"))
PASSED, FAILED = [], []


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
# fixtures
# --------------------------------------------------------------------------

def make_sim_unit(path, n_params=37, sweep_group="synapse_astro", mode="Full",
                  conn_rule="weibull", simtime=180.0, seed_master=1000,
                  n_topo=2, n_iter=2, swept=("O_N", "U_A", "w_e")):
    os.makedirs(path, exist_ok=True)
    job = {"simtime": simtime, "mode": mode, "sweep_group": sweep_group,
           "conn_rule": conn_rule,
           "conn_prob_lo": 0.1, "conn_prob_hi": 0.6,
           "p0_conn_lo": None, "p0_conn_hi": None,
           "d0_conn_lo": None, "d0_conn_hi": None,
           "beta_conn_lo": None, "beta_conn_hi": None,
           "_resolved_seed_master": seed_master,
           "_axis_declaration": {"swept_axes": list(swept),
                                 "consumed_axes": list(swept),
                                 "fixed_axes": ["DeltaT", "VT", "gL"],
                                 "inert_axes": []}}
    json.dump(job, open(os.path.join(path, "job_args.json"), "w"))
    json.dump({"manifest_version": 5, "sweep_group": sweep_group,
               "active_indices": list(range(22))},
              open(os.path.join(path, "manifest.json"), "w"))
    for t in range(n_topo):
        tdir = os.path.join(path, "topo_%05d" % t)
        os.makedirs(tdir, exist_ok=True)
        for i in range(n_iter):
            np.savez_compressed(os.path.join(tdir, "iter_%05d.npz" % i),
                                spk_N_t=np.array([0.1], dtype=np.float32),
                                spk_N_i=np.array([0], dtype=np.int32),
                                params=np.arange(n_params, dtype=np.float64),
                                theta=np.arange(22, dtype=np.float64),
                                p0_conn=np.float64(0.5),
                                d0_conn=np.float64(80.0),
                                beta_conn=np.float64(2.0))


def make_mea_unit(path, n_e=4, n_det=400, simtime_npz=180.0, fs=10110.09,
                  n_topo=2, n_iter=2, n_side=2):
    os.makedirs(path, exist_ok=True)
    json.dump({"n_topos": n_topo, "total_iters": n_topo * n_iter,
               "total_done": n_topo * n_iter},
              open(os.path.join(path, "mea_manifest.json"), "w"))
    centers = np.array([[50.0 + 200.0 * (k % 2), 50.0 + 200.0 * (k // 2)]
                        for k in range(n_e)], dtype=np.float32)
    meta = {"probe": {"n_side": n_side, "pitch": 200.0, "edge": 26.59,
                      "n_sub": 4},
            "noise": {"target_noise_uv": 5.0, "snr_ref": 15.0, "gamma": 0.5}}
    rng = np.random.default_rng(0)
    for t in range(n_topo):
        tdir = os.path.join(path, "topo_%05d" % t)
        os.makedirs(tdir, exist_ok=True)
        for i in range(n_iter):
            np.savez_compressed(
                os.path.join(tdir, "mea_iter_%05d.npz" % i),
                det_ch=rng.integers(0, n_e, size=n_det).astype(np.int32),
                det_t=np.sort(rng.uniform(0, simtime_npz,
                                          size=n_det)).astype(np.float32),
                det_amp=rng.uniform(30, 300, size=n_det).astype(np.float32),
                det_src_neuron=np.where(rng.random(n_det) < 0.1,
                                        -1, 3).astype(np.int32),
                sigma=np.full(n_e, 5.0, dtype=np.float32),
                electrode_centers=centers,
                params=np.arange(37, dtype=np.float64),
                theta=np.arange(22, dtype=np.float64),
                fs=np.float64(fs), simtime=np.float64(simtime_npz),
                meta_json=np.array(json.dumps(meta)))


def make_real_archive(root, n_files=3, fs_ifr=100.0, T_rec=1200.0, n_e=None,
                      full_meta=None, meta_in="npz"):
    """full_meta: None (legacy archive, preprocessing unrecorded) or a dict
    as the PATCHED run_channel_subset_extraction.py writes it, e.g.
    {"w_size": 0.01, "gaussian_window": 0.02, "electrodes_per_subset": 9,
     "n_subsets": 9, "mfr_threshold": 0.1, "fs_raw": 10110.09}.
    meta_in: "npz" embeds it in the archive, "json" writes traces_meta.json
    beside it, "both" does both."""
    for k in range(n_files):
        d = os.path.join(root, "control", "batch1", "ptrain_A%d" % k)
        os.makedirs(d, exist_ok=True)
        payload = {"ifr_trace": np.zeros(int(fs_ifr * T_rec), dtype=np.float32),
                   "fs_ifr": np.float64(fs_ifr),
                   "T_rec": np.float64(T_rec),
                   "culture_id": np.array("A%d" % k)}
        if n_e is not None:
            payload["electrodes_per_subset"] = np.int64(n_e)
        if full_meta and meta_in in ("npz", "both"):
            for kk, vv in full_meta.items():
                payload[kk] = np.asarray(vv)
        np.savez_compressed(os.path.join(d, "trace_subregion_00.npz"),
                            **payload)
        if full_meta and meta_in in ("json", "both"):
            json.dump(dict(full_meta, fs_ifr=fs_ifr, T_rec=T_rec),
                      open(os.path.join(d, "traces_meta.json"), "w"))


def make_export_shard(stem, p=26, E=10, sha="9d9e0a7f" + "0" * 56, n_e=9,
                      fs_ifr=100.0, sigma_sm=0.05, rows=8, parquet="valid",
                      unit_norm=True, key_style="producer"):
    """parquet: 'valid' writes a real one (needs pyarrow), 'corrupt' writes a
    4-byte stub, 'none' writes no parquet at all.

    The first version of this fixture always wrote the 4-byte stub.  With no
    pyarrow in the authoring sandbox the reader's ImportError branch swallowed
    it and the test passed without ever exercising the read path -- which is
    how a crash shipped.  'valid' now exercises it wherever pyarrow exists.
    """
    names = ["a%02d" % j for j in range(p)]
    # key_style="producer": the keys export_embeddings.py ACTUALLY writes,
    # i.e. dsn_frozen.FrozenDSN.observable_block() merged with the
    # electrode-geometry fields from record_resolved_n_e(). "legacy": the
    # keys this fixture used to write, which matched the profiler's reader
    # and nothing else -- kept so the fallback path stays exercised.
    # [CORRECTION 2026-09-11] the fixture was written to match the reader,
    # not the producer, and so never caught the reader profiling every real
    # export shard as fs_ifr = sigma_sm = window_s = None.
    if key_style == "producer":
        observable = {"ifr_dt_s": 1.0 / fs_ifr,
                      "ifr_smooth_sigma_s": sigma_sm,
                      "ifr_smooth_sigma_bins": sigma_sm * fs_ifr,
                      "fs_ifr_hz": fs_ifr,
                      "window_s": 180.0,
                      "window_length_samples": int(round(180.0 * fs_ifr)),
                      "ifr_units": "spikes per bin per electrode",
                      "T": 180.0}
        if n_e is not None:
            observable["n_electrodes"] = n_e
            observable["n_electrodes_source"] = "electrode_centers"
        embedding = {"embedding_dim": E, "dsn_checkpoint_sha256": sha}
    else:
        observable = {"n_electrodes": n_e, "fs_ifr": fs_ifr,
                      "sigma_sm": sigma_sm, "T": 180.0}
        embedding = {"embedding_dim": E, "dsn_checkpoint_sha256": sha,
                     "window_s": 180.0}
    side = {"param_names": names,
            "coord": ["ln"] * 17 + ["linear"] * (p - 17),
            "bounds_theta": [[0.0, 1.0]] * p,
            "embedding": embedding,
            "observable": observable}
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    json.dump(side, open(stem + ".json", "w"))

    if parquet == "corrupt":
        open(stem + ".parquet", "wb").write(b"PAR1")
    elif parquet == "valid":
        if not HAVE_PYARROW:
            open(stem + ".parquet", "wb").write(b"PAR1")
        else:
            rng = np.random.default_rng(1)
            Z = rng.normal(size=(rows, E))
            if unit_norm:
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            else:
                Z *= 3.0                      # deliberately off the sphere
            cols = {"z_%03d" % j: pa.array(Z[:, j].astype("float32"))
                    for j in range(E)}
            for j, nm in enumerate(names):
                cols["th_" + nm] = pa.array(
                    np.full(rows, 0.1 * j, dtype="float64"))
            cols["window_idx"] = pa.array(np.zeros(rows, dtype="int32"))
            pq.write_table(pa.table(cols), stem + ".parquet")
    return side


def build(root):
    sim = os.path.join(root, "sim")
    mea4 = os.path.join(root, "mea_out")
    mea1 = os.path.join(root, "mea_out_1electrode")
    for camp, seed in (("campaign_v1", 1000), ("campaign_v2", 2000)):
        for k in range(2):
            make_sim_unit(os.path.join(sim, camp, "sweep_intel_task%04d" % k),
                          seed_master=seed + k)
    for camp in ("campaign_v1", "campaign_v2"):
        for k in range(2):
            if camp == "campaign_v2" and k == 1:
                continue                       # sim unit with no MEA output
            make_mea_unit(os.path.join(mea4, camp, "sweep_intel_task%04d" % k),
                          n_e=4, n_side=2)
            make_mea_unit(os.path.join(mea1, camp, "sweep_intel_task%04d" % k),
                          n_e=1, n_side=1, n_det=100)
    make_mea_unit(os.path.join(mea4, "campaign_ghost", "sweep_intel_task0000"),
                  n_e=4)                       # MEA unit with no sim
    real = os.path.join(root, "real")
    make_real_archive(real)
    reg = os.path.join(root, "HPC_single_run.py")
    open(reg, "w").write("PARAM_NAMES = [%s]\n"
                         % ", ".join("'p%d'" % i for i in range(37)))
    return sim, mea4, mea1, real, reg


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def main():
    root = tempfile.mkdtemp(prefix="dsprof_")
    try:
        sim, mea4, mea1, real, reg = build(root)
        exp_a = os.path.join(root, "export_a", "sbi_v1_0000")
        exp_b = os.path.join(root, "export_b", "sbi_v2_0000")
        make_export_shard(exp_a)
        make_export_shard(exp_b, sha="deadbeef" + "0" * 56)

        p_sim = D.profile_dataset(sim, registry_src=reg, iters_per_unit=2)
        p_m4 = D.profile_dataset(mea4, sim_root=sim, registry_src=reg,
                                 iters_per_unit=2)
        p_m1 = D.profile_dataset(mea1, sim_root=sim, registry_src=reg,
                                 iters_per_unit=2)
        p_real = D.profile_dataset(real)
        p_exp = D.profile_dataset(os.path.join(root, "export_a"))

        def T1():
            assert D.detect_kind(sim) == "sim_campaign"
            assert D.detect_kind(mea4) == "mea_output"
            assert D.detect_kind(real) == "real_extracted"
            assert D.detect_kind(os.path.join(root, "export_a")) == "export_shard"
            assert D.detect_kind(os.path.join(root, "nope")) == "unknown"
        check("T1 all four kinds auto-detect by content, unknown stays unknown", T1)

        def T2():
            L = p_sim["labels"]
            assert L["mode"] == "Full" and L["sweep_group"] == "synapse_astro"
            assert L["conn_rule"] == "weibull" and L["n_active_indices"] == 22
            assert L["swept_axes"] == ["O_N", "U_A", "w_e"], L["swept_axes"]
            assert L["p"] == 3, L["p"]
            assert L["registry_width"] == 37
            assert p_sim["detail"]["len_params_tally"] == {37: 8}, \
                p_sim["detail"]["len_params_tally"]
        check("T2 sim profile: axes, prior, registry width, params tally", T2)

        def T3():
            assert list(p_sim["labels"]["prior_box"]) == ["conn_prob_lo",
                                                          "conn_prob_hi"], \
                p_sim["labels"]["prior_box"]
        check("T3 present-but-null kernel bounds excluded from the prior box", T3)

        def T4():
            bare = D.profile_dataset(mea4)            # no sim_root
            assert bare["observable"]["n_e"] == 4
            assert "INFERRED" in bare["observable"]["T_source"], bare
            assert bare["labels"]["p"] is None
        check("T4 MEA alone gives n_e but flags simtime as inferred, p unknown", T4)

        def T5():
            assert p_m4["observable"]["n_e"] == 4
            assert p_m1["observable"]["n_e"] == 1
            assert p_m4["observable"]["T"] == 180.0
            assert p_m4["observable"]["T_source"].startswith("job_args")
            assert p_m4["labels"]["sweep_group"] == "synapse_astro"
            expect = round(400 / (180.0 * 4), 4)
            assert abs(p_m4["observable"]["rate_hz_per_electrode"]
                       - expect) < 1e-9, p_m4["observable"]
        check("T5 MEA + sim_root: labels join, T from job_args, rate per electrode", T5)

        def T6():
            assert any("no paired sim unit" in w for w in p_m4["warnings"]), \
                p_m4["warnings"]
            assert any("no MEA output" in w for w in p_m4["warnings"]), \
                p_m4["warnings"]
        check("T6 unpaired units warn in both directions", T6)

        def T7():
            u = os.path.join(mea4, "campaign_v1", "sweep_intel_task0000")
            shutil.rmtree(u)
            make_mea_unit(u, n_e=4, simtime_npz=3.0)
            q = D.profile_dataset(mea4, sim_root=sim, iters_per_unit=1)
            assert any("simtime disagrees" in w for w in q["warnings"]), \
                q["warnings"]
            shutil.rmtree(u)
            make_mea_unit(u, n_e=4)
        check("T7 the inferred-simtime trap warns (npz 3 s vs job_args 180 s)", T7)

        def T8():
            odd = os.path.join(root, "sim_bad")
            make_sim_unit(os.path.join(odd, "c", "t0"), n_params=36)
            q = D.profile_dataset(odd, registry_src=reg)
            assert any("registry PARAM_NAMES has 37" in w
                       for w in q["warnings"]), q["warnings"]
        check("T8 registry width vs recorded len(params) mismatch warns", T8)

        def T9():
            odd = os.path.join(root, "sim_dup")
            make_sim_unit(os.path.join(odd, "c1", "t0"), seed_master=7)
            make_sim_unit(os.path.join(odd, "c2", "t0"), seed_master=7)
            q = D.profile_dataset(odd)
            assert q["provenance"]["duplicate_seeds"] == [7], q["provenance"]
            assert any("byte-identical replays" in w for w in q["warnings"])
        check("T9 replayed seed_master flagged as duplicate", T9)

        def T10():
            odd = os.path.join(root, "sim_het")
            make_sim_unit(os.path.join(odd, "c1", "t0"),
                          sweep_group="synapse_astro", mode="Full")
            make_sim_unit(os.path.join(odd, "c2", "t0"),
                          sweep_group="neuron_synapse", mode="Neuronal",
                          swept=("w_e", "tau"))
            q = D.profile_dataset(odd)
            assert D.is_nonconstant(q["labels"]["sweep_group"]), q["labels"]
            assert any("labels.sweep_group is NOT constant" in w
                       for w in q["warnings"]), q["warnings"]
            assert any("labels.swept_axes is NOT constant" in w
                       for w in q["warnings"]), q["warnings"]
        check("T10 a heterogeneous dataset is reported, not silently averaged", T10)

        def T11():
            assert p_real["kind"] == "real_extracted"
            assert p_real["observable"]["fs_ifr"] == 100.0
            assert p_real["observable"]["T"] == 1200.0
            assert p_real["observable"]["n_e"] is None
            assert any("NOT recorded" in w and "n_e" in w
                       for w in p_real["warnings"]), p_real["warnings"]
            r2 = os.path.join(root, "real9")
            make_real_archive(r2, n_e=9)
            q = D.profile_dataset(r2)
            assert q["observable"]["n_e"] == 9, q["observable"]
        check("T11 real archives: fs_ifr/T read; missing n_e warns, present n_e read", T11)

        def T12():
            assert p_exp["labels"]["p"] == 26
            assert p_exp["embedding"]["E"] == 10
            assert p_exp["embedding"]["checkpoint_sha256"].startswith("9d9e0a7f")
            assert p_exp["observable"]["n_e"] == 9
            assert len(p_exp["labels"]["coord"]) == 26
        check("T12 export shard contract read from the sidecar", T12)

        def T12b():
            if HAVE_PYARROW:
                assert p_exp["detail"]["n_rows"] == 8, p_exp["detail"]
                z = p_exp["detail"]["max_abs_znorm_minus_1"]
                assert isinstance(z, float) and z < 1e-5, z
                assert p_exp["detail"]["parquet_errors"] == []
            else:
                assert "pyarrow not installed" in str(
                    p_exp["detail"]["n_rows"]), p_exp["detail"]
        check("T12b with pyarrow: rows counted and ||z|| checked; without: "
              "sidecar-only path", T12b)

        def T13():
            c = D.compare_profiles(p_m4, p_m1)
            assert c["verdict"] == "BLOCKED", c
            fields = [e["field"] for e in c["hard_breaks"]]
            assert "observable.n_e" in fields, fields
        check("T13 n_e 4 vs 1 is a HARD break -> BLOCKED", T13)

        def T14():
            c = D.compare_profiles(p_m4, p_m4)
            assert c["verdict"] == "POOLABLE" and not c["hard_breaks"], c
            unknown = [e["field"] for e in c["not_comparable"]]
            assert "embedding.checkpoint_sha256" in unknown, unknown
            assert "observable.fs_ifr" in unknown, unknown
        check("T14 identical profiles pool; unknown fields are NOT counted as agreement", T14)

        def T15():
            a = D.profile_dataset(os.path.join(root, "export_a"))
            b = D.profile_dataset(os.path.join(root, "export_b"))
            c = D.compare_profiles(a, b)
            assert c["verdict"] == "BLOCKED", c
            assert [e["field"] for e in c["hard_breaks"]] == \
                ["embedding.checkpoint_sha256"], c["hard_breaks"]
        check("T15 two shards under different checkpoints are BLOCKED (A8)", T15)

        def T16():
            q = D.profile_dataset(mea4, sim_root=sim, window_s=1000.0)
            assert any("yields zero windows" in w for w in q["warnings"]), \
                q["warnings"]
        check("T16 T < window_s is caught before any export runs", T16)

        def T17():
            """The regression: a truncated shard must be REPORTED, never raise
            out of profile_dataset. This is the cluster failure of 2026-09-08."""
            bad = os.path.join(root, "export_bad")
            make_export_shard(os.path.join(bad, "sbi_bad_0000"),
                              parquet="corrupt")
            q = D.profile_dataset(bad)                  # must not raise
            assert q["labels"]["p"] == 26, q["labels"]  # sidecar still read
            if HAVE_PYARROW:
                assert q["detail"]["parquet_errors"], q["detail"]
                assert any("could not be read" in w for w in q["warnings"]), \
                    q["warnings"]
        check("T17 a truncated parquet warns instead of crashing the profiler", T17)

        def T18():
            if not HAVE_PYARROW:
                return
            off = os.path.join(root, "export_offsphere")
            make_export_shard(os.path.join(off, "sbi_off_0000"),
                              unit_norm=False)
            q = D.profile_dataset(off)
            assert q["detail"]["max_abs_znorm_minus_1"] > 1e-5, q["detail"]
            assert any("A7 would fail" in w for w in q["warnings"]), \
                q["warnings"]
        check("T18 non-unit ||z|| trips the A7 warning", T18)

        # ---- [2026-09-11] the three defects found on the first real run ----
        def T19():
            # producer keys are read: fs_ifr, sigma_sm, window_s come back
            # with the values the producer wrote, not None
            pa = D.profile_dataset(os.path.dirname(exp_a))
            assert pa["observable"]["fs_ifr"] == 100.0, pa["observable"]
            assert pa["observable"]["sigma_sm"] == 0.05, pa["observable"]
            assert pa["embedding"]["window_s"] == 180.0, pa["embedding"]
            assert pa["detail"].get("n_e_source") == "electrode_centers"
            # and the legacy keys still work through the fallbacks
            leg = os.path.join(root, "export_legacy", "sbi_v0_0000")
            make_export_shard(leg, key_style="legacy")
            pl = D.profile_dataset(os.path.dirname(leg))
            assert pl["observable"]["fs_ifr"] == 100.0, pl["observable"]
            assert pl["observable"]["sigma_sm"] == 0.05, pl["observable"]
            assert pl["embedding"]["window_s"] == 180.0, pl["embedding"]
        check("T19 reader uses the producer's sidecar keys (legacy still works)", T19)

        def T20():
            # a pre-fix shard with no n_e must WARN, not pass silently
            nofix = os.path.join(root, "export_nofix", "sbi_v1_0000")
            make_export_shard(nofix, n_e=None)
            pn = D.profile_dataset(os.path.dirname(nofix))
            assert pn["observable"]["n_e"] is None
            ws = D.check_internal_consistency(pn)
            assert any("n_e is ABSENT" in w for w in ws), ws
            # and a shard WITH n_e does not trip it
            pa = D.profile_dataset(os.path.dirname(exp_a))
            assert not any("n_e is ABSENT" in w
                           for w in D.check_internal_consistency(pa))
        check("T20 an export shard without n_e is flagged (trap 6.7)", T20)

        def T21():
            # a path with nothing to profile is a loud failure, not a
            # success with all-None fields
            ghost = os.path.join(root, "does_not_exist")
            pg = D.profile_dataset(ghost, kind="export_shard")
            assert pg["n_units"] == 0
            assert any("NO UNITS PROFILED" in w for w in pg["warnings"]), pg
            rc = D.main(["--kind", "export_shard", ghost])
            assert rc != 0, "main() returned %r on zero units" % rc
            rc_ok = D.main(["--kind", "export_shard", os.path.dirname(exp_a)])
            assert rc_ok == 0, rc_ok
        check("T21 zero units profiled exits non-zero", T21)

        FULL = {"w_size": 0.01, "gaussian_window": 0.02,
                "electrodes_per_subset": 9, "n_subsets": 9,
                "mfr_threshold": 0.1, "fs_raw": 10110.09,
                "extractor_version": "test"}

        def T22():
            # a patched-extractor archive: every preprocessing parameter is
            # read, with its source recorded, and NO unrecorded warning
            r = os.path.join(root, "real_full_npz")
            make_real_archive(r, full_meta=FULL, meta_in="npz")
            q = D.profile_dataset(r)
            o = q["observable"]
            assert o["sigma_sm"] == 0.02 and o["dt"] == 0.01 and o["n_e"] == 9 \
                and o["mfr_threshold"] == 0.1, o
            src = q["detail"]["preprocessing_sources"]
            assert src["sigma_sm"].startswith("archive:"), src
            assert not any("NOT recorded" in w for w in q["warnings"]), q["warnings"]
            # same, delivered as a companion traces_meta.json
            rj = os.path.join(root, "real_full_json")
            make_real_archive(rj, full_meta=FULL, meta_in="json")
            qj = D.profile_dataset(rj)
            assert qj["observable"]["sigma_sm"] == 0.02, qj["observable"]
            assert qj["detail"]["preprocessing_sources"]["sigma_sm"].startswith(
                "meta.json:"), qj["detail"]["preprocessing_sources"]
        check("T22 patched-extractor metadata read from npz and from traces_meta.json", T22)

        def T23():
            # a legacy archive: sigma_sm / n_e / mfr_threshold unrecorded ->
            # explicit warning naming each, dt still derived from fs_ifr
            o = p_real["observable"]
            assert o["sigma_sm"] is None and o["mfr_threshold"] is None, o
            assert abs(o["dt"] - 0.01) < 1e-12, o
            ws = [w for w in p_real["warnings"] if "NOT recorded" in w]
            assert ws and "sigma_sm" in ws[0] and "mfr_threshold" in ws[0], ws
            assert p_real["detail"]["preprocessing_sources"]["dt"] == "1/fs_ifr"
        check("T23 legacy archive: unrecorded parameters are named, not silently None", T23)

        def T24():
            # THE check the whole exercise exists for: archive sigma_sm vs the
            # export sidecar's checkpoint-declared sigma_sm is a HARD break
            r = os.path.join(root, "real_sig04")
            make_real_archive(r, full_meta=dict(FULL, gaussian_window=0.04),
                              meta_in="npz")
            q = D.profile_dataset(r)
            pe = D.profile_dataset(os.path.dirname(exp_a))   # sigma_sm 0.05
            c = D.compare_profiles(q, pe)
            broken = [b["field"] for b in c["hard_breaks"]]
            assert "observable.sigma_sm" in broken, c
            assert c["verdict"] == "BLOCKED", c["verdict"]
            # and equal values do not break
            r2 = os.path.join(root, "real_sig05")
            make_real_archive(r2, full_meta=dict(FULL, gaussian_window=0.05),
                              meta_in="npz")
            c2 = D.compare_profiles(D.profile_dataset(r2), pe)
            assert "observable.sigma_sm" not in [b["field"] for b in c2["hard_breaks"]], c2
        check("T24 archive sigma_sm vs export-declared sigma_sm is a HARD parity break", T24)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
