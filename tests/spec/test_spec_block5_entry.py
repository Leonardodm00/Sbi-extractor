"""Block 5 (entry point): oracles from SPEC.md Block 5, run through the real
command line on synthetic data and on a small fabricated campaign."""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.ndimage import gaussian_filter1d

from conftest import REPO, save_demo_checkpoint

DT, SIG, W = 0.02, 0.04, 100          # demo checkpoint: T_win = 2 s


def run_cli(args, cwd):
    p = subprocess.run([sys.executable, os.path.join(REPO, "example_export.py")] + args,
                       capture_output=True, text=True, cwd=cwd, env=dict(os.environ))
    return p


def oracle_x(groups, n_e, T, dt, sigma):
    K = int(np.floor(T / dt))
    edges = np.arange(K + 1) * dt
    C = sum(np.histogram(g, bins=edges)[0].astype(float) for g in groups if len(g))
    R = np.clip(gaussian_filter1d(np.asarray(C, float) * np.ones(K), sigma / dt), 0, None)
    return (R.astype(np.float32) / float(n_e)).astype(np.float32)


# ------------------------------------------------------------------ synthetic
def test_synthetic_end_to_end_and_reproducible(tmp_path, sim_dir, sbi_hpc_dir):
    outs = []
    for k in range(2):
        stem = str(tmp_path / ("sbi_demo_%d" % k))
        p = run_cli(["--mode", "synthetic", "--out", stem, "--n_sims", "16"], str(tmp_path))
        assert p.returncode == 0, p.stderr[-2000:]
        assert os.path.isfile(stem + ".parquet") and os.path.isfile(stem + ".json")
        t = pq.read_table(stem + ".parquet")
        sc = json.load(open(stem + ".json"))
        assert t.num_rows == 16 == sc["n_rows"]
        assert sc["assertions_passed"] == ["A2", "A3", "A4", "A5", "A7", "A9"]
        assert sc["embedding"]["dsn_checkpoint_sha256"] == "0" * 64
        assert any("DEMO MODE" in w for w in sc["warnings"])
        for name in t.column_names:
            col = t.column(name).to_numpy()
            if col.dtype.kind == "f":
                assert np.all(np.isfinite(col)), name
        outs.append(t)
    # same seed -> identical results (embeddings may differ only if the demo
    # backbone's init is unseeded; SPEC 4: export path deterministic, demo rng(0))
    a, b = outs
    for name in a.column_names:
        if name.startswith(("th_", "window_idx", "iter_idx", "topo_idx", "seed_run")):
            assert a.column(name).equals(b.column(name)), name
    zdiff = [n for n in a.column_names if n.startswith("z") and not a.column(n).equals(b.column(n))]
    assert not zdiff, "z columns differ between two synthetic runs: %s" % zdiff[:3]


# ------------------------------------------------------------------ campaign
def _consistent(registry, rng):
    L = set(registry.log_param_indices)
    B = registry.param_bounds
    params = np.where(B[:, 0] == B[:, 1], B[:, 0], rng.uniform(B[:, 0], B[:, 1]))
    theta = np.array([np.log(params[k]) if k in L else params[k] for k in range(len(params))])
    return params, theta


def make_campaign(root, registry, simtime=6.0, n_topo=2, n_iter=2, job_args_extra=None,
                  det_ch_offset=0, write_simtime=True):
    rng = np.random.default_rng(42)
    camp, mea = root / "camp", root / "mea"
    act = list(registry.sweep_groups["neuron_synapse"])
    camp.mkdir(); mea.mkdir()
    json.dump({"active_indices": act, "sweep_group": "neuron_synapse"},
              open(camp / "manifest.json", "w"))
    ja = {"conn_prob_lo": 0.1, "conn_prob_hi": 0.6}
    if write_simtime:
        ja["simtime"] = simtime
    ja.update(job_args_extra or {})
    json.dump(ja, open(camp / "job_args.json", "w"))
    truth = {}
    for t in range(n_topo):
        (camp / ("topo_%d" % t)).mkdir(); (mea / ("topo_%d" % t)).mkdir()
        # inside the registry kernel bounds [[0.1,1],[60,300],[1,2]] (default prior box)
        p0, d0, beta = rng.uniform(0.1, 0.9), rng.uniform(60, 290), rng.uniform(1.0, 1.9)
        for i in range(n_iter):
            cp = rng.uniform(0.1, 0.6)
            np.savez(camp / ("topo_%d" % t) / ("iter_%d.npz" % i), conn_prob=cp,
                     p0_conn=p0 + 0.01 * i, d0_conn=d0 + i, beta_conn=beta + 0.01 * i)
            params, theta = _consistent(registry, rng)
            n_det = int(rng.poisson(40 * simtime))
            det_t = np.sort(rng.uniform(0, simtime * (1 - 1e-9), n_det))
            det_ch = rng.integers(0, 9, n_det) + det_ch_offset
            np.savez(mea / ("topo_%d" % t) / ("mea_iter_%d.npz" % i), det_t=det_t,
                     det_ch=det_ch, electrode_centers=np.zeros((9, 2)), theta=theta,
                     params=params, topo_idx=t, iter_idx=i, seed_run=100 * t + i,
                     simtime=float(np.ceil(det_t.max())) / 2)   # decoy simtime
            truth[(t, i)] = dict(conn_prob=cp, p0_conn=p0 + 0.01 * i, d0_conn=d0 + i,
                                 beta_conn=beta + 0.01 * i, det_t=det_t, det_ch=det_ch,
                                 theta=theta)
    return camp, mea, truth


@pytest.fixture(scope="module")
def ckpt(dsn_dir, tmp_path_factory):
    return save_demo_checkpoint(dsn_dir, str(tmp_path_factory.mktemp("c5") / "best.pt"),
                                W=W, dt=DT, sigma=SIG)


def test_campaign_mode_join_and_K(tmp_path, registry, ckpt, sim_dir):
    camp, mea, truth = make_campaign(tmp_path, registry)
    stem = str(tmp_path / "out" / "sbi_c_0000")
    p = run_cli(["--mode", "campaign", "--checkpoint", ckpt, "--campaign", str(camp),
                 "--mea_out", str(mea), "--label_axes", "none", "--campaign_id", "c",
                 "--out", stem], str(tmp_path))
    assert p.returncode == 0, p.stderr[-3000:]
    t = pq.read_table(stem + ".parquet").to_pydict()
    # simtime 6 s -> K = 300 -> 3 windows of W = 100 per record (decoy simtime ignored)
    assert len(t["iter_idx"]) == 4 * 3
    for r in range(len(t["iter_idx"])):
        key = (t["topo_idx"][r], t["iter_idx"][r])
        for a in ("conn_prob", "p0_conn", "d0_conn", "beta_conn"):
            assert t["th_" + a][r] == truth[key][a], (key, a)
    # Z of record (1, 1) equals an independent build of x from det_t/det_ch
    from dsn_frozen import load_frozen_dsn
    f = load_frozen_dsn(ckpt)
    tr = truth[(1, 1)]
    groups = [tr["det_t"][tr["det_ch"] == e] for e in range(9)]
    x = oracle_x(groups, 9, 6.0, DT, SIG)
    Z, _ = f.embed(np.stack([x[i * W:(i + 1) * W] for i in range(3)]))
    rows = [r for r in range(len(t["iter_idx"])) if (t["topo_idx"][r], t["iter_idx"][r]) == (1, 1)]
    Zb = np.array([[t["z_%03d" % j][r] for j in range(16)] for r in rows], dtype=np.float32)
    # the shard is embedded in one batch, here a smaller one: SPEC Block 3
    # allows batch-size differences up to 1e-6 (float32 rounding, Q4).
    assert_allclose(Zb, Z, rtol=0, atol=1e-6)
    sc = json.load(open(stem + ".json"))
    assert sc["observable"]["n_electrodes"] == 9


def test_campaign_requires_label_axes(tmp_path, registry, ckpt, sim_dir):
    camp, mea, _ = make_campaign(tmp_path, registry)
    p = run_cli(["--mode", "campaign", "--checkpoint", ckpt, "--campaign", str(camp),
                 "--mea_out", str(mea), "--out", str(tmp_path / "x")], str(tmp_path))
    assert p.returncode != 0 and "label_axes" in p.stderr
    assert not os.path.exists(str(tmp_path / "x.parquet"))


def test_iter_campaign_records_direct(tmp_path, registry, sim_dir, dsn_dir):
    import sbi_labels
    from example_export import iter_campaign_records
    camp, mea, truth = make_campaign(tmp_path, registry, n_topo=1, n_iter=2)
    spec = sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]))
    # dt and sigma_sm are required (no defaults)
    with pytest.raises(TypeError):
        iter_campaign_records(str(camp), str(mea), spec, 6.0, "c")
    recs = list(iter_campaign_records(str(camp), str(mea), spec, 6.0, "c", dt=DT, sigma_sm=SIG))
    tr = truth[(0, 0)]
    x = oracle_x([tr["det_t"][tr["det_ch"] == e] for e in range(9)], 9, 6.0, DT, SIG)
    assert_array_equal(recs[0].trace, x)
    assert_array_equal(recs[0].theta_A[:len(spec.active_indices)], tr["theta"][spec.active_indices])
    # trim: removed after building on the full grid, before windowing
    recs_t = list(iter_campaign_records(str(camp), str(mea), spec, 6.0, "c", dt=DT,
                                        sigma_sm=SIG, trim_head_s=0.5))
    assert_array_equal(recs_t[0].trace, x[int(round(0.5 / DT)):])


def test_join_must_be_unique_and_complete(tmp_path, registry, sim_dir):
    import sbi_labels
    from example_export import iter_campaign_records
    camp, mea, _ = make_campaign(tmp_path, registry, n_topo=1, n_iter=1)
    spec = sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]))
    np.savez(camp / "topo_0" / "iter_000.npz", conn_prob=0.2, p0_conn=0.2, d0_conn=80.0,
             beta_conn=1.5)                                  # second file with index 0
    with pytest.raises(Exception):
        list(iter_campaign_records(str(camp), str(mea), spec, 6.0, "c", dt=DT, sigma_sm=SIG))
    os.remove(camp / "topo_0" / "iter_000.npz")
    np.savez(camp / "topo_0" / "iter_0.npz", conn_prob=0.2, p0_conn=0.2, d0_conn=80.0)
    with pytest.raises(KeyError):
        list(iter_campaign_records(str(camp), str(mea), spec, 6.0, "c", dt=DT, sigma_sm=SIG))


def test_out_of_range_det_ch_not_silently_dropped(tmp_path, registry, sim_dir):
    """SPEC section 3: det_ch in {0, ..., n_e-1}; section 4: 'Nothing is silently
    clipped, repaired or dropped.' 1-based channel indices (1..9 with n_e = 9)
    must not silently lose every spike on channel 9."""
    import sbi_labels
    from example_export import iter_campaign_records
    camp, mea, truth = make_campaign(tmp_path, registry, n_topo=1, n_iter=1, det_ch_offset=1)
    spec = sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]))
    with pytest.raises(ValueError):
        list(iter_campaign_records(str(camp), str(mea), spec, 6.0, "c", dt=DT, sigma_sm=SIG))


def test_missing_simtime_not_silently_defaulted(tmp_path, registry, ckpt, sim_dir):
    """SPEC Block 5: 'T from job_args.json (or --simtime)'; section 4: failures
    raise, no silent defaults. A job_args.json without simtime and no --simtime
    must be refused, not replaced by 180 s."""
    # the campaign really ran 200 s; job_args.json does not record simtime.
    camp, mea, _ = make_campaign(tmp_path, registry, simtime=200.0, n_topo=1,
                                 n_iter=2, write_simtime=False)
    stem = str(tmp_path / "out" / "s")
    p = run_cli(["--mode", "campaign", "--checkpoint", ckpt, "--campaign", str(camp),
                 "--mea_out", str(mea), "--label_axes", "none", "--out", stem], str(tmp_path))
    assert p.returncode != 0, "export ran with T silently defaulted:\n" + p.stdout[-900:]


def test_swap_checkpoint_geometry_and_label_axes(tmp_path, registry, dsn_dir, sim_dir):
    """Directive audit, swappability: a checkpoint with a different Delta_t /
    sigma_sm / W and a frozen label_axes.json with 3 topology axes; the rest of
    the pipeline must follow without code changes (SPEC 2.3, 4)."""
    ck = save_demo_checkpoint(dsn_dir, str(tmp_path / "c01.pt"), W=200, dt=0.01, sigma=0.02)
    camp, mea, truth = make_campaign(tmp_path, registry)
    la = tmp_path / "label_axes.json"
    json.dump({"topology_axes": ["p0_conn", "d0_conn", "beta_conn"],
               "excluded_axes": {"conn_prob": {"reason": "inert under weibull"}}},
              open(la, "w"))
    stem = str(tmp_path / "out" / "swap")
    p = run_cli(["--mode", "campaign", "--checkpoint", ck, "--campaign", str(camp),
                 "--mea_out", str(mea), "--label_axes", str(la), "--out", stem,
                 "--trim_head_s", "0.5"], str(tmp_path))
    assert p.returncode == 0, p.stderr[-2000:]
    t = pq.read_table(stem + ".parquet")
    sc = json.load(open(stem + ".json"))
    # K = 600, trim 50 -> 550 samples -> 2 windows of 200 per record
    assert t.num_rows == 4 * 2
    assert sc["observable"]["ifr_dt_s"] == 0.01 and sc["observable"]["window_length_samples"] == 200
    assert "th_conn_prob" not in t.column_names and sc["label_axes"]["p"] == len(sc["param_names"])
    tr = truth[(0, 0)]
    from dsn_frozen import load_frozen_dsn
    x = oracle_x([tr["det_t"][tr["det_ch"] == e] for e in range(9)], 9, 6.0, 0.01, 0.02)[50:]
    Z, _ = load_frozen_dsn(ck).embed(np.stack([x[:200], x[200:400]]))
    d = t.to_pydict()
    rows = [r for r in range(t.num_rows) if (d["topo_idx"][r], d["iter_idx"][r]) == (0, 0)]
    Zb = np.array([[d["z_%03d" % j][r] for j in range(16)] for r in rows], dtype=np.float32)
    # the shard is embedded in one batch, here a smaller one: SPEC Block 3
    # allows batch-size differences up to 1e-6 (float32 rounding, Q4).
    assert_allclose(Zb, Z, rtol=0, atol=1e-6)
