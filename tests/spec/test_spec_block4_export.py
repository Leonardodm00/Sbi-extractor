"""Block 4 (export): oracles from SPEC.md section 2.4 and Block 4."""
import copy
import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal

import sbi_labels
from export_embeddings import TraceRecord, export_embeddings


@pytest.fixture(scope="module")
def spec(registry):
    return sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]))


def _theta(spec, rng):
    lo, hi = spec.bounds_theta[:, 0], spec.bounds_theta[:, 1]
    return rng.uniform(lo, hi)


def _records(spec, W, lengths, seed=0, labelled=True):
    rng = np.random.default_rng(seed)
    out = []
    for i, K in enumerate(lengths):
        out.append(TraceRecord(trace=np.abs(rng.normal(size=K)).astype(np.float32),
                               ident={"campaign_id": "c", "topo_idx": i // 2, "iter_idx": i},
                               theta_A=_theta(spec, rng) if labelled else None))
    return out


IDENT = ("campaign_id", "topo_idx", "iter_idx")


def test_rows_columns_order_and_roundtrip(frozen, spec, tmp_path):
    W = frozen.window_length
    lengths = [3 * W, W, 2 * W + W // 2, W - 1, 5 * W - 1]
    recs = _records(spec, W, lengths)
    res = export_embeddings(frozen, recs, str(tmp_path / "s"), label_spec=spec,
                            ident_columns=IDENT)
    n_win = [(K - W) // W + 1 if K >= W else 0 for K in lengths]
    assert res.n_rows == sum(n_win) == 3 + 1 + 2 + 0 + 4
    assert res.n_traces_used == 4 and res.n_traces_skipped_short == 1
    t = pq.read_table(res.parquet_path)
    E = frozen.embedding_dim
    want_cols = (list(IDENT) + ["window_idx"] + ["z_%03d" % j for j in range(E)]
                 + ["zraw_%03d" % j for j in range(E)] + spec.column_names)
    assert t.column_names == want_cols
    sch = t.schema
    for j in range(E):
        assert str(sch.field("z_%03d" % j).type) == "float"
        assert str(sch.field("zraw_%03d" % j).type) == "float"
    for c in spec.column_names:
        assert str(sch.field(c).type) == "double"
    assert str(sch.field("window_idx").type).startswith("int")
    df = t.to_pydict()
    # window_idx restarts per record, rows keep record order, ident/theta inherited
    exp_idx, exp_iter, exp_theta = [], [], []
    for r, n in zip(recs, n_win):
        exp_idx += list(range(n)); exp_iter += [r.ident["iter_idx"]] * n
        exp_theta += [r.theta_A] * n
    assert df["window_idx"] == exp_idx and df["iter_idx"] == exp_iter
    Th = np.column_stack([np.asarray(df[c]) for c in spec.column_names])
    assert_array_equal(Th, np.asarray(exp_theta))
    # Z read back equals an independent embed of the same windows, exactly
    Xs = []
    for r, n in zip(recs, n_win):
        for i in range(n):
            Xs.append(r.trace[i * W:(i + 1) * W])
    Z, _ = frozen.embed(np.stack(Xs))
    Zback = np.column_stack([np.asarray(df["z_%03d" % j], dtype=np.float32) for j in range(E)])
    assert_array_equal(Zback, Z)
    # sidecar
    sc = json.load(open(res.sidecar_path))
    assert sc["schema_version"] == 1 and sc["n_rows"] == res.n_rows
    assert sc["n_traces_used"] == 4 and sc["n_traces_skipped_too_short"] == 1
    assert sc["window_stride_samples"] == W
    assert sc["embedding"]["dsn_checkpoint_sha256"] == frozen.ckpt_sha256
    assert sc["observable"]["pooling"] == "mean_over_electrodes"
    assert "spike_source" in sc["observable"]
    assert sc["assertions_passed"] == sorted(sc["assertions_passed"])
    assert set(sc["assertions_passed"]) == {"A2", "A3", "A4", "A5", "A7", "A9"}
    assert sc["param_names"] == spec.param_names and sc["coord"] == spec.coord
    assert np.asarray(sc["bounds_theta"]).shape == (spec.p, 2)
    assert "registry" in sc and "param_units" in sc
    assert any("short" in w.lower() or "SHORTER" in w for w in sc["warnings"])


def test_unlabelled(frozen, tmp_path):
    W = frozen.window_length
    # SPEC Block 4 oracle: "With label_spec = None: no th_* columns, and A9 is
    # the only assertion in assertions_passed."
    rng = np.random.default_rng(1)
    recs = [TraceRecord(trace=np.abs(rng.normal(size=2 * W)).astype(np.float32),
                        ident={"rec": "a"})]
    res = export_embeddings(frozen, recs, str(tmp_path / "u"), ident_columns=("rec",))
    t = pq.read_table(res.parquet_path)
    assert not [c for c in t.column_names if c.startswith("th_")]
    assert json.load(open(res.sidecar_path))["assertions_passed"] == ["A9"]


def test_stride_overlap(frozen, spec, tmp_path):
    W = frozen.window_length
    recs = _records(spec, W, [3 * W, 3 * W])
    res = export_embeddings(frozen, recs, str(tmp_path / "o"), label_spec=spec,
                            window_stride=W // 2)
    assert res.n_rows == 2 * ((3 * W - W) // (W // 2) + 1)
    assert any("OVERLAP" in w for w in res.warnings)


def _assert_no_files(stem):
    assert not os.path.exists(stem + ".parquet")
    assert not os.path.exists(stem + ".json")
    leftovers = [f for f in os.listdir(os.path.dirname(stem)) if f.endswith(".tmp")]
    assert not leftovers


def test_errors_and_atomicity(frozen, spec, tmp_path):
    W = frozen.window_length
    stem = str(tmp_path / "e")
    with pytest.raises(ValueError):                         # nothing long enough
        export_embeddings(frozen, _records(spec, W, [W - 1]), stem, label_spec=spec)
    _assert_no_files(stem)
    with pytest.raises(ValueError):                         # theta None with spec
        export_embeddings(frozen, _records(spec, W, [W, W], labelled=False), stem, label_spec=spec)
    _assert_no_files(stem)
    recs = _records(spec, W, [W, W])
    recs[1].theta_A = recs[1].theta_A[:-1]
    with pytest.raises(ValueError):                         # wrong length
        export_embeddings(frozen, recs, stem, label_spec=spec)
    _assert_no_files(stem)


def test_assertions_fire(frozen, spec, tmp_path):
    W = frozen.window_length
    stem = str(tmp_path / "a")
    # A9: NaN theta
    recs = _records(spec, W, [W, W, W])
    recs[1].theta_A = recs[1].theta_A.copy(); recs[1].theta_A[0] = np.nan
    with pytest.raises(AssertionError, match="A9"):
        export_embeddings(frozen, recs, stem, label_spec=spec)
    _assert_no_files(stem)
    # A4: one constant column
    recs = _records(spec, W, [W, W, W])
    for r in recs:
        r.theta_A = r.theta_A.copy(); r.theta_A[2] = recs[0].theta_A[2]
    with pytest.raises(AssertionError, match="A4"):
        export_embeddings(frozen, recs, stem, label_spec=spec)
    _assert_no_files(stem)
    res = export_embeddings(frozen, recs, stem, label_spec=spec,
                            allow_constant=[spec.param_names[2]])
    assert any("A4" in w for w in res.warnings)
    os.remove(res.parquet_path); os.remove(res.sidecar_path)
    # A5: out of box
    recs = _records(spec, W, [W, W, W])
    recs[0].theta_A = recs[0].theta_A.copy()
    recs[0].theta_A[0] = spec.bounds_theta[0, 1] + 1.0
    with pytest.raises(AssertionError, match="A5"):
        export_embeddings(frozen, recs, stem, label_spec=spec)
    _assert_no_files(stem)
    res = export_embeddings(frozen, recs, stem, label_spec=spec, strict_in_box=False)
    assert "A5" not in res.assertions_passed
    assert any("A5" in w for w in res.warnings)
    t = pq.read_table(res.parquet_path)
    assert t.column("th_" + spec.param_names[0]).to_pylist()[0] == recs[0].theta_A[0]  # not clipped
    os.remove(res.parquet_path); os.remove(res.sidecar_path)
    # A2 / A3: corrupt spec after construction
    bad = copy.deepcopy(spec); bad.coord = bad.coord[:-1]
    with pytest.raises(AssertionError, match="A2"):
        export_embeddings(frozen, _records(spec, W, [W, W]), stem, label_spec=bad)
    bad = copy.deepcopy(spec); bad.bounds_theta = bad.bounds_theta.copy()
    bad.bounds_theta[1, 1] = bad.bounds_theta[1, 0]
    with pytest.raises(AssertionError, match="A3"):
        export_embeddings(frozen, _records(spec, W, [W, W]), stem, label_spec=bad)
    _assert_no_files(stem)


def test_A7_fires(frozen, spec, tmp_path):
    f2 = copy.copy(frozen)
    orig = frozen.embed

    def bad_embed(X, batch_size=256, want_zraw=True):
        Z, Zr = orig(X, batch_size, want_zraw)
        return Z * 1.001, Zr
    f2.embed = bad_embed
    stem = str(tmp_path / "a7")
    with pytest.raises(AssertionError, match="A7"):
        export_embeddings(f2, _records(spec, frozen.window_length, [frozen.window_length] * 2),
                          stem, label_spec=spec)
    _assert_no_files(stem)


def test_atomic_when_sidecar_write_fails(frozen, spec, tmp_path):
    """SPEC Block 4: 'Writes are atomic: after a failure, no partial .parquet
    or .json exists at out_stem.' A sidecar that cannot be serialised (here a
    numpy integer in extra_sidecar, as an np.load-derived value would be) makes
    the call fail; no .parquet may be left behind."""
    stem = str(tmp_path / "atom")
    W = frozen.window_length
    with pytest.raises(TypeError):
        export_embeddings(frozen, _records(spec, W, [W, W]), stem, label_spec=spec,
                          extra_sidecar={"provenance": {"n": np.int64(3)}})
    _assert_no_files(stem)


def test_sidecar_registry_block_consistent_with_label(frozen, registry, tmp_path):
    """With an excluded topology axis, the sidecar must not declare it as part
    of the label anywhere (SPEC 2.2: Tau fixed by label_axes.json; Block 4
    sidecar carries param_names and registry)."""
    sp = sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]),
                                     topology_axes=["p0_conn", "d0_conn", "beta_conn"],
                                     excluded_axes={"conn_prob": "inert"})
    W = frozen.window_length
    res = export_embeddings(frozen, _records(sp, W, [W, W]), str(tmp_path / "r"), label_spec=sp)
    sc = json.load(open(res.sidecar_path))
    assert "conn_prob" not in sc["param_names"]
    assert sc["registry"]["topology_axes"] == ["p0_conn", "d0_conn", "beta_conn"]


# ---------------- added by the 2026-10-06 15:27 run ----------------
def test_multichannel_trace(dsn_dir, tmp_path):
    """Block 4 Inputs: trace of shape (C_in, K); windows cut along the last axis."""
    from conftest import save_demo_checkpoint
    from dsn_frozen import load_frozen_dsn
    ck = save_demo_checkpoint(dsn_dir, str(tmp_path / "c3.pt"), in_channels=3)
    f = load_frozen_dsn(ck)
    W = f.window_length
    rng = np.random.default_rng(5)
    tr = np.abs(rng.normal(size=(3, 2 * W + 7))).astype(np.float32)
    res = export_embeddings(f, [TraceRecord(trace=tr, ident={"r": 1})], str(tmp_path / "m"),
                            ident_columns=("r",))
    assert res.n_rows == 2
    Z, _ = f.embed(np.stack([tr[:, :W], tr[:, W:2 * W]]))
    t = pq.read_table(res.parquet_path)
    Zb = np.column_stack([t.column("z_%03d" % j).to_numpy() for j in range(f.embedding_dim)])
    np.testing.assert_allclose(Zb, Z, rtol=0, atol=1e-6)   # batch-size tolerance (Q4)


def test_missing_ident_value_not_silently_null(frozen, spec, tmp_path):
    """SPEC 4: nothing silently dropped. A record lacking a requested ident
    column must not be written with a null provenance value without notice."""
    W = frozen.window_length
    recs = _records(spec, W, [W, W])
    del recs[1].ident["topo_idx"]
    try:
        res = export_embeddings(frozen, recs, str(tmp_path / "n"), label_spec=spec,
                                ident_columns=IDENT)
    except (KeyError, ValueError):
        return
    t = pq.read_table(res.parquet_path).to_pydict()
    assert None not in t["topo_idx"] or any("topo_idx" in w for w in res.warnings)
