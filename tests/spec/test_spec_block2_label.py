"""Block 2 (label): oracles from SPEC.md section 2.2 and Block 2."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import sbi_labels

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rule1(lo, hi):
    """SPEC 2.2 rule 1 with the confirmed rounding-robust threshold."""
    return lo > 0 and hi > 0 and np.log10(hi / lo) >= 1 - 1e-12


def test_L_matches_rule1(registry):
    B = registry.param_bounds
    want = [k for k in range(B.shape[0]) if rule1(B[k, 0], B[k, 1])]
    assert list(registry.log_param_indices) == want


def test_registry_snapshot_values(registry):
    # SPEC: "Current registry values (2026-10 ...) NOT invariants" -- recorded only.
    assert len(registry.param_names) == 36
    assert len(registry.log_param_indices) == 26


def _reg_with_bounds(registry, k, lo, hi):
    from dataclasses import replace
    pb = registry.param_bounds.copy()
    pb[k] = (lo, hi)
    return pb


def test_rule1_threshold_registry_from_manifest(registry):
    """SPEC 2.2 (confirmed): lo=0.07, hi=0.7 is an ln axis although
    hi/lo evaluates to 9.999999999999998. registry_from_manifest (called by the
    entry point whenever manifest.json carries param_bounds) must classify it ln."""
    assert 0.7 / 0.07 < 10.0  # the floating-point premise of the oracle
    k = next(i for i in range(len(registry.param_names))
             if i not in registry.log_param_indices
             and registry.param_bounds[i, 0] > 0
             and registry.param_bounds[i, 0] < registry.param_bounds[i, 1])
    pb = _reg_with_bounds(registry, k, 0.07, 0.7)
    manifest = {"param_bounds": pb.tolist(), "param_names": list(registry.param_names)}
    reg2, changed, flipped = sbi_labels.registry_from_manifest(registry, manifest)
    assert k in reg2.log_param_indices, (
        "axis %s with bounds [0.07, 0.7] (one decade) classified linear"
        % registry.param_names[k])
    assert_allclose(reg2.param_bounds_theta[k], np.log([0.07, 0.7]), rtol=1e-12)


FAKE_SW = textwrap.dedent('''
    import numpy as np
    PARAM_BOUNDS = np.array([[0.07, 0.7], [1.0, 5.0], [2.0, 200.0]])
    LOG_PARAMS = {k for k in range(3) if PARAM_BOUNDS[k, 0] > 0 and
                  np.log10(PARAM_BOUNDS[k, 1] / PARAM_BOUNDS[k, 0]) >= 1 - 1e-12}
    LOG_BASE = "natural"
    PARAM_BOUNDS_THETA = PARAM_BOUNDS.copy()
    for k in LOG_PARAMS:
        PARAM_BOUNDS_THETA[k] = np.log(PARAM_BOUNDS[k])
    KERNEL_BOUNDS = np.array([[0.1, 1.0], [10.0, 100.0], [0.5, 3.0]])
''')
FAKE_SR = textwrap.dedent('''
    PARAM_NAMES = ["a", "b", "c"]
    PARAM_UNITS = ["", "", ""]
    SWEEP_GROUPS = {"g": [0, 1, 2]}
''')


def test_rule1_threshold_load_registry(tmp_path):
    """load_registry on a simulator whose LOG_PARAMS follow the confirmed
    rule 1 (robust threshold) must yield L = {0, 2}."""
    (tmp_path / "HPC_main_sweep.py").write_text(FAKE_SW)
    (tmp_path / "HPC_single_run.py").write_text(FAKE_SR)
    code = ("import sys; sys.path.insert(0, %r); import sbi_labels;"
            "r = sbi_labels.load_registry(%r); print(r.log_param_indices)"
            % (REPO, str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    assert p.stdout.strip() == "[0, 2]"


def test_topology_axes_linear_and_bounds_by_name(registry):
    act = list(registry.sweep_groups["neuron_synapse"])
    kb = np.array([[0.1, 1.0], [20.0, 200.0], [0.5, 4.0]])
    spec = sbi_labels.build_label_spec(registry, act, conn_prob_bounds=(0.15, 0.55),
                                       kernel_bounds=kb,
                                       topology_axes=["beta_conn", "p0_conn", "d0_conn"],
                                       excluded_axes={"conn_prob": "inert"})
    p = len(act) + 3
    assert spec.p == p and len(spec.coord) == p and spec.bounds_theta.shape == (p, 2)
    assert spec.param_names[-3:] == ["beta_conn", "p0_conn", "d0_conn"]
    assert spec.coord[-3:] == ["linear"] * 3
    assert_array_equal(spec.bounds_theta[-3:], [[0.5, 4.0], [0.1, 1.0], [20.0, 200.0]])
    assert spec.excluded_axes == {"conn_prob": "inert"}
    spec4 = sbi_labels.build_label_spec(registry, act, conn_prob_bounds=(0.15, 0.55),
                                        kernel_bounds=kb)
    assert_array_equal(spec4.bounds_theta[-4:],
                       [[0.15, 0.55], [0.1, 1.0], [20.0, 200.0], [0.5, 4.0]])
    assert spec4.bounds_theta.dtype == np.float64


def test_active_rows_follow_rule1(registry):
    act = list(registry.sweep_groups["neuron_synapse"])
    spec = sbi_labels.build_label_spec(registry, act)
    for j, k in enumerate(act):
        lo, hi = registry.param_bounds[k]
        if rule1(lo, hi):
            assert spec.coord[j] == "ln"
            assert_allclose(spec.bounds_theta[j], np.log([lo, hi]), rtol=1e-12)
        else:
            assert spec.coord[j] == "linear"
            assert_array_equal(spec.bounds_theta[j], [lo, hi])
        assert spec.param_names[j] == registry.param_names[k]


def test_A1_round_trip(registry, sim_dir):
    """natural -> theta -> natural at both edges, rtol 1e-9, using the rule
    (ln on L, identity elsewhere) and the registry's own theta bounds."""
    L = set(registry.log_param_indices)
    for k in range(len(registry.param_names)):
        for e in (0, 1):
            v = registry.param_bounds[k, e]
            th = np.log(v) if k in L else v
            assert_allclose(th, registry.param_bounds_theta[k, e], rtol=1e-12, atol=0)
            back = np.exp(th) if k in L else th
            assert_allclose(back, v, rtol=1e-9)
    from export_embeddings import run_assertion_A1
    run_assertion_A1(registry)


@pytest.mark.parametrize("name", ["DeltaT", "VT", "gL"])
def test_A3_frozen_axes(registry, name):
    k = registry.param_names.index(name)
    assert registry.param_bounds[k, 0] == registry.param_bounds[k, 1]
    with pytest.raises(ValueError):
        sbi_labels.build_label_spec(registry, [k])


def _spec(registry):
    return sbi_labels.build_label_spec(registry, list(registry.sweep_groups["neuron_synapse"]))


def _consistent_params_theta(registry, rng):
    L = set(registry.log_param_indices)
    B = registry.param_bounds
    params = np.where(B[:, 0] == B[:, 1], B[:, 0], rng.uniform(B[:, 0], B[:, 1]))
    theta = np.array([np.log(params[k]) if k in L else params[k]
                      for k in range(len(params))])
    return params, theta


def test_assemble_theta_slices_unchanged(registry):
    spec = _spec(registry)
    rng = np.random.default_rng(0)
    params, theta = _consistent_params_theta(registry, rng)
    topo = {"conn_prob": 0.3, "p0_conn": 0.5, "d0_conn": 50.0, "beta_conn": 1.5}
    row = sbi_labels.assemble_theta_A(spec, theta, topo, params_registry=params)
    assert row.dtype == np.float64 and row.shape == (spec.p,)
    assert_array_equal(row[:len(spec.active_indices)], theta[spec.active_indices])
    assert_array_equal(row[len(spec.active_indices):], [0.3, 0.5, 50.0, 1.5])


def test_A6_mismatch_raises(registry):
    spec = _spec(registry)
    rng = np.random.default_rng(1)
    params, theta = _consistent_params_theta(registry, rng)
    topo = {"conn_prob": 0.3, "p0_conn": 0.5, "d0_conn": 50.0, "beta_conn": 1.5}
    for k in (spec.active_indices[0], spec.active_indices[-1]):
        bad = theta.copy()
        bad[k] *= 1.001
        with pytest.raises(ValueError):
            sbi_labels.assemble_theta_A(spec, bad, topo, params_registry=params)
    # A log axis stored as log10 instead of ln is the classic error
    k = next(a for a in spec.active_indices if a in registry.log_param_indices)
    bad = theta.copy()
    bad[k] = np.log10(params[k])
    with pytest.raises(ValueError):
        sbi_labels.assemble_theta_A(spec, bad, topo, params_registry=params)


def test_label_errors(registry):
    n = len(registry.param_names)
    act = list(registry.sweep_groups["neuron_synapse"])
    for bad_act in ([], [act[0], act[0]], [n], [-1]):
        with pytest.raises(ValueError):
            sbi_labels.build_label_spec(registry, bad_act)
    with pytest.raises(ValueError):
        sbi_labels.build_label_spec(registry, act, kernel_bounds=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        sbi_labels.build_label_spec(registry, act, topology_axes=["foo"])
    with pytest.raises(ValueError):
        sbi_labels.build_label_spec(registry, act, topology_axes=["p0_conn", "p0_conn"])
    spec = _spec(registry)
    topo = {"conn_prob": 0.3, "p0_conn": 0.5, "d0_conn": 50.0, "beta_conn": 1.5}
    with pytest.raises(ValueError):
        sbi_labels.assemble_theta_A(spec, np.zeros(n - 1), topo)
    with pytest.raises(ValueError):
        sbi_labels.assemble_theta_A(spec, np.zeros(n), topo, params_registry=np.ones(n + 1))
    with pytest.raises(KeyError):
        sbi_labels.assemble_theta_A(spec, np.zeros(n), {"conn_prob": 0.3})


def test_noninteger_active_index_refused(registry):
    """SPEC: active_indices, each in {0, ..., n-1}. A float 1.7 is not an index;
    it must not be silently truncated to axis 1."""
    with pytest.raises((ValueError, TypeError)):
        sbi_labels.build_label_spec(registry, [1.7, 3])


# ---------------- added by the 2026-10-06 15:27 run ----------------
def test_registry_from_manifest_refusals_and_flip(registry):
    pb = registry.param_bounds.copy()
    with pytest.raises(ValueError):                      # width mismatch
        sbi_labels.registry_from_manifest(registry, {"param_bounds": pb[:-1].tolist()})
    names = list(registry.param_names)
    with pytest.raises(ValueError):                      # reordered names
        sbi_labels.registry_from_manifest(registry, {"param_bounds": pb.tolist(),
                                                     "param_names": names[::-1]})
    with pytest.raises(ValueError):                      # log10 base
        sbi_labels.registry_from_manifest(registry, {"param_bounds": pb.tolist(),
                                                     "log_transform": "log10"})
    # recorded log_params inconsistent with the recorded bounds
    lp = [names[k] for k in registry.log_param_indices][1:]
    with pytest.raises(ValueError):
        sbi_labels.registry_from_manifest(registry, {"param_bounds": pb.tolist(),
                                                     "log_params": lp})
    # identical bounds -> nothing changed, nothing flipped
    r2, changed, flipped = sbi_labels.registry_from_manifest(registry, {"param_bounds": pb.tolist()})
    assert changed == [] and flipped == []
    assert r2.log_param_indices == registry.log_param_indices
    # Sigma widened from [3, 15] to [3, 30] crosses one decade -> flips to ln
    k = names.index("Sigma")
    pb2 = pb.copy(); pb2[k] = (3.0, 30.0)
    r3, changed, flipped = sbi_labels.registry_from_manifest(registry, {"param_bounds": pb2.tolist()})
    assert "Sigma" in changed and "Sigma" in flipped and k in r3.log_param_indices
    assert_allclose(r3.param_bounds_theta[k], np.log([3.0, 30.0]))
