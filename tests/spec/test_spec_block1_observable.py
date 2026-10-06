"""Block 1 (observable): oracles from SPEC.md section 2.1 and Block 1.

Expected values are computed with numpy / scipy directly, never with the code
under test.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.ndimage import gaussian_filter1d


def oracle_x(groups, n_e, T, dt, sigma):
    """SPEC 2.1 literally: histogram on edges k*dt, gaussian_filter1d(reflect,
    truncate 4), clip at 0, float32, divide by n_e, float32."""
    K = int(np.floor(T / dt))
    edges = np.arange(K + 1, dtype=np.float64) * dt
    C = np.zeros(K)
    for g in groups:
        g = np.asarray(g, dtype=np.float64)
        if g.size:
            C += np.histogram(g, bins=edges)[0]
    R = np.clip(gaussian_filter1d(C, sigma=sigma / dt), 0.0, None).astype(np.float32)
    # float32 R divided by a Python float: numpy keeps float32 (NEP 50), then cast.
    return (R / float(n_e)).astype(np.float32)


def rand_groups(rng, n_e, T, rate=3.0):
    # Uniform on [0, T*(1-1e-9)): never exactly K*dt (outside the contract).
    return [np.sort(rng.uniform(0.0, T * (1 - 1e-9), rng.poisson(rate * T)))
            for _ in range(n_e)]


@pytest.fixture(scope="module")
def so(dsn_dir):
    import sim_observable
    return sim_observable


@pytest.mark.parametrize("dt,sigma,T", [(0.02, 0.04, 30.0), (0.01, 0.02, 12.0),
                                        (0.05, 0.3, 7.3)])
def test_bit_exact_recomputation(so, dt, sigma, T):
    rng = np.random.default_rng(1)
    for trial in range(10):
        n_e = int(rng.integers(1, 12))
        groups = rand_groups(rng, n_e, T)
        x = so.build_pooled_ifr(groups, n_e, T, dt, sigma)
        want = oracle_x(groups, n_e, T, dt, sigma)
        assert x.dtype == np.float32 and x.shape == want.shape
        # SPEC: "equal to x bit for bit"
        assert_array_equal(x, want)


def test_pooled_equals_per_electrode(so):
    rng = np.random.default_rng(2)
    groups = rand_groups(rng, 5, 20.0)
    pooled = np.sort(np.concatenate(groups))
    a = so.build_pooled_ifr(groups, 5, 20.0, 0.02, 0.04)
    b = so.build_pooled_ifr(pooled, 5, 20.0, 0.02, 0.04)
    assert_array_equal(a, b)


def test_mean_not_sum(so):
    rng = np.random.default_rng(3)
    S = np.sort(rng.uniform(0, 10.0 - 1e-6, 400))
    one = so.build_pooled_ifr([S], 1, 10.0, 0.02, 0.04)
    for n_e in (2, 9):
        many = so.build_pooled_ifr([S] * n_e, n_e, 10.0, 0.02, 0.04)
        # float32 rounding: counts n_e*c are exact, smoothing is linear, two
        # float32 roundings -> a few ulp; rtol 1e-6 ~ 8 float32 ulp.
        assert_allclose(many, one, rtol=1e-6, atol=1e-7)


def test_silent_electrode_counts(so):
    rng = np.random.default_rng(4)
    groups = rand_groups(rng, 4, 10.0)
    a = so.build_pooled_ifr(groups, 4, 10.0, 0.02, 0.04)
    b = so.build_pooled_ifr(groups + [np.array([])], 5, 10.0, 0.02, 0.04)
    assert_allclose(b, a * 4.0 / 5.0, rtol=1e-6, atol=1e-7)  # float32 rounding


@pytest.mark.parametrize("sigma", [0.0, 0.04, 0.5, 5.0])
def test_count_conservation(so, sigma):
    rng = np.random.default_rng(5)
    T, dt = 6.0, 0.02
    K = int(np.floor(T / dt))
    for n_e in (1, 9):
        groups = rand_groups(rng, n_e, T)
        # spikes near both grid ends
        groups[0] = np.sort(np.concatenate([groups[0], [0.0, 1e-9, dt * 0.5,
                                                       K * dt - 1e-9, K * dt - dt / 2]]))
        n_in = sum(int(np.sum((g >= 0) & (g < K * dt))) for g in groups)
        x = so.build_pooled_ifr(groups, n_e, T, dt, sigma)
        # SPEC: relative tolerance 1e-6 (float32)
        assert_allclose(n_e * np.sum(x, dtype=np.float64), n_in, rtol=1e-6)


def test_sigma_zero_is_no_smoothing(so):
    """sigma_sm >= 0 is a valid input (SPEC Block 1 Inputs); sigma = 0 must give
    x = C / n_e (gaussian of width 0 is the identity)."""
    rng = np.random.default_rng(6)
    groups = rand_groups(rng, 3, 4.0)
    x = so.build_pooled_ifr(groups, 3, 4.0, 0.02, 0.0)
    K = 200
    C = sum(np.histogram(g, bins=np.arange(K + 1) * 0.02)[0] for g in groups)
    assert np.all(np.isfinite(x))
    assert_allclose(x, (C / 3.0).astype(np.float32), rtol=1e-6)


def test_declared_duration(so):
    S = np.array([0.5, 1.0, 3.0])
    x = so.build_pooled_ifr([S], 9, 180.0, 0.02, 0.04)
    assert x.shape == (9000,)


def test_K_is_floor_of_T_over_dt(so):
    # SPEC: K = floor(T / Delta_t), evaluated as stated.
    for T, dt in [(180.0, 0.02), (0.3, 0.1), (1.0, 0.1), (7.3, 0.05), (12.0, 0.01)]:
        x = so.build_pooled_ifr([np.array([0.0])], 1, T, dt, 0.0 + 0.01)
        assert x.shape == (int(np.floor(T / dt)),)


def test_output_contract(so):
    rng = np.random.default_rng(7)
    x = so.build_pooled_ifr(rand_groups(rng, 9, 10.0), 9, 10.0, 0.01, 0.02)
    assert x.dtype == np.float32 and x.ndim == 1
    assert np.all(np.isfinite(x)) and np.all(x >= 0)


def test_inputs_not_mutated(so):
    rng = np.random.default_rng(8)
    groups = rand_groups(rng, 3, 5.0)
    copies = [g.copy() for g in groups]
    so.build_pooled_ifr(groups, 3, 5.0, 0.02, 0.04)
    for a, b in zip(groups, copies):
        assert_array_equal(a, b)


def test_parity_with_compute_ifr_trace(so, dsn_dir):
    from dataclasses import replace
    from generate_burst_data import CONTROL_PARAMS, compute_ifr_trace
    rng = np.random.default_rng(9)
    groups = rand_groups(rng, 9, 30.0)
    p = replace(CONTROL_PARAMS, duration_s=30.0, w_size=0.01, gaussian_window=0.02)
    ifr, fs = compute_ifr_trace(groups, p)
    x = so.build_pooled_ifr(groups, 9, 30.0, 0.01, 0.02)
    assert_array_equal(x, (ifr / 9.0).astype(np.float32))
    assert fs == pytest.approx(100.0)


# ---------------------------------------------------------------- windows
@pytest.mark.parametrize("K,W,S", [(100, 10, None), (100, 10, 3), (100, 10, 10),
                                   (99, 10, 7), (10, 10, None), (9, 10, None),
                                   (25, 1, 1), (31, 5, 50)])
def test_window_index_rule(so, K, W, S):
    x = np.arange(K, dtype=np.float32)
    Xw, starts = so.window_trace(x, W, S)
    s = W if S is None else S
    n_win = (K - W) // s + 1 if K >= W else 0
    assert Xw.shape == (n_win, W) and Xw.dtype == np.float32
    assert list(starts) == [i * s for i in range(n_win)]
    for i in range(n_win):
        assert_array_equal(Xw[i], x[i * s:i * s + W])


def test_window_multichannel_last_axis(so):
    x = np.arange(3 * 50, dtype=np.float64).reshape(3, 50)
    Xw, starts = so.window_trace(x, 20)
    assert Xw.shape == (2, 3, 20) and Xw.dtype == np.float32
    assert_array_equal(Xw[1], x[:, 20:40].astype(np.float32))


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("kw", [
    dict(spike_times_s=[np.array([0.1, np.nan])]),
    dict(spike_times_s=[np.array([0.1, np.inf])]),
    dict(spike_times_s=np.array([np.nan])),
    dict(T=np.nan), dict(T=np.inf), dict(T=0.0), dict(T=-1.0),
    dict(dt=np.nan), dict(dt=np.inf), dict(dt=0.0), dict(dt=-0.01),
    dict(T=0.01, dt=0.02),        # floor(T/dt) < 1
    dict(n_electrodes=0), dict(n_electrodes=-1),
    dict(sigma_sm=-0.01),
])
def test_invalid_inputs_raise(so, kw):
    args = dict(spike_times_s=[np.array([0.1, 0.2])], n_electrodes=1, T=1.0,
                dt=0.02, sigma_sm=0.04)
    args.update(kw)
    with pytest.raises(ValueError):
        so.build_pooled_ifr(**args)


def test_nan_sigma_raises(so):
    """SPEC: sigma_sm >= 0 [s]; Errors: 'sigma_sm < 0' -> ValueError and section 4
    'Failures raise'. A NaN sigma_sm is not >= 0; it must not slip past a
    comparison-based check and yield a non-finite x."""
    with pytest.raises(ValueError):
        so.build_pooled_ifr([np.array([0.1, 0.2])], 1, 1.0, 0.02, float("nan"))


@pytest.mark.parametrize("W,S", [(0, None), (-1, None), (5, 0), (5, -2)])
def test_window_invalid_raise(so, W, S):
    with pytest.raises(ValueError):
        so.window_trace(np.zeros(20, np.float32), W, S)
