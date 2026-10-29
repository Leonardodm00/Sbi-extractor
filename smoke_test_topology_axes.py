"""Smoke test for the topology axis stoa_gate_p -- Block 2 in specs/SPEC.md
(the 2026-10-29 additions, decision D-107), and the known axes of
preflight_label_axes.py (out of coverage; it reads them from sbi_labels).

Covers sbi_labels: the constants KNOWN_TOPOLOGY_AXES, JOB_ARGS_TOPOLOGY_BOUNDS
and DETECTION_TOPOLOGY_AXES, topology_bounds_from_job_args, and
build_label_spec / assemble_theta_A with topology_bounds; and
preflight_label_axes.py run as a script on small synthetic campaign trees
written to a temporary folder (a gated campaign, a gate-less one, a fixed
gate). Everything runs on a stand-in registry built here: no simulator, no
cluster file, no torch.

Run (from the Sbi-extractor root, any environment with numpy):
    python3 smoke_test_topology_axes.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table; exit
code 1 if any check failed, errored or is TODO. The temporary folders are
removed at exit.

Keep this file pure ASCII and LF only (hpc-python-compat).
"""
from __future__ import annotations

import atexit
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np
from numpy.testing import assert_allclose  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sbi_labels as SL               # noqa: E402

SEED = 20261029
REPORT_PACKAGES = ("numpy",)
PREFLIGHT = os.path.join(HERE, "preflight_label_axes.py")
GATE_BOX = (0.2, 1.0)                 # the MEAMA launcher's default [STOA_GATE_LO, STOA_GATE_HI]
_TMP = []


# --------------------------------------------------------------- helpers ---

def stand_in_registry():
    """A 5-axis registry: axes 0-2 linear on [0.5, 2.0], axis 3 frozen
    (lo == hi), axis 4 'O_N' a log axis on [0.03, 3.0] (two decades)."""
    names = ["a0", "a1", "a2", "frozen", "O_N"]
    bounds = np.array([[0.5, 2.0], [0.5, 2.0], [0.5, 2.0], [1.0, 1.0], [0.03, 3.0]])
    theta = bounds.copy()
    theta[4] = np.log(bounds[4])
    return SL.Registry(param_names=names, param_units=["a.u."] * 5, param_bounds=bounds,
                       param_bounds_theta=theta, log_param_indices=[4], log_base="natural",
                       kernel_bounds=np.array([[0.1, 1.0], [20.0, 200.0], [0.5, 3.0]]),
                       sweep_groups={"g": [0, 1, 2, 4]})


ACTIVE = [0, 1, 2, 4]


def expect_error(exc_type, fragment, func, *args, **kwargs):
    """func(*args) must raise exc_type with fragment in its message."""
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        assert fragment in str(exc), "message %r lacks %r" % (str(exc), fragment)
        return
    raise AssertionError("%s did not raise %s (%s)" % (func.__name__, exc_type.__name__,
                                                       fragment))


def tmpdir():
    d = tempfile.mkdtemp(prefix="topology_axes_")
    _TMP.append(d)
    return d


@atexit.register
def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def write_campaign(sim_main, name, gates, conn_rule="flat", job_args_extra=None):
    """A campaign of one sweep task, one topology per entry of gates (None: no
    stoa_gate_p key), two iterations per topology, as HPC_main_sweep writes them
    (the keys the preflight scan reads)."""
    sd = os.path.join(sim_main, name, "sweep_cpu_task0000")
    os.makedirs(sd)
    ja = {"simtime": 200.0, "conn_rule": conn_rule, "conn_prob_lo": 0.1, "conn_prob_hi": 0.6}
    ja.update(job_args_extra or {})
    with open(os.path.join(sd, "job_args.json"), "w") as fh:
        json.dump(ja, fh)
    rng = np.random.default_rng(SEED)
    for k, g in enumerate(gates):
        td = os.path.join(sd, "topo_%05d" % k)
        os.makedirs(td)
        cp = float(rng.uniform(0.1, 0.6))
        for i in range(2):
            extra = {} if g is None else {"stoa_gate_p": np.float64(g)}
            np.savez_compressed(os.path.join(td, "iter_%05d.npz" % i),
                                conn_rule=np.array(conn_rule), conn_prob=np.float64(cp),
                                topo_idx=np.int32(k), seed_run=np.int64(1000 * k + i),
                                theta=np.zeros(5), params=np.ones(5), **extra)
    return sd


def run_preflight(sim_main, *extra):
    out = os.path.join(sim_main, "freeze.json")
    r = subprocess.run([sys.executable, PREFLIGHT, "--sim_main", sim_main,
                        "--campaigns", "campaign_*", "--require-conn-rule", "flat",
                        "--out", out] + list(extra),
                       capture_output=True, text=True, timeout=300, cwd=tempfile.gettempdir())
    doc = None
    if r.returncode == 0 and os.path.isfile(out):
        with open(out) as fh:
            doc = json.load(fh)
    return r.returncode, (r.stdout + r.stderr).splitlines(), doc


# ---------------------------------------------------------------- checks ---

def test_2_constants():
    assert SL.TOPOLOGY_AXES == ("conn_prob", "p0_conn", "d0_conn", "beta_conn")   # unchanged
    assert SL.JOB_ARGS_TOPOLOGY_BOUNDS == {"stoa_gate_p": ("stoa_gate_lo", "stoa_gate_hi")}
    assert SL.KNOWN_TOPOLOGY_AXES == SL.TOPOLOGY_AXES + ("stoa_gate_p",)
    assert SL.DETECTION_TOPOLOGY_AXES == ("conn_prob",)
    assert SL.TOPOLOGY_UNITS["stoa_gate_p"] == "(dimensionless)"
    assert set(SL.KNOWN_TOPOLOGY_AXES) <= set(SL.TOPOLOGY_UNITS)


def test_2_known_answer_gate_label():
    """p = |A| + 2; the gate's row is its job_args box, linear, dimensionless."""
    reg = stand_in_registry()
    spec = SL.build_label_spec(reg, ACTIVE, "g", conn_prob_bounds=(0.05, 0.4),
                               topology_axes=["conn_prob", "stoa_gate_p"],
                               topology_bounds={"stoa_gate_p": GATE_BOX})
    assert spec.p == len(ACTIVE) + 2 == 6, spec.p
    assert spec.param_names == ["a0", "a1", "a2", "O_N", "conn_prob", "stoa_gate_p"]
    assert spec.coord == ["linear", "linear", "linear", "ln", "linear", "linear"], spec.coord
    assert spec.units[-1] == "(dimensionless)"
    assert np.array_equal(spec.bounds_theta[-1], list(GATE_BOX))          # copied, exact
    assert np.array_equal(spec.bounds_theta[-2], [0.05, 0.4])
    assert np.array_equal(spec.bounds_theta[3], np.log([0.03, 3.0]))
    theta_reg = np.array([0.7, 0.8, 0.9, 1.0, np.log(0.3)])
    params = np.array([0.7, 0.8, 0.9, 1.0, 0.3])
    row = SL.assemble_theta_A(spec, theta_reg, {"conn_prob": 0.2, "stoa_gate_p": 0.37,
                                                "ignored": 9.0}, params_registry=params)
    assert row.tolist() == [0.7, 0.8, 0.9, np.log(0.3), 0.2, 0.37], row


def test_2_reference_legacy_unchanged():
    """The default call (no topology_axes) is the legacy 4-axis block, and bounds
    for an axis outside topology_axes change nothing."""
    reg = stand_in_registry()
    legacy = SL.build_label_spec(reg, ACTIVE, "g", conn_prob_bounds=(0.05, 0.4))
    assert legacy.p == len(ACTIVE) + 4 and legacy.topology_axes == list(SL.TOPOLOGY_AXES)
    assert np.array_equal(legacy.bounds_theta[-3:], reg.kernel_bounds)
    a = SL.build_label_spec(reg, ACTIVE, "g", conn_prob_bounds=(0.05, 0.4),
                            topology_axes=["conn_prob"])
    b = SL.build_label_spec(reg, ACTIVE, "g", conn_prob_bounds=(0.05, 0.4),
                            topology_axes=["conn_prob"],
                            topology_bounds={"stoa_gate_p": GATE_BOX})
    assert (a.param_names, a.coord, a.units, a.topology_axes) == (
        b.param_names, b.coord, b.units, b.topology_axes)
    assert np.array_equal(a.bounds_theta, b.bounds_theta)


def test_2_job_args_helper():
    ja = {"stoa_gate_lo": 0.2, "stoa_gate_hi": 1.0, "conn_prob_lo": 0.1}
    assert SL.topology_bounds_from_job_args(ja, ["conn_prob", "stoa_gate_p"]) == {
        "stoa_gate_p": (0.2, 1.0)}
    assert SL.topology_bounds_from_job_args({"stoa_gate_lo": 1}, ["conn_prob"]) == {}
    assert SL.topology_bounds_from_job_args({}, []) == {}
    ok = SL.topology_bounds_from_job_args({"stoa_gate_lo": 0, "stoa_gate_hi": 1},
                                          ["stoa_gate_p"])
    assert ok == {"stoa_gate_p": (0.0, 1.0)} and all(type(v) is float for v in ok["stoa_gate_p"])
    for bad in ("missing", "absent", None, True, "0.2", float("nan"), float("inf")):
        ja = {"stoa_gate_lo": 0.2, "stoa_gate_hi": 1.0}
        if bad == "missing":
            del ja["stoa_gate_hi"]
        else:
            ja["stoa_gate_hi"] = bad
        expect_error(ValueError, "'stoa_gate_hi'", SL.topology_bounds_from_job_args, ja,
                     ["stoa_gate_p"])


def test_2_edge_cases():
    reg = stand_in_registry()
    kw = dict(sweep_group="g", conn_prob_bounds=(0.05, 0.4))
    expect_error(ValueError, "unknown topology axis/axes ['stoa_gate_p']",
                 SL.build_label_spec, reg, ACTIVE, topology_axes=["conn_prob", "stoa_gate_p"],
                 **kw)
    expect_error(ValueError, "comes from conn_prob_bounds", SL.build_label_spec, reg, ACTIVE,
                 topology_axes=["conn_prob"], topology_bounds={"conn_prob": (0.1, 0.2)}, **kw)
    expect_error(ValueError, "comes from kernel_bounds", SL.build_label_spec, reg, ACTIVE,
                 topology_axes=["conn_prob"], topology_bounds={"d0_conn": (1.0, 2.0)}, **kw)
    expect_error(ValueError, "no unit", SL.build_label_spec, reg, ACTIVE,
                 topology_axes=["conn_prob"], topology_bounds={"gate_x": (0.0, 1.0)}, **kw)
    for pair in ((0.2, float("nan")), (float("-inf"), 1.0), (0.2, True)):
        expect_error(ValueError, "not two finite numbers", SL.build_label_spec, reg, ACTIVE,
                     topology_axes=["conn_prob", "stoa_gate_p"],
                     topology_bounds={"stoa_gate_p": pair}, **kw)
    expect_error(ValueError, "not a (lo, hi) pair", SL.build_label_spec, reg, ACTIVE,
                 topology_axes=["conn_prob", "stoa_gate_p"],
                 topology_bounds={"stoa_gate_p": (0.2, 0.5, 1.0)}, **kw)
    expect_error(ValueError, "A3", SL.build_label_spec, reg, ACTIVE,
                 topology_axes=["conn_prob", "stoa_gate_p"],
                 topology_bounds={"stoa_gate_p": (0.7, 0.7)}, **kw)
    spec = SL.build_label_spec(reg, ACTIVE, topology_axes=["conn_prob", "stoa_gate_p"],
                               topology_bounds={"stoa_gate_p": GATE_BOX}, **kw)
    theta_reg = np.array([0.7, 0.8, 0.9, 1.0, np.log(0.3)])
    expect_error(KeyError, "stoa_gate_p", SL.assemble_theta_A, spec, theta_reg,
                 {"conn_prob": 0.2})
    expect_error(ValueError, "NaN or Inf", SL.assemble_theta_A, spec, theta_reg,
                 {"conn_prob": 0.2, "stoa_gate_p": float("nan")})


def test_2_contract():
    reg = stand_in_registry()
    spec = SL.build_label_spec(reg, ACTIVE, "g", conn_prob_bounds=(0.05, 0.4),
                               topology_axes=["stoa_gate_p", "conn_prob"],
                               topology_bounds={"stoa_gate_p": GATE_BOX})
    assert spec.bounds_theta.shape == (6, 2) and spec.bounds_theta.dtype == np.float64
    assert spec.param_names[-2:] == ["stoa_gate_p", "conn_prob"]       # the freeze's order
    assert np.array_equal(spec.bounds_theta[-2:], [list(GATE_BOX), [0.05, 0.4]])
    row = SL.assemble_theta_A(spec, np.zeros(5), {"conn_prob": 0.2, "stoa_gate_p": 0.37})
    assert row.shape == (6,) and row.dtype == np.float64 and row[-2:].tolist() == [0.37, 0.2]


def test_preflight_known_axes():
    """The scan's known axes are sbi_labels': the gate is a candidate, not a NOTE."""
    sim = tmpdir()
    write_campaign(sim, "campaign_meama_x_n01", gates=[0.31, 0.52, 0.77],
                   job_args_extra={"stoa_gate_lo": 0.2, "stoa_gate_hi": 1.0})
    rc, out, doc = run_preflight(sim)
    assert rc == 0 and doc is not None, out[-5:]
    assert doc["topology_axes"] == ["conn_prob", "stoa_gate_p"], doc["topology_axes"]
    assert doc["axis_stats"]["stoa_gate_p"] == {"n_distinct": 3, "min": 0.31, "max": 0.77}
    notes = [ln for ln in out if ln.startswith("NOTE:")]
    assert all("stoa_gate_p" not in ln for ln in notes), notes
    assert any(ln.split()[:1] == ["stoa_gate_p"] and "in theta" in ln for ln in out), out


def test_preflight_gate_absent_or_fixed():
    sim = tmpdir()
    write_campaign(sim, "campaign_cadex_hhgap_v1", gates=[None, None])
    rc, out, doc = run_preflight(sim)
    assert rc == 0 and doc["topology_axes"] == ["conn_prob"], out[-5:]
    assert "stoa_gate_p" not in doc["axis_stats"] and "stoa_gate_p" not in doc["excluded_axes"]
    assert any(ln.split()[:1] == ["stoa_gate_p"] and "ABSENT" in ln for ln in out), out
    sim = tmpdir()                        # a fixed gate (lo == hi): a delta prior, excluded
    write_campaign(sim, "campaign_meama_y_n01", gates=[0.7, 0.7],
                   job_args_extra={"stoa_gate_lo": 0.7, "stoa_gate_hi": 0.7})
    rc, out, doc = run_preflight(sim)
    assert rc == 0 and doc["topology_axes"] == ["conn_prob"], out[-5:]
    assert "constant across the whole export" in doc["excluded_axes"]["stoa_gate_p"]["reason"]
    sim = tmpdir()                        # declared inert by the user: excluded with the reason
    write_campaign(sim, "campaign_meama_z_n01", gates=[0.3, 0.6])
    rc, out, doc = run_preflight(sim, "--exclude", "stoa_gate_p=test reason")
    assert rc == 0 and doc["excluded_axes"]["stoa_gate_p"]["reason"] == "test reason", out[-5:]
    assert doc["topology_axes"] == ["conn_prob"]


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_determinism():
    raise unittest.SkipTest("deterministic bookkeeping: no randomness in the code under test")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds,
                                           lines[0] if lines else "")).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
