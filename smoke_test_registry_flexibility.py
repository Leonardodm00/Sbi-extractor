"""
smoke_test_registry_flexibility.py -- the label chain does not know any counts.

    python3 smoke_test_registry_flexibility.py

Expect: ALL 14 CHECKS PASSED, and no --sim_dir. This suite needs neither the
simulator repository nor the DSN tree: it writes throwaway HPC_main_sweep /
HPC_single_run stubs into a temp directory and points load_registry at them,
so the registry width n, the log set L, the active set A and the topology
block can be varied freely and the invariants checked at each setting.

WHY THIS EXISTS
---------------
2026-09-20. smoke_test_sbi_export.py's T9 asserted |L| = 27, n = 36, |A| = 23
and 19 active log axes against literals. A bounds edit in the simulator moved
|L| to 26 and T9 failed -- reporting a change in PARAM_BOUNDS as if it were a
defect in the export, which is exactly backwards: sbi_labels derives L
"MECHANICALLY from PARAM_BOUNDS, never from a hard-coded list, so it tracks
any future bounds edit". That test also conflated two objects behind one
literal:

    |L|    = how many registry axes rule (1) calls log coordinates
    p      = |A| + |eta|, the label width

which both happened to equal 27 in the registry of record and are not the
same number. This suite pins the RELATIONS and never the counts, and does it
at several widths so that a count creeping back in is caught here rather than
on the cluster.

WHAT IS CHECKED
---------------
F1  a small registry (n = 6) loads and satisfies every T9 invariant
F2  a wide registry (n = 41) does too, with a different |L| and |A|
F3  |L| = 0 (no axis spans a decade) is legal
F4  |L| = n (every axis spans a decade) is legal
F5  rule (1)'s threshold is >= 1 decade: exactly 1.000 is IN, just under is OUT
F6  the p0_conn trap is reported, not asserted: a kernel box spanning less
    than a decade leaves the topology block linear and still PASSES
F7  a 3-axis topology block gives p = |A| + 3, i.e. p tracks the block
F8  NEGATIVE: a point-interval axis inside the sweep group is refused
F9  NEGATIVE: LOG_PARAMS disagreeing with PARAM_BOUNDS is refused at load
F10 NEGATIVE: PARAM_BOUNDS_THETA that is not the coordinate rule applied to
    PARAM_BOUNDS passes T9 and is caught by T9b
F11 NEGATIVE: PARAM_NAMES shorter than PARAM_BOUNDS is refused
F12 NEGATIVE: PARAM_UNITS shorter than PARAM_NAMES is refused
F13 the margin m_k = log10(hi_k/lo_k) - 1 restates rule (1): m_k >= 0 iff
    k is in L. An axis at EXACTLY one decade is IN, since (1) is >= and not
    >. Point intervals and non-positive bounds are reported, not dropped
F14 at_risk_axes finds the axes within eps of the threshold, and the
    active-index filter keeps only those whose flip would change an
    exported theta column

HPC note (hpc-python-compat): pure ASCII, LF-only. numpy only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
_PASS = []
_FAIL = []


def check(name, fn):
    try:
        detail = fn()
    except Exception as exc:                                 # noqa: BLE001
        _FAIL.append((name, "%s: %s" % (type(exc).__name__, exc)))
        print("  [FAIL] %-5s %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
        return
    _PASS.append((name, detail))
    print("  [PASS] %-5s %s" % (name, detail))


# --------------------------------------------------------------------------- #
# the stub simulator
# --------------------------------------------------------------------------- #
_SWEEP_TMPL = '''\
"""Throwaway stand-in for HPC_main_sweep, written by a smoke test."""
import numpy as np

PARAM_BOUNDS = np.asarray(%(bounds)r, dtype=np.float64)
PARAM_BOUNDS_THETA = np.asarray(%(bounds_theta)r, dtype=np.float64)
LOG_PARAMS = %(log_params)r
LOG_BASE = %(log_base)r
KERNEL_BOUNDS = np.asarray(%(kernel)r, dtype=np.float64)

# The coordinate transforms, the same rule the real simulator applies: ln on
# the LOG_PARAMS axes, identity elsewhere. Present so assertion A1
# (export_embeddings.run_assertion_A1) can run against a stub.
_LOG_MASK = np.zeros(PARAM_BOUNDS.shape[0], dtype=bool)
_LOG_MASK[list(LOG_PARAMS)] = True


def natural_to_theta(v):
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    out[_LOG_MASK] = np.log(v[_LOG_MASK])
    return out


def theta_to_natural(t):
    t = np.asarray(t, dtype=np.float64)
    out = t.copy()
    out[_LOG_MASK] = np.exp(t[_LOG_MASK])
    return out
'''

_RUN_TMPL = '''\
"""Throwaway stand-in for HPC_single_run, written by a smoke test."""
PARAM_NAMES = %(names)r
PARAM_UNITS = %(units)r
SWEEP_GROUPS = %(groups)r
'''


def rule_one_bounds(bounds):
    """Rule (1), independently of sbi_labels and of the test under test."""
    b = np.asarray(bounds, dtype=np.float64)
    return sorted(k for k in range(b.shape[0])
                  if b[k, 0] > 0.0 and b[k, 1] > 0.0
                  and np.log10(b[k, 1] / b[k, 0]) >= 1.0)


def theta_of(bounds, log_indices):
    """The coordinate rule applied to a natural box, for each fixed axis k."""
    b = np.asarray(bounds, dtype=np.float64)
    out = b.copy()
    for k in log_indices:
        out[k] = np.log(b[k])
    return out


def write_stub(root, bounds, active, kernel=((0.1, 1.0), (50.0, 400.0),
                                             (0.5, 3.0)),
               log_params=None, bounds_theta=None, names=None, units=None,
               log_base="natural", group="neuron_synapse"):
    """Write a (HPC_main_sweep, HPC_single_run) pair describing one registry.

    Every argument that defaults to None is DERIVED from `bounds`, so a
    positive case is consistent by construction and a negative case is made by
    overriding exactly one of them.
    """
    b = np.asarray(bounds, dtype=np.float64)
    n = b.shape[0]
    lp = rule_one_bounds(b) if log_params is None else list(log_params)
    bt = theta_of(b, rule_one_bounds(b)) if bounds_theta is None \
        else np.asarray(bounds_theta, dtype=np.float64)
    nm = ["ax%02d" % k for k in range(n)] if names is None else list(names)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "HPC_main_sweep.py"), "w") as fh:
        fh.write(_SWEEP_TMPL % {"bounds": b.tolist(),
                                "bounds_theta": bt.tolist(),
                                "log_params": lp,
                                "log_base": log_base,
                                "kernel": np.asarray(kernel,
                                                     dtype=np.float64).tolist()})
    un = ["(dimensionless)"] * len(nm) if units is None else list(units)
    with open(os.path.join(root, "HPC_single_run.py"), "w") as fh:
        fh.write(_RUN_TMPL % {"names": nm,
                              "units": un,
                              "groups": {group: list(active)}})
    return root


def _forget_stub(root):
    """Drop the stub from sys.modules and sys.path.

    load_registry does `import HPC_main_sweep`, so without this the FIRST stub
    would be cached and every later case would silently re-test it -- the
    failure mode that makes a parametrised import test worthless.
    """
    for mod in ("HPC_main_sweep", "HPC_single_run"):
        sys.modules.pop(mod, None)
    root = os.path.abspath(root)
    while root in sys.path:
        sys.path.remove(root)


class _Case(object):
    """A stub directory that is torn down, and forgotten, on exit."""

    def __init__(self, **kw):
        self.kw = kw
        self.root = None

    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="regflex_")
        write_stub(self.root, **self.kw)
        return self.root

    def __exit__(self, *exc):
        _forget_stub(self.root)
        shutil.rmtree(self.root, ignore_errors=True)
        return False


# --------------------------------------------------------------------------- #
# bounds builders
# --------------------------------------------------------------------------- #
def decade_box(n, log_mask, lo=2.0):
    """(n, 2) natural bounds: axis k spans 2 decades if log_mask[k] else 1.5x."""
    b = np.zeros((n, 2), dtype=np.float64)
    for k in range(n):
        b[k, 0] = lo + k
        b[k, 1] = b[k, 0] * (100.0 if log_mask[k] else 1.5)
    return b


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def _invariants(root, group="neuron_synapse"):
    """Run the two tests under test and return T9's report line."""
    from smoke_test_sbi_export import (test_T9_registry_invariants,
                                       test_T9b_coordinate_map)
    line = test_T9_registry_invariants(root, sweep_group=group)
    test_T9b_coordinate_map(root, sweep_group=group)
    return line


def f1_small():
    mask = [True, False, True, False, False, False]
    with _Case(bounds=decade_box(6, mask), active=[0, 1, 3]) as root:
        line = _invariants(root)
    if "n=6 |L|=2" not in line or "p=7=3+4" not in line:
        raise AssertionError("report line %r" % line)
    return line


def f2_wide():
    n = 41
    mask = [(k % 3 == 0) for k in range(n)]
    active = [k for k in range(n) if k % 2 == 0][:17]
    with _Case(bounds=decade_box(n, mask), active=active) as root:
        line = _invariants(root)
    if "n=41" not in line or "p=21=17+4" not in line:
        raise AssertionError("report line %r" % line)
    return line


def f3_no_log():
    n = 9
    with _Case(bounds=decade_box(n, [False] * n), active=[1, 2, 3]) as root:
        line = _invariants(root)
    if "|L|=0" not in line or "0 ln + 3 linear" not in line:
        raise AssertionError("report line %r" % line)
    return line


def f4_all_log():
    n = 9
    with _Case(bounds=decade_box(n, [True] * n), active=[1, 2, 3]) as root:
        line = _invariants(root)
    if "|L|=9" not in line or "3 ln + 0 linear" not in line:
        raise AssertionError("report line %r" % line)
    return line


def f5_threshold():
    # axis 0 spans EXACTLY one decade (in L, since the rule is >= 1);
    # axis 1 spans 0.9999 decades (out of L); axis 2 spans two decades.
    b = np.asarray([[1.0, 10.0],
                    [1.0, 10.0 ** 0.9999],
                    [1.0, 100.0],
                    [1.0, 1.5]], dtype=np.float64)
    with _Case(bounds=b, active=[0, 1, 2]) as root:
        from sbi_labels import load_registry
        reg = load_registry(root)
        if reg.log_param_indices != [0, 2]:
            raise AssertionError("L = %r, expected [0, 2]"
                                 % reg.log_param_indices)
        line = _invariants(root)
    if "|L|=2" not in line:
        raise AssertionError("report line %r" % line)
    return "exactly 1.000 decade is IN L, 0.9999 is OUT; " + line.split(";")[0]


def f6_trap_dormant():
    # p0_conn spanning half a decade: the trap is NOT live, and that must be
    # reported rather than failing the way the old 'expected exactly 1' did.
    n = 5
    with _Case(bounds=decade_box(n, [True, False, True, False, False]),
               active=[0, 1, 2],
               kernel=((0.2, 0.632), (50.0, 400.0), (0.5, 3.0))) as root:
        line = _invariants(root)
    if "would NOT misclassify" not in line:
        raise AssertionError("trap not reported as dormant: %r" % line)
    return line.split("; ")[-1]


def f7_three_axis_topology():
    from sbi_labels import load_registry, build_label_spec
    n = 8
    with _Case(bounds=decade_box(n, [True] * 4 + [False] * 4),
               active=[0, 1, 2, 5]) as root:
        reg = load_registry(root)
        spec = build_label_spec(reg, reg.sweep_groups["neuron_synapse"],
                                "neuron_synapse", conn_prob_bounds=(0.1, 0.6),
                                topology_axes=["p0_conn", "d0_conn",
                                               "beta_conn"])
        if spec.p != 4 + 3:
            raise AssertionError("p = %d, expected len(A)+3 = 7" % spec.p)
        if spec.coord[-3:] != ["linear"] * 3:
            raise AssertionError("topology block not linear: %r" % spec.coord[-3:])
        if spec.param_names[-3:] != ["p0_conn", "d0_conn", "beta_conn"]:
            raise AssertionError("topology order: %r" % spec.param_names[-3:])
    return "p = |A| + |eta| = 4 + 3 = 7 with conn_prob excluded"


def f13_margin_restates_rule_one():
    # Bounds with known margins, including one EXACTLY on the threshold and
    # one just inside eps on the other side, plus the two undefined cases.
    b = np.asarray([[1.0, 10.0],     # 1.000000 dec, margin  0.000000, ln, RISK
                    [1.0, 100.0],    # 2.000000 dec, margin +1.000000, ln
                    [1.0, 2.0],      # 0.301030 dec, margin -0.698970, linear
                    [1.0, 9.0],      # 0.954243 dec, margin -0.045757, RISK
                    [5.0, 5.0],      # point interval -> undefined
                    [-2.0, 3.0]],    # non-positive lo -> undefined
                   dtype=np.float64)
    with _Case(bounds=b, active=[0, 1, 2]) as root:
        from sbi_labels import load_registry, coordinate_margins
        reg = load_registry(root)
        m = coordinate_margins(reg)
        L = set(reg.log_param_indices)
        # (2) restates (1): m_k >= 0 iff k in L, for every defined margin.
        for a in m:
            if a.margin is None:
                continue
            if (a.margin >= 0.0) != (a.index in L):
                raise AssertionError("axis %s margin %+.6f but in L = %s"
                                     % (a.name, a.margin, a.index in L))
        defined = [a for a in m if a.margin is not None]
        undef = [a for a in m if a.margin is None]
        if [a.name for a in defined] != ["ax00", "ax03", "ax02", "ax01"]:
            raise AssertionError("not sorted by |margin|: %r"
                                 % ([a.name for a in defined],))
        if abs(defined[0].margin) != 0.0:
            raise AssertionError("ax00 margin is %r, expected exactly 0.0"
                                 % (defined[0].margin,))
        if defined[0].coord != "ln":
            raise AssertionError("an axis at exactly 1 decade must be in L "
                                 "(rule (1) is >=), got %r" % defined[0].coord)
        if not np.isclose(defined[1].margin, np.log10(9.0) - 1.0):
            raise AssertionError("ax03 margin %r" % (defined[1].margin,))
        if len(undef) != 2 or not all(a.why_undefined for a in undef):
            raise AssertionError("undefined axes not reported: %r"
                                 % ([(a.name, a.why_undefined) for a in undef],))
    return ("margin >= 0 iff in L on 4 defined axes; exactly-1-decade is IN; "
            "2 undefined axes reported, not dropped")


def f14_at_risk_and_the_active_filter():
    b = np.asarray([[1.0, 10.0],     # margin  0.000000  -> at risk, ACTIVE
                    [1.0, 100.0],
                    [1.0, 2.0],
                    [1.0, 9.0],      # margin -0.045757  -> at risk, inactive
                    [5.0, 5.0],
                    [-2.0, 3.0]], dtype=np.float64)
    with _Case(bounds=b, active=[0, 1, 2]) as root:
        from sbi_labels import load_registry, at_risk_axes, format_margins
        reg = load_registry(root)
        allrisk = [a.name for a in at_risk_axes(reg)]
        if allrisk != ["ax00", "ax03"]:
            raise AssertionError("at_risk_axes gave %r" % (allrisk,))
        act = [a.name for a in at_risk_axes(reg, active_indices=[0, 1, 2])]
        if act != ["ax00"]:
            raise AssertionError("the active filter gave %r" % (act,))
        tight = [a.name for a in at_risk_axes(reg, eps=0.001)]
        if tight != ["ax00"]:
            raise AssertionError("eps=0.001 gave %r" % (tight,))
        tbl = format_margins(reg, active_indices=[0, 1, 2])
        if "ACTIVE" not in tbl or "ax00" not in tbl:
            raise AssertionError("format_margins lost the ACTIVE column")
    return ("2 axes at risk, 1 of them active; eps=0.001 narrows to the "
            "exactly-on-threshold one")


def _expect(fn, exc_type, needle):
    try:
        fn()
    except exc_type as exc:
        if needle not in str(exc):
            raise AssertionError("raised %s but without %r: %s"
                                 % (exc_type.__name__, needle, exc))
        return str(exc).splitlines()[0][:78]
    raise AssertionError("expected %s mentioning %r, nothing raised"
                         % (exc_type.__name__, needle))


def f8_frozen_axis_active():
    b = decade_box(5, [True, False, True, False, False])
    b[3] = [7.0, 7.0]                      # a point interval, made active
    with _Case(bounds=b, active=[0, 1, 3]) as root:
        return _expect(lambda: _invariants(root), AssertionError, "ax03")


def f9_log_params_drift():
    b = decade_box(5, [True, False, True, False, False])
    with _Case(bounds=b, active=[0, 1, 2], log_params=[0]) as root:
        return _expect(lambda: _invariants(root), RuntimeError, "LOG_PARAMS")


def f10_theta_bounds_drift():
    from smoke_test_sbi_export import (test_T9_registry_invariants,
                                       test_T9b_coordinate_map)
    b = decade_box(5, [True, False, True, False, False])
    bt = theta_of(b, rule_one_bounds(b))
    bt[2] = b[2]                           # a log axis left in natural units
    with _Case(bounds=b, active=[0, 1, 2], bounds_theta=bt) as root:
        # T9 must still pass: it says nothing about PARAM_BOUNDS_THETA.
        test_T9_registry_invariants(root)
        msg = _expect(lambda: test_T9b_coordinate_map(root),
                      AssertionError, "ax02")
    return "T9 passes, T9b catches it -- " + msg


def f11_bounds_width_mismatch():
    # PARAM_NAMES is 5 long, PARAM_BOUNDS is 6: n comes from the names, so the
    # bounds-shape assertion is the one that must fire.
    b = decade_box(6, [True, False, True, False, False, False])
    with _Case(bounds=b, active=[0, 1],
               names=["ax%02d" % k for k in range(5)],
               units=["(dimensionless)"] * 5) as root:
        return _expect(lambda: _invariants(root), AssertionError,
                       "param_bounds has shape")


def f12_units_width_mismatch():
    b = decade_box(5, [True, False, True, False, False])
    with _Case(bounds=b, active=[0, 1, 2],
               units=["(dimensionless)"] * 4) as root:
        return _expect(lambda: _invariants(root), AssertionError,
                       "param_units has 4 entries")


def main():
    print("smoke_test_registry_flexibility -- no sim repo, no DSN tree needed")
    print("-" * 70)
    check("F1", f1_small)
    check("F2", f2_wide)
    check("F3", f3_no_log)
    check("F4", f4_all_log)
    check("F5", f5_threshold)
    check("F6", f6_trap_dormant)
    check("F7", f7_three_axis_topology)
    check("F8", f8_frozen_axis_active)
    check("F9", f9_log_params_drift)
    check("F10", f10_theta_bounds_drift)
    check("F11", f11_bounds_width_mismatch)
    check("F12", f12_units_width_mismatch)
    check("F13", f13_margin_restates_rule_one)
    check("F14", f14_at_risk_and_the_active_filter)
    print("-" * 70)
    if _FAIL:
        print("FAILED %d of %d" % (len(_FAIL), len(_PASS) + len(_FAIL)))
        for n, d in _FAIL:
            print("  %-5s %s" % (n, d))
        return 1
    print("ALL %d CHECKS PASSED" % len(_PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
