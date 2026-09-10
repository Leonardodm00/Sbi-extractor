#!/usr/bin/env python3
"""
smoke_test_resolved_n_e.py

The sidecar must record the n_e the export ACTUALLY used, not the value of a
flag that is normally omitted. Before this fix a shard exported without
--n_electrodes wrote "n_electrodes": null, even though the pooled IFR had been
correctly divided by n_e read from electrode_centers -- leaving the shard
unable to be checked for preprocessing parity against the real arm afterwards.

Run:
    python3 smoke_test_resolved_n_e.py
    SMOKE_VERBOSE=1 python3 smoke_test_resolved_n_e.py

Expect: ALL 8 CHECKS PASSED
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import traceback
import types

# example_export imports dsn_frozen, which imports torch. None of the code
# under test here touches torch: record_resolved_n_e is pure dict manipulation
# and the rest is static source inspection. Under sbi_export the real torch is
# present and imported normally; on a bare checkout a stub keeps this file
# runnable rather than skipping the checks. Which path was taken is printed, so
# a stubbed run can never be mistaken for a full one.
try:
    import example_export as EX
    _IMPORT_MODE = "real torch"
except ImportError:
    _t = types.ModuleType("torch")
    _t.Tensor = object
    _t.nn = types.ModuleType("torch.nn")
    _t.nn.functional = types.ModuleType("torch.nn.functional")
    _t.no_grad = lambda *a, **k: (lambda f: f)
    sys.modules.setdefault("torch", _t)
    sys.modules.setdefault("torch.nn", _t.nn)
    sys.modules.setdefault("torch.nn.functional", _t.nn.functional)
    import example_export as EX
    _IMPORT_MODE = "STUBBED torch (no torch in this environment)"

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


def main():
    print("  import mode: %s" % _IMPORT_MODE)

    def T1():
        obs = {"electrode_forward_model": True}
        EX.record_resolved_n_e(obs, 1, "electrode_centers")
        assert obs["n_electrodes"] == 1, obs
        assert obs["n_electrodes_source"] == "electrode_centers", obs
    check("T1 the resolved n_e and its source are written", T1)

    def T2():
        obs = {}
        EX.record_resolved_n_e(obs, 9, "--n_electrodes")
        assert obs["n_electrodes"] == 9 and isinstance(obs["n_electrodes"], int)
        assert obs["n_electrodes_source"] == "--n_electrodes"
    check("T2 an explicit flag is recorded as such, not as a guess", T2)

    def T3():
        obs = {}
        for _ in range(5):
            EX.record_resolved_n_e(obs, 4, "electrode_centers")
        assert obs["n_electrodes"] == 4, obs
    check("T3 repeated identical calls are idempotent", T3)

    def T4():
        obs = {}
        EX.record_resolved_n_e(obs, 4, "electrode_centers")
        try:
            EX.record_resolved_n_e(obs, 1, "electrode_centers")
        except ValueError as exc:
            assert "not constant within this shard" in str(exc), str(exc)
            return
        raise AssertionError("a mixed-n_e shard must be refused")
    check("T4 n_e changing mid-shard raises instead of silently overwriting", T4)

    def T5():
        EX.record_resolved_n_e(None, 1, "electrode_centers")   # must not raise
    check("T5 observable_out=None is a no-op (synthetic mode)", T5)

    def T6():
        sig = inspect.signature(EX.iter_campaign_records)
        assert "observable_out" in sig.parameters, list(sig.parameters)
        assert sig.parameters["observable_out"].default is None
    check("T6 iter_campaign_records accepts observable_out, defaulting to None", T6)

    # --- the ordering contract this design depends on ---------------------
    # extra["observable"] is filled DURING generator consumption, and merged
    # into the sidecar afterwards. If export_embeddings ever merged
    # extra_sidecar before consuming the records, the resolved value would be
    # lost and the sidecar would silently go back to null. Pin it statically:
    # no fixture can catch a future reordering, but the source can.
    def T7():
        src = open(os.path.join(os.path.dirname(os.path.abspath(EX.__file__)),
                                "export_embeddings.py")).read()
        i_iter = src.index("for rec in records")
        i_merge = src.index("if extra_sidecar:")
        assert i_iter < i_merge, (
            "export_embeddings merges extra_sidecar at offset %d but consumes "
            "the records at %d. The resolved n_e is written during "
            "consumption, so merging first would drop it." % (i_merge, i_iter))
    check("T7 export_embeddings consumes records BEFORE merging extra_sidecar", T7)

    def T8():
        """extra must be built before the generator, or referencing
        extra['observable'] in the call raises NameError at run time."""
        src = open(os.path.abspath(EX.__file__)).read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "main")
        assign_line = min(
            n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "extra" for t in n.targets))
        use_line = min(
            n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.keyword) and n.arg == "observable_out")
        assert assign_line < use_line, (
            "extra is assigned at line %d but observable_out=extra[...] is "
            "used at line %d" % (assign_line, use_line))
    check("T8 extra is assigned before observable_out=extra[...] is referenced", T8)

    n = len(PASSED) + len(FAILED)
    print("\n%s  %d/%d checks passed"
          % ("ALL %d CHECKS PASSED" % n if not FAILED else "FAILURES",
             len(PASSED), n))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
