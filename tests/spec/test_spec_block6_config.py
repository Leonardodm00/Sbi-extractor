"""Spec-derived tests of Block 6.1 for the two profile keys added by Blocks 9 and
10 (cohort_map, drop_partial_topologies). Tester-written from specs/SPEC.md,
Block 6.1 ("cohort_map -- an absolute path or null ... drop_partial_topologies --
a bool"; "ConfigError for an unknown or missing key"), and contract amendment 2 of
2026-11-17 (F1/S1: cohort_map is the one path key not made absolute; a relative or
empty value is refused at load time). Extended 2026-11-17_1605 (commit 5b61d6c).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

import _bank_helpers as H


def _cfg():
    import bank_plan_config as CFG      # noqa: E402
    return CFG


@pytest.fixture(scope="module")
def base_profile(tmp_path_factory):
    """A valid profile (the fixture's paths; no file is opened by load_profile)."""
    root = tmp_path_factory.mktemp("cfg")
    fx = {"root": str(root), "record": "/x/rec.json", "freeze": "/x/frz.json",
          "manifest": "/x/man.json", "digests": {"record": "a" * 64, "freeze": "b" * 64,
                                                  "manifest": "c" * 64}}
    path = str(root / "base.json")
    H.fixture_module().write_profile(fx, path)
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def _load(tmp_path, prof):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(prof), encoding="ascii")
    return _cfg().load_profile(str(p))


def test_giulia_profile_values():
    """C1: Giulia: cohort_map null, drop_partial_topologies false (Block 6.1)."""
    cfg = _cfg().load_profile("giulia")
    assert cfg.cohort_map is None
    assert cfg.drop_partial_topologies is False


@pytest.mark.parametrize("value", [0, 1, "false", "true", None, 1.0])
def test_drop_partial_topologies_must_be_a_bool(tmp_path, base_profile, value):
    """C2."""
    prof = dict(base_profile, drop_partial_topologies=value)
    with pytest.raises(_cfg().ConfigError):
        _load(tmp_path, prof)


@pytest.mark.parametrize("value", [True, False])
def test_drop_partial_topologies_bools_load(tmp_path, base_profile, value):
    assert _load(tmp_path, dict(base_profile, drop_partial_topologies=value)).drop_partial_topologies is value


@pytest.mark.parametrize("value", [3, ["a.csv"], {"p": "a.csv"}, True])
def test_cohort_map_must_be_a_path_or_null(tmp_path, base_profile, value):
    """C3: a number, a list, a dict, a bool are refused."""
    with pytest.raises(_cfg().ConfigError):
        _load(tmp_path, dict(base_profile, cohort_map=value))


def test_cohort_map_absolute_path_and_null_load(tmp_path, base_profile):
    assert _load(tmp_path, dict(base_profile, cohort_map=None)).cohort_map is None
    assert _load(tmp_path, dict(base_profile, cohort_map="/abs/map.csv")).cohort_map == "/abs/map.csv"


# --- contract amendment 2 (2026-11-17, F1/S1): cohort_map is NOT made absolute against
# the repository root; a relative or empty value is refused when the profile is loaded,
# with ConfigError `cohort_map must be an absolute path or null, got <value>`.
# (2026-11-17_1605: the strict xfail of report 2026-11-17_1500 replaced by these.)

AMEND2_MSG = "cohort_map must be an absolute path or null, got "


@pytest.mark.parametrize("value", ["maps/linddup_cohort_map.csv", "", "x.csv", "./x.csv",
                                   "../x.csv", "~/x.csv", " ", " /abs/x.csv"])
def test_cohort_map_relative_or_empty_refused_at_load(tmp_path, base_profile, value):
    """C3, C4: every value os.path.isabs rejects (relative, empty, blank, a leading
    blank before "/", "~" unexpanded) raises ConfigError from load_profile itself, and
    the message is amendment 2's, naming the value (as given, or its repr)."""
    with pytest.raises(_cfg().ConfigError) as ei:
        _load(tmp_path, dict(base_profile, cohort_map=value))
    msg = str(ei.value)
    assert msg in (AMEND2_MSG + value, AMEND2_MSG + repr(value)), msg


@pytest.mark.parametrize("repo_root", [None, "/some/other/root"])
def test_cohort_map_absolute_kept_as_given_and_never_resolved(tmp_path, base_profile, repo_root):
    """C2: an absolute path is returned as given, whatever the repository root, and is
    accepted although no such file exists (that refusal is Block 9's, at bank time)."""
    p = tmp_path / "p.json"
    p.write_text(json.dumps(dict(base_profile, cohort_map="/no/such/dir/map.csv")),
                 encoding="ascii")
    cfg = _cfg().load_profile(str(p), repo_root=repo_root)
    assert cfg.cohort_map == "/no/such/dir/map.csv"


def test_other_path_keys_still_resolved_against_the_repo_root(tmp_path, base_profile):
    """C6 (regression): Block 6.1's general rule still holds for every other path key:
    a relative value is made absolute against the repository root (here given)."""
    rel = dict(base_profile, plan_out="work/bank_plan.json", bank_root="banks/sim",
               real_bank_root="banks/real", cohort_map=None)
    p = tmp_path / "p.json"
    p.write_text(json.dumps(rel), encoding="ascii")
    cfg = _cfg().load_profile(str(p), repo_root="/r")
    assert cfg.plan_out == "/r/work/bank_plan.json"
    assert cfg.bank_root == "/r/banks/sim" and cfg.real_bank_root == "/r/banks/real"
    assert cfg.cohort_map is None


@pytest.mark.parametrize("value", [0, 1.5, False, [], {}])
def test_cohort_map_other_types_refused(tmp_path, base_profile, value):
    """C5: a value that is neither null nor a string raises ConfigError."""
    with pytest.raises(_cfg().ConfigError):
        _load(tmp_path, dict(base_profile, cohort_map=value))


def _consumer_fixture(tmp_path):
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    prof = H.write_profile(fx, str(tmp_path / "rel.json"), cohort_map="maps/m.csv")
    return fx, prof


def test_relative_cohort_map_refuses_every_consumer(tmp_path):
    """Boundary 6.1 -> 6.5, 8, 7.4: the refusal at load time reaches every entry point
    that loads the profile -- bank_plan.py and bank_real.py print REFUSED (exit 1)
    naming cohort_map, write no plan and no shard; launch_bank.sh plan aborts (exit 2)
    before running anything."""
    fx, prof = _consumer_fixture(tmp_path)
    rc, so, se = H.run_py(os.path.join(H.EXPORT, "bank_plan.py"), "--profile", prof)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert AMEND2_MSG in so and "Traceback" not in so + se
    assert not os.path.exists(os.path.join(fx["root"], "plan", "bank_plan.json"))
    out = str(tmp_path / "real")
    rc, so, se = H.run_py(os.path.join(H.EXPORT, "bank_real.py"), "--profile", prof,
                          "--out-root", out)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:") and AMEND2_MSG in so, so + se
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))
    env = dict(os.environ, PROFILE=prof)
    r = subprocess.run(["bash", os.path.join(H.EXPORT, "launch_bank.sh"), "plan"],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stdout + r.stderr
    assert AMEND2_MSG in r.stdout and "ABORT" in r.stdout, r.stdout
    assert not os.path.exists(os.path.join(fx["root"], "plan", "bank_plan.json"))


@pytest.mark.parametrize("key", ["cohort_map", "drop_partial_topologies"])
def test_missing_new_key_refused(tmp_path, base_profile, key):
    """C4: "ConfigError for an unknown or missing key"."""
    prof = dict(base_profile)
    del prof[key]
    with pytest.raises(_cfg().ConfigError) as ei:
        _load(tmp_path, prof)
    assert key in str(ei.value)
