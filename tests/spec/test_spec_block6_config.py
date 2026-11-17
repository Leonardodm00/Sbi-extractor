"""Spec-derived tests of Block 6.1 for the two profile keys added by Blocks 9 and
10 (cohort_map, drop_partial_topologies). Tester-written from specs/SPEC.md,
Block 6.1 ("cohort_map -- an absolute path or null ... drop_partial_topologies --
a bool"; "ConfigError for an unknown or missing key").
"""
from __future__ import annotations

import json
import os

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


@pytest.mark.xfail(strict=True, reason="finding F1: a relative (or empty) cohort_map is resolved "
                   "against the repository root instead of being refused; the ConfigError in "
                   "bank_plan_config._validate cannot fire")
@pytest.mark.parametrize("value", ["maps/linddup_cohort_map.csv", ""])
def test_cohort_map_relative_path_refused(tmp_path, base_profile, value):
    """C3, literal reading of Block 6.1: "cohort_map -- an absolute path or null"."""
    with pytest.raises(_cfg().ConfigError):
        _load(tmp_path, dict(base_profile, cohort_map=value))


@pytest.mark.parametrize("key", ["cohort_map", "drop_partial_topologies"])
def test_missing_new_key_refused(tmp_path, base_profile, key):
    """C4: "ConfigError for an unknown or missing key"."""
    prof = dict(base_profile)
    del prof[key]
    with pytest.raises(_cfg().ConfigError) as ei:
        _load(tmp_path, prof)
    assert key in str(ei.value)
