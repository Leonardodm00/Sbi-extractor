"""Spec-derived tests of Block 8 (the real bank) as amended by Block 9, without a
cohort map. Tester-written from specs/SPEC.md sections 2.6 and 5 (Blocks 8, 9).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import _bank_helpers as H
from test_spec_block9_cohort_map import (expected_real, run_real, load_shard, check_rows,
                                         _manifest_doc)

REAL = os.path.join(H.EXPORT, "bank_real.py")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    H.sbi_hpc_dir()
    return H.write_fixture(str(tmp_path_factory.mktemp("b8_fixture")))


@pytest.mark.parametrize("K,W,want", [(30000, 18000, [0]), (36000, 18000, [0, 18000]),
                                      (60000, 18000, [0, 18000, 36000]), (17999, 18000, []),
                                      (18000, 18000, [0]), (35999, 18000, [0]), (0, 18000, []),
                                      (54000, 18000, [0, 18000, 36000])])
def test_windows_of(K, W, want):
    """R1: j W for j in 0..floor((K - W) / W); none when K < W."""
    import bank_real as BR
    assert BR.windows_of(K, W) == want


def test_no_map_rows_ids_sidecar_and_printout(fx, tmp_path):
    """R2-R5, R7, R8, M8, M9: without a map donor = well = rep_group = n_w; 18, 27, 27
    windows; 9 below the floor (the 0.05 Hz well); latent_spec cohort_map null."""
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, side = load_shard(out)
    rows, W, dt, man = expected_real(fx)
    check_rows(arrays, rows, W)
    classes = man["classes"]
    counts = [sum(1 for r in rows if r["cls"] == i) for i in range(len(classes))]
    assert counts == [18, 27, 27]
    assert sum(r["below"] for r in rows) == 9
    np.testing.assert_array_equal(arrays["rep_group"], arrays["donor"])
    np.testing.assert_array_equal(arrays["donor"], arrays["well"])
    assert side["latent_spec"]["cohort_map"] is None
    assert side["provenance"]["real_version"] == 2 and side["provenance"]["n_wells"] == 4
    assert side["provenance"]["below_floor"] == 9
    assert side["generator_sha256"] == H.sha256_file(fx["manifest"])
    lines = H.lines_of(so)
    assert "[real] cohort map: none (donor = rep_group = well)" in lines
    i = lines.index("[real] cohort map: none (donor = rep_group = well)")
    first_class = min(n for n, ln in enumerate(lines) if ln.startswith("[real] %s" % classes[0]))
    assert i < first_class


def test_no_map_rows_equal_real_version_1(fx, tmp_path):
    """R3 (regression): without a map the rows are REAL_VERSION 1's, array for
    array, bit for bit (the pre-wave commit run on the same fixture)."""
    base = H.extract_commit(H.BASE, str(tmp_path / "base"))
    if base is None:
        pytest.skip("git history not available for git archive")
    prof_old = H.write_profile(fx, str(tmp_path / "old.json"), drop_new_keys=True)
    out_old = str(tmp_path / "real_v1")
    rc, so, se = H.run_py(os.path.join(base, "export_windows", "bank_real.py"),
                          "--profile", prof_old, "--out-root", out_old)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    out_new = str(tmp_path / "real_v2")
    rc, so, se = run_real(fx, str(tmp_path), out_new)
    assert rc == 0, so + se
    a1, s1 = load_shard(out_old)
    a2, s2 = load_shard(out_new)
    assert set(a2) == set(a1) | {"rep_group"}
    for k in a1:
        assert a1[k].dtype == a2[k].dtype, k
        np.testing.assert_array_equal(a1[k], a2[k])
    assert s1["provenance"]["per_class"] == s2["provenance"]["per_class"]
    ls1 = dict(s1["latent_spec"])
    ls2 = dict(s2["latent_spec"])
    assert ls2.pop("cohort_map") is None
    assert ls1 == ls2


def test_refusals_write_nothing(fx, tmp_path):
    """R6: another manifest digest prefix; n_e != electrodes_per_subset; an existing
    shard_0000.npz (never overwritten: bytes unchanged). Each REFUSED, exit 1."""
    man = _manifest_doc(fx)
    cases = {"digest": {"cohort_manifest_sha256_prefix": "0123456789abcdef"},
             "n_e": {"n_e": 2}}
    for name, kw in cases.items():
        out = str(tmp_path / name)
        rc, so, se = run_real(fx, str(tmp_path), out, name="p_%s.json" % name, **kw)
        assert rc == 1 and H.last_line(so).startswith("REFUSED:"), (name, so + se)
        assert not os.path.exists(os.path.join(out, "shard_0000.npz")), name
    out = str(tmp_path / "exists")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0, so + se
    before = H.sha256_file(os.path.join(out, "shard_0000.npz"))
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert H.sha256_file(os.path.join(out, "shard_0000.npz")) == before
    assert man["n_wells"] == 4


def _rewrite_manifest(fx, mutate):
    """Change the frozen manifest (and its .sha256, via cohort_manifest.write_manifest,
    a library out of coverage); returns the new digest for the profile's prefix."""
    import cohort_manifest as CM
    man = _manifest_doc(fx)
    mutate(man)
    for k in [k for k in man if k.startswith("_")]:
        man.pop(k)
    fx["digests"]["manifest"] = CM.write_manifest(man, fx["manifest"])
    return man


@pytest.mark.xfail(strict=True, reason="finding F4: a well whose every trace is shorter than "
                   "one window is skipped without a count or a line (SPEC sec. 4)")
def test_trace_shorter_than_a_window_is_counted_and_reported(tmp_path):
    """SPEC sec. 4: "a trace too short for one window is counted and reported, never
    silently skipped". One 600 s well cut to 100 s (T_rec 100, K 10000 < W): the
    bank is built (none of its windows exist) and the skip must show somewhere."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))

    def cut(man):
        w = man["wells"][3]
        d = os.path.join(man["extract_root"], w["out_dir"])
        for s in range(int(man["n_subsets_per_well"])):
            p = os.path.join(d, "trace_subregion_%02d.npz" % s)
            with np.load(p) as z:
                a = {k: np.array(z[k]) for k in z.files}
            a["ifr_trace"], a["T_rec"] = a["ifr_trace"][:10000], np.float64(100.0)
            np.savez_compressed(p, **a)
        w["T_rec"] = 100.0
    man = _rewrite_manifest(fx, cut)
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    _, side = load_shard(out)
    prov = side["provenance"]
    accounted = sum(d["wells"] for d in prov["per_class"].values()) == prov["n_wells"]
    named = man["wells"][3]["out_dir"] in so          # (tmp paths hold the test's name)
    assert accounted or named


@pytest.mark.xfail(strict=True, reason="finding F5: an out_dir without a root component "
                   "raises IndexError (traceback) instead of REFUSED")
def test_out_dir_without_root_is_refused_not_a_traceback(tmp_path):
    """Block 8 errors: a well whose folder names no class (root) is REFUSED, exit 1,
    last line `REFUSED: ...` -- here an out_dir of one component."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    _rewrite_manifest(fx, lambda man: man["wells"][1].__setitem__("out_dir", "100N0A_wo"))
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se


def test_trace_length_disagreeing_with_T_rec_refuses(tmp_path):
    """R6: K differing from T_rec / Delta_t refuses (one archive cut short)."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    w = man["wells"][2]
    p = os.path.join(man["extract_root"], w["out_dir"], "trace_subregion_04.npz")
    with np.load(p) as z:
        d = {k: np.array(z[k]) for k in z.files}
    d["ifr_trace"] = d["ifr_trace"][:-100]
    np.savez_compressed(p, **d)
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))
