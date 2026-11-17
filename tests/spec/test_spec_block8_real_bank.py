"""Spec-derived tests of Block 8 (the real bank) as amended by Block 9, without a
cohort map. Tester-written from specs/SPEC.md sections 2.6 and 5 (Blocks 8, 9).
Extended 2026-11-17_1605 (commit 5b61d6c) for contract amendments 2 and 3: traces
shorter than one window counted and reported (provenance, outside the digest), and an
out_dir without <class>/<root>/ refused before any archive is read.
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


# --- contract amendments 2 and 3 (2026-11-17; the final check's F4/S2 and F5) -------
# (2026-11-17_1605: the strict xfails of report 2026-11-17_1500 replaced by these.)

PRE_AMENDMENT = "db4b418c5ea3b8132de7acfd394194d1e5eb53c8"   # Block 9 in, amendment 2 not
SHORT_LINE = "[real] %s: %d trace(s) shorter than one window (K < W, W = %d samples): no row"
WELL_LINE = "[real]   %s (%s): every trace shorter than one window, no row"


def _rewrite_archive(path, n_samples, dt=0.01):
    """Keep the first n_samples of one archive (an extractor's shorter recording)."""
    with np.load(path) as z:
        a = {k: np.array(z[k]) for k in z.files}
    a["ifr_trace"] = a["ifr_trace"][:n_samples]
    a["T_rec"] = np.float64(n_samples * dt)
    np.savez_compressed(path, **a)


def _archive(man, wi, s):
    return os.path.join(man["extract_root"], man["wells"][wi]["out_dir"],
                        "trace_subregion_%02d.npz" % s)


def _short_fixture(tmp_path):
    """Block 8's fixture with traces shorter than one window (W = 18000), T_rec kept
    consistent with every archive (a well without T_rec when its traces differ):
      well 0 (100N0A_wo): T_rec dropped; s0 exactly W samples (one window, not short),
                          s1 W - 1 (short); the other seven 300 s (one window each);
      well 1 (100N0A_wo): unchanged;
      well 2 (50N50A):    T_rec dropped; s0-s2 150 s (short); the other six 600 s;
      well 3 (70N30A):    T_rec 100 s, every archive 100 s: no row at all.
    By hand: 1 + 3 + 9 = 13 short traces; the one well without rows is well 3."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    _rewrite_archive(_archive(man, 0, 0), 18000)
    _rewrite_archive(_archive(man, 0, 1), 17999)
    for s in range(3):
        _rewrite_archive(_archive(man, 2, s), 15000)
    for s in range(9):
        _rewrite_archive(_archive(man, 3, s), 10000)

    def mutate(m):
        m["wells"][0]["T_rec"] = None
        m["wells"][2]["T_rec"] = None
        m["wells"][3]["T_rec"] = 100.0
    man = _rewrite_manifest(fx, mutate)
    return fx, man


def _run_pre_amendment(fx, tmp_path, out, cohort_map=None):
    base = H.extract_commit(PRE_AMENDMENT, str(tmp_path / "pre"))
    if base is None:
        pytest.skip("git history not available for git archive")
    prof = H.write_profile(fx, str(tmp_path / "pre.json"), cohort_map=cohort_map)
    return H.run_py(os.path.join(base, "export_windows", "bank_real.py"), "--profile", prof,
                    "--out-root", out)


def test_short_traces_counted_reported_and_kept_out_of_the_digest(tmp_path):
    """R4-R6 (amendments 2 and 3): no row from a trace with K < W, a trace with K = W
    gives its window; per class the count line (only where n > 0), the well left with
    no row named under its class; provenance.too_short = {13, [well 3's out_dir]};
    per_class wells count the wells with rows, so n_wells = sum of per_class wells +
    the wells named; the rows equal the archives' slices (numpy) and, with latent_spec,
    the contract digest and every other provenance value, the pre-amendment commit's
    on the same inputs, bit for bit."""
    fx, man = _short_fixture(tmp_path)
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, side = load_shard(out)
    rows, W, dt, _ = expected_real(fx)
    assert W == 18000
    check_rows(arrays, rows, W)
    w0 = [r for r in rows if r["well"] == 0]
    assert sorted((r["subregion"], r["window_idx"]) for r in w0) == [(s, 0) for s in (0, 2, 3, 4, 5, 6, 7, 8)]
    assert not any(r["well"] == 3 for r in rows)
    out3 = man["wells"][3]["out_dir"]
    prov = side["provenance"]
    assert prov["too_short"] == {"traces": 13, "wells_without_rows": [out3]}
    classes = man["classes"]
    assert [prov["per_class"][c]["wells"] for c in classes] == [2, 1, 0]
    assert prov["n_wells"] == 4 == sum(d["wells"] for d in prov["per_class"].values()) + 1
    assert "too_short" not in json.dumps(side["latent_spec"])
    lines = H.lines_of(so)
    for c, n in zip(classes, (1, 3, 9)):
        assert SHORT_LINE % (c, n, W) in lines, (c, so)
    want_well = WELL_LINE % (man["wells"][3]["culture"], out3)
    assert want_well in lines and sum("every trace shorter" in ln for ln in lines) == 1
    assert lines.index(SHORT_LINE % (classes[2], 9, W)) < lines.index(want_well)
    # the pre-amendment commit (Block 9 already in) on the very same inputs
    out_pre = str(tmp_path / "real_pre")
    rc, so_pre, se_pre = _run_pre_amendment(fx, tmp_path, out_pre)
    assert rc == 0 and H.last_line(so_pre) == "REAL BANK OK", so_pre + se_pre
    a_pre, s_pre = load_shard(out_pre)
    assert sorted(a_pre) == sorted(arrays)
    for k in a_pre:
        assert a_pre[k].dtype == arrays[k].dtype, k
        np.testing.assert_array_equal(a_pre[k], arrays[k])
    assert side["latent_spec"] == s_pre["latent_spec"]
    assert side["contract_digest"] == s_pre["contract_digest"] == H.latent_bank().contract_digest(side)
    p_new = {k: v for k, v in prov.items() if k not in ("host", "built", "too_short")}
    p_old = {k: v for k, v in s_pre["provenance"].items() if k not in ("host", "built")}
    assert p_new == p_old
    for k in [k for k in side if k not in ("provenance",)]:
        assert side[k] == s_pre[k], k
    # "every other output unchanged": the shard file byte for byte, and the printout but
    # for the new too-short lines (the output folder's name aside)
    assert H.sha256_file(os.path.join(out, "shard_0000.npz")) == \
        H.sha256_file(os.path.join(out_pre, "shard_0000.npz"))

    def strip(ln):
        return ln.replace(out_pre, "<out>").replace(out, "<out>")
    assert [strip(ln) for ln in lines if "shorter than one window" not in ln] == \
        [strip(ln) for ln in H.lines_of(so_pre)]


def test_no_short_trace_gives_zero_and_no_line(tmp_path):
    """R5: on the uncut fixture too_short = {0, []}, no count line, no well line; the
    contract digest equals the pre-amendment commit's."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    _, side = load_shard(out)
    assert side["provenance"]["too_short"] == {"traces": 0, "wells_without_rows": []}
    assert "shorter than one window" not in so
    rc, so_pre, se_pre = _run_pre_amendment(fx, tmp_path, str(tmp_path / "pre_out"))
    assert rc == 0, so_pre + se_pre
    assert load_shard(str(tmp_path / "pre_out"))[1]["contract_digest"] == side["contract_digest"]


def test_two_wells_without_rows_named_in_manifest_order(tmp_path):
    """R4: two wells left with no row (wells 1 and 3, different classes) are both
    named, each under its class, and listed in wells_without_rows in the manifest's
    order of wells (the order the spec leaves open; recorded here as observed)."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    for wi in (1, 3):
        for s in range(9):
            _rewrite_archive(_archive(man, wi, s), 10000)

    def mutate(m):
        m["wells"][1]["T_rec"] = 100.0
        m["wells"][3]["T_rec"] = 100.0
    man = _rewrite_manifest(fx, mutate)
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, side = load_shard(out)
    rows, W, _, _ = expected_real(fx)
    check_rows(arrays, rows, W)
    ods = [man["wells"][wi]["out_dir"] for wi in (1, 3)]
    assert side["provenance"]["too_short"] == {"traces": 18, "wells_without_rows": ods}
    lines = H.lines_of(so)
    for wi in (1, 3):
        assert WELL_LINE % (man["wells"][wi]["culture"], man["wells"][wi]["out_dir"]) in lines


def _map(fx, tmp_path, include):
    w = [x["out_dir"] for x in _manifest_doc(fx)["wells"]]
    path = str(tmp_path / "map.csv")
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write("out_dir,donor,rep_group,include\n")
        for wi, d in enumerate(w):
            fh.write("%s,d%d,g%d,%s\n" % (d, wi, wi, "yes" if wi in include else "no"))
    return path


def test_short_traces_with_a_cohort_map(tmp_path):
    """R4, R6 with Block 9: an excluded well is never read, so its short traces are not
    counted (map without well 3: 1 + 3 = 4 traces, no well named); n_wells counts the
    included wells and equals the per_class wells plus the wells named; a map that
    includes only well 3 leaves no window at all: REFUSED, exit 1, no shard."""
    fx, man = _short_fixture(tmp_path)
    path = _map(fx, tmp_path, include={0, 1, 2})
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=path)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, side = load_shard(out)
    ids = {wi: (wi, wi) for wi in (0, 1, 2)}          # labels d0..d2, g0..g2: sorted = index
    rows, W, _, _ = expected_real(fx, ids)
    check_rows(arrays, rows, W)
    prov = side["provenance"]
    assert prov["too_short"] == {"traces": 4, "wells_without_rows": []}
    assert prov["n_wells"] == 3 == sum(d["wells"] for d in prov["per_class"].values())
    assert "every trace shorter" not in so
    only3 = str(tmp_path / "only3.csv")
    os.replace(_map(fx, tmp_path, include={3}), only3)
    out = str(tmp_path / "real_only3")
    rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=only3)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:") and "no window" in H.last_line(so), so + se
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))


@pytest.mark.parametrize("bad", ["control", "70N30A/AraC_70N30A"])
@pytest.mark.parametrize("with_map", [False, True])
def test_out_dir_without_class_root_prefix_refused_before_any_archive(tmp_path, bad, with_map):
    """R2 (amendment 2, F5): a manifest well (the last, well 3) whose out_dir has fewer
    than three components is refused with amendment 2's text, exit 1, no traceback, no
    shard -- before any archive is read (well 0's archive 04 removed: a reader that
    went first would refuse "missing archive" instead), and also when a cohort map
    leaves that well out."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    os.remove(_archive(man, 0, 4))
    man = _rewrite_manifest(fx, lambda m: m["wells"][3].__setitem__("out_dir", bad))
    path = _map(fx, tmp_path, include={0, 1, 2}) if with_map else None
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=path)
    want = "REFUSED: %s: its folder %s has no <class>/<root>/ prefix" % (man["wells"][3]["culture"], bad)
    assert rc == 1 and H.last_line(so) == want, so + se
    assert "Traceback" not in so + se
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))


def test_out_dir_of_four_components_is_accepted(tmp_path):
    """R3: "<class>/<root>/<well...>" -- a well folder one level deeper (four
    components) builds; its batch is its root's rank among every well's roots."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    src = os.path.join(man["extract_root"], man["wells"][2]["out_dir"])
    dst = os.path.join(src, "deeper")
    os.makedirs(dst)
    for f in os.listdir(src):
        if f.endswith(".npz"):
            os.replace(os.path.join(src, f), os.path.join(dst, f))
    _rewrite_manifest(fx, lambda m: m["wells"][2].__setitem__(
        "out_dir", m["wells"][2]["out_dir"] + "/deeper"))
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, _ = load_shard(out)
    rows, W, _, _ = expected_real(fx)
    check_rows(arrays, rows, W)


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


@pytest.mark.xfail(strict=True, reason="finding F1 (2026-11-17_1605, NOTE): an out_dir whose "
                   "split('/') holds an empty component ('<class>//<well>', '<class>/<root>/') "
                   "passes amendment 2's three-component check and the bank builds; with "
                   "'<class>//<well>' the empty string becomes a class root and a batch id")
@pytest.mark.parametrize("form", ["double_slash", "trailing_slash"])
def test_out_dir_with_an_empty_component_is_refused(tmp_path, form):
    """Amendment 2 (F5) reads `<class>/<root>/<well...>`: a root and a well folder are
    named. Here well 2's archives sit in a two-level folder <class>/<well> and the
    manifest names it '<class>//<well>' (double_slash), or they sit in its root folder
    named '<class>/<root>/' (trailing_slash): either way no <class>/<root>/<well>
    prefix -- expected REFUSED, observed REAL BANK OK."""
    H.sbi_hpc_dir()
    fx = H.write_fixture(str(tmp_path / "fx"))
    man = _manifest_doc(fx)
    cls, root, well = man["wells"][2]["out_dir"].split("/")
    src = os.path.join(man["extract_root"], cls, root, well)
    dst = os.path.join(man["extract_root"], cls, well) if form == "double_slash" \
        else os.path.join(man["extract_root"], cls, root)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        os.replace(os.path.join(src, f), os.path.join(dst, f))
    new = "%s//%s" % (cls, well) if form == "double_slash" else "%s/%s/" % (cls, root)
    _rewrite_manifest(fx, lambda m: m["wells"][2].__setitem__("out_dir", new))
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
