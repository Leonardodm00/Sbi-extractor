"""Spec-derived tests of Block 9 (the cohort map of the real bank) and of its
boundary with Block 8 (bank_real.py) and with the stage-1 shard contract.
Tester-written from specs/SPEC.md section 5, Block 9 (and 2.6, Block 8).

Expected ids are computed by hand from the map text (Python's sorted() on the
labels of the INCLUDED wells, SPEC: "the sorted order of the distinct donor
labels of the included wells"); expected rows from the archives read with numpy.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

import _bank_helpers as H

REAL = os.path.join(H.EXPORT, "bank_real.py")
WELLS = ["A/r1/w0", "A/r1/w1", "B/r2/w2", "B/r3/w3", "C/r3/w4"]
MAN = {"wells": [{"out_dir": d} for d in WELLS]}


def _cmap():
    import cohort_map as CMAP          # noqa: E402
    return CMAP


def _write(tmp_path, text, name="map.csv"):
    p = tmp_path / name
    p.write_bytes(text.encode("latin-1") if isinstance(text, str) else text)
    return str(p)


# --------------------------------------------------------------- unit level

KNOWN = ('note,include,rep_group,out_dir,donor\n'
         '"free, text",yes,g_b,A/r1/w0,  10 \n'
         'x,YES,g_a,A/r1/w1,9\n'
         'y,no,AAA,B/r2/w2,ZZZ\n'
         '"multi\nline",True,g_b,B/r3/w3,B\n'
         'z,1,g_c,C/r3/w4,a\n')


def test_known_answer_ids_by_sorted_labels_of_included_wells(tmp_path):
    """M1, M3, M5: columns in any order, extra column ignored (quoted comma and a
    quoted newline included), labels stripped, ids by Python's sorted() of the
    INCLUDED wells' labels ("10" < "9" < "B" < "a"); the excluded well's labels
    ("ZZZ", "AAA") take no id; sha256 of the bytes; excluded in manifest order."""
    path = _write(tmp_path, KNOWN)
    m = _cmap().read_cohort_map(path, MAN)
    assert m.donor_labels == ("10", "9", "B", "a")
    assert m.rep_group_labels == ("g_a", "g_b", "g_c")
    assert m.donor_of == {0: 0, 1: 1, 3: 2, 4: 3}
    assert m.rep_group_of == {0: 1, 1: 0, 3: 1, 4: 2}
    assert m.excluded == ("B/r2/w2",)
    assert m.sha256 == hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert os.path.isabs(m.path)


def test_crlf_map_gives_the_same_ids(tmp_path):
    """A spreadsheet's CRLF export of the same map is the same map (the csv module)."""
    lf = _cmap().read_cohort_map(_write(tmp_path, KNOWN, "lf.csv"), MAN)
    crlf = _cmap().read_cohort_map(_write(tmp_path, KNOWN.replace("\n", "\r\n"), "crlf.csv"), MAN)
    for f in ("donor_labels", "rep_group_labels", "donor_of", "rep_group_of", "excluded"):
        assert getattr(lf, f) == getattr(crlf, f)
    assert lf.sha256 != crlf.sha256


def _map(rows, header="out_dir,donor,rep_group,include"):
    return header + "\n" + "".join(r + "\n" for r in rows)


def _all(include=("yes",) * 5, donors=("d0", "d1", "d2", "d3", "d4"),
         groups=("g0", "g1", "g2", "g3", "g4")):
    return ["%s,%s,%s,%s" % (w, d, g, i) for w, d, g, i in zip(WELLS, donors, groups, include)]


@pytest.mark.parametrize("token,value", [("yes", True), ("YES", True), ("Yes", True),
                                         ("true", True), ("TRUE", True), ("1", True),
                                         ("no", False), ("No", False), ("false", False),
                                         ("FALSE", False), ("0", False)])
def test_include_tokens_accepted(tmp_path, token, value):
    """M4: yes, no, true, false, 1, 0, case-insensitive."""
    inc = ["yes"] * 5
    inc[2] = token
    m = _cmap().read_cohort_map(_write(tmp_path, _map(_all(include=inc))), MAN)
    assert (2 in m.donor_of) is value
    assert (WELLS[2] in m.excluded) is (not value)


@pytest.mark.parametrize("token", ["y", "n", "2", "", "on", " yes", "yes ", "t"])
def test_include_tokens_refused(tmp_path, token):
    """M4: anything else is refused (only donor and rep_group are stripped)."""
    inc = ["yes"] * 5
    inc[1] = token
    with pytest.raises(_cmap().CohortMapError) as ei:
        _cmap().read_cohort_map(_write(tmp_path, _map(_all(include=inc))), MAN)
    assert "line 3" in str(ei.value)


def _refused(tmp_path, text, manifest=MAN, name="map.csv"):
    path = _write(tmp_path, text, name) if text is not None else str(tmp_path / "absent.csv")
    with pytest.raises(_cmap().CohortMapError) as ei:
        _cmap().read_cohort_map(path, manifest)
    return str(ei.value)


def test_refusals(tmp_path):
    """M7: each refusal of Block 9 (with amendment 1) on a small map."""
    ok = _all()
    # a required column missing
    msg = _refused(tmp_path, _map([r.rsplit(",", 1)[0] for r in ok], "out_dir,donor,rep_group"))
    assert "include" in msg
    # a required column named twice
    msg = _refused(tmp_path, _map([r + ",d9" for r in ok], "out_dir,donor,rep_group,include,donor"))
    assert "twice" in msg and "donor" in msg
    # an out_dir naming no well
    msg = _refused(tmp_path, _map(ok + ["Z/r9/w9,d,g,yes"]))
    assert "Z/r9/w9" in msg and "line 7" in msg
    # a well with no row
    msg = _refused(tmp_path, _map(ok[:-1]))
    assert "C/r3/w4" in msg
    # a well with two rows
    msg = _refused(tmp_path, _map(ok + [ok[1]]))
    assert "A/r1/w1" in msg and "line 7" in msg and "line 3" in msg
    # an empty label (blank after stripping), for an included and an excluded well
    msg = _refused(tmp_path, _map(_all(donors=("d0", "  ", "d2", "d3", "d4"))))
    assert "donor" in msg and "line 3" in msg
    msg = _refused(tmp_path, _map(_all(include=("yes", "no", "yes", "yes", "yes"),
                                       groups=("g0", "", "g2", "g3", "g4"))))
    assert "rep_group" in msg
    # every well excluded
    msg = _refused(tmp_path, _map(_all(include=("no",) * 5)))
    assert "excluded" in msg
    # a non-ASCII file, and a UTF-8 byte-order mark
    _refused(tmp_path, _map(_all(donors=("d0", "d\xe91", "d2", "d3", "d4"))))
    _refused(tmp_path, b"\xef\xbb\xbf" + _map(ok).encode("ascii"), name="bom.csv")
    # a row whose number of fields differs from the header's, naming the line and both counts
    five = [r + ",n" for r in ok]
    msg = _refused(tmp_path, _map(five[:2] + [ok[2] + ",free, text"] + five[3:],
                                  "out_dir,donor,rep_group,include,note"))
    assert "line 4" in msg and "6" in msg and "5" in msg
    _cmap().read_cohort_map(_write(tmp_path, _map(five, "out_dir,donor,rep_group,include,note"),
                                   "five.csv"), MAN)          # the same map, well formed: accepted
    msg = _refused(tmp_path, _map(ok[:3] + [ok[3].rsplit(",", 1)[0]] + ok[4:]))
    assert "line 5" in msg and "3" in msg and "4" in msg
    # a line the csv module cannot parse (strict)
    msg = _refused(tmp_path, _map(ok[:4] + ['"C/r3/w4"x,d4,g4,yes']))
    assert "line 6" in msg
    # a missing file
    _refused(tmp_path, None)
    # a manifest without wells, a well without out_dir, an out_dir listed twice
    _refused(tmp_path, _map(ok), manifest={"wells": []})
    _refused(tmp_path, _map(ok), manifest={"wells": [{"out_dir": d} for d in WELLS[:-1]] + [{}]})
    _refused(tmp_path, _map(ok), manifest={"wells": [{"out_dir": d} for d in WELLS + [WELLS[0]]]})


def test_blank_lines_and_whitespace_only_lines_never_change_the_ids(tmp_path):
    """A line that holds no well cannot change the result: an empty line is skipped
    (the csv module yields no field), and -- since the 2026-11-17 amendment of Block 9
    (S6) -- a line of blanks only is a row of one field, refused by the field-count
    rule, naming the line and both counts (1 field, the header's 4).
    (Tightened 2026-11-17_1605: the earlier version accepted either outcome.)"""
    base = _cmap().read_cohort_map(_write(tmp_path, _map(_all()), "a.csv"), MAN)
    rows = _all()
    blank = _cmap().read_cohort_map(_write(tmp_path, _map(rows[:2] + [""] + rows[2:]), "b.csv"), MAN)
    assert blank.donor_of == base.donor_of and blank.rep_group_of == base.rep_group_of
    for ws in ("   ", "\t", " \t "):
        msg = _refused(tmp_path, _map(rows[:2] + [ws] + rows[2:]), name="c.csv")
        assert "line 4" in msg and "1 fields" in msg and "has 4" in msg, (repr(ws), msg)


def test_excluded_listed_in_manifest_order_whatever_the_map_order(tmp_path):
    """S6 (Block 9, 2026-11-17): latent_spec.cohort_map.excluded lists the excluded
    out_dirs in the manifest's order of wells -- here three excluded wells (0, 2, 4)
    written in the map in the reverse order (4, 2, 0, then the included 3 and 1)."""
    inc = ("no", "yes", "no", "yes", "no")
    rows = _all(include=inc)
    m = _cmap().read_cohort_map(_write(tmp_path, _map(rows[::-1])), MAN)
    assert m.excluded == (WELLS[0], WELLS[2], WELLS[4])
    assert m.donor_of == {1: 0, 3: 1} and m.rep_group_of == {1: 0, 3: 1}


def test_out_dir_matching_is_exact_after_the_separator_rule(tmp_path):
    """M2: out_dir is matched on the manifest's out_dir with os.sep -> "/"; a
    surrounding blank or another separator does not match (refused, not guessed)."""
    rows = _all()
    _refused(tmp_path, _map([" " + rows[0]] + rows[1:]), name="blank.csv")
    if os.sep == "/":
        _refused(tmp_path, _map([rows[0].replace("/", "\\", 2)] + rows[1:]), name="bs.csv")


# --------------------------------------------------------------- bank level

def _manifest_doc(fx):
    with open(fx["manifest"], "r", encoding="ascii") as fh:
        return json.load(fh)


def expected_real(fx, ids=None, t_win_s=180.0, floor=0.1, rtol=1e-6):
    """Section 2.6 recomputed from the manifest and the archives (numpy only).
    ids: None (no map), else {well index: (donor id, rep_group id)} of the included."""
    man = _manifest_doc(fx)
    dt = float(man["preprocessing"]["w_size"])
    W = int(round(t_win_s / dt))
    classes, root = man["classes"], man["extract_root"]
    roots = sorted({w["out_dir"].replace(os.sep, "/").split("/")[1] for w in man["wells"]})
    rows = []
    for wi, w in enumerate(man["wells"]):
        if ids is not None and wi not in ids:
            continue
        parts = w["out_dir"].replace(os.sep, "/").split("/")
        for s in range(int(man["n_subsets_per_well"])):
            with np.load(os.path.join(root, w["out_dir"], "trace_subregion_%02d.npz" % s)) as z:
                y = np.array(z["ifr_trace"])
            n_win = (y.shape[0] - W) // W + 1 if y.shape[0] >= W else 0
            for j in range(n_win):
                x = y[j * W:(j + 1) * W]
                donor, group = (wi, wi) if ids is None else ids[wi]
                rows.append({"x": x, "cls": classes.index(parts[0]), "well": wi,
                             "realisation_id": wi, "donor": donor, "rep_group": group,
                             "batch": roots.index(parts[1]), "subregion": s, "window_idx": j,
                             "below": np.sum(x, dtype=np.float64) / (W * dt) < floor * (1 - rtol)})
    return rows, W, dt, man


def run_real(fx, root, out_root, cohort_map=None, name="real_profile.json", env=None, **kw):
    prof = H.write_profile(fx, os.path.join(root, name), cohort_map=cohort_map, **kw)
    rc, so, se = H.run_py(REAL, "--profile", prof, "--out-root", out_root, env=env)
    return rc, so, se


def load_shard(out_root):
    with np.load(os.path.join(out_root, "shard_0000.npz")) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    with open(os.path.join(out_root, "shard_0000.json"), "r", encoding="ascii") as fh:
        side = json.load(fh)
    return arrays, side


def check_rows(arrays, rows, W):
    n = len(rows)
    assert arrays["x"].shape == (n, W) and arrays["x"].dtype == np.float32
    assert arrays["theta"].shape == (n, 0) and arrays["theta"].dtype == np.float32
    assert arrays["nu"].shape == (n, 5) and not np.any(arrays["nu"])
    assert arrays["realisation_id"].dtype == np.uint64
    for key in ("cls", "donor", "well", "batch", "subregion", "window_idx", "rep_group"):
        assert arrays[key].dtype == np.int64 and arrays[key].shape == (n,), key
    assert arrays["contaminated"].dtype == bool and not np.any(arrays["contaminated"])
    for r, row in enumerate(rows):
        np.testing.assert_array_equal(arrays["x"][r], row["x"])      # bit for bit
        for key in ("cls", "well", "realisation_id", "donor", "rep_group", "batch",
                    "subregion", "window_idx"):
            assert int(arrays[key][r]) == row[key], (key, r)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    H.sbi_hpc_dir()
    root = str(tmp_path_factory.mktemp("b9_fixture"))
    return H.write_fixture(root)


def _map_a(fx, tmp_path):
    """Well 1 (the 0.05 Hz class-0 well) excluded with labels that would sort first;
    the other classes share donor "d_a" (one padded with blanks); three groups."""
    w = [x["out_dir"] for x in _manifest_doc(fx)["wells"]]
    text = ("rep_group,note,out_dir,include,donor\n"
            'g2,"plate 1, well A",%s,yes,d_b\n'
            'aaa_excluded,"copy, left out",%s,no,AAA_excluded\n'
            "g1,,%s,Yes,d_a\n"
            "g3,x,%s,TRUE, d_a \n" % (w[0], w[1], w[2], w[3]))
    path = str(tmp_path / "map_a.csv")
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write(text)
    # hand: donors sorted ["d_a", "d_b"]; groups sorted ["g1", "g2", "g3"]
    ids = {0: (1, 1), 2: (0, 0), 3: (0, 2)}
    return path, ids, w


def test_bank_with_map_rows_ids_and_sidecar(fx, tmp_path):
    """M5, M6, M8, M9, M10, R2, R4, R5 on Block 8's fixture with map A."""
    path, ids, w = _map_a(fx, tmp_path)
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=path)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, side = load_shard(out)
    rows, W, dt, man = expected_real(fx, ids)
    assert len(rows) == 9 + 27 + 27
    check_rows(arrays, rows, W)
    assert 1 not in set(arrays["well"].tolist())
    # sidecar (2.6, Block 9)
    lb = H.latent_bank()
    assert side["arm"] == "R" and side["coord"] == "unit_box" and side["param_names"] == []
    assert side["bounds_theta"] == [] and side["realisation_spec"]["kind"] == "recording"
    assert side["generator_sha256"] == H.sha256_file(fx["manifest"])
    ls = side["latent_spec"]
    assert ls["kind"] == "campaign_real"
    assert ls["cohort_map"] == {"sha256": H.sha256_file(path), "donor_labels": ["d_a", "d_b"],
                                "rep_group_labels": ["g1", "g2", "g3"], "excluded": [w[1]],
                                "n_wells_included": 3}
    assert side["contract_digest"] == lb.contract_digest(side)
    prov = side["provenance"]
    assert prov["real_version"] == 2 and prov["n_wells"] == 3
    classes = man["classes"]
    want = {c: {"windows": sum(1 for r in rows if r["cls"] == i),
                "below_floor": sum(1 for r in rows if r["cls"] == i and r["below"]),
                "wells": len({r["well"] for r in rows if r["cls"] == i})}
            for i, c in enumerate(classes)}
    assert prov["per_class"] == want
    assert want[classes[0]]["wells"] == 1
    # printed lines (Block 9): the map's lines, then the per-class lines, then OK
    lines = H.lines_of(so)
    head = ("[real] cohort map %s sha256 %s: 1 of 4 wells excluded; 2 donors; 3 pairing groups"
            % (os.path.abspath(path), H.sha256_file(path)[:16]))
    assert head in lines
    i = lines.index(head)
    assert lines[i + 1] == "[real]   excluded %s" % w[1]
    first_class = min(n for n, ln in enumerate(lines) if ln.startswith("[real] %s" % classes[0]))
    assert i < first_class
    # stage 1 (boundary B3): check_arrays with rep_group, read back by concat_shards
    assert lb.check_arrays(arrays) == len(rows)
    merged, _ = lb.concat_shards([os.path.join(out, "shard_0000.npz")])
    np.testing.assert_array_equal(merged["rep_group"], arrays["rep_group"])


def test_bank_with_map_is_deterministic_and_the_map_enters_the_digest(fx, tmp_path):
    """M8, M11: two runs give identical arrays and digest; no map gives another digest."""
    path, ids, _ = _map_a(fx, tmp_path)
    digests, shards = [], []
    for name in ("r1", "r2"):
        out = str(tmp_path / name)
        rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=path)
        assert rc == 0, so + se
        a, side = load_shard(out)
        digests.append(side["contract_digest"])
        shards.append(a)
    assert digests[0] == digests[1]
    for k in shards[0]:
        np.testing.assert_array_equal(shards[0][k], shards[1][k])
    out = str(tmp_path / "nomap")
    rc, so, se = run_real(fx, str(tmp_path), out)
    assert rc == 0, so + se
    assert load_shard(out)[1]["contract_digest"] != digests[0]


def test_excluded_well_archives_are_not_read(tmp_path):
    """M6: an excluded well with no archive at all builds; batch ids still number
    every manifest well's root; an INCLUDED well without its archive refuses."""
    H.sbi_hpc_dir()
    fxl = H.write_fixture(str(tmp_path / "fx"))
    path, ids, w = _map_a(fxl, tmp_path)
    man = _manifest_doc(fxl)
    gone = os.path.join(man["extract_root"], w[1])
    for f in os.listdir(gone):
        os.remove(os.path.join(gone, f))
    out = str(tmp_path / "real")
    rc, so, se = run_real(fxl, str(tmp_path), out, cohort_map=path)
    assert rc == 0 and H.last_line(so) == "REAL BANK OK", so + se
    arrays, _ = load_shard(out)
    rows, W, _, _ = expected_real(fxl, ids)
    check_rows(arrays, rows, W)                  # batch of well 0 is still 2 (its root's rank of 4)
    assert sorted(set(arrays["batch"].tolist())) == [0, 1, 2]
    # without the map the missing archive refuses
    rc, so, se = run_real(fxl, str(tmp_path), str(tmp_path / "real2"))
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert not os.path.exists(os.path.join(str(tmp_path / "real2"), "shard_0000.npz"))


@pytest.mark.parametrize("case", ["no_row", "unknown_well", "bad_include", "all_excluded",
                                  "ragged", "absent"])
def test_bank_refuses_a_bad_map(fx, tmp_path, case):
    """M7 through bank_real.py: REFUSED, exit 1, no shard written."""
    w = [x["out_dir"] for x in _manifest_doc(fx)["wells"]]
    rows = ["%s,d%d,g%d,yes" % (d, n, n) for n, d in enumerate(w)]
    if case == "no_row":
        rows = rows[:-1]
    elif case == "unknown_well":
        rows.append("X/Y/Z,d9,g9,yes")
    elif case == "bad_include":
        rows[0] = rows[0][:-3] + "maybe"
    elif case == "all_excluded":
        rows = [r[:-3] + "no" for r in rows]
    elif case == "ragged":
        rows[2] = rows[2] + ",extra"
    path = str(tmp_path / "bad.csv")
    if case != "absent":
        with open(path, "w", encoding="ascii") as fh:
            fh.write("out_dir,donor,rep_group,include\n" + "\n".join(rows) + "\n")
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out, cohort_map=path)
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert "Traceback" not in so + se
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))


def test_older_stage1_contract_is_refused_naming_the_commit(fx, tmp_path):
    """Boundary B3 / SPEC sec. 6: with a latent_bank older than SBI 88c32e5 (no
    optional rep_group) bank_real.py refuses, naming the commit."""
    sbi = os.path.dirname(H.sbi_hpc_dir())
    old = H.extract_commit("88c32e5^", str(tmp_path / "sbi_old"), repo=sbi,
                           paths=["hpc/dsn", "hpc/joint"])
    if old is None:
        pytest.skip("Simulation-Based-Inference history not available")
    out = str(tmp_path / "real")
    rc, so, se = run_real(fx, str(tmp_path), out, env={"SBI_HPC_DIR": os.path.join(old, "hpc")})
    assert rc == 1 and H.last_line(so).startswith("REFUSED:"), so + se
    assert "88c32e5" in H.last_line(so)
    assert not os.path.exists(os.path.join(out, "shard_0000.npz"))
