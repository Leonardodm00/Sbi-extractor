"""Smoke test for the real bank's cohort map -- Block 9 in specs/SPEC.md (SBI
decision D-148 (4); D-144..D-146 for LindDUP).

Covers cohort_map.read_cohort_map (the CSV rules, every refusal, the ids by
sorted labels, the sha256) and bank_real.py applying it (REAL_VERSION 2): the
rows of the included wells only, donor and rep_group from the map, the
sidecar's latent_spec "cohort_map" block, the printed lines; without a map
the rows of REAL_VERSION 1 and rep_group == donor == well. The bank checks run
on Block 8's fixture of bank_plan_fixture.py (two 300 s wells of class 0, one
600 s well of each other class, 9 subregions each), written to a temporary
folder; the reader's checks also run on small synthetic manifests. Nothing
here reads the cluster's files.

Two checks compare with something outside this repository's working tree,
and are skipped (SKIP, with the reason) when it is absent:
  test_94_no_map_equals_version_1   bank_real.py at REAL_VERSION 1, read with
                                    `git show ae71f33:export_windows/bank_real.py`
                                    (the contract commit of Block 9);
  test_95_linddup_map_accepted      the committed LindDUP map of the
                                    Simulation-Based-Inference checkout,
                                    $SBI_HPC_DIR/pipeline/projects/linddup/
                                    linddup_cohort_map.csv (read-only).

Run (from the Sbi-extractor root, in the sbi_export environment):
    export SBI_HPC_DIR=<Simulation-Based-Inference>/hpc   # latent_bank with rep_group (88c32e5 or later)
    source env.sh
    python3 export_windows/smoke_test_cohort_map.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table; exit
code 1 if any check failed, errored or is TODO. Each check removes its own
temporary folders when it ends.

Keep this file pure ASCII and LF only (hpc-python-compat).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bank_export as EXP             # noqa: E402
import bank_plan_fixture as FX        # noqa: E402
import bank_real as REAL              # noqa: E402
import cohort_manifest as CM          # noqa: E402
import cohort_map as CMAP             # noqa: E402
import smoke_test_bank_plan as SP     # noqa: E402  (its helpers; its checks run in its own suite)

SEED = 20261117                         # the synthetic maps' labels and row orders
REPORT_PACKAGES = ("numpy", "scipy")
W = 18000                               # 180 s at 0.01 s (D-041)
V1_COMMIT = "ae71f33"                   # Block 9's contract commit: bank_real.py at REAL_VERSION 1
HEADER = ("out_dir", "donor", "rep_group", "include")
TOKENS_IN = ("yes", "YES", "Yes", "true", "TRUE", "1")
TOKENS_OUT = ("no", "NO", "No", "false", "FALSE", "0")


# --------------------------------------------------------------- helpers ---

def od(wi):
    """The fixture's well wi as the map names it: its manifest out_dir, "/" separators."""
    return "/".join(FX.REAL_WELLS[wi][:3])


# The map of the bank checks: well 0 (class 0) left out; the two other classes
# one donor label ("arac") and two pairing groups. Columns in another order
# than REQUIRED_COLUMNS, a free-text column, a quoted comma, labels with
# blanks, include in mixed case, rows in another order than the manifest's.
# The ids by sorted labels differ from both the manifest order and the rows'
# order of first appearance; the left-out well's labels sort first, so they
# would shift every id if they were counted.
MAIN_MAP = ("note,include,rep_group,out_dir,donor\n"
            "AraC 70/30,True,arac_30,%s,arac\n"
            "\"left out, as the copies of D-145\",No,aaa_left_out,%s,aaa_left_out\n"
            "AraC 50/50,1,arac_50,%s, arac\n"
            ",YES, ctrl_b2 ,%s,ctrl \n" % (od(3), od(0), od(2), od(1)))
MAIN_DONOR = {1: 1, 2: 0, 3: 0}                       # "arac" 0, "ctrl" 1
MAIN_GROUP = {1: 2, 2: 1, 3: 0}                       # "arac_30" 0, "arac_50" 1, "ctrl_b2" 2


def write_text(path, text):
    """Bytes as given (str: ASCII), so that the digest is known."""
    with open(path, "wb") as fh:
        fh.write(text.encode("ascii") if isinstance(text, str) else text)
    return path


def map_text(rows, header=HEADER, eol="\n"):
    """A map from rows (tuples in the header's order), csv-quoted where needed."""
    buf = []
    for r in [header] + list(rows):
        out = []
        for v in r:
            v = str(v)
            out.append('"%s"' % v.replace('"', '""') if ("," in v or '"' in v) else v)
        buf.append(",".join(out))
    return eol.join(buf) + eol


def synthetic_manifest(n):
    return {"wells": [{"out_dir": "cls%d/root_%d/well_%02d" % (i % 3, i % 4, i)} for i in range(n)]}


def expect_map_error(fragment, path, manifest):
    try:
        CMAP.read_cohort_map(path, manifest)
    except CMAP.CohortMapError as exc:
        assert fragment in str(exc), "message %r lacks %r" % (str(exc), fragment)
        assert path in str(exc), "message %r does not name the file" % str(exc)
        return str(exc)
    raise AssertionError("read_cohort_map did not refuse (%s)" % fragment)


class MapFixture(object):
    """Block 8's fixture with a cohort map beside it, named by its profile."""

    def __init__(self, text=MAIN_MAP):
        self.f = SP.Fixture()
        self.map_path = write_text(os.path.join(self.f.dir, "cohort_map.csv"), text)
        FX.write_profile(self.f.fx, self.f.profile, cohort_map=self.map_path)
        self.dir, self.profile = self.f.dir, self.f.profile

    def cfg(self):
        return self.f.cfg()

    def close(self):
        self.f.close()


def expected_rows(root, donor_of, group_of):
    """Section 2.6's rows re-derived from the fixture's archives, for the wells
    of donor_of in manifest order: subregions 0-8, windows of W samples back to
    back from t = 0 (D-056: floor(K / W) of them)."""
    roots = sorted({w[1] for w in FX.REAL_WELLS})        # over every well of the manifest
    cols = dict((k, []) for k in ("x", "cls", "well", "donor", "rep_group", "batch",
                                  "subregion", "window_idx"))
    for wi in sorted(donor_of):
        cls_name, root_name, well = FX.REAL_WELLS[wi][:3]
        for s in range(9):
            with np.load(os.path.join(root, "real", cls_name, root_name, well,
                                      "trace_subregion_%02d.npz" % s)) as z:
                trace = np.asarray(z["ifr_trace"])
            for j in range(trace.size // W):
                for k, v in (("x", trace[j * W:(j + 1) * W]),
                             ("cls", FX.REAL_CLASSES.index(cls_name)), ("well", wi),
                             ("donor", donor_of[wi]), ("rep_group", group_of[wi]),
                             ("batch", roots.index(root_name)), ("subregion", s),
                             ("window_idx", j)):
                    cols[k].append(v)
    n = len(cols["x"])
    return {"x": np.stack(cols["x"]).astype(np.float32), "theta": np.zeros((n, 0), np.float32),
            "cls": np.asarray(cols["cls"], np.int64), "nu": np.zeros((n, 5), np.float32),
            "realisation_id": np.asarray(cols["well"], np.uint64),
            "donor": np.asarray(cols["donor"], np.int64), "well": np.asarray(cols["well"], np.int64),
            "batch": np.asarray(cols["batch"], np.int64),
            "subregion": np.asarray(cols["subregion"], np.int64),
            "window_idx": np.asarray(cols["window_idx"], np.int64),
            "contaminated": np.zeros(n, bool), "rep_group": np.asarray(cols["rep_group"], np.int64)}


def assert_arrays_equal(got, want, skip=()):
    """Same names, dtypes, shapes and values (exact: the rows are copies, no arithmetic)."""
    names = sorted(k for k in want if k not in skip)
    assert sorted(k for k in got if k not in skip) == names, (sorted(got), names)
    for k in names:
        assert got[k].dtype == want[k].dtype, (k, got[k].dtype, want[k].dtype)
        assert got[k].shape == want[k].shape, (k, got[k].shape, want[k].shape)
        assert np.array_equal(got[k], want[k]), k


def version_1_module():
    """bank_real.py at REAL_VERSION 1 (git show V1_COMMIT), imported as
    bank_real_v1 against this tree's other modules; (module, its folder)."""
    repo = os.path.dirname(HERE)
    try:
        r = subprocess.run(["git", "-C", repo, "show", "%s:export_windows/bank_real.py" % V1_COMMIT],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise unittest.SkipTest("git is not usable here: %s" % exc) from exc
    if r.returncode != 0:
        raise unittest.SkipTest("commit %s is not in this checkout's history: %s"
                                % (V1_COMMIT, r.stderr.decode("ascii", "replace").strip()))
    src = r.stdout.decode("ascii")
    assert "\nREAL_VERSION = 1\n" in src, "bank_real.py at %s is not REAL_VERSION 1" % V1_COMMIT
    d = tempfile.mkdtemp(prefix="bank_real_v1_")
    path = write_text(os.path.join(d, "bank_real_v1.py"), src)
    spec = importlib.util.spec_from_file_location("bank_real_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    while d in sys.path:                # the module put its own folder on sys.path
        sys.path.remove(d)
    return mod, d


# ------------------------------------------------- the reader (cohort_map) ---

def test_91_read_known_answer():
    f = MapFixture()
    try:
        man = CM.read_manifest(f.cfg().cohort_manifest_path)
        m = CMAP.read_cohort_map(f.map_path, man)
        with open(f.map_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert m.path == os.path.abspath(f.map_path), m.path
        assert m.sha256 == digest and len(m.sha256) == 64 and m.sha256 == m.sha256.lower()
        assert m.donor_labels == ("arac", "ctrl"), m.donor_labels      # stripped, sorted
        assert m.rep_group_labels == ("arac_30", "arac_50", "ctrl_b2"), m.rep_group_labels
        assert m.donor_of == MAIN_DONOR, m.donor_of
        assert m.rep_group_of == MAIN_GROUP, m.rep_group_of
        assert m.excluded == (od(0),), m.excluded
        try:
            m.sha256 = "0" * 64
        except Exception:                # dataclasses.FrozenInstanceError
            pass
        else:
            raise AssertionError("CohortMap is not frozen")
    finally:
        f.close()


def test_91_ids_reference():
    """The ids against numpy.unique (sorted distinct labels, the inverse index)
    over the included wells, on random maps with rows in random order."""
    rng = np.random.default_rng(SEED)
    d = tempfile.mkdtemp(prefix="cohort_map_ids_")
    try:
        for trial in range(6):
            n = int(rng.integers(5, 40))
            man = synthetic_manifest(n)
            donors = ["d%d" % v for v in rng.integers(0, 6, n)]
            groups = ["g%02d" % v for v in rng.integers(0, 9, n)]
            incl = rng.random(n) < 0.7
            incl[int(rng.integers(0, n))] = True               # at least one well included
            rows = [(man["wells"][i]["out_dir"], donors[i], groups[i],
                     TOKENS_IN[i % 6] if incl[i] else TOKENS_OUT[i % 6]) for i in range(n)]
            order = rng.permutation(n)
            path = write_text(os.path.join(d, "map_%d.csv" % trial), map_text([rows[i] for i in order]))
            m = CMAP.read_cohort_map(path, man)
            inc = np.flatnonzero(incl)
            for labels, got_labels, got_of in ((donors, m.donor_labels, m.donor_of),
                                               (groups, m.rep_group_labels, m.rep_group_of)):
                uniq, inverse = np.unique(np.asarray(labels)[inc], return_inverse=True)
                assert got_labels == tuple(uniq.tolist()), (trial, got_labels, uniq)
                assert got_of == dict(zip(inc.tolist(), inverse.ravel().tolist())), trial
            assert m.excluded == tuple(man["wells"][i]["out_dir"] for i in range(n) if not incl[i])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_91_tokens_and_columns():
    """Every include token in several cases; labels stripped; columns in any
    order with other columns ignored; a free-text field holding a comma (and
    a doubled quote), quoted as a spreadsheet writes it, read as one field
    (amendment 1); a CRLF file reads the same."""
    n = 12
    man = synthetic_manifest(n)
    d = tempfile.mkdtemp(prefix="cohort_map_tokens_")
    try:
        lines = ["out_dir,extra,include,rep_group,donor,note"]
        for i in range(n):
            token = (TOKENS_IN + TOKENS_OUT)[i]
            note = '"copy of well_%02d (cmp of every file, 2026-11-17; ""D-145"")"' % i   # quoted
            lines.append("%s,x%d,%s,  grp_%d ,\tdon_%d  ,%s"
                         % (man["wells"][i]["out_dir"], i, token, i % 2, i % 3, note))
        text = "\n".join(lines) + "\n"
        with open(write_text(os.path.join(d, "lf.csv"), text), newline="") as fh:
            parsed = list(csv.reader(fh))
        assert set(len(r) for r in parsed) == {6}, "the fixture's notes are not one field each"
        assert parsed[1][5] == 'copy of well_00 (cmp of every file, 2026-11-17; "D-145")', parsed[1]
        a = CMAP.read_cohort_map(os.path.join(d, "lf.csv"), man)
        b = CMAP.read_cohort_map(write_text(os.path.join(d, "crlf.csv"), text.replace("\n", "\r\n")),
                                 man)
        assert a.excluded == tuple(man["wells"][i]["out_dir"] for i in range(6, 12)), a.excluded
        assert sorted(a.donor_of) == list(range(6)), a.donor_of
        assert a.donor_labels == ("don_0", "don_1", "don_2"), a.donor_labels   # blanks and tab stripped
        assert a.rep_group_labels == ("grp_0", "grp_1"), a.rep_group_labels
        assert a.donor_of == dict((i, i % 3) for i in range(6)), a.donor_of
        assert a.rep_group_of == dict((i, i % 2) for i in range(6)), a.rep_group_of
        for field in ("donor_labels", "rep_group_labels", "donor_of", "rep_group_of", "excluded"):
            assert getattr(a, field) == getattr(b, field), field
        assert a.sha256 != b.sha256                      # the digest is of the bytes
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_91_invariants():
    """The ids do not depend on the rows' order; two reads agree."""
    rng = np.random.default_rng(SEED + 1)
    n = 20
    man = synthetic_manifest(n)
    rows = [(man["wells"][i]["out_dir"], "d%d" % (i % 4), "g%d" % (i % 7), "yes" if i % 5 else "no")
            for i in range(n)]
    d = tempfile.mkdtemp(prefix="cohort_map_order_")
    try:
        ref = CMAP.read_cohort_map(write_text(os.path.join(d, "ref.csv"), map_text(rows)), man)
        again = CMAP.read_cohort_map(os.path.join(d, "ref.csv"), man)
        assert ref == again
        for trial in range(4):
            perm = rng.permutation(n)
            m = CMAP.read_cohort_map(write_text(os.path.join(d, "p%d.csv" % trial),
                                                map_text([rows[i] for i in perm])), man)
            for field in ("donor_labels", "rep_group_labels", "donor_of", "rep_group_of", "excluded"):
                assert getattr(m, field) == getattr(ref, field), (trial, field)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_93_refusals():
    """Each refusal of the entry on a small map, plus the reader's own (a header
    naming a column twice, a line the csv module cannot parse, no file, a
    manifest without wells or listing an out_dir twice)."""
    man = synthetic_manifest(3)
    o = [w["out_dir"] for w in man["wells"]]
    good = [(o[0], "a", "g", "yes"), (o[1], "b", "g", "no"), (o[2], "a", "h", "1")]
    d = tempfile.mkdtemp(prefix="cohort_map_refusals_")
    try:
        def at(name, text):
            return write_text(os.path.join(d, name), text)
        CMAP.read_cohort_map(at("good.csv", map_text(good)), man)        # the base map reads
        for col in HEADER:                                               # a required column missing
            keep = [i for i, c in enumerate(HEADER) if c != col]
            rows = [tuple(r[i] for i in keep) for r in good]
            expect_map_error("required column(s) %s missing" % col,
                             at("no_%s.csv" % col, map_text(rows, tuple(HEADER[i] for i in keep))), man)
        expect_map_error("required column(s) out_dir, donor, rep_group, include missing",
                         at("empty.csv", ""), man)
        expect_map_error("names no well of the manifest",
                         at("stray.csv", map_text(good + [("cls9/root_9/well_99", "a", "g", "yes")])), man)
        expect_map_error("1 well(s) of the manifest have no row: %s" % o[1],
                         at("missing.csv", map_text([good[0], good[2]])), man)
        msg = expect_map_error("a second row for the well", at("twice.csv", map_text(good + [good[1]])), man)
        assert "line 5" in msg and "first on line 3" in msg, msg
        for name, row, fragment in (("empty_donor", (o[1], "", "g", "no"), "empty donor label"),
                                    ("blank_group", (o[1], "b", "   ", "no"), "empty rep_group label"),
                                    ("bad_include", (o[1], "b", "g", "maybe"), "is not one of"),
                                    ("empty_include", (o[1], "b", "g", ""), "is not one of"),
                                    ("y_include", (o[1], "b", "g", "y"), "is not one of")):
            expect_map_error(fragment, at(name + ".csv", map_text([good[0], row, good[2]])), man)
        msg = expect_map_error("is not one of",                  # blank lines: skipped, and counted
                               at("blank.csv", map_text([good[0]]) + "\n\n%s,b,g,maybe\n%s,a,h,1\n"
                                  % (o[1], o[2])), man)
        assert "line 5:" in msg, msg
        expect_map_error("every well is excluded",
                         at("none.csv", map_text([(r[0], r[1], r[2], "no") for r in good])), man)
        utf8 = map_text(good).replace(",b,", ",b" + chr(0xE9) + ",").encode("utf-8")
        assert expect_map_error("not an ASCII file", at("utf8.csv", utf8), man).find("line 3") > 0
        expect_map_error("not an ASCII file", at("bom.csv", b"\xef\xbb\xbf" + map_text(good).encode()), man)
        expect_map_error("names column(s) donor twice",
                         at("dup.csv", "out_dir,donor,rep_group,include,donor\n"
                                       + "".join("%s,%s,%s,%s,x\n" % r for r in good)), man)
        expect_map_error("line 2", at("quote.csv", map_text([]) + '"%s"x,a,g,yes\n' % o[0]), man)
        expect_map_error("no such file", os.path.join(d, "absent.csv"), man)
        path = at("good2.csv", map_text(good))
        expect_map_error("lists no well", path, {"wells": []})
        expect_map_error("lists out_dir %r twice" % o[0], path,
                         {"wells": man["wells"] + [{"out_dir": o[0]}]})
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_93_field_counts():
    """Amendment 1: a row whose number of fields differs from the header's is
    refused, naming its line and both counts -- a longer row (an unquoted
    comma in a free-text column, the very column that is otherwise ignored;
    a trailing comma) and a shorter one (a missing trailing field; a row cut
    short); the same map with the free-text field quoted, as a spreadsheet
    writes it, reads, and a quoted label keeps its comma."""
    man = synthetic_manifest(3)
    o = [w["out_dir"] for w in man["wells"]]
    head = "out_dir,donor,rep_group,include,note\n"
    row0 = "%s,a,g,yes,first well\n" % o[0]
    row2 = "%s,a,h,1,third well\n" % o[2]
    d = tempfile.mkdtemp(prefix="cohort_map_fields_")
    try:
        def at(name, text):
            return write_text(os.path.join(d, name), text)
        for name, row1, n_fields in (
                ("long_note", "%s,b,g,no,copy of well 0, compared on 2026-11-17\n" % o[1], 6),
                ("trailing_comma", "%s,b,g,no,copy of well 0,\n" % o[1], 6),
                ("missing_note", "%s,b,g,no\n" % o[1], 4),
                ("cut_short", "%s,b\n" % o[1], 2)):
            msg = expect_map_error("line 3: %d fields, the header has 5" % n_fields,
                                   at(name + ".csv", head + row0 + row1 + row2), man)
            assert "must be quoted" in msg, msg
        expect_map_error("line 4: 6 fields, the header has 5",          # the line counts blank lines
                         at("after_blank.csv", head + row0 + "\n" + row2.replace(
                             "third well", "third well, again")), man)
        quoted = '%s,"b, the second",g,no,"copy of well 0, compared on 2026-11-17"\n' % o[1]
        m = CMAP.read_cohort_map(at("quoted.csv", head + row0 + quoted + row2), man)
        assert m.excluded == (o[1],) and m.donor_labels == ("a",), m
        m = CMAP.read_cohort_map(at("quoted_in.csv", head + row0 + quoted.replace(",no,", ",yes,")
                                    + row2), man)
        assert m.donor_labels == ("a", "b, the second"), m.donor_labels
        assert m.donor_of == {0: 0, 1: 1, 2: 0}, m.donor_of
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------- the bank (bank_real, v2) ---

def test_92_bank_with_map():
    f = MapFixture()
    try:
        arrays, info = REAL.build_real(f.cfg())
        assert not np.any(arrays["well"] == 0), "the excluded well has rows"
        want = expected_rows(f.dir, MAIN_DONOR, MAIN_GROUP)
        assert arrays["x"].shape == (63, W), arrays["x"].shape          # 9 + 27 + 27
        assert_arrays_equal(arrays, want)
        per_well = dict((w, int(np.sum(arrays["well"] == w))) for w in range(4))
        assert per_well == {0: 0, 1: 9, 2: 27, 3: 27}, per_well
        assert info["per_class"] == {
            "100N0A_wo": {"windows": 9, "below_floor": 9, "wells": 1},
            "50N50A": {"windows": 27, "below_floor": 0, "wells": 1},
            "70N30A": {"windows": 27, "below_floor": 0, "wells": 1}}, info["per_class"]
        assert info["n_wells"] == 3 and info["n_wells_manifest"] == 4, info
        assert info["roots"] == sorted(w[1] for w in FX.REAL_WELLS), info["roots"]   # every well's
    finally:
        f.close()


def test_92_cli_with_map():
    f = MapFixture()
    try:
        lb, _ln = EXP.stage1_modules()
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 0 and out[-1] == "REAL BANK OK", out[-3:]
        with open(f.map_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert out[0] == ("[real] cohort map %s sha256 %s: 1 of 4 wells excluded; 2 donors; "
                          "3 pairing groups" % (f.map_path, digest[:16])), out[0]
        assert out[1] == "[real]   excluded %s" % od(0), out[1]
        assert out[2].startswith("[real] 100N0A_wo ") and "1 wells,    9 windows,   9 below" in out[2], out[2]
        assert len(out) == 7, out                                       # 2 + 3 classes + wrote + OK
        p = os.path.join(f.cfg().real_bank_root, "shard_0000.npz")
        a, side = lb.read_shard(p)                                      # check_arrays, the digest
        assert lb.check_arrays(a) == 63
        assert a["rep_group"].dtype == np.int64 and a["rep_group"].shape == (63,)
        try:                                        # rep_group is checked, not only allowed
            lb.check_arrays(dict(a, rep_group=a["rep_group"].astype(np.float32)))
        except lb.ContractError:
            pass
        else:
            raise AssertionError("check_arrays let a float rep_group through")
        assert_arrays_equal(a, expected_rows(f.dir, MAIN_DONOR, MAIN_GROUP))
        assert side["latent_spec"]["cohort_map"] == {
            "sha256": digest, "donor_labels": ["arac", "ctrl"],
            "rep_group_labels": ["arac_30", "arac_50", "ctrl_b2"], "excluded": [od(0)],
            "n_wells_included": 3}, side["latent_spec"]["cohort_map"]
        assert side["latent_spec"]["kind"] == "campaign_real" and side["arm"] == "R"
        assert side["latent_spec"]["class_roots"] == sorted(w[1] for w in FX.REAL_WELLS)
        assert side["generator_sha256"] == SP.sha256_of(f.f.fx["manifest"])
        prov = side["provenance"]
        assert prov["real_version"] == 2 and prov["n_wells"] == 3, prov
        assert prov["per_class"]["100N0A_wo"] == {"windows": 9, "below_floor": 9, "wells": 1}
        assert prov["below_floor"] == 9, prov
    finally:
        f.close()


def test_93_refusals_cli():
    """A map refusal reaches the last line as REFUSED (exit 1) and writes no
    shard; so does a stage-1 latent_bank without the optional rep_group."""
    stray = MAIN_MAP + "x,1,g,100N0A_wo/Batch9/ptrain_9,d\n"
    for text, fragment in ((stray, "names no well of the manifest"),
                           (MAIN_MAP.replace("True", "no").replace(",1,", ",0,").replace("YES", "false"),
                            "every well is excluded"),
                           (MAIN_MAP.replace(",YES, ctrl_b2 ,%s,ctrl \n" % od(1), "\n"),
                            "have no row: %s" % od(1))):
        f = MapFixture(text)
        try:
            rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
            assert rc == 1 and out[-1].startswith("REFUSED: ") and fragment in out[-1], out[-1:]
            assert not os.path.exists(os.path.join(f.cfg().real_bank_root, "shard_0000.npz"))
        finally:
            f.close()
    f = MapFixture()
    try:
        os.remove(f.map_path)
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 1 and "no such file" in out[-1], out[-1:]
    finally:
        f.close()
    lb, ln = EXP.stage1_modules()
    old = types.ModuleType("latent_bank")
    old.__file__ = lb.__file__
    saved = EXP.stage1_modules
    EXP.stage1_modules = lambda: (old, ln)
    f = SP.Fixture()
    try:
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 1 and "predates the shard's optional rep_group" in out[-1], out[-1:]
    finally:
        EXP.stage1_modules = saved
        f.close()


def test_94_no_map_rows():
    """Without a map: the rows of section 2.6 (Block 8's oracle, re-derived),
    rep_group == donor == well; the line, the sidecar's null block, version 2."""
    f = SP.Fixture()
    try:
        assert f.cfg().cohort_map is None
        arrays, info = REAL.build_real(f.cfg())
        ident = dict((w, w) for w in range(4))
        assert_arrays_equal(arrays, expected_rows(f.dir, ident, ident))
        assert np.array_equal(arrays["rep_group"], arrays["donor"])
        assert np.array_equal(arrays["donor"], arrays["well"])
        assert info["n_wells"] == 4 and info["cohort_map"] is None, info["n_wells"]
        lb, _ln = EXP.stage1_modules()
        rc, out = SP.run_main(REAL.main, ["--profile", f.profile])
        assert rc == 0 and out[-1] == "REAL BANK OK", out[-3:]
        assert out[0] == "[real] cohort map: none (donor = rep_group = well)", out[0]
        assert out[1].startswith("[real] 100N0A_wo "), out[1]
        a, side = lb.read_shard(os.path.join(f.cfg().real_bank_root, "shard_0000.npz"))
        assert "cohort_map" in side["latent_spec"] and side["latent_spec"]["cohort_map"] is None
        assert side["provenance"]["real_version"] == 2 and side["provenance"]["n_wells"] == 4
        assert np.array_equal(a["rep_group"], a["well"])
    finally:
        f.close()


def test_94_no_map_equals_version_1():
    """Without a map, against REAL_VERSION 1's own code on the same fixture:
    the same arrays (rep_group aside), the same per-class counts and lines,
    the same sidecar but latent_spec's "cohort_map": null (and the digest it
    moves) and real_version."""
    v1, d = version_1_module()
    f = SP.Fixture()
    try:
        cfg = f.cfg()
        a1, i1 = v1.build_real(cfg)
        a2, i2 = REAL.build_real(cfg)
        assert_arrays_equal(a2, a1, skip=("rep_group",))
        assert sorted(a2) == sorted(list(a1) + ["rep_group"]), sorted(a2)
        for key in i1:
            assert i1[key] == i2[key], key
        lb, _ln = EXP.stage1_modules()
        r1 = os.path.join(f.dir, "real_v1")
        r2 = os.path.join(f.dir, "real_v2")
        rc1, out1 = SP.run_main(v1.main, ["--profile", f.profile, "--out-root", r1])
        rc2, out2 = SP.run_main(REAL.main, ["--profile", f.profile, "--out-root", r2])
        assert rc1 == 0 and rc2 == 0, (out1[-1:], out2[-1:])
        assert out2[0] == "[real] cohort map: none (donor = rep_group = well)"
        assert out2[1:4] == out1[0:3] and out2[-1] == out1[-1] == "REAL BANK OK", (out1, out2)
        s1 = lb.read_shard(os.path.join(r1, "shard_0000.npz"))
        s2 = lb.read_shard(os.path.join(r2, "shard_0000.npz"))
        assert_arrays_equal(s2[0], s1[0], skip=("rep_group",))
        assert s2[1]["latent_spec"] == dict(s1[1]["latent_spec"], cohort_map=None)
        for k in lb.CONTRACT_FIELDS:
            if k != "latent_spec":
                assert s1[1][k] == s2[1][k], k
        p1, p2 = s1[1]["provenance"], s2[1]["provenance"]
        assert (p1["real_version"], p2["real_version"]) == (1, 2)
        for k in ("per_class", "below_floor", "n_wells"):
            assert p1[k] == p2[k], k
    finally:
        f.close()
        shutil.rmtree(d, ignore_errors=True)


def test_95_linddup_map_accepted():
    """The committed LindDUP map (SBI, read-only) reads under amendment 1,
    against a manifest of its own out_dirs: every row holds the header's five
    fields (SBI f2f7203 took the commas out of the left-out wells' notes; an
    earlier checkout fails here, naming the lines). The expected sets come
    from csv.reader by column position."""
    path = os.path.join(os.environ.get("SBI_HPC_DIR", ""), "pipeline", "projects", "linddup",
                        "linddup_cohort_map.csv")
    if not os.path.isfile(path):
        raise unittest.SkipTest("no LindDUP map at %s (SBI_HPC_DIR)" % path)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = [(reader.line_num, r) for r in reader]
    header = rows[0][1]
    ragged = ["line %d: %d fields" % (n, len(r)) for n, r in rows[1:] if r and len(r) != len(header)]
    assert not ragged, ("%s: rows of another length than the header's %d: %s"
                        % (path, len(header), ragged))
    col = dict((name, i) for i, name in enumerate(header))
    body = [r for _n, r in rows[1:] if r]
    man = {"wells": [{"out_dir": r[col["out_dir"]]} for r in body]}
    m = CMAP.read_cohort_map(path, man)
    keep = [r[col["include"]].lower() in ("yes", "true", "1") for r in body]
    assert m.excluded == tuple(r[col["out_dir"]] for r, k in zip(body, keep) if not k), m.excluded
    assert m.donor_labels == tuple(sorted(set(r[col["donor"]].strip() for r, k in zip(body, keep) if k)))
    assert m.rep_group_labels == tuple(sorted(set(r[col["rep_group"]].strip()
                                                  for r, k in zip(body, keep) if k)))
    assert len(m.donor_of) + len(m.excluded) == len(body)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_determinism():
    raise unittest.SkipTest("no randomness in the code under test; the reads' agreement and "
                            "the row-order invariance are test_91_invariants")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append("SBI_HPC_DIR=%s" % os.environ.get("SBI_HPC_DIR", "(unset)"))
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
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
