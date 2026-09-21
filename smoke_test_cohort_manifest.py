"""
smoke_test_cohort_manifest.py -- Stage D end to end on a synthetic cohort.

    python3 smoke_test_cohort_manifest.py

Expect: ALL 14 CHECKS PASSED. Needs the DSN tree (for cohort.py, via
dsn_tree.py) and scipy; no real recordings, no torch.

WHAT IT DOES
------------
Writes three synthetic wells (100 electrodes each, MATLAB v5 rasters exactly
as the extractor's own stage-1 tests do) under a two-class cohort layout,
writes a config whose cohort block carries the davinci numbers
(w_size 0.01, gaussian_window 0.02, base 1, grid 48, 9 x 9, mfr 0.1), then
runs the REAL chain by subprocess -- list_extraction_jobs.py, then
run_channel_subset_extraction.py per well with the generated flags, exactly
the command run_extractor_array_mea.pbs issues -- and builds, writes, reads
back and asserts against the cohort manifest. Then it breaks things one at a
time and checks each is refused with a message naming the offender.

CHECKS
------
M1  list_extraction_jobs.py (torch-free) writes 3 rows and flags byte-
    identical to cohort_config.build_extra_flags
M2  the real extractor writes, per well, 9 subregion archives + a fragment
    carrying every PREPROCESSING_FIELD, extractor_version 3, a commit and
    manifest_version 1
M3  build_manifest passes: 3 wells x 9 = 27 units, measured == configured
M4  write_manifest / read_manifest round trip, digest verified
M5  assert_archive_matches_manifest passes on all 27 archives
M6  sim_preprocessing_from_manifest returns (0.01, 0.02, 9, recorded);
    assert_sim_geometry(9) passes
M7  NEGATIVE  assert_sim_geometry(4) is refused
M8  NEGATIVE  a fragment whose gaussian_window differs -> "NOT constant"
M9  NEGATIVE  a well with no fragment -> LegacyArchive
M10 NEGATIVE  a config whose w_size differs from what was applied ->
              "disagrees with the configured plan"
M11 NEGATIVE  an archive whose mfr_threshold differs -> ManifestError
M12 NEGATIVE  an archive without the recorded keys -> LegacyArchive
M13 NEGATIVE  an edited extraction_flags.sh -> "not byte-identical"
M14 NEGATIVE  a well missing one subregion archive -> multiplicity refused

HPC note (hpc-python-compat): pure ASCII, LF-only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_PASS, _FAIL = [], []


def check(name, fn):
    try:
        d = fn()
    except Exception as exc:                                  # noqa: BLE001
        _FAIL.append((name, "%s: %s" % (type(exc).__name__, exc)))
        print("  [FAIL] %-4s %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
        return
    _PASS.append((name, d))
    print("  [PASS] %-4s %s" % (name, d))


def _expect(fn, exc_type, needle):
    try:
        fn()
    except exc_type as exc:
        if needle not in str(exc):
            raise AssertionError("raised %s without %r: %s"
                                 % (exc_type.__name__, needle, exc))
        return str(exc).splitlines()[0][:90]
    raise AssertionError("expected %s mentioning %r; nothing raised"
                         % (exc_type.__name__, needle))


# --------------------------------------------------------------------------- #
# synthetic cohort
# --------------------------------------------------------------------------- #
FS_RAW = 10110.09
T_S = 20.0
N_ELEC = 100
COHORT = {
    "fs_raw": FS_RAW, "index_base": 1, "grid_width": 48, "n_subsets": 9,
    "electrodes_per_subset": 9, "mfr_threshold": 0.1, "w_size": 0.01,
    "gaussian_window": 0.02,
}


def write_well(folder, rng):
    import scipy.io as sio
    sys.path.insert(0, os.path.join(_HERE, "extractor"))
    from channel_subset_extraction import PTRAIN_VARNAME
    os.makedirs(folder, exist_ok=True)
    n = int(round(T_S * FS_RAW))
    for k in range(1, N_ELEC + 1):                     # base 1, grid 48
        n_sp = int(rng.integers(20, 80))                # >= 1 Hz, above mfr 0.1
        idx = np.sort(rng.choice(n, size=n_sp, replace=False))
        r = np.zeros((n, 1), dtype=np.uint8); r[idx, 0] = 1
        sio.savemat(os.path.join(folder, "ptrain_%d.mat" % k),
                    {PTRAIN_VARNAME: r}, do_compression=True, format="5")


def build_cohort(root, rng):
    wells = [("DATA_C/Batch1", "ptrain_A1"), ("DATA_C/Batch1", "ptrain_A2"),
             ("DATA_P/Batch3", "ptrain_B1")]
    for sub, w in wells:
        write_well(os.path.join(root, sub, w), rng)
    cfg = {"cohort": dict(COHORT, **{
        "class_roots": {"0": [os.path.join(root, "DATA_C/Batch1")],
                        "1": [os.path.join(root, "DATA_P/Batch3")]},
        "class_names": ["control", "pathological"],
        "well_glob": "ptrain_*",
        "extract_root": os.path.join(root, "extracted"),
        "extract_layout": "{class_name}/{root_name}/{well}",
        "culture_template": "{root_name}__{well}",
    })}
    with open(os.path.join(root, "cfg.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    return os.path.join(root, "cfg.json")


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("rc=%d: %s\n%s\n%s" % (r.returncode, " ".join(cmd),
                                                  r.stdout[-800:], r.stderr[-800:]))
    return r.stdout


class State(object):
    pass


S = State()


def m1_list_jobs():
    S.root = tempfile.mkdtemp(prefix="stageD_")
    S.cfg = build_cohort(S.root, np.random.default_rng(7))
    S.extract_root = os.path.join(S.root, "extracted_v2")
    S.tsv = os.path.join(S.root, "m.tsv"); S.flags = os.path.join(S.root, "f.sh")
    out = run([sys.executable, "list_extraction_jobs.py", "--config", S.cfg,
               "--out-manifest", S.tsv, "--out-flags", S.flags,
               "--extract-root", S.extract_root], os.path.join(_HERE, "extractor"))
    if "wrote 3 well(s)" not in out:
        raise AssertionError(out)
    import cohort_config as CC
    c = CC.load_cohort(S.cfg, extract_root=S.extract_root)
    if open(S.flags, "rb").read() != CC.build_extra_flags(c).encode():
        raise AssertionError("flags differ from build_extra_flags")
    S.rows = [ln.rstrip("\n").split("\t") for ln in open(S.tsv) if ln.strip()]
    return "3 rows; flags == build_extra_flags; out_dirs under extracted_v2"


def m2_extract():
    flags = open(S.flags).read().split('EXTRA_FLAGS="')[1].split('"')[0].split()
    for folder, out_dir, _c in S.rows:
        run([sys.executable, "run_channel_subset_extraction.py", folder,
             "--out-dir", out_dir, "--mode", "per_region_single", "--no-plots"]
            + flags, os.path.join(_HERE, "extractor"))
    from cohort_manifest import read_fragment
    for _f, out_dir, _c in S.rows:
        fr = read_fragment(out_dir)
        m = fr["meta"]
        if fr["n_archives"] != 9:
            raise AssertionError("%s has %d archives" % (out_dir, fr["n_archives"]))
        if m.get("extractor_version") != "run_channel_subset_extraction/3":
            raise AssertionError("extractor_version %r" % m.get("extractor_version"))
        if m.get("manifest_version") != 1 or "extractor_commit" not in m:
            raise AssertionError("fragment lacks Stage D keys: %r" % sorted(m))
    return "3 wells x 9 archives; fragments carry version 3, commit, manifest_version 1"


def m3_build():
    from cohort_manifest import build_manifest
    S.doc = build_manifest(S.cfg, S.tsv, S.flags, extract_root=S.extract_root)
    d = S.doc
    if (d["n_wells"], d["n_subsets_per_well"], d["n_units"]) != (3, 9, 27):
        raise AssertionError((d["n_wells"], d["n_subsets_per_well"], d["n_units"]))
    for k, v in COHORT.items():
        if d["preprocessing"][k] != v:
            raise AssertionError("%s: %r != %r" % (k, d["preprocessing"][k], v))
    if d["derived"] != {"fs_ifr": 100.0, "sigma_sm_bins": 2.0}:
        raise AssertionError(d["derived"])
    return "3 x 9 = 27 units; measured == configured on all 8 fields; fs_ifr 100, sigma 2 bins"


def m4_roundtrip():
    from cohort_manifest import write_manifest, read_manifest, MANIFEST_NAME
    S.mpath = os.path.join(S.extract_root, MANIFEST_NAME)
    dg = write_manifest(S.doc, S.mpath)
    S.man = read_manifest(S.mpath)
    if S.man["_digest"] != dg or S.man["preprocessing"] != S.doc["preprocessing"]:
        raise AssertionError("round trip")
    return "written, read back, sha256 %s verified" % dg[:12]


def m5_assert_all():
    from cohort_manifest import assert_archive_matches_manifest
    n = 0
    for _f, out_dir, _c in S.rows:
        for nm in sorted(os.listdir(out_dir)):
            if nm.startswith("trace_subregion_") and nm.endswith(".npz"):
                assert_archive_matches_manifest(os.path.join(out_dir, nm), S.man, what=nm)
                n += 1
    if n != 27:
        raise AssertionError(n)
    return "all 27 archives match the manifest"


def m6_sim():
    from cohort_manifest import sim_preprocessing_from_manifest, assert_sim_geometry
    dt, sg, ne, rec = sim_preprocessing_from_manifest(S.man)
    if (dt, sg, ne) != (0.01, 0.02, 9) or rec["mfr_threshold"] != 0.1 or rec["n_subsets"] != 9:
        raise AssertionError((dt, sg, ne, rec))
    assert_sim_geometry(S.man, 9)
    return "conform dt=0.01 sigma=0.02; assert n_e=9 ok; mfr/n_subsets recorded only"


def m7_sim_geometry_refused():
    from cohort_manifest import assert_sim_geometry, ManifestError
    return _expect(lambda: assert_sim_geometry(S.man, 4), ManifestError, "n_e = 4")


def _with_copy(fn):
    """Run fn on a throwaway copy of the extracted tree + tsv."""
    root2 = tempfile.mkdtemp(prefix="stageD_neg_")
    ex2 = os.path.join(root2, "extracted_v2")
    shutil.copytree(S.extract_root, ex2)
    rows2 = [(f, o.replace(S.extract_root, ex2), c) for f, o, c in S.rows]
    tsv2 = os.path.join(root2, "m.tsv")
    with open(tsv2, "w") as fh:
        for r in rows2:
            fh.write("\t".join(r) + "\n")
    try:
        return fn(root2, ex2, rows2, tsv2)
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def m8_fragment_not_constant():
    from cohort_manifest import build_manifest, ManifestError
    def go(root2, ex2, rows2, tsv2):
        fp = os.path.join(rows2[1][1], "traces_meta.json")
        m = json.load(open(fp)); m["gaussian_window"] = 0.04; json.dump(m, open(fp, "w"))
        return _expect(lambda: build_manifest(S.cfg, tsv2, S.flags, extract_root=ex2),
                       ManifestError, "'gaussian_window' is NOT constant")
    return _with_copy(go)


def m9_legacy_well():
    from cohort_manifest import build_manifest, LegacyArchive
    def go(root2, ex2, rows2, tsv2):
        os.remove(os.path.join(rows2[0][1], "traces_meta.json"))
        return _expect(lambda: build_manifest(S.cfg, tsv2, S.flags, extract_root=ex2),
                       LegacyArchive, "has no traces_meta.json")
    return _with_copy(go)


def m10_plan_disagrees():
    from cohort_manifest import build_manifest, ManifestError
    cfg2 = os.path.join(S.root, "cfg_wsize.json")
    d = json.load(open(S.cfg)); d["cohort"]["w_size"] = 0.02; json.dump(d, open(cfg2, "w"))
    return _expect(lambda: build_manifest(cfg2, S.tsv, S.flags, extract_root=S.extract_root),
                   ManifestError, "disagrees with the configured plan")


def m11_archive_disagrees():
    from cohort_manifest import assert_archive_matches_manifest, ManifestError
    p = os.path.join(S.rows[0][1], "trace_subregion_00.npz")
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["mfr_threshold"] = np.float64(0.5)
    return _expect(lambda: assert_archive_matches_manifest(d, S.man, what="tampered"),
                   ManifestError, "('mfr_threshold', 0.5, 0.1)")


def m12_archive_legacy():
    from cohort_manifest import assert_archive_matches_manifest, LegacyArchive
    d = {"ifr_trace": np.zeros(10, np.float32), "fs_ifr": 100.0}
    return _expect(lambda: assert_archive_matches_manifest(d, S.man), LegacyArchive, "lacks")


def m13_flags_edited():
    from cohort_manifest import build_manifest, ManifestError
    f2 = os.path.join(S.root, "f_edited.sh")
    s = open(S.flags).read().replace("--mfr-threshold 0.1", "--mfr-threshold 0.2")
    open(f2, "w").write(s)
    return _expect(lambda: build_manifest(S.cfg, S.tsv, f2, extract_root=S.extract_root),
                   ManifestError, "not byte-identical")


def m14_missing_archive():
    from cohort_manifest import build_manifest, ManifestError
    def go(root2, ex2, rows2, tsv2):
        os.remove(os.path.join(rows2[2][1], "trace_subregion_08.npz"))
        return _expect(lambda: build_manifest(S.cfg, tsv2, S.flags, extract_root=ex2),
                       ManifestError, "do not hold exactly n_subsets = 9")
    return _with_copy(go)


def main():
    print("smoke_test_cohort_manifest -- Stage D on a synthetic 3-well cohort")
    print("-" * 70)
    for nm, fn in (("M1", m1_list_jobs), ("M2", m2_extract), ("M3", m3_build),
                   ("M4", m4_roundtrip), ("M5", m5_assert_all), ("M6", m6_sim),
                   ("M7", m7_sim_geometry_refused), ("M8", m8_fragment_not_constant),
                   ("M9", m9_legacy_well), ("M10", m10_plan_disagrees),
                   ("M11", m11_archive_disagrees), ("M12", m12_archive_legacy),
                   ("M13", m13_flags_edited), ("M14", m14_missing_archive)):
        check(nm, fn)
    shutil.rmtree(getattr(S, "root", "/nonexistent"), ignore_errors=True)
    print("-" * 70)
    if _FAIL:
        print("FAILED %d of %d" % (len(_FAIL), len(_PASS) + len(_FAIL)))
        return 1
    print("ALL %d CHECKS PASSED" % len(_PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
