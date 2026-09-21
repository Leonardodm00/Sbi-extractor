"""
cohort_manifest.py -- the cohort manifest: build it from the extraction
fragments, read it, and assert an archive (or a sim export) against it.

Stage D (2026-09-21). One module owns every rule about the manifest so the
aggregation job, the real-arm export and the sim-arm export cannot disagree.

THE DESIGN (handoff S6)
-----------------------
    CohortConfig (plan)  -- what the extraction was configured to do
    per-archive fragment  -- what each array task actually applied
                             (traces_meta.json, extractor version >= 3)
    cohort_manifest.json  -- the trace: written ONLY if every fragment agrees
                             with every other AND with the plan; otherwise
                             build_manifest() raises and nothing is written.

Mismatch RAISES. There is no --assume-preprocessing. An archive without the
recorded keys (extractor version 1, the archives of record before Stage D)
is a LegacyArchive and is not exportable.

Gating fields, real arm (raise on mismatch): every name in
cohort_config.PREPROCESSING_FIELDS, plus n_units and manifest_version.

Sim arm (decision 2026-09-21): the export CONFORMS w_size and gaussian_window
(Delta_t, sigma_sm) to the manifest; ASSERTS electrodes_per_subset against
the campaign's own n_e (geometry, which can only be checked); and RECORDS
mfr_threshold and n_subsets without applying either -- the sim tile has no
electrode-level MFR rule and no multiplicity, and never will.

Pure ASCII, LF only (hpc-python-compat). numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from cohort_config import (PREPROCESSING_FIELDS, SUBREGION_PREFIX,     # noqa: E402
                           load_cohort, preprocessing_dict)

__all__ = [
    "MANIFEST_VERSION", "FRAGMENT_NAME", "MANIFEST_NAME",
    "ManifestError", "LegacyArchive",
    "read_fragment", "build_manifest", "write_manifest", "read_manifest",
    "assert_archive_matches_manifest", "sim_preprocessing_from_manifest",
    "assert_sim_geometry", "sha256_file",
]

MANIFEST_VERSION = 1
FRAGMENT_NAME = "traces_meta.json"
MANIFEST_NAME = "cohort_manifest.json"

# Written into the manifest so the reader knows which fields are which.
GATING_REAL = tuple(PREPROCESSING_FIELDS) + ("n_units", "manifest_version")
CONFORMED_SIM = ("w_size", "gaussian_window")
ASSERTED_SIM = ("electrodes_per_subset",)
RECORDED_SIM = ("mfr_threshold", "n_subsets", "fs_raw", "index_base", "grid_width")


class ManifestError(RuntimeError):
    """The fragments disagree with each other, with the plan, or the manifest
    disagrees with an archive. Nothing is written or exported past this."""


class LegacyArchive(ManifestError):
    """An archive without recorded preprocessing (extractor version < 2).
    Not exportable under the manifest design: re-extract it."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# fragments
# --------------------------------------------------------------------------- #
def read_fragment(out_dir):
    """The per-well fragment (traces_meta.json) plus what the directory holds.

    Returns dict with keys: meta (the JSON), n_archives (count of
    trace_subregion_*.npz), out_dir. Raises LegacyArchive if the fragment is
    absent or predates the recorded-preprocessing schema, ManifestError if a
    PREPROCESSING_FIELD is missing from it.
    """
    fp = os.path.join(out_dir, FRAGMENT_NAME)
    if not os.path.isfile(fp):
        raise LegacyArchive(
            "%s has no %s: it was written by an extractor that recorded no "
            "preprocessing (version 1). Re-extract it; it is not exportable."
            % (out_dir, FRAGMENT_NAME))
    with open(fp, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    missing = [k for k in PREPROCESSING_FIELDS if k not in meta]
    if missing:
        raise LegacyArchive(
            "%s lacks %s: extractor_version %r predates the recorded-"
            "preprocessing schema. Re-extract it."
            % (fp, missing, meta.get("extractor_version")))
    n_arch = sum(1 for n in os.listdir(out_dir)
                 if n.startswith(SUBREGION_PREFIX) and n.endswith(".npz"))
    return {"meta": meta, "n_archives": n_arch, "out_dir": out_dir}


def _read_rows(manifest_tsv):
    rows = []
    with open(manifest_tsv, "r", encoding="ascii") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) != 3:
                raise ManifestError("%s: malformed line %r" % (manifest_tsv, ln))
            rows.append(tuple(parts))
    if not rows:
        raise ManifestError("%s is empty" % manifest_tsv)
    return rows


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_manifest(config_json, manifest_tsv, flags_path, extract_root=None,
                   extractor_commit=None):
    """Aggregate every well's fragment into ONE cohort manifest, or raise.

    Asserts, in this order, each with a message naming the offender:
      1. every well in manifest_tsv has a fragment with every PREPROCESSING_FIELD
      2. every PREPROCESSING_FIELD, fs_ifr, extractor_version and
         manifest_version is CONSTANT across wells
      3. the constant measured values EQUAL the configured ones (the plan)
      4. every well has exactly n_subsets archives, so
         n_units == n_wells * n_subsets
      5. the tracked extraction_flags.sh is byte-identical to what the plan
         would generate now (the flags the array job actually sourced)

    Returns the manifest dict. Does NOT write; write_manifest() does.
    """
    from cohort_config import build_extra_flags
    cohort = load_cohort(config_json, extract_root=extract_root)
    plan = preprocessing_dict(cohort)
    rows = _read_rows(manifest_tsv)

    # 1. every well has a usable fragment. EVERY offender is collected before
    # raising, not just the first. Measured on davinci 2026-09-21: PBS Pro
    # does not propagate a subjob's exit status into the array job's, so
    # depend=afterok on the extraction array releases THIS job even when
    # extraction tasks died (probe_array_depend.sh, verdict NOT GATED). A
    # partial cohort is therefore the EXPECTED failure path, not a rare one,
    # and reporting one well per run would mean one re-run per lost task.
    frags = []
    unusable = []
    for folder, out_dir, culture in rows:
        try:
            f = read_fragment(out_dir)
        except LegacyArchive as exc:
            unusable.append((culture, str(exc)))
            continue
        f["culture"] = culture
        f["folder"] = folder
        frags.append(f)
    if unusable:
        detail = "".join("\n    %s\n        %s" % (c, m) for c, m in unusable)
        raise LegacyArchive(
            "%d of %d well(s) have no usable %s, so the cohort is INCOMPLETE "
            "and no manifest is written. Re-extract these wells (the array "
            "task that owns each one is the row of the same out_dir in "
            "extraction_manifest.tsv):%s"
            % (len(unusable), len(rows), FRAGMENT_NAME, detail))

    # 2. constancy
    def const(key, getter):
        vals = {}
        for f in frags:
            v = getter(f)
            vals.setdefault(json.dumps(v, sort_keys=True), []).append(f["culture"])
        if len(vals) != 1:
            desc = "; ".join("%s in %d well(s) e.g. %s" % (k, len(c), c[0])
                             for k, c in sorted(vals.items()))
            raise ManifestError("%r is NOT constant across the cohort: %s"
                                % (key, desc))
        return getter(frags[0])

    measured = {k: const(k, lambda f, k=k: f["meta"][k]) for k in PREPROCESSING_FIELDS}
    fs_ifr = const("fs_ifr", lambda f: f["meta"].get("fs_ifr"))
    ext_ver = const("extractor_version", lambda f: f["meta"].get("extractor_version"))
    frag_mv = const("manifest_version", lambda f: f["meta"].get("manifest_version"))
    ext_commit = const("extractor_commit", lambda f: f["meta"].get("extractor_commit"))
    if frag_mv is None:
        raise LegacyArchive(
            "fragments carry no manifest_version (extractor_version %r); "
            "Stage D needs extractor version >= 3" % (ext_ver,))
    if int(frag_mv) != MANIFEST_VERSION:
        raise ManifestError("fragment manifest_version %r != this module's %d"
                            % (frag_mv, MANIFEST_VERSION))

    # 3. measured == configured
    bad = []
    for k in PREPROCESSING_FIELDS:
        m, p = measured[k], plan[k]
        same = (int(m) == int(p)) if isinstance(p, int) else np.isclose(
            float(m), float(p), rtol=0.0, atol=0.0)
        if not same:
            bad.append((k, m, p))
    if bad:
        raise ManifestError(
            "measured preprocessing disagrees with the configured plan on %s "
            "(field, measured, configured). The manifest is NOT written."
            % (bad,))
    if fs_ifr is None or abs(float(fs_ifr) * float(measured["w_size"]) - 1.0) > 1e-9:
        raise ManifestError("fs_ifr %r is not 1 / w_size (%r)"
                            % (fs_ifr, measured["w_size"]))

    # 4. multiplicity
    n_sub = int(measured["n_subsets"])
    short = [(f["culture"], f["n_archives"]) for f in frags if f["n_archives"] != n_sub]
    if short:
        raise ManifestError(
            "%d well(s) do not hold exactly n_subsets = %d subregion archives: %s"
            % (len(short), n_sub, short[:5]))
    n_wells = len(frags)
    n_units = sum(f["n_archives"] for f in frags)
    assert n_units == n_wells * n_sub

    # 5. the flags the array sourced are what the plan generates
    if not os.path.isfile(flags_path):
        raise ManifestError("flags file missing: %s" % flags_path)
    with open(flags_path, "rb") as fh:
        tracked = fh.read()
    regen = build_extra_flags(cohort).encode("ascii")
    if tracked != regen:
        raise ManifestError(
            "%s is not byte-identical to what the plan generates now; the "
            "array job sourced flags that do not match the config" % flags_path)

    root = os.path.abspath(str(cohort.extract_root))
    wells = []
    for f in frags:
        m = f["meta"]
        wells.append({
            "culture": f["culture"],
            "out_dir": os.path.relpath(f["out_dir"], root),
            "n_archives": int(f["n_archives"]),
            "T_rec": m.get("T_rec"),
            "n_present": m.get("n_present"),
            "n_samples_raw": m.get("n_samples_raw"),
        })

    with open(config_json, "r", encoding="utf-8") as fh:
        cohort_block = json.load(fh).get("cohort", {})

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "extract_root": root,
        "preprocessing": measured,
        "derived": {"fs_ifr": float(fs_ifr),
                    "sigma_sm_bins": float(measured["gaussian_window"]) / float(measured["w_size"])},
        "n_wells": n_wells,
        "n_subsets_per_well": n_sub,
        "n_units": n_units,
        "classes": [cohort.name_of_class(i) for i in range(cohort.n_classes())],
        "extractor_version": ext_ver,
        "extractor_commit": extractor_commit or ext_commit,
        "sha256": {
            "extraction_flags_sh": sha256_file(flags_path),
            "config_file": sha256_file(config_json),
            "cohort_block": _sha256_text(json.dumps(cohort_block, sort_keys=True)),
        },
        "gating": {"real": list(GATING_REAL),
                   "sim_conformed": list(CONFORMED_SIM),
                   "sim_asserted": list(ASSERTED_SIM),
                   "sim_recorded": list(RECORDED_SIM)},
        "wells": wells,
        "T_rec_range": [min(w["T_rec"] for w in wells if w["T_rec"] is not None),
                        max(w["T_rec"] for w in wells if w["T_rec"] is not None)],
    }


def write_manifest(doc, out_path):
    """Write the manifest and its .sha256 sidecar. Returns the digest."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    blob = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with open(out_path, "w", encoding="ascii") as fh:
        fh.write(blob)
    digest = _sha256_text(blob)
    with open(out_path + ".sha256", "w", encoding="ascii") as fh:
        fh.write("%s  %s\n" % (digest, os.path.basename(out_path)))
    return digest


def read_manifest(path):
    """Load a manifest, verify its sidecar digest, check its version."""
    with open(path, "r", encoding="ascii") as fh:
        blob = fh.read()
    doc = json.loads(blob)
    side = path + ".sha256"
    if os.path.isfile(side):
        with open(side, "r", encoding="ascii") as fh:
            want = fh.read().split()[0]
        got = _sha256_text(blob)
        if want != got:
            raise ManifestError("%s does not match its .sha256 (%s != %s)"
                                % (path, got[:12], want[:12]))
    if int(doc.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ManifestError("%s is manifest_version %r, this code reads %d"
                            % (path, doc.get("manifest_version"), MANIFEST_VERSION))
    doc["_digest"] = _sha256_text(blob)
    doc["_path"] = os.path.abspath(path)
    return doc


# --------------------------------------------------------------------------- #
# assertions used by the exports and the training loader
# --------------------------------------------------------------------------- #
def _native(v):
    """numpy scalar / 0-d array -> plain Python int or float, for messages
    and sidecars. Anything else is returned unchanged."""
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, TypeError):
            return v
    if isinstance(v, (bool, int)):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return v


def _archive_preprocessing(archive):
    """PREPROCESSING_FIELDS out of an .npz path, an npz object or a dict,
    as plain Python numbers (see _native)."""
    if isinstance(archive, str):
        with np.load(archive, allow_pickle=False) as z:
            keys = set(z.files)
            missing = [k for k in PREPROCESSING_FIELDS if k not in keys]
            if missing:
                raise LegacyArchive(
                    "%s lacks %s: written by an extractor that recorded no "
                    "preprocessing. Not exportable; re-extract." % (archive, missing))
            return {k: _native(z[k]) for k in PREPROCESSING_FIELDS}
    if hasattr(archive, "files"):
        keys = set(archive.files)
        missing = [k for k in PREPROCESSING_FIELDS if k not in keys]
        if missing:
            raise LegacyArchive("archive lacks %s" % (missing,))
        return {k: _native(archive[k]) for k in PREPROCESSING_FIELDS}
    missing = [k for k in PREPROCESSING_FIELDS if k not in archive]
    if missing:
        raise LegacyArchive("archive dict lacks %s" % (missing,))
    return {k: _native(archive[k]) for k in PREPROCESSING_FIELDS}


def assert_archive_matches_manifest(archive, manifest, what="archive"):
    """REAL ARM. Every PREPROCESSING_FIELD in the archive equals the manifest's.

    `archive` is a path to a trace_subregion_XX.npz / traces.npz, an open npz,
    or a dict of its keys. `manifest` is read_manifest()'s dict. Raises
    ManifestError naming every disagreeing field; LegacyArchive if the archive
    recorded nothing. Returns the archive's preprocessing dict on success.
    """
    got = _archive_preprocessing(archive)
    want = manifest["preprocessing"]
    bad = []
    for k in PREPROCESSING_FIELDS:
        g, w = got[k], want[k]
        same = (int(g) == int(w)) if isinstance(w, int) else np.isclose(
            float(g), float(w), rtol=0.0, atol=0.0)
        if not same:
            bad.append((k, g, w))
    if bad:
        raise ManifestError(
            "%s disagrees with cohort manifest %s on %s (field, archive, "
            "manifest). Not exported." % (what, manifest.get("_path", "?"), bad))
    return got


def sim_preprocessing_from_manifest(manifest):
    """SIM ARM. The values the sim export must CONFORM to, and those it only
    records. Returns (dt, sigma_sm, n_e_expected, recorded) with recorded a
    dict of the RECORDED_SIM fields for the sidecar."""
    p = manifest["preprocessing"]
    return (float(p["w_size"]), float(p["gaussian_window"]),
            int(p["electrodes_per_subset"]),
            {k: p[k] for k in RECORDED_SIM})


def assert_sim_geometry(manifest, n_e, what="sim campaign"):
    """SIM ARM. The campaign's electrode count equals the manifest's
    electrodes_per_subset. n_e is geometry (electrode_centers) and can only be
    checked, never conformed. Raises ManifestError."""
    want = int(manifest["preprocessing"]["electrodes_per_subset"])
    if int(n_e) != want:
        raise ManifestError(
            "%s has n_e = %d electrodes but the cohort manifest's "
            "electrodes_per_subset is %d. The pooled IFR is a mean over "
            "exactly n_e electrodes; these arms are not comparable."
            % (what, int(n_e), want))
    return want


# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="build (or check) the cohort manifest")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True, help="extraction_manifest.tsv")
    ap.add_argument("--flags", required=True, help="extraction_flags.sh")
    ap.add_argument("--extract-root", default=None)
    ap.add_argument("--out", default=None,
                    help="default <extract_root>/%s" % MANIFEST_NAME)
    ap.add_argument("--check-only", action="store_true",
                    help="run every assertion, write nothing")
    ap.add_argument("--extractor-commit", default=None)
    a = ap.parse_args(argv)
    try:
        doc = build_manifest(a.config, a.manifest, a.flags,
                             extract_root=a.extract_root,
                             extractor_commit=a.extractor_commit)
    except ManifestError as exc:
        print("REFUSED: %s" % exc)
        return 1
    print("cohort manifest: %d wells x %d subregions = %d units; "
          "preprocessing %s; extractor %s @ %s"
          % (doc["n_wells"], doc["n_subsets_per_well"], doc["n_units"],
             json.dumps(doc["preprocessing"], sort_keys=True),
             doc["extractor_version"], doc["extractor_commit"]))
    if a.check_only:
        print("(check-only; nothing written)")
        return 0
    out = a.out or os.path.join(doc["extract_root"], MANIFEST_NAME)
    digest = write_manifest(doc, out)
    print("wrote %s  sha256 %s" % (out, digest[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
