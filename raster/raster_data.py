"""
raster_data.py -- the data layer of the real-cohort raster plots. No plotting.

What it reads, in order of authority
------------------------------------
1. The cohort manifest of record, extracted_v2/cohort_manifest.json, checked
   against its .sha256 sidecar by cohort_manifest.read_manifest. It names
   the wells and carries the preprocessing every number here uses: fs_raw,
   index_base, grid_width, mfr_threshold (decision D-002: the manifest is
   the preprocessing contract).
2. Each well's extraction record, traces_meta.json in the well's out_dir,
   read by cohort_manifest.read_fragment. It names the raw folder the
   extraction read (source_folder) and records n_present, n_samples_raw and
   discarded (the present electrodes whose MFR is below mfr_threshold).
3. The raw 3Brain rasters of one well, ptrain_<k>.mat, read by the
   extractor's own load_ptrain_folder; the MFR is the extractor's own
   mean_firing_rates. Neither is re-implemented here.

What it writes
--------------
One cache per well, <out_root>/cache/<culture>.npz: every spike of the well
as (raw sample index, electrode linear index), sorted by time, plus the
present electrodes, their spike counts, the active mask and the provenance.
Before a cache is written, the spike table is checked against the well's
extraction record -- same raster length, same number of present electrodes,
same sub-threshold set. A well that fails is refused, never plotted.

The plotting side (raster_plot.py, raster_viewer.py) reads only caches and
wells.tsv, so it needs neither the manifest nor the raw data nor the DSN
tree: copy <out_root> to any machine with numpy and matplotlib to view it.

Decisions implemented (claude/SBI_decisions_and_ideas_log.md)
-------------------------------------------------------------
D-026  select_typical_wells: per class and per batch folder, the lower
       median of n_active = n_present - len(discarded).
D-027  raster_view: the rows are the active electrodes (MFR over the WHOLE
       recording >= mfr_threshold), in ascending linear index, which is
       row-major order on the extractor's grid mapping; grid row 0 first.
D-028  window_bounds: the half-open window [t0, t1) in seconds,
       0 <= t0 < t1 <= T_rec; a spike at raw sample s is at s / fs_raw.
D-029  nothing here computes a rate trace.

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_EXTRACTOR = os.path.join(_REPO, "extractor")
for _p in (_HERE, _REPO, _EXTRACTOR):
    if _p not in sys.path:
        sys.path.append(_p)

__all__ = [
    "CACHE_VERSION", "TOOL_VERSION", "DEFAULT_MANIFEST", "DEFAULT_OUT_ROOT",
    "RULE_TYPICAL", "RULE_NAMED",
    "RasterDataError", "WellRecord", "CohortInfo", "SpikeTable", "RasterView",
    "load_cohort", "select_typical_wells", "select_named_wells",
    "group_by_class", "write_wells_tsv", "read_wells_tsv",
    "build_spike_table", "check_against_record", "cache_path", "save_cache",
    "load_cache", "cache_is_current", "build_and_save_cache",
    "window_bounds", "window_slice", "raster_view", "fmt_window",
    "common_preprocessing", "figure_sidecar",
]

CACHE_VERSION = 1
TOOL_VERSION = "raster_plots/1"

# The cohort manifest of record and the default output root. The manifest's
# path has a space in it ("Deep Summary Network"); nothing here passes it
# through qsub -v, so that is harmless. The output root has none.
DEFAULT_MANIFEST = os.path.join(
    os.path.expanduser("~"), "Deep Summary Network", "Deep_bio",
    "extracted_v2", "cohort_manifest.json")
DEFAULT_OUT_ROOT = os.path.join(os.path.expanduser("~"), "raster_plots")

RULE_TYPICAL = "D-026: per class and batch folder, lower median of n_active"
RULE_NAMED = "named with --wells"

WELLS_TSV_COLUMNS = ("culture", "class_name", "batch", "well", "n_present",
                     "n_active", "n_samples_raw", "source_folder",
                     "selected_by")


class RasterDataError(RuntimeError):
    """Every refusal of this tool: a layout it does not recognise, a record
    that disagrees with the data, a window outside the recording."""


# --------------------------------------------------------------------------- #
# the cohort: manifest + extraction records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WellRecord:
    """One well of the cohort, as its extraction recorded it.

    n_active = n_present - len(discarded): the size of the extractor's valid
    set V (MFR over the whole recording >= mfr_threshold), read from the
    record, not recomputed. raw_root is the folder that holds the well's
    folder (a batch folder of the cohort config's class_roots).
    """

    culture: str
    class_name: str
    batch: str
    well: str
    out_dir: str
    source_folder: str
    raw_root: str
    n_present: int
    n_active: int
    n_samples_raw: int
    T_rec: float
    discarded: Tuple[int, ...]


@dataclass(frozen=True)
class CohortInfo:
    """The manifest's identity and the four preprocessing numbers used here."""

    manifest_path: str
    manifest_digest: str
    classes: Tuple[str, ...]
    fs_raw: float
    index_base: int
    grid_width: int
    mfr_threshold: float
    wells: Tuple[WellRecord, ...]


def _same_number(a, b):
    """Exact equality for the recorded preprocessing (ints or floats)."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def load_cohort(manifest_path=DEFAULT_MANIFEST, require_sidecar=True):
    """Read the cohort manifest and every well's extraction record.

    Each well's out_dir must read <class>/<batch>/<well> (the cohort
    config's extract_layout "{class_name}/{root_name}/{well}") and its
    culture must be "<batch>__<well>" (culture_template "{root_name}__{well}");
    anything else raises, because the class and batch of a well are read from
    that layout. The fragment's fs_raw, index_base, grid_width and
    mfr_threshold must equal the manifest's, and its n_present and
    n_samples_raw the manifest's per-well copy.

    Imports cohort_manifest lazily: that module imports cohort_config, which
    needs the DSN tree (SBI_HPC_DIR or artifacts/sbi_hpc). Only select and
    cache need this function; plot and view never call it.
    """
    manifest_path = os.path.abspath(os.path.expanduser(str(manifest_path)))
    if not os.path.isfile(manifest_path):
        raise RasterDataError(
            "cohort manifest not found: %s (pass --manifest; the one of record "
            "is extracted_v2/cohort_manifest.json under Deep_bio)" % manifest_path)
    if require_sidecar and not os.path.isfile(manifest_path + ".sha256"):
        raise RasterDataError(
            "%s has no .sha256 sidecar beside it, so its identity cannot be "
            "checked; use the manifest of record, or a copy made WITH its "
            "sidecar" % manifest_path)
    import cohort_manifest as CM                                  # lazy, see above

    doc = CM.read_manifest(manifest_path)
    prep = doc["preprocessing"]
    classes = tuple(str(c) for c in doc["classes"])
    root = str(doc["extract_root"])
    keys = ("fs_raw", "index_base", "grid_width", "mfr_threshold")

    wells: List[WellRecord] = []
    for w in doc["wells"]:
        rel = str(w["out_dir"]).replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        if len(parts) != 3:
            raise RasterDataError(
                "well %r: out_dir %r is not <class>/<batch>/<well>; this tool "
                "reads the class and batch from that layout" % (w.get("culture"), rel))
        cls, batch, well = parts
        if cls not in classes:
            raise RasterDataError("well %r: class folder %r is not one of the "
                                  "manifest's classes %r" % (w.get("culture"), cls, classes))
        culture = str(w["culture"])
        if culture != "%s__%s" % (batch, well):
            raise RasterDataError(
                "well %r: culture is not <batch>__<well> = %r; the culture "
                "template differs from the one this tool reads"
                % (culture, "%s__%s" % (batch, well)))
        out_dir = os.path.join(root, *parts)
        frag = CM.read_fragment(out_dir)
        meta = frag["meta"]
        for k in keys:
            if not _same_number(meta.get(k), prep.get(k)):
                raise RasterDataError(
                    "well %s: its record has %s = %r, the manifest %r"
                    % (culture, k, meta.get(k), prep.get(k)))
        src = meta.get("source_folder")
        if not src:
            raise RasterDataError("well %s: traces_meta.json names no "
                                  "source_folder" % culture)
        src = os.path.normpath(str(src))
        if os.path.basename(src) != well:
            raise RasterDataError(
                "well %s: source_folder %r does not end in the well's name %r"
                % (culture, src, well))
        n_present = int(meta["n_present"])
        n_samples_raw = int(meta["n_samples_raw"])
        if int(w["n_present"]) != n_present or int(w["n_samples_raw"]) != n_samples_raw:
            raise RasterDataError(
                "well %s: the manifest's n_present/n_samples_raw (%r, %r) differ "
                "from its record's (%d, %d)"
                % (culture, w["n_present"], w["n_samples_raw"], n_present, n_samples_raw))
        if "discarded" not in meta:
            raise RasterDataError("well %s: traces_meta.json records no discarded "
                                  "list, so its active set is unknown" % culture)
        discarded = tuple(sorted(int(e) for e in meta["discarded"]))
        if len(set(discarded)) != len(discarded) or len(discarded) > n_present:
            raise RasterDataError("well %s: its discarded list is malformed"
                                  % culture)
        wells.append(WellRecord(
            culture=culture, class_name=cls, batch=batch, well=well,
            out_dir=out_dir, source_folder=src,
            raw_root=os.path.dirname(src),
            n_present=n_present, n_active=n_present - len(discarded),
            n_samples_raw=n_samples_raw, T_rec=float(meta["T_rec"]),
            discarded=discarded))

    if len({w.culture for w in wells}) != len(wells):
        raise RasterDataError("the manifest names a culture twice")
    return CohortInfo(
        manifest_path=manifest_path, manifest_digest=str(doc["_digest"]),
        classes=classes, fs_raw=float(prep["fs_raw"]),
        index_base=int(prep["index_base"]), grid_width=int(prep["grid_width"]),
        mfr_threshold=float(prep["mfr_threshold"]), wells=tuple(wells))


# --------------------------------------------------------------------------- #
# which wells (D-026)
# --------------------------------------------------------------------------- #
def _batches_in_order(wells_of_class: Sequence[WellRecord]) -> List[str]:
    """Batch folders of one class, ordered by their raw folder path, so the
    rows follow the data tree (DATA_C/Batch3 before DATA_C/Batch4/SubBatch1)."""
    first_root: Dict[str, str] = {}
    for w in wells_of_class:
        first_root.setdefault(w.batch, w.raw_root)
    return sorted(first_root, key=lambda b: (first_root[b], b))


def select_typical_wells(wells: Sequence[WellRecord],
                         classes: Sequence[str]) -> List[WellRecord]:
    """D-026: for each class, for each batch folder, the lower-median well.

    Within a batch folder the wells are sorted by (n_active, culture) and the
    one at index (n - 1) // 2 is taken: the median for odd n, the lower of
    the two middle wells for even n, ties on n_active broken by culture id.
    Output order: classes as given, then batch folders by raw path.
    """
    out: List[WellRecord] = []
    for c in classes:
        in_c = [w for w in wells if w.class_name == c]
        if not in_c:
            raise RasterDataError("class %r has no well in the manifest" % c)
        for b in _batches_in_order(in_c):
            group = sorted((w for w in in_c if w.batch == b),
                           key=lambda w: (w.n_active, w.culture))
            out.append(group[(len(group) - 1) // 2])
    return out


def select_named_wells(wells: Sequence[WellRecord],
                       cultures: Sequence[str]) -> List[WellRecord]:
    """The wells named on the command line, in the order given."""
    by_id = {w.culture: w for w in wells}
    unknown = [c for c in cultures if c not in by_id]
    if unknown:
        raise RasterDataError(
            "not in the manifest: %s. Known cultures: %s"
            % (", ".join(unknown), ", ".join(sorted(by_id))))
    if len(set(cultures)) != len(cultures):
        raise RasterDataError("a culture is named twice in --wells")
    return [by_id[c] for c in cultures]


def group_by_class(items, classes: Sequence[str]) -> Dict[str, list]:
    """{class: [items of that class, in input order]} for every class that
    has at least one item, keyed in the order of `classes`. Works for
    WellRecord, SpikeTable and RasterView (all carry class_name)."""
    out: Dict[str, list] = {}
    for c in classes:
        sel = [x for x in items if x.class_name == c]
        if sel:
            out[c] = sel
    stray = [x for x in items if x.class_name not in classes]
    if stray:
        raise RasterDataError("items of unknown class: %s"
                              % sorted({x.class_name for x in stray}))
    return out


def write_wells_tsv(path: str, selected: Sequence[WellRecord], selected_by: str,
                    cohort: CohortInfo, classes: Sequence[str]) -> str:
    """The selection, one row per well, plus the manifest identity as
    comment lines. plot and view read it to know which caches to show."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["# raster selection written %s by %s"
             % (datetime.now(timezone.utc).isoformat(timespec="seconds"), TOOL_VERSION),
             "# manifest\t%s" % cohort.manifest_path,
             "# manifest_sha256\t%s" % cohort.manifest_digest,
             "# classes\t%s" % ",".join(classes),
             "\t".join(WELLS_TSV_COLUMNS)]
    for w in selected:
        lines.append("\t".join(str(v) for v in (
            w.culture, w.class_name, w.batch, w.well, w.n_present, w.n_active,
            w.n_samples_raw, w.source_folder, selected_by)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


def read_wells_tsv(path: str):
    """-> (rows, header): rows are dicts over WELLS_TSV_COLUMNS, header the
    comment lines as {key: value} (manifest, manifest_sha256, classes)."""
    if not os.path.isfile(path):
        raise RasterDataError(
            "no selection at %s; run  run_raster_plots.py select  first" % path)
    rows, header, cols = [], {}, None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split("\t", 1)
                if len(parts) == 2:
                    header[parts[0].strip()] = parts[1].strip()
                continue
            if cols is None:
                cols = line.split("\t")
                if tuple(cols) != WELLS_TSV_COLUMNS:
                    raise RasterDataError("%s: unexpected columns %r" % (path, cols))
                continue
            vals = line.split("\t")
            if len(vals) != len(cols):
                raise RasterDataError("%s: malformed row %r" % (path, line))
            rows.append(dict(zip(cols, vals)))
    if not rows:
        raise RasterDataError("%s lists no well" % path)
    return rows, header


# --------------------------------------------------------------------------- #
# the spike table (raw -> cache)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpikeTable:
    """Every spike of one well, sorted by time.

    sample[i] is the 0-based raw sample index of spike i and electrode[i]
    its electrode's linear index (the k of ptrain_<k>.mat); sample is
    non-decreasing, ties in ascending electrode. present lists the
    electrodes that have a raster file (ascending), n_spikes their spike
    counts over the whole recording, and active_mask marks those with
    MFR = n_spikes / T_rec >= mfr_threshold, evaluated by the extractor's
    mean_firing_rates when the cache was built.
    """

    culture: str
    class_name: str
    batch: str
    sample: np.ndarray
    electrode: np.ndarray
    present: np.ndarray
    n_spikes: np.ndarray
    active_mask: np.ndarray
    n_samples: int
    fs_raw: float
    index_base: int
    grid_width: int
    mfr_threshold: float
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def T_rec(self) -> float:
        """Recording duration [s] = n_samples / fs_raw, as the extractor
        computes it."""
        return float(self.n_samples) / float(self.fs_raw)

    @property
    def active(self) -> np.ndarray:
        """Linear indices of the active electrodes, ascending."""
        return self.present[self.active_mask]

    @property
    def n_present(self) -> int:
        return int(self.present.size)

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_mask))


def build_spike_table(well: WellRecord, cohort: CohortInfo) -> SpikeTable:
    """Read one well's ptrain_<k>.mat files into a SpikeTable.

    The reading is the extractor's (load_ptrain_folder: scipy.io.loadmat,
    variable "ptrain", binary raster, every file of one length), the grid
    check is validate_grid, the MFR is mean_firing_rates over the whole
    recording -- all with the manifest's fs_raw, index_base, grid_width.
    Imported lazily so that plot/view do not need the extractor.
    """
    from channel_subset_extraction import (load_ptrain_folder,
                                           mean_firing_rates, validate_grid)

    inv = load_ptrain_folder(well.source_folder, fs_raw=cohort.fs_raw,
                             index_base=cohort.index_base)
    validate_grid(inv.indices, width=cohort.grid_width, base=cohort.index_base)
    idx = inv.indices                                    # ascending
    present = np.asarray(idx, dtype=np.int32)
    n_spikes = np.asarray([inv.spikes[k].size for k in idx], dtype=np.int64)
    mfrs = mean_firing_rates(inv, inv.T_rec)
    active_mask = np.asarray([mfrs[k] >= cohort.mfr_threshold for k in idx],
                             dtype=bool)
    total = int(n_spikes.sum())
    if total:
        sample = np.concatenate([inv.spikes[k] for k in idx]).astype(np.int64)
        electrode = np.repeat(present, n_spikes)
        order = np.argsort(sample, kind="stable")        # ties keep electrode order
        sample = sample[order]
        electrode = electrode[order]
    else:
        sample = np.zeros(0, dtype=np.int64)
        electrode = np.zeros(0, dtype=np.int32)
    return SpikeTable(
        culture=well.culture, class_name=well.class_name, batch=well.batch,
        sample=sample, electrode=electrode.astype(np.int32), present=present,
        n_spikes=n_spikes, active_mask=active_mask, n_samples=int(inv.n_samples),
        fs_raw=float(cohort.fs_raw), index_base=int(cohort.index_base),
        grid_width=int(cohort.grid_width),
        mfr_threshold=float(cohort.mfr_threshold),
        meta={"source_folder": well.source_folder,
              "manifest_path": cohort.manifest_path,
              "manifest_digest": cohort.manifest_digest,
              "cache_version": CACHE_VERSION, "tool_version": TOOL_VERSION,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "record_n_present": well.n_present,
              "record_n_active": well.n_active,
              "record_n_samples_raw": well.n_samples_raw})


def check_against_record(table: SpikeTable, well: WellRecord) -> None:
    """Refuse a spike table that is not the data the extraction read.

    Three checks against the well's traces_meta.json: the raster length,
    the number of present electrodes, and the sub-threshold set (present
    and MFR < mfr_threshold) equal to the recorded discarded list, element
    for element. The last one is what makes the raster's rows (D-027) the
    extractor's valid set V.
    """
    problems = []
    if table.n_samples != well.n_samples_raw:
        problems.append("n_samples %d != recorded n_samples_raw %d"
                        % (table.n_samples, well.n_samples_raw))
    if table.n_present != well.n_present:
        problems.append("present electrodes %d != recorded n_present %d"
                        % (table.n_present, well.n_present))
    sub = set(int(e) for e in table.present[~table.active_mask])
    rec = set(well.discarded)
    if sub != rec:
        only_here = sorted(sub - rec)[:10]
        only_rec = sorted(rec - sub)[:10]
        problems.append("sub-threshold set differs from the recorded discarded "
                        "list (%d vs %d; only here %s; only recorded %s)"
                        % (len(sub), len(rec), only_here, only_rec))
    if problems:
        raise RasterDataError("well %s REFUSED -- its raw folder %s is not what "
                              "its extraction recorded: %s"
                              % (well.culture, well.source_folder, "; ".join(problems)))


def cache_path(out_root: str, culture: str) -> str:
    return os.path.join(out_root, "cache", culture + ".npz")


def save_cache(table: SpikeTable, path: str) -> str:
    """Write the cache atomically (temp file, then rename)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, cache_version=np.int64(CACHE_VERSION),
            sample=table.sample, electrode=table.electrode,
            present=table.present, n_spikes=table.n_spikes,
            active_mask=table.active_mask, n_samples=np.int64(table.n_samples),
            fs_raw=np.float64(table.fs_raw), index_base=np.int64(table.index_base),
            grid_width=np.int64(table.grid_width),
            mfr_threshold=np.float64(table.mfr_threshold),
            culture=np.array(table.culture), class_name=np.array(table.class_name),
            batch=np.array(table.batch),
            meta_json=np.array(json.dumps(table.meta, sort_keys=True)))
    os.replace(tmp, path)
    return path


def load_cache(path: str) -> SpikeTable:
    """Read a cache written by save_cache; refuse another version."""
    if not os.path.isfile(path):
        raise RasterDataError("cache not found: %s (run  run_raster_plots.py "
                              "cache  first)" % path)
    with np.load(path, allow_pickle=False) as z:
        ver = int(z["cache_version"])
        if ver != CACHE_VERSION:
            raise RasterDataError("%s is cache version %d, this code reads %d; "
                                  "rebuild it with  cache --force" % (path, ver, CACHE_VERSION))
        return SpikeTable(
            culture=str(z["culture"]), class_name=str(z["class_name"]),
            batch=str(z["batch"]), sample=z["sample"].astype(np.int64),
            electrode=z["electrode"].astype(np.int32),
            present=z["present"].astype(np.int32),
            n_spikes=z["n_spikes"].astype(np.int64),
            active_mask=z["active_mask"].astype(bool),
            n_samples=int(z["n_samples"]), fs_raw=float(z["fs_raw"]),
            index_base=int(z["index_base"]), grid_width=int(z["grid_width"]),
            mfr_threshold=float(z["mfr_threshold"]),
            meta=json.loads(str(z["meta_json"])))


def cache_is_current(path: str, well: WellRecord, cohort: CohortInfo) -> Tuple[bool, str]:
    """(True, "") if the cache at path was built from this well, this
    manifest and this tool version; else (False, the reason)."""
    if not os.path.isfile(path):
        return False, "absent"
    try:
        t = load_cache(path)
    except Exception as exc:                                     # noqa: BLE001
        return False, "unreadable (%s)" % exc
    checks = (
        ("culture", t.culture, well.culture),
        ("source_folder", t.meta.get("source_folder"), well.source_folder),
        ("manifest_digest", t.meta.get("manifest_digest"), cohort.manifest_digest),
        ("fs_raw", t.fs_raw, cohort.fs_raw),
        ("index_base", t.index_base, cohort.index_base),
        ("grid_width", t.grid_width, cohort.grid_width),
        ("mfr_threshold", t.mfr_threshold, cohort.mfr_threshold),
        ("n_samples", t.n_samples, well.n_samples_raw),
    )
    for name, have, want in checks:
        if have != want:
            return False, "%s differs (%r != %r)" % (name, have, want)
    return True, ""


def build_and_save_cache(well: WellRecord, cohort: CohortInfo, out_root: str,
                         force: bool = False) -> Dict[str, object]:
    """Build, check and write one well's cache; the unit of parallel work.

    Returns a summary dict for the log. Raises RasterDataError (refused) or
    the extractor's PtrainLoadError / GeometryError.
    """
    path = cache_path(out_root, well.culture)
    if not force:
        ok, why = cache_is_current(path, well, cohort)
        if ok:
            t = load_cache(path)
            return {"culture": well.culture, "status": "kept", "path": path,
                    "n_present": t.n_present, "n_active": t.n_active,
                    "n_spikes": int(t.sample.size), "seconds": 0.0}
    t0 = time.time()
    table = build_spike_table(well, cohort)
    check_against_record(table, well)
    save_cache(table, path)
    return {"culture": well.culture, "status": "built", "path": path,
            "n_present": table.n_present, "n_active": table.n_active,
            "n_spikes": int(table.sample.size), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# windows and the raster view (D-027, D-028)
# --------------------------------------------------------------------------- #
def window_bounds(t0, t1, T_rec: float, fs_raw: Optional[float] = None) -> Tuple[float, float]:
    """Validate the half-open window [t0, t1) in seconds.

    0 <= t0 < t1 <= T_rec. T_rec = n_samples / fs_raw is a float quotient,
    so t1 may exceed it by up to one sample period (1 / fs_raw) and still be
    accepted as "the end of the recording".
    """
    try:
        a, b = float(t0), float(t1)
    except (TypeError, ValueError):
        raise RasterDataError("window bounds must be numbers, got %r, %r" % (t0, t1))
    if not (np.isfinite(a) and np.isfinite(b)):
        raise RasterDataError("window bounds must be finite, got %r, %r" % (t0, t1))
    slack = (1.0 / float(fs_raw)) if fs_raw else 1e-9
    if a < 0.0 or b <= a or b > float(T_rec) + slack:
        raise RasterDataError(
            "window [%g, %g) s is not inside the recording: need 0 <= t0 < t1 "
            "<= T_rec = %.6g s" % (a, b, T_rec))
    return a, b


def window_slice(table: SpikeTable, t0: float, t1: float) -> slice:
    """Index range of the spikes with t0 * fs_raw <= sample < t1 * fs_raw,
    i.e. spike time s / fs_raw in [t0, t1)."""
    fs = table.fs_raw
    lo = int(np.searchsorted(table.sample, t0 * fs, side="left"))
    hi = int(np.searchsorted(table.sample, t1 * fs, side="left"))
    return slice(lo, hi)


@dataclass(frozen=True)
class RasterView:
    """What one raster panel draws (computed here, drawn by raster_plot).

    t[i] [s] and row[i] are the time and row of spike i of the window; rows
    0..n_rows-1 are the active electrodes in ascending linear index, so row 0
    is the first active electrode of grid row 0. row_ticks are the rows at
    which the grid rows in row_tick_labels begin (only grid rows that hold an
    active electrode get a tick).
    """

    culture: str
    class_name: str
    batch: str
    t0: float
    t1: float
    t: np.ndarray
    row: np.ndarray
    n_rows: int
    n_present: int
    row_ticks: Tuple[int, ...]
    row_tick_labels: Tuple[str, ...]

    @property
    def n_spikes(self) -> int:
        return int(self.t.size)


def raster_view(table: SpikeTable, t0: float, t1: float,
                grid_row_step: int = 12) -> RasterView:
    """The spikes of the active electrodes in [t0, t1), mapped to rows (D-027)."""
    a, b = window_bounds(t0, t1, table.T_rec, table.fs_raw)
    active = table.active                                     # ascending
    n_rows = int(active.size)
    size = int(table.present.max()) + 1 if table.present.size else 1
    row_of = np.full(size, -1, dtype=np.int32)
    row_of[active] = np.arange(n_rows, dtype=np.int32)
    sl = window_slice(table, a, b)
    rows = row_of[table.electrode[sl]]
    keep = rows >= 0
    t = table.sample[sl][keep].astype(np.float64) / table.fs_raw
    rows = rows[keep]

    grid_row = (active.astype(np.int64) - table.index_base) // table.grid_width
    ticks, labels = [], []
    for g in range(0, table.grid_width, max(int(grid_row_step), 1)):
        r = int(np.searchsorted(grid_row, g, side="left"))
        if r < n_rows and (not ticks or r > ticks[-1]):
            ticks.append(r)
            labels.append(str(int(grid_row[r])))
    return RasterView(
        culture=table.culture, class_name=table.class_name, batch=table.batch,
        t0=a, t1=b, t=t, row=rows.astype(np.int32), n_rows=n_rows,
        n_present=table.n_present, row_ticks=tuple(ticks),
        row_tick_labels=tuple(labels))


def fmt_window(t0: float, t1: float) -> str:
    """File-name stamp of a window: t0000.0-0060.0s."""
    return "t%06.1f-%06.1fs" % (t0, t1)


# --------------------------------------------------------------------------- #
# what every figure records about itself
# --------------------------------------------------------------------------- #
def common_preprocessing(tables: Sequence[SpikeTable]) -> Dict[str, object]:
    """fs_raw, index_base, grid_width, mfr_threshold shared by all tables;
    raises if two wells disagree (they would come from different manifests)."""
    if not tables:
        raise RasterDataError("no well to draw")
    keys = ("fs_raw", "index_base", "grid_width", "mfr_threshold")
    ref = {k: getattr(tables[0], k) for k in keys}
    for t in tables[1:]:
        for k in keys:
            if getattr(t, k) != ref[k]:
                raise RasterDataError(
                    "wells %s and %s disagree on %s (%r != %r): their caches "
                    "come from different manifests; rebuild with  cache --force"
                    % (tables[0].culture, t.culture, k, ref[k], getattr(t, k)))
    return ref


def figure_sidecar(tables: Sequence[SpikeTable], views_by_class: Dict[str, list],
                   t0: float, t1: float, rows: Sequence[dict],
                   header: Dict[str, str], how: str) -> Dict[str, object]:
    """The JSON written beside every figure: window, rules, wells, provenance."""
    prep = common_preprocessing(tables)
    by_table = {t.culture: t for t in tables}
    by_row = {r["culture"]: r for r in rows}
    wells = []
    for c, views in views_by_class.items():
        for v in views:
            t = by_table[v.culture]
            r = by_row.get(v.culture, {})
            wells.append({
                "culture": v.culture, "class_name": v.class_name, "batch": v.batch,
                "selected_by": r.get("selected_by"), "n_present": v.n_present,
                "n_active": v.n_rows, "n_spikes_in_window": v.n_spikes,
                "T_rec_s": t.T_rec, "source_folder": t.meta.get("source_folder"),
                "cache_created_utc": t.meta.get("created_utc"),
                "cache_manifest_sha256": t.meta.get("manifest_digest")})
    return {
        "tool_version": TOOL_VERSION,
        "made_by": how,
        "window_s": [float(t0), float(t1)],
        "window_rule": "half-open [t0, t1); spike time = raw sample index / fs_raw (D-028)",
        "rows": ("active electrodes: MFR over the whole recording >= mfr_threshold; "
                 "ascending linear index = row-major order on the extractor's grid "
                 "mapping with index_base; grid row 0 at the top (D-027)"),
        "wells_rule": sorted({w["selected_by"] for w in wells if w["selected_by"]}),
        "decisions": {"wells": "D-026", "rows": "D-027", "window": "D-028",
                      "no_rate_panel": "D-029"},
        "manifest": header.get("manifest"),
        "manifest_sha256": header.get("manifest_sha256"),
        "preprocessing": prep,
        "wells": wells,
    }
