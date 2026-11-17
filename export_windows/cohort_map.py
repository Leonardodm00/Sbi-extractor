"""Block 9 -- the real bank's cohort map (specs/SPEC.md, Block 9; SBI
decision D-148 (4), with D-144..D-146 for LindDUP).

A cohort map is a CSV file with one row per well of the cohort manifest:

    out_dir,donor,rep_group,include[,any other column]

`out_dir` is the manifest's well folder (relative to its extract_root, "/"
separators: `<class>/<root_name>/<well>`); `donor` and `rep_group` are
labels (non-empty strings); `include` is yes or no. bank_real.py reads it
through the profile's `cohort_map`: the donor and the replicate pairing
group of each row come from the map, and a well whose `include` is no
contributes no row.

    read_cohort_map(path, manifest) -> CohortMap      (CohortMapError)

The rules (SPEC Block 9). The file is ASCII and is read by the csv module
(csv.reader, the excel dialect, strict) under its header row, each row
mapped to the header's names as csv.DictReader maps it; the four required
columns may come in any order, and any other column is ignored -- so are
the fields of a row beyond the header's last column, which an unquoted
comma in a free-text column makes (DictReader's restkey), while a row
shorter than the header reads its missing columns as empty; blank lines are
skipped. A row is matched on the manifest's `out_dir` with os.sep replaced
by "/"; its
`donor` and `rep_group` are stripped of surrounding blanks and must not be
empty (for an excluded well too); `include` is one of yes, no, true, false,
1, 0, case-insensitive. Refused (CohortMapError, naming the file, the line
and the reason): a required column missing (or named twice in the header);
a row whose out_dir names no well; a well with no row, or with two rows; an
empty label; an include value outside the set; every well excluded; a
non-ASCII file; a line the csv module cannot parse; a missing file; a
manifest without wells, with a well without out_dir, or listing one out_dir
twice.

The ids: donor ids 0..D-1 in the sorted order of the distinct donor labels
of the included wells (Python's str order, which is byte order on ASCII),
rep_group ids 0..G-1 likewise; labels of excluded wells take no id. The
digest is the sha256 of the file's bytes, read once with the text parsed.

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = ["CohortMap", "CohortMapError", "read_cohort_map", "REQUIRED_COLUMNS"]

REQUIRED_COLUMNS = ("out_dir", "donor", "rep_group", "include")
INCLUDE_VALUES = {"yes": True, "true": True, "1": True,
                  "no": False, "false": False, "0": False}      # case-insensitive


class CohortMapError(ValueError):
    """A cohort map the real bank cannot be built on; the message names the
    file, the row and the reason."""


@dataclass(frozen=True)
class CohortMap:
    path: str                             # the file read
    sha256: str                           # its digest (64 lowercase hex)
    donor_labels: Tuple[str, ...]         # id -> label: sorted distinct labels of the included wells
    rep_group_labels: Tuple[str, ...]     # id -> label, likewise
    donor_of: Dict[int, int]              # included well's manifest index -> donor id
    rep_group_of: Dict[int, int]          # included well's manifest index -> rep_group id
    excluded: Tuple[str, ...]             # out_dir of every excluded well, in manifest order


def _read_ascii(path) -> Tuple[str, str]:
    """(text, sha256 of the bytes): one read, so the digest is the text's."""
    if not os.path.isfile(path):
        raise CohortMapError("%s: no such file" % path)
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        text = blob.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CohortMapError("%s: not an ASCII file (byte 0x%02x on line %d)"
                             % (path, blob[exc.start], blob[:exc.start].count(b"\n") + 1)) from exc
    return text, hashlib.sha256(blob).hexdigest()


def _well_index(manifest, path) -> Dict[str, int]:
    """{the manifest's out_dir with os.sep replaced by "/": its index}."""
    wells = manifest.get("wells") if isinstance(manifest, dict) else None
    if not wells:
        raise CohortMapError("%s: the manifest lists no well to map" % path)
    index = {}
    for wi, w in enumerate(wells):
        if not isinstance(w, dict) or "out_dir" not in w:
            raise CohortMapError("%s: the manifest's well %d has no out_dir" % (path, wi))
        key = str(w["out_dir"]).replace(os.sep, "/")
        if key in index:
            raise CohortMapError("%s: the manifest lists out_dir %r twice (wells %d and %d)"
                                 % (path, key, index[key], wi))
        index[key] = wi
    return index


def _rows(text, path) -> List[Tuple[int, Dict[str, str]]]:
    """[(line number, {column: value})] under the header row, as csv.DictReader
    maps them (blank lines skipped; the fields beyond the header's last column
    dropped; a row shorter than the header lacks its last columns), read with
    csv.reader for its line_num, which also numbers a line it cannot parse."""
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise CohortMapError("%s: required column(s) %s missing; the header is %r"
                                 % (path, ", ".join(missing), header))
        twice = [c for c in REQUIRED_COLUMNS if header.count(c) > 1]
        if twice:
            raise CohortMapError("%s: the header names column(s) %s twice"
                                 % (path, ", ".join(twice)))
        # zip stops at the shorter: the fields beyond the header are dropped and
        # a short row lacks its last columns -- DictReader's mapping, on purpose
        return [(reader.line_num, dict(zip(header, fields))) for fields in reader if fields]
    except csv.Error as exc:
        raise CohortMapError("%s: line %d: %s" % (path, reader.line_num, exc)) from exc


def read_cohort_map(path, manifest) -> CohortMap:
    """Read and check the map against the manifest (`manifest["wells"]`, each
    with its `out_dir`); see specs/SPEC.md Block 9 for every refusal."""
    text, sha256 = _read_ascii(path)
    index = _well_index(manifest, path)
    line_of, entry = {}, {}                  # manifest index -> line; -> (donor, rep_group, include)
    for line, row in _rows(text, path):
        where = "%s line %d" % (path, line)
        out_dir = row.get("out_dir", "")    # absent on a row shorter than the header
        if out_dir not in index:
            raise CohortMapError("%s: out_dir %r names no well of the manifest" % (where, out_dir))
        wi = index[out_dir]
        if wi in line_of:
            raise CohortMapError("%s: a second row for the well %r (the first on line %d)"
                                 % (where, out_dir, line_of[wi]))
        line_of[wi] = line
        donor, group = (row.get(c, "").strip() for c in ("donor", "rep_group"))
        for column, label in (("donor", donor), ("rep_group", group)):
            if not label:
                raise CohortMapError("%s: empty %s label" % (where, column))
        include = INCLUDE_VALUES.get(row.get("include", "").lower())
        if include is None:
            raise CohortMapError("%s: include %r is not one of yes, no, true, false, 1, 0"
                                 % (where, row.get("include", "")))
        entry[wi] = (donor, group, include)
    in_order = sorted(index, key=index.get)  # out_dir in manifest order
    no_row = [d for d in in_order if index[d] not in entry]
    if no_row:
        raise CohortMapError("%s: %d well(s) of the manifest have no row: %s%s"
                             % (path, len(no_row), ", ".join(no_row[:5]),
                                " ..." if len(no_row) > 5 else ""))
    included = [wi for wi in sorted(entry) if entry[wi][2]]
    if not included:
        raise CohortMapError("%s: every well is excluded" % path)
    donor_labels = tuple(sorted({entry[wi][0] for wi in included}))
    group_labels = tuple(sorted({entry[wi][1] for wi in included}))
    donor_id = {label: i for i, label in enumerate(donor_labels)}
    group_id = {label: i for i, label in enumerate(group_labels)}
    return CohortMap(path=os.path.abspath(path), sha256=sha256,
                     donor_labels=donor_labels, rep_group_labels=group_labels,
                     donor_of={wi: donor_id[entry[wi][0]] for wi in included},
                     rep_group_of={wi: group_id[entry[wi][1]] for wi in included},
                     excluded=tuple(d for d in in_order if not entry[index[d]][2]))
