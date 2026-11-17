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

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["CohortMap", "CohortMapError", "read_cohort_map", "REQUIRED_COLUMNS"]

REQUIRED_COLUMNS = ("out_dir", "donor", "rep_group", "include")


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


def read_cohort_map(path, manifest) -> CohortMap:
    """Read and check the map against the manifest (`manifest["wells"]`, each
    with its `out_dir`); see specs/SPEC.md Block 9 for every refusal."""
    raise NotImplementedError("Block 9")
