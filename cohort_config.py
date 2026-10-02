"""
cohort_config.py -- read the real-MEA cohort block WITHOUT importing torch.

Stage D (2026-09-21). Until now list_extraction_jobs.py did
`from config import ExperimentConfig`, and config.py's first imports are
backbone.py and augmentation.py, so listing 35 wells required the training
environment. The cohort dataclass now lives in the DSN tree's torch-free
`cohort.py` (Simulation-Based-Inference/hpc/dsn/cohort.py); this module is
the extractor-side loader for it, and the ONLY place in Sbi-extractor that
knows the JSON layout `{"cohort": {...}}`.

Provides
--------
load_cohort(config_json, extract_root=None) -> CohortConfig
    Reads ONLY the "cohort" object of the JSON, constructs the very same
    CohortConfig class the DSN uses (one definition, imported), and runs its
    validation. `extract_root` overrides the JSON's value -- Stage D writes
    to extracted_v2/ while the config keeps pointing at extracted/ until the
    new archives exist (one variable at a time).

build_extra_flags(cohort) -> str
    The EXTRA_FLAGS line for run_extractor_array_mea.pbs, byte for byte what
    list_extraction_jobs.py used to write inline. Factored out so a smoke
    test can compare it against the tracked extraction_flags.sh without
    touching the raw data.

PREPROCESSING_FIELDS
    The cohort fields that describe HOW a trace was made, in one fixed order.
    The per-archive fragment, the cohort manifest and the manifest assertion
    all iterate this tuple, so they cannot disagree about which numbers count.

Re-exports root_name_for, find_wells, classify_output_dir, expand,
SUBREGION_PREFIX, MULTICHANNEL_NAME from cohort.py, so callers here need one
import.

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import dsn_tree                                                   # noqa: E402
dsn_tree.add_dsn_to_path()          # raises DSNTreeMissing, naming the fix

from cohort import (CohortConfig, SUBREGION_PREFIX, MULTICHANNEL_NAME,  # noqa: E402
                    PTRAIN_FORMATS, DEFAULT_PTRAIN_FORMAT,
                    DEFAULT_PTRAIN_VARNAME, DEFAULT_PTRAIN_NAME_PATTERN,
                    validate_ptrain_fields,
                    root_name_for, find_wells, classify_output_dir, expand)

__all__ = [
    "CohortConfig", "SUBREGION_PREFIX", "MULTICHANNEL_NAME",
    "PTRAIN_FORMATS", "DEFAULT_PTRAIN_FORMAT", "DEFAULT_PTRAIN_VARNAME",
    "DEFAULT_PTRAIN_NAME_PATTERN", "validate_ptrain_fields",
    "root_name_for", "find_wells", "classify_output_dir", "expand",
    "PREPROCESSING_FIELDS", "PTRAIN_FIELDS", "load_cohort",
    "build_extra_flags", "preprocessing_dict", "ptrain_dict",
    "ptrain_is_default",
]

# The fields of CohortConfig that describe HOW a trace was computed -- the
# object pi of the Stage C/D design. Order is the order they are written
# everywhere. `n_subsets` is here because it is a property of the REAL arm's
# extraction (C subregions per culture) that the manifest RECORDS; the sim
# arm is never gated on it (decision 2026-09-21), nor on mfr_threshold.
PREPROCESSING_FIELDS = (
    "fs_raw",
    "index_base",
    "grid_width",
    "n_subsets",
    "electrodes_per_subset",
    "mfr_threshold",
    "w_size",
    "gaussian_window",
)

# [2026-10-01, Giulia cohort] HOW the per-electrode files are read: the
# format, the variable, the filename pattern. Recorded in every fragment
# (extractor version 4) and in the cohort manifest's "source_format" block;
# NOT part of PREPROCESSING_FIELDS, whose gating role (D-002) is unchanged --
# the format decides how the spike sample indices are obtained from a file,
# and once they are, the trace is computed exactly as before.
PTRAIN_FIELDS = (
    "ptrain_format",
    "ptrain_varname",
    "ptrain_name_pattern",
)
_PTRAIN_DEFAULTS = {
    "ptrain_format": DEFAULT_PTRAIN_FORMAT,
    "ptrain_varname": DEFAULT_PTRAIN_VARNAME,
    "ptrain_name_pattern": DEFAULT_PTRAIN_NAME_PATTERN,
}


def load_cohort(config_json, extract_root=None):
    """The cohort block of a DSN JSON config as a validated CohortConfig.

    Parameters
    ----------
    config_json : str
        Path to e.g. $SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json
    extract_root : str or None
        Overrides cohort.extract_root. Stage D passes the extracted_v2 path
        here so the config can keep pointing at the archives that exist.

    Raises
    ------
    FileNotFoundError, ValueError (from CohortConfig.__post_init__), KeyError
    if the JSON has no "cohort" object.
    """
    with open(config_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "cohort" not in data or not isinstance(data["cohort"], dict):
        raise KeyError("%s has no \"cohort\" object" % (config_json,))
    block = dict(data["cohort"])
    if extract_root is not None:
        block["extract_root"] = str(extract_root)
    # CohortConfig is a plain dataclass; unknown keys must not be silently
    # dropped, they are a sign the config is from another vintage.
    known = set(CohortConfig.__dataclass_fields__.keys())
    unknown = sorted(k for k in block if k not in known)
    if unknown:
        raise ValueError("cohort block has unknown field(s) %r; known: %r"
                         % (unknown, sorted(known)))
    return CohortConfig(**block)


def preprocessing_dict(cohort):
    """{field: value} over PREPROCESSING_FIELDS, with the JSON-native types
    the fragment and the manifest store (int for counts, float otherwise)."""
    out = {}
    for k in PREPROCESSING_FIELDS:
        v = getattr(cohort, k)
        out[k] = int(v) if k in ("index_base", "grid_width", "n_subsets",
                                 "electrodes_per_subset") else float(v)
    return out


def ptrain_dict(cohort):
    """{field: value} over PTRAIN_FIELDS, as plain str."""
    return {k: str(getattr(cohort, k)) for k in PTRAIN_FIELDS}


def ptrain_is_default(cohort):
    """True when the cohort reads the one pre-2026-10-01 format (DUP15HD)."""
    return ptrain_dict(cohort) == _PTRAIN_DEFAULTS


def build_extra_flags(cohort):
    """The exact EXTRA_FLAGS text list_extraction_jobs.py writes.

    Byte for byte the string the pre-Stage-D script wrote inline; a smoke test
    compares this against the tracked extraction_flags.sh.

    [2026-10-01] A cohort whose three PTRAIN_FIELDS are all at their defaults
    gets the pre-2026-10-01 line unchanged, so the DUP15HD extraction_flags.sh
    of record stays byte-identical. Any other cohort gets all three appended
    explicitly (--ptrain-format, --ptrain-varname, --ptrain-name-pattern),
    the pattern having been checked by validate_ptrain_fields to be safe as
    an unquoted shell word.
    """
    line = (
        "--fs-raw %.10g --base %d --grid-width %d "
        "--n-subsets %d --electrodes-per-subset %d "
        "--mfr-threshold %.10g --w-size %.10g --gaussian-window %.10g"
        % (float(cohort.fs_raw), int(cohort.index_base), int(cohort.grid_width),
           int(cohort.n_subsets), int(cohort.electrodes_per_subset),
           float(cohort.mfr_threshold), float(cohort.w_size),
           float(cohort.gaussian_window))
    )
    if not ptrain_is_default(cohort):
        validate_ptrain_fields(cohort.ptrain_format, cohort.ptrain_varname,
                               cohort.ptrain_name_pattern)
        line += (" --ptrain-format %s --ptrain-varname %s --ptrain-name-pattern %s"
                 % (cohort.ptrain_format, cohort.ptrain_varname,
                    cohort.ptrain_name_pattern))
    return (
        "# generated by list_extraction_jobs.py -- do not edit by hand;\n"
        "# re-run the generator if cohort.* changes in the config.\n"
        "EXTRA_FLAGS=\"%s\"\n" % line
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the cohort block of a config, torch-free")
    ap.add_argument("config")
    ap.add_argument("--extract-root", default=None)
    a = ap.parse_args()
    c = load_cohort(a.config, extract_root=a.extract_root)
    print("classes      : %d %r" % (c.n_classes(), [c.name_of_class(i) for i in range(c.n_classes())]))
    print("extract_root : %s" % c.extract_root)
    for k, v in preprocessing_dict(c).items():
        print("%-22s %r" % (k, v))
    for k, v in ptrain_dict(c).items():
        print("%-22s %r" % (k, v))
    print("exclude_wells          %r" % (list(c.exclude_wells),))
    print("fs_ifr %.10g Hz   sigma_bins %.10g" % (c.fs_ifr(), c.sigma_bins()))
    sys.stdout.write(build_extra_flags(c))
