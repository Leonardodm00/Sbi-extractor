#!/usr/bin/env python3
"""
real_source.py -- the trace source over the EXTRACTED REAL recordings.

This is the piece both handoffs list as "not yet implemented": README.md
Section 10 ("Real-recording export ... needs a trace source over the
extracted .npz archives") and the misspecification handoff Section 7 ("A3
has never been run").

WHAT IT DOES AND, MORE IMPORTANTLY, WHAT IT DOES NOT DO

It does NOT compute an observable. The pooled, smoothed, per-electrode-mean
IFR already exists on disk: run_channel_subset_extraction.py wrote it under
the key 'ifr_trace' at fs_ifr = 1 / w_size, with w_size = 0.02 s,
gaussian_window = 0.04 s and electrodes_per_subset = 9. Those are exactly
the constants build_pooled_ifr uses on the simulated side, so the two arms
are already commensurable and any recomputation here could only break that.
This module therefore LOADS and YIELDS; every array it hands to
export_embeddings is the bytes that came off disk.

Separation of concerns, matching the rest of the package:

    load_specs / validate_cohort   cohort bookkeeping, no I/O of traces
    read_real_trace                one .npz -> one 1-D trace + its meta
    build_real_records             (trace, ident) pairs, numpy only
    iter_trace_records             the same as TraceRecord (imports torch)
    main                           the export run

Everything above iter_trace_records is importable without torch, which is
what lets smoke_test_real_source.py run on a login node in a second.

THREE TRAPS THIS MODULE GUARDS, EACH FOUND IN THE ACTUAL DATA

1. THE GROUP IDENTIFIER IS NOT THE ONE IN THE NPZ.
   run_channel_subset_extraction.py writes culture_id = basename of the
   ptrain folder, e.g. 'ptrain_A1'. That name REPEATS across batches:
   DATA_C_Batch3/ptrain_A1 and Batch4_SubBatch1/ptrain_A1 both carry
   'ptrain_A1'. Grouping on it would silently merge two different cultures
   into one group, which is precisely the dependence structure the
   misspecification gate exists to respect. The authoritative group is the
   'culture' field of the specs file (e.g. 'DATA_C_Batch3__ptrain_A1'),
   which is unique. The npz value is read and cross-checked as a SUFFIX
   only, never used as the group.

2. THE 9 SUBREGIONS OF ONE CULTURE ARE NOT INDEPENDENT.
   They are spatially disjoint 9-electrode patches of ONE culture sharing
   ONE unknown theta_r, exactly as dependent as the windows within a
   subregion. Every row of one culture therefore carries the SAME 'culture'
   value, and the gate must be called with group_col='culture'. Passing
   'name' (which is per subregion) would give R = 9 * n_cultures and rebuild
   the anticonservative error the group-aware null was written to remove.

3. THE REAL SIDECAR MUST STILL DECLARE THE SHARD CONTRACT.
   export_embeddings(label_spec=None) writes no param_names / coord /
   bounds_theta, but npe_contract.Contract.from_dict REQUIRES all three and
   raises KeyError on a label-free sidecar, so load_shard() cannot read the
   real export at all. Passing --sim_sidecar copies that block over from a
   simulated sidecar and, on the way, asserts the DSN checkpoint digests
   agree. That turns assertion A8 (same frozen h_psi in both arms) from an
   eyeball check into a hard one at export time.

ASCII-only by policy (HPC transfer safety).

USAGE

    python3 real_source.py \
        --specs        /path/to/specs_real.json \
        --checkpoint   "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/Main/out/refit_mea_A_best/checkpoints/seed_0/best.pt" \
        --dsn_main_dir "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/Main" \
        --sim_sidecar  /path/to/sbi_campaign_cadex_rho1300v1.json \
        --out          /path/to/export/sbi_real_cohort

Add --max_records 12 for a dry run first. Note the quoting: the DSN path
contains a space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RealSpec",
    "load_specs",
    "validate_cohort",
    "read_real_trace",
    "build_real_records",
    "iter_trace_records",
    "contract_block_from_sidecar",
]

# Keys the extractor may have used for the trace array. 'ifr_trace' is the
# name NumpyTraceProvider reads; 'X' is the extractor's own historical name
# for the SAME array. Preference order is deliberate, not alphabetical.
TRACE_KEYS = ("ifr_trace", "X")

# Fields copied from a simulated sidecar so the real shard is loadable by
# npe_contract. See trap 3 in the module docstring.
CONTRACT_FIELDS = ("param_names", "coord", "bounds_theta", "param_units",
                   "registry")


# --------------------------------------------------------------------------- #
# cohort bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class RealSpec:
    """One extracted subregion trace, as described by the specs file."""
    path: str
    name: str
    condition: int
    culture: str


def load_specs(specs_path: str,
               path_from: Optional[str] = None,
               path_to: Optional[str] = None) -> List[RealSpec]:
    """Read the DSN specs JSON into RealSpec records.

    Parameters
    ----------
    specs_path : str
        The same specs file the DSN training/evaluation used. Every entry
        must carry 'path', 'name', 'condition' and 'culture'.
    path_from, path_to : str or None
        Optional literal prefix substitution applied to every 'path', for
        the case where the extracted archives were moved after the specs
        file was written. Both must be given together or neither.

    Raises
    ------
    KeyError
        If an entry is missing a required field. The index and the offending
        entry are named, because a specs file is usually machine-generated
        and a silent skip would drop a culture without trace.
    """
    if (path_from is None) != (path_to is None):
        raise ValueError("path_from and path_to must be given together")

    with open(specs_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise TypeError("specs file must hold a JSON list; got %s"
                        % type(raw).__name__)
    if not raw:
        raise ValueError("specs file %r is empty" % (specs_path,))

    out: List[RealSpec] = []
    for i, d in enumerate(raw):
        for key in ("path", "name", "condition", "culture"):
            if key not in d:
                raise KeyError("specs entry %d has no %r (entry: %r)"
                               % (i, key, d))
        p = str(d["path"])
        if path_from is not None and p.startswith(path_from):
            p = path_to + p[len(path_from):]
        out.append(RealSpec(path=p, name=str(d["name"]),
                            condition=int(d["condition"]),
                            culture=str(d["culture"])))
    return out


def validate_cohort(specs: Sequence[RealSpec]) -> Dict[str, object]:
    """Refuse a cohort that would break the gate's group structure.

    Two conditions are fatal rather than warned about:

    - a duplicate 'name', because names index rows and a duplicate makes the
      export un-joinable back to its source file;
    - a culture carrying more than one 'condition', because the per-class
      split would then cut a group in half, and every argument in the
      group-aware null assumes a group lies wholly inside one class.
      misspecification_gate() raises on this too; failing here is cheaper.

    Returns a summary dict; raises on either failure.
    """
    seen: Dict[str, int] = {}
    for i, s in enumerate(specs):
        if s.name in seen:
            raise ValueError(
                "duplicate spec name %r at entries %d and %d; names must be "
                "unique because they index the exported rows"
                % (s.name, seen[s.name], i))
        seen[s.name] = i

    by_culture: Dict[str, set] = {}
    order: List[str] = []
    for s in specs:
        if s.culture not in by_culture:
            by_culture[s.culture] = set()
            order.append(s.culture)
        by_culture[s.culture].add(s.condition)
    straddle = [c for c in order if len(by_culture[c]) > 1]
    if straddle:
        raise ValueError(
            "culture(s) %r carry more than one condition. A culture is one "
            "experimental unit and must lie wholly inside one class, or the "
            "per-class verdict breaks the group structure the gate rests on."
            % (straddle[:5],))

    per_class_rows: Dict[int, int] = {}
    per_class_cultures: Dict[int, int] = {}
    for s in specs:
        per_class_rows[s.condition] = per_class_rows.get(s.condition, 0) + 1
    for c in order:
        cond = next(iter(by_culture[c]))
        per_class_cultures[cond] = per_class_cultures.get(cond, 0) + 1

    subs = {}
    for s in specs:
        subs[s.culture] = subs.get(s.culture, 0) + 1
    counts = sorted(set(subs.values()))

    return {
        "n_specs": len(specs),
        "n_cultures": len(order),
        "cultures": order,
        "per_class_specs": per_class_rows,
        "per_class_cultures": per_class_cultures,
        "subregions_per_culture": counts,
    }


# --------------------------------------------------------------------------- #
# one archive -> one trace
# --------------------------------------------------------------------------- #
def read_real_trace(npz_path: str,
                    expect_fs: Optional[float] = None,
                    fs_tol: float = 1e-6) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load one extracted subregion trace and its metadata.

    Parameters
    ----------
    npz_path : str
        A trace_subregion_XX.npz written by run_channel_subset_extraction.py
        in mode 'per_region_single'.
    expect_fs : float or None
        fs_ifr the checkpoint implies, i.e. 1 / cohort.w_size. When given,
        a disagreement RAISES. This is the one parity check that cannot be
        recovered downstream: a trace sampled at a different rate is a
        different observable, and the embedding of it is meaningless while
        looking entirely normal.
    fs_tol : float
        Absolute tolerance on the fs comparison.

    Returns
    -------
    (x, meta)
        x    : (K,) float32, the trace EXACTLY as stored. Not recomputed,
               not rescaled, not smoothed again.
        meta : the npz scalars that matter downstream, as plain Python.
    """
    with np.load(npz_path, allow_pickle=False) as d:
        files = set(d.files)
        key = None
        for cand in TRACE_KEYS:
            if cand in files:
                key = cand
                break
        if key is None:
            raise KeyError(
                "%s holds none of %r (found %r). Was this written by "
                "run_channel_subset_extraction.py?"
                % (npz_path, list(TRACE_KEYS), sorted(files)))

        arr = np.asarray(d[key])
        meta: Dict[str, object] = {"trace_key": key}
        for k in ("fs_ifr", "T_rec", "n_present", "n_samples_raw",
                  "in_channels", "n_samples", "subregion_index", "mode"):
            if k in files:
                v = d[k]
                meta[k] = v.item() if getattr(v, "shape", ()) == () else v
        if "culture_id" in files:
            meta["culture_id"] = str(d["culture_id"])
        if "discarded" in files:
            meta["n_discarded"] = int(np.asarray(d["discarded"]).size)

    # Shape. A (1, K) array is a single-channel trace stored two-dimensionally
    # and is accepted; a genuine (C, K) multichannel array is NOT, because the
    # simulated arm is one pooled channel and a C-channel real arm would be a
    # different observable entirely.
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(
            "%s holds a %r array under %r. This module expects the "
            "'per_region_single' extraction mode, one pooled channel per "
            "file, to match the simulated 3x3 probe. A multichannel export "
            "is a different observable and must not be mixed in."
            % (npz_path, arr.shape, key))

    x = np.ascontiguousarray(arr, dtype=np.float32)
    if x.size == 0:
        raise ValueError("%s holds an empty trace" % (npz_path,))
    if not np.all(np.isfinite(x)):
        raise ValueError("%s holds non-finite samples" % (npz_path,))
    if np.any(x < 0.0):
        raise ValueError(
            "%s holds negative IFR samples (min %.6g). R_norm is a clipped "
            "smoothed count and cannot be negative; this file is corrupt or "
            "was not produced by the Stage 4 IFR routine."
            % (npz_path, float(x.min())))

    if expect_fs is not None:
        if "fs_ifr" not in meta:
            raise KeyError(
                "%s records no fs_ifr, so the sampling rate cannot be checked "
                "against the checkpoint's 1 / cohort.w_size = %.6g Hz. Refusing "
                "rather than assuming." % (npz_path, expect_fs))
        got = float(meta["fs_ifr"])
        if abs(got - float(expect_fs)) > fs_tol:
            raise ValueError(
                "%s was extracted at fs_ifr = %.6g Hz but the checkpoint "
                "implies %.6g Hz. The two arms would be different observables. "
                "Re-extract with --w-size %.6g, or use the checkpoint the "
                "extraction matched."
                % (npz_path, got, float(expect_fs), 1.0 / float(expect_fs)))

    return x, meta


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
def build_real_records(specs: Sequence[RealSpec],
                       expect_fs: Optional[float] = None,
                       strict_culture_id: bool = False
                       ) -> Iterator[Tuple[np.ndarray, Dict[str, object]]]:
    """Yield (trace, ident) for every spec, in specs-file order.

    ident carries, per row:
        culture    the GROUP for the gate (unique across batches)
        condition  the CLASS for the gate (0 control, 1 pathological)
        subregion  which 9-electrode patch of that culture
        name       the specs-file name, one per source file
        T_rec_s    the recording duration the extractor recorded
        fs_ifr     the extraction sampling rate, carried for audit

    strict_culture_id : bool
        The npz's own culture_id is the ptrain folder name and is NOT unique
        across batches (see trap 1). It is checked to be a suffix of the
        specs 'culture'; when True a mismatch raises, otherwise it is
        returned in the ident as culture_id_npz for inspection.
    """
    for s in specs:
        x, meta = read_real_trace(s.path, expect_fs=expect_fs)

        cid = meta.get("culture_id")
        if cid is not None and not str(s.culture).endswith(str(cid)):
            msg = ("specs culture %r does not end with the npz culture_id %r "
                   "for %s. One of the two is describing a different file."
                   % (s.culture, cid, s.path))
            if strict_culture_id:
                raise ValueError(msg)

        sub = meta.get("subregion_index")
        if sub is None:
            base = os.path.basename(s.path)
            digits = "".join(ch for ch in base if ch.isdigit())
            sub = int(digits) if digits else -1

        ident: Dict[str, object] = {
            "culture": s.culture,
            "condition": int(s.condition),
            "subregion": int(sub),
            "name": s.name,
            "T_rec_s": float(meta.get("T_rec", float("nan"))),
            "fs_ifr": float(meta.get("fs_ifr", float("nan"))),
        }
        if cid is not None:
            ident["culture_id_npz"] = str(cid)
        yield x, ident


def iter_trace_records(specs: Sequence[RealSpec],
                       expect_fs: Optional[float] = None,
                       strict_culture_id: bool = False,
                       max_records: Optional[int] = None):
    """build_real_records, wrapped as TraceRecord for export_embeddings.

    theta_A is None for every record: real recordings have no parameters.
    export_embeddings then writes no th_* columns and skips assertions
    A3 / A4 / A5 as inapplicable, which is the documented label-free path.

    TraceRecord is imported lazily because export_embeddings pulls in
    dsn_frozen and therefore torch; the functions above stay torch-free so
    the smoke test can exercise them without a GPU node.
    """
    from export_embeddings import TraceRecord  # noqa: E402

    n = 0
    for x, ident in build_real_records(
            specs, expect_fs=expect_fs,
            strict_culture_id=strict_culture_id):
        yield TraceRecord(trace=x, theta_A=None, ident=ident)
        n += 1
        if max_records is not None and n >= max_records:
            return


# --------------------------------------------------------------------------- #
# the shard contract (trap 3)
# --------------------------------------------------------------------------- #
def contract_block_from_sidecar(sim_sidecar_path: str,
                                expect_digest: Optional[str] = None
                                ) -> Dict[str, object]:
    """Copy the label block out of a simulated sidecar, checking the encoder.

    npe_contract.Contract.from_dict requires param_names, coord and
    bounds_theta. A label-free real sidecar has none of them, so load_shard
    cannot open the real export at all. Copying them across says "this shard
    belongs to the same family", which is exactly what A10 then checks.

    expect_digest : str or None
        The real export's own DSN checkpoint SHA-256. When given, it must
        equal the simulated sidecar's, or this raises. That is assertion A8,
        enforced at export time instead of by eye afterwards.
    """
    with open(sim_sidecar_path, "r", encoding="utf-8") as fh:
        sim = json.load(fh)

    missing = [k for k in ("param_names", "coord", "bounds_theta")
               if k not in sim]
    if missing:
        raise KeyError(
            "simulated sidecar %s has no %r. Point --sim_sidecar at a sidecar "
            "written WITH a label_spec, i.e. one of the campaign exports."
            % (sim_sidecar_path, missing))

    sim_digest = (sim.get("embedding", {}) or {}).get("dsn_checkpoint_sha256")
    if expect_digest is not None:
        if sim_digest is None:
            raise KeyError(
                "simulated sidecar %s records no dsn_checkpoint_sha256, so "
                "assertion A8 cannot be enforced." % (sim_sidecar_path,))
        if str(sim_digest) != str(expect_digest):
            raise ValueError(
                "A8 FAILED: the simulated shard was embedded with checkpoint "
                "%s... and this real export with %s.... Two arms embedded by "
                "different encoders measure the difference between the "
                "encoders and nothing about the simulator. Re-export one of "
                "them." % (str(sim_digest)[:16], str(expect_digest)[:16]))

    block = {k: sim[k] for k in CONTRACT_FIELDS if k in sim}
    block["contract_source_sidecar"] = os.path.abspath(sim_sidecar_path)
    block["contract_note"] = (
        "param_names / coord / bounds_theta are COPIED from the simulated "
        "sidecar named above so that npe_contract.load_shard can open this "
        "file. This shard carries NO th_* columns: it is real data and has "
        "no theta. Load it with require_theta=False.")
    return block


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _banner(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export DSN embeddings for the extracted REAL recordings.")
    ap.add_argument("--specs", required=True,
                    help="the DSN specs JSON (path/name/condition/culture)")
    ap.add_argument("--checkpoint", required=True,
                    help="the frozen DSN .pt; must be the SAME file the "
                         "simulated shards used")
    ap.add_argument("--out", required=True,
                    help="output stem, no extension")
    ap.add_argument("--dsn_main_dir",
                    default=os.environ.get("DSN_MAIN_DIR"),
                    help="<Deep-Summary-Network>/Main")
    ap.add_argument("--sim_sidecar", default=None,
                    help="a simulated sidecar .json to copy the shard "
                         "contract from; strongly recommended (see trap 3)")
    ap.add_argument("--expect_config_json", default=None,
                    help="training config JSON to cross-check the checkpoint "
                         "geometry against")
    ap.add_argument("--path_from", default=None,
                    help="literal prefix to replace in every specs path")
    ap.add_argument("--path_to", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_records", type=int, default=None,
                    help="stop after N traces; use for a dry run")
    ap.add_argument("--strict_culture_id", action="store_true",
                    help="raise instead of noting when the npz culture_id is "
                         "not a suffix of the specs culture")
    ap.add_argument("--no_zraw", action="store_true",
                    help="skip the pre-normalisation activations. Do NOT use "
                         "for the misspecification gate: an amplitude "
                         "mismatch is invisible in the L2-normalised z.")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dsn_frozen import load_frozen_dsn          # noqa: E402
    from export_embeddings import export_embeddings  # noqa: E402

    # ---- 1. cohort --------------------------------------------------------
    specs = load_specs(args.specs, path_from=args.path_from,
                       path_to=args.path_to)
    summary = validate_cohort(specs)

    _banner("COHORT")
    print("specs file            : %s" % os.path.abspath(args.specs))
    print("subregion traces      : %d" % summary["n_specs"])
    print("cultures (the GROUPS) : %d" % summary["n_cultures"])
    print("subregions per culture: %s" % (summary["subregions_per_culture"],))
    print("specs per class       : %s" % (summary["per_class_specs"],))
    print("cultures per class    : %s" % (summary["per_class_cultures"],))

    missing = [s.path for s in specs if not os.path.isfile(s.path)]
    if missing:
        print("")
        print("ERROR: %d specs path(s) do not exist, first few:" % len(missing))
        for p in missing[:5]:
            print("   %s" % p)
        return 3

    # ---- 2. the frozen encoder -------------------------------------------
    dsn = load_frozen_dsn(args.checkpoint, device=args.device,
                          dsn_main_dir=args.dsn_main_dir,
                          expect_config_json=args.expect_config_json)
    fs_ifr = 1.0 / float(dsn.w_size)

    _banner("ENCODER")
    print("checkpoint : %s" % os.path.abspath(args.checkpoint))
    print("sha256     : %s" % dsn.ckpt_sha256)
    print("E          : %d" % dsn.embedding_dim)
    print("in_channels: %d" % dsn.in_channels)
    print("l2_normalize: %s" % dsn.l2_normalize)
    print("window_s   : %.4f s   -> W = %d samples"
          % (dsn.window_s, dsn.window_length))
    print("w_size     : %.4f s   -> fs_ifr = %.4f Hz" % (dsn.w_size, fs_ifr))
    print("sigma_sm   : %.4f s" % dsn.gaussian_window)
    for w in dsn.warnings:
        print("WARN: %s" % w)

    if int(dsn.in_channels) != 1:
        print("")
        print("ERROR: the checkpoint expects %d input channels, but the "
              "extracted real traces are single-channel "
              "('per_region_single'). Refusing." % (dsn.in_channels,))
        return 4

    # ---- 3. expected geometry, computed BEFORE the run -------------------
    # Stated up front so the printed result can be compared against it. A
    # trace shorter than W yields ZERO windows and MEAWindowDataset drops it
    # with a silent `continue`; the arithmetic here is what makes that
    # visible instead.
    W = int(dsn.window_length)
    expected_rows = 0
    per_trace: Dict[int, int] = {}
    for s in specs if args.max_records is None else specs[:args.max_records]:
        with np.load(s.path, allow_pickle=False) as d:
            key = "ifr_trace" if "ifr_trace" in d.files else "X"
            L = int(np.asarray(d[key]).reshape(-1).shape[0])
        n_win = 0 if L < W else (L - W) // W + 1
        per_trace[n_win] = per_trace.get(n_win, 0) + 1
        expected_rows += n_win

    _banner("EXPECTED GEOMETRY")
    print("windows per trace : %s   (count of traces per value)"
          % (per_trace,))
    print("expected rows     : %d" % expected_rows)
    if 0 in per_trace:
        print("")
        print("ERROR: %d trace(s) are shorter than W = %d samples "
              "(%.1f s) and would be SILENTLY DROPPED. Refusing."
              % (per_trace[0], W, dsn.window_s))
        return 5

    # ---- 4. export --------------------------------------------------------
    extra = {
        "cohort": {
            "source": "real recordings, extracted subregion IFR",
            "specs_file": os.path.abspath(args.specs),
            "n_cultures": summary["n_cultures"],
            "n_subregion_traces": summary["n_specs"],
            "per_class_cultures": {str(k): v for k, v in
                                   summary["per_class_cultures"].items()},
            "group_column": "culture",
            "class_column": "condition",
            "group_note": (
                "The 9 subregions of one culture share one culture and one "
                "unknown theta_r and are NOT independent. Group on "
                "'culture'; grouping on 'name' or on the npz culture_id is "
                "invalid."),
        },
        "observable": {
            "n_electrodes": 9,
            "electrode_source": "electrodes_per_subset = 9, greedy disjoint "
                                "partition of the 48x48 array",
            "ifr_recomputed_here": False,
            "ifr_note": "loaded verbatim from the extractor's 'ifr_trace'",
        },
    }
    if args.sim_sidecar is not None:
        extra.update(contract_block_from_sidecar(
            args.sim_sidecar, expect_digest=dsn.ckpt_sha256))
        print("")
        print("contract copied from : %s" % os.path.abspath(args.sim_sidecar))
        print("A8 (same encoder both arms): PASS")

    records = iter_trace_records(specs, expect_fs=fs_ifr,
                                 strict_culture_id=args.strict_culture_id,
                                 max_records=args.max_records)

    result = export_embeddings(
        dsn, records, args.out,
        label_spec=None,
        ident_columns=("culture", "condition", "subregion", "name",
                       "T_rec_s", "fs_ifr"),
        extra_sidecar=extra,
        batch_size=args.batch_size,
        want_zraw=(not args.no_zraw))

    _banner("RESULT")
    print("rows written  : %d   (expected %d)"
          % (result.n_rows, expected_rows))
    print("parquet       : %s.parquet" % args.out)
    print("sidecar       : %s.json" % args.out)
    for w in getattr(result, "warnings", []) or []:
        print("WARN: %s" % w)

    if int(result.n_rows) != int(expected_rows):
        print("")
        print("ERROR: row count disagrees with the arithmetic above. Do not "
              "use this shard until the difference is explained.")
        return 6

    print("")
    print("NEXT: load with npe_contract.load_shard(path, require_theta=False),")
    print("      then groups_from_table(df, group_col='culture', "
          "class_col='condition').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
