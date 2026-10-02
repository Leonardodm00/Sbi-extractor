"""CLI driver -- extract channel-subset IFR traces from a ptrain folder and save.

Runs the full extractor (load -> geometry/MFR -> partition -> IFR) on ONE folder
of ptrain_<idx>.mat files and writes a self-describing .npz plus (for the
partition modes) the electrode-map and per-subregion IFR PNGs.

Usage
-----
    python3 run_channel_subset_extraction.py FOLDER --out-dir OUT \
        [--mode multichannel|per_region_single|whole_culture] \
        [--n-subsets 9] [--electrodes-per-subset 9] [--mfr-threshold 0.1] \
        [--fs-raw 10110.09] [--base 0] [--no-plots] \
        [--ptrain-format raster|sparse_peaks] [--ptrain-varname ptrain] \
        [--ptrain-name-pattern REGEX]

Output (in OUT)
---------------
    traces.npz : arrays
        X            : (rows, K) float32 IFR traces
        row_meaning  : "channels" (multichannel: rows are channels of ONE sample)
                       or "samples" (per_region_single / whole_culture: rows are
                       independent single-channel samples)
        in_channels  : C for multichannel, else 1
        n_samples    : number of training samples this recording yields
        fs_ifr, mode, index_base, grid_width, T_rec, n_samples_raw
        centers, center_mfr, discarded  (empty for whole_culture)
    subregion_map.png, subregion_ifrs.png  (unless --no-plots / whole_culture)

Only numpy / scipy / matplotlib are required (no torch): this is data extraction,
not training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

from channel_subset_extraction import (DEFAULT_FS_RAW, DEFAULT_PTRAIN_FORMAT,
                                       DEFAULT_PTRAIN_NAME_PATTERN,
                                       PTRAIN_FORMATS, PTRAIN_VARNAME,
                                       compile_name_pattern,
                                       extract_channel_subsets)

EXTRACTOR_VERSION = "run_channel_subset_extraction/4"   # 1 = pre-2026-09-11, no metadata; 2 = metadata; 3 = + extractor_commit, manifest_version (Stage D); 4 = + ptrain_format / ptrain_varname / ptrain_name_pattern (2026-10-01, Giulia cohort)
MANIFEST_VERSION = 1      # schema version of the per-archive fragment AND of cohort_manifest.json


def _git_commit_of(path):
    """Short git HEAD of the repository containing `path`, or "unknown".

    Recorded in every fragment so the cohort manifest can name the extractor
    commit that produced the archives. Never raises: a checkout without git
    on PATH still extracts, it just records "unknown".
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(path)),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                    # noqa: BLE001
        return "unknown"


def extraction_metadata(args, fs_ifr, argv=None):
    """Every parameter that decides what the trace IS, as one dict.

    [CORRECTION 2026-09-11] Until this version the archives recorded fs_ifr
    and T_rec but NOT gaussian_window (sigma_sm), electrodes_per_subset (the
    n_e each pooled trace is a mean over), n_subsets, mfr_threshold or
    fs_raw. The consumer (Sbi-extractor) stamps its sidecars with the encoder
    checkpoint's sigma_sm instead, so an archive smoothed at one value and an
    export declaring another could never be told apart from the files. This
    dict is merged into `meta`, so it lands in traces.npz AND in every
    trace_subregion_XX.npz (both spread **meta), and is also written to
    traces_meta.json beside them.

    Pure: no I/O, so it is testable without a ptrain folder.
    """
    w = float(args.w_size)
    g = float(args.gaussian_window)
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "w_size": w,                        # Delta_t [s]
        "gaussian_window": g,               # sigma_sm [s]
        "sigma_sm_bins": g / w,             # sigma_sm / Delta_t
        "fs_raw": float(args.fs_raw),
        "n_subsets": int(args.n_subsets),
        "electrodes_per_subset": int(args.electrodes_per_subset),
        "mfr_threshold": float(args.mfr_threshold),
        "source_folder": os.path.abspath(str(args.folder)),
        "argv": " ".join(argv if argv is not None else sys.argv),
        # [2026-10-01] how the files were READ (version 4). Not gating: the
        # cohort manifest records them as "source_format" (D-002 unchanged).
        "ptrain_format": str(args.ptrain_format),
        "ptrain_varname": str(args.ptrain_varname),
        "ptrain_name_pattern": str(args.ptrain_name_pattern),
        # Stage D: the fragment names the code that wrote it, and its schema.
        "extractor_commit": _git_commit_of(__file__),
        "manifest_version": MANIFEST_VERSION,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract channel-subset IFR traces from a ptrain folder.")
    p.add_argument("folder", help="directory of ptrain_<idx>.mat files (one recording)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--mode", default="multichannel",
                   choices=["multichannel", "per_region_single", "whole_culture"])
    p.add_argument("--n-subsets", type=int, default=9)
    p.add_argument("--electrodes-per-subset", type=int, default=9)
    p.add_argument("--mfr-threshold", type=float, default=0.1)
    p.add_argument("--fs-raw", type=float, default=DEFAULT_FS_RAW)
    p.add_argument("--base", type=int, default=0, choices=[0, 1])
    p.add_argument("--grid-width", type=int, default=48)
    p.add_argument("--w-size", type=float, default=0.02)
    p.add_argument("--gaussian-window", type=float, default=0.04)
    p.add_argument("--no-plots", action="store_true", help="skip PNG rendering")
    # [2026-10-01] how the per-electrode files are named and stored (the
    # cohort block's ptrain_* fields; defaults = the pre-2026-10-01 behaviour)
    p.add_argument("--ptrain-format", default=DEFAULT_PTRAIN_FORMAT,
                   choices=list(PTRAIN_FORMATS),
                   help="storage of the train: dense binary raster, or "
                        "scipy.sparse with every stored nonzero a spike")
    p.add_argument("--ptrain-varname", default=PTRAIN_VARNAME,
                   help="MATLAB variable holding the train")
    p.add_argument("--ptrain-name-pattern", default=DEFAULT_PTRAIN_NAME_PATTERN,
                   help="regex on the file basename with ONE capture group, "
                        "the electrode's integer index")
    args = p.parse_args(argv)
    compile_name_pattern(args.ptrain_name_pattern)     # refuse a bad pattern before any I/O

    traces, fs_ifr, diag = extract_channel_subsets(
        args.folder, mode=args.mode, n_subsets=args.n_subsets,
        electrodes_per_subset=args.electrodes_per_subset,
        mfr_threshold=args.mfr_threshold, fs_raw=args.fs_raw, index_base=args.base,
        grid_width=args.grid_width, w_size=args.w_size,
        gaussian_window=args.gaussian_window, return_diagnostics=True,
        ptrain_name_pattern=args.ptrain_name_pattern,
        ptrain_varname=args.ptrain_varname, ptrain_format=args.ptrain_format)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "multichannel":
        X = np.asarray(traces[0], dtype=np.float32)          # (C, K), rows = channels
        row_meaning = "channels"
        in_channels = int(X.shape[0])
        n_samples = 1
    else:
        X = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in traces],
                     axis=0)                                  # (n_samples, K), rows = samples
        row_meaning = "samples"
        in_channels = 1
        n_samples = int(X.shape[0])

    centers = np.array([s.center for s in diag.subregions], dtype=np.int64)
    center_mfr = np.array([s.center_mfr for s in diag.subregions], dtype=np.float64)
    discarded = np.array(diag.discarded, dtype=np.int64)

    pre = extraction_metadata(args, fs_ifr)
    if abs(float(fs_ifr) * pre["w_size"] - 1.0) > 1e-6:
        raise RuntimeError(
            "fs_ifr = %r from the extractor is not 1 / w_size = 1 / %r; refusing "
            "to write archives whose declared bin width disagrees with their "
            "sampling rate" % (float(fs_ifr), pre["w_size"]))
    meta = dict(
        fs_ifr=float(fs_ifr), mode=args.mode, index_base=int(diag.index_base),
        grid_width=int(diag.grid_width), T_rec=float(diag.T_rec),
        **pre,
        n_samples_raw=int(diag.n_samples), n_present=int(diag.n_present),
        centers=centers, center_mfr=center_mfr, discarded=discarded)

    # The pipeline's NumpyTraceProvider reads the array under the key
    # "ifr_trace"; "X" is kept as the extractor's own historical name so the
    # viz code and the stage smoke tests keep working. Both name the SAME array.
    npz_path = os.path.join(args.out_dir, "traces.npz")
    np.savez_compressed(
        npz_path, X=X, ifr_trace=X, row_meaning=row_meaning,
        in_channels=in_channels, n_samples=n_samples, **meta)
    written = [npz_path]
    meta_path = os.path.join(args.out_dir, "traces_meta.json")
    with open(meta_path, "w") as fh:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in meta.items()}, fh, indent=2, sort_keys=True)
    written.append(meta_path)

    # mode == "per_region_single": the C rows are INDEPENDENT single-channel
    # samples, not channels of one sample. Each becomes its own trace record,
    # so one .npz per subregion is written alongside the combined file. Every
    # sibling carries the SAME culture_id (the recording), which is what keeps
    # them out of each other's positive pairs downstream.
    if args.mode == "per_region_single":
        culture_id = os.path.basename(os.path.normpath(args.folder))
        for r in range(int(X.shape[0])):
            row = np.ascontiguousarray(X[r], dtype=np.float32)      # (K,)
            rp = os.path.join(args.out_dir, "trace_subregion_%02d.npz" % r)
            np.savez_compressed(
                rp, X=row, ifr_trace=row, row_meaning="samples",
                in_channels=1, n_samples=1, subregion_index=int(r),
                culture_id=culture_id, **meta)
            written.append(rp)

    if (not args.no_plots) and diag.subregions:
        from channel_subset_viz import plot_subregion_ifrs, plot_subregion_map
        plot_subregion_map(diag, os.path.join(args.out_dir, "subregion_map.png"))
        arr = traces[0] if args.mode == "multichannel" else X
        plot_subregion_ifrs(arr, fs_ifr,
                            os.path.join(args.out_dir, "subregion_ifrs.png"),
                            centers=[s.center for s in diag.subregions])

    print("mode=%s  X.shape=%s  row_meaning=%s  in_channels=%d  n_samples=%d  fs_ifr=%.3f"
          % (args.mode, X.shape, row_meaning, in_channels, n_samples, fs_ifr))
    print("present=%d  discarded(<theta)=%d  centres=%s"
          % (diag.n_present, discarded.size, centers.tolist()))
    print("wrote", npz_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
