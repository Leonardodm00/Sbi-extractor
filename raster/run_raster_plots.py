#!/usr/bin/env python3
"""
run_raster_plots.py -- spike rasters of the real DUP15HD cohort, control vs
pathological: choose the wells, cache their spikes, draw, browse.

Run from Sbi-extractor/raster in the sbi_export environment. Five commands:

    select   which wells (D-026: per class and batch folder, the well with
             the median number of active electrodes, read from its extraction
             record; or --wells to name them). Prints every batch folder's
             wells with n_active and marks the chosen ones; writes
             <out_root>/wells.tsv. Seconds; reads no raw data.
    cache    reads the chosen wells' raw ptrain_<k>.mat rasters with the
             extractor's own loader, checks them against each well's
             extraction record, writes <out_root>/cache/<culture>.npz.
             Minutes per well: run it as the PBS job (run_raster_plots.pbs).
             A cache already built from the same raw folder and manifest is
             kept unless --force.
    plot     static figures for each --window T0 T1 (seconds, repeatable;
             default 0 60): <out_root>/figures/raster_compare_<window>.png,
             .pdf and .json (default window: the first 60 s, or the whole
             recording if shorter); --per-well also draws each well alone.
    view     the interactive viewer: a slider and two boxes set the window;
             Save writes the same files `plot` would (D-028). Needs a display
             (X11 forwarding, e.g. MobaXterm).
    all      select (unless wells.tsv exists) + cache + plot; what the PBS
             job runs.

plot and view read only <out_root>/wells.tsv and <out_root>/cache/, so the
output folder can be copied to any machine with numpy and matplotlib.

Pass conditions (lines that must APPEAR):
    select : [raster] selected 6 well(s) -> .../wells.tsv
    cache  : [raster] cache: 6/6 well(s) ok
    plot   : [raster] plot: wrote N file(s) for M window(s)

Exit status: 0 ok; 1 a well failed or was refused; 2 bad input (missing
manifest, window outside the recording, no display).

Decisions: D-026 (wells), D-027 (rows), D-028 (windows, viewer), D-029 (no
rate panel) -- claude/SBI_decisions_and_ideas_log.md.

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import raster_data as RD                                          # noqa: E402

DEFAULT_WINDOW = (0.0, 60.0)


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
def _print_selection(cohort, selected, rule) -> None:
    chosen = {w.culture for w in selected}
    print("[raster] manifest %s (sha256 %s...)" % (cohort.manifest_path,
                                                   cohort.manifest_digest[:16]))
    print("[raster] preprocessing: fs_raw %g Hz, index_base %d, grid %dx%d, "
          "mfr_threshold %g Hz" % (cohort.fs_raw, cohort.index_base,
                                   cohort.grid_width, cohort.grid_width,
                                   cohort.mfr_threshold))
    print("[raster] rule: %s" % rule)
    for c in cohort.classes:
        in_c = [w for w in cohort.wells if w.class_name == c]
        for b in RD._batches_in_order(in_c):
            group = sorted((w for w in in_c if w.batch == b),
                           key=lambda w: (w.n_active, w.culture))
            print("[raster]   %-13s %-18s %s" % (c, b, "  ".join(
                "%s:%d%s" % (w.well, w.n_active, "*" if w.culture in chosen else "")
                for w in group)))
    print("[raster]   (well:n_active, sorted; * = chosen; n_active = electrodes "
          "with MFR >= %g Hz over the whole recording)" % cohort.mfr_threshold)


def do_select(args):
    cohort = RD.load_cohort(args.manifest)
    if args.wells:
        selected = RD.select_named_wells(cohort.wells, args.wells)
        rule = RD.RULE_NAMED
    else:
        selected = RD.select_typical_wells(cohort.wells, cohort.classes)
        rule = RD.RULE_TYPICAL
    _print_selection(cohort, selected, rule)
    path = os.path.join(args.out_root, "wells.tsv")
    if getattr(args, "dry_run", False):
        print("[raster] dry run: %d well(s) chosen, nothing written" % len(selected))
        return cohort, selected
    RD.write_wells_tsv(path, selected, rule, cohort, cohort.classes)
    for w in selected:
        print("[raster]   %-13s %s  (n_active %d of %d present)"
              % (w.class_name, w.culture, w.n_active, w.n_present))
    print("[raster] selected %d well(s) -> %s" % (len(selected), path))
    return cohort, selected


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _selection_for_cache(args):
    """(cohort, wells): the existing wells.tsv matched to the manifest, or a
    fresh selection when there is none (or --wells / --reselect)."""
    path = os.path.join(args.out_root, "wells.tsv")
    if args.wells or args.reselect or not os.path.isfile(path):
        return do_select(args)
    rows, header = RD.read_wells_tsv(path)
    print("[raster] using the selection in %s (pass --reselect to choose again)"
          % path)
    cohort = RD.load_cohort(args.manifest)
    if header.get("manifest_sha256") and header["manifest_sha256"] != cohort.manifest_digest:
        raise RD.RasterDataError(
            "%s was written from another manifest (sha256 %s..., this one %s...); "
            "run  select  again" % (path, header["manifest_sha256"][:16],
                                    cohort.manifest_digest[:16]))
    by_id = {w.culture: w for w in cohort.wells}
    wells = []
    for r in rows:
        w = by_id.get(r["culture"])
        if w is None or os.path.normpath(r["source_folder"]) != w.source_folder:
            raise RD.RasterDataError("%s: row %s does not match the manifest; run "
                                     "select  again" % (path, r["culture"]))
        wells.append(w)
    return cohort, wells


def do_cache(args):
    cohort, wells = _selection_for_cache(args)
    workers = max(1, min(int(args.workers), len(wells)))
    print("[raster] caching %d well(s) with %d worker(s) -> %s"
          % (len(wells), workers, os.path.join(args.out_root, "cache")))
    t0 = time.time()
    results, failures = [], []
    if workers == 1:
        for w in wells:
            try:
                results.append(RD.build_and_save_cache(w, cohort, args.out_root, args.force))
                _report(results[-1])
            except Exception as exc:                              # noqa: BLE001
                failures.append((w.culture, exc))
                print("[raster] cache FAIL %s: %s: %s" % (w.culture, type(exc).__name__, exc))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(RD.build_and_save_cache, w, cohort, args.out_root,
                                args.force): w for w in wells}
            for f in as_completed(futs):
                w = futs[f]
                try:
                    results.append(f.result())
                    _report(results[-1])
                except Exception as exc:                          # noqa: BLE001
                    failures.append((w.culture, exc))
                    print("[raster] cache FAIL %s: %s: %s"
                          % (w.culture, type(exc).__name__, exc))
    print("[raster] cache: %d/%d well(s) ok in %.0f s"
          % (len(results), len(wells), time.time() - t0))
    return 0 if not failures else 1


def _report(r) -> None:
    print("[raster] cache %-5s %s  present %d  active %d  spikes %s  (%.0f s)  %s"
          % (r["status"], r["culture"], r["n_present"], r["n_active"],
             format(r["n_spikes"], ","), r["seconds"], r["path"]))


# --------------------------------------------------------------------------- #
# plot / view: caches only
# --------------------------------------------------------------------------- #
def _load_selected(out_root):
    """(rows, header, classes, tables) from wells.tsv and the caches."""
    rows, header = RD.read_wells_tsv(os.path.join(out_root, "wells.tsv"))
    classes = [c for c in header.get("classes", "").split(",") if c]
    if not classes:
        classes = []
        for r in rows:
            if r["class_name"] not in classes:
                classes.append(r["class_name"])
    tables = []
    for r in rows:
        t = RD.load_cache(RD.cache_path(out_root, r["culture"]))
        if t.culture != r["culture"] or t.class_name != r["class_name"]:
            raise RD.RasterDataError("cache of %s is not that well's; rebuild it "
                                     "with  cache --force" % r["culture"])
        if (header.get("manifest_sha256")
                and t.meta.get("manifest_digest") != header["manifest_sha256"]):
            raise RD.RasterDataError(
                "cache of %s was built from another manifest than wells.tsv "
                "names; rebuild it with  cache --force" % r["culture"])
        tables.append(t)
    RD.common_preprocessing(tables)
    return rows, header, classes, tables


def _windows(args, tables):
    """The --window list, or [0, min(60 s, shortest recording))."""
    if args.window:
        return [tuple(w) for w in args.window]
    T = min(t.T_rec for t in tables)
    return [(DEFAULT_WINDOW[0], min(DEFAULT_WINDOW[1], T))]


def do_plot(args):
    import raster_plot as RP
    rows, header, classes, tables = _load_selected(args.out_root)
    mfr = tables[0].mfr_threshold
    figdir = os.path.join(args.out_root, "figures")
    written, n_win = [], 0
    for (a, b) in _windows(args, tables):
        views = [RD.raster_view(t, a, b) for t in tables]       # validates [a, b)
        vbc = RD.group_by_class(views, classes)
        stamp = RD.fmt_window(a, b)
        fig, lay = RP.figure_compare(vbc, classes, mfr, width_in=args.width,
                                     panel_height_in=args.panel_height, dpi=args.dpi,
                                     return_layout=True)
        side = RD.figure_sidecar(tables, vbc, a, b, rows, header, how="plot")
        side.update(marker_alpha=lay.alpha, marker_alpha_rule=RP.ALPHA_RULE)
        written += RP.save_figure(fig, os.path.join(figdir, "raster_compare_" + stamp),
                                  args.formats, side)
        if lay.alpha < 1.0:
            print("[raster] window [%g, %g) s is dense (up to %.1f spikes per pixel "
                  "cell): marks drawn with opacity %.3f" % (a, b, 1.0 / lay.alpha, lay.alpha))
        if args.per_well:
            for v in views:
                one = {v.class_name: [v]}
                fig, lay1 = RP.figure_single(v, classes, mfr, width_in=args.width,
                                             dpi=args.dpi, return_layout=True)
                side = RD.figure_sidecar(tables, one, a, b, rows, header,
                                         how="plot --per-well")
                side.update(marker_alpha=lay1.alpha, marker_alpha_rule=RP.ALPHA_RULE)
                written += RP.save_figure(
                    fig, os.path.join(figdir, "per_well", "%s_%s" % (v.culture, stamp)),
                    args.formats, side)
        n_win += 1
        print("[raster] window [%g, %g) s: %s spikes drawn over %d well(s)"
              % (a, b, format(sum(v.n_spikes for v in views), ","), len(views)))
    for p in written:
        print("[raster]   %s" % p)
    print("[raster] plot: wrote %d file(s) for %d window(s)" % (len(written), n_win))
    return 0


def do_view(args):
    import raster_viewer as RV
    RV.require_display()                         # before reading any cache
    rows, header, classes, tables = _load_selected(args.out_root)
    a, b = _windows(args, tables)[0]
    RD.window_bounds(a, b, min(t.T_rec for t in tables), tables[0].fs_raw)
    v = RV.run_viewer(tables, classes, a, b, args.out_root, rows, header,
                      formats=args.formats, dpi=args.dpi, save_width_in=args.width,
                      save_panel_height_in=args.panel_height)
    print("[raster] viewer closed; %d file(s) saved" % len(v.saved))
    return 0


def do_all(args):
    rc = do_cache(args)
    if rc != 0:
        print("[raster] all: cache failed for some well(s); not plotting")
        return rc
    return do_plot(args)


# --------------------------------------------------------------------------- #
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-root", default=RD.DEFAULT_OUT_ROOT,
                        help="output folder: wells.tsv, cache/, figures/ "
                             "(default %(default)s)")
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--manifest", default=RD.DEFAULT_MANIFEST,
                     help="cohort manifest of record (default %(default)s)")
    sel.add_argument("--wells", nargs="+", metavar="CULTURE",
                     help="name the wells instead of D-026's rule, e.g. "
                          "DATA_C_Batch3__ptrain_A1")
    draw = argparse.ArgumentParser(add_help=False)
    draw.add_argument("--window", nargs=2, type=float, action="append",
                      metavar=("T0", "T1"),
                      help="time window [T0, T1) in seconds; repeatable for "
                           "plot, the first is the viewer's start (default 0 60)")
    draw.add_argument("--formats", nargs="+", default=["png", "pdf"],
                      choices=["png", "pdf", "svg"])
    draw.add_argument("--dpi", type=float, default=300.0)
    draw.add_argument("--width", type=float, default=7.2,
                      help="figure width in inches (default %(default)s)")
    draw.add_argument("--panel-height", type=float, default=1.9,
                      help="height of one raster panel in inches (default %(default)s)")
    work = argparse.ArgumentParser(add_help=False)
    work.add_argument("--workers", type=int,
                      default=int(os.environ.get("NCPUS", "1") or 1),
                      help="wells cached in parallel (default $NCPUS or 1)")
    work.add_argument("--force", action="store_true",
                      help="rebuild caches even when current")
    work.add_argument("--reselect", action="store_true",
                      help="choose the wells again even if wells.tsv exists")

    p = argparse.ArgumentParser(
        description="Spike rasters of the real cohort, control vs pathological: "
                    "select, cache, plot, view, all. Details: README.md beside this file.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", parents=[common, sel], help="choose the wells")
    s.add_argument("--dry-run", action="store_true",
                   help="print the choice, write nothing")
    s.set_defaults(func=do_select)
    sub.add_parser("cache", parents=[common, sel, work],
                   help="raw rasters -> per-well spike caches").set_defaults(func=do_cache)
    s = sub.add_parser("plot", parents=[common, draw], help="static figures")
    s.add_argument("--per-well", action="store_true",
                   help="also one figure per well")
    s.set_defaults(func=do_plot)
    sub.add_parser("view", parents=[common, draw],
                   help="interactive viewer (needs a display)").set_defaults(func=do_view)
    s = sub.add_parser("all", parents=[common, sel, work, draw],
                       help="select + cache + plot")
    s.add_argument("--per-well", action="store_true")
    s.set_defaults(func=do_all)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out_root = os.path.abspath(os.path.expanduser(args.out_root))
    try:
        rc = args.func(args)
    except RD.RasterDataError as exc:
        print("[raster] ERROR: %s" % exc)
        return 2
    if args.cmd == "select":
        return 0
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
