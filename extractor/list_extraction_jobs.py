#!/usr/bin/env python3
"""
list_extraction_jobs.py -- enumerate every well the cohort needs extracted, and
exactly where each one's output must land, using the SAME find_wells()/expand()
functions make_mea_specs.py calls later (both now import them from the DSN
tree's torch-free cohort.py; this script needs no torch since Stage D).

WHY THIS EXISTS
---------------
extract_root + extract_layout in the cohort config define where extraction
output is supposed to go. Nothing enforces that the extraction JOB actually
writes there, other than a human copying the template by hand -- which is
exactly how a mismatch happens (a typo'd path, a stale --out-dir template) and
it fails silently: make_mea_specs.py just reports "0 wells found" with no
indication why. This script removes the copying: it imports the same
root_name_for()/find_wells()/expand() that make_mea_specs.py uses (from the
DSN tree's cohort.py) and computes the SAME path every well would resolve to,
so the extraction driver and the specs generator cannot disagree about where
a well's output belongs.

Writes two plain-ASCII, LF-only files:

    extraction_manifest.tsv
        one line per well: <raw_folder> TAB <out_dir> TAB <culture_id>
        Line N (1-indexed) is array task N-1.

    extraction_flags.sh
        a single EXTRA_FLAGS="..." line built from cohort.* in the config
        (fs_raw, base, grid_width, n_subsets, electrodes_per_subset,
        mfr_threshold, w_size), sourced by the PBS array job. The flags used
        at extraction time therefore cannot drift from what cohort.*
        declares -- there is only one place these numbers are written.

Usage
-----
    python3 list_extraction_jobs.py \
        --config $SBI_HPC_DIR/dsn/hpc/Config/config_mea_joint_full.davinci.json

    --out-manifest PATH   default: extraction_manifest.tsv (next to this script)
    --out-flags PATH      default: extraction_flags.sh     (next to this script)
    --extract-root PATH   override cohort.extract_root (Stage D: extracted_v2/)
    --mode {per_region_single,multichannel}
                          default: per_region_single -- what K3 requires.
                          multichannel is accepted for completeness but is
                          NOT what the cohort-grouping design in this repo
                          consumes; only pass it if you know you want it.

Prints the well count. Size the PBS array to it:
    qsub -J 0-$(($(wc -l < extraction_manifest.tsv) - 1)) run_extractor_array_mea.pbs

Pure ASCII, LF only (hpc-python-compat).
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# [Stage D, 2026-09-21] The cohort block is read through ../cohort_config.py,
# which imports the DSN tree's torch-free cohort.py (resolved by ../dsn_tree.py
# through SBI_HPC_DIR). Nothing here imports config.py or make_mea_specs.py
# any more, so this script runs in the extractor's own environment.
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cohort_config as CC                                        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-manifest",
                    default=os.path.join(_HERE, "extraction_manifest.tsv"))
    ap.add_argument("--out-flags",
                    default=os.path.join(_HERE, "extraction_flags.sh"))
    ap.add_argument("--mode", default="per_region_single",
                    choices=["per_region_single", "multichannel"])
    ap.add_argument("--extract-root", default=None,
                    help="override cohort.extract_root (Stage D: point at "
                         "extracted_v2/ while the config still names "
                         "extracted/). Affects the manifest's out_dir column "
                         "only; the flags file does not carry the root.")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.config):
        print("ABORT: config not found: %s" % args.config)
        return 2
    cohort = CC.load_cohort(args.config, extract_root=args.extract_root)
    if args.extract_root:
        print("extract_root overridden -> %s" % cohort.extract_root)
    if not cohort.class_roots:
        print("ABORT: cohort.class_roots is empty in %s" % args.config)
        return 2
    if "REPLACE/ME" in str(cohort.extract_root) or any(
            "REPLACE/ME" in str(r)
            for roots in cohort.class_roots.values() for r in roots):
        print("ABORT: cohort still has REPLACE/ME placeholder path(s). "
              "Fill in class_roots and extract_root before listing jobs.")
        return 2

    rows = []          # (folder, out_dir, culture_id)
    missing_roots, empty_roots = [], []
    for c in range(cohort.n_classes()):
        cname = cohort.name_of_class(c)
        for root in cohort.class_roots[str(c)]:
            root = str(root)
            root_name = CC.root_name_for(root)
            wells = CC.find_wells(root, cohort.well_glob)
            if wells is None:
                missing_roots.append(root)
                continue
            if not wells:
                empty_roots.append(root)
                continue
            for well in wells:
                folder = os.path.join(root, well)
                rel = CC.expand(cohort.extract_layout, c, cname, root_name, well)
                out_dir = os.path.join(str(cohort.extract_root), rel)
                culture = CC.expand(cohort.culture_template, c, cname,
                                     root_name, well)
                rows.append((folder, out_dir, culture))

    for r in missing_roots:
        print("WARNING: root not found, skipped: %s" % r)
    for r in empty_roots:
        print("WARNING: root has no well matching %r, skipped: %s"
              % (cohort.well_glob, r))
    if not rows:
        print("ABORT: found zero wells to extract. Check class_roots and "
              "well_glob in the config.")
        return 1

    n_uniq_culture = len(set(c for _f, _o, c in rows))
    if n_uniq_culture != len(rows):
        # A collision here means at least two wells share BOTH culture id and
        # out_dir (both come from the same root_name). Writing the manifest
        # anyway would let extraction quietly overwrite one well's output
        # with another well's -- data loss with no error message anywhere.
        # This must stop the run, not warn past it.
        seen, dupes = {}, {}
        for folder, out_dir, culture in rows:
            seen.setdefault(culture, []).append(folder)
        for culture, folders in seen.items():
            if len(folders) > 1:
                dupes[culture] = folders
        print("ABORT: %d well(s) but only %d distinct culture id(s) -- "
              "culture_template collides for these:" % (len(rows), n_uniq_culture))
        for culture, folders in dupes.items():
            print("  %s" % culture)
            for f in folders:
                print("    %s" % f)
        print("No manifest written. root_name_for() already disambiguates by "
              "parent+leaf, not leaf alone; a collision surviving that means "
              "two wells share BOTH their parent AND leaf directory names. "
              "Fix the colliding paths, or tell me the exact folders above "
              "and I will extend the disambiguator.")
        return 1

    parent = os.path.dirname(os.path.abspath(args.out_manifest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out_manifest, "w", encoding="ascii", newline="\n") as fh:
        for folder, out_dir, culture in rows:
            fh.write("%s\t%s\t%s\n" % (folder, out_dir, culture))

    flags = CC.build_extra_flags(cohort)
    parent = os.path.dirname(os.path.abspath(args.out_flags))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out_flags, "w", encoding="ascii", newline="\n") as fh:
        fh.write(flags)

    print("")
    print("wrote %d well(s) -> %s" % (len(rows), args.out_manifest))
    print("wrote extraction flags -> %s" % args.out_flags)
    print("")
    print("size the PBS array to this count, e.g.:")
    print("  qsub -J 0-%d run_extractor_array_mea.pbs" % (len(rows) - 1))
    print("mode for the array job: %s "
          "(pass --mode via the .pbs file's extractor call, not here -- "
          "this script only lists jobs, it does not run the extractor)"
          % args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
