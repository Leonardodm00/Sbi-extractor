#!/usr/bin/env python3
"""
verify_origin_manifest.py -- every row of ORIGIN_MANIFEST.tsv still true?

The manifest is this directory's byte-identity claim: each row names a file,
its sha256, the DSN blob it came from (or "-" for files with no DSN source),
and a status. Nothing verified it, so it drifted: migration step 5b edited
run_extractor_array_mea.pbs and run_extractor_smoke.pbs (the ENV_NAME default)
without refreshing their shas, and that went unnoticed from 2026-09-19 to
2026-09-21. A claim nobody checks is not a guarantee.

Run:
    python3 verify_origin_manifest.py            # this directory's manifest
    python3 verify_origin_manifest.py PATH.tsv   # another one

Exit 0 and one PASS line if every listed file exists and hashes as recorded.
Exit 1 and one line per problem otherwise. Needs nothing but the standard
library, so it runs in any environment, including one without numpy.

It checks the CURRENT sha, not the source blob: verifying "verbatim" against
the retired DSN repo needs that repo, which is the point of recording the
source path rather than re-deriving it. A file whose status is "verbatim" and
whose sha has changed is therefore reported as a changed file, and whether
that is a defect or an unrecorded edit is for the reader to decide.

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
import hashlib
import os
import sys


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest_path):
    base = os.path.dirname(os.path.abspath(manifest_path))
    problems, n = [], 0
    with open(manifest_path) as fh:
        for lineno, raw in enumerate(fh, 1):
            row = raw.rstrip("\n")
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            parts = row.split("\t")
            if len(parts) < 2:
                problems.append("line %d: not a tab-separated row: %r"
                                % (lineno, row[:60]))
                continue
            name, recorded = parts[0], parts[1]
            path = os.path.join(base, name)
            n += 1
            if not os.path.exists(path):
                problems.append("%-34s LISTED BUT ABSENT" % name)
                continue
            actual = sha256_of(path)
            if actual != recorded:
                problems.append("%-34s SHA CHANGED  recorded %s  actual %s"
                                % (name, recorded[:16], actual[:16]))
    # the mirror: a file present here but claimed by no row. GENERATED names
    # are machine state (gitignored, rewritten by list_extraction_jobs.py), so
    # they are deliberately not claimed by any row.
    GENERATED = ("extraction_manifest.tsv",)
    listed = set()
    with open(manifest_path) as fh:
        for raw in fh:
            if raw.strip() and not raw.lstrip().startswith("#") and "\t" in raw:
                listed.add(raw.split("\t")[0])
    unlisted = sorted(
        f for f in os.listdir(base)
        if os.path.isfile(os.path.join(base, f))
        and f not in listed
        and not f.startswith(".")
        and f not in GENERATED
        and os.path.splitext(f)[1] in (".py", ".sh", ".pbs", ".md", ".tsv")
        and f != os.path.basename(manifest_path))
    return n, problems, unlisted


def main(argv):
    manifest = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ORIGIN_MANIFEST.tsv")
    if not os.path.exists(manifest):
        print("[manifest] ABSENT: %s" % manifest)
        return 1
    n, problems, unlisted = verify(manifest)
    for p in problems:
        print("[manifest] %s" % p)
    if unlisted:
        print("[manifest] NOTE: %d file(s) here are in no row, so nothing "
              "claims anything about them:" % len(unlisted))
        for f in unlisted:
            print("[manifest]   %s" % f)
    if problems:
        print("[manifest] FAIL %d row(s) checked, %d problem(s)"
              % (n, len(problems)))
        return 1
    print("[manifest] PASS %d row(s), every file present and unchanged" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
