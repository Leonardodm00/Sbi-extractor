#!/usr/bin/env python3
"""c8_cleanup.py -- after the C8 record: list, then delete, what the record makes redundant.

The user's call of 2026-10-06 (decisions log D-065): once C8's record exists,
remove the old extractions of everything the re-extraction replaced, the old
Giulia real-arm partial roots, and the raw simulations of the seed duplicates
(the replays the record drops). Two steps, list then delete:

    python3 c8_cleanup.py                 # LIST: nothing is deleted
    python3 c8_cleanup.py --delete CODE   # DELETE exactly the listed paths

The list goes to out/c8_cleanup_list.txt (group, path, files, bytes, note) and
ends with a CODE, 12 hex digits of the sha256 of the listed paths. --delete
recomputes the list with every check, refuses unless it lists the same paths
and CODE matches, then removes each path and logs it to
out/c8_cleanup_deleted.txt. Re-running after a deletion lists nothing that is
gone (idempotent).

What is listed, each only if its checks pass:
  A  the raw simulation folder (<sim_main>/<campaign>/<sweep>) of every replay
     in the record's `excluded` (an entry with `replay_of`): the folder is a
     real directory exactly where the record says, its kept copy is a record
     task whose raw folder exists, and the two job_args.json carry the same
     seed values (sim_reextract_plan.task_seed). A replay with files_lost > 0
     holds iteration files no kept copy has; it is listed with that note
     (D-064 left them out of the extraction; deleting the folder loses them).
  B  the old detections <old_root>/<campaign>/<sweep> of every task the record
     covers -- re-extracted into the new root, or dropped as a replay.
     Anything else under <old_root> is printed as "kept" and not touched.
  C  <out_root>_replays_moved/<campaign>/<sweep> for the record's replays (the
     10-03 test task, moved there by M1).
  D  the Giulia real-arm partial roots beside the cohort of record, only if
     extracted_giulia/cohort_manifest.json matches its .sha256.

Refused outright: no record, or a record that does not match its .sha256;
a record task whose output is not complete (mea_manifest.json done ==
planned); a listed path that is a symlink, lies inside the new output root,
or contains a kept task's raw or output folder.

Needs env.sh sourced (sim_reextract_plan imports cohort_manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "record": "/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs_v2/REEXTRACTION_RECORD.json",
    "old_root": "/davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs",
    "giulia_dir": "/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro",
    "giulia_record_root": "extracted_giulia",
    "giulia_partials": ("extracted_giulia_partial_20261002", "extracted_giulia_partial2_20261002"),
    "out_dir": os.path.join(HERE, "out"),
}
LIST_NAME = "c8_cleanup_list.txt"
DELETED_NAME = "c8_cleanup_deleted.txt"


class Refused(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sidecar_ok(path):
    side = path + ".sha256"
    if not (os.path.isfile(path) and os.path.isfile(side)):
        return False
    want = open(side).read().split()
    return bool(want) and want[0] == sha256_file(path)


def tree_size(path):
    n = b = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                b += os.lstat(os.path.join(root, f)).st_size
                n += 1
            except OSError:
                pass
    return n, b


def inside(child, parent):
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def complete(task):
    mp = os.path.join(task["out_dir"], "mea_manifest.json")
    try:
        m = json.load(open(mp))
    except (OSError, ValueError):
        return False
    return int(m.get("total_done", -1)) == int(m.get("total_iters", -2)) == int(task["n_iters"])


# ------------------------------------------------------------ the list ---
def build_list(cfg, task_seed, say=print):
    """[(group, path, note)] of what may go; raises Refused on a blocking problem."""
    rp = cfg["record"]
    if not os.path.isfile(rp):
        raise Refused("no record at %s -- C8 is not closed; nothing to clean yet" % rp)
    if not sidecar_ok(rp):
        raise Refused("%s does not match its .sha256" % rp)
    rec = json.load(open(rp))
    if rec.get("record") != "sim_reextraction":
        raise Refused("%s is not a sim_reextraction record" % rp)
    tasks = {"%s/%s" % (t["campaign"], t["sweep"]): t for t in rec["tasks"]}
    bad = [n for n, t in tasks.items() if not complete(t)]
    if bad:
        raise Refused("%d record task(s) not complete on disk, e.g. %s" % (len(bad), bad[0]))
    out_root, sim_main = rec["out_root"], rec["sim_main"]
    replays = [e for e in rec.get("excluded", []) if e.get("replay_of")]
    say("[clean] record %s: %d task(s), %d replay(s) dropped" % (rp, len(tasks), len(replays)))
    items = []

    # A: raw simulations of the replays
    for e in replays:
        name, keep = "%s/%s" % (e["campaign"], e["sweep"]), tasks.get(e["replay_of"])
        d = e["campaign_dir"]
        why = None
        if keep is None:
            why = "its kept copy %s is not a record task" % e["replay_of"]
        elif os.path.normpath(d) != os.path.normpath(os.path.join(sim_main, e["campaign"], e["sweep"])):
            why = "campaign_dir %s is not <sim_main>/<campaign>/<sweep>" % d
        elif not os.path.exists(d):
            continue                                   # already gone
        elif os.path.islink(d) or not os.path.isdir(d):
            why = "not a real directory"
        elif not os.path.isdir(keep["campaign_dir"]):
            why = "its kept copy's raw folder %s is missing" % keep["campaign_dir"]
        else:
            try:
                if task_seed(d) != task_seed(keep["campaign_dir"]):
                    why = "its seed values differ from %s's" % e["replay_of"]
            except Exception as exc:  # noqa: BLE001
                why = "job_args.json unreadable: %r" % (exc,)
        if why:
            say("[clean] NOT LISTED (A) %s: %s" % (name, why))
            continue
        note = "replay of %s" % e["replay_of"]
        if int(e.get("files_lost") or 0):
            note += "; holds %d iteration file(s) no kept copy has -- lost with it (D-064)" % int(e["files_lost"])
        items.append(("A", d, note))

    # B: old detections of every task the record covers
    old = cfg["old_root"]
    covered = {}
    for n in tasks:
        covered[n] = "re-extracted into %s" % out_root
    for e in replays:
        covered["%s/%s" % (e["campaign"], e["sweep"])] = "replay of %s, not re-extracted" % e["replay_of"]
    if os.path.isdir(old):
        if inside(old, out_root) or inside(out_root, old):
            raise Refused("old root %s and new root %s overlap" % (old, out_root))
        kept = []
        for c in sorted(os.listdir(old)):
            cd = os.path.join(old, c)
            if not os.path.isdir(cd) or os.path.islink(cd):
                kept.append(c)
                continue
            for s in sorted(os.listdir(cd)):
                p = os.path.join(cd, s)
                n = "%s/%s" % (c, s)
                if n in covered and os.path.isdir(p) and not os.path.islink(p):
                    items.append(("B", p, covered[n]))
                else:
                    kept.append(n)
        say("[clean] kept under %s (not covered by the record): %d entr%s%s"
            % (old, len(kept), "y" if len(kept) == 1 else "ies",
               (": " + ", ".join(kept[:12]) + (" ..." if len(kept) > 12 else "")) if kept else ""))
    else:
        say("[clean] no old root at %s" % old)

    # C: moved replay detections
    moved = out_root.rstrip(os.sep) + "_replays_moved"
    for e in replays:
        p = os.path.join(moved, e["campaign"], e["sweep"])
        if os.path.isdir(p) and not os.path.islink(p):
            items.append(("C", p, "moved detections of a replay of %s" % e["replay_of"]))

    # D: Giulia partial roots
    if cfg.get("giulia_dir"):
        gm = os.path.join(cfg["giulia_dir"], cfg["giulia_record_root"], "cohort_manifest.json")
        if not sidecar_ok(gm):
            say("[clean] NOT LISTED (D): %s missing or not matching its .sha256" % gm)
        else:
            for pn in cfg["giulia_partials"]:
                p = os.path.join(cfg["giulia_dir"], pn)
                if os.path.isdir(p) and not os.path.islink(p):
                    items.append(("D", p, "refused Giulia real-arm run; cohort of record %s" % os.path.dirname(gm)))

    # safety: nothing listed may hold what the record keeps
    keep_dirs = [t["campaign_dir"] for t in tasks.values()] + [t["out_dir"] for t in tasks.values()]
    if cfg.get("giulia_dir"):
        keep_dirs.append(os.path.join(cfg["giulia_dir"], cfg["giulia_record_root"]))
    for g, p, _ in items:
        if os.path.islink(p):
            raise Refused("%s is a symlink" % p)
        if inside(p, out_root):
            raise Refused("%s lies inside the new output root" % p)
        for k in keep_dirs:
            if inside(k, p):
                raise Refused("%s contains %s, which the record keeps" % (p, k))
    return items


def list_code(items):
    return hashlib.sha256("\n".join(sorted(p for _, p, _ in items)).encode()).hexdigest()[:12]


# ------------------------------------------------------------- the steps ---
def do_list(cfg, task_seed):
    items = build_list(cfg, task_seed)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    tot = {}
    lines = ["# c8_cleanup list, %s; group<TAB>path<TAB>files<TAB>bytes<TAB>note" % time.strftime("%Y-%m-%dT%H:%M:%S")]
    for g, p, note in items:
        n, b = tree_size(p)
        t = tot.setdefault(g, [0, 0, 0])
        t[0] += 1; t[1] += n; t[2] += b
        lines.append("%s\t%s\t%d\t%d\t%s" % (g, p, n, b, note))
    code = list_code(items)
    lines.append("# CODE %s" % code)
    lp = os.path.join(cfg["out_dir"], LIST_NAME)
    with open(lp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    names = {"A": "raw simulations of replays", "B": "old detections", "C": "moved replay detections",
             "D": "Giulia partial roots"}
    for g in sorted(tot):
        print("[clean] %s %-28s: %4d folder(s), %9d file(s), %8.1f GB"
              % (g, names[g], tot[g][0], tot[g][1], tot[g][2] / 1e9))
    for g, p, note in items:
        if "lost with it" in note:
            print("[clean] NOTE %s: %s" % (p, note))
    print("[clean] LISTED %d folder(s) in %s -- nothing deleted. Read it, then:" % (len(items), lp))
    print("        python3 c8_cleanup.py --delete %s" % code)
    return 0


def do_delete(cfg, task_seed, code):
    lp = os.path.join(cfg["out_dir"], LIST_NAME)
    if not os.path.isfile(lp):
        raise Refused("no %s: run the list step first" % lp)
    saved = [x.split("\t")[1] for x in open(lp).read().splitlines() if x and not x.startswith("#")]
    items = build_list(cfg, task_seed, say=lambda s: None)
    now = sorted(p for _, p, _ in items)
    if now != sorted(saved):
        raise Refused("the list changed since %s was written (%d now, %d listed): run the list step again"
                      % (lp, len(now), len(saved)))
    if code != list_code(items):
        raise Refused("CODE %s is not this list's (%s)" % (code, list_code(items)))
    dp = os.path.join(cfg["out_dir"], DELETED_NAME)
    failed = 0
    with open(dp, "a") as log:
        for i, (g, p, note) in enumerate(items, 1):
            try:
                shutil.rmtree(p)
                log.write("%s\tdeleted\t%s\t%s\t%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), g, p, note))
                print("[clean] %d/%d deleted %s" % (i, len(items), p))
            except OSError as exc:
                failed += 1
                log.write("%s\tFAILED\t%s\t%s\t%r\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), g, p, exc))
                print("[clean] %d/%d FAILED %s: %r" % (i, len(items), p, exc))
    for g, p, _ in items:                      # empty campaign folders left behind
        parent = os.path.dirname(p)
        if g in ("B", "C") and os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
    print("[clean] DONE: %d deleted, %d failed; log %s" % (len(items) - failed, failed, dp))
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delete", metavar="CODE", help="delete what the last list holds; CODE from the list step")
    p.add_argument("--record", default=DEFAULTS["record"])
    p.add_argument("--old-root", default=DEFAULTS["old_root"])
    p.add_argument("--giulia-dir", default=DEFAULTS["giulia_dir"], help="'' to leave Giulia out")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    args = p.parse_args(argv)
    cfg = dict(DEFAULTS, record=args.record, old_root=args.old_root, giulia_dir=args.giulia_dir,
               out_dir=args.out_dir)
    sys.path.insert(0, HERE)
    try:
        import sim_reextract_plan as PLAN  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        print("[clean] REFUSED: sim_reextract_plan not importable (source env.sh first): %r" % (exc,))
        return 2
    try:
        return do_delete(cfg, PLAN.task_seed, args.delete) if args.delete else do_list(cfg, PLAN.task_seed)
    except Refused as exc:
        print("[clean] REFUSED: %s -- nothing deleted" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
