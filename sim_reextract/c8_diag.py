#!/usr/bin/env python3
"""c8_diag.py -- read-only diagnosis of a REFUSED C8 gate (Stage C, C8).

When sim_reextract_gate.py refuses, its log names the bad tasks but not why
they went bad. This reads, and changes nothing:

  - out/sim_reextract_gate.log and plan.json in this folder -- the plan the
    gate used. Run this BEFORE any launcher mode: every mode re-plans and
    rewrites plan.json;
  - per bad task, its output folder: mea_iter_*.npz files with data and EMPTY
    (zero bytes), tmp* leftovers, mea_manifest.json, the host, workers and
    start time its job wrote to mea_env.json, the last write, seconds per
    iteration, and whether a re-plan with RESUME=1 would re-run it or keep it
    (sim_reextract_plan.task_is_complete: the manifest alone decides);
  - the same timing over the tasks that passed;
  - the _failures.log entries of the bad tasks;
  - `qstat -xft <array>` (the array id from out/submissions.txt): exit status
    of every subjob, and the bad subjobs' walltime used, host, start, comment;
  - the job logs ~/c8_mea.[oe]<seq>.<index>: the bad tasks' lines, and the
    error lines of every .e log of the array;
  - df and mmlsquota.

Why empty files matter: process_campaign.py writes every mea_iter_*.npz to a
temp file and renames it (_atomic_savez), so a killed process leaves at most a
tmp* file. A zero-byte file under the final name is data lost after the rename
(the node, or the filesystem), which np.load reports as
EOFError('No data left in file').

    cd .../Sbi-extractor && source env.sh && conda activate sbi_export
    cd sim_reextract && python3 c8_diag.py > out/c8_diag_d1.txt 2>&1

The last line is "== D1 done". Options: --here DIR (default: this file's
folder), --logs DIR (default: ~).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

FAIL_RE = re.compile(r"\[gate\] FAIL (\S+)$")
PROBLEM_PREFIX = "[gate]      - "
SUMMARY_PREFIXES = ("[gate] tasks", "[gate] read", "[gate] REFUSED", "[job]")
O_PATTERN = r"workers|dispatching|iters processed|done\."
E_PATTERN = r"PBS|[Kk]ill|Error|Terminat|Traceback|signal|Bus|memory"
E_CENSUS = r"Errno|Error|PBS|[Kk]illed|Terminat|signal"


def hm(x):
    return time.strftime("%m-%d %H:%M", time.localtime(x)) if x else "?"


# --------------------------------------------------------------- loading ---
def load_plan_module(here):
    sys.path.insert(0, here)
    try:
        import sim_reextract_plan as PLAN  # noqa: E402
        return PLAN.task_is_complete
    except Exception as exc:  # noqa: BLE001
        print("(sim_reextract_plan not importable here: %r; the RESUME column is blank)" % (exc,))
        return None


def parse_gate_log(lines, names):
    """{task: [problem lines]} for every '[gate] FAIL <task>' block."""
    bad, cur = collections.OrderedDict(), None
    for line in lines:
        m = FAIL_RE.match(line)
        if m and m.group(1) in names:
            cur = m.group(1)
            bad[cur] = []
        elif cur and line.startswith(PROBLEM_PREFIX):
            bad[cur].append(line[len(PROBLEM_PREFIX):])
        else:
            cur = None
    return bad


def task_facts(t, complete):
    d = t["out_dir"]
    r = {"idx": t.get("array_index"), "plan": "%d/%d" % (t["n_iters"], t["n_topos"])}
    files = glob.glob(os.path.join(d, "topo_*", "mea_iter_*.npz"))
    st = [(os.path.getmtime(f), os.path.getsize(f)) for f in files]
    r["full"] = sum(1 for _, s in st if s > 0)
    r["empty"] = sum(1 for _, s in st if s == 0)
    r["tmp"] = len(glob.glob(os.path.join(d, "topo_*", "tmp*")))
    r["flog"] = sorted(glob.glob(os.path.join(d, "topo_*", "_failures.log")))
    try:
        env = json.load(open(os.path.join(d, "mea_env.json")))
    except Exception:  # noqa: BLE001
        env = {}
    r["host"], r["workers"] = env.get("host", "?"), env.get("workers", "?")
    w = env.get("written")
    r["start"] = time.mktime(time.strptime(w[:19], "%Y-%m-%dT%H:%M:%S")) if w else None
    mp = os.path.join(d, "mea_manifest.json")
    if os.path.isfile(mp):
        m = json.load(open(mp))
        r["man"] = "manifest %s/%s" % (m.get("total_done"), m.get("total_iters"))
        r["end"] = os.path.getmtime(mp)
        n = m.get("total_done") or 0
    else:
        r["man"] = "NO manifest"
        r["end"] = max([x for x, _ in st] or [0]) or None
        n = r["full"]
    r["resume"] = "?" if complete is None else ("KEPT as done by RESUME" if complete(t) else "re-run by RESUME")
    r["h"] = (r["end"] - r["start"]) / 3600.0 if r["start"] and r["end"] else None
    r["rate"] = r["h"] * 3600.0 / n if r["h"] and n else None
    return r


def read_qstat(array_id):
    try:
        txt = subprocess.run(["qstat", "-xft", array_id], capture_output=True, text=True,
                             timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        print("qstat -xft %s failed: %r" % (array_id, exc))
        return {}
    recs, rec, key = {}, None, None
    for line in txt.split("\n"):
        if line.startswith("Job Id:"):
            jid = line.split(":", 1)[1].strip()
            m = re.search(r"\[(\d+)\]", jid)
            rec, key = {}, None
            if m:
                recs[int(m.group(1))] = rec
        elif line.startswith("\t") and rec is not None and key:
            rec[key] += line.strip()          # qstat -f wraps long values: newline + TAB
        elif " = " in line and rec is not None:
            key, v = line.strip().split(" = ", 1)
            rec[key] = v
    return recs


def grep_log(path, pattern, n=4):
    if not os.path.isfile(path):
        return None
    lines = [x.strip() for x in open(path, errors="replace").read().splitlines() if x.strip()]
    hits = [x for x in lines if re.search(pattern, x)][:n]
    return " / ".join(hits)[:300], (lines[-1] if lines else "<empty>")[:200]


# -------------------------------------------------------------- printing ---
def report(here, logs):
    os.chdir(here)
    complete = load_plan_module(here)
    gate = open("out/sim_reextract_gate.log").read().splitlines()
    plan = json.load(open("plan.json"))
    T = {t["campaign"] + "/" + t["sweep"]: t for t in plan["tasks"]}
    print("== D1, read-only. plan.json last written %s, %d task(s) in it"
          % (hm(os.path.getmtime("plan.json")), len(T)))
    print("\n".join(x for x in gate if x.startswith(SUMMARY_PREFIXES)))
    bad = parse_gate_log(gate, T)

    print("--- the %d bad task(s): array index | task | planned iters/topos | on disk: mea_iter files "
          "with data + EMPTY, tmp leftovers | manifest | host, workers | started -> last write (h) | "
          "s/iter | RESUME | gate lines" % len(bad))
    F = {}
    for name, probs in bad.items():
        r = F[name] = task_facts(T[name], complete)
        un = sum("unreadable" in p for p in probs)
        nd = sum("no output directory" in p for p in probs)
        oth = [p for p in probs if "unreadable" not in p and "no output directory" not in p
               and "mea_iter_*.npz, raw has" not in p]
        print("%3s | %-45s | %11s | %5d + %d EMPTY, %d tmp | %-17s | %s, w=%s | %s -> %s (%s h) | %s | %s "
              "| unreadable %d, no dir %d%s" % (
                  r["idx"], name, r["plan"], r["full"], r["empty"], r["tmp"], r["man"], r["host"],
                  r["workers"], hm(r["start"]), hm(r["end"]),
                  "%.1f" % r["h"] if r["h"] is not None else "?",
                  "%.2f" % r["rate"] if r["rate"] else "?", r["resume"], un, nd,
                  ("; " + "; ".join(oth))[:160] if oth else ""))

    good = [n for n in T if n not in bad]
    R = {n: task_facts(T[n], complete) for n in good}
    rates = sorted(r["rate"] for r in R.values() if r["rate"])
    hours = sorted(r["h"] for r in R.values() if r["h"])
    f2 = lambda xs, f, fmt="%.2f": (fmt % f(xs)) if xs else "?"  # noqa: E731
    print("--- the %d task(s) that passed: s/iter min %s median %s max %s; run time (h) median %s max %s; "
          "workers %s" % (len(good), f2(rates, min), f2(rates, statistics.median), f2(rates, max),
                          f2(hours, statistics.median, "%.1f"), f2(hours, max, "%.1f"),
                          dict(collections.Counter(r["workers"] for r in R.values()))))
    allr = list(R.values()) + list(F.values())
    starts = sorted(r["start"] for r in allr if r["start"])
    ends = sorted(r["end"] for r in R.values() if r["end"])
    print("    first task started %s, last started %s; last good task finished %s"
          % (hm(starts[0] if starts else None), hm(starts[-1] if starts else None),
             hm(ends[-1] if ends else None)))
    hosts = collections.Counter(r["host"] for r in F.values())
    hall = collections.Counter(r["host"] for r in allr)
    print("    hosts of the bad tasks: %s (all tasks on those hosts: %s)"
          % (dict(hosts), {h: hall[h] for h in hosts}))
    print("    last writes of the bad tasks, sorted: %s"
          % ", ".join(hm(r["end"]) for r in sorted(F.values(), key=lambda r: r["end"] or 0)))

    print("--- _failures.log files in the bad tasks (path | entries | first entry):")
    for name, r in F.items():
        for fl in r["flog"]:
            txt = open(fl, errors="replace").read()
            ent = re.findall(r"^\S*iter_\d+\.npz: .*$", txt, flags=re.M)
            print("%s/%s | %d | %s" % (name, fl.split("/")[-2], len(ent), (ent[0] if ent else txt[:200])[:400]))
            last = [x for x in txt.splitlines() if x.strip()][-1:] or [""]
            print("    last line: %s" % last[0][:300])

    sub = [x.split() for x in open("out/submissions.txt").read().splitlines() if x.strip()]
    print("--- out/submissions.txt, last 3:")
    print("\n".join(" ".join(s) for s in sub[-3:]))
    arr = [s[2] for s in sub if len(s) > 2 and s[1] == "array"]
    if arr:
        A = arr[-1]
        seq = A.split("[")[0].split(".")[0]
        recs = read_qstat(A)
        print("--- PBS, array %s: %d subjob record(s); Exit_status counts: %s"
              % (A, len(recs), dict(collections.Counter(r.get("Exit_status", "?") for r in recs.values()))))
        bidx = {F[n]["idx"] for n in F}
        for i in sorted(recs):
            r = recs[i]
            if i in bidx or r.get("Exit_status") != "0":
                print("%3d | exit %s | walltime used %s | %s | start %s | %s" % (
                    i, r.get("Exit_status"), r.get("resources_used.walltime"),
                    r.get("exec_host", "?").split("/")[0], r.get("stime"), r.get("comment", "")[:150]))
        pat = os.path.join(logs, "c8_mea.[oe]%s*" % seq)
        lo = sorted(glob.glob(pat))
        print("--- job logs: %d file(s) %s (e.g. %s)" % (len(lo), pat, os.path.basename(lo[0]) if lo else "-"))
        for name, r in F.items():
            out = []
            for k, p in (("o", O_PATTERN), ("e", E_PATTERN)):
                g = grep_log(os.path.join(logs, "c8_mea.%s%s.%s" % (k, seq, r["idx"])), p)
                out.append(".%s missing" % k if g is None else ".%s: %s || last: %s" % (k, g[0], g[1]))
            print("%3s %s\n      %s" % (r["idx"], name, "\n      ".join(out)))
        allE = sorted(glob.glob(os.path.join(logs, "c8_mea.e%s*" % seq)))
        C = collections.Counter()
        for p in allE:
            for x in open(p, errors="replace").read().splitlines():
                if re.search(E_CENSUS, x):
                    C[re.sub(r"\d+", "N", re.sub(r"/\S+", "<path>", x.strip()))[:160]] += 1
        print("--- error lines over all %d .e logs of the array (count | line, numbers and paths masked):" % len(allE))
        print("\n".join("%5d | %s" % (n, x) for x, n in C.most_common(10)) or "  (none)")
    print("--- storage now:")
    for cmd in (["df", "-h", os.path.dirname(plan["out_root"])], ["mmlsquota", "--block-size", "auto"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print("$ " + " ".join(cmd) + "\n" + (r.stdout + r.stderr).strip()[:1200])
        except Exception as exc:  # noqa: BLE001
            print("$ %s: %r" % (" ".join(cmd), exc))
    print("== D1 done")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--here", default=os.path.dirname(os.path.abspath(__file__)),
                   help="the sim_reextract folder holding plan.json and out/ (default: this file's)")
    p.add_argument("--logs", default="~", help="where the array's .o/.e logs are (default: ~, #PBS -k eo)")
    args = p.parse_args(argv)
    report(os.path.abspath(args.here), os.path.expanduser(args.logs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
