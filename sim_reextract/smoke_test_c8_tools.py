#!/usr/bin/env python3
"""smoke_test_c8_tools.py -- checks for c8_diag.py and c8_cleanup.py on a synthetic tree.

    cd .../Sbi-extractor && source env.sh && conda activate sbi_export
    cd sim_reextract && python3 smoke_test_c8_tools.py

Last line "ALL 13 CHECKS PASSED", or "FAILED n of 13" with the failures above.
Needs no cluster, no ANN tools, no qstat: the tree is built in a temp folder
(raw campaign folders with job_args.json, an old and a new output root, a
record with its .sha256, a moved replay, a Giulia folder with a cohort
manifest and two partial roots, a gate log, a plan.json, fake job logs).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_reextract_plan  # noqa: E402,F401  (needs env.sh: it imports cohort_manifest)
import c8_cleanup  # noqa: E402
import c8_diag  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%s %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if (detail and not ok) else ""))


def write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh)


def write_with_sidecar(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    with open(path, "w") as fh:
        fh.write(blob)
    with open(path + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (hashlib.sha256(blob.encode()).hexdigest(), os.path.basename(path)))


def touch(path, nbytes=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * nbytes)


class Tree:
    """K kept (v9), R1 and R2 replays of K (R2 with files_lost 1), R3 a replay whose
    job_args seed was tampered with, U an old-root task the record does not cover."""

    def __init__(self, root):
        self.root = root
        self.main = os.path.join(root, "Main")
        self.new = os.path.join(root, "Outputs_v2")
        self.old = os.path.join(root, "Outputs")
        self.giu = os.path.join(root, "Giulia_Astro")
        self.out = os.path.join(root, "out")
        seed = {"_resolved_seed_master": 4000000, "seed_master": None, "seed_device": 1,
                "seed_neuron": 2, "seed_synapse": 3, "seed_astro": 4}
        self.names = {"K": ("campaign_cadex_rho1300v9", "sweep_cpu_task0000"),
                      "R1": ("campaign_cadex_rho1300v1", "sweep_cpu_task0000"),
                      "R2": ("campaign_cadex_rho1300v2", "sweep_cpu_task0000"),
                      "R3": ("campaign_cadex_rho1300v3", "sweep_cpu_task0000"),
                      "U": ("campaign_cadex_rho1300v4", "sweep_cpu_task0009")}
        for k, (c, s) in self.names.items():
            if k == "U":
                continue
            d = os.path.join(self.main, c, s)
            write_json(os.path.join(d, "job_args.json"), dict(seed, seed_astro=99) if k == "R3" else seed)
            touch(os.path.join(d, "topo_00001", "iter_00000.npz"))
        self.raw = lambda k: os.path.join(self.main, *self.names[k])
        self.oldp = lambda k: os.path.join(self.old, *self.names[k])
        for k in ("K", "R1", "R2", "U"):
            touch(os.path.join(self.oldp(k), "topo_00001", "mea_iter_00000.npz"))
        kout = os.path.join(self.new, *self.names["K"])
        write_json(os.path.join(kout, "mea_manifest.json"), {"total_done": 1, "total_iters": 1, "n_topos": 1})
        write_json(os.path.join(kout, "mea_env.json"), {"host": "n1", "workers": 1})
        touch(os.path.join(kout, "topo_00001", "mea_iter_00000.npz"))
        self.moved = os.path.join(root, "Outputs_v2_replays_moved", *self.names["R1"])
        touch(os.path.join(self.moved, "topo_00001", "mea_iter_00000.npz"))
        write_with_sidecar(os.path.join(self.giu, "extracted_giulia", "cohort_manifest.json"), {"n_wells": 16})
        for pn in c8_cleanup.DEFAULTS["giulia_partials"]:
            touch(os.path.join(self.giu, pn, "w1", "trace_subregion_00.npz"))
        task = lambda k: {"campaign": self.names[k][0], "sweep": self.names[k][1],  # noqa: E731
                          "campaign_dir": self.raw(k), "out_dir": os.path.join(self.new, *self.names[k]),
                          "n_topos": 1, "n_iters": 1}
        exc = []
        for k, lost in (("R1", 0), ("R2", 1), ("R3", 0)):
            e = task(k)
            e.update(reason="replay (D-061) of K", replay_of="%s/%s" % self.names["K"], files_lost=lost)
            exc.append(e)
        exc.append({"campaign": "campaign_cadex_rho1300v7", "sweep": "sweep_cpu_task0009",
                    "reason": "no iteration files"})
        self.record = {"record": "sim_reextraction", "out_root": self.new, "sim_main": self.main,
                       "tasks": [task("K")], "excluded": exc}
        self.rp = os.path.join(self.new, "REEXTRACTION_RECORD.json")
        write_with_sidecar(self.rp, self.record)

    def args(self, *extra):
        return ["--record", self.rp, "--old-root", self.old, "--giulia-dir", self.giu,
                "--out-dir", self.out] + list(extra)


def run(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(argv)
    return rc, buf.getvalue()


def code_of(text):
    for line in text.splitlines():
        if line.strip().startswith("python3 c8_cleanup.py --delete "):
            return line.split()[-1]
    return None


def cleanup_checks(base):
    t = Tree(os.path.join(base, "c1"))
    os.rename(t.rp, t.rp + ".aside")
    rc, out = run(c8_cleanup.main, t.args())
    check("CL1 no record: refused", rc == 2 and "no record" in out, out)
    os.rename(t.rp + ".aside", t.rp)

    with open(t.rp, "a") as fh:
        fh.write(" ")
    rc, out = run(c8_cleanup.main, t.args())
    check("CL2 record not matching its .sha256: refused", rc == 2 and "does not match" in out, out)
    write_with_sidecar(t.rp, t.record)

    mp = os.path.join(t.new, *t.names["K"], "mea_manifest.json")
    write_json(mp, {"total_done": 0, "total_iters": 1, "n_topos": 1})
    rc, out = run(c8_cleanup.main, t.args())
    check("CL3 a record task incomplete on disk: refused", rc == 2 and "not complete" in out, out)
    write_json(mp, {"total_done": 1, "total_iters": 1, "n_topos": 1})

    rc, out = run(c8_cleanup.main, t.args())
    lst = [x.split("\t") for x in open(os.path.join(t.out, c8_cleanup.LIST_NAME)).read().splitlines()
           if x and not x.startswith("#")]
    got = sorted((g, p) for g, p, *_ in lst)
    want = sorted([("A", t.raw("R1")), ("A", t.raw("R2")), ("B", t.oldp("K")), ("B", t.oldp("R1")),
                   ("B", t.oldp("R2")), ("C", t.moved)]
                  + [("D", os.path.join(t.giu, pn)) for pn in c8_cleanup.DEFAULTS["giulia_partials"]])
    check("CL4 the list is exactly A R1 R2, B K R1 R2, C R1, D both partials", rc == 0 and got == want,
          "%r\n%s" % (got, out))
    check("CL5 R3 (seed differs) not listed and said so; U kept; R2's lost file noted",
          "NOT LISTED (A) %s/%s: its seed values differ" % t.names["R3"] in out
          and "%s/%s" % t.names["U"] in out and "lost with it" in out, out)
    still = all(os.path.isdir(p) for _, p in want)
    check("CL6 the list step deleted nothing", still)

    code = code_of(out)
    rc, out2 = run(c8_cleanup.main, t.args("--delete", "000000000000"))
    check("CL7 delete with a wrong CODE: refused, nothing deleted",
          rc == 2 and "is not this list's" in out2 and all(os.path.isdir(p) for _, p in want), out2)

    extra = os.path.join(t.root, "Outputs_v2_replays_moved", *t.names["R2"])
    touch(os.path.join(extra, "f.npz"))
    rc, out3 = run(c8_cleanup.main, t.args("--delete", code))
    check("CL8 delete after the tree changed: refused", rc == 2 and "list changed" in out3, out3)
    shutil.rmtree(extra)

    rc, out4 = run(c8_cleanup.main, t.args("--delete", code))
    gone = all(not os.path.exists(p) for _, p in want)
    kept = all(os.path.isdir(p) for p in (t.raw("K"), t.raw("R3"), t.oldp("U"), os.path.join(t.new, *t.names["K"]),
                                          os.path.join(t.giu, "extracted_giulia")))
    check("CL9 delete with the right CODE: exactly the listed paths gone, the rest intact",
          rc == 0 and gone and kept and "DONE: %d deleted, 0 failed" % len(want) in out4, out4)

    rc, out5 = run(c8_cleanup.main, t.args())
    check("CL10 listing again: nothing left to list (idempotent)", rc == 0 and "LISTED 0 folder(s)" in out5, out5)

    t2 = Tree(os.path.join(base, "c2"))
    with open(os.path.join(t2.giu, "extracted_giulia", "cohort_manifest.json"), "a") as fh:
        fh.write(" ")
    os.symlink(t2.raw("K"), os.path.join(t2.old, "campaign_cadex_rho1300v3"))
    rc, out6 = run(c8_cleanup.main, t2.args())
    lst2 = open(os.path.join(t2.out, c8_cleanup.LIST_NAME)).read()
    check("CL11 Giulia manifest not matching: D not listed; a symlinked campaign folder is kept, not followed",
          rc == 0 and "NOT LISTED (D)" in out6 and "\tD\t" not in "\t" + lst2 and "Giulia" not in lst2
          and "campaign_cadex_rho1300v3" in out6.split("kept under")[1], out6)


def diag_checks(base):
    root = os.path.join(base, "d1")
    here = os.path.join(root, "sim_reextract")
    new = os.path.join(root, "Outputs_v2")
    tasks = []
    for i, (s, manifest, empty) in enumerate((("sweep_cpu_task0001", None, 2),
                                              ("sweep_cpu_task0002", (2, 2), 1),
                                              ("sweep_cpu_task0003", (2, 2), 0))):
        od = os.path.join(new, "campaign_cadex_rho1300v5", s)
        for j in range(2):
            p = os.path.join(od, "topo_00001", "mea_iter_%05d.npz" % j)
            touch(p, 0 if j < empty else 8)
        write_json(os.path.join(od, "mea_env.json"), {"host": "n%d" % i, "workers": 1,
                                                      "written": "2026-10-05T20:00:00+0200"})
        if manifest:
            write_json(os.path.join(od, "mea_manifest.json"),
                       {"total_done": manifest[0], "total_iters": manifest[1], "n_topos": 1})
        tasks.append({"campaign": "campaign_cadex_rho1300v5", "sweep": s, "out_dir": od, "n_iters": 2,
                      "n_topos": 1, "array_index": i, "status": "run"})
    write_json(os.path.join(here, "plan.json"), {"tasks": tasks, "out_root": new})
    log = ["[gate] tasks     : 3 (6 iterations planned), 4 worker(s)"]
    for s in ("sweep_cpu_task0001", "sweep_cpu_task0002"):
        log += ["[gate] FAIL campaign_cadex_rho1300v5/%s" % s,
                "[gate]      - topo_00001/mea_iter_00000.npz: unreadable: EOFError('No data left in file')"]
    log += ["[gate] read 3 mea_iter_*.npz file(s) across 3 task(s); 2 task(s) bad", "[job] sim_reextract_gate_exit=1"]
    os.makedirs(os.path.join(here, "out"))
    with open(os.path.join(here, "out", "sim_reextract_gate.log"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    with open(os.path.join(here, "out", "submissions.txt"), "w") as fh:
        fh.write("2026-10-05T16:00:00+02:00 array 777[].fake %s 3\n" % new)
    logs = os.path.join(root, "home")
    touch(os.path.join(logs, "c8_mea.e777.0"))
    with open(os.path.join(logs, "c8_mea.e777.0"), "w") as fh:
        fh.write("=>> PBS: job killed: node down\n")
    cwd = os.getcwd()
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.path.join(root, "no_qstat_here")     # qstat absent: reported, not fatal
    try:
        rc, out = run(c8_diag.main, ["--here", here, "--logs", logs])
    finally:
        os.environ["PATH"] = old_path
        os.chdir(cwd)
    lines = {x.split("|")[1].strip(): x for x in out.splitlines() if x.startswith(("  0 |", "  1 |"))}
    a = lines.get("campaign_cadex_rho1300v5/sweep_cpu_task0001", "")
    b = lines.get("campaign_cadex_rho1300v5/sweep_cpu_task0002", "")
    check("DG1 c8_diag: empty files counted; no manifest -> re-run by RESUME; complete manifest with an "
          "empty file -> KEPT as done by RESUME",
          "2 EMPTY" in a and "re-run by RESUME" in a and "1 EMPTY" in b and "KEPT as done by RESUME" in b, out)
    check("DG2 c8_diag: no qstat is reported, the job log's PBS line is read, and it ends '== D1 done'",
          rc == 0 and "qstat -xft 777[].fake failed" in out and "PBS: job killed: node down" in out
          and out.rstrip().endswith("== D1 done"), out)


def main():
    base = tempfile.mkdtemp(prefix="c8_tools_")
    try:
        cleanup_checks(base)
        diag_checks(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    n_bad = sum(1 for _, ok in RESULTS if not ok)
    print("ALL %d CHECKS PASSED" % len(RESULTS) if not n_bad else "FAILED %d of %d" % (n_bad, len(RESULTS)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
