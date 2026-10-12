#!/usr/bin/env python3
"""Block 7.2 -- one array element of the Giulia simulated bank (B5.2-B5.3;
specs/SPEC.md, Block 7.2).

    python3 export_windows/bank_task.py --plan export_windows/giulia/bank_plan.json --task-index N

Exports task N of the plan: its shards into the plan's bank_root (the profile's,
~/banks/giulia_sim, D-046) and its marker tasks/task_<NNNN>.json. A task whose
marker is complete for this plan (same plan digest, every file's sha256
unchanged) is skipped, so the array can be resubmitted after a failure.

Printed last line: `TASK <N> OK` (exit 0) or `FAILED: <reason>` (exit 1).

HPC note (hpc-python-compat): pure ASCII, LF only; python 3.11 on the cluster.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bank_export as EXP               # noqa: E402
import bank_plan_load as LOAD           # noqa: E402
import bank_plan_model as MODEL         # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Export one task of the Giulia simulated bank.")
    p.add_argument("--plan", required=True, help="bank_plan.json written by bank_plan.py")
    p.add_argument("--task-index", type=int, required=True, help="the task's index in the plan")
    p.add_argument("--bank-root", default=None, help="default: the plan's profile bank_root")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    try:
        plan = EXP.load_plan(args.plan)
        n = len(plan["tasks"])
        if not 0 <= args.task_index < n:
            raise EXP.ExportError("--task-index %d is not a task of the plan (0-%d)"
                                  % (args.task_index, n - 1))
        bank_root = os.path.abspath(args.bank_root or plan["profile"]["bank_root"])
        t = plan["tasks"][args.task_index]
        print("[task] plan %s  sha256 %s" % (plan["_path"], plan["_sha256"][:16]))
        print("[task] %d: %s  (%d simulations, %d shard(s)) -> %s"
              % (args.task_index, t["name"], t["n_iters"], len(t["shards"]), bank_root))
        if EXP.marker_is_complete(plan, args.task_index, bank_root):
            print("[task] already complete for this plan: nothing to do")
            print("TASK %d OK" % args.task_index)
            return 0
        os.makedirs(bank_root, exist_ok=True)
        m = EXP.export_task(plan, args.task_index, bank_root)
        tot = m["totals"]
        print("[task] read %d, kept %d, dropped by the floor %d (%.1f%% kept), %.0f s"
              % (tot["read"], tot["kept"], tot["dropped_floor"],
                 100.0 * tot["kept"] / max(1, tot["read"]), time.time() - t0))
        print("TASK %d OK" % args.task_index)
        return 0
    except (EXP.ExportError, LOAD.LoadError, MODEL.PlanRefused, ValueError, OSError) as exc:
        print("FAILED: %s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
