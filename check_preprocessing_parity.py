#!/usr/bin/env python3
"""
check_preprocessing_parity.py -- does the SIMULATED arm get exactly the same
smoothing and downsampling as the REAL arm?

Run this on the login node BEFORE exporting anything. It writes nothing,
submits nothing, and needs no allocation.

WHAT IS BEING CHECKED, AND WHY IT MATTERS

Both arms turn spike times into an instantaneous firing rate by the same
three-step recipe: bin at Delta_t (the DOWNSAMPLING, from the raw
10110.09 Hz sample grid to 1/Delta_t Hz), convolve with a Gaussian of width
sigma_sm/Delta_t bins (the SMOOTHING), and divide by the number of pooled
electrodes n_e. The two arms reach that recipe through DIFFERENT code:

    real : channel_subset_extraction.subregion_ifr
             -> generate_burst_data.compute_ifr_trace  (histogram, filter,
                clip, float32)  then  / n_e, cast float32 AGAIN
    sim  : sim_observable.build_pooled_ifr
             -> pooled_spike_counts (histogram with an explicit domain clip),
                filter, clip, / n_e in float64, cast float32 ONCE

The encoder is a convolutional network and is NOT scale-invariant, so any
disagreement between these two paths -- in Delta_t, in sigma_sm, in n_e, or
in amplitude -- separates the two embedding clouds by arithmetic rather than
by biology, and the misspecification gate reports it as a decisive
rejection. That failure mode has already happened once in this project
(the per-electrode mean versus sum discrepancy, a factor of n_e = 9).

WHAT THIS SCRIPT ESTABLISHES, PER MODE

  config      The three declarations agree: the checkpoint's cohort block,
              the fs_ifr recorded in the extracted real archives, and the
              campaign's declared simtime. Reports the resulting window
              arithmetic on both arms.
  primitives  The two IFR implementations are run on the SAME synthetic
              spike trains and compared numerically. They are NOT expected
              to agree bit-for-bit -- see the note in report_primitives --
              so the agreement is measured, not asserted.
  data        Real and simulated windows as they will actually be embedded,
              compared on amplitude statistics. This is the check that would
              have caught the factor-of-9.
  all         All three, in that order (default).

USAGE (MobaXterm / login node)

    cd /davinci-1/home/ldellamea/repos/Sbi-extractor
    python3 check_preprocessing_parity.py --mode all \
        --checkpoint "$HOME/dsn_main/out/refit_mea_A_best/checkpoints/seed_0/best.pt" \
        --dsn_main_dir "$HOME/dsn_main" \
        --real_npz "/davinci-1/home/ldellamea/Deep Summary Network/Deep_bio/extracted/control/Batch4_SubBatch1/ptrain_A1/trace_subregion_00.npz" \
        --specs /path/to/specs_real.json \
        --mea_out /davinci-1/home/ldellamea/ANN/MEA_analysis/Outputs/campaign_cadex_rho1300v1/sweep_cpu_task0000 \
        --campaign /davinci-1/home/ldellamea/ANN/Phenomenological/Main/campaign_cadex_rho1300v1/sweep_cpu_task0000 \
        --trim_head_s 20 --n_sample 60

Every path with a space in it must be quoted. --checkpoint may be omitted if
--w_size / --gaussian_window / --window_s are given explicitly, which lets
the whole script run without torch.

Exit code 0 only if every check passed.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILURES: List[str] = []
NOTES: List[str] = []


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)


def verdict(ok: bool, label: str, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def note(msg: str) -> None:
    print("  [note] %s" % msg)
    NOTES.append(msg)


def summarise(x: np.ndarray) -> Dict[str, float]:
    """Amplitude statistics of a set of windows, pooled over all samples."""
    f = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "n": float(f.size),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "p50": float(np.percentile(f, 50)),
        "p99": float(np.percentile(f, 99)),
        "max": float(f.max()),
        "zero_frac": float((f <= 0.0).mean()),
    }


def print_stats(label: str, s: Dict[str, float]) -> None:
    print("    %-10s n=%-9d mean=%.5g  std=%.5g  p50=%.5g  p99=%.5g  "
          "max=%.5g  zeros=%.1f%%"
          % (label, int(s["n"]), s["mean"], s["std"], s["p50"], s["p99"],
             s["max"], 100.0 * s["zero_frac"]))


# --------------------------------------------------------------------------- #
# geometry, from wherever it is declared
# --------------------------------------------------------------------------- #
def read_checkpoint_geometry(ckpt_path: str) -> Dict[str, object]:
    """Delta_t, sigma_sm, T_win, E, C from the checkpoint's own config block.

    Read directly rather than through load_frozen_dsn so this works without
    the DSN repo importable and without building the model.
    """
    import torch  # noqa: E402  -- only this function needs it
    c = torch.load(ckpt_path, map_location="cpu")
    cfg = c.get("config", {}) or {}

    def dig(*keys):
        d = cfg
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    return {
        "window_s": dig("data", "window_s"),
        "eval_stride_s": dig("data", "eval_stride_s"),
        "n_channels": dig("data", "n_channels"),
        "w_size": dig("cohort", "w_size"),
        "gaussian_window": dig("cohort", "gaussian_window"),
        "embedding_size": dig("backbone", "embedding_size"),
        "l2_normalize": dig("backbone", "l2_normalize"),
    }


def n_windows(length: int, w: int, stride: int) -> int:
    """floor((L - W) / S) + 1 when L >= W, else 0. The dataset's own rule."""
    return 0 if length < w else (length - w) // stride + 1


# --------------------------------------------------------------------------- #
# mode: config
# --------------------------------------------------------------------------- #
def mode_config(args, geom: Dict[str, object]) -> None:
    banner("1. CONFIG -- do the three declarations agree?")

    dt = float(geom["w_size"])
    sm = float(geom["gaussian_window"])
    win_s = float(geom["window_s"])
    fs_ckpt = 1.0 / dt
    W = int(round(win_s / dt))

    print("  checkpoint : Delta_t=%.5g s  sigma_sm=%.5g s  T_win=%.5g s"
          % (dt, sm, win_s))
    print("               -> fs_ifr=%.5g Hz   W=%d samples   sigma=%.4g bins"
          % (fs_ckpt, W, sm / dt))
    if geom.get("embedding_size") is not None:
        print("               E=%s  in_channels=%s  l2_normalize=%s"
              % (geom["embedding_size"], geom["n_channels"],
                 geom["l2_normalize"]))

    ok = verdict(sm / dt >= 1.0, "smoothing is at least one bin wide",
                 "sigma = %.4g bins" % (sm / dt))
    if geom.get("eval_stride_s") is not None:
        s_s = float(geom["eval_stride_s"])
        verdict(abs(s_s - win_s) < 1e-9,
                "eval stride equals the window (disjoint windows)",
                "stride=%.5g s, window=%.5g s" % (s_s, win_s))
    if geom.get("n_channels") is not None:
        verdict(int(geom["n_channels"]) == 1,
                "checkpoint expects single-channel input",
                "n_channels=%s" % geom["n_channels"])

    # ---- the real archives ------------------------------------------------
    paths = []
    if args.specs:
        with open(args.specs, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        paths = [e["path"] for e in entries]
    elif args.real_npz:
        paths = [args.real_npz]

    if paths:
        print("")
        print("  real archives (%d listed, checking up to %d):"
              % (len(paths), args.n_sample))
        seen_fs, seen_T, seen_K = set(), set(), set()
        n_checked = 0
        for p in paths[:args.n_sample]:
            if not os.path.isfile(p):
                continue
            with np.load(p, allow_pickle=False) as d:
                key = "ifr_trace" if "ifr_trace" in d.files else "X"
                seen_K.add(int(np.asarray(d[key]).reshape(-1).shape[0]))
                if "fs_ifr" in d.files:
                    seen_fs.add(round(float(d["fs_ifr"]), 6))
                if "T_rec" in d.files:
                    seen_T.add(round(float(d["T_rec"]), 3))
            n_checked += 1
        print("    checked %d file(s): fs_ifr=%s  T_rec=%s  K=%s"
              % (n_checked, sorted(seen_fs), sorted(seen_T), sorted(seen_K)))

        verdict(len(seen_fs) == 1, "one fs_ifr across the real cohort",
                "values: %s" % sorted(seen_fs))
        if len(seen_fs) == 1:
            fs_real = list(seen_fs)[0]
            ok = verdict(abs(fs_real - fs_ckpt) < 1e-6,
                         "real fs_ifr matches the checkpoint's 1/Delta_t",
                         "real=%.6g Hz, checkpoint=%.6g Hz" % (fs_real, fs_ckpt))
            if not ok:
                note("A mismatch here is NOT cosmetic: W is a count of "
                     "SAMPLES, so at fs_real=%.6g Hz a window of W=%d samples "
                     "spans %.4g s of biology, while the simulated arm spans "
                     "%.4g s. Re-extract with --w-size %.6g."
                     % (fs_real, W, W / fs_real, win_s, dt))
        if len(seen_K) == 1 and len(seen_T) == 1:
            K = list(seen_K)[0]
            nw = n_windows(K, W, W)
            print("    real windowing: K=%d -> %d window(s) of %d samples; "
                  "%d sample(s) unused at the tail"
                  % (K, nw, W, K - nw * W))
            verdict(nw >= 1, "each real trace yields at least one window",
                    "%d window(s)" % nw)
        note("The extractor does NOT record n_e in the archive. It is fixed "
             "at electrodes_per_subset (9) because partition_subregions "
             "raises rather than under-filling a subregion, so every real "
             "trace is a mean over exactly 9 electrodes -- the same n_e as "
             "the simulated 3x3 probe. Verified in source, not in the file.")

    # ---- the simulated side ----------------------------------------------
    if args.campaign:
        jpath = os.path.join(args.campaign, "job_args.json")
        print("")
        if os.path.isfile(jpath):
            with open(jpath, "r", encoding="utf-8") as fh:
                ja = json.load(fh)
            T_sim = ja.get("simtime")
            print("  campaign job_args.json: simtime=%s" % (T_sim,))
            if T_sim is not None:
                T_sim = float(T_sim)
                K_sim = int(T_sim / dt)
                n_trim = int(round(float(args.trim_head_s) / dt))
                K_kept = K_sim - n_trim
                nw = n_windows(K_kept, W, W)
                print("    sim windowing: K=%d, trim %d bin(s) (%.4g s), "
                      "keeps %d -> %d window(s); %d sample(s) unused"
                      % (K_sim, n_trim, args.trim_head_s, K_kept, nw,
                         K_kept - nw * W))
                verdict(nw == 1,
                        "each simulation yields exactly one window",
                        "%d window(s)" % nw)
                verdict(K_kept >= W,
                        "usable duration is at least the DSN window",
                        "%.4g s usable vs %.4g s window"
                        % (K_kept * dt, win_s))
        else:
            note("no job_args.json under %s; the declared simtime could not "
                 "be checked. Do NOT fall back on the npz 'simtime' field: "
                 "it is ceil(last detected spike)." % args.campaign)


# --------------------------------------------------------------------------- #
# mode: primitives
# --------------------------------------------------------------------------- #
def report_primitives(args, geom: Dict[str, object]) -> None:
    banner("2. PRIMITIVES -- do the two IFR implementations agree?")

    dt = float(geom["w_size"])
    sm = float(geom["gaussian_window"])
    n_e = int(args.n_electrodes)
    T = float(args.primitive_T)

    try:
        from sim_observable import build_pooled_ifr
    except Exception as exc:                       # noqa: BLE001
        verdict(False, "import sim_observable (simulated arm)", repr(exc))
        return

    dsn_main = args.dsn_main_dir
    if not dsn_main:
        note("--dsn_main_dir not given; skipping the real-arm import. Pass it "
             "to compare the two implementations directly.")
        return
    for sub in ("", os.path.join("hpc", "MultiChannel")):
        p = os.path.join(dsn_main, sub) if sub else dsn_main
        if os.path.isdir(p):
            sys.path.insert(0, p)
    try:
        from generate_burst_data import CONTROL_PARAMS, compute_ifr_trace
        from dataclasses import replace as dc_replace
    except Exception as exc:                       # noqa: BLE001
        verdict(False, "import generate_burst_data (real arm)", repr(exc))
        note("Add the directory containing generate_burst_data.py to "
             "--dsn_main_dir. Both <Main> and <Main>/hpc/MultiChannel hold a "
             "copy; they must be the same file for this check to mean "
             "anything -- diff them if unsure.")
        return

    rng = np.random.default_rng(0)
    trains = []
    for _ in range(n_e):
        n = rng.poisson(4.0 * T)
        trains.append(np.sort(rng.uniform(0.0, T, size=n)))

    # real path: compute_ifr_trace then divide, casting float32 TWICE
    params = dc_replace(CONTROL_PARAMS, duration_s=T, w_size=dt,
                        gaussian_window=sm)
    ifr_real, fs_real = compute_ifr_trace(trains, params)
    ifr_real = (ifr_real / float(n_e)).astype(np.float32)

    # simulated path: divide in float64, cast float32 ONCE
    ifr_sim = build_pooled_ifr(trains, n_electrodes=n_e, T=T, dt=dt,
                               sigma_sm=sm)

    verdict(ifr_real.shape == ifr_sim.shape, "same number of bins",
            "real %r vs sim %r" % (ifr_real.shape, ifr_sim.shape))
    verdict(abs(fs_real - 1.0 / dt) < 1e-9, "same fs_ifr",
            "real %.6g Hz" % fs_real)
    if ifr_real.shape != ifr_sim.shape:
        return

    a = ifr_real.astype(np.float64)
    b = ifr_sim.astype(np.float64)
    d_abs = float(np.abs(a - b).max())
    scale = float(np.abs(b).max()) or 1.0
    d_rel = d_abs / scale
    n_diff = int((ifr_real != ifr_sim).sum())

    print("    max |real - sim|      : %.6g" % d_abs)
    print("    relative to peak      : %.3g  (peak %.6g)" % (d_rel, scale))
    print("    bins differing at all : %d / %d" % (n_diff, ifr_real.size))

    # The two are NOT expected to be bit-identical, for two reasons found in
    # the source: (i) pooled_spike_counts clips the domain to [0, K*Delta_t)
    # while compute_ifr_trace lets np.histogram close its last bin on the
    # right, so a spike landing exactly on the final edge is counted by one
    # and not the other; (ii) the real path rounds to float32 BEFORE
    # dividing by n_e and again after, the simulated path divides in float64
    # and rounds once. Both are last-ulp effects. Asserting exact equality
    # here would send you chasing a phantom.
    tol = 1e-6
    ok = verdict(d_rel < tol,
                 "the two implementations agree to float32 rounding",
                 "relative difference %.3g < %.3g" % (d_rel, tol))
    if ok and n_diff:
        note("%d bin(s) differ in the last float32 ulp only. Expected: the "
             "real path casts to float32 before dividing by n_e and again "
             "after; the simulated path divides in float64 and casts once."
             % n_diff)

    # The boundary asymmetry, exhibited rather than described.
    K = int(T / dt)
    edge = np.array([K * dt])
    only_edge = [edge] + [np.empty(0) for _ in range(n_e - 1)]
    p_edge = dc_replace(CONTROL_PARAMS, duration_s=T, w_size=dt,
                        gaussian_window=sm)
    e_real, _ = compute_ifr_trace(only_edge, p_edge)
    e_sim = build_pooled_ifr(only_edge, n_electrodes=n_e, T=T, dt=dt,
                             sigma_sm=sm)
    mass_real = float(np.asarray(e_real, dtype=np.float64).sum())
    mass_sim = float(np.asarray(e_sim, dtype=np.float64).sum()) * n_e
    note("right-edge convention: a spike at exactly t = K*Delta_t contributes "
         "%.4g to the real path and %.4g to the simulated one. One spike per "
         "trace at worst; it matters only if your spike times are quantised "
         "onto the bin grid." % (mass_real, mass_sim))


# --------------------------------------------------------------------------- #
# mode: data
# --------------------------------------------------------------------------- #
def mode_data(args, geom: Dict[str, object]) -> None:
    banner("3. DATA -- do the windows that will be embedded have the same scale?")

    dt = float(geom["w_size"])
    sm = float(geom["gaussian_window"])
    win_s = float(geom["window_s"])
    W = int(round(win_s / dt))
    n_e = int(args.n_electrodes)

    # ---- real -------------------------------------------------------------
    real_paths: List[str] = []
    if args.specs:
        with open(args.specs, "r", encoding="utf-8") as fh:
            real_paths = [e["path"] for e in json.load(fh)]
    elif args.real_npz:
        real_paths = [args.real_npz]
    real_paths = [p for p in real_paths if os.path.isfile(p)][:args.n_sample]

    real_win: List[np.ndarray] = []
    for p in real_paths:
        with np.load(p, allow_pickle=False) as d:
            key = "ifr_trace" if "ifr_trace" in d.files else "X"
            x = np.asarray(d[key]).reshape(-1)
        nw = n_windows(x.shape[0], W, W)
        for i in range(nw):
            real_win.append(x[i * W:(i + 1) * W])
    if not real_win:
        note("no real windows built; pass --specs or --real_npz")
        return
    s_real = summarise(np.concatenate(real_win))

    # ---- simulated --------------------------------------------------------
    if not args.mea_out:
        note("no --mea_out given; the simulated side of this comparison was "
             "skipped, so the scale check did NOT run.")
        print_stats("real", s_real)
        return

    try:
        from sim_observable import build_pooled_ifr
    except Exception as exc:                       # noqa: BLE001
        verdict(False, "import sim_observable", repr(exc))
        return

    T_sim = args.simtime
    if T_sim is None and args.campaign:
        jpath = os.path.join(args.campaign, "job_args.json")
        if os.path.isfile(jpath):
            with open(jpath, "r", encoding="utf-8") as fh:
                T_sim = json.load(fh).get("simtime")
    if T_sim is None:
        note("no simtime available (pass --simtime or --campaign); the "
             "simulated side was skipped.")
        print_stats("real", s_real)
        return
    T_sim = float(T_sim)
    n_trim = int(round(float(args.trim_head_s) / dt))

    mea = sorted(glob.glob(os.path.join(args.mea_out, "topo_*",
                                        "mea_iter_*.npz")))[:args.n_sample]
    if not mea:
        verdict(False, "found mea_iter_*.npz under --mea_out", args.mea_out)
        return

    sim_win: List[np.ndarray] = []
    n_short = 0
    for p in mea:
        with np.load(p, allow_pickle=False) as m:
            if "det_t" not in m.files or "det_ch" not in m.files:
                continue
            det_t = np.asarray(m["det_t"], dtype=np.float64)
            det_ch = np.asarray(m["det_ch"], dtype=np.int64)
        per_e = [det_t[det_ch == e] for e in range(n_e)]
        x = build_pooled_ifr(per_e, n_electrodes=n_e, T=T_sim, dt=dt,
                             sigma_sm=sm)
        x = x[n_trim:]
        nw = n_windows(x.shape[0], W, W)
        if nw == 0:
            n_short += 1
            continue
        for i in range(nw):
            sim_win.append(x[i * W:(i + 1) * W])
    if not sim_win:
        verdict(False, "built at least one simulated window",
                "%d trace(s) too short" % n_short)
        return
    s_sim = summarise(np.concatenate(sim_win))

    print("  windows built: real=%d (from %d traces), sim=%d (from %d files, "
          "%d too short)" % (len(real_win), len(real_paths), len(sim_win),
                             len(mea), n_short))
    print("")
    print_stats("real", s_real)
    print_stats("sim", s_sim)

    verdict(n_short == 0, "no simulated trace was dropped for being short",
            "%d dropped" % n_short)

    for key in ("mean", "p99"):
        r = s_sim[key] / s_real[key] if s_real[key] > 0 else float("inf")
        print("    ratio sim/real on %-4s : %.4g" % (key, r))
        if not np.isfinite(r) or r <= 0:
            verdict(False, "sim/real %s ratio is finite and positive" % key,
                    "%r" % r)
            continue
        # A factor near n_e in either direction is the pooling-convention bug
        # (mean versus sum over the 9 electrodes) reappearing.
        near_ne = abs(r - n_e) < 0.25 * n_e or abs(r - 1.0 / n_e) < 0.25 / n_e
        verdict(not near_ne,
                "sim/real %s ratio is not a factor of n_e" % key,
                "ratio %.4g against n_e = %d" % (r, n_e))
        verdict(0.1 < r < 10.0,
                "sim/real %s ratio is within one order of magnitude" % key,
                "ratio %.4g" % r)

    note("A ratio inside an order of magnitude is NOT evidence that the "
         "simulator is right -- the two clouds can still be far apart in "
         "summary space. It only rules out a units or pooling error large "
         "enough to make the gate's answer meaningless. The gate itself is "
         "what answers the scientific question.")


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check that the simulated IFR preprocessing matches the "
                    "real one.")
    ap.add_argument("--mode", default="all",
                    choices=("all", "config", "primitives", "data"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dsn_main_dir", default=os.environ.get("DSN_MAIN_DIR"))
    ap.add_argument("--specs", default=None,
                    help="the real specs JSON; preferred over --real_npz")
    ap.add_argument("--real_npz", default=None,
                    help="a single trace_subregion_XX.npz")
    ap.add_argument("--mea_out", default=None,
                    help="a sweep task dir holding topo_*/mea_iter_*.npz")
    ap.add_argument("--campaign", default=None,
                    help="the matching sweep dir, for job_args.json")
    ap.add_argument("--simtime", type=float, default=None,
                    help="override T [s]; normally read from job_args.json")
    ap.add_argument("--trim_head_s", type=float, default=0.0)
    ap.add_argument("--n_electrodes", type=int, default=9)
    ap.add_argument("--n_sample", type=int, default=40,
                    help="how many files to sample per arm")
    ap.add_argument("--primitive_T", type=float, default=200.0,
                    help="duration of the synthetic trains in --mode primitives")
    # explicit geometry, so the script can run without torch
    ap.add_argument("--w_size", type=float, default=None)
    ap.add_argument("--gaussian_window", type=float, default=None)
    ap.add_argument("--window_s", type=float, default=None)
    args = ap.parse_args(argv)

    del FAILURES[:]
    del NOTES[:]

    geom: Dict[str, object]
    if args.checkpoint:
        geom = read_checkpoint_geometry(args.checkpoint)
        src = "checkpoint %s" % args.checkpoint
    else:
        if None in (args.w_size, args.gaussian_window, args.window_s):
            ap.error("give --checkpoint, or all of --w_size, "
                     "--gaussian_window and --window_s")
        geom = {"w_size": args.w_size, "gaussian_window": args.gaussian_window,
                "window_s": args.window_s}
        src = "command line"
    for k in ("w_size", "gaussian_window", "window_s"):
        if geom.get(k) is None:
            print("FATAL: %s does not declare %s" % (src, k))
            return 2
    if args.w_size is not None:
        geom["w_size"] = args.w_size
    if args.gaussian_window is not None:
        geom["gaussian_window"] = args.gaussian_window
    if args.window_s is not None:
        geom["window_s"] = args.window_s

    print("geometry source: %s" % src)

    if args.mode in ("all", "config"):
        mode_config(args, geom)
    if args.mode in ("all", "primitives"):
        report_primitives(args, geom)
    if args.mode in ("all", "data"):
        mode_data(args, geom)

    banner("VERDICT")
    if FAILURES:
        print("  %d check(s) FAILED:" % len(FAILURES))
        for f in FAILURES:
            print("    - %s" % f)
        print("")
        print("  Do NOT export until these are resolved. A preprocessing")
        print("  mismatch separates the two embedding clouds by arithmetic,")
        print("  and the gate will report it as decisive misspecification.")
        return 1
    print("  every check passed (%d note(s) above are informational)"
          % len(NOTES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
