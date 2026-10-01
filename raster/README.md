# raster/ -- spike rasters of the real cohort, control vs pathological

Raster plots of the real DUP15HD cohort (the 35 3Brain wells of the cohort
manifest, `DATA_C` / `DATA_P`): one raster per well, a mark per spike, the
active electrodes on the y axis, time on the x axis, a time window you choose
on the command line or with a slider. Static PNG/PDF figures and an
interactive viewer share one drawing code path.

Decisions it implements (`claude/SBI_decisions_and_ideas_log.md`):

| ID | what | in the code |
|---|---|---|
| D-026 | 3 control + 3 pathological wells: per class and batch folder, the well with the median `n_active` (lower median for an even count, ties by culture id), read from each well's extraction record | `raster_data.select_typical_wells`; `--wells` overrides |
| D-027 | rows = active electrodes (MFR over the whole recording >= `mfr_threshold`, 0.1 Hz), ascending linear index = grid row-major order, grid row 0 at the top | `raster_data.raster_view` |
| D-028 | the window is chosen with `--window T0 T1` (static figures) and with an interactive viewer whose Save writes the same figure | `run_raster_plots.py plot / view`, `raster_viewer.py` |
| D-029 | no rate panel under the rasters | `raster_plot.py` draws one raster axes per well and nothing else |

## Where the numbers come from

1. **The cohort manifest of record**, `Deep_bio/extracted_v2/cohort_manifest.json`,
   checked against its `.sha256` sidecar (`cohort_manifest.read_manifest`; a
   manifest without its sidecar is refused). It lists the wells and gives
   `fs_raw` (10110.09 Hz), `index_base` (1), `grid_width` (48) and
   `mfr_threshold` (0.1 Hz) -- D-002: the manifest is the preprocessing
   contract. Nothing here hard-codes those numbers.
2. **Each well's extraction record**, `traces_meta.json` in its `extracted_v2`
   folder: the raw folder it was extracted from (`source_folder`),
   `n_present`, `n_samples_raw`, and `discarded` (present electrodes with MFR
   below the threshold). `n_active = n_present - len(discarded)` is what D-026
   ranks by -- no raw data is read to choose the wells.
3. **The raw rasters**, `ptrain_<k>.mat` in the well's raw folder, read by the
   extractor's own `load_ptrain_folder` (scipy `loadmat`, variable `ptrain`,
   binary `(n_samples, 1)` raster), grid-checked by `validate_grid`, MFR by
   `mean_firing_rates` -- the very functions the Stage D extraction used. A
   spike at raw sample `s` (0-based) is at `s / fs_raw` seconds.

Before a well's cache is written it is **checked against its extraction
record**: the same raster length, the same number of present electrodes, and
the same sub-threshold set, element for element. A well that fails is
refused (`REFUSED` in the log, exit status 1) and never plotted. So the rows
of every raster are exactly the extractor's valid set V for that well.

## Quick start on davinci

Code route: commit on the laptop, push, pull on davinci (HPC_PATHS sec. L).
On davinci, every command below runs from `~/repos/Sbi-extractor/raster`
with `sbi_export` active.

```bash
cd ~/repos/Sbi-extractor && module load proxy && git pull && cd raster && conda activate sbi_export
python3 smoke_test_raster.py                      # ~10 s; must end with: [smoke] 12/12 checks passed
python3 run_raster_plots.py select --dry-run      # seconds; shows every batch folder and the 6 chosen wells
python3 run_raster_plots.py select                # writes ~/raster_plots/wells.tsv
mkdir -p out && qsub run_raster_plots.pbs         # caches the 6 wells + the figure for [0, 60) s
```

When the job has finished (`raster/out/raster_plots.log`):

```bash
python3 run_raster_plots.py plot --window 300 330 --window 600 605 --per-well
python3 run_raster_plots.py view                  # X11 needed; see "The viewer"
```

## Pass conditions -- lines that must APPEAR

| step | line |
|---|---|
| smoke test | `[smoke] 12/12 checks passed` |
| `select` | `[raster] selected 6 well(s) -> /davinci-1/home/ldellamea/raster_plots/wells.tsv` |
| job | `[raster] cache: 6/6 well(s) ok`, then `[raster] plot: wrote 3 file(s) for 1 window(s)`, then `[job] raster_exit=0` |
| `plot` | `[raster] plot: wrote N file(s) for M window(s)` |

`cache` prints one line per well, `cache built` (or `cache kept`), with its
present / active electrode counts and spike count. For the cohort of record
`present` should equal the well's `n_present` in the manifest (1724-2299).

## Commands

All commands take `--out-root DIR` (default `~/raster_plots`).

| command | what | reads | time |
|---|---|---|---|
| `select [--wells C ...] [--dry-run]` | the wells (D-026, or named), printed per batch folder with `n_active`; writes `wells.tsv` | manifest + records | seconds |
| `cache [--workers N] [--force] [--reselect]` | raw rasters -> `cache/<culture>.npz`, checked against the records; a cache built from the same raw folder and manifest is kept | manifest + records + raw | ~2 min per well per worker in a sandbox test; run it as the job |
| `plot [--window T0 T1 ...] [--per-well] [--formats png pdf svg] [--dpi] [--width] [--panel-height]` | static figures | `wells.tsv` + caches | seconds per window |
| `view [--window T0 T1]` | the viewer, starting at the first window | `wells.tsv` + caches | -- |
| `all` | `cache` (selecting first if there is no `wells.tsv`) + `plot`; what the job runs | everything | the job |

`--manifest PATH` (select, cache, all) points at another manifest; its
`.sha256` must sit beside it. `--wells` names wells by culture id, e.g.
`--wells DATA_C_Batch3__ptrain_A1 DATA_P_Batch4__ptrain_A2`; the figures'
JSON then records "named with --wells" instead of D-026.

Exit status: 0 ok; 1 a well failed or was refused; 2 bad input -- manifest
missing or without sidecar, a window outside the recording, a `wells.tsv`
written from another manifest, caches built from another manifest than
`wells.tsv` names, no display for `view`.

## Outputs

```
~/raster_plots/
  wells.tsv                                   the selection (+ manifest path and sha256 as # lines)
  cache/<culture>.npz                         every spike of the well: sample, electrode (sorted by time),
                                              present, n_spikes, active_mask, preprocessing, provenance
  figures/raster_compare_t0000.0-0060.0s.png  the comparison: classes as columns, wells as rows
  figures/raster_compare_t0000.0-0060.0s.pdf  same, vector text, rasterized marks
  figures/raster_compare_t0000.0-0060.0s.json window, rules, decisions, wells, counts, manifest sha256,
                                              marker opacity
  figures/per_well/<culture>_t....s.{png,pdf,json}   with --per-well
```

A cache is ~15 MB per well (sandbox test: 4.8 M spikes, 2284 electrodes).
`plot` and `view` read only `wells.tsv` and `cache/`: copy the folder to any
machine with numpy and matplotlib (scipy is not needed) to make figures or
browse there.

## How the figure reads

* **Rows** (D-027): the active electrodes, in ascending linear index. With
  the extractor's mapping `row = (k - index_base) // 48`, `col = (k -
  index_base) % 48`, that is grid row-major order; the y ticks mark where grid
  rows 0, 12, 24, 36 begin. Whether grid row 0 is the physical top of the
  chip is the extractor's own open orientation question
  (`extractor/channel_subset_viz.py`, ORIENTATION FLAG); the labels say "grid
  row", not "chip row", for that reason.
* **Window** (D-028): half-open, `[T0, T1)` seconds from the start of the
  recording; `T1` may equal the recording length (1200 s for the cohort).
  Default: the first 60 s (or the whole recording if shorter).
* **Marks**: one per spike. In a 2000-electrode panel a row is thinner than a
  pixel, so a spike is a one-pixel square; with fewer rows it becomes a
  vertical tick. Opacity: one value per figure, `alpha = min(1, max(0.05,
  1 / lambda_max))`, `lambda` the mean number of spikes per pixel cell of a
  panel. Windows up to about a minute stay fully opaque; long windows (the
  whole 20 min) would otherwise turn every panel into a solid block, so marks
  become translucent and stretches with more spikes (bursts, busy electrodes)
  stay darker. Because the value is shared across the figure, a denser well
  still looks denser than a sparser one. The JSON records the value.
* **Colour** carries the class only (control `#2a78d6`, pathological
  `#eb6834`, a pair checked for colour-vision deficiency); text is ink, and
  each column is named in a header.
* **Counts** under each well's name: active / present electrodes, spikes in
  the window.

Memory: a 60 s window of six wells is a few hundred MB; the whole recording
draws ~5 M spikes per well and peaked at ~2.6 GB in the sandbox test --
fine on a node, heavy for a shared login node.

## The viewer

`python3 run_raster_plots.py view [--window T0 T1]` opens the comparison
with controls underneath:

| control | effect |
|---|---|
| slider | drag a handle or the band; the panels follow 0.25 s after you stop |
| `t0` / `t1` boxes | type seconds, press Enter |
| left / right | pan by half a window |
| up / down (also `+` / `-`) | zoom in / out by 2 about the centre |
| home | back to the first window |
| toolbar zoom / pan | the window follows the x axis, spikes included |
| Save PNG/PDF | writes `figures/raster_compare_<window>.*` through the same function as `plot` |

It needs a display. On davinci: an SSH session with X11 forwarding (MobaXterm
does it by default; `echo $DISPLAY` must print something like
`localhost:10.0`). If matplotlib finds no GUI toolkit in `sbi_export`, try
`MPLBACKEND=QtAgg python3 run_raster_plots.py view`. Without a display the
command refuses (exit 2) and names `plot` instead. On a laptop: copy
`~/raster_plots` (wells.tsv + cache/) and this folder, then run `view
--out-root <copy>`.

## The smoke test

`smoke_test_raster.py` builds a 12-well cohort in a temporary folder, laid
out like davinci's (raw `DATA_C/Batch3/ptrain_*/ptrain_<k>.mat`,
`extracted_v2/<class>/<batch>/<well>/traces_meta.json`, a manifest with its
sidecar), with every spike planted, and checks the tool against the planted
spikes, never against its own output:

| check | what |
|---|---|
| R1 | the fixture manifest passes `read_manifest` |
| R2 | `load_cohort` reads class / batch / `n_active`; another layout, another `fs_raw`, a missing sidecar are refused |
| R3 | D-026 by hand (lower median, ties by culture, batch folders in raw-path order); `--wells`; unknown names refused |
| R4 | spike table == planted spikes; parity with `load_ptrain_folder`, `mean_firing_rates`, `partition_subregions`; 2 spikes in 20 s = 0.1 Hz counts as active |
| R5 | wrong raster length, `n_present`, discarded set: refused |
| R6 | cache round trip; stale manifest digest detected |
| R7 | `[t0, t1)` edges, the last sample, five bad windows refused |
| R8 | rows = active electrodes in linear order, every spike at its row, grid-row ticks |
| R9 | every panel draws exactly its planted spikes; one raster axes per well (D-029); PNG/PDF/JSON; opacity 1 when sparse, one shared value when dense |
| R10 | the CLI end to end, keep / `--force`, a tampered record refused (exit 1), exit 2 cases |
| R11 | the viewer's controls without a display; Save gives the same PNG bytes as `plot` |
| R12 | every file here ASCII and LF-only |

```bash
python3 smoke_test_raster.py                 # all
python3 smoke_test_raster.py --only R7 R9    # some (their prerequisites run too)
python3 smoke_test_raster.py --keep -v       # keep the fixture, print each CLI call's tail
```

It imports `cohort_manifest`, which needs the DSN tree (`../env.sh`:
`SBI_HPC_DIR`, default `artifacts/sbi_hpc`), as `select` and `cache` do.

## Troubleshooting

| message | cause | fix |
|---|---|---|
| `DSNTreeMissing` / "the DSN tree is <SBI_HPC_DIR>/dsn" | `cohort_manifest` cannot import `cohort.py` | `source ../env.sh`; `python3 ../dsn_tree.py` must print `resolves : yes` |
| `cohort manifest not found` | wrong `--manifest` | the default is `~/Deep Summary Network/Deep_bio/extracted_v2/cohort_manifest.json` |
| `has no .sha256 sidecar` | a copy without its sidecar | copy both files |
| `REFUSED -- its raw folder ... is not what its extraction recorded` | the raw folder changed since extraction, or another `fs_raw` | look at that folder before plotting it; nothing is written for it |
| `written from another manifest` | `wells.tsv` older than the manifest | `select` again |
| `the viewer needs a display` | no X11 | MobaXterm X11 forwarding, or `plot` |

Pure ASCII, LF only (hpc-python-compat).
