"""
raster_plot.py -- the drawing layer of the real-cohort raster plots.

Draws RasterView objects (raster_data.raster_view) onto matplotlib axes. It
reads no file and computes no data: what to draw is decided upstream, how to
draw it is decided here. It uses matplotlib's object API (Figure, not
pyplot), so it works under any backend: the static command saves with Agg,
the viewer (raster_viewer.py) draws into a pyplot window through the same
build_comparison(), and its Save button calls figure_compare(), the very
function `plot` uses (decision D-028).

Look
----
* One raster per well: a mark per spike, x = time [s], y = row; the rows are
  the active electrodes in grid row-major order, grid row 0 at the TOP, as in
  the extractor's electrode map (D-027). No rate strip (D-029).
* Colour carries the class and nothing else: control and pathological take
  slots 1 and 2 of the dataviz reference palette (#2a78d6, #eb6834),
  validated for colour-vision deficiency on a light surface (dataviz
  validator, 2026-10-01: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1
  against the surface). Text is always ink, never the class colour; each
  column is named by a legend entry (a colour swatch beside an ink label), so
  colour is never the only cue.
* Recessive chrome: left and bottom spines only, hairline, light grey; no
  grid; time ticks on the bottom row.
* Mark size follows the row density: when a row is at least 2 pt tall the
  spike is a vertical tick 0.8 row tall; when rows are denser than that
  (thousands of electrodes in a panel) it is a filled square one row tall,
  never smaller than one device pixel. Raster artists are rasterized, so a
  PDF stays small and its text stays vector.
* Mark opacity follows the window length, ONE value per figure. With lambda
  the mean number of spikes per pixel cell of a panel, and lambda_max the
  largest over the figure's panels,
      alpha = min(1, max(ALPHA_MIN, 1 / lambda_max)).
  Short windows (lambda_max <= 1, e.g. 60 s of a 2000-electrode well at
  300 dpi) keep alpha = 1: every spike is an opaque mark. Long windows,
  where opaque marks would fill every cell and the panel would turn into a
  solid block, make each mark translucent; overlapping marks then build up
  ink, so denser stretches (bursts) stay darker. Being shared, alpha never
  rescales one well against another: a denser well still looks denser.

Pure ASCII, LF only (hpc-python-compat): the two non-ASCII glyphs drawn are
built with chr().
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

__all__ = [
    "CLASS_COLORS", "INK", "ALPHA_MIN", "ALPHA_RULE", "class_color", "marker_style",
    "shared_alpha", "fmt_seconds", "ComparisonLayout", "comparison_size",
    "build_comparison", "figure_compare", "figure_single", "save_figure",
]

# dataviz reference palette, light mode: categorical slots and the ink.
CLASS_COLORS = {"control": "#2a78d6", "pathological": "#eb6834"}
_SPARE_COLORS = ("#1baf7a", "#4a3aa7")          # slots 3 and 7, for other names
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "axis": "#c3c2b7", "surface": "#ffffff"}
GE = chr(0x2265)                                # greater-or-equal sign
DOT = chr(0x00B7)                               # middle dot
ALPHA_MIN = 0.05
ALPHA_RULE = ("alpha = min(1, max(%g, 1 / lambda_max)); lambda = spikes per pixel "
              "cell of a panel; one alpha per figure" % ALPHA_MIN)

# Layout, in inches (manual GridSpec, so every axes size is known before
# drawing and the mark size can be derived from it).
_LEFT, _RIGHT = 0.62, 0.16
_TOP_TITLE, _TOP_HEADER = 0.30, 0.62     # suptitle band; column header + titles
_ROW_GAP, _COL_GAP, _BOTTOM = 0.46, 0.52, 0.50
_HEADER_LIFT = 0.38                       # column header above the top axes


def class_color(class_name: str, classes: Sequence[str]) -> str:
    """The class's colour: control/pathological fixed; any other class name
    takes the spare slots in the order the classes are listed."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    others = [c for c in classes if c not in CLASS_COLORS]
    i = others.index(class_name) if class_name in others else 0
    return _SPARE_COLORS[i % len(_SPARE_COLORS)]


def fmt_seconds(x: float) -> str:
    """1200 -> '1200', 12.5 -> '12.5', 0.125 -> '0.125' (ms resolution)."""
    s = ("%.3f" % float(x)).rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def marker_style(axes_height_pt: float, n_rows: int, dpi: float) -> Dict[str, object]:
    """Line2D marker keywords for one panel (see "Look" in the module note)."""
    px_pt = 72.0 / float(dpi)                         # one device pixel, in points
    row_pt = float(axes_height_pt) / max(int(n_rows), 1)
    if row_pt >= 2.0:
        return {"marker": "|", "markersize": 0.8 * row_pt,
                "markeredgewidth": max(min(0.12 * row_pt, 0.9), px_pt)}
    return {"marker": "s", "markersize": max(row_pt, px_pt),
            "markeredgewidth": 0.0}


def _cells(n_rows: int, w_px: float, h_px: float) -> float:
    """Pixel cells a panel's marks can fall in: rows denser than pixels share
    a pixel row, so the row count is capped by the height in pixels."""
    return max(min(float(max(n_rows, 1)), max(h_px, 1.0)) * max(w_px, 1.0), 1.0)


def shared_alpha(views, w_px: float, h_px: float) -> float:
    """alpha = min(1, max(ALPHA_MIN, 1 / lambda_max)) over the given panels."""
    lam = max((v.n_spikes / _cells(v.n_rows, w_px, h_px) for v in views), default=0.0)
    if lam <= 1.0:
        return 1.0
    return float(max(ALPHA_MIN, 1.0 / lam))


def _style_axes(ax, view, show_xlabels: bool, show_ylabel: bool) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", colors=INK["muted"], labelcolor=INK["secondary"],
                   labelsize=7, width=0.6, length=2.5)
    ax.set_ylim(max(view.n_rows, 1) - 0.5, -0.5)      # row 0 at the top
    ax.set_yticks(list(view.row_ticks))
    ax.set_yticklabels(list(view.row_tick_labels))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    if not show_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("time (s)", fontsize=8, color=INK["secondary"])
    if show_ylabel:
        ax.set_ylabel("grid row", fontsize=8, color=INK["secondary"])
    ax.set_facecolor(INK["surface"])


def _panel_titles(ax, view) -> Tuple[object, object]:
    """Two left-aligned lines above the panel: the culture id (ink), then
    its counts (secondary ink, smaller). Returns (name, counts) artists."""
    name = ax.annotate(view.culture, xy=(0.0, 1.0), xycoords="axes fraction",
                       xytext=(0, 12), textcoords="offset points",
                       ha="left", va="bottom", fontsize=8, color=INK["primary"])
    counts = ax.annotate(_count_text(view), xy=(0.0, 1.0),
                         xycoords="axes fraction", xytext=(0, 3),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=6.5, color=INK["secondary"])
    return name, counts


def _count_text(view) -> str:
    return "%s active / %s present  %s  %s spikes" % (
        format(view.n_rows, ","), format(view.n_present, ","), DOT,
        format(view.n_spikes, ","))


def _suptitle_text(t0: float, t1: float, mfr_threshold: Optional[float]) -> str:
    rule = ("active electrodes (MFR %s %g Hz)" % (GE, mfr_threshold)
            if mfr_threshold is not None else "active electrodes")
    return "Spike rasters, %s  |  window %s-%s s" % (rule, fmt_seconds(t0),
                                                     fmt_seconds(t1))


class ComparisonLayout:
    """The axes and artists of one comparison figure, so a window can be
    redrawn in place (the viewer) without rebuilding the figure."""

    def __init__(self, fig, axes, lines, titles, suptitle, classes,
                 mfr_threshold, panel_px: Tuple[float, float], alpha: float):
        self.fig = fig
        self.axes = axes                 # (class, i) -> Axes
        self.lines = lines               # (class, i) -> Line2D
        self.titles = titles             # (class, i) -> counts annotation
        self.suptitle = suptitle
        self.classes = list(classes)
        self.mfr_threshold = mfr_threshold
        self.panel_px = panel_px         # (width, height) of a panel in pixels
        self.alpha = alpha

    def first_axes(self):
        return next(iter(self.axes.values()))

    def update(self, views_by_class: Dict[str, list], t0: float, t1: float) -> None:
        """Replace every panel's spikes with those of a new window."""
        flat = [v for vs in views_by_class.values() for v in vs]
        self.alpha = shared_alpha(flat, *self.panel_px)
        for c, views in views_by_class.items():
            for i, v in enumerate(views):
                key = (c, i)
                if key not in self.lines:
                    raise KeyError("no panel for %r" % (key,))
                self.lines[key].set_data(v.t, v.row)
                self.lines[key].set_alpha(self.alpha)
                self.titles[key].set_text(_count_text(v))
        for ax in self.axes.values():
            ax.set_xlim(t0, t1)
        self.suptitle.set_text(_suptitle_text(t0, t1, self.mfr_threshold))


def comparison_size(n_panel_rows: int, width_in: float, panel_height_in: float,
                    extra_bottom_in: float = 0.0) -> Tuple[float, float]:
    h = (_TOP_TITLE + _TOP_HEADER + n_panel_rows * panel_height_in
         + max(n_panel_rows - 1, 0) * _ROW_GAP + _BOTTOM + extra_bottom_in)
    return float(width_in), float(h)


def build_comparison(fig, views_by_class: Dict[str, list], classes: Sequence[str],
                     mfr_threshold: Optional[float] = None,
                     panel_height_in: float = 1.9,
                     extra_bottom_in: float = 0.0) -> ComparisonLayout:
    """Draw one column per class and one panel per well into `fig`.

    fig must already have the size comparison_size() returns (the viewer
    adds extra_bottom_in for its widgets). All panels share the x axis, so
    zooming one zooms all.
    """
    cols = [c for c in classes if c in views_by_class]
    if not cols:
        raise ValueError("nothing to draw")
    n_rows = max(len(views_by_class[c]) for c in cols)
    W, H = fig.get_size_inches()
    panel_w_in = (W - _LEFT - _RIGHT - _COL_GAP * (len(cols) - 1)) / len(cols)
    gs = fig.add_gridspec(
        n_rows, len(cols),
        left=_LEFT / W, right=1.0 - _RIGHT / W,
        top=1.0 - (_TOP_TITLE + _TOP_HEADER) / H,
        bottom=(_BOTTOM + extra_bottom_in) / H,
        hspace=_ROW_GAP / panel_height_in,
        wspace=_COL_GAP / panel_w_in)
    panel_px = (panel_w_in * fig.dpi, panel_height_in * fig.dpi)
    alpha = shared_alpha([v for c in cols for v in views_by_class[c]], *panel_px)

    first = views_by_class[cols[0]][0]
    t0, t1 = first.t0, first.t1
    axes, lines, titles = {}, {}, {}
    share = None
    for j, c in enumerate(cols):
        color = class_color(c, classes)
        views = views_by_class[c]
        for i in range(n_rows):
            ax = fig.add_subplot(gs[i, j], sharex=share)
            if share is None:
                share = ax
            if i >= len(views):
                ax.set_visible(False)
                continue
            v = views[i]
            last = (i == len(views) - 1)
            _style_axes(ax, v, show_xlabels=last, show_ylabel=(j == 0))
            (line,) = ax.plot(v.t, v.row, linestyle="none", color=color, alpha=alpha,
                              rasterized=True, antialiased=False,
                              **marker_style(panel_height_in * 72.0, v.n_rows, fig.dpi))
            _, counts = _panel_titles(ax, v)
            axes[(c, i)], lines[(c, i)], titles[(c, i)] = ax, line, counts
        # column header: a legend entry, swatch in the class colour, ink label
        top_ax = axes.get((c, 0))
        if top_ax is not None:
            pos = top_ax.get_position()
            handle = Line2D([], [], linestyle="none", marker="s", markersize=7,
                            markerfacecolor=color, markeredgewidth=0)
            label = "%s  (%d well%s)" % (c.capitalize(), len(views),
                                         "" if len(views) == 1 else "s")
            fig.legend([handle], [label], loc="lower left",
                       bbox_to_anchor=(pos.x0 - 0.01, pos.y1 + _HEADER_LIFT / H),
                       frameon=False, fontsize=9, handlelength=0.8,
                       handletextpad=0.5, borderaxespad=0.0,
                       labelcolor=INK["primary"])
    share.set_xlim(t0, t1)
    sup = fig.text(_LEFT / W, 1.0 - 0.12 / H, _suptitle_text(t0, t1, mfr_threshold),
                   ha="left", va="top", fontsize=9.5, color=INK["primary"],
                   fontweight="semibold")
    return ComparisonLayout(fig, axes, lines, titles, sup, cols, mfr_threshold,
                            panel_px, alpha)


def figure_compare(views_by_class: Dict[str, list], classes: Sequence[str],
                   mfr_threshold: Optional[float] = None, width_in: float = 7.2,
                   panel_height_in: float = 1.9, dpi: float = 300.0,
                   return_layout: bool = False):
    """The comparison figure, classes as columns, wells as rows. With
    return_layout, returns (fig, layout) -- layout.alpha is the opacity used."""
    n_rows = max(len(v) for v in views_by_class.values())
    fig = Figure(figsize=comparison_size(n_rows, width_in, panel_height_in),
                 dpi=dpi, facecolor=INK["surface"])
    layout = build_comparison(fig, views_by_class, classes, mfr_threshold,
                              panel_height_in)
    return (fig, layout) if return_layout else fig


def figure_single(view, classes: Sequence[str], mfr_threshold: Optional[float] = None,
                  width_in: float = 7.2, panel_height_in: float = 2.6,
                  dpi: float = 300.0, return_layout: bool = False):
    """One well on its own, same look as a comparison panel."""
    return figure_compare({view.class_name: [view]}, classes, mfr_threshold,
                          width_in, panel_height_in, dpi, return_layout)


def save_figure(fig, stem: str, formats: Sequence[str] = ("png", "pdf"),
                sidecar: Optional[dict] = None) -> List[str]:
    """Write stem.<fmt> for every format (at the figure's dpi), and
    stem.json with the sidecar if one is given. Returns the paths."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    written = []
    for fmt in formats:
        path = "%s.%s" % (stem, fmt)
        fig.savefig(path, dpi=fig.dpi, facecolor=fig.get_facecolor())
        written.append(path)
    if sidecar is not None:
        doc = dict(sidecar)
        doc.setdefault("written_utc",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        doc["files"] = [os.path.basename(p) for p in written]
        path = stem + ".json"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        written.append(path)
    return written
