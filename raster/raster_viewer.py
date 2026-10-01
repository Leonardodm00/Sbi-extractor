"""
raster_viewer.py -- choose the time window of the raster comparison by hand
(decision D-028, the interactive half; the command-line half is `plot`).

Opens the same comparison figure `plot` saves -- one column per class, one
raster per well, built by raster_plot.build_comparison -- with controls
underneath:

    slider      drag either handle, or the band between them; the panels
                follow 0.25 s after you stop moving
    t0 / t1     type a bound in seconds, press Enter
    keys        left / right: pan by half a window; up / down: zoom in /
                out by 2 about the centre; home: back to the first window
    toolbar     matplotlib's zoom and pan work too: the window follows the
                x axis, so zooming out fetches the spikes that came into view
    Save        writes the current window as PNG/PDF (+ JSON) through
                raster_plot.figure_compare and save_figure, the functions
                `plot` uses, so a saved window and `plot --window` of the
                same span are the same figure

It needs a display. On davinci that is an SSH session with X11 forwarding
(MobaXterm's default; `echo $DISPLAY` must print something such as
localhost:10.0). Without one it refuses and names `plot` instead. It can
also run on any machine that has the output folder (wells.tsv + cache/)
and numpy + matplotlib: it reads no raw data and no manifest.

Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import raster_data as RD                                          # noqa: E402
import raster_plot as RP                                          # noqa: E402

__all__ = ["NON_INTERACTIVE", "require_display", "clamp_window", "RasterViewer",
           "run_viewer"]

NON_INTERACTIVE = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}
MIN_WIDTH_S = 0.01          # narrowest window the controls will set [s]
DEBOUNCE_MS = 250

# Widget band under the panels, in inches.
_WIDGET_BAND = 1.05


def require_display() -> str:
    """Import pyplot with an interactive backend, or refuse.

    Returns the backend name. Raises RasterDataError when there is no
    display or matplotlib fell back to a non-interactive backend.
    """
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise _no_display_error("$DISPLAY is empty")
    import matplotlib
    import matplotlib.pyplot as plt                               # noqa: F401
    backend = str(matplotlib.get_backend())
    if backend.lower() in NON_INTERACTIVE:
        raise _no_display_error("matplotlib chose the non-interactive "
                                "backend %r" % backend)
    return backend


def _no_display_error(why: str) -> RD.RasterDataError:
    return RD.RasterDataError(
        "the viewer needs a display (%s). On davinci, open the SSH session with "
        "X11 forwarding (MobaXterm does it by default; `echo $DISPLAY` must "
        "print something like localhost:10.0) and, if Tk is missing, try "
        "MPLBACKEND=QtAgg. Or make static figures instead:  "
        "python3 run_raster_plots.py plot --window T0 T1" % why)


def clamp_window(t0: float, t1: float, T: float,
                 min_width: float = MIN_WIDTH_S) -> Tuple[float, float]:
    """Fit [t0, t1) inside [0, T], keeping its width where possible.

    A window wider than the recording becomes [0, T]; one narrower than
    min_width is widened about its centre; one that sticks out on either side
    is slid back inside without changing its width.
    """
    a, b = float(t0), float(t1)
    if b < a:
        a, b = b, a
    width = min(max(b - a, min_width), T)
    if b - a < width:                                    # widen about the centre
        c = 0.5 * (a + b)
        a, b = c - 0.5 * width, c + 0.5 * width
    if a < 0.0:
        a, b = 0.0, width
    if b > T:
        a, b = T - width, T
    return max(a, 0.0), min(b, T)


class RasterViewer:
    """State and callbacks of one viewer window (testable without a display:
    pass any Figure and call apply / on_key / on_text directly)."""

    def __init__(self, tables: Sequence[RD.SpikeTable], classes: Sequence[str],
                 t0: float, t1: float, out_root: str, rows: List[dict],
                 header: Dict[str, str], formats: Sequence[str] = ("png", "pdf"),
                 dpi: float = 300.0, panel_height_in: float = 1.7,
                 width_in: float = 11.0, save_width_in: float = 7.2,
                 save_panel_height_in: float = 1.9):
        if not tables:
            raise RD.RasterDataError("nothing to view")
        self.tables = list(tables)
        self.classes = list(classes)
        self.T = min(t.T_rec for t in self.tables)
        self.t0, self.t1 = clamp_window(t0, t1, self.T)
        self.home = (self.t0, self.t1)
        self.out_root = out_root
        self.rows = rows
        self.header = header
        self.formats = tuple(formats)
        self.dpi = float(dpi)
        self.panel_height_in = float(panel_height_in)
        self.width_in = float(width_in)
        self.save_width_in = float(save_width_in)
        self.save_panel_height_in = float(save_panel_height_in)
        self.mfr = RD.common_preprocessing(self.tables)["mfr_threshold"]
        self.layout = None
        self.slider = self.box0 = self.box1 = self.button = self.status = None
        self.timer = None
        self._pending = None
        self._busy = False
        self.saved: List[str] = []

    # -- data -------------------------------------------------------------
    def views(self, t0: float, t1: float) -> Dict[str, list]:
        return RD.group_by_class([RD.raster_view(t, t0, t1) for t in self.tables],
                                 self.classes)

    # -- figure -----------------------------------------------------------
    def figure_size(self) -> Tuple[float, float]:
        n = max(len(v) for v in RD.group_by_class(self.tables, self.classes).values())
        return RP.comparison_size(n, self.width_in, self.panel_height_in, _WIDGET_BAND)

    def build(self, fig) -> None:
        """Draw the panels and the controls into fig (sized by figure_size)."""
        from matplotlib.widgets import Button, RangeSlider, TextBox

        self.layout = RP.build_comparison(
            fig, self.views(self.t0, self.t1), self.classes, self.mfr,
            panel_height_in=self.panel_height_in, extra_bottom_in=_WIDGET_BAND)
        W, H = fig.get_size_inches()
        x0, x1 = RP._LEFT / W, 1.0 - RP._RIGHT / W

        def box(left_in, bottom_in, width_in, height_in):
            return fig.add_axes([left_in / W, bottom_in / H, width_in / W, height_in / H])

        ax_s = box(RP._LEFT + 0.55, 0.70, W - RP._LEFT - RP._RIGHT - 0.55 - 1.4, 0.16)
        self.slider = RangeSlider(ax_s, "window (s)", 0.0, self.T,
                                  valinit=(self.t0, self.t1))
        self.slider.label.set_fontsize(8)
        self.slider.valtext.set_fontsize(8)
        self.box0 = TextBox(box(RP._LEFT + 0.55, 0.30, 0.9, 0.24), "t0 ",
                            initial="%.3f" % self.t0)
        self.box1 = TextBox(box(RP._LEFT + 1.95, 0.30, 0.9, 0.24), "t1 ",
                            initial="%.3f" % self.t1)
        self.button = Button(box(RP._LEFT + 3.1, 0.30, 1.3, 0.24),
                             "Save %s" % "/".join(f.upper() for f in self.formats))
        fig.text(x0, 0.08 / H,
                 "arrows: pan / zoom   home: first window   toolbar zoom works too   "
                 "Save writes to %s" % os.path.join(self.out_root, "figures"),
                 fontsize=7, color=RP.INK["muted"], ha="left", va="bottom")
        self.status = fig.text(x1, 0.08 / H, "", fontsize=7, color=RP.INK["secondary"],
                               ha="right", va="bottom")

        self.slider.on_changed(self.on_slider)
        self.box0.on_submit(lambda s: self.on_text(s, None))
        self.box1.on_submit(lambda s: self.on_text(None, s))
        self.button.on_clicked(lambda _ev: self.save())
        for ax in self.layout.axes.values():
            ax.callbacks.connect("xlim_changed", self.on_xlim)
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        from matplotlib.backend_bases import TimerBase
        try:
            timer = fig.canvas.new_timer(interval=DEBOUNCE_MS)
        except Exception:                                         # noqa: BLE001
            timer = None
        if timer is None or type(timer) is TimerBase:             # no event loop:
            self.timer = None                                     # apply at once
        else:
            timer.single_shot = True
            timer.add_callback(self._apply_pending)
            self.timer = timer

    # -- the one place a window is applied --------------------------------
    def apply(self, t0: float, t1: float) -> Tuple[float, float]:
        a, b = clamp_window(t0, t1, self.T)
        self._busy = True
        try:
            self.t0, self.t1 = a, b
            self.layout.update(self.views(a, b), a, b)
            if self.slider is not None and tuple(self.slider.val) != (a, b):
                self.slider.set_val((a, b))
            if self.box0 is not None:
                self.box0.set_val("%.3f" % a)
                self.box1.set_val("%.3f" % b)
        finally:
            self._busy = False
        self.layout.fig.canvas.draw_idle()
        return a, b

    def _schedule(self, t0: float, t1: float) -> None:
        self._pending = (t0, t1)
        if self.timer is None:
            self._apply_pending()
        else:
            self.timer.stop()
            self.timer.start()

    def _apply_pending(self) -> None:
        if self._pending is not None:
            p, self._pending = self._pending, None
            if not np.allclose(p, (self.t0, self.t1)):
                self.apply(*p)

    # -- callbacks ----------------------------------------------------------
    def on_slider(self, val) -> None:
        if self._busy:
            return
        self._schedule(float(val[0]), float(val[1]))

    def on_xlim(self, ax) -> None:
        if self._busy:
            return
        lo, hi = ax.get_xlim()
        if not np.allclose((lo, hi), (self.t0, self.t1)):
            self._schedule(lo, hi)

    def on_text(self, s0, s1) -> None:
        if self._busy:
            return
        try:
            a = float(s0) if s0 is not None else self.t0
            b = float(s1) if s1 is not None else self.t1
        except ValueError:
            self._say("not a number: %r" % (s0 if s0 is not None else s1))
            return
        self.apply(a, b)

    def on_key(self, event) -> None:
        if self._busy:
            return
        for tb in (self.box0, self.box1):
            if tb is not None and getattr(tb, "capturekeystrokes", False):
                return                                   # typing into a box
        w = self.t1 - self.t0
        c = 0.5 * (self.t0 + self.t1)
        key = event.key
        if key == "left":
            self.apply(self.t0 - 0.5 * w, self.t1 - 0.5 * w)
        elif key == "right":
            self.apply(self.t0 + 0.5 * w, self.t1 + 0.5 * w)
        elif key in ("up", "+", "="):
            self.apply(c - 0.25 * w, c + 0.25 * w)
        elif key in ("down", "-"):
            self.apply(c - w, c + w)
        elif key == "home":
            self.apply(*self.home)

    def save(self) -> List[str]:
        """The current window, through the same code as `plot`."""
        views = self.views(self.t0, self.t1)
        fig, lay = RP.figure_compare(views, self.classes, self.mfr,
                                     width_in=self.save_width_in,
                                     panel_height_in=self.save_panel_height_in,
                                     dpi=self.dpi, return_layout=True)
        stem = os.path.join(self.out_root, "figures",
                            "raster_compare_" + RD.fmt_window(self.t0, self.t1))
        side = RD.figure_sidecar(self.tables, views, self.t0, self.t1,
                                 self.rows, self.header, how="viewer Save")
        side.update(marker_alpha=lay.alpha, marker_alpha_rule=RP.ALPHA_RULE)
        written = RP.save_figure(fig, stem, self.formats, side)
        self.saved.extend(written)
        self._say("saved %s" % os.path.basename(written[0]))
        print("[raster] viewer saved: %s" % ", ".join(written))
        return written

    def _say(self, text: str) -> None:
        if self.status is not None:
            self.status.set_text(text)
            self.layout.fig.canvas.draw_idle()


def run_viewer(tables, classes, t0, t1, out_root, rows, header,
               formats=("png", "pdf"), dpi=300.0, save_width_in=7.2,
               save_panel_height_in=1.9) -> RasterViewer:
    """Open the window and block until it is closed."""
    require_display()
    import matplotlib.pyplot as plt

    for k, drop in (("keymap.back", "left"), ("keymap.forward", "right"),
                    ("keymap.home", "home")):
        try:
            plt.rcParams[k] = [x for x in plt.rcParams[k] if x != drop]
        except KeyError:
            pass
    v = RasterViewer(tables, classes, t0, t1, out_root, rows, header, formats, dpi,
                     save_width_in=save_width_in,
                     save_panel_height_in=save_panel_height_in)
    fig = plt.figure(figsize=v.figure_size(), facecolor=RP.INK["surface"])
    try:
        fig.canvas.manager.set_window_title("raster viewer -- %s" % out_root)
    except Exception:                                             # noqa: BLE001
        pass
    v.build(fig)
    plt.show()
    return v
