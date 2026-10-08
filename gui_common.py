# -*- coding: utf-8 -*-
"""Shared look and plumbing for every GUI in this folder (Concentration_GUI.py
and the ones that follow), so they look and behave as one family.  Nothing
here is specific to one window: each GUI keeps its own settings (DEFAULTS,
dialogs) in its own file and takes only the common parts from here.

    STYLESHEET, palette    the look (same family as Voigt_GUI.py / Alignment_GUI.py)
    PlotCanvas             matplotlib figure in the publication style, no toolbar
    attach_zoom_pan        scroll-zoom / drag-pan of the x axis without a toolbar
    style_axes, legend     axis and legend styling (= pub_axes / pub_legend)
    PUB_*, PubCanvas, pub_axes, pub_legend, pub_figure, pub_savefig, figure_png_bytes
                           the publication plot style of Camera_calibration_GUI
                           (compact 7/6 pt, grid, framed legend, 120 dpi),
                           see "Publication plots" below
    style_table, table_cell
    JobWorker              runs (state, callable) jobs off the GUI thread
    fmt                    'value ± error'
    config_dir, results_dir, load_json, save_json
                           Config/ and Results/ next to this file, JSON settings

A new window starts from:

    from gui_common import STYLESHEET, PlotCanvas, JobWorker, config_dir, load_json, save_json

    DEFAULTS = {...}                                       # this GUI's own settings
    SETTINGS = os.path.join(config_dir(), "mygui_settings.json")

    class MyWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setStyleSheet(STYLESHEET)
            self._settings = load_json(SETTINGS, DEFAULTS)
            ...
            b = QPushButton("Run")                          # primary (blue)
            b = QPushButton("Open…"); b.setObjectName("secondary")   # secondary (grey)
"""

import json
import math
import os
import time

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QHeaderView, QAbstractItemView, QTableWidgetItem
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

HERE = os.path.dirname(os.path.abspath(__file__))

# Plot colours: matplotlib's own tab10 (C0 blue, C1 orange, ...), as in the
# publication style below, assigned in fixed order.
SERIES_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#17becf")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#000000", "#404040", "#8a8983"
GRID_INK, SURFACE = "#b0b0b0", "#ffffff"
C_DATA, C_FIT, C_DECONV, C_RESID, C_REGION = "#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#dbeafe"


def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def config_dir():  return ensure_dir(os.path.join(HERE, "Config"))
def results_dir(): return ensure_dir(os.path.join(HERE, "Results"))


def load_json(path, defaults):
    """Settings from path, with every missing key taken from defaults."""
    try:
        s = json.load(open(path, "r", encoding="utf-8")) if os.path.isfile(path) else {}
        if not isinstance(s, dict): s = {}
    except Exception:
        s = {}
    for k, v in defaults.items():
        s.setdefault(k, v)
    return s


def save_json(path, s):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(s, f, indent=2)
    os.replace(tmp, path)


def fmt(v, err=None, digits=6):
    """'1.2345e-02 ± 3.1e-04', or '—' for nothing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = "%.*g" % (digits, v)
    if err is not None and not (isinstance(err, float) and math.isnan(err)):
        s += " ± %.2g" % err
    return s


# =============================================================================
# Workers
# =============================================================================
class JobWorker(QThread):
    """Runs a list of (state, callable returning a result dict) in order."""
    done = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self._jobs = jobs

    def run(self):
        for st, job in self._jobs:
            t = time.perf_counter()
            try:
                r = job()
            except Exception as e:
                self.failed.emit("%s: %s: %s" % (st.name, type(e).__name__, e)); continue
            r["elapsed_s"] = time.perf_counter() - t
            self.done.emit(st, r)


# =============================================================================
# Plots
# =============================================================================
def style_axes(ax, base_font=None):
    """The publication style (pub_axes) on ax; call again after setting the
    labels, or pass them to pub_axes directly.  base_font is ignored (kept for
    old callers)."""
    pub_axes(ax)
    ax.set_axisbelow(True)
    ax.ticklabel_format(useOffset=False, axis="x")


def legend(ax, base_font=None, loc="best", **kw):
    """The publication style's framed legend (pub_legend)."""
    return pub_legend(ax, loc=loc, **kw)


class PlotCanvas(QWidget):
    """A matplotlib figure in the publication style (see PubCanvas), without
    the matplotlib toolbar.  zoom_pan=True: scroll zooms the x axis around the
    cursor, a left drag pans it, a double click calls on_reset (see
    attach_zoom_pan)."""

    def __init__(self, height_in=4.0, toolbar=False, parent=None, zoom_pan=False,
                 on_change=None, on_reset=None):
        super().__init__(parent)
        # constrained layout is recomputed at every draw, so the axes always
        # fill the canvas at its current size (tight_layout ran once, too early)
        self.fig = Figure(figsize=(6, height_in), dpi=PUB_DPI, facecolor="white", layout="constrained")
        pub_layout(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self.canvas, 1)
        self.setStyleSheet("background: #ffffff;")
        if zoom_pan:
            attach_zoom_pan(self.canvas, on_change=on_change, on_reset=on_reset)

    def draw(self):
        self.canvas.draw_idle()


def attach_zoom_pan(canvas, on_change=None, on_reset=None, step=1.25):
    """Zoom and pan the x axis of a canvas without a toolbar: scroll zooms
    around the cursor, a left drag pans, a double click calls on_reset().
    on_change(ax) runs after every change (e.g. to rescale y); axes sharing x
    follow automatically."""
    st = {"ax": None, "x0": None, "xlim": None}

    def axes_at(ev):
        return ev.inaxes if ev.inaxes is not None and ev.inaxes.get_navigate() else None

    def changed(ax):
        if on_change is not None:
            on_change(ax)
        canvas.draw_idle()

    def on_scroll(ev):
        ax = axes_at(ev)
        if ax is None or ev.xdata is None:
            return
        f = 1.0 / step if ev.button == "up" else step
        lo, hi = ax.get_xlim()
        ax.set_xlim(ev.xdata - (ev.xdata - lo) * f, ev.xdata + (hi - ev.xdata) * f)
        changed(ax)

    def on_press(ev):
        ax = axes_at(ev)
        if ax is None or ev.button != 1:
            return
        if ev.dblclick:
            if on_reset is not None:
                on_reset()
            return
        st.update(ax=ax, x0=ev.x, xlim=ax.get_xlim())

    def on_move(ev):
        ax = st["ax"]
        if ax is None:
            return
        lo, hi = st["xlim"]
        width_px = ax.bbox.width or 1.0
        d = (ev.x - st["x0"]) / width_px * (hi - lo)
        ax.set_xlim(lo - d, hi - d)
        changed(ax)

    def on_release(ev):
        st.update(ax=None, x0=None, xlim=None)

    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_move)
    canvas.mpl_connect("button_release_event", on_release)


# =============================================================================
# Publication plots (the Camera_calibration_GUI style)
# =============================================================================
# The plots of Camera_calibration_GUI (scale_factor.py) as building blocks:
# matplotlib's own colours (tab10: C0 blue, C1 orange, ...) and font, full
# frame, grid, compact 7 pt labels and 6 pt ticks/legend, a framed legend with
# tight spacing, units in square brackets, the figure drawn at 120 dpi at the
# exact pixel size of the widget, no toolbar.  Every setting is applied to
# the axes explicitly, so the look does not depend on when the rcParams were
# active.  Use:
#
#     canvas = PubCanvas()                       # in the window, instead of PlotCanvas
#     fig = canvas.fig; fig.clear()
#     ax = fig.add_subplot(111)
#     ax.plot(x, y, "o", ms=PUB_MARKER, label="Data")
#     ax.plot(x, fit, "-", lw=PUB_FIT_LW, label="Fit")
#     pub_axes(ax, "Wavelength [nm]", "Optical depth [-]")
#     pub_legend(ax)
#     canvas.draw()
#
#     fig = pub_figure(1000, 650)                # the same for a file
#     ... ; pub_savefig(fig, "plot.png")         # 300 dpi PNG, or .pdf / .svg with real text
PUB_DPI = 120
PUB_FONT, PUB_LABEL, PUB_TICK, PUB_LEGEND, PUB_TITLE = 7.0, 7.0, 6.0, 6.0, 8.0
PUB_LW, PUB_FIT_LW, PUB_MARKER, PUB_GRID_LW = 1.2, 1.0, 3.0, 0.6
PUB_RC = {
    "font.size": PUB_FONT, "axes.labelsize": PUB_LABEL, "axes.titlesize": PUB_TITLE,
    "xtick.labelsize": PUB_TICK, "ytick.labelsize": PUB_TICK,
    "legend.fontsize": PUB_LEGEND, "legend.framealpha": 0.9,
    "lines.linewidth": PUB_LW, "grid.linewidth": PUB_GRID_LW,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def pub_style():
    """with pub_style(): ... - the rcParams of the style, for code that makes
    its own artists (pub_axes / pub_legend set everything explicitly anyway)."""
    import matplotlib
    return matplotlib.rc_context(PUB_RC)


def pub_axes(ax, xlabel=None, ylabel=None, title=None, grid=True):
    """Labels (labelpad 2), ticks (6 pt, pad 1), grid and frame of the style."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PUB_LABEL, labelpad=2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PUB_LABEL, labelpad=2)
    if title is not None:
        ax.set_title(title, fontsize=PUB_TITLE)
    ax.tick_params(axis="both", which="both", labelsize=PUB_TICK, pad=1)
    for t in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        t.set_fontsize(PUB_TICK)
    ax.xaxis.label.set_fontsize(PUB_LABEL); ax.yaxis.label.set_fontsize(PUB_LABEL)
    if grid:
        ax.grid(True, linewidth=PUB_GRID_LW)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_facecolor("white")


def pub_legend(ax, loc="upper left", **kw):
    """The style's compact framed legend (nothing when no artist has a label)."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return None
    opts = dict(loc=loc, frameon=True, framealpha=0.9, fontsize=PUB_LEGEND, borderpad=0.3,
                handlelength=1.2, handletextpad=0.4, labelspacing=0.25)
    opts.update(kw)
    return ax.legend(**opts)


def pub_layout(fig, pad_px=3):
    """Tight constrained-layout margins (the style's narrow borders)."""
    eng = fig.get_layout_engine()
    if eng is not None and hasattr(eng, "set"):
        eng.set(w_pad=pad_px / PUB_DPI, h_pad=pad_px / PUB_DPI, wspace=0.02, hspace=0.02)


def pub_figure(px_w=1000, px_h=650, dpi=PUB_DPI):
    """A Figure of exactly px_w x px_h pixels at the style's dpi (no pyplot)."""
    fig = Figure(figsize=(max(200, px_w) / dpi, max(140, px_h) / dpi), dpi=dpi, facecolor="white",
                 layout="constrained")
    pub_layout(fig)
    return fig


def pub_savefig(fig, path, dpi=300):
    """PNG at dpi (the figure keeps its layout, only the resolution goes up),
    or PDF / SVG with the text kept as text."""
    import matplotlib
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path, dpi=dpi, facecolor="white")


def figure_png_bytes(fig, dpi=None):
    """The figure as PNG bytes (e.g. for QPixmap.loadFromData)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi or fig.dpi, facecolor="white")
    return buf.getvalue()


class PubCanvas(QWidget):
    """A matplotlib canvas in the publication style: 120 dpi, so 7/6 pt text
    looks on screen as in the Camera_calibration_GUI images; no toolbar; the
    figure always fills the widget at its pixel size."""

    def __init__(self, parent=None, min_height=140):
        super().__init__(parent)
        self.fig = Figure(dpi=PUB_DPI, facecolor="white", layout="constrained")
        pub_layout(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(min_height)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self.canvas, 1)
        self.setStyleSheet("background: #ffffff;")

    def draw(self):
        self.canvas.draw_idle()


# =============================================================================
# Widgets
# =============================================================================
STYLESHEET = """
    QMainWindow, QDialog { background-color: #f3f5f7; }
    QWidget { font-family: 'Arial'; font-size: 13px; color: #0f172a; }
    QLabel { font-family: 'Arial'; font-size: 13px; color: #0f172a; }
    QLabel[muted="true"] { color: #64748b; font-size: 12px; }
    QLabel[cap="true"]   { color: #0f172a; font-weight: 700; }
    QLineEdit {
        font-family: 'Courier New'; font-size: 13px; padding: 7px 10px;
        border: 1px solid #d7dee8; border-radius: 10px; background-color: #fff; color: #0f172a;
    }
    QLineEdit:focus { border: 1px solid #3b82f6; background-color: #eff6ff; }
    QSpinBox, QDoubleSpinBox, QComboBox {
        font-family: 'Arial'; font-size: 13px; padding: 6px 8px;
        border: 1px solid #d7dee8; border-radius: 10px; background-color: #fff; color: #0f172a;
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #3b82f6; }
    QTextEdit {
        background: #f7fafc; border: 1px solid #dde3ea;
        border-radius: 8px; padding: 8px; font-size: 12px; color: #334155;
    }
    QListWidget {
        background: #ffffff; border: 1px solid #dde3ea; border-radius: 8px; padding: 4px;
    }
    QListWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
    QPushButton {
        font-family: 'Arial'; font-size: 13px; font-weight: 750; padding: 6px 14px;
        border: none; border-radius: 10px;
        background-color: #3b82f6; color: white;
    }
    QPushButton:hover    { background-color: #2563eb; }
    QPushButton:pressed  { background-color: #1d4ed8; }
    QPushButton:disabled { background-color: #cbd5e1; color: #f1f5f9; }
    QPushButton:checked  { background-color: #1d4ed8; }
    QPushButton#secondary {
        background-color: #eef2f7; color: #223046;
        border: 1px solid #d7dee8;
    }
    QPushButton#secondary:hover { background-color: #dde6f2; }
    QPushButton#secondary:checked { background-color: #bfdbfe; color: #1e3a8a; }
    QGroupBox {
        font-family: 'Arial'; font-size: 13px; font-weight: 650; color: #223046;
        background-color: #ffffff; border: 1px solid #dde3ea;
        border-radius: 14px; margin-top: 14px; padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 14px; padding: 2px 10px;
        background-color: #dbeafe; color: #1e3a8a;
        border: 1px solid #bfdbfe; border-radius: 10px;
    }
    QTabWidget::pane { border: 1px solid #ccc; background-color: #fff; border-radius: 4px; }
    QTabBar::tab {
        background-color: #e0e0e0; padding: 6px 14px; margin-right: 2px;
        border-top-left-radius: 5px; border-top-right-radius: 5px;
    }
    QTabBar::tab:selected { background-color: #0078d7; color: white; }
    QCheckBox { spacing: 6px; font-size: 13px; }
    QMenuBar { background-color: #ffffff; }
    QMenuBar::item:selected { background-color: #dbeafe; }
"""


def style_table(t, editable=False):
    t.verticalHeader().setVisible(False)
    h = t.horizontalHeader()
    for c in range(t.columnCount()):
        h.setSectionResizeMode(c, QHeaderView.Stretch)
    if t.columnCount() > 4 and not editable:
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    if not editable:
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True); t.setShowGrid(True)
    t.setStyleSheet("""
        QTableWidget {
            background-color: #ffffff; border: 1px solid #dde3ea; border-radius: 0px;
            gridline-color: #dde3ea; alternate-background-color: #f7fafc;
            selection-background-color: #dbeafe; selection-color: #0f172a;
        }
    """)
    h.setStyleSheet("""
        QHeaderView::section {
            background-color: #f0f4f8; color: #475569; font-weight: 800;
            border: none; border-right: 1px solid #dde3ea; border-bottom: 2px solid #94a3b8;
            padding: 6px 10px; font-size: 12px;
        }
    """)
    t.verticalHeader().setDefaultSectionSize(26)


def table_cell(text, align=Qt.AlignCenter):
    it = QTableWidgetItem(text); it.setTextAlignment(align)
    f = QFont("Consolas"); f.setPointSize(9); it.setFont(f)
    return it
