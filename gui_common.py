# -*- coding: utf-8 -*-
"""Shared look and plumbing for every GUI in this folder (Concentration_GUI.py
and the ones that follow), so they look and behave as one family.  Nothing
here is specific to one window: each GUI keeps its own settings (DEFAULTS,
dialogs) in its own file and takes only the common parts from here.

    STYLESHEET, palette    the look (same family as Voigt_GUI.py / Alignment_GUI.py)
    PlotCanvas             matplotlib figure + zoom/pan toolbar
    style_axes, legend     axis and legend styling
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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

HERE = os.path.dirname(os.path.abspath(__file__))

# Same reference palette as Alignment_GUI.py, assigned in fixed order.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                  "#7b61c9", "#3fb6c6", "#a36a3d")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID_INK, SURFACE = "#e6e6e3", "#ffffff"
C_DATA, C_FIT, C_DECONV, C_RESID, C_REGION = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#dbeafe"


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
def style_axes(ax, base_font=9.0):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_INK, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=base_font, colors=INK_SECONDARY, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_PRIMARY); ax.yaxis.label.set_color(INK_PRIMARY)
    ax.ticklabel_format(useOffset=False, axis="x")


def legend(ax, base_font=9.0, loc="best"):
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    leg = ax.legend(loc=loc, fontsize=base_font - 0.5, frameon=True, framealpha=1.0,
                    edgecolor=GRID_INK, facecolor=SURFACE, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)


class PlotCanvas(QWidget):
    """A matplotlib figure with the standard zoom/pan toolbar."""

    def __init__(self, height_in=4.0, toolbar=True, parent=None):
        super().__init__(parent)
        # constrained layout is recomputed at every draw, so the axes always
        # fill the canvas at its current size (tight_layout ran once, too early)
        self.fig = Figure(figsize=(6, height_in), dpi=100, facecolor=SURFACE, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setStyleSheet("QToolBar { background: #ffffff; border: none; }")
            v.addWidget(self.toolbar)
        v.addWidget(self.canvas, 1)

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
