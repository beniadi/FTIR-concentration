# -*- coding: utf-8 -*-
"""Concentration GUI - gas mixing ratio from an FTIR spectrum by spectral fitting.

The look (stylesheet, buttons, group boxes, tables, menus that own their
settings) follows Voigt_GUI.py through gui_common.py, so the two windows stay
one family.  The work is new: a HITRAN line-by-line forward model, convolved
with the instrument line shape, is fitted to the measured transmittance and
the mixing ratio comes out of the fit (retrieval.fit_spectrum).  With other
gases added (HITRAN -> Add Other Gas), every gas gets its own mixing ratio in
one fit (retrieval.fit_multigas).

    python Concentration_GUI.py
    python Concentration_GUI.py spectrum.txt     # open a spectrum at start-up
    python Concentration_GUI.py --no-example     # start empty
    python Concentration_GUI.py --selftest       # the retrieval on the example, no screen

At start-up the window opens the worked example - Input/N2O_simulated_500ppm.txt
(simulated by simulate_spectrum.py with 500 ppm N2O), the N2O line list in
Input/HITRAN and the LINEFIT ILS - and retrieves it, so there is always a
known answer to compare against.  Help -> Load Example repeats it.

WHAT IT DOES
------------
1. Opens a spectrum (two-column text or Bruker OPUS binary), transmittance or
   absorbance (log10 or ln).
2. Takes a HITRAN line list - a .data/.par file, a CSV, or downloaded from
   HITRANonline through HAPI (HITRAN -> Download Lines).
3. Takes the instrument line shape: a measured file (ILS_LINEFIT.txt) or a
   synthetic Gaussian / sinc.
4. With the cell temperature, pressure and path length, fits
       T_model = baseline * [(1 - z) * (exp(-x N L sum S V) (*) ILS) + z]
   to the data in the analysis region and reports x (ppm) with its
   uncertainty, the fitted shift, broadening, zero offset and baseline, and
   a consistency check (Retrieval -> Consistency Check): every strong line
   group refitted on its own, one ppm per segment, for every gas.
5. Gases that overlap the target's lines (H2O, CO, CO2 ...) can be added with
   their own line lists; they are then fitted together with the target, and
   the plot can show each gas's own transmittance.

Requires numpy, scipy, matplotlib, PyQt5, and hitran-api for TIPS partition
sums and downloads (the fit falls back to a power law without it).
"""

import os, sys, re, math
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QGridLayout, QMenuBar, QAction, QMessageBox, QSpinBox, QComboBox,
    QCheckBox, QTabWidget, QTextEdit, QTableWidget, QHeaderView, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox, QInputDialog, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spectrum_io as sio                                   # noqa: E402
import ils as ils_mod                                       # noqa: E402
import concentration as conc                                # noqa: E402
import retrieval as rt                                      # noqa: E402
import hitran_fetch                                         # noqa: E402
from gui_common import (STYLESHEET, PlotCanvas, JobWorker, style_table,                      # noqa: E402
                        table_cell, fmt, config_dir, results_dir, load_json, save_json,
                        INK_SECONDARY, INK_MUTED,
                        C_DATA, C_FIT, C_RESID, C_REGION, SERIES_COLOURS,
                        PUB_FONT, PUB_LW, PUB_FIT_LW, PUB_MARKER,
                        pub_axes, pub_legend, pub_figure, pub_savefig)

APP_TITLE = "FTIR Concentration 1.0"
WIN_W, WIN_H = 1440, 940
PRESSURE_UNITS = tuple(conc.PRESSURE_UNITS)                 # atm, hPa, mbar, Torr, kPa, Pa


def settings_path(): return os.path.join(config_dir(), "concentration_settings.json")


DEFAULTS = {
    # where things were last
    "last_open_dir": None, "last_save_dir": None,
    # data convention for absorbance files
    "abs_base": "10",
    # the cell
    "T_K": 297.15, "P_value": 9456.5556, "P_unit": "Pa", "L_cm": 316.9,
    # the fit (retrieval.DEFAULTS)
    "fit_shift": True, "max_shift": 0.1,
    "fit_broadening": True, "broadening0": 0.003, "max_broadening": 0.05,
    "zero_mode": "auto", "fit_lorentz": False,
    "baseline_order": 1, "oversample": 8,
    "q_mode": "tips", "iso": "", "s_min_rel": 1e-5, "mass_amu": 44.0,
    "ils_centre": "peak", "windows": "auto",
    # synthetic ILS
    "gauss_fwhm": 0.026, "sinc_mopd_cm": 9.47, "ils_half_width": 0.25,
    # HITRAN download
    "dl_molecule": "N2O", "dl_other_molecule": "H2O", "dl_main_iso": False,
    # start-up
    "example_on_start": True,
}

# The worked example: a spectrum simulated with HAPI at a known 500 ppm, so
# the retrieval has a right answer.  Paths are relative to this file.
EXAMPLE = {
    "spectrum": ("Input", "N2O_simulated_500ppm.txt"),
    "lines": ("Input", "HITRAN", "N2O_2214.00-2221.00.data"),
    "ils": ("Input", "ILS_LINEFIT.txt"),
    "region": (2215.5, 2220.0),
}
# The multi-gas example (multigas_example.py): 500 ppm N2O, 2 % H2O and 30 ppm
# CO, with the H2O line between two N2O lines.  "others" are the other gases.
EXAMPLE_MULTI = {
    "spectrum": ("Input", "Mix_N2O_H2O_CO_simulated.txt"),
    "lines": ("Input", "HITRAN", "N2O_2156.00-2168.00.data"),
    "others": (("Input", "HITRAN", "H2O_2156.00-2168.00.data"), ("Input", "HITRAN", "CO_2156.00-2168.00.data")),
    "ils": ("Input", "ILS_LINEFIT.txt"),
    "region": (2160.0, 2163.5),
}


def example_paths(ex=EXAMPLE):
    out = {k: os.path.join(HERE, *v) for k, v in ex.items() if k not in ("region", "others")}
    if "others" in ex:
        out["others"] = [os.path.join(HERE, *v) for v in ex["others"]]
    files = [p for v in out.values() for p in (v if isinstance(v, list) else [v])]
    return out if all(os.path.isfile(p) for p in files) else None


def load_settings():
    s = load_json(settings_path(), DEFAULTS)
    if not s["last_open_dir"]: s["last_open_dir"] = os.path.join(HERE, "Input")
    if not s["last_save_dir"]: s["last_save_dir"] = results_dir()
    return s


def fit_options(s):
    """retrieval.fit_spectrum's options from the settings."""
    return {"fit_shift": bool(s["fit_shift"]), "max_shift": float(s["max_shift"]),
            "fit_broadening": bool(s["fit_broadening"]), "broadening0": float(s["broadening0"]),
            "max_broadening": float(s["max_broadening"]),
            "fit_zero": {"fit": True, "fixed": False}.get(s["zero_mode"], "auto"),
            "fit_lorentz": bool(s["fit_lorentz"]),
            "baseline_order": int(s["baseline_order"]), "oversample": int(s["oversample"]),
            "q_mode": s["q_mode"], "iso": str(s["iso"]) or None,
            "s_min_rel": float(s["s_min_rel"]), "mass_amu": float(s["mass_amu"]),
            "ils_centre": s["ils_centre"],
            "windows": "auto" if s["windows"] == "auto" else None}


_HEADER_RE = re.compile(r"ppm\s+([-+\d.eE]+)\s+T\s+([-+\d.eE]+)\s*K\s+P\s+([-+\d.eE]+)\s*Pa"
                        r"\s+L\s+([-+\d.eE]+)\s*cm")


def read_sim_header(path):
    """(ppm, T_K, P_Pa, L_cm) from a simulate_spectrum.py header, else None."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(30):
                ln = fh.readline()
                if not ln:
                    break
                m = _HEADER_RE.search(ln)
                if m:
                    return tuple(float(g) for g in m.groups())
    except OSError:
        pass
    return None


_GAS_RE = re.compile(r"^\s*#?\s*gas\s+(\S+)\s+([-+\d.eE]+)\s*ppm")


def read_sim_gases(path):
    """{gas: ppm} from the "gas <line list> <ppm> ppm" lines of a simulated
    mixture's header (the gas named after its line-list file), else {}."""
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(30):
                m = _GAS_RE.match(fh.readline())
                if m:
                    name = os.path.basename(m.group(1).replace("\\", "/")).split("_")[0]
                    out[name] = float(m.group(2))
    except OSError:
        pass
    return out


def gas_name(lines, path):
    """The molecule's HITRAN name if the list holds one molecule, else the
    file name up to its first '_' ('H2O_2156.00-2168.00.data' -> 'H2O')."""
    mols = set(int(m) for m in lines["mol"])
    if len(mols) == 1 and 0 not in mols:
        try:
            n = hitran_fetch.molecule_name(mols.pop())
            if n:
                return n
        except Exception:
            pass
    return os.path.basename(path).split("_")[0].split(".")[0] or "gas"


def retrieve(x, y, target, lines, others, T_K, P_pa, L_cm, ils, opts):
    """fit_spectrum, or fit_multigas when there are other gases.  A multi-gas
    result is given the single-gas keys for the target (ppm, ppm_err, tau_max
    as the largest of any gas, n_lines as the total), with the per-gas values
    under gas_ppm, gas_err, gas_tau and gas_lines."""
    if not others:
        r = rt.fit_spectrum(x, y, lines, T_K, P_pa, L_cm, ils, opts)
        r.update(multi=False, target=target)
        return r
    gases = [(target, lines)] + [(g["name"], g["lines"]) for g in others]
    r = rt.fit_multigas(x, y, gases, T_K, P_pa, L_cm, ils, opts)
    r.update(multi=True, target=target, gas_ppm=r["ppm"], gas_err=r["ppm_err"], gas_tau=r["tau_max"],
             gas_lines=r["n_lines"])
    r["ppm"], r["ppm_err"] = r["gas_ppm"][target], r["gas_err"][target]
    r["tau_max"], r["n_lines"] = max(r["gas_tau"].values()), sum(r["gas_lines"].values())
    return r


_LIST_RE = re.compile(r"^(.+?)_(\d+(?:\.\d*)?)-(\d+(?:\.\d*)?)\.(?:data|par)$", re.I)


def scan_line_lists(folder=None):
    """{path: {"gas", "lo", "hi"}} for the line lists in Input/HITRAN, read
    from their names ('H2O_2156.00-2168.00.data', as hitran_fetch saves them)."""
    folder = folder or hitran_fetch.HITRAN_DIR
    out = {}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return out
    for n in names:
        m = _LIST_RE.match(n)
        if m:
            out[os.path.join(folder, n)] = {"gas": m.group(1), "lo": float(m.group(2)), "hi": float(m.group(3))}
    return out


def region_cover(info, region):
    """Fraction of the region inside the list's wavenumber range (1 without a region)."""
    if region is None:
        return 1.0
    return max(0.0, min(info["hi"], region[1]) - max(info["lo"], region[0])) / (region[1] - region[0])


class Job:
    """What JobWorker needs to name a job in its error messages."""
    def __init__(self, name):
        self.name = name


# =============================================================================
# Plot
# =============================================================================
WN_LABEL = "Wavenumber [cm$^{-1}$]"


def build_main_figure(fig, x, T, region, r, show, keep_xlim=None):
    """Measured and modelled transmittance, with the residual underneath.
    Drawn in the gui_common publication style."""
    fig.clear()
    with_res = bool(show.get("residual")) and r is not None
    if with_res:
        gs = fig.add_gridspec(2, 1, height_ratios=(3, 1))
        ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot(111); axr = None
    if x is None:
        ax.text(0.5, 0.5, "File → Open Spectrum", transform=ax.transAxes, ha="center", va="center",
                color=INK_MUTED, fontsize=PUB_FONT + 1)
        ax.set_xticks([]); ax.set_yticks([])
        return
    if region:
        ax.axvspan(region[0], region[1], color=C_REGION, alpha=0.6, lw=0, label="Region")
    if show.get("data", True):
        ax.plot(x, T, color=C_DATA, lw=PUB_LW, label="Measured")
    if r is not None:
        if show.get("fit", True):
            ax.plot(r["x"], r["fit"], color=C_FIT, lw=PUB_FIT_LW, label="Model  %s%s ppm" % (
                r["target"] + " " if r.get("multi") else "", fmt(r["ppm"], digits=5)))
        if show.get("baseline"):
            ax.plot(r["x"], r["baseline_curve"], color=INK_SECONDARY, lw=PUB_FIT_LW, ls="--", label="Baseline")
        if show.get("gases") and r.get("components"):
            for k, (g, c) in enumerate(r["components"].items()):
                ax.plot(r["x"], c, color=SERIES_COLOURS[(k + 2) % len(SERIES_COLOURS)], lw=PUB_FIT_LW,
                        label="%s  %s ppm" % (g, fmt(r["gas_ppm"][g], digits=5)))
        if show.get("windows"):
            for w in r.get("windows") or []:
                for e in (w["lo"], w["hi"]):
                    ax.axvline(e, color=INK_MUTED, lw=0.6, ls=":")
    pub_axes(ax, None if axr is not None else WN_LABEL, "Transmittance [-]")
    ax.ticklabel_format(useOffset=False, axis="x")
    pub_legend(ax, loc="lower left")
    if axr is not None:
        axr.axhline(0.0, color="k", lw=0.6)
        axr.plot(r["x"], r["residual"], color=C_RESID, lw=PUB_FIT_LW)
        pub_axes(axr, WN_LABEL, "Residual [-]")
        axr.ticklabel_format(useOffset=False, axis="x")
        ax.tick_params(labelbottom=False)
    if keep_xlim:
        ax.set_xlim(*keep_xlim)
    elif region:
        pad = 0.05 * (region[1] - region[0])
        ax.set_xlim(max(x[0], region[0] - pad), min(x[-1], region[1] + pad))
    else:
        ax.set_xlim(x[0], x[-1])
    _autoscale_y(ax)


def _autoscale_y(ax):
    """y limits from what is visible in the current x range."""
    lo, hi = ax.get_xlim()
    ys = []
    for ln in ax.get_lines():
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(xd) < 2:
            continue
        m = (xd >= lo) & (xd <= hi)
        if m.any():
            ys.append(yd[m])
    if ys:
        y = np.concatenate(ys); y = y[np.isfinite(y)]
        if len(y):
            a, b = float(y.min()), float(y.max())
            pad = 0.05 * (b - a) if b > a else 0.05
            ax.set_ylim(a - pad, b + pad)


# =============================================================================
# Dialogs
# =============================================================================
class SettingsDialog(QDialog):
    """The retrieval settings.  Edits a copy and hands it back on OK, so
    Cancel really cancels."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Retrieval Settings")
        self.setMinimumWidth(560)
        self._s = dict(settings)
        self._w = {}
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)

        g, f = self._page("Free parameters (besides the concentration and the baseline)")
        row = QHBoxLayout(); row.addWidget(self._check("fit_shift", "Fit"))
        row.addWidget(QLabel("max |shift|")); row.addWidget(self._dspin("max_shift", 0.001, 5, 0.01, 4)); row.addStretch(1)
        f.addRow("Wavenumber shift (cm⁻¹)", row)
        row = QHBoxLayout(); row.addWidget(self._check("fit_broadening", "Fit"))
        row.addWidget(QLabel("start")); row.addWidget(self._dspin("broadening0", 0, 1, 0.001, 5))
        row.addWidget(QLabel("max")); row.addWidget(self._dspin("max_broadening", 0.0001, 1, 0.005, 5))
        row.addStretch(1)
        f.addRow("Extra ILS Gaussian HWHM (cm⁻¹)", row)
        f.addRow("Zero-level offset", self._combo("zero_mode", (
            ("auto", "Auto - fit only with a saturated line (peak τ > 1)"),
            ("fit", "Fit"), ("fixed", "Fixed at 0")),
            "Transmittance offset z (detector non-linearity, stray light).\n"
            "It matters most when lines are black at their centre. With only\n"
            "optically thin lines it cannot be told apart from a common scale on\n"
            "the concentrations, and fitting it then biases every gas."))
        f.addRow("Lorentz width scale", self._check("fit_lorentz", "Fit",
                 "Scale factor on the HITRAN pressure-broadened widths\n"
                 "(a pressure reading or broadening coefficients that are off)."))
        f.addRow("Baseline polynomial order", self._ispin("baseline_order", 0, 6,
                 "0 = constant scale, 1 = linear, ... in the normalised wavenumber."))
        v.addWidget(g)

        g, f = self._page("Forward model")
        f.addRow("Partition function", self._combo("q_mode", (
            ("tips", "TIPS from HAPI (exact, per isotopologue)"),
            ("linear", "Power law, linear molecule  (T₀/T)¹"),
            ("nonlinear", "Power law, non-linear molecule  (T₀/T)¹·⁵"))))
        f.addRow("Isotopologues", self._combo("iso", (
            ("", "All in the line list"), ("1", "Main isotopologue only"))))
        f.addRow("Skip lines weaker than (× strongest)", self._dspin("s_min_rel", 0, 0.1, 1e-5, 8))
        f.addRow("Doppler mass if unknown (amu)", self._dspin("mass_amu", 1, 1000, 1, 3,
                 "Used only when HAPI cannot give the isotopologue mass."))
        f.addRow("Fine-grid oversampling", self._ispin("oversample", 1, 64,
                 "Model points per data point before the ILS convolution."))
        f.addRow("Centre the ILS at", self._combo("ils_centre", (
            ("peak", "Its maximum"), ("centroid", "Its centroid"), ("zero", "The tabulated zero"))))
        v.addWidget(g)

        g, f = self._page("Consistency check")
        f.addRow("Segment refit", self._combo("windows", (
            ("auto", "One segment per strong line group"), ("none", "Off")),
            "Each line (group) refitted on its own with the shift, broadening and\n"
            "zero fixed: the segment concentrations should agree.\n"
            "In a multi-gas fit every gas gets its own segments; gases that\n"
            "absorb in a segment too are refitted with it."))
        f.addRow(self._check("example_on_start", "Load and retrieve the worked example at start-up"))
        v.addWidget(g)
        v.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore)
        v.addWidget(bb)

    # -- small builders ------------------------------------------------------
    @staticmethod
    def _page(title):
        w = QGroupBox(title)
        f = QFormLayout(w); f.setContentsMargins(16, 16, 16, 14); f.setSpacing(10)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return w, f

    def _dspin(self, key, lo, hi, step, decimals, tip=None):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(decimals); s.setValue(float(self._s[key]))
        if tip: s.setToolTip(tip)
        self._w[key] = (s, float); return s

    def _ispin(self, key, lo, hi, tip=None):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(int(self._s[key]))
        if tip: s.setToolTip(tip)
        self._w[key] = (s, int); return s

    def _check(self, key, text, tip=None):
        c = QCheckBox(text); c.setChecked(bool(self._s[key]))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, bool); return c

    def _combo(self, key, items, tip=None):
        c = QComboBox()
        for val, lab in items: c.addItem(lab, val)
        c.setCurrentIndex(max(0, c.findData(self._s[key])))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, str); return c

    def _restore(self):
        for k, (wd, cast) in self._w.items():
            val = DEFAULTS[k]
            if isinstance(wd, QComboBox): wd.setCurrentIndex(max(0, wd.findData(val)))
            elif isinstance(wd, QCheckBox): wd.setChecked(bool(val))
            else: wd.setValue(cast(val))

    def values(self):
        s = dict(self._s)
        for k, (wd, cast) in self._w.items():
            if isinstance(wd, QComboBox): s[k] = wd.currentData()
            elif isinstance(wd, QCheckBox): s[k] = wd.isChecked()
            else: s[k] = cast(wd.value())
        return s


class DownloadDialog(QDialog):
    """Molecule and wavenumber range for a HITRAN download."""

    def __init__(self, molecules, molecule, lo, hi, main_iso, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download HITRAN Lines")
        self.setMinimumWidth(440)
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)
        g = QGroupBox("HITRANonline (through HAPI)")
        f = QFormLayout(g); f.setContentsMargins(16, 16, 16, 14); f.setSpacing(10)
        self.combo_mol = QComboBox(); self.combo_mol.setEditable(True)
        self.combo_mol.addItems(molecules or [molecule])
        self.combo_mol.setCurrentText(molecule)
        f.addRow("Molecule", self.combo_mol)
        self.spin_lo = QDoubleSpinBox(); self.spin_hi = QDoubleSpinBox()
        for sp, val in ((self.spin_lo, lo), (self.spin_hi, hi)):
            sp.setRange(0, 1e5); sp.setDecimals(3); sp.setSingleStep(0.5); sp.setValue(val)
        row = QHBoxLayout(); row.addWidget(self.spin_lo); row.addWidget(QLabel("to")); row.addWidget(self.spin_hi)
        f.addRow("Wavenumber (cm⁻¹)", row)
        self.chk_main = QCheckBox("Main isotopologue only"); self.chk_main.setChecked(main_iso)
        f.addRow("", self.chk_main)
        v.addWidget(g)
        note = QLabel("The lines are saved in Input/HITRAN as a .data file (160-character HITRAN "
                      "format) with its .header. Take the range a little wider than the region, "
                      "so the wings of lines just outside it are in the model.")
        note.setProperty("muted", True); note.setWordWrap(True); v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Download")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def values(self):
        return (self.combo_mol.currentText().strip(), self.spin_lo.value(), self.spin_hi.value(),
                self.chk_main.isChecked())


class GasDialog(QDialog):
    """Which gases go into the fit: one target (the radio button) and any
    others (the ticks).  Each gas's line list is chosen here from the files
    known for it - the one that covers most of the region - so the window
    only ever shows chemical formulas."""

    def __init__(self, lists, target, chosen, region, add_file, download, parent=None):
        """lists {path: {"gas", "lo", "hi"}}; target the target gas; chosen
        {gas: path} the gases fitted now; add_file(path) -> info or None;
        download(on_done) starts a download that calls on_done(path)."""
        super().__init__(parent)
        self.setWindowTitle("Select Gases")
        self.setMinimumWidth(520)
        self._lists, self._region = dict(lists), region
        self._add_file, self._download = add_file, download
        self._current = dict(chosen)
        self._target = target
        self._fit = set(chosen)
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)
        box = QGroupBox("Gases with HITRAN lines")
        bv = QVBoxLayout(box); bv.setContentsMargins(16, 16, 16, 14)
        self._grid = QGridLayout(); self._grid.setHorizontalSpacing(18); self._grid.setVerticalSpacing(6)
        bv.addLayout(self._grid); bv.addStretch(1)
        v.addWidget(box, 1)
        note = QLabel("Target: the gas the result is reported for. Fit: gases fitted together with it, "
                      "each with its own concentration - tick the ones whose lines overlap the target's.")
        note.setProperty("muted", True); note.setWordWrap(True); v.addWidget(note)
        row = QHBoxLayout()
        b = QPushButton("Add from File…"); b.setObjectName("secondary"); b.clicked.connect(self._on_add)
        b.setToolTip("A HITRAN .data/.par or CSV line list outside Input/HITRAN.")
        row.addWidget(b)
        b = QPushButton("Download…"); b.setObjectName("secondary"); b.clicked.connect(self._on_download)
        b.setToolTip("Lines of another molecule from HITRANonline, saved in Input/HITRAN.")
        row.addWidget(b)
        row.addStretch(1)
        self._bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._bb.accepted.connect(self.accept); self._bb.rejected.connect(self.reject)
        row.addWidget(self._bb)
        v.addLayout(row)
        self._populate()

    def _best(self, gas):
        """The list for a gas: most of the region covered, the one in use on a
        tie, then the narrowest."""
        cands = [(p, i) for p, i in self._lists.items() if i["gas"] == gas]
        return min(cands, key=lambda c: (-round(region_cover(c[1], self._region), 3),
                                         c[0] != self._current.get(gas), c[1]["hi"] - c[1]["lo"]))

    def _populate(self):
        g = self._grid
        while g.count():
            w = g.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for c, t in enumerate(("Gas", "Target", "Fit", "")):
            h = QLabel(t); h.setProperty("muted", True); g.addWidget(h, 0, c)
        self._group = QButtonGroup(self)
        self._rows = {}
        gases = sorted(set(i["gas"] for i in self._lists.values()), key=str.lower)
        if not gases:
            g.addWidget(QLabel("No line lists in Input/HITRAN - download or add one."), 1, 0, 1, 4)
        for r, gas in enumerate(gases, 1):
            path, info = self._best(gas)
            cover = region_cover(info, self._region)
            name = QLabel(gas); name.setStyleSheet("font-weight: 700;")
            name.setToolTip("%s\n%.2f–%.2f cm⁻¹" % (path, info["lo"], info["hi"]))
            rb = QRadioButton(); chk = QCheckBox()
            self._group.addButton(rb)
            if cover <= 0:
                note = "no lines in the region"
            elif cover < 0.999:
                note = "lines cover %.0f %% of the region" % (100 * cover)
            else:
                note = ""
            lab = QLabel(note); lab.setProperty("muted", True)
            rb.setEnabled(cover > 0); chk.setEnabled(cover > 0)
            rb.setChecked(gas == self._target and cover > 0)
            chk.setChecked(gas in self._fit and cover > 0)
            rb.toggled.connect(lambda on, gas=gas: self._on_target(gas, on))
            chk.toggled.connect(lambda on, gas=gas: self._on_fit(gas, on))
            g.addWidget(name, r, 0); g.addWidget(rb, r, 1); g.addWidget(chk, r, 2); g.addWidget(lab, r, 3)
            self._rows[gas] = (path, rb, chk)
        g.setColumnStretch(3, 1)
        self._sync()

    def _on_target(self, gas, on):
        if on:
            self._target = gas
            self._fit.add(gas)
        self._sync()

    def _on_fit(self, gas, on):
        (self._fit.add if on else self._fit.discard)(gas)

    def _sync(self):
        """The target is always fitted: its tick is set and locked."""
        for gas, (_p, rb, chk) in self._rows.items():
            t = rb.isChecked()
            chk.blockSignals(True)
            if t:
                chk.setChecked(True)
            chk.setEnabled(rb.isEnabled() and not t)
            chk.blockSignals(False)
        self._bb.button(QDialogButtonBox.Ok).setEnabled(any(rb.isChecked() for _p, rb, _c in self._rows.values()))

    def add_list(self, path, info, tick=True):
        self._lists[path] = info
        self._current[info["gas"]] = path
        if tick:
            self._fit.add(info["gas"])
        self._populate()

    def _on_add(self):
        start = hitran_fetch.HITRAN_DIR if os.path.isdir(hitran_fetch.HITRAN_DIR) else HERE
        path, _ = QFileDialog.getOpenFileName(self, "Add HITRAN line list", start,
                                              "HITRAN (*.data *.par *.csv *.txt);;All files (*)")
        if path:
            info = self._add_file(path)
            if info:
                self.add_list(path, info)

    def _on_download(self):
        def done(path):
            info = self._add_file(path)
            if info:
                self.add_list(path, info)
        self._download(done)

    def values(self):
        """(target, {gas: path}) - the target is in the dict too."""
        chosen = {gas: path for gas, (path, rb, chk) in self._rows.items() if chk.isChecked()}
        return self._target, chosen


# =============================================================================
# The window
# =============================================================================
class ConcentrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(WIN_W, WIN_H)
        self.setStyleSheet(STYLESHEET)
        self._settings = load_settings()
        self._spec = None              # spectrum_io.Spectrum
        self._lines = None             # concentration.load_hitran dict
        self._lines_path = ""
        self._target = ""              # the target gas's name
        self._others = []              # other gases fitted with the target: {"name", "lines", "path"}
        self._extra_lists = {}         # line lists added from outside Input/HITRAN: {path: {"gas", "lo", "hi"}}
        self._ils = None              # (offset_cm1, values)
        self._ils_src = ""
        self._reference = None         # (ppm, T, P, L) from a simulated file's header
        self._ref_gases = {}           # {gas: ppm} from a simulated mixture's header
        self._result = None
        self._worker = None
        self._busy = False
        self._build_menu_bar(); self._build_central()
        self._show_others()
        self._redraw()
        self.statusBar().showMessage("Ready. Open a spectrum, a HITRAN line list and an ILS, then Retrieve.")

    # ---------------------------------------------------------------- menu --
    def _build_menu_bar(self):
        mb = QMenuBar(self); self.setMenuBar(mb)

        fm = mb.addMenu("File")
        a = QAction("Open Spectrum…", self); a.setShortcut("Ctrl+O")
        a.triggered.connect(self.open_spectrum); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Save Result…", self); a.setShortcut("Ctrl+S")
        a.triggered.connect(self.save_result); fm.addAction(a)
        a = QAction("Export Curves…", self)
        a.setToolTip("Data, model, baseline and residual in the region as text columns\n"
                     "(and each gas's own model in a multi-gas fit).")
        a.triggered.connect(self.export_curves); fm.addAction(a)
        a = QAction("Save Plot…", self); a.triggered.connect(self.save_plot); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Exit", self); a.triggered.connect(self.close); fm.addAction(a)

        hm = mb.addMenu("HITRAN")
        a = QAction("Select Gases…", self); a.setShortcut("Ctrl+G")
        a.setToolTip("The target gas and the other gases fitted with it.")
        a.triggered.connect(self.select_gases); hm.addAction(a)
        hm.addSeparator()
        a = QAction("Open Line List…", self); a.setShortcut("Ctrl+L")
        a.setToolTip("160-character HITRAN .par/.data, or a CSV with nu, sw, elower columns.")
        a.triggered.connect(self.open_lines); hm.addAction(a)
        a = QAction("Download Lines…", self); a.triggered.connect(self.download_lines); hm.addAction(a)
        hm.addSeparator()
        a = QAction("Add Other Gas from File…", self)
        a.setToolTip("A gas whose lines overlap the target's: it is fitted with its own concentration.")
        a.triggered.connect(self.add_other_gas); hm.addAction(a)
        a = QAction("Download Other Gas…", self)
        a.triggered.connect(lambda: self.download_lines(other=True)); hm.addAction(a)
        a = QAction("Clear Other Gases", self); a.triggered.connect(self.clear_other_gases); hm.addAction(a)

        im = mb.addMenu("ILS")
        a = QAction("Load ILS File…", self); a.setShortcut("Ctrl+I")
        a.setToolTip("Two columns: offset from line centre (cm⁻¹), value - e.g. ILS_LINEFIT.txt.")
        a.triggered.connect(self.load_ils); im.addAction(a)
        sm = im.addMenu("Synthetic ILS")
        a = QAction("Gaussian (FWHM)…", self); a.triggered.connect(lambda: self.synthetic_ils("gauss"))
        sm.addAction(a)
        a = QAction("Ideal sinc (max OPD)…", self); a.triggered.connect(lambda: self.synthetic_ils("sinc"))
        sm.addAction(a)

        rm = mb.addMenu("Retrieval")
        a = QAction("Retrieve Concentration", self); a.setShortcut("Ctrl+R")
        a.triggered.connect(self.run_retrieval); rm.addAction(a)
        a = QAction("Consistency Check…", self); a.setShortcut("Ctrl+K")
        a.setToolTip("Every strong line group refitted on its own - the concentrations should agree.")
        a.triggered.connect(self.show_consistency); rm.addAction(a)
        rm.addSeparator()
        a = QAction("Retrieval Settings…", self); a.triggered.connect(self.show_settings); rm.addAction(a)

        hp = mb.addMenu("Help")
        a = QAction("Load Example", self)
        a.setToolTip("N2O_simulated_500ppm.txt with the N2O line list and ILS_LINEFIT.txt - true answer 500 ppm.")
        a.triggered.connect(self.load_example); hp.addAction(a)
        a = QAction("Load Multi-gas Example", self)
        a.setToolTip("Mix_N2O_H2O_CO_simulated.txt: 500 ppm N2O, 20000 ppm H2O, 30 ppm CO fitted together.")
        a.triggered.connect(self.load_multi_example); hp.addAction(a)
        hp.addSeparator()
        a = QAction("About", self); a.triggered.connect(self.show_about); hp.addAction(a)
        a = QAction("Method && Caveats", self); a.triggered.connect(self.show_method); hp.addAction(a)

    # ------------------------------------------------------------- central --
    def _build_central(self):
        c = QWidget(); self.setCentralWidget(c)
        main = QHBoxLayout(c); main.setContentsMargins(10, 10, 10, 10); main.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(10)
        right = QVBoxLayout(); right.setSpacing(10)
        main.addLayout(left, 58); main.addLayout(right, 42)
        self._build_plot_box(left)
        self._build_action_row(left)
        self._build_inputs_box(right)
        self._build_conditions_box(right)
        self._build_result_tabs(right)

    # -- left column ---------------------------------------------------------
    def _build_plot_box(self, parent_layout):
        box = QGroupBox("Spectrum")
        v = QVBoxLayout(box); v.setContentsMargins(12, 12, 12, 10); v.setSpacing(6)
        # no toolbar: scroll zooms the wavenumber axis, a drag pans it, a double click resets
        self.plot = PlotCanvas(5.0, zoom_pan=True,
                               on_change=lambda _ax: [_autoscale_y(a) for a in self.plot.fig.axes],
                               on_reset=self._redraw)
        self.plot.setToolTip("Scroll to zoom, drag to pan, double-click to reset the view")
        v.addWidget(self.plot, 1)
        row = QHBoxLayout(); row.setSpacing(10)
        self.chk_show = {}
        for key, text, on in (("data", "Data", True), ("fit", "Model", True), ("baseline", "Baseline", False),
                              ("windows", "Segments", False), ("gases", "Each gas", False),
                              ("residual", "Residual", True)):
            cb = QCheckBox(text); cb.setChecked(on); cb.toggled.connect(lambda _c: self._redraw(True))
            row.addWidget(cb); self.chk_show[key] = cb
        row.addStretch(1)
        v.addLayout(row)
        parent_layout.addWidget(box, 1)

    def _build_action_row(self, parent_layout):
        row = QHBoxLayout(); row.setSpacing(8)
        cap = QLabel("Region (cm⁻¹)"); cap.setProperty("muted", True); row.addWidget(cap)
        self.spin_lo = QDoubleSpinBox(); self.spin_hi = QDoubleSpinBox()
        for sp in (self.spin_lo, self.spin_hi):
            sp.setDecimals(4); sp.setRange(0, 1e6); sp.setSingleStep(0.01); sp.setMinimumWidth(120)
            sp.editingFinished.connect(lambda: self._redraw())
        row.addWidget(self.spin_lo); row.addWidget(QLabel("to")); row.addWidget(self.spin_hi)
        b = QPushButton("Use view"); b.setObjectName("secondary")
        b.setToolTip("Set the region to the x range currently shown in the plot.")
        b.clicked.connect(self._region_from_view); row.addWidget(b)
        b = QPushButton("Full"); b.setObjectName("secondary")
        b.clicked.connect(self._region_full); row.addWidget(b)
        row.addSpacing(12)
        self.btn_run = QPushButton("Retrieve Concentration"); self.btn_run.setMinimumHeight(38)
        self.btn_run.setToolTip("Fit the HITRAN x ILS forward model to the data in the region.")
        self.btn_run.clicked.connect(self.run_retrieval)
        row.addWidget(self.btn_run, 1)
        parent_layout.addLayout(row)
        self.lbl_status = QLabel("Open a spectrum first."); self.lbl_status.setProperty("muted", True)
        self.lbl_status.setWordWrap(True)
        parent_layout.addWidget(self.lbl_status)

    # -- right column --------------------------------------------------------
    def _build_inputs_box(self, parent_layout):
        box = QGroupBox("Inputs")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)

        def file_row(r, caption, slot, extra=None, open_text="Open…"):
            cap = QLabel(caption); cap.setProperty("cap", True); g.addWidget(cap, r, 0)
            lbl = QLabel("none"); lbl.setProperty("muted", True); lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            g.addWidget(lbl, r, 1)
            b = QPushButton(open_text); b.setObjectName("secondary"); b.clicked.connect(slot)
            g.addWidget(b, r, 2)
            if extra:
                b2 = QPushButton(extra[0]); b2.setObjectName("secondary"); b2.clicked.connect(extra[1])
                g.addWidget(b2, r, 3)
            return lbl

        self.lbl_spec = file_row(0, "Spectrum", self.open_spectrum)
        self.lbl_gases = file_row(1, "Gases", self.select_gases, open_text="Select…")
        self.lbl_ils = file_row(2, "ILS", self.load_ils)
        cap = QLabel("Data is"); cap.setProperty("muted", True); g.addWidget(cap, 3, 0)
        self.combo_kind = QComboBox()
        self.combo_kind.addItem("Transmittance", "T")
        self.combo_kind.addItem("Absorbance  log₁₀(1/T)", "A10")
        self.combo_kind.addItem("Absorbance  ln(1/T)", "Ae")
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)
        g.addWidget(self.combo_kind, 3, 1, 1, 2)
        g.setColumnStretch(1, 1)
        parent_layout.addWidget(box)

    def _build_conditions_box(self, parent_layout):
        box = QGroupBox("Gas cell")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        s = self._settings
        self.spin_T = QDoubleSpinBox(); self.spin_T.setRange(1, 5000); self.spin_T.setDecimals(2)
        self.spin_T.setSingleStep(0.5); self.spin_T.setValue(float(s["T_K"]))
        self.spin_P = QDoubleSpinBox(); self.spin_P.setRange(0, 1e9); self.spin_P.setDecimals(6)
        self.spin_P.setValue(float(s["P_value"]))
        self.combo_P = QComboBox()
        for u in PRESSURE_UNITS: self.combo_P.addItem(u, u)
        self.combo_P.setCurrentIndex(max(0, self.combo_P.findData(s["P_unit"])))
        self._p_unit = self.combo_P.currentData()
        self.combo_P.currentIndexChanged.connect(self._on_p_unit_changed)
        self.spin_L = QDoubleSpinBox(); self.spin_L.setRange(1e-4, 1e9); self.spin_L.setDecimals(3)
        self.spin_L.setValue(float(s["L_cm"]))
        for r, (name, w, unit, tip) in enumerate((
                ("Temperature", self.spin_T, QLabel("K"), "Gas temperature in the cell."),
                ("Pressure", self.spin_P, self.combo_P, "Total pressure in the cell."),
                ("Path length", self.spin_L, QLabel("cm"), "Optical path length (multipass: the total path)."))):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            g.addWidget(cap, r, 0); g.addWidget(w, r, 1); g.addWidget(unit, r, 2)
        g.setColumnStretch(1, 1)
        parent_layout.addWidget(box)

    def _build_result_tabs(self, parent_layout):
        tabs = QTabWidget(); self._tabs = tabs
        parent_layout.addWidget(tabs, 1)
        self._build_result_tab(tabs)
        self._build_consistency_dialog()
        self._build_ils_tab(tabs)
        self._build_log_tab(tabs)

    def _summary_grid(self, box, rows):
        sg = QGridLayout(box); sg.setContentsMargins(12, 10, 12, 10)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(5)
        mono = QFont("Consolas"); mono.setPointSize(10)
        out = {}
        for r, (name, tip) in enumerate(rows):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            val = QLabel("—"); val.setFont(mono); val.setProperty("cap", True); val.setToolTip(tip)
            val.setWordWrap(True); val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sg.addWidget(cap, r, 0); sg.addWidget(val, r, 1)
            out[name] = val
        sg.setColumnStretch(1, 1)
        return out

    def _build_result_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Result")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        box = QGroupBox("Concentration")
        bv = QVBoxLayout(box); bv.setContentsMargins(12, 12, 12, 10); bv.setSpacing(6)
        # one row per retrieved gas: name, ppm ± u, deviation from a simulated reference
        self.grid_ppm = QGridLayout(); self.grid_ppm.setHorizontalSpacing(14); self.grid_ppm.setVerticalSpacing(4)
        bv.addLayout(self.grid_ppm)
        self.lbl_ppm_sub = QLabel("± one standard error of the fit")
        self.lbl_ppm_sub.setProperty("muted", True); self.lbl_ppm_sub.setAlignment(Qt.AlignCenter)
        bv.addWidget(self.lbl_ppm_sub)
        row = QHBoxLayout(); row.addStretch(1)
        self.btn_details = QPushButton("Details…"); self.btn_details.setObjectName("secondary")
        self.btn_details.setToolTip("Fit parameters, quality and warnings of the current result.")
        self.btn_details.clicked.connect(self._show_details); row.addWidget(self.btn_details)
        row.addStretch(1); bv.addLayout(row)
        v.addWidget(box)
        v.addStretch(1)
        self._set_gas_rows([])

        # the details live in a non-modal dialog, filled by _show_result so it is always current
        self._dlg_details = QDialog(self)
        self._dlg_details.setWindowTitle("Result Details")
        self._dlg_details.setMinimumWidth(560)
        self._dlg_details.setStyleSheet(STYLESHEET)
        dv = QVBoxLayout(self._dlg_details); dv.setContentsMargins(14, 14, 14, 14); dv.setSpacing(10)
        box = QGroupBox("Details")
        self.lbl_res = self._summary_grid(box, [
            ("Number density", "x · N_total, molecule/cm³  (N_total = P / k_B T)."),
            ("Column", "x · N_total · L, molecule/cm²."),
            ("Reference", "The true concentration written in a simulated file's header, and the\n"
                          "deviation of the retrieval from it."),
            ("Other gases", "Multi-gas fit: each other gas's concentration, its deviation from a\n"
                            "simulated reference, and its correlation r with the target.\n"
                            "|r| near 1 means the two cannot be separated in this region."),
            ("Consistency", "Mean ± standard deviation of the segment concentrations\n"
                            "(Retrieval → Consistency Check) - they should agree with the global fit."),
            ("Shift", "Wavenumber offset, spectrum − HITRAN (cm⁻¹)."),
            ("Extra broadening", "Gaussian HWHM added to the ILS (cm⁻¹)."),
            ("Zero offset", "Transmittance zero-level offset z."),
            ("Lorentz scale", "Scale factor on the HITRAN Lorentz widths (1 when not fitted)."),
            ("Baseline", "Polynomial coefficients in the normalised wavenumber (−1..1 over the region)."),
            ("RMS / R²", "√(Σ(data − model)² / N), and the coefficient of determination."),
            ("Peak τ", "Largest optical depth of the intrinsic (un-convolved) model in the region.\n"
                       "τ ≫ 1 means saturated lines: the zero level and the ILS then matter a lot."),
            ("Lines / solver", "HITRAN lines in the model; least_squares status, evaluations, time."),
            ("Warnings", "Parameters that ended on a bound, and other things worth a look."),
        ])
        dv.addWidget(box, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self._dlg_details.close); dv.addWidget(bb)

    def _show_details(self):
        d = self._dlg_details
        d.show(); d.raise_(); d.activateWindow()

    def _set_gas_rows(self, rows):
        """rows: (gas, 'value ± u ppm', reference note or '').  Empty: a dash."""
        g = self.grid_ppm
        while g.count():
            w = g.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if not rows:
            rows = [("", "—", "")]
        big = len(rows) == 1
        # the window stylesheet sets QLabel's font, so the size has to be set the same way
        css = "font-size: %dpx; font-weight: 800; color: #1e3a8a;" % (28 if big else 20)
        for i, (gas, val, note) in enumerate(rows):
            for c, (text, style, align) in enumerate((
                    (gas, css, Qt.AlignRight), (val, css, Qt.AlignLeft), (note, None, Qt.AlignLeft))):
                lab = QLabel(text); lab.setAlignment(align | Qt.AlignVCenter)
                lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
                if style: lab.setStyleSheet(style)
                else: lab.setProperty("muted", True)
                g.addWidget(lab, i, c)
        g.setColumnStretch(0, 1); g.setColumnStretch(2, 1)

    def _build_consistency_dialog(self):
        """Non-modal, filled by _show_result, so it always shows the current result."""
        d = self._dlg_check = QDialog(self)
        d.setWindowTitle("Consistency Check")
        d.resize(760, 640)
        d.setStyleSheet(STYLESHEET)
        v = QVBoxLayout(d); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(8)
        hint = QLabel("Each strong line group refitted on its own, with the shift, broadening and zero fixed "
                      "at the global fit. The segments - weak and saturated lines alike - should agree with "
                      "the global result; that is the evidence the model and the inputs are right. In a "
                      "multi-gas fit every gas has its own segments, and gases that absorb in a segment too "
                      "are refitted with it (Co-fitted).")
        hint.setProperty("muted", True); hint.setWordWrap(True); v.addWidget(hint)
        self.lbl_check = QLabel("—"); self.lbl_check.setWordWrap(True)
        self.lbl_check.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_check)
        self.tbl_win = QTableWidget(0, 9)
        self.tbl_win.setHorizontalHeaderLabels(["#", "Gas", "From", "To", "ppm", "± u", "Peak τ", "RMS",
                                                "Co-fitted"])
        style_table(self.tbl_win)
        self.tbl_win.setMinimumHeight(150)
        v.addWidget(self.tbl_win, 1)
        self.win_plot = PlotCanvas(2.6); self.win_plot.setMinimumHeight(220)
        v.addWidget(self.win_plot, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(d.close); v.addWidget(bb)

    def show_consistency(self):
        d = self._dlg_check
        d.show(); d.raise_(); d.activateWindow()

    def _build_ils_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "ILS")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        self.ils_plot = PlotCanvas(2.4); self.ils_plot.setMinimumHeight(220)
        v.addWidget(self.ils_plot, 1)
        self.lbl_ils_stats = QLabel("—"); self.lbl_ils_stats.setProperty("muted", True)
        self.lbl_ils_stats.setWordWrap(True)
        v.addWidget(self.lbl_ils_stats)

    def _build_log_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Log")
        lv = QVBoxLayout(tab); lv.setContentsMargins(8, 10, 8, 8); lv.setSpacing(8)
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap)
        self.txt_log.setFontFamily("Consolas"); self.txt_log.setFontPointSize(9)
        lv.addWidget(self.txt_log, 1)
        row = QHBoxLayout(); row.addStretch(1)
        b = QPushButton("Clear"); b.setObjectName("secondary"); b.setMaximumWidth(100)
        b.clicked.connect(self.txt_log.clear); row.addWidget(b)
        lv.addLayout(row)

    # ------------------------------------------------------------- helpers --
    def log(self, msg):
        self.txt_log.append("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg))

    def region(self):
        lo, hi = self.spin_lo.value(), self.spin_hi.value()
        return (lo, hi) if hi > lo else None

    def pressure_pa(self):
        return self.spin_P.value() * conc.PRESSURE_UNITS[self.combo_P.currentData()]

    def transmittance(self):
        """The spectrum as transmittance, whatever the file held."""
        if self._spec is None:
            return None
        k = self.combo_kind.currentData()
        y = self._spec.y
        return y if k == "T" else sio.to_transmittance(y, "10" if k == "A10" else "e")

    def _set_busy(self, busy, text=""):
        self._busy = busy
        self.btn_run.setEnabled(not busy)
        if text:
            self.lbl_status.setText(text)
        QApplication.setOverrideCursor(Qt.WaitCursor) if busy else QApplication.restoreOverrideCursor()

    def _remember_dir(self, key, path):
        self._settings[key] = os.path.dirname(path)
        self._save_settings()

    def _save_settings(self):
        s = self._settings
        s["T_K"], s["P_value"], s["P_unit"], s["L_cm"] = (self.spin_T.value(), self.spin_P.value(),
                                                          self.combo_P.currentData(), self.spin_L.value())
        try: save_json(settings_path(), s)
        except Exception as e: self.log("could not save settings: %s" % e)

    def _start(self, name, job, on_done, busy_text):
        if self._busy:
            return
        self._set_busy(True, busy_text)
        self._worker = JobWorker([(Job(name), job)], self)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _on_failed(self, msg):
        self.log("FAILED %s" % msg)
        self.lbl_status.setText("Failed: %s" % msg)
        QMessageBox.warning(self, "Failed", msg)

    # -- settings ------------------------------------------------------------
    def show_settings(self):
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        before = dict(self._settings)
        self._settings = dlg.values()
        self._save_settings()
        changed = sorted(k for k in self._settings if self._settings[k] != before.get(k))
        if changed:
            self.log("settings changed: %s" % ", ".join(changed))

    def _on_p_unit_changed(self, _i):
        """Keep the physical pressure, change only the unit it is shown in."""
        new = self.combo_P.currentData()
        pa = self.spin_P.value() * conc.PRESSURE_UNITS[self._p_unit]
        self._p_unit = new
        self.spin_P.setValue(pa / conc.PRESSURE_UNITS[new])

    # -- region --------------------------------------------------------------
    def _set_region(self, lo, hi):
        self.spin_lo.setValue(lo); self.spin_hi.setValue(hi)

    def _region_from_view(self):
        if self._spec is None or not self.plot.fig.axes:
            return
        lo, hi = self.plot.fig.axes[0].get_xlim()
        self._set_region(max(lo, self._spec.x[0]), min(hi, self._spec.x[-1]))
        self._redraw(True)

    def _region_full(self):
        if self._spec is None:
            return
        self._set_region(self._spec.x[0], self._spec.x[-1])
        self._redraw()

    # -- spectrum ------------------------------------------------------------
    def open_spectrum(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open spectrum", self._settings["last_open_dir"] or HERE,
                                              "Spectra (*.txt *.dat *.csv *.dpt *.0 *.1 *.2 *.3);;All files (*)")
        if path:
            self._remember_dir("last_open_dir", path)
            self.load_spectrum_file(path)

    def load_spectrum_file(self, path):
        try:
            sp = sio.load_spectrum(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", "%s: %s" % (type(e).__name__, e)); return False
        self._spec = sp
        self._result = None
        self.combo_kind.blockSignals(True)
        self.combo_kind.setCurrentIndex(0 if sp.kind == "transmittance" else
                                        (1 if str(self._settings["abs_base"]) == "10" else 2))
        self.combo_kind.blockSignals(False)
        self.lbl_spec.setText("%s  ·  %d points, %.4f–%.4f cm⁻¹" % (sp.name, len(sp.x), sp.x[0], sp.x[-1]))
        self.lbl_spec.setToolTip(path)
        self._reference = read_sim_header(path)
        if self._reference:
            ppm, T, P, L = self._reference
            self.spin_T.setValue(T); self.spin_L.setValue(L)
            self.spin_P.setValue(P / conc.PRESSURE_UNITS[self.combo_P.currentData()])
            self.log("%s: simulated at %g ppm, T %g K, P %g Pa, L %g cm - cell conditions taken from it"
                     % (sp.name, ppm, T, P, L))
        self._ref_gases = read_sim_gases(path)
        if self._ref_gases:
            self.log("%s: simulated mixture %s" % (sp.name, ", ".join(
                "%s %g ppm" % kv for kv in self._ref_gases.items())))
        r = self.region()
        if r is None or r[1] < sp.x[0] or r[0] > sp.x[-1]:
            self._set_region(sp.x[0], sp.x[-1])
        self.log("opened %s (%s, %d points)" % (path, sp.kind, len(sp.x)))
        self._show_result(None)
        self._redraw()
        self._update_status()
        return True

    def _on_kind_changed(self, _i):
        k = self.combo_kind.currentData()
        if k != "T":
            self._settings["abs_base"] = "10" if k == "A10" else "e"
        self._result = None
        self._show_result(None)
        self._redraw(True)

    # -- line list -----------------------------------------------------------
    def open_lines(self):
        start = hitran_fetch.HITRAN_DIR if os.path.isdir(hitran_fetch.HITRAN_DIR) else HERE
        path, _ = QFileDialog.getOpenFileName(self, "Open HITRAN line list", start,
                                              "HITRAN (*.data *.par *.csv *.txt);;All files (*)")
        if path:
            self.load_lines_file(path)

    def load_lines_file(self, path):
        try:
            lines = conc.load_hitran(path)
        except Exception as e:
            QMessageBox.critical(self, "Line list", "%s: %s" % (type(e).__name__, e)); return False
        self._lines, self._lines_path = lines, path
        self._target = gas_name(lines, path)
        mols = sorted(set(int(m) for m in lines["mol"]))
        self._note_list(path, self._target, lines)
        self.log("line list %s: %s, %d lines, molecule id(s) %s" % (path, self._target, len(lines["nu"]), mols))
        self._others = [g for g in self._others if g["name"] != self._target]
        self._show_others()
        self._update_status()
        return True

    # -- other gases (multi-gas fit) -----------------------------------------
    def add_other_gas(self):
        start = hitran_fetch.HITRAN_DIR if os.path.isdir(hitran_fetch.HITRAN_DIR) else HERE
        paths, _ = QFileDialog.getOpenFileNames(self, "Add other gas (HITRAN line list)", start,
                                                "HITRAN (*.data *.par *.csv *.txt);;All files (*)")
        for p in paths:
            self.load_other_gas_file(p)

    def load_other_gas_file(self, path):
        try:
            lines = conc.load_hitran(path)
        except Exception as e:
            QMessageBox.critical(self, "Other gas", "%s: %s" % (type(e).__name__, e)); return False
        name = gas_name(lines, path)
        self._note_list(path, name, lines)
        if self._lines is not None and name == self._target:
            QMessageBox.information(self, "Other gas", "%s is already the target gas." % name); return False
        old = [g for g in self._others if g["name"] == name]
        self._others = [g for g in self._others if g["name"] != name] + [
            {"name": name, "lines": lines, "path": path}]
        self.log("%s gas %s: %s, %d lines" % ("replaced" if old else "added", name, path, len(lines["nu"])))
        self._show_others()
        return True

    def clear_other_gases(self):
        if self._others:
            self._others = []
            self.log("other gases cleared - single-gas fit")
            self._show_others()

    def _show_others(self):
        """The Gases row: chemical formulas only, the files in the tooltip."""
        if self._lines is None:
            self.lbl_gases.setText("none"); self.lbl_gases.setToolTip(""); return
        others = [g["name"] for g in self._others]
        self.lbl_gases.setText("%s  +  %s" % (self._target, ", ".join(others)) if others
                               else "%s  ·  single-gas fit" % self._target)
        self.lbl_gases.setToolTip("\n".join(["Target %s: %s" % (self._target, self._lines_path)]
                                            + ["%s: %s" % (g["name"], g["path"]) for g in self._others]))

    def _note_list(self, path, gas, lines):
        """Remember a line list that is not in Input/HITRAN, so Select Gases lists it."""
        known = scan_line_lists()
        if os.path.normcase(os.path.abspath(path)) not in {os.path.normcase(p) for p in known}:
            self._extra_lists[path] = {"gas": gas, "lo": float(lines["nu"].min()), "hi": float(lines["nu"].max())}

    def _list_info(self, path):
        """Gas and range of a line list picked in Select Gases, or None after an error message."""
        try:
            lines = conc.load_hitran(path)
        except Exception as e:
            QMessageBox.critical(self, "Line list", "%s: %s" % (type(e).__name__, e)); return None
        info = {"gas": gas_name(lines, path), "lo": float(lines["nu"].min()), "hi": float(lines["nu"].max())}
        if os.path.normcase(os.path.dirname(os.path.abspath(path))) != os.path.normcase(hitran_fetch.HITRAN_DIR):
            self._extra_lists[path] = info
        return info

    def select_gases(self):
        lists = scan_line_lists()
        lists.update(self._extra_lists)
        chosen = {g["name"]: g["path"] for g in self._others}
        if self._lines is not None:
            chosen[self._target] = self._lines_path
        dlg = GasDialog(lists, self._target, chosen, self.region(), self._list_info,
                        lambda done: self.download_lines(other=True, on_done=done), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        target, picked = dlg.values()
        same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        if self._lines is None or target != self._target or not same(picked[target], self._lines_path):
            if not self.load_lines_file(picked[target]):
                return
        want = {g: p for g, p in picked.items() if g != target}
        have = {g["name"]: g["path"] for g in self._others}
        if set(want) != set(have) or any(not same(p, have[g]) for g, p in want.items()):
            self._others = [g for g in self._others if g["name"] in want and same(g["path"], want[g["name"]])]
            for g, p in want.items():
                if g not in [o["name"] for o in self._others]:
                    self.load_other_gas_file(p)
            self._show_others()
            self.log("gases: %s" % " + ".join([self._target] + [g["name"] for g in self._others]))

    def download_lines(self, other=False, on_done=None):
        try:
            mols = list(hitran_fetch.molecules())
        except ImportError as e:
            QMessageBox.warning(self, "HITRAN", str(e)); return
        r = self.region()
        lo, hi = (math.floor(r[0]) - 1.0, math.ceil(r[1]) + 1.0) if r else (2214.0, 2221.0)
        key = "dl_other_molecule" if other else "dl_molecule"
        dlg = DownloadDialog(mols, self._settings[key], lo, hi, bool(self._settings["dl_main_iso"]), self)
        if other:
            dlg.setWindowTitle("Download Other Gas")
        if dlg.exec_() != QDialog.Accepted:
            return
        mol, lo, hi, main = dlg.values()
        self._settings[key], self._settings["dl_main_iso"] = mol, main
        self._save_settings()
        self.log("downloading %s %.3f–%.3f cm⁻¹ from HITRANonline…" % (mol, lo, hi))
        self._start("download %s" % mol,
                    lambda: {"path": hitran_fetch.fetch(mol, lo, hi, isos=[1] if main else None)},
                    (lambda _job, r: (self.log("saved %s (%.1f s)" % (r["path"], r["elapsed_s"])),
                                      on_done(r["path"]))) if on_done else
                    self._on_downloaded_other if other else self._on_downloaded,
                    "Downloading %s lines from HITRANonline…" % mol)

    def _on_downloaded(self, _job, r):
        self.log("saved %s (%.1f s)" % (r["path"], r["elapsed_s"]))
        self.load_lines_file(r["path"])

    def _on_downloaded_other(self, _job, r):
        self.log("saved %s (%.1f s)" % (r["path"], r["elapsed_s"]))
        self.load_other_gas_file(r["path"])

    # -- ILS -----------------------------------------------------------------
    def load_ils(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ILS", self._settings["last_open_dir"] or HERE,
                                              "Text (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        try:
            x, y = sio.load_ils(path)
        except Exception as e:
            QMessageBox.critical(self, "ILS", "%s: %s" % (type(e).__name__, e)); return
        self.set_ils(x, y, os.path.basename(path))

    def synthetic_ils(self, kind):
        s = self._settings
        if kind == "gauss":
            v, ok = QInputDialog.getDouble(self, "Gaussian ILS", "FWHM (cm⁻¹):", float(s["gauss_fwhm"]), 1e-5, 10, 5)
            if not ok: return
            s["gauss_fwhm"] = v
            x, y = ils_mod.ils_gauss(v)
            src = "Gaussian, FWHM %g cm⁻¹" % v
        else:
            v, ok = QInputDialog.getDouble(self, "Sinc ILS", "Maximum OPD (cm):", float(s["sinc_mopd_cm"]), 0.1, 1000, 3)
            if not ok: return
            s["sinc_mopd_cm"] = v
            x, y = ils_mod.ils_sinc(v, half_width=float(s["ils_half_width"]))
            src = "ideal sinc, max OPD %g cm" % v
        self._save_settings()
        self.set_ils(x, y, src)

    def set_ils(self, x, y, src):
        self._ils, self._ils_src = (np.asarray(x, float), np.asarray(y, float)), src
        st = ils_mod.ils_stats(*self._ils)
        self.lbl_ils.setText("%s  ·  FWHM %.5f cm⁻¹" % (src, st["fwhm"]))
        self.lbl_ils_stats.setText("FWHM %.5f cm⁻¹ · peak at %+.5f · centroid %+.5f · asymmetry %+.4f · area %.5g"
                                   % (st["fwhm"], st["peak"], st["centroid"], st["asymmetry"], st["area"]))
        self.log("ILS %s (%d points, FWHM %.5f cm⁻¹)" % (src, len(x), st["fwhm"]))
        self._draw_ils()
        self._update_status()

    def _draw_ils(self):
        fig = self.ils_plot.fig; fig.clear()
        ax = fig.add_subplot(111)
        if self._ils is not None:
            ax.plot(self._ils[0], self._ils[1], color=C_DATA, lw=PUB_LW)
            ax.axvline(0.0, color="k", lw=0.6)
        pub_axes(ax, "Offset from line centre [cm$^{-1}$]", "ILS [-]")
        self.ils_plot.draw()

    # -- retrieval -----------------------------------------------------------
    def _missing(self):
        return [n for n, v in (("a spectrum", self._spec), ("a HITRAN line list", self._lines),
                               ("an ILS", self._ils)) if v is None]

    def _update_status(self):
        miss = self._missing()
        self.lbl_status.setText("Still needed: %s." % ", ".join(miss) if miss else
                                "Ready: set the region and the gas cell, then Retrieve Concentration.")

    def run_retrieval(self):
        miss = self._missing()
        if miss:
            QMessageBox.information(self, "Retrieve", "Still needed: %s." % ", ".join(miss)); return
        reg = self.region()
        if reg is None:
            QMessageBox.information(self, "Retrieve", "The region is empty."); return
        T = self.transmittance()
        m = (self._spec.x >= reg[0]) & (self._spec.x <= reg[1])
        if m.sum() < 10:
            QMessageBox.information(self, "Retrieve", "Fewer than 10 points in the region."); return
        x, y = self._spec.x[m].copy(), T[m].copy()
        T_K, P_pa, L_cm = self.spin_T.value(), self.pressure_pa(), self.spin_L.value()
        lines, ils, opts = self._lines, self._ils, fit_options(self._settings)
        target, others = self._target, list(self._others)
        self._save_settings()
        self.log("retrieve %s, %.4f–%.4f cm⁻¹ (%d points), T %g K, P %g Pa, L %g cm, %s"
                 % (self._spec.name, reg[0], reg[1], len(x), T_K, P_pa, L_cm,
                    "gases %s" % " + ".join([target] + [g["name"] for g in others]) if others else target))
        self._run_args = {"T_K": T_K, "P_Pa": P_pa, "L_cm": L_cm, "region": reg,
                          "others_paths": {g["name"]: g["path"] for g in others}}
        self._start("retrieval", lambda: retrieve(x, y, target, lines, others, T_K, P_pa, L_cm, ils, opts),
                    self._on_retrieved, "Fitting %s…" % (
                        "%d gases together" % (1 + len(others)) if others else "the forward model"))

    def _on_retrieved(self, _job, r):
        r.update(self._run_args)
        r["spectrum"], r["lines_path"], r["ils_src"] = self._spec.path, self._lines_path, self._ils_src
        self._result = r
        self._show_result(r)
        self._redraw(True)
        if r["multi"]:
            self.log("→ %s   shift %+.5f   rms %.3g   (%s, %d evaluations, %.2f s)"
                     % ("   ".join("%s %s ppm" % (g, fmt(r["gas_ppm"][g], r["gas_err"][g])) for g in r["gases"]),
                        r["shift"], r["rms"], r["message"], r["nfev"], r["elapsed_s"]))
        else:
            self.log("→ %s ppm   shift %+.5f   rms %.3g   (%s, %d evaluations, %.2f s)"
                     % (fmt(r["ppm"], r["ppm_err"]), r["shift"], r["rms"], r["message"], r["nfev"], r["elapsed_s"]))
        self.lbl_status.setText("Done in %.2f s: %s ppm." % (r["elapsed_s"], fmt(r["ppm"], r["ppm_err"])))
        self.statusBar().showMessage("Retrieved %s ppm" % fmt(r["ppm"], r["ppm_err"]))

    def _ref_ppm(self, gas):
        """The simulated (true) ppm of a gas from the spectrum's header, else None."""
        if self._ref_gases:
            return self._ref_gases.get(gas)
        if self._reference and gas == self._target:
            return self._reference[0]
        return None

    def _show_result(self, r):
        L = self.lbl_res
        if r is None:
            self._set_gas_rows([])
            for w in L.values(): w.setText("—")
            self.lbl_check.setText("—")
            self.tbl_win.setRowCount(0)
            self._draw_windows(None)
            return
        gases = r["gases"] if r.get("multi") else [r.get("target") or ""]
        rows = []
        for g in gases:
            v, e = (r["gas_ppm"][g], r["gas_err"][g]) if r.get("multi") else (r["ppm"], r["ppm_err"])
            ref = self._ref_ppm(g)
            rows.append((g, "%s ppm" % fmt(v, e, digits=6),
                         "ref %g  (%+.2f %%)" % (ref, 100 * (v - ref) / ref) if ref else ""))
        self._set_gas_rows(rows)
        x = r["ppm"] * 1e-6
        L["Number density"].setText("%.5g molecule/cm³" % (x * r["n_total"]))
        L["Column"].setText("%.5g molecule/cm²" % (x * r["n_total"] * r["L_cm"]))
        ref = self._ref_ppm(r.get("target"))
        L["Reference"].setText("%g ppm  →  %+.3f %%" % (ref, 100 * (r["ppm"] - ref) / ref) if ref else "—")
        if r.get("multi"):
            t = r["target"]
            rows = []
            for k, g in enumerate(r["gases"]):
                if g == t:
                    continue
                v, e = r["gas_ppm"][g], r["gas_err"][g]
                ref = self._ref_ppm(g)
                rows.append("%s  %s ppm%s  ·  r = %+.2f" % (
                    g, fmt(v, e, digits=6), "  (ref %g, %+.3f %%)" % (ref, 100 * (v - ref) / ref) if ref else "",
                    r["corr"][0, k]))
            L["Other gases"].setText("\n".join(rows))
        else:
            L["Other gases"].setText("none (single-gas fit)")
        wins = r.get("windows") or []
        spread = self._segment_spread(r)
        L["Consistency"].setText("\n".join(spread) if spread else "off (Retrieval Settings)")
        self.lbl_check.setText("<br>".join(spread) if spread else "No segments - the check is off in "
                               "Retrieval Settings, or no line group is strong enough.")
        free = set(r["free"])
        L["Shift"].setText(fmt(r["shift"], r["shift_err"]) + ("" if "shift" in free else "  (fixed)"))
        L["Extra broadening"].setText(fmt(r["broadening"], r["broadening_err"])
                                      + ("" if "broadening" in free else "  (fixed)"))
        L["Zero offset"].setText(fmt(r["zero"], r["zero_err"]) + ("" if "zero" in free else "  (fixed)")
                                 + ("  · %s" % r["zero_mode"] if r.get("zero_mode") else ""))
        L["Lorentz scale"].setText(fmt(r["lorentz_scale"], r["lorentz_scale_err"])
                                   + ("" if "lorentz_scale" in free else "  (fixed)"))
        L["Baseline"].setText("  ".join("%.6g" % b for b in r["baseline"]))
        L["RMS / R²"].setText("%.4g  /  %.6f" % (r["rms"], r["r2"]))
        if r.get("multi"):
            L["Peak τ"].setText("  ".join("%s %.3g" % kv for kv in r["gas_tau"].items()))
            L["Lines / solver"].setText("%s lines · status %d, %d evaluations, %.2f s" % (
                " + ".join("%d %s" % (n, g) for g, n in r["gas_lines"].items()),
                r["status"], r["nfev"], r["elapsed_s"]))
        else:
            L["Peak τ"].setText("%.3g" % r["tau_max"])
            L["Lines / solver"].setText("%d lines · status %d, %d evaluations, %.2f s"
                                        % (r["n_lines"], r["status"], r["nfev"], r["elapsed_s"]))
        warn = []
        if r["at_bound"]:
            warn.append("on a bound: %s" % ", ".join(r["at_bound"]))
        if r["status"] <= 0:
            warn.append("solver did not converge: %s" % r["message"])
        if r["tau_max"] > 5 and "zero" not in free:
            warn.append("saturated lines with the zero level fixed")
        if r["tau_max"] < rt.AUTO_ZERO_TAU and "zero" in free:
            warn.append("zero level fitted with only optically thin lines - it trades against the "
                        "concentration scale; set it to Auto or Fixed")
        if r.get("multi"):
            k = len(r["gases"])
            bad = ["%s/%s" % (r["gases"][i], r["gases"][j]) for i in range(k) for j in range(i + 1, k)
                   if abs(r["corr"][i, j]) > 0.9]
            if bad:
                warn.append("strongly correlated (|r| > 0.9): %s" % ", ".join(bad))
        L["Warnings"].setText("; ".join(warn) if warn else "none")
        self.tbl_win.setRowCount(len(wins))
        for i, w in enumerate(wins):
            for c, t in enumerate(("%d" % (i + 1), w.get("gas", r.get("target", "")), "%.4f" % w["lo"],
                                   "%.4f" % w["hi"], "%.5g" % w["ppm"], "%.2g" % w["ppm_err"],
                                   "%.3g" % w["tau_max"], "%.3g" % w["rms"], ", ".join(w.get("cofit", [])))):
                self.tbl_win.setItem(i, c, table_cell(t))
        self._draw_windows(r)

    @staticmethod
    def _segment_spread(r):
        """One line per gas: mean ± sd of its segments against the global fit."""
        wins = r.get("windows") or []
        gases = r["gases"] if r.get("multi") else [r.get("target", "")]
        out = []
        for g in gases:
            ws = [q for q in wins if q.get("gas", g) == g]
            if not ws:
                continue
            w = np.array([q["ppm"] for q in ws])
            glob = r["gas_ppm"][g] if r.get("multi") else r["ppm"]
            dev = 100 * (w.mean() - glob) / glob if glob else float("nan")
            if len(w) == 1:                     # no spread from one segment: its own uncertainty
                out.append("%s%s ppm  (1 segment, %+.2f %% from the fit)" % (
                    g + "  " if g else "", fmt(ws[0]["ppm"], ws[0]["ppm_err"], digits=5), dev))
                continue
            sd = w.std(ddof=1)
            out.append("%s%.5g ± %.2g ppm  (%d segments, spread %.2f %%, mean %+.2f %% from the fit)" % (
                g + "  " if g else "", w.mean(), sd, len(w), 100 * sd / w.mean() if w.mean() else float("nan"),
                dev))
        return out

    def _draw_windows(self, r):
        fig = self.win_plot.fig; fig.clear()
        ax = fig.add_subplot(111)
        wins = (r or {}).get("windows") or []
        if wins and r.get("multi"):
            # gases at ppm and at per cent: each segment as its deviation from its gas's global fit
            ax.axhline(0.0, color=C_FIT, lw=PUB_FIT_LW, label="Global fit")
            for k, g in enumerate(r["gases"]):
                ws = [w for w in wins if w["gas"] == g]
                if not ws:
                    continue
                ref = r["gas_ppm"][g]
                ax.errorbar([0.5 * (w["lo"] + w["hi"]) for w in ws],
                            [100 * (w["ppm"] - ref) / ref for w in ws],
                            yerr=[100 * w["ppm_err"] / ref for w in ws], fmt="o", ms=PUB_MARKER,
                            color=SERIES_COLOURS[(k + 2) % len(SERIES_COLOURS)], capsize=2, lw=PUB_FIT_LW,
                            label=g)
            pub_axes(ax, "Segment centre [cm$^{-1}$]", "Deviation from global fit [%]")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.4 * (hi - lo))
            pub_legend(ax, loc="upper center", ncol=len(r["gases"]) + 1)
        elif wins:
            c = [0.5 * (w["lo"] + w["hi"]) for w in wins]
            ax.axhline(r["ppm"], color=C_FIT, lw=PUB_FIT_LW, label="Global fit")
            ax.axhspan(r["ppm"] - r["ppm_err"], r["ppm"] + r["ppm_err"], color=C_FIT, alpha=0.12, lw=0)
            sat = np.array([w["tau_max"] > 3 for w in wins])
            for mask, col, lab in ((~sat, SERIES_COLOURS[0], "Segment, peak τ ≤ 3"),
                                   (sat, SERIES_COLOURS[2], "Segment, peak τ > 3")):
                if mask.any():
                    ax.errorbar(np.array(c)[mask], np.array([w["ppm"] for w in wins])[mask],
                                yerr=np.array([w["ppm_err"] for w in wins])[mask], fmt="o", ms=PUB_MARKER,
                                color=col, capsize=2, lw=PUB_FIT_LW, label=lab)
            if self._reference:
                ax.axhline(self._reference[0], color=INK_SECONDARY, lw=PUB_FIT_LW, ls="--", label="Reference")
            pub_axes(ax, "Segment centre [cm$^{-1}$]", "Concentration [ppm]")
            lo, hi = ax.get_ylim()                  # headroom so the legend sits above the points
            ax.set_ylim(lo, hi + 0.6 * (hi - lo))
            pub_legend(ax, loc="upper center", ncol=2)
        else:
            ax.text(0.5, 0.5, "No segments" if r else "—", transform=ax.transAxes,
                    ha="center", va="center", color=INK_MUTED, fontsize=PUB_FONT + 1)
            ax.set_xticks([]); ax.set_yticks([])
        self.win_plot.draw()

    # -- drawing -------------------------------------------------------------
    def _redraw(self, keep_view=False):
        keep = None
        if keep_view and self.plot.fig.axes and self._spec is not None:
            keep = self.plot.fig.axes[0].get_xlim()
        show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
        x = self._spec.x if self._spec is not None else None
        build_main_figure(self.plot.fig, x, self.transmittance(), self.region(), self._result, show, keep_xlim=keep)
        self.plot.draw()

    # -- example -------------------------------------------------------------
    def load_example(self):
        p = example_paths()
        if p is None:
            QMessageBox.information(self, "Example", "The example files are not in Input/."); return
        lo, hi = EXAMPLE["region"]
        if not self.load_spectrum_file(p["spectrum"]):
            return
        self.load_lines_file(p["lines"])
        self.clear_other_gases()
        x, y = sio.load_ils(p["ils"]); self.set_ils(x, y, os.path.basename(p["ils"]))
        self._set_region(lo, hi)
        self._redraw()
        self.run_retrieval()

    def load_multi_example(self):
        p = example_paths(EXAMPLE_MULTI)
        if p is None:
            QMessageBox.information(self, "Multi-gas example",
                                    "The example files are not in Input/ - run  python multigas_example.py  "
                                    "once to download the lines and simulate the spectrum."); return
        if not self.load_spectrum_file(p["spectrum"]):
            return
        self.load_lines_file(p["lines"])
        self.clear_other_gases()
        for q in p["others"]:
            self.load_other_gas_file(q)
        x, y = sio.load_ils(p["ils"]); self.set_ils(x, y, os.path.basename(p["ils"]))
        self._set_region(*EXAMPLE_MULTI["region"])
        self._redraw()
        self.run_retrieval()

    # -- saving --------------------------------------------------------------
    def save_result(self):
        r = self._result
        if r is None:
            QMessageBox.information(self, "Save", "Retrieve a concentration first."); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save result", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "concentration_%s.txt" % stamp),
            "Text (*.txt);;All files (*)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        rows = [("date", datetime.now().isoformat(timespec="seconds")),
                ("spectrum", r["spectrum"]), ("line_list", r["lines_path"]), ("ils", r["ils_src"]),
                ("region_cm-1", "%.6f %.6f" % r["region"]),
                ("T_K", "%.6g" % r["T_K"]), ("P_Pa", "%.8g" % r["P_Pa"]), ("L_cm", "%.6g" % r["L_cm"]),
                ("ppm", "%.8g" % r["ppm"]), ("ppm_err", "%.4g" % r["ppm_err"]),
                ("number_density_cm-3", "%.6g" % (r["ppm"] * 1e-6 * r["n_total"])),
                ("column_cm-2", "%.6g" % (r["ppm"] * 1e-6 * r["n_total"] * r["L_cm"])),
                ("shift_cm-1", "%.6g ± %.2g" % (r["shift"], r["shift_err"])),
                ("broadening_cm-1", "%.6g ± %.2g" % (r["broadening"], r["broadening_err"])),
                ("zero", "%.6g ± %.2g" % (r["zero"], r["zero_err"])),
                ("lorentz_scale", "%.6g ± %.2g" % (r["lorentz_scale"], r["lorentz_scale_err"])),
                ("baseline", " ".join("%.8g" % b for b in r["baseline"])),
                ("rms", "%.6g" % r["rms"]), ("r2", "%.8f" % r["r2"]), ("tau_max", "%.4g" % r["tau_max"]),
                ("n_lines", "%d" % r["n_lines"]), ("free", " ".join(r["free"])),
                ("at_bound", " ".join(r["at_bound"]) or "none"), ("solver", r["message"])]
        if r.get("multi"):
            rows.insert(3, ("target_gas", r["target"]))
            rows.append(("fit", "multi-gas: %s" % " + ".join(r["gases"])))
            for k, g in enumerate(r["gases"]):
                rows.append(("ppm_%s" % g, "%.8g ± %.4g" % (r["gas_ppm"][g], r["gas_err"][g])))
                rows.append(("tau_max_%s" % g, "%.4g" % r["gas_tau"][g]))
                if g != r["target"]:
                    rows.append(("line_list_%s" % g, r["others_paths"].get(g, "")))
                    rows.append(("corr_%s_%s" % (r["target"], g), "%+.4f" % r["corr"][0, k]))
        if r.get("zero_mode"):
            rows.append(("zero_mode", r["zero_mode"]))
        ref = self._ref_ppm(r.get("target"))
        if ref:
            rows.append(("reference_ppm", "%g" % ref))
        for g, v in self._ref_gases.items():
            rows.append(("reference_ppm_%s" % g, "%g" % v))
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for k, v in rows:
                    fh.write("%-22s %s\n" % (k, v))
                wins = r.get("windows") or []
                if wins:
                    fh.write("\n# consistency check: segment  gas  lo_cm-1  hi_cm-1  ppm  ppm_err  tau_max  rms"
                             "  co-fitted\n")
                    for i, w in enumerate(wins):
                        fh.write("%d\t%s\t%.6f\t%.6f\t%.8g\t%.4g\t%.4g\t%.4g\t%s\n"
                                 % (i + 1, w.get("gas", r.get("target", "")), w["lo"], w["hi"], w["ppm"],
                                    w["ppm_err"], w["tau_max"], w["rms"], ",".join(w.get("cofit", [])) or "-"))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("saved %s" % path)
        self.statusBar().showMessage("Saved %s" % os.path.basename(path))

    def export_curves(self):
        r = self._result
        if r is None:
            QMessageBox.information(self, "Export", "Retrieve a concentration first."); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export curves", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "concentration_curves_%s.txt" % stamp),
            "Text (*.txt);;All files (*)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            comps = r.get("components") or {}
            what = ("  ".join("%s %s ppm" % (g, fmt(r["gas_ppm"][g], r["gas_err"][g])) for g in r["gases"])
                    if r.get("multi") else "%s ppm" % fmt(r["ppm"], r["ppm_err"]))
            sio.save_columns(path, [r["x"], r["data"], r["fit"], r["baseline_curve"], r["residual"]]
                             + list(comps.values()),
                             header="%s  %s\nwavenumber_cm-1\tdata_T\tmodel_T\tbaseline\tresidual%s"
                                    % (os.path.basename(r["spectrum"]), what,
                                       "".join("\tmodel_T_%s_only" % g for g in comps)))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("exported %s" % path)

    def save_plot(self):
        if self._spec is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "concentration_plot_%s.png" % stamp),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            # the current view, drawn again at a fixed size in the same publication style
            fig = pub_figure(1000, 650)
            show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
            build_main_figure(fig, self._spec.x, self.transmittance(), self.region(), self._result, show,
                              keep_xlim=self.plot.fig.axes[0].get_xlim())
            pub_savefig(fig, path, dpi=600 if path.lower().endswith(".png") else 300)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("saved %s" % path)

    # -- help ----------------------------------------------------------------
    def show_about(self):
        mb = QMessageBox(self)
        mb.setWindowTitle("About")
        mb.setTextFormat(Qt.RichText)
        mb.setTextInteractionFlags(Qt.TextBrowserInteraction)     # the paper link opens in the browser
        mb.setText((
            "<b>%s</b><br><br>"
            "Gas mixing ratio from an FTIR transmittance spectrum by fitting a HITRAN "
            "line-by-line model convolved with the instrument line shape.<br><br>"
            "Developed inspired by:<br>"
            "Trisna, B. A., et al.: Measurement report: Radiative efficiencies of "
            "(CF<sub>3</sub>)<sub>2</sub>CFCN, CF<sub>3</sub>OCFCF<sub>2</sub>, and "
            "CF<sub>3</sub>OCF<sub>2</sub>CF<sub>3</sub>, Atmos. Chem. Phys., 23, 4489, 2023, "
            "<a href=\"https://acp.copernicus.org/articles/23/4489/2023/\">"
            "https://acp.copernicus.org/articles/23/4489/2023/</a><br><br>"
            "<b>If you use this code for a publication, you must cite this paper.</b><br><br>"
            "Modules: retrieval · concentration · hitran_fetch · ils · spectrum_io.<br>"
            "Line data and partition sums: HITRAN / HAPI.<br>"
        ) % APP_TITLE)
        mb.exec_()

    def show_method(self):
        QMessageBox.information(self, "Method && Caveats", (
            "<b>Forward model</b><br>"
            "τ(ν) = x·N<sub>total</sub>·L·Σ S<sub>j</sub>(T)·V<sub>j</sub>(ν − shift), with HITRAN "
            "intensities, air/self broadening and pressure shifts, Voigt profiles, "
            "N<sub>total</sub> = P / k<sub>B</sub>T.<br>"
            "T<sub>model</sub> = baseline · [(1 − z)·(e<sup>−τ</sup> ⊛ ILS ⊛ G<sub>w</sub>) + z]<br><br>"
            "<b>Several gases</b><br>"
            "With other gases added, τ = Σ<sub>k</sub> x<sub>k</sub>·τ<sub>k</sub>: one mixing ratio per "
            "gas, with the shift, broadening, zero, Lorentz scale and baseline shared. Leaving out a gas "
            "whose lines overlap the target's biases the target. The correlation r between gases tells "
            "whether the region separates them.<br><br>"
            "<b>Why fit instead of integrate</b><br>"
            "Lines that are black at their centre lose their core to the ILS; integrating the "
            "measured absorbance then reads low and no deconvolution recovers it. The non-linear "
            "Beer–Lambert law is in the model, so saturation is handled exactly.<br><br>"
            "<b>Inputs that matter</b><br>"
            "x scales as 1/(N<sub>total</sub>·L): the pressure, temperature and path length enter "
            "the answer directly. The ILS and the zero level matter most for saturated lines "
            "(peak τ ≫ 1). With only thin lines (peak τ < 1) the zero level cannot be told apart from "
            "the concentration scale; the default Auto setting then keeps it at 0.<br><br>"
            "<b>Uncertainty</b><br>"
            "± is one standard error from the fit residuals only - not the uncertainty of T, P, L, "
            "the HITRAN intensities or the ILS. The spread of the segment values (Retrieval → "
            "Consistency Check) is a better guide to the model error."))

    def closeEvent(self, e):
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
        except RuntimeError:
            pass
        self._save_settings()
        super().closeEvent(e)


# =============================================================================
# Headless self-test
# =============================================================================
def self_test():
    """The retrieval on the worked example, whose true answer is in its header."""
    p = example_paths()
    if p is None:
        print("example files missing"); return None
    sp = sio.load_spectrum(p["spectrum"])
    ppm_true, T, P, L = read_sim_header(p["spectrum"])
    lo, hi = EXAMPLE["region"]
    m = (sp.x >= lo) & (sp.x <= hi)
    r = rt.fit_spectrum(sp.x[m], sp.y[m], conc.load_hitran(p["lines"]), T, P, L, sio.load_ils(p["ils"]),
                        fit_options(DEFAULTS))
    print("example: true %g ppm   retrieved %s ppm (%+.3f %%)   rms %.3g   %d segments   %.2f s"
          % (ppm_true, fmt(r["ppm"], r["ppm_err"]), 100 * (r["ppm"] - ppm_true) / ppm_true, r["rms"],
             len(r.get("windows") or []), r["elapsed_s"]))
    q = example_paths(EXAMPLE_MULTI)
    if q is None:
        print("multi-gas example files missing (python multigas_example.py)"); return r
    sp = sio.load_spectrum(q["spectrum"])
    true = read_sim_gases(q["spectrum"])
    _ppm, T, P, L = read_sim_header(q["spectrum"])
    lo, hi = EXAMPLE_MULTI["region"]
    m = (sp.x >= lo) & (sp.x <= hi)
    lines = conc.load_hitran(q["lines"])
    others = []
    for path in q["others"]:
        ln = conc.load_hitran(path)
        others.append({"name": gas_name(ln, path), "lines": ln, "path": path})
    rm = retrieve(sp.x[m], sp.y[m], gas_name(lines, q["lines"]), lines, others, T, P, L,
                  sio.load_ils(q["ils"]), fit_options(DEFAULTS))
    print("multi-gas example: %s   rms %.3g   zero %s   %d segments   %.2f s" % (
        "   ".join("%s %s ppm (%+.3f %%)" % (g, fmt(rm["gas_ppm"][g], rm["gas_err"][g]),
                                            100 * (rm["gas_ppm"][g] - true[g]) / true[g]) for g in rm["gases"]),
        rm["rms"], rm.get("zero_mode"), len(rm.get("windows") or []), rm["elapsed_s"]))
    return r


def main():
    if "--selftest" in sys.argv:
        self_test(); return
    app = QApplication(sys.argv)
    w = ConcentrationWindow(); w.show()
    files = [p for p in sys.argv[1:] if os.path.isfile(p)]
    if files:
        w.load_spectrum_file(files[0])
    elif "--no-example" not in sys.argv and w._settings.get("example_on_start", True):
        QTimer.singleShot(0, w.load_example)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
