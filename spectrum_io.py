# -*- coding: utf-8 -*-
"""Reading and writing spectra, instrument line shapes and par0 files.

What the MATLAB scripts did with load('...txt') happens here, plus a reader for
the Bruker OPUS binary files (*.dpt / *.0 / *.1 ...) the spectra were exported
from, so they no longer have to go through a text export first.

    load_spectrum(path)       -> Spectrum(x, y, name, path, kind)
    load_ils(path)            -> (offset_cm1, ils)   two columns, e.g. ILS_LINEFIT.txt
    load_linefit_params(path) -> (modulation, phase)  LINEFIT ilsparms.dat
    load_par0(path) / save_par0(path, par0)            the 4 x g MATLAB par0 text format
    to_absorbance / to_transmittance                   with a selectable log base
"""

import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    x: np.ndarray                 # wavenumber, increasing
    y: np.ndarray                 # as read
    name: str = ""
    path: str = ""
    kind: str = "transmittance"   # "transmittance" or "absorbance"
    meta: dict = field(default_factory=dict)


# =============================================================================
# Absorbance <-> transmittance
# =============================================================================
def to_absorbance(T, base="10"):
    """A = log_base(1/T).  voigtfit_test.m uses log10; the 2020 deconvolution
    script uses the natural log - so the base is a choice, not a constant."""
    T = np.clip(np.asarray(T, dtype=float), 1e-12, None)
    return -np.log10(T) if str(base) == "10" else -np.log(T)


def to_transmittance(A, base="10"):
    A = np.asarray(A, dtype=float)
    return 10.0 ** (-A) if str(base) == "10" else np.exp(-A)


def guess_kind(y):
    """Transmittance sits in roughly [0, 1.2] with its baseline near 1;
    absorbance has its baseline near 0."""
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    return "transmittance" if 0.5 < med < 1.5 and np.nanmin(y) > -0.2 else "absorbance"


# =============================================================================
# Text
# =============================================================================
_NUM = re.compile(r"^[\s]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def load_columns(path, min_cols=2):
    """All numeric rows of a text file as a float array.

    Comment lines (#, %, ;), header lines and blank lines are skipped;
    separators may be whitespace, tab, comma or semicolon.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#%;!" or not _NUM.match(s):
                continue
            parts = re.split(r"[\s,;]+", s)
            try:
                vals = [float(p) for p in parts if p != ""]
            except ValueError:
                continue
            if len(vals) >= min_cols:
                rows.append(vals[:max(min_cols, len(vals))])
    if not rows:
        raise ValueError("no numeric data in %s" % path)
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def is_opus(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x0a\x0a\xfe\xfe"
    except OSError:
        return False


def load_spectrum(path, kind="auto", block=None):
    """Spectrum from a two-column text file or a Bruker OPUS binary file.

    The x axis is sorted increasing (fit2voigt.m did the same) and duplicate
    points are dropped.  kind = "auto" guesses transmittance/absorbance.
    """
    meta = {}
    if is_opus(path):
        blocks = read_opus(path)
        if not blocks:
            raise ValueError("no spectral data block found in %s" % path)
        b = blocks[block] if block is not None else blocks[-1]
        x, y = b["x"], b["y"]
        meta = {"opus_block": b["type"], "opus_blocks": [bb["type"] for bb in blocks]}
    else:
        d = load_columns(path)
        x, y = d[:, 0], d[:, 1]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if kind == "auto":
        kind = guess_kind(y)
    return Spectrum(x=x, y=y, name=os.path.basename(path), path=path, kind=kind, meta=meta)


def save_columns(path, cols, header=""):
    arr = np.column_stack(cols)
    np.savetxt(path, arr, delimiter="\t", fmt="%.10g", header=header, comments="# ")


# =============================================================================
# Bruker OPUS binary
# =============================================================================
def _opus_params(d, off, length_bytes):
    """A parameter block: NAME\\0, type (int16), size in words (int16), value."""
    out = {}
    p, end = off, min(len(d), off + length_bytes)
    while p + 8 <= end:
        name = d[p:p + 3].decode("latin-1", errors="replace")
        if name == "END" or not name.isalnum():
            break
        typ, size = struct.unpack("<hh", d[p + 4:p + 8])
        raw = d[p + 8:p + 8 + 2 * size]
        if typ == 0 and len(raw) >= 4:
            val = struct.unpack("<i", raw[:4])[0]
        elif typ == 1 and len(raw) >= 8:
            val = struct.unpack("<d", raw[:8])[0]
        else:
            val = raw.split(b"\x00", 1)[0].decode("latin-1", errors="replace")
        out[name] = val
        p += 8 + 2 * size
    return out


def read_opus(path):
    """Every spectral data block of an OPUS file, as a list of dicts with x, y.

    The directory at offset 24 lists blocks as (type bytes, length in 4-byte
    words, byte offset).  A data block's parameters ('data status' block: NPT,
    FXV, LXV, CSF) carry the same type bytes with 16 added to the first one.
    """
    d = open(path, "rb").read()
    if d[:4] != b"\x0a\x0a\xfe\xfe":
        raise ValueError("%s is not an OPUS file" % path)
    p_dir, _max, n = struct.unpack("<iii", d[12:24])
    entries = []
    for i in range(n):
        e = d[p_dir + 12 * i: p_dir + 12 * i + 12]
        if len(e) < 12:
            break
        t = struct.unpack("<BBBB", e[:4])
        ln, off = struct.unpack("<ii", e[4:])
        entries.append((t, ln * 4, off))

    params = {}
    for t, nb, off in entries:
        if d[off:off + 3].isalpha() and d[off + 3:off + 4] == b"\x00":
            params[t] = _opus_params(d, off, nb)

    out = []
    for t, nb, off in entries:
        if t in params or t[0] == 0:
            continue
        status = params.get((t[0] + 16, t[1], t[2], t[3]))
        if status is None or "NPT" not in status:
            continue
        npt = int(status["NPT"])
        if npt <= 1 or nb < 4 * npt:
            continue
        y = np.frombuffer(d, dtype="<f4", count=npt, offset=off).astype(float)
        y = y * float(status.get("CSF", 1.0) or 1.0)
        x = np.linspace(float(status["FXV"]), float(status["LXV"]), npt)
        out.append({"type": t, "x": x, "y": y, "params": status})
    return out


# =============================================================================
# Instrument line shape
# =============================================================================
def load_ils(path):
    """ILS as (offset from line centre in cm-1, value), e.g. ILS_LINEFIT.txt."""
    d = load_columns(path)
    x, y = d[:, 0], d[:, 1]
    o = np.argsort(x)
    return x[o], y[o]


def load_linefit_params(path):
    """LINEFIT ilsparms.dat: modulation efficiency and phase error (rad)
    at equidistant optical path differences from 0 to MOPD."""
    d = load_columns(path)
    return d[:, 0], d[:, 1]


# =============================================================================
# par0 - the MATLAB initial-parameter file
# =============================================================================
def load_par0(path):
    """4 x g matrix: positions, intensities, Gaussian widths, Lorentzian widths."""
    d = load_columns(path, min_cols=1)
    if d.shape[0] == 4:
        return d
    if d.shape[1] == 4:                    # one line per row instead
        return d.T
    raise ValueError("par0 must have 4 rows (or 4 columns): got %s" % (d.shape,))


def save_par0(path, par0):
    np.savetxt(path, np.asarray(par0, dtype=float), fmt="%15.7e", delimiter="\t")
