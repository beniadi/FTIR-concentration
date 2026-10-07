# -*- coding: utf-8 -*-
"""Instrument line shape (ILS): building it, sampling it, convolving with it.

    ils_kernel(ils_x, ils_y, dx)      the ILS on a grid of spacing dx, centred,
                                      normalised to unit sum (area preserving)
    convolve(y, kernel)               edge-padded discrete convolution
    ils_from_linefit_params(...)      ILS from LINEFIT modulation/phase
    ils_sinc / ils_gauss              synthetic ILS when none was measured
    ils_stats(x, y)                   FWHM, centroid, area, asymmetry

THE CONVOLUTION DOMAIN
----------------------
An FTS records T_meas = T_true (*) ILS - the convolution acts on the
TRANSMITTANCE, which is what retrieval.py and simulate_spectrum.py do.
Convolving the absorbance instead is only the weak-line approximation.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
# Sampling and convolution
# =============================================================================
def is_uniform(x, rtol=1e-2):
    d = np.diff(np.asarray(x, dtype=float))
    return len(d) > 0 and np.all(d > 0) and np.ptp(d) <= rtol * abs(np.mean(d)) + 1e-15


def ils_kernel(ils_x, ils_y, dx, normalize=True, centre="peak"):
    """The ILS sampled at k*dx, k = -M..M, zero outside the tabulated range.

    centre = "peak" puts the ILS maximum at k = 0 (a measured ILS is often
    tabulated with a small offset), "zero" keeps the tabulated origin and
    "centroid" uses the first moment.  Returns (offsets, kernel).
    """
    ils_x = np.asarray(ils_x, dtype=float)
    ils_y = np.asarray(ils_y, dtype=float)
    if centre == "peak":
        c = _peak_position(ils_x, ils_y)
    elif centre == "centroid":
        c = _trapz(ils_x * ils_y, ils_x) / _trapz(ils_y, ils_x)
    else:
        c = 0.0
    xs = ils_x - c
    M = int(np.floor(min(abs(xs[0]), abs(xs[-1])) / dx))
    M = max(M, 1)
    k = np.arange(-M, M + 1) * dx
    ker = CubicSpline(xs, ils_y)(k)
    ker[(k < xs[0]) | (k > xs[-1])] = 0.0
    if normalize:
        s = ker.sum()
        if s != 0:
            ker = ker / s
    return k, ker


def _peak_position(x, y):
    """Sub-sample position of the maximum (parabola through the top 3)."""
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return x[i] + 0.5 * (y0 - y2) / den * (x[i + 1] - x[i - 1]) / 2
    return x[i]


def convolve(y, kernel, pad="edge"):
    """y (*) kernel, same length as y.  The ends are padded so a baseline
    does not droop where the kernel runs off the data."""
    y = np.asarray(y, dtype=float)
    M = len(kernel) // 2
    yp = np.pad(y, M, mode=pad) if pad else np.pad(y, M)
    return fftconvolve(yp, kernel, mode="same")[M:M + len(y)]


def resample_uniform(x, y, n=None):
    """y on a uniform grid spanning x (cubic spline)."""
    x = np.asarray(x, dtype=float)
    n = n or len(x)
    xu = np.linspace(x[0], x[-1], n)
    return xu, CubicSpline(x, y)(xu)


# =============================================================================
# Synthetic and LINEFIT-derived ILS
# =============================================================================
def ils_sinc(mopd_cm, half_width=0.25, dnu=None):
    """Ideal unapodised FTS: ILS(nu) = 2L sinc(2 nu L), L = maximum OPD (cm)."""
    dnu = dnu or 1.0 / (2 * mopd_cm) / 40.0
    x = np.arange(-half_width, half_width + dnu / 2, dnu)
    y = 2 * mopd_cm * np.sinc(2 * x * mopd_cm)
    return x, y / _trapz(y, x)


def ils_gauss(fwhm, half_width=None, dnu=None):
    """Gaussian ILS of the given FWHM (cm-1)."""
    half_width = half_width or 5 * fwhm
    dnu = dnu or fwhm / 40.0
    x = np.arange(-half_width, half_width + dnu / 2, dnu)
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = np.exp(-0.5 * (x / s) ** 2)
    return x, y / _trapz(y, x)


def ils_from_linefit_params(modulation, phase, mopd_cm, half_width=0.25, dnu=None,
                            n_opd=400):
    """ILS from LINEFIT's modulation efficiency and phase error.

    LINEFIT reports M(x) and phi(x) at equidistant OPD x_k = k*L/(n-1),
    k = 0..n-1.  The ILS is the cosine transform of the complex modulation
    over -L..L, with M even and phi odd:

        ILS(nu) = 2 * integral_0^L M(x) cos(2 pi nu x + phi(x)) dx
    """
    modulation = np.asarray(modulation, dtype=float)
    phase = np.asarray(phase, dtype=float)
    xk = np.linspace(0.0, mopd_cm, len(modulation))
    x = np.linspace(0.0, mopd_cm, n_opd)
    M = CubicSpline(xk, modulation)(x)
    P = CubicSpline(xk, phase)(x)
    dnu = dnu or 1.0 / (2 * mopd_cm) / 40.0
    nu = np.arange(-half_width, half_width + dnu / 2, dnu)
    integrand = M[None, :] * np.cos(2 * np.pi * nu[:, None] * x[None, :] + P[None, :])
    y = 2 * _trapz(integrand, x, axis=1)
    return nu, y / _trapz(y, nu)


def ils_stats(x, y):
    """FWHM, centroid, area, and the asymmetry (left/right half areas)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    area = _trapz(y, x)
    pk = _peak_position(x, y)
    i = int(np.argmax(y)); half = y[i] / 2
    left = np.where(y[:i] < half)[0]
    right = np.where(y[i:] < half)[0]
    fwhm = np.nan
    if len(left) and len(right):
        a, b = left[-1], i + right[0]
        xl = np.interp(half, [y[a], y[a + 1]], [x[a], x[a + 1]])
        xr = np.interp(half, [y[b], y[b - 1]], [x[b], x[b - 1]])
        fwhm = xr - xl
    cen = _trapz(x * y, x) / area if area else np.nan
    m = x <= pk
    al, ar = _trapz(y[m], x[m]), _trapz(y[~m], x[~m])
    return {"area": area, "peak": pk, "fwhm": fwhm, "centroid": cen,
            "asymmetry": (ar - al) / (ar + al) if (ar + al) else np.nan}
