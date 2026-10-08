# -*- coding: utf-8 -*-
"""Concentration by spectral fitting: HITRAN forward model, convolved with the ILS.

The integrated-area route (concentration.py) needs the ILS removed from the
spectrum first.  For lines that are black at their centre (optical depth >> 1)
the instrument hides the core, and no deconvolution recovers it: those lines
read low.  The usual FTIR retrieval avoids the problem by never integrating:

    tau(v)     = x * N_total * L * sum_j S_j(T) * V_j(v - shift)      (Voigt, HITRAN widths)
    T_model(v) = baseline(v) * [ (1 - z) * (exp(-tau) (*) ILS (*) G_w) + z ]

and x (the mixing ratio) is fitted to the measured transmittance, together with

    shift      wavenumber offset spectrum - HITRAN
    w          optional extra Gaussian broadening of the ILS (HWHM, cm-1), for
               an ILS slightly narrower than the one in force during the measurement
    z          optional zero-level offset of the transmittance (detector
               non-linearity, stray light) - it matters most for saturated lines
    g          optional scale factor on the HITRAN Lorentz widths (pressure
               broadening differing from HITRAN's, or a pressure reading that is off)
    baseline   polynomial of chosen order in the normalised wavenumber

Saturation is handled exactly because the non-linear Beer-Lambert law is in the
model.  fit_spectrum also refits each line (or unresolved line group) in its own
window, with shift, w and z fixed, as a consistency check: the per-window
concentrations should agree.

    fit_spectrum(x, T_meas, lines, T, P_pa, L_cm, ils, options) -> dict
    fit_multigas(x, T_meas, [(name, lines), ...], T, P_pa, L_cm, ils, options) -> dict
                         several gases at once, tau = sum_k x_k tau_k, one x per gas
    auto_windows(...)    windows around the strong line groups of a line list

fit_spectrum scales every line of its list by one x, so a line list holding
several molecules is treated as one gas at fixed ratios - use fit_multigas
when another absorber (H2O, CO, CO2 ...) overlaps the target's lines.
"""

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import wofz

import concentration as conc
import ils as ils_mod

SQRT_LN2 = np.sqrt(np.log(2.0))
SQRT_PI = np.sqrt(np.pi)
C_LIGHT, K_B, AMU = 2.99792458e8, 1.380649e-23, 1.66053906660e-27

DEFAULTS = {
    "vmr0": None,              # starting mixing ratio, ppm (None = coarse log scan)
    "shift0": None,            # starting shift, cm-1 (None = correlation scan)
    "fit_shift": True,
    "max_shift": 0.1,          # |shift| bound, cm-1
    "fit_broadening": True,
    "broadening0": 0.003,      # starting extra ILS HWHM, cm-1
    "max_broadening": 0.05,
    "fit_zero": True,          # True, False, or "auto": fixed at 0 unless a line is saturated
    "fit_lorentz": False,      # scale factor on the HITRAN Lorentz widths
    "baseline_order": 1,       # 0 = constant scale, 1 = linear, ...
    "oversample": 8,           # fine-grid points per data point
    "q_mode": "tips", "q_value": None,
    "iso": None,               # restrict to one isotopologue ("1"), None = all
    "s_min_rel": 1e-5,         # lines weaker than this fraction of the strongest are skipped
    "mass_amu": 44.0,          # Doppler mass when HAPI cannot tell
    "ils_centre": "peak",
    "windows": None,           # list of (lo, hi) for the per-window check, "auto", or None
}

# fit_zero "auto": the zero level is fitted only when the model's peak optical
# depth exceeds this.  With optically thin lines (1 - z) e^-tau + z ~ 1 - (1 - z) tau,
# so z and a common scale on the mixing ratios cannot be told apart.
AUTO_ZERO_TAU = 1.0


def _auto_zero(fit, args, o):
    """Run fit(*args, options) with the zero fixed, and again with it free when
    the result has a saturated line.  None when fit_zero is not "auto"."""
    if o.get("fit_zero") != "auto":
        return None
    r = fit(*args, dict(o, fit_zero=False, windows=None))
    tau = r["tau_max"]
    tau = max(tau.values()) if isinstance(tau, dict) else tau
    if tau > AUTO_ZERO_TAU:
        r = fit(*args, dict(o, fit_zero=True))
    elif o.get("windows"):                              # the per-window check was skipped above
        r = fit(*args, dict(o, fit_zero=False))
    r["zero_mode"] = "auto: %s (peak tau %.3g)" % ("fitted" if tau > AUTO_ZERO_TAU else "fixed at 0", tau)
    return r


def _masses(lines, idx, fallback):
    try:
        import hitran_fetch
        cache = {}
        out = np.empty(len(idx))
        for n, (m, i) in enumerate(zip(lines["mol"][idx], lines["iso"][idx])):
            key = (int(m), str(i))
            if key not in cache:
                cache[key] = hitran_fetch.iso_mass(*key)
            out[n] = cache[key]
        return out
    except Exception:
        return np.full(len(idx), float(fallback))


def _tau_per_ppm(xf, lines, idx, T, P_pa, S_T, masses, vmr_guess, g_scale=1.0):
    """Optical depth per ppm on the fine grid, divided by N_total*L (added later)."""
    P_atm = P_pa / 101325.0
    nu = lines["nu"][idx] + lines["delta_air"][idx] * P_atm
    p_self = vmr_guess * 1e-6 * P_atm
    gL = (296.0 / T) ** lines["n_air"][idx] * (lines["gamma_air"][idx] * (P_atm - p_self)
                                               + lines["gamma_self"][idx] * p_self) * g_scale
    gD = nu * np.sqrt(2 * np.log(2) * K_B * T / (masses * AMU * C_LIGHT ** 2))
    tau = np.zeros_like(xf)
    for a in range(0, len(idx), 40):                    # chunks keep the memory small
        b = slice(a, a + 40)
        z = (SQRT_LN2 * (xf[None, :] - nu[b, None]) + 1j * SQRT_LN2 * gL[b, None]) / gD[b, None]
        prof = SQRT_LN2 / (SQRT_PI * gD[b, None]) * wofz(z).real       # unit-area Voigt
        tau += (S_T[b, None] * prof).sum(axis=0)
    return tau


def _gauss_kernel(w, df):
    if w <= df * 0.05:
        return None
    half = max(1, int(np.ceil(4 * w / df)))
    k = np.exp(-np.log(2) * (df * np.arange(-half, half + 1) / w) ** 2)
    return k / k.sum()


class ForwardModel:
    """Everything that does not change during the fit, computed once."""

    def __init__(self, x, lines, T, P_pa, L_cm, ils, opts):
        o = self.o = opts
        self.x = np.asarray(x, float)
        dx = float(np.median(np.diff(self.x)))
        if dx <= 0:
            raise ValueError("the wavenumber axis must increase")
        self.df = df = dx / max(1, int(o["oversample"]))
        _, self.ker_ils = ils_mod.ils_kernel(ils[0], ils[1], df, centre=o["ils_centre"])
        margin = (len(self.ker_ils) // 2) * df + 4 * o["max_broadening"] + o["max_shift"] + 0.02
        self.xf = np.arange(self.x[0] - margin, self.x[-1] + margin + df / 2, df)
        # lines whose centre or wings reach the grid
        ok = np.ones(len(lines["nu"]), bool)
        if o["iso"]:
            ok &= lines["iso"].astype(str) == str(o["iso"])
        ok &= (lines["nu"] > self.xf[0] - 1.0) & (lines["nu"] < self.xf[-1] + 1.0)
        idx = np.flatnonzero(ok)
        if not len(idx):
            raise ValueError("no HITRAN lines within 1 cm-1 of %.4f-%.4f cm-1" % (self.x[0], self.x[-1]))
        S_T = conc.strengths_T(lines, idx, T, o["q_mode"], o["q_value"])
        keep = S_T >= o["s_min_rel"] * S_T.max()
        self.idx, self.S_T = idx[keep], S_T[keep]
        self.n_total = conc.number_density_total(P_pa, T)
        self.scale = self.n_total * L_cm * 1e-6            # per ppm
        self.masses = _masses(lines, self.idx, o["mass_amu"])
        self.lines, self.T, self.P_pa = lines, T, P_pa
        self._g = None
        self.tau1 = self.tau_unit(1.0)

    def tau_unit(self, g_scale):
        """Optical depth per ppm for one Lorentz scale (the last one is kept)."""
        if self._g != g_scale:
            self._tau = self.scale * _tau_per_ppm(self.xf, self.lines, self.idx, self.T, self.P_pa,
                                                  self.S_T, self.masses, self.o["vmr0"] or 0.0, g_scale)
            self._g = g_scale
        return self._tau

    def kernel(self, w):
        g = _gauss_kernel(w, self.df)
        return self.ker_ils if g is None else np.convolve(self.ker_ils, g)

    def transmittance(self, vmr, shift, w, z, xq=None, ker=None, g=1.0):
        """Instrument-convolved transmittance (no baseline) at xq (default: the data)."""
        tau = np.interp(self.xf - shift, self.xf, self.tau_unit(g), left=0.0, right=0.0)
        Tt = np.exp(-vmr * tau)
        Tc = fftconvolve(Tt, self.kernel(w) if ker is None else ker, mode="same")
        Tq = np.interp(self.x if xq is None else xq, self.xf, Tc)
        return Tq * (1 - z) + z

    def true_transmittance(self, vmr, shift, g=1.0):
        """Without the instrument, on the fine grid (for plotting)."""
        tau = np.interp(self.xf - shift, self.xf, self.tau_unit(g), left=0.0, right=0.0)
        return self.xf, np.exp(-vmr * tau)


def initial_guess(fm, x, Tm, o):
    """(ppm, shift) to start from.  The shift maximises the correlation of the
    measured absorption 1 - T with the instrument-convolved optical depth, which
    does not depend on the concentration; the ppm then comes from a log scan
    with the baseline scaled to the data."""
    shift = o["shift0"]
    if shift is None:
        if o["fit_shift"]:
            ker = fm.ker_ils
            tc = fftconvolve(fm.tau1, ker, mode="same")
            a = 1.0 - Tm / np.percentile(Tm, 95)
            a = a - a.mean()
            best, shift = -np.inf, 0.0
            for sh in np.arange(-o["max_shift"], o["max_shift"] + fm.df / 2, fm.df):
                b = np.interp(x, fm.xf + sh, tc)
                b = b - b.mean()
                c = float(a @ b) / (np.linalg.norm(b) + 1e-300)
                if c > best:
                    best, shift = c, float(sh)
        else:
            shift = 0.0
    vmr = o["vmr0"]
    if vmr is None:
        top = float(np.percentile(Tm, 95))
        best, vmr = np.inf, 1.0
        for v in np.logspace(-4, 6, 41):
            r = float(np.sum((top * fm.transmittance(v, shift, 0.0, 0.0) - Tm) ** 2))
            if r < best:
                best, vmr = r, float(v)
    return vmr, shift


def _xn(x, lo, hi):
    return 2 * (x - 0.5 * (lo + hi)) / (hi - lo) if hi > lo else x * 0


def _cov_err(r, n_par):
    res = r.fun
    dof = max(1, len(res) - n_par)
    s2 = float(res @ res) / dof
    try:
        cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
    except np.linalg.LinAlgError:
        cov = np.full((n_par, n_par), np.nan)
    return cov, np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_spectrum(x, T_meas, lines, T, P_pa, L_cm, ils, options=None):
    """Fit the mixing ratio to a measured transmittance spectrum.

    x, T_meas  the data inside the analysis region (transmittance)
    lines      HITRAN line list (concentration.load_hitran)
    T, P_pa, L_cm   cell temperature (K), total pressure (Pa), path length (cm)
    ils        (offset_cm1, values) of the instrument line shape
    Returns ppm, ppm_err, shift, broadening, zero, baseline, rms, r2, the model
    curves and, if options["windows"], one ppm per window."""
    t0 = time.perf_counter()
    o = dict(DEFAULTS); o.update(options or {})
    r = _auto_zero(fit_spectrum, (x, T_meas, lines, T, P_pa, L_cm, ils), o)
    if r is not None:
        r["elapsed_s"] = time.perf_counter() - t0
        return r
    x = np.asarray(x, float); Tm = np.asarray(T_meas, float)
    order = np.argsort(x); x, Tm = x[order], Tm[order]
    fm = ForwardModel(x, lines, T, P_pa, L_cm, ils, o)
    vmr0, shift0 = initial_guess(fm, x, Tm, o)
    lo, hi = x[0], x[-1]
    xn = _xn(x, lo, hi)
    nb = int(o["baseline_order"]) + 1

    NB0 = 5                                             # index of the first baseline coefficient
    names = ["ppm", "shift", "broadening", "zero", "lorentz_scale"] + ["b%d" % k for k in range(nb)]
    free = [True, bool(o["fit_shift"]), bool(o["fit_broadening"]), bool(o["fit_zero"]),
            bool(o["fit_lorentz"])] + [True] * nb
    p_all = np.array([vmr0, shift0, o["broadening0"] if o["fit_broadening"] else 0.0, 0.0, 1.0]
                     + [float(np.percentile(Tm, 95))] + [0.0] * (nb - 1))
    lb_all = np.array([0.0, -o["max_shift"], 0.0, -0.3, 0.2] + [0.0] + [-2.0] * (nb - 1))
    ub_all = np.array([1e7, o["max_shift"], o["max_broadening"], 0.3, 5.0] + [2.0] + [2.0] * (nb - 1))
    p_all = np.clip(p_all, lb_all, ub_all)
    fi = np.flatnonzero(free)

    def unpack(v):
        p = p_all.copy(); p[fi] = v
        return p

    def model(p):
        base = np.polyval(p[NB0:][::-1], xn)
        return base * fm.transmittance(p[0], p[1], p[2], p[3], g=p[4])

    r = least_squares(lambda v: model(unpack(v)) - Tm, p_all[fi], bounds=(lb_all[fi], ub_all[fi]),
                      x_scale="jac", method="trf", max_nfev=200 * len(fi),
                      diff_step=1e-4 if o["fit_lorentz"] else None)
    p = unpack(r.x)
    cov, err_f = _cov_err(r, len(fi))
    err = np.zeros(len(p)); err[fi] = err_f
    fit = model(p)
    res = Tm - fit
    ss_tot = float(np.sum((Tm - Tm.mean()) ** 2))
    out = {
        "ppm": float(p[0]), "ppm_err": float(err[0]),
        "shift": float(p[1]), "shift_err": float(err[1]),
        "broadening": float(p[2]), "broadening_err": float(err[2]),
        "zero": float(p[3]), "zero_err": float(err[3]),
        "lorentz_scale": float(p[4]), "lorentz_scale_err": float(err[4]),
        "baseline": p[NB0:].tolist(), "params": dict(zip(names, p.tolist())),
        "free": [n for n, f in zip(names, free) if f],
        "rms": float(np.sqrt(np.mean(res ** 2))),
        "r2": 1.0 - float(res @ res) / ss_tot if ss_tot > 0 else float("nan"),
        "x": x, "data": Tm, "fit": fit, "residual": res,
        "baseline_curve": np.polyval(p[NB0:][::-1], xn),
        "n_lines": int(len(fm.idx)), "n_total": fm.n_total,
        "status": int(r.status), "message": r.message, "nfev": int(r.nfev),
        "options": {k: v for k, v in o.items() if k != "windows"},
    }
    # parameters that ended on a bound: the fit traded them against each other
    out["at_bound"] = [n for n, v, a, b, f in zip(names, p, lb_all, ub_all, free)
                       if f and n not in ("ppm",) and not n.startswith("b")
                       and (abs(v - a) < 1e-6 * max(1, abs(a)) or abs(v - b) < 1e-6 * max(1, abs(b)))
                       and not (n == "broadening" and v == 0.0)]
    # peak optical depth in the region, at the fitted ppm - how saturated the lines are
    xf, Tt = fm.true_transmittance(p[0], p[1], p[4])
    m = (xf >= lo) & (xf <= hi)
    out["tau_max"] = float(-np.log(max(Tt[m].min(), 1e-300))) if m.any() else float("nan")

    if isinstance(o["windows"], str) and o["windows"] == "auto":
        o["windows"] = auto_windows(lines, lo, hi, T, shift=p[1], q_mode=o["q_mode"], iso=o["iso"])
    if o["windows"]:
        ker = fm.kernel(p[2])
        win = []
        for wlo, whi in o["windows"]:
            mw = (x >= wlo) & (x <= whi)
            if mw.sum() < 6:
                continue
            xw, Tw = x[mw], Tm[mw]
            xnw = _xn(xw, wlo, whi)

            def mod_w(v):
                return (v[1] + v[2] * xnw) * fm.transmittance(v[0], p[1], p[2], p[3], xq=xw, ker=ker, g=p[4])

            rw = least_squares(lambda v: mod_w(v) - Tw, [p[0], float(np.percentile(Tw, 95)), 0.0],
                               bounds=([0, 0, -2], [1e7, 2, 2]), x_scale="jac")
            _c, ew = _cov_err(rw, 3)
            _xf, Tt_w = fm.true_transmittance(rw.x[0], p[1], p[4])
            mm = (_xf >= wlo) & (_xf <= whi)
            win.append({"lo": float(wlo), "hi": float(whi), "ppm": float(rw.x[0]), "ppm_err": float(ew[0]),
                        "rms": float(np.sqrt(np.mean(rw.fun ** 2))),
                        "tau_max": float(-np.log(max(Tt_w[mm].min(), 1e-300))) if mm.any() else float("nan")})
        out["windows"] = win
    out["elapsed_s"] = time.perf_counter() - t0
    return out


# =============================================================================
# Several gases at once
# =============================================================================
class MultiForwardModel:
    """One ForwardModel per gas on a common fine grid; the optical depths add:

        tau(v) = sum_k x_k * tau_k(v - shift)

    vmr_self: {name: ppm} assumed for each gas's self-broadening (it matters
    for H2O at the per-cent level, whose self-broadening is ~5x its air one)."""

    def __init__(self, x, gases, T, P_pa, L_cm, ils, opts, vmr_self):
        self.names = [n for n, _ in gases]
        self.fms = [ForwardModel(x, lines, T, P_pa, L_cm, ils, dict(opts, vmr0=vmr_self.get(n) or 0.0))
                    for n, lines in gases]
        f0 = self.fms[0]
        self.x, self.xf, self.df, self.ker_ils, self.n_total = f0.x, f0.xf, f0.df, f0.ker_ils, f0.n_total
        self.kernel = f0.kernel

    def tau(self, vmrs, shift, g=1.0, only=None):
        t = np.zeros_like(self.xf)
        for k, (fm, v) in enumerate(zip(self.fms, vmrs)):
            if only is None or k == only:
                t += v * fm.tau_unit(g)
        return np.interp(self.xf - shift, self.xf, t, left=0.0, right=0.0)

    def transmittance(self, vmrs, shift, w, z, xq=None, ker=None, g=1.0, only=None):
        Tc = fftconvolve(np.exp(-self.tau(vmrs, shift, g, only)), self.kernel(w) if ker is None else ker,
                         mode="same")
        return np.interp(self.x if xq is None else xq, self.xf, Tc) * (1 - z) + z


def _multi_initial_guess(mf, x, Tm, o, vmr0):
    """Shift and mixing ratios to start from: for each trial shift, the
    optically thin absorbance -ln(T/top) is fitted as a non-negative sum of the
    ILS-convolved optical depths per ppm (NNLS); the shift with the smallest
    residual wins.  Saturated lines read low here, the full fit corrects them."""
    from scipy.optimize import nnls
    top = float(np.percentile(Tm, 95))
    a = -np.log(np.clip(Tm / top, 1e-4, None))
    taus = [fftconvolve(fm.tau_unit(1.0), mf.ker_ils, mode="same") for fm in mf.fms]
    shifts = [o["shift0"]] if o["shift0"] is not None else (
        np.arange(-o["max_shift"], o["max_shift"] + mf.df / 2, mf.df) if o["fit_shift"] else [0.0])
    best = (np.inf, 0.0, None)
    for sh in shifts:
        A = np.column_stack([np.interp(x, mf.xf + sh, t) for t in taus])
        v, rn = nnls(A, a)
        if rn < best[0]:
            best = (rn, float(sh), v)
    _, shift, v = best
    v = [float(vmr0[n]) if vmr0.get(n) is not None else max(float(vk), 1e-3)
         for n, vk in zip(mf.names, v)]
    return v, shift


def fit_multigas(x, T_meas, gases, T, P_pa, L_cm, ils, options=None):
    """Fit the mixing ratios of several gases to one transmittance spectrum.

    gases      list of (name, lines) - one HITRAN line list per gas
    options    as fit_spectrum, except
                 "vmr0"       {name: starting ppm}, missing names from an NNLS guess
                 "self_iter"  refits after updating each gas's self-broadening to its
                              fitted mixing ratio (default 1; 0 = keep the start values)
    Shift, broadening, zero, Lorentz scale and baseline are common to all gases.

    Returns ppm / ppm_err ({name: value}), the correlation matrix of the
    mixing ratios ("corr", names in "gases"), each gas's own convolved
    transmittance times the baseline ("components"), and the same fit
    diagnostics as fit_spectrum."""
    t0 = time.perf_counter()
    o = dict(DEFAULTS); o.update(options or {})
    r = _auto_zero(fit_multigas, (x, T_meas, gases, T, P_pa, L_cm, ils), o)
    if r is not None:
        r["elapsed_s"] = time.perf_counter() - t0
        return r
    vmr0 = dict(o["vmr0"] or {}) if isinstance(o["vmr0"], dict) else {}
    x = np.asarray(x, float); Tm = np.asarray(T_meas, float)
    order = np.argsort(x); x, Tm = x[order], Tm[order]
    names = [n for n, _ in gases]
    if len(set(names)) != len(names):
        raise ValueError("gas names must be unique")
    K = len(gases)
    lo, hi = x[0], x[-1]
    xn = _xn(x, lo, hi)
    nb = int(o["baseline_order"]) + 1

    mf = MultiForwardModel(x, gases, T, P_pa, L_cm, ils, o, vmr0)
    v0, shift0 = _multi_initial_guess(mf, x, Tm, o, vmr0)

    NB0 = K + 4
    pnames = list(names) + ["shift", "broadening", "zero", "lorentz_scale"] + ["b%d" % k for k in range(nb)]
    free = [True] * K + [bool(o["fit_shift"]), bool(o["fit_broadening"]), bool(o["fit_zero"]),
                         bool(o["fit_lorentz"])] + [True] * nb
    lb_all = np.array([0.0] * K + [-o["max_shift"], 0.0, -0.3, 0.2] + [0.0] + [-2.0] * (nb - 1))
    ub_all = np.array([1e7] * K + [o["max_shift"], o["max_broadening"], 0.3, 5.0] + [2.0] + [2.0] * (nb - 1))
    fi = np.flatnonzero(free)

    def model(p, only=None):
        base = np.polyval(p[NB0:][::-1], xn)
        return base * mf.transmittance(p[:K], p[K], p[K + 1], p[K + 2], g=p[K + 3], only=only)

    p_all = np.clip(np.array(list(v0) + [shift0, o["broadening0"] if o["fit_broadening"] else 0.0, 0.0, 1.0]
                             + [float(np.percentile(Tm, 95))] + [0.0] * (nb - 1)), lb_all, ub_all)
    n_iter = 1 + max(0, int(o.get("self_iter", 1)))
    nfev = 0
    for it in range(n_iter):
        if it:                                          # self-broadening at the fitted mixing ratios
            mf = MultiForwardModel(x, gases, T, P_pa, L_cm, ils, o, dict(zip(names, p_all[:K])))

        def unpack(v, base=p_all):
            p = base.copy(); p[fi] = v
            return p

        r = least_squares(lambda v: model(unpack(v)) - Tm, p_all[fi], bounds=(lb_all[fi], ub_all[fi]),
                          x_scale="jac", method="trf", max_nfev=200 * len(fi),
                          diff_step=1e-4 if o["fit_lorentz"] else None)
        p_all = unpack(r.x)
        nfev += int(r.nfev)
    p = p_all
    cov_f, err_f = _cov_err(r, len(fi))
    err = np.zeros(len(p)); err[fi] = err_f
    cov = np.zeros((len(p), len(p))); cov[np.ix_(fi, fi)] = cov_f
    sd = np.sqrt(np.clip(np.diag(cov)[:K], 1e-300, None))
    corr = cov[:K, :K] / np.outer(sd, sd)

    fit = model(p)
    res = Tm - fit
    ss_tot = float(np.sum((Tm - Tm.mean()) ** 2))
    base = np.polyval(p[NB0:][::-1], xn)
    out = {
        "gases": names,
        "ppm": dict(zip(names, p[:K].tolist())), "ppm_err": dict(zip(names, err[:K].tolist())),
        "corr": corr,
        "shift": float(p[K]), "shift_err": float(err[K]),
        "broadening": float(p[K + 1]), "broadening_err": float(err[K + 1]),
        "zero": float(p[K + 2]), "zero_err": float(err[K + 2]),
        "lorentz_scale": float(p[K + 3]), "lorentz_scale_err": float(err[K + 3]),
        "baseline": p[NB0:].tolist(), "params": dict(zip(pnames, p.tolist())),
        "free": [n for n, f in zip(pnames, free) if f],
        "rms": float(np.sqrt(np.mean(res ** 2))),
        "r2": 1.0 - float(res @ res) / ss_tot if ss_tot > 0 else float("nan"),
        "x": x, "data": Tm, "fit": fit, "residual": res, "baseline_curve": base,
        "components": {n: model(p, only=k) for k, n in enumerate(names)},
        "n_lines": {n: int(len(fm.idx)) for n, fm in zip(names, mf.fms)}, "n_total": mf.n_total,
        "status": int(r.status), "message": r.message, "nfev": nfev,
    }
    out["at_bound"] = [n for n, v, a, b, f in zip(pnames, p, lb_all, ub_all, free)
                       if f and n not in names and not n.startswith("b")
                       and (abs(v - a) < 1e-6 * max(1, abs(a)) or abs(v - b) < 1e-6 * max(1, abs(b)))
                       and not (n == "broadening" and v == 0.0)]
    m = (mf.xf >= lo) & (mf.xf <= hi)
    out["tau_max"] = {n: float(mf.tau(p[:K], p[K], p[K + 3], only=k)[m].max()) for k, n in enumerate(names)}
    if o["windows"]:
        out["windows"] = _multi_windows(mf, gases, x, Tm, p, o, T, lo, hi)
    out["elapsed_s"] = time.perf_counter() - t0
    return out


# a gas is refitted in a window when its peak optical depth there is at least
# this fraction of the window's own gas; weaker ones stay at the global fit
COFIT_REL = 0.2
# a window is kept for a gas only when the gas's peak optical depth in it is at
# least this fraction of its peak in the region (its weak lines say little)
WINDOW_MIN_REL = 0.1


def _multi_windows(mf, gases, x, Tm, p, o, T, lo, hi):
    """The per-window check of a multi-gas fit.  Windows are put around the
    strong line groups of every gas ("auto", from its own line list), and each
    window belongs to that gas.  In a window the gas's concentration and a
    linear baseline are refitted, with shift, broadening, zero and Lorentz
    scale fixed at the global fit; other gases that absorb there too (peak tau
    >= COFIT_REL of the window's gas) are refitted with it, the rest are fixed.
    A window is dropped when its gas's lines there are weak (peak tau below
    WINDOW_MIN_REL of the gas's peak in the region) or when another gas absorbs
    more than 1 / COFIT_REL times as much - it then says little about the gas.
    An explicit list of (lo, hi) windows is given to the gas absorbing most."""
    names = mf.names
    K = len(names)
    shift, w, z, g = p[K], p[K + 1], p[K + 2], p[K + 3]
    ker = mf.kernel(w)
    if isinstance(o["windows"], str) and o["windows"] == "auto":
        wins = [(n, a, b) for n, lines in gases
                for a, b in auto_windows(lines, lo, hi, T, shift=shift, q_mode=o["q_mode"], iso=o["iso"])]
    else:
        wins = [(None, a, b) for a, b in o["windows"]]
    taus = [mf.tau(p[:K], shift, g, only=k) for k in range(K)]
    mr = (mf.xf >= lo) & (mf.xf <= hi)
    gas_peak = np.array([t[mr].max() for t in taus])
    out = []
    for owner, wlo, whi in sorted(wins, key=lambda t: t[1]):
        mw = (x >= wlo) & (x <= whi)
        mm = (mf.xf >= wlo) & (mf.xf <= whi)
        if mw.sum() < 6 or not mm.any():
            continue
        peak = np.array([t[mm].max() for t in taus])
        ko = int(np.argmax(peak)) if owner is None else names.index(owner)
        if peak[ko] <= 0 or peak[ko] < WINDOW_MIN_REL * gas_peak[ko] or peak[ko] < COFIT_REL * peak.max():
            continue
        free = [k for k in range(K) if k == ko or peak[k] >= COFIT_REL * peak[ko]]
        nf = len(free)
        xw, Tw = x[mw], Tm[mw]
        xnw = _xn(xw, wlo, whi)

        def vm_of(v):
            vm = p[:K].copy(); vm[free] = v[:nf]
            return vm

        def mod_w(v):
            return (v[nf] + v[nf + 1] * xnw) * mf.transmittance(vm_of(v), shift, w, z, xq=xw, ker=ker, g=g)

        rw = least_squares(lambda v: mod_w(v) - Tw, list(p[free]) + [float(np.percentile(Tw, 95)), 0.0],
                           bounds=([0.0] * nf + [0, -2], [1e7] * nf + [2, 2]), x_scale="jac")
        _c, ew = _cov_err(rw, nf + 2)
        i = free.index(ko)
        out.append({"gas": names[ko], "lo": float(wlo), "hi": float(whi),
                    "ppm": float(rw.x[i]), "ppm_err": float(ew[i]),
                    "rms": float(np.sqrt(np.mean(rw.fun ** 2))),
                    "tau_max": float(mf.tau(vm_of(rw.x), shift, g)[mm].max()),
                    "cofit": [names[k] for k in free if k != ko]})
    return out


def auto_windows(lines, lo, hi, T, shift=0.0, rel=0.02, group=0.03, half=0.08, q_mode="tips", iso=None):
    """Windows around the line groups that matter: lines with S(T) above rel
    times the strongest in [lo, hi], lines closer than `group` cm-1 merged
    (unresolved doublets), each window centred on the group and at most
    `half` wide on either side, split halfway to its neighbours."""
    nu = lines["nu"] + shift
    ok = (nu >= lo) & (nu <= hi)
    if iso:
        ok &= lines["iso"].astype(str) == str(iso)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    S = conc.strengths_T(lines, idx, T, q_mode)
    k = S >= rel * S.max()
    idx, S = idx[k], S[k]
    o = np.argsort(nu[idx]); idx, S = idx[o], S[o]
    groups, cur = [], [0]
    for j in range(1, len(idx)):
        if nu[idx[j]] - nu[idx[j - 1]] <= group:
            cur.append(j)
        else:
            groups.append(cur); cur = [j]
    groups.append(cur)
    centres = [float(np.sum(nu[idx[g]] * S[g]) / np.sum(S[g])) for g in groups]
    spans = [(float(nu[idx[g[0]]]), float(nu[idx[g[-1]]])) for g in groups]
    out = []
    for i, (c, (a, b)) in enumerate(zip(centres, spans)):
        left = a - half if i == 0 else max(a - half, 0.5 * (spans[i - 1][1] + a))
        right = b + half if i == len(groups) - 1 else min(b + half, 0.5 * (b + spans[i + 1][0]))
        out.append((max(lo, left), min(hi, right)))
    return out
