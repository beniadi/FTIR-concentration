# -*- coding: utf-8 -*-
"""Gas concentration from an integrated absorbance and HITRAN line intensities.

The integrated Beer-Lambert law, for absorbance in base e (A = ln(I0/I)):

    integral A dv = S(T) * N * L

    S(T)   line intensity at the gas temperature  (cm-1 / (molecule cm-2) = cm / molecule),
           summed over the lines inside the integrated area
    N      number density of the absorbing gas   (molecule / cm3)
    L      optical path length                   (cm)

and the ideal gas gives the number density of all molecules in the cell,

    N_total = P / (k_B T)                       (molecule / cm3 after the m3 -> cm3 factor)

so the volume mixing ratio is  x = N / N_total,  ppm = 1e6 x.

    to_base_e(area, base)            areas in log10 are ln(10) = 2.3026 x smaller
    line_strength_T(S, E", v0, T)    HITRAN temperature conversion of S(296 K)
    q_ratio(T, mode)                 Q(296)/Q(T) - power-law approximation
    load_hitran(path)                .par (160-character) or a CSV with a header
    estimate_shift(lines, positions) wavenumber offset spectrum - HITRAN
    match_lines(lines, positions)    the HITRAN lines under each fitted line
    select_lines(lines, lo, hi, ...) the lines that belong to an area
    strength_sum(lines, idx, T, ...) sum of S(T) over those lines (TIPS or power law)
    concentration(...)               N, N_total, ppm and the propagated uncertainty

HITRAN intensities already contain the natural isotopic abundance, so the
result is the mixing ratio of the molecule (all isotopologues at natural
abundance) when the lines used are those of the main isotopologue.
"""

import csv
import math

import numpy as np

K_B = 1.380649e-23          # J / K
C2 = 1.4387769              # second radiation constant hc/k, cm K
T_REF = 296.0               # K, HITRAN reference temperature
LN10 = math.log(10.0)

# Pressure units -> Pa
PRESSURE_UNITS = {"atm": 101325.0, "hPa": 100.0, "mbar": 100.0, "Torr": 101325.0 / 760.0,
                  "kPa": 1000.0, "Pa": 1.0}

# Exponent n of Q(T) ~ T^n for the rotational partition function:
# 1 for linear molecules (N2O, CO2, CO, HCN ...), 1.5 for non-linear ones (H2O, CH4, O3 ...).
Q_EXPONENT = {"linear": 1.0, "nonlinear": 1.5}


def to_base_e(area, base):
    """Integrated absorbance in base e.  The radiative-transfer law uses ln(I0/I)."""
    return area * LN10 if str(base) == "10" else area


def q_ratio(T, mode="linear", value=None):
    """Q(T_ref) / Q(T).  mode "manual" returns value (e.g. from the HITRAN
    partition-function tables, q-files); otherwise the rotational power law
    (T_ref / T)^n, which ignores the vibrational partition function - good to
    a few tenths of a per cent within ~±30 K of 296 K for small molecules."""
    if mode == "manual":
        return float(value) if value else 1.0
    return (T_REF / float(T)) ** Q_EXPONENT.get(mode, 1.0)


def line_strength_T(S_ref, E_lower, v0, T, qr=None):
    """HITRAN line intensity at temperature T from its value at 296 K:

    S(T) = S(Tref) * Q(Tref)/Q(T) * exp(-c2 E"/T) / exp(-c2 E"/Tref)
                   * (1 - exp(-c2 v0/T)) / (1 - exp(-c2 v0/Tref))

    S_ref, E_lower (cm-1) and v0 (cm-1) may be arrays.  qr = Q(Tref)/Q(T),
    default the linear-molecule power law."""
    T = float(T)
    S_ref = np.asarray(S_ref, float); E_lower = np.asarray(E_lower, float); v0 = np.asarray(v0, float)
    if qr is None:
        qr = q_ratio(T)
    boltz = np.exp(-C2 * E_lower / T) / np.exp(-C2 * E_lower / T_REF)
    stim = (1.0 - np.exp(-C2 * v0 / T)) / (1.0 - np.exp(-C2 * v0 / T_REF))
    return S_ref * qr * boltz * stim


# =============================================================================
# HITRAN line lists
# =============================================================================
# Line-shape parameters a file may not carry, and what is assumed then:
# air- and self-broadening HWHM at 1 atm, 296 K (cm-1/atm), the temperature
# exponent, and the air pressure shift (cm-1/atm).
SHAPE_DEFAULTS = {"gamma_air": 0.07, "gamma_self": 0.07, "n_air": 0.75, "delta_air": 0.0}
FIELDS = ("nu", "sw", "elower", "mol", "iso", "gamma_air", "gamma_self", "n_air", "delta_air")


def _f(text, default):
    try:
        return float(text)
    except ValueError:
        return default


def _parse_par_line(ln):
    """One 160-character HITRAN2004+ record: molecule, isotopologue, v0, S, E",
    and the Voigt line-shape parameters."""
    d = SHAPE_DEFAULTS
    return {"mol": int(ln[0:2]), "iso": ln[2:3].strip(), "nu": float(ln[3:15]),
            "sw": float(ln[15:25]), "elower": float(ln[45:55]),
            "gamma_air": _f(ln[35:40], d["gamma_air"]), "gamma_self": _f(ln[40:45], d["gamma_self"]),
            "n_air": _f(ln[55:59], d["n_air"]), "delta_air": _f(ln[59:67], d["delta_air"])}


_CSV_NAMES = {"nu": ("nu", "wavenumber", "v0", "nu0", "position"),
              "sw": ("sw", "s", "intensity", "line_intensity", "s296"),
              "elower": ("elower", "e\"", "e''", "e_lower", "lower_state_energy", "epp"),
              "mol": ("molec_id", "mol", "molecule_id", "m"),
              "iso": ("local_iso_id", "iso", "i"),
              "gamma_air": ("gamma_air", "g_air"), "gamma_self": ("gamma_self", "g_self"),
              "n_air": ("n_air",), "delta_air": ("delta_air", "d_air")}


def load_hitran(path):
    """Lines from a HITRAN file, as a dict of arrays {nu, sw, elower, mol, iso,
    gamma_air, gamma_self, n_air, delta_air} (SHAPE_DEFAULTS where a CSV lacks them).

    Reads the native 160-character .par format (HITRANonline's default output,
    also what HAPI stores as .data) and comma/tab separated tables with a
    header row naming the columns (nu, sw, elower - the HITRANonline
    custom-output names - or similar)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = [ln.rstrip("\r\n") for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not text:
        raise ValueError("%s: no lines" % path)
    rows = []
    first = text[0]
    if len(first) >= 100 and first[3:15].strip().replace(".", "", 1).isdigit():
        for ln in text:
            if len(ln) >= 55:
                rows.append(_parse_par_line(ln))
    else:
        delim = "," if "," in first else ("\t" if "\t" in first else None)
        if delim is None:
            reader = [ln.split() for ln in text]
        else:
            reader = list(csv.reader(text, delimiter=delim))
        head = [h.strip().lower() for h in reader[0]]
        col = {}
        for key, names in _CSV_NAMES.items():
            for i, h in enumerate(head):
                if h in names:
                    col[key] = i; break
        missing = [k for k in ("nu", "sw", "elower") if k not in col]
        if missing:
            raise ValueError("%s: no column for %s in the header %s" % (path, ", ".join(missing), reader[0]))
        for r in reader[1:]:
            try:
                rows.append({"nu": float(r[col["nu"]]), "sw": float(r[col["sw"]]),
                             "elower": float(r[col["elower"]]),
                             "mol": int(r[col["mol"]]) if "mol" in col else 0,
                             "iso": r[col["iso"]].strip() if "iso" in col else "",
                             **{k: _f(r[col[k]], v) if k in col else v for k, v in SHAPE_DEFAULTS.items()}})
            except (ValueError, IndexError):
                continue
    if not rows:
        raise ValueError("%s: no readable HITRAN lines" % path)
    return {k: np.array([r[k] for r in rows]) for k in FIELDS}


def _iso_mask(lines, iso):
    ok = np.ones(len(lines["nu"]), bool)
    if iso:
        ok &= lines["iso"].astype(str) == str(iso)
    return ok


def estimate_shift(lines, positions, max_shift=0.1, iso=None, width=0.003):
    """Wavenumber offset (spectrum - HITRAN, cm-1) that lines the HITRAN lines up
    with the fitted positions: a grid search maximising the intensity-weighted
    overlap, refined by the intensity-weighted mean offset of the lines that
    match.  An uncalibrated FTIR axis is typically off by a few 0.01 cm-1,
    enough to miss every line with a ±0.01 cm-1 tolerance."""
    ok = _iso_mask(lines, iso)
    nu, sw = lines["nu"][ok], lines["sw"][ok]
    pos = np.atleast_1d(positions).astype(float)
    if not len(nu) or not len(pos):
        return 0.0
    grid = np.arange(-max_shift, max_shift + 1e-12, width / 10)
    d = pos[None, :, None] - nu[None, None, :] - grid[:, None, None]
    score = np.sum(sw[None, None, :] * np.exp(-(d / width) ** 2), axis=(1, 2))
    sh = float(grid[np.argmax(score)])
    off = pos[:, None] - nu[None, :] - sh
    m = np.abs(off) <= 2 * width
    if m.any():
        w = np.broadcast_to(sw[None, :], m.shape)[m]
        sh += float(np.sum(w * off[m]) / np.sum(w))
    return sh


def match_lines(lines, positions, tol=0.015, shift=0.0, iso=None):
    """For each fitted line, the indices of every HITRAN line within ±tol cm-1
    of it (after the shift), each HITRAN line given to its nearest fitted line.
    Lines closer together than the resolution (e.g. hot-band doublets) are
    fitted as one Voigt, so all of them count towards that line's area."""
    ok = _iso_mask(lines, iso)
    pos = np.atleast_1d(positions).astype(float)
    groups = [[] for _ in pos]
    if not len(pos):
        return []
    for j in np.flatnonzero(ok):
        d = np.abs(pos - (lines["nu"][j] + shift))
        k = int(np.argmin(d))
        if d[k] <= tol:
            groups[k].append(j)
    return [np.array(g, int) for g in groups]


def select_lines(lines, lo=None, hi=None, positions=None, tol=0.015, iso=None, shift=0.0):
    """Indices of the lines that belong to an area.

    positions=None: every line with lo <= v0 + shift <= hi (for an area
    integrated over a region).  positions given: the lines under the fitted
    lines (match_lines), for the analytic area of the fitted lines, so that
    lines nobody fitted are not counted.  iso restricts to one isotopologue
    (HITRAN's local id, "1" = most abundant)."""
    if positions is not None:
        g = match_lines(lines, positions, tol, shift, iso)
        return np.concatenate(g) if g else np.array([], int)
    nu = lines["nu"] + shift
    ok = _iso_mask(lines, iso)
    if lo is not None: ok &= nu >= lo
    if hi is not None: ok &= nu <= hi
    return np.flatnonzero(ok)


def strength_sum(lines, idx, T, q_mode="tips", q_value=None):
    """Sum of S(T) over lines[idx].  q_mode "tips" takes Q(296)/Q(T) per
    isotopologue from HAPI's TIPS tables (falls back to the linear power law
    when HAPI is missing or the line list has no molecule ids)."""
    return float(np.sum(strengths_T(lines, idx, T, q_mode, q_value)))


def strengths_T(lines, idx, T, q_mode="tips", q_value=None):
    """S(T) of each of lines[idx] (array); see strength_sum."""
    idx = np.asarray(idx, int)
    if not len(idx):
        return np.zeros(0)
    nu, sw, el = lines["nu"][idx], lines["sw"][idx], lines["elower"][idx]
    if q_mode == "tips":
        qr = np.empty(len(idx))
        try:
            import hitran_fetch
            cache = {}
            for n, (m, i) in enumerate(zip(lines["mol"][idx], lines["iso"][idx])):
                key = (int(m), str(i))
                if key not in cache:
                    cache[key] = hitran_fetch.partition_ratio(key[0], key[1], T, T_REF)
                qr[n] = cache[key]
        except Exception:
            qr[:] = q_ratio(T, "linear")
    else:
        qr = q_ratio(T, q_mode, q_value)
    return line_strength_T(sw, el, nu, T, qr)


# =============================================================================
# The retrieval
# =============================================================================
def number_density_total(P_pa, T):
    """Ideal gas, molecule / cm3."""
    return P_pa / (K_B * float(T)) * 1e-6


def concentration(area_e, S_T, L_cm, T, P_pa, area_err=None, rel_u=None, S_of_T=None):
    """Mixing ratio from an integrated absorbance in base e.

    area_e    integral A dv, base e (cm-1)
    S_T       summed line intensity at T (cm / molecule)
    rel_u     optional relative standard uncertainties {"S", "L", "P", "T_K"}:
              S, L, P as fractions; T_K as an absolute uncertainty in K
    S_of_T    optional callable T -> S(T), so the temperature uncertainty
              also acts through the line intensity, not only through N_total

    Returns {N, N_total, column, vmr, ppm, ppm_err, budget}.  The uncertainty
    is first-order and treats the inputs as uncorrelated; budget lists each
    input's contribution to ppm (same units)."""
    N = area_e / (S_T * L_cm)
    N_tot = number_density_total(P_pa, T)
    vmr = N / N_tot
    ppm = vmr * 1e6
    rel_u = rel_u or {}
    budget = {}
    if area_err:
        budget["area"] = abs(ppm) * area_err / abs(area_e) if area_e else float("nan")
    for k in ("S", "L", "P"):
        if rel_u.get(k):
            budget[k] = abs(ppm) * float(rel_u[k])
    uT = rel_u.get("T_K")
    if uT:
        def ppm_at(t):
            s = S_of_T(t) if S_of_T else S_T
            return area_e / (s * L_cm) / number_density_total(P_pa, t) * 1e6
        budget["T"] = abs(ppm_at(T + uT) - ppm_at(T - uT)) / 2.0
    ppm_err = math.sqrt(sum(v ** 2 for v in budget.values())) if budget else None
    return {"N": N, "N_total": N_tot, "column": N * L_cm, "vmr": vmr, "ppm": ppm,
            "ppm_err": ppm_err, "budget": budget}
