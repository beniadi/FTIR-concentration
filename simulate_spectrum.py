# -*- coding: utf-8 -*-
"""Simulated FTIR transmittance spectrum from a HITRAN line list.

The spectrum is computed with HAPI's own line-by-line code (absorption cross
sections with Voigt profiles, HITRAN broadening and shifts, TIPS partition
sums) - independently of retrieval.py - so it is a fair test of the
concentration retrieval:

    sigma(v)   = HAPI absorptionCoefficient_Voigt (cm2/molecule) at T, P
    tau(v)     = sigma(v) * x * N_total * L,        N_total = P / (k_B T)
    T_meas(v)  = baseline(v) * (exp(-tau) (*) ILS)  + noise, sampled on the output grid

    python simulate_spectrum.py                       # the N2O example in Input/
    python simulate_spectrum.py --ppm 2 --lines Input/HITRAN/CH4_....data --lo 3000 --hi 3010 ...
    python simulate_spectrum.py --gas Input/HITRAN/N2O_....data 500 --gas Input/HITRAN/H2O_....data 20000 ...

Writes two columns (wavenumber cm-1, transmittance) with a commented header
that records every input, so the true concentration is known.
"""

import argparse
import contextlib
import io
import os

import numpy as np
from scipy.signal import fftconvolve

import concentration as conc
import ils as ils_mod
import spectrum_io as sio

HERE = os.path.dirname(os.path.abspath(__file__))


def simulate(lines_path, ppm, T, P_pa, L_cm, ils, lo, hi, step=0.00188, fine=0.0002,
             noise=1e-3, baseline=(1.0, 0.0), seed=1):
    """(x, T_meas, T_true_on_x) for a gas of `ppm` in the cell.

    lines_path   HITRAN .data/.par file (HAPI reads the table next to its .header)
    ils          (offset_cm1, values)
    baseline     (level, slope per cm-1 about the centre)"""
    return simulate_mix([(lines_path, ppm)], T, P_pa, L_cm, ils, lo, hi, step, fine, noise, baseline, seed)


def simulate_mix(components, T, P_pa, L_cm, ils, lo, hi, step=0.00188, fine=0.0002,
                 noise=1e-3, baseline=(1.0, 0.0), seed=1):
    """As simulate, for a mixture: components = [(lines_path, ppm), ...].  The
    optical depths add; each gas is self-broadened by its own mixing ratio
    and air-broadened by the rest."""
    import hitran_fetch
    h = hitran_fetch.hapi()
    margin = 1.0
    nu_f = np.arange(lo - margin, hi + margin, fine)
    tau = np.zeros_like(nu_f)
    for lines_path, ppm in components:
        folder, name = os.path.split(os.path.abspath(lines_path))
        table = os.path.splitext(name)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            if table not in h.LOCAL_TABLE_CACHE:
                h.db_begin(folder)
            x_vmr = ppm * 1e-6
            nu_k, sigma = h.absorptionCoefficient_Voigt(
                SourceTables=table, Environment={"p": P_pa / 101325.0, "T": T},
                Diluent={"air": 1.0 - x_vmr, "self": x_vmr},
                WavenumberGrid=nu_f, HITRAN_units=True,
                WavenumberWing=25.0)          # full line wings (HAPI's default cuts at 50 half-widths)
        tau += sigma * x_vmr * conc.number_density_total(P_pa, T) * L_cm
    T_true = np.exp(-tau)
    _, ker = ils_mod.ils_kernel(ils[0], ils[1], fine)
    T_conv = fftconvolve(T_true, ker, mode="same")
    x = np.arange(lo, hi + step / 2, step)
    base = baseline[0] + baseline[1] * (x - 0.5 * (lo + hi))
    rng = np.random.default_rng(seed)
    T_meas = base * np.interp(x, nu_f, T_conv) + rng.normal(0.0, noise, len(x))
    return x, T_meas, np.interp(x, nu_f, T_true)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lines", default=os.path.join(HERE, "Input", "HITRAN", "N2O_2214.00-2221.00.data"))
    ap.add_argument("--ils", default=os.path.join(HERE, "Input", "ILS_LINEFIT.txt"))
    ap.add_argument("--ppm", type=float, default=500.0)
    ap.add_argument("--gas", nargs=2, action="append", metavar=("LINES", "PPM"),
                    help="one gas of a mixture (repeat); replaces --lines/--ppm")
    ap.add_argument("--T", type=float, default=297.15, help="K")
    ap.add_argument("--P", type=float, default=9456.5556, help="total pressure, Pa")
    ap.add_argument("--L", type=float, default=316.9, help="path length, cm")
    ap.add_argument("--lo", type=float, default=2215.5)
    ap.add_argument("--hi", type=float, default=2220.0)
    ap.add_argument("--step", type=float, default=0.00188, help="output grid spacing, cm-1")
    ap.add_argument("--noise", type=float, default=1e-3, help="standard deviation of the noise (transmittance)")
    ap.add_argument("--baseline", type=float, nargs=2, default=(0.985, 0.002), metavar=("LEVEL", "SLOPE"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ils = sio.load_ils(a.ils)
    comps = [(p, float(v)) for p, v in a.gas] if a.gas else [(a.lines, a.ppm)]
    x, Tm, _ = simulate_mix(comps, a.T, a.P, a.L, ils, a.lo, a.hi, a.step, noise=a.noise,
                            baseline=tuple(a.baseline), seed=a.seed)
    a.lines, a.ppm = comps[0]                 # the first gas is the one the header's "ppm" names
    out = a.out or os.path.join(HERE, "Input", "N2O_simulated_%gppm.txt" % a.ppm)
    mix = "".join("gas %s %.10g ppm\n" % (os.path.relpath(p, HERE), v) for p, v in comps) if len(comps) > 1 else ""
    header = ("Simulated transmittance (simulate_spectrum.py, HAPI line-by-line)\n"
              "line list %s\n%sILS %s\n"
              "ppm %.10g   T %.10g K   P %.10g Pa   L %.10g cm\n"
              "baseline %.6g + %.6g*(v - %.6g)   noise %.3g   seed %d\n"
              "wavenumber_cm-1\ttransmittance"
              % (os.path.relpath(a.lines, HERE), mix, os.path.relpath(a.ils, HERE), a.ppm, a.T, a.P, a.L,
                 a.baseline[0], a.baseline[1], 0.5 * (a.lo + a.hi), a.noise, a.seed))
    sio.save_columns(out, [x, Tm], header=header)
    print("wrote %s (%d points, %.4f-%.4f cm-1, %g ppm)" % (out, len(x), x[0], x[-1], a.ppm))


if __name__ == "__main__":
    main()
