# -*- coding: utf-8 -*-
"""Multi-gas retrieval on a simulated N2O + H2O + CO spectrum.

The window 2160.0-2163.5 cm-1 (low-wavenumber side of the N2O v3 band) holds
N2O lines, the CO R(4) line at 2161.97 cm-1 and the H2O line at
2161.73 cm-1, which sits between the N2O lines at 2161.68 and 2161.57 cm-1.
The spectrum is simulated with HAPI (simulate_spectrum.simulate_mix), then
retrieved three ways:

    multi-gas   N2O, H2O and CO fitted together (retrieval.fit_multigas)
    no H2O      N2O and CO only - H2O absorption left in the residual
    N2O only    the single-gas fit (retrieval.fit_spectrum)

    python multigas_example.py                 # one spectrum, plot in Results/
    python multigas_example.py --mc 20         # also 20 noise realisations
"""

import argparse
import os

import numpy as np

import concentration as conc
import hitran_fetch
import retrieval as rt
import simulate_spectrum as sim
import spectrum_io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
FETCH = (2156.0, 2168.0)                  # line lists: the window plus 2.5 cm-1 or more each side
REGION = (2160.0, 2163.5)
TRUE = {"N2O": 500.0, "H2O": 20000.0, "CO": 30.0}       # ppm; 2 % H2O ~ humid air
T_K, P_PA, L_CM = 297.15, 9456.5556, 316.9               # the cell of the N2O example
NOISE, BASELINE = 1e-3, (0.985, 0.002)


def line_file(mol):
    path = os.path.join(hitran_fetch.HITRAN_DIR, hitran_fetch.table_name(mol, *FETCH) + ".data")
    if not os.path.isfile(path):
        print("downloading %s %.2f-%.2f cm-1 from HITRAN ..." % (mol, *FETCH))
        path = hitran_fetch.fetch(mol, *FETCH)
    return path


# Every line here is optically thin (peak tau < 0.4), where (1 - z) e^-tau + z
# ~ 1 - (1 - z) tau: a zero offset z and a common scale on all the mixing
# ratios are the same thing.  Only saturated lines tell them apart, so z is
# held at 0 (fit it only when some line is black at its centre).
OPTS = {"fit_zero": False}


def fit_all(x, y, lines, ils):
    m = (x >= REGION[0]) & (x <= REGION[1])
    x, y = x[m], y[m]
    multi = rt.fit_multigas(x, y, [(g, lines[g]) for g in ("N2O", "H2O", "CO")], T_K, P_PA, L_CM, ils, OPTS)
    no_h2o = rt.fit_multigas(x, y, [(g, lines[g]) for g in ("N2O", "CO")], T_K, P_PA, L_CM, ils, OPTS)
    single = rt.fit_spectrum(x, y, lines["N2O"], T_K, P_PA, L_CM, ils, OPTS)
    return multi, no_h2o, single


def report(multi, no_h2o, single):
    def row(name, r):
        cells = []
        for g in TRUE:
            if isinstance(r["ppm"], dict) and g in r["ppm"]:
                v, e = r["ppm"][g], r["ppm_err"][g]
                cells.append("%10.2f +/- %-7.2f (%+6.2f %%)" % (v, e, 100 * (v - TRUE[g]) / TRUE[g]))
            elif g == "N2O" and not isinstance(r["ppm"], dict):
                v, e = r["ppm"], r["ppm_err"]
                cells.append("%10.2f +/- %-7.2f (%+6.2f %%)" % (v, e, 100 * (v - TRUE[g]) / TRUE[g]))
            else:
                cells.append("%-30s" % "        -")
        print("%-10s %s   rms %.2e" % (name, " ".join(cells), r["rms"]))

    print("\nregion %.2f-%.2f cm-1   T %.2f K   P %.1f Pa   L %.1f cm   noise %.0e"
          % (*REGION, T_K, P_PA, L_CM, NOISE))
    print("%-10s %s" % ("", " ".join("%-30s" % ("%s  (true %g ppm)" % (g, v)) for g, v in TRUE.items())))
    row("multi-gas", multi)
    row("no H2O", no_h2o)
    row("N2O only", single)
    print("\nmulti-gas: peak optical depth  %s"
          % "   ".join("%s %.3f" % kv for kv in multi["tau_max"].items()))
    print("multi-gas: correlation of the mixing ratios")
    g = multi["gases"]
    print("        " + "".join("%8s" % n for n in g))
    for i, n in enumerate(g):
        print("%8s" % n + "".join("%8.3f" % c for c in multi["corr"][i]))
    print("multi-gas: shift %.5f cm-1  broadening %.5f cm-1  zero %.4f  %d evaluations  %.1f s"
          % (multi["shift"], multi["broadening"], multi["zero"], multi["nfev"], multi["elapsed_s"]))


def plot(multi, no_h2o, single, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = multi["x"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1, 1]})
    ax[0].plot(x, multi["data"], ".", ms=2, color="0.4", label="simulated")
    ax[0].plot(x, multi["fit"], color="C3", lw=1, label="multi-gas fit")
    ax[0].set_ylabel("transmittance"); ax[0].legend(loc="lower left")
    base = multi["baseline_curve"]
    for k, (g, c) in enumerate(multi["components"].items()):
        ax[1].plot(x, c / base, color="C%d" % k, lw=1, label="%s %.1f ppm" % (g, multi["ppm"][g]))
    ax[1].set_ylabel("per gas / baseline"); ax[1].legend(loc="lower left", fontsize=8)
    ax[2].plot(x, multi["residual"], color="C3", lw=0.7)
    ax[2].set_ylabel("resid. multi")
    ax[3].plot(x, single["residual"], color="C0", lw=0.7, label="N2O only")
    ax[3].plot(x, no_h2o["residual"], color="C2", lw=0.7, label="N2O + CO")
    ax[3].set_ylabel("resid."); ax[3].legend(loc="lower left", fontsize=8)
    ax[3].set_xlabel("wavenumber (cm$^{-1}$)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print("plot: %s" % os.path.relpath(path, HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mc", type=int, default=0, help="number of extra noise realisations")
    ap.add_argument("--ils", default=os.path.join(HERE, "Input", "ILS_LINEFIT.txt"))
    a = ap.parse_args()

    paths = {g: line_file(g) for g in TRUE}
    lines = {g: conc.load_hitran(p) for g, p in paths.items()}
    ils = sio.load_ils(a.ils)
    comps = [(paths[g], v) for g, v in TRUE.items()]

    x, y, _ = sim.simulate_mix(comps, T_K, P_PA, L_CM, ils, *REGION, noise=NOISE, baseline=BASELINE, seed=1)
    out = os.path.join(HERE, "Input", "Mix_N2O_H2O_CO_simulated.txt")
    header = ("Simulated transmittance (multigas_example.py, HAPI line-by-line)\n"
              + "".join("gas %s %.10g ppm\n" % (os.path.relpath(p, HERE), v) for p, v in comps)
              # the "ppm ... T ... P ... L" line is what the GUI reads the cell conditions from;
              # its ppm is the first gas's
              + "ILS %s\nppm %.10g   T %.10g K   P %.10g Pa   L %.10g cm\n"
                "baseline %.6g + %.6g*(v - %.6g)   noise %.3g   seed 1\nwavenumber_cm-1\ttransmittance"
              % (os.path.relpath(a.ils, HERE), comps[0][1], T_K, P_PA, L_CM, *BASELINE, 0.5 * sum(REGION),
                 NOISE))
    sio.save_columns(out, [x, y], header=header)
    print("spectrum: %s" % os.path.relpath(out, HERE))

    multi, no_h2o, single = fit_all(x, y, lines, ils)
    report(multi, no_h2o, single)
    plot(multi, no_h2o, single, os.path.join(HERE, "Results", "multigas_example.png"))

    if a.mc:
        res = {k: {g: [] for g in TRUE} for k in ("multi-gas", "no H2O", "N2O only")}
        for s in range(2, 2 + a.mc):
            x, y, _ = sim.simulate_mix(comps, T_K, P_PA, L_CM, ils, *REGION, noise=NOISE, baseline=BASELINE,
                                       seed=s)
            m, n, o = fit_all(x, y, lines, ils)
            for g in TRUE:
                res["multi-gas"][g].append(m["ppm"][g])
                if g in n["ppm"]:
                    res["no H2O"][g].append(n["ppm"][g])
            res["N2O only"]["N2O"].append(o["ppm"])
        print("\n%d noise realisations: mean bias +/- standard deviation (%% of the true value)" % a.mc)
        for k, d in res.items():
            print("%-10s " % k + "   ".join("%s %+6.2f +/- %5.2f" % (g, 100 * (np.mean(v) / TRUE[g] - 1),
                                                                 100 * np.std(v) / TRUE[g])
                                           for g, v in d.items() if v))


if __name__ == "__main__":
    main()
