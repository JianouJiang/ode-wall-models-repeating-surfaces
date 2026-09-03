#!/usr/bin/env python3
r"""Figure 4 (two panels): the ODE wall-model failure DEEPENS with Reynolds number.
Reads only codes/results/reynolds_scaling.npz (real DNS, a-priori).

 (a) the closure-free cancellation eps_C = |tau_w|/|C| ~ Re^-1.15 (R2=0.975) FALLS
     through the O(1) catastrophe threshold across 5 DNS (factor 15 in Re), because
     the impulse Phi is Re-invariant while |tau_w| ~ C_f(Re) -> 0.
 (b) the closure-independent bound B = beta_floor/eps_C is a RISING wall
     (d lnB / d lnRe = +1.15, 95% MC band), crossing B=1 (relErr>100%) at
     engineering Re -- inverting the high-Re "rescue".

The matching-height-confound and separation-sign-split checks (formerly panels c,d;
only two Reynolds numbers have full 2-D fields) are reported in the text of
section 3.5, not as sparse plots.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_reynolds_deepening.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-size parity with the dose-response reference (~7.1 pt printed labels):
# this canvas is 11.0 in wide at full \textwidth, so labels need ~15 pt.
plt.rcParams.update({"font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                     "legend.fontsize": 13.5})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
d = np.load(os.path.join(ROOT, "codes/results/reynolds_scaling.npz"), allow_pickle=True)

C_DNS, C_CF, C_EXTRA, C_SEP = "#d8732e", "#333333", "#7a3fbf", "#c0392b"

Re = np.asarray(d["Re_pts"], float); eps = np.asarray(d["epsC_pts"], float)
cf = np.asarray(d["cf_pts"], float); SHORT = [s.split()[0] for s in json.loads(str(d["series_src"]))]
p_re = float(d["slope_re"]); r2_re = float(d["r2_re"]); beta = float(d["beta_floor"])
Bb = np.asarray(d["B_bound"], float); Rev = np.asarray(d["Re_eval"], float)
Blo, Bmed, Bhi = (np.asarray(d[k], float) for k in ("B_band_lo", "B_band_med", "B_band_hi"))
is_dns = Rev <= Re.max() + 1.0

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.35))

# ----------------------------------------------------------- (a) running-eps law
a = ax[0]
xs = np.logspace(np.log10(Re.min() * 0.9), np.log10(Re.max() * 1.1), 50)
a.plot(xs, np.exp(np.polyval([p_re, np.log(eps[0]) - p_re * np.log(Re[0])], np.log(xs))),
       "-", color=C_DNS, lw=1.0, alpha=0.5,
       label=fr"$\varepsilon_C\propto Re^{{{p_re:.2f}}}$ ($R^2={r2_re:.3f}$)")
a.plot(Re, eps, "o", color=C_DNS, ms=8.5, zorder=5,
       label=r"$\varepsilon_C=|\tau_w|/|C|$ (5 DNS)")
a.axhline(1.0, color="k", lw=0.7, ls=":")
a.text(0.97, 0.57, r"$\varepsilon_C=1$ threshold", transform=a.transAxes, ha="right",
       fontsize=11.5, color="0.3")
a.axvspan(2800, 5600, color="0.5", alpha=0.10, lw=0)
seen = set()
for r, e, s in zip(Re, eps, SHORT):
    if s not in seen:
        a.annotate(s, (r, e), textcoords="offset points", xytext=(7, 5),
                   fontsize=11, color="#999"); seen.add(s)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$Re_b$"); a.set_ylabel(r"$\varepsilon_C$ (residual / dropped convection)")
a.grid(alpha=0.16, which="both")
a.legend(loc="lower left", fontsize=11.5, framealpha=0.92)
a.set_title(r"(a) $\varepsilon_C\propto Re^{-1.15}$ falls through $O(1)$", fontsize=15, loc="left")
a2 = a.twinx()
a2.plot(Re, cf * 1e3, "s--", color=C_CF, ms=5, lw=1.0, alpha=0.8, label=r"$C_f\times10^3$")
a2.set_ylabel(r"$C_f\times10^3$", color=C_CF, fontsize=13.5)
a2.tick_params(axis="y", labelcolor=C_CF, labelsize=12)

# ----------------------------------------------------------- (b) impossibility bound
b = ax[1]
b.fill_between(Rev, Blo, Bhi, color=C_EXTRA, alpha=0.13, lw=0, label="95% MC band")
b.plot(Rev[~is_dns], Bmed[~is_dns], "--", color=C_EXTRA, lw=1.5, label="extrapolated")
b.plot(Re, Bb, "o", color=C_DNS, ms=8.5, zorder=5,
       label=fr"$\mathcal{{B}}=\beta_\mathrm{{floor}}/\varepsilon_C$ ($\beta_\mathrm{{floor}}={beta:.2f}$)")
b.axhline(1.0, color="k", lw=0.8, ls=":")
b.text(0.03, 0.57, r"$\mathcal{B}=1$: relErr $>100\%$", transform=b.transAxes, fontsize=11.5, color="0.3")
b.axvspan(Re.max(), Rev.max(), color=C_EXTRA, alpha=0.05, lw=0)
b.text(2.0e5, 0.05, "extrapolation\n(not simulated)", fontsize=11.5, color=C_EXTRA,
       ha="center", va="center")
b.annotate(fr"$d\ln\mathcal{{B}}/d\ln Re=+{-p_re:.2f}>0$", xy=(8e3, 0.47), xytext=(1.0e3, 5.0),
           fontsize=12.5, color=C_SEP, weight="bold",
           arrowprops=dict(arrowstyle="->", color=C_SEP, lw=1.2))
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel(r"$Re_b$"); b.set_ylabel(r"closure-indep. bound $\mathcal{B}=\beta_\mathrm{floor}/\varepsilon_C$")
b.grid(alpha=0.16, which="both")
b.legend(loc="lower right", fontsize=11, framealpha=0.92)
b.set_title("(b) the impossibility bound is a rising wall", fontsize=15, loc="left")

fig.tight_layout()
MAND = os.path.join(ROOT, "manuscript", "figures")
NODE = os.path.join(ROOT, "development", "nodes", "node_002")
for outd in (MAND, NODE):
    os.makedirs(outd, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outd, f"fig_reynolds_deepening.{ext}"), dpi=160, bbox_inches="tight")
print("wrote 2-panel fig_reynolds_deepening to manuscript/figures and node_002")
