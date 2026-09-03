#!/usr/bin/env python3
"""
Thrust #31 L3 figure — STATISTICAL ADJUDICATION of the Reynolds-asymptotic
catastrophe.  Reads ONLY codes/results/reynolds_adjudication.npz (+ the canonical
reynolds_scaling.npz for the fit line), so the figure cannot drift from the
audited numbers.

Four panels:
 (a) eps_C(Re) with station-bootstrap 95% CIs, the power-law fit, the O(1)
     catastrophe threshold, and the onset Reynolds number Re* (with CI band).
 (b) bootstrap posterior of the power-law slope (deepening sign robust to n=5).
 (c) bootstrap posterior of the catastrophe-onset Re* (eps_C = 1).
 (d) cross-group / cross-code validation: a law fit on Breuer (2009) alone
     predicts the independent Krank (2018) points.

Colour convention: orange = DNS/ground-truth severity; grey = thresholds/nulls;
purple = the extrapolated onset; Breuer = filled circles, Krank = open squares.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")
OUT     = os.path.join(ROOT, "development", "nodes", "node_003")

ORANGE = "#E8820C"; GREY = "#6E6E6E"; PURPLE = "#7B3FA0"; BLUE = "#2C5F8A"

def load(fn):
    d = np.load(os.path.join(RESULTS, fn), allow_pickle=True)
    return {k: d[k] for k in d.files}

A = load("reynolds_adjudication.npz")
C = load("reynolds_scaling.npz")

Re   = np.asarray(A["Re_sorted"], float)
eps  = np.asarray(A["epsC_agg"], float)
elo  = np.asarray(A["epsC_ci_lo"], float)
ehi  = np.asarray(A["epsC_ci_hi"], float)
slope = float(C["slope_re"]); inter_eps = eps[0] / Re[0] ** slope
Restar = float(A["Restar_point"]); Rstar_lo = float(A["Restar_ci_lo"]); Rstar_hi = float(A["Restar_ci_hi"])
slopes = np.asarray(A["slopes"], float)
Rdraws = np.asarray(A["Restar_draws"], float)

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "figure.dpi": 130})
fig, ax = plt.subplots(2, 2, figsize=(10.4, 8.0))

# ---- (a) eps_C(Re) with CIs, fit, threshold, Re* -----------------------------
a = ax[0, 0]
is_breuer = Re <= 2800
xr = np.logspace(np.log10(600), np.log10(12000), 200)
a.plot(xr, inter_eps * xr ** slope, "-", color=ORANGE, lw=1.6, zorder=2,
       label=fr"fit $\varepsilon_C\propto Re^{{{slope:.2f}}}$")
a.errorbar(Re[is_breuer], eps[is_breuer],
           yerr=[eps[is_breuer]-elo[is_breuer], ehi[is_breuer]-eps[is_breuer]],
           fmt="o", ms=7, color=ORANGE, mec="k", ecolor=GREY, capsize=3, lw=1,
           label="Breuer DNS (2009)", zorder=4)
a.errorbar(Re[~is_breuer], eps[~is_breuer],
           yerr=[eps[~is_breuer]-elo[~is_breuer], ehi[~is_breuer]-eps[~is_breuer]],
           fmt="s", ms=8, mfc="white", color=ORANGE, mec=ORANGE, ecolor=GREY,
           capsize=3, lw=1, label="Krank DNS (2018)", zorder=4)
a.axhline(1.0, ls="--", color=GREY, lw=1.1)
a.text(700, 1.15, r"catastrophe threshold $\varepsilon_C=1$", color=GREY, fontsize=8.5)
a.axvspan(Rstar_lo, Rstar_hi, color=PURPLE, alpha=0.12, zorder=0)
a.axvline(Restar, color=PURPLE, lw=1.3, ls="-")
a.text(Restar*1.04, 7, fr"$Re^\ast={Restar:.0f}$"+"\n"+fr"[{Rstar_lo:.0f},{Rstar_hi:.0f}]",
       color=PURPLE, fontsize=8.5)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$Re_b$"); a.set_ylabel(r"$\varepsilon_C=\sum|\tau_w|/\sum|C|$")
a.set_title("(a) running cancellation with station-bootstrap 95% CI", fontsize=10)
a.legend(fontsize=7.8, loc="lower left", framealpha=0.9)
a.grid(alpha=0.25, which="both")

# ---- (b) slope posterior -----------------------------------------------------
b = ax[0, 1]
b.hist(slopes, bins=60, color=ORANGE, alpha=0.8, edgecolor="none", density=True)
s_lo, s_hi = float(A["slope_ci_lo"]), float(A["slope_ci_hi"])
b.axvspan(s_lo, s_hi, color=ORANGE, alpha=0.18)
b.axvline(0.0, color=GREY, ls="--", lw=1.2)
b.text(0.02, 0.92, f"P(slope<0)={float(A['frac_negative_slope'])*100:.2f}%\n"
                   f"95% CI [{s_lo:.2f}, {s_hi:.2f}]",
       transform=b.transAxes, fontsize=8.5, va="top",
       bbox=dict(fc="white", ec=GREY, alpha=0.85))
b.set_xlabel(r"power-law slope  $d\ln\varepsilon_C/d\ln Re$")
b.set_ylabel("posterior density")
b.set_title("(b) deepening sign robust to n=5 (two-level bootstrap)", fontsize=10)
b.grid(alpha=0.25)

# ---- (c) Re* posterior -------------------------------------------------------
c = ax[1, 0]
Rclip = Rdraws[(Rdraws > 500) & (Rdraws < 20000)]
c.hist(Rclip, bins=70, color=PURPLE, alpha=0.75, edgecolor="none", density=True)
c.axvspan(Rstar_lo, Rstar_hi, color=PURPLE, alpha=0.16)
c.axvline(Restar, color=PURPLE, lw=1.4)
for rr in (2800, 5600):
    c.axvline(rr, color=GREY, ls=":", lw=1.0)
c.text(0.97, 0.95, f"$Re^*$ = {Restar:.0f}\n95% CI [{Rstar_lo:.0f}, {Rstar_hi:.0f}]\n"
                   "(inside measured\nRe window)", transform=c.transAxes,
       fontsize=8.5, va="top", ha="right",
       bbox=dict(fc="white", ec=PURPLE, alpha=0.85))
c.set_xscale("log")
c.set_xlabel(r"catastrophe-onset Reynolds number  $Re^*$  ($\varepsilon_C=1$)")
c.set_ylabel("posterior density")
c.set_title("(c) a-priori onset of ODE catastrophe", fontsize=10)
c.grid(alpha=0.25, which="both")

# ---- (d) cross-code validation Breuer -> Krank -------------------------------
d = ax[1, 1]
pB = float(A["breuer_slope"]); bR = np.array([700., 1400., 2800.])
bE = eps[is_breuer]; kR = np.array([5600., 10595.]); kE = np.asarray(A["krank_meas"], float)
kP = np.asarray(A["krank_pred"], float); kLo = np.asarray(A["krank_pred_lo"], float)
kHi = np.asarray(A["krank_pred_hi"], float)
xr2 = np.logspace(np.log10(650), np.log10(12000), 200)
interB = bE[0] / bR[0] ** pB
d.plot(xr2, interB * xr2 ** pB, "-", color=BLUE, lw=1.5,
       label=fr"Breuer-only law $\propto Re^{{{pB:.2f}}}$")
d.fill_between(kR, kLo, kHi, color=BLUE, alpha=0.13, label="Breuer-law 95% band")
d.plot(bR, bE, "o", ms=7, color=BLUE, mec="k", label="Breuer (2009), fit set", zorder=4)
d.plot(kR, kE, "s", ms=9, mfc="white", mec=ORANGE, mew=1.8,
       label="Krank (2018), held-out", zorder=5)
for i in range(2):
    d.annotate(f"{float(A['krank_pred_relerr'][i])*100:.0f}%", (kR[i], kE[i]),
               textcoords="offset points", xytext=(6, 8), fontsize=8, color=ORANGE)
d.axhline(1.0, ls="--", color=GREY, lw=1.0)
d.set_xscale("log"); d.set_yscale("log")
d.set_xlabel(r"$Re_b$"); d.set_ylabel(r"$\varepsilon_C$")
d.set_title("(d) cross-code check: Breuer law predicts Krank", fontsize=10)
d.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
d.grid(alpha=0.25, which="both")

fig.suptitle("Statistical adjudication of the Reynolds-asymptotic ODE-wall-model "
             "catastrophe (periodic hills, a-priori)", fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_reynolds_adjudication.{ext}"), bbox_inches="tight")
print("saved fig_reynolds_adjudication.{pdf,png} ->", OUT)
