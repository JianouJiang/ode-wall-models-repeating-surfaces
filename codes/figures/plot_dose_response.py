#!/usr/bin/env python3
"""
Dose-response figure: the cancellation diagnostic and ODE wall-stress skill as
continuous functions of the geometric repeat parameter, across the Xiao et al.
(2020) 29-case parameterised periodic-hill DNS family (Re_b = 5600).

This is the figure that upgrades the cancellation parameter from a binary
*label* (periodic hills fail / others succeed) into a *dose-response law*: the
dominant geometric control is the recirculation length relative to the outer
scale, L_sep/delta. As L_sep/delta grows, the cancellation parameter epsilon
falls into the residual regime (epsilon < 0.1) and the ODE wall-stress R^2 drops
far below zero.

The converging-diverging (wavy) channel is plotted as a CROSS-FAMILY ANCHOR
(open star) — a different repeating geometry at the benign end (small
L_sep/delta, epsilon = O(1), R^2 > 0). It is NOT part of the Xiao family and is
excluded from the regression / Spearman statistic, consistent with the
methodology (it anchors, it does not interpolate).

HONESTY (G1): every periodic-hill point is real Xiao DNS; the conv-div point is
real Marquillie/Laval DNS. No blade/city/exchanger is plotted. (G2): all numbers
are read from codes/results/corrected_family_sweep_l0_20260825.npz and
repeating_structure_contrast.npz; the trend fit is recomputed deterministically
from those arrays. (G4): a-priori protocol (reference profile + dp/dx in).

REFERENCE (2026-08-25): the family points come from the REPAIRED estimator
(curvature-aware cubic on six fluid points).  The superseded through-origin
four-point fit that previously supplied them under-resolves the wall traction,
and its bias correlates with the very geometric variables plotted here, so a
figure drawn from it overstates the trend it is used to show.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/plot_dose_response.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                            # codes/
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "..", "manuscript", "figures")

sys.path.insert(0, os.path.join(ROOT, "utils"))
try:
    from plotting_utils import setup_style, save_figure
    # Drawn at the printed width (2026-08-26).  This was drawn on an 11.4 in
    # canvas and printed at 3.76 in, so its 15 pt type reached the page at
    # 4.9 pt.  Canvas and type are now the printed ones, and main.tex includes
    # it at \\textwidth.
    setup_style(font_size=7.2, use_tex=False)
except Exception:
    def save_figure(fig, name, fig_dir=FIG_DIR, formats=("pdf", "png")):
        os.makedirs(fig_dir, exist_ok=True)
        for fmt in formats:
            p = os.path.join(fig_dir, f"{name}.{fmt}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"  Saved: {p}")
        plt.close(fig)


def main():
    d = np.load(os.path.join(RESULTS, "corrected_family_sweep_l0_20260825.npz"),
                allow_pickle=True)
    g = d["agg_cv_Lsep_over_delta"]
    eps = d["agg_eps_median"]
    r2 = d["agg_r2"]
    alpha = d["agg_cv_alpha"]
    rho_eps = float(d["collapse_Lsep_over_delta_rho_eps_median"])
    rho_r2 = float(d["collapse_Lsep_over_delta_rho_r2"])
    rho_r2_lo = float(d["collapse_Lsep_over_delta_rho_r2_ci_lo"])
    rho_r2_hi = float(d["collapse_Lsep_over_delta_rho_r2_ci_hi"])

    # cross-family conv-div anchor (NOT in the Xiao family / regression)
    c = np.load(os.path.join(RESULTS, "repeating_structure_contrast.npz"), allow_pickle=True)
    cv_g = float(c["convdiv_Lsep_over_delta"])
    cv_eps = float(c["convdiv_eps_median"])
    cv_r2 = float(c["convdiv_r2"])

    # deterministic OLS trend of log10(eps) on L_sep/delta (Xiao points only)
    slope, intercept = np.polyfit(g, np.log10(eps), 1)
    fit_pred = slope * g + np.log10(eps).mean() * 0 + intercept
    ss_res = np.sum((np.log10(eps) - (slope * g + intercept)) ** 2)
    ss_tot = np.sum((np.log10(eps) - np.log10(eps).mean()) ** 2)
    fit_r2 = 1.0 - ss_res / ss_tot
    print(f"OLS log10(eps) ~ L_sep/delta:  slope={slope:.3f} dex,  "
          f"intercept={intercept:.3f},  fit R^2={fit_r2:.3f}")
    print(f"Spearman rho: eps_median {rho_eps:+.3f} | R2 {rho_r2:+.3f} "
          f"[{rho_r2_lo:+.3f}, {rho_r2_hi:+.3f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4757, 2.25))
    cmap = plt.cm.viridis
    amin, amax = 0.5, 1.5

    # ---- panel (a): cancellation parameter eps vs L_sep/delta ----
    ax1.axhspan(1e-3, 0.1, color="#d62728", alpha=0.06, zorder=0)   # residual regime
    ax1.axhline(0.1, color="0.5", ls=":", lw=0.8, zorder=1)
    gg = np.linspace(g.min() * 0.95, g.max() * 1.05, 50)
    ax1.plot(gg, 10 ** (slope * gg + intercept), "k--", lw=1.0, zorder=2,
             label=f"trend (fit $R^2$={fit_r2:.2f})")
    sc = ax1.scatter(g, eps, c=alpha, cmap=cmap, vmin=amin, vmax=amax,
                     s=26, edgecolors="k", linewidths=0.4, zorder=3)
    ax1.scatter([cv_g], [cv_eps], marker="*", s=150, facecolors="none",
                edgecolors="k", linewidths=1.3, zorder=4,
                label="conv-div (cross-family)")
    ax1.set_yscale("log")
    ax1.set_ylim(8e-4, 12.0)
    ax1.set_xlabel(r"$L_{\rm sep}/\delta$  (recirculation length / outer scale)")
    ax1.set_ylabel(r"median cancellation $\tilde{\varepsilon}$")
    ax1.text(0.30, 0.06, f"Spearman " + r"$\rho$" + f" = {rho_eps:+.2f}",
             transform=ax1.transAxes, fontsize=7.2, va="bottom")
    ax1.set_title("(a) cancellation depth", loc="left", fontsize=7.6)
    ax1.annotate("residual regime\n" + r"$\tilde{\varepsilon}<0.1$",
                 xy=(g.max() * 0.7, 0.012), fontsize=7.2, color="#d62728")
    ax1.legend(loc="upper right", fontsize=7.2)

    # ---- panel (b): ODE wall-stress R^2 vs L_sep/delta ----
    ax2.axhline(0.0, color="0.4", ls="-", lw=0.8, zorder=1)
    ax2.scatter(g, r2, c=alpha, cmap=cmap, vmin=amin, vmax=amax,
                s=26, edgecolors="k", linewidths=0.4, zorder=3)
    ax2.scatter([cv_g], [cv_r2], marker="*", s=150, facecolors="none",
                edgecolors="k", linewidths=1.3, zorder=4)
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_xlabel(r"$L_{\rm sep}/\delta$")
    ax2.set_ylabel(r"ODE wall-stress $R^2(\tau_w)$")
    # placed in the empty band above the (clustered, strongly-negative) hill points
    ax2.text(0.33, 0.55,
             f"Spearman " + r"$\rho$" + f" = {rho_r2:+.2f}\n[{rho_r2_lo:+.2f}, {rho_r2_hi:+.2f}]",
             transform=ax2.transAxes, fontsize=7.2, va="center", ha="left")
    ax2.set_title("(b) wall-stress accuracy", loc="left", fontsize=7.6)
    ax2.annotate(r"success ($R^2>0$)", xy=(cv_g + 0.2, 2.0), fontsize=7.2, color="0.3")

    cbar = fig.colorbar(sc, ax=[ax1, ax2], shrink=0.85, pad=0.02,
                        ticks=[0.5, 0.75, 1.0, 1.25, 1.5])
    cbar.set_label(r"hill steepness $\alpha$")

    save_figure(fig, "fig_amplitude_pitch_sweep", fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
