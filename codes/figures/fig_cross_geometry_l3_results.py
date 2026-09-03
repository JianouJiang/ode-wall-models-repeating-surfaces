#!/usr/bin/env python3
r"""
fig_cross_geometry_l3_results.py
================================
L3 results figure for the shape-agnostic conditioning-floor iteration.  Three
panels, all from results/cross_geometry_l3_results.npz (no fabrication):

 (a) Denominator-robust failure metric relRMSE vs the standard R2, per geometry.
     The near-flat wavy anchor (cv(tau_w)=0.006) carries a catastrophic R2 but a
     control-grade relRMSE: its R2 is a SS_tot->0 denominator artefact, not a
     failure.  relRMSE >= 1 cleanly flags the three genuine repeating failures.
 (b) Within-geometry 1/eps amplification law on the hill (n=512): kappa_closure
     vs 1/eps, decisive (Spearman 0.89, p<1e-178).  The "deeper -> larger error"
     statement is NOT n=2.
 (c) Consolidation with the severity collapse: the conditioning floor prefactor
     band coincides with the independently-measured severity-law floor constant.

Colour convention (CLAUDE.md): orange = ground truth / failure family,
bluish-gray = tolerated/controls, green = the denominator-robust metric.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
MSFIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

ORANGE = "#e8731a"
GRAY = "#6b7a8f"
GREEN = "#2a8a4a"
RED = "#c0392b"

d = np.load(os.path.join(RESULTS, "cross_geometry_l3_results.npz"), allow_pickle=True)
tags = [str(t) for t in d["tags"]]
roles = [str(r) for r in d["roles"]]
relrmse_lo = d["relrmse_lo"]; relrmse_hi = d["relrmse_hi"]
r2_worst = d["r2_worst"]
cv = d["cv_tau"]

SHORT = {"periodic_hill_1p0": "hill", "sharp_rib_dtype": "rib", "wavy_a10": "wavy",
         "bfs_Re13700": "BFS", "conv_div_Re12600": "conv-div", "nasa_hump": "hump",
         "sep_bubble_caseB": "bubble", "wavy_flat": "near-flat\n(a$\\to$0)"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# ---- (a) relRMSE vs R2 ---------------------------------------------------- #
ax = axes[0]
relrmse_mid = 0.5 * (relrmse_lo + relrmse_hi)
for i, tag in enumerate(tags):
    role = roles[i]
    if role == "failure":
        c, mk = ORANGE, "o"
    elif role == "control":
        c, mk = GRAY, "s"
    else:
        c, mk = RED, "D"
    # clip R2 for display (log of -R2)
    x = -r2_worst[i]
    yerr = [[relrmse_mid[i] - relrmse_lo[i]], [relrmse_hi[i] - relrmse_lo[i] - (relrmse_mid[i] - relrmse_lo[i])]]
    ax.errorbar(max(x, 0.3), relrmse_mid[i],
                yerr=[[relrmse_mid[i] - relrmse_lo[i]], [relrmse_hi[i] - relrmse_mid[i]]],
                fmt=mk, color=c, ms=9, capsize=3, mec="k", mew=0.5, zorder=3)
    dx = 1.15 if tag != "wavy_flat" else 0.55
    ax.annotate(SHORT.get(tag, tag), (max(x, 0.3), relrmse_mid[i]),
                textcoords="offset points", xytext=(7, 4), fontsize=8.5)
ax.axhline(1.0, ls="--", color="k", lw=1, alpha=0.7)
ax.text(0.45, 1.08, "relRMSE = 1 (failure threshold)", fontsize=8, alpha=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$-R^2(\tau_w)$  (worst model closure; larger $=$ worse)")
ax.set_ylabel(r"relRMSE $= \mathrm{RMS}(\tau_w^{\rm pred}-\tau_w^{\rm true})/\langle|\tau_w|\rangle$")
ax.set_title("(a) denominator-robust failure metric", fontsize=11)
# annotate the artefact
ax.annotate("near-flat anchor:\n$R^2$ degenerate (cv$=0.006$)\nbut relRMSE control-grade",
            xy=(2500, 0.25), xytext=(120, 3.2), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2), color=RED)
# legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", mfc=ORANGE, mec="k", ms=9, label="repeating failure"),
       Line2D([0], [0], marker="s", color="w", mfc=GRAY, mec="k", ms=9, label="control (tolerated)"),
       Line2D([0], [0], marker="D", color="w", mfc=RED, mec="k", ms=9, label="near-flat anchor")]
ax.legend(handles=leg, fontsize=8.5, loc="lower right", framealpha=0.9)

# ---- (b) within-hill 1/eps law -------------------------------------------- #
ax = axes[1]
eps = d["eps__periodic_hill_1p0"]; kmed = d["kmed__periodic_hill_1p0"]
m = np.isfinite(eps) & (eps > 0) & np.isfinite(kmed) & (kmed > 0)
inv = 1.0 / eps[m]
ax.loglog(inv, kmed[m], "o", color=ORANGE, ms=4, alpha=0.55, mec="none", label="hill stations ($n=512$)")
# floor guide kappa = beta/eps = beta*inv
beta = float(np.nanmedian((kmed[m] * eps[m])[eps[m] < 0.3]))
xg = np.array([inv.min(), inv.max()])
ax.loglog(xg, beta * xg, "-", color="k", lw=1.6, label=r"$\kappa=\beta/\varepsilon$,  $\beta=%.3f$" % beta)
rho = float(d["within_rho"][list(d["within_tags"]).index("periodic_hill_1p0")])
p = float(d["within_p"][list(d["within_tags"]).index("periodic_hill_1p0")])
ax.set_xlabel(r"$1/\varepsilon$  (cancellation depth; right $=$ deeper)")
ax.set_ylabel(r"$\kappa_{\mathrm{closure}}$  (median over closures A--D)")
ax.set_title("(b) within-hill amplification law (not $n{=}2$)", fontsize=11)
ax.text(0.04, 0.93, "Spearman$(\\kappa,1/\\varepsilon)=%.2f$\n$p<10^{-178}$,  $n=512$" % rho,
        transform=ax.transAxes, fontsize=9.5, va="top",
        bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))
ax.legend(fontsize=8.5, loc="lower right")

# ---- (c) severity consolidation ------------------------------------------- #
ax = axes[2]
cond_lo = float(d["cond_beta_lo"]); cond_hi = float(d["cond_beta_hi"])
b_emp = float(d["severity_beta_emp"]); b_p5 = float(d["severity_beta_p5"])
# horizontal floor bands on a log-beta axis
ax.axhspan(cond_lo, cond_hi, xmin=0.08, xmax=0.45, color=ORANGE, alpha=0.35,
           label=r"conditioning floor $\beta$ (closure perturbation)")
ax.axhspan(b_emp, b_p5, xmin=0.55, xmax=0.92, color=GREEN, alpha=0.30,
           label=r"severity-law $\beta$ (deployed rel-err, 882 stn)")
ax.plot([0.26], [np.sqrt(cond_lo * cond_hi)], "o", color=ORANGE, ms=10, mec="k")
ax.plot([0.73], [np.sqrt(b_emp * b_p5)], "s", color=GREEN, ms=10, mec="k")
ax.set_yscale("log")
ax.set_xlim(0, 1); ax.set_xticks([0.26, 0.73])
ax.set_xticklabels(["conditioning\n(this work)", "severity law\n(node_004)"], fontsize=9)
ax.set_ylabel(r"closure-blind floor constant $\beta$")
ax.set_title(r"(c) one floor, two probes of $\mathrm{relErr}\geq\beta/\varepsilon$", fontsize=11)
ax.axhline(0.1, ls=":", color="k", lw=0.8, alpha=0.5)
ax.text(0.5, 0.105, r"$O(10^{-1})$", fontsize=8, ha="center", alpha=0.7)
ax.axhline(0.01, ls=":", color="k", lw=0.8, alpha=0.5)
ax.text(0.5, 0.0105, r"$O(10^{-2})$", fontsize=8, ha="center", alpha=0.7)
ax.set_ylim(5e-3, 3e-1)
ax.legend(fontsize=8, loc="upper center", framealpha=0.9)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_cross_geometry_l3_results." + ext), dpi=160, bbox_inches="tight")
    os.makedirs(MSFIG, exist_ok=True)
    fig.savefig(os.path.join(MSFIG, "fig_cross_geometry_l3_results." + ext), dpi=160, bbox_inches="tight")
print("wrote fig_cross_geometry_l3_results.{pdf,png} to codes/figures and manuscript/figures")
