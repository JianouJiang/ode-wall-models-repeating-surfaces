#!/usr/bin/env python3
r"""
fig_ym_robustness.py
====================
Thrust #12 (L3) -- matching-height robustness of the closure--structure
compensation.  Consumes results/compensation_ym_robustness.npz (produced by
codes/analysis/compensation_ym_robustness.py) and renders a 2-panel figure
showing that the exact-DNS-stress paradox and its convective explanation are
INVARIANT to the wall-model matching height across the coupled-WMLES first-cell
band (y_m^+ ~ 5-17) and into the log layer.

Neutral diagnostic palette (NOT the reserved model-role colours
orange/black/bluish-gray/green, which encode wall-model roles elsewhere): this
figure compares matching heights, not model roles.

Run: MPLBACKEND=Agg python3 codes/figures/fig_ym_robustness.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGDIR = os.path.join(os.path.dirname(ROOT), "manuscript", "figures")

d = np.load(os.path.join(RES, "compensation_ym_robustness.npz"), allow_pickle=True)
yp = d["med_y_m_plus"]
ratio = d["compensation_ratio"]
r2_ml = d["r2_ml"]
r2_dns = d["r2_dns"]
r_sc = d["corr_struct_C"]
rho_sc = d["spearman_struct_C"]
idx = d["idx_sweep"]

# coupled-WMLES first-cell band measured in our a-posteriori runs
WMLES_LO, WMLES_HI = 5.0, 17.0
C_RATIO = "#b2182b"     # crimson  -- compensation ratio
C_ML = "#4d4d4d"        # dark grey -- standard ODE skill
C_DNS = "#2166ac"       # blue      -- exact-stress ODE skill
C_CORR = "#1a9850"      # green-ish -- independent cross-check (diagnostic, not Spalding role)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.7))

# ---- panel (a): exact-stress paradox vs matching height --------------------
ax0.axvspan(WMLES_LO, WMLES_HI, color="0.85", alpha=0.6, lw=0,
            label=r"coupled-WMLES band")
axr = ax0.twinx()
l1, = ax0.plot(yp, -r2_ml, "o-", color=C_ML, mfc="white", lw=1.6,
               label=r"$-R^2$ standard ODE")
l2, = ax0.plot(yp, -r2_dns, "s-", color=C_DNS, lw=1.6,
               label=r"$-R^2$ exact-stress ODE")
ax0.set_yscale("log")
ax0.set_ylabel(r"$-R^2(\tau_w)$  (skill deficit; higher $=$ worse)")
ax0.set_xlabel(r"matching height  $y_m^+$  (median over separated stations)")
l3, = axr.plot(yp, ratio, "^--", color=C_RATIO, lw=1.6,
               label=r"ratio $\overline{|E_{\rm struct}|}/\overline{|E_{\rm ML}|}$")
axr.axhline(1.0, color=C_RATIO, ls=":", lw=1.0)
axr.set_ylabel(r"compensation ratio  (exact stress worse $\Leftrightarrow >1$)",
               color=C_RATIO)
axr.tick_params(axis="y", labelcolor=C_RATIO)
axr.set_ylim(0, ratio.max() * 1.25)
# mark the manuscript value y_idx=10
i10 = int(np.where(idx == 10)[0][0])
ax0.annotate(r"$Y_{\rm IDX}{=}10$", xy=(yp[i10], -r2_ml[i10]),
             xytext=(yp[i10] + 1.5, -r2_ml[i10] * 0.35), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax0.set_title("(a) paradox holds at every matching height", fontsize=10)
lns = [l1, l2, l3]
ax0.legend(lns, [x.get_label() for x in lns], fontsize=7.5, loc="upper left",
           framealpha=0.9)

# ---- panel (b): independent convective cross-check vs matching height ------
ax1.axvspan(WMLES_LO, WMLES_HI, color="0.85", alpha=0.6, lw=0)
ax1.plot(yp, r_sc, "o-", color=C_CORR, lw=1.6,
         label=r"Pearson $r(E_{\rm struct},\,C)$")
ax1.plot(yp, rho_sc, "s--", color=C_CORR, mfc="white", lw=1.6,
         label=r"Spearman $\rho(E_{\rm struct},\,C)$")
ax1.axhline(0.0, color="0.5", lw=0.8)
ax1.set_ylim(0, 1.0)
ax1.set_xlabel(r"matching height  $y_m^+$  (median over separated stations)")
ax1.set_ylabel(r"correlation with independent convective integral $C$")
ax1.set_title("(b) exact-stress error IS the dropped convection", fontsize=10)
ax1.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax1.text(0.97, 0.05, "$C$ from the 2-D budget;\nnever enters the ODE",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.5,
         color="0.35")

fig.tight_layout()
out_pdf = os.path.join(FIGDIR, "fig_ym_robustness.pdf")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out_pdf)
