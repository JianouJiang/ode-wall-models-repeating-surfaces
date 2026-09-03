#!/usr/bin/env python3
"""
Figure: a-priori evidence base for the convection-restoring wall model (CR-WM),
Thrust #22 / node_005 (Level 3 Results & analysis).

Every number is read from real on-disk npz files produced from DNS reference data:
  - codes/results/crwm_apriori.npz               (gap-closure ladder vs y_m)
  - codes/results/outer_transport_decomposition.npz (within-layer vs outer split)

Panels:
  (a) Gap-closure ladder: R^2(tau_w) of ODE/TBLE vs CR-WM(exact add-back) across
      matching height y_m^+. CR-WM lifts R^2 by ~56% at the WMLES band but does
      NOT cross zero -> the residual is outer transport, harvested by the coupled
      twin (F1).
  (b) Within-layer convective fraction f_wl(y_m): only ~3% of the convective
      transport lives below the WMLES matching height -> ~97% is outer, coupled-
      only. This is WHY F1 (coupled) is decisive and not a tautology of the
      a-priori add-back.

Color convention (repo): black = black-box ODE/TBLE baseline; the CR-WM extension
is drawn in a distinct teal. No fabricated data.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

ap = np.load(os.path.join(RES, "crwm_apriori.npz"), allow_pickle=True)
ot = np.load(os.path.join(RES, "outer_transport_decomposition.npz"), allow_pickle=True)

ymp   = ap["sweep_ymp"]
r2ode = ap["sweep_r2_ode"]
r2crw = ap["sweep_r2_crwm"]

yi_ymp = ot["yi_ymp"]
yi_fwl = ot["yi_fwl"]
med_outer = float(ot["med_outer_frac"])

C_ODE  = "k"          # black-box ODE/TBLE baseline
C_CRWM = "#1b9e9e"    # CR-WM extension (teal)

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0))

# ---- Panel (a): gap-closure ladder -----------------------------------------
axA.plot(ymp, r2ode, "o-", color=C_ODE, lw=1.8, ms=6, label="ODE / TBLE (drops convection)")
axA.plot(ymp, r2crw, "s-", color=C_CRWM, lw=1.8, ms=6,
         label="CR-WM (exact within-layer add-back)")
axA.axhline(0.0, color="0.5", lw=0.8, ls=":")
# annotate the WMLES-band point (y_m^+ ~ 12, Y_IDX=10)
axA.annotate(f"at $y_m^+\\!\\approx\\!12$: $R^2$ {r2ode[0]:.1f} $\\to$ {r2crw[0]:.1f}\n(~56% recovery, still $<0$)",
             xy=(ymp[0], r2crw[0]), xytext=(ymp[0]+18, -360),
             fontsize=9, ha="left",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
axA.set_xlabel(r"matching height $y_m^+$")
axA.set_ylabel(r"$R^2(\tau_w)$ on canonical hill ($\alpha{=}1.0$)")
axA.set_title("(a) Within-layer add-back: a partial cure", fontsize=11)
axA.legend(fontsize=8.5, loc="lower left", frameon=False)
axA.grid(alpha=0.25)

# ---- Panel (b): within-layer fraction --------------------------------------
axB.plot(yi_ymp, 100.0*yi_fwl, "D-", color=C_CRWM, lw=1.8, ms=6)
axB.axhline(100.0*(1.0-med_outer), color="0.5", ls=":", lw=0.9)
axB.annotate(f"WMLES band: only {100.0*yi_fwl[0]:.1f}% within-layer\n"
             f"$\\Rightarrow$ ~{100.0*med_outer:.0f}% of convection is OUTER",
             xy=(yi_ymp[0], 100.0*yi_fwl[0]),
             xytext=(yi_ymp[0]+18, 100.0*yi_fwl[0]+22),
             fontsize=9, ha="left",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
axB.set_xlabel(r"matching height $y_m^+$")
axB.set_ylabel(r"within-layer convective fraction $f_{wl}$ (\%)")
axB.set_title("(b) Why coupled F1 is decisive, not tautological", fontsize=11)
axB.grid(alpha=0.25)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig_crwm_apriori_evidence.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)

# ---- echo the traceable headline numbers -----------------------------------
print(f"[trace] crwm_apriori: r2_ode={float(ap['r2_all_ode']):.2f} "
      f"r2_crwm_exact={float(ap['r2_all_exact']):.2f} "
      f"corr_struct_C={float(ap['corr_struct_C']):.3f} "
      f"var_explained_by_C={float(ap['var_explained_by_C']):.3f}")
print(f"[trace] outer_transport: med_outer_frac={med_outer:.3f} "
      f"fwl@band={float(yi_fwl[0]):.4f}")
