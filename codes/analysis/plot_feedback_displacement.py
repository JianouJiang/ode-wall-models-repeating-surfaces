#!/usr/bin/env python3
r"""
plot_feedback_displacement.py  --  Thrust #16 L2 diagnostic figure.

Two panels, both from codes/results/feedback_displacement.npz (real Krank
Re10595 DNS + harvested coupled WMLES):

  (a) The CANCELLATION BAND.  R^2(tau_w) of the convection-blind one-way model
      M(U_m^DNS) and the median cancellation depth eps, as the matching height
      y_m^+ is swept.  The model is well-conditioned at y_m^+ <~ 5 (eps ~ O(1))
      and becomes catastrophic deep in the band (y_m^+ >~ 20, eps <~ 0.2).  The
      coupled WMLES matches at y_m^+ ~ 5 -- BELOW the band.

  (b) One-way vs two-way per station at the coupled matching height.  The
      truth-fed one-way model tracks DNS tau_w; the coupled two-way stress
      carries the LES's own near-wall error.  There is no catastrophe to heal
      at this height -> the a-posteriori survival is matching-height placement,
      not feedback suppression.

Output: development/nodes/node_002/fig_feedback_displacement.png

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/plot_feedback_displacement.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.normpath(os.path.join(ROOT, "..", "development", "nodes",
                                    "node_002", "fig_feedback_displacement.png"))

d = np.load(os.path.join(RES, "feedback_displacement.npz"), allow_pickle=True)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- panel (a): the cancellation band -----------------------------------
yp = d["sweep_ymp"]
r2 = d["sweep_r2"]
eps = d["sweep_eps"]
ym_coupled = float(np.nanmedian(d["ymp"]))

ax.axhline(0, color="0.7", lw=0.8, zorder=0)
l1, = ax.plot(yp, r2, "o-", color="black", lw=2, label=r"$R^2(\tau_w)$ one-way $M(U_m^{\rm DNS})$")
ax.set_xscale("log")
ax.set_xlabel(r"matching height $y_m^+$")
ax.set_ylabel(r"$R^2(\tau_w)$  (convection-blind model)")
ax.set_ylim(-5, 1.2)

axb = ax.twinx()
l2, = axb.plot(yp, eps, "s--", color="tab:orange", lw=1.6,
               label=r"median $\varepsilon=|\tau_w|/(|dp/dx|\,y_m)$")
axb.set_ylabel(r"cancellation depth  $\varepsilon$", color="tab:orange")
axb.tick_params(axis="y", labelcolor="tab:orange")
axb.set_yscale("log")

# shade the catastrophe band (R^2 < 0)
band = yp[r2 < 0]
if band.size:
    ax.axvspan(band.min(), yp.max(), color="red", alpha=0.07, zorder=0)
    ax.text(band.min() * 1.15, -3.5, "cancellation\nband\n(catastrophe)",
            color="firebrick", fontsize=8, ha="left", va="center")
ax.axvline(ym_coupled, color="tab:green", lw=2, ls=":")
ax.text(ym_coupled * 1.1, 0.7, "coupled\nWMLES\nmatches here",
        color="tab:green", fontsize=8, ha="left", va="center")
ax.legend(handles=[l1, l2], loc="lower left", fontsize=8, framealpha=0.9)
ax.set_title("(a) the convection-blind catastrophe is a matching-height band")

# ---- panel (b): one-way vs two-way per station --------------------------
x = d["x"]
order = np.argsort(x)
td = d["tau_dns"][order]
t1 = d["tau_oneway"][order]
tc = d["tau_coupled"][order]
xs = x[order]
ix = np.arange(len(xs))
ax2.axhline(0, color="0.7", lw=0.8)
ax2.plot(ix, td, "o-", color="tab:orange", lw=2, ms=6, label=r"DNS truth $\tau_w$")
ax2.plot(ix, t1, "s--", color="black", lw=1.5, ms=5,
         label=r"one-way $M(U_m^{\rm DNS})$ (truth-fed)")
ax2.plot(ix, tc, "^:", color="tab:blue", lw=1.5, ms=6,
         label=r"two-way coupled $\tau_w^{c}$ (LES)")
ax2.set_xticks(ix)
ax2.set_xticklabels([f"{v:g}" for v in xs], fontsize=7)
ax2.set_xlabel(r"station $x/H$")
ax2.set_ylabel(r"$\tau_w$")
ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
r2_1 = float(d["r2_oneway"]); r2_2 = float(d["r2_twoway"])
ax2.set_title(r"(b) at $y_m^+\!\sim\!%.0f$: one-way $R^2$=%.2f $>$ two-way $R^2$=%.2f"
              % (ym_coupled, r2_1, r2_2), fontsize=10)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("saved ->", OUT)
