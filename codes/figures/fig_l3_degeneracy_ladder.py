#!/usr/bin/env python3
r"""
Figure (Thrust #27, L3): two on-disk facts that explain WHY the cross-geometry
static placement ratio fails, and why the failure is structural not a tunable
placement artifact.

Consumes codes/results/thrust27_l3_transmission.npz.

(a) MATCHING-HEIGHT DEGENERACY.  Across all five coupled WMLES the deployed
    matching height self-selects the buffer layer: y_m+ in [4.5,7.4] (a factor
    <2), while the ODE's own catastrophe height y_crit+ spans a factor ~175.
    Consequence: the static ratio r=y_m/y_crit collapses onto 1/y_crit
    (corr(log r, -log y_crit)=+1.00) and carries essentially no y_m signal -- you
    cannot escape the failure by re-placing y_m, because the coupled solver will
    not place it in the log layer at a feasible near-wall resolution.  The
    failure is therefore a property of the geometry (eps), not of the match
    height.

(b) COUPLED SOPHISTICATION LADDER (alpha=1.0 hill).  Replacing the algebraic
    Spalding law by the full TBLE ODE does NOT rescue reattachment
    (e_reatt: 0.333 -> 0.387; premium Delta_{S->T}=+0.05 >= 0).  The
    convection-restoring CR-WM arm -- the only model that adds the missing
    physics -- is still LIVE; its bar is shown hatched/open with NO number, to be
    harvested when the run lands.

Colours: Spalding = green (paper convention); TBLE ODE = bluish-gray (gray-box
convention); CR-WM = open/hatched (unlanded).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "thrust27_l3_transmission.npz")
OUTPDF = os.path.join(HERE, "fig_l3_degeneracy_ladder.pdf")
OUTPNG = os.path.join(HERE, "fig_l3_degeneracy_ladder.png")

d = np.load(RES, allow_pickle=True)
plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))

# ---------------------------------------------------------------- panel (a)
C_labels = [str(x) for x in d["C_labels"]]
ym = d["C_ym"]; ycrit = d["C_ycrit"]
short = []
for s in C_labels:
    if "Re10595" in s:
        short.append(r"hill $\alpha$=1.0 (Re10595)")
    elif "conv" in s.lower() or "Laval" in s:
        short.append("conv-div (control)")
    else:
        short.append(s.replace("hill ", "hill $").replace("alpha=", r"\alpha=")
                      .replace(" (Re5600)", "$ (Re5600)"))
yy = np.arange(len(C_labels))[::-1]
ax[0].hlines(yy, ym, ycrit, color="0.7", lw=1.0, zorder=1)
ax[0].plot(ym, yy, "o", ms=8, color="darkorange", mec="k", mew=0.5,
           label=r"deployed $y_m^+$ (buffer layer)", zorder=3)
ax[0].plot(ycrit, yy, "s", ms=8, color="crimson", mec="k", mew=0.5,
           label=r"catastrophe $y_{\rm crit}^+$", zorder=3)
ax[0].axvspan(4.5, 7.4, color="darkorange", alpha=0.10, lw=0)
ax[0].set_xscale("log")
ax[0].set_yticks(yy)
ax[0].set_yticklabels(short, fontsize=7.4)
ax[0].set_xlabel(r"wall-normal location  $y^+$")
ym_span = float(d["C_ym_span"]); yc_span = float(d["C_ycrit_span"])
ax[0].set_title(fr"(a) $y_m^+$ degeneracy: span $\times${ym_span:.1f} vs "
                fr"$y_{{\rm crit}}^+$ $\times${yc_span:.0f}", fontsize=9.5)
ax[0].legend(fontsize=7.4, loc="center right", bbox_to_anchor=(1.0, 0.40),
             framealpha=0.92)
ax[0].grid(True, which="both", axis="x", alpha=0.18)

# ---------------------------------------------------------------- panel (b)
spald = float(d["D_spalding"]); tble = float(d["D_tble"])
crwm_present = bool(d["D_crwm_present"])
names = ["Spalding\n(algebraic)", "TBLE ODE\n(full)", "CR-WM\n(convection)"]
vals = [spald, tble, np.nan]
colors = ["green", "#6a7b8a", "none"]
xb = np.arange(3)
for i in range(3):
    if i < 2:
        ax[1].bar(xb[i], vals[i], color=colors[i], edgecolor="k", width=0.62,
                  zorder=3)
        ax[1].text(xb[i], vals[i] + 0.012, f"{vals[i]:.3f}", ha="center",
                   fontsize=8.5)
    else:
        ax[1].bar(xb[i], spald, color="none", edgecolor="0.5", width=0.62,
                  hatch="////", zorder=3)
        ax[1].text(xb[i], spald * 0.5,
                   "live\n(no number\nuntil landed)", ha="center", va="center",
                   fontsize=6.8, color="0.4")
ax[1].set_xticks(xb)
ax[1].set_xticklabels(names, fontsize=7.8)
ax[1].set_ylabel(r"coupled reattachment error  $e_{\rm reatt}$")
prem = float(d["D_premium"])
ax[1].set_title(fr"(b) sophistication ladder  $\Delta_{{S\to T}}={prem:+.3f}$",
                fontsize=9.5)
ax[1].set_ylim(0, max(spald, tble) * 1.32)
ax[1].grid(True, axis="y", alpha=0.18)
ax[1].annotate("more physics,\nno rescue", xy=(0.5, tble), xytext=(0.5, tble * 1.18),
               fontsize=7.2, ha="center", color="0.3")

fig.tight_layout()
fig.savefig(OUTPDF, bbox_inches="tight")
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print("wrote", OUTPDF)
print("wrote", OUTPNG)
