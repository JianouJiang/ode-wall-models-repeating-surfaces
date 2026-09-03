#!/usr/bin/env python3
"""fig_fidelity_ladder.py  —  synthesis figure (single panel, connected dot plot).

a-priori R^2(tau_w) on the canonical hill against wall-model sophistication, from
codes/results/fidelity_ladder.npz.  CLOSURE axis (5 closures, keep dp/dx, drop
convection): every rung is catastrophically negative and the exact-DNS-stress
rung is the worst.  CONVECTION axis (CR-WM add-back): the only rung that lifts
R^2 off the floor.  Scalar a-posteriori results (sophistication premium, cure
split) are stated in the manuscript caption, not embedded in the plot.

Colours: black-box ODE, bluish-gray closures, teal CR-WM (convection).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
FIGD = os.path.abspath(os.path.join(ROOT, "..", "manuscript", "figures"))
os.makedirs(FIGD, exist_ok=True)

C_BB = "#1a1a1a"      # black-box ODE (van Driest mixing length)
C_GRAY = "#6b7a8f"     # bluish-gray closures / exact-stress
C_CRWM = "#1f9e9e"     # teal CR-WM (convection axis)

d = np.load(os.path.join(RES, "fidelity_ladder.npz"), allow_pickle=True)
def g(k):
    v = d[k]; return v.item() if v.shape == () else v

clab = [str(x) for x in g("closure_axis_labels")]
cr2 = np.asarray(g("closure_axis_r2"), float)
order = np.argsort(cr2)[::-1]                 # least-negative (best) first
clab = [clab[i] for i in order]; cr2 = cr2[order]
r2_crwm = float(g("r2_apriori_crwm_exact"))   # CR-WM oracle ceiling

def short(L):
    if "van Driest" in L:
        return "van Driest\n(black-box)"
    if "Spalart" in L:
        return "SA-like"
    if "exact DNS" in L:
        return "exact DNS\nstress"
    return L

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4))

xcl = np.arange(len(clab))
xcr = len(clab) + 0.7

# closure branch: connected dots (flat, failing)
ax.plot(xcl, cr2, "-", color=C_GRAY, lw=1.2, zorder=2)
for i, (x_, v_) in enumerate(zip(xcl, cr2)):
    mc = C_BB if "van Driest" in clab[i] else C_GRAY
    ax.plot(x_, v_, "o", color=mc, ms=7, zorder=3)
    if v_ < -500:                                   # exact DNS stress
        ax.annotate(f"{v_:.0f}", (x_, v_), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center", fontsize=7, color=mc)
    else:
        ax.annotate(f"{v_:.0f}", (x_, v_), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=mc, va="bottom")

# convection axis: the CR-WM rung, lifted; arrow from the best closure up to it
ax.annotate("", xy=(xcr, r2_crwm), xytext=(xcl[0], cr2[0]),
            arrowprops=dict(arrowstyle="-|>", color=C_CRWM, lw=1.8,
                            connectionstyle="arc3,rad=-0.18"), zorder=2)
ax.plot([xcr], [r2_crwm], "D", color=C_CRWM, ms=9, zorder=4)
ax.annotate(f"CR-WM: {r2_crwm:.0f}", (xcr, r2_crwm),
            textcoords="offset points", xytext=(0, 9), ha="center", va="bottom",
            fontsize=7.6, color=C_CRWM, fontweight="bold")

ax.axhline(0, color="k", lw=0.8)
ax.text(xcr + 0.4, 1.4, "success", fontsize=7, color="0.45", ha="right", va="bottom")
ax.set_yscale("symlog")
ax.set_ylim(-1500, 9)
ax.set_xlim(-0.6, xcr + 0.9)
ax.set_xticks(list(xcl) + [xcr])
ax.set_xticklabels([short(L) for L in clab] + ["CR-WM\n(restore\nconv.)"],
                   fontsize=7)

ax.set_ylabel(r"a-priori $R^2(\tau_w)$  (canonical hill)")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGD, f"fig_fidelity_ladder.{ext}"), dpi=200,
                bbox_inches="tight")
print("WROTE", os.path.join(FIGD, "fig_fidelity_ladder.pdf"))
