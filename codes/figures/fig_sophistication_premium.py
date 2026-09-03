#!/usr/bin/env python3
r"""
fig_sophistication_premium.py  --  Thrust #24 L2 (node_002) figure.

The sophistication premium across the repeating-structure class.  Reads ONLY
codes/results/sophistication_premium.npz (built by analysis/sophistication_premium.py
from the coupled-WMLES harvests) and codes/results/premium_class_tble.npz.

 (a) Fidelity ladder, per geometry: the coupled deployment error e (bulk
     reattachment-location error vs DNS) of the Spalding rung (green, drops
     dp/dx + convection) and the TBLE/ODE rung (black, keeps dp/dx, drops
     convection).  The premium Delta_{S->T}=e_Spalding-e_TBLE is the gap; on every
     O(delta)-pitch hill it is <= 0 -- paying for dp/dx buys NO deployment error
     back (it slightly hurts).  Geometries whose TBLE rung is still running are
     drawn with the Spalding bar solid and the TBLE bar HATCHED/open ("queued"):
     no number is invented.

 (b) Magnitude law: |Delta_{S->T}|/e versus the a-priori cancellation depth
     eps_median, with the order-of-magnitude prediction |Delta|/e ~ eps shown as
     the diagonal and a within-a-factor-of-2 band.  The two complete hills fall
     in the band.

Colour scheme (project convention): green=Spalding, black=TBLE/black-box,
orange=DNS truth reference, bluish-gray reserved for the (pending) CR-WM cure.
No fabrication: every plotted number is read from the npz; pending rungs are
shown as hatched placeholders, never as data.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_sophistication_premium.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
NPZ = os.path.join(RES, "sophistication_premium.npz")
OUT = os.path.join(CODES, "..", "manuscript", "figures", "fig_sophistication_premium.pdf")

C_SPALD = "green"
C_TBLE = "black"
C_TRUTH = "orange"

d = np.load(NPZ, allow_pickle=True)
table = list(d["premium_table"])   # list of dicts: geom, e_spalding, e_tble, delta_S_to_T, ...

# display order: canonical hill, second hill (Breuer), steepness family, control
order = ["xiao_1p0", "breuer", "xiao_0p8", "xiao_1p2", "convdiv"]
rows = {r["geom"]: r for r in table}
order = [g for g in order if g in rows]
pretty = {
    "xiao_1p0": "hills 1.0\n(Re5600)",
    "breuer":   "hills 1.0\n(Re10595)",
    "xiao_0p8": r"hills $\alpha$0.8",
    "xiao_1p2": r"hills $\alpha$1.2",
    "convdiv":  "conv-div\n(control)",
}

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.7))

# ---- panel (a): fidelity-ladder bars per geometry ----
xs = np.arange(len(order))
w = 0.38
for i, g in enumerate(order):
    r = rows[g]
    eS = r["e_spalding"]
    axA.bar(i - w / 2, eS, w, color=C_SPALD, alpha=0.85,
            label="Spalding" if i == 0 else None)
    if r.get("complete"):
        eT = r["e_tble"]
        axA.bar(i + w / 2, eT, w, color=C_TBLE, alpha=0.85,
                label="TBLE / ODE" if i == 0 else None)
        # annotate the premium (gap)
        dl = r["delta_S_to_T"]
        axA.annotate(f"$\\Delta$={dl:+.3f}", (i, max(eS, eT) + 0.012),
                     ha="center", fontsize=6.5, color="0.2")
    else:
        # queued TBLE rung -- hatched placeholder at the Spalding level (NOT data)
        axA.bar(i + w / 2, eS, w, facecolor="none", edgecolor=C_TBLE,
                hatch="////", lw=0.8,
                label="TBLE (queued)" if g == order[2] else None)
        axA.annotate("queued", (i + w / 2, eS + 0.012), ha="center",
                     fontsize=6, color="0.4", rotation=90, va="bottom")

axA.set_xticks(xs)
axA.set_xticklabels([pretty[g] for g in order], fontsize=7.5)
axA.set_ylabel(r"coupled deployment error  $e=|x_{r}^{\rm WMLES}-x_{r}^{\rm DNS}|/x_{r}^{\rm DNS}$")
axA.set_title("(a) fidelity ladder: premium $\\leq 0$ on every $O(\\delta)$-pitch hill",
              fontsize=8.5)
axA.legend(fontsize=6.8, loc="upper center", framealpha=0.9)
axA.set_ylim(0, 0.50)
axA.axvline(3.5, color="0.6", ls=":", lw=0.8)
axA.text(4.0, 0.43, "control\n$\\epsilon\\sim O(1)$", ha="center", fontsize=6.5, color="0.35")

# ---- panel (b): |Delta|/e vs eps_median, within-factor-2 band ----
epsg, ratg, labg = [], [], []
for g in order:
    r = rows[g]
    if r.get("complete") and np.isfinite(r.get("eps_median", np.nan)) and r["eps_median"] > 0:
        epsg.append(r["eps_median"]); ratg.append(r["rel_premium"]); labg.append(g)
epsg = np.array(epsg); ratg = np.array(ratg)

xx = np.logspace(-2, 0.3, 50)
axB.plot(xx, xx, color="0.3", lw=1.2, label=r"$|\Delta|/e=\epsilon$ (prediction)")
axB.fill_between(xx, 0.5 * xx, 2.0 * xx, color="0.7", alpha=0.3,
                 label=r"within factor 2")
blab = {"xiao_1p0": "hills 1.0 (Re5600)", "breuer": "hills 1.0 (Re10595)",
        "xiao_0p8": r"hills $\alpha$0.8", "xiao_1p2": r"hills $\alpha$1.2",
        "convdiv": "conv-div"}
if len(epsg):
    axB.scatter(epsg, ratg, s=70, color=C_TRUTH, edgecolor="k", zorder=5)
    for e, rr, g in zip(epsg, ratg, labg):
        axB.annotate(blab.get(g, g), (e, rr), textcoords="offset points",
                     xytext=(9, -3), fontsize=6.5)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"a-priori cancellation depth  $\epsilon_{\rm median}$")
axB.set_ylabel(r"relative premium  $|\Delta_{S\to T}|/e$")
axB.set_title("(b) magnitude law: $|\\Delta|/e$ order-1 consistent with $\\epsilon$",
              fontsize=8.5)
axB.legend(fontsize=6.8, loc="upper left")
axB.set_xlim(0.03, 1.2); axB.set_ylim(0.03, 1.2)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"[fig] wrote {OUT}")
print(f"[fig] complete geometries plotted: {labg}")
print(f"[fig] queued (hatched) geometries: "
      f"{[g for g in order if not rows[g].get('complete')]}")
