#!/usr/bin/env python3
r"""
plot_condition_number.py
========================
Figure: the wall-stress map is ill-conditioned in the cancellation regime.

Left  : closure-channel condition number kappa_closure vs cancellation parameter
        eps, per station of the canonical periodic hill (red), with the
        cross-geometry medians (success geometries + the wide-pitch repeating
        conv-div control) overlaid.  kappa_closure rises as eps -> 0 and is
        bounded below ~0.5 on every success geometry.
Right : the three input channels (pressure gradient, matching velocity,
        turbulence closure) binned by eps on periodic hills, showing that the
        pressure and closure channels are amplified by the cancellation while
        the velocity channel stays O(1).

Consumes results/condition_number.npz (real DNS/LES, produced by
codes/analysis/condition_number.py).  Colours follow the manuscript convention.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
FIGDIR = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

d = np.load(os.path.join(RESULTS, "condition_number.npz"), allow_pickle=True)
eps = d["hills_eps"]
kK = d["hills_kappa_closure"]
kP = d["hills_kappa_P"]
kU = d["hills_kappa_U"]
geom_rows = d["geom_rows"]

ok = np.isfinite(eps) & (eps > 0) & np.isfinite(kK)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

# ---- left: kappa_closure vs eps, hills stations + cross-geometry medians ----
ax.scatter(eps[ok], np.clip(kK[ok], 1e-3, None), s=14, c="#cc3311",
           alpha=0.5, edgecolors="none", label="periodic hills (per station)")
# guide line kappa ~ c/eps (cancellation-regime prefactor from the npz)
c = float(d["hills_closure_prefactor_eps_lt_0p3"])
xg = np.logspace(np.log10(0.01), np.log10(0.3), 50)
ax.plot(xg, c / xg, "k--", lw=1.3, label=r"$\kappa_{\rm closure}\sim %.2f/\varepsilon$" % c)

markers = {"conv_div_Re12600": ("D", "#4477aa", "conv-div (wide-pitch repeat)")}
for g in geom_rows:
    lbl = g["label"]
    mk, col, name = markers.get(lbl, ("s", "0.25", None))
    ax.scatter(g["eps_med"], max(g["kK_med"], 1.3e-2), s=70, marker=mk,
               facecolors=col, edgecolors="k", zorder=5,
               label=name if name else ("success geometries (median)"
                                        if lbl == "bfs_Re13700" else None))
ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"cancellation parameter $\varepsilon=|\tau_w|/(|dp/dx|\,y_m)$")
ax.set_ylabel(r"closure-channel condition number $\kappa_{\rm closure}$")
ax.set_title("(a) conditioning vs cancellation")
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
ax.set_ylim(1e-2, 1e1)

# ---- right: channel-resolved condition number binned by eps (hills) ----
edges = [0.0, 0.05, 0.1, 0.3, 1.0]
centers, mP, mU, mK = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = ok & (eps >= lo) & (eps < hi)
    if m.sum() == 0:
        continue
    centers.append(np.sqrt(max(lo, 0.02) * hi))
    mP.append(np.nanmedian(kP[m])); mU.append(np.nanmedian(kU[m]))
    mK.append(np.nanmedian(kK[m]))
centers = np.array(centers)
ax2.plot(centers, mP, "-o", color="#ee6677", label="pressure gradient")
ax2.plot(centers, mK, "-s", color="#888888", label="turbulence closure")
ax2.plot(centers, mU, "-^", color="#228833", label="matching velocity")
ax2.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel(r"cancellation parameter $\varepsilon$")
ax2.set_ylabel(r"condition number $\kappa_{\rm channel}$ (hills, binned)")
ax2.set_title("(b) every retained channel is amplified")
ax2.legend(fontsize=7, loc="upper left")

fig.tight_layout()
out = os.path.join(FIGDIR, "fig_condition_number")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("wrote", out + ".pdf")
