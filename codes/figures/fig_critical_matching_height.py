#!/usr/bin/env python3
r"""
fig_critical_matching_height.py  --  Thrust #17 L2 multi-geometry figure.

The critical matching height y_crit GENERALISES across geometries (it is computed
per geometry from the variance balance gamma=sigma_tau, not fitted).  Two panels:

 (a) The matching-height sweep relRMS(y_m+) = gamma/sigma_tau for every geometry.
     The variance balance relRMS=1 (R^2=0) is the horizontal line.  The deployed
     first-cell band (y_m+ <~ 60) is shaded.  The O(delta)-pitch repeating hills
     cross INSIDE the deployed band; the wide-pitch control and the non-repeating
     single-feature controls cross an order of magnitude higher (or never within
     the resolved range) -> safe.

 (b) The critical height y_crit+ per geometry on a log axis, with the deployed
     band shaded.  A clean order-of-magnitude gap separates the bifurcating
     O(delta)-pitch hills (y_crit+ <= 16) from the controls (y_crit+ >= 148).

Source: codes/results/critical_matching_height_map.npz
        (built by codes/analysis/critical_matching_height.py).

Palette (geometry classes, documented; not the model-curve convention):
  red    = repeating O(delta)-pitch hills (the bifurcating failure class)
  blue   = wide-pitch repeating control (conv-div)
  gray   = non-repeating single-feature controls (success class)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
OUT = os.path.join(ROOT, "development", "nodes", "node_002")
os.makedirs(OUT, exist_ok=True)

RED = "#D55E00"      # O(delta)-pitch hills (bifurcate in range)
BLUE = "#0072B2"     # wide-pitch control
GRAY = "#7A7A7A"     # single-feature controls

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.0, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
})

d = np.load(os.path.join(RES, "critical_matching_height_map.npz"), allow_pickle=True)
keys = [str(k) for k in d["keys"]]
klass = [str(k) for k in d["klass"]]
ycrit = d["ycrit"].astype(float)
eps_c = d["eps_c"].astype(float)
grid = d["ymp_grid"].astype(float)
DEP = float(d["deployed_ymp_max"])

LABELS = {
    "krank_pehill_Re10595": r"periodic hill (Krank, Re$_H$=10595)",
    "periodic_hills_1p0": r"periodic hill ($h/L_x$=1, resolved)",
    "conv_div_channel": "conv-div channel (wide pitch)",
    "bfs_Re13700": "backward-facing step",
    "curved_bfs_LES": "curved BFS",
    "nasa_hump": "NASA wall hump",
}
def color_of(kl):
    return {"repeating_Odelta": RED, "repeating_wide": BLUE}.get(kl, GRAY)
def style_of(kl):
    return {"repeating_Odelta": "-", "repeating_wide": "--"}.get(kl, ":")

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

# ---- (a) the sweep relRMS(y_m+) per geometry ------------------------------- #
ax = axes[0]
ax.axvspan(grid.min(), DEP, color="0.85", alpha=0.6, lw=0, zorder=0)
ax.text(np.sqrt(grid.min() * DEP), 5e1, "deployed\nfirst-cell band",
        ha="center", va="top", color="0.35", fontsize=8.0)
ax.axhline(1.0, color="0.25", lw=1.1, ls="-")
ax.text(grid.max() * 0.95, 1.12, r"variance balance  $\gamma=\sigma_\tau$  ($R^2=0$)",
        ha="right", va="bottom", color="0.25", fontsize=8.0)
for k, kl in zip(keys, klass):
    rr = d[f"sweep_relrms__{k}"].astype(float)
    m = np.isfinite(rr)
    ax.plot(grid[m], rr[m], style_of(kl), color=color_of(kl), lw=1.9,
            label=LABELS.get(k, k))
    # mark the crossing
    yc = ycrit[keys.index(k)]
    if np.isfinite(yc) and yc <= grid.max():
        ax.plot([yc], [1.0], "o", color=color_of(kl), ms=7, mfc="white",
                mec=color_of(kl), mew=1.6, zorder=5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"matching height $y_m^+$")
ax.set_ylabel(r"$\mathrm{relRMS}(\tau_w)=\gamma/\sigma_\tau$")
ax.set_title("(a) variance-balance sweep, all geometries")
ax.set_ylim(0.05, 30)
ax.legend(loc="lower right", frameon=False, fontsize=7.3)

# ---- (b) y_crit+ per geometry (log) ---------------------------------------- #
ax = axes[1]
order = np.argsort(np.where(np.isfinite(ycrit), ycrit, 1e6))
yvals = []
labels = []
colors = []
for idx in order:
    yc = ycrit[idx]
    yplot = yc if np.isfinite(yc) and yc <= grid.max() else grid.max() * 1.6
    yvals.append(yplot)
    labels.append(LABELS.get(keys[idx], keys[idx]))
    colors.append(color_of(klass[idx]))
ypos = np.arange(len(yvals))
ax.axvspan(grid.min(), DEP, color="0.85", alpha=0.6, lw=0, zorder=0)
ax.text(np.sqrt(grid.min() * DEP), len(yvals) - 0.4, "deployed range",
        ha="center", va="center", color="0.35", fontsize=8.0)
for yp, yv, c, idx in zip(ypos, yvals, colors, order):
    capped = not (np.isfinite(ycrit[idx]) and ycrit[idx] <= grid.max())
    ax.plot([grid.min() * 0.6, yv], [yp, yp], "-", color=c, lw=1.2, alpha=0.6, zorder=1)
    ax.plot([yv], [yp], ">" if capped else "o", color=c, ms=11, zorder=3)
    if capped:
        ax.text(yv * 1.05, yp, "(beyond grid)", va="center", fontsize=7, color=c)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8.2)
ax.set_xscale("log")
ax.set_xlabel(r"critical matching height $y_\mathrm{crit}^+$")
ax.set_title("(b) $y_\\mathrm{crit}^+$: an order-of-magnitude gap")
ax.set_xlim(grid.min() * 0.5, grid.max() * 3)
ax.invert_yaxis()

fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(OUT, f"fig_critical_matching_height.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
