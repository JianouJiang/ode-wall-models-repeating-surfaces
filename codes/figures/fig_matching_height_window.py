#!/usr/bin/env python3
r"""
fig_matching_height_window.py  --  Thrust #17, Level-3 (results & analysis)

THE DEPLOYABLE MATCHING-HEIGHT WINDOW (one panel, the practical headline).

Per geometry, the usable WMLES matching-height window is [y_LLM, y_crit]:
  * lower bound y_LLM = Kawai & Larsson (2012) log-layer-mismatch guidance
    (match in the inertial log layer, y_m^+ ~ a few tens) -- a literature range
    (gray band), NOT computed here;
  * upper bound y_crit = force-cancellation cliff, COMPUTED per geometry from the
    reference profiles (critical_matching_height_map.npz) -- the marker.

The window is OPEN (green) where y_crit > y_LLM (controls: classical guidance
safe), and CLOSED (red, hatched) where y_crit < y_LLM (O(delta)-pitch hills:
following "match deeper" drives you into the catastrophe -- the inverted rule).
Measured coupled operating cells overlaid as points (real bottomWall yPlus).

Source: codes/results/matching_height_window.npz  (a-priori; honest synthesis).
Colors: project convention (orange truth not used here; red=closed/cliff,
green=open/safe, gray=literature lower bound).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
OUT = os.path.join(ROOT, "development", "nodes", "node_003")
os.makedirs(OUT, exist_ok=True)

RED = "#D55E00"
GREEN = "#2C9D3A"
GRAYB = "#56708A"
ORANGE = "#E69F00"

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11.5, "axes.titlesize": 12,
    "legend.fontsize": 9, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.linewidth": 0.9,
})

d = np.load(os.path.join(RES, "matching_height_window.npz"), allow_pickle=True)
keys = [str(k) for k in d["keys"]]
pitch_Od = d["pitch_O_delta"].astype(bool)
repeating = d["repeating"].astype(bool)
ycrit = d["ycrit"].astype(float)
y_llm_min = float(d["y_llm_min"]); y_llm_max = float(d["y_llm_max"])
operating = d["operating"][0]

# pretty labels
PRETTY = {
    "krank_pehill_Re10595": "periodic hill\nKrank Re$_H$=10595",
    "periodic_hills_1p0": "periodic hill\n$h/L_x$=1 (resolved)",
    "conv_div_channel": "conv-div\nchannel",
    "bfs_Re13700": "backward-facing\nstep",
    "curved_bfs_LES": "curved\nBFS",
    "nasa_hump": "NASA\nwall hump",
}
# order: O(delta) hills first, then controls
order = [i for i in range(len(keys)) if (pitch_Od[i] and repeating[i])] + \
        [i for i in range(len(keys)) if not (pitch_Od[i] and repeating[i])]

fig, ax = plt.subplots(figsize=(9.2, 5.2))
YTOP = 600.0   # cap for "inf" y_crit (beyond resolved grid)

# classical log-layer-matching guidance band (lower bound)
ax.axhspan(y_llm_min, y_llm_max, color=GRAYB, alpha=0.18, lw=0, zorder=0)
ax.text(len(order) - 0.4, np.sqrt(y_llm_min * y_llm_max),
        "Kawai & Larsson (2012)\nlog-layer matching\n($y_m^+\\!\\sim$ few tens)",
        color=GRAYB, fontsize=8.5, ha="right", va="center")

xpos = np.arange(len(order))
for xi, i in enumerate(order):
    yc = ycrit[i]
    yc_plot = yc if np.isfinite(yc) else YTOP
    closed = (np.isfinite(yc) and yc < y_llm_min)
    col = RED if closed else GREEN
    # the window [y_llm_min, y_crit]
    lo, hi = y_llm_min, yc_plot
    if hi > lo:   # OPEN window: green span from y_LLM up to y_crit
        ax.add_patch(plt.Rectangle((xi - 0.28, lo), 0.56, hi - lo,
                                   facecolor=GREEN, alpha=0.22, edgecolor=GREEN, lw=1.2))
    else:         # CLOSED window: red hatched span from y_crit up to y_LLM
        ax.add_patch(plt.Rectangle((xi - 0.28, yc_plot), 0.56, y_llm_min - yc_plot,
                                   facecolor=RED, alpha=0.20, edgecolor=RED, lw=1.2,
                                   hatch="////"))
    # y_crit marker
    if np.isfinite(yc):
        ax.plot(xi, yc, "v", color=col, ms=11, mec="black", mew=0.7, zorder=5)
        ax.text(xi, yc * 0.78, rf"$y_{{\rm crit}}^+$={yc:.0f}", ha="center", va="top",
                fontsize=8.5, color=col)
    else:
        ax.annotate("", xy=(xi, YTOP * 1.18), xytext=(xi, YTOP * 0.85),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.text(xi, YTOP * 0.80, r"$y_{\rm crit}^+\!>$grid", ha="center", va="top",
                fontsize=8, color=col)

# Measured coupled operating cells belong to the Xiao alpha=1.0 hill (a separate
# O(delta) member, not one of the 6 mapped columns); shown in the band-
# confirmation twin figure, not here, to avoid conflating two distinct hills.

ax.set_yscale("log")
ax.set_ylim(0.5, YTOP * 2.4)
ax.set_xticks(xpos)
ax.set_xticklabels([PRETTY.get(keys[i], keys[i]) for i in order], fontsize=8.5)
ax.set_ylabel(r"matching height $y_m^+$")
ax.set_title("Deployable matching-height window  $[y_{\\rm LLM},\\,y_{\\rm crit}]$:\n"
             "closed (red) on $O(\\delta)$-pitch repeating hills, wide (green) on controls")

# legend proxies
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=RED, alpha=0.20, edgecolor=RED, hatch="////",
          label="window CLOSED (no safe match)"),
    Patch(facecolor=GREEN, alpha=0.22, edgecolor=GREEN, label="window OPEN (classical guidance safe)"),
    Patch(facecolor=GRAYB, alpha=0.18, label="log-layer matching band (lower bound)"),
    plt.Line2D([], [], marker="v", color="black", ls="none", ms=10, label=r"$y_{\rm crit}^+$ (cancellation cliff, computed)"),
]
ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=8.3)

# divider between O(delta) hills and controls
n_od = sum(1 for i in order if (pitch_Od[i] and repeating[i]))
ax.axvline(n_od - 0.5, color="0.5", lw=0.8, ls=":")
ax.text(n_od / 2 - 0.5, 0.66, "$O(\\delta)$-pitch repeating", ha="center",
        fontsize=9.5, color=RED, fontweight="bold")
ax.text(n_od + (len(order) - n_od) / 2 - 0.5, YTOP * 1.9, "wide-pitch / single-feature controls",
        ha="center", fontsize=9.5, color=GREEN, fontweight="bold")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_matching_height_window.{ext}"), bbox_inches="tight")
print("wrote fig_matching_height_window.{png,pdf} to", OUT)
