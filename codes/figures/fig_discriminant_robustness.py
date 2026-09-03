#!/usr/bin/env python3
r"""
fig_discriminant_robustness.py -- thrust #10, Level-2 headline experiments figure.

A genuine structural wall-model failure must be robust to BOTH a streamwise and a
wall-normal perturbation of the a-priori sampling; only the eps<1 force-cancellation
cases are. Three panels, every number from
  codes/results/discriminant_robustness_battery.npz   (this thrust's L2 experiment)

  (a) AXIS 1 -- streamwise decimation. R^2(tau_w) vs station spacing Dx_sta/delta,
      offset-ensemble band [min,max]. periodic hills + 3-D diffuser stay flat-
      catastrophic (survive); curved-BFS heals upward (sampling artefact).
  (b) AXIS 2 -- wall-normal matching height. R^2 vs y_m^+ on a COMMON physical
      height grid. hills + diffuser dive deeper as y_m grows (error ~ y_m, never
      heals); curved-BFS stays eps-safe; separation-bubble caseE climbs OUT of the
      catastrophic band -- its fixed-index y_m^+~1 sublayer failure was an artefact.
  (c) DISCRIMINANT MAP. eps_med (log) vs fixed-index R^2 for all 12 geometries;
      eps*=1 splits the deep-cancellation structural failures (robust on both
      axes) from every eps-safe geometry. The caseE arrow shows the named-falsifier
      candidate moving from "fail" (fixed index) to "pass" (deployment y_m^+).

Colour convention here is by FAILURE CLASS (stated in caption), kept distinct from
the project model-colour scheme (orange=truth, black=black-box, bluish-gray=gray-box,
green=Spalding) which does not apply to this geometry-class scatter:
  crimson  = structural force-cancellation failure (eps<1, robust both axes)
  steel    = streamwise-sampling artefact (curved-BFS, heals on axis 1)
  purple   = matching-height artefact (caseE, heals on axis 2)
  grey/green = eps-safe geometries the ODE handles.

No fabricated numbers; a-priori, read-only inputs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "fig_discriminant_robustness.png"))

C_STRUCT = "#b3122a"   # structural force-cancellation failure (crimson)
C_SAMP = "#2f6fb0"     # streamwise-sampling artefact (steel)
C_MATCH = "#7a3b9a"    # matching-height artefact (purple)
C_PASS = "#4a4a4a"     # eps-safe geometries handled by the ODE (dark grey)
C_OK = "#2a9d4a"       # success markers (green-ish)

d = np.load(os.path.join(RES, "discriminant_robustness_battery.npz"),
            allow_pickle=True)
R2S = float(d["r2_success"])          # 0.88
EPS_STAR = float(d["eps_star"])       # 1.0
RR_FAIL = float(d["relrms_fail"])     # 0.5

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

# ----------------------------------------------------------------------------
# Panel (a): AXIS 1 -- streamwise decimation, offset-ensemble R2 bands
# ----------------------------------------------------------------------------
a = ax[0]
dec_geoms = [
    ("periodic_hills",  "periodic hills ($\\varepsilon{=}0.08$)", C_STRUCT, "o", "H"),
    ("kth_3d_diffuser", "3-D diffuser ($\\varepsilon{=}0.21$)",   "#e0708a", "s", "L"),
    ("curved_bfs_LES",  "curved BFS ($\\varepsilon{=}3.8$)",      C_SAMP,   "^", "h"),
    ("sep_bubble_caseE","sep.\\ bubble E ($\\varepsilon{=}26$)",  C_MATCH,  "D", "L"),
]
for key, lab, col, mk, unit in dec_geoms:
    dx = d[f"{key}_dec_dx"]
    lo = d[f"{key}_dec_R2_min"]
    hi = d[f"{key}_dec_R2_max_arr"]
    me = d[f"{key}_dec_R2_mean"]
    order = np.argsort(dx)
    dx, lo, hi, me = dx[order], lo[order], hi[order], me[order]
    a.fill_between(dx, lo, hi, color=col, alpha=0.18, lw=0)
    a.plot(dx, me, mk + "-", color=col, ms=4, lw=1.3, label=lab, mfc=col,
           mec="white", mew=0.5)
a.axhline(R2S, color=C_OK, ls="--", lw=1.0)
a.axhline(0.0, color="0.4", ls=":", lw=0.8)
a.text(a.get_xlim()[1], R2S, " success $R^2{=}0.88$", color=C_OK, fontsize=7,
       va="bottom", ha="right")
a.axhspan(-16, 0, color="0.93", zorder=0)
a.set_ylim(-16, 2.5)
a.set_xscale("log")
a.set_xlabel(r"station spacing  $\Delta x_{\rm sta}/\delta$")
a.set_ylabel(r"$R^2(\tau_w)$  (offset-ensemble band)")
a.set_title("(a) axis 1: streamwise decimation", fontsize=9, loc="left")
a.legend(fontsize=6.4, loc="lower right", framealpha=0.9)
a.annotate("curved BFS upper edge\nheals toward success",
           xy=(0.45, 0.7), xytext=(0.12, 1.7), fontsize=6.2, color=C_SAMP,
           ha="center", arrowprops=dict(arrowstyle="->", color=C_SAMP, lw=0.9))
a.text(0.02, -14.2, "structural cases never\nreach success (survive)",
       fontsize=6.4, color=C_STRUCT, ha="left", va="bottom")

# ----------------------------------------------------------------------------
# Panel (b): AXIS 2 -- wall-normal matching height, R2 vs y_m^+
# ----------------------------------------------------------------------------
b = ax[1]
yg = d["yplus_grid"]
mh_geoms = [
    ("periodic_hills",  "periodic hills",   C_STRUCT, "o"),
    ("kth_3d_diffuser", "3-D diffuser",     "#e0708a", "s"),
    ("curved_bfs_LES",  "curved BFS",       C_SAMP,   "^"),
    ("sep_bubble_caseE","sep.\\ bubble E",  C_MATCH,  "D"),
]
# shade the catastrophic region (R2<0)
b.axhspan(-80, 0, color="0.88", zorder=0)
for key, lab, col, mk in mh_geoms:
    r2g = d[f"{key}_yplus_R2"]
    b.plot(yg, r2g, mk + "-", color=col, ms=4, lw=1.3, label=lab, mfc=col,
           mec="white", mew=0.5)
# caseE fixed-index sublayer point (the artefact) + arrow up into the safe band
cE_fix_yp = float(d["sep_bubble_caseE_yplus_m"])
cE_fix_R2 = float(d["sep_bubble_caseE_R2_full"])
b.plot([cE_fix_yp], [cE_fix_R2], "x", color=C_MATCH, ms=8, mew=1.8, zorder=6)
b.annotate("caseE fixed index\n$y_m^+{\\approx}1$ (sublayer)",
           xy=(cE_fix_yp, cE_fix_R2), xytext=(8, -1.4), fontsize=6.4,
           color=C_MATCH, ha="left",
           arrowprops=dict(arrowstyle="->", color=C_MATCH, lw=1.0))
b.axhline(R2S, color=C_OK, ls="--", lw=1.0)
b.axhline(0.0, color="0.4", ls=":", lw=0.8)
b.set_ylim(-3.0, 1.2)
b.set_xlabel(r"matching height  $y_m^+$")
b.set_ylabel(r"$R^2(\tau_w)$")
b.set_title("(b) axis 2: wall-normal matching height", fontsize=9, loc="left")
b.legend(fontsize=6.6, loc="lower left", framealpha=0.9)
b.text(60, -2.7, "catastrophic ($R^2{<}0$)", fontsize=6.4, color="0.45",
       ha="right", va="bottom")
# inset showing the hills full dive (R2 to -73)
bins = b.inset_axes([0.60, 0.60, 0.37, 0.36])
bins.plot(yg, d["periodic_hills_yplus_R2"], "o-", color=C_STRUCT, ms=2.5, lw=1.0)
bins.plot(yg, d["kth_3d_diffuser_yplus_R2"], "s-", color="#e0708a", ms=2.5, lw=1.0)
bins.set_title(r"full scale", fontsize=6)
bins.tick_params(labelsize=5.5)
bins.set_xlabel(r"$y_m^+$", fontsize=6)
bins.axhline(0, color="0.5", ls=":", lw=0.6)

# ----------------------------------------------------------------------------
# Panel (c): DISCRIMINANT MAP -- eps_med (log) vs fixed-index R2, all geometries
# ----------------------------------------------------------------------------
c = ax[2]
labels = [str(x) for x in d["geometry_labels"]]
struct_set = set(str(x) for x in d["structural_set"])
art_set = set(str(x) for x in d["artifact_set"])
for lab in labels:
    e = float(d[f"{lab}_eps_med"])
    r = float(d[f"{lab}_R2_full"])
    if not (np.isfinite(e) and np.isfinite(r)):
        continue
    if lab in struct_set:
        col, mk, z = C_STRUCT, "o", 5
    elif lab == "curved_bfs_LES":
        col, mk, z = C_SAMP, "^", 5
    elif lab == "sep_bubble_caseE":
        col, mk, z = C_MATCH, "D", 5
    else:
        col, mk, z = C_PASS, ".", 3
    rr = max(r, -3.2)   # clip for display
    c.scatter([e], [rr], c=col, marker=mk, s=42 if mk != "." else 30,
              edgecolors="white", linewidths=0.5, zorder=z)
# caseE healed location (deployment y+ best R2) + arrow from fixed-index point
cE_e_fix = float(d["sep_bubble_caseE_eps_med"])
cE_heal_R2 = float(np.nanmax(d["sep_bubble_caseE_yplus_R2"]))
cE_heal_eps = float(np.nanmedian(d["sep_bubble_caseE_yplus_eps_med"]))
c.scatter([cE_heal_eps], [cE_heal_R2], facecolors="none", edgecolors=C_MATCH,
          marker="D", s=60, linewidths=1.4, zorder=6)
c.annotate("", xy=(cE_heal_eps, cE_heal_R2), xytext=(cE_e_fix, max(cE_fix_R2, -3.2)),
           arrowprops=dict(arrowstyle="->", color=C_MATCH, lw=1.2,
                           connectionstyle="arc3,rad=-0.3"), zorder=6)
c.text(cE_heal_eps, cE_heal_R2 + 0.15, "caseE heals at\ndeployment $y_m^+$",
       fontsize=6.2, color=C_MATCH, ha="center", va="bottom")
c.axvline(EPS_STAR, color="0.3", ls="--", lw=1.0)
c.axhline(R2S, color=C_OK, ls="--", lw=1.0)
c.axhline(0.0, color="0.4", ls=":", lw=0.8)
c.text(EPS_STAR * 1.15, -3.0, r"$\varepsilon^\ast{=}1$", fontsize=7, color="0.3")
c.text(0.012, R2S, " success", fontsize=6.5, color=C_OK, va="bottom")
c.set_xscale("log")
c.set_ylim(-3.4, 1.4)
c.set_xlabel(r"cancellation parameter  $\varepsilon_{\rm med}$")
c.set_ylabel(r"fixed-index $R^2(\tau_w)$")
c.set_title("(c) discriminant map", fontsize=9, loc="left")
# legend proxies
from matplotlib.lines import Line2D
proxies = [
    Line2D([], [], marker="o", color="none", mfc=C_STRUCT, mec="white",
           label="structural ($\\varepsilon{<}1$, both axes)", ms=7),
    Line2D([], [], marker="^", color="none", mfc=C_SAMP, mec="white",
           label="sampling artefact (curved BFS)", ms=7),
    Line2D([], [], marker="D", color="none", mfc=C_MATCH, mec="white",
           label="matching-height artefact (caseE)", ms=7),
    Line2D([], [], marker=".", color="none", mfc=C_PASS, mec="white",
           label="$\\varepsilon$-safe, ODE handles", ms=9),
]
c.legend(handles=proxies, fontsize=6.0, loc="lower right", framealpha=0.9)
c.text(0.012, -3.25, "deep cancellation", fontsize=6.2, color=C_STRUCT)

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
pdf = os.path.splitext(OUT)[0] + ".pdf"
fig.savefig(pdf, bbox_inches="tight")
print("wrote", OUT, "and", pdf)
