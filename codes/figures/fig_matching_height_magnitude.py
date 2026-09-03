#!/usr/bin/env python3
r"""
fig_matching_height_magnitude.py -- thrust #10, Level-3 headline results figure.

The wall-normal matching-height error is GOVERNED BY eps, not by a binary
threshold.  Every number from
  codes/results/matching_height_magnitude.npz   (this thrust's L3 experiment)

  (a) eps-COLLAPSE.  Pool every (geometry x matching-height) aggregate over the
      WMLES deployment window y_m^+ in [20,60] (108 points, 12 geometries x 9
      heights).  relRMS vs height-resolved eps_med collapses onto a single power
      law relRMS = C eps^p across four decades of eps -- one parameter orders the
      whole cloud, across geometries AND heights.
  (b) WORST-CASE eps-GRADING + THE EMPTY GAP.  Per-geometry worst-case error
      relRMS_wc = max over the window vs the intrinsic fixed-index eps_med.  The
      two eps<1 structural failures sit at O(4-8); every eps>1 geometry caps at
      the generic log-mismatch floor (<=0.7).  The shaded band (0.67, 3.60) is
      EMPTY -- the >5x gap between the O(1/eps) cancellation catastrophe and the
      bounded equilibrium-model floor.  The eps>1 'pass' cases that dip below the
      nominal threshold (open markers) reach only the floor, ~13x below hills.

Colour by FAILURE CLASS (kept distinct from the model-colour scheme
orange/black/bluish-gray/green, which labels tau_w model curves, not geometries):
  crimson = structural force-cancellation (eps<1)
  steel   = eps-safe geometry the ODE handles (eps>1)
A-priori, read-only inputs; nothing fabricated.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "fig_matching_height_magnitude.png"))

C_STRUCT = "#b3122a"   # structural force-cancellation failure (crimson)
C_SAFE = "#2f6fb0"     # eps-safe geometry handled by the ODE (steel)
C_FIT = "#222222"

d = np.load(os.path.join(RES, "matching_height_magnitude.npz"), allow_pickle=True)
EPS_STAR = float(d["eps_star"])
RR_FAIL = float(d["relrms_fail"])
labels = [str(x) for x in d["geometry_labels"]]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))

# ---------------------------------------------------------------------------
# Panel (a): pooled eps-collapse
# ---------------------------------------------------------------------------
a = ax[0]
pe = d["pooled_eps"]; pr = d["pooled_relRMS"]
# colour each pooled point by the INTRINSIC class of its geometry
eps_fixed = {l: float(d[f"{l}_eps_fixed"]) for l in labels}
pt_class = []
# rebuild geometry tag per pooled point by matching eps_h arrays
for l in labels:
    eh = d[f"{l}_eps_h"]; rh = d[f"{l}_relRMS_h"]; nh = d[f"{l}_n_h"]
    col = C_STRUCT if eps_fixed[l] < EPS_STAR else C_SAFE
    m = np.isfinite(rh) & np.isfinite(eh) & (nh >= 5)
    a.scatter(eh[m], rh[m], s=26, c=col, alpha=0.75,
              edgecolors="white", linewidths=0.4, zorder=3)
# power-law fit line
C = float(d["powerlaw_C"]); p = float(d["powerlaw_slope"])
xx = np.logspace(np.log10(pe.min()), np.log10(pe.max()), 100)
a.plot(xx, C * xx ** p, "-", color=C_FIT, lw=1.8, zorder=4,
       label=(r"$\mathrm{relRMS}=%.2f\,\widetilde{\varepsilon}^{\,%.2f}$"
              "\n" r"$(\rho=%.2f,\ R^2_{\log}=%.2f)$"
              % (C, p, float(d["collapse_spearman"]), float(d["powerlaw_loglog_R2"]))))
a.axhline(RR_FAIL, color="grey", ls=":", lw=1.0)
a.axvline(EPS_STAR, color="grey", ls="--", lw=1.0)
a.text(EPS_STAR * 1.15, 0.013, r"$\widetilde{\varepsilon}=1$", color="grey",
       fontsize=8, rotation=90, va="bottom")
a.text(pe.min() * 1.1, RR_FAIL * 1.1, r"relRMS$=0.5$ (failure)", color="grey", fontsize=7.5)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"height-resolved median cancellation parameter $\widetilde{\varepsilon}(y_m^+)$")
a.set_ylabel(r"relative RMS error in $\tau_w$")
a.set_title(r"(a) matching-height error collapses on $\widetilde{\varepsilon}$"
            "\n" r"$108$ points $=12$ geom $\times\,9$ heights, $y_m^+\!\in[20,60]$",
            fontsize=8.5)
a.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.9)
# class proxies
a.scatter([], [], s=26, c=C_STRUCT, label="struct")
a.grid(True, which="both", alpha=0.18)

# ---------------------------------------------------------------------------
# Panel (b): worst-case grading + the empty gap
# ---------------------------------------------------------------------------
b = ax[1]
gap_lo = float(d["gap_lo"]); gap_hi = float(d["gap_hi"])
b.axhspan(gap_lo, gap_hi, color="0.85", alpha=0.7, zorder=0)
b.text(2.0e2, np.sqrt(gap_lo * gap_hi),
       "EMPTY GAP\n%.1f$\\times$" % float(d["gap_ratio"]),
       ha="center", va="center", fontsize=8, color="0.35", style="italic")
for l in labels:
    ef = eps_fixed[l]; wc = float(d[f"{l}_relRMS_wc"])
    struct = ef < EPS_STAR
    col = C_STRUCT if struct else C_SAFE
    # open marker if this eps>1 'pass' geometry crosses the nominal threshold in-window
    rh = d[f"{l}_relRMS_h"]; r2h = d[f"{l}_R2_h"]
    crossed = (np.nanmin(r2h) < 0) or (np.nanmax(rh) > RR_FAIL)
    face = col if (struct or not crossed) else "white"
    b.scatter(ef, wc, s=80, marker="o", facecolors=face, edgecolors=col,
              linewidths=1.6, zorder=3)
b.axvline(EPS_STAR, color="grey", ls="--", lw=1.0)
b.axhline(RR_FAIL, color="grey", ls=":", lw=1.0)
b.text(EPS_STAR * 1.15, 0.13, r"$\widetilde{\varepsilon}=1$", color="grey",
       fontsize=8, rotation=90, va="bottom")
# annotate key geometries
ann = {"periodic_hills": "periodic hills", "kth_3d_diffuser": "KTH diffuser",
       "conv_div_channel": "conv-div", "curved_bfs_LES": "curved BFS",
       "gaussian_bump_Re1M": "Gauss bump 1M", "sep_bubble_caseA": "sep-bubble A"}
for l, txt in ann.items():
    ef = eps_fixed[l]; wc = float(d[f"{l}_relRMS_wc"])
    dy = 1.18 if l in ("periodic_hills", "kth_3d_diffuser") else 0.84
    b.annotate(txt, (ef, wc), (ef, wc * dy), fontsize=7.3, ha="center",
               color="0.2")
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel(r"intrinsic (fixed-index) $\widetilde{\varepsilon}_{\mathrm{med}}$")
b.set_ylabel(r"worst-case in-window relRMS$_{\rm wc}$")
b.set_title(r"(b) worst-case deployment error is $\widetilde{\varepsilon}$-graded"
            "\n" r"$\rho=%.2f$; open $=$ $\widetilde{\varepsilon}{>}1$ pass crossing threshold"
            % float(d["wc_spearman"]), fontsize=8.5)
b.grid(True, which="both", alpha=0.18)

# shared legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_STRUCT,
              markersize=8, label=r"structural ($\widetilde{\varepsilon}{<}1$)"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SAFE,
              markersize=8, label=r"$\widetilde{\varepsilon}$-safe ($\widetilde{\varepsilon}{>}1$)"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
              markeredgecolor=C_SAFE, markersize=8, label="threshold-crosser")]
b.legend(handles=leg, loc="lower left", fontsize=7.4, frameon=True, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
