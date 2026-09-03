#!/usr/bin/env python3
r"""
fig_error_mode_decomposition.py  --  L3 node_008 error-mode decomposition figure.

Two panels, from codes/results/heldout_error_mode_decomposition_l3.npz (the L3
re-analysis of the 20 held-out operating-map geometries):

  (a) ERROR-MODE SEPARATION BY THE CONDITIONING FLOOR.  med kappa (y, log) vs the
      variance-robust error relRMS (x).  The closure-conditioning floor -- NOT the
      R^2 magnitude -- separates the two error modes:
        * CLASS I  (red)  : eps_med<1 structural force-cancellation failures,
                            med kappa >= 0.11 (ill-conditioned);
        * CLASS II (gray) : eps_med>>1 raw-R^2<0 over-counts, med kappa <= 0.0035,
                            an order of magnitude better-conditioned.  ONE is a flat
                            wall (open diamond, of_flat_l22 -- vanishing-variance R^2
                            degeneracy); THREE are non-flat amplitude-driven misses
                            (gray triangles);
        * CLASS III (blue): eps_med>1, R^2>=0, tolerated and accurate.
      A shaded band marks the ~32x conditioning gap with no overlap.

  (b) THE AMPLITUDE ERROR MODE (well-conditioned, eps>>1, no cancellation).  relRMS
      vs a/delta on the well-conditioned subset (med kappa<0.05): the error rises
      with surface amplitude [Spearman(a/delta,relRMS)=+0.82, p<1e-3, n=13], ~2.6x
      from a/delta=0.10 to 0.40 at matched wide pitch -- a SECOND, milder, well-
      conditioned error mode that is NOT the cancellation mechanism.

Writes PNG+PDF to node_008/ and to manuscript/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_008")
MSFIG = os.path.join(PROJ, "manuscript", "figures")
os.makedirs(NODE, exist_ok=True)
os.makedirs(MSFIG, exist_ok=True)

d = np.load(os.path.join(RESULTS, "heldout_error_mode_decomposition_l3.npz"),
            allow_pickle=True)
tags = np.array([str(t) for t in d["tags"]])
a = d["a_over_delta"]; lam = d["lambda_over_delta"]
eps = d["eps_med"]; mk = d["med_kappa"]; rr = d["relRMS"]
classI = d["classI"].astype(bool)
IIa = d["IIa"].astype(bool)            # flat-wall vanishing-variance degeneracy
IIb = d["IIb"].astype(bool)            # non-flat amplitude-driven inaccuracy
classIII = d["classIII"].astype(bool)
gap = float(d["conditioning_gap"])
min_mk_I = float(d["min_medkappa_classI"]); max_mk_II = float(d["max_medkappa_classII"])
rho_a = float(d["spearman_a_relRMS_wellcond"])

C_FAIL = "#c0392b"     # Class I : structural cancellation failure (eps<1)
C_MISS = "#7f7f7f"     # Class II: non-cancellation R^2 over-count
C_TOL = "#2c6fbb"     # Class III: tolerated and accurate

fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.3))

# ---- panel (a): the conditioning floor separates the two error modes --------
A = ax[0]
# shaded conditioning gap, restricted to the OVERLAPPING error band where both
# Class-I failures and Class-II misses live: at comparable relRMS, med kappa splits
# the ill-conditioned cancellation failures (above) from the well-conditioned
# non-cancellation misses (below) -- a ~32x gap with no geometry in between.
rr_lo = float(min(rr[IIb].min(), rr[classI].min()) * 0.92)
rr_hi = float(max(rr[IIb].max(), rr[classI].max()) * 1.05)
A.fill_between([rr_lo, rr_hi], max_mk_II, min_mk_I, color="0.85", alpha=0.7, zorder=0)
A.text(rr_hi, np.sqrt(max_mk_II * min_mk_I),
       r"$\approx%.0f\times$ gap" % gap, ha="right", va="center",
       fontsize=8.5, color="0.35", style="italic")
A.scatter(rr[classIII], mk[classIII], s=58, marker="o", facecolor=C_TOL,
          edgecolor="k", linewidth=0.5, zorder=3, label="Class III tolerated+accurate")
A.scatter(rr[classI], mk[classI], s=80, marker="o", facecolor=C_FAIL,
          edgecolor="k", linewidth=0.6, zorder=4,
          label=r"Class I  $\varepsilon{<}1$ cancellation failure")
A.scatter(rr[IIb], mk[IIb], s=92, marker="^", facecolor=C_MISS,
          edgecolor="k", linewidth=0.7, zorder=4,
          label=r"Class IIb non-flat amplitude miss")
A.scatter(rr[IIa], mk[IIa], s=150, marker="D", facecolor="none",
          edgecolor="0.25", linewidth=1.5, zorder=5,
          label=r"Class IIa flat wall ($a/\delta{=}0.001$)")
A.set_yscale("log")
A.set_xlabel(r"rel. wall-stress error  ${\rm RMS}/\overline{|\tau_w|}$")
A.set_ylabel(r"median closure condition number  $\kappa$")
A.set_title("(a) the conditioning floor separates the modes\n"
            r"($R^2$ magnitude does not)", fontsize=9.6)
A.legend(loc="lower right", fontsize=7.0, frameon=True, framealpha=0.92)
A.grid(True, which="both", alpha=0.2)

# ---- panel (b): the amplitude error mode (well-conditioned subset) ----------
B = ax[1]
wc = mk < 0.05
# colour the well-conditioned points by class; size encodes pitch
for i in np.where(wc)[0]:
    col = C_MISS if IIb[i] else C_TOL
    mark = "^" if IIb[i] else "o"
    B.scatter(a[i], rr[i], s=60, marker=mark, facecolor=col, edgecolor="k",
              linewidth=0.5, zorder=3)
# trend guide: mean relRMS at each amplitude level in the well-conditioned subset
amps = sorted(set(np.round(a[wc], 2)))
means = [float(np.mean(rr[wc & (np.round(a, 2) == av)])) for av in amps]
B.plot(amps, means, "k--", lw=1.1, alpha=0.7, zorder=2,
       label="mean relRMS per $a/\\delta$")
B.set_xlabel(r"surface amplitude  $a/\delta$")
B.set_ylabel(r"rel. wall-stress error  ${\rm RMS}/\overline{|\tau_w|}$")
B.set_title("(b) the amplitude error mode (well-conditioned, $\\varepsilon{\\gg}1$)\n"
            r"Spearman$(a/\delta,{\rm relRMS})=%.2f$" % rho_a, fontsize=9.6)
B.grid(True, which="both", alpha=0.2)
handles_b = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TOL, markeredgecolor="k",
           markersize=8, label="tolerated+accurate"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_MISS, markeredgecolor="k",
           markersize=9, label="non-flat amplitude miss"),
    Line2D([0], [0], color="k", ls="--", lw=1.1, label=r"mean per $a/\delta$"),
]
B.legend(handles=handles_b, loc="upper left", fontsize=7.4, frameon=True, framealpha=0.92)

fig.tight_layout()
for outdir in (NODE, MSFIG):
    fig.savefig(os.path.join(outdir, "fig_error_mode_decomposition.png"), dpi=160)
    fig.savefig(os.path.join(outdir, "fig_error_mode_decomposition.pdf"))
print("wrote fig_error_mode_decomposition.{png,pdf} to node_008/ and manuscript/figures/")
