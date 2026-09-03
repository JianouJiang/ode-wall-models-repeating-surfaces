#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fig_fidelity_consistency_l3.py  --  L3 supplementary figure for the sharp-rib
fidelity-consistency thread.  Two panels, all numbers from
``codes/results/rib_fidelity_consistency_l3.npz`` (a-priori, Y_IDX=10):

  (a) The per-CELL resolved-fraction distribution on the converged d-type rib
      LES, plotted against the local total turbulent shear stress each cell
      carries.  The cells with f_res < 0.5 are the stress-STARVED sublayer floor
      (turbulent stress -> 0, so f_res is a 0/0 value); they carry ~0% of the
      near-wall turbulent stress.  The band-median (0.992) and the
      stress-WEIGHTED aggregate (0.993) coincide -> band-median is the correct
      representative statistic for eq:fres, not the cell-level min (0.246).

  (b) The RANS-certifies / WRLES-fails sign flip is robust to the matching
      height: across Y_IDX in {8..12} the RANS reference certifies the ODE
      (R^2 > 0, gray-box) while the wall-resolved LES exposes the failure
      (R^2 < 0, black-box) at EVERY height.

Colour scheme (project convention): black = black-box (ODE on the true WRLES
reference), bluish-gray = gray-box (RANS reference).

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_fidelity_consistency_l3.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
MSFIG = os.path.abspath(os.path.join(CODES, "..", "manuscript", "figures"))

BLACK = "#1a1a1a"        # black-box (ODE on the true WRLES reference)
GRAY = "#6b7a8f"         # gray-box (RANS reference)
ORANGE = "#e8820c"       # ground truth accent

d = np.load(os.path.join(RESULTS, "rib_fidelity_consistency_l3.npz"),
            allow_pickle=True)
dec = json.loads(str(d["f_res_cell_deciles_json"]))
bd = json.loads(str(d["per_cell_breakdown_json"]))
f_band = float(d["f_res_band_median"])
f_w = float(d["f_res_weighted"])
f_cells = np.asarray(d["f_cells"], float)
ttot = np.asarray(d["cell_ttot"], float)

y_idx = np.asarray(d["signflip_y_idx"], float)
r2_rans = np.asarray(d["signflip_r2_rans"], float)
r2_les = np.asarray(d["signflip_r2_les"], float)

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.9))

# ---- panel (a): per-cell f_res vs stress carried -------------------------
# normalise the total turbulent stress by its pooled median for a readable axis
tnorm = ttot / np.median(ttot)
lo = f_cells < 0.5
axA.scatter(tnorm[~lo], f_cells[~lo], s=11, c=GRAY, alpha=0.55,
            edgecolors="none", label=r"$f_{\rm res}\geq0.5$ (stress-bearing)")
axA.scatter(tnorm[lo], f_cells[lo], s=26, c=ORANGE, alpha=0.95,
            edgecolors=BLACK, linewidths=0.4,
            label=r"$f_{\rm res}<0.5$ (sublayer floor, %.1f%% of cells)"
            % (100 * bd["frac_cells_below_0p5"]))
axA.axhline(f_band, color=BLACK, lw=1.4, ls="-",
            label=r"band-median $=%.3f$ (the eq. statistic)" % f_band)
axA.axhline(f_w, color=ORANGE, lw=1.3, ls="--",
            label=r"stress-weighted $=%.3f$" % f_w)
axA.set_xscale("log")
axA.set_xlim(8e-5, 60)
axA.set_ylim(0.0, 1.04)
axA.set_xlabel(r"cell turbulent stress $\,/\,$ median")
axA.set_ylabel(r"resolved fraction $f_{\rm res}$ (per cell)")
axA.set_title(r"(a) per-cell $f_{\rm res}$: low values are stress-starved",
              fontsize=9.5)
axA.legend(loc="lower right", fontsize=6.6, framealpha=0.9)
axA.text(1.4e-4, 0.10,
         "sub-0.5 cells carry\n%.2f%% of the stress" % (100 * bd["stress_fraction_in_lo_cells"]),
         fontsize=7.0, color=BLACK,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRAY, lw=0.6))

# ---- panel (b): sign-flip robustness across matching height --------------
axB.axhspan(0, 1.1, color=GRAY, alpha=0.07)
axB.axhspan(-1.8, 0, color=BLACK, alpha=0.05)
axB.axhline(0, color="k", lw=0.8)
axB.plot(y_idx, r2_rans, "-s", color=GRAY, lw=1.8, ms=7, mec="k", mew=0.5,
         label=r"RANS reference (gray-box): certifies")
axB.plot(y_idx, r2_les, "-o", color=BLACK, lw=1.8, ms=7, mfc=BLACK,
         label=r"WRLES reference (black-box ODE): fails")
axB.set_xlabel(r"matching-cell index $y_{\rm idx}$")
axB.set_ylabel(r"$R^2(\tau_w)$ at $p/k=3$")
axB.set_xticks(y_idx)
axB.set_ylim(-1.8, 1.1)
axB.set_title(r"(b) sign flip robust at every matching height", fontsize=9.5)
axB.text(11.0, 0.88, "certifies\n($R^2>0$)", fontsize=7.2, color=GRAY, ha="center")
axB.text(8.4, -1.5, "fails\n($R^2<0$)", fontsize=7.2, color=BLACK, ha="center")
axB.legend(loc="center left", fontsize=7.0, framealpha=0.9)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(HERE, "fig_fidelity_consistency_l3." + ext)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("wrote", os.path.relpath(out, os.path.join(CODES, "..")))
# manuscript copy (supplementary)
os.makedirs(MSFIG, exist_ok=True)
for ext in ("png", "pdf"):
    out = os.path.join(MSFIG, "fig_fidelity_consistency_l3." + ext)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("wrote", os.path.relpath(out, os.path.join(CODES, "..")))
print("band-median=%.4f  stress-weighted=%.4f  cell-min=%.3f  p1=%.3f"
      % (f_band, f_w, float(d["f_res_cell_min"]), dec["p1"]))
