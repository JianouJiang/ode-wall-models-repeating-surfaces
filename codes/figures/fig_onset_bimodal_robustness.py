#!/usr/bin/env python3
r"""
fig_onset_bimodal_robustness.py  (L2 node_006 att2)
===================================================
Two-panel diagnostic for the L2 robustness result.

  (a) Threshold robustness of the crossing-direction branch classification.
      Each amplitude ladder's direction (F->T / T->F / mixed / allT) is shown
      against the catastrophe screen FAIL_RELRMS.  The shaded band marks the
      window where the BIMODAL classification (>=1 F->T AND >=1 T->F) holds:
      FAIL_RELRMS <= 0.50.
  (b) The closure-independent conditioning floor kappa cleanly separates the
      two error modes (gap 31.9x, NO overlap, AUC=1.000, exact p_two=0.057),
      whereas the crossing direction does not.  op_a10_l22 -- the case that
      destabilises the geometric label -- is resolved by kappa to Mode II.

Reads codes/results/onset_bimodal_robustness_l2.npz (CFD-free).
Writes codes/figures/fig_onset_bimodal_robustness.{png,pdf} and a node copy.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NODE = os.path.join(os.path.dirname(CODES), "development", "nodes", "node_006")

d = np.load(os.path.join(RESULTS, "onset_bimodal_robustness_l2.npz"),
            allow_pickle=True)
thr = d["thresholds"].astype(float)
amps = d["amps"].astype(float)
dirs = d["branch_dirs"]            # (n_thr, n_amp) strings
bim_ok = d["bimodal_ok"].astype(bool)
bim_max = float(d["bimodal_max_thr"])
mkI = d["kappa_classI"].astype(float)
mkII = d["kappa_classIIb"].astype(float)
gap = float(d["kappa_gap"])
auc = float(d["kappa_auc"])
p_two = float(d["kappa_p_two"])
k22 = float(d["op_a10_l22_kappa"])
KILL = float(d["KAPPA_ILL"])
KWELL = float(d["KAPPA_WELL"])
I_tags = [str(t) for t in d["classI_tags"]]
II_tags = [str(t) for t in d["classIIb_tags"]]

# colour map for branch direction
DCOL = {"F->T": "#1f77b4", "T->F": "#d62728", "mixed": "#999999",
        "allT": "#dddddd", "allF": "#333333"}
DLAB = {"F->T": "F$\\to$T (Mode I, short-pitch fail)",
        "T->F": "T$\\to$F (cand. Mode II, long-pitch fail)",
        "mixed": "mixed", "allT": "all tolerated", "allF": "all fail"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.5))

# ---- panel (a): threshold robustness grid -----------------------------------
nT, nA = dirs.shape
for it in range(nT):
    for ia in range(nA):
        dd = str(dirs[it, ia])
        axA.add_patch(plt.Rectangle((ia - 0.45, it - 0.45), 0.9, 0.9,
                      facecolor=DCOL.get(dd, "#cccccc"),
                      edgecolor="white", linewidth=1.5))
# shade the bimodal window on the threshold axis
ymax_idx = max(i for i in range(nT) if bim_ok[i])
axA.axhspan(-0.5, ymax_idx + 0.5, color="gold", alpha=0.18, zorder=0)
axA.text(nA - 0.5, ymax_idx + 0.5,
         "  bimodal window\n  (FAIL$_{relRMS}\\leq%.2f$)" % bim_max,
         va="bottom", ha="right", fontsize=9, color="#8a6d00")
axA.set_xticks(range(nA))
axA.set_xticklabels(["%.2f" % a for a in amps])
axA.set_yticks(range(nT))
axA.set_yticklabels(["%.2f" % t for t in thr])
axA.set_xlabel("amplitude  $a/\\delta$")
axA.set_ylabel("catastrophe screen  FAIL$_{relRMS}$")
axA.set_xlim(-0.6, nA - 0.4)
axA.set_ylim(-0.6, nT - 0.4)
axA.set_title("(a) crossing-direction label is threshold-fragile", fontsize=10.5)
handles = [Patch(facecolor=DCOL[k], label=DLAB[k])
           for k in ["F->T", "T->F", "mixed", "allT"]]
axA.legend(handles=handles, fontsize=7.6, loc="upper left",
           framealpha=0.95, handlelength=1.2)

# ---- panel (b): kappa separation --------------------------------------------
rng = np.random.default_rng(0)
xI = 1 + 0.06 * rng.standard_normal(len(mkI))
xII = 2 + 0.06 * rng.standard_normal(len(mkII))
axB.scatter(xI, mkI, s=70, color="#1f77b4", edgecolor="k", zorder=3,
            label="Mode I (cancellation, ill-cond.)")
axB.scatter(xII, mkII, s=70, color="#d62728", edgecolor="k", zorder=3,
            label="Mode II (R$^2$-miss, well-cond.)")
# the gap band
axB.axhspan(mkII.max(), mkI.min(), color="0.85", alpha=0.7, zorder=0)
axB.text(2.48, np.sqrt(mkII.max() * mkI.min()),
         "no-overlap\ngap %.0f$\\times$" % gap, ha="right", va="center",
         fontsize=9)
axB.axhline(KILL, color="#1f77b4", ls="--", lw=1, alpha=0.6)
axB.axhline(KWELL, color="#d62728", ls="--", lw=1, alpha=0.6)
# highlight op_a10_l22
i22 = II_tags.index("op_a10_l22")
axB.annotate("op_a10_l22\n($R^2{=}{-}8$, but $\\kappa{=}%.4f$\n$\\Rightarrow$ Mode II)"
             % k22, xy=(xII[i22], mkII[i22]), xytext=(1.35, 0.006),
             fontsize=8, ha="left",
             arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
axB.set_yscale("log")
axB.set_xticks([1, 2])
axB.set_xticklabels(["Mode I\n($n{=}%d$)" % len(mkI),
                     "Mode II\n($n{=}%d$, 2 ampl.)" % len(mkII)])
axB.set_xlim(0.5, 2.6)
axB.set_ylabel("conditioning floor  median $\\kappa$")
axB.set_title("(b) $\\kappa$ is the robust mode discriminant", fontsize=10.5)
axB.text(0.55, 0.04,
         "AUC$=%.3f$, exact $p_{\\rm two}=%.3f$" % (auc, p_two),
         transform=axB.transAxes, fontsize=8.5, va="top")
axB.legend(fontsize=7.8, loc="lower left")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "fig_onset_bimodal_robustness.%s" % ext),
                dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(NODE, "fig_onset_bimodal_robustness.png"),
            dpi=150, bbox_inches="tight")
print("wrote fig_onset_bimodal_robustness.{png,pdf} + node copy")
