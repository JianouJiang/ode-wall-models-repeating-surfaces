#!/usr/bin/env python3
r"""
fig_heldout_discriminant.py  --  L2 node_007 held-out validation figure.

Two panels, from codes/results/rib_discriminant_heldout_l2.npz (20 held-out
operating-map geometries, NEVER used to set the eps<1 threshold):

  (a) the OOS closure-conditioning law: med kappa vs eps_med (log-log).  The
      theory-frozen boundary eps=1 (vertical line) separates the 4 ill-
      conditioned eps<1 geometries from the 16 well-conditioned eps>1 ones;
      Spearman(eps, med kappa) = -0.84 (kappa~1/eps confirmed OOS).
  (b) the OOS dose-response: relRMS (variance-robust wall-stress error) vs
      eps_med.  Deeper cancellation -> larger error (Spearman(1/eps,relRMS)=+0.61).
      The raw-R^2<0 "misses" (open squares) are all at high eps with tiny
      conditioning -- the documented low-variance R^2 degeneracy, NOT failure.

Writes PNG+PDF to node_007/ and to manuscript/figures/ (B-L2-4 promotion).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_007")
MSFIG = os.path.join(PROJ, "manuscript", "figures")
os.makedirs(NODE, exist_ok=True)
os.makedirs(MSFIG, exist_ok=True)

d = np.load(os.path.join(RESULTS, "rib_discriminant_heldout_l2.npz"), allow_pickle=True)
tags = [str(t) for t in d["tags"]]
eps = d["eps_med"]; medk = d["med_kappa"]; relRMS = d["relRMS"]
ad = d["a_over_delta"]; r2b = d["r2_best"]
pred_fail = d["pred_fail"].astype(bool)
r2_fail = d["r2_fail"].astype(bool)
miss = pred_fail != r2_fail          # raw-R^2 disagreements (all high-eps over-counts)
rho_k = float(d["spearman_eps_medkappa"]); rho_e = float(d["spearman_inv_eps_relRMS"])

# marker by amplitude band, colour by the theory-frozen verdict
amp_marker = {0.0: "s", 0.05: "v", 0.10: "o", 0.15: "^", 0.20: "D", 0.40: "P"}
C_FAIL = "#c0392b"      # eps<1 : predicted structural failure (deep cancellation)
C_TOL = "#2c6fbb"      # eps>1 : predicted tolerated

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))

# ---- panel (a): conditioning floor vs eps (the kappa~1/eps mechanism, OOS) ---
a = ax[0]
for i in range(len(tags)):
    col = C_FAIL if pred_fail[i] else C_TOL
    mk = amp_marker.get(round(float(ad[i]), 2), "o")
    a.scatter(eps[i], medk[i], s=64, marker=mk, facecolor=col,
              edgecolor="k", linewidth=0.6, zorder=3)
# 1/eps guide (kappa ~ c/eps), anchored at the failure cloud
cguide = np.median(medk[pred_fail] * eps[pred_fail])
xx = np.logspace(np.log10(eps.min()), np.log10(eps.max()), 50)
a.plot(xx, cguide / xx, "k--", lw=1.0, alpha=0.6, zorder=1,
       label=r"$\kappa\sim 1/\varepsilon$ guide")
a.axvline(1.0, color="0.4", lw=1.2, ls=":", zorder=1)
a.text(1.0, medk.max() * 1.4, r"$\varepsilon=1$" + "\n(theory)", ha="center",
       va="top", fontsize=8.5, color="0.3")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$\varepsilon_{\rm med}$  (held-out, $a/\delta\times\lambda/\delta$ family)")
a.set_ylabel(r"median closure condition number  $\kappa$")
a.set_title(r"(a) conditioning floor, out-of-sample"
            "\n" + r"Spearman$(\varepsilon,\kappa)=%.2f$" % rho_k, fontsize=10)
a.legend(loc="lower left", fontsize=8, frameon=False)
a.grid(True, which="both", alpha=0.2)

# ---- panel (b): dose-response on the robust error metric --------------------
b = ax[1]
for i in range(len(tags)):
    col = C_FAIL if pred_fail[i] else C_TOL
    mk = amp_marker.get(round(float(ad[i]), 2), "o")
    if miss[i]:
        # raw-R^2<0 over-count: open square overlay (high-eps degeneracy)
        b.scatter(eps[i], relRMS[i], s=140, marker="s", facecolor="none",
                  edgecolor="0.45", linewidth=1.3, zorder=2)
    b.scatter(eps[i], relRMS[i], s=64, marker=mk, facecolor=col,
              edgecolor="k", linewidth=0.6, zorder=3)
b.axvline(1.0, color="0.4", lw=1.2, ls=":", zorder=1)
b.axhline(1.0, color="0.7", lw=0.8, ls="-", zorder=0)
b.set_xscale("log")
b.set_xlabel(r"$\varepsilon_{\rm med}$  (held-out)")
b.set_ylabel(r"rel. wall-stress error  ${\rm RMS}/\overline{|\tau_w|}$")
b.set_title(r"(b) dose-response, out-of-sample"
            "\n" + r"Spearman$(1/\varepsilon,{\rm relRMS})=%.2f$" % rho_e, fontsize=10)
b.grid(True, which="both", alpha=0.2)

# shared legend for the colour code + the open-square overlay
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAIL,
           markeredgecolor="k", markersize=9, label=r"$\varepsilon<1$ : predicted failure"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TOL,
           markeredgecolor="k", markersize=9, label=r"$\varepsilon>1$ : predicted tolerated"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
           markeredgecolor="0.45", markersize=11, markeredgewidth=1.3,
           label=r"raw $R^2<0$ over-count (low-var. degeneracy)"),
]
b.legend(handles=handles, loc="upper right", fontsize=7.6, frameon=True, framealpha=0.9)

fig.tight_layout()
for outdir in (NODE, MSFIG):
    fig.savefig(os.path.join(outdir, "fig_heldout_discriminant.png"), dpi=160)
    fig.savefig(os.path.join(outdir, "fig_heldout_discriminant.pdf"))
print("wrote fig_heldout_discriminant.{png,pdf} to node_007/ and manuscript/figures/")
