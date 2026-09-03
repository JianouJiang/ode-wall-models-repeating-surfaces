#!/usr/bin/env python3
r"""
fig_conditioning_discriminant.py  --  node_006 (L1 attempt 2) figure.

Two panels for the closure-conditioning-floor methodology that replaces the
fragile two-factor phi_span severity of node_001 (attempt 1):

  (a) THE ORACLE-AMPLIFICATION LAW A_E(eps): the factor by which the EXACT
      resolved Reynolds stress (maximal information) amplifies the wall-stress
      error relative to the eddy closures, versus the cancellation depth eps.
      A_E grows as eps->0; the sharp rib (A_E=3.9) lands on the smooth-hill law.

  (b) THE ISO-DEPTH CONDITIONING FLIP: at matched median depth (eps_med ~ 0.52)
      the d-type rib (fails) carries an ill-conditioned 90th-percentile tail
      8-24x the k-type's (tolerated) across all four eddy closures -- the robust
      replacement for the convex-hull phi_span.

Reads codes/results/rib_conditioning_discriminant_l1.npz (produced by
codes/analysis/rib_conditioning_discriminant_l1.py).  No fabrication.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
NODE = os.path.join(os.path.dirname(ROOT), "development", "nodes", "node_006")
MSFIG = os.path.join(os.path.dirname(ROOT), "manuscript", "figures")
os.makedirs(MSFIG, exist_ok=True)
os.makedirs(NODE, exist_ok=True)

d = np.load(os.path.join(RESULTS, "rib_conditioning_discriminant_l1.npz"),
            allow_pickle=True)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3))

# ---------- panel (a): oracle-amplification law A_E(eps) ---------------------
eps = d["law_eps"]; A_E = d["law_A_E"]; tags = d["law_tags"]; roles = d["law_roles"]
n = float(d["power_n"]); c = float(d["power_c"]); rho = float(d["spearman_AE_inv_eps"])
col = {"failure": "#d1495b", "control": "#2e86ab"}
for e, a, tg, ro in zip(eps, A_E, tags, roles):
    axA.scatter(e, a, s=95, c=col[str(ro)], edgecolor="k", zorder=3,
                label=None)
    axA.annotate(str(tg).replace("_", " "), (e, a), fontsize=7,
                 xytext=(6, 4), textcoords="offset points")
xx = np.logspace(np.log10(eps.min() * 0.7), np.log10(eps.max() * 1.4), 100)
axA.plot(xx, c * xx ** (-n), "k--", lw=1.2, zorder=2,
         label=r"$A_E\sim%.1f\,\varepsilon^{-%.2f}$ ($\rho=%.2f$)" % (c, n, rho))
axA.axhline(1.0, color="grey", ls=":", lw=1.0)
axA.text(eps.max(), 1.06, r"$A_E=1$ (oracle neutral)", fontsize=7,
         ha="right", va="bottom", color="grey")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel(r"median cancellation depth  $\varepsilon_{\rm med}$")
axA.set_ylabel(r"oracle amplification  $A_E=\kappa_E/\langle\kappa_{\rm eddy}\rangle$")
axA.set_title("(a) exact stress is the worst closure where cancellation is deep")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", ls="", mfc=col["failure"], mec="k",
              label="ODE fails ($\\varepsilon<1$)"),
       Line2D([0], [0], marker="o", ls="", mfc=col["control"], mec="k",
              label="ODE tolerated ($\\varepsilon>1$)"),
       Line2D([0], [0], ls="--", color="k", label="power-law fit")]
axA.legend(handles=leg, fontsize=7.5, loc="lower left")
axA.grid(alpha=0.25, which="both")

# ---------- panel (b): iso-depth conditioning-tail flip ----------------------
labels = ["ML\nvan Driest", "alg\nCebeci", "alg\nSA", "alg\nReichardt"]
p90_d = d["dtype_p90_kappa"][:4]      # eddy closures A-D
p90_k = d["ktype_p90_kappa"]
x = np.arange(4); w = 0.38
axB.bar(x - w / 2, p90_d, w, color="#d1495b", edgecolor="k",
        label=r"d-type rib (fails, $R^2=-0.94$)")
axB.bar(x + w / 2, p90_k, w, color="#2e86ab", edgecolor="k",
        label=r"k-type rib (tolerated, $R^2=+0.59$)")
for i in range(4):
    axB.text(i, max(p90_d[i], p90_k[i]) * 1.05,
             r"$%.0f\times$" % (p90_d[i] / p90_k[i]), ha="center",
             fontsize=8, fontweight="bold")
axB.set_yscale("log")
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8)
axB.set_ylabel(r"conditioning tail  $\kappa_{90}$ (90th pct.)")
axB.set_title(r"(b) iso-depth ($\varepsilon_{\rm med}\!\approx\!0.52$ both): the"
              "\nconditioning tail flips with reattachment")
axB.legend(fontsize=8, loc="upper right")
axB.grid(alpha=0.25, axis="y", which="both")

fig.tight_layout()
for ext in ("png", "pdf"):
    for outdir in (NODE, MSFIG):
        fig.savefig(os.path.join(outdir, "fig_conditioning_discriminant.%s" % ext),
                    dpi=160, bbox_inches="tight")
print("wrote fig_conditioning_discriminant.{png,pdf} to node_006/ and manuscript/figures/")
print("  panel (a): A_E ~ %.2f eps^-%.2f, rho=%.2f" % (c, n, rho))
print("  panel (b): p90 tail ratio d/k =",
      [round(float(a / b), 1) for a, b in zip(p90_d, p90_k)])
