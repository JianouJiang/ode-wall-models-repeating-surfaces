#!/usr/bin/env python3
r"""
Figure (Thrust #27, L3): the TRANSMISSION SPECTRUM of the force-cancellation
diagnostic eps -- three panels at decreasing statistical power.

Consumes codes/results/thrust27_l3_transmission.npz (built by
codes/analysis/thrust27_l3_transmission.py from real DNS/WMLES npz only).

(a) WITHIN-GEOMETRY (n=30, powered).  Sweep the matching height y_m on the fixed
    alpha=1.0 hill.  The a-priori traction error follows a power law
    relRMS = A*eps^-b with a perfect rank correlation (rho=-1.0); bootstrap band
    on the exponent b.

(b) WITHIN-FAMILY (n=29, powered, a-priori).  Vary the repeating-structure shape
    (Xiao steepness/pitch family).  The GEOMETRIC predictor L_sep/delta collapses
    the a-priori failure depth R^2(tau_w): rho=-0.75, CI excludes 0.  Every case
    lives in the O(delta) trigger window L_sep/delta ~ 1-5.

(c) CROSS-GEOMETRY (n=5, underpowered -- honest negative).  Five DISTINCT coupled
    WMLES geometries.  The local cancellation depth eps does NOT cross-rank the
    GLOBAL coupled reattachment error e_reatt (Spearman=+0.10): the alpha=0.8
    hill has the deepest cancellation of the Re5600 triplet yet the smallest
    reattachment error -- a local measure read against a global one.

Colours encode geometry class (no model curves here, so the reserved model
colours are not used): repeating pitch~O(delta) = crimson; cross-Re hill =
firebrick; conv-div wide-pitch control = steelblue.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "thrust27_l3_transmission.npz")
OUTPDF = os.path.join(HERE, "fig_l3_transmission_spectrum.pdf")
OUTPNG = os.path.join(HERE, "fig_l3_transmission_spectrum.png")

d = np.load(RES, allow_pickle=True)
plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9))

# ---------------------------------------------------------------- panel (a)
A_eps = d["A_eps"]; A_relrms = d["A_relrms"]
A_b = float(d["A_b"]); A_logA = float(d["A_logA"]); A_r2 = float(d["A_r2"])
A_ci = d["A_b_ci"]; A_rho = float(d["A_rho"])
xs = np.logspace(np.log10(A_eps.min()), np.log10(A_eps.max()), 100)
ax[0].plot(A_eps, A_relrms, "o", ms=5, color="crimson", mec="k", mew=0.4,
           zorder=3, label="WMLES $y_m$ sweep ($n=30$)")
ax[0].plot(xs, np.exp(A_logA) * xs ** (-A_b), "-", color="k", lw=1.6,
           label=fr"$\propto\bar\varepsilon^{{-{A_b:.2f}}}$  ($R^2={A_r2:.3f}$)")
# bootstrap band on exponent
ylo = np.exp(A_logA) * xs ** (-A_ci[1])
yhi = np.exp(A_logA) * xs ** (-A_ci[0])
ax[0].fill_between(xs, ylo, yhi, color="k", alpha=0.13, lw=0,
                   label=fr"95% CI $b\in[{A_ci[0]:.2f},{A_ci[1]:.2f}]$")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"local cancellation depth  $\bar\varepsilon$")
ax[0].set_ylabel(r"a-priori traction error  relRMS$(\tau_w)$")
ax[0].set_title(r"(a) within-geometry  $\rho=-1.00$", fontsize=10)
ax[0].legend(fontsize=7.2, loc="upper right", framealpha=0.9)
ax[0].grid(True, which="both", alpha=0.18)

# ---------------------------------------------------------------- panel (b)
B_Lsep = d["B_Lsep_over_delta"]; B_r2 = d["B_r2"]; B_alpha = d["B_alpha"]
B_rho = float(d["B_rho_Lsep_r2"]); B_ci = d["B_Lsep_r2_ci"]
sc = ax[1].scatter(B_Lsep, B_r2, c=B_alpha, cmap="viridis", s=38,
                   edgecolor="k", linewidth=0.4, zorder=3)
cb = fig.colorbar(sc, ax=ax[1], pad=0.02)
cb.set_label(r"hill steepness $\alpha$", fontsize=8)
ax[1].set_xlabel(r"separation length / b.l. thickness  $L_{\rm sep}/\delta$")
ax[1].set_ylabel(r"a-priori failure depth  $R^2(\tau_w)$")
ax[1].set_title(fr"(b) within-family  $\rho={B_rho:.2f}$ "
                fr"CI$[{B_ci[0]:.2f},{B_ci[1]:.2f}]$", fontsize=10)
ax[1].grid(True, alpha=0.18)
ax[1].text(0.50, 0.06, r"$L_{\rm sep}/\delta\sim\mathcal{O}(\delta)$ trigger window",
           transform=ax[1].transAxes, fontsize=7.6, ha="center",
           bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.6", alpha=0.85))

# ---------------------------------------------------------------- panel (c)
C_labels = [str(x) for x in d["C_labels"]]
C_eps = d["C_eps"]; C_er = d["C_ereatt"]
col = []
for s in C_labels:
    if "Re10595" in s:
        col.append("firebrick")
    elif "conv" in s.lower() or "convdiv" in s.lower() or "Laval" in s:
        col.append("steelblue")
    else:
        col.append("crimson")
ax[2].scatter(C_eps, C_er, c=col, s=70, edgecolor="k", linewidth=0.6, zorder=3)
# manual offsets to avoid the alpha=1.0 / alpha=1.2 label collision near the top
offsets = {
    "0.8": (6, -14),
    "1.0 (Re5600)": (-4, 10),
    "1.2": (8, -2),
    "Re10595": (8, -2),
    "conv": (-6, -16),
}
for s, x, y in zip(C_labels, C_eps, C_er):
    tag = (s.replace("hill ", "").replace(" (Re5600)", "")
            .replace("(Re10595, cross-Re)", "Re10595")
            .replace("conv-div channel (Laval2011, control)", "conv-div"))
    key = next((k for k in offsets if k in s), "0.8")
    ha = "right" if offsets[key][0] < 0 else "left"
    ax[2].annotate(tag, (x, y), fontsize=6.8, xytext=offsets[key],
                   textcoords="offset points", ha=ha)
ax[2].set_xscale("log")
ax[2].set_xlabel(r"local cancellation depth  $\bar\varepsilon$")
ax[2].set_ylabel(r"coupled reattachment error  $e_{\rm reatt}$")
ax[2].set_title(r"(c) cross-geometry  $\rho=+0.10$ ($n=5$)", fontsize=10)
ax[2].grid(True, which="both", alpha=0.18)
# annotate the local-vs-global decoupling
ax[2].annotate("deepest cancellation,\nsmallest global error",
               xy=(C_eps[0], C_er[0]), xytext=(0.30, 0.62),
               textcoords="axes fraction", fontsize=6.8, color="crimson",
               ha="center",
               arrowprops=dict(arrowstyle="->", color="crimson", lw=0.9))

fig.tight_layout()
fig.savefig(OUTPDF, bbox_inches="tight")
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print("wrote", OUTPDF)
print("wrote", OUTPNG)
