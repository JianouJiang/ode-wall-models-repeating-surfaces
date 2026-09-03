#!/usr/bin/env python3
r"""
fig_regularisation_impossibility.py
===================================
Thrust #30, Level 3 results figure.  The closure-independent IMPOSSIBILITY
result, shown on the POWERED per-station axis (n = 512 hill-surface stations),
with bootstrap CIs and significance -- not the underpowered n=5 cross-closure
correlations.  Four panels, all read from results/impossibility_results_l3.npz
(produced by development/nodes/node_003/results_impossibility.py); the gain-shape
panel and the kappa scatter also use closure_conditioning_floor.npz.

 (a) COHERENCE SPLIT with 95% station-bootstrap CIs: Spearman(kappa,1/eps) per
     closure.  Eddy closures 0.85-0.97 (CIs disjoint from) exact-DNS 0.19.
 (b) EXACT-DNS IS THE WORST: per-station kappa distribution per closure (log y);
     the exact stress sits ~41x above the eddy band (Mann-Whitney p~1e-194).
 (c) KAWAI-LARSSON INVERSION: |R^2(tau_w)| deepens monotonically with matching
     height y_m+ for ALL five closures -- matching higher makes it worse.
 (d) THE MECHANISM: inverse-operator gain w_g(y).y_m at the median-eps station.
     The exact stress is FLAT (Tikhonov lambda=0, full ill-posedness); every eddy
     closure is outer-damped (the eddy viscosity is a low-pass regulariser).

Colour scheme (project convention): orange = the DNS-truth-derived closure
(exact stress), bluish-grays = the eddy / gray-box closures.
Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_regularisation_impossibility.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
L3 = os.path.join(RES, "impossibility_results_l3.npz")
FLOOR = os.path.join(RES, "closure_conditioning_floor.npz")
OUT = os.path.join(CODES, "..", "manuscript", "figures",
                   "fig_regularisation_impossibility.pdf")

COL = {
    "A_ml_vandriest":  ("#1f3b57", "o", "ML van Driest"),
    "B_alg_cebeci":    ("#3a6ea5", "s", "Cebeci"),
    "C_alg_sa":        ("#6699cc", "^", "SA-like"),
    "D_alg_reichardt": ("#9bbcd8", "D", "Reichardt"),
    "E_dns_stress":    ("#e8820c", "*", "exact DNS stress"),
}

d = np.load(L3, allow_pickle=True)
fl = np.load(FLOOR, allow_pickle=True)
keys = [str(k) for k in d["closure_keys"]]
eddy = [str(k) for k in d["eddy_keys"]]
dns = str(d["dns_key"])
order = eddy + [dns]                       # eddy first, exact stress last

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm"})
fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4))
axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# ----------------------------------------------------------------- (a) coherence
rho = {k: float(v) for k, v in zip(keys, d["coh_rho"])}
lo = {k: float(v) for k, v in zip(keys, d["coh_ci_lo"])}
hi = {k: float(v) for k, v in zip(keys, d["coh_ci_hi"])}
xs = np.arange(len(order))
for i, k in enumerate(order):
    c, _, lab = COL[k]
    err = [[rho[k] - lo[k]], [hi[k] - rho[k]]]
    # point estimate + 95% CI as a dot-with-error-bar (no bars)
    axA.errorbar(i, rho[k], yerr=err, fmt="o", color=c, ms=9, mec="k",
                 mew=0.6, ecolor=c, elinewidth=1.6, capsize=4, zorder=3)
# shaded band marking the gap between the DNS CI top and the lowest eddy CI bottom
axA.axhspan(float(d["RA_dns_hi"]), float(d["RA_eddy_lo"]), color="0.85",
            alpha=0.6, zorder=0)
axA.text(0.05, 0.5 * (float(d["RA_dns_hi"]) + float(d["RA_eddy_lo"])),
         "disjoint\n95% CIs", fontsize=6.6, va="center", color="0.35")
axA.set_xticks(xs)
axA.set_xticklabels([COL[k][2] for k in order], rotation=30, ha="right", fontsize=6.6)
axA.set_ylabel(r"$\mathrm{Spearman}(\kappa_{\rm closure},\,1/\varepsilon)$")
axA.set_ylim(0, 1.05)
axA.set_title("(a) coherence split (n=512, bootstrap 95% CI)", fontsize=8.6)
axA.grid(True, axis="y", alpha=0.2)

# -------------------------------------------------------- (b) kappa distributions
data, cols, ticklab = [], [], []
for k in order:
    kap = fl[f"hills_kappa_{k}"]
    kap = kap[np.isfinite(kap) & (kap > 0)]
    data.append(kap); cols.append(COL[k][0]); ticklab.append(COL[k][2])
bp = axB.boxplot(data, positions=xs, widths=0.6, patch_artist=True,
                 showfliers=False, medianprops=dict(color="k", lw=1.1),
                 whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.85)
axB.set_yscale("log")
axB.set_xticks(xs)
axB.set_xticklabels(ticklab, rotation=30, ha="right", fontsize=6.6)
axB.set_ylabel(r"per-station $\kappa_{\rm closure}$")
ratio = float(d["RB_ratio"]); rci = d["RB_ratio_ci"]
axB.set_title(r"(b) exact stress is the worst: $%.0f\times$ amplified" % ratio,
              fontsize=8.6)
axB.annotate(r"$%.0f\times$ (95%% CI $%.0f$-$%.0f$)" % (ratio, rci[0], rci[1]) +
             "\n" + r"Mann-Whitney $p\approx10^{-194}$",
             xy=(len(order) - 1, np.median(data[-1])),
             xytext=(0.02, 0.93), textcoords="axes fraction",
             fontsize=6.6, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))
axB.grid(True, axis="y", which="both", alpha=0.18)

# ----------------------------------------------------------- (c) y_m inversion
yms = np.asarray(d["ym_plus"], float)
ym_r2 = np.asarray(d["ym_r2"], float)       # rows in `keys` order
for i, k in enumerate(keys):
    c, mk, lab = COL[k]
    axC.plot(yms, np.abs(ym_r2[i]), marker=mk, color=c, lw=1.3, ms=5,
             label=lab)
axC.set_yscale("log")
axC.set_xlabel(r"matching height $y_m^+$")
axC.set_ylabel(r"$|R^2(\tau_w)|$  (catastrophe depth)")
axC.set_title("(c) matching higher makes it worse", fontsize=8.6)
axC.annotate("Kawai-Larsson\nprescription inverted",
             xy=(yms[-1], np.abs(ym_r2[0, -1])), xytext=(0.05, 0.78),
             textcoords="axes fraction", fontsize=6.6,
             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
axC.legend(fontsize=6.0, loc="lower right", framealpha=0.9, ncol=1)
axC.grid(True, which="both", alpha=0.18)
axC.set_xticks(yms)

# ---------------------------------------------------------- (d) gain-shape w_g
if "wg_y_over_ym" in d.files:
    yy = np.asarray(d["wg_y_over_ym"], float)
    for k in keys:
        wk = f"wg_{k}"
        if wk not in d.files:
            continue
        c, _, lab = COL[k]
        lw = 2.0 if k == dns else 1.2
        axD.plot(yy, d[wk], color=c, lw=lw, label=lab,
                 zorder=3 if k == dns else 2)
    axD.axhline(1.0, color="0.6", ls=":", lw=0.8)
    axD.text(0.62, 1.04, r"flat gain $w_g y_m\!=\!1$ (exact stress, $\lambda\!=\!0$)",
             fontsize=6.2, color="0.35")
    axD.set_xlabel(r"$y/y_m$")
    axD.set_ylabel(r"inverse-operator gain $w_g(y)\,y_m$")
    axD.set_title(r"(d) the eddy viscosity is a Tikhonov regulariser",
                  fontsize=8.6)
    axD.legend(fontsize=6.0, loc="upper left", framealpha=0.9)
    axD.grid(True, alpha=0.18)
    axD.set_xlim(0, 1)
else:
    axD.text(0.5, 0.5, "w_g not available", ha="center", va="center")

fig.tight_layout(pad=0.6)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", OUT)
print("wrote", OUT.replace(".pdf", ".png"))
