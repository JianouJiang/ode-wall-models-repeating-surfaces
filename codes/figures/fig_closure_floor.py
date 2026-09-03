#!/usr/bin/env python3
r"""
fig_closure_floor.py
====================
Figure for the conditioning section: the model-independent accuracy floor.

Reads results/closure_conditioning_floor.npz (produced by
analysis/closure_conditioning_floor.py) and draws two panels:

 (a) closure-channel condition number kappa_closure vs 1/eps on the periodic-hill
     failure geometry, one series per closure.  The four standard eddy-viscosity
     closures rise as kappa_closure ~ 1/eps (Spearman 0.85-0.97); the exact DNS
     stress sits uniformly hyper-amplified (median kappa ~ 24, ~38x the model
     closures) -- the mechanism of the exact-stress paradox.

 (b) a-priori R^2(tau_w) for every closure on three geometries: the O(delta)-pitch
     periodic hill (catastrophic for ALL closures, incl. exact DNS), the BFS
     success geometry (all model closures fine), and the wide-pitch conv-div
     repeating control (all closures fine).  The floor is regime-specific to
     O(delta)-pitch repetition, not "closures are bad" and not "any repeating wall".

No fabrication: every number is read from the npz.
Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_closure_floor.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "closure_conditioning_floor.npz")
OUT = os.path.join(CODES, "..", "manuscript", "figures", "fig_closure_floor.pdf")

d = np.load(RES, allow_pickle=True)
keys = list(d["closure_keys"])
labels = list(d["closure_labels"])
eps = d["hills_eps"]
inv = 1.0 / eps

# colour scheme: model closures in bluish-grays (the gray-box family), the exact
# DNS stress in orange (it is the DNS-truth-derived closure).
COL = {
    "A_ml_vandriest": ("#1f3b57", "o", "ML van Driest"),
    "B_alg_cebeci":   ("#3a6ea5", "s", "Cebeci"),
    "C_alg_sa":       ("#6699cc", "^", "SA-like"),
    "D_alg_reichardt":("#9bbcd8", "D", "Reichardt"),
    "E_dns_stress":   ("#e8820c", "*", "exact DNS stress"),
}

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.1))

# ---- panel (a): kappa_closure vs 1/eps, per closure -------------------------
for k in keys:
    kap = d[f"hills_kappa_{k}"]
    m = np.isfinite(kap) & np.isfinite(inv) & (kap > 0)
    c, mk, lab = COL[k]
    axA.scatter(inv[m], kap[m], s=8, c=c, marker=mk, alpha=0.45,
                edgecolors="none", label=lab, rasterized=True)
# 1/eps reference guide (slope-1 in log-log), anchored to the model-closure band
xr = np.array([inv[np.isfinite(inv)].min(), inv[np.isfinite(inv)].max()])
axA.plot(xr, 0.08 * xr, "k--", lw=1.0, label=r"$\kappa\propto 1/\varepsilon$")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel(r"$1/\varepsilon$")
axA.set_ylabel(r"closure-channel condition number $\kappa_{\rm closure}$")
axA.set_title("(a) hills: every closure amplified as $\\varepsilon\\!\\to\\!0$", fontsize=9)
axA.legend(fontsize=6.2, loc="upper left", framealpha=0.9, ncol=1)
axA.grid(True, which="both", alpha=0.18)

# ---- panel (b): a-priori R^2(tau_w) per closure x geometry ------------------
hrows = {r["label"]: r for r in d["hills_rows"]}
geom = list(d["geom_rows"])
bfs = next(g for g in geom if g.get("eps_med", 0) > 5 and g["role"] == "success")
ctrl = next(g for g in geom if g["role"] == "control")

geom_sets = [
    ("periodic hill\n$O(\\delta)$ pitch", "hills"),
    ("BFS\n(success)", "bfs"),
    ("conv-div\nwide-pitch control", "ctrl"),
]
x = np.arange(len(geom_sets))
w = 0.16
for j, k in enumerate(keys):
    vals = []
    for _, tag in geom_sets:
        if tag == "hills":
            vals.append(hrows[labels[j]]["r2"])
        elif tag == "bfs":
            vals.append(bfs[f"r2_{k}"])
        else:
            vals.append(ctrl[f"r2_{k}"])
    c, mk, lab = COL[k]
    # symlog-ish display: clip catastrophic R2 to a floor for readability, annotate
    disp = [max(v, -3.0) for v in vals]
    bars = axB.bar(x + (j - 2) * w, disp, w, color=c, label=lab, edgecolor="k", lw=0.3)
    # annotate the catastrophic hills bar with the true value
    if vals[0] < -3.0:
        axB.text(x[0] + (j - 2) * w, -2.95, f"{vals[0]:.0f}", ha="center",
                 va="bottom", fontsize=5.0, rotation=90, color="white")
axB.axhline(0, color="k", lw=0.7)
axB.set_xticks(x)
axB.set_xticklabels([g[0] for g in geom_sets], fontsize=7.5)
axB.set_ylim(-3.2, 1.25)
axB.set_ylabel(r"a-priori $R^2(\tau_w)$  (clipped at $-3$)")
axB.set_title("(b) the floor is $O(\\delta)$-pitch specific", fontsize=9)
axB.legend(fontsize=6.0, loc="lower right", ncol=1, framealpha=0.9)
axB.grid(True, axis="y", alpha=0.18)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("wrote", os.path.normpath(OUT))
