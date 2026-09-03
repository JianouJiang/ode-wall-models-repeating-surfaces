#!/usr/bin/env python3
r"""
plot_periodic_solvability.py  --  L2 diagnostic figure for Thrust #15.

Reads results/periodic_solvability.npz (produced by periodic_solvability.py) and
renders a 3-panel diagnostic:

  (a) F2: |rho(log|C|,log eps)| for streamwise C_x (annihilated) vs wall-normal
      C_y (survives) per geometry, with bootstrap 95% CIs.  C_y is the stronger
      eps-correlate for the canonical hill + steeper hills + control; the
      shallowest alph05 is the marginal exception (shown, not hidden).
  (b) CV band closure: |P[C_x]|/P|C_x| (streamwise, annihilated, ~0) vs
      |P[C_y]|/P|C_y| (wall-normal, survives, O(1)) per geometry -- the direct,
      theorem-grade signature of the period-operator projection.
  (c) eps regime: median eps per geometry, trigger (eps<<1) vs control (eps~O(1)).

A priori, real DNS, every number from the npz.  Saved to the node directory.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
NODE = os.path.join(os.path.dirname(ROOT), "development", "nodes", "node_002")
os.makedirs(NODE, exist_ok=True)

d = np.load(os.path.join(RES, "periodic_solvability.npz"), allow_pickle=True)
labels = [str(s) for s in d["geom_labels"]]

pretty = {
    "periodic_hills_1p0": "hills 1p0\n(canonical)",
    "xiao_alph05": "hills a0.5\n(shallow)",
    "xiao_alph10": "hills a1.0",
    "xiao_alph10b": "hills a1.0'",
    "xiao_alph15": "hills a1.5\n(steep)",
    "conv_div": "conv-div\n(control)",
}
order = ["periodic_hills_1p0", "xiao_alph05", "xiao_alph10", "xiao_alph10b",
         "xiao_alph15", "conv_div"]
order = [l for l in order if l in labels]
xs = np.arange(len(order))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# ---- (a) F2 correlation magnitudes with CIs -------------------------------
ax0 = ax[0]
for k, geo in enumerate(order):
    rx = abs(float(d[f"{geo}__rho_x"]))
    ry = abs(float(d[f"{geo}__rho_y"]))
    cix = np.abs(d[f"{geo}__ci_x"])
    ciy = np.abs(d[f"{geo}__ci_y"])

    def yerr(r, ci):
        lo = max(0.0, r - ci.min())
        hi = max(0.0, ci.max() - r)
        return [[lo], [hi]]
    ax0.errorbar(k - 0.13, rx, yerr=yerr(rx, cix),
                 fmt="s", color="0.45", ms=7, capsize=3,
                 label="streamwise $C_x$ (annihilated)" if k == 0 else None)
    ax0.errorbar(k + 0.13, ry, yerr=yerr(ry, ciy),
                 fmt="o", color="#1f6f3f", ms=7, capsize=3,
                 label="wall-normal $C_y$ (survives)" if k == 0 else None)
ax0.set_xticks(xs)
ax0.set_xticklabels([pretty[l] for l in order], fontsize=8)
ax0.set_ylabel(r"$|\rho(\log|C|,\ \log\varepsilon)|$")
ax0.set_title("(a) F2: wall-normal $C_y$ is the stronger\n$\\varepsilon$-correlate (95% CI)")
ax0.legend(fontsize=8, loc="upper right")
ax0.grid(alpha=0.3)

# ---- (b) CV band closure --------------------------------------------------
ax1 = ax[1]
Rx = [float(d[f"{geo}__cv_Rx"]) for geo in order]
Ry = [float(d[f"{geo}__cv_Ry"]) for geo in order]
ax1.bar(xs - 0.18, Rx, width=0.36, color="0.45",
        label=r"$|P[C_x]|/P|C_x|$ (annihilated)")
ax1.bar(xs + 0.18, Ry, width=0.36, color="#1f6f3f",
        label=r"$|P[C_y]|/P|C_y|$ (survives)")
ax1.set_xticks(xs)
ax1.set_xticklabels([pretty[l] for l in order], fontsize=8)
ax1.set_ylabel("period-mean / mean-abs")
ax1.set_title("(b) CV band: period operator annihilates\nstreamwise, keeps wall-normal")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, axis="y")

# ---- (c) eps regime -------------------------------------------------------
ax2 = ax[2]
emed = [float(d[f"{geo}__eps_median"]) for geo in order]
cols = ["#1f6f3f" if str(d[f"{geo}__kind"]) == "trigger" else "#b03030"
        for geo in order]
ax2.bar(xs, emed, color=cols)
ax2.axhline(0.1, ls="--", color="k", lw=1)
ax2.text(len(order) - 1.0, 0.115, r"$\varepsilon=0.1$", fontsize=8)
ax2.set_yscale("log")
ax2.set_xticks(xs)
ax2.set_xticklabels([pretty[l] for l in order], fontsize=8)
ax2.set_ylabel(r"median $\varepsilon$")
ax2.set_title("(c) $\\varepsilon$ regime: hills trigger ($\\ll1$),\nconv-div control stays $O(1)$ [F3]")
ax2.grid(alpha=0.3, axis="y")

fig.suptitle(
    "Periodicity-solvability (Thrust #15): the period operator annihilates the "
    "streamwise seesaw; the surviving wall-normal advection sets $\\tau_w$ "
    "(a priori, real DNS)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(NODE, "fig_periodic_solvability.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
