#!/usr/bin/env python3
r"""
fig_periodicity_solvability.py
==============================
Manuscript figure for the solvability section (\S sec:periodicity, Thrust #15).

Reads results/periodic_solvability.npz (produced by
analysis/periodic_solvability.py) and draws the three pieces of evidence that the
period operator P annihilates the streamwise pressure-convection seesaw and leaves
the wall-normal advection as the agent that sets the residual wall stress:

 (a) CV band-closure -- the direct, theorem-grade signature.  Per geometry, the
     period-mean-to-mean-abs ratio of the two transports on the wall-following
     matching band: streamwise C_x = int U dU/dx dy is annihilated
     (|P[C_x]|/P|C_x| ~ 0.02-0.04, i.e. 20-44x suppressed) while wall-normal
     C_y = int V dU/dy dy survives (|P[C_y]|/P|C_y| ~ 0.7-0.9).  Universal across
     all six geometries, INCLUDING the conv-div control and the shallow a0.5 hill
     that the noisier station-level test (panel b) does not resolve.

 (b) F2 -- the surviving wall-normal term is the stronger eps-correlate.  Per
     geometry, |rho(log|C|,log eps)| (Spearman) for C_x vs C_y with bootstrap 95%
     CIs.  C_y dominates for the canonical hill, the steeper hills, the pooled
     hill family (N=3392) and the control; the shallowest hill a0.5 (nearest the
     eps~O(1) boundary) is the marginal exception, shown not hidden.

 (c) eps regime -- annihilation is necessary, not sufficient.  Median eps per
     geometry on a log axis; the O(delta)-pitch hills sit at eps~0.08-0.21 (the
     failure regime, eps<0.1 over 26-56% of stations) while the wide-pitch
     conv-div control, which ALSO has the annihilated seesaw, stays eps~3.3.

Colour scheme: this is a mechanism decomposition, not a model comparison, so it
does NOT use the reserved model colours (orange=truth, green=Spalding,
black=black-box, bluish-gray=gray-box).  Streamwise (annihilated) C_x is a muted
slate; wall-normal (surviving) C_y is a dark crimson; the conv-div control is
flagged with a hatch / open marker throughout.

No fabrication: every number is read from the npz.
Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_periodicity_solvability.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "periodic_solvability.npz")
OUT = os.path.join(CODES, "..", "manuscript", "figures",
                   "fig_periodicity_solvability.pdf")

d = np.load(RES, allow_pickle=True)
labels = [str(s) for s in d["geom_labels"]]

# display order: canonical hill first, steepness family, control last
order = ["periodic_hills_1p0", "xiao_alph05", "xiao_alph10", "xiao_alph10b",
         "xiao_alph15", "conv_div"]
order = [l for l in order if l in labels]

pretty = {
    "periodic_hills_1p0": "hills 1.0\n(canonical)",
    "xiao_alph05": r"hills $\alpha$0.5" + "\n(shallow)",
    "xiao_alph10": r"hills $\alpha$1.0",
    "xiao_alph10b": r"hills $\alpha$1.0$'$",
    "xiao_alph15": r"hills $\alpha$1.5" + "\n(steep)",
    "conv_div": "conv-div\n(control)",
}
is_ctrl = {g: (str(d[f"{g}__kind"]) == "control") for g in order}

C_X = "#5b6b7a"   # streamwise advection -- annihilated (muted slate)
C_Y = "#b2182b"   # wall-normal advection -- survives (dark crimson)
xs = np.arange(len(order))

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.2, 3.4))

# ====================================================================== (a)
# CV band-closure: period-mean / mean-abs of C_x (annihilated) vs C_y (survives)
Rx = np.array([float(d[f"{g}__cv_Rx"]) for g in order])
Ry = np.array([float(d[f"{g}__cv_Ry"]) for g in order])
# dumbbell dot plot: per geometry, the annihilated C_x (low) and surviving C_y
# (high) joined by a line -- the vertical gap IS the suppression (no bars).
axA.plot([], [], "s", color=C_X, label=r"streamwise $C_x$ (annihilated)")
axA.plot([], [], "o", color=C_Y, label=r"wall-normal $C_y$ (survives)")
for k, g in enumerate(order):
    ctrl = is_ctrl[g]
    axA.plot([k, k], [Rx[k], Ry[k]], "-", color="0.6", lw=1.1, zorder=1)
    axA.plot(k, Rx[k], "s", color=C_X, mfc=("none" if ctrl else C_X), mew=1.2,
             ms=8, zorder=3)
    axA.plot(k, Ry[k], "o", color=C_Y, mfc=("none" if ctrl else C_Y), mew=1.2,
             ms=8, zorder=3)
    axA.text(k, Ry[k] + 0.05, f"{Ry[k] / Rx[k]:.0f}$\\times$", ha="center",
             va="bottom", fontsize=6.5, color="0.2")
axA.set_xticks(xs)
axA.set_xticklabels([pretty[l] for l in order], fontsize=6.6)
axA.set_ylabel(r"period-mean / mean-abs,  $|P[C]|/P|C|$")
axA.set_ylim(0, 1.18)
axA.set_title("(a) CV band: $P$ annihilates streamwise,\nkeeps wall-normal "
              "(universal)", fontsize=8.5)
axA.legend(fontsize=6.6, loc="upper center", framealpha=0.92)
axA.grid(True, axis="y", alpha=0.18)

# ====================================================================== (b)
# F2: |rho| for C_x vs C_y with bootstrap 95% CIs; add the pooled family group
def yerr(r, ci):
    ci = np.abs(np.asarray(ci, float))
    lo = max(0.0, r - ci.min()); hi = max(0.0, ci.max() - r)
    return [[lo], [hi]]

for k, g in enumerate(order):
    rx = abs(float(d[f"{g}__rho_x"])); ry = abs(float(d[f"{g}__rho_y"]))
    open_marker = is_ctrl[g]
    axB.errorbar(k - 0.14, rx, yerr=yerr(rx, d[f"{g}__ci_x"]), fmt="s", color=C_X,
                 ms=6, capsize=2.5, mfc=("none" if open_marker else C_X),
                 label="streamwise $C_x$" if k == 0 else None)
    axB.errorbar(k + 0.14, ry, yerr=yerr(ry, d[f"{g}__ci_y"]), fmt="o", color=C_Y,
                 ms=6, capsize=2.5, mfc=("none" if open_marker else C_Y),
                 label="wall-normal $C_y$" if k == 0 else None)
# pooled hill family (N=3392) as a highlighted extra group
kp = len(order)
prx = abs(float(d["pool_rho_x"])); pry = abs(float(d["pool_rho_y"]))
axB.errorbar(kp - 0.14, prx, yerr=yerr(prx, d["pool_ci_x"]), fmt="s", color=C_X,
             ms=7, mec="k", mew=0.6)
axB.errorbar(kp + 0.14, pry, yerr=yerr(pry, d["pool_ci_y"]), fmt="o", color=C_Y,
             ms=7, mec="k", mew=0.6)
axB.axvspan(kp - 0.45, kp + 0.45, color="0.92", zorder=0)
axB.set_xticks(list(xs) + [kp])
axB.set_xticklabels([pretty[l] for l in order] +
                    ["pooled\nfamily\n$N{=}3392$"], fontsize=6.6)
axB.set_ylabel(r"$|\rho(\log|C|,\ \log\varepsilon)|$  (Spearman)")
axB.set_ylim(0, 0.62)
axB.set_title("(b) F2: wall-normal $C_y$ is the stronger\n"
              "$\\varepsilon$-correlate (95% CI)", fontsize=8.5)
axB.legend(fontsize=6.8, loc="upper left", framealpha=0.92)
axB.grid(True, axis="y", alpha=0.18)
# mark the one honest exception
ka = order.index("xiao_alph05")
axB.annotate("marginal\nexception", xy=(ka + 0.14, abs(float(d["xiao_alph05__rho_y"]))),
             xytext=(ka + 0.1, 0.40), fontsize=5.8, ha="center", color="0.3",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))

# ====================================================================== (c)
# eps regime: median eps per geometry, log axis, with 0.1 and 1 guides
epsm = np.array([float(d[f"{g}__eps_median"]) for g in order])
# dot/stem plot of median eps (no bars); colour flags trigger vs control
for k, g in enumerate(order):
    ctrl = is_ctrl[g]
    col = "#b2182b" if ctrl else "#5b6b7a"
    axC.plot([k, k], [0.03, epsm[k]], "-", color=col, lw=0.9, alpha=0.35,
             zorder=1)
    axC.plot(k, epsm[k], "o", color=col, mfc=("none" if ctrl else col),
             mew=1.2, ms=9, zorder=3)
    axC.text(k, epsm[k] * 1.13, f"{epsm[k]:.2f}", ha="center", va="bottom",
             fontsize=6.5)
axC.axhline(0.1, color=C_Y, ls="--", lw=1.0)
axC.axhline(1.0, color="0.4", ls=":", lw=1.0)
axC.text(len(order) - 0.5, 0.105, r"$\varepsilon=0.1$ (failure)", fontsize=6.2,
         color=C_Y, va="bottom", ha="right")
axC.text(len(order) - 0.5, 1.05, r"$\varepsilon=1$", fontsize=6.2, color="0.4",
         va="bottom", ha="right")
axC.set_yscale("log")
axC.set_xticks(xs)
axC.set_xticklabels([pretty[l] for l in order], fontsize=6.6)
axC.set_ylabel(r"median $\varepsilon = |\tau_w| / (|{\rm d}p/{\rm d}x|\,y_m)$")
axC.set_ylim(0.03, 6.0)
axC.set_title("(c) annihilation necessary, not sufficient:\n"
              "$O(\\delta)$-pitch triggers, wide-pitch does not", fontsize=8.5)
axC.grid(True, axis="y", which="both", alpha=0.18)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("wrote", os.path.normpath(OUT))

# ---- console summary (traceability) ---------------------------------------
print("\n--- traceability (from npz) ---")
for g in order:
    print(f"{g:20s} cvRx={float(d[f'{g}__cv_Rx']):.3f} cvRy={float(d[f'{g}__cv_Ry']):.3f} "
          f"|rhoCx|={abs(float(d[f'{g}__rho_x'])):.3f} |rhoCy|={abs(float(d[f'{g}__rho_y'])):.3f} "
          f"gap={float(d[f'{g}__gap']):+.3f} eps_med={float(d[f'{g}__eps_median']):.3f}")
print(f"pooled: |rhoCx|={abs(float(d['pool_rho_x'])):.3f} |rhoCy|={abs(float(d['pool_rho_y'])):.3f} "
      f"gap={float(d['pool_gap']):+.3f} CI={d['pool_ci_gap']}")
