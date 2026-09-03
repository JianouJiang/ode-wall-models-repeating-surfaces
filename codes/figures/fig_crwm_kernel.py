#!/usr/bin/env python3
r"""
fig_crwm_kernel.py  (Thrust #26, L2)
Two-panel a-priori figure for the MEASURED wall-model Green's function.

(a) Measured kernel |G(xi)|/|G(0)| vs xi/y_m (mean over stations) with the
    analytic linearised kernel overlaid -- leverage peaks at the wall and
    vanishes at y_m (P-K1, P-K2).
(b) Kernel-predicted within-layer cure  d tau_w^pred = int G conv dxi  vs the
    independently-measured CR-WM(exact) cure (P-K3), with the uniform-leverage
    counterfactual to show the kernel SHAPE carries the recovery.

All inputs trace to codes/results/crwm_kernel_experiment.npz (real DNS).
Colour scheme: gray-box (CR-WM) = bluish-gray.  A-priori; no coupled number.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GRAY = "#5b6b7a"      # bluish-gray = gray-box model
ORANGE = "#e08214"    # ground truth / measured

d = np.load(os.path.join(RES, "crwm_kernel_experiment.npz"), allow_pickle=True)
xi = d["xi_norm"]
Gn = np.abs(d["G_num_mean"]); Gn /= Gn[0]
Ga = np.abs(d["G_an_mean"]); Ga /= Ga[0]
cm, cp, cu = d["cure_meas"], d["cure_pred"], d["cure_unif"]
r2c = float(d["r2_cure"]); r2u = float(d["r2_cure_uniform"])
kc = float(d["kernel_corr_med"]); decay = float(d["frac_decay_to_ym"])

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))

# ---- panel (a): kernel shape -------------------------------------------------
ax[0].plot(xi, Gn, "o-", color=ORANGE, ms=4, lw=1.8,
           label="measured  $|G(\\xi)|$ (perturbed solver)")
ax[0].plot(xi, Ga, "--", color=GRAY, lw=2.0,
           label="analytic linearised kernel")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xlabel(r"$\xi / y_m$  (height in the wall layer)")
ax[0].set_ylabel(r"normalised leverage $|G(\xi)|/|G(0)|$")
ax[0].set_title(f"(a) leverage vanishes at $y_m$ "
                f"({100*decay:.1f}% decay; corr={kc:.3f})")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")
ax[0].set_ylim(-0.05, 1.08)

# ---- panel (b): kernel predicts the cure ------------------------------------
lim = 1.05 * np.nanmax(np.abs(np.concatenate([cm, cp])))
ax[1].plot([-lim, lim], [-lim, lim], "k-", lw=0.8, alpha=0.6)
ax[1].scatter(cm, cp, s=14, color=GRAY, alpha=0.7,
              label=f"kernel-weighted ($R^2$={r2c:.2f})")
ax[1].scatter(cm, cu, s=10, color="#bbbbbb", alpha=0.5, marker="x",
              label=f"uniform leverage ($R^2$={r2u:.0f})")
ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
ax[1].set_xlabel(r"measured CR-WM(exact) cure  $\Delta\tau_w$")
ax[1].set_ylabel(r"kernel-predicted cure  $\int G\,\mathrm{conv}\,d\xi$")
ax[1].set_title("(b) the kernel reproduces the measured cure")
ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")

fig.tight_layout()
out = os.path.join(HERE, "fig_crwm_kernel.pdf")
fig.savefig(out, dpi=200)
fig.savefig(out.replace(".pdf", ".png"), dpi=160)
print("wrote", out)
