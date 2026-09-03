#!/usr/bin/env python3
r"""Figure 3: the dropped term is the dispersive (form-induced) stress, and its
magnitude tracks failure.  Two panels, all from real DNS/LES
(codes/results/dispersive_budget_l2.npz and dispersive_l3_results.npz).

 (a) Near-wall stress identity on the canonical hill,
     tau/rho = nu d<U>/deta - <u'v'> - <u~v~>.  The Reynolds stress the ODE keeps
     (bluish-gray) and the dispersive stress it drops (orange) are nearly equal and
     opposite, so the net shear (black dashed) is a tiny residual.  (A Reynolds-only
     reconstruction over-predicts tau_w ~45x -- quoted in the text.)

 (b) The dispersive over-prediction Delta = |<u~v~>(eta_m)|/mean|tau_w| versus the
     a-priori R^2(tau_w) for the 29 Xiao hills (circles) and the 9 probe/control
     geometries (squares).  Delta is large only where the ODE fails (R^2<0); every
     tolerated geometry (R^2~1) has Delta<~1.

Colour scheme (paper-wide): orange = dispersive / failing (the omitted truth term),
bluish-gray = Reynolds / tolerated (the gray-box kept term), black = residual.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_dispersive_budget.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-size parity with the dose-response reference (~7.1 pt printed labels):
# this canvas is 10.4 in wide at full \textwidth, so labels need ~14 pt.
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                     "legend.fontsize": 12.5})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
FIG = os.path.join(ROOT, "codes", "figures")
MAN = os.path.join(ROOT, "manuscript", "figures")

ORANGE = "#E69F00"   # dispersive / failing (omitted truth term)
GRAY = "#56707A"     # bluish-gray: Reynolds / tolerated (kept term)
BLACK = "#111111"    # residual

d = np.load(os.path.join(RES, "dispersive_budget_l2.npz"), allow_pickle=True)

# ---- panel (a): stress identity on the canonical hill -------------------------
eta = d["f2_hill_eta"]; disp = d["f2_hill_disp"]; reyn = d["f2_hill_reyn"]
eta_m = float(d["f2_eta_m"])
mA = (eta > 0) & (eta <= 0.30) & np.isfinite(disp) & np.isfinite(reyn)

# ---- panel (b): Delta vs a-priori R^2 -----------------------------------------
a29 = d["lad29_alpha"].astype(float); Del29 = d["lad29_Delta"].astype(float)
r229 = d["lad29_r2"].astype(float)
bfail = d["bt_fail"].astype(bool); bDel = d["bt_Delta"].astype(float)
br2 = d["bt_r2"].astype(float)
THR = 0.5 * (np.nanmin(Del29) + bDel[~bfail].max())   # ~5.9

fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.15))

# ----- (a) ---------------------------------------------------------------------
a = ax[0]
a.plot(-reyn[mA], eta[mA], color=GRAY, lw=2.2,
       label=r"$-\langle u'v'\rangle$ (Reynolds, kept)")
a.plot(-disp[mA], eta[mA], color=ORANGE, lw=2.2,
       label=r"$-\langle\tilde u\tilde v\rangle$ (dispersive, dropped)")
a.plot(-(reyn[mA] + disp[mA]), eta[mA], color=BLACK, lw=2.2, ls="--",
       label=r"net shear $\tau/\rho$ (residual)")
a.axvline(0.0, color="0.8", lw=0.8, zorder=0)
a.axhline(eta_m, color="0.55", lw=0.9, ls=":")
a.text(0.985, eta_m + 0.006, r"$\eta_m$", color="0.4", ha="right", va="bottom",
       fontsize=12.5, transform=a.get_yaxis_transform())
a.set_xticks([-0.015, 0, 0.015])
a.set_xlabel(r"near-wall shear stress  $\tau/\rho$")
a.set_ylabel(r"$\eta=y-y_{\rm wall}\;(\delta)$")
a.set_ylim(0, 0.30)
a.legend(fontsize=10.5, loc="upper right", framealpha=0.92, handlelength=1.5)
a.set_title("(a) dispersive stress cancels the Reynolds stress", fontsize=13.5, loc="left")

# ----- (b) ---------------------------------------------------------------------
b = ax[1]
b.scatter(r229, Del29, s=40, color=ORANGE, edgecolor="k", lw=0.3, alpha=0.85,
          zorder=3, label="Xiao hill family (29, fail)")
b.scatter(br2[bfail], bDel[bfail], marker="s", s=72, color=ORANGE, edgecolor="k",
          lw=0.6, zorder=4, label="canonical hill")
b.scatter(br2[~bfail], bDel[~bfail], marker="s", s=64, facecolor="none",
          edgecolor=GRAY, lw=1.6, zorder=4, label="tolerated controls (8)")
b.axvline(0, color="0.7", lw=0.9, ls=":")
b.axhline(THR, color="0.55", lw=0.9, ls="--")
b.text(float(np.nanmin(r229)) + 2, THR + 0.5, "decision boundary", color="0.45",
       ha="left", va="bottom", fontsize=11)
b.set_xlabel(r"a-priori  $R^2(\tau_w)$")
b.set_ylabel(r"dispersive over-prediction  $\Delta$")
b.set_ylim(-1.5, 27)
b.legend(fontsize=11, loc="lower left", framealpha=0.94)
b.set_title(r"(b) $\Delta$ is large only where the ODE fails ($R^2<0$)", fontsize=13.5, loc="left")

fig.tight_layout()
for outdir in (FIG, MAN):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "fig_dispersive_budget.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "fig_dispersive_budget.png"), dpi=160,
                bbox_inches="tight")
print("WROTE fig_dispersive_budget.{pdf,png} to codes/figures and manuscript/figures")
