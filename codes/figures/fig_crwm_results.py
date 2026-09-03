#!/usr/bin/env python3
r"""
fig_crwm_results.py
===================
Thrust #26 (L3) results figure -- four panels, all from real data on disk:

  (a) measured vs analytic wall-stress sensitivity kernel G(xi) (Green's
      function): leverage peaks at the wall, vanishes at the matching height.
  (b) kernel-predicted vs measured within-layer cure (R^2 = 0.89 with bootstrap
      CI) against the collapsed uniform-leverage null (R^2 = -23).
  (c) the eps-resolved cure law: the ODE catastrophe (|R^2_ode|) and the
      within-layer recovery fraction, per eps-quartile -- the cure self-targets
      the deep-cancellation failure regime.
  (d) coupled a-posteriori reattachment error: landed baselines (Spalding, TBLE)
      and the PRE-REGISTERED prediction band for the unlanded CR-WM cure.

Colours follow the project convention: orange = ground truth (DNS),
black = black-box / ODE-TBLE, bluish-grey = gray-box / CR-WM, green = Spalding.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_crwm_results.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figures"))

ORANGE = "#E69F00"      # ground truth / DNS
BLACK = "#000000"       # black-box / ODE-TBLE
GRAY = "#56708A"        # gray-box / CR-WM (bluish-grey)
GREEN = "#1B7837"       # Spalding

ke = np.load(os.path.join(RESULTS, "crwm_kernel_experiment.npz"), allow_pickle=True)
ra = np.load(os.path.join(RESULTS, "crwm_results_analysis.npz"), allow_pickle=True)
tw = np.load(os.path.join(RESULTS, "aposteriori_crwm_twin.npz"), allow_pickle=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))

# ---- (a) the kernel ------------------------------------------------------
xi = ke["xi_norm"]
ax[0, 0].plot(xi, ke["G_num_mean"], "o", color=BLACK, ms=4, mfc="white",
              mew=1.2, label="measured  $G_{\\rm num}(\\xi)$")
ax[0, 0].plot(xi, ke["G_an_mean"], "-", color=GRAY, lw=2.0,
              label="analytic  $G_{\\rm an}(\\xi)$")
ax[0, 0].axhline(0, color="k", lw=0.5, ls=":")
ax[0, 0].set_xlabel("$\\xi / y_m$  (height of convective source)")
ax[0, 0].set_ylabel("normalised leverage  $|G(\\xi)|/|G(0)|$")
ax[0, 0].set_title("(a)  wall-stress sensitivity kernel", loc="left")
ax[0, 0].text(0.50, 0.55,
              f"corr$(G_{{\\rm num}},G_{{\\rm an}})$ = "
              f"{float(ke['kernel_corr_med']):.4f}\n"
              f"decay wall$\\to y_m$ = {100*float(ke['frac_decay_to_ym']):.1f}%",
              transform=ax[0, 0].transAxes, fontsize=7.5, va="top",
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax[0, 0].legend(loc="upper right", frameon=False)
ax[0, 0].annotate("leverage $\\to 0$\nat $y_m$", xy=(0.98, 0.02),
                  xytext=(0.62, 0.18), fontsize=7.5,
                  arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

# ---- (b) cure prediction vs measured ------------------------------------
cm, cp, cu = ke["cure_meas"], ke["cure_pred"], ke["cure_unif"]
ci_cure = ra["ci_r2_cure"]
lim = 1.05 * np.max(np.abs(np.concatenate([cm, cp])))
ax[0, 1].plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=0.8, zorder=0)
ax[0, 1].scatter(cm, cu, s=9, color="0.75", alpha=0.5, edgecolors="none",
                 label=f"uniform-leverage null\n$R^2$ = {float(ke['r2_cure_uniform']):.0f}")
ax[0, 1].scatter(cm, cp, s=11, color=GRAY, alpha=0.8, edgecolors="none",
                 label=f"kernel cure  $\\int G\\,{{\\rm conv}}\\,d\\xi$\n"
                       f"$R^2$ = {float(ke['r2_cure']):.3f} "
                       f"[{ci_cure[0]:.2f}, {ci_cure[1]:.2f}]")
ax[0, 1].set_xlim(-lim, lim); ax[0, 1].set_ylim(-lim, lim)
ax[0, 1].set_xlabel("measured cure  $\\Delta\\tau_w$  (exact within-layer)")
ax[0, 1].set_ylabel("predicted cure  $\\Delta\\tau_w$")
ax[0, 1].set_title("(b)  the kernel predicts the cure", loc="left")
ax[0, 1].legend(loc="upper left", frameon=False)

# ---- (c) eps-resolved cure law : curves vs eps (no bars) ----------------
r2ode = np.abs(np.asarray(ra["bin_r2_ode"], float))
rec = np.asarray(ra["bin_rec"], float)
rlo, rhi = np.asarray(ra["bin_rec_lo"], float), np.asarray(ra["bin_rec_hi"], float)
epsm = np.asarray(ra["bin_eps_med"], float)
o = np.argsort(epsm)                       # ascending cancellation depth
e, ro, rc, rl, rh = epsm[o], r2ode[o], rec[o], rlo[o], rhi[o]
axc = ax[1, 0]
ln1, = axc.plot(e, ro, "o-", color=BLACK, ms=5, lw=1.6,
                label="ODE catastrophe $|R^2_{\\rm ode}|$")
axc.set_xscale("log"); axc.set_yscale("log")
axc.set_ylabel("$|R^2_{\\rm ode}|$  (catastrophe, log)")
axc.set_xlabel("cancellation depth  $\\epsilon=|\\tau_w|/(|dp/dx|\\,y_m)$  "
               "(quartile median)")
axc.set_title("(c)  the cure self-targets the failure regime", loc="left")
axc.invert_xaxis()                         # deepest cancellation on the right
axr = axc.twinx()
ln2 = axr.errorbar(e, rc, yerr=[rc - rl, rh - rc], fmt="s-", color=GRAY,
                   ms=5, capsize=3, lw=1.4,
                   label="within-layer recovery fraction")
axr.axhline(0, color="0.5", lw=0.6, ls=":")
axr.set_ylabel("recovery fraction")
axr.set_ylim(-2.0, 1.0)
axc.legend([ln1, ln2], [ln1.get_label(), "within-layer recovery fraction"],
           loc="lower left", frameon=False, fontsize=7)
axc.annotate("deeper $\\epsilon\\Rightarrow$\nworse failure,\nbetter cure",
             xy=(e[0], ro[0]), xytext=(0.08, 0.62),
             textcoords="axes fraction", fontsize=7, color="0.3",
             arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))

# ---- (d) coupled reattachment error: landed three-rung dot plot ----------
axd = ax[1, 1]
# all three rungs from the SAME coupled twin, for internal consistency
sp = float(tw["spalding_e_reatt"]); tb = float(tw["tble_e_reatt"])
crwm_e = float(tw["crwm_e_reatt"])                       # now landed: 0.295
hlo, hhi = float(ra["heur_lo"]), float(ra["heur_hi"])    # a-priori forecast band
axd.axvline(0, color=ORANGE, lw=2.0, ls="--", zorder=2)
axd.text(0, 2.62, "DNS\n(perfect)", color=ORANGE, ha="center", va="bottom",
         fontsize=7.5)
axd.axvspan(hlo, hhi, color=GRAY, alpha=0.16, zorder=0)
axd.text((hlo + hhi) / 2, -0.62, "a-priori\nforecast band", color=GRAY,
         ha="center", va="top", fontsize=6.8)
rungs = [("Spalding (equil.)", sp, GREEN, 2),
         ("ODE / TBLE", tb, BLACK, 1),
         ("CR-WM (landed)", crwm_e, GRAY, 0)]
for lab, val, col, y in rungs:
    axd.plot([0, val], [y, y], "-", color=col, lw=0.9, alpha=0.4, zorder=2)
    axd.plot(val, y, "o", color=col, ms=13, zorder=4)
    axd.annotate(f"{val:.3f}", (val, y), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
                 color=col)
axd.annotate("lands $\\it{above}$ the\nforecast band\n(cure weaker than predicted)",
             (crwm_e, 0), textcoords="offset points", xytext=(6, -26),
             ha="center", fontsize=6.8, color=GRAY)
axd.set_yticks([2, 1, 0])
axd.set_yticklabels(["Spalding", "ODE/TBLE", "CR-WM\n(landed)"], fontsize=8)
axd.set_xlabel("coupled reattachment error  $e_{\\rm reatt}$")
axd.set_xlim(-0.03, 0.45)
axd.set_ylim(-0.9, 3.0)
axd.set_title("(d)  coupled cure: convection is the only mover", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"fig_crwm_results.{ext}"), dpi=200,
                bbox_inches="tight")
print("wrote", os.path.join(FIGDIR, "fig_crwm_results.{pdf,png}"))
