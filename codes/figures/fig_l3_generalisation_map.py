#!/usr/bin/env python3
r"""
fig_l3_generalisation_map.py  --  Thrust #33 L3 headline figure (two panels).

Panel (a)  THE OPERATING MAP / GENERALISATION (G7).  The a-priori cancellation
  diagnostic epsilon_med (x, log) vs the a-priori wall-stress skill R^2 (y) for
  the 15-geometry benchmark (cross_geometry_collapse.npz), with the 4 coupled
  a-posteriori repeating-hill points + conv-div control overlaid as their
  coupled reattachment error (aposteriori_dose_response.npz).  Markers encode
  the honest taxonomy: filled = repeating structure with pitch ~ O(delta)
  (triggers the catastrophe), open = non-repeating or wide-pitch (tolerated).
  The mechanism reproduces across the repeating-hill family AND correctly spares
  the conv-div control whose pitch is NOT O(delta) -- the generalisable, honestly
  bounded claim.

Panel (b)  THE COUPLED CURE (causal close).  Reattachment error e_reatt on the
  canonical h/L_x=1.0 hill for the single-variable coupled rungs
  Spalding -> TBLE -> CR-WM (aposteriori_crwm_twin.npz).  CR-WM restores
  within-layer convection with the CLOSURE UNTOUCHED; the drop TBLE->CR-WM is
  the measured causal effect of convection, the residual is the OUTER (above
  y_m) transport the 1-D ODE cannot see.  Pre-registered band shown.  If the
  CR-WM rung has not landed it is drawn as the pre-registered band only (no
  fabricated bar).

Colours (project convention): orange = ground truth/DNS, black = black-box/TBLE,
bluish-grey = gray-box/CR-WM, green = Spalding.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_l3_generalisation_map.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = [os.path.join(ROOT, "figures", "fig_l3_generalisation_map.pdf"),
        os.path.join(ROOT, "..", "manuscript", "figures", "fig_l3_generalisation_map.pdf")]

ORANGE = "#E69F00"
BLACK = "#000000"
GRAY = "#56708A"     # bluish-grey CR-WM
GREEN = "#1B7837"
RED = "#B2182B"

BAND = (0.134, 0.232)
P3_POINT = 0.170


def main():
    cg = np.load(os.path.join(RES, "cross_geometry_collapse.npz"), allow_pickle=True)
    dr = np.load(os.path.join(RES, "aposteriori_dose_response.npz"), allow_pickle=True)
    tw = np.load(os.path.join(RES, "aposteriori_crwm_twin.npz"), allow_pickle=True)

    fig, (axa, axc, axb) = plt.subplots(1, 3, figsize=(14.2, 4.3))

    # -------- panel (a): a-priori operating map (15 geometries) -------------
    eps = cg["eps_med"]; r2 = cg["r2"]; rep = cg["repeating"]; po = cg["pitch_O_delta"]
    keys = cg["keys"]
    for i in range(len(keys)):
        trig = (rep[i] == 1 and po[i] == 1)
        face = RED if trig else "none"
        edge = RED if trig else "0.45"
        axa.scatter(eps[i], r2[i], s=70, facecolors=face, edgecolors=edge,
                    linewidths=1.4, zorder=3)
    axa.axhline(0.88, color="0.5", ls=":", lw=1.0)
    axa.axhline(0.0, color="0.7", ls="-", lw=0.6)
    axa.axvline(1.0, color="0.5", ls="--", lw=1.0)
    axa.set_xscale("log")
    axa.set_yscale("symlog", linthresh=1.0)
    axa.text(1.15, -400, r"$\epsilon\sim O(1)$", rotation=90,
             fontsize=8, color="0.4", va="center")
    axa.text(3.0, 0.9, "$R^2=0.88$ success floor", fontsize=7, color="0.4", va="bottom")
    axa.set_xlabel(r"a-priori cancellation diagnostic  $\epsilon_{\rm med}$")
    axa.set_ylabel(r"a-priori wall-stress skill  $R^2(\tau_w)$  (symlog)")
    axa.set_title("(a) A-priori operating map\n($\\epsilon\\ll1$ over O($\\delta$)-pitch repeats "
                  "$\\Rightarrow$ catastrophe)", fontsize=9)
    rho = float(cg["spearman_rho"]); p = float(cg["spearman_p"]); n = int(cg["spearman_n"])
    axa.text(0.03, 0.04, f"Spearman $\\rho={rho:.2f}$\n$p={p:.1e}$, $n={n}$",
             transform=axa.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    axa.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor=RED,
               markeredgecolor=RED, ms=9, label="repeating, pitch $\\sim O(\\delta)$"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="0.45", ms=9, label="non-rep. / wide pitch"),
    ], fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.16))

    # -------- panel (c): coupled multi-geometry generalisation (G7) ---------
    labels = [str(s) for s in dr["labels"]]
    ceps = dr["eps_med"]; cerr = dr["e_reatt"]; ctl = dr["is_control"]
    for i in range(len(labels)):
        is_ctl = bool(ctl[i])
        face = "none" if is_ctl else GRAY
        edge = GREEN if is_ctl else GRAY
        axc.scatter(ceps[i], cerr[i], s=110, facecolors=face, edgecolors=edge,
                    linewidths=1.8, zorder=3)
        short = labels[i].replace(" (Re5600)", "").replace("hill ", "")
        dy = 0.012 if not is_ctl else -0.022
        axc.annotate(short, (ceps[i], cerr[i]), fontsize=6.8,
                     xytext=(0, 9 if dy > 0 else -12), textcoords="offset points",
                     ha="center", color=("0.2" if not is_ctl else GREEN))
    axc.axvline(1.0, color="0.5", ls="--", lw=1.0)
    axc.set_xscale("log")
    axc.set_xlabel(r"a-priori cancellation diagnostic  $\epsilon_{\rm med}$")
    axc.set_ylabel(r"coupled a-posteriori reattachment error  $e_{\rm reatt}$")
    axc.set_title("(c) Coupled generalisation (G7):\nfailure reproduces across the "
                  "repeating-hill family", fontsize=9)
    axc.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor=GRAY,
               markeredgecolor=GRAY, ms=9, label="repeating hill, $\\epsilon\\!<\\!1$ (fails)"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=GREEN, ms=9, label="conv-div control, $\\epsilon\\!\\sim\\!1.7$ (spared)"),
    ], fontsize=7.5, loc="upper right")
    axc.set_ylim(-0.02, 0.42)

    # -------- panel (b): coupled cure (single-variable twin) ----------------
    e_sp = float(tw["spalding_e_reatt"]); e_tb = float(tw["tble_e_reatt"])
    crwm_present = bool(tw["crwm_present"]) and np.isfinite(float(tw["crwm_e_reatt"]))
    e_cr = float(tw["crwm_e_reatt"]) if crwm_present else np.nan

    xs = [0, 1, 2]
    axb.bar(0, e_sp, width=0.6, color=GREEN, alpha=0.9, label="Spalding (drops conv.+dp/dx)")
    axb.bar(1, e_tb, width=0.6, color=BLACK, alpha=0.85, label="TBLE / ODE (drops convection)")
    # pre-registered band
    axb.axhspan(BAND[0], BAND[1], xmin=0.62, xmax=0.95, color=GRAY, alpha=0.18, zorder=0)
    if crwm_present:
        axb.bar(2, e_cr, width=0.6, color=GRAY, alpha=0.9,
                label=f"CR-WM (restores conv.): {e_cr:.3f}")
        impr = (e_tb - e_cr) / e_tb
        axb.annotate("", xy=(2, e_cr), xytext=(2, e_tb),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=2.0))
        axb.text(2.34, 0.5 * (e_tb + e_cr),
                 f"$-{100*impr:.0f}\\%$\n(convection,\nclosure fixed)",
                 fontsize=8, color=RED, va="center")
        in_band = BAND[0] <= e_cr <= BAND[1]
        axb.text(2, e_cr - 0.03, "in pre-reg.\nband" if in_band else "outside\nband",
                 ha="center", va="top", fontsize=7,
                 color=("green" if in_band else RED))
        ttl = f"(b) Coupled cure LANDED: convection removes {100*impr:.0f}% of\n" \
              f"the reattachment catastrophe (h/$L_x$=1.0 hill)"
    else:
        axb.text(2, 0.5 * (BAND[0] + BAND[1]), "CR-WM\npre-registered\nband\n[%.3f, %.3f]" % BAND,
                 ha="center", va="center", fontsize=8, color=GRAY)
        ttl = "(b) Coupled cure: pre-registered band (run not yet landed)"
    axb.axhline(0, color=ORANGE, ls="--", lw=1.8, label="DNS (perfect reattachment)")
    axb.set_xticks(xs)
    axb.set_xticklabels(["Spalding", "TBLE", "CR-WM"])
    axb.set_ylabel(r"coupled reattachment error  $e_{\rm reatt}$")
    axb.set_title(ttl, fontsize=9)
    axb.legend(fontsize=7.5, loc="upper left")
    axb.set_ylim(-0.03, max(0.45, e_tb * 1.15))

    fig.tight_layout()
    for out in FIGS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(os.path.abspath(out), dpi=200)
        print(f"[fig] wrote {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
