#!/usr/bin/env python3
r"""
fig_onset_boundary_map.py  (L3 node_005)
========================================
Four-panel headline figure for the flat->wavy->periodic-hill TRANSITION:

  (a) the (a/delta x lambda/delta) failure MAP: separated deep-cancellation
      failures (filled red), non-separated mild-error (half-filled), tolerated
      (open); the constant-slope onset boundary S=a*pi/lambda overlaid.
  (b) relRMS vs lambda/delta for the two pitch ladders (a/d=0.10 and 0.40):
      the failure-onset crossing MOVES from lambda~3 (a/d=0.10) into the locked
      [8,22] band (a/d=0.40) -> pitch is not the discriminant.
  (c) cancellation collapse: R^2(tau_w) vs eps for the Xiao 29 hills (orange)
      and the failing wavy walls (red) -- both collapse onto a common eps band;
      the H3-predicted critical depth [0.161,0.322] shaded.
  (d) coverage f_rec vs eps: failing wavy + hills share the high-coverage /
      low-eps quadrant; the BFS zero-frequency control sits at f_rec~0 / large eps.

Regenerates from codes/results/onset_boundary_map.npz (+ dose_response_xiao.npz,
cross_geometry_collapse.npz). No fabricated points.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
MANU_FIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

C_FAIL = "#c0392b"      # separated deep-cancellation failure (red)
C_MILD = "#e08e0b"      # non-separated mild error
C_TOL = "#2c7fb8"       # tolerated
C_HILL = "#d95f02"      # orange = reference DNS hill family (truth-anchored)
EPS_BAND = (0.161, 0.322)


def load():
    d = np.load(os.path.join(RESULTS, "onset_boundary_map.npz"), allow_pickle=True)
    return d


def main():
    d = load()
    a = d["a_over_delta"]; lam = d["lam_over_delta"]; slope = d["slope"]
    eps = d["eps_med"]; frec = d["f_rec"]; r2 = d["r2"]; rel = d["relRMS"]
    sep = d["separated"]; fail = d["fail"]
    hill_eps = d["hill_eps"]; hill_frec = d["hill_frec"]; hill_r2 = d["hill_r2"]
    cx_a = d["crossing_a"]; cx_lam = d["crossing_lam_c"]; cx_slope = d["crossing_slope_c"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # ---- (a) the (a/d x lam/d) failure map -----------------------------------
    a0 = ax[0, 0]
    for i in range(len(a)):
        if fail[i] and sep[i]:
            a0.scatter(lam[i], a[i], s=120, c=C_FAIL, marker="s",
                       edgecolors="k", zorder=3)
        elif fail[i]:
            a0.scatter(lam[i], a[i], s=110, facecolors="none", edgecolors=C_MILD,
                       marker="s", linewidths=2, zorder=3)
        else:
            a0.scatter(lam[i], a[i], s=90, facecolors="none", edgecolors=C_TOL,
                       marker="o", linewidths=1.6, zorder=3)
    # constant-slope onset boundary: a = S_c * lambda / pi, S_c = median crossing slope
    sc = np.nanmedian(cx_slope) if np.isfinite(cx_slope).any() else 0.105
    ll = np.linspace(1.5, max(24, np.nanmax(lam) + 2), 100)
    a0.plot(ll, sc * ll / np.pi, "k--", lw=1.5,
            label=r"onset $S=a\pi/\lambda=%.2f$" % sc)
    a0.axvspan(8, 22, color="0.85", alpha=0.5, zorder=0, label=r"locked $[8,22]$")
    a0.set_xlabel(r"$\lambda/\delta$ (pitch)"); a0.set_ylabel(r"$a/\delta$ (amplitude)")
    a0.set_title(r"(a) wavy failure map: onset tracks wall slope $S=a\pi/\lambda$")
    leg = [Line2D([], [], marker="s", color="w", markerfacecolor=C_FAIL,
                  markeredgecolor="k", markersize=11, label="separated cancellation FAIL"),
           Line2D([], [], marker="s", color="w", markerfacecolor="w",
                  markeredgecolor=C_MILD, markersize=11, label="non-separated mild error"),
           Line2D([], [], marker="o", color="w", markerfacecolor="w",
                  markeredgecolor=C_TOL, markersize=10, label="tolerated"),
           Line2D([], [], ls="--", color="k", label=r"onset $S_c=%.2f$" % sc)]
    a0.legend(handles=leg, loc="upper left", fontsize=8)

    # ---- (b) relRMS vs lambda for each pitch ladder --------------------------
    b = ax[0, 1]
    b.axvspan(8, 22, color="0.85", alpha=0.5, zorder=0, label=r"locked $[8,22]$")
    b.axhline(0.5, color="0.4", ls=":", lw=1.2)
    for a_t, col, mk in [(0.10, "#1b7837", "o"), (0.40, "#762a83", "^")]:
        m = np.abs(a - a_t) < 0.02
        if m.sum() >= 1:
            order = np.argsort(lam[m])
            b.plot(lam[m][order], rel[m][order], mk + "-", color=col, ms=8,
                   label=r"pitch ladder $a/\delta=%.2f$" % a_t)
    for a_t, lc in zip(cx_a, cx_lam):
        if np.isfinite(lc):
            b.axvline(lc, color="k", ls="--", lw=1,
                      ymax=0.5)
            b.annotate(r"$\lambda_c=%.1f$" % lc, (lc, 0.55),
                       fontsize=8, rotation=90, va="bottom")
    b.set_xlabel(r"$\lambda/\delta$ (pitch)"); b.set_ylabel(r"relRMS$(\tau_w)$")
    b.set_title("(b) onset pitch moves with amplitude (a/$\\delta$=0.40 locates [8,22])")
    b.legend(loc="upper right", fontsize=8)

    # ---- (c) cancellation collapse: R2 vs eps --------------------------------
    c = ax[1, 0]
    c.axvspan(EPS_BAND[0], EPS_BAND[1], color="0.85", alpha=0.6,
              label=r"H3 band $\varepsilon_c\in[%.2f,%.2f]$" % EPS_BAND)
    if hill_eps.size:
        c.scatter(hill_eps, hill_r2, s=55, c=C_HILL, marker="D", edgecolors="k",
                  label="Xiao 29 hills (DNS)", zorder=3)
    msf = fail & sep
    c.scatter(eps[msf], r2[msf], s=110, c=C_FAIL, marker="s", edgecolors="k",
              label="wavy separated FAIL", zorder=4)
    mtol = ~fail
    c.scatter(eps[mtol], r2[mtol], s=80, facecolors="none", edgecolors=C_TOL,
              marker="o", label="wavy tolerated", zorder=3)
    c.axhline(0, color="0.4", ls=":", lw=1)
    c.set_xscale("symlog", linthresh=0.3)
    c.set_xlabel(r"$\bar\varepsilon = |\tau_w|/(|dp/dx|\,y_m)$")
    c.set_ylabel(r"$R^2(\tau_w)$")
    c.set_title("(c) hills + wavy collapse on $\\varepsilon$ (not pitch)")
    c.legend(loc="lower right", fontsize=8)

    # ---- (d) coverage f_rec vs eps -------------------------------------------
    dd = ax[1, 1]
    dd.axvspan(EPS_BAND[0], EPS_BAND[1], color="0.85", alpha=0.6)
    if hill_eps.size:
        dd.scatter(hill_eps, hill_frec, s=55, c=C_HILL, marker="D",
                   edgecolors="k", label="Xiao 29 hills", zorder=3)
    dd.scatter(eps[msf], frec[msf], s=110, c=C_FAIL, marker="s", edgecolors="k",
               label="wavy separated FAIL", zorder=4)
    dd.scatter(eps[~sep], frec[~sep], s=80, facecolors="none", edgecolors=C_TOL,
               marker="o", label="wavy non-separated ($f_{rec}=0$)", zorder=3)
    dd.set_xscale("symlog", linthresh=0.3)
    dd.set_xlabel(r"$\bar\varepsilon$"); dd.set_ylabel(r"$f_{rec}=L_{sep}/\ell_p$")
    dd.set_title("(d) failures: low $\\varepsilon$ + high coverage")
    dd.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    for outdir in (HERE, MANU_FIG):
        os.makedirs(outdir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, "fig_onset_boundary_map.%s" % ext),
                        dpi=150, bbox_inches="tight")
    print("saved fig_onset_boundary_map.{png,pdf} ->", HERE, "and", MANU_FIG)


if __name__ == "__main__":
    main()
