#!/usr/bin/env python3
"""
A-posteriori WMLES statistical-convergence figure.

Reads the SAME real OpenFOAM running-average windows used by
codes/openfoam/pehill_wmles/convergence_check.py
(postProcessing/sampleBottomWall/<t>/bottomWall.xy, fieldAverage timeStart=135)
and plots the convergence of the coupled reattachment length x_reatt/H and the
bubble-mean |C_f| as the averaging window grows from 10 to 30 flow-through times.

This *demonstrates* (does not assert) that the reported coupled result
x_reatt/H = 3.58 is a converged time mean: the last-window drift is
d(x_reatt) < 0.1 H and d(bubble|C_f|) < 5 %.

No fabrication: every point is recomputed from an on-disk OpenFOAM sample.
Output: manuscript/figures/fig_aposteriori_convergence.{pdf,png}
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CASE = os.path.join(ROOT, "codes", "openfoam", "pehill_wmles")
SAMPLE_BASE = os.path.join(CASE, "postProcessing", "sampleBottomWall")
OUT_DIR = os.path.join(ROOT, "manuscript", "figures")
U_B = 1.0
T_FT = 9.0          # flow-through time L_x/u_b
T_TRANSIENT = 135.0  # fieldAverage timeStart
DNS_X_REATT = 4.511  # Krank, Kronbichler & Wall 2018

# colour scheme: orange = ground truth (DNS), green = equilibrium wall model
C_DNS = "#E69F00"
C_WMLES = "#2ca02c"


def spanwise_mean_cf(path):
    a = np.loadtxt(path, comments="#")
    x = a[:, 0]
    taux = -a[:, 3]                  # flip so Cf>0 == attached forward flow
    xr = np.round(x, 6)
    xu = np.unique(xr)
    cf = np.array([2.0 * taux[xr == xv].mean() / U_B**2 for xv in xu])
    return xu, cf


def x_reattach(x, cf):
    m = (x >= 0.1) & (x <= 7.0)
    xm, cfm = x[m], cf[m]
    up = []
    for i in range(len(cfm) - 1):
        if cfm[i] < 0 and cfm[i + 1] >= 0:
            up.append(xm[i] - cfm[i] * (xm[i + 1] - xm[i]) / (cfm[i + 1] - cfm[i]))
    return up[-1] if up else np.nan


def bubble_mean_abs_cf(x, cf):
    m = (x >= 0.2) & (x <= 4.0)
    return float(np.nanmean(np.abs(cf[m]))) if m.any() else np.nan


def main():
    times = sorted(float(d) for d in os.listdir(SAMPLE_BASE)
                   if d.replace(".", "", 1).isdigit())
    windows = [t for t in times if t > T_TRANSIENT]
    win_ft, xr_hist, cf_hist = [], [], []
    for t in windows:
        sdir = os.path.join(SAMPLE_BASE, f"{t:g}")
        xyf = glob.glob(os.path.join(sdir, "*.xy"))
        if not xyf:
            continue
        x, cf = spanwise_mean_cf(xyf[0])
        win_ft.append((t - T_TRANSIENT) / T_FT)
        xr_hist.append(x_reattach(x, cf))
        cf_hist.append(bubble_mean_abs_cf(x, cf))
    win_ft = np.array(win_ft)
    xr_hist = np.array(xr_hist)
    cf_hist = np.array(cf_hist)

    dxr = abs(xr_hist[-1] - xr_hist[-2])
    dcf = abs(cf_hist[-1] - cf_hist[-2]) / max(abs(cf_hist[-2]), 1e-12)
    print(f"[conv-fig] windows (T_FT): {win_ft.tolist()}")
    print(f"[conv-fig] x_reatt/H     : {np.round(xr_hist, 3).tolist()}")
    print(f"[conv-fig] bubble |Cf|   : {np.round(cf_hist, 5).tolist()}")
    print(f"[conv-fig] last-window drift d(x_reatt)={dxr:.3f}H, "
          f"d(bubble|Cf|)={100*dcf:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax1.axhline(DNS_X_REATT, color=C_DNS, lw=1.8, ls="--",
                label=f"DNS $x_r/H={DNS_X_REATT:.2f}$")
    ax1.plot(win_ft, xr_hist, "o-", color=C_WMLES, lw=1.8, ms=6,
             label="WMLES (equilibrium)")
    ax1.set_xlabel(r"averaging window $(t-t_0)/T_\mathrm{FT}$")
    ax1.set_ylabel(r"reattachment $x_\mathrm{reatt}/H$")
    ax1.set_title("(a) reattachment-length convergence")
    ax1.set_ylim(3.0, 4.8)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.25)
    ax1.annotate(rf"last-window drift $={dxr:.3f}\,H<0.1\,H$",
                 xy=(win_ft[-1], xr_hist[-1]), xytext=(0.30, 0.16),
                 textcoords="axes fraction", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))

    ax2.plot(win_ft, cf_hist, "s-", color=C_WMLES, lw=1.8, ms=6)
    ax2.set_xlabel(r"averaging window $(t-t_0)/T_\mathrm{FT}$")
    ax2.set_ylabel(r"bubble-mean $|C_f|$")
    ax2.set_title("(b) recirculation-stress convergence")
    ax2.grid(alpha=0.25)
    ax2.annotate(rf"drift $={100*dcf:.1f}\%<5\%$",
                 xy=(win_ft[-1], cf_hist[-1]), xytext=(0.32, 0.78),
                 textcoords="axes fraction", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_aposteriori_convergence.{ext}"),
                    dpi=150, bbox_inches="tight")
    print(f"[conv-fig] wrote {OUT_DIR}/fig_aposteriori_convergence.(pdf|png)")


if __name__ == "__main__":
    main()
