#!/usr/bin/env python3
r"""
fig_matching_height_band.py  --  node_003 (Thrust #16 L3) headline figure.

The matching-height band: the convection-blind ODE catastrophe is switched on/off
by WHERE the wall model matches.  Three panels:

 (a) COUPLED one-way sweep (Krank Re10595, 10 stations): R^2(tau_w) of the
     production ODE fed the true matching velocity at a range of heights.
     R^2 collapses monotonically as the match deepens into the cancellation band
     (eps -> 0).  The well-resolved coupled WMLES matches BELOW the band (marker).
     Source: codes/results/feedback_displacement.npz (sweep_*).

 (b) A-PRIORI closure-independent matching-height robustness (Xiao 512-station
     DNS): R^2(tau_w) vs matching-cell index for FIVE closures (4 algebraic/ML +
     exact DNS stress).  All collapse, and deepen monotonically with the match
     height -- the band is closure-independent (exact-DNS-stress is WORST).
     Source: codes/results/closure_conditioning_floor.npz (ym_robustness).

 (c) The DECISIVE a-posteriori twin (Xiao alpha=1.0 hill, Re5600): C_f(x) of two
     coupled WMLES identical in EVERYTHING except the near-wall grading -- the
     well-resolved run matches BELOW the band (y_m+~5), the band run matches IN
     the band (y_m+~25-35).  Reattachment vs DNS.  This is the deployed proof
     that matching-height placement, not closure, controls the failure.
     Source: codes/results/aposteriori_band_confirmation.npz (+ _xiao_1p0.npz).
     Rendered as "pending" until the band run completes.

Colors (project convention): orange = DNS truth; black = black-box ODE;
bluish-gray = gray-box; green = Spalding.  Closures in (b) use a sequential map.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
OUT = os.path.join(ROOT, "development", "nodes", "node_003")
os.makedirs(OUT, exist_ok=True)

ORANGE = "#E69F00"
BLACK = "#000000"
GRAYB = "#56708A"
GREEN = "#2C9D3A"
BAND = "#D55E00"

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
})

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))

# ---- (a) coupled one-way sweep --------------------------------------------- #
ax = axes[0]
d = np.load(os.path.join(RES, "feedback_displacement.npz"), allow_pickle=True)
ymp, r2, eps = d["sweep_ymp"], d["sweep_r2"], d["sweep_eps"]
ax.axhline(0, color="0.6", lw=0.8, ls=":")
# band shading: y_m+ >= 20 (eps <~ 0.2)
ax.axvspan(20, ymp.max() * 1.15, color=BAND, alpha=0.12, lw=0)
ax.text(0.97, 0.06, "cancellation\nband", transform=ax.transAxes,
        ha="right", va="bottom", color=BAND, fontsize=8.5)
ax.plot(ymp, r2, "-o", color=BLACK, mfc="white", ms=5, label=r"ODE one-way $R^2(\tau_w)$")
# well-resolved coupled operating height (median y_m+ ~3, two-way R2 ~+0.22; one-way +0.95)
ym_c = float(np.median(d["ymp"]))
ax.axvline(ym_c, color=GRAYB, lw=1.4, ls="--")
ax.annotate("well-resolved\nWMLES match\n" + rf"$y_m^+\!\approx\!{ym_c:.0f}$",
            xy=(ym_c, 0.5), xytext=(6.5, 0.62), color=GRAYB, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=GRAYB, lw=1.0))
ax.set_xscale("log")
ax.set_xlabel(r"matching height $y_m^+$")
ax.set_ylabel(r"$R^2(\tau_w)$")
ax.set_title("(a) coupled one-way sweep\n(Krank hill, Re$_H$=10595)")
# secondary eps annotations at a few heights
for xh, e in zip(ymp, eps):
    if xh in (1, 5, 20, 60):
        ax.text(xh, r2[list(ymp).index(xh)] + 0.18, rf"$\varepsilon$={e:.2f}",
                fontsize=7, ha="center", color="0.35")
ax.legend(loc="lower left", frameon=False)
ax.set_ylim(-4.6, 1.2)

# ---- (b) a-priori closure-independent matching-height robustness ----------- #
ax = axes[1]
d2 = np.load(os.path.join(RES, "closure_conditioning_floor.npz"), allow_pickle=True)
ymr = d2["ym_robustness"][0]  # dict: {Yidx: {closure: R2}}
labels = list(d2["closure_labels"])
keys = list(d2["closure_keys"])
yidx = sorted(ymr.keys())
cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(keys)))
for ci, (k, lab) in enumerate(zip(keys, labels)):
    r2s = [ymr[yy][k] for yy in yidx]
    is_dns = "dns" in k
    ax.plot(yidx, r2s, "-s" if is_dns else "-o",
            color=(ORANGE if is_dns else cmap[ci]),
            lw=2.4 if is_dns else 1.6, ms=6 if is_dns else 4,
            label=lab + (" (exact)" if is_dns else ""))
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_yscale("symlog")
ax.set_xlabel(r"matching-cell index $j_m$ (deeper $\rightarrow$)")
ax.set_ylabel(r"$R^2(\tau_w)$  (symlog)")
ax.set_title("(b) a-priori, closure-independent\n(Xiao 512-station DNS)")
ax.legend(loc="lower left", frameon=False, ncol=1)
ax.set_xticks(yidx)

# ---- (c) decisive a-posteriori twin ---------------------------------------- #
ax = axes[2]
band_path = os.path.join(RES, "aposteriori_band_confirmation.npz")
wr_path = os.path.join(RES, "aposteriori_dose_response_xiao_1p0.npz")
have_band = os.path.exists(band_path)
have_wr = os.path.exists(wr_path)
if have_band:
    b = np.load(band_path, allow_pickle=True)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.plot(b["x"], b["cf"], "-", color=BAND, lw=2.0,
            label=rf"band WMLES ($y_m^+\!\approx\!{float(b['ym_plus_avg']):.0f}$)")
    if np.isfinite(float(b["x_reatt_wmles"])):
        ax.axvline(float(b["x_reatt_wmles"]), color=BAND, ls="--", lw=1.2)
    if have_wr:
        w = np.load(wr_path, allow_pickle=True)
        ax.plot(w["wmles_x"], w["wmles_cf"], "-", color=GRAYB, lw=1.6,
                label=r"well-resolved WMLES ($y_m^+\!\approx\!5$)")
        if np.isfinite(float(w["wmles_x_reatt"])):
            ax.axvline(float(w["wmles_x_reatt"]), color=GRAYB, ls="--", lw=1.0)
    xrd = float(b["x_reatt_dns"])
    ax.axvline(xrd, color=ORANGE, ls="-", lw=1.6, label=rf"DNS reattach. $x$={xrd:.2f}")
    ax.set_xlabel(r"$x/h$")
    ax.set_ylabel(r"$C_f$ (bottom wall)")
    ax.set_title("(c) decisive a-posteriori twin\n(Xiao $\\alpha$=1.0 hill, Re$_H$=5600)")
    ax.legend(loc="upper right", frameon=False)
else:
    ax.text(0.5, 0.5, "band-confirmation\ncoupled WMLES\n(running — panel filled\non completion)",
            ha="center", va="center", transform=ax.transAxes, color=BAND, fontsize=10)
    ax.set_title("(c) decisive a-posteriori twin\n(Xiao $\\alpha$=1.0 hill, Re$_H$=5600)")
    ax.set_xlabel(r"$x/h$"); ax.set_ylabel(r"$C_f$ (bottom wall)")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_matching_height_band.{ext}"), bbox_inches="tight")
print("wrote fig_matching_height_band.{png,pdf} to", OUT,
      "| band panel:", "DATA" if have_band else "PENDING")
