#!/usr/bin/env python3
"""
Level-3 (Thrust #32) results figure.  All panels from real codes/results/*.npz,
using the files' ACTUAL keys.

  (a) A-priori benchmark   : R^2(tau_w) vs eps_median over the 15-geometry set;
                            the ODE succeeds (R^2>=0.88) off the repeating family
                            and fails catastrophically on the periodic hill.
  (b) Coupled a-posteriori : reattachment error e_reatt vs eps for the hill
                            steepness family (LANDED) + wide-pitch conv-div
                            control; the cross-pitch ordering is non-monotone
                            (honest), with the a-priori-derived predicted CR-WM
                            cure band shaded.
  (c) Cure ceiling         : separated-station R^2 -- exact DNS stress fails too;
                            within-layer convection (CR-WM) only partially closes
                            the gap (residual = outer transport above y_m).

Colours: orange=truth/reference, black=black-box ODE/TBLE, bluish-gray=gray-box,
green=Spalding.  Each panel is wrapped so one missing key cannot crash the figure.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES  = os.path.join(ROOT, "codes", "results")
OUT  = os.path.join(ROOT, "manuscript", "figures", "fig_l3_coupled_thrust32.pdf")

C_TRUTH = "#E69F00"; C_ODE = "#000000"; C_GRAY = "#56729B"
C_SPALD = "#1B9E77"; C_BAND = "#E69F00"

def load(stem):
    p = os.path.join(RES, stem + ".npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def fc(d, k): return np.array(d[k]).astype(float).ravel()
def f1(d, k): return float(np.array(d[k]).astype(float).ravel()[0])
def ss(d, k): return [str(x) for x in np.array(d[k]).ravel()]

fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.2))

# --------------------------------------------------------------- panel (a) ---
axa = ax[0]
try:
    g = load("cross_geometry_collapse")
    if g is None: raise FileNotFoundError("cross_geometry_collapse.npz")
    r2 = fc(g, "r2"); eps = fc(g, "eps_med")
    succ = r2 >= 0.88
    r2d = np.clip(r2, -100, 1.05)
    axa.scatter(eps[succ], r2d[succ], s=52, c=C_TRUTH, edgecolor="k", lw=0.5,
                zorder=3, label=f"ODE succeeds (n={int(succ.sum())})")
    axa.scatter(eps[~succ], r2d[~succ], s=72, marker="X", c=C_ODE, zorder=3,
                label=f"ODE catastrophic (n={int((~succ).sum())})")
    axa.axhline(0.88, ls="--", c=C_GRAY, lw=1.2)
    axa.set_xscale("log")
    axa.set_ylim(-105, 8)
    axa.set_xlabel(r"cancellation parameter $\varepsilon_{\mathrm{median}}$ (log)")
    axa.set_ylabel(r"$R^2(\tau_w)$ (clipped at $-100$)")
    axa.set_title("(a) A-priori benchmark (15 geometries)")
    axa.legend(loc="lower right", fontsize=8, framealpha=0.9)
except Exception as e:
    axa.text(0.5, 0.5, f"panel (a):\n{e}", ha="center", va="center",
             fontsize=8, transform=axa.transAxes)

# --------------------------------------------------------------- panel (b) ---
axb = ax[1]
try:
    d = load("aposteriori_dose_response")
    if d is None: raise FileNotFoundError("aposteriori_dose_response.npz")
    labels = ss(d, "labels"); eps = fc(d, "eps_med"); er = fc(d, "e_reatt")
    ctrl = fc(d, "is_control")
    hill = ctrl == 0
    # order hill points by eps for the connecting line
    hi = np.where(hill)[0]
    order = hi[np.argsort(eps[hi])]
    axb.plot(eps[order], er[order], "-o", color=C_ODE, ms=8, lw=1.4,
             label="coupled TBLE (hill family)")
    for i in np.where(ctrl == 1)[0]:
        axb.scatter([eps[i]], [er[i]], s=95, marker="s", c=C_GRAY, zorder=4,
                    label="conv-div control")
    # predicted CR-WM cure band from json
    jp = os.path.join(RES, "results_l3_thrust32.json")
    if os.path.exists(jp):
        cf = json.load(open(jp)).get("cure_forecast", {})
        band = cf.get("band")
        try:
            band = [float(b) for b in band]
            if all(np.isfinite(band)):
                axb.axhspan(band[0], band[1], color=C_BAND, alpha=0.20, zorder=1)
                axb.text(0.97, 0.62, "predicted CR-WM\ncure band (a-priori)",
                         transform=axb.transAxes, ha="right", va="center",
                         fontsize=7.5, color="#8a5a00")
        except Exception:
            pass
    axb.set_xscale("log")
    axb.set_ylim(0, 0.45)
    axb.set_xlabel(r"a-priori $\varepsilon_{\mathrm{median}}$ (log)")
    axb.set_ylabel(r"reattachment error $e_{\mathrm{reatt}}$")
    axb.set_title("(b) Coupled dose-response — non-monotone (honest)")
    axb.legend(loc="lower right", fontsize=8, framealpha=0.9)
except Exception as e:
    axb.text(0.5, 0.5, f"panel (b):\n{e}", ha="center", va="center",
             fontsize=8, transform=axb.transAxes)

# --------------------------------------------------------------- panel (c) ---
axc = ax[2]
try:
    d = load("crwm_apriori")
    if d is None: raise FileNotFoundError("crwm_apriori.npz")
    r2_ode  = f1(d, "r2_sep_ode")
    r2_ex   = f1(d, "r2_sep_exact")
    r2_lin  = f1(d, "r2_sep_linear")
    vals = [r2_ode, r2_ex, r2_lin]
    names = ["ODE\n(no conv.)", "exact DNS\nstress", "CR-WM\n(+within-layer\nconv.)"]
    axc.bar(names, vals, color=[C_ODE, C_TRUTH, C_GRAY], edgecolor="k")
    axc.axhline(0, color="k", lw=0.8)
    # value label inside the top of each bar (white)
    for i, v in enumerate(vals):
        axc.text(i, v * 0.06, f"{v:.1f}", ha="center", va="top",
                 color="white", fontsize=9, fontweight="bold")
    try:
        gap = f1(d, "gap_closed_linear")
        axc.text(0.5, 0.04, f"exact-DNS stress fails too; CR-WM closes only "
                 f"~{gap*100:.0f}% of the gap\n=> structural, not closure",
                 transform=axc.transAxes, ha="center", va="bottom",
                 color=C_SPALD, fontsize=8)
    except Exception:
        pass
    axc.set_ylim(min(vals) * 1.12, 1.5)
    axc.set_ylabel(r"separated-station $R^2(\tau_w)$")
    axc.set_title("(c) Cure ceiling: structure, not closure")
except Exception as e:
    axc.text(0.5, 0.5, f"panel (c):\n{e}", ha="center", va="center",
             fontsize=8, transform=axc.transAxes)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print("wrote", OUT)
