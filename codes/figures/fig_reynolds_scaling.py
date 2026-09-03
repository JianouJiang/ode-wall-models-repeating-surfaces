#!/usr/bin/env python3
"""
Thrust #19 L2 node figure — the Reynolds-scaling falsifier battery.
Reads ONLY development/nodes/node_003/reynolds_scaling.npz (real DNS, a-priori).
Four panels:
  (a) ASYMPTOTIC DEEPENING: eps_C(Re) over five DNS Reynolds numbers (Breuer
      700/1400/2800 + Krank 5600/10595) with bootstrap CIs; C_f(Re) on a twin axis.
  (b) BINDING CONDITION: |dp/dx| ratio per station + leave-one-out vs the viscous
      expectation (the denominator is outer-pinned, not viscous).
  (c) WALL-UNIT DECOMPOSITION: the tau_w deterioration is over-cancelled by the
      y_m-shrink -> the deepening is masked in the y_m+-fixed convention.
  (d) SPECIFICITY: separated stations deepen with Re; attached stations do not.

Run: OMP_NUM_THREADS=2 python3 codes/figures/fig_reynolds_scaling.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
NODE = os.path.join(ROOT, "development", "nodes", "node_003")
NPZ  = os.path.join(NODE, "reynolds_scaling.npz")
d = np.load(NPZ, allow_pickle=True)

C_DNS = "#d8732e"      # DNS / ground truth (orange, per project convention)
C_CF  = "#444444"      # skin friction
C_VISC = "#b03030"

fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))

# ---------------------------------------------------------------- (a) deepening
Re = d["Re_pts"]; eps = d["epsC_pts"]; lo = d["epsC_lo"]; hi = d["epsC_hi"]
cf = d["cf_pts"]; src = json.loads(str(d["series_src"]))
a = ax[0, 0]
yerr = np.vstack([eps - lo, hi - eps])
a.errorbar(Re, eps, yerr=yerr, fmt="o", color=C_DNS, ms=8, capsize=3,
           lw=1.5, label=r"$\varepsilon_C=|\tau_w|/|C|$  (closure-free)")
# power-law guide
s_re = float(d["slope_re"])
xx = np.array([Re.min(), Re.max()])
a.plot(xx, eps[0]*(xx/Re[0])**s_re, "--", color=C_DNS, lw=1,
       label=fr"$\varepsilon_C\sim Re^{{{s_re:.2f}}}$ ($R^2$={float(d['r2_re']):.2f})")
for r, e, s in zip(Re, eps, src):
    a.annotate(s.split()[0], (r, e), textcoords="offset points",
               xytext=(6, 6), fontsize=7, color="#666")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$Re_b$"); a.set_ylabel(r"$\varepsilon_C$  (residual / dropped convection)")
a.axhline(1.0, color="k", lw=0.6, ls=":")
a.set_title("(a) Cancellation DEEPENS with Re  (5 DNS, factor 15)", fontsize=10)
a2 = a.twinx()
a2.plot(Re, cf*1e3, "s-", color=C_CF, ms=6, lw=1.2, alpha=0.8,
        label=r"$C_f(Re)\times10^3$")
a2.set_ylabel(r"$C_f\times10^3$ (near-wall slope)", color=C_CF)
a2.tick_params(axis="y", labelcolor=C_CF)
a.legend(loc="lower left", fontsize=8)
a2.legend(loc="upper right", fontsize=8)

# -------------------------------------------------------------- (b) binding cond
b = ax[0, 1]
xs = d["xs_common"]; ratio = d["dpx_ratio"]; loo = d["dpx_loo"]
g = float(d["dpx_gmean"]); gw = float(d["dpx_gmean_weighted"])
Rr = float(d["Re_ratio"])
xi = np.arange(len(xs))
b.bar(xi, ratio, color=C_DNS, alpha=0.8, label=r"per-station $|\partial_x p|$ ratio")
b.axhline(g, color="k", lw=1.5, label=fr"geom-mean = {g:.2f}")
b.axhline(gw, color="k", lw=1.0, ls="--", label=fr"$|\partial_x p|$-weighted = {gw:.2f}")
b.axhline(Rr, color=C_VISC, lw=1.5, ls=":", label=fr"viscous would give {Rr:.2f}")
b.scatter(xi, loo, marker="x", color="navy", zorder=5, s=55,
          label="leave-one-out geom-mean")
b.set_xticks(xi); b.set_xticklabels([f"{x:g}" for x in xs])
b.set_xlabel(r"station $x/h$"); b.set_ylabel(r"$|\partial_x p|_{10595}/|\partial_x p|_{5600}$")
b.set_title("(b) Binding condition: denominator outer-pinned", fontsize=10)
b.legend(fontsize=7.5, loc="upper right")

# ----------------------------------------------------------- (c) wall-unit decomp
c = ax[1, 0]
pieces = [("$\\tau_w$\n(asymptotic\nsignal)", float(d["decomp_tau"]), C_CF),
          ("$|\\partial_x p|$\n(binding ~0)", float(d["decomp_dpx"]), "#888"),
          ("$y_m$-shift\n(grid effect)", float(d["decomp_ym"]), "#3a7d3a"),
          ("net\n$\\Delta\\ln\\varepsilon$", float(d["decomp_total"]), C_DNS)]
labels = [p[0] for p in pieces]; vals = [p[1] for p in pieces]; cols = [p[2] for p in pieces]
c.bar(range(4), vals, color=cols, alpha=0.85)
c.axhline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    c.annotate(f"{v:+.2f}", (i, v), textcoords="offset points",
               xytext=(0, 8 if v > 0 else -14), ha="center", fontsize=9)
c.set_xticks(range(4)); c.set_xticklabels(labels, fontsize=8)
c.set_ylabel(r"$\Delta\ln\varepsilon$  (5600$\to$10595)")
c.set_title("(c) Why $y_m^+$-fixed practice masks the deepening", fontsize=10)

# --------------------------------------------------------------- (d) specificity
e = ax[1, 1]
sep5, sep10 = float(d["prov_eps_outer_5600"]), float(d["prov_eps_outer_10595"])
att5, att10 = float(d["att_eps_5600"]), float(d["att_eps_10595"])
e.plot([5600, 10595], [sep5, sep10], "o-", color=C_DNS, ms=9, lw=2,
       label=f"separated (cancellation): {sep5:.2f}$\\to${sep10:.2f}  FALLS")
e.plot([5600, 10595], [att5, att10], "s--", color="#2a6fb0", ms=8, lw=2,
       label=f"attached (control): {att5:.2f}$\\to${att10:.2f}  rises")
e.set_xscale("log"); e.set_xticks([5600, 10595]); e.set_xticklabels(["5600", "10595"])
e.set_xlabel(r"$Re_b$"); e.set_ylabel(r"$\varepsilon$ (outer, $y_m=0.10H$)")
e.set_title("(d) Deepening is separation-specific (G7)", fontsize=10)
e.legend(fontsize=8, loc="center right")

fig.suptitle("Thrust #19 L2 — ODE wall-model failure is an ASYMPTOTIC (Reynolds) effect "
             "(a-priori, real DNS)", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(NODE, f"fig_reynolds_scaling.{ext}"), dpi=150,
                bbox_inches="tight")
print("traceability: all numbers read from", os.path.relpath(NPZ, ROOT))
print("eps_C(Re) =", np.round(eps, 3), "over Re =", Re.astype(int))
print("saved fig_reynolds_scaling.pdf/.png ->", os.path.relpath(NODE, ROOT))
