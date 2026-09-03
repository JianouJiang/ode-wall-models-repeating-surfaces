#!/usr/bin/env python3
r"""
Figure: cross-geometry collapse of a-priori ODE wall-model error onto the
cancellation diagnostic.

Consumes  codes/results/cross_geometry_collapse.npz  (produced by
codes/analysis/cross_geometry_collapse.py) and renders two panels:

  (a) relRMS(tau_w) vs median epsilon-bar, log-log, one point per distinct
      wall geometry.  Demonstrates the NON-TAUTOLOGICAL cross-geometry ordering
      (Spearman rho = -0.77, p < 1e-3, n = 17) -- geometry is varied, not the
      matching height.

  (b) relRMS(tau_w) vs domain coverage fraction frac[eps<1], the manuscript's
      domain-wide-cancellation discriminant (Spearman rho = +0.81).  The
      repeating periodic-hills family sits at coverage = 1 (cancellation over
      the WHOLE domain) with catastrophic error; conv-div (wide pitch) and the
      single-feature separated flows sit at low coverage with usable error.

Colour key here is by GEOMETRY CLASS (this figure carries no model curves, so
the reserved model colours orange/black/bluish-gray/green are not used):
  repeating pitch~O(delta) = crimson;  repeating wide pitch = darkorange;
  single separated feature = steelblue;  attached PG boundary layer = grey.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "cross_geometry_collapse.npz")
OUT = os.path.join(os.path.dirname(CODES), "manuscript", "figures",
                   "fig_cross_geometry_collapse.pdf")

d = np.load(RES, allow_pickle=True)
keys = d["keys"]; klass = d["klass"]
eps = d["eps_med"]; frac1 = d["frac_eps_lt1"]; rr = d["relRMS"]
pitch = d["pitch_O_delta"]; repeating = d["repeating"]
rho = float(d["spearman_rho"]); p = float(d["spearman_p"])
rho_f1 = float(d["spearman_rho_frac1"]); p_f1 = float(d["spearman_p_frac1"])
tol = float(d["relrms_tol"])

# colour / marker by class
def style(k, rep, pit):
    if rep and pit:
        return "crimson", "o", "repeating, pitch $\\sim\\!\\mathcal{O}(\\delta)$"
    if rep and not pit:
        return "darkorange", "s", "repeating, wide pitch"
    if k == "attached":
        return "0.45", "^", "attached PG boundary layer"
    return "steelblue", "D", "single separated feature"

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))

# ----- panel (a): relRMS vs eps_med (log-log) -------------------------------
for i in range(len(keys)):
    c, m, _ = style(klass[i], bool(repeating[i]), bool(pitch[i]))
    ax[0].scatter(eps[i], rr[i], c=c, marker=m, s=46, edgecolor="k",
                  linewidth=0.4, zorder=3)
ax[0].axhline(tol, color="k", ls=":", lw=1.0, zorder=1)
ax[0].text(2e-3, tol * 1.4, "relRMS $=0.5$ (usability)", fontsize=7.5)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"median cancellation parameter $\bar\varepsilon$")
ax[0].set_ylabel(r"a-priori $\mathrm{relRMS}(\tau_w)$")
ax[0].set_title(f"(a) cross-geometry ordering\n"
                r"$\rho_s(\bar\varepsilon,\,\mathrm{relRMS})="
                f"{rho:+.2f}$, $p<10^{{-3}}$, $n={len(keys)}$", fontsize=9)
ax[0].grid(True, which="major", ls="-", lw=0.3, alpha=0.4)

# annotate periodic-hills cluster
ph = np.array([k.startswith("periodic_hills") for k in keys])
ax[0].annotate("periodic hills\n(domain-wide\ncancellation)",
               xy=(eps[ph].mean(), rr[ph].mean()),
               xytext=(3e-3, 30), fontsize=7.5, color="crimson",
               arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))

# ----- panel (b): relRMS vs coverage fraction -------------------------------
for i in range(len(keys)):
    c, m, _ = style(klass[i], bool(repeating[i]), bool(pitch[i]))
    ax[1].scatter(frac1[i], rr[i], c=c, marker=m, s=46, edgecolor="k",
                  linewidth=0.4, zorder=3)
ax[1].axhline(tol, color="k", ls=":", lw=1.0, zorder=1)
ax[1].set_yscale("log")
ax[1].set_xlabel(r"domain coverage fraction $f(\bar\varepsilon<1)$")
ax[1].set_ylabel(r"a-priori $\mathrm{relRMS}(\tau_w)$")
ax[1].set_title("(b) domain-wide-cancellation discriminant\n"
                r"$\rho_s(f,\,\mathrm{relRMS})="
                f"{rho_f1:+.2f}$, $p<10^{{-3}}$", fontsize=9)
ax[1].grid(True, which="major", ls="-", lw=0.3, alpha=0.4)
ax[1].set_xlim(-0.05, 1.08)

# shared legend (de-duplicated)
seen = {}
handles = []
for i in range(len(keys)):
    c, m, lab = style(klass[i], bool(repeating[i]), bool(pitch[i]))
    if lab not in seen:
        seen[lab] = True
        handles.append(Line2D([0], [0], marker=m, color="w",
                              markerfacecolor=c, markeredgecolor="k",
                              markersize=8, label=lab))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.06, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved -> {OUT}")
