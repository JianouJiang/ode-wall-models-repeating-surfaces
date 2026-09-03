#!/usr/bin/env python3
r"""
Figure: the FORM-DRAG NEGATIVE CONTROL.  Does the streamwise form-drag fraction
phi_FD order the a-priori ODE failure over repeating structures?  Answer: NO --
the catastrophic failure is drag-partition / roughness INDEPENDENT.  This is the
data answer to the hardest open referee objection (G6): "isn't the sharp-rib
failure just form drag / roughness?"

Consumes  codes/results/formdrag_negative_control.npz  (produced by
codes/analysis/formdrag_negative_control.py) and renders two panels:

  (a) phi_FD vs a-priori ODE skill R2(tau_w) for the 10-case smooth+sharp family.
      The two CATASTROPHIC and the two TOLERATED outcomes OVERLAP across nearly
      the whole phi_FD axis: at phi_FD~0.93 the sharp ladder FAILS while the
      k-type rib and blade are TOLERATED (Delta phi_FD as small as 0.002, yet
      opposite verdict), and at phi_FD=0 the smooth hill is the DEEPEST failure
      while the smooth krank / conv-div are TOLERATED.  No phi_FD threshold
      separates the classes -- at matched form drag the verdict is set by the
      cancellation depth (eps, coverage), not by phi_FD.

  (b) class-separation AUC (catastrophic R2<0 vs tolerated R2>0) with bootstrap
      95% CI and exact-permutation p for the champion's a-priori discriminants
      (eps, deep-cancellation coverage f(eps<0.1)) versus phi_FD.  eps and
      coverage separate the classes at p<0.01; phi_FD sits at chance (CI brackets
      0.5, p=0.23).  The discriminant is the cancellation depth, NOT the drag
      partition.

This figure carries no model curves, so the reserved model colours
(orange/black/bluish-gray/green) are not used.  Colour key: catastrophic
failure = crimson; tolerated = steelblue.  Marker: smooth = circle, sharp = square.
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
NPZ = os.path.join(RESULTS, "formdrag_negative_control.npz")

CRIMSON = "#c0392b"      # catastrophic (R2 < 0)
STEEL = "#2c6fb0"        # tolerated (R2 > 0)

d = np.load(NPZ, allow_pickle=True)
keys = [str(k) for k in d["keys"]]
phi = d["phi_FD"]
r2 = d["r2"]
cov = d["coverage"]
eps = d["eps"]
cat = d["catastrophic"].astype(bool)
sharp = d["is_sharp"].astype(bool)

disc_names = [str(x) for x in d["disc_names"]]
auc = d["disc_auc"]
perm_p = d["disc_perm_p"]
ci_lo = d["disc_ci_lo"]
ci_hi = d["disc_ci_hi"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0))

# ----------------------------------------------------------------- panel (a)
# symlog y so the catastrophic R2 (down to -48) and the tolerated R2 (~+0.9)
# both read; threshold R2=0 marked.
for i, k in enumerate(keys):
    col = CRIMSON if cat[i] else STEEL
    mk = "s" if sharp[i] else "o"
    axA.scatter(phi[i], r2[i], s=95, c=col, marker=mk, edgecolors="k",
                linewidths=0.6, zorder=3)
axA.axhline(0.0, color="0.4", ls="--", lw=1.0, zorder=1)
axA.set_yscale("symlog", linthresh=1.0)
axA.set_xlim(-0.06, 1.06)
axA.set_xlabel(r"form-drag fraction  $\phi_{\mathrm{FD}}=|F_{p,x}|/(|F_{p,x}|+|F_{v,x}|)$")
axA.set_ylabel(r"a-priori ODE skill  $R^2(\tau_w)$")
axA.set_ylim(-110, 16)
axA.set_title("(a)  form drag does NOT order the failure", fontsize=10)

# --- the corrected, STRONGER falsifier: the classes OVERLAP at matched phi_FD ---
# iso-phi_FD pair at phi_FD~0.93: tolerated k-type rib SITS BETWEEN catastrophic
# ladder cases (rib 0.931 between ladder 0.929 and 0.968) -> Delta phi_FD=0.002,
# opposite verdict.  Mark the overlap with a vertical span and label both clusters.
ki = keys.index("rib_rans_ktype")              # tolerated, phi_FD=0.931
li0 = keys.index("ladder_rk0")                 # catastrophic, phi_FD=0.929 (matched)
axA.axvspan(0.918, 0.972, color="0.85", zorder=0)   # the high-phi_FD overlap band
axA.annotate("$\\phi_{FD}\\!\\approx\\!0.93$ overlap:\nrib/blade TOLERATED",
             (phi[ki], r2[ki]), xytext=(0.40, 7.0), fontsize=7.2, color=STEEL,
             ha="left", arrowprops=dict(arrowstyle="->", color=STEEL, lw=0.7))
axA.annotate("ladder CATASTROPHIC\n($\\Delta\\phi_{FD}{=}0.002$, opposite verdict)",
             (phi[li0], r2[li0]), xytext=(0.30, -62), fontsize=7.2, color=CRIMSON,
             ha="left", arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.7))
# iso-phi_FD pair at phi_FD=0: smooth hill (deepest fail) vs smooth tolerated
hi = keys.index("periodic_hills_1p0")
kr = keys.index("krank_pehill_Re10595")
axA.annotate("$\\phi_{FD}{=}0$ overlap:\nhill DEEPEST fail, smooth TOLERATED",
             (phi[hi], r2[hi]), xytext=(0.04, -25), fontsize=7.2, color="0.25",
             ha="left", arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.7))
axA.annotate("", (phi[kr], r2[kr]), xytext=(0.135, -18.5),
             arrowprops=dict(arrowstyle="->", color=STEEL, lw=0.7))

legA = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="k", markersize=8, label="smooth wall"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.5",
               markeredgecolor="k", markersize=8, label="sharp element"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CRIMSON,
               markeredgecolor="k", markersize=8, label="catastrophic ($R^2<0$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STEEL,
               markeredgecolor="k", markersize=8, label="tolerated ($R^2>0$)")]
axA.legend(handles=legA, fontsize=6.6, loc="center right", framealpha=0.9,
           handletextpad=0.3, ncol=1)

# ----------------------------------------------------------------- panel (b)
# AUC forest: champion discriminants vs phi_FD, with bootstrap CI + exact p.
pretty = {"eps": r"$\varepsilon$  (champion)",
          "coverage": r"coverage $f(\varepsilon{<}0.1)$  (champion)",
          "phi_FD": r"$\phi_{\mathrm{FD}}$  (form drag)"}
order = ["eps", "coverage", "phi_FD"]
ypos = np.arange(len(order))[::-1]
for y, name in zip(ypos, order):
    j = disc_names.index(name)
    col = STEEL if perm_p[j] < 0.05 else CRIMSON
    axB.errorbar(auc[j], y, xerr=[[auc[j] - ci_lo[j]], [ci_hi[j] - auc[j]]],
                 fmt="o", color=col, ecolor=col, capsize=4, ms=8, lw=1.6, zorder=3)
    tag = f"AUC={auc[j]:.2f}\n$p$={perm_p[j]:.3f}"
    axB.text(auc[j], y + 0.18, tag, ha="center", va="bottom", fontsize=7.3, color=col)
axB.axvline(0.5, color="0.4", ls="--", lw=1.0, zorder=1)
axB.text(0.5, len(order) - 0.55, "chance", color="0.4", fontsize=7.5,
         ha="center", rotation=90, va="top")
axB.set_yticks(ypos)
axB.set_yticklabels([pretty[k] for k in order], fontsize=8.5)
axB.set_ylim(-0.5, len(order) - 0.2)
axB.set_xlim(0.18, 1.07)
axB.set_xlabel("class-separation AUC  (catastrophic vs tolerated)\n"
               "exact 210-permutation $p$;  bars = stratified bootstrap 95% CI")
axB.set_title("(b)  the discriminant is cancellation depth, not drag partition",
              fontsize=10)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"fig_formdrag_negative_control.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
# manuscript copy
MAN = os.path.join(os.path.dirname(CODES), "manuscript", "figures")
if os.path.isdir(MAN):
    for ext in ("pdf", "png"):
        out = os.path.join(MAN, f"fig_formdrag_negative_control.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
