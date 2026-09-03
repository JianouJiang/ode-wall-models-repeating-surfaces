#!/usr/bin/env python3
r"""
Manuscript figure: the closure--structure compensation is regime-specific, and
the exact-stress error is the independently-integrated dropped convection.

Two panels, both a-priori, all numbers traced to on-disk npz:

  (a) Skill (relRMS of tau_w) of the ODE vs the cancellation diagnostic
      eps_med, across the 9 multi-station high-fidelity geometries of the
      P3 cross-geometry control (error_decomposition_cross_geometry.npz).
      OPEN markers = standard mixing-length closure; FILLED = exact-DNS-stress
      closure.  The relRMS=1 line splits "useful" from "useless".  Feeding the
      ODE the EXACT near-wall stress is catastrophic ONLY on the eps<<1
      repeating-hill geometry (relRMS 6.8 -> 29.8); on every eps>>1 geometry
      (8/8) it stays useful (relRMS_dns < 1, max 0.90).  So the closure error
      and the dropped convection compensate everywhere, but that compensation
      is OUTCOME-DECISIVE only in the deep-cancellation regime.

  (b) The independent convection cross-check (P4) on the 233 separated hill
      stations (error_decomposition.npz): the exact-stress structural error
      E_struct vs the convective integral C computed INDEPENDENTLY from the 2-D
      momentum budget (C never enters the ODE).  corr = +0.68
      (95% CI [+0.64,+0.74], Spearman +0.81, rank-robust) confirms the
      exact-stress error IS the dropped convection, not a numerical artefact.

CI / correlation annotations are read from compensation_l3_stats.npz
(produced by codes/analysis/compensation_l3_stats.py).

Colour key (this figure carries closure CONTROLS, not the reserved model roles
orange=truth/black=black-box/bluish-gray=gray-box/green=Spalding, so it uses a
neutral closure palette and says so):
  standard mixing-length closure = open black markers;
  exact-DNS-stress closure       = filled, coloured by geometry class
    (repeating O(delta) = crimson; repeating wide pitch = darkorange;
     single separated feature = steelblue).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
OUT = os.path.join(os.path.dirname(CODES), "manuscript", "figures",
                   "fig_compensation_regime.pdf")

cg = np.load(os.path.join(RES, "error_decomposition_cross_geometry.npz"),
             allow_pickle=True)
ed = np.load(os.path.join(RES, "error_decomposition.npz"), allow_pickle=True)
st = np.load(os.path.join(RES, "compensation_l3_stats.npz"), allow_pickle=True)

keys = [str(k) for k in cg["keys"]]
klass = [str(k) for k in cg["klass"]]
eps = cg["geom_eps_med"].astype(float)
rr_ml = cg["geom_relrms_ml"].astype(float)
rr_dns = cg["geom_relrms_dns"].astype(float)
r2_ml = cg["geom_r2_ml"].astype(float)

CLASS_COLOR = {"repeating_Odelta": "crimson",
               "repeating_wide": "darkorange",
               "single_feature": "steelblue"}
CLASS_MARK = {"repeating_Odelta": "*",
              "repeating_wide": "D",
              "single_feature": "o"}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                     "axes.titlesize": 10, "legend.fontsize": 7.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5))

# ===================== panel (a): regime-specific skill ====================
ax1.axhline(1.0, color="0.45", ls="--", lw=1.0, zorder=1)
ax1.text(2e3, 1.18, "useless", color="0.45", fontsize=7, ha="right")
ax1.text(2e3, 0.62, "useful", color="0.45", fontsize=7, ha="right")
ax1.axvspan(0.0, 1.0, color="crimson", alpha=0.06, zorder=0)
ax1.text(0.13, 55, r"$\bar\varepsilon\!<\!1$" + "\ncancellation",
         color="crimson", fontsize=7, ha="center")

for i, k in enumerate(keys):
    c = CLASS_COLOR[klass[i]]
    mk = CLASS_MARK[klass[i]]
    sz = 165 if mk == "*" else 55
    # standard mixing-length closure: OPEN
    ax1.scatter(eps[i], rr_ml[i], marker=mk, s=sz, facecolors="none",
                edgecolors="black", linewidths=1.1, zorder=4)
    # exact-DNS-stress closure: FILLED, class-coloured
    ax1.scatter(eps[i], rr_dns[i], marker=mk, s=sz, facecolors=c,
                edgecolors="black", linewidths=0.6, alpha=0.95, zorder=5)
    # connect the two closures for the same geometry
    ax1.plot([eps[i], eps[i]], [rr_ml[i], rr_dns[i]], color=c, lw=0.8,
             alpha=0.55, zorder=2)

# annotate the hills jump
ih = klass.index("repeating_Odelta")
ax1.annotate("exact stress\n$6.8\\!\\to\\!29.8$",
             xy=(eps[ih], rr_dns[ih]), xytext=(0.22, 9.5),
             fontsize=7.5, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"cancellation diagnostic  $\bar\varepsilon_{\rm med}=|\tau_w|/(|{\rm d}p/{\rm d}x|\,y_m)$")
ax1.set_ylabel(r"$\mathrm{relRMS}(\tau_w)$")
ax1.set_title("(a) exact-stress skill is regime-specific")
ax1.set_ylim(3e-3, 1e2)

leg_class = [Line2D([], [], marker="*", ls="none", mfc="crimson",
                    mec="black", ms=12, label="repeating, $O(\\delta)$ pitch (hills)"),
             Line2D([], [], marker="D", ls="none", mfc="darkorange",
                    mec="black", ms=6, label="repeating, wide pitch"),
             Line2D([], [], marker="o", ls="none", mfc="steelblue",
                    mec="black", ms=7, label="single separated feature")]
leg_clos = [Line2D([], [], marker="o", ls="none", mfc="none", mec="black",
                   ms=7, label="mixing-length closure"),
            Line2D([], [], marker="o", ls="none", mfc="0.4", mec="black",
                   ms=7, label="exact DNS stress")]
l1 = ax1.legend(handles=leg_class, loc="lower right", framealpha=0.9,
                handletextpad=0.3)
ax1.add_artist(l1)
ax1.legend(handles=leg_clos, loc="upper right", framealpha=0.9,
           handletextpad=0.3)

# ===================== panel (b): P4 independent convection ================
m = ed["sep_mask_with_C"]
Es = ed["E_struct"][m].astype(float)
C = ed["C"][m].astype(float)
r = float(st["p4_r_struct_C"])
ci = st["p4_r_struct_C_ci"]
rho = float(st["p4_rho_struct_C"])

ax2.scatter(C, Es, s=16, color="steelblue", alpha=0.55, edgecolors="none",
            zorder=3)
# 1:1 reference (E_struct = C would be exact)
lim = np.array([min(C.min(), Es.min()), max(C.max(), Es.max())])
ax2.plot(lim, lim, color="0.6", ls=":", lw=1.0, zorder=1, label="$E_{\\rm struct}=C$")
# least-squares guide
A = np.polyfit(C, Es, 1)
xs = np.linspace(C.min(), C.max(), 50)
ax2.plot(xs, np.polyval(A, xs), color="crimson", lw=1.3, zorder=2,
         label="LS fit")
ax2.axhline(0, color="0.85", lw=0.7, zorder=0)
ax2.axvline(0, color="0.85", lw=0.7, zorder=0)
ax2.set_xlabel(r"independent convective integral  $C$")
ax2.set_ylabel(r"exact-stress structural error $E_{\rm struct}$")
ax2.set_title("(b) the exact-stress error IS the dropped convection")
ax2.text(0.04, 0.95,
         f"$r={r:+.2f}$  (95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}])\n"
         f"Spearman $\\rho={rho:+.2f}$  ($n=233$ hill stations)",
         transform=ax2.transAxes, va="top", ha="left", fontsize=7.5,
         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", OUT)
