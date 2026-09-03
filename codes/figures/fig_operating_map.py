#!/usr/bin/env python3
r"""
fig_operating_map.py  --  Thrust #28 L3 headline figure: THE WALL-MODEL OPERATING MAP.

Four panels, all from real DNS/LES reference fields via the common Sec.6 pipeline
(operating_map_dataset.npz) and its L3 inferential layer (operating_map_l3.npz):

 (a) THE MAP.  R^2(tau_w) of the convection-blind ODE vs matching height y_m+,
     one column per geometry.  The O(delta)-pitch repeating hills (warm) plunge
     through the ignition line R^2=0 at small y_m+ = y_crit+; the wide-pitch +
     single-feature controls (bluish-gray) stay safe across the whole deployed
     range.  Deployed coupled WMLES heights marked on the axis (all sit low).

 (b) CLASS SEPARATION (F6).  The ignition height y_crit+ per geometry on a log
     axis: every O(delta) hill ignites below 16, every control above 147 --
     rank-AUC = 1.0, multiplicative margin 9.3x [bootstrap 95% CI].

 (c) DEPLOYMENT PHASE BOUNDARY (F5).  Crossover Reynolds number
     Re_tau* = y_crit+/r_g vs near-wall grid r_g = Dy_1/delta (DERIVED-LAW
     EXTRAPOLATION).  On a coarse grid the O(delta) hills cross into the
     catastrophe (shaded) at moderate engineering Re, while controls never do in
     any realistic range -- invisible at the cheap DNS-Re grids used to validate.

 (d) THE INFORMATIVE AXIS (F2) vs THE HONEST NEGATIVE (F3).  Within one geometry
     the error follows a measured power law relRMS = A eps^(-b), b=0.44 [95% CI];
     across geometries the static sort is uninformative (inset: eps vs coupled
     e_reatt, exact n=5 permutation p~1) -- so the decisive test is the
     within-geometry band traverse (F4, live/unlanded; NO number shown).

Colour convention: the O(delta)-pitch FAILURE class in a warm (orange/red)
sequence; the SAFE controls in bluish-gray; DNS/ignition reference in orange.
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
GRAYB = "#56708A"
GREEN = "#2C9D3A"
BAND = "#D55E00"

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10.5,
    "legend.fontsize": 7.6, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.linewidth": 0.9, "lines.linewidth": 1.7,
})

d = np.load(os.path.join(RES, "operating_map_dataset.npz"), allow_pickle=True)
L3 = np.load(os.path.join(RES, "operating_map_l3.npz"), allow_pickle=True)
keys = [str(s) for s in d["keys"]]
klass = [str(s) for s in d["klass"]]
ymp = d["ymp_grid"].astype(float)
ycrit = d["ycrit"].astype(float)
ceps = d["c_eps_plus"].astype(float)

# display names + per-geometry colour
NAME = {
    "krank_pehill_Re10595": r"Krank hill, Re$_H$=10595",
    "xiao_a0p8": r"Xiao $\alpha$=0.75", "xiao_a1p0": r"Xiao $\alpha$=1.0",
    "xiao_a1p2": r"Xiao $\alpha$=1.25", "periodic_hills_1p0": r"periodic hill (res.)",
    "conv_div_channel": "conv-div (wide pitch)", "curved_bfs_LES": "curved BFS",
    "nasa_hump": "NASA hump",
}
HILLS = [k for k in keys if klass[keys.index(k)] == "repeating_Odelta"]
CTRL = [k for k in keys if klass[keys.index(k)] != "repeating_Odelta"]
warm = plt.cm.autumn(np.linspace(0.0, 0.62, len(HILLS)))
cool = [GRAYB, "#7FA0B8", GREEN]
COL = {k: warm[i] for i, k in enumerate(HILLS)}
for i, k in enumerate(CTRL):
    COL[k] = cool[i % len(cool)]

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))

# =================== (a) THE MAP: R^2(y_m+) columns ========================== #
ax = axes[0, 0]
ax.axhspan(-50, 0, color=BAND, alpha=0.07, lw=0)
ax.axhline(0, color="0.4", lw=1.0, ls="--")
ax.text(1.05, -2.0, "ignition: $R^2<0$\n(catastrophe)", color=BAND, fontsize=8, va="top")
for k in HILLS + CTRL:
    r2 = d[f"sweep_r2__{k}"].astype(float)
    m = np.isfinite(r2)
    is_hill = k in HILLS
    ax.plot(ymp[m], r2[m], color=COL[k], lw=2.0 if is_hill else 1.5,
            ls="-" if is_hill else "--",
            label=NAME[k], zorder=3 if is_hill else 2)
    yc = ycrit[keys.index(k)]
    if np.isfinite(yc) and yc <= ymp.max():
        ax.plot([yc], [0], "v", color=COL[k], ms=6, mec="k", mew=0.4, zorder=5)
# deployed coupled y_m+ markers (a-posteriori), along the bottom
ymp5 = d["ymp5"].astype(float)
ax.scatter(ymp5, np.full_like(ymp5, -47), marker="D", s=34, color="k",
           zorder=6, label=r"deployed WMLES $y_m^+$")
ax.set_xscale("log")
ax.set_ylim(-50, 1.2)
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xlabel(r"matching height $y_m^+$")
ax.set_ylabel(r"$R^2(\tau_w)$  (symlog)")
ax.set_title(r"(a) the operating map: ODE $R^2(\tau_w)$ vs $y_m^+$")
ax.legend(loc="lower left", frameon=False, ncol=1, fontsize=6.8)

# =================== (b) class separation (F6) ============================== #
ax = axes[0, 1]
auc = float(L3["auc_ycrit"]); margin = float(L3["margin_ycrit"])
mlo, mhi = L3["margin_ycrit_ci"]
ax.axvspan(0.8, 16.5, color=BAND, alpha=0.08, lw=0)
ax.axvspan(140, 1e4, color=GRAYB, alpha=0.07, lw=0)
# lollipop layout: one geometry per row, names as y-tick labels (no overlap)
order = HILLS + CTRL                         # hills on top, controls below
ypos = {k: len(order) - i for i, k in enumerate(order)}
XMIN = 0.45
for k in order:
    yc = ycrit[keys.index(k)]
    yp = ypos[k]
    if not np.isfinite(yc):                  # NASA hump: y_crit+=inf (relrms criterion)
        ax.annotate("", xy=(9000, yp), xytext=(3500, yp),
                    arrowprops=dict(arrowstyle="->", color=COL[k], lw=1.8))
        ax.text(3200, yp, r"$y_{\rm crit}^+\!=\!\infty$", ha="right", va="center",
                fontsize=7.6, color=COL[k])
        continue
    ax.plot([XMIN, yc], [yp, yp], "-", color=COL[k], lw=1.4, alpha=0.6, zorder=2)
    ax.plot([yc], [yp], "o", color=COL[k], ms=11, mec="k", mew=0.6, zorder=4)
    ax.text(yc * 1.35, yp, f"{yc:.1f}", ha="left", va="center", fontsize=8,
            color=COL[k], fontweight="bold")
ax.axvline(16.0, color="0.3", lw=1.0, ls=":")
ax.set_xscale("log")
ax.set_xlim(XMIN, 1.2e4)
ax.set_ylim(0.3, len(order) + 0.7)
ax.set_yticks([ypos[k] for k in order])
ax.set_yticklabels([NAME[k] for k in order], fontsize=8)
for tick, k in zip(ax.get_yticklabels(), order):
    tick.set_color(COL[k])
ax.set_xlabel(r"ignition height $y_{\rm crit}^+$  (geometry-readable, a priori)")
ax.set_title("(b) class separation (F6)")
ax.text(0.97, 0.06,
        f"rank-AUC = {auc:.2f}\nmargin = {margin:.1f}$\\times$  (95% CI [{mlo:.0f}, {mhi:.0f}])",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
ax.text(0.06, 0.97, r"$O(\delta)$-pitch hills (fail)", color=BAND, fontsize=8.2,
        transform=ax.transAxes, va="top")
ax.text(0.97, 0.40, "controls (safe)", color=GRAYB, fontsize=8.2,
        transform=ax.transAxes, ha="right")

# =================== (c) Re_tau* deployment phase boundary (F5) ============== #
ax = axes[1, 0]
r_g = d["r_g_grid"].astype(float)
re_star = d["re_tau_star"].astype(float)
re_dep = float(d["re_tau_deployed"])
rg_fine = np.geomspace(r_g.min() * 0.9, r_g.max() * 1.1, 50)
ax.axhspan(1, re_dep, color="0.85", alpha=0.5, lw=0)
for k in HILLS + CTRL:
    yc = ycrit[keys.index(k)]
    if not np.isfinite(yc):
        continue
    is_hill = k in HILLS
    ax.plot(rg_fine, yc / rg_fine, "-" if is_hill else "--",
            color=COL[k], lw=2.0 if is_hill else 1.4, label=NAME[k])
ax.axhline(re_dep, color="k", lw=1.0, ls=":")
ax.text(r_g.max() * 0.5, re_dep * 1.15, f"DNS-Re validations (Re$_\\tau\\approx${re_dep:.0f})",
        fontsize=7.6, color="k")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"near-wall grid  $r_g=\Delta y_1/\delta$")
ax.set_ylabel(r"crossover  Re$_\tau^*=y_{\rm crit}^+/r_g$")
ax.set_title(r"(c) deployment phase boundary (F5, extrapolation)")
ax.legend(loc="upper right", frameon=False, fontsize=6.8, ncol=1)
ax.text(0.04, 0.05,
        r"$O(\delta)$ hills enter catastrophe" + "\nat moderate Re on coarse grids",
        transform=ax.transAxes, fontsize=7.8, color=BAND, va="bottom")

# =================== (d) informative axis (F2) vs honest negative (F3) ======= #
ax = axes[1, 1]
pr = np.load(os.path.join(RES, "epsilon_coupled_predictor.npz"), allow_pickle=True)
se = pr["sweep_eps"].astype(float); sr = pr["sweep_relrms"].astype(float)
ok = np.isfinite(se) & np.isfinite(sr) & (se > 0) & (sr > 0)
se, sr = se[ok], sr[ok]
b = float(L3["F2_b_point"]); blo, bhi = L3["F2_b_ci"]
A = float(np.exp(np.polyfit(np.log(se), np.log(sr), 1)[1]))
xx = np.geomspace(se.min(), se.max(), 50)
ax.fill_between(xx, A * xx**(-bhi), A * xx**(-blo), color=BAND, alpha=0.18, lw=0)
ax.plot(xx, A * xx**(-b), "-", color=BAND, lw=2.0,
        label=rf"relRMS $= A\,\varepsilon^{{-b}}$, $b={b:.2f}$" + "\n" +
              rf"  95% CI [{blo:.2f}, {bhi:.2f}]")
ax.plot(se, sr, "o", color="k", mfc="white", ms=5, label="within-geometry sweep (Krank hill)")
ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
ax.text(se.min(), 1.06, r"variance balance (relRMS=1)", fontsize=7.4, color="0.4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"cancellation depth $\varepsilon$ (within geometry)")
ax.set_ylabel(r"relRMS$(\tau_w)$")
ax.set_title(r"(d) informative axis (F2) vs honest negative (F3)")
ax.legend(loc="upper right", frameon=False, fontsize=7.4)
# inset: F3 cross-geometry honest negative
axin = ax.inset_axes([0.155, 0.205, 0.37, 0.35])
eps5 = L3["eps5"].astype(float); err5 = L3["err5"].astype(float)
p_eps = float(L3["p_eps5"])
axin.scatter(eps5, err5 * 100, s=30, color=GRAYB, zorder=3)
axin.set_xlabel(r"a-priori $\varepsilon$", fontsize=7)
axin.set_ylabel(r"coupled $e_{\rm reatt}$ [%]", fontsize=7)
axin.tick_params(labelsize=6.5)
axin.set_title(f"cross-geom (n=5):\nexact perm. p={p_eps:.2f}", fontsize=7)

fig.suptitle("The wall-model operating map across the repeating-structure class "
             "(a priori; coupled band traverse F4 pending)", fontsize=11.5, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.985])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_operating_map.{ext}"), bbox_inches="tight")
print("wrote fig_operating_map.{png,pdf} to", OUT)
