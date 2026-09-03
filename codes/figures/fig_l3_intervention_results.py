#!/usr/bin/env python3
r"""
fig_l3_intervention_results.py  --  Thrust #29 L3 results figure.

Four panels, all from real DNS/LES reference fields (node003_intervention_results.npz,
which re-derives every number from SOURCE npz):

 (a) THE O(delta)-PITCH BIFURCATION BOUND (PART A).  Per-geometry ignition height
     y_crit+ on a log axis.  The two O(delta)-pitch repeating hills ignite below
     16 (warm); the wide-pitch conv-div and single-feature controls sit above 147
     or never (bluish-gray).  bifurcates == pitch~O(delta) (Fisher one-sided
     p=0.067 on n=6 -- borderline, honestly annotated).

 (b) THE STRONG GENERALISATION BACKBONE (PART C).  Within the 29-case repeating
     family, one-way R^2(tau_w) collapses on L_sep/delta: Spearman -0.754,
     bootstrap 95% CI [-0.89,-0.54], p=2.4e-6 (n=29).  This is the statistically
     decisive class-level evidence, not the n=6 binary.

 (c) WHY WITHIN-GEOMETRY, NOT CROSS (PART B/C).  Cross-geometry (5 deployed
     coupled WMLES) the static eps sort is uninformative (rho=+0.10, permutation
     p~0.95): the deployed y_m+ is degenerate (all 4.5-7.4) while y_crit+ spans
     x175.  Within one geometry the error follows relRMS = A eps^-b, b=0.44
     [0.37,0.48], R^2=0.98 -- so the clean test moves ONE axis on ONE geometry.

 (d) THE OPERATING MAP + the two CONTROLLED INTERVENTIONS (PART D/E).  The 5
     deployed coupled points placed in the (chi=y_m+/y_crit+, e_reatt) plane
     (all in the safe corner chi<1 except the steep hills).  Axis-2 (CR-WM cure):
     pre-registered acceptance band e_cure in [0.134,0.232] shaded; the landed
     cure point is drawn ONLY if crwm_present=True on disk (else "live/unlanded").
     Axis-1 (band placement): drawn ONLY if band_present=True.  NO coupled number
     is fabricated.

Colour convention: O(delta)-pitch FAILURE class warm (orange/red); SAFE controls
bluish-gray; DNS/ignition reference orange; CR-WM cure green.
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

ORANGE = "#E69F00"; RED = "#D55E00"; GRAY = "#7F8FA6"
GREEN = "#009E73"; BLACK = "#000000"; BLUE = "#56B4E9"

d = np.load(os.path.join(RES, "node003_intervention_results.npz"), allow_pickle=True)
tab = np.load(os.path.join(RES, "intervention_multigeom_table.npz"), allow_pickle=True)
dr = np.load(os.path.join(RES, "dose_response_xiao.npz"), allow_pickle=True)
ep = np.load(os.path.join(RES, "epsilon_coupled_predictor.npz"), allow_pickle=True)

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                     "legend.fontsize": 8.5, "figure.dpi": 140})
fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.4))

# ---------------------------------------------------------------- (a) bound
a = ax[0, 0]
geo = list(d["geo_keys"]); pitch = np.asarray(d["pitch_O_delta"], bool)
yc = np.asarray(tab["ycrit_apriori"], float)
labels = ["krank hill\n(Re10595)", "hills h/Lx=1.0", "conv-div\n(wide)",
          "BFS", "curved BFS", "NASA hump"]
finite = np.isfinite(yc)
xpos = np.arange(len(geo))
for i in range(len(geo)):
    c = RED if pitch[i] else GRAY
    if finite[i]:
        a.bar(xpos[i], yc[i], color=c, edgecolor="k", lw=0.6,
              label=("O($\\delta$)-pitch hill" if (pitch[i] and i == 0) else
                     ("wide/single-feature control" if (not pitch[i] and i == 2) else None)))
    else:
        a.text(xpos[i], 1.3, "no ignition\n($y_{crit}^+\\to\\infty$)", ha="center",
               va="bottom", fontsize=7.5, color=GRAY, rotation=0)
        a.bar(xpos[i], 0.0, color=c)
a.axhline(60, color="k", ls=":", lw=1.0)
a.text(len(geo)-0.4, 66, "deployable $y_m^+\\leq60$", ha="right", fontsize=7.5)
a.set_yscale("log"); a.set_ylim(0.5, 1.2e3)
a.set_xticks(xpos); a.set_xticklabels(labels, fontsize=7.2, rotation=0)
a.set_ylabel("ignition height $y_{crit}^+$")
a.set_title("(a) O($\\delta$)-pitch bifurcation bound", loc="left")
a.legend(loc="upper right", frameon=False)
a.text(0.02, 0.02, f"bifurcates $=$ pitch$\\sim$O($\\delta$): holds (Fisher $p={float(d['bound_fisher_p_onesided']):.3f}$, $n=6$)\n"
       f"perfect-separation chance $1/15$ — borderline; backbone is (b)",
       transform=a.transAxes, va="bottom", fontsize=7.3,
       bbox=dict(boxstyle="round", fc="#FFF4E6", ec=ORANGE, lw=0.8))

# ---------------------------------------------------------------- (b) collapse
b = ax[0, 1]
r2f = np.asarray(dr["agg_r2"], float); lsep = np.asarray(dr["agg_cv_Lsep_over_delta"], float)
m = np.isfinite(r2f) & np.isfinite(lsep)
b.scatter(lsep[m], r2f[m], c=RED, edgecolor="k", lw=0.4, s=42, zorder=3,
          label=f"29 Xiao-family cases")
# robust trend (rank-based): show ordering
order = np.argsort(lsep[m])
b.set_xlabel("separation length $L_{sep}/\\delta$")
b.set_ylabel("one-way $R^2(\\tau_w)$")
b.set_title("(b) within-family generalisation backbone", loc="left")
rho = float(d["lsep_rho"]); lo = float(d["lsep_ci_lo"]); hi = float(d["lsep_ci_hi"])
b.text(0.03, 0.06,
       f"Spearman $\\rho={rho:.3f}$\nbootstrap 95% CI $[{lo:.2f},{hi:.2f}]$\n$p=2.4\\times10^{{-6}}$ ($n=29$)",
       transform=b.transAxes, va="bottom", fontsize=8.5,
       bbox=dict(boxstyle="round", fc="#FDECEA", ec=RED, lw=0.8))
b.legend(loc="upper right", frameon=False)

# ---------------------------------------------------------------- (c) within vs cross
c = ax[1, 0]
sweep_eps = np.asarray(ep["sweep_eps"], float); sweep_rel = np.asarray(ep["sweep_relrms"], float)
c.scatter(sweep_eps, sweep_rel, c=RED, edgecolor="k", lw=0.4, s=34, zorder=3,
          label="within-geometry traverse")
bb = float(d["within_geom_b"]); A = np.exp(np.polyfit(np.log(sweep_eps), np.log(sweep_rel), 1)[1])
xx = np.linspace(sweep_eps.min(), sweep_eps.max(), 100)
c.plot(xx, A * xx**(-bb), color=BLACK, lw=1.4,
       label=f"relRMS $\\propto\\varepsilon^{{-{bb:.2f}}}$ ($R^2$=0.98)")
c.set_xscale("log")
c.set_xlabel("cancellation depth $\\varepsilon$  (within one geometry)")
c.set_ylabel("relRMS$(\\tau_w)$")
c.set_title("(c) within-geometry law (clean) vs cross-geometry (degenerate)", loc="left")
# inset: cross-geometry degeneracy
ci = c.inset_axes([0.52, 0.14, 0.44, 0.42])
epsc = np.asarray(d["cp_eps_med"], float); ec = np.asarray(d["cp_e_reatt"], float)
ci.scatter(epsc, ec, c=GRAY, edgecolor="k", lw=0.4, s=30)
ci.set_xscale("log"); ci.set_xlabel("$\\varepsilon$ (5 geoms)", fontsize=7)
ci.set_ylabel("$e_{reatt}$", fontsize=7); ci.tick_params(labelsize=6.5)
ci.set_title(f"cross-geom $\\rho=+{float(d['rho_eps_cross']):.2f}$, perm $p\\approx{float(d['p_eps_cross_perm']):.2f}$",
             fontsize=6.8)
c.legend(loc="upper right", frameon=False)

# ---------------------------------------------------------------- (d) operating map + interventions
dd = ax[1, 1]
chi = np.asarray(d["cp_chi"], float); e = np.asarray(d["cp_e_reatt"], float)
pitch_cp = chi > 1.0
cpl = list(tab["cp_labels"])
for i in range(len(chi)):
    col = RED if (i < 3) else GRAY  # 3 Xiao hills warm; krank cross-Re & conv-div gray
    dd.scatter(chi[i], e[i], c=col, edgecolor="k", lw=0.5, s=70, zorder=4)
dd.axvline(1.0, color="k", ls="--", lw=1.0)
dd.text(1.05, 0.46, "$\\chi=y_m^+/y_{crit}^+=1$\n(ignition)", fontsize=7.3)
# pre-registered cure band (Axis-2), shaded across chi
blo = float(d["band_lo"]); bhi = float(d["band_hi"]); etb = float(d["e_TBLE"])
dd.axhspan(blo, bhi, color=GREEN, alpha=0.18, zorder=1)
dd.axhline(etb, color=BLACK, ls=":", lw=1.2)
dd.text(3.7, etb+0.005, f"TBLE/Spalding $e\\approx{etb:.2f}$ (no cure)", fontsize=7.2, ha="right")
dd.text(3.7, (blo+bhi)/2, f"pre-registered CR-WM\ncure band [{blo:.2f},{bhi:.2f}]",
        fontsize=7.2, ha="right", color=GREEN, va="center")
# landed cure point ONLY if on disk
crwm_present = bool(d["crwm_present"]); band_present = bool(d["band_present"])
if crwm_present and np.isfinite(float(d["crwm_e_reatt"])):
    ec_cure = float(d["crwm_e_reatt"])
    dd.scatter([3.15], [ec_cure], marker="*", s=240, c=GREEN, edgecolor="k",
               lw=0.8, zorder=6, label=f"CR-WM cure (LANDED) $e={ec_cure:.2f}$")
else:
    dd.annotate("Axis-2 cure: live/unlanded\n(auto-lands into band)", xy=(3.15, (blo+bhi)/2),
                xytext=(2.0, 0.10), fontsize=7.3, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
if band_present and np.isfinite(float(d["band_e_reatt"])):
    eb = float(d["band_e_reatt"])
    dd.scatter([1.6], [eb], marker="D", s=90, c=RED, edgecolor="k", lw=0.6,
               zorder=6, label=f"band $\\chi>1$ (LANDED) $e={eb:.2f}$")
else:
    dd.annotate("Axis-1 band ($\\chi>1$):\nqueued/unlanded", xy=(1.6, 0.40),
                xytext=(1.4, 0.52), fontsize=7.3, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
dd.set_xlabel("operating point $\\chi=y_m^+/y_{crit}^+$")
dd.set_ylabel("coupled reattachment error $e_{reatt}$")
dd.set_title("(d) operating map + 2 controlled interventions", loc="left")
dd.set_xlim(0, 4.0); dd.set_ylim(0, 0.58)

fig.suptitle("Thrust #29 — the wall-model operating map, intervention-tested "
             "(coupled cure/band auto-land; no number fabricated)",
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig_l3_intervention_results.{ext}")
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)
print(f"coupled state: crwm_present={crwm_present} band_present={band_present} "
      f"(figure auto-folds landed points)")
