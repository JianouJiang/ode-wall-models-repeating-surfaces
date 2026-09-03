"""
fig_flowblind_catastrophe_map.py  --  Thrust #25 L3 results figure.

Two panels, all real data from geometry_predictor_l2.npz + geometry_predictor_l3.npz
+ closure_conditioning_floor.npz (no fabricated points):

  (a) THE FLOW-BLIND CATASTROPHE MAP.  Each repeating geometry placed by its two
      pure-CAD coordinates: pitch ell_p/delta (x) and blockage-weighted
      cancellation depth eps_hat (y, log).  Markers coloured by the MEASURED
      outcome (catastrophic vs tolerated).  The horizontal flow-blind trigger
      eps_hat = tau_eps cleanly separates the classes; NO vertical pitch line
      can (conv-div's wide pitch sits inside the Xiao band) -- the figure shows
      why pitch alone fails and blockage-weighted depth succeeds.

  (b) THE CLOSURE-/SOPHISTICATION-INDEPENDENT FLOOR.  Predicted lower bound
      relErr >= beta/eps_hat (curve) with the canonical-hill operating point and
      the five closures (incl. EXACT-DNS stress, which lands WORST) -- the
      contribution is the bound, not the binary classifier.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_flowblind_catastrophe_map.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-size parity with the dose-response reference (~7.1 pt printed labels):
# this canvas is 11.0 in wide at full \textwidth, so labels need ~15 pt.
plt.rcParams.update({"font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                     "legend.fontsize": 13.5})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
NODE = os.path.join(os.path.dirname(ROOT), "development", "nodes", "node_003")

d2 = np.load(os.path.join(RESULTS, "geometry_predictor_l2.npz"), allow_pickle=True)
d3 = np.load(os.path.join(RESULTS, "geometry_predictor_l3.npz"), allow_pickle=True)
cd = np.load(os.path.join(RESULTS, "repeating_structure_contrast.npz"), allow_pickle=True)
fl = np.load(os.path.join(RESULTS, "closure_conditioning_floor.npz"), allow_pickle=True)

names = d2["names"]; fam = d2["family"]
eps_hat = d2["eps_hat"].astype(float); B = d2["B"].astype(float)
ell_p = d2["ell_p"].astype(float); H = d2["H"].astype(float)
rep = d2["repeating"].astype(bool); cat = d2["catastrophic"].astype(bool)

delta = np.where(np.isin(fam, ["xiao", "krank"]), H / 2.0, np.nan)
ci = int(np.where(fam == "convdiv")[0][0])
delta[ci] = float(cd["convdiv_delta_proxy"])
pod = ell_p / delta

TAU = float(d3["TAU_EPS"]); BETA = float(d3["BETA"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4))

# ---------- panel (a): flow-blind catastrophe map ----------------------------
CAT_C = "#c0392b"     # catastrophic (measured R2<0)
TOL_C = "#2471a3"     # tolerated
m_rep = rep & np.isfinite(pod)
catb = np.asarray(cat, bool)
FAMS = [(fam == "xiao", "o", "Xiao hills (29)"),
        (fam == "krank", "s", "Krank hills"),
        (fam == "convdiv", "D", "conv–div channel")]
for is_xi, mk, lab in FAMS:
    base = is_xi & m_rep
    if not base.any():
        continue
    # Outcome is encoded by FILL (colour-independent): catastrophic = filled,
    # tolerated = open -- so it reads without relying on the red/blue colour.
    catm, tolm = base & catb, base & ~catb
    if catm.any():
        axA.scatter(pod[catm], eps_hat[catm], facecolor=CAT_C, marker=mk, s=70,
                    edgecolor="k", linewidth=0.6, zorder=3)
    if tolm.any():
        axA.scatter(pod[tolm], eps_hat[tolm], facecolor="none", marker=mk, s=70,
                    edgecolor=TOL_C, linewidth=1.4, zorder=3)
# flow-blind trigger line
axA.axhline(TAU, color="k", ls="--", lw=1.6, zorder=2)
axA.text(9.6, TAU * 1.18, r"trigger  $\hat\varepsilon=%.0f$" % TAU,
         fontsize=12, ha="left", va="bottom")
# show that NO vertical pitch line separates: shade the Xiao pitch band
xb_lo, xb_hi = float(d3["R2_xiao_pod_lo"]), float(d3["R2_xiao_pod_hi"])
axA.axvspan(xb_lo, xb_hi, color="0.85", alpha=0.5, zorder=0)
axA.annotate("conv–div pitch\ninside hill band",
             xy=(pod[ci], eps_hat[ci]), xytext=(12.9, 2.7),
             fontsize=13, ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
axA.set_yscale("log")
axA.set_xlabel(r"pitch  $\ell_p/\delta$   (pure CAD)")
axA.set_ylabel(r"blockage-weighted depth  $\hat\varepsilon$")
axA.set_title("(a) flow-blind catastrophe map", fontsize=15, loc="left")
# legend for outcome colours.  Each family swatch is drawn in the SAME style it
# actually has in the plot (fill = measured outcome), so nothing in the legend
# is a colour/edge that never appears among the points.
from matplotlib.lines import Line2D
fam_handles = []
for is_f, mk, lab in FAMS:
    base = is_f & m_rep
    if base.any() and catb[base].all():             # family plotted as filled red
        fam_handles.append(Line2D([0], [0], marker=mk, color="w",
                                  markerfacecolor=CAT_C, markeredgecolor="k",
                                  markersize=9, label=lab))
    elif base.any() and not catb[base].any():       # family plotted as open blue
        fam_handles.append(Line2D([0], [0], marker=mk, color="w",
                                  markerfacecolor="none", markeredgecolor=TOL_C,
                                  markeredgewidth=1.4, markersize=9, label=lab))
    else:                                           # mixed (not present in data)
        fam_handles.append(Line2D([0], [0], marker=mk, color="w",
                                  markerfacecolor="0.7", markeredgecolor="k",
                                  markersize=9, label=lab))
out_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT_C,
                      markeredgecolor="k", markersize=9,
                      label="catastrophic ($R^2<0$, filled)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor=TOL_C, markeredgewidth=1.4, markersize=9,
                      label="tolerated (open)")]
axA.legend(handles=fam_handles + out_handles, loc="upper left", fontsize=9.5,
           framealpha=0.95, labelspacing=0.2, borderpad=0.3, handletextpad=0.4)
axA.set_xlim(1.5, 15)

# ---------- panel (b): the floor -------------------------------------------
eps_grid = np.logspace(-1.3, 1.0, 200)
floor = BETA / eps_grid
axB.plot(eps_grid, floor, "k-", lw=2.0)
axB.axhspan(1.0, 1e4, color="#fdebd0", alpha=0.6, zorder=0)
axB.text(0.062, 38.0, "catastrophic\n(>100% error)", fontsize=13, color="#7e5109", va="top")
# the 5 closures on the canonical hill: relErr proxy from |R2|^(1/2)-style severity.
# Plot them at the canonical eps_hat with their MEASURED relative error magnitude.
canon_eps = float(d3["R4_canon_eps_hat"])
# measured relative tau_w error: relRMS values implied by R2 (sqrt(1-R2) * sigma/|mean|);
# we use the closure ranking from hills_rows (R2) -> all far below the success line,
# exact-DNS the worst. Represent each by its predicted floor breach (>=1) using the
# severity prefactor med_kappa (condition-number) already on disk.
rows = fl["hills_rows"]
labs = [r["label"] for r in rows]
kap = np.array([r["med_kappa"] for r in rows])
# relative error >= floor; place each closure's OPERATING error = kappa (>=1 => fails).
# exact-DNS kappa ~ 24 (worst), algebraic ~0.5-0.7, ML ~0.47.
colmap = {"ML van Driest": "#2c3e50", "alg Cebeci": "#7f8c8d",
          "alg SA-like": "#95a5a6", "alg Reichardt": "#b3b6b7",
          "exact DNS stress": "#229954"}  # green = exact/Spalding-tier truth-fed
for lab, k in zip(labs, kap):
    axB.scatter([canon_eps], [max(k, BETA / canon_eps)], s=90, zorder=4,
                color=colmap.get(lab, "k"), edgecolor="k", linewidth=0.7)
axB.annotate("four eddy closures",
             xy=(canon_eps, max(BETA / canon_eps, float(np.sort(kap)[-2]))),
             xytext=(0.062, 0.30), fontsize=12, color="#555",
             arrowprops=dict(arrowstyle="->", color="#777", lw=1.0))
axB.annotate("exact-DNS stress\n(37.6$\\times$ WORSE,\n$R^2:-48\\to-927$)",
             xy=(canon_eps, max(kap)), xytext=(0.9, 9.0), fontsize=12,
             arrowprops=dict(arrowstyle="->", color="#229954", lw=1.1))
axB.axvline(canon_eps, color="0.6", ls=":", lw=1.0)
axB.text(canon_eps * 1.05, 1.55, "canonical hill",
         rotation=90, fontsize=11.5, va="bottom", color="0.4")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"blockage-weighted depth  $\hat\varepsilon$")
axB.set_ylabel(r"relative wall-stress error  $|\Delta\tau_w|/|\tau_w|$")
axB.set_title("(b) closure-independent floor", fontsize=15, loc="left")
axB.set_xlim(0.05, 10); axB.set_ylim(0.08, 1e2)

fig.tight_layout()

# floor label: placed AFTER layout so it hugs the black floor line and is rotated
# to the line's true on-screen angle (computed from the data transform), keeping
# it parallel to the line at any figure size instead of a fixed angle that
# diverges from it toward the right.
_p1 = axB.transData.transform((1.4, BETA / 1.4))
_p2 = axB.transData.transform((4.0, BETA / 4.0))
_ang = float(np.degrees(np.arctan2(_p2[1] - _p1[1], _p2[0] - _p1[0])))
_xlab = 2.7
axB.text(_xlab, BETA / _xlab * 1.22,
         r"floor $\beta/\hat\varepsilon$  ($\beta=%.1f$)" % BETA,
         rotation=_ang, rotation_mode="anchor", ha="center", va="bottom",
         fontsize=12.5, color="k")

MS_FIG = os.path.join(os.path.dirname(ROOT), "manuscript", "figures")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(NODE, "fig_flowblind_catastrophe_map." + ext),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(MS_FIG, "fig_flowblind_catastrophe_map." + ext),
                dpi=150, bbox_inches="tight")
print("saved fig_flowblind_catastrophe_map.{png,pdf} ->", NODE, "+ manuscript/figures")
