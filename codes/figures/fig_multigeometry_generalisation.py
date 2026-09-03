#!/usr/bin/env python3
r"""
fig_multigeometry_generalisation.py  --  node_003 (Thrust #16 L3) G7 figure.

Generalisation across the repeating-structure class + the honest O(delta)-pitch
boundary.  Two panels:

 (a) A-PRIORI dose-response across the Xiao periodic-hill steepness family
     (29 cases, 7 alpha-shapes x matching heights): R^2(tau_w) vs L_sep/delta.
     The catastrophe is governed by the geometric separation length: deeper
     cancellation (smaller eps, larger L_sep/delta) -> larger error.
     Spearman rho(R^2, L_sep/delta) annotated.  Source: dose_response_xiao.npz.

 (b) A-POSTERIORI coupled-WMLES deployment error vs a-priori eps across distinct
     repeating geometries that HAVE reference data: the Xiao hill family
     (alpha=0.8/1.0/1.2 @ Re5600) and the converging-diverging channel
     (Re12600) as the NEGATIVE control.  The conv-div channel has widely-spaced
     repeats (eps ~ O(1)) -> benign; the O(delta)-pitch hills -> at-risk.  This
     is the honest boundary of the claim.  Sources:
     aposteriori_dose_response_xiao_{0p8,1p0,1p2}.npz, aposteriori_wmles_convdiv.npz.

Colors: orange = DNS truth; hills shaded by alpha; conv-div control in green.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
OUT = os.path.join(ROOT, "development", "nodes", "node_003")
os.makedirs(OUT, exist_ok=True)

ORANGE = "#E69F00"; GREEN = "#2C9D3A"; BAND = "#D55E00"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10.5,
                     "legend.fontsize": 8.5, "savefig.dpi": 200, "lines.linewidth": 1.8})

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

# ---- (a) a-priori family dose-response ------------------------------------- #
ax = axes[0]
d = np.load(os.path.join(RES, "dose_response_xiao.npz"), allow_pickle=True)
cases = list(d["agg_case"])
r2 = d["agg_r2"]; lsep = d["agg_cv_Lsep_over_delta"]; epsm = d["agg_eps_median"]
# alpha from case label "alphXX..."
def alpha_of(c):
    import re
    m = re.match(r"alph(\d+)", c)
    if not m: return np.nan
    s = m.group(1)
    return float(s[0] + "." + s[1:]) if len(s) > 1 else float(s)
alphas = np.array([alpha_of(c) for c in cases])
uniq = sorted(set(alphas[np.isfinite(alphas)]))
cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(uniq)))
for a, col in zip(uniq, cmap):
    m = alphas == a
    ax.scatter(lsep[m], r2[m], s=42, color=col, edgecolor="k", lw=0.4,
               label=rf"$\alpha$={a:g}", zorder=3)
rho, p = spearmanr(lsep, r2)
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_yscale("symlog")
ax.set_xlabel(r"separation length $L_{\rm sep}/\delta$")
ax.set_ylabel(r"$R^2(\tau_w)$  (a priori, symlog)")
ax.set_title(f"(a) a-priori dose–response, Xiao family\n"
             f"$\\rho(R^2, L_{{\\rm sep}}/\\delta)$ = {rho:.2f} ($n$={len(r2)})")
ax.legend(loc="lower right", frameon=False, ncol=2)

# ---- (b) a-posteriori across geometries ------------------------------------ #
ax = axes[1]
pts = []  # (eps, e_reatt, label, color, marker)
xiao = {"0p8": (0.8, "#7E1E9C"), "1p0": (1.0, "#C0392B"), "1p2": (1.2, "#E67E22")}
for tag, (al, col) in xiao.items():
    p = os.path.join(RES, f"aposteriori_dose_response_xiao_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        e_re = float(z["e_reatt"]) * 100.0
        eps = float(z["apriori_eps_med"])
        pts.append((eps, e_re, rf"hill $\alpha$={al:g}", col, "o"))
# band-confirmation point (same a1p0 geometry, matched IN band)
bpath = os.path.join(RES, "aposteriori_band_confirmation.npz")
if os.path.exists(bpath):
    b = np.load(bpath, allow_pickle=True)
    if np.isfinite(float(b["e_reatt"])):
        pts.append((float(b["apriori_eps_med"]), float(b["e_reatt"]) * 100.0,
                    rf"hill $\alpha$=1.0 IN-BAND ($y_m^+\!\approx\!{float(b['ym_plus_avg']):.0f}$)",
                    BAND, "*"))
# conv-div negative control
cd = os.path.join(RES, "aposteriori_wmles_convdiv.npz")
if os.path.exists(cd):
    z = np.load(cd, allow_pickle=True)
    pts.append((float(z["wmles_eps_median"]), float(z["reattachment_rel_err_pct"]),
                "conv–div control", GREEN, "D"))
for eps, ere, lab, col, mk in pts:
    ax.scatter(eps, ere, s=(200 if mk == "*" else 80), color=col, marker=mk,
               edgecolor="k", lw=0.6, label=lab, zorder=3)
ax.axvspan(0, 0.2, color=BAND, alpha=0.10, lw=0)
ax.text(0.18, ax.get_ylim()[1] * 0.5 if False else 0, "", )
ax.set_xlabel(r"a-priori cancellation depth  $\varepsilon_{\rm med}$")
ax.set_ylabel(r"a-posteriori reattachment error (\%)")
ax.set_title("(b) a-posteriori coupled WMLES\n(reference-validated geometries)")
ax.legend(loc="upper right", frameon=False)
ax.annotate("O($\\delta$)-pitch\n(at-risk)", xy=(0.1, ax.get_ylim()[1]),
            ha="center", va="top", color=BAND, fontsize=8)

fig.tight_layout()
out = os.path.join(OUT, "fig_multigeometry_generalisation.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| a-posteriori points:", len(pts))
