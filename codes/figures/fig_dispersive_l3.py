#!/usr/bin/env python3
r"""L3 (Results and analysis) figure for the dispersive-stress thesis — the four
results that discharge the L2 Judge's L3 binds.  Reads ONLY
codes/results/dispersive_l3_results.npz (produced by dispersive_l3_results.py).

Panels:
 (a) B-L2-1 — closure-ratio DECOMPOSITION: the double-averaged total shear stress
     decays through the matching layer, and the (1-0.66)=34% gap splits into the
     viscous handoff (66.8% of tau_w lost) minus the slow turbulent+dispersive
     rebuild (32.3%); the slow rebuild IS the cancellation.
 (b) B-L2-2 — proper 29-vs-8 separation of the dispersive over-prediction Delta;
     AUC and Mann-Whitney p annotated; the thin empirical gap shaded honestly.
 (c) B-L2-3 — Delta vs the coverage discriminant frac[eps<0.1]: strong but not
     identical (Spearman rho), so Delta is a severity magnitude, not a re-brand.
 (d) reason-#4 — Delta non-monotone in steepness alpha is a DENOMINATOR effect:
     numerator (dispersive stress) saturates, denominator (mean wall stress) grows.

Colours per project convention: orange = ground-truth/failure (dispersive),
bluish-gray = tolerated/Reynolds, black = net residual.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
FIGM = os.path.join(ROOT, "manuscript", "figures")
NODE = os.path.join(ROOT, "development", "nodes", "node_004")

ORANGE = "#E8820C"     # ground truth / dispersive / failure
GRAY = "#6A7B8C"       # bluish-gray / tolerated / Reynolds
BLACK = "#1A1A1A"

d = np.load(os.path.join(RES, "dispersive_l3_results.npz"), allow_pickle=True)

fig, ax = plt.subplots(1, 4, figsize=(15.2, 3.7))

# ---- (a) closure decomposition -------------------------------------------
eta = d["prof_eta"]
tot = d["prof_tau_tot"]
tau_w = float(d["dec_tau_w_mean"])
etam = float(d["dec_eta_m"])
a0 = ax[0]
m = (eta > 0) & (eta <= 0.30)
a0.plot(tot[m] / tau_w, eta[m], "-", color=BLACK, lw=2.0,
        label=r"$\tau_{\rm tot}(\eta)/\tau_w$")
a0.axvline(1.0, color=ORANGE, ls="--", lw=1.4, label=r"$\tau_w$ (wall)")
a0.axhline(etam, color=GRAY, ls=":", lw=1.2)
cr = float(d["dec_closure_ratio"])
a0.plot([cr], [etam], "o", color=GRAY, ms=8, zorder=5)
a0.annotate(rf"matching height $\eta_m$" + "\n" + rf"ratio $={cr:.2f}$",
            xy=(cr, etam), xytext=(cr + 0.15, etam + 0.07),
            fontsize=8.5, color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
a0.set_xlabel(r"$\tau_{\rm tot}/\tau_w$")
a0.set_ylabel(r"wall distance $\eta=y-y_w(x)$")
a0.set_title(r"(a) closure ratio $0.66$ decomposed", fontsize=10)
a0.set_ylim(0, 0.30)
a0.set_xlim(0, 1.25)
# inset: the 34% gap = viscous decay - residual rebuild
fv = float(d["dec_frac_visc_decay"]) * 100
fr = float(d["dec_frac_resid_rebuild"]) * 100
fg = float(d["dec_frac_gap"]) * 100
txt = (rf"$34\%$ gap $=$ viscous handoff $({fv:.0f}\%)$" + "\n"
       rf"$-$ slow turb$+$disp rebuild $({fr:.0f}\%)$" + "\n"
       rf"$=$ {fg:.0f}\% (the cancellation)")
a0.text(0.04, 0.015, txt, fontsize=7.6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF6E8", ec=ORANGE, lw=0.8))
a0.legend(fontsize=8, loc="upper left", framealpha=0.9)

# ---- (b) 29-vs-8 separation ----------------------------------------------
a1 = ax[1]
D29 = d["lad29_Delta"]
tolD = d["tol_Delta"]
auc = float(d["auc_29_vs_8"])
pmw = float(d["mannwhitney_p"])
minf = float(d["sep_min_fail"])
maxt = float(d["sep_max_tol"])
rng = np.random.default_rng(0)
x29 = 1 + 0.12 * rng.standard_normal(len(D29))
x8 = 2 + 0.10 * rng.standard_normal(len(tolD))
a1.axhspan(maxt, minf, color="0.85", alpha=0.7, zorder=0)
a1.scatter(x29, D29, s=34, color=ORANGE, edgecolor="k", lw=0.4,
           label=f"failing hills (n={len(D29)})", zorder=3)
a1.scatter(x8, tolD, s=44, color=GRAY, marker="s", edgecolor="k", lw=0.4,
           label=f"tolerated (n={len(tolD)})", zorder=3)
a1.axhline(minf, color=ORANGE, ls="--", lw=1.0)
a1.axhline(maxt, color=GRAY, ls="--", lw=1.0)
a1.text(2.5, 0.5 * (minf + maxt),
        rf"gap ${minf/maxt:.2f}\times$" + "\n(thin)", fontsize=7.8,
        ha="center", va="center", color="0.25")
a1.set_xticks([1, 2])
a1.set_xticklabels(["failing", "tolerated"])
a1.set_xlim(0.5, 3.1)
a1.set_ylabel(r"$\Delta=|\langle\tilde u\tilde v\rangle(\eta_m)|/\overline{|\tau_w|}$")
a1.set_title(rf"(b) $\Delta$ separates $29$ vs $8$", fontsize=10)
a1.text(0.04, 0.96, rf"AUC$={auc:.2f}$" + "\n" + rf"$p={pmw:.0e}$ (MWU)",
        transform=a1.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.6", lw=0.7))
a1.legend(fontsize=7.6, loc="upper right", framealpha=0.9)

# ---- (c) Delta vs coverage -----------------------------------------------
a2 = ax[2]
cov = d["lad29_coverage"]
r2 = d["lad29_r2"]
rho_cov = float(d["rho_Delta_coverage"])
p_cov = float(d["p_Delta_coverage"])
sc = a2.scatter(cov, D29, c=r2, cmap="viridis_r", s=46, edgecolor="k", lw=0.4)
cb = fig.colorbar(sc, ax=a2, fraction=0.046, pad=0.03)
cb.set_label(r"a-priori $R^2(\tau_w)$", fontsize=8)
a2.set_xlabel(r"coverage  frac$[\varepsilon<0.1]$")
a2.set_ylabel(r"$\Delta$")
a2.set_title(r"(c) $\Delta$ tracks coverage, not a re-brand", fontsize=10)
a2.text(0.04, 0.96, rf"$\rho={rho_cov:+.2f}$ ($p={p_cov:.0e}$)" + "\n"
        + r"$\neq +1$: magnitude vs extent",
        transform=a2.transAxes, fontsize=8.3, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.6", lw=0.7))

# ---- (d) non-monotonicity = denominator effect ---------------------------
a3 = ax[3]
alpha = d["lad29_alpha"]
num = d["lad29_num"]
den = d["lad29_den"]
avals = sorted(set(alpha.tolist()))
Dm = [np.nanmean(d["lad29_Delta"][alpha == a]) for a in avals]
numm = [np.nanmean(num[alpha == a]) for a in avals]
denm = [np.nanmean(den[alpha == a]) for a in avals]
ln1 = a3.plot(avals, Dm, "o-", color=BLACK, lw=1.8, ms=6,
              label=r"$\Delta$ (ratio)")
a3.set_xlabel(r"steepness $\alpha$")
a3.set_ylabel(r"$\Delta=$ num$/$den", color=BLACK)
a3.set_title(r"(d) $\Delta\!\downarrow$ is a denominator effect", fontsize=10)
a3b = a3.twinx()
ln2 = a3b.plot(avals, np.array(numm) * 1e2, "s--", color=ORANGE, lw=1.4, ms=5,
               label=r"num $|\langle\tilde u\tilde v\rangle|\times10^2$ (saturates)")
ln3 = a3b.plot(avals, np.array(denm) * 1e3, "^:", color=GRAY, lw=1.4, ms=5,
               label=r"den $\overline{|\tau_w|}\times10^3$ (grows)")
a3b.set_ylabel(r"numerator / denominator", fontsize=8.5)
rho_n = float(d["nm_rho_alpha_num"])
rho_d = float(d["nm_rho_alpha_den"])
p_d = float(d["nm_p_alpha_den"])
lns = ln1 + ln2 + ln3
a3.legend(lns, [l.get_label() for l in lns], fontsize=7.3, loc="upper center",
          framealpha=0.9)
a3.text(0.03, 0.04,
        rf"$\rho(\alpha,$den$)={rho_d:+.2f}$ ($p={p_d:.0e}$)" + "\n"
        + rf"$\rho(\alpha,$num$)={rho_n:+.2f}$ (n.s.)",
        transform=a3.transAxes, fontsize=7.6, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.6", lw=0.7))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(NODE, f"fig_dispersive_l3.{ext}"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGM, f"fig_dispersive_l3.{ext}"), dpi=160,
                bbox_inches="tight")
print("WROTE fig_dispersive_l3.{pdf,png} to node_004/ and manuscript/figures/")
