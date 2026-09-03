"""Thrust #21 L2 figure — the naive spectral predictor fails; the dropped-vs-kept
coherence gamma succeeds.  Consumes codes/results/spectral_wallstress.npz only.

(a) The PRE-REGISTERED high-k convective predictor S anti-discriminates: failing
    (red) and tolerated (blue) geometries overlap completely -- localized features
    are broadband high-k too.
(b) The corrected tau_w-free discriminant gamma=|corr(Cbar,Pibar)| vs the VERIFIED
    a-priori ODE error (relRMS): decoupled (gamma~0) => catastrophic; slaved
    (gamma->1) => tolerated.  Out-of-sample bumps (open) confirm it.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
d = np.load(os.path.join(RES, "spectral_wallstress.npz"), allow_pickle=True)

keys = [str(k) for k in d["keys"]]
gamma = d["gamma"]
S = d["naive_S"]
rel = d["ode_relRMS"]
fail = d["fail_real"].astype(bool)
oos = ~d["in_discovery"].astype(bool)

RED, BLUE = "#c0392b", "#2c6fbb"
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.5))

# ---- (a) naive S falsification ----
m = np.isfinite(S)
for sel, c, lab in [(m & fail, RED, "ODE fails"), (m & ~fail, BLUE, "ODE tolerated")]:
    ax[0].scatter(S[sel], np.zeros(sel.sum()) + np.random.default_rng(1).uniform(-0.1, 0.1, sel.sum()),
                  c=c, s=70, edgecolor="k", lw=0.5, label=lab, zorder=3)
auc_S = float(d["F2naive_auc_S"])
ax[0].set_title(f"(a) pre-registered predictor $S$ FAILS\nAUC$(S,\\mathrm{{fail}})={auc_S:.3f}$ "
                f"(chance 0.5; anti-discriminates)", fontsize=11)
ax[0].set_xlabel(r"$S$ = high-$k$ convective energy fraction (Eq. S)")
ax[0].set_yticks([])
ax[0].set_ylim(-0.6, 0.6)
ax[0].legend(loc="upper center", fontsize=9, ncol=2, frameon=False)
ax[0].annotate("failing & tolerated\ngeometries OVERLAP", xy=(0.5, 0.0), xytext=(0.45, 0.42),
               fontsize=9, ha="center", color="gray")

# ---- (b) gamma success ----
relc = np.clip(rel, 1e-2, 1e4)
for sel, c, lab in [(fail, RED, "ODE fails"), (~fail, BLUE, "ODE tolerated")]:
    for openpt in [False, True]:
        s2 = sel & (oos == openpt)
        if s2.sum() == 0:
            continue
        ax[1].scatter(gamma[s2], relc[s2], s=85,
                      facecolor=("none" if openpt else c),
                      edgecolor=(c if openpt else "k"), lw=1.4 if openpt else 0.5,
                      label=(lab if not openpt else None), zorder=3)
# threshold band
ax[1].axhline(0.5, color="gray", ls="--", lw=1, zorder=1)
ax[1].axvspan(-0.02, 0.2, color=RED, alpha=0.06, zorder=0)
ax[1].axvspan(0.5, 1.02, color=BLUE, alpha=0.06, zorder=0)
for k, x, y in zip(keys, gamma, relc):
    if k in ("pehill_a1p0", "conv_div", "bump_h38", "nasa_hump", "curved_bfs"):
        ax[1].annotate(k.replace("pehill_", "hill ").replace("_", " "),
                       (x, y), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax[1].set_yscale("log")
ax[1].set_xlim(-0.03, 1.02)
auc_g = float(d["F2_gamma_auc"])
rho = float(d["F2_rho_gamma_relRMS"])
ax[1].set_title(f"(b) coherence $\\gamma=|\\mathrm{{corr}}(\\bar C,\\bar\\Pi)|$ SUCCEEDS\n"
                f"AUC$={auc_g:.3f}$,  Spearman$(\\gamma,\\mathrm{{relRMS}})={rho:+.2f}$", fontsize=11)
ax[1].set_xlabel(r"$\gamma$ = dropped-convection / kept-pressure coherence ($\tau_w$-free)")
ax[1].set_ylabel(r"verified a-priori ODE error  relRMS$(\tau_w)$")
ax[1].text(0.06, 1.5e3, "decoupled\n$\\Rightarrow$ FAILS", color=RED, fontsize=9, ha="center")
ax[1].text(0.78, 0.02, "slaved $\\Rightarrow$ tolerated", color=BLUE, fontsize=9, ha="center")
h = [plt.Line2D([], [], marker="o", ls="", mfc=RED, mec="k", label="ODE fails ($R^2<0.5$)"),
     plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec="k", label="ODE tolerated"),
     plt.Line2D([], [], marker="o", ls="", mfc="none", mec="gray", label="out-of-sample")]
ax[1].legend(handles=h, loc="center right", fontsize=8.5, frameon=False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(HERE, f"fig_spectral_coherence.{ext}"), dpi=160, bbox_inches="tight")
print("saved fig_spectral_coherence.{pdf,png}")
