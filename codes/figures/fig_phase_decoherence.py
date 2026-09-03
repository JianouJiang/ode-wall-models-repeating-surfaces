"""Figure: the tau_w-free coherence discriminant gamma = rho_E * R.

(a) The factorisation plane (rho_E = spectral co-location, R = cross-phase
    concentration); each geometry sits at (rho_E, R) with grey hyperbolic
    iso-gamma contours.  The periodic hills collapse to the low-R corner while
    the tolerated flows occupy the high-R/high-rho_E corner: failure is a phase
    (R) collapse, not a co-location collapse.

(b) The co-energy-weighted cross-spectral phase distribution P(phi) for a
    periodic hill (alpha=1.0, fails) versus the wide-pitch conv-div control
    (tolerated).  The hill's cross-phase is spread around the circle (R~0); the
    conv-div is sharply concentrated near phi=pi (anti-phase, R~1).

Reads codes/results/spectral_phase_decoherence.npz (real, a-priori).
Run: OMP_NUM_THREADS=2 python3 codes/figures/fig_phase_decoherence.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
MS_FIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

C_FAIL = "#c0392b"   # ODE fails  (crimson)
C_PASS = "#2c6fbb"   # ODE tolerated (blue)
C_HILL = "#e08214"   # periodic-hills family highlight (orange ring)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

d = np.load(os.path.join(RES, "spectral_phase_decoherence.npz"), allow_pickle=True)
keys = [str(k) for k in d["keys"]]
rho_E = d["rho_E"]; R = d["R"]; fail = d["fail_real"].astype(bool)
hill = np.array([k.startswith("pehill_a") for k in keys])

fig, (ax, axp) = plt.subplots(1, 2, figsize=(7.2, 3.2))
fig.subplots_adjust(wspace=0.30, bottom=0.18, top=0.92)

# ----------------------------------------------------------------- panel (a)
rr = np.linspace(0.02, 1.0, 200)
for g0 in (0.1, 0.3, 0.6, 0.9):
    ax.plot(rr, np.clip(g0 / rr, 0, 1.05), color="0.80", lw=0.7, zorder=1)
    x_lab = np.clip(g0 / 0.98, 0.27, 0.97)
    ax.text(x_lab, 1.01, f"$\\gamma={g0:g}$", color="0.55", fontsize=6.5,
            ha="center", va="bottom")

for i, k in enumerate(keys):
    col = C_FAIL if fail[i] else C_PASS
    ax.scatter(rho_E[i], R[i], s=46, c=col, edgecolors="k", linewidths=0.5,
               zorder=4)
    if hill[i]:
        ax.scatter(rho_E[i], R[i], s=140, facecolors="none", edgecolors=C_HILL,
                   linewidths=1.6, zorder=3)

ax.set_xlabel(r"spectral co-location  $\rho_E=\sum_k|c_k||p_k|$")
ax.set_ylabel(r"phase concentration  $R$")
ax.set_xlim(0.25, 1.02); ax.set_ylim(0.0, 1.06)
ax.set_title(r"(a)  $\gamma=\rho_E R$", fontsize=10)

from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", ls="", mfc=C_FAIL, mec="k", ms=6, label="ODE fails"),
       Line2D([0], [0], marker="o", ls="", mfc=C_PASS, mec="k", ms=6, label="ODE tolerated"),
       Line2D([0], [0], marker="o", ls="", mfc="none", mec=C_HILL, mew=1.6, ms=8,
              label="periodic hills")]
ax.legend(handles=leg, fontsize=6.8, loc="lower left", framealpha=0.95,
          handletextpad=0.4, borderpad=0.4)

# ----------------------------------------------------------------- panel (b)
nb = 36
edges = np.linspace(-np.pi, np.pi, nb + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
for ex, col, lab in (("pehill_a1p0", C_FAIL, r"hill $\alpha{=}1.0$ (fails)"),
                     ("conv_div", C_PASS, r"conv--div (tolerated)")):
    phi = d[f"fig_{ex}_phi"]; co = d[f"fig_{ex}_co"]
    Rex = R[keys.index(ex)]
    w, _ = np.histogram(phi, bins=edges, weights=co)
    w = w / w.sum()
    axp.plot(centers, w, color=col, lw=1.6, label=lab + rf", $R={Rex:.2f}$")

axp.set_xlim(-np.pi, np.pi)
axp.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
axp.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
axp.set_xlabel(r"cross-spectral phase  $\varphi_k$")
axp.set_ylabel(r"co-energy weight  $P(\varphi)$")
axp.set_title(r"(b)  phase distribution", fontsize=10)
axp.legend(fontsize=6.8, loc="upper center", framealpha=0.95,
           handletextpad=0.4, borderpad=0.4)

for out in (os.path.join(HERE, "fig_phase_decoherence.pdf"),
            os.path.join(HERE, "fig_phase_decoherence.png"),
            os.path.join(MS_FIG, "fig_phase_decoherence.pdf")):
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print("wrote", out)
