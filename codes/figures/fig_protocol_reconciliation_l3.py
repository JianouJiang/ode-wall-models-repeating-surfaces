#!/usr/bin/env python3
r"""L3 results figure (3-panel):
 (a) protocol reconciliation: the 37% onset-vs-rib relRMS gap is a deterministic
     RMS-vs-mean|tau| normalisation; rank ordering and the 0.5 screen survive.
 (b) kappa mode separation as a DIAGNOSTIC: 31.9x gap, zero overlap (lead with
     the gap; p is parenthetical, bind B-L3-3).
 (c) the directly-computed neglected convective term C* separates the modes and
     confirms the mechanism: Mode-I C*~O(1) (cancellation), Mode-II C*<<0.1.

Reads only codes/results/*.npz produced by the L3 analysis scripts.  No usetex.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_007")

REC = np.load(os.path.join(RESULTS, "protocol_reconciliation_l3.npz"), allow_pickle=True)
RIB = np.load(os.path.join(RESULTS, "rib_discriminant_heldout_l2.npz"), allow_pickle=True)
CON = np.load(os.path.join(RESULTS, "mode2_convective_confirmation_l3.npz"), allow_pickle=True)

C_I = "#c0392b"      # Mode-I (catastrophic cancellation)  -- distinct from model-type palette
C_II = "#2471a3"     # Mode-II (mild well-conditioned miss)
C_O = "#9aa0a6"      # other / intermediate

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5))

# ---------- (a) protocol reconciliation --------------------------------------
on = REC["onset_stored"]; rb = REC["rib_stored"]; ratio = REC["ratio_rms_over_mab"]
sc = ax[0].scatter(on, rb, c=ratio, cmap="viridis", s=42, edgecolor="k",
                   linewidth=0.4, zorder=3)
lim = [0, 1.45]
ax[0].plot(lim, lim, "k--", lw=0.9, zorder=1, label="identity")
ax[0].axhline(0.5, color="0.6", lw=0.7, ls=":"); ax[0].axvline(0.5, color="0.6", lw=0.7, ls=":")
# annotate the single straddler
flip = list(REC["screen_flip_tags"])
for i, t in enumerate(REC["shared"]):
    if str(t) in [str(f) for f in flip]:
        ax[0].annotate("op_a10_l22\n(only 0.5 straddler)", (on[i], rb[i]),
                       textcoords="offset points", xytext=(8, -22), fontsize=7,
                       arrowprops=dict(arrowstyle="->", lw=0.6))
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel(r"onset protocol  relRMS  ($/\sqrt{\langle\tau_w^2\rangle}$)")
ax[0].set_ylabel(r"$\kappa$ protocol  relRMS  ($/\langle|\tau_w|\rangle$)")
ax[0].set_title("(a) protocol reconciliation", fontsize=9)
cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.03)
cb.set_label(r"$\sqrt{\langle\tau_w^2\rangle}/\langle|\tau_w|\rangle\ \geq 1$", fontsize=7)
ax[0].text(0.04, 0.96, "gap = 100%% normalisation\n(closure term = 0, bit-exact)\n"
           r"rank $\rho=+%.3f$;  1/17 relabel" % REC["spearman_rank"],
           transform=ax[0].transAxes, va="top", ha="left", fontsize=7,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))

# ---------- (b) kappa diagnostic separation ----------------------------------
tags = [str(t) for t in RIB["tags"]]
medk = RIB["med_kappa"]
MI = ["oa_a05_l02", "oa_a15_l02", "oa_a20_l02", "op_a10_l03"]
MII = ["op_a10_l22", "op_a40_l14", "op_a40_l16"]
kI = np.array([medk[tags.index(t)] for t in MI])
kII = np.array([medk[tags.index(t)] for t in MII])
rng = np.random.default_rng(0)
xI = 1 + 0.06 * rng.standard_normal(len(kI))
xII = 2 + 0.06 * rng.standard_normal(len(kII))
ax[1].axhspan(kII.max(), kI.min(), color="0.9", zorder=0)  # the empty gap band
ax[1].scatter(xI, kI, c=C_I, s=55, edgecolor="k", linewidth=0.4, label="Mode I (ill-cond.)", zorder=3)
ax[1].scatter(xII, kII, c=C_II, s=55, edgecolor="k", linewidth=0.4, label="Mode II (well-cond.)", zorder=3)
ax[1].set_yscale("log")
ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["Mode I", "Mode II"])
ax[1].set_xlim(0.5, 2.5)
ax[1].set_ylabel(r"conditioning floor  $\kappa$  (median)")
ax[1].set_title("(b) $\\kappa$ separates the modes", fontsize=9)
gap = kI.min() / kII.max()
ax[1].text(0.5, 0.5, r"$%.0f\times$ gap" % gap + "\nzero overlap\nAUC$=1.000$",
           transform=ax[1].transAxes, ha="center", va="center", fontsize=8,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))
ax[1].text(0.5, 0.04, "(rank test for completeness: $p=0.057$, $n{=}4$ vs $3$)",
           transform=ax[1].transAxes, ha="center", va="bottom", fontsize=6.3, color="0.4")
ax[1].legend(fontsize=6.5, loc="upper right", framealpha=0.9)

# ---------- (c) C* mechanistic confirmation ----------------------------------
allt = [str(t) for t in CON["tags"]]
C = CON["Cstar"]; E = CON["eps_med"]
modeI = set(str(t) for t in CON["mode_I"]); modeII = set(str(t) for t in CON["mode_II"])
for t, c, e in zip(allt, C, E):
    col = C_I if t in modeI else (C_II if t in modeII else C_O)
    mk = "o" if t in modeI else ("s" if t in modeII else "^")
    ax[2].scatter(e, c, c=col, marker=mk, s=46, edgecolor="k", linewidth=0.4, zorder=3)
ax[2].axhline(0.5, color="0.55", lw=0.7, ls="--")
ax[2].axvline(1.0, color="0.55", lw=0.7, ls=":")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel(r"$\bar\varepsilon$  (median, cancellation depth)")
ax[2].set_ylabel(r"$C^\ast=\mathrm{med}\,|U\,\partial_x U|/\mathrm{med}\,|\partial_x p|$")
ax[2].set_title("(c) neglected convection confirms mechanism", fontsize=9)
ax[2].text(0.04, 0.06, r"Mode-I $C^\ast\!\sim\!O(1)$ (cancellation)" + "\n" +
           r"Mode-II $C^\ast\!\ll\!0.1$ (no cancellation)" + "\n" +
           r"$\rho(C^\ast,\log\bar\varepsilon)=%.2f$" % CON["spearman_C_logeps"],
           transform=ax[2].transAxes, va="bottom", ha="left", fontsize=7,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_I, markeredgecolor="k", label="Mode I", markersize=7),
       Line2D([0], [0], marker="s", color="w", markerfacecolor=C_II, markeredgecolor="k", label="Mode II", markersize=7),
       Line2D([0], [0], marker="^", color="w", markerfacecolor=C_O, markeredgecolor="k", label="other", markersize=7)]
ax[2].legend(handles=leg, fontsize=6.5, loc="upper right", framealpha=0.9)

fig.tight_layout()
for d in (HERE, NODE):
    fig.savefig(os.path.join(d, "fig_protocol_reconciliation_l3.png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(d, "fig_protocol_reconciliation_l3.pdf"), bbox_inches="tight")
print("saved fig_protocol_reconciliation_l3.{png,pdf} -> codes/figures/ and node_007/")
