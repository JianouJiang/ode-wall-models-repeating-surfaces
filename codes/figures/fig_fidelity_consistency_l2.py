#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fig_fidelity_consistency_l2.py -- L2 figure for the firmed fidelity-consistency
criterion.  All numbers from codes/results/rib_fidelity_consistency_l2.npz
(a-priori, Y_IDX=10), produced by rib_fidelity_consistency_l2.py.  No fabrication.

Three panels:
  (a) matched-geometry verdict flip with the MEASURED resolved fraction f_res
      (B-L2-1): RANS reference certifies (R^2>0), WRLES reference fails (R^2<0),
      same geometry and Re.
  (b) closure-independence substitution on the d-type rib (B-L2-5/G3): the exact
      resolved + SGS-completed LES stress does NOT rescue the ODE.
  (c) the co-failure window (B-L2-2): R^2 of the ODE against each RANS reference
      vs pitch p/k; masking is read only at the matched p/k=3 (WRLES disagrees),
      while at the tightest p/k=2 the ODE fails against RANS itself (no mask).

Run: OMP_NUM_THREADS=2 python3 codes/figures/fig_fidelity_consistency_l2.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
NODE = os.path.abspath(os.path.join(HERE, "..", "..", "development", "nodes", "node_002"))

# model-type colour scheme (project convention)
C_TRUTH = "#E69F00"     # orange  -- high-fidelity reference / measured
C_BLACK = "#000000"     # black   -- black-box ODE
C_GRAY = "#7A8B99"      # bluish-gray -- gray-box / RANS pilot
C_GOOD = "#117733"      # green   -- tolerated / pass


def main():
    d = np.load(os.path.join(RESULTS, "rib_fidelity_consistency_l2.npz"),
                allow_pickle=True)
    fid = json.loads(str(d["measured_fidelity_json"]))
    win = json.loads(str(d["cofailure_window_json"]))
    mt = win["matched"]

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # ---- (a) matched-geometry verdict flip with measured f_res ----
    refs = ["RANS\n($f_{res}=0$)", "WRLES\n($f_{res}=%.2f$)" % mt["f_res_les"]]
    vals = [mt["r2_rans"], mt["r2_les"]]
    cols = [C_GRAY, C_TRUTH]
    bars = ax[0].bar(refs, vals, color=cols, edgecolor="k", width=0.6)
    ax[0].axhline(0, color="k", lw=0.8)
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2,
                   v + (0.06 if v >= 0 else -0.06), "%+.2f" % v,
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    ax[0].set_ylabel(r"$R^2(\tau_w)$ of the ODE", fontsize=11)
    ax[0].set_title("(a) Same geometry ($p/k=3$, matched $Re$):\n"
                    "fidelity flips the verdict", fontsize=10.5)
    ax[0].set_ylim(-1.3, 1.0)
    ax[0].text(0.5, 0.90, r"$\Delta R^2=%+.2f$" % mt["delta_r2"],
               transform=ax[0].transAxes, ha="center", fontsize=10,
               bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    # ---- (b) closure-independence substitution ----
    labels = ["standard\nclosure", "+ exact\nresolved", "+ resolved\n+ SGS"]
    rs = [fid["standard_ml_r2"], fid["controlled_dns_r2"],
          fid["controlled_dns_total_r2"]]
    bars = ax[1].bar(labels, rs, color=[C_BLACK, C_TRUTH, C_TRUTH],
                     edgecolor="k", width=0.6)
    ax[1].axhline(0, color="k", lw=0.8)
    for b, v in zip(bars, rs):
        ax[1].text(b.get_x() + b.get_width() / 2, v - 0.2, "%.2f" % v,
                   ha="center", va="top", fontsize=11, color="white"
                   if v < -1 else "k")
    ax[1].set_ylabel(r"$R^2(\tau_w)$ on the $d$-type rib", fontsize=11)
    ax[1].set_title("(b) Exact LES stress does NOT rescue\n"
                    "(closure-independent failure)", fontsize=10.5)
    ax[1].set_ylim(-6.4, 0.4)

    # ---- (c) co-failure window ----
    pk = np.asarray(d["sweep_pk"], float)
    r2 = np.asarray(d["sweep_r2"], float)
    o = np.argsort(pk)
    pk, r2 = pk[o], r2[o]
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].plot(pk, r2, "o-", color=C_GRAY, mfc=C_GRAY, mec="k", ms=8,
               label="ODE vs RANS reference")
    # matched p/k=3: WRLES disagrees (masking)
    ax[2].plot([3], [mt["r2_les"]], "*", color=C_TRUTH, mec="k", ms=18,
               label="WRLES at $p/k=3$ (truth)")
    ax[2].annotate("RANS masks\n(spurious pass)", xy=(3, mt["r2_rans"]),
                   xytext=(3.6, 0.55), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[2].annotate("$p/k=2$: ODE fails\nvs RANS too\n(no mask)", xy=(2, r2[0]),
                   xytext=(2.0, -0.55), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[2].set_xlabel(r"pitch ratio $p/k$", fontsize=11)
    ax[2].set_ylabel(r"$R^2(\tau_w)$", fontsize=11)
    ax[2].set_title("(c) Co-failure window:\nRANS $can$ mask, not always",
                    fontsize=10.5)
    ax[2].legend(fontsize=8, loc="lower right")
    ax[2].set_ylim(-1.5, 1.0)

    fig.tight_layout()
    os.makedirs(NODE, exist_ok=True)
    for out in [os.path.join(NODE, "fig_fidelity_consistency_l2.png"),
                os.path.join(NODE, "fig_fidelity_consistency_l2.pdf")]:
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print("wrote", os.path.relpath(out, os.path.join(HERE, "..", "..")))


if __name__ == "__main__":
    main()
