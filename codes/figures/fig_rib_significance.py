#!/usr/bin/env python3
r"""
Node-local L3 diagnostic figure (NOT a manuscript figure -- kept out of the PDF
to respect the page-count bind B-L3-5): is the sharp-rib vs smooth-hill
cancellation-completeness gap sampling noise or a systematic offset (B-L3-1)?

Three panels, all from cancellation_completeness_significance.npz:
  (a) the two failure-band C distributions (hill n=504, rib n=37) with medians
      and IQRs -- the rib sits modestly low but overlaps the hill body.
  (b) BOTH autocorrelations: the rib (lag1~0.66, tau_int~3, n_eff~13) AND the
      hill (lag1~0.82, tau_int~41, n_eff~12) pools are equally non-independent,
      so an i.i.d. bootstrap on EITHER side is anti-conservative -- both must be
      block-resampled (B-L4-1).
  (c) the SYMMETRIC block-block bootstrap distribution of Delta = C_hill - C_rib
      vs the i.i.d. reference; the 95% CI excludes zero but the 99% CI includes
      it -> a small, threshold-sensitive ~12% systematic geometry offset (honest
      concession), while both medians stay O(0.30).

Writes: codes/figures/fig_rib_significance.{pdf,png}
        development/nodes/node_005/fig_rib_significance.png
Run:  OMP_NUM_THREADS=2 python3 codes/figures/fig_rib_significance.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
NODE = os.path.join(os.path.dirname(CODES), "development", "nodes", "node_005")

d = np.load(os.path.join(RES, "cancellation_completeness_significance.npz"),
            allow_pickle=True)
H, R = d["hill_C"], d["rib_C"]
H_med, R_med = float(d["hill_median"]), float(d["rib_median"])
H_iqr, R_iqr = d["hill_iqr"], d["rib_iqr"]
d_iid, d_blk = d["d_iid"], d["d_blk"]
ci_blk, ci_iid = d["delta_ci_block"], d["delta_ci_iid"]
ci_blk99 = d["delta_ci_block_99"]
p_blk = float(d["delta_p_block"])
rlag1, rtau, rn_eff, LR = float(d["rib_lag1"]), float(d["rib_tau_int"]), float(d["rib_n_eff"]), int(d["rib_block_len"])
hlag1, htau, hn_eff, LH = float(d["hill_lag1"]), float(d["hill_tau_int"]), float(d["hill_n_eff"]), int(d["hill_block_len"])

HILL_C, RIB_C = "darkorange", "crimson"
plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.8,
                     "mathtext.fontset": "cm", "font.family": "serif"})
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.2, 3.5))
fig.subplots_adjust(wspace=0.33)

# (a) the two distributions
bins = np.linspace(0, 0.8, 33)
axA.hist(H, bins=bins, density=True, color=HILL_C, alpha=0.45, label=f"hills (n={H.size})")
axA.hist(R, bins=bins, density=True, color=RIB_C, alpha=0.45, label=f"rib (n={R.size})")
axA.axvline(H_med, color=HILL_C, lw=2)
axA.axvline(R_med, color=RIB_C, lw=2)
axA.axvspan(H_iqr[0], H_iqr[1], color=HILL_C, alpha=0.10)
axA.set_xlabel(r"failure-band $C=|\Delta\tau_w|/\Phi$")
axA.set_ylabel("density")
axA.set_title("(a) failure-band completeness", fontsize=9.5, loc="left")
axA.legend(fontsize=8, frameon=False)
axA.text(0.97, 0.74, f"hill med {H_med:.3f}\nrib med {R_med:.3f}\n"
         f"gap {100*(H_med-R_med)/H_med:.0f}%", transform=axA.transAxes,
         ha="right", va="top", fontsize=8.0)

# (b) BOTH autocorrelations -- hill is as correlated as the rib (B-L4-1)
def _ac(a):
    x = np.asarray(a, float) - np.mean(a)
    c = np.correlate(x, x, "full"); c = c[c.size // 2:]; return c / c[0]
acR, acH = _ac(R), _ac(H)
lags = np.arange(min(10, acR.size, acH.size))
axB.bar(lags - 0.18, acH[:lags.size], color=HILL_C, alpha=0.8, width=0.36,
        label=f"hill ($n_{{\\rm eff}}\\approx${hn_eff:.0f})")
axB.bar(lags + 0.18, acR[:lags.size], color=RIB_C, alpha=0.8, width=0.36,
        label=f"rib ($n_{{\\rm eff}}\\approx${rn_eff:.0f})")
axB.axhline(0, color="0.4", lw=0.8)
axB.set_xlabel("lag (wall stations)")
axB.set_ylabel("$C$ autocorrelation")
axB.set_title("(b) both pools are autocorrelated", fontsize=9.5, loc="left")
axB.legend(fontsize=7.6, frameon=False, loc="upper right")
axB.text(0.95, 0.66, f"hill lag1 {hlag1:.2f}, " r"$\tau_{\rm int}$" f" {htau:.0f}\n"
         f"rib  lag1 {rlag1:.2f}, " r"$\tau_{\rm int}$" f" {rtau:.0f}",
         transform=axB.transAxes, ha="right", va="top", fontsize=7.8)

# (c) bootstrap Delta
bins2 = np.linspace(min(d_iid.min(), d_blk.min()), max(d_iid.max(), d_blk.max()), 40)
axC.hist(d_iid, bins=bins2, density=True, color="0.6", alpha=0.55,
         label="i.i.d. (both, ref)")
axC.hist(d_blk, bins=bins2, density=True, color=RIB_C, alpha=0.45,
         label=f"block (both, $L$={LH}/{LR})")
axC.axvline(0, color="k", lw=1.2, ls="--")
axC.axvline(ci_blk[0], color=RIB_C, lw=1.2, ls=":")
axC.axvline(ci_blk[1], color=RIB_C, lw=1.2, ls=":")
axC.axvline(ci_blk99[0], color="0.3", lw=1.0, ls="-.")
axC.set_xlabel(r"$\Delta = C_{\rm hill}-C_{\rm rib}$")
axC.set_ylabel("density")
axC.set_title("(c) small, threshold-sensitive offset", fontsize=9.5, loc="left")
axC.legend(fontsize=7.6, frameon=False, loc="upper right")
axC.text(0.03, 0.70, f"block 95% CI\n[{ci_blk[0]:.3f}, {ci_blk[1]:.3f}]\n"
         f"excludes 0 ($p$={p_blk:.3f})\n"
         f"99% CI [{ci_blk99[0]:.3f}, {ci_blk99[1]:.3f}]\nincludes 0",
         transform=axC.transAxes, ha="left", va="top", fontsize=7.6)

fig.savefig(os.path.join(HERE, "fig_rib_significance.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_rib_significance.png"), dpi=160, bbox_inches="tight")
os.makedirs(NODE, exist_ok=True)
fig.savefig(os.path.join(NODE, "fig_rib_significance.png"), dpi=160, bbox_inches="tight")
print("wrote fig_rib_significance.{pdf,png} to codes/figures and node_005")
print(f"  hill {H_med:.3f}  rib {R_med:.3f}  block-CI Delta [{ci_blk[0]:.3f},{ci_blk[1]:.3f}]")
