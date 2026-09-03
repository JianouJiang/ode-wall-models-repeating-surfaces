#!/usr/bin/env python3
r"""
Figure: decomposing the a-priori -> a-posteriori wall-stress "healing" on
periodic hills into (i) the matching-height (y_m) effect and (ii) the genuine
coupling-feedback effect.

Reads ONLY codes/results/ym_feedback_decomposition.npz (produced by
codes/analysis/ym_feedback_decomposition.py from real DNS + coupled-run data).

Panel (a): a-priori R^2(tau_w) of the SAME production ODE model as a continuous
function of the matching height y_m/H (the sweep). The a-priori protocol height
and the coupled WMLES first-cell band are marked. The curve shows the structural
error is ~linear in y_m: the coupled run sits ~7x lower, where the geometric
1/y_m factor alone lifts R^2 from -48 toward 0 -- WITHOUT curing the defect
(R^2 is still < 0 at the coupled height).

Panel (b): the transmission ladder as a waterfall: Rung0 (a-priori, DNS-protocol
y_m) -> +y_m effect -> Rung1 (a-priori model @ coupled y_m) -> +feedback ->
Rung2 (fully coupled). The y_m effect dominates (77-92% of the healing); coupling
feedback is the smaller, real term that finally tips R^2 positive.

Colour convention: orange = ground-truth axis, black = black-box ODE model,
green = equilibrium/coupled wall model.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")
FIG_DIR = os.path.join(ROOT, "manuscript", "figures")

C_BLACK = "#000000"   # black-box ODE model
C_COUP = "#2ca02c"    # green — coupled / equilibrium wall model
C_BAND = "#9ecae1"    # bluish — coupled matching-height band


def main():
    d = np.load(os.path.join(RESULTS, "ym_feedback_decomposition.npz"))
    ym_sweep = d["sweep_ym"]
    R2_sweep = d["sweep_R2"]
    R0, R1, R2c = float(d["R2_rung0_apriori"]), float(d["R2_rung1_coupled_ym"]), \
        float(d["R2_rung2_coupled"])
    ym_ap = float(d["ym_apriori_median"])
    ym_cp = float(d["ym_coupled_median"])
    ym_cp_lo, ym_cp_hi = float(d["ym_coupled_on_x"].min()), \
        float(d["ym_coupled_on_x"].max())
    dy, dfb = float(d["delta_ym"]), float(d["delta_feedback"])
    fy, ffb = float(d["frac_ym"]), float(d["frac_feedback"])
    fy_rr, ffb_rr = float(d["frac_ym_relrms"]), float(d["frac_feedback_relrms"])

    # station-matched control (L3 fix for the 512-vs-80 confound)
    sm = np.load(os.path.join(RESULTS, "station_matched_decomposition.npz"),
                 allow_pickle=True)
    R1_512 = float(sm["R2_rung1_512"])
    R1_80 = float(sm["R2_rung1_80"])
    R2_80 = float(sm["R2_rung2_80"])
    d_samp = float(sm["delta_sampling"])
    d_fb_m = float(sm["delta_feedback_matched"])
    ff_m = float(sm["frac_feedback_matched"])
    ff_L2 = float(sm["frac_feedback_L2"])

    # Single continuous-curve panel: the entire a-priori -> a-posteriori
    # "healing" rendered ON the R^2(y_m) curve as annotated points + arrows
    # (no bars / no waterfall). Rung0 and Rung1 sit on the a-priori curve; the
    # coupling feedback is the vertical jump from the curve up to Rung2.
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.0))

    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvspan(ym_cp_lo, ym_cp_hi, color=C_BAND, alpha=0.45, zorder=0,
               label="coupled first-cell band")
    ax.plot(ym_sweep, R2_sweep, "-", color=C_BLACK, lw=2.0, zorder=3,
            label=r"a-priori $R^2(\tau_w)$, ODE model (sweep)")

    # Rung 0 sits far down-right on the curve; Rungs 1 & 2 are nearly
    # coincident in y_m near the wall, so they go in a zoom inset.
    ax.plot([ym_ap], [R0], "s", color=C_BLACK, ms=9, zorder=5)
    ax.annotate(f"Rung 0: a-priori protocol, $R^2={R0:.0f}$", (ym_ap, R0),
                textcoords="offset points", xytext=(12, 6), ha="left",
                fontsize=9)

    # "y_m effect": travelling along the curve from the coupled cell to the
    # a-priori protocol height (a guide arrow following the curve)
    ax.annotate("", xy=(ym_ap, R0), xytext=(ym_cp, R1),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.6),
                zorder=4)
    ax.text(0.118, -30.0,
            f"$y_m$ effect: $\\Delta R^2={dy:+.0f}$\n"
            f"({100*fy:.0f}% of the healing --\n"
            f"the geometric $1/y_m$ factor,\nnot a cure)",
            ha="center", va="center", fontsize=9, color="0.30")

    # ---- zoom inset on the coupled near-wall band: Rung 1 -> Rung 2 -------
    axin = ax.inset_axes([0.40, 0.55, 0.40, 0.40])
    axin.axhline(0, color="0.6", lw=0.8)
    axin.axvspan(ym_cp_lo, ym_cp_hi, color=C_BAND, alpha=0.45)
    mask = ym_sweep <= 0.03
    axin.plot(ym_sweep[mask], R2_sweep[mask], "-", color=C_BLACK, lw=1.6)
    axin.plot([ym_cp], [R1], "o", color=C_COUP, ms=8, mfc="white",
              mec=C_COUP, mew=1.8, zorder=5)
    axin.plot([ym_cp], [R2c], "o", color=C_COUP, ms=9, zorder=5)
    axin.annotate("", xy=(ym_cp, R2c), xytext=(ym_cp, R1),
                  arrowprops=dict(arrowstyle="-|>", color=C_COUP, lw=2.0))
    axin.text(ym_cp + 0.0016, 0.5 * (R1 + R2c),
              f"coupling\nfeedback\n$\\Delta R^2={dfb:+.1f}$\n({100*ffb:.0f}%)",
              ha="left", va="center", fontsize=8, color=C_COUP)
    axin.text(ym_cp, R1 - 0.5, f"Rung 1: $R^2={R1:.1f}$\n(still $<0$: not cured)",
              ha="center", va="top", fontsize=7.5, color=C_COUP)
    axin.text(ym_cp - 0.0016, R2c + 0.3, f"Rung 2: coupled WMLES $R^2={R2c:+.2f}$",
              ha="left", va="bottom", fontsize=7.5, color=C_COUP)
    axin.set_xlim(0, 0.026)
    axin.set_ylim(R1 - 2.0, 2.2)
    axin.set_title("zoom: coupled near-wall band", fontsize=8)
    axin.tick_params(labelsize=7)
    ax.indicate_inset_zoom(axin, edgecolor="0.5")

    ax.set_xlabel(r"matching height $y_m/H$")
    ax.set_ylabel(r"$R^2$ (wall stress $\tau_w$ / coupled $C_f$)")
    ax.set_title(r"The a-priori$\to$a-posteriori healing is mostly the "
                 r"geometric $1/y_m$ factor, not a cure", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, ym_sweep.max())
    ax.set_ylim(R0 - 6, 6)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "fig_ym_decomposition.pdf")
    out_png = os.path.join(FIG_DIR, "fig_ym_decomposition.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"  y_m effect: dR2={dy:+.1f} ({100*fy:.0f}% R2 / {100*fy_rr:.0f}% relRMS)")
    print(f"  feedback  : dR2={dfb:+.1f} ({100*ffb:.0f}% R2 / {100*ffb_rr:.0f}% relRMS)")


if __name__ == "__main__":
    main()
