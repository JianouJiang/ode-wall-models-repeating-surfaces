#!/usr/bin/env python3
r"""
fig_rib_coverage_l3.py
======================
LEVEL 3 results figure: the d-type vs k-type sharp-rib divergence is a
deep-cancellation COVERAGE effect, not a median-depth or a recirculation-length
effect.  All numbers from results/transition_map_l3_results.npz (read-only;
no fabrication).

 (a)  The three sharp square ribs ordered by pitch lambda/delta.  Median
      cancellation depth eps_med (open bars) is DEGENERATE -- it is identical
      (0.521) for the catastrophic d-type LES rib and the tolerated k-type RANS
      rib -- this is the 2.0e-4-decade "razor-thin" gap the L2 Judge flagged.
      The deep-cancellation COVERAGE frac[eps<0.1] (filled bars) instead
      separates them by a factor 2.2 and orders all three ribs monotonically
      with the ODE skill R^2 (annotated): wider pitch -> less of the inter-rib
      gap held in deep force cancellation -> ODE tolerated.

 (b)  Per-station cancellation parameter eps(x) across one pitch.  The deep
      regime eps<0.1 (shaded) -- where the closure-independent bound makes the
      ODE catastrophic -- covers a LARGER fraction of the narrow d-type pitch
      than of the wide k-type pitch.  The coverage of this band, not the
      reversed-flow (recirculation) fraction, is what tracks the failure.

Colours: warm (#d7301f) = catastrophic (R^2<0), cool (#4575b4) = tolerated.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NPZ = os.path.join(RESULTS, "transition_map_l3_results.npz")
MS_FIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

FAIL_C = "#d7301f"
PASS_C = "#4575b4"

LABELS = {
    "rib_rans_dtype": "d-type\n$\\lambda/\\delta=0.4$ (RANS)",
    "rib_les_dtype":  "d-type\n$\\lambda/\\delta=0.6$ (LES)",
    "rib_rans_ktype": "k-type\n$\\lambda/\\delta=1.6$ (RANS)",
}


def main():
    d = np.load(NPZ, allow_pickle=True)
    rib_keys = [str(k) for k in d["rib_keys"]]
    lam = np.asarray(d["rib_lambda"], float)
    eps_med = np.asarray(d["rib_eps_med"], float)
    frac01 = np.asarray(d["rib_frac01"], float)
    r2 = np.asarray(d["rib_r2"], float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # ===================== panel (a): depth degenerate, coverage discriminates
    x = np.arange(len(rib_keys))
    w = 0.38
    cols = [FAIL_C if rr < 0 else PASS_C for rr in r2]
    # open bars: median depth eps_med (degenerate)
    axA.bar(x - w / 2, eps_med, w, facecolor="none", edgecolor="0.35",
            hatch="//", linewidth=1.1, label=r"median depth $\tilde\varepsilon$")
    # filled bars: deep-cancellation coverage frac[eps<0.1] (discriminates)
    axA.bar(x + w / 2, frac01, w, color=cols, edgecolor="k", linewidth=1.0,
            label=r"coverage $f(\varepsilon<0.1)$")
    for xi, rr, fr in zip(x, r2, frac01):
        axA.annotate(rf"$R^2={rr:+.2f}$", xy=(xi + w / 2, fr),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8.2,
                     color=(FAIL_C if rr < 0 else PASS_C), fontweight="bold")
    # annotate the degeneracy of median depth between LES d-type & k-type
    axA.annotate(r"$\tilde\varepsilon$ degenerate (0.521$\equiv$0.521,"
                 "\n$2{\\times}10^{-4}$ dec)",
                 xy=(1 - w / 2, eps_med[1]), xytext=(0.55, 0.72),
                 fontsize=7.6, ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
    axA.set_xticks(x)
    axA.set_xticklabels([LABELS[k] for k in rib_keys], fontsize=8)
    axA.set_ylabel(r"value")
    axA.set_ylim(0, 0.85)
    axA.set_title(r"(a)  depth is degenerate; coverage discriminates",
                  loc="left", fontsize=10)
    axA.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axA.grid(axis="y", alpha=0.25)

    # ===================== panel (b): per-station eps over one pitch =========
    axB.axhspan(1e-4, 0.1, color="0.86", zorder=0,
                label=r"deep cancellation $\varepsilon<0.1$")
    for k in rib_keys:
        xop = np.asarray(d[f"perst_{k}_x"], float)
        epsx = np.asarray(d[f"perst_{k}_eps"], float)
        rr = r2[rib_keys.index(k)]
        fr = frac01[rib_keys.index(k)]
        order = np.argsort(xop)
        c = FAIL_C if rr < 0 else PASS_C
        ls = "-" if rr < 0 else "--"
        axB.plot(xop[order], epsx[order], ls, color=c, lw=1.5,
                 marker="o", ms=3, alpha=0.85,
                 label=rf"{LABELS[k].splitlines()[0]} "
                       rf"$\lambda/\delta={lam[rib_keys.index(k)]:.1f}$, "
                       rf"$f(\varepsilon<0.1)={fr:.3f}$")
    axB.set_yscale("log")
    axB.set_ylim(1e-3, 1e2)
    axB.set_xlim(0, 1)
    axB.set_xlabel(r"streamwise position within one pitch  $x/\lambda$")
    axB.set_ylabel(r"$\varepsilon(x)=|\tau_w|/(|dp/dx|\,y_m)$")
    axB.set_title(r"(b)  deep-cancellation coverage over the pitch",
                  loc="left", fontsize=10)
    axB.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axB.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(MS_FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_rib_coverage_l3.{ext}"),
                    dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(MS_FIG, f"fig_rib_coverage_l3.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("Saved fig_rib_coverage_l3.{pdf,png} -> codes/figures + manuscript/figures")


if __name__ == "__main__":
    main()
