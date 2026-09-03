#!/usr/bin/env python3
"""Plot the terminal Re_H=19,000 M13 validation subset.

The experiment has no wall-traction measurement.  The figure therefore shows
only quantities that can be validated: station-wise mean-velocity error and
the reattachment event inferred from the lowest valid PIV point.  G1c/G2c are
displayed as a two-grid sensitivity pair, never as a convergence sequence.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
SUMMARY = RESULTS / "m13_highre_coupled_20260824_summary.json"
ARRAYS = RESULTS / "m13_highre_coupled_20260824.npz"
OUT = ROOT / "manuscript" / "figures"

ORANGE = "#E69F00"       # experiment / reference
GRAY = "#56708A"         # total-gradient TBLE
GREEN = "#1B7837"        # equilibrium Spalding
BLACK = "#000000"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    record = summary["campaigns"]["19000"]
    arrays = np.load(ARRAYS, allow_pickle=False)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 2.75),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    models = (("equilibrium", "Equilibrium", GREEN),
              ("total_gradient_tble", "Total-gradient TBLE", GRAY))
    grids = (("G1c", "307,200 cells", "--", "s", "none"),
             ("G2c", "819,200 cells", "-", "o", None))
    for model, label, colour in models:
        for grid, grid_label, line, marker, face in grids:
            stem = f"re19000_{grid}_{model}_rapp_19000"
            x = arrays[f"{stem}_station_x"]
            rms = arrays[f"{stem}_station_u_rms"]
            ax0.plot(x, rms, line, color=colour, marker=marker, ms=4,
                     mfc=("white" if face == "none" else colour), mec=colour,
                     lw=1.25)
    ax0.axhline(0.12, color=BLACK, ls=":", lw=1.0,
                zorder=0)
    ax0.set(xlabel=r"$x/H$", ylabel=r"station $E_U$",
            xlim=(0, 8.2), ylim=(0, 0.13),
            title=r"(a) Mean velocity against Rapp PIV")
    ax0.legend(handles=[
        Line2D([], [], color=GREEN, lw=1.5, label="Equilibrium"),
        Line2D([], [], color=GRAY, lw=1.5, label="Total-gradient TBLE"),
        Line2D([], [], color=BLACK, lw=1.0, ls=":",
               label=r"registered $0.12U_b$ bound"),
    ], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False,
       columnspacing=0.8, handlelength=2.0)
    ax0.text(0.98, 0.05, "open dashed: G1c (307,200)\nfilled solid: G2c (819,200)",
             transform=ax0.transAxes, ha="right", va="bottom", fontsize=6.8)

    exp = record["experimental_reattachment"]
    ax1.axhspan(*exp["bracket_x_over_H"], color=ORANGE, alpha=0.16,
                label="PIV sign-change bracket")
    ax1.axhline(exp["estimate_x_over_H"], color=ORANGE, lw=1.4,
                label="PIV linear estimate")
    xloc = {"equilibrium": 0.0, "total_gradient_tble": 1.0}
    for model, _, colour in models:
        ys = [record["metrics"][f"{grid}:{model}"]["reattachment_x_over_H"]
              for grid, *_ in grids]
        ax1.plot([xloc[model] - 0.07, xloc[model] + 0.07], ys,
                 color=colour, lw=1.0)
        ax1.plot(xloc[model] - 0.07, ys[0], marker="s", ms=5,
                 mfc="white", mec=colour, color=colour)
        win = record["averaging"][f"G2c:{model}"]
        window_change = abs(win["270"]["reattachment_x_over_H"]
                            - win["225"]["reattachment_x_over_H"])
        ax1.errorbar(xloc[model] + 0.07, ys[1], yerr=window_change,
                     marker="o", ms=5, mfc=colour, mec=colour,
                     ecolor=colour, elinewidth=0.9, capsize=2, ls="none")
    ax1.set_xticks([0, 1], ["Equilibrium", "Total-gradient\nTBLE"])
    ax1.set(ylabel=r"reattachment $x_r/H$", ylim=(3.0, 4.25),
            title=r"(b) Event location; two-grid sensitivity")
    ax1.text(0.98, 0.04, "squares: G1c\ncircles: G2c\nbars: 225--270 window",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=6.8)
    ax1.legend(loc="upper left", frameon=False)

    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT / f"fig_m13_re19000_validation.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"saved {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
