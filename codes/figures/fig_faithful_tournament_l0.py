#!/usr/bin/env python3
"""Figure: faithful published families, the shape/amplitude factorial, and the
geometry holdout.

(a) every faithfully implemented published family and the two candidate
    regularisers on the paper's a-priori surface, with paired phase-block
    intervals, against the primary reference and the bracket;
(b) the exact two-by-two factorial -- source shape crossed with source
    amplitude -- which is the experiment that separates "how much" from "what";
(c) the frozen operators on a different repeating wall, across three grids and
    four matching heights.

Every value is read from the deposited artifacts; nothing is recomputed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260825"
TOURNAMENT = ROOT / f"codes/results/faithful_tournament_l0_{STAMP}.json"
HOLDOUT = ROOT / f"codes/results/wavy_geometry_holdout_l0_{STAMP}.json"
OUT = ROOT / "manuscript/figures/fig_faithful_tournament_l0"
PRIMARY = "archive_index10"

LABELS = {
    "M0_equilibrium": "equilibrium law",
    "M1_pressure_gradient": "pressure-gradient ODE",
    "M2_hickel": "parametrised-convection ODE",
    "M2_hickel_Aplus26_variant": "parametrised convection (other damping)",
    "M3_yang_integral": "integral momentum model",
    "M4_park_moin": "non-equilibrium wall layer",
    "M5_meneveau": "generalised-Moody law",
    "Xall": "measured transport, in full",
    "Xc_exact_convection": "measured convection only",
    "Xcp_pressure_plus_convection": "measured pressure and convection",
    "Xcpr_plus_normal_stress": "measured, plus normal stress",
    "ORACLE_closure_free": "closure-free identity (oracle)",
    "FAC_exactshape_modelnorm": "measured shape at modelled amplitude",
    "FAC_modelshape_exactnorm": "modelled shape at measured amplitude",
}
GREY = "#4d4d4d"
BLUEGREY = "#6b7f95"
ORANGE = "#e07b39"
GREEN = "#3f8f52"


def label_of(name: str, c_star: dict) -> str:
    if name in LABELS:
        return LABELS[name]
    if name.startswith("NLWH_"):
        host = "measured transport" if "_Xall_" in name else "pressure gradient"
        return f"source horizon on {host}"
    if name.startswith("NLWM_"):
        host = "measured transport" if "_Xall_" in name else "pressure gradient"
        return f"uniform norm limit on {host}"
    return name


def main() -> int:
    T = json.loads(TOURNAMENT.read_text())
    prim = T["surfaces"][PRIMARY]
    c_star = T["c_star"]
    scores_B = prim["scores"]["B_mglet"]
    scores_C = prim["scores"]["C_xiao_repaired_cubic6"]

    # the integral family is deliberately absent: the vertically integrated
    # momentum identity does not close to the wall-traction scale on this
    # archive even with the measured profile, so it is not adjudicated here
    published = [a for a in ("M0_equilibrium", "M5_meneveau",
                             "M1_pressure_gradient", "M2_hickel",
                             "M4_park_moin")
                 if a in scores_B]
    instruments = [a for a in ("Xc_exact_convection", "Xall",
                               "ORACLE_closure_free") if a in scores_B]
    candidates = [a for a in scores_B
                  if (a.startswith("NLWH_") or a.startswith("NLWM_"))]
    order = published + candidates + instruments

    fig = plt.figure(figsize=(7.2, 8.4))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.0, 1.0], hspace=0.55)

    # ---------------- (a) the ladder ------------------------------------- #
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(order))[::-1]
    for k, name in enumerate(order):
        m = scores_B[name]
        colour = (GREEN if name in candidates else
                  ORANGE if name == "ORACLE_closure_free" else
                  BLUEGREY if name in instruments else GREY)
        ax.plot([m["interval"]["low"], m["interval"]["high"]], [y[k], y[k]],
                color=colour, lw=2.0, solid_capstyle="butt", alpha=0.75)
        ax.plot(m["relative_rms"], y[k], "o", color=colour, ms=5.0, zorder=3)
        if name in scores_C:
            ax.plot(scores_C[name]["relative_rms"], y[k], "|", color=colour,
                    ms=9, mew=1.4, zorder=3)
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([label_of(a, c_star) for a in order], fontsize=7.6)
    ax.set_xlabel(r"relative wall-traction error $E_\tau$"
                  "\n(circle: full-wall reference; tick: curvature-aware bracket)",
                  fontsize=8)
    ax.set_xscale("log")
    ax.set_title("(a)  faithfully implemented families and the two candidates",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=7.6)

    # ---------------- (b) the factorial ----------------------------------- #
    ax = fig.add_subplot(grid[1])
    factorial = prim["shape_amplitude_factorial"]
    cells = factorial["cells_relative_rms"]
    norms = factorial["cells_source_norm"]
    modelled = norms["M2_hickel"]
    measured = norms["Xall"]
    ax.plot([modelled, measured],
            [cells["M2_hickel"], cells["FAC_modelshape_exactnorm"]],
            "-o", color=GREY, lw=1.6, ms=5, label="modelled source shape")
    ax.plot([modelled, measured],
            [cells["FAC_exactshape_modelnorm"], cells["Xall"]],
            "-s", color=BLUEGREY, lw=1.6, ms=5, label="measured source shape")
    for x, value, text in ((modelled, cells["M2_hickel"], "published surrogate"),
                           (measured, cells["Xall"], "exact completion")):
        ax.annotate(text, (x, value), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"assembled source norm $N$", fontsize=8)
    ax.set_ylabel(r"$E_\tau$", fontsize=8)
    ax.legend(fontsize=7.4, frameon=False)
    ax.set_title("(b)  source shape crossed with source amplitude, at exactly "
                 "matched norms", fontsize=9, loc="left")
    ax.tick_params(labelsize=7.6)

    # ---------------- (c) the geometry holdout ---------------------------- #
    ax = fig.add_subplot(grid[2])
    if HOLDOUT.exists():
        H = json.loads(HOLDOUT.read_text())
        marks = {"G0": "o", "G1": "s", "G2": "D"}
        for grid_name, entry in sorted(H["grids"].items()):
            heights, candidate, best = [], [], []
            for key, record in sorted(entry["matching_heights"].items()):
                heights.append(float(key))
                candidate.append(record["candidate_relative_rms"])
                best.append(record["best_published_relative_rms"])
            ax.plot(heights, candidate, marks.get(grid_name, "o"), ls="-",
                    color=GREEN, ms=4.5, lw=1.2,
                    label=f"source horizon ({grid_name})")
            ax.plot(heights, best, marks.get(grid_name, "o"), ls="--",
                    color=GREY, ms=4.5, lw=1.2,
                    label=f"best published family ({grid_name})")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"matching height $\eta_m/\delta$", fontsize=8)
        ax.set_ylabel(r"$E_\tau$", fontsize=8)
        ax.legend(fontsize=6.6, frameon=False, ncol=2)
        ax.set_title("(c)  a different repeating wall, constants frozen on the "
                     "hill", fontsize=9, loc="left")
        ax.tick_params(labelsize=7.6)
    else:
        ax.axis("off")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT.with_suffix(".pdf").name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
