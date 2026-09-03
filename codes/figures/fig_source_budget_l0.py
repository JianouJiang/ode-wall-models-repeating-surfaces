#!/usr/bin/env python3
"""Source-norm budget: the a-priori tournament, the two interventions and the
surface dependence.

(a) the norm plane.  Absolute wall-traction error against the assembled source
    norm N for every arm at the paper's a-priori surface.  The straight line is
    fitted ONLY on the amplitude-sweep arms, where the physics is held fixed and
    the norm is moved by rescaling; the published wall-model families and the
    exact-completion rungs are out-of-sample points on it.
(b) the two interventions.  Paired phase-block 95% intervals for a rigid phase
    shift of the source, applied to the modelled source and to the exact one.
    A shift leaves the norm and the term composition alone and destroys only the
    correspondence between the source and the station it is used at.
(c) the surface dependence.  How much of the assembled norm returns as error, on
    the flat inter-hill floor and on the curved flanks of the same simulation.

Colour convention: orange = the reference/fitted law, green = the algebraic
published families, bluish grey = modelled-source families, black = arms built
from exact measured transport.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
STAMP = "20260825"
SURFACE = "archive_index10"
REF = "B_mglet"
ORANGE, GRAY, GREEN, BLACK = "#e07b1a", "#6b7c93", "#2e8b57", "#111111"

FAMILIES = {
    "M0_equilibrium": ("equilibrium (Spalding)", GREEN, "s"),
    "M5_meneveau": ("generalised Moody", GREEN, "D"),
    "M1_pressure_gradient": ("pressure-gradient ODE", GRAY, "o"),
    "M2_hickel": ("modelled convection", GRAY, "^"),
    "M4_park_moin": ("non-equilibrium PDE", BLACK, "v"),
    "M3_yang_integral": ("integral momentum", BLACK, "P"),
}
COMPLETION = {
    "Xc_exact_convection": "+ exact convection",
    "Xcp_pressure_plus_convection": "+ exact pressure profile",
    "Xcpr_plus_normal_stress": "+ normal-stress gradient",
    "Xall": "all omitted transport",
    "Xfull_closure_free": "no closure at all",
}


def main() -> int:
    tj = json.loads((RESULTS / f"source_budget_tournament_l0_{SURFACE}_{STAMP}.json").read_text())
    rj = json.loads((RESULTS / f"source_budget_regions_l0_{STAMP}.json").read_text())
    scores = tj["scores"][REF]
    norms = tj["source_norm"]
    law = tj["norm_law"][REF]["affine_norm_law"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.5,
        "axes.titlesize": 8.6, "legend.fontsize": 6.4, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.75), constrained_layout=True)

    # ---------------- (a) the norm plane -----------------------------------
    a = ax[0]
    fit_arms = law["fitted_on"]
    xs = np.array([norms[k]["N_rms"] for k in fit_arms])
    ys = np.array([scores[k]["absolute_rms"] for k in fit_arms])
    a.plot(xs, ys, ls="none", marker="x", ms=5, mew=1.1, color=ORANGE,
           label="amplitude sweep (the fit set)")
    grid = np.linspace(0.0, max(xs.max(), 0.026) * 1.05, 64)
    a.plot(grid, law["E0"] + law["delta"] * grid, color=ORANGE, lw=1.3,
           label=rf"$E_{{\rm abs}}={law['E0']:.2e}+{law['delta']:.2f}\,N$")
    for key, (label, colour, marker) in FAMILIES.items():
        if key not in scores:
            continue
        a.plot(norms[key]["N_rms"], scores[key]["absolute_rms"], marker=marker,
               ms=6, ls="none", color=colour, label=label, zorder=4)
    for key, label in COMPLETION.items():
        if key not in scores or key in FAMILIES:
            continue
        a.plot(norms[key]["N_rms"], scores[key]["absolute_rms"], marker="o", ms=5.5,
               ls="none", mfc="none", mec=BLACK, mew=1.0, zorder=3)
    matched = [k for k in scores if k.startswith("CTL_term_")]
    a.plot([norms[k]["N_rms"] for k in matched],
           [scores[k]["absolute_rms"] for k in matched], marker="_", ms=12, mew=1.4,
           ls="none", color=GRAY, label="four distinct terms at one common norm")
    a.set_yscale("log")
    a.set_xlabel(r"assembled source norm $N$ (traction units)")
    a.set_ylabel(r"$\mathrm{RMS}(\tau_w^{\rm model}-\tau_w^{\rm DNS})$")
    a.set_title("(a) Skill is set by the norm of the source, not its content", loc="left")
    a.set_xlim(-0.0012, max(xs.max(), 0.026) * 1.06)
    a.legend(frameon=False, loc="upper left", ncol=1)
    a.grid(which="both", color="0.93", lw=0.5, zorder=0)

    # ---------------- (b) the phase-shift interventions --------------------
    b = ax[1]
    shifts = [c for c in tj["contrasts"] if c["kind"] == "phase_shift"]
    order = sorted(shifts, key=lambda c: (c["second"], c["shift_fraction_of_period"]))
    y = np.arange(len(order))
    for i, c in enumerate(order):
        colour = GRAY if c["second"] == "M2_hickel" else BLACK
        for dy, rname, alpha in ((-0.15, "B_mglet", 1.0),
                                 (0.15, "C_xiao_repaired_cubic6", 0.45)):
            d = c["delta"][rname]
            b.plot([d["low"], d["high"]], [i + dy, i + dy], color=colour, lw=2.0,
                   alpha=alpha, solid_capstyle="butt")
            b.plot([d["median"]], [i + dy], marker="|", ms=8, color=colour, alpha=alpha)
    b.axvline(0.0, color=ORANGE, lw=1.1, ls=":")
    b.set_yticks(y, [("modelled source" if c["second"] == "M2_hickel" else "exact source")
                     + f", shift ${c['shift_fraction_of_period']:g}L_x$" for c in order])
    b.set_xlabel(r"paired $\Delta E$ (shifted $-$ unshifted); $>0$ means the shift hurt")
    b.set_title("(b) Only the modelled source loses by being misplaced", loc="left")
    b.text(0.98, 0.02, "dark: full-wall DNS reference\nfaint: same-simulation bracket",
           transform=b.transAxes, ha="right", va="bottom", fontsize=6.2)
    b.grid(axis="x", color="0.93", lw=0.5, zorder=0)

    # ---------------- (c) flat floor against curved flank ------------------
    c_ax = ax[2]
    reg = rj["surfaces"][SURFACE]["references"][REF]
    attain = reg["post_hoc_not_registered"]["attainment_E_abs_over_N_by_arm"]
    keys = [k for k in ("M2_hickel", "M1_pressure_gradient", "Xc_exact_convection",
                        "Xall", "M4_park_moin", "Xfull_closure_free") if k in attain]
    labels = {"M2_hickel": "modelled\nconvection",
              "M1_pressure_gradient": "pressure-\ngradient ODE",
              "Xc_exact_convection": "+ exact\nconvection",
              "Xall": "all omitted\ntransport",
              "M4_park_moin": "non-equilibrium\nPDE",
              "Xfull_closure_free": "no closure\nat all"}
    xpos = np.arange(len(keys))
    w = 0.36
    c_ax.bar(xpos - w / 2, [attain[k]["flat_floor"] for k in keys], width=w,
             color=GREEN, edgecolor="black", lw=0.4, label="flat inter-hill floor", zorder=3)
    c_ax.bar(xpos + w / 2, [attain[k]["sloped_wall"] for k in keys], width=w,
             color=BLACK, edgecolor="black", lw=0.4, label="curved flanks", zorder=3)
    c_ax.axhline(1.0, color=ORANGE, lw=1.0, ls=":")
    c_ax.set_yscale("log")
    c_ax.set_xticks(xpos, [labels[k] for k in keys], fontsize=6.0, rotation=30, ha="right")
    c_ax.set_ylabel(r"fraction of the assembled norm returned as error")
    ratio = reg["post_hoc_not_registered"]["closure_free_attainment_ratio"]
    c_ax.set_title(f"(c) The same reduction, two surfaces "
                   f"($\\times{ratio:.0f}$ on the closure-free arm)", loc="left")
    c_ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    c_ax.grid(axis="y", which="both", color="0.93", lw=0.5, zorder=0)

    figs = ROOT / "manuscript" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(figs / f"fig_source_budget_l0{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print("WROTE fig_source_budget_l0.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
