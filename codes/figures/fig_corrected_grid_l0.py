#!/usr/bin/env python3
"""Corrected crest-bulk coupled grid assessment at Re_H = 5600.

Replaces the rendering that the superseded reference produced.  That figure
asserted "failure survives refinement" and printed a fixed significance value
in its own axis annotation; both were properties of a wall traction that has
since been withdrawn, and both contradicted the corrected table in the same
paper.  A figure may not carry a conclusion its own data no longer supports,
and an annotation that states an outcome cannot be re-derived when the data
change --- so every number drawn here is read from the corrected coupled
artifact at plot time and nothing about the verdict is written into the code.

Panel (a) is the RMS-normalised traction error against mesh count with paired
phase-block 95% intervals and the reference-RMS unit threshold.  Panel (b) is
reattachment against the spread of the available references rather than a
single one, because the references themselves disagree by about 10% on the
bubble length.

Reads codes/results/m13_highre_coupled_20260825_summary.json.
Run:  MPLBACKEND=Agg python3 codes/figures/fig_corrected_grid_l0.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "codes/results/m13_highre_coupled_20260825_summary.json"
FIG_DIR = ROOT / "manuscript" / "figures"
CODE_FIG_DIR = ROOT / "codes" / "figures"
MANUSCRIPT_PDF = FIG_DIR / "fig_common_surface_grid_l3.pdf"
MANUSCRIPT_PNG = FIG_DIR / "fig_common_surface_grid_l3.png"
CODE_PDF = CODE_FIG_DIR / "fig_common_surface_grid_l3.pdf"
CODE_PNG = CODE_FIG_DIR / "fig_common_surface_grid_l3.png"

# project colour convention: orange = reference, green = equilibrium,
# bluish grey = total-gradient TBLE
C_REF, C_EQ, C_TBLE = "#e8820c", "#2e8b57", "#6b7f95"
GRIDS = ["G0", "G1c", "G2c"]
CELLS = {"G0": 92_160, "G1c": 307_200, "G2c": 819_200}
MODELS = [("equilibrium", "equilibrium", C_EQ, "--", "o"),
          ("total_gradient_tble", "total-gradient TBLE", C_TBLE, "-", "s")]
WINDOW = "270"


def main() -> int:
    d = json.loads(SUMMARY.read_text())["campaigns"]["5600"]
    av, ci = d["averaging"], d["phase_bootstrap_primary_intervals"]
    ref_xr = d["experimental_reattachment"]["estimate_x_over_H"]
    ref_lo, ref_hi = d["experimental_reattachment"]["bracket_x_over_H"]

    x = np.arange(len(GRIDS), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # ---- (a) traction error with paired phase-block intervals -------------
    ax = axes[0]
    for key, label, colour, ls, mk in MODELS:
        med = np.array([av[f"{g}:{key}"][WINDOW]["relative_rms"] for g in GRIDS])
        lo = np.array([ci[f"{g}:{key}"]["low"] for g in GRIDS])
        hi = np.array([ci[f"{g}:{key}"]["high"] for g in GRIDS])
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=colour, ls=ls,
                    marker=mk, ms=5.0, lw=1.4, capsize=3.0, label=label)
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    ax.text(0.02, 1.02, "error $=$ reference RMS", color="0.35", fontsize=8.5)
    ax.set_xticks(x, [f"{CELLS[g]:,}" for g in GRIDS])
    ax.set_xlabel("mesh cells")
    ax.set_ylabel(r"RMS$(\tau_s-\tau_s^{\rm ref})\,/\,$RMS$(\tau_s^{\rm ref})$")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("(a) traction error and refinement", loc="left", fontsize=11)
    ax.legend(loc="center right", fontsize=9, frameon=False)

    # ---- (b) reattachment against the reference spread --------------------
    ax = axes[1]
    ax.axhspan(ref_lo, ref_hi, color=C_REF, alpha=0.16, zorder=0,
               label="reference bracket")
    ax.axhline(ref_xr, color=C_REF, lw=1.4, zorder=1, label="reference estimate")
    for key, label, colour, ls, mk in MODELS:
        xr = np.array([av[f"{g}:{key}"][WINDOW]["reattachment_x_over_H"]
                       for g in GRIDS])
        ax.plot(x, xr, color=colour, ls=ls, marker=mk, ms=5.0, lw=1.4,
                label=label, zorder=3)
    ax.set_xticks(x, [f"{CELLS[g]:,}" for g in GRIDS])
    ax.set_xlabel("mesh cells")
    ax.set_ylabel(r"reattachment $x_r/H$")
    ax.set_title("(b) reattachment against the reference spread", loc="left",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    # written out one call per target, not in a loop, so that a static
    # provenance audit can see which figure this script produces without
    # executing it
    fig.savefig(MANUSCRIPT_PDF, bbox_inches="tight")
    fig.savefig(MANUSCRIPT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(CODE_PDF, bbox_inches="tight")
    fig.savefig(CODE_PNG, dpi=300, bbox_inches="tight")
    print(f"  Saved: {MANUSCRIPT_PDF.relative_to(ROOT)} (+ png, + codes copy)")
    plt.close(fig)

    for key, label, _, _, _ in MODELS:
        med = [av[f"{g}:{key}"][WINDOW]["relative_rms"] for g in GRIDS]
        lo = [ci[f"{g}:{key}"]["low"] for g in GRIDS]
        hi = [ci[f"{g}:{key}"]["high"] for g in GRIDS]
        print(f"{label:22s} E_tau " + "  ".join(f"{m:.3f}[{a:.3f},{b:.3f}]"
                                                for m, a, b in zip(med, lo, hi)))
        print(f"{'':22s} all intervals below unity: "
              f"{all(b < 1.0 for b in hi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
