#!/usr/bin/env python3
"""Figure: same inputs, same station -- the deployed ODEs and a geometry-blind
function of exactly their own two input groups, on a hill neither has seen.

Colour scheme of the project: orange = ground truth, black = black-box
(the empirical same-input function), bluish-grey = grey-box (the
pressure-gradient boundary-layer equation), green = Spalding equilibrium.

Reads codes/results/input_sufficiency_bracket.{npz,_summary.json}; writes
manuscript/figures/fig_input_sufficiency.{pdf,png}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "codes" / "results"
OUT = ROOT / "manuscript" / "figures"

ORANGE = "#E8820C"
BLACK = "#000000"
GREY = "#6E7F91"
GREEN = "#2E8B22"


def main() -> int:
    d = np.load(RES / "input_sufficiency_bracket.npz", allow_pickle=True)
    S = json.loads((RES / "input_sufficiency_bracket_summary.json").read_text())
    tag = "eta0.05"
    rec = S["heights"][tag]["canonical"]

    tau = d[f"{tag}_canonical_tau_ref"]
    emp = d[f"{tag}_canonical_pred_empirical"]
    m0 = d[f"{tag}_canonical_pred_equilibrium"]
    m1 = d[f"{tag}_canonical_pred_tble"]
    phase = np.linspace(0.0, 1.0, len(tau), endpoint=False)

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))

    ax[0].plot(phase, tau, color=ORANGE, lw=2.0, label="reference DNS")
    ax[0].plot(phase, emp, color=BLACK, lw=1.2, ls="-",
               label=f"same inputs, empirical ($R^2={rec['r2_empirical']:.2f}$)")
    ax[0].plot(phase, m0, color=GREEN, lw=1.2, ls="--",
               label=f"equilibrium ($R^2={rec['r2_equilibrium']:.2f}$)")
    ax[0].plot(phase, m1, color=GREY, lw=1.0, ls="-.",
               label=(f"pressure-gradient ODE ($R^2={rec['r2_tble']:.1f}$,"
                      " clipped)"))
    ax[0].set_xlabel("streamwise phase within one period")
    ax[0].set_ylabel(r"$\tau_w$")
    ax[0].set_title(f"{rec['case']}, $\\eta_m/\\delta=0.05$", fontsize=9)
    lim = 4.0 * float(np.sqrt(np.mean(tau ** 2)))
    ax[0].set_ylim(-lim, lim)
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")

    grp = S["heights"][tag]["transfer"]["a_and_b"]["leave_one_group_out"]
    vals = np.array(list(grp["per_case"].values()))
    ax[1].hist(vals, bins=12, color=BLACK, alpha=0.75)
    ax[1].axvline(np.median(vals), color=ORANGE, lw=2.0,
                  label=f"median {np.median(vals):.3f}")
    ax[1].set_xlabel(r"held-out $R^2(\tau_w)$, same inputs")
    ax[1].set_ylabel("cases")
    ax[1].set_title("29 hills, steepness group held out", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_input_sufficiency.{ext}", dpi=200)
    print("wrote manuscript/figures/fig_input_sufficiency.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
