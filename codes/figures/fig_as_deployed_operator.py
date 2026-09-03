#!/usr/bin/env python3
"""fig_as_deployed_operator.py -- the wall model as published vs as deployed.

(a) the three wall-traction curves on the canonical matching surface: what the
    DNS says, what the wall model requests, what the boundary condition
    delivers, and what the solver actually carried;
(b) the regime census of the delivery map against matching height, for both
    deployed architectures and both walls;
(c) the estimand bridge: how much of the request-to-measured gap the delivery
    map accounts for.

Colours follow the paper's convention: orange = ground truth, green = the
Spalding/equilibrium arm.  The three operator curves of panel (a) are
distinguished as black (requested, i.e. the model's own answer), bluish-grey
(delivered) and dashed black (measured).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
# The wall traction this panel is scored against is read from the published
# reference artifact and from nowhere else.  Rebuilding a reference inside a
# figure producer is what let the withdrawn four-point estimator survive here.
REFERENCES = RESULTS / "wall_traction_references_20260825.npz"
ORANGE, GRAY, GREEN = "#e07b1a", "#6b7c93", "#2e8b57"


def latest(pattern: str) -> Path:
    hits = [h for h in sorted(glob.glob(str(RESULTS / pattern)))
            if "pilot" not in h]
    if not hits:
        raise SystemExit(f"no producer output matching {pattern}")
    return Path(hits[-1])


def main() -> int:
    npz = np.load(latest("as_deployed_evaluation_*[0-9].npz"))
    summary = json.loads(latest("as_deployed_evaluation_*_summary.json").read_text())
    records = summary["records"]

    hill = [r for r in records if r["patch"] == "bottomWall"]
    top = [r for r in records if r["patch"] == "topWall"]
    tble = [r for r in hill if r["model"] == "total_gradient_tble"]
    equil = [r for r in hill if r["model"] == "equilibrium"]

    # panel (a): the canonical a-priori surface, longest window, finest common grid
    canon = sorted(
        [r for r in tble if abs(r["ym_median"] - 0.0935) < 0.02],
        key=lambda r: (-r["time"], -r["n_faces"]))
    if not canon:
        canon = sorted(tble, key=lambda r: (-r["time"]))
    c = canon[0]
    key = f"{c['case']}__{c['time']:.0f}__bottomWall__"
    refs = np.load(REFERENCES)

    def reference(name):
        ph = np.asarray(refs[f"{name}__phase"], float)
        tw = np.asarray(refs[f"{name}__tau"], float)
        o = np.argsort(ph)
        return ph[o], tw[o]

    phase_B, tau_B = reference("B_mglet")
    phase_C, tau_C = reference("C_xiao_repaired_cubic6")

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.9))

    a = ax[0]
    a.plot(phase_B, tau_B, color=ORANGE, lw=2.2,
           label="DNS wall traction (Peller \\& Manhart)")
    a.plot(phase_C, tau_C, color=ORANGE, lw=1.1, ls=":",
           label="same-simulation bracket (Xiao et al., cubic)")
    a.plot(npz[key + "request__phase"], npz[key + "request__tau"], color="k",
           lw=1.5, label=r"requested $\tau_w$ (wall model)")
    a.plot(npz[key + "deliver__phase"], npz[key + "deliver__tau"], color=GRAY,
           lw=1.8, label=r"delivered $\tau_s$ (boundary condition)")
    a.plot(npz[key + "measured__phase"], npz[key + "measured__tau"], color="k",
           lw=1.3, ls="--", label=r"measured $\langle\tau_s\rangle$ (solver)")
    a.axhline(0.0, color="0.7", lw=0.7)
    a.set_xlabel(r"$x/L_x$")
    a.set_ylabel(r"$\tau_s$")
    cells = "819{,}200" if "819200" in c["case"] else "307{,}200"
    a.set_title(f"(a) TBLE at $\\eta_m/H={c['ym_median']:.3g}$, "
                f"${cells}$ cells, window $t\\in[{c['averaging_window'][0]:.0f},"
                f"{c['averaging_window'][1]:.0f}]$", fontsize=9)
    a.legend(fontsize=7, frameon=False, loc="best")

    b = ax[1]
    series = (
        (tble, "k", "o", "TBLE, hill wall"),
        (equil, GREEN, "s", "equilibrium, hill wall"),
        ([r for r in top if r["model"] == "total_gradient_tble"],
         "k", "^", "TBLE, flat top wall"),
    )
    for rows, colour, marker, label in series:
        if not rows:
            continue
        xs = np.array([r["ym_median"] for r in rows])
        refused = np.array([r["regime_fraction"]["sign_refused"] for r in rows])
        flipped = np.array([r["sign_disagreement_request_vs_deliver"]
                            for r in rows])
        o = np.argsort(xs)
        b.plot(xs[o], refused[o], marker=marker, ls="none", color=colour, ms=5,
               label=f"{label}: request discarded")
        b.plot(xs[o], flipped[o], marker=marker, ls="none", color=colour, ms=5,
               markerfacecolor="none",
               label=f"{label}: delivered sign opposite")
    b.set_xlabel(r"$\eta_m/H$ of the patch's own matching surface")
    b.set_ylabel("fraction of wall faces")
    b.set_ylim(-0.03, 1.0)
    b.set_title("(b) what the delivery map does to the request", fontsize=9)
    b.legend(fontsize=6, frameon=False, ncol=1, loc="center right")

    c2 = ax[2]
    for rows, colour, marker, label in ((tble, "k", "o", "TBLE"),
                                        (equil, GREEN, "s", "equilibrium")):
        rows = [r for r in rows if "bridge" in r]
        if not rows:
            continue
        xs = np.array([r["ym_median"] for r in rows])
        d = np.array([r["bridge"]["delivery_deficiency_rms_over_measured"]
                      for r in rows])
        n = np.array([r["bridge"]["averaging_residual_rms_over_measured"]
                      for r in rows])
        o = np.argsort(xs)
        c2.plot(xs[o], d[o], marker=marker, ls="none", color=colour, ms=5,
                label=f"{label}: delivery deficiency $D$")
        c2.plot(xs[o], n[o], marker=marker, ls="none", color=colour, ms=5,
                markerfacecolor="none", label=f"{label}: residual $N$")
    c2.set_xlabel(r"$\eta_m/H$")
    c2.set_ylabel(r"r.m.s. / r.m.s. of measured $\tau_s$")
    c2.set_title("(c) the bridge from request to measurement", fontsize=9)
    c2.legend(fontsize=6.5, frameon=False, loc="upper left")

    for axis in ax:
        axis.tick_params(labelsize=8)
    fig.tight_layout()
    # Render once per format, then COPY, so every deposited asset is
    # byte-identical (the figure-provenance verifier requires it).
    import shutil
    figs = ROOT / "manuscript" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    master_pdf = figs / "fig_as_deployed_operator.pdf"
    master_png = figs / "fig_as_deployed_operator.png"
    fig.savefig(master_pdf, bbox_inches="tight")
    fig.savefig(master_png, dpi=150, bbox_inches="tight")
    # Deposits live outside development/nodes/: that tree is rotated by the
    # pipeline, which has already broken two harvests that kept evidence there.
    for dest in (ROOT / "manuscript" / "submission_flat",
                 ROOT / "codes" / "figures" / "node_generators"):
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master_pdf, dest / master_pdf.name)
        shutil.copyfile(master_png, dest / master_png.name)
    print("WROTE fig_as_deployed_operator.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
