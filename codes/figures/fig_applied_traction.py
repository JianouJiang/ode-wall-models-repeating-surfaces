#!/usr/bin/env python3
"""
Figure: the wall stress the model requests is not the wall stress the
simulation receives.

(a) Requested versus applied wall-tangential traction, face by face, on the
    canonical matching surface.  Points off the diagonal are faces whose
    traction the realizability projection changed; points in the shaded
    quadrants are faces where the applied traction opposes the request.
(b) Mean departure |applied - requested| per wall face through the run, for
    each matching height.
(c) The same departure, time-averaged over the analysis window, against the
    matching height.

Reads only codes/results/applied_traction_reproduction.npz and
codes/results/clipping_timeseries.npz.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(PROJECT, "codes", "results")

REPRO = os.path.join(RESULTS, "applied_traction_reproduction")
CLIP = os.path.join(RESULTS, "clipping_timeseries")

# Matching heights, in the case-name convention used by the campaign bundle.
HEIGHTS = {"ym0300": 0.03, "ym0600": 0.06, "ym0935": 0.0935,
           "ym1500": 0.15, "ym2500": 0.25}
FEATURE = "rswm_r23m6_ym0935_g1_tble_307200_v1"

# Project colour convention: orange = ground truth / reference quantity,
# bluish-grey = model-side quantity, green = Spalding.
C_TRUTH = "#E8820C"
C_GRAY = "#5B7C99"
C_DARK = "#222222"


def grid_of(case: str) -> str:
    return "G2c" if "819200" in case else "G1c"


def height_of(case: str) -> float:
    for key, val in HEIGHTS.items():
        if key in case:
            return val
    return float("nan")


def main() -> int:
    repro = np.load(REPRO + ".npz")
    summary = json.load(open(REPRO + "_summary.json"))
    clip = np.load(CLIP + ".npz")
    clip_sum = json.load(open(CLIP + "_summary.json"))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # ---- (a) requested versus applied ------------------------------------
    ax = axes[0]
    case = FEATURE if f"{FEATURE}__requested_tau" in repro else \
        sorted({k.split("__")[0] for k in repro.files})[0]
    req = repro[f"{case}__requested_tau"]
    app = repro[f"{case}__applied_tau"]
    lower = repro[f"{case}__lower_clipped"] > 0
    vector = repro[f"{case}__vector_capped"] > 0

    lim = 1.05 * max(np.max(np.abs(req)), np.max(np.abs(app)))
    # quadrants where the applied traction opposes the requested one
    ax.add_patch(plt.Rectangle((-lim, 0), lim, lim, fc="0.90", ec="none", zorder=0))
    ax.add_patch(plt.Rectangle((0, -lim), lim, lim, fc="0.90", ec="none", zorder=0))
    ax.plot([-lim, lim], [-lim, lim], color=C_TRUTH, lw=1.6, zorder=3,
            label="applied = requested")
    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    ax.scatter(req[lower], app[lower], s=3, c=C_DARK, alpha=0.45, lw=0,
               zorder=2, label="negative eddy viscosity refused")
    ax.scatter(req[vector], app[vector], s=3, c=C_GRAY, alpha=0.45, lw=0,
               zorder=2, label="complete traction bounded")
    neither = ~(lower | vector)
    if neither.any():
        ax.scatter(req[neither], app[neither], s=3, c="#B03060", alpha=0.6,
                   lw=0, zorder=2, label="unchanged")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"requested $\tau_w$ (wall model)")
    ax.set_ylabel(r"applied $\tau_s$ (boundary condition)")
    n_flip = int(np.sum(np.sign(req) != np.sign(app)))
    ax.set_title(f"(a) $\\eta_m/H={height_of(case):g}$: "
                 f"{100*n_flip/req.size:.1f}% of faces opposed", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)

    # ---- (b) departure through the run -----------------------------------
    ax = axes[1]
    cases = sorted({k.split("__")[0] for k in clip.files})
    cases = [c for c in cases if not np.isnan(height_of(c))]
    cases.sort(key=height_of)
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(cases):
        key_t, key_m = f"{c}__bottomWall__t", f"{c}__bottomWall__mismatch_per_face"
        if key_t not in clip.files:
            continue
        t, m = clip[key_t], clip[key_m]
        keep = t >= clip_sum["window_start"]
        ax.plot(t[keep], m[keep], lw=1.0,
                color=cmap(i / max(len(cases) - 1, 1)),
                label=rf"$\eta_m/H={height_of(c):g}$ ({grid_of(c)})")
    ax.set_xlabel(r"$t\,U_b/H$")
    ax.set_ylabel(r"mean $|\tau_s-\tau_w|$ per wall face")
    ax.set_title("(b) departure persists through the run", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)

    # ---- (c) departure against matching height ---------------------------
    ax = axes[2]
    recs = [r for r in clip_sum["records"] if r["patch"] == "bottomWall"]
    pts = sorted(((height_of(r["case"]), r["mismatch_per_face_mean"],
                   r["frac_clipped_mean"], grid_of(r["case"])) for r in recs
                  if not np.isnan(height_of(r["case"]))))
    fine = [p for p in pts if p[3] == "G2c"]
    base = [p for p in pts if p[3] != "G2c"]
    ax.plot([p[0] for p in base], [p[1] for p in base], "o-", color=C_GRAY,
            lw=1.6, ms=6, label="307,200 cells")
    if fine:
        ax.plot([p[0] for p in fine], [p[1] for p in fine], "s", color=C_DARK,
                ms=6, mfc="none", mew=1.4, label="819,200 cells")
    for h, m, _f, g in base:
        ax.annotate(f"{m:.2e}", (h, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5)
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_xlabel(r"matching height $\eta_m/H$")
    ax.set_ylabel(r"time-mean $|\tau_s-\tau_w|$ per face")
    frac_min = 100 * min(p[2] for p in pts)
    ax.set_title(f"(c) grows with $\\eta_m$; "
                 f"$\\geq${frac_min:.2f}% of faces altered", fontsize=10)
    ax.margins(x=0.12, y=0.18)

    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(PROJECT, "manuscript", "figures",
                           f"fig_applied_traction.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
