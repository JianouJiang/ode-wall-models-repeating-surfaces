#!/usr/bin/env python3
"""fig_coupling_gain.py -- how much of a wall model's error the coupling carries.

(a) the measured response of the coupled solution to a controlled open-loop
    perturbation of the wall model: paired restarts in which the delivered wall
    traction is multiplied by G = 1.25 at fixed matching velocity.  At the
    instant the gain is switched on the flow has not yet moved, so the
    transmission is exactly 1; it then relaxes over about one convective time
    unit.  Curves are running means over 0.5 convective units; the marker on
    each curve is the lag at which the two arms stop being one trajectory plus
    a deterministic difference.
(b) the transmission read at a lag of two convective units against matching
    height, for both deployed architectures, with the lag-1 to lag-5 spread as a
    band and the late-window mean and its moving-block interval as ticks.  A
    cross on the axis marks a configuration whose paired arms have already
    separated at that lag, so it has no deterministic reading.
(c) the wall model's response exponent s = dln|tau_w|/dln u, computed from the
    deployed kernels with no simulation, and the fraction of hill-wall faces on
    which it is not positive.

Colours follow the paper's convention: green = the Spalding/equilibrium arm,
bluish-grey = the total-gradient (grey-box) arm.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
GRAY, GREEN = "#6b7c93", "#2e8b57"
STYLE = {"tble": dict(color=GRAY, marker="s", label="total-gradient TBLE"),
         "eq": dict(color=GREEN, marker="o", label="equilibrium (Spalding)")}
DASH = {0.03: ":", 0.06: "-.", 0.0935: "-", 0.15: "--",
        0.25: (0, (3, 1, 1, 1, 1, 1))}
ARCH_OF = {"equilibrium": "eq", "tble": "tble"}
# The lag at which panel (b) reads T, and the scatter above which a
# reading is not deterministic.
READ = "2.0"
SD_MAX = 0.15


def latest(pattern: str) -> Path:
    hits = sorted(glob.glob(str(RESULTS / pattern)))
    if not hits:
        raise SystemExit("no producer output matching %s" % pattern)
    return Path(hits[-1])


def running_mean(t, y, width):
    out = np.empty_like(y)
    for i, ti in enumerate(t):
        m = (t >= ti - 0.5 * width) & (t <= ti + 0.5 * width)
        out[i] = y[m].mean()
    return out


def main() -> int:
    tr_path = latest("gain_probe_transmission_*_summary.json")
    tr = json.loads(tr_path.read_text())
    npz = np.load(str(tr_path).replace("_summary.json", ".npz"))
    ex_path = latest("gain_probe_model_exponent_*_summary.json")
    ex = json.loads(ex_path.read_text())

    pairs = [p for p in tr["pairs"]
             if p.get("status") == "ok" and p.get("patch") == "wallForceBottom"
             and p["gain"] == 1.25]
    if not pairs:
        raise SystemExit("no complete pairs to plot")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    # ---- (a) response --------------------------------------------------------
    ax = axes[0]
    for p in sorted(pairs, key=lambda r: (r["architecture"], r["ym_over_H"])):
        key = "%s__%s__series_time" % (p["perturbed"], p["patch"])
        if key not in npz.files:
            continue
        t = npz[key] - tr["gain_start_time"]
        T = npz[key.replace("series_time", "series_T")]
        m = t <= 10.0
        col = STYLE[p["architecture"]]["color"]
        ax.plot(t[m], running_mean(t[m], T[m], 0.5), lw=1.4, color=col,
                alpha=0.9, ls=DASH.get(round(p["ym_over_H"], 4), "-"))
        d = p.get("decorrelation_lag")
        if d is not None and np.isfinite(d) and d <= 10.0:
            j = int(np.argmin(np.abs(t - d)))
            ax.plot([t[j]], [running_mean(t[m], T[m], 0.5)[j]], marker="|",
                    ms=8, color=col)
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$(t-t_0)\,U_b/H$")
    ax.set_ylabel(r"transmission $T(t)$")
    ax.set_title(r"(a) response to $G=1.25$", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.0, 1.08)
    for ym, ls in sorted(DASH.items()):
        if any(round(p["ym_over_H"], 4) == ym for p in pairs):
            ax.plot([], [], ls=ls, color="0.45", lw=1.2,
                    label=r"$\eta_m/H=%.4g$" % ym)
    ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=False)

    # ---- (b) transmission vs matching height --------------------------------
    ax = axes[1]
    for arch in ("eq", "tble"):
        pts = sorted([p for p in pairs if p["architecture"] == arch],
                     key=lambda r: r["ym_over_H"])
        if not pts:
            continue
        det = [p for p in pts if p["T_sd_at_lag"][READ] < SD_MAX]
        x = np.array([p["ym_over_H"] for p in det])
        y = np.array([p["T_at_lag"][READ] for p in det])
        lo = np.array([min(p["T_at_lag"]["1.0"], p["T_at_lag"]["5.0"])
                       for p in det])
        hi = np.array([max(p["T_at_lag"]["1.0"], p["T_at_lag"]["5.0"])
                       for p in det])
        if x.size:
            ax.fill_between(x, lo, hi, color=STYLE[arch]["color"], alpha=0.16,
                            lw=0)
            ax.plot(x, y, lw=1.6, **STYLE[arch])
        for p in det:
            ax.plot([p["ym_over_H"]] * 2, [p["T_ci_lo"], p["T_ci_hi"]], lw=0.9,
                    color=STYLE[arch]["color"], alpha=0.35)
            ax.plot([p["ym_over_H"]], [p["T_late"]], marker="_", ms=8,
                    color=STYLE[arch]["color"], alpha=0.6)
        # configurations with no deterministic reading are shown as open
        # symbols at the axis, not as measurements
        for p in pts:
            if p in det:
                continue
            ax.plot([p["ym_over_H"]], [0.0], marker="x", ms=7, mew=1.4,
                    color=STYLE[arch]["color"], ls="none")
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$\eta_m/H$")
    ax.set_ylabel(r"transmission $T$")
    ax.set_title(r"(b) $T$ at lag $2H/U_b$ (band: lags 1--5)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    # ---- (c) a-priori response exponent -------------------------------------
    ax = axes[2]
    for arch in ("eq", "tble"):
        recs = sorted([r for r in ex["records"]
                       if ARCH_OF[r["architecture"]] == arch],
                      key=lambda r: r["ym_over_H"])
        if not recs:
            continue
        x = [r["ym_over_H"] for r in recs]
        y = [r["s_median"] for r in recs]
        ax.fill_between(x, [r["s_p10"] for r in recs],
                        [r["s_p90"] for r in recs],
                        color=STYLE[arch]["color"], alpha=0.16, lw=0)
        ax.plot(x, y, lw=1.6, **STYLE[arch])
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
    ax.axhline(2.0, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel(r"$\eta_m/H$")
    ax.set_ylabel(r"response exponent $s$")
    ax.set_title(r"(c) $s=\mathrm{d}\ln|\tau_w|/\mathrm{d}\ln u$, "
                 "median and 10--90 per cent", fontsize=10)
    ax.set_ylim(-2.2, 3.0)

    axb = ax.twinx()
    for arch in ("eq", "tble"):
        recs = sorted([r for r in ex["records"]
                       if ARCH_OF[r["architecture"]] == arch],
                      key=lambda r: r["ym_over_H"])
        axb.plot([r["ym_over_H"] for r in recs],
                 [100.0 * r["fraction_s_nonpositive"] for r in recs],
                 lw=1.0, ls="--", color=STYLE[arch]["color"], alpha=0.55,
                 marker=".", ms=4)
    axb.set_ylabel("faces with $s\\leq0$ (per cent)", fontsize=9)
    axb.set_ylim(-5, 100)

    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = ROOT / "manuscript" / "figures" / "fig_coupling_gain.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("FIGURE_RC=0 wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
