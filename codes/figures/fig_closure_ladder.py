#!/usr/bin/env python3
r"""
fig_closure_ladder.py  --  L2 a-posteriori closure-ladder figure
=================================================================

(a) Bottom-wall C_f(x) on periodic hills (Re_H=10595): DNS (orange) vs the two
    coupled-WMLES rungs that share the geometry/mesh/SGS and differ ONLY in the
    wall-model closure -- equilibrium ODE (green) and TBLE / non-equilibrium ODE
    (blue).  Reattachment markers show neither local closure recovers the DNS
    reattachment length.
(b) A-posteriori skill bars: |reattachment error| (%), one bar per rung, with
    the cancellation verdict -- climbing the LOCAL closure ladder (adding dp/dx)
    does not rescue the repeating geometry, because the missing physics is the
    NON-LOCAL convective flux.

Colours: orange = DNS truth, green = equilibrium ODE (Spalding family), blue =
TBLE/non-equilibrium ODE.  (The orange/black/bluish-gray/green ML-model scheme
governs the a-priori model-comparison figures; this panel compares two PHYSICS
wall-model closures, so the second physics rung gets its own blue.)

Consumes codes/results/closure_ladder_aposteriori.npz.  Rungs flagged
present=False are simply omitted -- no fabricated curve.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "codes", "results", "closure_ladder_aposteriori.npz")
OUT = os.path.join(ROOT, "manuscript", "figures", "fig_closure_ladder.pdf")
PREV = os.path.join(ROOT, "development", "nodes", "node_002",
                    "fig_closure_ladder_preview.png")

C_DNS = "#ff7f0e"   # orange  — DNS ground truth
C_EQ = "#2ca02c"    # green   — equilibrium ODE (Spalding family)
C_TBLE = "#1f4e9c"  # blue    — TBLE / non-equilibrium ODE

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})


def main():
    d = np.load(NPZ, allow_pickle=True)
    d = {k: d[k] for k in d.files}
    dns_x, dns_cf = d["dns_x"], d["dns_cf"]
    dns_xr = float(d["dns_x_reatt"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5),
                                   gridspec_kw={"width_ratios": [2.0, 1.0]})

    # ---- panel (a): C_f(x) ladder -----------------------------------------
    ax1.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax1.plot(dns_x, dns_cf, color=C_DNS, lw=1.8, label="DNS (Krank et al. 2018)", zorder=5)
    ax1.axvline(dns_xr, color=C_DNS, ls=":", lw=1.0, zorder=2)

    rungs = []
    if bool(d.get("eq_present", False)):
        rungs.append(("eq", "Equilibrium ODE (no $\\,dp/dx$)", C_EQ, "s"))
    if bool(d.get("tble_present", False)):
        rungs.append(("tble", "TBLE / non-eq. ODE (keeps $dp/dx$)", C_TBLE, "^"))

    for pre, lab, col, mk in rungs:
        x, cf = d[f"{pre}_x"], d[f"{pre}_cf"]
        xr = float(d[f"{pre}_x_reatt"])
        rel = float(d[f"{pre}_reatt_rel_err_pct"])
        ax1.plot(x, cf, color=col, lw=1.4, marker=mk, ms=3.2, mfc="none",
                 label=f"{lab}: $x_r/H$={xr:.2f} ({rel:+.0f}%)", zorder=4)
        ax1.axvline(xr, color=col, ls="--", lw=0.9, zorder=2)

    ax1.set_xlabel("$x/H$")
    ax1.set_ylabel("$C_f$")
    ax1.set_title("(a) Coupled-WMLES wall friction on periodic hills", fontsize=10)
    ax1.legend(fontsize=7.2, loc="upper right", framealpha=0.9)
    ax1.set_xlim(dns_x.min(), dns_x.max())

    # ---- panel (b): |reattachment error| bars -----------------------------
    labels, errs, cols = [], [], []
    for pre, lab, col, mk in rungs:
        labels.append("Equil.\nODE" if pre == "eq" else "TBLE\n(non-eq)")
        errs.append(abs(float(d[f"{pre}_reatt_rel_err_pct"])))
        cols.append(col)
    xb = np.arange(len(labels))
    ax2.bar(xb, errs, color=cols, width=0.6, edgecolor="k", lw=0.6)
    for i, e in enumerate(errs):
        ax2.text(i, e + 0.6, f"{e:.0f}%", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("$|$reattachment error$|$  (%)")
    ax2.set_title("(b) Local closure ladder", fontsize=10)
    if errs:
        ax2.set_ylim(0, max(errs) * 1.35 + 1)

    if len(rungs) == 2:
        dpp = abs(errs[1]) - abs(errs[0])
        msg = ("$dp/dx$ upgrade\ndoes not rescue\n($\\Delta=%+.0f$ pp)" % dpp)
        ax2.text(0.5, 0.92, msg, transform=ax2.transAxes, ha="center", va="top",
                 fontsize=8, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f3f3f3", ec="0.6", lw=0.6))
    else:
        ax2.text(0.5, 0.92, "TBLE rung\nstill solving", transform=ax2.transAxes,
                 ha="center", va="top", fontsize=8, color="0.4",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="0.7", lw=0.6))

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    os.makedirs(os.path.dirname(PREV), exist_ok=True)
    fig.savefig(PREV, dpi=130, bbox_inches="tight")
    print(f"[OUT] {os.path.relpath(OUT, ROOT)}  ({len(rungs)} rung(s) plotted)")
    print(f"[OUT] {os.path.relpath(PREV, ROOT)}")


if __name__ == "__main__":
    main()
