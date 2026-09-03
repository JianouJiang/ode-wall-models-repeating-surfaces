#!/usr/bin/env python3
r"""
fig_crwm_coupled_twin.py  --  Thrust #18 L2/L3 figure: the coupled F1 twin.

Plots the decisive a-posteriori contrast on the canonical h/L_x=1.0 periodic
hill: mean skin-friction C_f(x) on the lee (hill) wall for the three coupled
rungs (Spalding / TBLE / CR-WM) with the reattachment points marked, and the
DNS reattachment reference.  Reads ONLY codes/results/aposteriori_crwm_twin.npz
— if a rung is absent it is skipped (no fabricated curve).

Colour scheme (project convention): green=Spalding, black=black-box/indicted
TBLE, bluish-gray=CR-WM (gray-box), orange=ground truth (DNS).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "codes", "results", "aposteriori_crwm_twin.npz")
OUT = os.path.join(ROOT, "manuscript", "figures", "fig_crwm_coupled_twin.pdf")

STYLE = {
    "spalding": ("green", "Spalding (drops conv.+dp/dx)"),
    "tble":     ("black", "TBLE / ODE (drops convection)"),
    "crwm":     ("#5b7", "CR-WM (restores convection)"),  # placeholder gray-blue
}
STYLE["crwm"] = ("#6a7b8c", "CR-WM (restores convection)")  # bluish-gray


def main():
    if not os.path.exists(NPZ):
        print(f"[fig] {NPZ} absent — run harvest_crwm_twin.py first. Nothing plotted.")
        return 1
    d = np.load(NPZ, allow_pickle=True)
    Lx = float(d["Lx"]); x_dns = float(d["dns_x_reatt"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for rung, (color, label) in STYLE.items():
        if not bool(d.get(f"{rung}_present", False)):
            continue
        x = d[f"{rung}_x"]; cf = d[f"{rung}_cf"]
        xr = float(d[f"{rung}_x_reatt"]); er = float(d[f"{rung}_e_reatt"])
        ax.plot(x / Lx, cf, color=color, lw=1.6,
                label=f"{label}: $x_r$={xr:.2f} (err {100*er:.0f}%)")
        if np.isfinite(xr):
            ax.plot([xr / Lx], [0], "o", color=color, ms=6)
    ax.axvline(x_dns / Lx, color="orange", ls="--", lw=1.8,
               label=f"DNS $x_r$={x_dns:.2f}")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$x/L_x$"); ax.set_ylabel(r"$C_f$ (mean, lee wall)")
    ax.set_xlim(0, 0.75); ax.legend(fontsize=7, loc="best")
    ax.set_title("Coupled a-posteriori WMLES: convection-dropping vs restored "
                 "(h/$L_x$=1.0 hill)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"[fig] wrote {OUT}")
    # echo the headline contrast (traceable, not fabricated)
    if "reatt_improvement_frac" in d.files:
        print(f"[fig] F1 contrast: {100*float(d['reatt_improvement_frac']):.1f}% "
              f"of reattachment error removed by restoring convection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
