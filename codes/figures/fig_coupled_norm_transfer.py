#!/usr/bin/env python3
r"""
fig_coupled_norm_transfer.py -- L3 (Thrust #11) results figure (3 panels, all REAL
data from coupled_norm_transfer.npz):

 (a) coupled deployed-WMLES skin friction C_f(x) vs Krank2018 DNS on the
     canonical hill -- R^2(C_f)>0 yet reattachment lands ~20% early;
 (b) the NORM TRANSFER: same model, same hill -- a-priori R^2(tau_w) catastrophe
     vs a-posteriori R^2(C_f), with the surviving integral bias annotated;
 (c) the dose-response: a-priori |R^2(tau_w)| vs cancellation depth eps across the
     Xiao steepness family + mechanism predictor A/eps+B + conv-div control.

Colors per repo convention: orange=DNS truth, green=Spalding/equilibrium,
bluish-gray=TBLE, black=annotation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.0,
    "figure.dpi": 150, "savefig.dpi": 300, "lines.linewidth": 1.4,
    "font.family": "serif", "mathtext.fontset": "cm",
})
ORANGE = "#E69F00"; GREEN = "#009E73"; GRAYBLUE = "#56759A"
BLACK = "#000000"; RED = "#C44E52"

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                      "manuscript", "figures"))
NODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "development", "nodes", "node_003"))


def main():
    d = np.load(os.path.join(RES, "coupled_norm_transfer.npz"), allow_pickle=True)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # ---- (a) coupled C_f(x) vs DNS ---------------------------------------- #
    a = ax[0]
    a.axhline(0, color="0.6", lw=0.7, zorder=0)
    a.plot(d["dns_x"], d["dns_cf"], color=ORANGE, lw=1.6, label="DNS (Krank 2018)")
    a.plot(d["cf_x"], d["cf_eq"], "o-", color=GREEN, ms=2.6, lw=1.2,
           label=r"WMLES, equilibrium ($R^2_{C_f}{=}%+.2f$)" % float(d["coupled_eq_R2_cf"]))
    a.plot(d["cf_x"], d["cf_tble"], "s--", color=GRAYBLUE, ms=2.4, lw=1.0,
           label=r"WMLES, TBLE ($R^2_{C_f}{=}%+.2f$)" % float(d["coupled_tble_R2_cf"]))
    xr_dns = float(d["dns_x_reatt"]); xr_eq = float(d["eq_x_reatt"])
    a.axvline(xr_dns, color=ORANGE, ls=":", lw=1.0)
    a.axvline(xr_eq, color=GREEN, ls=":", lw=1.0)
    ytop = float(np.max(d["dns_cf"])) * 0.7
    a.annotate("", xy=(xr_eq, ytop), xytext=(xr_dns, ytop),
               arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.9))
    a.text((xr_dns + xr_eq) / 2, ytop * 1.15,
           r"$%.0f\%%$" % float(d["coupled_eq_reatt_rel_err_pct"]),
           ha="center", va="bottom", fontsize=8, color=BLACK)
    a.set_xlabel(r"$x/H$"); a.set_ylabel(r"$C_f$")
    a.set_title(r"(a) Coupled $C_f(x)$, hill $h/L_x{=}1$")
    a.set_xlim(0, 9)
    a.legend(loc="upper right", framealpha=0.9)

    # ---- (b) the norm transfer ------------------------------------------- #
    b = ax[1]
    R2_local = float(d["apriori_R2_tauw"])
    R2_global = float(d["coupled_eq_R2_cf"])
    b.bar([0, 1], [R2_local, R2_global], width=0.55,
          color=[RED, GREEN], edgecolor=BLACK, lw=0.8, zorder=3)
    b.axhline(0, color=BLACK, lw=0.8)
    b.set_xticks([0, 1])
    b.set_xticklabels(["a-priori\n$R^2(\\tau_w)$\n(local)",
                       "a-posteriori\n$R^2(C_f)$\n(global)"])
    b.set_ylabel("coefficient of determination")
    b.set_title("(b) Norm transfer (same model, same hill)")
    b.text(0, R2_local + 2.5, r"$%.1f$" % R2_local, ha="center", va="bottom",
           fontsize=9, color=RED, fontweight="bold")
    b.text(1, R2_global + 1.5, r"$%+.2f$" % R2_global, ha="center", va="bottom",
           fontsize=9, color=GREEN, fontweight="bold")
    b.annotate(("coupling clamps the pointwise\n"
                r"$O(1/\varepsilon)$ blow-up, but a" "\n"
                r"$%.0f\%%$ reattachment bias +" "\n"
                r"$%.2f\,u_b$ profile error survive")
               % (float(d["coupled_eq_reatt_rel_err_pct"]),
                  float(d["coupled_eq_profile_rms_mean"])),
               xy=(1, R2_global), xytext=(0.05, -30),
               fontsize=6.8, color=BLACK,
               arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    b.set_ylim(R2_local - 7, 13)

    # ---- (c) dose-response across eps ------------------------------------ #
    c = ax[2]
    eps = np.asarray(d["fam_eps"], float)
    absr2 = np.abs(np.asarray(d["fam_r2"], float))
    c.scatter(eps, absr2, s=30, color=GREEN, edgecolor=BLACK, lw=0.5, zorder=4,
              label="Xiao steepness family\n(a-priori, 29 configs)")
    A = float(d["predictor_A"]); B = float(d["predictor_B"])
    xx = np.linspace(eps.min() * 0.85, 4.0, 200)
    c.plot(xx, np.clip(A / xx + B, 0.5, None), color=BLACK, lw=1.1, ls="--",
           label=r"$|R^2|\sim A/\varepsilon{+}B$ (mechanism)")
    ce = float(d["convdiv_eps"])
    c.axvspan(1.0, 4.5, color="0.93", zorder=0)
    c.scatter([ce], [1.0 - float(d["convdiv_r2"]) + 0.066], s=70, marker="*",
              color=GRAYBLUE, edgecolor=BLACK, lw=0.7, zorder=5,
              label=r"conv-div control ($\varepsilon{=}%.1f$)" % ce)
    c.axhline(1.0, color="0.6", lw=0.7, ls=":")
    c.text(2.3, 1.4, "tolerable", fontsize=7, color="0.35", ha="center")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel(r"cancellation depth $\varepsilon_{\mathrm{med}}$")
    c.set_ylabel(r"$|R^2(\tau_w)|$ (a-priori)")
    c.set_title(r"(c) Dose-response ($\rho_s{=}%+.2f$)" % float(d["fam_spearman"]))
    c.legend(loc="lower left", framealpha=0.9, fontsize=6.3)

    fig.tight_layout()
    for outdir, name in [(FIGDIR, "fig_coupled_norm_transfer.pdf"),
                         (NODE, "fig_coupled_norm_transfer.png")]:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), bbox_inches="tight")
        print("wrote", os.path.join(outdir, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
