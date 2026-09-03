#!/usr/bin/env python3
r"""
fig_cross_geometry_conditioning_floor.py
========================================
The cross-geometry conditioning floor (L2, node_007).  Two panels, all numbers
from results/cross_geometry_conditioning_floor.npz (no fabrication):

 (a) kappa_closure vs eps for the model closures (A-D), pooled per geometry.
     The repeating-O(delta) FAILURE class -- smooth DNS hill, sharp WRLES rib,
     RANS wavy wall (three different shapes) -- sits on the conditioning FLOOR
     kappa ~ beta/eps (the shaded beta band, O(10^-2)); the non-repeating /
     wide-pitch / localised-separation CONTROLS (BFS, conv-div, NASA hump,
     separation bubble) sit BELOW it (well-conditioned, kappa bounded O(1)).

 (b) floor prefactor beta = median(kappa*eps) vs eps_median per geometry.  The
     failure class collapses into one decade (O(10^-2)) regardless of shape; the
     prefactor TRACKS the cancellation depth among the high-fidelity members
     (deeper eps -> larger beta), which is why the hill (0.062) and rib (0.013)
     differ by ~4x while sharing the floor.

Colour key is geometry-based (this figure is about geometries, not the reserved
model-type palette).  Failure class in warm hues, controls in cool/grey.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NPZ = os.path.join(RESULTS, "cross_geometry_conditioning_floor.npz")
MS_FIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

KEYS_AD = ["A_ml_vandriest", "B_alg_cebeci", "C_alg_sa", "D_alg_reichardt"]

# geometry display: label, colour, marker, role
GEO = {
    "periodic_hill_1p0": dict(lab="periodic hill (DNS)",      c="#d7301f", m="o", role="failure"),
    "sharp_rib_dtype":   dict(lab="sharp rib (WRLES)",        c="#ef6548", m="s", role="failure"),
    "wavy_a10":          dict(lab="wavy wall (RANS)",         c="#fc8d59", m="^", role="failure"),
    "bfs_Re13700":       dict(lab="backward step (LES)",      c="#4575b4", m="v", role="control"),
    "conv_div_Re12600":  dict(lab="conv-div channel (DNS)",   c="#74add1", m="D", role="control"),
    "nasa_hump":         dict(lab="NASA hump (LES)",          c="#878787", m="P", role="control"),
    "sep_bubble_caseB":  dict(lab="separation bubble (DNS)",  c="#bababa", m="X", role="control"),
    "wavy_flat":         dict(lab="near-flat $a\\to0$ (RANS)", c="#1a9850", m="*", role="transition"),
}


def main():
    d = np.load(NPZ, allow_pickle=True)
    tags = list(d["tags"])
    roles = {t: r for t, r in zip(d["tags"], d["roles"])}
    eps_med = {t: e for t, e in zip(d["tags"], d["eps_med"])}
    beta = {t: b for t, b in zip(d["tags"], d["beta_model"])}
    beta_lo, beta_hi = float(d["beta_lo"]), float(d["beta_hi"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- panel (a): kappa vs eps, pooled model closures, per geometry --------
    eps_all_fail = []
    for t in tags:
        g = GEO.get(t)
        if g is None or t == "wavy_flat":
            continue
        eps = d["eps__" + t]
        kk = []
        for key in KEYS_AD:
            arr = d.get("kappa__%s__%s" % (t, key))
            if arr is not None:
                kk.append(arr)
        if not kk:
            continue
        kpool = np.nanmedian(np.vstack(kk), axis=0)        # median over A-D per station
        m = np.isfinite(eps) & np.isfinite(kpool) & (eps > 0) & (kpool > 0)
        ax.scatter(eps[m], kpool[m], s=16, c=g["c"], marker=g["m"],
                   alpha=0.55, edgecolors="none",
                   label=g["lab"] + ("  [fail]" if g["role"] == "failure" else "  [ctrl]"))
        if g["role"] == "failure":
            eps_all_fail.append(eps[m])

    # the conditioning-floor band kappa = beta/eps over the cancellation regime
    epsline = np.logspace(np.log10(2e-3), np.log10(1.0), 100)
    ax.fill_between(epsline, beta_lo / epsline, beta_hi / epsline,
                    color="0.55", alpha=0.25, zorder=0,
                    label=r"floor $\kappa=\beta/\varepsilon$, $\beta\in[%.3f,%.3f]$" % (beta_lo, beta_hi))
    ax.plot(epsline, np.median([beta_lo, beta_hi]) / epsline, "--", color="0.35", lw=1.0, zorder=1)
    ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
    ax.text(1.05, ax.get_ylim()[0] if False else 2e-3, r"$\varepsilon=1$", color="0.5", fontsize=8)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"force-residual ratio  $\varepsilon=|\tau_w|/(|dp/dx|\,y_m)$")
    ax.set_ylabel(r"closure-channel condition number  $\kappa_{\rm closure}$ (median A--D)")
    ax.set_title("(a) one conditioning floor across shapes")
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9, ncol=1)
    ax.grid(True, which="both", alpha=0.2)

    # ---- panel (b): floor prefactor beta vs eps_median per geometry ----------
    bx.axhspan(1e-2, 1e-1, color="0.7", alpha=0.18, zorder=0)
    bx.text(0.03, 3.0e-2, r"$\beta\sim O(10^{-2})$ floor", fontsize=8, color="0.35")
    for t in tags:
        g = GEO.get(t)
        if g is None:
            continue
        b = beta[t]
        if not np.isfinite(b) or b <= 0:
            continue
        bx.scatter([eps_med[t]], [b], s=90, c=g["c"], marker=g["m"],
                   edgecolors="k", linewidths=0.6, zorder=3, label=g["lab"])
    # the high-fidelity depth trend (hill -> rib): deeper eps -> larger beta
    hi = [("periodic_hill_1p0",), ("sharp_rib_dtype",)]
    xs = [eps_med[t[0]] for t in hi if t[0] in eps_med]
    ys = [beta[t[0]] for t in hi if t[0] in beta]
    bx.plot(xs, ys, "-", color="#d7301f", lw=1.2, alpha=0.7, zorder=2)
    bx.annotate("deeper cancellation\n$\\Rightarrow$ larger prefactor",
                xy=(xs[0], ys[0]), xytext=(0.2, 0.011), fontsize=7.5, color="#d7301f",
                arrowprops=dict(arrowstyle="->", color="#d7301f", lw=0.9))

    bx.axvline(1.0, color="0.7", lw=0.8, ls=":")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(r"cancellation depth  $\varepsilon_{\rm median}$")
    bx.set_ylabel(r"floor prefactor  $\beta=\mathrm{median}(\kappa_{\rm closure}\,\varepsilon)$")
    bx.set_title("(b) shape-agnostic floor; prefactor tracks depth")
    bx.legend(fontsize=6.4, loc="upper left", framealpha=0.9)
    bx.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    for outdir in (os.path.join(CODES, "figures"), MS_FIG):
        os.makedirs(outdir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, "fig_cross_geometry_conditioning_floor." + ext),
                        dpi=160, bbox_inches="tight")
    print("wrote fig_cross_geometry_conditioning_floor.{pdf,png} to codes/figures and manuscript/figures")


if __name__ == "__main__":
    main()
