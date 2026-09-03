#!/usr/bin/env python3
r"""
fig_transition_map_l2.py
========================
Headline L2 figure for the transition-map iteration.  All numbers from
results/transition_map_l2.npz (no fabrication; read-only).

 (a) The (amplitude a/delta  x  pitch lambda/delta) failure map of the
     repeating-structure class.  The 29 Xiao parameterised DNS hills tile the
     interior (all catastrophic, coloured by R2 severity); the smooth WAVY wall
     and the SHARP square ribs are overlaid as the genuinely-in-framework
     intermediate geometries.  The 3-D diffuser is NOT shown -- it is out of the
     quasi-2-D framework (it fails by spanwise transport) and is no longer needed
     to anchor the boundary (bind B-L2-1).

 (b) The cancellation-depth discriminant: R2(tau_w) vs eps_med for the
     in-framework geometries.  Among SMOOTH structures eps cleanly separates fail
     (low eps) from pass (high eps) -- the shaded bracket.  The SHARP rib (LES)
     BREAKS the smooth ordering: it is catastrophic at eps=0.52 where the smooth
     krank hill is tolerated -- the skin-friction eps under-counts the
     cancellation for sharp edges (named second group = edge-sharpness/form-drag,
     the pre-registered H2 break, gate G11).

Geometry-based colour key (warm = catastrophic, cool = tolerated); sharp ribs use
square markers to make the smooth-vs-sharp break legible.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Print-size parity with the dose-response reference figure (11 pt on a 7.0 in
# canvas at 0.85\textwidth => ~7.1 pt printed labels): this canvas is 11.4 in
# wide at full \textwidth, so labels need ~15 pt (ticks/legend ~13.5 pt).
plt.rcParams.update({"font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                     "legend.fontsize": 13.5})

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NPZ = os.path.join(RESULTS, "transition_map_l2.npz")
MS_FIG = os.path.join(os.path.dirname(CODES), "manuscript", "figures")

FAIL_C = "#d7301f"     # catastrophic (warm)
PASS_C = "#4575b4"     # tolerated (cool)


def main():
    d = np.load(NPZ, allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    shape = [str(s) for s in d["shape"]]
    eps = np.asarray(d["eps"], float)
    r2 = np.asarray(d["r2"], float)
    fail = np.asarray(d["fail"], int).astype(bool)

    xa = np.asarray(d["xiao_a_over_delta"], float)
    xl = np.asarray(d["xiao_lambda_over_delta"], float)
    xr2 = np.asarray(d["xiao_r2"], float)

    def get(k):
        i = keys.index(k)
        return eps[i], r2[i], fail[i]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # ===================== panel (a): a/delta x lambda/delta map ============
    sc = axA.scatter(xl, xa, c=xr2, cmap="autumn", vmin=xr2.min(), vmax=0,
                     s=70, marker="o", edgecolor="k", linewidth=0.4, zorder=3,
                     label="_nolegend_")
    cb = fig.colorbar(sc, ax=axA, pad=0.02)
    cb.set_label(r"$R^2(\tau_w)$  (Xiao DNS hills)")

    # overlay the in-framework intermediates with real (a/delta, lambda/delta)
    overlays = [
        ("wavy_a10", 0.10, 2.0, "^", "wavy wall (RANS)"),
        ("rib_les_dtype", 0.20, 0.60, "s", "sharp rib d-type (LES)"),
        ("rib_rans_dtype", 0.20, 0.60, "s", None),
        ("rib_rans_ktype", 0.20, 1.60, "D", "sharp rib k-type (RANS)"),
        ("krank_pehill_Re10595", 0.659, 5.93, "P", "krank hill (DNS, pass)"),
    ]
    for k, a, l, m, lab in overlays:
        _, rr, fl = get(k)
        # pass/fail encoded by FILL as well as colour (colour-independent):
        # fail = filled, pass = open.
        if fl:
            axA.scatter(l, a, s=170, marker=m, facecolor=FAIL_C,
                        edgecolor="k", linewidth=1.1, zorder=5, label=lab)
        else:
            axA.scatter(l, a, s=170, marker=m, facecolor="none",
                        edgecolor=PASS_C, linewidth=1.8, zorder=5, label=lab)
    # pitch symbol: \ell_p (paper-wide convention; \lambda is the
    # forcing-to-stress ratio of eq. eps_hat)
    axA.set_xlabel(r"pitch  $\ell_p/\delta$")
    axA.set_ylabel(r"amplitude  $a/\delta$")
    axA.set_title("(a)  repeating-structure failure map", loc="left", fontsize=15)
    # legend: the 29 Xiao circles are coloured by the R^2 colourbar (yellow->red),
    # so their key is a NEUTRAL grey circle pointing at the colourbar, not a single
    # red marker; the overlays keep their pass/fail colours. Placed lower-right in
    # the empty corner so it covers no data points.
    xiao_proxy = Line2D([], [], marker="o", ls="none", mfc="0.8", mec="k",
                        mew=0.6, markersize=8,
                        label=r"Xiao hills (DNS): colour $=R^2$")
    h, l = axA.get_legend_handles_labels()
    axA.legend([xiao_proxy] + h, [xiao_proxy.get_label()] + l, fontsize=11,
               loc="lower right", framealpha=0.92, borderpad=0.5,
               labelspacing=0.35)
    axA.grid(alpha=0.25)

    # ===================== panel (b): R2 vs eps discriminant ================
    # smooth in-framework
    smooth = {
        "periodic_hills_1p0": ("o", "periodic hill"),
        "wavy_a10": ("^", "wavy wall"),
        "krank_pehill_Re10595": ("P", "krank hill"),
        "conv_div_channel": ("D", "conv–div channel"),
    }
    sharp = {
        "rib_les_dtype": ("s", "sharp rib d-type"),
        "rib_rans_dtype": ("s", None),
        "rib_rans_ktype": ("X", "sharp rib k-type"),
    }
    # shaded SMOOTH separation bracket, over ALL smooth members: the named
    # cases plotted here AND the 29-hill Xiao family (max failing eps 0.41,
    # panel a), so the shaded gap is empty family-wide -- the (0.41, 0.52)
    # bracket quoted in the text.
    xiao_eps = np.asarray(d["xiao_eps"], float)
    emf = max(float(d["br_smooth_eps_max_fail"]), float(np.nanmax(xiao_eps)))
    emp = float(d["br_smooth_eps_min_pass"])
    axB.axvspan(emf, emp, color="0.85", zorder=0,
                label=f"smooth bracket\n[{emf:.2f}, {emp:.2f}]")
    axB.axhline(0.0, color="k", lw=0.8, ls="--", zorder=1)

    for k, (m, lab) in smooth.items():
        e, rr, fl = get(k)
        axB.scatter(e, max(rr, -2.2), s=120, marker=m,
                    c=(FAIL_C if fl else PASS_C), edgecolor="k",
                    linewidth=0.9, zorder=4, label=lab)
    for k, (m, lab) in sharp.items():
        e, rr, fl = get(k)
        axB.scatter(e, max(rr, -2.2), s=150, marker=m,
                    facecolor=(FAIL_C if fl else PASS_C), edgecolor="k",
                    linewidth=1.6, zorder=5, label=lab)

    # annotate the sharp break
    e_rib, r_rib, _ = get("rib_les_dtype")
    e_kr, r_kr, _ = get("krank_pehill_Re10595")
    axB.annotate("sharp rib FAILS where\nsmooth PASSES (named\n2nd group: edge-sharpness)",
                 xy=(e_rib, max(r_rib, -2.2)), xytext=(1.15, -2.05),
                 fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    axB.set_xscale("log")
    axB.set_xlabel(r"cancellation depth  $\tilde\varepsilon = \mathrm{med}\,|\tau_w|/(|dp/dx|\,y_m)$")
    axB.set_ylabel(r"$R^2(\tau_w)$  (clipped at $-2.2$)")
    axB.set_title("(b)  smooth transition + sharp break", loc="left", fontsize=15)
    axB.set_ylim(-2.4, 1.15)
    axB.legend(fontsize=11, loc="upper left", framealpha=0.9, ncol=1)
    axB.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(MS_FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_transition_map_l2.{ext}"),
                    dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(MS_FIG, f"fig_transition_map_l2.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("Saved fig_transition_map_l2.{pdf,png} -> codes/figures + manuscript/figures")


if __name__ == "__main__":
    main()
