#!/usr/bin/env python3
"""
Figure 14 (rebuilt): the genuinely 3-D class member -- aligned wall-mounted cube
array (Coceal 2006), OpenFOAM RANS.

(a) The 3-D geometry with the cancellation severity rendered as a "dusty
    pollutant" cloud (cf. the error-tracer style of fig. sharp_pair): error dust
    seeded in proportion to the local ODE relative-error severity ~ 1/epsilon
    from the MEASURED floor field, pooling across the canopy floor where the
    1-D ODE wall model fails domain-wide.
(b) Pitch sweep: the floor cancellation epsilon and the model failure R^2(tau_w)
    versus pitch p/h -- the mechanism switches off continuously as the array is
    widened (data: cube_p*.npz + the packed/sparse endpoints).
"""
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(PROJ, "manuscript", "figures")
RES = os.path.join(PROJ, "codes", "results")
RED = "#c0271e"; GRN = "#11823b"

# warm "pollutant/error" colormap: pale (low error) -> deep red (high error)
CMAP_ERR = LinearSegmentedColormap.from_list(
    "err", ["#fff0b3", "#fec44f", "#fb9a29", "#e6550d", "#a50f15", "#67000d"])

# ---- isometric projection -------------------------------------------------
C30, S30 = np.cos(np.radians(30)), np.sin(np.radians(30))
def iso(X, Y, Z):
    return (X - Z) * C30, (X + Z) * S30 + Y

def cube(ax, x0, z0, s, h, zbase, alpha=1.0):
    top = [iso(x0, h, z0), iso(x0 + s, h, z0), iso(x0 + s, h, z0 + s), iso(x0, h, z0 + s)]
    left = [iso(x0, 0, z0), iso(x0, h, z0), iso(x0, h, z0 + s), iso(x0, 0, z0 + s)]
    right = [iso(x0, 0, z0), iso(x0, h, z0), iso(x0 + s, h, z0), iso(x0 + s, 0, z0)]
    for k, (poly, fc) in enumerate([(left, "#7f8794"), (right, "#a6acb7"), (top, "#dde1e7")]):
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="#23262d", lw=0.7,
                             zorder=zbase + k, alpha=alpha))


# ---- panel (a): dusty cancellation cloud ----------------------------------
def panel_a(ax, rng):
    d = np.load(os.path.join(RES, "cube_array_wall_profiles.npz"))
    x0s, z0s, tw, dp = d["x"], d["z"], d["tau_w"], d["dp_dx"]; ym = float(d["y_m"])
    eps = np.abs(tw) / (np.maximum(np.abs(dp), 1e-9) * ym)        # measured floor cancellation
    # severity = ODE relative-error scaling ~ 1/eps (capped); 0..1 for colour
    sev = np.clip(1.0 / np.clip(eps, 1e-3, None), 0, 60.0)
    logc = np.clip(np.log10(np.clip(eps, 1e-3, 1e2)), -2, 1)      # colour by eps (log)
    cval = (1.0 - (logc + 2) / 3.0)                              # 1 = high error (eps small)

    # interpolate the 108-station unit cell onto a fine grid (nearest neighbour)
    from scipy.interpolate import NearestNDInterpolator
    fsev = NearestNDInterpolator(np.c_[x0s, z0s], sev)
    fcol = NearestNDInterpolator(np.c_[x0s, z0s], cval)

    NC, P, s, h = 3, 2.0, 1.0, 1.0          # 3x3 cubes, pitch 2h
    cube_lo, cube_hi = 0.5, 1.5             # cube footprint within each 2x2 cell

    # ground plane
    L = NC * P
    ground = [iso(0, 0, 0), iso(L, 0, 0), iso(L, 0, L), iso(0, 0, L)]
    ax.add_patch(Polygon(ground, closed=True, fc="#f4f5f7", ec="#d3d7de", lw=0.8, zorder=1))

    # seed dust over every cell's open floor, weighted by severity
    per_cell = 4200
    DUST = []
    for I in range(NC):
        for J in range(NC):
            # candidate points in the 2x2 cell, excluding the cube footprint
            xc = rng.uniform(0, P, per_cell * 3); zc = rng.uniform(0, P, per_cell * 3)
            inside = (xc > cube_lo) & (xc < cube_hi) & (zc > cube_lo) & (zc < cube_hi)
            xc, zc = xc[~inside], zc[~inside]
            w = fsev(xc, zc); w = np.clip(w, 0, None)
            if w.sum() <= 0:
                continue
            pk = w / w.sum()
            take = min(per_cell, len(xc))
            idx = rng.choice(len(xc), size=take, replace=True, p=pk)
            xx, zz = xc[idx] + I * P, zc[idx] + J * P
            cc = fcol(xc[idx] + 0, zc[idx] + 0)
            DUST.append((xx, zz, cc))
    DX = np.concatenate([d[0] for d in DUST])
    DZ = np.concatenate([d[1] for d in DUST])
    DC = np.concatenate([d[2] for d in DUST])

    # vertical haze: most dust near the floor, a little lifted into the canopy
    yy = rng.exponential(0.13, size=DX.size); yy = np.clip(yy, 0, 0.9)
    order = np.argsort(DX + DZ)   # far-to-near for occlusion against cubes
    DX, DZ, DC, yy = DX[order], DZ[order], DC[order], yy[order]
    sx, sy = iso(DX, yy, DZ)
    # two passes: a soft glow then a fine grain (dust UNDER the cubes)
    ax.scatter(sx, sy, c=DC, cmap=CMAP_ERR, vmin=0, vmax=1, s=26,
               alpha=0.035, edgecolors="none", zorder=3, rasterized=True)
    ax.scatter(sx, sy, c=DC, cmap=CMAP_ERR, vmin=0, vmax=1, s=4.0,
               alpha=0.16, edgecolors="none", zorder=3, rasterized=True)

    # cubes far-to-near so near occlude far (and the dust behind them)
    cells = [(i, j) for j in range(NC) for i in range(NC)]
    cells.sort(key=lambda c: (c[0] + c[1]))
    for d_, (i, j) in enumerate(cells):
        cube(ax, i * P + cube_lo, j * P + cube_lo, s, h, zbase=5 + 3 * d_)

    ax.set_xlim(-L * S30 - 0.8, L * C30 + 0.8); ax.set_ylim(-1.4, L * S30 + 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, 1.00, r"Cancellation severity $\sim 1/\varepsilon$ pools across the canopy floor",
            transform=ax.transAxes, ha="center", fontsize=9.6)
    ax.text(0.02, 0.05, r"packed: $\ell_p/\delta=1$,  $\tilde\varepsilon=0.017$,  $R^2(\tau_w)=-72$",
            transform=ax.transAxes, ha="left", fontsize=8.6, color=RED)
    # compact horizontal colour key, inset bottom-right
    cax = ax.inset_axes([0.62, 0.10, 0.34, 0.030])
    sm = plt.cm.ScalarMappable(cmap=CMAP_ERR, norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.06, 0.94]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=7.5, length=0)
    cax.set_title(r"ODE error severity $\sim 1/\varepsilon$", fontsize=7.8, pad=2)


def _load_sweep():
    """Collect (pitch, eps_med, r2, frac_lt0p1) from the endpoints + cube_p*.npz."""
    pts = {}
    def add(path):
        if not os.path.exists(path):
            return
        d = np.load(path, allow_pickle=True)
        if "ok" not in d.files or "pitch_over_h" not in d.files:
            return
        if not bool(np.atleast_1d(d["ok"])[0]):
            return
        p = float(d["pitch_over_h"])
        pts[round(p, 3)] = dict(pitch=p, eps=float(d["eps_med"]), r2=float(d["r2"]),
                                frac=float(d["frac_eps_lt0p1"]), lam=float(d["lambda_p"]))
    add(os.path.join(RES, "cube_array.npz"))     # packed  P=2
    add(os.path.join(RES, "cube_sparse.npz"))    # sparse  P=12
    for f in sorted(glob.glob(os.path.join(RES, "cube_p[0-9]*.npz"))):
        add(f)
    rows = sorted(pts.values(), key=lambda r: r["pitch"])
    return rows


def panel_b(ax):
    rows = _load_sweep()
    P = np.array([r["pitch"] for r in rows])
    E = np.array([r["eps"] for r in rows])
    F = np.array([r["frac"] for r in rows])
    n = len(rows)

    # shaded regime bands by the eps~O(1) criterion
    ax.axhspan(1e-3, 1.0, color="#fdecea", zorder=0)
    ax.axhspan(1.0, 1e2, color="#eef6ef", zorder=0)
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--", zorder=1)

    # median floor cancellation eps vs pitch (left, log)
    ax.plot(P, E, "-o", color=RED, mfc="white", mec=RED, mew=1.4, lw=1.8,
            ms=6, zorder=5, label=r"median $\varepsilon$ (measured)")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(1.7, 14); ax.set_ylim(8e-3, 12)
    ax.set_xticks([2, 3, 4, 6, 8, 12]); ax.set_xticklabels(["2", "3", "4", "6", "8", "12"])
    ax.set_xlabel(r"pitch $p/h$", fontsize=10)
    ax.set_ylabel(r"floor cancellation  median $\varepsilon$", fontsize=10, color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.text(13.5, 1.35, "ODE tolerated", color=GRN, ha="right", va="bottom", fontsize=8.4)
    ax.text(13.5, 0.74, "ODE fails", color=RED, ha="right", va="top", fontsize=8.4)

    # fraction of floor in deep cancellation (right, linear)
    ax2 = ax.twinx()
    ax2.plot(P, F, "-s", color="#1f5fa8", mfc="white", mec="#1f5fa8", mew=1.3,
             lw=1.4, ms=5, zorder=4, label=r"floor fraction $\varepsilon<0.1$")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel(r"floor fraction $\varepsilon<0.1$", fontsize=9.5, color="#1f5fa8")
    ax2.tick_params(axis="y", colors="#1f5fa8")

    # endpoint annotations
    if n:
        ax.annotate("packed\n(fails)", (P[0], E[0]), xytext=(2.05, 0.03),
                    fontsize=7.8, color=RED, ha="left")
        ax.annotate("sparse\n(tolerated)", (P[-1], E[-1]), xytext=(8.4, 5.2),
                    fontsize=7.8, color=GRN, ha="left")
    ax.set_title(r"Pitch alone switches the mechanism off", fontsize=9.6)
    if n < 4:
        ax.text(0.5, 0.5, f"sweep filling in\n({n} of 6 points)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.55", style="italic")


def _load_re():
    """Reynolds sweep at packed pitch: endpoint Re=5000 (cube_array.npz) + cube_re*.npz."""
    pts = {}
    def add(path):
        if not os.path.exists(path):
            return
        d = np.load(path, allow_pickle=True)
        if "ok" not in d.files or "Re_h" not in d.files or not bool(np.atleast_1d(d["ok"])[0]):
            return
        if abs(float(d["pitch_over_h"]) - 2.0) > 1e-6:
            return
        re = float(d["Re_h"])
        pts[round(re)] = dict(re=re, eps=float(d["eps_med"]), r2=float(d["r2"]),
                              frac=float(d["frac_eps_lt0p1"]))
    add(os.path.join(RES, "cube_array.npz"))
    for f in sorted(glob.glob(os.path.join(RES, "cube_re[0-9]*.npz"))):
        add(f)
    return sorted(pts.values(), key=lambda r: r["re"])


def panel_c(ax):
    rows = _load_re()
    R = np.array([r["re"] for r in rows]); E = np.array([r["eps"] for r in rows])
    F = np.array([r["frac"] for r in rows]); n = len(rows)
    ax.axhspan(1e-3, 1.0, color="#fdecea", zorder=0)
    ax.axhspan(1.0, 1e2, color="#eef6ef", zorder=0)
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--", zorder=1)
    ax.plot(R, E, "-o", color=RED, mfc="white", mec=RED, mew=1.4, lw=1.8, ms=6, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2000, 26000); ax.set_ylim(8e-3, 1.6)
    ax.set_xticks([2500, 5000, 10000, 20000]); ax.set_xticklabels(["2.5k", "5k", "10k", "20k"])
    ax.set_xlabel(r"Reynolds number $\mathit{Re}_h$  (packed, $p/h=2$)", fontsize=9.5)
    ax.set_ylabel(r"median $\varepsilon$", fontsize=9.5, color=RED)
    ax.tick_params(axis="y", colors=RED, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(r"and deepens with Reynolds number", fontsize=9.4)
    if n < 3:
        ax.text(0.5, 0.5, f"Re sweep filling in\n({n} of 4 points)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.55", style="italic")


def main():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(12.6, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0],
                          left=0.01, right=0.93, top=0.92, bottom=0.12,
                          wspace=0.30, hspace=0.55)
    axa = fig.add_subplot(gs[:, 0]); panel_a(axa, rng)
    axb = fig.add_subplot(gs[0, 1]); panel_b(axb)
    axc = fig.add_subplot(gs[1, 1]); panel_c(axc)
    fig.text(0.30, 0.95, "(a)", fontsize=11, fontweight="bold")
    fig.text(0.555, 0.95, "(b)", fontsize=11, fontweight="bold")
    fig.text(0.555, 0.47, "(c)", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig_cube_array_TEST.png"), dpi=170)
    plt.close(fig); print("wrote fig_cube_array_TEST.png")


if __name__ == "__main__":
    main()
