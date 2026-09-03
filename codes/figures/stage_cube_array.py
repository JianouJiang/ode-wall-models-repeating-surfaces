#!/usr/bin/env python3
"""
Restored CUBE-ARRAY figure (the genuinely 3-D class member, not pseudo-2-D).
Was panel (b) of the removed blade+cube schematic; rebuilt clean + from REAL data.

(a) clean isometric schematic of the aligned wall-mounted cube array (urban canopy,
    pitch p=2h), mesh-free.
(b) the MEASURED floor cancellation: epsilon = |tau_w|/(|dp/dx| y_m) over the 108
    floor stations of the OpenFOAM unit-cell RANS -- eps<0.1 domain-wide (median
    0.017, 92% of the floor) => catastrophic (R^2=-72). The sparse control
    (pitch 6h) sits at eps_med=4.1, tolerated: pitch alone switches it off.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(PROJ, "manuscript", "figures")
RES = os.path.join(PROJ, "codes", "results")
RED = "#c0271e"; GRN = "#11823b"

C30, S30 = np.cos(np.radians(30)), np.sin(np.radians(30))
def iso(X, Y, Z):
    return (X - Z) * C30, (X + Z) * S30 + Y

def cube(ax, x0, z0, s, h, zbase):
    # three visible faces of a cube sitting on the floor at (x0,z0)
    top = [iso(x0, h, z0), iso(x0 + s, h, z0), iso(x0 + s, h, z0 + s), iso(x0, h, z0 + s)]
    # viewer-facing vertical faces: X=x0 (front-left) and Z=z0 (front-right)
    left = [iso(x0, 0, z0), iso(x0, h, z0), iso(x0, h, z0 + s), iso(x0, 0, z0 + s)]
    right = [iso(x0, 0, z0), iso(x0, h, z0), iso(x0 + s, h, z0), iso(x0 + s, 0, z0)]
    for k, (poly, fc) in enumerate([(left, "#7f8794"), (right, "#a6acb7"), (top, "#dde1e7")]):
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="#23262d", lw=0.7, zorder=zbase + k))


def panel_a(ax):
    # ground plane
    n, p, s, h = 2, 2.2, 1.0, 1.0
    L = n * p
    ground = [iso(0, 0, 0), iso(L, 0, 0), iso(L, 0, L), iso(0, 0, L)]
    ax.add_patch(Polygon(ground, closed=True, fc="#f1f2f4", ec="#cfd3da", lw=0.8, zorder=1))
    # cubes drawn FAR-to-NEAR (large x+z first) with increasing zorder so near occludes far
    cells = [(i, j) for j in range(n) for i in range(n)]
    cells.sort(key=lambda c: -(c[0] + c[1]))
    for d, (i, j) in enumerate(cells):
        cube(ax, i * p + 0.6, j * p + 0.6, s, h, zbase=5 + 3 * d)
    ax.annotate("", iso(0.6, 0, -0.35), iso(2.8, 0, -0.35),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
    ax.text(*iso(1.7, 0, -1.05), r"pitch $p=2h$", ha="center", fontsize=9.5)
    ax.set_xlim(-L * S30 - 1.0, L * C30 + 1.0); ax.set_ylim(-1.8, L * S30 + 1.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, 1.02, "Aligned wall-mounted cube array (urban canopy) --- fully 3-D",
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(0.98, 0.94, "ODE FAILS", transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec="none"))
    ax.text(0.02, 0.06, r"$\ell_p/\delta=1$,  $\tilde\varepsilon=0.017$,  $R^2(\tau_w)=-72$",
            transform=ax.transAxes, ha="left", fontsize=9.0)


def panel_b(ax):
    d = np.load(os.path.join(RES, "cube_array_wall_profiles.npz"))
    x, z, tw, dp = d["x"], d["z"], d["tau_w"], d["dp_dx"]; ym = float(d["y_m"])
    eps = np.abs(tw) / (np.maximum(np.abs(dp), 1e-9) * ym)
    # cube footprint (side 1, centred in the 2x2 cell)
    ax.add_patch(Rectangle((0.5, 0.5), 1.0, 1.0, fc="#8f95a3", ec="#2b2f38", lw=1.0, zorder=4))
    ax.text(1.0, 1.0, "cube", ha="center", va="center", fontsize=8.5, color="white", zorder=5)
    sc = ax.scatter(x, z, c=np.log10(np.clip(eps, 1e-3, 1e2)), s=70, cmap="RdYlBu",
                    vmin=-2, vmax=1, edgecolors="#555", linewidths=0.3, zorder=3)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xlabel(r"$x/h$", fontsize=10); ax.set_ylabel(r"$z/h$", fontsize=10)
    ax.set_title(r"Measured floor cancellation $\varepsilon=|\tau_w|/(|\mathrm{d}p/\mathrm{d}x|\,y_m)$"
                 + "\n" + r"$\varepsilon<0.1$ over $92\%$ of the floor $\Rightarrow$ domain-wide",
                 fontsize=9.5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([-2, -1, 0, 1]); cb.set_ticklabels(["0.01", "0.1", "1", "10"])
    cb.set_label(r"$\varepsilon$  (log scale)", fontsize=9)
    cb.ax.axhline((np.log10(0.1) + 2) / 3, color="#c0271e", lw=1.4)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                   gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    fig.subplots_adjust(left=0.02, right=0.97, top=0.9, bottom=0.12, wspace=0.18)
    panel_a(ax1); panel_b(ax2)
    fig.text(0.27, 0.965, "(a)", fontsize=11, fontweight="bold")
    fig.text(0.59, 0.965, "(b)", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig_cube_array.pdf"))
    fig.savefig(os.path.join(OUT, "fig_cube_array.png"), dpi=170)
    plt.close(fig); print("wrote fig_cube_array.pdf/.png")


if __name__ == "__main__":
    main()
