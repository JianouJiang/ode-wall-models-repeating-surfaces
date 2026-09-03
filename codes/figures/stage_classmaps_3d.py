#!/usr/bin/env python3
"""
Pseudo-3-D CLASS MAPS for the repeating-structure family (figs 5 & 6).

Each grid cell is the SAME geometry member as the flat version, but drawn as an
oblique (cabinet-projection) extrusion of the wall profile in the spanwise
direction.  The receding top surface carries a streamwise x spanwise mesh that
reads as a 3-D wall; the *near-reader* front cross-section face is left as a
clean solid fill (NO mesh grid) per supervisor request -- the front-face grid
was the cluttered part.

Only cells backed by an on-disk simulation carry a FAIL/HOLD badge.  The
remaining analytic geometry sketches are labelled NOT SIMULATED; no predicted
verdict is drawn.  This prevents the parameter grid from being mistaken for a
simulation matrix.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figures"))

# natural (Lambert) shading: every visible surface element is tinted by how its
# OUTWARD NORMAL faces a fixed light (upper-left), giving smooth slopes a continuous
# gradient instead of flat per-face tones, while flat tops / risers still separate.
LIGHT = np.array([-0.42, 0.907]); LIGHT = LIGHT / np.hypot(*LIGHT)   # from upper-left


def _gray(inten):
    inten = float(np.clip(inten, 0.0, 1.0))
    return (min(inten * 0.97, 1.0), min(inten * 0.99, 1.0), min(inten * 1.06, 1.0))


def skin_color(dx, dy):
    # outward normal of the z-extruded profile segment, in the x-y plane: (-dy, dx)
    nx, ny = -dy, dx; nn = (nx * nx + ny * ny) ** 0.5
    if nn < 1e-12:
        return _gray(0.72)
    lam = max(0.0, (nx * LIGHT[0] + ny * LIGHT[1]) / nn)
    return _gray(0.58 + 0.28 * lam)


FRONT_COLOR = _gray(0.72)     # near cross-section: faces the viewer, evenly lit
DEPTH_COLOR = _gray(0.55)     # hidden back / left occluders (deep shadow)
GROUND     = "#ecedf1"
EDGE       = "#2b2f38"
RED = "#c0271e"; GRN = "#11823b"; UNSIM = "#5f6670"

# oblique depth vector (data units): sweep z in [0,1] -> shift up-right.
# Kept shallow so dense (high-repeat) cells read as short 3-D prisms rather
# than a long diagonal wireframe.
DX, DY = 0.16, 0.15
# the solid body extends below the valley floor to BASE, so every member reads as
# a continuous SOLID wall with a structured top (not hollow teeth on an empty floor).
BASE = -0.13
ALPHA = 0.80             # semi-transparent fill -> clean translucent read


# ---------- wall-profile generators (cell width 1, baseline 0) ----------
def smooth_hill(n, a):
    x = np.linspace(0, 1, 800); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; hw = p * 0.40; d = (x - c) / hw
        y += a * np.where(np.abs(d) <= 1, 0.5 * (1 + np.cos(np.pi * np.clip(d, -1, 1))), 0.0)
    return x, y

def wavy(n, a):
    x = np.linspace(0, 1, 800); return x, a * 0.5 * (1 - np.cos(2 * np.pi * n * x))

def sharp_rib(n, a):
    x = np.linspace(0, 1, 2400); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; w = p * 0.42
        y[np.abs(x - c) <= w / 2] = a
    return x, y

def rounded_rib(n, a):
    x = np.linspace(0, 1, 2400); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; w = p * 0.42; r = w * 0.42; d = np.abs(x - c)
        flat = d <= (w / 2 - r); slope = (d > (w / 2 - r)) & (d <= w / 2)
        y[flat] = a
        y[slope] = a * 0.5 * (1 + np.cos(np.pi * (d[slope] - (w / 2 - r)) / r))
    return x, y

def gauss_bump(a):
    x = np.linspace(0, 1, 800); return x, a * np.exp(-((x - 0.5) / 0.13) ** 2)

def bfs(a):
    x = np.linspace(0, 1, 800); y = np.where(x < 0.32, a, 0.0); return x, y

def single_round(a):
    x = np.linspace(0, 1, 800); d = np.abs(x - 0.5); w = 0.34; r = 0.14
    y = np.where(d <= (w / 2 - r), a, 0.0)
    sl = (d > (w / 2 - r)) & (d <= w / 2)
    y[sl] = a * 0.5 * (1 + np.cos(np.pi * (d[sl] - (w / 2 - r)) / r)); return x, y


# ---------- pseudo-3-D cell renderer ----------
def draw_cell(ax, x, y, badge, filled, label, tag):
    # ground shadow plane at the solid base
    ax.fill([0, 1, 1 + DX, DX], [BASE, BASE, BASE + DY, BASE + DY], color=GROUND, ec="none", zorder=0)

    # Render the SOLID extruded prism as explicit, fully-filled faces.  A dense
    # resample lets the sharp risers come through as their own (side-shaded) faces.
    NS = 200
    xs = np.linspace(0, 1, NS); ys = np.interp(xs, x, y)

    # (i) BACK cross-section + LEFT end face: occluders so nothing shows through
    ax.fill(np.r_[DX, xs + DX, 1 + DX], np.r_[BASE + DY, ys + DY, BASE + DY],
            color=DEPTH_COLOR, ec="none", zorder=1)
    ax.fill([0, DX, DX, 0], [BASE, BASE + DY, ys[0] + DY, ys[0]],
            color=DEPTH_COLOR, ec="none", zorder=1)

    # (ii) swept SKIN, one quad per profile segment: each tinted by Lambert shading
    # of its outward normal -> smooth slopes get a continuous gradient (ec=face
    # colour closes the thin anti-alias seams between adjacent quads)
    for i in range(NS - 1):
        dy = ys[i + 1] - ys[i]; dx = xs[i + 1] - xs[i]
        c = skin_color(dx, dy)
        ax.fill([xs[i], xs[i + 1], xs[i + 1] + DX, xs[i] + DX],
                [ys[i], ys[i + 1], ys[i + 1] + DY, ys[i] + DY], color=c, ec=c, lw=0.5, zorder=2)

    # (iii) RIGHT end face (x=1 swept): right-facing -> in shadow
    rc = skin_color(0.0, -1.0)
    ax.fill([1, 1, 1 + DX, 1 + DX], [BASE, ys[-1], ys[-1] + DY, BASE + DY],
            color=rc, ec=rc, lw=0.5, zorder=2)

    # (iv) near FRONT cross-section face, on top
    ax.fill(np.r_[0.0, x, 1.0], np.r_[BASE, y, BASE], color=FRONT_COLOR, ec="none", zorder=5)
    ax.plot(xs + DX, ys + DY, color=EDGE, lw=0.5, alpha=0.5, zorder=4)  # back top edge
    ax.plot(x, y, color=EDGE, lw=1.0, zorder=6)                        # front top edge
    ax.plot([0, 1], [BASE, BASE], color=EDGE, lw=1.0, zorder=6)        # solid base
    ax.plot([0, 0], [BASE, y[0]], color=EDGE, lw=1.0, zorder=6)        # left side
    ax.plot([1, 1], [BASE, y[-1]], color=EDGE, lw=1.0, zorder=6)       # right side

    # frame / housekeeping
    ax.set_xlim(-0.05, 1.27); ax.set_ylim(BASE - 0.06, 1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cfd3da"); s.set_linewidth(0.8)

    # badges / tags / labels (axes-fraction => independent of the geometry)
    if filled:
        col = RED if badge == "FAIL" else GRN
        badge_text = badge
        bstyle = dict(fc=col, ec=col, lw=1.2)
        badge_size = 13.5
    else:
        # An unrun cell is a geometry sketch, not a model prediction.  Never
        # pass its former hard-coded FAIL/HOLD value to the renderer.
        assert badge is None
        badge_text = "NOT SIMULATED"
        bstyle = dict(fc="white", ec=UNSIM, lw=1.2, ls=(0, (3, 2)))
        badge_size = 9.2
    badge_y = 0.97 if filled else (0.74 if label else 0.925)
    ax.text(0.975, badge_y, badge_text,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=badge_size, fontweight="bold",
            color=("white" if filled else UNSIM),
            bbox=dict(boxstyle="round,pad=0.22", **bstyle))
    ax.text(0.025, 0.97, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, color="#444")
    if label:
        # named reference case -> placed INSIDE, just under the top of the panel
        # (above the geometry), to save the inter-row room the bottom labels took
        ax.text(0.5, 0.885, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=12, color="#333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#999", lw=0.8, alpha=0.92))




ANN = "#8a5a00"   # dimension-arrow colour (matches Fig 1's pitch arrows)

def dim_arrows(ax, n, a, hlab):
    """Mark the pitch (crest-to-crest) and the element height/amplitude on one
    2-repeat cell, so the grid's ratio axes are defined visually."""
    p = 1.0 / n
    x1, x2 = 0.5 * p, 1.5 * p                     # the two crest centres
    ya = a + 0.17
    ax.annotate("", xy=(x2, ya), xytext=(x1, ya),
                arrowprops=dict(arrowstyle="<|-|>", color=ANN, lw=1.3,
                                mutation_scale=10), zorder=9)
    ax.text(0.5 * (x1 + x2), ya + 0.035, r"$\ell_p$", ha="center", va="bottom",
            fontsize=15.5, color=ANN, zorder=9)
    for xc in (x1, x2):                            # dotted risers from the crests
        ax.plot([xc, xc], [a + 0.02, ya], ls=(0, (2, 2)), color=ANN, lw=0.9,
                zorder=9)
    xv = 0.055                                     # height arrow at the left edge
    ax.annotate("", xy=(xv, a), xytext=(xv, 0.0),
                arrowprops=dict(arrowstyle="<|-|>", color=ANN, lw=1.3,
                                mutation_scale=10), zorder=9)
    ax.plot([xv, x1], [a, a], ls=(0, (2, 2)), color=ANN, lw=0.9, zorder=9)
    ax.text(xv + 0.025, 0.5 * a, hlab, ha="left", va="center",
            fontsize=15.5, color=ANN, zorder=9)

# ---------- figure 5: morphology ----------
def make_morphology():
    ncol = [10, 5, 2, 1]
    col_lab = [r"$\ell_p/h=1.0$" + "\n(10 repeats)", r"$\ell_p/h=2.0$" + "\n(5 repeats)",
               r"$\ell_p/h=5.0$" + "\n(2 repeats)", r"$\ell_p\to\infty$" + "\n(non-repeating)"]
    rows = ["smooth hill", "rounded rib", "sharp square rib"]
    spec = [
        [("FAIL",1,"periodic hill"),("FAIL",1,"wavy wall"),("HOLD",1,"krank hill"),(None,0,"Gaussian bump")],
        [(None,0,None),(None,0,None),(None,0,None),(None,0,None)],
        [("FAIL",1,"rib $d$-type"),("HOLD",1,"rib $k$-type"),(None,0,None),("HOLD",1,"BFS")],
    ]
    shapef = [smooth_hill, rounded_rib, sharp_rib]
    tags = "abcdefghijkl"
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 6.2))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.84, bottom=0.075, hspace=0.22, wspace=0.10)
    t = 0
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]; n = ncol[c]
            if c == 3:
                x, y = (gauss_bump(0.55) if r == 0 else single_round(0.55) if r == 1 else bfs(0.55))
            else:
                x, y = shapef[r](n, 0.55 if r != 0 else 0.5)
            b, f, lab = spec[r][c]
            draw_cell(ax, x, y, b, f, lab, f"({tags[t]})"); t += 1
            if r == 0 and c == 2:
                dim_arrows(ax, 2, 0.5, r"$h$")
            if r == 2 and c == 1:
                # rib height k on the k-type cell (rib side, roughness convention)
                xv, ak = 0.035, 0.55
                ax.annotate("", xy=(xv, ak), xytext=(xv, 0.0),
                            arrowprops=dict(arrowstyle="<|-|>", color=ANN,
                                            lw=1.3, mutation_scale=10), zorder=9)
                ax.plot([xv, 0.10], [ak, ak], ls=(0, (2, 2)), color=ANN,
                        lw=0.9, zorder=9)
                ax.text(xv - 0.02, 0.5 * ak, r"$k$", ha="right", va="center",
                        fontsize=15.5, color=ANN, zorder=9)
            if r == 0:
                ax.set_title(col_lab[c], fontsize=16, pad=6)
        axes[r, 0].text(-0.145, 0.5, rows[r], transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=15.5)
    fig.text(0.54, 0.965, r"increasing pitch  $\ell_p/h$  (to the non-repeating limit) $\rightarrow$",
             ha="center", fontsize=18.5)
    # rotation=90 -> reads bottom->top like the row labels and other figures' row
    # axes; the arrow is $\leftarrow$, which renders pointing DOWN at rotation=90,
    # i.e. toward the sharp-edged rows (edge sharpness increases downward).
    fig.text(0.018, 0.46, r"increasing edge sharpness $\leftarrow$", rotation=90, va="center", fontsize=16.5)
    fig.savefig(os.path.join(OUT, "fig_repeating_class_morphology.pdf"))
    fig.savefig(os.path.join(OUT, "fig_repeating_class_morphology.png"), dpi=170)
    plt.close(fig); print("wrote fig_repeating_class_morphology (pseudo-3D)")


# ---------- figure 6: amplitude ----------
def make_amplitude():
    ncol = [8, 4, 2, 1]
    col_lab = [r"$\ell_p/h=1.25$" + "\n(8 repeats)", r"$\ell_p/h=2.50$" + "\n(4 repeats)",
               r"$\ell_p/h=5.00$" + "\n(2 repeats)", r"$\ell_p\to\infty$" + "\n(non-repeating)"]
    amps = [0.30, 0.52, 0.78, 1.02]; row_lab = [r"$a/h=0.30$", r"$a/h=0.55$", r"$a/h=0.80$", r"$a/h=1.05$"]
    spec = [
        [(None,0,None),("FAIL",1,"wavy wall"),(None,0,None),(None,0,None)],
        [(None,0,None),(None,0,None),("HOLD",1,"krank hill"),("HOLD",1,"Gaussian bump")],
        [(None,0,None),(None,0,None),("HOLD",1,"conv--div ch."),(None,0,None)],
        [(None,0,None),("FAIL",1,"periodic hill"),(None,0,None),(None,0,None)],
    ]
    tags = "abcdefghijklmnop"
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 7.8))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.86, bottom=0.07, hspace=0.24, wspace=0.10)
    t = 0
    for r in range(4):
        for c in range(4):
            ax = axes[r, c]; n = ncol[c]
            x, y = (gauss_bump(amps[r]) if c == 3 else smooth_hill(n, amps[r]))
            b, f, lab = spec[r][c]
            draw_cell(ax, x, y, b, f, lab, f"({tags[t]})"); t += 1
            if r == 3 and c == 2:
                dim_arrows(ax, 2, 1.02, r"$a$")
            if r == 0:
                ax.set_title(col_lab[c], fontsize=16, pad=6)
        axes[r, 0].text(-0.13, 0.5, row_lab[r], transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=15.5)
    fig.text(0.54, 0.97, r"increasing pitch  $\ell_p/h$  (to the non-repeating limit) $\rightarrow$",
             ha="center", fontsize=18.5)
    # rotation=90 to read bottom->top like the row labels; the arrow $\leftarrow$
    # renders pointing DOWN (toward larger a/h; amplitude increases downward).
    fig.text(0.018, 0.47, r"increasing amplitude $a/h$ $\leftarrow$", rotation=90, va="center", fontsize=16.5)
    fig.savefig(os.path.join(OUT, "fig_repeating_class_amplitude.pdf"))
    fig.savefig(os.path.join(OUT, "fig_repeating_class_amplitude.png"), dpi=170)
    plt.close(fig); print("wrote fig_repeating_class_amplitude (pseudo-3D)")


if __name__ == "__main__":
    make_morphology(); make_amplitude()
