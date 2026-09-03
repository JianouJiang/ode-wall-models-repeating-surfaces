#!/usr/bin/env python3
"""
geom3d.py — pseudo-3-D wall geometry, shared by every figure in the paper.

The renderer is the cabinet-projection extruder written for the submitted
version's class maps (`stage_classmaps_3d.py`): the wall profile is swept in the
spanwise direction along a fixed oblique offset, every swept quad is tinted by
Lambert shading of its own outward normal so smooth slopes get a continuous
gradient while risers separate, and the near cross-section face is left as a
clean solid so the front reads as a cut.  Only the rendering is reused — no
verdict, label or asset from those figures survives, because their badges are
falsified by the current results.

Rescaled here for the present paper: 7.2 pt type minimum, thin rules, and cells
about 1.4 in wide instead of 3 in.

Element height is held FIXED across cells and the pitch is drawn to scale, so
$\\ell_p/h$ is a true visual proportion from cell to cell.  Drawing each cell at
its own true aspect instead would make the periodic hill one part in eighteen of
the cell width and it would read as a flat line.
"""
import numpy as np

# cabinet projection: the spanwise sweep direction, in cell units
DX, DY = 0.16, 0.15
BASE = -0.13                      # the solid continues below the valley floor

LIGHT = np.array([-0.42, 0.907]); LIGHT = LIGHT / np.hypot(*LIGHT)


def _gray(i):
    i = float(np.clip(i, 0.0, 1.0))
    return (min(i * 0.97, 1.0), min(i * 0.99, 1.0), min(i * 1.06, 1.0))


def skin_color(dx, dy):
    nx, ny = -dy, dx
    nn = (nx * nx + ny * ny) ** 0.5
    if nn < 1e-12:
        return _gray(0.72)
    lam = max(0.0, (nx * LIGHT[0] + ny * LIGHT[1]) / nn)
    return _gray(0.58 + 0.28 * lam)


FRONT = _gray(0.72)
DEPTH = _gray(0.55)
GROUND = "#ecedf1"
EDGE = "#2b2f38"


# ── wall profiles on a cell of unit width, pitch and height in cell units ──
def sinusoid(pitch, h, x0=0.0):
    x = np.linspace(0, 1, 900)
    return x, 0.5 * h * (1.0 - np.cos(2 * np.pi * (x - x0) / pitch))


def hills(pitch, h, halfwidth=0.215):
    """Periodic hill: a smooth crest on a flat inter-hill floor.  The default
    half-width is 0.215 of the pitch, so the hill occupies about 43% of the
    period as the real geometry does (hill length ~3.9h of a 9h pitch) and the
    flat floor between hills is a real feature, not a sliver."""
    x = np.linspace(0, 1, 1100); y = np.zeros_like(x)
    k = 0
    while (k + 0.5) * pitch < 1.0 + pitch:
        c = (k + 0.5) * pitch; hw = pitch * halfwidth
        d = (x - c) / hw
        y += h * np.where(np.abs(d) <= 1,
                          0.5 * (1 + np.cos(np.pi * np.clip(d, -1, 1))), 0.0)
        k += 1
    return x, y


def square_ribs(pitch, h, duty=0.42):
    x = np.linspace(0, 1, 3000); y = np.zeros_like(x)
    k = 0
    while (k + 0.5) * pitch < 1.0 + pitch:
        c = (k + 0.5) * pitch; w = pitch * duty
        y[np.abs(x - c) <= w / 2] = h
        k += 1
    return x, y


def round_ribs(pitch, h, duty=0.42):
    x = np.linspace(0, 1, 3000); y = np.zeros_like(x)
    k = 0
    while (k + 0.5) * pitch < 1.0 + pitch:
        c = (k + 0.5) * pitch; w = pitch * duty; r = w * 0.45
        d = np.abs(x - c)
        y[d <= (w / 2 - r)] = h
        sl = (d > (w / 2 - r)) & (d <= w / 2)
        y[sl] = h * 0.5 * (1 + np.cos(np.pi * (d[sl] - (w / 2 - r)) / r))
        k += 1
    return x, y


def gauss_bump(h, c=0.5, s=0.13):
    x = np.linspace(0, 1, 900)
    return x, h * np.exp(-((x - c) / s) ** 2)


def step_down(h, c=0.34):
    x = np.linspace(0, 1, 900)
    return x, np.where(x < c, h, 0.0)


# ── the extruder ───────────────────────────────────────────────────────────
def extrude(ax, x, y, crest_colour=None, faint=False, base=BASE, lw=0.6):
    """Draw the profile (x, y) as an oblique extruded solid.

    `crest_colour` paints the near top edge — the wall the model is scored on.
    """
    a = 0.42 if faint else 1.0
    ax.fill([0, 1, 1 + DX, DX], [base, base, base + DY, base + DY],
            color=GROUND, ec="none", zorder=0, alpha=a)

    NS = 260
    xs = np.linspace(0, 1, NS); ys = np.interp(xs, x, y)

    ax.fill(np.r_[DX, xs + DX, 1 + DX], np.r_[base + DY, ys + DY, base + DY],
            color=DEPTH, ec="none", zorder=1, alpha=a)
    ax.fill([0, DX, DX, 0], [base, base + DY, ys[0] + DY, ys[0]],
            color=DEPTH, ec="none", zorder=1, alpha=a)

    for i in range(NS - 1):
        c = skin_color(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        ax.fill([xs[i], xs[i + 1], xs[i + 1] + DX, xs[i] + DX],
                [ys[i], ys[i + 1], ys[i + 1] + DY, ys[i] + DY],
                color=c, ec=c, lw=0.4, zorder=2, alpha=a)

    rc = skin_color(0.0, -1.0)
    ax.fill([1, 1, 1 + DX, 1 + DX], [base, ys[-1], ys[-1] + DY, base + DY],
            color=rc, ec=rc, lw=0.4, zorder=2, alpha=a)

    ax.fill(np.r_[0.0, x, 1.0], np.r_[base, y, base], color=FRONT, ec="none",
            zorder=5, alpha=a)
    ax.plot(xs + DX, ys + DY, color=EDGE, lw=0.35, alpha=0.45 * a, zorder=4)
    ax.plot(x, y, color=(crest_colour or EDGE), lw=(1.5 if crest_colour else lw),
            zorder=6, alpha=a, solid_capstyle="round")
    for xa, xb, ya, yb in ((0, 1, base, base), (0, 0, base, y[0]),
                           (1, 1, base, y[-1])):
        ax.plot([xa, xb], [ya, yb], color=EDGE, lw=lw, zorder=6, alpha=a)


# oblique unit direction for depth: a length L recedes by (L*UX, L*UY).  A cube
# must recede by its OWN side, or it is drawn as a bar.
UX, UY = 0.58, 0.54


def cube_array(ax, crest_colour=None, pitch=0.30, h=0.115, rows=3,
               faint=False, stagger=True):
    """Cube array in the cabinet projection.

    The cubes are cubes: the plan width, the height and the receding depth are
    all `h`.  The pitch follows the plan density, pitch = h / sqrt(lambda_p), so
    a sparse array shows fewer cubes across the same cell -- which is what
    sparse means.  Rows are separated by the same pitch in the spanwise
    direction, so the array reads as a field of separated blocks rather than a
    slab.  The scored surface is the floor between the cubes, so it is the floor
    that carries the colour.
    """
    a = 0.42 if faint else 1.0
    w = h
    dxc, dyc = w * UX, w * UY                     # a cube's own depth
    spanx, spany = pitch * UX, pitch * UY         # row-to-row separation
    Dx, Dy = spanx * (rows - 1) + dxc, spany * (rows - 1) + dyc

    # In the extruded 2-D cells the scored wall is seen edge-on, so the verdict
    # is a coloured line.  Here the scored surface is the FLOOR BETWEEN THE
    # CUBES, which this projection shows as an area -- so it is tinted as an
    # area and outlined, and the cubes drawn over it occlude their own
    # footprints.  A single line along the front would say the wrong thing.
    ax.fill([0, 1, 1 + Dx, Dx], [0, 0, Dy, Dy],
            color=GROUND, ec="none", zorder=0, alpha=a)
    if crest_colour:
        ax.fill([0, 1, 1 + Dx, Dx], [0, 0, Dy, Dy], color=crest_colour,
                ec=crest_colour, lw=1.0, zorder=1, alpha=0.16 * a)
    else:
        ax.plot([0, 1, 1 + Dx, Dx, 0], [0, 0, Dy, Dy, 0], color=EDGE, lw=0.4,
                zorder=1, alpha=a)

    top, front, side = _gray(0.88), _gray(0.70), _gray(0.52)
    for r in range(rows - 1, -1, -1):             # back rows first
        ox, oy = spanx * r, spany * r
        shift = 0.5 * pitch if (stagger and r % 2) else 0.0
        k = 0
        while shift + k * pitch + w < 1.001:
            X, Y = shift + k * pitch + ox, oy
            z = 3 + (rows - r)
            ax.fill([X, X + w, X + w, X], [Y, Y, Y + h, Y + h],
                    color=front, ec=EDGE, lw=0.35, zorder=z, alpha=a)
            ax.fill([X, X + w, X + w + dxc, X + dxc],
                    [Y + h, Y + h, Y + h + dyc, Y + h + dyc],
                    color=top, ec=EDGE, lw=0.35, zorder=z, alpha=a)
            ax.fill([X + w, X + w + dxc, X + w + dxc, X + w],
                    [Y, Y + dyc, Y + h + dyc, Y + h],
                    color=side, ec=EDGE, lw=0.35, zorder=z, alpha=a)
            k += 1


def frame(ax, ytop=0.60, xpad=0.06, equal=False, xmax=None):
    """Frame a geometry panel.

    `equal=True` forces a true 1:1 data aspect.  Without it a cube drawn as a
    square in data coordinates comes out as a slab whenever the axes box is not
    proportioned like the data range -- which is how the cube arrays ended up
    looking like dominoes.
    """
    ax.set_xlim(-xpad, (1 + DX if xmax is None else xmax) + xpad)
    ax.set_ylim(BASE - 0.03, ytop)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")


# ── plan-tile cube arrays ──────────────────────────────────────────────────
# `cube_array` above places its rows by receding a full pitch per row at the
# projection depth (UX, UY).  When the pitch is twice the cube height -- which
# is what lambda_p = 0.25 means -- each row then clears the top of the row in
# front of it and the array reads as stacked tiers rather than a field of
# blocks on a floor.  The renderer below fixes that by (i) halving the
# receding length, which is the cabinet convention, and (ii) drawing a FIXED
# unit floor tile and placing cubes on it at the true pitch in BOTH plan
# directions, so the only thing that changes between panels is the density.
PX, PY = 0.365, 0.335


def cube_tile(ax, crest_colour=None, pitch=0.32, h=0.16, stagger=False,
              side=1.0):
    """A unit-square floor tile in cabinet projection carrying cubes of side
    `h` at spacing `pitch` in both plan directions.

    The cubes are cubes: plan width, height and receding depth are all `h`.
    The tile is identical in every panel, so a sparse array shows fewer cubes
    on the same floor -- which is what sparse means, and what a per-panel
    frame would hide.  The scored surface is the floor between the cubes, so
    the floor is what carries the verdict colour.

    `side` is the plan width of the floor patch.  A caller that has to share a
    fixed scale with neighbouring cells (figure 2) can shrink the patch without
    shrinking the cubes, which is what keeps the element height comparable
    across that figure.
    """
    dxc, dyc = h * PX, h * PY                      # a cube's own depth

    sx, sy = side * PX, side * PY
    tile = ([0, side, side + sx, sx], [0, 0, sy, sy])
    ax.fill(*tile, color=GROUND, ec="none", zorder=0)
    if crest_colour:
        ax.fill(*tile, color=crest_colour, ec=crest_colour, lw=0.8, zorder=1,
                alpha=0.16)
    ax.plot(np.r_[tile[0], tile[0][0]], np.r_[tile[1], tile[1][0]],
            color=EDGE, lw=0.4, zorder=1, alpha=0.55)

    c_top, c_front, c_side = _gray(0.88), _gray(0.70), _gray(0.52)
    n = max(1, int(np.floor(side / pitch)))        # cubes per side on the tile
    off = 0.5 * (side - n * pitch) + 0.5 * pitch   # centre the lattice

    for r in range(n - 1, -1, -1):                 # back rows first
        d = off + r * pitch
        ox, oy = d * PX, d * PY
        shift = 0.5 * pitch if (stagger and r % 2) else 0.0
        for c in range(n):
            X = off + c * pitch + shift - 0.5 * h
            if X < -1e-9 or X + h > side + 1e-9:   # whole cubes on the tile only
                continue
            X += ox
            Y = oy
            z = 3 + (n - r)
            ax.fill([X, X + h, X + h, X], [Y, Y, Y + h, Y + h],
                    color=c_front, ec=EDGE, lw=0.35, zorder=z)
            ax.fill([X, X + h, X + h + dxc, X + dxc],
                    [Y + h, Y + h, Y + h + dyc, Y + h + dyc],
                    color=c_top, ec=EDGE, lw=0.35, zorder=z)
            ax.fill([X + h, X + h + dxc, X + h + dxc, X + h],
                    [Y, Y + dyc, Y + h + dyc, Y + h],
                    color=c_side, ec=EDGE, lw=0.35, zorder=z)


def cube_tile_frame(ax, h=0.16, headroom=0.22):
    """Frame a `cube_tile` panel.  `headroom` leaves room for the name label
    above the back row instead of setting it on top of the cubes."""
    ax.set_xlim(-0.05, 1 + PX + 0.05)
    ax.set_ylim(-0.06, PY + h + h * PY + headroom)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
