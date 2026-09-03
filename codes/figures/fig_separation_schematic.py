#!/usr/bin/env python3
"""
Figure 1: four-regime landscape comparison (no figure title -- framing lives in
the caption).

Four columns, left to right, ordered by how strongly the geometry forces the
near-wall force cancellation.  Each column keeps the geometry shape and its
title; the regime description (epsilon and the geometric scale) is annotated
INSIDE the schematic, and only the verdict badge plus the example cases sit
below it.  Separation is a realistic recirculation contour (separation
streamline + nested closed inner streamlines + rotation sense).

  1. Thin-BL / localised separation (BFS)      : eps ~ O(1)   -> ODE valid
  2. Weak constriction (shallow wavy wall)      : weak blockage -> ODE sufficient
  3. Wide-pitch / sparse element                : localised     -> ODE tolerated
  4. Repeating structure (periodic hill)        : eps << 1      -> ODE fails
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from plotting_utils import setup_style, save_figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

setup_style(font_size=11, use_tex=True)
import matplotlib as mpl
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_WALL_EDGE = '#33373b'
C_WALL_FILL = '#d9dde0'
C_STREAM    = '#2c7fb8'
C_RECIRC    = '#d62728'
C_OK        = '#2e7d32'
C_OK_FILL   = '#eef6ef'
C_BAD       = '#c62828'
C_BAD_FILL  = '#fdeceb'

# ---------------------------------------------------------------------------
# Canvas (landscape, no header)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13.8, 4.0))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 138)
ax.set_ylim(0, 40)
ax.set_aspect('equal')
ax.axis('off')


# ===========================================================================
# Helpers
# ===========================================================================
def rbox(cx, cy, w, h, fc, ec, lw=1.2, z=2, rounding=1.4, alpha=1.0):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle=f'round,pad=0,rounding_size={rounding}',
                         fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
                         mutation_aspect=1.0)
    ax.add_patch(box)
    return box


def check(cx, cy, s, color, lw=1.7):
    ax.plot([cx - 0.9 * s, cx - 0.2 * s, cx + s],
            [cy + 0.0 * s, cy - 0.7 * s, cy + 0.8 * s],
            '-', color=color, lw=lw, solid_capstyle='round', zorder=7, clip_on=False)


def cross(cx, cy, s, color, lw=1.7):
    ax.plot([cx - s, cx + s], [cy - s, cy + s], '-', color=color, lw=lw,
            solid_capstyle='round', zorder=7, clip_on=False)
    ax.plot([cx - s, cx + s], [cy + s, cy - s], '-', color=color, lw=lw,
            solid_capstyle='round', zorder=7, clip_on=False)


def streamlines(rect, levels, rad=0.03, color=C_STREAM, lw=0.9):
    x0, x1, y0, y1 = rect
    for t in levels:
        y = y0 + t * (y1 - y0)
        ax.annotate('', xy=(x0 + 0.95 * (x1 - x0), y - 0.02 * (y1 - y0)),
                    xytext=(x0 + 0.05 * (x1 - x0), y),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'), zorder=3)


def vprofile(x0, ybase, h, w, kind, color, label=None):
    """Near-wall velocity profile glyph (rooted on the wall at (x0, ybase))."""
    eta = np.linspace(0, 1, 40)
    if kind == 'healthy':
        u = eta ** (1.0 / 7.0)
    elif kind == 'mild':
        u = eta ** (1.0 / 3.0)
    elif kind == 'flat':
        u = -0.18 + 0.26 * eta + 0.92 * eta ** 3
    ax.plot([x0, x0], [ybase, ybase + h], ':', color='0.55', lw=0.6, zorder=4)
    ax.plot(x0 + u * w, ybase + eta * h, '-', color=color, lw=1.4, zorder=5)
    for tt in (0.4, 0.85):
        uu = np.interp(tt, eta, u)
        ax.annotate('', xy=(x0 + uu * w, ybase + tt * h), xytext=(x0, ybase + tt * h),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.7), zorder=5)
    if label:
        ax.text(x0 + 0.4, ybase + h + 0.6, label, fontsize=6.6, color=color,
                ha='center', va='bottom', style='italic', zorder=5)


def recirc(xs, xr, yfloor, H, color=C_RECIRC, sense=1, nloops=2, fill=True):
    """Realistic recirculation: separation streamline + nested closed inner
    streamlines + rotation sense (sense=+1 => clockwise)."""
    t = np.linspace(0, 1, 140)
    bx = xs + t * (xr - xs)
    by = yfloor + H * np.sin(np.pi * t)
    if fill:
        ax.fill_between(bx, yfloor, by, color=color, alpha=0.13, zorder=1)
    ax.plot(bx, by, color=color, lw=1.0, alpha=0.75, zorder=2)
    cx = 0.5 * (xs + xr); cyc = yfloor + 0.47 * H
    Rx = 0.33 * (xr - xs); Ry = 0.40 * H
    th = np.linspace(0, 2 * np.pi, 120)
    for f in np.linspace(1.0, 0.42, nloops):
        ax.plot(cx + f * Rx * np.cos(th), cyc + f * Ry * np.sin(th),
                color=color, lw=0.8, alpha=0.85, zorder=2)
    ax.annotate('', xy=(cx + sense * 0.45 * Rx, cyc + Ry),
                xytext=(cx - sense * 0.45 * Rx, cyc + Ry),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0.9), zorder=3)
    ax.annotate('', xy=(cx - sense * 0.45 * Rx, cyc - Ry),
                xytext=(cx + sense * 0.45 * Rx, cyc - Ry),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0.9), zorder=3)


# ===========================================================================
# Geometry schematics
# ===========================================================================
def draw_flow(rect, kind):
    x0, x1, y0, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    X = lambda s: x0 + s * Lx
    Y = lambda t: y0 + t * Ly
    s = np.linspace(0, 1, 400)

    if kind == 'bfs':
        wt = np.where(s < 0.30, 0.55, 0.16)
        ax.fill_between(X(s), Y(0), Y(wt), color=C_WALL_FILL, zorder=0)
        ax.plot(X(s), Y(wt), '-', color=C_WALL_EDGE, lw=1.7, zorder=3)
        streamlines((X(0.02), X(1.0), Y(0.64), Y(0.82)), [0.0, 0.55, 1.0])
        recirc(X(0.31), X(0.68), Y(0.16), 0.18 * Ly, sense=+1, nloops=2)
        vprofile(X(0.90), Y(0.16), 0.32 * Ly, 0.075 * Lx, 'healthy', C_OK,
                 label=r'strong $\partial_y U$')

    elif kind == 'wavy':
        wt = 0.22 + 0.075 * np.sin(2 * np.pi * 1.6 * s - 0.5)
        ax.fill_between(X(s), Y(0), Y(wt), color=C_WALL_FILL, zorder=0)
        ax.plot(X(s), Y(wt), '-', color=C_WALL_EDGE, lw=1.7, zorder=3)
        xx = np.linspace(0.05, 0.95, 120)
        for t0 in (0.50, 0.70):
            yy = t0 + 0.07 * np.sin(2 * np.pi * 1.6 * xx - 0.5)
            ax.plot(X(xx), Y(yy), '-', color=C_STREAM, lw=0.95, zorder=3)
            ax.annotate('', xy=(X(xx[-1]), Y(yy[-1])), xytext=(X(xx[-4]), Y(yy[-4])),
                        arrowprops=dict(arrowstyle='-|>', color=C_STREAM, lw=0.95), zorder=3)
        # root the profile exactly on the wavy wall
        xp = 0.5
        wtp = 0.22 + 0.075 * np.sin(2 * np.pi * 1.6 * xp - 0.5)
        vprofile(X(xp), Y(wtp), 0.34 * Ly, 0.075 * Lx, 'mild', C_OK, label=r'attached')

    elif kind == 'wide':
        # one continuous wall outline (floor + two ribs), white interior
        base, ribtop = 0.16, 0.56
        wt = np.full_like(s, base)
        for c in (0.16, 0.84):
            wt[np.abs(s - c) <= 0.05] = ribtop
        ax.fill_between(X(s), Y(0), Y(wt), color=C_WALL_FILL, zorder=0)
        ax.plot(X(s), Y(wt), '-', color=C_WALL_EDGE, lw=1.7, zorder=3)
        streamlines((X(0.02), X(1.0), Y(0.66), Y(0.82)), [0.0, 0.5, 1.0])
        recirc(X(0.22), X(0.42), Y(base), 0.14 * Ly, sense=+1, nloops=2)
        vprofile(X(0.55), Y(base), 0.34 * Ly, 0.075 * Lx, 'healthy', C_OK, label=r'attached')

    elif kind == 'hill':
        # tight-pitch: several closely packed crests with a recirculation bubble
        # trapped in every valley -> cancellation is domain-wide
        def crest(c, w0):
            return np.where(np.abs(s - c) < w0,
                            0.50 * np.cos((s - c) / w0 * np.pi / 2) ** 2, 0.0)
        cc = [0.125, 0.375, 0.625, 0.875]
        wt = np.full_like(s, 0.08)
        for c in cc:
            wt = wt + crest(c, 0.125)
        wt = wt + crest(-0.125, 0.125) + crest(1.125, 0.125)
        ax.fill_between(X(s), Y(0), Y(wt), color=C_WALL_FILL, zorder=0)
        ax.plot(X(s), Y(wt), '-', color=C_WALL_EDGE, lw=1.9, zorder=3)
        streamlines((X(0.02), X(1.0), Y(0.82), Y(0.95)), [0.0, 0.5, 1.0], rad=0.012)
        for vc in (0.25, 0.50, 0.75):
            recirc(X(vc - 0.105), X(vc + 0.105), Y(0.08), 0.20 * Ly,
                   sense=+1, nloops=2)


# ===========================================================================
# Geometric-parameter annotations (delta, pitch) -- Fig 1 doubles as the
# geometry-definition reference for sections 2.2-2.3
# ===========================================================================
C_SCALE = '#5b6b7a'      # boundary-layer / delta annotations
C_PITCH = '#8a5a00'      # pitch / height annotations
DELTA_FIG = 7.6          # boundary-layer thickness, drawn at one fixed visual
                         # size across panels so pitch/delta ratios are comparable


def delta_marker(rect, ywall_frac, x_at=0.10, show_label=False):
    """Dashed boundary-layer edge + a vertical delta double-arrow."""
    x0, x1, y0, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    yw = y0 + ywall_frac * Ly
    yedge = yw + DELTA_FIG
    ax.plot([x0 + 0.03 * Lx, x1 - 0.03 * Lx], [yedge, yedge], '--',
            color=C_SCALE, lw=0.9, dashes=(5, 3), zorder=4)
    if show_label:
        ax.text(x1 - 0.04 * Lx, yedge + 0.45, r'BL edge', fontsize=6.2,
                color=C_SCALE, ha='right', va='bottom', zorder=4)
    xd = x0 + x_at * Lx
    ax.annotate('', xy=(xd, yedge), xytext=(xd, yw),
                arrowprops=dict(arrowstyle='<->', color=C_SCALE, lw=1.1), zorder=5)
    ax.text(xd - 1.0, 0.5 * (yw + yedge), r'$\delta$', fontsize=9.5,
            color=C_SCALE, ha='right', va='center', zorder=5)


def pitch_marker(rect, sa, sb, yfrac, label=r'$\ell_p$'):
    """Horizontal pitch double-arrow between two repeating elements."""
    x0, x1, y0, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    xa, xb = x0 + sa * Lx, x0 + sb * Lx
    yy = y0 + yfrac * Ly
    ax.annotate('', xy=(xb, yy), xytext=(xa, yy),
                arrowprops=dict(arrowstyle='<->', color=C_PITCH, lw=1.1), zorder=6)
    ax.text(0.5 * (xa + xb), yy + 0.5, label, fontsize=8.6, color=C_PITCH,
            ha='center', va='bottom', zorder=6)


def height_marker(rect, s_at, t_lo, t_hi, label=r'$k$'):
    """Vertical element-height double-arrow."""
    x0, x1, y0, y1 = rect
    Lx, Ly = x1 - x0, y1 - y0
    xv = x0 + s_at * Lx
    ax.annotate('', xy=(xv, y0 + t_hi * Ly), xytext=(xv, y0 + t_lo * Ly),
                arrowprops=dict(arrowstyle='<->', color=C_PITCH, lw=1.0), zorder=6)
    ax.text(xv + 0.7, y0 + 0.5 * (t_lo + t_hi) * Ly, label, fontsize=8.4,
            color=C_PITCH, ha='left', va='center', zorder=6)


# ===========================================================================
# Columns
# ===========================================================================
cols = [
    dict(kind='bfs', good=True,
         title=r'Thin-BL / localised sep.',
         verdict=r'ODE valid',
         regime=r'single feature' + '\n' + r'$L_\mathrm{sep}\!\gg\!\delta$',
         rpos=(0.45, 0.90)),
    dict(kind='wavy', good=True,
         title=r'Weak constriction',
         verdict=r'ODE sufficient',
         regime=r'shallow, attached' + '\n' + r'weak blockage',
         rpos=(0.50, 0.90)),
    dict(kind='wide', good=True,
         title=r'Wide-pitch / sparse',
         verdict=r'ODE tolerated',
         regime=r'sparse, localised' + '\n' + r'pitch $\ell_p\!\gg\!\delta$',
         rpos=(0.50, 0.92)),
    dict(kind='hill', good=False,
         title=r'Tight-pitch / packed',
         verdict=r'ODE fails',
         regime=r'repeats domain-wide' + '\n' + r'pitch $\ell_p\!\sim\!\delta$',
         rpos=(0.50, 0.70)),
]

CXS = [19, 53, 87, 121]
CW = 32
GY0, GY1 = 15.5, 33.5          # geometry sub-rect (per column)
for cx, col in zip(CXS, cols):
    good = col['good']
    cmain = C_OK if good else C_BAD
    cfill = C_OK_FILL if good else C_BAD_FILL
    # grouping card + top accent bar
    rbox(cx, 21.5, CW, 35.0, '#ffffff', cmain, lw=1.6, rounding=1.6, z=1)
    rbox(cx, 38.2, CW, 1.5, cmain, cmain, lw=0, rounding=0.4, z=2)
    # column title
    ax.text(cx, 36.0, col['title'], fontsize=10.2, ha='center', va='center',
            color=cmain, fontweight='bold')
    # geometry schematic
    gx0, gx1 = cx - 15.5, cx + 15.5
    rect = (gx0, gx1, GY0, GY1)
    draw_flow(rect, col['kind'])
    # geometric-parameter annotations: boundary-layer thickness on every panel,
    # pitch where repetition is the point, element height on the rib panel
    if col['kind'] == 'bfs':
        delta_marker(rect, 0.16, x_at=0.10)
    elif col['kind'] == 'wavy':
        delta_marker(rect, 0.30, x_at=0.08)
    elif col['kind'] == 'wide':
        delta_marker(rect, 0.16, x_at=0.06)
        pitch_marker(rect, 0.16, 0.84, 0.40)
        height_marker(rect, 0.16, 0.16, 0.56, label=r'$k$')
    elif col['kind'] == 'hill':
        delta_marker(rect, 0.08, x_at=0.04)
        pitch_marker(rect, 0.125, 0.375, 0.78)
    # regime description embedded in the schematic
    rfx, rfy = col['rpos']
    ax.text(gx0 + rfx * (gx1 - gx0), GY0 + rfy * (GY1 - GY0), col['regime'],
            fontsize=7.6, ha='center', va='center', color='0.18', zorder=6,
            linespacing=1.25)
    # divider
    ax.plot([cx - 15, cx + 15], [13.5, 13.5], '-', color='#d7dade', lw=0.8, zorder=2)
    # verdict badge (lowest element in every column)
    rbox(cx, 9.3, 27, 5.2, cfill, cmain, lw=1.0, rounding=1.0, z=3)
    if good:
        check(cx - 10.5, 9.3, 1.35, cmain)
    else:
        cross(cx - 10.5, 9.3, 1.25, cmain)
    ax.text(cx + 1.6, 9.3, col['verdict'], fontsize=9.4, ha='center', va='center',
            color=cmain, fontweight='bold')

save_figure(fig, 'fig_separation_schematic', fig_dir=FIG_DIR)
print(f'\nSaved to {FIG_DIR}/fig_separation_schematic.pdf')
