#!/usr/bin/env python3
"""
fig_regime_map.py — Figure 1 of the manuscript.

The three near-wall situations the paper distinguishes, drawn in the house style
of the submitted version's opening figure (grey solid with hatch ticks, wall
surface coloured by local verdict, dashed boundary-layer edge, blue near-wall
velocity glyph, force legs beneath each panel).

The ORGANISING AXIS is new.  The submitted figure ordered its panels by PITCH --
localised separation, wide pitch, tight pitch -- and the current results falsify
that ordering in both directions: the sinusoidal wall at a pitch of order delta
is reproduced, and the WIDE-pitch k-type rib is the catastrophic failure.  What
does order the outcome is whether the one-dimensional reduction is a closed
balance on the wall in question, and the sign of the plan-integrated viscous
wall force.  Panels (a) and (b) are therefore the two walls of the SAME periodic
hill, and (c) is the wide-pitch rib.

Every number drawn is transcribed from the manuscript:
  (a) flat inter-hill floor : the residual survives; the reduction closes.
  (b) sloped hill wall      : the retained and dropped terms nearly cancel;
      the residual is what is left of them, and the reduction does not close.
  (c) wide-pitch k-type rib : the residual is OPPOSITE in sign to the drive.

  This is the paper's opening figure and it is conceptual: it carries no
  measured value.  It used to print six -- a traction percentage, two forces to
  four decimals, two bound ratios and a form-drag fraction -- which is not what
  a reader needs before the results exist.  The one fact that is measured
  rather than schematic is the SIGN of (c)'s residual; everything else is
  proportion, and the caption says so.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geom3d as g3

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def rib_forces():
    """Panel (c)'s numbers, read from the rib campaign rather than typed.

    `r2_4_m20_les_20260823.json` carries the plan-integrated floor viscous force,
    the rib-face pressure force, the form-drag fraction, the reattachment length
    and the pitch for the wide-pitch k-type rib.
    """
    import json
    with open(os.path.join(RESULTS, "r2_4_m20_les_20260823.json")) as fh:
        c = json.load(fh)["cases"]["r24_rib_ktype_p8_G1"]
    f = c["drag"]["forces"]["forcesBottom"]
    out = dict(viscous=round(f["viscous_x"], 4),
               pressure=round(f["pressure_x"], 4),
               form=round(c["drag"]["form_drag_fraction"], 3),
               x_r=round(c["windows"]["cum_140"]["validation"]["x_reattach_over_k"], 2),
               p_over_k=int(round(c["p_over_k"])))
    assert out["viscous"] == -0.0268 and out["pressure"] == 0.6136, out
    assert out["form"] == 1.046 and out["x_r"] == 4.12 and out["p_over_k"] == 8, out
    return out

TEXTWIDTH_IN = 468.0 / 72.27
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "manuscript", "figures")

C_EDGE  = "#33373b"      # wall edge
C_FILL  = "#d9dde0"      # solid below the wall
C_BLE   = "#5a6068"      # boundary-layer edge
C_U     = "#2c7fb8"      # velocity glyph
C_OK    = "#2e7d32"      # wall on which the reduction closes
C_BAD   = "#c62828"      # wall on which it does not
C_P     = "#1f6fb2"      # pressure-gradient leg
C_CONV  = "#d98c00"      # dropped convection leg
C_GREY  = "0.45"

# Minimum type size in every figure of this paper (user rule, 2026-08-26): no
# text smaller than the panel-title size.  Nothing here may go below MINPT.
MINPT = 7.2

matplotlib.rcParams.update({
    "font.size": MINPT, "font.family": "serif", "text.usetex": False,
    "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 400,
})


# ── geometry ───────────────────────────────────────────────────────────────
H = 0.085                      # element height, cell units, the same in (a)-(c)


def hill_xy(pitch=9 * 0.085, h=0.085):
    return g3.hills(pitch, h)


def wall3d(ax, xs, ys, mask, colour, other="0.45"):
    """Extrude the wall, then over-draw the coloured segment of the near edge."""
    g3.extrude(ax, xs, ys, crest_colour=None, lw=0.55)
    ax.plot(xs, np.where(mask, ys, np.nan), "-", color=colour, lw=1.8, zorder=7,
            solid_capstyle="round")
    ax.plot(xs, np.where(~mask, ys, np.nan), "-", color=other, lw=0.8, zorder=7)


def normal_arrow(ax, x, y, nx, ny, L, colour, label):
    ax.annotate("", xy=(x + nx * L, y + ny * L), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1), zorder=9)
    ax.plot([x, x], [y, y + L], ":", color="0.45", lw=0.7, zorder=8)
    ax.text(x + nx * L, y + ny * L + 0.012, label, fontsize=MINPT, color=colour,
            ha="center", va="bottom", zorder=9)


def vglyph(ax, x0, ybase, height, width, kind="log"):
    eta = np.linspace(0, 1, 40)
    u = eta ** (1.0 / 7.0) if kind == "log" else -0.16 + 0.24 * eta + 0.92 * eta ** 3
    ax.plot([x0, x0], [ybase, ybase + height], ":", color="0.6", lw=0.5, zorder=8)
    ax.plot(x0 + u * width, ybase + eta * height, "-", color=C_U, lw=1.0, zorder=9)
    for tt in (0.35, 0.8):
        uu = np.interp(tt, eta, u)
        ax.annotate("", xy=(x0 + uu * width, ybase + tt * height),
                    xytext=(x0, ybase + tt * height),
                    arrowprops=dict(arrowstyle="->", color=C_U, lw=0.55), zorder=9)


# ── the figure ─────────────────────────────────────────────────────────────
def legs(ax, retained, dropped, colour):
    """The near-wall balance, drawn about a common origin so it ADDS UP.

    All three panels use this one diagram, so the row differs in the single
    thing the figure is about -- the residual.  Panel (c) used to carry a
    different construction entirely (a signed force axis with two printed
    forces), which made the reader learn a second diagram for the third case.

    The residual arrow is drawn as `retained + dropped`, so what the reader
    measures off the page is what the balance says.  Lengths are schematic;
    only the SIGN of the residual is a measured fact.
    """
    x0, y0 = 0.50, 0.84
    residual = retained + dropped
    ax.plot([x0, x0], [y0 - 0.52, y0 + 0.10], "-", color="0.55", lw=0.7,
            zorder=4)
    for y, dx, col in ((y0, retained, C_P),
                       (y0 - 0.21, dropped, C_CONV),
                       (y0 - 0.42, residual, colour)):
        ax.annotate("", xy=(x0 + dx, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7),
                    zorder=5)


def verdict(ax, colour, text):
    ax.text(0.5, 0.10, text, fontsize=MINPT, color=colour, ha="center",
            va="center")


def main():
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.30))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], hspace=0.02,
                          wspace=0.045, left=0.004, right=0.996,
                          top=0.930, bottom=0.115)
    G = [fig.add_subplot(gs[0, i]) for i in range(3)]
    T = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in T:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- (a), (b): the two walls of one periodic hill ----------
    xs, ys = hill_xy()
    slope = np.gradient(ys, xs)
    on_slope = np.abs(slope) > 1e-3

    for ax, hi in ((G[0], ~on_slope), (G[1], on_slope)):
        wall3d(ax, xs, ys, hi, C_BAD if hi is on_slope else C_OK)
        xb = np.linspace(0.03, 0.97, 300)
        ax.plot(xb + g3.DX * 0.5, np.interp(xb, xs, ys) * 0.30 + 0.245
                + g3.DY * 0.5, "--", color=C_BLE, lw=0.7, zorder=8)
        g3.frame(ax, ytop=0.435)
        ax.annotate("", xy=(0.16, 0.375), xytext=(0.03, 0.375),
                    arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.0))
        ax.text(0.095, 0.385, "$U$", fontsize=MINPT, ha="center", va="bottom")

    a = G[0]
    xa = 0.78
    normal_arrow(a, xa, float(np.interp(xa, xs, ys)), 0.0, 1.0, 0.115, C_OK, "$n$")
    vglyph(a, 0.60, float(np.interp(0.60, xs, ys)), 0.19, 0.15, "log")
    a.annotate("", xy=(0.055, 0.0), xytext=(0.055, 0.245),
               arrowprops=dict(arrowstyle="<->", color=C_BLE, lw=0.7))
    a.text(0.075, 0.12, r"$\delta$", fontsize=MINPT, color=C_BLE, va="center")

    b = G[1]
    xb0 = 0.462
    yb = float(np.interp(xb0, xs, ys)); sb = float(np.interp(xb0, xs, slope))
    nrm = np.hypot(sb, 1.0)
    normal_arrow(b, xb0, yb, -sb / nrm, 1.0 / nrm, 0.115, C_BAD, "$n$")
    vglyph(b, 0.86, float(np.interp(0.86, xs, ys)), 0.19, 0.15, "flat")

    # ---------------- (c) the wide-pitch k-type rib -------------------------
    c = G[2]
    RF = rib_forces()
    HK = 0.058
    xr, yr = g3.square_ribs(8 * HK, HK)
    wall3d(c, xr, yr, np.ones_like(xr, dtype=bool), C_BAD)
    c.plot(np.linspace(0.03, 0.97, 200) + g3.DX * 0.5,
           0.0 * np.linspace(0, 1, 200) + 0.265 + g3.DY * 0.5, "--",
           color=C_BLE, lw=0.7, zorder=8)
    g3.frame(c, ytop=0.435)
    c.annotate("", xy=(0.16, 0.375), xytext=(0.03, 0.375),
               arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.0))
    c.text(0.095, 0.385, "$U$", fontsize=MINPT, ha="center", va="bottom")
    t = np.linspace(0, 1, 160)
    x0 = 8 * HK * 0.5 + 8 * HK * 0.42 / 2
    c.plot(x0 + t * RF["x_r"] * HK, np.sin(np.pi * t) ** 0.85 * HK * 0.9, "-",
           color=C_BAD, lw=0.8, alpha=0.9, zorder=8)
    c.annotate("", xy=(x0 + 0.10 * HK, 0.016), xytext=(x0 + 2.2 * HK, 0.016),
               arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=0.8), zorder=8)
    # p and k are the only two symbols this panel uses, so both are drawn ON
    # the front cross-section, against the profile the reader can actually
    # read.  They used to float over the receding top face -- p's label landed
    # on a dark side face and k's two-headed arrow collapsed to a blob across
    # its 0.09 in span, which is what made "what is p?" a fair question.
    p, gold = 8 * HK, "#8a6d1f"
    xa, xb = p * 0.5, p * 1.5              # crest centre to crest centre
    yp = HK + 0.022
    c.annotate("", xy=(xb, yp), xytext=(xa, yp),
               arrowprops=dict(arrowstyle="<|-|>", color=gold, lw=0.8,
                               mutation_scale=7), zorder=9)
    for xg in (xa, xb):                    # tie each end to the crest it marks
        c.plot([xg, xg], [HK, yp], "-", color=gold, lw=0.6, zorder=9)
    c.text(0.5 * (xa + xb) + 0.035, yp + 0.008,
           "$p = %dk$" % RF["p_over_k"], fontsize=MINPT, color=gold,
           ha="center", va="bottom", zorder=10,
           bbox=dict(boxstyle="square,pad=0.10", fc="w", ec="none", alpha=0.85))

    xk = xb + p * 0.42 / 2                 # the downstream face of that rib
    c.plot([xk, xk], [0.0, HK], "-", color=C_EDGE, lw=0.7, zorder=9)
    for yv in (0.0, HK):                   # serifs, so the extent is unmistakable
        c.plot([xk - 0.008, xk + 0.008], [yv, yv], "-", color=C_EDGE, lw=0.7,
               zorder=9)
    c.text(xk + 0.013, HK / 2, "$k$", fontsize=MINPT, va="center", ha="left",
           zorder=10)

    # ---------------- the balance beneath each panel ------------------------
    # One diagram, three times.  The panels differ in the RESIDUAL and in
    # nothing else, which is the whole point of the figure: what the wall model
    # must return is the leftover of two large opposing terms, and it is
    # healthy, then tiny, then of the wrong sign.
    legs(T[0], +0.40, -0.270, C_OK)     # residual +0.130, plainly there
    legs(T[1], +0.40, -0.365, C_BAD)    # residual +0.035, a stub
    legs(T[2], +0.40, -0.475, C_BAD)    # residual -0.075, the other way

    verdict(T[0], C_OK,  "residual survives")
    verdict(T[1], C_BAD, "residual near-cancels")
    verdict(T[2], C_BAD, "residual reverses")

    # the three arrows are named ONCE, for the whole row
    key = fig.add_axes([0.004, 0.008, 0.992, 0.075]); key.axis("off")
    key.set_xlim(0, 1); key.set_ylim(0, 1)
    for x, col, lab in ((0.175, C_P, "retained pressure gradient"),
                        (0.470, C_CONV, "dropped convection"),
                        (0.700, "0.35", r"$\tau_w$, the residual")):
        key.annotate("", xy=(x + 0.028, 0.5), xytext=(x, 0.5),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7))
        key.text(x + 0.038, 0.5, lab, fontsize=MINPT, color="0.3", va="center",
                 ha="left")

    for ax, t_ in zip(G, ("(a) flat inter-hill floor", "(b) sloped hill wall",
                          "(c) wide-pitch $k$-type rib")):
        ax.set_title(t_, fontsize=MINPT + 0.6, loc="center", pad=1.5)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "fig_regime_map." + ext))
    plt.close(fig)
    print("  wrote fig_regime_map.pdf  (%.2f x 2.30 in)" % TEXTWIDTH_IN)


if __name__ == "__main__":
    main()
