#!/usr/bin/env python3
"""
fig_class_map.py — Figure 2 of the manuscript: the repeating-surface class.

Each cell is the real wall, drawn as a pseudo-3-D oblique extrusion by the
shared renderer in `geom3d.py` (the cabinet projection of the submitted
version's class maps).  Element height is held FIXED across every cell and the
pitch is drawn to scale, so the pitch-to-height ratio is a true visual
proportion from cell to cell; the near top edge carries the local verdict.

Two design decisions the results force:

  * Columns are ordered by pitch WITHIN each row, and the caption says in words
    that pitch does not order the verdict.

    THERE IS NO PANEL (j), and it must not come back.  It plotted five
    hand-picked cases on the signed viscous share of the wall force and claimed
    that axis ordered the verdict.  It does not.  Put every case on ONE
    definition -- the whole-surface share, from
    `force_partition_conditioning_l0_20260825.json` -- and the ordering reads

        -4.57 % k-type rib        FAIL
        +1.77 % periodic hill     FAIL      <- the same hill, 0.4 points apart,
        +2.20 % periodic hill     HOLD      <- opposite verdicts
        +6.80 % sinusoid 0.20     HOLD
       +15.22 % cube staggered    FAIL      <- a FAIL between two HOLDs
       +22.23 % d-type rib        HOLD
       +28.27 % sinusoid 0.10     HOLD

    The old panel reached its clean split only by omitting both hills, the bump
    and the step, and by plotting the cube on a FLOOR-ONLY share (-0.20 %)
    while every other point was whole-surface (+15.22 %).  The same artifact
    already records `kappa_orders_verdict_monotonically = False` and the
    prediction, registered before the ordering was read, that "no geometry-only
    screen can predict it".  The rib-pair result the panel was built on is
    real and is stated in words in section 5.4, where two points belong.
  * THERE IS NO ROUNDED-RIB ROW, and it must not come back either.  It drew a
    wall we never scored and labelled it "NOT SIMULATED: the untested bridge
    between the two families".  That label was not true: a fillet ladder WAS
    run, in steady RANS, and ledger row M8 deleted it from the active evidence
    because its windowed-mean statistics were unsound.  "Not simulated" and
    "simulated, then withdrawn" are different statements and the figure was
    making the wrong one, in a panel the body never referred to.  The gap is
    real and is now stated in the caption in words, which is where a statement
    about absent evidence belongs.

  * The periodic hill appears ONCE.  The paper reports two entries for it that
    score -1.76 and +0.68, and says explicitly that the difference is one of
    matching surface (0.094H against 0.05H) and station count, "not of flow".
    Two identical cells with opposite badges would assert that the geometry
    ordered the outcome, which is the same error as ordering by pitch, so the
    second reading is stated in the caption instead.

Every verdict and score is READ from the artifact that owns it; see _scores().

Run:  cd codes/figures && python3 fig_class_map.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_regime_map import MINPT, TEXTWIDTH_IN, OUT, C_OK, C_BAD
import geom3d as g3

C_HOLD = "#2f7d32"
C_FAIL = "#c62828"
C_EXCL = "#8a8a8a"
BADGE = {"HOLD": C_HOLD, "FAIL": C_FAIL, "NOT SIMULATED": C_EXCL}

matplotlib.rcParams.update({
    "font.size": MINPT, "font.family": "serif", "text.usetex": False,
    "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 400,
})


# ── the badge values, READ from the artifacts that own them ───────────────
# These used to be typed into SMOOTH/SHARP as literals.  That is precisely the
# defect ledger row M17 was opened for -- "unrun verdict badges were a
# hard-coded literal table" -- and by 2026-08-27 the transcription had drifted
# twice: the Gaussian bump was printed +0.98 against the archive's 0.9747, and
# the steep sinusoid +0.81 against its finest grid's 0.798.  Nothing is typed
# here now; every score is computed, and the figure fails to build if an
# artifact goes missing.
#
# Convention, applied uniformly: the FINEST grid of each case, which is the
# convention section 5.4 states for the rib pair.
def _scores():
    import json
    import fig_blueprint_20260826 as _B

    with open(os.path.join(_B.RESULTS,
              "signed_wall_error_metrics_m2.summary.json")) as fh:
        em = {r["name"]: r["r2_descriptive"] for r in json.load(fh)["rows"]}
    with open(os.path.join(_B.RESULTS,
              "force_partition_conditioning_l0_20260825.json")) as fh:
        fp = json.load(fh)
    hill = [r for r in fp["scored"]
            if r["case"].startswith("hill_pehill_MGLET")][0]["r2_tau"]

    W = _B._wavy_rows()

    def wavy(amp):                     # finest grid, own-machine, eta_m/d=0.05
        c = max((r for r in W if r["amp"] == amp and not r["repeat"]),
                key=lambda r: r["cells"])
        return c["r2"][0]

    RIBS, CUBES = _B._sharp_rows()

    def rib(tag):                      # finest grid of that pitch
        rs = [r for r in RIBS if tag in r[0]]
        return max(rs, key=lambda r: float(r[0].split(",")[-1].split()[0]))[1]

    staggered = [c for c in CUBES if c[0].startswith("staggered")][0][4]

    return {"sin5": wavy(0.20), "hill": hill, "sin10": wavy(0.10),
            "bump": em["gaussian_bump_Re2M"], "cube": staggered,
            "rib3": rib("$d$-type"), "rib8": rib("$k$-type"),
            "bfs": em["bfs_Re13700"]}


S = _scores()


def _badge(kind):
    v = S[kind]
    return ("HOLD" if v > 0 else "FAIL"), ("%+.1f" if abs(v) >= 10 else "%+.2f") % v


# ── the cells: fixed element height, pitch drawn to scale ─────────────────
H = 0.095                     # element height, the same in every cell

SMOOTH = [
    ("sinusoid $2a/\\lambda{=}0.20$", "$\\ell_p/h=5$",  *_badge("sin5"),  "sin5"),
    ("periodic hill",                 "$\\ell_p/h=9$",  *_badge("hill"),  "hill"),
    ("sinusoid $2a/\\lambda{=}0.10$", "$\\ell_p/h=10$", *_badge("sin10"), "sin10"),
    ("Gaussian bump",                 "non-repeating",  *_badge("bump"),  "bump"),
]
SHARP = [
    ("cube array, staggered", "$\\ell_p/h=2$", *_badge("cube"), "cube"),
    ("square rib, $d$-type",  "$\\ell_p/k=3$", *_badge("rib3"), "rib3"),
    ("square rib, $k$-type",  "$\\ell_p/k=8$", *_badge("rib8"), "rib8"),
    ("backward-facing step",  "non-repeating",  *_badge("bfs"),  "bfs"),
]


def geometry(ax, kind, verdict, faint=False):
    col = None if verdict == "NOT SIMULATED" else (C_OK if verdict == "HOLD" else C_BAD)
    if kind == "sin5":
        g3.extrude(ax, *g3.sinusoid(5 * H, H), crest_colour=col)
    elif kind == "sin10":
        g3.extrude(ax, *g3.sinusoid(10 * H, H), crest_colour=col)
    elif kind == "hill":
        g3.extrude(ax, *g3.hills(9 * H, H), crest_colour=col)
    elif kind == "bump":
        g3.extrude(ax, *g3.gauss_bump(H * 1.5), crest_colour=col)
    elif kind == "rib3":
        g3.extrude(ax, *g3.square_ribs(3 * H, H), crest_colour=col)
    elif kind == "rib8":
        g3.extrude(ax, *g3.square_ribs(8 * H, H), crest_colour=col)
    elif kind == "bfs":
        g3.extrude(ax, *g3.step_down(H * 1.6), crest_colour=col)
    elif kind == "cube":
        # cube_tile, not cube_array: the latter recedes a full pitch per row, so
        # at pitch = 2h each row cleared the top of the one in front and the
        # array read as three stacked tiers -- the class of physically wrong
        # schematic Referee 3 named in figures 1-2.  The tile is 0.85 wide so
        # its total extent matches the extruder's 1 + DX, which is what keeps
        # this cell on the same scale as its neighbours.
        g3.cube_tile(ax, crest_colour=col, pitch=2 * H, h=H, side=0.85,
                     stagger=True)
    elif kind.startswith("roundrib"):
        pitch = {"roundrib2": 2, "roundrib4": 4, "roundrib7": 7}.get(kind, 0)
        if pitch:
            g3.extrude(ax, *g3.round_ribs(pitch * H, H), crest_colour=col,
                       faint=faint)
        else:                       # the isolated limit
            g3.extrude(ax, *g3.round_ribs(1.0, H, duty=0.30), crest_colour=col,
                       faint=faint)
    g3.frame(ax, ytop=0.585)


def main():
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.52))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.02, 0.38, 1.02, 0.38],
                          hspace=0.36, wspace=0.09, left=0.072, right=0.996,
                          top=0.984, bottom=0.012)

    def put(rowg, rowt, cells, letters, ylab):
        for c, (name, pitch, verdict, score, kind) in enumerate(cells):
            g = fig.add_subplot(gs[rowg, c])
            geometry(g, kind, verdict)
            g.text(0.005, 1.00, f"({letters[c]})", transform=g.transAxes,
                   fontsize=MINPT, va="top", ha="left", color="0.35")
            g.text(1.00, 1.00, verdict, transform=g.transAxes, fontsize=MINPT,
                   ha="right", va="top", color="w", weight="bold",
                   bbox=dict(boxstyle="round,pad=0.16", fc=BADGE[verdict],
                             ec="none"))
            g.text(0.50, 0.995, "$R^2 = $" + score, transform=g.transAxes,
                   fontsize=MINPT, ha="center", va="top",
                   color=BADGE[verdict])
            if c == 0:
                g.text(-0.045, 0.42, ylab, transform=g.transAxes, fontsize=MINPT,
                       rotation=90, va="center", ha="right", color="0.3")
            t = fig.add_subplot(gs[rowt, c]); t.axis("off")
            t.set_xlim(0, 1); t.set_ylim(0, 1)
            t.text(0.5, 0.92, name, fontsize=MINPT, ha="center", va="top",
                   color="0.15")
            t.text(0.5, 0.22, pitch, fontsize=MINPT, ha="center", va="center",
                   color="0.35")

    put(0, 1, SMOOTH, "abcd", "smooth")
    put(2, 3, SHARP, "efgh", "sharp / 3-D")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "fig_class_map." + ext))
    plt.close(fig)
    print("  wrote fig_class_map.pdf  (%.2f x 2.52 in)" % TEXTWIDTH_IN)


if __name__ == "__main__":
    main()
