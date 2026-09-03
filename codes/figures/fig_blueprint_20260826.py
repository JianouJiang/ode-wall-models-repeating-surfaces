#!/usr/bin/env python3
"""
fig_blueprint_20260826.py — figure/table rebalance, BLUEPRINT pass.

Each figure here REPLACES a table and must therefore carry every column that
table carried: not only the trend, but the intervals, the secondary columns and
the qualifications that live in the table caption.  Nothing may be dropped to
make a chart tidy.

Everything is drawn at PRINTED size (textwidth = 468 pt = 6.4757 in), so layout
and page cost are honest.  Panels made from data already in the paper are drawn
for real; geometry illustrations are labelled boxes at the true size and
position the artwork will occupy, and are what makes the no-placeholder check
fail until phase 2 replaces them.

Every number is transcribed verbatim from the table it replaces (build of
2026-08-26).  Nothing is recomputed.

Run:  cd codes/figures && python3 fig_blueprint_20260826.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geom3d as g3

TEXTWIDTH_IN = 468.0 / 72.27          # 6.4757 in — measured from the class

# Minimum type size in every figure of this paper (user rule, 2026-08-26): no
# text may be set smaller than the panel-title size of figure 1.  Every explicit
# fontsize below is clamped to this, and the tick/legend defaults sit on it too.
MINPT = 7.2
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "manuscript", "figures")


def style(fs=8.4):
    plt.rcdefaults()
    matplotlib.rcParams.update({
        "font.size": fs, "font.family": "serif", "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": MINPT, "ytick.labelsize": MINPT,
        "legend.fontsize": MINPT, "figure.dpi": 150, "savefig.dpi": 400,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        # A minus inside mathtext is U+2212, which pdftotext drops: "-1.757"
        # extracts as "1.757".  Every number a reader or a text search must be
        # able to find is therefore drawn OUTSIDE mathtext, and the tick
        # formatter is put on the ASCII hyphen here.
        "axes.unicode_minus": False,
    })


def num(v, dp=3, signed=True):
    """A number as PLAIN text (never inside $...$), so it survives extraction."""
    return ("%+.*f" if signed else "%.*f") % (dp, v)

C_EQ   = "#2ca02c"      # equilibrium  (green, as in the existing figures)
C_TB   = "#4C6D8C"      # total-gradient TBLE (bluish grey)
C_REF  = "#ff7f0e"      # reference / DNS
C_A    = "#b0b0b0"      # withdrawn estimator A
C_B    = "#1f77b4"      # primary reference B
C_C    = "#d62728"      # bracket reference C
C_HOLD = "#2f7d32"
C_FAIL = "#c62828"
C_EXCL = "#8a8a8a"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        # no bbox='tight': the printed size must be exactly the drawn size
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf  ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)")


def placeholder(ax, text="", fc="#f4f1e8", fs=MINPT, wrap=True):
    """A labelled box standing where real artwork will go."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linestyle((0, (3, 2))); s.set_linewidth(0.8); s.set_color("#8a7a55")
    ax.set_facecolor(fc)
    if text:
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                fontsize=fs, color="#5a4a25", wrap=wrap, linespacing=1.4)


# ══════════════════════════════════════════════════════════════════════════
# F1  the three near-wall situations                          [BOXES: G2]
# ══════════════════════════════════════════════════════════════════════════
# F3  replaces the physical-error table                             [REAL]
#     carries R2, q50, q95, qmax, E_D, E_S and both location errors
# ══════════════════════════════════════════════════════════════════════════
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _error_operator_rows():
    """Build the physical-error table from the archive rather than by hand.

    Seventeen of the eighteen cases come straight out of the frozen operator
    output, `signed_wall_error_metrics_m2.summary.json`, and reproduce every
    printed digit.  The periodic hill does NOT: that file scores it against the
    WITHDRAWN four-point traction estimator (R2 = -47.686), the value the paper
    explicitly retracts.  Its corrected score is taken from the reference-rebase
    output instead, and its station-error columns are carried as literals,
    because no archived operator output holds them against the corrected
    reference.  See the report for that provenance gap.
    """
    import json
    with open(os.path.join(RESULTS,
              "signed_wall_error_metrics_m2.summary.json")) as fh:
        rows = {r["name"]: r for r in json.load(fh)["rows"]}
    with open(os.path.join(RESULTS,
              "reference_rebase_headlines_l0_20260825.json")) as fh:
        rebase = json.load(fh)["headlines"]["scores"]["pg_ode_mixing_length"]

    C = ("r2_descriptive", "station_abs_p50", "station_abs_p95",
         "station_abs_max", "viscous_drag_signed_error",
         "separated_set_symmetric_difference", "separation_error_over_span",
         "reattachment_error_over_span")

    def one(name, label):
        r = rows[name]
        v = [r[c] for c in C]
        return (label,) + tuple(None if x != x else round(x, 3) for x in v)

    def family(prefix, label):
        rs = [r for r in rows.values() if r["name"].startswith(prefix)]
        out = [label]
        for c in C:
            vals = [r[c] for r in rs if r[c] == r[c]]
            out.append(None if not vals else
                       (round(min(vals), 3), round(max(vals), 3)))
        return tuple(out)

    R = [
        family("apg_tbl", "APG b1n--m18n (5)"),
        family("jaxa_sep_bubble", "JAXA $Re_\\theta$ 300--900 (3)"),
        one("separation_bubble_caseC", "Separation bubble C"),
        one("swept_separation_caseC35", "Swept bubble C35"),
        one("gaussian_bump_Re2M", "Gaussian bump"),
        one("nasa_hump_Re936000", "NASA hump"),
        one("conv_div_channel_Re12600", "Conv.--div. channel"),
        one("curved_bfs_Re13700_DNS", "Curved backward-facing step"),
        one("bfs_Re13700", "Backward-facing step"),
        one("krank_pehill_Re10595", "Krank hill, $Re$ 10595"),
        one("krank_pehill_Re5600", "Krank hill, $Re$ 5600"),
        # corrected R2 from the rebase; the remaining columns have no archived
        # counterpart against the corrected reference and are carried as stated
        ("Periodic hill 1.0", round(rebase["B_mglet"]["r2"], 3),
         .309, 2.918, 18.786, -.221, .338, .188, .273),
    ]
    # the figure must still print what the paper printed
    assert R[-1][1] == -1.757, R[-1][1]
    assert R[8][1:5] == (.889, .148, .349, .483), R[8]
    return R


def f3_errors():
    style(8.4)
    R = _error_operator_rows()
    y = np.arange(len(R))[::-1]
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(TEXTWIDTH_IN, 2.72),
        gridspec_kw={"width_ratios": [1.75, 0.72, 0.72]}, sharey=True)

    def val(v):
        return sum(v) / 2 if isinstance(v, tuple) else v

    for i, row in enumerate(R):
        r2, q50, q95, qmax = row[1], row[2], row[3], row[4]
        col = C_FAIL if val(r2) < 0 else C_TB
        a1.plot([val(q50), val(qmax)], [y[i], y[i]], "-", color=col, lw=0.9,
                alpha=0.6)
        for v, mk, ms in ((q50, "o", 4.4), (q95, "|", 6.0), (qmax, "D", 3.0)):
            if isinstance(v, tuple):
                a1.plot(list(v), [y[i]] * 2, "-", color=col, lw=2.6, alpha=0.5,
                        solid_capstyle="butt")
            a1.plot(val(v), y[i], mk, color=col, ms=ms,
                    mfc="w" if mk == "D" else col, mew=0.9)
        if val(qmax) > 5.0:            # the one case off the common scale
            a1.annotate(num(val(qmax), signed=False), (val(qmax), y[i]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=MINPT, color=col)
        txt = ("$R^2 = $" + num(r2) if not isinstance(r2, tuple)
               else "$R^2 = $" + num(r2[0], signed=False) + "-"
                    + num(r2[1], signed=False))
        a1.text(1.015, y[i], txt, transform=a1.get_yaxis_transform(),
                fontsize=MINPT, va="center", ha="left", color=col)
    a1.axvline(1.0, color="0.5", ls="--", lw=0.7, zorder=0)
    a1.set_xscale("log"); a1.set_xlim(6e-3, 45)
    a1.set_yticks(y); a1.set_yticklabels([r[0] for r in R], fontsize=MINPT)
    a1.set_ylim(-1.75, len(R) - 0.3)
    a1.set_xlabel(r"station error $|e_i|$  in units of the reference RMS "
                  r"$\tau_\star$")
    a1.set_title("(a) station errors, case by case", fontsize=MINPT, loc="left")
    # the marker shapes are the statistic; colour is the verdict, so the key is
    # drawn in neutral grey and put in the empty upper right of the panel
    a1.legend(handles=[
        plt.Line2D([], [], ls="none", marker="o", ms=4.4, color="0.35",
                   label="median"),
        plt.Line2D([], [], ls="none", marker="|", ms=6.0, color="0.35",
                   label="95th percentile"),
        plt.Line2D([], [], ls="none", marker="D", ms=3.0, mfc="w", mew=0.9,
                   color="0.35", label="maximum")],
        loc="upper right", fontsize=MINPT, handletextpad=0.3, borderpad=0.3,
        labelspacing=0.28, handlelength=1.0)

    # A legend in either upper corner of these narrow panels sat on the data, so
    # it goes to the bottom of the panel, where the last row leaves a clear band.
    panels = [
        (a2, "(b) $E_D$, $E_S$", [(5, "s", "#7b4ea3", r"$E_D$"),
                                  (6, "^", "#c47f17", r"$E_S$")]),
        (a3, "(c) location", [(7, "v", "#3a8a8a", r"$\Delta x_s/L$"),
                              (8, "P", "#8c564b", r"$\Delta x_r/L$")]),
    ]
    for ax, ttl, marks in panels:
        for j, mk, col, lab in marks:
            for i, row in enumerate(R):
                v = row[j]
                if v is None:
                    continue
                yy = y[i] + (0.16 if j in (5, 7) else -0.16)
                if isinstance(v, tuple):
                    ax.plot(list(v), [yy] * 2, "-", color=col, lw=1.8,
                            alpha=0.6, solid_capstyle="butt")
                ax.plot(val(v), yy, mk, color=col, ms=3.6, mew=0.7)
            ax.plot([], [], mk, color=col, ms=3.6, mew=0.7, label=lab)
        ax.axvline(0.0, color="0.35", lw=0.8)
        ax.set_xscale("symlog", linthresh=0.02, linscale=0.45)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels(["$-0.5$", "0", "0.5"], fontsize=MINPT)
        ax.set_xticks([-0.1, 0.1], minor=True)
        ax.set_xlim(-0.9, 0.6)
        ax.set_title(ttl, fontsize=MINPT, loc="left")
        ax.legend(loc="lower center", ncol=2, fontsize=MINPT,
                  handletextpad=0.25, columnspacing=0.9, borderpad=0.25,
                  handlelength=1.0, borderaxespad=0.25)
    save(fig, "fig_error_hierarchy")


# ══════════════════════════════════════════════════════════════════════════
# F4  replaces the amplitude-ladder table                    [REAL + BOX G3]
# ══════════════════════════════════════════════════════════════════════════
def _wavy_rows():
    """The amplitude ladder, read from its artifact.

    `r1_sta2_wavy_amplitude_*.json` carries every column: the cell count, the
    reversed-shear fraction, the pressure share of the wall force, the four
    scores and their standard errors over six disjoint windows.  It also carries
    the machine each calculation ran on, so the second-machine repeat identifies
    itself instead of being flagged by hand.
    """
    import glob, json
    src = sorted(glob.glob(os.path.join(RESULTS,
                 "r1_sta2_wavy_amplitude_*.json")))[-1]
    with open(src) as fh:
        art = json.load(fh)
    ETA = ("0.05", "0.1", "0.2", "0.3")
    rows = []
    for fam, cases in art["families"].items():
        for name, c in cases.items():
            rows.append(dict(
                amp=round(c["two_a_over_lambda"], 2),
                cells=int(c["cells"]),
                f_sep=round(c["wall"]["f_reversed"], 3),
                form=round(c["wall"]["form_fraction"], 2),
                r2=[round(c["ode"][e]["standard_ml"], 3) for e in ETA],
                se=[round(c["uncertainty"]["block_windows_ode"][e]
                          ["standard_ml"]["sem"], 3) for e in ETA],
                machine=c.get("machine"),
            ))
    seen = {}
    for r in rows:
        seen.setdefault((r["amp"], r["cells"]), []).append(r)
    for group in seen.values():
        for r in group:
            r["repeat"] = len(group) > 1 and r["machine"] != "archer2"
    # mild grids by size, then the second-machine repeat, then the steep pair
    rows.sort(key=lambda r: (r["amp"], r["repeat"], r["cells"]))
    assert len(rows) == 6, len(rows)
    assert rows[0]["r2"][0] == 0.874 and rows[-1]["r2"][3] == -0.981, rows[-1]
    assert sum(r["repeat"] for r in rows) == 1, [r["repeat"] for r in rows]
    assert rows[3]["repeat"] and rows[3]["cells"] == 786432, rows[3]
    return rows


# ── figure 5: (a) as it was, (b) and (c) rebuilt ───────────────────────────
# Operator, 2026-08-27: (c) looks flattened with white space above it, (b) looks
# odd, (a) is fine.
#
#  (c) was framed with equal=False, so the pseudo-3-D solid took whatever aspect
#      the axes box happened to have and UNDERSTATED the steepness by about
#      2.6x.  Worse, the extruder's depth offset DY = 0.15 is LARGER than the
#      wave it was drawing (peak-to-trough 0.046 and 0.092 in cell units), so
#      the sketch was mostly slab.  A wave this shallow is the wrong subject for
#      that renderer: both walls are now 2-D cross-sections on ONE axes with the
#      aspect locked, which is the only honest way to show that one slope is
#      twice the other, and it costs a third of the height.
#
#  (b) drew one marker per case, so a two- or three-grid cluster read as scatter
#      rather than as one quantity with a range.  It is a before/after
#      comparison and is now drawn as one: a dumbbell per row, mild to steep,
#      the grid spread as a bar behind the marker.
def _wavy_wall_pair(ax, amps, cols, nper=2.0):
    """Both wall cross-sections on one axes at TRUE aspect, stacked.

    2a/lambda is a slope, so the only honest way to show that one is twice the
    other is to draw them to the same scale with the aspect locked.
    """
    x = np.linspace(0.0, nper, 1400)
    slab = 0.045
    tops = []
    y0 = 0.0
    for amp, col in zip(amps[::-1], cols[::-1]):          # steepest at the foot
        y = 0.5 * amp * (1.0 - np.cos(2 * np.pi * x)) + y0
        ax.fill_between(x, y0 - slab, y, color=g3.FRONT, lw=0, zorder=1)
        ax.plot(x, y, "-", color=col, lw=1.5, zorder=3, solid_capstyle="round")
        ax.plot([0, nper], [y0 - slab] * 2, "-", color=g3.EDGE, lw=0.6, zorder=3)
        tops.append((y0 + amp, amp, col))
        y0 += amp + 0.13
    ax.set_xlim(-0.04, nper + 0.70)
    ax.set_ylim(-slab - 0.03, y0 - 0.13 + 0.10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    return tops


def f4_wavy():
    style(8.4)
    eta = np.array([0.05, 0.10, 0.20, 0.30])
    W = _wavy_rows()
    matplotlib.rcParams["figure.constrained_layout.use"] = False

    H = 2.48
    fig = plt.figure(figsize=(TEXTWIDTH_IN, H))
    X0 = 0.02
    AW = 3.02                       # (a) block, unchanged in size
    RX = X0 + AW + 0.30             # right column
    RW = TEXTWIDTH_IN - RX - 0.04

    ax = fig.add_axes([(X0 + 0.52) / TEXTWIDTH_IN, 0.60 / H, (AW - 0.56) / TEXTWIDTH_IN, 1.72 / H])
    CH = RW * (0.545 / 2.74)          # what equal aspect gives at this width
    # (c) sits on the same line as (a)'s axes, so the two blocks share a foot
    # and the right column does not hang below the left one
    CBOT = 0.60
    bx = fig.add_axes([(RX + 0.52) / TEXTWIDTH_IN, 1.56 / H,
                       (RW - 0.60) / TEXTWIDTH_IN, 0.62 / H])
    cx = fig.add_axes([RX / TEXTWIDTH_IN, CBOT / H, RW / TEXTWIDTH_IN, CH / H])

    # ---- (a) unchanged ---------------------------------------------------
    shown = set()
    for r in W:
        col = C_B if r["amp"] == 0.10 else C_C
        mk = "o" if r["amp"] == 0.10 else "s"
        ls = ":" if r["repeat"] else "-"
        lab = None
        if (r["amp"], r["repeat"]) not in shown:
            shown.add((r["amp"], r["repeat"]))
            if r["repeat"]:
                lab = "repeat, second machine"
            else:
                cells = "/".join("%.2f" % (q["cells"] / 1e6) for q in W
                                 if q["amp"] == r["amp"] and not q["repeat"])
                lab = "$2a/\\lambda=%.2f$   %s M cells" % (r["amp"], cells)
        ax.errorbar(eta, r["r2"], yerr=r["se"], fmt=ls + mk, color=col,
                    mfc="w", mew=0.9, capsize=1.6, elinewidth=0.8,
                    alpha=0.55 if r["repeat"] else 0.9, label=lab)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(eta); ax.set_xticklabels(["0.05", "0.10", "0.20", "0.30"])
    ax.minorticks_off()
    ax.set_ylim(-1.15, 1.02)
    ax.set_xlabel(r"matching surface  $\eta_m/\delta$")
    ax.set_ylabel(r"$R^2(\tau_w)$")
    ax.set_title("(a) score against matching surface, two amplitudes",
                 fontsize=MINPT, loc="left")
    ax.legend(fontsize=MINPT, loc="lower left", handlelength=1.6,
              labelspacing=0.3, borderpad=0.3)

    # ---- (b) one dumbbell per quantity, mild -> steep ---------------------
    for k, key in enumerate(("f_sep", "form")):
        yy = 1 - k
        vals = {a: [r[key] for r in W if r["amp"] == a] for a in (0.10, 0.20)}
        lo, hi = (sum(vals[a]) / len(vals[a]) for a in (0.10, 0.20))
        bx.annotate("", xy=(hi, yy), xytext=(lo, yy),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.9,
                                    shrinkA=3.2, shrinkB=3.2), zorder=1)
        for a, col, mk in ((0.10, C_B, "o"), (0.20, C_C, "s")):
            v = vals[a]
            if max(v) - min(v) > 5e-4:
                bx.plot([min(v), max(v)], [yy, yy], "-", color=col, lw=4.0,
                        alpha=0.40, zorder=2, solid_capstyle="round")
            bx.plot(sum(v) / len(v), yy, mk, color=col, ms=4.6, zorder=3)
            # No printed value.  The prose of this section already gives both
            # endpoints ("rises from 0.486 to 0.693 ... from 72% to 93%"), and
            # a min-max range set beside a marker reads as two separate
            # measurements rather than as one quantity's grid spread.
    bx.set_yticks([1, 0])
    bx.set_yticklabels([r"$f_{\rm sep}$", "form share"], fontsize=MINPT)
    bx.set_ylim(-0.70, 1.70); bx.set_xlim(0.42, 1.06)
    bx.set_xticks([0.5, 0.7, 0.9])
    bx.tick_params(labelsize=MINPT, left=False, right=False, top=False)
    bx.set_title("(b) reversed-shear fraction and form share", fontsize=MINPT, loc="left")
    for sp in ("left", "right", "top"):
        bx.spines[sp].set_visible(False)
    bx.spines["bottom"].set_bounds(0.45, 1.0)

    # ---- (c) both surfaces, one axes, true aspect ------------------------
    tops = _wavy_wall_pair(cx, [0.10, 0.20], [C_B, C_C])
    for ytop, amp, col in tops:
        cx.text(2.08, ytop, "$2a/\\lambda=%.2f$" % amp, fontsize=MINPT,
                color=col, ha="left", va="center")
    cx.set_title("(c) the two walls at $\\lambda=2\\delta$, to scale",
                 fontsize=MINPT, loc="left", pad=2)

    save(fig, "fig_wavy_amplitude")


# ══════════════════════════════════════════════════════════════════════════
# F5  replaces BOTH ladder tables (a priori and coupled)            [REAL]
# ══════════════════════════════════════════════════════════════════════════
def _ladder_rows():
    """Both ladders, read from the rescored artifact rather than typed in.

    `r2m4_ladder_rescored_20260825.json` reproduces every printed digit of both
    tables this figure replaced: the a-priori ladder at the mesh-recorded
    surface (`apriori/ladder_L1`) and the coupled ladder on the two grids
    (`coupled/L1`, `coupled/L2`).  The constant-profile case simply has no key
    on the finer grid, which is how the artifact records that it did not
    complete -- so the figure's blank comes from the data, not from a flag.
    """
    import json
    with open(os.path.join(RESULTS, "r2m4_ladder_rescored_20260825.json")) as fh:
        d = json.load(fh)
    REF = ("A_xiao_linear4_deposited", "B_mglet_deposited",
           "C_xiao_cubic6_repaired")
    ap_src = d["apriori"]["ladder_L1"]
    AP_KEYS = [
        ("M0_equilibrium", "Equilibrium", ""),
        ("M1_pressure_gradient_ode", "Pressure-gradient ODE",
         "vs equilibrium: unresolved"),
        ("M2_hickel_modelled_convection", "Modelled convection",
         "vs ODE: better"),
        ("Xc_exact_convection_profile", "ODE $+$ exact within-layer convection",
         "vs ODE: unresolved"),
        ("Xall_all_omitted_transport", "$+$ every omitted transport term",
         "vs equilibrium: worse"),
        ("Xfull_all_transport_plus_exact_shear_stress",
         "$+$ exact resolved shear stress", "vs previous row: worse"),
    ]
    AP = [(lab,) + tuple(round(ap_src[r]["metrics"][k]["relative_rms"], 3)
                         for r in REF) + (note,)
          for k, lab, note in AP_KEYS]

    CP_KEYS = [("equilibrium", "Equilibrium"),
               ("totalGradient", "Pressure-gradient ODE"),
               ("hickel", "Modelled convection"),
               ("resolvedConvectionLinear", "$+$ resolved convection, linear"),
               ("resolvedConvectionConstant", "$+$ resolved convection, constant")]

    def pair(grid, ref, key):
        m = d["coupled"][grid][ref]["metrics"].get(key)
        return None if m is None else (round(m["relative_rms"], 3),
                                       round(m["r2"], 2))
    CP = [(lab, pair("L1", REF[1], k), pair("L1", REF[2], k),
           pair("L2", REF[1], k), pair("L2", REF[2], k))
          for k, lab in CP_KEYS]

    assert AP[0][1:4] == (0.535, 0.689, 0.655), AP[0]
    assert AP[5][1:4] == (12.731, 4.881, 7.517), AP[5]
    assert CP[2][1] == (0.440, 0.80), CP[2]
    assert CP[4][3] is None and CP[4][4] is None, CP[4]
    return AP, CP


def f5_ladder():
    style(8.4)
    AP, CP = _ladder_rows()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.32],
                          width_ratios=[1.0, 1.0], wspace=0.06)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1], sharey=axB)

    y = np.arange(len(AP))[::-1]
    h = 0.26
    for i, (name, a, b, c, note) in enumerate(AP):
        axA.barh(y[i] + h, a, height=h, color=C_A, edgecolor="none")
        axA.barh(y[i],      b, height=h, color=C_B, edgecolor="none")
        axA.barh(y[i] - h,  c, height=h, color=C_C, edgecolor="none")
        if note:
            axA.text(17.0, y[i], note, fontsize=MINPT, va="center", ha="left",
                     color="0.30")
    axA.axvline(1.0, color="0.35", ls="--", lw=0.7, zorder=0)
    axA.set_yticks(y); axA.set_yticklabels([r[0] for r in AP], fontsize=MINPT)
    axA.set_xscale("log"); axA.set_xlim(0.35, 150)
    axA.set_ylim(-0.62, len(AP) - 0.38)
    axA.set_xlabel(r"a-priori traction error  $E_\tau$")
    axA.set_title("(a) a priori, at the mesh-recorded surface, median "
                  "$\\eta_m/H=0.0935$", fontsize=7.4, loc="left")
    axA.invert_yaxis()
    axA.annotate("", xy=(3.40, y[5] + 0.32), xytext=(0.72, y[0] + 0.30),
                 arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0,
                                 connectionstyle="arc3,rad=-0.15"))
    axA.text(40.0, y[0], "adding the missing physics\nmakes it worse",
             fontsize=MINPT, color="0.25", ha="center", va="center", style="italic")
    fig.legend(handles=[
        plt.Line2D([], [], color=C_A, lw=5, label="$A$ withdrawn estimator (negative control)"),
        plt.Line2D([], [], color=C_B, lw=5, label="$B$ full-wall DNS traction (primary)"),
        plt.Line2D([], [], color=C_C, lw=5, label="$C$ curvature-aware bracket")],
        loc="outside lower center", ncol=3, fontsize=MINPT,
        handlelength=1.4, columnspacing=1.8)

    y2 = np.arange(len(CP))[::-1]
    hh = 0.24
    for i, (name, b1, c1, b2, c2) in enumerate(CP):
        # One number per entry, not two.  The bar and the dot are the FINER
        # grid, which is the grid the paper's headline numbers come from
        # (table 3 reports finest-grid values); the coarser grid is the tick
        # behind them, so the gap between tick and bar end IS the grid
        # sensitivity and needs no second printed value.  Printing
        # "coarse/fine" beside every entry put ten numbers in a panel that is
        # making one point.
        for coarse, fine, yy, col in ((b1, b2, y2[i] + hh, C_B),
                                      (c1, c2, y2[i] - hh, C_C)):
            prim = fine if fine is not None else coarse
            axB.barh(yy, prim[0], height=hh, color=col, edgecolor="none")
            axC.plot(prim[1], yy, "o", color=col, ms=4.2)
            if fine is not None:
                axB.plot(coarse[0], yy, "|", color="0.15", ms=6, mew=1.0)
                axC.plot(coarse[1], yy, "|", color="0.15", ms=6, mew=1.0)
            axB.text(prim[0] + 0.075, yy, num(prim[0], 3, False),
                     fontsize=MINPT, color=col, va="center", ha="left",
                     zorder=6, bbox=dict(boxstyle="square,pad=0.08", fc="w",
                                         ec="none", alpha=0.85))
            right = prim[1] >= 0
            axC.text(prim[1] + (0.19 if right else -0.19), yy, num(prim[1], 2),
                     fontsize=MINPT, color=col, va="center",
                     ha="left" if right else "right", zorder=6,
                     bbox=dict(boxstyle="square,pad=0.08", fc="w", ec="none",
                               alpha=0.85))
    for a_, xl, ttl in ((axB, r"coupled traction error  $E_\tau$",
                         "(b) coupled, same surface"),
                        (axC, r"coupled $R^2(\tau_s)$",
                         "(c) the second registered score")):
        a_.set_yticks(y2)
        a_.set_ylim(-0.62, len(CP) - 0.38)
        a_.set_xlabel(xl)
        a_.set_title(ttl, fontsize=7.4, loc="left")
        a_.invert_yaxis()
    axB.axvline(1.0, color="0.35", ls="--", lw=0.7, zorder=0)
    axC.axvline(0.0, color="0.35", ls="--", lw=0.7, zorder=0)
    axB.set_yticklabels([r[0] + (r"$^{\dagger}$" if r[3] is None else "")
                         for r in CP], fontsize=MINPT)
    axB.set_xlim(0, 3.05)
    axC.set_xlim(-3.55, 2.85)
    plt.setp(axC.get_yticklabels(), visible=False)
    # the two grids are identified in the caption, not on the axes
    save(fig, "fig_ladder_apriori_coupled")


# ══════════════════════════════════════════════════════════════════════════
# F6  replaces the rib-pair table and carries the cube arrays [REAL + BOXES]
# ══════════════════════════════════════════════════════════════════════════
def _sharp_rows():
    """The rib pair and the cube arrays, read from their artifact.

    `r2_4_m20_les_20260823.json` is the complete campaign: it carries all eight
    cases, it is what the paper's tables were built from, and it is the default
    of `verify_r2_4_m20.py`.  A later partial re-run exists
    (`..._20260825.json`); its point estimates agree but its bootstrap intervals
    differ and it is missing the sparse cube, so it is NOT the source here.

    Ribs are read at the common physical matching height over the cumulative
    window `cum_140`; cubes over `cum_260`.
    """
    import json
    with open(os.path.join(RESULTS, "r2_4_m20_les_20260823.json")) as fh:
        art = json.load(fh)
    K = art["cases"]

    def rib(case, label, colour):
        c = K[case]
        w = c["windows"]["cum_140"]
        m = w["matched_ym"]
        ci = m["station_block_bootstrap"]["r2_ci95"]
        f = c["drag"]["forces"]["forcesBottom"]
        share = 100.0 * f["viscous_x"] / (f["viscous_x"] + f["pressure_x"])
        # the paper prints one decimal above 10 and three below
        dp = lambda v: round(v, 1 if abs(v) >= 10 else 3)
        return (f"{label}, {c['n_cells'] / 1e6:.2f} M",
                dp(m["standard_ml_r2"]), dp(ci[0]), dp(ci[1]),
                # the paper prints this share to one decimal above 10%, two below
                ("%+.1f%%" if abs(share) >= 10 else "%+.2f%%") % share,
                "%.2f" % w["validation"]["x_reattach_over_k"],
                "%.3f" % w["eps_median"], colour)

    ribs = [rib("r24_rib_dtype_p3_G1", "$d$-type $p/k{=}3$", C_HOLD),
            rib("r24_rib_dtype_p3_G0", "$d$-type $p/k{=}3$", C_HOLD),
            rib("r24_rib_ktype_p8_G1", "$k$-type $p/k{=}8$", C_FAIL),
            rib("r24_rib_ktype_p8_G0", "$k$-type $p/k{=}8$", C_FAIL)]

    def cube(case, label, colour, excluded):
        c = K[case]
        fl = c["windows"]["cum_260"]["floor"]
        ci = fl["station_block_bootstrap"]["r2_ci95"]
        return (label % c["lambda_p"], round(fl["standard_ml_r2"], 3),
                round(ci[0], 3), round(ci[1], 3),
                round(fl["matched_ym_rib"]["standard_ml_r2"], 3),
                colour, excluded)

    cubes = [cube("r24_cube_aligned_G1", "aligned  $\\lambda_p{=}%.2f$", C_EXCL, True),
             cube("r24_cube_staggered_G1", "staggered $\\lambda_p{=}%.2f$", C_FAIL, False),
             cube("r24_cube_sparse_G1", "sparse  $\\lambda_p{=}%.3f$", C_FAIL, False)]

    assert ribs[0][1:4] == (0.588, 0.021, 0.777), ribs[0]
    # the paper's table rounds the two small upper bounds one place further
    # (-0.288 -> -0.29, -0.462 -> -0.46); the figure keeps the artifact's digit
    assert ribs[1][1:4] == (0.226, -1.304, 0.549), ribs[1]
    assert (ribs[2][1], ribs[2][2], round(ribs[2][3], 2)) == (-50.7, -139.1, -0.29), ribs[2]
    assert (ribs[3][1], ribs[3][2], round(ribs[3][3], 2)) == (-47.9, -142.1, -0.46), ribs[3]
    assert cubes[2][1:5] == (-0.932, -1.619, -0.432, -1.546), cubes[2]
    return ribs, cubes


# ── figure 9: two panels, then two stacked geometry groups ─────────────────
# Operator instructions, 2026-08-27, and what each one cost:
#
#  (1) SIDE BY SIDE, not stacked.  An earlier attempt was reverted with the
#      note "keep the stack" because the labels and annotation columns clipped.
#      That was a consequence of laying the figure out in FIGURE FRACTIONS and
#      letting text hang outside the axes at x = 1.03, not of the arrangement.
#      Every band here is sized in inches; `_sw_col_x` places the number
#      columns from the MEASURED width of the widest cell and asserts rather
#      than overflowing.
#
#  (2) NO FREE-STANDING TABLE.  What numbers remain live inside their panel,
#      on the band of the case they belong to.
#
#  (3) FEWER NUMBERS.  Panel (b) prints none: every value it used to show is
#      already in the prose of this section (+0.702, -0.013, -0.932 on the
#      floor; +0.552, -0.376, -1.546 matched), so it was duplicating the text.
#      Panel (a)'s numbers are NOT in the prose, so they could not simply go;
#      instead the two columns that were CONTROLS -- the reattachment length
#      and the cancellation ratio, both there to prove a negative -- moved into
#      the sentence that states that negative, and the panel kept the score,
#      its interval, and the signed viscous share the verdict does follow.
#
#  (4) BOTH GEOMETRY GROUPS STACKED.  Stacking the rib pair halves its width,
#      which is what pays for the larger cube tiles.  Every cube panel shows a
#      3x3 patch at ONE common cube size, so each floor is three pitches across
#      and the sparse floor is ~3x the packed ones: the panels differ in
#      spacing, not in block size, and deliberately do not share a floor size.
#
#  (5) The rib sketches used to sit under a third of a panel of empty air,
#      because frame(ytop=0.52) framed a solid that only reaches y = 0.265.
SW_NCELL = 3
SW_H = 3.68
SW_LAB_A, SW_LAB_B = 0.80, 0.54
SW_X0, SW_GAP = 0.02, 0.18
SW_AX = (TEXTWIDTH_IN - SW_X0 - SW_LAB_A - SW_GAP - SW_LAB_B - 0.08) / 2.0
SW_UNIT_A = 0.355                 # (a): a band is a text line plus its interval
SW_AX_BOT = 1.82
SW_GEO_BOT = 0.02
SW_RIBW = 1.32                    # width of the stacked rib column
SW_NAME_GAP = 0.015


def _sw_rect(x, y, w, h):
    """inches -> figure fraction, for the figure-9 layout above."""
    return [x / TEXTWIDTH_IN, y / SW_H, w / TEXTWIDTH_IN, h / SW_H]


_SW_MEASURE_FIG = None


def _sw_tw(s, fs=MINPT):
    global _SW_MEASURE_FIG
    if _SW_MEASURE_FIG is None:
        _SW_MEASURE_FIG = plt.figure(figsize=(1, 1))
    t = _SW_MEASURE_FIG.text(0, 0, s, fontsize=fs)
    bb = t.get_window_extent(renderer=_SW_MEASURE_FIG.canvas.get_renderer()); t.remove()
    return bb.width / _SW_MEASURE_FIG.dpi


def _sw_col_x(headers, rows, total, pad=0.10, inset=0.035):
    """Left edge of each number column, in AXES FRACTION, from measured text.
    `inset` keeps the first column off the axes frame, which a leading minus
    sign otherwise sits on top of."""
    n = len(headers)
    wid = [max([_sw_tw(headers[j])] + [_sw_tw(r[j]) for r in rows]) for j in range(n)]
    slack = total - inset - (sum(wid) + pad * (n - 1))
    assert slack > -1e-9, f"columns overflow by {-slack:.3f} in: {headers}"
    xs, x = [], inset
    for j in range(n):
        xs.append(x / total); x += wid[j] + pad
    return xs


def _sw_key(ax, entries):
    """A marker key INSIDE the axes.  It used to hang off the x-label, which is
    the caption's job, not the axis's."""
    handles = [plt.Line2D([], [], color="0.35", marker=mk, ls="none", ms=4.0,
                          mfc=("w" if mk == "^" else "0.35"), mew=1.0)
               for mk, _ in entries]
    ax.legend(handles, [lb for _, lb in entries], loc="lower left",
              fontsize=MINPT, frameon=False, handletextpad=0.4,
              borderpad=0.15, labelspacing=0.22, borderaxespad=0.3)


def _sw_zero_bands(ax, ys, half=0.30):
    for ym in ys:
        ax.plot([0.0, 0.0], [ym - half, ym + half], "-", color="0.45", lw=0.8,
                zorder=0)


def f6_sharp():
    style(8.4)
    RIBS, CUBES = _sharp_rows()
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(TEXTWIDTH_IN, SW_H))

    xa = SW_X0 + SW_LAB_A
    xb = xa + SW_AX + SW_GAP + SW_LAB_B
    ha = (len(RIBS) - 1 + 1.62) * SW_UNIT_A
    ax = fig.add_axes(_sw_rect(xa, SW_AX_BOT, SW_AX, ha))
    bx = fig.add_axes(_sw_rect(xb, SW_AX_BOT, SW_AX, ha))

    # ---- (a) score + interval, and the one covariate the verdict follows ----
    hdr = ["$R^2$  [95% interval]", "viscous share"]
    cells = [[num(v, 3 if abs(v) < 10 else 1) + "  ["
              + num(lo, 3 if abs(lo) < 10 else 1) + ", "
              + num(hi, 3 if abs(hi) < 10 else 1) + "]", vis]
             for (lab, v, lo, hi, vis, xr, eps, col) in RIBS]
    xs = _sw_col_x(hdr, cells, SW_AX)
    n = len(RIBS)
    ax.set_ylim(-(n - 1) - 0.95, 1.30)
    for j, h in enumerate(hdr):
        ax.text(xs[j], 1.18, h, transform=ax.get_yaxis_transform(),
                fontsize=MINPT, va="top", ha="left", color="0.35")
    for i, (lab, v, lo, hi, vis, xr, eps, col) in enumerate(RIBS):
        yt, ym = -i + 0.60, -i + 0.16
        for j, s in enumerate(cells[i]):
            ax.text(xs[j], yt, s, transform=ax.get_yaxis_transform(),
                    fontsize=MINPT, va="center", ha="left", color=col)
        ax.plot([lo, hi], [ym, ym], "-", color=col, lw=1.3, alpha=0.85)
        ax.plot(v, ym, "o", color=col, ms=4.0, zorder=3)
    _sw_zero_bands(ax, [-i + 0.16 for i in range(n)])
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks([-100, -1, 0, 1]); ax.minorticks_off()
    ax.set_xticklabels(["-100", "-1", "0", "1"], fontsize=MINPT)
    ax.set_xlim(-3.2e2, 2.6)
    ax.set_yticks([-i + 0.38 for i in range(n)])
    ax.set_yticklabels([r[0].replace(" $p/k{=}3$", "").replace(" $p/k{=}8$", "")
                        for r in RIBS], fontsize=MINPT)
    ax.set_xlabel(r"$R^2(\tau_w)$", fontsize=MINPT, labelpad=1.5)
    _sw_key(ax, [("o", r"$R^2$ at the matching height $\eta_m/k=0.1456$")])
    ax.set_title("(a) square ribs, two pitches", fontsize=MINPT,
                 loc="left", pad=4)
    ax.tick_params(labelsize=MINPT)

    # ---- (b) no numbers: every one of them is in the text of this section ---
    m = len(CUBES)
    bx.set_ylim(-(m - 1) - 0.85, 0.85)
    for i, (lab, v, lo, hi, mv, col, exc) in enumerate(CUBES):
        bx.plot([lo, hi], [-i, -i], "-", color=col, lw=1.4, alpha=0.85)
        bx.plot(v,  -i, "o", color=col, ms=4.4, zorder=3)
        bx.plot(mv, -i, "^", color=col, ms=4.4, mfc="w", mew=1.0, zorder=3)
        if exc:
            bx.text(-2.60, -i, "excluded", fontsize=MINPT, va="center",
                    ha="left", color=col, style="italic")
    _sw_zero_bands(bx, [-i for i in range(m)], half=0.42)
    bx.set_yticks([-i for i in range(m)])
    bx.set_yticklabels([c[0].split("  ")[0].split(" $")[0] for c in CUBES],
                       fontsize=MINPT)
    bx.set_xticks([-2, -1, 0, 1]); bx.set_xlim(-2.75, 1.30)
    bx.tick_params(labelsize=MINPT)
    bx.set_xlabel(r"$R^2(\tau_w)$", fontsize=MINPT, labelpad=1.5)
    _sw_key(bx, [("o", "array floor"), ("^", "rib-matched height")])
    bx.set_title("(b) cube arrays, three packings",
                 fontsize=MINPT, loc="left", pad=4)

    # ---- geometry band: both groups stacked --------------------------------
    WP = (TEXTWIDTH_IN - SW_X0 - SW_RIBW - 0.20 - 0.12) / 4.0        # one packed cube tile
    rib_h = 0.367 * SW_RIBW                            # drawn height of one sketch
    band_h = 2 * rib_h + 0.20
    name_y_top = SW_GEO_BOT + band_h + SW_NAME_GAP
    title_y = name_y_top + 0.13

    HR = 0.115
    sub = (band_h - 0.20) / 2.0
    for j2, (pk, lab, col) in enumerate(
            ((3, "$d$-type, $p/k=3$", C_HOLD), (8, "$k$-type, $p/k=8$", C_FAIL))):
        ybot = SW_GEO_BOT + (1 - j2) * (sub + 0.20)
        axg = fig.add_axes(_sw_rect(SW_X0, ybot, SW_RIBW, sub))
        g3.extrude(axg, *g3.square_ribs(pk * HR, HR), crest_colour=col)
        g3.frame(axg, ytop=0.31, equal=True); axg.set_anchor("N")
        fig.text((SW_X0 + 0.5 * SW_RIBW) / TEXTWIDTH_IN, (ybot + sub + SW_NAME_GAP) / SW_H, lab,
                 fontsize=MINPT, ha="center", va="bottom", color=col)
    fig.text(SW_X0 / TEXTWIDTH_IN, title_y / SW_H, "(c) the rib pair", fontsize=MINPT,
             ha="left", va="bottom")

    dx0 = SW_X0 + SW_RIBW + 0.20
    for j2, (lam, lab, col, stag) in enumerate(
            ((0.25, "aligned", C_EXCL, False), (0.25, "staggered", C_FAIL, True))):
        ybot = SW_GEO_BOT + (1 - j2) * (sub + 0.20)
        axc = fig.add_axes(_sw_rect(dx0, ybot, WP, sub))
        h_rel = lam ** 0.5 / SW_NCELL
        g3.cube_tile(axc, crest_colour=col, pitch=1.0 / SW_NCELL, h=h_rel,
                     stagger=stag)
        g3.cube_tile_frame(axc, h=h_rel, headroom=0.03); axc.set_anchor("N")
        fig.text((dx0 + 0.5 * WP) / TEXTWIDTH_IN, (ybot + sub + SW_NAME_GAP) / SW_H, lab,
                 fontsize=MINPT, ha="center", va="bottom", color=col)

    sx = dx0 + WP + 0.12
    axs = fig.add_axes(_sw_rect(sx, SW_GEO_BOT, 3 * WP, band_h))
    h_rel = 0.028 ** 0.5 / SW_NCELL
    g3.cube_tile(axs, crest_colour=C_FAIL, pitch=1.0 / SW_NCELL, h=h_rel)
    g3.cube_tile_frame(axs, h=h_rel, headroom=0.03); axs.set_anchor("N")
    fig.text((sx + 1.5 * WP) / TEXTWIDTH_IN, name_y_top / SW_H, "sparse", fontsize=MINPT,
             ha="center", va="bottom", color=C_FAIL)
    fig.text(dx0 / TEXTWIDTH_IN, title_y / SW_H, "(d) the three cube arrays", fontsize=MINPT,
             ha="left", va="bottom")

    # The (c)/(d) titles sit under the score panels' x-labels, and the gap
    # between them was guessed twice and wrong twice (the second guess
    # overlapped by 0.060 in).  Measure it instead, and refuse to ship a build
    # where the two collide.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lab_bottom = min(a.xaxis.label.get_window_extent(renderer=r).y0 / fig.dpi
                     for a in (ax, bx))
    clearance = lab_bottom - (title_y + 0.10)
    assert clearance >= 0.05, (
        f"figure 9: the (c)/(d) titles are {clearance:.3f} in from the score "
        f"panels' x-labels; raise SW_AX_BOT or shorten the geometry band")

    save(fig, "fig_sharp_walls")
    matplotlib.rcParams["figure.constrained_layout.use"] = True


# ══════════════════════════════════════════════════════════════════════════
# F7  replaces the coupled grid table AND the matching-surface sweep table   [REAL]
# ══════════════════════════════════════════════════════════════════════════
def _coupled_rows():
    """Both coupled panels, read from the archive rather than typed in.

    Top row: `m13_highre_coupled_20260825_summary.json`, campaigns 5600 and
    10595, grids G0/G1c/G2c = 92 160 / 307 200 / 819 200 cells.  Every printed
    digit of the grid table is reproduced, and the archive carries a bootstrap
    interval for all TWELVE rows where the table printed only seven, so all
    twelve are drawn.

    Bottom row: `scoring_reference_conditioning_l0_20260825.json`, the sweep
    RE-SCORED against the full-wall reference B.  This matters: the earlier
    sweep artifacts (`r2_3_ym_window_*`, `metric_station_results_l3_*`) hold the
    scoring against the withdrawn reconstruction, under which the equilibrium
    trend runs the OTHER WAY (Spearman -1 against +1) -- the conclusion the
    paper explicitly withdraws.  Reading those files here would have put the
    retracted result back into the figure.
    """
    import json
    with open(os.path.join(RESULTS,
              "m13_highre_coupled_20260825_summary.json")) as fh:
        camp = json.load(fh)["campaigns"]
    with open(os.path.join(RESULTS,
              "scoring_reference_conditioning_l0_20260825.json")) as fh:
        pts = json.load(fh)["points"]

    GRIDS = (("G0", 92160), ("G1c", 307200), ("G2c", 819200))
    grid = {}
    for re_h in (5600, 10595):
        for model, tag in (("equilibrium", "eq"), ("total_gradient_tble", "tb")):
            e, iv, r2, xr, fm = [], [], [], [], []
            for g, _ in GRIDS:
                m = camp[str(re_h)]["metrics"][f"{g}:{model}"]
                b = camp[str(re_h)]["phase_bootstrap_primary_intervals"][f"{g}:{model}"]
                e.append(round(m["relative_rms"], 3))
                iv.append((round(b["low"], 3), round(b["high"], 3)))
                r2.append(round(m["r2"], 3))
                xr.append(round(m["reattachment_x_over_H"], 3))
                fm.append(round(m["reversed_fraction"], 3) if re_h == 5600 else None)
            grid[(re_h, tag)] = (e, iv, r2, xr, fm)

    # the reattachment bias is measured against the shortest published bubble
    XR_REF = 4.473
    TAGS = ("0145", "0300", "0600", "0935", "1500", "2500")
    sweep = {}
    for model, tag in (("equilibrium", "eq"), ("total_gradient_tble", "tb")):
        eta, e, lo, hi, r2, dxr = [], [], [], [], [], []
        for t in TAGS:
            pt = pts[f"ym{t}_G1c_{model}"]
            w = pt["wall"]["B_mglet_dns"]
            eta.append(round(pt["ym_over_H"], 4))
            e.append(round(w["relative_rms"], 3))
            lo.append(round(w["relative_rms_interval"]["low"], 3))
            hi.append(round(w["relative_rms_interval"]["high"], 3))
            r2.append(round(w["r2"], 3))
            dxr.append(round(pt["model_reattachment_x_over_H"] - XR_REF, 3))
        sweep[tag] = (np.array(eta), np.array(e), np.array(lo), np.array(hi),
                      np.array(r2), np.array(dxr))

    assert grid[(5600, "eq")][0] == [0.226, 0.234, 0.252], grid[(5600, "eq")][0]
    assert grid[(10595, "tb")][2] == [0.976, 0.970, 0.966], grid[(10595, "tb")][2]
    assert list(sweep["eq"][1]) == [0.234, 0.503, 0.649, 0.705, 0.747, 0.795]
    assert list(sweep["tb"][4]) == [0.962, 0.938, 0.877, 0.861, 0.789, 0.758]
    return grid, sweep


def f7_coupled():
    """What moves the coupled score: refining the mesh (top row) and raising
    the matching surface (bottom row).  One subsection, one figure."""
    style(8.4)
    GR, SW = _coupled_rows()
    cells = np.array([92160, 307200, 819200], dtype=float)
    D = {(5600, "eq"): GR[(5600, "eq")], (5600, "tb"): GR[(5600, "tb")],
         (10595, "eq"): GR[(10595, "eq")], (10595, "tb"): GR[(10595, "tb")]}
    S = {(5600, "eq"): ("-o", C_EQ, "w", r"equilibrium, $Re_H=5600$"),
         (5600, "tb"): ("-s", C_TB, "w", r"TBLE, $Re_H=5600$"),
         (10595, "eq"): ("--o", C_EQ, C_EQ, r"equilibrium, $Re_H=10595$"),
         (10595, "tb"): ("--s", C_TB, C_TB, r"TBLE, $Re_H=10595$")}
    eta = SW["eq"][0]
    etap = np.array([1.9, 3.9, 7.7, 12.0, 19.3, 32.1])
    eq_e, eq_lo, eq_hi, eq_r2, eq_dxr = SW["eq"][1:]
    tb_e, tb_lo, tb_hi, tb_r2, tb_dxr = SW["tb"][1:]

    fig, AX = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 4.35))
    (a1, a2, a3), (b1, b2, b3) = AX
    for k, (e, iv, r2, xr, fm) in D.items():
        ls, col, mfc, lab = S[k]
        a1.plot(cells, e, ls, color=col, mfc=mfc, mew=1.0, label=lab)
        for x, v, i2 in zip(cells, e, iv):
            if i2:
                a1.plot([x, x], i2, color=col, lw=0.9, alpha=0.75, zorder=0)
        a2.plot(cells, r2, ls, color=col, mfc=mfc, mew=1.0)
        a3.plot(cells, xr, ls, color=col, mfc=mfc, mew=1.0)
    a1.set_yscale("log")
    a1.set_yticks([0.1, 0.2, 0.4, 0.8]); a1.set_yticklabels(["0.1", "0.2", "0.4", "0.8"])
    a1.set_ylim(0.09, 0.95); a1.set_ylabel(r"$E_\tau$")
    a1.set_title("(a) refinement: traction error\nand its 95% intervals",
                 fontsize=MINPT, loc="left")
    a1.legend(handles=[
        plt.Line2D([], [], color=C_EQ, marker="o", mfc="w", mew=1.0,
                   label="equilibrium"),
        plt.Line2D([], [], color=C_TB, marker="s", mfc="w", mew=1.0,
                   label="total-gradient TBLE")],
        fontsize=MINPT, loc="lower left", labelspacing=0.22,
        handlelength=1.5, borderpad=0.22)
    a2.set_ylabel(r"$R^2(\tau_s)$"); a2.set_ylim(0.83, 1.0)
    a2.set_title("(b) the second registered\nscore", fontsize=MINPT, loc="left")
    a3.set_ylabel(r"$x_r/H$"); a3.set_ylim(3.9, 4.95)
    a3.set_title("(c) reattachment, and\nreversed-shear coverage",
                 fontsize=MINPT, loc="left")
    a4 = a3.twinx()
    for k in ((5600, "eq"), (5600, "tb")):
        ls, col, mfc, lab = S[k]
        a4.plot(cells, D[k][4], color=col, ls=":", mfc="none", mew=0.8,
                alpha=0.55, marker="x", ms=4)
    a4.set_ylabel(r"$f_-$  (dotted, $Re_H=5600$)", fontsize=MINPT)
    a4.set_ylim(0.40, 0.72); a4.tick_params(labelsize=MINPT)
    for ax in (a1, a2, a3):
        ax.set_xscale("log"); ax.set_xticks(cells)
        ax.set_xticklabels(["92 k", "307 k", "819 k"], fontsize=MINPT)
        ax.minorticks_off(); ax.set_xlim(7.4e4, 1.02e6)
        ax.set_xlabel("cells", fontsize=MINPT)

    for e, lo, hi, col, lab in ((eq_e, eq_lo, eq_hi, C_EQ, "equilibrium"),
                                (tb_e, tb_lo, tb_hi, C_TB, "total-gradient TBLE")):
        b1.fill_between(eta, lo, hi, color=col, alpha=0.16, lw=0)
        b1.plot(eta, e, "-o", color=col, label=lab, mfc="w", mew=1.0)
    b1.axhline(1.0, color=C_REF, ls=":", lw=0.9)
    b1.set_yscale("log"); b1.set_ylabel(r"$E_\tau$")
    b1.set_title("(d) matching surface: error\nand 95% bands", fontsize=MINPT,
                 loc="left")
    b1.legend(loc="lower right", fontsize=MINPT)
    b2.plot(eta, eq_r2, "-o", color=C_EQ, mfc="w", mew=1.0)
    b2.plot(eta, tb_r2, "-s", color=C_TB, mfc="w", mew=1.0)
    b2.set_ylabel(r"$R^2(\tau_s)$"); b2.set_ylim(0.30, 1.0)
    b2.set_title("(e) both scores fall as the\nsurface rises", fontsize=MINPT,
                 loc="left")
    b3.axhspan(-0.667, 0.0, color="0.80", alpha=0.55, lw=0)
    b3.axhline(0.0, color="0.35", lw=0.7)
    b3.plot(eta, eq_dxr, "-o", color=C_EQ, mfc="w", mew=1.0)
    b3.plot(eta, tb_dxr, "-s", color=C_TB, mfc="w", mew=1.0)
    b3.set_ylabel(r"$\Delta x_r/H$"); b3.set_ylim(-1.35, 0.35)
    b3.set_title("(f) reattachment bias (grey:\npublished bubble spread)",
                 fontsize=MINPT, loc="left")
    for ax in (b1, b2, b3):
        ax.set_xscale("log"); ax.set_xticks(eta)
        ax.set_xticklabels(["0.0145", "0.0300", "0.0600", "0.0935", "0.1500",
                            "0.2500"], fontsize=MINPT, rotation=90)
        ax.minorticks_off()
        ax.set_xlabel(r"matching surface  $\eta_m/H$", fontsize=MINPT)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(eta)
        sec.set_xticklabels([f"{v:.0f}" for v in etap], fontsize=MINPT)
        sec.set_xlabel(r"$\eta_m^{+}$", fontsize=MINPT)
        ax.tick_params(top=False)
    save(fig, "fig_coupled_score")


if __name__ == "__main__":
    print("blueprint figures (printed width = %.3f in):" % TEXTWIDTH_IN)
    f3_errors(); f4_wavy(); f5_ladder()
    f6_sharp(); f7_coupled()
    print("done.")
