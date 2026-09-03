#!/usr/bin/env python3
"""
Clean (mesh-free) replacements for the repeating-structure CLASS MAPS
(figs 3 & 4).  Supervisor feedback: the 3-D mesh overlay on the geometry
blocks is misleading -- show clean wall-profile shapes instead.

Message preserved: the ODE verdict (FAIL/HOLD) tracks the PITCH (columns),
not the boundary shape (morphology fig) or the amplitude (amplitude fig).
Badge: filled = simulated case, outline = predicted by the geometry rule.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figures"))
WALL = "#8f95a3"; EDGE = "#2b2f38"
RED = "#c0271e"; GRN = "#11823b"

# ---------- clean wall-profile generators (normalised cell width 1, baseline 0) ----------
def smooth_hill(n, a):
    x = np.linspace(0, 1, 1000); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; hw = p * 0.40; d = (x - c) / hw
        y += a * np.where(np.abs(d) <= 1, 0.5 * (1 + np.cos(np.pi * np.clip(d, -1, 1))), 0.0)
    return x, y

def wavy(n, a):
    x = np.linspace(0, 1, 1000); return x, a * 0.5 * (1 - np.cos(2 * np.pi * n * x))

def sharp_rib(n, a):
    x = np.linspace(0, 1, 3000); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; w = p * 0.42
        y[np.abs(x - c) <= w / 2] = a
    return x, y

def rounded_rib(n, a):
    x = np.linspace(0, 1, 3000); y = np.zeros_like(x); p = 1.0 / n
    for k in range(n):
        c = (k + 0.5) * p; w = p * 0.42; r = w * 0.42; d = np.abs(x - c)
        flat = d <= (w / 2 - r); slope = (d > (w / 2 - r)) & (d <= w / 2)
        y[flat] = a
        y[slope] = a * 0.5 * (1 + np.cos(np.pi * (d[slope] - (w / 2 - r)) / r))
    return x, y

def gauss_bump(a):
    x = np.linspace(0, 1, 1000); return x, a * np.exp(-((x - 0.5) / 0.13) ** 2)

def bfs(a):
    x = np.linspace(0, 1, 1000); y = np.where(x < 0.32, a, 0.0); return x, y

def single_round(a):
    x = np.linspace(0, 1, 1000); d = np.abs(x - 0.5); w = 0.34; r = 0.14
    y = np.where(d <= (w / 2 - r), a, 0.0)
    sl = (d > (w / 2 - r)) & (d <= w / 2)
    y[sl] = a * 0.5 * (1 + np.cos(np.pi * (d[sl] - (w / 2 - r)) / r)); return x, y


def draw_cell(ax, x, y, badge, filled, label, tag):
    ax.fill_between(x, 0, y, color=WALL, zorder=2)
    ax.plot(x, y, color=EDGE, lw=0.9, zorder=3)
    ax.axhline(0, color=EDGE, lw=0.9, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.35); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cfd3da"); s.set_linewidth(0.8)
    col = RED if badge == "FAIL" else GRN
    fc = col if filled else "none"; tc = "white" if filled else col
    ax.text(0.985, 1.21, badge, transform=ax.transData, ha="right", va="top",
            fontsize=7.6, fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec=col, lw=1.0))
    ax.text(0.03, 1.2, tag, transform=ax.transData, ha="left", va="top", fontsize=8, color="#444")
    if label:
        ax.text(0.5, -0.16, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.8, color=col,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.8))


def make_morphology():
    ncol = [10, 5, 2, 1]; col_lab = [r"$\lambda/h=1.0$" + "\n(10 repeats)", r"$\lambda/h=2.0$" + "\n(5 repeats)",
             r"$\lambda/h=5.0$" + "\n(2 repeats)", r"$\lambda\to\infty$" + "\n(non-repeating)"]
    rows = ["smooth\n(periodic hill / wavy)", "rounded rib\n(intermediate edge)", "sharp square rib\n($d$-/$k$-type)"]
    # (badge, filled, label) per (r,c); None label = blank
    spec = [
        [("FAIL",1,"periodic hill"),("FAIL",1,"wavy wall"),("HOLD",1,"krank hill"),("HOLD",0,"Gaussian bump")],
        [("FAIL",0,None),("FAIL",0,None),("HOLD",0,None),("HOLD",0,None)],
        [("FAIL",1,"rib $d$-type"),("HOLD",1,"rib $k$-type"),("HOLD",0,None),("HOLD",1,"BFS")],
    ]
    shapef = [smooth_hill, rounded_rib, sharp_rib]
    tags = "abcdefghijkl"
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 6.3))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.07, hspace=0.42, wspace=0.10)
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
            if r == 0:
                ax.set_title(col_lab[c], fontsize=9.0, pad=6)
        axes[r, 0].text(-0.13, 0.5, rows[r], transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=8.8)
    fig.text(0.54, 0.95, r"increasing pitch  $\lambda/h$  (to the non-repeating limit) $\rightarrow$",
             ha="center", fontsize=10.5)
    fig.text(0.018, 0.46, r"increasing edge sharpness $\rightarrow$", rotation=90, va="center", fontsize=9.5)
    fig.text(0.5, 0.015, "FAIL / HOLD badge: filled = simulated case, outline = predicted by the geometry rule "
             r"($\varepsilon\gtrsim0.5$); quantitative $\tilde\varepsilon$, $R^2$ in table~2.",
             ha="center", fontsize=8.2, color="#444")
    fig.savefig(os.path.join(OUT, "fig_repeating_class_morphology.pdf"))
    fig.savefig(os.path.join(OUT, "fig_repeating_class_morphology.png"), dpi=170)
    plt.close(fig); print("wrote fig_repeating_class_morphology")


def make_amplitude():
    ncol = [8, 4, 2, 1]; col_lab = [r"$\lambda/h=1.25$" + "\n(8 repeats)", r"$\lambda/h=2.50$" + "\n(4 repeats)",
             r"$\lambda/h=5.00$" + "\n(2 repeats)", r"$\lambda\to\infty$" + "\n(non-repeating)"]
    amps = [0.30, 0.52, 0.78, 1.02]; row_lab = [r"$a/h=0.30$", r"$a/h=0.55$", r"$a/h=0.80$", r"$a/h=1.05$"]
    spec = [
        [("FAIL",0,None),("FAIL",1,"wavy wall"),("HOLD",0,None),("HOLD",0,None)],
        [("FAIL",0,None),("FAIL",0,None),("HOLD",1,"krank hill"),("HOLD",1,"Gaussian bump")],
        [("FAIL",0,None),("FAIL",0,None),("HOLD",1,"conv--div ch."),("HOLD",0,None)],
        [("FAIL",0,None),("FAIL",1,"periodic hill"),("HOLD",0,None),("HOLD",0,None)],
    ]
    tags = "abcdefghijklmnop"
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.07, hspace=0.45, wspace=0.10)
    t = 0
    for r in range(4):
        for c in range(4):
            ax = axes[r, c]; n = ncol[c]
            x, y = (gauss_bump(amps[r]) if c == 3 else smooth_hill(n, amps[r]))
            b, f, lab = spec[r][c]
            draw_cell(ax, x, y, b, f, lab, f"({tags[t]})"); t += 1
            if r == 0:
                ax.set_title(col_lab[c], fontsize=9.0, pad=6)
        axes[r, 0].text(-0.13, 0.5, row_lab[r], transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=8.8)
    fig.text(0.54, 0.955, r"increasing pitch  $\lambda/h$  (to the non-repeating limit) $\rightarrow$",
             ha="center", fontsize=10.5)
    fig.text(0.018, 0.47, r"increasing amplitude $a/h$ $\rightarrow$", rotation=90, va="center", fontsize=9.5)
    fig.text(0.5, 0.013, "FAIL / HOLD badge: filled = simulated, outline = predicted. "
             "The verdict tracks the PITCH (columns), not the amplitude (rows).",
             ha="center", fontsize=8.2, color="#444")
    fig.savefig(os.path.join(OUT, "fig_repeating_class_amplitude.pdf"))
    fig.savefig(os.path.join(OUT, "fig_repeating_class_amplitude.png"), dpi=170)
    plt.close(fig); print("wrote fig_repeating_class_amplitude")


if __name__ == "__main__":
    # SUPERSEDED: the live generator for fig_repeating_class_morphology /
    # fig_repeating_class_amplitude is stage_classmaps_3d.py. This older variant
    # writes the SAME output filenames but with stale content (rotation=90 up-arrows,
    # $\lambda/h$ pitch symbol), so running it would silently overwrite the corrected
    # figures. Guarded off; run stage_classmaps_3d.py instead.
    raise SystemExit(
        "stage_classmaps_clean.py is superseded by stage_classmaps_3d.py "
        "(the live Fig 5/6 generator) and is guarded off so it cannot overwrite "
        "the corrected figures. Run stage_classmaps_3d.py instead.")
