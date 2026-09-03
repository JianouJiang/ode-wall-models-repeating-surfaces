#!/usr/bin/env python3
r"""
Figure: geometry-class generalisation -- the established cancellation rule and
its PRE-REGISTERED extrapolation to a structurally distinct second instance
(transverse square ribs).  L3, Thrust #23.

Consumes  codes/results/rib_class_prediction.npz  (produced by
codes/analysis/rib_class_prediction.py) and renders two panels:

  (a) THE ESTABLISHED RULE (real data only, n=15).  Deep-cancellation coverage
      frac[eps<0.1] vs a-priori ODE skill R2(tau_w).  A single threshold
      tau* = 0.31 (gap midpoint) perfectly separates the structural failures
      (R2 < -1) from the tolerated geometries (Spearman rho = +0.78,
      p < 1e-3).  Colour = intrinsic geometry class.

  (b) THE PRE-REGISTERED PREDICTION (geometry axis only).  Streamwise
      length-scale parameter p/k (cavity pitch over rib height) for the
      established 2-D repeating geometries and the two rib designs.  The rib
      points are placed by GEOMETRY (fixed by the mesh); their failure outcome
      is a PREDICTION -- the wall-resolved LES that tests it is in-flight.
      The d-/k-type roughness transition (Perry 1969; Leonardi 2003, p/k ~ 7)
      coincides with the predicted wall-model-validity boundary: that
      coincidence is the new, falsifiable claim.

Colour key (by geometry class; this figure carries no model curves, so the
reserved model colours orange/black/bluish-gray/green are not used):
  repeating pitch~O(delta) = crimson;  repeating wide pitch = darkorange;
  single separated feature = steelblue;  attached boundary layer = grey.
Rib PREDICTIONS are drawn as open, hatched markers -- never filled data.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results", "rib_class_prediction.npz")
OUT = os.path.join(os.path.dirname(CODES), "manuscript", "figures",
                   "fig_rib_class_prediction.pdf")

d = np.load(RES, allow_pickle=True)
keys = [str(k) for k in d["keys"]]
klass = [str(k) for k in d["klass"]]
repeating = d["repeating"].astype(bool)
pitchO = d["pitch_O_delta"].astype(bool)
frac01 = d["frac_eps_lt0p1"].astype(float)
r2 = d["r2"].astype(float)
tau = float(d["tau_star"])
rho = float(d["spearman_rho_frac0p1"]); p = float(d["spearman_p_frac0p1"])
R2_FAIL = float(d["R2_FAIL"])

# ---- LANDED measured rib outcome (converged LES + RANS pilots) -------------
# rib_eps_regime_l2.npz carries the measured a-priori verdict on the now-
# CONVERGED wall-resolved rib LES (averaging window 100 t.u.) plus the RANS
# pilots, all scored by the frozen ODE/eps instrument.  These are DATA, drawn
# filled, over the pre-registered prediction zones.
REG = os.path.join(CODES, "results", "rib_eps_regime_l2.npz")
rib_meas = {}
if os.path.exists(REG):
    g = np.load(REG, allow_pickle=True)
    if bool(g["les_is_converged_final"]):
        labs = [str(s) for s in g["labels"]]
        for want, key in [("rib LES d-type (resolved)", "les_d"),
                          ("rib RANS k-type", "rans_k"),
                          ("rib RANS d-type", "rans_d")]:
            if want in labs:
                j = labs.index(want)
                rib_meas[key] = dict(cov=float(g["coverage"][j]),
                                     r2=float(g["r2"][j]),
                                     pk=float(g["p_over_k"][j]))


def style(k, rep, pit):
    if rep and pit:
        return "crimson", "o", r"repeating, pitch $\sim\!\mathcal{O}(\delta)$"
    if rep and not pit:
        return "darkorange", "s", "repeating, wide pitch"
    if k == "attached":
        return "0.45", "^", "attached boundary layer"
    return "steelblue", "D", "single separated feature"


def _loop(ax, cx, cy, rx, ry, color):
    """A small clockwise recirculation-loop arrow."""
    th = np.linspace(np.deg2rad(130), np.deg2rad(130 - 300), 60)
    ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color=color, lw=1.0)
    ax.annotate("", xy=(cx + rx * np.cos(th[-1]), cy + ry * np.sin(th[-1])),
                xytext=(cx + rx * np.cos(th[-3]), cy + ry * np.sin(th[-3])),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                mutation_scale=7))


def draw_ribs(ax, kind):
    """Cavity-flow schematic for square-rib roughness.
    kind='d': tight pitch -> a vortex FILLS the cavity, flow skims the crests,
              no reattachment (domain-wide cancellation -> ODE fails).
    kind='k': wide pitch  -> the flow separates behind a rib, REATTACHES on the
              floor, then re-develops before the next rib (localised -> tolerated)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.patch.set_visible(False)          # transparent -> blends with the panel bg
    fy, rh, rw = 0.4, 1.6, 0.7
    ax.fill_between([0, 10], 0, fy, color="0.82", lw=0)          # floor
    xr = [2.6, 4.9] if kind == "d" else [1.4, 8.0]              # tight vs wide
    for x0 in xr:
        ax.add_patch(Rectangle((x0, fy), rw, rh, facecolor="0.72",
                               edgecolor="0.4", lw=0.8))
        for t in np.linspace(0, rw, 4):                         # solid hatch ticks
            ax.plot([x0 + t, x0 + t - 0.18], [fy, fy - 0.18], color="0.4", lw=0.5)
    top = fy + rh
    ax.annotate("", xy=(9.4, top + 0.32), xytext=(0.4, top + 0.32),
                arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=1.0,
                                mutation_scale=8))
    if kind == "d":
        ax.plot([xr[0] + rw, xr[1]], [top, top], color="steelblue", lw=1.0)
        _loop(ax, 0.5 * (xr[0] + rw + xr[1]), fy + 0.55 * rh, 0.72, 0.5, "#c0392b")
        ax.text(5, 4.35, "$d$-type:\nvortex fills cavity", ha="center",
                va="top", fontsize=9.5, color="0.15", linespacing=1.15)
    else:
        x1 = xr[0] + rw
        xs = np.linspace(x1, x1 + 3.0, 40)
        ys = fy + (top - fy) * np.exp(-((xs - x1) / 1.5) ** 2)
        ax.plot(xs, np.maximum(ys, fy + 0.03), color="0.3", ls=(0, (4, 2)), lw=1.0)
        _loop(ax, x1 + 1.0, fy + 0.42, 0.6, 0.32, "#c0392b")
        xre = x1 + 3.0
        ax.plot([xre], [fy], marker="|", color="seagreen", ms=9, mew=1.4)
        ax.annotate("", xy=(xr[1] - 0.1, fy + 0.35), xytext=(xre, fy + 0.35),
                    arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=0.9,
                                    mutation_scale=7))
        ax.text(xre, fy - 0.22, "reattach", ha="center", va="top",
                fontsize=8, color="seagreen")
        ax.text(5, 4.35, "$k$-type:\nreattaches between ribs", ha="center",
                va="top", fontsize=9.5, color="0.15", linespacing=1.15)


# Print-size parity with the dose-response reference (~7.1 pt printed labels):
# this canvas is 10.0 in wide at full \textwidth, so labels need ~13 pt.
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "legend.fontsize": 11.5, "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6))

# ===== panel (a): the established rule (real data only) =====================
axa = ax[0]
# clip R2 for display so the catastrophic failures don't crush the OK cluster
r2disp = np.clip(r2, -60, 1.05)
for i in range(len(keys)):
    c, m, _ = style(klass[i], bool(repeating[i]), bool(pitchO[i]))
    axa.scatter(frac01[i], r2disp[i], c=c, marker=m, s=52, edgecolor="k",
                linewidth=0.4, zorder=3)
axa.axvline(tau, color="k", ls="--", lw=1.1, zorder=2)
axa.axhline(R2_FAIL, color="0.5", ls=":", lw=0.9, zorder=1)
# shade the structural-failure quadrant (high coverage, catastrophic R2)
axa.axvspan(tau, 0.62, ymin=0.0, ymax=(R2_FAIL + 60) / (1.05 + 60),
            color="crimson", alpha=0.07, zorder=0)
axa.text(tau + 0.012, 1.5, r"$\tau^\ast=%.2f$" % tau, fontsize=11)
axa.text(0.605, R2_FAIL - 2.5, r"$R^2\!=\!-1$ skill floor", fontsize=9.5,
         color="0.45", ha="right", va="top")
axa.text(0.605, -60, "structural-failure region", color="crimson",
         fontsize=10.5, va="bottom", ha="right")
# structural failures -- short labels in clear spots (detail lives in the text)
axa.annotate("periodic hills",
             xy=(frac01[keys.index("periodic_hills_1p0")],
                 r2disp[keys.index("periodic_hills_1p0")]),
             xytext=(0.40, -43), fontsize=10, ha="right",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
axa.annotate("3-D diffuser", xy=(frac01[keys.index("kth_3d_diffuser")],
             r2disp[keys.index("kth_3d_diffuser")]),
             xytext=(0.505, -12), fontsize=10, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
# landed rib d-type LES star (short label; R^2 / "left of tau*" nuance in text)
if "les_d" in rib_meas:
    m = rib_meas["les_d"]
    axa.scatter(m["cov"], np.clip(m["r2"], -60, 1.05), marker="*", s=320,
                facecolor="crimson", edgecolor="k", linewidth=0.8, zorder=6)
    axa.annotate("rib $d$-type LES\n(landed, fails)",
                 xy=(m["cov"], np.clip(m["r2"], -60, 1.05)),
                 xytext=(0.02, -32), fontsize=10, ha="left", color="crimson",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="crimson"))
axa.set_xlim(-0.02, 0.62)
axa.set_ylim(-62, 8)
axa.set_xlabel(r"deep-cancellation coverage  $\mathrm{frac}[\bar\varepsilon<0.1]$")
axa.set_ylabel(r"a-priori ODE skill  $R^2(\tau_w)$  (clipped at $-60$)")
axa.set_title("(a) the established rule  ($n=15$, real data)",
              fontsize=13, loc="left")
axa.grid(True, ls="-", lw=0.3, alpha=0.35)

# ===== panel (b): the pre-registered prediction (geometry axis) =============
axb = ax[1]
DKT = float(d["dk_transition_pk"])
x_reatt = float(d["rib_x_reatt_k"][0])
axb.set_xlim(0, 13)
axb.set_ylim(-2.0, 3.0)
YT = 0.63          # axes-fraction top of the DATA region (top strip = schematics)

# GEOMETRY FORECAST (before the LES): fail for p/k below the d/k transition,
# tolerated above -- shaded FULL height so the colour runs continuously behind
# the (transparent) cavity schematics.
axb.axvspan(0, DKT, color="crimson", alpha=0.06, zorder=0)
axb.axvspan(DKT, 13, color="seagreen", alpha=0.06, zorder=0)
axb.text(0.3, 1.02, "forecast:\nFAIL", color="crimson", fontsize=9.5, va="top")
axb.text(12.7, 1.02, "forecast:\ntolerated", color="seagreen", fontsize=9.5,
         va="top", ha="right")
# outcome split (R2=0); d/k transition band (in the gap between the schematics)
axb.axhline(0.0, color="k", lw=0.9, ls="--", zorder=1)
axb.axvspan(DKT - 0.6, DKT + 0.6, color="0.6", alpha=0.16, zorder=0)
axb.axvline(DKT, color="0.4", ls="--", lw=1.0, zorder=1)
axb.text(DKT, -0.55, "$d$/$k$\ntransition", fontsize=9.5, ha="center",
         va="center", color="0.3",
         bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", alpha=0.92))
# reattachment length; label sits mid-height along the line (clear of the point
# annotations at the bottom, and of the schematic strip on top)
axb.plot([x_reatt, x_reatt], [-2.0, 1.2], color="seagreen", ls=":", lw=1.0, zorder=1)
axb.text(x_reatt - 0.22, 0.15, r"$x_{\rm reatt}\approx%.1fk$" % x_reatt,
         fontsize=9, color="seagreen", rotation=90, va="bottom", ha="center")

# landed data at real (p/k, R2): LES = star (high fidelity); RANS = open hatched
# circle (a pilot -- unreliable in separation, cannot certify the ODE).
order = [("les_d", "$d$-type LES", "crimson", "*", (0.35, 0.55)),
         ("rans_d", "$d$-type RANS", "crimson", "o", (-1.9, 0.85)),
         ("rans_k", "$k$-type RANS", "darkorange", "o", (0.5, 0.5))]
for key, name, c, mk, (dx, dy) in order:
    if key not in rib_meas:
        continue
    m = rib_meas[key]
    pk, r2v = m["pk"], float(np.clip(m["r2"], -2.0, 3.0))
    if mk == "*":
        axb.scatter(pk, r2v, s=210, marker="*", facecolor=c, edgecolor="k",
                    linewidth=0.8, zorder=6)
    else:
        axb.scatter(pk, r2v, s=95, marker="o", facecolors="none", edgecolors=c,
                    linewidth=1.8, hatch="///", zorder=6)
    axb.annotate("%s\n$R^2\\!=\\!%.2f$" % (name, m["r2"]), xy=(pk, r2v),
                 xytext=(pk + dx, r2v + dy), fontsize=9.5, color=c, ha="left",
                 va="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3,
                                             mutation_scale=13, shrinkB=4))

# the visual DEFINITION of d-/k-type: cavity-flow schematics (top strip)
draw_ribs(axb.inset_axes([0.02, 0.74, 0.37, 0.24]), "d")
draw_ribs(axb.inset_axes([0.61, 0.74, 0.37, 0.24]), "k")

axb.set_xlabel(r"streamwise length-scale ratio  $p/k$  (pitch / rib height)")
axb.set_ylabel(r"a-priori ODE skill  $R^2(\tau_w)$")
axb.set_title("(b) pre-registered rib prediction, now LANDED", fontsize=13, loc="left")
axb.grid(True, ls="-", lw=0.3, alpha=0.3)

# legend (shared, classes for panel a + prediction marker)
handles = []
seen = set()
for i in range(len(keys)):
    c, m, lab = style(klass[i], bool(repeating[i]), bool(pitchO[i]))
    if lab not in seen:
        seen.add(lab)
        handles.append(Line2D([], [], marker=m, color="w", markerfacecolor=c,
                              markeredgecolor="k", markersize=8, label=lab))
handles.append(Line2D([], [], marker="*", color="w", markerfacecolor="crimson",
                      markeredgecolor="k", markersize=13,
                      label="rib LES (high-fidelity)"))
handles.append(Line2D([], [], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="0.4", markeredgewidth=1.8,
                      markersize=10, label="rib RANS (pilot)"))
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
# also drop a PNG next to the script for quick inspection
png = os.path.join(HERE, "fig_rib_class_prediction.png")
fig.savefig(png, dpi=150, bbox_inches="tight")
print(f"Wrote {png}")
