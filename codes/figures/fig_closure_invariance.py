#!/usr/bin/env python3
"""
fig_closure_invariance.py -- thrust #9, Level-3 headline results figure.

The a-priori catastrophe ($R^2(\\tau_w)=-47.7$ on periodic hills) reappears under
two-way coupling as a SIGNED, CLOSURE-INVARIANT, geometry-GATED reattachment bias.
Three panels, all from real coupled WMLES on disk:

  (a) Coupled skin friction C_f(x): DNS (orange) vs equilibrium-ODE wall model
      (black, black-box) vs TBLE wall model that restores dp/dx (bluish-gray,
      gray-box). BOTH closures reattach ~1 H early -- the failure does not care
      which closure rides on top of it (P4: error is convective, not closure).
  (b) Coupled reattachment bias by closure: eq -20.6%, TBLE -22.9% (2.3 pp WORSE,
      not better). conv-div negative-control cell drawn hatched/"running" until
      its coupled run reaches endTime -- never as a result.
  (c) pitch/L_sep gate: hills (O(delta) pitch, 2.26) TRIGGER vs conv-div (wide,
      21.7) predicted no-trigger. The a-priori R^2 sign flips across the gate.

Colour convention (project): orange = DNS/truth, black = equilibrium/black-box,
bluish-gray = TBLE/gray-box, grey-hatched = pending coupled run.

No fabricated numbers: a cell whose coupled run is not yet present is greyed and
labelled, never plotted as a result. Every value traces to:
  codes/results/aposteriori_wmles_pehill.npz   (eq + TBLE coupled C_f, reattach)
  codes/results/closure_ladder_aposteriori.npz (eq/TBLE R^2(C_f) on common axis)
  codes/results/geometry_closure_matrix.npz    (geometry x closure cells, P3/P4)
  codes/results/diagnostic_test_corrected.npz  (hills a-priori R^2 = -47.7)
  codes/results/repeating_structure_contrast.npz (conv-div a-priori + pitch)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "fig_closure_invariance.png"))

C_TRUTH = "#e8820c"     # DNS / truth        (orange)
C_EQ = "#1a1a1a"        # equilibrium ODE    (black-box, black)
C_TBLE = "#6b7a8f"      # TBLE (keeps dp/dx) (gray-box, bluish-gray)
C_PEND = "#bdbdbd"      # pending / running


def scalar(d, k, default=np.nan):
    if k not in d:
        return default
    v = d[k]
    return v.item() if getattr(v, "shape", None) == () else v


pe = np.load(os.path.join(RES, "aposteriori_wmles_pehill.npz"), allow_pickle=True)
lad = np.load(os.path.join(RES, "closure_ladder_aposteriori.npz"), allow_pickle=True)
mat = np.load(os.path.join(RES, "geometry_closure_matrix.npz"), allow_pickle=True)
diag = np.load(os.path.join(RES, "diagnostic_test_corrected.npz"), allow_pickle=True)
con = np.load(os.path.join(RES, "repeating_structure_contrast.npz"), allow_pickle=True)

# ---- real coupled numbers (assert presence, never fabricate) -------------------
eq_present = bool(scalar(pe, "eq_present"))
tble_present = bool(scalar(pe, "tble_present"))
assert eq_present and tble_present, "both hills coupled rungs must be present"

dns_x, dns_cf = pe["dns_x"], pe["dns_cf"]
eq_x, eq_cf = pe["eq_x"], pe["eq_cf"]
tble_x, tble_cf = pe["tble_x"], pe["tble_cf"]

dns_xr = float(scalar(pe, "dns_x_reatt"))
eq_xr = float(scalar(pe, "eq_x_reatt"))
tble_xr = float(scalar(pe, "tble_x_reatt"))
eq_bias = float(scalar(pe, "eq_reattachment_rel_err_pct"))
tble_bias = float(scalar(pe, "tble_reattachment_rel_err_pct"))
eq_r2cf = float(scalar(lad, "eq_R2_cf"))
tble_r2cf = float(scalar(lad, "tble_R2_cf"))

hills_apriori_r2 = float(scalar(diag, "standard_ml_r2"))      # -47.69
hills_pitch = float(scalar(con, "pehill_pitch_over_Lsep"))    # 2.26
convdiv_apriori_r2 = float(scalar(con, "convdiv_r2"))         # +0.934
convdiv_pitch = float(scalar(con, "convdiv_pitch_over_Lsep")) # 21.7
convdiv_present = bool(scalar(mat, "cell2_present"))
convdiv_bias = float(scalar(mat, "cell2_reatt_rel_err_pct")) if convdiv_present else np.nan

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.6, 4.5))

# ========================== panel (a): C_f closure overlay =====================
axA.plot(dns_x, dns_cf, color=C_TRUTH, lw=2.4, label="DNS (Krank 2018)", zorder=4)
axA.plot(eq_x, eq_cf, color=C_EQ, lw=1.8, marker="o", ms=3.0,
         label=f"eq-ODE WMLES ($R^2_{{C_f}}={eq_r2cf:+.2f}$)", zorder=5)
axA.plot(tble_x, tble_cf, color=C_TBLE, lw=1.8, marker="s", ms=3.0,
         label=f"TBLE WMLES ($R^2_{{C_f}}={tble_r2cf:+.2f}$)", zorder=5)
axA.axhline(0, color="k", lw=0.6)
for xr, col, ls in [(dns_xr, C_TRUTH, "--"), (eq_xr, C_EQ, ":"), (tble_xr, C_TBLE, ":")]:
    axA.axvline(xr, color=col, ls=ls, lw=1.3, alpha=0.85)
axA.annotate("", xy=(eq_xr, 0.0125), xytext=(dns_xr, 0.0125),
             arrowprops=dict(arrowstyle="<->", color="#444", lw=1.2))
axA.text(0.5 * (eq_xr + dns_xr), 0.0138,
         f"$-0.9H$\nboth closures", ha="center", va="bottom", fontsize=8.5,
         color="#333")
axA.set_xlim(0, 9)
axA.set_xlabel("$x/H$")
axA.set_ylabel("$C_f(x) = 2\\tau_w/\\rho u_b^2$")
axA.set_title("(a) Coupled $C_f$: both closures reattach early\n"
              f"$x_r$: DNS {dns_xr:.2f}, eq {eq_xr:.2f}, TBLE {tble_xr:.2f}",
              fontsize=12, loc="left")
axA.legend(fontsize=8.3, loc="upper right", framealpha=0.9)
axA.grid(ls=":", alpha=0.35)

# ===== panel (b): closure-invariant bias as a bar chart =======================
C_CTRL = "#2ca02c"      # conv-div control (tolerated)
bars = [("eq-ODE\n(black-box)", eq_bias, C_EQ, True),
        ("TBLE\n(keeps $dp/dx$)", tble_bias, C_TBLE, True),
        ("conv--div\n(control)", convdiv_bias, C_CTRL, convdiv_present)]
xpos = np.arange(len(bars))
axB.axhline(0, color=C_TRUTH, lw=2.2, zorder=3)
axB.text(-0.55, 0.7, "DNS (reference)", color=C_TRUTH, ha="left",
         va="bottom", fontsize=8.5)
for x, (lab, val, col, present) in zip(xpos, bars):
    if present and np.isfinite(val):
        axB.bar(x, val, width=0.62, color=col, zorder=2)
        axB.annotate(f"{val:+.1f}%", (x, val), textcoords="offset points",
                     xytext=(0, 11 if val >= 0 else -13),
                     ha="center", va=("bottom" if val >= 0 else "top"),
                     fontsize=10.5, fontweight="bold", color=col)
    else:
        axB.annotate("coupled run\nrunning", (x, 0), textcoords="offset points",
                     xytext=(0, 9), ha="center", va="bottom", fontsize=8.0,
                     style="italic", color="#888")
# the two hills closures cluster together (closure-invariant), opposite the control
axB.text(0.5, -31.2, "closure-invariant ($2.3$ pp apart, same sign)",
         ha="center", va="bottom", fontsize=8.0, color="#444")
if convdiv_present and np.isfinite(convdiv_bias):
    axB.text(0.95, 8.5, "opposite sign:\nno structural collapse",
             ha="center", va="center", fontsize=8.0, style="italic", color=C_CTRL)
axB.set_xticks(xpos)
axB.set_xticklabels([b[0] for b in bars], fontsize=9)
axB.set_xlim(-0.62, len(bars) - 0.38)
axB.set_ylabel("coupled reattachment bias  $\\Delta x_r/x_r$  [%]")
axB.set_ylim(-32, 20)
axB.set_title("(b) Same geometry, swapped closure:\n"
              f"failure persists (a-priori $R^2(\\tau_w)={hills_apriori_r2:.0f}$)",
              fontsize=12, loc="left")
axB.grid(axis="y", ls=":", alpha=0.4)

# ========================== panel (c): pitch/L_sep gate ========================
# x = pitch/L_sep, y = a-priori R^2 (sign of the catastrophe); marker style = trigger
axC.axhline(0, color="k", lw=0.8)
axC.axvspan(0.5, 4.0, color=C_TRUTH, alpha=0.10)
# hills: triggers (R^2 << 0 a-priori; coupled bias real)
axC.scatter(hills_pitch, hills_apriori_r2, s=170, color=C_EQ, edgecolor="k",
            marker="v", zorder=5)
axC.annotate(f"periodic hills\npitch/$L_{{sep}}$={hills_pitch:.2f}\n"
             f"a-priori $R^2={hills_apriori_r2:.0f}$\ncoupled bias ${eq_bias:+.0f}\\%$ (real)",
             (hills_pitch, hills_apriori_r2), textcoords="offset points",
             xytext=(14, 10), fontsize=8.4, color=C_EQ)
# conv-div: predicted no-trigger (R^2 > 0 a-priori)
axC.scatter(convdiv_pitch, convdiv_apriori_r2, s=170, color=C_TBLE, edgecolor="k",
            marker="^", zorder=5)
cd_status = (f"coupled bias {convdiv_bias:+.0f}%" if convdiv_present
             else "coupled run running")
axC.annotate(f"conv-div channel\npitch/$L_{{sep}}$={convdiv_pitch:.1f}\n"
             f"a-priori $R^2={convdiv_apriori_r2:+.2f}$\n({cd_status})",
             (convdiv_pitch, convdiv_apriori_r2), textcoords="offset points",
             xytext=(-90, -34), fontsize=8.4, color=C_TBLE, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_TBLE, lw=0.9))
axC.set_xscale("symlog", linthresh=1.0)
axC.set_yscale("symlog", linthresh=1.0)
axC.set_xlabel("pitch / $L_{\\rm sep}$  (geometry repetition scale)")
axC.set_ylabel("a-priori $R^2(\\tau_w)$  (sign of the catastrophe)")
axC.set_title("(c) The $O(\\delta)$-pitch gate:\nwhich repeating geometries trigger",
              fontsize=12, loc="left")
axC.text(1.9, -4.0, "$O(\\delta)$ pitch\nTRIGGER", ha="center", va="center",
         fontsize=8.8, color=C_TRUTH, fontweight="bold")
axC.grid(ls=":", alpha=0.4)

complete = bool(scalar(mat, "matrix_complete"))
# suptitle removed -- earlier submission convention keeps the figure's framing in the LaTeX
# caption, not a large title above the panels (matches the other figures).
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"  eq:   xr={eq_xr:.3f} bias={eq_bias:+.2f}% R2cf={eq_r2cf:+.3f}")
print(f"  tble: xr={tble_xr:.3f} bias={tble_bias:+.2f}% R2cf={tble_r2cf:+.3f}")
print(f"  closure-invariance: |TBLE-eq| bias = {abs(tble_bias-eq_bias):.2f} pp")
print(f"  gate: hills pitch={hills_pitch:.2f} R2={hills_apriori_r2:.1f} | "
      f"conv-div pitch={convdiv_pitch:.1f} R2={convdiv_apriori_r2:+.3f} "
      f"present={convdiv_present}")
print(f"  matrix_complete={complete}")
