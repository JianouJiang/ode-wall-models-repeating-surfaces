#!/usr/bin/env python3
"""
fig_geometry_closure_matrix.py -- thrust #9 L2/L3 figure.

Renders the coupled GEOMETRY x CLOSURE matrix from geometry_closure_matrix.npz:
  (a) a-priori R^2(tau_w) catastrophe  ->  coupled finite signed reattachment bias
      (the renormalisation: catastrophe is not removed, it reappears as a bounded
       signed bias);
  (b) pitch/L_sep gate: hills (O(delta) pitch) TRIGGER vs conv-div (wide) no-trigger.

A cell whose coupled run is not yet present is drawn hatched/greyed and labelled
"running" -- never as a result.  No fabricated numbers.

Colour convention (project): equilibrium ODE = black (black-box closure proxy),
TBLE = bluish-gray (gray-box), DNS/truth anchor = orange.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "fig_geometry_closure_matrix.png"))

C_EQ = "#1a1a1a"        # equilibrium ODE  (black-box)
C_TBLE = "#6b7a8f"      # TBLE             (bluish-gray)
C_TRUTH = "#e8820c"     # DNS truth        (orange)
C_PEND = "#bdbdbd"      # pending/running

d = np.load(os.path.join(RES, "geometry_closure_matrix.npz"), allow_pickle=True)


def g(k, default=np.nan):
    v = d[k]
    return v.item() if getattr(v, "shape", None) == () else v


n = int(g("n_cells"))
cells = []
for i in range(n):
    cells.append(dict(
        geometry=str(g(f"cell{i}_geometry")),
        closure=str(g(f"cell{i}_closure")),
        present=bool(g(f"cell{i}_present")),
        apriori_r2=float(g(f"cell{i}_apriori_r2")),
        reatt=float(g(f"cell{i}_reatt_rel_err_pct")),
        pitch=float(g(f"cell{i}_pitch_over_Lsep")),
        eps=float(g(f"cell{i}_apriori_eps_med")),
    ))

labels = {
    ("periodic_hills", "equilibrium_ODE"): "hills × eq-ODE",
    ("periodic_hills", "TBLE_dpdx"): "hills × TBLE",
    ("conv_div_channel", "equilibrium_ODE"): "conv-div × eq-ODE",
}
colmap = {"equilibrium_ODE": C_EQ, "TBLE_dpdx": C_TBLE}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

# ---- panel (a): a-priori R^2 catastrophe -> coupled signed reattachment bias ----
xs = np.arange(n)
for i, c in enumerate(cells):
    lab = labels.get((c["geometry"], c["closure"]), f"{c['geometry']}×{c['closure']}")
    col = colmap.get(c["closure"], C_EQ)
    if c["present"] and np.isfinite(c["reatt"]):
        axA.bar(i, c["reatt"], color=col, edgecolor="k", width=0.6, zorder=3)
        axA.text(i, c["reatt"] - 1.4, f"{c['reatt']:+.1f}%", ha="center", va="top",
                 fontsize=10, fontweight="bold")
    else:
        axA.bar(i, -10, color=C_PEND, edgecolor="k", hatch="//", width=0.6,
                alpha=0.6, zorder=3)
        axA.text(i, -5, "running\n(P3)", ha="center", va="center", fontsize=9,
                 style="italic", color="#444")
axA.axhline(0, color="k", lw=0.8)
axA.set_xticks(xs)
axA.set_xticklabels([labels.get((c["geometry"], c["closure"]), "") for c in cells],
                    rotation=12, fontsize=9)
axA.set_ylabel("coupled reattachment bias  $\\Delta x_r/x_r$  [%]")
axA.set_title("(a) a-priori catastrophe $\\to$ coupled signed bias\n"
              "hills a-priori $R^2(\\tau_w)=-47.7$ (both closures)", fontsize=10)
axA.set_ylim(-28, 4)
axA.grid(axis="y", ls=":", alpha=0.4)

# ---- panel (b): pitch/L_sep gate ------------------------------------------------
for c in cells:
    col = colmap.get(c["closure"], C_EQ)
    yval = abs(c["reatt"]) if (c["present"] and np.isfinite(c["reatt"])) else np.nan
    mk = "o" if c["closure"] == "equilibrium_ODE" else "s"
    if np.isfinite(yval):
        axB.scatter(c["pitch"], yval, s=130, color=col, edgecolor="k", marker=mk,
                    zorder=3, label=labels.get((c["geometry"], c["closure"])))
        axB.annotate(labels.get((c["geometry"], c["closure"])),
                     (c["pitch"], yval), textcoords="offset points",
                     xytext=(8, 6), fontsize=9)
    else:
        axB.scatter(c["pitch"], 1.0, s=130, color=C_PEND, edgecolor="k", marker=mk,
                    hatch="//", zorder=3, alpha=0.7)
        axB.annotate(labels.get((c["geometry"], c["closure"])) + "\n(running)",
                     (c["pitch"], 1.0), textcoords="offset points",
                     xytext=(8, 6), fontsize=8, style="italic", color="#444")
axB.axvspan(0.5, 4, color=C_TRUTH, alpha=0.10)
axB.text(2.0, 26, "O($\\delta$) pitch\nTRIGGER", ha="center", fontsize=9,
         color=C_TRUTH, fontweight="bold")
axB.text(21.7, 26, "wide pitch\nno trigger", ha="center", fontsize=9, color="#444")
axB.set_xscale("log")
axB.set_xlabel("pitch / $L_{\\rm sep}$  (geometry repetition scale)")
axB.set_ylabel("|coupled reattachment bias|  [%]")
axB.set_title("(b) pitch/$L_{\\rm sep}$ gate: which repeating\ngeometries trigger the bias",
              fontsize=10)
axB.set_ylim(-1, 30)
axB.grid(ls=":", alpha=0.4)

comp = bool(g("matrix_complete"))
fig.suptitle("Coupled WMLES geometry × closure matrix  "
             + ("(complete)" if comp else "(conv-div control still running — P3 pending)"),
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}  (matrix_complete={comp})")
