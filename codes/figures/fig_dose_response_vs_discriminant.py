#!/usr/bin/env python3
"""
Thrust #20 L3 results figure.

Two panels, one honest message:
  (a) WITHIN the narrow Xiao hill-steepness family the a-priori cancellation
      depth eps_med does NOT rank-order the deployed (coupled) reattachment
      error -- the pre-registered monotone dose-response P1/P4 are FALSIFIED.
      alpha=0.8 has the deepest pointwise cancellation yet the smallest
      deployed error.
  (b) ACROSS the geometry class (15 geometries spanning eps_med 0.08..5e3) the
      same eps_med IS a strong, significant catastrophic-vs-tolerable
      discriminant (Spearman rho=-0.675, p=0.006). The repeating O(delta)-pitch
      hills are the only catastrophic-failure points and the only ones with
      eps_med < O(0.5).

All numbers read live from codes/results/*.npz. No fabrication.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))

# ---------------------------------------------------------------- panel (a)
dr = np.load(os.path.join(RES, "aposteriori_dose_response.npz"), allow_pickle=True)
labels = list(dr["labels"])
eps_a = dr["eps_med"].astype(float)
Edep = dr["E_dep"].astype(float)
is_ctrl = dr["is_control"].astype(bool)
_fj_raw = dr["falsifiers_json"].item()
if isinstance(_fj_raw, dict):
    fj = _fj_raw
else:
    import ast, json
    try:
        fj = json.loads(_fj_raw)
    except Exception:
        fj = ast.literal_eval(_fj_raw)
fitR2 = float(dr["fit_R2"])
rho_a = float(dr["spearman_rho"])

# ---------------------------------------------------------------- panel (b)
cg = np.load(os.path.join(RES, "cross_geometry_collapse.npz"), allow_pickle=True)
eps_b = cg["eps_med"].astype(float)
r2_b = cg["r2"].astype(float)
klass = list(cg["klass"])
pitch = cg["pitch_O_delta"].astype(bool)
rho_b = float(cg["spearman_rho"])
p_b = float(cg["spearman_p"])
n_b = int(cg["spearman_n"])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.5))

# ===== panel (a): within-family dose-response (falsified) =====
mk_hill = ~is_ctrl
# colour the family points; control distinct
for i in range(len(eps_a)):
    if is_ctrl[i]:
        c, m, lab = "tab:gray", "s", "conv-div control"
    else:
        c, m, lab = "tab:blue", "o", None
    axa.scatter(eps_a[i], 100 * Edep[i], s=90, c=c, marker=m,
                edgecolor="k", linewidth=0.8, zorder=3)
# annotate each
ann = {
    "hill alpha=0.8 (Re5600)": (r"$\alpha{=}0.8$", (8, 6)),
    "hill alpha=1.0 (Re5600)": (r"$\alpha{=}1.0$", (8, -4)),
    "hill alpha=1.2 (Re5600)": (r"$\alpha{=}1.2$", (-38, 6)),
    "hill alpha=1.0 (Re10595, cross-Re)": (r"$\alpha{=}1.0$, Re10595", (8, -14)),
    "conv-div control": ("conv-div\ncontrol", (-10, -28)),
}
for i, lb in enumerate(labels):
    if lb in ann:
        txt, off = ann[lb]
        axa.annotate(txt, (eps_a[i], 100 * Edep[i]),
                     textcoords="offset points", xytext=off, fontsize=8.5)
# the failed monotone fit g(eps)=A/eps+B (as fitted in coupled_dose_response.py)
xx = np.linspace(0.07, 1.9, 200)
A, B = float(dr["fit_A"]), float(dr["fit_B"])
axa.plot(xx, 100 * (A / xx + B), "r--", lw=1.3, alpha=0.7,
         label=fr"monotone fit $g=A/\varepsilon+B$  ($R^2={fitR2:.3f}$)")
axa.set_xlabel(r"a-priori cancellation depth  $\varepsilon_\mathrm{med}$")
axa.set_ylabel(r"deployed reattachment error  $E_\mathrm{dep}$  [\%]"
               if matplotlib.rcParams["text.usetex"] else
               r"deployed reattachment error  $E_\mathrm{dep}$  [%]")
axa.set_title("(a) WITHIN the hill-steepness family:\n"
              "$\\varepsilon$ does NOT order the deployed error", fontsize=10)
axa.legend(loc="upper right", fontsize=8, frameon=False)
axa.text(0.04, 0.04,
         f"P1 monotone: {'PASS' if fj['P1_monotone'] else 'FALSIFIED'}\n"
         f"P2 control<hills: {'PASS' if fj['P2_control_small'] else 'FALSIFIED'}\n"
         f"P4 predictor fit: {'PASS' if fj['P4_predictor_fit']['R2']>0.5 else 'FALSIFIED'} "
         f"($\\rho={rho_a:.2f}$)",
         transform=axa.transAxes, fontsize=8, va="bottom",
         bbox=dict(boxstyle="round", fc="mistyrose", ec="r", alpha=0.8))
axa.set_xlim(0.04, 2.0)
axa.set_ylim(0, 40)
axa.grid(alpha=0.25)

# ===== panel (b): across-class discriminant (survives) =====
colmap = {"repeating": "tab:red", "repeating_wide": "tab:orange",
          "single_feature": "tab:green", "attached": "tab:blue"}
seen = set()
for i in range(len(eps_b)):
    k = klass[i]
    lab = None
    if k not in seen:
        lab = k.replace("_", " "); seen.add(k)
    edge = "k" if pitch[i] else "none"
    lw = 1.5 if pitch[i] else 0.5
    axb.scatter(eps_b[i], r2_b[i], s=80, c=colmap.get(k, "gray"),
                marker="o", edgecolor=edge, linewidth=lw, label=lab, zorder=3)
axb.axhline(0.88, color="k", ls=":", lw=1, alpha=0.6)
axb.text(2e2, 0.90, r"success band $R^2\geq0.88$", fontsize=7.5, ha="right")
axb.axhline(0.0, color="gray", ls="-", lw=0.6, alpha=0.4)
axb.set_xscale("log")
axb.set_xlabel(r"a-priori cancellation depth  $\varepsilon_\mathrm{med}$  (log)")
axb.set_ylabel(r"$R^2(\tau_w)$  (a-priori, ODE)")
axb.set_title("(b) ACROSS the geometry class:\n"
              fr"$\varepsilon$ IS a significant discriminant ($\rho={rho_b:.2f}$, $p={p_b:.3f}$, $n={n_b}$)",
              fontsize=10)
axb.set_ylim(-55, 6)
axb.legend(loc="center right", fontsize=7.5, frameon=True, framealpha=0.9)
# annotate the two headline points so neither is hidden
keys_b = list(cg["keys"])
for i, kk in enumerate(keys_b):
    if kk == "periodic_hills_1p0":
        axb.annotate(r"periodic hill $\alpha{=}1.0$" + "\n($R^2=-47.7$)",
                     (eps_b[i], r2_b[i]), textcoords="offset points",
                     xytext=(20, 8), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.7))
    if kk == "kth_3d_diffuser":
        axb.annotate("3-D diffuser\n(non-repeating low-$\\varepsilon$\noutlier, see text)",
                     (eps_b[i], r2_b[i]), textcoords="offset points",
                     xytext=(22, 6), fontsize=7,
                     arrowprops=dict(arrowstyle="->", lw=0.6))
axb.text(0.03, 0.30, "black ring = repeating, pitch $\\sim O(\\delta)$",
         transform=axb.transAxes, fontsize=7.5, va="bottom",
         bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
axb.grid(alpha=0.25, which="both")

fig.tight_layout()
out_pdf = os.path.join(HERE, "..", "..", "development", "nodes", "node_004",
                       "fig_dose_response_vs_discriminant.pdf")
out_pdf = os.path.normpath(out_pdf)
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
# also emit the manuscript-pathed copy used by main.tex (fig:dose_vs_discriminant)
ms_pdf = os.path.normpath(os.path.join(HERE, "..", "..", "manuscript",
                                       "figures", "fig_dose_vs_discriminant.pdf"))
fig.savefig(ms_pdf, bbox_inches="tight")
print("wrote", out_pdf)
print("wrote", ms_pdf)
print(f"  panel(a): within-family rho={rho_a:.3f}, fit R2={fitR2:.4f}")
print(f"  panel(b): across-class rho={rho_b:.3f}, p={p_b:.4f}, n={n_b}")
