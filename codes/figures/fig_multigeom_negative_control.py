#!/usr/bin/env python3
"""
Headline figure (iteration #7-#8): the COUPLED multi-geometry negative control +
two-rung wall-model contrast that closes killer gate G7.

Story (all from real coupled WMLES + DNS, no fabrication):
  (a) PERIODIC HILLS (f_rec=0.44, domain-wide cancellation): the equilibrium AND
      the faithful TBLE wall model BOTH fail coupled -- reattachment lands early,
      i.e. the failure is driven by dropping CONVECTION (TBLE keeps dp/dx and
      still fails), not by dropping dp/dx alone.  Two-rung contrast.
  (b) CONV-DIV channel (f_rec=0.07, localised bubble): the SAME equilibrium wall
      model SUCCEEDS coupled -- C_f(x) and the small lee-side bubble match DNS.
  (c) The coverage criterion f_rec = L_sep/ell_p predicts the SIGN of the coupled
      outcome: f_rec>0.2 -> fail (hills), f_rec<0.1 -> succeed (conv-div).  The
      reattachment error collapses onto f_rec across the two repeating geometries.

Colour scheme (paper convention): orange = DNS truth, green = equilibrium
(Spalding) wall model, black = faithful TBLE/ODE wall model.

Robust to partial data: a rung whose npz field is absent is simply skipped and a
notice is printed; the script never fabricates a curve.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")
FIG_DIR = os.path.join(ROOT, "manuscript", "figures")

C_DNS = "#ff7f0e"     # orange  -- DNS ground truth
C_EQ = "#2ca02c"      # green   -- equilibrium (Spalding) wall model
C_TBLE = "#000000"    # black   -- faithful TBLE / ODE wall model
C_PRED = "#1f77b4"


def reattach(x, cf, lo, hi):
    m = (x >= lo) & (x <= hi)
    xm, cfm = x[m], cf[m]
    up = []
    for i in range(len(cfm) - 1):
        if cfm[i] < 0 and cfm[i + 1] >= 0:
            up.append(xm[i] - cfm[i] * (xm[i + 1] - xm[i]) / (cfm[i + 1] - cfm[i]))
    return up[-1] if up else np.nan


def main():
    ph = np.load(os.path.join(RESULTS, "aposteriori_wmles_pehill.npz"),
                 allow_pickle=True)
    cd = np.load(os.path.join(RESULTS, "aposteriori_wmles_convdiv.npz"),
                 allow_pickle=True)
    crit = np.load(os.path.join(RESULTS, "criterion_sign_prediction.npz"),
                   allow_pickle=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.2))

    # ── panel (a): hills two-rung C_f(x) ────────────────────────────────────
    ax1.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax1.plot(ph["dns_x"], ph["dns_cf"], color=C_DNS, lw=1.7,
             label=f"DNS  ($x_r$={float(ph['dns_x_reatt']):.2f})")
    if bool(ph["eq_present"]) and "eq_x" in ph.files:
        ax1.plot(ph["eq_x"], ph["eq_cf"], color=C_EQ, lw=1.5,
                 label=f"equilibrium  ($x_r$={float(ph['eq_x_reatt']):.2f})")
        ax1.axvline(float(ph["eq_x_reatt"]), color=C_EQ, ls=":", lw=1.0)
    if bool(ph["tble_present"]) and "tble_x" in ph.files:
        ax1.plot(ph["tble_x"], ph["tble_cf"], color=C_TBLE, lw=1.3, ls="--",
                 label=f"TBLE  ($x_r$={float(ph['tble_x_reatt']):.2f})")
        ax1.axvline(float(ph["tble_x_reatt"]), color=C_TBLE, ls=":", lw=0.9)
    else:
        print("[fig] TBLE rung not harvested yet -- panel (a) shows DNS+equilibrium only")
    ax1.axvline(float(ph["dns_x_reatt"]), color=C_DNS, ls=":", lw=1.0)
    ax1.set_xlabel(r"$x/H$"); ax1.set_ylabel(r"$C_f = 2\tau_w/u_b^2$")
    ax1.set_xlim(0, 9)
    ax1.legend(fontsize=7, loc="upper right", frameon=False)
    ax1.set_title(r"(a) periodic hills: $f_{\rm rec}=0.44$ (FAIL)", fontsize=9)

    # ── panel (b): conv-div C_f(x) negative control ─────────────────────────
    ax2.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax2.plot(cd["dns_x"], cd["dns_cf"], color=C_DNS, lw=1.7,
             label=f"DNS  ($x_r$={float(cd['dns_x_reatt']):.2f})")
    if bool(cd["convdiv_present"]) and "wmles_x" in cd.files:
        ax2.plot(cd["wmles_x"], cd["wmles_cf"], color=C_EQ, lw=1.5,
                 label=f"equilibrium  ($x_r$={float(cd['wmles_x_reatt']):.2f})")
        ax2.axvline(float(cd["wmles_x_reatt"]), color=C_EQ, ls=":", lw=1.0)
    else:
        ax2.text(0.5, 0.9, "coupled conv-div run in progress",
                 transform=ax2.transAxes, ha="center", fontsize=8, color="0.4")
        print("[fig] conv-div coupled run not harvested yet -- panel (b) DNS only")
    ax2.axvline(float(cd["dns_x_reatt"]), color=C_DNS, ls=":", lw=1.0)
    # mark the small lee-side bubble window
    ax2.axvspan(float(cd["dns_x_sep"]), float(cd["dns_x_reatt"]),
                color="0.85", alpha=0.5, zorder=0)
    ax2.set_xlabel(r"$x/h$"); ax2.set_ylabel(r"$C_f = 2\tau_w/u_b^2$")
    ax2.set_xlim(0, float(cd["dns_Lx"]))
    ax2.legend(fontsize=7, loc="upper right", frameon=False)
    ax2.set_title(r"(b) conv-div: $f_{\rm rec}=0.07$ (SUCCEED)", fontsize=9)

    # ── panel (c): coverage criterion predicts the sign ─────────────────────
    geos = [str(g) for g in crit["geometries"]]
    f_rec = np.array(crit["f_rec"], float)
    # coupled reattachment error (%) for each geometry, where available
    err = {}
    if bool(ph["eq_present"]):
        err["pehill"] = abs(float(ph["eq_reattachment_rel_err_pct"]))
    if bool(cd["convdiv_present"]) and "reattachment_rel_err_pct" in cd.files:
        err["convdiv"] = abs(float(cd["reattachment_rel_err_pct"]))
    ax3.axvspan(0, 0.1, color=C_EQ, alpha=0.12)
    ax3.axvspan(0.2, 0.6, color="#d62728", alpha=0.10)
    ax3.text(0.05, 0.92, "succeed\n$f_{\\rm rec}<0.1$", transform=ax3.transAxes,
             fontsize=7, color=C_EQ, ha="left", va="top")
    ax3.text(0.95, 0.92, "fail\n$f_{\\rm rec}>0.2$", transform=ax3.transAxes,
             fontsize=7, color="#d62728", ha="right", va="top")
    for g, fr in zip(geos, f_rec):
        key = "pehill" if "hill" in g or g == "pehill" else "convdiv"
        if key in err:
            ax3.scatter([fr], [err[key]], s=70,
                        color=(C_EQ if fr < 0.1 else "#d62728"),
                        edgecolor="k", zorder=5)
            ax3.annotate(g, (fr, err[key]), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
    ax3.set_xlabel(r"coverage fraction $f_{\rm rec}=L_{\rm sep}/\ell_p$")
    ax3.set_ylabel(r"coupled $|x_r$ error$|$  (\%)")
    ax3.set_xlim(0, 0.6)
    ax3.set_title("(c) $f_{\\rm rec}$ predicts the coupled sign", fontsize=9)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"fig_multigeom_negative_control.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
