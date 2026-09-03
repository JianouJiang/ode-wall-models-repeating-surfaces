#!/usr/bin/env python3
r"""
onset_steepness_falsification_l2.py   (L2 implementation & experiments -- node_003)
==================================================================================
The n>=4 (a/delta x lambda/delta) wavy sweep is COMPLETE.  This script runs the
PRE-REGISTERED F1 steepness-collapse verdict (frozen at L1, node_001) AND the
controlled matched-steepness falsification that the new sweep makes possible.

WHY a NEW analysis object (vs node_002's onset_steepness_sweep_l2.py):
  node_002 only evaluated the per-amplitude boundary crossings and the F1 CV
  test.  The new sweep adds something the n=2 data could NOT: CASES AT MATCHED
  MAX-SLOPE S* WITH DIFFERENT eps.  That lets us test, as a controlled
  experiment, whether S* or eps governs ODE failure -- the make-or-break the
  L1-Judge flagged ("two amplitudes define one ratio; a coincidence cannot be
  ruled out").  At n>=4 the coincidence is RULED OUT.

NOTHING is re-implemented: evaluate(.,Y_IDX=10), coverage_metrics, and the
boundary-crossing algorithm are imported VERBATIM from the production pipeline
(onset_boundary_map_l3).  F1 thresholds are FROZEN from L1.

OUTPUT: codes/results/onset_steepness_falsification_l2.npz
        development/nodes/node_003/fig_onset_falsification.png  (also codes/figures)
"""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
FIGS = os.path.join(CODES, "figures")
NODE = os.path.join(os.path.dirname(CODES), "development", "nodes", "node_003")
sys.path.insert(0, HERE)

from onset_boundary_map_l3 import (                                   # noqa: E402
    evaluate, Y_IDX, FAIL_RELRMS, DELTA, md5,
    discover_cases, score_case, pitch_crossing,
)

assert Y_IDX == 10, "matching index drifted from the paper-wide standard"
assert FAIL_RELRMS == 0.5, "catastrophe screen drifted"
assert DELTA == 1.0, "outer scale drifted"

# pre-registered F1 thresholds (FROZEN at L1, not tunable here)
CV_S_CONFIRM = 0.15
RATIO_CONFIRM = 5.0
CV_S_KILL = 0.30
A_TARGETS = [0.05, 0.10, 0.20, 0.40]


def cv(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / np.mean(x))


def f1_verdict(cv_s, cv_eps):
    ratio = cv_eps / cv_s if (np.isfinite(cv_s) and cv_s > 0) else np.nan
    if np.isfinite(cv_s) and np.isfinite(cv_eps):
        if (cv_s >= cv_eps) or (cv_s > CV_S_KILL):
            return "KILL", ratio
        if (cv_s <= CV_S_CONFIRM) and (ratio >= RATIO_CONFIRM):
            return "CONFIRM", ratio
    return "DIRECTIONAL", ratio


def auc(score, label):
    """AUC of `score` as a classifier of label==1 (fail) vs 0 (tol), via the
    Mann-Whitney U / rank statistic.  Returns AUC oriented so >0.5 means
    'higher score => more failure'.  Also returns the |AUC-0.5|-symmetric
    discriminability D = max(AUC, 1-AUC) so a perfect SEPARATOR (either sign)
    reads 1.0 and a non-separator reads 0.5."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    pos = score[label == 1]
    neg = score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan, np.nan
    # U = sum over pairs of [pos>neg] + 0.5[pos==neg]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    a = (gt + 0.5 * eq) / (pos.size * neg.size)
    return float(a), float(max(a, 1.0 - a))


def main():
    print("=" * 80)
    print("ONSET-STEEPNESS FALSIFICATION (L2 node_003)")
    print("  pre-registered F1 + matched-steepness controlled test")
    print("  locked: evaluate(Y_IDX=%d), crossing algo from onset_boundary_map_l3"
          % Y_IDX)
    print("=" * 80)

    # ---- regression guards (bit-exact) --------------------------------------
    hill = os.path.join(RESULTS,
                        "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    r2_hill = evaluate(hill)["r2"]
    print("[guard 1] canonical periodic-hill R2 = %.8f (headline -47.69)" % r2_hill)
    assert abs(r2_hill - (-47.68617253)) < 1e-6, "canonical hill R2 drifted!"
    blade_md5 = md5(os.path.join(RESULTS, "blade_severance_l3.npz"))
    print("[guard 2] blade_severance_l3.npz md5 =", blade_md5[:8])
    assert blade_md5 == "60427e650592c2fdc0db301c228a273c", "blade npz drifted!"
    rib = os.path.join(RESULTS, "rib_les_dtype_wall_profiles.npz")
    r2_rib = np.nan
    if os.path.exists(rib):
        r2_rib = evaluate(rib)["r2"]
        print("[guard 3] rib LES (sharp d-type) R2 = %.8f" % r2_rib)
        assert abs(r2_rib - (-0.94317196)) < 1e-6, "rib LES R2 drifted!"

    # ---- score every wavy case at the 4 pre-registered amplitudes -----------
    tags = discover_cases()
    rows = [r for r in (score_case(t) for t in tags) if r is not None]
    rows = [r for r in rows
            if any(abs(r["a_over_delta"] - a) < 0.005 for a in A_TARGETS)]
    rows.sort(key=lambda r: (r["a_over_delta"], r["lam_over_delta"]))
    print("\nscored %d wavy cases at a/d in %s" % (len(rows), A_TARGETS))
    print("%-13s %5s %7s %7s %8s %8s %7s %s" %
          ("tag", "a/d", "lam/d", "S*", "eps", "relRMS", "R2", "cls"))
    for r in rows:
        cls = "FAIL" if r["fail"] else "tol"
        print("%-13s %5.2f %7.2f %7.3f %8.3f %8.3f %7.2f  %s" %
              (r["tag"], r["a_over_delta"], r["lam_over_delta"], r["slope"],
               r["eps_med"], r["relRMS"], r["r2"], cls))

    Sall = np.array([r["slope"] for r in rows])
    Eall = np.array([r["eps_med"] for r in rows])
    Fall = np.array([1 if r["fail"] else 0 for r in rows])
    Aall = np.array([r["a_over_delta"] for r in rows])
    Lall = np.array([r["lam_over_delta"] for r in rows])
    R2all = np.array([r["r2"] for r in rows])

    # ---- per-amplitude boundary-#1 (steep-onset) crossings ------------------
    print("\n" + "-" * 72)
    print("BOUNDARY CROSSINGS (first relRMS=0.5 crossing per amplitude):")
    cross = {}
    for a_t in A_TARGETS:
        pc = pitch_crossing(rows, a_t)
        if pc is None:
            print("  a/d=%.2f : <2 ladder pts" % a_t)
            continue
        lc = pc.get("lam_c", np.nan)
        lo, hi = pc["bracket"]
        bracketed = bool(np.isfinite(lc) and lo < lc < hi)
        cross[a_t] = dict(pc, bracketed=bracketed)
        order = np.argsort(pc["lam"])
        lad = " ".join("%.2f:%.2f" % (pc["lam"][i], pc["relRMS"][i])
                       for i in order)
        print("  a/d=%.2f  lam_c=%6s  S*_c=%7s  eps_c=%7s  bracket=[%s,%s] %s"
              % (a_t, ("%.3f" % lc) if np.isfinite(lc) else "  --  ",
                 ("%.4f" % pc.get("slope_c", np.nan)) if np.isfinite(lc) else " -- ",
                 ("%.3f" % pc.get("eps_c", np.nan)) if np.isfinite(lc) else " -- ",
                 ("%.2f" % lo) if np.isfinite(lo) else "--",
                 ("%.2f" % hi) if np.isfinite(hi) else "--",
                 "BRACKETED" if bracketed else "(coarse/none)"))
        print("           ladder lam:relRMS = %s" % lad)

    ok = [a for a in A_TARGETS if a in cross and cross[a]["bracketed"]]
    Sc = [cross[a]["slope_c"] for a in ok]
    Ec = [cross[a]["eps_c"] for a in ok]
    Lc = [cross[a]["lam_c"] for a in ok]
    cv_s, cv_e = cv(Sc), cv(Ec)
    verdict, ratio = f1_verdict(cv_s, cv_e)

    print("\n" + "-" * 72)
    print("PRE-REGISTERED F1 at n=%d bracketed crossings (a/d=%s)"
          % (len(ok), [round(a, 2) for a in ok]))
    print("  S*_c  = %s" % np.array2string(np.array(Sc), precision=4))
    print("  eps_c = %s" % np.array2string(np.array(Ec), precision=3))
    if len(ok) >= 2:
        print("  CV(S*)=%.4f (max/min %.2fx)   CV(eps_c)=%.4f (max/min %.1fx)"
              % (cv_s, max(Sc) / min(Sc), cv_e, max(Ec) / min(Ec)))
        print("  dissociation ratio CV(eps_c)/CV(S*) = %.2f" % ratio)
    print("  >>> F1 = %s <<<" % verdict)
    if verdict == "KILL":
        print("  -> constant-S* onset collapse FALSIFIED (CV(S*)>%.2f or "
              ">=CV(eps)). The n=2 collapse was a coincidence." % CV_S_KILL)
    elif verdict == "CONFIRM":
        print("  -> constant-S* onset collapse CONFIRMED.")
    else:
        print("  -> DIRECTIONAL: report honestly, soften to 'approximate'.")

    # ---- THE CONTROLLED TEST: does S* or eps discriminate fail vs tol? -------
    print("\n" + "-" * 72)
    print("MATCHED-STEEPNESS CONTROLLED TEST (the new n>=4 leverage):")
    auc_s, D_s = auc(Sall, Fall)
    # eps fails LOW, so feed -eps so 'higher score => more fail'
    auc_e, D_e = auc(-Eall, Fall)
    print("  classify FAIL(relRMS>0.5) vs tolerate over %d wavy cases:" % len(rows))
    print("    discriminability D(S*)   = %.3f  (AUC=%.3f) -- 0.5=no skill" %
          (D_s, auc_s))
    print("    discriminability D(eps)  = %.3f  (AUC=%.3f, low eps => fail)" %
          (D_e, auc_e))
    # explicit matched-S* pairs: cases whose S* are within 5% but outcomes differ
    print("  matched-S* pairs (|dS*/S*|<0.06, OPPOSITE outcome):")
    npair = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if Fall[i] == Fall[j]:
                continue
            if abs(Sall[i] - Sall[j]) / (0.5 * (Sall[i] + Sall[j])) < 0.06:
                npair += 1
                fi = "FAIL" if Fall[i] else "tol"
                fj = "FAIL" if Fall[j] else "tol"
                print("    S*~%.3f : %-12s eps=%5.2f %-4s   vs   %-12s eps=%6.2f %-4s"
                      % (0.5 * (Sall[i] + Sall[j]), rows[i]["tag"], Eall[i], fi,
                         rows[j]["tag"], Eall[j], fj))
    if npair == 0:
        print("    (none within 6%; see overlap range below)")
    # overlap range of S* between fail and tol
    s_fail = Sall[Fall == 1]
    s_tol = Sall[Fall == 0]
    e_fail = Eall[Fall == 1]
    e_tol = Eall[Fall == 0]
    print("  S* overlap:  FAIL in [%.3f,%.3f]  TOL in [%.3f,%.3f]  overlap=%s"
          % (s_fail.min(), s_fail.max(), s_tol.min(), s_tol.max(),
             "YES (S* cannot separate)"
             if (s_tol.min() < s_fail.max()) else "no"))
    print("  eps overlap: FAIL in [%.3f,%.3f]  TOL in [%.3f,%.3f]"
          % (e_fail.min(), e_fail.max(), e_tol.min(), e_tol.max()))

    # ---- save ---------------------------------------------------------------
    out = os.path.join(RESULTS, "onset_steepness_falsification_l2.npz")
    np.savez(
        out,
        tag=np.array([r["tag"] for r in rows]),
        a_over_delta=Aall, lam_over_delta=Lall, slope=Sall, eps_med=Eall,
        relRMS=np.array([r["relRMS"] for r in rows]), r2=R2all, fail=Fall,
        cross_a=np.array(ok), Sc=np.array(Sc), Ec=np.array(Ec), Lc=np.array(Lc),
        cv_S=cv_s, cv_eps=cv_e, dissociation_ratio=ratio, f1_verdict=verdict,
        auc_slope=auc_s, D_slope=D_s, auc_eps=auc_e, D_eps=D_e,
        s_fail_range=np.array([s_fail.min(), s_fail.max()]),
        s_tol_range=np.array([s_tol.min(), s_tol.max()]),
        eps_fail_range=np.array([e_fail.min(), e_fail.max()]),
        eps_tol_range=np.array([e_tol.min(), e_tol.max()]),
        cv_s_confirm=CV_S_CONFIRM, ratio_confirm=RATIO_CONFIRM, cv_s_kill=CV_S_KILL,
        canonical_hill_r2=float(r2_hill), rib_r2=float(r2_rib), blade_md5=blade_md5,
        note=("L2 node_003: n>=4 (a/d x lam/d) wavy sweep. Pre-registered F1 "
              "constant-S* collapse evaluated at bracketed crossings + matched-"
              "steepness controlled test (D(S*) vs D(eps) classifying ODE "
              "catastrophe). RANS k-omegaSST; own wall-resolved LES at a/d=0.10 "
              "lam/d=3.0 in-flight anchors S*_c vs separation bias. No fabrication."),
    )
    print("\nSaved -> results/%s" % os.path.basename(out))

    # ---- figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

        # panel (a): (a/d x lam/d) map, color = R2 sign, marker = fail/tol
        sc = ax[0].scatter(Lall, Aall, c=np.clip(R2all, -2, 1), cmap="RdYlGn",
                           s=90, edgecolors="k",
                           marker="o", vmin=-2, vmax=1, zorder=3)
        for i in range(len(rows)):
            if Fall[i]:
                ax[0].scatter(Lall[i], Aall[i], s=190, facecolors="none",
                              edgecolors="red", linewidths=1.8, zorder=4)
        ax[0].set_xlabel(r"$\lambda/\delta$ (pitch)")
        ax[0].set_ylabel(r"$a/\delta$ (amplitude)")
        ax[0].set_title("(a) wavy (a/d x lam/d) map\nred ring = ODE catastrophe")
        cb = fig.colorbar(sc, ax=ax[0]); cb.set_label(r"$R^2(\tau_w)$ (clip $-2..1$)")

        # panel (b): the falsifier -- fail/tol vs S* (overlap) and vs eps (clean)
        jit = (np.arange(len(rows)) % 2) * 0.0
        ax[1].scatter(Sall[Fall == 1], np.ones(s_fail.size) * 1.0, c="red",
                      s=70, label="FAIL", zorder=3)
        ax[1].scatter(Sall[Fall == 0], np.ones(s_tol.size) * 0.0, c="green",
                      s=70, label="tolerate", zorder=3)
        ax[1].axvspan(max(s_tol.min(), s_fail.min()),
                      min(s_tol.max(), s_fail.max()),
                      color="gray", alpha=0.25, zorder=1,
                      label="S* overlap")
        ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["tol", "FAIL"])
        ax[1].set_xlabel(r"max wall slope $S^*=\pi a/\lambda$")
        ax[1].set_title("(b) S* does NOT separate\nD(S*)=%.2f" % D_s)
        ax[1].legend(fontsize=8, loc="center right")

        ax[2].scatter(Efail_x := Eall[Fall == 1], np.ones(s_fail.size) * 1.0,
                      c="red", s=70, zorder=3)
        ax[2].scatter(Etol_x := Eall[Fall == 0], np.ones(s_tol.size) * 0.0,
                      c="green", s=70, zorder=3)
        ax[2].set_xscale("log")
        ax[2].axvline(0.5, ls="--", c="k", lw=1)
        ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["tol", "FAIL"])
        ax[2].set_xlabel(r"cancellation depth $\varepsilon$ (median)")
        ax[2].set_title("(c) eps separates\nD(eps)=%.2f" % D_e)

        fig.suptitle("F1 = %s : constant-S* onset %s; eps governs failure "
                     "(CV(S*)=%.2f, CV(eps)=%.2f)"
                     % (verdict,
                        "FALSIFIED" if verdict == "KILL" else verdict.lower(),
                        cv_s, cv_e), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(FIGS, exist_ok=True)
        os.makedirs(NODE, exist_ok=True)
        for d in (FIGS, NODE):
            fig.savefig(os.path.join(d, "fig_onset_falsification.png"), dpi=130)
        print("Saved -> figures/fig_onset_falsification.png (+ node_003/)")
    except Exception as e:
        print("figure skipped:", e)

    print("=" * 80)


if __name__ == "__main__":
    main()
