#!/usr/bin/env python3
r"""
onset_steepness_sweep_l2.py   (Level-2 implementation & experiments -- node_002)
==============================================================================
Evaluate the PRE-REGISTERED F1 steepness-collapse verdict at n>=4 boundary
crossings, using the COMPLETED a/delta in {0.05, 0.20} pitch ladders that this
node ran in OpenFOAM (run_onset_ladder.sh onset_a05.txt / onset_a20.txt) added
to the on-disk a/delta in {0.10, 0.40} crossings.

NOTHING is re-implemented.  The a-priori ODE evaluator evaluate(.,Y_IDX=10), the
coverage metrics, and the boundary-crossing algorithm
(discover_cases / score_case / pitch_crossing / interp_crossing) are imported
VERBATIM from onset_boundary_map_l3 (the on-disk producer of
onset_boundary_map.npz).  The F1 thresholds were FROZEN at L1 (node_001,
methodology.md) BEFORE these runs and are NOT tuned here.

PRE-REGISTERED F1 (from node_001/methodology.md, copied verbatim):
  spread measure : CV = std(ddof=1)/mean across the >=4 boundary crossings
  CONFIRM   iff  CV(S*) <= 0.15  AND  CV(eps_c)/CV(S*) >= 5
  KILL      iff  CV(S*) >= CV(eps_c)   OR  CV(S*) > 0.30
  otherwise DIRECTIONAL (report honestly, do not claim "collapse")

Each crossing is accepted only if its ladder BRACKETS relRMS=0.5 (a case above
and an adjacent case below); lambda_c is interp_crossing (never extrapolated).

OUTPUT: codes/results/onset_steepness_sweep_l2.npz
"""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
sys.path.insert(0, HERE)

# verbatim locked pipeline (no re-implementation)
from onset_boundary_map_l3 import (                                   # noqa: E402
    evaluate, Y_IDX, FAIL_RELRMS, DELTA, md5,
    discover_cases, score_case, pitch_crossing, interp_crossing,
)

assert Y_IDX == 10, "matching index drifted from the paper-wide standard"
assert FAIL_RELRMS == 0.5, "catastrophe screen drifted"
assert DELTA == 1.0, "outer scale drifted"

# pre-registered F1 thresholds (FROZEN at L1, not tunable here)
CV_S_CONFIRM = 0.15
RATIO_CONFIRM = 5.0
CV_S_KILL = 0.30

# the four amplitudes whose pitch ladders define the onset boundary
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


def main():
    print("=" * 80)
    print("ONSET-STEEPNESS SWEEP (L2 node_002) -- F1 verdict at n>=4 crossings")
    print("  locked: evaluate(Y_IDX=%d), crossing algo from onset_boundary_map_l3"
          % Y_IDX)
    print("  pre-registered (L1): CONFIRM iff CV(S*)<=%.2f & ratio>=%.0f ; "
          "KILL iff CV(S*)>=CV(eps) or CV(S*)>%.2f"
          % (CV_S_CONFIRM, RATIO_CONFIRM, CV_S_KILL))
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
    if os.path.exists(rib):
        r2_rib = evaluate(rib)["r2"]
        print("[guard 3] rib LES (sharp d-type) R2 = %.8f" % r2_rib)
        assert abs(r2_rib - (-0.94317196)) < 1e-6, "rib LES R2 drifted!"

    # ---- score every harvested wavy case ------------------------------------
    tags = discover_cases()
    rows = [r for r in (score_case(t) for t in tags) if r is not None]
    print("\nscored %d wavy cases" % len(rows))

    # ---- per-amplitude pitch crossings (the boundary curve) -----------------
    print("\n%-6s %5s %9s %10s %10s %10s %9s %s" %
          ("a/d", "n", "lam_c/d", "bracket_lo", "bracket_hi", "S*=pi a/lam",
           "eps_c", "bracketed?"))
    cross = {}
    for a_t in A_TARGETS:
        pc = pitch_crossing(rows, a_t)
        if pc is None:
            print("  a/d=%.2f : <2 ladder points -- NOT RUN" % a_t)
            continue
        lc = pc.get("lam_c", np.nan)
        lo, hi = pc["bracket"]
        bracketed = (np.isfinite(lc) and np.isfinite(lo) and np.isfinite(hi)
                     and lo < lc < hi)
        cross[a_t] = dict(pc, bracketed=bool(bracketed))
        print("%-6.2f %5d %9.4f %10s %10s %10s %9s   %s"
              % (a_t, pc["n"], lc,
                 ("%.3f" % lo) if np.isfinite(lo) else "  --  ",
                 ("%.3f" % hi) if np.isfinite(hi) else "  --  ",
                 ("%.5f" % pc.get("slope_c", np.nan)) if np.isfinite(lc) else "  --  ",
                 ("%.4f" % pc.get("eps_c", np.nan)) if np.isfinite(lc) else "  --  ",
                 "YES" if bracketed else "NO (refine ladder)"))
        # show the ladder so a referee sees the bracket
        order = np.argsort(pc["lam"])
        lad = "  ".join("%.2f:%.3f" % (pc["lam"][i], pc["relRMS"][i])
                        for i in order)
        print("        ladder lam:relRMS = %s" % lad)

    # ---- F1 verdict at n>=4 -------------------------------------------------
    ok = [a for a in A_TARGETS if a in cross and cross[a]["bracketed"]]
    S = [cross[a]["slope_c"] for a in ok]
    E = [cross[a]["eps_c"] for a in ok]
    L = [cross[a]["lam_c"] for a in ok]
    cv_s, cv_e = cv(S), cv(E)
    verdict, ratio = f1_verdict(cv_s, cv_e)

    print("\n" + "-" * 72)
    print("F1 VERDICT at n=%d bracketed crossings  (a/d=%s)"
          % (len(ok), [round(a, 2) for a in ok]))
    print("  S*  per crossing : %s" % np.array2string(np.array(S), precision=5))
    print("  eps_c per crossing: %s" % np.array2string(np.array(E), precision=4))
    print("  lam_c per crossing: %s" % np.array2string(np.array(L), precision=3))
    if len(ok) >= 2:
        print("  CV(S*)            = %.4f   (max/min = %.3fx)"
              % (cv_s, max(S) / min(S)))
        print("  CV(eps_c)         = %.4f   (max/min = %.2fx)"
              % (cv_e, max(E) / min(E)))
        print("  dissociation ratio CV(eps_c)/CV(S*) = %.2f" % ratio)
    print("\n  >>> F1 = %s <<<" % verdict)
    if verdict == "CONFIRM":
        print("  steepness collapse CONFIRMED at n=%d: onset boundary is a curve "
              "of ~constant max wall slope S*; eps dissociates (severity, not "
              "onset)." % len(ok))
    elif verdict == "KILL":
        print("  steepness collapse KILLED: backtrack to L0, honest reframe.")
    else:
        print("  DIRECTIONAL: report honestly, soften to 'approximate' collapse.")

    # ---- B-L2-4 small-amplitude limit --------------------------------------
    print("\n" + "-" * 72)
    print("B-L2-4 small-amplitude limit (a/d=0.05):")
    if 0.05 in cross and cross[0.05]["bracketed"]:
        s05 = cross[0.05]["slope_c"]
        others = [cross[a]["slope_c"] for a in ok if a != 0.05]
        if others:
            dev = abs(s05 - np.mean(others)) / np.mean(others)
            print("  S*(a/d=0.05)=%.5f vs mean(S* a/d>=0.10)=%.5f -> deviation %.1f%%"
                  % (s05, np.mean(others), 100 * dev))
            print("  %s" % ("within collapse (no viscous/Re anomaly)" if dev < 0.15
                            else "DEVIATES -> report small-a viscous-damping caveat"))
    else:
        print("  a/d=0.05 crossing not yet bracketed (ladder still running).")

    # ---- save ---------------------------------------------------------------
    out = os.path.join(RESULTS, "onset_steepness_sweep_l2.npz")
    np.savez(
        out,
        a_targets=np.array(A_TARGETS),
        bracketed=np.array([cross.get(a, {}).get("bracketed", False)
                            for a in A_TARGETS]),
        lam_c=np.array([cross.get(a, {}).get("lam_c", np.nan) for a in A_TARGETS]),
        slope_c=np.array([cross.get(a, {}).get("slope_c", np.nan) for a in A_TARGETS]),
        eps_c=np.array([cross.get(a, {}).get("eps_c", np.nan) for a in A_TARGETS]),
        bracket_lo=np.array([cross.get(a, {}).get("bracket", (np.nan, np.nan))[0]
                             for a in A_TARGETS]),
        bracket_hi=np.array([cross.get(a, {}).get("bracket", (np.nan, np.nan))[1]
                             for a in A_TARGETS]),
        ok_a=np.array(ok), S_ok=np.array(S), E_ok=np.array(E), L_ok=np.array(L),
        cv_S=cv_s, cv_eps=cv_e, dissociation_ratio=ratio,
        f1_verdict=verdict,
        cv_s_confirm=CV_S_CONFIRM, ratio_confirm=RATIO_CONFIRM, cv_s_kill=CV_S_KILL,
        canonical_hill_r2=float(r2_hill), blade_md5=blade_md5,
        note=("L2 onset-steepness sweep: PRE-REGISTERED F1 evaluated at n=%d "
              "bracketed boundary crossings (a/d in %s). Two NEW OpenFOAM pitch "
              "ladders a/d in {0.05,0.20} (k-omegaSST steady RANS, run_onset_ladder.sh) "
              "added to the on-disk {0.10,0.40} crossings. RANS-predicted onset; "
              "own wall-resolved WALE LES at a/d=0.10,lam/d=3.0 anchors S*_c vs "
              "k-omegaSST separation bias (B-L2-3, in-flight). No fabrication; "
              "only bracketed crossings enter the verdict." % (len(ok), ok)),
    )
    print("\nSaved -> results/%s" % os.path.basename(out))
    print("=" * 80)


if __name__ == "__main__":
    main()
