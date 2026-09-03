#!/usr/bin/env python3
r"""
onset_steepness_methodology_l1.py   (Level-1 Core methodology -- node_001)
=========================================================================

ITERATION THESIS (L0, node_000, Judge YES 6/10 -> L1)
-----------------------------------------------------
The 8/10 champion established that the cancellation depth eps / deep-cancellation
coverage orders the SEVERITY of ODE-wall-model failure *among the cases that
already fail* (the Xiao 29-case periodic-hill family).  This iteration adds the
ORTHOGONAL axis the champion left open: what GEOMETRIC invariant locates the
failure-ONSET boundary R^2(tau_w)=0 in the (a/delta, lambda/delta) plane of the
smooth flat->wavy->hill family -- and *why eps is NOT that invariant*.

THE CLAIM (and its dissociation headline -- B-L1-4)
---------------------------------------------------
The flat->hill failure ONSET collapses onto a critical maximum surface STEEPNESS

      S* == max|dy_wall/dx|  ~  const

WHILE the onset cancellation-depth eps_c varies by ~30x along the very same
boundary.  Steepness sets the geometric ONSET (is the separated, O(delta)-pitch
recirculation imposed at all?); eps/coverage sets the SEVERITY once above it (the
champion's validated result).  The non-obvious contribution is the DISSOCIATION
of onset (geometry, steepness) from severity (flow, eps) -- two orthogonal
controls of one catastrophe -- NOT the truism "steeper walls separate earlier".

WHAT THIS L1 NODE DELIVERS (methodology only; the sweep is L2)
-------------------------------------------------------------
This script LOCKS the onset protocol and discharges the L0-Judge's five binds BY
COMPUTATION on the EXISTING on-disk npz.  It runs no new OpenFOAM.

  B-L1-1 (FATAL)  FIX the steepness formula.  The OpenFOAM wall is the
                  raised-cosine  y_w(x) = a*(1 - cos(2*pi*x/lambda))/2,  whose
                  maximum slope is  S* = max|dy_w/dx| = pi*a/lambda  -- NOT
                  2*pi*a/lambda.  We define steepness() = pi*a/lambda, reproduce
                  the on-disk crossing slopes [0.1051, 0.0949] bit-exact, and
                  HARD-ASSERT that the 2*pi form does NOT match the data.  Any
                  2*pi*a/lambda is a hard error here.
  B-L1-2          LOCK the n>=4 amplitude-sweep protocol: amplitude set, pitch
                  ladders straddling each crossing, the (verbatim, locked) R^2=0
                  / relRMS=0.5 crossing algorithm, and the pre-registered F1
                  collapse metric (CV of S* across amplitudes) with explicit
                  collapse / kill thresholds.
  B-L1-3 (CRIT)   RANS reliability.  The crossing is RANS(k-omegaSST)-located;
                  k-omegaSST delays separation.  We SCOPE the boundary as "the
                  RANS-predicted onset boundary", record the resolved-DNS anchor
                  (Maass & Schumann 1996 wavy DNS: separation physical; plus the
                  closure-independent rib LES), and register the own-LES anchor
                  AT the crossing pitch as an explicit L2/L3 deliverable.
  B-L1-4          LEAD with the dissociation: report CV(S*) vs CV(eps_c) and the
                  1.11x-vs-31x ratio as the headline; "steepness controls
                  separation onset" alone is a truism and is NOT the claim.
  B-L1-5 (FATAL)  Anti-empty: emit a real npz + reproduce the n=2 crossing.

THE LOCKED A-PRIORI PIPELINE
----------------------------
`evaluate(., Y_IDX=10)` and `spearman` are imported VERBATIM from
cross_geometry_collapse (the production a-priori ODE evaluator used by every
figure in the paper).  The crossing algorithm (interp_crossing, pitch_crossing,
score_case, discover_cases) is imported VERBATIM from onset_boundary_map_l3 (the
on-disk producer of onset_boundary_map.npz).  Nothing is re-implemented: this
node re-derives the boundary through the SAME frozen functions and proves the
result is bit-identical to the on-disk artifact, then layers the steepness
analysis + protocol lock on top.

OUTPUT
------
  codes/results/onset_steepness_methodology_l1.npz
"""
from __future__ import annotations

import os
import sys
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # codes/analysis
CODES = os.path.dirname(HERE)                                # codes/
RESULTS = os.path.join(CODES, "results")

# --- LOCK the production pipeline + the on-disk crossing algorithm -----------
sys.path.insert(0, HERE)
from cross_geometry_collapse import evaluate, Y_IDX, spearman          # noqa: E402
from onset_boundary_map_l3 import (                                    # noqa: E402
    discover_cases, score_case, pitch_crossing, interp_crossing,
    FAIL_RELRMS, DELTA, BETA_FLOOR,
)

assert Y_IDX == 10, "matching index drifted from the paper-wide standard"
assert FAIL_RELRMS == 0.5, "catastrophe screen drifted from paper-wide 0.5"
assert DELTA == 1.0, "delta convention drifted from channel half-height H/2"


# ===========================================================================
# B-L1-1 (FATAL): the steepness invariant, CORRECT for the raised-cosine wall.
# ===========================================================================
def steepness(a_over_delta, lam_over_delta):
    r"""Maximum wall slope S* = max|dy_w/dx| of the OpenFOAM raised-cosine wall
        y_w(x) = a*(1 - cos(2*pi*x/lambda))/2.
    dy_w/dx = a*(pi/lambda)*sin(2*pi*x/lambda)  ->  max|.| = pi*a/lambda.
    (a and lambda are both normalised by the SAME delta = H/2, so the ratio is
     dimensionless and delta cancels.)
    """
    return np.pi * np.asarray(a_over_delta, float) / np.asarray(lam_over_delta, float)


def steepness_WRONG_2pi(a_over_delta, lam_over_delta):
    """The L0 research-direction's ERRONEOUS form, kept ONLY to prove it does
    NOT match the data.  Never use this for any reported number."""
    return 2.0 * np.pi * np.asarray(a_over_delta, float) \
        / np.asarray(lam_over_delta, float)


# ===========================================================================
# Pre-registered F1 collapse / kill thresholds (B-L1-2).
# ===========================================================================
# Spread measure = coefficient of variation CV = std(ddof=1)/mean across the
# >=4 boundary crossings.  CV is scale-free so S* (~0.1) and eps_c (~O(1-10))
# are compared on the same footing.  We also report the robust max/min ratio.
CV_COLLAPSE_MAX = 0.15     # S* is "constant to engineering tolerance" iff CV<=0.15
DISSOC_RATIO_MIN = 5.0     # eps_c must be >=5x MORE variable than S* (dissociation)
CV_KILL_MAX = 0.30         # CV(S*)>0.30 => steepness is not a tight invariant
# F1 KILL (route to honest reframe / backtrack) iff CV(S*) >= CV(eps_c) (the L0
# pre-registration: "S* varies by MORE than eps_c") OR CV(S*) > CV_KILL_MAX.
# F1 CONFIRM iff CV(S*) <= CV_COLLAPSE_MAX AND CV(eps_c)/CV(S*) >= DISSOC_RATIO_MIN.
# Anything between = DIRECTIONAL: report honestly, do not over-claim "collapse".

# Pre-registered amplitude set + pitch ladders for the L2 sweep ----------------
# Crossing scales as lambda_c ~ (pi/S*)*a ~ 31.4*a (since S*~0.10), so each
# ladder is centred on its expected crossing and must BRACKET relRMS=0.5.
AMP_SET = [0.05, 0.10, 0.20, 0.40]                      # a/delta
PITCH_LADDERS = {                                       # lambda/delta per amplitude
    0.05: [1.0, 1.5, 2.0, 3.0, 4.0],                    # expected lam_c ~ 1.6
    0.10: [2.0, 3.0, 4.0, 5.0, 6.0],                    # expected lam_c ~ 3.1  (HAVE)
    0.20: [4.0, 5.0, 6.0, 7.0, 8.0, 10.0],              # expected lam_c ~ 6.3
    0.40: [8.0, 11.0, 14.0, 16.0, 18.0],                # expected lam_c ~ 12.6 (HAVE)
}
# Crossing convergence rule: a crossing is ACCEPTED only if the ladder contains
# >=1 case with relRMS>0.5 immediately adjacent (in lambda order) to a case with
# relRMS<0.5; the crossing is the linear interpolant (interp_crossing).  If no
# such bracket exists the ladder is refined (extra pitch) at L2 before a crossing
# is claimed -- never extrapolated.


def md5(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cv(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.mean(x)) if x.size >= 2 else np.nan


def main():
    print("=" * 80)
    print("ONSET-STEEPNESS METHODOLOGY (L1 node_001) -- protocol lock")
    print("  locked: evaluate(Y_IDX=%d) ; crossing algo from onset_boundary_map_l3"
          % Y_IDX)
    print("  steepness invariant  S* = pi*a/lambda  (raised-cosine wall)")
    print("=" * 80)

    # ---------------------------------------------------------------- guards --
    hill = os.path.join(RESULTS,
                        "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    r2_hill = evaluate(hill)["r2"]
    print("\n[guard 1] canonical periodic-hill R2 = %+.8f (headline -47.69)"
          % r2_hill)
    assert abs(r2_hill - (-47.68617253416459)) < 1e-6, "canonical hill R2 drifted!"
    blade_md5 = md5(os.path.join(RESULTS, "blade_severance_l3.npz"))
    print("[guard 2] blade_severance_l3.npz md5 =", blade_md5)
    assert blade_md5 == "60427e650592c2fdc0db301c228a273c", "blade npz drifted!"
    # closure-independent rib LES (sharp d-type) -- the third standing guard
    anch = np.load(os.path.join(RESULTS, "onset_resolved_anchor.npz"),
                   allow_pickle=True)
    rib_r2 = float(anch["rib_les_r2"])
    print("[guard 3] rib LES (closure-indep, sharp d-type) R2 = %+.8f" % rib_r2)
    assert abs(rib_r2 - (-0.9431719607410027)) < 1e-6, "rib LES R2 drifted!"

    # ======================================================================
    # B-L1-1 (FATAL): re-derive the n=2 boundary crossings THROUGH the locked
    # algorithm, compute S* = pi*a/lambda, and prove BOTH (a) bit-exact match to
    # the on-disk onset_boundary_map.npz, and (b) the 2*pi form does NOT match.
    # ======================================================================
    tags = discover_cases()
    rows = [r for r in (score_case(t) for t in tags) if r is not None]
    print("\n[recompute] scored %d wavy cases through the frozen pipeline" % len(rows))

    crossings = {}
    for a_t in AMP_SET:
        pc = pitch_crossing(rows, a_t)
        if pc is not None and np.isfinite(pc.get("lam_c", np.nan)):
            crossings[a_t] = pc

    a_cs = np.array(sorted(crossings.keys()))
    lam_cs = np.array([crossings[a]["lam_c"] for a in a_cs])
    eps_cs = np.array([crossings[a]["eps_c"] for a in a_cs])
    # steepness via the CORRECT formula, from the recomputed crossings
    S_cs = steepness(a_cs, lam_cs)
    S_cs_wrong = steepness_WRONG_2pi(a_cs, lam_cs)

    print("\n[B-L1-1] re-derived boundary crossings (locked relRMS=0.5 interp):")
    print("  %-6s %-12s %-12s %-12s %-12s" %
          ("a/d", "lam_c/d", "S*=pi*a/lam", "(WRONG 2pi)", "eps_c"))
    for a, lc, s, sw, e in zip(a_cs, lam_cs, S_cs, S_cs_wrong, eps_cs):
        print("  %-6.2f %-12.8f %-12.8f %-12.8f %-12.6f" % (a, lc, s, sw, e))

    # cross-check against the on-disk artifact -> BIT-EXACT
    obm = np.load(os.path.join(RESULTS, "onset_boundary_map.npz"),
                  allow_pickle=True)
    disk_a = np.asarray(obm["crossing_a"], float)
    disk_lam = np.asarray(obm["crossing_lam_c"], float)
    disk_slope = np.asarray(obm["crossing_slope_c"], float)
    disk_eps = np.asarray(obm["crossing_eps_c"], float)
    # align on amplitude
    order = np.array([int(np.argmin(np.abs(disk_a - a))) for a in a_cs])
    d_lam, d_slope, d_eps = disk_lam[order], disk_slope[order], disk_eps[order]
    err_lam = float(np.max(np.abs(lam_cs - d_lam)))
    err_slope = float(np.max(np.abs(S_cs - d_slope)))
    err_eps = float(np.max(np.abs(eps_cs - d_eps)))
    print("\n[B-L1-1 bit-exact] re-derived vs on-disk onset_boundary_map.npz:")
    print("  max|lam_c - disk|   = %.3e" % err_lam)
    print("  max|S* - disk slope|= %.3e   (disk slope = %s)"
          % (err_slope, np.array2string(d_slope, precision=8)))
    print("  max|eps_c - disk|   = %.3e" % err_eps)
    assert err_lam < 1e-9 and err_slope < 1e-9 and err_eps < 1e-9, \
        "re-derivation is NOT bit-exact with the on-disk boundary map!"

    # the FATAL assertion: pi*a/lambda matches; 2*pi*a/lambda does NOT.
    assert np.allclose(S_cs, d_slope, atol=1e-9), "pi form must match data"
    assert not np.allclose(S_cs_wrong, d_slope, atol=1e-3), \
        "2*pi form must NOT match -- B-L1-1 fix failed"
    print("  => CONFIRMED  S* = pi*a/lambda  (the 2*pi form is wrong by exactly 2x)")

    # ======================================================================
    # B-L1-4: the DISSOCIATION headline -- S* ~ const while eps_c varies ~30x.
    # ======================================================================
    cv_S = cv(S_cs)
    cv_eps = cv(eps_cs)
    ratio_S = float(S_cs.max() / S_cs.min())
    ratio_eps = float(eps_cs.max() / eps_cs.min())
    dissoc_ratio = float(cv_eps / cv_S) if cv_S > 0 else np.inf
    print("\n[B-L1-4 DISSOCIATION] (current n=%d; F1 demands n>=4 at L2)" % len(a_cs))
    print("  steepness S*:  CV = %.4f   max/min = %.3fx   <- the ONSET invariant"
          % (cv_S, ratio_S))
    print("  depth   eps_c: CV = %.4f   max/min = %.2fx   <- varies along boundary"
          % (cv_eps, ratio_eps))
    print("  dissociation ratio CV(eps_c)/CV(S*) = %.1f" % dissoc_ratio)

    # ======================================================================
    # B-L1-2: pre-registered F1 verdict logic (applied to current n=2; the
    # binding test is at L2 with n>=4).
    # ======================================================================
    f1_confirm = (cv_S <= CV_COLLAPSE_MAX) and (dissoc_ratio >= DISSOC_RATIO_MIN)
    f1_kill = (cv_S >= cv_eps) or (cv_S > CV_KILL_MAX)
    f1_verdict = ("KILL" if f1_kill else "CONFIRM" if f1_confirm else "DIRECTIONAL")
    print("\n[B-L1-2 F1 protocol] pre-registered thresholds:")
    print("  CONFIRM  iff CV(S*)<=%.2f AND CV(eps_c)/CV(S*)>=%.1f"
          % (CV_COLLAPSE_MAX, DISSOC_RATIO_MIN))
    print("  KILL     iff CV(S*)>=CV(eps_c)  OR  CV(S*)>%.2f" % CV_KILL_MAX)
    print("  amplitude set a/d = %s" % AMP_SET)
    for a in AMP_SET:
        lad = PITCH_LADDERS[a]
        have = a in crossings
        print("    a/d=%.2f : pitch ladder lambda/d=%s   expected lam_c~%.1f  %s"
              % (a, lad, np.pi * a / 0.10,
                 "[HAVE crossing]" if have else "[L2 TO RUN]"))
    print("  => current (n=%d) provisional F1 verdict: %s" % (len(a_cs), f1_verdict))

    # ======================================================================
    # B-L1-3 (CRIT): RANS reliability scope + resolved-DNS anchor plan.
    # ======================================================================
    print("\n[B-L1-3] RANS-onset scope + resolved-DNS anchor:")
    print("  crossing fidelity   : %s" % str(anch["crossing_fidelity"]))
    print("  Maass1996 wavy DNS  : a/lambda=%.2f -> a/d=%.1f, lam/d=%.1f ; "
          "separates(DNS)=%s ; overlaps crossing=%s"
          % (float(anch["maass_a_over_lambda"]),
             float(anch["maass_a_over_delta_mapped"]),
             float(anch["maass_lambda_over_delta_mapped"]),
             bool(anch["maass_separates_DNS"]),
             bool(anch["maass_overlaps_crossing"])))
    print("  => the boundary is reported as the RANS-PREDICTED onset boundary;")
    print("     Maass1996 DNS confirms separation is PHYSICAL at the failing")
    print("     steepness; the closure-independent rib LES (R2=%.3f) confirms a"
          % rib_r2)
    print("     real ODE failure in a sharp repeating geometry; an own wall-")
    print("     resolved LES AT the crossing pitch is the registered L2/L3")
    print("     anchor against k-omegaSST separation bias (not yet claimed).")
    rans_scope = ("RANS(k-omegaSST)-predicted onset boundary; separation physical "
                  "(Maass1996 DNS); own-LES at crossing = L2/L3 deliverable")

    # ----------------------------------------------------------------- save ---
    out = os.path.join(RESULTS, "onset_steepness_methodology_l1.npz")
    np.savez(
        out,
        protocol_y_idx=Y_IDX,
        fail_relRMS=FAIL_RELRMS,
        delta_convention="channel_half_height_H_over_2",
        steepness_formula="S_star = pi*a/lambda (raised cosine, max|dy/dx|)",
        # re-derived boundary crossings (bit-exact vs onset_boundary_map.npz)
        crossing_a=a_cs, crossing_lam_c=lam_cs, crossing_eps_c=eps_cs,
        crossing_S_star=S_cs, crossing_S_star_WRONG_2pi=S_cs_wrong,
        # bit-exact reproduction residuals
        reproduction_err_lam=err_lam, reproduction_err_slope=err_slope,
        reproduction_err_eps=err_eps,
        # dissociation headline (B-L1-4)
        cv_S_star=cv_S, cv_eps_c=cv_eps,
        ratio_S_star=ratio_S, ratio_eps_c=ratio_eps,
        dissociation_ratio=dissoc_ratio,
        # pre-registered F1 protocol (B-L1-2)
        cv_collapse_max=CV_COLLAPSE_MAX, dissoc_ratio_min=DISSOC_RATIO_MIN,
        cv_kill_max=CV_KILL_MAX, f1_verdict_provisional=f1_verdict,
        amp_set=np.array(AMP_SET),
        pitch_ladder_a05=np.array(PITCH_LADDERS[0.05]),
        pitch_ladder_a10=np.array(PITCH_LADDERS[0.10]),
        pitch_ladder_a20=np.array(PITCH_LADDERS[0.20]),
        pitch_ladder_a40=np.array(PITCH_LADDERS[0.40]),
        amplitudes_have_crossing=np.array(sorted(crossings.keys())),
        # B-L1-3 scope
        rans_scope=rans_scope,
        maass_separates_DNS=bool(anch["maass_separates_DNS"]),
        maass_overlaps_crossing=bool(anch["maass_overlaps_crossing"]),
        rib_les_r2=rib_r2,
        # guards
        canonical_hill_r2=float(r2_hill), blade_md5=blade_md5,
        note=("L1 onset-STEEPNESS methodology lock (node_001). Steepness invariant "
              "S*=pi*a/lambda (raised-cosine wall; the 2*pi form in the L0 "
              "research-direction was wrong by 2x and is HARD-disproven here). "
              "Boundary crossings re-derived bit-exact through the locked "
              "evaluate(Y_IDX=10) + onset_boundary_map_l3 crossing algorithm. "
              "DISSOCIATION headline: along the n=2 boundary S* CV=%.3f (ratio "
              "%.2fx) while eps_c CV=%.2f (ratio %.1fx) -> onset(geometry,S*) is "
              "orthogonal to severity(flow,eps), the champion's eps role. F1 "
              "pre-registered: CONFIRM if CV(S*)<=0.15 & CV(eps_c)/CV(S*)>=5; "
              "KILL if CV(S*)>=CV(eps_c). n>=4 amplitude sweep {0.05,0.1,0.2,0.4} "
              "is the L2 binding test. Boundary is RANS-predicted; separation "
              "physical (Maass1996); own-LES at crossing = L2/L3 anchor. No new "
              "sims; no fabrication." % (cv_S, ratio_S, cv_eps, ratio_eps)),
    )
    print("\nSaved -> results/%s" % os.path.basename(out))
    print("=" * 80)
    print("L1 STEEPNESS METHODOLOGY LOCK COMPLETE:")
    print("  B-L1-1 FATAL  steepness fixed to pi*a/lambda, bit-exact vs disk, "
          "2pi disproven")
    print("  B-L1-2        n>=4 sweep + crossing algo + F1 CV thresholds locked")
    print("  B-L1-3 CRIT   RANS scope + DNS/LES anchor plan recorded")
    print("  B-L1-4        dissociation headline (CV %.3f vs %.2f) reported"
          % (cv_S, cv_eps))
    print("  B-L1-5 FATAL  npz emitted; n=2 crossing reproduced")


if __name__ == "__main__":
    main()
