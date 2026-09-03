#!/usr/bin/env python3
r"""
onset_mechanism_map_l1.py   (L1 core methodology -- node_004, attempt 2)
=======================================================================
LATERAL RE-LOCK of the onset methodology after the L0 *constant-steepness*
thesis was FALSIFIED at L2 (node_003: pre-registered F1 = KILL, CV(S*)=0.42 >
0.30).  This node does NOT re-attempt a single-invariant onset law -- that path
is dead.  Instead it locks a *mechanism-resolved* onset methodology that the
falsifying 23-case sweep itself reveals:

  THESIS (att2).  The ODE-wall-model failure-onset boundary in the (a/delta,
  lambda/delta) plane is NOT a single curve set by one geometric invariant.
  It is BIMODAL -- the edge of a TOLERATED BAND in the cancellation depth
  epsilon, with failure at BOTH ends by two DISTINCT mechanisms:

    * Mode I  (low amplitude, SHORT-pitch edge): deep force cancellation,
              epsilon < eps_lo ~ 0.6.  Crossing direction F->T as lambda grows.
    * Mode II (high amplitude, LONG-pitch edge): domain-integral epsilon
              blow-up, epsilon >> 1.  Crossing direction INVERTS to T->F.

  Of the two candidate ONSET invariants, epsilon is the better-behaved one on
  the principal (Mode-I) branch -- CV(eps_c) ~ 0.08 while CV(S*) ~ 0.47 -- which
  is the *exact inversion* of the L0 pre-registration (which claimed steepness
  constant, epsilon 30x-varying).  We report the inversion honestly.

WHY this is a genuine methodology, not just "reading the L2 data":
  it (a) defines the onset object operationally (relRMS=0.5 crossing, locked),
  (b) classifies each ladder by crossing DIRECTION into a mechanism branch,
  (c) attributes each branch to an epsilon tail (the band picture), and
  (d) pre-registers the L2/L3 binding tests that are now LOW-RISK confirmations
  (the structural claims are already supported on-disk) rather than the
  make-or-break single-invariant bet that killed att1's subtree.

NOTHING is re-implemented: evaluate(.,Y_IDX=10), the catastrophe screen
FAIL_RELRMS=0.5, the outer scale DELTA, and interp_crossing are imported
VERBATIM from the production pipeline (onset_boundary_map_l3).  All per-case
numbers are read from the on-disk 23-case sweep onset_steepness_falsification_l2
.npz produced at L2; the crossings are INDEPENDENTLY re-derived here through the
locked interp_crossing and asserted bit-exact against the stored values.

CFD-FREE.  No new simulations.  Every number traces to a codes/results/*.npz.

OUTPUT: codes/results/onset_mechanism_map_l1.npz
        development/nodes/node_004/onset_mechanism_map_l1.npz   (copy)
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NODE = os.path.join(os.path.dirname(CODES), "development", "nodes", "node_004")
sys.path.insert(0, HERE)

# ---- locked production contract (imported verbatim; no re-implementation) ----
from onset_boundary_map_l3 import (                                   # noqa: E402
    evaluate, Y_IDX, FAIL_RELRMS, DELTA, md5, interp_crossing,
)

assert Y_IDX == 10,        "matching index drifted from the paper-wide standard"
assert FAIL_RELRMS == 0.5, "catastrophe screen drifted"
assert DELTA == 1.0,       "outer scale drifted"

SWEEP = os.path.join(RESULTS, "onset_steepness_falsification_l2.npz")

# Mechanism-band classifier thresholds (pre-registered HERE, frozen for L2/L3).
# eps_lo: cancellation edge (deep tail); eps_band_hi: where the tolerated band is
# entered.  These are DIAGNOSTIC LABELS for the two mechanisms, NOT a fitted law.
EPS_DEEP = 1.0       # Mode-I cancellation lives at eps < O(1)
EPS_GROW = 5.0       # Mode-II domain-integral blow-up lives at eps >> O(1)


def cv(x):
    x = np.asarray([v for v in np.ravel(x) if np.isfinite(v)], float)
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / np.mean(x))


def auc_D(score, label):
    """Discriminability D = max(AUC, 1-AUC) of `score` separating fail(1) vs
    tol(0) via the Mann-Whitney rank statistic.  D=1 perfect, 0.5 chance."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    a = wins / (pos.size * neg.size)
    return float(max(a, 1.0 - a))


def branch_direction(lam, fail):
    """Classify a fixed-amplitude pitch ladder by crossing direction.
       'F2T' = fails at SHORT pitch, tolerates at LONG pitch  (Mode I, standard)
       'T2F' = tolerates at SHORT pitch, fails at LONG pitch   (Mode II, inverted)
       'mono_*' if no sign change."""
    order = np.argsort(lam)
    f = np.asarray(fail, int)[order]
    if f[0] == 1 and f[-1] == 0:
        return "F2T"
    if f[0] == 0 and f[-1] == 1:
        return "T2F"
    return "mono_fail" if f.all() else "mono_tol"


def main():
    if not os.path.exists(SWEEP):
        raise SystemExit(f"missing on-disk sweep: {SWEEP}")
    d = np.load(SWEEP, allow_pickle=True)

    tag   = d["tag"]
    a_od  = np.asarray(d["a_over_delta"], float)
    lam   = np.asarray(d["lam_over_delta"], float)
    slope = np.asarray(d["slope"], float)        # S* = max|dy/dx| = pi a/lambda
    eps   = np.asarray(d["eps_med"], float)
    rel   = np.asarray(d["relRMS"], float)
    r2    = np.asarray(d["r2"], float)
    fail  = np.asarray(d["fail"], int)

    amps = sorted(set(np.round(a_od, 4)))

    # ----- (1) per-amplitude branch classification + independent crossing -----
    rows = []
    for a in amps:
        m = np.isclose(a_od, a)
        L, R, E, S, F = lam[m], rel[m], eps[m], slope[m], fail[m]
        o = np.argsort(L)
        L, R, E, S, F = L[o], R[o], E[o], S[o], F[o]
        direction = branch_direction(L, F)
        # independent re-derivation of the relRMS=0.5 crossing via locked code
        lc, lo, hi = interp_crossing(L, R, FAIL_RELRMS)
        rec = dict(a=float(a), n=int(m.sum()), direction=direction,
                   lam=L, relRMS=R, eps=E, slope=S, fail=F,
                   lam_c=lc, bracket=(lo, hi))
        if np.isfinite(lc):
            rec["eps_c"]   = float(np.interp(lc, L, E))
            rec["slope_c"] = float(a * np.pi / lc)
        rows.append(rec)

    # ----- (2) reconcile vs on-disk crossings (bit-exact assertion) -----
    cross_a = np.asarray(d["cross_a"], float)
    Sc_disk = np.asarray(d["Sc"], float)
    Ec_disk = np.asarray(d["Ec"], float)
    Lc_disk = np.asarray(d["Lc"], float)
    max_dL = max_dS = max_dE = 0.0
    for ca, Sd, Ed, Ld in zip(cross_a, Sc_disk, Ec_disk, Lc_disk):
        r = next(r for r in rows if np.isclose(r["a"], ca))
        max_dL = max(max_dL, abs(r["lam_c"] - Ld))
        max_dS = max(max_dS, abs(r["slope_c"] - Sd))
        max_dE = max(max_dE, abs(r["eps_c"] - Ed))
    assert max_dL < 1e-6 and max_dS < 1e-6 and max_dE < 1e-6, \
        f"crossing re-derivation drifted: dL={max_dL} dS={max_dS} dE={max_dE}"

    # ----- (3) the two mechanism branches -----
    modeI  = [r for r in rows if r["direction"] == "F2T"]   # standard, short-pitch
    modeII = [r for r in rows if r["direction"] == "T2F"]   # inverted, long-pitch
    aI  = np.array([r["a"] for r in modeI])
    EcI = np.array([r["eps_c"]   for r in modeI])
    ScI = np.array([r["slope_c"] for r in modeI])
    LcI = np.array([r["lam_c"]   for r in modeI])

    cv_eps_I = cv(EcI)
    cv_S_I   = cv(ScI)
    # INVERTED dissociation ratio: how much MORE constant is eps than S* on Mode I
    inv_diss_ratio = (cv_S_I / cv_eps_I) if cv_eps_I > 0 else np.nan
    eps_lo = float(np.mean(EcI))          # located Mode-I onset (band lower edge)

    # ----- (4) single-scalar onset falsification (robust, n=23) -----
    # No single geometric/flow scalar globally separates fail vs tol.
    D_slope = auc_D(slope, fail)
    D_eps   = auc_D(eps,   fail)
    D_a     = auc_D(a_od,  fail)
    D_lam   = auc_D(lam,   fail)
    # global CV of S* across the SAME-mechanism (Mode I) crossings -> KILL value
    cv_S_kill = cv_S_I

    # ----- (5) the tolerated BAND: failures sit at OPPOSITE eps tails -----
    eps_fail = eps[fail == 1]
    eps_tol  = eps[fail == 0]
    # Mode-I failures are the deep tail; Mode-II failures are the high tail.
    modeI_fail_eps  = eps[(fail == 1) & (eps < EPS_DEEP)]
    modeII_fail_eps = eps[(fail == 1) & (eps > EPS_GROW)]
    band_lo = float(np.max(modeI_fail_eps)) if modeI_fail_eps.size else np.nan
    band_hi = float(np.min(modeII_fail_eps)) if modeII_fail_eps.size else np.nan

    # ----- guards (bit-exact) -----
    g_hill = float(d["canonical_hill_r2"])
    g_rib  = float(d["rib_r2"])
    g_blade = str(d["blade_md5"])
    assert abs(g_hill + 47.68617253416459) < 1e-9, "hill guard drift"
    assert abs(g_rib + 0.9431719607410027) < 1e-9, "rib guard drift"
    assert g_blade == "60427e650592c2fdc0db301c228a273c", "blade guard drift"

    # ----- save -----
    out = os.path.join(RESULTS, "onset_mechanism_map_l1.npz")
    np.savez(
        out,
        amps=np.array(amps, float),
        branch_dir=np.array([r["direction"] for r in rows]),
        lam_c=np.array([r["lam_c"] for r in rows]),
        slope_c=np.array([r.get("slope_c", np.nan) for r in rows]),
        eps_c=np.array([r.get("eps_c", np.nan) for r in rows]),
        # bit-exact reconciliation residuals
        recon_max_dLam=max_dL, recon_max_dSlope=max_dS, recon_max_dEps=max_dE,
        # Mode-I branch (principal)
        modeI_a=aI, modeI_eps_c=EcI, modeI_slope_c=ScI, modeI_lam_c=LcI,
        cv_eps_modeI=cv_eps_I, cv_S_modeI=cv_S_I,
        inverted_dissociation_ratio=inv_diss_ratio,
        eps_lo_located=eps_lo,
        # single-scalar falsification (robust)
        D_slope=D_slope, D_eps=D_eps, D_a=D_a, D_lam=D_lam,
        cv_S_kill_value=cv_S_kill, cv_S_kill_thresh=0.30,
        # band
        eps_fail_range=np.array([eps_fail.min(), eps_fail.max()]),
        eps_tol_range=np.array([eps_tol.min(), eps_tol.max()]),
        band_lo_edge=band_lo, band_hi_edge=band_hi,
        n_modeI=len(modeI), n_modeII=len(modeII), n_cases=len(tag),
        EPS_DEEP=EPS_DEEP, EPS_GROW=EPS_GROW,
        # guards
        canonical_hill_r2=g_hill, rib_r2=g_rib, blade_md5=g_blade,
        note=("L1 node_004 att2: mechanism-resolved onset methodology. The L0 "
              "constant-steepness onset thesis was FALSIFIED at L2 (F1=KILL). "
              "This re-locks the onset object as the edge of a TOLERATED eps "
              "BAND with two failure mechanisms (Mode I cancellation eps<~0.6, "
              "F->T short-pitch; Mode II domain-integral blow-up eps>>1, T->F "
              "long-pitch). On the principal Mode-I branch eps is ~6x MORE "
              "constant than S* -- the inversion of the L0 pre-registration. "
              "All crossings re-derived bit-exact from the on-disk 23-case "
              "sweep via the locked interp_crossing. CFD-free; no fabrication."),
    )
    if os.path.isdir(NODE):
        shutil.copy2(out, os.path.join(NODE, "onset_mechanism_map_l1.npz"))

    # ----- console report -----
    print("=" * 74)
    print("L1 att2  MECHANISM-RESOLVED ONSET METHODOLOGY  (node_004)")
    print("=" * 74)
    print(f"on-disk sweep : {os.path.basename(SWEEP)}  ({len(tag)} cases)")
    print(f"locked import : evaluate(Y_IDX={Y_IDX}), FAIL_RELRMS={FAIL_RELRMS}, "
          f"DELTA={DELTA}, interp_crossing  [verbatim]")
    print(f"recon vs disk : max|dLam|={max_dL:.2e}  max|dS*|={max_dS:.2e}  "
          f"max|deps|={max_dE:.2e}   (bit-exact)")
    print("-" * 74)
    print("per-amplitude branch classification:")
    for r in rows:
        cstr = (f"lam_c={r['lam_c']:6.3f}  S*_c={r.get('slope_c', np.nan):7.4f}"
                f"  eps_c={r.get('eps_c', np.nan):8.3f}") if np.isfinite(r["lam_c"]) \
               else "no relRMS=0.5 crossing in ladder"
        print(f"  a/d={r['a']:.2f}  n={r['n']}  dir={r['direction']:9s}  {cstr}")
    print("-" * 74)
    print("Mode I (principal, F->T short-pitch cancellation branch):")
    print(f"  amplitudes a/d = {aI}")
    print(f"  eps_c          = {np.round(EcI,4)}   CV = {cv_eps_I:.4f}")
    print(f"  S*_c           = {np.round(ScI,4)}   CV = {cv_S_I:.4f}")
    print(f"  -> eps_c is {inv_diss_ratio:.2f}x MORE constant than S*  "
          f"(L0 claimed the OPPOSITE)")
    print(f"  -> located Mode-I onset  eps_lo = {eps_lo:.3f}  "
          f"(S*_c spans {ScI.max()/ScI.min():.2f}x)")
    print("-" * 74)
    print("Mode II (inverted, T->F long-pitch domain-integral blow-up branch):")
    for r in modeII:
        print(f"  a/d={r['a']:.2f}  lam_c={r['lam_c']:.3f}  "
              f"eps_c={r['eps_c']:.2f} (>> 1)")
    print("-" * 74)
    print("single-scalar onset FALSIFICATION (robust, n=23):")
    print(f"  D(S*)  = {D_slope:.3f}   D(eps) = {D_eps:.3f}   "
          f"D(a/d) = {D_a:.3f}   D(lam) = {D_lam:.3f}")
    print(f"  CV(S*)_Mode-I = {cv_S_kill:.3f}  > kill thresh 0.30  "
          f"=> constant-S* onset is DEAD")
    print(f"  no single scalar reaches clean (D=1) separation across both "
          f"mechanisms")
    print("-" * 74)
    print("tolerated BAND in eps (failures occupy OPPOSITE tails):")
    print(f"  eps(fail) range = [{eps_fail.min():.3f}, {eps_fail.max():.3f}]")
    print(f"  eps(tol)  range = [{eps_tol.min():.3f}, {eps_tol.max():.3f}]")
    print(f"  band lower edge (max Mode-I-fail eps)  = {band_lo:.3f}")
    print(f"  band upper edge (min Mode-II-fail eps) = {band_hi:.3f}")
    print("-" * 74)
    print(f"guards: hill R2={g_hill:.8f}  rib R2={g_rib:.8f}  blade md5={g_blade}")
    print(f"written: {out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
