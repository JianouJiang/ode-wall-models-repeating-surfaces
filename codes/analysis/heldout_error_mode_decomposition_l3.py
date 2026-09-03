#!/usr/bin/env python3
r"""
heldout_error_mode_decomposition_l3.py  --  L3 (Results and analysis, node_008)
===============================================================================
RESULTS-LEVEL decomposition of the twenty held-out operating-map geometries into
DISTINCT error modes, discharging the L2 Judge's four L3 binds (node_007 verdict).

WHY THIS NODE EXISTS (the L2 Judge's L3 binds)
----------------------------------------------
The L2 held-out validation (node_007, rib_discriminant_heldout_l2.py / .npz) is
CORRECT science -- the epsilon<1 rule has zero false positives and reproduces the
kappa~1/eps mechanism out of sample -- but its narrative committed a verifiable
FACTUAL ERROR and a misdiagnosis:

  B-L3-1 (FATAL): the manuscript said "TWO flat walls" are among the four raw-R^2
    over-counts.  Only ONE is flat (of_flat_l22, a/delta=0.001).  The other three
    (op_a10_l22, op_a40_l14, op_a40_l16) are NOT flat (a/delta=0.10-0.40).
  B-L3-1 (cont.): the blanket "vanishing-variance R^2 degeneracy" diagnosis is
    WRONG for those three -- op_a40_l14/l16 have cv_tw~0.47, relRMS~0.55-0.64, i.e.
    GENUINE pointwise inaccuracy, not a low-variance artifact.
  B-L3-2 (CRIT): the held-out set is coarse steady RANS; this tests "eps predicts
    closure conditioning on RANS" -- the DNS-validated calibration set supplies the
    ground-truth failure labels.  State the provenance.
  B-L3-3 (CRIT): op_a40_l14/l16 (a/delta=0.4, wide pitch, f_sep=0, eps>>1) raise a
    legitimate question -- does eps MISS an amplitude-driven error mode?  Name it or
    attribute it.

This script answers all of them QUANTITATIVELY by decomposing the twenty geometries
into three physically distinct classes and measuring the instrument (the closure-
conditioning floor med kappa) that separates them:

  CLASS I  -- structural force-CANCELLATION failures (eps_med < 1).  n=4.  The eps
              rule flags exactly these.  med kappa >= 0.11 (ill-conditioned: the
              cancellation residual is small => the closure inverse is singular).
  CLASS II -- non-cancellation R^2 over-counts at eps_med >> 1.  n=4.  These are the
              raw-R^2<0 "misses".  med kappa <= 0.0035 (WELL-conditioned: an order of
              magnitude below Class I).  eps CORRECTLY does NOT flag them because the
              cancellation mechanism is NOT active.  They split into:
                (IIa) ONE flat wall  : of_flat_l22 (a/delta=0.001) -- model EXCELLENT
                      (relRMS=0.6%, cv_tw=0.001); R^2<0 is the genuine vanishing-
                      tau_w-variance degeneracy (the documented a->0 wavy_flat case).
                (IIb) THREE non-flat : op_a10_l22, op_a40_l14, op_a40_l16 -- a SECOND,
                      well-conditioned, amplitude-driven error mode (named below),
                      NOT the cancellation mechanism this paper diagnoses.
  CLASS III -- tolerated and accurate (eps_med > 1, R^2 >= 0).  n=12.

THE AMPLITUDE ERROR MODE (B-L3-3, named not hand-waved)
-------------------------------------------------------
Within the WELL-conditioned subset (med kappa < 0.05, i.e. eps>>1, no cancellation),
the relative wall-stress error rises with surface amplitude:
  Spearman(a/delta, relRMS | well-conditioned) ~ +0.82 (p<1e-3, n=13),
and at matched wide pitch (lambda/delta>=6) raising a/delta from 0.10 to 0.40 lifts
mean relRMS ~2.6x.  This is a SEPARATE, milder, well-conditioned amplitude effect:
the ODE makes a moderate pointwise phase/amplitude error over a steep crest-trough
even with no domain-wide cancellation.  It is NOT ill-conditioned (med kappa<=0.004
=> the residual is well defined) and NOT the O(1/eps) structural failure -- so the
eps<1 discriminant is CORRECT to leave it tolerated.  Consistent with the paper's
already-documented finding that eps is a BINARY catastrophic-vs-tolerable
discriminant, not a continuous accuracy meter.

THE SEPARATOR IS THE CONDITIONING FLOOR, NOT R^2 MAGNITUDE (the key L3 finding)
------------------------------------------------------------------------------
In this held-out RANS set the Class-I cancellation failures have R^2 in [-1.45,-0.16]
(mild -- they are RANS SHALLOW wavy walls; the catastrophic R^2 in [-84,-10] lives in
the DNS Xiao STEEP-hill calibration family, not here), and the Class-IIb amplitude
misses have R^2 in [-0.80,-0.33].  These R^2 ranges OVERLAP, so R^2 magnitude does
NOT separate the two error modes out of sample.  med kappa DOES: Class I >= 0.11,
Class II <= 0.0035 -- a ~32x gap with no overlap.  This is exactly why the
methodology reads the closure-conditioning floor rather than raw R^2.

NON-TAUTOLOGY / NO-REGRESSION
-----------------------------
This is a results-level RE-ANALYSIS of the byte-identical L2 npz (the instrument is
not changed).  The canonical anchors are re-asserted from their own results files
(hill closure-A R2=-47.68617253, d-type rib closure-A R2=-0.94317196), and the
protected blade_severance_l3.npz md5=60427e65... is logged (drift 0).  No fabrication;
RANS a-priori only; DNS symlinks untouched.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/heldout_error_mode_decomposition_l3.py
"""
import hashlib
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_008")
os.makedirs(NODE, exist_ok=True)

# canonical anchors (the upstream instrument must still reproduce these)
HILL_A_R2 = -47.68617253416459
RIB_A_R2 = -0.9431719607410027
BLADE_MD5 = "60427e650592c2fdc0db301c228a273c"

FLAT_TOL = 0.01          # a/delta below this == a nominally flat wall
WELLCOND = 0.05          # med kappa below this == well-conditioned (no cancellation)


def md5(path):
    if not os.path.exists(path):
        return "absent"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    print("=" * 78)
    print("heldout_error_mode_decomposition_l3 -- L3 node_008 (error-mode decomposition)")
    print("=" * 78)

    # ---- (0) guards: anchors + protected signature unchanged ----------------
    cg = np.load(os.path.join(RESULTS, "cross_geometry_conditioning_floor.npz"),
                 allow_pickle=True)
    hi = list(cg["tags"]).index("periodic_hill_1p0")
    hill_A = float(cg["rows"][hi]["A_ml_vandriest"]["r2"])
    cf = np.load(os.path.join(RESULTS, "rib_conditioning_floor.npz"), allow_pickle=True)
    rib_rows = {r["label"]: r for r in cf["les_rows"]}
    rib_A = float(rib_rows["ML van Driest"]["r2"])
    blade_md5 = md5(os.path.join(RESULTS, "blade_severance_l3.npz"))
    assert abs(hill_A - HILL_A_R2) < 1e-9, "hill anchor drift %.3e" % (hill_A - HILL_A_R2)
    assert abs(rib_A - RIB_A_R2) < 1e-6, "rib anchor drift %.3e" % (rib_A - RIB_A_R2)
    assert blade_md5 == BLADE_MD5, "blade md5 drift (regression): %s" % blade_md5
    print("[guard] hill closure-A R2 = %.8f (OK)" % hill_A)
    print("[guard] d-type rib closure-A R2 = %.8f (OK)" % rib_A)
    print("[guard] blade_severance_l3.npz md5 = %s (drift 0)" % blade_md5)

    # ---- (1) load the byte-identical held-out scores ------------------------
    d = np.load(os.path.join(RESULTS, "rib_discriminant_heldout_l2.npz"),
                allow_pickle=True)
    tags = np.array([str(t) for t in d["tags"]])
    a = d["a_over_delta"].astype(float)
    lam = d["lambda_over_delta"].astype(float)
    fsep = d["f_sep"].astype(float)
    eps = d["eps_med"].astype(float)
    mk = d["med_kappa"].astype(float)
    rr = d["relRMS"].astype(float)
    cv = d["cv_tw"].astype(float)
    r2 = d["r2_best"].astype(float)
    pf = d["pred_fail"].astype(bool)         # eps_med < 1 (theory-frozen)
    rf = d["r2_fail"].astype(bool)           # raw R^2_best < 0
    n = len(tags)
    assert n == 20, "expected 20 held-out geometries, got %d" % n

    # ---- (2) the three-class decomposition ----------------------------------
    miss = pf != rf                          # raw-R^2 disagreements with the eps rule
    classI = pf                              # structural cancellation failures (eps<1)
    classII = miss & (~pf)                   # non-cancellation R^2 over-counts (eps>>1)
    classIII = (~pf) & (~rf)                 # tolerated AND accurate
    # sanity: every geometry lands in exactly one class
    assert np.all(classI.astype(int) + classII.astype(int) + classIII.astype(int) == 1), \
        "classes are not a partition"

    flat = a < FLAT_TOL                      # nominally flat wall
    IIa = classII & flat                     # flat-wall vanishing-variance degeneracy
    IIb = classII & (~flat)                  # non-flat amplitude-driven inaccuracy

    print("\n[2] three-class decomposition of the twenty held-out geometries:")
    print("    CLASS I  (eps_med<1, structural cancellation failure): n=%d  %s"
          % (classI.sum(), list(tags[classI])))
    print("    CLASS II (eps_med>1, raw-R^2<0 over-count)           : n=%d  %s"
          % (classII.sum(), list(tags[classII])))
    print("       IIa flat-wall (a/delta<%.2f) vanishing-variance   : n=%d  %s"
          % (FLAT_TOL, IIa.sum(), list(tags[IIa])))
    print("       IIb non-flat amplitude-driven inaccuracy          : n=%d  %s"
          % (IIb.sum(), list(tags[IIb])))
    print("    CLASS III (eps_med>1, R^2>=0, tolerated+accurate)    : n=%d"
          % classIII.sum())

    # ---- (3) B-L3-1 FATAL: EXACTLY one flat wall, three non-flat ------------
    n_flat_in_II = int(IIa.sum())
    n_nonflat_in_II = int(IIb.sum())
    print("\n[3] B-L3-1 (the corrected count):")
    print("    flat walls among the four R^2 over-counts = %d  (%s)"
          % (n_flat_in_II, list(tags[IIa])))
    print("    non-flat R^2 over-counts                  = %d  (%s)"
          % (n_nonflat_in_II, list(tags[IIb])))
    for t in tags[IIa]:
        i = list(tags).index(t)
        print("      [IIa] %-12s a/d=%.3f relRMS=%.3f cv_tw=%.3f R2=%+.2f"
              " -> model EXCELLENT, R^2 degenerate (vanishing tau_w variance)"
              % (t, a[i], rr[i], cv[i], r2[i]))
    for t in tags[IIb]:
        i = list(tags).index(t)
        print("      [IIb] %-12s a/d=%.3f lam/d=%.0f relRMS=%.3f cv_tw=%.3f R2=%+.2f"
              " -> GENUINE inaccuracy, but well-conditioned (med_kappa=%.4f)"
              % (t, a[i], lam[i], rr[i], cv[i], r2[i], mk[i]))

    # ---- (4) the CONDITIONING FLOOR separates the two error modes -----------
    min_mk_I = float(mk[classI].min())       # least ill-conditioned cancellation fail
    max_mk_II = float(mk[classII].max())     # most ill-conditioned non-cancellation miss
    gap = min_mk_I / max_mk_II
    # do the R^2 ranges overlap? (they do -> R^2 magnitude is NOT a clean separator)
    r2_I = (float(r2[classI].min()), float(r2[classI].max()))
    r2_IIb = (float(r2[IIb].min()), float(r2[IIb].max()))
    r2_overlap = (r2_IIb[0] <= r2_I[1]) and (r2_I[0] <= r2_IIb[1])
    print("\n[4] the conditioning floor separates the two error modes (R^2 does NOT):")
    print("    CLASS I  med_kappa in [%.4f, %.4f]" % (min_mk_I, float(mk[classI].max())))
    print("    CLASS II med_kappa in [%.4f, %.4f]" % (float(mk[classII].min()), max_mk_II))
    print("    => conditioning gap (min_I / max_II) = %.1fx  (NO overlap)" % gap)
    print("    CLASS I  R^2_best in [%.2f, %.2f]  (RANS shallow wavy -- mild;"
          % r2_I)
    print("       catastrophic R^2<=-10 lives in the DNS Xiao steep-hill calibration set)")
    print("    CLASS IIb R^2_best in [%.2f, %.2f]" % r2_IIb)
    print("    => R^2 ranges OVERLAP (%s): R^2 magnitude is NOT a clean OOS separator;"
          % ("yes" if r2_overlap else "no"))
    print("       the closure-conditioning floor (med_kappa) is. This is exactly why")
    print("       the discriminant reads the conditioning floor, not raw R^2.")

    # ---- (5) B-L3-3: NAME the amplitude-driven error mode -------------------
    wc = mk < WELLCOND                       # well-conditioned subset (no cancellation)
    rho_a, p_a = spearmanr(a[wc], rr[wc])
    # matched wide pitch (lambda/delta >= 6): a/delta 0.10 vs 0.40 contrast
    a10 = np.array(["op_a10" in t for t in tags]) & (lam >= 6)
    a40 = np.array(["op_a40" in t for t in tags]) & (lam >= 6)
    mean_rr_a10 = float(rr[a10].mean())
    mean_rr_a40 = float(rr[a40].mean())
    amp_amp = mean_rr_a40 / mean_rr_a10
    mk_a10 = float(mk[a10].max())
    mk_a40 = float(mk[a40].max())
    print("\n[5] B-L3-3: the amplitude-driven error mode (named):")
    print("    Within the well-conditioned subset (med_kappa<%.2f, n=%d, eps>>1):"
          % (WELLCOND, int(wc.sum())))
    print("      Spearman(a/delta, relRMS) = %+.3f (p=%.2e) -- error rises with amplitude"
          % (rho_a, p_a))
    print("    At matched wide pitch lambda/delta>=6:")
    print("      op_a10 (a/d=0.10): mean relRMS=%.3f (max med_kappa=%.4f, well-conditioned)"
          % (mean_rr_a10, mk_a10))
    print("      op_a40 (a/d=0.40): mean relRMS=%.3f (max med_kappa=%.4f, well-conditioned)"
          % (mean_rr_a40, mk_a40))
    print("      => amplitude amplification = %.1fx, at ~constant (low) conditioning" % amp_amp)
    print("    DIAGNOSIS: a SECOND, milder, WELL-CONDITIONED amplitude error mode --")
    print("    pointwise phase/amplitude error over steep crests with NO domain-wide")
    print("    cancellation. NOT the O(1/eps) structural failure; eps<1 correctly leaves")
    print("    it tolerated (binary discriminant, not a continuous accuracy meter).")

    # ---- (6) B-L3-2: RANS provenance of the held-out set --------------------
    print("\n[6] B-L3-2: provenance of the held-out set:")
    print("    all 20 operating-map geometries are coarse steady RANS (no resolved")
    print("    <u'v'>). This test validates 'eps predicts the closure-CONDITIONING")
    print("    floor on RANS reference data' (oracle-free, a different fidelity).")
    print("    The DNS-validated 7-geometry calibration set supplies the GROUND-TRUTH")
    print("    catastrophic-failure labels. The two together are a strength: the rule")
    print("    is calibrated on DNS truth and survives on independent RANS geometries.")

    # ---- (7) save -----------------------------------------------------------
    summary = dict(
        method="L3 error-mode decomposition of the 20 held-out operating-map geometries",
        node="node_008", level=3,
        guards=dict(hill_closure_A_r2=hill_A, rib_closure_A_r2=rib_A, blade_md5=blade_md5),
        classes=dict(
            classI_cancellation_failures=dict(
                n=int(classI.sum()), tags=list(tags[classI]),
                med_kappa_min=min_mk_I, med_kappa_max=float(mk[classI].max()),
                r2_best_range=list(r2_I)),
            classII_noncancellation_overcounts=dict(
                n=int(classII.sum()), tags=list(tags[classII]),
                med_kappa_min=float(mk[classII].min()), med_kappa_max=max_mk_II,
                flat_walls=dict(n=n_flat_in_II, tags=list(tags[IIa])),
                nonflat_amplitude=dict(n=n_nonflat_in_II, tags=list(tags[IIb]),
                                       r2_best_range=list(r2_IIb))),
            classIII_tolerated_accurate=dict(n=int(classIII.sum()))),
        conditioning_gap=dict(min_medkappa_classI=min_mk_I, max_medkappa_classII=max_mk_II,
                              gap_factor=gap, r2_ranges_overlap=bool(r2_overlap)),
        amplitude_error_mode=dict(
            spearman_a_relRMS_wellcond=float(rho_a), p=float(p_a),
            n_wellcond=int(wc.sum()),
            mean_relRMS_a10_widepitch=mean_rr_a10,
            mean_relRMS_a40_widepitch=mean_rr_a40,
            amplification=amp_amp,
            max_medkappa_a10=mk_a10, max_medkappa_a40=mk_a40),
        provenance="20 held-out cases are coarse steady RANS (oracle-free); DNS-validated "
                   "7-geom calibration set supplies ground-truth catastrophic-failure labels",
    )
    with open(os.path.join(NODE, "error_mode_decomposition.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    np.savez(
        os.path.join(RESULTS, "heldout_error_mode_decomposition_l3.npz"),
        tags=tags, a_over_delta=a, lambda_over_delta=lam, f_sep=fsep,
        eps_med=eps, med_kappa=mk, relRMS=rr, cv_tw=cv, r2_best=r2,
        classI=classI, classII=classII, classIII=classIII, flat=flat, IIa=IIa, IIb=IIb,
        n_flat_in_II=n_flat_in_II, n_nonflat_in_II=n_nonflat_in_II,
        min_medkappa_classI=min_mk_I, max_medkappa_classII=max_mk_II, conditioning_gap=gap,
        r2_range_classI=np.array(r2_I), r2_range_classIIb=np.array(r2_IIb),
        r2_ranges_overlap=bool(r2_overlap),
        spearman_a_relRMS_wellcond=float(rho_a), p_a_relRMS=float(p_a), n_wellcond=int(wc.sum()),
        mean_relRMS_a10_widepitch=mean_rr_a10, mean_relRMS_a40_widepitch=mean_rr_a40,
        amplitude_amplification=amp_amp,
        hill_anchor_r2=hill_A, rib_anchor_r2=rib_A, blade_md5=blade_md5,
    )
    print("\nwrote node_008/error_mode_decomposition.json and "
          "codes/results/heldout_error_mode_decomposition_l3.npz")

    # ---- (8) headline assertions (the L3 claims that go to print) -----------
    assert n_flat_in_II == 1, "B-L3-1: expected EXACTLY one flat wall, got %d" % n_flat_in_II
    assert n_nonflat_in_II == 3, "B-L3-1: expected three non-flat misses, got %d" % n_nonflat_in_II
    assert "of_flat_l22" in tags[IIa], "B-L3-1: the flat wall must be of_flat_l22"
    assert set(tags[IIb]) == {"op_a10_l22", "op_a40_l14", "op_a40_l16"}, \
        "B-L3-1: non-flat misses must be op_a10_l22, op_a40_l14, op_a40_l16"
    assert gap > 10.0, "conditioning floor does not separate the modes (gap=%.1f)" % gap
    assert max_mk_II < min_mk_I, "a Class-II miss is as ill-conditioned as a Class-I fail"
    assert r2_overlap, "R^2 ranges should overlap (R^2 is NOT the clean separator)"
    assert rho_a > 0.6 and p_a < 0.01, \
        "amplitude error mode not significant (rho=%.3f, p=%.3g)" % (rho_a, p_a)
    assert amp_amp > 1.5, "amplitude amplification too weak (%.2f)" % amp_amp
    print("\nALL L3 decomposition assertions PASS:")
    print("  B-L3-1: ONE flat wall (of_flat_l22) + THREE non-flat misses (op_a10_l22,"
          " op_a40_l14, op_a40_l16)")
    print("  conditioning gap = %.1fx (Class I >= %.3f, Class II <= %.4f); R^2 ranges overlap"
          % (gap, min_mk_I, max_mk_II))
    print("  amplitude mode: Spearman(a/d, relRMS | well-cond) = %+.3f (p=%.2e), %.1fx at wide pitch"
          % (rho_a, p_a, amp_amp))


if __name__ == "__main__":
    main()
