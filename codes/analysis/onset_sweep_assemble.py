#!/usr/bin/env python3
r"""
onset_sweep_assemble.py   (Level-2 implementation -- node_003)
==============================================================

Execute the onset-boundary program the L1 methodology
(onset_boundary_methodology.py) LOCKED:  score every wavy-wall case in the
amplitude x pitch sweep with the FROZEN a-priori pipeline, emit the per-case
onset schema, and LOCATE the R^2 = 0 (relRMS = 0.5) failure-onset crossing in
the pitch ladder -- turning the champion's *bracket* into a *measured* boundary.

Nothing here re-implements the model or the metric:  `evaluate(., Y_IDX=10)` is
imported VERBATIM from cross_geometry_collapse (the production a-priori ODE used
by every other figure), and the coverage descriptors (f_rec, L_sep/delta) come
from coverage_metrics in onset_boundary_methodology (a byte-for-byte copy of the
validated dose_response_xiao span function).  This script only *applies* them to
the new geometry and reads off the crossing.

It scores WHATEVER cases are already harvested to
codes/results/<tag>_wall_profiles.npz, so it runs incrementally while the RANS
sweep is still in flight (anti-empty B-L2-4) and again at the end for the full
boundary.

Per-case onset schema (B-L1-1):
  case_tag, a_over_delta, ell_p_over_delta, fidelity,
  L_sep, L_sep_over_delta, f_rec, f_sep,
  eps_med, frac_eps_lt0p1, relRMS, r2, n_stations

OUTPUT:  codes/results/onset_sweep_results.npz
"""
from __future__ import annotations

import os
import sys
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))     # codes/analysis
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")

sys.path.insert(0, HERE)
from cross_geometry_collapse import evaluate, Y_IDX, spearman       # noqa: E402
from onset_boundary_methodology import coverage_metrics              # noqa: E402

assert Y_IDX == 10, "matching index drifted from the paper-wide standard"

FAIL_RELRMS = 0.5      # catastrophe screen (artefact-robust; same as paper-wide)
DELTA = 1.0            # channel half-height convention (B-L1-3), x already in H/2


# ---------------------------------------------------------------------------
# The wavy sweep:  (tag, role).  a/delta, lambda/delta read from the npz itself.
#   pitch  : a/delta ~ 0.10 fixed, lambda/delta varies   (locate crossing)
#   amp    : lambda/delta = 2 fixed, a/delta varies       (boundary is a corner)
#   flat   : a -> 0 control                               (R2 denominator artefact)
# Pre-existing on-disk anchors (wavy_a10 = pitch&amp node at lam/d=2; wavy_flat)
# are included automatically.  New ladder cases appear once harvested.
# ---------------------------------------------------------------------------
EXPECTED = [
    # pre-existing
    ("wavy_a10",     "both"),     # a/d=0.10 lam/d=2  (failing anchor, R2=-0.34)
    ("wavy_flat",    "flat"),     # a->0   lam/d=2
    # pitch ladder (a/d=0.10)
    ("op_a10_l03",   "pitch"), ("op_a10_l04", "pitch"), ("op_a10_l05", "pitch"),
    ("op_a10_l06",   "pitch"), ("op_a10_l08", "pitch"), ("op_a10_l11", "pitch"),
    ("op_a10_l14",   "pitch"), ("op_a10_l16", "pitch"), ("op_a10_l18", "pitch"),
    ("op_a10_l22",   "pitch"),
    # amplitude ladder (lam/d=2)
    ("oa_a05_l02",   "amp"), ("oa_a15_l02", "amp"),
    ("oa_a20_l02",   "amp"), ("oa_a30_l02", "amp"),
    # flat controls at wide pitch
    ("of_flat_l08",  "flat"), ("of_flat_l22", "flat"),
]


def md5(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def score_case(tag, role):
    """Return the onset-schema dict for one wavy case, or None if not yet on disk."""
    path = os.path.join(RESULTS, "%s_wall_profiles.npz" % tag)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    ev = evaluate(path)
    cv = coverage_metrics(path, DELTA)
    a_over = float(d["a_over_delta"]) if "a_over_delta" in d else np.nan
    return dict(
        case_tag=tag, role=role,
        a_over_delta=a_over,
        ell_p_over_delta=cv["ell_p_over_delta"],
        fidelity="RANS-komegaSST",
        L_sep=cv["L_sep"], L_sep_over_delta=cv["L_sep_over_delta"],
        f_rec=cv["f_rec"], f_sep=ev["f_sep"],
        eps_med=ev["eps_med"], frac_eps_lt0p1=ev["frac_eps_lt0p1"],
        relRMS=ev["relRMS"], r2=ev["r2"], n_stations=ev["n"],
        provenance=str(d["provenance"]) if "provenance" in d else "RANS",
        verdict=("FAIL" if ev["relRMS"] > FAIL_RELRMS else "tolerate"),
    )


def interp_crossing(x, val, target):
    """Linearly interpolate the x at which monotone-ish val(x) crosses target.
    Returns (x_cross, bracket_lo, bracket_hi) using the first sign change of
    (val-target) along x-sorted points; NaN if no crossing bracketed."""
    order = np.argsort(x)
    xs, vs = np.asarray(x)[order], np.asarray(val)[order]
    g = vs - target
    for i in range(1, len(xs)):
        if np.isfinite(g[i - 1]) and np.isfinite(g[i]) and g[i - 1] * g[i] <= 0 \
                and g[i - 1] != g[i]:
            t = -g[i - 1] / (g[i] - g[i - 1])
            xc = xs[i - 1] + t * (xs[i] - xs[i - 1])
            return float(xc), float(xs[i - 1]), float(xs[i])
    return np.nan, np.nan, np.nan


def main():
    print("=" * 80)
    print("ONSET SWEEP ASSEMBLE (L2 node_003)  --  locate the ODE-failure onset")
    print("  frozen pipeline: evaluate(., Y_IDX=%d) ; delta = H/2 = %.1f ; "
          "catastrophe relRMS>%.1f" % (Y_IDX, DELTA, FAIL_RELRMS))
    print("=" * 80)

    # -- regression guard: locked evaluate reproduces the canonical hill --------
    hill = os.path.join(RESULTS,
                        "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    r2_hill = evaluate(hill)["r2"]
    print("[guard] canonical periodic hill R2 = %+.2f (paper headline -47.7)"
          % r2_hill)
    assert -49.0 < r2_hill < -46.0, "canonical hill R2 drifted!"
    blade_md5 = md5(os.path.join(RESULTS, "blade_severance_l3.npz"))
    print("[guard] blade_severance_l3.npz md5 =", blade_md5)

    # -- score every available case ---------------------------------------------
    rows = []
    missing = []
    for tag, role in EXPECTED:
        r = score_case(tag, role)
        if r is None:
            missing.append(tag)
        else:
            rows.append(r)

    print("\nscored %d / %d expected cases  (missing/in-flight: %s)"
          % (len(rows), len(EXPECTED), ", ".join(missing) if missing else "none"))
    print("\n%-13s %5s %7s %8s %7s %8s %8s %9s  %s" %
          ("tag", "a/d", "lam/d", "L_sep/d", "f_rec", "eps_md", "relRMS",
           "R2", "verdict"))
    for r in sorted(rows, key=lambda r: (r["role"], r["ell_p_over_delta"],
                                         r["a_over_delta"])):
        print("%-13s %5.3f %7.2f %8.3f %7.3f %8.3f %8.3f %+9.2f  %s" %
              (r["case_tag"], r["a_over_delta"], r["ell_p_over_delta"],
               r["L_sep_over_delta"], r["f_rec"], r["eps_med"], r["relRMS"],
               r["r2"], r["verdict"]))

    # =======================================================================
    # LOCATE THE PITCH-LADDER CROSSING (a/delta ~ 0.10):  R2=0 & relRMS=0.5.
    # The pitch sweep = the a/d=0.10 cases (wavy_a10 at lam/d=2 included).
    # =======================================================================
    pitch = [r for r in rows if abs(r["a_over_delta"] - 0.10) < 1e-6]
    pitch.sort(key=lambda r: r["ell_p_over_delta"])
    lpd = np.array([r["ell_p_over_delta"] for r in pitch])
    frec_p = np.array([r["f_rec"] for r in pitch])
    relrms_p = np.array([r["relRMS"] for r in pitch])
    r2_p = np.array([r["r2"] for r in pitch])
    eps_p = np.array([r["eps_med"] for r in pitch])

    print("\n" + "-" * 70)
    print("PITCH LADDER (a/delta=0.10):  %d points, lam/d in [%.1f, %.1f]"
          % (len(pitch), lpd.min() if len(lpd) else np.nan,
             lpd.max() if len(lpd) else np.nan))
    n_in_band = int(np.sum((lpd >= 8) & (lpd <= 22)))
    print("  points in the locked [8,22] band (B-L2-1): %d" % n_in_band)

    lpd_c_relrms, blo, bhi = interp_crossing(lpd, relrms_p, FAIL_RELRMS)
    lpd_c_r2, _, _ = interp_crossing(lpd, r2_p, 0.0)
    frec_c, _, _ = interp_crossing(lpd, frec_p, np.nan) if False else (np.nan,)*3
    # f_rec AT the relRMS=0.5 crossing pitch (interpolate f_rec vs lam/d there)
    if np.isfinite(lpd_c_relrms):
        frec_c = float(np.interp(lpd_c_relrms, lpd, frec_p))
        eps_c_meas = float(np.interp(lpd_c_relrms, lpd, eps_p))
        print("  >>> CROSSING located (relRMS=0.5):  (lam/d)_c = %.2f  "
              "in bracket [%.1f, %.1f]" % (lpd_c_relrms, blo, bhi))
        print("      at the crossing:  f_rec,c = %.3f   eps_c(measured) = %.3f"
              % (frec_c, eps_c_meas))
        if np.isfinite(lpd_c_r2):
            print("      (R2=0 crossing pitch = %.2f, consistent)" % lpd_c_r2)
    else:
        frec_c = eps_c_meas = np.nan
        print("  >>> NO relRMS=0.5 crossing bracketed yet "
              "(need points straddling fail/tolerate; sweep may be in flight).")

    # =======================================================================
    # H3 prediction check (held-out beta floor from L1):  predicted eps_c band.
    # =======================================================================
    BETA_FLOOR = 0.161           # closure-independent floor (manuscript)
    eps_c_pred_lo, eps_c_pred_hi = BETA_FLOOR / 1.0, BETA_FLOOR / 0.5
    print("\n[H3] predicted critical depth eps_c in [%.3f, %.3f] "
          "(closure floor beta=%.3f)" % (eps_c_pred_lo, eps_c_pred_hi, BETA_FLOOR))
    if np.isfinite(eps_c_meas):
        inside = eps_c_pred_lo <= eps_c_meas <= eps_c_pred_hi
        ratio = eps_c_meas / np.sqrt(eps_c_pred_lo * eps_c_pred_hi)
        print("     measured eps_c = %.3f  -> %s predicted band  (ratio to band "
              "centre = %.2f; F3 ok if within order unity)"
              % (eps_c_meas, "INSIDE" if inside else "outside", ratio))

    # =======================================================================
    # AMPLITUDE LADDER (lam/d=2):  the boundary is a CORNER, not a vertical line.
    # =======================================================================
    amp = [r for r in rows if abs(r["ell_p_over_delta"] - 2.0) < 0.15
           and r["role"] in ("amp", "both", "flat")]
    amp.sort(key=lambda r: r["a_over_delta"])
    aod = np.array([r["a_over_delta"] for r in amp])
    relrms_a = np.array([r["relRMS"] for r in amp])
    print("\n" + "-" * 70)
    print("AMPLITUDE LADDER (lam/d=2):  %d points, a/d in [%.3f, %.3f]"
          % (len(amp), aod.min() if len(aod) else np.nan,
             aod.max() if len(aod) else np.nan))
    aod_c, alo, ahi = interp_crossing(aod, relrms_a, FAIL_RELRMS)
    if np.isfinite(aod_c):
        print("  >>> amplitude onset (relRMS=0.5):  (a/d)_c = %.3f  "
              "in bracket [%.3f, %.3f]" % (aod_c, alo, ahi))
    else:
        print("  >>> amplitude crossing not bracketed yet.")

    # ----------------------------------------------------------------- save ----
    def col(key):
        return np.array([r[key] for r in rows])

    out = os.path.join(RESULTS, "onset_sweep_results.npz")
    np.savez(
        out,
        # per-case onset schema (B-L1-1)
        case_tag=col("case_tag"), role=col("role"),
        a_over_delta=col("a_over_delta"),
        ell_p_over_delta=col("ell_p_over_delta"),
        fidelity=col("fidelity"),
        L_sep=col("L_sep"), L_sep_over_delta=col("L_sep_over_delta"),
        f_rec=col("f_rec"), f_sep=col("f_sep"),
        eps_med=col("eps_med"), frac_eps_lt0p1=col("frac_eps_lt0p1"),
        relRMS=col("relRMS"), r2=col("r2"), n_stations=col("n_stations"),
        verdict=col("verdict"), provenance=col("provenance"),
        # pitch-ladder crossing (the headline measured boundary)
        pitch_lpd=lpd, pitch_frec=frec_p, pitch_relRMS=relrms_p,
        pitch_r2=r2_p, pitch_eps=eps_p,
        n_pitch_in_band_8_22=int(n_in_band),
        lpd_c_relRMS=float(lpd_c_relrms), lpd_c_bracket_lo=float(blo),
        lpd_c_bracket_hi=float(bhi), lpd_c_r2=float(lpd_c_r2),
        frec_c=float(frec_c), eps_c_measured=float(eps_c_meas),
        # H3 predicted band
        beta_floor=BETA_FLOOR,
        eps_c_pred_lo=eps_c_pred_lo, eps_c_pred_hi=eps_c_pred_hi,
        # amplitude-ladder crossing
        amp_aod=aod, amp_relRMS=relrms_a, aod_c_relRMS=float(aod_c),
        # regression-guard signatures
        canonical_hill_r2=float(r2_hill), blade_md5=blade_md5,
        n_scored=len(rows), n_expected=len(EXPECTED),
        missing=np.array(missing),
        note=("L2 onset sweep: a-priori ODE (evaluate Y_IDX=10) on the wavy "
              "amplitude x pitch family, delta=H/2. Locates the R2=0 / relRMS=0.5 "
              "failure-onset crossing in the pitch ladder (a/d=0.10) and reads off "
              "f_rec,c and the measured eps_c, compared to the held-out closure-floor "
              "prediction eps_c in [0.161,0.322]. RANS (k-omegaSST) -- separation may "
              "be under-predicted; resolved-DNS anchor assessed separately. No "
              "fabrication; only harvested converged/time-averaged cases scored."),
    )
    print("\nSaved -> results/%s  (%d cases scored)" % (os.path.basename(out),
                                                        len(rows)))
    print("=" * 80)


if __name__ == "__main__":
    main()
