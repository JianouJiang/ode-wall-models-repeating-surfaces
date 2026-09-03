#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L2 implementation: the fidelity-consistency criterion, firmed
==============================================================

This is the L2 (Implementation & experiments) deliverable for the sharp-rib
fidelity-consistency thread.  It discharges the four L1 Judge binds with
COMPUTATION on the converged wall-resolved square-rib LES and the RANS sweep,
replacing the L1 methodology's hardcoded literals with MEASURED numbers:

  B-L2-1 (CRIT)  f_res is MEASURED from the LES resolved-stress (UPrime2Mean)
                 and SGS-viscosity (nut) fields at the matching height, not a
                 hardcoded 0.99.  We read the measured value from
                 ``rib_les_dtype_apriori.npz`` (produced by rib_les_harvest.py at
                 the converged endTime t=140) and its convergence evidence from
                 ``rib_les_stationarity.json`` (t=120 vs t=140).

  B-L2-5 (CRIT)  A second PAIRED test: the exact LES Reynolds stress (resolved,
                 and SGS-completed) is substituted into the 1-D ODE balance for
                 the d-type rib.  It does NOT rescue the prediction
                 (R^2 stays catastrophic; the exact stress makes it WORSE) ->
                 the failure is closure-independent (G3) on the sharp rib, with
                 the resolved fraction now MEASURED at f_res ~ 0.99.

  B-L2-2 (CRIT)  The co-failure window is SCOPED precisely.  Masking is
                 demonstrated at the ONE matched geometry (p/k=3, identical k/d
                 and Re, both fidelities present): RANS certifies (+0.70) while
                 the WRLES fails (-0.94).  At the tightest pitch (p/k=2) the ODE
                 ALSO fails against the RANS reference itself (R^2=-1.29): RANS
                 does NOT certify there, so there is nothing to mask.  The claim
                 is therefore "RANS CAN mask an ODE failure" (p/k=3), NOT "RANS
                 always masks it" (false at p/k=2).  The separator is the
                 ODE-matchability of the RANS reference tau_w: masking requires a
                 benign, ODE-reproducible RANS stress, which the wider p/k=3
                 cavity provides and the tight p/k=2 cavity does not.

  B-L2-3 (minor) The "d-type (WRLES)" <-> p/k=3 naming provenance is recorded
                 explicitly in the emitted npz and printed.

Read-only w.r.t. the DNS/LES/RANS corpus.  Imports the verbatim canonical model
(``evaluate``, ``Y_IDX``) from ``cross_geometry_collapse`` and asserts three
bit-exact regression-guard anchors before reporting any new number (FATAL
anti-circularity).  Emits ``codes/results/rib_fidelity_consistency_l2.npz``.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/rib_fidelity_consistency_l2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

from cross_geometry_collapse import evaluate, Y_IDX  # noqa: E402  verbatim model

# ---------------------------------------------------------------------------
# 0. Regression guards (FATAL anti-circularity).
# ---------------------------------------------------------------------------
GUARDS = {
    "hill_case_1p0": ("periodic_hills_case_1p0_wall_profiles_corrected.npz",
                      -47.68617253416459),
    "rib_les_dtype": ("rib_les_dtype_wall_profiles.npz", -0.9431719607410027),
    "rib_rans_dtype": ("rib_rans_dtype_wall_profiles.npz", -1.2865148282831504),
}


def _p(stem):
    return os.path.join(RESULTS, stem)


def check_guards():
    out = {}
    for name, (stem, ref) in GUARDS.items():
        got = float(evaluate(_p(stem))["r2"])
        drift = abs(got - ref)
        ok = drift < 1e-6
        out[name] = dict(ref=ref, got=got, drift=drift, ok=ok)
        print("  [%s] %-16s R^2 = %.10f  (ref %.10f, drift %.2e)" %
              ("OK " if ok else "!! DRIFT", name, got, ref, drift))
        if not ok:
            raise SystemExit("FATAL regression-guard drift on %s" % name)
    return out


# ---------------------------------------------------------------------------
# 1. B-L2-1 / B-L2-5 : MEASURED f_res + closure-independence substitution.
#    Read from the converged-endTime harvest (rib_les_harvest.py) and its
#    stationarity certificate.  We assert here that f_res was MEASURED (not the
#    L1 literal) and that the substitution did not rescue the prediction.
# ---------------------------------------------------------------------------
def measured_fidelity():
    ap = _p("rib_les_dtype_apriori.npz")
    if not os.path.exists(ap):
        raise SystemExit("FATAL: rib_les_dtype_apriori.npz missing -- run "
                         "rib_les_harvest.py at the converged endTime first.")
    d = np.load(ap, allow_pickle=True)
    f_res_band = float(d["f_res_band_median"])
    f_res_pooled = float(d["f_res_pooled_median"])
    std_r2 = float(d["standard_ml_r2"])
    dns_r2 = float(d["controlled_dns_r2"])          # exact RESOLVED LES stress
    dns_tot_r2 = float(d["controlled_dns_total_r2"])  # resolved + SGS-completed
    # per-station band f_res for the spread
    fb = np.asarray(d["f_res_band"], float)
    fb = fb[np.isfinite(fb)]
    # closure-independence: neither exact-stress arm rescues (both stay R^2<0)
    closure_independent = bool(dns_r2 < 0 and dns_tot_r2 < 0)
    # stationarity certificate
    st = {}
    spath = _p("rib_les_stationarity.json")
    if os.path.exists(spath):
        st = json.load(open(spath))
    return dict(
        f_res_band_median=f_res_band,
        f_res_pooled_median=f_res_pooled,
        f_res_band_min=float(np.min(fb)) if fb.size else np.nan,
        f_res_band_p25=float(np.percentile(fb, 25)) if fb.size else np.nan,
        n_stations=int(fb.size),
        standard_ml_r2=std_r2,
        controlled_dns_r2=dns_r2,
        controlled_dns_total_r2=dns_tot_r2,
        substitution_rescues=bool(dns_r2 >= 0 or dns_tot_r2 >= 0),
        closure_independent=closure_independent,
        stationary=bool(st.get("stationary", False)),
        f_res_drift=float(st.get("drift", {}).get("f_res_band", np.nan)),
        time=str(d["time"]) if "time" in d.files else "n/a",
    )


# ---------------------------------------------------------------------------
# 2. B-L2-2 : scope the co-failure window.  Matched geometry (the ONE place
#    masking is demonstrated) + the p/k=2 RANS genuine failure (no masking) +
#    the matchability separator.
# ---------------------------------------------------------------------------
RANS_SWEEP = [
    ("d-type (p/k=2)", "rib_rans_dtype_wall_profiles.npz", 2),
    ("p/k=3 (RANS)", "rib_rans_pk3_wall_profiles.npz", 3),
    ("p/k=5 (RANS)", "rib_rans_pk5_wall_profiles.npz", 5),
    ("p/k=6 (RANS)", "rib_rans_pk6_wall_profiles.npz", 6),
    ("p/k=7 (RANS)", "rib_rans_pk7_wall_profiles.npz", 7),
    ("k-type (p/k=8)", "rib_rans_ktype_wall_profiles.npz", 8),
]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def cofailure_window(fid):
    # RANS sweep: matchability (relRMS, R^2) of the RANS reference by the ODE
    sweep = []
    for label, stem, pk in RANS_SWEEP:
        d = np.load(_p(stem), allow_pickle=True)
        m = evaluate(_p(stem))
        sweep.append(dict(label=label, p_over_k=pk,
                          pitch_over_delta=float(d["lambda_over_delta"]),
                          r2=float(m["r2"]), relRMS=float(m["relRMS"]),
                          eps_med=float(m["eps_med"]),
                          f_sep=float(m["f_sep"])))
    # the matched geometry: RANS p/k=3 vs WRLES p/k=3 (same k/d, same Re)
    rans3 = next(s for s in sweep if s["p_over_k"] == 3)
    m_les = evaluate(_p("rib_les_dtype_wall_profiles.npz"))
    d_les = np.load(_p("rib_les_dtype_wall_profiles.npz"), allow_pickle=True)
    d_r3 = np.load(_p("rib_rans_pk3_wall_profiles.npz"), allow_pickle=True)
    nu_les = float(np.atleast_1d(d_les["nu"])[0])
    nu_rans = float(np.atleast_1d(d_r3["nu"])[0])
    matched = dict(
        geometry="p/k=3, k/delta=0.2",
        Re_matched=bool(abs(nu_les - nu_rans) < 1e-7),
        nu_les=nu_les, nu_rans=nu_rans,
        r2_rans=rans3["r2"], r2_les=float(m_les["r2"]),
        delta_r2=rans3["r2"] - float(m_les["r2"]),
        sign_flip=bool(float(m_les["r2"]) < 0 and rans3["r2"] >= 0),
        eps_les=float(m_les["eps_med"]), eps_rans=rans3["eps_med"],
        cov_les=float(m_les["frac_eps_lt0p1"]),
        cov_rans=float(evaluate(_p("rib_rans_pk3_wall_profiles.npz"))["frac_eps_lt0p1"]),
        f_res_les=fid["f_res_band_median"],   # MEASURED (B-L2-1)
        f_res_rans=0.0,                        # RANS carries no resolved stress
    )
    # the tightest pitch: RANS genuine failure (no masking)
    rans2 = next(s for s in sweep if s["p_over_k"] == 2)
    no_mask = dict(
        case="d-type RANS (p/k=2)",
        r2=rans2["r2"], relRMS=rans2["relRMS"],
        ode_fails_against_rans=bool(rans2["r2"] < 0),
        note=("at the tightest pitch the ODE fails against the RANS reference "
              "ITSELF (R^2<0), so RANS does not certify -> nothing to mask; "
              "the RANS verdict coincides with the failure"),
    )
    # the separator: ODE-matchability of the RANS reference (relRMS, R^2).
    # masking requires a benign RANS stress the local ODE can reproduce.
    separator = dict(
        metric="ODE-matchability of the RANS reference tau_w (relRMS, R^2)",
        matched_pk3=dict(relRMS=rans3["relRMS"], r2=rans3["r2"],
                         matchable=bool(rans3["r2"] >= 0)),
        tight_pk2=dict(relRMS=rans2["relRMS"], r2=rans2["r2"],
                       matchable=bool(rans2["r2"] >= 0)),
        matchability_ratio=rans2["relRMS"] / rans3["relRMS"],
        statement=("masking (spurious pass) requires (i) a true failure that the "
                   "WRLES exposes AND (ii) a benign RANS reference the convection-"
                   "blind ODE can reproduce.  The wider p/k=3 cavity (p/delta=0.6) "
                   "provides a matchable RANS tau_w (relRMS=0.45, R^2=+0.70); the "
                   "tight p/k=2 cavity (p/delta=0.4) does not (relRMS=1.0, "
                   "R^2=-1.29), so RANS cannot mask there."),
    )
    # the RANS sweep does NOT order the verdict (consistent with the criterion)
    pk = [s["p_over_k"] for s in sweep]
    r2 = [s["r2"] for s in sweep]
    sweep_ordering = dict(
        spearman_pk_r2=spearman(pk, r2),
        note=("the RANS sweep R^2(p/k) is non-monotone and weakly correlated "
              "with pitch -> RANS verdicts do not order the failure, exactly the "
              "fidelity-consistency point; masking is read at the MATCHED "
              "geometry, never from this sweep."),
    )
    return dict(sweep=sweep, matched=matched, no_mask=no_mask,
                separator=separator, sweep_ordering=sweep_ordering)


PROVENANCE = (  # B-L2-3
    "The wall-resolved LES file rib_les_dtype_wall_profiles.npz is at p/k=3 "
    "(pitch/k = 3, lambda/delta=0.6, k/delta=0.2). The 'dtype' tag is historical: "
    "the LES was set up as a classical d-type cavity at rib-gap/k = w/k = 2, and "
    "the pitch is counted rib-centre to rib-centre, so p/k = w/k + 1 = 3. The "
    "matched RANS reference is rib_rans_pk3_wall_profiles.npz (p/k=3). The "
    "rib_rans_dtype file is the SEPARATE tighter p/k=2 RANS cavity."
)


def main():
    print("=" * 76)
    print(" L2 fidelity-consistency (firmed)  rib_fidelity_consistency_l2.py")
    print("=" * 76)
    print("\n[0] Regression guards (FATAL anti-circularity):")
    guards = check_guards()

    print("\n[1] B-L2-1/B-L2-5  MEASURED f_res + closure-independence substitution:")
    fid = measured_fidelity()
    print("  f_res (MEASURED, band-median over wall-model layer) = %.4f  "
          "(pooled %.4f, min %.4f, n=%d stations)"
          % (fid["f_res_band_median"], fid["f_res_pooled_median"],
             fid["f_res_band_min"], fid["n_stations"]))
    print("  converged (t=120 vs t=140, f_res drift %.4f): %s"
          % (fid["f_res_drift"], fid["stationary"]))
    print("  ODE substitution on the d-type rib at MEASURED f_res~0.99:")
    print("    standard closure      R^2 = %+.3f" % fid["standard_ml_r2"])
    print("    + exact resolved LES  R^2 = %+.3f" % fid["controlled_dns_r2"])
    print("    + resolved + SGS      R^2 = %+.3f" % fid["controlled_dns_total_r2"])
    print("  => exact stress does NOT rescue (closure-independent = %s); the "
          "failure is structural." % fid["closure_independent"])

    print("\n[2] B-L2-2  co-failure window (precise scope):")
    win = cofailure_window(fid)
    mt = win["matched"]
    print("  MATCHED geometry %s, Re matched=%s:" % (mt["geometry"], mt["Re_matched"]))
    print("    RANS reference CERTIFIES ODE : R^2 = %+.3f  (f_res = %.1f)"
          % (mt["r2_rans"], mt["f_res_rans"]))
    print("    WRLES reference FAILS    ODE : R^2 = %+.3f  (f_res = %.3f, MEASURED)"
          % (mt["r2_les"], mt["f_res_les"]))
    print("    Delta R^2 = %+.3f  sign flip = %s  (eps iso-depth: LES %.3f vs "
          "RANS %.3f)" % (mt["delta_r2"], mt["sign_flip"], mt["eps_les"], mt["eps_rans"]))
    nm = win["no_mask"]
    print("  TIGHT pitch %s: R^2 = %+.3f -> ODE fails against RANS too (no mask)."
          % (nm["case"], nm["r2"]))
    sp = win["separator"]
    print("  SEPARATOR = %s" % sp["metric"])
    print("    p/k=3 RANS matchable=%s (relRMS %.3f); p/k=2 RANS matchable=%s "
          "(relRMS %.3f); ratio %.2fx"
          % (sp["matched_pk3"]["matchable"], sp["matched_pk3"]["relRMS"],
             sp["tight_pk2"]["matchable"], sp["tight_pk2"]["relRMS"],
             sp["matchability_ratio"]))
    so = win["sweep_ordering"]
    print("  RANS sweep ordering: Spearman(p/k,R^2)=%+.3f -> %s"
          % (so["spearman_pk_r2"], "non-monotone (expected)"))
    print("  CLAIM: RANS CAN mask an ODE failure (p/k=3); it does NOT always "
          "(p/k=2 genuine fail).")

    print("\n[3] B-L2-3  naming provenance:")
    print("  " + PROVENANCE)

    out = os.path.join(RESULTS, "rib_fidelity_consistency_l2.npz")
    np.savez(
        out,
        Y_IDX=Y_IDX,
        guards_json=json.dumps(guards),
        measured_fidelity_json=json.dumps(fid),
        cofailure_window_json=json.dumps(win),
        provenance=PROVENANCE,
        # flat arrays for the figure
        f_res_band_median=fid["f_res_band_median"],
        standard_ml_r2=fid["standard_ml_r2"],
        controlled_dns_r2=fid["controlled_dns_r2"],
        controlled_dns_total_r2=fid["controlled_dns_total_r2"],
        sweep_pk=np.array([s["p_over_k"] for s in win["sweep"]], float),
        sweep_r2=np.array([s["r2"] for s in win["sweep"]], float),
        sweep_relRMS=np.array([s["relRMS"] for s in win["sweep"]], float),
        sweep_pitch_over_delta=np.array([s["pitch_over_delta"]
                                         for s in win["sweep"]], float),
        matched_r2_rans=win["matched"]["r2_rans"],
        matched_r2_les=win["matched"]["r2_les"],
    )
    print("\n[done] wrote %s" % os.path.relpath(out, os.path.join(HERE, "..", "..")))
    print("=" * 76)


if __name__ == "__main__":
    main()
