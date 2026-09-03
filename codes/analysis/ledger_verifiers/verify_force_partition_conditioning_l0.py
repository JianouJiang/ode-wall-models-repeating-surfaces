#!/usr/bin/env python3
r"""
verify_force_partition_conditioning_l0.py
=========================================

Independent check of the L0 node_001 force-partition conditioning result.

Design rules observed (they are the recurring lessons of this project):
  * No gate encodes an outcome.  Where a direction is asserted it is asserted
    because it is what the identity forces (an exact algebraic relation), never
    because a previous run happened to produce it.  The two-factor prediction is
    checked as "the registered prediction and the measured outcome agree", with
    the measured value reported either way.
  * Nothing is proved by a file name or a tool name.
  * Load-bearing numbers are RE-DERIVED here from the raw archives by an
    independent implementation, not read back from the artifact under test.
  * Six control cases perturb the artifact and require the corresponding check to
    fail, so a check that cannot fail is itself a defect.

Run:  python3 codes/analysis/ledger_verifiers/verify_force_partition_conditioning_l0.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
CODES = os.path.dirname(ANALYSIS)
ROOT = os.path.dirname(CODES)
RESULTS = os.path.join(CODES, "results")
RAW = os.path.join(CODES, "raw_data")

sys.path.insert(0, os.path.join(
    RAW, "geometry_driven", "xiao_pehill_parameterized",
    "utility", "hill-geometry-gereration"))
from hillShape import profile as hill_profile          # noqa: E402

ART = os.path.join(RESULTS, "force_partition_conditioning_l0_20260825.json")
PRODUCER = os.path.join(ANALYSIS, "force_partition_conditioning_l0.py")
MGLET = os.path.join(RAW, "periodic_hill_ufr3_30", "ercoftac_ufr3_30",
                     "UFR3-30_data-NP-Re5600-DNS2-11.dat")
KRANK = os.path.join(RAW, "periodic_hill_ufr3_30", "krank_2018_re10595",
                     "KKW_DNS_Periodic_Hill_Re10595_cf_cp_bottom.dat")
LEGACY = os.path.join(RESULTS, "formdrag_partition.npz")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def by_case(payload, key="partitions"):
    return {p["case"]: p for p in payload[key]}


# ---------------------------------------------------------------------------
# independent re-implementation of the force partition (does NOT import the
# producer): a second code path for the two load-bearing hill numbers
# ---------------------------------------------------------------------------
def independent_hill_partition(path, comments=None, delimiter=None,
                               strip_placeholders=False):
    a = np.loadtxt(path, comments=comments, delimiter=delimiter) if comments \
        else np.loadtxt(path)
    if strip_placeholders:
        a = a[~((a[:, 1] == 0.0) & (a[:, 2] == 0.0))]
    o = np.argsort(a[:, 0])
    x, tau, p = a[o, 0], a[o, 1], a[o, 2]
    yw = np.asarray(hill_profile(x.copy()), float)
    # independent slope: centred differences written out, not np.gradient
    hp_ = np.empty_like(yw)
    hp_[1:-1] = (yw[2:] - yw[:-2]) / (x[2:] - x[:-2])
    hp_[0] = (yw[1] - yw[0]) / (x[1] - x[0])
    hp_[-1] = (yw[-1] - yw[-2]) / (x[-1] - x[-2])
    Ff = float(np.trapezoid(p * hp_, x))
    Fv = float(np.trapezoid(tau, x))
    return {"F_form": Ff, "F_visc": Fv, "f_form": Ff / (Ff + Fv),
            "kappa": (Ff + Fv) / Fv}


def run_checks(payload, tag=""):
    """All content checks.  Returns the number of failures introduced."""
    n0 = len(FAIL)
    P = by_case(payload)

    # ---- C1 structure -------------------------------------------------
    check(f"{tag}schema and terminal status",
          payload.get("schema") == "force-partition-conditioning-l0-v1"
          and payload.get("status") == "FORCE_PARTITION_CONDITIONING_L0_OK")

    # ---- C2/C3 independent re-derivation of the hill partition --------
    ind_m = independent_hill_partition(MGLET, strip_placeholders=True)
    got_m = P["hill_pehill_MGLET_Re5600"]
    d_m = abs(ind_m["f_form"] - got_m["f_form"])
    check(f"{tag}hill form-drag fraction (Re 5,600) reproduced by an independent "
          f"implementation", d_m < 5e-4, f"|delta| = {d_m:.2e}")

    ind_k = independent_hill_partition(KRANK, comments="%", delimiter=",")
    got_k = P["hill_pehill_KRANK_Re10595"]
    d_k = abs(ind_k["f_form"] - got_k["f_form"])
    check(f"{tag}hill form-drag fraction (Re 10,595) reproduced by an independent "
          f"implementation", d_k < 5e-4, f"|delta| = {d_k:.2e}")

    # ---- C4 two independent DNS agree ---------------------------------
    agree = payload["hill_two_reference_agreement"]["absolute_difference"]
    check(f"{tag}two independent hill DNS agree on the partition",
          agree < 0.02, f"|delta f_form| = {agree:.4f}")

    # ---- C5 the hill is form-drag dominated, on both references -------
    check(f"{tag}the periodic hill is form-drag dominated on both references",
          got_m["f_form"] > 0.9 and got_k["f_form"] > 0.9,
          f"{got_m['f_form']:.4f} / {got_k['f_form']:.4f}")

    # ---- C6 pressure-datum invariance ---------------------------------
    dat = [(p["case"], p["datum_drift_per_unit_shift_relative"])
           for p in payload["partitions"]
           if "datum_drift_per_unit_shift_relative" in p]
    worst_dat = max(v for _, v in dat)
    check(f"{tag}form drag is invariant to the pressure datum",
          worst_dat < 1e-6,
          f"worst relative drift per unit shift {worst_dat:.2e} over {len(dat)} cases")

    # ---- C7 exact factorisation ---------------------------------------
    fac = payload["factorisation_worst_relative_residual"]
    check(f"{tag}kappa factorises exactly into partition and sign factors",
          fac < 1e-12, f"worst relative residual {fac:.2e}")

    # ---- C8 the transfer identity -------------------------------------
    tr = payload["transfer_worst_relative_deviation"]
    check(f"{tag}the transfer identity holds to machine precision",
          tr < 1e-12, f"worst relative deviation {tr:.2e}")

    # ---- C9 momentum-closure gate actually excludes -------------------
    gate = payload["momentum_closure_gate"]
    bad = [p["case"] for p in payload["partitions"]
           if p.get("drive_known") and not p.get("momentum_closure_pass")]
    kept = [p for p in payload["partitions"]
            if p.get("drive_known") and p.get("momentum_closure_pass")]
    check(f"{tag}every retained case with a known drive closes its momentum "
          f"balance within the gate",
          all(abs(p["momentum_closure_relative_residual"]) <= gate for p in kept)
          and len(kept) > 0,
          f"{len(kept)} retained, worst residual "
          f"{max(abs(p['momentum_closure_relative_residual']) for p in kept):.2e}")
    check(f"{tag}the excluded set is reported rather than silently dropped",
          set(bad) == set(payload["excluded_on_momentum_closure"]),
          f"excluded: {bad}")

    # ---- C10 exact within-run flat-wall control -----------------------
    ctrls = [p["flat_wall_within_run_control"] for p in payload["partitions"]
             if p.get("flat_wall_within_run_control")]
    check(f"{tag}the smooth wall of the same simulations gives kappa = 1 exactly",
          len(ctrls) >= 2 and all(c["F_form"] == 0.0 and c["kappa"] == 1.0
                                  for c in ctrls),
          f"{len(ctrls)} within-run controls")

    # ---- C11/C12 the partition reproduces the deposited fractions -----
    # Split by construction: rows formed from the solver's own signed patch
    # integrals must reproduce EXACTLY; rows re-formed here from the sampled wall
    # line are a different quadrature of the same surface integral, so they must
    # agree only to a difference that FALLS with grid refinement.  A single loose
    # tolerance would hide the distinction.
    dep = payload["deposited_agreement"]
    exact = {p["case"]: p["deposited_form_fraction_reproduced_to"]
             for p in payload["partitions"]
             if p["case"] in dep["exact_rows"]}
    check(f"{tag}rows formed from the solver's own signed patch integrals "
          f"reproduce the deposited fractions exactly",
          len(exact) >= 5 and all(v == 0.0 for v in exact.values()),
          f"{len(exact)} cases, all exact")
    conv = dep["recomputed_rows_convergence"]
    check(f"{tag}the re-formed wall-line quadrature agrees with the deposited "
          f"face-based integral to a difference that falls with refinement",
          dep["recomputed_difference_falls_with_refinement"] is True
          and min(r["abs_difference"] for r in conv) < 5e-5,
          "; ".join(f"{r['grid']} {r['abs_difference']:.2e}" for r in conv))

    # ---- C13 closure independence, in BOTH normalisations -------------
    ci = payload["closure_independence"]
    check(f"{tag}substituting the exact resolved stress is worse on the wall-"
          f"stress score in every pair",
          ci["exact_stress_worse_on_r2"] == ci["n_pairs"] and ci["n_pairs"] >= 6,
          f"{ci['exact_stress_worse_on_r2']}/{ci['n_pairs']}")
    check(f"{tag}substituting the exact resolved stress is worse in force units "
          f"in every pair",
          ci["exact_stress_worse_in_force_units"] == ci["n_pairs"],
          f"{ci['exact_stress_worse_in_force_units']}/{ci['n_pairs']}")

    # ---- C14/C15/C16 the ordering inversion ---------------------------
    inv = payload["ordering_inversion"]
    check(f"{tag}the wall-stress ranking and the misplaced-force ranking disagree "
          f"on a majority of geometry pairs",
          inv["discordant_pairs"] > inv["concordant_pairs"],
          f"{inv['discordant_pairs']} discordant vs {inv['concordant_pairs']} concordant")
    dc = inv["decisive_contrast"]
    check(f"{tag}the geometry with the best wall-stress score misplaces more wall "
          f"force than the geometry with the worst",
          dc["best_r2_misplaced_force_fraction"]
          > dc["worst_r2_misplaced_force_fraction"],
          f"{100 * dc['best_r2_misplaced_force_fraction']:.1f}% (R2 "
          f"{dc['best_r2']:+.2f}) vs {100 * dc['worst_r2_misplaced_force_fraction']:.1f}% "
          f"(R2 {dc['worst_r2']:+.2f}), factor {dc['inversion_factor']:.1f}")
    check(f"{tag}the decisive contrast survives every grid combination",
          inv["contrast_holds_on_every_grid_combination"] is True)
    check(f"{tag}the decisive contrast survives both admissible period "
          f"quadratures",
          inv["contrast_holds_under_other_quadrature"] is True
          and inv["worst_quadrature_sensitivity_unsigned"] < 0.05,
          f"worst quadrature sensitivity "
          f"{100 * inv['worst_quadrature_sensitivity_unsigned']:.2f}%")
    check(f"{tag}the decisive contrast survives every matching height on the "
          f"available ladder",
          inv["wavy_minimum_misplaced_force_fraction_over_ladder"]
          > dc["worst_r2_misplaced_force_fraction"],
          f"ladder minimum {100 * inv['wavy_minimum_misplaced_force_fraction_over_ladder']:.1f}%")
    check(f"{tag}the cross-geometry matching-height confound is stated",
          "matching surface" in inv.get("caveat", ""))

    # ---- C17 the registered two-factor prediction ---------------------
    # Outcome-neutral: the identity forces the outcome to be a PRODUCT, so a
    # geometry-only number cannot order it.  We report the measured value and
    # require the artifact to state the prediction before the outcome.
    mono = payload["kappa_orders_verdict_monotonically"]
    check(f"{tag}the registered two-factor prediction is recorded",
          "conditioning factor, not a predictor"
          in payload.get("registered_prediction", ""))
    check(f"{tag}the measured ordering behaviour of kappa is reported "
          f"(monotone = {mono})", isinstance(mono, bool))

    # ---- C18/C19 the archived negative-control audit ------------------
    aud = payload["legacy_negative_control_audit"]
    d1 = aud["D1_smooth_wall_zero"]
    check(f"{tag}the archived table's zero form drag for the smooth curved walls "
          f"is contradicted by measurement",
          d1["archived_phi_FD_hill"] == 0.0
          and min(d1["measured_f_form_hill_mglet"],
                  d1["measured_f_form_hill_krank"]) > 0.9,
          f"archived {d1['archived_phi_FD_hill']:.3f} vs measured "
          f"{d1['measured_f_form_hill_mglet']:.4f}")
    if os.path.exists(LEGACY):
        L = np.load(LEGACY, allow_pickle=True)
        lk = [str(k) for k in L["keys"]]
        lp = np.asarray(L["phi_FD"], float)
        zeros = sorted(k for k, v in zip(lk, lp) if v == 0.0)
        check(f"{tag}the archived zero-assignment set is reported exactly",
              zeros == sorted(d1["archived_cases_assigned_zero"]),
              f"{len(zeros)} cases")
    d2 = aud["D2_absolute_value_estimator"]
    check(f"{tag}at least one measured wall has a viscous traction opposing the "
          f"drive", d2["n_sign_inverted_walls_in_set"] >= 1,
          f"{d2['n_sign_inverted_walls_in_set']} case(s)")
    check(f"{tag}the absolute-value estimator maps every case below unity and so "
          f"cannot express that inversion",
          d2["max_absvalue_estimate"] < 1.0,
          f"max = {d2['max_absvalue_estimate']:.4f}")
    inv_cases = [p for p in payload["partitions"] if p["f_form"] > 1.0]
    check(f"{tag}a sign-inverted wall carries a negative conditioning number",
          len(inv_cases) >= 1 and all(p["kappa"] < 0 for p in inv_cases),
          f"{[round(p['kappa'], 2) for p in inv_cases]}")

    # ---- C20 the constrained model attains its exact bound ------------
    cw = payload["constrained_wall_model"]
    check(f"{tag}the projection attains its own exact bound",
          cw["bound_attained_to"] < 1e-3,
          f"worst gap {cw['bound_attained_to']:.2e}")
    check(f"{tag}the pointwise payoff of the integral constraint is reported "
          f"rather than claimed", "registered_before_reading" in cw
          and isinstance(cw["median_pointwise_rms_reduction"], float),
          f"median {100 * cw['median_pointwise_rms_reduction']:.1f}%")

    return len(FAIL) - n0


def static_checks():
    """The producer must not take wall traction from the withdrawn estimator."""
    src = open(PRODUCER).read()
    check("the producer scores the hill against a published full-wall DNS",
          "MGLET_5600" in src and "np.interp(x, mglet_x, mglet_tau)" in src)
    check("the producer never reads wall traction from the partly withdrawn "
          "archive",
          'd["tau_w"]' not in src and "d['tau_w']" not in src)
    check("the absolute-value estimator appears only as a named negative control",
          src.count("f_form_absvalue_estimator") >= 1
          and "negative control" in src)


def red_fixtures(payload):
    """Each fixture perturbs the artifact and REQUIRES a check to fail."""
    print("\n  control cases (each must trip the corresponding check):")
    fixtures = []

    def fixture(name, mutate):
        p = copy.deepcopy(payload)
        mutate(p)
        base_fail = len(FAIL)
        sink = []
        real_pass, real_fail = PASS[:], FAIL[:]
        PASS.clear()
        FAIL.clear()
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_checks(p, tag="[fixture] ")
            tripped = len(FAIL) > 0
        finally:
            sink = list(FAIL)
            PASS.clear()
            PASS.extend(real_pass)
            FAIL.clear()
            FAIL.extend(real_fail)
        ok = tripped
        (PASS if ok else FAIL).append(f"control case: {name}")
        print(f"    [{'PASS' if ok else 'FAIL'}] control case: {name}"
              f"  -- {'tripped ' + str(len(sink)) + ' check(s)' if ok else 'NOT DETECTED'}")
        fixtures.append((name, ok))
        _ = base_fail

    def r1(p):                       # the hill is quietly made smooth again
        for q in p["partitions"]:
            if q["case"] == "hill_pehill_MGLET_Re5600":
                q["f_form"] = 0.0
        p["legacy_negative_control_audit"]["D1_smooth_wall_zero"][
            "measured_f_form_hill_mglet"] = 0.0
    fixture("assigning the hill zero form drag is detected", r1)

    def r2(p):                       # the sign inversion is estimated away
        for q in p["partitions"]:
            if q["f_form"] > 1.0:
                q["f_form"] = q["f_form_absvalue_estimator"]
                q["kappa"] = abs(q["kappa"])
        p["legacy_negative_control_audit"][
            "D2_absolute_value_estimator"]["n_sign_inverted_walls_in_set"] = 0
    fixture("losing the sign-inverted wall is detected", r2)

    def r3(p):                       # the transfer identity is broken by 1%
        p["transfer_worst_relative_deviation"] = 1e-2
    fixture("a broken transfer identity is detected", r3)

    def r4(p):                       # the factorisation is broken
        p["factorisation_worst_relative_residual"] = 1e-3
    fixture("a broken exact factorisation is detected", r4)

    def r5(p):                       # a non-stationary case is smuggled back in
        for q in p["partitions"]:
            if q.get("drive_known") and not q.get("momentum_closure_pass"):
                q["momentum_closure_pass"] = True
        # the excluded list is left as it was, so the two disagree
    fixture("re-admitting a case that fails the momentum gate is detected", r5)

    def r6(p):                       # the inversion is reversed
        dc = p["ordering_inversion"]["decisive_contrast"]
        dc["best_r2_misplaced_force_fraction"], \
            dc["worst_r2_misplaced_force_fraction"] = (
                dc["worst_r2_misplaced_force_fraction"],
                dc["best_r2_misplaced_force_fraction"])
    fixture("a reversed decisive contrast is detected", r6)

    def r8(p):                       # a patch-integral row silently drifts
        for q in p["partitions"]:
            if q["case"] in p["deposited_agreement"]["exact_rows"]:
                q["deposited_form_fraction_reproduced_to"] = 1e-9
                break
    fixture("drift in a row that must reproduce exactly is detected", r8)

    def r9(p):                       # the quadrature difference stops converging
        p["deposited_agreement"][
            "recomputed_difference_falls_with_refinement"] = False
    fixture("a non-converging quadrature difference is detected", r9)

    def r7(p):                       # the pressure datum starts to matter
        for q in p["partitions"]:
            if "datum_drift_per_unit_shift_relative" in q:
                q["datum_drift_per_unit_shift_relative"] = 1e-3
    fixture("loss of pressure-datum invariance is detected", r7)


def main():
    print("=" * 78)
    print("VERIFY  force-partition conditioning of the wall-model target (L0)")
    print("=" * 78)
    if not os.path.exists(ART):
        print(f"  [FAIL] artifact missing: {ART}")
        return 2
    payload = json.load(open(ART))

    print("\n  static checks on the producer:")
    static_checks()
    print("\n  content checks:")
    run_checks(payload)
    red_fixtures(payload)

    n = len(PASS) + len(FAIL)
    print(f"\n{len(PASS)}/{n} checks passed "
          f"(force-partition conditioning, L0 node_001)")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
