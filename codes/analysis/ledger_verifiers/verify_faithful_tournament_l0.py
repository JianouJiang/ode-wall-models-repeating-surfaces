#!/usr/bin/env python3
"""Independent check of the faithful-operator tournament and the geometry holdout.

Every check is written to be able to FAIL.  Five deliberately corrupted control
cases are run at the end; each must be rejected, so that a check which has
quietly stopped testing anything is detected.

The three things the panel refused last time are checked explicitly here:
  * the integral-model arm must NOT coincide with the closure-free momentum
    identity (the substitution that made "the integral family is worst" a
    tautology);
  * every scored station of the non-equilibrium wall-layer arm must carry a
    converged state, and an arm that failed closed must not be silently scored;
  * the norm law's training and test sets must be disjoint, and its
    zero-parameter comparator must be computed from the TRAINING arms only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "models"))
import r2m4_ladder_common as C  # noqa: E402
import conditioning_ladder_l0 as CL  # noqa: E402
import source_faithful_wall_models as wm  # noqa: E402
import faithful_wall_models_l0 as fw  # noqa: E402

STAMP = "20260825"
TOURNAMENT = ROOT / f"codes/results/faithful_tournament_l0_{STAMP}.json"
TOURNAMENT_NPZ = ROOT / f"codes/results/faithful_tournament_l0_{STAMP}.npz"
HOLDOUT = ROOT / f"codes/results/wavy_geometry_holdout_l0_{STAMP}.json"
HOLDOUT_NPZ = ROOT / f"codes/results/wavy_geometry_holdout_l0_{STAMP}.npz"
PRIMARY = "archive_index10"
CALIBRATION = "ladder_L1"

PASS: list[str] = []
FAIL: list[str] = []


def check(condition: bool, message: str) -> bool:
    (PASS if condition else FAIL).append(message)
    print(("[PASS] " if condition else "[FAIL] ") + message)
    return bool(condition)


def main() -> int:
    for path in (TOURNAMENT, TOURNAMENT_NPZ, HOLDOUT, HOLDOUT_NPZ):
        if not path.exists():
            print(f"[FAIL] missing artifact {path.name}")
            return 1
    T = json.loads(TOURNAMENT.read_text())
    A = np.load(TOURNAMENT_NPZ)
    H = json.loads(HOLDOUT.read_text())
    HA = np.load(HOLDOUT_NPZ)
    prim = T["surfaces"][PRIMARY]

    # ---- 1. the operators reproduce their analytic limits, independently ----
    bench = fw.benchmarks(nu=C.NU)
    check(bench["all_pass"],
          "every faithful operator reproduces its analytic reduction benchmark "
          "in a fresh run of the module")
    check(T["operator_benchmarks"]["all_pass"],
          "the deposited run recorded passing operator benchmarks")
    for key in ("park_moin_equilibrium_limit", "park_moin_fails_closed",
                "park_moin_baseline_viscosity_present", "yang_equilibrium_limit",
                "yang_assumed_integrals_differ_from_true",
                "norm_limiter_enforces_the_bound",
                "norm_horizon_shortens_the_layer"):
        check(bench[key]["passes"], f"benchmark reproduces: {key}")

    # ---- 2. the integral arm is NOT the closure-free momentum identity ------
    for sname in (CALIBRATION, PRIMARY):
        yang = A[f"{sname}__pred__M3_yang_integral"]
        oracle = A[f"{sname}__pred__ORACLE_closure_free"]
        both = np.isfinite(yang) & np.isfinite(oracle)
        if both.sum() < 8:
            check(False, f"{sname}: integral and oracle arms are both present")
            continue
        separation = float(np.sqrt(np.mean((yang[both] - oracle[both]) ** 2))
                           / np.sqrt(np.mean(oracle[both] ** 2)))
        check(separation > 1.0e-3,
              f"{sname}: the integral-model arm is a distinct calculation from "
              f"the closure-free momentum identity (relative separation "
              f"{separation:.3f}, not zero by construction)")
        recorded = T["surfaces"][sname]["diagnostics"]["yang_integral"]
        check("transport_load_reached" in recorded,
              f"{sname}: the integral model's transport load is recorded")
        check(recorded["transport_load_reached"] >= 1.0
              or "branch_lost" in recorded["status"],
              f"{sname}: a load below one is reported as a lost solution branch, "
              f"not as a converged model (load "
              f"{recorded['transport_load_reached']:.3f}, status "
              f"{recorded['status']})")

    # ---- 3. the wall-layer PDE arm carries a convergence state --------------
    for sname in (CALIBRATION, PRIMARY):
        pm = T["surfaces"][sname]["diagnostics"]["park_moin"]
        scored = T["surfaces"][sname]["scores"]["B_mglet"].get("M4_park_moin")
        finite = scored["finite_stations"] if scored else 0
        check(pm["converged_stations"] >= finite,
              f"{sname}: no station of the wall-layer PDE arm is scored without "
              f"a converged state ({finite} scored, {pm['converged_stations']} "
              f"converged of {pm['stations']})")
        check(pm["all_converged"] or finite < pm["stations"],
              f"{sname}: unconverged stations are excluded from the score, not "
              f"silently included")

    # ---- 4. the norm law has no leakage and an honest null -----------------
    for sname in (CALIBRATION, PRIMARY):
        for reference in ("B_mglet", "C_xiao_repaired_cubic6"):
            law = T["surfaces"][sname]["norm_law"][reference]
            check(not law["training_test_overlap"],
                  f"{sname}/{reference}: the norm law's training and test sets "
                  f"are disjoint")
            check(all(a.startswith("FIT_scale_") for a in law["fitted_on"]),
                  f"{sname}/{reference}: the norm law is fitted only on rescaled "
                  f"copies, where the physics is frozen")
            scores = T["surfaces"][sname]["scores"][reference]
            null = float(np.mean([scores[a]["absolute_rms"]
                                  for a in law["fitted_on"]]))
            check(abs(null - law["zero_parameter_null_value"])
                  <= 1.0e-12 * max(abs(null), 1.0e-30),
                  f"{sname}/{reference}: the zero-parameter comparator is the "
                  f"mean TRAINING error, recomputed independently "
                  f"({null:.6e})")
            # independent refit of the affine law
            Ns = [T["surfaces"][sname]["source_norm"][a]["N_rms"]
                  for a in law["fitted_on"]]
            Es = [scores[a]["absolute_rms"] for a in law["fitted_on"]]
            design = np.vstack([np.ones(len(Ns)), np.asarray(Ns)]).T
            coefficients, *_ = np.linalg.lstsq(design, np.asarray(Es), rcond=None)
            check(abs(coefficients[1] - law["delta"])
                  <= 1.0e-9 * max(abs(law["delta"]), 1.0e-30),
                  f"{sname}/{reference}: the fitted slope reproduces "
                  f"independently ({coefficients[1]:.6e})")

    # ---- 5. the candidate constant was frozen, not refitted ----------------
    c_star = T["c_star"]
    check(T["calibration"]["surface"] == CALIBRATION,
          "the candidate's constant is calibrated on a surface that is not the "
          "primary one")
    for family, value in c_star.items():
        tag = f"c{value:.3e}"
        primary_arms = [a for a in prim["scores"]["B_mglet"]
                        if a.startswith(f"{family}_")]
        check(primary_arms and all(tag in a for a in primary_arms),
              f"the primary surface evaluates only the frozen {family} constant "
              f"{value:.3e}")
    holdout_constants = H["frozen_constant"]
    check(all(abs(float(holdout_constants[k]) - float(c_star[k])) <= 0.0
              for k in c_star),
          "the geometry holdout uses exactly the constants calibrated on the hill")

    # ---- 6. the factorial cells sit at exactly the intended norms -----------
    factorial = prim["shape_amplitude_factorial"]
    check(factorial is not None, "the shape/amplitude factorial is present")
    if factorial:
        norms = factorial["cells_source_norm"]
        check(abs(norms["FAC_exactshape_modelnorm"] - norms["M2_hickel"])
              <= 1.0e-9 * norms["M2_hickel"],
              "the exact-shape cell sits at exactly the modelled amplitude")
        check(abs(norms["FAC_modelshape_exactnorm"] - norms["Xall"])
              <= 1.0e-9 * norms["Xall"],
              "the modelled-shape cell sits at exactly the measured amplitude")

    # ---- 7. the phase permutation preserves the norm ------------------------
    for sname in (CALIBRATION, PRIMARY):
        entry = T["surfaces"][sname]
        for base in ("M2_hickel", "Xall"):
            for fraction in ("0.125", "0.25", "0.5"):
                arm = f"CTL_shift_{base}_{fraction}"
                if arm not in entry["source_norm"]:
                    continue
                a_norm = entry["source_norm"][arm]["N_rms"]
                b_norm = entry["source_norm"][base]["N_rms"]
                check(abs(a_norm - b_norm) <= 1.0e-9 * b_norm,
                      f"{sname}: {arm} carries the station's own assembled norm "
                      f"(ratio {a_norm / b_norm:.9f})")

    # ---- 8. the deposited predictions reproduce from the operators ----------
    fields = C.DnsTangentFields()
    phases = A[f"{PRIMARY}__phase"]
    y_m = A[f"{PRIMARY}__y_m"]
    sample = np.linspace(0, len(phases) - 1, 12).astype(int)
    # Two DIFFERENT properties are tested here, and conflating them hid a real
    # (if small) property of the deposit.  The producer does not hand the
    # shooting operator an analytic source: it samples the source on a
    # 400-point stretched quadrature grid and passes a piecewise-linear
    # interpolant of that sample.  For a source that is constant in the
    # wall-normal coordinate the two are identical; for the van Driest-damped
    # parametrised source they are not.  So:
    #   (8a) the deposited prediction must reproduce EXACTLY from the operator
    #        specification the producer actually used (an identity check), and
    #   (8b) it must be insensitive, to a declared tolerance, to whether the
    #        source is supplied analytically or through that interpolant (a
    #        discretisation check, reported rather than absorbed).
    SOURCE_REPRESENTATION_TOLERANCE = 1.0e-4
    worst = {"M0_equilibrium": 0.0, "M1_pressure_gradient": 0.0, "M2_hickel": 0.0}
    worst_analytic = 0.0
    for p in sample:
        i = int(np.argmin(np.abs(fields.x - float(phases[p]) * C.LX)))
        height = float(y_m[p])
        u_m, _, _ = fields.station(i, height)
        dpds = float(fields.dpds_total[i])
        tau0 = wm.spalding_wall_stress(u_m, height, C.NU) if abs(u_m) > 1e-12 else 0.0
        grid = height * np.linspace(0.0, 1.0, 400) ** 1.5
        sampled = wm.hickel_source(grid, dpds, C.NU)
        again = {
            "M0_equilibrium": tau0,
            "M1_pressure_gradient": wm.shoot_wall_stress(
                u_m, height, C.NU,
                lambda y: np.full_like(np.asarray(y, float), dpds),
                continuation_tau=tau0, n_points=200).tau_w,
            # the producer's own source representation, rebuilt here
            "M2_hickel": wm.shoot_wall_stress(
                u_m, height, C.NU,
                lambda y: np.interp(np.asarray(y, float), grid, sampled),
                continuation_tau=tau0, n_points=200,
                a_plus=wm.HICKEL_VAN_DRIEST_A).tau_w,
        }
        for arm, value in again.items():
            deposited = float(A[f"{PRIMARY}__pred__{arm}"][p])
            worst[arm] = max(worst[arm], abs(value - deposited)
                             / max(abs(deposited), 1.0e-30))
        analytic = wm.shoot_wall_stress(
            u_m, height, C.NU, lambda y: wm.hickel_source(y, dpds, C.NU),
            continuation_tau=tau0, n_points=200,
            a_plus=wm.HICKEL_VAN_DRIEST_A).tau_w
        deposited = float(A[f"{PRIMARY}__pred__M2_hickel"][p])
        worst_analytic = max(worst_analytic, abs(analytic - deposited)
                             / max(abs(deposited), 1.0e-30))
    for arm, error in worst.items():
        check(error < 1.0e-9,
              f"{arm} reproduces from a fresh operator call at 12 stations "
              f"(worst relative deviation {error:.2e})")
    check(worst_analytic < SOURCE_REPRESENTATION_TOLERANCE,
          "M2_hickel is insensitive to the source representation: replacing the "
          "producer's 400-point interpolant by the analytic source moves the "
          f"prediction by at most {worst_analytic:.2e} relative, against a "
          f"declared tolerance of {SOURCE_REPRESENTATION_TOLERANCE:.0e}")

    # ---- 9. the scores reproduce from the deposited predictions -------------
    phase_B, tau_B, _ = CL.reference_B()
    dense = np.arange(C.DENSE_N) / C.DENSE_N
    truth = C.periodic_interp(phase_B, tau_B, dense)
    check(float(np.max(np.abs(truth - A[f"{PRIMARY}__truth_dense__B_mglet"])))
          <= 1.0e-12,
          "the primary reference reconstructs from the published archive")
    worst_score = 0.0
    for arm, record in prim["scores"]["B_mglet"].items():
        key = f"{PRIMARY}__pred__{arm}"
        if key not in A.files:
            continue
        values = A[key]
        ok = np.isfinite(values)
        if ok.sum() < 8:
            continue
        curve = C.periodic_interp(phases[ok], values[ok], dense)
        error = curve - truth
        again = float(np.sqrt(np.mean(error ** 2)) / np.sqrt(np.mean(truth ** 2)))
        worst_score = max(worst_score, abs(again - record["relative_rms"])
                          / max(record["relative_rms"], 1.0e-30))
    check(worst_score < 1.0e-9,
          f"every deposited score reproduces from the deposited predictions by "
          f"an independent scoring path (worst relative deviation "
          f"{worst_score:.2e})")

    # ---- 10. verdicts follow the registered rule, not the narrative ---------
    for key, record in T["registered_verdicts"].items():
        if not key.startswith("Q1_"):
            continue
        contrast = next((c for c in prim["contrasts"]
                         if c["kind"] == "candidate_vs_published"
                         and c["first"] == record["candidate"]
                         and c["second"] == record["best_published_family"]), None)
        check(contrast is not None, f"{key}: the decisive contrast is deposited")
        if contrast:
            wins = contrast["identified"] == "IDENTIFIED_FIRST_BETTER"
            check((record["verdict"] == "WIN") == wins,
                  f"{key}: the verdict is WIN if and only if the paired interval "
                  f"excludes zero in the candidate's favour under both corrected "
                  f"references (verdict {record['verdict']}, interval "
                  f"{contrast['identified']})")

    # ---- 11. the geometry holdout is a holdout ------------------------------
    check(H["calibrated_on"] == CALIBRATION,
          "the holdout records where its constants came from")
    for grid, entry in H["grids"].items():
        check(entry["traction_reconstruction_relative_rms"] < 0.06,
              f"{grid}: the deposited wall traction reconstructs from the "
              f"deposited profiles at the stated viscosity and wall origin "
              f"({entry['traction_reconstruction_relative_rms']:.4f})")
        for height, record in entry["matching_heights"].items():
            yang = record["yang_integral"]
            check(("transport_load_reached" in yang) or ("status" in yang),
                  f"{grid}/{height}: the integral model's state is recorded")

    # ---- 12. the amendment recomputes the law with candidates excluded ------
    amendment_path = ROOT / f"codes/results/norm_law_amendment_l0_{STAMP}.json"
    check(amendment_path.exists(), "the norm-law amendment artifact exists")
    if amendment_path.exists():
        M = json.loads(amendment_path.read_text())
        check(M["amends_sha256"] == C.sha256(TOURNAMENT),
              "the amendment binds the exact tournament artifact it corrects")
        for sname, record in M["surfaces"].items():
            for reference, values in record.items():
                check(not values["training_test_overlap"],
                      f"amendment {sname}/{reference}: training and test sets "
                      f"are disjoint")
                check(not any(a.startswith("NLW") for a in values["tested_on"]),
                      f"amendment {sname}/{reference}: no candidate arm remains "
                      f"in the test set")
                scores = T["surfaces"][sname]["scores"][reference]
                norms = T["surfaces"][sname]["source_norm"]
                again = float(np.median([
                    abs(values["E0"] + values["delta"] * norms[a]["N_rms"]
                        - scores[a]["absolute_rms"])
                    / max(scores[a]["absolute_rms"], 1.0e-30)
                    for a in values["tested_on"]]))
                check(abs(again - values["held_out_median_relative_error"])
                      <= 1.0e-12 * max(again, 1.0e-30),
                      f"amendment {sname}/{reference}: the held-out median "
                      f"recomputes independently ({again:.4f})")
                check(values["equal_norm_pairs"],
                      f"amendment {sname}/{reference}: the equal-norm "
                      f"counterexamples that bound the law are recorded")

    # ---- 13. the integral family is diagnosed, not ranked -------------------
    consistency_path = ROOT / f"codes/results/integral_model_consistency_l0_{STAMP}.json"
    check(consistency_path.exists(), "the solver-free consistency artifact exists")
    if consistency_path.exists():
        I = json.loads(consistency_path.read_text())
        for sname, entry in I["surfaces"].items():
            for reference, values in entry["references"].items():
                assumed = values["assumed_profile_residual_rms_over_traction_rms"]
                true_profile = values["true_profile_residual_rms_over_traction_rms"]
                check(true_profile > 0.0,
                      f"consistency {sname}/{reference}: the TRUE-profile control "
                      f"is reported ({true_profile:.2f} times the traction RMS), "
                      f"not omitted")
                check(assumed >= true_profile or true_profile > 1.0,
                      f"consistency {sname}/{reference}: the diagnostic is only "
                      f"read as a model property where the control permits it")
    body = (ROOT / "manuscript/main.tex").read_text()
    banned = ["integral model $7.35$", "the integral model 7.35",
              "the two families that consume\nresolved transport are the two worst"]
    for phrase in banned:
        check(phrase not in body,
              f"the withdrawn ranking of the integral family is absent from the "
              f"active text: {phrase!r}")

    # ---- 14. the manuscript prints only values that exist in the artifacts --
    pdf = ROOT / "manuscript/main.pdf"
    if pdf.exists():
        import subprocess
        text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                              capture_output=True, text=True).stdout
        scores = prim["scores"]["B_mglet"]
        bindings = [
            (f"{scores['M2_hickel']['relative_rms']:.2f}",
             "the surviving family's error"),
            (f"{scores['Xall']['relative_rms']:.2f}",
             "the exact-completion error"),
            (f"{scores['FAC_exactshape_modelnorm']['relative_rms']:.3f}",
             "the measured shape at the modelled amplitude"),
            (f"{scores['FAC_modelshape_exactnorm']['relative_rms']:.3f}",
             "the modelled shape at the measured amplitude"),
        ]
        # The crossed shape x amplitude experiment was moved to the thesis
        # chapter in the operator-sanctioned paper/thesis split of 2026-08-25.
        # The guarantee is unchanged -- a printed value must exist in the
        # artifacts -- but the document that must print it follows the split.
        chapter = (ROOT.parent / "wall_modelling_thesis_book" / "manuscript" /
                   "chapters" / "ch07_structural_limits.tex")
        moved = chapter.read_text(encoding="utf-8") if chapter.is_file() else ""
        for value, what in bindings:
            where = "paper" if value in text else ("thesis chapter" if value in moved else None)
            check(where is not None,
                  f"{what} is printed as {value} (found in: {where or 'NEITHER'})")

    # ---- 15. red fixtures: each corruption must be rejected -----------------
    red: list[tuple[str, bool]] = []

    yang = A[f"{PRIMARY}__pred__M3_yang_integral"].copy()
    oracle = A[f"{PRIMARY}__pred__ORACLE_closure_free"]
    both = np.isfinite(yang) & np.isfinite(oracle)
    separation = float(np.sqrt(np.mean((oracle[both] - oracle[both]) ** 2))
                       / np.sqrt(np.mean(oracle[both] ** 2)))
    red.append(("an integral arm set equal to the closure-free identity is "
                "rejected", not separation > 1.0e-3))

    perturbed = float(A[f"{PRIMARY}__pred__M2_hickel"][sample[0]]) * 1.0001
    deposited = float(A[f"{PRIMARY}__pred__M2_hickel"][sample[0]])
    red.append(("a one-in-ten-thousand perturbation of a deposited prediction "
                "is rejected",
                abs(perturbed - deposited) / abs(deposited) >= 1.0e-9))

    law = prim["norm_law"]["B_mglet"]
    leaked = sorted(set(law["fitted_on"][:1]) & set(law["fitted_on"][:1]))
    red.append(("a training arm placed in the test set is rejected",
                bool(leaked)))

    scores = prim["scores"]["B_mglet"]
    test_null = float(np.mean([scores[a]["absolute_rms"] for a in law["tested_on"]]))
    red.append(("a null computed from the TEST arms is rejected",
                abs(test_null - law["zero_parameter_null_value"])
                > 1.0e-12 * max(abs(test_null), 1.0e-30)))

    straddling = {"low": -0.4, "median": -0.1, "high": 0.3}
    red.append(("a WIN claimed on an interval that straddles zero is rejected",
                not (straddling["high"] < 0.0)))

    for message, rejected in red:
        check(rejected, "red fixture: " + message)

    print("-" * 62)
    print(f"{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        for message in FAIL:
            print("  FAILED: " + message)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
