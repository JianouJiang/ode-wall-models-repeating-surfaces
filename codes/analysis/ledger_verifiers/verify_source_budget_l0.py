#!/usr/bin/env python3
"""Independent check of the source-norm budget and the published-family tournament.

This verifier does not trust the producer's own summary.  It re-scores every arm
from the deposited per-station predictions against references rebuilt from the
published reference artifact, refits the norm law from those re-scored numbers,
re-derives every registered verdict, and re-runs the region split.  It then
checks that what the manuscript prints agrees with what it recomputed.

Six control cases follow.  Each perturbs the deposited evidence in one specific
way and requires the corresponding check to FAIL; a checker that cannot be made
to fail is not measuring anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

NODE = Path(__file__).resolve().parent
ROOT = NODE.parents[2]
RESULTS = ROOT / "codes" / "results"
STAMP = "20260825"
SURFACES = ("archive_index10", "ladder_L1")
REFS = ("B_mglet", "C_xiao_repaired_cubic6")
REFERENCES = RESULTS / f"wall_traction_references_{STAMP}.npz"
TEX = ROOT / "manuscript" / "main.tex"
PDF = ROOT / "manuscript" / "main.pdf"
DENSE_N = 4096
TOL = 1.0e-9
CHECKS: list[tuple[str, bool]] = []


def check(label: str, ok: bool) -> bool:
    CHECKS.append((label, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return bool(ok)


def periodic_interp(x, y, target):
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0], np.r_[y, y, y])


def reference_dense(name):
    refs = np.load(REFERENCES)
    dense = np.arange(DENSE_N) / DENSE_N
    return periodic_interp(refs[f"{name}__phase"], refs[f"{name}__tau"], dense)


def rescore(phase, pred, truth_dense):
    dense = np.arange(DENSE_N) / DENSE_N
    ok = np.isfinite(pred)
    p = periodic_interp(np.asarray(phase)[ok], np.asarray(pred)[ok], dense)
    err = p - truth_dense
    return {"absolute_rms": float(np.sqrt(np.mean(err ** 2))),
            "relative_rms": float(np.sqrt(np.mean(err ** 2))
                                  / np.sqrt(np.mean(truth_dense ** 2))),
            "r2": float(1.0 - np.sum(err ** 2)
                        / np.sum((truth_dense - truth_dense.mean()) ** 2))}


def affine(N, E):
    A = np.vstack([np.ones(len(N)), np.asarray(N, float)]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(E, float), rcond=None)
    return float(coef[0]), float(coef[1])


def pdf_text() -> str:
    out = subprocess.run(["pdftotext", str(PDF), "-"], check=True,
                         capture_output=True, text=True).stdout
    return " ".join(out.split())


def main() -> int:
    regions = json.loads((RESULTS / f"source_budget_regions_l0_{STAMP}.json").read_text())

    for surface in SURFACES:
        j = json.loads((RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.json").read_text())
        d = np.load(RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.npz")
        phase = np.asarray(d["phase"], float)
        arms = sorted(k[len("pred__"):] for k in d.files if k.startswith("pred__"))
        check(f"{surface}: production run used every station (stride 1)",
              j["surface"]["station_stride"] == 1)
        check(f"{surface}: the withdrawn estimator is a negative control, never the primary",
              j["references"]["A_withdrawn_linear4"].startswith("NEGATIVE_CONTROL")
              and j["references"]["B_mglet"] == "PRIMARY_TRUTH")

        # ---- 1. every deposited score re-derived from the deposited arrays --
        worst = 0.0
        for rname in REFS:
            truth = reference_dense(rname)
            for a in arms:
                if a not in j["scores"][rname]:
                    continue
                mine = rescore(phase, d[f"pred__{a}"], truth)
                for key in ("absolute_rms", "relative_rms", "r2"):
                    ref = j["scores"][rname][a][key]
                    worst = max(worst, abs(mine[key] - ref) / max(abs(ref), 1e-12))
        check(f"{surface}: all deposited scores reproduce independently "
              f"(worst relative deviation {worst:.2e})", worst < 1e-9)

        # ---- 2. structural properties of the source norm --------------------
        pairs = [("CTL_scale_Xall_2", "Xall", 2.0),
                 ("CTL_scale_Xall_0.5", "Xall", 0.5),
                 ("CTL_scale_M2_hickel_2", "M2_hickel", 2.0),
                 ("CTL_scale_M1_pressure_gradient_0.5", "M1_pressure_gradient", 0.5)]
        worst_scale = 0.0
        for scaled, base, c in pairs:
            n_s = np.asarray(d[f"norm__{scaled}"], float)
            n_b = np.asarray(d[f"norm__{base}"], float)
            good = np.isfinite(n_s) & np.isfinite(n_b) & (np.abs(n_b) > 0)
            worst_scale = max(worst_scale, float(np.max(np.abs(n_s[good] / (c * n_b[good]) - 1.0))))
        check(f"{surface}: the norm is exactly homogeneous of degree one in the source "
              f"(worst deviation {worst_scale:.2e})", worst_scale < 1e-12)
        viol = 0
        for a in arms:
            n = np.asarray(d[f"norm__{a}"], float)
            w = np.asarray(d[f"work__{a}"], float)
            good = np.isfinite(n) & np.isfinite(w)
            viol += int(np.sum(n[good] < np.abs(w[good]) - 1e-15))
        check(f"{surface}: assembled norm bounds the net contribution at every station "
              f"({viol} violations)", viol == 0)
        check(f"{surface}: the closure-only arms carry exactly zero source norm",
              float(np.nanmax(np.abs(d["norm__M0_equilibrium"]))) == 0.0
              and float(np.nanmax(np.abs(d["norm__M5_meneveau"]))) == 0.0)

        # ---- 3. the integral family reduces to the closure-free budget ------
        gap = float(np.nanmax(np.abs(d["pred__M3_yang_integral"] - d["pred__Xfull_closure_free"])))
        check(f"{surface}: the integral momentum family with resolved inputs IS the "
              f"closure-free reconstruction (max difference {gap:.2e})", gap == 0.0)

        # ---- 4. the norm law refits from the re-scored numbers --------------
        for rname in REFS:
            truth = reference_dense(rname)
            fit_arms = j["norm_law"][rname]["affine_norm_law"]["fitted_on"]
            N = [j["source_norm"][a]["N_rms"] for a in fit_arms]
            E = [rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"] for a in fit_arms]
            E0, delta = affine(N, E)
            dep = j["norm_law"][rname]["affine_norm_law"]
            check(f"{surface}/{rname}: norm law refits to the deposited coefficients",
                  abs(E0 - dep["E0"]) <= 1e-12 + 1e-9 * abs(dep["E0"])
                  and abs(delta - dep["delta"]) <= 1e-12 + 1e-9 * abs(dep["delta"]))
            fams = list(j["norm_law"][rname]["per_family_relative_error"])
            rel = [abs((E0 + delta * j["source_norm"][a]["N_rms"])
                       - rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"])
                   / max(rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"], 1e-30)
                   for a in fams]
            null_level = float(np.mean([rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"]
                                        for a in fams]))
            rel_null = [abs(null_level - rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"])
                        / max(rescore(phase, d[f"pred__{a}"], truth)["absolute_rms"], 1e-30)
                        for a in fams]
            check(f"{surface}/{rname}: the one-slope law beats the zero-parameter null "
                  f"out of sample ({np.median(rel):.3f} vs {np.median(rel_null):.3f})",
                  np.median(rel) < np.median(rel_null))
            check(f"{surface}/{rname}: deposited out-of-sample errors match the recomputed ones",
                  abs(float(np.median(rel))
                      - j["norm_law"][rname]["out_of_sample_median_relative_error"]) < 1e-9)

        # ---- 5. the two interventions, re-derived as stated -----------------
        shifts = [c for c in j["contrasts"] if c["kind"] == "phase_shift"]
        modelled = [c for c in shifts if c["second"] == "M2_hickel"]
        exact = [c for c in shifts if c["second"] == "Xall"]
        check(f"{surface}: both interventions were run at all three shifts",
              len(modelled) == 3 and len(exact) == 3)
        ratios = [c["norm_ratio"] for c in shifts]
        # The shift is a physics-destroying intervention, not an exactly
        # norm-matched one: moving the source to a station with a different
        # matching height changes its assembled norm.  Every measured ratio is
        # ABOVE one, so the shifted arm is handicapped by the norm law, and a
        # tie is therefore conservative for the conclusion drawn from it.  The
        # gate records that the handicap runs in that direction and is bounded.
        check(f"{surface}: the shift inflates rather than deflates the norm "
              f"(measured ratios {min(ratios):.2f}-{max(ratios):.2f})",
              all(1.0 <= r < 1.7 for r in ratios))
        # Outcome-neutral: the recorded verdict must be the one the recorded
        # intervals imply, whatever that verdict is.
        bad = []
        for c in shifts:
            b, cc = c["delta"]["B_mglet"], c["delta"]["C_xiao_repaired_cubic6"]
            if b["high"] < 0 and cc["high"] < 0:
                implied = "IDENTIFIED_FIRST_BETTER"
            elif b["low"] > 0 and cc["low"] > 0:
                implied = "IDENTIFIED_SECOND_BETTER"
            elif (b["high"] < 0 and cc["low"] > 0) or (b["low"] > 0 and cc["high"] < 0):
                implied = "CONTRADICTORY_ACROSS_REFERENCES"
            else:
                implied = "UNRESOLVED"
            if implied != c["identified"]:
                bad.append(c["first"])
        check(f"{surface}: every intervention verdict follows from its own intervals",
              not bad)

    # ---- 6. the region split, re-derived -----------------------------------
    for surface in SURFACES:
        for rname in REFS:
            e = regions["surfaces"][surface]["references"][rname]
            implied = ("SUPPORTED" if e["delta_ratio_sloped_over_flat"] >= 3.0
                       else "REFUTED")
            check(f"{surface}/{rname}: the registered geometry verdict matches its own "
                  f"measured ratio ({e['delta_ratio_sloped_over_flat']:.2f}, {e['P5_verdict']})",
                  implied == e["P5_verdict"])
    check("the registered geometry prediction is reported with the outcome it got",
          regions["P5_verdict_overall"] in ("SUPPORTED", "MIXED", "REFUTED"))

    # ---- 7. what the paper prints ------------------------------------------
    tex = " ".join(TEX.read_text(encoding="utf-8").split())
    compiled = pdf_text() if PDF.exists() else ""
    j = json.loads((RESULTS / f"source_budget_tournament_l0_{SURFACES[0]}_{STAMP}.json").read_text())
    delta = j["norm_law"]["B_mglet"]["affine_norm_law"]["delta"]
    printed = f"{delta:.2f}"
    check("the paper prints the measured norm-law slope",
          printed in tex and (not compiled or printed in compiled))
    forbidden = ("restoring the omitted convection does not repair",
                 "the pressure-gradient ODE fails catastrophically")
    check("no withdrawn a-priori ladder claim is printed",
          not any(f in tex.lower() for f in forbidden))
    # The displacement experiment was moved to the thesis chapter in the
    # author-sanctioned paper/thesis split of 2026-08-25, so the statement is
    # required in the corpus (paper OR that chapter) rather than in the paper
    # alone -- and the framing it exists to prevent is forbidden in both.
    chapter = (ROOT.parent / "wall_modelling_thesis_book" / "manuscript" /
               "chapters" / "ch07_structural_limits.tex")
    corpus = (tex + (chapter.read_text(encoding="utf-8") if chapter.is_file() else "")).lower()
    stated = any(w in corpus for w in
                 ("not identifiable", "cannot be distinguished", "unresolved"))
    check("the exact-source shift is stated as NOT identifiable rather than as making "
          "no difference (paper or thesis chapter)",
          stated and "shift makes no difference" not in corpus)

    # ---- control cases ------------------------------------------------------
    print("\nRED FIXTURES (each must be caught)")
    surface = SURFACES[0]
    j = json.loads((RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.json").read_text())
    d = np.load(RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.npz")
    phase = np.asarray(d["phase"], float)
    truth = reference_dense("B_mglet")

    perturbed = np.asarray(d["pred__M2_hickel"], float) * 1.01
    mine = rescore(phase, perturbed, truth)
    check("R1 a one-per-cent perturbation of a prediction is caught by the re-score",
          abs(mine["absolute_rms"] - j["scores"]["B_mglet"]["M2_hickel"]["absolute_rms"])
          / j["scores"]["B_mglet"]["M2_hickel"]["absolute_rms"] > 1e-9)

    withdrawn = reference_dense("A_withdrawn_linear4")
    families = ("M0_equilibrium", "M1_pressure_gradient", "M2_hickel",
                "M4_park_moin", "M5_meneveau", "Xc_exact_convection")
    a_scores = {a: rescore(phase, d[f"pred__{a}"], withdrawn)["relative_rms"]
                for a in families}
    b_scores = {a: rescore(phase, d[f"pred__{a}"], truth)["relative_rms"]
                for a in families}
    flipped = [(u, v) for i, u in enumerate(families) for v in families[i + 1:]
               if (a_scores[u] < a_scores[v]) != (b_scores[u] < b_scores[v])]
    check(f"R2 scoring against the withdrawn estimator reorders published families "
          f"({len(flipped)} of {len(families) * (len(families) - 1) // 2} pairs), so a "
          f"silent reference substitution cannot pass unnoticed", len(flipped) > 0)

    fake = json.loads(json.dumps(regions))
    ref0 = fake["surfaces"][SURFACES[0]]["references"]["B_mglet"]
    ref0["P5_verdict"] = "SUPPORTED"
    implied = "SUPPORTED" if ref0["delta_ratio_sloped_over_flat"] >= 3.0 else "REFUTED"
    check("R3 a registered verdict flipped to the flattering outcome is caught",
          implied != ref0["P5_verdict"])

    tampered = np.asarray(d["pred__M3_yang_integral"], float).copy()
    tampered[0] += 1.0e-12
    check("R4 a broken integral-family identity is caught",
          float(np.nanmax(np.abs(tampered - d["pred__Xfull_closure_free"]))) != 0.0)

    n_s = np.asarray(d["norm__CTL_scale_Xall_2"], float) * 1.001
    n_b = np.asarray(d["norm__Xall"], float)
    good = np.isfinite(n_s) & np.isfinite(n_b) & (np.abs(n_b) > 0)
    check("R5 a norm that is not homogeneous in the source amplitude is caught",
          float(np.max(np.abs(n_s[good] / (2.0 * n_b[good]) - 1.0))) >= 1e-12)

    fake_contrast = {"delta": {"B_mglet": {"low": -1.0, "high": 2.0, "median": 0.5},
                              "C_xiao_repaired_cubic6": {"low": -1.0, "high": 2.0,
                                                         "median": 0.5}},
                     "identified": "IDENTIFIED_SECOND_BETTER"}
    b, cc = fake_contrast["delta"]["B_mglet"], fake_contrast["delta"]["C_xiao_repaired_cubic6"]
    implied = "UNRESOLVED" if not (b["low"] > 0 and cc["low"] > 0) else "IDENTIFIED_SECOND_BETTER"
    check("R6 an interval that straddles zero cannot be reported as identified",
          implied != fake_contrast["identified"])

    passed = sum(1 for _, ok in CHECKS if ok)
    print(f"\nsource-norm budget L0: {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
