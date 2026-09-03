#!/usr/bin/env python3
"""AMENDMENT: recompute the source-norm law's held-out test with a corrected
test-set definition.

Defect found in this node's own instrument, after the production run
-------------------------------------------------------------------
``faithful_tournament_l0.py`` builds the norm law's test set by excluding the
fitted arms, the rescaled controls and the candidate family whose names begin
``NLWM_``.  The second candidate family, the wall-normal source horizon, is
named ``NLWH_`` and was NOT excluded.  Its arms are not in the fit, so this is
not training leakage -- the fit and test sets remain disjoint -- but on the
calibration surface, where the whole constant grid is evaluated, fifty candidate
arms enter a test set that is supposed to measure how well the law predicts
PUBLISHED families and completion variants.  The reported held-out median is
then dominated by arms of the law's own construction.

This amendment recomputes the law from the DEPOSITED scores and norms, with no
new model evaluation of any kind, under a test-set definition that excludes both
candidate families.  It supersedes the ``norm_law`` block of the tournament
artifact; the superseded values are retained here so the change is auditable.

Read-only on every input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402

STAMP = "20260825"
TINY = 1.0e-30
NORMLESS = ("M3_yang_integral", "M5_meneveau")
CANDIDATE_PREFIXES = ("NLWM_", "NLWH_")
CONTROL_PREFIXES = ("CTL_", "FIT_scale_")


def affine_fit(N, E):
    N = np.asarray(N, float)
    E = np.asarray(E, float)
    design = np.vstack([np.ones_like(N), N]).T
    coefficients, *_ = np.linalg.lstsq(design, E, rcond=None)
    residual = E - design @ coefficients
    return (float(coefficients[0]), float(coefficients[1]),
            float(np.sqrt(np.mean(residual ** 2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=STAMP)
    args = ap.parse_args()
    source = ROOT / f"codes/results/faithful_tournament_l0_{args.stamp}.json"
    T = json.loads(source.read_text())

    result = {
        "schema": "norm_law_amendment_l0/1",
        "amends": source.name,
        "amends_sha256": C.sha256(source),
        "defect": ("the tournament's test-set filter excluded the uniformly "
                   "rescaling candidate family but not the wall-normal horizon "
                   "family, so candidate arms entered a test set intended for "
                   "published families and completion variants"),
        "correction": ("both candidate families are excluded; the fit set is "
                       "unchanged and was already disjoint from the test set, so "
                       "this is a change of what is being predicted, not a "
                       "leakage repair"),
        "no_new_model_evaluation": True,
        "surfaces": {},
    }
    for sname, entry in T["surfaces"].items():
        record = {}
        for reference in ("B_mglet", "C_xiao_repaired_cubic6"):
            scores = entry["scores"][reference]
            norms = entry["source_norm"]
            train = sorted(a for a in scores if a.startswith("FIT_scale_"))
            test = sorted(
                a for a in scores
                if not any(a.startswith(p) for p in CONTROL_PREFIXES)
                and not any(a.startswith(p) for p in CANDIDATE_PREFIXES)
                and a not in NORMLESS
                and norms.get(a, {}).get("N_rms") is not None)
            E0, delta, fit_rms = affine_fit(
                [norms[a]["N_rms"] for a in train],
                [scores[a]["absolute_rms"] for a in train])
            null = float(np.mean([scores[a]["absolute_rms"] for a in train]))
            relative = {a: abs(E0 + delta * norms[a]["N_rms"]
                               - scores[a]["absolute_rms"])
                        / max(scores[a]["absolute_rms"], TINY) for a in test}
            relative_null = {a: abs(null - scores[a]["absolute_rms"])
                             / max(scores[a]["absolute_rms"], TINY) for a in test}
            superseded = entry["norm_law"][reference]
            record[reference] = {
                "fitted_on": train,
                "tested_on": test,
                "training_test_overlap": sorted(set(train) & set(test)),
                "excluded_candidate_arms": sorted(
                    a for a in scores
                    if any(a.startswith(p) for p in CANDIDATE_PREFIXES)),
                "excluded_normless_arms": list(NORMLESS),
                "E0": E0, "delta": delta, "fit_rms": fit_rms,
                "zero_parameter_null_value": null,
                "held_out_median_relative_error":
                    float(np.median(list(relative.values()))),
                "zero_parameter_null_median_relative_error":
                    float(np.median(list(relative_null.values()))),
                "per_arm_relative_error": relative,
                "superseded_by_this_amendment": {
                    "tested_on": superseded["tested_on"],
                    "held_out_median_relative_error":
                        superseded["held_out_median_relative_error"],
                    "zero_parameter_null_median_relative_error":
                        superseded["zero_parameter_null_median_relative_error"],
                },
            }
            # a same-norm counterexample: two arms with identical assembled norm
            # and different error are, by themselves, a bound on any law of N alone
            pairs = []
            for a in test:
                for b in test:
                    if a >= b:
                        continue
                    if abs(norms[a]["N_rms"] - norms[b]["N_rms"]) \
                            <= 1.0e-9 * max(norms[a]["N_rms"], TINY):
                        pairs.append({
                            "arms": [a, b], "N": norms[a]["N_rms"],
                            "relative_rms": [scores[a]["relative_rms"],
                                             scores[b]["relative_rms"]],
                            "error_ratio": (max(scores[a]["absolute_rms"],
                                                scores[b]["absolute_rms"])
                                            / max(min(scores[a]["absolute_rms"],
                                                      scores[b]["absolute_rms"]), TINY)),
                        })
            record[reference]["equal_norm_pairs"] = pairs
        result["surfaces"][sname] = record

    out = ROOT / "codes/results" / f"norm_law_amendment_l0_{args.stamp}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    print("wrote", out.name)
    for sname, record in result["surfaces"].items():
        for reference, values in record.items():
            print(f"{sname}/{reference}: delta = {values['delta']:.4f}, "
                  f"held-out median {values['held_out_median_relative_error']:.3f} "
                  f"(was {values['superseded_by_this_amendment']['held_out_median_relative_error']:.3f}"
                  f" over {len(values['superseded_by_this_amendment']['tested_on'])} arms), "
                  f"null {values['zero_parameter_null_median_relative_error']:.3f}, "
                  f"{len(values['tested_on'])} test arms, "
                  f"{len(values['equal_norm_pairs'])} equal-norm pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
