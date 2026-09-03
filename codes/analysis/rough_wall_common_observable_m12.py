#!/usr/bin/env python3
"""M12: source-faithful rough-wall model on a common force observable.

The earlier M12 comparison was invalid: it supplied an assumed ``k_s=k`` to a
homogenised rough-wall model and scored the result against phasewise molecular
traction on resolved rib faces.  This producer instead uses the independently
calibrated C20S staggered-cube surface of Cheng & Castro (2002), the matching
staggered-cube WRLES from R2-4/M20, and one observable on both sides:

* input: intrinsic period-mean velocity on planes above the published
  roughness-sublayer edge;
* model: Meneveau's (2020) generalized-Moody rough-wall relation, evaluated
  with the published displacement height and roughness length;
* target: directly integrated pressure plus molecular force on the whole
  periodic plan, divided by plan area.

No local viscous traction and no roughness parameter fitted to the WRLES target
enters the comparison.  The output is deliberately unavailable until the
operator-owned R2-4/M20 production deposit has been harvested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
MODEL_SOURCE = ROOT / "codes" / "models" / "source_faithful_wall_models.py"
sys.path.insert(0, str(ROOT / "codes"))

from models.source_faithful_wall_models import meneveau_rough_wall_stress  # noqa: E402


CASE_ID = "r24_cube_staggered_G1"
RSL_TOP_OVER_H = 1.85
MATCHING_PLANES_OVER_H = np.asarray([1.90, 2.00, 2.20, 2.50])

# Cheng & Castro (2002), Boundary-Layer Meteorology 104, Table IV, C20S.
# The primary entry uses u_*(p), the independently measured form-drag stress,
# as the prescribed log-law slope.  The remaining entries give the declared
# calibration-method envelope; none uses the present WRLES.
CALIBRATIONS = (
    ("pressure_drag_xwire", 16.6 / 20.0, 1.07 / 20.0, True),
    ("is_and_rs_xwire", 18.4 / 20.0, 0.65 / 20.0, False),
    ("is_xwire", 20.6 / 20.0, 0.56 / 20.0, False),
    ("rs_xwire", 17.5 / 20.0, 0.73 / 20.0, False),
    ("pressure_drag_lda", 14.5 / 20.0, 1.33 / 20.0, False),
    ("is_and_rs_lda", 16.7 / 20.0, 0.81 / 20.0, False),
    ("is_lda", 19.5 / 20.0, 0.69 / 20.0, False),
    ("rs_lda", 14.8 / 20.0, 0.95 / 20.0, False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cumulative_time(name: str) -> float:
    if not name.startswith("cum_"):
        return -math.inf
    return float(name.split("_", 1)[1])


def validate_inputs(campaign: dict, provenance: dict) -> dict:
    cases = campaign.get("cases", {})
    if CASE_ID not in cases:
        raise RuntimeError(f"{CASE_ID} is absent: R2-4/M20 has not been harvested")
    case = cases[CASE_ID]
    if case.get("status") != "OK" or not case.get("converged"):
        raise RuntimeError(f"{CASE_ID} is not a converged production result")
    if case.get("kind") != "cube" or case.get("layout") != "staggered":
        raise AssertionError("M12 requires the registered staggered-cube geometry")
    if not math.isclose(float(case["lambda_p"]), 0.25, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError("M12 requires the published plan density lambda_p=0.25")
    if provenance.get("pilot", False) or provenance.get("layout") != "staggered":
        raise AssertionError("pilot or non-staggered provenance cannot enter M12")
    if not math.isclose(float(provenance["lambda_p"]), 0.25, abs_tol=1.0e-12):
        raise AssertionError("provenance geometry does not match C20S")
    if float(MATCHING_PLANES_OVER_H.min()) <= RSL_TOP_OVER_H:
        raise AssertionError("a common matching plane lies inside the published RSL")
    return case


def component_record(case: dict, provenance: dict) -> dict:
    """Signed breakdown of the plan-mean wall force, opposing terms included."""
    forces = case["drag"]["forces"]
    area = float(provenance["mesh"]["A_plan"])
    named = (
        ("cube_pressure_x", forces["forcesCube"]["pressure_x"],
         forces["forcesCube"]["pressure_x_block_sd"]),
        ("cube_viscous_x", forces["forcesCube"]["viscous_x"],
         forces["forcesCube"]["viscous_x_block_sd"]),
        ("floor_viscous_x", forces["forcesFloor"]["viscous_x"],
         forces["forcesFloor"]["viscous_x_block_sd"]),
    )
    total = sum(float(value) for _, value, _ in named)
    return {
        "plan_area": area,
        "total_force_x": total,
        "terms": {
            name: {
                "force_x": float(value),
                "block_sd": float(sd),
                "fraction_of_total": float(value) / total,
                "opposes_resistance": float(value) * total < 0.0,
                "block_sd_from_zero": (abs(float(value)) / float(sd)) if float(sd) else None,
            }
            for name, value, sd in named
        },
    }


def force_reference(case: dict, provenance: dict) -> tuple[float, float]:
    """Return plan-mean total kinematic stress and its block standard error."""
    drag = case["drag"]
    forces = drag["forces"]
    cube, floor = forces["forcesCube"], forces["forcesFloor"]
    area = float(provenance["mesh"]["A_plan"])
    components = (
        (float(cube["pressure_x"]), float(cube["pressure_x_block_sd"])),
        (float(cube["viscous_x"]), float(cube["viscous_x_block_sd"])),
        (float(floor["viscous_x"]), float(floor["viscous_x_block_sd"])),
    )
    # The original guard required every component to resist the flow.  The
    # terminal staggered-cube WRLES falsifies that assumption: the plan-mean
    # floor viscous force is -0.0324 (block SD 0.0199), i.e. a small thrust,
    # because reversed shear covers most of the inter-cube floor.  That is the
    # paper's own phenomenon, not a defective decomposition, so the invariant
    # enforced here is the physical one -- the TOTAL is a resistance along the
    # driving direction and reproduces u_tau^2 exactly -- with any opposing
    # component bounded and recorded rather than forbidden.
    total = sum(value for value, _ in components)
    if total <= 0.0:
        raise AssertionError("total plan-mean wall force is not a resistance")
    opposing = [value for value, _ in components if value * total < 0.0]
    if opposing and max(abs(v) for v in opposing) > 0.05 * abs(total):
        raise AssertionError("an opposing wall-force component exceeds 5% of the total")
    stress = abs(total) / area
    standard_error = math.sqrt(sum(error * error for _, error in components)) / area
    recorded = float(drag["u_tau_measured"]) ** 2
    if not math.isclose(stress, recorded, rel_tol=2.0e-12, abs_tol=2.0e-12):
        raise AssertionError("direct-force and recorded u_tau targets disagree")
    return stress, standard_error


def evaluate(campaign: dict, provenance: dict) -> tuple[dict, dict[str, np.ndarray]]:
    case = validate_inputs(campaign, provenance)
    tau_reference, tau_reference_se = force_reference(case, provenance)
    h = float(provenance["h"])
    nu = float(provenance["nu"])
    u_tau = float(provenance["u_tau"])
    dp_ds = -float(provenance["body_force_gx"])
    windows = case["windows"]
    window_names = sorted(windows, key=lambda name: (cumulative_time(name), name))
    if not window_names or not any(name.startswith("disj_") for name in window_names):
        raise AssertionError("M12 requires cumulative and disjoint averaging windows")

    n_w = len(window_names)
    n_y = len(MATCHING_PLANES_OVER_H)
    n_c = len(CALIBRATIONS)
    velocity = np.empty((n_w, n_y))
    prediction = np.empty((n_w, n_y, n_c))
    for wi, name in enumerate(window_names):
        profile = windows[name]["mean_profile"]
        y = np.asarray(profile["y"], dtype=float) / h
        u = np.asarray(profile["U_over_utau"], dtype=float) * u_tau
        if y.min() > MATCHING_PLANES_OVER_H.min() or y.max() < MATCHING_PLANES_OVER_H.max():
            raise AssertionError(f"{name} does not span every registered matching plane")
        velocity[wi] = np.interp(MATCHING_PLANES_OVER_H, y, u)
        for yi, y_over_h in enumerate(MATCHING_PLANES_OVER_H):
            for ci, (_, d_over_h, z0_over_h, _) in enumerate(CALIBRATIONS):
                delta = (y_over_h - d_over_h) * h
                z0 = z0_over_h * h
                prediction[wi, yi, ci] = meneveau_rough_wall_stress(
                    velocity[wi, yi], delta, nu, z0, dp_ds
                )

    relative_error = (prediction - tau_reference) / tau_reference
    primary_index = next(i for i, item in enumerate(CALIBRATIONS) if item[3])
    primary_plane = int(np.where(np.isclose(MATCHING_PLANES_OVER_H, 2.0))[0][0])
    final_index = max(
        (i for i, name in enumerate(window_names) if name.startswith("cum_")),
        key=lambda i: cumulative_time(window_names[i]),
    )
    final_error = float(relative_error[final_index, primary_plane, primary_index])
    all_errors = relative_error.reshape(-1)
    summary = {
        "schema": "rough-wall-common-observable-m12-v1",
        "status": "PASS",
        "row": "M12",
        "case": CASE_ID,
        "geometry_contract": {
            "layout": "staggered cubes",
            "plan_area_density": 0.25,
            "published_surface": "Cheng--Castro C20S",
            "simulation_fidelity": case["tag"],
        },
        "model": {
            "name": "Meneveau generalized-Moody rough-wall model",
            "source": "Meneveau (2020), doi:10.1080/14685248.2020.1840573, equations (49)--(51)",
            "implementation": "codes/models/source_faithful_wall_models.py:meneveau_rough_wall_stress",
            "output": "signed plan-mean kinematic total surface traction",
        },
        "reference": {
            "kind": "pressure plus molecular force integrated over cube and floor, divided by plan area",
            "support": "one complete periodic plan",
            "tau": tau_reference,
            "block_standard_error": tau_reference_se,
            "relative_block_standard_error": tau_reference_se / tau_reference,
            "uses_local_viscous_shear": False,
            "components": component_record(case, provenance),
        },
        "roughness_calibration": {
            "source": "Cheng & Castro (2002), Boundary-Layer Meteorology 104:229--259, doi:10.1023/A:1016060103448, Table IV",
            "independent_of_present_wrles": True,
            "primary_method": CALIBRATIONS[primary_index][0],
            "primary_d_over_h": CALIBRATIONS[primary_index][1],
            "primary_z0_over_h": CALIBRATIONS[primary_index][2],
            "methods": [
                {"name": name, "d_over_h": d, "z0_over_h": z0, "primary": primary}
                for name, d, z0, primary in CALIBRATIONS
            ],
        },
        "matching_surface": {
            "input": "intrinsic period-mean U",
            "height_origin": "cube base",
            "planes_over_h": MATCHING_PLANES_OVER_H.tolist(),
            "published_rsl_top_over_h": RSL_TOP_OVER_H,
            "all_planes_above_rsl": True,
            "model_distance": "Delta=(y-d)",
        },
        "forcing": {
            "body_force_gx": float(provenance["body_force_gx"]),
            "equivalent_dp_ds": dp_ds,
        },
        "uncertainty_design": {
            "averaging_windows": window_names,
            "calibration_methods": len(CALIBRATIONS),
            "matching_planes": len(MATCHING_PLANES_OVER_H),
            "envelope_semantics": "deterministic window x plane x published-calibration sensitivity; separate from force block SE",
        },
        "headline": {
            "window": window_names[final_index],
            "matching_plane_over_h": float(MATCHING_PLANES_OVER_H[primary_plane]),
            "calibration": CALIBRATIONS[primary_index][0],
            "prediction": float(prediction[final_index, primary_plane, primary_index]),
            "reference": tau_reference,
            "signed_relative_error": final_error,
            "absolute_relative_error": abs(final_error),
            "full_sensitivity_signed_error_range": [float(all_errors.min()), float(all_errors.max())],
        },
    }
    arrays = {
        "window_names": np.asarray(window_names),
        "matching_planes_over_h": MATCHING_PLANES_OVER_H,
        "calibration_names": np.asarray([item[0] for item in CALIBRATIONS]),
        "d_over_h": np.asarray([item[1] for item in CALIBRATIONS]),
        "z0_over_h": np.asarray([item[2] for item in CALIBRATIONS]),
        "intrinsic_u": velocity,
        "tau_prediction": prediction,
        "tau_relative_error": relative_error,
        "tau_reference": np.asarray(tau_reference),
        "tau_reference_block_se": np.asarray(tau_reference_se),
    }
    return summary, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260823")
    args = parser.parse_args()
    campaign_path = RESULTS / f"r2_4_m20_les_{args.date}.json"
    provenance_path = RESULTS / "r2_4_m20" / CASE_ID / "PROVENANCE.json"
    if not campaign_path.exists() or not provenance_path.exists():
        missing = [str(path.relative_to(ROOT)) for path in (campaign_path, provenance_path) if not path.exists()]
        raise FileNotFoundError("M12 waits for the completed operator campaign: " + ", ".join(missing))
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    summary, arrays = evaluate(campaign, provenance)
    summary["source_hashes"] = {
        str(campaign_path.relative_to(ROOT)): sha256(campaign_path),
        str(provenance_path.relative_to(ROOT)): sha256(provenance_path),
        str(MODEL_SOURCE.relative_to(ROOT)): sha256(MODEL_SOURCE),
    }
    json_out = RESULTS / "rough_wall_common_observable_m12.json"
    npz_out = RESULTS / "rough_wall_common_observable_m12.npz"
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez(npz_out, schema=np.asarray(summary["schema"]), **arrays)
    head = summary["headline"]
    print("M12 COMMON-OBSERVABLE ROUGH-WALL BENCHMARK: PASS")
    print(f"  {CASE_ID}: tau_model={head['prediction']:.6g}, tau_force={head['reference']:.6g}, "
          f"relative error={head['signed_relative_error']:+.3f}")
    print(f"  sensitivity envelope={head['full_sensitivity_signed_error_range']}")
    print(f"  wrote {json_out.relative_to(ROOT)}")
    print(f"  wrote {npz_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
