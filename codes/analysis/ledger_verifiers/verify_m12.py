#!/usr/bin/env python3
"""Independent verifier for the M12 common-observable rough-wall benchmark."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
JSON_PATH = RESULTS / "rough_wall_common_observable_m12.json"
NPZ_PATH = RESULTS / "rough_wall_common_observable_m12.npz"
sys.path.insert(0, str(ROOT / "codes"))

from models.source_faithful_wall_models import meneveau_rough_wall_stress  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_valid(document: dict) -> bool:
    reference = document.get("reference", {})
    calibration = document.get("roughness_calibration", {})
    surface = document.get("matching_surface", {})
    planes = np.asarray(surface.get("planes_over_h", []), dtype=float)
    return bool(
        document.get("status") == "PASS"
        and reference.get("uses_local_viscous_shear") is False
        and "pressure plus molecular force" in reference.get("kind", "")
        and calibration.get("independent_of_present_wrles") is True
        and len(planes) >= 3
        and np.all(planes > float(surface.get("published_rsl_top_over_h", math.inf)))
        and surface.get("input") == "intrinsic period-mean U"
    )


def main() -> int:
    if not JSON_PATH.exists() or not NPZ_PATH.exists():
        raise SystemExit("M12 FAIL: production artifact is missing")
    doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    data = np.load(NPZ_PATH, allow_pickle=False)
    checks: list[tuple[str, bool]] = []
    checks.append(("schema/status", doc.get("schema") == "rough-wall-common-observable-m12-v1"
                   and str(data["schema"]) == doc.get("schema") and doc.get("status") == "PASS"))
    checks.append(("common-observable contract", contract_valid(doc)))
    checks.append(("registered geometry", doc.get("case") == "r24_cube_staggered_G1"
                   and doc["geometry_contract"]["plan_area_density"] == 0.25))

    for relpath, expected in doc.get("source_hashes", {}).items():
        path = ROOT / relpath
        checks.append((f"source hash {relpath}", path.exists() and sha256(path) == expected))

    names = [str(item) for item in data["window_names"]]
    planes = np.asarray(data["matching_planes_over_h"], dtype=float)
    calibrations = [str(item) for item in data["calibration_names"]]
    d_over_h = np.asarray(data["d_over_h"], dtype=float)
    z0_over_h = np.asarray(data["z0_over_h"], dtype=float)
    velocity = np.asarray(data["intrinsic_u"], dtype=float)
    prediction = np.asarray(data["tau_prediction"], dtype=float)
    error = np.asarray(data["tau_relative_error"], dtype=float)
    tau_reference = float(data["tau_reference"])
    checks.append(("array shapes", velocity.shape == (len(names), len(planes))
                   and prediction.shape == (len(names), len(planes), len(calibrations))))
    checks.append(("window independence", len(names) >= 3 and any(name.startswith("disj_") for name in names)))
    checks.append(("published plane guard", np.all(planes > 1.85) and np.min(planes) < 2.0 < np.max(planes)))
    checks.append(("published calibration", calibrations[0] == "pressure_drag_xwire"
                   and math.isclose(d_over_h[0], 0.83, abs_tol=1e-14)
                   and math.isclose(z0_over_h[0], 0.0535, abs_tol=1e-14)))

    campaign_rel = next(path for path in doc["source_hashes"] if "r2_4_m20_les_" in path)
    provenance_rel = next(path for path in doc["source_hashes"] if path.endswith("PROVENANCE.json"))
    campaign = json.loads((ROOT / campaign_rel).read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / provenance_rel).read_text(encoding="utf-8"))
    case = campaign["cases"][doc["case"]]
    h, nu, u_tau = (float(provenance[key]) for key in ("h", "nu", "u_tau"))
    dp_ds = -float(provenance["body_force_gx"])

    rebuilt_u = np.empty_like(velocity)
    for wi, name in enumerate(names):
        profile = case["windows"][name]["mean_profile"]
        y = np.asarray(profile["y"], dtype=float) / h
        u = np.asarray(profile["U_over_utau"], dtype=float) * u_tau
        rebuilt_u[wi] = np.interp(planes, y, u)
    checks.append(("intrinsic velocity rebuild", np.allclose(rebuilt_u, velocity, rtol=2e-13, atol=2e-13)))

    forces = case["drag"]["forces"]
    area = float(provenance["mesh"]["A_plan"])
    force_components = np.asarray([
        float(forces["forcesCube"]["pressure_x"]),
        float(forces["forcesCube"]["viscous_x"]),
        float(forces["forcesFloor"]["viscous_x"]),
    ])
    # The former check required every component to resist the flow.  The
    # terminal staggered-cube WRLES falsifies that: its plan-mean floor viscous
    # force is negative (a small thrust, reversed shear over most of the
    # inter-cube floor).  Enforce the invariant that was actually intended --
    # the TOTAL is a resistance along the drive -- and bound, rather than
    # forbid, any opposing term.
    total_force = float(np.sum(force_components))
    opposing = force_components[force_components * total_force < 0.0]
    checks.append(("direct force totals to a resistance", total_force > 0.0))
    checks.append(("any opposing wall-force component is below 5 % of the total",
                   opposing.size == 0 or float(np.max(np.abs(opposing))) <= 0.05 * abs(total_force)))
    recorded_components = doc["reference"]["components"]
    checks.append(("signed component record matches the deposit", math.isclose(
        float(recorded_components["total_force_x"]), total_force, rel_tol=1e-12, abs_tol=1e-12)
        and all(math.isclose(float(recorded_components["terms"][n]["force_x"]), float(v),
                             rel_tol=1e-12, abs_tol=1e-12)
                for n, v in zip(("cube_pressure_x", "cube_viscous_x", "floor_viscous_x"),
                                force_components))
        and any(recorded_components["terms"][n]["opposes_resistance"]
                for n in recorded_components["terms"])))
    rebuilt_reference = abs(total_force) / area
    checks.append(("total plan-force rebuild", math.isclose(rebuilt_reference, tau_reference,
                                                            rel_tol=2e-13, abs_tol=2e-13)))

    rebuilt_prediction = np.empty_like(prediction)
    for wi in range(len(names)):
        for yi, y_over_h in enumerate(planes):
            for ci in range(len(calibrations)):
                rebuilt_prediction[wi, yi, ci] = meneveau_rough_wall_stress(
                    rebuilt_u[wi, yi], (y_over_h - d_over_h[ci]) * h,
                    nu, z0_over_h[ci] * h, dp_ds
                )
    checks.append(("source-model rebuild", np.allclose(rebuilt_prediction, prediction,
                                                       rtol=2e-13, atol=2e-13)))
    checks.append(("error rebuild", np.allclose((prediction - tau_reference) / tau_reference,
                                                error, rtol=2e-13, atol=2e-13)))
    checks.append(("finite published range", np.all(np.isfinite(prediction))
                   and np.all((z0_over_h[None, :] / (planes[:, None] - d_over_h[None, :]) > 1e-5)
                              & (z0_over_h[None, :] / (planes[:, None] - d_over_h[None, :]) < 0.1))))

    source = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    # This gate used to require the producer's FILE NAME to appear in the paper.
    # File names were removed from the body on the operator's register directive,
    # and proving content with a tool name is a defect in its own right: it
    # passes for a paper that names the script and says nothing, and fails for a
    # paper that describes the measurement completely.  It now tests the three
    # things the row is actually about --- the source benchmark, the model under
    # test, and the common observable both are evaluated on.
    flat = " ".join(source.split())
    checks.append(("manuscript source anchors", "chengcastro2002" in source
                   and "meneveau2020moody" in source
                   and "intrinsic period-mean velocity" in flat
                   and "plan-mean" in flat))

    # Red fixtures: each resurrects one defect of the invalid resolved-rib test.
    local_target = copy.deepcopy(doc)
    local_target["reference"]["uses_local_viscous_shear"] = True
    checks.append(("red fixture: local shear rejected", not contract_valid(local_target)))
    fitted = copy.deepcopy(doc)
    fitted["roughness_calibration"]["independent_of_present_wrles"] = False
    checks.append(("red fixture: fitted roughness rejected", not contract_valid(fitted)))
    inside = copy.deepcopy(doc)
    inside["matching_surface"]["planes_over_h"][0] = 1.5
    checks.append(("red fixture: sublayer plane rejected", not contract_valid(inside)))
    # A genuinely broken force decomposition -- an opposing term of the same
    # order as the resistance -- must still be rejected by the replacement of
    # the old all-same-sign guard.
    broken = force_components.copy()
    broken[2] = -0.30 * abs(total_force)
    broken_total = float(np.sum(broken))
    broken_opposing = broken[broken * broken_total < 0.0]
    checks.append(("red fixture: large opposing force component rejected",
                   broken_opposing.size > 0
                   and float(np.max(np.abs(broken_opposing))) > 0.05 * abs(broken_total)))

    failed = [name for name, passed in checks if not passed]
    if failed:
        for name in failed:
            print(f"FAIL {name}")
        print(f"M12: {len(checks) - len(failed)}/{len(checks)} checks passed")
        return 1
    print(f"M12: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
