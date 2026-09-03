#!/usr/bin/env python3
"""Build the source-faithful M0--M5 registry and the real rough-wall test.

This is the Level-1 comparison contract.  It does not rank unavailable coupled
models.  It freezes equations, runtime inputs, outputs, numerical policies and
source hashes, verifies every reference operator, and evaluates the deployable
M0/M1/M2/M5 subset on the deposited 48-station d-type-rib WRLES profiles.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from models.source_faithful_wall_models import (  # noqa: E402
    distributed_force_moments,
    hickel_source,
    meneveau_rough_wall_stress,
    ode_velocity,
    shoot_wall_stress,
    spalding_uplus,
    spalding_wall_stress,
    wall_layer_pde_residual,
    yang_integral_wall_stress,
)


RESULTS = ROOT / "codes" / "results"
MODULE = ROOT / "codes" / "models" / "source_faithful_wall_models.py"
RIB = RESULTS / "rib_les_dtype_wall_profiles.npz"
JSON_OUT = RESULTS / "source_faithful_model_registry_l1.json"
NPZ_OUT = RESULTS / "source_faithful_model_registry_l1.npz"

COMMON_FIELDS = [
    "u_m", "y_m", "nu", "dp_ds", "tau_matching", "storage_rate",
    "momentum_flux", "z0", "wall_layer_state", "volume_force",
]
OUTPUT = "signed kinematic wall traction tau_w"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "M0",
            "name": "equilibrium Spalding wall law",
            "source": "Spalding (1961), doi:10.1115/1.3641728",
            "equation": "y+ = u+ + exp(-kappa B)[exp(kappa u+)-1-kappa u+-(kappa u+)^2/2-(kappa u+)^3/6]",
            "runtime_inputs": ["u_m", "y_m", "nu"],
            "retained_terms": ["equilibrium local stress"],
            "matching_surface": "physical wall-normal distance y_m",
            "numerics": "one scalar Brent solve; sign(tau_w)=sign(u_m)",
            "failure_policy": "fatal if the positive friction-velocity root is not bracketed",
            "complexity": "O(N_root) scalar residual evaluations per face",
            "implementation": "codes/models/source_faithful_wall_models.py:spalding_wall_stress",
        },
        {
            "id": "M1",
            "name": "pressure-gradient-retaining TBLE ODE",
            "source": "Balaras et al. (1996), doi:10.2514/3.13200; Mukha et al. (2019), doi:10.1016/j.cpc.2019.01.016",
            "equation": "d tau/dy = dp/ds; tau=(nu+nu_t)dU/dy",
            "runtime_inputs": ["u_m", "y_m", "nu", "dp_ds"],
            "retained_terms": ["streamwise pressure gradient", "wall-normal stress divergence"],
            "matching_surface": "physical wall-normal distance y_m",
            "numerics": "wall-clustered trapezoidal integration plus audited all-bracket Brent shooting",
            "failure_policy": "fatal on no root; multiple roots reported and selected by declared continuation state",
            "complexity": "O(N_y N_root) operations per face",
            "implementation": "codes/models/source_faithful_wall_models.py:shoot_wall_stress",
        },
        {
            "id": "M2",
            "name": "Hickel parametrised-convection ODE",
            "source": "Hickel et al. (2013), equation (4.1)",
            "equation": "d tau/dy = (dp/ds)[1-min(y/y_pg,1)], y_pg(nu^2/|dp/ds|)^(-1/3)=4",
            "runtime_inputs": ["u_m", "y_m", "nu", "dp_ds"],
            "retained_terms": ["streamwise pressure gradient", "modelled convection", "wall-normal stress divergence"],
            "matching_surface": "physical wall-normal distance y_m",
            "numerics": "same grid, closure and root policy as M1; only the source term changes",
            "failure_policy": "same as M1",
            "complexity": "O(N_y N_root) operations per face",
            "implementation": "codes/models/source_faithful_wall_models.py:hickel_source",
        },
        {
            "id": "M3",
            "name": "Yang integral/history wall model",
            "source": "Yang et al. (2015), doi:10.1063/1.4908072",
            "equation": "tau_w = tau_m - dL_s/dt - M_s",
            "runtime_inputs": ["tau_matching", "storage_rate", "momentum_flux"],
            "retained_terms": ["matching-plane traction", "resolved storage", "integrated momentum flux/history"],
            "matching_surface": "top of the integral wall-model layer",
            "numerics": "conservative algebraic traction update after profile-moment evolution",
            "failure_policy": "fatal on a non-finite moment or matching traction",
            "complexity": "O(N_y) moment reduction per face plus moment evolution",
            "implementation": "codes/models/source_faithful_wall_models.py:yang_integral_wall_stress",
        },
        {
            "id": "M4",
            "name": "unsteady wall-layer PDE model",
            "source": "Park and Moin (2014), doi:10.1063/1.4861069",
            "equation": "dU_s/dt + div(U_s U) + dp/ds - d(tau_sn)/dn - f_s = 0",
            "runtime_inputs": ["wall_layer_state", "dp_ds", "volume_force"],
            "retained_terms": ["resolved storage", "resolved convection", "streamwise pressure gradient", "wall-normal stress divergence"],
            "matching_surface": "Dirichlet velocity at the wall-layer outer boundary",
            "numerics": "time-advanced wall-parallel PDE; reference residual verified by manufactured balance",
            "failure_policy": "PDE residual and nonlinear convergence must be reported, never replaced by zero traction",
            "complexity": "O(N_x N_y N_z N_iter) per wall-layer update",
            "implementation": "codes/models/source_faithful_wall_models.py:wall_layer_pde_residual",
        },
        {
            "id": "M5",
            "name": "Meneveau generalized-Moody rough-wall model",
            "source": "Meneveau (2020), doi:10.1080/14685248.2020.1840573, equations (7)--(8), (40)--(51)",
            "equation": "Re_tauDelta=[Re_tauDelta,pres^6+(Re_Delta Theta_fit)^6]^(1/6); tau_w=sign(U_m)(nu Re_tauDelta/y_m)^2",
            "runtime_inputs": ["u_m", "y_m", "nu", "dp_ds", "z0"],
            "retained_terms": ["streamwise pressure-gradient fit", "roughness length", "viscous-to-fully-rough transition"],
            "matching_surface": "distance Delta=y_m above the roughness origin",
            "numerics": "explicit equations (49)--(51); no iteration",
            "failure_policy": "reject inputs outside Re_Delta, psi_p and z0/Delta ranges printed by the source",
            "complexity": "O(1) elementary operations per face",
            "implementation": "codes/models/source_faithful_wall_models.py:meneveau_rough_wall_stress",
        },
    ]


def instruments() -> list[dict[str, object]]:
    return [
        {
            "id": "Xs",
            "name": "exact-stress substitution",
            "role": "causal instrument, not deployable model",
            "runtime_inputs": ["u_m", "y_m", "nu", "dp_ds", "tau_matching"],
            "implementation": "codes/analysis/diagnostic_test_corrected.py",
        },
        {
            "id": "Xc",
            "name": "exact-convection substitution",
            "role": "causal instrument, not deployable model",
            "runtime_inputs": ["u_m", "y_m", "nu", "dp_ds", "wall_layer_state"],
            "implementation": "codes/openfoam/pehill_wmles/wallmodel_tble/tbleShootCRWM.H",
        },
        {
            "id": "D0",
            "name": "distributed volume-force comparator",
            "role": "non-local representation preserving force moments",
            "runtime_inputs": ["wall_layer_state", "volume_force"],
            "implementation": "codes/analysis/physical_face_force_migration.py",
        },
    ]


def validate_registry(specs: list[dict[str, object]]) -> None:
    ids = [str(spec["id"]) for spec in specs]
    if ids != [f"M{index}" for index in range(6)]:
        raise AssertionError("registry must contain ordered M0--M5 exactly once")
    input_sets: list[frozenset[str]] = []
    term_sets: list[frozenset[str]] = []
    for spec in specs:
        inputs = frozenset(str(value) for value in spec["runtime_inputs"])
        terms = frozenset(str(value) for value in spec["retained_terms"])
        if not inputs.issubset(COMMON_FIELDS):
            raise AssertionError(f"{spec['id']} requests an undeclared observation")
        if not terms:
            raise AssertionError(f"{spec['id']} has no retained-term declaration")
        input_sets.append(inputs)
        term_sets.append(terms)
    if len(set(term_sets)) != len(term_sets):
        raise AssertionError("two model labels hide the same retained-term set")
    if "dp_ds" in input_sets[0] or "dp_ds" not in input_sets[1]:
        raise AssertionError("M0/M1 pressure-input distinction is not locked")
    if "modelled convection" not in term_sets[2]:
        raise AssertionError("M2 must add modelled convection")
    if "integrated momentum flux/history" not in term_sets[3]:
        raise AssertionError("M3 must receive an actual history/flux state")
    if "resolved convection" not in term_sets[4]:
        raise AssertionError("M4 must retain resolved convection")
    if "z0" not in input_sets[5] or "roughness length" not in term_sets[5]:
        raise AssertionError("M5 must receive roughness rather than a hidden smooth-wall proxy")


def manufactured_verification() -> dict[str, float]:
    errors: dict[str, float] = {}
    nu = 1.0e-5
    y_m = 0.03

    # M0: create U_m from a prescribed friction velocity and recover tau_w.
    u_tau = 0.045
    y_plus = y_m * u_tau / nu
    u_m = u_tau * spalding_uplus(y_plus)
    errors["M0_tau_abs_error"] = abs(
        spalding_wall_stress(u_m, y_m, nu) - u_tau * u_tau
    )

    # M1 and M2: synthesize U_m with their own source but solve independently.
    tau_exact = 0.003
    dp_ds = 0.04
    source_m1 = lambda y: np.full_like(y, dp_ds, dtype=float)
    u_m1 = ode_velocity(tau_exact, y_m, nu, source_m1, n_points=900)
    m1 = shoot_wall_stress(u_m1, y_m, nu, source_m1,
                           continuation_tau=tau_exact, n_points=900)
    errors["M1_tau_abs_error"] = abs(m1.tau_w - tau_exact)
    source_m2 = lambda y: hickel_source(y, dp_ds, nu)
    u_m2 = ode_velocity(tau_exact, y_m, nu, source_m2, n_points=900)
    m2 = shoot_wall_stress(u_m2, y_m, nu, source_m2,
                           continuation_tau=tau_exact, n_points=900)
    errors["M2_tau_abs_error"] = abs(m2.tau_w - tau_exact)

    # M3: conservative integral update.
    tau_matching, storage, flux = 0.011, -0.0025, 0.004
    expected_m3 = 0.0095
    errors["M3_tau_abs_error"] = abs(
        yang_integral_wall_stress(tau_matching, storage, flux) - expected_m3
    )

    # M4: a signed manufactured balance; every retained term is non-zero.
    residual = wall_layer_pde_residual(0.2, -0.3, 0.7, 0.1, 0.5)
    errors["M4_residual_abs"] = abs(float(residual))

    # M5: high-Re ZPG must approach the fully rough log-law asymptote.
    u_m5, delta5, nu5, z05 = 5.0, 1.0, 1.0e-6, 0.01
    tau_m5 = meneveau_rough_wall_stress(u_m5, delta5, nu5, z05, 0.0)
    tau_log = (0.4 * u_m5 / np.log(delta5 / z05)) ** 2
    errors["M5_fully_rough_relative_error"] = abs(tau_m5 - tau_log) / tau_log

    # D0: direct quadrature of a signed force distribution and its moment.
    y = np.linspace(0.0, 1.0, 2001)
    force = 2.0 + 3.0 * y
    zeroth, first = distributed_force_moments(y, force)
    errors["D0_zeroth_abs_error"] = abs(zeroth - 3.5)
    errors["D0_first_abs_error"] = abs(first - 2.0)
    return errors


def score(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    return {
        "r2_descriptive": float(1.0 - np.sum((prediction - reference) ** 2) / denominator),
        "relative_rms": float(np.sqrt(np.mean((prediction - reference) ** 2)) / scale),
        "signed_drag_error": float(np.sum(prediction - reference) / np.sum(np.abs(reference))),
        "sign_mismatch_fraction": float(np.mean(np.signbit(prediction) != np.signbit(reference))),
    }


def rib_test() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(RIB, allow_pickle=False) as data:
        y = np.asarray(data["y"], dtype=float)
        u = np.asarray(data["U"], dtype=float)
        reference = np.asarray(data["tau_w"], dtype=float)
        dp_ds = np.asarray(data["dp_dx"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
        nu = float(data["nu"])

    # The deposited geometry has k/delta=0.2.  The source's fully rough
    # equivalence ks_infty=30 z0 is applied once, before looking at tau_w:
    # ks_infty=k -> z0=k/30.  Index 15 is the lowest stored plane for which
    # every station meets z0/y_m<0.1; no station is dropped.
    k = 0.2
    z0 = k / 30.0
    matching_index = 15
    y_m = y[:, matching_index]
    u_m = u[:, matching_index]
    if not np.all((z0 / y_m > 1.0e-5) & (z0 / y_m < 0.1)):
        raise AssertionError("registered rib matching plane violates M5 roughness range")

    predictions = {name: np.full(reference.shape, np.nan)
                   for name in ("M0", "M1", "M2", "M5")}
    root_counts = {name: np.zeros(reference.shape, dtype=int)
                   for name in ("M1", "M2")}
    elapsed: dict[str, float] = {}

    start = time.perf_counter()
    for index in range(reference.size):
        predictions["M0"][index] = spalding_wall_stress(
            u_m[index], y_m[index], nu
        )
    elapsed["M0"] = time.perf_counter() - start

    for model in ("M1", "M2"):
        start = time.perf_counter()
        for index in range(reference.size):
            if model == "M1":
                source = lambda yy, value=dp_ds[index]: np.full_like(
                    yy, value, dtype=float
                )
            else:
                source = lambda yy, value=dp_ds[index]: hickel_source(
                    yy, value, nu
                )
            result = shoot_wall_stress(
                u_m[index], y_m[index], nu, source,
                continuation_tau=predictions["M0"][index], n_points=500
            )
            predictions[model][index] = result.tau_w
            root_counts[model][index] = len(result.roots)
        elapsed[model] = time.perf_counter() - start

    start = time.perf_counter()
    for index in range(reference.size):
        predictions["M5"][index] = meneveau_rough_wall_stress(
            u_m[index], y_m[index], nu, z0, dp_ds[index]
        )
    elapsed["M5"] = time.perf_counter() - start

    metrics = {name: score(reference, values)
               for name, values in predictions.items()}
    summary: dict[str, object] = {
        "case": "rib_les_dtype",
        "reference_fidelity": "deposited wall-resolved LES (WALE), 48 stations",
        "matching_index": matching_index,
        "station_count": int(reference.size),
        "common_valid_station_count": int(reference.size),
        "roughness_mapping": "specified before scoring: ks_infinity=k, z0=ks_infinity/30, k/delta=0.2",
        "z0": z0,
        "z0_over_y_m_min": float(np.min(z0 / y_m)),
        "z0_over_y_m_max": float(np.max(z0 / y_m)),
        "metrics": metrics,
        "reference_runtime_seconds": elapsed,
        "max_root_count": {name: int(np.max(counts))
                           for name, counts in root_counts.items()},
        "target_used_to_set_z0": False,
    }
    arrays = {
        "rib_x": x,
        "rib_y_m": y_m,
        "rib_u_m": u_m,
        "rib_dp_ds": dp_ds,
        "rib_tau_reference": reference,
        "rib_tau_M0": predictions["M0"],
        "rib_tau_M1": predictions["M1"],
        "rib_tau_M2": predictions["M2"],
        "rib_tau_M5": predictions["M5"],
        "rib_M1_root_count": root_counts["M1"],
        "rib_M2_root_count": root_counts["M2"],
    }
    return summary, arrays


def main() -> int:
    specs = model_specs()
    validate_registry(specs)
    verification = manufactured_verification()
    thresholds = {
        "M0_tau_abs_error": 1.0e-11,
        "M1_tau_abs_error": 2.0e-10,
        "M2_tau_abs_error": 2.0e-10,
        "M3_tau_abs_error": 1.0e-15,
        "M4_residual_abs": 1.0e-15,
        "M5_fully_rough_relative_error": 0.01,
        "D0_zeroth_abs_error": 1.0e-6,
        "D0_first_abs_error": 1.0e-6,
    }
    failures = [name for name, value in verification.items()
                if value > thresholds[name]]
    if failures:
        raise AssertionError("manufactured verification failed: " + ", ".join(failures))

    rib_summary, rib_arrays = rib_test()
    source_paths = [
        MODULE,
        ROOT / "codes" / "openfoam" / "xiao_wmles_a1p0" / "0" / "nut",
        ROOT / "codes" / "openfoam" / "pehill_wmles_tble" / "0" / "nut",
        ROOT / "codes" / "openfoam" / "pehill_wmles" / "wallmodel_tble" / "tbleShootCRWM.H",
        ROOT / "codes" / "analysis" / "physical_face_force_migration.py",
        RIB,
    ]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source-locked artifact: " + str(missing))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    registry = {
        "schema": "source-faithful-model-registry-l1-v1",
        "status": "PASS",
        "idea": "source-locked term-nested M0--M5 registry under one observation/output contract",
        "common_observation_fields": COMMON_FIELDS,
        "common_output": OUTPUT,
        "density_convention": "kinematic: every traction is divided by density",
        "pressure_direction": "streamwise/wall-tangential dp/ds; never wall-normal",
        "models": specs,
        "causal_instruments_and_comparator": instruments(),
        "manufactured_errors": verification,
        "manufactured_thresholds": thresholds,
        "rough_wall_real_data_test": rib_summary,
        "source_hashes": hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    np.savez(
        NPZ_OUT,
        schema=np.array(registry["schema"]),
        model_ids=np.asarray([spec["id"] for spec in specs]),
        common_observation_fields=np.asarray(COMMON_FIELDS),
        manufactured_names=np.asarray(list(verification)),
        manufactured_errors=np.asarray(list(verification.values()), dtype=float),
        manufactured_thresholds=np.asarray([thresholds[name] for name in verification],
                                           dtype=float),
        rib_matching_index=np.array(rib_summary["matching_index"]),
        rib_z0=np.array(rib_summary["z0"]),
        rib_z0_over_y_m_min=np.array(rib_summary["z0_over_y_m_min"]),
        rib_z0_over_y_m_max=np.array(rib_summary["z0_over_y_m_max"]),
        rib_source_sha256=np.array(hashes[str(RIB.relative_to(ROOT))]),
        **rib_arrays,
    )
    print("SOURCE-FAITHFUL MODEL REGISTRY: PASS")
    print(f"  manufactured checks: {len(verification)}/{len(verification)}")
    print("  real rough-wall test: 48/48 WRLES stations in published input range")
    for model, metrics in rib_summary["metrics"].items():
        print(f"  {model}: R2={metrics['r2_descriptive']:+.6f}, "
              f"relRMS={metrics['relative_rms']:.6f}, "
              f"sign-mismatch={metrics['sign_mismatch_fraction']:.3f}")
    print(f"  wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"  wrote {NPZ_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
