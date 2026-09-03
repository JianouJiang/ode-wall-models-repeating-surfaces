#!/usr/bin/env python3
"""Build executable source benchmarks for referee row R1-SCI-2.

The earlier registry verified names and term masks but treated Yang's integral
model as a supplied-traction rearrangement and Park--Moin as a residual only.
This producer instead exercises three output-bearing operators:

* Hickel et al.'s ODE with the source's A+=17 coefficient;
* Yang et al.'s published fully rough profile/moment equations (20)--(22);
* Park & Moin's dynamic closure (9) and an implicit wall-layer traction step.

Matched-operator interventions remain in the R2-m4 ladder and are deliberately
not relabelled as published models here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from models.source_faithful_wall_models import (  # noqa: E402
    HICKEL_VAN_DRIEST_A,
    hickel_source,
    hickel_wall_stress,
    ode_velocity,
    park_moin_dynamic_eddy_viscosity,
    park_moin_wall_layer_step,
    shoot_wall_stress,
    yang_rough_integral_wall_stress,
)


RESULTS = ROOT / "codes" / "results"
MODULE = ROOT / "codes" / "models" / "source_faithful_wall_models.py"
JSON_OUT = RESULTS / "source_operator_benchmarks_r1sci2.json"
NPZ_OUT = RESULTS / "source_operator_benchmarks_r1sci2.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Hickel: manufacture the matching velocity with A+=17, then recover the
    # wall traction through an independent all-bracket shooting call.
    nu = 1.0e-5
    y_m = 0.03
    dp_ds = 0.04
    hickel_tau_exact = 0.003
    source = lambda y: hickel_source(y, dp_ds, nu)
    hickel_u_m = ode_velocity(
        hickel_tau_exact, y_m, nu, source, n_points=1600,
        a_plus=HICKEL_VAN_DRIEST_A,
    )
    hickel = hickel_wall_stress(
        hickel_u_m, y_m, nu, dp_ds,
        continuation_tau=hickel_tau_exact, n_points=1600,
    )
    hickel_wrong_a = shoot_wall_stress(
        hickel_u_m, y_m, nu, source,
        continuation_tau=hickel_tau_exact, n_points=1600, a_plus=26.0,
    )

    # Yang: the equilibrium branch must collapse exactly to the fully rough
    # log law. A non-zero moment then exercises both equations (21) and (22).
    yang_u = 5.0
    yang_delta = 1.0
    yang_y0 = 0.01
    yang_equilibrium = yang_rough_integral_wall_stress(
        yang_u, yang_delta, yang_y0, 0.0,
    )
    yang_moment = 0.05
    yang_nonequilibrium = yang_rough_integral_wall_stress(
        yang_u, yang_delta, yang_y0, yang_moment,
        continuation_u_tau=yang_equilibrium.u_tau,
    )
    yang_y = np.linspace(yang_y0, yang_delta, 257)
    yang_profile = yang_nonequilibrium.u_tau * (
        np.log(yang_y / yang_y0) / 0.40
        + yang_nonequilibrium.linear_coefficient
        * (yang_y - yang_y0) / yang_delta
    )
    yang_log_tau = (0.40 * yang_u / np.log(yang_delta / yang_y0)) ** 2

    # Park--Moin equation (9): retain a full deviatoric tensor contraction.
    strain = np.asarray([[1.0, 0.2], [0.2, -1.0]])
    resolved_stress = np.asarray([[-0.3, 0.1], [0.1, 0.2]])
    mu_star = 0.5
    density = 1.0
    park_mu = float(park_moin_dynamic_eddy_viscosity(
        mu_star, density, resolved_stress, strain,
        clip_negative=False,
    ))

    # A quadratic pressure/body/convection balance is an exact solution of the
    # centred wall-normal operator; it makes the returned traction auditable.
    park_y = np.linspace(0.0, 1.0, 65)
    park_nu = 0.1
    park_convection = 0.3
    park_pressure = 0.5
    park_body = 2.8
    park_net_source = -park_convection - park_pressure + park_body
    park_exact_velocity = (
        park_net_source * park_y * (1.0 - park_y) / (2.0 * park_nu)
    )
    park = park_moin_wall_layer_step(
        park_exact_velocity, park_y, 0.4, park_nu, 0.0, 0.0,
        convective_term=park_convection,
        pressure_gradient=park_pressure,
        volume_force=park_body,
    )
    park_drop_convection = park_moin_wall_layer_step(
        park_exact_velocity, park_y, 0.4, park_nu, 0.0, 0.0,
        convective_term=0.0,
        pressure_gradient=park_pressure,
        volume_force=park_body,
    )
    park_wrong_pressure_sign = park_moin_wall_layer_step(
        park_exact_velocity, park_y, 0.4, park_nu, 0.0, 0.0,
        convective_term=park_convection,
        pressure_gradient=-park_pressure,
        volume_force=park_body,
    )
    park_tau_exact = 0.5 * park_net_source

    checks = {
        "hickel_a_plus_is_17": HICKEL_VAN_DRIEST_A == 17.0,
        "hickel_recovers_manufactured_traction": (
            abs(hickel.tau_w - hickel_tau_exact) <= 2.0e-10
            and abs(hickel.residual) <= 2.0e-10
        ),
        "hickel_wrong_a_plus_fixture_moves_output": (
            abs(hickel_wrong_a.tau_w - hickel.tau_w) >= 1.0e-5
        ),
        "yang_equilibrium_reduces_to_log_law": (
            abs(yang_equilibrium.tau_w - yang_log_tau) <= 2.0e-13
            and abs(yang_equilibrium.linear_coefficient) <= 2.0e-13
        ),
        "yang_profile_matches_les_velocity": (
            abs(yang_profile[-1] - yang_u) <= 2.0e-12
        ),
        "yang_nonequilibrium_closes_both_equations": (
            abs(yang_nonequilibrium.matching_residual) <= 2.0e-12
            and abs(yang_nonequilibrium.momentum_residual) <= 2.0e-12
            and len(yang_nonequilibrium.roots) >= 1
        ),
        "park_dynamic_coefficient_is_finite": np.isfinite(park_mu),
        "park_wall_layer_linear_solve_closes": park.linear_residual <= 2.0e-12,
        "park_returns_exact_quadratic_wall_traction": (
            abs(park.tau_w - park_tau_exact) <= 2.0e-12
            and np.max(np.abs(park.velocity - park_exact_velocity)) <= 2.0e-12
        ),
        "park_drop_convection_fixture_moves_output": (
            abs(park_drop_convection.tau_w - park.tau_w) >= 1.0e-2
        ),
        "park_pressure_sign_fixture_moves_output": (
            abs(park_wrong_pressure_sign.tau_w - park.tau_w) >= 1.0e-2
        ),
    }

    checks = {name: bool(value) for name, value in checks.items()}
    record = {
        "schema": "source-operator-benchmarks-r1sci2-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "purpose": (
            "Output-bearing source benchmarks; matched-operator causal "
            "instruments are excluded and remain in the R2-m4 ladder."
        ),
        "source_module": str(MODULE.relative_to(ROOT)),
        "source_module_sha256": sha256(MODULE),
        "operators": {
            "hickel_2013": {
                "source": "Hickel et al. (2013), equations (3)--(4)",
                "closure_coefficient_a_plus": HICKEL_VAN_DRIEST_A,
                "inputs": {"u_m": hickel_u_m, "y_m": y_m,
                           "nu": nu, "dp_ds": dp_ds},
                "tau_exact": hickel_tau_exact,
                "tau_computed": hickel.tau_w,
                "velocity_residual": hickel.residual,
                "root_count": len(hickel.roots),
                "wrong_a_plus_tau": hickel_wrong_a.tau_w,
            },
            "yang_2015": {
                "source": "Yang et al. (2015), doi:10.1063/1.4908072, equations (20)--(22)",
                "branch": "fully rough high-Re profile/moment solve",
                "inputs": {"u_les": yang_u, "delta_y": yang_delta,
                           "roughness_length": yang_y0,
                           "moment_rate": yang_moment},
                "equilibrium_tau": yang_equilibrium.tau_w,
                "equilibrium_log_tau": yang_log_tau,
                "tau_computed": yang_nonequilibrium.tau_w,
                "u_tau": yang_nonequilibrium.u_tau,
                "linear_coefficient": yang_nonequilibrium.linear_coefficient,
                "matching_residual": yang_nonequilibrium.matching_residual,
                "momentum_residual": yang_nonequilibrium.momentum_residual,
                "positive_root_count": len(yang_nonequilibrium.roots),
                "forbidden_shortcut": "no supplied tau_matching input",
            },
            "park_moin_2014": {
                "source": "Park and Moin (2014), doi:10.1063/1.4861069, equations (2), (5), (9)",
                "numerics": "backward Euler; second-order wall-normal diffusion; no-slip molecular traction",
                "dynamic_mu_star": mu_star,
                "dynamic_mu": park_mu,
                "inputs": {"nu": park_nu, "dt": 0.4,
                           "convective_term": park_convection,
                           "pressure_gradient": park_pressure,
                           "volume_force": park_body},
                "tau_exact": park_tau_exact,
                "tau_computed": park.tau_w,
                "linear_residual": park.linear_residual,
                "drop_convection_tau": park_drop_convection.tau_w,
                "wrong_pressure_sign_tau": park_wrong_pressure_sign.tau_w,
            },
        },
        "checks": checks,
    }
    JSON_OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        NPZ_OUT,
        schema=np.asarray(record["schema"]),
        source_module_sha256=np.asarray(record["source_module_sha256"]),
        hickel_roots=np.asarray(hickel.roots),
        yang_roots=np.asarray(yang_nonequilibrium.roots),
        yang_y=yang_y,
        yang_profile=yang_profile,
        park_y=park_y,
        park_velocity=park.velocity,
        park_exact_velocity=park_exact_velocity,
        check_names=np.asarray(list(checks)),
        check_values=np.asarray(list(checks.values()), dtype=np.int8),
    )
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"wrote {NPZ_OUT.relative_to(ROOT)}")
    print(f"R1-SCI-2 source operators: {sum(checks.values())}/{len(checks)} checks passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
