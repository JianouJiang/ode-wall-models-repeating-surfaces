#!/usr/bin/env python3
"""Independent verifier for output-bearing Hickel/Yang/Park source models."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[3]
JSON_RESULT = ROOT / "codes/results/source_operator_benchmarks_r1sci2.json"
NPZ_RESULT = ROOT / "codes/results/source_operator_benchmarks_r1sci2.npz"
PRODUCER = ROOT / "codes/analysis/source_operator_benchmarks_r1sci2.py"
MODULE = ROOT / "codes/models/source_faithful_wall_models.py"
TEX = ROOT / "manuscript/main.tex"
PDF = ROOT / "manuscript/main.pdf"
checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_tex(text: str) -> str:
    kept: list[str] = []
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(r"\iffalse"):
            depth += 1
            continue
        if stripped.startswith(r"\fi") and depth:
            depth -= 1
            continue
        if not depth:
            kept.append(re.sub(r"(?<!\\)%.*$", "", line))
    if depth:
        raise RuntimeError("unclosed inactive TeX block")
    return "\n".join(kept)


def hickel_tau(u_m: float, y_m: float, nu: float, dp_ds: float,
                a_plus: float) -> float:
    eta = np.linspace(0.0, 1.0, 1600)
    y = y_m * eta ** 1.5
    y_pg = 4.0 * (nu * nu / abs(dp_ds)) ** (1.0 / 3.0)
    source = dp_ds * (1.0 - np.minimum(y / y_pg, 1.0))
    impulse = cumulative_trapezoid(source, y, initial=0.0)

    def velocity(tau: float) -> float:
        u_tau = np.sqrt(abs(tau))
        length = 0.41 * y * (1.0 - np.exp(-y * u_tau / nu / a_plus))
        stress = tau + impulse
        strain = np.sign(stress) * 2.0 * np.abs(stress) / (
            nu + np.sqrt(nu * nu + 4.0 * length * length * np.abs(stress))
        )
        return float(trapezoid(strain, y))

    candidates = np.concatenate((
        -np.logspace(3, -12, 500), np.asarray([0.0]),
        np.logspace(-12, 3, 500),
    ))
    values = np.asarray([velocity(value) - u_m for value in candidates])
    roots = [brentq(lambda value: velocity(value) - u_m, left, right,
                    xtol=1.0e-13, rtol=1.0e-13)
             for left, right, f_left, f_right in zip(
                 candidates[:-1], candidates[1:], values[:-1], values[1:])
             if f_left * f_right < 0.0]
    if not roots:
        raise RuntimeError("independent Hickel scan found no root")
    return float(min(roots, key=lambda value: abs(value - 0.003)))


def yang_roots(u_les: float, delta: float, y0: float,
               moment: float) -> tuple[list[float], list[float]]:
    kappa = 0.40
    log_coefficient = np.log(delta / y0) / kappa
    fraction = 1.0 - y0 / delta
    c = kappa / fraction
    polynomial = [
        c * c * log_coefficient ** 2 - 2.0 * c * log_coefficient,
        -2.0 * c * c * u_les * log_coefficient + 2.0 * c * u_les,
        c * c * u_les ** 2 - moment,
    ]
    roots = sorted(float(root.real) for root in np.roots(polynomial)
                   if abs(root.imag) < 1.0e-11 and root.real > 0.0)
    coefficients = [
        (u_les / root - log_coefficient) / fraction for root in roots
    ]
    return roots, coefficients


def park_rebuild(operator: dict[str, object]) -> tuple[float, float, float]:
    values = operator["inputs"]
    nu = float(values["nu"])
    dt = float(values["dt"])
    convection = float(values["convective_term"])
    pressure = float(values["pressure_gradient"])
    body = float(values["volume_force"])
    y = np.linspace(0.0, 1.0, 65)
    dy = y[1] - y[0]
    net = -convection - pressure + body
    old = net * y * (1.0 - y) / (2.0 * nu)
    matrix = np.zeros((y.size, y.size))
    rhs = old + dt * net
    matrix[0, 0] = matrix[-1, -1] = 1.0
    rhs[0] = rhs[-1] = 0.0
    coefficient = dt * nu / dy ** 2
    for index in range(1, y.size - 1):
        matrix[index, index - 1:index + 2] = (
            -coefficient, 1.0 + 2.0 * coefficient, -coefficient
        )
    velocity = np.linalg.solve(matrix, rhs)
    tau = nu * (4.0 * velocity[1] - velocity[2]) / (2.0 * dy)

    strain = np.asarray([[1.0, 0.2], [0.2, -1.0]])
    reynolds = np.asarray([[-0.3, 0.1], [0.1, 0.2]])
    dynamic_mu = (0.5 + np.sum(reynolds * strain)
                  / (2.0 * np.sum(strain * strain)))
    return float(tau), float(np.max(np.abs(velocity - old))), float(dynamic_mu)


def function_arguments(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [argument.arg for argument in node.args.args]
    raise KeyError(name)


def main() -> int:
    required = [JSON_RESULT, NPZ_RESULT, PRODUCER, MODULE, TEX, PDF]
    check("all source-operator artifacts exist", all(path.exists() for path in required))
    if not all(path.exists() for path in required):
        return 2
    record = json.loads(JSON_RESULT.read_text(encoding="utf-8"))
    with np.load(NPZ_RESULT, allow_pickle=False) as arrays:
        check("artifact schema, status and source hash are current",
              record.get("schema") == "source-operator-benchmarks-r1sci2-v1"
              and record.get("status") == "PASS"
              and str(arrays["schema"]) == record["schema"]
              and record["source_module_sha256"] == sha256(MODULE)
              and str(arrays["source_module_sha256"]) == sha256(MODULE))
        check("producer records eleven green operator/fixture checks",
              len(record["checks"]) == 11
              and all(record["checks"].values())
              and np.all(np.asarray(arrays["check_values"]) == 1))

        hickel = record["operators"]["hickel_2013"]
        inputs = hickel["inputs"]
        rebuilt_h17 = hickel_tau(float(inputs["u_m"]), float(inputs["y_m"]),
                                 float(inputs["nu"]), float(inputs["dp_ds"]), 17.0)
        rebuilt_h26 = hickel_tau(float(inputs["u_m"]), float(inputs["y_m"]),
                                 float(inputs["nu"]), float(inputs["dp_ds"]), 26.0)
        check("independent Hickel A+=17 solve reproduces signed traction",
              hickel["closure_coefficient_a_plus"] == 17.0
              and abs(rebuilt_h17 - hickel["tau_computed"]) < 2.0e-11
              and abs(rebuilt_h17 - hickel["tau_exact"]) < 2.0e-10)
        check("coefficient control case rejects inherited A+=26",
              abs(rebuilt_h26 - rebuilt_h17) > 1.0e-5
              and abs(rebuilt_h26 - hickel["wrong_a_plus_tau"]) < 2.0e-11)

        yang = record["operators"]["yang_2015"]
        y_inputs = yang["inputs"]
        roots, linear = yang_roots(
            float(y_inputs["u_les"]), float(y_inputs["delta_y"]),
            float(y_inputs["roughness_length"]),
            float(y_inputs["moment_rate"]),
        )
        branch = int(np.argmin(np.abs(np.asarray(roots)
                                     - np.sqrt(yang["equilibrium_tau"]))))
        selected = roots[branch]
        chosen_a = linear[branch]
        match = selected * (
            np.log(y_inputs["delta_y"] / y_inputs["roughness_length"]) / 0.40
            + chosen_a * (1.0 - y_inputs["roughness_length"] / y_inputs["delta_y"])
        ) - y_inputs["u_les"]
        moment = (0.40 * selected ** 2 * chosen_a
                  * (0.40 * chosen_a + 2.0) - y_inputs["moment_rate"])
        check("independent Yang equations (20)--(22) select and close one branch",
              len(roots) == yang["positive_root_count"]
              and abs(selected ** 2 - yang["tau_computed"]) < 2.0e-13
              and abs(match) < 2.0e-12 and abs(moment) < 2.0e-12)
        check("Yang profile array reaches its matching velocity",
              np.all(np.diff(arrays["yang_y"]) > 0.0)
              and abs(float(arrays["yang_profile"][-1])
                      - y_inputs["u_les"]) < 2.0e-12)

        park = record["operators"]["park_moin_2014"]
        park_tau, park_velocity_error, dynamic_mu = park_rebuild(park)
        check("independent Park--Moin equation (9) contraction is reproduced",
              abs(dynamic_mu - park["dynamic_mu"]) < 2.0e-15)
        check("independent backward-Euler wall layer returns molecular traction",
              abs(park_tau - park["tau_computed"]) < 2.0e-12
              and abs(park_tau - park["tau_exact"]) < 2.0e-12
              and park_velocity_error < 2.0e-12
              and np.max(np.abs(arrays["park_velocity"]
                                - arrays["park_exact_velocity"])) < 2.0e-12)
        check("Park retained-term control cases move the output",
              abs(park["drop_convection_tau"] - park["tau_computed"]) > 1.0e-2
              and abs(park["wrong_pressure_sign_tau"]
                      - park["tau_computed"]) > 1.0e-2)

    module_source = MODULE.read_text(encoding="utf-8")
    yang_args = function_arguments(module_source, "yang_rough_integral_wall_stress")
    park_args = function_arguments(module_source, "park_moin_wall_layer_step")
    check("source functions are output-bearing rather than supplied-output relabels",
          "tau_matching" not in yang_args and "moment_rate" in yang_args
          and "matching_velocity" in park_args and "pressure_gradient" in park_args
          and "convective_term" in park_args)

    source = active_tex(TEX.read_text(encoding="utf-8"))
    compact = re.sub(r"\s+", " ", source)
    check("active Methods distinguishes published models from causal instruments",
          "Source-model benchmarks return wall traction" in source
          and "A^+=17" in compact
          and "equations~(20)--(22)" in compact
          and "backward Euler" in compact
          and "causal instruments" in compact
          and "hickel2013" in source and "yang2015integral" in source
          and "park2014" in source)

    compiled = subprocess.run(
        ["pdftotext", str(PDF), "-"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    compact_pdf = re.sub(r"\s+", " ", compiled)
    check("compiled PDF contains the source-model benchmark contract",
          "Source-model benchmarks return wall traction" in compact_pdf
          and "A+ = 17" in compact_pdf
          and "backward Euler" in compact_pdf
          and PDF.stat().st_mtime_ns >= TEX.stat().st_mtime_ns)

    # Explicit control cases for the verifier itself.
    red_args = list(yang_args) + ["tau_matching"]
    check("control case rejects a Yang supplied-traction shortcut",
          not ("tau_matching" not in red_args and "moment_rate" in red_args))
    red_source = source.replace("A^+=17", "A^+=26", 1)
    check("control case rejects the wrong Hickel coefficient in Methods",
          not ("A^+=17" in re.sub(r"\s+", " ", red_source)))

    failed = [name for name, passed in checks if not passed]
    print(f"R1-SCI-2 source operators: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
