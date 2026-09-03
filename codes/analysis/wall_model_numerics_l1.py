#!/usr/bin/env python3
"""Numerical verification of the deployed pressure-gradient TBLE wall model.

This is the M5 artifact required by the JCP revision.  It audits the exact C++
header used by the OpenFOAM boundary condition against a Python reference on
every registered y-index-10 wall face from ten real DNS/LES profile families.
The audit measures spatial order, root-tolerance sensitivity, iteration counts,
branch selection, failure handling, and warmed cost per face.

The main repair is numerical, not physical: the quadratic strain formula is
rationalised to remove near-wall subtractive cancellation, and the runtime
shoot changes from plain bisection to safeguarded Brent--Dekker.  No closure,
input, or wall-model equation is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
CODES = HERE.parent
PROJECT = CODES.parent
RESULTS = CODES / "results"
CPP_SOURCE = CODES / "openfoam" / "verify_tble_numerics.cpp"
TBLE_HEADER = CODES / "openfoam" / "pehill_wmles" / "wallmodel_tble" / "tbleShoot.H"
CRWM_HEADER = CODES / "openfoam" / "pehill_wmles" / "wallmodel_tble" / "tbleShootCRWM.H"
Y_INDEX = 10
KAPPA = 0.41
A_PLUS = 26.0
GRID_LEVELS = np.array([50, 100, 200, 400, 500, 800, 1600, 3200, 6400], dtype=int)
REFERENCE_GRID = 6400
TAU_TOLERANCES = np.array([1e-6, 1e-8, 1e-10, 1e-12, 1e-14])

sys.path.insert(0, str(CODES))
import manifest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_profile_files() -> list[tuple[str, Path]]:
    vendor = CODES / "vendor" / "universal_wall_function" / "codes" / "results"
    entries: list[tuple[str, Path]] = []
    for name in manifest.core_multistation_names():
        if name == "periodic_hills_case_1p0":
            path = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
        else:
            path = vendor / f"{name}_wall_profiles.npz"
        entries.append((name, path))
    entries.append(("conv_div_channel_Re12600",
                    CODES / "new_data_download" /
                    "conv_div_channel_Re12600_wall_profiles.npz"))
    return entries


def load_wall_faces():
    family, station, u_match, y_match, dpdx, nu, tau_reference = [], [], [], [], [], [], []
    sources = []
    for name, path in registered_profile_files():
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        y, velocity = data["y"], data["U"]
        pressure, viscosity = np.asarray(data["dp_dx"]), np.asarray(data["nu"])
        tau = np.asarray(data["tau_w"])
        count_before = len(family)
        for index in range(len(pressure)):
            yi = y[index] if y.ndim == 2 else y
            ui = velocity[index] if velocity.ndim == 2 else velocity
            nui = viscosity[index] if viscosity.ndim else viscosity
            if (Y_INDEX >= len(yi) or not np.isfinite(ui[Y_INDEX])
                    or not np.isfinite(yi[Y_INDEX]) or yi[Y_INDEX] <= 0):
                continue
            family.append(name)
            station.append(index)
            u_match.append(float(ui[Y_INDEX]))
            y_match.append(float(yi[Y_INDEX]))
            dpdx.append(float(pressure[index]))
            nu.append(float(nui))
            tau_reference.append(float(tau[index]))
        sources.append({"family": name, "path": str(path.relative_to(PROJECT)),
                        "sha256": sha256(path),
                        "n_faces": len(family) - count_before})
    return {
        "family": np.asarray(family, dtype="U48"),
        "station": np.asarray(station, dtype=int),
        "u_match": np.asarray(u_match),
        "y_match": np.asarray(y_match),
        "dpdx": np.asarray(dpdx),
        "nu": np.asarray(nu),
        "tau_reference": np.asarray(tau_reference),
    }, sources


def integrated_velocity(tau_w: float, y_match: float, dpdx: float, nu: float,
                        n_grid: int, stable: bool = True) -> float:
    """Return U(y_m) using the deployed eta^(3/2) grid and trapezoidal rule."""
    eta = np.linspace(0.0, 1.0, n_grid)
    y = y_match * eta ** 1.5
    u_tau = np.sqrt(max(abs(tau_w), 1e-30))
    damping = 1.0 - np.exp(-y * u_tau / (nu * A_PLUS))
    lm2 = (KAPPA * y * damping) ** 2
    forcing = tau_w + dpdx * y
    discriminant = nu * nu + 4.0 * lm2 * np.abs(forcing)
    if stable:
        # Stable positive root of lm2*|S|^2 + nu*|S| - |F| = 0.
        abs_shear = 2.0 * np.abs(forcing) / (nu + np.sqrt(discriminant))
    else:
        safe_lm2 = np.where(lm2 < 1e-30, 1.0, lm2)
        abs_shear = (-nu + np.sqrt(discriminant)) / (2.0 * safe_lm2)
        abs_shear = np.where(lm2 < 1e-30, np.abs(forcing) / nu, abs_shear)
    shear = np.sign(forcing) * abs_shear
    return float(np.trapezoid(shear, y))


def initial_bracket(u_match: float, y_match: float, dpdx: float, nu: float,
                    n_grid: int, stable: bool):
    residual = lambda tau: integrated_velocity(
        tau, y_match, dpdx, nu, n_grid, stable) - u_match
    viscous = abs(nu * u_match / y_match)
    pressure = abs(dpdx * y_match)
    scale = max(viscous, pressure, 1e-8) * 5.0
    lo, hi = -scale, scale
    f_lo, f_hi = residual(lo), residual(hi)
    calls, expansions = 2, 0
    for _ in range(12):
        if f_lo * f_hi < 0.0:
            return residual, lo, hi, calls, expansions, False
        lo *= 3.0
        hi *= 3.0
        f_lo, f_hi = residual(lo), residual(hi)
        calls += 2
        expansions += 1
    samples = np.linspace(lo, hi, 30)
    values = np.asarray([residual(value) for value in samples])
    calls += len(samples)
    changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    if not len(changes):
        return residual, np.nan, np.nan, calls, expansions, True
    index = int(changes[0])
    return residual, samples[index], samples[index + 1], calls, expansions, True


def solve_python(u_match: float, y_match: float, dpdx: float, nu: float,
                 n_grid: int, stable: bool = True, tau_tol: float = 1e-14):
    if abs(u_match) < 1e-15 and abs(dpdx) < 1e-15:
        return 0.0, 0, 0, 0, False, True
    residual, lo, hi, calls, expansions, used_scan = initial_bracket(
        u_match, y_match, dpdx, nu, n_grid, stable)
    if not np.isfinite(lo):
        return np.nan, 0, calls, expansions, used_scan, False
    root, report = brentq(residual, lo, hi, xtol=tau_tol, rtol=1e-14,
                          maxiter=100, full_output=True, disp=False)
    return (float(root), int(report.iterations), calls + int(report.function_calls),
            expansions, used_scan, bool(report.converged))


def solve_all(faces, n_grid: int, stable: bool = True, tau_tol: float = 1e-14):
    n = len(faces["u_match"])
    tau = np.empty(n)
    iterations = np.empty(n, dtype=int)
    calls = np.empty(n, dtype=int)
    expansions = np.empty(n, dtype=int)
    scans = np.empty(n, dtype=bool)
    converged = np.empty(n, dtype=bool)
    for index in range(n):
        result = solve_python(faces["u_match"][index], faces["y_match"][index],
                              faces["dpdx"][index], faces["nu"][index],
                              n_grid, stable, tau_tol)
        tau[index], iterations[index], calls[index], expansions[index], scans[index], converged[index] = result
    return tau, iterations, calls, expansions, scans, converged


def observed_order(coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray):
    first = np.abs(coarse - medium)
    second = np.abs(medium - fine)
    valid = (first > 1e-14) & (second > 1e-14)
    order = np.full(len(first), np.nan)
    order[valid] = np.log2(first[valid] / second[valid])
    return order


def root_count(faces, index: int, n_samples: int = 401) -> int:
    residual, lo, hi, _, _, _ = initial_bracket(
        faces["u_match"][index], faces["y_match"][index],
        faces["dpdx"][index], faces["nu"][index], 200, True)
    samples = np.linspace(lo, hi, n_samples)
    values = np.asarray([residual(value) for value in samples])
    return int(np.sum(values[:-1] * values[1:] < 0.0))


def plain_bisection(faces, index: int):
    residual, lo, hi, _, _, _ = initial_bracket(
        faces["u_match"][index], faces["y_match"][index],
        faces["dpdx"][index], faces["nu"][index], 200, True)
    f_lo = residual(lo)
    mid = 0.5 * (lo + hi)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(hi - lo) < 1e-12 or abs(f_mid) < 1e-14:
            break
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return mid


def continuation_root(faces, index: int) -> float:
    """Track the physical branch from dp/dx=0 for a detected multi-root face."""
    u_match = faces["u_match"][index]
    y_match = faces["y_match"][index]
    dpdx = faces["dpdx"][index]
    nu = faces["nu"][index]
    previous = solve_python(u_match, y_match, 0.0, nu, 200)[0]
    for fraction in np.linspace(0.02, 1.0, 50):
        residual, lo, hi, _, _, _ = initial_bracket(
            u_match, y_match, fraction * dpdx, nu, 200, True)
        samples = np.linspace(lo, hi, 801)
        values = np.asarray([residual(value) for value in samples])
        changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
        roots = np.asarray([brentq(residual, samples[j], samples[j + 1],
                                   xtol=1e-14, rtol=1e-14)
                            for j in changes])
        if not len(roots):
            raise RuntimeError("homotopy branch lost its bracket")
        previous = float(roots[np.argmin(np.abs(roots - previous))])
    return previous


def run_cpp(faces, repeats: int):
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the runtime-header audit")
    with tempfile.TemporaryDirectory(prefix="tble_numerics_") as temp_name:
        temp = Path(temp_name)
        input_path = temp / "faces.tsv"
        with input_path.open("w", encoding="utf-8") as stream:
            for index in range(len(faces["u_match"])):
                stream.write(
                    f"{index} {faces['u_match'][index]:.17g} {faces['dpdx'][index]:.17g} "
                    f"{faces['y_match'][index]:.17g} {faces['nu'][index]:.17g}\n")
        executable = temp / "verify_tble_numerics"
        build = subprocess.run(
            [compiler, "-O3", "-std=c++14", str(CPP_SOURCE),
             f"-I{CODES / 'openfoam'}", "-o", str(executable)],
            check=True, capture_output=True, text=True)
        run = subprocess.run([str(executable), str(input_path), str(repeats)],
                             check=True, capture_output=True, text=True)

    records = []
    benchmark = None
    for line in run.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "CASE":
            records.append(fields[1:])
        elif fields[0] == "BENCH_BLOCK":
            if benchmark is None:
                benchmark = []
            benchmark.append(fields[1:])
    if len(records) != len(faces["u_match"]) or not benchmark:
        raise RuntimeError("incomplete C++ harness output")
    array = np.asarray(records, dtype=float)
    bench = np.asarray(benchmark, dtype=float)
    cost_samples = bench[:, 4]
    return {
        "tau": array[:, 1],
        "iterations": array[:, 2].astype(int),
        "calls": array[:, 3].astype(int),
        "expansions": array[:, 4].astype(int),
        "used_scan": array[:, 5].astype(bool),
        "bracketed": array[:, 6].astype(bool),
        "converged": array[:, 7].astype(bool),
        "velocity_residual": array[:, 8],
        "bracket_width": array[:, 9],
        "benchmark_blocks": len(bench),
        "benchmark_repeats_per_block": int(bench[0, 1]),
        "benchmark_face_evaluations": int(np.sum(bench[:, 2])),
        "benchmark_seconds": float(np.sum(bench[:, 3])),
        "microseconds_per_face_samples": cost_samples,
        "microseconds_per_face": float(np.median(cost_samples)),
        "benchmark_checksum": float(bench[-1, 5]),
        "compiler": compiler,
        "compiler_stdout": build.stdout,
    }


def percentile(values, q):
    return float(np.nanpercentile(values, q))


def make_figure(path_stem: Path, grid_solutions, unstable_solutions,
                tolerance_solutions, reference, cpp_iterations):
    relative_floor = np.maximum(np.abs(reference), 1e-8)
    stable_p95 = [percentile(np.abs(grid_solutions[int(n)] - reference)
                             / relative_floor, 95) for n in GRID_LEVELS[:-1]]
    unstable_p95 = [percentile(np.abs(unstable_solutions[int(n)] - reference)
                               / relative_floor, 95)
                    for n in GRID_LEVELS[:-1] if int(n) in unstable_solutions]
    unstable_n = [int(n) for n in GRID_LEVELS[:-1] if int(n) in unstable_solutions]
    tol_reference = tolerance_solutions[1e-14]
    tol_p95 = [percentile(np.abs(tolerance_solutions[float(t)] - tol_reference), 95)
               for t in TAU_TOLERANCES[:-1]]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))
    axes[0].loglog(GRID_LEVELS[:-1], stable_p95, "o-", color="#243746",
                   label="rationalised root")
    axes[0].loglog(unstable_n, unstable_p95, "s--", color="#c55a11",
                   label="subtractive form")
    guide_n = np.asarray([50.0, 800.0])
    axes[0].loglog(guide_n, stable_p95[0] * (guide_n / 50.0) ** -2,
                   ":", color="0.4", label=r"$N^{-2}$")
    axes[0].set(xlabel="wall-normal points $N$",
                ylabel="95th-percentile relative error")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].loglog(TAU_TOLERANCES[:-1], tol_p95, "o-", color="#243746")
    axes[1].plot(TAU_TOLERANCES[:-1], TAU_TOLERANCES[:-1], ":", color="0.4")
    axes[1].invert_xaxis()
    axes[1].set(xlabel=r"root tolerance in $\tau_w$",
                ylabel=r"95th-percentile $|\Delta\tau_w|$")

    bins = np.arange(cpp_iterations.min() - 0.5, cpp_iterations.max() + 1.5)
    axes[2].hist(cpp_iterations, bins=bins, color="#607d8b", edgecolor="white")
    axes[2].set(xlabel="Brent iterations per wall face", ylabel="face count")
    axes[2].axvline(np.median(cpp_iterations), color="black", linestyle="--",
                    linewidth=1)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes,
                  va="top", fontweight="bold")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-dir", type=Path, default=None,
                        help="also copy the generated artifacts into this node")
    parser.add_argument("--benchmark-repeats", type=int, default=30,
                        help="repeats per timing block; seven blocks are measured")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.node_dir is not None:
        args.node_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    faces, sources = load_wall_faces()
    n_faces = len(faces["u_match"])
    print(f"Registered corpus: {n_faces} faces, {len(sources)} families")

    grid_solutions = {}
    grid_iterations = {}
    for n_grid in GRID_LEVELS:
        tau, iterations, *_ = solve_all(faces, int(n_grid), stable=True)
        grid_solutions[int(n_grid)] = tau
        grid_iterations[int(n_grid)] = iterations
        print(f"  stable grid N={n_grid}: complete")
    reference = grid_solutions[REFERENCE_GRID]

    unstable_solutions = {}
    for n_grid in (50, 100, 200, 400, 500, 800, 1600):
        unstable_solutions[n_grid] = solve_all(
            faces, n_grid, stable=False)[0]

    tolerance_solutions = {}
    tolerance_iterations = {}
    for tolerance in TAU_TOLERANCES:
        tau, iterations, *_ = solve_all(
            faces, 200, stable=True, tau_tol=float(tolerance))
        tolerance_solutions[float(tolerance)] = tau
        tolerance_iterations[float(tolerance)] = iterations

    cpp = run_cpp(faces, args.benchmark_repeats)
    print(f"C++ harness: {np.sum(cpp['converged'])}/{n_faces} converged; "
          f"{cpp['microseconds_per_face']:.3f} microseconds/face")

    # A dense root census is offline verification, not part of runtime cost.
    root_counts = np.asarray([root_count(faces, index) for index in range(n_faces)])
    multi_indices = np.flatnonzero(root_counts > 1)
    bisection_tau = np.asarray([plain_bisection(faces, index)
                               for index in range(n_faces)])
    continuation = np.full(n_faces, np.nan)
    for index in multi_indices:
        continuation[index] = continuation_root(faces, int(index))

    stable_order = observed_order(grid_solutions[200], grid_solutions[400],
                                  grid_solutions[800])
    unstable_order = observed_order(unstable_solutions[200],
                                    unstable_solutions[400],
                                    unstable_solutions[800])
    rel_floor = np.maximum(np.abs(reference), 1e-8)
    grid200_rel = np.abs(grid_solutions[200] - reference) / rel_floor
    grid500_rel = np.abs(grid_solutions[500] - reference) / rel_floor
    cpp_python_abs = np.abs(cpp["tau"] - grid_solutions[200])
    formula_rel = np.abs(unstable_solutions[200] - grid_solutions[200]) / rel_floor
    branch_delta = np.abs(bisection_tau - cpp["tau"])

    hill = faces["family"] == "periodic_hills_case_1p0"
    truth = faces["tau_reference"][hill]
    stable_hill = grid_solutions[500][hill]
    hill_r2_stable = 1.0 - np.sum((stable_hill - truth) ** 2) / np.sum(
        (truth - truth.mean()) ** 2)
    archived_hill_r2 = -47.68617253416459

    checks = {
        "registered_real_faces_786": n_faces == 786,
        "registered_families_10": len(sources) == 10,
        "cpp_all_bracketed": bool(np.all(cpp["bracketed"])),
        "cpp_all_converged": bool(np.all(cpp["converged"])),
        "cpp_no_zero_fallback": bool(np.all(np.isfinite(cpp["tau"]))),
        "cpp_python_same_grid_max_abs_lt_2e-11": float(np.max(cpp_python_abs)) < 2e-11,
        "rationalised_observed_order_gt_1p95": percentile(stable_order, 50) > 1.95,
        "rationalisation_repairs_order": percentile(stable_order, 50) >
            percentile(unstable_order, 50) + 1.5,
        "N200_p95_relative_error_lt_1e-4": percentile(grid200_rel, 95) < 1e-4,
        "N200_max_relative_error_lt_3e-4": float(np.max(grid200_rel)) < 3e-4,
        "N500_p95_relative_error_lt_2e-5": percentile(grid500_rel, 95) < 2e-5,
        "registered_multi_root_detected": len(multi_indices) >= 1,
        "plain_bisection_branch_error_detected": float(np.max(branch_delta)) > 1e-3,
        "brent_matches_zero_pressure_continuation": bool(
            len(multi_indices) and
            np.max(np.abs(cpp["tau"][multi_indices] - continuation[multi_indices])) < 2e-10),
        "p95_iterations_lt_20": percentile(cpp["iterations"], 95) < 20,
        "kernel_cost_lt_200us_per_face": cpp["microseconds_per_face"] < 200.0,
        "printed_hill_R2_unchanged": abs(hill_r2_stable - archived_hill_r2) < 1e-4,
    }
    checks = {name: bool(value) for name, value in checks.items()}

    npz_path = RESULTS / "wall_model_numerics_l1.npz"
    np.savez(
        npz_path,
        **faces,
        grid_levels=GRID_LEVELS,
        tau_tolerances=TAU_TOLERANCES,
        tau_python_N50=grid_solutions[50],
        tau_python_N100=grid_solutions[100],
        tau_python_N200=grid_solutions[200],
        tau_python_N400=grid_solutions[400],
        tau_python_N500=grid_solutions[500],
        tau_python_N800=grid_solutions[800],
        tau_python_N1600=grid_solutions[1600],
        tau_python_N3200=grid_solutions[3200],
        tau_python_N6400=reference,
        tau_unstable_N200=unstable_solutions[200],
        tau_unstable_N400=unstable_solutions[400],
        tau_unstable_N800=unstable_solutions[800],
        observed_order_stable_200_400_800=stable_order,
        observed_order_unstable_200_400_800=unstable_order,
        grid200_relative_error=grid200_rel,
        grid500_relative_error=grid500_rel,
        cpp_tau=cpp["tau"],
        cpp_iterations=cpp["iterations"],
        cpp_residual_evaluations=cpp["calls"],
        cpp_bracket_expansions=cpp["expansions"],
        cpp_used_scan=cpp["used_scan"],
        cpp_bracketed=cpp["bracketed"],
        cpp_converged=cpp["converged"],
        cpp_velocity_residual=cpp["velocity_residual"],
        cpp_terminal_bracket_width=cpp["bracket_width"],
        root_count_401_scan=root_counts,
        multi_root_indices=multi_indices,
        plain_bisection_tau=bisection_tau,
        continuation_tau=continuation,
        formula_relative_difference=formula_rel,
        tolerance_tau_1e6=tolerance_solutions[1e-6],
        tolerance_tau_1e8=tolerance_solutions[1e-8],
        tolerance_tau_1e10=tolerance_solutions[1e-10],
        tolerance_tau_1e12=tolerance_solutions[1e-12],
        tolerance_tau_1e14=tolerance_solutions[1e-14],
        benchmark_microseconds_per_face=np.asarray(cpp["microseconds_per_face"]),
        benchmark_microseconds_per_face_samples=cpp["microseconds_per_face_samples"],
        benchmark_face_evaluations=np.asarray(cpp["benchmark_face_evaluations"]),
        hill_r2_stable=np.asarray(hill_r2_stable),
        hill_r2_archived=np.asarray(archived_hill_r2),
        y_index=np.asarray(Y_INDEX),
    )

    figure_stem = RESULTS / "fig_wall_model_numerics_l1"
    make_figure(figure_stem, grid_solutions, unstable_solutions,
                tolerance_solutions, reference, cpp["iterations"])

    summary = {
        "schema": "wall-model-numerics-l1-v1",
        "approach": "rationalised quadratic plus safeguarded Brent-Dekker shooting",
        "equation_changed": False,
        "closure_changed": False,
        "matching_index": Y_INDEX,
        "n_families": len(sources),
        "n_faces": n_faces,
        "sources": sources,
        "spatial_scheme": "composite trapezoidal on y=y_m*eta^(3/2)",
        "runtime_grid_points": 200,
        "python_reference_grid_points": REFERENCE_GRID,
        "root_method": "safeguarded Brent-Dekker with bracket expansion",
        "tau_tolerance": 1e-12,
        "velocity_residual_tolerance": 1e-14,
        "maximum_iterations": 100,
        "failure_handling": "the report records converged=false; the scalar OpenFOAM API returns NaN rather than a false zero; zero failures in the registered corpus",
        "stable_order_median": percentile(stable_order, 50),
        "stable_order_p10_p90": [percentile(stable_order, 10), percentile(stable_order, 90)],
        "unstable_order_median": percentile(unstable_order, 50),
        "runtime_N200_relative_error": {
            "median": percentile(grid200_rel, 50),
            "p95": percentile(grid200_rel, 95),
            "max": float(np.max(grid200_rel)),
        },
        "python_N500_relative_error": {
            "median": percentile(grid500_rel, 50),
            "p95": percentile(grid500_rel, 95),
            "max": float(np.max(grid500_rel)),
        },
        "cpp_python_same_grid_max_abs": float(np.max(cpp_python_abs)),
        "iterations": {
            "median": percentile(cpp["iterations"], 50),
            "p95": percentile(cpp["iterations"], 95),
            "max": int(np.max(cpp["iterations"])),
            "residual_evaluations_median": percentile(cpp["calls"], 50),
        },
        "bracketing": {
            "converged": int(np.sum(cpp["converged"])),
            "failed": int(np.sum(~cpp["converged"])),
            "expansion_max": int(np.max(cpp["expansions"])),
            "scan_count": int(np.sum(cpp["used_scan"])),
        },
        "branch_audit": {
            "multi_root_face_count": int(len(multi_indices)),
            "multi_root_global_indices": multi_indices.tolist(),
            "plain_bisection_max_delta": float(np.max(branch_delta)),
            "brent_continuation_max_delta": float(
                np.max(np.abs(cpp["tau"][multi_indices] - continuation[multi_indices]))
                if len(multi_indices) else np.nan),
        },
        "formula_audit": {
            "direct_subtraction_p95_relative_difference_N200": percentile(formula_rel, 95),
            "direct_subtraction_max_relative_difference_N200": float(np.max(formula_rel)),
        },
        "cost": {
            "compiler": cpp["compiler"],
            "benchmark_blocks": cpp["benchmark_blocks"],
            "benchmark_repeats_per_block": cpp["benchmark_repeats_per_block"],
            "face_evaluations": cpp["benchmark_face_evaluations"],
            "seconds": cpp["benchmark_seconds"],
            "microseconds_per_face": cpp["microseconds_per_face"],
            "microseconds_per_face_p25_p75": [
                percentile(cpp["microseconds_per_face_samples"], 25),
                percentile(cpp["microseconds_per_face_samples"], 75)],
            "scope": "warmed standalone single-thread wall-model kernel; excludes OpenFOAM field access and communication",
        },
        "canonical_hill_R2": {
            "stable_solver": float(hill_r2_stable),
            "archived_solver": archived_hill_r2,
            "difference": float(hill_r2_stable - archived_hill_r2),
        },
        "source_hashes": {
            "tble_header": sha256(TBLE_HEADER),
            "crwm_header": sha256(CRWM_HEADER),
            "cpp_harness": sha256(CPP_SOURCE),
        },
        "runtime_seconds": time.perf_counter() - started,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    summary_path = RESULTS / "wall_model_numerics_l1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    products = [npz_path, summary_path, figure_stem.with_suffix(".pdf"),
                figure_stem.with_suffix(".png")]
    if args.node_dir is not None:
        for product in products:
            shutil.copy2(product, args.node_dir / product.name)

    print("\nWALL-MODEL NUMERICS")
    print(f"  observed order, rationalised form : {summary['stable_order_median']:.4f}")
    print(f"  observed order, subtractive form  : {summary['unstable_order_median']:.4f}")
    print(f"  N=200 relative error p95/max       : {percentile(grid200_rel,95):.3e} / {np.max(grid200_rel):.3e}")
    print(f"  C++/Python max |delta tau_w|       : {np.max(cpp_python_abs):.3e}")
    print(f"  iterations median/p95/max          : {percentile(cpp['iterations'],50):.0f} / {percentile(cpp['iterations'],95):.0f} / {np.max(cpp['iterations'])}")
    print(f"  multi-root faces / branch repair   : {len(multi_indices)} / {np.max(branch_delta):.3e}")
    print(f"  kernel cost                        : {cpp['microseconds_per_face']:.3f} us/face")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"  Saved -> {npz_path.relative_to(PROJECT)}")
    print(f"  Saved -> {summary_path.relative_to(PROJECT)}")
    if not all(checks.values()):
        raise SystemExit("wall-model numerical verification failed")


if __name__ == "__main__":
    main()
