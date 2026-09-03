#!/usr/bin/env python3
"""Independent, conservative wall-following balance for ledger row R1-STA-3.

The earlier wall-following calculation assessed closure against a wall force
reconstructed from the same outer flux operator.  This producer breaks that
loop.  Volume and matching-surface terms are evaluated from the Xiao 29-case
768 x 385 archive.  Wall pressure and molecular traction are evaluated from
the separately deposited 5-case 512 x 257 archive.  Both use the analytic
physical hill surface eta = y - h(x), and only conservative phase averages are
compared across the two grids.

No eddy viscosity, Boussinesq reconstruction, thin-layer deletion, wall-
gradient pressure proxy, or model prediction enters the balance.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
NODE = ROOT / "development" / "nodes" / "node_005"
ANALYSIS = ROOT / "codes" / "analysis"
RAW_VOLUME = (ROOT / "codes" / "raw_data" / "geometry_driven" /
              "xiao_pehill_parameterized" / "pehill-29-cases-DNS" /
              "alph10-9-3036")
RAW_SURFACE = (ROOT / "codes" / "raw_data" / "geometry_driven" /
               "xiao_pehill_parameterized" / "pehill-5-cases-DNS" /
               "case_1p0" / "dns-data")
HILL_UTIL = (ROOT / "codes" / "raw_data" / "geometry_driven" /
             "xiao_pehill_parameterized" / "utility" /
             "hill-geometry-gereration")
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(HILL_UTIL))

from da_budget import (  # noqa: E402
    Config,
    ETA_MATCH,
    NU,
    build_certificate,
    load_documented_raw,
    periodic_derivative,
)
from hillShape import profile as hill_profile  # noqa: E402


IDEA = ("Conservative phase integration evaluated from independent volume and "
        "surface archives makes the complete wall-following force balance close "
        "below the physical small-residual scale before wavelength telescoping.")
PERIOD = 9.0
PHASE_COUNTS = np.array([9, 12, 18, 24, 32, 36, 48, 64, 96], dtype=int)
PRIMARY_PHASE_COUNT = 48
# Operator 2026-08-25: this was a hard-coded copy of `median_eps` from the WITHDRAWN
# wall-profile archive, whose tau_w is a 4-point through-origin linear fit with NO tangent
# correction (tau*cos^2(theta) on the flanks).  A clean verifier was being measured against a
# pasted contaminated threshold.  Corrected value on the MGLET DNS deposit, phase-block
# [0.086, 0.221]; the repaired-Xiao cubic gives 0.12856, i.e. the two valid references agree to
# 2% here even though they differ ~60% on traction RMS.  The closure envelope itself is rebuilt
# from the raw archives and is NOT a function of any traction estimator, so this change moves
# only the threshold - and it moves the margin the RIGHT way (ratio 0.762 -> 0.484).
EPSILON_REFERENCE = 0.13154
EPSILON_REFERENCE_INTERVAL = (0.086, 0.221)
EPSILON_REFERENCE_SOURCE = ("MGLET DNS deposit (ERCOFTAC UFR3-30, Re_H=5600); superseded value "
                            "0.08364189563744982 from the withdrawn 4-point estimator")
VOLUME_CONFIGS = tuple(
    Config(degree, points, mode)
    for degree, points, mode in itertools.product(
        (1, 2), (8, 12, 16), (192, None)))
SURFACE_CONFIGS = tuple(
    itertools.product((1, 2), (8, 10, 12, 16), (2, 3, 4, 5, 6)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reshape(table: np.ndarray, column: int, nx: int, ny: int) -> np.ndarray:
    return table[:, column].reshape(ny, nx).T


def load_independent_surface() -> dict[str, np.ndarray]:
    """Load the 512-grid surface archive without using processed wall profiles."""
    mean_path = RAW_SURFACE / "mean_files.dat"
    table = np.loadtxt(mean_path)
    x = np.unique(table[:, 0])
    y = np.unique(table[:, 1])
    nx, ny = x.size, y.size
    if table.shape != (nx * ny, 6) or (nx, ny) != (512, 257):
        raise RuntimeError("unexpected independent-surface archive schema")
    return {
        "x": x,
        "y": y,
        "U": reshape(table, 2, nx, ny),
        "V": reshape(table, 3, nx, ny),
        "P": reshape(table, 5, nx, ny),
        "source_sha256": np.array(sha256(mean_path)),
    }


def independent_wall_force(
    raw: dict[str, np.ndarray],
    pressure_degree: int,
    pressure_points: int,
    shear_points: int,
) -> dict[str, np.ndarray]:
    """Integrate wall pressure and molecular traction on one physical surface.

    The pressure fit and the through-origin velocity fit are independent of the
    volume/matching-plane operator.  Both are anchored at the analytic hill
    surface h(x), not at a fixed Cartesian row or a last-solid-node surrogate.
    The returned sign is the lower-boundary value B(s,0) in the transformed
    conservative identity.  No slip and incompressibility give its molecular
    part as -nu*(1+h'^2)*partial_eta(U), rather than the flat-wall expression.
    """
    x, y = raw["x"], raw["y"]
    U, V, P = raw["U"], raw["V"], raw["P"]
    h = hill_profile(x.copy())
    hp = periodic_derivative(h, x)
    speed = np.hypot(U, V)
    p_wall = np.empty(x.size)
    tau_conventional = np.empty(x.size)
    first_fluid = np.empty(x.size, dtype=int)
    for i in range(x.size):
        fluid = np.flatnonzero(speed[i] > 1.0e-6)
        if fluid.size < max(pressure_points, shear_points):
            raise RuntimeError(f"insufficient surface points at station {i}")
        k0 = int(fluid[0])
        first_fluid[i] = k0
        distance_p = y[k0:k0 + pressure_points] - h[i]
        p_wall[i] = np.polyval(
            np.polyfit(distance_p, P[i, k0:k0 + pressure_points],
                       pressure_degree), 0.0)
        distance_u = y[k0:k0 + shear_points] - h[i]
        velocity_u = U[i, k0:k0 + shear_points]
        denominator = float(np.sum(distance_u ** 2))
        if denominator <= 0.0:
            raise RuntimeError(f"degenerate wall fit at station {i}")
        tau_conventional[i] = NU * float(
            np.sum(distance_u * velocity_u) / denominator)

    pressure_force = -hp * p_wall
    molecular_force = -(1.0 + hp ** 2) * tau_conventional
    return {
        "x": x,
        "h": h,
        "h_prime": hp,
        "p_wall": p_wall,
        "pressure_force": pressure_force,
        "molecular_force": molecular_force,
        "total_force": pressure_force + molecular_force,
        "first_fluid": first_fluid,
    }


def phase_average(x: np.ndarray, field: np.ndarray, count: int) -> np.ndarray:
    """Conservative rectangle-rule average on endpoint-excluded periodic grids."""
    x = np.asarray(x, dtype=float)
    field = np.asarray(field, dtype=float)
    edges = np.linspace(0.0, PERIOD, count + 1)
    values = np.empty((count,) + field.shape[1:], dtype=float)
    for index in range(count):
        right = x < edges[index + 1] if index < count - 1 else x <= edges[index + 1]
        mask = (x >= edges[index]) & right
        if not np.any(mask):
            raise RuntimeError(f"empty phase bin {index}/{count}")
        values[index] = np.mean(field[mask], axis=0)
    return values


def parent_terms(certificate: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return all signed wall-to-matching-surface contributions."""
    return {
        "mean": certificate["mean_transport"],
        "pressure": certificate["pressure_total"],
        "reynolds": certificate["reynolds_transport"],
        "molecular": certificate["viscous_transport"],
        "body": certificate["body_force"],
    }


def closure_record(
    x_volume: np.ndarray,
    terms: dict[str, np.ndarray],
    x_surface: np.ndarray,
    surface: dict[str, np.ndarray],
    count: int,
) -> dict[str, np.ndarray | float]:
    blocked = {name: phase_average(x_volume, value, count)
               for name, value in terms.items()}
    direct_pressure = phase_average(x_surface, surface["pressure_force"], count)
    direct_molecular = phase_average(x_surface, surface["molecular_force"], count)
    direct = direct_pressure + direct_molecular
    parent = sum(blocked.values())
    residual = parent - direct
    all_legs = np.stack([*blocked.values(), direct])
    full_leg_ratio = float(np.linalg.norm(residual) / np.linalg.norm(all_legs))
    wavelength_ratio = float(abs(np.mean(residual)) /
                             max(abs(np.mean(direct)), 1.0e-30))
    return {
        **blocked,
        "direct_pressure": direct_pressure,
        "direct_molecular": direct_molecular,
        "direct_total": direct,
        "parent_total": parent,
        "residual": residual,
        "phase_full_leg_ratio": full_leg_ratio,
        "wavelength_direct_ratio": wavelength_ratio,
        "correlation": float(np.corrcoef(parent, direct)[0, 1]),
    }


def matching_plane_inventory(c: dict[str, np.ndarray]) -> dict[str, float]:
    """Split the matching-plane flux, including the dispersive mean flux."""
    k = int(c["eta_match_index"])
    hp = c["h_prime"]
    U, W = c["U"], c["W"]
    mean_flux = U[:, k] * W[:, k]
    reynolds_flux = c["stress_reynolds"][:, k]
    molecular_flux = c["stress_viscous"][:, k]
    pressure_flux = -hp * c["P"][:, k]
    ubar = float(np.mean(U[:, k]))
    wbar = float(np.mean(W[:, k]))
    dispersive = float(np.mean((U[:, k] - ubar) * (W[:, k] - wbar)))
    return {
        "mean_flux": float(np.mean(mean_flux)),
        "mean_product": ubar * wbar,
        "dispersive_flux": dispersive,
        "reynolds_flux": float(np.mean(reynolds_flux)),
        "molecular_flux": float(np.mean(molecular_flux)),
        "pressure_flux": float(np.mean(pressure_flux)),
    }


def save_outputs(arrays: dict[str, np.ndarray], summary: dict) -> None:
    for directory in (RESULTS, NODE):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "independent_phase_balance_r1_sta3.npz", **arrays)
        (directory / "independent_phase_balance_r1_sta3.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    volume_raw = load_documented_raw(RAW_VOLUME)
    surface_raw = load_independent_surface()
    central_volume = build_certificate(volume_raw, Config())
    central_surface = independent_wall_force(surface_raw, 1, 12, 4)
    central_terms = parent_terms(central_volume)

    central_records = {
        int(count): closure_record(
            central_volume["x"], central_terms,
            central_surface["x"], central_surface, int(count))
        for count in PHASE_COUNTS
    }
    primary = central_records[PRIMARY_PHASE_COUNT]

    # A deterministic operator envelope.  It is not called a confidence
    # interval: pressure extrapolation, Fourier cut-off, and wall-shear fit
    # width are numerical choices, not random samples.
    volume_ensemble = [build_certificate(volume_raw, config)
                       for config in VOLUME_CONFIGS]
    surface_ensemble = [independent_wall_force(surface_raw, *config)
                        for config in SURFACE_CONFIGS]
    envelope = np.empty((len(PHASE_COUNTS), len(volume_ensemble),
                         len(surface_ensemble)))
    wavelength_envelope = np.empty_like(envelope)
    for i, count in enumerate(PHASE_COUNTS):
        for j, volume in enumerate(volume_ensemble):
            terms = parent_terms(volume)
            for k, surface in enumerate(surface_ensemble):
                record = closure_record(volume["x"], terms,
                                        surface["x"], surface, int(count))
                envelope[i, j, k] = record["phase_full_leg_ratio"]
                wavelength_envelope[i, j, k] = record["wavelength_direct_ratio"]

    phase_central = np.array([
        central_records[int(count)]["phase_full_leg_ratio"]
        for count in PHASE_COUNTS])
    wave_central = np.array([
        central_records[int(count)]["wavelength_direct_ratio"]
        for count in PHASE_COUNTS])
    phase_min = envelope.min(axis=(1, 2))
    phase_max = envelope.max(axis=(1, 2))
    wave_min = wavelength_envelope.min(axis=(1, 2))
    wave_max = wavelength_envelope.max(axis=(1, 2))
    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            RAW_VOLUME / "mean_files.dat",
            RAW_VOLUME / "rms_files1.dat",
            RAW_VOLUME / "rms_files2.dat",
            RAW_SURFACE / "mean_files.dat",
        )
    }

    summary = {
        "schema": "independent-phase-balance-r1-sta3-v1",
        "ledger_row": "R1-STA-3",
        "idea": IDEA,
        "volume_source": "Xiao 29-case alph10-9-3036, 768x385",
        "surface_source": "Xiao 5-case case_1p0, 512x257",
        "source_hashes": source_hashes,
        "physical_surface": "eta=y-h(x), analytic Xiao hill surface",
        "matching_height_over_H": ETA_MATCH,
        "primary_phase_count": PRIMARY_PHASE_COUNT,
        "phase_counts": PHASE_COUNTS.tolist(),
        "primary_phase_closure_over_full_leg": float(
            primary["phase_full_leg_ratio"]),
        "primary_wavelength_closure_over_direct": float(
            primary["wavelength_direct_ratio"]),
        "primary_parent_direct_correlation": float(primary["correlation"]),
        "phase_closure_central_range": [float(phase_central.min()),
                                         float(phase_central.max())],
        "phase_closure_operator_envelope": [float(envelope.min()),
                                              float(envelope.max())],
        "wavelength_closure_operator_envelope": [
            float(wavelength_envelope.min()),
            float(wavelength_envelope.max()),
        ],
        "epsilon_reference_median": EPSILON_REFERENCE,
        "max_phase_closure_over_epsilon": float(envelope.max() /
                                                 EPSILON_REFERENCE),
        "volume_operator_count": len(volume_ensemble),
        "surface_operator_count": len(surface_ensemble),
        "operator_pair_count": int(envelope.size),
        "matching_plane_wavelength_inventory": matching_plane_inventory(
            central_volume),
        "uses_modelled_eddy_viscosity": False,
        "uses_boussinesq_reference_stress": False,
        "uses_thin_layer_deletion": False,
        "uses_pressure_height_proxy": False,
        "uses_same_operator_wall_reference": False,
        "status": "PASS" if (
            envelope.max() < EPSILON_REFERENCE and
            primary["correlation"] > 0.995 and
            primary["phase_full_leg_ratio"] < 0.06 and
            primary["wavelength_direct_ratio"] < 0.06
        ) else "FAIL",
    }

    arrays: dict[str, np.ndarray] = {
        "phase_counts": PHASE_COUNTS,
        "phase_closure_central": phase_central,
        "phase_closure_operator_min": phase_min,
        "phase_closure_operator_max": phase_max,
        "wavelength_closure_central": wave_central,
        "wavelength_closure_operator_min": wave_min,
        "wavelength_closure_operator_max": wave_max,
        "closure_operator_ensemble": envelope,
        "wavelength_operator_ensemble": wavelength_envelope,
        "phase_x": (np.arange(PRIMARY_PHASE_COUNT) + 0.5) *
                   PERIOD / PRIMARY_PHASE_COUNT,
        "direct_pressure": np.asarray(primary["direct_pressure"]),
        "direct_molecular": np.asarray(primary["direct_molecular"]),
        "direct_total": np.asarray(primary["direct_total"]),
        "parent_total": np.asarray(primary["parent_total"]),
        "residual": np.asarray(primary["residual"]),
        "mean_transport": np.asarray(primary["mean"]),
        "pressure_transport": np.asarray(primary["pressure"]),
        "reynolds_transport": np.asarray(primary["reynolds"]),
        "molecular_transport": np.asarray(primary["molecular"]),
        "body_force": np.asarray(primary["body"]),
        "volume_x": central_volume["x"],
        "surface_x": central_surface["x"],
        "surface_h": central_surface["h"],
        "surface_pressure_force": central_surface["pressure_force"],
        "surface_molecular_force": central_surface["molecular_force"],
        "source_volume_mean_sha256": np.array(source_hashes[
            str((RAW_VOLUME / "mean_files.dat").relative_to(ROOT))]),
        "source_surface_mean_sha256": np.array(source_hashes[
            str((RAW_SURFACE / "mean_files.dat").relative_to(ROOT))]),
        "schema": np.array(summary["schema"]),
    }
    save_outputs(arrays, summary)

    print("R1-STA-3 INDEPENDENT PHASE BALANCE")
    print(f"volume / surface grids       : 768x385 / 512x257")
    print(f"primary phase closure        : {100*summary['primary_phase_closure_over_full_leg']:.3f}%")
    print(f"primary wavelength closure   : {100*summary['primary_wavelength_closure_over_direct']:.3f}%")
    print(f"all-operator phase envelope  : [{100*envelope.min():.3f}, {100*envelope.max():.3f}]%")
    print(f"closure / epsilon (maximum)  : {summary['max_phase_closure_over_epsilon']:.3f}")
    print(f"operator pairs               : {envelope.size}")
    print(f"STATUS                       : {summary['status']}")
    if summary["status"] != "PASS":
        raise SystemExit("independent phase balance failed registered gates")


if __name__ == "__main__":
    main()
