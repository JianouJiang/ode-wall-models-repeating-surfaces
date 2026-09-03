#!/usr/bin/env python3
"""Mimetic physical-face force accounting and affine-exact deposition.

This is the Level-1 reference implementation of the force-migration method.
The conserved quantity is a fixed Cartesian momentum component.  Physical
wall force is integrated on actual faces; outward momentum flux and
wall-on-fluid force are stored with different names and signs.  A local
constrained projection deposits each integrated face force to cell centres
while reproducing constants and affine coordinates exactly.

The manufactured tests cover flat, sloped, curved, vertical-faced and
three-dimensional surfaces.  A separate raw-Xiao audit measures the two
streamwise terms commonly removed by a thin-layer reduction.  It makes no
claim of independent wall-force closure; that requires a direct IBM or
body-fitted force and remains a later empirical gate.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
NODE = ROOT / "development" / "nodes" / "node_003"
RAW = (ROOT / "codes" / "raw_data" / "geometry_driven" /
       "xiao_pehill_parameterized" / "pehill-29-cases-DNS" /
       "alph10-9-3036")
ANALYSIS = ROOT / "codes" / "analysis"
sys.path.insert(0, str(ANALYSIS))

from da_budget import (  # noqa: E402
    Config,
    NU,
    load_documented_raw,
    periodic_derivative,
    surface_interpolate,
)


IDEA = ("An affine-exact physical-face coarse-graining preserves signed wall "
        "force and its first moment under resolution change, while a scalar "
        "surrogate-wall traction loses the wall-normal force moment even when "
        "given the exact net streamwise force.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float) ** 2)))


def wall_force_from_faces(
    area_vectors: np.ndarray,
    pressure: np.ndarray,
    molecular_stress: np.ndarray,
) -> np.ndarray:
    """Return integrated wall-on-fluid force on each physical face.

    ``area_vectors`` point outward from the fluid and already include face
    area.  The returned sign follows (-p I + sigma_nu) dot n dA.
    """
    area_vectors = np.asarray(area_vectors, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    molecular_stress = np.asarray(molecular_stress, dtype=float)
    nface, ndim = area_vectors.shape
    if pressure.shape != (nface,):
        raise ValueError("pressure must have one value per face")
    if molecular_stress.shape != (nface, ndim, ndim):
        raise ValueError("molecular_stress has incompatible shape")
    cauchy = molecular_stress - pressure[:, None, None] * np.eye(ndim)[None]
    return np.einsum("fij,fj->fi", cauchy, area_vectors)


def outward_momentum_flux(
    area_vectors: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    reynolds_stress: np.ndarray,
    molecular_stress: np.ndarray,
    component: int = 0,
    density: float = 1.0,
) -> np.ndarray:
    """Integrated outward flux of one fixed Cartesian momentum component."""
    area_vectors = np.asarray(area_vectors, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    reynolds_stress = np.asarray(reynolds_stress, dtype=float)
    molecular_stress = np.asarray(molecular_stress, dtype=float)
    nface, ndim = area_vectors.shape
    if not (velocity.shape == (nface, ndim) and
            reynolds_stress.shape == (nface, ndim, ndim) and
            molecular_stress.shape == (nface, ndim, ndim)):
        raise ValueError("face-field shape mismatch")
    row = (density * velocity[:, component, None] * velocity +
           density * reynolds_stress[:, component, :] -
           molecular_stress[:, component, :])
    row[:, component] += pressure
    return np.einsum("fj,fj->f", row, area_vectors)


def certificate_wall_force(
    storage: float,
    nonwall_outward_flux: np.ndarray,
    volume_body_force: float,
) -> float:
    """Wall-on-fluid force from the complete control-volume certificate."""
    return float(storage + np.sum(nonwall_outward_flux) - volume_body_force)


def force_moments(positions: np.ndarray, integrated_forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zeroth vector moment and first position-by-force tensor moment."""
    positions = np.asarray(positions, dtype=float)
    integrated_forces = np.asarray(integrated_forces, dtype=float)
    return (np.sum(integrated_forces, axis=0),
            np.einsum("ci,cj->ij", positions, integrated_forces))


@dataclass(frozen=True)
class DepositResult:
    cell_forces: np.ndarray
    weights: np.ndarray
    support_size: np.ndarray
    constraint_condition: np.ndarray


def affine_exact_deposit(
    face_centres: np.ndarray,
    face_forces: np.ndarray,
    cell_centres: np.ndarray,
    initial_support: int | None = None,
) -> DepositResult:
    """Deposit integrated face forces with exact constant/affine precision.

    For each face, minimise ||w-w0||_2 subject to

        sum_c w_c = 1,       sum_c r_c w_c = r_face.

    The Gaussian ``w0`` controls locality but not conservation.  Signed weights
    are permitted and reported: exact wall-normal first-moment reproduction is
    impossible with non-negative weights when all fluid cell centres lie on one
    side of the wall.
    """
    face_centres = np.asarray(face_centres, dtype=float)
    face_forces = np.asarray(face_forces, dtype=float)
    cell_centres = np.asarray(cell_centres, dtype=float)
    nface, ndim = face_centres.shape
    ncell = cell_centres.shape[0]
    if face_forces.shape != (nface, ndim) or cell_centres.shape[1] != ndim:
        raise ValueError("force-deposition dimension mismatch")
    start = initial_support or max(2 * (ndim + 1), 8)
    full_weights = np.zeros((nface, ncell))
    supports = np.zeros(nface, dtype=int)
    conditions = np.zeros(nface)
    cell_forces = np.zeros((ncell, ndim))

    for g, point in enumerate(face_centres):
        distance = np.linalg.norm(cell_centres - point[None], axis=1)
        order = np.argsort(distance)
        solution = None
        for support_size in range(start, ncell + 1):
            ids = order[:support_size]
            support = cell_centres[ids]
            constraint = np.vstack((np.ones(support_size), support.T))
            if np.linalg.matrix_rank(constraint) < ndim + 1:
                continue
            scale = max(float(distance[ids[-1]]), np.finfo(float).eps)
            base = np.exp(-(distance[ids] / scale) ** 2)
            base /= np.sum(base)
            gram = constraint @ constraint.T
            condition = float(np.linalg.cond(gram))
            if not np.isfinite(condition) or condition > 1.0e12:
                continue
            target = np.r_[1.0, point]
            correction = constraint.T @ np.linalg.solve(
                gram, target - constraint @ base)
            local_weights = base + correction
            if np.max(np.abs(constraint @ local_weights - target)) > 2.0e-12:
                continue
            solution = (ids, local_weights, support_size, condition)
            break
        if solution is None:
            raise RuntimeError(f"no full-rank affine support for face {g}")
        ids, local_weights, support_size, condition = solution
        full_weights[g, ids] = local_weights
        cell_forces[ids] += local_weights[:, None] * face_forces[g]
        supports[g] = support_size
        conditions[g] = condition

    return DepositResult(cell_forces, full_weights, supports, conditions)


def box_affine_certificate(ndim: int) -> dict[str, np.ndarray | float]:
    """Exact affine-flux finite-volume certificate on a unit box."""
    if ndim not in (2, 3):
        raise ValueError("manufactured box must be two- or three-dimensional")
    centres = []
    area_vectors = []
    labels = []
    for axis in range(ndim):
        for side in (0.0, 1.0):
            point = np.full(ndim, 0.5)
            point[axis] = side
            area = np.zeros(ndim)
            area[axis] = -1.0 if side == 0.0 else 1.0
            centres.append(point)
            area_vectors.append(area)
            labels.append(f"axis{axis}_{'minus' if side == 0.0 else 'plus'}")
    centres = np.asarray(centres)
    area_vectors = np.asarray(area_vectors)
    constant = np.linspace(0.17, 0.17 * ndim, ndim)
    gradient = np.arange(1, ndim * ndim + 1, dtype=float).reshape(ndim, ndim) / 17.0
    flux_rows = constant[None] + centres @ gradient.T
    face_flux = np.einsum("fi,fi->f", flux_rows, area_vectors)
    body_force = float(np.trace(gradient))
    wall_id = labels.index("axis1_minus")
    direct_wall_force = float(-face_flux[wall_id])
    nonwall = np.delete(face_flux, wall_id)
    reconstructed = certificate_wall_force(0.0, nonwall, body_force)
    return {
        "centres": centres,
        "area_vectors": area_vectors,
        "face_flux": face_flux,
        "body_force": body_force,
        "direct_wall_force": direct_wall_force,
        "certificate_wall_force": reconstructed,
        "residual": reconstructed - direct_wall_force,
    }


def curved_wall_faces(nx: int = 32, nz: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic wavy wall extruded in z, with physical outward area vectors."""
    edges_x = np.linspace(0.0, 1.0, nx + 1)
    edges_z = np.linspace(0.0, 1.0, nz + 1)
    height = lambda x: 0.18 + 0.08 * np.sin(2.0 * np.pi * x)  # noqa: E731
    centres = []
    areas = []
    pressure = []
    stress = []
    for i in range(nx):
        x0, x1 = edges_x[i], edges_x[i + 1]
        y0, y1 = height(x0), height(x1)
        for j in range(nz):
            z0, z1 = edges_z[j], edges_z[j + 1]
            centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0,
                               (z0 + z1) / 2.0])
            # Fluid is above the wall.  This is n dA for the bilinear strip.
            area = np.array([(y1 - y0) * (z1 - z0),
                             -(x1 - x0) * (z1 - z0), 0.0])
            unit = area / np.linalg.norm(area)
            p = 0.8 + 0.25 * centre[0] - 0.1 * centre[2]
            qx = 0.45 + 0.12 * np.cos(2.0 * np.pi * centre[0])
            sigma = np.zeros((3, 3))
            sigma[0, :] = qx * unit
            sigma[:, 0] = qx * unit
            sigma[0, 0] = qx * unit[0]
            centres.append(centre)
            areas.append(area)
            pressure.append(p)
            stress.append(sigma)
    centres = np.asarray(centres)
    areas = np.asarray(areas)
    pressure = np.asarray(pressure)
    stress = np.asarray(stress)
    forces = wall_force_from_faces(areas, pressure, stress)
    return centres, areas, forces, pressure


def target_grid(nx: int) -> np.ndarray:
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    # All deposition centres lie on the fluid side of the highest wall point.
    # Exact wall-normal first moments therefore require signed weights.
    y = np.array([0.30, 0.45, 0.65])
    z = (np.arange(4, dtype=float) + 0.5) / 4.0
    return np.asarray(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T


def phase_histogram(positions: np.ndarray, forces: np.ndarray, nbins: int) -> np.ndarray:
    phase = np.mod(positions[:, 0], 1.0)
    index = np.minimum((phase * nbins).astype(int), nbins - 1)
    out = np.zeros((nbins, forces.shape[1]))
    for i, force in zip(index, forces):
        out[i] += force
    return out


def cube_pressure_test() -> dict[str, np.ndarray | float]:
    """Direct pressure-force integration on a three-dimensional cube."""
    centres = []
    areas = []
    for axis in range(3):
        for side in (-0.5, 0.5):
            point = np.zeros(3)
            point[axis] = side
            # Outward from fluid points into the solid cube.
            area = np.zeros(3)
            area[axis] = -np.sign(side)
            centres.append(point)
            areas.append(area)
    centres = np.asarray(centres)
    areas = np.asarray(areas)
    gradient = np.array([0.31, -0.17, 0.09])
    pressure = 1.2 + centres @ gradient
    force = wall_force_from_faces(areas, pressure, np.zeros((6, 3, 3)))
    # With normals into the cube, integral(-p n_fluid) = integral(p n_solid)
    # = grad(p) times unit cube volume.
    return {
        "centres": centres,
        "area_vectors": areas,
        "pressure": pressure,
        "forces": force,
        "expected": gradient,
        "residual": np.sum(force, axis=0) - gradient,
    }


def physical_face_manufactured_tests() -> tuple[np.ndarray, np.ndarray]:
    """Independent analytic checks of physical-face signs and orientations."""
    names: list[str] = []
    residuals: list[float] = []

    # Flat wall: constant and affine viscous traction integrate analytically.
    x_edges = np.linspace(0.0, 1.0, 17)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = np.diff(x_edges)
    centres = np.column_stack((x_mid, np.zeros_like(x_mid),
                               np.full_like(x_mid, 0.5)))
    areas = np.column_stack((np.zeros_like(dx), -dx, np.zeros_like(dx)))
    for label, traction, expected in (
        ("flat_constant_traction", np.full_like(x_mid, 0.7), 0.7),
        ("flat_affine_traction", 0.4 + 0.6 * x_mid, 0.7),
    ):
        stress = np.zeros((x_mid.size, 3, 3))
        stress[:, 0, 1] = -traction
        force = wall_force_from_faces(areas, np.zeros(x_mid.size), stress)
        names.append(label)
        residuals.append(float(np.sum(force[:, 0]) - expected))

    # A planar surface sloped in both wall-parallel directions.
    area = np.array([[0.25, -1.0, -0.15]])
    pressure = np.array([1.3])
    force = wall_force_from_faces(area, pressure, np.zeros((1, 3, 3)))
    expected = -pressure[0] * area[0]
    names.append("streamwise_and_spanwise_slopes")
    residuals.append(float(np.max(np.abs(force[0] - expected))))

    # Periodic curved wall: constant pressure has zero net streamwise force.
    curved_centres, curved_areas, _, _ = curved_wall_faces(nx=64, nz=2)
    curved_force = wall_force_from_faces(
        curved_areas, np.ones(curved_centres.shape[0]),
        np.zeros((curved_centres.shape[0], 3, 3)))
    names.append("periodic_curvature_constant_pressure")
    residuals.append(float(np.sum(curved_force[:, 0])))

    # Vertical rib face and explicit normal reversal.
    rib_area = np.array([[-0.8, 0.0, 0.0]])
    rib_pressure = np.array([0.9])
    rib_force = wall_force_from_faces(rib_area, rib_pressure,
                                      np.zeros((1, 3, 3)))
    names.append("vertical_rib_pressure")
    residuals.append(float(rib_force[0, 0] - 0.72))
    reversed_force = wall_force_from_faces(-rib_area, rib_pressure,
                                           np.zeros((1, 3, 3)))
    names.append("normal_reversal")
    residuals.append(float(np.max(np.abs(reversed_force + rib_force))))

    # Paired periodic faces telescope only after each signed flux is stored.
    periodic_areas = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocity = np.array([[0.8, -0.1, 0.2], [0.8, -0.1, 0.2]])
    p = np.array([0.3, 0.3])
    reynolds = np.zeros((2, 3, 3))
    reynolds[:, 0, 0] = 0.04
    molecular = np.zeros((2, 3, 3))
    molecular[:, 0, 0] = 0.01
    paired_flux = outward_momentum_flux(periodic_areas, velocity, p,
                                        reynolds, molecular)
    names.append("periodic_face_telescoping")
    residuals.append(float(np.sum(paired_flux)))

    box2 = box_affine_certificate(2)
    box3 = box_affine_certificate(3)
    names.extend(("storage_body_force_2d", "storage_body_force_3d"))
    residuals.extend((float(box2["residual"]), float(box3["residual"])))
    cube = cube_pressure_test()
    names.append("three_dimensional_cube")
    residuals.append(float(np.max(np.abs(cube["residual"]))))
    return np.asarray(names), np.asarray(residuals)


def thin_layer_term_audit() -> dict[str, np.ndarray | str]:
    """Measure, rather than delete, streamwise RANS terms in raw hill DNS."""
    raw = load_documented_raw()
    heights = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
    modes: list[int | None] = [96, 128, 192, 256, 320, None]
    mode_values = np.array([96, 128, 192, 256, 320, -1])
    names = ("mean_convection", "pressure_gradient", "reynolds_streamwise",
             "reynolds_normal", "viscous_streamwise", "viscous_normal")
    integrated_rms = np.empty((len(modes), heights.size, len(names)))
    reynolds_fraction = np.empty((len(modes), heights.size))
    viscous_ratio = np.empty_like(reynolds_fraction)
    central_station_terms = None

    for imode, mode in enumerate(modes):
        config = Config(pressure_degree=1, pressure_points=12,
                        max_fourier_mode=mode)
        mapped = surface_interpolate(raw, config)
        x = mapped["x"]
        eta = mapped["eta"]
        h_prime = periodic_derivative(mapped["h"], x, mode)

        def derivative_s(field: np.ndarray) -> np.ndarray:
            return periodic_derivative(field, x, mode)

        def derivative_eta(field: np.ndarray) -> np.ndarray:
            return np.gradient(field, eta, axis=1, edge_order=2)

        U, V, P, Rxx, Rxy = (mapped[key] for key in
                              ("U", "V", "P", "Rxx", "Rxy"))
        U_eta = derivative_eta(U)
        U_x = derivative_s(U) - h_prime[:, None] * U_eta
        terms = {
            "mean_convection": U * U_x + V * U_eta,
            "pressure_gradient": (derivative_s(P) -
                                  h_prime[:, None] * derivative_eta(P)),
            "reynolds_streamwise": (derivative_s(Rxx) -
                                      h_prime[:, None] * derivative_eta(Rxx)),
            "reynolds_normal": derivative_eta(Rxy),
            "viscous_streamwise": -NU * (derivative_s(U_x) -
                                           h_prime[:, None] * derivative_eta(U_x)),
            "viscous_normal": -NU * derivative_eta(U_eta),
        }
        for ih, height in enumerate(heights):
            k = int(np.argmin(np.abs(eta - height)))
            station = np.stack([
                np.trapezoid(terms[name][:, :k + 1], eta[:k + 1], axis=1)
                for name in names
            ], axis=1)
            integrated_rms[imode, ih] = [rms(station[:, j])
                                         for j in range(len(names))]
            lead = max(integrated_rms[imode, ih, 0],
                       integrated_rms[imode, ih, 1],
                       integrated_rms[imode, ih, 3])
            reynolds_fraction[imode, ih] = integrated_rms[imode, ih, 2] / lead
            viscous_ratio[imode, ih] = (integrated_rms[imode, ih, 4] /
                                        integrated_rms[imode, ih, 5])
            if mode is None and np.isclose(height, 0.10):
                central_station_terms = station

    assert central_station_terms is not None
    return {
        "heights_over_H": heights,
        "fourier_modes": mode_values,
        "term_names": np.array(names),
        "integrated_term_rms": integrated_rms,
        "reynolds_streamwise_fraction": reynolds_fraction,
        "viscous_streamwise_to_normal_ratio": viscous_ratio,
        "central_station_terms_eta_0p1": central_station_terms,
        "x": raw["x"],
        "source_mean_sha256": str(raw["source_mean_sha256"]),
        "source_rms1_sha256": str(raw["source_rms1_sha256"]),
        "source_rms2_sha256": str(raw["source_rms2_sha256"]),
        "schema": "full-rans-thin-layer-audit-v1",
    }


def save_json(name: str, payload: dict) -> None:
    for directory in (RESULTS, NODE):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)


def save_npz(name: str, **payload: np.ndarray) -> None:
    for directory in (RESULTS, NODE):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / name, **payload)


def main() -> None:
    NODE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    certificate_2d = box_affine_certificate(2)
    certificate_3d = box_affine_certificate(3)
    cube = cube_pressure_test()
    manufactured_names, manufactured_residuals = physical_face_manufactured_tests()
    source_centres, source_areas, source_forces, source_pressure = curved_wall_faces()

    filter_nx = np.array([8, 16, 32])
    moment0_error = []
    moment1_error = []
    phase_error = []
    finest = None
    source_m0, source_m1 = force_moments(source_centres, source_forces)
    for nx in filter_nx:
        cells = target_grid(int(nx))
        deposited = affine_exact_deposit(source_centres, source_forces, cells)
        deposited_m0, deposited_m1 = force_moments(cells, deposited.cell_forces)
        moment0_error.append(np.linalg.norm(deposited_m0 - source_m0) /
                             max(np.linalg.norm(source_m0), 1.0e-14))
        moment1_error.append(np.linalg.norm(deposited_m1 - source_m1) /
                             max(np.linalg.norm(source_m1), 1.0e-14))
        source_phase = phase_histogram(source_centres, source_forces, int(nx))
        cell_phase = phase_histogram(cells, deposited.cell_forces, int(nx))
        phase_error.append(np.linalg.norm(cell_phase - source_phase) /
                           max(np.linalg.norm(source_phase), 1.0e-14))
        if nx == filter_nx[-1]:
            finest = (cells, deposited, deposited_m0, deposited_m1)
    assert finest is not None
    cells, deposited, deposited_m0, deposited_m1 = finest

    scalar_centres = source_centres.copy()
    scalar_centres[:, 1] = 0.0
    scalar_forces = np.zeros_like(source_forces)
    scalar_forces[:, 0] = source_forces[:, 0]
    scalar_m0, scalar_m1 = force_moments(scalar_centres, scalar_forces)
    scalar_x_net_error = abs(scalar_m0[0] - source_m0[0])
    scalar_yx_moment_loss = abs(scalar_m1[1, 0] - source_m1[1, 0])

    weights = deposited.weights
    row_sum_error = np.max(np.abs(np.sum(weights, axis=1) - 1.0))
    coordinate_error = np.max(np.abs(weights @ cells - source_centres))
    negative_weight_fraction = float(np.mean(weights[weights != 0.0] < 0.0))
    max_weight_l1 = float(np.max(np.sum(np.abs(weights), axis=1)))

    thin = thin_layer_term_audit()
    reynolds_min = float(np.min(thin["reynolds_streamwise_fraction"]))
    viscous_min = float(np.min(thin["viscous_streamwise_to_normal_ratio"]))

    contract = {
        "approach": "coordinate-free physical-face force migration",
        "idea": IDEA,
        "schema": "physical-face-force-migration-v1",
        "conserved_component": "fixed Cartesian streamwise momentum",
        "wall_force": "sum_f (-p I + sigma_nu) dot S_f; S_f outward from fluid",
        "outward_flux": "sum_f [rho U_x U + p e_x + rho R_x - sigma_nu,x] dot S_f",
        "certificate": "T_w = storage + nonwall outward flux - volume body force",
        "deposition_constraints": ["sum_c w_cg = 1", "sum_c r_c w_cg = r_g"],
        "deposition_solver": "minimum-L2 correction of local Gaussian weights",
        "signed_weight_policy": ("signed weights are allowed and reported because one-sided "
                                  "fluid support cannot reproduce wall-normal first moments "
                                  "with non-negative weights"),
        "scalar_comparator": ("streamwise traction on y=0 surrogate; exact net streamwise "
                              "force is oracle-only; no normal force or wall-normal moment"),
        "oracle_and_deployed_separate": True,
        "reference_modelled_quantities": "none",
        "thin_layer_terms_deleted": "none",
        "thin_layer_audit_terms": ["partial_x R_xx", "nu partial_xx U"],
        "manufactured_tests": [
            "flat constant traction", "flat affine traction",
            "streamwise slope", "spanwise slope", "non-zero curvature",
            "vertical rib face", "three-dimensional cube pressure force",
            "periodic face telescoping", "normal reversal and owner orientation",
            "storage and body force", "constant and affine deposition",
        ],
        "claim_boundary": ("manufactured preservation is a numerical contract, not evidence "
                           "that a deployed distributed model is accurate; empirical scalar-"
                           "versus-distributed ranking requires later coupled runs"),
        "ledger_row_closed": "R3-3",
        "ledger_verifier": "python3 codes/analysis/ledger_verifiers/verify_r3_3.py",
    }

    representation_dictionary = {
        "schema": "force-representation-dictionary-v1",
        "conserved_quantity": "signed integrated Cartesian force",
        "representations": {
            "resolved_physical_surface": {
                "support": "actual wall faces",
                "force": "(-p I + sigma_nu) dot S_f",
                "inputs": "physical face geometry, pressure, molecular traction",
                "invariants": ["signed zeroth force moment", "physical first force moment"],
                "status": "reference representation",
            },
            "affine_exact_deposition_oracle": {
                "support": "volume-cell centres",
                "force": "sum_g w_cg f_g",
                "inputs": "resolved integrated face forces and face centroids",
                "invariants": ["signed zeroth force moment", "physical first force moment"],
                "status": "oracle representation; not a deployed roughness closure",
            },
            "scalar_surrogate_oracle": {
                "support": "surrogate plane y=0",
                "force": "streamwise component only",
                "inputs": "exact net streamwise reference force",
                "invariants": ["signed net streamwise force only"],
                "lost_quantities": ["normal force", "wall-normal streamwise force moment"],
                "status": "oracle capability comparator",
            },
            "deployed_scalar_wall_model": {
                "support": "surrogate wall",
                "force": "model-predicted streamwise wall traction",
                "inputs": "runtime matching-plane fields",
                "invariants": [],
                "status": "kept separate from both oracle comparisons",
            },
        },
        "comparison_rule": ("oracle-input and matched-runtime-input results must be tabulated "
                            "separately; conservation is not phase accuracy"),
    }

    summary = {
        "idea": IDEA,
        "certificate_2d_abs_residual": abs(float(certificate_2d["residual"])),
        "certificate_3d_abs_residual": abs(float(certificate_3d["residual"])),
        "cube_pressure_max_abs_residual": float(np.max(np.abs(cube["residual"]))),
        "physical_face_manufactured_max_abs_residual": float(
            np.max(np.abs(manufactured_residuals))),
        "deposition_row_sum_max_abs_error": float(row_sum_error),
        "deposition_coordinate_max_abs_error": float(coordinate_error),
        "deposition_m0_relative_error_max": float(np.max(moment0_error)),
        "deposition_m1_relative_error_max": float(np.max(moment1_error)),
        "deposition_phase_relative_errors": [float(v) for v in phase_error],
        "deposition_negative_nonzero_weight_fraction": negative_weight_fraction,
        "deposition_max_weight_l1": max_weight_l1,
        "deposition_max_constraint_condition": float(np.max(deposited.constraint_condition)),
        "scalar_oracle_streamwise_net_force_abs_error": float(scalar_x_net_error),
        "scalar_oracle_wall_normal_streamwise_moment_loss": float(scalar_yx_moment_loss),
        "thin_layer_reynolds_streamwise_fraction_min": reynolds_min,
        "thin_layer_reynolds_streamwise_fraction_max": float(np.max(
            thin["reynolds_streamwise_fraction"])),
        "thin_layer_viscous_streamwise_to_normal_min": viscous_min,
        "thin_layer_viscous_streamwise_to_normal_max": float(np.max(
            thin["viscous_streamwise_to_normal_ratio"])),
        "thin_layer_heights_over_H": thin["heights_over_H"].tolist(),
        "thin_layer_fourier_modes": thin["fourier_modes"].tolist(),
        "source_hashes": {
            "mean_files.dat": thin["source_mean_sha256"],
            "rms_files1.dat": thin["source_rms1_sha256"],
            "rms_files2.dat": thin["source_rms2_sha256"],
        },
        "status": "PASS",
    }

    tests_ok = [
        summary["certificate_2d_abs_residual"] < 1.0e-13,
        summary["certificate_3d_abs_residual"] < 1.0e-13,
        summary["cube_pressure_max_abs_residual"] < 1.0e-13,
        summary["physical_face_manufactured_max_abs_residual"] < 1.0e-13,
        row_sum_error < 2.0e-12,
        coordinate_error < 2.0e-12,
        np.max(moment0_error) < 2.0e-12,
        np.max(moment1_error) < 2.0e-12,
        scalar_x_net_error < 1.0e-13,
        scalar_yx_moment_loss > 1.0e-4,
        reynolds_min > 0.20,
        viscous_min > 0.20,
    ]
    if not all(tests_ok):
        summary["status"] = "FAIL"

    save_json("physical_face_operator_contract.json", contract)
    save_json("force_representation_dictionary.json", representation_dictionary)
    save_json("physical_face_force_migration_summary.json", summary)
    save_npz(
        "force_deposition_manufactured.npz",
        source_centres=source_centres,
        source_area_vectors=source_areas,
        source_pressure=source_pressure,
        source_forces=source_forces,
        target_centres=cells,
        target_forces=deposited.cell_forces,
        weights=weights,
        support_size=deposited.support_size,
        constraint_condition=deposited.constraint_condition,
        source_m0=source_m0,
        source_m1=source_m1,
        deposited_m0=deposited_m0,
        deposited_m1=deposited_m1,
        scalar_centres=scalar_centres,
        scalar_forces=scalar_forces,
        scalar_m0=scalar_m0,
        scalar_m1=scalar_m1,
        filter_nx=filter_nx,
        moment0_relative_error=np.asarray(moment0_error),
        moment1_relative_error=np.asarray(moment1_error),
        phase_relative_error=np.asarray(phase_error),
        certificate_2d_residual=np.array(certificate_2d["residual"]),
        certificate_3d_residual=np.array(certificate_3d["residual"]),
        cube_pressure_residual=np.asarray(cube["residual"]),
        manufactured_test_names=manufactured_names,
        manufactured_test_residuals=manufactured_residuals,
        schema=np.array("affine-exact-force-deposition-v1"),
    )
    save_npz(
        "full_rans_thin_layer_audit.npz",
        heights_over_H=thin["heights_over_H"],
        fourier_modes=thin["fourier_modes"],
        term_names=thin["term_names"],
        integrated_term_rms=thin["integrated_term_rms"],
        reynolds_streamwise_fraction=thin["reynolds_streamwise_fraction"],
        viscous_streamwise_to_normal_ratio=thin["viscous_streamwise_to_normal_ratio"],
        central_station_terms_eta_0p1=thin["central_station_terms_eta_0p1"],
        x=thin["x"],
        source_mean_sha256=np.array(thin["source_mean_sha256"]),
        source_rms1_sha256=np.array(thin["source_rms1_sha256"]),
        source_rms2_sha256=np.array(thin["source_rms2_sha256"]),
        schema=np.array(thin["schema"]),
    )

    print("PHYSICAL-FACE FORCE-MIGRATION METHODOLOGY")
    print(f"2-D / 3-D certificate residuals : {summary['certificate_2d_abs_residual']:.3e} / "
          f"{summary['certificate_3d_abs_residual']:.3e}")
    print(f"cube pressure residual          : {summary['cube_pressure_max_abs_residual']:.3e}")
    print(f"all physical-face tests         : {summary['physical_face_manufactured_max_abs_residual']:.3e}")
    print(f"deposition M0/M1 max rel error  : {summary['deposition_m0_relative_error_max']:.3e} / "
          f"{summary['deposition_m1_relative_error_max']:.3e}")
    print(f"scalar oracle lost M_yx         : {scalar_yx_moment_loss:.6e}")
    print(f"thin-layer dRxx/dx fraction     : {reynolds_min:.3f}--"
          f"{summary['thin_layer_reynolds_streamwise_fraction_max']:.3f}")
    print(f"thin-layer nu*d2U/dx2 ratio     : {viscous_min:.3f}--"
          f"{summary['thin_layer_viscous_streamwise_to_normal_max']:.3f}")
    print(f"STATUS                          : {summary['status']}")
    if summary["status"] != "PASS":
        raise SystemExit("methodology gates failed")


if __name__ == "__main__":
    main()
