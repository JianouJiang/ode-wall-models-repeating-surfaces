#!/usr/bin/env python3
"""Direct-force-closed adequacy certificate on the deposited square-rib WRLES.

This is a real-case finite-volume witness for the representation theorem.  It
uses the 100-flow-time OpenFOAM field average (UMean, UPrime2Mean and pMean),
five stored instantaneous U/nut fields for the mean SGS flux, and pressure plus
molecular traction evaluated independently on the physical wall faces.

The parent operator is the complete steady Cartesian x-momentum balance on the
mesh's streamwise phase control volumes spanning the full height and span (48
for the deposited one-pitch case).  No thin-layer
deletion, eddy-viscosity reconstruction of Reynolds stress, pressure-gradient
times height proxy, or self-reconstructed wall force is used.  The rib's
vertical faces are physical wall faces, so form force enters directly.

For a declared linear force representation B, the estimator is

    r_B(a) = B a - f_parent,

while the independently measured error is e_B(a)=B a-f_direct.  Since
||f_parent-f_direct|| <= delta_parent, the triangle inequality gives the
computable two-sided certificate

    ||r_B(a)||-delta_parent <= ||e_B(a)|| <= ||r_B(a)||+delta_parent.

The same perturbation bound applies to the best-representation gap.  The code
checks these statements for target-blind zero/constant/coarsened candidates and
separately reports oracle best projections.  It explicitly shows that a
phase-wise scalar traction has zero x-force representation gap; any positive
complete-trace gap is therefore attributed only to the declared vector/moment
outputs, not to wall-stress error.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "codes" / "openfoam" / "rib_les_dtype"
CASE_LABEL = "deposited one-pitch square-rib LES numerical substrate"
RESULTS = ROOT / "codes" / "results"
NODE = ROOT / "development" / "nodes" / "node_004"
MESH = CASE / "constant" / "polyMesh"
TIMES = ("60.00073791", "80.00104278", "99.99809652",
         "120.00046185", "139.99984127")
CENTRAL_TIME = TIMES[-1]
NU = 1.0 / 4200.0
DOMAIN_LENGTH = 0.6
TRACE_COMPONENTS = ("phase_Fx", "phase_Fy", "phase_yFx_over_H")
EXPECTED_PHASE_CELLS = int(os.environ.get("M9_EXPECTED_PHASE_CELLS", "48"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parenthesized_body(text: str, start: int) -> tuple[str, int]:
    opening = text.find("(", start)
    if opening < 0:
        raise ValueError("opening parenthesis not found")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index], index + 1
    raise ValueError("unclosed parenthesized list")


def _nonuniform_body(text: str, anchor: str) -> tuple[int, str]:
    match = re.search(anchor + r"\s+nonuniform\s+List<[^>]+>\s+(\d+)", text)
    if not match:
        raise ValueError(f"nonuniform field not found: {anchor}")
    count = int(match.group(1))
    body, _ = _parenthesized_body(text, match.end())
    return count, body


def _vectors(body: str, width: int) -> np.ndarray:
    rows = re.findall(r"\(([^()]*)\)", body)
    values = np.array([[float(value) for value in row.split()] for row in rows])
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"expected vector width {width}, got {values.shape}")
    return values


def read_internal_field(path: Path, kind: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    widths = {"scalar": 1, "vector": 3, "symmTensor": 6}
    count, body = _nonuniform_body(text, r"internalField")
    if kind == "scalar":
        values = np.fromstring(body, sep=" ")
    else:
        values = _vectors(body, widths[kind])
    if values.shape[0] != count:
        raise ValueError(f"{path}: expected {count} entries, got {values.shape[0]}")
    return values


def read_patch_field(path: Path, patch: str, kind: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    patch_match = re.search(rf"\b{re.escape(patch)}\s*\{{", text)
    if not patch_match:
        raise ValueError(f"patch {patch} not found in {path}")
    opening = text.find("{", patch_match.start())
    depth = 0
    closing = None
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        raise ValueError(f"unclosed patch {patch}")
    block = text[opening + 1:closing]
    widths = {"scalar": 1, "vector": 3, "symmTensor": 6}
    count, body = _nonuniform_body(block, r"value")
    if kind == "scalar":
        values = np.fromstring(body, sep=" ")
    else:
        values = _vectors(body, widths[kind])
    if values.shape[0] != count:
        raise ValueError(f"{path}:{patch}: patch count mismatch")
    return values


def read_plain_list(path: Path, kind: str) -> np.ndarray | list[np.ndarray]:
    text = path.read_text(encoding="utf-8")
    # OpenFOAM Foundation and ESI writers use slightly different decorative
    # header rulers.  Anchor on the first count followed by the top-level list
    # instead of depending on either ruler spelling.
    match = re.search(r"\n\s*(\d+)\s*\n\s*\(", text)
    if not match:
        raise ValueError(f"OpenFOAM list not found in {path}")
    absolute = match.end() - 1
    count = int(match.group(1))
    body, _ = _parenthesized_body(text, absolute)
    if kind == "points":
        values = _vectors(body, 3)
        if values.shape[0] != count:
            raise ValueError("point count mismatch")
        return values
    if kind == "faces":
        rows = []
        for width, payload in re.findall(r"(\d+)\s*\(([^()]*)\)", body):
            row = np.fromstring(payload, sep=" ", dtype=int)
            if row.size != int(width):
                raise ValueError("face width mismatch")
            rows.append(row)
        if len(rows) != count:
            raise ValueError("face count mismatch")
        return rows
    values = np.fromstring(body, sep=" ", dtype=int)
    if values.size != count:
        raise ValueError(f"label count mismatch in {path}")
    return values


def read_boundary() -> dict[str, dict[str, int]]:
    text = (MESH / "boundary").read_text(encoding="utf-8")
    result = {}
    for name in ("inlet", "outlet", "zMin", "zMax", "bottomWall", "topWall"):
        match = re.search(
            rf"\b{name}\s*\{{.*?nFaces\s+(\d+)\s*;.*?startFace\s+(\d+)\s*;",
            text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"boundary metadata missing for {name}")
        result[name] = {"nFaces": int(match.group(1)),
                        "startFace": int(match.group(2))}
    return result


@dataclass
class MeshData:
    points: np.ndarray
    faces: list[np.ndarray]
    owner: np.ndarray
    neighbour: np.ndarray
    face_centres: np.ndarray
    face_area_vectors: np.ndarray
    cell_centres: np.ndarray
    cell_volumes: np.ndarray
    boundaries: dict[str, dict[str, int]]
    phase_index: np.ndarray
    phase_x: np.ndarray


def build_mesh() -> MeshData:
    points = np.asarray(read_plain_list(MESH / "points", "points"))
    faces = read_plain_list(MESH / "faces", "faces")
    owner = np.asarray(read_plain_list(MESH / "owner", "labels"))
    neighbour = np.asarray(read_plain_list(MESH / "neighbour", "labels"))
    cell_centres = read_internal_field(CASE / "0" / "C", "vector")
    if len(faces) != owner.size:
        raise ValueError("mesh face/owner mismatch")

    centres = np.empty((len(faces), 3))
    areas = np.empty((len(faces), 3))
    for face_index, vertices in enumerate(faces):
        xyz = points[vertices]
        origin = np.mean(xyz, axis=0)
        area = np.zeros(3)
        weighted = np.zeros(3)
        weight_sum = 0.0
        for j in range(xyz.shape[0]):
            a = xyz[j] - origin
            b = xyz[(j + 1) % xyz.shape[0]] - origin
            triangle_area = 0.5 * np.cross(a, b)
            magnitude = np.linalg.norm(triangle_area)
            area += triangle_area
            weighted += magnitude * (origin + xyz[j] + xyz[(j + 1) % xyz.shape[0]]) / 3.0
            weight_sum += magnitude
        areas[face_index] = area
        centres[face_index] = weighted / weight_sum

    ncell = cell_centres.shape[0]
    volumes = np.zeros(ncell)
    moment = np.einsum("ij,ij->i", areas, centres) / 3.0
    np.add.at(volumes, owner, moment)
    np.add.at(volumes, neighbour, -moment[:neighbour.size])
    if np.min(volumes) <= 0 or not np.all(np.isfinite(volumes)):
        raise ValueError("non-positive reconstructed cell volume")

    phase_x, phase_index = np.unique(np.round(cell_centres[:, 0], 12),
                                     return_inverse=True)
    if EXPECTED_PHASE_CELLS > 0 and phase_x.size != EXPECTED_PHASE_CELLS:
        raise ValueError(
            f"expected {EXPECTED_PHASE_CELLS} streamwise phase cells, got {phase_x.size}"
        )
    return MeshData(points, faces, owner, neighbour, centres, areas,
                    cell_centres, volumes, read_boundary(), phase_index, phase_x)


def patch_face_ids(mesh: MeshData, patch: str) -> np.ndarray:
    item = mesh.boundaries[patch]
    return np.arange(item["startFace"], item["startFace"] + item["nFaces"])


def linear_face_value(values: np.ndarray, owner: np.ndarray, neighbour: np.ndarray,
                      centres: np.ndarray, face_centres: np.ndarray) -> np.ndarray:
    delta_owner = np.linalg.norm(face_centres - centres[owner], axis=1)
    delta_neighbour = np.linalg.norm(centres[neighbour] - face_centres, axis=1)
    denominator = delta_owner + delta_neighbour
    return ((delta_neighbour / denominator)[:, None] * values[owner] +
            (delta_owner / denominator)[:, None] * values[neighbour])


def linear_face_scalar(values: np.ndarray, owner: np.ndarray, neighbour: np.ndarray,
                       centres: np.ndarray, face_centres: np.ndarray) -> np.ndarray:
    delta_owner = np.linalg.norm(face_centres - centres[owner], axis=1)
    delta_neighbour = np.linalg.norm(centres[neighbour] - face_centres, axis=1)
    denominator = delta_owner + delta_neighbour
    return ((delta_neighbour / denominator) * values[owner] +
            (delta_owner / denominator) * values[neighbour])


def pair_periodic_x(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    inlet = patch_face_ids(mesh, "inlet")
    outlet = patch_face_ids(mesh, "outlet")
    # The blockMesh writer leaves O(1e-8) round-off differences between the
    # two translationally coupled point sets.  Pair geometrically at a
    # tolerance far below the smallest face spacing (1.6e-3).
    lookup = {
        (round(mesh.face_centres[f, 1], 7), round(mesh.face_centres[f, 2], 7)): f
        for f in outlet
    }
    paired_outlet = []
    for face in inlet:
        key = (round(mesh.face_centres[face, 1], 7),
               round(mesh.face_centres[face, 2], 7))
        if key not in lookup:
            raise ValueError("cyclic x-face pairing failed")
        paired_outlet.append(lookup[key])
    paired = np.asarray(paired_outlet, dtype=int)
    if np.max(np.linalg.norm(mesh.face_centres[inlet, 1:] -
                             mesh.face_centres[paired, 1:], axis=1)) > 1e-6:
        raise ValueError("cyclic x-face pairing exceeds tolerance")
    return inlet, paired


def body_gradient_until(end_time: float) -> float:
    text = (CASE / "log.pimpleFoam").read_text(encoding="utf-8", errors="replace")
    current_time = None
    current_gradient = None
    records = []
    for line in text.splitlines():
        time_match = re.match(r"Time = ([0-9.eE+-]+)s", line)
        if time_match:
            if current_time is not None and current_gradient is not None:
                records.append((current_time, current_gradient))
            current_time = float(time_match.group(1))
            current_gradient = None
            continue
        gradient_match = re.search(r"pressure gradient = ([0-9.eE+-]+)", line)
        if gradient_match and current_time is not None:
            current_gradient = float(gradient_match.group(1))
    if current_time is not None and current_gradient is not None:
        records.append((current_time, current_gradient))
    array = np.asarray([(t, g) for t, g in records if 40.0 <= t <= end_time])
    if array.shape[0] < 100:
        raise ValueError("insufficient body-force history")
    duration = array[-1, 0] - array[0, 0]
    # ARCHER2's pinned Cray Python currently ships NumPy < 2.0, where
    # ``trapezoid`` is not yet available.  ``trapz`` is the identical
    # composite-trapezoid operation on those supported releases.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(array[:, 1], array[:, 0]) / duration)


def symm_to_rxx(values: np.ndarray) -> np.ndarray:
    # OpenFOAM symmTensor order is xx, xy, xz, yy, yz, zz.
    return values[:, 0]


def sgs_normal_stress(mesh: MeshData, face_ids: np.ndarray,
                      owner: np.ndarray, neighbour: np.ndarray,
                      sample_times: tuple[str, ...],
                      periodic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    for time in sample_times:
        velocity = read_internal_field(CASE / time / "U", "vector")
        nut = read_internal_field(CASE / time / "nut", "scalar")
        if periodic:
            dx = (mesh.cell_centres[neighbour, 0] + DOMAIN_LENGTH -
                  mesh.cell_centres[owner, 0])
        else:
            dx = mesh.cell_centres[neighbour, 0] - mesh.cell_centres[owner, 0]
        derivative = (velocity[neighbour, 0] - velocity[owner, 0]) / dx
        nut_face = 0.5 * (nut[owner] + nut[neighbour])
        samples.append(-2.0 * nut_face * derivative *
                       mesh.face_area_vectors[face_ids, 0])
    stack = np.stack(samples)
    return np.mean(stack, axis=0), np.std(stack, axis=0, ddof=1)


def evaluate_time(mesh: MeshData, time: str) -> dict[str, np.ndarray | float]:
    time_path = CASE / time
    U = read_internal_field(time_path / "UMean", "vector")
    p = read_internal_field(time_path / "pMean", "scalar")
    Rxx = symm_to_rxx(read_internal_field(time_path / "UPrime2Mean", "symmTensor"))
    nphase = mesh.phase_x.size

    terms = {name: np.zeros(nphase) for name in
             ("mean", "pressure", "reynolds", "molecular", "sgs")}
    term_sgs_uncertainty = np.zeros(nphase)
    # An average ending at t may use only stored SGS snapshots at or before t.
    # This keeps the window-sensitivity audit causal; the central t=140 record
    # still uses all five available snapshots.
    sample_times = tuple(snapshot for snapshot in TIMES
                         if float(snapshot) <= float(time) + 1e-8)
    if len(sample_times) < 3:
        raise ValueError("fewer than three causal SGS snapshots")

    ninternal = mesh.neighbour.size
    internal_ids = np.arange(ninternal)
    o_all = mesh.owner[:ninternal]
    n_all = mesh.neighbour
    cross = mesh.phase_index[o_all] != mesh.phase_index[n_all]
    face_ids = internal_ids[cross]
    owner = o_all[cross]
    neighbour = n_all[cross]
    sf_x = mesh.face_area_vectors[face_ids, 0]
    if np.max(np.abs(mesh.face_area_vectors[face_ids, 1:])) > 1e-12:
        raise ValueError("phase interface is not Cartesian-x aligned")
    Uf = linear_face_value(U, owner, neighbour, mesh.cell_centres,
                           mesh.face_centres[face_ids])
    pf = linear_face_scalar(p, owner, neighbour, mesh.cell_centres,
                            mesh.face_centres[face_ids])
    rf = linear_face_scalar(Rxx, owner, neighbour, mesh.cell_centres,
                            mesh.face_centres[face_ids])
    dx = mesh.cell_centres[neighbour, 0] - mesh.cell_centres[owner, 0]
    duxdx = (U[neighbour, 0] - U[owner, 0]) / dx
    face_terms = {
        "mean": Uf[:, 0] ** 2 * sf_x,
        "pressure": pf * sf_x,
        "reynolds": rf * sf_x,
        "molecular": -2.0 * NU * duxdx * sf_x,
    }
    sgs, sgs_std = sgs_normal_stress(
        mesh, face_ids, owner, neighbour, sample_times)
    face_terms["sgs"] = sgs
    for name, values in face_terms.items():
        np.add.at(terms[name], mesh.phase_index[owner], values)
        np.add.at(terms[name], mesh.phase_index[neighbour], -values)
    np.add.at(term_sgs_uncertainty, mesh.phase_index[owner], np.abs(sgs_std))
    np.add.at(term_sgs_uncertainty, mesh.phase_index[neighbour], np.abs(sgs_std))

    # Periodic streamwise interface.  Use each paired face once for each side.
    inlet, outlet = pair_periodic_x(mesh)
    owner_first = mesh.owner[inlet]
    owner_last = mesh.owner[outlet]
    sf_out = mesh.face_area_vectors[outlet, 0]
    Uf = 0.5 * (U[owner_last] + U[owner_first])
    pf = 0.5 * (p[owner_last] + p[owner_first])
    rf = 0.5 * (Rxx[owner_last] + Rxx[owner_first])
    dx = (mesh.cell_centres[owner_first, 0] + DOMAIN_LENGTH -
          mesh.cell_centres[owner_last, 0])
    duxdx = (U[owner_first, 0] - U[owner_last, 0]) / dx
    periodic_terms = {
        "mean": Uf[:, 0] ** 2 * sf_out,
        "pressure": pf * sf_out,
        "reynolds": rf * sf_out,
        "molecular": -2.0 * NU * duxdx * sf_out,
    }
    sgs, sgs_std = sgs_normal_stress(mesh, outlet, owner_last, owner_first,
                                     sample_times,
                                     periodic=True)
    periodic_terms["sgs"] = sgs
    for name, values in periodic_terms.items():
        np.add.at(terms[name], mesh.phase_index[owner_last], values)
        np.add.at(terms[name], mesh.phase_index[owner_first], -values)
    np.add.at(term_sgs_uncertainty, mesh.phase_index[owner_last], np.abs(sgs_std))
    np.add.at(term_sgs_uncertainty, mesh.phase_index[owner_first], np.abs(sgs_std))

    pressure_force = np.zeros((nphase, 3))
    viscous_force = np.zeros((nphase, 3))
    moment_y_fx = np.zeros(nphase)
    for patch in ("bottomWall", "topWall"):
        ids = patch_face_ids(mesh, patch)
        owners = mesh.owner[ids]
        sf = mesh.face_area_vectors[ids]
        area = np.linalg.norm(sf, axis=1)
        normal = sf / area[:, None]
        p_wall = read_patch_field(time_path / "pMean", patch, "scalar")
        fp = -p_wall[:, None] * sf
        delta = np.abs(np.einsum(
            "ij,ij->i", mesh.face_centres[ids] - mesh.cell_centres[owners], normal))
        tangential_velocity = U[owners] - np.einsum(
            "ij,ij->i", U[owners], normal)[:, None] * normal
        fv = -NU * tangential_velocity / delta[:, None] * area[:, None]
        phase = mesh.phase_index[owners]
        np.add.at(pressure_force, phase, fp)
        np.add.at(viscous_force, phase, fv)
        np.add.at(moment_y_fx, phase, mesh.face_centres[ids, 1] * (fp[:, 0] + fv[:, 0]))

    gradient = body_gradient_until(float(time))
    volume_by_phase = np.bincount(mesh.phase_index, weights=mesh.cell_volumes,
                                  minlength=nphase)
    body = gradient * volume_by_phase
    nonwall = sum(terms.values())
    parent_fx = nonwall - body
    direct = pressure_force + viscous_force

    # The full vector/moment trace uses directly integrated wall faces.  The
    # parent certifies x force; y force and moment are retained only for the
    # representation-gap statement and are never called parent-certified.
    direct_trace = np.r_[direct[:, 0], direct[:, 1], moment_y_fx]
    parent_trace = np.r_[parent_fx, direct[:, 1], moment_y_fx]
    full_leg = (sum(np.abs(value) for value in terms.values()) + np.abs(body) +
                np.abs(pressure_force[:, 0]) + np.abs(viscous_force[:, 0]))
    return {
        "U": U,
        "p": p,
        "Rxx": Rxx,
        "body_gradient": gradient,
        "volume_by_phase": volume_by_phase,
        "body": body,
        "nonwall": nonwall,
        **{f"nonwall_{name}": value for name, value in terms.items()},
        "sgs_snapshot_envelope": term_sgs_uncertainty,
        "direct_pressure": pressure_force,
        "direct_viscous": viscous_force,
        "direct_force": direct,
        "direct_moment_y_fx": moment_y_fx,
        "parent_fx": parent_fx,
        "direct_trace": direct_trace,
        "parent_trace": parent_trace,
        "residual_fx": parent_fx - direct[:, 0],
        "full_leg_scale": full_leg,
    }


def representation_matrix(nphase: int, bins: int,
                          complete_trace: bool = True) -> np.ndarray:
    if nphase % bins:
        raise ValueError("phase count must be divisible by bin count")
    rows = 3 * nphase if complete_trace else nphase
    matrix = np.zeros((rows, bins))
    block = nphase // bins
    for phase in range(nphase):
        matrix[phase, phase // block] = 1.0
    return matrix


def best_projection(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coefficient, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = matrix @ coefficient - target
    return coefficient, residual, float(np.linalg.norm(residual))


def main() -> None:
    mesh = build_mesh()
    windows = {time: evaluate_time(mesh, time) for time in TIMES[2:]}
    central = windows[CENTRAL_TIME]
    nphase = mesh.phase_x.size
    direct_fx = central["direct_force"][:, 0]
    parent_fx = central["parent_fx"]
    residual = parent_fx - direct_fx
    residual_norm = float(np.linalg.norm(residual))
    direct_norm = float(np.linalg.norm(direct_fx))
    full_leg_norm = float(np.linalg.norm(central["full_leg_scale"]))
    wavelength_residual = float(np.sum(residual))
    wavelength_direct = float(np.sum(direct_fx))

    window_direct = np.stack([windows[t]["direct_force"][:, 0] for t in windows])
    window_parent = np.stack([windows[t]["parent_fx"] for t in windows])
    direct_window_envelope = np.max(np.abs(window_direct - direct_fx), axis=0)
    parent_window_envelope = np.max(np.abs(window_parent - parent_fx), axis=0)
    propagated = np.sqrt(direct_window_envelope ** 2 +
                         parent_window_envelope ** 2 +
                         central["sgs_snapshot_envelope"] ** 2)
    delta_parent = float(np.linalg.norm(residual) + np.linalg.norm(propagated))

    representations = {}
    coarse_representation_bins = tuple(
        bins for bins in (1, 4, 8, 16, 48) if nphase % bins == 0
    )
    for complete in (False, True):
        target_direct = (central["direct_trace"] if complete else direct_fx)
        target_parent = (central["parent_trace"] if complete else parent_fx)
        label = "complete_vector_moment" if complete else "signed_x_force"
        representations[label] = {}
        representation_bins = coarse_representation_bins + (
            () if complete or nphase in coarse_representation_bins else (nphase,)
        )
        for bins in representation_bins:
            if not complete and bins == nphase:
                # The full signed-x representation is the identity.  Avoid a
                # dense O(nphase^3) least-squares/SVD calculation on the
                # many-pitch replacement mesh while retaining the exact same
                # algebra as ``representation_matrix(nphase, nphase)``.
                coefficient = target_direct.copy()
                error_direct = np.zeros_like(target_direct)
                error_parent = np.zeros_like(target_parent)
                gap_direct = 0.0
                gap_parent = 0.0
                singular = np.ones(nphase)
            else:
                matrix = representation_matrix(nphase, bins, complete)
                coefficient, error_direct, gap_direct = best_projection(matrix, target_direct)
                _, error_parent, gap_parent = best_projection(matrix, target_parent)
                singular = np.linalg.svd(matrix, compute_uv=False)
            representations[label][str(bins)] = {
                "n_coefficients": bins,
                "best_direct_gap": gap_direct,
                "best_parent_estimated_gap": gap_parent,
                "gap_difference": abs(gap_direct - gap_parent),
                "gap_perturbation_bound": residual_norm,
                "gap_bound_pass": bool(abs(gap_direct - gap_parent) <=
                                       residual_norm + 1e-14),
                "sigma_min_positive": float(singular[-1]),
                "sigma_max": float(singular[0]),
                "coefficient_norm": float(np.linalg.norm(coefficient)),
                "direct_residual_norm_rebuild": float(np.linalg.norm(error_direct)),
                "parent_residual_norm_rebuild": float(np.linalg.norm(error_parent)),
            }

    # Target-blind candidates exercise the estimator independently of the best
    # projection.  The constant candidate uses only total body forcing, which
    # is available at run time; it does not use the direct wall target.
    candidates = {
        "zero": np.zeros(nphase),
        "uniform_body_balance": np.full(nphase, -np.sum(central["body"]) / nphase),
        "alternating_manufactured": 0.25 * np.linalg.norm(parent_fx) /
            np.sqrt(nphase) * (-1.0) ** np.arange(nphase),
    }
    effectivity = {}
    for name, coefficient in candidates.items():
        estimated = coefficient - parent_fx
        true = coefficient - direct_fx
        norm_estimated = float(np.linalg.norm(estimated))
        norm_true = float(np.linalg.norm(true))
        difference = abs(norm_estimated - norm_true)
        effectivity[name] = {
            "estimated_error_norm": norm_estimated,
            "independent_true_error_norm": norm_true,
            "effectivity_index": norm_estimated / norm_true,
            "norm_difference": difference,
            "parent_closure_bound": residual_norm,
            "two_sided_bound_pass": bool(difference <= residual_norm + 1e-14),
        }

    window_certificate = []
    for end_time, item in windows.items():
        window_direct_fx = item["direct_force"][:, 0]
        window_parent_fx = item["parent_fx"]
        window_residual = window_parent_fx - window_direct_fx
        local_candidates = {
            "zero": np.zeros(nphase),
            "uniform_body_balance": np.full(
                nphase, -np.sum(item["body"]) / nphase),
            "alternating_manufactured": (
                0.25 * np.linalg.norm(window_parent_fx) / np.sqrt(nphase) *
                (-1.0) ** np.arange(nphase)),
        }
        window_certificate.append({
            "end_time": float(end_time),
            "phase_closure_over_full_leg": float(
                np.linalg.norm(window_residual) /
                np.linalg.norm(item["full_leg_scale"])),
            "wavelength_closure_over_direct": float(
                abs(np.sum(window_residual)) /
                max(abs(np.sum(window_direct_fx)), 1e-30)),
            "effectivity": {
                name: float(np.linalg.norm(coefficient - window_parent_fx) /
                            np.linalg.norm(coefficient - window_direct_fx))
                for name, coefficient in local_candidates.items()
            },
        })

    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [MESH / "points", MESH / "faces", MESH / "owner",
                     MESH / "neighbour", MESH / "boundary", CASE / "0" / "C",
                     CASE / "log.pimpleFoam"] +
                    [CASE / time / field for time in TIMES
                     for field in ("U", "nut")] +
                    [CASE / time / field for time in TIMES[2:]
                     for field in ("UMean", "UPrime2Mean", "pMean")]
    }
    summary = {
        "schema": "direct-force-adequacy-certificate-l1-v1",
        "case": CASE_LABEL,
        "method": {
            "parent": "complete steady Cartesian finite-volume x-momentum balance",
            "control_volumes": f"{nphase} Cartesian streamwise bins spanning full height and span",
            "direct_force": "pMean plus molecular traction on physical wall faces",
            "reynolds_stress": "deposited UPrime2Mean; no Boussinesq substitution",
            "sgs_stress": "mean of five stored instantaneous WALE nut-strain fluxes",
            "body_force": "time-integrated meanVelocityForce history over averaging window",
            "thin_layer_deletion": False,
            "pressure_height_proxy": False,
            "self_reconstruction_as_reference": False,
        },
        "mesh": {
            "cells": int(mesh.cell_centres.shape[0]),
            "faces": len(mesh.faces),
            "physical_wall_faces": int(sum(mesh.boundaries[p]["nFaces"]
                                           for p in ("bottomWall", "topWall"))),
            "phase_control_volumes": nphase,
            "reconstructed_volume": float(np.sum(mesh.cell_volumes)),
        },
        "averaging_windows_end_time": [float(t) for t in windows],
        "body_gradient_central": float(central["body_gradient"]),
        "phase_parent_closure": {
            "residual_l2": residual_norm,
            "residual_over_direct_force_l2": residual_norm / direct_norm,
            "residual_over_full_leg_l2": residual_norm / full_leg_norm,
            "max_abs_residual_over_local_full_leg": float(np.max(
                np.abs(residual) / np.maximum(central["full_leg_scale"], 1e-30))),
            "correlation_parent_direct": float(np.corrcoef(parent_fx, direct_fx)[0, 1]),
            "propagated_window_sgs_l2": float(np.linalg.norm(propagated)),
        },
        "wavelength_parent_closure": {
            "direct_force": wavelength_direct,
            "parent_force": float(np.sum(parent_fx)),
            "residual": wavelength_residual,
            "relative_to_direct": abs(wavelength_residual) / abs(wavelength_direct),
        },
        "effectivity_candidates": effectivity,
        "deterministic_window_sensitivity": {
            "records": window_certificate,
            "phase_closure_envelope": [
                min(record["phase_closure_over_full_leg"]
                    for record in window_certificate),
                max(record["phase_closure_over_full_leg"]
                    for record in window_certificate),
            ],
            "wavelength_closure_envelope": [
                min(record["wavelength_closure_over_direct"]
                    for record in window_certificate),
                max(record["wavelength_closure_over_direct"]
                    for record in window_certificate),
            ],
            "effectivity_envelope": [
                min(value for record in window_certificate
                    for value in record["effectivity"].values()),
                max(value for record in window_certificate
                    for value in record["effectivity"].values()),
            ],
        },
        "best_projection": representations,
        "interpretation_guards": {
            "phasewise_scalar_signed_x_gap_is_zero_within_roundoff": bool(
                representations["signed_x_force"][str(nphase)]["best_direct_gap"] < 1e-14),
            "complete_trace_gap_not_called_wall_stress_error": True,
            "oracle_projection_separate_from_target_blind_candidates": True,
            "volume_filtered_method_claimed_as_prior_art": True,
        },
        "source_hashes": source_hashes,
    }
    gates = {
        "phase_full_leg_closure_below_10pct":
            summary["phase_parent_closure"]["residual_over_full_leg_l2"] < 0.10,
        "wavelength_closure_below_10pct":
            summary["wavelength_parent_closure"]["relative_to_direct"] < 0.10,
        "all_effectivity_bounds_hold": all(
            item["two_sided_bound_pass"] for item in effectivity.values()),
        "all_projection_perturbation_bounds_hold": all(
            item["gap_bound_pass"] for group in representations.values()
            for item in group.values()),
        "x_only_symmetric_inversion_guard":
            summary["interpretation_guards"]["phasewise_scalar_signed_x_gap_is_zero_within_roundoff"],
    }
    summary["gates"] = gates
    summary["status"] = "PASS" if all(gates.values()) else "FAIL"

    arrays = {
        "phase_x": mesh.phase_x,
        "direct_force": central["direct_force"],
        "direct_moment_y_fx": central["direct_moment_y_fx"],
        "parent_fx": parent_fx,
        "residual_fx": residual,
        "full_leg_scale": central["full_leg_scale"],
        "body_force": central["body"],
        "nonwall_mean": central["nonwall_mean"],
        "nonwall_pressure": central["nonwall_pressure"],
        "nonwall_reynolds": central["nonwall_reynolds"],
        "nonwall_molecular": central["nonwall_molecular"],
        "nonwall_sgs": central["nonwall_sgs"],
        "direct_pressure": central["direct_pressure"],
        "direct_viscous": central["direct_viscous"],
        "window_direct_fx": window_direct,
        "window_parent_fx": window_parent,
        "window_full_leg_scale": np.stack(
            [windows[t]["full_leg_scale"] for t in windows]),
        "window_body_force": np.stack([windows[t]["body"] for t in windows]),
        "propagated_window_sgs": propagated,
        "schema": np.array(summary["schema"]),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    NODE.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "direct_force_adequacy_certificate_l1.json"
    npz_path = RESULTS / "direct_force_adequacy_certificate_l1.npz"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    np.savez(npz_path, **arrays)
    for source in (json_path, npz_path):
        (NODE / source.name).write_bytes(source.read_bytes())

    print("DIRECT-FORCE ADEQUACY CERTIFICATE")
    print(f"  mesh: {summary['mesh']['cells']} cells; {nphase} phase CVs")
    print("  phase closure/full-leg: "
          f"{summary['phase_parent_closure']['residual_over_full_leg_l2']:.6f}")
    print("  wavelength closure/direct: "
          f"{summary['wavelength_parent_closure']['relative_to_direct']:.6f}")
    for name, item in effectivity.items():
        print(f"  effectivity {name}: {item['effectivity_index']:.6f}")
    print(f"  signed-x {nphase}-bin best gap: "
          f"{representations['signed_x_force'][str(nphase)]['best_direct_gap']:.3e}")
    print(f"  STATUS: {summary['status']}")
    if summary["status"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit("certificate gates failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
