#!/usr/bin/env python3
"""Reduce the terminal common-surface 3-grid x 2-model ARCHER2 campaign.

The wall stress is projected onto the downstream-oriented *physical* wall
tangent.  The OpenFOAM surface sampler returns the wall-on-fluid traction with
the opposite sign, hence ``tau_s = -wallShearStressMean . t_s``.  Matching
heights are measured from mesh topology at every phase, not inferred from the
flat-floor grading check.  R^2 is descriptive; signed force, zero crossings,
reversed-shear coverage and RMS-normalised error are the primary estimands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT / "codes" / "results" /
    "rswm_xiao_dns_grid_campaign_final_l2"
)
DNS_FILE = ROOT / "codes" / "results" / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
DNS_EXPECTED_SHA256 = "d039cefb93ec1a8555555deed79041921bf8ce98cd1477479087a9804ca7ff85"
GEOMETRY_README = (
    ROOT / "codes" / "raw_data" / "geometry_driven" /
    "xiao_pehill_parameterized" / "pehill-29-cases-DNS" / "README_NEWDATABASE.pdf"
)
GEOMETRY_SOURCE = (
    ROOT / "codes" / "raw_data" / "geometry_driven" /
    "xiao_pehill_parameterized" / "utility" /
    "hill-geometry-gereration" / "hillShape.py"
)
MESH_GENERATOR = ROOT / "codes" / "openfoam" / "make_xiao_dns_wmles_case.py"
OUT_NPZ = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2.npz"
OUT_JSON = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2_summary.json"
OUT_MANIFEST = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2_manifest.json"
NODE_JSON = ROOT / "development" / "nodes" / "node_003" / "results_summary.json"
OUT_FIG = ROOT / "development" / "nodes" / "node_003" / "fig_common_surface_grid_l2.pdf"
OUT_PNG = ROOT / "development" / "nodes" / "node_003" / "fig_common_surface_grid_l2.png"

GRIDS = ("G0", "G1c", "G2c")
MODELS = ("equilibrium", "total_gradient_tble")
CASES = {
    ("G0", "equilibrium"): "rswm_xiao_dns_g0_equilibrium_92160_l2_v1",
    ("G0", "total_gradient_tble"): "rswm_xiao_dns_g0_tble_92160_l2_v1",
    ("G1c", "equilibrium"): "rswm_xiao_dns_g1_equilibrium_307200_l2_v1",
    ("G1c", "total_gradient_tble"): "rswm_xiao_dns_g1_tble_307200_l2_v1",
    ("G2c", "equilibrium"): "rswm_xiao_dns_g2_equilibrium_819200_l2_v1",
    ("G2c", "total_gradient_tble"): "rswm_xiao_dns_g2_tble_819200_l2_v1",
}
CELLS = {"G0": 92160, "G1c": 307200, "G2c": 819200}
LX = 9.0
LY = 3.036
LZ = 4.5
HILL_HALF_WIDTH = 54.0 / 28.0

sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(
    ROOT / "codes" / "raw_data" / "geometry_driven" /
    "xiao_pehill_parameterized" / "utility" / "hill-geometry-gereration"
))
from verify_common_matching_surface import (  # noqa: E402
    newell_normal,
    read_faces,
    read_labels,
    read_patch,
    read_points,
    read_vector_field,
)
from da_budget import periodic_derivative  # noqa: E402
from hillShape import profile as xiao_hill_profile  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_interp(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0], np.r_[y, y, y])


def sample_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append([float(token) for token in line.split()])
    values = np.asarray(rows, float)
    if values.ndim != 2 or values.shape[1] < 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid sampled wall data: {path}")
    return values


def face_area(vertices: np.ndarray) -> float:
    area_vector = np.zeros(3)
    for index, current in enumerate(vertices):
        area_vector += np.cross(current, vertices[(index + 1) % len(vertices)])
    return 0.5 * float(np.linalg.norm(area_vector))


def mesh_bottom(case: Path) -> dict[str, np.ndarray]:
    mesh = case / "input" / "polyMesh"
    centres = read_vector_field(case / "input" / "C")
    points = read_points(mesh / "points")
    faces = read_faces(mesh / "faces")
    owners = read_labels(mesh / "owner")
    start, count = read_patch(mesh / "boundary", "bottomWall")
    xyz, tangent, distance, area = [], [], [], []
    for face_index in range(start, start + count):
        vertices = points[faces[face_index]]
        face_centre = vertices.mean(axis=0)
        normal = newell_normal(vertices)
        t_s = np.array([-normal[1], normal[0], 0.0])
        if t_s[0] < 0.0:
            t_s *= -1.0
        t_s /= np.linalg.norm(t_s)
        xyz.append(face_centre)
        tangent.append(t_s)
        distance.append(abs(float(np.dot(centres[owners[face_index]] - face_centre, normal))))
        area.append(face_area(vertices))
    arrays = {
        "xyz": np.asarray(xyz),
        "tangent": np.asarray(tangent),
        "ym": np.asarray(distance),
        "area": np.asarray(area),
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError(f"non-finite mesh extraction in {case}")
    return arrays


def align_sample(mesh: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    mesh_order = np.lexsort((mesh["xyz"][:, 2], mesh["xyz"][:, 0]))
    sample_order = np.lexsort((rows[:, 2], rows[:, 0]))
    if len(mesh_order) != len(sample_order):
        raise ValueError("mesh/sample face-count mismatch")
    mismatch = np.max(np.linalg.norm(mesh["xyz"][mesh_order] - rows[sample_order, :3], axis=1))
    if mismatch > 2.0e-6:
        raise ValueError(f"mesh/sample centre mismatch {mismatch:.3e}")
    aligned = np.empty_like(rows)
    aligned[mesh_order] = rows[sample_order]
    return aligned


def phase_reduce(mesh: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray | float]:
    rows = align_sample(mesh, rows)
    traction = -rows[:, 3:6]
    tau_s_face = np.einsum("ij,ij->i", traction, mesh["tangent"])
    tau_x_face = traction[:, 0]
    rounded = np.round(mesh["xyz"][:, 0], 9)
    x_unique, inverse = np.unique(rounded, return_inverse=True)
    z_unique = np.unique(np.round(mesh["xyz"][:, 2], 9))
    if len(z_unique) < 2:
        raise ValueError("spanwise extent cannot be recovered from one face centre")
    dz = float(np.median(np.diff(z_unique)))
    span = float(np.ptp(z_unique) + dz)
    tau_s, tau_x, ywall, ym, wall_ds, tangent_x, tangent_y = [], [], [], [], [], [], []
    for index in range(len(x_unique)):
        chosen = inverse == index
        weights = mesh["area"][chosen]
        tau_s.append(np.average(tau_s_face[chosen], weights=weights))
        tau_x.append(np.average(tau_x_face[chosen], weights=weights))
        ywall.append(np.average(mesh["xyz"][chosen, 1], weights=weights))
        ym.append(np.average(mesh["ym"][chosen], weights=weights))
        wall_ds.append(np.sum(weights) / span)
        mean_tangent = np.average(mesh["tangent"][chosen], axis=0, weights=weights)
        mean_tangent /= np.linalg.norm(mean_tangent)
        tangent_x.append(mean_tangent[0])
        tangent_y.append(mean_tangent[1])
    area = mesh["area"]
    return {
        "phase": np.asarray(x_unique) / LX,
        "x": np.asarray(x_unique),
        "tau_s": np.asarray(tau_s),
        "tau_x": np.asarray(tau_x),
        "ywall": np.asarray(ywall),
        "ym": np.asarray(ym),
        "wall_ds": np.asarray(wall_ds),
        "tangent_x": np.asarray(tangent_x),
        "tangent_y": np.asarray(tangent_y),
        "dz": dz,
        "signed_tangent_force_per_span": float(np.sum(tau_s_face * area) / span),
        "signed_x_force_per_span": float(np.sum(tau_x_face * area) / span),
    }


def zero_crossings(phase: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    phase = np.asarray(phase)
    values = np.asarray(values)
    sep, rea = [], []
    for index in range(len(values)):
        nxt = (index + 1) % len(values)
        a, b = values[index], values[nxt]
        if a == 0.0 or a * b < 0.0:
            xa = phase[index]
            xb = phase[nxt] + (1.0 if nxt == 0 else 0.0)
            fraction = 0.0 if a == b else -a / (b - a)
            crossing = (xa + fraction * (xb - xa)) % 1.0
            (sep if a >= 0.0 and b < 0.0 else rea).append(crossing)
    if not sep or not rea:
        return math.nan, math.nan
    candidates = []
    for separation in sep:
        downstream = [((reattachment - separation) % 1.0, reattachment)
                      for reattachment in rea
                      if (reattachment - separation) % 1.0 > 1.0e-12]
        if downstream:
            length, reattachment = min(downstream)
            candidates.append((length, separation, reattachment))
    if not candidates:
        return math.nan, math.nan
    _, separation, reattachment = max(candidates, key=lambda item: item[0])
    return float(separation), float(reattachment)


def metrics(
    curve: dict[str, Any],
    truth_phase: np.ndarray,
    truth_tau: np.ndarray,
    truth_tau_x_legacy: np.ndarray | None = None,
) -> dict[str, float]:
    # Score every grid on the same dense, uniformly-spaced phase array.  This
    # prevents a changing number/distribution of wall faces from changing the
    # metric quadrature as the mesh is refined.
    dense_phase = np.arange(4096, dtype=float) / 4096.0
    tau_ref = periodic_interp(truth_phase, truth_tau, dense_phase)
    tau = periodic_interp(np.asarray(curve["phase"]), np.asarray(curve["tau_s"]), dense_phase)
    error = tau - tau_ref
    denom = float(np.sum((tau_ref - np.mean(tau_ref)) ** 2))
    sep, rea = zero_crossings(dense_phase, tau)
    sep_ref, rea_ref = zero_crossings(dense_phase, tau_ref)
    reference_on_faces = periodic_interp(
        truth_phase, truth_tau, np.asarray(curve["phase"])
    )
    wall_ds = np.asarray(curve["wall_ds"])
    tangent_x = np.asarray(curve["tangent_x"])
    truth_tangent_force = float(np.sum(reference_on_faces * wall_ds))
    truth_x_force = float(np.sum(reference_on_faces * tangent_x * wall_ds))
    tangent_force_scale = max(float(np.sum(np.abs(reference_on_faces) * wall_ds)), 1.0e-14)
    x_force_scale = max(
        float(np.sum(np.abs(reference_on_faces * tangent_x) * wall_ds)), 1.0e-14
    )
    model_tangent_force = float(curve["signed_tangent_force_per_span"])
    model_x_force = float(curve["signed_x_force_per_span"])
    result = {
        "r2": float(1.0 - np.sum(error**2) / denom),
        "relative_rms": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(tau_ref**2))),
        "cf_rms": float(2.0 * np.sqrt(np.mean(tau**2))),
        "cf_error_rms": float(2.0 * np.sqrt(np.mean(error**2))),
        "sign_accuracy": float(np.mean(np.sign(tau) == np.sign(tau_ref))),
        "reversed_fraction": float(np.mean(tau < 0.0)),
        "truth_reversed_fraction": float(np.mean(tau_ref < 0.0)),
        "separation_phase": sep,
        "reattachment_phase": rea,
        "truth_separation_phase": sep_ref,
        "truth_reattachment_phase": rea_ref,
        "separation_x_over_H": float(sep * LX),
        "reattachment_x_over_H": float(rea * LX),
        "truth_separation_x_over_H": float(sep_ref * LX),
        "truth_reattachment_x_over_H": float(rea_ref * LX),
        "signed_tangent_force_per_span": model_tangent_force,
        "truth_signed_tangent_force_per_span": truth_tangent_force,
        "signed_tangent_force_error_over_abs_truth": (
            model_tangent_force - truth_tangent_force
        ) / tangent_force_scale,
        "signed_x_force_per_span": model_x_force,
        "truth_signed_x_force_per_span": truth_x_force,
        "signed_x_force_error_over_abs_truth": (
            model_x_force - truth_x_force
        ) / x_force_scale,
    }
    if truth_tau_x_legacy is not None:
        legacy_ref = periodic_interp(truth_phase, truth_tau_x_legacy, dense_phase)
        model_x = periodic_interp(
            np.asarray(curve["phase"]), np.asarray(curve["tau_x"]), dense_phase
        )
        legacy_error = model_x - legacy_ref
        legacy_variance = float(np.sum((legacy_ref - np.mean(legacy_ref)) ** 2))
        result["legacy_x_r2"] = float(
            1.0 - np.sum(legacy_error**2) / legacy_variance
        )
        result["legacy_x_relative_rms"] = float(
            np.sqrt(np.mean(legacy_error**2)) /
            max(np.sqrt(np.mean(legacy_ref**2)), 1.0e-14)
        )
    return result


def path_convergence(values: list[float], cells: list[int]) -> dict[str, Any]:
    """Describe the registered anisotropic refinement path without false GCI.

    The three meshes change streamwise, wall-normal and spanwise resolution at
    different ratios and also change wall-normal grading.  There is therefore
    no defensible scalar h or formal observed order.  We report the complete
    path, successive signed changes, monotonicity and its measured envelope.
    """
    fc, fm, ff = map(float, values)
    if not np.all(np.isfinite([fc, fm, ff])):
        return {"status": "undefined_nonfinite", "values": [fc, fm, ff]}
    coarse_to_middle = fm - fc
    middle_to_fine = ff - fm
    tolerance = 1.0e-14 * max(abs(fc), abs(fm), abs(ff), 1.0)
    monotone = (
        abs(coarse_to_middle) <= tolerance
        or abs(middle_to_fine) <= tolerance
        or coarse_to_middle * middle_to_fine > 0.0
    )
    diminishing = abs(middle_to_fine) <= abs(coarse_to_middle) + tolerance
    return {
        "status": (
            "monotone_diminishing" if monotone and diminishing else
            "monotone_non_diminishing" if monotone else "nonmonotone"
        ),
        "cells": [int(value) for value in cells],
        "values": [fc, fm, ff],
        "coarse_to_middle": float(coarse_to_middle),
        "middle_to_fine": float(middle_to_fine),
        "envelope": float(max(values) - min(values)),
        "fine_change_over_fine_magnitude": float(
            abs(middle_to_fine) / max(abs(ff), 1.0e-14)
        ),
    }


def dns_tangent_reference(dns: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover physical tangent traction from deposited Xiao U,V profiles.

    The canonical corrected deposit stores vertical wall columns and its
    legacy ``tau_w`` is ``nu*dU/dy``.  On a sloped no-slip wall the mechanical
    scalar needed here is ``nu*dU_t/dn``.  At the wall ``dU_t/ds=0`` and
    ``dy/dn=t_x``, so the first-four-point through-origin fit gives
    ``tau_s=nu*(dU_t/dy)/t_x``.  The tangent is computed from the independent
    analytic Xiao surface, not imported from any mesh being evaluated.
    """
    phase = np.mod(
        (np.asarray(dns["x"], float) - float(np.min(dns["x"]))) /
        LX,
        1.0,
    )
    x = np.asarray(dns["x"], float)
    wall = np.asarray(xiao_hill_profile(x.copy()), float)
    slope = periodic_derivative(wall, x)
    magnitude = np.sqrt(1.0 + slope**2)
    tx, ty = 1.0 / magnitude, slope / magnitude
    if np.any(tx <= 0.0):
        raise ValueError("downstream wall tangent has a non-positive x component")

    y = np.asarray(dns["y"], float)
    u = np.asarray(dns["U"], float)
    v = np.asarray(dns["V"], float)
    nu = np.asarray(dns["nu"], float)
    tau_s = np.full(len(phase), np.nan)
    for index in range(len(phase)):
        yi = y[index] if y.ndim == 2 else y
        nui = float(nu[index] if nu.ndim else nu)
        use = np.arange(1, min(5, len(yi)))
        valid = np.isfinite(yi[use]) & np.isfinite(u[index, use]) & np.isfinite(v[index, use])
        use = use[valid]
        if len(use) < 2:
            continue
        wall_distance_y = yi[use]
        tangent_velocity = u[index, use] * tx[index] + v[index, use] * ty[index]
        denominator = float(np.sum(wall_distance_y**2))
        if denominator > 0.0:
            dut_dy = float(np.sum(wall_distance_y * tangent_velocity) / denominator)
            tau_s[index] = nui * dut_dy / tx[index]
    if not np.all(np.isfinite(tau_s)):
        raise ValueError("DNS tangent-traction reconstruction is incomplete")

    tau_x_derived = tau_s * tx
    tau_x_legacy = np.asarray(dns["tau_w"], float)
    scale = max(float(np.sqrt(np.mean(tau_x_legacy**2))), 1.0e-14)
    audit = {
        "derived_vs_legacy_x_relative_rms": float(
            np.sqrt(np.mean((tau_x_derived - tau_x_legacy) ** 2)) / scale
        ),
        "legacy_definition": "nu*dU/dy, first four fluid points",
        "tangent_definition": "nu*dU_t/dn from the same four points and independent analytic Xiao tangent",
        "period_over_H": LX,
    }
    return tau_s, audit


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def json_ready(value: Any) -> Any:
    """Return strict-JSON data, representing undefined scalars as null."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def profile_metrics(
    case: Path, checkpoint: str, dns: Any, surface_curve: dict[str, Any]
) -> dict[str, Any]:
    """Score harvested global-x mean-velocity profiles against Xiao DNS.

    OpenFOAM's line sampler records global ``y`` and omits points inside the
    hill, whereas the corrected DNS deposit is wall-reanchored.  The harvested
    wall shape is therefore subtracted before interpolation.  Stations are
    registered by periodic phase on the documented common period $L_x/H=9$.
    """
    directory = case / "postProcessing_sampleProfiles" / checkpoint
    all_paths = sorted(directory.glob("*.xy"))
    dns_x = np.asarray(dns["x"], float)
    dns_period = LX
    dns_phase = np.mod((dns_x - dns_x.min()) / dns_period, 1.0)
    dns_y = np.asarray(dns["y"], float)
    dns_u = np.asarray(dns["U"], float)
    dictionary = (case / "input" / "sampleProfiles").read_text()
    registered_x = {
        name: float(x_value)
        for name, x_value in re.findall(
            r"\b(x[0-9]+p[0-9]+)\s*\{[^}]*?start\s*\(\s*([0-9.eE+-]+)",
            dictionary,
            re.DOTALL,
        )
    }
    primary_x = np.asarray([LX * (index + 0.5) / 10.0 for index in range(10)])
    primary_names = {
        name for name, value in registered_x.items()
        if np.min(np.abs(primary_x - value)) < 1.0e-8
    }
    paths = [path for path in all_paths if path.stem in primary_names]
    if len(primary_names) != 10 or len(paths) != 10:
        raise ValueError(
            f"could not recover the ten registered Xiao phase profiles from {case}; "
            f"registered={len(registered_x)}, sampled={len(all_paths)}, primary={len(paths)}"
        )
    station_x, station_rms, station_points = [], [], []
    for path in paths:
        if path.stem not in registered_x:
            raise ValueError(f"unregistered sampled profile {path.name}")
        x_value = registered_x[path.stem]
        phase = (x_value / LX) % 1.0
        wall_y = float(periodic_interp(
            np.asarray(surface_curve["phase"]), np.asarray(surface_curve["ywall"]),
            np.asarray([phase]),
        )[0])
        sampled = np.loadtxt(path)
        if sampled.ndim != 2 or sampled.shape[1] != 4:
            raise ValueError(f"invalid mean-velocity profile {path}: {sampled.shape}")
        local_y = sampled[:, 0] - wall_y
        phase_distance = np.abs((dns_phase - phase + 0.5) % 1.0 - 0.5)
        index = int(np.argmin(phase_distance))
        valid_dns = np.isfinite(dns_y[index]) & np.isfinite(dns_u[index])
        yd, ud = dns_y[index, valid_dns], dns_u[index, valid_dns]
        valid = ((local_y >= yd.min()) & (local_y <= yd.max())
                 & np.isfinite(sampled[:, 1]))
        if np.count_nonzero(valid) < 32:
            raise ValueError(f"insufficient overlap in profile {path}")
        reference = np.interp(local_y[valid], yd, ud)
        station_x.append(x_value)
        station_rms.append(float(np.sqrt(np.mean((sampled[valid, 1] - reference) ** 2))))
        station_points.append(int(np.count_nonzero(valid)))
    return {
        "profile_x": np.asarray(station_x),
        "profile_u_rms_by_station": np.asarray(station_rms),
        "profile_points_by_station": np.asarray(station_points),
        "profile_u_rms_mean": float(np.mean(station_rms)),
        "profile_u_rms_max": float(np.max(station_rms)),
    }


def load_case(case_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    case = RESULT_ROOT / case_id
    manifest = json.loads((case / "MANIFEST.json").read_text())
    mesh = mesh_bottom(case)
    checkpoint_names = (case / "checkpoint_times_l2.txt").read_text().split()
    curves: dict[str, dict[str, Any]] = {}
    for name in checkpoint_names:
        path = case / "postProcessing_sampleBottomWall" / name / "bottomWall.xy"
        curves[name] = phase_reduce(mesh, sample_rows(path))
    return manifest, mesh, curves


def make_figure(
    curves: dict[tuple[str, str], dict[str, Any]],
    metrics_by_case: dict[tuple[str, str], dict[str, float]],
    truth_phase: np.ndarray,
    truth_tau: np.ndarray,
    surface_error: dict[str, dict[str, float]],
) -> None:
    colors = {"equilibrium": "#247a4d", "total_gradient_tble": "#687f95"}
    labels = {
        "equilibrium": "equilibrium (Spalding)",
        "total_gradient_tble": "vector-realizable total-gradient TBLE",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(truth_phase, 2.0 * truth_tau, color="#d97706", lw=2.2, label="DNS")
    for model in MODELS:
        curve = curves[("G2c", model)]
        ax.plot(curve["phase"], 2.0 * np.asarray(curve["tau_s"]), color=colors[model], lw=1.5,
                ls="-" if model == "total_gradient_tble" else "--", label=labels[model])
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set(xlabel=r"phase $x/L_x$", ylabel=r"signed $C_{f,s}=2\tau_s/U_b^2$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) Finest-grid wall traction", loc="left", fontsize=9)

    ax = axes[0, 1]
    for model in MODELS:
        values = [metrics_by_case[(grid, model)]["relative_rms"] for grid in GRIDS]
        ax.plot([CELLS[g] for g in GRIDS], values, "o-", color=colors[model], label=labels[model])
    ax.set_xscale("log")
    ax.set(xlabel="cells", ylabel="RMS error / DNS RMS")
    ax.set_title("(b) Error under refinement", loc="left", fontsize=9)

    ax = axes[1, 0]
    for model in MODELS:
        tangent = [metrics_by_case[(g, model)]["signed_tangent_force_per_span"] for g in GRIDS]
        xforce = [metrics_by_case[(g, model)]["signed_x_force_per_span"] for g in GRIDS]
        ax.plot([CELLS[g] for g in GRIDS], tangent, "o-", color=colors[model])
        ax.plot([CELLS[g] for g in GRIDS], xforce, "s:", color=colors[model], alpha=0.75)
    ax.set_xscale("log")
    ax.set(xlabel="cells", ylabel="integrated viscous shear / span")
    ax.set_title("(c) Tangential (circles) and x-shear (squares)", loc="left", fontsize=9)

    ax = axes[1, 1]
    rms = [surface_error[g]["rms_relative"] for g in GRIDS]
    maximum = [surface_error[g]["max_relative"] for g in GRIDS]
    ax.plot([CELLS[g] for g in GRIDS], rms, "o-", color="#2f3b46", label="RMS")
    ax.plot([CELLS[g] for g in GRIDS], maximum, "s--", color="#2f3b46", alpha=0.7, label="max")
    ax.set_xscale("log")
    # G0 is the reference matching surface and therefore has exactly zero
    # mismatch.  A linear ordinate displays that datum honestly and avoids the
    # duplicated near-zero tick labels produced by a symmetric-log transform.
    ax.set_ylim(-2.5e-4, 1.15 * max(maximum))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.set(xlabel="cells", ylabel="relative matching-height mismatch")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(d) Full curved matching surface", loc="left", fontsize=9)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if not RESULT_ROOT.is_dir():
        raise SystemExit(f"terminal campaign absent: {RESULT_ROOT}")

    if sha256(DNS_FILE) != DNS_EXPECTED_SHA256:
        raise SystemExit("authoritative corrected DNS file identity changed")
    dns = np.load(DNS_FILE)
    truth_x = np.asarray(dns["x"], float)
    if not (
        truth_x.shape == (512,)
        and np.all(np.isfinite(truth_x))
        and np.all(np.diff(truth_x) > 0.0)
        and abs(float(truth_x[0])) < 1.0e-12
        and abs(float(np.median(np.diff(truth_x))*truth_x.size) - LX) < 1.0e-3
    ):
        raise SystemExit("corrected DNS is not the registered 512-phase Lx/H=9 deposit")
    truth_period = LX
    truth_phase = np.mod((truth_x - truth_x.min()) / truth_period, 1.0)
    truth_tau_x_legacy = np.asarray(dns["tau_w"], float)

    manifests, meshes = {}, {}
    curves: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_curves: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for key, case_id in CASES.items():
        manifest, mesh, case_curves = load_case(case_id)
        if manifest["grid"] != key[0] or manifest["model"] != key[1]:
            raise ValueError(f"manifest identity mismatch for {case_id}")
        if args.verify_hashes:
            for relative, record in manifest["files"].items():
                path = RESULT_ROOT / case_id / relative
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError(f"manifest mismatch: {case_id}/{relative}")
        manifests[key], meshes[key], checkpoint_curves[key] = manifest, mesh, case_curves
        curves[key] = case_curves[(RESULT_ROOT / case_id / "checkpoint_times_l2.txt").read_text().split()[-1]]

    truth_tau, truth_tangent_audit = dns_tangent_reference(dns)

    reference = curves[("G0", "equilibrium")]
    surface_error: dict[str, dict[str, float]] = {}
    geometry_error: dict[str, dict[str, float]] = {}
    for grid in GRIDS:
        current = curves[(grid, "equilibrium")]
        ref_ym = periodic_interp(np.asarray(reference["phase"]), np.asarray(reference["ym"]),
                                 np.asarray(current["phase"]))
        difference = np.asarray(current["ym"]) - ref_ym
        scale = float(np.mean(ref_ym))
        surface_error[grid] = {
            "rms_relative": float(np.sqrt(np.mean(difference**2)) / scale),
            "max_relative": float(np.max(np.abs(difference)) / scale),
        }
        analytic_wall = np.asarray(xiao_hill_profile(np.asarray(current["x"]).copy()), float)
        geometry_difference = np.asarray(current["ywall"]) - analytic_wall
        geometry_error[grid] = {
            "rms_over_H": float(np.sqrt(np.mean(geometry_difference**2))),
            "max_over_H": float(np.max(np.abs(geometry_difference))),
        }
        paired = curves[(grid, "total_gradient_tble")]
        if not np.allclose(current["ym"], paired["ym"], rtol=0.0, atol=2e-12):
            raise ValueError(f"two models do not share a matching surface on {grid}")

    metrics_by_case = {
        key: metrics(curve, truth_phase, truth_tau, truth_tau_x_legacy)
        for key, curve in curves.items()
    }
    profile_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    for key, case_id in CASES.items():
        checkpoint = (RESULT_ROOT / case_id / "checkpoint_times_l2.txt").read_text().split()[-1]
        profile = profile_metrics(RESULT_ROOT / case_id, checkpoint, dns, curves[key])
        profile_by_case[key] = profile
        metrics_by_case[key]["profile_u_rms_mean"] = profile["profile_u_rms_mean"]
        metrics_by_case[key]["profile_u_rms_max"] = profile["profile_u_rms_max"]
    grid_resolution: dict[str, dict[str, Any]] = {}
    nu = float(np.median(np.asarray(dns["nu"], float)))
    for grid in GRIDS:
        curve = curves[(grid, "equilibrium")]
        tau_ref = periodic_interp(truth_phase, truth_tau, np.asarray(curve["phase"]))
        u_tau = np.sqrt(np.abs(tau_ref))
        grid_resolution[grid] = {
            "matching_cell_index": 1,
            "nx": int(len(curve["phase"])),
            "ny": int(manifests[(grid, "equilibrium")]["ny"]),
            "nz": int(manifests[(grid, "equilibrium")]["nz"]),
            "wall_arclength_dx_plus": quantiles(np.asarray(curve["wall_ds"]) * u_tau / nu),
            "first_cell_ym_plus": quantiles(np.asarray(curve["ym"]) * u_tau / nu),
            "spanwise_dz_plus": quantiles(float(curve["dz"]) * u_tau / nu),
            "wall_arclength_dx_over_H": quantiles(np.asarray(curve["wall_ds"])),
            "first_cell_ym_over_H": quantiles(np.asarray(curve["ym"])),
            "spanwise_dz_over_H": float(curve["dz"]),
        }
    averaging = {}
    for key, by_time in checkpoint_curves.items():
        names = list(by_time)
        last = np.asarray(by_time[names[-1]]["tau_s"])
        previous = np.asarray(by_time[names[-2]]["tau_s"])
        earlier = np.asarray(by_time[names[-3]]["tau_s"])
        norm = max(float(np.sqrt(np.mean(last**2))), 1.0e-14)
        checkpoint_metrics = [
            metrics(by_time[name], truth_phase, truth_tau, truth_tau_x_legacy)
            for name in names
        ]
        averaging[key] = {
            "change_180_to_225": float(np.sqrt(np.mean((previous - earlier) ** 2)) / norm),
            "change_225_to_270": float(np.sqrt(np.mean((last - previous) ** 2)) / norm),
            "reattachment_change_180_to_225_H": float(abs(
                checkpoint_metrics[1]["reattachment_x_over_H"] -
                checkpoint_metrics[0]["reattachment_x_over_H"]
            )),
            "reattachment_change_225_to_270_H": float(abs(
                checkpoint_metrics[2]["reattachment_x_over_H"] -
                checkpoint_metrics[1]["reattachment_x_over_H"]
            )),
            "r2_at_180": checkpoint_metrics[0]["r2"],
            "r2_at_225": checkpoint_metrics[1]["r2"],
            "r2_at_270": checkpoint_metrics[2]["r2"],
        }

    grid_path_convergence = {}
    for model in MODELS:
        for quantity in (
            "reattachment_x_over_H", "r2", "relative_rms", "reversed_fraction",
            "legacy_x_r2", "legacy_x_relative_rms", "profile_u_rms_mean",
            "signed_tangent_force_per_span",
            "signed_x_force_per_span",
        ):
            values = [metrics_by_case[(grid, model)][quantity] for grid in GRIDS]
            grid_path_convergence[f"{model}:{quantity}"] = path_convergence(
                values, [CELLS[grid] for grid in GRIDS]
            )

    computational_cost = {}
    for grid in GRIDS:
        eq_cost = manifests[(grid, "equilibrium")]["solver_cost"]
        tble_cost = manifests[(grid, "total_gradient_tble")]["solver_cost"]
        eq_unit = float(eq_cost["clock_seconds_per_bottom_face_step"])
        tble_unit = float(tble_cost["clock_seconds_per_bottom_face_step"])
        computational_cost[grid] = {
            "equilibrium": eq_cost,
            "total_gradient_tble": tble_cost,
            "tble_to_equilibrium_clock_per_face_step": tble_unit / eq_unit,
            "producer_elapsed_seconds": {
                model: int(manifests[(grid, model)]["producer_elapsed_seconds"])
                for model in MODELS
            },
        }

    summary = {
        "status": "RSWM_COMMON_SURFACE_GRID_L2_OK",
        "scope": "mapped initialisation followed by coupled WMLES; Xiao alpha=1 Re_H=5600 deposit geometry only",
        "models": list(MODELS),
        "grids": list(GRIDS),
        "cells": CELLS,
        "metrics": {f"{g}:{m}": values for (g, m), values in metrics_by_case.items()},
        "averaging_window_convergence": {f"{g}:{m}": values for (g, m), values in averaging.items()},
        "grid_resolution": grid_resolution,
        "full_surface_matching": surface_error,
        "grid_path_convergence": grid_path_convergence,
        "tble_realizability": {
            grid: manifests[(grid, "total_gradient_tble")]["tble_realizability_summary"]
            for grid in GRIDS
        },
        "wall_sample_sanity": {
            f"{grid}:{model}": manifests[(grid, model)]["wall_sample_sanity"]
            for grid in GRIDS for model in MODELS
        },
        "analytic_geometry_error": geometry_error,
        "computational_cost": computational_cost,
        "dns_tangent_reconstruction_audit": truth_tangent_audit,
        "dns_reference": str(DNS_FILE.relative_to(ROOT)),
        "dns_reference_file_sha256": sha256(DNS_FILE),
        "documented_period_over_H": LX,
        "documented_domain_over_H": {"Lx": LX, "Ly": LY, "Lz": LZ},
        "documented_hill_half_width_over_H": HILL_HALF_WIDTH,
        "geometry_readme": str(GEOMETRY_README.relative_to(ROOT)),
        "geometry_readme_sha256": sha256(GEOMETRY_README),
        "geometry_source": str(GEOMETRY_SOURCE.relative_to(ROOT)),
        "geometry_source_sha256": sha256(GEOMETRY_SOURCE),
        "mesh_generator": str(MESH_GENERATOR.relative_to(ROOT)),
        "mesh_generator_sha256": sha256(MESH_GENERATOR),
        "dns_tau_sha256": hashlib.sha256(np.ascontiguousarray(truth_tau).tobytes()).hexdigest(),
        "dns_legacy_tau_x_sha256": hashlib.sha256(
            np.ascontiguousarray(truth_tau_x_legacy).tobytes()
        ).hexdigest(),
        "producer_jobs": {f"{g}:{m}": manifests[(g, m)]["producer_job_id"] for g in GRIDS for m in MODELS},
    }
    encoded_summary = json.dumps(
        json_ready(summary), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    OUT_JSON.write_text(encoded_summary)
    NODE_JSON.write_text(encoded_summary)

    payload: dict[str, Any] = {
        "status": np.array(summary["status"]),
        "truth_phase": truth_phase,
        "truth_tau_s": truth_tau,
        "truth_tau_x_legacy": truth_tau_x_legacy,
        "grids": np.asarray(GRIDS),
        "models": np.asarray(MODELS),
        "cells": np.asarray([CELLS[grid] for grid in GRIDS]),
        "dns_tangent_vs_legacy_x_relative_rms": np.asarray(
            truth_tangent_audit["derived_vs_legacy_x_relative_rms"]
        ),
    }
    for grid in GRIDS:
        for model in MODELS:
            prefix = f"{grid}_{model}"
            curve = curves[(grid, model)]
            for name in (
                "phase", "x", "tau_s", "tau_x", "ym", "ywall", "wall_ds",
                "tangent_x", "tangent_y",
            ):
                payload[f"{prefix}_{name}"] = np.asarray(curve[name])
            for name, value in metrics_by_case[(grid, model)].items():
                payload[f"{prefix}_{name}"] = np.asarray(value)
            for name in ("profile_x", "profile_u_rms_by_station", "profile_points_by_station"):
                payload[f"{prefix}_{name}"] = np.asarray(profile_by_case[(grid, model)][name])
            for name, value in averaging[(grid, model)].items():
                payload[f"{prefix}_{name}"] = np.asarray(value)
            payload[f"{prefix}_dz"] = np.asarray(curve["dz"])
        payload[f"{grid}_matching_rms_relative"] = np.asarray(surface_error[grid]["rms_relative"])
        payload[f"{grid}_matching_max_relative"] = np.asarray(surface_error[grid]["max_relative"])
        payload[f"{grid}_tble_to_equilibrium_clock_per_face_step"] = np.asarray(
            computational_cost[grid]["tble_to_equilibrium_clock_per_face_step"]
        )
        for resolution_name, values in grid_resolution[grid].items():
            if isinstance(values, dict):
                for quantile, value in values.items():
                    payload[f"{grid}_{resolution_name}_{quantile}"] = np.asarray(value)
            else:
                payload[f"{grid}_{resolution_name}"] = np.asarray(values)
    np.savez_compressed(OUT_NPZ, **payload)
    make_figure(curves, metrics_by_case, truth_phase, truth_tau, surface_error)
    manifest_files = (
        RESULT_ROOT / "CAMPAIGN_MANIFEST.json",
        DNS_FILE,
        GEOMETRY_README,
        GEOMETRY_SOURCE,
        MESH_GENERATOR,
        Path(__file__).resolve(),
        OUT_NPZ,
        OUT_JSON,
        NODE_JSON,
        OUT_FIG,
        OUT_PNG,
    )
    analysis_manifest = {
        "status": "RSWM_COMMON_SURFACE_GRID_L2_ANALYSIS_HASH_OK",
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in manifest_files
        },
    }
    OUT_MANIFEST.write_text(json.dumps(
        analysis_manifest, indent=2, sort_keys=True
    ) + "\n")
    print(
        "RSWM_COMMON_SURFACE_GRID_L2_OK cases=6 "
        f"finest_eq_relRMS={metrics_by_case[('G2c', 'equilibrium')]['relative_rms']:.6g} "
        f"finest_tble_relRMS={metrics_by_case[('G2c', 'total_gradient_tble')]['relative_rms']:.6g} "
        f"matching_max={max(item['max_relative'] for item in surface_error.values()):.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
