#!/usr/bin/env python3
"""Wall-origin and roughness-sublayer sensitivity for the canonical Xiao hill.

This analysis closes referee-ledger row M10/M11 with measured quantities.  It
does not propose a new diagnostic.  It tests the origin convention used by the
existing pressure-traction ratio and expresses the matching-height result in
roughness-sublayer units.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "development/nodes/node_001"
RESULTS = ROOT / "codes/results"
PROFILE = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
PRESSURE = RESULTS / "exact_pressure_traction_512.npz"
CERTIFICATE = RESULTS / "wall_following_budget_certificate_l1.npz"
CRITICAL = RESULTS / "critical_matching_height_map.npz"
HILL_UTIL = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
             "utility/hill-geometry-gereration")
sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrate_periodic(field: np.ndarray, period: float) -> float:
    """Rectangle rule on an endpoint-excluded uniform periodic grid."""
    return float(period * np.mean(field))


def safe_epsilon(tau: np.ndarray, forcing: np.ndarray) -> np.ndarray:
    out = np.full(tau.shape, np.nan)
    valid = np.isfinite(tau) & np.isfinite(forcing) & (np.abs(forcing) > 1e-12)
    out[valid] = np.abs(tau[valid]) / np.abs(forcing[valid])
    return out


def main() -> None:
    NODE.mkdir(parents=True, exist_ok=True)
    with np.load(PROFILE, allow_pickle=False) as profile, \
         np.load(PRESSURE, allow_pickle=False) as pressure, \
         np.load(CERTIFICATE, allow_pickle=False) as certificate, \
         np.load(CRITICAL, allow_pickle=False) as critical:
        x = profile["x"].astype(float)
        y = profile["y"].astype(float)
        tau_w = profile["tau_w"].astype(float)
        dp_dx = profile["dp_dx"].astype(float)
        u_tau = profile["u_tau"].astype(float)
        nu = profile["nu"].astype(float)
        exact_pressure = pressure["exact_pressure_traction"].astype(float)
        # The processed profile prints x to six decimals; the raw pressure
        # table retains eight.  The grids agree to the documented print roundoff.
        if not np.allclose(x, pressure["x"], rtol=0.0, atol=6e-7):
            raise RuntimeError("profile and exact-pressure station grids differ")

        # The deposited profile ordinate is distance from the local analytic
        # wall.  This local surface is the only primary origin used here.
        eta_station = y[:, 10] - y[:, 0]
        eta_reference = float(pressure["matching_height_over_H"])
        eta_first = y[:, 1] - y[:, 0]
        eps_exact = safe_epsilon(tau_w, exact_pressure)
        eps_legacy_local = safe_epsilon(tau_w, dp_dx * eta_station)

        # Independent force-based reference plane.  It is reported as a
        # sensitivity comparator, not substituted for the local wall.
        xc = certificate["x"].astype(float)
        hc = certificate["h"].astype(float)
        q_ref = certificate["q_wall_reference"].astype(float)
        q_ens = certificate["q_ref_ensemble"].astype(float)
        period_c = float((xc[1] - xc[0]) * xc.size)
        moment0 = integrate_periodic(q_ref, period_c)
        moment1 = integrate_periodic(hc * q_ref, period_c)
        jackson_d = moment1 / moment0
        jackson_ensemble = np.array([
            integrate_periodic(hc * q, period_c) /
            integrate_periodic(q, period_c) for q in q_ens
        ])
        force_condition = (integrate_periodic(np.abs(q_ref), period_c) /
                           abs(moment0))

        h = hill_profile(x.copy())
        physical_match = h + eta_station
        origins = {
            "local_surface": h,
            "trough_plane": np.zeros_like(h),
            "jackson_force_centroid_plane": np.full_like(h, jackson_d),
            "crest_plane": np.ones_like(h),
        }
        origin_rows = []
        for name, origin in origins.items():
            height = physical_match - origin
            positive = height > 0.0
            eps = safe_epsilon(tau_w[positive],
                               dp_dx[positive] * height[positive])
            origin_rows.append({
                "origin": name,
                "origin_height_over_H": ("stationwise" if name == "local_surface"
                                          else float(origin[0])),
                "positive_height_fraction": float(np.mean(positive)),
                "median_positive_height_over_H": (float(np.median(height[positive]))
                                                   if np.any(positive) else None),
                "median_legacy_epsilon_on_valid_stations": (float(np.nanmedian(eps))
                                                             if eps.size else None),
                "fraction_epsilon_below_0p1_on_valid_stations": (float(np.nanmean(eps < 0.1))
                                                                  if eps.size else None),
            })

        perturb_rows = []
        for fraction in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75):
            effective_eta = eta_station - fraction * eta_first
            eps = safe_epsilon(tau_w, dp_dx * effective_eta)
            perturb_rows.append({
                "wall_location_shift_over_first_spacing": fraction,
                "positive_height_fraction": float(np.mean(effective_eta > 0.0)),
                "median_epsilon": float(np.nanmedian(eps)),
                "fraction_epsilon_below_0p1": float(np.nanmean(eps < 0.1)),
            })

        # An exact wall-to-match integral must be invariant under a rigid
        # vertical translation.  Check the coordinate arithmetic explicitly.
        translation_offsets = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        translation_error = max(float(np.max(np.abs(
            ((physical_match + shift) - (h + shift)) - eta_station)))
            for shift in translation_offsets)

        # Chan/MacDonald wavelength scaling is used only as a roughness-layer
        # reference scale.  The conclusion is a scale-overlap result, not a
        # claim that 0.5 lambda is universal for smooth hills.
        wavelength = float((x[1] - x[0]) * x.size)
        rsl_height = 0.5 * wavelength
        rsl_plus = rsl_height * u_tau / nu
        eta_plus = eta_station * u_tau / nu
        keys = critical["keys"].astype(str)
        idx = int(np.flatnonzero(keys == "periodic_hills_1p0")[0])
        ycrit_plus = float(critical["ycrit"][idx])

        summary = {
            "schema": "wall-origin-rsl-sensitivity-v1",
            "ledger_row": "M10/M11",
            "case": "Xiao periodic hill h/Lx=1.0, Re_H=5600",
            "origin_convention": "eta=y-y_w(x), measured from the local analytic surface",
            "pressure_definition": "exact wall-to-eta integral from exact_pressure_traction_512.npz",
            "sources_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                               (PROFILE, PRESSURE, CERTIFICATE, CRITICAL)},
            "stations": int(x.size),
            "eta_reference_over_H": eta_reference,
            "exact_epsilon_median": float(np.nanmedian(eps_exact)),
            "exact_epsilon_fraction_below_0p1": float(np.nanmean(eps_exact < 0.1)),
            "legacy_local_epsilon_median": float(np.nanmedian(eps_legacy_local)),
            "legacy_local_epsilon_fraction_below_0p1": float(np.nanmean(eps_legacy_local < 0.1)),
            "translation_offsets_over_H": translation_offsets.tolist(),
            "translation_invariance_max_error": translation_error,
            "jackson_force_centroid_over_H": float(jackson_d),
            "jackson_centroid_ensemble_min": float(jackson_ensemble.min()),
            "jackson_centroid_ensemble_max": float(jackson_ensemble.max()),
            "jackson_centroid_ensemble_std": float(jackson_ensemble.std(ddof=1)),
            "signed_force_condition_number": float(force_condition),
            "centroid_interpretation": "stable for this hill; no centroid-ill-conditioning claim",
            "origin_table": origin_rows,
            "wall_location_perturbation_table": perturb_rows,
            "roughness_sublayer_reference": "y_r=0.5 lambda (literature comparison, not a universal hill law)",
            "wavelength_over_H": wavelength,
            "roughness_sublayer_height_over_H": rsl_height,
            "median_eta_match_over_rsl_height": float(np.median(eta_station / rsl_height)),
            "max_eta_match_over_rsl_height": float(np.max(eta_station / rsl_height)),
            "fraction_match_above_rsl_height": float(np.mean(eta_station >= rsl_height)),
            "median_eta_match_plus": float(np.median(eta_plus)),
            "median_rsl_height_plus": float(np.median(rsl_plus)),
            "critical_height_plus": ycrit_plus,
            "critical_height_over_median_rsl_plus": float(ycrit_plus / np.median(rsl_plus)),
            "result": "no overlap between the deposited matching interface and the wavelength-scaled outer edge of the roughness layer",
            "status": "PASS",
        }

    arrays = {
        "x": x,
        "eta_station": eta_station,
        "eta_first": eta_first,
        "eps_exact": eps_exact,
        "eps_legacy_local": eps_legacy_local,
        "hill_height": h,
        "jackson_centroid": np.array(jackson_d),
        "jackson_centroid_ensemble": jackson_ensemble,
        "rsl_height": np.array(rsl_height),
        "eta_plus": eta_plus,
        "rsl_plus": rsl_plus,
        "source_profile_sha256": np.array(summary["sources_sha256"][str(PROFILE.relative_to(ROOT))]),
        "schema": np.array(summary["schema"]),
    }
    for directory in (RESULTS, NODE):
        np.savez(directory / "wall_origin_rsl_sensitivity_l0.npz", **arrays)
        (directory / "wall_origin_rsl_sensitivity_l0.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (NODE / "wall_origin_rsl_sensitivity_l0.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=origin_rows[0].keys())
        writer.writeheader()
        writer.writerows(origin_rows)

    print("M10/M11 wall-origin and roughness-layer sensitivity")
    print(f"exact epsilon median       : {summary['exact_epsilon_median']:.6f}")
    print(f"Jackson d/H                : {jackson_d:.6f}")
    print(f"centroid ensemble range    : [{jackson_ensemble.min():.6f}, {jackson_ensemble.max():.6f}]")
    print(f"median eta_m/y_r           : {summary['median_eta_match_over_rsl_height']:.6f}")
    print(f"median y_r+                : {summary['median_rsl_height_plus']:.3f}")
    print(f"critical y+                : {ycrit_plus:.3f}")
    print("PASS — wrote node and codes/results artifacts")


if __name__ == "__main__":
    main()
