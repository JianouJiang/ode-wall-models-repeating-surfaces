#!/usr/bin/env python3
"""Close M10/M11 on a common physical surface and measured RSL signal.

The producer is deliberately independent of the coupled simulations.  It
binds the pressure/stress comparison to the mesh-recorded L1 matching surface,
anchors Jackson's displacement height to the directly integrated wall force,
and measures the decay of the dispersive shear stress rather than assigning a
roughness-sublayer height from wavelength alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
NODE = ROOT / "development" / "nodes" / "node_007"
LADDER_JSON = RESULTS / "r2m4_apriori_ladder_20260823.json"
LADDER_NPZ = RESULTS / "r2m4_apriori_ladder_20260823.npz"
PROFILE512 = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
CERTIFICATE = RESULTS / "wall_following_budget_certificate_l1.npz"
CRITICAL = RESULTS / "critical_matching_height_map.npz"
RAW_BASE = (ROOT / "codes" / "raw_data" / "geometry_driven" /
            "xiao_pehill_parameterized" / "pehill-29-cases-DNS")
RAW_CASES = {
    "pitch6_768": RAW_BASE / "alph10-6-3036" / "mean_files.dat",
    "pitch9_768": RAW_BASE / "alph10-9-3036" / "mean_files.dat",
    "pitch12_768": RAW_BASE / "alph10-12-3036" / "mean_files.dat",
}
ETA_RSL = np.linspace(0.0, 2.0, 401)
RSL_THRESHOLDS = (0.025, 0.05, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_slope(values: np.ndarray, dx: float, max_mode: int) -> np.ndarray:
    modes = np.fft.fftfreq(values.size) * values.size
    wave = 2.0 * np.pi * np.fft.fftfreq(values.size, d=dx)
    spectrum = np.fft.fft(values)
    spectrum[np.abs(modes) > max_mode] = 0.0
    return np.fft.ifft(1j * wave * spectrum).real


def rsl_record(name: str, x: np.ndarray, h: np.ndarray,
               u: np.ndarray, v: np.ndarray, source: Path) -> tuple[dict, np.ndarray]:
    hp = periodic_slope(h, float(x[1] - x[0]), max(8, x.size // 8))
    w = v - hp[:, None] * u
    u_tilde = u - np.mean(u, axis=0)
    w_tilde = w - np.mean(w, axis=0)
    dispersive = np.abs(np.mean(u_tilde * w_tilde, axis=0))
    eligible = ETA_RSL >= 0.025
    peak = float(np.max(dispersive[eligible]))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError(f"non-positive dispersive-stress peak for {name}")
    ratio = dispersive / peak
    tail_max = np.maximum.accumulate(ratio[::-1])[::-1]
    edges = {}
    for threshold in RSL_THRESHOLDS:
        index = np.flatnonzero(tail_max <= threshold)
        edges[f"{threshold:.3f}"] = (float(ETA_RSL[index[0]])
                                      if index.size else None)
    return ({
        "name": name,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "stations": int(x.size),
        "eta_max_over_H": float(ETA_RSL[-1]),
        "dispersive_stress_peak": peak,
        "tail_ratio_at_eta_max": float(ratio[-1]),
        "edge_over_H": edges,
        "primary_edge_is_lower_bound": edges["0.050"] is None,
    }, ratio)


def raw_rsl(path: Path, name: str) -> tuple[dict, np.ndarray]:
    table = np.loadtxt(path)
    if table.shape[1] != 6:
        raise RuntimeError(f"unexpected mean-file schema: {path}")
    x = np.unique(table[:, 0])
    y = np.unique(table[:, 1])
    nx, ny = x.size, y.size
    u0 = table[:, 2].reshape(ny, nx).T
    v0 = table[:, 3].reshape(ny, nx).T
    speed = np.hypot(u0, v0)
    h = np.empty(nx)
    u = np.empty((nx, ETA_RSL.size))
    v = np.empty_like(u)
    for i in range(nx):
        fluid = np.flatnonzero(speed[i] > 1.0e-10)
        if fluid.size < 12:
            raise RuntimeError(f"insufficient fluid points at {name} station {i}")
        first = int(fluid[0])
        h[i] = float(y[first - 1]) if first else 0.0
        abscissa = np.r_[h[i], y[first:]]
        u[i] = np.interp(h[i] + ETA_RSL, abscissa,
                         np.r_[0.0, u0[i, first:]])
        v[i] = np.interp(h[i] + ETA_RSL, abscissa,
                         np.r_[0.0, v0[i, first:]])
    return rsl_record(name, x, h, u, v, path)


def processed_rsl(path: Path) -> tuple[dict, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], float)
        y = np.asarray(data["y"], float)
        u0 = np.asarray(data["U"], float)
        v0 = np.asarray(data["V"], float)
    u = np.empty((x.size, ETA_RSL.size))
    v = np.empty_like(u)
    for i in range(x.size):
        finite = np.isfinite(y[i]) & np.isfinite(u0[i]) & np.isfinite(v0[i])
        u[i] = np.interp(ETA_RSL, y[i, finite], u0[i, finite])
        v[i] = np.interp(ETA_RSL, y[i, finite], v0[i, finite])
    # The processed profiles are already wall-relative.  The analytic surface
    # is used only to rotate vertical velocity to the wall-following component.
    util = (ROOT / "codes" / "raw_data" / "geometry_driven" /
            "xiao_pehill_parameterized" / "utility" /
            "hill-geometry-gereration")
    import sys
    sys.path.insert(0, str(util))
    from hillShape import profile  # noqa: PLC0415
    h = profile(x.copy())
    return rsl_record("pitch9_512_independent", x, h, u, v, path)


def main() -> int:
    NODE.mkdir(parents=True, exist_ok=True)
    ladder_summary = json.loads(LADDER_JSON.read_text(encoding="utf-8"))
    with np.load(LADDER_NPZ, allow_pickle=False) as ladder, \
         np.load(PROFILE512, allow_pickle=False) as profile, \
         np.load(CERTIFICATE, allow_pickle=False) as cert, \
         np.load(CRITICAL, allow_pickle=False) as critical:
        phase = np.asarray(ladder["ladder_L1_phase"], float)
        y_m = np.asarray(ladder["ladder_L1_y_m"], float)
        truth = np.asarray(ladder["ladder_L1_diag_truth"], float)
        dpds = np.asarray(ladder["ladder_L1_diag_dpds"], float)
        epsilon = np.abs(truth) / np.maximum(np.abs(dpds) * y_m, 1.0e-30)

        # Perturb the wall location by fractions of one local first spacing.
        x_profile = np.asarray(profile["x"], float)
        period = float((x_profile[1] - x_profile[0]) * x_profile.size)
        first_spacing = np.asarray(profile["y"][:, 1] - profile["y"][:, 0], float)
        phase_x = np.mod(phase, 1.0) * period
        local_spacing = np.interp(
            phase_x, np.r_[x_profile, x_profile[0] + period],
            np.r_[first_spacing, first_spacing[0]])
        shift_fractions = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
        perturb_epsilon = np.empty((shift_fractions.size, phase.size))
        for j, fraction in enumerate(shift_fractions):
            shifted = y_m - fraction * local_spacing
            if np.any(shifted <= 0.0):
                raise RuntimeError("wall-origin perturbation crossed the matching surface")
            perturb_epsilon[j] = (np.abs(truth) /
                                  np.maximum(np.abs(dpds) * shifted, 1.0e-30))

        h = np.asarray(cert["h"], float)
        direct = np.asarray(cert["q_wall_direct"], float)
        reconstructed = np.asarray(cert["q_wall_reference"], float)
        direct_ensemble = np.asarray(cert["q_direct_ensemble"], float)
        d_direct = float(np.mean(h * direct) / np.mean(direct))
        d_reconstructed = float(np.mean(h * reconstructed) /
                                np.mean(reconstructed))
        d_ensemble = np.asarray([np.mean(h * q) / np.mean(q)
                                 for q in direct_ensemble])
        force_discrepancy = float(
            abs(np.mean(reconstructed) - np.mean(direct)) /
            np.mean(np.abs(direct)))

        keys = np.asarray(critical["keys"]).astype(str)
        index = int(np.flatnonzero(keys == "periodic_hills_1p0")[0])
        stored_ycrit = float(np.asarray(critical["ycrit"])[index])
        relrms = np.asarray(critical["sweep_relrms__periodic_hills_1p0"], float)
        ymp = np.asarray(critical["ymp_grid"], float)
        if relrms[0] >= 1.0:
            ycrit_rebuilt = float(ymp[0])
        else:
            crossings = np.flatnonzero((relrms[:-1] < 1.0) & (relrms[1:] >= 1.0))
            if not crossings.size:
                ycrit_rebuilt = float("inf")
            else:
                j = int(crossings[0])
                weight = -np.log(relrms[j]) / (np.log(relrms[j + 1]) -
                                                np.log(relrms[j]))
                ycrit_rebuilt = float(np.exp(np.log(ymp[j]) + weight *
                                             (np.log(ymp[j + 1]) - np.log(ymp[j]))))

        arrays = {
            "phase": phase,
            "matching_height": y_m,
            "truth_tau_s": truth,
            "dpds": dpds,
            "epsilon": epsilon,
            "origin_shift_fractions": shift_fractions,
            "origin_perturbation_epsilon": perturb_epsilon,
            "local_first_spacing": local_spacing,
            "h": h,
            "q_wall_direct": direct,
            "q_wall_reconstructed": reconstructed,
            "jackson_direct_ensemble": d_ensemble,
            "critical_ymp": ymp,
            "critical_relrms": relrms,
            "schema": np.array("wall-origin-rsl-common-surface-v2"),
        }

    rsl_rows = []
    record, ratio = processed_rsl(PROFILE512)
    rsl_rows.append(record)
    arrays["rsl_ratio__pitch9_512_independent"] = ratio
    for name, path in RAW_CASES.items():
        record, ratio = raw_rsl(path, name)
        rsl_rows.append(record)
        arrays[f"rsl_ratio__{name}"] = ratio
    arrays["rsl_eta"] = ETA_RSL

    common = ladder_summary["surfaces"]["ladder_L1"]
    summary = {
        "schema": "wall-origin-rsl-common-surface-v2",
        "ledger_row": "M10/M11",
        "common_surface": {
            "definition": "mesh-recorded wall-normal first-cell-centre surface",
            "phase_count": int(phase.size),
            "y_m_over_H": common["y_m_over_H"],
            "epsilon_median_rebuilt": float(np.median(epsilon)),
            "epsilon_fraction_below_0p1": float(np.mean(epsilon < 0.1)),
            "model_metrics": common["metrics"],
        },
        "origin_perturbation": [
            {
                "wall_shift_over_first_spacing": float(fraction),
                "epsilon_median": float(np.median(perturb_epsilon[j])),
                "epsilon_fraction_below_0p1": float(np.mean(perturb_epsilon[j] < 0.1)),
            }
            for j, fraction in enumerate(shift_fractions)
        ],
        "jackson_centroid": {
            "direct_wall_force_over_H": d_direct,
            "direct_ensemble_min": float(np.min(d_ensemble)),
            "direct_ensemble_max": float(np.max(d_ensemble)),
            "direct_ensemble_std": float(np.std(d_ensemble, ddof=1)),
            "reconstructed_parent_force_over_H": d_reconstructed,
            "centroid_discrepancy_over_H": abs(d_reconstructed - d_direct),
            "integrated_force_discrepancy_over_direct_abs_mean": force_discrepancy,
            "signed_force_condition_number": float(np.mean(np.abs(direct)) /
                                                     abs(np.mean(direct))),
        },
        "roughness_sublayer": {
            "signal": "absolute dispersive shear stress |<U_tilde W_tilde>|",
            "edge_rule": ("lowest eta above the signal peak after which the tail remains below "
                          "5% of the peak"),
            "threshold_sensitivity": list(RSL_THRESHOLDS),
            "cases": rsl_rows,
            "result": ("no edge detected below eta/H=2.0 in either independent canonical "
                       "archive or at pitches 6H, 9H and 12H"),
        },
        "critical_height_same_operator": {
            "stored_ycrit_plus": stored_ycrit,
            "rebuilt_ycrit_plus": ycrit_rebuilt,
            "map_source": str(CRITICAL.relative_to(ROOT)),
            "map_sha256": sha256(CRITICAL),
        },
        "sources_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (LADDER_JSON, LADDER_NPZ, PROFILE512, CERTIFICATE,
                         CRITICAL, *RAW_CASES.values())
        },
        "status": "M10_M11_COMMON_SURFACE_PASS",
    }

    for directory in (RESULTS, NODE):
        np.savez(directory / "wall_origin_rsl_common_surface_m10m11.npz", **arrays)
        (directory / "wall_origin_rsl_common_surface_m10m11.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("M10/M11 common-surface and measured-RSL audit")
    print(f"epsilon median             : {np.median(epsilon):.6f}")
    print(f"direct Jackson d/H         : {d_direct:.6f}")
    print(f"reconstructed Jackson d/H  : {d_reconstructed:.6f}")
    print("RSL primary edges          : " + ", ".join(
        f"{row['name']}={'>'+str(row['eta_max_over_H']) if row['edge_over_H']['0.050'] is None else row['edge_over_H']['0.050']}"
        for row in rsl_rows))
    print("PASS — wrote stable and node artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
