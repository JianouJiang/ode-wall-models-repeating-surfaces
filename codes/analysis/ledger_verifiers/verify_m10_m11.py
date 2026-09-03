#!/usr/bin/env python3
"""Independent verifier for wall-origin and roughness-sublayer row M10/M11."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
SUMMARY = RESULTS / "wall_origin_rsl_common_surface_m10m11.json"
DATA = RESULTS / "wall_origin_rsl_common_surface_m10m11.npz"
LADDER = RESULTS / "r2m4_apriori_ladder_20260823.npz"
PROFILE = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
CERT = RESULTS / "wall_following_budget_certificate_l1.npz"
CRITICAL = RESULTS / "critical_matching_height_map.npz"
TEX = ROOT / "manuscript" / "main.tex"
PDF = ROOT / "manuscript" / "main.pdf"
RAW_BASE = (ROOT / "codes" / "raw_data" / "geometry_driven" /
            "xiao_pehill_parameterized" / "pehill-29-cases-DNS")
RAW = {
    "pitch6_768": RAW_BASE / "alph10-6-3036" / "mean_files.dat",
    "pitch9_768": RAW_BASE / "alph10-9-3036" / "mean_files.dat",
    "pitch12_768": RAW_BASE / "alph10-12-3036" / "mean_files.dat",
}
ETA = np.linspace(0.0, 2.0, 401)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slope(values: np.ndarray, dx: float) -> np.ndarray:
    modes = np.fft.fftfreq(values.size) * values.size
    wave = 2.0 * np.pi * np.fft.fftfreq(values.size, d=dx)
    spectrum = np.fft.fft(values)
    spectrum[np.abs(modes) > max(8, values.size // 8)] = 0.0
    return np.fft.ifft(1j * wave * spectrum).real


def ratio_from_fields(x: np.ndarray, h: np.ndarray,
                      u: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = v - slope(h, float(x[1] - x[0]))[:, None] * u
    ut = u - np.mean(u, axis=0)
    wt = w - np.mean(w, axis=0)
    stress = np.abs(np.mean(ut * wt, axis=0))
    return stress / np.max(stress[ETA >= 0.025])


def raw_ratio(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    x = np.unique(table[:, 0])
    y = np.unique(table[:, 1])
    nx, ny = x.size, y.size
    u0 = table[:, 2].reshape(ny, nx).T
    v0 = table[:, 3].reshape(ny, nx).T
    speed = np.hypot(u0, v0)
    h = np.empty(nx)
    u = np.empty((nx, ETA.size))
    v = np.empty_like(u)
    for i in range(nx):
        first = int(np.flatnonzero(speed[i] > 1.0e-10)[0])
        h[i] = float(y[first - 1]) if first else 0.0
        yy = np.r_[h[i], y[first:]]
        u[i] = np.interp(h[i] + ETA, yy, np.r_[0.0, u0[i, first:]])
        v[i] = np.interp(h[i] + ETA, yy, np.r_[0.0, v0[i, first:]])
    return ratio_from_fields(x, h, u, v)


def processed_ratio(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as profile:
        x = np.asarray(profile["x"], float)
        y = np.asarray(profile["y"], float)
        u0 = np.asarray(profile["U"], float)
        v0 = np.asarray(profile["V"], float)
    u = np.empty((x.size, ETA.size))
    v = np.empty_like(u)
    for i in range(x.size):
        valid = np.isfinite(y[i]) & np.isfinite(u0[i]) & np.isfinite(v0[i])
        u[i] = np.interp(ETA, y[i, valid], u0[i, valid])
        v[i] = np.interp(ETA, y[i, valid], v0[i, valid])
    util = (ROOT / "codes" / "raw_data" / "geometry_driven" /
            "xiao_pehill_parameterized" / "utility" /
            "hill-geometry-gereration")
    sys.path.insert(0, str(util))
    from hillShape import profile as hill_profile  # noqa: PLC0415
    return ratio_from_fields(x, hill_profile(x.copy()), u, v)


def surface_matches(phase: np.ndarray, height: np.ndarray,
                    reference_phase: np.ndarray,
                    reference_height: np.ndarray) -> bool:
    return (phase.shape == reference_phase.shape == height.shape == reference_height.shape
            and np.allclose(phase, reference_phase, rtol=0.0, atol=1.0e-14)
            and np.allclose(height, reference_height, rtol=0.0, atol=1.0e-14))


def active_source(path: Path) -> str:
    sys.path.insert(0, str(ROOT / "codes" / "analysis" / "ledger_verifiers"))
    from _active_build import active_source as extract  # noqa: PLC0415
    return extract(path)


def main() -> int:
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    required = [SUMMARY, DATA, LADDER, PROFILE, CERT, CRITICAL, TEX, PDF, *RAW.values()]
    check("all stable, raw and manuscript artifacts exist", all(path.is_file() for path in required))
    if not all(path.is_file() for path in required):
        return 2
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    check("schema and terminal status are pinned",
          summary.get("schema") == "wall-origin-rsl-common-surface-v2"
          and summary.get("status") == "M10_M11_COMMON_SURFACE_PASS")
    check("all declared source hashes remain current",
          all((ROOT / rel).is_file() and sha256(ROOT / rel) == digest
              for rel, digest in summary["sources_sha256"].items()))

    with np.load(DATA, allow_pickle=False) as result, \
         np.load(LADDER, allow_pickle=False) as ladder, \
         np.load(PROFILE, allow_pickle=False) as profile, \
         np.load(CERT, allow_pickle=False) as cert, \
         np.load(CRITICAL, allow_pickle=False) as critical:
        phase = np.asarray(result["phase"], float)
        y_m = np.asarray(result["matching_height"], float)
        phase_ref = np.asarray(ladder["ladder_L1_phase"], float)
        y_ref = np.asarray(ladder["ladder_L1_y_m"], float)
        check("comparison uses the mesh-recorded physical surface",
              surface_matches(phase, y_m, phase_ref, y_ref))
        check("red fixture rejects fixed/phase-mismatched heights",
              not surface_matches(phase, np.full_like(y_m, np.median(y_m)),
                                  phase_ref, y_ref)
              and not surface_matches(np.roll(phase, 1), y_m, phase_ref, y_ref))

        truth = np.asarray(ladder["ladder_L1_diag_truth"], float)
        dpds = np.asarray(ladder["ladder_L1_diag_dpds"], float)
        epsilon = np.abs(truth) / np.maximum(np.abs(dpds) * y_ref, 1.0e-30)
        check("common-surface pressure ratio is independently rebuilt",
              np.allclose(result["epsilon"], epsilon)
              and abs(np.median(epsilon) -
                      summary["common_surface"]["epsilon_median_rebuilt"]) < 1.0e-14)

        first = np.asarray(profile["y"][:, 1] - profile["y"][:, 0], float)
        x = np.asarray(profile["x"], float)
        period = float((x[1] - x[0]) * x.size)
        spacing = np.interp(np.mod(phase, 1.0) * period,
                            np.r_[x, x[0] + period], np.r_[first, first[0]])
        shifts = np.asarray(result["origin_shift_fractions"], float)
        rebuilt = np.asarray([
            np.abs(truth) / np.maximum(np.abs(dpds) * (y_ref - f * spacing), 1.0e-30)
            for f in shifts])
        check("origin perturbation table is rebuilt station by station",
              np.allclose(result["local_first_spacing"], spacing)
              and np.allclose(result["origin_perturbation_epsilon"], rebuilt)
              and np.allclose([row["epsilon_median"]
                               for row in summary["origin_perturbation"]],
                              np.median(rebuilt, axis=1)))

        h = np.asarray(cert["h"], float)
        q_direct = np.asarray(cert["q_wall_direct"], float)
        q_recon = np.asarray(cert["q_wall_reference"], float)
        direct_d = float(np.mean(h * q_direct) / np.mean(q_direct))
        recon_d = float(np.mean(h * q_recon) / np.mean(q_recon))
        jackson = summary["jackson_centroid"]
        check("Jackson centroid is anchored to direct wall pressure plus viscous force",
              np.allclose(result["q_wall_direct"], q_direct)
              and abs(direct_d - jackson["direct_wall_force_over_H"]) < 1.0e-14)
        check("independent reconstructed-force discrepancy is reported",
              abs(recon_d - jackson["reconstructed_parent_force_over_H"]) < 1.0e-14
              and abs(abs(recon_d - direct_d) -
                      jackson["centroid_discrepancy_over_H"]) < 1.0e-14
              and jackson["centroid_discrepancy_over_H"] > 0.01)
        direct_ensemble = np.asarray(cert["q_direct_ensemble"], float)
        d_ensemble = np.asarray([np.mean(h * q) / np.mean(q) for q in direct_ensemble])
        check("direct-force centroid ensemble is independently rebuilt",
              np.allclose(result["jackson_direct_ensemble"], d_ensemble)
              and abs(d_ensemble.min() - jackson["direct_ensemble_min"]) < 1.0e-14
              and abs(d_ensemble.max() - jackson["direct_ensemble_max"]) < 1.0e-14)

        saved_ratios = {
            "pitch9_512_independent": np.asarray(result["rsl_ratio__pitch9_512_independent"], float),
            **{name: np.asarray(result[f"rsl_ratio__{name}"], float) for name in RAW},
        }
        check("independent 512-station dispersive decay is rebuilt",
              np.allclose(saved_ratios["pitch9_512_independent"],
                          processed_ratio(PROFILE), rtol=2.0e-12, atol=2.0e-14))
        raw_ok = all(np.allclose(saved_ratios[name], raw_ratio(path),
                                 rtol=2.0e-12, atol=2.0e-14)
                     for name, path in RAW.items())
        check("three-pitch 768-station dispersive decays are rebuilt", raw_ok)
        tail_ok = all(np.min(np.maximum.accumulate(ratio[::-1])[::-1]) > threshold
                      for ratio in saved_ratios.values()
                      for threshold in (0.025, 0.05, 0.10))
        check("RSL lower bound survives archive, pitch and threshold sensitivity",
              tail_ok and all(row["primary_edge_is_lower_bound"]
                              for row in summary["roughness_sublayer"]["cases"]))

        keys = np.asarray(critical["keys"]).astype(str)
        i = int(np.flatnonzero(keys == "periodic_hills_1p0")[0])
        rel = np.asarray(critical["sweep_relrms__periodic_hills_1p0"], float)
        ymp = np.asarray(critical["ymp_grid"], float)
        rebuilt_ycrit = float(ymp[0]) if rel[0] >= 1.0 else np.nan
        critical_summary = summary["critical_height_same_operator"]
        check("critical-height map is opened and same-operator floor rebuilt",
              sha256(CRITICAL) == critical_summary["map_sha256"]
              and abs(float(critical["ycrit"][i]) - rebuilt_ycrit) < 1.0e-14
              and abs(critical_summary["rebuilt_ycrit_plus"] - rebuilt_ycrit) < 1.0e-14)

    source = active_source(TEX)
    compiled = subprocess.run(["pdftotext", str(PDF), "-"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
    compact_source = re.sub(r"\s+", " ", source)
    compact_pdf = re.sub(r"\s+", " ", compiled)
    required_text = ["0.1437", "0.37531", "0.39185", "0.01654", "pitch-$6H$", "pitch-$12H$"]
    check("active Methods prints the direct-origin/RSL audit",
          all(token in compact_source for token in required_text))
    check("compiled PDF prints the direct-origin/RSL audit",
          all(token.replace("$", "") in compact_pdf.replace(" ", "")
              if token.startswith("pitch") else token in compact_pdf
              for token in required_text)
          and PDF.stat().st_mtime_ns >= TEX.stat().st_mtime_ns)

    failed = [name for name, passed in checks if not passed]
    print(f"M10/M11: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
