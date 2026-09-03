#!/usr/bin/env python3
"""Verify the corrected crest-bulk Re_H=5600 three-grid/two-model matrix.

This replaces the original M3 verifier after the domain-volume-drive defect.
It binds all six finalizer manifests, independently rebuilds the native L2/L3
traction metrics, and tests the failure verdict without pinning its magnitude.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "codes/results/rswm_xiao_highre_campaign_m13_final/re5600"
L2_NPZ = ROOT / "codes/results/rswm_common_surface_grid_l2.npz"
L2_JSON = ROOT / "codes/results/rswm_common_surface_grid_l2_summary.json"
L3_NPZ = ROOT / "codes/results/rswm_grid_results_l3.npz"
L3_JSON = ROOT / "codes/results/rswm_grid_results_l3_summary.json"
FIGURE = ROOT / "manuscript/figures/fig_common_surface_grid_l3.pdf"
DNS = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"

CASES = {
    "rswm_m13_re5600_g0_tble_92160_v2": ("14889013", "G0", "total_gradient_tble", 92160),
    "rswm_m13_re5600_g0_equilibrium_92160_v2": ("14889015", "G0", "equilibrium", 92160),
    "rswm_m13_re5600_g1_tble_307200_v2": ("14889021", "G1c", "total_gradient_tble", 307200),
    "rswm_m13_re5600_g1_equilibrium_307200_v2": ("14889022", "G1c", "equilibrium", 307200),
    "rswm_m13_re5600_g2_tble_819200_v2": ("14889025", "G2c", "total_gradient_tble", 819200),
    "rswm_m13_re5600_g2_equilibrium_819200_v2": ("14889026", "G2c", "equilibrium", 819200),
}
JOBS = {record[0] for record in CASES.values()}
GRIDS = ("G0", "G1c", "G2c")
MODELS = ("equilibrium", "total_gradient_tble")
DENSE_N = 4096
BLOCK = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_interp(x, y, target):
    order = np.argsort(np.asarray(x, float))
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    return np.interp(target, np.r_[x - 1, x, x + 1], np.r_[y, y, y])


def exact_one_sided(values: np.ndarray) -> float:
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    return float(np.mean(null >= np.mean(values)))


def drive_registration(directory: Path, manifest: dict) -> tuple[float, float, float]:
    """Read the corrected drive from the immutable terminal evidence.

    Finalizer 14889058 omitted convenience fields from ``MANIFEST.json``.
    The input dictionary and two independent volume prints are themselves in
    the manifest's byte/hash registry, so they remain the authoritative record.
    """
    fv = (directory / "input/fvConstraints").read_text(errors="replace")
    match = re.findall(
        r"\bUbar\s*\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)\s*;", fv)
    if len(match) != 1:
        return math.nan, math.nan, math.nan
    ubar = float(match[0][0])
    solver = (directory / "log.pimpleFoam").read_text(errors="replace")
    selected = re.findall(
        r"selected\s+([0-9]+)\s+cell\(s\)\s+with volume\s+([0-9.eE+-]+)", solver)
    checkmesh = (directory / "log.checkMesh").read_text(errors="replace")
    checked = re.findall(
        r"Total volume\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)", checkmesh)
    if len(selected) != 1 or len(checked) != 1:
        return ubar, math.nan, math.nan
    selected_cells, volume = int(selected[0][0]), float(selected[0][1])
    if selected_cells != int(manifest.get("grid_cells", -1)):
        return ubar, volume, math.nan
    if not math.isclose(volume, float(checked[0]), rel_tol=1e-8):
        return ubar, volume, math.nan
    crest = ubar * volume / (2.036 * 4.5 * 9.0)
    return ubar, volume, crest


def main() -> int:
    checks: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        checks.append((label, bool(condition)))

    root_manifest_path = CAMPAIGN / "CAMPAIGN_MANIFEST.json"
    check("corrected six-case root manifest exists", root_manifest_path.is_file())
    if not root_manifest_path.is_file():
        return report(checks)
    root_manifest = json.loads(root_manifest_path.read_text())
    check("terminal six-case campaign", root_manifest.get("status") == "TERMINAL_SIX_CASE_CAMPAIGN_OK")
    check("corrected producer registry", set(root_manifest.get("producer_job_ids", [])) == JOBS)
    check("six root case records", set(root_manifest.get("cases", {})) == set(CASES))

    for case_id, (job, grid, model, cells) in CASES.items():
        directory = CAMPAIGN / case_id
        manifest_path = directory / "MANIFEST.json"
        check(f"{case_id}: manifest exists", manifest_path.is_file())
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        record = root_manifest["cases"].get(case_id, {})
        check(f"{case_id}: root binds manifest", record.get("manifest_sha256") == sha256(manifest_path)
              and record.get("manifest_bytes") == manifest_path.stat().st_size)
        check(f"{case_id}: identities", manifest.get("producer_job_id") == job
              and manifest.get("grid") == grid and manifest.get("model") == model
              and manifest.get("grid_cells") == cells)
        check(f"{case_id}: corrected Reynolds inputs", manifest.get("Re_H") == 5600
              and math.isclose(float(manifest.get("nu", math.nan)), 1 / 5600, rel_tol=2e-6))
        ubar, volume, crest = drive_registration(directory, manifest)
        expected_ubar = 2.036 * 4.5 * 9.0 / volume
        check(f"{case_id}: crest-bulk drive reconstructed from registered files",
              math.isclose(crest, 1.0, rel_tol=1e-5)
              and math.isclose(ubar, expected_ubar, rel_tol=1e-5))
        check(f"{case_id}: terminal averaging", manifest.get("latest_time") == 405.0
              and manifest.get("average_start") == 135.0 and manifest.get("average_end") == 405.0
              and manifest.get("terminal_state") ==
              "producer_exit0_solver_end_checkpoint_sampling_and_hash_gates_passed")
        check(f"{case_id}: Courant gate", float(manifest.get("maximum_courant", math.inf)) <= 0.56)
        files = manifest.get("files", {})
        check(f"{case_id}: nonempty byte registry", bool(files))
        for relative, registered in files.items():
            path = directory / relative
            check(f"{case_id}: {relative}", path.is_file()
                  and path.stat().st_size == registered.get("bytes")
                  and sha256(path) == registered.get("sha256"))
        samples = manifest.get("wall_sample_sanity", {})
        check(f"{case_id}: three checkpoints with 20 profiles", set(samples) == {"315", "360", "405"}
              and all(len(entry.get("profile_rows", {})) == 20 for entry in samples.values()))

    check("canonical native products exist", all(path.is_file() and path.stat().st_size > 0
          for path in (L2_NPZ, L2_JSON, L3_NPZ, L3_JSON, FIGURE)))
    if not all(path.is_file() for path in (L2_NPZ, L2_JSON, L3_NPZ, L3_JSON)):
        return report(checks)
    l2s = json.loads(L2_JSON.read_text())
    l3s = json.loads(L3_JSON.read_text())
    l2 = np.load(L2_NPZ)
    l3 = np.load(L3_NPZ)
    check("native L2/L3 statuses", l2s.get("status") == "RSWM_COMMON_SURFACE_GRID_L2_OK"
          and l3s.get("status") == "RSWM_GRID_RESULTS_L3_OK")
    check("native products bind corrected jobs", set(l2s.get("producer_jobs", {}).values()) == JOBS
          and set(l3s.get("producer_jobs", {}).values()) == JOBS)
    check("registered grid sizes", np.array_equal(l2["cells"], np.asarray([92160, 307200, 819200])))

    dense = np.arange(DENSE_N, dtype=float) / DENSE_N
    truth = periodic_interp(l2["truth_phase"], l2["truth_tau_s"], dense)
    for grid in GRIDS:
        for model in MODELS:
            prefix = f"{grid}_{model}"
            pred = periodic_interp(l2[f"{prefix}_phase"], l2[f"{prefix}_tau_s"], dense)
            error = pred - truth
            rel = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(truth**2)))
            r2 = float(1 - np.sum(error**2) / np.sum((truth - truth.mean())**2))
            recorded = l3s["base_metrics"][f"{grid}:{model}"]
            check(f"{prefix}: independent E_tau/R2 rebuild", abs(rel - recorded["relative_rms"]) < 2e-12
                  and abs(r2 - recorded["r2"]) < 2e-11)
            check(f"{prefix}: coupled point estimate exceeds DNS RMS", rel > 1.0)
            draws = l3[f"{prefix}_primary_bootstrap_relative_rms"]
            interval = l3s["phase_bootstrap_primary_intervals"][f"{grid}:{model}"]
            quantile = np.quantile(draws, (0.025, 0.5, 0.975))
            check(f"{prefix}: phase-block interval rebuild", np.allclose(
                quantile, [interval["low"], interval["median"], interval["high"]], atol=2e-14))

    raw_failure_p = {}
    for model in MODELS:
        prefix = f"G2c_{model}"
        pred = periodic_interp(l2[f"{prefix}_phase"], l2[f"{prefix}_tau_s"], dense)
        contrast = (pred - truth) ** 2 - truth**2
        blocks = np.asarray([np.mean(contrast[i * BLOCK:(i + 1) * BLOCK])
                             for i in range(DENSE_N // BLOCK)])
        p_raw = exact_one_sided(blocks)
        raw_failure_p[model] = p_raw
        report_p = l3s["failure_significance_tests"][model]
        check(f"{model}: exact failure test rebuilt",
              abs(p_raw - report_p["p_one_sided"]) < 1e-15)

    ordered = sorted(raw_failure_p, key=raw_failure_p.get)
    holm = {}
    previous = 0.0
    for index, model in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - index) * raw_failure_p[model])
        previous = max(previous, adjusted)
        holm[model] = previous
    for model in MODELS:
        reported = l3s["failure_significance_tests"][model]["p_one_sided_holm_two_models"]
        check(f"{model}: Holm adjustment rebuilt", abs(reported - holm[model]) < 1e-15)

    verdict = l3s.get("registered_verdicts", {})
    check("corrected inferential verdict is not overstated",
          bool(verdict.get("both_finest_point_estimates_above_unit"))
          and not bool(verdict.get("both_finest_phase_intervals_above_unit"))
          and not bool(verdict.get("both_one_sided_holm_tests_resolved_at_0p05"))
          and "not an interval-separated" in l3s.get("conclusion", ""))

    check("publication figure is a nonempty PDF", FIGURE.is_file()
          and FIGURE.read_bytes().startswith(b"%PDF") and FIGURE.stat().st_size > 10000)
    return report(checks)


def report(checks: list[tuple[str, bool]]) -> int:
    failures = [label for label, passed in checks if not passed]
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    print(f"CORRECTED_GRID_M3: {len(checks) - len(failures)}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
