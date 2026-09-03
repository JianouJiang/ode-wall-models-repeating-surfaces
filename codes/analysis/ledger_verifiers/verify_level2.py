#!/usr/bin/env python3
"""Independent terminal verifier for the Level-2 ARCHER2 experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
NODE = Path(__file__).resolve().parents[3] / "codes" / "results" / "rswm_level2_node003_evidence"
CAMPAIGN = (
    ROOT / "codes" / "results" /
    "rswm_xiao_dns_grid_campaign_final_l2"
)
VALIDATION = NODE / "validation_job_14868798"
DNS_FILE = ROOT / "codes" / "results" / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
DNS_SHA256 = "d039cefb93ec1a8555555deed79041921bf8ce98cd1477479087a9804ca7ff85"
GEOMETRY_GENERATOR = ROOT / "codes" / "openfoam" / "make_xiao_dns_wmles_case.py"
GEOMETRY_VERIFIER = ROOT / "codes" / "openfoam" / "verify_xiao_dns_mesh.py"
GEOMETRY_SOURCE = (
    ROOT / "codes" / "raw_data" / "geometry_driven" /
    "xiao_pehill_parameterized" / "utility" /
    "hill-geometry-gereration" / "hillShape.py"
)
SUMMARY = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2_summary.json"
NODE_SUMMARY = NODE / "results_summary.json"
RESULT = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2.npz"
ANALYSIS_MANIFEST = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2_manifest.json"
FIGURE = NODE / "fig_common_surface_grid_l2.pdf"
FIGURE_PNG = NODE / "fig_common_surface_grid_l2.png"
SCHEDULER = NODE / "SCHEDULER_TERMINAL_L2.txt"
JOB_REGISTRY = NODE / "JOB_REGISTRY.json"

CASES = {
    "rswm_xiao_dns_g0_tble_92160_l2_v1":
        ("14868882", "G0", "total_gradient_tble", 92160, 80, 24),
    "rswm_xiao_dns_g0_equilibrium_92160_l2_v1":
        ("14868883", "G0", "equilibrium", 92160, 80, 24),
    "rswm_xiao_dns_g1_tble_307200_l2_v1":
        ("14868884", "G1c", "total_gradient_tble", 307200, 120, 40),
    "rswm_xiao_dns_g1_equilibrium_307200_l2_v1":
        ("14868885", "G1c", "equilibrium", 307200, 120, 40),
    "rswm_xiao_dns_g2_tble_819200_l2_v1":
        ("14868887", "G2c", "total_gradient_tble", 819200, 160, 64),
    "rswm_xiao_dns_g2_equilibrium_819200_l2_v1":
        ("14868888", "G2c", "equilibrium", 819200, 160, 64),
}
BATCH_BY_JOB = {
    "14868882": "rswm_xiao_dns_g0_tble_l2.slurm",
    "14868883": "rswm_xiao_dns_g0_equilibrium_l2.slurm",
    "14868884": "rswm_xiao_dns_g1_tble_l2.slurm",
    "14868885": "rswm_xiao_dns_g1_equilibrium_l2.slurm",
    "14868887": "rswm_xiao_dns_g2_tble_l2.slurm",
    "14868888": "rswm_xiao_dns_g2_equilibrium_l2.slurm",
}

checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, bool(condition)))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(directory: Path, manifest: dict, label: str) -> None:
    records = manifest.get("files", {})
    check(f"{label}: nonempty file registry", isinstance(records, dict) and bool(records))
    if not isinstance(records, dict):
        return
    for relative, record in records.items():
        path = directory / relative
        check(f"{label}: exists {relative}", path.is_file())
        if path.is_file():
            check(f"{label}: bytes {relative}", path.stat().st_size == record.get("bytes"))
            check(f"{label}: sha256 {relative}", file_hash(path) == record.get("sha256"))


# Freeze the corrected 512-phase DNS identity before any coupled comparison.
check("DNS: authoritative corrected file", (
    DNS_FILE.is_file() and file_hash(DNS_FILE) == DNS_SHA256
))
if DNS_FILE.is_file():
    dns_identity = np.load(DNS_FILE, allow_pickle=False)
    dns_x = np.asarray(dns_identity["x"], float)
    check("DNS: 512-phase documented period", (
        dns_x.shape == (512,) and np.all(np.isfinite(dns_x))
        and np.all(np.diff(dns_x) > 0.0) and abs(float(dns_x[0])) < 1.0e-12
        and abs(float(np.median(np.diff(dns_x))*dns_x.size) - 9.0) < 1.0e-3
    ))
check("geometry: frozen generator/verifier/source", (
    GEOMETRY_GENERATOR.is_file()
    and file_hash(GEOMETRY_GENERATOR) ==
    "fbfb1b330557832ec2c96a923f1ca294abaab5fde0b63a4f3ca8cc98d69fab23"
    and GEOMETRY_VERIFIER.is_file()
    and file_hash(GEOMETRY_VERIFIER) ==
    "878feac4f880447619fb53c9b3f79c5e77a43a0d73e05d68e53e433a54765445"
    and GEOMETRY_SOURCE.is_file()
    and file_hash(GEOMETRY_SOURCE) ==
    "4354400e9aee021e003f8b535bab6ebda7aabe8d513b1877d6dd5795d5ef288c"
))
if GEOMETRY_GENERATOR.is_file() and GEOMETRY_SOURCE.is_file():
    generator_spec = importlib.util.spec_from_file_location(
        "independent_mesh_generator", GEOMETRY_GENERATOR
    )
    source_spec = importlib.util.spec_from_file_location(
        "independent_deposit_hill", GEOMETRY_SOURCE
    )
    generator_module = importlib.util.module_from_spec(generator_spec)
    source_module = importlib.util.module_from_spec(source_spec)
    assert generator_spec.loader is not None and source_spec.loader is not None
    generator_spec.loader.exec_module(generator_module)
    source_spec.loader.exec_module(source_module)
    geometry_x = np.linspace(0.0, 54.0/28.0, 10_001)
    generated_y = np.asarray([
        generator_module.xiao_profile(value) for value in geometry_x
    ])
    source_y = np.asarray(source_module.profile(geometry_x.copy()))
    check("geometry: polynomial formula bit-close at 10001 points", (
        np.max(np.abs(generated_y - source_y)) == 0.0
    ))


# The performance gate is a result in its own right, not just prose.
validation_manifest_path = VALIDATION / "MANIFEST.json"
check("validation: manifest", validation_manifest_path.is_file())
if validation_manifest_path.is_file():
    validation_manifest = json.loads(validation_manifest_path.read_text())
    check("validation: job", validation_manifest.get("producer_job_id") == "14868798")
    check("validation: target", validation_manifest.get("latest_time") == 5.2)
    check("validation: documented period", validation_manifest.get("Lx") == 9.0)
    check("validation: measured wall time", validation_manifest.get("solver_wall_seconds") == 135)
    verify_manifest(VALIDATION, validation_manifest, "validation")
    validation_log = (VALIDATION / "log.pimpleFoam").read_text(errors="replace")
    check("validation: one End", len(re.findall(r"^End$", validation_log, re.M)) == 1)
    check("validation: census audit 500", re.search(
        r"TBLE_CENSUS_AUDIT patch=bottomWall .*timeIndex=500", validation_log
    ) is not None)
    check("validation: exact face telemetry", validation_log.count("TOTAL_GRADIENT_TBLE_FACE") == 3840)
    slurm = (VALIDATION / "slurm-14868798.out").read_text(errors="replace")
    check("validation: wrapper sentinels", all(marker in slurm for marker in (
        "TOTAL_GRADIENT_SANITY_OK", "max_projection_error=0.000e+00",
        "vector_capped=2106", "TOTAL_GRADIENT_RUN_OK", "JOB_DONE_OK"
    )))
    validation_driver = VALIDATION / "execution_driver_xiao_dns.sh"
    base_validation_driver = ROOT / "jobs" / "rswm_total_gradient_common_driver.sh"
    expected_validation_driver = base_validation_driver.read_text()
    expected_validation_driver = expected_validation_driver.replace("8.96484375", "9.0")
    expected_validation_driver = expected_validation_driver.replace(
        "import make_xiao_wmles_case as generator",
        "import make_xiao_dns_wmles_case as generator",
    ).replace("verify_common_matching_surface.py", "verify_xiao_dns_mesh.py")
    check("validation: exact executed Xiao-DNS driver", (
        validation_driver.is_file()
        and validation_driver.read_text() == expected_validation_driver
        and file_hash(validation_driver) ==
        "6970dc6e68f5904084d90dcdef8bfaf6f39083ab042238a81332a5420c4933d3"
    ))
    validation_mesh = VALIDATION / "blockMeshDict.executed"
    check("validation: exact Xiao-DNS mesh dictionary", (
        validation_mesh.is_file()
        and file_hash(validation_mesh) ==
        "f240baf4f63357478197e37136b3d672b4c242a01f4cd1c14383bbb089eb9870"
        and "half_width=1.928571428571" in validation_mesh.read_text()
        and "(7.071428571429 0.000000000000" in validation_mesh.read_text()
        and "(9.000000000000 1.000000000000" in validation_mesh.read_text()
        and "3.036000000000" in validation_mesh.read_text()
    ))
    validation_matching_path = VALIDATION / "matching_surface.json"
    validation_matching = (
        json.loads(validation_matching_path.read_text())
        if validation_matching_path.is_file() else {}
    )
    check("validation: measured deposit geometry", (
        validation_matching.get("documented_extent") == {"Lx": 9.0, "Ly": 3.036, "Lz": 4.5}
        and validation_matching.get("wall_phase_count") == 80
        and validation_matching.get("bottom_wall_faces") == 1920
        and validation_matching.get("analytic_geometry_rms_over_H", math.inf) < 2.0e-3
        and validation_matching.get("analytic_geometry_max_over_H", math.inf) < 5.0e-3
    ))

check("campaign: directory", CAMPAIGN.is_dir())
campaign_manifest_path = CAMPAIGN / "CAMPAIGN_MANIFEST.json"
check("campaign: root manifest", campaign_manifest_path.is_file())
if campaign_manifest_path.is_file():
    campaign_manifest = json.loads(campaign_manifest_path.read_text())
    check("campaign: terminal root status", (
        campaign_manifest.get("status") == "TERMINAL_SIX_CASE_CAMPAIGN_OK"
        and len(campaign_manifest.get("cases", {})) == 6
    ))
    check("campaign: exact producer IDs", set(
        campaign_manifest.get("producer_job_ids", [])
    ) == {"14868882", "14868883", "14868884", "14868885", "14868887", "14868888"})
for case_id, (job, grid, model, cells, nx, nz) in CASES.items():
    directory = CAMPAIGN / case_id
    manifest_path = directory / "MANIFEST.json"
    check(f"{case_id}: manifest", manifest_path.is_file())
    if not manifest_path.is_file():
        continue
    manifest = json.loads(manifest_path.read_text())
    if campaign_manifest_path.is_file():
        root_record = campaign_manifest.get("cases", {}).get(case_id, {})
        check(f"{case_id}: root manifest binding", (
            root_record.get("producer_job_id") == job
            and root_record.get("manifest_bytes") == manifest_path.stat().st_size
            and root_record.get("manifest_sha256") == file_hash(manifest_path)
            and manifest.get("finalizer_job_id") == campaign_manifest.get("finalizer_job_id")
        ))
    check(f"{case_id}: identities", all((
        manifest.get("producer_job_id") == job,
        manifest.get("grid") == grid,
        manifest.get("model") == model,
        manifest.get("grid_cells") == cells,
        manifest.get("Lx") == 9.0,
        manifest.get("Ly") == 3.036,
        manifest.get("Lz") == 4.5,
        math.isclose(manifest.get("hill_half_width", math.nan), 54.0/28.0,
                     rel_tol=0.0, abs_tol=1.0e-15),
        manifest.get("geometry") == "xiao_alpha1_deposit_polynomial",
        manifest.get("latest_time") == 405.0,
        manifest.get("average_start") == 135.0,
        manifest.get("average_end") == 405.0,
    )))
    check(f"{case_id}: terminal state", manifest.get("terminal_state") ==
          "producer_exit0_solver_end_checkpoint_sampling_and_hash_gates_passed")
    solver_cost = manifest.get("solver_cost", {})
    check(f"{case_id}: solver cost", all(
        isinstance(solver_cost.get(name), (int, float))
        and math.isfinite(float(solver_cost[name]))
        and float(solver_cost[name]) > 0.0
        for name in (
            "execution_seconds", "clock_seconds", "time_steps",
            "clock_seconds_per_step", "clock_seconds_per_bottom_face_step",
        )
    ))
    verify_manifest(directory, manifest, case_id)
    registered_files = set(manifest.get("files", {}))
    actual_files = {
        str(path.relative_to(directory)) for path in directory.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    check(f"{case_id}: exact manifest file set", actual_files == registered_files)
    source_force = directory / "input" / "uniform" / "momentumForceProperties"
    check(f"{case_id}: common source-force bytes", (
        source_force.is_file()
        and file_hash(source_force) ==
        "9da8e6131425dafc3ca2b9937c4659ff912c90a2db005b1be8ee2a63c2cf7f8f"
    ))
    check(f"{case_id}: reproducible initial/physics fields", all(
        (directory / "input" / name).is_file()
        for name in ("U", "p", "nut", "C", "physicalProperties", "momentumTransport")
    ))
    execution_files = list((directory / "input" / "execution").glob("*"))
    check(f"{case_id}: execution sources", (
        len(execution_files) == 12
        and all(path.is_file() and path.stat().st_size > 0 for path in execution_files)
    ))
    named_execution = [path for path in execution_files if path.name not in (
        "execution_driver_xiao_dns.sh", "submitted_batch_script.slurm"
    )]
    check(f"{case_id}: execution sources match workspace", (
        len(named_execution) == 10
        and all((ROOT / "jobs" / path.name).is_file()
                and file_hash(path) == file_hash(ROOT / "jobs" / path.name)
                for path in named_execution)
    ))
    deployed_xiao_dns = directory / "input" / "execution" / "execution_driver_xiao_dns.sh"
    base_driver = ROOT / "jobs" / (
        "rswm_total_gradient_common_driver.sh" if model == "total_gradient_tble"
        else "rswm_grid_common_surface_driver.sh"
    )
    expected_xiao_dns = base_driver.read_text().replace("8.96484375", "9.0")
    expected_xiao_dns = expected_xiao_dns.replace(
        "import make_xiao_wmles_case as generator",
        "import make_xiao_dns_wmles_case as generator",
    ).replace("verify_common_matching_surface.py", "verify_xiao_dns_mesh.py")
    check(f"{case_id}: executed Xiao-DNS driver reconstructed", (
        deployed_xiao_dns.is_file() and deployed_xiao_dns.read_text() == expected_xiao_dns
        and expected_xiao_dns.count("8.96484375") == 0
    ))
    submitted_batch = directory / "input" / "execution" / "submitted_batch_script.slurm"
    expected_batch = ROOT / "jobs" / BATCH_BY_JOB[job]
    check(f"{case_id}: scheduler-spooled batch bytes", (
        submitted_batch.is_file() and expected_batch.is_file()
        and file_hash(submitted_batch) == file_hash(expected_batch)
    ))
    scheduler_output = directory / "producer_scheduler_output.txt"
    scheduler_text = scheduler_output.read_text(errors="replace") if scheduler_output.is_file() else ""
    check(f"{case_id}: producer scheduler script evidence", (
        scheduler_output.is_file()
        and f"XIAO_DNS_DRIVER_OK model={model} sha256={file_hash(deployed_xiao_dns)}" in scheduler_text
        and ": OK" in scheduler_text
        and "JOB_DONE_OK" in scheduler_text
    ))
    generator_files = list((directory / "input" / "generator_sources").glob("*.py"))
    check(f"{case_id}: generator sources", len(generator_files) == 3 and all(
        (ROOT / "codes" / "openfoam" / path.name).is_file()
        and file_hash(path) == file_hash(ROOT / "codes" / "openfoam" / path.name)
        for path in generator_files
    ))
    matching = manifest.get("matching_surface_setup_check", {})
    check(f"{case_id}: deposit-faithful mesh certificate", (
        matching.get("status") == "MESH_MATCHING_SURFACE_OK"
        and matching.get("documented_extent") == {"Lx": 9.0, "Ly": 3.036, "Lz": 4.5}
        and matching.get("bottom_wall_faces") == nx*nz
        and matching.get("wall_phase_count") == nx
        and matching.get("analytic_geometry_rms_over_H", math.inf) < 2.0e-3
        and matching.get("analytic_geometry_max_over_H", math.inf) < 5.0e-3
    ))

    log = (directory / "log.pimpleFoam").read_text(errors="replace")
    times = [float(value) for value in re.findall(r"^Time = ([0-9.eE+-]+)s?$", log, re.M)]
    courant = [float(value) for value in re.findall(
        r"Courant Number mean:\s*[0-9.eE+-]+\s+max:\s*([0-9.eE+-]+)", log
    )]
    check(f"{case_id}: solver terminal", (
        len(re.findall(r"^End$", log, re.M)) == 1
        and bool(times) and math.isclose(times[-1], 405.0, abs_tol=1e-6)
        and "FOAM FATAL" not in log and "Segmentation fault" not in log
    ))
    check(f"{case_id}: Courant", bool(courant) and max(courant) <= 0.56)
    sample_sanity = manifest.get("wall_sample_sanity", {})
    check(f"{case_id}: wall-sample sanity", (
        isinstance(sample_sanity, dict) and len(sample_sanity) == 3
        and all(
            record.get("faces") == nx*nz
            and 0.0 < record.get("raw_vector_peak", math.inf) < 0.1
            and 0.0 <= record.get("raw_vector_q99", math.inf) < 0.1
            and 0.0 < record.get("phase_mean_vector_peak", math.inf) < 0.1
            and isinstance(record.get("profile_rows"), dict)
            and len(record.get("profile_rows", {})) == 10
            and min(record.get("profile_rows", {"": 0}).values()) >= 32
            for record in sample_sanity.values()
        )
    ))
    if model == "total_gradient_tble":
        check(f"{case_id}: native operator fired", all(marker in log for marker in (
            "Initial registered pressure-gradient source",
            "TOTAL_GRADIENT_TBLE_FACE patch=bottomWall",
            "TBLE_CENSUS_AUDIT patch=bottomWall",
        )))
        telemetry = manifest.get("tble_realizability_summary")
        check(f"{case_id}: realizability reported", (
            isinstance(telemetry, dict)
            and telemetry.get("records", 0) > 0
            and isinstance(telemetry.get("last"), dict)
            and 403.9 <= telemetry["last"].get("time", -math.inf) <= 405.1
            and telemetry["last"].get("faces") == nx*nz
            and 0.0 <= telemetry["last"].get("fraction", -1.0) <= 1.0
            and telemetry["last"].get("vector_capped", -1) >= 0
            and 0.0 <= telemetry["last"].get("applied_traction_max", math.inf) < 0.1
            and 0.0 <= telemetry.get("minimum_clipped_fraction", -1.0)
            <= telemetry.get("maximum_clipped_fraction", 2.0) <= 1.0
            and 0.0 <= telemetry.get("minimum_lower_clipped_fraction", -1.0)
            <= telemetry.get("maximum_lower_clipped_fraction", 2.0) <= 1.0
            and 0.0 <= telemetry.get("minimum_vector_capped_fraction", -1.0)
            <= telemetry.get("maximum_vector_capped_fraction", 2.0) <= 1.0
            and math.isfinite(telemetry.get("maximum_mean_absolute_mismatch", math.nan))
            and telemetry.get("maximum_mean_absolute_mismatch", -1.0) >= 0.0
            and math.isfinite(telemetry.get("maximum_mismatch", math.nan))
            and telemetry.get("maximum_mismatch", -1.0) >= 0.0
            and 0.0 <= telemetry.get("maximum_applied_traction", math.inf) < 0.1
        ))
        source_pairs = (
            (directory / "input/registeredMeanVelocityForce/registeredMeanVelocityForce.H",
             ROOT / "codes/openfoam/registeredMeanVelocityForce/registeredMeanVelocityForce.H"),
            (directory / "input/registeredMeanVelocityForce/registeredMeanVelocityForce.C",
             ROOT / "codes/openfoam/registeredMeanVelocityForce/registeredMeanVelocityForce.C"),
            (directory / "input/registeredMeanVelocityForce/totalGradientTbleNutFvPatchScalarField.H",
             ROOT / "codes/openfoam/registeredMeanVelocityForce/totalGradientTbleNutFvPatchScalarField.H"),
            (directory / "input/registeredMeanVelocityForce/totalGradientTbleNutFvPatchScalarField.C",
             ROOT / "codes/openfoam/registeredMeanVelocityForce/totalGradientTbleNutFvPatchScalarField.C"),
            (directory / "input/wallmodel_tble/tbleShootContinuation.H",
             ROOT / "jobs/rswm_continuation_tbleShoot.H"),
        )
        check(f"{case_id}: deployed sources match workspace", all(
            deployed.is_file() and workspace.is_file()
            and file_hash(deployed) == file_hash(workspace)
            for deployed, workspace in source_pairs
        ))

    checkpoint_names = (directory / "checkpoint_times_l2.txt").read_text().split()
    check(f"{case_id}: three checkpoints", len(checkpoint_names) == 3)
    for target, checkpoint in zip((315.0, 360.0, 405.0), checkpoint_names):
        check(f"{case_id}: checkpoint registration {target}",
              abs(float(checkpoint) - target) <= 0.05)
        wall = directory / "postProcessing_sampleBottomWall" / checkpoint / "bottomWall.xy"
        check(f"{case_id}: wall sample {checkpoint}", wall.is_file())
        if wall.is_file():
            values = np.loadtxt(wall)
            check(f"{case_id}: wall shape {checkpoint}", values.shape == (nx * nz, 7))
            check(f"{case_id}: finite wall {checkpoint}", np.all(np.isfinite(values)))
        profiles = directory / "postProcessing_sampleProfiles" / checkpoint
        profile_files = list(profiles.glob("*.xy")) if profiles.is_dir() else []
        check(f"{case_id}: ten profiles {checkpoint}", len(profile_files) == 10)
        check(f"{case_id}: nonempty profiles {checkpoint}",
              bool(profile_files) and all(path.stat().st_size > 0 for path in profile_files))
        average_state = directory / "checkpoints" / checkpoint / "fieldAverageProperties"
        check(f"{case_id}: average state {checkpoint}", average_state.is_file())
        if average_state.is_file():
            total_times = [float(value) for value in re.findall(
                r"totalTime\s+([0-9.eE+-]+);", average_state.read_text()
            )]
            expected_average = target - 135.0
            check(f"{case_id}: average duration {checkpoint}", (
                len(total_times) == 3
                and all(abs(value - expected_average) <= 0.05 for value in total_times)
            ))

check("analysis: summary", SUMMARY.is_file())
check("analysis: node summary", NODE_SUMMARY.is_file())
if SUMMARY.is_file() and NODE_SUMMARY.is_file():
    check("analysis: summary copies identical", file_hash(SUMMARY) == file_hash(NODE_SUMMARY))
check("analysis: npz", RESULT.is_file())
check("analysis: hash manifest", ANALYSIS_MANIFEST.is_file())
check("analysis: figure", FIGURE.is_file() and FIGURE.stat().st_size > 10_000)
check("analysis: PNG figure", FIGURE_PNG.is_file() and FIGURE_PNG.stat().st_size > 50_000)
if SUMMARY.is_file():
    summary = json.loads(SUMMARY.read_text())
    check("analysis: status", summary.get("status") == "RSWM_COMMON_SURFACE_GRID_L2_OK")
    check("analysis: six metrics", len(summary.get("metrics", {})) == 6)
    check("analysis: three grid resolutions", set(summary.get("grid_resolution", {})) == {
        "G0", "G1c", "G2c"
    })
    check("analysis: three cost comparisons", set(summary.get("computational_cost", {})) == {
        "G0", "G1c", "G2c"
    })
    check("analysis: refinement classifications", (
        len(summary.get("grid_path_convergence", {})) == 18
        and all("observed_order" not in record and "fine_gci" not in record
                for record in summary.get("grid_path_convergence", {}).values())
    ))
    check("analysis: analytic geometry audit", set(
        summary.get("analytic_geometry_error", {})
    ) == {"G0", "G1c", "G2c"} and all(
        record.get("rms_over_H", math.inf) < 2.0e-3
        and record.get("max_over_H", math.inf) < 5.0e-3
        for record in summary.get("analytic_geometry_error", {}).values()
    ))
    geometry_readme = ROOT / summary.get("geometry_readme", "__missing__")
    check("analysis: documented period source", (
        summary.get("documented_period_over_H") == 9.0
        and geometry_readme.is_file()
        and file_hash(geometry_readme) == summary.get("geometry_readme_sha256")
        and summary.get("documented_domain_over_H") == {"Lx": 9.0, "Ly": 3.036, "Lz": 4.5}
        and math.isclose(summary.get("documented_hill_half_width_over_H", math.nan),
                         54.0/28.0, rel_tol=0.0, abs_tol=1.0e-15)
    ))
    geometry_source = ROOT / summary.get("geometry_source", "__missing__")
    mesh_generator = ROOT / summary.get("mesh_generator", "__missing__")
    check("analysis: geometry source identity", (
        geometry_source.is_file()
        and file_hash(geometry_source) == summary.get("geometry_source_sha256")
        and mesh_generator.is_file()
        and file_hash(mesh_generator) == summary.get("mesh_generator_sha256")
    ))
    check("analysis: producer set", set(summary.get("producer_jobs", {}).values()) == {
        "14868882", "14868883", "14868884", "14868885", "14868887", "14868888"
    })
if ANALYSIS_MANIFEST.is_file():
    analysis_manifest = json.loads(ANALYSIS_MANIFEST.read_text())
    analysis_records = analysis_manifest.get("files", {})
    expected_analysis_records = {
        "codes/results/rswm_xiao_dns_grid_campaign_final_l2/CAMPAIGN_MANIFEST.json",
        "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz",
        "codes/raw_data/geometry_driven/xiao_pehill_parameterized/pehill-29-cases-DNS/README_NEWDATABASE.pdf",
        "codes/raw_data/geometry_driven/xiao_pehill_parameterized/utility/hill-geometry-gereration/hillShape.py",
        "codes/openfoam/make_xiao_dns_wmles_case.py",
        "codes/analysis/rswm_common_surface_grid_l2.py",
        "codes/results/rswm_common_surface_grid_l2.npz",
        "codes/results/rswm_common_surface_grid_l2_summary.json",
        "development/nodes/node_003/results_summary.json",
        "development/nodes/node_003/fig_common_surface_grid_l2.pdf",
        "development/nodes/node_003/fig_common_surface_grid_l2.png",
    }
    check("analysis: hash manifest status", (
        analysis_manifest.get("status") == "RSWM_COMMON_SURFACE_GRID_L2_ANALYSIS_HASH_OK"
        and set(analysis_records) == expected_analysis_records
    ))
    for relative, record in analysis_records.items():
        path = ROOT / relative
        check(f"analysis: hash-bound {relative}", (
            path.is_file() and path.stat().st_size == record.get("bytes")
            and file_hash(path) == record.get("sha256")
        ))
if RESULT.is_file():
    result = np.load(RESULT, allow_pickle=False)
    check("analysis: npz status", str(result["status"]) == "RSWM_COMMON_SURFACE_GRID_L2_OK")
    core_names = [name for name in result.files if any(token in name for token in (
        "_phase", "_tau_s", "_tau_x", "_ym", "_ywall", "relative_rms",
        "sign_accuracy", "reversed_fraction", "signed_tangent_force",
        "signed_x_force", "matching_", "change_",
    )) and not any(token in name for token in (
        "separation_phase", "reattachment_phase",
    ))]
    check("analysis: finite core npz", bool(core_names) and all(
        np.all(np.isfinite(result[name])) for name in core_names
    ))
    check("analysis: registered cell ladder", np.array_equal(
        result["cells"], np.asarray([92160, 307200, 819200])
    ))

    def periodic_interp(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        x, y = np.asarray(x)[order], np.asarray(y)[order]
        return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0],
                         np.r_[y, y, y])

    dense = np.arange(4096, dtype=float) / 4096.0
    truth = periodic_interp(result["truth_phase"], result["truth_tau_s"], dense)
    legacy_truth = periodic_interp(
        result["truth_phase"], result["truth_tau_x_legacy"], dense
    )
    denom = float(np.sum((truth - np.mean(truth)) ** 2))

    # Rebuild each signed physical-tangent curve directly from the harvested
    # mesh and raw seven-column wall sample.  This is intentionally separate
    # from the reducer and catches a shared sign, face-order or integration bug
    # that recomputing R2 from already-reduced NPZ arrays cannot detect.
    sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
    sys.path.insert(0, str(
        ROOT / "codes" / "raw_data" / "geometry_driven" /
        "xiao_pehill_parameterized" / "utility" / "hill-geometry-gereration"
    ))
    from verify_common_matching_surface import (  # type: ignore
        newell_normal as independent_newell_normal,
        read_faces as independent_read_faces,
        read_labels as independent_read_labels,
        read_patch as independent_read_patch,
        read_points as independent_read_points,
        read_vector_field as independent_read_vector_field,
    )
    from hillShape import profile as independent_xiao_profile  # type: ignore

    def independent_raw_curve(directory: Path) -> tuple[np.ndarray, np.ndarray, float]:
        mesh_dir = directory / "input" / "polyMesh"
        centres = independent_read_vector_field(directory / "input" / "C")
        points = independent_read_points(mesh_dir / "points")
        check(f"analysis: exact mesh extent {directory.name}", (
            np.allclose(points.min(axis=0), [0.0, 0.0, 0.0], rtol=0.0, atol=5e-11)
            and np.allclose(points.max(axis=0), [9.0, 3.036, 4.5],
                            rtol=0.0, atol=5e-11)
            and np.min(np.abs(points[:, 0] - 54.0/28.0)) < 1.0e-8
            and np.min(np.abs(points[:, 0] - (9.0 - 54.0/28.0))) < 1.0e-8
        ))
        block_dictionary = (directory / "input" / "blockMeshDict").read_text()
        check(f"analysis: archived Xiao block dictionary {directory.name}", all(
            marker in block_dictionary for marker in (
                "half_width=1.928571428571", "3.036000000000",
                "(7.071428571429 0.000000000000", "(9.000000000000 1.000000000000",
            )
        ))
        faces = independent_read_faces(mesh_dir / "faces")
        owners = independent_read_labels(mesh_dir / "owner")
        start, count = independent_read_patch(mesh_dir / "boundary", "bottomWall")
        xyz, tangent, area, owner_distance = [], [], [], []
        for face_index in range(start, start + count):
            vertices = points[faces[face_index]]
            centre = vertices.mean(axis=0)
            normal = independent_newell_normal(vertices)
            downstream = np.array([-normal[1], normal[0], 0.0])
            if downstream[0] < 0.0:
                downstream *= -1.0
            downstream /= np.linalg.norm(downstream)
            area_vector = np.zeros(3)
            for index, vertex in enumerate(vertices):
                area_vector += np.cross(vertex, vertices[(index + 1) % len(vertices)])
            xyz.append(centre)
            tangent.append(downstream)
            area.append(0.5 * np.linalg.norm(area_vector))
            owner_distance.append(abs(float(
                np.dot(centres[owners[face_index]] - centre, normal)
            )))
        xyz, tangent, area = map(np.asarray, (xyz, tangent, area))
        analytic_wall = np.asarray(independent_xiao_profile(xyz[:, 0].copy()), float)
        geometry_error = xyz[:, 1] - analytic_wall
        check(f"analysis: independent wall geometry {directory.name}", (
            float(np.sqrt(np.mean(geometry_error**2))) < 2.0e-3
            and float(np.max(np.abs(geometry_error))) < 5.0e-3
        ))
        check(f"analysis: independent positive owner distance {directory.name}", (
            bool(owner_distance) and min(owner_distance) > 0.0
        ))
        checkpoint = (directory / "checkpoint_times_l2.txt").read_text().split()[-1]
        rows = np.loadtxt(
            directory / "postProcessing_sampleBottomWall" / checkpoint / "bottomWall.xy"
        )
        mesh_order = np.lexsort((xyz[:, 2], xyz[:, 0]))
        sample_order = np.lexsort((rows[:, 2], rows[:, 0]))
        if np.max(np.linalg.norm(xyz[mesh_order] - rows[sample_order, :3], axis=1)) > 2e-6:
            raise RuntimeError("independent wall-sample registration failed")
        aligned = np.empty_like(rows)
        aligned[mesh_order] = rows[sample_order]
        tau_face = np.einsum("ij,ij->i", -aligned[:, 3:6], tangent)
        x_rounded = np.round(xyz[:, 0], 9)
        x_unique, inverse = np.unique(x_rounded, return_inverse=True)
        tau = np.asarray([
            np.average(tau_face[inverse == index], weights=area[inverse == index])
            for index in range(len(x_unique))
        ])
        z_unique = np.unique(np.round(xyz[:, 2], 9))
        span = np.ptp(z_unique) + np.median(np.diff(z_unique))
        integral = float(np.sum(tau_face * area) / span)
        return x_unique / 9.0, tau, integral

    for grid in ("G0", "G1c", "G2c"):
        ym_by_model = []
        for model in ("equilibrium", "total_gradient_tble"):
            prefix = f"{grid}_{model}"
            phase = result[f"{prefix}_phase"]
            tau = result[f"{prefix}_tau_s"]
            reconstructed = periodic_interp(phase, tau, dense)
            r2 = 1.0 - float(np.sum((reconstructed - truth) ** 2)) / denom
            check(f"analysis: independent R2 {prefix}", math.isclose(
                r2, float(result[f"{prefix}_r2"]), rel_tol=0.0, abs_tol=5e-13
            ))
            model_x = periodic_interp(phase, result[f"{prefix}_tau_x"], dense)
            legacy_denom = float(np.sum((legacy_truth - np.mean(legacy_truth)) ** 2))
            legacy_r2 = 1.0 - float(np.sum((model_x - legacy_truth) ** 2)) / legacy_denom
            check(f"analysis: independent legacy-x R2 {prefix}", math.isclose(
                legacy_r2, float(result[f"{prefix}_legacy_x_r2"]),
                rel_tol=0.0, abs_tol=5e-13
            ))
            tangent_norm = np.hypot(result[f"{prefix}_tangent_x"],
                                    result[f"{prefix}_tangent_y"])
            check(f"analysis: unit tangent {prefix}", np.allclose(
                tangent_norm, 1.0, rtol=0.0, atol=5e-12
            ))
            check(f"analysis: positive resolution {prefix}", (
                np.all(result[f"{prefix}_wall_ds"] > 0.0)
                and np.all(result[f"{prefix}_ym"] > 0.0)
                and float(result[f"{prefix}_dz"]) > 0.0
            ))
            check(f"analysis: ten profile errors {prefix}", (
                result[f"{prefix}_profile_x"].shape == (10,)
                and np.allclose(
                    result[f"{prefix}_profile_x"],
                    0.45 + 0.9*np.arange(10), rtol=0.0, atol=5e-12,
                )
                and result[f"{prefix}_profile_u_rms_by_station"].shape == (10,)
                and np.all(result[f"{prefix}_profile_points_by_station"] >= 32)
            ))
            case_id = next(
                name for name, values in CASES.items()
                if values[1] == grid and values[2] == model
            )
            raw_phase, raw_tau, raw_integral = independent_raw_curve(CAMPAIGN / case_id)
            check(f"analysis: raw phase rebuild {prefix}", np.allclose(
                raw_phase, phase, rtol=0.0, atol=5e-13
            ))
            check(f"analysis: raw tangent traction rebuild {prefix}", np.allclose(
                raw_tau, tau, rtol=0.0, atol=5e-13
            ))
            check(f"analysis: raw tangent integral rebuild {prefix}", math.isclose(
                raw_integral, float(result[f"{prefix}_signed_tangent_force_per_span"]),
                rel_tol=0.0, abs_tol=5e-13
            ))
            ym_by_model.append(result[f"{prefix}_ym"])
        check(f"analysis: paired matching surface {grid}", np.array_equal(
            ym_by_model[0], ym_by_model[1]
        ))

check("scheduler: terminal evidence", SCHEDULER.is_file())
if SCHEDULER.is_file():
    scheduler = SCHEDULER.read_text()
    for job in ("14868882", "14868883", "14868884", "14868885", "14868887", "14868888"):
        check(f"scheduler: {job} completed", re.search(
            rf"^{job}\|COMPLETED\|0:0\|", scheduler, re.M
        ) is not None)

check("jobs: registry", JOB_REGISTRY.is_file())
if JOB_REGISTRY.is_file():
    registry = json.loads(JOB_REGISTRY.read_text())
    matrix = registry.get("production_matrix", [])
    check("jobs: six registered producers", (
        len(matrix) == 6
        and {item.get("job_id") for item in matrix} == {
            "14868882", "14868883", "14868884", "14868885", "14868887", "14868888"
        }
    ))
    check("jobs: producers terminal in registry", all(
        item.get("state") == "COMPLETED" and item.get("exit_code") == "0:0"
        for item in matrix
    ))
    check("jobs: finalizer terminal in registry", (
        registry.get("finalizer", {}).get("state") == "COMPLETED"
        and registry.get("finalizer", {}).get("exit_code") == "0:0"
        and (not campaign_manifest_path.is_file()
             or registry.get("finalizer", {}).get("job_id") ==
             campaign_manifest.get("finalizer_job_id"))
    ))

failed = [name for name, passed in checks if not passed]
for name, passed in checks:
    if "--verbose" in sys.argv or not passed:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
raise SystemExit(1 if failed else 0)
