#!/usr/bin/env python3
"""Produce the M5 audit for the deployed TBLE branch policy.

The artifact joins three independent evidence streams: an exact C++ replay of
the v1 failure states through the v3 scale-invariant census, live OpenFOAM face
telemetry from two completed ARCHER2 pilots, and four fail-closed production
aborts with scheduler records.  No manuscript number is entered by hand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
OPENFOAM = ROOT / "codes" / "openfoam"
FIXTURE = OPENFOAM / "verify_wall_model_branch_policy_m5.cpp"
V3 = OPENFOAM / "ladderWallModels_v3"
FAIL_DIR = RESULTS / "m5_live_failure_evidence"
PILOT_DIR = RESULTS / "r2m4_ladder_campaign"
PORTABILITY = RESULTS / "m5_portability_compile_20260823.txt"
OUT_JSON = RESULTS / "wall_model_branch_policy_m5.json"
OUT_NPZ = RESULTS / "wall_model_branch_policy_m5.npz"
NU = 0.0001785714


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_fixture() -> tuple[str, list[dict[str, float | int | str]], dict[str, float]]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required")
    with tempfile.TemporaryDirectory(prefix="m5_branch_") as name:
        executable = Path(name) / "verify_m5"
        subprocess.run(
            [compiler, "-O2", "-std=c++14", f"-I{OPENFOAM}",
             str(FIXTURE), "-o", str(executable)], check=True)
        run = subprocess.run([str(executable)], check=True,
                             capture_output=True, text=True)
    records: list[dict[str, float | int | str]] = []
    for line in run.stdout.splitlines():
        if not line.startswith("BRANCH "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        records.append({
            "case": fields["case"], "model": fields["model"],
            "scale": float(fields["scale"]), "roots": int(fields["roots"]),
            "distinct": int(fields["distinct"]), "scan": float(fields["scan"]),
            "expansions": int(fields["expansions"]),
            "continued": float(fields["continued"]),
            "homotopy": float(fields["homotopy"]),
            "residual": float(fields["residual"]),
            "steps": int(fields["steps"]), "pass": int(fields["pass"]),
        })
    bench_lines = [line for line in run.stdout.splitlines()
                   if line.startswith("BENCH ")]
    if (len(records) != 20 or len(bench_lines) != 1
            or "M5_BRANCH_POLICY_FIXTURE_OK" not in run.stdout):
        raise RuntimeError("incomplete M5 C++ fixture")
    bench_fields = dict(token.split("=", 1)
                        for token in bench_lines[0].split()[1:])
    bench = {key: float(value) for key, value in bench_fields.items()}
    return run.stdout, records, bench


def key_values(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z][A-Za-z0-9]*)=([^ ]+)", line))


def parse_pilot(path: Path) -> dict[str, object]:
    face_records, census_records, census_audits, realizability_records = [], [], [], []
    with path.open(errors="replace") as stream:
        for line in stream:
            if "LADDER_TBLE_FACE" in line:
                face_records.append(key_values(line))
            elif "LADDER_ROOT_CENSUS" in line:
                census_records.append(key_values(line))
            elif "LADDER_CENSUS_AUDIT" in line:
                census_audits.append(key_values(line))
            elif "LADDER_REALIZABILITY" in line:
                realizability_records.append(key_values(line))
    if (not face_records or not realizability_records
            or (not census_records and not census_audits)):
        raise RuntimeError(f"missing live telemetry in {path}")

    errors = {"rawNut": [], "upperNut": [], "nut": [],
              "appliedTau": [], "appliedTractionMag": []}
    roots, steps = [], []
    for item in face_records:
        tau = float(item["tauW"]); um = float(item["UMatch"])
        ut = float(item["UtMag"]); ym = float(item["ym"])
        raw = tau*ym/um - NU if abs(um) > 1.0e-14 else -np.finfo(float).max
        upper = max(abs(tau)*ym/max(ut, 1.0e-14)-NU, 0.0)
        nut = min(max(raw, 0.0), upper)
        expected = {
            "rawNut": raw, "upperNut": upper, "nut": nut,
            "appliedTau": (NU+nut)*um/ym,
            "appliedTractionMag": (NU+nut)*ut/ym,
        }
        for key, value in expected.items():
            errors[key].append(abs(float(item[key])-value))
        roots.append(int(item["roots"])); steps.append(int(item["homotopySteps"]))
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "face_records": len(face_records),
        "census_records": len(census_records),
        "census_audits": len(census_audits),
        "realizability_records": len(realizability_records),
        "max_roots_first_solve": max(roots),
        "homotopy_steps_unique": sorted(set(steps)),
        "projection_max_abs_error": {k: max(v) for k, v in errors.items()},
        "census_max_velocity_residual": (max(
            float(item["maxVelocityResidual"]) for item in census_records)
            if census_records else None),
        "census_max_roots": (max(int(item["maxRoots"])
                                 for item in census_records)
                             if census_records else None),
        "kernel_fixture_ok": "LADDER_KERNEL_FIXTURE_OK" in
            (path.parent / "log.ladderKernelFixture").read_text(errors="replace"),
    }


def parse_fatal() -> list[dict[str, object]]:
    scheduler = {}
    for line in (FAIL_DIR / "sacct_20260823.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        job, name, state, elapsed, nodes, code = line.split("|")
        scheduler[job] = {"job_name": name, "state": state,
                          "elapsed": elapsed, "nodes": int(nodes),
                          "exit_code": code}
    job_for_case = {
        "r2m4_L1_resolvedConvectionConstant_153600_v1": "14888773",
        "r2m4_L2_hickel_327680_v1": "14888778",
        "r2m4_L1_hickel_153600_v1": "14888771",
        "r2m4_L1_totalGradient_153600_v1": "14888767",
    }
    records = []
    for case, job in job_for_case.items():
        path = FAIL_DIR / f"{case}.log.pimpleFoam"
        matches = [key_values(line) for line in path.read_text(errors="replace").splitlines()
                   if "LADDER branch failure" in line]
        if len(matches) != 1:
            raise RuntimeError(f"expected one fatal record in {path}")
        item = matches[0]
        record = {
            "case": case, "job": job, "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path), "model": item["model"],
            "roots": int(item["roots"]),
            "branch_loss": int(item["branchLoss"]),
            "ambiguous": int(item["ambiguous"]),
            "truncated": int(item["truncated"]),
            "finite": int(item["finite"]), **scheduler[job],
        }
        records.append(record)
    return records


def main() -> None:
    fixture_stdout, branches, benchmark = compile_fixture()
    pilot_paths = sorted(PILOT_DIR.glob("r2m4_pilot_*_L1_v2/log.pimpleFoam"))
    pilots = [parse_pilot(path) for path in pilot_paths]
    fatals = parse_fatal()
    portability = PORTABILITY.read_text()
    scales = sorted({float(item["scale"]) for item in branches})
    by_case = {}
    for case in sorted({str(item["case"]) for item in branches}):
        group = [item for item in branches if item["case"] == case]
        by_case[case] = {
            "continued_span": float(np.ptp([item["continued"] for item in group])),
            "homotopy_span": float(np.ptp([item["homotopy"] for item in group])),
            "root_counts": [int(item["roots"]) for item in group],
            "distinct_counts": [int(item["distinct"]) for item in group],
        }
    checks = {
        "four_live_fail_closed_jobs": len(fatals) == 4,
        "scheduler_states_failed_nonzero": all(
            item["state"] == "FAILED" and item["exit_code"] != "0:0"
            for item in fatals),
        "fatal_records_are_complete_ambiguities": all(
            item["roots"] == 3 and item["branch_loss"] == 0
            and item["ambiguous"] == 1 and item["truncated"] == 0
            and item["finite"] == 1 for item in fatals),
        "four_cases_five_scan_scales": len(branches) == 20
            and scales == [2.0, 4.0, 8.0, 16.0, 32.0],
        "all_branch_fixture_rows_pass": all(item["pass"] == 1 for item in branches),
        "continued_branch_invariant": all(
            item["continued_span"] < 2e-12 for item in by_case.values()),
        "homotopy_branch_invariant": all(
            item["homotopy_span"] < 2e-12 for item in by_case.values()),
        "default_v2_v3_bit_identity": fixture_stdout.count(
            "DEFAULT_IDENTITY") == 4 and "pass=0" not in fixture_stdout,
        "nonfinite_path_is_explicit_failure":
            "NONFINITE pass=1 converged=0 finite=0 tauFinite=0" in fixture_stdout,
        "two_live_openfoam_pilots": len(pilots) == 2
            and all(item["face_records"] == 9600 for item in pilots),
        "live_census_and_realizability_telemetry": all(
            (item["census_records"] >= 4 or item["census_audits"] >= 2)
            and item["realizability_records"] >= 4
            for item in pilots),
        "applied_projection_rebuild": all(
            max(item["projection_max_abs_error"].values()) < 5e-12
            for item in pilots),
        "live_kernel_fixture_and_face_census": all(
            item["kernel_fixture_ok"] and item["max_roots_first_solve"] >= 1
            and item["homotopy_steps_unique"] == [33] for item in pilots),
        "foundation_and_esi_compile": portability.count("M5_V3_COMPILE_OK") == 3
            and "org-v10" in portability and "com-v2512" in portability,
        "standalone_initial_and_local_cost_recorded":
            0 < benchmark["initial_us"] < 1e6
            and 0 < benchmark["local_us"] < 1e3,
        "single_shared_v3_call_site": "../ladderWallModels_v2/ladderTbleNutFvPatchScalarField.C"
            in (V3 / "ladderTbleNutFvPatchScalarField.C").read_text(),
    }
    summary = {
        "schema": "wall-model-branch-policy-m5-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "scales": scales, "branch_cases": by_case,
        "pilots": pilots, "fatal_jobs": fatals, "benchmark": benchmark,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in
            [FIXTURE, V3 / "ladderTbleShootScaleInvariant.H",
             V3 / "ladderTbleNutFvPatchScalarField.C",
             V3 / "registeredMeanVelocityForceCompat.C", PORTABILITY]
        },
        "interpretation": (
            "The v1 boundary condition failed closed on four finite three-root "
            "states. V3 merges duplicate representations of the zero-stress "
            "branch and preserves the production-width v2 result bit for bit, "
            "while an adaptive census makes pressure-homotopy and previous-time "
            "selection invariant to scan multipliers 2--32."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez(
        OUT_NPZ,
        case=np.asarray([item["case"] for item in branches]),
        model=np.asarray([item["model"] for item in branches]),
        scale=np.asarray([item["scale"] for item in branches]),
        roots=np.asarray([item["roots"] for item in branches]),
        distinct=np.asarray([item["distinct"] for item in branches]),
        continued=np.asarray([item["continued"] for item in branches]),
        homotopy=np.asarray([item["homotopy"] for item in branches]),
        residual=np.asarray([item["residual"] for item in branches]),
        fatal_job=np.asarray([item["job"] for item in fatals]),
        fatal_exit=np.asarray([item["exit_code"] for item in fatals]),
        check_name=np.asarray(list(checks)),
        check_pass=np.asarray(list(checks.values()), dtype=bool),
    )
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"M5 branch-policy producer: {sum(checks.values())}/{len(checks)} checks passed")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
