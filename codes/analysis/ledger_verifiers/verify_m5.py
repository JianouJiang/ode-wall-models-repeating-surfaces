#!/usr/bin/env python3
"""Independent verifier for M5 wall-model numerics and branch policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OPENFOAM = ROOT / "codes" / "openfoam"
RESULTS = ROOT / "codes" / "results"
SUMMARY = RESULTS / "wall_model_branch_policy_m5.json"
ARRAYS = RESULTS / "wall_model_branch_policy_m5.npz"
FAIL_DIR = RESULTS / "m5_live_failure_evidence"
PILOTS = RESULTS / "r2m4_ladder_campaign"
FIXTURE = OPENFOAM / "verify_wall_model_branch_policy_m5.cpp"
NU = 0.0001785714


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def kv(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)", line))


def compile_fixture(root: Path, fixture: Path) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required")
    with tempfile.TemporaryDirectory(prefix="verify_m5_") as name:
        exe = Path(name) / "m5"
        built = subprocess.run(
            [compiler, "-O2", "-std=c++14", f"-I{root}", str(fixture),
             "-o", str(exe)], capture_output=True, text=True)
        if built.returncode:
            return built
        return subprocess.run([str(exe)], capture_output=True, text=True)


def projection_error(item: dict[str, str]) -> float:
    tau = float(item["tauW"]); um = float(item["UMatch"])
    ut = float(item["UtMag"]); ym = float(item["ym"])
    raw = tau*ym/um-NU if abs(um) > 1e-14 else -np.finfo(float).max
    upper = max(abs(tau)*ym/max(ut, 1e-14)-NU, 0.0)
    nut = min(max(raw, 0.0), upper)
    expected = [raw, upper, nut, (NU+nut)*um/ym, (NU+nut)*ut/ym]
    observed = [float(item[key]) for key in
                ["rawNut", "upperNut", "nut", "appliedTau",
                 "appliedTractionMag"]]
    return max(abs(a-b) for a, b in zip(expected, observed))


def fatal_valid(text: str) -> bool:
    rows = [kv(line) for line in text.splitlines()
            if "LADDER branch failure" in line]
    return len(rows) == 1 and rows[0].get("roots") == "3" \
        and rows[0].get("branchLoss") == "0" \
        and rows[0].get("ambiguous") == "1" \
        and rows[0].get("truncated") == "0" \
        and rows[0].get("finite") == "1"


def active_tex() -> str:
    text = (ROOT / "manuscript" / "main.tex").read_text()
    # Remove inactive migration blocks so stale claims cannot satisfy M5.
    while "\\iffalse" in text:
        start = text.index("\\iffalse")
        end = text.index("\\fi", start) + len("\\fi")
        text = text[:start] + text[end:]
    return text


def main() -> int:
    checks: list[tuple[str, bool]] = []
    summary = json.loads(SUMMARY.read_text())
    arrays = np.load(ARRAYS)
    check = lambda name, ok: checks.append((name, bool(ok)))

    check("producer status and 16 checks", summary.get("status") == "PASS"
          and len(summary.get("checks", {})) == 16
          and all(summary["checks"].values()))
    check("npz carries 20 passing scale rows", len(arrays["scale"]) == 20
          and set(arrays["scale"].tolist()) == {2., 4., 8., 16., 32.}
          and bool(np.all(arrays["check_pass"])))
    check("stored source hashes", all(
        (ROOT / path).is_file() and digest(ROOT / path) == value
        for path, value in summary["source_hashes"].items()))

    run = compile_fixture(OPENFOAM, FIXTURE)
    check("independent C++ replay", run.returncode == 0
          and run.stdout.count("BRANCH case=") == 20
          and run.stdout.count("DEFAULT_IDENTITY") == 4
          and "M5_BRANCH_POLICY_FIXTURE_OK" in run.stdout)
    benches = [kv(line) for line in run.stdout.splitlines()
               if line.startswith("BENCH ")]
    check("standalone cost independently measured", len(benches) == 1
          and 0 < float(benches[0]["initial_us"]) < 1e6
          and 0 < float(benches[0]["local_us"]) < 1e3)
    branch = [kv(line) for line in run.stdout.splitlines()
              if line.startswith("BRANCH ")]
    check("independent bracket invariance", len(branch) == 20
          and all(item["pass"] == "1" for item in branch)
          and all(np.ptp([float(item[field]) for item in branch
                          if item["case"] == case]) < 2e-12
                  for case in {item["case"] for item in branch}
                  for field in ["continued", "homotopy"]))

    fatal_paths = sorted(FAIL_DIR.glob("r2m4_*_v1.log.pimpleFoam"))
    check("four independent fail-closed logs", len(fatal_paths) == 4
          and all(fatal_valid(path.read_text(errors="replace"))
                  for path in fatal_paths))
    scheduler = [line for line in (FAIL_DIR / "sacct_20260823.txt")
                 .read_text().splitlines() if line and not line.startswith("#")]
    check("scheduler failure state and nonzero exit", len(scheduler) == 4
          and all("|FAILED|" in line and not line.endswith("|0:0")
                  for line in scheduler))

    pilot_paths = sorted(PILOTS.glob("r2m4_pilot_*_L1_v2/log.pimpleFoam"))
    face_count = 0; census_audits = 0; realizability = 0; max_error = 0.0
    first_face = None
    for path in pilot_paths:
        for line in path.read_text(errors="replace").splitlines():
            if "LADDER_TBLE_FACE" in line:
                item = kv(line); first_face = first_face or item
                face_count += 1; max_error = max(max_error, projection_error(item))
            elif "LADDER_CENSUS_AUDIT" in line:
                census_audits += 1
            elif "LADDER_REALIZABILITY" in line:
                realizability += 1
    check("live OpenFOAM telemetry independently parsed", len(pilot_paths) == 2
          and face_count == 19200 and census_audits >= 4
          and realizability >= 8 and max_error < 5e-12)

    portability = (RESULTS / "m5_portability_compile_20260823.txt").read_text()
    check("Foundation 10 and ESI v2512 builds", portability.count(
          "M5_V3_COMPILE_OK") == 3 and "org-v10" in portability
          and "com-v2512" in portability)

    # Control case 1: remove adaptive resolution.  At least one 16--32 scan
    # must then miss the nonzero branch, making the exact same fixture fail.
    with tempfile.TemporaryDirectory(prefix="m5_red_scan_") as name:
        temp = Path(name)
        shutil.copytree(OPENFOAM / "ladderWallModels_v2",
                        temp / "ladderWallModels_v2")
        shutil.copytree(OPENFOAM / "ladderWallModels_v3",
                        temp / "ladderWallModels_v3")
        red_fixture = temp / FIXTURE.name
        shutil.copy2(FIXTURE, red_fixture)
        header = temp / "ladderWallModels_v3" / "ladderTbleShootScaleInvariant.H"
        altered = header.read_text().replace(
            "const int scanIntervals = TBLE_SCAN_INTERVALS*widthFactor;",
            "const int scanIntervals = TBLE_SCAN_INTERVALS;")
        header.write_text(altered)
        red = compile_fixture(temp, red_fixture)
    check("control case rejects fixed-resolution wide bracket", red.returncode != 0
          and "M5_BRANCH_POLICY_FIXTURE_FAIL" in red.stdout)

    fatal_text = fatal_paths[0].read_text(errors="replace")
    check("control case rejects hidden ambiguity", fatal_valid(fatal_text)
          and not fatal_valid(fatal_text.replace("ambiguous=1", "ambiguous=0", 1)))
    if first_face:
        tampered = dict(first_face)
        tampered["nut"] = str(float(tampered["nut"])+1e-3)
        check("control case rejects wrong applied projection",
              projection_error(first_face) < 5e-12
              and projection_error(tampered) > 1e-4)
    else:
        check("control case rejects wrong applied projection", False)

    tex = active_tex()
    check("active manuscript states registered branch policy",
          "pressure-source homotopy" in tex
          and "closest previously converged root" in tex
          and "terminates" in tex)

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(ok for _, ok in checks)
    print(f"M5: {passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
