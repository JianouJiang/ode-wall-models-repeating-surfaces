#!/usr/bin/env python3
"""Run the M9 parent/direct-force certificate on the corrected multi-pitch rib.

The scientific operator is ``direct_force_adequacy_certificate_l1.py``.  This
driver supplies only the replacement case path, its recorded box length and
viscosity, and the five latest complete cumulative-average snapshots.  It is
intended to run on ARCHER2 after ``r24_rib_dtype_p3_G1`` reaches t=140 because
the reconstructed fields are deliberately not copied into the compact R2-4
deposit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "codes/analysis/direct_force_adequacy_certificate_l1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_producer():
    spec = importlib.util.spec_from_file_location("m9_corrected_operator", PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_times(case: Path) -> list[str]:
    required = ("U", "nut", "UMean", "UPrime2Mean", "pMean")
    times = []
    for path in case.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and all((path / name).is_file() for name in required):
            times.append((value, path.name))
    return [name for _, name in sorted(times)[-5:]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default=str(ROOT / "jobs/r24_rib_dtype_p3_G1"),
        help="terminal reconstructed OpenFOAM case",
    )
    parser.add_argument(
        "--node-output",
        default=str(ROOT / "development/nodes/node_007/m9_corrected"),
    )
    args = parser.parse_args()
    case = Path(args.case).resolve()
    provenance_path = case / "PROVENANCE.json"
    if not provenance_path.is_file():
        raise SystemExit(f"missing provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    times = complete_times(case)
    if len(times) != 5 or float(times[-1]) < 139.0:
        raise SystemExit(f"need five complete snapshots through t=140; found {times}")
    log = case / "log.pimpleFoam"
    if not log.is_file() or "\nEnd\n" not in log.read_text(errors="replace"):
        raise SystemExit("solver log is not terminal")

    producer = load_producer()
    producer.CASE = case
    producer.MESH = case / "constant/polyMesh"
    producer.TIMES = tuple(times)
    producer.CENTRAL_TIME = times[-1]
    producer.DOMAIN_LENGTH = float(provenance["Lx_box"])
    producer.NU = float(provenance["nu"])
    producer.EXPECTED_PHASE_CELLS = 0
    producer.CASE_LABEL = (
        "matched-numerics multi-pitch d-type square-rib WRLES "
        f"({case.name}, OpenFOAM 10, WALE)"
    )
    producer.NODE = Path(args.node_output).resolve()
    producer.main()

    summary_path = ROOT / "codes/results/direct_force_adequacy_certificate_l1.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "PASS" or summary["mesh"]["cells"] <= 94976:
        raise SystemExit("corrected M9 certificate did not replace the one-pitch substrate")
    npz_path = ROOT / "codes/results/direct_force_adequacy_certificate_l1.npz"
    config = {
        "schema": "m9-corrected-rib-remote-rebuild-v1",
        "status": "M9_CORRECTED_RIB_REBUILD_OK",
        "case": str(case.relative_to(ROOT)),
        "times": times,
        "domain_length": producer.DOMAIN_LENGTH,
        "nu": producer.NU,
        "phase_control_volumes": summary["mesh"]["phase_control_volumes"],
        "cells": summary["mesh"]["cells"],
        "phase_closure": summary["phase_parent_closure"]["residual_over_full_leg_l2"],
        "wavelength_closure": summary["wavelength_parent_closure"]["relative_to_direct"],
        "source_hashes": summary["source_hashes"],
        "producer_sha256": sha256(PRODUCER),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "case_provenance_sha256": sha256(provenance_path),
        "solver_log_sha256": sha256(log),
        "result_sha256": {
            "direct_force_adequacy_certificate_l1.json": sha256(summary_path),
            "direct_force_adequacy_certificate_l1.npz": sha256(npz_path),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
    }
    producer.NODE.mkdir(parents=True, exist_ok=True)
    (producer.NODE / "rebuild_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print("M9_CORRECTED_RIB_REBUILD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
