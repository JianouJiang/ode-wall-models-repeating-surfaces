#!/usr/bin/env python3
"""Lightweight stable-path guard for the completed WP0 deposit repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scalar(path: Path, key: str) -> float:
    with np.load(path, allow_pickle=False) as data:
        return float(np.asarray(data[key]).reshape(-1)[0])


def main() -> int:
    reconcile = json.loads((RESULTS / "headline_number_reconciliation.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "reproduce_audit.json").read_text(encoding="utf-8"))
    links = [path for path in (ROOT / "codes/new_data_download").iterdir()
             if path.is_symlink()]
    broken = [path for path in links if not path.resolve().exists()]
    values = {
        "core_ode_results_surface_aware": scalar(RESULTS / "core_ode_results_surface_aware.npz", "canonical_hill_r2"),
        "criterion_evaluation": scalar(RESULTS / "criterion_evaluation.npz", "canonical_hill_r2"),
        "diagnostic_test_corrected": scalar(RESULTS / "diagnostic_test_corrected.npz", "standard_ml_r2"),
        "pehill_5case_corrected": scalar(RESULTS / "pehill_5case_corrected.npz", "case_1p0_r2"),
    }
    canonical = float(reconcile["canonical"]["value"])
    checks = [
        ("eight logical raw-data links exist", len(links) == 8),
        ("no raw-data link is broken", not broken),
        ("reproduction audit artifacts present", audit["all_artifacts_present"]),
        ("reproduction audit artifacts written in run window", audit["all_artifacts_written_in_run_window"]),
        ("canonical score is corrected", canonical == -47.68617253416459),
        ("four independent score paths agree bit-exactly", all(value == canonical for value in values.values())),
        ("canonical source hash matches", digest(RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz") == reconcile["canonical"]["source_sha256"]),
        ("legacy score is labelled superseded", reconcile["superseded"]["status"] == "historical_artifact_only_not_for_manuscript_or_figures"),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
