#!/usr/bin/env python3
"""Write the machine-readable WP0 reconciliation of the hill headline score.

The corrected, hill-surface-aware value is the only canonical value.  The
wall-pinned value is retained solely as provenance for the extraction defect.
All numbers below are read from regenerated artifacts; none is duplicated as a
free-standing constant except for the acceptance assertion.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
OUT = os.path.join(RESULTS, "headline_number_reconciliation.json")
CANONICAL = -47.68617253416459


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def main():
    ce_path = artifact("criterion_evaluation.npz")
    dc_path = artifact("diagnostic_test_corrected.npz")
    pc_path = artifact("pehill_5case_corrected.npz")
    wp_path = artifact("periodic_hills_case_1p0_wall_profiles_corrected.npz")
    wa_path = artifact("wall_extraction_artifact.npz")
    core_path = artifact("core_ode_results_surface_aware.npz")

    ce = np.load(ce_path, allow_pickle=True)
    names = [str(x) for x in ce["geom_names"]]
    hill_idx = names.index("periodic_hills_case_1p0")
    corrected = float(ce["ode_r2"][hill_idx])
    wa = np.load(wa_path, allow_pickle=True)
    legacy = float(wa["legacy_wall_pinned_r2_tau"])
    dc = np.load(dc_path, allow_pickle=True)
    pc = np.load(pc_path, allow_pickle=True)
    pc_idx = [str(x) for x in pc["case"]].index("case_1p0")
    independent = {
        "criterion_evaluation": corrected,
        "diagnostic_test_corrected": float(dc["standard_ml_r2"]),
        "pehill_5case_corrected": float(pc["r2"][pc_idx]),
        "core_ode_results_surface_aware": float(
            np.load(core_path, allow_pickle=True)["canonical_hill_r2"]),
    }
    if not all(abs(value - CANONICAL) <= 1e-9
               for value in independent.values()):
        raise RuntimeError(f"Corrected canonical score disagrees: {independent}")

    record = {
        "schema": "wp0-headline-number-reconciliation-v1",
        "canonical": {
            "case": "periodic_hills_case_1p0",
            "metric": "R2_tau_w",
            "value": corrected,
            "matching_index": int(dc["Y_IDX"]),
            "extraction": "hill_surface_aware",
            "status": "canonical_for_JCP",
            "source_artifact": os.path.relpath(wp_path, CODES),
            "source_sha256": sha256(wp_path),
        },
        "independent_reproductions": independent,
        "superseded": {
            "value": legacy,
            "extraction": "global_y_zero_wall_pinning",
            "status": "historical_artifact_only_not_for_manuscript_or_figures",
            "documented_by": os.path.relpath(wa_path, CODES),
            "documentation_sha256": sha256(wa_path),
        },
        "live_artifact_hashes": {
            os.path.relpath(path, CODES): sha256(path)
            for path in (ce_path, core_path, dc_path, pc_path, wp_path, wa_path)
        },
    }
    with open(OUT, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"canonical R2(tau_w): {corrected:.14f}")
    print(f"superseded wall-pinned R2: {legacy:.9g}")
    print(f"Saved -> {os.path.relpath(OUT, CODES)}")


if __name__ == "__main__":
    main()
