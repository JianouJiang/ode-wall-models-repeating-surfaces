#!/usr/bin/env python3
"""Build the project-owned ODE-result bundle consumed by the JCP pipeline.

Unlike the retired vendor summary, this producer never evaluates a periodic
hill from a globally y=0-pinned profile.  The canonical hill is evaluated from
the regenerated surface-aware profile, and the other three Xiao variants are
evaluated directly from their raw DNS columns with the same surface locator.
All non-hill core cases are recomputed from the declared profile products.
"""
from __future__ import annotations

import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
OUT = os.path.join(RESULTS, "core_ode_results_surface_aware.npz")
Y_INDICES = (5, 10, 20)

sys.path.insert(0, CODES)
import manifest  # noqa: E402

sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                               "codes", "analysis"))
from ode_wall_model import evaluate_on_dataset  # noqa: E402

from pehill_5case_corrected import BASE, VARIANTS, evaluate as evaluate_hill  # noqa: E402


def store_standard(out, name, by_index):
    for y_idx, result in by_index.items():
        if result.get("n_valid", 0) == 0:
            continue
        prefix = f"{name}_yidx{y_idx}"
        for source, suffix in (
            ("r2_tau", "r2_tau"),
            ("r2_cf", "r2_cf"),
            ("sign_acc_sep", "sign_acc_sep"),
            ("n_valid", "n_valid"),
            ("n_sep", "n_sep"),
        ):
            out[f"{prefix}_{suffix}"] = np.asarray(result[source])


def main():
    out = {
        "schema": np.array("core-ode-results-surface-aware-v1"),
        "matching_indices": np.asarray(Y_INDICES, dtype=int),
        "periodic_hill_extraction": np.array("first_fluid_node_above_local_surface"),
        "non_hill_extraction": np.array("declared_profile_coordinates"),
    }

    vendor_results = os.path.join(CODES, "vendor", "universal_wall_function",
                                  "codes", "results")
    canonical_hill = "periodic_hills_case_1p0"
    for name in manifest.core_names():
        if name == canonical_hill:
            profile = os.path.join(
                RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
        else:
            profile = os.path.join(vendor_results, f"{name}_wall_profiles.npz")
        if not os.path.isfile(profile):
            raise FileNotFoundError(profile)
        store_standard(out, name,
                       evaluate_on_dataset(profile, y_indices=list(Y_INDICES)))
        print(f"[core] {name}: recomputed", flush=True)

    # Include every periodic variant named in the benchmark, using raw-DNS,
    # local-surface extraction.  The canonical case above deliberately remains
    # under its manifest name; these fields make the full variant audit explicit.
    for _label, case_name, _h_over_lx in VARIANTS:
        case_dir = os.path.join(BASE, case_name, "dns-data")
        for y_idx in Y_INDICES:
            result = evaluate_hill(case_dir, y_idx=y_idx)
            prefix = f"periodic_hills_{case_name}_yidx{y_idx}"
            for key in ("r2", "rel_err", "sign_acc", "eps_median", "f_sep", "n"):
                out[f"{prefix}_{key}"] = np.asarray(result[key])
        print(f"[hill/raw] {case_name}: y_idx={Y_INDICES}", flush=True)

    canonical = float(out["periodic_hills_case_1p0_yidx10_r2_tau"])
    raw_crosscheck = float(out["periodic_hills_case_1p0_yidx10_r2"])
    if abs(canonical - raw_crosscheck) > 1e-9:
        raise RuntimeError(
            f"canonical corrected profile/raw mismatch: {canonical} vs {raw_crosscheck}")
    out["canonical_hill_r2"] = np.asarray(canonical)
    out["canonical_hill_protocol"] = np.array("surface_aware_yidx10")
    np.savez(OUT, **out)
    print(f"canonical hill R2(tau_w) = {canonical:.14f}")
    print(f"Saved -> results/{os.path.basename(OUT)} ({len(out)} fields)")


if __name__ == "__main__":
    main()
