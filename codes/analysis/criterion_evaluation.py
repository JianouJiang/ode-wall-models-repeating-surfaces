#!/usr/bin/env python3
"""
Evaluate how the manuscript's geometry-level diagnostics separate successful
and failed ODE wall-model cases across the core benchmark.

The core benchmark is the 31 datasets declared in codes/manifest.py (the 34 of
manuscript Table 1 minus three periodic-hills variants whose DNS lacks Reynolds
shear stress, <u'v'> = 0, making the budget ratio R undefined). The dataset set
is taken from the manifest, NOT by globbing, so the result is deterministic
regardless of which other artifacts live in the data directory.

Outputs:
  - codes/results/criterion_evaluation.npz

PRIMARY classification: evaluated on the 9 non-trivial multi-station datasets
only (APG TBLs, separation bubble, NASA hump, BFS, periodic hills) using
f_sep. The 22 single-station cases (channels, pipes, Couette, ZPG TBL) are
excluded from the primary classification because the ODE reduces to the
equilibrium solution there.

SECONDARY (NEGATIVE RESULT): the composite metric C = Rbar * f_sep was
explored but does not improve classification beyond f_sep alone. It
mis-orders BFS (C=0.26, accurate) and periodic hills (C=0.23, fails).
The threshold machinery below is retained for reproducibility only.
"""
import os
import numpy as np
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'utils')))
# Single source of truth: which datasets belong to the manuscript analyses.
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..')))
import manifest

BUDGET_PATH = os.path.join(RESULTS_DIR, "momentum_budget_all_geometries.npz")
ODE_PATH = os.path.join(RESULTS_DIR, "core_ode_results_surface_aware.npz")
CORRECTED_HILL_PATH = os.path.join(
    RESULTS_DIR, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
Y_IDX = 10

# The vendor analysis directory contains the production ODE implementation used
# throughout the paper.  Re-evaluating the corrected profile here prevents the
# project-owned surface-aware bundle independently cross-checked below.
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "vendor",
                               "universal_wall_function", "codes", "analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402


def separating_interval(values, labels):
    """Return the open interval that perfectly separates success and failure."""
    success_max = float(values[labels == 1].max())
    failure_min = float(values[labels == 0].min())
    width = failure_min - success_max
    return success_max, failure_min, width


def summarize_subset(values, labels, mask, threshold):
    """Summarize separation properties on a given subset."""
    sub_values = values[mask]
    sub_labels = labels[mask]
    success_max, failure_min, width = separating_interval(sub_values, sub_labels)
    pred = (sub_values < threshold).astype(int)
    accuracy = float((pred == sub_labels).mean())
    return {
        "count": int(mask.sum()),
        "success_count": int((sub_labels == 1).sum()),
        "failure_count": int((sub_labels == 0).sum()),
        "success_max": success_max,
        "failure_min": failure_min,
        "interval_width": width,
        "accuracy": accuracy,
    }


def corrected_hill_r2(path, y_idx=Y_IDX):
    """Recompute the canonical hill score from the surface-aware profiles."""
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["y"], dtype=float)
    U = np.asarray(d["U"], dtype=float)
    tau_true = np.asarray(d["tau_w"], dtype=float)
    dp_dx = np.asarray(d["dp_dx"], dtype=float)
    nu = np.asarray(d["nu"], dtype=float)
    if nu.ndim == 0:
        nu = np.full(tau_true.size, float(nu))
    tau_pred = np.full(tau_true.size, np.nan)
    for i in range(tau_true.size):
        if y_idx >= y.shape[1]:
            continue
        y_m, U_m = y[i, y_idx], U[i, y_idx]
        if not (np.isfinite(y_m) and np.isfinite(U_m) and y_m > 0):
            continue
        tau_pred[i] = predict_tau_w(U_m, y_m, dp_dx[i], nu[i])
    valid = np.isfinite(tau_pred) & np.isfinite(tau_true)
    truth = tau_true[valid]
    residual = truth - tau_pred[valid]
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    if valid.sum() < 2 or ss_tot <= 0:
        raise RuntimeError("Corrected canonical-hill profiles cannot define R^2")
    return 1.0 - float(np.sum(residual ** 2)) / ss_tot, int(valid.sum())


budget = np.load(BUDGET_PATH, allow_pickle=True)
ode = np.load(ODE_PATH, allow_pickle=True)

# ---------------------------------------------------------------------------
# Manifest-driven selection (SINGLE SOURCE OF TRUTH).
#
# Previously this script iterated over every name in the budget file, which had
# silently grown to include extended/exploratory datasets that have no entry in
# the vendor ODE-result file -> KeyError, and a stale, non-regenerable
# criterion_evaluation.npz. We now select EXACTLY the core benchmark declared in
# codes/manifest.py and join it against the budget by name. Anything in the
# budget that is not a core dataset is reported and ignored.
# ---------------------------------------------------------------------------
budget_by_name = {str(n): i for i, n in enumerate(budget["geom_names"])}

core_names = manifest.core_names()
missing_from_budget = [n for n in core_names if n not in budget_by_name]
if missing_from_budget:
    raise RuntimeError(
        "Core datasets missing from the momentum budget; rerun "
        "momentum_budget_all_geometries.py first: " + ", ".join(missing_from_budget))

ignored = sorted(set(budget_by_name) - set(core_names))
if ignored:
    print(f"[manifest] ignoring {len(ignored)} non-core budget entries "
          f"(extended/exploratory): {', '.join(ignored)}")

sel = np.array([budget_by_name[n] for n in core_names])
all_geom_names = np.array(core_names)
# Geometry type comes from the manifest (authoritative), not the budget file.
all_geom_types = np.array([manifest.core_geom_type(n) for n in core_names])
all_mean_ratio = np.array(budget["mean_ratio"], dtype=float)[sel]
all_n_stations = np.array(budget["n_stations"], dtype=int)[sel]
all_n_separated = np.array(budget["n_separated"], dtype=int)[sel]
all_frac_sep = np.where(all_n_stations > 0, all_n_separated / all_n_stations, 0.0)
all_conv_intensity = all_mean_ratio * all_frac_sep

ode_r2 = {
    key.replace("_yidx10_r2_tau", ""): float(ode[key])
    for key in ode.files
    if key.endswith("_yidx10_r2_tau")
}

hill_key = "periodic_hills_case_1p0"
if hill_key not in ode_r2:
    raise KeyError(f"Missing surface-aware comparison value for {hill_key} in {ODE_PATH}")
if not os.path.exists(CORRECTED_HILL_PATH):
    raise FileNotFoundError(
        "Run build_corrected_pehill_profiles.py before criterion_evaluation.py: "
        + CORRECTED_HILL_PATH)
canonical_hill_r2, canonical_hill_n = corrected_hill_r2(CORRECTED_HILL_PATH)
bundled_hill_r2 = float(ode_r2[hill_key])
if abs(bundled_hill_r2 - canonical_hill_r2) > 1e-9:
    raise RuntimeError("Project-owned hill score disagrees with profile recomputation: "
                       f"{bundled_hill_r2} vs {canonical_hill_r2}")
ode_r2[hill_key] = canonical_hill_r2
print(f"[protocol] canonical surface-aware hill R^2 = "
      f"{canonical_hill_r2:.9f} (N={canonical_hill_n}, Y_IDX={Y_IDX})")

# Multi-station "primary" status comes from the manifest; cross-check the budget.
all_multistation_flag = np.array([manifest.core_is_multistation(n) for n in core_names])
if not np.array_equal(all_multistation_flag, all_n_stations > 1):
    mism = [n for n, f, ns in zip(core_names, all_multistation_flag, all_n_stations)
            if f != (ns > 1)]
    raise RuntimeError(f"manifest multi_station flag disagrees with budget n_stations "
                       f"for: {', '.join(mism)}")

all_labels = np.zeros(len(all_geom_names), dtype=int)
for i, name in enumerate(all_geom_names):
    if not all_multistation_flag[i]:
        # Single-station equilibrium case: ODE reduces to the equilibrium
        # solution, so it is a success by construction.
        all_labels[i] = 1
    else:
        if name not in ode_r2:
            raise KeyError(
                f"No ODE result '{name}_yidx10_r2_tau' for multi-station core "
                f"dataset '{name}' in {ODE_PATH}.")
        all_labels[i] = int(ode_r2[name] > 0.0)

metrics = {
    "frac_sep": all_frac_sep,
    "mean_ratio": all_mean_ratio,
    "conv_intensity": all_conv_intensity,
}

# PRIMARY metric: f_sep (fraction of separated stations)
fsep_threshold = 0.30  # any value in (0.24, 0.40) works on this sample

# SECONDARY (exploratory) metric: C = Rbar * f_sep
c_threshold = 0.30  # kept for reference; does NOT improve over f_sep

all_multi_station = all_n_stations > 1
all_separated_multi = all_multi_station & (all_frac_sep > 0.0)
primary_mask = all_multi_station

# PRIMARY accuracy (f_sep): 9 non-trivial multi-station datasets only
pred_fsep_multi = (all_frac_sep[all_multi_station] < fsep_threshold).astype(int)
primary_accuracy_fsep = float((pred_fsep_multi == all_labels[all_multi_station]).mean())

# SECONDARY accuracy (C): 9 non-trivial multi-station datasets only
pred_c_multi = (all_conv_intensity[all_multi_station] < c_threshold).astype(int)
secondary_accuracy_c = float((pred_c_multi == all_labels[all_multi_station]).mean())

# Sanity check: all datasets (includes 22 trivial single-station)
pred_fsep_all = (all_frac_sep < fsep_threshold).astype(int)
sanity_accuracy = float((pred_fsep_all == all_labels).mean())

subset_summary_fsep = {
    "all": summarize_subset(all_frac_sep, all_labels, np.ones(len(all_labels), dtype=bool), fsep_threshold),
    "multi_station": summarize_subset(all_frac_sep, all_labels, all_multi_station, fsep_threshold),
    "separated_multi": summarize_subset(all_frac_sep, all_labels, all_separated_multi, fsep_threshold),
}
subset_summary_c = {
    "all": summarize_subset(all_conv_intensity, all_labels, np.ones(len(all_labels), dtype=bool), c_threshold),
    "multi_station": summarize_subset(all_conv_intensity, all_labels, all_multi_station, c_threshold),
    "separated_multi": summarize_subset(all_conv_intensity, all_labels, all_separated_multi, c_threshold),
}

summary = {}
for name, values in metrics.items():
    success_max, failure_min, width = separating_interval(values[primary_mask], all_labels[primary_mask])
    all_success_max, all_failure_min, all_width = separating_interval(values, all_labels)
    summary[name] = {
        "primary_success_max": success_max,
        "primary_failure_min": failure_min,
        "primary_interval_width": width,
        "primary_separable": bool(width > 0.0),
        "all_success_max": all_success_max,
        "all_failure_min": all_failure_min,
        "all_interval_width": all_width,
        "all_separable": bool(all_width > 0.0),
    }

primary_names = all_geom_names[primary_mask]
primary_geom_types = all_geom_types[primary_mask]
primary_labels = all_labels[primary_mask]
primary_frac_sep = all_frac_sep[primary_mask]
primary_mean_ratio = all_mean_ratio[primary_mask]
primary_conv_intensity = all_conv_intensity[primary_mask]
primary_ode_r2 = np.array([ode_r2[name] for name in primary_names], dtype=float)
primary_ode_r2_protocol = np.array([
    "hill_surface_aware_yidx10" if name == hill_key
    else "project_owned_profile_yidx10"
    for name in primary_names
])

for name in all_geom_names[np.argsort(all_conv_intensity)]:
    i = int(np.where(all_geom_names == name)[0][0])
    status = "success" if all_labels[i] else "failure"
    print(
        f"{name:30s}  type={all_geom_types[i]:18s}  "
        f"C={all_conv_intensity[i]:8.4f}  Rbar={all_mean_ratio[i]:8.4f}  "
        f"f_sep={all_frac_sep[i]:7.4f}  {status}"
    )

print("\nMetric separation summary:")
for name, stats in summary.items():
    print(
        f"  {name:15s} primary_success_max={stats['primary_success_max']:.6f}  "
        f"primary_failure_min={stats['primary_failure_min']:.6f}  "
        f"primary_width={stats['primary_interval_width']:.6f}  "
        f"primary_separable={stats['primary_separable']}"
    )

print(f"\nPRIMARY metric: f_sep (threshold = {fsep_threshold:.2f})")
print(f"  Accuracy (9 multi-station): {primary_accuracy_fsep:.3f}")
print(f"  Accuracy (all {len(all_labels)}):           {sanity_accuracy:.3f}")
print(f"\nSECONDARY (exploratory) metric: C (threshold = {c_threshold:.2f}) -- (DEPRECATED)")
print(f"  Accuracy (9 multi-station): {secondary_accuracy_c:.3f}")
print("\nSubset summary for f_sep (primary):")
for name, stats in subset_summary_fsep.items():
    print(
        f"  {name:15s} count={stats['count']:2d}  "
        f"success={stats['success_count']:2d}  failure={stats['failure_count']:2d}  "
        f"acc={stats['accuracy']:.3f}  "
        f"success_max={stats['success_max']:.6f}  "
        f"failure_min={stats['failure_min']:.6f}"
    )
print("\nSubset summary for C (exploratory):")
for name, stats in subset_summary_c.items():
    print(
        f"  {name:15s} count={stats['count']:2d}  "
        f"success={stats['success_count']:2d}  failure={stats['failure_count']:2d}  "
        f"acc={stats['accuracy']:.3f}  "
        f"success_max={stats['success_max']:.6f}  "
        f"failure_min={stats['failure_min']:.6f}"
    )

np.savez(
    os.path.join(RESULTS_DIR, "criterion_evaluation.npz"),
    geom_names=primary_names,
    geom_types=primary_geom_types,
    labels=primary_labels,
    frac_sep=primary_frac_sep,
    mean_ratio=primary_mean_ratio,
    conv_intensity=primary_conv_intensity,
    ode_r2=primary_ode_r2,
    ode_r2_protocol=primary_ode_r2_protocol,
    canonical_hill_r2=np.array(canonical_hill_r2),
    canonical_hill_n=np.array(canonical_hill_n),
    canonical_hill_source=np.array(
        "results/periodic_hills_case_1p0_wall_profiles_corrected.npz"),
    canonical_hill_extraction=np.array("hill_surface_aware"),
    ode_result_source=np.array("results/core_ode_results_surface_aware.npz"),
    # PRIMARY metric: f_sep
    fsep_threshold=fsep_threshold,
    fsep_primary_accuracy=primary_accuracy_fsep,
    frac_sep_success_max=summary["frac_sep"]["primary_success_max"],
    frac_sep_failure_min=summary["frac_sep"]["primary_failure_min"],
    # SECONDARY (exploratory) metric: C = Rbar * f_sep
    c_threshold=c_threshold,
    c_secondary_accuracy=secondary_accuracy_c,
    mean_ratio_success_max=summary["mean_ratio"]["primary_success_max"],
    mean_ratio_failure_min=summary["mean_ratio"]["primary_failure_min"],
    conv_intensity_success_max=summary["conv_intensity"]["primary_success_max"],
    conv_intensity_failure_min=summary["conv_intensity"]["primary_failure_min"],
    primary_dataset_count=len(primary_names),
    primary_success_count=int((primary_labels == 1).sum()),
    primary_failure_count=int((primary_labels == 0).sum()),
    primary_accuracy_fsep_multi_station=primary_accuracy_fsep,
    secondary_accuracy_c_multi_station=secondary_accuracy_c,
    sanity_accuracy_fsep_all=sanity_accuracy,
    multi_station=np.ones(len(primary_names), dtype=bool),
    separated_multi=primary_frac_sep > 0.0,
    multi_station_count=subset_summary_fsep["multi_station"]["count"],
    multi_station_success_count=subset_summary_fsep["multi_station"]["success_count"],
    multi_station_failure_count=subset_summary_fsep["multi_station"]["failure_count"],
    multi_station_accuracy_fsep=subset_summary_fsep["multi_station"]["accuracy"],
    multi_station_success_max=subset_summary_fsep["multi_station"]["success_max"],
    multi_station_failure_min=subset_summary_fsep["multi_station"]["failure_min"],
    separated_multi_count=subset_summary_fsep["separated_multi"]["count"],
    separated_multi_success_count=subset_summary_fsep["separated_multi"]["success_count"],
    separated_multi_failure_count=subset_summary_fsep["separated_multi"]["failure_count"],
    separated_multi_accuracy_fsep=subset_summary_fsep["separated_multi"]["accuracy"],
    separated_multi_success_max=subset_summary_fsep["separated_multi"]["success_max"],
    separated_multi_failure_min=subset_summary_fsep["separated_multi"]["failure_min"],
    all_geom_names=all_geom_names,
    all_geom_types=all_geom_types,
    all_labels=all_labels,
    all_frac_sep=all_frac_sep,
    all_mean_ratio=all_mean_ratio,
    all_conv_intensity=all_conv_intensity,
    all_fsep_accuracy=sanity_accuracy,
    all_frac_sep_success_max=summary["frac_sep"]["all_success_max"],
    all_frac_sep_failure_min=summary["frac_sep"]["all_failure_min"],
    all_mean_ratio_success_max=summary["mean_ratio"]["all_success_max"],
    all_mean_ratio_failure_min=summary["mean_ratio"]["all_failure_min"],
    all_conv_intensity_success_max=summary["conv_intensity"]["all_success_max"],
    all_conv_intensity_failure_min=summary["conv_intensity"]["all_failure_min"],
)
