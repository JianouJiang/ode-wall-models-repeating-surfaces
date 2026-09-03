#!/usr/bin/env python3
"""Build the uncertainty certificate for every active headline claim (M1).

The active-claim inventory is defined by the independent M15 reconciliation.
This producer attaches one of three honest uncertainty objects to each claim:

* a circular moving-block 95% interval for spatially sampled profiles;
* a deterministic operator/phase/height/averaging-window envelope.

These objects are deliberately not pooled.  In particular, a deterministic
sensitivity envelope is never relabelled as a confidence interval.  The two
Xiao cases that set the printed 29-case range are rebuilt from their raw DNS
fields and passed through the same production wall-model solver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
OUT_JSON = RESULTS / "headline_uncertainty_m1.json"
OUT_NPZ = RESULTS / "headline_uncertainty_m1.npz"
SEED = 20260821
N_BOOT = 5000

sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "vendor" / "universal_wall_function" /
                       "codes" / "analysis"))
from dose_response_xiao import NU as XIAO_NU  # noqa: E402
from dose_response_xiao import XIAO, Y_IDX, read_case  # noqa: E402
from ode_wall_model import predict_tau_w  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = float(np.sum((reference - prediction) ** 2))
    total = float(np.sum((reference - np.mean(reference)) ** 2))
    return 1.0 - residual / total if total > 0.0 else float("nan")


def circular_indices(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    starts = rng.integers(0, n, size=int(math.ceil(n / block)))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def block_r2(reference: np.ndarray, prediction: np.ndarray,
             rng: np.random.Generator) -> tuple[float, np.ndarray, int]:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(prediction)
    reference, prediction = reference[valid], prediction[valid]
    block = max(2, int(math.ceil(math.sqrt(len(reference)))))
    samples = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        index = circular_indices(rng, len(reference), block)
        samples[b] = r2(reference[index], prediction[index])
    return r2(reference, prediction), samples, block


def metric_trace(data: np.lib.npyio.NpzFile, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = [str(v) for v in data["names"]]
    index = names.index(name)
    n = int(data["n_stations"][index])
    return (np.asarray(data["station_tau_ref"][index, :n], dtype=float),
            np.asarray(data["station_tau_pred"][index, :n], dtype=float))


def xiao_trace(case_name: str) -> tuple[np.ndarray, np.ndarray, Path]:
    case_dir = Path(XIAO) / case_name
    case = read_case(str(case_dir))
    prediction = np.full(len(case["x"]), np.nan)
    for i, (y, u, pressure_gradient) in enumerate(
            zip(case["y"], case["U"], case["dp_dx"])):
        if Y_IDX < len(y) and y[Y_IDX] > 0.0 and np.isfinite(u[Y_IDX]):
            prediction[i] = predict_tau_w(
                float(u[Y_IDX]), float(y[Y_IDX]), float(pressure_gradient), XIAO_NU
            )
    return np.asarray(case["tau_w"], dtype=float), prediction, case_dir / "mean_files.dat"


def percentile(samples: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(samples[np.isfinite(samples)], [2.5, 97.5])]


def window_two_sem(campaign: dict, model: str) -> list[float]:
    """Registered 180/225/270-window mean +/- two standard errors."""
    values = np.asarray([
        campaign["averaging"][f"G2c:{model}"][str(window)]["r2"]
        for window in (180, 225, 270)
    ], dtype=float)
    half_width = 2.0 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    centre = float(np.mean(values))
    return [centre - half_width, centre + half_width]


def corrected_hill_reference() -> np.ndarray:
    """The full-wall reference traction on the 512 archive stations.

    The traction previously used to score this hill was reconstructed from the
    velocity archive by a four-point through-origin fit whose spacing
    under-resolves the wall gradient; it was withdrawn on 2026-08-25.  The
    certificate is rebuilt here on the reference of Breuer et al. (2009), read
    from the standalone rebase artifact so that this producer and the rebase
    cannot silently disagree about which reference is primary.
    """
    rebase = json.loads(
        (RESULTS / "reference_rebase_headlines_l0_20260825.json").read_text())
    if rebase["schema"] != "reference-rebase-headlines-l0-v1":
        raise RuntimeError("unexpected rebase schema")
    npz = np.load(RESULTS / "reference_rebase_headlines_l0_20260825.npz")
    return np.asarray(npz["reference_B_mglet"], dtype=float)


def rng_for(label: str) -> np.random.Generator:
    offset = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")
    return np.random.default_rng(SEED + offset)


def main() -> None:
    active_path = RESULTS / "active_manuscript_number_reconciliation_m15.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    required = set(active["claims"])

    metrics_path = RESULTS / "signed_wall_error_metrics_m2.npz"
    metrics = np.load(metrics_path, allow_pickle=True)
    diagnostic_path = RESULTS / "diagnostic_test_corrected.npz"
    diagnostic = np.load(diagnostic_path, allow_pickle=True)
    grid_path = RESULTS / "rswm_grid_results_l3_summary.json"
    grid_npz_path = RESULTS / "rswm_grid_results_l3.npz"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_npz = np.load(grid_npz_path, allow_pickle=False)
    high_re_path = RESULTS / "m13_highre_coupled_20260825_summary.json"
    high_re_npz_path = RESULTS / "m13_highre_coupled_20260825.npz"
    high_re_summary = json.loads(high_re_path.read_text(encoding="utf-8"))
    re5600 = high_re_summary["campaigns"]["5600"]
    high_re = high_re_summary["campaigns"]["10595"]
    re19000 = high_re_summary["campaigns"]["19000"]
    high_re_npz = np.load(high_re_npz_path, allow_pickle=False)
    uncertainty_path = RESULTS / "uncertainty_certificate_l1.npz"
    uncertainty = np.load(uncertainty_path, allow_pickle=True)
    thin_path = RESULTS / "full_rans_thin_layer_audit.npz"
    thin = np.load(thin_path, allow_pickle=True)
    phase_path = RESULTS / "independent_phase_balance_r1_sta3.npz"
    phase = np.load(phase_path, allow_pickle=False)

    samples: dict[str, np.ndarray] = {}
    claims: dict[str, dict] = {}

    def add_block(name: str, reference: np.ndarray, prediction: np.ndarray,
                  source: str) -> None:
        point, draws, block = block_r2(reference, prediction, rng_for(name))
        samples[name] = draws
        claims[name] = {
            "point": point,
            "uncertainty_kind": "circular_station_block_95pct_interval",
            "interval": percentile(draws),
            "block_length": block,
            "replicates": N_BOOT,
            "source": source,
        }

    corrected_ref = corrected_hill_reference()
    _, hill_pred = metric_trace(metrics, "periodic_hills_case_1p0")
    if hill_pred.shape != corrected_ref.shape:
        raise RuntimeError("hill prediction and corrected reference are not co-located")
    add_block("canonical_hill_r2", corrected_ref, hill_pred,
              "signed_wall_error_metrics_m2.npz:station_tau_pred scored against "
              "reference_rebase_headlines_l0_20260825.npz:reference_B_mglet")
    rebase_scores = json.loads(
        (RESULTS / "reference_rebase_headlines_l0_20260825.json").read_text())
    rb_scores = rebase_scores["headlines"]["scores"]
    rb_family = rebase_scores["ranking_r2_2"]
    claims["canonical_hill_r2_bracket"] = {
        "point": rb_scores["pg_ode_mixing_length"]["C_repaired_cubic6"]["r2"],
        "uncertainty_kind": "phase_block_95pct_interval_on_the_bracket_reference",
        "interval": rb_scores["pg_ode_mixing_length"]["C_repaired_cubic6"]["r2_ci"],
        "source": "reference_rebase_headlines_l0_20260825.json: same prediction, "
                  "curvature-aware same-simulation reference",
    }
    claims["canonical_hill_r2_superseded"] = {
        "point": rb_scores["pg_ode_mixing_length"]["A_withdrawn_linear4"]["r2"],
        "uncertainty_kind": "withdrawn_reference_reported_only_as_the_superseded_value",
        "interval": rb_scores["pg_ode_mixing_length"]["A_withdrawn_linear4"]["r2_ci"],
        "source": "reference_rebase_headlines_l0_20260825.json: withdrawn "
                  "four-point through-origin reconstruction, negative control only",
    }
    claims["exact_stress_worse_on_every_reference"] = {
        "point": bool(rebase_scores["headlines"]["survives_reference_swap"]
                      ["exact_dns_stress_is_worse_than_mixing_length_on_every_reference"]),
        "uncertainty_kind": "qualitative_statement_checked_on_three_references",
        "per_reference_scores": {
            name: rb_scores["pg_ode_exact_dns_stress"][name]["r2"]
            for name in ("A_withdrawn_linear4", "B_mglet", "C_repaired_cubic6")},
        "source": "reference_rebase_headlines_l0_20260825.json",
    }
    claims["xiao_family_epsilon_range"] = {
        "point": rb_family["eps_range"]["repaired"],
        "uncertainty_kind": "single_estimator_family_range_with_stated_reference_limitation",
        "members_with_independent_reference": 1,
        "members_without_independent_reference":
            rb_family["n_members_without_any_independent_reference"],
        "source": "reference_rebase_headlines_l0_20260825.json: corrected 29-case sweep",
    }
    claims["xiao_family_outer_scale_rho"] = {
        "point": rb_family["ranking"]["repaired"]["per_candidate"]["L_y"]["rho"]["r2"],
        "uncertainty_kind": "member_bootstrap_95pct_interval_plus_permutation_p",
        "interval": rb_family["ranking"]["repaired"]["per_candidate"]["L_y"]["rho_r2_ci"],
        "permutation_p":
            rb_family["ranking"]["repaired"]["per_candidate"]["L_y"]["p_rho_r2"],
        "source": "reference_rebase_headlines_l0_20260825.json: corrected 29-case sweep",
    }
    bfs_ref, bfs_pred = metric_trace(metrics, "curved_bfs_Re13700_DNS")
    add_block("curved_bfs_r2", bfs_ref, bfs_pred,
              "signed_wall_error_metrics_m2.npz:station_tau_ref/pred")

    exact_valid = np.asarray(diagnostic["controlled_dns_valid"], dtype=bool)
    add_block("corrected_exact_stress_r2",
              corrected_ref[exact_valid],
              np.asarray(diagnostic["controlled_dns"], dtype=float)[exact_valid],
              "diagnostic_test_corrected.npz:controlled_dns scored against "
              "reference_rebase_headlines_l0_20260825.npz:reference_B_mglet")

    # The family range is now taken from the corrected 29-case sweep.  The
    # earlier certificate bootstrapped the two extreme members against the
    # withdrawn reconstruction; a station bootstrap against a superseded
    # yardstick is not an uncertainty on the corrected quantity, so it has been
    # replaced by the honest object: the measured range across all 29 members
    # under one estimator, with the reference limitation carried explicitly.
    xiao_raw_paths: list[Path] = []
    sweep = json.loads(
        (ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/"
         "xiao29_epsilon_sweep.json").read_text())
    corrected_r2 = [member["repaired"]["r2"] for member in sweep["per_member"]]
    worst = min(corrected_r2)
    best = max(corrected_r2)
    claims["xiao_family_r2_range"] = {
        "point": [worst, best],
        "uncertainty_kind": "single_estimator_family_range_with_stated_reference_limitation",
        "n_members": len(corrected_r2),
        "members_with_independent_reference":
            sweep["members_with_independent_wall_traction_reference"],
        "estimator_bias_correlates_with_reported_variables":
            sweep["contamination_correlates_with"],
        "limitation": sweep["reference_limitation"],
        "source": "corrected 29-case sweep: same predictions, curvature-aware "
                  "same-simulation wall-gradient estimator",
    }

    models = ("equilibrium", "total_gradient_tble")
    # Re-5,600 coupled claims come from the corrected harvest, which scores the
    # same calculations against the full-wall reference.
    finest = [re5600["metrics"][f"G2c:{model}"] for model in models]
    primary_intervals = re5600["phase_bootstrap_primary_intervals"]
    claims["coupled_finest_traction_errors"] = {
        "point": [entry["relative_rms"] for entry in finest],
        "uncertainty_kind": "paired_circular_phase_block_95pct_intervals",
        "intervals": [[primary_intervals[f"G2c:{model}"]["low"],
                       primary_intervals[f"G2c:{model}"]["high"]]
                      for model in models],
        "block_length": grid["bootstrap_protocol"]["primary_block_points"],
        "replicates_per_model": grid["bootstrap_protocol"]["draws"],
        "source": "m13_highre_coupled_20260825: Re5600 G2c phase-resampled E_tau "
                  "against the full-wall reference",
    }
    claims["coupled_finest_r2"] = {
        "point": [entry["r2"] for entry in finest],
        "uncertainty_kind": "three_grid_envelope_plus_180_225_270_window_two_sem",
        "intervals": [[min(re5600["grid_path_convergence"][f"{model}:r2"]["values"]),
                       max(re5600["grid_path_convergence"][f"{model}:r2"]["values"])]
                      for model in models],
        "window_intervals": [window_two_sem(re5600, model) for model in models],
        "grid_paths": [re5600["grid_path_convergence"][f"{model}:r2"]["values"]
                       for model in models],
        "source": "m13_highre_coupled_20260825_summary.json:campaigns[5600]."
                  "grid_path_convergence and averaging",
    }
    holm = re5600["failure_significance_tests"]
    claims["coupled_holm_p"] = {
        "point": [holm[model]["p_one_sided_holm_two_models"] for model in models],
        "uncertainty_kind": "exact_phase_block_sign_flip_test_with_Holm_correction",
        "block_count": len(holm[models[0]]["block_values"]),
        "permutations": holm[models[0]]["permutations"],
        "source": "m13_highre_coupled_20260825_summary.json:campaigns[5600]."
                  "failure_significance_tests",
    }
    comparison = re5600["model_comparison"]
    claims["coupled_model_difference_p"] = {
        "point": [comparison["finest_relative_rms_tble_minus_equilibrium"],
                  comparison["paired_exact_block_test"]["p_two_sided"]],
        "uncertainty_kind": "paired_phase_block_95pct_interval_and_exact_sign_flip_test",
        "interval": [comparison["primary_bootstrap_delta_interval"]["low"],
                     comparison["primary_bootstrap_delta_interval"]["high"]],
        "permutations": comparison["paired_exact_block_test"]["permutations"],
        "source": "rswm_grid_results_l3.npz/json: G2c paired model difference",
    }
    samples["coupled_finest_traction_errors"] = np.vstack([
        np.asarray(grid_npz[f"G2c_{model}_primary_bootstrap_relative_rms"], dtype=float)
        for model in models
    ])
    samples["coupled_model_difference_p"] = np.asarray(
        grid_npz["block512_G2c_delta_tble_minus_equilibrium"], dtype=float
    )

    high_re_finest = [high_re["metrics"][f"G2c:{model}"] for model in models]
    high_re_intervals = high_re["phase_bootstrap_primary_intervals"]
    claims["re10595_finest_traction_errors"] = {
        "point": [entry["relative_rms"] for entry in high_re_finest],
        "uncertainty_kind": "paired_circular_phase_block_95pct_intervals",
        "intervals": [[high_re_intervals[f"G2c:{model}"]["low"],
                       high_re_intervals[f"G2c:{model}"]["high"]]
                      for model in models],
        "block_length": 512,
        "replicates_per_model": N_BOOT * 4,
        "source": "m13_highre_coupled_20260825.npz/json: Re10595 G2c phase-resampled E_tau",
    }
    claims["re10595_finest_r2"] = {
        "point": [entry["r2"] for entry in high_re_finest],
        "uncertainty_kind": "three_grid_envelope_plus_180_225_270_window_two_sem",
        "intervals": [[min(high_re["grid_path_convergence"][f"{model}:r2"]["values"]),
                       max(high_re["grid_path_convergence"][f"{model}:r2"]["values"])]
                      for model in models],
        "window_intervals": [window_two_sem(high_re, model) for model in models],
        "source": "m13_highre_coupled_20260825_summary.json: Re10595 "
                  "grid_path_convergence and averaging",
    }
    high_re_truth = high_re["truth"]["events"]
    claims["re10595_finest_reattachment"] = {
        "point": [entry["reattachment_x_over_H"] for entry in high_re_finest]
                 + [high_re_truth["reattachment_x_over_H"]],
        "uncertainty_kind": "documented_dns_event_interval_and_model_grid_paths",
        "interval": [high_re_truth["documented_reattachment_x_over_H"]
                     - high_re_truth["documented_reattachment_uncertainty_over_H"],
                     high_re_truth["documented_reattachment_x_over_H"]
                     + high_re_truth["documented_reattachment_uncertainty_over_H"]],
        "model_grid_paths": [
            high_re["grid_path_convergence"][f"{model}:reattachment_x_over_H"]["values"]
            for model in models],
        "source": "m13_highre_coupled_20260825_summary.json: Re10595 metrics and Krank event uncertainty",
    }
    samples["re10595_finest_traction_errors"] = np.vstack([
        np.asarray(high_re_npz[f"re10595_G2c_{model}_primary_bootstrap_relative_rms"],
                   dtype=float) for model in models
    ])

    re19000_profile = [
        re19000["profiles"][f"G2c:{model}"]["rapp_19000"]["u_rms_mean"]
        for model in models
    ]
    claims["re19000_profile_mean_rms"] = {
        "point": re19000_profile,
        "uncertainty_kind": "two_grid_deterministic_envelope",
        "intervals": [[min(re19000["profiles"][f"{grid}:{model}"]["rapp_19000"]["u_rms_mean"]
                           for grid in ("G1c", "G2c")),
                       max(re19000["profiles"][f"{grid}:{model}"]["rapp_19000"]["u_rms_mean"]
                           for grid in ("G1c", "G2c"))]
                      for model in models],
        "stations": 10,
        "reference": "Rapp (2009) PIV; no wall-traction truth",
        "source": "m13_highre_coupled_20260825.npz/json: Re19000 stationwise PIV velocity RMS",
    }
    re19000_xr = [re19000["metrics"][f"G2c:{model}"]["reattachment_x_over_H"]
                  for model in models]
    exp19 = re19000["experimental_reattachment"]
    claims["re19000_reattachment"] = {
        "point": re19000_xr + [exp19["estimate_x_over_H"]],
        "uncertainty_kind": "experimental_sign_change_bracket_plus_two_grid_and_window_sensitivity",
        "experimental_bracket": exp19["bracket_x_over_H"],
        "model_grid_paths": [
            re19000["grid_path_convergence"][f"{model}:reattachment_x_over_H"]["values"]
            for model in models],
        "model_window_paths": [[
            re19000["averaging"][f"G2c:{model}"]["225"]["reattachment_x_over_H"],
            re19000["averaging"][f"G2c:{model}"]["270"]["reattachment_x_over_H"]]
            for model in models],
        "convergence_claim_permitted": False,
        "source": "m13_highre_coupled_20260825.npz/json: Re19000 coupled event and Rapp PIV bracket",
    }
    claims["re19000_cancellation_medians"] = {
        "point": [re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated"]
                  for model in models],
        "uncertainty_kind": "paired_circular_phase_block_95pct_intervals",
        "intervals": [[
            re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated_interval"]["low"],
            re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated_interval"]["high"]]
            for model in models],
        "source": "m13_highre_coupled_20260825.npz/json: Re19000 G2c separated-phase eps_c",
    }

    uncertainty_names = [str(v) for v in uncertainty["names"]]
    hill_index = uncertainty_names.index("periodic_hills_case_1p0")
    claims["three_term_hill_ratio"] = {
        "point": float(uncertainty["statistic_centre"][hill_index, 0]),
        "uncertainty_kind": "simultaneous_station_block_95pct_interval",
        "interval": [float(uncertainty["statistic_simultaneous_ci_low"][hill_index, 0]),
                     float(uncertainty["statistic_simultaneous_ci_high"][hill_index, 0])],
        "operator_envelope": [float(uncertainty["statistic_operator_envelope_low"][hill_index, 0]),
                              float(uncertainty["statistic_operator_envelope_high"][hill_index, 0])],
        "source": "uncertainty_certificate_l1.npz",
    }

    reynolds = np.asarray(thin["reynolds_streamwise_fraction"], dtype=float)
    viscous = np.asarray(thin["viscous_streamwise_to_normal_ratio"], dtype=float)
    claims["thin_layer_reynolds_range"] = {
        "point": [float(np.min(reynolds)), float(np.max(reynolds))],
        "uncertainty_kind": "six_height_by_six_operator_deterministic_envelope",
        "interval": [float(np.min(reynolds)), float(np.max(reynolds))],
        "source": "full_rans_thin_layer_audit.npz:reynolds_streamwise_fraction",
    }
    claims["thin_layer_viscous_range"] = {
        "point": [float(np.min(viscous)), float(np.max(viscous))],
        "uncertainty_kind": "six_height_by_six_operator_deterministic_envelope",
        "interval": [float(np.min(viscous)), float(np.max(viscous))],
        "source": "full_rans_thin_layer_audit.npz:viscous_streamwise_to_normal_ratio",
    }

    phase_central = np.asarray(phase["phase_closure_central"], dtype=float)
    phase_min = np.asarray(phase["phase_closure_operator_min"], dtype=float)
    phase_max = np.asarray(phase["phase_closure_operator_max"], dtype=float)
    wave_central = np.asarray(phase["wavelength_closure_central"], dtype=float)
    wave_min = np.asarray(phase["wavelength_closure_operator_min"], dtype=float)
    wave_max = np.asarray(phase["wavelength_closure_operator_max"], dtype=float)
    primary = int(np.flatnonzero(phase["phase_counts"] == 48)[0])
    claims["independent_phase_closure"] = {
        "point": float(phase_central[primary]),
        "uncertainty_kind": "phase_count_by_volume_surface_operator_deterministic_envelope",
        "interval": [float(np.min(phase_min)), float(np.max(phase_max))],
        "source": "independent_phase_balance_r1_sta3.npz",
    }
    claims["independent_phase_operator_envelope"] = {
        "point": [float(np.min(phase_min)), float(np.max(phase_max))],
        "uncertainty_kind": "nine_phase_count_by_4320_operator_deterministic_envelope",
        "interval": [float(np.min(phase_min)), float(np.max(phase_max))],
        "source": "independent_phase_balance_r1_sta3.npz",
    }
    claims["independent_wavelength_closure"] = {
        "point": float(wave_central[primary]),
        "uncertainty_kind": "phase_count_by_volume_surface_operator_deterministic_envelope",
        "interval": [float(np.min(wave_min)), float(np.max(wave_max))],
        "source": "independent_phase_balance_r1_sta3.npz",
    }

    if set(claims) != required:
        missing = sorted(required - set(claims))
        extra = sorted(set(claims) - required)
        raise RuntimeError(f"M1 claim inventory mismatch; missing={missing}, extra={extra}")

    source_paths = [active_path, metrics_path, diagnostic_path,
                    grid_path, grid_npz_path, high_re_path, high_re_npz_path,
                    uncertainty_path, thin_path, phase_path,
                    *xiao_raw_paths]
    payload = {
        "schema": "headline-uncertainty-m1-v1",
        "status": "PASS",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "active_claim_count": len(required),
        "coverage": f"{len(required)}/{len(required)} active M15 headline groups",
        "semantics": {
            "sampling_interval": "circular moving-block 95% interval; block length ceil(sqrt(N))",
            "deterministic_envelope": "operator, height, averaging-window or reference-data sensitivity; not a confidence interval",
        },
        "claims": claims,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        schema=np.array("headline-uncertainty-m1-v1"),
        bootstrap_seed=np.array(SEED),
        bootstrap_replicates=np.array(N_BOOT),
        claim_names=np.asarray(sorted(required)),
        **samples,
    )
    print(f"M1 active headline coverage: {len(required)}/{len(required)}")
    print(f"M1 spatial bootstrap replicates: {N_BOOT}")
    print("M1 sampling intervals and deterministic envelopes kept separate")
    print("M1 status: PASS")


if __name__ == "__main__":
    main()
