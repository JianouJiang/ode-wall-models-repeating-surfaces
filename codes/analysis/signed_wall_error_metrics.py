#!/usr/bin/env python3
"""Physical error operator for sign-changing wall stress.

The legacy manuscript used :math:`R^2` both as a descriptive score and as the
pass/fail definition.  That is unsafe when the reference wall stress changes
sign: an arbitrarily small variance makes ``R2`` arbitrarily negative, while a
large ``R2`` does not certify the separation topology.  This module freezes a
model-independent metric operator before the new coupled calculations.

The operator is evaluated on every multi-station case in the canonical core
and extended manifests.  Predictions are rebuilt with the deposited ODE
solver at matching index 10; saved headline scores are used only as guards.

Outputs
-------
``codes/results/signed_wall_error_metrics_m2.npz``
    Machine-readable case table and padded stationwise arrays.
``codes/results/signed_wall_error_metrics_m2.summary.json``
    Human-readable table with source hashes and metric definitions.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CODES = ROOT / "codes"
RESULTS = CODES / "results"
VENDOR_RESULTS = CODES / "vendor/universal_wall_function/codes/results"
OUT = RESULTS / "signed_wall_error_metrics_m2.npz"
SUMMARY = RESULTS / "signed_wall_error_metrics_m2.summary.json"
Y_IDX = 10
R2_SUCCESS = 0.88
N_BOOT = 2000
BOOT_SEED = 20260821

sys.path.insert(0, str(CODES))
import manifest  # noqa: E402

sys.path.insert(0, str(CODES / "vendor/universal_wall_function/codes/analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def savez_deterministic(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write an NPZ whose bytes do not depend on the execution time.

    ``numpy.savez`` records the current timestamp in every ZIP member, which
    defeats byte-level reproduction even when all arrays are identical.  A
    fixed member timestamp and sorted keys make the deposited product stable.
    """
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(payload):
            stream = io.BytesIO()
            np.save(stream, np.asarray(payload[key]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, stream.getvalue())


def scalar_at(value: np.ndarray, i: int) -> float:
    array = np.asarray(value, dtype=float)
    return float(array if array.ndim == 0 else array[i])


def profile_path(name: str, extended_record: tuple[str, str, str, str] | None) -> Path:
    if name == "periodic_hills_case_1p0":
        return RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
    if extended_record is not None:
        return CODES / extended_record[2]
    return VENDOR_RESULTS / f"{name}_wall_profiles.npz"


def rebuild_case(path: Path) -> dict[str, np.ndarray]:
    """Rebuild the deposited pressure-gradient ODE prediction at ``Y_IDX``."""
    with np.load(path, allow_pickle=False) as data:
        required = ("x", "y", "U", "tau_w", "dp_dx", "nu")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        velocity = np.asarray(data["U"], dtype=float)
        truth = np.asarray(data["tau_w"], dtype=float)
        pressure = np.asarray(data["dp_dx"], dtype=float)
        nu = np.asarray(data["nu"], dtype=float)

    if velocity.ndim != 2 or truth.ndim != 1 or truth.size != velocity.shape[0]:
        raise ValueError(f"{path}: unsupported profile schema")
    if x.size != truth.size:
        raise ValueError(f"{path}: x/tau_w length mismatch")

    prediction = np.full(truth.size, np.nan)
    edge_velocity = np.nanmax(np.abs(velocity), axis=1)
    for i in range(truth.size):
        yi = y[i] if y.ndim == 2 else y
        if Y_IDX >= yi.size or Y_IDX >= velocity.shape[1]:
            continue
        ym = float(yi[Y_IDX])
        um = float(velocity[i, Y_IDX])
        if not (np.isfinite(ym) and ym > 0.0 and np.isfinite(um)):
            continue
        prediction[i] = predict_tau_w(
            um, ym, scalar_at(pressure, i), scalar_at(nu, i))

    valid = (np.isfinite(x) & np.isfinite(truth) & np.isfinite(prediction)
             & np.isfinite(edge_velocity) & (edge_velocity > 0.0))
    order = np.argsort(x[valid], kind="mergesort")
    result = {
        "x": x[valid][order],
        "tau_ref": truth[valid][order],
        "tau_pred": prediction[valid][order],
        "u_edge": edge_velocity[valid][order],
    }
    if result["x"].size < 4 or np.any(np.diff(result["x"]) <= 0.0):
        raise ValueError(f"{path}: need at least four strictly ordered stations")
    return result


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Positive open-interval trapezoidal weights, normalised to sum to one."""
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    if np.any(weights <= 0.0):
        raise ValueError("non-positive station quadrature weight")
    return weights / np.sum(weights)


def r2_score(reference: np.ndarray, prediction: np.ndarray) -> float:
    variance = np.sum((reference - np.mean(reference)) ** 2)
    return (float(1.0 - np.sum((prediction - reference) ** 2) / variance)
            if variance > 0.0 else math.nan)


def negative_intervals(x: np.ndarray, values: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Piecewise-linear negative components with endpoint bracket widths.

    Each tuple is ``(left_root, right_root, left_bracket, right_bracket)``.
    Components touching the sampled boundary are retained with zero endpoint
    bracket so that a missing external crossing is never invented.
    """
    negative = values < 0.0
    if not np.any(negative):
        return []
    components: list[tuple[float, float, float, float]] = []
    starts = np.flatnonzero(negative & np.r_[True, ~negative[:-1]])
    ends = np.flatnonzero(negative & np.r_[~negative[1:], True])
    for start, end in zip(starts, ends):
        if start == 0:
            left, left_bracket = float(x[0]), 0.0
        else:
            x0, x1 = float(x[start - 1]), float(x[start])
            y0, y1 = float(values[start - 1]), float(values[start])
            left = x0 - y0 * (x1 - x0) / (y1 - y0)
            left_bracket = x1 - x0
        if end == x.size - 1:
            right, right_bracket = float(x[-1]), 0.0
        else:
            x0, x1 = float(x[end]), float(x[end + 1])
            y0, y1 = float(values[end]), float(values[end + 1])
            right = x0 - y0 * (x1 - x0) / (y1 - y0)
            right_bracket = x1 - x0
        components.append((left, right, left_bracket, right_bracket))
    return components


def overlap(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def topology_metrics(x: np.ndarray, reference: np.ndarray,
                     prediction: np.ndarray) -> dict[str, float | int | bool]:
    """Compare the largest reference reversed-shear interval without pairing leakage."""
    span = float(x[-1] - x[0])
    ref_intervals = negative_intervals(x, reference)
    pred_intervals = negative_intervals(x, prediction)
    dense_x = np.linspace(x[0], x[-1], max(4097, 16 * x.size + 1))
    ref_negative = np.interp(dense_x, x, reference) < 0.0
    pred_negative = np.interp(dense_x, x, prediction) < 0.0
    symmetric_difference = float(np.mean(ref_negative != pred_negative))

    out: dict[str, float | int | bool] = {
        "n_ref_components": len(ref_intervals),
        "n_pred_components": len(pred_intervals),
        "separated_set_symmetric_difference": symmetric_difference,
        "reference_has_separation": bool(ref_intervals),
        "event_missed": False,
        "x_separation_ref": math.nan,
        "x_reattachment_ref": math.nan,
        "x_separation_pred": math.nan,
        "x_reattachment_pred": math.nan,
        "separation_error_over_span": math.nan,
        "reattachment_error_over_span": math.nan,
        "separation_resolution_uncertainty_over_span": math.nan,
        "reattachment_resolution_uncertainty_over_span": math.nan,
    }
    if not ref_intervals:
        return out

    ref = max(ref_intervals, key=lambda interval: interval[1] - interval[0])
    out["x_separation_ref"], out["x_reattachment_ref"] = ref[0], ref[1]
    if not pred_intervals:
        out["event_missed"] = True
        return out

    # The predicted component is paired solely by geometric overlap with the
    # registered reference event; no stress amplitude or target error is used.
    pred = max(pred_intervals, key=lambda interval: (overlap(ref, interval),
                                                     interval[1] - interval[0]))
    if overlap(ref, pred) <= 0.0:
        out["event_missed"] = True
        return out
    out["x_separation_pred"], out["x_reattachment_pred"] = pred[0], pred[1]
    out["separation_error_over_span"] = (pred[0] - ref[0]) / span
    out["reattachment_error_over_span"] = (pred[1] - ref[1]) / span
    out["separation_resolution_uncertainty_over_span"] = (
        0.5 * math.hypot(ref[2], pred[2]) / span)
    out["reattachment_resolution_uncertainty_over_span"] = (
        0.5 * math.hypot(ref[3], pred[3]) / span)
    return out


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(math.ceil(n / block)))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def bootstrap_intervals(reference: np.ndarray, prediction: np.ndarray,
                        weights: np.ndarray, scale: float,
                        seed: int) -> dict[str, tuple[float, float]]:
    """Circular moving-block intervals for distribution and drag metrics."""
    rng = np.random.default_rng(seed)
    n = reference.size
    block = max(3, int(math.ceil(math.sqrt(n))))
    draws = np.empty((N_BOOT, 4))
    for ib in range(N_BOOT):
        index = circular_block_indices(n, block, rng)
        ref, pred = reference[index], prediction[index]
        ww = weights[index]
        ww = ww / np.sum(ww)
        absolute = np.abs(pred - ref) / scale
        denominator = np.sum(ww * np.abs(ref))
        draws[ib] = (
            np.median(absolute),
            np.percentile(absolute, 95),
            np.sum(ww * (pred - ref)) / denominator,
            np.sum(ww * (np.signbit(pred) != np.signbit(ref))),
        )
    quantiles = np.percentile(draws, [2.5, 97.5], axis=0)
    names = ("station_abs_p50", "station_abs_p95", "viscous_drag_signed_error",
             "station_sign_mismatch_fraction")
    return {name: (float(quantiles[0, i]), float(quantiles[1, i]))
            for i, name in enumerate(names)}


def evaluate_metrics(case: dict[str, np.ndarray], seed: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x, truth, prediction, u_edge = (case[key] for key in
                                    ("x", "tau_ref", "tau_pred", "u_edge"))
    weights = trapezoid_weights(x)
    tau_scale = float(np.sqrt(np.sum(weights * truth ** 2)))
    if not np.isfinite(tau_scale) or tau_scale <= 0.0:
        raise ValueError("reference wall-stress RMS must be positive")
    cf_ref = 2.0 * truth / u_edge ** 2
    cf_pred = 2.0 * prediction / u_edge ** 2
    cf_scale = float(np.sqrt(np.sum(weights * cf_ref ** 2)))
    signed_station = (prediction - truth) / tau_scale
    signed_cf_station = (cf_pred - cf_ref) / cf_scale
    abs_station = np.abs(signed_station)
    abs_cf_station = np.abs(signed_cf_station)
    drag_denominator = float(np.sum(weights * np.abs(truth)))
    topology = topology_metrics(x, truth, prediction)
    intervals = bootstrap_intervals(truth, prediction, weights, tau_scale, seed)

    metrics: dict[str, object] = {
        "n_stations": int(x.size),
        "tau_rms_scale": tau_scale,
        "cf_rms_scale": cf_scale,
        "r2_descriptive": r2_score(truth, prediction),
        "relrms_tau": float(np.sqrt(np.sum(weights * (prediction - truth) ** 2)) / tau_scale),
        "station_signed_median": float(np.median(signed_station)),
        "station_abs_p50": float(np.median(abs_station)),
        "station_abs_p95": float(np.percentile(abs_station, 95)),
        "station_abs_max": float(np.max(abs_station)),
        "cf_station_abs_p50": float(np.median(abs_cf_station)),
        "cf_station_abs_p95": float(np.percentile(abs_cf_station, 95)),
        "cf_station_abs_max": float(np.max(abs_cf_station)),
        "viscous_drag_ref": float(np.sum(weights * truth)),
        "viscous_drag_pred": float(np.sum(weights * prediction)),
        "viscous_drag_signed_error": float(np.sum(weights * (prediction - truth)) /
                                                   drag_denominator),
        "viscous_drag_absolute_error": float(abs(np.sum(weights * (prediction - truth))) /
                                                     drag_denominator),
        "station_sign_mismatch_fraction": float(np.sum(weights *
                                                 (np.signbit(prediction) != np.signbit(truth)))),
        **topology,
    }
    for name, (lo, hi) in intervals.items():
        metrics[f"{name}_ci_lo"] = lo
        metrics[f"{name}_ci_hi"] = hi
    station = {
        "x": x,
        "tau_ref": truth,
        "tau_pred": prediction,
        "cf_ref": cf_ref,
        "cf_pred": cf_pred,
        "signed_station_error": signed_station,
        "quadrature_weight": weights,
    }
    return metrics, station


def case_records() -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for name in manifest.core_multistation_names():
        records.append((name, manifest.core_geom_type(name), profile_path(name, None)))
    for record in manifest.EXTENDED_DATASETS:
        name, family = record[0], record[1]
        records.append((name, family, profile_path(name, record)))
    return records


def main() -> int:
    rows: list[dict[str, object]] = []
    stations: list[dict[str, np.ndarray]] = []
    for i, (name, family, path) in enumerate(case_records()):
        if not path.is_file():
            raise FileNotFoundError(path)
        case = rebuild_case(path)
        metrics, station = evaluate_metrics(case, BOOT_SEED + i)
        row = {
            "name": name,
            "family": family,
            "profile_path": str(path.relative_to(ROOT)),
            "profile_sha256": sha256(path),
            **metrics,
        }
        rows.append(row)
        stations.append(station)
        print(f"{name:34s} n={metrics['n_stations']:4d} "
              f"R2={metrics['r2_descriptive']:+9.3f} "
              f"p95={metrics['station_abs_p95']:.3f} "
              f"drag={metrics['viscous_drag_signed_error']:+.3f} "
              f"set={metrics['separated_set_symmetric_difference']:.3f}")

    max_n = max(int(row["n_stations"]) for row in rows)
    station_keys = tuple(stations[0])
    padded = {key: np.full((len(rows), max_n), np.nan) for key in station_keys}
    for i, station in enumerate(stations):
        n = station["x"].size
        for key in station_keys:
            padded[key][i, :n] = station[key]

    scalar_keys = [key for key in rows[0]
                   if key not in ("name", "family", "profile_path", "profile_sha256")]
    payload: dict[str, np.ndarray] = {
        "schema": np.array("signed-wall-error-operator-v1"),
        "matching_index": np.array(Y_IDX),
        "r2_success_reference": np.array(R2_SUCCESS),
        "bootstrap_replicates": np.array(N_BOOT),
        "bootstrap_seed": np.array(BOOT_SEED),
        "names": np.array([row["name"] for row in rows]),
        "families": np.array([row["family"] for row in rows]),
        "profile_paths": np.array([row["profile_path"] for row in rows]),
        "profile_sha256": np.array([row["profile_sha256"] for row in rows]),
    }
    for key in scalar_keys:
        sample = rows[0][key]
        if isinstance(sample, bool):
            payload[key] = np.array([bool(row[key]) for row in rows], dtype=bool)
        elif isinstance(sample, int):
            payload[key] = np.array([int(row[key]) for row in rows], dtype=int)
        else:
            payload[key] = np.array([float(row[key]) for row in rows], dtype=float)
    payload.update({f"station_{key}": value for key, value in padded.items()})
    savez_deterministic(OUT, payload)

    bfs = next(row for row in rows if row["name"] == "bfs_Re13700")
    hill = next(row for row in rows if row["name"] == "periodic_hills_case_1p0")
    summary = {
        "schema": "signed-wall-error-operator-v1",
        "idea": ("Replace variance-only scoring by a sign-topology-aware physical metric "
                 "operator for wall stress."),
        "metric_definitions": {
            "station_error": "(tau_model-tau_ref)/RMS_x(tau_ref)",
            "cf_station_error": "(Cf_model-Cf_ref)/RMS_x(Cf_ref), Cf=2 tau/Ue^2",
            "viscous_drag_signed_error": ("integral(tau_model-tau_ref) dx / "
                                           "integral(abs(tau_ref)) dx"),
            "separated_set_symmetric_difference": ("measure({tau_model<0} symmetric-difference "
                                                    "{tau_ref<0}) / sampled span"),
            "event_location": ("piecewise-linear zero crossings; largest reference component "
                               "paired to prediction by geometric overlap only"),
            "r2_descriptive": "reported as a descriptive column; never used alone as adequacy",
            "intervals": ("95% circular moving-block percentile intervals; block length "
                          "ceil(sqrt(n)), 2000 replicates"),
        },
        "n_cases": len(rows),
        "case_scope": "all multi-station cases in CORE_DATASETS and EXTENDED_DATASETS",
        "bfs_r2": bfs["r2_descriptive"],
        "bfs_margin_above_0p88": float(bfs["r2_descriptive"]) - R2_SUCCESS,
        "bfs_station_abs_p50_p95_max": [bfs["station_abs_p50"],
                                            bfs["station_abs_p95"],
                                            bfs["station_abs_max"]],
        "canonical_hill_r2": hill["r2_descriptive"],
        "canonical_hill_station_abs_p50_p95_max": [hill["station_abs_p50"],
                                                       hill["station_abs_p95"],
                                                       hill["station_abs_max"]],
        "rows": rows,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT.relative_to(ROOT)}")
    print(f"Saved {SUMMARY.relative_to(ROOT)}")
    print(f"BFS R2 margin above 0.88 = {summary['bfs_margin_above_0p88']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
