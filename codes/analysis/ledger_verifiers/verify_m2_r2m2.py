#!/usr/bin/env python3
"""Independent verifier for the sign-changing wall-stress metric operator.

Closes claim M2 and R2-m2 only if the deposited table can be rebuilt
from its profile sources, the station distributions cover the entire declared
multi-station corpus, and negative fixtures exercise singular and topological
failure modes.  This verifier deliberately does not import the producer.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "codes"
RESULT = CODES / "results/signed_wall_error_metrics_m2.npz"
SUMMARY = CODES / "results/signed_wall_error_metrics_m2.summary.json"
LEDGER = ROOT / "REFEREE_POINT_LEDGER.md"
MAIN = ROOT / "manuscript/main.tex"
Y_IDX = 10

sys.path.insert(0, str(CODES))
import manifest  # noqa: E402

sys.path.insert(0, str(CODES / "vendor/universal_wall_function/codes/analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402


checks: list[tuple[str, bool]] = []


def check(label: str, condition: bool) -> None:
    checks.append((label, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    result[0] = 0.5 * (x[1] - x[0])
    result[-1] = 0.5 * (x[-1] - x[-2])
    result[1:-1] = 0.5 * (x[2:] - x[:-2])
    return result / np.sum(result)


def r2(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((reference - np.mean(reference)) ** 2)
    if denominator <= 0.0:
        raise ValueError("R2 is undefined for a zero-variance reference")
    return float(1.0 - np.sum((prediction - reference) ** 2) / denominator)


def metrics(x: np.ndarray, reference: np.ndarray,
            prediction: np.ndarray) -> dict[str, float]:
    ww = weights(x)
    scale = float(np.sqrt(np.sum(ww * reference ** 2)))
    if scale <= 0.0:
        raise ValueError("reference RMS must be positive")
    error = (prediction - reference) / scale
    drag_denom = np.sum(ww * np.abs(reference))
    return {
        "r2_descriptive": r2(reference, prediction),
        "relrms_tau": float(np.sqrt(np.sum(ww * (prediction - reference) ** 2)) / scale),
        "station_signed_median": float(np.median(error)),
        "station_abs_p50": float(np.median(np.abs(error))),
        "station_abs_p95": float(np.percentile(np.abs(error), 95)),
        "station_abs_max": float(np.max(np.abs(error))),
        "viscous_drag_signed_error": float(np.sum(ww * (prediction - reference)) / drag_denom),
        "station_sign_mismatch_fraction": float(np.sum(ww *
                                              (np.signbit(prediction) != np.signbit(reference)))),
    }


def intervals(x: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    negative = values < 0.0
    starts = np.flatnonzero(negative & np.r_[True, ~negative[:-1]])
    ends = np.flatnonzero(negative & np.r_[~negative[1:], True])
    result: list[tuple[float, float]] = []
    for start, end in zip(starts, ends):
        left = float(x[0]) if start == 0 else float(
            x[start - 1] - values[start - 1] * (x[start] - x[start - 1]) /
            (values[start] - values[start - 1]))
        right = float(x[-1]) if end == x.size - 1 else float(
            x[end] - values[end] * (x[end + 1] - x[end]) /
            (values[end + 1] - values[end]))
        result.append((left, right))
    return result


def topology(x: np.ndarray, reference: np.ndarray,
             prediction: np.ndarray) -> tuple[float, float, float, bool]:
    dense = np.linspace(x[0], x[-1], max(4097, 16 * x.size + 1))
    set_error = float(np.mean((np.interp(dense, x, reference) < 0.0) !=
                              (np.interp(dense, x, prediction) < 0.0)))
    ref = intervals(x, reference)
    pred = intervals(x, prediction)
    if not ref:
        return set_error, math.nan, math.nan, False
    main = max(ref, key=lambda item: item[1] - item[0])
    if not pred:
        return set_error, math.nan, math.nan, True
    paired = max(pred, key=lambda item: (max(0.0, min(main[1], item[1]) -
                                                   max(main[0], item[0])),
                                         item[1] - item[0]))
    overlap = max(0.0, min(main[1], paired[1]) - max(main[0], paired[0]))
    if overlap <= 0.0:
        return set_error, math.nan, math.nan, True
    span = x[-1] - x[0]
    return set_error, (paired[0] - main[0]) / span, (paired[1] - main[1]) / span, False


def scalar_at(value: np.ndarray, index: int) -> float:
    array = np.asarray(value, dtype=float)
    return float(array if array.ndim == 0 else array[index])


def rebuild_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], float)
        y = np.asarray(data["y"], float)
        velocity = np.asarray(data["U"], float)
        reference = np.asarray(data["tau_w"], float)
        pressure = np.asarray(data["dp_dx"], float)
        nu = np.asarray(data["nu"], float)
    prediction = np.full(reference.size, np.nan)
    # Surface-aware hill columns contain NaN inside the solid.  The edge
    # velocity is the maximum over fluid samples, exactly as the documented
    # profile protocol requires.
    u_edge = np.nanmax(np.abs(velocity), axis=1)
    for i in range(reference.size):
        yi = y[i] if y.ndim == 2 else y
        if Y_IDX < yi.size and yi[Y_IDX] > 0.0:
            prediction[i] = predict_tau_w(float(velocity[i, Y_IDX]), float(yi[Y_IDX]),
                                          scalar_at(pressure, i), scalar_at(nu, i))
    valid = np.isfinite(x + reference + prediction + u_edge) & (u_edge > 0.0)
    order = np.argsort(x[valid], kind="mergesort")
    return x[valid][order], reference[valid][order], prediction[valid][order], u_edge[valid][order]


def active_tex(text: str) -> str:
    kept: list[str] = []
    depth = 0
    for line in text.splitlines():
        token = line.split("%", 1)[0].strip()
        if token.startswith(r"\iffalse"):
            depth += 1
        elif token.startswith(r"\fi") and depth:
            depth -= 1
        elif depth == 0:
            kept.append(line)
    if depth:
        raise RuntimeError("unclosed iffalse block")
    return "\n".join(kept)


with np.load(RESULT, allow_pickle=False) as saved:
    names = saved["names"].astype(str)
    expected = manifest.core_multistation_names() + manifest.extended_names()
    check("all declared multi-station cases are present once",
          names.tolist() == expected and len(set(names)) == 18)
    check("operator schema and matching height are frozen",
          str(saved["schema"]) == "signed-wall-error-operator-v1" and
          int(saved["matching_index"]) == 10)

    # Every source is byte-addressed and every deposited station distribution
    # has exactly the registered valid length.
    source_ok = True
    station_ok = True
    formula_ok = True
    topology_ok = True
    for i, name in enumerate(names):
        path = ROOT / str(saved["profile_paths"][i])
        source_ok &= path.is_file() and sha256(path) == str(saved["profile_sha256"][i])
        n = int(saved["n_stations"][i])
        x = saved["station_x"][i, :n]
        ref = saved["station_tau_ref"][i, :n]
        pred = saved["station_tau_pred"][i, :n]
        station_ok &= (n >= 10 and np.all(np.isfinite(x + ref + pred)) and
                       np.all(np.diff(x) > 0.0) and
                       np.all(np.isnan(saved["station_x"][i, n:])))
        rebuilt = metrics(x, ref, pred)
        formula_ok &= all(np.isclose(rebuilt[key], saved[key][i], rtol=3e-13, atol=3e-14)
                          for key in rebuilt)
        set_error, sep_error, reatt_error, missed = topology(x, ref, pred)
        topology_ok &= np.isclose(set_error,
                                  saved["separated_set_symmetric_difference"][i],
                                  rtol=0.0, atol=3e-7)
        if np.isfinite(sep_error):
            topology_ok &= np.isclose(sep_error, saved["separation_error_over_span"][i],
                                      rtol=3e-13, atol=3e-14)
            topology_ok &= np.isclose(reatt_error, saved["reattachment_error_over_span"][i],
                                      rtol=3e-13, atol=3e-14)
        topology_ok &= missed == bool(saved["event_missed"][i])
    check("every profile source hash is exact", source_ok)
    check("all padded station distributions obey the schema", station_ok)
    check("independent physical-metric formulas reproduce every case", formula_ok)
    check("independent event topology reproduces every case", topology_ok)

    # Re-execute the expensive model only for three structurally different
    # anchors: flat-with-step, curved repeating, and smooth repeating control.
    for name in ("bfs_Re13700", "periodic_hills_case_1p0", "conv_div_channel_Re12600"):
        i = int(np.flatnonzero(names == name)[0])
        path = ROOT / str(saved["profile_paths"][i])
        x, ref, pred, u_edge = rebuild_prediction(path)
        n = int(saved["n_stations"][i])
        check(f"raw/profile ODE rebuild: {name}",
              n == x.size and
              np.allclose(x, saved["station_x"][i, :n], rtol=0.0, atol=0.0) and
              np.allclose(ref, saved["station_tau_ref"][i, :n], rtol=0.0, atol=0.0) and
              np.allclose(pred, saved["station_tau_pred"][i, :n], rtol=3e-13, atol=3e-14))

    bfs_i = int(np.flatnonzero(names == "bfs_Re13700")[0])
    hill_i = int(np.flatnonzero(names == "periodic_hills_case_1p0")[0])
    check("BFS marginal aggregate pass is recorded, not rounded away",
          0.008 < float(saved["r2_descriptive"][bfs_i] - 0.88) < 0.009)
    check("canonical-hill corrected headline is preserved",
          abs(float(saved["r2_descriptive"][hill_i]) + 47.68617253416459) < 1e-10)
    check("all p50/p95/max station summaries are ordered and finite",
          np.all(np.isfinite(saved["station_abs_p50"])) and
          np.all(saved["station_abs_p50"] <= saved["station_abs_p95"]) and
          np.all(saved["station_abs_p95"] <= saved["station_abs_max"]))
    check("all moving-block intervals contain ordered finite endpoints",
          all(np.all(np.isfinite(saved[f"{key}_ci_lo"])) and
              np.all(saved[f"{key}_ci_lo"] <= saved[f"{key}_ci_hi"])
              for key in ("station_abs_p50", "station_abs_p95",
                          "viscous_drag_signed_error", "station_sign_mismatch_fraction")))

# Negative fixtures: these fail the assumptions that R2 and pointwise relative
# errors silently make, and exercise the event matcher without using target
# stress amplitudes.
try:
    r2(np.ones(8), np.ones(8))
    zero_variance_rejected = False
except ValueError:
    zero_variance_rejected = True
check("negative fixture: zero-variance R2 is rejected", zero_variance_rejected)

x_fixture = np.linspace(0.0, 1.0, 101)
reference_fixture = (x_fixture - 0.20) * (x_fixture - 0.70)
prediction_fixture = (x_fixture - 0.25) * (x_fixture - 0.75)
base = metrics(x_fixture, reference_fixture, prediction_fixture)
scaled = metrics(x_fixture, 17.0 * reference_fixture, 17.0 * prediction_fixture)
check("negative fixture: physical metrics are stress-unit invariant",
      all(np.isclose(base[key], scaled[key], rtol=2e-14, atol=2e-14) for key in base))

zero_fixture = reference_fixture.copy()
zero_fixture[20] = 0.0
finite_fixture = metrics(x_fixture, zero_fixture, prediction_fixture)
check("negative fixture: a local zero causes no singular station error",
      all(np.isfinite(value) for value in finite_fixture.values()))

set_error, sep_error, reatt_error, missed = topology(
    x_fixture, reference_fixture, prediction_fixture)
check("topology fixture: shifted bubble locations are recovered",
      not missed and abs(sep_error - 0.05) < 2e-15 and
      abs(reatt_error - 0.05) < 2e-15 and 0.09 < set_error < 0.11)
_, _, _, missed = topology(x_fixture, reference_fixture, np.ones_like(x_fixture))
check("topology fixture: a missed event cannot receive a finite location score", missed)

summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
check("summary records definitions rather than verdict thresholds",
      summary["n_cases"] == 18 and
      "RMS_x" in summary["metric_definitions"]["station_error"] and
      "never used alone" in summary["metric_definitions"]["r2_descriptive"])

source = re.sub(r"\s+", " ", active_tex(MAIN.read_text(encoding="utf-8")))
pdf = subprocess.run(["pdftotext", str(ROOT / "manuscript/main.pdf"), "-"],
                     check=True, capture_output=True, text=True).stdout
check("active manuscript defines the sign-topology metric operator",
      "symmetric difference" in source and "reference RMS wall stress" in source)
check("active manuscript states the BFS 0.0087 margin",
      "0.0087" in source and "0.349" in source)
check("compiled PDF contains the physical metric result",
      "0.0087" in pdf and "0.349" in pdf)

ledger = LEDGER.read_text(encoding="utf-8")
check("ledger closes M2 and R2-m2 by this verifier",
      all(token in ledger for token in ("**M2**", "**R2-m2**", "verify_m2_r2m2.py")) and
      ledger.count("verify_m2_r2m2.py") >= 2)

passed = sum(ok for _, ok in checks)
print(f"{passed}/{len(checks)} checks passed")
raise SystemExit(0 if passed == len(checks) else 1)
