#!/usr/bin/env python3
"""Reproduce the public Hausmann--van Wachem periodic-hill benchmark.

The input is the CC-BY-4.0 deposit associated with Phys. Rev. Fluids 10,
044604 (2025), DOI 10.5281/zenodo.15094241.  This script deliberately uses
the authors' deposited profiles and the normalization in their plotting
script.  It does not treat the published volume-filtered method as ours.

The comparison covers both deposited filter widths, all nine phase stations
and all three deposited subfilter closures.  Each LES profile is compared on
its own y grid with the explicitly filtered DNS interpolated to that grid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "codes" / "vendor" / "hausmann_vfwmles_zenodo15094241"
DATA = SOURCE / "PeriodicHill"
RESULTS = ROOT / "codes" / "results"

FILTERS = {"0035": 0.035, "0070": 0.070}
POSITIONS = ("005", "100", "200", "300", "400", "500", "600", "700", "800")
MODELS = ("LESVRE", "LESNL", "LESNLVRE")
UB = 1.0595  # value in the deposited PlotProfilesPeriodicHill.py
ZENODO_RECORD = 15094241
ZENODO_DOI = "10.5281/zenodo.15094241"
ZENODO_ZIP_MD5 = "4aeba3f2e0be214b6188f17e0380b2c8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(y: np.ndarray, prediction: np.ndarray,
                reference: np.ndarray) -> float:
    numerator = np.trapezoid((prediction - reference) ** 2, y)
    denominator = np.trapezoid(reference ** 2, y)
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sqrt(numerator / denominator))


def evaluate_profile(reference_path: Path, prediction_path: Path) -> dict:
    reference = np.loadtxt(reference_path)
    prediction = np.loadtxt(prediction_path)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise RuntimeError(f"unexpected filtered-DNS schema: {reference_path}")
    if prediction.ndim != 2 or prediction.shape[1] != 6:
        raise RuntimeError(f"unexpected VF-WMLES schema: {prediction_path}")

    lower = max(float(reference[:, 0].min()), float(prediction[:, 0].min()))
    upper = min(float(reference[:, 0].max()), float(prediction[:, 0].max()))
    keep = (prediction[:, 0] >= lower) & (prediction[:, 0] <= upper)
    if np.count_nonzero(keep) < 8:
        raise RuntimeError("insufficient common profile support")
    y = prediction[keep, 0]
    u_dns = np.interp(y, reference[:, 0], reference[:, 1])
    v_dns = np.interp(y, reference[:, 0], reference[:, 2])
    # This is the exact normalization used by the deposited plotting script.
    u_les = prediction[keep, 1] / UB
    v_les = prediction[keep, 2] / UB

    i_dns = int(np.argmin(np.abs(reference[:, 0])))
    i_les = int(np.argmin(np.abs(prediction[:, 0])))
    return {
        "relative_l2_u": relative_l2(y, u_les, u_dns),
        "relative_l2_v": relative_l2(y, v_les, v_dns),
        "wall_u_error": float(prediction[i_les, 1] / UB - reference[i_dns, 1]),
        "wall_v_error": float(prediction[i_les, 2] / UB - reference[i_dns, 2]),
        "y_overlap_min": lower,
        "y_overlap_max": upper,
        "n_les_overlap": int(y.size),
        "reference_sha256": sha256(reference_path),
        "prediction_sha256": sha256(prediction_path),
    }


def main() -> None:
    required = [SOURCE / "PlotProfilesPeriodicHill.py", DATA / "README"]
    required += [DATA / width / f"filt_{position}.txt"
                 for width in FILTERS for position in POSITIONS]
    required += [DATA / width / f"{model}_{position}.txt"
                 for width in FILTERS for model in MODELS for position in POSITIONS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing deposited Hausmann files:\n" + "\n".join(missing))

    records: list[dict] = []
    for width, sigma in FILTERS.items():
        for model in MODELS:
            for position in POSITIONS:
                record = evaluate_profile(
                    DATA / width / f"filt_{position}.txt",
                    DATA / width / f"{model}_{position}.txt",
                )
                record.update(filter_id=width, sigma_over_H=sigma,
                              model=model, x_over_H=float(position) / 100.0)
                records.append(record)

    summaries = {}
    for width, sigma in FILTERS.items():
        summaries[width] = {}
        for model in MODELS:
            subset = [r for r in records
                      if r["filter_id"] == width and r["model"] == model]
            summaries[width][model] = {
                "sigma_over_H": sigma,
                "n_phase_stations": len(subset),
                "median_relative_l2_u": float(np.median(
                    [r["relative_l2_u"] for r in subset])),
                "max_relative_l2_u": float(np.max(
                    [r["relative_l2_u"] for r in subset])),
                "median_relative_l2_v": float(np.median(
                    [r["relative_l2_v"] for r in subset])),
                "max_relative_l2_v": float(np.max(
                    [r["relative_l2_v"] for r in subset])),
                "rms_wall_u_error": float(np.sqrt(np.mean(
                    [r["wall_u_error"] ** 2 for r in subset]))),
                "rms_wall_v_error": float(np.sqrt(np.mean(
                    [r["wall_v_error"] ** 2 for r in subset]))),
            }

    fine_u = summaries["0035"]["LESVRE"]["median_relative_l2_u"]
    coarse_u = summaries["0070"]["LESVRE"]["median_relative_l2_u"]
    summary = {
        "schema": "hausmann-vfwmles-source-reproduction-l1-v1",
        "source": {
            "authors": "Max Hausmann and Berend van Wachem",
            "paper": "Physical Review Fluids 10, 044604 (2025)",
            "paper_doi": "10.1103/PhysRevFluids.10.044604",
            "zenodo_record": ZENODO_RECORD,
            "zenodo_doi": ZENODO_DOI,
            "zenodo_actualdata_zip_md5": ZENODO_ZIP_MD5,
            "license": "CC-BY-4.0",
            "deposited_plot_script_sha256": sha256(
                SOURCE / "PlotProfilesPeriodicHill.py"),
            "deposited_readme_sha256": sha256(DATA / "README"),
        },
        "protocol": {
            "filter_widths_sigma_over_H": list(FILTERS.values()),
            "phase_stations_x_over_H": [float(p) / 100 for p in POSITIONS],
            "subfilter_models": list(MODELS),
            "bulk_velocity_from_deposited_plot_script": UB,
            "comparison": "LES grid; linearly interpolated explicitly filtered DNS",
            "metric": "profile-integrated relative L2 on common y support",
        },
        "n_profile_comparisons": len(records),
        "summaries": summaries,
        "source_claim_checks": {
            "fine_vreman_median_u_rel_l2_below_0p05": bool(fine_u < 0.05),
            "coarse_vreman_u_error_exceeds_fine": bool(coarse_u > fine_u),
            "vreman_lowest_median_u_error_both_filters": bool(all(
                summaries[w]["LESVRE"]["median_relative_l2_u"] == min(
                    summaries[w][m]["median_relative_l2_u"] for m in MODELS)
                for w in FILTERS)),
        },
    }
    summary["status"] = ("PASS" if all(summary["source_claim_checks"].values())
                         else "FAIL")

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "hausmann_vfwmles_reproduction_l1.json"
    npz_path = RESULTS / "hausmann_vfwmles_reproduction_l1.npz"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    arrays = {
        key: np.array([record[key] for record in records])
        for key in ("filter_id", "sigma_over_H", "model", "x_over_H",
                    "relative_l2_u", "relative_l2_v", "wall_u_error",
                    "wall_v_error", "n_les_overlap", "reference_sha256",
                    "prediction_sha256")
    }
    np.savez(npz_path, **arrays,
             schema=np.array(summary["schema"]),
             source_zip_md5=np.array(ZENODO_ZIP_MD5))
    print("HAUSMANN VF-WMLES SOURCE REPRODUCTION")
    print(f"  profiles: {len(records)}/54")
    for width in FILTERS:
        print(f"  sigma/H={FILTERS[width]:.3f}")
        for model in MODELS:
            item = summaries[width][model]
            print(f"    {model:8s} median relL2(U)="
                  f"{item['median_relative_l2_u']:.6f}; "
                  f"max={item['max_relative_l2_u']:.6f}")
    print(f"  STATUS: {summary['status']}")
    if summary["status"] != "PASS":
        raise SystemExit("source reproduction failed preregistered checks")


if __name__ == "__main__":
    main()
