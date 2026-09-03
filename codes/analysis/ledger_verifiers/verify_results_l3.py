#!/usr/bin/env python3
"""Independent checks for the Level-3 Xiao grid-results reduction."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


NODE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
L2_NPZ = ROOT / "codes" / "results" / "rswm_common_surface_grid_l2.npz"
L3_NPZ = NODE / "grid_results_l3.npz"
L3_JSON = NODE / "grid_results_l3_summary.json"
PAPER_NPZ = ROOT / "codes" / "results" / "rswm_grid_results_l3.npz"
PAPER_JSON = ROOT / "codes" / "results" / "rswm_grid_results_l3_summary.json"
FIGURE_PDF = NODE / "fig_common_surface_grid_l3.pdf"
FIGURE_PNG = NODE / "fig_common_surface_grid_l3.png"
PAPER_FIGURE_PDF = ROOT / "codes" / "figures" / "fig_common_surface_grid_l3.pdf"
PAPER_FIGURE_PNG = ROOT / "codes" / "figures" / "fig_common_surface_grid_l3.png"
RESULTS_MD = NODE / "results.md"
HANDOFF = NODE / "STAGE_HANDOFF.md"
TERMINAL = NODE / "ARCHER2_TERMINAL_EVIDENCE.txt"
LEDGER_CHECK = NODE / "LEDGER_CHECK.txt"
OUT = NODE / "verification.json"

EXPECTED_L2_SHA256 = "8951ea66d8de5f79fb66106dcb3760d70753b818560dd13cb34db9db5b64e396"
GRIDS = ("G0", "G1c", "G2c")
MODELS = ("equilibrium", "total_gradient_tble")
EXPECTED_JOBS = ("14868882", "14868883", "14868884", "14868885", "14868887", "14868888")
DENSE_N = 4096
BLOCK_POINTS = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_interp(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(x, float))
    xs = np.asarray(x, float)[order]
    ys = np.asarray(y, float)[order]
    return np.interp(
        np.mod(target, 1.0),
        np.r_[xs - 1.0, xs, xs + 1.0],
        np.r_[ys, ys, ys],
    )


def exact_sign_p(block_values: np.ndarray, two_sided: bool = False) -> float:
    values = np.asarray(block_values, float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    observed = float(np.mean(values))
    if two_sided:
        return float(np.mean(np.abs(null) >= abs(observed)))
    return float(np.mean(null >= observed))


def main() -> int:
    checks: list[str] = []

    def check(label: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    for path in (
        L2_NPZ, L3_NPZ, L3_JSON, PAPER_NPZ, PAPER_JSON, FIGURE_PDF, FIGURE_PNG,
        PAPER_FIGURE_PDF, PAPER_FIGURE_PNG, RESULTS_MD, HANDOFF, TERMINAL, LEDGER_CHECK,
    ):
        check(f"exists:{path.name}", path.is_file() and path.stat().st_size > 0)
    check("locked Level-2 hash", sha256(L2_NPZ) == EXPECTED_L2_SHA256)

    summary = json.loads(L3_JSON.read_text())
    check("summary status", summary["status"] == "RSWM_GRID_RESULTS_L3_OK")
    check("staged NPZ identical", sha256(L3_NPZ) == sha256(PAPER_NPZ))
    check("staged JSON identical", sha256(L3_JSON) == sha256(PAPER_JSON))
    check("bootstrap draws", summary["bootstrap_protocol"]["draws"] == 20000)
    check("primary physical block", abs(
        summary["bootstrap_protocol"]["primary_block_length_over_H"] - 1.125
    ) < 1.0e-14)
    check("registered producer set", set(summary["producer_jobs"].values()) == set(EXPECTED_JOBS))
    check("registered finalizer", summary["finalizer_job"] == "14869295")

    terminal = TERMINAL.read_text()
    for job in EXPECTED_JOBS + ("14869295",):
        check(f"terminal:{job}", f"{job}|COMPLETED" in terminal and f"{job}|COMPLETED" in terminal)
        line = next(item for item in terminal.splitlines() if item.startswith(job + "|"))
        check(f"exit-zero:{job}", line.endswith("|0:0"))

    l2 = np.load(L2_NPZ)
    l3 = np.load(L3_NPZ)
    check("npz status", str(l3["status"]) == "RSWM_GRID_RESULTS_L3_OK")
    check("npz draw count", int(l3["bootstrap_draws"]) == 20000)
    dense_phase = np.arange(DENSE_N, dtype=float) / DENSE_N
    truth = periodic_interp(l2["truth_phase"], l2["truth_tau_s"], dense_phase)
    check("truth rebuilt", np.allclose(truth, l3["truth_tau_s"], rtol=0.0, atol=1.0e-15))

    direct_predictions: dict[tuple[str, str], np.ndarray] = {}
    for grid in GRIDS:
        for model in MODELS:
            prefix = f"{grid}_{model}"
            prediction = periodic_interp(
                l2[f"{prefix}_phase"], l2[f"{prefix}_tau_s"], dense_phase
            )
            direct_predictions[(grid, model)] = prediction
            error = prediction - truth
            relative_rms = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(truth**2)))
            variance = float(np.sum((truth - np.mean(truth)) ** 2))
            r2 = float(1.0 - np.sum(error**2) / variance)
            record = summary["base_metrics"][f"{grid}:{model}"]
            check(f"relRMS:{prefix}", abs(relative_rms - record["relative_rms"]) < 2.0e-13)
            check(f"r2:{prefix}", abs(r2 - record["r2"]) < 2.0e-12)
            check(f"curve:{prefix}", np.allclose(
                prediction, l3[f"{prefix}_tau_s"], rtol=0.0, atol=1.0e-15
            ))
            samples = l3[f"{prefix}_primary_bootstrap_relative_rms"]
            check(f"bootstrap finite:{prefix}", samples.shape == (20000,) and np.all(np.isfinite(samples)))
            low, median, high = np.quantile(samples, (0.025, 0.5, 0.975))
            reported = summary["phase_bootstrap_primary_intervals"][f"{grid}:{model}"]
            check(f"bootstrap interval:{prefix}", np.allclose(
                (low, median, high),
                (reported["low"], reported["median"], reported["high"]),
                rtol=0.0,
                atol=1.0e-14,
            ))

    # Rebuild the exact tests without calling the producer.
    raw_p = {}
    for model in MODELS:
        difference = (direct_predictions[("G2c", model)] - truth) ** 2 - truth**2
        blocks = np.asarray([
            np.mean(difference[i * BLOCK_POINTS:(i + 1) * BLOCK_POINTS])
            for i in range(DENSE_N // BLOCK_POINTS)
        ])
        p_one = exact_sign_p(blocks)
        p_two = exact_sign_p(blocks, two_sided=True)
        report = summary["failure_significance_tests"][model]
        check(f"exact one-sided:{model}", abs(p_one - report["p_one_sided"]) < 1.0e-15)
        check(f"exact two-sided:{model}", abs(p_two - report["p_two_sided"]) < 1.0e-15)
        raw_p[model] = p_one
    check("Holm equilibrium", abs(
        summary["failure_significance_tests"]["equilibrium"]["p_one_sided_holm_two_models"]
        - 0.0078125
    ) < 1.0e-15)
    check("Holm TBLE", abs(
        summary["failure_significance_tests"]["total_gradient_tble"]["p_one_sided_holm_two_models"]
        - 0.0078125
    ) < 1.0e-15)

    loss_difference = (
        (direct_predictions[("G2c", "total_gradient_tble")] - truth) ** 2
        - (direct_predictions[("G2c", "equilibrium")] - truth) ** 2
    )
    loss_blocks = np.asarray([
        np.mean(loss_difference[i * BLOCK_POINTS:(i + 1) * BLOCK_POINTS])
        for i in range(DENSE_N // BLOCK_POINTS)
    ])
    check("model comparison two-sided", abs(
        exact_sign_p(loss_blocks, two_sided=True)
        - summary["model_comparison"]["paired_exact_block_test"]["p_two_sided"]
    ) < 1.0e-15)
    check("model ranking not overclaimed", "do not establish" in summary["model_comparison"]["interpretation"])

    for model in MODELS:
        robust = summary["grid_robustness"][model]
        check(f"favorable envelope remains failed:{model}",
              robust["favorable_one_observed_envelope_relative_rms"] > 1.0)
        check(f"favorable R2 remains negative:{model}",
              robust["favorable_one_observed_envelope_r2"] < 0.0)
        check(f"terminal drift small:{model}",
              abs(robust["terminal_window_change"]) < 0.011)

    check("results approach line", RESULTS_MD.read_text().startswith("APPROACH: "))
    handoff = HANDOFF.read_text()
    for heading in (
        "CURRENT-LEVEL REQUIRED — COMPLETE",
        "CURRENT-LEVEL REQUIRED — OPEN",
        "LIVE REMOTE JOBS",
        "NEXT-LEVEL READY (NOT A CURRENT GATE)",
        "BLOCKED/DEPENDENCIES",
    ):
        check(f"handoff heading:{heading}", heading in handoff)
    check("no current-level open work", "# CURRENT-LEVEL REQUIRED — OPEN\n\nNONE" in handoff)
    check("no live remote work", "# LIVE REMOTE JOBS\n\nNONE" in handoff)
    check("ledger row passes", "[PASS] R1-SCI-3 / M3: 2685/2685 checks passed" in LEDGER_CHECK.read_text())
    check("node verifier recorded", "RSWM_GRID_RESULTS_L3_VERIFIED checks=" in LEDGER_CHECK.read_text())

    check("PDF signature", FIGURE_PDF.read_bytes().startswith(b"%PDF"))
    check("staged PDF identical", sha256(FIGURE_PDF) == sha256(PAPER_FIGURE_PDF))
    check("staged PNG identical", sha256(FIGURE_PNG) == sha256(PAPER_FIGURE_PNG))
    image = mpimg.imread(FIGURE_PNG)
    check("publication raster size", image.shape[0] >= 1500 and image.shape[1] >= 2000)
    check("publication raster finite", np.all(np.isfinite(image)))

    verification = {
        "status": "RSWM_GRID_RESULTS_L3_VERIFIED",
        "checks_passed": len(checks),
        "checks": checks,
        "hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                L2_NPZ, L3_NPZ, L3_JSON, PAPER_NPZ, PAPER_JSON, FIGURE_PDF, FIGURE_PNG,
                PAPER_FIGURE_PDF, PAPER_FIGURE_PNG,
                RESULTS_MD, HANDOFF, TERMINAL, LEDGER_CHECK,
                NODE / "analyze_grid_results_l3.py",
            )
        },
    }
    OUT.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"RSWM_GRID_RESULTS_L3_VERIFIED checks={len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
