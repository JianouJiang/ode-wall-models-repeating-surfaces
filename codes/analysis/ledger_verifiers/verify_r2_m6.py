#!/usr/bin/env python3
"""Stable-path coupled-metric verifier for referee row R2-m6."""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"


def active_tex(text: str) -> str:
    kept, depth = [], 0
    for line in text.splitlines():
        token = line.split("%", 1)[0].strip()
        if token.startswith(r"\iffalse"):
            depth += 1
        elif token.startswith(r"\fi") and depth:
            depth -= 1
        elif depth == 0:
            kept.append(line)
    if depth:
        raise RuntimeError("unclosed \\iffalse block")
    return "\n".join(kept)


def main() -> int:
    grid = json.loads((RESULTS / "rswm_grid_results_l3_summary.json").read_text())
    eq = grid["base_metrics"]["G2c:equilibrium"]
    tb = grid["base_metrics"]["G2c:total_gradient_tble"]
    eq_bias = 100 * (eq["reattachment_x_over_H"] - eq["truth_reattachment_x_over_H"]) / eq["truth_reattachment_x_over_H"]
    tb_bias = 100 * (tb["reattachment_x_over_H"] - tb["truth_reattachment_x_over_H"]) / tb["truth_reattachment_x_over_H"]
    source = re.sub(r"\s+", " ", active_tex((ROOT / "manuscript/main.tex").read_text(encoding="utf-8")))
    pdf = subprocess.run(["pdftotext", str(ROOT / "manuscript/main.pdf"), "-"],
                         check=True, capture_output=True, text=True).stdout
    expected_jobs = {"14889013", "14889015", "14889021", "14889022", "14889025", "14889026"}
    r2_tokens = [f"{eq['r2']:.3f}", f"{tb['r2']:.3f}"]
    error_tokens = [f"{eq['relative_rms']:.3f}", f"{tb['relative_rms']:.3f}"]
    bias_tokens = [f"{eq_bias:.2f}", f"{tb_bias:.2f}"]
    # TeX source uses ASCII '-', whereas pdftotext commonly emits U+2212.
    pdf_ascii = pdf.replace("−", "-").replace(" ", "")
    checks = [
        ("corrected three-grid coupled matrix harvested", grid["status"] == "RSWM_GRID_RESULTS_L3_OK"
         and len(grid["producer_jobs"]) == 6 and set(grid["producer_jobs"].values()) == expected_jobs),
        ("finite finest-grid signed reattachment biases", all(map(math.isfinite, (eq_bias, tb_bias)))),
        ("finite physical traction metrics", all(map(math.isfinite,
         (eq["r2"], tb["r2"], eq["relative_rms"], tb["relative_rms"])))
         and eq["relative_rms"] > 0 and tb["relative_rms"] > 0),
        ("active body prints both finest-grid signed biases",
         all(token in source for token in bias_tokens)),
        ("active body prints R2 and physical error from the corrected matrix",
         all(token in source for token in r2_tokens + error_tokens)),
        ("superseded coarse interpretation removed", all(token not in source for token in (r"$-20.6\%$", r"$-22.9\%$", r"$\Rtwo(C_f)=0.762$"))),
        ("compiled PDF contains both corrected finest-grid biases",
         all(token.replace("-", "-") + "%" in pdf_ascii for token in bias_tokens)),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
