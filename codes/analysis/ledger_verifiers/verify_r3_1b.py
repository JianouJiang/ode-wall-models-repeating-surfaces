#!/usr/bin/env python3
"""Independent verification of the R3-1b figure-provenance repair."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "codes/error_spreading_viz/stageM1_forcebalance.py"
BFS = ROOT / "codes/error_spreading_viz/out/fields_bfs.npz"
HILL = ROOT / "codes/results/momentum_budget_pehill.npz"
SUMMARY = ROOT / "codes/results/figure_streamline_provenance_r3_1b.json"
OUT_PNG = ROOT / "codes/error_spreading_viz/out/figM1_forcebalance.png"
PAPER_PNG = ROOT / "manuscript/figures/figM1_forcebalance.png"
MAIN = ROOT / "manuscript/main.tex"
FLS = ROOT / "manuscript/main.fls"
PDF = ROOT / "manuscript/main.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = GENERATOR.read_text(encoding="utf-8")
    manuscript = MAIN.read_text(encoding="utf-8")
    recorder = FLS.read_text(encoding="utf-8") if FLS.exists() else ""
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    bfs = np.load(BFS, allow_pickle=True)
    x = np.asarray(bfs["x"], dtype=float)
    extent = float(x[x > 0.0].max() - x[x > 0.0].min())
    pdf_text = subprocess.run(
        ["pdftotext", str(PDF), "-"], check=True, capture_output=True, text=True
    ).stdout

    forbidden_maps = (
        "fig_repeating_class_morphology", "fig_repeating_class_amplitude"
    )
    checks = [
        ("both streamline layers transform V by x extent",
         source.count("x_extent * Vu") == 2),
        ("generator contains no illustration/not-to-scale badge",
         "illustration / not to scale" not in source.lower()),
        ("BFS correction factor independently equals physical x extent",
         abs(payload["bfs"]["old_to_corrected_factor"] - extent) < 1e-12 and
         abs(extent - 29.96078431372549) < 1e-10),
        ("hill correction factor independently equals physical x extent",
         abs(payload["hill"]["old_to_corrected_factor"] -
             payload["hill"]["x_extent"]) < 1e-12),
        ("provenance source hashes are live",
         payload["sources"]["bfs_npz"]["sha256"] == sha256(BFS) and
         payload["sources"]["hill_npz"]["sha256"] == sha256(HILL) and
         payload["sources"]["generator"]["sha256"] == sha256(GENERATOR)),
        ("published and producer PNGs are byte-identical",
         sha256(OUT_PNG) == sha256(PAPER_PNG) ==
         payload["sources"]["rendered_png"]["sha256"]),
        ("hand-drawn overview is absent from the active LaTeX build",
         "fig_separation_schematic" not in recorder),
        ("criticised force-balance figure is absent from the active build",
         "figM1_forcebalance" not in recorder),
        ("archived caption discloses DNS hill and matched-RANS BFS provenance",
         "Xiao DNS hill field" in manuscript and
         "matched-RANS" in manuscript),
        ("hard-coded class-map assets remain cut",
         all(token not in recorder for token in forbidden_maps)),
        ("compiled paper does not present the cut overview caption",
         "The three near-wall regimes" not in pdf_text),
        ("compiled paper does not present the cut force-balance caption",
         "Near-wall momentum balance (log scale)" not in pdf_text),
        ("resolution records no random or generative imagery",
         payload["status"] == "PASS" and
         payload["uses_random_or_generative_image_model"] is False),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
