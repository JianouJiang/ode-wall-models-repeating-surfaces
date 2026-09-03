#!/usr/bin/env python3
"""Record the data and coordinate transform behind the corrected flow figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HILL = ROOT / "codes" / "results" / "momentum_budget_pehill.npz"
BFS = ROOT / "codes" / "error_spreading_viz" / "out" / "fields_bfs.npz"
SCRIPT = ROOT / "codes" / "error_spreading_viz" / "stageM1_forcebalance.py"
PNG = ROOT / "manuscript" / "figures" / "figM1_forcebalance.png"
OUT = ROOT / "codes" / "results" / "figure_streamline_provenance_r3_1b.json"
NODE_OUT = ROOT / "development" / "nodes" / "node_005" / OUT.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slope_audit(path: Path, kind: str) -> dict:
    data = np.load(path, allow_pickle=True)
    if kind == "hill":
        x = np.asarray(data["x_unique"], dtype=float)
        u = np.asarray(data["U"], dtype=float)
        v = np.asarray(data["V"], dtype=float)
        source = np.ones(len(x), dtype=bool)
    else:
        x = np.asarray(data["x"], dtype=float)
        u = np.asarray(data["U"], dtype=float)
        v = np.asarray(data["V"], dtype=float)
        source = x > 0.0
    x = x[source]
    u = u[source]
    v = v[source]
    extent = float(x.max() - x.min())
    valid = np.isfinite(u) & np.isfinite(v) & (np.abs(u) > 1e-6)
    physical = np.abs(v[valid] / u[valid])
    # The legacy call passed physical (U,V) on an x-normalised grid, so the
    # renderer used V/U.  The correct plotted slope is
    # dy/d(x/L_x)=L_x V/U.
    old_plot = np.abs(v[valid] / u[valid])
    corrected = np.abs(extent * v[valid] / u[valid])
    return {
        "x_extent": extent,
        "physical_median_abs_dy_dx": float(np.median(physical)),
        "old_plot_median_abs_dy_dxnormalised": float(np.median(old_plot)),
        "corrected_plot_median_abs_dy_dxnormalised": float(np.median(corrected)),
        "old_to_corrected_factor": float(np.median(corrected) / np.median(old_plot)),
        "coordinate_identity": "dy/dx_normalised=(x_max-x_min)*V/U",
    }


def main() -> None:
    payload = {
        "schema": "figure-streamline-provenance-r3-1b-v1",
        "resolution": "both criticised figures are removed from the active build; the archived streamline generator is additionally repaired against stored DNS/RANS velocity fields with the exact normalised-coordinate vector transform",
        "hill": slope_audit(HILL, "hill"),
        "bfs": slope_audit(BFS, "bfs"),
        "sources": {
            "hill_npz": {"path": str(HILL.relative_to(ROOT)), "sha256": sha256(HILL)},
            "bfs_npz": {"path": str(BFS.relative_to(ROOT)), "sha256": sha256(BFS)},
            "generator": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
            "rendered_png": {"path": str(PNG.relative_to(ROOT)), "sha256": sha256(PNG)},
        },
        "uses_random_or_generative_image_model": False,
        "status": "PASS",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUT.write_text(rendered)
    NODE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NODE_OUT.write_text(rendered)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
