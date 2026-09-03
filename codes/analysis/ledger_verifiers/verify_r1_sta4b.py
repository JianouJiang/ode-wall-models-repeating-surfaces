#!/usr/bin/env python3
"""Stable-path guard for closure-free total stress row R1-STA-4b."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "codes/results/closure_free_total_stress_r1sta4b.npz"
SUMMARY = ROOT / "codes/results/closure_free_total_stress_r1sta4b_summary.json"
RAW = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
       "pehill-29-cases-DNS/alph10-9-3036")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    data = np.load(RESULT, allow_pickle=False)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks = [
        ("raw mean hash", digest(RAW / "mean_files.dat") == summary["source_sha256"]["mean"]),
        ("raw rms1 hash", digest(RAW / "rms_files1.dat") == summary["source_sha256"]["rms1"]),
        ("documented uv source hash", digest(RAW / "rms_files2.dat") == summary["source_sha256"]["rms2"]),
        ("profile grid", data["eta"].shape == (401,)),
        ("finite components", all(np.all(np.isfinite(data[k])) for k in
                                  ("viscous_stress", "reynolds_stress", "dispersive_stress", "total_stress"))),
        ("component identity", np.allclose(data["total_stress"],
                                           data["viscous_stress"] + data["reynolds_stress"] + data["dispersive_stress"],
                                           atol=2e-15, rtol=0.0)),
        ("independent raw audit passed", summary["checks_passed"] == summary["checks_total"] == 13),
        ("no model stress in definition", "molecular" in summary["stress_definition"] or
                                          "viscous" in summary["stress_definition"]),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
