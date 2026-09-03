#!/usr/bin/env python3
"""Stable-path verifier for exact-pressure referee row R1-STA-4a."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "codes/results/exact_pressure_traction_512.npz"
SUMMARY = ROOT / "codes/results/exact_pressure_traction_512_summary.json"
RAW = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
       "pehill-5-cases-DNS/case_1p0/dns-data/mean_files.dat")
HILL_UTIL = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
             "utility/hill-geometry-gereration")
sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402

ETA = np.linspace(0.0, 0.1, 101)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derivative(field: np.ndarray, x: np.ndarray) -> np.ndarray:
    work = field[:, None] if field.ndim == 1 else field
    wave = 2 * np.pi * np.fft.fftfreq(x.size, d=float(x[1] - x[0]))
    value = np.fft.ifft(1j * wave[:, None] * np.fft.fft(work, axis=0), axis=0).real
    return value[:, 0] if field.ndim == 1 else value


def main() -> int:
    checks: list[tuple[str, bool]] = []
    def check(name: str, value: bool) -> None:
        checks.append((name, bool(value)))
        print(f"[{'PASS' if value else 'FAIL'}] {name}")

    table = np.loadtxt(RAW)
    x, y = np.unique(table[:, 0]), np.unique(table[:, 1])
    U = table[:, 2].reshape(y.size, x.size).T
    V = table[:, 3].reshape(y.size, x.size).T
    P = table[:, 5].reshape(y.size, x.size).T
    h = hill_profile(x.copy())
    mapped = np.empty((x.size, ETA.size))
    p_wall = np.empty(x.size)
    speed = np.hypot(U, V)
    for i in range(x.size):
        k0 = int(np.flatnonzero(speed[i] > 1e-10)[0])
        distance = y[k0:k0 + 12] - h[i]
        p_wall[i] = np.polyval(np.polyfit(distance, P[i, k0:k0 + 12], 1), 0.0)
        mapped[i] = np.interp(h[i] + ETA, np.r_[h[i], y[k0:]],
                              np.r_[p_wall[i], P[i, k0:]])
    hp = derivative(h, x)
    exact = derivative(np.trapezoid(mapped, ETA, axis=1), x) - hp * (mapped[:, -1] - mapped[:, 0])

    saved = np.load(RESULT, allow_pickle=False)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    check("raw grid is 512 by 257", table.shape == (512 * 257, 6))
    check("raw source hash is bound", digest(RAW) == summary["source_sha256"])
    check("exact pressure traction rebuilds from raw P", np.allclose(exact, saved["exact_pressure_traction"], atol=2e-12, rtol=0.0))
    check("matching height is eta/H=0.1", float(saved["matching_height_over_H"]) == 0.1)
    check("18-operator uncertainty ensemble is recorded", summary["exact_operator_ensemble_count"] == 18)
    check("exact/direct mapped identity closes", summary["identity_vs_direct_relative_rms"] < 5e-3)
    check("wall-gradient approximation is materially different", summary["relative_rms_error"] > 0.25)
    check("no model stress or wall-stress input", not summary["uses_modelled_eddy_viscosity"] and not summary["uses_wall_stress"])
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
