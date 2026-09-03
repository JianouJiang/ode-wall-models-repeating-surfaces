#!/usr/bin/env python3
"""Independent audit for referee-ledger row R1-STA-4b.

The audit reconstructs the wavelength-averaged total-stress profile directly
from the documented Xiao DNS columns.  It deliberately does not import the
production budget module.  The phase-resolved Reynolds stress is read from
``rms_files2.dat``; the dispersive stress is introduced only after the second,
wavelength average.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RAW = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
       "pehill-29-cases-DNS/alph10-9-3036")
HILL_UTIL = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
             "utility/hill-geometry-gereration")
PRODUCER = ROOT / "codes/analysis/da_budget.py"
REFERENCE = ROOT / "codes/results/matching_plane_stress_l1.npz"
NODE_NPZ = Path(__file__).with_name("closure_free_total_stress_r1sta4b.npz")
NODE_JSON = Path(__file__).with_name("closure_free_total_stress_r1sta4b_summary.json")
RESULT_NPZ = ROOT / "codes/results/closure_free_total_stress_r1sta4b.npz"
RESULT_JSON = ROOT / "codes/results/closure_free_total_stress_r1sta4b_summary.json"
NU = 1.0 / 5600.0
VELOCITY_TOL = 1.0e-10

sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reshape_column(table: np.ndarray, column: int, nx: int, ny: int) -> np.ndarray:
    return table[:, column].reshape(ny, nx).T


def periodic_derivative(field: np.ndarray, x: np.ndarray) -> np.ndarray:
    vector = field.ndim == 1
    work = field[:, None] if vector else field
    dx = float(x[1] - x[0])
    wave = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    derivative = np.fft.ifft(
        1j * wave[:, None] * np.fft.fft(work, axis=0), axis=0
    ).real
    return derivative[:, 0] if vector else derivative


def interpolate_from_wall(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    field: np.ndarray,
    wall: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    mapped = np.empty((x.size, eta.size))
    for i in range(x.size):
        fluid = np.flatnonzero(speed[i] > VELOCITY_TOL)
        if fluid.size < 16:
            raise RuntimeError(f"insufficient fluid nodes at station {i}")
        k0 = int(fluid[0])
        ordinate = np.r_[0.0, field[i, k0:]]
        abscissa = np.r_[wall[i], y[k0:]]
        mapped[i] = np.interp(wall[i] + eta, abscissa, ordinate)
    return mapped


def executable_identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
    return names


def main() -> None:
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    mean_path = RAW / "mean_files.dat"
    rms1_path = RAW / "rms_files1.dat"
    rms2_path = RAW / "rms_files2.dat"
    mean = np.loadtxt(mean_path)
    rms1 = np.loadtxt(rms1_path)
    rms2 = np.loadtxt(rms2_path)

    check("documented raw column counts are 6/6/5",
          mean.shape[1] == 6 and rms1.shape[1] == 6 and rms2.shape[1] == 5)
    check("raw coordinates align",
          np.allclose(mean[:, :2], rms1[:, :2], rtol=0.0, atol=2e-7) and
          np.allclose(mean[:, :2], rms2[:, :2], rtol=0.0, atol=2e-7))

    x = np.unique(mean[:, 0])
    y = np.unique(mean[:, 1])
    nx, ny = x.size, y.size
    check("raw array is a complete tensor-product grid", nx * ny == mean.shape[0])

    U0 = reshape_column(mean, 2, nx, ny)
    V0 = reshape_column(mean, 3, nx, ny)
    Rxx0 = reshape_column(rms1, 2, nx, ny)
    # Documented schema: rms_files2 = x,y,uv,uw,vw.
    Rxy0 = reshape_column(rms2, 2, nx, ny)
    wrong_Rxy0 = reshape_column(rms1, 5, nx, ny)
    speed = np.hypot(U0, V0)

    with np.load(REFERENCE, allow_pickle=False) as saved:
        eta = saved["eta"].astype(float)
        wall = hill_profile(x.copy())
        hp = periodic_derivative(wall, x)

        U = interpolate_from_wall(x, y, speed, U0, wall, eta)
        V = interpolate_from_wall(x, y, speed, V0, wall, eta)
        Rxx = interpolate_from_wall(x, y, speed, Rxx0, wall, eta)
        Rxy = interpolate_from_wall(x, y, speed, Rxy0, wall, eta)
        wrong_Rxy = interpolate_from_wall(x, y, speed, wrong_Rxy0, wall, eta)

        U_eta = np.gradient(U, eta, axis=1, edge_order=2)
        V_eta = np.gradient(V, eta, axis=1, edge_order=2)
        U_x = periodic_derivative(U, x) - hp[:, None] * U_eta
        V_x = periodic_derivative(V, x) - hp[:, None] * V_eta
        W = V - hp[:, None] * U

        reynolds_flux = Rxy - hp[:, None] * Rxx
        wrong_reynolds_flux = wrong_Rxy - hp[:, None] * Rxx
        viscous_flux = (-NU * (U_eta + V_x) +
                        2.0 * hp[:, None] * NU * U_x)
        U_bar = np.mean(U, axis=0)
        W_bar = np.mean(W, axis=0)
        dispersive_flux = np.mean(
            (U - U_bar[None, :]) * (W - W_bar[None, :]), axis=0
        )

        reynolds_mean = np.mean(reynolds_flux, axis=0)
        viscous_mean = np.mean(viscous_flux, axis=0)
        total_stress = -(viscous_mean + reynolds_mean + dispersive_flux)

        check("saved contravariant velocity reproduces from raw fields",
              np.max(np.abs(W - saved["W"])) < 2e-13)
        check("saved Reynolds flux reproduces from documented uv column",
              np.max(np.abs(reynolds_mean -
                            saved["reynolds_contravariant_mean"])) < 2e-13)
        check("saved viscous flux reproduces from raw mean velocities",
              np.max(np.abs(viscous_mean -
                            saved["viscous_contravariant_mean"])) < 2e-13)
        check("saved dispersive flux reproduces only after wavelength average",
              np.max(np.abs(dispersive_flux - saved["dispersive"])) < 2e-13)
        check("mean-advection decomposition is exact",
              np.max(np.abs(np.mean(U * W, axis=0) -
                            (U_bar * W_bar + dispersive_flux))) < 2e-14)
        check("legacy rms_files1 pressure-variance column is not uv",
              np.max(np.abs(np.mean(wrong_reynolds_flux, axis=0) -
                            saved["reynolds_contravariant_mean"])) > 1e-3)
        check("closure-free total stress is finite at every height",
              np.all(np.isfinite(total_stress)))
        check("closure-free profile contains molecular, Reynolds and dispersive terms",
              all(np.max(np.abs(term)) > 0.0 for term in
                  (viscous_mean, reynolds_mean, dispersive_flux)))

        saved_hashes = {
            "mean": sha256(mean_path),
            "rms1": sha256(rms1_path),
            "rms2": sha256(rms2_path),
        }

    identifiers = executable_identifiers(PRODUCER)
    banned = {"nu_t", "nut", "eddy_viscosity", "boussinesq"}
    check("production exact-balance arithmetic has no model-stress identifier",
          identifiers.isdisjoint(banned))

    source_text = PRODUCER.read_text(encoding="utf-8")
    component_expression = source_text.split("component_sum =", 1)[1].split("\n\n", 1)[0]
    check("phase-resolved component sum does not double-count dispersive flux",
          "dispersive" not in component_expression)

    match_index = int(np.argmin(np.abs(eta - 0.10)))
    payload = {
        "schema": "closure-free-total-stress-r1sta4b-v1",
        "coordinate_map": "xi=x, eta=y-h(x), J=1; eta is vertical, not true-normal",
        "momentum_component": "fixed Cartesian x",
        "phase_identity": "RANS: mean advection plus measured Reynolds stress; no dispersive term",
        "wavelength_identity": "dispersive flux emerges from <UW>=<U><W>+<U~W~>",
        "stress_definition": "tau_xeta=-(<viscous momentum flux>+<Reynolds momentum flux>+<U~W~>)",
        "raw_uv_source": "rms_files2.dat column 2 (x,y,uv,uw,vw)",
        "source_sha256": saved_hashes,
        "grid": [int(nx), int(ny)],
        "eta_points": int(eta.size),
        "eta_match": float(eta[match_index]),
        "tau_total_at_match": float(total_stress[match_index]),
        "viscous_at_match": float(-viscous_mean[match_index]),
        "reynolds_at_match": float(-reynolds_mean[match_index]),
        "dispersive_at_match": float(-dispersive_flux[match_index]),
        "max_raw_reproduction_error": float(max(
            np.max(np.abs(W - np.load(REFERENCE)["W"])),
            np.max(np.abs(reynolds_mean - np.load(REFERENCE)["reynolds_contravariant_mean"])),
            np.max(np.abs(viscous_mean - np.load(REFERENCE)["viscous_contravariant_mean"])),
            np.max(np.abs(dispersive_flux - np.load(REFERENCE)["dispersive"])),
        )),
        "checks_passed": int(sum(ok for _, ok in checks)),
        "checks_total": len(checks),
        "status": "PASS" if all(ok for _, ok in checks) else "FAIL",
    }

    arrays = {
        "eta": eta,
        "viscous_stress": -viscous_mean,
        "reynolds_stress": -reynolds_mean,
        "dispersive_stress": -dispersive_flux,
        "total_stress": total_stress,
        "source_mean_sha256": np.array(saved_hashes["mean"]),
        "source_rms1_sha256": np.array(saved_hashes["rms1"]),
        "source_rms2_sha256": np.array(saved_hashes["rms2"]),
        "schema": np.array(payload["schema"]),
    }
    for path in (NODE_NPZ, RESULT_NPZ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
    for path in (NODE_JSON, RESULT_JSON):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nR1-STA-4b: {payload['checks_passed']}/{payload['checks_total']} checks passed")
    print(f"tau_total(eta/H=0.10) = {payload['tau_total_at_match']:+.8e}")
    print(f"wrote {NODE_NPZ.relative_to(ROOT)}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
