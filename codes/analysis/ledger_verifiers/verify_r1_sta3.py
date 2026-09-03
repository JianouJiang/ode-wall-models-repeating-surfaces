#!/usr/bin/env python3
"""Independent raw rebuild for the R1-STA-3 phase-balance closure."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / "codes" / "analysis" / "independent_phase_balance_r1_sta3.py"
RESULT = ROOT / "codes" / "results" / "independent_phase_balance_r1_sta3.npz"
SUMMARY = ROOT / "codes" / "results" / "independent_phase_balance_r1_sta3.json"
RAW_VOLUME = (ROOT / "codes" / "raw_data" / "geometry_driven" /
              "xiao_pehill_parameterized" / "pehill-29-cases-DNS" /
              "alph10-9-3036")
RAW_SURFACE = (ROOT / "codes" / "raw_data" / "geometry_driven" /
               "xiao_pehill_parameterized" / "pehill-5-cases-DNS" /
               "case_1p0" / "dns-data")
HILL_UTIL = (ROOT / "codes" / "raw_data" / "geometry_driven" /
             "xiao_pehill_parameterized" / "utility" /
             "hill-geometry-gereration")
sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402


NU = 1.0 / 5600.0
ETA = np.linspace(0.0, 1.0, 401)
ETA_MATCH = 0.1
PERIOD = 9.0
PRIMARY_PHASE_COUNT = 48
def _epsilon_reference() -> float:
    """The wall-force residual scale, read from the corrected reference object.

    This was a hard-coded literal, and it went stale when the scoring reference
    was withdrawn: a clean verifier was still measuring a clean producer
    against a superseded constant.  Reading it makes the threshold move with
    the evidence instead of with an edit.
    """
    path = (Path(__file__).resolve().parents[3] /
            "codes/results/reference_rebase_headlines_l0_20260825.json")
    return float(json.loads(path.read_text())["epsilon"]["N3_mglet_deposited"]["median"])


EPSILON_REFERENCE = _epsilon_reference()


checks: list[tuple[str, bool]] = []


def check(label: str, condition: bool) -> None:
    checks.append((label, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reshape(table: np.ndarray, column: int, nx: int, ny: int) -> np.ndarray:
    return table[:, column].reshape(ny, nx).T


def derivative(field: np.ndarray, x: np.ndarray) -> np.ndarray:
    vector = field.ndim == 1
    work = field[:, None] if vector else field
    wave = 2.0 * np.pi * np.fft.fftfreq(x.size, d=float(x[1] - x[0]))
    value = np.fft.ifft(1j * wave[:, None] * np.fft.fft(work, axis=0),
                         axis=0).real
    return value[:, 0] if vector else value


def map_volume() -> dict[str, np.ndarray | float]:
    """Rebuild the central 768-grid operator without importing producer code."""
    mean = np.loadtxt(RAW_VOLUME / "mean_files.dat")
    rms1 = np.loadtxt(RAW_VOLUME / "rms_files1.dat")
    rms2 = np.loadtxt(RAW_VOLUME / "rms_files2.dat")
    x, y = np.unique(mean[:, 0]), np.unique(mean[:, 1])
    nx, ny = x.size, y.size
    check("volume source is 768 by 385", (nx, ny) == (768, 385))
    fields = {
        "U": reshape(mean, 2, nx, ny),
        "V": reshape(mean, 3, nx, ny),
        "P": reshape(mean, 5, nx, ny),
        "Rxx": reshape(rms1, 2, nx, ny),
        "Rxy": reshape(rms2, 2, nx, ny),
    }
    h = hill_profile(x.copy())
    mapped = {name: np.empty((nx, ETA.size)) for name in fields}
    p_wall = np.empty(nx)
    speed = np.hypot(fields["U"], fields["V"])
    for i in range(nx):
        k0 = int(np.flatnonzero(speed[i] > 1.0e-10)[0])
        distance = y[k0:k0 + 12] - h[i]
        p_wall[i] = np.polyval(
            np.polyfit(distance, fields["P"][i, k0:k0 + 12], 1), 0.0)
        for name, wall_value in (("U", 0.0), ("V", 0.0),
                                 ("P", p_wall[i]), ("Rxx", 0.0),
                                 ("Rxy", 0.0)):
            mapped[name][i] = np.interp(
                h[i] + ETA, np.r_[h[i], y[k0:]],
                np.r_[wall_value, fields[name][i, k0:]])

    U, V, P, Rxx, Rxy = (mapped[k] for k in
                          ("U", "V", "P", "Rxx", "Rxy"))
    hp = derivative(h, x)
    Ue = np.gradient(U, ETA, axis=1, edge_order=2)
    Ve = np.gradient(V, ETA, axis=1, edge_order=2)
    Us, Vs = derivative(U, x), derivative(V, x)
    Ux, Vx = Us - hp[:, None] * Ue, Vs - hp[:, None] * Ve
    W = V - hp[:, None] * U
    flux_s = U ** 2 + Rxx + P - 2.0 * NU * Ux
    flux_y = U * V + Rxy - NU * (Ue + Vx)
    flux_eta = flux_y - hp[:, None] * flux_s
    fit = (ETA >= 0.4) & (ETA <= 0.9)
    forcing = float(np.polyfit(ETA[fit], np.mean(flux_eta, axis=0)[fit], 1)[0])
    km = int(np.argmin(np.abs(ETA - ETA_MATCH)))

    def integral_derivative(field: np.ndarray) -> np.ndarray:
        return derivative(np.trapezoid(field[:, :km + 1], ETA[:km + 1],
                                       axis=1), x)

    return {
        "x": x,
        "mean": integral_derivative(U ** 2) + U[:, km] * W[:, km],
        "pressure": integral_derivative(P) - hp * P[:, km],
        "reynolds": (integral_derivative(Rxx) + Rxy[:, km] -
                      hp * Rxx[:, km]),
        "molecular": (integral_derivative(-2.0 * NU * Ux) -
                       NU * (Ue[:, km] + Vx[:, km]) +
                       2.0 * hp * NU * Ux[:, km]),
        "body": np.full(nx, -forcing * ETA[km]),
    }


def map_surface() -> dict[str, np.ndarray]:
    """Rebuild physical pressure and molecular force on the 512-grid source."""
    table = np.loadtxt(RAW_SURFACE / "mean_files.dat")
    x, y = np.unique(table[:, 0]), np.unique(table[:, 1])
    nx, ny = x.size, y.size
    check("surface source is 512 by 257", (nx, ny) == (512, 257))
    U = reshape(table, 2, nx, ny)
    V = reshape(table, 3, nx, ny)
    P = reshape(table, 5, nx, ny)
    h = hill_profile(x.copy())
    hp = derivative(h, x)
    speed = np.hypot(U, V)
    p_wall, tau = np.empty(nx), np.empty(nx)
    for i in range(nx):
        k0 = int(np.flatnonzero(speed[i] > 1.0e-6)[0])
        distance_p = y[k0:k0 + 12] - h[i]
        p_wall[i] = np.polyval(
            np.polyfit(distance_p, P[i, k0:k0 + 12], 1), 0.0)
        distance_u = y[k0:k0 + 4] - h[i]
        velocity_u = U[i, k0:k0 + 4]
        tau[i] = NU * np.sum(distance_u * velocity_u) / np.sum(distance_u ** 2)
    return {
        "x": x,
        "pressure": -hp * p_wall,
        "molecular": -(1.0 + hp ** 2) * tau,
        "total": -hp * p_wall - (1.0 + hp ** 2) * tau,
    }


def phase_average(x: np.ndarray, field: np.ndarray, count: int) -> np.ndarray:
    edges = np.linspace(0.0, PERIOD, count + 1)
    result = []
    for index in range(count):
        right = x < edges[index + 1] if index < count - 1 else x <= edges[index + 1]
        result.append(np.mean(field[(x >= edges[index]) & right]))
    return np.asarray(result)


run = subprocess.run([sys.executable, str(PRODUCER)], cwd=ROOT,
                     capture_output=True, text=True, timeout=180,
                     env={**__import__("os").environ,
                          "OMP_NUM_THREADS": "2",
                          "OPENBLAS_NUM_THREADS": "2",
                          "MKL_NUM_THREADS": "2"})
check("fresh producer rebuild", run.returncode == 0 and "STATUS                       : PASS" in run.stdout)
if run.returncode:
    print(run.stdout)
    print(run.stderr)

summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
stored = np.load(RESULT, allow_pickle=False)
volume = map_volume()
surface = map_surface()
blocked = {name: phase_average(volume["x"], volume[name], PRIMARY_PHASE_COUNT)
           for name in ("mean", "pressure", "reynolds", "molecular", "body")}
direct_pressure = phase_average(surface["x"], surface["pressure"],
                                PRIMARY_PHASE_COUNT)
direct_molecular = phase_average(surface["x"], surface["molecular"],
                                 PRIMARY_PHASE_COUNT)
direct = direct_pressure + direct_molecular
parent = sum(blocked.values())
residual = parent - direct
full_legs = np.stack([*blocked.values(), direct])
phase_ratio = float(np.linalg.norm(residual) / np.linalg.norm(full_legs))
wavelength_ratio = float(abs(np.mean(residual)) / abs(np.mean(direct)))

check("four raw source hashes bind current bytes",
      all((ROOT / path).is_file() and sha256(ROOT / path) == value
          for path, value in summary["source_hashes"].items()))
check("independent physical surface is common",
      summary["physical_surface"] == "eta=y-h(x), analytic Xiao hill surface")
check("all five parent terms independently rebuild",
      all(np.allclose(blocked[name], stored[f"{name}_transport"]
                      if name != "body" else stored["body_force"],
                      rtol=0.0, atol=3.0e-12)
          for name in ("mean", "pressure", "reynolds", "molecular", "body")))
check("independent wall pressure and molecular force rebuild",
      np.allclose(direct_pressure, stored["direct_pressure"], rtol=0.0,
                  atol=3.0e-12) and
      np.allclose(direct_molecular, stored["direct_molecular"], rtol=0.0,
                  atol=3.0e-12))
check("stored parent/direct/residual identities",
      np.allclose(parent, stored["parent_total"], rtol=0.0, atol=3.0e-12) and
      np.allclose(direct, stored["direct_total"], rtol=0.0, atol=3.0e-12) and
      np.allclose(residual, stored["residual"], rtol=0.0, atol=3.0e-12))
check("primary phase closure independently below six percent",
      phase_ratio < 0.06 and np.isclose(
          phase_ratio, summary["primary_phase_closure_over_full_leg"],
          rtol=2.0e-12, atol=2.0e-14))
check("wavelength closure independently below six percent",
      wavelength_ratio < 0.06 and np.isclose(
          wavelength_ratio, summary["primary_wavelength_closure_over_direct"],
          rtol=2.0e-12, atol=2.0e-14))
check("complete operator envelope stays below physical epsilon",
      float(stored["closure_operator_ensemble"].max()) < EPSILON_REFERENCE and
      summary["max_phase_closure_over_epsilon"] < 1.0)
check("phase closure precedes wavelength telescoping",
      stored["phase_counts"].shape == (9,) and
      stored["phase_closure_central"].shape == (9,) and
      np.all(np.isfinite(stored["phase_closure_central"])))
check("no modelled or thin-layer reference quantity",
      not summary["uses_modelled_eddy_viscosity"] and
      not summary["uses_boussinesq_reference_stress"] and
      not summary["uses_thin_layer_deletion"] and
      not summary["uses_pressure_height_proxy"] and
      not summary["uses_same_operator_wall_reference"])

# Red fixtures use the same full-leg closure measure as the real record.
def closure(candidate_parent: np.ndarray) -> float:
    return float(np.linalg.norm(candidate_parent - direct) /
                 np.linalg.norm(full_legs))

check("red fixture: omitting mean transport is rejected",
      closure(parent - blocked["mean"]) > EPSILON_REFERENCE)
check("red fixture: pressure sign reversal is rejected",
      closure(parent - 2.0 * blocked["pressure"]) > EPSILON_REFERENCE)
check("red fixture: same-operator wall reference is forbidden",
      summary["uses_same_operator_wall_reference"] is False)

main_tex = ROOT / "manuscript" / "main.tex"
check("stable result declares the R1-STA-3 idea and status",
      summary.get("ledger_row") == "R1-STA-3" and
      summary.get("status") == "PASS" and
      isinstance(summary.get("idea"), str) and
      len(summary["idea"].strip()) > 40)
check("active manuscript states independent phase balance",
      "independent 512" in main_tex.read_text(encoding="utf-8") and
      "phase integration precedes" in main_tex.read_text(encoding="utf-8"))

print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(ok for _, ok in checks) else 1)
