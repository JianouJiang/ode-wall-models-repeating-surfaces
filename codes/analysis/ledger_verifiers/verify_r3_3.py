#!/usr/bin/env python3
"""Raw-rebuilding verifier for claim R3-3 (thin-layer deletions)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RAW = (ROOT / "codes" / "raw_data" / "geometry_driven" /
       "xiao_pehill_parameterized" / "pehill-29-cases-DNS" /
       "alph10-9-3036")
RESULT = ROOT / "codes" / "results" / "full_rans_thin_layer_audit.npz"
SUMMARY = ROOT / "codes" / "results" / "physical_face_force_migration_summary.json"
CONTRACT = ROOT / "codes" / "results" / "physical_face_operator_contract.json"
LEDGER = ROOT / "REFEREE_POINT_LEDGER.md"
MAIN = ROOT / "manuscript" / "main.tex"
HILL_UTIL = (ROOT / "codes" / "raw_data" / "geometry_driven" /
             "xiao_pehill_parameterized" / "utility" /
             "hill-geometry-gereration")
sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402


NU = 1.0 / 5600.0
ETA = np.linspace(0.0, 1.0, 401)
HEIGHTS = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
NAMES = ("mean_convection", "pressure_gradient", "reynolds_streamwise",
         "reynolds_normal", "viscous_streamwise", "viscous_normal")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reshape(table: np.ndarray, column: int, nx: int, ny: int) -> np.ndarray:
    return table[:, column].reshape(ny, nx).T


def ddx(field: np.ndarray, x: np.ndarray, max_mode: int | None) -> np.ndarray:
    one_dimensional = field.ndim == 1
    work = field[:, None] if one_dimensional else field
    wave = 2.0 * np.pi * np.fft.fftfreq(x.size, d=float(x[1] - x[0]))
    modes = np.fft.fftfreq(x.size) * x.size
    spectrum = np.fft.fft(work, axis=0)
    if max_mode is not None:
        spectrum[np.abs(modes) > max_mode, :] = 0.0
    derivative = np.fft.ifft(1j * wave[:, None] * spectrum, axis=0).real
    return derivative[:, 0] if one_dimensional else derivative


def raw_fields() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    mean_path = RAW / "mean_files.dat"
    rms1_path = RAW / "rms_files1.dat"
    rms2_path = RAW / "rms_files2.dat"
    mean = np.loadtxt(mean_path)
    rms1 = np.loadtxt(rms1_path)
    rms2 = np.loadtxt(rms2_path)
    if not (mean.shape[1] == 6 and rms1.shape[1] == 6 and rms2.shape[1] == 5):
        raise RuntimeError("unexpected raw column schema")
    x = np.unique(mean[:, 0])
    y = np.unique(mean[:, 1])
    nx, ny = x.size, y.size
    fields = {
        "x": x,
        "y": y,
        "U": reshape(mean, 2, nx, ny),
        "V": reshape(mean, 3, nx, ny),
        "P": reshape(mean, 5, nx, ny),
        "Rxx": reshape(rms1, 2, nx, ny),
        "Rxy": reshape(rms2, 2, nx, ny),
    }
    hashes = {"mean": sha256(mean_path), "rms1": sha256(rms1_path),
              "rms2": sha256(rms2_path)}
    return fields, hashes


def interpolate(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x, y = fields["x"], fields["y"]
    h = hill_profile(x.copy())
    speed = np.hypot(fields["U"], fields["V"])
    out = {name: np.empty((x.size, ETA.size)) for name in
           ("U", "V", "P", "Rxx", "Rxy")}
    for i in range(x.size):
        fluid = np.flatnonzero(speed[i] > 1.0e-10)
        if fluid.size < 12:
            raise RuntimeError("insufficient fluid nodes")
        k0 = int(fluid[0])
        yy = y[k0:]
        pressure_fit = np.polyfit(yy[:12] - h[i], fields["P"][i, k0:k0 + 12], 1)
        wall_pressure = np.polyval(pressure_fit, 0.0)
        for name, wall_value in (("U", 0.0), ("V", 0.0),
                                 ("P", wall_pressure), ("Rxx", 0.0),
                                 ("Rxy", 0.0)):
            out[name][i] = np.interp(h[i] + ETA, np.r_[h[i], yy],
                                     np.r_[wall_value, fields[name][i, k0:]])
    out["h"] = h
    out["x"] = x
    return out


def rebuild(mode: int | None, mapped: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = mapped["x"]
    hp = ddx(mapped["h"], x, mode)

    def de(field: np.ndarray) -> np.ndarray:
        return np.gradient(field, ETA, axis=1, edge_order=2)

    U, V, P, Rxx, Rxy = (mapped[key] for key in
                          ("U", "V", "P", "Rxx", "Rxy"))
    Ueta = de(U)
    Ux = ddx(U, x, mode) - hp[:, None] * Ueta
    terms = (
        U * Ux + V * Ueta,
        ddx(P, x, mode) - hp[:, None] * de(P),
        ddx(Rxx, x, mode) - hp[:, None] * de(Rxx),
        de(Rxy),
        -NU * (ddx(Ux, x, mode) - hp[:, None] * de(Ux)),
        -NU * de(Ueta),
    )
    table = np.empty((HEIGHTS.size, len(NAMES)))
    reynolds_fraction = np.empty(HEIGHTS.size)
    viscous_ratio = np.empty(HEIGHTS.size)
    for ih, height in enumerate(HEIGHTS):
        k = int(np.argmin(np.abs(ETA - height)))
        station = [np.trapezoid(term[:, :k + 1], ETA[:k + 1], axis=1)
                   for term in terms]
        table[ih] = [np.sqrt(np.mean(value ** 2)) for value in station]
        reynolds_fraction[ih] = table[ih, 2] / max(table[ih, 0], table[ih, 1], table[ih, 3])
        viscous_ratio[ih] = table[ih, 4] / table[ih, 5]
    return table, reynolds_fraction, viscous_ratio


checks: list[tuple[str, bool]] = []


def check(label: str, condition: bool) -> None:
    checks.append((label, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")


fields, hashes = raw_fields()
mapped = interpolate(fields)
with np.load(RESULT, allow_pickle=False) as saved:
    check("raw mean hash", str(saved["source_mean_sha256"]) == hashes["mean"])
    check("raw normal-stress hash", str(saved["source_rms1_sha256"]) == hashes["rms1"])
    check("raw shear-stress hash", str(saved["source_rms2_sha256"]) == hashes["rms2"])
    check("full conservative term dictionary",
          tuple(saved["term_names"].astype(str)) == NAMES)
    check("six registered matching heights", np.allclose(saved["heights_over_H"], HEIGHTS))

    for saved_index, mode in ((0, 96), (5, None)):
        table, rfrac, vratio = rebuild(mode, mapped)
        check(f"raw rebuild integrated terms, mode {mode}",
              np.allclose(table, saved["integrated_term_rms"][saved_index],
                          rtol=2.0e-12, atol=2.0e-13))
        check(f"raw rebuild streamwise Reynolds ratios, mode {mode}",
              np.allclose(rfrac, saved["reynolds_streamwise_fraction"][saved_index],
                          rtol=2.0e-12, atol=2.0e-13))
        check(f"raw rebuild streamwise viscous ratios, mode {mode}",
              np.allclose(vratio, saved["viscous_streamwise_to_normal_ratio"][saved_index],
                          rtol=2.0e-12, atol=2.0e-13))

    check("streamwise Reynolds term is not asymptotically negligible",
          float(np.min(saved["reynolds_streamwise_fraction"])) > 0.20)
    check("streamwise viscous term is not asymptotically negligible",
          float(np.min(saved["viscous_streamwise_to_normal_ratio"])) > 0.20)

summary = json.loads(SUMMARY.read_text())
contract = json.loads(CONTRACT.read_text())
check("summary agrees with raw lower bound for Reynolds term",
      summary["thin_layer_reynolds_streamwise_fraction_min"] > 0.20)
check("summary agrees with raw lower bound for viscous term",
      summary["thin_layer_viscous_streamwise_to_normal_min"] > 0.20)
check("parent method deletes no thin-layer term", contract["thin_layer_terms_deleted"] == "none")

ledger = LEDGER.read_text()
main = MAIN.read_text()
check("claim R3-3 is closed by this command",
      "**R3-3**" in ledger and
      "verify_r3_3.py" in ledger and
      "**CLOSED 2026-08-21**" in ledger)
check("manuscript states the full conservative balance",
      "partial_x R_{xx}" in main or "partial_xR_{xx}" in main)
check("manuscript prints measured streamwise-term ranges",
      "0.263" in main and "0.484" in main and "0.304" in main and "0.483" in main)

passed = sum(ok for _, ok in checks)
print(f"{passed}/{len(checks)} checks passed")
if passed != len(checks):
    raise SystemExit(1)
