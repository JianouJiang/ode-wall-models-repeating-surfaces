#!/usr/bin/env python3
"""Independent guard for the source-faithful taxonomy and rough-wall test.

The verifier does not import the registry producer.  It independently rebuilds
M0 and Meneveau M5 from the deposited rib profiles, checks all source hashes and
manufactured thresholds, audits the input/term masks, and exercises two red
fixtures that reproduce the reviewer's taxonomy and pressure-direction failures.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
REGISTRY = RESULTS / "source_faithful_model_registry_l1.json"
DATA = RESULTS / "source_faithful_model_registry_l1.npz"
RIB = RESULTS / "rib_les_dtype_wall_profiles.npz"
TEX = ROOT / "manuscript" / "main.tex"
PDF = ROOT / "manuscript" / "main.pdf"
checks: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    checks.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_tex(text: str) -> str:
    kept: list[str] = []
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(r"\iffalse"):
            depth += 1
            continue
        if stripped.startswith(r"\fi") and depth:
            depth -= 1
            continue
        if depth:
            continue
        kept.append(re.sub(r"(?<!\\)%.*$", "", line))
    if depth:
        raise RuntimeError("unclosed inactive TeX block")
    return "\n".join(kept)


def spalding_stress(u_m: float, y_m: float, nu: float) -> float:
    if u_m == 0.0:
        return 0.0
    speed = abs(u_m)

    def residual(u_tau: float) -> float:
        up = speed / u_tau
        ku = 0.41 * up
        if ku > 700.0:
            return -np.inf
        yp_model = up + np.exp(-0.41 * 5.0) * (
            np.exp(ku) - 1.0 - ku - 0.5 * ku * ku - ku ** 3 / 6.0
        )
        return y_m * u_tau / nu - yp_model

    lo = max(np.finfo(float).tiny, speed * 1.0e-10, nu / y_m * 1.0e-10)
    hi = max(speed * 10.0, np.sqrt(speed * nu / y_m) * 100.0, nu / y_m)
    while residual(hi) <= 0.0:
        hi *= 10.0
    u_tau = brentq(residual, lo, hi, xtol=1.0e-13, rtol=1.0e-13)
    return float(np.sign(u_m) * u_tau ** 2)


def m5_manual(u_m: float, delta: float, nu: float, z0: float,
              dp_ds: float) -> float:
    """Independent transcription of Meneveau (2020), equations (7)--(51)."""
    if u_m == 0.0:
        return 0.0
    speed = abs(u_m)
    re_d = speed * delta / nu
    zeta = z0 / delta
    psi = np.sign(u_m) * dp_ds * delta ** 3 / nu ** 2
    if not (0.0 < re_d < 1.0e7 and abs(psi) < 2.0e7
            and 1.0e-5 < zeta < 0.1):
        raise ValueError("outside published M5 domain")

    beta1 = 1.0 / (1.0 + 0.155 * re_d ** (-0.03))
    beta2 = 1.7 - 1.0 / (1.0 + 36.0 * re_d ** (-0.75))
    k3 = 0.005
    refit = (k3 ** (beta1 - 0.5) * re_d ** beta1
             * (1.0 + (k3 * re_d) ** (-beta2))
             ** ((beta1 - 0.5) / beta2))
    if psi < 0.0:
        mag = -psi
        lower = 1.5 * mag ** 0.39 * (1.0 + (1000.0 / mag) ** 2) ** (-0.055)
        power = 2.5 - 0.6 * (1.0 + np.tanh(2.0 * (np.log10(mag) - 6.0)))
        re_pressure = (lower ** power + refit ** power) ** (1.0 / power)
    elif psi > 0.0:
        minimum = 2.5 * psi ** 0.54 * (1.0 + (30.0 / psi) ** 0.5) ** (-0.88)
        re_pressure = (refit * (1.0 - 1.0 /
                       (1.0 + np.log(re_d / minimum)) ** 1.9)
                       if re_d > minimum else 0.0)
    else:
        re_pressure = refit

    xi = 1.0 / np.log(1.0 / zeta)
    big_psi = psi / re_d ** 2
    if big_psi == 0.0:
        theta = 0.4 * xi
    else:
        mag = abs(big_psi)
        alpha = 1.15 * np.sqrt(mag)
        theta1 = (-np.sign(big_psi) * np.sqrt(mag * alpha)
                  * (1.0 + (2.25 * xi / alpha) ** (-1.35)) ** (-1.0 / 1.35))
        if big_psi < 0.0:
            coefficient, xi_m = 0.085 * np.sqrt(mag), 0.95 * np.sqrt(mag)
        else:
            coefficient, xi_m = -0.63 * big_psi ** 1.24, 0.20
        theta2 = coefficient * (xi / xi_m) * np.exp(
            -0.5 * (1.0 - xi / xi_m) ** 2
        )
        theta = max(0.4 * xi + theta1 + theta2, 0.0)
    re_rough = re_d * theta
    re_tau = (re_pressure ** 6 + re_rough ** 6) ** (1.0 / 6.0)
    u_tau = speed * re_tau / re_d
    return float(np.sign(u_m) * u_tau ** 2)


def metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    r2 = 1.0 - np.sum((prediction - reference) ** 2) / np.sum(
        (reference - np.mean(reference)) ** 2
    )
    rel = np.sqrt(np.mean((prediction - reference) ** 2)) / np.sqrt(
        np.mean(reference ** 2)
    )
    mismatch = np.mean(np.signbit(prediction) != np.signbit(reference))
    return float(r2), float(rel), float(mismatch)


def taxonomy_ok(models: list[dict[str, object]], pressure_direction: str) -> bool:
    by_id = {str(model["id"]): model for model in models}
    if list(by_id) != [f"M{i}" for i in range(6)]:
        return False
    terms = {key: set(map(str, value["retained_terms"]))
             for key, value in by_id.items()}
    inputs = {key: set(map(str, value["runtime_inputs"]))
              for key, value in by_id.items()}
    return (
        "streamwise" in pressure_direction.lower()
        and "wall-normal pressure gradient" not in pressure_direction.lower()
        and "dp_ds" not in inputs["M0"]
        and "dp_ds" in inputs["M1"]
        and "modelled convection" in terms["M2"]
        and "integrated momentum flux/history" in terms["M3"]
        and "resolved convection" in terms["M4"]
        and "z0" in inputs["M5"]
        and len({frozenset(value) for value in terms.values()}) == 6
    )


def main() -> int:
    required = [REGISTRY, DATA, RIB, TEX, PDF]
    check("all required artifacts exist", all(path.exists() for path in required))
    if not all(path.exists() for path in required):
        return 2
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with np.load(DATA, allow_pickle=False) as result, np.load(RIB, allow_pickle=False) as rib:
        check("registry and array schemas agree",
              registry.get("schema") == "source-faithful-model-registry-l1-v1"
              and registry.get("status") == "PASS"
              and str(result["schema"]) == registry["schema"])
        models = registry["models"]
        check("ordered M0--M5 taxonomy and term masks are valid",
              taxonomy_ok(models, registry["pressure_direction"]))
        check("every deployable model returns one signed kinematic traction",
              registry["common_output"] == "signed kinematic wall traction tau_w"
              and registry["density_convention"].startswith("kinematic"))
        check("source citations identify all six formulations",
              "10.1115/1.3641728" in models[0]["source"]
              and "10.1016/j.cpc.2019.01.016" in models[1]["source"]
              and "equation (4.1)" in models[2]["source"]
              and "10.1063/1.4908072" in models[3]["source"]
              and "10.1063/1.4861069" in models[4]["source"]
              and "10.1080/14685248.2020.1840573" in models[5]["source"])
        check("all source and raw-data hashes remain current",
              all((ROOT / rel).exists() and sha256(ROOT / rel) == digest
                  for rel, digest in registry["source_hashes"].items())
              and str(result["rib_source_sha256"]) == sha256(RIB))
        names = list(map(str, result["manufactured_names"]))
        errors = np.asarray(result["manufactured_errors"], dtype=float)
        thresholds = np.asarray(result["manufactured_thresholds"], dtype=float)
        check("all eight manufactured/operator checks clear registered thresholds",
              len(names) == 8 and np.all(np.isfinite(errors))
              and np.all(errors <= thresholds))

        index = int(result["rib_matching_index"])
        z0 = float(result["rib_z0"])
        y_m = np.asarray(rib["y"][:, index], dtype=float)
        u_m = np.asarray(rib["U"][:, index], dtype=float)
        dp_ds = np.asarray(rib["dp_dx"], dtype=float)
        reference = np.asarray(rib["tau_w"], dtype=float)
        nu = float(rib["nu"])
        check("rough-wall protocol is target-free and admits all 48 WRLES stations",
              index == 15 and abs(z0 - 0.2 / 30.0) < 1.0e-15
              and reference.size == 48
              and np.all((z0 / y_m > 1.0e-5) & (z0 / y_m < 0.1))
              and registry["rough_wall_real_data_test"]["target_used_to_set_z0"] is False)

        rebuilt_m0 = np.asarray([spalding_stress(uu, yy, nu)
                                 for uu, yy in zip(u_m, y_m)])
        rebuilt_m5 = np.asarray([m5_manual(uu, yy, nu, z0, pp)
                                 for uu, yy, pp in zip(u_m, y_m, dp_ds)])
        saved_m0 = np.asarray(result["rib_tau_M0"], dtype=float)
        saved_m5 = np.asarray(result["rib_tau_M5"], dtype=float)
        check("independent Spalding rebuild matches every stored station",
              np.allclose(rebuilt_m0, saved_m0, rtol=2.0e-12, atol=2.0e-14))
        check("independent Meneveau equations (7)--(51) match every stored station",
              np.allclose(rebuilt_m5, saved_m5, rtol=2.0e-12, atol=2.0e-14))
        r2, rel, mismatch = metrics(reference, rebuilt_m5)
        reported = registry["rough_wall_real_data_test"]["metrics"]["M5"]
        check("real-data rough-wall score is independently reproduced",
              abs(r2 - reported["r2_descriptive"]) < 1.0e-12
              and abs(rel - reported["relative_rms"]) < 1.0e-12
              and abs(mismatch - reported["sign_mismatch_fraction"]) < 1.0e-15
              and r2 < -50.0 and rel > 6.0)

    red_models = json.loads(json.dumps(registry["models"]))
    red_models[2]["retained_terms"].remove("modelled convection")
    check("control case rejects M2 relabelled as a second M1",
          not taxonomy_ok(red_models, registry["pressure_direction"]))
    check("control case rejects the reviewer-caught wall-normal pressure label",
          not taxonomy_ok(registry["models"], "wall-normal pressure gradient"))

    source = active_tex(TEX.read_text(encoding="utf-8"))
    compact_source = re.sub(r"\s+", " ", source)
    check("active Methods prints the source-locked M0--M5 comparison",
          "Source-locked model comparison" in source
          and all(f"M{index}" in source for index in range(6))
          and "streamwise pressure gradient" in compact_source
          and "wall-normal pressure gradient" not in compact_source)
    check("active source prints the real 48-station rough-wall result",
          "48-station" in source and "-51.93" in source
          and "meneveau2020moody" in source)

    pdf_run = subprocess.run(["pdftotext", str(PDF), "-"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    compiled = re.sub(r"\s+", " ", pdf_run.stdout)
    pages = int(subprocess.run(["pdfinfo", str(PDF)], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout
                .split("Pages:", 1)[1].splitlines()[0].strip())
    check("compiled PDF carries the registry and rough-wall result within 20 pages",
          "Source-locked model comparison" in compiled
          and ("48-station" in compiled or "48station" in compiled)
          and "−51.93" in compiled
          and pages <= 20 and PDF.stat().st_mtime_ns >= TEX.stat().st_mtime_ns)

    failed = [name for name, passed in checks if not passed]
    print(f"R1-SCI-2/M12: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
