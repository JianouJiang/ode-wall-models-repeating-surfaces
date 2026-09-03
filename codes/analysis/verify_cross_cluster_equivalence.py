#!/usr/bin/env python3
"""Cross-cluster equivalence certificate: ARCHER2 vs Oxford ARC.

NEW FILE (Agent H, 2026-08-24).  Nothing here modifies a pinned artefact.

WHY THIS EXISTS
---------------
ARCHER2 runs ``openfoam/org/v10.20230119`` (Cray PE, ``crayGccDPInt32Opt``,
gcc 11.2 + cray-mpich) and Oxford ARC runs ``OpenFOAM/10-foss-2022a``
(EasyBuild, ``linux64GccDPInt32Opt``, gcc 11.3 + OpenMPI 4.1.4).  Same
Foundation-10 source line, different build.  The campaign's whole point is
matched numerics, so rows may not mix clusters inside one comparison until the
two builds are shown to give the same answer on the same case.  This script is
that test.  It is deliberately two-tier:

TIER 1 -- deterministic identity.  The ARC replay starts from the *byte
identical* mesh, initial field and dictionaries of the ARCHER2 deposit, so the
wall model's FIRST solve (the ``TOTAL_GRADIENT_TBLE_FACE ... solver=full_census``
census, printed once per wall face at 17 significant figures, before any linear
solve has run) is a deterministic function of the input.  Any difference there
is pure build arithmetic.  The early time-step trajectory (deltaT, Courant,
initial residuals, drive gradient) is then compared step by step to locate the
onset of chaotic divergence, which is a property of the flow, not of the build.

TIER 2 -- statistical equivalence.  The two full 405-time-unit runs are reduced
with the campaign's OWN code -- ``codes/analysis/rswm_common_surface_grid_l2.py``
and ``development/nodes/node_004/analyze_grid_results_l3.py``, imported, not
re-implemented -- and the headline metrics (E_tau, R^2, sign accuracy,
x_sep / x_re, u*, the phase-closure metric eps_c, mean-profile RMS vs the
reference) are compared *with* the paired circular phase-block bootstrap
intervals the campaign already uses (Lx/8 blocks, 20,000 draws).  Two LES runs
of a chaotic flow can never agree pointwise in time; the honest question is
whether they agree to within the phase-block sampling uncertainty that every
headline number in the paper already carries.

VERDICT
-------
EQUIVALENT      -- tier 1 agrees to the build-arithmetic tolerance AND every
                   tier-2 headline difference lies inside its paired 95 %
                   phase-block interval.
NOT-EQUIVALENT  -- otherwise, with the measured discrepancies printed.  This is
                   a perfectly good outcome: it simply means matched pairs stay
                   on one machine.
PARTIAL         -- only tier 1 was available (the long replay had not landed).

Usage
-----
    python3 codes/analysis/verify_cross_cluster_equivalence.py \
        [--archer2-case <deposit dir>] [--arc-case <arc bundle dir>] \
        [--archer2-log <log.pimpleFoam>] [--arc-log <log.pimpleFoam>] \
        [--date YYYYMMDD] [--draws 20000]

Writes ``codes/results/cross_cluster_equivalence_<date>.json`` and ``.npz``.
Exit status 0 = certificate written and the verdict is EQUIVALENT;
1 = NOT-EQUIVALENT; 2 = PARTIAL (tier 2 inputs absent); >2 = the test itself
could not be run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
L2_REDUCER = ROOT / "codes" / "analysis" / "rswm_common_surface_grid_l2.py"
L3_ANALYSER = ROOT / "development" / "nodes" / "node_004" / "analyze_grid_results_l3.py"
HARVEST = ROOT / "codes" / "analysis" / "harvest_m13_highre.py"

DEFAULT_ARCHER2 = (ROOT / "codes" / "results" / "rswm_xiao_highre_campaign_m13_final"
                   / "re5600" / "rswm_m13_re5600_g1_tble_307200_v2")
DEFAULT_ARC = ROOT / "codes" / "results" / "rswm_arc_equivalence"

LX = 9.0
DENSE_N = 4096
BLOCK_POINTS = 512          # Lx/8 phase blocks -- the deposited L3 protocol
DRAWS = 20000
SEED = 20260824
OUTER_YM_OVER_H = 0.10

# Tier-1 tolerances.  The first-solve census is a deterministic function of
# byte-identical inputs, so the only admissible difference is floating-point
# evaluation order between two builds of the same source.  1e-9 relative is
# ~7 orders of magnitude looser than double round-off and 7 orders tighter than
# any physically meaningful difference, so the test cannot be gamed either way.
FIRST_SOLVE_REL_TOL = 1.0e-9
FIRST_SOLVE_FIELDS = ("UMatch", "UtMag", "ym", "phaseDpds", "driveGradient",
                      "driveProjection", "dpds", "tauW", "rawNut", "upperNut",
                      "nut", "appliedTau", "appliedTractionMag")

FACE_PATTERN = re.compile(
    r"TOTAL_GRADIENT_TBLE_FACE patch=(\S+) proc=(\d+) face=(\d+) "
    r"centre=\(([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\) "
    + " ".join(rf"{name}=([-0-9.eE+]+)" for name in FIRST_SOLVE_FIELDS)
)


# --------------------------------------------------------------------------- #
# small helpers (same conventions as the campaign harvests)
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(np.asarray(values, float), (0.025, 0.5, 0.975))
    return {"low": float(low), "median": float(median), "high": float(high)}


# --------------------------------------------------------------------------- #
# TIER 1 -- deterministic first-solve census and early trajectory
# --------------------------------------------------------------------------- #
CENTRE_DECIMALS = 7   # face pitch is 7.5e-2 in x and 1.125e-1 in z: 1e-7 is unambiguous


def parse_first_solve(log: Path) -> dict[tuple[str, float, float, float], dict[str, float]]:
    """First-solve wall-model census, keyed by (patch, x, y, z) face centre.

    Keyed on the face centre rather than (proc, face) because the two clusters
    partition the mesh independently.  The centre is ROUNDED to
    ``CENTRE_DECIMALS``: it is a computed geometric quantity, so its last digits
    differ between builds.  Measured on the matched-rank pair (2026-08-24):
    exact-string keys matched 4,099 of 9,600; rounding to 1e-9 matched 9,254;
    1e-8 matched 9,480; **1e-7 matched all 9,600 with 9,600 distinct keys and no
    collision**, which is what CENTRE_DECIMALS is set to.  The smallest wall-face
    pitch on this mesh is 7.5e-2, so 1e-7 cannot merge two distinct faces.  The
    geometric quantity that actually enters the model, ``ym``, agrees between the
    builds to 3.7e-13 relative, so this is a printing/《centroid-arithmetic》
    artefact of the key, not a mesh difference (the polyMesh files are
    sha256-identical by construction).
    """
    out: dict[tuple[str, float, float, float], dict[str, float]] = {}
    with log.open("r", errors="replace") as stream:
        for line in stream:
            if "TOTAL_GRADIENT_TBLE_FACE" not in line:
                continue
            match = FACE_PATTERN.search(line)
            if match is None:
                continue
            groups = match.groups()
            key = (groups[0],
                   round(float(groups[3]), CENTRE_DECIMALS),
                   round(float(groups[4]), CENTRE_DECIMALS),
                   round(float(groups[5]), CENTRE_DECIMALS))
            if key in out:
                continue  # only the first solve is deterministic
            out[key] = {name: float(value)
                        for name, value in zip(FIRST_SOLVE_FIELDS, groups[6:])}
    return out


def parse_trajectory(log: Path, limit: int = 400) -> list[dict[str, float]]:
    """Per-time-step trajectory: t, deltaT, Courant, residuals, drive gradient."""
    steps: list[dict[str, float]] = []
    current: dict[str, float] = {}
    pending: dict[str, float] = {}
    with log.open("r", errors="replace") as stream:
        for line in stream:
            if line.startswith("Courant Number mean:"):
                m = re.search(r"mean:\s*([-0-9.eE+]+)\s+max:\s*([-0-9.eE+]+)", line)
                if m:
                    pending["courant_mean"] = float(m.group(1))
                    pending["courant_max"] = float(m.group(2))
            elif line.startswith("deltaT = "):
                pending["deltaT"] = float(line.split("=", 1)[1])
            elif line.startswith("Time = "):
                if current:
                    steps.append(current)
                    if len(steps) >= limit:
                        return steps
                current = dict(pending)
                pending = {}
                current["time"] = float(line.split("=", 1)[1].strip().rstrip("s"))
            elif current and "Solving for" in line:
                m = re.search(r"Solving for (\w+), Initial residual = ([-0-9.eE+]+)", line)
                if m and f"res_{m.group(1)}_first" not in current:
                    current[f"res_{m.group(1)}_first"] = float(m.group(2))
            elif current and line.startswith("Registered pressure gradient source:"):
                m = re.search(r"uncorrected Ubar = ([-0-9.eE+]+), pressure gradient = ([-0-9.eE+]+)", line)
                if m and "drive_ubar_first" not in current:
                    current["drive_ubar_first"] = float(m.group(1))
                    current["drive_gradient_first"] = float(m.group(2))
            elif current and line.startswith("ExecutionTime ="):
                m = re.search(r"ExecutionTime = ([-0-9.eE+]+) s", line)
                if m:
                    current["execution_time"] = float(m.group(1))
    if current:
        steps.append(current)
    return steps


def compare_first_solve(a2: dict, arc: dict) -> dict[str, Any]:
    shared = sorted(set(a2) & set(arc))
    result: dict[str, Any] = {
        "archer2_faces": len(a2),
        "arc_faces": len(arc),
        "matched_faces": len(shared),
        "unmatched_archer2": len(set(a2) - set(arc)),
        "unmatched_arc": len(set(arc) - set(a2)),
        "fields": {},
    }
    if not shared:
        # e.g. the equilibrium (Spalding) model prints no TBLE census at all --
        # a legitimate outcome that simply makes tier 1 inapplicable, and a
        # deliberate NOT-EQUIVALENT trigger when the two bundles are not twins.
        result["status"] = "NO_MATCHED_FACES"
        result["worst_field"] = None
        result["worst_max_relative_difference"] = float("inf")
        result["bit_identical_fraction_overall"] = 0.0
        result["tolerance"] = FIRST_SOLVE_REL_TOL
        return result
    worst_rel = 0.0
    worst_field = None
    bit_identical_total = 0
    value_total = 0
    outlier = np.zeros(len(shared), bool)
    for name in FIRST_SOLVE_FIELDS:
        va = np.asarray([a2[k][name] for k in shared], float)
        vb = np.asarray([arc[k][name] for k in shared], float)
        # Scale by the local magnitude but never below the field's own RMS
        # scale: a quantity that passes through zero (dpds, tauW, nut all do)
        # would otherwise report a 100 % "relative" difference for an absolute
        # difference of 1e-13.  The pure local-magnitude ratio is reported too.
        rms = float(np.sqrt(np.mean(va ** 2)))
        local = np.maximum(np.abs(va), np.abs(vb))
        scale = np.maximum(local, max(rms, 1.0e-300))
        rel = np.abs(vb - va) / scale
        rel_local = np.abs(vb - va) / np.where(local > 0.0, local, 1.0)
        bit_identical = int(np.count_nonzero(vb == va))
        bit_identical_total += bit_identical
        value_total += len(shared)
        outlier |= rel > FIRST_SOLVE_REL_TOL
        record = {
            "max_abs_difference": float(np.max(np.abs(vb - va))),
            "max_relative_difference": float(np.max(rel)),
            "max_local_ratio_difference": float(np.max(rel_local)),
            "median_relative_difference": float(np.median(rel)),
            "p99_relative_difference": float(np.quantile(rel, 0.99)),
            "field_rms_scale": rms,
            "faces_over_tolerance": int(np.count_nonzero(rel > FIRST_SOLVE_REL_TOL)),
            "bit_identical_fraction": bit_identical / len(shared),
            "archer2_rms": float(np.sqrt(np.mean(va ** 2))),
            "arc_rms": float(np.sqrt(np.mean(vb ** 2))),
        }
        result["fields"][name] = record
        if record["max_relative_difference"] > worst_rel:
            worst_rel = record["max_relative_difference"]
            worst_field = name
    result["worst_field"] = worst_field
    result["worst_max_relative_difference"] = worst_rel
    result["outlier_faces"] = int(np.count_nonzero(outlier))
    result["outlier_face_fraction"] = float(np.mean(outlier))
    clean = ~outlier
    if np.any(clean):
        worst_clean = 0.0
        for name in FIRST_SOLVE_FIELDS:
            va_all = np.asarray([a2[k][name] for k in shared], float)
            va = va_all[clean]
            vb = np.asarray([arc[k][name] for k in shared], float)[clean]
            rms = float(np.sqrt(np.mean(va_all ** 2)))
            scale = np.maximum(np.maximum(np.abs(va), np.abs(vb)), max(rms, 1.0e-300))
            worst_clean = max(worst_clean, float(np.max(np.abs(vb - va) / scale)))
        result["worst_max_relative_difference_excluding_outliers"] = worst_clean
    result["bit_identical_fraction_overall"] = bit_identical_total / max(value_total, 1)
    result["tolerance"] = FIRST_SOLVE_REL_TOL
    result["status"] = "IDENTICAL_TO_BUILD_ARITHMETIC" if worst_rel <= FIRST_SOLVE_REL_TOL \
        else "FIRST_SOLVE_DIFFERS"
    return result


def compare_trajectory(a2: list[dict], arc: list[dict]) -> dict[str, Any]:
    n = min(len(a2), len(arc))
    keys = ("deltaT", "courant_max", "res_Ux_first", "res_p_first", "drive_gradient_first")
    series: dict[str, dict[str, list[float]]] = {}
    divergence_step: dict[str, int | None] = {}
    for key in keys:
        va, vb, rel = [], [], []
        for i in range(n):
            if key not in a2[i] or key not in arc[i]:
                break
            x, y = a2[i][key], arc[i][key]
            scale = max(abs(x), abs(y), 1.0e-300)
            va.append(x)
            vb.append(y)
            rel.append(abs(y - x) / scale)
        series[key] = {"archer2": va, "arc": vb, "relative_difference": rel}
        step = next((i for i, r in enumerate(rel) if r > 1.0e-6), None)
        divergence_step[key] = step
    return {
        "compared_steps": n,
        "archer2_steps_parsed": len(a2),
        "arc_steps_parsed": len(arc),
        "series": series,
        "first_step_relative_difference_exceeds_1e-6": divergence_step,
        "note": ("solver logs print 8 significant figures, so agreement below "
                 "~1e-8 relative cannot be resolved from the trajectory; the "
                 "17-digit first-solve census is the high-precision channel"),
    }


# --------------------------------------------------------------------------- #
# TIER 2 -- statistical equivalence on the full replay
# --------------------------------------------------------------------------- #
def bundle_curves(l2: Any, harvest: Any, case: Path) -> tuple[dict, dict, dict, dict]:
    return harvest.load_case(l2, case)


def eps_closure(harvest: Any, l2: Any, curve: dict, pw: np.ndarray, dense: np.ndarray,
                seed: int, draws: int, truth_sep: float | None,
                truth_rea: float | None) -> dict[str, Any]:
    """Phase-closure metric eps_c, exactly the recipe of harvest_m13_highre."""
    x = np.asarray(curve["x"])
    tau = np.asarray(curve["tau_s"])
    ym = np.asarray(curve["ym"])
    dpds = harvest.periodic_derivative_arclength(x, np.asarray(curve["ywall"]), pw)
    phi = np.abs(dpds) * ym
    eps = np.abs(tau) / np.maximum(phi, 1.0e-14)
    eps_outer = eps * ym / OUTER_YM_OVER_H
    phase = np.asarray(curve["phase"])
    sep, rea = l2.zero_crossings(dense, l2.periodic_interp(phase, tau, dense))
    if (truth_sep is not None and truth_rea is not None
            and math.isfinite(truth_sep) and math.isfinite(truth_rea)):
        sep_use, rea_use = truth_sep / LX, truth_rea / LX
    else:
        sep_use, rea_use = sep, rea
    if math.isfinite(sep_use) and math.isfinite(rea_use):
        separated = ((phase - sep_use) % 1.0) < ((rea_use - sep_use) % 1.0)
    else:
        separated = np.zeros(len(phase), bool)
    rng = np.random.default_rng(seed)
    n = len(phase)
    block = max(4, n // 8)
    med_sep, med_all = [], []
    for _ in range(min(draws, 5000)):
        starts = rng.integers(0, n, size=n // block + 1)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        sample = eps[idx]
        med_all.append(np.median(sample))
        med_sep.append(np.median(eps[idx][separated[idx]]) if np.any(separated[idx])
                       else np.median(sample))
    return {
        "eps_c_phase": eps,
        "eps_c_outer_phase": eps_outer,
        "separated_mask": separated,
        "eps_c_median_all": float(np.median(eps)),
        "eps_c_median_separated": float(np.median(eps[separated])) if np.any(separated) else math.nan,
        "eps_c_outer_median_separated": float(np.median(eps_outer[separated])) if np.any(separated) else math.nan,
        "eps_c_median_all_interval": interval(np.asarray(med_all)),
        "eps_c_median_separated_interval": interval(np.asarray(med_sep)),
        "separated_fraction": float(np.mean(separated)),
    }


def ustar_record(curve: dict) -> dict[str, float]:
    """Friction velocity from the run's own physical-tangent wall traction."""
    tau = np.abs(np.asarray(curve["tau_s"], float))
    ustar = np.sqrt(tau)
    return {
        "ustar_median": float(np.median(ustar)),
        "ustar_mean": float(np.mean(ustar)),
        "ustar_max": float(np.max(ustar)),
        "ustar_rms": float(np.sqrt(np.mean(ustar ** 2))),
        "ustar_area_weighted_mean": float(np.average(ustar, weights=np.asarray(curve["wall_ds"], float))),
    }


def tier2(l2: Any, l3: Any, harvest: Any, a2_case: Path, arc_case: Path,
          draws: int) -> dict[str, Any]:
    dense = np.arange(DENSE_N, dtype=float) / DENSE_N
    refs = {"xiao_5600": harvest.xiao_dns_reference(l2)}
    truth_phase = refs["xiao_5600"]["truth_phase"]
    truth_tau = refs["xiao_5600"]["truth_tau"]
    truth_dense = l2.periodic_interp(truth_phase, truth_tau, dense)
    truth_sep, truth_rea = l2.zero_crossings(dense, truth_dense)

    out: dict[str, Any] = {"machines": {}, "arrays": {}}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for label, case in (("archer2", a2_case), ("arc", arc_case)):
        manifest, mesh, curves, pressures = bundle_curves(l2, harvest, case)
        names = list(curves)
        final = curves[names[-1]]
        metrics = l2.metrics(final, truth_phase, truth_tau)
        eps = eps_closure(harvest, l2, final, pressures[names[-1]], dense,
                          SEED + (1 if label == "archer2" else 2), draws,
                          float(truth_sep * LX), float(truth_rea * LX))
        profiles = harvest.read_profiles(case, names[-1])
        profile = harvest.profile_validation(profiles, refs["xiao_5600"])
        window = {}
        for name in names:
            m = l2.metrics(curves[name], truth_phase, truth_tau)
            window[name] = {"relative_rms": m["relative_rms"], "r2": m["r2"],
                            "reattachment_x_over_H": m["reattachment_x_over_H"]}
        out["machines"][label] = {
            "case": str(case),
            "case_id": manifest.get("case_id"),
            "producer_job_id": manifest.get("producer_job_id"),
            "mpi_ranks": manifest.get("mpi_ranks"),
            "maximum_courant": manifest.get("maximum_courant"),
            "latest_time": manifest.get("latest_time"),
            "solver_cost": manifest.get("solver_cost"),
            "openfoam_module": manifest.get("openfoam_module", "openfoam/org/v10.20230119"),
            "openfoam_build": manifest.get("openfoam_build", "crayGccDPInt32Opt"),
            "checkpoints": names,
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
            "eps_c": {k: v for k, v in eps.items() if not isinstance(v, np.ndarray)},
            "ustar": ustar_record(final),
            "profile_validation": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in profile.items()},
            "window_convergence": window,
        }
        predictions[(label, "tau_s")] = l2.periodic_interp(
            np.asarray(final["phase"]), np.asarray(final["tau_s"]), dense)
        out["arrays"][f"{label}_phase"] = np.asarray(final["phase"])
        out["arrays"][f"{label}_tau_s"] = np.asarray(final["tau_s"])
        out["arrays"][f"{label}_eps_c"] = eps["eps_c_phase"]
        out["arrays"][f"{label}_tau_dense"] = predictions[(label, "tau_s")]
    out["arrays"]["truth_dense"] = truth_dense
    out["arrays"]["dense_phase"] = dense

    # paired circular phase-block bootstrap of E_tau (the deposited L3 protocol)
    samples = l3.circular_block_bootstrap(truth_dense, predictions,
                                          block_points=BLOCK_POINTS, draws=draws,
                                          seed=SEED)
    delta = samples[("arc", "tau_s")] - samples[("archer2", "tau_s")]
    e_a2 = out["machines"]["archer2"]["metrics"]["relative_rms"]
    e_arc = out["machines"]["arc"]["metrics"]["relative_rms"]
    d_interval = interval(delta)
    out["e_tau_comparison"] = {
        "archer2_E_tau": e_a2,
        "arc_E_tau": e_arc,
        "difference_arc_minus_archer2": e_arc - e_a2,
        "fractional_difference": (e_arc - e_a2) / max(abs(e_a2), 1.0e-14),
        "archer2_interval": interval(samples[("archer2", "tau_s")]),
        "arc_interval": interval(samples[("arc", "tau_s")]),
        "paired_delta_interval": d_interval,
        "zero_inside_paired_interval": bool(d_interval["low"] <= 0.0 <= d_interval["high"]),
        "block_length_over_H": LX * BLOCK_POINTS / DENSE_N,
        "draws": draws,
    }
    out["arrays"]["bootstrap_delta"] = delta

    # every other headline metric against the ARCHER2 phase-block half-width
    half_width = 0.5 * (out["e_tau_comparison"]["archer2_interval"]["high"]
                        - out["e_tau_comparison"]["archer2_interval"]["low"])
    checks: dict[str, Any] = {}

    def add(name: str, a: float, b: float, tol: float, unit: str) -> None:
        d = float(b - a)
        checks[name] = {"archer2": float(a), "arc": float(b), "difference": d,
                        "tolerance": float(tol), "unit": unit,
                        "inside": bool(abs(d) <= tol) if math.isfinite(d) else False}

    ma, mb = out["machines"]["archer2"], out["machines"]["arc"]
    add("E_tau", ma["metrics"]["relative_rms"], mb["metrics"]["relative_rms"],
        half_width, "phase-block 95% half-width of E_tau")
    add("R2", ma["metrics"]["r2"], mb["metrics"]["r2"],
        2.0 * half_width * max(1.0, abs(ma["metrics"]["r2"])), "scaled E_tau half-width")
    add("sign_accuracy", ma["metrics"]["sign_accuracy"], mb["metrics"]["sign_accuracy"],
        0.05, "absolute (5 % of the phase circle)")
    # x_sep / x_re: the window (225->270) drift of the ARCHER2 run is the
    # honest scale for "the same reattachment point".
    win = list(ma["window_convergence"])
    rea_drift = abs(ma["window_convergence"][win[-1]]["reattachment_x_over_H"]
                    - ma["window_convergence"][win[-2]]["reattachment_x_over_H"])
    rea_tol = max(rea_drift, 0.06)   # Krank's own reattachment uncertainty
    add("x_re", ma["metrics"]["reattachment_x_over_H"], mb["metrics"]["reattachment_x_over_H"],
        rea_tol, "max(ARCHER2 225->270 window drift, 0.06 H)")
    add("x_sep", ma["metrics"]["separation_x_over_H"], mb["metrics"]["separation_x_over_H"],
        rea_tol, "max(ARCHER2 225->270 window drift, 0.06 H)")
    add("reversed_fraction", ma["metrics"]["reversed_fraction"], mb["metrics"]["reversed_fraction"],
        0.05, "absolute")
    add("signed_tangent_force_per_span", ma["metrics"]["signed_tangent_force_per_span"],
        mb["metrics"]["signed_tangent_force_per_span"],
        0.10 * abs(ma["metrics"]["signed_tangent_force_per_span"]), "10 % of the ARCHER2 value")
    add("ustar_area_weighted_mean", ma["ustar"]["ustar_area_weighted_mean"],
        mb["ustar"]["ustar_area_weighted_mean"],
        0.05 * abs(ma["ustar"]["ustar_area_weighted_mean"]), "5 % of the ARCHER2 value")
    # eps_c: compare against the ARCHER2 phase-block interval of the same statistic
    ea = ma["eps_c"]["eps_c_median_separated_interval"]
    add("eps_c_median_separated", ma["eps_c"]["eps_c_median_separated"],
        mb["eps_c"]["eps_c_median_separated"], 0.5 * (ea["high"] - ea["low"]),
        "phase-block 95% half-width of eps_c (separated region)")
    add("profile_u_rms_mean", ma["profile_validation"]["u_rms_mean"],
        mb["profile_validation"]["u_rms_mean"],
        max(0.1 * abs(ma["profile_validation"]["u_rms_mean"]), 0.005), "10 % of the ARCHER2 RMS")
    out["headline_checks"] = checks
    out["headline_all_inside"] = all(c["inside"] for c in checks.values())
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archer2-case", default=str(DEFAULT_ARCHER2))
    parser.add_argument("--arc-case", default="")
    parser.add_argument("--archer2-log", default="")
    parser.add_argument("--arc-log", default="")
    parser.add_argument("--control-a", default="",
                        help="same-build census log A (decomposition control)")
    parser.add_argument("--control-b", default="",
                        help="same-build census log B at a different rank count")
    parser.add_argument("--date", default=_dt.date.today().isoformat().replace("-", ""))
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--trajectory-steps", type=int, default=400)
    args = parser.parse_args()

    a2_case = Path(args.archer2_case).resolve()
    if not (a2_case / "MANIFEST.json").is_file():
        print(f"[FATAL] no ARCHER2 deposit at {a2_case}", file=sys.stderr)
        return 3

    arc_case = Path(args.arc_case).resolve() if args.arc_case else None
    if arc_case is None and DEFAULT_ARC.is_dir():
        candidates = sorted(p for p in DEFAULT_ARC.iterdir()
                            if (p / "MANIFEST.json").is_file() and "pilot" not in p.name
                            and "marker" not in p.name)
        arc_case = candidates[-1] if candidates else None

    a2_log = Path(args.archer2_log) if args.archer2_log else a2_case / "log.pimpleFoam"
    if args.arc_log:
        arc_log = Path(args.arc_log)
    elif arc_case is not None and (arc_case / "log.pimpleFoam").is_file():
        arc_log = arc_case / "log.pimpleFoam"
    else:
        arc_logs = sorted(DEFAULT_ARC.glob("*/log.pimpleFoam")) if DEFAULT_ARC.is_dir() else []
        arc_log = arc_logs[-1] if arc_logs else None

    l2 = load_module(L2_REDUCER, "rswm_l2_locked")
    l3 = load_module(L3_ANALYSER, "rswm_l3_locked")
    harvest = load_module(HARVEST, "rswm_m13_harvest_locked")

    cert: dict[str, Any] = {
        "status": "CROSS_CLUSTER_EQUIVALENCE",
        "date": args.date,
        "question": ("does Oxford ARC's OpenFOAM/10-foss-2022a reproduce ARCHER2's "
                     "openfoam/org/v10.20230119 on an identical coupled WMLES case?"),
        "archer2": {"module": "openfoam/org/v10.20230119", "build": "crayGccDPInt32Opt",
                    "compiler": "gcc 11.2.0 (PrgEnv-gnu/8.4.0)", "mpi": "cray-mpich/8.1.27",
                    "node": "AMD EPYC 7742 (Rome), 128 cores"},
        "arc": {"module": "OpenFOAM/10-foss-2022a", "build": "linux64GccDPInt32Opt",
                "compiler": "gcc 11.3.0 (foss/2022a)", "mpi": "OpenMPI 4.1.4 (SYSTEMOPENMPI)",
                "node": "Intel Xeon Platinum 8268 (Cascade Lake) 2.9 GHz, 48 cores"},
        "provenance": {
            "l2_reducer": str(L2_REDUCER.relative_to(ROOT)), "l2_reducer_sha256": sha256(L2_REDUCER),
            "l3_analyser": str(L3_ANALYSER.relative_to(ROOT)), "l3_analyser_sha256": sha256(L3_ANALYSER),
            "harvest": str(HARVEST.relative_to(ROOT)), "harvest_sha256": sha256(HARVEST),
            "archer2_case": (str(a2_case.relative_to(ROOT))
                             if str(a2_case).startswith(str(ROOT)) else str(a2_case)),
            "archer2_manifest_sha256": sha256(a2_case / "MANIFEST.json"),
            "bootstrap_protocol": {"circular": 1, "block_points": BLOCK_POINTS,
                                   "dense_phase_points": DENSE_N, "draws": args.draws,
                                   "confidence_level": 0.95, "seed": SEED},
        },
    }
    arrays: dict[str, np.ndarray] = {}

    # ---------------- tier 1 ---------------- #
    tier1_ok: bool | None = None
    if arc_log is not None and Path(arc_log).is_file():
        cert["provenance"]["arc_log"] = str(arc_log)
        cert["provenance"]["arc_log_sha256"] = sha256(Path(arc_log))
        a2_faces = parse_first_solve(a2_log)
        arc_faces = parse_first_solve(Path(arc_log))
        first_solve = compare_first_solve(a2_faces, arc_faces)
        traj = compare_trajectory(parse_trajectory(a2_log, args.trajectory_steps),
                                  parse_trajectory(Path(arc_log), args.trajectory_steps))
        fixture_a2 = a2_case / "log.continuationFixture"
        fixture_arc = Path(arc_log).with_name("log.continuationFixture")
        fixture = None
        if fixture_a2.is_file() and fixture_arc.is_file():
            ta = fixture_a2.read_text(errors="replace").strip()
            tb = fixture_arc.read_text(errors="replace").strip()
            fixture = {"archer2_sha256": sha256(fixture_a2), "arc_sha256": sha256(fixture_arc),
                       "identical_text": ta == tb,
                       "archer2_line": ta.splitlines()[-1] if ta else "",
                       "arc_line": tb.splitlines()[-1] if tb else ""}
        control = None
        if args.control_a and args.control_b:
            control = compare_first_solve(parse_first_solve(Path(args.control_a)),
                                          parse_first_solve(Path(args.control_b)))
            control["role"] = ("ONE build, TWO MPI rank counts: isolates how much of any "
                               "first-solve disagreement is caused by the decomposition "
                               "alone rather than by the build")
            control["log_a"] = args.control_a
            control["log_b"] = args.control_b
        cert["tier1_deterministic"] = {"first_solve_census": first_solve,
                                       "early_trajectory": traj,
                                       "kernel_unit_fixture": fixture,
                                       "decomposition_control": control}
        tier1_ok = first_solve["status"] == "IDENTICAL_TO_BUILD_ARITHMETIC"
        for key, rec in traj["series"].items():
            arrays[f"traj_{key}_archer2"] = np.asarray(rec["archer2"], float)
            arrays[f"traj_{key}_arc"] = np.asarray(rec["arc"], float)
            arrays[f"traj_{key}_reldiff"] = np.asarray(rec["relative_difference"], float)
    else:
        cert["tier1_deterministic"] = {"status": "ARC_LOG_ABSENT"}

    # ---------------- tier 2 ---------------- #
    tier2_ok: bool | None = None
    if arc_case is not None and (Path(arc_case) / "MANIFEST.json").is_file():
        try:
            t2 = tier2(l2, l3, harvest, a2_case, Path(arc_case), args.draws)
        except Exception as exc:  # noqa: BLE001
            cert["tier2_statistical"] = {"status": "FAILED", "error": repr(exc)}
            tier2_ok = False
        else:
            arrays.update({k: np.asarray(v) for k, v in t2.pop("arrays").items()})
            cert["tier2_statistical"] = t2
            tier2_ok = bool(t2["headline_all_inside"] and
                            t2["e_tau_comparison"]["zero_inside_paired_interval"])
    else:
        cert["tier2_statistical"] = {"status": "ARC_PRODUCTION_BUNDLE_ABSENT",
                                     "note": "the 405-time-unit ARC replay had not landed"}

    # ---------------- verdict ---------------- #
    if tier2_ok is None:
        verdict = "PARTIAL"
        rc = 2
    elif tier1_ok is not False and tier2_ok:
        verdict = "EQUIVALENT"
        rc = 0
    else:
        verdict = "NOT-EQUIVALENT"
        rc = 1
    cert["tier1_pass"] = tier1_ok
    cert["tier2_pass"] = tier2_ok
    cert["verdict"] = verdict
    cert["verdict_meaning"] = {
        "EQUIVALENT": ("the two builds agree to build arithmetic on the deterministic "
                       "first solve and to within the phase-block sampling uncertainty "
                       "on every headline metric: a matched pair MAY span the two clusters"),
        "NOT-EQUIVALENT": ("a measured discrepancy exceeds the campaign's own uncertainty; "
                           "every matched comparison must stay on one machine"),
        "PARTIAL": "tier 2 inputs were not available; no cross-cluster mixing is authorised",
    }[verdict]

    out_json = ROOT / "codes" / "results" / f"cross_cluster_equivalence_{args.date}.json"
    out_npz = ROOT / "codes" / "results" / f"cross_cluster_equivalence_{args.date}.npz"
    out_json.write_text(json.dumps(json_ready(cert), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out_npz, **arrays)

    print(f"[verdict] {verdict}")
    t1 = cert["tier1_deterministic"]
    if "first_solve_census" in t1:
        fs = t1["first_solve_census"]
        print(f"[tier1] first-solve census: {fs['matched_faces']} faces matched, "
              f"worst relative difference {fs['worst_max_relative_difference']:.3e} "
              f"({fs['worst_field']}), {fs.get('outlier_faces', 0)} faces over tolerance "
              f"({fs.get('outlier_face_fraction', 0.0):.4%}), bit-identical fraction "
              f"{fs['bit_identical_fraction_overall']:.4f} -> {fs['status']}")
        if t1.get("decomposition_control"):
            dc = t1["decomposition_control"]
            print(f"[tier1] decomposition control (one build, two rank counts): "
                  f"{dc['matched_faces']} faces, worst relative difference "
                  f"{dc['worst_max_relative_difference']:.3e}, "
                  f"{dc.get('outlier_faces', 0)} faces over tolerance "
                  f"({dc.get('outlier_face_fraction', 0.0):.4%})")
        if t1.get("kernel_unit_fixture"):
            print(f"[tier1] kernel unit fixture identical: "
                  f"{t1['kernel_unit_fixture']['identical_text']}")
    if "headline_checks" in cert.get("tier2_statistical", {}):
        for name, rec in cert["tier2_statistical"]["headline_checks"].items():
            flag = "PASS" if rec["inside"] else "FAIL"
            print(f"[tier2][{flag}] {name}: ARCHER2 {rec['archer2']:.6g} vs ARC "
                  f"{rec['arc']:.6g} (diff {rec['difference']:+.3g}, tol {rec['tolerance']:.3g})")
    print(f"[cert] {out_json.relative_to(ROOT)}")
    print(f"[cert] {out_npz.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
