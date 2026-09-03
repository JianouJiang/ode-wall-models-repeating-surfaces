#!/usr/bin/env python3
"""M13 / R2-m3 harvest: higher-Reynolds-number coupled WMLES on the Xiao hill.

Consumes the finalized ARCHER2 bundles
``codes/results/rswm_xiao_highre_campaign_m13_final/re{5600,10595,19000,37000}/``
(G0/G1c/G2c x 2 wall models at 5600 and 10595, G1c/G2c at 19000 and 37000;
same producer/finalizer gates as the deposited Re_H = 5600 matrix, but with
the CREST bulk velocity held at u_b = 1 -- the deposited matrix constrained
the domain-volume average, i.e. ran at crest bulk velocity 1.387 and
Re_H,eff ~ 7,770; it is carried here only as a documented legacy anchor) and
writes

    codes/results/m13_highre_coupled_<date>.npz
    codes/results/m13_highre_coupled_<date>_summary.json

with the same estimands, uncertainty protocol and provenance fields as the
deposited certificates:

* wall traction on the physical downstream tangent, phase-reduced with
  area weights (imported from ``rswm_common_surface_grid_l2.py``);
* at Re_H = 5,600 the reference is the Xiao et al. (2020) alpha=1 DNS wall
  traction (deposited 512-phase reconstruction) and at Re_H = 10,595 the
  Krank, Kronbichler & Wall (2018) DNS skin friction (c_f along the bottom
  wall, 1153 points); the metric is the
  deposited RMS-normalised phase error with the paired circular phase-block
  bootstrap (Lx/8 blocks, 20,000 draws) and the exact eight-block sign-flip
  failure test (error energy > DNS energy), imported from the Level-3
  analyser;
At Re_H=37,000 the registered TBLE branch policy terminated both G1c and G2c
producers on a three-root ambiguous continuation before statistics began. The
failure-aware terminal bundle preserves those scheduler-bound failures, while
the physical metrics at that Reynolds number therefore contain equilibrium
only. This is a numerical-admissibility failure, not a physical TBLE traction
result; no TBLE value is imputed. At every Re_H/model combination that completed,
the coupled cancellation parameter
  eps_c(x) = |tau_s| / (|dp_w/ds| y_m) is measured from the run's own
  wall-sampled ``wallShearStressMean`` and ``pMean`` (no modelled quantity),
  with phase-block bootstrap intervals, at the coupled matching height and
  rescaled to the paper's outer fraction y_m/H = 0.10;
* at Re_H = 10,595 / 19,000 / 37,000 the coupled mean-velocity profiles are
  validated against the Rapp (2009) / Rapp & Manhart (2011) PIV experiments
  (ERCOFTAC UFR3-30, ten stations, absolute y, masked points are zero rows),
  and additionally against the Krank DNS and Breuer LESOCC LES (10,595) and
  the Manhart MGLET LES (37,000);
* reattachment/separation from the coupled zero crossings versus the Krank
  DNS (10,595) and the experimental near-wall sign-change bracket (19k/37k);
* averaging-window (180/225/270) and drive-force stationarity checks;
* wall-unit resolution (Δs+, y_m+, Δz+) from the run's own traction.

Nothing here modifies the pinned drivers or the deposited results.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import os as _os

def _strip_mglet_placeholders(arr):
    """Drop the ERCOFTAC UFR3-30 deposit's trailing plot-axis closure rows.

    The file ends with (0,0,0) and (9,0,0): axis endpoints for the published
    figure, not measurements (real data ends at x/H = 8.9909916).  Left in, they
    inject a spurious tau_w = 0 crossing at x = 0 into separation detection.
    Effect on E_tau is < 0.001 but the separation list is wrong.
    Operator fix 2026-08-25, found by the independent truth-reference audit.
    """
    import numpy as _np
    a = _np.asarray(arr)
    while len(a) and _np.all(a[-1, 1:] == 0.0):
        a = a[:-1]
    return a

CAMPAIGN_ROOT = Path(_os.environ.get("M13_CAMPAIGN_ROOT", ROOT / "codes" / "results" / "rswm_xiao_highre_campaign_m13_final"))
LEGACY_RE5600_ROOT = ROOT / "codes" / "results" / "legacy_volume_average_re5600_20260823"


def legacy_or_canonical(name: str) -> Path:
    """Keep the superseded volume-driven matrix available as history.

    The corrected Re=5600 rebase replaces the canonical L2/L3 filenames.  M13
    still reports the old matrix as an explicitly labelled legacy anchor, so a
    later full-ladder harvest must read the preserved copy rather than silently
    relabelling the corrected matrix as the old run.
    """
    legacy = LEGACY_RE5600_ROOT / name
    return legacy if legacy.is_file() else ROOT / "codes" / "results" / name


DEPOSITED_L2_NPZ = legacy_or_canonical("rswm_common_surface_grid_l2.npz")
DEPOSITED_L2_JSON = legacy_or_canonical("rswm_common_surface_grid_l2_summary.json")
DEPOSITED_L3_JSON = legacy_or_canonical("rswm_grid_results_l3_summary.json")
L2_REDUCER = ROOT / "codes" / "analysis" / "rswm_common_surface_grid_l2.py"
def _resolve_l3_analyser():
    """Locate analyze_grid_results_l3.py.

    The pipeline ROTATES development/nodes/ (levels get archived into
    development/exhausted_*/ when a tree is exhausted), so the live node path is
    not a durable address.  Search, in order: the live node dir, the stable
    generator copy under codes/figures/node_generators/, then the newest
    exhausted archive.  All copies are byte-identical; the harvest records
    l3_analyser_sha256, so provenance is unchanged by which one is found.
    Operator fix 2026-08-25 (the live path vanished at 00:52 and broke the
    R2-3/M6 and M13 harvests).
    """
    cands = [ROOT / "development" / "nodes" / "node_004" / "analyze_grid_results_l3.py",
             ROOT / "codes" / "figures" / "node_generators" / "analyze_grid_results_l3.py"]
    arch = sorted((ROOT / "development").glob("exhausted_*/nodes/node_004/analyze_grid_results_l3.py"))
    cands.extend(reversed(arch))
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError("analyze_grid_results_l3.py not found in: "
                            + ", ".join(str(c) for c in cands))


L3_ANALYSER = _resolve_l3_analyser()
REFERENCE_ROOT = ROOT / "codes" / "raw_data" / "periodic_hill_ufr3_30"
REFERENCE_MANIFEST = REFERENCE_ROOT / "MANIFEST.json"
MESH_GENERATOR = ROOT / "codes" / "openfoam" / "make_xiao_dns_wmles_case.py"
WRAPPER = ROOT / "jobs" / "rswm_xiao_highre_production_wrapper.sh"
LIBRARY_RE = ROOT / "codes" / "openfoam" / "registeredMeanVelocityForceRe"

RES = (5600, 10595, 19000, 37000)
MATRIX_GRIDS = {5600: ("G0", "G1c", "G2c"), 10595: ("G0", "G1c", "G2c"), 19000: ("G1c", "G2c"), 37000: ("G1c", "G2c")}
DNS_5600 = ROOT / "codes" / "results" / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
MGLET_5600_WALL = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
GRIDS = ("G0", "G1c", "G2c")
MODELS = ("equilibrium", "total_gradient_tble")
CELLS = {"G0": 92160, "G1c": 307200, "G2c": 819200}
GRID_TAG = {"G0": "g0", "G1c": "g1", "G2c": "g2"}
LX, LY, LZ = 9.0, 3.036, 4.5
DENSE_N = 4096
BLOCK_POINTS = 512
BLOCK_SENSITIVITY = (256, 512, 1024)
DRAWS = 20000
SEED = 20260823
OUTER_YM_OVER_H = 0.10
ERCOFTAC_STATIONS = (0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
AVERAGING_LENGTHS = (180, 225, 270)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
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


def grid_path_descriptor(l2: Any, values: list[float], cells: list[int]) -> dict[str, Any]:
    """Report a registered grid path without promoting two grids to convergence.

    The Re_H=5,600 and 10,595 matrices contain the registered G0/G1c/G2c
    path and therefore use the deposited three-grid descriptor.  The higher-Re
    campaigns contain G1c/G2c only: those two points measure sensitivity, but
    cannot establish a trend or observed order.  Keeping a distinct status is
    deliberate so prose and verifiers cannot silently call the latter a
    convergence result.
    """
    if len(values) != len(cells):
        raise ValueError("grid-path values and cell counts have different lengths")
    if len(values) == 3:
        return l2.path_convergence(values, cells)
    if len(values) != 2:
        raise ValueError(f"grid-path descriptor requires two or three grids, got {len(values)}")
    coarse, fine = map(float, values)
    if not np.all(np.isfinite([coarse, fine])):
        return {
            "status": "undefined_nonfinite_two_grid",
            "cells": [int(value) for value in cells],
            "values": [coarse, fine],
        }
    change = fine - coarse
    return {
        "status": "two_grid_sensitivity_only",
        "cells": [int(value) for value in cells],
        "values": [coarse, fine],
        "coarse_to_fine": float(change),
        "envelope": float(abs(change)),
        "fine_change_over_fine_magnitude": float(
            abs(change) / max(abs(fine), 1.0e-14)
        ),
        "convergence_claim_permitted": False,
    }


def periodic_derivative_arclength(x: np.ndarray, y_wall: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Central periodic derivative d(values)/ds along the wall arclength."""
    n = len(x)
    xp = np.r_[x[-1] - LX, x, x[0] + LX]
    yp = np.r_[y_wall[-1], y_wall, y_wall[0]]
    vp = np.r_[values[-1], values, values[0]]
    ds = np.sqrt(np.diff(xp) ** 2 + np.diff(yp) ** 2)
    out = np.empty(n)
    for i in range(n):
        out[i] = (vp[i + 2] - vp[i]) / (ds[i] + ds[i + 1])
    return out


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def read_rapp_station(path: Path) -> dict[str, Any]:
    x = None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            m = re.search(r"x/h\s*=\s*([0-9.]+)", line)
            if m:
                x = float(m.group(1))
            continue
        if line.strip():
            rows.append([float(v) for v in line.split(",")])
    a = np.asarray(rows, float)
    valid = np.any(a[:, 1:] != 0.0, axis=1)
    return {"x": x, "y": a[:, 0], "u": a[:, 1], "v": a[:, 2], "uu": a[:, 3],
            "vv": a[:, 4], "uv": a[:, 5], "valid": valid, "file": str(path.relative_to(ROOT))}


def read_space_station(path: Path, x: float) -> dict[str, Any]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append([float(v) for v in s.split()])
        except ValueError:
            continue
    a = np.asarray(rows, float)
    order = np.argsort(a[:, 0])
    a = a[order]
    return {"x": x, "y": a[:, 0], "u": a[:, 1], "v": a[:, 2],
            "valid": np.ones(len(a), bool), "file": str(path.relative_to(ROOT))}


def read_krank_station(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    m = re.search(r"Station:\s*x/H\s*=\s*([0-9eE+.\-]+)", text)
    x = float(m.group(1)) if m else None
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("-"):
            continue
        try:
            rows.append([float(v) for v in s.split(",") if v.strip()])
        except ValueError:
            continue
    a = np.asarray(rows, float)
    order = np.argsort(a[:, 0])
    a = a[order]
    return {"x": x, "y": a[:, 0], "u": a[:, 1], "v": a[:, 2],
            "valid": np.ones(len(a), bool), "file": str(path.relative_to(ROOT))}


def xiao_dns_reference(l2: Any) -> dict[str, Any]:
    """Xiao alpha=1 DNS (Re_H=5600) as ten absolute-y stations at the ERCOFTAC x/h."""
    dns = np.load(DNS_5600)
    sys.path.insert(0, str(ROOT / "codes" / "raw_data" / "geometry_driven" / "xiao_pehill_parameterized" / "utility" / "hill-geometry-gereration"))
    from hillShape import profile as hill_profile  # type: ignore
    x_dns = np.asarray(dns["x"], float)
    stations = []
    for xs in ERCOFTAC_STATIONS:
        i = int(np.argmin(np.abs(x_dns - xs)))
        y = np.asarray(dns["y"][i], float)
        u = np.asarray(dns["U"][i], float)
        v = np.asarray(dns["V"][i], float)
        ok = np.isfinite(y) & np.isfinite(u)
        wall = float(hill_profile(np.asarray([x_dns[i]]).copy())[0])
        stations.append({"x": float(xs), "x_dns": float(x_dns[i]), "y": y[ok] + wall, "u": u[ok], "v": v[ok],
                         "valid": np.ones(int(np.count_nonzero(ok)), bool), "file": str(DNS_5600.relative_to(ROOT))})
    truth_tau, audit = l2.dns_tangent_reference(dns)
    truth_phase = np.mod((x_dns - x_dns.min()) / LX, 1.0)
    return {"label": "Xiao et al. (2020) alpha=1 DNS Re=5600", "stations": stations,
            "truth_phase": truth_phase, "truth_tau": truth_tau, "audit": audit}


def load_references() -> dict[str, Any]:
    erc = REFERENCE_ROOT / "ercoftac_ufr3_30"
    krank_dir = REFERENCE_ROOT / "krank_2018_re10595"
    refs: dict[str, Any] = {}
    exp_tag = {5600: "5600", 10595: "10600", 19000: "19000", 37000: "37000"}
    for re_h, tag in exp_tag.items():
        stations = [read_rapp_station(erc / f"UFR3-30_X_{tag}_data_CR-{i:03d}.dat") for i in range(1, 11)]
        refs[f"rapp_{re_h}"] = {"label": f"Rapp (2009) PIV Re={tag}", "stations": stations}
    refs["breuer_10595"] = {
        "label": "Breuer et al. (2009) LESOCC LES Re=10595",
        "stations": [read_space_station(erc / f"UFR3-30_C_10595_data_MB-{i:03d}.dat", xs)
                     for i, xs in enumerate(ERCOFTAC_STATIONS, start=1)],
    }
    refs["mglet_37000"] = {
        "label": "Manhart/Peller MGLET LES Re=37000",
        "stations": [read_space_station(erc / f"UFR3-30_data-NP-re37000-v40-{i:02d}.dat", xs)
                     for i, xs in enumerate(ERCOFTAC_STATIONS, start=1)],
    }
    refs["mglet_10595"] = {
        "label": "Manhart/Peller MGLET LES Re=10595",
        "stations": [read_space_station(erc / f"UFR3-30_data-NP-re10595-v28-{i:02d}.dat", xs)
                     for i, xs in enumerate(ERCOFTAC_STATIONS, start=1)],
    }
    krank_stations = [read_krank_station(p) for p in sorted(krank_dir.glob("KKW_DNS_Periodic_Hill_Re10595_Station*.dat"))]
    refs["krank_10595"] = {"label": "Krank, Kronbichler & Wall (2018) DNS Re=10595", "stations": krank_stations}
    # Krank bottom-wall skin friction
    rows = []
    for line in (krank_dir / "KKW_DNS_Periodic_Hill_Re10595_cf_cp_bottom.dat").read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("-"):
            continue
        try:
            rows.append([float(v) for v in s.split(",") if v.strip()])
        except ValueError:
            continue
    cf = np.asarray(rows, float)
    refs["krank_cf"] = {
        "x": cf[:, 0], "cf": cf[:, 1], "cp": cf[:, 2],
        "file": str((krank_dir / "KKW_DNS_Periodic_Hill_Re10595_cf_cp_bottom.dat").relative_to(ROOT)),
        "documented_separation_x_over_H": 0.20,
        "documented_reattachment_x_over_H": 4.51,
        "documented_reattachment_uncertainty_over_H": 0.06,
        "definition": "c_f = tau_w/(0.5 rho u_b^2), bottom wall, x/H = 0 at the crest; treated here as the signed wall-tangential traction (x-projection reported as sensitivity)",
    }
    return refs


def experimental_reattachment_bracket(stations: list[dict[str, Any]]) -> dict[str, Any]:
    """Reattachment from the sign change of the lowest valid PIV point between stations.

    The PIV cannot resolve tau_w; Rapp & Manhart located reattachment from the
    near-wall velocity.  Here the lowest valid point (y/h ~ 0.03--0.05 above
    the flat floor) is used at the floor stations x/h = 2..7; the estimate is
    the linear zero crossing between the last reversed and the first forward
    station and the bracket is the pair of stations.
    """
    floor = [s for s in stations if 2.0 <= s["x"] <= 7.0]
    near = []
    for s in floor:
        idx = np.nonzero(s["valid"])[0]
        near.append((s["x"], float(s["u"][idx[0]]), float(s["y"][idx[0]])))
    for (x0, u0, y0), (x1, u1, y1) in zip(near[:-1], near[1:]):
        if u0 < 0.0 <= u1:
            return {"estimate_x_over_H": x0 + (0.0 - u0) / (u1 - u0) * (x1 - x0),
                    "bracket_x_over_H": [x0, x1],
                    "near_wall_points": near,
                    "probe_height_over_H": [y0, y1]}
    return {"estimate_x_over_H": math.nan, "bracket_x_over_H": [math.nan, math.nan], "near_wall_points": near}


# --------------------------------------------------------------------------- #
# campaign loading
# --------------------------------------------------------------------------- #
def case_id(re_h: int, grid: str, model: str) -> str:
    mtag = "tble" if model == "total_gradient_tble" else "equilibrium"
    stem = f"rswm_m13_re{re_h}_{GRID_TAG[grid]}_{mtag}_{CELLS[grid]}"
    # 37,000 TBLE producers were resubmitted as _v3 with the fold-degeneracy
    # guard after the _v2 ambiguous-root aborts; every other case is _v2.
    for suffix in ("v2", "v3", "v4"):
        if (CAMPAIGN_ROOT / f"re{re_h}" / f"{stem}_{suffix}").is_dir():
            return f"{stem}_{suffix}"
    return f"{stem}_v2"


def load_case(l2: Any, case: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    manifest = json.loads((case / "MANIFEST.json").read_text())
    mesh = l2.mesh_bottom(case)
    checkpoints = (case / "checkpoint_times_l2.txt").read_text().split()
    curves, pressures = {}, {}
    for name in checkpoints:
        rows = l2.sample_rows(case / "postProcessing_sampleBottomWall" / name / "bottomWall.xy")
        curves[name] = l2.phase_reduce(mesh, rows)
        if rows.shape[1] < 7:
            raise ValueError(f"{case}: wall sample lacks the pMean column")
        aligned = l2.align_sample(mesh, rows)
        rounded = np.round(mesh["xyz"][:, 0], 9)
        x_unique, inverse = np.unique(rounded, return_inverse=True)
        pw = np.asarray([np.average(aligned[inverse == i, 6], weights=mesh["area"][inverse == i])
                         for i in range(len(x_unique))])
        pressures[name] = pw
    return manifest, mesh, curves, pressures


def read_profiles(case: Path, checkpoint: str) -> dict[str, np.ndarray]:
    directory = case / "postProcessing_sampleProfiles" / checkpoint
    dictionary = (case / "input" / "sampleProfiles").read_text()
    registered = {name: float(x) for name, x in re.findall(
        r"\b(x[0-9]+p[0-9]+)\s*\{[^}]*?start\s*\(\s*([0-9.eE+-]+)", dictionary, re.DOTALL)}
    out = {}
    for path in sorted(directory.glob("*.xy")):
        if path.stem not in registered:
            raise ValueError(f"unregistered profile {path}")
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"bad profile {path}")
        out[f"{registered[path.stem]:.3f}"] = data
    return out


def profile_validation(profiles: dict[str, np.ndarray], reference: dict[str, Any]) -> dict[str, Any]:
    """RMS of U_coupled - U_ref over the overlapping absolute-y range, per station."""
    station_rms, station_x, station_n, station_max, near_wall_sign = [], [], [], [], []
    for st in reference["stations"]:
        key = f"{st['x']:.3f}"
        if key not in profiles:
            continue
        data = profiles[key]
        y, u = data[:, 0], data[:, 1]
        valid = st["valid"] & np.isfinite(st["u"])
        yr, ur = st["y"][valid], st["u"][valid]
        if len(yr) < 8:
            continue
        mask = (y >= yr.min()) & (y <= yr.max())
        if np.count_nonzero(mask) < 8:
            continue
        ref_on = np.interp(y[mask], yr, ur)
        err = u[mask] - ref_on
        station_x.append(float(st["x"]))
        station_rms.append(float(np.sqrt(np.mean(err ** 2))))
        station_max.append(float(np.max(np.abs(err))))
        station_n.append(int(np.count_nonzero(mask)))
        # near-wall sign agreement at the reference's lowest valid point
        u_c_low = float(np.interp(yr[0], y, u))
        near_wall_sign.append(float(np.sign(u_c_low) == np.sign(ur[0])))
    if not station_rms:
        raise ValueError(f"no overlapping stations with {reference['label']}")
    return {
        "reference": reference["label"],
        "station_x": np.asarray(station_x),
        "station_u_rms": np.asarray(station_rms),
        "station_u_max_abs_error": np.asarray(station_max),
        "station_points": np.asarray(station_n),
        "u_rms_mean": float(np.mean(station_rms)),
        "u_rms_max": float(np.max(station_rms)),
        "near_wall_sign_agreement": float(np.mean(near_wall_sign)),
    }


def reference_spread(refs: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Reference-to-reference mean-velocity spread (same stations, absolute y)."""
    out_rms = []
    for sa, sb in zip(refs[a]["stations"], refs[b]["stations"]):
        va = sa["valid"]
        vb = sb["valid"]
        ya, ua = sa["y"][va], sa["u"][va]
        yb, ub = sb["y"][vb], sb["u"][vb]
        lo, hi = max(ya.min(), yb.min()), min(ya.max(), yb.max())
        grid = np.linspace(lo, hi, 200)
        out_rms.append(float(np.sqrt(np.mean((np.interp(grid, ya, ua) - np.interp(grid, yb, ub)) ** 2))))
    return {"pair": [refs[a]["label"], refs[b]["label"]], "station_u_rms": np.asarray(out_rms),
            "u_rms_mean": float(np.mean(out_rms)), "u_rms_max": float(np.max(out_rms))}


def registered_drive(case: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the immutable drive registration from three deposited files.

    The first corrected finalizer omitted the three convenience fields promised
    by ``M13/REBASE.md`` from its JSON manifest.  The physical evidence is still
    present, hash-addressed by that manifest: ``fvConstraints`` contains the
    requested domain-volume velocity, while the solver and ``checkMesh`` logs
    independently print the selected-cell and mesh volumes.  Deriving the crest
    bulk velocity from those files is stricter than inserting metadata into a
    terminal result bundle, which must remain byte-identical to the deposit.
    """
    paths = {
        "fvConstraints": case / "input" / "fvConstraints",
        "solver_log": case / "log.pimpleFoam",
        "checkmesh_log": case / "log.checkMesh",
    }
    for label, path in paths.items():
        relative = str(path.relative_to(case))
        registered = manifest.get("files", {}).get(relative)
        if not path.is_file() or registered is None:
            raise SystemExit(f"{case.name}: drive-evidence file is not registered: {relative}")
        if path.stat().st_size != int(registered["bytes"]) or sha256(path) != registered["sha256"]:
            raise SystemExit(f"{case.name}: drive-evidence file changed: {relative}")

    fv_text = paths["fvConstraints"].read_text(errors="replace")
    targets = re.findall(
        r"\bUbar\s*\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)\s*;",
        fv_text,
    )
    if len(targets) != 1:
        raise SystemExit(f"{case.name}: expected one vector Ubar registration, found {len(targets)}")
    target = tuple(float(value) for value in targets[0])
    if abs(target[1]) > 1.0e-14 or abs(target[2]) > 1.0e-14 or target[0] <= 0.0:
        raise SystemExit(f"{case.name}: invalid streamwise Ubar target {target}")

    solver_text = paths["solver_log"].read_text(errors="replace")
    selected = re.findall(
        r"selected\s+([0-9]+)\s+cell\(s\)\s+with volume\s+([0-9.eE+-]+)",
        solver_text,
    )
    if len(selected) != 1:
        raise SystemExit(f"{case.name}: expected one selected-cell volume, found {len(selected)}")
    selected_cells, selected_volume = int(selected[0][0]), float(selected[0][1])
    if selected_cells != int(manifest["grid_cells"]):
        raise SystemExit(
            f"{case.name}: drive selected {selected_cells} cells, expected {manifest['grid_cells']}")

    check_text = paths["checkmesh_log"].read_text(errors="replace")
    checked = [float(value) for value in re.findall(
        r"Total volume\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)",
        check_text,
    )]
    if len(checked) != 1:
        raise SystemExit(f"{case.name}: expected one checkMesh total volume, found {len(checked)}")
    checked_volume = checked[0]
    if abs(selected_volume - checked_volume) > 1.0e-8 * checked_volume:
        raise SystemExit(
            f"{case.name}: selected volume {selected_volume} != checkMesh volume {checked_volume}")

    crest_registered = target[0] * selected_volume / (LX * LZ * 2.036)
    if abs(crest_registered - 1.0) > 1.0e-5:
        raise SystemExit(
            f"{case.name}: deposited drive implies crest bulk velocity {crest_registered}, not 1")

    # Future finalizers may carry the convenience fields.  If present, bind
    # them to the independently reconstructed values rather than trusting them.
    optional = {
        "crest_bulk_velocity": crest_registered,
        "volume_average_Ubar": target[0],
        "mesh_total_volume": selected_volume,
    }
    for key, value in optional.items():
        if key in manifest and abs(float(manifest[key]) - value) > 1.0e-8 * max(1.0, abs(value)):
            raise SystemExit(f"{case.name}: manifest {key} disagrees with immutable drive evidence")

    return {
        "volume_average_Ubar": target[0],
        "mesh_total_volume": selected_volume,
        "crest_bulk_velocity_registered": crest_registered,
        "registration_source": "hash-addressed fvConstraints + solver selected volume + checkMesh total volume",
        "registration_evidence_sha256": {label: sha256(path) for label, path in paths.items()},
    }


def branch_policy(case: Path) -> dict[str, Any]:
    """Which TBLE root-selection kernel the case actually ran, from its own log.

    Production libraries print TOTAL_GRADIENT_TBLE_KERNEL version=<id> once at
    startup; a log without the marker came from the pinned continuation kernel.
    Degeneracy selections (and, for the alternative homotopy kernel, branch
    restarts) are counted so the frequency of the near-zero twin-root event is a
    measured quantity rather than an anecdote.
    """
    log = case / "log.pimpleFoam"
    if not log.is_file():
        return {"kernel_version": "unknown", "log_present": False}
    text = log.read_text(errors="replace")
    marker = re.search(r"TOTAL_GRADIENT_TBLE_KERNEL version=(\S+)", text)
    degen = [int(v) for v in re.findall(r"degenerateRoots=([0-9]+)", text)]
    restarts = [int(v) for v in re.findall(r"homotopyRestarts=([0-9]+)", text)]
    return {
        "kernel_version": marker.group(1) if marker else "pinned-continuation",
        "degenerate_selection_reports": len(degen),
        "degenerate_selections_max_per_report": max(degen) if degen else 0,
        "homotopy_restart_reports": len(restarts),
        "homotopy_restarts_max_per_report": max(restarts) if restarts else 0,
        "branch_failure_in_log": "TBLE branch failure" in text,
    }


def crest_bulk_velocity(profiles: dict[str, np.ndarray]) -> dict[str, float]:
    """Crest bulk velocity from the volumetric flux, measured at every station.

    The flow is incompressible, periodic and statistically steady, so the flux per
    unit span Q = int u dy is the same at every streamwise station; the crest bulk
    velocity is Q divided by the crest section height 2.036.  Reading Q from a
    SINGLE sampled line (z = 2.25, x/h = 0.05) is a noisy estimator of that
    constant: two runs on the same mesh under the same exact constraint differ by
    ~4 % in it, because one spanwise line does not average the spanwise
    inhomogeneity of the mean field.  We therefore report the station median as the
    measurement, the station spread as its sampling uncertainty, and keep the
    single-slice value for continuity with the earlier certificates.
    """
    out: dict[str, float] = {}
    fluxes = []
    for key, data in sorted(profiles.items()):
        y, u = data[:, 0], data[:, 1]
        q = float(np.trapz(u, y))
        fluxes.append(q)
        out[f"Q_per_span_x{key}"] = q
    if fluxes:
        arr = np.asarray(fluxes, float)
        out["Q_per_span_station_median"] = float(np.median(arr))
        out["Q_per_span_station_p05"] = float(np.quantile(arr, 0.05))
        out["Q_per_span_station_p95"] = float(np.quantile(arr, 0.95))
        out["Q_per_span_station_relative_spread"] = float(np.ptp(arr) / max(abs(np.median(arr)), 1e-14))
        out["crest_bulk_velocity_stations"] = float(np.median(arr) / 2.036)
        out["crest_bulk_velocity_station_count"] = int(len(arr))
    if "Q_per_span_x0.050" in out:
        out["crest_bulk_velocity_measured"] = out["Q_per_span_x0.050"] / 2.036
        out["crest_bulk_velocity_single_slice_note"] = (
            "single z=2.25 line at x/h=0.05; superseded as the measurement by "
            "crest_bulk_velocity_stations, retained for continuity")
    return out


def _halves_block_statistics(t: np.ndarray, g: np.ndarray, window: np.ndarray) -> dict[str, Any]:
    """Block-scatter uncertainty of the first-half/second-half drive difference.

    The driving gradient wanders on eddy-turnover timescales, so the raw halves
    percentage is not interpretable against a fixed threshold.  Six 45-unit block
    means give the within-window scatter, hence a standard error for the difference
    of the two half-means and a z-score that says whether the drift is
    distinguishable from that wander.
    """
    edges = np.linspace(135.0, 405.0, 7)
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (t >= lo) & (t < hi) if hi < 405.0 else (t >= lo) & (t <= hi)
        if np.any(sel):
            means.append(float(np.mean(g[sel])))
    means = np.asarray(means, float)
    scale = abs(float(np.mean(g[window])))
    if len(means) < 4 or scale <= 0.0:
        return {"halves_block_count": int(len(means)), "halves_difference_z": math.nan,
                "halves_relative_difference_block_se": math.nan}
    first, second = means[: len(means) // 2], means[len(means) // 2:]
    difference = float(np.mean(second) - np.mean(first))
    var = float(np.var(means, ddof=1))
    se = math.sqrt(var * (1.0 / len(first) + 1.0 / len(second)))
    return {
        "halves_block_count": int(len(means)),
        "halves_block_means": [float(v) for v in means],
        "halves_difference": difference,
        "halves_difference_standard_error": se,
        "halves_relative_difference_block_se": float(se / scale),
        "halves_difference_z": float(difference / se) if se > 0.0 else math.nan,
    }


def drive_stationarity(log: Path, target_ubar: float) -> dict[str, Any]:
    """Mean driving pressure gradient in the two halves of the averaging window."""
    times, grads, ubar = [], [], []
    current = None
    pattern = re.compile(r"[Pp]ressure gradient source: uncorrected Ubar = ([0-9.eE+-]+), pressure gradient = ([0-9.eE+-]+)")
    with log.open(errors="replace") as stream:
        for line in stream:
            if line.startswith("Time = "):
                current = float(line[7:].rstrip().rstrip("s"))
            elif "ressure gradient source:" in line and current is not None:
                m = pattern.search(line)
                if m:
                    times.append(current)
                    ubar.append(float(m.group(1)))
                    grads.append(float(m.group(2)))
    t = np.asarray(times)
    g = np.asarray(grads)
    u = np.asarray(ubar)
    if len(t) < 100:
        raise ValueError(f"insufficient force telemetry in {log}")
    first = (t >= 135.0) & (t < 270.0)
    second = (t >= 270.0) & (t <= 405.0)
    window = (t >= 135.0) & (t <= 405.0)
    transient = (t >= 90.0) & (t < 135.0)
    return {
        "samples": int(len(t)),
        "window_mean_gradient": float(np.mean(g[window])),
        "first_half_mean_gradient": float(np.mean(g[first])),
        "second_half_mean_gradient": float(np.mean(g[second])),
        "halves_relative_difference": float(abs(np.mean(g[second]) - np.mean(g[first])) / abs(np.mean(g[window]))),
        # The halves difference is a noisy statistic: the driving gradient wanders on
        # eddy-turnover timescales, so an absolute threshold is not interpretable on its
        # own.  Six 45-unit sub-blocks give the within-window block scatter, hence a
        # standard error for the difference of the two half-means and a z-score.  A case
        # is non-stationary when |z| is large, NOT when the raw percentage exceeds a
        # number chosen from a couple of cases.
        **_halves_block_statistics(t, g, window),
        "pre_window_mean_gradient": float(np.mean(g[transient])) if np.any(transient) else math.nan,
        "window_mean_uncorrected_Ubar": float(np.mean(u[window])),
        "registered_volume_average_Ubar": float(target_ubar),
        "window_Ubar_max_abs_deviation": float(np.max(np.abs(u[window] - target_ubar))),
        "telemetry_samples_in_window": int(np.count_nonzero(window)),
    }


# --------------------------------------------------------------------------- #
# main reduction
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=_dt.date.today().isoformat().replace("-", ""))
    parser.add_argument("--res", default=",".join(str(r) for r in RES),
                        help="comma-separated Reynolds numbers with finalized bundles")
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    res = tuple(int(r) for r in args.res.split(","))

    l2 = load_module(L2_REDUCER, "rswm_l2_locked")
    l3 = load_module(L3_ANALYSER, "rswm_l3_locked")
    refs = load_references()
    refs["xiao_5600"] = xiao_dns_reference(l2)
    ref_manifest = json.loads(REFERENCE_MANIFEST.read_text())
    for relative, rec in ref_manifest["files"].items():
        path = REFERENCE_ROOT / relative
        if sha256(path) != rec["sha256"]:
            raise SystemExit(f"reference file changed: {relative}")

    dense = np.arange(DENSE_N, dtype=float) / DENSE_N
    out: dict[str, Any] = {"status": "M13_HIGHRE_COUPLED_OK", "reynolds_numbers": list(res)}
    payload: dict[str, Any] = {"dense_phase": dense}
    provenance: dict[str, Any] = {
        "campaign_root": str(CAMPAIGN_ROOT.relative_to(ROOT)) if str(CAMPAIGN_ROOT).startswith(str(ROOT)) else str(CAMPAIGN_ROOT),
        "deposited_re5600_l2": str(DEPOSITED_L2_NPZ.relative_to(ROOT)),
        "deposited_re5600_l2_sha256": sha256(DEPOSITED_L2_NPZ),
        "deposited_re5600_l3_summary_sha256": sha256(DEPOSITED_L3_JSON),
        "l2_reducer_sha256": sha256(L2_REDUCER),
        "l3_analyser_sha256": sha256(L3_ANALYSER),
        "mesh_generator": str(MESH_GENERATOR.relative_to(ROOT)),
        "mesh_generator_sha256": sha256(MESH_GENERATOR),
        "highre_wrapper_sha256": sha256(WRAPPER),
        "tble_library_re_sha256": {p.name: sha256(p) for p in sorted(LIBRARY_RE.glob("*.[CH]"))},
        "reference_manifest_sha256": sha256(REFERENCE_MANIFEST),
        "xiao_dns_5600_file": str(DNS_5600.relative_to(ROOT)),
        "xiao_dns_5600_sha256": sha256(DNS_5600),
        "krank_cf_file": refs["krank_cf"]["file"],
        "krank_cf_sha256": sha256(ROOT / refs["krank_cf"]["file"]),
        "bootstrap_protocol": {
            "circular": 1, "confidence_level": 0.95, "dense_phase_points": DENSE_N,
            "draws": args.draws, "primary_block_points": BLOCK_POINTS,
            "block_sensitivity_points": list(BLOCK_SENSITIVITY), "seed": SEED,
            "estimand": "phase-averaged RMS-normalised physical-tangent wall-traction error; phase-block medians of eps_c",
        },
        "matching_height_convention": (
            "wall model matched at the bottom-wall owner-cell centre on the common physical "
            "surface (first-cell-centre fraction d0/2 = 0.004781207506 of Ly on all grids, "
            "y_m/H ~ 0.0145); y_m/H is fixed across Re_H, so y_m+ grows with Re_H and is "
            "reported from the run's own traction"
        ),
        "averaging_window": "discard 0<=t<135, average 135<=t<=405 (30 flow-throughs of Lx=9H); checkpoints 315/360/405 = 180/225/270 units",
    }

    # ------------------------------------------------------------------ #
    # Re = 5600 anchor from the deposited L2/L3 artefacts
    # ------------------------------------------------------------------ #
    dep = np.load(DEPOSITED_L2_NPZ)
    dep_l3 = json.loads(DEPOSITED_L3_JSON.read_text())
    dep_summary = json.loads(DEPOSITED_L2_JSON.read_text())
    anchor: dict[str, Any] = {"Re_H_nominal": 5600, "Re_H_effective_crest_bulk": 5600 * (114.359 / (LZ * LX)) / 2.036,
                              "bulk_velocity_constraint": "domain-volume average Ubar=1 (deposited); crest bulk velocity = V_mesh/(Lx Lz)/2.036 = 1.387",
                              "role": "legacy anchor only -- NOT part of the corrected Reynolds ladder",
                              "source": "deposited rswm_common_surface_grid_l2 / rswm_grid_results_l3",
                              "producer_jobs": dep_summary["producer_jobs"], "metrics": {}, "eps_c": {}}
    truth5600 = l2.periodic_interp(dep["truth_phase"], dep["truth_tau_s"], dense)
    for grid in GRIDS:
        for model in MODELS:
            prefix = f"{grid}_{model}"
            anchor["metrics"][f"{grid}:{model}"] = {
                k: float(dep[f"{prefix}_{k}"]) for k in (
                    "r2", "relative_rms", "cf_error_rms", "sign_accuracy", "reversed_fraction",
                    "separation_x_over_H", "reattachment_x_over_H", "truth_reattachment_x_over_H",
                    "truth_separation_x_over_H", "signed_tangent_force_per_span")}
            payload[f"legacy5600_{prefix}_phase"] = dep[f"{prefix}_phase"]
            payload[f"legacy5600_{prefix}_tau_s"] = dep[f"{prefix}_tau_s"]
            payload[f"legacy5600_{prefix}_ym"] = dep[f"{prefix}_ym"]
    anchor["primary_intervals"] = dep_l3["phase_bootstrap_primary_intervals"]
    anchor["failure_tests"] = {m: {k: v for k, v in dep_l3["failure_significance_tests"][m].items() if k != "block_values"}
                               for m in MODELS}
    anchor["grid_resolution"] = dep_summary["grid_resolution"]
    payload["legacy5600_truth_phase"] = dep["truth_phase"]
    payload["legacy5600_truth_tau_s"] = dep["truth_tau_s"]
    out["legacy_re5600_volume_average_anchor"] = anchor

    # deposited Re=5600 eps_c needs pMean: recompute from the deposited bundles
    dep_root = ROOT / "codes" / "results" / "rswm_xiao_dns_grid_campaign_final_l2"
    dep_cases = {
        ("G0", "equilibrium"): "rswm_xiao_dns_g0_equilibrium_92160_l2_v1",
        ("G0", "total_gradient_tble"): "rswm_xiao_dns_g0_tble_92160_l2_v1",
        ("G1c", "equilibrium"): "rswm_xiao_dns_g1_equilibrium_307200_l2_v1",
        ("G1c", "total_gradient_tble"): "rswm_xiao_dns_g1_tble_307200_l2_v1",
        ("G2c", "equilibrium"): "rswm_xiao_dns_g2_equilibrium_819200_l2_v1",
        ("G2c", "total_gradient_tble"): "rswm_xiao_dns_g2_tble_819200_l2_v1",
    }

    def eps_record(curve: dict[str, Any], pw: np.ndarray, rng_seed: int, truth_sep: float | None = None,
                   truth_rea: float | None = None) -> dict[str, Any]:
        x = np.asarray(curve["x"])
        tau = np.asarray(curve["tau_s"])
        ym = np.asarray(curve["ym"])
        dpds = periodic_derivative_arclength(x, np.asarray(curve["ywall"]), pw)
        phi = np.abs(dpds) * ym
        eps = np.abs(tau) / np.maximum(phi, 1.0e-14)
        eps_outer = eps * ym / OUTER_YM_OVER_H
        phase = np.asarray(curve["phase"])
        sep, rea = l2.zero_crossings(dense, l2.periodic_interp(phase, tau, dense))
        if truth_sep is not None and truth_rea is not None and math.isfinite(truth_sep) and math.isfinite(truth_rea):
            sep_use, rea_use = truth_sep / LX, truth_rea / LX
        else:
            sep_use, rea_use = sep, rea
        separated = ((phase - sep_use) % 1.0) < ((rea_use - sep_use) % 1.0) if math.isfinite(sep_use) and math.isfinite(rea_use) else np.zeros(len(phase), bool)
        rng = np.random.default_rng(rng_seed)
        n = len(phase)
        block = max(4, n // 8)
        draws_median_sep, draws_median_all, draws_logmean_all = [], [], []
        for _ in range(min(args.draws, 5000)):
            starts = rng.integers(0, n, size=n // block + 1)
            idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
            sample = eps[idx]
            sample_sep = eps[idx][separated[idx]] if np.any(separated[idx]) else sample
            draws_median_all.append(np.median(sample))
            draws_median_sep.append(np.median(sample_sep))
            draws_logmean_all.append(np.exp(np.mean(np.log(np.maximum(sample, 1.0e-12)))))
        return {
            "eps_c_phase": eps, "eps_c_outer_phase": eps_outer, "phi_phase": phi, "dpds_wall_phase": dpds,
            "pw_phase": pw, "separated_mask": separated,
            "eps_c_median_all": float(np.median(eps)),
            "eps_c_median_separated": float(np.median(eps[separated])) if np.any(separated) else math.nan,
            "eps_c_outer_median_separated": float(np.median(eps_outer[separated])) if np.any(separated) else math.nan,
            "eps_c_geometric_mean_all": float(np.exp(np.mean(np.log(np.maximum(eps, 1.0e-12))))),
            "eps_c_median_all_interval": interval(np.asarray(draws_median_all)),
            "eps_c_median_separated_interval": interval(np.asarray(draws_median_sep)),
            "eps_c_geometric_mean_interval": interval(np.asarray(draws_logmean_all)),
            "separated_fraction": float(np.mean(separated)),
            "separation_x_over_H": float(sep * LX) if math.isfinite(sep) else math.nan,
            "reattachment_x_over_H": float(rea * LX) if math.isfinite(rea) else math.nan,
            "cf_rms": float(2.0 * np.sqrt(np.mean(tau ** 2))),
            "cf_mean_abs": float(2.0 * np.mean(np.abs(tau))),
            "phi_mean": float(np.mean(phi)),
        }

    for (grid, model), cid in dep_cases.items():
        _, _, curves, pressures = load_case(l2, dep_root / cid)
        last = list(curves)[-1]
        rec = eps_record(curves[last], pressures[last], SEED + 5600 + CELLS[grid],
                         anchor["metrics"][f"{grid}:{model}"]["truth_separation_x_over_H"],
                         anchor["metrics"][f"{grid}:{model}"]["truth_reattachment_x_over_H"])
        anchor["eps_c"][f"{grid}:{model}"] = {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
        for k, v in rec.items():
            if isinstance(v, np.ndarray):
                payload[f"legacy5600_{grid}_{model}_{k}"] = v

    # ------------------------------------------------------------------ #
    # higher Re campaigns
    # ------------------------------------------------------------------ #
    by_re: dict[int, dict[str, Any]] = {}
    for re_h in res:
        root = CAMPAIGN_ROOT / f"re{re_h}"
        campaign_manifest = json.loads((root / "CAMPAIGN_MANIFEST.json").read_text())
        manifest_status = campaign_manifest.get("status")
        failure_aware = (re_h == 37000 and
                         manifest_status == "TERMINAL_TWO_CASES_PLUS_TWO_REGISTERED_FAILURES_OK")
        if manifest_status != "TERMINAL_SIX_CASE_CAMPAIGN_OK" and not failure_aware:
            raise SystemExit(f"Re={re_h}: campaign not terminal")
        completed_models = ("equilibrium",) if failure_aware else MODELS
        nu = 1.0 / re_h
        record: dict[str, Any] = {"Re_H": re_h, "nu": nu, "producer_jobs": {}, "finalizer_job": campaign_manifest["finalizer_job_id"],
                                  "cases": {}, "metrics": {}, "eps_c": {}, "profiles": {}, "averaging": {},
                                  "drive_stationarity": {}, "grid_resolution": {}, "cost": {}, "manifest_checks": {},
                                  "terminal_status": manifest_status, "available_models": list(completed_models),
                                  "registered_failures": []}
        curves_by: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        press_by: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        manifests: dict[tuple[str, str], dict[str, Any]] = {}
        grids = tuple(campaign_manifest.get("grids", MATRIX_GRIDS[re_h]))
        if grids != MATRIX_GRIDS[re_h]:
            raise SystemExit(f"Re={re_h}: grids {grids} != registered {MATRIX_GRIDS[re_h]}")
        record["grids"] = list(grids)
        GRIDS_RE = grids
        if failure_aware:
            failure_entries = campaign_manifest.get("registered_failures", [])
            if len(failure_entries) != 2:
                raise SystemExit("Re=37000: failure-aware manifest requires two failure records")
            expected_failures = {
                "14889048": ("G1c", 1, 124.0, 125.0),
                "14889051": ("G2c", 76, 78.0, 79.0),
            }
            for entry in failure_entries:
                job = str(entry.get("producer_job_id", ""))
                if job not in expected_failures:
                    raise SystemExit(f"Re=37000: unexpected registered failure producer {job}")
                failure_path = root / f"{entry['case_id']}_failure" / "FAILURE_RECORD.json"
                data = failure_path.read_bytes()
                if len(data) != int(entry["record_bytes"]) or hashlib.sha256(data).hexdigest() != entry["record_sha256"]:
                    raise SystemExit(f"Re=37000: changed failure record for {job}")
                failure = json.loads(data)
                grid, face, time_lo, time_hi = expected_failures[job]
                branch = failure.get("branch_record", {})
                if (failure.get("status") != "REGISTERED_TBLE_BRANCH_AMBIGUITY_FAIL_CLOSED"
                        or failure.get("producer_state") != "FAILED"
                        or failure.get("producer_exit_code") != "15:0"
                        or failure.get("entered_registered_average") is not False
                        or failure.get("grid") != grid
                        or failure.get("model") != "total_gradient_tble"
                        or int(branch.get("face", -1)) != face
                        or tuple(branch.get(k) for k in ("roots", "branchLoss", "ambiguous", "truncated", "finite")) != (3, 0, 1, 0, 1)
                        or not time_lo < float(failure.get("latest_time", -1)) < time_hi):
                    raise SystemExit(f"Re=37000: invalid registered branch-failure record for {job}")
                record["producer_jobs"][f"{grid}:total_gradient_tble"] = job
                record["registered_failures"].append({
                    "case_id": failure["case_id"], "producer_job_id": job,
                    "grid": grid, "model": "total_gradient_tble",
                    "latest_time": failure["latest_time"], "average_start": failure["average_start"],
                    "branch_record": branch, "interpretation": failure["interpretation"],
                    "record_bytes": len(data), "record_sha256": hashlib.sha256(data).hexdigest(),
                })
            if {x["producer_job_id"] for x in record["registered_failures"]} != set(expected_failures):
                raise SystemExit("Re=37000: registered failure producer set is incomplete")
        for grid in GRIDS_RE:
            for model in completed_models:
                cid = case_id(re_h, grid, model)
                case = root / cid
                manifest, mesh, curves, pressures = load_case(l2, case)
                if manifest["grid"] != grid or manifest["model"] != model or int(manifest["Re_H"]) != re_h:
                    raise SystemExit(f"manifest identity mismatch {cid}")
                if abs(float(manifest["nu"]) - nu) > 1.0e-5 * nu:
                    raise SystemExit(f"{cid}: nu {manifest['nu']} != 1/{re_h}")
                if manifest["terminal_state"] != "producer_exit0_solver_end_checkpoint_sampling_and_hash_gates_passed":
                    raise SystemExit(f"{cid}: terminal gate not passed")
                drive = registered_drive(case, manifest)
                record["cases"][f"{grid}:{model}_bulk"] = drive
                for relative, rec_f in manifest["files"].items():
                    p = case / relative
                    if not p.is_file() or p.stat().st_size != rec_f["bytes"]:
                        raise SystemExit(f"{cid}: bundle file missing/changed: {relative}")
                record["manifest_checks"][cid] = {"files": len(manifest["files"]), "maximum_courant": manifest["maximum_courant"],
                                                  "latest_time": manifest["latest_time"]}
                record["producer_jobs"][f"{grid}:{model}"] = manifest["producer_job_id"]
                record["cases"][f"{grid}:{model}"] = cid
                record["cost"][f"{grid}:{model}"] = {"solver_cost": manifest["solver_cost"],
                                                     "producer_elapsed_seconds": manifest["producer_elapsed_seconds"],
                                                     "mpi_ranks": manifest["mpi_ranks"]}
                curves_by[(grid, model)] = curves
                press_by[(grid, model)] = pressures
                manifests[(grid, model)] = manifest
                log = case / "log.pimpleFoam"
                record["drive_stationarity"][f"{grid}:{model}"] = drive_stationarity(
                    log, float(drive["volume_average_Ubar"]))
                record["averaging"][f"{grid}:{model}"] = {}
                if manifest["model"] == "total_gradient_tble":
                    record["cases"][f"{grid}:{model}_tble_realizability"] = manifest["tble_realizability_summary"]
                    log_text = log.read_text(errors="replace")
                    degen = [int(v) for v in re.findall(r"degenerateRoots=([0-9]+)", log_text)]
                    restarts = [int(v) for v in re.findall(r"homotopyRestarts=([0-9]+)", log_text)]
                    kernel_marker = re.search(r"TOTAL_GRADIENT_TBLE_KERNEL version=(\S+)", log_text)
                    record["cases"][f"{grid}:{model}_branch_policy"] = branch_policy(case)
                    record["cases"][f"{grid}:{model}_degenerate_roots"] = {
                        "telemetry_reports": len(degen),
                        "max_per_report": max(degen) if degen else 0,
                        "total": int(sum(degen)),
                        "kernel": ("jobs/rswm_m13_tbleShoot_degenerate.H (fold-degeneracy guard)"
                                   if degen else "pinned rswm_continuation_tbleShoot.H (no guard)"),
                    }
        # matching surface shared by both models on each grid
        if len(completed_models) == 2:
            for grid in GRIDS_RE:
                a = curves_by[(grid, "equilibrium")][list(curves_by[(grid, "equilibrium")])[-1]]
                b = curves_by[(grid, "total_gradient_tble")][list(curves_by[(grid, "total_gradient_tble")])[-1]]
                if not np.allclose(a["ym"], b["ym"], rtol=0.0, atol=2.0e-12):
                    raise SystemExit(f"Re={re_h} {grid}: models do not share a matching surface")

        # ---- wall-traction certification against Krank DNS at 10,595 ----
        truth_phase = truth_tau = None
        truth_events = None
        analytic_tangent_x = None
        if re_h == 5600:
            # The wall-traction truth at 5,600 is the Peller & Manhart MGLET DNS
            # deposited on the ERCOFTAC UFR3-30 page, NOT the tau reconstructed from
            # the public Xiao velocity archive.  codes/analysis/audit_m13_truth_references.py
            # shows the reconstruction is ~2.8x low in RMS against two independent DNS at
            # this Reynolds number, disagrees in sign at a station, puts separation at
            # x/H=0.38 instead of 0.18, and inverts the C_f(Re) ordering against the
            # 10,595 truth -- scoring the two ends against it is not like-for-like.  The
            # run geometry and the reference geometry are the identical hill polynomial
            # (audit test 4, max|dy|/H = 0).  The Xiao reconstruction is retained below
            # as a documented secondary score, never as the primary.
            mglet = np.loadtxt(MGLET_5600_WALL)
            mglet = _strip_mglet_placeholders(mglet)
            kx = mglet[:, 0]
            truth_phase = np.mod(kx / LX, 1.0)
            truth_tau = mglet[:, 1]
            truth_label = "Peller & Manhart MGLET DNS Re=5600 (ERCOFTAC UFR3-30 bottom wall)"
            truth_file = str(MGLET_5600_WALL.relative_to(ROOT))
            truth_definition = ("deposited bottom-wall tau_w, H=1, u_b=1 (crest bulk), rho=1; "
                                "cross-checked against Krank et al. (2018) DNS at the same Re "
                                "(station RMS ratio 0.94)")
            documented = {"documented_separation_x_over_H": 0.18, "documented_reattachment_x_over_H": None,
                          "documented_reattachment_uncertainty_over_H": None}
        elif re_h == 10595:
            kx = refs["krank_cf"]["x"]
            kcf = refs["krank_cf"]["cf"]
            truth_phase = np.mod(kx / LX, 1.0)
            truth_tau = 0.5 * kcf  # rho = 1, u_b = 1
            truth_label, truth_file = refs["krank_10595"]["label"], refs["krank_cf"]["file"]
            truth_definition = refs["krank_cf"]["definition"]
            documented = {k: refs["krank_cf"][k] for k in ("documented_separation_x_over_H", "documented_reattachment_x_over_H", "documented_reattachment_uncertainty_over_H")}
        if truth_tau is not None:
            # x-projection sensitivity: tau_x = tau_s * t_x on the analytic surface
            try:
                sys.path.insert(0, str(ROOT / "codes" / "raw_data" / "geometry_driven" / "xiao_pehill_parameterized" / "utility" / "hill-geometry-gereration"))
                from hillShape import profile as hill_profile  # type: ignore
                from da_budget import periodic_derivative  # type: ignore
                wall = np.asarray(hill_profile(np.asarray(kx, float).copy()), float)
                slope = periodic_derivative(wall, np.asarray(kx, float))
                analytic_tangent_x = 1.0 / np.sqrt(1.0 + slope ** 2)
            except Exception as exc:  # noqa: BLE001
                record["krank_tangent_projection_note"] = f"analytic tangent unavailable: {exc}"
            truth_dense = l2.periodic_interp(truth_phase, truth_tau, dense)
            sep_ref, rea_ref = l2.zero_crossings(dense, truth_dense)
            truth_events = {"separation_x_over_H": float(sep_ref * LX), "reattachment_x_over_H": float(rea_ref * LX),
                            "reversed_fraction": float(np.mean(truth_dense < 0.0)),
                            "cf_rms": float(2.0 * np.sqrt(np.mean(truth_dense ** 2)))}
            truth_events.update(documented)
            payload[f"re{re_h}_truth_phase"] = truth_phase
            payload[f"re{re_h}_truth_tau_s"] = truth_tau
            record["truth"] = {"reference": truth_label, "file": truth_file,
                               "definition": truth_definition, "events": truth_events,
                               "points": int(len(kx))}
            predictions: dict[tuple[str, str], np.ndarray] = {}
            for grid in GRIDS_RE:
                for model in completed_models:
                    curves = curves_by[(grid, model)]
                    names = list(curves)
                    for avg_len, name in zip(AVERAGING_LENGTHS, names):
                        m = l2.metrics(curves[name], truth_phase, truth_tau)
                        record["averaging"][f"{grid}:{model}"][str(avg_len)] = {
                            "relative_rms": m["relative_rms"], "r2": m["r2"],
                            "reattachment_x_over_H": m["reattachment_x_over_H"]}
                    final = l2.metrics(curves[names[-1]], truth_phase, truth_tau)
                    if analytic_tangent_x is not None:
                        final_x = l2.metrics(curves[names[-1]], truth_phase, truth_tau * analytic_tangent_x)
                        final["x_projected_truth_relative_rms"] = final_x["relative_rms"]
                        final["x_projected_truth_r2"] = final_x["r2"]
                    if re_h == 5600:
                        # documented secondary: the same run scored against the superseded
                        # Xiao reconstruction, so the size of the reference artefact stays visible
                        secondary = l2.metrics(curves[names[-1]], refs["xiao_5600"]["truth_phase"],
                                               refs["xiao_5600"]["truth_tau"])
                        final["secondary_score_vs_xiao_reconstruction"] = {
                            "relative_rms": secondary["relative_rms"], "r2": secondary["r2"],
                            "note": "superseded reference; see codes/results/m13_truth_reference_audit_*.json"}
                    record["metrics"][f"{grid}:{model}"] = final
                    predictions[(grid, model)] = l2.periodic_interp(np.asarray(curves[names[-1]]["phase"]),
                                                                    np.asarray(curves[names[-1]]["tau_s"]), dense)
            # paired phase-block bootstrap (imported from the deposited L3 analyser)
            sensitivity = {}
            primary = None
            for block in BLOCK_SENSITIVITY:
                samples = l3.circular_block_bootstrap(truth_dense, predictions, block_points=block,
                                                      draws=args.draws, seed=SEED + block)
                delta = samples[("G2c", "total_gradient_tble")] - samples[("G2c", "equilibrium")]
                sensitivity[str(block)] = {
                    "block_length_over_H": LX * block / DENSE_N,
                    "relative_rms_intervals": {f"{g}:{m}": interval(samples[(g, m)]) for g in GRIDS_RE for m in completed_models},
                    "finest_tble_minus_equilibrium_interval": interval(delta),
                    "probability_tble_error_exceeds_equilibrium": float(np.mean(delta > 0.0)),
                }
                if block == BLOCK_POINTS:
                    primary = samples
            record["phase_bootstrap_block_sensitivity"] = sensitivity
            record["phase_bootstrap_primary_intervals"] = sensitivity[str(BLOCK_POINTS)]["relative_rms_intervals"]
            for (g, m), s in primary.items():
                payload[f"re{re_h}_{g}_{m}_primary_bootstrap_relative_rms"] = s
            # exact eight-block sign-flip failure test on the finest grid
            fixed_blocks = DENSE_N // BLOCK_POINTS
            tests, raw_p = {}, {}
            for model in completed_models:
                difference = (predictions[("G2c", model)] - truth_dense) ** 2 - truth_dense ** 2
                block_values = np.asarray([np.mean(difference[i * BLOCK_POINTS:(i + 1) * BLOCK_POINTS]) for i in range(fixed_blocks)])
                t = l3.exact_block_sign_flip(block_values)
                tests[model] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()}
                raw_p[model] = t["p_one_sided"]
            adjusted = l3.holm_adjust(raw_p)
            for model in completed_models:
                tests[model]["p_one_sided_holm_two_models"] = adjusted[model]
                tests[model]["null"] = "error energy <= DNS signal energy (block means of e^2 - tau_DNS^2)"
            record["failure_significance_tests"] = tests
            # paired model comparison on the finest grid (deposited L3 protocol):
            # exact two-sided eight-block test on the squared-error difference and
            # the paired phase-block bootstrap interval of E_tau(TBLE) - E_tau(eq).
            loss_difference = ((predictions[("G2c", "total_gradient_tble")] - truth_dense) ** 2
                               - (predictions[("G2c", "equilibrium")] - truth_dense) ** 2)
            pair_blocks = np.asarray([np.mean(loss_difference[i * BLOCK_POINTS:(i + 1) * BLOCK_POINTS]) for i in range(fixed_blocks)])
            pair_test = l3.exact_block_sign_flip(pair_blocks)
            delta = primary[("G2c", "total_gradient_tble")] - primary[("G2c", "equilibrium")]
            eq_fine = record["metrics"]["G2c:equilibrium"]
            tb_fine = record["metrics"]["G2c:total_gradient_tble"]
            record["model_comparison"] = {
                "finest_relative_rms_tble_minus_equilibrium": tb_fine["relative_rms"] - eq_fine["relative_rms"],
                "finest_relative_rms_fractional_increase": tb_fine["relative_rms"] / eq_fine["relative_rms"] - 1.0,
                "paired_exact_block_test": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pair_test.items()},
                "primary_bootstrap_delta_interval": interval(delta),
                "primary_bootstrap_probability_tble_worse": float(np.mean(delta > 0.0)),
                "finest_signed_tangent_force_ratio": {m: record["metrics"][f"G2c:{m}"]["signed_tangent_force_per_span"] / record["metrics"][f"G2c:{m}"]["truth_signed_tangent_force_per_span"] for m in completed_models},
                "finest_one_envelope_subtracted_relative_rms": {
                    m: record["metrics"][f"G2c:{m}"]["relative_rms"] - (max(record["metrics"][f"{g}:{m}"]["relative_rms"] for g in GRIDS_RE) - min(record["metrics"][f"{g}:{m}"]["relative_rms"] for g in GRIDS_RE))
                    for m in completed_models},
            }
            # grid path convergence
            record["grid_path_convergence"] = {}
            for model in completed_models:
                for q in ("relative_rms", "r2", "reattachment_x_over_H", "reversed_fraction", "signed_tangent_force_per_span"):
                    vals = [record["metrics"][f"{g}:{model}"][q] for g in GRIDS_RE]
                    record["grid_path_convergence"][f"{model}:{q}"] = grid_path_descriptor(
                        l2, vals, [CELLS[g] for g in GRIDS_RE]
                    )
        else:
            # no wall-stress truth: events and window convergence from the coupled curves
            for grid in GRIDS_RE:
                for model in completed_models:
                    curves = curves_by[(grid, model)]
                    names = list(curves)
                    final = curves[names[-1]]
                    tau_dense = l2.periodic_interp(np.asarray(final["phase"]), np.asarray(final["tau_s"]), dense)
                    sep, rea = l2.zero_crossings(dense, tau_dense)
                    record["metrics"][f"{grid}:{model}"] = {
                        "separation_x_over_H": float(sep * LX), "reattachment_x_over_H": float(rea * LX),
                        "reversed_fraction": float(np.mean(tau_dense < 0.0)),
                        "cf_rms": float(2.0 * np.sqrt(np.mean(tau_dense ** 2))),
                        "signed_tangent_force_per_span": float(final["signed_tangent_force_per_span"]),
                    }
                    for avg_len, name in zip(AVERAGING_LENGTHS, names):
                        c = curves[name]
                        td = l2.periodic_interp(np.asarray(c["phase"]), np.asarray(c["tau_s"]), dense)
                        s_, r_ = l2.zero_crossings(dense, td)
                        record["averaging"][f"{grid}:{model}"][str(avg_len)] = {
                            "reattachment_x_over_H": float(r_ * LX), "cf_rms": float(2.0 * np.sqrt(np.mean(td ** 2)))}
            record["grid_path_convergence"] = {}
            for model in completed_models:
                for q in ("reattachment_x_over_H", "reversed_fraction", "cf_rms", "signed_tangent_force_per_span"):
                    vals = [record["metrics"][f"{g}:{model}"][q] for g in GRIDS_RE]
                    record["grid_path_convergence"][f"{model}:{q}"] = grid_path_descriptor(
                        l2, vals, [CELLS[g] for g in GRIDS_RE]
                    )

        # ---- inter-model traction discrepancy (all Re) with bootstrap ----
        inter = {}
        if len(completed_models) == 2:
            for grid in GRIDS_RE:
                ce = curves_by[(grid, "equilibrium")]
                ct = curves_by[(grid, "total_gradient_tble")]
                e = l2.periodic_interp(np.asarray(ce[list(ce)[-1]]["phase"]), np.asarray(ce[list(ce)[-1]]["tau_s"]), dense)
                t = l2.periodic_interp(np.asarray(ct[list(ct)[-1]]["phase"]), np.asarray(ct[list(ct)[-1]]["tau_s"]), dense)
                samples = l3.circular_block_bootstrap(e, {("x", "tble"): t}, block_points=BLOCK_POINTS, draws=min(args.draws, 5000), seed=SEED + re_h + CELLS[grid])
                inter[grid] = {"tble_vs_equilibrium_relative_rms": float(np.sqrt(np.mean((t - e) ** 2)) / np.sqrt(np.mean(e ** 2))),
                               "interval": interval(samples[("x", "tble")])}
        record["inter_model_traction"] = inter

        # ---- window convergence of the traction itself ----
        for grid in GRIDS_RE:
            for model in completed_models:
                curves = curves_by[(grid, model)]
                names = list(curves)
                last = np.asarray(curves[names[-1]]["tau_s"])
                prev = np.asarray(curves[names[-2]]["tau_s"])
                earl = np.asarray(curves[names[-3]]["tau_s"])
                norm = max(float(np.sqrt(np.mean(last ** 2))), 1.0e-14)
                record["averaging"][f"{grid}:{model}"]["change_180_to_225"] = float(np.sqrt(np.mean((prev - earl) ** 2)) / norm)
                record["averaging"][f"{grid}:{model}"]["change_225_to_270"] = float(np.sqrt(np.mean((last - prev) ** 2)) / norm)

        # ---- eps_c from the coupled fields (all Re) ----
        for grid in GRIDS_RE:
            for model in completed_models:
                curves = curves_by[(grid, model)]
                names = list(curves)
                te = truth_events["separation_x_over_H"] if truth_events else None
                tr = truth_events["reattachment_x_over_H"] if truth_events else None
                rec = eps_record(curves[names[-1]], press_by[(grid, model)][names[-1]], SEED + re_h + CELLS[grid] + (1 if model == "equilibrium" else 2), te, tr)
                record["eps_c"][f"{grid}:{model}"] = {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
                for k, v in rec.items():
                    if isinstance(v, np.ndarray):
                        payload[f"re{re_h}_{grid}_{model}_{k}"] = v
                final = curves[names[-1]]
                for k in ("phase", "x", "tau_s", "tau_x", "ym", "ywall", "wall_ds", "tangent_x", "tangent_y"):
                    payload[f"re{re_h}_{grid}_{model}_{k}"] = np.asarray(final[k])
                payload[f"re{re_h}_{grid}_{model}_dz"] = np.asarray(final["dz"])

        # ---- grid resolution in wall units from the run's own traction (equilibrium) ----
        for grid in GRIDS_RE:
            curves = curves_by[(grid, "equilibrium")]
            final = curves[list(curves)[-1]]
            u_tau = np.sqrt(np.abs(np.asarray(final["tau_s"])))
            if truth_tau is not None:
                u_tau_truth = np.sqrt(np.abs(l2.periodic_interp(truth_phase, truth_tau, np.asarray(final["phase"]))))
            else:
                u_tau_truth = u_tau
            record["grid_resolution"][grid] = {
                "matching_cell_index": 1,
                "nx": int(len(final["phase"])), "ny": int(manifests[(grid, "equilibrium")]["ny"]), "nz": int(manifests[(grid, "equilibrium")]["nz"]),
                "wall_arclength_dx_plus": l2.quantiles(np.asarray(final["wall_ds"]) * u_tau / nu),
                "first_cell_ym_plus": l2.quantiles(np.asarray(final["ym"]) * u_tau / nu),
                "spanwise_dz_plus": l2.quantiles(float(final["dz"]) * u_tau / nu),
                "first_cell_ym_plus_truth_based": l2.quantiles(np.asarray(final["ym"]) * u_tau_truth / nu) if truth_tau is not None else None,
                "wall_arclength_dx_over_H": l2.quantiles(np.asarray(final["wall_ds"])),
                "first_cell_ym_over_H": l2.quantiles(np.asarray(final["ym"])),
                "spanwise_dz_over_H": float(final["dz"]),
                "u_tau_source": "coupled equilibrium run wall traction" + ("; DNS-based variant also given" if truth_tau is not None else ""),
            }

        # ---- profile validation ----
        ref_keys = {5600: ["rapp_5600", "xiao_5600"],
                    10595: ["rapp_10595", "krank_10595", "breuer_10595", "mglet_10595"],
                    19000: ["rapp_19000"], 37000: ["rapp_37000", "mglet_37000"]}[re_h]
        for grid in GRIDS_RE:
            for model in completed_models:
                cid = case_id(re_h, grid, model)
                case = root / cid
                checkpoint = (case / "checkpoint_times_l2.txt").read_text().split()[-1]
                profiles = read_profiles(case, checkpoint)
                record["cases"][f"{grid}:{model}_bulk"].update(crest_bulk_velocity(profiles))
                record["profiles"][f"{grid}:{model}"] = {}
                for rk in ref_keys:
                    v = profile_validation(profiles, refs[rk])
                    record["profiles"][f"{grid}:{model}"][rk] = {k: val for k, val in v.items() if not isinstance(val, np.ndarray)}
                    for k, val in v.items():
                        if isinstance(val, np.ndarray):
                            payload[f"re{re_h}_{grid}_{model}_{rk}_{k}"] = val
                if grid == "G2c":
                    for key, data in profiles.items():
                        payload[f"re{re_h}_{grid}_{model}_profile_x{key.replace('.', 'p')}"] = data
        # reference-to-reference spread (how far apart the references themselves are)
        if re_h == 5600:
            record["reference_spread"] = {
                "rapp_vs_xiao": {k: v for k, v in reference_spread(refs, "rapp_5600", "xiao_5600").items() if not isinstance(v, np.ndarray)},
            }
        if re_h == 10595:
            record["reference_spread"] = {
                "rapp_vs_krank": {k: v for k, v in reference_spread(refs, "rapp_10595", "krank_10595").items() if not isinstance(v, np.ndarray)},
                "breuer_vs_krank": {k: v for k, v in reference_spread(refs, "breuer_10595", "krank_10595").items() if not isinstance(v, np.ndarray)},
                "rapp_vs_breuer": {k: v for k, v in reference_spread(refs, "rapp_10595", "breuer_10595").items() if not isinstance(v, np.ndarray)},
            }
        if re_h == 37000:
            record["reference_spread"] = {
                "rapp_vs_mglet": {k: v for k, v in reference_spread(refs, "rapp_37000", "mglet_37000").items() if not isinstance(v, np.ndarray)},
            }
        # experimental reattachment bracket
        record["experimental_reattachment"] = experimental_reattachment_bracket(refs[f"rapp_{re_h}"]["stations"])
        record["experimental_reattachment"]["reference"] = refs[f"rapp_{re_h}"]["label"]
        by_re[re_h] = record

    out["campaigns"] = {str(k): v for k, v in by_re.items()}

    # ------------------------------------------------------------------ #
    # Reynolds-number trend of the coupled cancellation parameter
    # ------------------------------------------------------------------ #
    trend: dict[str, Any] = {}
    for model in MODELS:
        for grid in ("G1c", "G2c"):
            xs, ys, lo, hi = [], [], [], []
            for re_h in res:
                key = f"{grid}:{model}"
                if key not in by_re[re_h]["eps_c"]:
                    continue
                e = by_re[re_h]["eps_c"][key]
                xs.append(re_h)
                ys.append(e["eps_c_median_separated"])
                lo.append(e["eps_c_median_separated_interval"]["low"])
                hi.append(e["eps_c_median_separated_interval"]["high"])
            xs_a, ys_a = np.asarray(xs, float), np.asarray(ys, float)
            ok = np.isfinite(ys_a) & (ys_a > 0)
            slope = float(np.polyfit(np.log(xs_a[ok]), np.log(ys_a[ok]), 1)[0]) if np.count_nonzero(ok) >= 2 else math.nan
            # slope uncertainty: resample each point uniformly within its bootstrap interval (log space)
            rng = np.random.default_rng(SEED + 7)
            slopes = []
            lo_a, hi_a = np.asarray(lo, float), np.asarray(hi, float)
            for _ in range(4000):
                draw = np.exp(rng.uniform(np.log(np.maximum(lo_a[ok], 1e-12)), np.log(np.maximum(hi_a[ok], 1e-12))))
                slopes.append(np.polyfit(np.log(xs_a[ok]), np.log(draw), 1)[0]) if np.count_nonzero(ok) >= 2 else None
            trend[f"{grid}:{model}"] = {
                "Re_H": xs, "eps_c_median_separated": ys, "interval_low": lo, "interval_high": hi,
                "log_slope": slope, "log_slope_interval": interval(np.asarray(slopes)) if slopes else None,
                "monotone_decreasing": bool(np.all(np.diff(ys_a[ok]) < 0)),
                "highest_interval_high_below_lowest_interval_low": bool(hi_a[ok][-1] < lo_a[ok][0]) if np.count_nonzero(ok) >= 2 else False,
            }
    legacy = anchor["eps_c"]["G2c:equilibrium"]
    trend["legacy_volume_average_re5600_G2c_equilibrium"] = {
        "eps_c_median_separated": legacy["eps_c_median_separated"],
        "note": "deposited run at crest bulk velocity 1.387 (Re_H,eff ~ 7,770); not on the ladder",
    }
    out["eps_c_reynolds_trend"] = trend
    out["provenance"] = provenance
    out["date"] = args.date

    # verdict fields used by the verifier (data-driven, no thresholds hidden here)
    verdict = {}
    for re_h in res:
        r = by_re[re_h]
        available_models = tuple(r["available_models"])
        if "truth" in r:
            verdict[f"re{re_h}_finest_relative_rms"] = {m: r["metrics"][f"G2c:{m}"]["relative_rms"] for m in available_models}
            verdict[f"re{re_h}_finest_relative_rms_interval"] = {m: r["phase_bootstrap_primary_intervals"][f"G2c:{m}"] for m in available_models}
            verdict[f"re{re_h}_failure_p_holm"] = {m: r["failure_significance_tests"][m]["p_one_sided_holm_two_models"] for m in available_models}
            verdict[f"re{re_h}_truth_events"] = r["truth"]["events"]
            verdict[f"re{re_h}_grid_path_relative_rms"] = {m: r["grid_path_convergence"][f"{m}:relative_rms"]["status"] for m in available_models}
        verdict[f"re{re_h}_available_models"] = list(available_models)
        verdict[f"re{re_h}_registered_failures"] = r["registered_failures"]
        verdict[f"re{re_h}_finest_reattachment"] = {m: r["metrics"][f"G2c:{m}"]["reattachment_x_over_H"] for m in available_models}
        verdict[f"re{re_h}_experimental_reattachment"] = r["experimental_reattachment"]
        verdict[f"re{re_h}_finest_profile_u_rms_mean_vs_rapp"] = {m: r["profiles"][f"G2c:{m}"][f"rapp_{re_h}"]["u_rms_mean"] for m in available_models}
        verdict[f"re{re_h}_finest_eps_c_median_separated"] = {m: r["eps_c"][f"G2c:{m}"]["eps_c_median_separated"] for m in available_models}
        verdict[f"re{re_h}_finest_eps_c_median_separated_interval"] = {m: r["eps_c"][f"G2c:{m}"]["eps_c_median_separated_interval"] for m in available_models}
        verdict[f"re{re_h}_finest_window_change_225_to_270"] = {m: r["averaging"][f"G2c:{m}"]["change_225_to_270"] for m in available_models}
        verdict[f"re{re_h}_finest_drive_halves_relative_difference"] = {m: r["drive_stationarity"][f"G2c:{m}"]["halves_relative_difference"] for m in available_models}
        verdict[f"re{re_h}_finest_crest_bulk_velocity_measured"] = {m: r["cases"][f"G2c:{m}_bulk"].get("crest_bulk_velocity_measured") for m in available_models}
        # Headline physical statement for the Re trend: at Re_H=37,000 the raw TBLE
        # eddy-viscosity request exceeds the vector-realizability bound at EVERY wall
        # face from t~108 onward, so the applied traction is the projected one
        # everywhere.  Reported, never gated.
        realizability = {}
        for m in available_models:
            summary_r = r["cases"].get(f"G2c:{m}_tble_realizability")
            if summary_r:
                realizability[m] = {
                    "minimum_clipped_fraction": summary_r.get("minimum_clipped_fraction"),
                    "maximum_clipped_fraction": summary_r.get("maximum_clipped_fraction"),
                    "maximum_vector_capped_fraction": summary_r.get("maximum_vector_capped_fraction"),
                    "maximum_mean_absolute_mismatch": summary_r.get("maximum_mean_absolute_mismatch"),
                }
        if realizability:
            verdict[f"re{re_h}_finest_tble_realizability"] = realizability
        verdict[f"re{re_h}_grid_resolution_G2c"] = {k: r["grid_resolution"]["G2c"][k] for k in ("first_cell_ym_plus", "wall_arclength_dx_plus", "spanwise_dz_plus")}
    out["verdict_inputs"] = verdict

    stem = f"m13_highre_coupled_{args.date}"
    out_dir = Path(_os.environ.get("M13_OUTPUT_DIR", ROOT / "codes" / "results"))
    out_json = out_dir / f"{stem}_summary.json"
    out_npz = out_dir / f"{stem}.npz"
    out_json.write_text(json.dumps(json_ready(out), indent=2, sort_keys=True, allow_nan=False) + "\n")
    payload["status"] = np.array(out["status"])
    payload["reynolds_numbers"] = np.asarray(list(res))
    payload["summary_json"] = np.array(out_json.name)
    np.savez_compressed(out_npz, **payload)
    print(f"M13_HIGHRE_COUPLED_OK res={res} summary={out_json} npz={out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
