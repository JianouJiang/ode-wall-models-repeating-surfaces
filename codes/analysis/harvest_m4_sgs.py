#!/usr/bin/env python3
"""Harvest the M4 SGS-sensitivity campaign into a ledger certificate.

Row M4 ("WM-SGS interaction untested") is closed when the coupled
wall-traction verdict of the deposited WALE campaign
(``codes/results/rswm_grid_results_l3_summary.json``) is reproduced, with the
same estimand, same DNS reference, same 4096-point phase grid, same paired
circular phase-block bootstrap and same exact sign-flip tests, under at least
two alternative subgrid-scale closures on the grids the verdict is stated on.

Inputs
------
* ``codes/results/rswm_m4_sgs_campaign_final/<case>/`` bundles written by
  ``jobs/rswm_m4_sgs_finalize.slurm`` (same layout as the deposited
  ``rswm_xiao_dns_grid_campaign_final_l2`` bundles plus ``sgs_swap_m4.json``).
* the locked Level-2 WALE reduction ``rswm_common_surface_grid_l2.npz`` and
  the Level-3 WALE certificate ``rswm_grid_results_l3_summary.json``.

Outputs
-------
``codes/results/m4_sgs_sensitivity_<date>.{json,npz}`` and
``codes/figures/fig_m4_sgs_sensitivity.{pdf,png}``.

Every reduction routine is imported from the locked Level-2/Level-3 producers;
nothing is retyped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
CAMPAIGN_ROOT = ROOT / "codes" / "results" / "rswm_m4_sgs_re_campaign_final"
# Corrected (crest-bulk u_b = 1, Re_H = 5600) WALE baseline of Agent B (M13), 2026-08-23.
# The original deposits (rswm_xiao_dns_grid_campaign_final_l2) held the DOMAIN-VOLUME average
# at 1 (crest bulk 1.39, Re_H,eff ~ 7,770) and are no longer the reference.
WALE_ROOT = ROOT / "codes" / "results" / "rswm_xiao_highre_campaign_m13_final" / "re5600"
L2_REDUCER = ROOT / "codes" / "analysis" / "rswm_common_surface_grid_l2.py"
def _resolve_l3_analyzer():
    """Locate analyze_grid_results_l3.py without depending on the pipeline's tree.

    development/nodes/ is ROTATED by the pipeline (levels are archived into
    development/exhausted_*/ when a tree is exhausted), so the live node path is not a
    durable address - it vanished on 2026-08-25 and broke this harvest.  Search the live
    node dir, then the stable generator copy, then the newest archive.  All copies are
    byte-identical and the recorded sha256 is unchanged by which one is found.
    Operator fix 2026-08-26 (same repair already applied to harvest_m13_highre.py).
    """
    cands = [ROOT / "development" / "nodes" / "node_004" / "analyze_grid_results_l3.py",
             ROOT / "codes" / "figures" / "node_generators" / "analyze_grid_results_l3.py"]
    cands.extend(sorted((ROOT / "development").glob(
        "exhausted_*/nodes/node_004/analyze_grid_results_l3.py"), reverse=True))
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError("analyze_grid_results_l3.py not found in: "
                            + ", ".join(str(c) for c in cands))


L3_ANALYZER = _resolve_l3_analyzer()
L3_JSON = ROOT / "codes" / "results" / "rswm_grid_results_l3_summary.json"   # superseded deposit, context only
MATRIX_FILE = ROOT / "jobs" / "rswm_m4_sgs_re_matrix.txt"
CREST_HEIGHT = 2.036
EXPECTED_UBAR = 0.721045

MODELS = ("equilibrium", "total_gradient_tble")
REFERENCE_SGS = "WALE"
WALE_CASE_IDS = {
    ("G0", "equilibrium"): "rswm_m13_re5600_g0_equilibrium_92160_v2",
    ("G0", "total_gradient_tble"): "rswm_m13_re5600_g0_tble_92160_v2",
    ("G1c", "equilibrium"): "rswm_m13_re5600_g1_equilibrium_307200_v2",
    ("G1c", "total_gradient_tble"): "rswm_m13_re5600_g1_tble_307200_v2",
    ("G2c", "equilibrium"): "rswm_m13_re5600_g2_equilibrium_819200_v2",
    ("G2c", "total_gradient_tble"): "rswm_m13_re5600_g2_tble_819200_v2",
}
DENSE_N = 4096
BOOTSTRAP_DRAWS = 20000
PRIMARY_BLOCK_POINTS = 512
BOOTSTRAP_SEED = 20260822 + PRIMARY_BLOCK_POINTS   # identical to the L3 primary draw
AVERAGING_LENGTHS = (180, 225, 270)

# Acceptance thresholds (stated in work_progress/archer2_campaign_20260823/M4/MANIFEST.md)
RATIO_LOW, RATIO_HIGH = 0.75, 1.3333333333
MARGIN_FRACTION = 0.5
ALPHA = 0.05


def verdict_side(point: float, low: float, high: float, threshold: float = 1.0) -> str:
    """Which side of the DNS-RMS threshold a case lands on, resolved by its interval.

    'above'  = coupled traction error exceeds the DNS RMS, interval-resolved
    'below'  = coupled traction error is under the DNS RMS, interval-resolved
    'straddles' = the 95% interval contains the threshold
    Deliberately outcome-neutral: M4 asks whether the SGS model changes the verdict,
    not which verdict it is.  (The 2026-08-25 reference correction moved the Re-5600
    coupled verdict from 'above' to 'below'; a criterion that hard-coded 'above' would
    have been a gate encoding an outcome.)
    """
    if low > threshold:
        return "above"
    if high < threshold:
        return "below"
    return "straddles"


def invariance_criterion(e_sgs: float, low_sgs: float, high_sgs: float, p_sgs: float,
                         e_wale: float, low_wale: float, high_wale: float, p_wale: float,
                         ratio_low: float, ratio_high: float, margin_fraction: float,
                         alpha: float) -> dict:
    """Is the coupled wall-traction verdict unchanged by the subgrid-scale model?

    A1 (sign / classification of the verdict) - the alternative-SGS run must land on the
        same side of the DNS-RMS threshold as the corrected WALE baseline, by point
        estimate, by interval classification, and by the exact block test's conclusion.
    A2 (magnitude class) - the error must stay inside a fixed ratio band of the WALE
        value, and the SGS-induced change must be smaller than half the distance from
        the WALE value to the threshold, so the SGS choice cannot be what decides the
        classification.
    """
    margin = abs(e_wale - 1.0)
    return {
        "A1_point_estimate_same_side_as_wale": bool((e_sgs > 1.0) == (e_wale > 1.0)),
        "A1_interval_classification_matches_wale": bool(
            verdict_side(e_sgs, low_sgs, high_sgs) == verdict_side(e_wale, low_wale, high_wale)),
        "A1_exact_test_conclusion_matches_wale": bool((p_sgs <= alpha) == (p_wale <= alpha)),
        "A2_ratio_within_class": bool(ratio_low <= e_sgs / e_wale <= ratio_high),
        "A2_change_below_half_threshold_margin": bool(abs(e_sgs - e_wale) < margin_fraction * margin),
    }


def resolve_ubar(bundle: Path, manifest: dict) -> tuple[float, str]:
    """Return the constrained volume-average velocity of a finalized bundle.

    The crest-bulk drive correction (Ubar = u_b*2.036*Lz*Lx/V_mesh = 0.721045 instead of
    the deposits' 1.0) is what puts the run at the STATED Re_H = 5600, so every bundle
    must prove it carries that constraint.  Two finalizer generations wrote two schemas:
    the M4 v2 finalizer hoists 'volume_average_Ubar'/'fvConstraints_Ubar' into the case
    manifest, while Agent B's M13 finalizer (job 14889058) keeps the deposited manifest
    schema and does not.  The authority in both cases is the dictionary the solver
    actually read, input/fvConstraints, so fall back to it and finally to the producer's
    own M13_BULK_VELOCITY telemetry line.  The check is never skipped: a bundle whose
    Ubar cannot be established anywhere is an error.
    """
    for key in ("volume_average_Ubar", "fvConstraints_Ubar"):
        if key in manifest:
            return float(manifest[key]), f"manifest['{key}']"
    driver = manifest.get("driver_manifest") or {}
    for key in ("volume_average_Ubar", "fvConstraints_Ubar"):
        if key in driver:
            return float(driver[key]), f"manifest['driver_manifest']['{key}']"
    constraints = bundle / "input" / "fvConstraints"
    if constraints.is_file():
        match = re.search(r"Ubar\s+\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)",
                          constraints.read_text())
        if match:
            if float(match.group(2)) or float(match.group(3)):
                raise SystemExit(f"{bundle.name}: drive is not streamwise: {match.group(0)}")
            return float(match.group(1)), "input/fvConstraints"
    telemetry = bundle / "producer_scheduler_output.txt"
    if telemetry.is_file():
        match = re.search(r"M13_BULK_VELOCITY .*volume_average_Ubar=([0-9.eE+-]+)",
                          telemetry.read_text(errors="replace"))
        if match:
            return float(match.group(1)), "producer_scheduler_output.txt"
    raise SystemExit(f"{bundle.name}: cannot establish the constrained volume-average velocity "
                     "(no manifest key, no input/fvConstraints, no producer telemetry)")


def check_ubar(bundle: Path, manifest: dict) -> dict:
    """Strict crest-bulk drive guard; returns the audited value and its source."""
    value, source = resolve_ubar(bundle, manifest)
    if abs(value - EXPECTED_UBAR) > 2.0e-3:
        raise SystemExit(f"{bundle.name} is not crest-bulk corrected: Ubar={value!r} "
                         f"(expected {EXPECTED_UBAR}) read from {source}")
    if int(manifest["Re_H"]) != 5600:
        raise SystemExit(f"{bundle.name} is not at Re_H = 5600: {manifest['Re_H']!r}")
    return {"volume_average_Ubar": value, "ubar_source": source}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_matrix(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        case_id, model, grid, cells, ny, nx, nz, ranks, job, sgs = line.split("|")
        rows.append(dict(case_id=case_id, model=model, grid=grid, cells=int(cells),
                         ny=int(ny), nx=int(nx), nz=int(nz), ranks=int(ranks),
                         producer_job=job, sgs=sgs))
    return rows


def read_scalar_field(path: Path) -> np.ndarray:
    """Read a nonuniform ascii volScalarField internalField."""
    import re
    text = path.read_text(errors="strict")
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", text)
    if match is None:
        raise ValueError(f"nonuniform scalar internalField not found in {path}")
    count = int(match.group(1))
    body = text[match.end():]
    values = np.fromstring(body[:body.index(")")], sep="\n")
    if values.shape != (count,):
        raise ValueError(f"expected {count} scalars in {path}, found {values.shape}")
    return values


def first_cell_nut_over_nu(l2: Any, case: Path, nu: float) -> dict[str, np.ndarray | float]:
    """Spanwise/area-averaged instantaneous SGS viscosity in the wall-adjacent
    (matching) cells at the terminal time, per streamwise phase, over nu.

    This is the 'power check': it documents that the SGS swap changes the
    eddy viscosity seen by the wall model at the matching height.
    """
    from verify_common_matching_surface import read_labels, read_patch, read_points, read_faces
    mesh_dir = case / "input" / "polyMesh"
    owners = read_labels(mesh_dir / "owner")
    start, count = read_patch(mesh_dir / "boundary", "bottomWall")
    nut = read_scalar_field(case / "latest_time" / "nut")
    mesh = l2.mesh_bottom(case)
    first = nut[owners[start:start + count]] / nu
    rounded = np.round(mesh["xyz"][:, 0], 9)
    x_unique, inverse = np.unique(rounded, return_inverse=True)
    phase_mean = np.asarray([
        np.average(first[inverse == i], weights=mesh["area"][inverse == i]) for i in range(len(x_unique))
    ])
    return {
        "phase": x_unique / l2.LX,
        "nut_over_nu_first_cell": phase_mean,
        "period_mean": float(np.average(phase_mean, weights=np.asarray([
            np.sum(mesh["area"][inverse == i]) for i in range(len(x_unique))]))),
        "period_median": float(np.median(phase_mean)),
        "period_max": float(np.max(phase_mean)),
    }


def crest_bulk_velocity(case: Path, checkpoint: str) -> float:
    """Q/2.036 from the mid-channel UMean profile (x = 4.05, full y column).

    Mass conservation on the periodic hill makes the volume flux per unit
    span the same at every station; dividing by the crest height 2.036 H
    gives the crest-section bulk velocity that defines Re_H.  Must be ~1.
    """
    path = case / "postProcessing_sampleProfiles" / checkpoint / "x04p050.xy"
    if not path.is_file():
        path = sorted((case / "postProcessing_sampleProfiles" / checkpoint).glob("x04p*.xy"))[0]
    data = np.loadtxt(path)
    y, u = data[:, 0], data[:, 1]
    return float(np.trapz(u, y) / CREST_HEIGHT)


def load_case(l2: Any, case: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads((case / "MANIFEST.json").read_text())
    mesh = l2.mesh_bottom(case)
    names = (case / "checkpoint_times_l2.txt").read_text().split()
    curves = {}
    for name in names:
        rows = l2.sample_rows(case / "postProcessing_sampleBottomWall" / name / "bottomWall.xy")
        curves[name] = l2.phase_reduce(mesh, rows)
    return manifest, curves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    out_json = ROOT / "codes" / "results" / f"m4_sgs_sensitivity_{args.date}.json"
    out_npz = ROOT / "codes" / "results" / f"m4_sgs_sensitivity_{args.date}.npz"
    fig_pdf = ROOT / "codes" / "figures" / "fig_m4_sgs_sensitivity.pdf"
    fig_png = ROOT / "codes" / "figures" / "fig_m4_sgs_sensitivity.png"

    if not CAMPAIGN_ROOT.is_dir():
        raise SystemExit(f"campaign bundle absent: {CAMPAIGN_ROOT} (run archer2_run.sh down)")
    campaign = json.loads((CAMPAIGN_ROOT / "CAMPAIGN_MANIFEST.json").read_text())
    if not WALE_ROOT.is_dir():
        raise SystemExit(f"corrected WALE baseline absent: {WALE_ROOT} (Agent B finalizer)")
    wale_campaign = json.loads((WALE_ROOT / "CAMPAIGN_MANIFEST.json").read_text())
    l2 = load_module("rswm_l2_locked", L2_REDUCER)
    l3 = load_module("rswm_l3_locked", L3_ANALYZER)
    l3_summary = json.loads(L3_JSON.read_text())

    if sha256(l2.DNS_FILE) != l2.DNS_EXPECTED_SHA256:
        raise SystemExit("authoritative corrected DNS file identity changed")
    dns = np.load(l2.DNS_FILE)

    # ---- wall-traction reference ------------------------------------------------
    # The 4-point through-origin fit on the Xiao velocity archive was WITHDRAWN as a
    # scoring reference on 2026-08-25 (its wall spacing is 7.5x the MGLET deposit's, so
    # the fit is unconverged and biased low).  The primary reference is now the Peller &
    # Manhart MGLET DNS bottom-wall traction; the curvature-aware repaired cubic on the
    # same Xiao archive is the sensitivity bracket; the withdrawn linear-4 estimator is
    # carried only to reproduce the superseded numbers.  All three come from the shared
    # module so M4 and R2-m4 cannot drift apart.  M4 asks whether the VERDICT is
    # invariant across SGS models, and that question is asked on each reference in turn.
    sys.path.insert(0, str(ROOT / "codes" / "analysis"))
    import r2m4_truth_references as truth_module  # noqa: E402
    all_references = truth_module.references()
    truth_phase, truth_tau, truth_label = all_references[truth_module.PRIMARY]
    truth_phase = np.asarray(truth_phase, float)
    truth_tau = np.asarray(truth_tau, float)
    # the archive tau_w column is the withdrawn linear-4 estimator on a different phase grid;
    # it is carried only as a named reference in reference_sensitivity, never as a metric input
    truth_tau_x_legacy = None
    tangent_audit = {
        "primary_reference": truth_module.PRIMARY,
        "primary_label": truth_label,
        "bracket_reference": truth_module.BRACKET,
        "superseded_reference": truth_module.SUPERSEDED,
        "note": ("the deposited Xiao 4-point through-origin wall-gradient fit was withdrawn as a "
                 "scoring reference on 2026-08-25; MGLET is primary, the repaired cubic is the "
                 "bracket, and every headline is reported on all three"),
        "module": "codes/analysis/r2m4_truth_references.py",
    }
    dense_phase = np.arange(DENSE_N, dtype=float) / DENSE_N
    truth_dense = l2.periodic_interp(truth_phase, truth_tau, dense_phase)
    reference_grids = {
        name: l2.periodic_interp(np.asarray(ph, float), np.asarray(tv, float), dense_phase)
        for name, (ph, tv, _lab) in all_references.items()
    }

    # ---- WALE reference curves from Agent B's corrected Re=5600 bundles ----------
    reference_curves: dict[tuple[str, str], np.ndarray] = {}
    reference_metrics: dict[tuple[str, str], dict[str, float]] = {}
    reference_manifests: dict[tuple[str, str], dict[str, Any]] = {}
    reference_terminal: dict[tuple[str, str], dict[str, Any]] = {}
    reference_bulk: dict[str, float] = {}
    reference_ubar: dict[tuple[str, str], dict] = {}
    for (grid, model), case_id in WALE_CASE_IDS.items():
        bundle = WALE_ROOT / case_id
        if not (bundle / "MANIFEST.json").is_file():
            continue
        manifest, by_time = load_case(l2, bundle)
        if manifest["grid"] != grid or manifest["model"] != model or int(manifest["Re_H"]) != 5600:
            raise SystemExit(f"WALE reference identity mismatch in {case_id}")
        reference_ubar[(grid, model)] = check_ubar(bundle, manifest)
        names = list(by_time)
        terminal = by_time[names[-1]]
        reference_terminal[(grid, model)] = terminal
        reference_manifests[(grid, model)] = manifest
        reference_curves[(grid, model)] = l2.periodic_interp(
            np.asarray(terminal["phase"]), np.asarray(terminal["tau_s"]), dense_phase)
        values = l2.metrics(terminal, truth_phase, truth_tau)
        profile = l2.profile_metrics(bundle, names[-1], dns, terminal)
        values["profile_u_rms_mean"] = profile["profile_u_rms_mean"]
        values["profile_u_rms_max"] = profile["profile_u_rms_max"]
        reference_metrics[(grid, model)] = values
        reference_bulk[f"{REFERENCE_SGS}:{grid}:{model}"] = crest_bulk_velocity(bundle, names[-1])
    if not reference_curves:
        raise SystemExit("no corrected WALE reference bundle found")

    # ---- campaign cases --------------------------------------------------------
    matrix = read_matrix(MATRIX_FILE) if MATRIX_FILE.is_file() else []
    cases: dict[tuple[str, str, str], dict[str, Any]] = {}
    case_ubar: dict[tuple[str, str, str], dict] = {}
    curves: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for case_dir in sorted(p for p in CAMPAIGN_ROOT.iterdir() if (p / "MANIFEST.json").is_file()):
        manifest, case_curves = load_case(l2, case_dir)
        key = (manifest["sgs_model"], manifest["grid"], manifest["model"])
        if key in cases:
            raise SystemExit(f"duplicate case for {key}")
        if args.verify_hashes:
            for relative, record in manifest["files"].items():
                path = case_dir / relative
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise SystemExit(f"manifest mismatch: {case_dir.name}/{relative}")
        swap = json.loads((case_dir / "sgs_swap_m4.json").read_text())
        if swap["sgs_model"] != manifest["sgs_model"] or swap["status"] != "SGS_SWAP_OK":
            raise SystemExit(f"SGS audit mismatch in {case_dir.name}")
        if swap["deposited_wale_sha256"] != "0147bd493928af1713ccd974703c3ea3378c24dc41e4fda69d44636c293b62b8":
            raise SystemExit("SGS swap did not start from the deposited WALE dictionary")
        case_ubar[(manifest["sgs_model"], manifest["grid"], manifest["model"])] = check_ubar(case_dir, manifest)
        cases[key] = {"manifest": manifest, "dir": case_dir, "swap": swap}
        curves[key] = case_curves
    if not cases:
        raise SystemExit("no finalized M4 case found")

    sgs_models = sorted({key[0] for key in cases})
    grids = sorted({key[1] for key in cases}, key=lambda g: {"G0": 0, "G1c": 1, "G2c": 2}[g])

    # ---- per-case metrics on the common phase grid -----------------------------
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    base: dict[str, Any] = {}
    temporal: dict[str, Any] = {}
    surface_check: dict[str, Any] = {}
    for key, by_time in curves.items():
        sgs, grid, model = key
        names = list(by_time)
        terminal = by_time[names[-1]]
        predictions[key] = l2.periodic_interp(
            np.asarray(terminal["phase"]), np.asarray(terminal["tau_s"]), dense_phase)
        values = l2.metrics(terminal, truth_phase, truth_tau)
        profile = l2.profile_metrics(cases[key]["dir"], names[-1], dns, terminal)
        values["profile_u_rms_mean"] = profile["profile_u_rms_mean"]
        values["profile_u_rms_max"] = profile["profile_u_rms_max"]
        base[":".join(key)] = values
        records = []
        for averaging_length, name in zip(AVERAGING_LENGTHS, names):
            current = l2.metrics(by_time[name], truth_phase, truth_tau)
            records.append({"averaging_length": averaging_length, "checkpoint": name,
                            "relative_rms": current["relative_rms"], "r2": current["r2"],
                            "reattachment_x_over_H": current["reattachment_x_over_H"]})
        rel = np.asarray([item["relative_rms"] for item in records])
        temporal[":".join(key)] = {
            "records": records,
            "relative_rms_envelope": float(np.ptp(rel)),
            "relative_rms_last_change": float(rel[-1] - rel[-2]),
            "relative_rms_last_change_fraction_of_terminal": float(abs(rel[-1] - rel[-2]) / rel[-1]),
        }
        # the swapped-SGS case must sit on the SAME physical matching surface as WALE
        if (grid, model) not in reference_terminal:
            raise SystemExit(f"no corrected WALE reference for {grid}:{model}")
        ref_terminal = reference_terminal[(grid, model)]
        wale_ym = l2.periodic_interp(np.asarray(ref_terminal["phase"]), np.asarray(ref_terminal["ym"]),
                                     np.asarray(terminal["phase"]))
        surface_check[":".join(key)] = {
            "max_relative_ym_mismatch_vs_wale": float(
                np.max(np.abs(np.asarray(terminal["ym"]) - wale_ym)) / np.mean(wale_ym)),
            "polyMesh_points_sha256_equals_wale_case": (
                sha256(cases[key]["dir"] / "input" / "polyMesh" / "points")
                == sha256(WALE_ROOT / WALE_CASE_IDS[(grid, model)] / "input" / "polyMesh" / "points")),
            "crest_bulk_velocity": crest_bulk_velocity(cases[key]["dir"], names[-1]),
            "wale_crest_bulk_velocity": reference_bulk[f"{REFERENCE_SGS}:{grid}:{model}"],
            "volume_average_Ubar": case_ubar[key]["volume_average_Ubar"],
            "volume_average_Ubar_source": case_ubar[key]["ubar_source"],
            "wale_volume_average_Ubar": reference_ubar[(grid, model)]["volume_average_Ubar"],
            "wale_volume_average_Ubar_source": reference_ubar[(grid, model)]["ubar_source"],
            "nu": float(cases[key]["manifest"]["nu"]),
            "wale_nu": float(reference_manifests[(grid, model)]["nu"]),
        }

    # ---- power check: first-cell SGS viscosity, new SGS vs deposited WALE ----------
    nu = float(np.median(np.asarray(dns["nu"], float)))
    wale_root = WALE_ROOT
    wale_case_ids = WALE_CASE_IDS
    power_check: dict[str, Any] = {}
    wale_nut: dict[tuple[str, str], dict[str, Any]] = {}
    nut_curves: dict[str, np.ndarray] = {}
    for key in cases:
        sgs, grid, model = key
        label = ":".join(key)
        if (grid, model) not in wale_nut:
            wale_nut[(grid, model)] = first_cell_nut_over_nu(l2, wale_root / wale_case_ids[(grid, model)], nu)
            nut_curves[f"{REFERENCE_SGS}_{grid}_{model}"] = wale_nut[(grid, model)]["nut_over_nu_first_cell"]
        current = first_cell_nut_over_nu(l2, cases[key]["dir"], nu)
        nut_curves["_".join(key)] = current["nut_over_nu_first_cell"]
        reference = wale_nut[(grid, model)]
        power_check[label] = {
            "first_cell_nut_over_nu_median": current["period_median"],
            "first_cell_nut_over_nu_max": current["period_max"],
            "wale_first_cell_nut_over_nu_median": reference["period_median"],
            "wale_first_cell_nut_over_nu_max": reference["period_max"],
            "median_ratio_to_wale": current["period_median"] / max(reference["period_median"], 1.0e-14),
            "note": "instantaneous t=405 SGS viscosity, area-weighted over the span, in the wall-adjacent cell that feeds the wall model",
        }

    # ---- paired phase-block bootstrap: all new cases + WALE references together --
    pool: dict[tuple[str, str, str], np.ndarray] = dict(predictions)
    for (grid, model), curve in reference_curves.items():
        pool[(REFERENCE_SGS, grid, model)] = curve
    samples = l3.circular_block_bootstrap(
        truth_dense, pool, block_points=PRIMARY_BLOCK_POINTS, draws=args.draws, seed=BOOTSTRAP_SEED)
    intervals = {":".join(key): l3.interval(values) for key, values in samples.items()}

    # ---- exact failure tests (error energy > DNS energy), Holm over new cases ---
    fixed_blocks = DENSE_N // PRIMARY_BLOCK_POINTS
    failure_tests: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    wale_raw_p: dict[str, float] = {}
    for key in pool:
        if key[0] != REFERENCE_SGS:
            continue
        difference = (pool[key] - truth_dense) ** 2 - truth_dense**2
        blocks = np.asarray([np.mean(difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
                             for i in range(fixed_blocks)])
        wale_raw_p[":".join(key)] = l3.exact_block_sign_flip(blocks)["p_one_sided"]
    for key, prediction in predictions.items():
        difference = (prediction - truth_dense) ** 2 - truth_dense**2
        blocks = np.asarray([np.mean(difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
                             for i in range(fixed_blocks)])
        test = l3.exact_block_sign_flip(blocks)
        failure_tests[":".join(key)] = test
        raw_p[":".join(key)] = test["p_one_sided"]
    # Primary correction matches the deposited WALE protocol: Holm over the two
    # wall models sharing one SGS closure and one grid.  The whole-table Holm
    # (all alternative-SGS cases at once) is reported as a secondary figure.
    adjusted_all = l3.holm_adjust(raw_p)
    adjusted: dict[str, float] = {}
    for sgs in {key[0] for key in predictions}:
        for grid in {key[1] for key in predictions}:
            family = {":".join(key): raw_p[":".join(key)] for key in predictions
                      if key[0] == sgs and key[1] == grid}
            if family:
                adjusted.update(l3.holm_adjust(family))
    for label in failure_tests:
        failure_tests[label]["p_one_sided_holm_two_models"] = adjusted[label]
        failure_tests[label]["p_one_sided_holm_all_sgs_cases"] = adjusted_all[label]

    # ---- invariance table: SGS x model x grid -----------------------------------
    table: dict[str, Any] = {}
    all_pass = True
    for key in sorted(predictions):
        sgs, grid, model = key
        label = ":".join(key)
        ref = reference_metrics[(grid, model)]
        e_sgs = base[label]["relative_rms"]
        e_wale = ref["relative_rms"]
        delta = samples[key] - samples[(REFERENCE_SGS, grid, model)]
        delta_interval = l3.interval(delta)
        loss_difference = (predictions[key] - truth_dense) ** 2 - (reference_curves[(grid, model)] - truth_dense) ** 2
        paired_sgs_test = l3.exact_block_sign_flip(np.asarray([
            np.mean(loss_difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
            for i in range(fixed_blocks)]))
        ratio = e_sgs / e_wale
        margin = abs(e_wale - 1.0)
        wale_label = f"{REFERENCE_SGS}:{grid}:{model}"
        wale_interval = intervals[wale_label]
        criterion = invariance_criterion(
            e_sgs, intervals[label]["low"], intervals[label]["high"], adjusted[label],
            e_wale, wale_interval["low"], wale_interval["high"], wale_raw_p[wale_label],
            RATIO_LOW, RATIO_HIGH, MARGIN_FRACTION, ALPHA)
        passed = all(criterion.values())
        all_pass &= passed
        table[label] = {
            "sgs": sgs, "grid": grid, "model": model,
            "relative_rms": e_sgs,
            "relative_rms_interval_95": intervals[label],
            "wale_relative_rms": e_wale,
            "wale_relative_rms_interval_95": intervals[f"{REFERENCE_SGS}:{grid}:{model}"],
            "sgs_minus_wale_interval_95": delta_interval,
            "sgs_minus_wale_paired_exact_p_two_sided": paired_sgs_test["p_two_sided"],
            "ratio_to_wale": ratio,
            "verdict_side": verdict_side(e_sgs, intervals[label]["low"], intervals[label]["high"]),
            "wale_verdict_side": verdict_side(e_wale, wale_interval["low"], wale_interval["high"]),
            "threshold_margin_over_abs_sgs_change": (margin / abs(e_sgs - e_wale)) if e_sgs != e_wale else float("inf"),
            "wale_p_one_sided_failure": wale_raw_p[wale_label],
            "p_one_sided_failure": failure_tests[label]["p_one_sided"],
            "p_one_sided_failure_holm": adjusted[label],
            "p_one_sided_failure_holm_all_cases": adjusted_all[label],
            "r2": base[label]["r2"], "wale_r2": ref["r2"],
            "reattachment_x_over_H": base[label]["reattachment_x_over_H"],
            "wale_reattachment_x_over_H": ref["reattachment_x_over_H"],
            "reversed_fraction": base[label]["reversed_fraction"],
            "wale_reversed_fraction": ref["reversed_fraction"],
            "sign_accuracy": base[label]["sign_accuracy"],
            "profile_u_rms_mean": base[label]["profile_u_rms_mean"],
            "wale_profile_u_rms_mean": ref["profile_u_rms_mean"],
            "averaging_window_envelope": temporal[label]["relative_rms_envelope"],
            "averaging_window_last_change_fraction": temporal[label]["relative_rms_last_change_fraction_of_terminal"],
            "criterion": criterion,
            "verdict_invariant": passed,
        }

    # ---- reference sensitivity: repeat the whole invariance question on each of the
    # three wall-traction references (MGLET primary, repaired cubic bracket, withdrawn
    # linear-4).  Bootstrap intervals are recomputed against each reference so the
    # classification is not carried over from the primary one.
    reference_sensitivity: dict[str, Any] = {}
    for reference_name, reference_truth in reference_grids.items():
        ref_samples = l3.circular_block_bootstrap(
            reference_truth, pool, block_points=PRIMARY_BLOCK_POINTS,
            draws=args.draws, seed=BOOTSTRAP_SEED)
        ref_p: dict[str, float] = {}
        for key, prediction in pool.items():
            difference = (prediction - reference_truth) ** 2 - reference_truth**2
            blocks = np.asarray([np.mean(difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
                                 for i in range(fixed_blocks)])
            ref_p[":".join(key)] = l3.exact_block_sign_flip(blocks)["p_one_sided"]
        rows: dict[str, Any] = {}
        every = True
        for key in sorted(predictions):
            sgs, grid, model = key
            label, wlabel = ":".join(key), f"{REFERENCE_SGS}:{grid}:{model}"
            e_s = float(np.sqrt(np.mean((pool[key] - reference_truth) ** 2))
                        / np.sqrt(np.mean(reference_truth ** 2)))
            e_w = float(np.sqrt(np.mean((pool[(REFERENCE_SGS, grid, model)] - reference_truth) ** 2))
                        / np.sqrt(np.mean(reference_truth ** 2)))
            iv_s, iv_w = l3.interval(ref_samples[key]), l3.interval(ref_samples[(REFERENCE_SGS, grid, model)])
            crit = invariance_criterion(e_s, iv_s["low"], iv_s["high"], ref_p[label],
                                        e_w, iv_w["low"], iv_w["high"], ref_p[wlabel],
                                        RATIO_LOW, RATIO_HIGH, MARGIN_FRACTION, ALPHA)
            every &= all(crit.values())
            rows[label] = {
                "relative_rms": e_s, "relative_rms_interval_95": iv_s,
                "wale_relative_rms": e_w, "wale_relative_rms_interval_95": iv_w,
                "ratio_to_wale": e_s / e_w,
                "verdict_side": verdict_side(e_s, iv_s["low"], iv_s["high"]),
                "wale_verdict_side": verdict_side(e_w, iv_w["low"], iv_w["high"]),
                "criterion": crit, "verdict_invariant": all(crit.values()),
            }
        reference_sensitivity[reference_name] = {
            "label": all_references[reference_name][2],
            "role": ("primary" if reference_name == truth_module.PRIMARY else
                     "bracket" if reference_name == truth_module.BRACKET else "superseded"),
            "all_cases_verdict_invariant": every,
            "wale_verdict_side": {f"{g}:{m}": rows[f"{s}:{g}:{m}"]["wale_verdict_side"]
                                  for (s, g, m) in sorted(predictions)},
            "rows": rows,
        }

    # ---- model ranking under each SGS (reported, not gating) --------------------
    ranking: dict[str, Any] = {}
    for sgs in sgs_models:
        for grid in grids:
            k_t, k_e = (sgs, grid, "total_gradient_tble"), (sgs, grid, "equilibrium")
            if k_t not in predictions or k_e not in predictions:
                continue
            loss_difference = (predictions[k_t] - truth_dense) ** 2 - (predictions[k_e] - truth_dense) ** 2
            test = l3.exact_block_sign_flip(np.asarray([
                np.mean(loss_difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
                for i in range(fixed_blocks)]))
            delta = samples[k_t] - samples[k_e]
            ranking[f"{sgs}:{grid}"] = {
                "tble_minus_equilibrium_relative_rms": base[":".join(k_t)]["relative_rms"] - base[":".join(k_e)]["relative_rms"],
                "tble_minus_equilibrium_interval_95": l3.interval(delta),
                "paired_exact_p_two_sided": test["p_two_sided"],
                "rankable_at_5pct": bool(test["p_two_sided"] <= ALPHA),
                "tble_point_estimate_worse": bool(base[":".join(k_t)]["relative_rms"] > base[":".join(k_e)]["relative_rms"]),
            }
    for grid in grids:
        k_t, k_e = (REFERENCE_SGS, grid, "total_gradient_tble"), (REFERENCE_SGS, grid, "equilibrium")
        if k_t not in pool or k_e not in pool:
            continue
        loss_difference = (pool[k_t] - truth_dense) ** 2 - (pool[k_e] - truth_dense) ** 2
        test = l3.exact_block_sign_flip(np.asarray([
            np.mean(loss_difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
            for i in range(fixed_blocks)]))
        ranking[f"{REFERENCE_SGS}:{grid}"] = {
            "tble_minus_equilibrium_relative_rms": reference_metrics[(grid, "total_gradient_tble")]["relative_rms"]
                                                   - reference_metrics[(grid, "equilibrium")]["relative_rms"],
            "tble_minus_equilibrium_interval_95": l3.interval(samples[k_t] - samples[k_e]),
            "paired_exact_p_two_sided": test["p_two_sided"],
            "rankable_at_5pct": bool(test["p_two_sided"] <= ALPHA),
            "tble_point_estimate_worse": bool(reference_metrics[(grid, "total_gradient_tble")]["relative_rms"]
                                              > reference_metrics[(grid, "equilibrium")]["relative_rms"]),
        }
    # WALE reference rows (same estimand, same draws) and their own failure tests
    wale_table: dict[str, Any] = {}
    for (grid, model), values in reference_metrics.items():
        key = (REFERENCE_SGS, grid, model)
        difference = (pool[key] - truth_dense) ** 2 - truth_dense**2
        blocks = np.asarray([np.mean(difference[i * PRIMARY_BLOCK_POINTS:(i + 1) * PRIMARY_BLOCK_POINTS])
                             for i in range(fixed_blocks)])
        test = l3.exact_block_sign_flip(blocks)
        wale_table[":".join(key)] = {
            "relative_rms": values["relative_rms"], "r2": values["r2"],
            "relative_rms_interval_95": intervals[":".join(key)],
            "p_one_sided_failure": test["p_one_sided"],
            "reattachment_x_over_H": values["reattachment_x_over_H"],
            "reversed_fraction": values["reversed_fraction"],
            "profile_u_rms_mean": values["profile_u_rms_mean"],
            "crest_bulk_velocity": reference_bulk[":".join(key)],
            "producer_job_id": reference_manifests[(grid, model)]["producer_job_id"],
            "superseded_deposit_relative_rms_volume_average_drive": (
                l3_summary["base_metrics"].get(f"{grid}:{model}", {}).get("relative_rms")),
        }

    # ---- completeness --------------------------------------------------------------
    alternative = [s for s in sgs_models if s != REFERENCE_SGS]
    required_grids = ("G1c", "G2c")
    missing = [f"{s}:{g}:{m}" for s in alternative for g in required_grids for m in MODELS
               if (s, g, m) not in predictions]
    complete = len(alternative) >= 2 and not missing
    cost = {}
    for key, record in cases.items():
        m = record["manifest"]
        cost[":".join(key)] = {
            "producer_elapsed_seconds": m["producer_elapsed_seconds"],
            "mpi_ranks": m["mpi_ranks"],
            "node_hours": m["producer_elapsed_seconds"] / 3600.0 * m["mpi_ranks"] / 128.0,
            "clock_seconds_per_step": m["solver_cost"]["clock_seconds_per_step"],
            "time_steps": m["solver_cost"]["time_steps"],
            "maximum_courant": m["maximum_courant"],
        }

    status = ("M4_SGS_SENSITIVITY_OK" if complete and all_pass else
              "M4_SGS_SENSITIVITY_INCOMPLETE" if not complete else
              "M4_SGS_SENSITIVITY_VERDICT_NOT_INVARIANT")
    summary = {
        "status": status,
        "row": "M4",
        "row_text": "WM-SGS interaction untested: decisive coupled cases repeated with >=2 SGS models; verdict invariant across SGS",
        "scope": ("Xiao alpha=1 Re_H=5600 (crest-bulk u_b=1) periodic hill; coupled Foundation-10 WMLES; "
                  "Agent B's corrected WALE matrix is the control; only constant/momentumTransport (plus the transported "
                  "SGS fields and their scheme/solver entries) differs; same mesh, matching surface, "
                  "schemes, dt/Co, averaging window 135-405 and post-processing."),
        "estimand": "phase-averaged RMS-normalised physical-tangent wall-traction error E_tau (eq. coupled_error)",
        "acceptance": {
            "A1": ("the alternative-SGS run lands on the same side of the DNS-RMS threshold as the "
                   "corrected WALE baseline - by point estimate, by 95% phase-block interval "
                   "classification, and by the conclusion of the exact one-sided block test at "
                   "alpha=0.05 (Holm over the two wall models per SGS and grid). Outcome-neutral: "
                   "M4 asks whether the SGS model CHANGES the verdict, not which verdict it is"),
            "A2": (f"{RATIO_LOW:.2f} <= E_tau(sgs)/E_tau(WALE) <= {RATIO_HIGH:.2f} and "
                   f"|E_tau(sgs)-E_tau(WALE)| < {MARGIN_FRACTION} |E_tau(WALE)-1| for the same model "
                   "and grid, so the SGS choice cannot be what decides the classification"),
            "completeness": "at least two alternative SGS closures x two wall models x grids G1c and G2c",
            "reported_not_gating": "TBLE-vs-equilibrium ranking under each SGS; reattachment, reversed-shear coverage, profile RMS; averaging-window envelopes",
        },
        "sgs_models": sgs_models + ([REFERENCE_SGS] if REFERENCE_SGS not in sgs_models else []),
        "sgs_descriptions": {key[0]: record["swap"]["sgs_description"] for key, record in cases.items()},
        "grids": grids,
        "models": list(MODELS),
        "complete": complete,
        "missing_cases": missing,
        "all_cases_verdict_invariant": all_pass,
        "table": table,
        "reference_sensitivity": reference_sensitivity,
        "invariance_holds_on_every_reference": all(
            block["all_cases_verdict_invariant"] for block in reference_sensitivity.values()),
        "wale_reference_table": wale_table,
        "wale_reference_root": str(WALE_ROOT.relative_to(ROOT)),
        "wale_reference_producer_jobs": {":".join((REFERENCE_SGS, g, m)): man["producer_job_id"]
                                         for (g, m), man in reference_manifests.items()},
        "wale_reference_campaign_status": wale_campaign.get("status"),
        "bulk_velocity_convention": ("crest-section u_b = 1 (meanVelocityForce Ubar = 2.036 Lz Lx / V_mesh = "
                                     f"{EXPECTED_UBAR}); the 2026-08-22 deposits drove the volume average and are superseded"),
        "base_metrics": base,
        "temporal_sensitivity": temporal,
        "matching_surface_check": surface_check,
        "sgs_power_check_first_cell_nut": power_check,
        "model_ranking_by_sgs": ranking,
        "failure_significance_tests": failure_tests,
        "phase_bootstrap_intervals": intervals,
        "bootstrap_protocol": {
            "estimand": "phase-averaged RMS-normalised physical-tangent wall-traction error",
            "dense_phase_points": DENSE_N, "draws": args.draws, "paired": True, "circular": True,
            "primary_block_points": PRIMARY_BLOCK_POINTS, "block_length_over_H": 9.0 * PRIMARY_BLOCK_POINTS / DENSE_N,
            "seed": BOOTSTRAP_SEED, "confidence_level": 0.95,
            "note": "identical to the WALE Level-3 primary protocol; WALE references resampled in the same draws",
        },
        "cost": cost,
        "producer_jobs": {":".join(key): record["manifest"]["producer_job_id"] for key, record in cases.items()},
        "finalizer_job": campaign.get("finalizer_job_id"),
        "campaign_status": campaign.get("status"),
        "failed_cases": campaign.get("failed_cases", []),
        "dns_tangent_reconstruction_audit": tangent_audit,
        "source_hashes": {
            "codes/analysis/rswm_common_surface_grid_l2.py": sha256(L2_REDUCER),
            # recorded under the path actually resolved: development/nodes/ is rotated by
            # the pipeline, so the nominal node path is not a durable address.  Every copy
            # is byte-identical, so the recorded sha256 is unaffected by which one is used.
            str(L3_ANALYZER.relative_to(ROOT)): sha256(L3_ANALYZER),
            "codes/openfoam/rswm_m4_apply_sgs.py": sha256(ROOT / "codes" / "openfoam" / "rswm_m4_apply_sgs.py"),
            "jobs/rswm_m4_sgs_re_wrapper.sh": sha256(ROOT / "jobs" / "rswm_m4_sgs_re_wrapper.sh"),
            "jobs/rswm_xiao_highre_production_wrapper.sh": sha256(ROOT / "jobs" / "rswm_xiao_highre_production_wrapper.sh"),
            str(CAMPAIGN_ROOT.relative_to(ROOT)) + "/CAMPAIGN_MANIFEST.json": sha256(CAMPAIGN_ROOT / "CAMPAIGN_MANIFEST.json"),
            str(WALE_ROOT.relative_to(ROOT)) + "/CAMPAIGN_MANIFEST.json": sha256(WALE_ROOT / "CAMPAIGN_MANIFEST.json"),
            **{str((WALE_ROOT / cid / "MANIFEST.json").relative_to(ROOT)): sha256(WALE_ROOT / cid / "MANIFEST.json")
               for (g, m), cid in WALE_CASE_IDS.items() if (g, m) in reference_curves},
            str(l2.DNS_FILE.relative_to(ROOT)): sha256(l2.DNS_FILE),
        },
        "case_manifest_sha256": {":".join(key): sha256(record["dir"] / "MANIFEST.json") for key, record in cases.items()},
        "acceptance_gate_sign_note": ("A1 is evaluated on the crest-bulk-corrected runs against the MGLET primary "
                                      "reference. The corrected WALE baseline no longer exceeds the DNS RMS, so the "
                                      "coupled 'wall-traction failure' verdict of 2026-08-22 is withdrawn independently "
                                      "of M4; what M4 certifies is that WHATEVER the corrected verdict is, it is the "
                                      "same under every subgrid-scale closure tested"),
        "conclusion": (
            "The coupled wall-traction verdict is invariant to the subgrid-scale closure: every "
            "alternative-SGS case lands on the same side of the DNS-RMS threshold as the corrected "
            "WALE baseline, by point estimate, interval classification and exact block test, and "
            "stays inside the WALE magnitude class. The WM-SGS interaction does not contaminate the "
            "coupled conclusion."
            if complete and all_pass else
            "INCOMPLETE or NOT INVARIANT - see missing_cases / table[*].criterion."
        ),
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    def arrays_to_lists(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: arrays_to_lists(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [arrays_to_lists(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    encoded = json.dumps(l2.json_ready(arrays_to_lists(summary)), indent=2, sort_keys=True, allow_nan=False) + "\n"
    out_json.write_text(encoded)

    payload: dict[str, Any] = {
        "status": np.array(status),
        "truth_phase": truth_phase, "truth_tau_s": truth_tau, "dense_phase": dense_phase,
        "truth_tau_s_dense": truth_dense,
        "sgs_models": np.asarray(sgs_models), "grids": np.asarray(grids), "models": np.asarray(MODELS),
    }
    for key, by_time in curves.items():
        prefix = "_".join(key)
        terminal = by_time[list(by_time)[-1]]
        for name in ("phase", "x", "tau_s", "tau_x", "ym", "ywall", "wall_ds", "tangent_x", "tangent_y"):
            payload[f"{prefix}_{name}"] = np.asarray(terminal[name])
        payload[f"{prefix}_tau_s_dense"] = predictions[key]
        payload[f"{prefix}_bootstrap_relative_rms"] = samples[key]
        for name, value in base[":".join(key)].items():
            payload[f"{prefix}_{name}"] = np.asarray(value)
        row = table[":".join(key)]
        for name in ("relative_rms", "ratio_to_wale", "p_one_sided_failure_holm", "wale_relative_rms"):
            payload[f"{prefix}_{name}"] = np.asarray(row[name])
        payload[f"{prefix}_interval_95"] = np.asarray([row["relative_rms_interval_95"][k] for k in ("low", "median", "high")])
        payload[f"{prefix}_sgs_minus_wale_interval_95"] = np.asarray([row["sgs_minus_wale_interval_95"][k] for k in ("low", "median", "high")])
        payload[f"{prefix}_verdict_invariant"] = np.asarray(row["verdict_invariant"])
    for name, curve in nut_curves.items():
        payload[f"{name}_first_cell_nut_over_nu"] = np.asarray(curve)
    for (grid, model), curve in reference_curves.items():
        payload[f"{REFERENCE_SGS}_{grid}_{model}_tau_s_dense"] = curve
        payload[f"{REFERENCE_SGS}_{grid}_{model}_bootstrap_relative_rms"] = samples[(REFERENCE_SGS, grid, model)]
    np.savez_compressed(out_npz, **payload)

    make_figure(table, sgs_models, grids, fig_pdf, fig_png)
    print(f"{status} cases={len(cases)} alternative_sgs={alternative} missing={len(missing)} "
          f"all_invariant={all_pass} json={out_json.relative_to(ROOT)}")
    for label, row in sorted(table.items()):
        iv = row["relative_rms_interval_95"]
        print(f"  {label:<40s} E_tau={row['relative_rms']:.3f} [{iv['low']:.3f},{iv['high']:.3f}] "
              f"WALE={row['wale_relative_rms']:.3f} ratio={row['ratio_to_wale']:.3f} "
              f"side={row['verdict_side']}/{row['wale_verdict_side']} "
              f"x_r={row['reattachment_x_over_H']:.3f} invariant={row['verdict_invariant']}")
    return 0 if status == "M4_SGS_SENSITIVITY_OK" else 2


def make_figure(table: dict[str, Any], sgs_models: list[str], grids: list[str],
                fig_pdf: Path, fig_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"equilibrium": "#247a4d", "total_gradient_tble": "#687f95"}
    labels = {"equilibrium": "equilibrium (Spalding)", "total_gradient_tble": "total-gradient TBLE"}
    order = ["WALE"] + [s for s in sgs_models if s != "WALE"]
    fig, axes = plt.subplots(1, len(grids), figsize=(4.0 * len(grids), 3.6), constrained_layout=True, squeeze=False)
    for ax, grid in zip(axes[0], grids):
        for j, model in enumerate(MODELS):
            xs, ys, lo, hi = [], [], [], []
            for i, sgs in enumerate(order):
                if sgs == "WALE":
                    row = next((r for r in table.values() if r["grid"] == grid and r["model"] == model), None)
                    if row is None:
                        continue
                    value, iv = row["wale_relative_rms"], row["wale_relative_rms_interval_95"]
                else:
                    row = table.get(f"{sgs}:{grid}:{model}")
                    if row is None:
                        continue
                    value, iv = row["relative_rms"], row["relative_rms_interval_95"]
                xs.append(i + (j - 0.5) * 0.22)
                ys.append(value)
                lo.append(value - iv["low"])
                hi.append(iv["high"] - value)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=colors[model], label=labels[model], capsize=3)
        ax.axhline(1.0, ls=":", color="#d97706", label="DNS RMS threshold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylim(0, None)
        ax.set_title({"G0": "92,160 cells", "G1c": "307,200 cells", "G2c": "819,200 cells"}.get(grid, grid))
        ax.set_ylabel(r"$E_\tau$")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_pdf)
    fig.savefig(fig_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
