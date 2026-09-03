#!/usr/bin/env python3
"""R2-3/M6 real resolution: coupled matching-height sweep on the Xiao hill.

Consumes ``codes/results/rswm_r23m6_ym_campaign_final/ym{0300,0600,0935,1500,2500}``
(Re_H = 5600, corrected crest-bulk drive, G1c everywhere, G2c at the two
extremes, equilibrium + total-gradient TBLE) plus the deposited corrected
y_m/H = 0.0145 baseline (M13 ``re5600`` bundles), and writes

    codes/results/r2_3_ym_window_<date>.{npz,_summary.json}

with, per matching height and model:
* the coupled physical-tangent wall-traction error against the Xiao DNS
  (deposited L2 metric; paired circular phase-block bootstrap, Lx/8 blocks),
  reattachment/separation, reversed fraction, signed force;
* the measured matching height (per-face median, flat-floor value, y_m+ from
  the DNS friction velocity);
* the coupled cancellation parameter eps_c at the actual matching height;
* mean-velocity profile RMS against the Xiao DNS stations;
* window/drive stationarity checks;
and, across heights, the A-PRIORI -> A-POSTERIORI TRANSFER RELATION: the
deposited a-priori sweeps (critical_matching_height_map.npz on the y+ axis
1..300 -- the reviewer's "R^2 < 0 at every y_m+ in [1,300]" object -- and
epsilon_coupled_predictor.npz on the y/H axis) interpolated at the coupled
heights, paired with the coupled errors, with rank statistics and the
data-driven window verdict.

Reuses the deposited reducers by import; nothing pinned is modified.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import re

import harvest_m13_highre as HM

ROOT = HM.ROOT
CAMPAIGN = Path(os.environ.get("R23M6_CAMPAIGN_ROOT", ROOT / "codes" / "results" / "rswm_r23m6_ym_campaign_final"))
# Drive-stationarity continuations under the prospective rule
# development/nodes/node_009/CONTINUATION_RULE_YM_SWEEP.md: case -> (bundle
# subdir, averaging-window origin t0).  The registered [135,405] window failed
# the drive-halves gate for this one case (0.05505 > 0.05); the continuation
# re-evaluates every gate on the equally long later window [405,675].
CONTINUATIONS = {
    "0300:G1c:equilibrium": {
        "bundle": "ym0300_cont",
        "case_id": "rswm_r23m6_ym0300_g1_equilibrium_307200_v1",
        "drive_t0": 405.0,
        "rule": "development/nodes/node_009/CONTINUATION_RULE_YM_SWEEP.md",
    },
}
# Registered model aborts (AMENDMENT_YM2500_G2C_TBLE_ABORT.md): the pinned
# TBLE kernel refused a genuine three-distinct-root state at the largest
# matching height on the fine grid.  The point is recorded as a structured
# abort — the model's measured operability boundary — never fabricated and
# never silently omitted.
MODEL_ABORTS = {
    "2500:G2c:total_gradient_tble": {
        "case_id": "rswm_r23m6_ym2500_g2_tble_819200_v1",
        "producer_job_id": "14904100",
        "slurm_log": "logs/archer2/slurm-14904100.out",
        "solver_log": "codes/results/registered_aborts/ym2500_g2c_tble_abort_log.pimpleFoam",
        "amendment": "codes/results/registered_aborts/AMENDMENT_YM2500_G2C_TBLE_ABORT.md",
        "kernel_sha256": "69edf4b532ca612fb8a735b338dc24d69de880dd97908fa8c3468848a5f1ff56",
    },
}


def model_abort_record(spec: dict[str, Any]) -> dict[str, Any]:
    slurm_path = ROOT / spec["slurm_log"]
    solver_path = ROOT / spec["solver_log"]
    slurm_text = slurm_path.read_text(errors="replace")
    solver_text = solver_path.read_text(errors="replace")
    abort_lines = [ln.strip() for ln in solver_text.splitlines() if "TBLE branch failure" in ln]
    if not abort_lines:
        raise SystemExit(f"{spec['case_id']}: registered abort line absent from the solver log")
    # The driver redirects srun output into log.pimpleFoam, so the scheduler
    # log carries job identity (driver marker), not the abort text.
    if "R23M6_YM_DRIVER_OK ym=0.25 model=total_gradient_tble" not in slurm_text:
        raise SystemExit(f"{spec['case_id']}: scheduler log lacks the registered driver marker")
    times = [float(m) for m in re.findall(r"^Time = ([0-9.eE+-]+)s?$", solver_text, re.MULTILINE)]
    if not times:
        raise SystemExit(f"{spec['case_id']}: no solver time history in evidence log")
    realizability = [ln.strip() for ln in solver_text.splitlines()
                     if "TBLE_REALIZABILITY patch=bottomWall" in ln]
    m = re.search(r"roots=(\d+) branchLoss=(\d+) ambiguous=(\d+) truncated=(\d+)", abort_lines[0])
    if m is None:
        raise SystemExit(f"{spec['case_id']}: abort line lacks the root census fields")
    amendment = ROOT / spec["amendment"]
    if not amendment.is_file():
        raise SystemExit(f"{spec['case_id']}: amendment file missing")
    return {
        "model_abort": {
            "case_id": spec["case_id"],
            "producer_job_id": spec["producer_job_id"],
            "abort_line": abort_lines[0].lstrip("[0-9] "),
            "roots": int(m.group(1)),
            "ambiguous": int(m.group(3)),
            "fold_degeneracy_guard_events": int(re.search(r"degenerateRoots=(\d+)", realizability[-1]).group(1)) if realizability else None,
            "last_solver_time": max(times),
            "registered_end_time": 405.0,
            "final_bottomwall_realizability": (realizability[-1] if realizability else None),
            "slurm_log": spec["slurm_log"],
            "slurm_log_sha256": HM.sha256(slurm_path),
            "solver_log": spec["solver_log"],
            "solver_log_sha256": HM.sha256(solver_path),
            "kernel_sha256": spec["kernel_sha256"],
            "amendment": spec["amendment"],
        }
    }
BASELINE = Path(os.environ.get("M13_CAMPAIGN_ROOT", ROOT / "codes" / "results" / "rswm_xiao_highre_campaign_m13_final")) / "re5600"
APRIORI_MAP = ROOT / "codes" / "results" / "critical_matching_height_map.npz"
APRIORI_PRED = ROOT / "codes" / "results" / "epsilon_coupled_predictor.npz"
YM_TAGS = {"0300": 0.03, "0600": 0.06, "0935": 0.0935, "1500": 0.15, "2500": 0.25}
G2_TAGS = ("0300", "2500")
MODELS = ("equilibrium", "total_gradient_tble")
LX = 9.0
DENSE_N = 4096
BLOCK = 512
SEED = 20260824
YCRIT_PAPER = 15.925931  # critical_matching_height_map ycrit[krank_pehill_Re10595]


def interval(v):
    return HM.interval(np.asarray(v))


def drive_stationarity_shifted(log: Path, target_ubar: float, t0: float) -> dict[str, Any]:
    """HM.drive_stationarity with the 270-unit window origin at t0 (135 = deposit).

    Identical telemetry parse, identical functional gates; the halves are
    [t0, t0+135) vs [t0+135, t0+270] and the pre-window transient is
    [t0-45, t0).  t0=135 reproduces HM.drive_stationarity bit-for-bit.
    """
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
    first = (t >= t0) & (t < t0 + 135.0)
    second = (t >= t0 + 135.0) & (t <= t0 + 270.0)
    window = (t >= t0) & (t <= t0 + 270.0)
    transient = (t >= t0 - 45.0) & (t < t0)
    return {
        "samples": int(len(t)),
        "window_origin": float(t0),
        "window_mean_gradient": float(np.mean(g[window])),
        "first_half_mean_gradient": float(np.mean(g[first])),
        "second_half_mean_gradient": float(np.mean(g[second])),
        "halves_relative_difference": float(abs(np.mean(g[second]) - np.mean(g[first])) / abs(np.mean(g[window]))),
        "pre_window_mean_gradient": float(np.mean(g[transient])) if np.any(transient) else math.nan,
        "window_mean_uncorrected_Ubar": float(np.mean(u[window])),
        "registered_volume_average_Ubar": float(target_ubar),
        "window_Ubar_max_abs_deviation": float(np.max(np.abs(u[window] - target_ubar))),
        "telemetry_samples_in_window": int(np.count_nonzero(window)),
    }


def point_record(l2, l3, case: Path, truth_phase, truth_tau, dense, draws: int, seed: int, drive_t0: float = 135.0) -> dict[str, Any]:
    manifest, mesh, curves, pressures = HM.load_case(l2, case)
    drive = HM.registered_drive(case, manifest)
    names = list(curves)
    final = curves[names[-1]]
    m = l2.metrics(final, truth_phase, truth_tau)
    pred = l2.periodic_interp(np.asarray(final["phase"]), np.asarray(final["tau_s"]), dense)
    truth_dense = l2.periodic_interp(truth_phase, truth_tau, dense)
    samples = l3.circular_block_bootstrap(truth_dense, {("c", "m"): pred}, block_points=BLOCK, draws=draws, seed=seed)
    ym_faces = np.asarray(final["ym"])
    u_tau_dns = np.sqrt(np.abs(l2.periodic_interp(truth_phase, truth_tau, np.asarray(final["phase"]))))
    nu = float(manifest["nu"])
    rec_eps = None
    try:
        te_sep, te_rea = l2.zero_crossings(dense, truth_dense)
        # eps at the ACTUAL matching height from the run's own wall fields
        x = np.asarray(final["x"]); tau = np.asarray(final["tau_s"])
        dpds = HM.periodic_derivative_arclength(x, np.asarray(final["ywall"]), pressures[names[-1]])
        phi = np.abs(dpds) * ym_faces
        eps = np.abs(tau) / np.maximum(phi, 1.0e-14)
        phase = np.asarray(final["phase"])
        separated = ((phase - te_sep) % 1.0) < ((te_rea - te_sep) % 1.0)
        rng = np.random.default_rng(seed + 1)
        n = len(phase); blk = max(4, n // 8)
        med = []
        for _ in range(min(draws, 4000)):
            starts = rng.integers(0, n, size=n // blk + 1)
            idx = ((starts[:, None] + np.arange(blk)[None, :]) % n).ravel()[:n]
            sel = eps[idx][separated[idx]]
            med.append(np.median(sel if len(sel) else eps[idx]))
        rec_eps = {"eps_c_median_separated": float(np.median(eps[separated])),
                   "eps_c_median_separated_interval": interval(np.asarray(med))}
    except Exception as exc:  # noqa: BLE001
        rec_eps = {"error": str(exc)}
    checkpoint = names[-1]
    profiles = HM.read_profiles(case, checkpoint)
    bulk = HM.crest_bulk_velocity(profiles)
    # AMENDMENT_CREST_BULK_SLICE_MEASUREMENT.md: the spanwise-mean flux through
    # every cross-section is pinned to Ubar*V/(Lx*Lz) = 2.036 by the exactly
    # held volume constraint; the z=2.25 slice fluxes below measure the
    # window-persistent spanwise inhomogeneity of the outer mean flow.
    slice_flux = {xkey: float(np.trapz(d[:, 1], d[:, 0])) for xkey, d in profiles.items()}
    bulk["slice_flux_stations"] = slice_flux
    bulk["slice_flux_expected_spanwise_mean"] = 2.036
    bulk["slice_flux_max_abs_deviation_fraction"] = float(
        max(abs(q / 2.036 - 1.0) for q in slice_flux.values()))
    refs = {"xiao_5600": HM.xiao_dns_reference(l2)}
    prof = HM.profile_validation(profiles, refs["xiao_5600"])
    last = np.asarray(final["tau_s"]); prev = np.asarray(curves[names[-2]]["tau_s"]); earl = np.asarray(curves[names[-3]]["tau_s"])
    norm = max(float(np.sqrt(np.mean(last ** 2))), 1e-14)
    return {
        "case_id": manifest["case_id"] if "case_id" in manifest else case.name,
        "producer_job_id": manifest["producer_job_id"],
        "ym_target_over_H": float(manifest.get("ym_target_over_H", json.loads((case / "matching_surface.json").read_text()).get("ym_target_over_H", math.nan))),
        "ym_measured_median_over_H": float(np.median(ym_faces)),
        "ym_measured_flat_over_H": float(json.loads((case / "matching_surface.json").read_text()).get("measured_flat_ym_over_H", np.median(ym_faces))),
        "ym_plus_dns": l2.quantiles(ym_faces * u_tau_dns / nu),
        "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float))},
        "relative_rms_interval": interval(samples[("c", "m")]),
        "eps_c": rec_eps,
        "profile_u_rms_mean_vs_xiao": prof["u_rms_mean"],
        "profile_u_rms_max_vs_xiao": prof["u_rms_max"],
        "crest_bulk": bulk,
        "drive_registration": {k: v for k, v in drive.items() if not isinstance(v, dict)},
        "maximum_courant": manifest["maximum_courant"],
        "change_225_to_270": float(np.sqrt(np.mean((last - prev) ** 2)) / norm),
        "change_180_to_225": float(np.sqrt(np.mean((prev - earl) ** 2)) / norm),
        "drive_stationarity": drive_stationarity_shifted(case / "log.pimpleFoam", float(drive["volume_average_Ubar"]), drive_t0),
        "solver_cost": manifest["solver_cost"],
        "branch_policy": HM.branch_policy(case),
        "_curve": {k: np.asarray(final[k]) for k in ("phase", "x", "tau_s", "ym", "ywall", "wall_ds")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=_dt.date.today().isoformat().replace("-", ""))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--tags", default=",".join(YM_TAGS))
    args = parser.parse_args()
    tags = [t for t in args.tags.split(",") if t]

    l2 = HM.load_module(HM.L2_REDUCER, "rswm_l2_locked")
    l3 = HM.load_module(HM.L3_ANALYSER, "rswm_l3_locked")
    dns = np.load(HM.DNS_5600)
    truth_tau, audit = l2.dns_tangent_reference(dns)
    truth_phase = np.mod((np.asarray(dns["x"]) - float(np.min(dns["x"]))) / LX, 1.0)
    dense = np.arange(DENSE_N, dtype=float) / DENSE_N

    # Status is derived AFTER harvesting: R23M6_YM_WINDOW_OK only when the FULL
    # registered matrix is present (all six heights, G2c at the extremes and
    # baseline, both models) and no registered continuation is pending.  A
    # partial harvest is labelled PARTIAL so it can never masquerade as
    # terminal evidence.
    out: dict[str, Any] = {"status": "R23M6_YM_WINDOW_PARTIAL", "points": {}, "Re_H": 5600}
    payload: dict[str, Any] = {"dense_phase": dense, "truth_phase": truth_phase, "truth_tau_s": truth_tau}

    # baseline y_m/H = 0.0145 from the corrected M13 matrix
    baseline_cases = {("G1c", "equilibrium"): "rswm_m13_re5600_g1_equilibrium_307200_v2",
                      ("G1c", "total_gradient_tble"): "rswm_m13_re5600_g1_tble_307200_v2",
                      ("G2c", "equilibrium"): "rswm_m13_re5600_g2_equilibrium_819200_v2",
                      ("G2c", "total_gradient_tble"): "rswm_m13_re5600_g2_tble_819200_v2"}
    for (grid, model), cid in baseline_cases.items():
        rec = point_record(l2, l3, BASELINE / cid, truth_phase, truth_tau, dense, args.draws, SEED)
        rec["ym_target_over_H"] = 0.014515746
        curve = rec.pop("_curve")
        for k, v in curve.items():
            payload[f"ym0145_{grid}_{model}_{k}"] = v
        out["points"][f"0145:{grid}:{model}"] = HM.json_ready(rec)

    def resolve_rung(tag: str) -> Path:
        """Most complete finalized bundle for a rung.

        The kernel-v4 rerun of the y_m/H=0.25 G2c TBLE case is packaged beside the
        original as ``ym<tag>_v4``; prefer whichever manifest carries most cases.
        """
        best, best_n = None, -1
        for name in (f"ym{tag}_v4", f"ym{tag}"):
            manifest = CAMPAIGN / name / "CAMPAIGN_MANIFEST.json"
            if manifest.is_file():
                n = len(json.loads(manifest.read_text()).get("cases", {}))
                if n > best_n:
                    best, best_n = CAMPAIGN / name, n
        if best is None:
            raise SystemExit(f"ym{tag}: no finalized bundle under {CAMPAIGN}")
        return best

    for tag in tags:
        root = resolve_rung(tag)
        cm = json.loads((root / "CAMPAIGN_MANIFEST.json").read_text())
        if cm.get("status") != "TERMINAL_SIX_CASE_CAMPAIGN_OK":
            raise SystemExit(f"ym{tag}: campaign not terminal")
        grids = ("G1c", "G2c") if tag in G2_TAGS else ("G1c",)
        for grid in grids:
            gtag = "g1" if grid == "G1c" else "g2"
            cells = 307200 if grid == "G1c" else 819200
            for model in MODELS:
                mtag = "tble" if model == "total_gradient_tble" else "equilibrium"
                cid = f"rswm_r23m6_ym{tag}_{gtag}_{mtag}_{cells}_v1"
                for suffix in ("v3", "v2", "v1"):
                    candidate = f"rswm_r23m6_ym{tag}_{gtag}_{mtag}_{cells}_{suffix}"
                    if (root / candidate).is_dir():
                        cid = candidate
                        break
                key = f"{tag}:{grid}:{model}"
                # A registered abort stands only until the same case is rerun with a
                # kernel whose tie-break resolves it.  The v3 case id is the kernel-v4
                # rerun (job 14909038) of the 14904100 abort: the tied roots there differ
                # by ~1e-11 (numerical twins at tau_w=0 on a separation face), so the
                # abort was an implementation artifact, not a model admissibility limit.
                rerun_cid = f"rswm_r23m6_ym{tag}_{gtag}_{mtag}_{cells}_v3"
                rerun_available = (root / rerun_cid).is_dir()
                if key in MODEL_ABORTS and not rerun_available:
                    out["points"][key] = HM.json_ready(model_abort_record(MODEL_ABORTS[key]))
                    continue
                rec = point_record(l2, l3, root / cid, truth_phase, truth_tau, dense, args.draws, SEED + int(tag) + cells)
                if key in MODEL_ABORTS and rerun_available:
                    rec["supersedes_registered_abort"] = {
                        "aborted_case_id": MODEL_ABORTS[key]["case_id"],
                        "aborted_producer_job_id": MODEL_ABORTS[key]["producer_job_id"],
                        "aborted_kernel_sha256": MODEL_ABORTS[key]["kernel_sha256"],
                        "rerun_case_id": cid,
                        "reason": ("the aborting tie was a numerical twin-root pair (~1e-11 apart) at "
                                   "tau_w=0; kernel v4 scales the degeneracy tolerance by "
                                   "census.scanHalfWidth and resolves it"),
                        "abort_evidence_retained": MODEL_ABORTS[key]["amendment"],
                    }
                if abs(rec["ym_measured_flat_over_H"] - YM_TAGS[tag]) > 0.01 * YM_TAGS[tag]:
                    raise SystemExit(f"{cid}: measured ym {rec['ym_measured_flat_over_H']} != target {YM_TAGS[tag]}")
                cont = CONTINUATIONS.get(key)
                if cont is not None:
                    cont_root = CAMPAIGN / cont["bundle"]
                    cont_manifest = cont_root / "CAMPAIGN_MANIFEST.json"
                    if cont_manifest.is_file() and json.loads(cont_manifest.read_text()).get("status") == "TERMINAL_SIX_CASE_CAMPAIGN_OK":
                        original = rec
                        rec = point_record(l2, l3, cont_root / cont["case_id"], truth_phase,
                                           truth_tau, dense, args.draws, SEED + int(tag) + cells,
                                           drive_t0=cont["drive_t0"])
                        if abs(rec["ym_measured_flat_over_H"] - YM_TAGS[tag]) > 0.01 * YM_TAGS[tag]:
                            raise SystemExit(f"{cont['case_id']} (continuation): measured ym off target")
                        orig_curve = original.pop("_curve")
                        for k, v in orig_curve.items():
                            payload[f"ym{tag}_{grid}_{model}_orig135405_{k}"] = v
                        rec["continuation"] = {
                            "rule": cont["rule"],
                            "window": [cont["drive_t0"], cont["drive_t0"] + 270.0],
                            "original_window": [135.0, 405.0],
                            "original_window_record": {k: v for k, v in original.items()},
                            "relative_rms_change_vs_original": rec["metrics"]["relative_rms"] - original["metrics"]["relative_rms"],
                        }
                    else:
                        rec["continuation_pending"] = {
                            "rule": cont["rule"],
                            "note": ("this point failed the registered drive-halves gate on [135,405] and its "
                                     "registered continuation bundle is not terminal yet; the sweep is NOT "
                                     "complete until the continuation lands"),
                        }
                curve = rec.pop("_curve")
                for k, v in curve.items():
                    payload[f"ym{tag}_{grid}_{model}_{k}"] = v
                out["points"][key] = HM.json_ready(rec)

    # ---- a-priori sweeps and the transfer relation --------------------------
    ap_map = np.load(APRIORI_MAP, allow_pickle=True)
    ap_pred = np.load(APRIORI_PRED, allow_pickle=True)
    keys = [str(k) for k in ap_map["keys"]]
    ymp = np.asarray(ap_map["ymp_grid"], float)
    ap_yplus_relrms = {k: np.asarray(ap_map[f"sweep_relrms__{k}"], float)
                       for k in ("periodic_hills_1p0", "krank_pehill_Re10595") if f"sweep_relrms__{k}" in ap_map.files}
    sweep_ym = np.asarray(ap_pred["sweep_ym"], float)
    sweep_relrms = np.asarray(ap_pred["sweep_relrms"], float)
    sweep_r2 = np.asarray(ap_pred["sweep_R2"], float)
    sweep_eps = np.asarray(ap_pred["sweep_eps"], float)

    transfer = []
    all_tags = ["0145"] + list(tags)
    ym_values = {"0145": 0.014515746, **YM_TAGS}
    for tag in all_tags:
        for model in MODELS:
            key = f"{tag}:G1c:{model}"
            if key not in out["points"]:
                continue
            p = out["points"][key]
            ym_h = ym_values[tag]
            ymp_med = p["ym_plus_dns"]["median"]
            row = {
                "ym_tag": tag, "ym_over_H": ym_h, "model": model,
                "ym_plus_dns_median": ymp_med,
                "coupled_relative_rms": p["metrics"]["relative_rms"],
                "coupled_relative_rms_interval": p["relative_rms_interval"],
                "coupled_r2": p["metrics"]["r2"],
                "coupled_reattachment_x_over_H": p["metrics"]["reattachment_x_over_H"],
                "coupled_reattachment_bias_over_H": p["metrics"]["reattachment_x_over_H"] - p["metrics"]["truth_reattachment_x_over_H"],
                "coupled_eps_c_median_separated": (p["eps_c"] or {}).get("eps_c_median_separated"),
                "apriori_relrms_yH_axis": float(np.interp(ym_h, sweep_ym, sweep_relrms)) if sweep_ym.min() <= ym_h <= sweep_ym.max() else None,
                "apriori_r2_yH_axis": float(np.interp(ym_h, sweep_ym, sweep_r2)) if sweep_ym.min() <= ym_h <= sweep_ym.max() else None,
                "apriori_eps_yH_axis": float(np.interp(ym_h, sweep_ym, sweep_eps)) if sweep_ym.min() <= ym_h <= sweep_ym.max() else None,
                "apriori_relrms_yplus_axis": {k: float(np.interp(ymp_med, ymp, v)) for k, v in ap_yplus_relrms.items()},
            }
            transfer.append(row)
    out["transfer_relation"] = transfer

    def spear(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(ok) < 3:
            return math.nan
        ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
        return float(np.corrcoef(ra, rb)[0, 1])

    stats: dict[str, Any] = {}
    for model in MODELS:
        rows = [r for r in transfer if r["model"] == model]
        rows.sort(key=lambda r: r["ym_over_H"])
        ymh = [r["ym_over_H"] for r in rows]
        cr = [r["coupled_relative_rms"] for r in rows]
        ap = [r["apriori_relrms_yH_axis"] for r in rows]
        rb = [abs(r["coupled_reattachment_bias_over_H"]) for r in rows]
        lo = [r["coupled_relative_rms_interval"]["low"] for r in rows]
        hi = [r["coupled_relative_rms_interval"]["high"] for r in rows]
        i_min = int(np.argmin(cr))
        beyond = [i for i, r in enumerate(rows) if r["ym_plus_dns_median"] > YCRIT_PAPER]
        inside = [i for i, r in enumerate(rows) if r["ym_plus_dns_median"] <= YCRIT_PAPER]
        stats[model] = {
            "ym_over_H": ymh, "coupled_relative_rms": cr,
            "interval_low": lo, "interval_high": hi,
            "apriori_relrms": ap,
            "abs_reattachment_bias_over_H": rb,
            "spearman_coupled_vs_apriori_relrms": spear(cr, ap),
            "spearman_coupled_vs_ym": spear(cr, ymh),
            "spearman_reatt_bias_vs_ym": spear(rb, ymh),
            "argmin_ym_over_H": ymh[i_min],
            "argmin_ym_plus": rows[i_min]["ym_plus_dns_median"],
            "min_relative_rms": cr[i_min],
            "worst_beyond_window": (max(cr[i] for i in beyond) if beyond else None),
            "best_inside_window": (min(cr[i] for i in inside) if inside else None),
            "beyond_window_worse_than_inside_best": (bool(max(cr[i] for i in beyond) > min(cr[i] for i in inside)) if beyond and inside else None),
            "all_relative_rms_above_1": bool(all(v > 1.0 for v in cr)),
            "all_interval_low_above_1": bool(all(v > 1.0 for v in lo)),
        }
    out["window_verdict"] = {
        "paper_ycrit_plus": YCRIT_PAPER,
        "per_model": stats,
        "note": ("coupled traction error vs matching height, with the a-priori sweep at the same heights; "
                 "the reviewer's object (a-priori R^2<0 at every y_m+ in [1,300]) is the "
                 "critical_matching_height_map y+ sweep carried in transfer_relation.apriori_relrms_yplus_axis"),
    }
    # G1c->G2c invariance at the extremes
    grid_check = {}
    for tag in [t for t in tags if t in G2_TAGS]:
        for model in MODELS:
            a = out["points"].get(f"{tag}:G1c:{model}"); b = out["points"].get(f"{tag}:G2c:{model}")
            if a and b and "model_abort" in b:
                grid_check[f"{tag}:{model}"] = {
                    "G1c_relative_rms": (a["metrics"]["relative_rms"] if "metrics" in a else None),
                    "G2c_model_abort": b["model_abort"]["abort_line"],
                    "G2c_abort_producer_job_id": b["model_abort"]["producer_job_id"],
                    "verdict_invariant_above_1": None,
                    "note": "no G2c verdict is fabricated for the aborted case; "
                            "see the registered amendment",
                }
            elif a and b:
                grid_check[f"{tag}:{model}"] = {
                    "G1c_relative_rms": a["metrics"]["relative_rms"], "G2c_relative_rms": b["metrics"]["relative_rms"],
                    "change": b["metrics"]["relative_rms"] - a["metrics"]["relative_rms"],
                    "verdict_invariant_above_1": bool(a["metrics"]["relative_rms"] > 1.0 and b["metrics"]["relative_rms"] > 1.0),
                }
    out["grid_invariance_extremes"] = grid_check
    out["provenance"] = {
        "campaign_root": str(CAMPAIGN), "baseline_root": str(BASELINE),
        "truth": "Xiao et al. (2020) alpha=1 DNS tangent reconstruction (deposited)",
        "dns_sha256": HM.sha256(HM.DNS_5600),
        "apriori_map_sha256": HM.sha256(APRIORI_MAP),
        "apriori_predictor_sha256": HM.sha256(APRIORI_PRED),
        "l2_reducer_sha256": HM.sha256(HM.L2_REDUCER),
        "l3_analyser_sha256": HM.sha256(HM.L3_ANALYSER),
        "wrapper_sha256": HM.sha256(ROOT / "jobs" / "rswm_r23m6_ym_wrapper.sh"),
        "grading_sha256": HM.sha256(ROOT / "codes" / "openfoam" / "rswm_ym_grading.py"),
        "mesh_verifier_sha256": HM.sha256(ROOT / "codes" / "openfoam" / "verify_r23m6_ym_mesh.py"),
        "bootstrap": {"draws": args.draws, "block_points": BLOCK, "dense": DENSE_N, "seed": SEED},
        "dns_tangent_audit": HM.json_ready(audit),
    }
    out["date"] = args.date
    required = [f"0145:{g}:{m}" for g in ("G1c", "G2c") for m in MODELS]
    required += [f"{t}:G1c:{m}" for t in YM_TAGS for m in MODELS]
    required += [f"{t}:G2c:{m}" for t in G2_TAGS for m in MODELS]
    missing = [k for k in required if k not in out["points"]]
    pending = [k for k, p in out["points"].items() if "continuation_pending" in p]
    aborted = [k for k, p in out["points"].items() if "model_abort" in p]
    if "2500" in tags and sorted(aborted) != sorted(MODEL_ABORTS):
        raise SystemExit(f"abort registry mismatch: recorded {aborted}, registered {sorted(MODEL_ABORTS)}")
    if not missing and not pending:
        out["status"] = "R23M6_YM_WINDOW_OK"
    out["completeness"] = {"required_points": required, "missing_points": missing,
                           "continuation_pending_points": pending,
                           "model_abort_points": aborted}
    stem = f"r2_3_ym_window_{args.date}"
    out_dir = Path(os.environ.get("R23M6_OUTPUT_DIR", ROOT / "codes" / "results"))
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(HM.json_ready(out), indent=2, sort_keys=True, allow_nan=False) + "\n")
    payload["status"] = np.array(out["status"])
    np.savez_compressed(out_dir / f"{stem}.npz", **payload)
    print(f"{out['status']} tags={['0145'] + tags} missing={missing} continuation_pending={pending} "
          f"summary={out_dir / (stem + '_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
