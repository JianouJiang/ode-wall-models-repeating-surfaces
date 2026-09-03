#!/usr/bin/env python3
"""Harvest the R2-m4 / R3-2 coupled model ladder (ARCHER2 r2m4_ladder_campaign).

Reads codes/results/r2m4_ladder_campaign/<case_id>/ (synced with
`archer2_run.sh down`), reduces the time-averaged bottom-wall traction to the
physical tangent exactly as the deposited three-grid analysis does
(tau_s = -wallShearStressMean . t_s, span average per phase, mesh tangent from
input/polyMesh when present), scores every case against the tangent-frame DNS
truth with the deposited metric set and phase-block bootstrap, pairs the
coupled ladder with the a-priori ladder evaluated on the same surface
(r2m4_apriori_ladder_<stamp>.json) and writes

  codes/results/r2m4_ladder_coupled_<stamp>.{json,npz}
  codes/figures/fig_r2m4_ladder.{pdf,png}

Cases that have not landed are reported as pending; the status is
R2M4_LADDER_HARVEST_OK only when the minimum set (five L1 models and the two
W1 ceiling cases, the latter taken from Agent B's corrected Re=5,600 G1c
matrix after the operator's rebase order) is complete,
R2M4_LADDER_HARVEST_PARTIAL otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402
import r2m4_truth_references as T  # noqa: E402
import rswm_common_surface_grid_l2 as L2  # noqa: E402  (deposit reduction protocol)

# 2026-08-25: the deposited Xiao 4-point wall-gradient reconstruction was withdrawn as a SCORING
# reference (work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/REPORT.md).  The case
# reduction below (raw tau_s curves, provenance, physical checks) is reference-independent and
# unchanged; the SCORES are now produced against every reference in r2m4_truth_references.py, with
# the MGLET DNS deposit primary and the repaired-Xiao cubic as the bracket.  The 20260823 artifact
# is left on disk verbatim as superseded.
STAMP = "20260825"
SUPERSEDED_STAMP = "20260823"
CAMPAIGN = ROOT / "codes/results/r2m4_ladder_campaign"
APRIORI = ROOT / "codes/results" / f"r2m4_apriori_ladder_{SUPERSEDED_STAMP}.json"
RESCORED = ROOT / "codes/results" / f"r2m4_ladder_rescored_{STAMP}.json"
OUT_JSON = ROOT / "codes/results" / f"r2m4_ladder_coupled_{STAMP}.json"
OUT_NPZ = ROOT / "codes/results" / f"r2m4_ladder_coupled_{STAMP}.npz"
FIG = ROOT / "codes/figures/fig_r2m4_ladder"
CASES = {
    # v1 = kernel v1 (four rungs aborted on the duplicated tau_w~0 branch, see MANIFEST 6b/6c);
    # v2 = kernel v2 resubmissions.  Tuples are searched in order; a completed bundle wins.
    ("L1", "equilibrium"): "r2m4_L1_equilibrium_153600_v1",
    ("L1", "totalGradient"): ("r2m4_ladder_campaign/r2m4_L1_totalGradient_153600_v2",
                              "r2m4_ladder_campaign/r2m4_L1_totalGradient_153600_v1"),
    ("L1", "hickel"): ("r2m4_ladder_campaign/r2m4_L1_hickel_153600_v2",
                       "r2m4_ladder_campaign/r2m4_L1_hickel_153600_v1"),
    ("L1", "resolvedConvectionLinear"): ("r2m4_ladder_campaign/r2m4_L1_resolvedConvectionLinear_153600_v1",
                                         "r2m4_ladder_campaign/r2m4_L1_resolvedConvectionLinear_153600_v2"),
    ("L1", "resolvedConvectionConstant"): "r2m4_L1_resolvedConvectionConstant_153600_v2",   # v1 (14888773) aborted at t=23.9; kernel v2
    # W1 = wall-resolved ceiling (deposited G1c grid, y_m^+ ~ 2, corrected mass flow).  Supplied by
    # Agent B's corrected Re=5,600 matrix (jobs 14889022 / 14889021, operator rebase 2026-08-23);
    # the Agent-C duplicates were cancelled.  Candidates are searched in order.
    ("W1", "equilibrium"): ("rswm_xiao_highre_campaign_m13_final/re5600/rswm_m13_re5600_g1_equilibrium_307200_v2",
                            "rswm_xiao_highre_campaign_m13/rswm_m13_re5600_g1_equilibrium_307200_v2",
                            "r2m4_ladder_campaign/r2m4_W1_equilibrium_307200_v1"),
    ("W1", "totalGradient"): ("rswm_xiao_highre_campaign_m13_final/re5600/rswm_m13_re5600_g1_tble_307200_v2",
                              "rswm_xiao_highre_campaign_m13/rswm_m13_re5600_g1_tble_307200_v2",
                              "r2m4_ladder_campaign/r2m4_W1_totalGradient_307200_v1"),
    ("L2", "equilibrium"): "r2m4_L2_equilibrium_327680_v1",
    ("L2", "totalGradient"): ("r2m4_ladder_campaign/r2m4_L2_totalGradient_327680_v1",
                              "r2m4_ladder_campaign/r2m4_L2_totalGradient_327680_v2"),
    ("L2", "hickel"): ("r2m4_ladder_campaign/r2m4_L2_hickel_327680_v2",
                       "r2m4_ladder_campaign/r2m4_L2_hickel_327680_v1"),
    ("L2", "resolvedConvectionLinear"): "r2m4_L2_resolvedConvectionLinear_327680_v2",   # kernel v2 (pre-emptive switch)
}
MINIMUM = [k for k in CASES if k[0] in ("L1", "W1")]
SURFACE_OF = {"L1": "ladder_L1", "L2": "ladder_L1", "W1": "common_W1"}
EXPECTED_UBAR = 0.721044918040774


def analytic_mesh(rows: np.ndarray) -> dict[str, np.ndarray]:
    """Fallback when input/polyMesh is absent: analytic tangent, uniform area."""
    x = rows[:, 0]
    _, slope, tx, ty = C.wall_tangent(np.sort(np.unique(np.round(x, 9))))
    xu = np.sort(np.unique(np.round(x, 9)))
    idx = np.searchsorted(xu, np.round(x, 9))
    tangent = np.stack([tx[idx], ty[idx], np.zeros_like(x)], axis=1)
    dxs = float(np.median(np.diff(xu)))
    zu = np.unique(np.round(rows[:, 2], 9))
    dz = float(np.median(np.diff(zu)))
    area = dxs * np.sqrt(1.0 + slope[idx] ** 2) * dz
    return {"xyz": rows[:, :3].copy(), "tangent": tangent, "ym": np.full(len(x), np.nan), "area": area}


def log_telemetry(path: Path) -> dict:
    text = path.read_text(errors="replace")
    ubar = [float(v) for v in re.findall(r"uncorrected Ubar = ([0-9.eE+-]+)", text)]
    co = [float(v) for v in re.findall(r"Courant Number mean:\s*[0-9.eE+-]+\s+max:\s*([0-9.eE+-]+)", text)]
    steps = len(re.findall(r"^Time = ", text, flags=re.M))
    real = re.findall(r"LADDER_REALIZABILITY model=\S+ patch=bottomWall time=([0-9.eE+-]+) faces=(\d+) clipped=(\d+)"
                      r".*?convectionImpulseL1=([0-9.eE+-]+) pressureImpulseL1=([0-9.eE+-]+)", text)
    real = np.asarray(real, float) if real else np.zeros((0, 5))
    window = real[real[:, 0] >= 135.0] if len(real) else real
    return {
        "time_steps": steps,
        "ubar_final": ubar[-1] if ubar else None,
        "ubar_window_mean": float(np.mean(ubar[-max(1, len(ubar) // 3):])) if ubar else None,
        "max_courant": max(co) if co else None,
        "bottom_clipped_fraction_window": float(np.mean(window[:, 2] / window[:, 1])) if len(window) else None,
        "bottom_convection_over_pressure_impulse_window": (
            float(np.mean(window[:, 3] / np.maximum(window[:, 4], 1e-30))) if len(window) else None),
    }


def resolve_case(case_id):
    """Return (directory, external) for a ladder case id or a tuple of candidates."""
    if isinstance(case_id, tuple):
        for rel in case_id:
            d = ROOT / "codes/results" / rel
            if (d / "MANIFEST.json").is_file():
                return d, not rel.startswith("r2m4_ladder_campaign")
        return None, False
    return CAMPAIGN / case_id, False


def harvest_case(case_id) -> tuple[dict, dict] | None:
    case, external = resolve_case(case_id)
    if case is None or not (case / "MANIFEST.json").is_file():
        return None
    manifest = json.loads((case / "MANIFEST.json").read_text())
    if external:   # pinned-driver bundle (Agent B's corrected matrix): map its fields
        tele0 = log_telemetry(case / "log.pimpleFoam")
        manifest = dict(manifest)
        manifest.setdefault("model", "equilibrium" if "equilibrium" in case.name else "totalGradient")
        manifest["model"] = "equilibrium" if "equilibrium" in case.name else "totalGradient"
        manifest.setdefault("grid", "W1")
        manifest["grid"] = "W1"
        manifest.setdefault("Ubar_volume", tele0["ubar_final"])
        manifest.setdefault("average_start", 135.0)
    times = sorted((case / "postProcessing_sampleBottomWall").iterdir(), key=lambda p: float(p.name))
    rows = L2.sample_rows(times[-1] / "bottomWall.xy")
    if (case / "input" / "polyMesh").is_dir() and (case / "input" / "C").is_file():
        mesh = L2.mesh_bottom(case)
        tangent_source = "mesh (deposit protocol)"
    else:
        mesh = analytic_mesh(rows)
        tangent_source = "analytic Xiao surface (input/polyMesh not staged)"
    curve = L2.phase_reduce(mesh, rows)
    surface = json.loads((case / "matching_surface.json").read_text()) if (case / "matching_surface.json").is_file() else {}
    record = {
        "case_id": case.name,
        "result_dir": str(case.relative_to(ROOT)),
        "external_source": ("Agent B corrected Re=5600 matrix (pinned driver, crest-bulk Ubar)" if external else None),
        "model": manifest["model"],
        "grid": manifest["grid"],
        "cells": manifest["grid_cells"],
        "producer_job_id": manifest.get("producer_job_id"),
        "nodes": manifest.get("nodes"),
        "solver_wall_seconds": manifest.get("solver_wall_seconds"),
        "latest_time": manifest["latest_time"],
        "average_start": manifest["average_start"],
        "Ubar_volume": manifest["Ubar_volume"],
        "tangent_source": tangent_source,
        "matching_height_over_H": {k: surface.get(f"matching_height_{k}") for k in ("min", "median", "max")},
        "driver_sha256": manifest.get("driver_sha256"),
        "ladder_kernel_sha256": manifest.get("ladder_kernel_sha256"),
        "sample_sha256": C.sha256(times[-1] / "bottomWall.xy"),
        "telemetry": log_telemetry(case / "log.pimpleFoam"),
    }
    return record, curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    fields = C.DnsTangentFields()
    references = T.references()
    dense_grid = np.arange(C.DENSE_N) / C.DENSE_N
    reference_dense = {k: C.periodic_interp(ph, ta, dense_grid) for k, (ph, ta, _) in references.items()}
    apriori = json.loads(APRIORI.read_text()) if APRIORI.is_file() else None
    result = {"schema": "r2m4-coupled-ladder-v2", "row": "R2-m4 / R3-2", "stamp": STAMP,
              "dns_sha256": C.sha256(C.DNS_FILE), "cases": {}, "pending": [],
              "primary_reference": T.PRIMARY, "bracket_reference": T.BRACKET,
              "withdrawn_reference": T.SUPERSEDED,
              "references": {k: v for k, v in T.LABELS.items()},
              "scoring_note": ("metrics[] and the verdicts are against the PRIMARY reference (MGLET DNS); "
                               "metrics_by_reference[] carries all three.  The 20260823 artifact scored "
                               "against the withdrawn reference and is retained as superseded."),
              "apriori_source": str(APRIORI.relative_to(ROOT)) if apriori else None,
              "rescore_source": str(RESCORED.relative_to(ROOT)) if RESCORED.is_file() else None}
    arrays = {"truth_phase": fields.phase, "truth_tau_s": fields.tau_s_truth}
    dense_by_grid: dict[str, dict[str, np.ndarray]] = {}
    truth_dense = None
    for key, case_id in CASES.items():
        harvested = harvest_case(case_id)
        if harvested is None:
            result["pending"].append(case_id if isinstance(case_id, str) else case_id[0])
            continue
        record, curve = harvested
        tele = record["telemetry"]
        checks = {
            "reached_405": abs(record["latest_time"] - 405.0) < 1e-6,
            "mass_flow_matched": tele["ubar_final"] is not None and abs(tele["ubar_final"] - EXPECTED_UBAR) < 1e-4*EXPECTED_UBAR + 5e-7,
            "courant_bounded": tele["max_courant"] is not None and tele["max_courant"] <= 0.56,
        }
        p_dense = C.periodic_interp(curve["phase"], curve["tau_s"], dense_grid)
        by_reference = {}
        for reference, truth_dense in reference_dense.items():
            e = p_dense - truth_dense
            tsep, trea = L2.zero_crossings(dense_grid, truth_dense)
            by_reference[reference] = {
                "relative_rms": float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(truth_dense ** 2))),
                "r2": float(1.0 - np.sum(e ** 2) / np.sum((truth_dense - truth_dense.mean()) ** 2)),
                "sign_accuracy": float(np.mean(np.sign(p_dense) == np.sign(truth_dense))),
                "signed_force_ratio": float(np.sum(p_dense) / np.sum(truth_dense)),
                "correlation": float(np.corrcoef(p_dense, truth_dense)[0, 1]),
                "truth_separation_x_over_H": tsep * C.LX,
                "truth_reattachment_x_over_H": trea * C.LX,
            }
        sep, rea = L2.zero_crossings(dense_grid, p_dense)
        m = dict(by_reference[T.PRIMARY])
        m.update({"separation_x_over_H": sep * C.LX, "reattachment_x_over_H": rea * C.LX,
                  "signed_tangent_force_per_span": curve["signed_tangent_force_per_span"]})
        record["checks"] = checks
        record["metrics"] = m
        record["metrics_by_reference"] = by_reference
        grid = key[0]
        dense_by_grid.setdefault(grid, {})[key[1]] = p_dense
        truth_dense = reference_dense[T.PRIMARY]
        arrays[f"{grid}_{key[1]}_phase"] = curve["phase"]
        arrays[f"{grid}_{key[1]}_tau_s"] = curve["tau_s"]
        arrays[f"{grid}_{key[1]}_ym"] = curve["ym"]
        result["cases"][f"{grid}:{key[1]}"] = record
    # phase-block bootstrap per grid (paired across models of one grid), on every reference
    for grid, preds in dense_by_grid.items():
        for reference, truth in reference_dense.items():
            b = C.block_bootstrap_relative_rms(truth, preds)
            for model, samples in b.items():
                result["cases"][f"{grid}:{model}"]["metrics_by_reference"][reference][
                    "relative_rms_interval"] = C.interval(samples)
            pairs = {}
            for a, bb in (("totalGradient", "equilibrium"), ("hickel", "totalGradient"),
                          ("resolvedConvectionLinear", "totalGradient"),
                          ("resolvedConvectionConstant", "totalGradient")):
                if a in b and bb in b:
                    pairs[f"{a}-minus-{bb}"] = C.interval(b[a] - b[bb])
            result.setdefault("paired_relative_rms_differences_by_reference", {}).setdefault(
                reference, {})[grid] = pairs
            if reference == T.PRIMARY and "resolvedConvectionLinear" in b:
                result.setdefault("verdict_by_reference", {}).setdefault(reference, {})[grid] = (
                    C.side_verdict(C.interval(b["resolvedConvectionLinear"]),
                                   pairs["resolvedConvectionLinear-minus-totalGradient"]))
            elif "resolvedConvectionLinear" in b:
                result.setdefault("verdict_by_reference", {}).setdefault(reference, {})[grid] = (
                    C.side_verdict(C.interval(b["resolvedConvectionLinear"]),
                                   pairs["resolvedConvectionLinear-minus-totalGradient"]))
        boot = C.block_bootstrap_relative_rms(truth_dense, preds)
        for model, samples in boot.items():
            result["cases"][f"{grid}:{model}"]["metrics"]["relative_rms_interval"] = C.interval(samples)
            arrays[f"{grid}_{model}_bootstrap_relative_rms"] = samples
        pairs = {}
        for a, b in (("totalGradient", "equilibrium"), ("hickel", "totalGradient"),
                     ("resolvedConvectionLinear", "totalGradient"), ("resolvedConvectionConstant", "totalGradient")):
            if a in boot and b in boot:
                pairs[f"{a}-minus-{b}"] = C.interval(boot[a] - boot[b])
        result.setdefault("paired_relative_rms_differences", {})[grid] = pairs
    # side-by-side with the a priori ladder on the same surface
    table = []
    for grid in ("L1", "L2", "W1"):
        for model, ladder_name in C.COUPLED_MODEL_OF.items():
            rec = result["cases"].get(f"{grid}:{model}")
            ap = None
            if RESCORED.is_file():
                _rs = json.loads(RESCORED.read_text())
                ap = _rs["apriori"][SURFACE_OF[grid]][T.PRIMARY]["metrics"].get(ladder_name)
            elif apriori:
                ap = apriori["surfaces"][SURFACE_OF[grid]]["metrics"].get(ladder_name)
            if rec is None and ap is None:
                continue
            table.append({"grid": grid, "model": model, "ladder": ladder_name,
                          "apriori_relative_rms": ap["relative_rms"] if ap else None,
                          "apriori_interval": ap["relative_rms_interval"] if ap else None,
                          "coupled_relative_rms": rec["metrics"]["relative_rms"] if rec else None,
                          "coupled_interval": rec["metrics"].get("relative_rms_interval") if rec else None,
                          "coupled_r2": rec["metrics"]["r2"] if rec else None,
                          "coupled_force_ratio": rec["metrics"]["signed_force_ratio"] if rec else None})
    result["ladder_table"] = table
    verdicts = result.get("verdict_by_reference", {}).get(T.PRIMARY, {})
    result["coupled_verdict"] = verdicts
    rescored = json.loads(RESCORED.read_text()) if RESCORED.is_file() else None
    if rescored and "L1" in verdicts:
        # the a-priori side must be scored against the SAME reference; it lives in the rescore artifact
        result["apriori_verdict_ladder_L1"] = (
            rescored["apriori"]["ladder_L1"][T.PRIMARY]["verdict_resolved_convection"])
        result["row_verdict"] = C.row_verdict(result["apriori_verdict_ladder_L1"], verdicts["L1"])
        result["row_verdict_bracket"] = C.row_verdict(
            rescored["apriori"]["ladder_L1"][T.BRACKET]["verdict_resolved_convection"],
            result.get("verdict_by_reference", {}).get(T.BRACKET, {}).get("L1", "INCONCLUSIVE"))
    elif apriori and "L1" in verdicts:
        result["apriori_verdict_ladder_L1"] = None
        result["row_verdict"] = None
    complete = all(f"{g}:{m}" in result["cases"] for g, m in MINIMUM)
    result["status"] = "R2M4_LADDER_HARVEST_OK" if complete else "R2M4_LADDER_HARVEST_PARTIAL"
    OUT_JSON.write_text(json.dumps(L2.json_ready(result), indent=2, sort_keys=True) + "\n")
    np.savez(OUT_NPZ, **arrays)
    for row in table:
        print("%-3s %-28s apriori=%-8s coupled=%-8s R2=%-8s force=%s" % (
            row["grid"], row["model"],
            "%.3f" % row["apriori_relative_rms"] if row["apriori_relative_rms"] is not None else "-",
            "%.3f" % row["coupled_relative_rms"] if row["coupled_relative_rms"] is not None else "pending",
            "%.2f" % row["coupled_r2"] if row["coupled_r2"] is not None else "-",
            "%.2f" % row["coupled_force_ratio"] if row["coupled_force_ratio"] is not None else "-"))
    print(result["status"], "pending:", result["pending"])
    print("verdicts:", {k: result.get(k) for k in ("apriori_verdict_ladder_L1", "coupled_verdict", "row_verdict")})
    if not args.no_figure and result["cases"]:
        make_figure(result, arrays, apriori)
    return 0


def make_figure(result, arrays, apriori):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(C.COUPLED_MODEL_OF)
    labels = ["M0 eq.", "M1 PG-ODE", "M2 Hickel", "Xc conv.\n(linear)", "Xc conv.\n(const.)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xs = np.arange(len(order))
    if apriori:
        ap = apriori["surfaces"]["ladder_L1"]["metrics"]
        vals = [ap[C.COUPLED_MODEL_OF[m]]["relative_rms"] for m in order]
        lo = [ap[C.COUPLED_MODEL_OF[m]]["relative_rms_interval"]["low"] for m in order]
        hi = [ap[C.COUPLED_MODEL_OF[m]]["relative_rms_interval"]["high"] for m in order]
        ax.errorbar(xs - 0.15, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="s", color="#1f77b4",
                    capsize=3, label="a priori (DNS inputs, same surface)")
    for grid, shift, color, mk in (("L1", 0.05, "#d62728", "o"), ("L2", 0.2, "#ff7f0e", "^")):
        pts = [(i, result["cases"][f"{grid}:{m}"]["metrics"]) for i, m in enumerate(order)
               if f"{grid}:{m}" in result["cases"]]
        if pts:
            v = [p[1]["relative_rms"] for p in pts]
            iv = [p[1].get("relative_rms_interval", {"low": p[1]["relative_rms"], "high": p[1]["relative_rms"]}) for p in pts]
            ax.errorbar([p[0] + shift for p in pts], v,
                        yerr=[np.subtract(v, [q["low"] for q in iv]), np.subtract([q["high"] for q in iv], v)],
                        fmt=mk, color=color, capsize=3, label=f"coupled WMLES {grid}")
    for m, ls in (("equilibrium", "--"), ("totalGradient", ":")):
        if f"W1:{m}" in result["cases"]:
            ax.axhline(result["cases"][f"W1:{m}"]["metrics"]["relative_rms"], ls=ls, color="k", lw=1,
                       label=f"wall-resolved ceiling W1 ({m})")
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"RMS$(\tau_s-\tau_s^{DNS})$ / RMS$(\tau_s^{DNS})$")
    ax.set_title("Model ladder, Xiao hill Re=5600, $y_m/H=0.094$")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(arrays["truth_phase"], 2 * arrays["truth_tau_s"], "k", lw=2, label="DNS")
    for m, color in zip(order, ("#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd")):
        if f"L1_{m}_phase" in arrays:
            ax.plot(arrays[f"L1_{m}_phase"], 2 * arrays[f"L1_{m}_tau_s"], color=color, lw=1, label=f"L1 {m}")
    if "W1_equilibrium_phase" in arrays:
        ax.plot(arrays["W1_equilibrium_phase"], 2 * arrays["W1_equilibrium_tau_s"], "k:", lw=1, label="W1 ceiling")
    ax.set_xlabel("phase $x/L_x$")
    ax.set_ylabel("$C_f$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(str(FIG) + ".pdf")
    fig.savefig(str(FIG) + ".png", dpi=160)


if __name__ == "__main__":
    raise SystemExit(main())
