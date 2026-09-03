#!/usr/bin/env python3
"""R2-m4 / R3-2 a-priori model ladder on the Xiao alpha=1 DNS (Re_H=5600).

Evaluates, at the SAME physical matching surface as the coupled ladder
(first-cell-centre heights of the L1 ladder mesh, y_m/H = 0.0935 on the flat
floor), the wall-traction prediction of
  M0  equilibrium (Spalding)            M1  pressure-gradient TBLE ODE
  M2  Hickel parametrised convection    Xc  ODE + resolved convection (linear,
  constant one-value reconstructions)   Xc* ODE + exact within-layer convection
against the tangent-frame DNS traction, with the deposited phase-block
uncertainty protocol.  Two further surfaces are reported: the paper's archive
index-10 surface (512 stations) and the deposited common surface (y_m/H =
0.0145) used by the wall-resolved ceiling grid W1.

Output: codes/results/r2m4_apriori_ladder_20260823.{json,npz}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402

STAMP = "20260823"
OUT_JSON = ROOT / "codes/results" / f"r2m4_apriori_ladder_{STAMP}.json"
OUT_NPZ = ROOT / "codes/results" / f"r2m4_apriori_ladder_{STAMP}.npz"
L1_SURFACE = (ROOT / "work_progress/archer2_campaign_20260823/R2-m4/"
              "ladder_surface_L1_local_blockmesh.json")
L2_NPZ = ROOT / "codes/results/rswm_common_surface_grid_l2.npz"


def surfaces(fields: C.DnsTangentFields):
    l1 = json.loads(L1_SURFACE.read_text())
    if l1["status"] != "MESH_MATCHING_SURFACE_OK":
        raise RuntimeError("L1 surface record is not verified")
    l2 = np.load(L2_NPZ)
    dns = np.load(C.DNS_FILE)
    y = np.asarray(dns["y"], float)
    offset10 = y[:, 10] - y[:, 0]
    return {
        "ladder_L1": (np.asarray(l1["phase_over_period"]), np.asarray(l1["phase_matching_height_over_H"]),
                      "first cell centre of the ladder mesh (y_m/H=0.0935 flat floor)"),
        "archive_index10": (fields.phase, offset10 * fields.tx,
                            "paper a-priori surface: archive wall index 10 (vertical offset x t_x)"),
        "common_W1": (np.asarray(l2["G1c_equilibrium_phase"]), np.asarray(l2["G1c_equilibrium_ym"]),
                      "deposited common surface (y_m/H=0.0145 flat floor), wall-resolved ceiling grid"),
    }


def main() -> int:
    fields = C.DnsTangentFields()
    result = {
        "schema": "r2m4-apriori-ladder-v1",
        "row": "R2-m4 / R3-2",
        "dns_source": str(C.DNS_FILE.relative_to(ROOT)),
        "dns_sha256": C.sha256(C.DNS_FILE),
        "driving_acceleration": C.DNS_DRIVING_ACCELERATION,
        "truth": "nu dU_t/dn, first four archive points, analytic Xiao tangent (deposit protocol)",
        "effective_gradient": "(dp_wall/dx - g) t_x, g from wall_following_budget_l1",
        "convection": "mean-field (U.grad)U . t_s from the 512x257 archive, derivatives in (x, eta), sampled on the wall-normal line",
        "extended_members_note": ("Xp/Xcp/Xcpr/Xall add the exact pressure-gradient profile, streamwise normal-stress "
                                  "gradient and streamwise viscous term; Xfull is the closure-free balance "
                                  "tau(y_m) - impulses.  Their pointwise normal-line balance closes only on the flat "
                                  "floor (flat_floor_metrics); the conservative wavelength balance is R1-STA-3."),
        "root_policy": "audited all-bracket census; root closest to the Spalding value",
        "ladder": list(C.LADDER),
        "bootstrap": {"draws": C.BOOTSTRAP_DRAWS, "block_points": C.BLOCK_POINTS,
                      "dense_points": C.DENSE_N, "seed": C.BOOTSTRAP_SEED},
        "surfaces": {},
    }
    arrays = {"truth_phase": fields.phase, "truth_tau_s": fields.tau_s_truth}
    for name, (phase, y_m, note) in surfaces(fields).items():
        preds, diag = C.ladder_predictions(fields, phase, y_m)
        dense_preds = {}
        metrics = {}
        for model, tau in preds.items():
            m, dense, t_dense, p_dense = C.phase_metrics(phase, tau, fields.phase, fields.tau_s_truth)
            metrics[model] = m
            dense_preds[model] = p_dense
            arrays[f"{name}_{model}"] = tau
        # flat-floor restriction (2.05 <= x/H <= 6.90, where wall-normal and
        # wall-following coordinates coincide): secondary table for the
        # extended members whose pointwise normal-line balance does not close
        # on the slopes (see closure diagnostic)
        flat = (dense * C.LX >= 2.05) & (dense * C.LX <= 6.90)
        flat_metrics = {}
        for model, p_dense in dense_preds.items():
            e = p_dense[flat] - t_dense[flat]
            flat_metrics[model] = {
                "relative_rms": float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(t_dense[flat] ** 2))),
                "r2": float(1.0 - np.sum(e ** 2) / np.sum((t_dense[flat] - t_dense[flat].mean()) ** 2)),
                "sign_accuracy": float(np.mean(np.sign(p_dense[flat]) == np.sign(t_dense[flat]))),
            }
        boot = C.block_bootstrap_relative_rms(t_dense, dense_preds)
        for model in preds:
            metrics[model]["relative_rms_interval"] = C.interval(boot[model])
        # paired differences along the ladder (same resampled blocks)
        deltas = {}
        for a, b in (("M1_pressure_gradient_ode", "M0_equilibrium"),
                     ("M2_hickel_modelled_convection", "M1_pressure_gradient_ode"),
                     ("Xc_resolved_convection_linear", "M1_pressure_gradient_ode"),
                     ("Xc_exact_convection_profile", "M1_pressure_gradient_ode")):
            deltas[f"{a}-minus-{b}"] = C.interval(boot[a] - boot[b])
        for k, v in diag.items():
            arrays[f"{name}_diag_{k}"] = v
        arrays[f"{name}_phase"] = phase
        arrays[f"{name}_y_m"] = y_m
        result["surfaces"][name] = {
            "note": note,
            "phase_count": int(len(phase)),
            "y_m_over_H": {"min": float(np.min(y_m)), "median": float(np.median(y_m)),
                           "max": float(np.max(y_m))},
            "y_m_plus_median": float(np.median(y_m * np.sqrt(np.abs(diag["truth"])) / C.NU)),
            "epsilon_median": float(np.median(np.abs(diag["truth"]) /
                                              np.maximum(np.abs(diag["dpds"]) * y_m, 1e-30))),
            "convection_impulse_over_pressure_impulse_median": float(
                np.median(diag["conv_impulse_over_pressure_impulse"])),
            "multi_root_M1_stations": int(np.sum(diag["roots_M1"] > 1)),
            "metrics": metrics,
            "flat_floor_metrics": flat_metrics,
            "paired_relative_rms_differences": deltas,
        }
        print(f"== {name}: {note}", flush=True)
        for model in C.LADDER:
            m = metrics[model]
            print(f"   {model:42s} relRMS={m['relative_rms']:.3f} "
                  f"[{m['relative_rms_interval']['low']:.3f},{m['relative_rms_interval']['high']:.3f}] "
                  f"R2={m['r2']:.2f} sign={m['sign_accuracy']:.3f} force={m['signed_force_ratio']:.2f} "
                  f"| flat: relRMS={flat_metrics[model]['relative_rms']:.3f} R2={flat_metrics[model]['r2']:.2f}",
                  flush=True)
    s = result["surfaces"]["ladder_L1"]
    result["apriori_verdict_ladder_L1"] = C.side_verdict(
        s["metrics"]["Xc_resolved_convection_linear"]["relative_rms_interval"],
        s["paired_relative_rms_differences"]["Xc_resolved_convection_linear-minus-M1_pressure_gradient_ode"])
    result["acceptance_rule"] = ("SUPPORTED iff interval(E(Xc_lin)-E(M1)).high<0 and interval(E(Xc_lin)).high<1; "
                                 "REFUTED iff interval(E(Xc_lin)-E(M1)).low>0 or interval(E(Xc_lin)).low>=1")
    result["status"] = "R2M4_APRIORI_LADDER_OK"
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez(OUT_NPZ, **arrays)
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_NPZ.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
