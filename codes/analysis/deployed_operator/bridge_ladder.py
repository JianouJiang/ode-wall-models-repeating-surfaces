#!/usr/bin/env python3
"""bridge_ladder.py -- the four-point estimand ladder, per coupled case.

Joins this node's as-deployed evaluation to the deposited model-matched
a-priori reduction so that all four points of the ladder are scored with one
truth, one tangent, one metric and one bootstrap:

    S(M_DNS)        a priori: DNS matching data, the published estimand
    S(M_LES)        the same model on the LES's own developed matching data
    P(S(M_LES))     what the boundary condition can deliver
    <tau_applied>   what the solver actually carried

and reports the three gaps

    (I) input transfer   = S(M_LES)  - S(M_DNS)
    (D) delivery deficit = P(S(M_LES)) - S(M_LES)
    (N) residual         = <tau_applied> - P(S(M_LES))

Inputs are two existing result files; nothing is recomputed from CFD.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import harvest_m13_highre as HM              # noqa: E402

L2 = HM.load_module(HM.L2_REDUCER, "rswm_l2_bridge")
L3 = HM.load_module(HM.L3_ANALYSER, "rswm_l3_bridge")
LX, DENSE_N = 9.0, 4096
BLOCK_POINTS, DRAWS, SEED = 512, 20000, 20260824
MATCHED = RESULTS / "model_matched_transfer_l2_20260824.npz"
CAMPAIGN = RESULTS / "rswm_r23m6_ym_campaign_final"


def latest(pattern: str) -> Path:
    hits = [h for h in sorted(glob.glob(str(RESULTS / pattern)))
            if "pilot" not in h]
    if not hits:
        raise SystemExit(f"no file matching {pattern}")
    return Path(hits[-1])


def rung_grid(case: str) -> tuple[str, str]:
    rung = case.split("_")[2]                       # rswm_r23m6_<rung>_...
    grid = "G2c" if "819200" in case else "G1c"
    return rung, grid


def score(dense_pred, truth_dense, dense_phase):
    err = dense_pred - truth_dense
    denom = float(np.sum((truth_dense - truth_dense.mean()) ** 2))
    samples = L3.circular_block_bootstrap(
        truth_dense, {("c", "m"): dense_pred},
        block_points=BLOCK_POINTS, draws=DRAWS, seed=SEED)[("c", "m")]
    return {
        "r2": float(1.0 - np.sum(err ** 2) / denom),
        "relative_rms": float(np.sqrt(np.mean(err ** 2))
                              / np.sqrt(np.mean(truth_dense ** 2))),
        "sign_accuracy": float(np.mean(np.sign(dense_pred)
                                       == np.sign(truth_dense))),
        "interval_relative_rms": {
            "median": float(np.median(samples)),
            "lo": float(np.quantile(samples, 0.025)),
            "hi": float(np.quantile(samples, 0.975))},
    }


def main() -> int:
    npz = np.load(latest("as_deployed_evaluation_*[0-9].npz"))
    summary = json.loads(latest("as_deployed_evaluation_*_summary.json").read_text())
    matched = np.load(MATCHED, allow_pickle=True)

    dns = np.load(HM.DNS_5600)
    truth_tau, _ = L2.dns_tangent_reference(dns)
    xs = np.asarray(dns["x"], float)
    truth_phase = np.mod((xs - float(np.min(xs))) / LX, 1.0)
    dense = np.arange(DENSE_N, dtype=float) / DENSE_N
    truth_dense = L2.periodic_interp(truth_phase, truth_tau, dense)

    rows = []
    for r in summary["records"]:
        if r["patch"] != "bottomWall":
            continue
        rung, grid = rung_grid(r["case"])
        akey = f"{rung}_{grid}_{r['model']}_apriori_matched_tau"
        pkey = f"{rung}_{grid}_{r['model']}_phase"
        if akey not in matched.files:
            continue
        apriori = L2.periodic_interp(matched[pkey], matched[akey], dense)
        base = f"{r['case']}__{r['time']:.0f}__"
        request = npz[base + "request__dense"]
        deliver = npz[base + "deliver__dense"]
        measured = npz[base + "measured__dense"]
        scale = float(np.sqrt(np.mean(measured ** 2)))

        def rms(a):
            return float(np.sqrt(np.mean(a ** 2)))

        # Commensurability check: the coupled curve this node reduces must be
        # the SAME curve the paper already reports.  Rebuild it independently
        # with the pinned L2 reducer from the deposit's own sampled wall file.
        deposit_dev = np.nan
        case_dir = CAMPAIGN / r["deposit"] / r["case"]
        sampled = (case_dir / "postProcessing_sampleBottomWall"
                   / f"{r['time']:g}" / "bottomWall.xy")
        if sampled.exists():
            mesh = L2.mesh_bottom(case_dir)
            pinned = L2.phase_reduce(mesh, L2.sample_rows(sampled))
            pinned_dense = L2.periodic_interp(pinned["phase"], pinned["tau_s"],
                                              dense)
            deposit_dev = float(np.max(np.abs(pinned_dense - measured))
                                / max(scale, 1e-30))

        rows.append({
            "case": r["case"], "deposit": r["deposit"], "rung": rung,
            "grid": grid, "model": r["model"], "time": r["time"],
            "ym_median": r["ym_median"],
            "averaging_window": r["averaging_window"],
            "scores": {
                "apriori_dns_input": score(apriori, truth_dense, dense),
                "request_les_input": score(request, truth_dense, dense),
                "delivered": score(deliver, truth_dense, dense),
                "measured": score(measured, truth_dense, dense)},
            "pinned_reducer_max_deviation_over_rms": deposit_dev,
            "gaps_over_measured_rms": {
                "I_input_transfer": rms(request - apriori) / scale,
                "D_delivery_deficiency": rms(deliver - request) / scale,
                "N_residual": rms(measured - deliver) / scale,
                "total_apriori_to_measured": rms(measured - apriori) / scale},
        })
        print(f"LADDER {r['case']} t={r['time']:.0f} "
              f"R2 apriori={rows[-1]['scores']['apriori_dns_input']['r2']:.3g} "
              f"request={rows[-1]['scores']['request_les_input']['r2']:.3g} "
              f"deliver={rows[-1]['scores']['delivered']['r2']:.3g} "
              f"measured={rows[-1]['scores']['measured']['r2']:.3g}", flush=True)

    out = RESULTS / "as_deployed_bridge_ladder.json"
    out.write_text(json.dumps({
        "generated_from": {
            "as_deployed": latest("as_deployed_evaluation_*[0-9].npz").name,
            "model_matched": MATCHED.name,
            "model_matched_sha256": HM.sha256(MATCHED)},
        "n_rows": len(rows), "rows": rows}, indent=1))
    print(f"WROTE {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
