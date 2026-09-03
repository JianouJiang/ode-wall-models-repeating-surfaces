#!/usr/bin/env python3
"""Re-score the four-point as-deployed ladder against the corrected references.

The ladder compares four estimands on one truth: the model on DNS matching
data, the same model on the simulation's own matching data, what the boundary
condition can deliver, and what the solver carried.  Its truth was built by the
withdrawn four-point through-origin wall-gradient fit, so the four published
scores are reference artifacts even though the GAPS between them are formed on
the simulation's own measured RMS and are therefore reference-free.

Nothing is recomputed from CFD and no prediction is touched.  Only the truth
changes:

    A  the withdrawn through-origin linear fit        (negative control)
    B  the wall-resolved full-wall DNS                (primary)
    C  a curvature-aware cubic on the same archive    (bracket)

Reproducing the deposited A column exactly is the instrument-fidelity check
that licenses reading the B and C columns.

Outputs codes/results/as_deployed_ladder_rebased_l0_<stamp>.json.
Run:  python3 codes/analysis/rebase_as_deployed_ladder_l0.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
STAMP = "20260825"
OUT = RESULTS / f"as_deployed_ladder_rebased_l0_{STAMP}.json"
REBASE_NPZ = RESULTS / f"reference_rebase_headlines_l0_{STAMP}.npz"
BRIDGE = ROOT / "codes" / "analysis" / "deployed_operator" / "bridge_ladder.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    BL = load(BRIDGE, "bridge_ladder_rebase")
    L2, HM = BL.L2, BL.HM
    dense = np.arange(BL.DENSE_N, dtype=float) / BL.DENSE_N

    # ---- the three truths, all on the same dense phase grid ---------------
    dns = np.load(HM.DNS_5600)
    xs = np.asarray(dns["x"], float)
    truth_phase = np.mod((xs - float(np.min(xs))) / BL.LX, 1.0)
    tau_A, _ = L2.dns_tangent_reference(dns)

    reb = np.load(REBASE_NPZ, allow_pickle=True)
    xr = np.asarray(reb["x"], float)
    reb_phase = np.mod((xr - float(np.min(xr))) / BL.LX, 1.0)
    if not np.allclose(np.sort(reb_phase), np.sort(truth_phase), atol=1e-9):
        raise SystemExit("corrected references are not on the archive's stations")

    truths = {
        "A_withdrawn_linear4": L2.periodic_interp(truth_phase, tau_A, dense),
        "B_mglet": L2.periodic_interp(
            reb_phase, np.asarray(reb["reference_B_mglet"], float), dense),
        "C_repaired_cubic6": L2.periodic_interp(
            reb_phase, np.asarray(reb["reference_C_repaired_cubic6"], float), dense),
    }

    npz = np.load(BL.latest("as_deployed_evaluation_*[0-9].npz"))
    summary = json.loads(
        BL.latest("as_deployed_evaluation_*_summary.json").read_text())
    matched = np.load(BL.MATCHED, allow_pickle=True)

    rows = []
    for r in summary["records"]:
        if r["patch"] != "bottomWall":
            continue
        rung, grid = BL.rung_grid(r["case"])
        akey = f"{rung}_{grid}_{r['model']}_apriori_matched_tau"
        pkey = f"{rung}_{grid}_{r['model']}_phase"
        if akey not in matched.files:
            continue
        preds = {
            "apriori_dns_input": L2.periodic_interp(matched[pkey],
                                                    matched[akey], dense),
            "request_les_input": npz[f"{r['case']}__{r['time']:.0f}__request__dense"],
            "delivered": npz[f"{r['case']}__{r['time']:.0f}__deliver__dense"],
            "measured": npz[f"{r['case']}__{r['time']:.0f}__measured__dense"],
        }
        row = {"case": r["case"], "rung": rung, "grid": grid,
               "model": r["model"], "time": r["time"],
               "ym_median": r["ym_median"], "scores": {}}
        for ref, truth in truths.items():
            row["scores"][ref] = {k: BL.score(p, truth, dense)
                                  for k, p in preds.items()}
        rows.append(row)
        a = row["scores"]["A_withdrawn_linear4"]
        b = row["scores"]["B_mglet"]
        print(f"{r['case'][:46]:46s} y_m={r['ym_median']:.4f} "
              f"A[{a['apriori_dns_input']['r2']:+8.3f} "
              f"{a['measured']['r2']:+8.3f}]  "
              f"B[{b['apriori_dns_input']['r2']:+8.3f} "
              f"{b['measured']['r2']:+8.3f}]", flush=True)

    # ---- instrument fidelity: reproduce the deposited A column ------------
    dep = json.loads((RESULTS / "as_deployed_bridge_ladder.json").read_text())
    key = {(d["case"], d["time"]): d for d in dep["rows"]}
    worst, n_checked = 0.0, 0
    for row in rows:
        d = key.get((row["case"], row["time"]))
        if not d:
            continue
        for point in ("apriori_dns_input", "request_les_input",
                      "delivered", "measured"):
            got = row["scores"]["A_withdrawn_linear4"][point]["r2"]
            want = d["scores"][point]["r2"]
            worst = max(worst, abs(got - want))
            n_checked += 1

    # ---- what the reference swap does to the ladder -----------------------
    def ladder(row, ref):
        s = row["scores"][ref]
        return [s[p]["r2"] for p in ("apriori_dns_input", "request_les_input",
                                     "delivered", "measured")]

    def delivery_moves_toward_truth(row, ref):
        """The section's actual claim: the delivery map improves the score."""
        s = row["scores"][ref]
        return bool(s["delivered"]["r2"] > s["request_les_input"]["r2"])

    canonical = [r for r in rows
                 if abs(r["ym_median"] - 0.0935) < 5e-3
                 and r["model"] == "total_gradient_tble" and r["grid"] == "G1c"]

    payload = {
        "generated_by": "codes/analysis/rebase_as_deployed_ladder_l0.py",
        "question": ("does the four-point as-deployed ladder survive the "
                     "reference correction, and which of its statements are "
                     "reference-free?"),
        "instrument_fidelity": {
            "reproduces_deposited_A_column": worst < 1e-9,
            "worst_absolute_r2_deviation": worst,
            "n_scores_checked": n_checked,
        },
        "canonical_surface_ladder": {
            ref: (ladder(canonical[0], ref) if canonical else None)
            for ref in truths
        },
        "delivery_moves_score_toward_truth": {
            ref: sum(delivery_moves_toward_truth(r, ref) for r in rows)
            for ref in truths
        },
        "n_rows": len(rows),
        "reference_free_note": (
            "the three gaps (I), (D), (N) are normalised by the simulation's own "
            "measured RMS and never touch a reference traction, so they are "
            "unchanged by this rebase; only the four R^2 columns move"),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"\ninstrument fidelity : worst |dR2| vs deposited A column = "
          f"{worst:.3e} over {n_checked} scores")
    if canonical:
        for ref in truths:
            vals = ladder(canonical[0], ref)
            print(f"canonical ladder {ref:22s} "
                  + "  ".join(f"{v:+8.3f}" for v in vals))
    print("delivery improves the score, cases out of "
          f"{len(rows)}: "
          + ", ".join(f"{k}={v}" for k, v in
                      payload["delivery_moves_score_toward_truth"].items()))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
