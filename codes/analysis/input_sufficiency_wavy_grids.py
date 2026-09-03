#!/usr/bin/env python3
"""Grid robustness of the cross-family arm of the input-sufficiency bracket.

The main measurement transfers a same-input empirical function from the 29-case
hill family to the wavy wall and finds that it does not transfer.  A referee's
first question is whether that collapse is a property of the wavy reference or
of the grid it was computed on.  The R1-STA-2 deposit carries three converged
grids (0.79 M, 2.36 M, 7.30 M cells, a 9.3x cell range), so the question is
answerable without any new simulation.

Output: codes/results/input_sufficiency_wavy_grids.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))

import input_sufficiency_bracket as isb  # noqa: E402

WAVY = ROOT / "codes/results/r1_sta2_wavy_wrles_20260824.npz"


def load_wavy_grid(grid: str, frac: float) -> dict:
    d = np.load(WAVY, allow_pickle=True)
    c = isb.station_state(d[f"{grid}_ycell"], d[f"{grid}_U"], d[f"{grid}_tau_t"],
                          np.gradient(d[f"{grid}_p_wall"], d[f"{grid}_x"]),
                          isb.NU_WAVY, frac)
    c.update(name=f"wavy_{grid}", family="wavy_wall", group="wavy")
    return isb.groups(c)


def main() -> int:
    out = {"schema": "input_sufficiency_wavy_grids_v1",
           "source_sha256": isb.sha256(WAVY), "heights": {}}
    for frac in isb.ETA_FRACTIONS:
        hills, _, _ = isb.build_cases(frac)
        rec = {}
        for grid in ("G0", "G1", "G2"):
            tgt = load_wavy_grid(grid, frac)
            dep = isb.deployed_predictions(tgt, frac)
            entry = {"n_station": tgt["n_station"]}
            for use_b in (False, True):
                pred, _, _ = isb.knn_transfer(hills, tgt, use_b)
                entry["a_and_b" if use_b else "a_only"] = {
                    "r2_hills_to_wavy": isb.r2_score(pred, tgt["tau"]),
                    "relrms_hills_to_wavy": isb.rel_rms(pred, tgt["tau"])}
            entry["r2_equilibrium"] = isb.r2_score(dep["m0"], tgt["tau"])
            entry["r2_tble"] = isb.r2_score(dep["m1"], tgt["tau"])
            rec[grid] = entry
            print(f"[eta{frac:.2f}] {grid} n={tgt['n_station']}: "
                  f"hills->wavy (a,b) R2={entry['a_and_b']['r2_hills_to_wavy']:.3f}; "
                  f"equilibrium {entry['r2_equilibrium']:.3f}; "
                  f"TBLE {entry['r2_tble']:.3f}", flush=True)
        vals = [rec[g]["a_and_b"]["r2_hills_to_wavy"] for g in ("G0", "G1", "G2")]
        rec["grid_spread_a_and_b"] = float(max(vals) - min(vals))
        rec["collapse_is_grid_robust"] = bool(max(vals) < 0.0)
        out["heights"][f"eta{frac:.2f}"] = rec
    dest = ROOT / "codes/results/input_sufficiency_wavy_grids.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
