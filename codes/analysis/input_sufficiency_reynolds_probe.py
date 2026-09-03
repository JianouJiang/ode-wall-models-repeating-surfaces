#!/usr/bin/env python3
"""Reynolds-number probe for the input-sufficiency bracket.

The 29-case hill family is entirely at Re_b = 5600, so a fair objection to the
main measurement is that the empirical instrument transfers only because every
training case shares one Reynolds number.  The deployed models share that
advantage and still fail, so the comparison is matched either way -- but the
objection is worth answering directly.

This probe trains the same geometry-blind function of (a, b) on the 29 Xiao
hills at Re_b = 5600 and evaluates it on the Krank et al. periodic hill at
Re_H = 10595, an independent DNS of a hill it has never seen at a Reynolds
number it has never seen, with the deployed models evaluated at the same
stations.  The Krank archive supplies only ten wall stations, so the scores are
reported with that n printed and are treated as a probe, not as a result.

Output: codes/results/input_sufficiency_reynolds_probe.json
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

KRANK = {
    "krank_pehill_Re5600": ROOT / ("codes/new_data_download/geometry_driven/"
                                   "krank_pehill_Re5600_wall_profiles.npz"),
    "krank_pehill_Re10595": ROOT / ("codes/new_data_download/geometry_driven/"
                                    "krank_pehill_Re10595_wall_profiles.npz"),
}


def load_krank(path: Path, frac: float) -> dict:
    d = np.load(path, allow_pickle=True)
    nu = float(np.atleast_1d(d["nu"]).reshape(-1)[0])
    c = isb.station_state(d["y"], d["U"], d["tau_w"], d["dp_dx"], nu, frac)
    c.update(name=path.stem, family="periodic_hill_krank", group="krank")
    return isb.groups(c)


def main() -> int:
    out = {"schema": "input_sufficiency_reynolds_probe_v1", "heights": {}}
    for frac in isb.ETA_FRACTIONS:
        hills, _, _ = isb.build_cases(frac)
        rec = {}
        for name, path in KRANK.items():
            if not path.exists():
                rec[name] = {"status": "ABSENT"}
                continue
            tgt = load_krank(path, frac)
            dep = isb.deployed_predictions(tgt, frac)
            entry = {"n_station": tgt["n_station"], "nu": tgt["nu"]}
            for use_b in (False, True):
                key = "a_and_b" if use_b else "a_only"
                pred, _, _ = isb.knn_transfer(hills, tgt, use_b)
                entry[key] = {
                    "r2_empirical": isb.r2_score(pred, tgt["tau"]),
                    "relrms_empirical": isb.rel_rms(pred, tgt["tau"]),
                }
            entry["r2_equilibrium"] = isb.r2_score(dep["m0"], tgt["tau"])
            entry["r2_tble"] = isb.r2_score(dep["m1"], tgt["tau"])
            entry["relrms_equilibrium"] = isb.rel_rms(dep["m0"], tgt["tau"])
            entry["relrms_tble"] = isb.rel_rms(dep["m1"], tgt["tau"])
            rec[name] = entry
            print(f"[eta{frac:.2f}] {name} n={tgt['n_station']}: "
                  f"empirical(a,b) R2={entry['a_and_b']['r2_empirical']:.3f}, "
                  f"equilibrium {entry['r2_equilibrium']:.3f}, "
                  f"TBLE {entry['r2_tble']:.3f}", flush=True)
        out["heights"][f"eta{frac:.2f}"] = rec
    out["caveat"] = ("ten wall stations per Krank case; a probe of Reynolds-number "
                     "transfer, not a scored result")
    dest = ROOT / "codes/results/input_sufficiency_reynolds_probe.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
