#!/usr/bin/env python3
"""
Full-span crest-section flux: the registered drive gate, measured properly.

The campaign registered a 3% acceptance gate on the measured crest-section
bulk velocity.  It was scored with a single spanwise line of the time-averaged
velocity at mid-span (z = 2.25), and five of seventeen scored points failed.
That estimator samples one spanwise station, so it carries the full spanwise
variability of the mean field; the campaign's own records show its
station-to-station scatter reaching 9.6%, i.e. as large as the deviation being
gated.

This script replaces it with the complete section integral

    u_b,crest = 1/(h_crest L_z) * \\int\\int UMean_x dy dz ,

evaluated by OpenFOAM's own plane quadrature (`surfaceFieldValue`,
`areaIntegrate`) on the retained three-dimensional time-averaged field of each
deposited case, at several streamwise stations.  Mass conservation makes the
section flux streamwise-invariant, so the spread across stations is a measured
uncertainty on the estimator rather than an assumption.

Input : the raw sweep log written by the remote plane-integration pass.
Output: codes/results/full_span_crest_flux.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(PROJECT, "codes", "results")

CREST_HEIGHT = 2.036   # H, Xiao alpha = 1 crest section
LZ = 4.5               # H, spanwise extent
GATE = 0.03            # registered acceptance threshold

# The hill body occupies x/H in [0, 1.929] and [7.071, 9].  A constant-x plane
# there cuts the solid, so those stations are reported but excluded from the
# gate statistic, which uses the unobstructed channel stations.
CLEAR_STATIONS = (2.00, 4.05, 6.00)

LINE = re.compile(
    r"^FLUX (\S+) x=([0-9.]+) ([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\s*$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(
        PROJECT, "development", "nodes", "node_011", "work", "planeflux.log"))
    ap.add_argument("--out", default=os.path.join(
        RESULTS, "full_span_crest_flux.json"))
    args = ap.parse_args()

    per_case: dict[str, dict[float, float]] = {}
    for line in open(args.log, errors="replace"):
        m = LINE.match(line.strip())
        if not m:
            continue
        case, x, qx = m.group(1), float(m.group(2)), float(m.group(3))
        per_case.setdefault(case, {})[x] = qx

    if not per_case:
        print("no FLUX records parsed")
        return 2

    records = []
    for case in sorted(per_case):
        stations = per_case[case]
        clear = {x: q for x, q in stations.items() if x in CLEAR_STATIONS}
        if not clear:
            continue
        ub = {x: q / (CREST_HEIGHT * LZ) for x, q in stations.items()}
        ub_clear = np.array([ub[x] for x in sorted(clear)])
        mean = float(np.mean(ub_clear))
        rec = dict(
            case=case,
            n_stations=len(stations),
            n_clear_stations=int(ub_clear.size),
            crest_bulk_full_span=mean,
            abs_deviation=abs(mean - 1.0),
            station_spread=float(np.max(ub_clear) - np.min(ub_clear)),
            per_station={f"{x:.2f}": ub[x] for x in sorted(ub)},
            gate=GATE,
            outcome="PASS" if abs(mean - 1.0) <= GATE else "FAIL",
        )
        records.append(rec)
        print(f"  {case:<48s} u_b,crest={mean:.6f} "
              f"|dev|={rec['abs_deviation']*100:.4f}%  "
              f"spread={rec['station_spread']*100:.4f}%  {rec['outcome']}")

    devs = np.array([r["abs_deviation"] for r in records])
    spreads = np.array([r["station_spread"] for r in records])
    out = dict(
        method=("full-span section integral of the retained 3-D time-averaged "
                "velocity, by OpenFOAM surfaceFieldValue/areaIntegrate; "
                "u_b,crest = Q_x/(h_crest L_z), h_crest=2.036H, L_z=4.5H"),
        crest_height=CREST_HEIGHT, span=LZ, gate=GATE,
        clear_stations=list(CLEAR_STATIONS),
        n_cases=len(records),
        n_pass=int(np.sum(devs <= GATE)),
        max_abs_deviation=float(np.max(devs)),
        median_abs_deviation=float(np.median(devs)),
        max_station_spread=float(np.max(spreads)),
        all_pass=bool(np.all(devs <= GATE)),
        records=records,
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\n{out['n_pass']}/{out['n_cases']} cases within the registered "
          f"{GATE*100:.0f}% gate; largest deviation "
          f"{out['max_abs_deviation']*100:.4f}%, largest station spread "
          f"{out['max_station_spread']*100:.4f}%")
    print("wrote", args.out)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
