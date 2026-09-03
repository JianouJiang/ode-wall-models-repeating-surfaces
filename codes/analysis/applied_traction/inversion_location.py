#!/usr/bin/env python3
"""
Where along the hill does the boundary condition discard the model's sign?

`reproduce_applied_traction.py` establishes that the deployed projection alters
the requested wall stress and, on a subset of faces, applies a traction that
opposes it.  Sign reversal happens exactly where the wall model asks for a
stress opposing the matching-point velocity, which a non-negative scalar eddy
viscosity cannot express.  This script locates those faces in the streamwise
phase x/L, so the reversal can be compared with the separated region rather
than quoted as a bulk percentage.

Parses only the solver's own per-face records (face centres included), so it
needs no kernel evaluation and runs in seconds.

Output: codes/results/inversion_location.npz / _summary.json
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

LX = 9.0  # Xiao alpha = 1 periodic-hill streamwise period, in H

REC = re.compile(
    r"TOTAL_GRADIENT_TBLE_FACE patch=(\w+).*?centre=\(([-0-9.eE+]+) "
    r"([-0-9.eE+]+) ([-0-9.eE+]+)\).*?UMatch=([-0-9.eE+]+).*?"
    r"tauW=([-0-9.eE+]+).*?appliedTau=([-0-9.eE+]+)"
)

NBINS = 60


def parse(log_path: str, patch: str = "bottomWall"):
    x, um, tau, app = [], [], [], []
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if "TOTAL_GRADIENT_TBLE_FACE" not in line:
                continue
            m = REC.search(line)
            if not m or m.group(1) != patch:
                continue
            x.append(float(m.group(2)))
            um.append(float(m.group(5)))
            tau.append(float(m.group(6)))
            app.append(float(m.group(7)))
    return (np.asarray(x), np.asarray(um), np.asarray(tau), np.asarray(app))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(
        RESULTS, "rswm_r23m6_ym_campaign_final"))
    ap.add_argument("--patch", default="bottomWall")
    ap.add_argument("--out", default=os.path.join(RESULTS, "inversion_location"))
    args = ap.parse_args()

    cases = []
    for root, _d, files in os.walk(args.bundle):
        if "log.pimpleFoam" in files:
            cases.append(root)
    cases.sort()

    summaries, blobs = [], {}
    for case_dir in cases:
        name = os.path.basename(case_dir)
        x, um, tau, app = parse(os.path.join(case_dir, "log.pimpleFoam"), args.patch)
        if x.size == 0:
            continue
        phase = (x % LX) / LX
        nz = (tau != 0) & (app != 0)
        inverted = np.zeros_like(phase, dtype=bool)
        inverted[nz] = np.sign(tau[nz]) != np.sign(app[nz])

        edges = np.linspace(0.0, 1.0, NBINS + 1)
        which = np.clip(np.digitize(phase, edges) - 1, 0, NBINS - 1)
        frac = np.full(NBINS, np.nan)
        reversed_request = np.full(NBINS, np.nan)
        for b in range(NBINS):
            sel = which == b
            if sel.sum():
                frac[b] = float(np.mean(inverted[sel]))
                reversed_request[b] = float(np.mean(tau[sel] < 0))

        # The separated region of the Xiao alpha=1 hill: the DNS mean
        # reattachment is at x/H = 4.473, the hill crest region ends near
        # x/H = 1.  Quote the reversal fraction inside and outside it.
        sep = (phase * LX >= 0.5) & (phase * LX <= 4.473)
        rec = dict(
            case=name,
            n_faces=int(x.size),
            frac_inverted_all=float(np.mean(inverted)),
            frac_inverted_separated=float(np.mean(inverted[sep]))
            if sep.any() else float("nan"),
            frac_inverted_attached=float(np.mean(inverted[~sep]))
            if (~sep).any() else float("nan"),
            frac_requested_negative=float(np.mean(tau < 0)),
            peak_bin_phase=float(0.5 * (edges[int(np.nanargmax(frac))]
                                        + edges[int(np.nanargmax(frac)) + 1])),
            peak_bin_fraction=float(np.nanmax(frac)),
        )
        summaries.append(rec)
        blobs[f"{name}__phase_bin"] = 0.5 * (edges[:-1] + edges[1:])
        blobs[f"{name}__frac_inverted"] = frac
        blobs[f"{name}__frac_requested_negative"] = reversed_request
        print(f"  {name:<46s} inverted all={rec['frac_inverted_all']*100:5.1f}% "
              f"separated={rec['frac_inverted_separated']*100:5.1f}% "
              f"attached={rec['frac_inverted_attached']*100:5.1f}% "
              f"peak at x/L={rec['peak_bin_phase']:.2f}", flush=True)

    if not summaries:
        print("no per-face records found")
        return 2
    out = dict(bundle=os.path.relpath(args.bundle, PROJECT), patch=args.patch,
               separated_window_xH=[0.5, 4.473],
               n_cases=len(summaries), records=summaries)
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out + ".npz", **blobs)
    print(f"\nwrote {args.out}.npz and {args.out}_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
