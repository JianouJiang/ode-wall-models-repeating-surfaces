#!/usr/bin/env python3
"""
Time history of the wall-model realizability projection in the coupled runs.

The per-face records replayed by `reproduce_applied_traction.py` are the
solver's first-solve states.  The boundary condition additionally writes a
periodic `TBLE_REALIZABILITY` summary for each wall patch throughout the run,
carrying its own `time=` stamp.  That series is the production statistic: it
says what fraction of wall faces had the requested wall stress altered before
it reached the large-eddy simulation, at every audited step of the run.

Reported per case and patch, restricted to the analysis window:
  * frac_clipped        -- faces whose traction the projection changed
  * frac_lower_clipped  -- faces where a negative eddy viscosity was refused
  * frac_vector_capped  -- faces where the complete traction was bounded
  * mismatch_per_face   -- mean |applied - requested| per wall face
  * mismatch_max        -- largest single-face departure

Output: codes/results/clipping_timeseries.npz / _summary.json
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

LINE = re.compile(
    r"TBLE_REALIZABILITY patch=(\w+) time=([0-9.eE+-]+) faces=(\d+) "
    r"clipped=(\d+) fraction=([0-9.eE+-]+) mismatchL1=([0-9.eE+-]+) "
    r"mismatchMax=([0-9.eE+-]+) lowerClipped=(\d+) vectorCapped=(\d+) "
    r"appliedTractionMax=([0-9.eE+-]+) degenerateRoots=(\d+)"
)

# Statistics are taken after the drive/averaging origin used by the campaign
# reduction, so that the start-up transient is not counted as production.
WINDOW_START = 135.0


def parse(log_path: str) -> dict:
    rows: dict[str, list] = {}
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if "TBLE_REALIZABILITY" not in line:
                continue
            m = LINE.search(line)
            if not m:
                continue
            patch = m.group(1)
            rows.setdefault(patch, []).append([
                float(m.group(2)), int(m.group(3)), int(m.group(4)),
                float(m.group(5)), float(m.group(6)), float(m.group(7)),
                int(m.group(8)), int(m.group(9)), float(m.group(10)),
                int(m.group(11)),
            ])
    return {p: np.asarray(v, dtype=float) for p, v in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(
        RESULTS, "rswm_r23m6_ym_campaign_final"))
    ap.add_argument("--window-start", type=float, default=WINDOW_START)
    ap.add_argument("--out", default=os.path.join(RESULTS, "clipping_timeseries"))
    args = ap.parse_args()

    cases = []
    for root, _dirs, files in os.walk(args.bundle):
        if "log.pimpleFoam" in files:
            cases.append(root)
    cases.sort()

    summaries, blobs = [], {}
    for case_dir in cases:
        data = parse(os.path.join(case_dir, "log.pimpleFoam"))
        if not data:
            continue
        name = os.path.basename(case_dir)
        for patch, arr in data.items():
            t = arr[:, 0]
            keep = t >= args.window_start
            if keep.sum() < 5:
                keep = np.ones_like(t, dtype=bool)
            faces = arr[keep, 1]
            rec = dict(
                case=name, patch=patch,
                n_samples=int(keep.sum()),
                t_first=float(t[keep].min()), t_last=float(t[keep].max()),
                faces=int(faces[0]),
                frac_clipped_mean=float(np.mean(arr[keep, 3])),
                frac_clipped_min=float(np.min(arr[keep, 3])),
                frac_lower_clipped_mean=float(np.mean(arr[keep, 6] / faces)),
                frac_vector_capped_mean=float(np.mean(arr[keep, 7] / faces)),
                mismatch_per_face_mean=float(np.mean(arr[keep, 4] / faces)),
                mismatch_max=float(np.max(arr[keep, 5])),
                applied_traction_max=float(np.max(arr[keep, 8])),
                degenerate_roots_total=int(np.sum(arr[keep, 9])),
            )
            rec["mismatch_max_over_applied_max"] = float(
                rec["mismatch_max"] / max(rec["applied_traction_max"], 1e-30))
            summaries.append(rec)
            blobs[f"{name}__{patch}__t"] = t
            blobs[f"{name}__{patch}__frac_clipped"] = arr[:, 3]
            blobs[f"{name}__{patch}__mismatch_per_face"] = arr[:, 4] / arr[:, 1]
            print(f"  {name:<46s} {patch:<10s} n={rec['n_samples']:4d} "
                  f"clipped={rec['frac_clipped_mean']*100:6.2f}% "
                  f"(min {rec['frac_clipped_min']*100:6.2f}%) "
                  f"mismatch/face={rec['mismatch_per_face_mean']:.3e}", flush=True)

    if not summaries:
        print("no TBLE realizability records found")
        return 2

    bottom = [s for s in summaries if s["patch"] == "bottomWall"]
    out = dict(
        bundle=os.path.relpath(args.bundle, PROJECT),
        window_start=args.window_start,
        n_case_patches=len(summaries),
        bottom_wall_min_frac_clipped=min(s["frac_clipped_mean"] for s in bottom),
        bottom_wall_max_frac_clipped=max(s["frac_clipped_mean"] for s in bottom),
        records=summaries,
    )
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out + ".npz", **blobs)
    print(f"\nbottom wall: time-mean clipped fraction ranges "
          f"{out['bottom_wall_min_frac_clipped']*100:.2f}% to "
          f"{out['bottom_wall_max_frac_clipped']*100:.2f}% across "
          f"{len(bottom)} coupled runs")
    print(f"wrote {args.out}.npz and {args.out}_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
