#!/usr/bin/env python3
"""Corrected pooled certified floor — the certificate for class incompatibility.

DEFECT THIS FILE REPAIRS.  `input_sufficiency_bracket.py` normalises the pooled
traction by the *pooled* root mean square before computing the certified floor.
Across families that is wrong: the wavy-wall wall stress is several times larger
in magnitude than the hill wall stress, so in a pooled normalisation the wavy
stations become large outliers and inflate the floor for a reason that has
nothing to do with identifiability.  The pooled floor reported by that producer
at eta_m/delta = 0.05 must therefore not be read as evidence of class
incompatibility.

CORRECTED PROTOCOL.  Each case's traction is normalised by *its own* root mean
square before pooling, so a positive floor contribution means what it should:
two stations with nearly the same inputs carry different traction *relative to
their own case's scale*.  Two further honesty requirements are enforced here and
were missing before:

  (a) OVERLAP.  A pooled floor is meaningless if the classes do not occupy the
      same region of input space -- with no near neighbours across classes there
      are no cross-class pairs and the floor is simply the within-class floor.
      The overlap fraction is measured and printed.
  (b) MATCHED COUNTS.  The within-class and cross-class pools are subsampled to
      the same number of stations, and the cross-class pool is *balanced* so the
      minority families are not reduced to a handful of points by uniform
      thinning.

Output: codes/results/input_sufficiency_pooled_floor.json
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

N_POOL = 900                     # stations retained in every pool (matched)
L_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
SEED = 20260825


def own_scale_norm(case: dict) -> np.ndarray:
    return case["tau"] / (np.sqrt(np.mean(case["tau"] ** 2)) + 1e-300)


def stack(cases: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.vstack([np.column_stack((isb.slog(c["a"]), isb.slog(c["b"])))
                   for c in cases])
    t = np.concatenate([own_scale_norm(c) for c in cases])
    fam = np.concatenate([np.full(c["n_station"], i) for i, c in enumerate(cases)])
    return d, t, fam


def balanced_subsample(fam: np.ndarray, n_total: int, rng) -> np.ndarray:
    """Equal share per family where possible, filled from the largest family."""
    fams = np.unique(fam)
    per = max(1, n_total // len(fams))
    keep = []
    for f in fams:
        idx = np.flatnonzero(fam == f)
        take = min(per, len(idx))
        keep.append(rng.choice(idx, size=take, replace=False))
    keep = np.concatenate(keep)
    if len(keep) < n_total:
        rest = np.setdiff1d(np.flatnonzero(np.isin(fam, fams)), keep)
        extra = rng.choice(rest, size=min(n_total - len(keep), len(rest)),
                           replace=False)
        keep = np.concatenate([keep, extra])
    return np.sort(keep[:n_total])


def overlap_fraction(d_minor: np.ndarray, d_major: np.ndarray) -> float:
    """Share of minority points whose nearest major-class neighbour is within
    the major class's own 95th-percentile nearest-neighbour distance."""
    dm = np.sqrt(((d_major[:, None, :] - d_major[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dm, np.inf)
    scale = float(np.percentile(dm.min(1), 95))
    cross = np.sqrt(((d_minor[:, None, :] - d_major[None, :, :]) ** 2).sum(-1))
    return float(np.mean(cross.min(1) <= scale)), scale


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {"schema": "input_sufficiency_pooled_floor_v1",
           "repairs": "per-case traction normalisation; balanced pools; overlap test",
           "n_pool": N_POOL, "seed": SEED, "heights": {}}

    for frac in isb.ETA_FRACTIONS:
        hills, wavy, convdiv = isb.build_cases(frac)
        rec = {}

        # standardise the input metric ONCE, on the union, so the two pools are
        # measured with the same ruler
        d_all, _, _ = stack(hills + [wavy, convdiv])
        mu, sd = d_all.mean(0), d_all.std(0) + 1e-30

        pools = {"hills_only": hills, "cross_family": hills + [wavy, convdiv]}
        for label, cases in pools.items():
            d, t, fam = stack(cases)
            d = (d - mu) / sd
            idx = balanced_subsample(fam, N_POOL, np.random.default_rng(SEED))
            entry = {"n_used": int(len(idx)),
                     "n_families": int(len(np.unique(fam[idx])))}
            for L in L_GRID:
                f, pairs = isb.certified_floor(d[idx], t[idx], L)
                entry[f"L={L}"] = {"floor": f, "pairs": pairs}
            rec[label] = entry
            print(f"[eta{frac:.2f}] {label:13s} n={len(idx)} "
                  + " ".join(f"L={L}:{entry[f'L={L}']['floor']:.3f}"
                             for L in L_GRID), flush=True)

        d_h, _, _ = stack(hills)
        d_w, _, _ = stack([wavy])
        d_c, _, _ = stack([convdiv])
        d_h = (d_h - mu) / sd
        sub = d_h[np.linspace(0, len(d_h) - 1, 3000).astype(int)]
        ov_w, scale = overlap_fraction((d_w - mu) / sd, sub)
        ov_c, _ = overlap_fraction((d_c - mu) / sd, sub)
        rec["overlap"] = {"wavy_in_hill_cloud": ov_w, "convdiv_in_hill_cloud": ov_c,
                          "hill_nn_scale_95pct": scale}
        print(f"[eta{frac:.2f}] overlap: wavy {ov_w:.3f}, conv-div {ov_c:.3f}",
              flush=True)

        margins = {f"L={L}": rec["cross_family"][f"L={L}"]["floor"]
                             - rec["hills_only"][f"L={L}"]["floor"] for L in L_GRID}
        rec["margin_cross_minus_hills"] = margins
        rec["P3b_supported"] = bool(all(v > 0.02 for v in margins.values()))
        print(f"[eta{frac:.2f}] margins {  {k: round(v,3) for k,v in margins.items()} } "
              f"-> P3b {'SUPPORTED' if rec['P3b_supported'] else 'NOT SUPPORTED'}",
              flush=True)
        out["heights"][f"eta{frac:.2f}"] = rec

    dest = ROOT / "codes/results/input_sufficiency_pooled_floor.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
