#!/usr/bin/env python3
"""Per-member corrected 29-case hill-family sweep --- the clean dependency root.

The corrected family statistics exist only as summary numbers inside the
reference-rebase object, so the two family figures had no admissible root to be
redrawn from and kept rendering the superseded estimator's points while the
prose beside them had already been corrected.  This module publishes the
per-member corrected table as a first-class artifact, under the same array
names the figure generator already consumes, so the figure can be rebuilt by
changing its input rather than its code.

Source: the repaired book of the family sweep (curvature-aware cubic on six
fluid points), never the legacy book.  Nothing is recomputed here; the point is
to give the corrected numbers a citable home.  The legacy columns are carried
alongside, clearly named, because the paper reports the size and the direction
of the correction and must be able to reproduce both.

Outputs codes/results/corrected_family_sweep_l0_<stamp>.{npz,json}.
Run:  python3 codes/analysis/corrected_family_sweep_l0.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAMP = "20260825"
SWEEP = (ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/"
         "xiao29_epsilon_sweep.json")
OUT_NPZ = ROOT / "codes/results" / f"corrected_family_sweep_l0_{STAMP}.npz"
OUT_JSON = ROOT / "codes/results" / f"corrected_family_sweep_l0_{STAMP}.json"

MEASURES = ("eps_median", "r2", "rel_err", "f_rec", "f_sep", "L_sep",
            "frac_eps_lt_0p1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- statistics
def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, so tied levels do not depend on input order."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def boot_ci(a, b, draws=20000, seed=20260825):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.array([spearman(a[i], b[i])
                     for i in (rng.integers(0, n, n) for _ in range(draws))])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    S = json.loads(SWEEP.read_text())
    mem = S["per_member"]
    n = len(mem)

    alpha = np.array([m["alpha"] for m in mem], float)
    ell_p = np.array([m["ell_p"] for m in mem], float)
    L_y = np.array([m["L_y"] for m in mem], float)
    delta = 0.5 * L_y
    names = np.array([m["member"] for m in mem])
    has_ref = np.array([bool(m["has_independent_reference"]) for m in mem])
    ratio = np.array([m["repaired_over_legacy_eps"] for m in mem], float)

    book = {}
    for b in ("legacy", "repaired"):
        book[b] = {k: np.array([m[b][k] for m in mem], float) for k in MEASURES}

    g_rep = book["repaired"]["L_sep"] / delta
    g_leg = book["legacy"]["L_sep"] / delta
    eps = book["repaired"]["eps_median"]
    r2 = book["repaired"]["r2"]

    rho_eps = spearman(g_rep, eps)
    rho_r2 = spearman(g_rep, r2)
    lo, hi = boot_ci(g_rep, r2)

    payload = {
        # names the existing family figure already reads, so the generator
        # needs an input swap and not a rewrite
        "agg_cv_Lsep_over_delta": g_rep,
        "agg_eps_median": eps,
        "agg_r2": r2,
        "agg_cv_alpha": alpha,
        "collapse_Lsep_over_delta_rho_eps_median": np.float64(rho_eps),
        "collapse_Lsep_over_delta_rho_r2": np.float64(rho_r2),
        "collapse_Lsep_over_delta_rho_r2_ci_lo": np.float64(lo),
        "collapse_Lsep_over_delta_rho_r2_ci_hi": np.float64(hi),
        # provenance and the superseded book, for the size-of-correction claims
        "member": names,
        "has_independent_reference": has_ref,
        "ell_p": ell_p,
        "L_y": L_y,
        "repaired_over_legacy_eps": ratio,
        "legacy_cv_Lsep_over_delta": g_leg,
        "legacy_eps_median": book["legacy"]["eps_median"],
        "legacy_r2": book["legacy"]["r2"],
        "reference_book": np.str_("repaired_cubic6"),
    }
    np.savez(OUT_NPZ, **payload)

    summary = {
        "generated_by": "codes/analysis/corrected_family_sweep_l0.py",
        "reference_book": "repaired (curvature-aware cubic, six fluid points)",
        "superseded_book": "legacy (through-origin linear fit, four fluid points)",
        "n_members": n,
        "eps_range_repaired": [float(eps.min()), float(eps.max())],
        "eps_range_legacy": [float(book["legacy"]["eps_median"].min()),
                             float(book["legacy"]["eps_median"].max())],
        "r2_max_repaired": float(r2.max()),
        "r2_min_repaired": float(r2.min()),
        "every_member_fails_repaired": bool(np.all(r2 < 0)),
        "spearman_Lsep_over_delta": {
            "eps_median": rho_eps, "r2": rho_r2, "r2_ci95": [lo, hi]},
        "limitation_independent_reference": {
            "n_with_independent_wall_traction": int(has_ref.sum()),
            "members": [str(x) for x in names[has_ref]],
            "statement": (
                "only one of the 29 members has an independently published wall "
                "traction, and it is among the least affected by the correction, "
                "so the validation does not cover the strongly affected end of "
                "the family"),
        },
        "estimator_bias_correlates_with_reported_variables": {
            "rho(alpha, repaired_over_legacy_eps)": spearman(alpha, ratio),
            "rho(L_sep_over_delta_legacy, repaired_over_legacy_eps)":
                spearman(g_leg, ratio),
            "statement": (
                "the correction is largest for the longest-bubble, shallowest "
                "members, the direction that manufactures the reported trend, so "
                "part of the previously published correlation was a property of "
                "the estimator and not of the flow"),
        },
        "inputs": {str(SWEEP.relative_to(ROOT)): sha256(SWEEP)},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print(f"members                 : {n}")
    print(f"eps range (repaired)    : [{eps.min():.4f}, {eps.max():.4f}]")
    print(f"eps range (legacy)      : [{book['legacy']['eps_median'].min():.4f}, "
          f"{book['legacy']['eps_median'].max():.4f}]")
    print(f"R2 range (repaired)     : [{r2.min():.3f}, {r2.max():.3f}]  "
          f"all negative = {bool(np.all(r2 < 0))}")
    print(f"rho(L_sep/delta, eps)   : {rho_eps:+.4f}")
    print(f"rho(L_sep/delta, R2)    : {rho_r2:+.4f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"members with own ref    : {int(has_ref.sum())} of {n}")
    print(f"wrote {OUT_NPZ.relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
