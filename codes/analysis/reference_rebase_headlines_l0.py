#!/usr/bin/env python3
"""L0 producer: rebase every wall-traction-bearing headline of the periodic-hill
family onto the corrected reference set, and settle the two claims the audit
reopened.

Motivation
----------
The wall traction used to score the canonical hill was reconstructed from the
public 512x257 velocity archive by a through-origin LINEAR fit of its first four
points.  At that archive's wall spacing (dy = 0.0093-0.0136 H, first fluid points
at y+ 2.4-44) the fit under-resolves the wall gradient, so the reconstruction is
unusable AS A SCORING REFERENCE.  The archive itself is fine, and every
PREDICTION in this study is untouched: only the yardstick changes.

That yardstick, however, is in the denominator of the paper's headline a-priori
numbers.  This producer re-scores the SAME deposited predictions against

    A  the withdrawn linear-4 reconstruction        (negative control only)
    B  Peller & Manhart MGLET full-wall DNS         (primary)
    C  the same archive read with a through-origin cubic on six fluid points
                                                    (same-simulation bracket)
    K  Krank's deposited 10-station traction        (sparse independent check)

and reports four things:

  1. HEADLINES        every canonical-hill a-priori score, on all references,
                      with phase-block intervals, plus which of the paper's
                      qualitative statements survive the swap.
  2. EPSILON          the cancellation parameter under each traction estimator
                      at the paper's own matching surface.
  3. RANKING (R2-2)   the geometric-control ranking over the 29-case family,
                      rebuilt with the family's own knobs included as
                      candidates, on legacy AND corrected estimators, with
                      permutation p-values and partial correlations.
  4. NULL (Lambda)    whether the fitted amplification law adds information
                      beyond a shared-denominator magnitude ratio.

Nothing here re-runs a simulation and nothing here changes a prediction.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/reference_rebase_headlines_l0.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes/analysis"))

STAMP = "20260825"
OUT_JSON = ROOT / "codes/results" / f"reference_rebase_headlines_l0_{STAMP}.json"
OUT_NPZ = ROOT / "codes/results" / f"reference_rebase_headlines_l0_{STAMP}.npz"

ARCHIVE = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
DIAG = ROOT / "codes/results/diagnostic_test_corrected.npz"
MGLET = (ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/"
         "UFR3-30_data-NP-Re5600-DNS2-11.dat")
KRANK = ROOT / "codes/raw_data/geometry_driven/krank_pehill_Re5600_wall_profiles.npz"
SWEEP = (ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/"
         "xiao29_epsilon_sweep.json")
LADDER = ROOT / "codes/results" / f"conditioning_ladder_l0_{STAMP}.json"
LADDER_NPZ = ROOT / "codes/results" / f"conditioning_ladder_l0_{STAMP}.npz"
DOSE = ROOT / "codes/results/dose_response_xiao.npz"
CLOSURE_FLOOR = ROOT / "codes/results/closure_conditioning_floor.npz"

LX = 9.0
NU = 1.0 / 5600.0
Y_IDX = 10                 # the paper's a-priori matching index
REPAIRED_DEG, REPAIRED_K = 3, 6      # audit-selected repaired estimator
BLOCK = 64                 # Lx/8 in stations of the 512-station archive
DRAWS = 20000
SEED = 20260825

# published values this producer must reproduce exactly on reference A
PUBLISHED = {
    "pg_ode_r2": -47.68617253416459,
    "exact_stress_r2": -482.97683571,
    "eps_median": 0.08364189563744982,
    "eps_frac_below_0p1": 0.564453125,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wall_tangent(x):
    """Downstream unit wall tangent at the archive stations.

    Delegates to the study's committed surface description so that this
    producer and the coupled/ladder analyses use one and the same wall.
    """
    import r2m4_ladder_common as _C
    _h, slope, tx, ty = _C.wall_tangent(np.asarray(x, float))
    return tx, ty, slope


def wrap_interp(src_phase, src_val, tgt_phase):
    """Periodic linear interpolation onto target phases in [0, 1)."""
    o = np.argsort(src_phase)
    p, v = np.asarray(src_phase, float)[o], np.asarray(src_val, float)[o]
    p_ext = np.concatenate([p - 1.0, p, p + 1.0])
    v_ext = np.concatenate([v, v, v])
    return np.interp(np.mod(tgt_phase, 1.0), p_ext, v_ext)


def poly_origin_slope(n, u, deg):
    A = np.vstack([np.asarray(n, float) ** (k + 1) for k in range(deg)]).T
    c, *_ = np.linalg.lstsq(A, np.asarray(u, float), rcond=None)
    return float(c[0])


def r2(pred, truth):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def rel_rms(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)) /
                 np.sqrt(np.mean(truth ** 2)))


def zero_crossings(x, y):
    """x locations where y changes sign, by linear interpolation."""
    out = []
    for i in range(len(y) - 1):
        if y[i] == 0.0:
            out.append(float(x[i]))
        elif y[i] * y[i + 1] < 0.0:
            t = y[i] / (y[i] - y[i + 1])
            out.append(float(x[i] + t * (x[i + 1] - x[i])))
    return out


def block_ci(fn, n, block=BLOCK, draws=DRAWS, seed=SEED):
    """Moving-block bootstrap 95% interval of fn(index_array)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nb))
    vals = np.empty(draws)
    offs = np.arange(block)
    for d in range(draws):
        idx = np.mod(starts[d][:, None] + offs[None, :], n).ravel()[:n]
        vals[d] = fn(idx)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# --------------------------------------------------------------------------- #
# rank statistics
# --------------------------------------------------------------------------- #
def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for g in np.flatnonzero(counts > 1):
        m = inv == g
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / den) if den > 0 else 0.0


def perm_p(a, b, draws=20000, seed=SEED):
    """Two-sided permutation p-value for Spearman rho."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(a, b))
    b = np.asarray(b, float)
    cnt = 0
    for _ in range(draws):
        if abs(spearman(a, rng.permutation(b))) >= obs - 1e-15:
            cnt += 1
    return float((cnt + 1) / (draws + 1))


def member_bootstrap_ci(a, b, draws=4000, seed=SEED):
    """Percentile bootstrap over family members for a Spearman rho."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    vals = np.empty(draws)
    for d in range(draws):
        idx = rng.integers(0, n, n)
        if np.all(a[idx] == a[idx][0]) or np.all(b[idx] == b[idx][0]):
            vals[d] = np.nan
            continue
        vals[d] = spearman(a[idx], b[idx])
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def partial_spearman(a, b, c):
    """Spearman correlation of a and b with the rank-linear effect of c removed."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)

    def resid(v):
        v = v - v.mean()
        z = rc - rc.mean()
        return v - (np.dot(v, z) / np.dot(z, z)) * z

    va, vb = resid(ra), resid(rb)
    den = np.sqrt(np.sum(va ** 2) * np.sum(vb ** 2))
    return float(np.sum(va * vb) / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# PART 1 — references and canonical-hill headline rebase
# --------------------------------------------------------------------------- #
def build_references():
    d = np.load(ARCHIVE)
    x = np.asarray(d["x"], float)
    y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float)
    V = np.asarray(d["V"], float)
    phase = np.mod(x / LX, 1.0)
    tx, ty, _ = wall_tangent(x)

    refs = {}
    # A — the withdrawn deposit (negative control)
    refs["A_withdrawn_linear4"] = np.asarray(d["tau_w"], float)

    # C — repaired same-simulation estimator: through-origin cubic, 6 fluid points
    tau_c = np.empty(x.size)
    for i in range(x.size):
        ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])
        yy, uu, vv = y[i, ok], U[i, ok], V[i, ok]
        off = yy[1:REPAIRED_K + 1] - yy[0]
        ut = uu[1:REPAIRED_K + 1] * tx[i] + vv[1:REPAIRED_K + 1] * ty[i]
        tau_c[i] = NU * poly_origin_slope(off, ut, REPAIRED_DEG) / tx[i]
    refs["C_repaired_cubic6"] = tau_c

    # B — MGLET deposited full-wall traction (strip the two plot-axis rows)
    raw = np.loadtxt(MGLET)
    tail = raw[-2:]
    if not (np.allclose(tail[0], [0.0, 0.0, 0.0]) and
            np.allclose(tail[1], [9.0, 0.0, 0.0])):
        raise RuntimeError("MGLET placeholder rows moved — re-audit before using")
    body = raw[:-2]
    refs["B_mglet"] = wrap_interp(np.mod(body[:, 0] / LX, 1.0), body[:, 1], phase)

    # K — Krank 10 deposited stations (sparse; kept separately, not interpolated)
    kd = np.load(KRANK, allow_pickle=True)
    krank = (np.asarray(kd["x"], float), np.asarray(kd["tau_w"], float))
    return d, x, phase, tx, ty, refs, krank, body.shape[0]


def score_prediction(name, pred, x, phase, refs, krank):
    out = {}
    for rname, truth in refs.items():
        err = pred - truth
        scale = float(np.sqrt(np.mean(truth ** 2)))
        rec = {
            "r2": r2(pred, truth),
            "rel_rms": rel_rms(pred, truth),
            "reference_rms": scale,
            "sign_mismatch_fraction": float(np.mean(np.sign(pred) != np.sign(truth))),
            "station_abs_p50_over_ref_rms": float(np.percentile(np.abs(err), 50) / scale),
            "station_abs_p95_over_ref_rms": float(np.percentile(np.abs(err), 95) / scale),
            "station_abs_max_over_ref_rms": float(np.max(np.abs(err)) / scale),
            "correlation_with_reference": float(np.corrcoef(pred, truth)[0, 1]),
            "reference_zero_crossings_x": zero_crossings(x, truth)[:6],
        }
        n = pred.size
        rec["r2_ci"] = block_ci(lambda idx: r2(pred[idx], truth[idx]), n)
        rec["rel_rms_ci"] = block_ci(lambda idx: rel_rms(pred[idx], truth[idx]), n)
        out[rname] = rec
    # K: station-restricted, no bootstrap claimed
    kx, ktau = krank
    kpred = wrap_interp(phase, pred, np.mod(kx / LX, 1.0))
    out["K_krank_10_stations"] = {
        "n_stations": int(kx.size),
        "r2": r2(kpred, ktau),
        "rel_rms": rel_rms(kpred, ktau),
        "reference_rms": float(np.sqrt(np.mean(ktau ** 2))),
        "note": "sparse independent cross-check; station-restricted, no interval claimed",
    }
    return out


def part1_headlines():
    d, x, phase, tx, ty, refs, krank, n_mglet = build_references()
    diag = np.load(DIAG, allow_pickle=True)

    preds = {
        "pg_ode_mixing_length": np.asarray(diag["standard_ml"], float),
        "pg_ode_exact_dns_stress": np.asarray(diag["controlled_dns"], float),
    }
    scored = {k: score_prediction(k, v, x, phase, refs, krank) for k, v in preds.items()}

    # instrument fidelity: reference A must reproduce the published headlines
    fid = {
        "pg_ode_r2_on_A": scored["pg_ode_mixing_length"]["A_withdrawn_linear4"]["r2"],
        "pg_ode_r2_published": PUBLISHED["pg_ode_r2"],
        "pg_ode_abs_deviation": abs(scored["pg_ode_mixing_length"]
                                    ["A_withdrawn_linear4"]["r2"] - PUBLISHED["pg_ode_r2"]),
        "exact_stress_r2_on_A": scored["pg_ode_exact_dns_stress"]["A_withdrawn_linear4"]["r2"],
        "exact_stress_r2_published": PUBLISHED["exact_stress_r2"],
        "exact_stress_abs_deviation": abs(scored["pg_ode_exact_dns_stress"]
                                          ["A_withdrawn_linear4"]["r2"]
                                          - PUBLISHED["exact_stress_r2"]),
    }

    # reference-to-reference distance, in units of the primary reference RMS
    B, Cc, A = refs["B_mglet"], refs["C_repaired_cubic6"], refs["A_withdrawn_linear4"]
    rms_B = float(np.sqrt(np.mean(B ** 2)))
    r2r = {
        "B_to_C_over_B_rms": float(np.sqrt(np.mean((B - Cc) ** 2)) / rms_B),
        "B_to_A_over_B_rms": float(np.sqrt(np.mean((B - A) ** 2)) / rms_B),
        "rms_ratio_A_over_B": float(np.sqrt(np.mean(A ** 2)) / rms_B),
        "rms_ratio_C_over_B": float(np.sqrt(np.mean(Cc ** 2)) / rms_B),
    }

    # which qualitative statements survive the reference swap?
    ml = scored["pg_ode_mixing_length"]
    ex = scored["pg_ode_exact_dns_stress"]
    survives = {
        "every_reference_gives_negative_r2_for_the_pg_ode": bool(
            all(ml[r]["r2"] < 0 for r in ("A_withdrawn_linear4", "B_mglet",
                                          "C_repaired_cubic6"))),
        "exact_dns_stress_is_worse_than_mixing_length_on_every_reference": bool(
            all(ex[r]["r2"] < ml[r]["r2"] for r in ("A_withdrawn_linear4", "B_mglet",
                                                    "C_repaired_cubic6"))),
        "r2_magnitude_is_reference_dependent_factor": float(
            ml["A_withdrawn_linear4"]["r2"] / ml["B_mglet"]["r2"]),
        "rel_rms_is_reference_dependent_factor": float(
            ml["A_withdrawn_linear4"]["rel_rms"] / ml["B_mglet"]["rel_rms"]),
        "sign_mismatch_moves_by": float(
            ml["A_withdrawn_linear4"]["sign_mismatch_fraction"]
            - ml["B_mglet"]["sign_mismatch_fraction"]),
    }
    # the five-closure family: the same rebase applied to the closure-independence
    # instrument, whose predictions live on the identical 512 stations
    cf = np.load(CLOSURE_FLOOR, allow_pickle=True)
    if not np.array_equal(np.asarray(cf["hills_tau_true"], float), refs["A_withdrawn_linear4"]):
        raise RuntimeError("the closure-family instrument is not on the archive stations")
    family = {}
    for key in [str(k) for k in cf["closure_keys"]]:
        p = np.asarray(cf[f"hills_pred_{key}"], float)
        family[key] = {rn: {"r2": r2(p, t), "rel_rms": rel_rms(p, t)}
                       for rn, t in refs.items()}
    order_ok = {}
    for rn in refs:
        scores = {k: family[k][rn]["r2"] for k in family}
        worst = min(scores, key=lambda k: scores[k])
        order_ok[rn] = {
            "all_negative": bool(all(v < 0 for v in scores.values())),
            "worst_closure": worst,
            "exact_stress_is_worst": bool(worst == "E_dns_stress"),
            "range": [float(min(scores.values())), float(max(scores.values()))],
        }

    return {
        "predictions_are_unchanged": ("every prediction array is read verbatim "
                                      "from its deposited a-priori artifact; only "
                                      "the scoring reference varies"),
        "closure_family": {"per_closure": family, "per_reference": order_ok},
        "n_stations": int(x.size),
        "n_mglet_rows_used": int(n_mglet),
        "instrument_fidelity": fid,
        "reference_to_reference": r2r,
        "scores": scored,
        "survives_reference_swap": survives,
    }, refs, x, phase, d


# --------------------------------------------------------------------------- #
# PART 2 — epsilon under each traction estimator
# --------------------------------------------------------------------------- #
def part2_epsilon(refs, d, x):
    y = np.asarray(d["y"], float)
    dpdx = np.asarray(d["dp_dx"], float)
    y_m = y[:, Y_IDX] - y[:, 0]
    den = np.maximum(np.abs(dpdx) * y_m, 1e-30)

    # N1: tangent-projected through-origin LINEAR fit on four fluid points —
    # the estimator the audit characterised, one rung better than the deposit.
    U, V = np.asarray(d["U"], float), np.asarray(d["V"], float)
    tx, ty, _ = wall_tangent(x)
    tau_n1 = np.empty(x.size)
    for i in range(x.size):
        ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])
        yy, uu, vv = y[i, ok], U[i, ok], V[i, ok]
        off = yy[1:5] - yy[0]
        ut = uu[1:5] * tx[i] + vv[1:5] * ty[i]
        tau_n1[i] = NU * poly_origin_slope(off, ut, 1) / tx[i]

    estimators = {
        "N0_archive_deposit_as_published": refs["A_withdrawn_linear4"],
        "N1_tangent_linear4": tau_n1,
        "N2_repaired_cubic6": refs["C_repaired_cubic6"],
        "N3_mglet_deposited": refs["B_mglet"],
    }
    out = {}
    for name, tau in estimators.items():
        eps = np.abs(tau) / den
        med = float(np.median(eps))
        out[name] = {
            "median": med,
            "median_ci": block_ci(lambda idx: float(np.median(eps[idx])), eps.size),
            "frac_below_0p1": float(np.mean(eps < 0.1)),
            "frac_below_1": float(np.mean(eps < 1.0)),
            "geometric_mean": float(np.exp(np.mean(np.log(np.maximum(eps, 1e-30))))),
            "factor_vs_published": med / PUBLISHED["eps_median"],
        }
    out["_reproduction_of_published"] = {
        "median": out["N0_archive_deposit_as_published"]["median"],
        "published": PUBLISHED["eps_median"],
        "frac_below_0p1": out["N0_archive_deposit_as_published"]["frac_below_0p1"],
        "frac_below_0p1_published": PUBLISHED["eps_frac_below_0p1"],
        "exact": bool(out["N0_archive_deposit_as_published"]["median"]
                      == PUBLISHED["eps_median"]),
    }
    out["_matching_surface"] = {
        "y_index": Y_IDX,
        "median_y_m_over_H": float(np.median(y_m)),
        "note": ("epsilon magnitudes are convention-bound: they scale with the "
                 "matching offset y_m of whichever archive is read, so only "
                 "orderings on ONE fixed surface are comparable"),
    }
    return out


# --------------------------------------------------------------------------- #
# PART 3 — the reopened ranking (R2-2)
# --------------------------------------------------------------------------- #
MEASURES = ("eps_median", "frac_eps_lt_0p1", "r2", "rel_err")


def part3_ranking():
    S = json.loads(SWEEP.read_text())
    mem = S["per_member"]
    n = len(mem)

    alpha = np.array([m["alpha"] for m in mem], float)
    ell_p = np.array([m["ell_p"] for m in mem], float)
    L_y = np.array([m["L_y"] for m in mem], float)
    delta = 0.5 * L_y

    def candidates(book):
        L_sep = np.array([m[book]["L_sep"] for m in mem], float)
        return {
            # the five groups the paper ranked
            "alpha": alpha,
            "h_over_Lx": 1.0 / ell_p,
            "ell_p_over_delta": ell_p / delta,
            "L_sep_over_delta": L_sep / delta,
            "f_rec": np.array([m[book]["f_rec"] for m in mem], float),
            # the family's own knobs, previously not offered as candidates
            "L_y": L_y,
            "h_over_delta": 1.0 / delta,          # hill height is unity in this family
            "ell_p": ell_p,
            "L_sep": L_sep,
        }

    result = {}
    for book in ("legacy", "repaired"):
        meas = {k: np.array([m[book][k] for m in mem], float) for k in MEASURES}
        cands = candidates(book)
        rows = {}
        for cname, g in cands.items():
            rho = {k: spearman(g, meas[k]) for k in MEASURES}
            # Coherence.  A genuine control of the cancellation must push the
            # four measures the way the mechanism says: a candidate that raises
            # the median cancellation must LOWER the fraction of deeply
            # cancelling stations, RAISE the skill and LOWER the relative
            # error.  frac(eps<0.1) and rel_err are therefore inversely
            # oriented with respect to eps_median and R^2.
            s = {k: np.sign(rho[k]) for k in MEASURES}
            coherent = bool(s["r2"] != 0 and
                            s["eps_median"] == s["r2"] and
                            s["frac_eps_lt_0p1"] == -s["r2"] and
                            s["rel_err"] == -s["r2"])
            rows[cname] = {
                "rho": rho,
                "p_rho_r2": perm_p(g, meas["r2"]),
                "abs_rho_r2": abs(rho["r2"]),
                "rho_r2_ci": member_bootstrap_ci(g, meas["r2"]),
                "min_abs_rho_over_measures": float(min(abs(v) for v in rho.values())),
                "sign_coherent_across_four_measures": coherent,
            }
        ranked = sorted(rows, key=lambda c: -rows[c]["abs_rho_r2"])
        coherent_ranked = [c for c in ranked if rows[c]["sign_coherent_across_four_measures"]]
        result[book] = {
            "per_candidate": rows,
            "ranked_by_abs_rho_r2": ranked,
            "sign_coherent_candidates_ranked": coherent_ranked,
            "strongest_overall": ranked[0],
            "strongest_sign_coherent": coherent_ranked[0] if coherent_ranked else None,
            "published_claim_only_L_sep_over_delta_is_strongest": bool(
                coherent_ranked and coherent_ranked[0] == "L_sep_over_delta"),
        }

    # does L_sep/delta add anything once the outer scale is controlled?
    part = {}
    for book in ("legacy", "repaired"):
        meas_r2 = np.array([m[book]["r2"] for m in mem], float)
        L_sep = np.array([m[book]["L_sep"] for m in mem], float)
        part[book] = {
            "rho(L_sep_over_delta, r2)": spearman(L_sep / delta, meas_r2),
            "rho(L_y, r2)": spearman(L_y, meas_r2),
            "rho(L_sep_raw, r2)": spearman(L_sep, meas_r2),
            "p(L_sep_raw, r2)": perm_p(L_sep, meas_r2),
            "partial_rho(L_sep_over_delta, r2 | L_y)": partial_spearman(
                L_sep / delta, meas_r2, L_y),
            "partial_rho(L_y, r2 | L_sep_over_delta)": partial_spearman(
                L_y, meas_r2, L_sep / delta),
            "partial_rho(L_sep_raw, r2 | L_y)": partial_spearman(L_sep, meas_r2, L_y),
            "rho(L_y, L_sep_over_delta)": spearman(L_y, L_sep / delta),
            "note": ("L_sep/delta and L_y are algebraically linked (delta = L_y/2), "
                     "so these partials separate a genuine bubble-length effect "
                     "from its denominator; the raw L_sep column is the "
                     "denominator-free version of the same question"),
        }

    # estimator bias vs the reported variables (the candour point)
    ratio = np.array([m["repaired_over_legacy_eps"] for m in mem], float)
    L_sep_leg = np.array([m["legacy"]["L_sep"] for m in mem], float)
    bias = {
        "rho(alpha, repaired_over_legacy_eps)": spearman(alpha, ratio),
        "rho(L_sep_over_delta_legacy, repaired_over_legacy_eps)":
            spearman(L_sep_leg / delta, ratio),
        "rho(median_dy_first, repaired_over_legacy_eps)":
            spearman(np.array([m["median_dy_first"] for m in mem], float), ratio),
        "interpretation": ("the estimator correction is largest exactly for the "
                           "longest-bubble, shallowest members, i.e. in the "
                           "direction that manufactures the published trend"),
    }

    eps_leg = np.array([m["legacy"]["eps_median"] for m in mem], float)
    eps_rep = np.array([m["repaired"]["eps_median"] for m in mem], float)
    r2_leg = np.array([m["legacy"]["r2"] for m in mem], float)
    r2_rep = np.array([m["repaired"]["r2"] for m in mem], float)
    return {
        "n_members": n,
        "members_with_independent_wall_traction": S["members_with_independent_wall_traction_reference"],
        "n_members_without_any_independent_reference": len(
            S["members_without_any_independent_reference"]),
        "reference_limitation": S["reference_limitation"],
        "ranking": result,
        "partial_correlations": part,
        "estimator_bias_correlates_with_reported_variables": bias,
        "every_member_fails": {
            "legacy_max_r2": float(r2_leg.max()),
            "repaired_max_r2": float(r2_rep.max()),
            "all_negative_legacy": bool(np.all(r2_leg < 0)),
            "all_negative_repaired": bool(np.all(r2_rep < 0)),
        },
        "eps_range": {
            "legacy": [float(eps_leg.min()), float(eps_leg.max())],
            "repaired": [float(eps_rep.min()), float(eps_rep.max())],
            "legacy_argmin_member": mem[int(np.argmin(eps_leg))]["member"],
            "repaired_argmin_member": mem[int(np.argmin(eps_rep))]["member"],
        },
        "headline_rho_r2_L_sep_over_delta": {
            "legacy": spearman(np.array([m["legacy"]["L_sep"] for m in mem], float) / delta,
                               r2_leg),
            "repaired": spearman(np.array([m["repaired"]["L_sep"] for m in mem], float) / delta,
                                 r2_rep),
        },
    }


# --------------------------------------------------------------------------- #
# PART 4 — is the amplification law more than a magnitude ratio?
# --------------------------------------------------------------------------- #
def part4_null():
    J = json.loads(LADDER.read_text())
    Z = np.load(LADDER_NPZ, allow_pickle=True)
    delta_fit = float(J["amplification_bound"]["delta_fitted_over_all_points"])

    ref_of = {"B_mglet": ("reference_B_phase", "reference_B_tau"),
              "C_xiao_repaired_cubic6": ("reference_C_phase", "reference_C_tau")}

    # Every quantity below is recomputed on ONE grid — the surface's own phase
    # stations — so that the law, the null and the error are strictly
    # comparable.  The deposited predictor is carried alongside for fidelity.
    raw = []
    for pt in J["amplification_bound"]["points"]:
        surf, refname = pt["surface"], pt["reference"]
        pred = np.asarray(Z[f"{surf}_pred_Xfull_all_transport_plus_exact_shear_stress"], float)
        phase = np.asarray(Z[f"{surf}_phase"], float)
        rp, rt = ref_of[refname]
        truth = wrap_interp(np.asarray(Z[rp], float), np.asarray(Z[rt], float), phase)
        assembled = np.asarray(Z[f"{surf}_impulse_S_abs_plus_tau_ym"], float)
        rms_truth = float(np.sqrt(np.mean(truth ** 2)))
        # second quadrature convention: a dense uniform phase grid, the
        # convention the deposited ladder metric uses
        dense = np.linspace(0.0, 1.0, 4096, endpoint=False)
        pred_d = wrap_interp(phase, pred, dense)
        truth_d = wrap_interp(np.asarray(Z[rp], float), np.asarray(Z[rt], float), dense)
        asm_d = wrap_interp(phase, assembled, dense)
        rms_truth_d = float(np.sqrt(np.mean(truth_d ** 2)))
        raw.append({
            "surface": surf, "reference": refname,
            "E": rel_rms(pred, truth),
            "Lambda_tau": float(np.sqrt(np.mean(assembled ** 2)) / rms_truth),
            "Lambda_tau_as_deposited": float(pt["predictor"]),
            "null": float(np.sqrt(np.mean(pred ** 2)) / rms_truth),
            "corr": float(np.corrcoef(pred, truth)[0, 1]),
            "attain": float(np.sqrt(np.mean((pred - truth) ** 2)) /
                            np.sqrt(np.mean(assembled ** 2))),
            "E_dense": rel_rms(pred_d, truth_d),
            "Lambda_tau_dense": float(np.sqrt(np.mean(asm_d ** 2)) / rms_truth_d),
            "null_dense": float(np.sqrt(np.mean(pred_d ** 2)) / rms_truth_d),
        })

    # refit the law's single free parameter on this grid, least squares through
    # the origin, so the comparison does not penalise it for a stale constant
    E_all = np.array([r["E"] for r in raw])
    L_all = np.array([r["Lambda_tau"] for r in raw])
    delta_here = float(np.dot(L_all, E_all) / np.dot(L_all, L_all))

    # the same contest under the dense-grid quadrature convention
    E_d = np.array([r["E_dense"] for r in raw])
    L_d = np.array([r["Lambda_tau_dense"] for r in raw])
    N_d = np.array([r["null_dense"] for r in raw])
    delta_d = float(np.dot(L_d, E_d) / np.dot(L_d, L_d))
    fit_err_d = np.abs(delta_d * L_d - E_d) / E_d
    null_err_d = np.abs(N_d - E_d) / E_d
    convention = {
        "station_grid": {
            "note": "each surface scored on its own phase stations",
        },
        "dense_uniform_grid": {
            "note": "both fields interpolated to 4096 uniform phases",
            "delta_refitted": delta_d,
            "fitted_law_mean_relative_error": float(fit_err_d.mean()),
            "null_mean_relative_error": float(null_err_d.mean()),
            "fitted_law_beats_null": bool(fit_err_d.mean() < null_err_d.mean()),
        },
        "E_convention_sensitivity_max_relative": float(
            np.max(np.abs(E_d - E_all) / E_all)),
    }

    rows = []
    for r in raw:
        fitted = delta_here * r["Lambda_tau"]
        rows.append({
            "surface": r["surface"],
            "reference": r["reference"],
            "measured_E": r["E"],
            "Lambda_tau_on_this_grid": r["Lambda_tau"],
            "Lambda_tau_as_deposited": r["Lambda_tau_as_deposited"],
            "fitted_law_prediction": fitted,
            "fitted_law_relative_error": abs(fitted - r["E"]) / r["E"],
            "shared_denominator_null": r["null"],
            "null_relative_error": abs(r["null"] - r["E"]) / r["E"],
            "correlation_prediction_with_truth": r["corr"],
            "attainment_error_over_assembled_magnitude": r["attain"],
        })

    fit_err = np.array([r["fitted_law_relative_error"] for r in rows])
    null_err = np.array([r["null_relative_error"] for r in rows])
    return {
        "question": ("does E = delta * Lambda_tau carry information beyond the "
                     "trivial magnitude ratio RMS(prediction)/RMS(truth), which "
                     "shares its denominator with the error metric?"),
        "points": rows,
        "quadrature_convention": convention,
        "delta_refitted_on_this_grid": delta_here,
        "delta_as_deposited": delta_fit,
        "mean_absolute_correlation_prediction_with_truth": float(
            np.mean([abs(r["correlation_prediction_with_truth"]) for r in rows])),
        "attainment_range": [float(min(r["attainment_error_over_assembled_magnitude"]
                                       for r in rows)),
                             float(max(r["attainment_error_over_assembled_magnitude"]
                                       for r in rows))],
        "fitted_law_worst_relative_error": float(fit_err.max()),
        "fitted_law_mean_relative_error": float(fit_err.mean()),
        "null_worst_relative_error": float(null_err.max()),
        "null_mean_relative_error": float(null_err.mean()),
        "null_free_parameters": 0,
        "fitted_law_free_parameters": 1,
        "fitted_law_beats_null": bool(fit_err.mean() < null_err.mean()),
        "verdict": None,          # filled below
        "denominator_free_alternative": (
            "attainment = RMS(prediction - truth)/RMS(sum_j|I_j| + |tau(y_m)|) "
            "shares no factor with the error metric's denominator and therefore "
            "is not exposed to this null"),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    print("[1/4] rebasing canonical-hill headlines ...", flush=True)
    headlines, refs, x, phase, d = part1_headlines()

    print("[2/4] epsilon under each traction estimator ...", flush=True)
    eps = part2_epsilon(refs, d, x)

    print("[3/4] rebuilding the geometric-control ranking (R2-2) ...", flush=True)
    ranking = part3_ranking()

    print("[4/4] amplification-law null test ...", flush=True)
    null = part4_null()
    conv = null["quadrature_convention"]
    both = null["fitted_law_beats_null"] and conv["dense_uniform_grid"]["fitted_law_beats_null"]
    # the margin must also exceed the sensitivity of the error metric itself to
    # the quadrature convention, or the contest is below the instrument's noise
    margin = null["null_mean_relative_error"] - null["fitted_law_mean_relative_error"]
    resolved = bool(both and margin > conv["E_convention_sensitivity_max_relative"])
    if resolved:
        null["verdict"] = "FITTED_LAW_ADDS_INFORMATION"
    elif both:
        null["verdict"] = "FITTED_LAW_BEATS_NULL_BUT_BY_LESS_THAN_THE_CONVENTION_SENSITIVITY"
    else:
        null["verdict"] = "FITTED_LAW_DOES_NOT_BEAT_A_ZERO_PARAMETER_NULL"
    null["margin_over_null_mean"] = float(margin)

    out = {
        "schema": "reference-rebase-headlines-l0-v1",
        "node": "L0 attempt 1 (node_000), 2026-08-25",
        "question": ("which of the paper's wall-traction-bearing claims are "
                     "properties of the flow, and which were properties of the "
                     "withdrawn scoring reference?"),
        "headlines": headlines,
        "epsilon": eps,
        "ranking_r2_2": ranking,
        "amplification_null": null,
        "inputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ARCHIVE, DIAG, MGLET, KRANK, SWEEP, LADDER, LADDER_NPZ,
                      CLOSURE_FLOOR)
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, sort_keys=True))

    np.savez_compressed(
        OUT_NPZ,
        x=x, phase=phase,
        **{f"reference_{k}": v for k, v in refs.items()},
        pred_pg_ode=np.asarray(np.load(DIAG)["standard_ml"], float),
        pred_exact_stress=np.asarray(np.load(DIAG)["controlled_dns"], float),
    )
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_NPZ.relative_to(ROOT)}")

    ml = headlines["scores"]["pg_ode_mixing_length"]
    print("\ncanonical hill, pressure-gradient ODE:")
    for r in ("A_withdrawn_linear4", "B_mglet", "C_repaired_cubic6"):
        print(f"  {r:24s} R2 = {ml[r]['r2']:10.3f}   relRMS = {ml[r]['rel_rms']:7.3f}")
    print(f"\nepsilon median: published {eps['N0_archive_deposit_as_published']['median']:.5f}"
          f" -> MGLET {eps['N3_mglet_deposited']['median']:.5f}"
          f"  (frac<0.1: {eps['N0_archive_deposit_as_published']['frac_below_0p1']:.3f}"
          f" -> {eps['N3_mglet_deposited']['frac_below_0p1']:.3f})")
    for book in ("legacy", "repaired"):
        rr = ranking["ranking"][book]
        print(f"ranking [{book}]: strongest overall = {rr['strongest_overall']}, "
              f"strongest sign-coherent = {rr['strongest_sign_coherent']}")
    print(f"amplification: {null['verdict']} "
          f"(fitted mean {null['fitted_law_mean_relative_error']:.4f} vs "
          f"null mean {null['null_mean_relative_error']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
