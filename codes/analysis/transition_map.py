#!/usr/bin/env python3
r"""
transition_map.py
=================
PILLAR B / Bind B-L1-4 : construct the ODE-wall-model FAILURE TRANSITION MAP in
the engineer-readable geometry plane (amplitude a/delta  x  pitch lambda/delta),
and demonstrate that a SINGLE shape-agnostic order parameter -- the
blockage-weighted cancellation depth eps -- governs the transition boundary
across all wall shapes (smooth sinusoidal hills, low-amplitude wavy walls, sharp
square ribs, and the zero-frequency BFS limit).

WHAT THIS SCRIPT DEFINES (the L1 methodology)
---------------------------------------------
1. AXES.  Every repeating geometry is placed at
       a/delta      = (peak wall displacement amplitude) / (outer scale delta),
       lambda/delta = (streamwise repeat pitch)          / (outer scale delta),
   with delta the channel half-height (hill/wavy/rib) or boundary-layer thickness
   proxy.  The non-repeating backward-facing step is the lambda/delta -> infinity
   anchor (spatial frequency -> 0).

2. PER-POINT METRIC.  At each geometry point we run the FROZEN a-priori ODE/TBLE
   wall model (Y_IDX = 10 matching cell, identical to the rest of the paper) and
   record three numbers, all a-priori-readable:
       eps_median      = median_x |tau_w| / (|dp/dx| y_m)   (cancellation depth),
       frac(eps<0.1)   = wall fraction in the residual regime,
       R^2(tau_w)      = ODE wall-stress skill            (the FAILURE measure).
   A point is TOLERATED if R^2 >= 0.88, MARGINAL if 0 <= R^2 < 0.88, FAILED if
   R^2 < 0.

3. TRANSITION BOUNDARY.  The boundary is NOT drawn by eye.  We test whether the
   single order parameter eps_median separates failed from non-failed points
   irrespective of wall shape (the shape-agnostic claim):
     - ROC AUC of (-eps_median) as a binary fail/not-fail classifier,
     - eps* = the eps_median threshold maximising balanced accuracy, with a
       deterministic bootstrap CI,
     - leave-one-geometry-class-out (LOGO): refit eps* with each shape class held
       out, to show the threshold is not carried by one family.
   In the (a/delta, lambda/delta) plane the boundary is then the level set
   eps_median(a/delta, lambda/delta) = eps*, reported as the family of points
   straddling eps*.

4. SPARSITY / HETEROGENEITY.  Points come from different Re, codes and fidelities
   (DNS, LES, coarse RANS).  We therefore (i) never compare absolute tau_w across
   sets -- only the dimensionless, within-set R^2 and eps; (ii) label each point
   by provenance (DNS / OpenFOAM-RANS); (iii) report the collapse with rank
   statistics (Spearman) that are invariant to per-set rescaling.

DATA SOURCES
------------
  - Xiao 29-case parameterised periodic hills  (dose_response_xiao.npz)  [DNS]
  - curved BFS Re_H=13700, lambda/delta=inf anchor      [WRLES, Bentaleb 2012]
  - conv-div channel, Krank hills, gaussian bumps, JAXA/Coleman bubbles,
    KTH diffuser  -- shape-agnostic collapse only                        [DNS/LES]
  - wavy_*_wall_profiles.npz, rib_*_wall_profiles.npz  (auto-discovered) [RANS]
All read-only; this script writes ONLY codes/results/transition_map_dataset.npz.

Output:  codes/results/transition_map_dataset.npz
Run:     OMP_NUM_THREADS=2 python3 codes/analysis/transition_map.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
GEOM = os.path.join(CODES, "new_data_download", "geometry_driven")
NDD = os.path.join(CODES, "new_data_download")
RESULTS = os.path.join(CODES, "results")

sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

Y_IDX = 10
EPS_RES = 0.1
R2_TOL = 0.88      # tolerated threshold (= ODE success bound used in the paper)


# ----------------------------------------------------------------------------
# unified a-priori evaluator (identical protocol to evaluate_geometry_driven.py)
# ----------------------------------------------------------------------------
def evaluate(path, y_idx=Y_IDX):
    d = np.load(path, allow_pickle=True)
    y, U = d["y"], d["U"]
    tau_true = np.asarray(d["tau_w"], float)
    dp_dx = np.asarray(d["dp_dx"], float)
    nu = d["nu"]
    n = len(tau_true)
    nu = np.full(n, float(nu)) if np.ndim(nu) == 0 else np.asarray(nu, float)

    y_m = np.array([(y[i] if y.ndim == 2 else y)[y_idx] for i in range(n)])
    tau_pred = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        if y_idx >= len(yi) or y_m[i] <= 0 or np.isnan(Ui[y_idx]):
            continue
        tau_pred[i] = predict_tau_w(Ui[y_idx], y_m[i], dp_dx[i], nu[i])

    denom = np.abs(dp_dx) * np.abs(y_m)
    eps = np.full(n, np.nan)
    m = denom > 1e-30
    eps[m] = np.abs(tau_true[m]) / denom[m]
    eps_v = eps[np.isfinite(eps)]

    valid = np.isfinite(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = float(np.sum((tw_t - tw_p) ** 2))
    ss_tot = float(np.sum((tw_t - tw_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # coefficient of variation of the TRUE wall stress, and the median relative
    # error of the ODE prediction.  Both are needed to classify degenerate
    # zero-variance controls (e.g. a flat-limit channel where tau_w is uniform):
    # there R^2 is meaningless (ss_tot -> 0) yet the ODE is accurate in absolute
    # terms.  A point only counts as a *failure* if it carries real spatial
    # variance AND R^2<0; a uniform-stress wall the ODE reproduces (small median
    # relative error) is tolerated by construction.
    cv_tau = float(np.std(tw_t) / np.abs(np.mean(tw_t))) if tw_t.size and np.mean(tw_t) != 0 else np.nan
    rel = np.abs(tw_p - tw_t) / np.where(np.abs(tw_t) > 1e-30, np.abs(tw_t), np.nan)
    med_relerr = float(np.nanmedian(rel)) if rel.size else np.nan
    return dict(
        r2=float(r2),
        eps_median=float(np.median(eps_v)) if eps_v.size else np.nan,
        frac_eps_lt_0p1=float(np.mean(eps_v < EPS_RES)) if eps_v.size else np.nan,
        f_sep=float(np.mean(tau_true < 0)),
        cv_tau=cv_tau, med_relerr=med_relerr,
        n_stations=n, n_valid=int(valid.sum()),
    )


# ----------------------------------------------------------------------------
# rank / classifier helpers (no sklearn dependency)
# ----------------------------------------------------------------------------
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_score(score, label):
    """AUC that `score` ranks positives (label==1) above negatives. Mann-Whitney."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {i: (csum[i] - cnt[i] + 1 + csum[i]) / 2.0 for i in range(len(cnt))}
    ranks = np.array([avg[i] for i in inv])
    sum_pos = ranks[label == 1].sum()
    return float((sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def best_threshold(eps, fail):
    """eps* maximising balanced accuracy of (eps < eps* => predict fail)."""
    eps, fail = np.asarray(eps, float), np.asarray(fail, int)
    m = np.isfinite(eps)
    eps, fail = eps[m], fail[m]
    cands = np.unique(eps)
    mids = np.concatenate([[cands[0] * 0.5],
                           np.sqrt(cands[:-1] * cands[1:]),
                           [cands[-1] * 1.5]])
    best_t, best_j = np.nan, -1.0
    P, N = (fail == 1).sum(), (fail == 0).sum()
    for t in mids:
        pred = eps < t
        tpr = np.sum(pred & (fail == 1)) / P if P else 0.0
        tnr = np.sum(~pred & (fail == 0)) / N if N else 0.0
        j = 0.5 * (tpr + tnr)
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t), float(best_j)


# ----------------------------------------------------------------------------
# dataset registry
# ----------------------------------------------------------------------------
def build_registry():
    """Return list of dataset rows. geom_known=True means it carries trustworthy
    (a/delta, lambda/delta) coordinates for the geometry-plane MAP; geom_known=
    False means it contributes only to the shape-agnostic eps collapse."""
    rows = []

    # --- Xiao 29-case periodic hills: the genuine 2-axis repeating family -----
    xz = np.load(os.path.join(RESULTS, "dose_response_xiao.npz"), allow_pickle=True)
    a_over = 1.0 / np.asarray(xz["agg_delta"], float)              # hill height 1 / delta
    lam_over = np.asarray(xz["agg_cv_ellp_over_delta"], float)
    for i, case in enumerate(np.asarray(xz["agg_case"])):
        rows.append(dict(
            tag=f"xiao_{case}", shape="smooth_hill", provenance="DNS",
            a_over_delta=float(a_over[i]), lambda_over_delta=float(lam_over[i]),
            geom_known=True,
            eps_median=float(xz["agg_eps_median"][i]),
            frac_eps_lt_0p1=float(xz["agg_frac_eps_lt_0p1"][i]),
            r2=float(xz["agg_r2"][i]), f_sep=float(xz["agg_f_sep"][i]),
            cv_tau=np.nan, med_relerr=np.nan,
            n_stations=int(xz["agg_case"].shape[0] and 0) or None,
        ))

    # --- BFS: the lambda/delta -> inf zero-frequency anchor -------------------
    bfs = evaluate(os.path.join(GEOM, "curved_bfs_Re13700_DNS_wall_profiles.npz"))
    rows.append(dict(tag="curved_bfs_Re13700", shape="bfs_zero_freq",
                     provenance="WRLES", a_over_delta=1.0,
                     lambda_over_delta=np.inf, geom_known=True, **_pick(bfs)))

    # --- collapse-only DNS/LES anchors (geometry coords not asserted) ---------
    collapse_only = [
        ("conv_div_channel", os.path.join(GEOM, "conv_div_channel_Re12600_wall_profiles.npz"), "smooth_wavy"),
        ("krank_pehill_Re5600", os.path.join(GEOM, "krank_pehill_Re5600_wall_profiles.npz"), "smooth_hill"),
        ("krank_pehill_Re10595", os.path.join(GEOM, "krank_pehill_Re10595_wall_profiles.npz"), "smooth_hill"),
        ("gaussian_bump_Re1M", os.path.join(NDD, "gaussian_bump_Re1M_wall_profiles.npz"), "single_bump"),
        ("gaussian_bump_Re2M", os.path.join(NDD, "gaussian_bump_Re2M_wall_profiles.npz"), "single_bump"),
        ("separation_bubble_caseA", os.path.join(NDD, "separation_bubble_caseA_wall_profiles.npz"), "single_bubble"),
        ("separation_bubble_caseB", os.path.join(NDD, "separation_bubble_caseB_wall_profiles.npz"), "single_bubble"),
        ("kth_3d_diffuser", os.path.join(NDD, "kth_3d_diffuser_Re18000_wall_profiles.npz"), "diffuser"),
    ]
    # Known geometry coordinates for the two Krank periodic hills (bind B-L2-4).
    # Both are the canonical ERCOFTAC/Breuer periodic hill that the Krank et al.
    # (2018) DNS resolves: hill height h=1, top wall L_y=3.036 (file y-max=3.036,
    # confirmed on disk) so delta=L_y/2=1.518; streamwise period L_x=9h. Hence
    #   a/delta      = h/delta      = 1/1.518   = 0.659
    #   lambda/delta = L_x/delta    = 9/1.518   = 5.929
    # identical to the Xiao alpha=1.0 baseline case (alph10-9-3036: a/d=0.659,
    # lam/d=5.92), differing only in Reynolds number. The two cases share the
    # SAME geometry, so they carry the same (a/delta, lambda/delta).
    KRANK_DELTA = 3.036 / 2.0
    krank_coords = {
        "krank_pehill_Re5600": (1.0 / KRANK_DELTA, 9.0 / KRANK_DELTA),
        "krank_pehill_Re10595": (1.0 / KRANK_DELTA, 9.0 / KRANK_DELTA),
    }
    for tag, path, shape in collapse_only:
        if not os.path.exists(path):
            continue
        try:
            ev = evaluate(path)
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue
        ad, ld = krank_coords.get(tag, (np.nan, np.nan))
        rows.append(dict(tag=tag, shape=shape, provenance="DNS",
                         a_over_delta=ad, lambda_over_delta=ld,
                         geom_known=bool(np.isfinite(ad) and np.isfinite(ld)),
                         **_pick(ev)))

    # --- auto-discover OpenFOAM-generated wavy + rib points (geometry known) --
    for path in sorted(glob.glob(os.path.join(RESULTS, "wavy_*_wall_profiles.npz")) +
                       glob.glob(os.path.join(RESULTS, "rib_*_wall_profiles.npz"))):
        d = np.load(path, allow_pickle=True)
        ev = evaluate(path)
        tag = os.path.basename(path).replace("_wall_profiles.npz", "")
        shape = "wavy_wall" if tag.startswith("wavy") else "square_rib"
        a_over = float(d["a_over_delta"]) if "a_over_delta" in d.files else np.nan
        lam_over = float(d["lambda_over_delta"]) if "lambda_over_delta" in d.files else np.nan
        rows.append(dict(tag=tag, shape=shape, provenance="OpenFOAM-RANS",
                         a_over_delta=a_over, lambda_over_delta=lam_over,
                         geom_known=np.isfinite(a_over) and np.isfinite(lam_over),
                         **_pick(ev)))
    return rows


def _pick(ev):
    return dict(eps_median=ev["eps_median"], frac_eps_lt_0p1=ev["frac_eps_lt_0p1"],
                r2=ev["r2"], f_sep=ev["f_sep"], n_stations=ev["n_stations"],
                cv_tau=ev.get("cv_tau", np.nan), med_relerr=ev.get("med_relerr", np.nan))


def main():
    rows = build_registry()
    eps = np.array([r["eps_median"] for r in rows], float)
    r2 = np.array([r["r2"] for r in rows], float)
    frac = np.array([r["frac_eps_lt_0p1"] for r in rows], float)
    shape = np.array([r["shape"] for r in rows])
    prov = np.array([r["provenance"] for r in rows])
    a_over = np.array([r["a_over_delta"] for r in rows], float)
    lam_over = np.array([r["lambda_over_delta"] for r in rows], float)
    tags = np.array([r["tag"] for r in rows])
    cv_tau = np.array([r.get("cv_tau", np.nan) for r in rows], float)
    med_relerr = np.array([r.get("med_relerr", np.nan) for r in rows], float)

    # Robust fail/tolerated labels (degeneracy-aware).  A flat-limit control with
    # near-uniform wall stress (CV(tau_w) < 5%) makes R^2 meaningless (zero
    # variance to explain) even though the ODE is accurate in absolute terms; it
    # must not be mislabelled a failure.  A point is a FAILURE only if it has real
    # spatial variance AND R^2<0; it is TOLERATED if R^2>=0.88, or if it is a
    # low-variance control the ODE reproduces (median relative error < 25%).
    CV_FLAT, RELERR_OK = 0.05, 0.25
    degenerate = np.isfinite(cv_tau) & (cv_tau < CV_FLAT)
    fail = ((r2 < 0.0) & (~degenerate)).astype(int)             # 1 = ODE fails
    tolerated = ((r2 >= R2_TOL) |
                 (degenerate & (med_relerr < RELERR_OK))).astype(int)

    print("=" * 92)
    print(f"TRANSITION-MAP dataset: {len(rows)} geometry points "
          f"({fail.sum()} fail, {tolerated.sum()} tolerated, "
          f"{len(rows)-fail.sum()-tolerated.sum()} marginal)")
    print("=" * 92)
    print(f"{'tag':30s} {'shape':14s} {'prov':14s} {'a/d':>6s} {'lam/d':>7s} "
          f"{'eps~':>7s} {'R2':>9s}")
    for r in rows:
        ld = "inf" if not np.isfinite(r["lambda_over_delta"]) else f"{r['lambda_over_delta']:.2f}"
        ad = "nan" if not np.isfinite(r["a_over_delta"]) else f"{r['a_over_delta']:.2f}"
        print(f"{r['tag'][:30]:30s} {r['shape']:14s} {r['provenance']:14s} "
              f"{ad:>6s} {ld:>7s} {r['eps_median']:7.3f} {r['r2']:9.2f}")

    # ---- shape-agnostic order-parameter test --------------------------------
    rho_eps = spearman(eps, r2)
    rho_frac = spearman(frac, r2)
    auc = auc_score(-eps, fail)                 # small eps should rank as fail
    eps_star, jstat = best_threshold(eps, fail)

    # deterministic bootstrap CI on eps*
    rng = np.random.default_rng(0)
    boot = []
    idx_all = np.arange(len(rows))
    for _ in range(2000):
        bi = rng.integers(0, len(rows), len(rows))
        t, _ = best_threshold(eps[bi], fail[bi])
        if np.isfinite(t):
            boot.append(t)
    boot = np.array(boot)
    eps_star_lo, eps_star_hi = np.percentile(boot, [2.5, 97.5])

    print("-" * 92)
    print("SHAPE-AGNOSTIC ORDER PARAMETER (eps governs failure across all shapes)")
    print(f"  Spearman rho(eps_median, R2) = {rho_eps:+.3f}   "
          f"rho(frac<0.1, R2) = {rho_frac:+.3f}")
    print(f"  AUC[eps separates fail/not] = {auc:.3f}")
    print(f"  eps* (max balanced acc)     = {eps_star:.3f}  "
          f"[{eps_star_lo:.3f}, {eps_star_hi:.3f}]  (J={jstat:.3f})")

    # ---- leave-one-shape-class-out ------------------------------------------
    print("  Leave-one-shape-class-out eps* (threshold not carried by one family):")
    logo = {}
    for sc in np.unique(shape):
        keep = shape != sc
        if fail[keep].sum() == 0 or (fail[keep] == 0).sum() == 0:
            continue
        t, j = best_threshold(eps[keep], fail[keep])
        logo[sc] = t
        print(f"     hold out {sc:14s} -> eps* = {t:.3f}")

    out = dict(
        tag=tags, shape=shape, provenance=prov,
        a_over_delta=a_over, lambda_over_delta=lam_over,
        eps_median=eps, frac_eps_lt_0p1=frac, r2=r2, fail=fail,
        tolerated=tolerated, cv_tau=cv_tau, med_relerr=med_relerr,
        degenerate=degenerate.astype(int),
        rho_eps_r2=rho_eps, rho_frac_r2=rho_frac, auc_eps=auc,
        eps_star=eps_star, eps_star_lo=float(eps_star_lo),
        eps_star_hi=float(eps_star_hi), youden_J=jstat,
        logo_shape=np.array(list(logo.keys())),
        logo_eps_star=np.array(list(logo.values()), float),
        Y_IDX=Y_IDX, R2_TOL=R2_TOL, EPS_RES=EPS_RES,
        n_geom_known=int(np.sum([r["geom_known"] for r in rows])),
    )
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, "transition_map_dataset.npz"), **out)
    print(f"\nSaved -> results/transition_map_dataset.npz "
          f"({out['n_geom_known']} points carry (a/d, lam/d) coordinates)")


if __name__ == "__main__":
    main()
