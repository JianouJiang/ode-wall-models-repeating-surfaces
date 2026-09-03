#!/usr/bin/env python3
"""
cube_pair_diagnostic.py -- the controlled 3-D pitch pair (packed vs sparse cube),
per-station error-mode diagnostic + form-drag control (F4).

Inputs (real, on disk):
  results/cube_array_wall_profiles.npz   packed  (pitch/delta = 1, lambda_p = 0.25)
  results/cube_sparse_wall_profiles.npz  sparse  (pitch/delta = 6, lambda_p = 0.0069)
  openfoam/{cube_array_prod,cube_sparse_prod}/postProcessing/forcesFD/*/forces.dat

For each cube, re-runs the verbatim a-priori ODE (cross_geometry_collapse.predict_tau_w,
Y_IDX) per floor station and records the Mode-I/Mode-II separators:
  kappa ~ 1/eps (median, p90)      -- conditioning
  amp   = med|pred| / med|true|    -- the O(1/eps) amplitude over-count
  bias  = mean(pred - true)/med|true|
  corr / spearman                  -- skill
  CV(tau_true)                     -- variance available to the R2 denominator
  C_canc = frac(dp/dx > 0 AND eps < 0.1)
  phi_FD = |Fp_x| / (|Fp_x| + |Fv_x|)  from the forcesFD integral (F4)

Writes results/cube_pair_diagnostic.npz. No simulation; pure read-only analysis.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
OF = os.path.join(CODES, "openfoam")
sys.path.insert(0, HERE)
from cross_geometry_collapse import predict_tau_w, Y_IDX, r2, rel_rms  # noqa: E402
from formdrag_completed_depth import parse_forces_dat                  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def per_station(path):
    d = np.load(path, allow_pickle=True)
    y, U = d["y"], d["U"]
    tt = np.asarray(d["tau_w"], float)
    dp = np.asarray(d["dp_dx"], float)
    nu = np.atleast_1d(np.asarray(d["nu"], float))
    n = len(tt)
    tp = np.full(n, np.nan)
    ym = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        ym[i] = yi[Y_IDX]
        if ym[i] > 0 and np.isfinite(Ui[Y_IDX]):
            tp[i] = predict_tau_w(Ui[Y_IDX], ym[i], dp[i],
                                  nu[i] if nu.size > 1 else nu[0])
    v = np.isfinite(tp) & np.isfinite(tt)
    eps = np.abs(tt) / np.maximum(np.abs(dp) * np.abs(ym), 1e-30)
    return tp[v], tt[v], eps[v], dp[v]


def phi_fd(case):
    base = os.path.join(OF, case, "postProcessing", "forcesFD")
    t = sorted(os.listdir(base), key=float)[-1]
    fp, fv = parse_forces_dat(os.path.join(base, t, "forces.dat"))
    return abs(fp) / (abs(fp) + abs(fv)), fp, fv, float(t)


def main():
    out = {}
    for tag, prof, case in (
            ("packed", "cube_array_wall_profiles.npz", "cube_array_prod"),
            ("sparse", "cube_sparse_wall_profiles.npz", "cube_sparse_prod")):
        tp, tt, eps, dp = per_station(os.path.join(RESULTS, prof))
        kappa = 1.0 / eps
        cstar = float(np.dot(tp, tt) / np.dot(tp, tp))
        row = dict(
            n=len(tt),
            r2=r2(tt, tp),
            relRMS=rel_rms(tt, tp),
            relRMS_rescaled=rel_rms(tt, cstar * tp),
            c_star=cstar,
            corr=float(np.corrcoef(tp, tt)[0, 1]),
            rho=spearman(tp, tt),
            amp_ratio=float(np.median(np.abs(tp)) / np.median(np.abs(tt))),
            bias=float(np.mean(tp - tt) / np.median(np.abs(tt))),
            cv_tau=float(np.std(tt) / abs(np.mean(tt))),
            eps_med=float(np.median(eps)),
            frac_eps_lt0p1=float(np.mean(eps < 0.1)),
            kappa_med=float(np.median(kappa)),
            kappa_p90=float(np.percentile(kappa, 90)),
            C_canc=float(np.mean((dp > 0) & (eps < 0.1))),
        )
        pfd, fp, fv, tF = phi_fd(case)
        row.update(phi_FD=float(pfd), Fp_x=float(fp), Fv_x=float(fv),
                   forces_time=tF)
        for k, vv in row.items():
            out[f"{tag}_{k}"] = vv
        print(f"{tag}: " + "  ".join(f"{k}={vv:.4g}" if isinstance(vv, float)
                                     else f"{k}={vv}" for k, vv in row.items()))

    out["kappa_ratio_med"] = out["packed_kappa_med"] / out["sparse_kappa_med"]
    out["eps_ratio_med"] = out["sparse_eps_med"] / out["packed_eps_med"]
    out["pitch_over_delta"] = np.array([1.0, 6.0])
    out["lambda_p"] = np.array([0.25, 1.0 / 144.0])
    out["note"] = (
        "Controlled 3-D pitch pair, identical Coceal ALIGNED unit cell, "
        "OpenFOAM-RANS kOmegaSST, iteration-mean harvests (verbatim evaluate, "
        "Y_IDX=10). Packed = Mode-I structural cancellation (kappa large, "
        "O(1/eps) amplitude over-count, no rank skill); sparse = Mode-II "
        "ordinary closure miss (kappa ordinary, amp ~1, real rank skill, "
        "R2<0 only via the variance-starved denominator CV(tau)~0.3). "
        "phi_FD stays form-drag-dominated on BOTH sides of the verdict flip "
        "(F4: form drag does not order the outcome).")
    np.savez(os.path.join(RESULTS, "cube_pair_diagnostic.npz"), **out)
    print(f"\nkappa ratio (packed/sparse, median) = {out['kappa_ratio_med']:.1f}x")
    print(f"eps ratio   (sparse/packed, median) = {out['eps_ratio_med']:.1f}x")
    print(f"wrote {os.path.join(RESULTS, 'cube_pair_diagnostic.npz')}")


if __name__ == "__main__":
    main()
