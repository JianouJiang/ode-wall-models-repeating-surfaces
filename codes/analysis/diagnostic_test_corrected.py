#!/usr/bin/env python3
r"""
diagnostic_test_corrected.py
============================
Hill-surface-aware re-run of the periodic-hills DIAGNOSTIC TEST
(``diagnostic_test.py``), which establishes the central structural claim (G3):
substituting the exact DNS Reynolds stress does NOT rescue the ODE wall model.

WHY THIS EXISTS
---------------
The legacy ``diagnostic_test.py`` reads the *preprocessed*
``periodic_hills_case_1p0_wall_profiles.npz`` for both the ground-truth tau_w
and the matching profile. That file pins the wall at y=0, so on the hill faces
it samples velocity *inside the solid hill* (U~0), manufacturing a spurious
tau_w~1e-6 -> R^2 ~ -4.7e6. The y=0-pinning artifact is documented in
``analysis/verify_wall_extraction.py`` and corrected for the headline
cancellation parameter in ``figures/plot_cancellation_parameter_corrected.py``.

This script repeats the three diagnostic solvers under the SAME hill-surface-aware
extraction convention (eq:hillsurface) used by ``dose_response_xiao.py`` and the
corrected cancellation figure, reading the SAME raw DNS (case_1p0 mean plus the
documented Reynolds-shear file).
The question it answers: does the closure-independence conclusion survive the
correction? (It must, for G3 to stand.)

OUTPUT
------
  codes/results/diagnostic_test_corrected.npz

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/diagnostic_test_corrected.py
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # codes/
RESULTS = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(ROOT, "new_data_download", "geometry_driven",
                       "xiao_pehill_parameterized", "pehill-5-cases-DNS",
                       "case_1p0", "dns-data")

# main (production) ODE solver, identical to the rest of the manuscript
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w as _main_predict_tau_w   # noqa: E402

RE_B = 5600.0
NU = 1.0 / RE_B
KAPPA, A_PLUS, N_GRID = 0.41, 26.0, 500
Y_IDX = 10
VEL_TOL = 1e-6


def extract_profiles():
    """Hill-surface-aware per-station extraction from raw case_1p0 DNS.

    Returns per-station: distance-from-wall fluid column (y, U, uv), the
    matching-height (y_m, U_m), the hill-surface tau_w (nu du/dy, LS fit through
    the no-slip origin), and the near-wall dp/dx (central difference)."""
    mean = np.loadtxt(os.path.join(RAW_DIR, "mean_files.dat"))   # x y u v w p
    # Xiao schema: rms_files1 = uu,vv,ww,pp; rms_files2 = uv,uw,vw.
    # The former rms_files1[:,5] read produced the superseded -927 score.
    rms2 = np.loadtxt(os.path.join(RAW_DIR, "rms_files2.dat"))   # x y uv uw vw
    x, y, u, v, p = mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3], mean[:, 5]
    uv = rms2[:, 2]
    xu = np.unique(np.round(x, 6))

    profs, xs, tau_w, p_wall = [], [], [], []
    for xv in xu:
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp, ww = y[m], u[m], v[m], p[m], uv[m]
        o = np.argsort(yy)
        yy, uu, vv, pp, ww = yy[o], uu[o], vv[o], pp[o], ww[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < Y_IDX + 2:
            continue
        k = max(fluid[0], 1)
        yw = yy[k - 1]
        ywall = yy[k - 1:] - yw
        ufl, pfl, uvfl = uu[k - 1:], pp[k - 1:], ww[k - 1:]
        nfit = min(4, len(ywall) - 1)
        yf, uf = ywall[1:1 + nfit], ufl[1:1 + nfit]
        dudy = float(np.sum(yf * uf) / np.sum(yf * yf)) if np.sum(yf * yf) > 0 else 0.0
        xs.append(xv)
        tau_w.append(NU * dudy)
        p_wall.append(pfl[1])
        profs.append(dict(y=ywall, U=ufl, uv=uvfl))
    xs = np.asarray(xs)
    tau_w = np.asarray(tau_w)
    dp_dx = np.gradient(np.asarray(p_wall), xs)
    for i, pr in enumerate(profs):
        pr["tau_w_dns"] = float(tau_w[i])
        pr["dpdx"] = float(dp_dx[i])
        pr["is_sep"] = bool(tau_w[i] < 0)
    return profs


def solve_controlled(tau_w_guess, y_match, dp_dx, mode, prof):
    eta = np.linspace(0, 1, N_GRID)
    y_grid = y_match * eta ** 1.5
    if mode == "mixing_length":
        u_tau = np.sqrt(max(abs(tau_w_guess), 1e-30))
        y_plus = y_grid * u_tau / NU
        D = 1.0 - np.exp(-y_plus / A_PLUS)
        l_m2 = (KAPPA * y_grid * D) ** 2
        F = tau_w_guess + dp_dx * y_grid
        absF, signF = np.abs(F), np.sign(F)
        disc = NU * NU + 4.0 * l_m2 * absF
        viscous = l_m2 < 1e-30
        l_m2_safe = np.where(viscous, 1.0, l_m2)
        absS = (-NU + np.sqrt(disc)) / (2.0 * l_m2_safe)
        absS = np.where(viscous, absF / NU, absS)
        dUdy = signF * absS
    elif mode == "dns_stress":
        uv_interp = np.interp(y_grid, prof["y"], prof["uv"])
        dUdy = (tau_w_guess + dp_dx * y_grid + uv_interp) / NU
    else:
        return np.nan
    dy = np.diff(y_grid)
    return np.sum(0.5 * (dUdy[:-1] + dUdy[1:]) * dy)


def predict_controlled(U_match, y_match, dp_dx, mode, prof):
    if abs(U_match) < 1e-15:
        return 0.0
    try:
        return brentq(lambda tw: solve_controlled(tw, y_match, dp_dx, mode, prof)
                      - U_match, -10.0, 10.0, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan


def main():
    profs = extract_profiles()
    print(f"Extracted {len(profs)} hill-surface-aware profiles")
    n = len(profs)
    gt = np.full(n, np.nan)
    pred = {k: np.full(n, np.nan) for k in
            ["standard_ml", "controlled_ml", "controlled_dns"]}
    is_sep = np.zeros(n, bool)
    valid = np.zeros(n, bool)

    t0 = time.time()
    for i, pr in enumerate(profs):
        if Y_IDX >= len(pr["y"]):
            continue
        y_m, U_m, dpdx = pr["y"][Y_IDX], pr["U"][Y_IDX], pr["dpdx"]
        pred["standard_ml"][i] = _main_predict_tau_w(U_m, y_m, dpdx, NU)
        pred["controlled_ml"][i] = predict_controlled(U_m, y_m, dpdx, "mixing_length", pr)
        pred["controlled_dns"][i] = predict_controlled(U_m, y_m, dpdx, "dns_stress", pr)
        gt[i] = pr["tau_w_dns"]
        is_sep[i] = pr["is_sep"]
        valid[i] = True
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} ({time.time()-t0:.1f}s)")

    print(f"\n{'method':<18s} {'R2':>14s} {'RMSE':>10s} {'SA_sep':>8s} {'N':>5s}")
    out = dict(tau_w_dns=gt, is_separated=is_sep, valid_mask=valid,
               n_profiles=n, n_separated=int(is_sep.sum()), Y_IDX=Y_IDX)
    for k in pred:
        mask = valid & np.isfinite(pred[k])
        p, t, s = pred[k][mask], gt[mask], is_sep[mask]
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((p - t) ** 2))
        sa = np.mean(np.sign(p[s]) == np.sign(t[s])) if s.sum() else np.nan
        print(f"{k:<18s} {r2:14.3f} {rmse:10.4f} {sa:8.3f} {mask.sum():5d}")
        out[k] = pred[k]
        out[f"{k}_valid"] = mask
        out[f"{k}_r2"] = float(r2)
        out[f"{k}_rmse"] = float(rmse)
        out[f"{k}_sa_sep"] = float(sa)

    np.savez(os.path.join(RESULTS, "diagnostic_test_corrected.npz"), **out)
    print(f"\nSaved -> results/diagnostic_test_corrected.npz")
    print(f"separated profiles (tau_w<0): {int(is_sep.sum())}/{n}")


if __name__ == "__main__":
    main()
