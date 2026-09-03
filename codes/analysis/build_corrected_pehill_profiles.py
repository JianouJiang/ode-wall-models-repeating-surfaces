#!/usr/bin/env python3
r"""
build_corrected_pehill_profiles.py
==================================
Generate a *hill-surface-aware* periodic-hills wall-profile file with the SAME
schema as the legacy ``periodic_hills_case_1p0_wall_profiles.npz`` so that every
figure that consumes it (error_vs_epsilon, fig_spatial_error,
cancellation_parameter_extended, ...) can be repointed at the corrected data with
a one-line path change.

WHY (the headline reconciliation, G2)
-------------------------------------
The legacy preprocessed file pins the matching column at the channel floor y=0.
On the hill faces that samples velocity *inside the solid hill* (U~0), giving a
spurious tau_w~1e-6 (Re_tau~5.6, unphysical) and hence a spurious eps~1e-4 and
blown-up relative errors O(1e4) (near-zero denominator). The artifact is
documented in ``verify_wall_extraction.py``; the corrected headline cancellation
number lives in ``plot_cancellation_parameter_corrected.py``. This script applies
the identical hill-surface-aware convention (eq:hillsurface) per station and
re-anchors each profile to the hill surface, reading the SAME raw DNS.

The re-anchored profiles are distance-from-wall columns (fluid only), padded to
the legacy wall-normal width with NaN. Consumers index the matching height
Y_IDX=10 only, so NaN padding beyond the fluid column is inert.

OUTPUT
------
  codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/build_corrected_pehill_profiles.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # codes/
RESULTS = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(ROOT, "new_data_download", "geometry_driven",
                       "xiao_pehill_parameterized", "pehill-5-cases-DNS",
                       "case_1p0", "dns-data")

RE_B = 5600.0
NU = 1.0 / RE_B
Y_IDX = 10
VEL_TOL = 1e-6
WIDTH = 257            # legacy wall-normal width, for drop-in schema


def main():
    mean = np.loadtxt(os.path.join(RAW_DIR, "mean_files.dat"))   # x y u v w p
    # Xiao archive schema: rms_files1 = uu,vv,ww,pp;
    # rms_files2 = uv,uw,vw.  The former project reader incorrectly used the
    # pressure-statistic column as uv (WP1 source-column audit).
    rms2 = np.loadtxt(os.path.join(RAW_DIR, "rms_files2.dat"))   # x y uv uw vw
    x, y, u, v, p = mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3], mean[:, 5]
    uv = rms2[:, 2]
    xu = np.unique(np.round(x, 6))
    n = len(xu)

    Y = np.full((n, WIDTH), np.nan)
    U = np.full((n, WIDTH), np.nan)
    V = np.full((n, WIDTH), np.nan)
    UV = np.full((n, WIDTH), np.nan)
    xs = np.zeros(n)
    tau_w = np.zeros(n)
    p_wall = np.full(n, np.nan)

    for s, xv in enumerate(xu):
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp, ww = y[m], u[m], v[m], p[m], uv[m]
        o = np.argsort(yy)
        yy, uu, vv, pp, ww = yy[o], uu[o], vv[o], pp[o], ww[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        xs[s] = xv
        if len(fluid) < 2:
            continue
        k = max(fluid[0], 1)
        yw = yy[k - 1]
        ywall = yy[k - 1:] - yw            # distance from wall (fluid column)
        ufl, vfl, pfl, wwfl = uu[k - 1:], vv[k - 1:], pp[k - 1:], ww[k - 1:]
        L = min(len(ywall), WIDTH)
        Y[s, :L] = ywall[:L]
        U[s, :L] = ufl[:L]
        V[s, :L] = vfl[:L]
        UV[s, :L] = wwfl[:L]
        # tau_w = nu du/dy, LS fit of first few fluid points through the origin
        nfit = min(4, len(ywall) - 1)
        yf, uf = ywall[1:1 + nfit], ufl[1:1 + nfit]
        tau_w[s] = NU * (float(np.sum(yf * uf) / np.sum(yf * yf))
                         if np.sum(yf * yf) > 0 else 0.0)
        p_wall[s] = pfl[1] if len(pfl) > 1 else pfl[0]

    dp_dx = np.gradient(p_wall, xs)
    is_sep = tau_w < 0
    nu_arr = np.full(n, NU)
    u_tau = np.sqrt(np.abs(tau_w))
    Re_tau = u_tau / NU

    # ── cross-check: eps median from this file must match the corrected headline
    y_m = Y[:, Y_IDX]
    denom = np.abs(dp_dx) * np.abs(y_m)
    valid = denom > 1e-30
    eps = np.full(n, np.nan)
    eps[valid] = np.abs(tau_w[valid]) / denom[valid]
    fin = np.isfinite(eps) & (eps > 0)
    med_eps = float(np.nanmedian(eps[fin]))
    f01 = float(np.mean(eps[fin] < 0.1))
    med_tau = float(np.nanmedian(np.abs(tau_w[tau_w != 0])))

    print(f"stations            : {n}")
    print(f"median eps          : {med_eps:.4f}   (corrected headline 0.084)")
    print(f"frac(eps<0.1)       : {f01:.3f}   (corrected headline 0.564)")
    print(f"median |tau_w|      : {med_tau:.3e}  -> Re_tau = "
          f"{float(np.sqrt(med_tau)/NU):.0f} (physical)")
    print(f"reversed-shear frac : {float(np.mean(is_sep)):.3f}")

    out = os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    np.savez(out, y=Y, U=U, V=V, uv=UV, x=xs, dp_dx=dp_dx, tau_w=tau_w,
             nu=nu_arr, u_tau=u_tau, Re_tau=Re_tau, is_separated=is_sep,
             geometry=np.array("periodic_hills"), re_identifier=np.array("1p0"),
             extraction=np.array("hill_surface_aware"),
             median_eps=med_eps, frac_below_01=f01)
    print(f"\nSaved -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
