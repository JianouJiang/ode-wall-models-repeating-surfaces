#!/usr/bin/env python3
r"""
build_corrected_xiao_profiles.py  --  Thrust #27, Level 2 (the load-bearing step)
=================================================================================
Generate *hill-surface-aware* wall-profile files for the Xiao et al. (2020)
parameterised periodic-hill family at h/L_x = 0.8, 1.0, 1.2 (Re_b = 5600), with
the EXACT same schema and extraction convention as
``periodic_hills_case_1p0_wall_profiles_corrected.npz`` (built by
``build_corrected_pehill_profiles.py``).

WHY THIS EXISTS (L1 Judge issue #3, the single load-bearing L2 task)
--------------------------------------------------------------------
The Thrust #27 cross-geometry collapse needs the a-priori critical matching
height y_crit for ALL FIVE coupled cases.  Three are already on disk
(critical_matching_height_map.npz: krank, periodic_hills_1p0, conv_div).  The
two MISSING ones are the Xiao alpha=0.8 and alpha=1.2 coupled cases: their
reference fields (tau_w(x), dp/dx(x), full U(y) profiles per station) were never
extracted in the corrected, hill-surface-aware convention.  This script extracts
them straight from the raw Xiao DNS (mean_files.dat + rms_files2.dat), using the
IDENTICAL protocol as the alpha=1.0 corrected file, so the variance-balance
y_crit sweep (critical_matching_height.py) can be run on all three steepnesses
under one fixed protocol.

PROTOCOL (identical to build_corrected_pehill_profiles.py; eq:hillsurface)
  * per streamwise station x, sort wall-normal, drop solid-region points
    (|u|,|v| < VEL_TOL) and re-anchor distance-from-wall at the first fluid cell;
  * tau_w = nu * du/dy via a through-origin least-squares fit of the first
    <=4 fluid points;
  * dp/dx = d/dx of the wall-adjacent pressure;
  * Y_IDX=10 matching height used only as a cross-check (the sweep re-derives its
    own matching height).

The alpha=1.0 rebuild is a SELF-CONSISTENCY GATE: it must reproduce the existing
periodic_hills_case_1p0_wall_profiles_corrected.npz (median eps ~ 0.084,
frac(eps<0.1) ~ 0.564) to prove the generalised builder is faithful.

Honesty (G1/G2): every number traces to the read-only raw Xiao DNS; no
fabrication; no fields invented.  Re-runnable, deterministic.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/build_corrected_xiao_profiles.py
Out:  codes/results/periodic_hills_case_{0p8,1p0,1p2}_wall_profiles_corrected.npz
      (1p0 written to a _rebuild.npz so the canonical file is not overwritten)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # codes/
RESULTS = os.path.join(ROOT, "results")
BASE = os.path.join(ROOT, "new_data_download", "geometry_driven",
                    "xiao_pehill_parameterized", "pehill-5-cases-DNS")

RE_B = 5600.0
NU = 1.0 / RE_B
Y_IDX = 10
VEL_TOL = 1e-6
WIDTH = 257            # legacy wall-normal width, drop-in schema

# (table tag, case dir, h/L_x, output filename)
VARIANTS = [
    ("0p8", "case_0p8", 0.8, "periodic_hills_case_0p8_wall_profiles_corrected.npz"),
    ("1p0", "case_1p0", 1.0, "periodic_hills_case_1p0_wall_profiles_corrected_rebuild.npz"),
    ("1p2", "case_1p2", 1.2, "periodic_hills_case_1p2_wall_profiles_corrected.npz"),
]


def build(case_dir, tag, hLx):
    raw = os.path.join(BASE, case_dir, "dns-data")
    mean = np.loadtxt(os.path.join(raw, "mean_files.dat"))   # x y u v w p
    rms2 = np.loadtxt(os.path.join(raw, "rms_files2.dat"))   # x y uv uw vw
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

    # cross-check: eps median at the Y_IDX matching height
    y_m = Y[:, Y_IDX]
    denom = np.abs(dp_dx) * np.abs(y_m)
    valid = denom > 1e-30
    eps = np.full(n, np.nan)
    eps[valid] = np.abs(tau_w[valid]) / denom[valid]
    fin = np.isfinite(eps) & (eps > 0)
    med_eps = float(np.nanmedian(eps[fin])) if fin.any() else np.nan
    f01 = float(np.mean(eps[fin] < 0.1)) if fin.any() else np.nan
    med_tau = float(np.nanmedian(np.abs(tau_w[tau_w != 0])))

    return dict(Y=Y, U=U, V=V, UV=UV, xs=xs, dp_dx=dp_dx, tau_w=tau_w,
                nu_arr=nu_arr, u_tau=u_tau, Re_tau=Re_tau, is_sep=is_sep,
                n=n, med_eps=med_eps, f01=f01, med_tau=med_tau,
                f_sep=float(np.mean(is_sep)))


def main():
    print(f"{'tag':5s} {'N':>5s} {'med_eps':>8s} {'f(eps<.1)':>9s} "
          f"{'Re_tau(med)':>11s} {'f_sep':>6s}")
    for tag, cdir, hLx, outname in VARIANTS:
        r = build(cdir, tag, hLx)
        print(f"{tag:5s} {r['n']:5d} {r['med_eps']:8.4f} {r['f01']:9.3f} "
              f"{float(np.sqrt(r['med_tau'])/NU):11.0f} {r['f_sep']:6.3f}", end="")
        if tag == "1p0":
            print("   (self-consistency gate: expect med_eps~0.084, f~0.564)", end="")
        print()
        # GUARD (G2): the 5-case case_0p8/case_1p2 ASCII files carry a broken,
        # ~zero pressure column (|dp/dx|_med ~ 4e-5 vs the real 1p0 ~6e-2),
        # inflating eps ~1000x (med_eps ~150-280, f(eps<0.1)=0).  Refuse to write
        # a corrected-looking file for them: the reference-consistent source is
        # the 29-case DNS (alph075/alph10-9/alph125), used by thrust27_collapse_l2.py.
        if tag in ("0p8", "1p2") and (not np.isfinite(r["med_eps"]) or r["med_eps"] > 5.0):
            print(f"      REFUSED to write '{outname}': broken pressure column "
                  f"(med_eps={r['med_eps']:.1f} >> physical). Use 29-case DNS instead.")
            continue
        out = os.path.join(RESULTS, outname)
        np.savez(out, y=r["Y"], U=r["U"], V=r["V"], uv=r["UV"], x=r["xs"],
                 dp_dx=r["dp_dx"], tau_w=r["tau_w"], nu=r["nu_arr"],
                 u_tau=r["u_tau"], Re_tau=r["Re_tau"], is_separated=r["is_sep"],
                 geometry=np.array("periodic_hills"),
                 re_identifier=np.array(tag),
                 extraction=np.array("hill_surface_aware"),
                 median_eps=r["med_eps"], frac_below_01=r["f01"])
        print(f"      saved -> results/{outname}")

    # gate: rebuilt 1p0 vs canonical
    canon = os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    reb = os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected_rebuild.npz")
    if os.path.exists(canon):
        a = np.load(canon, allow_pickle=True)
        b = np.load(reb, allow_pickle=True)
        d_eps = abs(float(a["median_eps"]) - float(b["median_eps"]))
        tau_ok = np.allclose(np.asarray(a["tau_w"]), np.asarray(b["tau_w"]),
                             rtol=1e-6, atol=1e-12, equal_nan=True)
        print(f"\nGATE: rebuilt 1p0 vs canonical: |d med_eps|={d_eps:.2e}, "
              f"tau_w identical={tau_ok}  "
              f"-> {'PASS' if (d_eps < 1e-6 and tau_ok) else 'CHECK'}")


if __name__ == "__main__":
    main()
