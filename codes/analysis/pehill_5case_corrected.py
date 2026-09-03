#!/usr/bin/env python3
r"""
pehill_5case_corrected.py
=========================
Hill-surface-aware re-evaluation of the four periodic-hills variants reported in
the manuscript benchmark table (tab:ode_results): h/L_x = 0.8, 1.0, 1.0-refined,
1.2, all at Re_b = 5600 (Xiao et al. 2020, pehill-5-cases-DNS).

WHY (G2)
--------
The retired benchmark summary for these four variants was computed from y=0-
pinned preprocessed wall-profile files, whose near-zero hill-face tau_w made its
scores invalid. This script
recomputes R^2, sign accuracy, median eps and the reversed-shear fraction for the
same four variants under the corrected hill-surface-aware extraction
(eq:hillsurface) + the same a-priori ODE solver used throughout the manuscript,
straight from the raw DNS.  It reports each result without assuming that all
four variants have the same verdict; case_1p0 is the canonical headline case.

OUTPUT
------
  codes/results/pehill_5case_corrected.npz

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/pehill_5case_corrected.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # codes/
RESULTS = os.path.join(ROOT, "results")
BASE = os.path.join(ROOT, "new_data_download", "geometry_driven",
                    "xiao_pehill_parameterized", "pehill-5-cases-DNS")

sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w               # noqa: E402

RE_B = 5600.0
NU = 1.0 / RE_B
Y_IDX = 10
VEL_TOL = 1e-6

# (table label, case directory, h/L_x)
VARIANTS = [
    (r"$h/L_x = 0.8$",            "case_0p8",         0.8),
    (r"$h/L_x = 1.0$",            "case_1p0",         1.0),
    (r"$h/L_x = 1.0$, refined",   "case_1p0_refined", 1.0),
    (r"$h/L_x = 1.2$",            "case_1p2",         1.2),
]


def evaluate(case_dir, y_idx=Y_IDX):
    mean = np.loadtxt(os.path.join(case_dir, "mean_files.dat"))   # x y u v w p
    x, y, u, v, p = mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3], mean[:, 5]
    xu = np.unique(np.round(x, 6))

    xs, tau_true, p_wall, y_m, U_m = [], [], [], [], []
    for xv in xu:
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp = y[m], u[m], v[m], p[m]
        o = np.argsort(yy)
        yy, uu, vv, pp = yy[o], uu[o], vv[o], pp[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < y_idx + 2:
            continue
        k = max(fluid[0], 1)
        ywall = yy[k - 1:] - yy[k - 1]
        ufl, pfl = uu[k - 1:], pp[k - 1:]
        nfit = min(4, len(ywall) - 1)
        yf, uf = ywall[1:1 + nfit], ufl[1:1 + nfit]
        dudy = float(np.sum(yf * uf) / np.sum(yf * yf)) if np.sum(yf * yf) > 0 else 0.0
        xs.append(xv)
        tau_true.append(NU * dudy)
        p_wall.append(pfl[1])
        y_m.append(ywall[y_idx])
        U_m.append(ufl[y_idx])

    xs = np.asarray(xs); tau_true = np.asarray(tau_true)
    y_m = np.asarray(y_m); U_m = np.asarray(U_m)
    dp_dx = np.gradient(np.asarray(p_wall), xs)

    tau_pred = np.array([predict_tau_w(U_m[i], y_m[i], dp_dx[i], NU)
                         for i in range(len(xs))])
    valid = ~np.isnan(tau_pred)
    tp, tt = tau_pred[valid], tau_true[valid]
    ss_res = np.sum((tt - tp) ** 2)
    ss_tot = np.sum((tt - tt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rel_err = np.sqrt(ss_res) / (np.sqrt(np.sum(tt ** 2)) + 1e-30)

    # sign accuracy over all profiles with non-zero reference tau_w
    nz = valid & (np.abs(tau_true) > 1e-30)
    sign_acc = float(np.mean(np.sign(tau_pred[nz]) == np.sign(tau_true[nz])))

    denom = np.abs(dp_dx) * np.abs(y_m)
    eps = np.where(denom > 1e-30, np.abs(tau_true) / denom, np.nan)
    eps_med = float(np.nanmedian(eps[np.isfinite(eps) & (eps > 0)]))
    f_sep = float(np.mean(tau_true < 0))
    return dict(r2=float(r2), rel_err=float(rel_err), sign_acc=sign_acc,
                eps_median=eps_med, f_sep=f_sep, n=int(len(xs)))


def main():
    print(f"{'variant':28s} {'R2':>12s} {'sign%':>6s} {'relerr':>7s} "
          f"{'eps~':>7s} {'f_sep':>6s} {'N':>5s}")
    out = {}
    labels, r2s, signs, relerrs, epss, fseps = [], [], [], [], [], []
    for label, cdir, hLx in VARIANTS:
        ev = evaluate(os.path.join(BASE, cdir, "dns-data"), y_idx=Y_IDX)
        print(f"{cdir:28s} {ev['r2']:12.2f} {ev['sign_acc']*100:6.1f} "
              f"{ev['rel_err']:7.2f} {ev['eps_median']:7.3f} {ev['f_sep']:6.3f} "
              f"{ev['n']:5d}")
        labels.append(cdir); r2s.append(ev["r2"]); signs.append(ev["sign_acc"])
        relerrs.append(ev["rel_err"]); epss.append(ev["eps_median"])
        fseps.append(ev["f_sep"])
        for k, val in ev.items():
            out[f"{cdir}_{k}"] = val
    out.update(case=np.array(labels), r2=np.array(r2s), sign_acc=np.array(signs),
               rel_err=np.array(relerrs), eps_median=np.array(epss),
               f_sep=np.array(fseps), Y_IDX=Y_IDX, Re_b=RE_B)
    np.savez(os.path.join(RESULTS, "pehill_5case_corrected.npz"), **out)
    print(f"\nR2 band: [{min(r2s):.1f}, {max(r2s):.1f}]   "
          f"(29-case family: [-84.3, -9.9])")
    print(f"Saved -> results/pehill_5case_corrected.npz")


if __name__ == "__main__":
    main()
