#!/usr/bin/env python3
"""
Read-only a-priori evaluation of the geometry-driven repeating-structure cases
that appear in manuscript Table 2 (``tab:extended``):

  * curved backward-facing step (Bentaleb et al. 2012, Re_H = 13,700)
  * Krank periodic hills (Krank et al. 2018) at Re_b = 5600 and 10,595.

Why this script exists
----------------------
The Level-2/3 review flagged a reproducibility seam: the curved-BFS and Krank
periodic-hills numbers in Table 2 were supplied by
``new_data_download/all_new_dataset_results.npz``, which is produced by
``fix_and_evaluate_all.py``. That driver's *first* step OVERWRITES the Xiao
periodic-hills ``*_wall_profiles.npz`` files in the ``geometry_driven/`` symlink,
i.e. it mutates the read-only source DNS database. It therefore must never run
inside the clean pipeline.

This script reproduces *only* the three geometry-driven Table 2 rows, strictly
read-only: it loads the relevant ``*_wall_profiles.npz`` files, runs the same
ODE wall model used everywhere else, and writes a single regenerable artifact
``results/geometry_driven_table2.npz``. It does not touch any source data.

The cancellation parameter is reported as the median over stations with
|dp/dx| > 0, matching the manuscript definition of \\tilde{epsilon}.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # codes/analysis
CODES = os.path.dirname(HERE)                                # codes/
GEOM_DIR = os.path.join(CODES, "new_data_download", "geometry_driven")
RESULTS = os.path.join(CODES, "results")

# Same ODE wall model used by every other evaluation in this paper.
sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

Y_IDX = 10  # matching index, y+ ~ 50, identical to the rest of the paper

# (key, wall-profiles file) — these files are read-only DNS, never written here.
CASES = [
    ("curved_bfs_Re13700_DNS", "curved_bfs_Re13700_DNS_wall_profiles.npz"),
    ("krank_pehill_Re5600",    "krank_pehill_Re5600_wall_profiles.npz"),
    ("krank_pehill_Re10595",   "krank_pehill_Re10595_wall_profiles.npz"),
]


def evaluate(filepath, y_idx=Y_IDX):
    """A-priori ODE evaluation of one dataset. Pure read; returns a metrics dict."""
    d = np.load(filepath, allow_pickle=True)
    y, U = d["y"], d["U"]
    tau_true = np.asarray(d["tau_w"], dtype=float)
    dp_dx = np.asarray(d["dp_dx"], dtype=float)
    nu_arr = d["nu"]
    n = len(tau_true)
    if np.ndim(nu_arr) == 0:
        nu_arr = np.full(n, float(nu_arr))

    tau_pred = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        if y_idx >= len(yi):
            continue
        y_m, U_m = yi[y_idx], Ui[y_idx]
        if y_m <= 0 or np.isnan(U_m):
            continue
        nu_i = nu_arr[i] if len(nu_arr) > 1 else nu_arr[0]
        tau_pred[i] = predict_tau_w(U_m, y_m, dp_dx[i], nu_i)

    valid = ~np.isnan(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = float(np.sum((tw_t - tw_p) ** 2))
    ss_tot = float(np.sum((tw_t - tw_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    f_sep = float(np.mean(tau_true < 0))

    # Cancellation parameter epsilon(x), masked to stations with |dp/dx| > 0
    # (the manuscript definition of \tilde{epsilon}).
    y_m_all = np.array([(y[i] if y.ndim == 2 else y)[y_idx] for i in range(n)])
    denom = np.abs(dp_dx) * np.abs(y_m_all)
    mask = denom > 1e-30
    epsilon = np.full(n, np.nan)
    epsilon[mask] = np.abs(tau_true[mask]) / denom[mask]
    eps_valid = epsilon[np.isfinite(epsilon)]
    eps_median = float(np.median(eps_valid)) if eps_valid.size else np.nan

    sep = tw_t < 0
    n_sep = int(sep.sum())
    sign_acc_sep = float(np.mean(np.sign(tw_p[sep]) == np.sign(tw_t[sep]))) \
        if n_sep else np.nan

    return {
        "r2": r2, "f_sep": f_sep, "eps_median": eps_median,
        "n_stations": n, "n_sep": n_sep, "sign_acc_sep": sign_acc_sep,
        "tau_true": tau_true, "tau_pred": tau_pred,
        "x": np.asarray(d["x"], dtype=float).ravel(),
        "dp_dx": dp_dx, "epsilon": epsilon,
    }


def main():
    print("Read-only geometry-driven Table 2 evaluation (no source data is modified)")
    print("=" * 78)
    print(f"{'Geometry':28s} {'N':>4s} {'f_sep':>7s} {'eps_med':>9s} "
          f"{'R2(tau_w)':>11s} {'SA_sep':>7s}")
    print("-" * 78)

    save = {}
    names = []
    for key, fname in CASES:
        path = os.path.join(GEOM_DIR, fname)
        if not os.path.exists(path):
            raise SystemExit(f"MISSING read-only DNS file: {path}")
        res = evaluate(path)
        names.append(key)
        sa = f"{res['sign_acc_sep']:.2f}" if not np.isnan(res['sign_acc_sep']) else "---"
        print(f"{key:28s} {res['n_stations']:>4d} {res['f_sep']:>7.3f} "
              f"{res['eps_median']:>9.3f} {res['r2']:>+11.4f} {sa:>7s}")
        for k, v in res.items():
            save[f"{key}_{k}"] = v
    save["dataset_names"] = np.array(names)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "geometry_driven_table2.npz")
    np.savez(out, **save)
    print("-" * 78)
    print(f"Saved -> results/{os.path.basename(out)}")


if __name__ == "__main__":
    main()
