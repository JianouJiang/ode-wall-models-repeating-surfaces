"""
dose_response_closure_sensitivity.py
====================================
Is the dose-response law a property of the GEOMETRY, or an artefact of the
particular turbulence-closure constants in the ODE/TBLE wall model?

This is the parametric counterpart of the manuscript's "structural, not closure"
result (G3 / sec:diagnostic): there, swapping in *exact DNS Reynolds stresses*
fails to rescue the periodic hills. Here we instead perturb the two closure
constants of the algebraic mixing-length model used everywhere in the paper,

    nu_t = (kappa * y * D)^2 |dU/dy|,   D = 1 - exp(-y+/A+),

across the literature-plausible ranges kappa in {0.38, 0.41, 0.44} and
A+ in {20, 26, 32}, re-run the FULL Xiao 29-case dose-response for every
(kappa, A+) combination, and ask two questions:

  1. Robustness of failure: does every case still fail (R^2 < 0) for every
     closure setting?  (If the failure survived only at kappa=0.41, A+=26 it
     would be a closure artefact.)
  2. Robustness of the LAW: does the dominant geometric control (L_sep/delta)
     keep the same sign and comparable magnitude of Spearman rho across all
     closure settings?  (If rho flipped sign or collapsed to 0 under a closure
     tweak, the "law" would be a closure artefact, not geometry.)

HONESTY: identical a-priori protocol and data to dose_response_xiao.py; only the
two closure constants are varied. No DNS data is modified (the constants are
monkey-patched in-memory on the imported model module; the read-only vendor
file is never touched). Every number is written to
codes/results/dose_response_closure_sensitivity.npz by this committed script.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/dose_response_closure_sensitivity.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RESULTS = os.path.join(ROOT, "results")

# Reuse the *validated* readers / scales / collapse stats from the main script,
# so this experiment differs from it by closure constants ONLY.
sys.path.insert(0, HERE)
from dose_response_xiao import (read_case, parse_alpha, spearman, XIAO,  # noqa: E402
                                NU, Y_IDX, RE_B)

# the model module whose module-level KAPPA / A_PLUS we perturb
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
import ode_wall_model as M                          # noqa: E402

KAPPAS = [0.38, 0.41, 0.44]
A_PLUSES = [20.0, 26.0, 32.0]
NOMINAL = (0.41, 26.0)


def evaluate_case_with_current_closure(case):
    """ODE skill (R^2) for one case at the model module's CURRENT KAPPA/A_PLUS.

    Mirrors evaluate_case() in dose_response_xiao.py but recomputes only what
    depends on the closure (tau_pred -> R^2). Geometry-only quantities
    (L_sep/delta, alpha) are closure-independent and computed once by the caller.
    """
    x = case["x"]
    tau_true = case["tau_w"]
    dp_dx = case["dp_dx"]
    n = len(x)
    tau_pred = np.full(n, np.nan)
    for i in range(n):
        yi, Ui = case["y"][i], case["U"][i]
        if Y_IDX >= len(yi):
            continue
        y_m, U_m = yi[Y_IDX], Ui[Y_IDX]
        if y_m <= 0 or np.isnan(U_m):
            continue
        tau_pred[i] = M.predict_tau_w(U_m, y_m, dp_dx[i], NU)
    valid = ~np.isnan(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = np.sum((tw_t - tw_p) ** 2)
    ss_tot = np.sum((tw_t - tw_t.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    cases = sorted(d for d in os.listdir(XIAO)
                   if os.path.isdir(os.path.join(XIAO, d)) and d.startswith("alph"))
    print(f"Closure-sensitivity of the dose-response law: {len(cases)} Xiao "
          f"periodic-hill cases, Re_b={RE_B:.0f}, Y_IDX={Y_IDX}")

    # --- read every case ONCE; cache profiles + closure-independent geometry ---
    cached, Lsep_over_delta, alpha = [], [], []
    for name in cases:
        case = read_case(os.path.join(XIAO, name))
        cached.append(case)
        delta = 0.5 * case["L_y"]
        sep = case["tau_w"] < 0
        # largest contiguous reversed-shear span (= L_sep), inline (geometry only)
        L_sep, _ = __import__("dose_response_xiao").largest_contiguous_span(case["x"], sep)
        Lsep_over_delta.append(L_sep / delta if delta > 0 else np.nan)
        alpha.append(parse_alpha(name))
    Lsep_over_delta = np.array(Lsep_over_delta)
    alpha = np.array(alpha)

    grid_kappa, grid_aplus = [], []
    grid_rho_r2_Lsep, grid_rho_r2_alpha = [], []
    grid_n_fail, grid_r2_median, grid_r2_max = [], [], []
    nominal_r2 = None

    print("-" * 78)
    print(f"{'kappa':>6s} {'A+':>5s} | {'#R2<0':>6s}/{len(cases):<3d} "
          f"{'median R2':>11s} {'max R2':>9s} | {'rho_R2(Lsep/d)':>14s} {'rho_R2(alpha)':>13s}")
    for kappa in KAPPAS:
        for aplus in A_PLUSES:
            M.KAPPA, M.A_PLUS = kappa, aplus            # in-memory perturbation
            r2 = np.array([evaluate_case_with_current_closure(c) for c in cached])
            n_fail = int(np.sum(r2 < 0))
            rho_L = spearman(Lsep_over_delta, r2)
            rho_a = spearman(alpha, r2)
            grid_kappa.append(kappa); grid_aplus.append(aplus)
            grid_rho_r2_Lsep.append(rho_L); grid_rho_r2_alpha.append(rho_a)
            grid_n_fail.append(n_fail)
            grid_r2_median.append(float(np.nanmedian(r2)))
            grid_r2_max.append(float(np.nanmax(r2)))
            if (kappa, aplus) == NOMINAL:
                nominal_r2 = r2.copy()
            print(f"{kappa:6.2f} {aplus:5.1f} | {n_fail:6d}/{len(cases):<3d} "
                  f"{np.nanmedian(r2):11.2f} {np.nanmax(r2):9.3f} | "
                  f"{rho_L:14.3f} {rho_a:13.3f}")
    # restore nominal so a later import in the same process is unsurprised
    M.KAPPA, M.A_PLUS = NOMINAL

    grid_rho_r2_Lsep = np.array(grid_rho_r2_Lsep)
    grid_n_fail = np.array(grid_n_fail)
    all_fail = bool(np.all(grid_n_fail == len(cases)))
    rho_sign_consistent = bool(np.all(grid_rho_r2_Lsep < 0))
    rho_spread = float(grid_rho_r2_Lsep.max() - grid_rho_r2_Lsep.min())

    print("-" * 78)
    print(f"All 29 cases fail (R2<0) for ALL {len(grid_n_fail)} closure settings: {all_fail}")
    print(f"rho_R2(L_sep/delta) negative for ALL closure settings: {rho_sign_consistent}")
    print(f"rho_R2(L_sep/delta) range across closures: "
          f"[{grid_rho_r2_Lsep.min():.3f}, {grid_rho_r2_Lsep.max():.3f}] "
          f"(spread {rho_spread:.3f})")

    out = dict(
        kappas=np.array(KAPPAS), a_pluses=np.array(A_PLUSES),
        grid_kappa=np.array(grid_kappa), grid_aplus=np.array(grid_aplus),
        grid_rho_r2_Lsep_over_delta=grid_rho_r2_Lsep,
        grid_rho_r2_alpha=np.array(grid_rho_r2_alpha),
        grid_n_fail=grid_n_fail,
        grid_r2_median=np.array(grid_r2_median),
        grid_r2_max=np.array(grid_r2_max),
        n_cases=len(cases),
        all_cases_fail_all_closures=all_fail,
        rho_r2_Lsep_sign_consistent=rho_sign_consistent,
        rho_r2_Lsep_spread=rho_spread,
        Lsep_over_delta=Lsep_over_delta, alpha=alpha,
        Y_IDX=Y_IDX, Re_b=RE_B,
        nominal_kappa=NOMINAL[0], nominal_aplus=NOMINAL[1],
    )
    np.savez(os.path.join(RESULTS, "dose_response_closure_sensitivity.npz"), **out)
    print(f"\nSaved -> {os.path.relpath(os.path.join(RESULTS, 'dose_response_closure_sensitivity.npz'), ROOT)}")


if __name__ == "__main__":
    main()
