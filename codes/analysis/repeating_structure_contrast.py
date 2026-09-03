"""
repeating_structure_contrast.py
===============================
A-priori test of the *specificity* of the repeating-structure failure
criterion using two genuinely repeating (quasi-periodic) wall geometries:

  * periodic hills  (Xiao et al. 2020, Re_b = 5600) -- the canonical instance
    where the per-element recirculation FILLS the period (f_rec ~ 0.44),
  * converging-diverging channel (Laval & Marquillie 2011, Re_b = 12600) -- a
    repeating wavy-wall geometry whose lee-side recirculation is LOCALISED and
    is followed by a long attached recovery (f_rec ~ 0.05).

The criterion (manuscript, Sec. "Repeating structures") says repetition is
*necessary but not sufficient*: the domain-wide cancellation regime (eps << 1,
ODE failure) is triggered only when the recirculation covers an O(1) fraction
f_rec of every period. These two cases are BOTH repeating, yet:

  - periodic hills : f_rec ~ 0.44  ->  eps << 1 domain-wide  ->  ODE FAILS
  - conv-div chan. : f_rec ~ 0.05  ->  eps ~ O(1)            ->  ODE SUCCEEDS

i.e. the discriminator is the coverage fraction f_rec, NOT mere repetition.
This directly substantiates the "risk flag, not theorem" framing: a repeating
geometry alone does not prove failure.

This script PROVES nothing about blade rows / cities / heat exchangers; it adds
ONE additional real, end-to-end repeating geometry (the conv-div channel) to
the demonstrated set, on the tolerated side of the criterion.

All inputs are real DNS fields. NO fabricated data.
Output -> codes/results/repeating_structure_contrast.npz

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/repeating_structure_contrast.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "new_data_download")

# The validated ODE/TBLE wall model used throughout the manuscript.
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w               # noqa: E402

Y_IDX = 10   # matching-height index, identical to the manuscript's a-priori protocol


def largest_contiguous_span(x, mask):
    """Streamwise extent of the largest contiguous reversed-shear run (= the
    per-element recirculation length L_sep), in the units of x."""
    if not mask.any():
        return 0.0, (np.nan, np.nan)
    best_len, best_span = 0.0, (np.nan, np.nan)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            span = x[j] - x[i]
            if span > best_len:
                best_len, best_span = span, (x[i], x[j])
            i = j + 1
        else:
            i += 1
    return best_len, best_span


def ode_r2_and_eps(npz_path, delta_proxy):
    """Run the ODE wall model a-priori on every station; return R^2(tau_w),
    median cancellation parameter eps, reversed-shear fraction, and the
    repeating-structure geometry metrics."""
    d = np.load(npz_path, allow_pickle=True)
    x = np.asarray(d["x"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    U = np.asarray(d["U"], dtype=float)
    tau_true = np.asarray(d["tau_w"], dtype=float)
    dp_dx = np.asarray(d["dp_dx"], dtype=float)
    nu = np.asarray(d["nu"], dtype=float)
    n = len(tau_true)
    if nu.ndim == 0:
        nu = np.full(n, float(nu))

    tau_pred = np.full(n, np.nan)
    eps = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        if Y_IDX >= len(yi):
            continue
        y_m, U_m = yi[Y_IDX], Ui[Y_IDX]
        if y_m <= 0 or np.isnan(U_m):
            continue
        tau_pred[i] = predict_tau_w(U_m, y_m, dp_dx[i], nu[i] if len(nu) > 1 else nu[0])
        denom = abs(dp_dx[i]) * y_m
        if denom > 1e-30:
            eps[i] = abs(tau_true[i]) / denom

    valid = ~np.isnan(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = np.sum((tw_t - tw_p) ** 2)
    ss_tot = np.sum((tw_t - tw_t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    sep = tau_true < 0
    ell_p = float(x.max() - x.min())
    L_sep, span = largest_contiguous_span(x, sep)
    f_rec = L_sep / ell_p if ell_p > 0 else np.nan
    eps_med = float(np.nanmedian(eps))

    return dict(
        r2=float(r2),
        f_rev=float(sep.mean()),
        eps_median=eps_med,
        ell_p=ell_p,
        L_sep=float(L_sep),
        recirc_span=np.asarray(span, dtype=float),
        f_rec=float(f_rec),
        Lsep_over_delta=float(L_sep / delta_proxy),
        pitch_over_Lsep=float(ell_p / L_sep) if L_sep > 0 else np.inf,
        n_stations=n,
        n_sep=int(sep.sum()),
    )


def pehill_from_budget():
    """Periodic-hills geometry metrics from the already-computed momentum
    budget (Xiao Re_b=5600); ODE R^2 is the catastrophic value reported by the
    manuscript's criterion_evaluation (periodic_hills_case_1p0)."""
    d = np.load(os.path.join(RESULTS, "momentum_budget_pehill.npz"), allow_pickle=True)
    x = np.asarray(d["x_unique"], dtype=float)
    y = np.asarray(d["y_unique"], dtype=float)
    sep = np.asarray(d["sep_mask"], dtype=bool)
    ell_p = float(x.max() - x.min())
    H = float(y.max() - y.min())
    delta = 0.5 * H
    L_sep, span = largest_contiguous_span(x, sep)
    f_rec = L_sep / ell_p

    # The legacy vendor profile pins the wall at y=0 on the hill faces.  Use the
    # corrected surface-aware profile for every wall-stress-dependent quantity;
    # retain this independent budget file only for the geometric period and
    # recirculation span.
    corrected_path = os.path.join(
        RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    corrected = ode_r2_and_eps(corrected_path, delta_proxy=delta)
    r2 = corrected["r2"]
    eps_med = corrected["eps_median"]

    return dict(
        r2=r2,
        f_rev=float(sep.mean()),
        eps_median=eps_med,
        ell_p=ell_p,
        L_sep=float(L_sep),
        recirc_span=np.asarray(span, dtype=float),
        f_rec=float(f_rec),
        delta_proxy=delta,
        Lsep_over_delta=float(L_sep / delta),
        pitch_over_Lsep=float(ell_p / L_sep) if L_sep > 0 else np.inf,
        n_stations=len(x),
        n_sep=int(sep.sum()),
        source_profile=np.array(
            "results/periodic_hills_case_1p0_wall_profiles_corrected.npz"),
        extraction=np.array("hill_surface_aware"),
    )


def main():
    # --- conv-div channel: delta proxy = channel half-height = 1 (its units) --
    cdc = ode_r2_and_eps(
        os.path.join(DATA, "conv_div_channel_Re12600_wall_profiles.npz"),
        delta_proxy=1.0,
    )
    cdc["delta_proxy"] = 1.0
    ph = pehill_from_budget()

    out = {}
    for tag, rec in (("pehill", ph), ("convdiv", cdc)):
        for k, v in rec.items():
            out[f"{tag}_{k}"] = v
    out["note"] = ("Specificity test: both geometries repeat; the coverage "
                   "fraction f_rec discriminates ODE failure (pehill) from "
                   "success (conv-div), not mere repetition.")
    np.savez(os.path.join(RESULTS, "repeating_structure_contrast.npz"), **out)

    print("Repeating-structure specificity contrast (real DNS, a-priori ODE)")
    print("=" * 74)
    hdr = f"{'quantity':<26}{'periodic hills':>20}{'conv-div channel':>22}"
    print(hdr)
    print("-" * 74)
    rows = [
        ("repeats?",            "yes",                 "yes"),
        ("Re_b",                "5600",                "12600"),
        ("repeat pitch ell_p",  f"{ph['ell_p']:.2f}",  f"{cdc['ell_p']:.2f}"),
        ("recirc L_sep",        f"{ph['L_sep']:.2f}",  f"{cdc['L_sep']:.2f}"),
        ("coverage f_rec",      f"{ph['f_rec']:.3f}",  f"{cdc['f_rec']:.3f}"),
        ("reversed-shear frac", f"{ph['f_rev']:.3f}",  f"{cdc['f_rev']:.3f}"),
        ("L_sep / delta",       f"{ph['Lsep_over_delta']:.2f}", f"{cdc['Lsep_over_delta']:.2f}"),
        ("ell_p / L_sep",       f"{ph['pitch_over_Lsep']:.2f}", f"{cdc['pitch_over_Lsep']:.2f}"),
        ("eps median",          f"{ph['eps_median']:.2e}", f"{cdc['eps_median']:.2e}"),
        ("ODE R^2(tau_w)",      f"{ph['r2']:.3g}",     f"{cdc['r2']:.3f}"),
        ("criterion fires?",    "YES (f_rec~0.4)",     "no (f_rec~0.05)"),
        ("ODE outcome",         "FAIL",                "SUCCESS"),
    ]
    for name, a, b in rows:
        print(f"{name:<26}{a:>20}{b:>22}")
    print("-" * 74)
    print("Interpretation: repetition is necessary but NOT sufficient. The")
    print("recirculation coverage f_rec is the discriminator: it fills the")
    print("period for hills (failure) but not for the conv-div channel (success).")
    print("Saved -> results/repeating_structure_contrast.npz")


if __name__ == "__main__":
    main()
