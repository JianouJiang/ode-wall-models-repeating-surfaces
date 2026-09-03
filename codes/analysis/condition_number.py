#!/usr/bin/env python3
r"""
condition_number.py
===================
Thrust #14 (L1, attempt 2) -- the *conditioning* of the wall-stress problem.

WHY THIS EXISTS
---------------
The manuscript already shows (i) the cancellation parameter
    eps(x) = |tau_w| / (|dp/dx| y_m)
collapses to O(0.1) over the periodic-hill family, and (ii) the ODE relative
error scales as O(1/eps) (Fig. error_vs_eps).  Both describe the *error of the
particular ODE that drops convection*.  The deeper, unresolved question a reviewer will ask is: **is the failure a deficiency of THIS model, or a property
of the PROBLEM?**  If only the dropped convective term mattered, a better local
closure (or the exact DNS stress) would help.  It does not -- the exact-DNS-
stress substitution makes hills WORSE (R^2_dns = -927 vs R^2_ml = -48,
diagnostic_test_corrected.npz).  That paradox is currently an empirical control
with no mechanism.

This module supplies the mechanism.  We treat the wall model as a map
    M : (U_m, y_m, dp/dx, closure) |--> tau_w
and measure its **condition number** -- the amplification of a small relative
input/closure error into the predicted wall stress, normalised by the TRUE wall
stress (the achievable-accuracy normalisation that controls R^2):
    kappa_X = | d tau_w^pred / tau_w^true | / | delta_X |    as delta_X -> 0.
kappa_X is obtained by directly PERTURBING the production mixing-length ODE
solver (ode_wall_model.predict_tau_w, UNCHANGED) by a relative delta on channel
X and re-solving -- an independent measurement, NOT an algebraic re-statement of
eps.

CENTRAL RESULT (real DNS, no fabrication)
-----------------------------------------
The cancellation makes tau_w a small residual, so the map is ill-conditioned on
*every* input channel -- pressure gradient, matching velocity, AND the turbulence
closure -- with kappa_X growing as eps -> 0.  The closure channel being amplified
is the new content: it explains the exact-stress paradox as a conditioning
consequence (perturbing the closure toward "exact" is itself amplified by
kappa_closure ~ 1/eps where eps << 1), and it sets a model-independent floor:
no local closure can be accurate where the problem is ill-conditioned.

This is closure-INDEPENDENT (it is demonstrated by perturbing the closure
itself), a-priori (the perturbation uses only reference inputs, never the answer),
and geometry-general: every success geometry -- including the wide-pitch repeating
conv-div negative control -- is well-conditioned, while only the O(delta)-pitch
periodic hill is not (G7 boundary).

FALSIFIERS (this thrust dies if any fire)
  F1  kappa_X does NOT rise as eps -> 0 (no monotone sensitivity-cancellation
      link)                                                  -> not a conditioning effect.
  F2  the CLOSURE channel is NOT amplified (kappa_closure flat in eps)
                                                             -> cannot explain the exact-stress paradox.
  F3  success geometries are as ill-conditioned as hills      -> kappa does not
      separate the regimes; the ill-conditioning is not specific to the cancellation.

A-priori only.  No new CFD.  <= 2 cores.  Every number traces to real DNS/LES
wall_profiles on disk via the SAME hill-surface / Y_IDX extraction used by the
production pipeline.

Usage
  smoke (hills only, no npz):
    OMP_NUM_THREADS=2 python3 codes/analysis/condition_number.py --smoke
  full (hills + cross-geometry, writes results/condition_number.npz):
    OMP_NUM_THREADS=2 python3 codes/analysis/condition_number.py
"""
import os
import sys
import argparse

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
VEND = os.path.join(CODES, "vendor", "universal_wall_function", "codes", "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D            # production hill-surface extraction
sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
import ode_wall_model as M                        # production mixing-length ODE (UNCHANGED)

Y_IDX = D.Y_IDX
DELTA = 1.0e-3            # relative perturbation for the finite-difference sensitivity
TW_FLOOR = 1.0e-12       # guard on |tau_w^true|

# Cross-geometry benchmark (success geometries + the wide-pitch repeating control).
# (label, path, is_repeating, pitch_is_O_delta)
CROSS = [
    ("bfs_Re13700",        os.path.join(VEND, "bfs_Re13700_wall_profiles.npz"),            False, False),
    ("curved_bfs_Re13700", os.path.join(VEND, "curved_bfs_Re13700_LES_wall_profiles.npz"), False, False),
    ("nasa_hump_Re936000", os.path.join(VEND, "nasa_hump_Re936000_wall_profiles.npz"),     False, False),
    ("conv_div_Re12600",   os.path.join(VEND, "conv_div_channel_Re12600_wall_profiles.npz"), True,  False),
    ("gaussian_bump_Re1M", os.path.join(VEND, "gaussian_speed_bump_Re1M_wall_profiles.npz"), False, False),
    ("sep_bubble_B",       os.path.join(VEND, "separation_bubble_caseB_wall_profiles.npz"),  False, False),
]


def _col(arr, i):
    return arr[i] if np.ndim(arr) == 2 else arr


def channel_sensitivities(U_m, y_m, dp_dx, nu, tau_true, delta=DELTA):
    r"""Finite-difference condition numbers of the production ODE on each input
    channel, normalised by the TRUE wall stress.

    kappa_X = |tau_w^pred(X(1+delta)) - tau_w^pred(X)| / |tau_w^true| / delta .

    Channels: pressure gradient (P), matching velocity (U), turbulence closure
    (the von Karman constant KAPPA, perturbed on the production module).  Returns
    NaN for any channel whose re-solve does not converge."""
    if abs(tau_true) < TW_FLOOR or y_m <= 0 or not np.isfinite(U_m):
        return np.nan, np.nan, np.nan, np.nan
    base = M.predict_tau_w(U_m, y_m, dp_dx, nu)
    if not np.isfinite(base):
        return np.nan, np.nan, np.nan, np.nan

    pP = M.predict_tau_w(U_m, y_m, dp_dx * (1.0 + delta), nu)
    pU = M.predict_tau_w(U_m * (1.0 + delta), y_m, dp_dx, nu)
    k0 = M.KAPPA
    M.KAPPA = k0 * (1.0 + delta)
    pK = M.predict_tau_w(U_m, y_m, dp_dx, nu)
    M.KAPPA = k0

    def amp(p):
        return abs(p - base) / abs(tau_true) / delta if np.isfinite(p) else np.nan

    kP, kU, kK = amp(pP), amp(pU), amp(pK)
    # worst-channel (sup-norm) condition number
    chans = [c for c in (kP, kU, kK) if np.isfinite(c)]
    kmax = max(chans) if chans else np.nan
    return kP, kU, kK, kmax


# --------------------------------------------------------------------------- #
#  Periodic hills (case_1p0): the failure geometry, per station                #
# --------------------------------------------------------------------------- #
def run_hills():
    NU = D.NU
    profs = D.extract_profiles()
    rows = []
    for pr in profs:
        if Y_IDX + 1 >= len(pr["y"]):
            continue
        y_m, U_m, dpdx = pr["y"][Y_IDX], pr["U"][Y_IDX], pr["dpdx"]
        tw = pr["tau_w_dns"]
        if y_m <= 0 or abs(dpdx) * y_m <= 0 or abs(tw) < TW_FLOOR:
            continue
        eps = abs(tw) / (abs(dpdx) * y_m)
        kP, kU, kK, kmax = channel_sensitivities(U_m, y_m, dpdx, NU, tw)
        base = M.predict_tau_w(U_m, y_m, dpdx, NU)
        relerr = abs(base - tw) / abs(tw) if np.isfinite(base) else np.nan
        rows.append((eps, tw < 0, kP, kU, kK, kmax, relerr))
    a = np.array(rows, float)
    return dict(eps=a[:, 0], sep=a[:, 1].astype(bool),
                kP=a[:, 2], kU=a[:, 3], kK=a[:, 4], kmax=a[:, 5], relerr=a[:, 6])


def summarise_hills(h):
    eps = h["eps"]
    ok = np.isfinite(h["kK"]) & np.isfinite(eps) & (eps > 0)
    inv = 1.0 / eps[ok]
    out = {}
    print("=" * 70)
    print("PERIODIC HILLS (case_1p0) -- conditioning of the wall-stress map")
    print(f"  stations: {ok.sum()}   separated: {int(h['sep'][ok].sum())}")
    print("-" * 70)
    # F1/F2: sensitivity rises as eps -> 0 (Spearman vs 1/eps), per channel
    for name, key in [("pressure", "kP"), ("velocity", "kU"),
                      ("closure ", "kK"), ("worst-ch", "kmax")]:
        k = h[key][ok]
        m = np.isfinite(k)
        rho = spearmanr(k[m], inv[m]).statistic
        out[f"spearman_{key}_invconv"] = float(rho)
        print(f"  Spearman(kappa_{name}, 1/eps) = {rho:+.3f}    "
              f"median kappa = {np.median(k[m]):.2f}")
    out["spearman_relerr_invconv"] = float(
        spearmanr(h["relerr"][ok], inv).statistic)
    print(f"  Spearman(ODE relerr, 1/eps)  = {out['spearman_relerr_invconv']:+.3f}"
          f"   (reference: the known error scaling)")
    print("-" * 70)
    # dose-response: kappa_closure binned by eps
    print("  closure-channel condition number, binned by eps:")
    edges = [0.0, 0.05, 0.1, 0.3, 1.0, np.inf]
    bin_eps, bin_kK, bin_n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (eps >= lo) & (eps < hi) & np.isfinite(h["kK"])
        if m.sum() == 0:
            continue
        mk = float(np.median(h["kK"][m]))
        me = float(np.median(eps[m]))
        bin_eps.append(me); bin_kK.append(mk); bin_n.append(int(m.sum()))
        print(f"    eps in [{lo:.2g},{hi:.2g}):  n={m.sum():3d}  "
              f"median kappa_closure={mk:5.2f}  (1/eps_med={1/me:5.1f})")
    out["bin_eps_med"] = np.array(bin_eps)
    out["bin_kK_med"] = np.array(bin_kK)
    out["bin_n"] = np.array(bin_n)
    # prefactor kappa_closure ~ c / eps in the cancellation regime (eps < 0.3)
    canc = ok & (eps < 0.3) & np.isfinite(h["kK"]) & (h["kK"] > 0)
    pref = float(np.median(h["kK"][canc] * eps[canc]))
    out["closure_prefactor_eps_lt_0p3"] = pref
    out["kK_p90_hills"] = float(np.nanpercentile(h["kK"][ok], 90))
    out["kK_med_hills"] = float(np.nanmedian(h["kK"][ok]))
    print(f"  => kappa_closure ~ {pref:.3f} / eps  in the cancellation regime "
          f"(eps<0.3); 90th-pct kappa_closure={out['kK_p90_hills']:.2f}")
    print("=" * 70)
    return out


# --------------------------------------------------------------------------- #
#  Cross-geometry: success geometries + wide-pitch repeating control           #
# --------------------------------------------------------------------------- #
def run_cross():
    geom_rows = []
    print("\nCROSS-GEOMETRY conditioning (median over stations)")
    print(f"  {'geometry':<22}{'n':>5}{'eps_med':>10}{'kP_med':>9}"
          f"{'kU_med':>9}{'kK_med':>9}{'kK_p90':>9}{'repeat':>8}")
    for label, path, rep, pitch_Odelta in CROSS:
        if not os.path.exists(path):
            print(f"  {label:<22}  MISSING")
            continue
        d = np.load(path, allow_pickle=True)
        y, U = d["y"], d["U"]
        dpdx = np.asarray(d["dp_dx"], float)
        tau = np.asarray(d["tau_w"], float)
        nu = np.atleast_1d(np.asarray(d["nu"], float))
        n = len(tau)
        ee, kp, ku, kk = [], [], [], []
        for i in range(n):
            yi, Ui = _col(y, i), _col(U, i)
            if Y_IDX + 1 >= len(yi):
                continue
            y_m, U_m = float(yi[Y_IDX]), float(Ui[Y_IDX])
            nu_i = float(nu[i]) if nu.size > 1 else float(nu[0])
            if y_m <= 0 or abs(dpdx[i]) * y_m <= 0 or abs(tau[i]) < TW_FLOOR:
                continue
            eps = abs(tau[i]) / (abs(dpdx[i]) * y_m)
            kP, kU, kK, _ = channel_sensitivities(U_m, y_m, dpdx[i], nu_i, tau[i])
            if not np.isfinite(kK):
                continue
            ee.append(eps); kp.append(kP); ku.append(kU); kk.append(kK)
        if not kk:
            print(f"  {label:<22}  no valid stations")
            continue
        ee, kp, ku, kk = map(np.array, (ee, kp, ku, kk))
        row = dict(label=label, n=int(len(kk)), eps_med=float(np.median(ee)),
                   kP_med=float(np.nanmedian(kp)), kU_med=float(np.nanmedian(ku)),
                   kK_med=float(np.nanmedian(kk)), kK_p90=float(np.nanpercentile(kk, 90)),
                   is_repeating=bool(rep), pitch_O_delta=bool(pitch_Odelta))
        geom_rows.append(row)
        print(f"  {label:<22}{row['n']:>5}{row['eps_med']:>10.3g}"
              f"{row['kP_med']:>9.2f}{row['kU_med']:>9.2f}{row['kK_med']:>9.2f}"
              f"{row['kK_p90']:>9.2f}{'yes' if rep else 'no':>8}")
    return geom_rows


def link_to_paradox(hills_summary, geom_rows):
    """Tie kappa_closure to the exact-DNS-stress paradox (G3, Thrust #12):
    the closure channel is amplified on hills, ~flat on successes, which is why
    substituting the exact stress (a finite closure change) is catastrophic only
    on hills."""
    diag = np.load(os.path.join(RESULTS, "diagnostic_test_corrected.npz"),
                   allow_pickle=True)
    r2_ml = float(diag["standard_ml_r2"])
    r2_dns = float(diag["controlled_dns_r2"])
    kK_hills = hills_summary["kK_p90_hills"]
    kK_success = np.median([g["kK_med"] for g in geom_rows]) if geom_rows else np.nan
    print("\nLINK TO THE EXACT-DNS-STRESS PARADOX (G3)")
    print(f"  hills 90th-pct kappa_closure = {kK_hills:.2f}  "
          f"(closure channel strongly amplified)")
    print(f"  success-geometry median kappa_closure = {kK_success:.2f}  "
          f"(closure channel benign)")
    print(f"  => exact-stress substitution catastrophic ONLY on hills: "
          f"R^2_ml={r2_ml:.0f} -> R^2_dns={r2_dns:.0f}")
    return dict(r2_ml=r2_ml, r2_dns=r2_dns,
                kK_p90_hills=float(kK_hills),
                kK_med_success=float(kK_success))


def main(smoke=False):
    print(f"condition_number.py  ({'SMOKE' if smoke else 'FULL'})  "
          f"Y_IDX={Y_IDX}  DELTA={DELTA}\n")
    h = run_hills()
    hs = summarise_hills(h)
    if smoke:
        print("\nSMOKE: hills only; no npz written; cross-geometry is the full run.")
        return
    geom_rows = run_cross()
    paradox = link_to_paradox(hs, geom_rows)

    # --- falsifier verdicts -------------------------------------------------
    f1 = hs["spearman_kmax_invconv"] > 0.5          # sensitivity rises as eps->0
    f2 = hs["spearman_kK_invconv"] > 0.5            # CLOSURE channel amplified
    kK_success = paradox["kK_med_success"]
    f3 = hs["kK_p90_hills"] > 3.0 * kK_success       # hills >> successes
    print("\nFALSIFIER VERDICTS")
    print(f"  F1 (sensitivity rises as eps->0):     "
          f"{'PASS' if f1 else 'FAIL'}  "
          f"(Spearman kappa_max,1/eps = {hs['spearman_kmax_invconv']:+.2f})")
    print(f"  F2 (closure channel amplified):       "
          f"{'PASS' if f2 else 'FAIL'}  "
          f"(Spearman kappa_closure,1/eps = {hs['spearman_kK_invconv']:+.2f})")
    print(f"  F3 (hills >> successes, regime-spec): "
          f"{'PASS' if f3 else 'FAIL'}  "
          f"(hills kK_p90={hs['kK_p90_hills']:.2f} vs success kK_med={kK_success:.2f})")

    out = os.path.join(RESULTS, "condition_number.npz")
    pack = dict(
        Y_IDX=Y_IDX, DELTA=DELTA,
        hills_eps=h["eps"], hills_sep=h["sep"],
        hills_kappa_P=h["kP"], hills_kappa_U=h["kU"],
        hills_kappa_closure=h["kK"], hills_kappa_max=h["kmax"],
        hills_relerr=h["relerr"],
        geom_rows=np.array(geom_rows, dtype=object),
        f1_sensitivity_rises=bool(f1),
        f2_closure_amplified=bool(f2),
        f3_regime_specific=bool(f3),
    )
    for k, v in hs.items():
        pack[f"hills_{k}"] = v
    for k, v in paradox.items():
        pack[f"paradox_{k}"] = v
    np.savez(out, **pack)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
