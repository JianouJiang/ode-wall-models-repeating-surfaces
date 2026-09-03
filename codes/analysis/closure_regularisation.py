#!/usr/bin/env python3
r"""
closure_regularisation.py
=========================
Level 1 (core methodology) for Thrust #30 -- the IMPOSSIBILITY / lower-bound
thesis.  This script gives the *mechanism* behind the on-disk
`closure_conditioning_floor.npz` numbers, which the manuscript's
\S\,sec:conditioning currently only asserts ("the exact stress has no benign
stations to anchor a rank law").  It turns that hand-wave into a measured,
first-principles statement:

    THE EDDY VISCOSITY IS A TIKHONOV REGULARISER OF THE WALL-STRESS INVERSE.

Derivation (made rigorous in methodology.md).  Integrate the thin-boundary-layer
balance d/dy[tau_tot] = F(y), with tau_tot(y) = tau_w + I_F(y),
I_F(y)=\int_0^y F.  A *local eddy-viscosity* closure relates the total stress to
the gradient multiplicatively, tau_tot = g(y) dU/dy with g = nu + nu_t(y):

        dU/dy = (tau_w + I_F)/g(y)
   =>   U_m   = tau_w * G0 + \int_0^{y_m} I_F/g dy ,   G0 = \int_0^{y_m} dy/g
   =>   tau_w = [ U_m - \int I_F/g dy ] / G0 .                         (eddy)

So a perturbation of the forcing/closure at height y propagates to tau_w with the
*inverse-operator weight*

        w_g(y) = (1/g(y)) / G0 ,    \int_0^{y_m} w_g dy = 1 .          (KEY)

A *frozen DNS stress* closure instead carries the turbulent stress as an ADDITIVE
source, tau_tot = nu dU/dy + tau_turb^DNS(y); the only thing dividing the
gradient is the molecular nu:

        dU/dy = (tau_w + I_F - tau_turb)/nu
   =>   tau_w = [ nu*U_m - \int (I_F - tau_turb) dy ] / y_m ,          (dns)

whose inverse-operator weight is w(y) = (1/nu)/(y_m/nu) = 1/y_m -- UNIFORM.

THE MECHANISM, IN ONE LINE.  A model eddy viscosity nu_t MULTIPLIES dU/dy, so it
grows outward (nu_t ~ kappa u_tau y) and DAMPS the inverse-operator gain exactly
in the outer part of [0,y_m] where the omitted convective integral I_C(y) is
largest -- a low-pass / Tikhonov filter w_g(y)\downarrow.  A frozen DNS stress is
an additive source; it provides NO such damping, so the inverse sees only the
(tiny) molecular nu and the gain weight is FLAT and maximal.  The "most faithful
closure" is the WORST because faithfulness here means giving up the regulariser.

THREE FALSIFIABLE, FIRST-PRINCIPLES PREDICTIONS (this script measures all three;
each is a number the manuscript can cite, and each can KILL the mechanism):

  R1  GAIN-SHAPE ORDERS THE CONDITIONING.  The closure-channel condition number
      kappa_closure measured on disk (closure_conditioning_floor.npz) is
      reproduced by the inverse-operator gain of w_g: rank-ordering the five
      closures by an *a-priori weight statistic* (outer-gain mass of w_g) gives
      the SAME ordering as kappa_med, with the exact-DNS (flat weight) strictly
      the largest.  KILL if the flat-weight closure is not the most-amplified.

  R2  REGULARISATION STRENGTH ANTI-CORRELATES WITH kappa.  Across closures, the
      eddy-to-total viscosity fraction carried in the matching layer,
      beta_reg = <nu_t/(nu+nu_t)>, anti-correlates with kappa_med (more
      regulariser -> smaller gain).  beta_reg(dns)=0 (no regulariser).  KILL if
      adding regulariser does not reduce the gain.

  R3  THE COHERENCE SPLIT IS A WEIGHT-INVARIANCE EFFECT.  For eddy closures w_g(y)
      is set by the LOCAL profile (u_tau, dp/dx), so it varies station-to-station
      and its gain tracks the local 1/eps -> Spearman(kappa,1/eps) high.  For the
      DNS-stress closure w(y)=1/y_m is IDENTICAL at every station (eps-blind), so
      the gain cannot track 1/eps -> Spearman collapses.  We measure the
      station-to-station shape variability of w_g and show eddy >> dns, which is
      the *cause* of the on-disk coherence split (rho 0.85-0.97 vs 0.19).
      KILL if the DNS-stress weight is not the most station-invariant.

A-priori only.  Real DNS hill-surface profiles on disk (the SAME extraction the
floor experiment uses).  No new CFD.  Cross-checks every claim against the
on-disk closure_conditioning_floor.npz so the mechanism cannot drift from the
headline numbers (killer gate G2).  Writes results/closure_regularisation.npz.

Usage
  smoke (canonical hill only, no npz):
    OMP_NUM_THREADS=2 python3 codes/analysis/closure_regularisation.py --smoke
  full (writes npz + cross-check):
    OMP_NUM_THREADS=2 python3 codes/analysis/closure_regularisation.py
"""
import os
import sys
import time
import argparse

import numpy as np
from scipy.stats import spearmanr
try:
    from numpy import trapezoid as _trapz       # numpy >= 2.0
except ImportError:                              # numpy < 2.0
    from numpy import trapz as _trapz

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D
import closure_conditioning_floor as F  # reuse the EXACT solver + closure family

Y_IDX = F.Y_IDX
NU = F.NU_HILLS
N_GRID = F.N_GRID
CLOSURES = F.CLOSURES
TW_FLOOR = F.TW_FLOOR


# --------------------------------------------------------------------------- #
#  The inverse-operator weight w_g(y) for a given closure at a given station   #
# --------------------------------------------------------------------------- #
def inverse_operator_weight(clo, tau_w, y_m, dp_dx, nu, prof):
    r"""Return (y_grid, w, g, nut_frac) for the converged closure at this station.

    g(y) is the EFFECTIVE total viscosity that maps the local total stress
    tau_tot = tau_w + I_F(y) to the gradient dU/dy (g = tau_tot/(dU/dy)), i.e.
    the denominator of the inverse operator.  For the DNS-stress closure the
    regularising viscosity is molecular only (the turbulent stress is an additive
    source, not a multiplicative viscosity), so g == nu and the weight is flat.
    w(y) = (1/g)/\int(1/g) is the normalised inverse-operator gain; nut_frac =
    nu_t/(nu+nu_t) is the local regularisation fraction (0 for the DNS closure).
    """
    eta = np.linspace(0.0, 1.0, N_GRID)
    y_grid = y_m * eta ** 1.5
    F_force = tau_w + dp_dx * y_grid               # local total stress tau_tot(y)

    if clo["kind"] == "dns":
        # additive turbulent source -> the inverse sees only molecular nu
        g = np.full_like(y_grid, nu)
        nut_frac = np.zeros_like(y_grid)
    else:
        uv_interp = None
        dUdy = F._dUdy_profile(clo, tau_w, y_grid, dp_dx, nu, uv_interp, 1.0)
        # g = tau_tot / (dU/dy); guard the near-wall where both -> 0 consistently
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(dUdy) > 1e-30, F_force / dUdy, nu)
        g = np.where(np.isfinite(g) & (g >= nu), g, nu)   # g >= nu physically
        nut_frac = np.clip((g - nu) / np.maximum(g, 1e-30), 0.0, 1.0)

    inv_g = 1.0 / g
    G0 = _trapz(inv_g, y_grid)
    w = inv_g / G0 if G0 > 0 else np.full_like(y_grid, np.nan)
    return y_grid, w, g, nut_frac


def weight_statistics(y_grid, w, nut_frac, y_m):
    r"""Scalar a-priori descriptors of the inverse-operator weight w_g(y):

      outer_mass : \int_{y_m/2}^{y_m} w dy  -- fraction of the inverse gain that
                   lands in the OUTER half, where the omitted convective integral
                   I_C(y) is largest.  Flat weight -> 0.5; outer-damped -> < 0.5.
      com        : centre of mass <y/y_m>_w  -- 0.5 for flat, < 0.5 for damped.
      gain_peak  : y_m * max(w)  -- the dimensionless peak gain (>= 1; =1 flat).
      beta_reg   : <nu_t/(nu+nu_t)>_w  -- regularisation strength (0 for dns).
    """
    outer = y_grid >= 0.5 * y_m
    outer_mass = float(_trapz(w[outer], y_grid[outer]))
    com = float(_trapz(w * (y_grid / y_m), y_grid))
    gain_peak = float(y_m * np.nanmax(w))
    beta_reg = float(_trapz(w * nut_frac, y_grid))   # w integrates to 1
    return dict(outer_mass=outer_mass, com=com, gain_peak=gain_peak, beta_reg=beta_reg)


# --------------------------------------------------------------------------- #
def run(smoke=False):
    t0 = time.time()
    print(f"closure_regularisation.py  ({'SMOKE' if smoke else 'FULL'})  "
          f"Y_IDX={Y_IDX}  N_GRID={N_GRID}  closures={len(CLOSURES)}\n")
    profs = D.extract_profiles()

    # per-station storage
    eps_l, tw_l = [], []
    W = {c["key"]: dict(outer_mass=[], com=[], gain_peak=[], beta_reg=[],
                        wshape=[]) for c in CLOSURES}
    n_grid_ref = np.linspace(0.0, 1.0, 64)        # common eta grid for shape var

    for pr in profs:
        if Y_IDX + 1 >= len(pr["y"]):
            continue
        y_m, U_m, dpdx = pr["y"][Y_IDX], pr["U"][Y_IDX], pr["dpdx"]
        tw = pr["tau_w_dns"]
        if y_m <= 0 or abs(dpdx) * y_m <= 0 or abs(tw) < TW_FLOOR:
            continue
        eps = abs(tw) / (abs(dpdx) * y_m)
        eps_l.append(eps); tw_l.append(tw)
        for c in CLOSURES:
            tau_pred = F.predict_tau_w(c, U_m, y_m, dpdx, NU, pr)
            if not np.isfinite(tau_pred):
                for k in ("outer_mass", "com", "gain_peak", "beta_reg"):
                    W[c["key"]][k].append(np.nan)
                W[c["key"]]["wshape"].append(np.full_like(n_grid_ref, np.nan))
                continue
            yg, w, g, nut_frac = inverse_operator_weight(c, tau_pred, y_m, dpdx, NU, pr)
            st = weight_statistics(yg, w, nut_frac, y_m)
            for k in ("outer_mass", "com", "gain_peak", "beta_reg"):
                W[c["key"]][k].append(st[k])
            # normalised weight SHAPE on a common grid (y_m*eta) for variability
            w_on_ref = np.interp(n_grid_ref, yg / y_m, w * y_m)   # dimensionless
            W[c["key"]]["wshape"].append(w_on_ref)

    eps = np.array(eps_l); inv = 1.0 / eps
    n = len(eps)
    print(f"stations: {n}\n")

    # cross-check vs the on-disk floor numbers (G2: the mechanism must reproduce
    # the headline conditioning, not invent its own)
    floor = np.load(os.path.join(RESULTS, "closure_conditioning_floor.npz"),
                    allow_pickle=True)
    disk_rows = {r["label"]: r for r in floor["hills_rows"]}

    print("=" * 92)
    print("INVERSE-OPERATOR WEIGHT  w_g(y)=(1/g)/INT(1/g)  vs  on-disk conditioning")
    print("=" * 92)
    hdr = (f"  {'closure':<18}{'outer_mass':>11}{'com':>7}{'gain_peak':>10}"
           f"{'beta_reg':>9}{'shape_CV':>9} | {'kappa_med(disk)':>15}{'rho(disk)':>10}")
    print(hdr)
    rows = {}
    for c in CLOSURES:
        d = W[c["key"]]
        om = np.array(d["outer_mass"], float)
        com = np.array(d["com"], float)
        gp = np.array(d["gain_peak"], float)
        br = np.array(d["beta_reg"], float)
        shp = np.array(d["wshape"], float)            # (n, 64)
        # station-to-station shape variability: mean across-station CV per node,
        # averaged over the layer.  Flat & invariant weight -> ~0; varying -> big.
        with np.errstate(invalid="ignore"):
            mean_shape = np.nanmean(shp, axis=0)
            std_shape = np.nanstd(shp, axis=0)
            shape_cv = float(np.nanmean(std_shape / np.maximum(mean_shape, 1e-9)))
        med_om = float(np.nanmedian(om))
        med_com = float(np.nanmedian(com))
        med_gp = float(np.nanmedian(gp))
        med_br = float(np.nanmedian(br))
        dlab = disk_rows.get(c["label"], {})
        kappa_med = float(dlab.get("med_kappa", np.nan))
        rho_disk = float(dlab.get("spearman", np.nan))
        rows[c["key"]] = dict(label=c["label"], outer_mass=med_om, com=med_com,
                              gain_peak=med_gp, beta_reg=med_br, shape_cv=shape_cv,
                              kappa_med=kappa_med, rho_disk=rho_disk)
        print(f"  {c['label']:<18}{med_om:>11.3f}{med_com:>7.3f}{med_gp:>10.2f}"
              f"{med_br:>9.3f}{shape_cv:>9.3f} | {kappa_med:>15.2f}{rho_disk:>10.3f}")
    print("=" * 92)

    keys = [c["key"] for c in CLOSURES]
    kap = np.array([rows[k]["kappa_med"] for k in keys])
    om = np.array([rows[k]["outer_mass"] for k in keys])
    gp = np.array([rows[k]["gain_peak"] for k in keys])
    br = np.array([rows[k]["beta_reg"] for k in keys])
    cv = np.array([rows[k]["shape_cv"] for k in keys])
    rho = np.array([rows[k]["rho_disk"] for k in keys])
    dns_i = keys.index("E_dns_stress")

    # ---- R1: gain shape orders the conditioning; flat weight is the worst ---- #
    worst_kappa = keys[int(np.argmax(kap))]
    flattest = keys[int(np.argmax(gp == np.nanmin(gp)))]  # smallest peak = flattest? no
    # "flat" = uniform weight => gain_peak ~ 1 and outer_mass ~ 0.5; the eddy
    # closures are outer-damped => gain_peak > 1, outer_mass < 0.5.  Predict the
    # MOST-amplified closure as the one whose weight is flattest (max outer_mass).
    pred_worst = keys[int(np.argmax(om))]
    rank_rho = spearmanr(om, kap).statistic
    R1 = (worst_kappa == "E_dns_stress") and (pred_worst == "E_dns_stress")
    print(f"\nR1  gain-shape orders conditioning:")
    print(f"    measured worst closure (max kappa_med) = {worst_kappa}")
    print(f"    predicted worst (max outer-gain mass)   = {pred_worst}")
    print(f"    Spearman(outer_mass, kappa_med) over 5 closures = {rank_rho:+.3f}")
    print(f"    exact-DNS outer_mass={om[dns_i]:.3f} (flat=0.5) vs eddy mean "
          f"{np.mean(np.delete(om,dns_i)):.3f}  ->  R1 {'PASS' if R1 else 'FAIL'}")

    # ---- R2: regularisation strength anti-correlates with kappa ------------- #
    rho_br = spearmanr(br, kap).statistic
    R2 = (br[dns_i] < 1e-6) and (rho_br < 0)
    print(f"\nR2  regularisation strength vs conditioning:")
    print(f"    beta_reg(dns)={br[dns_i]:.4f} (no regulariser); eddy beta_reg in "
          f"[{np.min(np.delete(br,dns_i)):.3f},{np.max(np.delete(br,dns_i)):.3f}]")
    print(f"    Spearman(beta_reg, kappa_med) = {rho_br:+.3f}  (expect < 0)  -> "
          f"R2 {'PASS' if R2 else 'FAIL'}")

    # ---- R3: coherence split = weight station-invariance of the dns closure -- #
    most_invariant = keys[int(np.argmin(cv))]
    rho_cv = spearmanr(cv, rho).statistic   # more shape variation -> higher disk rho
    R3 = (most_invariant == "E_dns_stress")
    print(f"\nR3  coherence split is a weight-invariance effect:")
    print(f"    most station-INVARIANT weight (min shape CV) = {most_invariant} "
          f"(dns CV={cv[dns_i]:.3f} vs eddy mean {np.mean(np.delete(cv,dns_i)):.3f})")
    print(f"    on-disk coherence: dns rho={rho[dns_i]:.3f} vs eddy "
          f"[{np.min(np.delete(rho,dns_i)):.3f},{np.max(np.delete(rho,dns_i)):.3f}]")
    print(f"    Spearman(shape_CV, disk_rho) = {rho_cv:+.3f}  ->  "
          f"R3 {'PASS' if R3 else 'FAIL'}")

    allpass = R1 and R2 and R3
    print("\n" + "=" * 92)
    print(f"MECHANISM VERDICT: R1={R1}  R2={R2}  R3={R3}  ->  "
          f"{'ALL PASS' if allpass else 'FAIL'}")
    print("=" * 92)

    if smoke:
        print(f"\nSMOKE done in {time.time()-t0:.1f}s; no npz written.")
        return allpass

    pack = dict(
        Y_IDX=Y_IDX, N_GRID=N_GRID, n_stations=n,
        closure_keys=np.array(keys),
        closure_labels=np.array([rows[k]["label"] for k in keys]),
        outer_mass=om, com=np.array([rows[k]["com"] for k in keys]),
        gain_peak=gp, beta_reg=br, shape_cv=cv,
        kappa_med_disk=kap, rho_disk=rho,
        R1_gainshape_orders_kappa=bool(R1),
        R1_spearman_outermass_kappa=float(rank_rho),
        R2_reg_anticorrelates_kappa=bool(R2),
        R2_spearman_betareg_kappa=float(rho_br),
        R3_dns_most_invariant=bool(R3),
        R3_dns_shape_cv=float(cv[dns_i]),
        R3_eddy_mean_shape_cv=float(np.mean(np.delete(cv, dns_i))),
        rows=np.array([rows[k] for k in keys], dtype=object),
    )
    out = os.path.join(RESULTS, "closure_regularisation.npz")
    np.savez(out, **pack)
    print(f"\nwrote {out}   [{time.time()-t0:.1f}s total]")
    return allpass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ok = run(smoke=ap.parse_args().smoke)
    sys.exit(0 if ok else 1)
