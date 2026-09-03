#!/usr/bin/env python3
r"""
rib_eps_ode.py  --  Thrust #23 (L2) geometry-agnostic ODE/eps evaluation core.

This module is the SHARED scientific instrument that scores BOTH the periodic
hills (the manuscript's headline failure case) and the new rib-roughened channel
(Thrust #23's second, structurally distinct repeating geometry).  It contains
*no* rib-specific or hill-specific logic: it takes a list of wall-normal station
profiles (distance-from-wall column y, streamwise velocity U, resolved shear
stress uv) plus the molecular viscosity nu, and returns the a-priori ODE wall-
model verdict exactly as ``diagnostic_test_corrected.py`` computes it for hills:

  * the manuscript's PRODUCTION ODE solver  ``ode_wall_model.predict_tau_w``
    (algebraic mixing-length closure, identical to every hill number);
  * a controlled mixing-length shooting solver  (closure-variation control);
  * a controlled exact-DNS/LES-stress shooting solver  (G3 closure-independence);
  * the cancellation parameter   eps(x) = |tau_w| / (|dp/dx| * y_m)   and the
    domain fractions f(eps<1), f(eps<0.1)  (manuscript eq:cancellation);
  * R^2(tau_w), relRMS(tau_w) per closure, separation-sign accuracy.

The closures below are copied VERBATIM from ``diagnostic_test_corrected.py``
(the corrected, hill-surface-aware instrument that yields the canonical hill
R^2=-47.7), with the ONLY change being that nu, KAPPA, A_PLUS, N_GRID, Y_IDX are
passed in rather than module globals -- so the rib (nu=2.381e-4) and the hill
(nu=1/5600) are scored on byte-identical numerics.  ``prove_instrument.py``
asserts this module reproduces the stored hill verdict bit-for-bit; that is the
non-tautology guarantee (methodology sec.6): the rib number, when harvested,
comes off the SAME falsifiable instrument as the hills, with no rib-specific
freedom.

a-priori only.  No fabrication: this module computes, it does not assert data.
"""
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # codes/
# the manuscript's production ODE solver -- imported, not re-implemented
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w as _main_predict_tau_w     # noqa: E402

# numerics identical to diagnostic_test_corrected.py / dose_response_xiao.py
KAPPA, A_PLUS, N_GRID = 0.41, 26.0, 500
Y_IDX_DEFAULT = 10


# ---------------------------------------------------------------------------
# controlled shooting closures  (copied verbatim from diagnostic_test_corrected,
# nu-parametrised so the same code scores hills AND ribs)
# ---------------------------------------------------------------------------
def _solve_controlled(tau_w_guess, y_match, dp_dx, mode, prof, nu):
    eta = np.linspace(0, 1, N_GRID)
    y_grid = y_match * eta ** 1.5
    if mode == "mixing_length":
        u_tau = np.sqrt(max(abs(tau_w_guess), 1e-30))
        y_plus = y_grid * u_tau / nu
        D = 1.0 - np.exp(-y_plus / A_PLUS)
        l_m2 = (KAPPA * y_grid * D) ** 2
        F = tau_w_guess + dp_dx * y_grid
        absF, signF = np.abs(F), np.sign(F)
        disc = nu * nu + 4.0 * l_m2 * absF
        viscous = l_m2 < 1e-30
        l_m2_safe = np.where(viscous, 1.0, l_m2)
        absS = (-nu + np.sqrt(disc)) / (2.0 * l_m2_safe)
        absS = np.where(viscous, absF / nu, absS)
        dUdy = signF * absS
    elif mode == "dns_stress":
        uv_interp = np.interp(y_grid, prof["y"], prof["uv"])
        dUdy = (tau_w_guess + dp_dx * y_grid + uv_interp) / nu
    elif mode == "dns_stress_total":
        # SGS-completed substitution (G3, sharp rib): the resolved Reynolds
        # shear stress <u'v'>_res is augmented by the modelled SGS shear stress
        # <u'v'>_sgs = -nu_sgs dU/dy so that the FULL turbulent stress the LES
        # carries (resolved + subgrid) enters the balance.  prof["uv_total"]
        # already holds <u'v'>_res + <u'v'>_sgs harvested on the LES grid.
        uv_interp = np.interp(y_grid, prof["y"], prof["uv_total"])
        dUdy = (tau_w_guess + dp_dx * y_grid + uv_interp) / nu
    else:
        return np.nan
    dy = np.diff(y_grid)
    return np.sum(0.5 * (dUdy[:-1] + dUdy[1:]) * dy)


def _predict_controlled(U_match, y_match, dp_dx, mode, prof, nu):
    if abs(U_match) < 1e-15:
        return 0.0
    try:
        return brentq(lambda tw: _solve_controlled(tw, y_match, dp_dx, mode, prof, nu)
                      - U_match, -10.0, 10.0, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan


def _r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def evaluate(profs, nu, Y_IDX=Y_IDX_DEFAULT, closures=("standard_ml",
                                                       "controlled_ml",
                                                       "controlled_dns")):
    """Score a set of wall-normal station profiles a-priori.

    Each ``prof`` in ``profs`` is a dict with:
        y        distance-from-wall column (y[0]=0 origin), increasing
        U        streamwise velocity on that column
        uv       resolved Reynolds shear stress on that column (for dns_stress)
        tau_w    reference (truth) wall stress at the station
        dpdx     near-wall streamwise pressure gradient at the station
    Returns a dict of arrays + scalar R^2/relRMS per closure + the eps field.
    """
    n = len(profs)
    gt = np.full(n, np.nan)
    dpdx = np.full(n, np.nan)
    y_m = np.full(n, np.nan)
    is_sep = np.zeros(n, bool)
    valid = np.zeros(n, bool)
    pred = {k: np.full(n, np.nan) for k in closures}

    for i, pr in enumerate(profs):
        if Y_IDX >= len(pr["y"]):
            continue
        ym, Um, dx = pr["y"][Y_IDX], pr["U"][Y_IDX], pr["dpdx"]
        if ym <= 0 or not np.isfinite(Um):
            continue
        if "standard_ml" in closures:
            pred["standard_ml"][i] = _main_predict_tau_w(Um, ym, dx, nu)
        if "controlled_ml" in closures:
            pred["controlled_ml"][i] = _predict_controlled(Um, ym, dx, "mixing_length", pr, nu)
        if "controlled_dns" in closures:
            pred["controlled_dns"][i] = _predict_controlled(Um, ym, dx, "dns_stress", pr, nu)
        if "controlled_dns_total" in closures:
            pred["controlled_dns_total"][i] = _predict_controlled(
                Um, ym, dx, "dns_stress_total", pr, nu)
        gt[i] = pr["tau_w"]
        dpdx[i] = dx
        y_m[i] = ym
        is_sep[i] = bool(pr["tau_w"] < 0)
        valid[i] = True

    # cancellation parameter eps = |tau_w| / (|dp/dx| y_m)  (manuscript def.)
    denom = np.abs(dpdx) * y_m
    eps = np.full(n, np.nan)
    good = valid & (denom > 1e-30)
    eps[good] = np.abs(gt[good]) / denom[good]
    eps_v = eps[np.isfinite(eps)]

    out = dict(tau_w_ref=gt, dpdx=dpdx, y_m=y_m, eps=eps, is_separated=is_sep,
               valid_mask=valid, n_profiles=n, Y_IDX=Y_IDX, nu=float(nu),
               eps_median=float(np.median(eps_v)) if eps_v.size else np.nan,
               frac_eps_lt1=float(np.mean(eps_v < 1.0)) if eps_v.size else np.nan,
               frac_eps_lt0p1=float(np.mean(eps_v < 0.1)) if eps_v.size else np.nan)
    for k in closures:
        m = valid & np.isfinite(pred[k])
        p, t = pred[k][m], gt[m]
        out[k] = pred[k]
        out[f"{k}_r2"] = float(_r2(p, t)) if m.sum() else np.nan
        rms_tau = np.sqrt(np.mean(t ** 2)) if m.sum() else np.nan
        out[f"{k}_relRMS"] = float(np.sqrt(np.mean((p - t) ** 2)) / rms_tau) \
            if (m.sum() and rms_tau > 0) else np.nan
        out[f"{k}_n"] = int(m.sum())
    return out
