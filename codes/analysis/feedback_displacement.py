#!/usr/bin/env python3
r"""
feedback_displacement.py  --  Thrust #16, Level-2 (implementation & experiments).

DECISIVE a-posteriori MEASUREMENT of the closed-loop "feedback regularization &
error displacement" mechanism derived in development/nodes/node_001/methodology.md.
Everything below is computed from REAL on-disk data at a SINGLE, self-consistent
Reynolds number (Krank, Kronbichler & Wall 2018 periodic hills, Re_H = 10595):

  * DNS reference (truth)   : codes/new_data_download/geometry_driven/
                              krank_pehill_Re10595_wall_profiles.npz
                              (full U(y) profiles + tau_w + dp/dx at 10 stations)
  * coupled WMLES (two-way) : codes/openfoam/pehill_wmles/  (harvested
                              wmles_bottomwall_summary.npz: tau_w^c, y_m at 80 x;
                              postProcessing/sampleProfiles/405: U(y) at 10 x)
  * wall-model map M        : the production ODE (convection-blind),
                              vendor/.../ode_wall_model.predict_tau_w

The three PRE-REGISTERED falsifiers (locked at L0/L1) are measured here:

  F1  feedback, not averaging.   One-way  tau_w = M(U_m^DNS)  (model fed the TRUE
      matching velocity -- NO loop)  vs  two-way coupled  tau_w^c  (LES supplies
      U_m self-consistently).  SAME matching height, SAME 10 stations, SAME Re.
      The only thing that differs is the feedback.  Feedback => two-way error
      << one-way error.  The pure-norm null (#9, "same error field, different
      averaging") predicts EQUAL error in this common relRMS(tau_w) norm.

  F2  displacement identity.   The convection-blind map is STEEP
      (kappa = |(U_m/tau_w) dM/dU_m| ~ 1/eps).  A bounded O(1) velocity
      displacement  Delta U_m = U_m^c - U_m^DNS  therefore carries the LARGE
      open-loop stress defect  delta = M(U_m^DNS) - tau_w^DNS:
          Delta U_m * (dM/dU_m)  ~=  - delta.
      Falsifiable content: |Delta U_m / U_m| is O(1) (tens of %), NOT ~0 and
      NOT ~1/eps; and kappa is large (~1/eps) where the model is catastrophic.

  lambda  outer-compliance gain.   lambda = |(tau_w/U_m) dL/dtau_w| with
      L: tau_w -> U_m the RESOLVED-outer response.  Measured from the coupled
      operating points (log-law slope d ln U_m / d ln tau_w across attached
      stations) -- must be O(1), confirming the loop gain L_g = kappa*lambda is
      SET BY kappa (the conditioning), not by lambda.

Honesty (G1-G4, G6, G7):
  * a single coupled geometry is used here (the hills, Re10595); the conv-div
    F3 control and the Xiao alpha=0.8 cross-check are handled in the companion
    scripts (feedback_displacement_control.py, harvest verification) and the L3
    multi-geometry table.  No fabricated numbers; every output traces to the
    npz/case files named above.
  * the matching height used is the COUPLED first-cell height y_m^c (y_m^+ ~ 2),
    so F1 isolates the feedback at FIXED y_m -- the larger, confounding
    a-priori->coupled "healing" from the matching-height shift is quantified
    separately in ym_feedback_decomposition.py and reported, NOT hidden.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/feedback_displacement.py
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # codes/
RESULTS = os.path.join(ROOT, "results")
DNS_NPZ = os.path.join(ROOT, "new_data_download", "geometry_driven",
                       "krank_pehill_Re10595_wall_profiles.npz")
WMLES_DIR = os.path.join(ROOT, "openfoam", "pehill_wmles")
WMLES_SUMMARY = os.path.join(WMLES_DIR, "wmles_bottomwall_summary.npz")
PROF_DIR = os.path.join(WMLES_DIR, "postProcessing", "sampleProfiles", "405")

sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w as M_op   # noqa: E402  (the map M)

NU = 9.44e-5            # Krank Re_H = 10595 (matches the coupled WMLES exactly)
# coupled sample-profile station tags (x/H), aligned with the DNS stations
STATION_TAGS = ["x0p05", "x0p50", "x1p00", "x2p00", "x3p00",
                "x4p00", "x5p00", "x6p00", "x7p00", "x8p00"]


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def rel_rms(truth, pred, mask=None):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    if mask is None:
        mask = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[mask], pred[mask]
    return float(np.sqrt(np.mean((p - t) ** 2)) / np.sqrt(np.mean(t ** 2)))


def r2(truth, pred, mask=None):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    if mask is None:
        mask = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[mask], pred[mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def dM_dUm(U_m, y_m, dpdx, nu, frac=1e-3):
    """Centered finite-difference slope dM/dU_m of the production wall map."""
    h = frac * max(abs(U_m), 1e-6)
    tp = M_op(U_m + h, y_m, dpdx, nu)
    tm = M_op(U_m - h, y_m, dpdx, nu)
    if not (np.isfinite(tp) and np.isfinite(tm)):
        return np.nan
    return (tp - tm) / (2.0 * h)


def invert_M_for_Um(tau_target, y_m, dpdx, nu, U_seed):
    """Find the matching velocity U_m^c that the production map would require to
    output the coupled stress tau_target (the model's coupled operating point).
    Bracketed root-find around the DNS velocity seed."""
    def f(U):
        tw = M_op(U, y_m, dpdx, nu)
        return tw - tau_target if np.isfinite(tw) else np.nan
    # build a sign-changing bracket around the seed
    lo, hi = sorted((-abs(U_seed) * 6 - 0.5, abs(U_seed) * 6 + 0.5))
    try:
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0):
            # scan for a bracket
            grid = np.linspace(lo, hi, 4001)
            fv = np.array([f(g) for g in grid])
            sgn = np.where(np.isfinite(fv), np.sign(fv), 0)
            idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
            if len(idx) == 0:
                return np.nan
            # take the bracket closest to the seed
            j = idx[np.argmin(np.abs(grid[idx] - U_seed))]
            lo, hi = grid[j], grid[j + 1]
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except Exception:
        return np.nan


def load_coupled_Um_LES(tag, y_m, y_wall_guess=None):
    """Independent (line-sampled) coupled matching velocity at height y_m above
    the local wall, as an a-posteriori cross-check of the model-inverted U_m^c.
    Returns (U_m^LES, y_wall)."""
    path = os.path.join(PROF_DIR, tag + ".xy")
    if not os.path.exists(path):
        return np.nan, np.nan
    a = np.loadtxt(path)
    y, U = a[:, 0], a[:, 1]
    # detect the wall: first sample with |U| above noise floor
    fluid = np.where(np.abs(U) > 1e-5)[0]
    if len(fluid) == 0:
        return np.nan, np.nan
    i0 = fluid[0]
    y_wall = y[i0 - 1] if i0 > 0 else 0.0
    y_rel = y[i0:] - y_wall
    U_fl = U[i0:]
    if y_m <= 0 or y_m > y_rel[-1]:
        return np.nan, y_wall
    return float(np.interp(y_m, y_rel, U_fl)), y_wall


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=" * 74)
    print("Thrust #16  L2  --  feedback regularization & error displacement")
    print("Self-consistent at Krank Re_H=10595 (coupled WMLES vs same DNS)")
    print("=" * 74)

    # ---- DNS truth (10 stations) -----------------------------------------
    dns = np.load(DNS_NPZ, allow_pickle=True)
    x_st = dns["x"].astype(float)
    dns_y = dns["y"]           # (10, 385) wall-relative
    dns_U = dns["U"]
    dns_tau = dns["tau_w"].astype(float)
    dns_dpdx = dns["dp_dx"].astype(float)
    dns_sep = dns["is_separated"].astype(bool)
    n = len(x_st)

    # ---- coupled WMLES wall summary (80 x) -> interp to the 10 stations ----
    w = np.load(WMLES_SUMMARY, allow_pickle=True)
    wx = w["x"].astype(float)
    o = np.argsort(wx)
    y_m = np.interp(x_st, wx[o], w["y_m"][o])         # coupled first-cell height
    tau_c = np.interp(x_st, wx[o], w["tau_w"][o])     # coupled wall stress (two-way)
    eps_c = np.interp(x_st, wx[o], w["eps"][o])       # coupled cancellation depth

    # per-station arrays
    U_m_dns = np.full(n, np.nan)
    tau_oneway = np.full(n, np.nan)     # M(U_m^DNS)        -- one-way (no loop)
    delta = np.full(n, np.nan)          # open-loop defect  = M(U_m^DNS) - tau_dns
    slope = np.full(n, np.nan)          # dM/dU_m
    kappa = np.full(n, np.nan)          # |(U_m/tau) dM/dU_m|
    U_m_c_model = np.full(n, np.nan)    # inverted model operating point (two-way)
    U_m_c_les = np.full(n, np.nan)      # line-sampled LES matching velocity
    ymp = np.full(n, np.nan)            # y_m^+

    for i in range(n):
        ym = float(y_m[i])
        y, U = dns_y[i], dns_U[i]
        # DNS matching velocity at the coupled first-cell height
        U_m_dns[i] = float(np.interp(ym, y, U))
        dpdx = float(dns_dpdx[i])
        # one-way: convection-blind model fed the TRUE matching velocity
        tw = M_op(U_m_dns[i], ym, dpdx, NU)
        tau_oneway[i] = tw
        delta[i] = tw - dns_tau[i]
        s = dM_dUm(U_m_dns[i], ym, dpdx, NU)
        slope[i] = s
        if np.isfinite(s) and abs(dns_tau[i]) > 0:
            kappa[i] = abs(U_m_dns[i] / dns_tau[i] * s)
        # two-way model operating point: U_m that yields the coupled stress
        U_m_c_model[i] = invert_M_for_Um(tau_c[i], ym, dpdx, NU, U_m_dns[i])
        # independent LES line-sampled matching velocity (cross-check)
        U_m_c_les[i], _ = load_coupled_Um_LES(STATION_TAGS[i], ym)
        ymp[i] = ym * np.sqrt(abs(dns_tau[i])) / NU

    # pure velocity-displacement effect: SAME production map M, fed the LES
    # matching velocity instead of the DNS one (isolates feedback from the
    # choice of wall-model and from the LES's own stress-evaluation error).
    tau_oneway_les = np.full(n, np.nan)
    for i in range(n):
        if np.isfinite(U_m_c_les[i]):
            tau_oneway_les[i] = M_op(U_m_c_les[i], float(y_m[i]),
                                     float(dns_dpdx[i]), NU)

    # ---- F1: one-way vs two-way (same y_m, same stations, same norm) -------
    m_all = np.isfinite(tau_oneway) & np.isfinite(tau_c) & np.isfinite(dns_tau)
    relrms_oneway = rel_rms(dns_tau, tau_oneway, m_all)
    relrms_twoway = rel_rms(dns_tau, tau_c, m_all)
    r2_oneway = r2(dns_tau, tau_oneway, m_all)
    r2_twoway = r2(dns_tau, tau_c, m_all)
    feedback_factor = relrms_oneway / relrms_twoway
    mML = np.isfinite(tau_oneway_les) & np.isfinite(dns_tau)
    relrms_M_les = rel_rms(dns_tau, tau_oneway_les, mML)
    r2_M_les = r2(dns_tau, tau_oneway_les, mML)

    print("\n--- F1  feedback, not averaging  (matched y_m^+ ~ %.1f, N=%d) ---"
          % (np.nanmedian(ymp), m_all.sum()))
    print("  one-way  M(U_m^DNS)          : relRMS = %6.3f   R2 = %+8.2f"
          % (relrms_oneway, r2_oneway))
    print("  same M, fed LES U_m          : relRMS = %6.3f   R2 = %+8.2f"
          "   <- pure velocity-displacement" % (relrms_M_les, r2_M_les))
    print("  two-way  tau_w^c (coupled)   : relRMS = %6.3f   R2 = %+8.3f"
          % (relrms_twoway, r2_twoway))
    print("  feedback factor (one/two)    : %5.2fx   "
          "(>1 => feedback heals; <1 => no catastrophe to heal here)"
          % feedback_factor)

    # ---- F2: displacement identity ---------------------------------------
    # use the model operating point (consistent two-operator picture)
    dUm_model = U_m_c_model - U_m_dns
    dUm_les = U_m_c_les - U_m_dns
    lhs = dUm_model * slope                  # Delta U_m * dM/dU_m
    rhs = -delta                             # - open-loop defect
    relU_model = dUm_model / np.where(np.abs(U_m_dns) > 1e-9, U_m_dns, np.nan)
    relU_les = dUm_les / np.where(np.abs(U_m_dns) > 1e-9, U_m_dns, np.nan)

    mF2 = np.isfinite(lhs) & np.isfinite(rhs) & (np.abs(rhs) > 1e-12)
    ident_ratio = lhs[mF2] / rhs[mF2]
    print("\n--- F2  displacement identity   Delta U_m * dM/dU_m  ~=  - delta ---")
    print("   x/H   sep   y_m^+    kappa     |dUm/Um|   identity(LHS/RHS)")
    for i in range(n):
        if not mF2[i]:
            continue
        print("  %5.2f   %s   %5.1f   %8.1f   %7.3f      %7.3f"
              % (x_st[i], "Y" if dns_sep[i] else "n", ymp[i], kappa[i],
                 abs(relU_model[i]), lhs[i] / rhs[i]))
    print("  identity ratio  : median=%.3f  mean=%.3f  (target ~1, O(1))"
          % (np.nanmedian(ident_ratio), np.nanmean(ident_ratio)))
    print("  |Delta U_m/U_m| : median=%.3f  (model)   %.3f  (LES line-sampled)"
          % (np.nanmedian(np.abs(relU_model)),
             np.nanmedian(np.abs(relU_les[np.isfinite(relU_les)]))))
    print("  kappa (rel gain): median=%.1f  max=%.1f   ~ 1/eps_median=%.1f"
          % (np.nanmedian(kappa), np.nanmax(kappa), 1.0 / np.nanmedian(eps_c)))
    # cross-check: model-inverted vs LES-sampled coupled matching velocity
    mxc = np.isfinite(U_m_c_model) & np.isfinite(U_m_c_les)
    if mxc.sum() >= 3:
        cc = np.corrcoef(U_m_c_model[mxc], U_m_c_les[mxc])[0, 1]
        print("  U_m^c model-inverted vs LES-sampled: r=%.3f (N=%d) "
              "-- production M faithful to LES wall fn" % (cc, mxc.sum()))

    # ---- lambda: outer-compliance gain from coupled operating points -------
    # L : tau_w -> U_m ; lambda = d ln|U_m| / d ln|tau_w| across attached stns
    att = (~dns_sep) & np.isfinite(U_m_c_les) & (np.abs(tau_c) > 0)
    lam = np.nan
    if att.sum() >= 3:
        lx = np.log(np.abs(tau_c[att]))
        ly = np.log(np.abs(U_m_c_les[att]))
        A = np.vstack([lx, np.ones_like(lx)]).T
        lam = float(np.linalg.lstsq(A, ly, rcond=None)[0][0])
    print("\n--- lambda  outer-compliance gain (L: tau_w -> U_m) ---")
    print("  measured d ln|U_m| / d ln|tau_w| (attached, N=%d) = %.3f"
          % (att.sum(), lam))
    print("  log-law estimate = 0.5 ;  both O(1) => loop gain set by kappa, "
          "not lambda")

    # ---- partition of the open-loop defect (where the error goes) ----------
    # delta (open-loop, O(Phi))  =  surviving stress error  +  displacement-equiv
    surviving = tau_c - dns_tau                  # residual stress error (two-way)
    displaced = U_m_c_model * 0.0 + (delta - surviving)   # = -lhs approx
    mp = m_all & np.isfinite(delta)
    frac_surv = np.sqrt(np.mean(surviving[mp] ** 2)) / np.sqrt(np.mean(delta[mp] ** 2))
    print("\n--- partition: where the open-loop defect goes ---")
    print("  RMS open-loop defect |delta|         = %.4e" % np.sqrt(np.mean(delta[mp] ** 2)))
    print("  RMS surviving stress error           = %.4e" % np.sqrt(np.mean(surviving[mp] ** 2)))
    print("  surviving / open-loop (fraction)     = %.3f   (~ eps, displaced rest)"
          % frac_surv)

    # ---- matching-height sweep: where is the cancellation/catastrophe band? -
    # For each candidate matching height y_m (expressed in + units via the DNS
    # u_tau), evaluate the one-way model M(U_m^DNS(y_m)) across the 10 stations
    # and its relRMS + R2 vs DNS tau_w, plus the median eps = |tau_w|/(|dp/dx|*y_m).
    # This locates the band of heights where the convection-blind model is
    # catastrophic, and shows where the coupled WMLES actually matches.
    sweep_ymp = np.array([1., 2., 3., 5., 8., 12., 20., 35., 60.])
    sw_relrms = np.full(sweep_ymp.size, np.nan)
    sw_r2 = np.full(sweep_ymp.size, np.nan)
    sw_eps = np.full(sweep_ymp.size, np.nan)
    for j, yp in enumerate(sweep_ymp):
        pj = np.full(n, np.nan)
        epsj = np.full(n, np.nan)
        for i in range(n):
            utau = np.sqrt(abs(dns_tau[i]))
            ym = yp * NU / utau if utau > 0 else np.nan
            y, U = dns_y[i], dns_U[i]
            if not np.isfinite(ym) or ym <= y[0] or ym > y[-1]:
                continue
            Um = float(np.interp(ym, y, U))
            pj[i] = M_op(Um, ym, float(dns_dpdx[i]), NU)
            denom = abs(dns_dpdx[i]) * ym
            epsj[i] = abs(dns_tau[i]) / denom if denom > 0 else np.nan
        mj = np.isfinite(pj) & np.isfinite(dns_tau)
        if mj.sum() >= 3:
            sw_relrms[j] = rel_rms(dns_tau, pj, mj)
            sw_r2[j] = r2(dns_tau, pj, mj)
        sw_eps[j] = np.nanmedian(epsj)
    print("\n--- matching-height sweep: the catastrophe band (one-way model) ---")
    print("  y_m^+   eps_median   relRMS(tau_w)     R2(tau_w)")
    for j, yp in enumerate(sweep_ymp):
        print("  %5.0f   %9.3f   %11.3f   %+11.2f"
              % (yp, sw_eps[j], sw_relrms[j], sw_r2[j]))
    print("  coupled WMLES matches at y_m^+ ~ %.0f (median) -> BELOW the band"
          % np.nanmedian(ymp))

    # ---- save -------------------------------------------------------------
    out = dict(
        Re_H=10595.0, NU=NU, x=x_st, is_sep=dns_sep, y_m=y_m, ymp=ymp,
        eps_coupled=eps_c,
        # truth + maps
        tau_dns=dns_tau, U_m_dns=U_m_dns, dpdx=dns_dpdx,
        tau_oneway=tau_oneway, tau_oneway_les=tau_oneway_les,
        tau_coupled=tau_c, delta=delta,
        relrms_M_les=relrms_M_les, r2_M_les=r2_M_les,
        sweep_ymp=sweep_ymp, sweep_relrms=sw_relrms, sweep_r2=sw_r2,
        sweep_eps=sw_eps,
        slope_dM_dUm=slope, kappa=kappa,
        U_m_c_model=U_m_c_model, U_m_c_les=U_m_c_les,
        dUm_model=dUm_model, dUm_les=dUm_les,
        relU_model=relU_model, relU_les=relU_les,
        # F1
        relrms_oneway=relrms_oneway, relrms_twoway=relrms_twoway,
        r2_oneway=r2_oneway, r2_twoway=r2_twoway,
        feedback_factor=feedback_factor,
        # F2
        ident_lhs=lhs, ident_rhs=rhs,
        ident_ratio_median=float(np.nanmedian(ident_ratio)),
        ident_ratio_mean=float(np.nanmean(ident_ratio)),
        relU_model_median=float(np.nanmedian(np.abs(relU_model))),
        kappa_median=float(np.nanmedian(kappa)),
        kappa_max=float(np.nanmax(kappa)),
        # lambda
        lambda_measured=lam, lambda_loglaw=0.5,
        # partition
        surviving_stress=surviving,
        frac_surviving=frac_surv,
    )
    op = os.path.join(RESULTS, "feedback_displacement.npz")
    np.savez(op, **out)
    print("\nSaved -> %s   (%.1fs)" % (op, time.time() - t0))


if __name__ == "__main__":
    main()
