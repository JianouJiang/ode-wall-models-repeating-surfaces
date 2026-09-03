#!/usr/bin/env python3
"""
Thrust #19 L3 — Results & analysis for the Reynolds-asymptotic deepening.

This script CONSUMES the L2 result files (it re-runs no DNS extraction) and
produces the L3 *analysis* products:

  1. EXTRAPOLATION BAND.  A Monte-Carlo power-law fit to the five-DNS eps_C(Re)
     series that PROPAGATES each point's bootstrap 95% CI, giving an honest
     (wide) predicted band for eps_C and the ODE error amplification 1/eps_C at
     engineering Reynolds numbers.  The engineering-Re numbers are flagged
     EXTRAPOLATION via the derived law -- never a simulated claim (G1/G4).

  2. TWO-AXIS REGIME MAP.  Folds the L2 Reynolds axis together with the #11
     steepness axis (dose_response_xiao.npz, Re=5600, 29 cases).  The real DNS
     coverage is L-shaped: a Reynolds row (canonical hill, 5 Re) and a steepness
     column (29 Xiao cases at Re=5600).  The dangerous corner -- engineering Re
     AND O(delta) pitch -- is the PREDICTED intersection of two independently
     validated 1-D trends, not a filled DNS grid.  Stored honestly as such.

  3. F1-F4 VERDICT TABLE with CIs, assembled from the L2 npz.

Inputs  (read-only):
  development/nodes/node_003/reynolds_scaling.npz   (L2 Reynolds series)
  codes/results/dose_response_xiao.npz              (#11 steepness axis, Re5600)
Output:
  development/nodes/node_004/results_reynolds_l3.npz

A-priori only.  No CFD, no fabrication.  OMP_NUM_THREADS=2.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RS   = os.path.join(ROOT, "development", "nodes", "node_003", "reynolds_scaling.npz")
DOSE = os.path.join(ROOT, "codes", "results", "dose_response_xiao.npz")
OUTD = os.path.join(ROOT, "development", "nodes", "node_004")
os.makedirs(OUTD, exist_ok=True)
OUT  = os.path.join(OUTD, "results_reynolds_l3.npz")

rng = np.random.default_rng(20260531)   # fixed seed -> reproducible band

# ----------------------------------------------------------------------------
# Load the L2 Reynolds series
# ----------------------------------------------------------------------------
d = np.load(RS, allow_pickle=True)
Re   = d["Re_pts"].astype(float)           # [700,1400,2800,5600,10595]
eps  = d["epsC_pts"].astype(float)         # closure-free residual/dropped-convection
lo   = d["epsC_lo"].astype(float)          # bootstrap 95% CI low
hi   = d["epsC_hi"].astype(float)          # bootstrap 95% CI high
cf   = d["cf_pts"].astype(float)
src  = json.loads(str(d["series_src"]))
slope_re_pt = float(d["slope_re"])         # point-estimate power-law exponent
r2_re_pt    = float(d["r2_re"])
eps_grid    = json.loads(str(d["eps_grid"]))   # outer-lambda pressure-eps (Krank pair)

print("=== L2 five-Reynolds series (input) ===")
for r, e, l, h, c, s in zip(Re, eps, lo, hi, cf, src):
    print(f"  Re={r:7.0f}  eps_C={e:6.3f}  CI[{l:5.2f},{h:6.2f}]  Cf={c*1e3:5.2f}e-3  ({s})")
print(f"  point power-law: eps_C ~ Re^{slope_re_pt:.3f}  (R^2={r2_re_pt:.3f})")

# ----------------------------------------------------------------------------
# 1.  Monte-Carlo extrapolation band (propagate each point's bootstrap CI)
# ----------------------------------------------------------------------------
# Each eps_C point is modelled lognormal so that its 2.5/97.5 percentiles match
# the stored bootstrap CI [lo,hi].  A power law is refit on every draw; the
# spread of the predictions at engineering Re is the honest extrapolation band.
logRe = np.log(Re)
mu_ln = np.log(eps)
# lognormal sigma so that exp(mu +/- 1.96 sigma) = [lo, hi] (use the geometric CI)
sig_ln = (np.log(hi) - np.log(lo)) / (2.0 * 1.959963985)

N = 20000
Re_eval = np.array([700., 1400., 2800., 5600., 10595.,
                    2.0e4, 5.0e4, 1.0e5, 1.0e6])     # last four = EXTRAPOLATION
logEval = np.log(Re_eval)

slopes = np.empty(N)
intercepts = np.empty(N)
pred = np.empty((N, Re_eval.size))
for i in range(N):
    y = rng.normal(mu_ln, sig_ln)                  # one noisy realisation of log eps_C
    A = np.vstack([np.ones_like(logRe), logRe]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)   # [intercept, slope]
    intercepts[i] = coef[0]
    slopes[i] = coef[1]
    pred[i] = np.exp(coef[0] + coef[1] * logEval)

q = lambda a, p: np.percentile(a, p)
slope_med = float(np.median(slopes))
slope_ci  = (float(q(slopes, 2.5)), float(q(slopes, 97.5)))
band_med  = np.median(pred, axis=0)
band_lo   = np.percentile(pred, 2.5, axis=0)
band_hi   = np.percentile(pred, 97.5, axis=0)

print("\n=== Monte-Carlo power-law fit (CI-propagated, N=%d) ===" % N)
print(f"  slope  = {slope_med:.3f}  95% CI [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
print(f"  (point-estimate slope was {slope_re_pt:.3f})")
print("  predicted eps_C and ODE error amplification 1/eps_C:")
print("   Re        eps_C(med)   eps_C 95%-band         1/eps_C(med)  1/eps_C 95%-band   note")
err_med = 1.0 / band_med
err_lo  = 1.0 / band_hi          # small eps_C -> large 1/eps_C : invert the band ends
err_hi  = 1.0 / band_lo
for j, r in enumerate(Re_eval):
    note = "DNS" if r <= 10595 else "EXTRAPOLATION"
    print(f"  {r:8.0f}  {band_med[j]:8.3f}   [{band_lo[j]:6.3f},{band_hi[j]:7.3f}]   "
          f"{err_med[j]:8.2f}   [{err_lo[j]:6.2f},{err_hi[j]:7.2f}]   {note}")

# slope strictly negative across the whole posterior?  (asymptotic, not chance)
frac_negative = float(np.mean(slopes < 0.0))
print(f"  fraction of MC fits with NEGATIVE slope (deepening) = {frac_negative*100:.2f}%")

# ----------------------------------------------------------------------------
# 2.  Two-axis regime map  (Reynolds row  x  steepness column)
# ----------------------------------------------------------------------------
ds = np.load(DOSE, allow_pickle=True)
st_hLx   = ds["agg_cv_h_over_Lx"].astype(float)     # steepness h/Lx
st_eps   = ds["agg_eps_median"].astype(float)       # pressure-based eps (Re5600)
st_r2    = ds["agg_r2"].astype(float)               # ODE R^2(tau_w) (Re5600)
st_Lsep  = ds["agg_cv_Lsep_over_delta"].astype(float)
st_alpha = ds["agg_cv_alpha"].astype(float)
Re_dose  = float(ds["Re_b"])

# Canonical periodic-hill (Froehlich/Breuer/Krank) steepness anchor for the
# Reynolds row: pitch L_x = 9h between crests -> h/L_x = 1/9.  Documented anchor,
# NOT a fitted quantity; the Reynolds family all share this geometry.
H_OVER_LX_CANON = 1.0 / 9.0

# Reynolds row points (5 DNS at canonical steepness), severity = eps_C
row_Re    = Re.copy()
row_hLx   = np.full_like(Re, H_OVER_LX_CANON)
row_sev   = eps.copy()                  # eps_C (closure-free) -- the Reynolds diagnostic

# Steepness column points (29 Xiao at Re5600), severity = pressure eps_median
col_Re    = np.full_like(st_eps, Re_dose)
col_hLx   = st_hLx
col_sev   = st_eps                      # pressure eps -- the steepness diagnostic

print("\n=== Regime map ===")
print(f"  Reynolds row : {row_Re.size} DNS @ h/Lx={H_OVER_LX_CANON:.3f}, "
      f"Re {row_Re.min():.0f}-{row_Re.max():.0f}, eps_C {row_sev.min():.2f}-{row_sev.max():.2f}")
print(f"  Steepness col: {col_Re.size} Xiao DNS @ Re={Re_dose:.0f}, "
      f"h/Lx {col_hLx.min():.3f}-{col_hLx.max():.3f}, eps_med {col_sev.min():.2f}-{col_sev.max():.2f}, "
      f"R2 {st_r2.min():.1f}..{st_r2.max():.1f}")
print("  NOTE: the two families use two diagnostics (eps_C convective / eps pressure);")
print("        both deepen.  No filled Re x steepness DNS grid is claimed (honesty).")

# Critical matching fraction (the usable window) from the MEASURED pressure-eps.
# eps = tau_w/(|dp/dx|*y_m) is EXACTLY proportional to 1/lambda (y_m=lambda*delta)
# at a fixed station, so eps(lambda) = eps(lambda_ref)*(lambda_ref/lambda).  The
# catastrophe sets in at eps = eps_c ~ 0.2 (manuscript), giving the largest
# tolerable matching fraction
#     lambda_c(Re) = lambda_ref * eps(lambda_ref; Re) / eps_c .
# This uses the MEASURED geometric prefactor (eps/C_f ~ 100, F2), NOT the bare
# 1/(2 lambda); available where pressure-eps exists (the two Krank Re).  It
# SHRINKS with Re because eps(lambda_ref) ~ C_f drains -> the usable matching
# window narrows toward engineering Re.
eps_c = 0.2
lam_ref = 0.10
eps_ref = {5600.0: eps_grid["outer_0.1"][0], 10595.0: eps_grid["outer_0.1"][1]}
lam_c_5600  = lam_ref * eps_ref[5600.0]  / eps_c
lam_c_10595 = lam_ref * eps_ref[10595.0] / eps_c
print("  usable matching window lambda_c = lambda_ref*eps(0.1)/eps_c (measured prefactor):")
print(f"    Re=  5600  eps(0.1)={eps_ref[5600.0]:.3f}  lambda_c={lam_c_5600:.4f}")
print(f"    Re= 10595  eps(0.1)={eps_ref[10595.0]:.3f}  lambda_c={lam_c_10595:.4f}")
print(f"    window narrows by factor {lam_c_10595/lam_c_5600:.3f} over Re 5600->10595")

# ----------------------------------------------------------------------------
# 3.  F1-F4 verdict table (assembled from L2 npz)
# ----------------------------------------------------------------------------
verdicts = {
    "F1_5Re_deepening": {
        "claim": "eps_C decreases monotonically across 5 DNS (Re 700->10595)",
        "evidence": f"eps_C = {list(np.round(eps,3))} (strictly decreasing)",
        "pass": bool(d["F1_5re_pass"]) and bool(np.all(np.diff(eps) < 0)),
    },
    "F1_outer_lambda": {
        "claim": "eps falls with Re at every fixed outer matching fraction",
        "evidence": {k: [round(v[0],3), round(v[1],3)] for k, v in eps_grid.items()},
        "pass": bool(d["F1_pass"]),
    },
    "F2_eps_over_Cf_invariant": {
        "claim": "pressure-eps/C_f is Re-invariant within O(1) (Krank pair)",
        "evidence": f"eps/Cf = {list(np.round(d['eps_press_over_cf'],1))}, spread {float(d['F2_spread']):.3f}",
        "pass": bool(d["F2_pass"]),
    },
    "F3_sign_stable": {
        "claim": "sign of d eps/d Re stable across the outer grid (not the y=0 artifact)",
        "evidence": "all three outer lambda fall (see F1_outer)",
        "pass": bool(d["F3_pass"]),
    },
    "F4_separation_specific": {
        "claim": "deepening is separation-specific: separated eps falls, attached eps rises",
        "evidence": f"separated {float(d['ym30_5600']):.3f}->{float(d['ym30_10595']):.3f} (falls); "
                    f"attached {float(d['att_eps_5600']):.3f}->{float(d['att_eps_10595']):.3f} (rises)",
        "pass": bool(d["F4_pass"]),
    },
    "binding_outer_pinned": {
        "claim": "denominator |dp/dx|*y_m outer-pinned (gmean ~1.2, not viscous 1.89)",
        "evidence": f"gmean {float(d['dpx_gmean']):.3f}, LOO {list(np.round(d['dpx_loo'],3))}, "
                    f"viscous {float(d['Re_ratio']):.3f}",
        "pass": bool(d["binding_pass"]),
    },
}
print("\n=== F1-F4 verdicts ===")
for k, v in verdicts.items():
    print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}: {v['claim']}")

all_pass = all(v["pass"] for v in verdicts.values())
print(f"\n  ALL falsifiers PASS: {all_pass}")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
np.savez(
    OUT,
    # series
    Re_pts=Re, epsC_pts=eps, epsC_lo=lo, epsC_hi=hi, cf_pts=cf,
    series_src=json.dumps(src),
    # extrapolation band
    Re_eval=Re_eval, band_med=band_med, band_lo=band_lo, band_hi=band_hi,
    err_med=err_med, err_lo=err_lo, err_hi=err_hi,
    slope_med=slope_med, slope_ci_lo=slope_ci[0], slope_ci_hi=slope_ci[1],
    slope_point=slope_re_pt, frac_negative_slope=frac_negative, mc_draws=N,
    # regime map
    row_Re=row_Re, row_hLx=row_hLx, row_sev=row_sev,
    col_Re=col_Re, col_hLx=col_hLx, col_sev=col_sev, col_r2=st_r2,
    col_Lsep=st_Lsep, col_alpha=st_alpha,
    h_over_Lx_canon=H_OVER_LX_CANON, Re_dose=Re_dose,
    lam_c_5600=lam_c_5600, lam_c_10595=lam_c_10595, lam_ref=lam_ref, eps_c=eps_c,
    # F4 specificity
    sep_5600=float(d["ym30_5600"]), sep_10595=float(d["ym30_10595"]),
    att_5600=float(d["att_eps_5600"]), att_10595=float(d["att_eps_10595"]),
    # verdicts
    verdicts=json.dumps(verdicts), all_pass=all_pass,
)
print(f"\nSaved -> {OUT}")
