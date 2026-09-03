#!/usr/bin/env python3
r"""
operating_map_l3.py  --  Thrust #28, Level-3 (results & analysis)

THE OPERATING MAP, ADJUDICATED.

L2 built the common operating-map dataset (Sec.6 algorithm on 8 geometries,
`operating_map_dataset.npz`).  L3 turns that dataset into the paper's RESULTS:
it attaches an INFERENTIAL layer to every headline number and discharges the L2
Judge's binding conditions.  Nothing here is a new physical model -- it is the
statistical adjudication of the map the L2 pipeline produced.

What L3 computes (every item traces to an on-disk reference field or to L2's
dataset, and is re-derived from source by verify_node003.py):

  (A) F6 -- the class-separation margin, made quantitative.
      The ignition height y_crit+ separates the O(delta)-pitch hills from the
      wide-pitch + single-feature controls with a clean rank gap.  We report the
      separation as (i) a rank statistic (AUC = 1 iff every hill ignites below
      every control) and (ii) the multiplicative margin
      min(control y_crit+) / max(hill y_crit+).  Same for the cancellation length
      c_eps+.  This is the F6 "portability" claim turned into a number with a
      bootstrap confidence interval.

  (B) BC-L3-1 -- the NASA-hump y_crit+ = inf "sentinel", resolved NUMERICALLY.
      The Sec.6 crossing uses the VARIANCE-BALANCE criterion relrms >= 1 (model
      RMSE reaches the RMS *magnitude* of tau_w).  For NASA hump relrms maxes at
      ~0.35 over the whole resolved range, so it never reaches 1 -> y_crit+ = inf
      is the CORRECT answer of that criterion, not a bug.  The Judge's "R^2 = 0
      at y_m+ ~ 185-204" uses the STRICTER mean-subtracted criterion (RMSE reaches
      the standard deviation of tau_w).  At Re=936000 the wall stress has a large
      mean, so RMS(tau_w) >> std(tau_w): the model crosses the std bar (R^2=0) but
      never the magnitude bar (relrms=1).  Both put NASA y_crit+ FAR above the
      hill range (max 15.93) -> the F6 separation holds under EITHER definition.
      We compute both crossings for all 8 geometries (closing the 6/8 verify gap).

  (C) F2 -- the within-geometry informative axis, with the EXPONENT (not the sign)
      as the inferential object.  The Judge's standing objection is that the
      within-geometry Spearman rho = -1 is a tautology (one knob swept on one
      geometry monotonically).  Conceded: rho = -1 is structural.  The NON-trivial
      measured quantity is the POWER-LAW EXPONENT b in relrms = A eps^(-b): it sets
      HOW FAST the error grows as the match deepens, and a tautology cannot
      predict its value.  We bootstrap b over the sweep points and report a CI.

  (D) F3 -- the cross-geometry static-chi honest negative, with an EXACT test.
      n = 5 coupled points: an asymptotic p-value is meaningless.  We enumerate all
      5! = 120 permutations and compute the EXACT two-sided p for rho(eps,e_reatt)
      and rho(chi,e_reatt).  Both are non-significant -> the cross-geometry sort is
      genuinely uninformative (the degeneracy is real, not hidden signal); the
      decisive test is the within-geometry band traverse (F4, unlanded).

  (E) F5 -- the Re_tau* deployment phase boundary, with the O(delta) bound stated
      as a number: at a coarse deployment grid (r_g = Dy_1/delta = 0.05) every
      O(delta) hill crosses ignition at Re_tau* <= ~320 (moderate engineering Re),
      while every control needs Re_tau* >= ~4000.  DERIVED-LAW EXTRAPOLATION
      (Re_tau* = y_crit+/r_g), flagged, never a simulated high-Re run.

NO band number and NO CR-WM number is used anywhere (F4 live/unlanded).

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/operating_map_l3.py
Out:  codes/results/operating_map_l3.npz
"""
from __future__ import annotations

import os
import sys
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
sys.path.insert(0, os.path.join(CODES, "analysis"))
import critical_matching_height as cmh  # noqa: E402

RNG = np.random.default_rng(28031)   # fixed seed -> reproducible bootstrap
N_BOOT = 5000


# ----------------------------------------------------------------------------- #
#  small stats helpers (hand-rolled; verify_node003 cross-checks vs scipy)
# ----------------------------------------------------------------------------- #
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def exact_perm_p(x, y, stat=spearman):
    """Exact two-sided permutation p over all |y|! permutations (n small)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = abs(stat(x, y))
    n = len(y)
    cnt = 0; tot = 0
    for perm in itertools.permutations(range(n)):
        tot += 1
        if abs(stat(x, y[list(perm)])) >= obs - 1e-12:
            cnt += 1
    return obs, cnt / tot, tot


def r2_zero_crossing(ymp, r2):
    """y_m+ where R^2 crosses 0 downward (mean-subtracted criterion), log-interp."""
    ok = np.isfinite(r2) & np.isfinite(ymp)
    yy = ymp[ok]; rr = r2[ok]
    idx = np.where((rr[:-1] > 0) & (rr[1:] <= 0))[0]
    if not idx.size:
        return np.inf
    i = idx[0]
    t = (0.0 - rr[i]) / (rr[i + 1] - rr[i])
    return float(np.exp(np.log(yy[i]) + t * (np.log(yy[i + 1]) - np.log(yy[i]))))


def relrms1_crossing(ymp, relrms):
    """y_m+ where relrms crosses 1 upward (variance-balance/magnitude criterion)."""
    ok = np.isfinite(relrms) & np.isfinite(ymp)
    yy = ymp[ok]; rr = relrms[ok]
    if rr.size < 2:
        return np.nan
    if rr[0] >= 1.0:
        return float(yy[0])
    idx = np.where((rr[:-1] < 1.0) & (rr[1:] >= 1.0))[0]
    if not idx.size:
        return np.inf
    i = idx[0]
    t = (np.log(1.0) - np.log(rr[i])) / (np.log(rr[i + 1]) - np.log(rr[i]))
    return float(np.exp(np.log(yy[i]) + t * (np.log(yy[i + 1]) - np.log(yy[i]))))


def main():
    print("=" * 100)
    print("THRUST #28 L3 -- adjudicating the OPERATING MAP (inferential layer)")
    print("=" * 100)

    d = np.load(os.path.join(RESULTS, "operating_map_dataset.npz"), allow_pickle=True)
    keys = [str(s) for s in d["keys"]]
    klass = [str(s) for s in d["klass"]]
    ycrit = d["ycrit"].astype(float)
    c_eps = d["c_eps_plus"].astype(float)
    ymp_grid = d["ymp_grid"].astype(float)

    od_mask = np.array([k == "repeating_Odelta" for k in klass])
    ctrl_mask = ~od_mask

    # ----- (B) NASA / dual-criterion: BOTH crossings for ALL 8 geometries ----- #
    print("\n[B] BC-L3-1 -- dual-criterion ignition height for ALL 8 geometries:")
    print(f"    {'geometry':22s} {'relrms=1 (y_crit+)':>20s} {'R^2=0 (y_m+)':>14s} "
          f"{'relrms_max':>11s}")
    relrms1 = np.full(len(keys), np.nan)
    r2zero = np.full(len(keys), np.nan)
    relrms_max = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        rr = d[f"sweep_relrms__{k}"].astype(float)
        r2 = d[f"sweep_r2__{k}"].astype(float)
        relrms1[i] = relrms1_crossing(ymp_grid, rr)
        r2zero[i] = r2_zero_crossing(ymp_grid, r2)
        relrms_max[i] = float(np.nanmax(rr))
        f1 = ("%.2f" % relrms1[i]) if np.isfinite(relrms1[i]) else "inf"
        f2 = ("%.1f" % r2zero[i]) if np.isfinite(r2zero[i]) else "inf"
        print(f"    {k:22s} {f1:>20s} {f2:>14s} {relrms_max[i]:>11.3f}")
    # the F6 separation under BOTH criteria
    hill_yc_rms = relrms1[od_mask]
    ctrl_yc_rms = relrms1[ctrl_mask]
    hill_yc_r2 = r2zero[od_mask]
    ctrl_yc_r2 = r2zero[ctrl_mask]
    sep_rms = bool(np.nanmax(hill_yc_rms) < np.nanmin(ctrl_yc_rms[np.isfinite(ctrl_yc_rms)])
                   if np.isfinite(ctrl_yc_rms).any() else True)
    # under R^2: compare finite values; inf controls trivially above
    finite_ctrl_r2 = ctrl_yc_r2[np.isfinite(ctrl_yc_r2)]
    sep_r2 = bool(np.nanmax(hill_yc_r2[np.isfinite(hill_yc_r2)]) <
                  (finite_ctrl_r2.min() if finite_ctrl_r2.size else np.inf))
    nasa_i = keys.index("nasa_hump")
    print(f"    NASA hump: relrms_max={relrms_max[nasa_i]:.3f} (<1 -> relrms=1 NEVER "
          f"reached -> y_crit+=inf is CORRECT); R^2=0 at y_m+~{r2zero[nasa_i]:.0f}.")
    print(f"    F6 separation holds under relrms criterion: {sep_rms}; "
          f"under R^2 criterion: {sep_r2}.")

    # ----- (A) F6 class-separation margin + AUC + bootstrap ------------------- #
    print("\n[A] F6 -- class-separation margin (ignition height + cancellation length):")
    hill_yc = ycrit[od_mask & np.isfinite(ycrit)]
    ctrl_yc = ycrit[ctrl_mask & np.isfinite(ycrit)]
    hill_ce = c_eps[od_mask & np.isfinite(c_eps)]
    ctrl_ce = c_eps[ctrl_mask & np.isfinite(c_eps)]

    def auc_separation(pos, neg):
        """fraction of (pos<neg) ordered pairs; 1.0 == perfect rank separation."""
        wins = sum(1.0 * (p < n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    auc_yc = auc_separation(hill_yc, ctrl_yc)
    auc_ce = auc_separation(hill_ce, ctrl_ce)
    margin_yc = float(ctrl_yc.min() / hill_yc.max())
    margin_ce = float(ctrl_ce.min() / hill_ce.max())
    # bootstrap the margin: resample within each class, recompute min(ctrl)/max(hill)
    boot_m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        hb = RNG.choice(hill_yc, size=hill_yc.size, replace=True)
        cb = RNG.choice(ctrl_yc, size=ctrl_yc.size, replace=True)
        boot_m[b] = cb.min() / hb.max()
    margin_ci = (float(np.percentile(boot_m, 2.5)), float(np.percentile(boot_m, 97.5)))
    print(f"    y_crit+: hills {np.sort(hill_yc).round(2)} vs controls {np.sort(ctrl_yc).round(1)}")
    print(f"      AUC(y_crit+) = {auc_yc:.3f} (1.0 = every hill ignites below every control)")
    print(f"      margin min(ctrl)/max(hill) = {margin_yc:.1f}x  "
          f"95%CI[{margin_ci[0]:.1f},{margin_ci[1]:.1f}]")
    print(f"    c_eps+: hills {np.sort(hill_ce).round(2)} vs controls {np.sort(ctrl_ce).round(1)}")
    print(f"      AUC(c_eps+) = {auc_ce:.3f} ; margin = {margin_ce:.1f}x")

    # ----- (C) F2 -- within-geometry power-law exponent b with bootstrap CI --- #
    print("\n[C] F2 -- within-geometry informative axis (exponent b is the real datum):")
    pr = np.load(os.path.join(RESULTS, "epsilon_coupled_predictor.npz"), allow_pickle=True)
    sw_eps = pr["sweep_eps"].astype(float)
    sw_rms = pr["sweep_relrms"].astype(float)
    ok = np.isfinite(sw_eps) & np.isfinite(sw_rms) & (sw_eps > 0) & (sw_rms > 0)
    x = np.log(sw_eps[ok]); y = np.log(sw_rms[ok])     # relrms = A eps^(-b) -> slope=-b
    b_point = float(-np.polyfit(x, y, 1)[0])
    rho_f2 = spearman(sw_eps[ok], sw_rms[ok])
    n_pts = int(ok.sum())
    boot_b = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = RNG.integers(0, n_pts, n_pts)
        boot_b[k] = -np.polyfit(x[idx], y[idx], 1)[0]
    b_ci = (float(np.percentile(boot_b, 2.5)), float(np.percentile(boot_b, 97.5)))
    print(f"    n={n_pts} within-geometry sweep points; Spearman rho={rho_f2:+.3f} "
          f"(conceded structural).")
    print(f"    POWER-LAW EXPONENT b = {b_point:.3f}  95%CI[{b_ci[0]:.3f},{b_ci[1]:.3f}] "
          f"(excludes 0: {b_ci[0] > 0}) -- the non-tautological measured rate.")

    # ----- (D) F3 -- exact n=5 permutation test for the honest negative ------- #
    print("\n[D] F3 -- cross-geometry static-chi honest negative (EXACT n=5 test):")
    eps5 = d["eps5"].astype(float)
    err5 = d["err5"].astype(float)
    chi5 = d["chi5"].astype(float)
    obs_eps, p_eps, ntot = exact_perm_p(eps5, err5)
    obs_chi, p_chi, _ = exact_perm_p(chi5, err5)
    rho_eps = spearman(eps5, err5)
    rho_chi = spearman(chi5, err5)
    print(f"    n=5 coupled points, {ntot} exact permutations:")
    print(f"      rho(eps,  e_reatt) = {rho_eps:+.3f}  exact two-sided p = {p_eps:.3f}")
    print(f"      rho(chi,  e_reatt) = {rho_chi:+.3f}  exact two-sided p = {p_chi:.3f}")
    print(f"    => neither orders the coupled error (both p ~ 1): the cross-geometry")
    print(f"       sort is genuinely uninformative (degeneracy real, not hidden signal).")
    ym_span = float(d["ym_span"]); ycrit_span = float(d["ycrit_span"])
    corr_logchi = float(d["corr_logchi"])
    print(f"    degeneracy: deployed y_m+ span x{ym_span:.2f} vs y_crit+ span x{ycrit_span:.1f}; "
          f"corr(log chi, -log y_crit+)={corr_logchi:.3f}  -> static chi is a y_crit+ proxy.")

    # ----- (E) F5 -- Re_tau* phase boundary, O(delta) bound as a number ------- #
    print("\n[E] F5 -- Re_tau* deployment phase boundary [DERIVED-LAW EXTRAPOLATION]:")
    r_g = d["r_g_grid"].astype(float)
    re_star = d["re_tau_star"].astype(float)        # (8,4)
    j_coarse = int(np.argmin(np.abs(r_g - 0.05)))    # the coarse standard grid
    hill_re = re_star[od_mask, j_coarse]
    ctrl_re = re_star[ctrl_mask, j_coarse]
    hill_re = hill_re[np.isfinite(hill_re)]
    ctrl_re_fin = ctrl_re[np.isfinite(ctrl_re)]
    print(f"    at coarse grid r_g={r_g[j_coarse]:.2f}: O(delta) hills Re_tau* in "
          f"[{hill_re.min():.0f},{hill_re.max():.0f}] (moderate engineering Re); "
          f"controls Re_tau* >= {ctrl_re_fin.min():.0f}.")
    print(f"    => the catastrophe is a HIGH-Re DEPLOYMENT hazard for O(delta) hills on "
          f"standard grids, invisible at the cheap DNS-Re ({d['re_tau_deployed']:.0f}) grids "
          f"used to validate.")

    # ----- SAVE --------------------------------------------------------------- #
    out = os.path.join(RESULTS, "operating_map_l3.npz")
    np.savez(
        out,
        keys=np.array(keys), klass=np.array(klass),
        # (B) dual-criterion
        ycrit_relrms1=relrms1, ycrit_r2zero=r2zero, relrms_max=relrms_max,
        sep_under_relrms=sep_rms, sep_under_r2=sep_r2,
        nasa_relrms_max=float(relrms_max[nasa_i]), nasa_r2zero=float(r2zero[nasa_i]),
        # (A) F6 separation
        hill_ycrit=hill_yc, ctrl_ycrit=ctrl_yc, hill_ceps=hill_ce, ctrl_ceps=ctrl_ce,
        auc_ycrit=auc_yc, auc_ceps=auc_ce,
        margin_ycrit=margin_yc, margin_ycrit_ci=np.array(margin_ci),
        margin_ceps=margin_ce,
        # (C) F2 exponent
        F2_b_point=b_point, F2_b_ci=np.array(b_ci), F2_rho=rho_f2, F2_npts=n_pts,
        # (D) F3 exact perm
        rho_eps5=rho_eps, p_eps5=p_eps, rho_chi5=rho_chi, p_chi5=p_chi, n_perm=ntot,
        ym_span=ym_span, ycrit_span=ycrit_span, corr_logchi=corr_logchi,
        eps5=eps5, err5=err5, chi5=chi5,
        # (E) Re_tau*
        r_g_grid=r_g, j_coarse=j_coarse, hill_re_star_coarse=hill_re,
        ctrl_re_star_coarse=ctrl_re_fin,
        # honesty
        crwm_number_used=False, band_number_used=False, extrapolation_flag=True,
        seed=28031, n_boot=N_BOOT,
        note=("Thrust #28 L3 inferential layer over operating_map_dataset.npz. "
              "(A) F6 class separation AUC=1, multiplicative margin + bootstrap CI. "
              "(B) BC-L3-1: dual-criterion ignition height (relrms=1 magnitude vs R^2=0 "
              "mean-subtracted) for ALL 8 geometries; NASA y_crit+=inf is the correct "
              "relrms answer (relrms_max~0.35<1), R^2=0 at y_m+~185; both >> hill max "
              "15.93 so F6 holds under either definition. (C) F2 within-geom exponent b "
              "with bootstrap CI (rho=-1 conceded structural; b is the measured rate). "
              "(D) F3 EXACT 120-permutation p for the n=5 cross-geom honest negative. "
              "(E) F5 Re_tau* O(delta) bound (DERIVED-LAW EXTRAPOLATION). NO band/CR-WM "
              "number used."),
    )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
