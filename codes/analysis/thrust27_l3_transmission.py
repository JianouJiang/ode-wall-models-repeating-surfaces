#!/usr/bin/env python3
r"""
Thrust #27 -- Level 3 (Results & analysis): the TRANSMISSION SPECTRUM.

Central question of this node: through which path does the a-priori force-
cancellation depth eps become a *coupled* wall-modelled-LES error, and where
does that path break?  We separate the question into three regimes of varying
statistical power, plus the landed coupled sophistication ladder, and report
each one honestly with the power it actually has.

Inputs (all already on disk, real DNS/WMLES; nothing fabricated):
  codes/results/epsilon_coupled_predictor.npz  -- within-geometry y_m sweep (n=30)
  codes/results/dose_response_xiao.npz         -- within-family steepness sweep (n=29)
  codes/results/thrust27_collapse.npz          -- cross-geometry coupled set (n=5)
  codes/results/aposteriori_crwm_twin.npz      -- coupled sophistication ladder
  codes/results/aposteriori_dose_response.npz  -- 5-point coupled e_reatt (cross-check)

Output: codes/results/thrust27_l3_transmission.npz

Three regimes
-------------
A. WITHIN-GEOMETRY (n=30, powered).  Hold the periodic hill (alpha=1.0) fixed,
   sweep the matching height y_m.  This varies eps along the model's OWN response
   and is the cleanest local test.  Result: a power law relRMS = A*eps^-b with a
   PERFECT rank correlation (Spearman rho=-1.0).  We bootstrap the exponent b.

B. WITHIN-FAMILY (n=29, powered, a-priori).  Hold the geometry CLASS fixed
   (Xiao periodic-hill family) but vary the repeating-structure shape: steepness
   alpha in {0.5..1.5} and pitch.  Result: deeper cancellation -> larger a-priori
   error (Spearman(eps_med, relErr) significant) and the GEOMETRIC predictor that
   collapses the failure depth is L_sep/delta (the O(delta)-pitch criterion of the
   brief), rho(L_sep/delta, R^2) strongly negative with a CI excluding zero.

C. CROSS-GEOMETRY (n=5, underpowered -- honest negative).  Five DISTINCT coupled
   WMLES geometries.  The naive static placement ratio r=y_m/y_crit does NOT
   cross-rank the coupled reattachment error (Spearman rho=0.00, F3 falsified at
   L2).  We DIAGNOSE why, with two on-disk facts:
     (i)  matching-height degeneracy: every coupled solver self-selects the
          buffer layer (y_m+ in [4.5,7.4], factor <2) while y_crit+ spans a
          factor ~175 -- so r collapses to ~1/y_crit and carries no y_m signal;
     (ii) local-vs-global gap: eps is a LOCAL traction measure, e_reatt is a
          GLOBAL bubble-integrated measure; the alpha=0.8 case (deepest eps,
          smallest bubble) is the lone regime outlier that proves the two
          decouple across geometries.

D. COUPLED SOPHISTICATION LADDER (landed).  At the matched coupled config,
   replacing the algebraic Spalding law by the full TBLE ODE does NOT rescue
   reattachment (e_reatt: Spalding 0.333 vs TBLE 0.387) -- sophistication-
   independence.  The convection-restoring CR-WM arm is still LIVE; no CR-WM
   coupled number is asserted here.
"""
import os
import numpy as np
from scipy.stats import spearmanr, linregress

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")


def L(name):
    return np.load(os.path.join(RES, name), allow_pickle=True)


# Deterministic bootstrap RNG (no wall-clock seed; Math.random/Date banned-equiv
# reproducibility -- fixed seed so the CI is identical on every re-run).
RNG = np.random.default_rng(20260531)


# ----------------------------------------------------------------------------
# Regime A: within-geometry y_m sweep (n=30, powered)
# ----------------------------------------------------------------------------
a = L("epsilon_coupled_predictor.npz")
A_eps = np.asarray(a["sweep_eps"], float)
A_relrms = np.asarray(a["sweep_relrms"], float)
A_n = A_eps.size
A_rho, A_p = spearmanr(A_eps, A_relrms)            # expect -1.0
# power-law fit relRMS = A * eps^-b  ->  log relRMS = logA - b log eps
lx = np.log(A_eps)
ly = np.log(A_relrms)
sl = linregress(lx, ly)
A_b = -sl.slope                                    # exponent (positive)
A_logA = sl.intercept
A_r2 = sl.rvalue ** 2
# bootstrap the exponent (resample the 30 sweep points with replacement)
B = 5000
boot_b = np.empty(B)
idx_all = np.arange(A_n)
for i in range(B):
    s = RNG.choice(idx_all, size=A_n, replace=True)
    ss = linregress(lx[s], ly[s])
    boot_b[i] = -ss.slope
A_b_ci = np.percentile(boot_b, [2.5, 97.5])

# ----------------------------------------------------------------------------
# Regime B: within-family steepness sweep (n=29, powered, a-priori)
# ----------------------------------------------------------------------------
b = L("dose_response_xiao.npz")
B_eps = np.asarray(b["agg_eps_median"], float)
B_relerr = np.asarray(b["agg_rel_err"], float)
B_r2 = np.asarray(b["agg_r2"], float)
B_alpha = np.asarray(b["agg_cv_alpha"], float)
B_Lsep_d = np.asarray(b["agg_cv_Lsep_over_delta"], float)
B_n = B_eps.size
B_rho_eps_relerr, B_p_eps_relerr = spearmanr(B_eps, B_relerr)
B_rho_eps_r2, B_p_eps_r2 = spearmanr(B_eps, B_r2)
# geometric collapse variable: L_sep/delta vs a-priori failure depth R^2
B_rho_Lsep_r2 = float(b["collapse_Lsep_over_delta_rho_r2"])
B_Lsep_r2_ci = (float(b["collapse_Lsep_over_delta_rho_r2_ci_lo"]),
                float(b["collapse_Lsep_over_delta_rho_r2_ci_hi"]))
B_rho_Lsep_eps = float(b["collapse_Lsep_over_delta_rho_eps_median"])
# all 29 cases sit at L_sep/delta = O(1) (the trigger window)
B_Lsep_d_range = (float(np.nanmin(B_Lsep_d)), float(np.nanmax(B_Lsep_d)))

# ----------------------------------------------------------------------------
# Regime C: cross-geometry coupled set (n=5, underpowered -- honest negative)
# ----------------------------------------------------------------------------
c = L("thrust27_collapse.npz")
C_labels = [str(x) for x in c["labels"]]
C_eps = np.asarray(c["eps_med"], float)
C_ereatt = np.asarray(c["e_reatt"], float)
C_ratio = np.asarray(c["ratio"], float)
C_ym = np.asarray(c["y_m_plus"], float)
C_ycrit = np.asarray(c["y_crit_plus"], float)
C_rho_ratio = float(c["rho_ratio"])               # F3 statistic (=0.00, falsified)
C_rho_eps = float(c["rho_eps"])
C_loo_ratio = np.asarray(c["loo_rho_ratio"], float)

# (i) matching-height degeneracy: spread of y_m+ vs spread of y_crit+
C_ym_span = float(C_ym.max() / C_ym.min())        # ~1.66
C_ycrit_span = float(C_ycrit.max() / C_ycrit.min())  # ~175
# Pearson check that the static ratio is dominated by 1/y_crit, not by y_m:
#   if y_m were the driver, log(r) would track log(y_m); if y_crit, -log(y_crit).
pr_ym = np.corrcoef(np.log(C_ratio), np.log(C_ym))[0, 1]
pr_ycrit = np.corrcoef(np.log(C_ratio), -np.log(C_ycrit))[0, 1]

# (ii) local-vs-global gap: identify the alpha=0.8 outlier explicitly by label.
#   Within the Re=5600 steepness triplet (alpha=0.8/1.0/1.2) the SHALLOWEST
#   geometry alpha=0.8 has the DEEPEST cancellation (smallest eps) yet the
#   SMALLEST coupled reattachment error -- deep LOCAL cancellation that does not
#   transmit to a large GLOBAL (bubble-integrated) error.
a0p8_g = int(np.where(["alpha=0.8" in s for s in C_labels])[0][0])
# the Re=5600 steepness triplet (exclude the cross-Re Re10595 and the conv-div control)
trip = np.array([("alpha=0.8" in s or "alpha=1.0 (Re5600" in s or "alpha=1.2" in s)
                 for s in C_labels])
C_trip_eps = C_eps[trip]
C_trip_ereatt = C_ereatt[trip]
C_a0p8_eps = float(C_eps[a0p8_g])
C_a0p8_ereatt = float(C_ereatt[a0p8_g])
C_a0p8_is_deepest_in_trip = bool(C_a0p8_eps == C_trip_eps.min())
C_a0p8_is_smallest_err_in_trip = bool(C_a0p8_ereatt == C_trip_ereatt.min())
# drop-the-outlier LOO: the other four order perfectly by r
C_loo_drop_a0p8 = float(C_loo_ratio[a0p8_g])       # expect +1.0

# ----------------------------------------------------------------------------
# Regime D: coupled sophistication ladder (landed; CR-WM still live)
# ----------------------------------------------------------------------------
t = L("aposteriori_crwm_twin.npz")
D_spalding = float(t["spalding_e_reatt"])
D_tble = float(t["tble_e_reatt"])
D_crwm_present = bool(t["crwm_present"])
D_crwm = float(t["crwm_e_reatt"]) if D_crwm_present else np.nan
# sophistication premium Delta_{S->T} = e_TBLE - e_Spalding  (>=0 means NO rescue)
D_premium = D_tble - D_spalding

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
print("=" * 70)
print("THRUST #27 L3 -- TRANSMISSION SPECTRUM")
print("=" * 70)
print("\n[A] WITHIN-GEOMETRY y_m sweep (n=%d, powered):" % A_n)
print(f"    Spearman(eps, relRMS) = {A_rho:+.3f} (p={A_p:.1e})")
print(f"    power law relRMS = {np.exp(A_logA):.3f} * eps^-{A_b:.3f}"
      f"   R^2(log-log)={A_r2:.4f}")
print(f"    exponent b = {A_b:.3f}  bootstrap95%CI=[{A_b_ci[0]:.3f},{A_b_ci[1]:.3f}]"
      f"  (B={B})")
print("\n[B] WITHIN-FAMILY steepness sweep (n=%d, powered, a-priori):" % B_n)
print(f"    Spearman(eps_med, relErr) = {B_rho_eps_relerr:+.3f} (p={B_p_eps_relerr:.2e})")
print(f"    Spearman(eps_med, R^2)    = {B_rho_eps_r2:+.3f} (p={B_p_eps_r2:.2e})")
print(f"    GEOMETRIC collapse: rho(L_sep/delta, R^2) = {B_rho_Lsep_r2:+.3f}"
      f"  CI[{B_Lsep_r2_ci[0]:+.2f},{B_Lsep_r2_ci[1]:+.2f}]")
print(f"    all 29 cases at L_sep/delta in [{B_Lsep_d_range[0]:.2f},{B_Lsep_d_range[1]:.2f}]"
      f"  (O(delta) trigger window)")
print("\n[C] CROSS-GEOMETRY coupled set (n=%d, UNDERPOWERED -- honest negative):" % C_eps.size)
print(f"    F3: Spearman(r=y_m/y_crit, e_reatt) = {C_rho_ratio:+.3f}  -> FALSIFIED")
print(f"    matching-height degeneracy: y_m+ span x{C_ym_span:.2f} vs y_crit+ span x{C_ycrit_span:.1f}")
print(f"    corr(log r, log y_m)= {pr_ym:+.3f}   corr(log r, -log y_crit)= {pr_ycrit:+.3f}")
print(f"      -> the static ratio is a y_crit proxy, carries no y_m signal")
print(f"    local-vs-global: in Re5600 triplet alpha=0.8 DEEPEST eps={C_a0p8_eps:.3f}"
      f" (deepest={C_a0p8_is_deepest_in_trip}) yet SMALLEST e_reatt={C_a0p8_ereatt:.3f}"
      f" (smallest={C_a0p8_is_smallest_err_in_trip})")
print(f"    LOO drop alpha=0.8 -> rho(r,e)={C_loo_drop_a0p8:+.2f} (other four order by r)")
print("\n[D] COUPLED SOPHISTICATION LADDER (landed; CR-WM live):")
print(f"    e_reatt: Spalding={D_spalding:.3f}  TBLE={D_tble:.3f}"
      f"  premium Delta_S->T={D_premium:+.3f} (>=0 -> NO rescue)")
print(f"    CR-WM present={D_crwm_present} (run live; no CR-WM number asserted)")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
out = os.path.join(RES, "thrust27_l3_transmission.npz")
np.savez(
    out,
    # A
    A_eps=A_eps, A_relrms=A_relrms, A_n=A_n, A_rho=A_rho, A_p=A_p,
    A_b=A_b, A_logA=A_logA, A_r2=A_r2, A_b_ci=A_b_ci, A_boot_B=B,
    # B
    B_eps=B_eps, B_relerr=B_relerr, B_r2=B_r2, B_alpha=B_alpha,
    B_Lsep_over_delta=B_Lsep_d, B_n=B_n,
    B_rho_eps_relerr=B_rho_eps_relerr, B_p_eps_relerr=B_p_eps_relerr,
    B_rho_eps_r2=B_rho_eps_r2, B_p_eps_r2=B_p_eps_r2,
    B_rho_Lsep_r2=B_rho_Lsep_r2, B_Lsep_r2_ci=np.array(B_Lsep_r2_ci),
    B_rho_Lsep_eps=B_rho_Lsep_eps, B_Lsep_d_range=np.array(B_Lsep_d_range),
    # C
    C_labels=np.array(C_labels, dtype=object), C_eps=C_eps, C_ereatt=C_ereatt,
    C_ratio=C_ratio, C_ym=C_ym, C_ycrit=C_ycrit,
    C_rho_ratio=C_rho_ratio, C_rho_eps=C_rho_eps, C_loo_ratio=C_loo_ratio,
    C_ym_span=C_ym_span, C_ycrit_span=C_ycrit_span,
    C_corr_logr_logym=pr_ym, C_corr_logr_neglogycrit=pr_ycrit,
    C_a0p8_eps=C_a0p8_eps, C_a0p8_ereatt=C_a0p8_ereatt,
    C_a0p8_is_deepest_in_trip=C_a0p8_is_deepest_in_trip,
    C_a0p8_is_smallest_err_in_trip=C_a0p8_is_smallest_err_in_trip,
    C_trip_eps=C_trip_eps, C_trip_ereatt=C_trip_ereatt,
    C_loo_drop_a0p8=C_loo_drop_a0p8,
    # D
    D_spalding=D_spalding, D_tble=D_tble, D_premium=D_premium,
    D_crwm_present=D_crwm_present, D_crwm=D_crwm,
    note=("Thrust27 L3 transmission spectrum. A=within-geom n=30 powered; "
          "B=within-family n=29 powered a-priori; C=cross-geom n=5 honest "
          "negative + degeneracy diagnosis; D=coupled sophistication landed, "
          "CR-WM live. All numbers trace to real DNS/WMLES npz."),
)
print(f"\nsaved -> {out}")
