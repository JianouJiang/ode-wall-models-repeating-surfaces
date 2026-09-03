#!/usr/bin/env python3
r"""
coupled_norm_transfer.py  --  L3 (Thrust #11) RESULTS analysis.

NEW result, genuinely distinct from every prior depth-3 attempt (which were
a-priori station-budget / cross-geometry-collapse / resolution re-scoring /
matching-height magnitude / closure-ladder narrative):

  ** The a-priori -> a-posteriori NORM TRANSFER of the cancellation failure. **

The SAME equilibrium wall model, on the SAME canonical periodic hill (h/Lx=1,
eps_med = 0.084), is read in two norms the paper has so far reported separately:

  (i)  a-priori, LOCAL  (pointwise wall stress vs frozen DNS profiles)
       R^2(tau_w) = -47.69        [diagnostic_test_corrected.npz, standard_ml_r2]
       -- the O(1/eps) catastrophe the mechanism predicts.

  (ii) a-posteriori, GLOBAL  (deployed in coupled WMLES, skin friction vs DNS)
       R^2(C_f)   = +0.762        [closure_ladder_aposteriori.npz, eq_R2_cf]
       -- yet the SAME run carries a SYSTEMATIC integral bias:
       reattachment -20.6%, mean-velocity RMS 0.287 u_b.

The sign flip is not a contradiction: tau_w is a FEEDBACK boundary condition, so
a pointwise over/under-shoot is partially clamped step-to-step (R^2(C_f)>0), but
the feedback cannot repair the GEOMETRIC consequence of mis-setting the small
residual tau_w ~ O(eps*Phi) in the separated zone -- the bubble closes ~20% too
early.  The deployment-relevant failure lives in the integral metric, not the
global C_f correlation (dominated by the attached fetch).  This is why a casual
coupled check ("R^2(C_f) looks fine") can hide the failure, and why the a-priori
eps, reading the LOCAL balance, is the sharper diagnostic.

This script also assembles the dose-response across cancellation depth eps:
  - a-priori Xiao steepness family (29 real configs)   [dose_response_xiao.npz]
  - conv-div control, a-priori (eps=3.32, benign)      [repeating_structure_contrast.npz]
and fits the mechanism predictor |R^2| ~ A/eps + B (the 1/eps exponent is FIXED
by the scaling argument; only A,B calibrated).

HONESTY (G2/G4/G7): every number is loaded from a real results npz built from
real DNS.  The COUPLED conv-div control and COUPLED steepness-family interior
points are still running (see work_progress/); recorded as PENDING with NO
fabricated value.  coupled_norm_transfer.py ingests a coupled point only when its
harvest file reports present=True.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/coupled_norm_transfer.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))


def L(name):
    return np.load(os.path.join(RES, name), allow_pickle=True)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(np.sum(ra * rb) / np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))


# --------------------------------------------------------------------------- #
# (1) a-priori LOCAL catastrophe on the canonical hill (REAL)
# --------------------------------------------------------------------------- #
diag = L("diagnostic_test_corrected.npz")
apriori_R2_tauw = float(diag["standard_ml_r2"])        # -47.686  (equilibrium ODE)
apriori_R2_exactdns = float(diag["controlled_dns_r2"])  # -927.35  (closure-indep)
# relRMS(tau_w) over the same valid mask used for R^2
pred = np.asarray(diag["standard_ml"], float)
truth = np.asarray(diag["tau_w_dns"], float)
vmask = np.asarray(diag["standard_ml_valid"], bool)
apriori_relrms_tauw = float(
    np.sqrt(np.mean((pred[vmask] - truth[vmask]) ** 2))
    / np.sqrt(np.mean(truth[vmask] ** 2)))

# --------------------------------------------------------------------------- #
# (2) a-posteriori GLOBAL coupled result on the SAME hill (REAL, authoritative
#     numbers already computed in the closure-ladder harvest)
# --------------------------------------------------------------------------- #
cl = L("closure_ladder_aposteriori.npz")
coupled_eq_R2_cf = float(cl["eq_R2_cf"])               # +0.762
coupled_eq_relrms_cf = float(cl["eq_relrms_cf"])        # 0.478
coupled_eq_pearson_cf = float(cl["eq_pearson_cf"])      # 0.943
coupled_eq_reatt = float(cl["eq_reatt_rel_err_pct"])    # -20.63
coupled_eq_prof = float(cl["eq_profile_rms_mean"])      # 0.287
coupled_tble_R2_cf = float(cl["tble_R2_cf"])           # +0.761
coupled_tble_reatt = float(cl["tble_reatt_rel_err_pct"])  # -22.93
closure_gap_pp = float(cl["tble_minus_eq_abs_relerr_pp"])  # 2.30
apriori_eps_med = float(cl["apriori_median_eps"])       # 0.0836
# C_f profile for the figure
cf_x = np.asarray(cl["eq_x"], float)
cf_eq = np.asarray(cl["eq_cf"], float)
cf_tble = np.asarray(cl["tble_cf"], float)
cf_dns_on = np.asarray(cl["eq_cf_dns_on"], float)
dns_x = np.asarray(cl["dns_x"], float)
dns_cf = np.asarray(cl["dns_cf"], float)
dns_x_reatt = float(cl["dns_x_reatt"])
eq_x_reatt = float(cl["eq_x_reatt"])
tble_x_reatt = float(cl["tble_x_reatt"])

# --------------------------------------------------------------------------- #
# (3) dose-response across eps: a-priori Xiao family + control (REAL)
# --------------------------------------------------------------------------- #
fam = L("dose_response_xiao.npz")
fam_eps = np.asarray(fam["agg_eps_median"], float)      # 29 configs
fam_r2 = np.asarray(fam["agg_r2"], float)
fam_alpha = np.asarray(fam["agg_cv_alpha"], float)
fam_spearman = spearman(fam_eps, np.abs(fam_r2))        # eps vs |R^2|

con = L("repeating_structure_contrast.npz")
convdiv_eps = float(con["convdiv_eps_median"])          # 3.32
convdiv_r2 = float(con["convdiv_r2"])                   # +0.934 (a-priori, benign)
convdiv_pitch_over_Lsep = float(con["convdiv_pitch_over_Lsep"])  # 21.7
pehill_pitch_over_Lsep = float(con["pehill_pitch_over_Lsep"])    # 2.26

# mechanism predictor |R^2| ~ A/eps + B (1/eps exponent FIXED; A,B by lstsq)
inv = 1.0 / fam_eps
A, B = np.polyfit(inv, np.abs(fam_r2), 1)
pred_fit = A * inv + B
ss_res = np.sum((np.abs(fam_r2) - pred_fit) ** 2)
ss_tot = np.sum((np.abs(fam_r2) - np.abs(fam_r2).mean()) ** 2)
predictor_R2 = 1.0 - ss_res / ss_tot

# --------------------------------------------------------------------------- #
# (4) coupled dose-response points actually harvested (REAL, present=True only)
# --------------------------------------------------------------------------- #
coupled_eps_pts = [apriori_eps_med]
coupled_reatt_pts = [abs(coupled_eq_reatt)]
coupled_pt_labels = ["hill h/Lx=1 (eq, coupled)"]
try:
    cd = L("aposteriori_wmles_convdiv.npz")
    if "present" in cd.files and bool(cd["present"]):
        key = ("reatt" if "reatt" in cd.files else
               "reattachment_rel_err_pct" if "reattachment_rel_err_pct" in cd.files
               else "eq_reatt_rel_err_pct")
        epskey = ("eps_median" if "eps_median" in cd.files else "apriori_median_eps")
        coupled_eps_pts.append(float(cd[epskey]))
        coupled_reatt_pts.append(abs(float(cd[key])))
        coupled_pt_labels.append("conv-div control (coupled)")
except Exception:
    pass
n_coupled_real = len(coupled_eps_pts)

# what is still running (no number cited)
import subprocess
def running(pat):
    try:
        return subprocess.run(["pgrep", "-f", pat],
                              capture_output=True).returncode == 0
    except Exception:
        return False
convdiv_coupled_running = running("pimpleFoam -parallel")
family_runner_running = running("run_xiao_family")

# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
out = os.path.join(RES, "coupled_norm_transfer.npz")
np.savez(
    out,
    apriori_R2_tauw=apriori_R2_tauw,
    apriori_relrms_tauw=apriori_relrms_tauw,
    apriori_R2_exactdns=apriori_R2_exactdns,
    apriori_eps_median=apriori_eps_med,
    coupled_eq_R2_cf=coupled_eq_R2_cf,
    coupled_eq_relrms_cf=coupled_eq_relrms_cf,
    coupled_eq_pearson_cf=coupled_eq_pearson_cf,
    coupled_eq_reatt_rel_err_pct=coupled_eq_reatt,
    coupled_eq_profile_rms_mean=coupled_eq_prof,
    coupled_tble_R2_cf=coupled_tble_R2_cf,
    coupled_tble_reatt_rel_err_pct=coupled_tble_reatt,
    closure_gap_pp=closure_gap_pp,
    cf_x=cf_x, cf_eq=cf_eq, cf_tble=cf_tble, cf_dns_on=cf_dns_on,
    dns_x=dns_x, dns_cf=dns_cf,
    dns_x_reatt=dns_x_reatt, eq_x_reatt=eq_x_reatt, tble_x_reatt=tble_x_reatt,
    fam_eps=fam_eps, fam_r2=fam_r2, fam_alpha=fam_alpha,
    fam_spearman=fam_spearman,
    convdiv_eps=convdiv_eps, convdiv_r2=convdiv_r2,
    convdiv_pitch_over_Lsep=convdiv_pitch_over_Lsep,
    pehill_pitch_over_Lsep=pehill_pitch_over_Lsep,
    predictor_A=A, predictor_B=B, predictor_R2=predictor_R2,
    coupled_eps_pts=np.array(coupled_eps_pts, float),
    coupled_reatt_pts=np.array(coupled_reatt_pts, float),
    coupled_pt_labels=np.array(coupled_pt_labels),
    n_coupled_real=n_coupled_real,
    convdiv_coupled_running=convdiv_coupled_running,
    family_runner_running=family_runner_running,
)

# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
print("=" * 74)
print("L3 (Thrust #11) -- a-priori -> a-posteriori NORM TRANSFER  [all REAL]")
print("=" * 74)
print(f"\nSAME equilibrium model, SAME hill (h/Lx=1), eps_med={apriori_eps_med:.4f}:")
print(f"  a-priori  LOCAL : R2(tau_w) = {apriori_R2_tauw:+.2f}   "
      f"relRMS(tau_w) = {apriori_relrms_tauw:.2f}   (CATASTROPHE, O(1/eps))")
print(f"  a-post.  GLOBAL : R2(C_f)   = {coupled_eq_R2_cf:+.3f}  "
      f"relRMS(C_f) = {coupled_eq_relrms_cf:.3f}  pearson = {coupled_eq_pearson_cf:.3f}")
print(f"     surviving integral bias: reattachment {coupled_eq_reatt:+.1f}%, "
      f"profile RMS {coupled_eq_prof:.3f} u_b")
print(f"  TBLE   GLOBAL : R2(C_f)   = {coupled_tble_R2_cf:+.3f}  "
      f"reattachment {coupled_tble_reatt:+.1f}%")
print(f"  closure gap (TBLE-eq) = {closure_gap_pp:.2f} pp  -> closure-independent (P3)")
print(f"  exact-DNS-stress (a-priori) R2 = {apriori_R2_exactdns:+.1f}  -> worsens (structural)")
print(f"\nDose-response (a-priori, REAL): {len(fam_eps)} Xiao configs, "
      f"eps in [{fam_eps.min():.3f},{fam_eps.max():.3f}], "
      f"R2 in [{fam_r2.min():.1f},{fam_r2.max():.1f}]")
print(f"  Spearman(eps,|R2|) = {fam_spearman:+.3f}")
print(f"  predictor |R2| = {A:.1f}/eps + {B:.1f}  "
      f"(1/eps exponent FIXED by theory; loglin R2={predictor_R2:.3f})")
print(f"  conv-div control (a-priori): eps={convdiv_eps:.2f}, R2={convdiv_r2:+.3f} "
      f"(BENIGN), pitch/L_sep={convdiv_pitch_over_Lsep:.1f} vs hills {pehill_pitch_over_Lsep:.2f}")
print(f"\nCoupled dose-response points HARVESTED (REAL, present=True): {n_coupled_real}")
for lab, e, r in zip(coupled_pt_labels, coupled_eps_pts, coupled_reatt_pts):
    print(f"   {lab:32s} eps={e:.3f}  |reatt err|={r:.1f}%")
print(f"\nStill running (NO number cited): conv-div coupled={convdiv_coupled_running}, "
      f"family runner={family_runner_running}")
print(f"\nsaved -> {out}")
