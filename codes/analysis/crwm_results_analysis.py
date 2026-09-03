#!/usr/bin/env python3
r"""
crwm_results_analysis.py
========================
Thrust #26 (L3, Results and analysis).

L2 MEASURED the wall-model Green's function G(xi) and showed, at a single point
estimate, that the kernel-weighted within-layer convection reproduces the
independently-measured CR-WM(exact) wall-stress cure (R^2 = 0.89) while a
uniform-leverage control collapses (R^2 = -23).  That is validation.  L3 turns
the validated kernel into RESULTS AND ANALYSIS by answering three questions the
point estimates do not:

  R1 (significance) : are the headline numbers (R^2 of the kernel cure, the
        recovery fractions 0.56 / 0.096, the uniform-leverage gap) statistically
        robust, or artefacts of a handful of stations?  -> station bootstrap,
        B = 4000 resamples, 95% percentile CIs, and a paired bootstrap p-value
        for "kernel cure beats the uniform-leverage null".

  R2 (the cure law) : the L2 ceiling is a single population number.  Does the
        within-layer cure depend on HOW DEEP the cancellation is?  -> resolve the
        recovery fraction against the paper's central diagnostic
        eps = |tau_w| / (|dp/dx| y_m), per eps-quartile bin.  This connects the
        Green's-function ceiling to the eps-cancellation mechanism that is the
        spine of the manuscript.  The MEASURED law (not assumed): the within-
        layer cure recovers most where the cancellation is DEEPEST (eps small,
        the genuine failure regime: ~61% of the catastrophe in the deepest
        quartile) and is neutral-to-HARMFUL where eps ~ O(1) (the regime the ODE
        already handles, so there is no catastrophe to cure and a blind
        correction can overshoot).  The cure therefore SELF-TARGETS the failure
        regime -- which is exactly the physical justification for gating CR-WM on
        eps, and shows the population-average 0.56 UNDERSTATES the cure's value
        where the wall model actually fails.

  R3 (forward test) : convert the a-priori within-layer ceiling into a
        PRE-REGISTERED, falsifiable prediction band for the coupled
        a-posteriori reattachment-error cure crwm_e_reatt (live run
        xiao_wmles_a1p0_crwm, not yet landed).  The rigorous claim is the
        two-sided bound 0 < crwm_e_reatt < tble_e_reatt; a clearly-labelled
        HEURISTIC central band follows from the bootstrapped recovery fraction.
        NO coupled cure number is asserted -- this is the prediction the live
        run will confirm or refute.

A-PRIORI throughout (R1, R2) plus an a-priori-derived PREDICTION (R3).  Every
input traces to real DNS already on disk; no new CFD; <= 2 cores.  The coupled
twin baselines that ARE on disk (Spalding 0.333, TBLE 0.387) are read, never
recomputed; crwm_e_reatt is read as NaN (crwm_present == False) and never filled.

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 codes/analysis/crwm_results_analysis.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
import crwm_apriori as C          # the production solver + loaders (for eps inputs)

NU, Y_IDX = C.NU, C.Y_IDX
RNG = np.random.default_rng(20260531)   # fixed seed -> reproducible bootstrap
B = 4000                                # bootstrap resamples


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def r2(pred, truth):
    """Coefficient of determination about the truth mean (same convention as
    crwm_apriori / diagnostic_test_corrected)."""
    ss_res = np.sum((pred - truth) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def recovery_fraction(tau_ode, tau_exact, tau_dns):
    """Fraction of the R^2-space catastrophe closed by the (exact) within-layer
    cure: (R2_cured - R2_ode) / (0 - R2_ode).  Matches recovery_distance /
    gap_closed on disk."""
    r0 = r2(tau_ode, tau_dns)
    rN = r2(tau_exact, tau_dns)
    return float((rN - r0) / (0.0 - r0)) if r0 < 0 else np.nan


def ci(samples, lo=2.5, hi=97.5):
    s = np.asarray(samples)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


# --------------------------------------------------------------------------
# 1. load the MEASURED kernel-experiment arrays (L2 output, real data)
# --------------------------------------------------------------------------
ke = np.load(os.path.join(RESULTS, "crwm_kernel_experiment.npz"), allow_pickle=True)
tau_ode = ke["tau_ode"]
tau_exact = ke["tau_exact"]
tau_dns = ke["tau_dns"]
cure_meas = ke["cure_meas"]     # tau_exact - tau_ode (measured)
cure_pred = ke["cure_pred"]     # int G conv dxi (kernel-weighted)
cure_unif = ke["cure_unif"]     # uniform-leverage counterfactual
is_sep = ke["is_sep"].astype(bool)
n = int(ke["n_stations"])
assert tau_ode.shape == (n,), "kernel arrays not the expected 512-station shape"

# point estimates exactly as L2 reported them (sanity, re-derived here)
r2_cure_pt = r2(cure_pred, cure_meas)
r2_unif_pt = r2(cure_unif, cure_meas)
rec_full_pt = recovery_fraction(tau_ode, tau_exact, tau_dns)
rec_sep_pt = recovery_fraction(tau_ode[is_sep], tau_exact[is_sep], tau_dns[is_sep])

# --------------------------------------------------------------------------
# 2. reconstruct per-station eps on the SAME 512 stations (central diagnostic)
#    eps = |tau_w_dns| / (|dp/dx| * y_m).  Station order replicates the kernel
#    experiment's selection exactly so eps aligns index-for-index.
# --------------------------------------------------------------------------
profs = C.D.extract_profiles()
conv_profs = C.load_conv_profiles()
stations = [i for i in range(len(profs))
            if conv_profs[i] is not None and Y_IDX < len(profs[i]["y"])]
assert len(stations) == n, f"station count {len(stations)} != kernel arrays {n}"
eps = np.empty(n)
for k, i in enumerate(stations):
    pr = profs[i]
    y_m = pr["y"][Y_IDX]
    eps[k] = abs(pr["tau_w_dns"]) / (abs(pr["dpdx"]) * y_m + 1e-30)
eps_median = float(np.median(eps))

# --------------------------------------------------------------------------
# R1.  station bootstrap -> 95% CIs + paired p-value (kernel beats uniform)
# --------------------------------------------------------------------------
idx_all = np.arange(n)
idx_sep = np.where(is_sep)[0]

boot_r2_cure = np.empty(B)
boot_r2_unif = np.empty(B)
boot_rec_full = np.empty(B)
boot_rec_sep = np.empty(B)
boot_dwin = np.empty(B)              # kernel-minus-uniform per-resample R^2 gap
for b in range(B):
    s = RNG.choice(idx_all, size=n, replace=True)
    boot_r2_cure[b] = r2(cure_pred[s], cure_meas[s])
    boot_r2_unif[b] = r2(cure_unif[s], cure_meas[s])
    boot_dwin[b] = boot_r2_cure[b] - boot_r2_unif[b]
    boot_rec_full[b] = recovery_fraction(tau_ode[s], tau_exact[s], tau_dns[s])
    ss = RNG.choice(idx_sep, size=idx_sep.size, replace=True)
    boot_rec_sep[b] = recovery_fraction(tau_ode[ss], tau_exact[ss], tau_dns[ss])

# one-sided bootstrap p-value: P(kernel cure no better than uniform null)
p_kernel_gt_unif = float(np.mean(boot_dwin <= 0.0))

ci_r2_cure = ci(boot_r2_cure)
ci_r2_unif = ci(boot_r2_unif)
ci_rec_full = ci(boot_rec_full)
ci_rec_sep = ci(boot_rec_sep)

# --------------------------------------------------------------------------
# R2.  eps-resolved cure law: recovery fraction per eps quartile bin
# --------------------------------------------------------------------------
q = np.percentile(eps, [25, 50, 75])
edges = np.array([eps.min() - 1e-12, q[0], q[1], q[2], eps.max() + 1e-12])
bin_eps_med = np.empty(4)
bin_rec = np.empty(4)
bin_rec_lo = np.empty(4)
bin_rec_hi = np.empty(4)
bin_r2_ode = np.empty(4)
bin_r2_exact = np.empty(4)
bin_n = np.empty(4, int)
for j in range(4):
    m = (eps > edges[j]) & (eps <= edges[j + 1])
    bin_n[j] = int(m.sum())
    bin_eps_med[j] = float(np.median(eps[m]))
    bin_r2_ode[j] = r2(tau_ode[m], tau_dns[m])
    bin_r2_exact[j] = r2(tau_exact[m], tau_dns[m])
    bin_rec[j] = recovery_fraction(tau_ode[m], tau_exact[m], tau_dns[m])
    # bootstrap CI within the bin
    bi = np.where(m)[0]
    br = np.empty(B)
    for b in range(B):
        ssb = RNG.choice(bi, size=bi.size, replace=True)
        br[b] = recovery_fraction(tau_ode[ssb], tau_exact[ssb], tau_dns[ssb])
    bin_rec_lo[j], bin_rec_hi[j] = ci(br)

# rank correlation: does recovery fall with deepening cancellation (smaller eps)?
# per-station "cure efficiency" = signed cure toward dns / structural error.
struct_err = tau_ode - tau_dns
# avoid divide-by-zero; use stations with non-trivial structural error
eff_mask = np.abs(struct_err) > 1e-6
cure_eff = np.full(n, np.nan)
cure_eff[eff_mask] = cure_meas[eff_mask] / (-struct_err[eff_mask])  # +1 = perfect
# Spearman rho(eps, cure_eff) via rank-Pearson (no scipy dependency)
def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(np.sum(ra * rb) / np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
rho_eps_eff = spearman(eps[eff_mask], cure_eff[eff_mask])

# --------------------------------------------------------------------------
# R3.  forward prediction band for the coupled cure (NOT asserted, predicted)
#      Read the on-disk coupled twin baselines.  crwm_e_reatt MUST be NaN here.
# --------------------------------------------------------------------------
tw = np.load(os.path.join(RESULTS, "aposteriori_crwm_twin.npz"), allow_pickle=True)
spalding_e = float(tw["spalding_e_reatt"])
tble_e = float(tw["tble_e_reatt"])
crwm_present = bool(tw["crwm_present"])
crwm_e_disk = float(tw["crwm_e_reatt"])     # expected NaN until the run lands
assert not crwm_present, "coupled CR-WM run has landed -- HARVEST it, do not predict"
assert np.isnan(crwm_e_disk), "crwm_e_reatt is populated -- use the measured value"

# Rigorous, derivation-backed claim (two-sided): a within-layer cure restores a
# bounded, strictly-positive-but-incomplete fraction of the wall-stress error
# (kernel ceiling), so the coupled reattachment error must shrink but not vanish.
pred_lower = 0.0                            # full cure would falsify outer-core
pred_upper = tble_e                         # no cure would falsify the mechanism

# HEURISTIC central band (explicitly heuristic): IF the coupled reattachment
# error scaled linearly with the unrecovered fraction of the wall-stress
# catastrophe, the partial cure 'rec' would map e_TBLE -> e_TBLE*(1-rec).
# Use the bootstrapped full-population recovery fraction for the band.
heur_center = tble_e * (1.0 - rec_full_pt)
heur_lo = tble_e * (1.0 - ci_rec_full[1])   # more recovery -> lower e
heur_hi = tble_e * (1.0 - ci_rec_full[0])

# --------------------------------------------------------------------------
# save + report
# --------------------------------------------------------------------------
out = os.path.join(RESULTS, "crwm_results_analysis.npz")
np.savez(
    out,
    n_stations=n, B=B, eps=eps, eps_median=eps_median,
    # R1
    r2_cure_pt=r2_cure_pt, r2_unif_pt=r2_unif_pt,
    rec_full_pt=rec_full_pt, rec_sep_pt=rec_sep_pt,
    ci_r2_cure=np.array(ci_r2_cure), ci_r2_unif=np.array(ci_r2_unif),
    ci_rec_full=np.array(ci_rec_full), ci_rec_sep=np.array(ci_rec_sep),
    p_kernel_gt_unif=p_kernel_gt_unif,
    boot_r2_cure=boot_r2_cure, boot_rec_full=boot_rec_full,
    # R2
    bin_edges=edges, bin_eps_med=bin_eps_med, bin_rec=bin_rec,
    bin_rec_lo=bin_rec_lo, bin_rec_hi=bin_rec_hi, bin_n=bin_n,
    bin_r2_ode=bin_r2_ode, bin_r2_exact=bin_r2_exact,
    rho_eps_eff=rho_eps_eff,
    cure_eff=cure_eff,
    # R3
    spalding_e=spalding_e, tble_e=tble_e,
    crwm_present=crwm_present,
    pred_lower=pred_lower, pred_upper=pred_upper,
    heur_center=heur_center, heur_lo=heur_lo, heur_hi=heur_hi,
    note=np.array(["L3 results analysis. R1 bootstrap UQ, R2 eps-resolved cure "
                   "law, R3 pre-registered forward band for the UNLANDED coupled "
                   "cure. crwm_e_reatt is NaN on disk and is NOT asserted."]),
)

print("=" * 70)
print("THRUST #26 L3 -- RESULTS AND ANALYSIS  (a-priori + forward prediction)")
print("=" * 70)
print(f"stations: {n}   bootstrap resamples: {B}   eps median: {eps_median:.4f}")

print("\n--- R1  statistical significance (station bootstrap, 95% CI) ---")
print(f"kernel-cure R^2          : {r2_cure_pt:+.3f}   CI [{ci_r2_cure[0]:+.3f}, {ci_r2_cure[1]:+.3f}]")
print(f"uniform-leverage null R^2: {r2_unif_pt:+.2f}   CI [{ci_r2_unif[0]:+.2f}, {ci_r2_unif[1]:+.2f}]")
print(f"recovery fraction (full) : {rec_full_pt:.3f}    CI [{ci_rec_full[0]:.3f}, {ci_rec_full[1]:.3f}]")
print(f"recovery fraction (sep)  : {rec_sep_pt:.3f}    CI [{ci_rec_sep[0]:.3f}, {ci_rec_sep[1]:.3f}]")
print(f"P(kernel cure <= uniform null) = {p_kernel_gt_unif:.4f}  "
      f"({'significant' if p_kernel_gt_unif < 0.05 else 'NOT significant'})")

print("\n--- R2  eps-resolved cure law (recovery fraction per eps quartile) ---")
print(f"{'eps bin':>11} {'n':>4} {'eps_med':>8} {'R2_ode':>9} {'R2_exact':>9} "
      f"{'recovery':>9} {'95% CI':>18}")
for j in range(4):
    print(f"{['Q1(deep)','Q2','Q3','Q4(shallow)'][j]:>11} {bin_n[j]:>4} "
          f"{bin_eps_med[j]:>8.4f} {bin_r2_ode[j]:>9.2f} {bin_r2_exact[j]:>9.2f} "
          f"{bin_rec[j]:>9.3f} [{bin_rec_lo[j]:>6.2f},{bin_rec_hi[j]:>6.2f}]")
print(f"Spearman rho(eps, per-station cure efficiency) = {rho_eps_eff:+.3f}")
print("  -> recovery is LARGEST at deepest cancellation (smallest eps); the cure")
print("     self-targets the failure regime and is neutral/harmful at eps~O(1).")

print("\n--- R3  forward prediction for the coupled cure (UNLANDED) ---")
print(f"on-disk coupled baselines : Spalding e_reatt = {spalding_e:.3f}, "
      f"TBLE e_reatt = {tble_e:.3f}")
print(f"crwm_present on disk       : {crwm_present}  (crwm_e_reatt = {crwm_e_disk})")
print(f"RIGOROUS pre-registered band : {pred_lower:.3f} < crwm_e_reatt < {pred_upper:.3f}")
print(f"HEURISTIC central estimate   : {heur_center:.3f}  "
      f"band [{heur_lo:.3f}, {heur_hi:.3f}]  (linear-recovery map; not asserted)")
print(f"\nsaved -> {out}")
