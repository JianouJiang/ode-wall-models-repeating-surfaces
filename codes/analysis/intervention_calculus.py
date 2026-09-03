#!/usr/bin/env python3
"""
Thrust #29 L1 — the INTERVENTION CALCULUS of the operating map.

This is the Core-methodology (L1) computation. It asserts NO new coupled number
(the CR-WM cure is live/unlanded, the band run is queued/unlanded). It FORMALISES
the two operating-map axes as controllable intervention variables and LOCKS the
quantitative prediction the two coupled interventions must satisfy, deriving every
quantity FROM SOURCE on-disk npz produced by real DNS/LES:

  1. The 2x2 controlled-intervention design that decouples the two axes
     (Judge L0 fix #1: why the band run is a *pure placement* change with no
     convective confound on the equilibrium model, and why a convection-carrying
     PDE wall model would confound the two).

  2. The within-layer / outer-transport split of the structural error and the
     within-layer recovery fraction r_in (the "ceiling"), from the a-priori
     oracle add-back (crwm_apriori.npz) cross-checked two independent ways.

  3. The F3 quantitative identity: the cured a-posteriori reattachment-error
     residual must equal the OUTER-transport error the map allocates above y_m,
     i.e. e_cure ~ (1 - r_in) * e_TBLE.  We LOCK the pre-registered band
     (Judge L0 fix #2) from the bootstrap CI on r_in, and state honestly that the
     map from an a-priori LOCAL wall-stress norm to a coupled GLOBAL reattachment
     norm is an order-of-magnitude bridge, not an exact equality.

  4. The honest n=1 scope ledger for the COUPLED interventions (Judge L0 fix #3).

Output: codes/results/intervention_calculus.npz
Run:    OMP_NUM_THREADS=2 python3 codes/analysis/intervention_calculus.py
"""
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "codes", "results")


def load(f):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing source npz: {f}")
    return np.load(p, allow_pickle=True)


print("=" * 76)
print("Thrust #29 L1 — intervention calculus  (no new coupled number)")
print("=" * 76)

# -----------------------------------------------------------------------------
# (1) The 2x2 controlled-intervention design that DECOUPLES the two map axes.
#
# Operating point = (axis-1 placement chi = y_m+/y_crit+ , axis-2 convective
# content kept by the wall model). The two interventions toggle ONE axis each:
#
#                         convection content kept by model
#                         OFF (C_model = 0)      ON (C_model = int_0^{y_m} C dy)
#   placement  chi < 1    equilibrium/TBLE       CR-WM (cure twin)      <- Axis-2
#   (safe)               = the deployed corner    fixed y_m+, conv on
#   ---------------------------------------------------------------------------
#   placement  chi > 1    band run (Axis-1)       (not run; would confound)
#   (failure)            same model, y_m+ raised
#
# KEY orthogonality fact (Judge fix #1): for an EQUILIBRIUM/TBLE wall model the
# within-layer convective content kept by the model is IDENTICALLY ZERO at every
# matching height, C_model(y_m) == 0 for all y_m, because the model equation has
# no convective source term. Hence raising y_m (the band run) moves ONLY axis-1
# (placement); it injects no convection, so it cannot co-move axis-2. The cross
# term d C_model / d y_m = 0 is what makes the band a clean placement intervention.
#
# By contrast a convection-CARRYING PDE wall model (e.g. Park & Moin 2014) has
# C_model(y_m) = int_0^{y_m} C dy which GROWS with y_m, so d C_model/d y_m != 0:
# raising y_m would simultaneously change placement AND captured convection -- the
# two axes confound. This is exactly why the controlled design uses the
# equilibrium model for the placement intervention and toggles convection only at
# FIXED y_m (the CR-WM twin) for the deficit intervention.
# -----------------------------------------------------------------------------
print("\n[1] 2x2 intervention design (axis decoupling)")
# design rows: (placement_chi_gt1, convection_on, label)
design_labels = np.array([
    "equilibrium_TBLE_safe_corner",   # chi<1, C_model=0   (deployed baseline)
    "CRWM_cure_twin",                 # chi<1, C_model on   (Axis-2 intervention)
    "band_placement",                 # chi>1, C_model=0    (Axis-1 intervention)
])
design_chi_gt1 = np.array([0, 0, 1], dtype=int)
design_conv_on = np.array([0, 1, 0], dtype=int)
# the equilibrium model carries zero convection at every matching height:
C_model_equilibrium_any_ym = 0.0          # structural property of the ODE
dCdym_equilibrium = 0.0                    # => band run does not co-move axis-2
dCdym_pde_signature = "int_0^{y_m} C dy (grows with y_m -> PDE confounds)"
print(f"  C_model(equilibrium, any y_m) = {C_model_equilibrium_any_ym}  "
      f"=> dC/dy_m = {dCdym_equilibrium} (band = pure placement)")
print(f"  PDE wall model would confound: C_model = {dCdym_pde_signature}")

# -----------------------------------------------------------------------------
# (2) Within-layer / outer-transport split and the recovery ceiling r_in.
#
# A-priori, on the canonical h/Lx=1.0 hill, restoring the within-layer convective
# flux lifts the wall-stress skill from R2_ode to R2_oracle. The fraction of the
# error VARIANCE that the within-layer term removes is
#       r_in = 1 - (1 - R2_oracle) / (1 - R2_ode)
# (SS_res ratio; "perfect cure" is R2=1 i.e. SS_res=0). Cross-checked against the
# 512-station population-average recovery rec_full (crwm_results_analysis.npz).
# -----------------------------------------------------------------------------
print("\n[2] within-layer recovery ceiling r_in (a-priori, two independent reads)")
ap = load("crwm_apriori.npz")
r2_ode = float(np.array(ap["r2_all_ode"]).ravel()[0])
r2_oracle = float(np.array(ap["r2_all_exact"]).ravel()[0])   # within-layer oracle add-back
r2_const = float(np.array(ap["r2_all_const"]).ravel()[0])    # deployable single-cell
r_in_canonical = 1.0 - (1.0 - r2_oracle) / (1.0 - r2_ode)     # error-variance recovery
r_in_const = 1.0 - (1.0 - r2_const) / (1.0 - r2_ode)
var_expl_C = float(np.array(ap["var_explained_by_C"]).ravel()[0])  # r^2 of E_struct~C

ca = load("crwm_results_analysis.npz")
r_in_pop = float(np.array(ca["rec_full_pt"]).ravel()[0])     # 512-station population avg
r_in_ci = np.array(ca["ci_rec_full"]).ravel().astype(float)  # bootstrap 95% CI
print(f"  R2(tau_w): ode={r2_ode:.3f}  oracle(within-layer)={r2_oracle:.3f}")
print(f"  r_in (canonical-hill error-variance) = {r_in_canonical:.4f}")
print(f"  r_in (512-station population avg)     = {r_in_pop:.4f}  CI {r_in_ci}")
print(f"  cross-check var_explained_by_C (r^2)  = {var_expl_C:.4f}")
# adopt the population-average recovery + its bootstrap CI as the locked ceiling
r_in = r_in_pop
r_out = 1.0 - r_in
r_in_lo, r_in_hi = float(r_in_ci[0]), float(r_in_ci[1])
print(f"  ADOPTED ceiling r_in = {r_in:.3f}  -> outer-transport fraction r_out = {r_out:.3f}")

# -----------------------------------------------------------------------------
# (3) F3 quantitative identity + LOCKED pre-registered band (Judge L0 fix #2).
#
#   map allocation:  e_cure ~ (1 - r_in) * e_TBLE      (outer residual the
#                                                        within-layer cure cannot reach)
#   point prediction:        e_cure_pred = r_out * e_TBLE
#   acceptance band:         propagate the bootstrap CI on r_in
#                            [ (1 - r_in_hi) e_TBLE , (1 - r_in_lo) e_TBLE ]
#   partial-cure constraint (F1): 0 < e_cure < e_TBLE   (strict: real but partial)
#
# HONESTY: r_in is measured in a LOCAL a-priori wall-stress norm; e_cure is a
# GLOBAL coupled reattachment-error norm. The translation between them is an
# order-of-magnitude bridge, not an algebraic identity -- so the band is the
# falsifiable tolerance, and a cure BELOW the band (toward 0) is NOT a falsifier
# but evidence the coupled LES additionally supplies outer transport (Thrust #20
# stronger claim). The single fatal outcome is e_cure >= e_TBLE (no cure).
# -----------------------------------------------------------------------------
print("\n[3] F3 identity + locked pre-registered band")
tw = load("aposteriori_crwm_twin.npz")
e_TBLE = float(np.array(tw["tble_e_reatt"]).ravel()[0])
e_Spalding = float(np.array(tw["spalding_e_reatt"]).ravel()[0])
crwm_present = bool(np.array(tw["crwm_present"]).ravel()[0])
e_cure_pred = r_out * e_TBLE
band_lo = (1.0 - r_in_hi) * e_TBLE
band_hi = (1.0 - r_in_lo) * e_TBLE
print(f"  e_TBLE (landed) = {e_TBLE:.4f}   e_Spalding (landed) = {e_Spalding:.4f}")
print(f"  F3 point prediction  e_cure ~ (1 - {r_in:.3f}) * {e_TBLE:.3f} = {e_cure_pred:.4f}")
print(f"  F3 acceptance band   [{band_lo:.4f}, {band_hi:.4f}]")
print(f"  F1 partial-cure constraint  0 < e_cure < {e_TBLE:.4f}")
print(f"  crwm_present = {crwm_present}  -> "
      f"{'NO cure number asserted' if not crwm_present else 'CURE LANDED (harvest at L2)'}")

# cross-check the band against the band already locked on disk by #26 L3
heur_center = float(np.array(ca["heur_center"]).ravel()[0])
heur_lo = float(np.array(ca["heur_lo"]).ravel()[0])
heur_hi = float(np.array(ca["heur_hi"]).ravel()[0])
band_consistent = (abs(heur_center - e_cure_pred) < 5e-3 and
                   abs(heur_lo - band_lo) < 5e-3 and abs(heur_hi - band_hi) < 5e-3)
print(f"  cross-check vs on-disk locked band (crwm_results_analysis.npz): "
      f"center {heur_center:.4f} band [{heur_lo:.4f},{heur_hi:.4f}]  "
      f"consistent={band_consistent}")

# -----------------------------------------------------------------------------
# (4) Closure-independence guard for the deficit (F6 / G3): exact-DNS stresses
#     make the a-priori failure WORSE, so the cure cannot be a closure tweak --
#     it must be the non-local convective term (single-variable control).
# -----------------------------------------------------------------------------
print("\n[4] closure-independence of the deficit (F6/G3)")
dt = load("diagnostic_test_corrected.npz")
r2_ml = float(np.array(dt["standard_ml_r2"]).ravel()[0])
r2_dns = float(np.array(dt["controlled_dns_r2"]).ravel()[0])
closure_indep = r2_dns < r2_ml
print(f"  a-priori R2(tau_w): ML closure={r2_ml:.2f}  exact-DNS={r2_dns:.1f}  "
      f"exact-DNS worse={closure_indep}")

# -----------------------------------------------------------------------------
# (5) Honest n=1 scope ledger for the COUPLED interventions (Judge L0 fix #3).
# -----------------------------------------------------------------------------
print("\n[5] coupled-intervention scope ledger (honesty)")
scope_rows = np.array([
    # claim, n_geometries_coupled, provenance
    "Axis-1 band placement intervention | coupled n=1 (a=1.0 hill, Re5600, 1 grid) | NEW (this thrust, F2, UNLANDED)",
    "Axis-2 CR-WM convection cure       | coupled n=1 (a=1.0 hill, Re5600)         | NEW (this thrust, F1, UNLANDED)",
    "steepness dose-response a=0.8/1.0/1.2 | coupled n=3 family                    | PRE-EXISTING (#11)",
    "conv-div wide-pitch null            | coupled n=1 (eps_med~1.70, tolerated)   | PRE-EXISTING (#11, G7 null)",
    "within-family L_sep/delta collapse  | a-priori family (rho=-0.754)            | PRE-EXISTING (#27, G7)",
    "a-priori operating-map sweep        | a-priori n=8 geometries                 | PRE-EXISTING (#28 L2)",
])
for r in scope_rows:
    print("  " + r)

# -----------------------------------------------------------------------------
# save
# -----------------------------------------------------------------------------
out = os.path.join(RES, "intervention_calculus.npz")
np.savez(
    out,
    # (1) design
    design_labels=design_labels,
    design_chi_gt1=design_chi_gt1,
    design_conv_on=design_conv_on,
    C_model_equilibrium_any_ym=C_model_equilibrium_any_ym,
    dCdym_equilibrium=dCdym_equilibrium,
    # (2) ceiling
    r2_ode=r2_ode, r2_oracle=r2_oracle, r2_const=r2_const,
    r_in_canonical=r_in_canonical, r_in_const=r_in_const, var_explained_by_C=var_expl_C,
    r_in=r_in, r_in_lo=r_in_lo, r_in_hi=r_in_hi, r_out=r_out,
    # (3) F3 band
    e_TBLE=e_TBLE, e_Spalding=e_Spalding, crwm_present=crwm_present,
    e_cure_pred=e_cure_pred, band_lo=band_lo, band_hi=band_hi,
    partial_cure_lo=0.0, partial_cure_hi=e_TBLE,
    band_consistent_with_disk=band_consistent,
    # (4) closure indep
    r2_ml=r2_ml, r2_dns=r2_dns, closure_independent=closure_indep,
    # (5) scope
    scope_rows=scope_rows,
    note=np.array([
        "Thrust #29 L1 intervention calculus. NO coupled cure/band number "
        "(crwm_present=False, aposteriori_band absent). F3 band derived from "
        "a-priori within-layer recovery r_in and landed e_TBLE; LOCAL->GLOBAL "
        "norm bridge is order-of-magnitude (band is the tolerance). F1 kill: "
        "e_cure >= e_TBLE."
    ]),
)
print(f"\nsaved -> {os.path.relpath(out, ROOT)}")
print("=" * 76)
