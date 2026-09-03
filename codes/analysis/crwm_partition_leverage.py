#!/usr/bin/env python3
"""
crwm_partition_leverage.py  --  L1 (core methodology) for thrust #26.

Computes the NEAR-WALL LEVERAGE of the wall-normal partition of the missing
convective forcing, from REAL on-disk a-priori files ONLY, using explicit keys.

Definition.  The wall-model BVP makes tau_w a linear functional of the forcing
F(y); its sensitivity kernel G(xi)=d tau_w/dF(xi) is maximal at the wall and
vanishes at y_m, so near-wall forcing is disproportionately effective on tau_w.
The within-layer band carries only a small share of the missing convection by
MAGNITUDE (f_within), yet restoring it recovers a large share of the wall-stress
ERROR (f_recovery).  Their ratio is the leverage:

    L = f_recovery / f_within.

Two honest recovery measures are reported (the manuscript uses both):
  * variance:  f_recovery_var = var_explained_by_C        (~0.46)
  * distance:  f_recovery_dist = (r2_exact - r2_ode)/(0 - r2_ode)  (~0.56,
               the exact-within-layer-flux oracle ceiling, R2 -47.7 -> -21.0)
and f_within = med_within_layer_frac (~0.029) from the decomposition file.

Inputs (must exist):
    codes/results/crwm_apriori.npz
    codes/results/outer_transport_decomposition.npz
Output:
    codes/results/crwm_partition_leverage.npz

No value is hardcoded; the script errors loudly if a key is absent.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
APRIORI = os.path.join(RES, "crwm_apriori.npz")
OTD = os.path.join(RES, "outer_transport_decomposition.npz")
OUT = os.path.join(RES, "crwm_partition_leverage.npz")


def scalar(d, key, path):
    if key not in d.files:
        sys.exit(f"KEY '{key}' ABSENT in {path}; keys={'|'.join(d.files)}. "
                 f"Refusing to write a fabricated value.")
    return float(np.ravel(d[key])[0])


def main():
    for p in (APRIORI, OTD):
        if not os.path.isfile(p):
            sys.exit(f"REQUIRED INPUT MISSING: {p}")

    a = np.load(APRIORI, allow_pickle=True)
    o = np.load(OTD, allow_pickle=True)

    r2_ode = scalar(a, "r2_all_ode", APRIORI)
    r2_exact = scalar(a, "r2_all_exact", APRIORI)
    r2_const = scalar(a, "r2_all_const", APRIORI)
    var_expl = scalar(a, "var_explained_by_C", APRIORI)
    corr = scalar(a, "corr_struct_C", APRIORI)

    f_within = scalar(o, "med_within_layer_frac", OTD)
    f_outer = scalar(o, "med_outer_frac", OTD)
    y50 = scalar(o, "med_y50_over_ym", OTD)

    if f_within <= 0:
        sys.exit(f"non-physical within-layer forcing fraction {f_within}")

    f_recovery_var = var_expl
    f_recovery_dist = (r2_exact - r2_ode) / (0.0 - r2_ode)

    leverage_var = f_recovery_var / f_within
    leverage_dist = f_recovery_dist / f_within

    np.savez(
        OUT,
        r2_ode=r2_ode, r2_const=r2_const, r2_exact=r2_exact,
        var_explained_by_C=var_expl, corr_struct_C=corr,
        within_forcing_fraction=f_within, outer_forcing_fraction=f_outer,
        y50_over_ym=y50,
        recovery_variance=f_recovery_var, recovery_distance=f_recovery_dist,
        near_wall_leverage_variance=leverage_var,
        near_wall_leverage_distance=leverage_dist,
        coupled_recovery_ceiling_variance=f_recovery_var,
        coupled_recovery_ceiling_distance=f_recovery_dist,
        R_coupled=np.nan,  # harvested from xiao_wmles_a1p0_crwm when it lands
        provenance="derived from crwm_apriori.npz + outer_transport_decomposition.npz; no new simulation",
    )
    print("LEVERAGE_RESULT "
          f"r2_ode={r2_ode:.2f} r2_const={r2_const:.2f} r2_exact={r2_exact:.2f} "
          f"f_within={f_within:.4f} f_outer={f_outer:.4f} y50/ym={y50:.2f} "
          f"recovery_var={f_recovery_var:.3f} recovery_dist={f_recovery_dist:.3f} "
          f"L_var={leverage_var:.1f} L_dist={leverage_dist:.1f} -> {OUT}")


if __name__ == "__main__":
    main()
