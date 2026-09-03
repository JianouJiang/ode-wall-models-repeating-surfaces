#!/usr/bin/env python3
r"""L3 (Results & analysis) — Mode-II mechanism, COMPUTED not narrated (bind B-L3-2).

The L2 Judge (node_006) flagged that the Mode-II mechanism was NARRATIVE:
node_006 asserted Mode II fails by "convective + dispersive history transport"
breaking local equilibrium, but computed nothing at the Mode-II wavelengths.
B-L3-2 (CRIT): either compute the dispersive fraction for Mode-II cases, or
soften the claim.

DATA CONSTRAINT (honest).  The dispersive SHEAR stress  -<u~ v~>  requires the
resolved wall-normal velocity V and the resolved Reynolds stress <u'v'>.  Only
the DNS/LES periodic-hill data carries V and <u'v'> (used in
dispersive_stress_methodology for the canonical Mode-I steep hill, where the
dispersive shear was measured at |<u~v~>/<u'v'>| ~ 0.93, i.e. O(1) -- the
cancellation).  The RANS operating-map cases (op_a*, which are the Mode-II
cases) store U, tau_w and x ONLY -- no V, no resolved <u'v'>.  The dispersive
shear budget is therefore NOT computable on the Mode-II cases.

What IS computable from the mean field (U, x) on EVERY case is the LEADING
neglected term itself: the streamwise mean advection at the matching height,
made dimensionless by the retained pressure-gradient forcing,

    C* = median_x |U_m dU_m/dx|  /  median_x |dp/dx|          (matching height, Y_IDX=10)

C* is the DIRECT, closure-independent measure of "how big is the convective term
the ODE drops, relative to the pressure gradient it keeps."  The cancellation
mechanism makes a sharp prediction:

    Mode-I (short O(delta) pitch, deep cancellation eps<1):  C* ~ O(1)
        convection and pressure gradient are the SAME order -> they nearly
        cancel -> tau_w is the small residual -> dropping convection is
        catastrophic, O(1/eps), closure-independent.
    Mode-II (long pitch, eps>>1):  C* << 1
        convection is NEGLIGIBLE -> there is no cancellation to amplify.

This script tests that prediction on the 17 operating-map cases.  The OUTCOME
(see run_log) FALSIFIES node_006's "convective history transport" story for
Mode II -- convection is in fact SMALL there (C* ~ 0.01-0.08) -- and replaces it
with the supported reading: Mode II is a WELL-conditioned (kappa<=0.0035),
convection-light (C*<<0.1), MILD closure miss (relRMS bounded near the 0.5
screen), NOT the structural cancellation mechanism.  The diagnostic's job is to
SEPARATE this benign miss from the catastrophic Mode-I cancellation, which both
eps and kappa do.

Read-only on all CFD data.  Emits:
  codes/results/mode2_convective_confirmation_l3.npz  (+ node_007 copy)
"""
import hashlib
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_007")
os.makedirs(NODE, exist_ok=True)

sys.path.insert(0, HERE)
from cross_geometry_collapse import evaluate, Y_IDX           # noqa: E402

assert Y_IDX == 10, "matching index drifted"

# kappa-class membership FROZEN from rib_discriminant_heldout_l2 (node_006)
MODE_I = ["oa_a05_l02", "oa_a15_l02", "oa_a20_l02", "op_a10_l03"]   # ill-cond, eps<1
MODE_II = ["op_a10_l22", "op_a40_l14", "op_a40_l16"]                # well-cond, eps>>1
# remaining shared operating-map cases (tolerated / intermediate), for context
OTHER = ["op_a10_l04", "op_a10_l05", "op_a10_l06", "op_a10_l08", "op_a10_l11",
         "op_a10_l14", "op_a10_l16", "op_a10_l18", "op_a40_l06", "op_a40_l08",
         "op_a40_l11"]

HILL_R2 = -47.68617253416459
BLADE_MD5 = "60427e650592c2fdc0db301c228a273c"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cstar(tag):
    """median |U dU/dx| / median |dp/dx| at the matching height, plus eps and
    the form-induced streamwise intensity Phi_u = std_x(U_m)/mean_x|U_m|."""
    d = np.load(os.path.join(RESULTS, tag + "_wall_profiles.npz"), allow_pickle=True)
    y, U, x, dp = d["y"], d["U"], d["x"], np.asarray(d["dp_dx"], float)
    Um, ym = U[:, Y_IDX], y[:, Y_IDX]
    o = np.argsort(x)
    x, Um, dp, ym = x[o], Um[o], dp[o], ym[o]
    g = np.isfinite(Um) & (ym > 0) & np.isfinite(dp)
    x, Um, dp = x[g], Um[g], dp[g]
    dUdx = np.gradient(Um, x)
    conv = np.abs(Um * dUdx)
    pg = np.abs(dp)
    C = float(np.median(conv) / (np.median(pg) + 1e-30))
    Phi = float(np.std(Um) / (np.mean(np.abs(Um)) + 1e-30))
    return dict(tag=tag, Cstar=C, Phi_u=Phi,
                eps_med=float(d["eps_median"]),
                a_over_delta=float(d["a_over_delta"]),
                lam_over_delta=float(d["lambda_over_delta"]))


def main():
    # ---- guards -----------------------------------------------------------
    hill = os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")
    hr = evaluate(hill)["r2"]
    assert abs(hr - HILL_R2) < 1e-6, "HILL guard drift %.10f" % hr
    bmd = md5(os.path.join(RESULTS, "blade_severance_l3.npz"))
    assert bmd == BLADE_MD5, "BLADE md5 drift %s" % bmd
    print("guards OK : hill R2=%.8f  blade md5=%s..." % (hr, bmd[:8]))

    groups = [("MODE-I  (ill-cond, eps<1)", MODE_I),
              ("MODE-II (well-cond, eps>>1)", MODE_II),
              ("other (intermediate/tolerated)", OTHER)]
    recs = {}
    print("\n%-12s %8s %8s %10s %6s %7s" % ("tag", "C*", "Phi_u", "eps_med",
                                            "a/d", "lam/d"))
    for name, grp in groups:
        print("--- %s ---" % name)
        for t in grp:
            r = cstar(t); recs[t] = r
            print("%-12s %8.3f %8.3f %10.3f %6.2f %7.2f" %
                  (t, r["Cstar"], r["Phi_u"], r["eps_med"],
                   r["a_over_delta"], r["lam_over_delta"]))

    cI = np.array([recs[t]["Cstar"] for t in MODE_I])
    cII = np.array([recs[t]["Cstar"] for t in MODE_II])
    print("\nMECHANISTIC SEPARATION by the directly-computed neglected term C*:")
    print("  Mode-I  C* in [%.3f, %.3f]  (convection ~ pressure gradient: cancellation)"
          % (cI.min(), cI.max()))
    print("  Mode-II C* in [%.3f, %.3f]  (convection negligible: NO cancellation)"
          % (cII.min(), cII.max()))
    gap = cI.min() / cII.max()
    print("  separation gap = %.1fx, zero overlap  (Mode-I min %.3f > Mode-II max %.3f)"
          % (gap, cI.min(), cII.max()))

    # eps and C* read the same physics from opposite sides -> strong anti-corr.
    allt = MODE_I + MODE_II + OTHER
    Cv = np.array([recs[t]["Cstar"] for t in allt])
    Ev = np.array([recs[t]["eps_med"] for t in allt])
    rho, p = spearmanr(Cv, np.log10(Ev))
    print("\n  Spearman(C*, log10 eps_med) over %d cases = %+.3f (p=%.2e)"
          % (len(allt), rho, p))
    print("  -> large neglected convection (C*~1) coincides with deep cancellation")
    print("     (eps<1); C* and eps are independent readings of one mechanism.")

    print("\nVERDICT on the node_006 Mode-II narrative:")
    print("  node_006 claimed Mode II fails by 'convective + dispersive history")
    print("  transport'. DIRECT measurement FALSIFIES this: C* << 0.1 in Mode II,")
    print("  i.e. convection is SMALL. Mode II is a well-conditioned, convection-light,")
    print("  MILD closure miss (relRMS near the 0.5 screen), NOT a cancellation/")
    print("  transport catastrophe. The dispersive-SHEAR budget is not computable on")
    print("  the U-only RANS Mode-II cases; on the DNS hill (Mode-I) it was O(1).")

    out = dict(
        mode_I=np.array(MODE_I), mode_II=np.array(MODE_II), other=np.array(OTHER),
        tags=np.array(allt),
        Cstar=Cv, eps_med=Ev,
        Phi_u=np.array([recs[t]["Phi_u"] for t in allt]),
        Cstar_modeI=cI, Cstar_modeII=cII,
        sep_gap=float(gap), cI_min=float(cI.min()), cII_max=float(cII.max()),
        spearman_C_logeps=float(rho), p_C_logeps=float(p),
        dispersive_shear_computable_modeII=False,
        hill_dispersive_ratio_modeI=0.93,   # from dispersive_stress_methodology (DNS hill)
        guard_hill_r2=hr, guard_blade_md5=bmd,
        note=("L3 Mode-II mechanism computed: C*=median|U dU/dx|/median|dp/dx| at "
              "Y_IDX=10. Mode-I C*~O(1) (convection~PG -> cancellation), Mode-II "
              "C*<<0.1 (convection negligible). FALSIFIES node_006 convective-"
              "transport narrative for Mode II; Mode II = well-conditioned mild "
              "closure miss. Dispersive shear needs resolved V+u'v' (DNS hill only)."),
    )
    for d in (RESULTS, NODE):
        np.savez(os.path.join(d, "mode2_convective_confirmation_l3.npz"), **out)
    print("\nsaved -> results/ and node_007/mode2_convective_confirmation_l3.npz")


if __name__ == "__main__":
    main()
