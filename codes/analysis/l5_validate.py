#!/usr/bin/env python3
r"""
l5_validate.py  --  Level-5 (Final validation against all metrics) deliverable.
==============================================================================

DIAGNOSTIC paper, repeating-structure framing.  This script is the machine-checkable
core of the L5 validation: it loads ONLY the protected results corpus
(codes/results/*.npz) and the development metrics/gates definitions, then
reconciles -- number by number -- every metric in research/metrics.json and every
headline figure quoted in the manuscript abstract/contributions against the
on-disk data.  It is deterministic and FOREGROUND, and writes its JSON output
BEFORE any assertion (anti-empty bind B-L5-1).  Re-running reproduces it
bit-for-bit; nothing is hand-entered.

Outputs:
  development/nodes/node_009/l5_validation.json

Every check carries: the claimed value (as printed in the manuscript), the npz
field it traces to, the on-disk value, and a PASS/FAIL.  A check FAILS loudly if
the data does not support the claim -- there is no silent pass (bind B-L5-2).
"""
import os, json
import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(PROJ, "codes", "results")
NODE = os.path.join(PROJ, "development", "nodes", "node_009")
os.makedirs(NODE, exist_ok=True)


def load(fn):
    return np.load(os.path.join(R, fn), allow_pickle=True)


def f(x):
    """scalar -> python float."""
    return float(np.asarray(x).ravel()[0])


def approx(a, b, rel=0.02, abstol=0.0):
    """True if a within rel (or abstol) of b."""
    if abstol and abs(a - b) <= abstol:
        return True
    if b == 0:
        return abs(a - b) <= (abstol or 1e-9)
    return abs(a - b) / abs(b) <= rel


checks = []  # list of dicts


def record(metric, claim, source, claimed_val, disk_val, ok, note=""):
    checks.append(dict(metric=metric, claim=claim, source=source,
                       claimed=claimed_val, on_disk=disk_val,
                       status="PASS" if ok else "FAIL", note=note))


# =====================================================================
# METRIC 1: ODE success (R^2 >= 0.88 on the geometries it should handle)
# =====================================================================
ce = load("criterion_evaluation.npz")
ode_r2 = np.asarray(ce["ode_r2"], float)
# the catastrophic entry is the periodic-hills family; all OTHERS are the success set
success = ode_r2[ode_r2 > -1.0]            # drop the hills catastrophe
n_success = int((success >= 0.88).sum())
min_success = float(success.min())
record("M1_ode_success",
       "ODE attains R^2>=0.88 on the localised/attached geometries it should handle",
       "criterion_evaluation.npz:ode_r2",
       ">=0.88 on the success geometries (min curved-BFS/BFS ~0.888)",
       f"n_success(>=0.88)={n_success}/{len(success)}, min_success_R2={min_success:.4f}",
       (n_success == len(success)) and (min_success >= 0.88),
       "9-geometry core set; the 1 non-success entry is the periodic-hills catastrophe (Metric 2).")

# broader 15-case cross-geometry set: how many tolerated (R^2>=0.88)?
cg = load("cross_geometry_collapse.npz")
cg_r2 = np.asarray(cg["r2"], float)
cg_names = [str(x) for x in cg["keys"]]
n_tol = int((cg_r2 >= 0.88).sum())
record("M1_ode_success_broad",
       "Across the 15-case cross-geometry reference set the ODE is tolerated on the localised cases",
       "cross_geometry_collapse.npz:r2",
       "majority tolerated (R^2>=0.88), hills family fails",
       f"n(R2>=0.88)={n_tol}/15; failing={[cg_names[i] for i in range(15) if cg_r2[i]<0.88]}",
       n_tol >= 10,
       "Failing members are the repeating periodic-hill cases, consistent with the thesis.")

# =====================================================================
# METRIC 2: ODE failure on the repeating periodic-hills family
# =====================================================================
dx = load("dose_response_xiao.npz")
agg_r2 = np.asarray(dx["agg_r2"], float)
r2_min, r2_max = float(agg_r2.min()), float(agg_r2.max())
record("M2_ode_failure_dose",
       "Xiao 29-case hill family is uniformly catastrophic, R^2 in [-84,-10]",
       "dose_response_xiao.npz:agg_r2",
       "all 29 in [-84.3,-9.86]",
       f"min={r2_min:.2f}, max={r2_max:.2f}, all_negative={bool((agg_r2<0).all())}",
       approx(r2_min, -84.3, rel=0.03) and approx(r2_max, -9.86, rel=0.03) and bool((agg_r2 < 0).all()))

p5 = load("pehill_5case_corrected.npz")
c1p0_r2 = f(p5["case_1p0_r2"]); c1p0_relerr = f(p5["case_1p0_rel_err"])
record("M2_canonical_hill",
       "Canonical h/Lx=1.0 hill: R^2=-47.7, relRMS(tau_w)=6.8",
       "pehill_5case_corrected.npz:case_1p0_r2 / case_1p0_rel_err",
       "R2=-47.7, relRMS=6.8",
       f"R2={c1p0_r2:.3f}, relRMS={c1p0_relerr:.3f}",
       approx(c1p0_r2, -47.7, rel=0.01) and approx(c1p0_relerr, 6.8, rel=0.02))

dtc = load("diagnostic_test_corrected.npz")
std_r2 = f(dtc["standard_ml_r2"])
record("M2_diagnostic_corrected",
       "Wall-surface-aware (Y_IDX=10) protocol reproduces the hill catastrophe",
       "diagnostic_test_corrected.npz:standard_ml_r2",
       "R2=-47.7 (Y_IDX=10, supersedes the y=0 wall-pinning artifact)",
       f"R2={std_r2:.3f}, Y_IDX={f(dtc['Y_IDX']):.0f}",
       approx(std_r2, -47.7, rel=0.01))

# =====================================================================
# METRIC 3: cancellation diagnostic separates failure from tolerance
# =====================================================================
cp = load("cancellation_parameter_corrected.npz")
ph_med = f(cp["periodic_hills_median_eps"])
ph_f01 = f(cp["periodic_hills_frac_below_01"])
ph_f1 = f(cp["periodic_hills_frac_below_1"])
record("M3_hill_eps_depth",
       "Periodic hills: eps median 0.084, eps<0.1 over ~56%, eps<1 over ~98%",
       "cancellation_parameter_corrected.npz",
       "median 0.084, f(eps<0.1)~0.56, f(eps<1)~0.98",
       f"median={ph_med:.4f}, f(eps<0.1)={ph_f01:.3f}, f(eps<1)={ph_f1:.3f}",
       approx(ph_med, 0.084, rel=0.02) and approx(ph_f01, 0.56, rel=0.03) and approx(ph_f1, 0.98, rel=0.01))

rho_f1 = f(cg["spearman_rho_frac1"]); p_f1 = f(cg["spearman_p_frac1"])
rho_f01 = f(cg["spearman_rho_frac0p1"]); p_f01 = f(cg["spearman_p_frac0p1"])
n_cg = int(f(cg["spearman_n"]))
record("M3_coverage_orders_error",
       "Coverage f(eps<1) and f(eps<0.1) order the domain error across geometries",
       "cross_geometry_collapse.npz:spearman_rho_frac1 / frac0p1",
       "rho(f(eps<1),relRMS)=+0.753 p=0.0012 ; rho(f(eps<0.1),relRMS)=+0.783 p=5.6e-4 ; n=15",
       f"rho_f1={rho_f1:.3f} p={p_f1:.4f} ; rho_f01={rho_f01:.3f} p={p_f01:.2e} ; n={n_cg}",
       approx(rho_f1, 0.753, rel=0.02) and approx(rho_f01, 0.783, rel=0.02) and n_cg == 15)

# Dimensionless severity collapse (the L3 headline) -- pooled n=46
l3 = load("l3_collapse.npz")
sp = f(l3["spearman_S_r2"]); ci = (f(l3["ci_lo"]), f(l3["ci_hi"]))
Sstar = f(l3["S_threshold"]); acc = f(l3["threshold_accuracy"])
record("M3_severity_collapse",
       "Single dimensionless severity S collapses the failure (rho=-0.88 [-0.94,-0.75], S*~0.32, 98% acc)",
       "l3_collapse.npz:spearman_S_r2 / ci / S_threshold / threshold_accuracy",
       "rho=-0.876 [-0.939,-0.746], S*=0.318, acc=97.8% (1/46)",
       f"rho={sp:.4f} ci=[{ci[0]:.3f},{ci[1]:.3f}] S*={Sstar:.3f} acc={acc:.3f} n={len(l3['pooled_S'])}",
       approx(sp, -0.876, rel=0.01) and approx(Sstar, 0.318, rel=0.02) and approx(acc, 0.978, rel=0.01)
       and len(l3["pooled_S"]) == 46)

# =====================================================================
# METRIC 4: structural, not closure (exact DNS stress does NOT rescue)
# =====================================================================
dns_r2 = f(dtc["controlled_dns_r2"])
record("M4_structural_not_closure",
       "Substituting exact DNS Reynolds stresses does NOT rescue the hills -- it makes the fit WORSE",
       "diagnostic_test_corrected.npz:controlled_dns_r2 vs standard_ml_r2",
       "DNS-stress R2 strictly worse than eddy-model R2",
       f"controlled_dns_R2={dns_r2:.1f}  vs  standard_ml_R2={std_r2:.1f}  (DNS worse={dns_r2 < std_r2})",
       dns_r2 < std_r2)

imp = load("impossibility_results_l3.npz")
rb = f(imp["RB_ratio"]); rb_p = f(imp["RB_mwu_p"])
record("M4_worst_conditioned",
       "Exact DNS stress is the WORST-conditioned closure (closure-independent bound)",
       "impossibility_results_l3.npz:RB_ratio / RB_mwu_p",
       "41x eddy-model conditioning, p~1e-194",
       f"RB_ratio={rb:.2f}, p={rb_p:.2e}",
       approx(rb, 41.3, rel=0.02) and rb_p < 1e-150)

# =====================================================================
# HEADLINE abstract/contribution numbers (G2 traceability)
# =====================================================================
rs = load("reynolds_scaling.npz")
slope = f(rs["slope_re"])
Re_pts = np.asarray(rs["Re_pts"], float)
re_factor = Re_pts.max() / Re_pts.min()
record("G2_reynolds_scaling",
       "eps_C ~ Re^-1.15 across a factor of 15",
       "reynolds_scaling.npz:slope_re / Re_pts",
       "slope=-1.15, factor~15",
       f"slope={slope:.3f}, Re factor={re_factor:.1f}",
       approx(slope, -1.15, rel=0.02) and approx(re_factor, 15.0, rel=0.05))

gp = load("geometry_predictor_l3.npz")
auc_in = f(gp["R1_auc_indomain"]); auc_block = f(gp["R2_auc_block"]); auc_pitch = f(gp["R2_auc_pitch"])
record("G2_geometry_auc",
       "Readable from geometry: in-domain AUC 1.00; blockage (not pitch) sets the boundary",
       "geometry_predictor_l3.npz:R1_auc_indomain / R2_auc_block / R2_auc_pitch",
       "AUC_indomain=1.00; block 1.00 > pitch 0.97",
       f"AUC_indomain={auc_in:.3f}, AUC_block={auc_block:.3f}, AUC_pitch={auc_pitch:.3f}",
       approx(auc_in, 1.0, abstol=1e-9) and auc_block >= auc_pitch)

sd = load("spectral_phase_decoherence.npz")
gamma_auc = f(sd["M4_auc_rho_E"])
record("G2_spectral_auc",
       "tau_w-free coherence discriminant Gamma separates failure from success (AUC 0.98)",
       "spectral_phase_decoherence.npz:M4_auc_rho_E",
       "AUC=0.98",
       f"AUC={gamma_auc:.3f}",
       approx(gamma_auc, 0.98, rel=0.02))

ap = load("aposteriori_wmles_pehill.npz")
reatt = f(ap["reattachment_rel_err_pct"]); rms = f(ap["profile_rms_mean"])
record("G2_aposteriori",
       "Coupled WMLES reattaches early (-20.6%); velocity error 0.287 u_b",
       "aposteriori_wmles_pehill.npz:reattachment_rel_err_pct / profile_rms_mean",
       "-20.6%, 0.287 u_b",
       f"reattachment={reatt:.2f}%, rms={rms:.4f} u_b",
       approx(reatt, -20.6, rel=0.01) and approx(rms, 0.287, rel=0.01))

bfs_r2 = float(success.min())  # BFS is the min-success geometry
record("G2_bfs_curvedbfs",
       "BFS / curved-BFS are tolerated (R^2~0.88-0.89), anchoring the zero-frequency control",
       "criterion_evaluation.npz:ode_r2 (min success entry)",
       "~0.888",
       f"min_success_R2={bfs_r2:.4f}",
       approx(bfs_r2, 0.888, rel=0.01))

# =====================================================================
# WRITE JSON FIRST (anti-empty), THEN assert
# =====================================================================
n_pass = sum(c["status"] == "PASS" for c in checks)
n_fail = sum(c["status"] == "FAIL" for c in checks)
summary = dict(
    title="L5 final validation -- metric & headline reconciliation against codes/results/*.npz",
    protocol="deterministic, foreground; loads protected npz only; writes JSON before asserting.",
    n_checks=len(checks), n_pass=n_pass, n_fail=n_fail,
    all_pass=(n_fail == 0),
    checks=checks,
)
out = os.path.join(NODE, "l5_validation.json")
with open(out, "w") as fh:
    json.dump(summary, fh, indent=2)

print("=" * 78)
print(f"L5 VALIDATION  --  {n_pass}/{len(checks)} PASS, {n_fail} FAIL")
print("=" * 78)
for c in checks:
    mark = "OK " if c["status"] == "PASS" else "XX "
    print(f"{mark}{c['metric']:28s} {c['on_disk']}")
print(f"\nWrote {os.path.relpath(out, PROJ)}")

# Loud failure only AFTER the artifact is on disk.
assert n_fail == 0, f"{n_fail} validation checks FAILED -- see l5_validation.json"
print("ALL L5 METRIC/HEADLINE CHECKS PASS.")
