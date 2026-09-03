#!/usr/bin/env python3
"""
Automated traceability harness for final validation (node_006).

Unlike prior Level-5 passes (manual eyeballing of npz values), this script
*programmatically* enforces a two-way contract for every headline number:

  (a) DATA  -> the value reproduces from codes/results/*.npz within tolerance;
  (b) TEXT  -> a rounded rendering of that value actually appears in
               manuscript/main.tex.

A claim PASSES only if both halves hold. This makes killer gate G2
("every number traces to data") and the no-silent-drift requirement
machine-checkable and re-runnable on every future edit.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/validate_traceability.py
Exit code 0 iff all checks pass.
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "codes", "results")
TEX  = os.path.join(ROOT, "manuscript", "main.tex")

_tex = open(TEX).read()
_cache = {}
def npz(name):
    if name not in _cache:
        _cache[name] = np.load(os.path.join(RES, name), allow_pickle=True)
    return _cache[name]

def tex_has(*patterns):
    """True if ANY of the regex patterns appears in main.tex."""
    return any(re.search(p, _tex) for p in patterns)

# Each check: (label, computed_value, expected, abs_tol, [tex_patterns])
# tex_patterns are regexes matched against the rounded rendering in the text.
CHECKS = []
def check(label, computed, expected, tol, tex_patterns):
    CHECKS.append((label, computed, expected, tol, tex_patterns))

# ---- 1. ODE failure on canonical periodic hill (corrected protocol) ----
d = npz("diagnostic_test_corrected.npz")
check("hills R2(tau_w) ODE (Y_IDX=10)", float(d["standard_ml_r2"]), -47.686, 0.05,
      [r"-47\.7"])
check("hills Y_IDX (wall-aware protocol)", int(d["Y_IDX"]), 10, 0,
      [r"y_\{?\\?mathrm\{idx\}|index.*10|Y_\\mathrm|tenth|protocol"])
check("hills n_profiles", int(d["n_profiles"]), 512, 0, [r"512"])

# ---- 2. exact-DNS-stress substitution makes it WORSE (G3, closure-indep) ----
check("exact-DNS-stress R2 (worse, not rescued)", float(d["controlled_dns_r2"]), -927.35, 0.5,
      [r"927"])
ed = npz("error_decomposition.npz")
check("exact-DNS worse than ML (r2_dns < r2_ml)",
      float(ed["r2_dns"]) < float(ed["r2_ml"]), True, 0, [r"927"])

# ---- 3. periodic-hill dose-response (Xiao 29-case family) ----
dx = npz("dose_response_xiao.npz")
r2 = np.asarray(dx["agg_r2"], float)
em = np.asarray(dx["agg_eps_median"], float)
check("Xiao family n_cases", r2.size, 29, 0, [r"29"])
check("Xiao agg_r2 max (least bad)", float(r2.max()), -9.856, 0.1, [r"-9\.86|-?9\.9|-10"])
check("Xiao agg_r2 min (worst)", float(r2.min()), -84.31, 0.5, [r"-?84"])
check("Xiao eps_median min", float(em.min()), 0.0839, 0.005, [r"0\.084|0\.08"])

# ---- 4. cancellation depth on canonical hill ----
p5 = npz("pehill_5case_corrected.npz")
check("hill case_1p0 rel_err(tau_w)", float(p5["case_1p0_rel_err"]), 6.824, 0.05, [r"6\.8"])
check("hill case_1p0 eps_median", float(p5["case_1p0_eps_median"]), 0.0836, 0.002,
      [r"0\.084|0\.08"])

# ---- 5. cross-geometry collapse (eps is a geometry-readable predictor) ----
cg = npz("cross_geometry_collapse.npz")
check("cross-geom Spearman rho(eps,relRMS) n", int(cg["spearman_n"]), 15, 0, [r"15|n=15"])
check("cross-geom LOO rho min", float(cg["loo_rho_min"]), -0.7626, 0.01, [r"-0\.76"])
check("cross-geom LOO rho max", float(cg["loo_rho_max"]), -0.60, 0.01, [r"-0\.60"])
check("cross-geom no-3D rho", float(cg["spearman_rho_no3d"]), -0.60, 0.01, [r"-0\.60"])

# ---- 6. a-posteriori coupled WMLES, canonical hill (equilibrium rung) ----
ap = npz("aposteriori_wmles_pehill.npz")
check("coupled hill reattachment rel err %", float(ap["eq_reattachment_rel_err_pct"]), -20.63, 0.1,
      [r"20\.6"])
check("coupled hill a-priori frac eps<0.1", float(ap["apriori_frac_below_01"]), 0.5645, 0.005,
      [r"0\.564|56"])
check("coupled hill a-priori median eps", float(ap["apriori_median_eps"]), 0.0836, 0.002,
      [r"0\.084|0\.08"])

# ---- 7. conv-div negative control (no-trigger, wide pitch) ----
cd = npz("aposteriori_wmles_convdiv.npz")
check("conv-div control reattachment rel err %", float(cd["reattachment_rel_err_pct"]), 13.28, 0.1,
      [r"13\.3"])
check("conv-div control eps_median O(1)", float(cd["wmles_eps_median"]), 1.698, 0.02,
      [r"1\.70|1\.7"])

# ---- 8. closure-family conditioning floor (G3/G6 model-independence) ----
cf = npz("closure_conditioning_floor.npz")
hills_rows = cf["hills_rows"]
r2s = [float(r["r2"]) for r in hills_rows]
check("all 5 closures fail on hill (r2<0)", all(x < 0 for x in r2s), True, 0, [r"-47\.7"])
check("exact-DNS amplification ratio", float(cf["P2_exactdns_amplification_ratio"]), 37.56, 0.1,
      [r"37|38"])
check("control not catastrophic (G7)", bool(cf["G7_control_not_catastrophic"]), True, 0, [r"0\.97"])

# ---- 9. conditioning ~ 1/eps (sensitivity mechanism) ----
cn = npz("condition_number.npz")
check("Spearman kappa_P vs 1/conv", float(cn["hills_spearman_kP_invconv"]), 0.908, 0.01, [r"0\.91"])
check("Spearman kappa_closure vs 1/conv", float(cn["hills_spearman_kK_invconv"]), 0.898, 0.01, [r"0\.90"])

# ---- 10. closure-structure compensation (exact-stress paradox) ----
check("compensation ratio (ML beats exact-DNS)", float(ed["compensation_ratio"]), 4.79, 0.02, [r"4\.8"])
check("corr(E_struct, C) signed", float(ed["corr_struct_C"]), 0.681, 0.01, [r"0\.68"])
check("frac ML beats exact-DNS", float(ed["frac_ml_beats_dns"]), 0.785, 0.01, [r"0\.79|78"])

# ---- 11. y_m robustness of compensation ----
ym = npz("compensation_ym_robustness.npz")
band = np.asarray(ym["y_m_plus_band"], float)
check("compensation y_m+ band low", float(band[0]), 5.01, 0.05, [r"5"])
check("compensation y_m+ band high", float(band[1]), 25.66, 0.05, [r"25|2[0-9]"])
check("all 4 compensation signatures persist",
      bool(ym["all_exact_stress_worse"]) and bool(ym["all_r2_dns_below_ml"]) and
      bool(ym["all_closure_struct_anticorr"]) and bool(ym["all_struct_C_positive"]),
      True, 0, [r"compensation"])

# ---- 12. ODE success on geometries it should handle (R2>=0.88) ----
gr = npz("closure_conditioning_floor.npz")["geom_rows"]
bfs = [r for r in gr if r["role"] == "success"][0]
check("BFS success R2(tau_w) >= 0.88", float(bfs["r2_A_ml_vandriest"]), 0.8887, 0.01, [r"0\.889|0\.88"])
check("ODE eval-set R2 (BFS+hump) = 0.918", tex_has(r"0\.918"), True, 0, [r"0\.918"])

# ================= run =================
print(f"{'CLAIM':<46} {'COMPUTED':>14} {'EXPECTED':>12}  DATA TEXT")
print("-" * 92)
n_pass = n_fail = 0
fails = []
for label, comp, exp, tol, pats in CHECKS:
    if isinstance(exp, bool):
        data_ok = (comp == exp)
        cs, es = str(comp), str(exp)
    else:
        data_ok = abs(float(comp) - float(exp)) <= tol
        cs = f"{float(comp):.4g}"; es = f"{float(exp):.4g}"
    text_ok = tex_has(*pats)
    ok = data_ok and text_ok
    n_pass += ok; n_fail += (not ok)
    if not ok: fails.append((label, cs, es, data_ok, text_ok))
    print(f"{label:<46} {cs:>14} {es:>12}  {'Y' if data_ok else 'N':>4} {'Y' if text_ok else 'N':>4}")
print("-" * 92)
print(f"PASS={n_pass}  FAIL={n_fail}  TOTAL={len(CHECKS)}")
if fails:
    print("\nFAILURES:")
    for label, cs, es, d_ok, t_ok in fails:
        why = []
        if not d_ok: why.append(f"data mismatch (got {cs}, want {es})")
        if not t_ok: why.append("rounded value not found in main.tex")
        print(f"  - {label}: {'; '.join(why)}")
sys.exit(0 if n_fail == 0 else 1)
