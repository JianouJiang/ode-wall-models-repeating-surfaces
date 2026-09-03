#!/usr/bin/env python3
r"""rib_class_prediction.py  --  L3 (Results & analysis), Thrust #23.

GEOMETRY-CLASS GENERALISATION: turn the established 15-geometry cancellation
rule into a single, pre-registered, *falsifiable* prediction for the two
rib-roughened-channel regimes whose wall-resolved LES is in-flight (queued
behind the #22 coupled twin; see work_progress/).

This is an a-priori analysis on data already on disk -- it asserts NO rib
number.  It does three things, all reproducible:

  (1) ESTABLISH the discriminant on the 15-geometry cloud
      (codes/results/cross_geometry_collapse.npz).  Failure is defined
      structurally as R2(tau_w) < R2_FAIL; the best geometry-readable
      separator is the deep-cancellation coverage  frac[eps<0.1].  We compute
      the empirical separation margin and the perfectly-separating threshold.

  (2) STATE the 2-D repeating-subclass rule (the disclosed 3-D diffuser is a
      different, corner-separation mechanism -- main.tex l.1133 -- and is
      excluded): within the 2-D repeating class, a pitch ~ O(delta) that makes
      the recirculation SPAN the cavity (no attached re-development) forces
      deep domain-wide cancellation -> structural ODE failure; a wide pitch
      that lets the flow reattach between features keeps cancellation
      localised -> tolerated.  This is the classical d-type / k-type roughness
      transition (Perry 1969; Leonardi 2003, transition near p/k ~ 7).

  (3) APPLY the rule to the two on-disk rib designs (PROVENANCE.json) and
      PRE-REGISTER the falsifiable LES test (F1/F2/F4) with thresholds derived
      from the established cloud -- BEFORE the rib LES produces any number.

Honesty guard (gates G1/G2): asserts no ribbed-channel result file exists, so
no rib eps/R2 is claimed here; only the geometry (pitch/delta, p/k,
cavity-spanning) -- which is fixed by the mesh -- and the published-DNS
transition are used.

Reads  : codes/results/cross_geometry_collapse.npz
         codes/openfoam/rib_channel_{dtype,ktype}/PROVENANCE.json
Writes : codes/results/rib_class_prediction.npz
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RES = os.path.join(CODES, "results")
OF = os.path.join(CODES, "openfoam")

R2_FAIL = -1.0      # structural failure: catastrophic, sign-flipped, far below skill
R2_OK = 0.0         # tolerated: at least sign-correct / usable

# ----------------------------------------------------------------------------
# (1) establish the discriminant on the 15-geometry cloud
# ----------------------------------------------------------------------------
d = np.load(os.path.join(RES, "cross_geometry_collapse.npz"), allow_pickle=True)
keys = [str(k) for k in d["keys"]]
klass = [str(k) for k in d["klass"]]
repeating = d["repeating"].astype(bool)
pitchO = d["pitch_O_delta"].astype(bool)
eps_med = d["eps_med"].astype(float)
frac1 = d["frac_eps_lt1"].astype(float)
frac01 = d["frac_eps_lt0p1"].astype(float)
r2 = d["r2"].astype(float)
n = len(keys)

is_struct = r2 < R2_FAIL
# 2-D class = everything except the disclosed 3-D diffuser
is_3d = np.array([k == "kth_3d_diffuser" for k in keys])

print("=" * 78)
print("(1) DISCRIMINANT ON THE ESTABLISHED 15-GEOMETRY CLOUD")
print("    (cross_geometry_collapse.npz; protocol Y_IDX=10, a-priori)")
print("=" * 78)
print(f"{'geometry':24s} {'class':15s} {'pitchO':6s} {'frac<.1':>7s} "
      f"{'eps_med':>8s} {'R2':>9s} {'struct?':>7s}")
for i in range(n):
    print(f"{keys[i]:24s} {klass[i]:15s} {str(bool(pitchO[i])):6s} "
          f"{frac01[i]:7.2f} {eps_med[i]:8.3g} {r2[i]:9.3g} "
          f"{str(bool(is_struct[i])):>7s}")

# empirical separation on frac[eps<0.1]
struct_f01 = frac01[is_struct]
ok_f01 = frac01[~is_struct]
gap_lo = float(ok_f01.max())          # highest non-structural coverage
gap_hi = float(struct_f01.min())      # lowest structural coverage
tau_star = 0.5 * (gap_lo + gap_hi)    # perfectly-separating threshold (midpoint)
sep_clean = gap_hi > gap_lo           # gap exists -> linearly separable on this axis
# confusion at tau_star
pred_struct = frac01 >= tau_star
n_err = int(np.sum(pred_struct != is_struct))

print("\n--- separation on deep-cancellation coverage frac[eps<0.1] ---")
print(f"  structural geometries (R2<{R2_FAIL}): "
      f"{[keys[i] for i in range(n) if is_struct[i]]}")
print(f"    their frac[eps<0.1]: {[round(float(x),3) for x in struct_f01]}  "
      f"(min = {gap_hi:.3f})")
print(f"  non-structural max frac[eps<0.1] = {gap_lo:.3f}")
print(f"  empirical gap = ({gap_lo:.3f}, {gap_hi:.3f}); "
      f"separating threshold tau* = {tau_star:.3f}")
print(f"  classification errors at tau*: {n_err}/{n}  "
      f"(clean separation: {sep_clean})")
print(f"  Spearman rho(frac[eps<0.1], R2) = {float(d['spearman_rho_frac0p1']):+.3f} "
      f"(p = {float(d['spearman_p_frac0p1']):.1e}, n={n})")
print(f"  Spearman rho(eps_med, relRMS)   = {float(d['spearman_rho']):+.3f} "
      f"(p = {float(d['spearman_p']):.1e})")
print(f"  leave-one-out rho in [{float(d['loo_rho_min']):+.3f}, "
      f"{float(d['loo_rho_max']):+.3f}]")

# robustness: same separation restricted to the 2-D class (drop 3-D diffuser)
m2d = ~is_3d
struct2d = is_struct & m2d
gap_lo_2d = float(frac01[m2d & ~is_struct].max())
gap_hi_2d = float(frac01[struct2d].min())
print(f"\n  within the 2-D class (drop 3-D diffuser, n={int(m2d.sum())}):")
print(f"    structural 2-D failures: {[keys[i] for i in range(n) if struct2d[i]]}")
print(f"    gap = ({gap_lo_2d:.3f}, {gap_hi_2d:.3f}); "
      f"still separable: {gap_hi_2d > gap_lo_2d}")

# ----------------------------------------------------------------------------
# (2)+(3) load the two rib designs and apply the rule
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(2)/(3) APPLY THE 2-D RULE TO THE TWO ON-DISK RIB DESIGNS")
print("=" * 78)

def load_prov(tag):
    with open(os.path.join(OF, f"rib_channel_{tag}", "PROVENANCE.json")) as f:
        return json.load(f)

ribs = {t: load_prov(t) for t in ("dtype", "ktype")}
DK_TRANSITION = 7.0   # Perry 1969 / Leonardi 2003: d-type p/k<~7, k-type p/k>~7

rib_rows = []
for tag, pv in ribs.items():
    k = pv["k"]; pitch = pv["pitch"]; delta = pv["delta"]
    pk = pitch / k
    x_reatt_k = pv["x_reatt_est_k"]          # ~4.5k, published rib-DNS
    cavity_w_k = pv["w"] / k                  # cavity width in rib heights
    spans = bool(pv["cavity_spans"])          # recirc spans cavity? (w < x_reatt)
    # physical class call from GEOMETRY ONLY (no eps measured):
    #   spans cavity  -> no attached re-development -> deep domain-wide cancel
    #                    -> predict STRUCTURAL FAIL (like the O(delta)-pitch hill)
    #   reattaches    -> attached stretch between ribs -> localised cancel
    #                    -> predict TOLERATED (like the wide-pitch conv-div)
    pred = "STRUCTURAL_FAIL" if spans else "TOLERATED"
    dk = "d-type" if pk < DK_TRANSITION else "k-type"
    rib_rows.append((tag, pitch / delta, pk, cavity_w_k, x_reatt_k, spans, dk, pred))
    print(f"\n  {tag}:  pitch/delta = {pitch/delta:.2f},  p/k = {pk:.1f}  "
          f"({dk}; transition p/k~{DK_TRANSITION:.0f})")
    print(f"    cavity width = {cavity_w_k:.1f}k  vs  reattachment ~ {x_reatt_k:.1f}k "
          f"(Leonardi 2003)  ->  recirculation spans cavity: {spans}")
    print(f"    => geometry-rule prediction: {pred}")

# ----------------------------------------------------------------------------
# pre-registered, falsifiable LES test (thresholds from the established cloud)
# ----------------------------------------------------------------------------
F1_FRAC01 = round(tau_star, 2)   # d-type must exceed the structural coverage threshold
F2_FRAC01 = round(gap_lo, 2)     # k-type must stay below the non-structural ceiling
print("\n" + "=" * 78)
print("PRE-REGISTERED FALSIFIABLE LES TEST  (cloud-derived thresholds)")
print("=" * 78)
print(f"  F1 (d-type, predict FAIL):  frac[eps<0.1] >= {F1_FRAC01:.2f}  AND  "
      f"R2(tau_w) < {R2_FAIL:.0f}")
print(f"  F2 (k-type, predict OK)  :  frac[eps<0.1] <  {F2_FRAC01:.2f}  AND  "
      f"R2(tau_w) > {R2_OK:.0f}")
print(f"  F3 (in-family control)   :  the ONLY independent variable is cavity "
      f"width (pitch); same k/delta=0.2, Re_delta=4200, closure, ODE solver.")
print(f"  F4 (fidelity gate)       :  rib reattachment length and d-/k-type "
      f"wall-pressure signature within published rib-DNS spread")
print(f"                              (Leonardi 2003; Ikeda & Durbin 2007) "
      f"BEFORE any eps/R2 is reported.")

# ----------------------------------------------------------------------------
# honesty guard: no rib result on disk -> no rib number claimed
# ----------------------------------------------------------------------------
rib_npz = [f for f in os.listdir(RES) if "rib" in f.lower() and f.endswith(".npz")
           and f != "rib_class_prediction.npz"]
rib_result_present = len(rib_npz) > 0
print("\n" + "=" * 78)
print("HONESTY GUARD (G1/G2)")
print("=" * 78)
print(f"  ribbed-channel RESULT npz on disk: {rib_npz if rib_npz else '(none)'}")
assert not rib_result_present, "rib result file present -- update analysis to harvest it"
print("  ASSERT PASS: no rib result file -> this analysis asserts NO rib eps/R2;")
print("  the rib markers are PREDICTIONS, the in-flight LES is the falsifiable test.")

# self-consistency assertions on the established cloud
assert sep_clean and n_err == 0, "frac[eps<0.1] must perfectly separate the cloud"
assert ribs["dtype"]["cavity_spans"] and not ribs["ktype"]["cavity_spans"], \
    "rib designs must straddle the spanning/reattaching boundary"
assert (ribs["dtype"]["pitch"] / ribs["dtype"]["k"]) < DK_TRANSITION < \
    (ribs["ktype"]["pitch"] / ribs["ktype"]["k"]), \
    "rib designs must straddle the d/k transition p/k~7"
print("  ASSERT PASS: cloud separates cleanly; rib designs straddle the "
      "spanning/reattaching (d/k) boundary.")

# ----------------------------------------------------------------------------
# write the reproducible artifact
# ----------------------------------------------------------------------------
np.savez(
    os.path.join(RES, "rib_class_prediction.npz"),
    keys=np.array(keys), klass=np.array(klass),
    repeating=repeating, pitch_O_delta=pitchO,
    eps_med=eps_med, frac_eps_lt0p1=frac01, frac_eps_lt1=frac1, r2=r2,
    is_structural=is_struct,
    R2_FAIL=R2_FAIL, R2_OK=R2_OK,
    sep_gap_lo=gap_lo, sep_gap_hi=gap_hi, tau_star=tau_star,
    sep_errors=n_err, sep_clean=sep_clean,
    spearman_rho_frac0p1=float(d["spearman_rho_frac0p1"]),
    spearman_p_frac0p1=float(d["spearman_p_frac0p1"]),
    rib_tags=np.array([r[0] for r in rib_rows]),
    rib_pitch_over_delta=np.array([r[1] for r in rib_rows]),
    rib_p_over_k=np.array([r[2] for r in rib_rows]),
    rib_cavity_w_k=np.array([r[3] for r in rib_rows]),
    rib_x_reatt_k=np.array([r[4] for r in rib_rows]),
    rib_cavity_spans=np.array([r[5] for r in rib_rows]),
    rib_dk_type=np.array([r[6] for r in rib_rows]),
    rib_prediction=np.array([r[7] for r in rib_rows]),
    dk_transition_pk=DK_TRANSITION,
    F1_frac01_threshold=F1_FRAC01, F2_frac01_threshold=F2_FRAC01,
    rib_result_present=rib_result_present,
    note=np.array(
        "L3 Thrust #23 geometry-class prediction. frac[eps<0.1]>=tau* (tau* from "
        "the 15-geom gap midpoint) perfectly separates structural failure on the "
        "established cloud. The two rib designs straddle the d/k (spanning vs "
        "reattaching) transition by GEOMETRY ONLY; F1/F2 pre-register the "
        "falsifiable LES test. No rib eps/R2 asserted (rib_result_present=False)."),
)
print(f"\nWrote {os.path.join(RES, 'rib_class_prediction.npz')}")
print("DONE -- a-priori prediction registered; LES in-flight is the test.")
