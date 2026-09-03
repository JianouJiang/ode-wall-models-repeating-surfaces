#!/usr/bin/env python3
r"""
coupled_dose_response.py  --  L1 (Thrust #11) methodology module.

Formalises the *a priori -> a posteriori predictor* claim: a single, locked
definition of (i) the coupled deployment-error metric  E_dep(G)  extracted from a
harvested coupled-WMLES result, and (ii) the geometry-readable predictor
g(eps_med) that the results section (L3) fits across the repeating-hill
steepness family + the converging-diverging negative control.

This module DEFINES and DEMONSTRATES the methodology on the anchors that already
exist on disk (the h/Lx=1.0 coupled hill, and the conv-div control once its run
is harvested).  It invents NO family numbers: the alpha=0.8 / alpha=1.2 coupled
points are produced by the L2 OpenFOAM runs (make_xiao_wmles_case.py) and only
then ingested here.  Every value written to the scaffold npz is either (a) read
directly from an existing results/*.npz, or (b) flagged PENDING.

Run:
    OMP_NUM_THREADS=2 python3 codes/analysis/coupled_dose_response.py
Outputs:
    codes/results/coupled_dose_response_scaffold.npz
    (prints a human-readable summary; no manuscript number is created here)

Definitions are frozen here so that L2 (harvest), L3 (fit + figure) and the
manuscript all cite ONE source of truth.
"""
import os
import json
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RES = os.path.abspath(RES)


# --------------------------------------------------------------------------- #
#  (1)  The coupled deployment-error metric  E_dep(G)                          #
# --------------------------------------------------------------------------- #
# A coupled WMLES of a repeating geometry G is scored against its wall-resolved
# reference by THREE dimensionless, sign-free deployment errors, each O(0.1-1):
#
#   e_reatt(G) = |x_reatt^WMLES - x_reatt^DNS| / x_reatt^DNS        (bulk topology)
#   e_Cf(G)    = relRMS( C_f^WMLES , C_f^DNS )                       (wall loading)
#              = sqrt(<(C_f^W - C_f^D)^2>) / sqrt(<C_f^D^2>)
#   e_prof(G)  = < ||U^WMLES(y) - U^DNS(y)|| >_stations / u_b         (mean field)
#
# These are the three faces of "did the deployed model get the flow right":
# the reattachment length is the separated-flow engineering number, C_f is the
# wall quantity the model is supposed to deliver, and the profile RMS is the
# bulk mean-field fidelity.  The scalar summary is their (unweighted) mean,
#
#   E_dep(G) = (e_reatt + e_Cf + e_prof) / 3 .
#
# Unweighted because the three are already commensurate dimensionless fractions
# and we refuse to tune weights to a desired trend (a referee-proof choice).
# All three components are reported individually so the composite never hides a
# disagreement among them.

def deployment_error(e_reatt, e_Cf, e_prof):
    """Composite coupled deployment error from its three locked components.
    Any missing component is carried as nan and excluded from the mean."""
    comps = np.array([e_reatt, e_Cf, e_prof], dtype=float)
    finite = comps[np.isfinite(comps)]
    summary = float(np.mean(finite)) if finite.size else np.nan
    return summary, comps


# --------------------------------------------------------------------------- #
#  (2)  The geometry-readable predictor  g(eps_med)                            #
# --------------------------------------------------------------------------- #
# The a-priori diagnostic is eps(x) = |tau_w| / (|dp/dx| y_m); eps_med is its
# domain median (the geometry-readable scalar a practitioner computes BEFORE any
# LES, from a cheap precursor / reference profile).  The locked predictor form is
#
#   g(eps_med) = A / eps_med + B ,                                   (Eq. predictor)
#
# i.e. the O(1/eps) divergence of the scaling argument (Sec. 4) plus a structural
# floor B.  This is NOT a free curve fit: the 1/eps branch is mechanism-derived
# (the deployed error inherits the residual-cancellation amplification), and B is
# the irreducible coupled floor.  L3 fits (A, B) across the >=4 coupled points
# {alpha=0.8, 1.0, 1.2 hills ; conv-div control} spanning eps_med in [0.08, 3.3]
# and reports the fit quality + Spearman rank correlation rho(eps_med, E_dep).

def predictor(eps_med, A, B):
    return A / np.asarray(eps_med, float) + B


def fit_predictor(eps_med, E_dep):
    """Fit g(eps)=A/eps+B by linear least squares in (1/eps).  Returns
    (A, B, R2, spearman_rho).  Requires >=3 finite points (an L3 operation;
    here it runs only if >=3 real anchors are already on disk)."""
    eps_med = np.asarray(eps_med, float)
    E_dep = np.asarray(E_dep, float)
    m = np.isfinite(eps_med) & np.isfinite(E_dep) & (eps_med > 0)
    if m.sum() < 3:
        return None
    x = 1.0 / eps_med[m]
    y = E_dep[m]
    Acoef, Bcoef = np.polyfit(x, y, 1)
    yhat = Acoef * x + Bcoef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Spearman rho on (eps, E): expect strong NEGATIVE (smaller eps -> larger E)
    rx = np.argsort(np.argsort(eps_med[m]))
    ry = np.argsort(np.argsort(E_dep[m]))
    rho = np.corrcoef(rx, ry)[0, 1]
    return dict(A=float(Acoef), B=float(Bcoef), R2=float(R2),
                spearman_rho=float(rho), n=int(m.sum()))


# --------------------------------------------------------------------------- #
#  (3)  The pre-registered falsifiers (P1-P4), encoded as checks on E_dep      #
# --------------------------------------------------------------------------- #
def evaluate_falsifiers(points):
    """points: list of dicts with keys {label, eps_med, E_dep, e_reatt,
    eq_tble_gap_pp, is_control}.  Returns a verdict dict.  Runs only on real,
    harvested points; PENDING entries are skipped and flagged."""
    real = [p for p in points if p.get("present", False)]
    out = {"n_real_points": len(real)}
    # P1: monotone dose-response across hill steepness family (controls excluded)
    hills = [p for p in real if not p.get("is_control", False)]
    hills = sorted(hills, key=lambda p: p["eps_med"])
    if len(hills) >= 2:
        Es = [p["E_dep"] for p in hills]
        # smaller eps must give larger E  =>  E decreasing in eps
        out["P1_monotone"] = bool(all(Es[i] >= Es[i + 1] - 1e-9
                                      for i in range(len(Es) - 1)))
    else:
        out["P1_monotone"] = "PENDING (need >=2 hill steepnesses)"
    # P2: conv-div control deployment error << hills
    ctrl = [p for p in real if p.get("is_control", False)]
    if ctrl and hills:
        out["P2_control_small"] = bool(
            max(p["E_dep"] for p in ctrl) < 0.5 * min(p["E_dep"] for p in hills))
    else:
        out["P2_control_small"] = "PENDING (need conv-div coupled harvest)"
    # P3: closure-independence -- eq vs TBLE coupled gap <= 3 pp at every steepness
    gaps = [p.get("eq_tble_gap_pp") for p in real if p.get("eq_tble_gap_pp") is not None]
    out["P3_closure_indep"] = (bool(max(abs(g) for g in gaps) <= 3.0) if gaps
                               else "PENDING")
    # P4: one monotone g(eps) fits family + control (>=4 points)
    if len(real) >= 4:
        fit = fit_predictor([p["eps_med"] for p in real],
                            [p["E_dep"] for p in real])
        out["P4_predictor_fit"] = fit
    else:
        out["P4_predictor_fit"] = "PENDING (need >=4 coupled points)"
    return out


# --------------------------------------------------------------------------- #
#  Demonstration on existing on-disk anchors (NO fabricated family numbers)    #
# --------------------------------------------------------------------------- #
def load_existing_anchors():
    points = []

    # --- anchor A: coupled periodic hill h/Lx=1.0 (alpha=1.0, real, harvested)
    f = os.path.join(RES, "aposteriori_wmles_pehill.npz")
    fcl = os.path.join(RES, "closure_ladder_aposteriori.npz")
    if os.path.exists(f):
        d = np.load(f)
        cl = np.load(fcl) if os.path.exists(fcl) else None
        present = bool(d["wmles_present"])
        e_reatt = abs(float(d["reattachment_rel_err_pct"])) / 100.0
        e_prof = float(d["profile_rms_mean"])  # already normalised by u_b
        e_Cf = float(cl["eq_relrms_cf"]) if cl is not None else np.nan
        summary, comps = deployment_error(e_reatt, e_Cf, e_prof)
        gap = (float(cl["tble_minus_eq_abs_relerr_pp"])
               if cl is not None else None)
        points.append(dict(
            label="pehill_alpha1.0 (Re_H=10595, coupled)",
            alpha=1.0, eps_med=float(d["apriori_median_eps"]),
            e_reatt=e_reatt, e_Cf=e_Cf, e_prof=e_prof, E_dep=summary,
            eq_tble_gap_pp=gap, is_control=False, present=present))

    # --- anchor B: conv-div negative control (real DNS; coupled run in progress)
    fcd = os.path.join(RES, "aposteriori_wmles_convdiv.npz")
    fct = os.path.join(RES, "repeating_structure_contrast.npz")
    eps_cd = (float(np.load(fct)["convdiv_eps_median"])
              if os.path.exists(fct) else np.nan)
    cd_present = False
    if os.path.exists(fcd):
        dd = np.load(fcd, allow_pickle=True)
        cd_present = bool(dd["convdiv_present"]) if "convdiv_present" in dd else False
    points.append(dict(
        label="convdiv control (Re=12600, coupled)",
        alpha=np.nan, eps_med=eps_cd,
        e_reatt=np.nan, e_Cf=np.nan, e_prof=np.nan, E_dep=np.nan,
        eq_tble_gap_pp=None, is_control=True, present=cd_present))

    # --- slots C, D: steepness family alpha=0.8, 1.2 (PENDING L2 OpenFOAM run)
    for a, tag in [(0.8, "0p8"), (1.2, "1p2")]:
        # a-priori eps_med for this steepness IS available from the Xiao DNS;
        # the COUPLED E_dep is PENDING the L2 run -> carried as nan/PENDING.
        eps_ap = np.nan
        ff = os.path.join(RES, f"aposteriori_dose_response_xiao_{tag}.npz")
        present = os.path.exists(ff)
        points.append(dict(
            label=f"pehill_alpha{a} (Re_H=5600, coupled)",
            alpha=a, eps_med=eps_ap,
            e_reatt=np.nan, e_Cf=np.nan, e_prof=np.nan, E_dep=np.nan,
            eq_tble_gap_pp=None, is_control=False, present=present))
    return points


def main():
    points = load_existing_anchors()
    print("=" * 72)
    print("L1 coupled dose-response scaffold -- DEFINITIONS demonstrated on")
    print("existing on-disk anchors (no family numbers invented).")
    print("=" * 72)
    for p in points:
        flag = "REAL" if p["present"] else "PENDING(L2)"
        E = p["E_dep"]
        Estr = f"{E:.4f}" if np.isfinite(E) else "  --  "
        print(f"[{flag:11s}] {p['label']:42s} "
              f"eps_med={p['eps_med'] if np.isfinite(p['eps_med']) else float('nan'):.4f}"
              f"  E_dep={Estr}")
        if p["present"] and np.isfinite(E):
            print(f"               components: e_reatt={p['e_reatt']:.4f} "
                  f"e_Cf={p['e_Cf']:.4f} e_prof={p['e_prof']:.4f}")
    verdict = evaluate_falsifiers(points)
    print("-" * 72)
    print("Pre-registered falsifier status (PENDING until L2 harvest):")
    print(json.dumps(verdict, indent=2, default=str))

    # save scaffold -- structure only + the one real composite that exists
    out = os.path.join(RES, "coupled_dose_response_scaffold.npz")
    np.savez(
        out,
        labels=np.array([p["label"] for p in points]),
        alpha=np.array([p["alpha"] for p in points], float),
        eps_med=np.array([p["eps_med"] for p in points], float),
        E_dep=np.array([p["E_dep"] for p in points], float),
        e_reatt=np.array([p["e_reatt"] for p in points], float),
        e_Cf=np.array([p["e_Cf"] for p in points], float),
        e_prof=np.array([p["e_prof"] for p in points], float),
        present=np.array([p["present"] for p in points], bool),
        is_control=np.array([p["is_control"] for p in points], bool),
        predictor_form=np.array("g(eps_med)=A/eps_med+B"),
        note=np.array("L1 scaffold; family (alpha=0.8,1.2) + conv-div coupled "
                      "points are PENDING the L2 OpenFOAM runs. NO fabricated "
                      "numbers. The single REAL composite is the alpha=1.0 hill."),
    )
    print("-" * 72)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
