#!/usr/bin/env python3
r"""
harvest_dose_response.py  --  L2 (Thrust #11) family aggregator.

Collect every harvested coupled-WMLES point of the repeating-hill steepness
family + controls into ONE dose-response result the L3 figure/table consume:

    codes/results/aposteriori_dose_response.npz

Points (each carried only if its run is actually harvested -- NO fabrication):
  - alpha=0.8, 1.0, 1.2 hills @ Re_H=5600 (Xiao DNS reference)
        <- aposteriori_dose_response_xiao_{0p8,1p0,1p2}.npz  (harvest_xiao_case.py)
  - alpha=1.0 hill @ Re_H=10595 (Krank DNS reference) -- cross-Re anchor
        <- aposteriori_wmles_pehill.npz (+ closure_ladder_aposteriori.npz for e_Cf)
  - converging-diverging channel -- the ECONOMY (eps>1, wide-pitch) NEGATIVE
    CONTROL: predicted NO catastrophic deployment error
        <- aposteriori_wmles_convdiv.npz (+ repeating_structure_contrast.npz for eps)

Tests the pre-registered predictions:
  P1  coupled E_dep rises monotonically as a-priori eps_med falls (hills)
  P2  conv-div control E_dep << hills
  P4  one g(eps_med)=A/eps_med+B fits family + control (>=4 points)
and reports Spearman rho(eps_med, E_dep).  All definitions are imported from
coupled_dose_response.py (single source of truth).

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/harvest_dose_response.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, HERE)
from coupled_dose_response import (deployment_error, fit_predictor,
                                   evaluate_falsifiers)


def load_point(npz, **meta):
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    if "wmles_present" in d.files and not bool(d["wmles_present"]):
        return None
    return d


def collect():
    pts = []

    # --- Xiao steepness family (same Re, same resolution, only alpha varies)
    for tag, alpha in [("0p8", 0.8), ("1p0", 1.0), ("1p2", 1.2)]:
        d = load_point(os.path.join(RES, f"aposteriori_dose_response_xiao_{tag}.npz"))
        if d is None:
            pts.append(dict(label=f"hill alpha={alpha} (Re5600)", alpha=alpha,
                            eps_med=np.nan, E_dep=np.nan, e_reatt=np.nan,
                            e_Cf=np.nan, e_prof=np.nan, is_control=False,
                            present=False, ref="Xiao2020"))
            continue
        pts.append(dict(
            label=f"hill alpha={alpha} (Re5600)", alpha=alpha,
            eps_med=float(d["apriori_eps_med"]), E_dep=float(d["E_dep"]),
            e_reatt=float(d["e_reatt"]), e_Cf=float(d["e_Cf"]),
            e_prof=float(d["e_prof"]), is_control=False, present=True,
            ref="Xiao2020", x_reatt=float(d["wmles_x_reatt"]),
            dns_x_reatt=float(d["dns_x_reatt"]),
            sample_time=float(d["wmles_sample_time"])))

    # --- cross-Re anchor: alpha=1.0 @ Re10595 (Krank DNS)
    fa = os.path.join(RES, "aposteriori_wmles_pehill.npz")
    fcl = os.path.join(RES, "closure_ladder_aposteriori.npz")
    if os.path.exists(fa):
        d = np.load(fa, allow_pickle=True)
        if bool(d["wmles_present"]):
            cl = np.load(fcl) if os.path.exists(fcl) else None
            e_reatt = abs(float(d["reattachment_rel_err_pct"])) / 100.0
            e_prof = float(d["profile_rms_mean"])
            e_Cf = float(cl["eq_relrms_cf"]) if cl is not None else np.nan
            E, _ = deployment_error(e_reatt, e_Cf, e_prof)
            pts.append(dict(
                label="hill alpha=1.0 (Re10595, cross-Re)", alpha=1.0,
                eps_med=float(d["apriori_median_eps"]), E_dep=E,
                e_reatt=e_reatt, e_Cf=e_Cf, e_prof=e_prof, is_control=False,
                present=True, ref="Krank2018", x_reatt=float(d["wmles_x_reatt"]),
                dns_x_reatt=float(d["dns_x_reatt"]),
                sample_time=float(d["eq_sample_time"])))

    # --- conv-div negative control (eps>1, wide pitch)
    fcd = os.path.join(RES, "aposteriori_wmles_convdiv.npz")
    fct = os.path.join(RES, "repeating_structure_contrast.npz")
    eps_cd = (float(np.load(fct)["convdiv_eps_median"])
              if os.path.exists(fct) else np.nan)
    cd_present = False
    cd = dict(label="conv-div control", alpha=np.nan, eps_med=eps_cd,
              E_dep=np.nan, e_reatt=np.nan, e_Cf=np.nan, e_prof=np.nan,
              is_control=True, present=False, ref="Laval2011")
    if os.path.exists(fcd):
        dd = np.load(fcd, allow_pickle=True)
        cd_present = bool(dd["convdiv_present"]) if "convdiv_present" in dd.files \
            else bool(dd.get("wmles_present", False))
        if cd_present:
            # accept either a direct E_dep or component fields
            def g(k):
                return float(dd[k]) if k in dd.files else np.nan
            e_reatt = (abs(g("reattachment_rel_err_pct")) / 100.0
                       if "reattachment_rel_err_pct" in dd.files else g("e_reatt"))
            e_prof = g("profile_rms_mean") if "profile_rms_mean" in dd.files \
                else g("e_prof")
            e_Cf = g("e_Cf")
            E, _ = deployment_error(e_reatt, e_Cf, e_prof)
            cd.update(E_dep=E, e_reatt=e_reatt, e_Cf=e_Cf, e_prof=e_prof,
                      present=True)
            # prefer the coupled run's OWN eps_median (self-consistent with the
            # harvested deployment error) over the a-priori contrast value
            if "eps_median" in dd.files:
                cd["eps_med"] = g("eps_median")
            elif "wmles_eps_median" in dd.files:
                cd["eps_med"] = g("wmles_eps_median")
    pts.append(cd)
    return pts


def main():
    pts = collect()
    real = [p for p in pts if p["present"]]
    print("=== coupled a-posteriori dose-response: harvested points ===")
    for p in pts:
        s = (f"  {p['label']:38s} eps_med="
             f"{p['eps_med']:6.3f}  E_dep="
             f"{p['E_dep']:6.3f}  (e_reatt={p['e_reatt']:.3f}, "
             f"e_Cf={p['e_Cf']!s:>6}, e_prof={p['e_prof']!s:>6})"
             if p["present"] else
             f"  {p['label']:38s} PENDING (run not harvested)")
        print(s)

    # ordering / fit on real points
    fit = None
    rho = np.nan
    if len(real) >= 3:
        epsv = np.array([p["eps_med"] for p in real])
        Ev = np.array([p["E_dep"] for p in real])
        fit = fit_predictor(epsv, Ev)
        if fit:
            rho = fit["spearman_rho"]

    falsifiers = evaluate_falsifiers([
        dict(label=p["label"], eps_med=p["eps_med"], E_dep=p["E_dep"],
             e_reatt=p["e_reatt"], is_control=p["is_control"], present=p["present"],
             eq_tble_gap_pp=(2.30 if (p["present"] and not p["is_control"]
                                      and "Re10595" in p["label"]) else None))
        for p in pts])

    out = dict(
        labels=np.array([p["label"] for p in pts]),
        alpha=np.array([p["alpha"] for p in pts], float),
        eps_med=np.array([p["eps_med"] for p in pts], float),
        E_dep=np.array([p["E_dep"] for p in pts], float),
        e_reatt=np.array([p["e_reatt"] for p in pts], float),
        e_Cf=np.array([p["e_Cf"] for p in pts], float),
        e_prof=np.array([p["e_prof"] for p in pts], float),
        is_control=np.array([p["is_control"] for p in pts]),
        present=np.array([p["present"] for p in pts]),
        ref=np.array([p["ref"] for p in pts]),
        n_real=len(real), spearman_rho=rho,
        falsifiers_json=str(falsifiers),
    )
    if fit:
        out.update(fit_A=fit["A"], fit_B=fit["B"], fit_R2=fit["R2"],
                   fit_n=fit["n"])
    npz = os.path.join(RES, "aposteriori_dose_response.npz")
    np.savez(npz, **out)
    if fit:
        fit_msg = ("g(eps)=%.3f/eps+%.3f  R2=%.3f  rho=%.3f  n=%d"
                   % (fit["A"], fit["B"], fit["R2"], fit["spearman_rho"],
                      fit["n"]))
    else:
        fit_msg = "PENDING (<3 real points)"
    print(f"\n[fit] {fit_msg}")
    print(f"[falsifiers] {falsifiers}")
    print(f"[OUT] wrote {npz}  ({len(real)} real / {len(pts)} slots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
