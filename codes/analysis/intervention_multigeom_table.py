#!/usr/bin/env python3
r"""
intervention_multigeom_table.py  --  Thrust #29 L2 (node_002).

Assembles the MULTI-GEOMETRY operating-map / intervention table (falsifier F5,
killer gate G7) from on-disk results only.  This is the generalisation evidence
that the two controllable map axes of Thrust #29 -- placement (chi = y_m+/y_crit+)
and convective content (m) -- structure the a-posteriori failure across the
repeating-structure CLASS, and that the catastrophe band is bounded to the
O(delta)-pitch sub-class.

It also stages the TWO NEW single-geometry coupled interventions of this thrust:
  * Axis-2 (CR-WM convection cure) -> aposteriori_crwm_twin.npz (crwm_*)
  * Axis-1 (band placement)        -> aposteriori_band_confirmation.npz
both reported ABSENT / NaN until their runs land on disk -- never fabricated.

Every quantity is RE-DERIVED from its SOURCE npz (no hand-typed headline numbers):
  critical_matching_height_map.npz   per-geometry y_crit+, eps_c, bifurcation
  thrust27_collapse.npz              5 deployed coupled points (y_m+, y_crit+, e_reatt)
  aposteriori_dose_response.npz      deployed eps_med + e_reatt cross-geom Spearman
  epsilon_coupled_predictor.npz      within-geometry monotone law (rho, power b)
  dose_response_xiao.npz             within-family L_sep/delta collapse (rho + CI)
  intervention_calculus.npz          L1 F3 band, r_in/r_out, e_TBLE
  aposteriori_crwm_twin.npz          Axis-2 cure (UNLANDED -> crwm_e_reatt=NaN)
  aposteriori_band_confirmation.npz  Axis-1 band (UNLANDED/absent)

Writes codes/results/intervention_multigeom_table.npz.  Exit 0 on success.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")


def L(fn):
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def fmt(v, nd=3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "  --   "
        return f"{float(v):7.{nd}f}"
    except (TypeError, ValueError):
        return f"{v}"


def main():
    # ---------------------------------------------------------------- sources
    cmap = L("critical_matching_height_map.npz")     # per-geom bifurcation
    t27 = L("thrust27_collapse.npz")                  # 5 deployed coupled pts
    dose = L("aposteriori_dose_response.npz")         # deployed eps + e_reatt
    eps_pred = L("epsilon_coupled_predictor.npz")     # within-geom law
    xiao = L("dose_response_xiao.npz")                # L_sep/delta collapse
    icalc = L("intervention_calculus.npz")            # L1 F3 band
    twin = L("aposteriori_crwm_twin.npz")             # Axis-2 cure
    band = L("aposteriori_band_confirmation.npz")     # Axis-1 placement

    for nm, obj in [("critical_matching_height_map", cmap),
                    ("thrust27_collapse", t27),
                    ("aposteriori_dose_response", dose),
                    ("epsilon_coupled_predictor", eps_pred),
                    ("intervention_calculus", icalc)]:
        assert obj is not None, f"required source missing: {nm}.npz"

    # ============================================================= PART A
    # Per-geometry a-priori operating-map structure (bifurcation classification).
    # This is the O(delta)-pitch generalisation BOUND (F4/F5): which geometries
    # even HAVE a reachable catastrophe band.
    geo_keys = [str(k) for k in cmap["keys"]]
    klass = [str(k) for k in cmap["klass"]]
    repeating = np.asarray(cmap["repeating"], dtype=bool)
    pitch_Od = np.asarray(cmap["pitch_O_delta"], dtype=bool)
    ycrit_ap = np.asarray(cmap["ycrit"], dtype=float)
    eps_c = np.asarray(cmap["eps_c"], dtype=float)
    bifurcates = np.asarray(cmap["bifurcates"], dtype=bool)

    # RE-DERIVE the bifurcation classification independently from the sweep R2:
    # a geometry "bifurcates" iff its a-priori one-way R2(tau_w) suffers a
    # catastrophe ONSET (a >=0 -> <0 crossing) -- OR is catastrophic throughout --
    # within the DEPLOYABLE matching-height range (y_m+ <= deployed_ymp_max).  The
    # full sweep grid runs to y_m+~300, far beyond any real WMLES first cell, so a
    # crossing out there (e.g. conv-div at y_crit+~208) is NOT deployment-reachable
    # and must not count.  Cross-check vs the stored flag so the table is not a copy.
    ymp_grid = np.asarray(cmap["ymp_grid"], dtype=float)
    ymp_max = float(cmap["deployed_ymp_max"])   # 60
    dep = ymp_grid <= ymp_max
    bif_rederived = []
    for k in geo_keys:
        r2 = cmap.get(f"sweep_r2__{k}")
        if r2 is None:
            bif_rederived.append(False)
            continue
        r2 = np.asarray(r2, dtype=float)[dep]
        r2 = r2[np.isfinite(r2)]
        if r2.size < 2:
            bif_rederived.append(False)
            continue
        onset = np.any((r2[:-1] >= 0) & (r2[1:] < 0))   # catastrophe ignition
        allneg = bool(np.all(r2 < 0))                   # catastrophic throughout
        bif_rederived.append(bool(onset or allneg))
    bif_rederived = np.asarray(bif_rederived, dtype=bool)
    bif_match = bool(np.array_equal(bif_rederived, bifurcates))

    # The O(delta)-pitch bound: bifurcation occurs ONLY for repeating O(delta)-pitch
    bound_holds = bool(np.array_equal(bifurcates, pitch_Od))

    # ============================================================= PART B
    # Deployed coupled operating points (the 5 real coupled WMLES).  These show
    # the safe-corner / degeneracy that motivates WITHIN-geometry interventions.
    cp_labels = [str(s) for s in t27["labels"]]
    cp_e_reatt = np.asarray(t27["e_reatt"], dtype=float)
    cp_eps = np.asarray(t27["eps_med"], dtype=float)
    cp_ycrit = np.asarray(t27["y_crit_plus"], dtype=float)
    cp_ymp = np.asarray(t27["y_m_plus"], dtype=float)
    cp_chi = cp_ymp / cp_ycrit                       # RE-DERIVE chi, not copied
    assert np.allclose(cp_chi, np.asarray(t27["ratio"], dtype=float), rtol=1e-6), \
        "chi = y_m+/y_crit+ disagrees with stored ratio"

    rho_eps_cross = float(t27["rho_eps"])            # raw eps cross-geom Spearman
    rho_chi_cross = float(t27["rho_ratio"])          # chi cross-geom Spearman
    loo_chi = np.asarray(t27["loo_rho_ratio"], dtype=float)
    steep_monotone = bool(t27["steepness_monotone"])

    # ============================================================= PART C
    # The clean within-geometry law (Axis traversed on ONE geometry).
    rho_within = float(eps_pred["spearman_rho"])     # -1.0
    b_within = float(eps_pred["fit_b"])              # 0.441
    r2ll_within = float(eps_pred["fit_r2_loglog"])   # 0.977

    # within-family L_sep/delta collapse (G7 generalisation within the family)
    # canonical anchor: Spearman(L_sep/delta, R2(tau_w)) = -0.754 CI[-0.88,-0.54]
    # (bigger separation relative to delta -> more negative R2 = worse).
    lsep_rho = lsep_lo = lsep_hi = np.nan
    if xiao is not None:
        lsep_rho = float(xiao["collapse_Lsep_over_delta_rho_r2"]) \
            if "collapse_Lsep_over_delta_rho_r2" in xiao else np.nan
        lsep_lo = float(xiao["collapse_Lsep_over_delta_rho_r2_ci_lo"]) \
            if "collapse_Lsep_over_delta_rho_r2_ci_lo" in xiao else np.nan
        lsep_hi = float(xiao["collapse_Lsep_over_delta_rho_r2_ci_hi"]) \
            if "collapse_Lsep_over_delta_rho_r2_ci_hi" in xiao else np.nan

    # ============================================================= PART D
    # The L1 F3 band + the two NEW single-geometry interventions (this thrust).
    e_TBLE = float(icalc["e_TBLE"])
    r_in = float(icalc["r_in"]); r_out = float(icalc["r_out"])
    band_lo = float(icalc["band_lo"]); band_hi = float(icalc["band_hi"])
    e_cure_pred = float(icalc["e_cure_pred"])

    # Axis-2 cure (CR-WM) -- UNLANDED unless crwm_present True on disk
    crwm_present = bool(twin["crwm_present"]) if (twin is not None and "crwm_present" in twin) else False
    crwm_e_reatt = float(twin["crwm_e_reatt"]) if (twin is not None and "crwm_e_reatt" in twin) else np.nan
    if not crwm_present:
        crwm_e_reatt = np.nan                        # never fabricate

    # Axis-1 band placement -- UNLANDED unless the npz exists with a number
    band_present = False
    band_e_reatt = np.nan
    band_e_reatt_fine = np.nan
    if band is not None:
        band_present = bool(band["present"]) if "present" in band else \
            bool(band.get("band_present", False))
        for k in ("band_e_reatt", "e_reatt", "coarse_e_reatt"):
            if k in band and np.isfinite(float(band[k])):
                band_e_reatt = float(band[k]); break
        for k in ("fine_e_reatt", "e_reatt_fine"):
            if k in band and np.isfinite(float(band[k])):
                band_e_reatt_fine = float(band[k]); break
    if not band_present:
        band_e_reatt = np.nan

    # ------------------------------------------------------- F-verdicts (state)
    # F1: cure partial (0 < e_cure < e_TBLE)
    if crwm_present:
        f1 = "PASS" if (0.0 < crwm_e_reatt < e_TBLE) else "KILL"
    else:
        f1 = "UNLANDED"
    # F3: cure in map allocation band
    if crwm_present:
        f3 = "PASS" if (band_lo <= crwm_e_reatt <= band_hi) else \
             ("BONUS" if crwm_e_reatt < band_lo else "KILL")
    else:
        f3 = "UNLANDED"
    # F2: band placement degrades vs fine in-corner twin
    if band_present:
        f2 = "PASS" if (np.isfinite(band_e_reatt_fine) and band_e_reatt > band_e_reatt_fine) \
            else ("LANDED-NOFINE" if not np.isfinite(band_e_reatt_fine) else "KILL")
    else:
        f2 = "UNLANDED"
    # F5 / G7: O(delta)-pitch bound (bifurcation == pitch_O_delta) + within-family collapse
    f5 = "PASS" if bound_holds else "FAIL"
    # F6 / G3: closure-independence already on disk
    f6 = "PASS" if bool(icalc.get("closure_independent", False)) else "FAIL"

    # =============================================================== PRINT
    print("=" * 78)
    print("THRUST #29 L2 -- MULTI-GEOMETRY INTERVENTION TABLE (F5 / G7)")
    print("=" * 78)
    print("\nPART A.  A-priori operating-map structure across geometries")
    print("  (does the geometry HAVE a reachable catastrophe band? = O(delta) bound)\n")
    print(f"  {'geometry':<26}{'class':<18}{'pitch~d':>8}{'y_crit+':>9}"
          f"{'eps_c':>8}{'bifurc':>8}")
    for i, k in enumerate(geo_keys):
        print(f"  {k:<26}{klass[i]:<18}{str(bool(pitch_Od[i])):>8}"
              f"{fmt(ycrit_ap[i],2)}{fmt(eps_c[i],3)}{str(bool(bifurcates[i])):>8}")
    print(f"\n  bifurcation re-derived from sweep_r2 matches stored flag: {bif_match}")
    print(f"  O(delta)-pitch BOUND holds (bifurcates == pitch_O_delta): {bound_holds}")

    print("\nPART B.  Deployed coupled operating points (5 real coupled WMLES)\n")
    print(f"  {'case':<34}{'eps_med':>9}{'y_m+':>7}{'y_crit+':>9}"
          f"{'chi':>7}{'e_reatt':>9}")
    for i, lab in enumerate(cp_labels):
        print(f"  {lab:<34}{fmt(cp_eps[i],3)}{fmt(cp_ymp[i],2)}"
              f"{fmt(cp_ycrit[i],2)}{fmt(cp_chi[i],2)}{fmt(cp_e_reatt[i],3)}")
    print(f"\n  cross-geom Spearman(raw eps, e_reatt) = {rho_eps_cross:+.2f}  (uninformative)")
    print(f"  cross-geom Spearman(chi,     e_reatt) = {rho_chi_cross:+.2f}  (degenerate; "
          f"per-geom y_crit+ varies x{cp_ycrit.max()/cp_ycrit.min():.0f})")
    print(f"  LOO Spearman(chi): {loo_chi}  (drop alpha=0.8 -> {loo_chi[0]:+.1f})")
    print(f"  steepness-monotone in raw eps: {steep_monotone}")

    print("\nPART C.  Clean WITHIN-geometry law + within-family collapse\n")
    print(f"  within-geometry  Spearman(eps, relRMS) = {rho_within:+.2f}, "
          f"relRMS ~ eps^-{b_within:.3f}  (log-log R2={r2ll_within:.3f})")
    print(f"  within-family    Spearman(L_sep/delta, R2(tau_w)) = {fmt(lsep_rho,3)} "
          f"CI[{fmt(lsep_lo,2)},{fmt(lsep_hi,2)}]")

    print("\nPART D.  The two NEW single-geometry interventions (this thrust)\n")
    print(f"  e_TBLE (landed baseline)              = {e_TBLE:.3f}")
    print(f"  r_in={r_in:.3f}  r_out={r_out:.3f}  -> F3 band [{band_lo:.3f},{band_hi:.3f}] "
          f"(point {e_cure_pred:.3f})")
    print(f"  Axis-2 CR-WM cure : crwm_present={crwm_present}  e_cure={fmt(crwm_e_reatt,3)}")
    print(f"  Axis-1 band place : band_present={band_present}  e_reatt={fmt(band_e_reatt,3)} "
          f"(fine twin {fmt(band_e_reatt_fine,3)})")

    print("\nFALSIFIER STATE")
    for nm, v in [("F1 cure partial", f1), ("F2 band degrades", f2),
                  ("F3 map allocation", f3), ("F5 O(d) bound/G7", f5),
                  ("F6 closure-indep/G3", f6)]:
        print(f"  {nm:<22}: {v}")

    # =============================================================== SAVE
    out = dict(
        # PART A
        geo_keys=np.array(geo_keys), klass=np.array(klass),
        repeating=repeating, pitch_O_delta=pitch_Od,
        ycrit_apriori=ycrit_ap, eps_c=eps_c, bifurcates=bifurcates,
        bif_rederived=bif_rederived, bif_match=bif_match, bound_holds=bound_holds,
        # PART B
        cp_labels=np.array(cp_labels), cp_e_reatt=cp_e_reatt, cp_eps_med=cp_eps,
        cp_ycrit_plus=cp_ycrit, cp_ym_plus=cp_ymp, cp_chi=cp_chi,
        rho_eps_cross=rho_eps_cross, rho_chi_cross=rho_chi_cross,
        loo_chi=loo_chi, steepness_monotone=steep_monotone,
        # PART C
        rho_within=rho_within, b_within=b_within, r2ll_within=r2ll_within,
        lsep_rho=lsep_rho, lsep_lo=lsep_lo, lsep_hi=lsep_hi,
        # PART D
        e_TBLE=e_TBLE, r_in=r_in, r_out=r_out,
        band_lo=band_lo, band_hi=band_hi, e_cure_pred=e_cure_pred,
        crwm_present=crwm_present, crwm_e_reatt=crwm_e_reatt,
        band_present=band_present, band_e_reatt=band_e_reatt,
        band_e_reatt_fine=band_e_reatt_fine,
        # verdicts
        F1=f1, F2=f2, F3=f3, F5=f5, F6=f6,
        note=("Thrust #29 L2 multi-geometry intervention table. PART A = O(delta)-pitch "
              "bifurcation bound (only repeating O(delta)-pitch geoms have a reachable "
              "catastrophe band). PART B = 5 deployed coupled points (raw-eps & chi "
              "cross-geom sorts uninformative -> degeneracy motivates within-geom "
              "interventions). PART C = clean within-geometry monotone law + within-family "
              "L_sep/delta collapse. PART D = the two NEW single-geometry coupled "
              "interventions (CR-WM cure Axis-2; band placement Axis-1), UNLANDED until on "
              "disk -- never fabricated. Every number re-derived from SOURCE npz."),
    )
    op = os.path.join(RESULTS, "intervention_multigeom_table.npz")
    np.savez(op, **out)
    print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
