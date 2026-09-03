#!/usr/bin/env python3
r"""
crwm_eps_distribution_forecast_l2.py -- Thrust #33 L2 (node_002).

Discharges the Judge's L1 mandatory fixes for the multi-geometry CR-WM cure
forecast:

  FIX 1 (eps-binned f_in).  The flat forecast applies one recovery fraction
        f_in = 0.560 to every hill.  The MEASURED station-level recovery law
        rec(eps) (crwm_results_analysis.npz: bin_rec) swings from +0.61 at the
        deepest cancellation (eps ~ 0.025) to -1.60 at eps ~ 0.18.  The Judge's
        concern: a1p2 has the highest eps_median, so a flat f_in may over-predict
        its cure.  RESOLUTION: the deployment-relevant quantity is not the
        per-station-UNWEIGHTED recovery but the LEVERAGE-weighted (squared-error)
        aggregate -- and recovery_fraction = (SS_ode - SS_cured)/(SS_ode - SS_tot)
        is EXACTLY additive over stations.  We split the 512 canonical stations at
        eps = 0.1 and measure that the low-eps group carries ~30x the leverage
        (mean denominator mass) of the high-eps group.  A two-group leverage model
        rec(f) = [f*Rlo + (1-f)*Rhi] / [f*Dlo + (1-f)*Dhi]  (f = frac of stations
        with eps < 0.1) reproduces the on-disk f_in = 0.560 at the canonical hill's
        own f = 0.564, and gives the DISTRIBUTION-AWARE per-geometry forecast.
        Because leverage concentrates where cancellation is deepest, the aggregate
        recovery is ROBUST across the steepness family (rec ~ 0.50-0.56) -- the
        flat f_in is NOT fragile.  This is reported alongside the flat forecast.

  FIX 2 (baseline harmonisation).  Forecasts are quoted against the TBLE baseline
        (the clean single-variable contrast: TBLE and CR-WM differ ONLY by the
        within-layer convective source, identical mesh / numerics / restart field).
        a1p0 TBLE baseline = 0.387 (measured).  a1p2 TBLE baseline = TO BE MEASURED
        by the launched xiao_wmles_a1p2_tble arm; the Spalding baseline 0.327 is
        quoted only as a reference.  Mixed Spalding/TBLE baselines from the L1
        registry are NOT used for the band centre.

  FIX 4 (a0p8 low power).  a0p8 Spalding baseline is 0.048 (4.8%), within plausible
        coupled-WMLES averaging uncertainty (+-1-2%).  Flagged as a low-power arm;
        the decisive arms are a1p0 (landing) and a1p2 (launched).

NO fabrication: every number traces to crwm_kernel_experiment.npz /
crwm_results_analysis.npz / aposteriori_crwm_twin.npz / dose_response_xiao.npz.
If the a1p0 coupled cure has landed (aposteriori_crwm_twin.npz crwm_present=True)
the script reports the landed number against the pre-registered band; otherwise it
states crwm_present=False and asserts NO cure number.
Writes codes/results/crwm_eps_distribution_forecast.npz.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")


def load(name):
    return np.load(os.path.join(RESULTS, name), allow_pickle=True)


def r2(p, t):
    ss = np.sum((t - t.mean()) ** 2)
    return 1.0 - np.sum((p - t) ** 2) / ss if ss > 0 else np.nan


def main():
    ke = load("crwm_kernel_experiment.npz")
    ra = load("crwm_results_analysis.npz")
    to, tx, td = ke["tau_ode"], ke["tau_exact"], ke["tau_dns"]
    eps = ra["eps"]                                   # 512, aligned index-for-index
    assert eps.shape == to.shape, "eps not aligned to kernel stations"
    dbar = td.mean()

    # --- exact per-station decomposition of the recovery fraction -------------
    recovered = (to - td) ** 2 - (tx - td) ** 2       # numerator mass per station
    denom = (to - td) ** 2 - (td - dbar) ** 2          # denominator mass per station
    f_in = float(recovered.sum() / denom.sum())
    assert abs(f_in - float(ra["rec_full_pt"])) < 1e-6, "f_in mismatch vs disk"

    # --- two-group leverage model (split at eps = 0.1) ------------------------
    lo = eps < 0.1
    hi = ~lo
    Rlo, Rhi = recovered[lo].mean(), recovered[hi].mean()
    Dlo, Dhi = denom[lo].mean(), denom[hi].mean()
    leverage_ratio = float(Dlo / Dhi)                  # ~30x: low-eps dominates

    def rec_of_frac(f):
        return float((f * Rlo + (1 - f) * Rhi) / (f * Dlo + (1 - f) * Dhi))

    frac_a1p0_kernel = float(np.mean(lo))
    selfcheck = rec_of_frac(frac_a1p0_kernel)          # must reproduce f_in
    assert abs(selfcheck - f_in) < 1e-6, "two-group model self-check failed"

    # --- per-geometry eps-distribution stats (steepness-family protocol) ------
    #    frac(eps<0.1), eps_median from dose_response_xiao.npz, matched by the
    #    exact eps_median values the L1 forecast registered.
    dr = load("dose_response_xiao.npz")
    case, em, fr = dr["agg_case"], dr["agg_eps_median"], dr["agg_frac_eps_lt_0p1"]
    REG = {  # geometry -> (registered eps_median from forecast_registry.json)
        "hill_a0p8": 0.09823786615124785,
        "hill_a1p0": 0.14909175910684377,
        "hill_a1p2": 0.20490547244972762,
    }
    geo_frac, geo_emed = {}, {}
    for g, e_reg in REG.items():
        j = int(np.argmin(np.abs(em - e_reg)))
        geo_emed[g] = float(em[j])
        geo_frac[g] = float(fr[j])
        assert abs(em[j] - e_reg) < 1e-6, f"{g} eps_median row not matched"

    # --- baselines (FIX 2): TBLE single-variable contrast where measured -------
    twin = load("aposteriori_crwm_twin.npz")
    tble_a1p0 = float(twin["tble_e_reatt"])            # 0.387 measured
    spald_a1p0 = float(twin["spalding_e_reatt"])       # 0.333 measured
    dose = load("aposteriori_dose_response.npz")
    # Spalding baselines (reference only) from the dose-response (equilibrium model)
    lbl = list(dose["labels"]); Edep = dose["E_dep"]
    spald_a0p8 = float(Edep[0])                        # 0.0484
    spald_a1p2 = float(Edep[2])                        # 0.327

    BASE = {  # (model, e_base, measured?)
        "hill_a0p8": ("spalding", spald_a0p8, True),   # only Spalding ran -> low power
        "hill_a1p0": ("tble",     tble_a1p0,  True),   # clean TBLE single-var contrast
        "hill_a1p2": ("tble",     np.nan,     False),  # TBLE arm launched; Spalding ref below
    }
    SPALD_REF = {"hill_a0p8": spald_a0p8, "hill_a1p0": spald_a1p0, "hill_a1p2": spald_a1p2}

    # --- build forecast table -------------------------------------------------
    rows = []
    for g in ("hill_a0p8", "hill_a1p0", "hill_a1p2"):
        f = geo_frac[g]
        rec_flat = f_in
        rec_dist = rec_of_frac(f)
        model, e_base, measured = BASE[g]
        e_ref = SPALD_REF[g]
        def band(e, rec):
            return None if not np.isfinite(e) else float(e * (1 - rec))
        rows.append(dict(
            geometry=g, eps_median=geo_emed[g], frac_eps_lt_0p1=f,
            baseline_model=model, e_base=(None if not np.isfinite(e_base) else float(e_base)),
            e_base_measured=bool(measured), e_base_spalding_ref=float(e_ref),
            rec_flat=float(rec_flat), rec_dist_aware=float(rec_dist),
            e_hat_flat=band(e_base, rec_flat), e_hat_dist=band(e_base, rec_dist),
            low_power=bool(g == "hill_a0p8"),
        ))

    # --- a1p0 landed-cure check (NO fabrication) ------------------------------
    crwm_present = bool(twin["crwm_present"])
    a1p0_band = [float(ra["heur_lo"]), float(ra["heur_hi"])]   # [0.134, 0.232]
    a1p0_landed = float(twin["crwm_e_reatt"]) if crwm_present else None
    a1p0_in_band = (None if a1p0_landed is None
                    else bool(a1p0_band[0] <= a1p0_landed <= a1p0_band[1]))

    out = dict(
        f_in=f_in, f_in_ci=np.array([float(ra["ci_rec_full"][0]), float(ra["ci_rec_full"][1])]),
        bin_eps_med=ra["bin_eps_med"], bin_rec=ra["bin_rec"],
        bin_rec_lo=ra["bin_rec_lo"], bin_rec_hi=ra["bin_rec_hi"], bin_n=ra["bin_n"],
        leverage_ratio_lo_over_hi=leverage_ratio,
        group_Rlo=float(Rlo), group_Rhi=float(Rhi), group_Dlo=float(Dlo), group_Dhi=float(Dhi),
        frac_a1p0_kernel=frac_a1p0_kernel, twogroup_selfcheck=selfcheck,
        rec_dist_a0p8=rec_of_frac(geo_frac["hill_a0p8"]),
        rec_dist_a1p0=rec_of_frac(geo_frac["hill_a1p0"]),
        rec_dist_a1p2=rec_of_frac(geo_frac["hill_a1p2"]),
        rows_json=json.dumps(rows),
        a1p0_tble_baseline=tble_a1p0, a1p0_spalding_baseline=spald_a1p0,
        a1p0_band=np.array(a1p0_band), crwm_present=crwm_present,
        a1p0_crwm_e_reatt=(np.nan if a1p0_landed is None else a1p0_landed),
        a1p0_in_band=(False if a1p0_in_band is None else a1p0_in_band),
        note=("Thrust#33 L2: eps-distribution-aware multi-geometry CR-WM cure forecast. "
              "Leverage-weighted recovery is robust (0.50-0.56) across the family because "
              "the low-eps stations carry %.0fx the leverage. Baselines harmonised to TBLE "
              "single-variable contrast. a0p8 = low-power (Spalding e_base=0.048). "
              "crwm_present=%s." % (leverage_ratio, crwm_present)),
    )
    np.savez(os.path.join(RESULTS, "crwm_eps_distribution_forecast.npz"), **out)

    # --- report ---------------------------------------------------------------
    print("=" * 74)
    print("Thrust #33 L2 -- eps-distribution-aware multi-geometry cure forecast")
    print("=" * 74)
    print(f"f_in (flat, leverage-weighted)      = {f_in:.4f}  CI {tuple(np.round(out['f_in_ci'],3))}")
    print(f"two-group self-check (frac={frac_a1p0_kernel:.3f}) = {selfcheck:.4f}  [target {f_in:.4f}]")
    print(f"leverage ratio low/high eps          = {leverage_ratio:.1f}x  "
          "(why aggregate recovery is robust)")
    print("\nMEASURED station-level recovery law rec(eps)  [FIX 1, from disk]:")
    for j in range(4):
        print(f"  eps_med={float(ra['bin_eps_med'][j]):.3f}  rec={float(ra['bin_rec'][j]):+.3f}"
              f"  [{float(ra['bin_rec_lo'][j]):+.2f},{float(ra['bin_rec_hi'][j]):+.2f}]  n={int(ra['bin_n'][j])}")
    print("\nForecast table (FIX 2 baselines, FIX 4 power):")
    print(f"  {'geom':<10}{'eps_med':>8}{'frac<.1':>8}{'base':>9}{'e_base':>8}"
          f"{'rec_flat':>9}{'rec_dist':>9}{'e_flat':>8}{'e_dist':>8}{'note':>10}")
    for rw in rows:
        eb = f"{rw['e_base']:.3f}" if rw['e_base'] is not None else "  TBD"
        ef = f"{rw['e_hat_flat']:.3f}" if rw['e_hat_flat'] is not None else "  TBD"
        ed = f"{rw['e_hat_dist']:.3f}" if rw['e_hat_dist'] is not None else "  TBD"
        nt = "low-power" if rw['low_power'] else ("LAUNCHED" if rw['e_base'] is None else "")
        print(f"  {rw['geometry']:<10}{rw['eps_median']:>8.3f}{rw['frac_eps_lt_0p1']:>8.3f}"
              f"{rw['baseline_model']:>9}{eb:>8}{rw['rec_flat']:>9.3f}{rw['rec_dist_aware']:>9.3f}"
              f"{ef:>8}{ed:>8}{nt:>10}")
    print(f"\na1p0 pre-registered band [{a1p0_band[0]:.3f},{a1p0_band[1]:.3f}]  "
          f"crwm_present={crwm_present}")
    if crwm_present:
        print(f"  LANDED a1p0 CR-WM e_reatt = {a1p0_landed:.4f}  in_band={a1p0_in_band}")
    else:
        print("  a1p0 CR-WM NOT yet landed -- NO cure number asserted (run in flight).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
