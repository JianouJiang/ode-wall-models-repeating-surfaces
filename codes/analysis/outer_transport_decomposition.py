#!/usr/bin/env python3
r"""
outer_transport_decomposition.py
================================
Thrust #20 (L1, Core methodology) -- the WITHIN-LAYER / OUTER-TRANSPORT
decomposition of the convective momentum flux that the near-wall ODE drops.

PURPOSE.  Thrust #18 established that restoring the convective source inside the
wall layer (the CR-WM) heals only PART of the a-priori catastrophe on the
canonical periodic hill (R^2(tau_w): -47.7 -> -31.3 const / -21.0 exact-within-
layer; crwm_apriori.npz).  Thrust #20's thesis is that the unrecovered part is
carried by the convective transport ABOVE the matching height -- the outer flux a
1-D wall model structurally cannot see but a COUPLED LES resolves.  This script
makes that claim QUANTITATIVE and FALSIFIABLE from real DNS, before any coupled
run.  It is a pure a-priori diagnostic on data already on disk; no CFD.

THE DECOMPOSITION (physically derived, not fitted).  Integrate the steady 2-D
x-momentum total-stress balance from the wall:
        dG/dy = dp/dx + conv(y),   G(y) = (nu+nu_t) dU/dy,   G(0) = tau_w,
        => G(y) = tau_w + (dp/dx) y + Pi(y),   Pi(y) = int_0^y conv dy'.
The cumulative convective flux Pi splits at the matching height y_m:
        Pi(infty) = Pi(y_m)            [within-layer: <= y_m, wall-model-accessible]
                  + (Pi(infty)-Pi(y_m))[outer: > y_m, LES-resolved only].
The 1-D wall model can reconstruct only Pi(y) for y <= y_m from the matching-cell
data; the outer part is invisible to it.

TWO DISTINCT MEASURES (kept separate on purpose -- see HONESTY).
  (A) CONVECTIVE-TRANSPORT CENTROID.  Sign-safe: cumulative |conv| flux
      A(y) = int_0^y |conv| dy'.  y_50 = height where A(y)/A(y_edge) = 0.5, i.e.
      where half the convective momentum transport has accumulated.  Report
      y_50 / y_m and the within-layer transport fraction f_wl = A(y_m)/A(y_edge).
      This is robust because |conv| does not cancel in sign through the
      recirculation (the signed Pi does -- conv reverses across the shear layer,
      so a signed within-layer fraction is fragile and reported only as context).
  (B) VARIANCE-EXPLAINED.  From error_decomposition.npz: corr(E_struct, C)=+0.68
      => r^2 ~= 0.46 of the structural wall-stress-error VARIANCE is tracked by
      the single matching-cell convective value C.  This is a SIGNATURE measure
      (how predictive the matching-cell value is of the error), NOT a flux
      fraction.  It is NOT equal to f_wl and we do not equate them.

WHY THIS REFINES (and corrects) the L0 heuristic -- HONEST.  The L0 thesis
loosely tied the r^2~=0.46 ceiling to "roughly half the missing momentum flux
lives above y_m".  The data say the FLUX-magnitude split is far more lopsided
than half: only ~3% of the convective transport is within the layer; ~97% is
outer (y_50 ~ 7.5 y_m).  So the two measures are DISTINCT and we report both
without conflating: (A) says the matching height captures only the foot of the
convective transport; (B) says the part it does capture is nonetheless a strong
predictor of the error.  Both independently support the same conclusion -- the
1-D within-layer model is information-limited and the curative transport is
predominantly outer -- but they are different quantities.

WHAT THIS DOES AND DOES NOT PROVE (the non-tautology guard).
  - It DOES establish, from DNS, that the convective momentum transport the ODE
    drops lives overwhelmingly ABOVE the matching height, so the a-priori
    within-layer CR-WM ceiling (-21.0/-31.3) is a within-layer-information
    property, not the closed coupled answer.
  - It does NOT claim the coupled CR-WM heals the failure.  Whether the coupled
    LES, which resolves that outer transport, lifts the ceiling is the decisive
    coupled twin (F1/F2), harvested at L2/L3.  This script makes that question
    well-posed and shows it is NOT pre-decided by the a-priori number -- the
    crisp reason the coupled run is decisive, not a foregone reparametrisation.

A-PRIORI ONLY.  Every number traces to momentum_budget_pehill.npz (real Krank
periodic-hill DNS, canonical alpha=1.0) and error_decomposition.npz.  No CFD, no
coupled number, <= 2 cores.

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 codes/analysis/outer_transport_decomposition.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # codes/
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D        # production extraction constants

Y_IDX = D.Y_IDX                               # canonical matching index (=10)


def station_profiles():
    """Yield (i, y_from_wall, conv(y)) for every separated DNS station with a
    usable wall-normal column, using the same wall-surface-aware extraction as
    the production diagnostic (wall_j offset, fluid mask)."""
    b = np.load(os.path.join(RESULTS, "momentum_budget_pehill.npz"),
                allow_pickle=True)
    conv, yu, wall_j, fluid = b["convection"], b["y_unique"], b["wall_j"], b["fluid"]
    sep = b["sep_mask"]
    for i in np.where(sep)[0]:
        j0 = int(wall_j[i])
        col = np.where(fluid[i])[0]
        col = col[col >= j0]
        if len(col) < Y_IDX + 2:
            continue
        y = yu[col] - yu[j0]                   # distance from the (lower) wall
        c = conv[i, col]                       # conv(y) = U dU/dx + V dU/dy
        yield i, y, c


def cum_trapz(yv, fv):
    """Running trapezoidal integral with a leading 0 (same length as yv)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (fv[1:] + fv[:-1]) * np.diff(yv))])


def decompose(y_edge, yi=Y_IDX):
    """Per-station split of the convective transport at the matching height for a
    given boundary-layer-edge cutoff y_edge and matching index yi.

    Returns dict of arrays over the stations that resolve y_edge."""
    y50_over_ym, fwl_abs, fwl_signed, ym_arr = [], [], [], []
    for i, y, c in station_profiles():
        if yi >= len(y):
            continue
        ym = y[yi]
        sel = y <= y_edge
        ys, cs = y[sel], c[sel]
        if len(ys) < yi + 2 or ym > ys[-1]:
            continue
        # (A) sign-safe cumulative |conv| transport
        A = cum_trapz(ys, np.abs(cs))
        if A[-1] <= 0:
            continue
        frac = A / A[-1]
        y50 = float(np.interp(0.5, frac, ys))
        y50_over_ym.append(y50 / ym)
        fwl_abs.append(float(np.interp(ym, ys, A) / A[-1]))
        # (signed, fragile -- conv reverses sign across the shear layer)
        Pi = cum_trapz(ys, cs)
        if abs(Pi[-1]) > 1e-4:
            fwl_signed.append(float(np.interp(ym, ys, Pi) / Pi[-1]))
        ym_arr.append(float(ym))
    return {
        "y50_over_ym": np.array(y50_over_ym),
        "fwl_abs": np.array(fwl_abs),
        "fwl_signed": np.array(fwl_signed),
        "ym": np.array(ym_arr),
    }


def summ(a):
    return (float(np.median(a)), float(np.mean(a)),
            float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def main():
    print("=" * 72)
    print("Thrust #20 L1 -- within-layer / outer-transport decomposition")
    print("canonical periodic hill (Krank DNS, alpha=1.0), a-priori, no CFD")
    print("=" * 72)

    # ---- headline decomposition at the canonical cutoff y_edge = 1 (one hill
    # height ~ delta) and canonical matching index Y_IDX -----------------------
    d = decompose(y_edge=1.0, yi=Y_IDX)
    n = len(d["y50_over_ym"])
    med_y50, mean_y50, q1_y50, q3_y50 = summ(d["y50_over_ym"])
    med_fwl, mean_fwl, q1_fwl, q3_fwl = summ(d["fwl_abs"])
    print(f"\nseparated stations used: {n}   y_m (median) = {np.median(d['ym']):.4f} H")
    print("-" * 72)
    print("(A) CONVECTIVE-TRANSPORT CENTROID (sign-safe |conv|):")
    print(f"    y_50 / y_m         median={med_y50:5.2f}  mean={mean_y50:5.2f}  "
          f"IQR=[{q1_y50:.2f},{q3_y50:.2f}]")
    print(f"    within-layer frac  median={med_fwl:.3f}  mean={mean_fwl:.3f}  "
          f"IQR=[{q1_fwl:.3f},{q3_fwl:.3f}]")
    print(f"    => outer fraction (above y_m) median = {1.0 - med_fwl:.3f}")
    print(f"    signed within-layer frac (fragile, context): "
          f"median={np.median(d['fwl_signed']):+.3f}")

    # ---- (B) variance-explained, a DISTINCT measure (kept separate) ----------
    ed = np.load(os.path.join(RESULTS, "error_decomposition.npz"),
                 allow_pickle=True)
    corr_struct_C = float(ed["corr_struct_C"])
    var_explained = corr_struct_C ** 2
    print("-" * 72)
    print("(B) VARIANCE-EXPLAINED (signature, NOT a flux fraction):")
    print(f"    corr(E_struct, C) = {corr_struct_C:+.2f}  ->  r^2 = {var_explained:.2f}")
    print("    (the matching-cell convective value predicts ~46% of the")
    print("     structural-error VARIANCE; this is distinct from f_wl above)")

    # ---- robustness 1: boundary-layer-edge cutoff sensitivity ----------------
    print("-" * 72)
    print("robustness vs BL-edge cutoff y_edge (median y_50/y_m, within-layer frac):")
    edge_grid, edge_y50, edge_fwl = [], [], []
    for ye in [0.5, 0.75, 1.0, 1.5, 2.0]:
        de = decompose(y_edge=ye, yi=Y_IDX)
        if len(de["y50_over_ym"]) == 0:
            continue
        m_y50 = float(np.median(de["y50_over_ym"]))
        m_fwl = float(np.median(de["fwl_abs"]))
        edge_grid.append(ye); edge_y50.append(m_y50); edge_fwl.append(m_fwl)
        print(f"    y_edge={ye:4.2f} H   y_50/y_m={m_y50:5.2f}   "
              f"within-layer frac={m_fwl:.3f}   (n={len(de['y50_over_ym'])})")

    # ---- robustness 2: matching-index sweep (ties to thrust #17 critical y) --
    print("-" * 72)
    print("robustness vs matching index Y_IDX (centroid is far above y_m for all):")
    yi_grid, yi_ymp, yi_y50, yi_fwl = [], [], [], []
    NU = D.NU
    bnpz = np.load(os.path.join(RESULTS, "momentum_budget_pehill.npz"),
                   allow_pickle=True)
    tau_dns = bnpz["tau_w_dns"]; sepm = bnpz["sep_mask"]
    for yi in [10, 20, 40, 60, 80]:
        de = decompose(y_edge=1.0, yi=yi)
        if len(de["y50_over_ym"]) == 0:
            continue
        # representative y_m+ at this index (median over separated stations)
        ymp = float(np.median(de["ym"]) *
                    np.sqrt(np.median(np.abs(tau_dns[sepm]))) / NU)
        yi_grid.append(yi); yi_ymp.append(ymp)
        yi_y50.append(float(np.median(de["y50_over_ym"])))
        yi_fwl.append(float(np.median(de["fwl_abs"])))
        print(f"    Y_IDX={yi:3d}  y_m~{np.median(de['ym']):.3f}H (y_m+~{ymp:4.0f})  "
              f"y_50/y_m={np.median(de['y50_over_ym']):5.2f}  "
              f"within-layer frac={np.median(de['fwl_abs']):.3f}")

    # ---- bridge to the CR-WM a-priori ceiling --------------------------------
    cz = np.load(os.path.join(RESULTS, "crwm_apriori.npz"), allow_pickle=True)
    r2_ode = float(cz["r2_all_ode"])
    r2_const = float(cz["r2_all_const"])
    r2_exact = float(cz["r2_all_exact"])
    print("-" * 72)
    print("BRIDGE to the within-layer CR-WM ceiling (crwm_apriori.npz):")
    print(f"    ODE (no conv)           R^2 = {r2_ode:7.2f}")
    print(f"    CR-WM(const, c_m only)  R^2 = {r2_const:7.2f}   "
          f"(info-truncation rung)")
    print(f"    CR-WM(exact within-lay) R^2 = {r2_exact:7.2f}   "
          f"(within-layer ceiling)")
    print("    The exact-within-layer restoration recovers the WITHIN-LAYER flux")
    print(f"    (only ~{med_fwl*100:.0f}% of the convective transport) yet still")
    print("    fails (R^2<0): the residual is the conditioning-amplified closure")
    print("    error PLUS the outer transport (~97%) that NO 1-D model can carry.")
    print("    Whether the COUPLED LES (which resolves that outer transport) lifts")
    print("    the ceiling is the decisive coupled twin (F1/F2), not this number.")

    out = os.path.join(RESULTS, "outer_transport_decomposition.npz")
    np.savez(
        out,
        # headline (y_edge=1, Y_IDX=10)
        n_stations=n,
        ym_median=float(np.median(d["ym"])),
        y50_over_ym=d["y50_over_ym"],
        fwl_abs=d["fwl_abs"],
        fwl_signed=d["fwl_signed"],
        med_y50_over_ym=med_y50, mean_y50_over_ym=mean_y50,
        q1_y50_over_ym=q1_y50, q3_y50_over_ym=q3_y50,
        med_within_layer_frac=med_fwl, mean_within_layer_frac=mean_fwl,
        q1_within_layer_frac=q1_fwl, q3_within_layer_frac=q3_fwl,
        med_outer_frac=1.0 - med_fwl,
        # variance-explained (distinct measure)
        corr_struct_C=corr_struct_C, var_explained_by_C=var_explained,
        # cutoff sensitivity
        edge_grid=np.array(edge_grid), edge_y50=np.array(edge_y50),
        edge_fwl=np.array(edge_fwl),
        # matching-index sweep
        yi_grid=np.array(yi_grid), yi_ymp=np.array(yi_ymp),
        yi_y50=np.array(yi_y50), yi_fwl=np.array(yi_fwl),
        # CR-WM ceiling bridge
        r2_ode=r2_ode, r2_crwm_const=r2_const, r2_crwm_exact=r2_exact,
        note=np.array(["a-priori within-layer/outer decomposition; sign-safe "
                       "|conv| centroid y50/ym~7.5, within-layer frac~0.03; "
                       "r^2(var)=0.46 is a DISTINCT signature measure, not a "
                       "flux fraction; no coupled number"]),
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
