#!/usr/bin/env python3
r"""
nonlocality_length.py
=====================
Thrust #13 (L1) -- the near-wall non-locality length.

*** RESULT: criterion is NON-DISCRIMINATING (F1 fails). ***
This script TESTED the L0 hypothesis that the ODE fails when the streamwise
domain of dependence ell_x = U_m y_m^2/(nu+nu_t) reaches the geometric scale L_x.
On the full real benchmark ell_x/L_x is a near-chance discriminator of ODE
failure: pooled station-level AUC(ell_x/L_x, eps<1) = 0.59 vs eps's trivial 1.00;
the near-wall Peclet Pe ANTI-discriminates (AUC 0.08 -- failure stations have the
LOWEST Pe, i.e. the ODE fails where the near-wall flow is MOST diffusion-
dominated). At geometry-median level ell_x/L_x ranks the failure geometry
(periodic hills, 0.090) 2nd of 9, but all values are <<1 (max 0.11, so the
predicted ell_x/L_x>~1 threshold is never reached) and it washes out station-
level. Turbulent nu_t keeps ell_x short and U_m->0 in recirculation shrinks Pe
further. ell_x is therefore NOT a usable failure criterion; the real driver stays
the force cancellation (eps<<1, hypersensitive residual). See
development/nodes/node_001/methodology.md. Kept as the reproducible record of the
negative result -- NOT a manuscript input.

The hypothesis was: turn the qualitative Goldstein/characteristics argument of
Sec.~\ref{sec:illposedness} into a quantitative, closure-independent, tau_w-free,
a-priori-deployable mechanism.  For every station of every multi-station
benchmark geometry it computes, from the matching cell ALONE:

    Pe     =  U_m y_m / (nu + nu_t(y_m))          near-wall Peclet number
    ell_x  =  Pe * y_m  =  U_m y_m^2 / (nu+nu_t)   non-locality (domain-of-
                                                   dependence) length
    ratio  =  ell_x / L_x                          dropped-convection / diffusion

with the *total* (closure-independent) eddy viscosity taken directly from the
reference Reynolds stress,

    nu_t(y_m)  =  -<u'v'>(y_m) / (dU/dy)(y_m)  =  -uv[Y_IDX] / dUdy ,

(the manuscript sign convention: the stored ``uv`` field is the raw <u'v'>, so
the Reynolds shear stress that drives momentum is -uv; cf. the dns_stress branch
of diagnostic_test_corrected.solve_controlled).  No tau_w enters ell_x -- it is
deployable inside a running WMLES, unlike eps = |tau_w|/(|dp/dx| y_m).

The 1-D ODE reduction is valid iff ratio << 1 and singular when ratio >~ 1; for
a repeating wall L_x = geometric pitch ~ O(delta), so the trigger is
ell_x >~ pitch (manuscript eq. 7, methodology.md sec. 2-3).

Streamwise scale L_x  (objective, declared per geometry; methodology.md sec. 4):
  * repeating walls  -> geometric repeat pitch (PITCH dict, fixed geometry prop.)
  * non-repeating    -> local pressure-gradient length  U_m^2 / |dp/dx|  (tau_w-free)

F3 robustness (methodology.md sec. 6.1): nu_t is singular where dU/dy -> 0
(reversed-flow core).  Stations with |dU/dy| < TAU_GRAD or nu+nu_t <= 0 are
MASKED and the masked fraction is reported per geometry as the deployability
bound.  Pre-registered decision rule: >30% masked on the failure geometry => F3
fires, narrow the claim to a theoretical interpretation.

A-priori only.  No new CFD.  <= 2 cores.  No fabrication: every output is
computed from real DNS/LES wall_profiles on disk, via the SAME hill-surface /
Y_IDX extraction used by the production pipeline.

Usage:
  smoke (L1 feasibility, hills only, NO npz written, prints F3 diagnostics):
    OMP_NUM_THREADS=2 python3 codes/analysis/nonlocality_length.py --smoke
  full (L2, all geometries, writes results/nonlocality_length.npz):
    OMP_NUM_THREADS=2 python3 codes/analysis/nonlocality_length.py
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
NDD = os.path.join(CODES, "new_data_download")
GEOM = os.path.join(NDD, "geometry_driven")
VEND = os.path.join(CODES, "vendor", "universal_wall_function", "codes", "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D            # noqa: E402  (production extraction)

Y_IDX = D.Y_IDX                                  # = 10, the manuscript matching index

# --- F3 masking parameters (methodology.md sec. 6.1) ------------------------
TAU_GRAD = 1e-3        # small-gradient guard on |dU/dy(y_m)|; swept for robustness
MASK_FRAC_KILL = 0.30  # >30% masked on the failure geometry => F3 fires

# --- benchmark geometries (SAME files as error_decomposition_cross_geometry) -
#   (label, path, is_repeating, pitch_in_ym_units_or_None)
# pitch values are GEOMETRIC repeat lengths in the same length unit as the
# stored y-column; L2 verifies them against the geometry headers / delta.  A
# value of None on a repeating wall defers L_x to the local pressure-gradient
# scale and flags it for L2 to fill from the geometry metadata.
GEOMS = [
    ("periodic_hills_1p0",
     os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
     True,  None),
    ("conv_div_channel",
     os.path.join(VEND, "conv_div_channel_Re12600_wall_profiles.npz"),
     True,  None),
    ("bfs_Re13700",
     os.path.join(VEND, "bfs_Re13700_wall_profiles.npz"),
     False, None),
    ("curved_bfs_Re13700",
     os.path.join(VEND, "curved_bfs_Re13700_LES_wall_profiles.npz"),
     False, None),
    ("nasa_hump_Re936000",
     os.path.join(VEND, "nasa_hump_Re936000_wall_profiles.npz"),
     False, None),
    ("gaussian_bump_Re1M",
     os.path.join(VEND, "gaussian_speed_bump_Re1M_wall_profiles.npz"),
     False, None),
    ("gaussian_bump_Re2M",
     os.path.join(VEND, "gaussian_speed_bump_Re2M_wall_profiles.npz"),
     False, None),
    ("separation_bubble_B",
     os.path.join(VEND, "separation_bubble_caseB_wall_profiles.npz"),
     False, None),
    ("jaxa_sep_bubble_Re600",
     os.path.join(VEND, "jaxa_separation_ReTheta600_wall_profiles.npz"),
     False, None),
]


def _col(arr, i):
    """Per-station column whether arr is 2-D (rows) or shared 1-D."""
    return arr[i] if np.ndim(arr) == 2 else arr


def _dudy_at(y, U, k):
    """Central-difference dU/dy at index k on the real (non-uniform) column."""
    if k <= 0 or k >= len(y) - 1:
        # one-sided fallback at the ends
        if k <= 0:
            return (U[1] - U[0]) / (y[1] - y[0])
        return (U[-1] - U[-2]) / (y[-1] - y[-2])
    return (U[k + 1] - U[k - 1]) / (y[k + 1] - y[k - 1])


def _dudy_fit_at(y, U, k, npts=3):
    """3-point local least-squares slope at index k (robustness estimator)."""
    lo, hi = max(0, k - npts), min(len(y), k + npts + 1)
    yy, uu = y[lo:hi], U[lo:hi]
    if len(yy) < 2:
        return np.nan
    A = np.vstack([yy, np.ones_like(yy)]).T
    slope, _ = np.linalg.lstsq(A, uu, rcond=None)[0]
    return float(slope)


def compute_geometry(label, path, is_repeating, pitch):
    """Per-station Pe, ell_x, ratio, eps, F3 mask for one wall_profiles file."""
    d = np.load(path, allow_pickle=True)
    y, U, uv = d["y"], d["U"], d["uv"]
    tau = np.asarray(d["tau_w"], float)
    dpdx = np.asarray(d["dp_dx"], float)
    nu = np.atleast_1d(np.asarray(d["nu"], float))
    n = len(tau)

    Pe = np.full(n, np.nan)
    ell_x = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    nu_t = np.full(n, np.nan)
    dudy = np.full(n, np.nan)
    ell_x_fit = np.full(n, np.nan)        # robustness estimator (3-pt fit nu_t)
    eps = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    ym_all = np.full(n, np.nan)

    for i in range(n):
        yi, Ui, uvi = _col(y, i), _col(U, i), _col(uv, i)
        if Y_IDX >= len(yi):
            continue
        y_m, U_m = float(yi[Y_IDX]), float(Ui[Y_IDX])
        ym_all[i] = y_m
        if not np.isfinite(U_m) or y_m <= 0:
            continue
        nu_i = float(nu[i]) if nu.size > 1 else float(nu[0])

        g = _dudy_at(yi, Ui, Y_IDX)
        dudy[i] = g
        tau_R = -float(uvi[Y_IDX])                 # Reynolds shear stress = -<u'v'>
        # eps (manuscript def.) -- diagnostic only, NOT used in ell_x
        den = abs(dpdx[i]) * abs(y_m)
        if den > 1e-30:
            eps[i] = abs(tau[i]) / den

        # --- F3 mask: nu_t singular where dU/dy -> 0 --------------------------
        if abs(g) < TAU_GRAD:
            continue
        nt = tau_R / g
        nu_t[i] = nt
        if nu_i + nt <= 0:                          # unphysical negative diffusivity
            continue

        diff = nu_i + nt
        Pe[i] = U_m * y_m / diff
        ell_x[i] = Pe[i] * y_m                      # = U_m y_m^2 / (nu+nu_t)
        valid[i] = True

        # robustness estimator: nu_t from a 3-pt local fit slope
        gf = _dudy_fit_at(yi, Ui, Y_IDX)
        if np.isfinite(gf) and abs(gf) >= TAU_GRAD:
            ntf = tau_R / gf
            if nu_i + ntf > 0:
                ell_x_fit[i] = U_m * y_m * y_m / (nu_i + ntf)

        # --- L_x and ratio ---------------------------------------------------
        if is_repeating and pitch is not None:
            Lx = pitch
        else:
            # local pressure-gradient streamwise length  U_m^2 / |dp/dx|
            Lx = (U_m * U_m / abs(dpdx[i])) if abs(dpdx[i]) > 1e-30 else np.nan
        if np.isfinite(Lx) and Lx > 0:
            ratio[i] = ell_x[i] / Lx

    v = valid
    masked_frac = float(1.0 - v.mean()) if n else np.nan
    ratio_v = ratio[v & np.isfinite(ratio)]
    eps_v = eps[np.isfinite(eps)]
    # estimator stability: relative spread between central-diff and 3-pt-fit ell_x
    both = v & np.isfinite(ell_x_fit) & (ell_x > 0)
    est_spread = (float(np.median(np.abs(ell_x_fit[both] - ell_x[both])
                                  / ell_x[both])) if both.sum() else np.nan)

    summary = dict(
        label=label, n=int(n), n_valid=int(v.sum()),
        masked_frac=masked_frac,
        Pe_med=float(np.nanmedian(Pe[v])) if v.any() else np.nan,
        ell_x_med=float(np.nanmedian(ell_x[v])) if v.any() else np.nan,
        ratio_med=float(np.median(ratio_v)) if ratio_v.size else np.nan,
        eps_med=float(np.median(eps_v)) if eps_v.size else np.nan,
        ym_med=float(np.nanmedian(ym_all[v])) if v.any() else np.nan,
        est_spread=est_spread,
        is_repeating=bool(is_repeating),
    )
    fields = dict(Pe=Pe, ell_x=ell_x, ratio=ratio, nu_t=nu_t, dudy=dudy,
                  eps=eps, valid=valid, ym=ym_all, ell_x_fit=ell_x_fit)
    return summary, fields


def run(smoke=False):
    geoms = GEOMS[:1] if smoke else GEOMS
    print(f"nonlocality_length.py  ({'SMOKE / L1 feasibility' if smoke else 'FULL / L2'})")
    print(f"Y_IDX={Y_IDX}  TAU_GRAD={TAU_GRAD}  geometries={len(geoms)}\n")
    print(f"{'geometry':<22}{'n':>5}{'valid':>7}{'mask%':>7}"
          f"{'Pe_med':>9}{'ellx_med':>10}{'ratio_med':>11}{'eps_med':>9}"
          f"{'estspread':>10}")
    summaries, all_fields = [], {}
    for label, path, rep, pitch in geoms:
        if not os.path.exists(path):
            print(f"{label:<22}  MISSING: {path}")
            continue
        s, f = compute_geometry(label, path, rep, pitch)
        summaries.append(s)
        all_fields[label] = f
        print(f"{s['label']:<22}{s['n']:>5}{s['n_valid']:>7}"
              f"{100*s['masked_frac']:>6.1f} {s['Pe_med']:>9.3g}"
              f"{s['ell_x_med']:>10.3g}{s['ratio_med']:>11.3g}"
              f"{s['eps_med']:>9.3g}{s['est_spread']:>10.3g}")

    if smoke:
        s = summaries[0]
        print("\n--- F3 feasibility (failure geometry) ---")
        print(f"masked fraction = {100*s['masked_frac']:.1f}%  "
              f"(kill threshold {100*MASK_FRAC_KILL:.0f}%): "
              f"{'PASS -> nu_t robustly extractable' if s['masked_frac'] < MASK_FRAC_KILL else 'FAIL -> F3 fires'}")
        print(f"estimator spread (central-diff vs 3-pt-fit nu_t) = "
              f"{s['est_spread']:.3g} (small => stable)")
        print(f"median Pe={s['Pe_med']:.3g}, median ell_x={s['ell_x_med']:.3g} "
              f"(matching-height units; L_x normalisation is L2)")
        print("\nSMOKE: no npz written; full multi-geometry run + "
              "nonlocality_length.npz are L2 deliverables.")
        return

    out = os.path.join(RESULTS, "nonlocality_length.npz")
    pack = {"summaries": np.array(summaries, dtype=object),
            "Y_IDX": Y_IDX, "TAU_GRAD": TAU_GRAD}
    for label, f in all_fields.items():
        for k, v in f.items():
            pack[f"{label}__{k}"] = v
    np.savez(out, **pack)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="L1 feasibility: hills only, no npz, print F3 diagnostics")
    run(smoke=ap.parse_args().smoke)
