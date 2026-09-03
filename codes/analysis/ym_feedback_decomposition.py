#!/usr/bin/env python3
r"""
ym_feedback_decomposition.py
============================
Decompose the a-priori -> a-posteriori wall-stress "healing" on periodic hills
into two physically distinct contributions:

    Rung 0  (a-priori, manuscript protocol)  R^2(tau_w) = -47.7
              |  matching height y_m moves from the a-priori protocol
              |  (y_m ~ 0.094 H, Y_IDX=10) down to the COUPLED WMLES
              v  first-cell height (y_m ~ 0.011-0.014 H)            [y_m effect]
    Rung 1  (a-priori model, evaluated at the COUPLED y_m)  R^2(tau_w) = ?
              |  inputs become self-consistent: the LES supplies U_m
              v  instead of pristine DNS U_m at the same height     [feedback]
    Rung 2  (fully coupled WMLES)  R^2(C_f) = +0.762

This answers the single most important question the L1 review raised: how much of
the +0.762-vs-(-47.7) gap is merely the finer coupled matching height (a protocol
/ geometric effect) and how much is genuine coupling feedback (the self-consistent
operating-point shift)?  The HONESTY FLAG carried from L0 is explicit: do NOT
over-attribute the healing to coupling feedback.

The decomposition is EXACT for the y_m effect: Rung 0 and Rung 1 use the SAME
512 hill-surface DNS stations, the SAME ground-truth tau_w, the SAME production
ODE wall model, the SAME near-wall dp/dx -- ONLY the matching height y_m changes.
Rung 1 -> Rung 2 (feedback) is the residual: same equilibrium closure, same y_m
band, but U_m now comes from the self-consistent LES rather than pristine DNS.

Why finer y_m heals (the physical "because"):
  In the near-wall layer the integrated convection + pressure-gradient forcing
  that the ODE NEGLECTS scales as ~ (C + dp/dx) * y_m.  The surviving wall stress
  the ODE must reproduce is tau_w.  The ODE error is therefore ~ |C+dp/dx|*y_m /
  |tau_w| = (1/eps)*(y_m/y_ref): LINEAR in y_m.  Halving y_m halves the structural
  error.  The coupled WMLES sits ~7x lower than the a-priori protocol, so a large
  part of its apparent "healing" is this geometric 1/y_m factor -- NOT a cure of
  the structural defect.  At fixed, coarse y_m the defect is unchanged.

Inputs  (real, on disk):
  raw DNS  : codes/new_data_download/.../case_1p0/dns-data/{mean,rms}_files.dat
             (via diagnostic_test_corrected.extract_profiles)
  coupled  : codes/results/matching_height_validation.npz  (ym_mesh(x), 80 stn)
             codes/results/coupled_wallstress_transmission.npz (eq_R2_coupled_cf)

Output:
  codes/results/ym_feedback_decomposition.npz

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/ym_feedback_decomposition.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # codes/
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D                       # noqa: E402
from diagnostic_test_corrected import _main_predict_tau_w   # noqa: E402

NU = D.NU
Y_IDX_APRIORI = D.Y_IDX            # =10, the manuscript a-priori protocol height


def r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rel_rms(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return np.sqrt(np.mean((yhat - y) ** 2)) / np.sqrt(np.mean(y ** 2))


def predict_at_height(profs, y_m_of_x, xs):
    """Predict tau_w with the production ODE model, matching at a prescribed
    physical height y_m (per station). y_m_of_x is either a scalar (same y_m for
    all stations) or an array aligned with `xs`. Returns (pred, gt, valid)."""
    n = len(profs)
    pred = np.full(n, np.nan)
    gt = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    ymp = np.full(n, np.nan)        # y_m^+ for bookkeeping
    for i, pr in enumerate(profs):
        y, U = pr["y"], pr["U"]
        y_m = float(y_m_of_x) if np.isscalar(y_m_of_x) else float(y_m_of_x[i])
        # need at least one fluid point above the requested height
        if y_m <= y[0] or y_m > y[-1]:
            continue
        U_m = float(np.interp(y_m, y, U))
        tw = _main_predict_tau_w(U_m, y_m, pr["dpdx"], NU)
        if not np.isfinite(tw):
            continue
        pred[i] = tw
        gt[i] = pr["tau_w_dns"]
        valid[i] = True
        ymp[i] = y_m * np.sqrt(abs(pr["tau_w_dns"])) / NU
    return pred, gt, valid, ymp


def score(pred, gt, valid):
    m = valid & np.isfinite(pred) & np.isfinite(gt)
    return (r2(gt[m], pred[m]), rel_rms(gt[m], pred[m]),
            float(np.mean(np.sign(pred[m]) == np.sign(gt[m]))), int(m.sum()))


def main():
    t0 = time.time()
    profs = D.extract_profiles()
    xs = np.array([0.0])  # placeholder; we rebuild xs from the extraction below
    # rebuild xs in the same order extract_profiles emits (unique rounded x)
    mean = np.loadtxt(os.path.join(D.RAW_DIR, "mean_files.dat"))
    xu = np.unique(np.round(mean[:, 0], 6))
    # extract_profiles drops stations with too few fluid points; realign by
    # re-deriving the surviving x via the same filter is unnecessary because
    # predict_at_height is indexed by profile, and we only need per-station y_m
    # for the COUPLED rung -> we map coupled ym(x) onto the surviving stations.
    # Recover the surviving x list:
    xs_keep = []
    VEL_TOL = D.VEL_TOL
    x, y_, u_, v_ = mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3]
    for xv in xu:
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv = y_[m], u_[m], v_[m]
        o = np.argsort(yy)
        uu, vv = uu[o], vv[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < D.Y_IDX + 2:
            continue
        xs_keep.append(xv)
    xs = np.asarray(xs_keep)
    assert len(xs) == len(profs), (len(xs), len(profs))
    print(f"Extracted {len(profs)} hill-surface stations  ({time.time()-t0:.1f}s)")

    # ---- Rung 0: a-priori protocol height (Y_IDX=10) -------------------------
    ym_apriori = np.array([pr["y"][Y_IDX_APRIORI] for pr in profs])
    p0, g0, v0, ymp0 = predict_at_height(profs, ym_apriori, xs)
    R0, rr0, sa0, n0 = score(p0, g0, v0)
    print(f"Rung 0  a-priori  y_m~{np.median(ym_apriori):.4f}H  "
          f"(y_m+~{np.nanmedian(ymp0):.1f})  R2={R0:+.2f}  relRMS={rr0:.3f}  "
          f"SA={sa0:.3f}  N={n0}")

    # ---- coupled matching height (per station, mapped onto a-priori x) -------
    mh = np.load(os.path.join(RESULTS, "matching_height_validation.npz"))
    cx, cym = mh["x"], mh["ym_mesh"]
    order = np.argsort(cx)
    ym_coupled_on_x = np.interp(xs, cx[order], cym[order])
    print(f"Coupled y_m band: median={np.median(cym):.4f}H  "
          f"range=[{cym.min():.4f},{cym.max():.4f}]H "
          f"(~{np.median(ym_apriori)/np.median(cym):.1f}x finer than a-priori)")

    # ---- Rung 1: a-priori MODEL at the COUPLED y_m (pristine DNS U_m) ---------
    p1, g1, v1, ymp1 = predict_at_height(profs, ym_coupled_on_x, xs)
    R1, rr1, sa1, n1 = score(p1, g1, v1)
    print(f"Rung 1  a-priori model @ coupled y_m~{np.median(ym_coupled_on_x):.4f}H "
          f"(y_m+~{np.nanmedian(ymp1):.1f})  R2={R1:+.2f}  relRMS={rr1:.3f}  "
          f"SA={sa1:.3f}  N={n1}")

    # ---- Rung 2: fully coupled WMLES (self-consistent LES U_m) ---------------
    tr = np.load(os.path.join(RESULTS, "coupled_wallstress_transmission.npz"))
    R2c = float(tr["eq_R2_coupled_cf"])
    rr2 = float(tr["eq_relrms_coupled_cf"])
    print(f"Rung 2  coupled WMLES (C_f vs same DNS)            R2={R2c:+.3f}  "
          f"relRMS={rr2:.3f}")

    # ---- y_m sweep (the ladder as a continuous curve) ------------------------
    # sweep target y_m from finer than coupled up to coarser than a-priori
    sweep_ym = np.linspace(0.006, 0.18, 30)
    sweep_R2 = np.full(sweep_ym.size, np.nan)
    sweep_relrms = np.full(sweep_ym.size, np.nan)
    sweep_ymp = np.full(sweep_ym.size, np.nan)
    sweep_N = np.zeros(sweep_ym.size, int)
    for j, ym in enumerate(sweep_ym):
        pj, gj, vj, ympj = predict_at_height(profs, ym, xs)
        Rj, rrj, saj, nj = score(pj, gj, vj)
        sweep_R2[j], sweep_relrms[j] = Rj, rrj
        sweep_ymp[j] = np.nanmedian(ympj)
        sweep_N[j] = nj
    print(f"Sweep: R2 ranges [{np.nanmin(sweep_R2):+.1f}, {np.nanmax(sweep_R2):+.2f}] "
          f"over y_m in [{sweep_ym.min():.3f},{sweep_ym.max():.3f}]H")

    # ---- decomposition of the healing ----------------------------------------
    delta_ym = R1 - R0           # pure matching-height (geometric/protocol) effect
    delta_feedback = R2c - R1    # coupling self-consistency (feedback) effect
    total = R2c - R0
    # fraction of the total healing attributable to each (on the R2 axis)
    frac_ym = delta_ym / total
    frac_feedback = delta_feedback / total
    print("\n=== Transmission ladder (R^2 on the common wall-stress axis) ===")
    print(f"  Rung 0  a-priori (DNS-protocol y_m)      R2 = {R0:+.2f}")
    print(f"  Rung 1  a-priori model @ coupled y_m     R2 = {R1:+.2f}")
    print(f"  Rung 2  coupled WMLES                    R2 = {R2c:+.3f}")
    print(f"  ----------------------------------------------------------")
    print(f"  Delta(y_m, geometric)   = {delta_ym:+.2f}   "
          f"({100*frac_ym:.0f}% of healing)")
    print(f"  Delta(feedback, coupling)= {delta_feedback:+.2f}   "
          f"({100*frac_feedback:.0f}% of healing)")
    print(f"  Total healing            = {total:+.2f}")

    # R^2 is a saturating metric; report the same split on the (more linear)
    # rel-RMS error axis as an honesty cross-check.
    d_ym_rr = rr0 - rr1          # error REMOVED by finer y_m
    d_fb_rr = rr1 - rr2          # error removed by feedback
    tot_rr = rr0 - rr2
    frac_ym_rr = d_ym_rr / tot_rr
    frac_fb_rr = d_fb_rr / tot_rr
    print("  -- rel-RMS axis (cross-check) --")
    print(f"  relRMS ladder: {rr0:.2f} -> {rr1:.2f} -> {rr2:.2f}")
    print(f"  Delta(y_m)={d_ym_rr:.2f} ({100*frac_ym_rr:.0f}%)  "
          f"Delta(feedback)={d_fb_rr:.2f} ({100*frac_fb_rr:.0f}%)")
    print(f"  Sign-agreement: {sa0:.2f} (a-priori) -> {sa1:.2f} (coupled y_m) "
          f"-- finer height fixes the separated-region sign")

    out = dict(
        # rel-RMS-axis decomposition (cross-check, more linear than R^2)
        delta_ym_relrms=d_ym_rr, delta_feedback_relrms=d_fb_rr,
        total_healing_relrms=tot_rr,
        frac_ym_relrms=frac_ym_rr, frac_feedback_relrms=frac_fb_rr,
        # rungs
        R2_rung0_apriori=R0, relrms_rung0=rr0, sa_rung0=sa0, n_rung0=n0,
        R2_rung1_coupled_ym=R1, relrms_rung1=rr1, sa_rung1=sa1, n_rung1=n1,
        R2_rung2_coupled=R2c, relrms_rung2=rr2,
        # heights
        ym_apriori=ym_apriori, ym_coupled_on_x=ym_coupled_on_x, xs=xs,
        ym_apriori_median=float(np.median(ym_apriori)),
        ym_coupled_median=float(np.median(cym)),
        ymp_rung0_median=float(np.nanmedian(ymp0)),
        ymp_rung1_median=float(np.nanmedian(ymp1)),
        # decomposition
        delta_ym=delta_ym, delta_feedback=delta_feedback, total_healing=total,
        frac_ym=frac_ym, frac_feedback=frac_feedback,
        # sweep
        sweep_ym=sweep_ym, sweep_R2=sweep_R2, sweep_relrms=sweep_relrms,
        sweep_ymp=sweep_ymp, sweep_N=sweep_N,
        # provenance
        Y_IDX_APRIORI=Y_IDX_APRIORI, NU=NU,
    )
    np.savez(os.path.join(RESULTS, "ym_feedback_decomposition.npz"), **out)
    print(f"\nSaved -> results/ym_feedback_decomposition.npz  "
          f"({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
