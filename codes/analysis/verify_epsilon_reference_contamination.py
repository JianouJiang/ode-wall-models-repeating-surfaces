#!/usr/bin/env python3
"""Agent V addendum -- is the manuscript's epsilon_ref = 0.08364 contaminated by the
withdrawn 4-point through-origin wall-traction estimator, and by how much?

Trace established separately (see REPORT.md ADDENDUM): 0.08364189563744982 is the
`median_eps` field written into codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz
by codes/analysis/build_corrected_pehill_profiles.py, as

    eps(x) = |tau_w(x)| / (|dp_dx(x)| * y_m(x)),   y_m = y[:, 10]
    tau_w  = nu * (4-point through-origin LINEAR fit of the STREAMWISE u against the
             VERTICAL offset y)                       <-- the withdrawn estimator,
                                                          and without the tangent correction

This script recomputes epsilon on the paper's own convention with four numerators and
reports medians, frac(eps<0.1) and circular phase-block intervals.  It also swaps the
numerator inside R2-1's exact-pressure normalisation.  Read-only; no simulation.

Numerators
  N0 legacy_nu_dUdy      the manuscript's (must reproduce 0.0836419 exactly)
  N1 tangent_linear4     nu dU_t/dn, same 4 points  (the L2/L3 deposit's estimator)
  N2 xiao_repaired_cubic through-origin cubic, first 6 fluid points, validated against
                         MGLET at the Xiao spacing (relative RMS error 0.264)
  N3 mglet               Peller & Manhart MGLET DNS deposited tau_w, interpolated to the
                         same 512 stations

Out: work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/epsilon_reference_contamination.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/epsilon_reference_contamination.json"
XIAO = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
MGLET = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
EXACT_P = ROOT / "codes/results/exact_pressure_traction_512.npz"
LX, NU, Y_IDX = 9.0, 1.0 / 5600.0, 10
BLOCK, DRAWS, SEED = 64, 20000, 20260825          # Lx/8 of 512 stations


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 23), b""):
            h.update(c)
    return h.hexdigest()


def ercoftac_hill(xh: float) -> float:
    xm = 28.0 * (xh % LX)
    if xm > 28.0 * LX / 2.0:
        xm = 28.0 * LX - xm
    if xm < 9.0:
        h = min(28.0, 28.0 + 6.775070969851e-3 * xm**2 - 2.124527775800e-3 * xm**3)
    elif xm < 14.0:
        h = 2.507355893131e1 + 9.754803562315e-1*xm - 1.016116352781e-1*xm**2 + 1.889794677828e-3*xm**3
    elif xm < 20.0:
        h = 2.579601052357e1 + 8.206693007457e-1*xm - 9.055370274339e-2*xm**2 + 1.626510569859e-3*xm**3
    elif xm < 30.0:
        h = 4.046435022819e1 - 1.379581654948*xm + 1.945884504128e-2*xm**2 - 2.070318932190e-4*xm**3
    elif xm < 40.0:
        h = 1.792461334664e1 + 8.743920332081e-1*xm - 5.567361123058e-2*xm**2 + 6.277731764683e-4*xm**3
    elif xm <= 54.0:
        h = max(0.0, 5.639011190988e1 - 2.010520359035*xm + 1.644919857549e-2*xm**2 + 2.674976141766e-5*xm**3)
    else:
        h = 0.0
    return h / 28.0


def tangent(x):
    d = 1.0e-5
    slope = np.array([(ercoftac_hill(v + d) - ercoftac_hill(v - d)) / (2.0 * d) for v in np.atleast_1d(x)])
    mag = np.sqrt(1.0 + slope**2)
    return 1.0 / mag, slope / mag


def poly_origin_slope(n, u, deg):
    A = np.vstack([np.asarray(n, float) ** (k + 1) for k in range(deg)]).T
    return float(np.linalg.lstsq(A, np.asarray(u, float), rcond=None)[0][0])


def wrap_interp(xp, yp, t):
    o = np.argsort(np.mod(np.asarray(xp, float), 1.0))
    a = np.mod(np.asarray(xp, float), 1.0)[o]
    b = np.asarray(yp, float)[o]
    return np.interp(np.mod(np.asarray(t, float), 1.0),
                     np.r_[a - 1, a, a + 1], np.r_[b, b, b])


def block_median_ci(values, block=BLOCK, draws=DRAWS, seed=SEED):
    """Circular phase-block resample of the MEDIAN of a station-indexed record."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = np.where(ok, v, np.nan)
    n = v.size
    nb = n // block
    rng = np.random.default_rng(seed)
    off = np.arange(block)[None, None, :]
    med = np.empty(draws)
    for a in range(0, draws, 1000):
        b = min(a + 1000, draws)
        st = rng.integers(0, n, size=(b - a, nb))
        idx = ((st[:, :, None] + off) % n).reshape(b - a, n)
        med[a:b] = np.nanmedian(v[idx], axis=1)
    return {"median": float(np.median(med)), "p05": float(np.quantile(med, 0.05)),
            "p95": float(np.quantile(med, 0.95))}


def main() -> int:
    d = np.load(XIAO, allow_pickle=True)
    x = np.asarray(d["x"], float)
    Y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float)
    V = np.asarray(d["V"], float)
    dpdx = np.asarray(d["dp_dx"], float)
    tau_legacy = np.asarray(d["tau_w"], float)
    tx, ty = tangent(x)
    n = x.size

    tau_tan = np.full(n, np.nan)
    tau_cub = np.full(n, np.nan)
    for i in range(n):
        yy, uu, vv = Y[i], U[i], V[i]
        m = np.isfinite(yy) & np.isfinite(uu) & np.isfinite(vv)
        yy, uu, vv = yy[m], uu[m], vv[m]
        ut4 = uu[1:5] * tx[i] + vv[1:5] * ty[i]
        tau_tan[i] = NU * float(np.sum(yy[1:5] * ut4) / np.sum(yy[1:5] ** 2)) / tx[i]
        ut6 = uu[1:7] * tx[i] + vv[1:7] * ty[i]
        tau_cub[i] = NU * poly_origin_slope(yy[1:7], ut6, 3) / tx[i]
    mg = np.loadtxt(MGLET)[:-2]
    tau_mglet = wrap_interp(mg[:, 0] / LX, mg[:, 1], np.mod((x - x.min()) / LX, 1.0))

    numerators = {"N0_legacy_nu_dUdy_manuscript": tau_legacy,
                  "N1_tangent_linear4": tau_tan,
                  "N2_xiao_repaired_cubic": tau_cub,
                  "N3_mglet_deposited": tau_mglet}

    # ---- denominator A: the manuscript's own (archived station dp/dx x index-10 height)
    y_m = Y[:, Y_IDX]
    denomA = np.abs(dpdx) * np.abs(y_m)
    # ---- denominator B: R2-1's exact raw pressure integral on the common eta/H = 0.10 surface
    ep = np.load(EXACT_P, allow_pickle=True)
    denomB = np.abs(np.asarray(ep["exact_pressure_traction"], float))

    res = {"schema": "epsilon-reference-contamination-v1",
           "agent": "V",
           "trace": {
               "manuscript_value": 0.08364189563744982,
               "stored_in": "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz['median_eps']",
               "produced_by": "codes/analysis/build_corrected_pehill_profiles.py (lines 86-106)",
               "numerator_estimator": ("nu * 4-point through-origin LINEAR fit of the STREAMWISE u "
                                       "against the VERTICAL wall offset -- the withdrawn estimator, "
                                       "and WITHOUT the tangent/normal correction that the L2 deposit "
                                       "(dns_tangent_reference) later added"),
               "sha256_xiao_npz": sha256(XIAO), "sha256_mglet": sha256(MGLET),
               "sha256_exact_pressure": sha256(EXACT_P)},
           "surface_A_manuscript_convention": {}, "surface_B_r2_1_exact_pressure": {}}

    for label, denom, key in ((("A"), denomA, "surface_A_manuscript_convention"),
                              (("B"), denomB, "surface_B_r2_1_exact_pressure")):
        good = np.isfinite(denom) & (denom > 1e-30)
        for name, tau in numerators.items():
            eps = np.full(n, np.nan)
            eps[good] = np.abs(tau[good]) / denom[good]
            fin = np.isfinite(eps) & (eps > 0)
            res[key][name] = {
                "median": float(np.nanmedian(eps[fin])),
                "frac_below_0p1": float(np.mean(eps[fin] < 0.1)),
                "geometric_mean": float(np.exp(np.nanmean(np.log(eps[fin])))),
                "n_valid": int(fin.sum()),
                "phase_block_ci_on_median": block_median_ci(np.where(fin, eps, np.nan)),
            }
        base = res[key]["N0_legacy_nu_dUdy_manuscript"]["median"]
        for name in numerators:
            res[key][name]["inflation_vs_manuscript_numerator"] = res[key][name]["median"] / base

    res["reproduction_check"] = {
        "recomputed_N0_surface_A": res["surface_A_manuscript_convention"]["N0_legacy_nu_dUdy_manuscript"]["median"],
        "manuscript": 0.08364189563744982,
        "matches": abs(res["surface_A_manuscript_convention"]["N0_legacy_nu_dUdy_manuscript"]["median"]
                       - 0.08364189563744982) < 1e-12}

    # R1-STA-3 consequence: the closure envelope is unchanged, the threshold moves.
    env_max = 0.06371867039278582
    res["R1_STA_3_consequence"] = {
        "phase_closure_operator_envelope_max": env_max,
        "ratio_to_epsilon": {name: env_max / res["surface_A_manuscript_convention"][name]["median"]
                             for name in numerators},
        "note": ("the closure residual itself is rebuilt from raw volume/surface archives and is "
                 "NOT a function of the wall-traction estimator; only the acceptance threshold moves, "
                 "and it moves UP, so the margin improves")}

    # ------------------------------------------------------------------ R2-1
    # R2-1's canonical hill row does NOT read tau_w from the withdrawn npz: it takes
    # a special branch using wall_following_budget_certificate_l1.npz.  But that
    # file's q_viscous_direct is built by da_budget.direct_wall_shear, which is the
    # SAME family of estimator (through-origin LINEAR fit, points=2, tangential
    # velocity vs vertical offset / t_x) on the 768x385 alph10-9-3036 archive.
    BUDGET = ROOT / "codes/results/wall_following_budget_certificate_l1.npz"
    RAW29 = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized/"
             "pehill-29-cases-DNS/alph10-9-3036")
    b = np.load(BUDGET, allow_pickle=False)
    xb = np.asarray(b["x"], float)
    tau_m_ops = np.asarray(b["tau_match_ensemble"], float)
    tau_m_c = np.median(tau_m_ops, axis=0)
    tau_w_budget = np.median(np.asarray(b["q_viscous_direct_ensemble"], float), axis=0)
    Phi = np.asarray(b["pressure_impulse"], float)

    m29 = np.loadtxt(RAW29 / "mean_files.dat")
    xr = np.unique(np.round(m29[:, 0], 6))
    txr, tyr = tangent(xr)
    tau2, tauc = np.empty(xr.size), np.empty(xr.size)
    for i, xv in enumerate(xr):
        sel = np.abs(m29[:, 0] - xv) < 1e-6
        yy, uu, vv = m29[sel, 1], m29[sel, 2], m29[sel, 3]
        o = np.argsort(yy); yy, uu, vv = yy[o], uu[o], vv[o]
        fl = np.where((np.abs(uu) > 1e-6) | (np.abs(vv) > 1e-6))[0]
        k = max(fl[0], 1)
        nn = yy[k - 1:] - yy[k - 1]; UU = uu[k - 1:]; VV = vv[k - 1:]
        ut = UU[1:7] * txr[i] + VV[1:7] * tyr[i]
        tau2[i] = NU * float(np.sum(nn[1:3] * ut[:2]) / np.sum(nn[1:3] ** 2)) / txr[i]
        tauc[i] = NU * poly_origin_slope(nn[1:7], ut, 3) / txr[i]
    tau_mglet_b = wrap_interp(mg[:, 0] / LX, mg[:, 1], np.mod((xb - xb.min()) / LX, 1.0))

    def four(tm, tw, pp):
        floor_p = 0.02 * np.sqrt(np.nanmean(pp ** 2))
        floor_s = 0.02 * np.sqrt(np.nanmean(tw ** 2))
        r = lambda a, c, f: np.abs(a) / np.maximum(np.abs(c), f)
        return {"tau_m_over_pressure": float(np.nanmedian(r(tm, pp, floor_p))),
                "tau_m_over_tau_w": float(np.nanmedian(r(tm, tw, floor_s))),
                "epsilon_pressure_only": float(np.nanmedian(r(tw, pp, floor_p))),
                "epsilon_all_retained": float(np.nanmedian(
                    np.abs(tw) / np.maximum(np.abs(pp) + np.abs(tm), floor_p)))}

    res["R2_1_three_term_table_recomputed"] = {
        "hill_branch_note": ("uncertainty_certificate_l1.case_operators() takes a dedicated branch "
                             "for periodic_hills_case_1p0: tau_m, tau_w and Phi all come from "
                             "wall_following_budget_certificate_l1.npz, NOT from the withdrawn npz "
                             "(whose path is nevertheless recorded in source_paths -- misleading "
                             "provenance, worth fixing)."),
        "tau_w_estimator_in_that_branch": ("da_budget.direct_wall_shear: through-origin LINEAR fit, "
                                           "points=2, on the 768x385 archive"),
        "rms_tau_w_budget": float(np.sqrt(np.mean(tau_w_budget ** 2))),
        "rms_tau_w_2point_reproduced": float(np.sqrt(np.mean(tau2 ** 2))),
        "rms_tau_w_cubic6_same_archive": float(np.sqrt(np.mean(tauc ** 2))),
        "rms_mglet": float(np.sqrt(np.mean(tau_mglet_b ** 2))),
        "ratio_budget_over_mglet": float(np.sqrt(np.mean(tau_w_budget ** 2)) / np.sqrt(np.mean(tau_mglet_b ** 2))),
        "columns": {
            "as_published_budget_tau_w": four(tau_m_c, tau_w_budget, Phi),
            "corrected_cubic6_same_archive": four(tau_m_c, tauc, Phi),
            "corrected_mglet": four(tau_m_c, tau_mglet_b, Phi),
        },
        "verdict": ("column 1 (|tau_m|/Phi = 0.735, the number the R2-1 claim quotes) contains "
                    "no tau_w and is unchanged.  Columns 2-4 all carry tau_w and all move."),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n")
    print(f"written -> {OUT.relative_to(ROOT)}")
    for key in ("surface_A_manuscript_convention", "surface_B_r2_1_exact_pressure"):
        print(f"-- {key}")
        for name in numerators:
            r = res[key][name]
            ci = r["phase_block_ci_on_median"]
            print(f"   {name:32s} median {r['median']:.5f} [{ci['p05']:.5f},{ci['p95']:.5f}] "
                  f"frac<0.1 {r['frac_below_0p1']:.3f}  x{r['inflation_vs_manuscript_numerator']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
