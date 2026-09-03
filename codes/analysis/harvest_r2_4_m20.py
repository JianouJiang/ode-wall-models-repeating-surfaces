#!/usr/bin/env python3
r"""
harvest_r2_4_m20.py -- R2-4 / M20: matched-numerics rib pair + cube-array WRLES.
===============================================================================
Turns the reduced ARCHER2 deposits in ``codes/results/r2_4_m20/<case>/``
(written in-job by ``codes/openfoam/r24_les_reduce.py``) into the row
artifact ``codes/results/r2_4_m20_les_<date>.{json,npz}``.

Every wall-model number goes through the SAME instrument as every hill and
the deposited d-type rib number: ``rib_eps_ode.evaluate`` (imported, not
retyped), with the deposit's conventions (``rib_harvest.Y_IDX = 10``,
least-squares wall gradient through the origin over the first four cells,
near-wall pressure gradient by ``numpy.gradient`` along the horizontal wall
set, ``rib_resolved_fraction.resolved_fraction`` for f_res).

Per case and per averaging window the harvest reports, with uncertainty:
  * R^2(tau_w), relRMS for standard_ml / controlled_ml / controlled_dns /
    controlled_dns_total; eps_median; f(eps<1); f(eps<0.1); f_res;
  * uncertainty = (i) phase-block replicates: one verdict per rib pitch (rib)
    or per cube cell (cube) -> mean, sd, min, max, t-interval;
    (ii) circular moving-block bootstrap over stations (M1 convention,
    block = ceil(sqrt(N)), 5000 replicates) -> 95 % interval;
    (iii) window envelope: the deposit's two cumulative windows plus two
    DISJOINT windows rebuilt exactly from the cumulative means.
  * validation observables: reattachment length x_r/k, cavity reversed-flow
    fraction, wall-drag partition (pressure / viscous) with block-std from the
    forces time series, u_tau and Re_tau, the momentum-balance closure
    (driving force vs integrated wall force), and the turbulence-sustainment
    audit (vv, ww, nut/nu in the upper half) that the deposited d-type fails.
  * cube: intrinsically averaged <U>(y)/u_tau with a log-law (kappa=0.41)
    fit of d and z0 above 2h, drag partition cube-pressure / cube-viscous /
    floor-viscous, floor-station verdicts at the index convention, at the
    rib-matched physical height y_m/h = 0.146 and at y+ = 50.
  * grid check G1 (production) vs G0 (r = 0.75): verdict invariance.

Usage:  python3 codes/analysis/harvest_r2_4_m20.py [--date 20260823] [--allow-pilot]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
DEPOSITS = RESULTS / "r2_4_m20"
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
from rib_eps_ode import evaluate                                      # noqa: E402
from rib_harvest import Y_IDX, NU as NU_RIB, K as K_RIB                # noqa: E402
from rib_resolved_fraction import resolved_fraction, summarise         # noqa: E402

SEED = 20260823
N_BOOT = 5000
CLOSURES = ("standard_ml", "controlled_ml", "controlled_dns", "controlled_dns_total")
NFIT = 4                      # deposit convention: LS wall gradient over first 4 cells
Y_M_OVER_K_RIB = None         # filled from the production d-type column (y[Y_IDX]/k)


# --------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def circular_indices(rng, n, block):
    starts = rng.integers(0, n, size=int(math.ceil(n / block)))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def block_bootstrap(values_true, values_pred, eps, rng):
    n = len(values_true)
    block = max(2, int(math.ceil(math.sqrt(n))))
    r2s = np.empty(N_BOOT)
    epsm = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = circular_indices(rng, n, block)
        r2s[b] = r2(values_pred[idx], values_true[idx])
        e = eps[idx]
        epsm[b] = np.nanmedian(e)
    return dict(block_length=block, n_boot=N_BOOT,
                r2_ci95=[float(np.nanpercentile(r2s, 2.5)), float(np.nanpercentile(r2s, 97.5))],
                eps_median_ci95=[float(np.nanpercentile(epsm, 2.5)), float(np.nanpercentile(epsm, 97.5))])


def t_interval(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    n = len(vals)
    if n == 0:
        return dict(n=0)
    out = dict(n=int(n), mean=float(vals.mean()), min=float(vals.min()), max=float(vals.max()))
    if n >= 2:
        sd = float(vals.std(ddof=1))
        # two-sided 95 % t quantiles for n-1 dof (small table, no scipy dependency)
        tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}.get(n - 1, 1.96)
        out.update(sd=sd, ci95=[out["mean"] - tq * sd / math.sqrt(n),
                                out["mean"] + tq * sd / math.sqrt(n)])
    return out


# --------------------------------------------------------------------------
# window algebra: cumulative fieldAverage means -> disjoint windows (exact)
# --------------------------------------------------------------------------
def load_windows(files, t0):
    wins = []
    for f in sorted(files, key=lambda p: float(np.load(p)["time"])):
        d = np.load(f, allow_pickle=False)
        wins.append(dict(file=f, t=float(d["time"]), data={k: d[k] for k in d.files}))
    return wins


def disjoint_window(w_a, w_b, t0, mean_keys, prime_pairs):
    """Window [t_a, t_b] from cumulative windows [t0,t_a] and [t0,t_b]."""
    Ta, Tb = w_a["t"] - t0, w_b["t"] - t0
    Tw = Tb - Ta
    out = {}
    A, B = w_a["data"], w_b["data"]
    for k in mean_keys:
        if k in A and k in B:
            out[k] = (Tb * B[k] - Ta * A[k]) / Tw
    # prime2Mean: <u'u'>_t = <uu>_t - U_t U_t  ->  <uu> additive
    for (pk, ua, ub) in prime_pairs:
        if pk in A and pk in B:
            uu_a = A[pk] + A[ua] * A[ub]
            uu_b = B[pk] + B[ua] * B[ub]
            uu_w = (Tb * uu_b - Ta * uu_a) / Tw
            out[pk] = uu_w - out[ua] * out[ub]
    return out


RIB_MEAN_KEYS = ("U", "V", "W", "p", "nutMean", "nut")
RIB_PRIME = (("uu", "U", "U"), ("uv", "U", "V"), ("uw", "U", "W"),
             ("vv", "V", "V"), ("vw", "V", "W"), ("ww", "W", "W"))


# --------------------------------------------------------------------------
# RIB: station profiles from the spanwise-averaged 2-D field
# --------------------------------------------------------------------------
def rib_profiles(fld, x, y, prov, fold=True, rib_index=None):
    """One enriched profile per streamwise station on the horizontal wall set
    (cavity floor y=0, rib top y=k), deposit conventions.  fold=True averages
    the N ribs onto one pitch (phase average); rib_index selects one pitch."""
    k, w, p = prov["k_rib_height"], prov["w_rib_width"], prov["p_pitch"]
    n_ribs = int(prov["n_ribs"])
    nu = prov["nu"]
    xp = np.mod(x, p)                                  # phase within the pitch
    rib = np.floor(x / p).astype(int)
    if rib_index is not None:
        sel_r = rib == rib_index
    else:
        sel_r = np.ones_like(x, bool)
    over_rib = (xp > (p - w) / 2 - 1e-9) & (xp < (p + w) / 2 + 1e-9)
    key_x = np.round(xp if fold else x, 6)
    ux = np.unique(key_x[sel_r])
    nutkey = "nutMean" if "nutMean" in fld else "nut"
    profs = []
    for x0 in ux:
        m = sel_r & (key_x == x0)
        ywall = k if over_rib[m][0] else 0.0
        m = m & (y > ywall - 1e-9)
        # phase-fold: average all cells with the same (phase, y)
        ykey = np.round(y[m] - ywall, 7)
        uy, inv = np.unique(ykey, return_inverse=True)
        cnt = np.bincount(inv).astype(float)

        def avg(a):
            return np.bincount(inv, weights=a[m]) / cnt
        U = avg(fld["U"]); uv = avg(fld["uv"]); pp = avg(fld["p"]); nut = avg(fld[nutkey])
        if len(uy) < Y_IDX + 2:
            continue
        ycol = np.concatenate([[0.0], uy]); ucol = np.concatenate([[0.0], U])
        uvcol = np.concatenate([[0.0], uv]); nutcol = np.concatenate([[nut[0]], nut])
        dUdy = np.gradient(ucol, ycol)
        rf = resolved_fraction(uvcol, dUdy, nutcol, nu)
        yf, uf = ycol[1:1 + NFIT], ucol[1:1 + NFIT]
        dudn = float(np.sum(yf * uf) / np.sum(yf * yf))
        band = slice(1, Y_IDX + 1)
        fb = rf["f_res"][band]
        profs.append(dict(x=float(x0), wall="ribtop" if ywall > 0 else "floor", y=ycol, U=ucol,
                          uv=uvcol, uv_total=rf["uv_total"], nut=nutcol, f_res_col=rf["f_res"],
                          f_res_band=float(np.nanmedian(fb)) if np.isfinite(fb).any() else np.nan,
                          tau_w=nu * dudn, p_near=float(pp[0])))
    profs.sort(key=lambda d: d["x"])
    xs = np.array([d["x"] for d in profs]); pn = np.array([d["p_near"] for d in profs])
    dpdx = np.gradient(pn, xs)
    for i, d in enumerate(profs):
        d["dpdx"] = float(dpdx[i])
    return profs


def resample_profiles(profs, y_m, n_above=3.0, n_pts=31):
    """Deposit-independent matching height: put y[Y_IDX] exactly at y_m by
    interpolating every column onto y_grid = y_m * arange(n_pts)/Y_IDX."""
    grid = y_m * np.arange(n_pts) / Y_IDX
    out = []
    for d in profs:
        if d["y"][-1] < grid[-1]:
            grid_use = grid[grid <= d["y"][-1]]
        else:
            grid_use = grid
        if len(grid_use) <= Y_IDX:
            continue
        e = dict(d)
        e["y"] = grid_use
        e["U"] = np.interp(grid_use, d["y"], d["U"])
        e["uv"] = np.interp(grid_use, d["y"], d["uv"])
        e["uv_total"] = np.interp(grid_use, d["y"], d["uv_total"])
        out.append(e)
    return out


def score(profs, nu, rng, label):
    res = evaluate(profs, nu, Y_IDX=Y_IDX, closures=CLOSURES)
    fres = summarise(profs, Y_IDX=Y_IDX)
    valid = res["valid_mask"] & np.isfinite(res["standard_ml"])
    boot = block_bootstrap(res["tau_w_ref"][valid], res["standard_ml"][valid],
                           res["eps"][valid], rng)
    out = dict(label=label, n_stations=int(res["n_profiles"]),
               y_m=float(np.nanmedian(res["y_m"])),
               eps_median=float(res["eps_median"]), frac_eps_lt1=float(res["frac_eps_lt1"]),
               frac_eps_lt0p1=float(res["frac_eps_lt0p1"]),
               f_res_band_median=fres["f_res_band_median"],
               f_res_pooled_median=fres["f_res_pooled_median"],
               f_sep=float(np.mean(res["tau_w_ref"][valid] < 0)),
               station_block_bootstrap=boot)
    for c in CLOSURES:
        out[f"{c}_r2"] = float(res[f"{c}_r2"]); out[f"{c}_relRMS"] = float(res[f"{c}_relRMS"])
    out["closure_independent"] = bool(out["controlled_dns_r2"] < 0 and out["controlled_dns_total_r2"] < 0)
    arrays = dict(x=np.array([d["x"] for d in profs]), tau_w=res["tau_w_ref"],
                  pred_standard_ml=res["standard_ml"], pred_controlled_dns=res["controlled_dns"],
                  dpdx=res["dpdx"], eps=res["eps"], y_m=res["y_m"])
    return out, arrays


def rib_validation(fld, x, y, prov, profs):
    k, p, w = prov["k_rib_height"], prov["p_pitch"], prov["w_rib_width"]
    floor = [d for d in profs if d["wall"] == "floor"]
    xs = np.array([d["x"] for d in floor]); tw = np.array([d["tau_w"] for d in floor])
    x2 = (p + w) / 2.0                                   # rib trailing edge
    s = np.mod(xs - x2, p); o = np.argsort(s); ss, ts = s[o], tw[o]
    xr = np.nan
    for i in range(1, len(ss)):
        if ts[i - 1] < 0 <= ts[i]:
            xr = ss[i - 1] + (ss[i] - ss[i - 1]) * (-ts[i - 1]) / (ts[i] - ts[i - 1]); break
    rev = float(np.mean(tw < 0))
    # turbulence-sustainment audit by y-band (spanwise+time averaged fields)
    bands = {}
    for lo, hi in ((0.0, k), (k, 0.6), (0.6, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.0)):
        m = (y >= lo) & (y < hi)
        nutkey = "nutMean" if "nutMean" in fld else "nut"
        bands[f"{lo:.1f}-{hi:.1f}"] = dict(U=float(fld["U"][m].mean()), uu=float(fld["uu"][m].mean()),
                                          vv=float(fld["vv"][m].mean()), ww=float(fld["ww"][m].mean()),
                                          uv=float(fld["uv"][m].mean()),
                                          nut_over_nu=float(fld[nutkey][m].mean() / prov["nu"]))
    upper = (y >= 1.4) & (y < 1.9)
    return dict(x_reattach_over_k=float(xr / k) if np.isfinite(xr) else None,
                reattaches_on_floor=bool(np.isfinite(xr)), floor_reversed_fraction=rev,
                y_bands=bands,
                upper_half_turbulent=bool(fld["vv"][upper].mean() > 1e-4 and fld["ww"][upper].mean() > 1e-4),
                deposit_rib_les_dtype_upper_half_vv="5.7e-08 (laminarised; see MANIFEST)")


def forces_series(case_dir, names, t_start, t_end):
    """Mean pressure/viscous x-force per patch over [t_start,t_end] with block std."""
    out = {}
    for nm in names:
        fs = sorted(glob.glob(str(case_dir / "postProcessing" / nm / "*" / "force*.dat")))
        if not fs:
            continue
        rows = []
        for f in fs:
            for line in open(f):
                if line.startswith("#"):
                    continue
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) >= 7:
                    rows.append([float(v) for v in nums[:7]])
        if not rows:
            continue
        a = np.array(rows); a = a[np.argsort(a[:, 0])]
        m = (a[:, 0] >= t_start) & (a[:, 0] <= t_end)
        if m.sum() < 10:
            m = a[:, 0] >= a[-1, 0] - 0.2 * (a[-1, 0] - a[0, 0])
        t = a[m, 0]; fp = a[m, 1]; fv = a[m, 4]
        nb = max(2, min(20, len(t) // 50))
        blocks = np.array_split(np.arange(len(t)), nb)
        bp = np.array([fp[b].mean() for b in blocks]); bv = np.array([fv[b].mean() for b in blocks])
        out[nm] = dict(t_window=[float(t[0]), float(t[-1])], n_samples=int(len(t)), n_blocks=nb,
                       pressure_x=float(fp.mean()), pressure_x_block_sd=float(bp.std(ddof=1) / math.sqrt(nb)),
                       viscous_x=float(fv.mean()), viscous_x_block_sd=float(bv.std(ddof=1) / math.sqrt(nb)))
    return out


def log_gradp(case_dir, t_start, t_end):
    f = case_dir / "log_series.npy"
    if not f.exists():
        return None
    s = np.load(f)
    m = (s[:, 0] >= t_start) & (s[:, 0] <= t_end) & np.isfinite(s[:, 4])
    if m.sum() == 0:
        return None
    return dict(gradP_mean=float(s[m, 4].mean()), gradP_sd=float(s[m, 4].std()),
                dt_mean=float(np.nanmean(s[m, 1])), Co_max_mean=float(np.nanmean(s[m, 3])),
                n_steps_total=int(len(s)), t_end=float(s[-1, 0]))


# --------------------------------------------------------------------------
def harvest_rib(case_dir, prov, man, rng, allow_pilot):
    files = sorted(glob.glob(str(case_dir / "span_t*.npz")))
    t0 = prov["averaging"]["avg_start"]
    t_end = prov["averaging"]["end_time"]
    wins = load_windows(files, t0)
    wins = [w for w in wins if w["t"] > t0 + 1e-6]
    if not wins:
        return dict(status="PENDING", reason="no averaged window")
    final = wins[-1]
    converged = final["t"] >= t_end - 1e-3
    if not converged and not allow_pilot:
        return dict(status="INFLIGHT", reason=f"latest window t={final['t']} < endTime {t_end}",
                    latest_time=final["t"])
    x, y = final["data"]["x"], final["data"]["y"]
    k = prov["k_rib_height"]
    windows = {f"cum_{final['t']:.4g}": final["data"]}
    if len(wins) >= 2:
        prev = wins[-2]
        windows[f"cum_{prev['t']:.4g}"] = prev["data"]
        split_t = prov["averaging"].get("split_time", 100.0)
        ws = min(wins[:-1], key=lambda w: abs(w["t"] - split_t))
        if ws["t"] < final["t"] - 1e-6:
            windows[f"disj_{t0:.4g}-{ws['t']:.4g}"] = ws["data"]
            windows[f"disj_{ws['t']:.4g}-{final['t']:.4g}"] = disjoint_window(
                ws, final, t0, RIB_MEAN_KEYS, RIB_PRIME)
    nu = prov["nu"]
    out = dict(status="OK", kind="rib", case=case_dir.name, tag=prov["geometry"],
               p_over_k=prov["p_over_k"], pitch_over_delta=prov["lambda_over_delta"],
               n_ribs=prov["n_ribs"], Lx=prov["Lx_box"], Lz=prov["Lz_span"], Re_delta=prov["Re_delta"],
               n_cells=int(man["n_cells"]), refine=prov["mesh"]["refine"], slurm_job_id=man.get("slurm_job_id"),
               windows={}, arrays={})
    global Y_M_OVER_K_RIB
    for wname, fld in windows.items():
        profs = rib_profiles(fld, x, y, prov, fold=True)
        sc, arr = score(profs, nu, rng, wname)
        ym_k = float(np.median([d["y"][Y_IDX] for d in profs])) / k
        sc["y_m_over_k_index_convention"] = ym_k
        if Y_M_OVER_K_RIB is None and prov["mesh"]["refine"] == 1.0:
            Y_M_OVER_K_RIB = ym_k
        # replicate (phase-block) verdicts: one per rib pitch
        reps = []
        for i in range(int(prov["n_ribs"])):
            pr = rib_profiles(fld, x, y, prov, fold=False, rib_index=i)
            if len(pr) >= Y_IDX + 2:
                r = evaluate(pr, nu, Y_IDX=Y_IDX, closures=("standard_ml",))
                reps.append(dict(rib=i, standard_ml_r2=float(r["standard_ml_r2"]),
                                 eps_median=float(r["eps_median"])))
        sc["rib_replicates"] = reps
        sc["rib_replicate_r2"] = t_interval([r["standard_ml_r2"] for r in reps])
        sc["rib_replicate_eps_median"] = t_interval([r["eps_median"] for r in reps])
        # matched physical matching height (grid-independent convention)
        if Y_M_OVER_K_RIB is not None:
            rp = resample_profiles(profs, Y_M_OVER_K_RIB * k)
            sc2, _ = score(rp, nu, rng, wname + "_ym_matched")
            sc["matched_ym"] = dict(y_m_over_k=Y_M_OVER_K_RIB, **{kk: sc2[kk] for kk in sc2
                                    if kk.endswith("_r2") or kk in ("eps_median", "frac_eps_lt0p1",
                                                                    "station_block_bootstrap")})
        sc["validation"] = rib_validation(fld, x, y, prov, profs)
        out["windows"][wname] = sc
        out["arrays"][wname] = arr
    # forces / drag / momentum closure over the final cumulative window
    fr = forces_series(case_dir, ("forcesBottom", "forcesTop"), t0, final["t"])
    gp = log_gradp(case_dir, t0, final["t"])
    A_plan = prov["Lx_box"] * prov["Lz_span"]
    V_fluid = prov["Lx_box"] * prov["H_channel_height"] * prov["Lz_span"] - prov["n_ribs"] * k * prov["w_rib_width"] * prov["Lz_span"]
    drag = dict(forces=fr, log=gp)
    if "forcesBottom" in fr:
        fb = fr["forcesBottom"]; ft = fr.get("forcesTop", {})
        # SIGNED sums.  The former abs-per-component sum silently reversed any
        # opposing term; on the wide-pitch k-type rib the plan-integrated
        # viscous force on the ribbed wall is genuinely negative (-0.0268,
        # -4.57 % of the signed total), so the abs form inflated u_tau by
        # 4.5 % and manufactured a -7.0 % momentum-closure residual where the
        # signed balance closes to +1.4e-5.  The negative sign
        # is the physical result, not a defect, and is retained.
        F_b = fb["pressure_x"] + fb["viscous_x"]
        F_t = ft.get("viscous_x", 0.0)
        u_tau = math.sqrt(F_b / A_plan)
        drag.update(F_bottom_total=F_b, form_drag_fraction=fb["pressure_x"] / F_b,
                    viscous_force_fraction=fb["viscous_x"] / F_b,
                    viscous_force_opposes_drive=bool(fb["viscous_x"] * F_b < 0.0),
                    u_tau_bottom=u_tau, Re_tau_bottom=u_tau * prov["delta_half_height"] / nu,
                    u_tau_top=math.sqrt(F_t / A_plan) if F_t > 0 else None,
                    Re_tau_top=(math.sqrt(F_t / A_plan) / nu) if F_t > 0 else None,
                    dx_plus=prov["mesh"]["dx_cavity"] * u_tau / nu, dz_plus=prov["mesh"]["dz"] * u_tau / nu)
        if gp:
            drag["momentum_closure"] = dict(driving=gp["gradP_mean"] * V_fluid, wall_total=F_b + F_t,
                                            relative_residual=(gp["gradP_mean"] * V_fluid - F_b - F_t) / (F_b + F_t))
    out["drag"] = drag
    # Reynolds-number definitions.  meanVelocityForce holds the FLUID-volume
    # average U_Vf = Ubar = 1 (OpenFOAM-10 meanVelocityForce.C divides by
    # set_.V()).  Flow rate Q = U_Vf V_fluid / Lx, so a flow-rate-based bulk
    # Reynolds number Q/(2 Lz nu) (= U_b delta/nu with U_b over the full 2 delta
    # cross-section, identical to U_b,crest (delta - k/2)/nu) is lower by the
    # fluid-volume fraction 1 - k w/(p H).  Both are reported; the measured
    # flow rate from the spanwise-averaged field is the cross-check.
    f_vol = 1.0 - k * prov["w_rib_width"] / (prov["p_pitch"] * prov["H_channel_height"])
    fld = final["data"]
    q_cols = []
    for x0 in np.unique(np.round(x, 6))[::max(1, len(np.unique(np.round(x, 6))) // 40)]:
        m = np.abs(x - x0) < 1e-6
        yy, uu = y[m], fld["U"][m]
        o = np.argsort(yy)
        ywall = 0.0 if yy.min() < k else k
        yint = np.concatenate([[ywall], yy[o], [prov["H_channel_height"]]])
        uint = np.concatenate([[0.0], uu[o], [0.0]])
        q_cols.append(float(np.trapz(uint, yint)))
    Q_over_Lz = float(np.mean(q_cols))
    out["reynolds"] = dict(
        held_quantity="fluid-volume-averaged U (meanVelocityForce Ubar = 1)",
        Re_fluid_volume_average=prov["Re_delta"],
        fluid_volume_fraction=f_vol,
        Re_flow_rate_nominal=prov["Re_delta"] * f_vol,
        Re_flow_rate_measured=Q_over_Lz / (2.0 * nu),
        Q_over_Lz_measured=Q_over_Lz, Q_over_Lz_nominal=f_vol * prov["H_channel_height"],
        note="Re_Q = Q/(2 Lz nu) is invariant to the choice of reference height; "
             "deposited rib_les_dtype used the same convention (Re_Q = 4060).")
    out["converged"] = bool(converged)
    return out


# --------------------------------------------------------------------------
# CUBE
# --------------------------------------------------------------------------
CUBE_MEAN_KEYS = ("UMean", "pMean", "nutMean", "nut")


def cube_disjoint(w_a, w_b, t0):
    Ta, Tb = w_a["t"] - t0, w_b["t"] - t0
    Tw = Tb - Ta
    A, B = w_a["data"], w_b["data"]
    out = {"C": B["C"]}
    for k in CUBE_MEAN_KEYS:
        if k in A and k in B:
            out[k] = (Tb * B[k].astype(float) - Ta * A[k].astype(float)) / Tw
    if "UPrime2Mean" in A:
        Ua, Ub = A["UMean"].astype(float), B["UMean"].astype(float)
        idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        prod_a = np.stack([Ua[:, i] * Ua[:, j] for i, j in idx], 1)
        prod_b = np.stack([Ub[:, i] * Ub[:, j] for i, j in idx], 1)
        uu_w = (Tb * (B["UPrime2Mean"].astype(float) + prod_b) - Ta * (A["UPrime2Mean"].astype(float) + prod_a)) / Tw
        Uw = out["UMean"]
        out["UPrime2Mean"] = uu_w - np.stack([Uw[:, i] * Uw[:, j] for i, j in idx], 1)
    return out


def cube_columns(fld, prov, fold=True, cell_index=None, wall="floor"):
    """Wall-normal columns above the floor (wall='floor') or above the cube tops
    (wall='cubetop'); fold=True phase-averages over the periodic cube cells."""
    C = fld["C"].astype(float); U = fld["UMean"].astype(float); p = fld["pMean"].astype(float)
    R = fld["UPrime2Mean"].astype(float); nut = (fld["nutMean"] if "nutMean" in fld else fld["nut"]).astype(float)
    h, L, P, layout = prov["h"], prov["L_box"], prov["pitch"], prov["layout"]
    x, y, z = C[:, 0], C[:, 1], C[:, 2]
    # periodic cell fold
    if layout == "staggered":
        px, pz = 2 * P, P           # rows alternate every P in x; z-period P
        # glide symmetry (x + P, z + P/2) maps row A onto row B: fold with it
        xg = np.mod(x, 2 * P); zg = np.where(xg >= P, np.mod(z + P / 2, P), np.mod(z, P)); xg = np.mod(xg, P)
        fx, fz = xg, zg
        n_cells = int(round((L / P) * (L / P)))
        cell_id = (np.floor(x / P) * (L / P) + np.floor(np.where(np.mod(x, 2 * P) >= P, z + P / 2, z) % L / P)).astype(int)
    else:
        fx, fz = np.mod(x, P), np.mod(z, P)
        n_cells = int(round((L / P) * (L / P)))
        cell_id = (np.floor(x / P) * (L / P) + np.floor(z / P)).astype(int)
    if cell_index is not None:
        sel = cell_id == cell_index
    else:
        sel = np.ones(len(x), bool)
    keyx = np.round(fx if fold else x, 6); keyz = np.round(fz if fold else z, 6)
    cols, inv = np.unique(np.stack([keyx[sel], keyz[sel]], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    ys = y[sel]; Us = U[sel, 0]; uvs = R[sel, 1]; ps = p[sel]; nuts = nut[sel]
    ymin = np.full(len(cols), np.inf); np.minimum.at(ymin, inv, ys)
    profs = []
    nu = prov["nu"]
    for ci in range(len(cols)):
        is_floor = ymin[ci] < 0.5 * h
        if (wall == "floor") != is_floor:
            continue
        ywall = 0.0 if is_floor else h
        m = (inv == ci) & (ys > ywall - 1e-9)
        ykey = np.round(ys[m] - ywall, 7)
        uy, inv2 = np.unique(ykey, return_inverse=True)
        cnt = np.bincount(inv2).astype(float)
        Uc = np.bincount(inv2, weights=Us[m]) / cnt
        uvc = np.bincount(inv2, weights=uvs[m]) / cnt
        pc = np.bincount(inv2, weights=ps[m]) / cnt
        nc = np.bincount(inv2, weights=nuts[m]) / cnt
        if len(uy) < Y_IDX + 2:
            continue
        ycol = np.concatenate([[0.0], uy]); ucol = np.concatenate([[0.0], Uc])
        uvcol = np.concatenate([[0.0], uvc]); nutcol = np.concatenate([[nc[0]], nc])
        dUdy = np.gradient(ucol, ycol)
        rf = resolved_fraction(uvcol, dUdy, nutcol, nu)
        yf, uf = ycol[1:1 + NFIT], ucol[1:1 + NFIT]
        dudn = float(np.sum(yf * uf) / np.sum(yf * yf))
        fb = rf["f_res"][1:Y_IDX + 1]
        profs.append(dict(x=float(cols[ci, 0]), z=float(cols[ci, 1]), wall=wall, y=ycol, U=ucol,
                          uv=uvcol, uv_total=rf["uv_total"], nut=nutcol, f_res_col=rf["f_res"],
                          f_res_band=float(np.nanmedian(fb)) if np.isfinite(fb).any() else np.nan,
                          tau_w=nu * dudn, p_near=float(pc[0])))
    # dp/dx along x within each z-lane, over contiguous floor segments
    profs.sort(key=lambda d: (d["z"], d["x"]))
    zs = np.array([d["z"] for d in profs]); xs = np.array([d["x"] for d in profs])
    dx = np.median(np.diff(np.unique(xs))) if len(np.unique(xs)) > 1 else 1.0
    for zl in np.unique(zs):
        idx = np.where(zs == zl)[0]
        seg_start = 0
        xi = xs[idx]
        for j in range(1, len(idx) + 1):
            if j == len(idx) or xi[j] - xi[j - 1] > 1.5 * dx:
                seg = idx[seg_start:j]
                if len(seg) >= 3:
                    g = np.gradient(np.array([profs[s]["p_near"] for s in seg]), xs[seg])
                else:
                    g = np.full(len(seg), np.nan)
                for s, gg in zip(seg, g):
                    profs[s]["dpdx"] = float(gg)
                seg_start = j
    profs = [d for d in profs if np.isfinite(d.get("dpdx", np.nan))]
    return profs, n_cells


def cube_mean_profile(fld, prov):
    C = fld["C"].astype(float); U = fld["UMean"].astype(float); R = fld["UPrime2Mean"].astype(float)
    y = np.round(C[:, 1], 6)
    uy, inv = np.unique(y, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    Um = np.bincount(inv, weights=U[:, 0]) / cnt           # intrinsic (fluid) average
    vv = np.bincount(inv, weights=R[:, 3]) / cnt; ww = np.bincount(inv, weights=R[:, 5]) / cnt
    uv = np.bincount(inv, weights=R[:, 1]) / cnt
    nfluid = cnt / cnt.max()
    h, ut = prov["h"], prov["u_tau"]
    # log-law fit above 2h: U/u_tau = (1/0.41) ln((y-d)/z0); scan d, fit z0
    m = (uy > 2 * h) & (uy < 3.5 * h)
    best = None
    for d in np.linspace(0.0, 1.0 * h, 101):
        yy = uy[m] - d
        if np.any(yy <= 0):
            continue
        # ln z0 = ln(y-d) - 0.41 U
        lz = np.log(yy) - 0.41 * Um[m] / ut
        resid = np.std(lz)
        if best is None or resid < best[0]:
            best = (resid, d, math.exp(lz.mean()))
    return dict(y=uy, U_over_utau=Um / ut, vv=vv, ww=ww, uv=uv, fluid_fraction=nfluid,
                loglaw=dict(d_over_h=best[1] / h, z0_over_h=best[2] / h, fit_rms_lnz0=best[0]) if best else None,
                U_lid_over_utau=float(Um[-1] / ut),
                U_bulk_over_utau=float(np.sum(Um * cnt) / np.sum(cnt) / ut))


def harvest_cube(case_dir, prov, man, rng, allow_pilot):
    files = sorted(glob.glob(str(case_dir / "field_t*.npz")))
    t0 = prov["averaging"]["avg_start"]; t_end = prov["averaging"]["end_time"]
    wins = load_windows(files, t0)
    wins = [w for w in wins if w["t"] > t0 + 1e-6]
    if not wins:
        return dict(status="PENDING", reason="no averaged window")
    final = wins[-1]
    converged = final["t"] >= t_end - 1e-3
    if not converged and not allow_pilot:
        return dict(status="INFLIGHT", reason=f"latest window t={final['t']} < endTime {t_end}",
                    latest_time=final["t"])
    windows = {f"cum_{final['t']:.4g}": final["data"]}
    if len(wins) >= 2:
        prev = wins[-2]; windows[f"cum_{prev['t']:.4g}"] = prev["data"]
        split_t = t0 + 0.5 * (t_end - t0)
        ws = min(wins[:-1], key=lambda w: abs(w["t"] - split_t))
        if ws["t"] < final["t"] - 1e-6:
            windows[f"disj_{t0:.4g}-{ws['t']:.4g}"] = ws["data"]
            windows[f"disj_{ws['t']:.4g}-{final['t']:.4g}"] = cube_disjoint(ws, final, t0)
    nu, h, ut = prov["nu"], prov["h"], prov["u_tau"]
    out = dict(status="OK", kind="cube", case=case_dir.name, layout=prov["layout"], tag=prov["geometry"],
               lambda_p=prov["lambda_p"], pitch_over_h=prov["pitch"], L_box=prov["L_box"],
               Re_tau_H=prov["Re_tau_H"], n_cells=int(man["n_cells"]), refine=prov["mesh"]["refine"],
               slurm_job_id=man.get("slurm_job_id"), wall_units=prov["wall_units"], windows={}, arrays={})
    for wname, fld in windows.items():
        sc_all = dict(label=wname)
        for wall in ("floor", "cubetop"):
            profs, n_cells = cube_columns(fld, prov, fold=True, wall=wall)
            if len(profs) < 5:
                continue
            sc, arr = score(profs, nu, rng, f"{wname}_{wall}")
            sc["y_m_over_h_index_convention"] = float(np.median([d["y"][Y_IDX] for d in profs])) / h
            sc["y_m_plus_index_convention"] = sc["y_m_over_h_index_convention"] * h * ut / nu
            reps = []
            for i in range(n_cells):
                pr, _ = cube_columns(fld, prov, fold=False, cell_index=i, wall=wall)
                if len(pr) >= 5:
                    r = evaluate(pr, nu, Y_IDX=Y_IDX, closures=("standard_ml",))
                    reps.append(dict(cell=i, standard_ml_r2=float(r["standard_ml_r2"]), eps_median=float(r["eps_median"])))
            sc["cell_replicates"] = reps
            sc["cell_replicate_r2"] = t_interval([r["standard_ml_r2"] for r in reps])
            sc["cell_replicate_eps_median"] = t_interval([r["eps_median"] for r in reps])
            ym_rib = (Y_M_OVER_K_RIB if Y_M_OVER_K_RIB else 0.146) * h
            for nm, ym in (("matched_ym_rib", ym_rib), ("ym_yplus50", 50.0 * nu / ut)):
                rp = resample_profiles(profs, ym)
                if len(rp) >= 5:
                    s2, _ = score(rp, nu, rng, f"{wname}_{wall}_{nm}")
                    sc[nm] = dict(y_m_over_h=ym / h, y_m_plus=ym * ut / nu,
                                  **{kk: s2[kk] for kk in s2 if kk.endswith("_r2") or kk in
                                     ("eps_median", "frac_eps_lt0p1", "station_block_bootstrap")})
            sc_all[wall] = sc
            out["arrays"][f"{wname}_{wall}"] = arr
        sc_all["mean_profile"] = cube_mean_profile(fld, prov)
        mp = sc_all["mean_profile"]
        upper = (mp["y"] > 2 * h)
        sc_all["upper_region_turbulent"] = bool(mp["vv"][upper].mean() > 0.05 * ut ** 2)
        out["windows"][wname] = sc_all
    fr = forces_series(case_dir, ("forcesFloor", "forcesCube"), t0, final["t"])
    V = prov["mesh"]["V_fluid"]; A = prov["mesh"]["A_plan"]
    drag = dict(forces=fr, driving_force=prov["body_force_gx"] * V, u_tau_nominal=ut)
    if "forcesCube" in fr and "forcesFloor" in fr:
        # SIGNED sums, as for the ribs above: the staggered array's plan-mean
        # floor viscous force is -0.0324 (block SD 0.0199), a small thrust.
        Fc = fr["forcesCube"]["pressure_x"] + fr["forcesCube"]["viscous_x"]
        Ff = fr["forcesFloor"]["viscous_x"]
        drag.update(F_cube_total=Fc, F_floor=Ff, form_drag_fraction=fr["forcesCube"]["pressure_x"] / (Fc + Ff),
                    floor_viscous_fraction=Ff / (Fc + Ff),
                    floor_viscous_opposes_drive=bool(Ff * (Fc + Ff) < 0.0),
                    u_tau_measured=math.sqrt((Fc + Ff) / A),
                    momentum_closure=dict(driving=prov["body_force_gx"] * V, wall_total=Fc + Ff,
                                          relative_residual=(prov["body_force_gx"] * V - Fc - Ff) / (Fc + Ff)))
    out["drag"] = drag
    out["log"] = log_gradp(case_dir, t0, final["t"])
    out["converged"] = bool(converged)
    return out


# --------------------------------------------------------------------------
def grid_check(cases):
    """Pair G1/G0 members of the same geometry: verdict invariance."""
    checks = {}
    by_geom = {}
    for c in cases.values():
        if c.get("status") != "OK":
            continue
        key = re.sub(r"_G[01]$", "", c["case"])
        by_geom.setdefault(key, {})["G1" if c["refine"] == 1.0 else "G0"] = c
    for key, g in by_geom.items():
        if "G1" in g and "G0" in g:
            w1 = g["G1"]["windows"]; w0 = g["G0"]["windows"]
            fin1 = sorted(k for k in w1 if k.startswith("cum_"))[-1]
            fin0 = sorted(k for k in w0 if k.startswith("cum_"))[-1]
            if g["G1"]["kind"] == "rib":
                a, b = w1[fin1], w0[fin0]
                ra, rb = a.get("matched_ym", a), b.get("matched_ym", b)
            else:
                a, b = w1[fin1]["floor"], w0[fin0]["floor"]
                ra, rb = a.get("matched_ym_rib", a), b.get("matched_ym_rib", b)
            checks[key] = dict(
                G1_cells=g["G1"]["n_cells"], G0_cells=g["G0"]["n_cells"],
                r2_G1=a["standard_ml_r2"], r2_G0=b["standard_ml_r2"],
                r2_G1_matched_ym=ra["standard_ml_r2"], r2_G0_matched_ym=rb["standard_ml_r2"],
                eps_median_G1=a["eps_median"], eps_median_G0=b["eps_median"],
                ci_G1=ra["station_block_bootstrap"]["r2_ci95"], ci_G0=rb["station_block_bootstrap"]["r2_ci95"],
                verdict_invariant=bool(np.sign(ra["standard_ml_r2"]) == np.sign(rb["standard_ml_r2"])),
                ci_overlap=bool(max(ra["station_block_bootstrap"]["r2_ci95"][0], rb["station_block_bootstrap"]["r2_ci95"][0])
                                <= min(ra["station_block_bootstrap"]["r2_ci95"][1], rb["station_block_bootstrap"]["r2_ci95"][1])))
    return checks


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items() if k != "arrays"}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260823")
    ap.add_argument("--allow-pilot", action="store_true", help="also harvest the 1-time-unit pipeline pilots (diagnostic only)")
    ap.add_argument("--deposits", default=str(DEPOSITS))
    ap.add_argument("--out-stem", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    dep = Path(a.deposits)
    cases = {}
    order = sorted(dep.glob("*/MANIFEST.json"))
    # production d-type first so the matched y_m/k is set from it
    order.sort(key=lambda p: (0 if "dtype" in p.parent.name and "G1" in p.parent.name else 1, str(p)))
    for mf in order:
        cdir = mf.parent
        if cdir.name.startswith("r24_pilot") and not a.allow_pilot:
            continue
        man = json.loads(mf.read_text()); prov = json.loads((cdir / "PROVENANCE.json").read_text())
        print("harvesting", cdir.name, man["kind"], "windows", man["time_dirs"])
        res = harvest_rib(cdir, prov, man, rng, a.allow_pilot) if man["kind"] == "rib" \
            else harvest_cube(cdir, prov, man, rng, a.allow_pilot)
        res["source_manifest_sha256"] = sha256(mf)
        res["source_provenance_sha256"] = sha256(cdir / "PROVENANCE.json")
        cases[cdir.name] = res
        if res.get("status") == "OK":
            for wname, w in res["windows"].items():
                if res["kind"] == "rib":
                    print(f"  {wname:>14s}: R2(std)={w['standard_ml_r2']:+.3f} CI{w['station_block_bootstrap']['r2_ci95']} "
                          f"R2(dns)={w['controlled_dns_r2']:+.3f} eps_med={w['eps_median']:.3f} f_res={w['f_res_band_median']:.3f} "
                          f"x_r/k={w['validation']['x_reattach_over_k']} upper_turb={w['validation']['upper_half_turbulent']}")
                else:
                    f = w.get("floor")
                    if f:
                        print(f"  {wname:>14s} floor: R2(std)={f['standard_ml_r2']:+.3f} CI{f['station_block_bootstrap']['r2_ci95']} "
                              f"eps_med={f['eps_median']:.3f} f_res={f['f_res_band_median']:.3f} "
                              f"loglaw d/h={w['mean_profile']['loglaw']['d_over_h']:.2f} z0/h={w['mean_profile']['loglaw']['z0_over_h']:.3f}")
        else:
            print("  ->", res["status"], res.get("reason"))
    summary = dict(row="R2-4 / M20", date=a.date, campaign="archer2_campaign_20260823/R2-4_M20",
                   fidelity="OpenFOAM-10 wall-resolved LES (WALE), NOT DNS; numerics of codes/openfoam/rib_les_dtype",
                   instrument="codes/analysis/rib_eps_ode.evaluate (Y_IDX=10, deposit conventions)",
                   bootstrap=dict(seed=SEED, n_boot=N_BOOT, kind="circular moving-block over stations, block ceil(sqrt(N))"),
                   matched_y_m_over_k=Y_M_OVER_K_RIB,
                   generators_sha256={f: sha256(ROOT / "codes" / "openfoam" / f) for f in
                                      ("make_rib_les_multi_case.py", "make_cube_les_case.py", "make_rib_les_case.py",
                                       "r24_les_reduce.py", "les_seed_fast.py", "of_ascii_fast.py")},
                   driver_sha256=sha256(ROOT / "jobs" / "r24_les_driver.sh"),
                   cases=cases, grid_check=grid_check(cases))
    n_ok = sum(1 for c in cases.values() if c.get("status") == "OK")
    summary["n_cases_ok"] = n_ok
    stem = a.out_stem or f"r2_4_m20_les_{a.date}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    arrays = {}
    for cname, c in cases.items():
        for wname, arr in c.get("arrays", {}).items():
            for k, v in arr.items():
                arrays[f"{cname}__{wname}__{k}"] = np.asarray(v, float)
    np.savez(RESULTS / f"{stem}.npz", **arrays, date=a.date, n_cases_ok=n_ok)
    print(f"\nsaved -> {RESULTS / stem}.json/.npz  (cases OK: {n_ok}/{len(cases)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
