#!/usr/bin/env python3
"""Agent V addendum 2 -- does the M2 / R2-m2 closure depend on the withdrawn hill reference?

signed_wall_error_metrics_m2.npz scores 18 cases.  For `periodic_hills_case_1p0` the
PREDICTION is the pressure-gradient ODE evaluated at Y_IDX=10 from the archive's own
RESOLVED velocity (clean), but the TRUTH is `tau_w` read from
codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz -- the withdrawn
4-point through-origin LINEAR fit of the streamwise u against the vertical offset,
with no tangent correction.  Every M2 statistic for that one case is normalised by the
weighted RMS of that truth, so all of them are exposed.

This script re-evaluates the identical metric operator for that case with the truth
replaced by (B) the MGLET DNS deposited tau_w and (C) my validated through-origin cubic
on the same archive, leaving the prediction, the quadrature and the formulas untouched.
`predict_tau_w` and the metric formulas are re-implemented here rather than imported from
the producer, except the ODE itself which is imported so the prediction is bit-identical.

Out: work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/m2_reference_contamination.json
Read-only on all inputs; no simulation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/m2_reference_contamination.json"
XIAO = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
MGLET = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
M2 = ROOT / "codes/results/signed_wall_error_metrics_m2.npz"
sys.path.insert(0, str(ROOT / "codes/vendor/universal_wall_function/codes/analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

LX, NU, Y_IDX = 9.0, 1.0 / 5600.0, 10
N_BOOT, BOOT_SEED = 2000, 20260821


def ercoftac_hill(xh):
    xm = 28.0 * (xh % LX)
    if xm > 28.0 * LX / 2.0:
        xm = 28.0 * LX - xm
    if xm < 9.0:
        h = min(28.0, 28.0 + 6.775070969851e-3*xm**2 - 2.124527775800e-3*xm**3)
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
    s = np.array([(ercoftac_hill(v + d) - ercoftac_hill(v - d)) / (2.0 * d) for v in np.atleast_1d(x)])
    m = np.sqrt(1.0 + s**2)
    return 1.0 / m, s / m


def poly_origin_slope(n, u, deg):
    A = np.vstack([np.asarray(n, float) ** (k + 1) for k in range(deg)]).T
    return float(np.linalg.lstsq(A, np.asarray(u, float), rcond=None)[0][0])


def wrap_interp(xp, yp, t):
    o = np.argsort(np.mod(np.asarray(xp, float), 1.0))
    a = np.mod(np.asarray(xp, float), 1.0)[o]
    b = np.asarray(yp, float)[o]
    return np.interp(np.mod(np.asarray(t, float), 1.0), np.r_[a-1, a, a+1], np.r_[b, b, b])


def trapezoid_weights(x):
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w / np.sum(w)


def negative_intervals(x, v):
    """Contiguous runs where v<0, with linearly interpolated end points."""
    neg = v < 0.0
    out = []
    i = 0
    while i < len(neg):
        if not neg[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(neg) and neg[j + 1]:
            j += 1
        if i == 0:
            a, da = x[0], 0.0
        else:
            t = v[i-1] / (v[i-1] - v[i]); a = x[i-1] + t * (x[i] - x[i-1]); da = x[i] - x[i-1]
        if j == len(neg) - 1:
            b, db = x[-1], 0.0
        else:
            t = v[j] / (v[j] - v[j+1]); b = x[j] + t * (x[j+1] - x[j]); db = x[j+1] - x[j]
        out.append((a, b, da, db))
        i = j + 1
    return out


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def topology(x, ref, pred):
    span = float(x[-1] - x[0])
    ri, pi = negative_intervals(x, ref), negative_intervals(x, pred)
    dx = np.linspace(x[0], x[-1], max(4097, 16 * x.size + 1))
    sym = float(np.mean((np.interp(dx, x, ref) < 0) != (np.interp(dx, x, pred) < 0)))
    o = {"n_ref_components": len(ri), "n_pred_components": len(pi),
         "separated_set_symmetric_difference": sym, "event_missed": False,
         "x_separation_ref": math.nan, "x_reattachment_ref": math.nan,
         "separation_error_over_span": math.nan, "reattachment_error_over_span": math.nan}
    if not ri:
        return o
    r = max(ri, key=lambda t: t[1] - t[0])
    o["x_separation_ref"], o["x_reattachment_ref"] = r[0], r[1]
    if not pi:
        o["event_missed"] = True; return o
    p = max(pi, key=lambda t: (overlap(r, t), t[1] - t[0]))
    if overlap(r, p) <= 0.0:
        o["event_missed"] = True; return o
    o["separation_error_over_span"] = (p[0] - r[0]) / span
    o["reattachment_error_over_span"] = (p[1] - r[1]) / span
    return o


def metrics(x, truth, pred, u_edge, seed=BOOT_SEED):
    w = trapezoid_weights(x)
    tau_scale = float(np.sqrt(np.sum(w * truth**2)))
    cf_ref, cf_pred = 2.0 * truth / u_edge**2, 2.0 * pred / u_edge**2
    cf_scale = float(np.sqrt(np.sum(w * cf_ref**2)))
    sgn = (pred - truth) / tau_scale
    a = np.abs(sgn)
    acf = np.abs((cf_pred - cf_ref) / cf_scale)
    var = np.sum((truth - truth.mean())**2)
    dd = float(np.sum(w * np.abs(truth)))
    m = {"n_stations": int(x.size), "tau_rms_scale": tau_scale, "cf_rms_scale": cf_scale,
         "r2_descriptive": float(1.0 - np.sum((pred - truth)**2) / var) if var > 0 else math.nan,
         "relrms_tau": float(np.sqrt(np.sum(w * (pred - truth)**2)) / tau_scale),
         "station_signed_median": float(np.median(sgn)),
         "station_abs_p50": float(np.median(a)), "station_abs_p95": float(np.percentile(a, 95)),
         "station_abs_max": float(np.max(a)),
         "cf_station_abs_p50": float(np.median(acf)), "cf_station_abs_p95": float(np.percentile(acf, 95)),
         "cf_station_abs_max": float(np.max(acf)),
         "viscous_drag_signed_error": float(np.sum(w * (pred - truth)) / dd),
         "station_sign_mismatch_fraction": float(np.sum(w * (np.signbit(pred) != np.signbit(truth)))),
         **topology(x, truth, pred)}
    # moving-block interval on p50/p95, same block rule and seed family as the producer
    rng = np.random.default_rng(seed)
    n = x.size
    block = max(2, int(round(n ** (1.0 / 3.0))))
    nb = int(np.ceil(n / block))
    p50 = np.empty(N_BOOT); p95 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        st = rng.integers(0, n, size=nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in st])[:n]
        p50[b] = np.median(a[idx]); p95[b] = np.percentile(a[idx], 95)
    m["station_abs_p50_ci"] = [float(np.percentile(p50, 2.5)), float(np.percentile(p50, 97.5))]
    m["station_abs_p95_ci"] = [float(np.percentile(p95, 2.5)), float(np.percentile(p95, 97.5))]
    return m


def main() -> int:
    d = np.load(XIAO, allow_pickle=False)
    x = np.asarray(d["x"], float); y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float); V = np.asarray(d["V"], float)
    dp = np.asarray(d["dp_dx"], float); tau_legacy = np.asarray(d["tau_w"], float)
    nu_arr = np.asarray(d["nu"], float)
    tx, ty = tangent(x)

    pred = np.full(x.size, np.nan)
    u_edge = np.nanmax(np.abs(U), axis=1)
    for i in range(x.size):
        ym, um = float(y[i, Y_IDX]), float(U[i, Y_IDX])
        if np.isfinite(ym) and ym > 0 and np.isfinite(um):
            pred[i] = predict_tau_w(um, ym, float(dp[i]), float(nu_arr[i]))

    tau_cub = np.empty(x.size)
    for i in range(x.size):
        yy, uu, vv = y[i], U[i], V[i]
        msk = np.isfinite(yy) & np.isfinite(uu) & np.isfinite(vv)
        yy, uu, vv = yy[msk], uu[msk], vv[msk]
        ut = uu[1:7] * tx[i] + vv[1:7] * ty[i]
        tau_cub[i] = NU * poly_origin_slope(yy[1:7], ut, 3) / tx[i]
    mg = np.loadtxt(MGLET)[:-2]
    tau_mg = wrap_interp(mg[:, 0] / LX, mg[:, 1], np.mod((x - x.min()) / LX, 1.0))

    ok = np.isfinite(x) & np.isfinite(pred) & np.isfinite(u_edge) & (u_edge > 0)
    o = np.argsort(x[ok], kind="mergesort")
    xs, pr, ue = x[ok][o], pred[ok][o], u_edge[ok][o]

    truths = {"A_withdrawn_legacy_as_published": tau_legacy[ok][o],
              "B_mglet": tau_mg[ok][o],
              "C_repaired_cubic": tau_cub[ok][o]}
    res = {"schema": "m2-reference-contamination-v1", "agent": "V",
           "case": "periodic_hills_case_1p0 (1 of 18 in signed_wall_error_metrics_m2.npz)",
           "prediction": "pressure-gradient ODE at Y_IDX=10 from the archive's resolved velocity -- UNCHANGED, clean",
           "metrics": {k: metrics(xs, t, pr, ue) for k, t in truths.items()}}

    pub = np.load(M2, allow_pickle=True)
    names = np.asarray(pub["names"]).astype(str)
    i = int(list(names).index("periodic_hills_case_1p0"))
    res["published_row"] = {k: float(pub[k][i]) for k in
                            ("r2_descriptive", "relrms_tau", "station_abs_p50", "station_abs_p95",
                             "station_abs_max", "viscous_drag_signed_error",
                             "separated_set_symmetric_difference", "separation_error_over_span",
                             "reattachment_error_over_span", "cf_rms_scale", "tau_rms_scale",
                             "station_sign_mismatch_fraction")}
    res["published_row"]["event_missed"] = bool(pub["event_missed"][i])
    res["reproduction_of_published_row"] = {
        k: res["metrics"]["A_withdrawn_legacy_as_published"][k] for k in res["published_row"] if k in
        res["metrics"]["A_withdrawn_legacy_as_published"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n")
    print(f"written -> {OUT.relative_to(ROOT)}")
    keys = ("r2_descriptive", "relrms_tau", "station_abs_p50", "station_abs_p95", "station_abs_max",
            "viscous_drag_signed_error", "separated_set_symmetric_difference",
            "separation_error_over_span", "reattachment_error_over_span", "event_missed",
            "station_sign_mismatch_fraction", "tau_rms_scale", "cf_rms_scale")
    print(f"{'metric':44s} {'published':>12s} {'A repro':>12s} {'B MGLET':>12s} {'C cubic':>12s}")
    for k in keys:
        p = res["published_row"].get(k, float("nan"))
        row = [res["metrics"][t].get(k, float("nan")) for t in
               ("A_withdrawn_legacy_as_published", "B_mglet", "C_repaired_cubic")]
        fmt = lambda v: (f"{v:12.4f}" if isinstance(v, float) else f"{str(v):>12s}")
        print(f"{k:44s} {fmt(p)} {fmt(row[0])} {fmt(row[1])} {fmt(row[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
