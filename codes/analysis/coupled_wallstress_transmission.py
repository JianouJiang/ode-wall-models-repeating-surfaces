#!/usr/bin/env python3
"""
Error-transmission diagnostic (Level-0 feasibility, iteration 2026-05-30).

Goal: put the COUPLED WMLES wall stress on the SAME axis as the A-PRIORI
wall-stress metric, so the a-priori catastrophe (R^2(tau_w) approx -48 with
pristine DNS inputs) and the coupled outcome can be compared in COMMENSURABLE
units (both are bottom-wall skin friction C_f(x) vs the same DNS).

This closes the gap the manuscript currently concedes in
sec:aposteriori_sign ("not in commensurable units / apples-to-oranges").

Inputs  (real, on disk):
  codes/results/aposteriori_wmles_pehill.npz
    eq_x, eq_cf      : coupled equilibrium-WMLES bottom-wall C_f(x), 80 stations
    dns_x, dns_cf    : Krank et al. (2018) wall-resolved DNS C_f(x), 1153 pts
    apriori_*        : a-priori epsilon summary already harvested

Output:
  codes/results/coupled_wallstress_transmission.npz

Nothing here is fabricated; every number is computed from the loaded arrays.
The faithful TBLE-coupled rung (tble_present currently False) and the
conv-div coupled non-trigger are harvested by separate runs; when present this
script extends automatically.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)               # codes/
RES = os.path.join(ROOT, "results")
PEHILL = os.path.join(RES, "aposteriori_wmles_pehill.npz")


def r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def rel_rms(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return np.sqrt(np.mean((yhat - y) ** 2)) / np.sqrt(np.mean(y ** 2))


def coupled_wallstress_fidelity(npz_path, cf_key="eq_cf", x_key="eq_x"):
    d = np.load(npz_path, allow_pickle=True)
    wx, wcf = np.asarray(d[x_key], float), np.asarray(d[cf_key], float)
    dx, dcf = np.asarray(d["dns_x"], float), np.asarray(d["dns_cf"], float)
    # both are bottom-wall C_f(x) over the same hill period x in [0, 9];
    # interpolate the (denser) DNS onto the WMLES station grid.
    order = np.argsort(dx)
    dns_on_w = np.interp(wx, dx[order], dcf[order])
    return {
        "R2_coupled_cf": r2(dns_on_w, wcf),
        "relrms_coupled_cf": rel_rms(dns_on_w, wcf),
        "pearson_coupled_cf": float(np.corrcoef(dns_on_w, wcf)[0, 1]),
        "n_stations": int(wx.size),
        "x": wx,
        "cf_wmles": wcf,
        "cf_dns_on_w": dns_on_w,
    }


def main():
    d = np.load(PEHILL, allow_pickle=True)
    out = {}

    # ---- commensurable coupled wall-stress fidelity (equilibrium rung) ----
    eq = coupled_wallstress_fidelity(PEHILL, "eq_cf", "eq_x")
    out["eq_R2_coupled_cf"] = eq["R2_coupled_cf"]
    out["eq_relrms_coupled_cf"] = eq["relrms_coupled_cf"]
    out["eq_pearson_coupled_cf"] = eq["pearson_coupled_cf"]
    out["eq_n_stations"] = eq["n_stations"]
    out["eq_x"] = eq["x"]
    out["eq_cf_wmles"] = eq["cf_wmles"]
    out["eq_cf_dns_on_w"] = eq["cf_dns_on_w"]

    # ---- bulk-transmission references already in the npz ----
    out["eq_reatt_rel_err_pct"] = float(d["eq_reattachment_rel_err_pct"])
    out["eq_profile_rms_mean"] = float(d["eq_profile_rms_mean"])
    out["eq_eps_median_coupled"] = float(d["eq_eps_median"])
    out["apriori_eps_median"] = float(d["apriori_median_eps"])

    # ---- faithful TBLE rung, if/when harvested (live run) ----
    if bool(d["tble_present"]):
        tb = coupled_wallstress_fidelity(PEHILL, "tble_cf", "tble_x")
        out["tble_R2_coupled_cf"] = tb["R2_coupled_cf"]
        out["tble_relrms_coupled_cf"] = tb["relrms_coupled_cf"]
        out["tble_present"] = True
    else:
        out["tble_present"] = False

    np.savez(os.path.join(RES, "coupled_wallstress_transmission.npz"), **out)

    # ---- report ----
    print("=== Coupled wall-stress transmission (periodic hills) ===")
    print(f"  equilibrium rung, {out['eq_n_stations']} stations")
    print(f"  R2(C_f coupled vs DNS)   = {out['eq_R2_coupled_cf']:+.3f}")
    print(f"  rel-RMS(C_f coupled)     = {out['eq_relrms_coupled_cf']:.3f}")
    print(f"  Pearson r (C_f coupled)  = {out['eq_pearson_coupled_cf']:+.3f}")
    print(f"  reattachment error       = {out['eq_reatt_rel_err_pct']:+.1f} %")
    print(f"  10-station mean-U RMS    = {out['eq_profile_rms_mean']:.3f} u_b")
    print(f"  coupled eps median       = {out['eq_eps_median_coupled']:.3f}")
    print(f"  a-priori eps median      = {out['apriori_eps_median']:.3f}")
    print(f"  TBLE rung harvested      = {out['tble_present']}")
    print()
    print("Reference (manuscript, a-priori, pristine inputs, SAME equilibrium")
    print("model): R2(tau_w) = -48, rel-err = 6.8.  The coupled R2 above is on")
    print("the SAME wall-stress axis -> the gap is the error-transmission effect.")


if __name__ == "__main__":
    main()
