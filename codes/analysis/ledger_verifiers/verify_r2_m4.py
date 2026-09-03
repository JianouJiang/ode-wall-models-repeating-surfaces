#!/usr/bin/env python3
"""Ledger verifier for row R2-m4 / R3-2 ("PDE models are the cure" is untested).

2026-08-25 revision: the deposited Xiao 4-point wall-gradient reconstruction was
withdrawn as a SCORING reference.  The verifier now checks the RE-SCORED ladder
(MGLET DNS primary, repaired-Xiao cubic bracket), rebuilding both corrected
references from their raw deposits with independent algebra, and re-derives the
verdict on each.  The 2026-08-23 artifacts are still checked for integrity but
their scores are treated as superseded, not as the row's answer.

Checks the a-priori model ladder artifact, the coupled ARCHER2 ladder
artifact, rebuilds the truth and the headline metrics independently from the
stored arrays, re-derives the pre-registered verdict and rejects fixtures.
The row is closable only when the a-priori and coupled verdicts agree
(both SUPPORTED or both REFUTED); a falsified statement is a complete answer
provided it is stated, so a consistent REFUTED pair passes.

Run: python3 codes/analysis/ledger_verifiers/verify_r2_m4.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"
STAMP = "20260823"
APRIORI_JSON = RESULTS / f"r2m4_apriori_ladder_{STAMP}.json"
APRIORI_NPZ = RESULTS / f"r2m4_apriori_ladder_{STAMP}.npz"
COUPLED_JSON = RESULTS / f"r2m4_ladder_coupled_{STAMP}.json"
COUPLED_NPZ = RESULTS / f"r2m4_ladder_coupled_{STAMP}.npz"
RESCORE_STAMP = "20260825"
RESCORED_JSON = RESULTS / f"r2m4_ladder_rescored_{RESCORE_STAMP}.json"
RESCORED_NPZ = RESULTS / f"r2m4_ladder_rescored_{RESCORE_STAMP}.npz"
CORRECTED_COUPLED_JSON = RESULTS / f"r2m4_ladder_coupled_{RESCORE_STAMP}.json"
MGLET_WALL = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
PRIMARY = "B_mglet_deposited"
BRACKET = "C_xiao_cubic6_repaired"
WITHDRAWN = "A_xiao_linear4_deposited"
DNS = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
DNS_SHA = "d039cefb93ec1a8555555deed79041921bf8ce98cd1477479087a9804ca7ff85"
EXPECTED_UBAR = 0.721044918040774
LX = 9.0
DENSE = 4096


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def periodic_interp(x, y, target):
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1, x, x + 1], np.r_[y, y, y])


def rel_rms(phase, pred, tphase, ttau):
    d = np.arange(DENSE) / DENSE
    t = periodic_interp(tphase, ttau, d)
    p = periodic_interp(phase, pred, d)
    return float(np.sqrt(np.mean((p - t) ** 2)) / np.sqrt(np.mean(t ** 2)))


def independent_truth():
    """Deposit protocol: nu dU_t/dn from the first four points on the analytic tangent."""
    sys.path.insert(0, str(ROOT / "codes/openfoam"))
    from make_xiao_dns_wmles_case import HALF_WIDTH, xiao_profile
    d = np.load(DNS)
    x, y, U, V = (np.asarray(d[k], float) for k in ("x", "y", "U", "V"))
    h = np.asarray([xiao_profile(v) if v <= HALF_WIDTH else xiao_profile(LX - v) if v >= LX - HALF_WIDTH else 0.0 for v in x])
    dx = float(np.median(np.diff(x)))
    slope = (8 * (np.roll(h, -1) - np.roll(h, 1)) - (np.roll(h, -2) - np.roll(h, 2))) / (12 * dx)
    tx, ty = 1 / np.sqrt(1 + slope ** 2), slope / np.sqrt(1 + slope ** 2)
    nu = 1.0 / 5600.0
    tau = np.empty(x.size)
    for i in range(x.size):
        off = y[i, 1:5] - y[i, 0]
        ut = U[i, 1:5] * tx[i] + V[i, 1:5] * ty[i]
        tau[i] = nu * float(np.sum(off * ut) / np.sum(off ** 2)) / tx[i]
    return np.mod((x - x.min()) / LX, 1.0), tau


def independent_mglet():
    """B: deposited MGLET bottom-wall tau_w, plot-axis closure rows dropped."""
    raw = np.loadtxt(MGLET_WALL)
    keep = [i for i in range(len(raw)) if not (i >= len(raw) - 2 and np.all(raw[i, 1:] == 0.0))]
    body = raw[keep]
    return np.mod(body[:, 0] / LX, 1.0), body[:, 1]


def independent_repaired_xiao():
    """C: through-origin cubic on the first six fluid points, solved from the
    explicit 3x3 normal equations (not lstsq) so the algebra is independent."""
    sys.path.insert(0, str(ROOT / "codes/openfoam"))
    from make_xiao_dns_wmles_case import HALF_WIDTH, xiao_profile
    d = np.load(DNS)
    x, y, U, V = (np.asarray(d[k], float) for k in ("x", "y", "U", "V"))
    h = np.asarray([xiao_profile(v) if v <= HALF_WIDTH else xiao_profile(LX - v) if v >= LX - HALF_WIDTH else 0.0 for v in x])
    dx = float(np.median(np.diff(x)))
    slope = (8 * (np.roll(h, -1) - np.roll(h, 1)) - (np.roll(h, -2) - np.roll(h, 2))) / (12 * dx)
    tx, ty = 1 / np.sqrt(1 + slope ** 2), slope / np.sqrt(1 + slope ** 2)
    nu = 1.0 / 5600.0
    tau = np.empty(x.size)
    for i in range(x.size):
        ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])
        n = (y[i, ok] - y[i, ok][0])[1:7]
        ut = (U[i, ok] * tx[i] + V[i, ok] * ty[i])[1:7]
        moments = np.array([[np.sum(n ** (a + b + 2)) for b in range(3)] for a in range(3)])
        rhs = np.array([np.sum(ut * n ** (a + 1)) for a in range(3)])
        tau[i] = nu * float(np.linalg.solve(moments, rhs)[0]) / tx[i]
    return np.mod((x - x.min()) / LX, 1.0), tau


def side_verdict(e, d):
    if d["high"] < 0 and e["high"] < 1:
        return "SUPPORTED"
    if d["low"] > 0 or e["low"] >= 1:
        return "REFUTED"
    return "INCONCLUSIVE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true",
                        help="do not fail on a partial coupled harvest (campaign still running)")
    args = parser.parse_args()
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    # --- a priori artifact -------------------------------------------------
    ok_files = APRIORI_JSON.is_file() and APRIORI_NPZ.is_file()
    check("a-priori ladder artifact present", ok_files)
    if not ok_files:
        return finish(checks)
    ap = json.loads(APRIORI_JSON.read_text())
    apz = np.load(APRIORI_NPZ)
    check("a-priori status OK", ap.get("status") == "R2M4_APRIORI_LADDER_OK")
    check("a-priori DNS source hash bound", ap.get("dns_sha256") == DNS_SHA == sha256(DNS))
    tp, tt = independent_truth()
    check("truth rebuilt independently (deposit tangent protocol)",
          np.allclose(apz["truth_phase"], tp) and np.allclose(apz["truth_tau_s"], tt, atol=1e-15, rtol=0))
    ladder = ("M0_equilibrium", "M1_pressure_gradient_ode", "M2_hickel_modelled_convection",
              "Xc_resolved_convection_linear", "Xc_resolved_convection_constant", "Xc_exact_convection_profile")
    s = ap["surfaces"]["ladder_L1"]
    check("a-priori ladder complete on the coupled surface", all(m in s["metrics"] for m in ladder))
    rebuilt_ok = True
    for m in ladder:
        r = rel_rms(apz["ladder_L1_phase"], apz[f"ladder_L1_{m}"], tp, tt)
        rebuilt_ok &= abs(r - s["metrics"][m]["relative_rms"]) < 1e-12
    check("[superseded reference A] a-priori relative-RMS metrics rebuilt from stored arrays", rebuilt_ok)
    check("a-priori intervals present (phase-block bootstrap)",
          all("relative_rms_interval" in s["metrics"][m] for m in ladder) and ap["bootstrap"]["draws"] >= 20000)
    check("a-priori surface is the coupled ladder surface (y_m/H median 0.0935)",
          abs(s["y_m_over_H"]["median"] - 0.0935321823) < 1e-6)
    dxc = s["paired_relative_rms_differences"]["Xc_resolved_convection_linear-minus-M1_pressure_gradient_ode"]
    exc = s["metrics"]["Xc_resolved_convection_linear"]["relative_rms_interval"]
    ap_verdict = side_verdict(exc, dxc)
    check(f"[superseded reference A] a-priori verdict re-derived = {ap_verdict}", ap.get("apriori_verdict_ladder_L1") == ap_verdict)
    # red fixture: swapping Xc for a copy of M1 must not be SUPPORTED
    fixture = side_verdict(s["metrics"]["M1_pressure_gradient_ode"]["relative_rms_interval"],
                           {"low": 0.0, "high": 0.0})
    check("fixture: unchanged-M1 substitute cannot be SUPPORTED", fixture != "SUPPORTED")

    # --- coupled artifact --------------------------------------------------
    if not (COUPLED_JSON.is_file() and COUPLED_NPZ.is_file()):
        check("coupled ladder artifact present", args.allow_partial)
        return finish(checks)
    cp = json.loads(COUPLED_JSON.read_text())
    cpz = np.load(COUPLED_NPZ)
    complete = cp.get("status") == "R2M4_LADDER_HARVEST_OK"
    check("coupled harvest complete (5 L1 models + 2 W1 ceiling cases)", complete or args.allow_partial)
    check("[superseded reference A] coupled truth identical to a-priori truth", np.allclose(cpz["truth_tau_s"], apz["truth_tau_s"], atol=0, rtol=0))
    kernel_shas = {sha256(ROOT / "codes/openfoam/ladderWallModels/ladderTbleShoot.H"),
                   sha256(ROOT / "codes/openfoam/ladderWallModels_v2/ladderTbleShoot.H")}
    driver_shas = {sha256(ROOT / "jobs/r2m4_ladder_driver.sh"), sha256(ROOT / "jobs/r2m4_ladder_driver_v2.sh")}
    all_cases_ok = True
    for key, rec in cp["cases"].items():
        c = rec["checks"]
        ok = c["reached_405"] and c["mass_flow_matched"] and c["courant_bounded"]
        if rec.get("external_source"):      # W1 ceiling from the corrected M13 matrix (pinned driver)
            ok &= abs(rec["Ubar_volume"] - EXPECTED_UBAR) < 1e-5
        else:
            ok &= abs(rec["Ubar_volume"] - EXPECTED_UBAR) < 1e-12
            ok &= rec["ladder_kernel_sha256"] in kernel_shas and rec["driver_sha256"] in driver_shas
        ok &= "relative_rms_interval" in rec["metrics"]
        g, m = key.split(":")
        r = rel_rms(cpz[f"{g}_{m}_phase"], cpz[f"{g}_{m}_tau_s"], tp, tt)
        ok &= abs(r - rec["metrics"]["relative_rms"]) < 1e-12
        if not ok:
            print(f"       case {key} failed its checks")
        all_cases_ok &= ok
    check("every coupled case: t=405 reached, DNS mass flow held, Co<=0.56, pinned hashes, metric rebuilt",
          all_cases_ok and (bool(cp["cases"]) or args.allow_partial))
    # --- corrected references and the re-scored ladder ---------------------
    if not (RESCORED_JSON.is_file() and RESCORED_NPZ.is_file()):
        check("re-scored ladder artifact present (2026-08-25 reference correction)", False)
        return finish(checks)
    rs = json.loads(RESCORED_JSON.read_text())
    rz = np.load(RESCORED_NPZ)
    check("re-scored ladder status OK", rs.get("status") == "R2M4_LADDER_RESCORED_OK")
    check("withdrawn reference is not primary",
          rs["primary_reference"] == PRIMARY and rs["withdrawn_reference"] == WITHDRAWN)
    dense = np.asarray(rz["dense_phase"])
    bph, btau = independent_mglet()
    cph, ctau = independent_repaired_xiao()
    b_dense = periodic_interp(bph, btau, dense)
    c_dense = periodic_interp(cph, ctau, dense)
    check("MGLET reference rebuilt independently (placeholders stripped)",
          np.allclose(b_dense, rz[f"reference_{PRIMARY}"], rtol=0, atol=1e-12))
    check("repaired-Xiao reference rebuilt independently (normal-equation cubic)",
          np.allclose(c_dense, rz[f"reference_{BRACKET}"], rtol=1e-9, atol=1e-12))
    ratio_a = float(np.sqrt(np.mean(rz[f"reference_{WITHDRAWN}"] ** 2)) / np.sqrt(np.mean(b_dense ** 2)))
    ratio_c = float(np.sqrt(np.mean(c_dense ** 2)) / np.sqrt(np.mean(b_dense ** 2)))
    check(f"withdrawn reference is {ratio_a:.3f} of MGLET in RMS (audit: 0.360)", abs(ratio_a - 0.360) < 0.01)
    check(f"bracket reference is {ratio_c:.3f} of MGLET in RMS (audit: 0.626)", abs(ratio_c - 0.626) < 0.01)
    check("withdrawn reference reproduces the a-priori truth of the superseded artifact",
          np.allclose(rz[f"reference_{WITHDRAWN}"], periodic_interp(apz["truth_phase"], apz["truth_tau_s"], dense),
                      rtol=0, atol=1e-15))
    rebuilt = True
    for ref, truth in ((PRIMARY, b_dense), (BRACKET, c_dense)):
        for m in ladder:
            stored = rs["apriori"]["ladder_L1"][ref]["metrics"][m]["relative_rms"]
            e = rz[f"apriori_ladder_L1_{m}_dense"] - truth
            rebuilt &= abs(float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(truth ** 2))) - stored) < 1e-12
        for g in ("L1", "L2"):
            for mm in rs["coupled"][g][ref]["metrics"]:
                stored = rs["coupled"][g][ref]["metrics"][mm]["relative_rms"]
                e = rz[f"coupled_{g}_{mm}_dense"] - truth
                rebuilt &= abs(float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(truth ** 2))) - stored) < 1e-12
    check("every re-scored relative-RMS rebuilt from the stored dense arrays", rebuilt)
    verdicts_ok = True
    for ref in (PRIMARY, BRACKET):
        a = rs["apriori"]["ladder_L1"][ref]
        v = side_verdict(a["metrics"]["Xc_resolved_convection_linear"]["relative_rms_interval"],
                         a["paired_relative_rms_differences"]["Xc_resolved_convection_linear-minus-M1_pressure_gradient_ode"])
        verdicts_ok &= v == a["verdict_resolved_convection"] == rs["verdicts"][ref]["apriori_ladder_L1"]
        c1 = rs["coupled"]["L1"][ref]
        vc = side_verdict(c1["metrics"]["resolvedConvectionLinear"]["relative_rms_interval"],
                          c1["paired_relative_rms_differences"]["resolvedConvectionLinear-minus-totalGradient"])
        verdicts_ok &= vc == c1["verdict_resolved_convection"] == rs["verdicts"][ref]["coupled_L1"]
    check("verdicts re-derived on both corrected references", verdicts_ok)

    # --- the reference-robust findings the row rests on ---------------------
    robust_conditioning = all(
        rs["apriori"]["ladder_L1"][r]["paired_relative_rms_differences"][k]["low"] > 0.0
        for r in (WITHDRAWN, PRIMARY, BRACKET)
        for k in ("Xall_all_omitted_transport-minus-M0_equilibrium",
                  "Xfull_all_transport_plus_exact_shear_stress-minus-M0_equilibrium"))
    check("ill-conditioning signature is reference-robust (adding exact transport is worse, interval > 0 "
          "under all three references)", robust_conditioning)
    robust_coupled = all(
        rs["coupled"]["L1"][r]["paired_relative_rms_differences"]["resolvedConvectionLinear-minus-totalGradient"]["low"] > 0.0
        for r in (WITHDRAWN, PRIMARY, BRACKET))
    check("coupled convection restoration is worse than the PG-ODE under all three references "
          "(paired interval > 0)", robust_coupled)
    catastrophe_gone = rs["apriori"]["ladder_L1"][PRIMARY]["metrics"]["M1_pressure_gradient_ode"]["relative_rms"] < 1.0
    check("superseded 'PG-ODE fails catastrophically (2.84)' is NOT reported as the corrected result "
          f"(primary reference gives {rs['apriori']['ladder_L1'][PRIMARY]['metrics']['M1_pressure_gradient_ode']['relative_rms']:.3f})",
          catastrophe_gone)

    # --- corrected coupled harvest -----------------------------------------
    if CORRECTED_COUPLED_JSON.is_file():
        cc = json.loads(CORRECTED_COUPLED_JSON.read_text())
        check("corrected coupled harvest scores against the primary reference",
              cc.get("primary_reference") == PRIMARY and cc.get("schema") == "r2m4-coupled-ladder-v2")
        agree = all(abs(cc["cases"][f"{g}:{m}"]["metrics_by_reference"][PRIMARY]["relative_rms"]
                        - rs["coupled"][g][PRIMARY]["metrics"][m]["relative_rms"]) < 1e-12
                    for g in ("L1", "L2") for m in rs["coupled"][g][PRIMARY]["metrics"]
                    if f"{g}:{m}" in cc["cases"])
        check("corrected harvest and independent re-score agree on every coupled score", agree)
        check("corrected harvest row verdict matches the re-score",
              cc.get("row_verdict") == rs["verdicts"][PRIMARY]["row"])
    else:
        check("corrected coupled harvest present", args.allow_partial)

    # --- closure ------------------------------------------------------------
    row_primary = rs["verdicts"][PRIMARY]["row"]
    row_bracket = rs["verdicts"][BRACKET]["row"]
    check(f"primary and bracket references give the same row verdict ({row_primary})",
          row_primary == row_bracket)
    print(f"       row verdict: primary {row_primary} | bracket {row_bracket} "
          f"| superseded reference gave {rs['verdicts'][WITHDRAWN]['row']}")
    check("row closable (a-priori and coupled agree and are not inconclusive, on the corrected reference)",
          row_primary.startswith("CLOSABLE"))
    return finish(checks)


def finish(checks):
    n = sum(ok for _, ok in checks)
    print(f"{n}/{len(checks)} checks passed")
    return 0 if n == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
