#!/usr/bin/env python3
"""Stable-path verifier for claim R1-STA-2 (wavy-wall WRLES, second failure geometry).

Checks that the deposited artifact codes/results/r1_sta2_wavy_wrles_<date>.{json,npz}
(i)   exists, is complete (3 converged grids of the ladder) and hash-bound to the
      reduced ARCHER2 outputs and the two public reference data sets;
(ii)  validates against the Hudson 1996 / Maass-Schumann 1996 / Cherukat 1998
      references within the tolerances the artifact itself declares;
(iii) carries the a-priori ODE diagnostic with block-window uncertainties at
      several matching heights, and states the failure-instance verdict plainly
      (the verdict itself is NOT a pass condition: a validated 'not a failure'
      would also close the row honestly);
(iv)  re-derives the headline numbers from the npz arrays (no trust in the json),
      and rejects control cases (sign-flipped traction, shuffled phase).

Usage:  python3 codes/analysis/ledger_verifiers/verify_r1_sta2.py [--artifact PATH.json]
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
sys.path.insert(0, str(ROOT / "codes/analysis"))


def digest(path: Path) -> str:
    if not path.exists():
        return "MISSING:" + str(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def denone(o):
    """JSON has no NaN: the harvest writes None for non-finite numbers; restore NaN."""
    if isinstance(o, dict):
        return {k: denone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [denone(v) for v in o]
    return float("nan") if o is None else o


def r2(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[m], true[m]
    return 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


def crossings(tau, waves=2):
    n = len(tau)
    xg = np.arange(n) / n * waves
    seps, res = [], []
    for i in range(n):
        j = (i + 1) % n
        t0, t1 = tau[i], tau[j]
        xc = ((xg[i] + (t0 / (t0 - t1)) * (waves / n)) % waves) % 1.0
        if t0 > 0 >= t1:
            seps.append(xc)
        elif t0 < 0 <= t1:
            res.append(xc)
    return np.array(seps), np.array(res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    if args.artifact:
        jpath = Path(args.artifact)
    else:
        cands = sorted(RESULTS.glob("r1_sta2_wavy_wrles_*.json"))
        if not cands:
            print("[FAIL] no r1_sta2_wavy_wrles_<date>.json artifact on disk")
            print("0/1 checks passed")
            return 1
        jpath = cands[-1]
    npath = jpath.with_suffix(".npz")
    art = json.loads(jpath.read_text())
    art["grids"] = denone(art["grids"])
    art["grid_convergence"] = denone(art.get("grid_convergence", {}))
    data = np.load(npath, allow_pickle=False)
    grids = art["grids"]
    order = [g for g in ("G0", "G1", "G2") if g in grids]
    tol = art["tolerances"]
    ref = art["reference_scalars"]
    checks = []

    # ---- completeness + provenance
    checks.append(("artifact schema / row", art.get("schema") == "r1_sta2_wavy_wrles_v1" and art.get("ledger_row") == "R1-STA-2"))
    checks.append(("three-grid ladder present and converged", len(order) == 3 and all(grids[g]["converged"] for g in order)))
    checks.append(("producer job ids recorded", all(isinstance(grids[g].get("slurm_job_id"), str) for g in order)))
    ok_hash = True
    for g in order:
        cdir = RESULTS / "r1_sta2_wavy_wrles" / grids[g]["case_id"]
        for fn, h in grids[g]["source_hashes"]["reduced"].items():
            p = cdir / "reduced" / fn
            ok_hash &= p.exists() and digest(p) == h
        ok_hash &= digest(cdir / "GEOMETRY.json") == grids[g]["source_hashes"]["geometry"]
    checks.append(("reduced ARCHER2 outputs hash-bound", ok_hash))
    ok_ref = True
    for fam in ("hudson", "maass"):
        for k, v in art["reference_files"][fam].items():
            p = ROOT / v["file"]
            ok_ref &= p.exists() and digest(p) == v["sha256"]
    checks.append(("ERCOFTAC case 76/77 reference files hash-bound", ok_ref))
    checks.append(("no modelled eddy viscosity in reference quantities",
                   art.get("uses_modelled_eddy_viscosity_in_reference") is False))

    # ---- matched numerics record
    okn = True
    for g in order:
        gp = RESULTS / "r1_sta2_wavy_wrles" / grids[g]["case_id"] / "GEOMETRY.json"
        if not gp.exists():
            okn = False
            continue
        geo = json.loads(gp.read_text())
        okn &= geo["sgs"].startswith("WALE") and geo["convection"] == "Gauss LUST grad(U)" and geo["ddt"] == "backward"
        okn &= abs(geo["maxCo"] - 0.5) < 1e-12 and abs(geo["maxDeltaT"] - 0.008) < 1e-12
        okn &= abs(geo["two_a_over_lambda"] - 0.1) < 1e-12 and abs(geo["lambda_over_delta"] - 2.0) < 1e-12
        okn &= abs(geo["Re_h"] - 3460.0) < 1e-9
    checks.append(("numerics = deposited rib WRLES (WALE/LUST/backward/maxCo 0.5) and Cherukat geometry", okn))

    fg = grids[order[-1]] if order else None
    if fg is not None:
        res = fg["resolution"]
        checks.append(("finest grid wall-resolved (y1+ <= %.1f, nut_wall ~ 0)" % tol["yplus_max"],
                       res["y1_plus"] <= tol["yplus_max"] and res["nut_wall_max_over_nu"] < 0.05))
        checks.append(("averaging window >= 20 flow-throughs", res["flow_throughs"] >= 20.0))
        w = fg["wall"]
        dns_sep = 0.5 * (ref["maass_schumann_1996"]["x_sep"] + ref["cherukat_1998"]["x_sep"])
        dns_re = 0.5 * (ref["maass_schumann_1996"]["x_re"] + ref["cherukat_1998"]["x_re"])
        checks.append(("separation phase within %.2f lambda of the two DNS" % tol["x_sep"], abs(w["x_sep"] - dns_sep) <= tol["x_sep"]))
        checks.append(("reattachment phase within %.2f lambda of the two DNS" % tol["x_re"], abs(w["x_re"] - dns_re) <= tol["x_re"]))
        uref = 0.5 * (ref["maass_schumann_1996"]["ustar_wavy"] + ref["hudson_1996"]["ustar_wavy"])
        checks.append(("wavy-wall total-drag u* within %.0f%% of Hudson/Maass" % (100 * tol["ustar_rel"]),
                       abs(w["ustar_wavy"] - uref) / uref <= tol["ustar_rel"]))
        checks.append(("global momentum balance closes (body force vs wall forces)",
                       w["momentum_closure_rel"] <= tol["momentum_closure_rel"]))
        v = fg["validation"]
        cross = art["hudson_dns_reference_l2"]
        checks.append(("Maass DNS mean-velocity profiles: median rel-L2 <= %.2f" % tol["maass_l2_median"],
                       v["maass_U_l2_median"] <= tol["maass_l2_median"]))
        # The experiment gate is relative to the reference DNS's OWN distance from the
        # experiment (recomputed by the harvest from the deposited files), because that
        # distance -- 0.101 median -- exceeds any absolute 0.10 gate one might impose.
        checks.append(("Hudson experiment: no worse than %.2fx the DNS's own distance (%.3f) from it"
                       % (tol["hudson_dns_margin"], cross["U_median"]),
                       v["hudson_U_l2_median"] <= tol["hudson_dns_margin"] * cross["U_median"]))
        checks.append(("experiment gate is anchored to a recomputed reference distance, not a literal",
                       0.0 < cross["U_median"] < 1.0 and len(cross["U_by_station"]) == 10))
        # ---- independent rebuild from npz
        tau = data["%s_tau_t" % order[-1]]
        seps, reat = crossings(tau)
        checks.append(("separation/reattachment rebuilt from npz traction",
                       seps.size == 2 and reat.size == 2 and abs(seps.mean() - w["x_sep"]) < 1e-9 and abs(reat.mean() - w["x_re"]) < 1e-9))
        eta = "0.1"
        od = fg["ode_diagnostic"][eta]
        pred = data["%s_eta0.1_pred_standard_ml" % order[-1]]
        truth = data["%s_eta0.1_tau_ref" % order[-1]]
        checks.append(("R2(standard ODE) at eta_m=0.1 rebuilt from npz", abs(r2(pred, truth) - od["standard_ml"]) < 1e-9))
        eps = data["%s_eta0.1_eps" % order[-1]]
        checks.append(("eps median rebuilt from npz", abs(np.median(eps[np.isfinite(eps)]) - od["eps_median"]) < 1e-9))
        # ---- uncertainty carried
        unc = fg["uncertainty"]
        checks.append(("block-window uncertainty on x_sep, x_re, u*, R2 (>= 4 blocks)",
                       unc["n_blocks"] >= 4 and all(unc["block_windows"][k]["n"] >= 4 for k in ("x_sep", "x_re", "ustar_wavy"))
                       and unc["block_windows_ode"]["0.1"]["standard_ml"]["n"] >= 4))
        checks.append(("verdict stated at >= 4 matching heights", len(fg["verdict"]["standard_ml_r2_by_eta"]) >= 4))
        checks.append(("wall-normal origin recovered from the meshed wall (fit resid << first cell)",
                       fg["wall"]["wall_origin_fit_residual_over_dy1"] < 0.05))
        checks.append(("top-level validation and ode_verdict blocks populated",
                       isinstance(art.get("validation"), dict) and isinstance(art.get("ode_verdict"), dict)
                       and len(art["validation"]["per_grid"]) == 3 and len(art["ode_verdict"]["per_grid"]) == 3
                       and isinstance(art["ode_verdict"].get("statement"), str)))
        conv = art["grid_convergence"]
        checks.append(("grid convergence: verdict invariant across the ladder", bool(conv.get("verdict_invariant"))))
        lc_sep = conv.get("x_sep", {}).get("last_change", float("nan"))
        lc_re = conv.get("x_re", {}).get("last_change", float("nan"))
        checks.append(("grid convergence: separation/reattachment change on last refinement <= %.2f lambda" % tol["x_sep"],
                       abs(lc_sep) <= tol["x_sep"] and abs(lc_re) <= tol["x_re"]))
        # ---- control cases
        fs, fr = crossings(-tau)
        checks.append(("control case: sign-flipped traction swaps separation/reattachment",
                       fs.size == 2 and abs(fs.mean() - w["x_re"]) < 1e-9))
        rng = np.random.default_rng(0)
        shuffled = r2(pred[rng.permutation(len(pred))], truth)
        checks.append(("control case: phase-shuffled prediction is not the deposited R2", abs(shuffled - od["standard_ml"]) > 1e-6))
        checks.append(("status flag consistent", art["status"] == "R1_STA2_WAVY_WRLES_OK"))
        print("verdict (finest grid %s, %d cells): second_failure_instance=%s ; R2(standard ODE) by eta_m = %s ; eps_med by eta_m = %s"
              % (order[-1], fg["cells"], fg["verdict"]["second_failure_instance"],
                 {k: round(v, 3) for k, v in fg["verdict"]["standard_ml_r2_by_eta"].items()},
                 {k: round(v, 3) for k, v in fg["verdict"]["eps_median_by_eta"].items()}))

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
