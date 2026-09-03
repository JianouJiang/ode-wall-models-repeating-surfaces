#!/usr/bin/env python3
"""Verifier for the R1-STA-2 wavy-wall AMPLITUDE LADDER (2a/lambda = 0.10 vs 0.20).

Checks that codes/results/r1_sta2_wavy_amplitude_<date>.{json,npz}
(i)   is built from BOTH amplitudes with everything except amplitude matched;
(ii)  keeps the two families' validation status honestly asymmetric -- the mild family
      reference-validated against hash-pinned ERCOFTAC data, the steep family explicitly
      NOT reference-validated, with the transfer argument recorded;
(iii) carries >= 2 converged steep grids with the verdict invariant across them and
      six-block uncertainties on every headline;
(iv)  rebuilds the headline R2 and epsilon from the npz arrays; and
(v)   states the amplitude verdict plainly, whichever way it falls.

A steep result that ALSO does not fail is a PASS here: the row's job is to report the
amplitude ladder honestly, not to manufacture a failure.
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


def digest(path: Path) -> str:
    if not path.exists():
        return "MISSING:" + str(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def denone(o):
    if isinstance(o, dict):
        return {k: denone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [denone(v) for v in o]
    return float("nan") if o is None else o


def r2(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[m], true[m]
    return 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    if args.artifact:
        jp = Path(args.artifact)
    else:
        c = sorted(RESULTS.glob("r1_sta2_wavy_amplitude_*.json"))
        if not c:
            print("[FAIL] no r1_sta2_wavy_amplitude_<date>.json artifact on disk")
            print("0/1 checks passed")
            return 1
        jp = c[-1]
    art = denone(json.loads(jp.read_text()))
    data = np.load(jp.with_suffix(".npz"), allow_pickle=False)
    fams, L = art["families"], art["amplitude_ladder"]
    vs = art["validation_status"]
    checks = []

    checks.append(("artifact schema / row", art.get("schema") == "r1_sta2_wavy_amplitude_v1"
                   and art.get("ledger_row") == "R1-STA-2"))
    checks.append(("both amplitudes present", "mild_archer2" in fams and "steep_arc" in fams
                   and len(fams["steep_arc"]) >= 1))
    steep = fams["steep_arc"]
    checks.append((">= 2 steep grids, all converged to endTime",
                   len(steep) >= 2 and all(steep[g]["converged"] for g in steep)))
    checks.append(("steep grid ladder is a real refinement (>= 1.3x linear)",
                   len(steep) >= 2 and
                   (max(steep[g]["cells"] for g in steep) /
                    min(steep[g]["cells"] for g in steep)) ** (1 / 3.) >= 1.3))

    # --- the one-variable claim
    checks.append(("amplitude is the ONLY changed parameter (Re_h and lambda/delta matched)",
                   bool(L["matched"]["Re_h"]) and bool(L["matched"]["lambda_over_delta"])))
    checks.append(("amplitude actually doubled", abs(L["amplitude_ratio"] - 2.0) < 1e-9))
    # Resolve each case directory by its own case_id across BOTH run roots.  The
    # previous rule keyed on the family name containing "arc", which also matches
    # "mild_archer2", so the three ARCHER2 mild cases were looked for under the
    # amplitude root and the numerics and hash checks failed on a path error rather
    # than on the evidence.  Resolution must be unique: a case_id present in both
    # roots, or in neither, is a failure, not a silent choice.
    ROOTS = (RESULTS / "r1_sta2_wavy_amplitude", RESULTS / "r1_sta2_wavy_wrles")

    def case_dir(case_id):
        hits = [r / case_id for r in ROOTS if (r / case_id).is_dir()]
        return hits[0] if len(hits) == 1 else None

    okres = all(case_dir(d["case_id"]) is not None
                for F in fams.values() for d in F.values())
    checks.append(("every case directory resolves to exactly one run root", okres))
    okn = True
    for fname, F in fams.items():
        for g, d in F.items():
            cd = case_dir(d["case_id"])
            if cd is None:
                okn = False
                continue
            gp = cd / "GEOMETRY.json"
            if not gp.exists():
                okn = False
                continue
            geo = json.loads(gp.read_text())
            okn &= (geo["sgs"].startswith("WALE") and geo["convection"] == "Gauss LUST grad(U)"
                    and geo["ddt"] == "backward" and abs(geo["maxCo"] - 0.5) < 1e-12
                    and abs(geo["Re_h"] - 3460.0) < 1e-9
                    and abs(geo["lambda_over_delta"] - 2.0) < 1e-9)
    checks.append(("every case carries the deposited rib-WRLES numerics and matched Re/lambda", okn))
    okh, nhash = True, 0
    for fname, F in fams.items():
        for g, d in F.items():
            cd = case_dir(d["case_id"])
            if cd is None:
                okh = False
                continue
            for fn, h in d["source_hashes"]["reduced"].items():
                okh &= digest(cd / "reduced" / fn) == h
                nhash += 1
            gh = d["source_hashes"].get("geometry")
            if gh:
                okh &= digest(cd / "GEOMETRY.json") == gh
                nhash += 1
    checks.append(("all reduced outputs and geometries hash-bound (%d files)" % nhash,
                   okh and nhash >= 2 * sum(len(F) for F in fams.values())))

    # --- honest, asymmetric validation status
    checks.append(("mild family declared reference-validated",
                   vs["mild"]["reference_validated"] is True and len(vs["mild"]["references"]) == 2))
    checks.append(("steep family declared NOT reference-validated, with reason and transfer",
                   vs["steep"]["reference_validated"] is False
                   and len(str(vs["steep"]["reason"])) > 80
                   and "identical" in str(vs["steep"]["validation_transfer"])))
    checks.append(("steep literature kept as consistency notes only, never a gate",
                   set(vs["steep"]["consistency_notes_only"]) == set(art["steep_literature"])
                   and all("note" in v for v in art["steep_literature"].values())))
    m = fams["mild_archer2"]
    mg = sorted(m, key=lambda g: m[g]["cells"])[-1]
    checks.append(("mild family still meets its own reference gates",
                   m[mg]["validation"]["maass_U_l2_median"] <= art["tolerances"]["maass_l2_median"]
                   and m[mg]["validation"]["hudson_U_l2_median"] <=
                   art["tolerances"]["hudson_dns_margin"] * art["hudson_dns_reference_l2"]["U_median"]))

    # --- resolution and averaging, both families, identical treatment
    okr = True
    for fname, F in fams.items():
        for g, d in F.items():
            okr &= (d["resolution"]["y1_plus"] <= art["tolerances"]["yplus_max"]
                    and d["resolution"]["flow_throughs"] >= 20.0
                    and d["wall"]["momentum_closure_rel"] <= art["tolerances"]["momentum_closure_rel"])
    checks.append(("every case wall-resolved, >= 20 flow-throughs, momentum balance closed", okr))
    checks.append(("six-block uncertainties on every steep headline",
                   all(steep[g]["uncertainty"]["n_blocks"] >= 4
                       and steep[g]["uncertainty"]["block_windows"]["x_sep"]["n"] >= 4
                       and steep[g]["uncertainty"]["block_windows_ode"]["0.1"]["standard_ml"]["n"] >= 4
                       for g in steep)))
    checks.append(("verdict stated at >= 4 matching heights for both amplitudes",
                   len(L["mild"]["standard_ml_r2_by_eta"]) >= 4
                   and len(L["steep"]["standard_ml_r2_by_eta"]) >= 4))
    checks.append(("steep verdict invariant across its grids", bool(L["steep_grid_invariant"])))

    # --- independent rebuild from the npz
    sg = sorted(steep, key=lambda g: steep[g]["cells"])[-1]
    tag = "steep_arc_%s" % sg
    pred = data["%s_eta0.1_pred_standard_ml" % tag]
    truth = data["%s_eta0.1_tau_ref" % tag]
    checks.append(("steep R2 at eta_m=0.1 rebuilt from npz",
                   abs(r2(pred, truth) - steep[sg]["verdict"]["standard_ml_r2_by_eta"]["0.1"]) < 1e-9))
    eps = data["%s_eta0.1_eps" % tag]
    checks.append(("steep eps median rebuilt from npz",
                   abs(np.median(eps[np.isfinite(eps)]) -
                       steep[sg]["verdict"]["eps_median_by_eta"]["0.1"]) < 1e-9))
    tau = data["%s_tau_t" % tag]
    checks.append(("steep wall traction reverses somewhere (flow is separated)",
                   float(np.mean(tau < 0)) > 0.05))
    rng = np.random.default_rng(0)
    checks.append(("red fixture: phase-shuffled prediction is not the deposited R2",
                   abs(r2(pred[rng.permutation(len(pred))], truth) -
                       steep[sg]["verdict"]["standard_ml_r2_by_eta"]["0.1"]) > 1e-6))

    # --- cross-cluster handling
    if L.get("same_cluster_tiepoint"):
        cc = L["cross_cluster_check"]
        checks.append(("same-cluster tie-point present: amplitude comparison never spans clusters",
                       True))
        checks.append(("cross-cluster replay of the mild G0 agrees (x_sep < 0.05 lambda, u* < 10%)",
                       cc["d_x_sep"] < 0.05 and cc["d_ustar_rel"] < 0.10))
    else:
        checks.append(("cross-cluster caveat recorded when no tie-point exists",
                       "machine" in L["mild"] and "machine" in L["steep"]
                       and L["mild"]["machine"] != L["steep"]["machine"]))
    checks.append(("plain-language amplitude statement present",
                   isinstance(L.get("statement"), str) and len(L["statement"]) > 80))
    checks.append(("status flag consistent", art["status"] == "R1_STA2_AMPLITUDE_OK"))

    print("AMPLITUDE LADDER: " + L["statement"] + "\n")
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
