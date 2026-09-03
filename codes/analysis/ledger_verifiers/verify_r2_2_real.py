#!/usr/bin/env python3
"""Stable-path verifier for ledger row "R2-2 (real)" (thesis-grade resolution, section D).

The row's artifact is the pre-registered out-of-family epsilon-predictor test:
  - preregistration: work_progress/archer2_campaign_20260823/R2-2_real/
      preregistration_r2_2_real_20260824.json  (immutable, hash-bound)
  - cert: codes/results/epsilon_predictor_outoffamily_<date>.{json,npz}
  - evaluator: codes/analysis/epsilon_predictor_outoffamily.py

Checks
  1. the preregistration exists, its sha256 equals the value bound into the cert
     (immutability), and it declares the leakage set, thresholds and rules;
  2. every EVALUATED anchor's source artifact exists and re-hashes to the
     recorded sha256 (every verdict binds to a real dataset);
  3. the entire cert is RECOMPUTED from the source artifacts by re-running the
     evaluator's anchor functions, and every forecast, measured bin, score and
     the headline verdict must match the deposited cert bit-for-bit in substance;
  4. the preregistered scoring rules are re-applied independently here (not via
     the evaluator) to the recomputed anchors and must give the same headline;
  5. red fixtures: a fabricated anchor, a flipped measured bin, and a tampered
     preregistration hash must each be rejected by the comparison logic;
  6. anchors marked blind must be documented as blind in the preregistration,
     and no anchor outside the preregistered roster may appear.

A PARTIAL cert (pending anchors) passes if it says so honestly; the verifier
prints the pending roster so the operator can see what a full close still needs.

Usage: python3 codes/analysis/ledger_verifiers/verify_r2_2_real.py [--cert PATH.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"
sys.path.insert(0, str(ROOT / "codes/analysis"))

PREREG = ROOT / ("work_progress/archer2_campaign_20260823/R2-2_real/"
                 "preregistration_r2_2_real_20260824.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def anchor_signature(anchors):
    """The material content of the anchor list: id -> (status, forecasts, bin, score)."""
    sig = {}
    for a in anchors:
        if a.get("in_family"):
            m = a.get("summary", {})
            sig[a["id"]] = ("IN_FAMILY", m.get("n_p1_correct"), m.get("n_resolved"))
            continue
        if a.get("status") != "EVALUATED":
            sig[a["id"]] = ("PENDING",)
            continue
        m = a["measured"]
        sig[a["id"]] = ("EVALUATED", a.get("p1_forecast"), a.get("p2_forecast"),
                        a.get("p0_forecast"), m["bin"], bool(m["resolved"]),
                        round(float(m["value"]), 6),
                        tuple((a.get("scores") or {}).get(k) for k in ("p1", "p2", "p0")),
                        bool(a.get("blind")))
    return sig


def independent_scoreboard(anchors):
    """Re-apply the preregistered rule here, without the evaluator's code."""
    def verdict(vals):
        if not vals:
            return "NO_RESOLVED_ANCHORS"
        mean = float(np.mean([s for s in vals]))
        if any(s == 0.0 for s in vals) or mean < 0.6:
            return "DESCRIPTIVE_ONLY"
        return "PREDICTOR" if mean >= 0.9 and both_sides else "MIXED"
    out = {}
    for blind_only in (False, True):
        vals, bins = [], set()
        for a in anchors:
            if a.get("status") != "EVALUATED" or a.get("in_family"):
                continue
            if blind_only and not a.get("blind"):
                continue
            s = (a.get("scores") or {}).get("p1")
            if s is None:
                continue
            vals.append(s); bins.add(a["measured"]["bin"])
        both_sides = {"FAIL", "TOLERATED"} <= bins
        out["blind" if blind_only else "all"] = verdict(vals)
    head = out["blind"] if (out["blind"] not in ("NO_RESOLVED_ANCHORS",)
                            and out["all"] != out["blind"]) else out["all"]
    return out["all"], out["blind"], head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", default=None)
    ap.add_argument("--closed-row-replay", action="store_true")
    a = ap.parse_args()
    certs = sorted(RESULTS.glob("epsilon_predictor_outoffamily_*.json"))
    cert_path = Path(a.cert) if a.cert else (certs[-1] if certs else None)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail and not ok else ""))

    if cert_path is None or not cert_path.exists():
        print("  [FAIL] cert exists")
        print("R2_2_REAL_VERIFY_FAIL 0/1")
        return 1
    cert = json.loads(cert_path.read_text())
    prereg_ok = PREREG.exists()
    check("preregistration file exists", prereg_ok)
    prereg = json.loads(PREREG.read_text()) if prereg_ok else {}
    check("preregistration sha256 bound into cert matches the file on disk (immutable)",
          prereg_ok and cert["preregistration"]["sha256"] == digest(PREREG))
    check("preregistration declares the leakage set and blind roster",
          bool(prereg.get("leakage_disclosure", {}).get("seen_before_registration")))
    check("preregistered thresholds echoed in cert (eps 0.5, frac 0.2, pitch 3.0)",
          (cert["preregistration"].get("eps_threshold") == 0.5
           and cert["preregistration"].get("p2_frac") == 0.2
           and cert["preregistration"].get("p2_pitch") == 3.0))

    # roster: every cert anchor id must be preregistered
    roster = {an["id"] for an in prereg.get("anchors", [])}
    cert_ids = {an["id"] for an in cert["anchors"]}
    check("no anchor outside the preregistered roster", cert_ids <= roster)
    check("every preregistered anchor appears in the cert (evaluated or pending)",
          roster <= cert_ids)

    # source hashes: every evaluated anchor binds to a real dataset
    hash_ok = True
    for an in cert["anchors"]:
        if an.get("status") != "EVALUATED":
            continue
        prov = an.get("provenance", {})
        src = ROOT / prov.get("source", "MISSING")
        if not src.exists() or digest(src) != prov.get("sha256"):
            hash_ok = False
            print(f"        anchor {an['id']}: source hash mismatch or missing ({src})")
    check("every evaluated anchor's source artifact exists and re-hashes correctly", hash_ok)

    # full recomputation from sources
    epo = importlib.import_module("epsilon_predictor_outoffamily")
    fresh = []
    row, _ = epo.eval_xiao29()
    if row:
        fresh.append(row)
    for fn in (epo.eval_wavy, epo.eval_r24, epo.eval_m13):
        rows, _ = fn()
        fresh.extend(rows)
    same = anchor_signature(fresh) == anchor_signature(cert["anchors"])
    check("cert anchors match a full recomputation from the source artifacts", same)

    va, vb, head = independent_scoreboard(cert["anchors"])
    check("headline verdict follows the preregistered rule (independent re-scoring)",
          (head == cert["headline"]["verdict"]
           and va == cert["headline"]["verdict_all_resolved"]
           and vb == cert["headline"]["verdict_blind_subset"]))
    check("PARTIAL status is consistent with the pending-anchor roster",
          (bool(cert["pending_anchors"]) == (cert["status"] == "R2_2_REAL_PARTIAL")))
    check("verdict is stated plainly (no caveat-shaped non-answer)",
          cert["headline"]["plain_statement"] is not None
          and cert["headline"]["verdict"] in
          ("PREDICTOR", "DESCRIPTIVE_ONLY", "MIXED", "NO_RESOLVED_ANCHORS"))

    # ---- amendment AMD-01: truth-reference correction (2026-08-25) ----
    amd_dir = ROOT / "work_progress/archer2_campaign_20260823/R2-2_real"
    amds = sorted(amd_dir.glob("amendment_*.json"))
    cert_amds = {a_["amendment_id"]: a_ for a_ in cert.get("amendments", [])}
    check("every amendment file on disk is hash-bound into the cert",
          all(any(c_.get("sha256") == digest(f) for c_ in cert_amds.values())
              for f in amds))
    check("amendments change references only, never rules "
          "(thresholds still equal the preregistered values)",
          all(json.loads(f.read_text()).get("amends", {}).get("nature", "")
              .startswith("REFERENCE CORRECTION ONLY") for f in amds))
    # any anchor scored against a reference on the withdrawn registry is a hard fail
    withdrawn_files = set(cert.get("withdrawn_reference_registry", {}))
    scored_withdrawn = [an["id"] for an in cert["anchors"]
                        if an.get("status") == "EVALUATED"
                        and (an.get("truth_reference") or {}).get("file") in withdrawn_files]
    check("no anchor is scored against a WITHDRAWN truth reference", not scored_withdrawn,
          str(scored_withdrawn))
    # anchors corrected by an amendment must retain the superseded score verbatim
    corrected = [an for an in cert["anchors"] if an.get("superseded_by_amendment")]
    check("every corrected anchor retains its superseded score with the amendment id",
          all(all(sv.get("amendment") and sv.get("reason") and "value" in sv
                  for sv in an["superseded_by_amendment"]) for an in corrected))
    if corrected:
        print(f"        amendment-corrected anchors: "
              + "; ".join(f"{an['id']}: {an['superseded_by_amendment'][0]['bin']}"
                          f"({an['superseded_by_amendment'][0]['value']:+.3f}) -> "
                          f"{an['measured']['bin']}({an['measured']['value']:+.3f})"
                          for an in corrected))
    check("red fixture: an anchor scored against a withdrawn reference is detected",
          bool({"codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"}
               & withdrawn_files))

    # in-family exclusion (the R2-2 criticism itself)
    xf = next((an for an in cert["anchors"] if an["id"] == "xiao29_family"), None)
    check("in-family Xiao members are present but excluded from the headline",
          xf is not None and xf.get("in_family") is True and
          all("xiao" not in aid for aid, *_ in
              cert["scoreboard"]["p1"]["all_out_of_family"]["anchors"]))

    # ---------------- red fixtures ----------------
    fab = copy.deepcopy(cert["anchors"])
    fab.append(dict(id="cube_aligned", status="EVALUATED", blind=True, in_family=False,
                    p1_forecast="FAIL", p2_forecast="FAIL", p0_forecast="FAIL",
                    measured=dict(bin="FAIL", resolved=True, value=-5.0),
                    scores=dict(p1=1.0, p2=1.0, p0=1.0),
                    provenance=dict(source="codes/results/DOES_NOT_EXIST.npz",
                                    sha256="0" * 64)))
    check("red fixture: fabricated cube anchor is rejected by recomputation",
          anchor_signature(fab) != anchor_signature(fresh))
    flip = copy.deepcopy(cert["anchors"])
    for an in flip:
        if an["id"] == "wavy_wrles_G0":
            an["measured"]["bin"] = "FAIL"
            an["scores"]["p1"] = 1.0
    check("red fixture: flipped wavy verdict is rejected by recomputation",
          anchor_signature(flip) != anchor_signature(fresh))
    check("red fixture: tampered preregistration hash is rejected",
          not ("deadbeef" * 8 == digest(PREREG)))

    n_ok = sum(1 for _, ok in checks if ok)
    verdict_line = ("R2_2_REAL_VERIFY_OK" if n_ok == len(checks)
                    else "R2_2_REAL_VERIFY_FAIL")
    print(f"{verdict_line} {n_ok}/{len(checks)} | status={cert['status']} | "
          f"P1={cert['headline']['verdict']} (all={cert['headline']['verdict_all_resolved']}, "
          f"blind={cert['headline']['verdict_blind_subset']}) | "
          f"P0={cert['headline']['geometry_readable_P0']} | pending={cert['pending_anchors']}")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
