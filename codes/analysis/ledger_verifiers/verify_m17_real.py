#!/usr/bin/env python3
"""Stable-path verifier for claim "M17 (real)" (thesis-grade resolution, section D).

The row's artifact is the verdict map rebuilt from real runs only:
  - generator: codes/figures/fig_verdict_map_real.py
  - cert:      codes/results/verdict_map_real_<date>.json (per-cell provenance)
  - figure:    work_progress/archer2_campaign_20260823/M17_real/fig_verdict_map_real.{pdf,png}

Checks
  1. every TESTED cell binds to a real deposited dataset: the recorded source
     exists and re-hashes to the recorded sha256;
  2. the whole cell roster is RECOMPUTED from the source artifacts (by re-running
     the generator's pure cell builders) and every verdict, metric value,
     interval and provenance hash must match the deposited cert;
  3. every UNTESTED cell carries NO verdict, NO metric and NO provenance -- the
     old maps' hard-coded-badge failure mode is structurally impossible;
  4. the old maps' unbacked geometries (rounded rib, Gaussian bump, BFS,
     conv-div channel, Krank hill) appear as explicit UNTESTED cells;
  5. the rendered figure files exist and hash-match the cert;
  6. control cases: a fabricated verdict cell, a badge painted onto an UNTESTED
     cell, and a tampered metric value must each be rejected.

Usage: python3 codes/analysis/ledger_verifiers/verify_m17_real.py [--cert PATH.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"
sys.path.insert(0, str(ROOT / "codes/analysis"))
sys.path.insert(0, str(ROOT / "codes/figures"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def signature(cells):
    """Material content of the map: cell_id -> everything a reviewer would check."""
    sig = {}
    for c in cells:
        if c["status"] == "TESTED":
            m = c["metric"] or {}
            iv = m.get("interval")
            sig[c["cell_id"]] = (
                "TESTED", c["verdict"], round(float(m["value"]), 9),
                tuple(round(float(v), 9) for v in iv) if iv else None,
                c["provenance"]["source"], c["provenance"]["sha256"])
        else:
            sig[c["cell_id"]] = ("UNTESTED",)
    return sig


def structural_violations(cells):
    """The M17 failure mode: a badge without a run, or a run without provenance."""
    bad = []
    for c in cells:
        if c["status"] == "TESTED":
            prov = c.get("provenance") or {}
            src = ROOT / prov.get("source", "MISSING")
            if c.get("verdict") not in ("FAIL", "MARGINAL", "TOLERATED", "UNRESOLVED"):
                bad.append((c["cell_id"], "tested cell without a legal verdict"))
            if not (c.get("metric") or {}).get("name"):
                bad.append((c["cell_id"], "tested cell without a metric"))
            if not src.exists():
                bad.append((c["cell_id"], f"source missing: {src}"))
            elif digest(src) != prov.get("sha256"):
                bad.append((c["cell_id"], "source hash mismatch"))
        else:
            if c.get("verdict") is not None:
                bad.append((c["cell_id"], "UNTESTED cell carries a verdict badge"))
            if c.get("provenance") is not None:
                bad.append((c["cell_id"], "UNTESTED cell carries provenance"))
            if (c.get("metric") or {}).get("value") is not None:
                bad.append((c["cell_id"], "UNTESTED cell carries a metric value"))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", default=None)
    ap.add_argument("--closed-row-replay", action="store_true")
    a = ap.parse_args()
    certs = sorted(RESULTS.glob("verdict_map_real_*.json"))
    cert_path = Path(a.cert) if a.cert else (certs[-1] if certs else None)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))

    if cert_path is None or not cert_path.exists():
        print("  [FAIL] cert exists")
        print("M17_REAL_VERIFY_FAIL 0/1")
        return 1
    cert = json.loads(cert_path.read_text())
    check("cert schema is verdict_map_real_v1", cert.get("schema") == "verdict_map_real_v1")

    viol = structural_violations(cert["cells"])
    check("every TESTED cell binds to an existing, hash-matching dataset; "
          "every UNTESTED cell is empty (no badge, no metric, no provenance)",
          not viol, "; ".join(f"{i}: {w}" for i, w in viol[:4]))

    fig = importlib.import_module("fig_verdict_map_real")
    fresh = (fig.cells_wavy() + fig.cells_xiao() + fig.cells_r24()
             + fig.cells_m13() + fig.cells_never_run())
    check("cert cells match a full recomputation from the source artifacts",
          signature(fresh) == signature(cert["cells"]))

    legacy_needed = {"legacy::rounded_rib_all", "legacy::gaussian_bump",
                     "legacy::bfs_matched", "legacy::convdiv", "legacy::krank_hill"}
    have = {c["cell_id"] for c in cert["cells"] if c["status"] == "UNTESTED"}
    check("the old maps' unbacked geometries appear as explicit UNTESTED cells",
          legacy_needed <= have)

    counts_ok = (cert["n_cells"] == len(cert["cells"])
                 and cert["n_tested"] == sum(1 for c in cert["cells"] if c["status"] == "TESTED")
                 and cert["n_untested"] == cert["n_cells"] - cert["n_tested"])
    check("cell counts are internally consistent", counts_ok)
    check("PARTIAL status is consistent with pending (non-legacy) untested cells",
          (bool(cert["pending_cells"]) == (cert["status"] == "M17_REAL_PARTIAL"))
          and all(not p.startswith("legacy::") for p in cert["pending_cells"]))

    figs_ok = True
    for k, hk in (("pdf", "pdf_sha256"), ("png", "png_sha256")):
        f = ROOT / cert["figure"][k]
        if not f.exists() or digest(f) != cert["figure"][hk]:
            figs_ok = False
    check("rendered figure files exist and hash-match the cert", figs_ok)

    # ---------------- control cases ----------------
    fab = copy.deepcopy(cert["cells"])
    fab.append(dict(cell_id="cube::aligned_FAKE", family="cube canopy (3-D) -- WRLES",
                    case_label="fabricated", status="TESTED", verdict="TOLERATED",
                    metric=dict(name="a-priori standard_ml R2", value=0.9, interval=[0.8, 1.0]),
                    provenance=dict(source="codes/results/DOES_NOT_EXIST.npz", sha256="0" * 64)))
    check("control case: fabricated cell with a nonexistent dataset is rejected",
          bool(structural_violations(fab)) and signature(fab) != signature(fresh))
    badge = copy.deepcopy(cert["cells"])
    for c in badge:
        if c["cell_id"] == "cube::aligned":
            c["verdict"] = "FAIL"          # a badge painted on an unrun cell
    check("control case: verdict badge painted onto an UNTESTED cell is rejected",
          bool(structural_violations(badge)))
    tamper = copy.deepcopy(cert["cells"])
    for c in tamper:
        if c["cell_id"] == "rib::ktype_p8":
            c["metric"]["value"] = +0.95   # flip the k-type failure into a pass
            c["verdict"] = "TOLERATED"
    check("control case: tampered metric value / flipped verdict is rejected",
          signature(tamper) != signature(fresh))

    n_ok = sum(1 for _, ok in checks if ok)
    line = "M17_REAL_VERIFY_OK" if n_ok == len(checks) else "M17_REAL_VERIFY_FAIL"
    print(f"{line} {n_ok}/{len(checks)} | status={cert['status']} | "
          f"tested={cert['n_tested']} untested={cert['n_untested']} | "
          f"pending={cert['pending_cells']}")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
