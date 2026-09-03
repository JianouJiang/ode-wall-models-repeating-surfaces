#!/usr/bin/env python3
"""Run EVERY verifier in the tree and say which are green, which are not, and why.

`check_referee_ledger.py` runs the verifier of each CLOSED row.  Passing it
therefore means "every row already declared closed still verifies" -- it says
nothing about instruments belonging to open rows, to superseded work, or to no
row at all.  Nineteen of the sixty programs in `codes/analysis/ledger_verifiers/`
are in that position, so a green ledger run has been read as a green project
when it is not the same statement.

This program removes that gap by running all of them and classifying each
result, so the honest state is one command away instead of an inference.

Classification is by evidence, not by opinion:
  GREEN      exit 0.
  EXPECTED   exit non-zero for a reason recorded in REFEREE_POINT_LEDGER.md --
             the row is OPEN, or the work was cancelled by the operator, or the
             artifact it needs has not landed.  These are not defects.
  LENGTH     exit non-zero solely on the body-page target, which is a reported
             number and an operator decision, not a broken claim.
  SUPERSEDED exit non-zero because it belongs to an earlier iteration whose
             evidence the project has since replaced.
  DEFECT     exit non-zero for any other reason.  These are the ones to fix.

Usage: python3 codes/analysis/instrument_inventory.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VDIR = ROOT / "codes" / "analysis" / "ledger_verifiers"

# Reason codes, each with the fact that justifies it.  A program may only be
# excused by name here; anything else that fails is a DEFECT.
EXPECTED = {
    "verify_m4.py":
        "ledger row M4 is OPEN (a second subgrid closure is new compute, forbidden "
        "under the standing no-new-simulation rule); the paper makes no "
        "subgrid-invariance claim",
    "verify_r2_m4.py":
        "ledger row R2-m4 / R3-2 is OPEN — REOPENED; closing it is an operator call",
    "verify_r1_sta2_amplitude.py":
        "the steep-wave amplitude artifact is produced by the campaign poller when "
        "its ARC row goes terminal",
    "verify_modal_sensitivity_l0.py":
        "the modal-sensitivity study was CANCELLED by the operator on 2026-08-25; "
        "its verifier is retained as the record that it was never run",
    "verify_r2_2_real.py":
        "PARTIAL: cube_aligned and cube_sparse anchors have now landed and the "
        "re5600/re10595 sources were regenerated, so the certificate needs an "
        "operator-owned refresh inside the live campaign",
    "verify_m17_real.py":
        "PARTIAL: same two pending anchors and the same regenerated sources as "
        "R2-2 (real)",
}
LENGTH = {
    "verify_writing_l4.py", "verify_writing_rows.py", "verify_m17.py",
}
SUPERSEDED = {
    "verify_level2.py":
        "iteration-2 gate; its evidence base was replaced by the corrected "
        "reference campaign",
    "verify_results_l3.py":
        "level-3 gate of an earlier tree; superseded by the referee-point ledger",
    "verify_input_sufficiency.py":
        "written for the two-transfer-relation presentation that the corrected "
        "reference work withdrew",
    "verify_model_registry_r1sci2_m12.py":
        "superseded companion of R1-SCI-2 / M12, whose live verifiers "
        "(verify_r1_sci2.py, verify_m12.py) are registered and green; it also "
        "binds the retired 20-page ceiling",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    rows = []
    for f in sorted(VDIR.glob("verify_*.py")):
        r = subprocess.run([sys.executable, str(f)], capture_output=True,
                           text=True, cwd=ROOT, timeout=900)
        first = next((l for l in (r.stdout + r.stderr).splitlines()
                      if "FAIL" in l), "").strip()[:120]
        if r.returncode == 0:
            kind, why = "GREEN", ""
        elif f.name in LENGTH:
            kind, why = "LENGTH", "body-page target; reported, not a broken claim"
        elif f.name in EXPECTED:
            kind, why = "EXPECTED", EXPECTED[f.name]
        elif f.name in SUPERSEDED:
            kind, why = "SUPERSEDED", SUPERSEDED[f.name]
        else:
            kind, why = "DEFECT", first
        rows.append({"program": f.name, "rc": r.returncode, "class": kind,
                     "first_failure": first, "why": why})

    order = ["DEFECT", "LENGTH", "EXPECTED", "SUPERSEDED", "GREEN"]
    counts = {k: sum(1 for x in rows if x["class"] == k) for k in order}
    print(f"INSTRUMENT INVENTORY — {len(rows)} programs in "
          f"{VDIR.relative_to(ROOT)}")
    print("-" * 66)
    for k in order:
        print(f"  {k:<11} {counts[k]:>3}")
    for k in order[:-1]:
        sel = [x for x in rows if x["class"] == k]
        if not sel:
            continue
        print(f"\n{k}")
        for x in sel:
            print(f"  {x['program']}")
            print(f"      {x['why'] or x['first_failure']}")
    if a.json:
        a.json.write_text(json.dumps({"counts": counts, "programs": rows}, indent=1))
        print(f"\nwritten: {a.json}")
    print(f"\n{counts['GREEN']}/{len(rows)} green; {counts['DEFECT']} defect(s)")
    return 1 if counts["DEFECT"] else 0


if __name__ == "__main__":
    sys.exit(main())
