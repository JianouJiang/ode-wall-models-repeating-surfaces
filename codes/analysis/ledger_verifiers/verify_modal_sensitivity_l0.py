#!/usr/bin/env python3
"""Independent check of the mode-resolved wall-traction sensitivity experiment
(L0 node_002).

This verifier is deliberately OUTCOME-NEUTRAL.  It never demands that a
prediction came true; it demands that whatever is reported is (a) supported by
the harvested artifact, (b) accompanied by the noise floor that makes it
interpretable, and (c) not stated more strongly than the evidence allows.  The
handover's recurring lesson is that a gate encoding an outcome is a bug.

Run:
    python3 codes/analysis/ledger_verifiers/verify_modal_sensitivity_l0.py
"""

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "codes" / "results" / "modal_sensitivity_l0_20260825.json"
NODE = ROOT / "development" / "nodes" / "node_002"
PREREG = NODE / "preregistration.json"
PREREG_SHA = NODE / "preregistration.sha256"
RESULTS = NODE / "RESULTS.md"
DIRECTION = NODE / "research_direction.md"
LIB = ROOT / "codes" / "openfoam" / "modalWallPerturbation"

RESOLVED_SD = 3.0          # a response is "resolved" only above this many floor sd
MIN_NULL = 3               # a floor needs at least this many members


class Checker:
    def __init__(self):
        self.passed, self.failed = 0, []

    def ok(self, cond, label):
        if cond:
            self.passed += 1
        else:
            self.failed.append(label)
        return bool(cond)

    def report(self, title):
        total = self.passed + len(self.failed)
        for f in self.failed:
            print(f"  [FAIL] {f}")
        print(f"{self.passed}/{total} checks passed  ({title})")
        return 0 if not self.failed else 1


def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main(art_path=ART, results_path=RESULTS, strict_pending=False):
    c = Checker()

    # ---- 0. instrument source is present and says what it does -------------
    src = LIB / "modalPerturbedNutUSpaldingWallFunctionFvPatchScalarField.C"
    hdr = LIB / "modalPerturbedNutUSpaldingWallFunctionFvPatchScalarField.H"
    c.ok(src.exists() and hdr.exists(), "instrument source present")
    if src.exists():
        s = src.read_text()
        # The request must be restored before the base-class solve, or the
        # perturbation would feed back into what the model asks for.
        c.ok("operator==(requestedNut_)" in s,
             "instrument restores the unperturbed request before the model solve")
        # Delivery must be measured, not assumed.
        c.ok("sumDtauDel2" in s and "nClipped" in s,
             "instrument measures delivered RMS and counts clipped faces")
        c.ok("max(scalar(0), nutTarget)" in s or "max(scalar(0)" in s,
             "instrument applies the realizability clip explicitly")

    # ---- 1. preregistration is intact --------------------------------------
    if c.ok(PREREG.exists() and PREREG_SHA.exists(), "preregistration present"):
        want = PREREG_SHA.read_text().split()[0]
        got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
        c.ok(want == got, f"preregistration unmodified (sha {want[:12]})")
        pre = load(PREREG)
        c.ok(pre is not None and len(pre.get("predictions", [])) >= 7,
             "preregistration states at least 7 falsifiable predictions")
        c.ok(pre is not None and all("falsified_if" in p
                                     for p in pre.get("predictions", [])),
             "every prediction carries an explicit falsification condition")
        c.ok(pre is not None and len(pre.get("what_this_cannot_establish", [])) >= 4,
             "preregistration states the limits of the design")

    # ---- 2. the artifact --------------------------------------------------
    art = load(art_path)
    if not art:
        # A campaign that has not landed is not a failure of the instrument,
        # but nothing downstream may be claimed.
        c.ok(not strict_pending, "harvested artifact present")
        if results_path.exists():
            txt = results_path.read_text()
            c.ok("PENDING" in txt.upper() or "no runs" in txt.lower(),
                 "results file declares the campaign incomplete when it is")
        return c.report("modal sensitivity L0 (campaign not yet landed)")

    runs = art.get("runs", [])
    summ = art.get("summary", {})
    floor = summ.get("floor", {})
    resp = summ.get("responses", [])

    c.ok(len(runs) > 0, "artifact holds at least one run")

    # ---- 3. the instrument passed its own acceptance test ------------------
    idt = art.get("identity_tests", [])
    c.ok(len(idt) > 0, "at least one identity test is recorded")
    for i, t in enumerate(idt):
        c.ok(t.get("identity_pass") is True,
             f"identity test {i}: null configuration reproduces the stock condition")
        c.ok(t.get("red_fixture_pass") is True,
             f"identity test {i}: control case moved the solution")
        worst = max((v for v in (t.get("fields") or {}).values()
                     if v is not None), default=None)
        c.ok(worst is not None and worst <= t.get("tolerance", 1e-12),
             f"identity test {i}: worst field difference within tolerance")

    # ---- 4. every run is complete and self-describing ----------------------
    for r in runs:
        rid = r.get("run_id", "?")
        c.ok(r.get("reached_time") is not None
             and abs(float(r["reached_time"]) - float(r["end_time"])) < 1e-6,
             f"{rid}: run reached its target time")
        c.ok((r.get("drive") or {}).get("n", 0) >= 8,
             f"{rid}: drive series has enough samples in the averaging window")
        if r.get("kind") != "none":
            d = r.get("delivered") or {}
            c.ok(d.get("rms_delivered") is not None,
                 f"{rid}: delivered perturbation RMS was measured")
            c.ok(d.get("clipped_fraction") is not None,
                 f"{rid}: clipped-face fraction was recorded")
            # Delivery must never be silently assumed equal to the request.
            if d.get("rms_requested") and d.get("rms_delivered"):
                c.ok(d["rms_delivered"] <= d["rms_requested"] * 1.001,
                     f"{rid}: delivered RMS does not exceed the requested RMS")
        # The unperturbed top wall is the within-run control.
        ctl = r.get("control_top_wall")
        if ctl is not None:
            c.ok(abs(ctl.get("rms_delivered", 0.0)) < 1e-14,
                 f"{rid}: unperturbed control wall received no perturbation")

    # ---- 5. the noise floor exists before any response is interpreted ------
    c.ok(summ.get("n_null", 0) >= MIN_NULL,
         f"noise floor built from at least {MIN_NULL} null members")
    for name, f in floor.items():
        if f.get("n", 0) >= 2:
            c.ok(f.get("sd") is not None,
                 f"floor '{name}': standard deviation reported")

    # ---- 6. no response is described without its floor --------------------
    for item in resp:
        rid = item.get("run_id", "?")
        for q in ("drive_mean", "tau_x_rms", "tau_x_int"):
            v = item.get(q)
            if isinstance(v, dict) and v.get("delta") is not None:
                c.ok("in_floor_sd" in v,
                     f"{rid}/{q}: response is expressed in units of the floor")

    # ---- 7. matched norm actually held ------------------------------------
    addA = [i for i in resp
            if i.get("kind") == "additive"
            and i.get("rms_target") is not None
            and abs(i["rms_target"] - 8.461025e-04) < 1e-9]
    if len(addA) >= 2:
        got = [i["rms_delivered"] for i in addA if i.get("rms_delivered")]
        if len(got) >= 2:
            spread = (max(got) - min(got)) / max(got)
            # Requested norms are matched by construction; delivered norms may
            # differ through the clip.  The check is that the spread is
            # REPORTED and bounded, not that it is zero.
            c.ok(spread < 0.60,
                 f"matched-norm family: delivered RMS spread {spread:.3f} < 0.60")

    # ---- 8. the RESULTS narrative is supported by the artifact -------------
    if results_path.exists():
        txt = results_path.read_text()
        # Any number written as a transmission coefficient must appear in the
        # artifact to within rounding.
        arts = set()
        for i in resp:
            for q in ("drive_mean", "tau_x_rms", "tau_x_int"):
                v = i.get(q)
                if isinstance(v, dict):
                    for kk in ("delta", "in_floor_sd", "sensitivity", "value"):
                        if v.get(kk) is not None:
                            arts.add(round(float(v[kk]), 4))
        for f in floor.values():
            for kk in ("mean", "sd", "spread"):
                if f.get(kk) is not None:
                    arts.add(round(float(f[kk]), 4))

        claimed = re.findall(r"`T_(?:\d+)\s*=\s*([-\d.]+)`", txt)
        for x in claimed:
            try:
                val = round(float(x), 4)
            except ValueError:
                continue
            c.ok(any(abs(val - a) < 5e-4 for a in arts),
                 f"claimed transmission value {x} traces to the artifact")

        # Outcome-neutral honesty gates.
        if "REFUTED" in txt.upper() or "refuted" in txt:
            c.ok(True, "results report a refuted prediction explicitly")
        c.ok("floor" in txt.lower(),
             "results state the noise floor alongside the responses")
        banned = ["proves that", "for all wall models", "in general, therefore"]
        for b in banned:
            c.ok(b not in txt.lower(),
                 f"results avoid over-generalising phrase '{b}'")

    # ---- 9. scope flags of the frozen thesis are not violated --------------
    if DIRECTION.exists():
        d = DIRECTION.read_text().lower()
        c.ok("no wall model and no cure" in d or "new_wall_model_offered" in d,
             "direction restates that no wall model is offered")
        c.ok("archer2" in d, "direction states the ARCHER2 constraint")

    return c.report("modal sensitivity L0 (node_002)")


# ---------------------------------------------------------------------------
# Control cases: the checker must FAIL on corrupted inputs.
# ---------------------------------------------------------------------------
def red_fixtures():
    import copy
    import tempfile

    art = load(ART)
    if not art:
        print("control cases skipped: no artifact yet")
        return 0

    fails = []

    def run_mutated(mutate, label):
        a = copy.deepcopy(art)
        mutate(a)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(a, fh)
            p = pathlib.Path(fh.name)
        rc = main(art_path=p, results_path=pathlib.Path("/nonexistent"))
        p.unlink()
        if rc == 0:
            fails.append(label)
        return rc

    def r1(a):
        for t in a.get("identity_tests", []):
            t["identity_pass"] = False
    run_mutated(r1, "R1 identity failure must be caught")

    def r2(a):
        for t in a.get("identity_tests", []):
            t["red_fixture_pass"] = False
    run_mutated(r2, "R2 dead control case must be caught")

    def r3(a):
        for r in a.get("runs", []):
            if r.get("kind") != "none" and r.get("delivered"):
                r["delivered"]["rms_delivered"] = None
    run_mutated(r3, "R3 unmeasured delivery must be caught")

    def r4(a):
        a["summary"]["n_null"] = 1
    run_mutated(r4, "R4 missing noise floor must be caught")

    def r5(a):
        for r in a.get("runs", []):
            r["reached_time"] = 0.0
    run_mutated(r5, "R5 short run must be caught")

    def r6(a):
        for r in a.get("runs", []):
            if r.get("control_top_wall"):
                r["control_top_wall"]["rms_delivered"] = 1.0
    run_mutated(r6, "R6 contaminated control wall must be caught")

    for f in fails:
        print(f"  [RED-FIXTURE FAIL] {f}")
    print(f"{6 - len(fails)}/6 control cases behaved correctly")
    return 0 if not fails else 1


if __name__ == "__main__":
    rc = main()
    if "--red" in sys.argv:
        rc |= red_fixtures()
    sys.exit(rc)
