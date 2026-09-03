#!/usr/bin/env python3
"""verify_model_matched_transfer_l2.py -- independent check of the node_010 instrument.

This verifier exists because the previous one had holes that let four real
defects through.  It therefore does two things that the previous one did not:

  1. It RECOMPUTES the load-bearing quantities from the deposited campaign and
     from the case dictionaries, rather than re-deriving a statistic from the
     same column the producer wrote.
  2. It carries RED FIXTURES: synthetic summaries that contain exactly the four
     defects the panel found.  A verifier that does not REJECT every red
     fixture is itself failing, and this script reports that as a failure.

Checks
------
A. model provenance -- every row's a-priori member is the ladder member of the
   architecture named in the case's own ``input/nut`` dictionary, re-read here.
B. a-priori columns are NOT byte-identical across the two models.
C. commensurate estimands -- station, grid and averaging-window sensitivities
   are all differences of E_tau; a mixed-normalisation ratio is rejected.
D. grid invariance is EQUALITY OF DECISION SIDES, ``(a>1) == (b>1)``, not
   ``a>1 and b>1``; a mixed pair asserted invariant is rejected.
E. case-correct critical height -- the ``periodic_hills_1p0`` row, never
   ``krank_pehill_Re10595``.
F. drive-gate provenance -- the registered single-slice 3% check is recorded
   with its true outcome, no post-outcome ceiling is applied as a pass rule,
   and no text calls a post-outcome amendment prospective.
G. the abort is described as one observed trajectory failure, not a located
   boundary and not a proof that no wall stress exists.

Usage:  python3 verify_model_matched_transfer_l2.py [summary.json]
Exit 0 only if every check passes AND every control case is rejected.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CMH_MAP = ROOT / "codes" / "results" / "critical_matching_height_map.npz"
CASE_ROW = "periodic_hills_1p0"
FOREIGN_ROW = "krank_pehill_Re10595"
DEPLOYED_BC = {"equilibrium": "nutUSpaldingWallFunction",
               "total_gradient_tble": "totalGradientTbleNut"}
APRIORI_MEMBER_OF = {"equilibrium": "M0_equilibrium",
                     "total_gradient_tble": "M1_pressure_gradient_ode"}
BANNED_PROSPECTIVE = re.compile(
    r"no (numerical )?threshold changed|registered before (its|the) outcome|"
    r"pre-?outcome amendment|prospective(ly)? registered ceiling", re.I)


def read_bc(case: Path, patch: str = "bottomWall") -> str | None:
    f = case / "input" / "nut"
    if not f.is_file():
        return None
    m = re.search(rf"^\s*{patch}\s*$\s*\{{\s*type\s+(\w+)\s*;",
                  f.read_text(errors="replace"), re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
def run_checks(s: dict, *, reread_cases: bool) -> list[tuple[str, bool, str]]:
    """Return [(name, ok, note)].  ``reread_cases`` is disabled for fixtures."""
    out: list[tuple[str, bool, str]] = []

    def c(name, ok, note=""):
        out.append((name, bool(ok), note))

    pts = {k: v for k, v in s.get("points", {}).items() if "model_abort" not in v}

    # --- A. model provenance -------------------------------------------------
    for k, v in pts.items():
        model = v.get("model")
        want_bc = DEPLOYED_BC.get(model)
        want_member = APRIORI_MEMBER_OF.get(model)
        c(f"A {k}: a-priori member is the deployed architecture",
          v.get("apriori_member_matched") == want_member
          and v.get("deployed_nut_bc") == want_bc,
          f"member={v.get('apriori_member_matched')} bc={v.get('deployed_nut_bc')}")
        if reread_cases and v.get("case_dir"):
            case = ROOT / v["case_dir"]
            bc = read_bc(case)
            c(f"A {k}: nut BC re-read from the case dictionary agrees",
              bc is not None and bc == want_bc, f"re-read {bc}")

    # --- B. the a-priori columns must differ between models ------------------
    cols = {}
    for m, t in (s.get("transfer_relation_model_matched") or {}).items():
        cols[m] = [round(x, 12) if isinstance(x, (int, float)) else x
                   for x in t.get("apriori_matched_relative_rms", [])]
    ms = sorted(cols)
    if len(ms) == 2 and cols[ms[0]] and len(cols[ms[0]]) == len(cols[ms[1]]):
        c("B a-priori columns are model-specific (not byte-identical)",
          cols[ms[0]] != cols[ms[1]],
          f"{ms[0]}[0]={cols[ms[0]][0]} {ms[1]}[0]={cols[ms[1]][0]}")
    else:
        c("B a-priori columns are model-specific (not byte-identical)", False,
          "columns absent or unequal length")

    # the producer must also DISCLOSE that the superseded column was model-blind
    aud = s.get("deposited_column_audit") or {}
    c("B deposited model-blind column is disclosed and superseded",
      aud.get("byte_identical_across_models") is True
      and "superseded" in str(aud.get("consequence", "")).lower())

    # --- C. commensurate estimands ------------------------------------------
    for k, v in pts.items():
        ws = v.get("window_sensitivity") or {}
        est = str(ws.get("estimand", ""))
        c(f"C {k}: window sensitivity is an E_tau difference",
          "E_tau" in est and "RMS(tau_model - tau_DNS)/RMS(tau_DNS)" in est
          and len(ws.get("per_checkpoint", {})) == 3, est[:60])
        # a curve-difference norm normalised by the LATEST curve is the defect
        c(f"C {k}: no curve-difference normalisation in the window estimand",
          "tau_270 - tau_225" not in est and "RMS(tau_270)" not in est)
    for m, sv in (s.get("sensitivity_common_scale") or {}).items():
        c(f"C {m}: the three sensitivities share one estimand",
          "differences of E_tau" in str(sv.get("estimand", "")))
        # ratio must be recomputable from the two E_tau numbers it claims
        span, win = sv.get("station_span_Etau"), sv.get("max_abs_window_change_Etau")
        r = sv.get("station_over_window")
        c(f"C {m}: station/window ratio recomputes from its own E_tau operands",
          all(isinstance(x, (int, float)) for x in (span, win, r)) and win > 0
          and abs(r - span / win) < 1e-9, f"{r} vs {None if not win else span / win}")
        c(f"C {m}: grid-ratio scope names the excluded abort",
          "abort" in str(sv.get("scope", "")).lower())

    # --- D. grid invariance = equality of decision sides ---------------------
    gi = s.get("grid_invariance_common_scale") or {}
    c("D grid-invariance block present", bool(gi))
    for k, g in gi.items():
        a, b = g.get("G1c_relative_rms"), g.get("G2c_relative_rms")
        if a is None or b is None:
            c(f"D {k}: unscored pair carries no fabricated verdict",
              g.get("verdict_sides_equal") is None)
            continue
        want = (float(a) > 1.0) == (float(b) > 1.0)
        c(f"D {k}: stored verdict equals recomputed (a>1)==(b>1)",
          g.get("verdict_sides_equal") is want,
          f"stored={g.get('verdict_sides_equal')} recomputed={want} a={a} b={b}")
        # and the buggy conjunction must not be what was stored
        buggy = bool(float(a) > 1.0 and float(b) > 1.0)
        if buggy != want:
            c(f"D {k}: verdict is not the a>1 and b>1 conjunction",
              g.get("verdict_sides_equal") != buggy)

    # --- E. case-correct critical height ------------------------------------
    ch = s.get("critical_matching_height") or {}
    c("E critical height uses the case's own row",
      ch.get("row_used") == CASE_ROW and ch.get("row_rejected") == FOREIGN_ROW,
      str(ch.get("row_used")))
    if CMH_MAP.is_file() and reread_cases:
        d = np.load(CMH_MAP, allow_pickle=True)
        rows = [str(x) for x in d["keys"]]
        yc = float(d["ycrit"][rows.index(CASE_ROW)])
        c("E y_crit+ re-read from the map agrees with the row used",
          isinstance(ch.get("ycrit_plus"), (int, float))
          and abs(ch["ycrit_plus"] - yc) < 1e-9, f"map={yc} stored={ch.get('ycrit_plus')}")
        sweep = np.asarray(d[f"sweep_relrms__{CASE_ROW}"], float)
        c("E interior-crossing flag agrees with the swept a-priori curve",
          ch.get("interior_crossing") is bool(np.any(sweep < 1.0)))
    # the foreign value must not be reused anywhere as this case's threshold
    blob = json.dumps(s)
    c("E foreign y_crit+ 15.9259 is not used as this case's threshold",
      "15.925931" not in blob.replace(str(ch.get("ycrit_plus_rejected", "")), "", 1)
      or ch.get("ycrit_plus_rejected") is not None)
    c("E no interior crossing is claimed for this case",
      ch.get("interior_crossing") is False
      and ch.get("all_sampled_surfaces_beyond_ycrit") is True)

    # --- F. drive-gate provenance -------------------------------------------
    roll = s.get("drive_gate_rollup") or {}
    gate = roll.get("registered_3pct_slice_gate") or {}
    c("F registered 3% single-slice gate outcomes are recorded",
      isinstance(gate.get("FAIL"), list) and isinstance(gate.get("PASS"), list))
    c("F the registered gate is reported as FAILED where it failed",
      len(gate.get("FAIL", [])) > 0
      and "FAILED" in str(roll.get("statement", "")).upper())
    c("F no replacement ceiling is applied as a pass rule",
      "12%" not in str(roll.get("statement", ""))
      and "no replacement ceiling" in str(roll.get("statement", "")).lower())
    c("F slice fluxes carry no pass/fail authority",
      all("no pass/fail authority" in str(
          (v.get("drive") or {}).get("slice_flux_homogeneity_diagnostic", {}).get("authority", ""))
          or "none" in str((v.get("drive") or {}).get(
              "slice_flux_homogeneity_diagnostic", {}).get("authority", "")).lower()
          for v in pts.values()))
    c("F admission rests on the exact controller/volume identity",
      not roll.get("exact_identity_fails"))
    c("F no false prospective-registration language",
      BANNED_PROSPECTIVE.search(blob) is None,
      (BANNED_PROSPECTIVE.search(blob).group(0) if BANNED_PROSPECTIVE.search(blob) else ""))

    # --- G. the abort is not over-read --------------------------------------
    aborts = {k: v for k, v in s.get("points", {}).items() if "model_abort" in v}
    for k, v in aborts.items():
        note = str(v.get("note", "")).lower()
        c(f"G {k}: abort framed as one observed trajectory failure",
          "not a located operability boundary" in note
          and "no wall stress exists" in note and "not a proof" in note)
    c("G no 'no wall stress exists' claim outside the disclaimer",
      blob.count("no wall stress exists") == len(aborts))

    return out


# ---------------------------------------------------------------------------
# H. the manuscript's PRINTED values must equal the measured ones.
# The node_009 reviewer asked for exactly this: "extend the verifier so it checks
# the manuscript's printed intervals/ratios".  Raw grep would read the \iffalse
# blocks, so the compiled PDF text is used.
# ---------------------------------------------------------------------------
MAIN_PDF = ROOT / "manuscript" / "main.pdf"


def manuscript_text() -> str | None:
    if not MAIN_PDF.is_file():
        return None
    import subprocess
    try:
        r = subprocess.run(["pdftotext", "-layout", str(MAIN_PDF), "-"],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return " ".join(r.stdout.split())


def check_manuscript(s: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    txt = manuscript_text()
    if txt is None:
        out.append(("H manuscript PDF readable", False, "main.pdf absent or pdftotext failed"))
        return out
    tr = s["transfer_relation_model_matched"]
    sens = s["sensitivity_common_scale"]

    def near(printed: str, value: float, tol: float) -> bool:
        return abs(float(printed) - float(value)) <= tol

    # the two transfer statistics
    tb, eq = tr["total_gradient_tble"], tr["equilibrium"]
    out.append(("H TBLE transfer rho printed as measured",
                (r"\rho = +1" in txt or "= +1," in txt or "+1, p = 0.003" in txt)
                and abs(tb["spearman_coupled_vs_apriori_MATCHED"] - 1.0) < 1e-9,
                f"measured {tb['spearman_coupled_vs_apriori_MATCHED']}"))
    out.append(("H TBLE transfer p printed as measured",
                f"{tb['spearman_coupled_vs_apriori_MATCHED_permutation_p']:.3f}" in txt,
                f"looking for {tb['spearman_coupled_vs_apriori_MATCHED_permutation_p']:.3f}"))
    out.append(("H equilibrium transfer rho printed as measured",
                f"{eq['spearman_coupled_vs_apriori_MATCHED']:.2f}".lstrip("-") in txt,
                f"measured {eq['spearman_coupled_vs_apriori_MATCHED']:.2f}"))
    out.append(("H equilibrium transfer p printed as measured",
                f"{eq['spearman_coupled_vs_apriori_MATCHED_permutation_p']:.2f}" in txt,
                f"measured {eq['spearman_coupled_vs_apriori_MATCHED_permutation_p']:.2f}"))
    # the four sensitivity ratios, rounded as printed
    for m, key, label in (("equilibrium", "station_over_window", "E_tau eq"),
                          ("total_gradient_tble", "station_over_window", "E_tau tble"),
                          ("equilibrium", "station_over_window_R2", "R2 eq"),
                          ("total_gradient_tble", "station_over_window_R2", "R2 tble")):
        v = sens[m][key]
        out.append((f"H sensitivity ratio printed as measured ({label})",
                    f"{round(v)}" in txt, f"measured {v:.2f} -> printed {round(v)}"))
    # the superseded values must NOT survive in the compiled body
    for bad, why in (("66 times", "the mixed-normalisation window ratio"),
                     ("p = 0.017", "the wrong-architecture transfer p"),
                     ("15.9259", "the foreign krank critical height"),
                     ("No wall stress exists", "the over-read abort claim")):
        out.append((f"H superseded value absent from the compiled body: {why}",
                    bad not in txt, f"found '{bad}'" if bad in txt else ""))
    return out


# ---------------------------------------------------------------------------
# RED FIXTURES -- each injects one of the four panel-found defects.
# The verifier is required to REJECT every one of them.
# ---------------------------------------------------------------------------
def red_fixtures(good: dict) -> list[tuple[str, dict]]:
    fx = []

    # 1. model-blind a-priori column: copy the TBLE sweep into equilibrium
    f = copy.deepcopy(good)
    t = f["transfer_relation_model_matched"]
    t["equilibrium"]["apriori_matched_relative_rms"] = list(
        t["total_gradient_tble"]["apriori_matched_relative_rms"])
    fx.append(("model-blind a-priori column duplicated across models", f))

    # 2. wrong architecture named for the equilibrium arm
    f = copy.deepcopy(good)
    for k, v in f["points"].items():
        if v.get("model") == "equilibrium":
            v["apriori_member_matched"] = "M1_pressure_gradient_ode"
    fx.append(("equilibrium row paired with the pressure-gradient a-priori model", f))

    # 3. mixed-normalisation window sensitivity
    f = copy.deepcopy(good)
    for v in f["points"].values():
        if "window_sensitivity" in v:
            v["window_sensitivity"]["estimand"] = "RMS(tau_270 - tau_225)/RMS(tau_270)"
    fx.append(("window sensitivity given as a curve-difference norm", f))

    # 4. mixed grid pair asserted invariant (the a>1 and b>1 hole)
    f = copy.deepcopy(good)
    gi = f.setdefault("grid_invariance_common_scale", {})
    gi["FIXTURE:mixed"] = {"G1c_relative_rms": 1.5, "G2c_relative_rms": 0.5,
                           "verdict_sides_equal": True}
    fx.append(("mixed grid pair (1.5, 0.5) asserted verdict-invariant", f))

    # 4b. a both-tolerated pair scored with the buggy conjunction
    f = copy.deepcopy(good)
    gi = f.setdefault("grid_invariance_common_scale", {})
    gi["FIXTURE:both_below"] = {"G1c_relative_rms": 0.5, "G2c_relative_rms": 0.5,
                                "verdict_sides_equal": False}   # a>1 and b>1
    fx.append(("both-tolerated pair scored with the a>1 and b>1 conjunction", f))

    # 5. foreign critical height reinstated
    f = copy.deepcopy(good)
    f["critical_matching_height"]["row_used"] = FOREIGN_ROW
    f["critical_matching_height"]["ycrit_plus"] = 15.925931
    fx.append(("foreign krank y_crit+ = 15.93 used for the Xiao case", f))

    # 6. post-outcome ceiling reinstated as a pass rule + prospective language
    f = copy.deepcopy(good)
    f["drive_gate_rollup"]["statement"] = (
        "all points pass: values outside 3% are admitted below the 12% ceiling; "
        "no numerical threshold changed and the amendment was registered before its outcome")
    fx.append(("post-outcome 12% ceiling reinstated with prospective language", f))

    # 7. abort over-read as a located boundary
    f = copy.deepcopy(good)
    for v in f["points"].values():
        if "model_abort" in v:
            v["note"] = "locates the upper operability boundary; no wall stress exists there"
    fx.append(("abort over-read as a located operability boundary", f))

    return fx


def main() -> int:
    default = sorted((ROOT / "codes" / "results").glob("model_matched_transfer_l2_*_summary.json"))
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (default[-1] if default else None)
    if path is None or not path.is_file():
        print("no model_matched_transfer_l2 summary found")
        return 2
    s = json.loads(path.read_text())
    print(f"verifying {path.relative_to(ROOT)}\n")

    results = run_checks(s, reread_cases=True) + check_manuscript(s)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, note in results:
        if not ok:
            print(f"  [FAIL] {name}   {note}")
    print(f"real summary: {n_ok}/{len(results)} checks passed")

    print("\ncontrol cases (each MUST be rejected):")
    fixture_ok = True
    for label, f in red_fixtures(s):
        r = run_checks(f, reread_cases=False)
        failures = [n for n, ok, _ in r if not ok]
        rejected = bool(failures)
        print(f"  [{'REJECTED' if rejected else 'ACCEPTED -- VERIFIER HOLE'}] {label}"
              + (f"  (tripped {len(failures)} check(s), first: {failures[0]})" if rejected else ""))
        fixture_ok &= rejected

    passed = (n_ok == len(results)) and fixture_ok
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}  "
          f"({n_ok}/{len(results)} checks, control cases {'all rejected' if fixture_ok else 'LEAKED'})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
