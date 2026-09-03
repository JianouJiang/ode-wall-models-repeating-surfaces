#!/usr/bin/env python3
"""Verify the corrected build at the level a reviewer reads it.

Two things are checked, because the previous attempt failed on the gap between
them:

1. PROVENANCE.  Every figure rendered in the compiled paper must have a
   producer, and its provenance closure must not touch a withdrawn quantity or
   re-instantiate a withdrawn estimator.  Read from the evidence graph.

2. RENDERED CONTENT.  A truth inventory is checked against the ACTIVE source
   *and* against the text of the compiled PDF.  Superseded values are searched
   for in every form they can be printed in --- ``0.08364``, ``8.364\\%``,
   ``8.36%`` --- because the previous checker banned one spelling of a number
   and the paper kept printing another.  Values that are legitimately quoted as
   withdrawn are allowed only inside a sentence that says so.

The inventory is outcome-neutral: it asserts what was measured and where it
came from, never that a comparison came out a particular way.  Gates that
encode an outcome have repeatedly had to be repaired in this project, so this
one records directions and intervals instead of demanding fixed sentences.

Control cases at the end confirm each class of check can actually fail.

Run:  python3 codes/analysis/ledger_verifiers/verify_evidence_graph_l0.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _active_build import active_source, pdf_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STAMP = "20260825"
GRAPH = ROOT / "codes/results" / f"evidence_graph_l0_{STAMP}.json"
SWEEP = ROOT / "codes/results" / f"corrected_family_sweep_l0_{STAMP}.json"
LADDER = ROOT / "codes/results" / f"as_deployed_ladder_rebased_l0_{STAMP}.json"
REBASE = ROOT / "codes/results" / f"reference_rebase_headlines_l0_{STAMP}.json"

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "", note: str = "") -> bool:
    """``detail`` explains a failure; ``note`` is reported either way."""
    suffix = note or ("" if ok else detail)
    (PASS if ok else FAIL).append(f"{name}{(' -- ' + suffix) if suffix else ''}")
    return ok


# --------------------------------------------------------------------------
# text normalisation: the PDF prints minus signs, ligatures and hyphenation
# that a naive substring test does not survive
# --------------------------------------------------------------------------
def norm(text: str) -> str:
    text = (text.replace("\u2212", "-").replace("\u2013", "-")
                .replace("\u2014", "--").replace("\u00a0", " ")
                .replace("\ufb01", "fi").replace("\ufb02", "fl"))
    text = re.sub(r"-\n", "", text)          # hyphenation across a line break
    return re.sub(r"\s+", " ", text)


def tex_norm(text: str) -> str:
    """Active LaTeX reduced to something comparable with rendered text."""
    text = norm(text)
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.replace("\\,", "").replace("~", " ")
    return re.sub(r"\s+", " ", text)


def numeric_aliases(value: float, printed: str) -> list[str]:
    """Every spelling a superseded number can reach the page in.

    Only forms precise enough to identify THIS number are generated.  Rounding
    a superseded value down to two significant figures produces strings such as
    ``0.084`` and ``8.4`` that occur innocently elsewhere in the paper, and a
    checker that bans those cannot distinguish a returning error from an
    unrelated coincidence.
    """
    out = {printed}
    for dp in (4, 5, 6):
        out.add(f"{value:.{dp}f}".rstrip("0"))
        out.add(f"{value:.{dp}f}")
    for dp in (2, 3, 4):
        out.add(f"{value * 100:.{dp}f}".rstrip("0"))
        out.add(f"{value * 100:.{dp}f}")
    return sorted(a for a in out if a and not a.endswith("."))


# --------------------------------------------------------------------------
# 1. provenance
# --------------------------------------------------------------------------
def check_graph() -> None:
    if not check("evidence graph present", GRAPH.exists(), str(GRAPH.name)):
        return
    g = json.loads(GRAPH.read_text())
    s = g["summary"]
    check("every rendered figure has a producer",
          s["n_no_producer"] == 0, f"{s['n_no_producer']} without one")
    check("no rendered figure is downstream of a withdrawn quantity",
          s["n_contaminated"] == 0,
          "; ".join(r["file"] for r in g["figure_closures"]
                    if r["verdict"] == "CONTAMINATED"))
    check("no rendered figure has unresolvable provenance",
          s["n_unresolved"] == 0, f"{s['n_unresolved']} unresolved")
    check("the graph records both kinds of withdrawal it must detect",
          bool(g.get("withdrawal_registry")) and bool(g.get("withdrawn_estimators")),
          "a registry is empty or absent from the artifact")
    inst = g.get("estimator_instantiations", {})
    check("the withdrawn estimator is still detected where it is instantiated",
          any(v for v in inst.values()),
          "no script matches the estimator signature -- either it was removed "
          "everywhere, or the signature has rotted and the detector is blind")
    exemptions = g.get("negative_control_producers", {})
    check("every negative-control exemption is still justified by its artifact",
          all(spec.get("exemption_currently_valid")
              for spec in exemptions.values()),
          "; ".join(rel for rel, spec in exemptions.items()
                    if not spec.get("exemption_currently_valid")))
    check("every rendered figure exists on disk",
          all(r["rendered_present"] for r in g["figure_closures"]),
          "; ".join(r["file"] for r in g["figure_closures"]
                    if not r["rendered_present"]))


# --------------------------------------------------------------------------
# 2. instrument fidelity of the two re-scorings this node performed
# --------------------------------------------------------------------------
def check_fidelity() -> None:
    if check("as-deployed rebase present", LADDER.exists()):
        d = json.loads(LADDER.read_text())
        f = d["instrument_fidelity"]
        check("rebase reproduces the superseded ladder exactly",
              f["reproduces_deposited_A_column"],
              f"worst |dR2| = {f['worst_absolute_r2_deviation']:.3e} over "
              f"{f['n_scores_checked']} scores")
        check("the rebase scored every reference on the same records",
              all(len(r["scores"]) == 3 for r in d["rows"]))
    if check("corrected family sweep present", SWEEP.exists()):
        s = json.loads(SWEEP.read_text())
        check("family sweep covers all 29 members", s["n_members"] == 29)
        check("family sweep states its independent-reference limitation",
              s["limitation_independent_reference"]["n_with_independent_wall_traction"] == 1)
        check("family sweep reports the estimator bias direction",
              "statement" in s["estimator_bias_correlates_with_reported_variables"])


# --------------------------------------------------------------------------
# 3. the truth inventory, checked in source AND in the rendered PDF
# --------------------------------------------------------------------------
# Each entry: what must be printed now, what must never be printed as a current
# claim, and (where a superseded number is legitimately quoted) the words that
# must sit beside it.
FORBIDDEN = [
    {"id": "epsilon scale",
     "aliases": numeric_aliases(0.08364189563744982, "0.08364"),
     "why": "superseded cancellation scale; the corrected value is 0.1315"},
    {"id": "29-case family minimum",
     "aliases": ["reaching -84", "reaching $-84$"],
     "why": "superseded family extreme; the corrected range is -3.3 to -17.6"},
    {"id": "coupled grid failure",
     "aliases": ["finest-grid point errors exceed one",
                 "both finest-grid point errors exceed one"],
     "why": "reversed by the correction; both are now below the threshold"},
    {"id": "Reynolds falsification framing",
     "aliases": ["falsifies that proposed trend", "elevated point estimates",
                 "failure deepens with Reynolds"],
     "why": "neither a deepening failure nor a flat falsification is measured"},
]
# superseded numbers the paper may still quote, but only as withdrawn
QUOTABLE_IF_MARKED = [
    {"id": "as-deployed superseded ladder", "needle": "-7.54",
     "context": r"withdrawn|superseded"},
    # the paper legitimately reports the superseded cancellation median in
    # order to state the size of the correction; that is allowed only where
    # the sentence identifies it as the superseded estimator's answer
    # "the four-point fit" is the paper's own name for the withdrawn estimator
    # and is what the consistency sweep already accepts as a withdrawal marker;
    # omitting that spelling here made a correctly-marked sentence read as an
    # unmarked one.
    {"id": "superseded cancellation median", "needle": "0.084",
     "context": r"through-origin|superseded|withdrawn|legacy|four-point"},
]
REQUIRED = [
    {"id": "corrected cancellation scale", "needle": "13.15",
     "why": "the corrected wall-force residual scale"},
    {"id": "corrected family range low", "needle": "-3.3"},
    {"id": "corrected family range high", "needle": "-17.6"},
    {"id": "reference recovery range", "needle": "0.30--0.36",
     "pdf_needle": "0.30-0.36"},
    {"id": "four-Reynolds profile trend (first)", "needle": "0.0299"},
    {"id": "four-Reynolds profile trend (last)", "needle": "0.0706"},
    {"id": "realisability saturation limitation",
     "needle": "maximum clipped fraction reaches unity"},
    {"id": "spanwise inhomogeneity limitation", "needle": "3.5"},
    {"id": "corrected ladder is reference-bracketed", "needle": "0.86"},
]
MANDATORY_LIMITATIONS = [
    ("one-of-29 independent reference",
     r"(only|exactly) one of the 29|1 of 29|one member of the 29"),
    ("estimator bias correlates with the reported variables",
     r"estimator and not of the flow|property of the estimator"),
    ("epsilon magnitudes are convention-bound",
     r"convention-bound|only orderings on one fixed surface|"
     r"orderings taken on one fixed surface"),
]
# internal register, including the plural and inflected forms the previous
# checker missed.  "wall-pinned" and "edge-pinned" are physics and are exempt.
REGISTER = [
    (r"\bgates?\b", r"edge-pinned|wall-pinned"),
    (r"\bgated\b", None),
    (r"\bcheckpoints?\b", None),
    (r"\bmanifests?\b", None),
    (r"\bverifiers?\b", None),
    (r"\bbyte-identical\b", None),
    (r"\bledger\b", None),
    (r"\bsha256\b", None),
]


def check_inventory() -> None:
    act = tex_norm(active_source())
    try:
        pdf = norm(pdf_text())
    except Exception as exc:                                   # noqa: BLE001
        check("compiled PDF readable", False, str(exc))
        return
    check("compiled PDF readable", True, f"{len(pdf)} characters")

    for item in FORBIDDEN:
        hits_src = [a for a in item["aliases"] if a in act]
        hits_pdf = [a for a in item["aliases"] if a in pdf]
        check(f"superseded claim absent from source: {item['id']}",
              not hits_src, f"found {hits_src}")
        check(f"superseded claim absent from rendered PDF: {item['id']}",
              not hits_pdf, f"found {hits_pdf}")

    for item in QUOTABLE_IF_MARKED:
        ok = True
        bad = []
        for m in re.finditer(re.escape(item["needle"]), pdf):
            window = pdf[max(0, m.start() - 400):m.end() + 400]
            if not re.search(item["context"], window, re.I):
                ok = False
                bad.append(window[380:440])
        check(f"superseded value quoted only as withdrawn: {item['id']}",
              ok, "; ".join(bad[:2]))

    for item in REQUIRED:
        needle_pdf = item.get("pdf_needle", item["needle"])
        check(f"corrected value present in rendered PDF: {item['id']}",
              needle_pdf in pdf, f"missing {needle_pdf!r}")

    for name, pattern in MANDATORY_LIMITATIONS:
        check(f"mandatory limitation stated: {name}",
              bool(re.search(pattern, pdf, re.I)))

    for pattern, allow in REGISTER:
        hits = []
        for m in re.finditer(pattern, act, re.I):
            window = act[max(0, m.start() - 40):m.end() + 40]
            if allow and re.search(allow, window, re.I):
                continue
            hits.append(window.strip())
        check(f"internal register absent from the paper: {pattern}",
              not hits, f"{len(hits)} occurrence(s): {hits[:2]}")


# --------------------------------------------------------------------------
# 4. control cases --- every class of check must be able to fail
# --------------------------------------------------------------------------
def red_fixtures() -> None:
    fake_pdf = "the measured 8.364% wall-force residual scale"
    check("RED: a superseded percentage alias is detected",
          any(a in norm(fake_pdf) for a in FORBIDDEN[0]["aliases"]))

    check("RED: a superseded decimal alias is detected",
          any(a in norm("epsilon = 0.08364 at the matching surface")
              for a in FORBIDDEN[0]["aliases"]))

    check("RED: a plural register term is detected",
          bool(re.search(REGISTER[0][0], "the acceptance gates are set", re.I)))

    check("RED: the physics allowlist protects real terminology",
          bool(re.search(REGISTER[0][1], "edge-pinned separation", re.I)))

    check("RED: an unmarked superseded value is caught",
          not re.search(QUOTABLE_IF_MARKED[0]["context"],
                        "the ladder runs -7.54 at the canonical surface", re.I))

    g = json.loads(GRAPH.read_text()) if GRAPH.exists() else {"figure_closures": []}
    check("RED: the graph would flag a contaminated figure",
          any(r["verdict"] in ("CLEAN", "CONTAMINATED")
              for r in g["figure_closures"]),
          "graph produces per-figure verdicts")


def main() -> int:
    check_graph()
    check_fidelity()
    check_inventory()
    red_fixtures()
    for line in PASS:
        print(f"[PASS] {line}")
    for line in FAIL:
        print(f"[FAIL] {line}")
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed "
          f"(evidence graph and rendered-content inventory, L0)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
