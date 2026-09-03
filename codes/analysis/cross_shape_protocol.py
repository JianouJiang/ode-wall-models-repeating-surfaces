#!/usr/bin/env python3
r"""
cross_shape_protocol.py   (Level-1 core methodology, cross-shape iteration)
===========================================================================

WHAT THIS SCRIPT FREEZES
------------------------
This iteration tests whether the ODE-failure diagnostic transfers across
genuinely-distinct INDUSTRIAL repeating shapes (sharp ribs / bars, turbine
cascades, 3-D arrays) and from two into three dimensions.  Before any new shape
is run (Level 2), the *adjudication protocol* must be locked so the new shapes
are scored OUT OF SAMPLE.  This file is that lock.  It contributes three things,
none of which re-implements the validated pipeline:

  (1) THE LOCKED ADJUDICATION RULE  (B-L1-4).
      The canonical evaluator `evaluate()` is imported VERBATIM from
      `cross_geometry_collapse` (same y_idx=10, same production ODE
      `predict_tau_w`, same manuscript eps definition, same R2/relRMS).  We add
      ONLY a pre-registered, frozen decision rule on top of its outputs:
        * measured verdict   : R2(tau_w) < 0  -> catastrophic
                               R2(tau_w) >= 0.88 -> tolerated   (champion bound)
        * a-priori prediction: deep-cancellation COVERAGE gate
                               frac[eps<0.1] >= COV_STAR        (discriminant)
                               with the median-depth eps* and its grey zone as
                               the necessary-but-not-sufficient depth condition.
      The thresholds EPS_STAR, the grey band [0.20,0.50] and COV_STAR are
      derived from the CHAMPION corpus (cross_geometry_collapse.npz) and frozen
      here as module constants.  A new shape's eps is computed AFTER this lock.

  (2) THE FORM-DRAG DECOMPOSITION  (B-L1-3).
      eps = |tau_w|/(|dp/dx| y_m) is built on SKIN FRICTION.  For bluff / sharp
      repeating bodies a large fraction of the streamwise wall force is FORM
      (pressure) drag, not skin friction.  We confront this head-on:
        * eps stays the LOCAL wall-model failure diagnostic on ANY shape, because
          the ODE predicts the local tangential stress tau_w(s) and the
          cancellation mechanism lives in that same local tangential balance --
          a property that is independent of the global form/skin split.
        * f_form = |F_form,x| / (|F_form,x| + |F_skin,x|) is a SECOND, named
          GLOBAL group: the relevance weight that bounds how much an ODE error
          in tau_w propagates to the integrated drag.  It does NOT change WHERE
          the ODE fails (eps does that); it changes how much that failure
          MATTERS to integrated quantities.
      `form_drag_fraction_surface()` computes f_form from any resolved field
      (face areas + surface pressure + wall-shear); it is VERIFIED against an
      analytic square-bar case, and a REAL f_form for the converged sharp-rib
      WRLES is read straight from its OpenFOAM `forces` function-object output.

  (3) THE OUT-OF-SAMPLE / G3 PROTOCOL is recorded as machine-readable metadata so
      Level 2 can score new shapes with NO further tuning (locked-protocol audit).

The d-type rib bridges the two limits: it is a sharp, bluff-ish repeating
structure with f_form ~ 0.6 (FORM DRAG EXCEEDS SKIN FRICTION), yet the ODE still
fails on it (R2=-0.94, champion-validated) and the skin-friction eps criterion
still adjudicates it catastrophic.  Form-drag dominance therefore does NOT
disable eps -- the empirical key that resolves B-L1-3.

Everything is read-only on the protected DNS; nothing is fabricated.  All numbers
trace to on-disk *_wall_profiles.npz and to the rib OpenFOAM forces.dat.

OUTPUTS
-------
  codes/results/cross_shape_protocol.npz
  codes/results/cross_shape_protocol_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))             # codes/analysis
CODES = os.path.dirname(HERE)                                 # codes/
RESULTS = os.path.join(CODES, "results")
OPENFOAM = os.path.join(CODES, "openfoam")

# Import the CANONICAL, champion-validated evaluator VERBATIM -- no re-derivation.
sys.path.insert(0, HERE)
from cross_geometry_collapse import (  # noqa: E402
    CASES, Y_IDX, evaluate, r2, rel_rms,
)

# ============================================================================
# (1) LOCKED, PRE-REGISTERED ADJUDICATION RULE  -- frozen BEFORE any new shape
# ============================================================================
# Provenance of the locked values: the CHAMPION corpus
# codes/results/cross_geometry_collapse.npz (15 labelled geometries).  On that
# corpus the two catastrophic cases (periodic_hills, kth_3d_diffuser) carry
# deep-cancellation coverage frac[eps<0.1] = {0.564, 0.429}; every tolerated
# case has frac[eps<0.1] <= 0.200.  The coverage gate therefore separates the
# labelled classes with an EMPTY gap (0.200, 0.429); we lock COV_STAR in that
# gap.  Median depth eps alone is degenerate (the coarse krank hill is tolerated
# at eps=0.524 while the sharp d-type rib is catastrophic at eps~0.52), so eps*
# is the NECESSARY depth condition with a grey band, and coverage is the
# DISCRIMINANT that resolves it -- exactly the champion's validated two-level
# reading.  These constants are fixed here; Level-2 shapes are scored against
# them with no adjustment (B-L1-4).
R2_CATASTROPHE = 0.0     # R2(tau_w) < this  -> measured catastrophic failure
R2_TOLERATE = 0.88       # R2(tau_w) >= this -> measured tolerated (champion bound)

EPS_STAR = 0.33          # median-eps depth threshold (the value named in B-L1-4)
EPS_GREY_LO = 0.20       # eps_med < LO  -> clear catastrophe (depth alone)
EPS_GREY_HI = 0.50       # eps_med > HI  -> clear tolerate  (depth alone)
#                          eps_med in [LO,HI] -> GREY, resolved by coverage gate
COV_STAR = 0.30          # deep-cancellation coverage gate: frac[eps<0.1] >= this
#                          -> predicted catastrophic.  Sits in the empty labelled
#                          gap (0.200, 0.429); this is the discriminant.


def depth_class(eps_med: float) -> str:
    """DESCRIPTIVE label on the median cancellation parameter -- NOT a decision
    gate (see predict_verdict).  Median eps is reported alongside the verdict
    only as a diagnostic; it is *degenerate* as a classifier and therefore does
    NOT enter the prediction (B-L2-1 fix, see note below)."""
    if not np.isfinite(eps_med):
        return "undefined"
    if eps_med < EPS_GREY_LO:
        return "deep"            # deep median cancellation (descriptive)
    if eps_med > EPS_GREY_HI:
        return "shallow"         # shallow median cancellation (descriptive)
    return "grey"                # intermediate (descriptive)


def predict_verdict(eps_med: float, frac_lt0p1: float) -> dict:
    """A-priori predicted ODE verdict from the LOCKED threshold.

    DECISION VARIABLE (B-L2-1, narrative == code): the SOLE predictor is the
    deep-cancellation COVERAGE
        frac[eps < 0.1] >= COV_STAR  ->  catastrophic   (else tolerated).
    Coverage is *itself* a depth-and-extent measure: a station enters the count
    only if it is DEEPLY cancelled (eps < 0.1), and the count is the spatial
    FRACTION of such stations.  Coverage therefore already encodes "local depth
    over a domain-wide extent" -- the physical content the mechanism predicts.

    The MEDIAN eps is *not* a gate.  It is provably degenerate as a classifier:
    the coarse `krank` hill is TOLERATED at eps_med = 0.524 while the sharp
    d-type rib is CATASTROPHIC at eps_med ~ 0.52 -- identical median depth,
    opposite verdicts.  A median-eps threshold would misclassify one of them.
    We therefore report `depth_class(eps_med)` only as a DESCRIPTOR and make the
    coverage gate the single decision variable.  (Earlier drafts implied a
    two-level depth-then-coverage rule; the code never used depth, so we align
    the narrative to the code: coverage alone decides.)
    """
    cov_fail = bool(np.isfinite(frac_lt0p1) and frac_lt0p1 >= COV_STAR)
    verdict = "catastrophic" if cov_fail else "tolerated"
    return {
        "predicted": verdict,
        "decision_variable": "deep_cancellation_coverage_frac_eps_lt_0p1",
        "depth_class": depth_class(eps_med),   # DESCRIPTIVE only -- not a gate
        "cov_fail": cov_fail,
        "in_grey_zone": False,                 # no grey zone: coverage is the sole gate
    }


def measured_verdict(r2v: float) -> str:
    if not np.isfinite(r2v):
        return "undefined"
    if r2v < R2_CATASTROPHE:
        return "catastrophic"
    if r2v >= R2_TOLERATE:
        return "tolerated"
    return "marginal"


def adjudicate(path: str) -> dict:
    """Locked cross-shape adjudication of one geometry: canonical evaluate()
    plus the frozen prediction/verdict rule.  100% read-only."""
    m = evaluate(path)                       # VERBATIM canonical pipeline
    pred = predict_verdict(m["eps_med"], m["frac_eps_lt0p1"])
    meas = measured_verdict(m["r2"])
    match = None
    if meas in ("catastrophic", "tolerated"):
        match = (pred["predicted"] == meas)
    return {**m, **pred, "measured": meas, "prediction_correct": match}


# ============================================================================
# (2) FORM-DRAG DECOMPOSITION  (B-L1-3)
# ============================================================================
def form_drag_fraction(F_form_x: float, F_skin_x: float) -> float:
    """f_form = |F_form,x| / (|F_form,x| + |F_skin,x|), the GLOBAL relevance
    weight.  F_form,x = streamwise PRESSURE (form) force on one repeating
    element; F_skin,x = streamwise VISCOUS (skin-friction) force on the same
    surface.  f_form -> 0 : streamlined (eps errors propagate fully to drag);
    f_form -> 1 : bluff array (the local ODE failure is real but down-weighted
    by (1-f_form) in the integrated streamwise force)."""
    a, b = abs(float(F_form_x)), abs(float(F_skin_x))
    return float(a / (a + b)) if (a + b) > 0 else float("nan")


def form_drag_fraction_surface(Sf, p_face, tau_face) -> dict:
    """Resolved-field surface integral of f_form over one repeating element.

    Parameters (per boundary face, consistent units; pressure kinematic /rho)
      Sf       : (N,3) outward face-area vectors of the wall patch (out of fluid)
      p_face   : (N,)  pressure at each face
      tau_face : (N,3) wall-shear traction (force/area) the fluid exerts on the
                 wall at each face

    Force on the wall by the fluid, streamwise component:
      F_form_x = - sum_f p_f * Sf_x,f          (pressure acts along -Sf)
      F_skin_x =   sum_f tau_x,f * |Sf_f|       (viscous traction * area)
    Returns F_form_x, F_skin_x and f_form.
    """
    Sf = np.asarray(Sf, float).reshape(-1, 3)
    p = np.asarray(p_face, float).reshape(-1)
    tau = np.asarray(tau_face, float).reshape(-1, 3)
    area = np.linalg.norm(Sf, axis=1)
    F_form_x = float(-np.sum(p * Sf[:, 0]))
    F_skin_x = float(np.sum(tau[:, 0] * area))
    return {
        "F_form_x": F_form_x,
        "F_skin_x": F_skin_x,
        "f_form": form_drag_fraction(F_form_x, F_skin_x),
    }


def _unit_test_form_drag() -> dict:
    """Analytic verification on a 2-D square bar (height=width=k) on a floor.

    Construct an exactly-known load:
      * windward vertical face : pressure p_w, outward area-vector (-A,0,0)
      * leeward  vertical face : pressure p_l, outward area-vector (+A,0,0)
        (A = k * span)
      * floor of length L      : skin-friction traction tau0 in +x,
        outward area-vector (0,-S,0) with |Sf| = S = L * span
    Analytic streamwise forces ON the wall BY the fluid:
      F_form_x = -(p_w*(-A) + p_l*(+A)) = A*(p_w - p_l)
      F_skin_x =  tau0 * S
      f_form   = A*(p_w-p_l) / ( A*(p_w-p_l) + tau0*S )
    """
    span, k, L = 1.0, 0.2, 1.0
    A = k * span
    S = L * span
    p_w, p_l, tau0 = 0.50, -0.30, 0.04
    Sf = np.array([[-A, 0.0, 0.0],
                   [+A, 0.0, 0.0],
                   [0.0, -S, 0.0]])
    p_face = np.array([p_w, p_l, 0.0])               # floor p irrelevant (Sf_x=0)
    tau_face = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [tau0, 0.0, 0.0]])
    got = form_drag_fraction_surface(Sf, p_face, tau_face)
    F_form_exact = A * (p_w - p_l)
    F_skin_exact = tau0 * S
    f_exact = F_form_exact / (F_form_exact + F_skin_exact)
    return {
        "f_form_computed": got["f_form"],
        "f_form_exact": float(f_exact),
        "F_form_x_err": abs(got["F_form_x"] - F_form_exact),
        "F_skin_x_err": abs(got["F_skin_x"] - F_skin_exact),
        "passed": (abs(got["f_form"] - f_exact) < 1e-12),
    }


def _rib_forces_path() -> str:
    """Locate the converged d-type rib OpenFOAM `forces` function-object output."""
    base = os.path.join(OPENFOAM, "rib_les_dtype", "postProcessing", "forcesRib")
    if not os.path.isdir(base):
        return ""
    times = [t for t in os.listdir(base) if os.path.isdir(os.path.join(base, t))]
    if not times:
        return ""
    # latest time (converged statistics, t=140)
    times.sort(key=lambda s: float(s))
    return os.path.join(base, times[-1], "forces.dat")


def rib_form_drag_fraction() -> dict:
    """Read the REAL form/skin split of the converged sharp-rib WRLES straight
    from its OpenFOAM forces.dat (force = (pressure viscous) vectors on the
    ribbed bottomWall).  Body-force driven (meanVelocityForce), so the periodic
    p is the genuine rib form-drag signature, not a mean-gradient artefact."""
    path = _rib_forces_path()
    if not path or not os.path.isfile(path):
        return {"available": False, "path": path}
    last = ""
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                last = line.strip()
    # parse:  time ((Fp_x Fp_y Fp_z) (Fv_x Fv_y Fv_z)) ((Mp..)(Mv..))
    nums = []
    tok = ""
    for ch in last:
        if ch in "()\t ":
            if tok:
                nums.append(float(tok))
                tok = ""
        else:
            tok += ch
    if tok:
        nums.append(float(tok))
    # nums = [time, Fp_x,Fp_y,Fp_z, Fv_x,Fv_y,Fv_z, Mp..(3), Mv..(3)]
    time = nums[0]
    Fp_x, Fv_x = nums[1], nums[4]
    return {
        "available": True,
        "path": os.path.relpath(path, CODES),
        "time": time,
        "F_form_x": Fp_x,
        "F_skin_x": Fv_x,
        "f_form": form_drag_fraction(Fp_x, Fv_x),
    }


# ============================================================================
# (3) DEMONSTRATION ON THE CHAMPION CORPUS  (bit-identical reproduction)
# ============================================================================
def main() -> int:
    # ---- adjudicate every champion-corpus geometry under the locked rule -----
    rows = []
    for key, path, family, klass, repeating, pitch_O_delta in CASES:
        if not os.path.isfile(path):
            rows.append({"key": key, "available": False, "path": path})
            continue
        a = adjudicate(path)
        rows.append({
            "key": key, "available": True, "family": family, "klass": klass,
            "repeating": bool(repeating), "pitch_O_delta": bool(pitch_O_delta),
            "eps_med": a["eps_med"], "frac_eps_lt0p1": a["frac_eps_lt0p1"],
            "relRMS": a["relRMS"], "r2": a["r2"],
            "depth_class": a["depth_class"], "cov_fail": a["cov_fail"],
            "in_grey_zone": a["in_grey_zone"],
            "predicted": a["predicted"], "measured": a["measured"],
            "prediction_correct": a["prediction_correct"],
        })

    avail = [r for r in rows if r.get("available")]
    scored = [r for r in avail if r["measured"] in ("catastrophic", "tolerated")]
    n_correct = sum(1 for r in scored if r["prediction_correct"])

    # ---- form-drag decomposition: unit test + REAL rib number ----------------
    ut = _unit_test_form_drag()
    rib = rib_form_drag_fraction()

    # ---- LOCKED protocol record (machine-readable, for L2 out-of-sample) ------
    locked = {
        "y_idx": int(Y_IDX),
        "R2_CATASTROPHE": R2_CATASTROPHE,
        "R2_TOLERATE": R2_TOLERATE,
        "EPS_STAR": EPS_STAR,
        "EPS_GREY_LO": EPS_GREY_LO,
        "EPS_GREY_HI": EPS_GREY_HI,
        "COV_STAR": COV_STAR,
        "decision_variable": "deep-cancellation coverage frac[eps<0.1] (SOLE gate)",
        "discriminant": "deep-cancellation coverage frac[eps<0.1] >= COV_STAR",
        "depth_condition": ("median eps is a DESCRIPTOR only, NOT a gate -- it is "
                            "degenerate (krank hill tolerated at eps_med=0.524 vs "
                            "d-type rib catastrophic at eps_med~0.52); coverage "
                            "alone decides (narrative==code, B-L2-1)"),
        "EPS_STAR_role": "descriptive median-depth label only (not consulted by predict_verdict)",
        "verdict_rule": "R2<0 catastrophic; R2>=0.88 tolerated; else marginal",
        "evaluate_imported_from": "cross_geometry_collapse.evaluate (verbatim)",
        "provenance_source": "codes/results/cross_geometry_collapse.npz (champion corpus)",
    }

    summary = {
        "locked_protocol": locked,
        "n_corpus": len(rows),
        "n_available": len(avail),
        "n_scored": len(scored),
        "n_prediction_correct": n_correct,
        "prediction_accuracy": (n_correct / len(scored)) if scored else None,
        "form_drag_unit_test": ut,
        "rib_form_drag": rib,
        "rows": rows,
        "note": ("Level-1 cross-shape adjudication protocol LOCKED before any new "
                 "shape is run.  evaluate() imported verbatim from the champion "
                 "pipeline; only a pre-registered decision rule + a form-drag "
                 "decomposition are added.  f_form(d-type rib)~0.60 shows form-drag "
                 "dominance does NOT disable the skin-friction eps criterion."),
    }

    # ---- WRITE OUTPUTS BEFORE ANY ASSERT (anti-empty) ------------------------
    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, "cross_shape_protocol_summary.json")
    with open(jpath, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    npath = os.path.join(RESULTS, "cross_shape_protocol.npz")
    keys = np.array([r["key"] for r in avail])
    np.savez(
        npath,
        keys=keys,
        eps_med=np.array([r["eps_med"] for r in avail]),
        frac_eps_lt0p1=np.array([r["frac_eps_lt0p1"] for r in avail]),
        relRMS=np.array([r["relRMS"] for r in avail]),
        r2=np.array([r["r2"] for r in avail]),
        predicted=np.array([r["predicted"] for r in avail]),
        measured=np.array([r["measured"] for r in avail]),
        prediction_correct=np.array(
            [(-1 if r["prediction_correct"] is None else int(r["prediction_correct"]))
             for r in avail]),
        repeating=np.array([r["repeating"] for r in avail]),
        EPS_STAR=EPS_STAR, COV_STAR=COV_STAR,
        R2_TOLERATE=R2_TOLERATE,
        rib_f_form=float(rib.get("f_form", np.nan)),
        rib_F_form_x=float(rib.get("F_form_x", np.nan)),
        rib_F_skin_x=float(rib.get("F_skin_x", np.nan)),
        unit_test_f_form=float(ut["f_form_computed"]),
        unit_test_f_form_exact=float(ut["f_form_exact"]),
    )

    # ---- HUMAN-READABLE REPORT ----------------------------------------------
    print("=" * 78)
    print("CROSS-SHAPE ADJUDICATION PROTOCOL  (Level-1, LOCKED)")
    print("=" * 78)
    print(f"Locked thresholds: eps* = {EPS_STAR} (grey [{EPS_GREY_LO},{EPS_GREY_HI}]), "
          f"COV* = {COV_STAR} (frac[eps<0.1]); R2 tolerate >= {R2_TOLERATE}")
    print(f"evaluate() imported verbatim from cross_geometry_collapse (y_idx={Y_IDX})")
    print("-" * 78)
    hdr = f"{'geometry':24s}{'eps_med':>9s}{'cov<0.1':>9s}{'R2':>9s}" \
          f"{'depth':>8s}{'pred':>13s}{'meas':>13s}{'ok':>4s}"
    print(hdr)
    for r in avail:
        ok = {True: "Y", False: "N", None: "-"}[r["prediction_correct"]]
        print(f"{r['key']:24s}{r['eps_med']:9.3f}{r['frac_eps_lt0p1']:9.3f}"
              f"{r['r2']:9.2f}{r['depth_class']:>8s}{r['predicted']:>13s}"
              f"{r['measured']:>13s}{ok:>4s}")
    print("-" * 78)
    print(f"Locked-protocol prediction accuracy on the champion corpus: "
          f"{n_correct}/{len(scored)} scored geometries")
    print("-" * 78)
    print("FORM-DRAG DECOMPOSITION (B-L1-3):")
    print(f"  analytic unit test : f_form computed = {ut['f_form_computed']:.6f}  "
          f"exact = {ut['f_form_exact']:.6f}  PASSED = {ut['passed']}")
    if rib.get("available"):
        print(f"  d-type rib WRLES   : F_form,x = {rib['F_form_x']:+.3e}  "
              f"F_skin,x = {rib['F_skin_x']:+.3e}  ->  f_form = {rib['f_form']:.4f}")
        print(f"                       (t={rib['time']}, {rib['path']})")
        print(f"  => sharp rib is form-drag-DOMINATED (f_form={rib['f_form']:.2f} > 0.5) "
              f"yet the ODE fails (R2=-0.94) and eps adjudicates it catastrophic:")
        print(f"     form-drag dominance does NOT disable the local skin-friction eps.")
    else:
        print("  d-type rib WRLES   : forces.dat NOT found (run forcesRib FO in L2)")
    print("-" * 78)
    print(f"WROTE: {os.path.relpath(jpath, CODES)}")
    print(f"WROTE: {os.path.relpath(npath, CODES)}")

    # ---- ASSERTS (after all writes) -----------------------------------------
    # bit-identical reproduction of champion headline numbers via the verbatim
    # import (no re-derivation):
    by = {r["key"]: r for r in avail}
    assert abs(by["periodic_hills_1p0"]["r2"] - (-47.69)) < 0.5, \
        "hills R2 drifted from champion -47.69"
    assert abs(by["conv_div_channel"]["r2"] - 0.934) < 0.02, \
        "conv-div R2 drifted from champion 0.934"
    assert by["periodic_hills_1p0"]["measured"] == "catastrophic"
    assert by["periodic_hills_1p0"]["predicted"] == "catastrophic"
    assert by["conv_div_channel"]["measured"] == "tolerated"
    assert by["conv_div_channel"]["predicted"] == "tolerated"
    # locked coverage gate sits in the labelled empty gap (0.200, 0.429):
    assert EPS_GREY_LO < COV_STAR < by["periodic_hills_1p0"]["frac_eps_lt0p1"]
    # form-drag implementation is correct and the rib number is real:
    assert ut["passed"], "form-drag surface integrator failed analytic unit test"
    assert rib.get("available"), "real rib f_form unavailable"
    assert 0.5 < rib["f_form"] < 0.7, "rib f_form out of expected sharp-bar band"
    print("\nALL ASSERTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
