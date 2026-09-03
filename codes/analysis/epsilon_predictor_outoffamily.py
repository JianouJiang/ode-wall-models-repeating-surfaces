#!/usr/bin/env python3
r"""
epsilon_predictor_outoffamily.py -- R2-2 (real): the honest, pre-registered
out-of-family test of epsilon as a PREDICTOR of ODE wall-model failure.
===========================================================================

History.  The the earlier submission build claimed a geometry-readable failure criterion
(hat-epsilon + classifier, AUC=1.00) and referee point R2-2 destroyed it:
parameter drift across 15 locations, two contradictions, hard-assigned
infinities, within-family-only evidence.  The JCP build deleted the claim.
This module runs the experiment the deleted claim never had: ONE fixed
definition of epsilon, ONE evaluation-surface convention, thresholds fixed in
advance in an immutable preregistration
(work_progress/archer2_campaign_20260823/R2-2_real/
 preregistration_r2_2_real_20260824.json, sha256-bound into the cert),
then a mechanical comparison of forecast vs measured verdict on out-of-family
high-fidelity anchors, with the verdict stated plainly either way:
PREDICTOR / DESCRIPTIVE_ONLY / MIXED.

The module NEVER computes a new flow quantity: every epsilon and every R^2 is
read from the campaign's deposited harvest artifacts (dose_response_xiao.npz,
r1_sta2_wavy_wrles_<date>.{json,npz}, r2_4_m20_les_<date>.{json,npz},
m13_highre_coupled_<date>_summary.json), so the numbers scored here are
byte-identical to the numbers those rows deposited.  It only applies the
preregistered rules.

PARTIAL mode: anchors whose deposits have not landed are reported PENDING and
the cert status is R2_2_REAL_PARTIAL.  Re-run (identical command) when the
poller lands new deposits; the preregistration is immutable, so a re-run may
only ADD anchors.

Usage:  python3 codes/analysis/epsilon_predictor_outoffamily.py [--date 20260824]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
PREREG = ROOT / ("work_progress/archer2_campaign_20260823/R2-2_real/"
                 "preregistration_r2_2_real_20260824.json")
AMENDMENTS = sorted((ROOT / "work_progress/archer2_campaign_20260823/R2-2_real")
                    .glob("amendment_*.json"))

# --------------------------------------------------------------------------
# Truth-reference registry (amendment AMD-01, 2026-08-25).
# A coupled anchor may NEVER be scored silently against a withdrawn reference:
# the evaluator classifies each candidate harvest's truth file, scores against a
# VALID one, and retains the withdrawn score verbatim under the amendment.
# --------------------------------------------------------------------------
WITHDRAWN_TRUTHS = {
    "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz": dict(
        amendment="AMD-01",
        reason=("wall traction RECONSTRUCTED from the public Xiao velocity archive; "
                "fails four independent tests (disagrees with two mutually-consistent "
                "DNS at the same Re, separation at x/H=0.379 vs 0.181, inverts the "
                "C_f(Re) ordering, geometry bit-identical) -- withdrawn by Agent B, "
                "audit codes/results/m13_truth_reference_audit_20260825.json"),
    ),
}


def classify_truth(truth: dict) -> dict:
    """VALID / WITHDRAWN verdict on a harvest's declared truth reference."""
    f = (truth or {}).get("file", "")
    w = WITHDRAWN_TRUTHS.get(f)
    return dict(file=f, definition=(truth or {}).get("definition"),
                reference=(truth or {}).get("reference"),
                status="WITHDRAWN" if w else "VALID",
                withdrawal=w)

EPS_CRIT = 0.5          # P1 threshold        (preregistered)
P2_FRAC = 0.2           # P2 deep-cancellation coverage threshold (preregistered)
P2_PITCH = 3.0          # P2 dense-coverage leg: pitch <= 3*delta (preregistered)
BINS = ("FAIL", "MARGINAL", "TOLERATED")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r2_bin(v: float) -> str:
    if not np.isfinite(v):
        return "INVALID"
    if v < 0.0:
        return "FAIL"
    if v < 0.5:
        return "MARGINAL"
    return "TOLERATED"


def interval_bin(lo: float, hi: float) -> str | None:
    """Single bin containing the whole interval, else None (unresolved)."""
    blo, bhi = r2_bin(lo), r2_bin(hi)
    return blo if blo == bhi else None


def p1_forecast(eps_median: float) -> str:
    return "FAIL" if eps_median < EPS_CRIT else "TOLERATED"


def p2_forecast(frac_lt0p1: float, pitch_over_delta: float) -> str:
    return ("FAIL" if (frac_lt0p1 >= P2_FRAC and pitch_over_delta <= P2_PITCH)
            else "TOLERATED")


def score(forecast: str, measured: str) -> float:
    if forecast == measured:
        return 1.0
    if {forecast, measured} == {"FAIL", "TOLERATED"}:
        return 0.0
    return 0.5   # adjacent bins


def latest(pattern: str) -> Path | None:
    hits = sorted(RESULTS.glob(pattern))
    return hits[-1] if hits else None


# --------------------------------------------------------------------------
# anchor evaluators -- each returns a cert row (dict) + npz arrays (dict)
# --------------------------------------------------------------------------
def eval_xiao29():
    src = RESULTS / "dose_response_xiao.npz"
    if not src.exists():
        return None, {}
    d = np.load(src, allow_pickle=True)
    rows, arrays = [], {}
    for i, case in enumerate(d["agg_case"]):
        eps = float(d["agg_eps_median"][i]); r2 = float(d["agg_r2"][i])
        measured = r2_bin(r2)
        resolved = abs(r2) > 1.0            # prereg fallback: no archived interval
        fc = p1_forecast(eps)
        rows.append(dict(
            member=str(case), eps_median=eps, frac_eps_lt0p1=float(d["agg_frac_eps_lt_0p1"][i]),
            pitch_over_delta=float(d["agg_cv_ellp_over_delta"][i]),
            measured_r2=r2, measured_bin=measured, resolved=bool(resolved),
            resolution_basis="point value, no archived interval; accepted because |R2|>1",
            p1_forecast=fc, p1_score=(score(fc, measured) if resolved else None),
            p2_forecast=p2_forecast(float(d["agg_frac_eps_lt_0p1"][i]),
                                    float(d["agg_cv_ellp_over_delta"][i])),
        ))
    arrays["xiao29_eps_median"] = np.asarray(d["agg_eps_median"], float)
    arrays["xiao29_r2"] = np.asarray(d["agg_r2"], float)
    row = dict(
        id="xiao29_family", status="EVALUATED", blind=False, in_family=True,
        role="in-family calibration set (excluded from headline by preregistration)",
        n_members=len(rows), members=rows,
        summary=dict(
            n_p1_correct=sum(1 for r in rows if r["p1_score"] == 1.0),
            n_resolved=sum(1 for r in rows if r["resolved"]),
            eps_median_range=[float(np.min(arrays["xiao29_eps_median"])),
                              float(np.max(arrays["xiao29_eps_median"]))],
            r2_range=[float(np.min(arrays["xiao29_r2"])),
                      float(np.max(arrays["xiao29_r2"]))]),
        provenance=dict(source=str(src.relative_to(ROOT)), sha256=sha256(src),
                        instrument="rib_eps_ode.evaluate, Y_IDX=10 (archived)"),
    )
    return row, arrays


def eval_wavy():
    jsrc = latest("r1_sta2_wavy_wrles_*.json")
    if jsrc is None:
        return [], {}
    nsrc = jsrc.with_suffix(".npz")
    art = json.loads(jsrc.read_text())
    npz = np.load(nsrc, allow_pickle=True)
    rows, arrays = [], {}
    for grid in ("G0", "G1", "G2"):
        aid = f"wavy_wrles_{grid}"
        if grid not in art.get("grids", {}):
            rows.append(dict(id=aid, status="PENDING", blind=True,
                             reason="grid not yet in the deposited wavy harvest "
                                    f"(absent_grids={art.get('absent_grids')})"))
            continue
        g = art["grids"][grid]
        od = g["ode_diagnostic"]["0.1"]                     # preregistered surface
        blk_key = f"{grid}_block_r2_standard_ml"
        r2v = float(od["standard_ml"])
        if blk_key in npz.files:
            blocks = np.asarray(npz[blk_key], float)[:, 1]  # eta index 1 == 0.1
            sem = float(blocks.std(ddof=1) / math.sqrt(len(blocks)))
            lo, hi = r2v - 2 * sem, r2v + 2 * sem
            basis = f"block-window replicates (n={len(blocks)}), +/-2*SEM"
        else:
            lo = hi = r2v
            basis = "point value (no block array archived for this grid)"
        measured = interval_bin(lo, hi)
        eps = float(od["eps_median"]); eps_exact = float(od["eps_exact_median"])
        form_invariant = p1_forecast(eps) == p1_forecast(eps_exact)
        fc = p1_forecast(eps_exact)     # primary = exact-integral form (preregistered)
        resolved = measured is not None and form_invariant
        mb = measured if measured is not None else "UNRESOLVED"
        rows.append(dict(
            id=aid, status="EVALUATED", blind=(grid != "G0"), in_family=False,
            leaked=(grid == "G0"),
            y_m="eta_m = 0.1*delta (actual %.4f)" % od["eta_m_actual"],
            eps=dict(median_pointwise=eps, median_exact_integral=eps_exact,
                     primary_form="exact_integral", frac_lt0p1=float(od["frac_eps_lt0p1"]),
                     form_bin_invariant=bool(form_invariant)),
            p1_forecast=fc, p2_forecast=p2_forecast(float(od["frac_eps_lt0p1"]), 2.0),
            p0_forecast="TOLERATED",
            measured=dict(metric="a-priori standard_ml R2 at eta_m=0.1*delta",
                          value=r2v, interval=[lo, hi], bin=mb,
                          resolved=bool(resolved), resolution_basis=basis),
            scores=dict(p1=(score(fc, mb) if resolved else None),
                        p2=(score(rows_p2 := p2_forecast(float(od["frac_eps_lt0p1"]), 2.0), mb)
                            if resolved else None),
                        p0=(score("TOLERATED", mb) if resolved else None)),
            provenance=dict(source=str(jsrc.relative_to(ROOT)), sha256=sha256(jsrc),
                            npz=str(nsrc.relative_to(ROOT)), npz_sha256=sha256(nsrc),
                            case_id=g.get("case_id"), slurm_job_id=g.get("slurm_job_id")),
        ))
        for k in ("eps", "eps_exact", "tau_ref", "pred_standard_ml"):
            key = f"{grid}_eta0.1_{k}"
            if key in npz.files:
                arrays[f"{aid}_{k}"] = np.asarray(npz[key], float)
        if blk_key in npz.files:
            arrays[f"{aid}_block_r2"] = np.asarray(npz[blk_key], float)[:, 1]
    return rows, arrays


RIB_PITCH = {"rib_dtype_p3": 0.6, "rib_ktype_p8": 1.6}
CUBE_PITCH = {"cube_aligned": 0.5, "cube_staggered": 0.5, "cube_sparse": 1.0}
P0_RIB = {"rib_dtype_p3": "FAIL", "rib_ktype_p8": "MARGINAL"}
P0_CUBE = {"cube_aligned": "FAIL", "cube_staggered": "FAIL", "cube_sparse": "TOLERATED"}


def eval_r24():
    jsrc = latest("r2_4_m20_les_*.json")
    rows, arrays = [], {}
    want = dict(rib_dtype_p3="r24_rib_dtype_p3", rib_ktype_p8="r24_rib_ktype_p8",
                cube_aligned="r24_cube_aligned", cube_staggered="r24_cube_staggered",
                cube_sparse="r24_cube_sparse")
    if jsrc is None:
        return [dict(id=a, status="PENDING", blind=True,
                     reason="r2_4_m20 harvest artifact not yet deposited")
                for a in want], {}
    art = json.loads(jsrc.read_text())
    nsrc = jsrc.with_suffix(".npz")
    npz = np.load(nsrc, allow_pickle=True) if nsrc.exists() else None
    for aid, stem in want.items():
        prod, check = f"{stem}_G1", f"{stem}_G0"
        c = art["cases"].get(prod)
        if c is None or c.get("status") != "OK":
            rows.append(dict(id=aid, status="PENDING", blind=True,
                             reason=f"case {prod} not yet in the deposited r2_4_m20 harvest"))
            continue
        is_rib = c["kind"] == "rib"
        # preregistered: longest cumulative averaging window of the deposit
        # (mechanical rule; reproduces 'cum_140' for the rib pair exactly)
        cums = [k for k in c["windows"] if k.startswith("cum_")]
        if not cums:
            rows.append(dict(id=aid, status="PENDING", blind=True,
                             reason=f"case {prod} carries no cumulative window yet"))
            continue
        wname = max(cums, key=lambda k: float(k.split("_")[1]))
        w = c["windows"][wname] if is_rib else c["windows"][wname]["floor"]
        ci = w["station_block_bootstrap"]["r2_ci95"]
        eci = w["station_block_bootstrap"].get("eps_median_ci95")
        r2v = float(w["standard_ml_r2"])
        measured = interval_bin(float(ci[0]), float(ci[1]))
        mb = measured if measured is not None else "UNRESOLVED"
        eps = float(w["eps_median"])
        fc = p1_forecast(eps)
        pitch = (RIB_PITCH if is_rib else CUBE_PITCH)[aid]
        p2 = p2_forecast(float(w["frac_eps_lt0p1"]), pitch)
        p0 = (P0_RIB if is_rib else P0_CUBE)[aid]
        resolved = measured is not None
        gc = art["cases"].get(check)
        gcheck = None
        if gc and gc.get("status") == "OK" and any(k.startswith("cum_") for k in gc["windows"]):
            gname = max([k for k in gc["windows"] if k.startswith("cum_")],
                        key=lambda k: float(k.split("_")[1]))
            gw = gc["windows"][gname] if is_rib else gc["windows"][gname]["floor"]
            gcheck = dict(case=check, window=gname, standard_ml_r2=float(gw["standard_ml_r2"]),
                          r2_ci95=[float(v) for v in gw["station_block_bootstrap"]["r2_ci95"]],
                          eps_median=float(gw["eps_median"]),
                          bin=r2_bin(float(gw["standard_ml_r2"])))
        rows.append(dict(
            id=aid, status="EVALUATED", blind=True, in_family=False, leaked=False,
            y_m="Y_IDX=10 deposit convention (y_m/k=0.146)" if is_rib
                else "cube floor stations, Y_IDX=10 convention",
            eps=dict(median_pointwise=eps, median_exact_integral=None,
                     primary_form="pointwise (only form archived by the deposited instrument)",
                     interval_ci95=eci, frac_lt0p1=float(w["frac_eps_lt0p1"]),
                     form_bin_invariant=None),
            p1_forecast=fc, p2_forecast=p2, p0_forecast=p0,
            measured=dict(metric=f"a-priori standard_ml R2, production grid G1, window {wname}",
                          value=r2v, interval=[float(ci[0]), float(ci[1])], bin=mb,
                          resolved=bool(resolved),
                          resolution_basis="station moving-block bootstrap 95% CI (archived)"),
            grid_check=gcheck,
            scores=dict(p1=(score(fc, mb) if resolved else None),
                        p2=(score(p2, mb) if resolved else None),
                        p0=(score(p0, mb) if resolved else None)),
            provenance=dict(source=str(jsrc.relative_to(ROOT)), sha256=sha256(jsrc),
                            case_id=prod,
                            deposit_manifest_sha256=c.get("source_manifest_sha256")),
        ))
        if npz is not None:
            for k in ("eps", "tau_w", "pred_standard_ml"):
                key = f"{prod}__{wname}__{k}"
                if key in npz.files:
                    arrays[f"{aid}_{k}"] = np.asarray(npz[key], float)
    return rows, arrays


def _m13_candidates(re_h):
    """All deposited m13 harvests carrying this Re, newest first, each classified
    by the validity of its declared truth reference."""
    out = []
    for js in sorted(RESULTS.glob("m13_highre_coupled_*_summary.json"), reverse=True):
        try:
            art = json.loads(js.read_text())
        except Exception:
            continue
        c = art.get("campaigns", {}).get(str(re_h))
        if c is None or "G2c:equilibrium" not in c.get("metrics", {}):
            continue
        out.append((js, art, c, classify_truth(c.get("truth"))))
    return out


def eval_m13():
    jsrc = latest("m13_highre_coupled_*_summary.json")
    apriori = RESULTS / "dose_response_xiao.npz"
    rows, arrays = [], {}
    if jsrc is None:
        return [dict(id=f"hill_coupled_re{r}", status="PENDING", blind=True,
                     reason="m13 coupled harvest not yet deposited")
                for r in (5600, 10595, 19000, 37000)], {}
    art = json.loads(jsrc.read_text())
    # a-priori epsilon for the canonical geometry (only archived at Re 5600)
    eps_apriori = frac_apriori = None
    if apriori.exists():
        d = np.load(apriori, allow_pickle=True)
        idx = [i for i, c in enumerate(d["agg_case"]) if str(c) == "alph10-9-3036"]
        if idx:
            eps_apriori = float(d["agg_eps_median"][idx[0]])
            frac_apriori = float(d["agg_frac_eps_lt_0p1"][idx[0]])
            pitch_apriori = float(d["agg_cv_ellp_over_delta"][idx[0]])
    for re_h in (5600, 10595, 19000, 37000):
        aid = f"hill_coupled_re{re_h}"
        cands = _m13_candidates(re_h)
        if not cands:
            rows.append(dict(id=aid, status="PENDING", blind=True,
                             reason=f"Re_H={re_h} bundle not yet in any m13 harvest"))
            continue
        valid = [t for t in cands if t[3]["status"] == "VALID"]
        withdrawn = [t for t in cands if t[3]["status"] == "WITHDRAWN"]
        if not valid:
            # never score against a withdrawn reference: report, do not silently use
            js0, _, _, cl0 = withdrawn[0]
            rows.append(dict(
                id=aid, status="WITHDRAWN_REFERENCE", blind=False, in_family=False,
                reason=("the only deposited truth reference for this anchor is WITHDRAWN "
                        f"({cl0['file']}); no valid-reference harvest exists yet"),
                truth_reference=cl0,
                provenance=dict(source=str(js0.relative_to(ROOT)), sha256=sha256(js0))))
            continue
        jsrc_a, art_a, c, truth_class = valid[0]
        key = "G2c:equilibrium"
        met = c["metrics"].get(key)
        if "r2" not in met or key not in c.get("phase_bootstrap_primary_intervals", {}):
            rows.append(dict(
                id=aid,
                status="PENDING",
                blind=True,
                reason=(f"Re_H={re_h} bundle is terminal but supplies profile/event truth, "
                        "not the registered wall-traction reference; the wall-traction "
                        "anchor remains unscored"),
            ))
            continue
        r2v = float(met["r2"])
        # window replicates (180/225/270 on the finest grid) -> +/-2*SEM
        wins = c["averaging"][key]
        wvals = np.array([wins[t]["r2"] for t in ("180", "225", "270")], float)
        sem = float(wvals.std(ddof=1) / math.sqrt(len(wvals)))
        lo, hi = float(wvals.mean() - 2 * sem), float(wvals.mean() + 2 * sem)
        wbin = interval_bin(lo, hi)
        # archived bootstrap interval is on the primary estimand (relRMS); its
        # side of 1.0 must be consistent with the R2 bin, else CONFLICTED
        pb = c["phase_bootstrap_primary_intervals"][key]
        rel_side = ("FAIL_side" if pb["low"] > 1.0 else
                    "TOLERATED_side" if pb["high"] < 1.0 else "straddles_1")
        consistent = ((wbin == "FAIL" and rel_side == "FAIL_side") or
                      (wbin in ("MARGINAL", "TOLERATED") and rel_side == "TOLERATED_side"))
        resolved = wbin is not None and consistent
        mb = (wbin if wbin is not None else "UNRESOLVED") if consistent else "UNRESOLVED"
        basis = (f"window replicates 180/225/270 +/-2*SEM -> [{lo:.3f},{hi:.3f}]; "
                 f"archived relRMS bootstrap [{pb['low']:.3f},{pb['high']:.3f}] {rel_side}"
                 + ("" if consistent else " -> CONFLICTED_UNCERTAINTY, unresolved"))
        p1 = p1_forecast(eps_apriori) if (re_h == 5600 and eps_apriori is not None) else None
        p2 = (p2_forecast(frac_apriori, pitch_apriori)
              if (re_h == 5600 and frac_apriori is not None) else None)
        # amendment AMD-01: retain, verbatim, any score that a WITHDRAWN reference gave
        superseded = []
        for js_w, _a_w, c_w, cl_w in withdrawn:
            m_w = c_w["metrics"].get(key, {})
            wv_w = np.array([c_w["averaging"][key][t]["r2"]
                             for t in ("180", "225", "270")], float) \
                if key in c_w.get("averaging", {}) else np.array([])
            pb_w = c_w.get("phase_bootstrap_primary_intervals", {}).get(key)
            b_w = None
            if wv_w.size:
                s_w = float(wv_w.std(ddof=1) / math.sqrt(len(wv_w)))
                b_w = interval_bin(float(wv_w.mean() - 2 * s_w), float(wv_w.mean() + 2 * s_w))
            side_w = (None if not pb_w else
                      ("FAIL_side" if pb_w["low"] > 1.0 else
                       "TOLERATED_side" if pb_w["high"] < 1.0 else "straddles_1"))
            cons_w = ((b_w == "FAIL" and side_w == "FAIL_side") or
                      (b_w in ("MARGINAL", "TOLERATED") and side_w == "TOLERATED_side"))
            superseded.append(dict(
                amendment=cl_w["withdrawal"]["amendment"],
                reason=cl_w["withdrawal"]["reason"],
                truth_reference=cl_w,
                value=float(m_w.get("r2", np.nan)),
                bin=(b_w if (b_w and cons_w) else "UNRESOLVED"),
                source=str(js_w.relative_to(ROOT)), sha256=sha256(js_w)))
        rows.append(dict(
            id=aid, status="EVALUATED", blind=(re_h >= 19000), in_family=False,
            leaked=(re_h <= 10595),
            y_m="coupled matching plane of the m13 campaign (finest grid G2c)",
            eps=dict(
                median_pointwise=eps_apriori if re_h == 5600 else None,
                median_exact_integral=None,
                primary_form=("a-priori canonical member alph10-9-3036 (Y_IDX=10)"
                              if re_h == 5600 else
                              "NOT ARCHIVED at this Re: P1 not evaluable (preregistered)"),
                frac_lt0p1=frac_apriori if re_h == 5600 else None,
                eps_c_coupled_sensitivity={
                    m: c["eps_c"][f"G2c:{m}"]["eps_c_median_separated"]
                    for m in c["available_models"]
                    if f"G2c:{m}" in c.get("eps_c", {})}),
            p1_forecast=p1, p2_forecast=p2, p0_forecast="FAIL",
            measured=dict(metric="coupled phase-averaged wall-traction R2 vs registered "
                                 "reference truth, G2c equilibrium, longest window",
                          value=r2v, interval=[lo, hi], bin=mb,
                          resolved=bool(resolved), resolution_basis=basis,
                          sensitivity_total_gradient_tble=float(
                              c["metrics"].get("G2c:total_gradient_tble", {}).get("r2", np.nan)),
                          failure_significance_p=c["failure_significance_tests"]
                              .get("equilibrium", {}).get("p_one_sided")),
            scores=dict(p1=(score(p1, mb) if (resolved and p1) else None),
                        p2=(score(p2, mb) if (resolved and p2) else None),
                        p0=(score("FAIL", mb) if resolved else None)),
            truth_reference=truth_class,
            superseded_by_amendment=(superseded or None),
            provenance=dict(source=str(jsrc_a.relative_to(ROOT)), sha256=sha256(jsrc_a),
                            case_id=c["cases"].get(key),
                            truth=c.get("truth", {}).get("reference")),
        ))
        arrays[f"{aid}_window_r2"] = wvals
    return rows, arrays


# --------------------------------------------------------------------------
def scoreboard(anchors):
    """Apply the preregistered overall-verdict rule mechanically."""
    def collect(pred, blind_only):
        vals = []
        for a in anchors:
            if a.get("status") != "EVALUATED" or a.get("in_family"):
                continue
            if blind_only and not a.get("blind"):
                continue
            s = (a.get("scores") or {}).get(pred)
            if s is None:
                continue
            vals.append((a["id"], s, a["measured"]["bin"]))
        return vals

    def verdict(vals):
        if not vals:
            return "NO_RESOLVED_ANCHORS", None
        scores = [s for _, s, _ in vals]
        mean = float(np.mean(scores))
        bins = {b for _, _, b in vals}
        if any(s == 0.0 for s in scores) or mean < 0.6:
            return "DESCRIPTIVE_ONLY", mean
        if mean >= 0.9 and {"FAIL", "TOLERATED"} <= bins:
            return "PREDICTOR", mean
        return "MIXED", mean

    out = {}
    for pred in ("p1", "p2", "p0"):
        va, ma = verdict(collect(pred, False))
        vb, mb = verdict(collect(pred, True))
        out[pred] = dict(all_out_of_family=dict(verdict=va, mean_score=ma,
                                                anchors=collect(pred, False)),
                         blind_subset=dict(verdict=vb, mean_score=mb,
                                           anchors=collect(pred, True)),
                         headline=(vb if (va != vb and vb != "NO_RESOLVED_ANCHORS") else va))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--out-stem", default=None)
    a = ap.parse_args()
    prereg = json.loads(PREREG.read_text())
    prereg_sha = sha256(PREREG)

    anchors, arrays = [], {}
    row, arr = eval_xiao29()
    if row:
        anchors.append(row); arrays.update(arr)
    for fn in (eval_wavy, eval_r24, eval_m13):
        rows, arr = fn()
        anchors.extend(rows); arrays.update(arr)

    pending = [a_["id"] for a_ in anchors if a_.get("status") == "PENDING"]
    board = scoreboard(anchors)
    status = "R2_2_REAL_PARTIAL" if pending else "R2_2_REAL_COMPLETE"

    cert = dict(
        schema="epsilon_predictor_outoffamily_v1",
        ledger_row="R2-2 (real)",
        idea=("Pre-registered out-of-family test of the cancellation coordinate epsilon "
              "as a predictor of ODE wall-model failure, on the campaign's high-fidelity "
              "anchors, with the verdict stated plainly either way."),
        date=a.date,
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        status=status,
        preregistration=dict(path=str(PREREG.relative_to(ROOT)), sha256=prereg_sha,
                             registered_utc=prereg["registered_utc"],
                             eps_threshold=EPS_CRIT, p2_frac=P2_FRAC, p2_pitch=P2_PITCH),
        amendments=[dict(path=str(x.relative_to(ROOT)), sha256=sha256(x),
                         amendment_id=json.loads(x.read_text()).get("amendment_id"),
                         title=json.loads(x.read_text()).get("title"))
                    for x in AMENDMENTS],
        withdrawn_reference_registry={k: v["amendment"] for k, v in WITHDRAWN_TRUTHS.items()},
        anchors=anchors,
        pending_anchors=pending,
        scoreboard=board,
        headline=dict(
            question="Is epsilon an out-of-family predictor of ODE wall-model failure?",
            verdict=board["p1"]["headline"],
            verdict_all_resolved=board["p1"]["all_out_of_family"]["verdict"],
            verdict_blind_subset=board["p1"]["blind_subset"]["verdict"],
            geometry_readable_P0=board["p0"]["headline"],
            partial=bool(pending),
            plain_statement=None),   # filled below
        rerun_command="python3 codes/analysis/epsilon_predictor_outoffamily.py --date <YYYYMMDD>",
    )
    v = cert["headline"]
    stmt = {
        "PREDICTOR": "epsilon predicted the out-of-family verdicts; the criterion survives an honest test.",
        "DESCRIPTIVE_ONLY": ("epsilon does NOT predict out of family: at least one high-fidelity anchor "
                             "with epsilon below threshold is tolerated (or vice versa). epsilon remains "
                             "a descriptive a-priori coordinate; the deleted earlier submission claim stays dead."),
        "MIXED": ("the evidence is split: epsilon called some out-of-family anchors correctly and missed "
                  "others (or the resolved set is one-sided). It cannot be advertised as a predictor."),
        "NO_RESOLVED_ANCHORS": "no out-of-family anchor is resolved yet.",
    }[v["verdict"]]
    if cert["status"] == "R2_2_REAL_PARTIAL":
        stmt += f"  [PARTIAL: pending anchors {pending}]"
    v["plain_statement"] = stmt

    stem = a.out_stem or f"epsilon_predictor_outoffamily_{a.date}"
    jp = RESULTS / f"{stem}.json"
    jp.write_text(json.dumps(cert, indent=1, default=float) + "\n")
    np.savez(RESULTS / f"{stem}.npz",
             **{k: np.asarray(v_, float) for k, v_ in arrays.items()},
             prereg_sha256=prereg_sha, date=a.date, status=status)
    print(f"\n=== R2-2 (real) {status} ===")
    print("P1 (is epsilon a predictor?):", v["verdict"], "-", stmt)
    print("P0 (geometry-readable?):     ", v["geometry_readable_P0"])
    for a_ in anchors:
        if a_.get("status") != "EVALUATED" or a_.get("in_family"):
            continue
        m = a_["measured"]
        print(f"  {a_['id']:22s} eps_fc={a_.get('p1_forecast')} p0={a_.get('p0_forecast'):>9s} "
              f"measured={m['bin']:10s} R2={m['value']:+.3f} resolved={m['resolved']} "
              f"blind={a_.get('blind')}")
    print("pending:", pending)
    print("saved ->", jp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
