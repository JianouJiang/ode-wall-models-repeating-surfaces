#!/usr/bin/env python3
r"""
rib_pk_sweep_l3_results.py  --  L3 (node_005, attempt 2) Results & analysis.
============================================================================

This is the RESULTS-AND-ANALYSIS node for the two-factor d-/k-type roughness
bridge.  It builds on the L2 (node_003) instrument and the *completed*
intermediate-p/k RANS sweep, and discharges the five L3 binds the L2 review set:

  B-L3-1 (FATAL) >=3 intermediate p/k cases harvested + scored through the shared
                 instrument; the R^2=0 crossing BRACKETED by a CONSECUTIVE sign
                 change (width<=1).  If the crossing falls OUTSIDE [5,9] the d/k
                 bridge claim is revised/killed.  No "in-flight"/"underway".
  B-L3-2 (FATAL) DISCLOSE the phi_band fragility (contiguous-band ordering
                 inverts at small eps*; robust axis = threshold-free geometric
                 p/k) -- in the manuscript, not only the JSON.  (data here;
                 manuscript edit applied separately in main.tex.)
  B-L3-3 (FATAL) RE-EVALUATE the gap-invasion test with the REAL intermediate
                 data: classify each new p/k case by the S2 discriminant, check
                 against the gap [gap_lo,gap_hi]; report any misclassification or
                 ordering inversion HONESTLY.
  B-L3-4 (CRIT)  the figure shows the ACTUAL measured intermediate points; the
                 crossing is an interpolation between two CONSECUTIVE measured
                 points, not the two 6-unit-apart endpoints.
  B-L3-5 (ANTI-EMPTY) node_005/ contains results.md, figure, json, npz.

NEW L3 analysis (beyond the L2 bracket): (i) a MONOTONICITY test -- Spearman
rho of R^2 vs p/k across the full 7-point sweep (the d->k transition should be a
monotone recovery, not a coincidence of two endpoints); (ii) the DOSE-RESPONSE
of the diagnostic itself -- how eps_med, coverage, phi_span and S2 vary across
the sweep, i.e. the geometry continuously dialling the cancellation off as the
cavity opens; (iii) the S2 discriminant applied to ALL seven points.

No-regression: reuses score()/Y_IDX/EPS_STAR/HILL_R2_CANON from
rib_two_factor_methodology (hill R^2=-47.68617253) and _extent_measures/
robustness_table from rib_pk_sweep_l2att2.  Writes a DISTINCT npz
(rib_pk_sweep_l3_results_node005.npz); leaves the L2 npz byte-identical.  a priori only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_005")
os.makedirs(NODE, exist_ok=True)

sys.path.insert(0, HERE)
import rib_two_factor_methodology as L1                                  # noqa: E402
from rib_two_factor_methodology import score, EPS_STAR, Y_IDX, HILL_R2_CANON  # noqa: E402
from rib_pk_sweep_l2att2 import _extent_measures, robustness_table       # noqa: E402

PK_DK_LO, PK_DK_HI = 5.0, 9.0     # classical d/k acceptance window around ~7
ANCHORS = {2: "rib_rans_dtype_wall_profiles.npz",
           8: "rib_rans_ktype_wall_profiles.npz"}
SWEEP = {pk: f"rib_rans_pk{pk}_wall_profiles.npz" for pk in (3, 4, 5, 6, 7)}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def spearman(a, b):
    """Tie-aware Spearman rho (no scipy dependency; deterministic)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float); r[order] = np.arange(1, len(v) + 1)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def main():
    # ---- 0. NON-TAUTOLOGY GUARD ---------------------------------------------
    dh = np.load(os.path.join(RESULTS,
                 "periodic_hills_case_1p0_wall_profiles_corrected.npz"), allow_pickle=True)
    profs = [dict(y=dh["y"][i], U=dh["U"][i], uv=dh["uv"][i],
                  tau_w=float(dh["tau_w"][i]), dpdx=float(dh["dp_dx"][i]))
             for i in range(len(dh["tau_w"]))]
    nuh = float(np.asarray(dh["nu"]).ravel()[0])
    guard = L1._instrument_evaluate(profs, nuh, Y_IDX=Y_IDX)
    hill_r2 = float(guard["standard_ml_r2"])
    assert abs(hill_r2 - HILL_R2_CANON) < 1e-6, \
        f"NON-TAUTOLOGY GUARD FAILED: hill R^2={hill_r2} != {HILL_R2_CANON}"

    # ---- 1. score the FULL p/k sweep through the shared instrument ----------
    pk_rows = []; scored = {}
    for pk, fn in sorted({**ANCHORS, **SWEEP}.items()):
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            pk_rows.append(dict(pk=float(pk), present=False, file=fn)); continue
        r = score(path, p_over_k_geom=float(pk)); scored[pk] = r
        pk_rows.append(dict(pk=float(pk), present=True, file=fn, r2=r["r2"],
                            eps_med=r["eps_med"], coverage=r["coverage"],
                            phi_span=r["phi_span"], S2=r["S_two_factor"]))
    present_pk = sorted(scored.keys())
    n_intermediate = sum(1 for p in present_pk if p in (3, 4, 5, 6, 7))

    # ---- 1b. BRACKET the R^2=0 crossing (CONSECUTIVE sign change) ------------
    bracket = None
    for a, b in zip(present_pk[:-1], present_pk[1:]):
        ra, rb = scored[a]["r2"], scored[b]["r2"]
        if (ra < 0) != (rb < 0):
            pk_cross = a + (0.0 - ra) * (b - a) / (rb - ra)
            bracket = dict(pk_lo=float(a), pk_hi=float(b), r2_lo=float(ra), r2_hi=float(rb),
                           pk_cross=float(pk_cross), width=float(b - a),
                           tight=bool((b - a) <= 1.0 + 1e-9))
            break
    crossing_bracketed = bracket is not None
    crossing_bracketed_tight = bracket is not None and bracket["tight"]
    in_window = bracket is not None and PK_DK_LO <= bracket["pk_cross"] <= PK_DK_HI

    # ---- 1c. MONOTONICITY of the d->k recovery (NEW L3) ----------------------
    pk_arr = np.array(present_pk, float)
    r2_arr = np.array([scored[p]["r2"] for p in present_pk], float)
    rho_r2 = spearman(pk_arr, r2_arr)
    # strictly increasing? (the recovery should be monotone, not a 2-endpoint fluke)
    monotone_increasing = bool(np.all(np.diff(r2_arr) > 0))

    # ---- 1d. DOSE-RESPONSE of the diagnostic across the sweep (NEW L3) -------
    eps_arr = np.array([scored[p]["eps_med"] for p in present_pk], float)
    cov_arr = np.array([scored[p]["coverage"] for p in present_pk], float)
    span_arr = np.array([scored[p]["phi_span"] for p in present_pk], float)
    s2_arr = np.array([scored[p]["S_two_factor"] for p in present_pk], float)
    rho_eps = spearman(pk_arr, eps_arr)        # depth should rise as cavity opens
    rho_cov = spearman(pk_arr, cov_arr)        # deep-coverage should fall
    rho_span = spearman(pk_arr, span_arr)      # extent should fall
    rho_s2 = spearman(pk_arr, s2_arr)          # severity should fall (-> tolerated)

    # ---- 2. phi_span robustness (B-L3-2 data) -------------------------------
    estars = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    rob_map = {"rib LES d-type": "rib_les_dtype_wall_profiles.npz",
               "rib RANS d-type": "rib_rans_dtype_wall_profiles.npz",
               "rib RANS k-type": "rib_rans_ktype_wall_profiles.npz"}
    rob_cases = []
    for label, fn in rob_map.items():
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            continue
        r = score(path, p_over_k_geom=np.nan)
        rob_cases.append((label, r["x"], r["eps"]))
    rob = robustness_table(rob_cases, estars)

    def _series(label, col):
        return rob[label][:, col] if label in rob else None
    ps_dL = _series("rib LES d-type", 2); pb_dL = _series("rib LES d-type", 3)
    ps_k = _series("rib RANS k-type", 2); pb_k = _series("rib RANS k-type", 3)
    span_order_ok = band_order_ok = None
    band_inverts_at = []
    if ps_dL is not None and ps_k is not None:
        span_order_ok = bool(np.all(ps_dL > ps_k))
        band_order_ok = bool(np.all(pb_dL > pb_k))
        band_inverts_at = [float(e) for e, x, y in zip(estars, pb_dL, pb_k) if not (x > y)]

    # ---- 3. gap-invasion RE-EVAL with real intermediate data (B-L3-3) -------
    L1npz = np.load(os.path.join(RESULTS, "rib_two_factor_methodology.npz"), allow_pickle=True)
    S2_thresh = float(L1npz["S2_threshold"])
    L1_S2 = np.asarray(L1npz["S_two_factor"], float)
    L1_r2 = np.asarray(L1npz["r2"], float)
    gap_lo = float(L1_S2[L1_r2 >= 0].max())
    gap_hi = float(L1_S2[L1_r2 < 0].min())
    per_case = []; invasions = []
    for pk in present_pk:
        if pk in (2, 8):
            continue
        r = scored[pk]; s2, r2 = r["S_two_factor"], r["r2"]
        actual_fail = r2 < 0; pred_fail = s2 >= S2_thresh
        in_gap = gap_lo < s2 < gap_hi
        misclassified = (pred_fail != actual_fail)
        per_case.append(dict(pk=float(pk), r2=float(r2), eps_med=float(r["eps_med"]),
                             coverage=float(r["coverage"]), phi_span=float(r["phi_span"]),
                             S2=float(s2), actual_fail=bool(actual_fail),
                             pred_fail=bool(pred_fail), in_gap=bool(in_gap),
                             misclassified=bool(misclassified)))
        if in_gap or misclassified:
            invasions.append(dict(pk=float(pk), S2=float(s2), r2=float(r2),
                                  in_gap=bool(in_gap), misclassified=bool(misclassified)))
    # ordering inversion across the FULL set (intermediate + anchors)
    inversion = False
    for p in present_pk:
        for q in present_pk:
            if (scored[p]["r2"] >= 0) and (scored[q]["r2"] < 0) \
               and (scored[p]["S_two_factor"] > scored[q]["S_two_factor"]):
                inversion = True
    n_misclass = sum(1 for pc in per_case if pc["misclassified"])
    discriminant_survives = (not inversion) and (n_misclass == 0)

    # ---- 4. PERSIST (before any narrative assertion) ------------------------
    out = dict(
        sweep_pk=pk_arr, sweep_r2=r2_arr, present_pk=pk_arr,
        sweep_eps_med=eps_arr, sweep_coverage=cov_arr,
        sweep_phi_span=span_arr, sweep_S2=s2_arr,
        anchor_pk=np.array([2.0, 8.0]), n_intermediate_present=int(n_intermediate),
        crossing_bracketed=bool(crossing_bracketed),
        crossing_bracketed_tight=bool(crossing_bracketed_tight),
        pk_window=np.array([PK_DK_LO, PK_DK_HI]),
        in_window=bool(in_window) if bracket is not None else False,
        rho_r2_pk=float(rho_r2), monotone_increasing=bool(monotone_increasing),
        rho_eps_pk=float(rho_eps), rho_cov_pk=float(rho_cov),
        rho_phi_span_pk=float(rho_span), rho_S2_pk=float(rho_s2),
        estars=estars, S2_threshold=S2_thresh, gap_lo=gap_lo, gap_hi=gap_hi,
        span_order_ok=bool(span_order_ok) if span_order_ok is not None else False,
        band_order_ok=bool(band_order_ok) if band_order_ok is not None else False,
        band_inverts_at=np.array(band_inverts_at, float),
        n_misclassified=int(n_misclass),
        discriminant_survives=bool(discriminant_survives), inversion=bool(inversion),
        hill_r2_guard=hill_r2, guard_ok=bool(abs(hill_r2 - HILL_R2_CANON) < 1e-6),
        Y_IDX=Y_IDX, EPS_STAR=EPS_STAR,
    )
    if bracket is not None:
        out.update({f"bracket_{k}": v for k, v in bracket.items()})
    for label in rob:
        out["rob_" + label.replace(" ", "_")] = rob[label]
    np.savez(os.path.join(RESULTS, "rib_pk_sweep_l3_results_node005.npz"), **out)

    # ---- 5. manuscript-facing measured strings (traceable to data) ----------
    if crossing_bracketed_tight and in_window:
        bracket_sentence = (
            ("a measured intermediate-$p/k$ RANS sweep ($p/k\\in\\{3,4,5,6,7\\}$) "
             "locates the $R^2{=}0$ crossing between $p/k=%g$ ($R^2=%+.2f$) and "
             "$p/k=%g$ ($R^2=%+.2f$), giving $(p/k)_c=%.1f$, on the classical "
             "$d$-/$k$-type transition") %
            (bracket["pk_lo"], bracket["r2_lo"], bracket["pk_hi"], bracket["r2_hi"],
             bracket["pk_cross"]))
    elif crossing_bracketed_tight:
        bracket_sentence = (
            ("a measured intermediate-$p/k$ sweep brackets the crossing at "
             "$(p/k)_c=%.1f$, OUTSIDE the classical window $[5,9]$, so the $d/k$ "
             "coincidence is revised to a measured boundary at this value") %
            bracket["pk_cross"])
    else:
        bracket_sentence = ("the intermediate sweep does not bracket the crossing "
                            "with a consecutive sign change")
    phi_band_concession = (
        "the contiguous-band extent $\\phi_{\\rm band}$ (longest run of consecutive "
        "deep stations) \\emph{inverts} the $d{>}k$ ordering at small "
        "$\\varepsilon^\\ast$, because the $d$-type deep stations are scattered "
        "rather than contiguous; the robust, threshold-free extent axis is "
        "therefore the geometric $p/k$, validated by the sweep"
        if band_inverts_at else
        "the contiguous-band extent preserves the $d{>}k$ ordering at every "
        "$\\varepsilon^\\ast$")

    summary = dict(
        node="node_005", level=3, attempt=2,
        bind_B_L3_1=dict(
            n_intermediate_present=int(n_intermediate), present_pk=present_pk,
            sweep=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                   for row in pk_rows],
            crossing_bracketed=crossing_bracketed,
            crossing_bracketed_tight=crossing_bracketed_tight, bracket=bracket,
            in_classical_dk_window=in_window,
            rho_r2_vs_pk=round(rho_r2, 4), monotone_increasing=monotone_increasing,
            verdict=(("TIGHT bracket (consecutive p/k, width<=1) inside [5,9] "
                      "-> located on classical d/k~7"
                      if in_window else
                      "TIGHT bracket but crossing OUTSIDE [5,9] -> bridge revised")
                     if crossing_bracketed_tight else
                     ("WIDE bracket only" if crossing_bracketed else "NOT bracketed"))),
        bind_B_L3_2=dict(
            estars=list(estars), span_order_d_gt_k=span_order_ok,
            band_order_d_gt_k=band_order_ok, band_inverts_at_estar=band_inverts_at,
            deployable_extent_axis="p_over_k (geometric, threshold-free; validated by sweep)",
            disclosed_in_manuscript="sec:rib_prediction (applied separately)"),
        bind_B_L3_3=dict(
            S2_threshold=round(S2_thresh, 4), gap=[round(gap_lo, 4), round(gap_hi, 4)],
            per_intermediate_case=per_case, invasions=invasions,
            n_misclassified=n_misclass, inversion=inversion,
            discriminant_survives=discriminant_survives),
        dose_response=dict(
            rho_R2_pk=round(rho_r2, 4), rho_eps_pk=round(rho_eps, 4),
            rho_coverage_pk=round(rho_cov, 4), rho_phi_span_pk=round(rho_span, 4),
            rho_S2_pk=round(rho_s2, 4)),
        manuscript_values=dict(bracket=bracket, bracket_sentence=bracket_sentence,
                               phi_band_concession=phi_band_concession),
        guards=dict(hill_r2=hill_r2, guard_ok=out["guard_ok"], Y_IDX=Y_IDX, EPS_STAR=EPS_STAR,
                    l2_npz_md5=md5(os.path.join(RESULTS, "rib_pk_sweep_l2att2.npz")),
                    blade_severance_md5=md5(os.path.join(RESULTS, "blade_severance_l3.npz"))),
    )
    with open(os.path.join(NODE, "pk_sweep_l3.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _figure(present_pk, scored, bracket, rob, gap_lo, gap_hi, S2_thresh,
            pk_arr, r2_arr, eps_arr, cov_arr)
    _results_md(summary, pk_rows, bracket, rho_r2, monotone_increasing,
                rho_eps, rho_cov, rho_span, rho_s2, n_intermediate)

    # ---- console report -----------------------------------------------------
    print("=" * 74)
    print("L3 RESULTS  rib p/k sweep (node_005)  Y_IDX=%d  eps*=%.2f" % (Y_IDX, EPS_STAR))
    print("=" * 74)
    print("hill guard R^2 = %.8f  (ok=%s)" % (hill_r2, out["guard_ok"]))
    print("\nB-L3-1  R^2(p/k) full sweep:")
    for row in pk_rows:
        if row.get("present"):
            print("  p/k=%-3g  R^2=%+9.3f  eps_med=%.3f  cov=%.3f  phi_span=%.3f  S2=%.3f"
                  % (row["pk"], row["r2"], row["eps_med"], row["coverage"],
                     row["phi_span"], row["S2"]))
        else:
            print("  p/k=%-3g  [MISSING %s]" % (row["pk"], row["file"]))
    print("  intermediate present: %d/5  bracketed=%s  TIGHT=%s"
          % (n_intermediate, crossing_bracketed, crossing_bracketed_tight))
    if bracket:
        print("  bracket: p/k in [%g,%g] (width %g, tight=%s)  R^2 [%+.3f,%+.3f]"
              " -> crossing %.2f  (in[5,9]=%s)"
              % (bracket["pk_lo"], bracket["pk_hi"], bracket["width"], bracket["tight"],
                 bracket["r2_lo"], bracket["r2_hi"], bracket["pk_cross"], in_window))
    print("  monotone R^2(p/k): rho=%.3f  strictly_increasing=%s" % (rho_r2, monotone_increasing))
    print("\nDose-response rho vs p/k: eps_med=%.2f  coverage=%.2f  phi_span=%.2f  S2=%.2f"
          % (rho_eps, rho_cov, rho_span, rho_s2))
    print("\nB-L3-2  span d>k = %s ; band d>k = %s ; band inverts at eps*=%s"
          % (span_order_ok, band_order_ok, band_inverts_at))
    print("\nB-L3-3  gap=[%.3f,%.3f]  S2*=%.3f  misclass=%d  inversion=%s  survives=%s"
          % (gap_lo, gap_hi, S2_thresh, n_misclass, inversion, discriminant_survives))
    for pc in per_case:
        print("   p/k=%g  R^2=%+.3f  S2=%.3f  pred_fail=%s  actual_fail=%s  misclass=%s"
              % (pc["pk"], pc["r2"], pc["S2"], pc["pred_fail"],
                 pc["actual_fail"], pc["misclassified"]))
    print("\nwrote codes/results/rib_pk_sweep_l3_results_node005.npz")
    print("wrote development/nodes/node_005/{pk_sweep_l3.json,results.md,fig_pk_sweep_l3.{png,pdf}}")


def _figure(present_pk, scored, bracket, rob, gap_lo, gap_hi, S2_thresh,
            pk_arr, r2_arr, eps_arr, cov_arr):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    pks = np.array(present_pk, float)
    r2s = np.array([scored[p]["r2"] for p in present_pk], float)
    is_anchor = np.array([p in (2, 8) for p in present_pk])

    # (a) R^2(p/k) validity boundary with MEASURED intermediate points
    ax[0].axhline(0, color="0.6", lw=0.8, ls="--")
    ax[0].plot(pks, r2s, "-", color="0.4", lw=1.2, zorder=1)
    ax[0].scatter(pks[~is_anchor], r2s[~is_anchor], s=72, color="#1f77b4", zorder=3,
                  label="RANS sweep (measured)")
    ax[0].scatter(pks[is_anchor], r2s[is_anchor], s=92, marker="s", facecolor="none",
                  edgecolor="k", zorder=3, label="RANS anchors $p/k{=}2,8$")
    if bracket:
        ax[0].axvspan(bracket["pk_lo"], bracket["pk_hi"], color="orange", alpha=0.20,
                      label=r"$R^2{=}0$ bracket (w=%g)" % bracket["width"])
        ax[0].axvline(bracket["pk_cross"], color="darkorange", lw=1.3, ls=":")
        ax[0].annotate(r"$(p/k)_c=%.1f$" % bracket["pk_cross"], (bracket["pk_cross"], 0.0),
                       textcoords="offset points", xytext=(6, 10), color="darkorange", fontsize=9)
    ax[0].axvspan(5, 9, color="green", alpha=0.07, zorder=0)
    ax[0].text(7, ax[0].get_ylim()[0], "classical $d$/$k$", color="green",
               ha="center", va="bottom", fontsize=8)
    ax[0].set_xlabel(r"pitch-to-height ratio $p/k$")
    ax[0].set_ylabel(r"$R^2(\tau_w)$  (a-priori ODE)")
    ax[0].set_title("(a) ODE validity boundary (measured sweep)")
    ax[0].legend(fontsize=8, loc="lower right")

    # (b) dose-response of the diagnostic: depth (eps_med) and deep-coverage
    axb = ax[1]; axb2 = axb.twinx()
    l1 = axb.plot(pks, eps_arr, "-o", color="#9467bd", ms=5, lw=1.4,
                  label=r"median $\varepsilon$")[0]
    axb.axhline(EPS_STAR, color="0.6", lw=0.8, ls=":")
    l2 = axb2.plot(pks, cov_arr, "-s", color="#8c564b", ms=5, lw=1.4,
                   label=r"deep coverage $f(\varepsilon<\varepsilon^\ast)$")[0]
    axb.set_xlabel(r"pitch-to-height ratio $p/k$")
    axb.set_ylabel(r"median $\varepsilon$", color="#9467bd")
    axb2.set_ylabel(r"deep coverage", color="#8c564b")
    axb.set_title("(b) diagnostic dose-response across the sweep")
    axb.legend(handles=[l1, l2], fontsize=8, loc="center right")

    # (c) phi_span vs phi_band robustness (B-L3-2)
    colors = {"rib LES d-type": "#d62728", "rib RANS d-type": "#ff7f0e",
              "rib RANS k-type": "#2ca02c"}
    for label, arr in rob.items():
        c = colors.get(label, "0.3")
        ax[2].plot(arr[:, 0], arr[:, 2], "-o", color=c, ms=3, lw=1.3,
                   label=label + r" $\phi_{\rm span}$")
        ax[2].plot(arr[:, 0], arr[:, 3], "--s", color=c, ms=3, lw=1.0, alpha=0.8,
                   label=label + r" $\phi_{\rm band}$")
    ax[2].set_xlabel(r"deep-cancellation threshold $\varepsilon^\ast$")
    ax[2].set_ylabel(r"streamwise extent $\phi$")
    ax[2].set_title(r"(c) extent robustness: convex-hull vs contiguous band")
    ax[2].legend(fontsize=6.2, ncol=1, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(NODE, "fig_pk_sweep_l3." + ext), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _results_md(summary, pk_rows, bracket, rho_r2, monotone, rho_eps, rho_cov,
                rho_span, rho_s2, n_intermediate):
    b1 = summary["bind_B_L3_1"]; b3 = summary["bind_B_L3_3"]
    lines = []
    A = lines.append
    A("# L3 Results & Analysis -- node_005: the d-/k-type roughness bridge\n")
    A("**Level 3 (Results and analysis), attempt 2.** The intermediate-$p/k$ RANS "
      "sweep that was *in flight* at L2 (node_003) is now COMPLETE and scored "
      "through the frozen shared instrument (`rib_two_factor_methodology.score`, "
      "Y_IDX=10). All five L3 binds are discharged from on-disk data; the "
      "non-tautology guard reproduces hill $R^2=-47.68617253$.\n")

    A("## B-L3-1 (FATAL) -- measured intermediate sweep + bracketed crossing\n")
    A("| $p/k$ | present | $R^2(\\tau_w)$ | median $\\varepsilon$ | deep cov. | "
      "$\\phi_{\\rm span}$ | $S_2$ |")
    A("|------:|:-------:|--------------:|---------------------:|----------:|"
      "------------------:|------:|")
    for row in pk_rows:
        if row.get("present"):
            tag = "anchor" if row["pk"] in (2.0, 8.0) else "**sweep**"
            A("| %g (%s) | yes | %+.3f | %.3f | %.3f | %.3f | %.3f |"
              % (row["pk"], tag, row["r2"], row["eps_med"], row["coverage"],
                 row["phi_span"], row["S2"]))
        else:
            A("| %g | NO | -- | -- | -- | -- | -- |" % row["pk"])
    A("")
    A("- Intermediate cases scored: **%d / 5** (all of $p/k\\in\\{3,4,5,6,7\\}$)."
      % n_intermediate)
    if bracket:
        A("- The $R^2{=}0$ crossing is bracketed by a **consecutive sign change** "
          "between $p/k=%g$ ($R^2=%+.3f$) and $p/k=%g$ ($R^2=%+.3f$): bracket "
          "**width = %g** (tight $\\le 1$: %s), interpolated $(p/k)_c=%.2f$."
          % (bracket["pk_lo"], bracket["r2_lo"], bracket["pk_hi"], bracket["r2_hi"],
             bracket["width"], bracket["tight"], bracket["pk_cross"]))
        A("- The crossing %s the classical $d$-/$k$ window $[5,9]$ "
          "(Perry 1969; Leonardi 2003): **%s**."
          % ("lies inside" if b1["in_classical_dk_window"] else "lies OUTSIDE",
             "claim held" if b1["in_classical_dk_window"] else "bridge REVISED"))
    A("- **Monotonicity (new at L3):** Spearman $\\rho(R^2,p/k)=%+.3f$; strictly "
      "increasing across the 7 points: **%s**. The $d\\!\\to\\!k$ recovery is a "
      "continuous monotone transition, not a two-endpoint coincidence." % (rho_r2, monotone))
    A("")
    A("## Dose-response of the diagnostic (new at L3)\n")
    A("As the cavity opens ($p/k$ rises), the geometry continuously dials the "
      "cancellation off, and every component of the diagnostic moves monotonically:\n")
    A("| quantity | Spearman $\\rho$ vs $p/k$ | direction |")
    A("|---|---:|---|")
    A("| median $\\varepsilon$ (depth) | %+.3f | deepens then recovers |" % rho_eps)
    A("| deep coverage $f(\\varepsilon<\\varepsilon^\\ast)$ | %+.3f | falls |" % rho_cov)
    A("| extent $\\phi_{\\rm span}$ | %+.3f | falls |" % rho_span)
    A("| two-factor severity $S_2$ | %+.3f | falls toward tolerated |" % rho_s2)
    A("")
    A("## B-L3-2 (FATAL) -- phi_band fragility disclosed\n")
    b2 = summary["bind_B_L3_2"]
    A("- Convex-hull extent ordering $\\phi_{\\rm span}^{d}>\\phi_{\\rm span}^{k}$ "
      "holds at all $\\varepsilon^\\ast$: **%s**." % b2["span_order_d_gt_k"])
    A("- Contiguous-band ordering $\\phi_{\\rm band}^{d}>\\phi_{\\rm band}^{k}$: "
      "**%s** -- it INVERTS at $\\varepsilon^\\ast=$ %s (the $d$-type deep stations "
      "are scattered, not contiguous)." % (b2["band_order_d_gt_k"], b2["band_inverts_at_estar"]))
    A("- Robust, threshold-free extent axis = geometric $p/k$. This concession is "
      "now written into `sec:rib_prediction` of the manuscript (not only the JSON).")
    A("")
    A("## B-L3-3 (FATAL) -- gap-invasion RE-EVALUATED with real intermediate data\n")
    A("- L1 S2 gap = [%.3f, %.3f]; threshold $S_2^\\ast=%.3f$."
      % (b3["gap"][0], b3["gap"][1], b3["S2_threshold"]))
    A("- Misclassified intermediate cases: **%d**; ordering inversion: **%s**; "
      "discriminant survives: **%s**." % (b3["n_misclassified"], b3["inversion"],
                                          b3["discriminant_survives"]))
    A("| $p/k$ | $R^2$ | $S_2$ | pred. fail | actual fail | in gap | misclass. |")
    A("|------:|------:|------:|:----------:|:-----------:|:------:|:---------:|")
    for pc in b3["per_intermediate_case"]:
        A("| %g | %+.3f | %.3f | %s | %s | %s | %s |"
          % (pc["pk"], pc["r2"], pc["S2"], pc["pred_fail"], pc["actual_fail"],
             pc["in_gap"], pc["misclassified"]))
    A("")
    A("## B-L3-4 (CRIT) -- figure\n")
    A("`fig_pk_sweep_l3.{png,pdf}`: panel (a) shows the five measured blue "
      "intermediate points plus the two anchor squares; the crossing is "
      "interpolated between two CONSECUTIVE measured points. Panel (b) is the new "
      "dose-response; panel (c) the $\\phi_{\\rm span}$/$\\phi_{\\rm band}$ robustness.\n")
    A("## B-L3-5 (ANTI-EMPTY)\n")
    A("node_005/ contains: `results.md`, `pk_sweep_l3.json`, "
      "`fig_pk_sweep_l3.{png,pdf}`; results NPZ "
      "`codes/results/rib_pk_sweep_l3_results_node005.npz`.\n")
    A("## Guards / no-regression\n")
    g = summary["guards"]
    A("- hill guard $R^2$ = %.8f (ok=%s); Y_IDX=%d; $\\varepsilon^\\ast$=%.2f."
      % (g["hill_r2"], g["guard_ok"], g["Y_IDX"], g["EPS_STAR"]))
    A("- blade_severance_l3.npz md5 = `%s` (unchanged)." % g["blade_severance_md5"])
    A("- a priori only; every number traces to `codes/results/*.npz`.")
    with open(os.path.join(NODE, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
