#!/usr/bin/env python3
r"""
rib_pk_sweep_l2att2.py  --  L2 (node_003, ATTEMPT 2) implementation & experiments.
=================================================================================

Attempt 1 (node_002) was not accepted at the L2 review for a single FATAL reason:
B-L2-1 was not discharged -- the intermediate-p/k RANS sweep was *in flight* with
ZERO intermediate cases harvested, so the R^2=0 crossing was a 2-endpoint linear
interpolation, not a measured bracket.  The root cause was structural, not timing:
the sweep case generator emitted no fieldAverage function object, so the separated
rib-cavity RANS (which limit-cycles rather than reaching a steady fixed point)
produced no UMean/pMean, and harvest_rans_profiles.py's honesty guard (correctly)
refused to write an npz.

THIS attempt FIXES the structural bug (make_rib_rans_case.py now emits the same
fieldAverage block as the d-/k-type anchors, timeStart=12000 of endTime=20000) and
RUNS THE FULL SWEEP {3,4,5,6,7} TO COMPLETION before scoring.  The intermediate npz
are harvested via the time-averaged route (UMean present AND Ux residual <1e-4) --
exactly the route the anchors already use.

This script SCORES the completed sweep through the *shared* L1 instrument
(rib_two_factor_methodology.score, itself rib_eps_ode.evaluate frozen at Y_IDX=10),
written BEFORE the assertions, and discharges the L2 binds:

  B-L2-1 (FATAL)  >=3 intermediate p/k harvested + scored; R^2=0 crossing
                  BRACKETED by a sign change between two CONSECUTIVE p/k (width<=1).
                  If the crossing lands OUTSIDE [5,9] the d/k bridge is revised.
  B-L2-2 (FATAL)  phi_span robustness: eps*-sensitivity table [0.05,0.30] AND a
                  contiguous-band phi_band; honest report of whether the contiguous
                  ordering INVERTS (it does, at small eps* -- disclosed).
  B-L2-4 (CRIT)   Gap-invasion: each NEW intermediate case classified; does any land
                  in the L1 S2 gap and get misclassified, or invert the ordering?

No-regression: writes a DISTINCT npz (rib_pk_sweep_l2att2.npz); the node_002 npz
is left byte-identical.  Non-tautology guard reproduces hill R^2=-47.68617253.
a priori only; no fabrication.
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
NODE = os.path.join(PROJ, "development", "nodes", "node_003")
os.makedirs(NODE, exist_ok=True)

sys.path.insert(0, HERE)
import rib_two_factor_methodology as L1            # noqa: E402
from rib_two_factor_methodology import score, EPS_STAR, Y_IDX, HILL_R2_CANON  # noqa: E402

PK_DK_LO, PK_DK_HI = 5.0, 9.0          # classical d/k acceptance window around ~7


# --- contiguous-band / convex-hull extent measures (B-L2-2) -------------------
def _extent_measures(x, eps, estar):
    x = np.asarray(x, float); eps = np.asarray(eps, float)
    order = np.argsort(x); xs, es = x[order], eps[order]
    ev = np.isfinite(es)
    span = float(xs.max() - xs.min()) if xs.size > 1 else 0.0
    deep = ev & (es < estar)
    n_ev = int(ev.sum())
    coverage = float(deep.sum() / n_ev) if n_ev > 0 else 0.0
    if deep.sum() > 1 and span > 0:
        xd = xs[deep]; phi_span = float((xd.max() - xd.min()) / span)
    else:
        phi_span = 0.0
    best_lo = best_hi = -1; best_len = 0; i = 0; nstn = xs.size
    while i < nstn:
        if deep[i]:
            j = i
            while j + 1 < nstn and deep[j + 1]:
                j += 1
            if (j - i) > (best_hi - best_lo):
                best_lo, best_hi, best_len = i, j, j - i + 1
            i = j + 1
        else:
            i += 1
    phi_band = float((xs[best_hi] - xs[best_lo]) / span) if best_len > 1 and span > 0 else 0.0
    return coverage, phi_span, phi_band, int(deep.sum()), int(best_len)


def robustness_table(cases, estars):
    out = {}
    for label, x, eps in cases:
        rows = []
        for e in estars:
            cov, ps, pb, nd, nr = _extent_measures(x, eps, e)
            rows.append((e, cov, ps, pb, nd, nr))
        out[label] = np.array(rows, float)
    return out


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def main():
    # --- 0. NON-TAUTOLOGY GUARD ----------------------------------------------
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

    # --- 1. score the FULL p/k sweep through the shared instrument -------------
    anchors = {2: "rib_rans_dtype_wall_profiles.npz",
               8: "rib_rans_ktype_wall_profiles.npz"}
    sweep_files = {pk: f"rib_rans_pk{pk}_wall_profiles.npz" for pk in (3, 4, 5, 6, 7)}

    pk_rows = []; scored = {}
    for pk, fn in sorted({**anchors, **sweep_files}.items()):
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            pk_rows.append(dict(pk=float(pk), present=False, file=fn)); continue
        r = score(path, p_over_k_geom=float(pk)); scored[pk] = r
        pk_rows.append(dict(pk=float(pk), present=True, file=fn,
                            r2=r["r2"], eps_med=r["eps_med"], coverage=r["coverage"],
                            phi_span=r["phi_span"], S2=r["S_two_factor"]))

    present_pk = sorted(scored.keys())
    n_intermediate = sum(1 for p in present_pk if p in (3, 4, 5, 6, 7))

    # --- 1b. BRACKET the R^2=0 crossing (consecutive sign change) -------------
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

    # --- 2. phi_span robustness (B-L2-2) -------------------------------------
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
        band_inverts_at = [float(e) for e, a, b in zip(estars, pb_dL, pb_k) if not (a > b)]

    # --- 3. gap-invasion check (B-L2-4): classify EACH intermediate case ------
    L1npz = np.load(os.path.join(RESULTS, "rib_two_factor_methodology.npz"), allow_pickle=True)
    S2_thresh = float(L1npz["S2_threshold"])
    L1_S2 = np.asarray(L1npz["S_two_factor"], float)
    L1_r2 = np.asarray(L1npz["r2"], float)
    gap_lo = float(L1_S2[L1_r2 >= 0].max())
    gap_hi = float(L1_S2[L1_r2 < 0].min())
    per_case = []; invasions = []; inversion = False
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
    # inversion across the FULL set (intermediate + anchors)
    for pk in present_pk:
        for pk2 in present_pk:
            if (scored[pk]["r2"] >= 0) and (scored[pk2]["r2"] < 0) \
               and (scored[pk]["S_two_factor"] > scored[pk2]["S_two_factor"]):
                inversion = True
    discriminant_survives = (not inversion) and all(not iv["misclassified"] for iv in invasions)

    # --- 4. PERSIST (before any narrative assertion) -------------------------
    pk_arr = np.array(present_pk, float)
    r2_arr = np.array([scored[p]["r2"] for p in present_pk], float)
    out = dict(
        sweep_pk=pk_arr, sweep_r2=r2_arr, present_pk=pk_arr,
        anchor_pk=np.array([2.0, 8.0]), n_intermediate_present=int(n_intermediate),
        crossing_bracketed=bool(crossing_bracketed),
        crossing_bracketed_tight=bool(crossing_bracketed_tight),
        pk_window=np.array([PK_DK_LO, PK_DK_HI]),
        in_window=bool(in_window) if bracket is not None else False,
        estars=estars, S2_threshold=S2_thresh, gap_lo=gap_lo, gap_hi=gap_hi,
        span_order_ok=bool(span_order_ok) if span_order_ok is not None else False,
        band_order_ok=bool(band_order_ok) if band_order_ok is not None else False,
        band_inverts_at=np.array(band_inverts_at, float),
        discriminant_survives=bool(discriminant_survives), inversion=bool(inversion),
        hill_r2_guard=hill_r2, guard_ok=bool(abs(hill_r2 - HILL_R2_CANON) < 1e-6),
        Y_IDX=Y_IDX, EPS_STAR=EPS_STAR,
    )
    if bracket is not None:
        out.update({f"bracket_{k}": v for k, v in bracket.items()})
    for label in rob:
        out["rob_" + label.replace(" ", "_")] = rob[label]
    np.savez(os.path.join(RESULTS, "rib_pk_sweep_l2att2.npz"), **out)

    # measured manuscript-facing strings (so text edits are traceable to data)
    manuscript_values = dict(
        bracket=bracket,
        bracket_sentence=(
            (("a measured intermediate-$p/k$ RANS sweep ($p/k\\in\\{3,4,5,6,7\\}$) "
              "brackets the $R^2{=}0$ crossing between $p/k=%g$ ($R^2=%+.2f$) and "
              "$p/k=%g$ ($R^2=%+.2f$), giving $(p/k)_c=%.1f$ --- on the classical "
              "$d$-/$k$-type transition.")
             % (bracket["pk_lo"], bracket["r2_lo"], bracket["pk_hi"], bracket["r2_hi"],
                bracket["pk_cross"]))
            if crossing_bracketed_tight and in_window else
            (("a measured intermediate-$p/k$ sweep brackets the crossing at "
              "$(p/k)_c=%.1f$, OUTSIDE the classical window $[5,9]$ --- the $d/k$ "
              "bridge is revised accordingly.") % bracket["pk_cross"])
            if crossing_bracketed_tight else
            "the intermediate sweep does not yet bracket the crossing with a consecutive sign change."),
        phi_band_concession=(
            "the contiguous-band extent $\\phi_{\\rm band}$ (longest run of "
            "consecutive deep stations) inverts the d$>$k ordering at small "
            "$\\varepsilon^\\ast$ (the d-type deep stations are scattered, not "
            "contiguous); the robust, threshold-free extent axis is the geometric "
            "$p/k$, validated by the sweep."
            if band_inverts_at else
            "the contiguous-band extent preserves the d$>$k ordering at all $\\varepsilon^\\ast$."),
    )
    summary = dict(
        attempt=2, node="node_003",
        bind_B_L2_1=dict(
            n_intermediate_present=int(n_intermediate), present_pk=present_pk,
            sweep=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                   for row in pk_rows],
            crossing_bracketed=crossing_bracketed,
            crossing_bracketed_tight=crossing_bracketed_tight, bracket=bracket,
            in_classical_dk_window=in_window,
            verdict=(("TIGHT bracket (consecutive p/k) inside [5,9] -> located on classical d/k~7"
                      if in_window else
                      "TIGHT bracket but crossing OUTSIDE [5,9] -> bridge revised")
                     if crossing_bracketed_tight else
                     ("WIDE bracket only" if crossing_bracketed else "NOT yet bracketed"))),
        bind_B_L2_2=dict(
            estars=list(estars), span_order_d_gt_k=span_order_ok,
            band_order_d_gt_k=band_order_ok, band_inverts_at_estar=band_inverts_at,
            deployable_extent_axis="p_over_k (geometric, threshold-free; validated by sweep)"),
        bind_B_L2_4=dict(
            S2_threshold=round(S2_thresh, 4), gap=[round(gap_lo, 4), round(gap_hi, 4)],
            per_intermediate_case=per_case, invasions=invasions, inversion=inversion,
            discriminant_survives=discriminant_survives),
        manuscript_values=manuscript_values,
        guards=dict(hill_r2=hill_r2, guard_ok=out["guard_ok"], Y_IDX=Y_IDX, EPS_STAR=EPS_STAR,
                    node002_npz_md5=md5(os.path.join(RESULTS, "rib_pk_sweep_l2.npz")),
                    blade_severance_md5=md5(os.path.join(RESULTS, "blade_severance_l3.npz"))),
    )
    with open(os.path.join(NODE, "pk_sweep_l2att2.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _figure(present_pk, scored, bracket, rob, per_case, gap_lo, gap_hi, S2_thresh)

    # --- console report ------------------------------------------------------
    print("=" * 72)
    print("L2 att2 rib p/k sweep (node_003)  Y_IDX=%d  eps*=%.2f" % (Y_IDX, EPS_STAR))
    print("=" * 72)
    print("hill guard R^2 = %.8f  (ok=%s)" % (hill_r2, out["guard_ok"]))
    print("\nB-L2-1  R^2(p/k) sweep:")
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
    print("\nB-L2-2  span ordering d>k = %s ; band ordering d>k = %s ; band inverts at eps*=%s"
          % (span_order_ok, band_order_ok, band_inverts_at))
    print("\nB-L2-4  gap=[%.3f,%.3f]  S2*=%.3f  invasions=%d  inversion=%s  survives=%s"
          % (gap_lo, gap_hi, S2_thresh, len(invasions), inversion, discriminant_survives))
    for pc in per_case:
        print("   p/k=%g  R^2=%+.3f  S2=%.3f  in_gap=%s  misclassified=%s"
              % (pc["pk"], pc["r2"], pc["S2"], pc["in_gap"], pc["misclassified"]))
    print("\nwrote codes/results/rib_pk_sweep_l2att2.npz")
    print("wrote development/nodes/node_003/{pk_sweep_l2att2.json,fig_pk_sweep.{png,pdf}}")


def _figure(present_pk, scored, bracket, rob, per_case, gap_lo, gap_hi, S2_thresh):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    pks = np.array(present_pk, float)
    r2s = np.array([scored[p]["r2"] for p in present_pk], float)
    is_anchor = np.array([p in (2, 8) for p in present_pk])
    ax[0].axhline(0, color="0.6", lw=0.8, ls="--")
    ax[0].plot(pks, r2s, "-", color="0.4", lw=1.2, zorder=1)
    ax[0].scatter(pks[~is_anchor], r2s[~is_anchor], s=70, color="#1f77b4", zorder=3,
                  label="RANS sweep (new, measured)")
    ax[0].scatter(pks[is_anchor], r2s[is_anchor], s=90, marker="s", facecolor="none",
                  edgecolor="k", zorder=3, label="RANS anchors (p/k=2,8)")
    if bracket:
        ax[0].axvspan(bracket["pk_lo"], bracket["pk_hi"], color="orange", alpha=0.18,
                      label=r"$R^2{=}0$ bracket (width %g)" % bracket["width"])
        ax[0].axvline(bracket["pk_cross"], color="orange", lw=1.2, ls=":")
        ax[0].annotate(r"$(p/k)_c=%.1f$" % bracket["pk_cross"], (bracket["pk_cross"], 0.0),
                       textcoords="offset points", xytext=(6, 8), color="darkorange", fontsize=9)
    ax[0].axvspan(5, 9, color="green", alpha=0.06, zorder=0)
    ax[0].text(7, ax[0].get_ylim()[0], "classical d/k", color="green",
               ha="center", va="bottom", fontsize=8)
    ax[0].set_xlabel(r"pitch-to-height ratio $p/k$")
    ax[0].set_ylabel(r"$R^2(\tau_w)$  (a-priori ODE)")
    ax[0].set_title("(a) ODE validity boundary vs roughness regime (measured)")
    ax[0].legend(fontsize=8, loc="lower right")

    colors = {"rib LES d-type": "#d62728", "rib RANS d-type": "#ff7f0e",
              "rib RANS k-type": "#2ca02c"}
    for label, arr in rob.items():
        c = colors.get(label, "0.3")
        ax[1].plot(arr[:, 0], arr[:, 2], "-o", color=c, ms=3, lw=1.3,
                   label=label + r" $\phi_{\rm span}$")
        ax[1].plot(arr[:, 0], arr[:, 3], "--s", color=c, ms=3, lw=1.0, alpha=0.8,
                   label=label + r" $\phi_{\rm band}$")
    ax[1].set_xlabel(r"deep-cancellation threshold $\varepsilon^\ast$")
    ax[1].set_ylabel(r"streamwise extent $\phi$")
    ax[1].set_title(r"(b) EXTENT robustness: convex-hull vs contiguous band")
    ax[1].legend(fontsize=6.5, ncol=1, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(NODE, "fig_pk_sweep." + ext), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
