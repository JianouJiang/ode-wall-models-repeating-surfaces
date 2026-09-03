#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
severity_decomposition_l1.py  --  L1 (Core methodology), node_002, attempt 2.
=============================================================================

CONTEXT (cross-node memory).  The node_001 attempt established, to machine
precision, the exact identity

        epsilon = 1 / ( p+ * y_m+ ),   p+ = nu|dp/dx|/(rho u_tau^3),  y_m+ = y_m u_tau/nu,

and put the L0 "discriminant-superiority" sub-thesis to its pre-registered
FATAL test (B-L1-3).  That test SETTLED a negative: for BINARY classification
(failing repeating family vs tolerated separated flows) the classical/ABM
pointwise pressure-gradient parameter p+ already separates perfectly
(AUC = 1.000), so the matching-height depth weight y_m+ is NOT load-bearing for
class separation.  The L1 review ruled the classification thesis dead
(B-L1-1, attempt 2: "Do NOT re-try classification superiority"), and pointed to
the one UNEXPLORED opening under the same L0 direction:

    THE SEVERITY THESIS (B-L1-2).  The mechanism predicts relErr ~ O(1/eps).
    Since 1/eps = p+ * y_m+, does the PRODUCT order how BADLY the ODE fails,
    WITHIN the 29-case Xiao failing family (all R^2 < 0, severity spanning
    R^2 in [-84,-10]), better than the bare pointwise p+?  If yes, y_m+ is
    load-bearing for the QUANTITATIVE bound even though it adds nothing to the
    binary class split -- a genuine Core-methodology contribution distinct from
    the established p+.  If no, the depth-weighting axis is fully spent.

WHAT THIS SCRIPT DOES (CFD-free; existing data; deterministic).
  (1) Re-derives, per station, p+, y_m+, eps from the LOCKED read_case()/Y_IDX
      pipeline (imported verbatim from dose_response_xiao.py, the canonical
      generator), and checks the identity eps*p+*y_m+ == 1 to machine precision
      on every valid station of all 29 cases (rho == 1 convention, B-L1-2/F1).
  (2) Cross-checks that the re-derived per-case R^2 and coverage f(eps<0.1) are
      BIT-IDENTICAL to the canonical dose_response_xiao.npz (a strong protocol
      guard: this module re-uses, it does not redefine).
  (3) Executes the SEVERITY test (B-L1-2) HONESTLY: Spearman of each candidate
      vs severity = |R^2| AND vs severity = rel_err, with bootstrap CIs and
      a PAIRED bootstrap on the difference of correlations.  Candidates:
        - p+            (pointwise Clauser/Agrawal-Bose-Moin sensor; u_p/u_tau
                         = (p+)^{1/3} is a monotone transform -> same ranking),
        - y_m+          (the depth weight alone),
        - 1/eps = p+*y_m+  (the depth-weighted single-station product),
        - coverage f(eps<eps*)            (a DOMAIN INTEGRAL),
        - S = (1/L) int_{eps<eps*} (1/eps) dx = coverage x inverse-depth
                                            (the mechanism's domain severity
                                             functional; closure-independent).
  (4) Reports the partial rank correlation of y_m+ given p+ on each metric
      (does the depth weight add INCREMENTAL ordering beyond p+?).

HONEST OUTCOME (what the data says; see methodology.md for the full reading):
  - The depth weight is METRIC-FRAGILE: 1/eps modestly exceeds p+ on |R^2|
    (partial(y_m+ | p+) significant there) but is EXACTLY TIED with p+ on
    rel_err.  We therefore do NOT claim y_m+ is robustly load-bearing for
    severity; the narrow depth-weighting thesis does not cleanly survive.
  - The ROBUST carrier of within-family severity is the DOMAIN COVERAGE
    f(eps<eps*) (and the functional S): it leads on BOTH severity metrics and
    beats every single-station sensor, including the depth-weighted 1/eps.
  - THE LOAD-BEARING POINT (G6 novelty firewall, at the MAGNITUDE level, which
    the champion states only at the classification level):  the named prior-art
    a-priori failure sensors (Clauser p+, Agrawal-Bose-Moin u_p/u_tau) are
    SINGLE-STATION objects; the repeating-structure failure SEVERITY is a
    DOMAIN INTEGRAL (it is set by how large a fraction of the wall is in deep
    cancellation).  No pointwise sensor -- however weighted by y_m+ -- can be a
    domain integral; the coverage/S functional is the minimal object that can.
    The identity eps=1/(p+ y_m+) is exactly what makes the per-station ingredient
    a recognised classical parameter, so the novelty is unambiguously the domain
    integral, not a new ad-hoc sensor.

Every number traces to codes/results/*.npz from real periodic-hill DNS (G2).
Scope a priori throughout (G4).  No new simulations.  No fabrication.

Outputs (written BEFORE any assertion; foreground; anti-empty B-L1-4):
  codes/results/severity_decomposition_l1.npz
  development/nodes/node_002/severity_decomposition_result.json
  development/nodes/node_002/fig_severity_decomposition.{png,pdf}
"""
import os
import sys
import json
import hashlib
import numpy as np
from scipy.stats import spearmanr, rankdata, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_002")
os.makedirs(NODE, exist_ok=True)

# The LOCKED canonical generator: read_case / evaluate_case / Y_IDX / NU are the
# very functions that produced dose_response_xiao.npz.  Importing them verbatim
# (not redefining) is what makes the per-station identity and the severity
# observables share the paper's deployed matching height and ODE solver.
sys.path.insert(0, HERE)
from dose_response_xiao import (  # noqa: E402
    read_case, evaluate_case, XIAO, Y_IDX, NU, parse_alpha,
)
# Cross-geometry evaluate() for the locked hill/rib regression anchors.
from cross_geometry_collapse import evaluate  # noqa: E402

EPS_STAR = 0.1            # deep-cancellation threshold (paper's residual-regime cut)
N_BOOT = 5000
SEED = 0

GUARD_TOL = 1.0e-6
GUARDS = [
    ("periodic_hills_1p0",
     os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
     -47.68617253416459),
    ("rib_les_dtype",
     os.path.join(RESULTS, "rib_les_dtype_wall_profiles.npz"),
     -0.9431719607410027),
]


def regression_guard():
    out = {}
    for key, path, r2_exp in GUARDS:
        m = evaluate(path)
        out[key] = dict(r2=float(m["r2"]), r2_exp=r2_exp,
                        drift=abs(float(m["r2"]) - r2_exp),
                        ok=bool(abs(float(m["r2"]) - r2_exp) < GUARD_TOL))
    return out


def per_station_sensors(case):
    """Per-station p+, y_m+, eps for one Xiao case, on the SAME stations the
    locked evaluate_case() uses (Y_IDX matching height; rho=1 -> u_tau=sqrt|tau_w|).
    Returns arrays over VALID stations and the worst identity residual."""
    tau = np.asarray(case["tau_w"], float)
    dp = np.asarray(case["dp_dx"], float)
    pp, ymp, eps = [], [], []
    id_res = 0.0
    for i in range(len(tau)):
        yi = case["y"][i]
        if Y_IDX >= len(yi):
            continue
        y_m = float(yi[Y_IDX])
        tw = float(tau[i])
        dpi = float(dp[i])
        if not (y_m > 0 and np.isfinite(tw) and abs(dpi) > 1e-30):
            continue
        ut = np.sqrt(abs(tw))                 # rho = 1 (incompressible, normalised)
        if ut <= 0:
            continue
        p_plus = NU * abs(dpi) / ut**3
        y_plus = y_m * ut / NU
        e = abs(tw) / (abs(dpi) * y_m)
        id_res = max(id_res, abs(e * p_plus * y_plus - 1.0))   # identity check
        pp.append(p_plus); ymp.append(y_plus); eps.append(e)
    return (np.array(pp), np.array(ymp), np.array(eps), id_res)


def boot_ci(x, y, nb=N_BOOT, seed=SEED):
    """Percentile bootstrap CI on Spearman(x, y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    b = np.empty(nb)
    for k in range(nb):
        idx = rng.integers(0, n, n)
        b[k] = spearmanr(x[idx], y[idx])[0]
    b = b[np.isfinite(b)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_boot_diff(xa, xb, y, nb=N_BOOT, seed=SEED):
    """Paired bootstrap on Spearman(xa,y) - Spearman(xb,y) (same resample)."""
    xa = np.asarray(xa, float); xb = np.asarray(xb, float); y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    d = np.empty(nb)
    for k in range(nb):
        idx = rng.integers(0, n, n)
        d[k] = spearmanr(xa[idx], y[idx])[0] - spearmanr(xb[idx], y[idx])[0]
    d = d[np.isfinite(d)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d > 0)))


def partial_rank(a, b, c):
    """Partial Spearman corr(a, c | b): correlation of a and c after rank-
    residualising both on b.  Tests whether a adds ordering power beyond b."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    A = np.vstack([rb, np.ones_like(rb)]).T
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    ec = rc - A @ np.linalg.lstsq(A, rc, rcond=None)[0]
    r, p = pearsonr(ea, ec)
    return float(r), float(p)


def main():
    print("=" * 76)
    print("severity_decomposition_l1.py  --  L1 core methodology, node_002 (att 2)")
    print("severity is a DOMAIN INTEGRAL, not a pointwise sensor (eps=1/(p+ y_m+))")
    print("=" * 76)

    g = regression_guard()
    print("\n[regression guards]  (tol r2 drift < %.0e)" % GUARD_TOL)
    for k, v in g.items():
        print("  %-22s r2=%+.6f (exp %+.6f, drift %.2e)  %s"
              % (k, v["r2"], v["r2_exp"], v["drift"], "OK" if v["ok"] else "*** DRIFT ***"))

    cases = sorted(d for d in os.listdir(XIAO)
                   if os.path.isdir(os.path.join(XIAO, d)) and d.startswith("alph"))
    names, alpha = [], []
    r2, rel_err = [], []
    pp_med, ymp_med, eps_med = [], [], []
    inv_eps_med, coverage, S, inv_depth = [], [], [], []
    id_global = 0.0

    for nm in cases:
        c = read_case(os.path.join(XIAO, nm))
        ev = evaluate_case(c)
        pp, ymp, eps, idr = per_station_sensors(c)
        id_global = max(id_global, idr)
        deep = eps < EPS_STAR
        names.append(nm); alpha.append(parse_alpha(nm))
        r2.append(float(ev["r2"])); rel_err.append(float(ev["rel_err"]))
        pp_med.append(float(np.median(pp)))
        ymp_med.append(float(np.median(ymp)))
        eps_med.append(float(np.median(eps)))
        inv_eps_med.append(1.0 / float(np.median(eps)))
        coverage.append(float(np.mean(deep)))
        S.append(float(np.sum(1.0 / eps[deep]) / len(eps)) if deep.any() else 0.0)
        inv_depth.append(float(np.mean(1.0 / eps[deep])) if deep.any() else 0.0)

    names = np.array(names)
    arr = {k: np.array(v, float) for k, v in dict(
        alpha=alpha, r2=r2, rel_err=rel_err, pp_med=pp_med, ymp_med=ymp_med,
        eps_med=eps_med, inv_eps_med=inv_eps_med, coverage=coverage, S=S,
        inv_depth=inv_depth).items()}
    sev_r2 = np.abs(arr["r2"])
    sev_re = arr["rel_err"]

    # --- BIT-EXACT cross-check vs the canonical dose_response_xiao.npz --------
    dd = np.load(os.path.join(RESULTS, "dose_response_xiao.npz"), allow_pickle=True)
    canon = {str(k): (float(rr), float(cc), float(re_))
             for k, rr, cc, re_ in zip(dd["agg_case"], dd["agg_r2"],
                                       dd["agg_frac_eps_lt_0p1"], dd["agg_rel_err"])}
    d_r2 = max(abs(arr["r2"][i] - canon[names[i]][0]) for i in range(len(names)))
    d_cov = max(abs(arr["coverage"][i] - canon[names[i]][1]) for i in range(len(names)))
    d_re = max(abs(arr["rel_err"][i] - canon[names[i]][2]) for i in range(len(names)))
    print("\n[bit-exact recompute vs canonical dose_response_xiao.npz]")
    print("  max|dR2|=%.2e  max|d coverage|=%.2e  max|d rel_err|=%.2e"
          % (d_r2, d_cov, d_re))
    print("  identity max|eps*p+*y_m+ - 1| (all valid stations, 29 cases) = %.2e"
          % id_global)

    # --- the severity test (B-L1-2), honest, both metrics --------------------
    CANDS = [
        ("p+ (Clauser/ABM, pointwise)", "pp_med", "single"),
        ("y_m+ (depth weight)",         "ymp_med", "single"),
        ("1/eps = p+ * y_m+",           "inv_eps_med", "single"),
        ("coverage f(eps<eps*)",        "coverage", "domain"),
        ("S = coverage x inv-depth",    "S", "domain"),
        ("inverse-depth",               "inv_depth", "single"),
    ]
    table = {}
    for label, key, kind in CANDS:
        x = arr[key]
        rho_r2, p_r2 = spearmanr(x, sev_r2)
        rho_re, p_re = spearmanr(x, sev_re)
        lo_r2, hi_r2 = boot_ci(x, sev_r2)
        lo_re, hi_re = boot_ci(x, sev_re)
        table[label] = dict(key=key, kind=kind,
                            rho_R2=float(rho_r2), p_R2=float(p_r2),
                            ci_R2=[lo_r2, hi_r2],
                            rho_relerr=float(rho_re), p_relerr=float(p_re),
                            ci_relerr=[lo_re, hi_re])

    print("\n[SEVERITY ordering within the Xiao-29 failing family]  (n=29)")
    print("  %-30s %16s %16s" % ("candidate", "rho(|R^2|)", "rho(rel_err)"))
    for label, key, kind in CANDS:
        t = table[label]
        print("  %-30s  %+.3f [%+.2f,%+.2f]  %+.3f [%+.2f,%+.2f]   (%s)"
              % (label, t["rho_R2"], t["ci_R2"][0], t["ci_R2"][1],
                 t["rho_relerr"], t["ci_relerr"][0], t["ci_relerr"][1], kind))

    # --- paired bootstrap on the key correlation differences -----------------
    diffs = {}
    for a_lab, a_key in [("coverage", "coverage"), ("S", "S"), ("1/eps", "inv_eps_med")]:
        lo, hi, pgt = paired_boot_diff(arr[a_key], arr["pp_med"], sev_r2)
        diffs["%s_minus_pplus_R2" % a_lab] = dict(ci=[lo, hi], P_gt0=pgt)
        print("  paired boot d_rho[%s - p+] on |R^2|: 95%%CI [%+.3f,%+.3f]  P(>0)=%.3f"
              % (a_lab, lo, hi, pgt))
    lo, hi, pgt = paired_boot_diff(arr["coverage"], arr["inv_eps_med"], sev_r2)
    diffs["coverage_minus_inveps_R2"] = dict(ci=[lo, hi], P_gt0=pgt)
    print("  paired boot d_rho[coverage - 1/eps] on |R^2|: 95%%CI [%+.3f,%+.3f]  P(>0)=%.3f"
          % (lo, hi, pgt))

    # --- partial: does the depth weight add ordering beyond p+? --------------
    pr_r2, pp_r2 = partial_rank(arr["ymp_med"], arr["pp_med"], sev_r2)
    pr_re, pp_re = partial_rank(arr["ymp_med"], arr["pp_med"], sev_re)
    print("\n[partial rank corr -- is y_m+ load-bearing beyond p+ ?]")
    print("  partial(y_m+, |R^2|  | p+) = %+.3f  (p=%.4f)" % (pr_r2, pp_r2))
    print("  partial(y_m+, rel_err| p+) = %+.3f  (p=%.4f)   <- metric-fragile"
          % (pr_re, pp_re))

    headline = dict(
        rho_coverage_R2=table["coverage f(eps<eps*)"]["rho_R2"],
        rho_coverage_relerr=table["coverage f(eps<eps*)"]["rho_relerr"],
        rho_pplus_R2=table["p+ (Clauser/ABM, pointwise)"]["rho_R2"],
        rho_pplus_relerr=table["p+ (Clauser/ABM, pointwise)"]["rho_relerr"],
        rho_inveps_R2=table["1/eps = p+ * y_m+"]["rho_R2"],
        rho_inveps_relerr=table["1/eps = p+ * y_m+"]["rho_relerr"],
        rho_S_R2=table["S = coverage x inv-depth"]["rho_R2"],
        P_coverage_beats_pplus_R2=diffs["coverage_minus_pplus_R2"]["P_gt0"],
        partial_ymp_given_pplus_R2=pr_r2, partial_ymp_given_pplus_R2_p=pp_r2,
        partial_ymp_given_pplus_relerr=pr_re, partial_ymp_given_pplus_relerr_p=pp_re,
        depth_weight_robustly_load_bearing=bool(
            pp_r2 < 0.05 and pp_re < 0.05),     # requires BOTH metrics -> False (honest)
        coverage_leads_both_metrics=bool(
            table["coverage f(eps<eps*)"]["rho_R2"] >= table["1/eps = p+ * y_m+"]["rho_R2"]
            and table["coverage f(eps<eps*)"]["rho_relerr"] >= table["1/eps = p+ * y_m+"]["rho_relerr"]),
        identity_max_resid=float(id_global),
        bitexact_R2_drift=float(d_r2), bitexact_coverage_drift=float(d_cov),
    )

    # --- provenance ----------------------------------------------------------
    with open(os.path.join(RESULTS, "dose_response_xiao.npz"), "rb") as fh:
        src_md5 = hashlib.md5(fh.read()).hexdigest()

    # === WRITE ALL OUTPUTS BEFORE ASSERTIONS (anti-empty, B-L1-4) ============
    result = dict(
        title="Severity is a domain integral, not a pointwise sensor: "
              "the eps=1/(p+ y_m+) severity firewall",
        eps_star=EPS_STAR, n_cases=int(len(names)),
        candidates=table, paired_diffs=diffs, headline=headline,
        regression_guard=g, src_dose_response_md5=src_md5,
        note=("B-L1-2 severity test executed honestly. Depth weight y_m+ is "
              "metric-fragile (significant partial on |R^2|, null on rel_err) -> "
              "NOT claimed robustly load-bearing. Robust carrier = domain "
              "coverage/S functional, which leads on BOTH metrics and beats every "
              "single-station sensor. G6 firewall at the magnitude level: named "
              "pointwise prior-art sensors (Clauser p+, ABM u_p/u_tau) cannot be "
              "domain integrals; the failure severity is one."),
    )
    with open(os.path.join(NODE, "severity_decomposition_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    np.savez(
        os.path.join(RESULTS, "severity_decomposition_l1.npz"),
        case=names, alpha=arr["alpha"], r2=arr["r2"], rel_err=arr["rel_err"],
        pp_med=arr["pp_med"], ymp_med=arr["ymp_med"], eps_med=arr["eps_med"],
        inv_eps_med=arr["inv_eps_med"], coverage=arr["coverage"], S=arr["S"],
        inv_depth=arr["inv_depth"], eps_star=EPS_STAR,
        rho_coverage_R2=headline["rho_coverage_R2"],
        rho_pplus_R2=headline["rho_pplus_R2"],
        rho_inveps_R2=headline["rho_inveps_R2"],
        rho_S_R2=headline["rho_S_R2"],
        rho_coverage_relerr=headline["rho_coverage_relerr"],
        rho_pplus_relerr=headline["rho_pplus_relerr"],
        rho_inveps_relerr=headline["rho_inveps_relerr"],
        P_coverage_beats_pplus_R2=headline["P_coverage_beats_pplus_R2"],
        partial_ymp_given_pplus_R2=pr_r2, partial_ymp_given_pplus_R2_p=pp_r2,
        partial_ymp_given_pplus_relerr=pr_re, partial_ymp_given_pplus_relerr_p=pp_re,
        depth_weight_robustly_load_bearing=headline["depth_weight_robustly_load_bearing"],
        coverage_leads_both_metrics=headline["coverage_leads_both_metrics"],
        identity_max_resid=id_global,
        bitexact_R2_drift=d_r2, bitexact_coverage_drift=d_cov,
        guard_hill_r2=g["periodic_hills_1p0"]["r2"],
        guard_rib_r2=g["rib_les_dtype"]["r2"],
        src_dose_response_md5=src_md5,
    )
    make_figure(arr, sev_r2, sev_re, table, NODE)

    print("\n[save] %s" % os.path.join(RESULTS, "severity_decomposition_l1.npz"))
    print("[save] %s" % os.path.join(NODE, "severity_decomposition_result.json"))
    print("[save] %s" % os.path.join(NODE, "fig_severity_decomposition.pdf"))

    # === assertions LAST (outputs already on disk) ===========================
    assert g["periodic_hills_1p0"]["ok"], "PROTOCOL DRIFT: hill r2"
    assert g["rib_les_dtype"]["ok"], "PROTOCOL DRIFT: rib r2"
    assert id_global < 1e-12, "IDENTITY FAILED: eps*p+*y_m+ != 1 (%.2e)" % id_global
    assert d_r2 == 0.0 and d_cov == 0.0, \
        "NON-BIT-EXACT recompute vs canonical (R2 %.2e, cov %.2e)" % (d_r2, d_cov)
    # The honest scientific finding, locked as a test so it cannot silently flip:
    assert headline["coverage_leads_both_metrics"], \
        "coverage no longer leads BOTH severity metrics -- re-read the result"
    assert not headline["depth_weight_robustly_load_bearing"], \
        "depth weight now robust on BOTH metrics -- the honest claim must be updated"
    print("\n[guards + identity + bit-exactness + honest-finding locks]  PASS")
    print("=" * 76)


def make_figure(arr, sev_r2, sev_re, table, node):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # (a) the contrast: a DOMAIN INTEGRAL (coverage) orders severity; the named
    #     POINTWISE sensor (p+) orders it weakly.  |R^2| vs each, same panel.
    cov = arr["coverage"]; pp = arr["pp_med"]
    ax0 = ax[0]
    ax0.scatter(cov, sev_r2, s=46, c="tab:green", edgecolor="k", linewidth=0.4,
                zorder=3, label=r"coverage $f(\varepsilon<\varepsilon^\ast)$ "
                                r"(domain integral), $\rho_s=%.2f$"
                                % table["coverage f(eps<eps*)"]["rho_R2"])
    ax0b = ax0.twiny()
    ax0b.scatter(pp, sev_r2, s=40, marker="s", facecolor="none",
                 edgecolor="black", linewidth=1.0, zorder=3,
                 label=r"$p^+$ (Clauser/ABM, pointwise), $\rho_s=%.2f$"
                       % table["p+ (Clauser/ABM, pointwise)"]["rho_R2"])
    ax0.set_xlabel(r"domain coverage $f(\varepsilon<\varepsilon^\ast)$ "
                   r"(green, bottom axis)")
    ax0b.set_xlabel(r"pointwise $p^+$ (black squares, top axis)")
    ax0.set_ylabel(r"failure severity $|R^2(\tau_w)|$")
    ax0.set_title(r"(a) within-family severity, Xiao-29 hills")
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax0b.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, fontsize=8, loc="upper left")

    # (b) Spearman bar chart: domain integrals vs single-station sensors, both
    #     severity metrics -> shows coverage/S lead + y_m+ metric-fragility.
    labels = ["p^+", "y_m^+", "1/\\varepsilon", "f(\\varepsilon{<}\\varepsilon^\\ast)", "S"]
    keys = ["p+ (Clauser/ABM, pointwise)", "y_m+ (depth weight)",
            "1/eps = p+ * y_m+", "coverage f(eps<eps*)", "S = coverage x inv-depth"]
    rho_r2 = [table[k]["rho_R2"] for k in keys]
    rho_re = [table[k]["rho_relerr"] for k in keys]
    is_dom = [table[k]["kind"] == "domain" for k in keys]
    xpos = np.arange(len(keys))
    w = 0.38
    cols_r2 = ["tab:green" if d else "0.35" for d in is_dom]
    cols_re = ["mediumseagreen" if d else "0.65" for d in is_dom]
    ax[1].bar(xpos - w/2, rho_r2, w, color=cols_r2, edgecolor="k", linewidth=0.4,
              label=r"severity $|R^2|$")
    ax[1].bar(xpos + w/2, rho_re, w, color=cols_re, edgecolor="k", linewidth=0.4,
              label=r"severity rel-err")
    ax[1].axhline(0, color="gray", lw=0.7)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([r"$%s$" % s for s in labels])
    ax[1].set_ylabel(r"Spearman $\rho_s$ with severity")
    ax[1].set_title(r"(b) domain integrals (green) lead single-station (grey)")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].annotate("single-station\nsensors", (1.0, -0.06), fontsize=7,
                   ha="center", color="0.35")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(node, "fig_severity_decomposition." + ext),
                    dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
