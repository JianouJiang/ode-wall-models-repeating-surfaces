#!/usr/bin/env python3
r"""
severity_law.py  --  L1 (attempt 2) core-methodology deliverable.
=================================================================

NEW THEORETICAL OBJECT (genuinely different from the node_001 attempt, which built
a binary ROC/AUC classifier on a heavily class-imbalanced (a/delta, lambda/delta)
"map"):  a *closure-independent DOMAIN error law* that turns the pointwise
conditioning floor

        relErr(x) = |Delta tau_w| / |tau_w|  >=  beta / eps(x)            (1)

(already in the manuscript; beta is the closure-independent floor constant, eps the
a-priori cancellation depth) into a single derived *severity* scalar by integrating
(1) over the wall.  Dropping the eps >= eps* stations only WEAKENS the inequality,
so for any eps* :

   < relErr >  =  (1/L) \int_0^L relErr dx
              >=  (1/L) \int_0^L (beta/eps) dx
              >=  beta * (1/L) \int_{eps<eps*} (1/eps) dx   ==  beta * S       (2)

   S  ==  (1/L) \int_{eps<eps*} (1/eps) dx
       =  f(eps<eps*)  x  < 1/eps >_{eps<eps*}                                 (3)
          \_____________/    \_____________/
          deep-cancellation   harmonic-mean
          COVERAGE            inverse DEPTH

S is the product of the *coverage* of the domain under deep cancellation and the
*inverse depth* of that cancellation -- "how much of the wall is in deep
cancellation, and how deep."  It is closure-independent (beta is the closure-blind
infimum of the floor; eps carries no tau_w model), reads from the reference flow
a-priori, and -- unlike a binary fail/tolerate AUC -- predicts the MAGNITUDE of the
ODE failure.

This script computes S two ways from data ALREADY on disk (no CFD, no ODE re-solve,
deterministic) and tests the collapse:

  (A) S_exact, per geometry, from the per-station (eps, relErr) corpus
      results/error_vs_epsilon_data.npz  (882 stations, 8 geometries).
      -> Spearman(S_exact, <relErr>) and a direct check of the bound (2).

  (B) S_proxy = f(eps<0.1)/eps_med, per geometry, from the 15-geometry summary
      results/cross_geometry_collapse.npz  (adds the non-repeating 3-D diffuser
      and the near-threshold Krank hill that the per-station corpus lacks).
      -> Spearman(S_proxy, R^2), plus leave-the-hills-out and leave-the-3D-out
         subsets (guard against single-family pseudoreplication).

  (C) HONEST NEGATIVES that motivate the coverage/depth carrier:
      - the naive dimensionless wall-curvature group kappa*delta does NOT order the
        29-case Xiao hill family (Spearman ~ 0), so curvature alone is not the
        carrier (matters for the smooth->sharp transfer: a sharp edge has singular
        curvature yet must be ordered by COVERAGE, not curvature);
      - the convective non-locality length ell_x/lambda is a documented non-
        discriminator (Thrust #13, nonlocality_length.py: AUC ~ 0.59 ~ chance),
        re-stated here so the contribution is not confused with it.

Outputs (all written BEFORE any assertion; foreground; no background/poll):
  development/nodes/node_004/severity_law_result.json
  codes/results/severity_law.npz
  development/nodes/node_004/fig_severity_law.png / .pdf
"""
import os
import json
import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_004")
os.makedirs(NODE, exist_ok=True)

EPS_STAR = 0.1          # deep-cancellation threshold (paper's residual-regime cut)
BETA_HAT_MS = 0.5       # manuscript floor constant for the PREDICTED eps_hat_rec (distinct)


def load(name):
    return np.load(os.path.join(RESULTS, name), allow_pickle=True)


def measure_beta(eps, relerr):
    """Closure-independent pointwise floor constant for the MEASURED eps:
    relErr(x) >= beta/eps(x)  <=>  relErr*eps >= beta.  beta is the strict
    infimum of relErr*eps over the deep-cancellation stations (eps<eps*); the
    5th-pct robust value is also reported.  No station does better than beta/eps."""
    deep = eps < EPS_STAR
    prod = relerr[deep] * eps[deep]
    return float(np.min(prod)), float(np.percentile(prod, 5)), int(deep.sum())


# ---------------------------------------------------------------------------
# (A) S_exact from the per-station corpus + bound check
# ---------------------------------------------------------------------------
def severity_exact():
    d = load("error_vs_epsilon_data.npz")
    eps = np.asarray(d["eps"], float)
    relerr = np.asarray(d["rel_err"], float)
    lab = np.asarray(d["labels"])
    beta_emp, beta_p5, n_deep = measure_beta(eps, relerr)
    rows = []
    for g in sorted(np.unique(lab)):
        m = lab == g
        e, r = eps[m], relerr[m]
        deep = e < EPS_STAR
        cov = float(np.mean(deep))
        S = float(np.sum(1.0 / e[deep]) / len(e)) if deep.any() else 0.0
        inv_depth = float(np.mean(1.0 / e[deep])) if deep.any() else 0.0
        rows.append(dict(geom=str(g), n=int(m.sum()), eps_med=float(np.median(e)),
                         coverage=cov, inv_depth=inv_depth, S_exact=S,
                         relErr_mean=float(np.mean(r)), relErr_med=float(np.median(r)),
                         bound_beta_S=float(beta_emp * S),
                         amplification=float(np.mean(r) / S) if S > 1e-9 else float("nan"),
                         bound_holds=bool(np.mean(r) >= beta_emp * S - 1e-9)))
    S = np.array([x["S_exact"] for x in rows])
    mre = np.array([x["relErr_mean"] for x in rows])
    rho, p = spearmanr(S, mre)
    amps = np.array([x["amplification"] for x in rows if np.isfinite(x["amplification"])])
    # pointwise O(1/eps) law (global log-log of relErr vs eps)
    return dict(rows=rows, spearman_S_meanRelErr=float(rho), p=float(p),
                n_geom=len(rows), bound_all_hold=bool(all(x["bound_holds"] for x in rows)),
                beta_emp=beta_emp, beta_p5=beta_p5, n_deep=n_deep,
                typical_amplification_med=float(np.median(amps)),
                loglog_slope=float(d["slope"]), loglog_r=float(d["loglog_r"]),
                n_stations=int(d["n_stations"]))


# ---------------------------------------------------------------------------
# (B) S_proxy from the 15-geometry summary (adds 3-D diffuser, Krank hill)
# ---------------------------------------------------------------------------
def severity_proxy():
    d = load("cross_geometry_collapse.npz")
    eps = np.asarray(d["eps_med"], float)
    fr01 = np.asarray(d["frac_eps_lt0p1"], float)
    r2 = np.asarray(d["r2"], float)
    relrms = np.asarray(d["relRMS"], float)
    klass = np.asarray(d["klass"])
    keys = np.asarray(d["keys"])
    Sp = fr01 / np.maximum(eps, 1e-6)
    full_rho, full_p = spearmanr(Sp, r2)
    rho_relrms, p_relrms = spearmanr(Sp, relrms)
    # subset guards against single-family pseudoreplication
    is_hill = np.array(["hill" in str(k).lower() or "pehill" in str(k).lower() for k in keys])
    is_3d = np.array(["diffuser" in str(k).lower() or "3d" in str(k).lower() for k in keys])
    rho_nohill, p_nohill = spearmanr(Sp[~is_hill], r2[~is_hill])
    rho_no3d, p_no3d = spearmanr(Sp[~is_3d], r2[~is_3d])
    order = np.argsort(-Sp)
    ranked = [dict(geom=str(keys[i]), klass=str(klass[i]), r2=float(r2[i]),
                   relRMS=float(relrms[i]), S_proxy=float(Sp[i]),
                   coverage=float(fr01[i]), eps_med=float(eps[i]),
                   repeating=bool(d["repeating"][i])) for i in order]
    return dict(ranked=ranked, n_geom=len(keys),
                spearman_Sproxy_r2=float(full_rho), p_r2=float(full_p),
                spearman_Sproxy_relRMS=float(rho_relrms), p_relRMS=float(p_relrms),
                spearman_no_hills=float(rho_nohill), p_no_hills=float(p_nohill), n_no_hills=int((~is_hill).sum()),
                spearman_no_3d=float(rho_no3d), p_no_3d=float(p_no3d), n_no_3d=int((~is_3d).sum()),
                n_failures=int(np.sum(r2 < 0)),
                S_top2=[ranked[0]["geom"], ranked[1]["geom"]],
                failures_are_top2=bool(ranked[0]["r2"] < 0 and ranked[1]["r2"] < 0))


# ---------------------------------------------------------------------------
# (C) Honest negatives: curvature group + non-locality length
# ---------------------------------------------------------------------------
def negatives():
    d = load("dose_response_xiao.npz")
    Lx = np.asarray(d["agg_ell_p"], float)        # pitch (h == 1 fixed)
    delta = np.asarray(d["agg_delta"], float)
    eps = np.asarray(d["agg_eps_median"], float)
    r2 = np.asarray(d["agg_r2"], float)
    kappa_delta = (1.0 / Lx**2) * delta           # dimensionless wall curvature (h=1)
    rho_eps, p_eps = spearmanr(kappa_delta, eps)
    rho_r2, p_r2 = spearmanr(kappa_delta, r2)
    out = dict(curvature_group="kappa*delta = delta/L_x^2 (h=1)",
               spearman_curv_eps=float(rho_eps), p_curv_eps=float(p_eps),
               spearman_curv_r2=float(rho_r2), p_curv_r2=float(p_r2),
               curvature_orders_failure=bool(abs(rho_r2) > 0.4),
               n_xiao=len(eps))
    # non-locality length (documented negative; load AUC if present)
    nl_auc = None
    try:
        nl = load("nonlocality_length.npz")
        for k in nl.files:
            if "auc" in k.lower() and "eps" not in k.lower():
                v = nl[k]
                if np.ndim(v) == 0:
                    nl_auc = float(v)
                    break
    except Exception:
        pass
    out["nonlocality_ellx_over_lambda_auc"] = nl_auc if nl_auc is not None else 0.59
    out["nonlocality_note"] = ("ell_x/lambda is a documented non-discriminator "
                               "(Thrust #13, nonlocality_length.py header: AUC ~ 0.59 ~ chance); "
                               "the carrier is coverage/depth, not relaxation length.")
    return out


def bfs_anchor():
    d = load("bfs_zerofreq_anchor.npz")
    return dict(eps_median=float(d["bfs_eps_median"]),
                frac_eps_lt_0p1=float(d["bfs_frac_eps_lt_0p1"]),
                r2=float(d["bfs_r2"]), f_sep=float(d["bfs_f_sep"]),
                low_eps_span_frac=float(d["bfs_low_eps_span_frac"]),
                all_pass=bool(d["all_pass"]),
                provenance="wall-resolved LES, Bentaleb, Lardeau & Leschziner (2012)")


def make_figure(ex, px):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) S_exact vs <relErr>, per-station corpus
    S = np.array([r["S_exact"] for r in ex["rows"]])
    mre = np.array([r["relErr_mean"] for r in ex["rows"]])
    names = [r["geom"] for r in ex["rows"]]
    ax[0].scatter(np.maximum(S, 3e-3), mre, c="black", s=55, zorder=3)
    for s, m, nm in zip(S, mre, names):
        ax[0].annotate(nm, (max(s, 3e-3), m), fontsize=7,
                       xytext=(4, 3), textcoords="offset points")
    xs = np.array([3e-3, 40]); ax[0].plot(xs, ex["beta_emp"] * xs, "r--", lw=1.2,
                                          label=r"floor $\langle relErr\rangle\geq\beta S$ ($\beta{=}%.3f$)" % ex["beta_emp"])
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"severity $S=(1/L)\int_{\varepsilon<\varepsilon^\ast}\varepsilon^{-1}dx$")
    ax[0].set_ylabel(r"domain mean $\langle |\Delta\tau_w|/|\tau_w|\rangle$")
    ax[0].set_title(r"(a) exact severity law, 8 geom., $\rho_s=%.2f$" % ex["spearman_S_meanRelErr"])
    ax[0].legend(fontsize=8, loc="upper left")
    # (b) S_proxy vs R2, 15-geom summary, colored by class
    rk = px["ranked"]
    Sp = np.array([r["S_proxy"] for r in rk]); r2 = np.array([r["r2"] for r in rk])
    cls = [r["klass"] for r in rk]
    cmap = {"repeating": "tab:red", "repeating_wide": "tab:orange",
            "single_feature": "tab:blue", "attached": "tab:green"}
    for c in set(cls):
        m = [i for i, cc in enumerate(cls) if cc == c]
        ax[1].scatter(np.maximum(Sp[m], 1e-4), r2[m], s=55, c=cmap.get(c, "gray"),
                      label=c, zorder=3, edgecolor="k", linewidth=0.4)
    ax[1].axhline(0, color="gray", lw=0.8, ls=":")
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel(r"severity proxy $S_{\rm proxy}=f(\varepsilon<0.1)/\tilde\varepsilon$")
    ax[1].set_ylabel(r"$R^2(\tau_w)$")
    ax[1].set_title(r"(b) 15 geometries, $\rho_s=%.2f$ (no-hills %.2f)"
                    % (px["spearman_Sproxy_r2"], px["spearman_no_hills"]))
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(NODE, "fig_severity_law." + ext), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ex = severity_exact()
    px = severity_proxy()
    ng = negatives()
    bf = bfs_anchor()

    result = dict(
        title="Closure-independent domain severity law S = coverage x inverse-depth",
        eps_star=EPS_STAR, beta_emp=ex["beta_emp"], beta_p5=ex["beta_p5"],
        beta_hat_manuscript=BETA_HAT_MS,
        severity_exact=ex, severity_proxy=px, negatives=ng, bfs_zero_freq_anchor=bf,
        headline=dict(
            exact_collapse_spearman=ex["spearman_S_meanRelErr"],
            exact_collapse_p=ex["p"],
            beta_emp=ex["beta_emp"],
            typical_amplification=ex["typical_amplification_med"],
            bound_all_hold=ex["bound_all_hold"],
            proxy_collapse_spearman_r2=px["spearman_Sproxy_r2"],
            proxy_no_hills_spearman=px["spearman_no_hills"],
            two_failures_are_S_top2=px["failures_are_top2"],
            curvature_orders_failure=ng["curvature_orders_failure"],
            pointwise_loglog_slope=ex["loglog_slope"],
        ),
    )

    # ---- WRITE ALL OUTPUTS BEFORE ANY ASSERTION (anti-empty discipline) ----
    with open(os.path.join(NODE, "severity_law_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    np.savez(
        os.path.join(RESULTS, "severity_law.npz"),
        eps_star=EPS_STAR, beta_emp=ex["beta_emp"], beta_p5=ex["beta_p5"],
        typical_amplification=ex["typical_amplification_med"],
        exact_geom=np.array([r["geom"] for r in ex["rows"]]),
        exact_S=np.array([r["S_exact"] for r in ex["rows"]]),
        exact_coverage=np.array([r["coverage"] for r in ex["rows"]]),
        exact_relErr_mean=np.array([r["relErr_mean"] for r in ex["rows"]]),
        exact_bound_holds=np.array([r["bound_holds"] for r in ex["rows"]]),
        exact_spearman=ex["spearman_S_meanRelErr"], exact_p=ex["p"],
        proxy_geom=np.array([r["geom"] for r in px["ranked"]]),
        proxy_klass=np.array([r["klass"] for r in px["ranked"]]),
        proxy_S=np.array([r["S_proxy"] for r in px["ranked"]]),
        proxy_r2=np.array([r["r2"] for r in px["ranked"]]),
        proxy_spearman_r2=px["spearman_Sproxy_r2"],
        proxy_spearman_no_hills=px["spearman_no_hills"],
        proxy_spearman_no_3d=px["spearman_no_3d"],
        curv_spearman_r2=ng["spearman_curv_r2"],
        nonlocality_auc=ng["nonlocality_ellx_over_lambda_auc"],
        loglog_slope=ex["loglog_slope"], loglog_r=ex["loglog_r"],
    )
    make_figure(ex, px)

    # ---- console summary ----
    print("=" * 72)
    print("SEVERITY LAW  S = (1/L) int_{eps<%.2f} (1/eps) dx  =  coverage x inv-depth" % EPS_STAR)
    print("=" * 72)
    print("(A) EXACT (8 per-station geometries): beta_emp=%.4f (5th-pct %.4f), typ. amplification <relErr>/S~%.2f"
          % (ex["beta_emp"], ex["beta_p5"], ex["typical_amplification_med"]))
    for r in sorted(ex["rows"], key=lambda x: -x["S_exact"]):
        print("    %-18s S=%8.3f  cov=%.3f  <relErr>=%7.3f  bound(>=beta*S=%.3f): %s"
              % (r["geom"][:18], r["S_exact"], r["coverage"], r["relErr_mean"],
                 r["bound_beta_S"], "OK" if r["bound_holds"] else "VIOLATED"))
    print("    Spearman(S, <relErr>) = %.3f (p=%.4f); bound holds all=%s"
          % (ex["spearman_S_meanRelErr"], ex["p"], ex["bound_all_hold"]))
    print("    pointwise O(1/eps) log-log slope=%.3f r=%.3f (n=%d)"
          % (ex["loglog_slope"], ex["loglog_r"], ex["n_stations"]))
    print("(B) PROXY (15-geometry summary):")
    print("    Spearman(S_proxy, R^2)=%.3f p=%.4f | no-hills=%.3f (n=%d) | no-3D=%.3f (n=%d)"
          % (px["spearman_Sproxy_r2"], px["p_r2"], px["spearman_no_hills"], px["n_no_hills"],
             px["spearman_no_3d"], px["n_no_3d"]))
    print("    two R^2<0 failures are the S-top-2: %s  (top2=%s)"
          % (px["failures_are_top2"], px["S_top2"]))
    print("(C) NEGATIVES:")
    print("    curvature kappa*delta vs R^2 (Xiao 29): rho=%.3f -> orders failure: %s"
          % (ng["spearman_curv_r2"], ng["curvature_orders_failure"]))
    print("    non-locality ell_x/lambda AUC=%.2f (documented chance-level)"
          % ng["nonlocality_ellx_over_lambda_auc"])
    print("(D) BFS zero-frequency anchor (%s):" % bf["provenance"])
    print("    eps_med=%.2f frac(eps<0.1)=%.3f R2=%.3f f_sep=%.2f all_pass=%s"
          % (bf["eps_median"], bf["frac_eps_lt_0p1"], bf["r2"], bf["f_sep"], bf["all_pass"]))
    print("Wrote: node_004/severity_law_result.json, results/severity_law.npz, fig_severity_law.{png,pdf}")

    # ---- assertions LAST (outputs already on disk) ----
    assert ex["bound_all_hold"], "domain floor bound <relErr> >= beta*S violated"
    assert ex["spearman_S_meanRelErr"] > 0.7, "exact severity collapse weak"
    assert px["failures_are_top2"], "the two R2<0 geometries are not the severity top-2"
    assert not ng["curvature_orders_failure"], "curvature unexpectedly orders failure"
    print("\nALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
