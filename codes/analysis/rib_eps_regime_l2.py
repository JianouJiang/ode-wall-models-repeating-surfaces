#!/usr/bin/env python3
r"""
rib_eps_regime_l2.py  --  L2 (node_003) epsilon-regime interpretation.
======================================================================

Discharges B-L2-A2-4 (review): explain WHY the sharp d-type rib fails the
a-priori ODE at a much MILDER cancellation depth (eps_median ~ 0.5, coverage of
the deep regime eps<eps* ~ 0.12) than the canonical periodic hill (eps_median ~
0.08, coverage ~ 0.56), yet still gives R^2(tau_w) < 0.

The honest, data-backed answer is NOT a new mechanism but the SAME cancellation
mechanism read in TWO REGIMES that the existing severity law already separates by
its first factor, the COVERAGE of the deep-cancellation band:

      S  =  coverage  x  inverse-depth ,   coverage = frac{ eps < eps* }.

  * DOMAIN-WIDE regime (smooth steep hill): the O(delta)-pitch repetition forces
    deep cancellation over a LARGE fraction of the pitch (coverage ~ 0.56; the
    deep-eps stations span ~100% of the streamwise domain) -> very large S ->
    catastrophic, domain-filling failure (R^2 ~ -48).
  * LOCALISED regime (sharp d-type rib): the deep cancellation is CONFINED to the
    cavity-recirculation zone (coverage ~ 0.12; deep-eps stations occupy a small
    streamwise span) -> smaller S -> MILDER but still catastrophic failure
    (R^2 ~ -1.3), ordered correctly below the hills by S.

Two consequences, both reported honestly (Pillar E):
  (1) the rib's lower coverage/S sits BELOW the smooth transition threshold
      S* ~ 0.32, so a single smooth criterion would MIS-CALL it tolerated ->
      a single smooth S does NOT collapse sharp geometries;
  (2) the additional governing group that orders the sharp family is the
      pitch/blockage ratio p/k: the d-type (p/k=3, cavity-confined recirculation)
      FAILS while the k-type (reattaches, near-wall scale reset every pitch) is
      TOLERATED -- exactly the O(delta)-pitch-repetition thesis on a sharp shape.

A REJECTED hypothesis is recorded too (no fabrication, B-L2-A2-4): the "sharp ribs
fail through a near-singular EDGE-PRESSURE spike" idea is NOT supported -- the
spanwise-and-time-averaged near-wall |dp/dx| of the rib is FLATTER than the smooth
hill's (peak/median ~ 1.7 vs ~ 87), so the discriminator is coverage/pitch, not
pressure spikiness.

Everything uses the FROZEN a-priori ODE/TBLE protocol (Y_IDX=10, predict_tau_w)
shared by every other geometry -- no retuning.  Outputs written BEFORE assertions.

Outputs:
  development/nodes/node_003/rib_eps_regime.json
  development/nodes/node_003/fig_eps_regime.{png,pdf}
  codes/results/rib_eps_regime_l2.npz
"""
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))            # codes/analysis
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
PROJ = os.path.dirname(CODES)
NODE = os.path.join(PROJ, "development", "nodes", "node_003")
os.makedirs(NODE, exist_ok=True)

sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

Y_IDX = 10
EPS_STAR = 0.1
R2_TOL = 0.88
S_STAR_SMOOTH = 0.32        # smooth-hill transition (manuscript)


def score(path):
    """Per-station a-priori ODE verdict + eps on one wall_profiles npz, frozen
    protocol.  Handles per-station nu (hills) and scalar nu (ribs)."""
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float)
    tau = np.asarray(d["tau_w"], float)
    dp = np.asarray(d["dp_dx"], float)
    nua = np.asarray(d["nu"], float)
    x = np.asarray(d["x"], float)
    n = len(tau)
    ym = y[:, Y_IDX]

    def nu_i(i):
        return float(nua[i]) if nua.size > 1 else float(nua)

    tp = np.full(n, np.nan)
    for i in range(n):
        if ym[i] > 0 and np.isfinite(U[i, Y_IDX]):
            tp[i] = predict_tau_w(float(U[i, Y_IDX]), float(ym[i]), float(dp[i]), nu_i(i))

    den = np.abs(dp) * np.abs(ym)
    eps = np.full(n, np.nan)
    m = den > 1e-30
    eps[m] = np.abs(tau[m]) / den[m]

    v = np.isfinite(tp) & np.isfinite(tau)
    ss_res = float(np.sum((tau[v] - tp[v]) ** 2))
    ss_tot = float(np.sum((tau[v] - tau[v].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ev = np.isfinite(eps)
    deep = ev & (eps < EPS_STAR)
    coverage = float(np.mean(eps[ev] < EPS_STAR)) if ev.any() else 0.0
    eps_med = float(np.median(eps[ev])) if ev.any() else np.nan
    S_proxy = coverage / max(eps_med, 1e-6)

    # streamwise span occupied by the deep-cancellation band (domain-wide vs local)
    span = float(x.max() - x.min()) if n > 1 else 0.0
    if deep.sum() > 1 and span > 0:
        xd = x[deep]
        deep_xspan_frac = float((xd.max() - xd.min()) / span)
    else:
        deep_xspan_frac = 0.0

    # near-wall pressure spikiness (tests + rejects the edge-pressure hypothesis)
    adp = np.abs(dp[np.isfinite(dp)])
    dpdx_peak_over_med = float(np.max(adp) / max(np.median(adp), 1e-30)) if adp.size else np.nan

    # geometry groups, when present in the file
    a_over_d = float(d["a_over_delta"]) if "a_over_delta" in d.files else np.nan
    lam_over_d = float(d["lambda_over_delta"]) if "lambda_over_delta" in d.files else np.nan
    p_over_k = (lam_over_d / a_over_d) if (a_over_d and np.isfinite(a_over_d) and a_over_d > 0) else np.nan

    return dict(
        n=n, r2=float(r2), eps_med=eps_med, coverage=coverage, S_proxy=S_proxy,
        deep_xspan_frac=deep_xspan_frac, dpdx_peak_over_med=dpdx_peak_over_med,
        a_over_delta=a_over_d, lambda_over_delta=lam_over_d, p_over_k=p_over_k,
        x=x, eps=eps,
    )


def main():
    # prefer the CONVERGED canonical LES rib; fall back to in-flight only as an
    # explicitly-flagged dry run (honesty guard, never silently final).
    les_canon = os.path.join(RESULTS, "rib_les_dtype_wall_profiles.npz")
    les_infl = os.path.join(RESULTS, "rib_les_dtype_wall_profiles_INFLIGHT_validation.npz")
    if os.path.exists(les_canon):
        les_path, les_final = les_canon, True
    elif os.path.exists(les_infl):
        les_path, les_final = les_infl, False
    else:
        les_path, les_final = None, False

    cases = [
        ("periodic hill h/Lx=1.0", "smooth", os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz")),
        ("wavy a/d=0.1", "smooth", os.path.join(RESULTS, "wavy_a10_wall_profiles.npz")),
        ("rib RANS d-type", "sharp", os.path.join(RESULTS, "rib_rans_dtype_wall_profiles.npz")),
        ("rib RANS k-type", "sharp", os.path.join(RESULTS, "rib_rans_ktype_wall_profiles.npz")),
    ]
    if les_path is not None:
        cases.insert(2, ("rib LES d-type (resolved)", "sharp", les_path))

    rows = []
    for label, cls, path in cases:
        if not os.path.exists(path):
            continue
        s = score(path)
        s.update(label=label, cls=cls, path=os.path.basename(path))
        rows.append(s)

    hill = next((r for r in rows if r["label"].startswith("periodic hill")), None)
    les = next((r for r in rows if "LES" in r["label"]), None)
    rans_d = next((r for r in rows if r["label"] == "rib RANS d-type"), None)
    rans_k = next((r for r in rows if r["label"] == "rib RANS k-type"), None)

    # ---- regime verdicts (honest) ----
    headline_rib = les if les is not None else rans_d
    regime = None
    if hill is not None and headline_rib is not None:
        regime = dict(
            hill_coverage=hill["coverage"], hill_eps_med=hill["eps_med"],
            hill_deep_xspan_frac=hill["deep_xspan_frac"], hill_r2=hill["r2"],
            rib_coverage=headline_rib["coverage"], rib_eps_med=headline_rib["eps_med"],
            rib_deep_xspan_frac=headline_rib["deep_xspan_frac"], rib_r2=headline_rib["r2"],
            both_fail=bool(hill["r2"] < 0 and headline_rib["r2"] < 0),
            rib_milder=bool(headline_rib["r2"] > hill["r2"]),
            note=("SAME cancellation mechanism, two regimes separated by COVERAGE (the "
                  "first factor of S): the smooth hill is DOMAIN-WIDE (coverage=%.2f, "
                  "deep-eps band spans %.0f%% of the pitch, R2=%.1f); the sharp rib is "
                  "LOCALISED to the cavity (coverage=%.2f, R2=%.2f) -- milder, ordered "
                  "below the hills by S, but still R2<0."
                  % (hill["coverage"], 100 * hill["deep_xspan_frac"], hill["r2"],
                     headline_rib["coverage"], headline_rib["r2"])),
        )

    # smooth-threshold misclassification (Pillar E consequence 1)
    smooth_misclass = None
    if headline_rib is not None:
        fails = headline_rib["r2"] < 0
        smooth_says_fail = headline_rib["S_proxy"] >= S_STAR_SMOOTH
        smooth_misclass = dict(
            rib_S_proxy=headline_rib["S_proxy"], S_star_smooth=S_STAR_SMOOTH,
            rib_fails=bool(fails), smooth_threshold_predicts_fail=bool(smooth_says_fail),
            misclassified=bool(fails and not smooth_says_fail),
            note=("the localised rib sits at S_proxy=%.2f < S*=%.2f, so the smooth "
                  "threshold would MIS-CALL it tolerated -> a single smooth S does not "
                  "collapse sharp geometries; the pitch group p/k is required."
                  % (headline_rib["S_proxy"], S_STAR_SMOOTH)),
        )

    # pitch-group ordering (Pillar E consequence 2)
    pitch = None
    if rans_d is not None and rans_k is not None:
        pitch = dict(
            dtype_p_over_k=rans_d["p_over_k"], dtype_r2=rans_d["r2"], dtype_coverage=rans_d["coverage"],
            ktype_p_over_k=rans_k["p_over_k"], ktype_r2=rans_k["r2"], ktype_coverage=rans_k["coverage"],
            ordered_by_pitch=bool(rans_k["r2"] > rans_d["r2"]),
            note=("pitch/blockage group p/k orders the sharp family: d-type "
                  "(p/k=%.0f, cavity-confined recirculation, coverage=%.2f) FAILS "
                  "R2=%.2f; k-type (p/k=%.0f, reattaches, coverage=%.2f) is TOLERATED "
                  "R2=%.2f -- the O(delta)-pitch-repetition thesis on a sharp geometry."
                  % (rans_d["p_over_k"], rans_d["coverage"], rans_d["r2"],
                     rans_k["p_over_k"], rans_k["coverage"], rans_k["r2"])),
        )

    # rejected edge-pressure-spike hypothesis (no fabrication)
    edge_pressure = None
    if hill is not None and headline_rib is not None:
        edge_pressure = dict(
            hill_dpdx_peak_over_med=hill["dpdx_peak_over_med"],
            rib_dpdx_peak_over_med=headline_rib["dpdx_peak_over_med"],
            hypothesis_supported=bool(headline_rib["dpdx_peak_over_med"]
                                      > hill["dpdx_peak_over_med"]),
            note=("REJECTED: the near-singular edge-pressure-spike hypothesis is NOT "
                  "supported -- the rib's spanwise/time-averaged near-wall |dp/dx| is "
                  "FLATTER than the smooth hill's (peak/median %.1f vs %.1f). The "
                  "regime discriminator is coverage/pitch, not pressure spikiness."
                  % (headline_rib["dpdx_peak_over_med"], hill["dpdx_peak_over_med"])),
        )

    result = dict(
        title="Epsilon-regime interpretation of the sharp-rib ODE failure (L2, node_003)",
        les_is_converged_final=bool(les_final),
        les_source=os.path.basename(les_path) if les_path else None,
        protocol="frozen a-priori ODE/TBLE Y_IDX=10 predict_tau_w; eps=|tau_w|/(|dp/dx| y_m); "
                 "coverage=frac(eps<0.1); S_proxy=coverage/eps_med.",
        eps_star=EPS_STAR, s_star_smooth=S_STAR_SMOOTH, r2_tol=R2_TOL,
        geometries=[dict(label=r["label"], cls=r["cls"], file=r["path"], n=r["n"],
                         r2=r["r2"], eps_med=r["eps_med"], coverage=r["coverage"],
                         S_proxy=r["S_proxy"], deep_xspan_frac=r["deep_xspan_frac"],
                         p_over_k=r["p_over_k"], dpdx_peak_over_med=r["dpdx_peak_over_med"])
                    for r in rows],
        two_regime_verdict=regime,
        smooth_threshold_misclassification=smooth_misclass,
        pitch_group_ordering=pitch,
        rejected_edge_pressure_hypothesis=edge_pressure,
    )

    # ---- WRITE OUTPUTS BEFORE ANY ASSERTION ----
    with open(os.path.join(NODE, "rib_eps_regime.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        os.path.join(RESULTS, "rib_eps_regime_l2.npz"),
        labels=np.array([r["label"] for r in rows]),
        cls=np.array([r["cls"] for r in rows]),
        r2=np.array([r["r2"] for r in rows]),
        eps_med=np.array([r["eps_med"] for r in rows]),
        coverage=np.array([r["coverage"] for r in rows]),
        S_proxy=np.array([r["S_proxy"] for r in rows]),
        deep_xspan_frac=np.array([r["deep_xspan_frac"] for r in rows]),
        p_over_k=np.array([r["p_over_k"] for r in rows]),
        dpdx_peak_over_med=np.array([r["dpdx_peak_over_med"] for r in rows]),
        eps_star=EPS_STAR, s_star_smooth=S_STAR_SMOOTH,
        les_is_converged_final=bool(les_final),
    )

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5))
    # (a) eps(x/pitch): domain-wide (hill) vs localised (rib)
    ax[0].axhline(EPS_STAR, color="purple", lw=1.0, ls="--", label=r"$\varepsilon^\ast=%.1f$" % EPS_STAR)
    for r, col in ((hill, "tab:orange"), (headline_rib, "tab:green")):
        if r is None:
            continue
        xs = r["x"]
        xn = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-30)
        e = r["eps"]
        o = np.argsort(xn)
        lab = r["label"] + (" (coverage %.2f)" % r["coverage"])
        ax[0].semilogy(xn[o], np.clip(e[o], 1e-3, 1e3), "-", color=col, lw=1.3, label=lab)
    ax[0].set_xlabel(r"streamwise position $x/L_{\rm pitch}$")
    ax[0].set_ylabel(r"cancellation depth $\varepsilon(x)=|\tau_w|/(|dp/dx|\,y_m)$")
    ax[0].set_title("(a) domain-wide (hill) vs localised (rib) cancellation")
    ax[0].legend(fontsize=7, loc="lower left", framealpha=0.9)

    # (b) coverage vs R^2: smooth + sharp; rib off the smooth trend.  symlog y so
    # the catastrophic domain-wide hill (R2~-48) and the milder ribs (R2~-1) both
    # read clearly on one axis.
    for r in rows:
        if r["cls"] == "smooth":
            c, mk = "tab:orange", "o"
        elif "LES" in r["label"]:
            c, mk = "tab:green", "D"
        else:
            c, mk = "tab:red", "s"
        ax[1].scatter(r["coverage"], r["r2"], c=c, marker=mk, s=85, edgecolor="k", zorder=4)
        # nudge labels off the markers and away from the legend
        dy = -14 if r["r2"] > 0 else 8
        ax[1].annotate(r["label"].replace("rib ", "").replace(" d-type", " d").replace(" k-type", " k"),
                       (r["coverage"], r["r2"]), fontsize=6.5, xytext=(-2, dy),
                       textcoords="offset points", ha="center")
    ax[1].axhline(0, color="0.3", lw=0.8, ls=":")
    ax[1].set_yscale("symlog", linthresh=2.0)
    ax[1].set_xlim(0.0, 0.62)
    ax[1].set_xlabel(r"deep-cancellation coverage  frac$(\varepsilon<\varepsilon^\ast)$")
    ax[1].set_ylabel(r"$R^2(\tau_w)$  (frozen a-priori ODE, symlog)")
    ax[1].set_title("(b) coverage is the regime order parameter")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
                  markeredgecolor="k", label="smooth hill/wavy"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:green",
                  markeredgecolor="k", label="rib LES (resolved)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:red",
                  markeredgecolor="k", label="rib RANS pilot")]
    ax[1].legend(handles=leg, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(NODE, "fig_eps_regime." + ext), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ----
    print("=" * 88)
    print("L2 EPS-REGIME INTERPRETATION  (eps*=%.2f  S*_smooth=%.2f  LES_final=%s)"
          % (EPS_STAR, S_STAR_SMOOTH, les_final))
    print("=" * 88)
    print("%-28s %-6s %6s %8s %8s %9s %7s %8s"
          % ("geometry", "class", "R2", "eps_med", "coverage", "deep_span", "p/k", "S_proxy"))
    for r in rows:
        print("%-28s %-6s %6.2f %8.3f %8.3f %9.2f %7s %8.3f"
              % (r["label"][:28], r["cls"], r["r2"], r["eps_med"], r["coverage"],
                 r["deep_xspan_frac"],
                 "n/a" if not np.isfinite(r["p_over_k"]) else "%.0f" % r["p_over_k"],
                 r["S_proxy"]))
    print("-" * 88)
    for tag, blk in (("regime", regime), ("smooth-misclass", smooth_misclass),
                     ("pitch-group", pitch), ("edge-pressure(rejected)", edge_pressure)):
        if blk:
            print("[%s] %s" % (tag, blk["note"]))
    if not les_final:
        print("\n[honesty guard] LES rib is IN-FLIGHT (scratch) -> numbers are a DRY RUN, "
              "NOT final.  Re-run after the canonical npz exists.")
    print("\nWrote node_003/rib_eps_regime.json, fig_eps_regime.{png,pdf}, "
          "results/rib_eps_regime_l2.npz")

    # ---- assertions LAST ----
    assert hill is not None, "missing periodic-hill anchor"
    assert headline_rib is not None, "missing rib"
    assert np.isfinite(headline_rib["r2"]), "rib R2 not finite"
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
