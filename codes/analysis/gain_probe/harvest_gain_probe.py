#!/usr/bin/env python3
"""Harvest the coupling-gain probe: closed-loop error transmission in WMLES.

Each probe run continues a deposited, fully developed coupled periodic-hill
case from the same t = 405 checkpoint.  In the perturbed arm the wall model's
*delivered* traction is multiplied by a constant gain G at fixed matching
velocity; the traction the model *requests* is unchanged.  The control arm has
G = 1.  Both arms are otherwise the identical simulation.

Write F(t) for the streamwise friction force on the hill wall (patch area
integral of the wall shear stress, per unit density).  Then

    T(t) = [F_G(t)/F_1(t) - 1] / (G - 1)

is the fraction of the open-loop perturbation that survives in the coupled
solution.  T = 1 means the coupling transmits the wall model's error intact;
T < 1 means the near-wall flow adjusts and absorbs part of it.  The window mean
T_inf and the loop gain Lambda = 1/T_inf - 1 are the reported quantities.

At the first sample after the gain is switched on the flow has not yet moved,
so T must equal 1 there; that is a check on the instrument, not a result.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "codes" / "results" / "rswm_gain_probe"
T0 = 405.0
# Moving-block bootstrap block length: half a streamwise period L_x/2 = 4.5
# convective time units at the crest bulk velocity.
BLOCK = 4.5
DRAWS = 4000
# Plateau window, in convective time units after the gain is switched on.
PLATEAU = (3.0, 8.0)
FIT_END = 12.0
# Lags after the gain onset at which T is read, and the averaging half-width.
LAGS = (1.0, 2.0, 5.0, 10.0)
LAG_WIDTH = 1.0
# Scatter of T inside a sliding one-unit window above which the paired arms
# are treated as decorrelated.
DECORR_SD = 0.05

VEC = re.compile(r"\(([^)]*)\)")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_surface_field_value(path):
    """Return (time, vector) from an OpenFOAM surfaceFieldValue .dat file."""
    times, vals = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = VEC.search(line)
        if not m:
            continue
        t = float(line.split()[0])
        v = [float(x) for x in m.group(1).split()]
        times.append(t)
        vals.append(v)
    if not times:
        raise RuntimeError("no samples in %s" % path)
    return np.asarray(times), np.asarray(vals)


def probe_series(probe_id, patch="wallForceBottom"):
    base = RESULT_ROOT / probe_id / "postProcessing" / patch
    if not base.exists():
        return None
    files = sorted(base.rglob("surfaceFieldValue.dat"))
    if not files:
        return None
    ts, vs = [], []
    for f in files:
        t, v = read_surface_field_value(f)
        ts.append(t)
        vs.append(v)
    t = np.concatenate(ts)
    v = np.concatenate(vs, axis=0)
    order = np.argsort(t)
    t, v = t[order], v[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], v[keep]


def moving_block_bootstrap(t, y, block, draws, rng):
    """95% interval for the mean of an autocorrelated series."""
    if y.size < 4:
        return float("nan"), float("nan")
    dt = np.median(np.diff(t))
    n_block = max(2, int(round(block / dt)))
    n_block = min(n_block, max(2, y.size // 3))
    starts_max = y.size - n_block
    if starts_max <= 0:
        return float("nan"), float("nan")
    n_draw_blocks = int(np.ceil(y.size / n_block))
    means = np.empty(draws)
    for b in range(draws):
        st = rng.integers(0, starts_max + 1, size=n_draw_blocks)
        idx = (st[:, None] + np.arange(n_block)[None, :]).ravel()[:y.size]
        means[b] = y[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exp_relaxation_fit(t, T, t0):
    """Least-squares fit T(t) = Tinf + (1-Tinf) exp(-(t-t0)/tau)."""
    best = (np.inf, np.nan, np.nan)
    for tau in np.geomspace(0.02, 40.0, 240):
        basis = np.exp(-(t - t0) / tau)
        # T = Tinf(1-basis) + basis  ->  (T - basis) = Tinf (1 - basis)
        num = np.dot(1.0 - basis, T - basis)
        den = np.dot(1.0 - basis, 1.0 - basis)
        if den <= 0:
            continue
        Tinf = num / den
        resid = float(np.sum((Tinf * (1.0 - basis) + basis - T) ** 2))
        if resid < best[0]:
            best = (resid, Tinf, tau)
    return best[1], best[2], best[0]


def analyse_pair(control, perturbed, gain, window, rng, patch="wallForceBottom"):
    c = probe_series(control, patch)
    p = probe_series(perturbed, patch)
    if c is None or p is None:
        return None
    tc, vc = c
    tp, vp = p
    Fc_raw, Fp_raw = vc[:, 0], vp[:, 0]

    lo, hi = T0, min(tc[-1], tp[-1])
    if hi - lo < 0.5 * window:
        return dict(status="short",
                    reached_control=float(tc[-1]), reached_perturbed=float(tp[-1]))

    grid = tc[(tc >= lo) & (tc <= hi)]
    Fc = np.interp(grid, tc, Fc_raw)
    Fp = np.interp(grid, tp, Fp_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = (Fp / Fc - 1.0) / (gain - 1.0)

    avg_start = T0 + 0.25 * window
    win = grid >= avg_start
    if win.sum() < 8:
        win = grid >= (lo + 0.5 * (hi - lo))

    mean_c = float(Fc[win].mean())
    mean_p = float(Fp[win].mean())
    T_late = (mean_p / mean_c - 1.0) / (gain - 1.0)
    ci_lo, ci_hi = moving_block_bootstrap(grid[win], T[win], BLOCK, DRAWS, rng)
    Tfit, tau_c, resid = exp_relaxation_fit(grid[grid <= T0 + FIT_END],
                                            T[grid <= T0 + FIT_END], T0)

    # Paired estimate.  The two arms start from the identical state and run the
    # identical discrete trajectory apart from the gain, so their difference is
    # deterministic until chaotic divergence sets in.  Rather than assume a
    # plateau, T is reported at a ladder of fixed lags after the gain is
    # switched on, together with the lag at which the pairing is lost.
    lag = grid - T0
    T_at_lag, T_sd_at_lag = {}, {}
    for centre in LAGS:
        m = (lag >= centre - 0.5 * LAG_WIDTH) & (lag <= centre + 0.5 * LAG_WIDTH)
        T_at_lag[centre] = float(T[m].mean()) if m.sum() >= 3 else float("nan")
        T_sd_at_lag[centre] = float(T[m].std()) if m.sum() >= 3 else float("nan")

    # Pairing is lost when the scatter of T inside a sliding one-unit window
    # first exceeds DECORR_SD; before that the difference between the arms is
    # deterministic to better than that tolerance.
    # The scatter is measured about a straight line inside each window, so a
    # fast but smooth relaxation is not mistaken for decorrelation.
    decorr = float("nan")
    for e in np.arange(0.0, lag.max(), 1.0):
        m = (lag >= e) & (lag < e + 1.0)
        if m.sum() < 4:
            continue
        coef = np.polyfit(lag[m], T[m], 1)
        window_resid = T[m] - np.polyval(coef, lag[m])
        if window_resid.std() > DECORR_SD:
            decorr = float(e)
            break

    plat = (grid >= T0 + PLATEAU[0]) & (grid <= T0 + PLATEAU[1])
    T_plateau = float(T[plat].mean()) if plat.sum() >= 4 else float("nan")
    T_plateau_sd = float(T[plat].std()) if plat.sum() >= 4 else float("nan")
    late_sd = float(T[win].std()) if win.sum() >= 4 else float("nan")
    # Is the response still falling where it is read?  (a negative slope means
    # the reported value is an upper bound on the transmitted fraction)
    still_falling = bool(T_at_lag.get(LAGS[-1], np.nan)
                         < T_at_lag.get(LAGS[-2], np.nan) - 0.02)

    return dict(
        status="ok",
        control=control,
        perturbed=perturbed,
        gain=float(gain),
        patch=patch,
        n_samples=int(grid.size),
        t_start=float(grid[0]),
        t_end=float(grid[-1]),
        average_start=float(avg_start),
        n_average_samples=int(win.sum()),
        F_control_mean=mean_c,
        F_perturbed_mean=mean_p,
        T_inf=float(T_plateau),
        T_at_lag={str(k): v for k, v in T_at_lag.items()},
        # The non-negativity floor of the scalar eddy viscosity can stop a
        # G < 1 request from being delivered in full.  T(t_0) measures the
        # gain the boundary condition actually applied, so dividing by it
        # normalises every arm to its own delivered perturbation.
        T_eff_at_lag={str(k): (v / float(T[0]) if T[0] != 0 else float("nan"))
                      for k, v in T_at_lag.items()},
        delivered_gain=float(1.0 + (gain - 1.0) * T[0]),
        T_sd_at_lag={str(k): v for k, v in T_sd_at_lag.items()},
        decorrelation_lag=decorr,
        still_falling_at_last_lag=still_falling,
        T_plateau=float(T_plateau),
        T_plateau_sd=T_plateau_sd,
        T_plateau_window=[T0 + PLATEAU[0], T0 + PLATEAU[1]],
        n_plateau_samples=int(plat.sum()),
        T_late=float(T_late),
        T_late_sd=late_sd,
        T_ci_lo=ci_lo,
        T_ci_hi=ci_hi,
        T_window_mean_of_ratio=float(np.mean(T[win])),
        # grid[0] is the checkpoint itself: both arms hold the identical state
        # there, so T must be 0.  The first sample after it is the one at which
        # the flow has barely moved and T must be ~1.
        T_at_checkpoint=float(T[0]),
        T_first_after_onset=float(T[1]) if T.size > 1 else float("nan"),
        t_first_after_onset=float(grid[1]) if grid.size > 1 else float("nan"),
        loop_gain=float(1.0 / T_plateau - 1.0) if T_plateau not in (0.0,) else float("inf"),
        loop_gain_late=float(1.0 / T_late - 1.0) if T_late not in (0.0,) else float("inf"),
        T_fit=float(Tfit),
        relaxation_time=float(tau_c),
        fit_residual=float(resid),
        series_time=grid,
        series_T=T,
        series_F_control=Fc,
        series_F_perturbed=Fp,
    )


def fidelity_check(deployed_id, control_id, patch="wallForceBottom"):
    """Does the gain = 1 derived boundary condition reproduce the deployed one?

    The two runs are compared only at time values they both sampled, so a
    mismatch cannot be an artefact of the adjustable time step.
    """
    a = probe_series(deployed_id, patch)
    b = probe_series(control_id, patch)
    if a is None or b is None:
        return dict(status="missing")
    ta, va = a
    tb, vb = b
    common, ia, ib = np.intersect1d(np.round(ta, 9), np.round(tb, 9),
                                    return_indices=True)
    if common.size < 4:
        return dict(status="no_common_times", n_deployed=int(ta.size),
                    n_control=int(tb.size),
                    t_deployed_last=float(ta[-1]), t_control_last=float(tb[-1]))
    fa, fb = va[ia, 0], vb[ib, 0]
    rel = np.abs(fa - fb) / np.maximum(np.abs(fb), 1e-30)
    return dict(status="ok", n_compared=int(common.size),
                t_first=float(common[0]), t_last=float(common[-1]),
                max_relative_difference=float(np.max(rel)),
                median_relative_difference=float(np.median(rel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="jobs/rswm_gain_probe_matrix.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    matrix = json.loads((ROOT / args.matrix).read_text())
    runs = {r["probe_id"]: r for r in matrix["runs"]}
    rng = np.random.default_rng(20260825)

    present = {p.name for p in RESULT_ROOT.iterdir()} if RESULT_ROOT.exists() else set()
    pairs, arrays = [], {}

    for run in matrix["runs"]:
        if run["role"] != "perturbed":
            continue
        tag, arch = run["probe_id"].split("_")[2], run["probe_id"].split("_")[3]
        control = "rswm_gain_%s_%s_g100" % (tag, arch)
        if run["probe_id"] not in present or control not in present:
            pairs.append(dict(status="missing", perturbed=run["probe_id"],
                              control=control, gain=run["gain"],
                              ym_over_H=run["ym"], architecture=arch))
            continue
        for patch in ("wallForceBottom", "wallForceTop"):
            res = analyse_pair(control, run["probe_id"], run["gain"],
                               run["window"], rng, patch)
            if res is None:
                pairs.append(dict(status="missing", perturbed=run["probe_id"],
                                  control=control, patch=patch,
                                  ym_over_H=run["ym"], architecture=arch))
                continue
            res["ym_over_H"] = run["ym"]
            res["architecture"] = arch
            key = "%s__%s" % (run["probe_id"], patch)
            for f in ("series_time", "series_T", "series_F_control",
                      "series_F_perturbed"):
                if f in res:
                    arrays["%s__%s" % (key, f)] = res.pop(f)
            pairs.append(res)

    fidelity = {}
    for arch in ("tble", "eq"):
        dep = "rswm_gain_ym0935_%s_deployed_v2" % arch
        if dep not in present:
            dep = "rswm_gain_ym0935_%s_deployed" % arch
        ctl = "rswm_gain_ym0935_%s_g100" % arch
        if dep in present and ctl in present:
            fidelity[arch] = fidelity_check(dep, ctl)
        else:
            fidelity[arch] = dict(status="missing")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d")
    out = pathlib.Path(args.out) if args.out else \
        ROOT / ("codes/results/gain_probe_transmission_%s.npz" % stamp)
    np.savez_compressed(out, **arrays)

    ok = [p for p in pairs if p.get("status") == "ok"
          and p.get("patch") == "wallForceBottom"]
    summary = dict(
        generated=datetime.datetime.utcnow().isoformat() + "Z",
        matrix_sha256=sha256(ROOT / args.matrix),
        driver_sha256=matrix["driver_sha256"],
        gain_start_time=T0,
        block_length=BLOCK,
        bootstrap_draws=DRAWS,
        definition="T = [F_G/F_1 - 1]/(G-1) on the patch-integrated streamwise "
                   "wall shear force; Lambda = 1/T - 1.  T_plateau is the mean "
                   "over the paired-plateau window; T_late is the registered "
                   "late-window mean with a moving-block interval.",
        plateau_window_after_onset=list(PLATEAU),
        lags=list(LAGS),
        lag_width=LAG_WIDTH,
        decorrelation_sd_threshold=DECORR_SD,
        n_pairs_complete=len(ok),
        n_pairs_expected=sum(1 for r in matrix["runs"] if r["role"] == "perturbed"),
        fidelity=fidelity,
        pairs=pairs,
    )
    (out.with_name(out.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float))

    print("complete pairs (hill wall): %d of %d"
          % (len(ok), summary["n_pairs_expected"]))
    for p in sorted(ok, key=lambda r: (r["architecture"], r["ym_over_H"], r["gain"])):
        L = p["T_at_lag"]
        print("  %-4s ym/H=%.4f G=%.2f | T(0)=%.4f T1=%.3f T2=%.3f T5=%.3f"
              " T10=%.3f | decorr@%.0f | late %.3f [%.3f,%.3f] | falling=%d"
              % (p["architecture"], p["ym_over_H"], p["gain"],
                 p["T_at_checkpoint"], L["1.0"], L["2.0"], L["5.0"], L["10.0"],
                 p["decorrelation_lag"], p["T_late"], p["T_ci_lo"],
                 p["T_ci_hi"], p["still_falling_at_last_lag"]))
    for arch, f in fidelity.items():
        print("  fidelity %-4s %s" % (arch, f))
    print("wrote %s (+ summary)" % out)


if __name__ == "__main__":
    main()
