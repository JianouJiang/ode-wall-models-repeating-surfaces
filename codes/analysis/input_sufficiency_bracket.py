#!/usr/bin/env python3
r"""Is an ODE wall model's failure information-limited or functional-form-limited?

A wall model of the standard local class reads, at one station, the matching
velocity $u_m$, the matching distance $y_m$, the kinematic viscosity $\nu$ and
the streamwise pressure gradient $\mathrm{d}p/\mathrm{d}x$, and returns the wall
traction $\tau_w$.  Any such model that is invariant under the similarity group
of the incompressible wall layer is a function of two dimensionless groups,

    a = u_m y_m / nu ,   b = (dp/dx) y_m^3 / nu^2   -->   t = tau_w y_m^2 / nu^2,

so the whole class -- equilibrium laws, pressure-gradient boundary-layer
equations, any closure inside them, any number of retained terms -- is a set of
functions on the SAME two-dimensional input space.  This program brackets what
that class can do on a given surface, from below and from above.

LOWER BOUND (certified, model-free).  For any model whose response is
L-Lipschitz in the standardised inputs and any two stations i, j,

    |e_i| + |e_j| >= |t_i - t_j| - L ||d_i - d_j|| ,

hence e_i^2 + e_j^2 >= g_ij^2/2 with g_ij the positive part of the right-hand
side.  Summing over any set of DISJOINT pairs certifies a lower bound on the
mean square error; maximising over matchings gives the tightest such bound
available from the retained edges.  Using a subset of stations or of edges only
weakens the bound, so every reported value is conservative.

UPPER BOUND (constructive, out-of-sample).  A geometry-blind empirical function
of exactly the same two inputs is fitted on other cases and evaluated on a case
it has never seen.  It is an instrument for measuring what the input set
determines, not a proposed wall model: it carries no closure, no equation and
no deployment path.

BASELINE (matched).  The two deployed models of the paper -- the Spalding
equilibrium relation (M0) and the pressure-gradient boundary-layer equation
solved by the shared shooting operator (M1) -- are evaluated at exactly the same
stations, from exactly the same inputs, with the constants the coupled cases
deployed.

If the empirical function of the same inputs scores where the deployed models do
not, the input set is sufficient and the failure lies in the functional form.
If it does not, the input set is insufficient and no model in the class can be
repaired without new inputs.  Both outcomes are reported as measured.

All evaluations are A PRIORI on mean reference fields: no coupled simulation is
run or claimed here (acceptance criterion G4), and no geometry outside those with real
reference data is evaluated (G1).

Outputs
-------
codes/results/input_sufficiency_bracket.npz
codes/results/input_sufficiency_bracket_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from models.source_faithful_wall_models import (  # noqa: E402
    shoot_wall_stress,
    spalding_wall_stress,
)

RESULTS = ROOT / "codes" / "results"
XIAO_DIR = (ROOT / "codes/new_data_download/geometry_driven/"
            "xiao_pehill_parameterized/pehill-29-cases-DNS")
WAVY_NPZ = RESULTS / "r1_sta2_wavy_wrles_20260824.npz"
CONVDIV_NPZ = ROOT / "codes/new_data_download/conv_div_channel_Re12600_wall_profiles.npz"
CACHE = RESULTS / "input_sufficiency_cache"

# ---------------------------------------------------------------- registered
ETA_FRACTIONS = (0.05, 0.10)          # matching surface, fraction of case delta
K_NEIGHBOURS = 15                     # empirical instrument, inverse-distance
LIPSCHITZ_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FLOOR_SUBSAMPLE = 1200                # stations retained for the matching bound
BOOTSTRAP_BLOCKS = 12                 # circular moving-block resampling
BOOTSTRAP_DRAWS = 2000
RNG_SEED = 20260825
NU_XIAO = 1.0 / 5600.0                # Re_b = 5600 (Xiao et al. family)
NU_WAVY = 1.0 / 3460.0                # U_b delta / Re_h, make_wavy_cherukat_wrles_case.py
WIDTH = 257
VEL_TOL = 1.0e-6
SPALDING_KAPPA, SPALDING_B = 0.41, 5.0
TBLE_KAPPA, TBLE_APLUS = 0.41, 26.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ loading
def _extract_columns(mean_path: Path, rms_path: Path) -> dict:
    """Hill-surface-aware extraction, protocol of build_corrected_xiao_profiles.py."""
    import pandas as pd

    mean = pd.read_csv(mean_path, sep=r"\s+", header=None).to_numpy()
    x, y, u, v, p = mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3], mean[:, 5]
    uv = (pd.read_csv(rms_path, sep=r"\s+", header=None).to_numpy()[:, 2]
          if rms_path.exists() else np.zeros_like(x))

    xu = np.unique(np.round(x, 6))
    n = len(xu)
    Y = np.full((n, WIDTH), np.nan)
    U = np.full((n, WIDTH), np.nan)
    V = np.full((n, WIDTH), np.nan)
    UV = np.full((n, WIDTH), np.nan)
    xs = np.zeros(n)
    tau_w = np.zeros(n)
    p_wall = np.full(n, np.nan)

    for s, xv in enumerate(xu):
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp, ww = y[m], u[m], v[m], p[m], uv[m]
        o = np.argsort(yy)
        yy, uu, vv, pp, ww = yy[o], uu[o], vv[o], pp[o], ww[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        xs[s] = xv
        if len(fluid) < 2:
            continue
        k = max(fluid[0], 1)
        ywall = yy[k - 1:] - yy[k - 1]
        ufl, vfl, pfl, wwfl = uu[k - 1:], vv[k - 1:], pp[k - 1:], ww[k - 1:]
        L = min(len(ywall), WIDTH)
        Y[s, :L], U[s, :L], V[s, :L], UV[s, :L] = (ywall[:L], ufl[:L],
                                                   vfl[:L], wwfl[:L])
        nfit = min(4, len(ywall) - 1)
        yf, uf = ywall[1:1 + nfit], ufl[1:1 + nfit]
        tau_w[s] = NU_XIAO * (float(np.sum(yf * uf) / np.sum(yf * yf))
                              if np.sum(yf * yf) > 0 else 0.0)
        p_wall[s] = pfl[1] if len(pfl) > 1 else pfl[0]

    return dict(x=xs, y=Y, U=U, V=V, uv=UV, tau_w=tau_w,
                dp_dx=np.gradient(p_wall, xs), p_wall=p_wall)


def load_hill_fields(case_dir: Path) -> dict:
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{case_dir.name}_full.npz"
    if not cached.exists():
        data = _extract_columns(case_dir / "mean_files.dat", case_dir / "rms_files2.dat")
        np.savez_compressed(cached, **data)
    d = np.load(cached, allow_pickle=True)
    return {k: d[k] for k in d.files}


def station_state(y, U, tau, dpdx, nu, frac):
    """Matching state at y_m = frac * delta, delta = half the tallest fluid column."""
    n = len(tau)
    tops = np.array([np.nanmax(y[i]) if np.any(np.isfinite(y[i])) else np.nan
                     for i in range(n)])
    delta = 0.5 * float(np.nanmax(tops))
    y_m = frac * delta
    u_m = np.full(n, np.nan)
    for i in range(n):
        good = np.isfinite(y[i]) & np.isfinite(U[i])
        if good.sum() >= 2 and np.nanmax(y[i][good]) >= y_m:
            u_m[i] = float(np.interp(y_m, y[i][good], U[i][good]))
    ok = np.isfinite(u_m) & np.isfinite(tau) & np.isfinite(dpdx)
    return dict(u_m=u_m[ok], y_m=np.full(int(ok.sum()), y_m), nu=nu,
                dp_dx=np.asarray(dpdx)[ok], tau=np.asarray(tau)[ok],
                delta=delta, n_station=int(ok.sum()))


def groups(case: dict) -> dict:
    a = case["u_m"] * case["y_m"] / case["nu"]
    b = case["dp_dx"] * case["y_m"] ** 3 / case["nu"] ** 2
    t = case["tau"] * case["y_m"] ** 2 / case["nu"] ** 2
    return dict(case, a=a, b=b, t=t)


def build_cases(frac: float) -> tuple[list[dict], dict, dict]:
    hills = []
    for case_dir in sorted(d for d in XIAO_DIR.iterdir() if d.is_dir()):
        if not (case_dir / "mean_files.dat").exists():
            continue
        f = load_hill_fields(case_dir)
        c = station_state(f["y"], f["U"], f["tau_w"], f["dp_dx"], NU_XIAO, frac)
        c.update(name=case_dir.name, family="periodic_hill",
                 group=case_dir.name.split("-")[0])
        hills.append(groups(c))

    w = np.load(WAVY_NPZ, allow_pickle=True)
    wavy = station_state(w["G2_ycell"], w["G2_U"], w["G2_tau_t"],
                         np.gradient(w["G2_p_wall"], w["G2_x"]), NU_WAVY, frac)
    wavy.update(name="wavy_wall_G2", family="wavy_wall", group="wavy")
    wavy = groups(wavy)

    cdv = np.load(CONVDIV_NPZ, allow_pickle=True)
    nu_cd = float(np.atleast_1d(cdv["nu"]).reshape(-1)[0])
    convdiv = station_state(cdv["y"], cdv["U"], cdv["tau_w"], cdv["dp_dx"], nu_cd, frac)
    convdiv.update(name="conv_div_Re12600", family="conv_div", group="convdiv")
    convdiv = groups(convdiv)
    return hills, wavy, convdiv


# --------------------------------------------------------- empirical instrument
def slog(v):
    return np.sign(v) * np.log10(1.0 + np.abs(v))


def features(case: dict, use_b: bool) -> np.ndarray:
    cols = [slog(case["a"])] + ([slog(case["b"])] if use_b else [])
    return np.column_stack(cols)


def knn_transfer(train: list[dict], test: dict, use_b: bool,
                 k: int = K_NEIGHBOURS, shuffle_rng=None):
    Xtr = np.vstack([features(c, use_b) for c in train])
    ytr = np.concatenate([slog(c["t"]) for c in train])
    if shuffle_rng is not None:                      # control case
        ytr = ytr[shuffle_rng.permutation(len(ytr))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-30
    A = (Xtr - mu) / sd
    B = (features(test, use_b) - mu) / sd
    kk = int(min(k, len(A) - 1))
    pred = np.empty(len(B))
    nn_dist = np.empty(len(B))
    for start in range(0, len(B), 256):
        chunk = B[start:start + 256]
        dd = np.sqrt(((chunk[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        idx = np.argpartition(dd, kk, axis=1)[:, :kk]
        rows = np.arange(len(chunk))[:, None]
        dsel = dd[rows, idx]
        w = 1.0 / (dsel + 1e-9)
        pred[start:start + len(chunk)] = (w * ytr[idx]).sum(1) / w.sum(1)
        nn_dist[start:start + len(chunk)] = dsel.min(1)
    t_hat = np.sign(pred) * (10.0 ** np.abs(pred) - 1.0)
    return t_hat * test["nu"] ** 2 / test["y_m"] ** 2, nn_dist, A


# ------------------------------------------------------------ deployed models
def deployed_predictions(case: dict, frac: float | None = None) -> dict:
    """M0/M1 at the case's own stations.  Cached: the shooting operator costs
    about 54 ms per station and the census runs to tens of thousands."""
    if frac is not None:
        CACHE.mkdir(exist_ok=True)
        store = CACHE / f"deployed_{case['name']}_eta{frac:.2f}.npz"
        if store.exists():
            z = np.load(store)
            if len(z["m0"]) == case["n_station"]:
                return dict(m0=z["m0"], m1=z["m1"],
                            n_model_failures=int(z["n_model_failures"]))
    out = _deployed_predictions_raw(case)
    if frac is not None:
        np.savez_compressed(store, **out)
    return out


def _deployed_predictions_raw(case: dict) -> dict:
    n = case["n_station"]
    m0 = np.full(n, np.nan)
    m1 = np.full(n, np.nan)
    nu, y_m = case["nu"], case["y_m"]
    n_fail = 0
    for i in range(n):
        try:
            m0[i] = spalding_wall_stress(float(case["u_m"][i]), float(y_m[i]), nu,
                                         kappa=SPALDING_KAPPA, b_const=SPALDING_B)
        except Exception:
            n_fail += 1
        try:
            g = float(case["dp_dx"][i])
            m1[i] = shoot_wall_stress(float(case["u_m"][i]), float(y_m[i]), nu,
                                      source=lambda yy, g=g: np.full_like(yy, g),
                                      a_plus=TBLE_APLUS).tau_w
        except Exception:
            n_fail += 1
    return dict(m0=m0, m1=m1, n_model_failures=n_fail)


# ------------------------------------------------------------------- metrics
def r2_score(pred, ref):
    good = np.isfinite(pred) & np.isfinite(ref)
    if good.sum() < 3:
        return float("nan")
    p, r = pred[good], ref[good]
    denom = np.sum((r - r.mean()) ** 2)
    return float("nan") if denom <= 0 else float(1.0 - np.sum((p - r) ** 2) / denom)


def rel_rms(pred, ref):
    good = np.isfinite(pred) & np.isfinite(ref)
    if good.sum() < 3:
        return float("nan")
    p, r = pred[good], ref[good]
    return float(np.sqrt(np.mean((p - r) ** 2)) / (np.sqrt(np.mean(r ** 2)) + 1e-300))


def block_bootstrap_r2(pred, ref, rng, n_blocks=BOOTSTRAP_BLOCKS,
                       draws=BOOTSTRAP_DRAWS):
    """Circular moving-block interval over streamwise stations."""
    good = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[good], ref[good]
    n = len(p)
    if n < 3 * n_blocks:
        return (float("nan"), float("nan"))
    length = n // n_blocks
    vals = np.empty(draws)
    for d in range(draws):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([(np.arange(s, s + length) % n) for s in starts])
        vals[d] = r2_score(p[idx], r[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ---------------------------------------------------- certified Lipschitz floor
def certified_floor(d_std: np.ndarray, t_norm: np.ndarray, L: float,
                    k_edges: int = 30):
    """Max-weight-matching lower bound on RMS error / RMS(tau) for L-Lipschitz F."""
    import networkx as nx

    n = len(t_norm)
    dist = np.sqrt(((d_std[:, None, :] - d_std[None, :, :]) ** 2).sum(-1))
    order = np.argsort(dist, axis=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in order[i, 1:k_edges + 1]:
            j = int(j)
            if j <= i:
                continue
            g = abs(t_norm[i] - t_norm[j]) - L * dist[i, j]
            if g > 0.0:
                G.add_edge(i, j, weight=0.5 * g * g)
    if G.number_of_edges() == 0:
        return 0.0, 0
    matching = nx.max_weight_matching(G, weight="weight")
    total = sum(G[u][v]["weight"] for u, v in matching)
    return float(np.sqrt(total / n)), len(matching)


def measured_lipschitz(d_std: np.ndarray, model_t: np.ndarray, k: int = 10) -> float:
    """Largest finite-difference slope of a deployed model over the observed data."""
    good = np.isfinite(model_t)
    d, m = d_std[good], model_t[good]
    if len(m) < k + 2:
        return float("nan")
    dist = np.sqrt(((d[:, None, :] - d[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    idx = np.argpartition(dist, k, axis=1)[:, :k]
    rows = np.arange(len(d))[:, None]
    slopes = np.abs(m[idx] - m[:, None]) / np.maximum(dist[rows, idx], 1e-12)
    return float(np.percentile(slopes, 95))


# ----------------------------------------------------------------------- main
def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(RNG_SEED)
    out: dict = dict(
        schema="input_sufficiency_bracket_v1",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        registered=dict(eta_fractions=list(ETA_FRACTIONS), k_neighbours=K_NEIGHBOURS,
                        lipschitz_grid=list(LIPSCHITZ_GRID),
                        floor_subsample=FLOOR_SUBSAMPLE, seed=RNG_SEED,
                        bootstrap=dict(blocks=BOOTSTRAP_BLOCKS, draws=BOOTSTRAP_DRAWS),
                        spalding=dict(kappa=SPALDING_KAPPA, b=SPALDING_B),
                        tble=dict(kappa=TBLE_KAPPA, a_plus=TBLE_APLUS),
                        nu=dict(xiao=NU_XIAO, wavy=NU_WAVY)),
        sources=dict(wavy=sha256(WAVY_NPZ), conv_div=sha256(CONVDIV_NPZ)),
        heights={},
    )
    arrays: dict = {}

    for frac in ETA_FRACTIONS:
        tag = f"eta{frac:.2f}"
        hills, wavy, convdiv = build_cases(frac)
        print(f"[{tag}] {len(hills)} hills, wavy {wavy['n_station']} st., "
              f"conv-div {convdiv['n_station']} st.", flush=True)
        rec: dict = dict(n_hill_cases=len(hills),
                         n_hill_stations=int(sum(c["n_station"] for c in hills)),
                         hill_delta=[float(c["delta"]) for c in hills])

        # --- matched deployed baselines, same stations, same inputs
        deployed = {}
        for c in hills + [wavy, convdiv]:
            dp = deployed_predictions(c, frac)
            deployed[c["name"]] = dp
            c["m0"], c["m1"] = dp["m0"], dp["m1"]
        rec["deployed"] = {
            name: dict(
                r2_equilibrium=r2_score(deployed[c["name"]]["m0"], c["tau"]),
                r2_tble=r2_score(deployed[c["name"]]["m1"], c["tau"]),
                relrms_equilibrium=rel_rms(deployed[c["name"]]["m0"], c["tau"]),
                relrms_tble=rel_rms(deployed[c["name"]]["m1"], c["tau"]),
                n_model_failures=deployed[c["name"]]["n_model_failures"])
            for name, c in ((c["name"], c) for c in hills + [wavy, convdiv])}
        print(f"[{tag}] deployed models evaluated "
              f"({time.time() - t_start:.0f}s)", flush=True)

        # --- constructive upper bound, two hold-out protocols, both input sets
        rec["transfer"] = {}
        for use_b in (False, True):
            key = "a_and_b" if use_b else "a_only"
            case_out, group_out, shuffled, extrap = [], [], [], []
            train_scale: dict[str, float] = {}
            for i, held in enumerate(hills):
                tr = [c for j, c in enumerate(hills) if j != i]
                pred, nn, A = knn_transfer(tr, held, use_b)
                case_out.append(r2_score(pred, held["tau"]))
                tr_g = [c for c in hills if c["group"] != held["group"]]
                pred_g, nn_g, A_g = knn_transfer(tr_g, held, use_b)
                group_out.append(r2_score(pred_g, held["tau"]))
                pred_s, _, _ = knn_transfer(tr_g, held, use_b, shuffle_rng=rng)
                shuffled.append(r2_score(pred_s, held["tau"]))
                # extrapolation: test nearest-neighbour distance against the
                # training set's own nearest-neighbour scale, estimated once per
                # hold-out group on a fixed 4000-point subsample.
                if held["group"] not in train_scale:
                    sub = A_g[np.linspace(0, len(A_g) - 1,
                                          min(4000, len(A_g))).astype(int)]
                    dtr = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
                    np.fill_diagonal(dtr, np.inf)
                    train_scale[held["group"]] = float(np.percentile(dtr.min(1), 95))
                extrap.append(float(np.mean(nn_g > train_scale[held["group"]])))
            case_out = np.array(case_out)
            group_out = np.array(group_out)
            shuffled = np.array(shuffled)
            cross = {}
            for tgt in (wavy, convdiv):
                p, _, _ = knn_transfer(hills, tgt, use_b)
                cross[f"hills_to_{tgt['family']}"] = r2_score(p, tgt["tau"])
            p, _, _ = knn_transfer([wavy, convdiv], hills[0], use_b)
            cross["nonhill_to_first_hill"] = r2_score(p, hills[0]["tau"])
            rec["transfer"][key] = dict(
                leave_one_case_out=dict(
                    per_case={c["name"]: float(v) for c, v in zip(hills, case_out)},
                    median=float(np.median(case_out)), min=float(case_out.min()),
                    max=float(case_out.max())),
                leave_one_group_out=dict(
                    per_case={c["name"]: float(v) for c, v in zip(hills, group_out)},
                    median=float(np.median(group_out)), min=float(group_out.min()),
                    max=float(group_out.max()),
                    n_above_half=int(np.sum(group_out > 0.5))),
                label_shuffled_control=dict(median=float(np.median(shuffled)),
                                            max=float(shuffled.max())),
                extrapolating_fraction_median=float(np.median(extrap)),
                cross_family=cross)
            print(f"[{tag}][{key}] group-out median R2 "
                  f"{np.median(group_out):.3f}; shuffled {np.median(shuffled):.3f}; "
                  f"cross {cross}", flush=True)

        # --- interval on the headline pair, canonical hill
        canon = next((c for c in hills if c["name"] == "alph10-6-3036"), hills[0])
        tr_g = [c for c in hills if c["group"] != canon["group"]]
        pred_c, _, _ = knn_transfer(tr_g, canon, True)
        rec["canonical"] = dict(
            case=canon["name"], n_station=canon["n_station"],
            r2_empirical=r2_score(pred_c, canon["tau"]),
            r2_empirical_ci=block_bootstrap_r2(pred_c, canon["tau"], rng),
            r2_equilibrium=r2_score(canon["m0"], canon["tau"]),
            r2_equilibrium_ci=block_bootstrap_r2(canon["m0"], canon["tau"], rng),
            r2_tble=r2_score(canon["m1"], canon["tau"]),
            r2_tble_ci=block_bootstrap_r2(canon["m1"], canon["tau"], rng),
            relrms_empirical=rel_rms(pred_c, canon["tau"]),
            relrms_equilibrium=rel_rms(canon["m0"], canon["tau"]),
            relrms_tble=rel_rms(canon["m1"], canon["tau"]))
        arrays[f"{tag}_canonical_tau_ref"] = canon["tau"]
        arrays[f"{tag}_canonical_pred_empirical"] = pred_c
        arrays[f"{tag}_canonical_pred_equilibrium"] = canon["m0"]
        arrays[f"{tag}_canonical_pred_tble"] = canon["m1"]
        arrays[f"{tag}_canonical_a"] = canon["a"]
        arrays[f"{tag}_canonical_b"] = canon["b"]

        # --- certified lower bound: within one case, and pooled across families
        floors = {}
        for label, pool in (("canonical_hill", [canon]),
                            ("hills_pooled", hills),
                            ("cross_family_pooled", hills + [wavy, convdiv])):
            a = np.concatenate([slog(c["a"]) for c in pool])
            b = np.concatenate([slog(c["b"]) for c in pool])
            t = np.concatenate([c["t"] for c in pool])
            tau = np.concatenate([c["tau"] for c in pool])
            idx = (np.arange(len(t)) if len(t) <= FLOOR_SUBSAMPLE
                   else np.linspace(0, len(t) - 1, FLOOR_SUBSAMPLE).astype(int))
            d = np.column_stack((a[idx], b[idx]))
            d = (d - d.mean(0)) / (d.std(0) + 1e-30)
            tn = tau[idx] / (np.sqrt(np.mean(tau[idx] ** 2)) + 1e-300)
            floors[label] = {f"L={L}": dict(zip(("floor", "pairs"),
                                                certified_floor(d, tn, L)))
                             for L in LIPSCHITZ_GRID}
            floors[label]["n_used"] = int(len(idx))
        # deployed models' own measured smoothness in the same coordinates
        dcan = np.column_stack((slog(canon["a"]), slog(canon["b"])))
        dcan = (dcan - dcan.mean(0)) / (dcan.std(0) + 1e-30)
        scale = np.sqrt(np.mean(canon["tau"] ** 2))
        floors["measured_lipschitz_canonical"] = dict(
            equilibrium=measured_lipschitz(dcan, canon["m0"] / scale),
            tble=measured_lipschitz(dcan, canon["m1"] / scale),
            reference=measured_lipschitz(dcan, canon["tau"] / scale))
        rec["certified_floor"] = floors
        print(f"[{tag}] floors done ({time.time() - t_start:.0f}s)", flush=True)
        out["heights"][tag] = rec

    out["runtime_seconds"] = time.time() - t_start
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "input_sufficiency_bracket.npz", **arrays)
    with (RESULTS / "input_sufficiency_bracket_summary.json").open("w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote codes/results/input_sufficiency_bracket.{{npz,_summary.json}} "
          f"in {out['runtime_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
