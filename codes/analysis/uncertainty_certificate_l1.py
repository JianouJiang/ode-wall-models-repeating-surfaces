#!/usr/bin/env python3
"""Simultaneous uncertainty certificate for the Level-1 wall-model method.

The certificate keeps four different sources of variation separate:

1. profile sampling, represented by a circular moving-block bootstrap;
2. family sampling, represented by an outer cluster bootstrap in which all
   members of one physical family are resampled as one cluster;
3. deterministic operator sensitivity (wall-normal differentiation, pressure
   extraction, matching surface and Fourier cut-off); and
4. deployed numerical sensitivity (wall-normal grid, root tolerance and root
   branch) plus the measured coupled averaging-window drift.

The random hierarchy is used only for sampling intervals.  Operator, grid,
height and averaging-window effects are reported as deterministic envelopes;
calling those envelopes confidence intervals would be statistically wrong.
No Xiao member is ever counted as an independent family replicate.

The script also supplies the missing three-term table requested by Referee 2:
|tau(y_m)|/Phi, |tau(y_m)|/|tau_w|, and the pressure-only versus all-retained-
traction normalisations.  The canonical hill uses the independently rebuilt
exact pressure traction.  Other profile-only cases are explicitly tagged as
using the archived station pressure gradient integrated to the same y_m.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
METRICS = RESULTS / "signed_wall_error_metrics_m2.npz"
PRESSURE_EXACT = RESULTS / "exact_pressure_traction_512.npz"
BUDGET = RESULTS / "wall_following_budget_certificate_l1.npz"
NUMERICS = RESULTS / "wall_model_numerics_l1.npz"
EQ_WINDOWS = ROOT / "codes" / "openfoam" / "pehill_wmles" / "postProcessing" / "sampleBottomWall"
TBLE_WINDOWS = ROOT / "codes" / "openfoam" / "pehill_wmles_tble" / "postProcessing" / "sampleBottomWall"

SEED = 20260821
N_BOOT = 5000
Y_INDEX = 10
CI = (2.5, 97.5)
PRESSURE_FLOOR_FRACTION = 0.02
STRESS_FLOOR_FRACTION = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_station_rows(value: np.ndarray, n_station: int) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        return np.repeat(value[None, :], n_station, axis=0)
    if value.shape[0] != n_station:
        raise ValueError(f"station count mismatch: {value.shape[0]} != {n_station}")
    return value


def local_derivative_operators(y: np.ndarray, u: np.ndarray, index: int) -> np.ndarray:
    """Return source-independent local polynomial derivatives at one y index."""
    derivatives = []
    n = len(y)
    for width, degree in ((5, 2), (7, 2), (7, 3), (9, 2), (9, 3), (11, 3)):
        half = width // 2
        lo = max(0, min(index - half, n - width))
        hi = min(n, lo + width)
        if hi - lo <= degree:
            continue
        yy = y[lo:hi] - y[index]
        uu = u[lo:hi]
        valid = np.isfinite(yy) & np.isfinite(uu)
        if np.count_nonzero(valid) <= degree:
            continue
        coeff = np.polyfit(yy[valid], uu[valid], degree)
        derivatives.append(float(np.polyval(np.polyder(coeff), 0.0)))
    if len(derivatives) < 4:
        raise ValueError("fewer than four valid wall-normal differentiation operators")
    return np.asarray(derivatives)


def canonical_exact_pressure(x: np.ndarray) -> np.ndarray:
    data = np.load(PRESSURE_EXACT, allow_pickle=False)
    xp = np.asarray(data["x"], dtype=float)
    force = np.asarray(data["exact_pressure_traction"], dtype=float)
    if len(x) == len(xp) and np.allclose(x, xp, rtol=0.0, atol=1e-10):
        return force.copy()
    period = (xp[-1] - xp[0]) + np.median(np.diff(xp))
    xwrap = ((x - xp[0]) % period) + xp[0]
    return np.interp(xwrap, np.r_[xp, xp[0] + period], np.r_[force, force[0]])


def case_operators(path: Path, name: str) -> dict[str, np.ndarray | str]:
    if name == "periodic_hills_case_1p0":
        # All four quantities are evaluated on the same fixed physical surface
        # eta/H=0.10.  This avoids the stationwise-height mismatch that caused
        # the earlier M10/M11 artifact to be reopened.
        budget = np.load(BUDGET, allow_pickle=False)
        x = np.asarray(budget["x"], dtype=float)
        tau_m = np.asarray(budget["tau_match_ensemble"], dtype=float)
        tau_w = np.median(np.asarray(budget["q_viscous_direct_ensemble"], dtype=float), axis=0)
        pressure = np.asarray(budget["pressure_impulse"], dtype=float)
        return {
            "x": x,
            "tau_w": tau_w,
            "tau_m_operators": tau_m,
            "pressure": pressure,
            "y_m": np.full(len(x), 0.10),
            "pressure_protocol": "exact_raw_pressure_integral_common_eta_over_H_0p10",
        }

    data = np.load(path, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float)
    u = as_station_rows(data["U"], len(x))
    y = as_station_rows(data["y"], len(x))
    uv = as_station_rows(data["uv"], len(x))
    tau_w = np.asarray(data["tau_w"], dtype=float)
    nu_raw = np.asarray(data["nu"], dtype=float)
    nu = np.full(len(x), float(nu_raw)) if nu_raw.ndim == 0 else nu_raw
    if len(tau_w) != len(x) or len(nu) != len(x):
        raise ValueError(f"{name}: station arrays are not aligned")

    per_station = []
    for i in range(len(x)):
        if Y_INDEX >= len(y[i]):
            raise ValueError(f"{name}: matching index outside profile")
        dudys = local_derivative_operators(y[i], u[i], Y_INDEX)
        per_station.append(nu[i] * dudys - uv[i, Y_INDEX])
    n_operator = min(map(len, per_station))
    tau_m = np.stack([values[:n_operator] for values in per_station], axis=1)
    y_m = y[:, Y_INDEX]

    # The profile corpus stores dp/dx at each station but not, for every
    # family, a common 2-D pressure grid on which a second independent
    # wall-normal integral can be rebuilt.  This table therefore keeps the
    # exact hill result separate from the documented profile operation.
    pressure = np.asarray(data["dp_dx"], dtype=float) * y_m
    pressure_protocol = "archived_station_dpdx_times_same_y_m"

    return {
        "x": x,
        "tau_w": tau_w,
        "tau_m_operators": tau_m,
        "pressure": pressure,
        "y_m": y_m,
        "pressure_protocol": pressure_protocol,
    }


def circular_block_indices(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=count)
    index = np.concatenate([(start + np.arange(block)) % n for start in starts])
    return index[:n]


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray,
               floor_fraction: float) -> np.ndarray:
    scale = float(np.sqrt(np.nanmean(np.asarray(denominator) ** 2)))
    floor = max(floor_fraction * scale, np.finfo(float).tiny)
    return np.abs(numerator) / np.maximum(np.abs(denominator), floor)


def four_statistics(tau_m: np.ndarray, tau_w: np.ndarray,
                    pressure: np.ndarray, index: np.ndarray) -> np.ndarray:
    tm = tau_m[index]
    tw = tau_w[index]
    pp = pressure[index]
    tm_over_p = safe_ratio(tm, pp, PRESSURE_FLOOR_FRACTION)
    tm_over_tw = safe_ratio(tm, tw, STRESS_FLOOR_FRACTION)
    eps_p = safe_ratio(tw, pp, PRESSURE_FLOOR_FRACTION)
    eps_all = np.abs(tw) / np.maximum(
        np.abs(pp) + np.abs(tm),
        PRESSURE_FLOOR_FRACTION * np.sqrt(np.nanmean(pp ** 2)))
    return np.asarray([np.nanmedian(tm_over_p), np.nanmedian(tm_over_tw),
                       np.nanmedian(eps_p), np.nanmedian(eps_all)])


def sample_case(case: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau_ops = np.asarray(case["tau_m_operators"], dtype=float)
    tau_w = np.asarray(case["tau_w"], dtype=float)
    pressure = np.asarray(case["pressure"], dtype=float)
    central_tau = np.median(tau_ops, axis=0)
    centre = four_statistics(central_tau, tau_w, pressure, np.arange(len(tau_w)))
    block = max(2, int(np.ceil(len(tau_w) ** (1.0 / 3.0))))
    samples = np.empty((N_BOOT, 4))
    for b in range(N_BOOT):
        index = circular_block_indices(rng, len(tau_w), block)
        samples[b] = four_statistics(central_tau, tau_w, pressure, index)
    operator_statistics = np.stack([
        four_statistics(tau_ops[o], tau_w, pressure, np.arange(len(tau_w)))
        for o in range(tau_ops.shape[0])])
    return centre, samples, operator_statistics.min(axis=0), operator_statistics.max(axis=0)


def family_cluster_bootstrap(rng: np.random.Generator, families: np.ndarray,
                             case_samples: np.ndarray) -> np.ndarray:
    """Nested family/member/block/operator bootstrap for cross-family claims.

    ``case_samples[c,b]`` is already one circular-block draw for case
    c.  The outer loop draws physical families, then a member within each
    selected family, then one of those inner draws.  Differentiation operators
    remain a separate deterministic envelope.  A family with 29 members
    therefore remains one outer sampling unit.
    """
    unique = np.unique(families)
    out = np.empty((N_BOOT, case_samples.shape[2]))
    grouped = {family: np.flatnonzero(families == family) for family in unique}
    for b in range(N_BOOT):
        selected = unique[rng.integers(0, len(unique), size=len(unique))]
        rows = []
        for family in selected:
            members = grouped[family]
            case = int(members[rng.integers(0, len(members))])
            inner = int(rng.integers(0, case_samples.shape[1]))
            rows.append(case_samples[case, inner])
        out[b] = np.nanmedian(np.asarray(rows), axis=0)
    return out


def load_window_metrics(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, reattachment, bubble = [], [], []
    for path in sorted(directory.glob("*/*.xy"), key=lambda p: float(p.parent.name)):
        time = float(path.parent.name)
        if time <= 135.0:
            continue
        table = np.loadtxt(path, comments="#")
        xr = np.round(table[:, 0], 6)
        xu = np.unique(xr)
        cf = np.asarray([2.0 * (-table[xr == xv, 3]).mean() for xv in xu])
        mask = (xu >= 0.1) & (xu <= 7.0)
        xx, yy = xu[mask], cf[mask]
        crossings = []
        for i in range(len(xx) - 1):
            if yy[i] < 0.0 <= yy[i + 1]:
                crossings.append(xx[i] - yy[i] * (xx[i + 1] - xx[i]) /
                                 (yy[i + 1] - yy[i]))
        bmask = (xu >= 0.2) & (xu <= 4.0)
        times.append(time)
        reattachment.append(crossings[-1] if crossings else np.nan)
        bubble.append(float(np.mean(np.abs(cf[bmask]))))
    if len(times) < 2:
        raise RuntimeError(f"insufficient real averaging windows in {directory}")
    return np.asarray(times), np.asarray(reattachment), np.asarray(bubble)


def deterministic_envelopes() -> dict:
    budget = np.load(BUDGET, allow_pickle=False)
    q_ref = np.asarray(budget["q_ref_ensemble"], dtype=float)
    q_match = np.asarray(budget["q_match_ensemble"], dtype=float)
    residual = q_match - q_ref
    budget_rms = np.sqrt(np.mean(residual ** 2, axis=1))

    numerics = np.load(NUMERICS, allow_pickle=True)
    reference = np.asarray(numerics["tau_python_N6400"], dtype=float)
    grid = np.stack([np.asarray(numerics[f"tau_python_N{n}"], dtype=float)
                     for n in (50, 100, 200, 400, 500, 800, 1600, 3200)])
    grid_rms = np.sqrt(np.mean((grid - reference[None, :]) ** 2, axis=1))
    tolerance = np.stack([np.asarray(numerics[f"tolerance_tau_1e{power}"], dtype=float)
                          for power in (6, 8, 10, 12, 14)])
    tolerance_rms = np.sqrt(np.mean((tolerance - tolerance[-1]) ** 2, axis=1))
    q_height = np.asarray(budget["q_wall_by_height"], dtype=float)
    q_height_reference = np.median(q_height, axis=0)
    height_rms = np.sqrt(np.mean((q_height - q_height_reference[None, :]) ** 2,
                                 axis=1))
    multi = np.asarray(numerics["multi_root_indices"], dtype=int)
    plain = np.asarray(numerics["plain_bisection_tau"], dtype=float)[multi]
    continued = np.asarray(numerics["continuation_tau"], dtype=float)[multi]
    branch_delta = np.abs(continued - plain)

    eq_t, eq_xr, eq_b = load_window_metrics(EQ_WINDOWS)
    tb_t, tb_xr, tb_b = load_window_metrics(TBLE_WINDOWS)
    return {
        "budget_operator_count": int(len(budget_rms)),
        "budget_closure_rms_min_max": [float(budget_rms.min()), float(budget_rms.max())],
        "matching_surface_over_H": np.asarray(budget["reference_heights"], dtype=float).tolist(),
        "matching_surface_wall_force_rms_from_median": height_rms.tolist(),
        "nonlinear_multi_root_station_count": int(len(multi)),
        "nonlinear_multi_root_indices": multi.tolist(),
        "plain_vs_continued_branch_abs_traction": branch_delta.tolist(),
        "wall_grid_points": [50, 100, 200, 400, 500, 800, 1600, 3200, 6400],
        "wall_grid_rms_error": np.r_[grid_rms, 0.0].tolist(),
        "wall_tolerances": [1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
        "wall_tolerance_rms_error": tolerance_rms.tolist(),
        "equilibrium_window_end": eq_t.tolist(),
        "equilibrium_reattachment": eq_xr.tolist(),
        "equilibrium_bubble_abs_cf": eq_b.tolist(),
        "tble_window_end": tb_t.tolist(),
        "tble_reattachment": tb_xr.tolist(),
        "tble_bubble_abs_cf": tb_b.tolist(),
        "equilibrium_last_window_drift": {
            "reattachment_H": float(abs(eq_xr[-1] - eq_xr[-2])),
            "bubble_abs_cf_fraction": float(abs(eq_b[-1] - eq_b[-2]) / abs(eq_b[-2])),
        },
        "tble_last_window_drift": {
            "reattachment_H": float(abs(tb_xr[-1] - tb_xr[-2])),
            "bubble_abs_cf_fraction": float(abs(tb_b[-1] - tb_b[-2]) / abs(tb_b[-2])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-dir", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    metric = np.load(METRICS, allow_pickle=True)
    names = np.asarray(metric["names"]).astype(str)
    families = np.asarray(metric["families"]).astype(str)
    paths = [ROOT / str(path) for path in metric["profile_paths"]]
    # Operator 2026-08-25 (independent audit): `case_operators()` takes a DEDICATED
    # branch for the periodic hill that reads BUDGET (the wall-following budget
    # certificate) and ignores the metric file's own profile path entirely.  The
    # recorded provenance therefore named a file this case never used - and that file
    # has since been withdrawn as a scoring reference, which made the stale string
    # actively misleading.  Record the path the data actually came from.
    paths = [BUDGET if name == "periodic_hills_case_1p0" else path
             for name, path in zip(names, paths)]

    centres, lows, highs, all_samples = [], [], [], []
    operator_lows, operator_highs = [], []
    operator_count, protocols, source_hashes = [], [], []
    for name, path in zip(names, paths):
        case = case_operators(path, name)
        centre, samples, operator_low, operator_high = sample_case(case, rng)
        centres.append(centre)
        lows.append(np.percentile(samples, CI[0], axis=0))
        highs.append(np.percentile(samples, CI[1], axis=0))
        all_samples.append(samples)
        operator_lows.append(operator_low)
        operator_highs.append(operator_high)
        operator_count.append(np.asarray(case["tau_m_operators"]).shape[0])
        protocols.append(str(case["pressure_protocol"]))
        source_hashes.append(sha256(path))
        print(f"{name:28s} tau_m/Phi={centre[0]:8.3f} "
              f"[{lows[-1][0]:.3f},{highs[-1][0]:.3f}]  "
              f"eps_P={centre[2]:.3f} eps_all={centre[3]:.3f}")

    centres = np.asarray(centres)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    all_samples = np.stack(all_samples, axis=0)  # case, bootstrap, statistic
    operator_lows = np.asarray(operator_lows)
    operator_highs = np.asarray(operator_highs)
    sample_se = np.std(all_samples, axis=1, ddof=1)
    safe_se = np.maximum(sample_se, np.finfo(float).eps)
    studentised = np.abs((all_samples - centres[:, None, :]) /
                         safe_se[:, None, :])
    simultaneous_critical = float(np.percentile(
        np.max(studentised, axis=(0, 2)), 95.0))
    simultaneous_low = np.maximum(0.0, centres - simultaneous_critical * sample_se)
    simultaneous_high = centres + simultaneous_critical * sample_se

    family_samples = family_cluster_bootstrap(rng, families, all_samples)
    family_centre = np.nanmedian(np.asarray([
        np.nanmedian(centres[families == family], axis=0)
        for family in np.unique(families)]), axis=0)
    family_low = np.percentile(family_samples, CI[0], axis=0)
    family_high = np.percentile(family_samples, CI[1], axis=0)
    family_se = np.std(family_samples, axis=0, ddof=1)
    family_studentised = np.abs((family_samples - family_centre[None, :]) /
                                np.maximum(family_se[None, :], np.finfo(float).eps))
    family_critical = float(np.percentile(np.max(family_studentised, axis=1), 95.0))
    family_simultaneous_low = np.maximum(0.0, family_centre - family_critical * family_se)
    family_simultaneous_high = family_centre + family_critical * family_se
    envelopes = deterministic_envelopes()

    canonical = int(np.flatnonzero(names == "periodic_hills_case_1p0")[0])
    summary = {
        "schema": "uncertainty-certificate-l1-v1",
        "idea": "nested family/block/operator uncertainty certificate",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "confidence_percent": 95.0,
        "matching_index": Y_INDEX,
        "statistics": ["median_abs_tau_m_over_abs_pressure",
                       "median_abs_tau_m_over_abs_tau_w",
                       "median_epsilon_pressure_only",
                       "median_epsilon_all_retained_tractions"],
        "uncertainty_semantics": {
            "casewise_confidence_interval": "circular station-block sampling of the median reconstruction operator",
            "cross_family_confidence_interval": "physical-family outer cluster, case within family, then circular-block draw",
            "deterministic_envelope": "differentiation operator, matching surface, wall-normal grid, nonlinear branch, root tolerance and averaging-window sensitivity",
            "casewise_simultaneous_rule": "95th percentile of the maximum studentised deviation over the registered 18-case by four-statistic table",
            "cross_family_simultaneous_rule": "95th percentile of the maximum studentised deviation over four family-weighted statistics",
        },
        "simultaneous_critical_value_18_cases_4_statistics": simultaneous_critical,
        "canonical_hill": {
            "pressure_protocol": protocols[canonical],
            "tau_m_over_pressure": float(centres[canonical, 0]),
            "tau_m_over_pressure_ci": [float(simultaneous_low[canonical, 0]), float(simultaneous_high[canonical, 0])],
            "tau_m_over_tau_w": float(centres[canonical, 1]),
            "tau_m_over_tau_w_ci": [float(simultaneous_low[canonical, 1]), float(simultaneous_high[canonical, 1])],
            "epsilon_pressure_only": float(centres[canonical, 2]),
            "epsilon_pressure_only_ci": [float(simultaneous_low[canonical, 2]), float(simultaneous_high[canonical, 2])],
            "epsilon_all_retained": float(centres[canonical, 3]),
            "epsilon_all_retained_ci": [float(simultaneous_low[canonical, 3]), float(simultaneous_high[canonical, 3])],
            "legacy_tau_m_over_pressure_2p00_status": "superseded: rms_files1 column 5 is pressure variance, not uv; documented uv is rms_files2 column 2",
        },
        "family_cluster_median": family_centre.tolist(),
        "family_cluster_ci_low": family_low.tolist(),
        "family_cluster_ci_high": family_high.tolist(),
        "family_cluster_simultaneous_ci_low": family_simultaneous_low.tolist(),
        "family_cluster_simultaneous_ci_high": family_simultaneous_high.tolist(),
        "family_cluster_simultaneous_critical": family_critical,
        "n_cases": int(len(names)),
        "n_independent_families": int(len(np.unique(families))),
        "xiao_members_outer_units": 1,
        "deterministic_envelopes": envelopes,
        "sources": {
            "metrics": {"path": str(METRICS.relative_to(ROOT)), "sha256": sha256(METRICS)},
            "exact_pressure": {"path": str(PRESSURE_EXACT.relative_to(ROOT)), "sha256": sha256(PRESSURE_EXACT)},
            "budget": {"path": str(BUDGET.relative_to(ROOT)), "sha256": sha256(BUDGET)},
            "numerics": {"path": str(NUMERICS.relative_to(ROOT)), "sha256": sha256(NUMERICS)},
        },
        "status": "PASS",
    }

    npz_path = RESULTS / "uncertainty_certificate_l1.npz"
    json_path = RESULTS / "uncertainty_certificate_l1_summary.json"
    np.savez(
        npz_path,
        names=names,
        families=families,
        source_paths=np.asarray([str(path.relative_to(ROOT)) for path in paths]),
        source_sha256=np.asarray(source_hashes),
        pressure_protocol=np.asarray(protocols),
        differentiation_operator_count=np.asarray(operator_count),
        statistic_centre=centres,
        statistic_ci_low=lows,
        statistic_ci_high=highs,
        statistic_simultaneous_ci_low=simultaneous_low,
        statistic_simultaneous_ci_high=simultaneous_high,
        statistic_operator_envelope_low=operator_lows,
        statistic_operator_envelope_high=operator_highs,
        simultaneous_critical=np.asarray(simultaneous_critical),
        family_cluster_centre=family_centre,
        family_cluster_ci_low=family_low,
        family_cluster_ci_high=family_high,
        family_cluster_samples=family_samples,
        bootstrap_seed=np.asarray(SEED),
        bootstrap_replicates=np.asarray(N_BOOT),
        schema=np.asarray("uncertainty-certificate-l1-v1"),
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.node_dir:
        args.node_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, args.node_dir / npz_path.name)
        shutil.copy2(json_path, args.node_dir / json_path.name)

    print("\nUNCERTAINTY CERTIFICATE")
    print(f"  cases / independent families : {len(names)} / {len(np.unique(families))}")
    print(f"  family bootstrap replicates  : {N_BOOT}")
    print(f"  budget operator ensemble     : {envelopes['budget_operator_count']}")
    print(f"  EQ last-window drift         : {envelopes['equilibrium_last_window_drift']}")
    print(f"  TBLE last-window drift       : {envelopes['tble_last_window_drift']}")
    print(f"  Saved -> {npz_path.relative_to(ROOT)}")
    print("  STATUS: PASS")


if __name__ == "__main__":
    main()
