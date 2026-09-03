#!/usr/bin/env python3
"""Independent guard for the Level-1 uncertainty and R2-1 three-term table."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
NPZ = RESULTS / "uncertainty_certificate_l1.npz"
SUMMARY = RESULTS / "uncertainty_certificate_l1_summary.json"
METRIC = RESULTS / "signed_wall_error_metrics_m2.npz"
EXACT_P = RESULTS / "exact_pressure_traction_512.npz"
RAW = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
       "pehill-29-cases-DNS/alph10-9-3036")
HILL_UTILITY = (ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized" /
                "utility/hill-geometry-gereration")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(label: str, condition: bool, counter: list[int]) -> None:
    counter[1] += 1
    if condition:
        counter[0] += 1
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")


def independent_canonical_ratio() -> float:
    """Rebuild the common-surface ratio from raw fields by a distinct scheme.

    This deliberately does not import the producer.  It uses a quadratic wall
    pressure extrapolation and a fourth-order finite-difference periodic
    derivative, whereas the producer's central member uses a linear pressure
    fit and a spectral derivative.
    """
    sys.path.insert(0, str(HILL_UTILITY))
    from hillShape import profile as hill_profile

    mean = np.loadtxt(RAW / "mean_files.dat")
    rms1 = np.loadtxt(RAW / "rms_files1.dat")
    rms2 = np.loadtxt(RAW / "rms_files2.dat")
    x, y = np.unique(mean[:, 0]), np.unique(mean[:, 1])
    nx, ny = len(x), len(y)

    def reshape(table: np.ndarray, column: int) -> np.ndarray:
        return table[:, column].reshape(ny, nx).T

    raw = {
        "U": reshape(mean, 2), "V": reshape(mean, 3),
        "P": reshape(mean, 5), "Rxx": reshape(rms1, 2),
        "Rxy": reshape(rms2, 2),
    }
    h = hill_profile(x.copy())
    eta = np.linspace(0.0, 0.10, 101)
    mapped = {name: np.empty((nx, len(eta))) for name in raw}
    for i in range(nx):
        speed = np.hypot(raw["U"][i], raw["V"][i])
        k0 = int(np.flatnonzero(speed > 1e-10)[0])
        yy = y[k0:]
        dy = yy[:10] - h[i]
        p_wall = np.polyval(np.polyfit(dy, raw["P"][i, k0:k0 + 10], 2), 0.0)
        for name, wall_value in (("U", 0.0), ("V", 0.0),
                                 ("P", p_wall), ("Rxx", 0.0),
                                 ("Rxy", 0.0)):
            mapped[name][i] = np.interp(
                h[i] + eta, np.r_[h[i], yy],
                np.r_[wall_value, raw[name][i, k0:]])

    dx = float(x[1] - x[0])

    def periodic_d4(field: np.ndarray) -> np.ndarray:
        return (-np.roll(field, -2, axis=0) + 8.0 * np.roll(field, -1, axis=0)
                - 8.0 * np.roll(field, 1, axis=0) + np.roll(field, 2, axis=0)) / (12.0 * dx)

    hp = periodic_d4(h)
    u, v, p = mapped["U"], mapped["V"], mapped["P"]
    ue = np.gradient(u, eta, axis=1, edge_order=2)
    ve = np.gradient(v, eta, axis=1, edge_order=2)
    ux = periodic_d4(u) - hp[:, None] * ue
    vx = periodic_d4(v) - hp[:, None] * ve
    tau_m = (mapped["Rxy"][:, -1] - hp * mapped["Rxx"][:, -1]
             - (ue[:, -1] + vx[:, -1]) / 5600.0
             + 2.0 * hp * ux[:, -1] / 5600.0)
    pressure = (periodic_d4(np.trapezoid(p, eta, axis=1))
                - hp * (p[:, -1] - p[:, 0]))
    floor = 0.02 * np.sqrt(np.mean(pressure ** 2))
    return float(np.median(np.abs(tau_m) / np.maximum(np.abs(pressure), floor)))


def clustered_median(families: np.ndarray, values: np.ndarray) -> float:
    return float(np.median([np.median(values[families == family])
                            for family in np.unique(families)]))


def main() -> int:
    counter = [0, 0]
    check("artifacts exist", NPZ.is_file() and SUMMARY.is_file(), counter)
    if not NPZ.is_file() or not SUMMARY.is_file():
        return 1
    data = np.load(NPZ, allow_pickle=False)
    summary = json.loads(SUMMARY.read_text())
    metric = np.load(METRIC, allow_pickle=True)

    names = np.asarray(data["names"]).astype(str)
    families = np.asarray(data["families"]).astype(str)
    centre = np.asarray(data["statistic_centre"], dtype=float)
    low = np.asarray(data["statistic_ci_low"], dtype=float)
    high = np.asarray(data["statistic_ci_high"], dtype=float)
    sim_low = np.asarray(data["statistic_simultaneous_ci_low"], dtype=float)
    sim_high = np.asarray(data["statistic_simultaneous_ci_high"], dtype=float)
    hill = int(np.flatnonzero(names == "periodic_hills_case_1p0")[0])

    check("schema and fixed seed", str(data["schema"]) == "uncertainty-certificate-l1-v1"
          and int(data["bootstrap_seed"]) == 20260821
          and int(data["bootstrap_replicates"]) == 5000, counter)
    check("complete registered case table", len(names) == 18 and centre.shape == (18, 4), counter)
    check("ten physical families, Xiao counted once", len(np.unique(families)) == 10
          and summary["xiao_members_outer_units"] == 1, counter)
    check("metric registry alignment", np.array_equal(names, np.asarray(metric["names"]).astype(str))
          and np.array_equal(families, np.asarray(metric["families"]).astype(str)), counter)
    check("all intervals finite and ordered", np.all(np.isfinite(low))
          and np.all(np.isfinite(high)) and np.all(low <= centre) and np.all(centre <= high), counter)
    check("maximum-studentised intervals are simultaneous and no narrower",
          float(data["simultaneous_critical"]) > 1.96
          and np.all(sim_low <= low + 1e-14) and np.all(sim_high >= high - 1e-14), counter)
    check("all-retained denominator cannot enlarge epsilon", np.all(centre[:, 3] <= centre[:, 2] + 1e-14), counter)
    check("canonical pressure is the exact raw-field operator",
          str(data["pressure_protocol"][hill]) == "exact_raw_pressure_integral_common_eta_over_H_0p10", counter)
    rebuilt = independent_canonical_ratio()
    check("independent raw-field common-surface rebuild lies in casewise interval",
          low[hill, 0] <= rebuilt <= high[hill, 0], counter)
    check("legacy 2.00 column error is superseded, not silently reused",
          centre[hill, 0] < 1.0 and "superseded" in
          summary["canonical_hill"]["legacy_tau_m_over_pressure_2p00_status"], counter)
    bfs = int(np.flatnonzero(names == "bfs_Re13700")[0])
    conv = int(np.flatnonzero(names == "conv_div_channel_Re12600")[0])
    check("BFS three-term size reproduced", low[bfs, 0] < 7.08 < high[bfs, 0], counter)
    check("converging-diverging three-term size reproduced", low[conv, 0] < 4.48 < high[conv, 0], counter)
    check("source hashes are live", all(
        sha256(ROOT / str(path)) == str(digest)
        for path, digest in zip(data["source_paths"], data["source_sha256"])), counter)
    check("registered parent artifacts are byte addressed",
          summary["sources"]["metrics"]["sha256"] == sha256(METRIC)
          and summary["sources"]["exact_pressure"]["sha256"] == sha256(EXACT_P), counter)

    # Negative fixture: duplicating one family 29 times must not change the
    # family-weighted statistic, whereas a naive case median does change.
    fixture_family = np.asarray(["hill", "control", "control"])
    fixture_value = np.asarray([0.1, 2.0, 4.0])
    duplicated_family = np.r_[np.repeat("hill", 29), ["control", "control"]]
    duplicated_value = np.r_[np.repeat(0.1, 29), [2.0, 4.0]]
    cluster_a = clustered_median(fixture_family, fixture_value)
    cluster_b = clustered_median(duplicated_family, duplicated_value)
    naive_a = float(np.median(fixture_value))
    naive_b = float(np.median(duplicated_value))
    check("family-cluster fixture is invariant to 29 duplicate hill members",
          cluster_a == cluster_b and naive_a != naive_b, counter)

    env = summary["deterministic_envelopes"]
    check("32-operator budget envelope retained", env["budget_operator_count"] == 32, counter)
    check("five matching surfaces remain a deterministic force envelope",
          len(env["matching_surface_over_H"]) == 5 and
          len(env["matching_surface_wall_force_rms_from_median"]) == 5 and
          max(env["matching_surface_wall_force_rms_from_median"]) > 0.0, counter)
    check("nonlinear branch ambiguity is detected and bounded separately",
          env["nonlinear_multi_root_station_count"] == 1 and
          env["nonlinear_multi_root_indices"] == [723] and
          env["plain_vs_continued_branch_abs_traction"][0] > 0.0, counter)
    check("real equilibrium averaging windows retained", len(env["equilibrium_window_end"]) == 4
          and env["equilibrium_last_window_drift"]["reattachment_H"] < 0.1, counter)
    check("real TBLE averaging windows retained", len(env["tble_window_end"]) == 3
          and env["tble_last_window_drift"]["bubble_abs_cf_fraction"] < 0.05, counter)
    check("grid and tolerance envelopes end at the reference", env["wall_grid_rms_error"][-1] == 0.0
          and env["wall_tolerance_rms_error"][-1] == 0.0, counter)

    print(f"{counter[0]}/{counter[1]} checks passed")
    return 0 if counter[0] == counter[1] else 1


if __name__ == "__main__":
    sys.exit(main())
