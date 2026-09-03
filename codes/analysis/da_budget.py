#!/usr/bin/env python3
"""Closure-free wall-following momentum certificate for the Xiao hill DNS.

This is the WP1 reference implementation for the JCP revision.  It starts from
the conservative, steady, two-dimensional Reynolds-averaged x-momentum equation

    d_j [ U_x U_j + R_xj + p delta_xj - 2 nu S_xj ] = f_x,

and maps it with s=x and eta=y-h(x).  The Jacobian is one and the normal
contravariant velocity is W=V-h' U.  No eddy viscosity, Boussinesq stress, or
thin-layer deletion occurs anywhere in the reference balance.

The source-column audit is deliberately explicit.  In the documented Xiao
29-case archive, rms_files1.dat contains uu,vv,ww,pp while rms_files2.dat
contains uv,uw,vw.  Earlier project scripts read rms_files1[:,5] as uv.  This
producer uses rms_files2[:,2] and emits a separate audit of the numerical impact.

Outputs are written both to codes/results (the manuscript source of truth) and
to development/nodes/node_003 (the level deliverable).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CODES = ROOT / "codes"
RESULTS = CODES / "results"
NODE = ROOT / "development" / "nodes" / "node_003"
FIGURES = ROOT / "manuscript" / "figures"
RAW29 = (CODES / "raw_data" / "geometry_driven" /
         "xiao_pehill_parameterized" / "pehill-29-cases-DNS" /
         "alph10-9-3036")
RAW5 = (CODES / "raw_data" / "geometry_driven" /
        "xiao_pehill_parameterized" / "pehill-5-cases-DNS" /
        "case_1p0" / "dns-data")
HILL_UTIL = (CODES / "raw_data" / "geometry_driven" /
             "xiao_pehill_parameterized" / "utility" /
             "hill-geometry-gereration")
sys.path.insert(0, str(HILL_UTIL))
from hillShape import profile as hill_profile  # noqa: E402


NU = 1.0 / 5600.0
ETA_MATCH = 0.10
ETA = np.linspace(0.0, 1.0, 401)
REFERENCE_HEIGHTS = np.array([0.40, 0.50, 0.60, 0.75, 0.90])
FORCING_FIT = (0.40, 0.90)
VELOCITY_TOL = 1.0e-10


@dataclass(frozen=True)
class Config:
    pressure_degree: int = 1
    pressure_points: int = 12
    max_fourier_mode: int | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reshape_column(table: np.ndarray, column: int, nx: int, ny: int) -> np.ndarray:
    """The archive is y-major: all x locations are listed for each y."""
    return table[:, column].reshape(ny, nx).T


def load_documented_raw(case: Path = RAW29) -> dict[str, np.ndarray]:
    mean_path = case / "mean_files.dat"
    rms1_path = case / "rms_files1.dat"
    rms2_path = case / "rms_files2.dat"
    mean = np.loadtxt(mean_path)
    rms1 = np.loadtxt(rms1_path)
    rms2 = np.loadtxt(rms2_path)
    if mean.shape[1] != 6 or rms1.shape[1] != 6 or rms2.shape[1] != 5:
        raise RuntimeError("unexpected Xiao raw-column schema")
    # The three ASCII writers use slightly different printed precision for y
    # (maximum round-off 1.22e-7 H); they still describe the same grid.
    if not (np.allclose(mean[:, :2], rms1[:, :2], rtol=0.0, atol=2.0e-7) and
            np.allclose(mean[:, :2], rms2[:, :2], rtol=0.0, atol=2.0e-7)):
        raise RuntimeError("mean/RMS coordinate columns are not aligned")
    x = np.unique(mean[:, 0])
    y = np.unique(mean[:, 1])
    nx, ny = x.size, y.size
    if nx * ny != mean.shape[0]:
        raise RuntimeError("raw field is not a complete tensor-product grid")
    return {
        "x": x,
        "y": y,
        "U": reshape_column(mean, 2, nx, ny),
        "V": reshape_column(mean, 3, nx, ny),
        "P": reshape_column(mean, 5, nx, ny),
        "Rxx": reshape_column(rms1, 2, nx, ny),
        "Ryy": reshape_column(rms1, 3, nx, ny),
        "Rzz": reshape_column(rms1, 4, nx, ny),
        "Pvar": reshape_column(rms1, 5, nx, ny),
        "Rxy": reshape_column(rms2, 2, nx, ny),
        "Rxz": reshape_column(rms2, 3, nx, ny),
        "Ryz": reshape_column(rms2, 4, nx, ny),
        "source_mean_sha256": np.array(sha256(mean_path)),
        "source_rms1_sha256": np.array(sha256(rms1_path)),
        "source_rms2_sha256": np.array(sha256(rms2_path)),
    }


def periodic_derivative(field: np.ndarray, x: np.ndarray,
                        max_mode: int | None = None) -> np.ndarray:
    """Fourier derivative on the endpoint-excluded periodic DNS grid."""
    vector = field.ndim == 1
    work = field[:, None] if vector else field
    dx = float(x[1] - x[0])
    modes = np.fft.fftfreq(x.size) * x.size
    wave = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    spectrum = np.fft.fft(work, axis=0)
    if max_mode is not None:
        spectrum[np.abs(modes) > max_mode, :] = 0.0
    deriv = np.fft.ifft(1j * wave[:, None] * spectrum, axis=0).real
    return deriv[:, 0] if vector else deriv


def surface_interpolate(raw: dict[str, np.ndarray], config: Config,
                        eta: np.ndarray = ETA) -> dict[str, np.ndarray]:
    """Interpolate documented fluid nodes to a common physical eta surface.

    The analytic wall is prepended with no-slip velocity and zero Reynolds
    stress.  Pressure is extrapolated from the first fluid nodes; the fit order
    and width are varied in the uncertainty ensemble.
    """
    x, y = raw["x"], raw["y"]
    h = hill_profile(x.copy())
    speed = np.hypot(raw["U"], raw["V"])
    names = ("U", "V", "P", "Rxx", "Rxy")
    mapped = {name: np.empty((x.size, eta.size)) for name in names}
    p_wall = np.empty(x.size)
    first_fluid = np.empty(x.size, dtype=int)
    for i in range(x.size):
        fluid = np.flatnonzero(speed[i] > VELOCITY_TOL)
        if fluid.size < max(config.pressure_points, 12):
            raise RuntimeError(f"insufficient fluid nodes at station {i}")
        k0 = int(fluid[0])
        first_fluid[i] = k0
        yy = y[k0:]
        dy = yy[:config.pressure_points] - h[i]
        coeff = np.polyfit(dy, raw["P"][i, k0:k0 + config.pressure_points],
                           config.pressure_degree)
        p_wall[i] = np.polyval(coeff, 0.0)
        for name, wall_value in (("U", 0.0), ("V", 0.0),
                                 ("P", p_wall[i]), ("Rxx", 0.0),
                                 ("Rxy", 0.0)):
            abscissa = np.r_[h[i], yy]
            ordinate = np.r_[wall_value, raw[name][i, k0:]]
            mapped[name][i] = np.interp(h[i] + eta, abscissa, ordinate)
    mapped.update(x=x, eta=eta, h=h, p_wall=p_wall,
                  first_fluid=first_fluid)
    return mapped


def direct_wall_shear(raw: dict[str, np.ndarray], mapped: dict[str, np.ndarray],
                      h_prime: np.ndarray, points: int = 2) -> np.ndarray:
    """Wall-on-fluid viscous x-traction from a tangent/normal no-slip fit."""
    y = raw["y"]
    tau = np.empty(raw["x"].size)
    for i, k0 in enumerate(mapped["first_fluid"]):
        dy_vertical = y[k0:k0 + points] - mapped["h"][i]
        tangent_velocity = ((raw["U"][i, k0:k0 + points] +
                             h_prime[i] * raw["V"][i, k0:k0 + points]) /
                            np.sqrt(1.0 + h_prime[i] ** 2))
        slope_vertical = (np.sum(dy_vertical * tangent_velocity) /
                          np.sum(dy_vertical ** 2))
        tau_fluid_on_wall = (NU * np.sqrt(1.0 + h_prime[i] ** 2) *
                             slope_vertical)
        tau[i] = -tau_fluid_on_wall
    return tau


def at_height(eta: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(eta - value)))


def build_certificate(raw: dict[str, np.ndarray], config: Config) -> dict[str, np.ndarray]:
    m = surface_interpolate(raw, config)
    x, eta, h = m["x"], m["eta"], m["h"]
    U, V, P, Rxx, Rxy = (m[k] for k in ("U", "V", "P", "Rxx", "Rxy"))
    hp = periodic_derivative(h, x, config.max_fourier_mode)
    Ue = np.gradient(U, eta, axis=1, edge_order=2)
    Ve = np.gradient(V, eta, axis=1, edge_order=2)
    Us = periodic_derivative(U, x, config.max_fourier_mode)
    Vs = periodic_derivative(V, x, config.max_fourier_mode)
    Ps = periodic_derivative(P, x, config.max_fourier_mode)
    Ux = Us - hp[:, None] * Ue
    Vx = Vs - hp[:, None] * Ve
    Pe = np.gradient(P, eta, axis=1, edge_order=2)
    Px = Ps - hp[:, None] * Pe
    W = V - hp[:, None] * U

    # Conservative Cartesian x-momentum flux transformed to (s,eta).
    flux_s = U ** 2 + Rxx + P - 2.0 * NU * Ux
    flux_y = U * V + Rxy - NU * (Ue + Vx)
    flux_eta = flux_y - hp[:, None] * flux_s

    fit_mask = (eta >= FORCING_FIT[0]) & (eta <= FORCING_FIT[1])
    mean_flux_eta = np.mean(flux_eta, axis=0)
    forcing_fit = np.polyfit(eta[fit_mask], mean_flux_eta[fit_mask], 1)
    forcing = float(forcing_fit[0])
    fitted = np.polyval(forcing_fit, eta[fit_mask])
    denom = np.sum((mean_flux_eta[fit_mask] -
                    np.mean(mean_flux_eta[fit_mask])) ** 2)
    forcing_r2 = 1.0 - np.sum((mean_flux_eta[fit_mask] - fitted) ** 2) / denom

    def reconstruct_wall(height: float) -> np.ndarray:
        k = at_height(eta, height)
        integral = np.trapezoid(flux_s[:, :k + 1], eta[:k + 1], axis=1)
        return (periodic_derivative(integral, x, config.max_fourier_mode) +
                flux_eta[:, k] - forcing * eta[k])

    q_wall_by_height = np.vstack([reconstruct_wall(z) for z in REFERENCE_HEIGHTS])
    q_wall_reference = np.median(q_wall_by_height, axis=0)
    q_wall_height_std = np.std(q_wall_by_height, axis=0, ddof=1)
    q_wall_match = reconstruct_wall(ETA_MATCH)

    km = at_height(eta, ETA_MATCH)

    def integral_derivative(field: np.ndarray) -> np.ndarray:
        integ = np.trapezoid(field[:, :km + 1], eta[:km + 1], axis=1)
        return periodic_derivative(integ, x, config.max_fourier_mode)

    mean_transport = integral_derivative(U ** 2) + U[:, km] * W[:, km]
    pressure_total = integral_derivative(P) - hp * P[:, km]
    reynolds_transport = (integral_derivative(Rxx) + Rxy[:, km] -
                          hp * Rxx[:, km])
    viscous_transport = (integral_derivative(-2.0 * NU * Ux) -
                         NU * (Ue[:, km] + Vx[:, km]) +
                         2.0 * hp * NU * Ux[:, km])
    body_force = np.full(x.size, -forcing * eta[km])
    component_sum = (mean_transport + pressure_total + reynolds_transport +
                     viscous_transport + body_force)

    q_pressure_wall = -hp * m["p_wall"]
    pressure_impulse = pressure_total - q_pressure_wall
    pressure_approx = eta[km] * Px[:, 0]
    q_viscous_reference = q_wall_reference - q_pressure_wall
    q_viscous_direct = direct_wall_shear(raw, m, hp)
    q_wall_direct = q_pressure_wall + q_viscous_direct

    # Matching-plane and double-averaged transfer terms.  W, not V, is the
    # normal flux across an eta=constant surface.
    stress_reynolds = Rxy - hp[:, None] * Rxx
    stress_viscous = (-NU * (Ue + Vx) + 2.0 * hp[:, None] * NU * Ux)
    Ubar = np.mean(U, axis=0)
    Wbar = np.mean(W, axis=0)
    dispersive = np.mean((U - Ubar[None, :]) *
                         (W - Wbar[None, :]), axis=0)
    naive_dispersive = np.mean((U - Ubar[None, :]) *
                               (V - np.mean(V, axis=0)[None, :]), axis=0)

    return {
        **m,
        "h_prime": hp,
        "W": W,
        "Ux": Ux,
        "Vx": Vx,
        "Px": Px,
        "flux_s": flux_s,
        "flux_eta": flux_eta,
        "forcing": np.array(forcing),
        "forcing_r2": np.array(forcing_r2),
        "forcing_intercept": np.array(forcing_fit[1]),
        "mean_flux_eta": mean_flux_eta,
        "q_wall_by_height": q_wall_by_height,
        "q_wall_reference": q_wall_reference,
        "q_wall_height_std": q_wall_height_std,
        "q_wall_match": q_wall_match,
        "q_wall_direct": q_wall_direct,
        "q_pressure_wall": q_pressure_wall,
        "q_viscous_reference": q_viscous_reference,
        "q_viscous_direct": q_viscous_direct,
        "mean_transport": mean_transport,
        "pressure_total": pressure_total,
        "pressure_impulse": pressure_impulse,
        "pressure_approx": pressure_approx,
        "reynolds_transport": reynolds_transport,
        "viscous_transport": viscous_transport,
        "body_force": body_force,
        "component_sum": component_sum,
        "stress_reynolds": stress_reynolds,
        "stress_viscous": stress_viscous,
        "Ubar": Ubar,
        "Wbar": Wbar,
        "dispersive": dispersive,
        "naive_dispersive": naive_dispersive,
        "eta_match_index": np.array(km),
    }


def legacy_stress_column_audit() -> dict[str, np.ndarray]:
    """Reproduce the old 2.00 ratio and recompute it with documented uv."""
    mean = np.loadtxt(RAW5 / "mean_files.dat")
    rms1 = np.loadtxt(RAW5 / "rms_files1.dat")
    rms2 = np.loadtxt(RAW5 / "rms_files2.dat")
    x_unique = np.unique(np.round(mean[:, 0], 6))

    def evaluate(uv_flat: np.ndarray) -> tuple[float, np.ndarray]:
        rows = []
        for xv in x_unique:
            mask = np.abs(mean[:, 0] - xv) < 1.0e-6
            y, u, v, p, uv = (mean[mask, 1], mean[mask, 2], mean[mask, 3],
                              mean[mask, 5], uv_flat[mask])
            order = np.argsort(y)
            y, u, v, p, uv = (z[order] for z in (y, u, v, p, uv))
            fluid = np.flatnonzero((np.abs(u) > 1.0e-6) |
                                   (np.abs(v) > 1.0e-6))
            k0 = max(int(fluid[0]), 1)
            distance = y[k0 - 1:] - y[k0 - 1]
            vel = u[k0 - 1:]
            shear = uv[k0 - 1:]
            nfit = min(4, distance.size - 1)
            tau_w = NU * np.sum(distance[1:1 + nfit] * vel[1:1 + nfit]) / np.sum(
                distance[1:1 + nfit] ** 2)
            tau_m = (NU * np.gradient(vel, distance, edge_order=2)[10] -
                     shear[10])
            rows.append((xv, distance[10], tau_w, tau_m, p[k0]))
        rows = np.asarray(rows)
        dpdx = np.gradient(rows[:, 4], rows[:, 0])
        phi = np.abs(dpdx) * rows[:, 1]
        valid = phi > 1.0e-8
        ratio = np.abs(rows[valid, 3]) / phi[valid]
        return float(np.median(ratio)), ratio

    legacy_median, legacy_ratio = evaluate(rms1[:, 5])
    corrected_median, corrected_ratio = evaluate(rms2[:, 2])
    return {
        "legacy_source": np.array("rms_files1_column_5_mislabelled_as_uv"),
        "correct_source": np.array("rms_files2_column_2_uvmean"),
        "legacy_median_abs_tau_m_over_phi": np.array(legacy_median),
        "corrected_median_abs_tau_m_over_phi": np.array(corrected_median),
        "legacy_ratio": legacy_ratio,
        "corrected_ratio": corrected_ratio,
    }


def save_npz(name: str, **fields: np.ndarray) -> None:
    for directory in (RESULTS, NODE):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / name, **fields)


def save_json(name: str, payload: dict) -> None:
    for directory in (RESULTS, NODE):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)


def make_figure(c: dict[str, np.ndarray], summary: dict) -> None:
    x = c["x"]
    residual = c["q_wall_match"] - c["q_wall_reference"]
    km = int(c["eta_match_index"])
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, c["pressure_impulse"], color="#4477AA", lw=1.1,
            label="exact pressure impulse")
    ax.plot(x, c["mean_transport"], color="#EE7733", lw=1.1,
            label="mean transport")
    ax.plot(x, c["reynolds_transport"], color="#667788", lw=1.1,
            label="Reynolds transport")
    ax.plot(x, c["q_viscous_reference"], color="black", lw=1.2,
            label="viscous wall traction")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set(xlabel=r"phase $x/H$", ylabel=r"signed force per projected area",
           title="(a) Complete wall-following balance")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(x, c["q_wall_reference"], color="#EE7733", lw=1.2,
            label="outer-height reference")
    ax.plot(x, c["q_wall_match"], color="black", lw=0.9, ls="--",
            label=r"held-out $\eta_m/H=0.10$")
    ax.fill_between(x, -summary["wall_traction_uncertainty_rms"],
                    summary["wall_traction_uncertainty_rms"], color="#BBBBBB",
                    alpha=0.35, label="RMS extraction uncertainty")
    ax.plot(x, residual, color="#4477AA", lw=0.7, label="closure residual")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set(xlabel=r"phase $x/H$", ylabel="traction / residual",
           title=f"(b) Held-out closure: {100*summary['station_closure_relative_rms']:.2f}% RMS")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 0]
    ax.plot(c["dispersive"], c["eta"], color="#EE7733", lw=1.2,
            label=r"$\langle\widetilde U\widetilde W\rangle$")
    ax.plot(np.mean(c["stress_reynolds"], axis=0), c["eta"],
            color="#667788", lw=1.2,
            label=r"$\langle R_{xy}-h'R_{xx}\rangle$")
    ax.plot(np.mean(c["stress_viscous"], axis=0), c["eta"],
            color="black", lw=1.0, label="viscous flux")
    ax.axhline(c["eta"][km], color="0.4", ls="--", lw=0.7)
    ax.set(xlabel="wavelength-mean flux", ylabel=r"$\eta/H$",
           ylim=(0, 0.35), title="(c) Matching-plane transfer")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.scatter(c["pressure_impulse"], c["pressure_approx"], s=7,
               color="#4477AA", alpha=0.65)
    lim = np.nanmax(np.abs(np.r_[c["pressure_impulse"], c["pressure_approx"]]))
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
    ax.set(xlabel=r"exact $\int_0^{\eta_m}\partial_x p\,d\eta$",
           ylabel=r"wall-gradient approximation $\eta_m(\partial_xp)_w$",
           title=f"(d) Pressure approximation: rel. RMS {summary['pressure_approx_rel_rms']:.2f}")

    for directory, suffix in ((NODE, "pdf"), (NODE, "png"),
                              (FIGURES, "pdf"), (FIGURES, "png")):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"fig_wall_following_certificate.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    raw = load_documented_raw()
    central_config = Config()
    central = build_certificate(raw, central_config)

    # Differentiation and surface-pressure uncertainty.  The held-out matching
    # surface is never included in the outer reference-height set.
    configs = [Config(degree, points, mode)
               for degree in (1, 2)
               for points in (8, 10, 12, 16)
               for mode in (192, 256, 320, None)]
    ensemble = [build_certificate(raw, cfg) for cfg in configs]
    q_ref_ensemble = np.stack([c["q_wall_reference"] for c in ensemble])
    q_match_ensemble = np.stack([c["q_wall_match"] for c in ensemble])
    q_direct_ensemble = np.stack([c["q_wall_direct"] for c in ensemble])
    q_viscous_direct_ensemble = np.stack([c["q_viscous_direct"] for c in ensemble])
    tau_match_ensemble = np.stack([
        c["stress_reynolds"][:, int(c["eta_match_index"])] +
        c["stress_viscous"][:, int(c["eta_match_index"])]
        for c in ensemble])
    forcing_ensemble = np.array([float(c["forcing"]) for c in ensemble])
    pressure_exact_ensemble = np.stack([c["pressure_impulse"] for c in ensemble])
    pressure_approx_ensemble = np.stack([c["pressure_approx"] for c in ensemble])

    q_ref_sigma = np.std(q_ref_ensemble, axis=0, ddof=1)
    surface_extraction_error = central["q_wall_direct"] - central["q_wall_reference"]
    wall_uncertainty = np.sqrt(q_ref_sigma ** 2 + surface_extraction_error ** 2)
    closure_residual = central["q_wall_match"] - central["q_wall_reference"]
    traction_rms = float(np.sqrt(np.mean(central["q_wall_reference"] ** 2)))
    closure_rms = float(np.sqrt(np.mean(closure_residual ** 2)))
    uncertainty_rms = float(np.sqrt(np.mean(wall_uncertainty ** 2)))

    exact_p = central["pressure_impulse"]
    approx_p = central["pressure_approx"]
    pressure_rel_rms = float(np.sqrt(np.mean((approx_p - exact_p) ** 2)) /
                             np.sqrt(np.mean(exact_p ** 2)))
    pressure_corr = float(np.corrcoef(exact_p, approx_p)[0, 1])

    # Wavelength-integrated quantities; derivative terms vanish independently.
    means = {name: float(np.mean(central[name])) for name in
             ("q_wall_reference", "q_wall_match", "q_wall_direct",
              "q_pressure_wall", "q_viscous_reference", "q_viscous_direct",
              "mean_transport", "pressure_total", "pressure_impulse",
              "reynolds_transport", "viscous_transport", "body_force")}
    form_fraction = abs(means["q_pressure_wall"] / means["q_wall_reference"])
    wavelength_residual = means["q_wall_match"] - means["q_wall_reference"]

    legacy = legacy_stress_column_audit()
    summary = {
        "approach": "full wall-following conservative momentum certificate",
        "source_case": "Xiao 29-case alph10-9-3036 DNS",
        "source_grid": [int(raw["x"].size), int(raw["y"].size)],
        "eta_match_over_H": ETA_MATCH,
        "reference_heights_over_H": REFERENCE_HEIGHTS.tolist(),
        "forcing_fit_interval_over_H": list(FORCING_FIT),
        "forcing": float(central["forcing"]),
        "forcing_fit_r2": float(central["forcing_r2"]),
        "forcing_ensemble_min": float(forcing_ensemble.min()),
        "forcing_ensemble_max": float(forcing_ensemble.max()),
        "station_closure_rms": closure_rms,
        "station_closure_relative_rms": closure_rms / traction_rms,
        "station_closure_correlation": float(np.corrcoef(
            central["q_wall_match"], central["q_wall_reference"])[0, 1]),
        "wall_traction_rms": traction_rms,
        "wall_traction_uncertainty_rms": uncertainty_rms,
        "closure_below_propagated_uncertainty": bool(closure_rms < uncertainty_rms),
        "station_fraction_below_local_uncertainty": float(np.mean(
            np.abs(closure_residual) <= wall_uncertainty)),
        "component_sum_max_abs_error": float(np.max(np.abs(
            central["component_sum"] - central["q_wall_match"]))),
        "wavelength_residual": wavelength_residual,
        "wavelength_residual_relative": abs(wavelength_residual) /
            abs(means["q_wall_reference"]),
        "wavelength_mean_terms": means,
        "pressure_form_fraction_of_total_wall_force": form_fraction,
        "pressure_approx_rel_rms": pressure_rel_rms,
        "pressure_approx_correlation": pressure_corr,
        "legacy_tau_m_over_phi_wrong_column": float(
            legacy["legacy_median_abs_tau_m_over_phi"]),
        "corrected_tau_m_over_phi_documented_uv": float(
            legacy["corrected_median_abs_tau_m_over_phi"]),
        "uses_modelled_eddy_viscosity": False,
        "uses_boussinesq_reference_stress": False,
        "uses_thin_layer_deletion": False,
        "status": "PASS" if (closure_rms / traction_rms < 0.02 and
                               closure_rms < uncertainty_rms and
                               float(central["forcing_r2"]) > 0.999 and
                               np.max(np.abs(central["component_sum"] -
                                             central["q_wall_match"])) < 1e-12)
                  else "FAIL",
    }

    common = {
        "x": central["x"], "eta": central["eta"], "h": central["h"],
        "h_prime": central["h_prime"], "eta_match": np.array(ETA_MATCH),
        "reference_heights": REFERENCE_HEIGHTS,
        "q_wall_by_height": central["q_wall_by_height"],
        "q_wall_reference": central["q_wall_reference"],
        "q_wall_match": central["q_wall_match"],
        "q_wall_direct": central["q_wall_direct"],
        "q_pressure_wall": central["q_pressure_wall"],
        "q_viscous_direct": central["q_viscous_direct"],
        "q_viscous_reference": central["q_viscous_reference"],
        "closure_residual": closure_residual,
        "wall_uncertainty": wall_uncertainty,
        "q_ref_ensemble": q_ref_ensemble,
        "q_match_ensemble": q_match_ensemble,
        "q_direct_ensemble": q_direct_ensemble,
        "q_viscous_direct_ensemble": q_viscous_direct_ensemble,
        "tau_match_ensemble": tau_match_ensemble,
        "forcing_ensemble": forcing_ensemble,
        "source_mean_sha256": raw["source_mean_sha256"],
        "source_rms1_sha256": raw["source_rms1_sha256"],
        "source_rms2_sha256": raw["source_rms2_sha256"],
        "schema": np.array("wall-following-certificate-v1"),
    }
    save_npz("wall_following_budget_certificate_l1.npz", **common,
             mean_transport=central["mean_transport"],
             pressure_total=central["pressure_total"],
             pressure_impulse=central["pressure_impulse"],
             reynolds_transport=central["reynolds_transport"],
             viscous_transport=central["viscous_transport"],
             body_force=central["body_force"],
             component_sum=central["component_sum"],
             forcing=central["forcing"], forcing_r2=central["forcing_r2"])
    save_npz("wavelength_integral_certificate_l1.npz", **{
        key: np.array(value) for key, value in means.items()},
        pressure_form_fraction_of_total_wall_force=np.array(form_fraction),
        wavelength_residual=np.array(wavelength_residual),
        wavelength_residual_relative=np.array(summary["wavelength_residual_relative"]),
        forcing=central["forcing"], forcing_fit_r2=central["forcing_r2"],
        forcing_intercept=central["forcing_intercept"],
        eta=central["eta"], mean_flux_eta=central["mean_flux_eta"])
    save_npz("pressure_impulse_approximation_l1.npz", x=central["x"],
             exact_pressure_impulse=exact_p, wall_gradient_approximation=approx_p,
             exact_ensemble=pressure_exact_ensemble,
             approximation_ensemble=pressure_approx_ensemble,
             relative_rms_error=np.array(pressure_rel_rms),
             correlation=np.array(pressure_corr))
    save_npz("matching_plane_stress_l1.npz", x=central["x"], eta=central["eta"],
             W=central["W"], Ubar=central["Ubar"], Wbar=central["Wbar"],
             dispersive=central["dispersive"],
             naive_cartesian_dispersive=central["naive_dispersive"],
             reynolds_contravariant_mean=np.mean(central["stress_reynolds"], axis=0),
             viscous_contravariant_mean=np.mean(central["stress_viscous"], axis=0),
             reynolds_contravariant_at_match=central["stress_reynolds"][:,
                 int(central["eta_match_index"])],
             viscous_contravariant_at_match=central["stress_viscous"][:,
                 int(central["eta_match_index"])])
    save_npz("legacy_stress_column_audit_l1.npz", **legacy)

    term_dictionary = {
        "coordinate_map": "s=x, eta=y-h(s), Jacobian=1",
        "contravariant_velocity": "W=V-h'(s)U",
        "parent_equation": "partial_s A + partial_eta Q = f_x",
        "A": "U^2+R_xx+p-2 nu partial_x U",
        "Q": "UV+R_xy-nu(partial_y U+partial_x V)-h' A",
        "certificate": "Q_w=partial_s integral_0^eta_m A d eta+Q_m-f_x eta_m",
        "pressure_impulse": "partial_s integral p d eta-h'(p_m-p_w)",
        "reynolds_transport": "partial_s integral R_xx d eta+R_xy,m-h'R_xx,m",
        "dispersive_flux": "<U_tilde W_tilde>; W is required, not Cartesian V",
        "wall_sign": "Q_w is wall-on-fluid x traction per projected wall length",
        "raw_columns": {
            "mean_files.dat": "x,y,U,V,W,p",
            "rms_files1.dat": "x,y,uu,vv,ww,pp",
            "rms_files2.dat": "x,y,uv,uw,vw",
        },
        "reference_model_quantities": "none",
        "thin_layer_terms_deleted": "none",
    }
    save_json("term_dictionary_l1.json", term_dictionary)
    save_json("wall_following_budget_l1_summary.json", summary)
    make_figure(central, summary)

    print("=" * 78)
    print("WP1 WALL-FOLLOWING CONSERVATION CERTIFICATE")
    print("=" * 78)
    print(f"raw grid                         : {raw['x'].size} x {raw['y'].size}")
    print(f"uniform forcing                 : {float(central['forcing']):.8f} "
          f"(fit R2={float(central['forcing_r2']):.8f})")
    print(f"held-out station closure RMS    : {closure_rms:.6e} "
          f"({100*closure_rms/traction_rms:.3f}% of traction RMS)")
    print(f"propagated traction uncertainty : {uncertainty_rms:.6e}")
    print(f"closure below uncertainty       : {closure_rms < uncertainty_rms}")
    print(f"component algebra max error     : {summary['component_sum_max_abs_error']:.3e}")
    print(f"wavelength residual             : {wavelength_residual:+.3e} "
          f"({100*summary['wavelength_residual_relative']:.3f}%)")
    print(f"pressure form-force fraction    : {form_fraction:.3f}")
    print(f"pressure approximation rel RMS  : {pressure_rel_rms:.3f}")
    print(f"tau_m/Phi old -> documented uv  : "
          f"{summary['legacy_tau_m_over_phi_wrong_column']:.3f} -> "
          f"{summary['corrected_tau_m_over_phi_documented_uv']:.3f}")
    print(f"STATUS                           : {summary['status']}")
    print(f"Saved manuscript and node artifacts under {NODE.relative_to(ROOT)}")
    if summary["status"] != "PASS":
        raise SystemExit("WP1 certificate failed its pre-registered numerical gates")


if __name__ == "__main__":
    main()
