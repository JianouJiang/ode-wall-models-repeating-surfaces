#!/usr/bin/env python3
"""GEOMETRY HOLDOUT for the norm-limited wall model.

The candidate's single constant is calibrated on the periodic hill and frozen.
This producer applies the frozen operator, unchanged and unrefitted, to a
DIFFERENT repeating wall -- the wall-resolved large-eddy simulation of the mild
sinusoidal channel (2a/lambda = 0.1, wavelength equal to the mean channel
height) already deposited for the campaign -- and asks whether it still beats
the strongest faithfully implemented published family there.

Three properties make this a real holdout rather than a second view of the same
data.  The geometry, the Reynolds number and the code path are different; the
reference wall traction is the simulation's OWN resolved wall stress, so none of
the wall-traction estimator questions that affect the hill archive arise here;
and the wall is TOLERATED by the ODE at small matching heights and NOT tolerated
at large ones, so the holdout contains both regimes.

Read-only on every input.  No new simulation, no remote job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "models"))
import r2m4_ladder_common as C  # noqa: E402
import source_faithful_wall_models as wm  # noqa: E402
import faithful_wall_models_l0 as fw  # noqa: E402

WAVY_NPZ = ROOT / "codes/results/r1_sta2_wavy_wrles_20260824.npz"
WAVY_JSON = ROOT / "codes/results/r1_sta2_wavy_wrles_20260824.json"
TOURNAMENT = ROOT / "codes/results/faithful_tournament_l0_{stamp}.json"

# geometry and physical constants of the deposited case, from its generator
# codes/openfoam/make_wavy_cherukat_wrles_case.py
DELTA = 1.0                 # mean channel half height
LAMBDA = 2.0 * DELTA        # wavelength = mean channel height
N_WAVES = 2
LX = N_WAVES * LAMBDA
NU = 1.0 * DELTA / 3460.0   # Re_h = U_b delta / nu = 3460 (Cherukat et al. 1998)
ETA_LIST = (0.05, 0.10, 0.20, 0.30)
FULL_SWEEP_GRID = "G2"
N_QUAD = 400
N_SHOOT = 200
N_PDE = 65
TINY = 1.0e-30


class WavyTangentFields:
    """Deposited wall-resolved fields on a wall-following grid, in the local
    wall-tangent frame, with the same term decomposition the hill analysis uses."""

    def __init__(self, data, grid: str, eta_max: float = 0.6, n_eta: int = 301):
        self.grid = grid
        self.x = np.asarray(data[f"{grid}_x"], float)
        # the MESHED wall, not the analytic sinusoid: the mesh represents the
        # wave by a spline, and the traction the simulation reports is evaluated
        # on the meshed faces, so that surface is the correct wall-normal origin
        key = (f"{grid}_y_wall_mesh" if f"{grid}_y_wall_mesh" in data.files
               else f"{grid}_y_wall")
        self.wall_origin_key = key
        self.y_wall = np.asarray(data[key], float)
        self.slope = np.asarray(data[f"{grid}_h_prime"], float)
        self.tau_truth = np.asarray(data[f"{grid}_tau_t"], float)
        self.p_wall = np.asarray(data[f"{grid}_p_wall"], float)
        y_cell = np.asarray(data[f"{grid}_ycell"], float)
        # The deposited stations are wall-FACE centres, which lie on the meshed
        # sinusoid and are therefore not exactly equispaced in x.  Streamwise
        # derivatives use the uniform surrogate spacing, as the deposit's own
        # reduction does; the departure from uniformity is recorded.
        self.dx = LX / self.x.size
        uniform = self.x[0] + self.dx * np.arange(self.x.size)
        self.spacing_nonuniformity = float(np.max(np.abs(self.x - uniform)) / self.dx)
        if self.spacing_nonuniformity > 0.25:
            raise RuntimeError("deposited wavy stations depart from a uniform "
                               "streamwise spacing by more than 5% of a cell")
        self.nu = NU
        self.eta = np.linspace(0.0, eta_max, n_eta)
        offset = y_cell - self.y_wall[:, None]
        if not np.all(offset > 0.0):
            raise RuntimeError("wall-following coordinate is not positive definite")
        n = self.x.size

        def to_eta(name, wall_value="zero"):
            raw = np.asarray(data[f"{grid}_{name}"], float)
            out = np.empty((n, n_eta))
            for i in range(n):
                if wall_value == "zero":
                    yy = np.concatenate(([0.0], offset[i]))
                    vv = np.concatenate(([0.0], raw[i]))
                else:                       # pressure: zero wall-normal gradient
                    yy = np.concatenate(([0.0], offset[i]))
                    vv = np.concatenate(([raw[i, 0]], raw[i]))
                out[i] = np.interp(self.eta, yy, vv)
            return out

        def ddx(field):
            """Periodic streamwise derivative on the ACTUAL station spacing."""
            pad = 2
            xs = np.concatenate((self.x[-pad:] - LX, self.x, self.x[:pad] + LX))
            fs = np.concatenate((field[-pad:], field, field[:pad]), axis=0)
            return np.gradient(fs, xs, axis=0, edge_order=2)[pad:-pad]

        self._ddx = ddx
        self.U = to_eta("U")
        self.V = to_eta("V")
        self.P = to_eta("P", wall_value="neumann")
        self.Ruu = to_eta("uu")
        self.Rvv = to_eta("vv")
        self.Ruv = to_eta("uv")
        magnitude = np.sqrt(1.0 + self.slope ** 2)
        self.tx = 1.0 / magnitude
        self.ty = self.slope / magnitude

        Ue = np.gradient(self.U, self.eta, axis=1)
        Ve = np.gradient(self.V, self.eta, axis=1)
        Ux = self._ddx(self.U) - self.slope[:, None] * Ue
        Vx = self._ddx(self.V) - self.slope[:, None] * Ve
        conv_x = self.U * Ux + self.V * Ue
        conv_y = self.U * Vx + self.V * Ve
        Pe = np.gradient(self.P, self.eta, axis=1)
        Px = self._ddx(self.P) - self.slope[:, None] * Pe
        self._grad_p = (Px, Pe)
        self._conv_x_y = (conv_x, conv_y)

        def grad2(f):
            fe = np.gradient(f, self.eta, axis=1)
            return self._ddx(f) - self.slope[:, None] * fe, fe

        self._dR_xy = grad2(self.Ruu) + grad2(self.Rvv) + grad2(self.Ruv)
        self._tau_xy = (Ux, Ue, Vx, Ve, self.Ruu, self.Rvv, self.Ruv)

        def second(f):
            fe = np.gradient(f, self.eta, axis=1)
            fx = self._ddx(f) - self.slope[:, None] * fe
            fxx = self._ddx(fx) - self.slope[:, None] * np.gradient(fx, self.eta, axis=1)
            fxy = np.gradient(fx, self.eta, axis=1)
            fyy = np.gradient(fe, self.eta, axis=1)
            return fxx, fxy, fyy

        self._second_xy = second(self.U) + second(self.V)
        self.driving = float(np.asarray(data.get(f"{grid}_gradP", np.nan)))
        self.phase = np.mod((self.x - self.x.min()) / LX, 1.0)
        self._interp_cache: dict = {}

    def set_driving(self, value: float) -> None:
        self.driving = float(value)
        Px, Pe = self._grad_p
        self.dpds_total = (Px[:, 0] * self.tx + Pe[:, 0] * self.ty
                           - self.driving * self.tx)

    def _interp(self, name):
        if name not in self._interp_cache:
            field = {"U": (self.U, self.V), "conv": self._conv_x_y,
                     "dpds": self._grad_p, "dRtt": self._dR_xy,
                     "visc": self._second_xy, "tau": self._tau_xy}[name]
            pad = 4
            xs = np.concatenate((self.x[-pad:] - LX, self.x, self.x[:pad] + LX))
            vals = [np.concatenate((f[-pad:], f, f[:pad]), axis=0) for f in field]
            self._interp_cache[name] = [
                RegularGridInterpolator((xs, self.eta), v, bounds_error=False,
                                        fill_value=None) for v in vals]
        return self._interp_cache[name]

    def _wall_height(self, x_values):
        pad = 4
        xs = np.concatenate((self.x[-pad:] - LX, self.x, self.x[:pad] + LX))
        hs = np.concatenate((self.y_wall[-pad:], self.y_wall, self.y_wall[:pad]))
        return np.interp(np.asarray(x_values, float), xs, hs)

    def _normal_points(self, i, n):
        n = np.atleast_1d(np.asarray(n, float))
        xp = self.x[i] - n * self.ty[i]
        yp = self.y_wall[i] + n * self.tx[i]
        eta_p = np.maximum(yp - self._wall_height(xp), 0.0)
        xw = ((xp - self.x[0]) % LX) + self.x[0]
        return np.stack([xw, eta_p], axis=-1)

    def profile_of(self, name, i):
        tx, ty = self.tx[i], self.ty[i]
        nu = self.nu
        driving = self.driving

        def profile(n):
            pts = self._normal_points(i, n)
            comps = [f(pts) for f in self._interp(name)]
            if name == "conv":
                return comps[0] * tx + comps[1] * ty
            if name == "dpds":
                return comps[0] * tx + comps[1] * ty - driving * tx
            if name == "dRtt":
                (uu_x, uu_y, vv_x, vv_y, uv_x, uv_y) = comps
                rtt_x = uu_x * tx * tx + 2.0 * uv_x * tx * ty + vv_x * ty * ty
                rtt_y = uu_y * tx * tx + 2.0 * uv_y * tx * ty + vv_y * ty * ty
                return rtt_x * tx + rtt_y * ty
            if name == "visc":
                (Uxx, Uxy, Uyy, Vxx, Vxy, Vyy) = comps
                utxx = Uxx * tx + Vxx * ty
                utxy = Uxy * tx + Vxy * ty
                utyy = Uyy * tx + Vyy * ty
                return -nu * (utxx * tx * tx + 2.0 * utxy * tx * ty + utyy * ty * ty)
            if name == "tau":
                (Ux, Uy, Vx, Vy, Ruu, Rvv, Ruv) = comps
                dUt_dn = (Ux * tx + Vx * ty) * (-ty) + (Uy * tx + Vy * ty) * tx
                Rtn = (Rvv - Ruu) * tx * ty + Ruv * (tx * tx - ty * ty)
                return nu * dUt_dn - Rtn
            raise KeyError(name)
        return profile

    def station(self, i, y_m):
        pts = self._normal_points(i, [y_m])
        u, v = (f(pts)[0] for f in self._interp("U"))
        return float(u * self.tx[i] + v * self.ty[i])


def arc_length(fields):
    ds_dx = np.sqrt(1.0 + fields.slope ** 2)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1])
                                         * np.diff(fields.x))))
    period = float(s[-1] + 0.5 * (ds_dx[-1] + ds_dx[0]) * fields.dx)
    return s, period


def evaluate(fields, eta_m: float, c_star: dict, log=print) -> dict:
    """Every arm at one matching height, on this grid."""
    n_st = fields.x.size
    xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
    n_grid = eta_m * xi
    arc, period = arc_length(fields)
    keys = ("dpds", "conv", "dRtt", "visc")
    pred = {}
    norm = {}
    names = ("M0_equilibrium", "M1_pressure_gradient", "M2_hickel", "Xall",
             "ORACLE_closure_free", "M3_yang_integral", "M4_park_moin",
             "NLWM_Xall_frozen", "NLWM_M1_frozen",
             "NLWH_Xall_frozen", "NLWH_M1_frozen",
             "CTL_exact_at_modelled_norm",
             "FAC_exactshape_modelnorm", "FAC_modelshape_exactnorm")
    for a in names:
        pred[a] = np.full(n_st, np.nan)
        norm[a] = np.full(n_st, np.nan)
    u_m = np.empty(n_st)
    tau_at = np.empty(n_st)
    pressure_impulse = np.empty(n_st)
    y_m = np.full(n_st, eta_m)
    limiter = np.zeros(n_st)
    pm_converged = np.zeros(n_st)
    pm_iterations = np.zeros(n_st)

    terms = {k: np.zeros((n_st, N_QUAD)) for k in keys}
    for i in range(n_st):
        for k in keys:
            terms[k][i] = np.asarray(fields.profile_of(k, i)(n_grid), float)
        u_m[i] = fields.station(i, eta_m)
        tau_at[i] = float(fields.profile_of("tau", i)([eta_m])[0])
        pressure_impulse[i] = float(np.trapezoid(terms["dpds"][i], n_grid))

    # one global scale that puts the exact source at the modelled source's norm
    def rms_norm(values_of_i):
        acc = np.empty(n_st)
        for i in range(n_st):
            tau0 = (wm.spalding_wall_stress(u_m[i], eta_m, fields.nu)
                    if abs(u_m[i]) > 1e-12 else 0.0)
            D = fw.equilibrium_diffusivity(n_grid, tau0, fields.nu)
            G = float(np.trapezoid(1.0 / D, n_grid))
            acc[i], _ = fw.assembled_source_norm(n_grid, D, G, values_of_i(i))
        return float(np.sqrt(np.mean(acc ** 2)))

    exact_norm = rms_norm(lambda i: sum(terms[k][i] for k in keys))
    modelled_norm = rms_norm(lambda i: wm.hickel_source(
        n_grid, float(fields.dpds_total[i]), fields.nu))
    matched = float(modelled_norm / max(exact_norm, TINY))

    t0 = time.time()
    for i in range(n_st):
        dpds = float(fields.dpds_total[i])
        tau0 = (wm.spalding_wall_stress(u_m[i], eta_m, fields.nu)
                if abs(u_m[i]) > 1e-12 else 0.0)
        D = fw.equilibrium_diffusivity(n_grid, tau0, fields.nu)
        G = float(np.trapezoid(1.0 / D, n_grid))
        exact_sum = sum(terms[k][i] for k in keys)
        sources = {
            "M1_pressure_gradient": (np.full(N_QUAD, dpds), wm.VAN_DRIEST_A),
            "M2_hickel": (wm.hickel_source(n_grid, dpds, fields.nu),
                          wm.HICKEL_VAN_DRIEST_A),
            "Xall": (exact_sum, wm.VAN_DRIEST_A),
            "CTL_exact_at_modelled_norm": (matched * exact_sum, wm.VAN_DRIEST_A),
        }
        pred["M0_equilibrium"][i] = tau0
        norm["M0_equilibrium"][i] = 0.0
        impulse = float(np.trapezoid(exact_sum, n_grid))
        pred["ORACLE_closure_free"][i] = tau_at[i] - impulse
        norm["ORACLE_closure_free"][i], _ = fw.assembled_source_norm(
            n_grid, D, G, exact_sum)
        for a, (vals, a_plus) in sources.items():
            norm[a][i], _ = fw.assembled_source_norm(n_grid, D, G, vals)
            src = (lambda v: (lambda y: np.interp(np.asarray(y, float), n_grid, v)))(vals)
            pred[a][i] = wm.shoot_wall_stress(
                u_m[i], eta_m, fields.nu, src, continuation_tau=tau0,
                n_points=N_SHOOT, a_plus=a_plus).tau_w
        for tag, vals in (("Xall", exact_sum), ("M1", np.full(N_QUAD, dpds))):
            out = fw.norm_limited_wall_stress(u_m[i], eta_m, fields.nu, n_grid,
                                              vals, c_norm=c_star["NLWM"],
                                              n_points=N_SHOOT)
            pred[f"NLWM_{tag}_frozen"][i] = out.tau_w
            norm[f"NLWM_{tag}_frozen"][i] = out.norm_after
            hz = fw.norm_horizon_wall_stress(u_m[i], eta_m, fields.nu, n_grid,
                                             vals, c_norm=c_star["NLWH"],
                                             n_points=N_SHOOT)
            pred[f"NLWH_{tag}_frozen"][i] = hz.tau_w
            norm[f"NLWH_{tag}_frozen"][i] = hz.norm_after
            if tag == "Xall":
                limiter[i] = float(hz.limiter_active)
        # exact 2x2 factorial of source shape against source amplitude
        n_model = norm["M2_hickel"][i]
        n_exact = norm["Xall"][i]
        if n_model > TINY and n_exact > TINY:
            for a, vals, a_plus in (
                    ("FAC_exactshape_modelnorm", exact_sum * (n_model / n_exact),
                     wm.VAN_DRIEST_A),
                    ("FAC_modelshape_exactnorm",
                     wm.hickel_source(n_grid, dpds, fields.nu) * (n_exact / n_model),
                     wm.HICKEL_VAN_DRIEST_A)):
                norm[a][i], _ = fw.assembled_source_norm(n_grid, D, G, vals)
                src = (lambda v: (lambda y: np.interp(np.asarray(y, float),
                                                      n_grid, v)))(vals)
                pred[a][i] = wm.shoot_wall_stress(
                    u_m[i], eta_m, fields.nu, src, continuation_tau=tau0,
                    n_points=N_SHOOT, a_plus=a_plus).tau_w
        n_uniform = np.linspace(0.0, eta_m, N_PDE)
        pts = fields._normal_points(i, n_uniform)
        Ux, Uy, Vx, Vy, Ruu, Rvv, Ruv = [f(pts) for f in fields._interp("tau")]
        divergence = 0.5 * (Ux + Vy)
        Sxx, Syy, Sxy = Ux - divergence, Vy - divergence, 0.5 * (Uy + Vx)
        numerator = Ruu * Sxx + Rvv * Syy + 2.0 * Ruv * Sxy
        denominator = 2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = np.where(denominator > TINY,
                                  numerator / np.maximum(denominator, TINY), 0.0)
        pm = fw.park_moin_wall_stress(
            u_m[i], eta_m, fields.nu, n_uniform,
            np.interp(n_uniform, n_grid, terms["conv"][i]),
            np.interp(n_uniform, n_grid, terms["dpds"][i]),
            dynamic_correction=np.nan_to_num(correction, nan=0.0, posinf=0.0,
                                             neginf=0.0),
            tolerance=1.0e-10, max_iterations=20000)
        pred["M4_park_moin"][i] = pm.tau_w
        norm["M4_park_moin"][i], _ = fw.assembled_source_norm(
            n_grid, D, G, terms["dpds"][i] + terms["conv"][i])
        pm_converged[i] = float(pm.converged)
        pm_iterations[i] = pm.iterations
    log(f"    stations {time.time() - t0:.0f}s")

    yang = fw.yang_integral_wall_stress_field(
        u_m, y_m, fields.nu, tau_at, pressure_impulse, arc, period,
        tolerance=1.0e-9, max_iterations=20000, relaxation=0.15)
    pred["M3_yang_integral"] = yang.tau_w

    return dict(pred=pred, norm=norm, u_m=u_m, limiter_fraction=float(limiter.mean()),
                matched_scale=matched, exact_norm=exact_norm,
                modelled_norm=modelled_norm,
                park_moin={"converged_stations": int(pm_converged.sum()),
                           "stations": int(n_st),
                           "median_iterations": float(np.median(pm_iterations)),
                           "all_converged": bool(pm_converged.all())},
                yang={"status": yang.status, "converged": bool(yang.converged),
                      "iterations": int(yang.iterations),
                      "residual": float(yang.residual)})


def score_against(phase, truth_tau, phase_pred, pred, arms):
    dense = np.arange(C.DENSE_N) / C.DENSE_N
    truth = C.periodic_interp(phase, truth_tau, dense)
    preds_dense, metrics = {}, {}
    for a in arms:
        v = pred[a]
        ok = np.isfinite(v)
        if ok.sum() < 8:
            continue
        p_d = C.periodic_interp(np.asarray(phase_pred)[ok], v[ok], dense)
        preds_dense[a] = p_d
        err = p_d - truth
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        metrics[a] = {
            "relative_rms": float(np.sqrt(np.mean(err ** 2))
                                  / np.sqrt(np.mean(truth ** 2))),
            "absolute_rms": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
            "sign_accuracy": float(np.mean(np.sign(p_d) == np.sign(truth))),
        }
    boots = C.block_bootstrap_relative_rms(truth, preds_dense)
    for a in metrics:
        metrics[a]["interval"] = C.interval(boots[a])
    return metrics, boots, truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="20260825")
    ap.add_argument("--c-star", type=float, default=None,
                    help="override; by default read from the tournament artifact")
    ap.add_argument("--grids", default="G0,G1,G2")
    args = ap.parse_args()
    t_start = time.time()

    if args.c_star is None:
        tour = json.loads(TOURNAMENT.as_posix().format(stamp=args.stamp)
                          and Path(str(TOURNAMENT).format(stamp=args.stamp)).read_text())
        c_star = {k: float(v) for k, v in tour["c_star"].items()}
        calibrated_on = tour["calibration"]["surface"]
    else:
        c_star = {"NLWM": float(args.c_star), "NLWH": float(args.c_star)}
        calibrated_on = "command line override"
    print("frozen c* = " + ", ".join(f"{k} {v:.4e}" for k, v in c_star.items())
          + f" (calibrated on {calibrated_on})")

    data = np.load(WAVY_NPZ)
    meta = json.loads(WAVY_JSON.read_text())
    result = {
        "schema": "wavy_geometry_holdout_l0/1",
        "question": ("does the norm-limited wall model, with its constant frozen "
                     "on the periodic hill, still beat the best faithful "
                     "published family on a different repeating wall?"),
        "frozen_constant": c_star,
        "calibrated_on": calibrated_on,
        "case": {
            "geometry": "sinusoidal channel, 2a/lambda = 0.10, lambda = mean height",
            "nu": NU, "delta": DELTA, "lambda": LAMBDA, "box_waves": N_WAVES,
            "fidelity": meta["fidelity"],
            "reference": ("the simulation's own resolved wall traction; no wall-"
                          "gradient estimator is involved"),
        },
        "inputs": {
            "wavy_wrles": {"path": str(WAVY_NPZ.relative_to(ROOT)),
                           "sha256": C.sha256(WAVY_NPZ)},
            "faithful_models": {"path": "codes/models/faithful_wall_models_l0.py",
                                "sha256": C.sha256(ROOT / "codes/models/faithful_wall_models_l0.py")},
        },
        "bootstrap": {"block_points": C.BLOCK_POINTS, "dense_points": C.DENSE_N,
                      "draws": C.BOOTSTRAP_DRAWS, "seed": C.BOOTSTRAP_SEED},
        "grids": {},
    }
    arrays = {}
    for grid in args.grids.split(","):
        if f"{grid}_x" not in data.files:
            continue
        fields = WavyTangentFields(data, grid)
        fields.set_driving(float(meta["grids"][grid]["wall"]["gradP_window_mean"]))
        # the deposited traction must be reproducible from the deposited profile
        # at this molecular viscosity: an independent check that nu, the wall
        # origin and the tangent convention are the ones the deposit used.
        recon = np.asarray([float(fields.profile_of("tau", i)([0.0])[0])
                            for i in range(fields.x.size)])
        check = float(np.sqrt(np.mean((recon - fields.tau_truth) ** 2))
                      / np.sqrt(np.mean(fields.tau_truth ** 2)))
        print(f"{grid}: {fields.x.size} stations, wall-traction reconstruction "
              f"relative RMS {check:.4f}")
        etas = ETA_LIST if grid == FULL_SWEEP_GRID else (0.10,)
        entry = {"stations": int(fields.x.size),
                 "cells": meta["grids"][grid]["cells"],
                 "driving_acceleration": fields.driving,
                 "traction_reconstruction_relative_rms": check,
                 "streamwise_spacing_nonuniformity_cells": fields.spacing_nonuniformity,
                 "wall_origin_key": fields.wall_origin_key,
                 "matching_heights": {}}
        for eta_m in etas:
            print(f"  eta_m/delta = {eta_m}")
            ev = evaluate(fields, eta_m, c_star)
            arms = [a for a in ev["pred"] if np.isfinite(ev["pred"][a]).any()]
            metrics, boots, truth = score_against(
                fields.phase, fields.tau_truth, fields.phase, ev["pred"], arms)
            published = [a for a in ("M0_equilibrium", "M1_pressure_gradient",
                                     "M2_hickel", "M3_yang_integral",
                                     "M4_park_moin") if a in metrics]
            best = min(published, key=lambda a: metrics[a]["relative_rms"])
            contrasts = []
            targets = published + ["Xall", "CTL_exact_at_modelled_norm",
                                   "FAC_exactshape_modelnorm",
                                   "FAC_modelshape_exactnorm"]
            for first in ("NLWH_Xall_frozen", "NLWM_Xall_frozen"):
              for a in targets:
                if a not in boots or first not in boots:
                    continue
                d = C.interval(boots[first] - boots[a])
                contrasts.append({
                    "first": first, "second": a, "delta": d,
                    "verdict": ("CANDIDATE_BETTER" if d["high"] < 0.0 else
                                "CANDIDATE_WORSE" if d["low"] > 0.0 else
                                "UNRESOLVED"),
                })
            entry["matching_heights"][f"{eta_m:.2f}"] = {
                "scores": metrics,
                "best_published_family": best,
                "candidate_relative_rms": metrics["NLWH_Xall_frozen"]["relative_rms"],
                "candidate_uniform_rescale_relative_rms":
                    metrics["NLWM_Xall_frozen"]["relative_rms"],
                "shape_amplitude_factorial_relative_rms": {
                    a: metrics[a]["relative_rms"] for a in
                    ("M2_hickel", "FAC_exactshape_modelnorm",
                     "FAC_modelshape_exactnorm", "Xall") if a in metrics},
                "best_published_relative_rms": metrics[best]["relative_rms"],
                "contrasts": contrasts,
                "limiter_active_fraction": ev["limiter_fraction"],
                "matched_norm_scale": ev["matched_scale"],
                "exact_source_norm_rms": ev["exact_norm"],
                "modelled_source_norm_rms": ev["modelled_norm"],
                "park_moin": ev["park_moin"], "yang_integral": ev["yang"],
                "source_norm": {a: float(np.sqrt(np.mean(
                    ev["norm"][a][np.isfinite(ev["norm"][a])] ** 2)))
                    if np.isfinite(ev["norm"][a]).any() else None for a in arms},
            }
            for a in arms:
                arrays[f"{grid}_eta{eta_m:.2f}__pred__{a}"] = ev["pred"][a]
            arrays[f"{grid}_eta{eta_m:.2f}__u_m"] = ev["u_m"]
        arrays[f"{grid}__phase"] = fields.phase
        arrays[f"{grid}__tau_truth"] = fields.tau_truth
        result["grids"][grid] = entry

    # registered holdout verdict, on the finest grid at the headline height
    fine = result["grids"].get(FULL_SWEEP_GRID, {}).get("matching_heights", {})
    verdict = {}
    for key, rec in fine.items():
        best = rec["best_published_family"]
        contrast = next((c for c in rec["contrasts"]
                         if c["second"] == best
                         and c["first"] == "NLWH_Xall_frozen"), None)
        verdict[key] = {
            "best_published_family": best,
            "candidate": rec["candidate_relative_rms"],
            "best_published": rec["best_published_relative_rms"],
            "verdict": contrast["verdict"] if contrast else None,
            "delta": contrast["delta"] if contrast else None,
        }
    result["holdout_verdict"] = verdict
    result["runtime_seconds"] = time.time() - t_start

    out_json = ROOT / "codes/results" / f"wavy_geometry_holdout_l0_{args.stamp}.json"
    out_npz = ROOT / "codes/results" / f"wavy_geometry_holdout_l0_{args.stamp}.npz"
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    np.savez_compressed(out_npz, **arrays)
    print("wrote", out_json.name, out_npz.name)
    for grid, entry in result["grids"].items():
        for key, rec in entry["matching_heights"].items():
            ordered = sorted(rec["scores"], key=lambda a: rec["scores"][a]["relative_rms"])
            print(f"--- {grid} eta_m={key} ---")
            for a in ordered:
                m = rec["scores"][a]
                print(f"   {a:32s} E={m['relative_rms']:8.3f}  R2={m['r2']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
