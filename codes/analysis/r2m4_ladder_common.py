#!/usr/bin/env python3
"""Shared R2-m4 ladder definitions: DNS tangent-frame fields, the wall-model
ladder operators and the phase-block metric protocol.

Used by r2m4_apriori_ladder.py (a priori, DNS inputs) and
harvest_r2m4_ladder.py (coupled ARCHER2 cases) so both halves of the
causal-sufficiency figure use one truth, one tangent, one metric and one
uncertainty protocol.  Metrics follow the deposited three-grid analysis
(rswm_common_surface_grid_l2 / analyze_grid_results_l3): physical-tangent
traction, RMS-normalised error, descriptive R^2, sign accuracy, signed force,
circular phase-block bootstrap (block Lx/8, 20000 draws).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "models"))
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
import source_faithful_wall_models as wm  # noqa: E402
from make_xiao_dns_wmles_case import HALF_WIDTH, xiao_profile  # noqa: E402

DNS_FILE = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
DNS_SHA256 = "d039cefb93ec1a8555555deed79041921bf8ce98cd1477479087a9804ca7ff85"
BUDGET_SUMMARY = ROOT / "codes/results/wall_following_budget_l1_summary.json"
LX = 9.0
NU = 1.0 / 5600.0
DNS_DRIVING_ACCELERATION = 0.011035961313934037   # wall_following_budget_l1 'forcing'
DENSE_N = 4096
BLOCK_POINTS = 512          # Lx/8, the deposited primary block
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260823
RAW_DIR = (ROOT / "codes/new_data_download/geometry_driven/xiao_pehill_parameterized/"
           "pehill-5-cases-DNS/case_1p0/dns-data")
LADDER = ("M0_equilibrium", "M1_pressure_gradient_ode", "M2_hickel_modelled_convection",
          "Xc_resolved_convection_linear", "Xc_resolved_convection_constant",
          "Xc_exact_convection_profile",
          "Xp_exact_pressure_profile", "Xcp_convection_plus_pressure_profile",
          "Xcpr_plus_streamwise_normal_stress", "Xall_all_omitted_transport",
          "Xfull_all_transport_plus_exact_shear_stress")
COUPLED_MODEL_OF = {
    "equilibrium": "M0_equilibrium",
    "totalGradient": "M1_pressure_gradient_ode",
    "hickel": "M2_hickel_modelled_convection",
    "resolvedConvectionLinear": "Xc_resolved_convection_linear",
    "resolvedConvectionConstant": "Xc_resolved_convection_constant",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def periodic_interp(x, y, target):
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0], np.r_[y, y, y])


def periodic_ddx(f, dx):
    """Fourth-order periodic central d/dx along axis 0 of a uniformly spaced record."""
    f = np.asarray(f, float)
    fp1, fm1 = np.roll(f, -1, axis=0), np.roll(f, 1, axis=0)
    fp2, fm2 = np.roll(f, -2, axis=0), np.roll(f, 2, axis=0)
    return (8.0 * (fp1 - fm1) - (fp2 - fm2)) / (12.0 * dx)


def wall_tangent(x):
    """Downstream unit tangent of the analytic Xiao surface at stations x."""
    h = np.asarray([xiao_profile(v) if v <= HALF_WIDTH else
                    xiao_profile(LX - v) if v >= LX - HALF_WIDTH else 0.0 for v in x])
    dx = float(np.median(np.diff(x)))
    slope = periodic_ddx(h, dx)
    mag = np.sqrt(1.0 + slope ** 2)
    return h, slope, 1.0 / mag, slope / mag


class DnsTangentFields:
    """Xiao 512x257 archive re-expressed on a common wall-offset grid in the
    local wall-tangent frame, with the mean convective acceleration."""

    def __init__(self, eta_max=0.6, n_eta=301):
        if sha256(DNS_FILE) != DNS_SHA256:
            raise RuntimeError("DNS archive hash mismatch")
        d = np.load(DNS_FILE)
        self.x = np.asarray(d["x"], float)
        y = np.asarray(d["y"], float)
        U = np.asarray(d["U"], float)
        V = np.asarray(d["V"], float)
        self.dp_dx_wall = np.asarray(d["dp_dx"], float)
        self.tau_x_legacy = np.asarray(d["tau_w"], float)
        self.dx = float(np.median(np.diff(self.x)))
        if abs(self.dx * self.x.size - LX) > 1e-3:
            raise RuntimeError("archive period is not 9H")
        self.eta = np.linspace(0.0, eta_max, n_eta)
        n = self.x.size
        self.U = np.empty((n, n_eta))
        self.V = np.empty((n, n_eta))
        for i in range(n):
            ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])   # NaN-padded hill columns
            off = y[i, ok] - y[i, 0]
            self.U[i] = np.interp(self.eta, off, U[i, ok])
            self.V[i] = np.interp(self.eta, off, V[i, ok])
        self.P, self.Ruu, self.Rvv, self.Ruv = self._raw_profiles(y, n_eta)
        self.h, self.slope, self.tx, self.ty = wall_tangent(self.x)
        # derivatives: d/dx|_y = d/dx|_eta - h' d/deta ; d/dy = d/deta
        Ue = np.gradient(self.U, self.eta, axis=1)
        Ve = np.gradient(self.V, self.eta, axis=1)
        Ux = periodic_ddx(self.U, self.dx) - self.slope[:, None] * Ue
        Vx = periodic_ddx(self.V, self.dx) - self.slope[:, None] * Ve
        conv_x = self.U * Ux + self.V * Ue
        conv_y = self.U * Vx + self.V * Ve
        self.Ut = self.U * self.tx[:, None] + self.V * self.ty[:, None]
        self.conv_t = conv_x * self.tx[:, None] + conv_y * self.ty[:, None]
        # Cartesian component fields on (x, eta) for normal-line sampling
        Pe = np.gradient(self.P, self.eta, axis=1)
        Px = periodic_ddx(self.P, self.dx) - self.slope[:, None] * Pe
        self._grad_p = (Px, Pe)
        self._conv_x_y = (conv_x, conv_y)
        # station-frame normal stress gradient needs the station tangent, which
        # varies per station; store the three Reynolds-stress component
        # gradients and combine per station below.
        def grad2(f):
            fe = np.gradient(f, self.eta, axis=1)
            return periodic_ddx(f, self.dx) - self.slope[:, None] * fe, fe
        self._dR_xy = grad2(self.Ruu) + grad2(self.Rvv) + grad2(self.Ruv)   # 6 fields
        Ux = periodic_ddx(self.U, self.dx) - self.slope[:, None] * Ue
        Vx = periodic_ddx(self.V, self.dx) - self.slope[:, None] * Ve
        self._tau_xy = (Ux, Ue, Vx, Ve, self.Ruu, self.Rvv, self.Ruv)
        def second(f):
            fe = np.gradient(f, self.eta, axis=1)
            fx = periodic_ddx(f, self.dx) - self.slope[:, None] * fe
            fxx = periodic_ddx(fx, self.dx) - self.slope[:, None] * np.gradient(fx, self.eta, axis=1)
            fxy = np.gradient(fx, self.eta, axis=1)
            fyy = np.gradient(fe, self.eta, axis=1)
            return fxx, fxy, fyy
        self._second_xy = second(self.U) + second(self.V)   # Uxx,Uxy,Uyy,Vxx,Vxy,Vyy
        # effective streamwise source along the tangent: wall dp/ds minus the
        # DNS driving acceleration (same sign convention as the coupled BC)
        self.dpds_total = self.dp_dx_wall * self.tx - DNS_DRIVING_ACCELERATION * self.tx
        self.phase = np.mod((self.x - self.x.min()) / LX, 1.0)
        # truth: nu dU_t/dn from the first four archive points (deposit protocol)
        tau_s = np.empty(n)
        for i in range(n):
            off = y[i, 1:5] - y[i, 0]
            ut = U[i, 1:5] * self.tx[i] + V[i, 1:5] * self.ty[i]
            tau_s[i] = NU * float(np.sum(off * ut) / np.sum(off ** 2)) / self.tx[i]
        self.tau_s_truth = tau_s

    def _raw_profiles(self, y_npz, n_eta):
        mean = np.loadtxt(RAW_DIR / "mean_files.dat")
        rms1 = np.loadtxt(RAW_DIR / "rms_files1.dat")
        rms2 = np.loadtxt(RAW_DIR / "rms_files2.dat")
        n = self.x.size
        P = np.empty((n, n_eta)); Ruu = np.empty((n, n_eta)); Rvv = np.empty((n, n_eta)); Ruv = np.empty((n, n_eta))
        for i in range(n):
            m = np.abs(mean[:, 0] - self.x[i]) < 1e-6
            yy = mean[m, 1]; o = np.argsort(yy); yy = yy[o]
            k0 = int(np.argmin(np.abs(yy - y_npz[i, 0])))
            if abs(yy[k0] - y_npz[i, 0]) > 1e-6:
                raise RuntimeError("raw/archive wall index mismatch at station %d" % i)
            off = yy[k0:] - yy[k0]
            # the archive stores p=0 on the wall row; use the fluid points with
            # dp/dn=0 (constant) extrapolation to the wall, as the deposit's
            # wall-pressure extraction (first fluid point) does
            P[i] = np.interp(self.eta, off[1:], mean[m, 5][o][k0 + 1:])
            Ruu[i] = np.interp(self.eta, off, rms1[m, 2][o][k0:])
            Rvv[i] = np.interp(self.eta, off, rms1[m, 3][o][k0:])
            Ruv[i] = np.interp(self.eta, off, rms2[m, 2][o][k0:])
        return P, Ruu, Rvv, Ruv

    # ---- sampling along the true wall-normal line of a station ------------
    # Cartesian fields are stored on (x, eta) with eta the vertical offset from
    # the wall.  A wall-normal line from station i reaches (x_i - n t_y,
    # h_i + n t_x); its vertical offset from the local wall is that height
    # minus h(x').  Periodic x is handled by wrapping the columns.
    def _interp(self, name):
        if not hasattr(self, "_interp_cache"):
            self._interp_cache = {}
        if name not in self._interp_cache:
            field = {"U": (self.U, self.V), "conv": self._conv_x_y, "dpds": self._grad_p,
                     "dRtt": self._dR_xy, "visc": self._second_xy, "tau": self._tau_xy}[name]
            pad = 4
            xs = np.concatenate((self.x[-pad:] - LX, self.x, self.x[:pad] + LX))
            vals = [np.concatenate((f[-pad:], f, f[:pad]), axis=0) for f in field]
            self._interp_cache[name] = [RegularGridInterpolator((xs, self.eta), v, bounds_error=False,
                                                                fill_value=None) for v in vals]
        return self._interp_cache[name]

    def _normal_points(self, i, n):
        n = np.atleast_1d(np.asarray(n, float))
        xp = self.x[i] - n * self.ty[i]
        yp = self.h[i] + n * self.tx[i]
        hp = np.asarray([xiao_profile(v % LX) if (v % LX) <= HALF_WIDTH else
                         xiao_profile(LX - (v % LX)) if (v % LX) >= LX - HALF_WIDTH else 0.0 for v in xp])
        eta_p = np.maximum(yp - hp, 0.0)
        xw = ((xp - self.x[0]) % LX) + self.x[0]
        return np.stack([xw, eta_p], axis=-1)

    def profile_of(self, name, i):
        """Tangent-frame profile (function of wall-normal n) of a named term,
        sampled on the wall-normal line and projected on the station tangent."""
        tx, ty = self.tx[i], self.ty[i]
        def profile(n):
            pts = self._normal_points(i, n)
            comps = [f(pts) for f in self._interp(name)]
            if name == "conv":            # (conv_x, conv_y) -> tangential
                return comps[0] * tx + comps[1] * ty
            if name == "dpds":            # (p_x, p_y) -> tangential minus driving force
                return comps[0] * tx + comps[1] * ty - DNS_DRIVING_ACCELERATION * tx
            if name == "dRtt":            # d/ds of the station-frame normal stress R_tt
                (uu_x, uu_y, vv_x, vv_y, uv_x, uv_y) = comps
                rtt_x = uu_x * tx * tx + 2.0 * uv_x * tx * ty + vv_x * ty * ty
                rtt_y = uu_y * tx * tx + 2.0 * uv_y * tx * ty + vv_y * ty * ty
                return rtt_x * tx + rtt_y * ty
            if name == "visc":            # -nu d2U_t/ds2 = -nu t.(grad grad U_t).t
                (Uxx, Uxy, Uyy, Vxx, Vxy, Vyy) = comps
                utxx = Uxx * tx + Vxx * ty
                utxy = Uxy * tx + Vxy * ty
                utyy = Uyy * tx + Vyy * ty
                return -NU * (utxx * tx * tx + 2.0 * utxy * tx * ty + utyy * ty * ty)
            if name == "tau":             # nu dU_t/dn - <u_t u_n> (U_t, U_n in the station frame)
                (Ux, Uy, Vx, Vy, Ruu, Rvv, Ruv) = comps
                dUt_dn = (Ux * tx + Vx * ty) * (-ty) + (Uy * tx + Vy * ty) * tx
                Rtn = (Rvv - Ruu) * tx * ty + Ruv * (tx * tx - ty * ty)
                return NU * dUt_dn - Rtn
            raise KeyError(name)
        return profile

    def station(self, i, y_m):
        """Inputs at wall-normal matching height y_m for station i."""
        pts = self._normal_points(i, [y_m])
        u, v = (f(pts)[0] for f in self._interp("U"))
        u_m = float(u * self.tx[i] + v * self.ty[i])
        c_m = float(self.profile_of("conv", i)([y_m])[0])
        return u_m, c_m, y_m / self.tx[i]

    def convection_profile(self, i, y_m):
        """c_t as a function of wall-normal distance n in [0, y_m]."""
        return self.profile_of("conv", i)


def ladder_predictions(fields: DnsTangentFields, phases, y_m_of_phase,
                       models=LADDER):
    """Evaluate the ladder at the given phases and matching heights."""
    x_targets = np.mod(phases, 1.0) * LX
    out = {m: np.full(len(phases), np.nan) for m in models}
    diag = {k: np.full(len(phases), np.nan) for k in
            ("u_m", "c_m", "dpds", "y_m", "roots_M1", "truth", "conv_impulse_over_pressure_impulse")}
    for p_index, (xt, y_m) in enumerate(zip(x_targets, y_m_of_phase)):
        i = int(np.argmin(np.abs(fields.x - xt)))
        u_m, c_m, _ = fields.station(i, y_m)
        dpds = float(fields.dpds_total[i])
        diag["u_m"][p_index] = u_m
        diag["c_m"][p_index] = c_m
        diag["dpds"][p_index] = dpds
        diag["y_m"][p_index] = y_m
        diag["truth"][p_index] = fields.tau_s_truth[i]
        diag["conv_impulse_over_pressure_impulse"][p_index] = (
            abs(0.5 * c_m * y_m) / max(abs(dpds) * y_m, 1e-30))
        tau0 = wm.spalding_wall_stress(u_m, y_m, NU) if abs(u_m) > 1e-12 else 0.0
        # cache every exact profile on one fine wall-normal grid so that the
        # shooting residuals interpolate instead of re-sampling the 2-D fields
        n_grid = y_m * np.linspace(0.0, 1.0, 400) ** 1.5
        cached = {}
        for term in ("conv", "dpds", "dRtt", "visc"):
            vals = np.asarray(fields.profile_of(term, i)(n_grid), float)
            cached[term] = (lambda v: (lambda y: np.interp(np.asarray(y, float), n_grid, v)))(vals)
        if "M0_equilibrium" in out:
            out["M0_equilibrium"][p_index] = tau0
        sources = {
            "M1_pressure_gradient_ode": lambda y: np.full_like(np.asarray(y, float), dpds),
            "M2_hickel_modelled_convection": lambda y: wm.hickel_source(y, dpds, NU),
            "Xc_resolved_convection_linear": lambda y: dpds + c_m * np.asarray(y, float) / y_m,
            "Xc_resolved_convection_constant": lambda y: dpds + c_m * np.ones_like(np.asarray(y, float)),
            "Xc_exact_convection_profile": (lambda prof: (lambda y: dpds + prof(y)))(cached["conv"]),
            "Xp_exact_pressure_profile": cached["dpds"],
            "Xcp_convection_plus_pressure_profile": (lambda a, b: (lambda y: a(y) + b(y)))(
                cached["dpds"], cached["conv"]),
            "Xcpr_plus_streamwise_normal_stress": (lambda a, b, c: (lambda y: a(y) + b(y) + c(y)))(
                cached["dpds"], cached["conv"], cached["dRtt"]),
            "Xall_all_omitted_transport": (lambda a, b, c, d: (lambda y: a(y) + b(y) + c(y) + d(y)))(
                cached["dpds"], cached["conv"], cached["dRtt"], cached["visc"]),
        }
        if "Xfull_all_transport_plus_exact_shear_stress" in out:
            # no closure, no shooting: tau_w = tau(y_m) - int_0^{y_m} (all sources) dn
            n_grid = y_m * np.linspace(0.0, 1.0, 400) ** 1.5
            src = sources["Xall_all_omitted_transport"](n_grid)
            impulse = float(np.trapezoid(src, n_grid))
            out["Xfull_all_transport_plus_exact_shear_stress"][p_index] = (
                float(fields.profile_of("tau", i)([y_m])[0]) - impulse)
        for name, source in sources.items():
            if name not in out or name.startswith("Xfull"):
                continue
            result = wm.shoot_wall_stress(u_m, y_m, NU, source, continuation_tau=tau0,
                                          n_points=200)
            out[name][p_index] = result.tau_w
            if name == "M1_pressure_gradient_ode":
                diag["roots_M1"][p_index] = len(result.roots)
    return out, diag


def phase_metrics(phase, pred, truth_phase, truth_tau):
    """Deposited coupled-metric set on the dense periodic phase grid."""
    dense = np.arange(DENSE_N) / DENSE_N
    t = periodic_interp(truth_phase, truth_tau, dense)
    p = periodic_interp(phase, pred, dense)
    err = p - t
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return {
        "relative_rms": float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(t ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
        "sign_accuracy": float(np.mean(np.sign(p) == np.sign(t))),
        "signed_force_ratio": float(np.sum(p) / np.sum(t)),
        "reversed_fraction": float(np.mean(p < 0.0)),
        "truth_reversed_fraction": float(np.mean(t < 0.0)),
        "correlation": float(np.corrcoef(p, t)[0, 1]),
    }, dense, t, p


def block_bootstrap_relative_rms(truth_dense, preds_dense, draws=BOOTSTRAP_DRAWS,
                                 block=BLOCK_POINTS, seed=BOOTSTRAP_SEED):
    """Paired circular phase-block bootstrap of RMS-normalised error (L3 protocol)."""
    n = len(truth_dense)
    per = n // block
    rng = np.random.default_rng(seed)
    err2 = {k: (v - truth_dense) ** 2 for k, v in preds_dense.items()}
    out = {k: np.empty(draws) for k in preds_dense}
    offsets = np.arange(block)[None, None, :]
    for b0 in range(0, draws, 250):
        b1 = min(b0 + 250, draws)
        starts = rng.integers(0, n, size=(b1 - b0, per))
        idx = ((starts[:, :, None] + offsets) % n).reshape(b1 - b0, n)
        den = np.sqrt(np.mean(truth_dense[idx] ** 2, axis=1))
        for k, e2 in err2.items():
            out[k][b0:b1] = np.sqrt(np.mean(e2[idx], axis=1)) / den
    return out


def interval(values):
    lo, med, hi = np.quantile(values, (0.025, 0.5, 0.975))
    return {"low": float(lo), "median": float(med), "high": float(hi)}


# ---------------------------------------------------------------------------
# Pre-registered acceptance rule (written 2026-08-23 before any coupled
# production case landed; see work_progress/archer2_campaign_20260823/R2-m4/
# MANIFEST.md).  The ledger statement under test is
#     "restoring the convective term the ODE drops removes its failure".
# On one surface with relative-RMS errors E(.) and the paired phase-block
# interval of  D = E(Xc_linear) - E(M1):
#     SUPPORTED   iff  D.high < 0  and  E(Xc_linear).high < 1
#     REFUTED     iff  D.low  > 0  or   E(Xc_linear).low  >= 1
#     INCONCLUSIVE otherwise.
# The row closes when the a-priori and the coupled L1 verdicts agree and are
# both SUPPORTED (statement demonstrated) or both REFUTED (statement
# falsified and stated); any INCONCLUSIVE or disagreeing pair keeps it open.
# ---------------------------------------------------------------------------
def side_verdict(e_xc: dict, d_xc_minus_m1: dict) -> str:
    if d_xc_minus_m1["high"] < 0.0 and e_xc["high"] < 1.0:
        return "SUPPORTED"
    if d_xc_minus_m1["low"] > 0.0 or e_xc["low"] >= 1.0:
        return "REFUTED"
    return "INCONCLUSIVE"


def row_verdict(apriori: str, coupled: str) -> str:
    if apriori == coupled and apriori in ("SUPPORTED", "REFUTED"):
        return "CLOSABLE_" + apriori
    return "OPEN_" + apriori + "_" + coupled
