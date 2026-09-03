#!/usr/bin/env python3
"""harvest_r1_sta2.py -- Hudson/Cherukat wavy-wall WRLES -> ledger artifact (row R1-STA-2).

Consumes the spanwise-averaged 2-D fields written in-job by wavy_wrles_reduce.py
(``codes/results/r1_sta2_wavy_wrles/<case_id>/reduced/wavy2d_<grid>_t<T>.npz`` after
``archer2_run.sh down``) for every available grid of the ladder G0/G1/G2 and produces

    codes/results/r1_sta2_wavy_wrles_<date>.npz / .json

containing, per grid and per averaging window (cumulative [avg_start, T] and the
independent 20-time-unit blocks recovered from the cumulative means):

  (i)  VALIDATION against the deposited reference data
         - separation / reattachment phase vs Maass & Schumann DNS (0.142 / 0.603),
           Cherukat DNS (0.14 / 0.59) and Hudson LDV (0.22 / 0.58);
         - wavy-wall total-drag friction velocity u*/U_b vs Hudson 0.1075 and
           Maass 0.104, plus the global momentum-balance closure (body force vs
           integrated wall forces: an honesty residual, not a model);
         - mean-velocity profiles U(eta) at the 10 Hudson stations and U(y) at the
           10 Maass stations (profile-integrated relative L2 error), Reynolds shear
           stress likewise;
         - first-harmonic amplitude and phase of the wall shear and wall pressure
           relative to the wave (the "phase of wall shear vs wave" key quantity).
  (ii) The paper's a-priori ODE wall-model diagnostic on this geometry, on the
       SAME instrument as the rib WRLES and every hill number
       (``codes/analysis/rib_eps_ode.evaluate``): R^2(tau_w), relRMS, eps = |tau_w|
       /(|dp/dx| y_m), f(eps<1), f(eps<0.1), separation-sign accuracy, for the
       standard mixing-length ODE, the controlled ML shooting, and the exact
       resolved / SGS-completed stress substitutions, at several physical matching
       heights eta_m; the exact pressure integral int_0^eta_m dp/dx deta is carried
       beside the wall-gradient proxy (R1-STA-4a) and the wall-following
       term-by-term balance to eta_m (mean transport, pressure, Reynolds, viscous,
       body force) with its closure residual (R1-STA-3 style).
  (iii) Grid convergence of every headline quantity across the ladder and
        block-window / wave-to-wave uncertainty for each.

No eddy viscosity enters any reference or validation quantity; nu_t appears only
inside the wall model being tested.  Nothing is fabricated: grids that are not on
disk are reported as absent and the status is set accordingly.

Usage:  python3 codes/analysis/harvest_r1_sta2.py [--date 20260823] [--cases-dir ...]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODES = ROOT / "codes"
RESULTS = CODES / "results"
RUNS = RESULTS / "r1_sta2_wavy_wrles"
REF = CODES / "raw_data" / "wavy_wall_reference"
sys.path.insert(0, str(CODES / "analysis"))
sys.path.insert(0, str(CODES / "openfoam"))
from rib_eps_ode import evaluate as ode_evaluate  # noqa: E402  (shared instrument)
from da_budget import periodic_derivative  # noqa: E402  (Fourier derivative, reused)

SCHEMA = "r1_sta2_wavy_wrles_v1"
LEDGER_ROW = "R1-STA-2"
IDEA = ("A wall-resolved LES of the Hudson/Cherukat smooth wavy wall at lambda = 2 delta, "
        "run with the deposited rib-WRLES numerics and validated against two public "
        "reference data sets, gives a second high-fidelity two-dimensional repeating "
        "geometry on which the a-priori ODE wall-model diagnostic is evaluated.")

# reference scalars (see codes/raw_data/wavy_wall_reference/README.md)
REF_SCALARS = {
    "maass_schumann_1996": dict(x_sep=0.142, x_re=0.603, ustar_wavy=0.104, ustar_flat=0.070,
                                Re_h=3380.0),
    "cherukat_1998": dict(x_sep=0.14, x_re=0.59, Re_h=3460.0),
    "hudson_1996": dict(x_sep=0.22, x_re=0.58, ustar_wavy=1.29 / 12.0, Re_h=3380.0,
                        crest_offset=0.0229),   # exp x/lambda=0 is 0.0229 lambda past the crest
}
HUDSON_UB_CM_S, HUDSON_H_CM, HUDSON_NU = 12.0, 2.54, 0.0088
ETA_MATCH_TARGETS = (0.05, 0.10, 0.20, 0.30)      # physical matching heights / delta
# Profile-agreement tolerances.  The original single gate (median rel-L2 <= 0.10 against
# BOTH reference sets) was superseded on 2026-08-24 with evidence: measured on the
# identical metric, the Maass-Schumann DNS is itself 0.101 (median, max 0.152) away from
# the Hudson LDV experiment, so a 0.10 absolute gate against an experiment is tighter than
# the reference DNS achieves and cannot discriminate.  The replacement is
#   (a) a TIGHTER absolute gate against the DNS (0.05), and
#   (b) a RELATIVE gate against the experiment: no worse than HUDSON_DNS_MARGIN times the
#       DNS's own measured distance from that experiment, recomputed from the deposited
#       reference files every run (hudson_dns_reference_l2 below), never hard-coded.
TOL = dict(x_sep=0.05, x_re=0.05, ustar_rel=0.10, maass_l2_median=0.05,
           hudson_dns_margin=1.25, momentum_closure_rel=0.05, yplus_max=1.5)
BLOCK = 20.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- reference data
def load_hudson():
    out = {}
    for i in range(10):
        f = REF / "ercoftac_case076_hudson1996" / ("xlam%02d.dat" % i)
        d = np.loadtxt(f, comments="#")
        out[i / 10.0] = dict(eta=d[:, 0] / HUDSON_H_CM, U=d[:, 1] / HUDSON_UB_CM_S,
                             V=d[:, 2] / HUDSON_UB_CM_S, urms=d[:, 3] / HUDSON_UB_CM_S,
                             vrms=d[:, 4] / HUDSON_UB_CM_S, uv=d[:, 5] / HUDSON_UB_CM_S ** 2,
                             file=str(f.relative_to(ROOT)), sha256=sha256(f))
    return out


def load_maass():
    out = {}
    for i in range(1, 11):
        f = REF / "ercoftac_case077_maass_schumann1996" / ("dat%02d.dat" % i)
        txt = f.read_text().splitlines()
        xl = float([l for l in txt if "x/lambda" in l][0].split("=")[-1])
        blocks, cur = [], []
        for line in txt:
            if line.startswith("#"):
                if cur:
                    blocks.append(np.array(cur))
                    cur = []
                continue
            if line.strip():
                cur.append([float(v) for v in line.split()])
        if cur:
            blocks.append(np.array(cur))
        m, r = blocks[0], blocks[1]
        # Maass frame: H=1 (full height), crest at x=0, mean wall at z=0 -> LES y = 0.1 + 2 z
        out[xl] = dict(y=0.1 + 2.0 * m[:, 1], U=m[:, 2], V=m[:, 4], P=m[:, 5],
                       uu=r[:, 2], vv=r[:, 4], ww=r[:, 3], uv=-r[:, 5], pp=r[:, 6],
                       file=str(f.relative_to(ROOT)), sha256=sha256(f))
    return out


def reference_cross_l2(hudson, maass):
    """Distance between the two REFERENCES on the verifier's own metric.

    Anchors the experiment gate: our WRLES cannot reasonably be asked to sit closer to
    the Hudson LDV profiles than the Maass-Schumann DNS of the same flow does.
    """
    off = REF_SCALARS["hudson_1996"]["crest_offset"]
    a = 0.05 * 2.0                      # amplitude in delta (2a/lambda = 0.1, lambda = 2 delta)
    mph = np.array(sorted(maass))
    uL, vL = [], []
    for ph in sorted(hudson):
        ph_les = (ph + off) % 1.0
        j = mph[np.argmin(np.abs(((mph - ph_les + 0.5) % 1.0) - 0.5))]
        d, e = maass[j], hudson[ph]
        eta_dns = d["y"] - a * (1.0 + np.cos(2.0 * np.pi * j))
        uL.append(rel_l2(e["eta"], e["U"], eta_dns, d["U"], ymax=1.0))
        vL.append(rel_l2(e["eta"], e["uv"], eta_dns, d["uv"], ymax=1.0))
    return dict(U_median=float(np.nanmedian(uL)), U_max=float(np.nanmax(uL)),
                uv_median=float(np.nanmedian(vL)), U_by_station=[float(v) for v in uL],
                note="Maass-Schumann DNS vs Hudson LDV on the identical rel-L2 metric")


# ---------------------------------------------------------------- field helpers
def load_grid(case_dir: Path):
    red = case_dir / "reduced"
    files = sorted(red.glob("wavy2d_*_t*.npz"), key=lambda p: float(p.stem.split("_t")[-1]))
    if not files:
        return None
    snaps = [dict(np.load(f, allow_pickle=False)) for f in files]
    for s, f in zip(snaps, files):
        s["_file"] = f
    geo = json.loads((case_dir / "GEOMETRY.json").read_text())
    man = json.loads((case_dir / "MANIFEST.json").read_text()) if (case_dir / "MANIFEST.json").exists() else {}
    ts = None
    tsf = list(red.glob("wavy_timeseries_*.npz"))
    if tsf:
        ts = dict(np.load(tsf[0], allow_pickle=False))
    return dict(snaps=snaps, geo=geo, manifest=man, timeseries=ts, case_dir=case_dir)


def block_windows(snaps, t0):
    """Recover independent block means [t_{k-1}, t_k] from cumulative means since t0.
    Returns list of dicts with the same field names as a snapshot (means and
    resolved stresses), plus the cumulative final window."""
    means = ("U", "V", "W", "P", "nut", "p_wall_bottom", "p_wall_top", "wss_bottom", "wss_top",
             "wss_bottom_z", "wss_top_z", "p_wall_bottom_z", "p_wall_top_z")
    second = {"uu": ("U", "U"), "vv": ("V", "V"), "ww": ("W", "W"), "uv": ("U", "V"),
              "uw": ("U", "W"), "vw": ("V", "W")}
    out = []
    prev, tprev = None, t0
    for s in snaps:
        t = float(s["time"])
        w = dict(t_start=tprev, t_end=t, kind="block")
        if prev is None:
            for k in means:
                if k in s:
                    w[k] = s[k]
            for k in second:
                if k in s:
                    w[k] = s[k]
        else:
            a, b = (t - t0), (tprev - t0)
            for k in means:
                if k in s and k in prev:
                    w[k] = (a * s[k] - b * prev[k]) / (t - tprev)
            for k, (p, q) in second.items():
                if k in s and k in prev:
                    raw_s = s[k] + s[p] * s[q]
                    raw_p = prev[k] + prev[p] * prev[q]
                    raw_w = (a * raw_s - b * raw_p) / (t - tprev)
                    w[k] = raw_w - w[p] * w[q]
        for k in ("x", "y", "y_wall", "nu", "lambda_", "amplitude", "delta", "H_mean", "y_top", "z"):
            w[k] = s[k]
        out.append(w)
        prev, tprev = s, t
    return out


def meshed_wall(y, y_top, w_analytic, iters=80):
    """Recover the ACTUAL meshed wall height per station from the deposited cell
    centres, and its slope.

    blockMesh distributes the cell centres of a single-block hex along each column as
    ``y[i,j] = w_i + f_j (y_top - w_i)`` with the SAME normalised distribution f_j in
    every column (simpleGrading is normalised on the block edge).  Alternating least
    squares on that rank-1 structure recovers w_i (the meshed wall) and f_j; the fit
    residual is reported and must be a small fraction of the first cell height.

    This matters because blockMesh represents the sinusoid by a spline through
    64 points per wavelength, which deviates from the analytic cosine by up to
    ~5e-4 delta (dominated by the wave's own harmonic plus the spline's 3rd
    harmonic, grid-independent).  On the finest grid that deviation exceeds the
    first cell-centre height, so an analytic origin puts eta < 0 at some stations.
    The wall traction is evaluated by OpenFOAM on the meshed faces, so the meshed
    surface -- not the analytic one -- is the correct wall-normal origin and the
    correct surface for the tangent/normal decomposition (verified: it lowers the
    spurious normal component of the viscous traction at every grid).
    """
    w = np.asarray(w_analytic, float).copy()
    for _ in range(iters):
        f = ((y - w[:, None]) * (y_top - w)[:, None]).sum(0) / ((y_top - w) ** 2).sum()
        w = ((y - y_top * f[None, :]) * (1 - f)[None, :]).sum(1) / ((1 - f) ** 2).sum()
    resid = float(np.abs(y - (w[:, None] + f[None, :] * (y_top - w)[:, None])).max())
    return w, f, resid


def wall_quantities(w):
    """Phase-resolved wall traction on the wavy wall: tangential viscous stress
    (fluid-on-wall, +x forward), Cartesian components, pressure, form/friction drag."""
    x, lam = w["x"], float(w["lambda_"])
    a = float(w["amplitude"])
    yw = w["y_wall_mesh"]                                          # meshed wall (see meshed_wall)
    hp = w["h_prime"]                                              # d y_wall/dx of the MESHED wall
    tx, ty = 1.0 / np.sqrt(1 + hp ** 2), hp / np.sqrt(1 + hp ** 2)  # unit tangent
    nx_, ny_ = -ty, tx                                               # unit normal into fluid
    wss = -w["wss_bottom"]                                           # fluid-on-wall traction
    tau_t = wss[:, 0] * tx + wss[:, 1] * ty
    tau_n = wss[:, 0] * nx_ + wss[:, 1] * ny_
    pw = w["p_wall_bottom"]
    ds = np.sqrt(1 + hp ** 2)                                        # surface length per dx
    # drag per unit span per unit streamwise length: friction = int tau_x ds,
    # form = int p (-n_x) ds  with -n_x = ty  (pressure pushes along the inward normal)
    Lx = float(x[-1] - x[0]) + float(np.median(np.diff(x)))
    fric = np.mean(wss[:, 0] * ds)
    form = np.mean(pw * (-nx_) * ds)
    top = -w["wss_top"][:, 0].mean()
    return dict(x=x, phase=(x / lam) % 1.0, hp=hp, tau_t=tau_t, tau_n=tau_n, tau_x=wss[:, 0],
                p_wall=pw, ds=ds, friction_drag=fric, form_drag=form, total_drag=fric + form,
                ustar_wavy=np.sqrt(max(fric + form, 0.0)), ustar_top=np.sqrt(max(top, 0.0)),
                tau_top=top, Lx=Lx)


def crossings(phase, tau, lam_waves):
    """Separation (+ -> -) and reattachment (- -> +) phases per wave (linear interp on
    the uniform periodic x grid); identical algorithm to the verifier's rebuild."""
    seps, res = [], []
    n = len(phase)
    xg = np.arange(n) / n * lam_waves            # x in wavelengths
    for i in range(n):
        j = (i + 1) % n
        t0, t1 = tau[i], tau[j]
        if t0 == t1:
            continue
        xc = ((xg[i] + (t0 / (t0 - t1)) * (lam_waves / n)) % lam_waves) % 1.0
        if t0 > 0 >= t1:
            seps.append(xc)
        elif t0 < 0 <= t1:
            res.append(xc)
    return np.array(seps), np.array(res)


def first_harmonic(phase, f):
    """f(phase) ~ mean + A cos(2 pi phase - phi); returns (mean, A, phi [rad])."""
    c = np.mean(f * np.cos(2 * np.pi * phase)) * 2
    s = np.mean(f * np.sin(2 * np.pi * phase)) * 2
    return float(np.mean(f)), float(np.hypot(c, s)), float(np.arctan2(s, c))


def column_profiles(w, n_eta=400, eta_max=None):
    """Interpolate every column to a common wall-following eta grid (distance above the
    local wall); returns dict of (nx, n_eta) arrays + eta."""
    x, y, yw = w["x"], w["y"], w["y_wall_mesh"]
    eta_loc = y - yw[:, None]
    if not np.all(eta_loc > 0):
        raise RuntimeError("wall-following coordinate is not positive definite: "
                           "min eta = %.3e (meshed-wall origin failed)" % eta_loc.min())
    if eta_max is None:
        eta_max = float(np.min(eta_loc[:, -1]))
    eta = np.concatenate([[0.0], np.geomspace(float(np.min(eta_loc[:, 0])) * 0.5, eta_max, n_eta - 1)])
    out = dict(eta=eta, x=x)
    for k in ("U", "V", "P", "uu", "vv", "ww", "uv", "nut"):
        if k not in w:
            continue
        arr = np.empty((len(x), n_eta))
        for i in range(len(x)):
            col = w[k][i]
            if k in ("U", "V", "uu", "vv", "ww", "uv", "nut"):
                ycol = np.concatenate([[0.0], eta_loc[i]])
                vcol = np.concatenate([[0.0], col])
            else:   # pressure: zero-gradient at the wall
                ycol = np.concatenate([[0.0], eta_loc[i]])
                vcol = np.concatenate([[col[0]], col])
            arr[i] = np.interp(eta, ycol, vcol)
        out[k] = arr
    return out


def station_profile(prof, phase_target, key, waves=2):
    """Profile at phase x/lambda = phase_target averaged over the `waves` identical waves."""
    ph = prof["x"] / (prof["x"][-1] + np.median(np.diff(prof["x"]))) * waves
    vals = []
    for wv in range(waves):
        xt = (phase_target + wv) % waves
        # periodic linear interpolation along x
        i1 = np.searchsorted(ph, xt) % len(ph)
        i0 = (i1 - 1) % len(ph)
        x0, x1 = ph[i0], ph[i1]
        if x1 < x0:
            x1 += waves
        if xt < x0:
            xt += waves
        f = (xt - x0) / (x1 - x0) if x1 > x0 else 0.0
        vals.append((1 - f) * prof[key][i0] + f * prof[key][i1])
    return np.mean(vals, axis=0)


def rel_l2(y_ref, f_ref, y_les, f_les, ymin=None, ymax=None):
    m = np.isfinite(f_ref)
    if ymin is not None:
        m &= y_ref >= ymin
    if ymax is not None:
        m &= y_ref <= ymax
    if m.sum() < 3:
        return np.nan
    fi = np.interp(y_ref[m], y_les, f_les)
    return float(np.sqrt(np.trapezoid((fi - f_ref[m]) ** 2, y_ref[m]) /
                         max(np.trapezoid(f_ref[m] ** 2, y_ref[m]), 1e-30)))


# ---------------------------------------------------------------- ODE diagnostic
def ode_profiles(w, prof, wallq, eta_m, nu, waves=2):
    """Build rib_eps_ode station dicts with Y_IDX pointing at the cell nearest eta_m.
    Truth tau_w = exact tangential traction; dp/dx = Fourier derivative of the wall
    pressure (periodic over the 2-wave box); uv_total = resolved + SGS (-nut dU/deta)."""
    x, eta = prof["x"], prof["eta"]
    Lx = float(x[-1] - x[0]) + float(np.median(np.diff(x)))
    xper = np.linspace(0, Lx, len(x), endpoint=False)
    dpdx_w = periodic_derivative(wallq["p_wall"], xper)
    # exact pressure-gradient integral to eta_m on the wall-following grid
    Px = periodic_derivative(prof["P"], xper)                  # d/ds at fixed eta
    km = int(np.argmin(np.abs(eta - eta_m)))
    p_int = np.trapezoid(Px[:, :km + 1], eta[:km + 1], axis=1)
    profs = []
    for i in range(len(x)):
        U = prof["U"][i]
        uv = prof["uv"][i]
        dUde = np.gradient(U, eta)
        uv_tot = uv - prof["nut"][i] * dUde if "nut" in prof else uv
        profs.append(dict(x=float(x[i]), y=eta, U=U, uv=uv, uv_total=uv_tot,
                          tau_w=float(wallq["tau_t"][i]), dpdx=float(dpdx_w[i]),
                          p_int=float(p_int[i])))
    return profs, km


def wall_following_balance(w, prof, wallq, eta_m, nu, gradP):
    """Term-by-term conservative x-momentum balance integrated from the wall to eta_m
    (da_budget.build_certificate formulation, generic wall, nu and body force passed in).
    Returns dict of (nx,) arrays; component_sum - q_wall_direct is the closure residual."""
    x, eta = prof["x"], prof["eta"]
    Lx = float(x[-1] - x[0]) + float(np.median(np.diff(x)))
    xper = np.linspace(0, Lx, len(x), endpoint=False)
    U, V, P, Rxx, Rxy = prof["U"], prof["V"], prof["P"], prof["uu"], prof["uv"]
    hp = wallq["hp"]
    Ue = np.gradient(U, eta, axis=1, edge_order=2)
    Ve = np.gradient(V, eta, axis=1, edge_order=2)
    Us, Vs, Ps = (periodic_derivative(f, xper) for f in (U, V, P))
    Ux = Us - hp[:, None] * Ue
    Vx = Vs - hp[:, None] * Ve
    W = V - hp[:, None] * U
    km = int(np.argmin(np.abs(eta - eta_m)))

    def integral_derivative(field):
        return periodic_derivative(np.trapezoid(field[:, :km + 1], eta[:km + 1], axis=1), xper)

    mean_transport = integral_derivative(U ** 2) + U[:, km] * W[:, km]
    pressure_total = integral_derivative(P) - hp * P[:, km]
    reynolds_transport = integral_derivative(Rxx) + Rxy[:, km] - hp * Rxx[:, km]
    viscous_transport = (integral_derivative(-2.0 * nu * Ux) - nu * (Ue[:, km] + Vx[:, km]) +
                         2.0 * hp * nu * Ux[:, km])
    body_force = np.full(len(x), -gradP * eta[km])           # meanVelocityForce is +x
    component_sum = mean_transport + pressure_total + reynolds_transport + viscous_transport + body_force
    q_pressure_wall = -hp * wallq["p_wall"]
    # flux convention of da_budget.direct_wall_shear: WALL-ON-FLUID x-traction
    # (= OpenFOAM wallShearStress sign) = minus the fluid-on-wall tau_x
    q_viscous_direct = -wallq["tau_x"]
    q_wall_direct = q_pressure_wall + q_viscous_direct
    return dict(eta_m=float(eta[km]), mean_transport=mean_transport, pressure_total=pressure_total,
                reynolds_transport=reynolds_transport, viscous_transport=viscous_transport,
                body_force=body_force, component_sum=component_sum, q_wall_direct=q_wall_direct,
                q_pressure_wall=q_pressure_wall, q_viscous_direct=q_viscous_direct,
                residual=component_sum - q_wall_direct)


# ---------------------------------------------------------------- per-window analysis
def analyse_window(w, geo, gradP, hudson, maass, waves=2):
    nu = float(w["nu"])
    lam = float(w["lambda_"])
    x = w["x"]
    n = len(x)
    Lx = float(x[-1] - x[0]) + float(np.median(np.diff(x)))
    yw_mesh, _, wall_fit_resid = meshed_wall(w["y"], float(w["y_top"]), w["y_wall"])
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=Lx / n)
    w["y_wall_mesh"] = yw_mesh
    w["h_prime"] = np.fft.ifft(1j * kx * np.fft.fft(yw_mesh)).real
    wallq = wall_quantities(w)
    prof = column_profiles(w)
    # First cell-centre height above the MESHED wall.  The reduced field
    # ``dy_first_cell`` is measured from the analytic cosine, which the spline
    # wall departs from by up to ~5e-4 delta at 2a/lambda = 0.10 and ~2.5e-3 at
    # 0.20; that offset is comparable with (and on the steep wall several times
    # larger than) the cell height itself, so an analytic origin returns
    # negative "cell heights" at some stations and a spuriously large maximum.
    # The solver computes traction on the meshed faces, so the meshed surface is
    # the origin for this record as well as for the profiles.
    dy1_mesh = w["y"][:, 0] - yw_mesh
    res = dict(t_start=w.get("t_start"), t_end=w.get("t_end"),
               wall_origin_fit_residual=wall_fit_resid,
               wall_origin_dev_max=float(np.abs(yw_mesh - w["y_wall"]).max()),
               dy1_mesh_min=float(np.min(dy1_mesh)),
               dy1_mesh_median=float(np.median(dy1_mesh)),
               dy1_mesh_max=float(np.max(dy1_mesh)),
               # block windows are differences of cumulative means and carry no
               # geometry field; the analytic record is a per-case constant
               dy1_analytic_min=(float(np.min(w["dy_first_cell"]))
                                 if "dy_first_cell" in w else float("nan")),
               dy1_analytic_max=(float(np.max(w["dy_first_cell"]))
                                 if "dy_first_cell" in w else float("nan")),
               wall_origin_fit_residual_over_dy1=wall_fit_resid /
               float(np.min(w["y"][:, 0] - yw_mesh)))
    # --- wall traction phase
    seps, res_ = crossings(wallq["phase"], wallq["tau_t"], waves)
    res["x_sep_per_wave"] = seps
    res["x_re_per_wave"] = res_
    res["x_sep"] = float(np.mean(seps)) if seps.size else np.nan
    res["x_re"] = float(np.mean(res_)) if res_.size else np.nan
    res["n_sep_crossings"] = int(seps.size)
    res["f_reversed"] = float(np.mean(wallq["tau_t"] < 0))
    res["tau_mean"], res["tau_A1"], res["tau_phi1"] = first_harmonic(wallq["phase"], wallq["tau_t"])
    res["pw_mean"], res["pw_A1"], res["pw_phi1"] = first_harmonic(wallq["phase"], wallq["p_wall"])
    res["Cf_mean"] = 2 * res["tau_mean"]
    res["ustar_wavy"] = float(wallq["ustar_wavy"])
    res["ustar_top"] = float(wallq["ustar_top"])
    res["friction_drag"] = float(wallq["friction_drag"])
    res["form_drag"] = float(wallq["form_drag"])
    res["form_fraction"] = float(wallq["form_drag"] / max(wallq["total_drag"], 1e-30))
    # global momentum balance: gradP * H_mean (per unit plan area) = bottom total + top
    H = float(w["H_mean"])
    lhs = gradP * H
    rhs = wallq["total_drag"] + wallq["tau_top"]
    res["momentum_closure_rel"] = float(abs(lhs - rhs) / max(abs(lhs), 1e-30))
    res["ustar_wavy_from_gradP"] = float(np.sqrt(max(lhs - wallq["tau_top"], 0.0)))
    res["Re_tau_wavy"] = res["ustar_wavy"] / nu
    res["Re_tau_top"] = res["ustar_top"] / nu
    res["tau_t"] = wallq["tau_t"]
    res["tau_n_over_tau_t_rms"] = float(np.sqrt(np.mean(wallq["tau_n"] ** 2)) /
                                        max(np.sqrt(np.mean(wallq["tau_t"] ** 2)), 1e-30))
    res["p_wall"] = wallq["p_wall"]
    res["phase"] = wallq["phase"]
    res["y_wall_mesh"] = yw_mesh
    res["h_prime"] = w["h_prime"]
    # --- reference profile comparison
    off = REF_SCALARS["hudson_1996"]["crest_offset"]
    hud = {}
    for ph, d in hudson.items():
        Ul = station_profile(prof, ph + off, "U", waves)
        uvl = station_profile(prof, ph + off, "uv", waves)
        hud[ph] = dict(U_l2=rel_l2(d["eta"], d["U"], prof["eta"], Ul, ymax=1.0),
                       uv_l2=rel_l2(d["eta"], d["uv"], prof["eta"], uvl, ymax=1.0),
                       U_les=np.interp(d["eta"], prof["eta"], Ul), eta=d["eta"], U_ref=d["U"],
                       uv_les=np.interp(d["eta"], prof["eta"], uvl), uv_ref=d["uv"])
    res["hudson"] = hud
    res["hudson_U_l2_median"] = float(np.nanmedian([v["U_l2"] for v in hud.values()]))
    res["hudson_U_l2_max"] = float(np.nanmax([v["U_l2"] for v in hud.values()]))
    res["hudson_uv_l2_median"] = float(np.nanmedian([v["uv_l2"] for v in hud.values()]))
    maa = {}
    for ph, d in maass.items():
        Ul = station_profile(prof, ph, "U", waves)
        uvl = station_profile(prof, ph, "uv", waves)
        # Maass y is absolute: convert the LES eta grid at this phase to absolute y
        # absolute height at this phase uses the same meshed wall as the profile origin
        yw_ph = float(np.interp(ph % 1.0, np.sort(wallq["phase"]),
                                yw_mesh[np.argsort(wallq["phase"])]))
        yabs = prof["eta"] + yw_ph
        maa[ph] = dict(U_l2=rel_l2(d["y"], d["U"], yabs, Ul, ymax=1.1),
                       uv_l2=rel_l2(d["y"], d["uv"], yabs, uvl, ymax=1.1),
                       U_les=np.interp(d["y"], yabs, Ul), y=d["y"], U_ref=d["U"],
                       uv_les=np.interp(d["y"], yabs, uvl), uv_ref=d["uv"])
    res["maass"] = maa
    res["maass_U_l2_median"] = float(np.nanmedian([v["U_l2"] for v in maa.values()]))
    res["maass_U_l2_max"] = float(np.nanmax([v["U_l2"] for v in maa.values()]))
    res["maass_uv_l2_median"] = float(np.nanmedian([v["uv_l2"] for v in maa.values()]))
    # --- a-priori ODE verdict at several physical matching heights
    ode = {}
    for eta_m in ETA_MATCH_TARGETS:
        profs, km = ode_profiles(w, prof, wallq, eta_m, nu, waves)
        closures = ("standard_ml", "controlled_ml", "controlled_dns", "controlled_dns_total")
        ev = ode_evaluate(profs, nu, Y_IDX=km, closures=closures)
        p_int = np.array([p["p_int"] for p in profs])
        tau = np.array([p["tau_w"] for p in profs])
        eps_exact = np.abs(tau) / np.maximum(np.abs(p_int), 1e-30)
        sign_ok = {k: float(np.mean(np.sign(ev[k]) == np.sign(tau))) for k in closures}
        bal = wall_following_balance(w, prof, wallq, eta_m, nu, gradP)
        # cancellation fraction Pi = |mean transport| / |pressure + reynolds + viscous| (data only)
        Pi = np.abs(bal["mean_transport"]) / np.maximum(
            np.abs(bal["pressure_total"]) + np.abs(bal["reynolds_transport"]) + np.abs(bal["viscous_transport"]), 1e-30)
        ode[eta_m] = dict(
            eta_m_actual=float(prof["eta"][km]), y_idx=km,
            eta_m_plus=float(prof["eta"][km] * res["ustar_wavy"] / nu),
            **{k: float(ev[k + "_r2"]) for k in closures},
            **{k + "_relRMS": float(ev[k + "_relRMS"]) for k in closures},
            **{k + "_sign_acc": sign_ok[k] for k in closures},
            eps_median=float(ev["eps_median"]), frac_eps_lt1=float(ev["frac_eps_lt1"]),
            frac_eps_lt0p1=float(ev["frac_eps_lt0p1"]),
            eps_exact_median=float(np.median(eps_exact)),
            frac_eps_exact_lt1=float(np.mean(eps_exact < 1)),
            pi_median=float(np.median(Pi)),
            balance_residual_rel=float(np.sqrt(np.mean(bal["residual"] ** 2)) /
                                       max(np.sqrt(np.mean(bal["q_wall_direct"] ** 2)), 1e-30)),
            term_rms={k: float(np.sqrt(np.mean(bal[k] ** 2))) for k in
                      ("mean_transport", "pressure_total", "reynolds_transport",
                       "viscous_transport", "body_force", "q_wall_direct")},
            pred={k: ev[k] for k in closures}, tau_ref=tau, eps=ev["eps"], eps_exact=eps_exact,
        )
    res["ode"] = ode
    return res


def summarise_blocks(block_res, key, sub=None):
    vals = np.array([(b[key] if sub is None else b["ode"][sub][key]) for b in block_res], float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(mean=np.nan, std=np.nan, sem=np.nan, n=0, min=np.nan, max=np.nan)
    return dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
                sem=float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                n=int(vals.size), min=float(vals.min()), max=float(vals.max()))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else dict(shape=list(o.shape), summary="array in npz")
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=_dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--cases-dir", default=str(RUNS))
    ap.add_argument("--expected-end", type=float, default=200.0)
    args = ap.parse_args()
    runs = Path(args.cases_dir)
    hudson, maass = load_hudson(), load_maass()
    cross = reference_cross_l2(hudson, maass)
    print("[reference cross-check] Maass DNS vs Hudson experiment: median rel-L2(U) = %.3f "
          "(max %.3f) -- the experiment gate is anchored to this, not to an absolute number"
          % (cross["U_median"], cross["U_max"]))

    grids, absent = {}, []
    for g in ("G0", "G1", "G2"):
        cands = sorted(runs.glob("r1sta2_wavy_%s_v*" % g))
        loaded = None
        for c in cands[::-1]:
            loaded = load_grid(c)
            if loaded:
                break
        if loaded is None:
            absent.append(g)
            continue
        grids[g] = loaded
    if not grids:
        raise SystemExit("no reduced wavy WRLES data under %s -- nothing harvested (no fabrication)" % runs)

    out_json = dict(schema=SCHEMA, ledger_row=LEDGER_ROW, idea=IDEA, status="PENDING",
                    generated=_dt.datetime.now().isoformat(timespec="seconds"),
                    reference_scalars=REF_SCALARS, tolerances=TOL,
                    hudson_dns_reference_l2=cross,
                    reference_files={"hudson": {str(k): dict(file=v["file"], sha256=v["sha256"]) for k, v in hudson.items()},
                                     "maass": {str(k): dict(file=v["file"], sha256=v["sha256"]) for k, v in maass.items()}},
                    grids={}, absent_grids=absent, producer_jobs={})
    out_npz = {}
    for g, G in grids.items():
        geo, snaps, ts = G["geo"], G["snaps"], G["timeseries"]
        t0 = float(geo["avg_start"])
        final = snaps[-1]
        t_end = float(final["time"])
        converged = abs(t_end - args.expected_end) < 1e-6
        # body force averaged over the final cumulative window / each block
        def gradP_between(a, b):
            if ts is None:
                return np.nan
            m = (ts["time"] > a) & (ts["time"] <= b)
            return float(np.mean(ts["gradP"][m])) if m.any() else np.nan
        blocks = block_windows(snaps, t0)
        cum = dict(final)
        cum.update(t_start=t0, t_end=t_end, kind="cumulative")
        gP_cum = gradP_between(t0, t_end)
        R = analyse_window(cum, geo, gP_cum, hudson, maass)
        block_res = []
        for b in blocks:
            if b["t_end"] - b["t_start"] < 0.5 * BLOCK:
                continue
            block_res.append(analyse_window(b, geo, gradP_between(b["t_start"], b["t_end"]), hudson, maass))
        # per-wave spread (the two identical waves are independent realisations of phase)
        wave_spread = dict(x_sep=float(np.ptp(R["x_sep_per_wave"])) if R["x_sep_per_wave"].size == 2 else np.nan,
                           x_re=float(np.ptp(R["x_re_per_wave"])) if R["x_re_per_wave"].size == 2 else np.nan)
        unc = {k: summarise_blocks(block_res, k) for k in
               ("x_sep", "x_re", "ustar_wavy", "ustar_top", "tau_A1", "tau_phi1", "pw_A1", "pw_phi1",
                "hudson_U_l2_median", "maass_U_l2_median", "form_fraction", "f_reversed")}
        unc_ode = {str(e): {k: summarise_blocks(block_res, k, sub=e) for k in
                            ("standard_ml", "controlled_ml", "controlled_dns", "controlled_dns_total",
                             "eps_median", "frac_eps_lt1", "frac_eps_lt0p1", "pi_median")}
                   for e in ETA_MATCH_TARGETS}
        # resolution record (final window)
        yp_max = float(final["yplus_bottom_max"]) if "yplus_bottom_max" in final else np.nan
        nut_wall_max = float(final["nut_wall_bottom_max"]) if "nut_wall_bottom_max" in final else np.nan
        ustar = R["ustar_wavy"]
        lv = float(geo["nu"]) / max(ustar, 1e-30)
        resol = dict(dx_plus=float(geo["dx"]) / lv, dz_plus=float(geo["dz"]) / lv,
                     # measured from the meshed wall (see analyse_window); the
                     # analytic-origin value is retained beside it, never dropped
                     y1_plus=R["dy1_mesh_max"] / lv * 2.0,   # first-cell height (2x centre)
                     y1_plus_centre_max=R["dy1_mesh_max"] / lv,
                     y1_plus_analytic_origin=float(np.max(final["dy_first_cell"])) / lv * 2.0,
                     dy1_mesh_min_over_max=R["dy1_mesh_min"] / R["dy1_mesh_max"],
                     dy_mid_plus=float(geo["dy_mid"]) / lv, yplus_fo_max=yp_max,
                     nut_wall_max_over_nu=nut_wall_max / float(geo["nu"]),
                     dt_final=float(ts["deltaT"][-1]) if ts is not None else np.nan,
                     dt_window_mean=float(np.mean(ts["deltaT"][ts["time"] > t0])) if ts is not None else np.nan,
                     CoMax_window_max=float(np.max(ts["CoMax"][ts["time"] > t0])) if ts is not None else np.nan,
                     n_steps=int(len(ts["time"])) if ts is not None else -1,
                     averaging_window=[t0, t_end],
                     flow_throughs=(t_end - t0) / (float(geo["Lx"]) / float(geo["Ub"])),
                     eddy_turnovers=(t_end - t0) * ustar / float(geo["delta"]))
        # validation gates (final cumulative window, tolerances stated above)
        ms, ch, hu = REF_SCALARS["maass_schumann_1996"], REF_SCALARS["cherukat_1998"], REF_SCALARS["hudson_1996"]
        gates = dict(
            converged_to_end_time=bool(converged),
            wall_resolved_y1_plus=bool(resol["y1_plus"] <= TOL["yplus_max"]),
            x_sep_vs_dns=bool(abs(R["x_sep"] - 0.5 * (ms["x_sep"] + ch["x_sep"])) <= TOL["x_sep"]),
            x_re_vs_dns=bool(abs(R["x_re"] - 0.5 * (ms["x_re"] + ch["x_re"])) <= TOL["x_re"]),
            ustar_vs_reference=bool(abs(R["ustar_wavy"] - 0.5 * (ms["ustar_wavy"] + hu["ustar_wavy"])) /
                                    (0.5 * (ms["ustar_wavy"] + hu["ustar_wavy"])) <= TOL["ustar_rel"]),
            momentum_closure=bool(R["momentum_closure_rel"] <= TOL["momentum_closure_rel"]),
            maass_profiles=bool(R["maass_U_l2_median"] <= TOL["maass_l2_median"]),
            hudson_profiles_no_worse_than_dns=bool(
                R["hudson_U_l2_median"] <= TOL["hudson_dns_margin"] * cross["U_median"]),
        )
        gates["validated"] = all(gates.values())
        # failure-instance verdict: standard ODE R^2 < 0 at every matching height AND eps_med < 1
        r2s = [R["ode"][e]["standard_ml"] for e in ETA_MATCH_TARGETS]
        epss = [R["ode"][e]["eps_median"] for e in ETA_MATCH_TARGETS]
        dns_r2 = [R["ode"][e]["controlled_dns"] for e in ETA_MATCH_TARGETS]
        verdict = dict(standard_ml_r2_by_eta={str(e): v for e, v in zip(ETA_MATCH_TARGETS, r2s)},
                       eps_median_by_eta={str(e): v for e, v in zip(ETA_MATCH_TARGETS, epss)},
                       controlled_dns_r2_by_eta={str(e): v for e, v in zip(ETA_MATCH_TARGETS, dns_r2)},
                       ode_fails_all_heights=bool(np.all(np.array(r2s) < 0)),
                       ode_fails_any_height=bool(np.any(np.array(r2s) < 0)),
                       cancellation_eps_lt1_all_heights=bool(np.all(np.array(epss) < 1)),
                       exact_stress_does_not_rescue=bool(np.all(np.array(dns_r2) < 0)),
                       reversed_fraction=R["f_reversed"])
        verdict["second_failure_instance"] = bool(verdict["ode_fails_all_heights"] and
                                                  verdict["cancellation_eps_lt1_all_heights"])
        grid_json = dict(
            case_id=G["case_dir"].name, grid=g, cells=int(geo["n_cells"]), mesh=[geo["nx"], geo["ny"], geo["nz"]],
            converged=converged, t_end=t_end, n_blocks=len(block_res), resolution=resol,
            slurm_job_id=G["manifest"].get("slurm_job_id"), solver_seconds=G["manifest"].get("solver_seconds"),
            ranks=G["manifest"].get("ranks"), nodes=G["manifest"].get("nodes"),
            wall=dict(wall_origin_fit_residual=R["wall_origin_fit_residual"],
                      wall_origin_dev_max=R["wall_origin_dev_max"],
                      wall_origin_fit_residual_over_dy1=R["wall_origin_fit_residual_over_dy1"],
                      x_sep=R["x_sep"], x_re=R["x_re"], x_sep_per_wave=R["x_sep_per_wave"], x_re_per_wave=R["x_re_per_wave"],
                      n_sep_crossings=R["n_sep_crossings"], f_reversed=R["f_reversed"],
                      tau_mean=R["tau_mean"], tau_A1=R["tau_A1"], tau_phi1_deg=np.degrees(R["tau_phi1"]),
                      pw_mean=R["pw_mean"], pw_A1=R["pw_A1"], pw_phi1_deg=np.degrees(R["pw_phi1"]),
                      Cf_mean=R["Cf_mean"], ustar_wavy=R["ustar_wavy"], ustar_top=R["ustar_top"],
                      ustar_wavy_from_gradP=R["ustar_wavy_from_gradP"], Re_tau_wavy=R["Re_tau_wavy"],
                      Re_tau_top=R["Re_tau_top"], friction_drag=R["friction_drag"], form_drag=R["form_drag"],
                      form_fraction=R["form_fraction"], momentum_closure_rel=R["momentum_closure_rel"],
                      tau_n_over_tau_t_rms=R["tau_n_over_tau_t_rms"], gradP_window_mean=gP_cum),
            validation=dict(hudson_U_l2_median=R["hudson_U_l2_median"], hudson_U_l2_max=R["hudson_U_l2_max"],
                            hudson_uv_l2_median=R["hudson_uv_l2_median"],
                            hudson_U_l2_by_station={str(k): v["U_l2"] for k, v in R["hudson"].items()},
                            maass_U_l2_median=R["maass_U_l2_median"], maass_U_l2_max=R["maass_U_l2_max"],
                            maass_uv_l2_median=R["maass_uv_l2_median"],
                            maass_U_l2_by_station={str(k): v["U_l2"] for k, v in R["maass"].items()},
                            gates=gates),
            ode_diagnostic={str(e): {k: v for k, v in R["ode"][e].items() if not isinstance(v, (np.ndarray, dict))}
                            | dict(term_rms=R["ode"][e]["term_rms"]) for e in ETA_MATCH_TARGETS},
            uncertainty=dict(block_windows=unc, block_windows_ode=unc_ode, wave_to_wave=wave_spread,
                             n_blocks=len(block_res), block_length=BLOCK),
            verdict=verdict,
            source_hashes=dict(reduced={s["_file"].name: sha256(s["_file"]) for s in snaps},
                               geometry=sha256(G["case_dir"] / "GEOMETRY.json"),
                               manifest=sha256(G["case_dir"] / "MANIFEST.json") if (G["case_dir"] / "MANIFEST.json").exists() else None),
        )
        out_json["grids"][g] = grid_json
        out_json["producer_jobs"][g] = G["manifest"].get("slurm_job_id")
        # npz arrays
        out_npz["%s_phase" % g] = R["phase"]
        out_npz["%s_tau_t" % g] = R["tau_t"]
        out_npz["%s_p_wall" % g] = R["p_wall"]
        out_npz["%s_x" % g] = final["x"]
        out_npz["%s_y_wall" % g] = final["y_wall"]
        out_npz["%s_y_wall_mesh" % g] = R["y_wall_mesh"]
        out_npz["%s_h_prime" % g] = R["h_prime"]
        for e in ETA_MATCH_TARGETS:
            for k in ("standard_ml", "controlled_ml", "controlled_dns", "controlled_dns_total"):
                out_npz["%s_eta%g_pred_%s" % (g, e, k)] = R["ode"][e]["pred"][k]
            out_npz["%s_eta%g_tau_ref" % (g, e)] = R["ode"][e]["tau_ref"]
            out_npz["%s_eta%g_eps" % (g, e)] = R["ode"][e]["eps"]
            out_npz["%s_eta%g_eps_exact" % (g, e)] = R["ode"][e]["eps_exact"]
        for ph, v in R["hudson"].items():
            out_npz["%s_hudson_%02d_eta" % (g, int(round(ph * 10)))] = v["eta"]
            out_npz["%s_hudson_%02d_U_les" % (g, int(round(ph * 10)))] = v["U_les"]
            out_npz["%s_hudson_%02d_U_ref" % (g, int(round(ph * 10)))] = v["U_ref"]
            out_npz["%s_hudson_%02d_uv_les" % (g, int(round(ph * 10)))] = v["uv_les"]
            out_npz["%s_hudson_%02d_uv_ref" % (g, int(round(ph * 10)))] = v["uv_ref"]
        for j, (ph, v) in enumerate(sorted(R["maass"].items())):
            out_npz["%s_maass_%02d_y" % (g, j)] = v["y"]
            out_npz["%s_maass_%02d_U_les" % (g, j)] = v["U_les"]
            out_npz["%s_maass_%02d_U_ref" % (g, j)] = v["U_ref"]
            out_npz["%s_maass_%02d_uv_les" % (g, j)] = v["uv_les"]
            out_npz["%s_maass_%02d_uv_ref" % (g, j)] = v["uv_ref"]
        out_npz["%s_block_x_sep" % g] = np.array([b["x_sep"] for b in block_res])
        out_npz["%s_block_x_re" % g] = np.array([b["x_re"] for b in block_res])
        out_npz["%s_block_ustar" % g] = np.array([b["ustar_wavy"] for b in block_res])
        out_npz["%s_block_r2_standard_ml" % g] = np.array([[b["ode"][e]["standard_ml"] for e in ETA_MATCH_TARGETS] for b in block_res])
        out_npz["%s_block_eps_median" % g] = np.array([[b["ode"][e]["eps_median"] for e in ETA_MATCH_TARGETS] for b in block_res])
        out_npz["%s_U" % g] = final["U"]
        out_npz["%s_V" % g] = final["V"]
        out_npz["%s_P" % g] = final["P"]
        out_npz["%s_uv" % g] = final["uv"]
        out_npz["%s_uu" % g] = final["uu"]
        out_npz["%s_vv" % g] = final["vv"]
        out_npz["%s_ww" % g] = final["ww"]
        out_npz["%s_ycell" % g] = final["y"]

    # ---- grid convergence across the ladder
    order = [g for g in ("G0", "G1", "G2") if g in out_json["grids"]]
    conv = dict(grids=order, n_grids=len(order))
    if len(order) >= 2:
        def seq(fn):
            return [fn(out_json["grids"][g]) for g in order]
        h = [out_json["grids"][g]["cells"] ** (-1.0 / 3.0) for g in order]
        for name, fn in (("x_sep", lambda d: d["wall"]["x_sep"]), ("x_re", lambda d: d["wall"]["x_re"]),
                         ("ustar_wavy", lambda d: d["wall"]["ustar_wavy"]), ("tau_A1", lambda d: d["wall"]["tau_A1"]),
                         ("tau_phi1_deg", lambda d: d["wall"]["tau_phi1_deg"]), ("pw_phi1_deg", lambda d: d["wall"]["pw_phi1_deg"]),
                         ("form_fraction", lambda d: d["wall"]["form_fraction"]),
                         ("hudson_U_l2_median", lambda d: d["validation"]["hudson_U_l2_median"]),
                         ("r2_standard_ml_eta0.1", lambda d: d["ode_diagnostic"]["0.1"]["standard_ml"]),
                         ("r2_controlled_dns_eta0.1", lambda d: d["ode_diagnostic"]["0.1"]["controlled_dns"]),
                         ("eps_median_eta0.1", lambda d: d["ode_diagnostic"]["0.1"]["eps_median"])):
            v = seq(fn)
            entry = dict(values=v, h=h)
            if len(v) >= 2:
                entry["last_change"] = float(v[-1] - v[-2])
                entry["last_change_rel"] = float(abs(v[-1] - v[-2]) / max(abs(v[-1]), 1e-30))
            if len(v) == 3:
                d1, d2 = v[1] - v[0], v[2] - v[1]
                entry["monotone"] = bool(d1 * d2 > 0)
                r = np.log(h[1] / h[2]) and np.log(h[0] / h[1])
                if abs(d2) > 1e-14 and abs(d1) > 1e-14 and d1 * d2 > 0:
                    p = np.log(abs(d1 / d2)) / np.log(h[0] / h[1])    # uniform-ratio assumption
                    entry["observed_order"] = float(p)
                    rr = (h[1] / h[2]) ** p
                    entry["richardson_extrapolated"] = float(v[2] + (v[2] - v[1]) / (rr - 1.0))
                    entry["gci_fine_rel"] = float(1.25 * abs((v[2] - v[1]) / max(abs(v[2]), 1e-30)) / (rr - 1.0))
            conv[name] = entry
        # verdict invariance across the ladder
        conv["verdict_invariant"] = bool(len({out_json["grids"][g]["verdict"]["second_failure_instance"] for g in order}) == 1)
        conv["validated_on_finest"] = bool(out_json["grids"][order[-1]]["validation"]["gates"]["validated"])
    out_json["grid_convergence"] = conv
    finest = order[-1]
    fg = out_json["grids"][finest]
    out_json["finest_grid"] = finest
    out_json["headline"] = dict(
        finest_grid=finest, cells=fg["cells"], validated=fg["validation"]["gates"]["validated"],
        x_sep=fg["wall"]["x_sep"], x_re=fg["wall"]["x_re"], ustar_wavy=fg["wall"]["ustar_wavy"],
        hudson_U_l2_median=fg["validation"]["hudson_U_l2_median"], maass_U_l2_median=fg["validation"]["maass_U_l2_median"],
        standard_ml_r2_eta0p1=fg["ode_diagnostic"]["0.1"]["standard_ml"],
        controlled_dns_r2_eta0p1=fg["ode_diagnostic"]["0.1"]["controlled_dns"],
        eps_median_eta0p1=fg["ode_diagnostic"]["0.1"]["eps_median"],
        second_failure_instance=fg["verdict"]["second_failure_instance"],
        x_sep_block_sem=fg["uncertainty"]["block_windows"]["x_sep"]["sem"],
        x_re_block_sem=fg["uncertainty"]["block_windows"]["x_re"]["sem"],
        r2_standard_ml_eta0p1_block_sem=fg["uncertainty"]["block_windows_ode"]["0.1"]["standard_ml"]["sem"],
    )
    # ---- top-level summary blocks (read by the campaign poller / operator)
    out_json["validation"] = dict(
        finest_grid=finest,
        gates=fg["validation"]["gates"],
        validated=fg["validation"]["gates"]["validated"],
        per_grid={g: dict(
            cells=out_json["grids"][g]["cells"],
            hudson_U_l2_median=out_json["grids"][g]["validation"]["hudson_U_l2_median"],
            hudson_U_l2_max=out_json["grids"][g]["validation"]["hudson_U_l2_max"],
            hudson_uv_l2_median=out_json["grids"][g]["validation"]["hudson_uv_l2_median"],
            maass_U_l2_median=out_json["grids"][g]["validation"]["maass_U_l2_median"],
            maass_U_l2_max=out_json["grids"][g]["validation"]["maass_U_l2_max"],
            maass_uv_l2_median=out_json["grids"][g]["validation"]["maass_uv_l2_median"],
            x_sep=out_json["grids"][g]["wall"]["x_sep"], x_re=out_json["grids"][g]["wall"]["x_re"],
            ustar_wavy=out_json["grids"][g]["wall"]["ustar_wavy"],
            momentum_closure_rel=out_json["grids"][g]["wall"]["momentum_closure_rel"],
            y1_plus=out_json["grids"][g]["resolution"]["y1_plus"],
            dx_plus=out_json["grids"][g]["resolution"]["dx_plus"],
            flow_throughs=out_json["grids"][g]["resolution"]["flow_throughs"],
            gates_passed=sum(1 for k, v in out_json["grids"][g]["validation"]["gates"].items()
                             if k != "validated" and v),
            gates_total=len(out_json["grids"][g]["validation"]["gates"]) - 1,
        ) for g in order},
        reference_scalars=REF_SCALARS)
    out_json["ode_verdict"] = dict(
        finest_grid=finest,
        second_failure_instance=fg["verdict"]["second_failure_instance"],
        grid_robust=bool(conv.get("verdict_invariant")),
        eta_m_targets=list(ETA_MATCH_TARGETS),
        per_grid={g: dict(
            standard_ml_r2=out_json["grids"][g]["verdict"]["standard_ml_r2_by_eta"],
            controlled_dns_r2=out_json["grids"][g]["verdict"]["controlled_dns_r2_by_eta"],
            eps_median=out_json["grids"][g]["verdict"]["eps_median_by_eta"],
            second_failure_instance=out_json["grids"][g]["verdict"]["second_failure_instance"],
            reversed_fraction=out_json["grids"][g]["verdict"]["reversed_fraction"],
            r2_block_sem={str(e): out_json["grids"][g]["uncertainty"]["block_windows_ode"][str(e)]["standard_ml"]["sem"]
                          for e in ETA_MATCH_TARGETS},
        ) for g in order},
        statement=None)
    out_json["ode_verdict"]["statement"] = (
        "On the mild wavy wall (2a/lambda=0.10, lambda=2delta, Re_h=3460) the pressure-gradient ODE "
        "wall model is %s at the finest grid: R2(tau_w) = %s across eta_m/delta = %s, with "
        "epsilon_median = %s. The verdict is %s across the %d-grid ladder."
        % ("NOT a failure instance" if not fg["verdict"]["second_failure_instance"] else "a failure instance",
           {k: round(v, 3) for k, v in fg["verdict"]["standard_ml_r2_by_eta"].items()},
           list(ETA_MATCH_TARGETS),
           {k: round(v, 3) for k, v in fg["verdict"]["eps_median_by_eta"].items()},
           "invariant" if conv.get("verdict_invariant") else "NOT invariant", len(order)))

    complete = (len(order) == 3 and all(out_json["grids"][g]["converged"] for g in order))
    if complete and conv.get("validated_on_finest") and conv.get("verdict_invariant"):
        out_json["status"] = "R1_STA2_WAVY_WRLES_OK"
    elif complete:
        out_json["status"] = "R1_STA2_COMPLETE_GATES_FAILED"
    else:
        out_json["status"] = "R1_STA2_PARTIAL_%d_OF_3_GRIDS" % len(order)
    out_json["uses_modelled_eddy_viscosity_in_reference"] = False
    out_json["fidelity"] = "OpenFOAM-10 pimpleFoam wall-resolved LES (WALE), NOT DNS; numerics = deposited rib_les_dtype"

    tag = "r1_sta2_wavy_wrles_%s" % args.date
    np.savez(RESULTS / (tag + ".npz"), **out_npz)
    (RESULTS / (tag + ".json")).write_text(json.dumps(jsonable(out_json), indent=2))
    print("status:", out_json["status"])
    for g in order:
        d = out_json["grids"][g]
        print("%s cells=%d conv=%s | x_sep=%.3f x_re=%.3f u*=%.4f (ref 0.104/0.1075) closure=%.3f | "
              "Hudson U L2 med=%.3f Maass U L2 med=%.3f | y1+=%.2f dx+=%.1f | R2(std ML) eta0.1=%.3f "
              "R2(exact stress)=%.3f eps_med=%.3f | failure=%s validated=%s"
              % (g, d["cells"], d["converged"], d["wall"]["x_sep"], d["wall"]["x_re"], d["wall"]["ustar_wavy"],
                 d["wall"]["momentum_closure_rel"], d["validation"]["hudson_U_l2_median"],
                 d["validation"]["maass_U_l2_median"], d["resolution"]["y1_plus"], d["resolution"]["dx_plus"],
                 d["ode_diagnostic"]["0.1"]["standard_ml"], d["ode_diagnostic"]["0.1"]["controlled_dns"],
                 d["ode_diagnostic"]["0.1"]["eps_median"], d["verdict"]["second_failure_instance"],
                 d["validation"]["gates"]["validated"]))
    print("saved ->", RESULTS / (tag + ".json"), "and .npz")


if __name__ == "__main__":
    main()
