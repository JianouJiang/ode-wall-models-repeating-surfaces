#!/usr/bin/env python3
"""deployed_operator.py -- the wall-model operator as the solver actually applies it.

A wall model is published as a map ``S`` from matching-plane data to a wall
shear stress.  A large-eddy simulation does not apply ``S``.  It applies

        W  =  P o S o M ,

where ``M`` extracts the matching data from the discrete solution and ``P`` is
the *delivery map* of the boundary condition through which the requested stress
is imposed.  In every scalar-eddy-viscosity implementation -- which is how
essentially all wall models are coupled to an incompressible finite-volume LES,
including both boundary conditions deployed in this paper -- ``P`` is not the
identity.  This module implements ``P`` and the deployed forms of ``S`` for the
two boundary conditions used in the coupled campaign, so that the requested and
the delivered stress can be scored side by side.

Both operators are transcribed from source, not from documentation:

  * ``project_tble``  <- ``tbleVectorRealizableNut`` in the case-owned header
    ``input/wallmodel_tble/tbleShootContinuation.H`` (identical in every case;
    hash recorded by the producer).
  * ``spalding_deployed`` <- OpenFOAM-10
    ``nutUSpaldingWallFunctionFvPatchScalarField.C``, vendored verbatim under
    ``vendor_openfoam/`` and hashed by the producer.

Notation used throughout (and in the paper):

    u_m   = U_c . t_s      signed matching velocity along the wall tangent
    q     = |U_t|          tangential speed at the matching cell (TBLE arm)
    s     = |U_c|          full relative speed at the matching cell (Spalding arm)
    y_m                    wall-normal distance to the matching cell centre
    nu                     kinematic viscosity
    tau_r                  requested (model) wall stress
    tau_d                  delivered wall stress, i.e. what the LES receives
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# OpenFOAM Foundation 10 defaults for nutUSpaldingWallFunction (no override
# appears in any deposited `nut` dictionary; the producer asserts this).
KAPPA_DEPLOYED = 0.41
E_DEPLOYED = 9.8
B_DEPLOYED = math.log(E_DEPLOYED) / KAPPA_DEPLOYED       # 5.5668...
ROOT_V_SMALL = 1.0e-150                                   # OpenFOAM rootVSmall
SPALDING_MAX_ITER = 10                                    # deployed iteration cap
SPALDING_TOL = 0.01                                       # deployed relative tol


# ---------------------------------------------------------------------------
# P : the delivery map of the total-gradient TBLE boundary condition
# ---------------------------------------------------------------------------
@dataclass
class Delivery:
    """Result of applying the boundary-condition delivery map."""

    raw_nut: np.ndarray
    upper_nut: np.ndarray
    nut: np.ndarray
    tau_delivered: np.ndarray
    traction_magnitude: np.ndarray
    lower_clipped: np.ndarray      # the requested stress opposed the flow
    vector_capped: np.ndarray      # realizability cap on the traction magnitude
    regime: np.ndarray             # 0 faithful, 1 alignment-capped, 2 sign-refused


def project_tble(tau_r, u_m, q, y_m, nu) -> Delivery:
    """Vectorised transcription of ``tbleVectorRealizableNut``.

    The boundary condition cannot impose a traction directly: it may only
    choose a non-negative scalar eddy viscosity ``nu_t``, after which the
    solver applies ``(nu + nu_t) u_m / y_m``.  Two guards act:

      lower guard   nu_t >= 0            (no negative total viscosity)
      upper guard   nu_t <= |tau_r| y_m / q - nu    (traction magnitude may not
                                                     exceed the requested one)
    """
    tau_r = np.asarray(tau_r, float)
    u_m = np.asarray(u_m, float)
    q = np.asarray(q, float)
    y_m = np.asarray(y_m, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(np.abs(u_m) > 1.0e-14, tau_r * y_m / u_m - nu, -np.inf)
    speed = np.maximum(q, 1.0e-14)
    upper = np.maximum(np.abs(tau_r) * y_m / speed - nu, 0.0)
    lower = np.maximum(raw, 0.0)
    nut = np.minimum(lower, upper)
    lower_clipped = raw < 0.0
    vector_capped = lower > upper
    tau_d = (nu + nut) * u_m / y_m
    regime = np.where(lower_clipped, 2, np.where(vector_capped, 1, 0))
    return Delivery(raw, upper, nut, tau_d, (nu + nut) * speed / y_m,
                    lower_clipped, vector_capped, regime)


# ---------------------------------------------------------------------------
# S and P for the stock equilibrium (Spalding) wall function
# ---------------------------------------------------------------------------
def _spalding_yplus(u_plus, kappa=KAPPA_DEPLOYED, b=B_DEPLOYED):
    ku = kappa * np.asarray(u_plus, float)
    return u_plus + math.exp(-kappa * b) * (
        np.exp(np.minimum(ku, 50.0)) - 1.0 - ku - 0.5 * ku * ku - ku ** 3 / 6.0)


def spalding_utau_converged(speed, y, nu, kappa=KAPPA_DEPLOYED, b=B_DEPLOYED):
    """Converged Spalding friction velocity (bisection, machine tolerance).

    This is the wall model *as published*: the exact root of Spalding's law.
    """
    speed = np.asarray(speed, float)
    out = np.zeros_like(speed)
    for i, sp in enumerate(np.atleast_1d(speed).ravel()):
        if not np.isfinite(sp) or sp <= 0.0:
            continue
        yi = float(np.atleast_1d(y).ravel()[i] if np.ndim(y) else y)

        def resid(ut):
            return yi * ut / nu - _spalding_yplus(sp / ut, kappa, b)

        lo, hi = max(np.finfo(float).tiny, sp * 1e-12), max(sp, nu / yi) * 10.0
        n = 0
        while resid(hi) <= 0.0 and n < 200:
            hi *= 2.0
            n += 1
        if resid(hi) <= 0.0:
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1e-15 * max(hi, 1.0):
                break
        out.ravel()[i] = 0.5 * (lo + hi)
    return out


def spalding_utau_deployed(speed, y, nu, nut_seed,
                           kappa=KAPPA_DEPLOYED, e=E_DEPLOYED):
    """The deployed Newton iteration of ``nutUSpaldingWallFunction::calcUTau``.

    Transcribed line for line, including the seed from the *previous* eddy
    viscosity, the 1 % relative tolerance and the ten-iteration cap.  It is
    therefore a state-dependent, deliberately loosely converged solve, not the
    exact root returned by :func:`spalding_utau_converged`.
    """
    speed = np.atleast_1d(np.asarray(speed, float))
    y = np.atleast_1d(np.asarray(y, float)) * np.ones_like(speed)
    nut_seed = np.atleast_1d(np.asarray(nut_seed, float)) * np.ones_like(speed)
    mag_grad_u = speed / y
    u_tau = np.zeros_like(speed)
    iters = np.zeros_like(speed, dtype=int)
    for i in range(speed.size):
        ut = math.sqrt(max(nut_seed[i] + nu, 0.0) * mag_grad_u[i])
        it = 0
        if ut > ROOT_V_SMALL:
            err = np.inf
            while ut > ROOT_V_SMALL and err > SPALDING_TOL and it < SPALDING_MAX_ITER:
                k_uu = min(kappa * speed[i] / ut, 50.0)
                f_kuu = math.exp(k_uu) - 1.0 - k_uu * (1.0 + 0.5 * k_uu)
                f = (-ut * y[i] / nu + speed[i] / ut
                     + (f_kuu - k_uu ** 3 / 6.0) / e)
                df = y[i] / nu + speed[i] / ut ** 2 + k_uu * f_kuu / (e * ut)
                ut_new = ut + f / df
                err = abs((ut - ut_new) / ut) if ut != 0.0 else np.inf
                ut = ut_new
                it += 1
            u_tau[i] = max(0.0, ut)
        iters[i] = it
    return u_tau, iters


def project_spalding(u_tau, u_m, s, y_m, nu) -> Delivery:
    """Delivery map of ``nutUSpaldingWallFunction``.

    ``nut = max(0, u_tau^2/magGradU - nu)`` with ``magGradU = s/y_m``; the
    solver then applies ``(nu + nut) u_m / y_m``.  The delivered traction is
    therefore always aligned with the local near-wall velocity: this boundary
    condition has no mechanism at all for imposing a stress that opposes it.
    """
    u_tau = np.asarray(u_tau, float)
    u_m = np.asarray(u_m, float)
    s = np.asarray(s, float)
    y_m = np.asarray(y_m, float)
    mag_grad_u = s / y_m
    nut = np.maximum(u_tau ** 2 / (mag_grad_u + ROOT_V_SMALL) - nu, 0.0)
    tau_d = (nu + nut) * u_m / y_m
    lower_clipped = (u_tau ** 2 / (mag_grad_u + ROOT_V_SMALL) - nu) < 0.0
    # the requested magnitude is u_tau^2; it is delivered scaled by u_m/s
    regime = np.where(lower_clipped, 2,
                      np.where(np.abs(np.abs(u_m) - s) > 1e-12 * np.maximum(s, 1e-30),
                               1, 0))
    return Delivery(u_tau ** 2 / (mag_grad_u + ROOT_V_SMALL) - nu,
                    np.full_like(nut, np.inf), nut, tau_d,
                    (nu + nut) * s / y_m, lower_clipped,
                    np.zeros_like(nut, dtype=bool), regime)


# ---------------------------------------------------------------------------
# Analytic statements that the producer checks numerically
# ---------------------------------------------------------------------------
def faithful_mask(tau_r, u_m, q, y_m, nu, rtol=1.0e-12):
    """Exact condition under which the TBLE delivery map is the identity.

    ``tau_d = u_m max(|tau_r|/q 1[tau_r u_m > 0], nu/y_m)``, so ``tau_d ==
    tau_r`` requires the request to agree in sign with ``u_m`` and then either

      * the projected branch to be selected and to project to itself:
        ``q == |u_m|`` with ``|tau_r| >= nu |u_m| / y_m``; or
      * the molecular branch to be selected and to coincide with the request:
        ``|tau_r| == nu |u_m| / y_m`` exactly.

    The second branch was omitted from the first version of this function and
    of the corresponding corollary; both are corrected here.  It is measure
    zero in floating point but it is not empty, and a manufactured case for it
    is carried in ``verify_as_deployed.py``.
    """
    tau_r = np.asarray(tau_r, float)
    u_m = np.asarray(u_m, float)
    q = np.asarray(q, float)
    floor = nu * np.abs(u_m) / np.asarray(y_m, float)
    sign_ok = tau_r * u_m > 0.0
    strong_enough = np.abs(tau_r) >= floor
    aligned = np.isclose(q, np.abs(u_m), rtol=rtol, atol=0.0)
    on_floor = np.isclose(np.abs(tau_r), floor, rtol=rtol, atol=0.0)
    return sign_ok & ((strong_enough & aligned) | on_floor)


def contraction_bound(tau_r, u_m, q, y_m, nu):
    """Upper bound on the delivered traction proved in the paper.

    ``|tau_d| <= max(|tau_r| * |u_m|/q , nu |u_m| / y_m)``.  Only the first
    branch is a contraction; on the second the delivered stress is the laminar
    Couette stress of the matching cell, which EXCEEDS a weaker request.  This
    is an envelope, not a proof that the map never amplifies.
    """
    tau_r = np.asarray(tau_r, float)
    u_m = np.asarray(u_m, float)
    q = np.maximum(np.asarray(q, float), 1.0e-14)
    return np.maximum(np.abs(tau_r) * np.abs(u_m) / q,
                      nu * np.abs(u_m) / np.asarray(y_m, float))
