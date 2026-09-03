#!/usr/bin/env python3
"""L0: THE SOURCE NORM AS A DESIGN CONSTRAINT, TESTED AGAINST FAITHFUL BASELINES.

What this replaces
------------------
The previous head-to-head comparison was refused, unanimously and correctly,
because two of the published architectures it ranked were substitutes:

  * the integral wall model was evaluated as ``tau(y_m) - impulse`` with the
    matching-plane traction AND the exact momentum flux both supplied, which is
    the closure-free momentum identity and therefore identical to it by
    construction; and
  * the non-equilibrium wall-layer model was advanced with only the dynamic
    CORRECTION to its eddy viscosity, for a fixed 60 iterations, with no
    convergence state retained (an independent replay found 0 of 8 sampled
    stations converged, and the returned traction moved by 10-26% when the
    iteration was continued).

It also fitted its "out-of-sample" law on arms that were in the test set, and
read a wide interval containing zero as evidence of no effect.

This producer answers the same scientific question with

  1. faithful operators (``codes/models/faithful_wall_models_l0.py``): the
     integral model solved as its own coupled profile/moment system over the
     whole periodic wall, the non-equilibrium wall layer marched to a declared
     steady state on its FULL eddy viscosity and failing closed otherwise, and
     the parametrised-convection ODE reported with its published damping
     constant.  Every operator passes an analytic reduction benchmark before it
     is allowed to score.
  2. a positive candidate rather than a diagnosis.  The norm-limited wall model
     caps the assembled source magnitude at ``N* = c |U_m|`` -- one dimensionless
     constant, calibrated once on a surface that is not the primary one, then
     frozen.  The registered question is whether it BEATS the best faithful
     published family.
  3. the decisive matched-norm contrast that separates "exact content helps once
     its magnitude is controlled" from "incompleteness is intrinsically good":
     the exact source rescaled to the modelled source's norm, and the modelled
     source rescaled up to the exact source's norm.
  4. statistics that can carry a negative: a norm law fitted ONLY on rescaled
     copies and tested on arms never used in the fit, with a zero-parameter null
     computed from the TRAINING arms; a norm-PRESERVING phase permutation; a
     two-one-sided-test equivalence margin declared in advance; and an explicit
     interaction contrast instead of comparing two significance verdicts.

Truth protocol is the campaign's: A = withdrawn four-point estimator (NEGATIVE
CONTROL ONLY), B = Peller & Manhart full-wall DNS (PRIMARY), C = repaired
same-simulation cubic (BRACKET), K = Krank stations (sparse cross-check).  A
contrast counts only when its paired phase-block interval excludes zero with the
same sign under BOTH B and C.

No new simulation, no remote job, read-only on every input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "models"))
import r2m4_ladder_common as C  # noqa: E402
import conditioning_ladder_l0 as CL  # noqa: E402
import source_faithful_wall_models as wm  # noqa: E402
import faithful_wall_models_l0 as fw  # noqa: E402

STAMP = "20260825"
N_QUAD = 400
N_SHOOT = 200
N_PDE = 65
TINY = 1.0e-30

SCALE_FACTORS = (0.5, 2.0)
SCALE_BASES = ("M1_pressure_gradient", "M2_hickel", "Xall")
SHIFT_FRACTIONS = (0.125, 0.25, 0.5)
SHIFT_BASES = ("M2_hickel", "Xall")

# The candidate's single constant is searched over this grid on the CALIBRATION
# surface only, and is then frozen for every other evaluation in the study.
C_NORM_GRID_FULL = tuple(float(v) for v in np.geomspace(2.0e-5, 2.0e-1, 25))
CALIBRATION_SURFACE = "ladder_L1"
PRIMARY_SURFACE = "archive_index10"

# Declared in advance: the smallest difference in relative-RMS wall-traction
# error that this study is prepared to call a difference.  Used for the
# two-one-sided-test equivalence verdicts, so that "the interval contains zero"
# can never be reported as "no effect" without the interval also being narrow
# enough to exclude an effect of this size.
EQUIVALENCE_MARGIN = 0.25

PREREGISTERED = {
    "Q1_candidate_beats_the_best_faithful_family": (
        "The norm-limited wall model, with its one constant calibrated on the "
        "coupled-mesh surface and frozen, has a LOWER relative-RMS wall-traction "
        "error on the primary surface than the best faithfully implemented "
        "published family, with the paired 95% phase-block interval excluding "
        "zero under BOTH corrected references. WIN / NO_ADVANTAGE / REFUTED."),
    "Q2_matched_norm_separates_content_from_magnitude": (
        "The exact source rescaled to the modelled source's assembled norm is "
        "compared with the modelled source at that same norm. "
        "CONTENT_HELPS_AT_MATCHED_NORM if the exact-content arm is identifiably "
        "better; CONTENT_HURTS_AT_MATCHED_NORM if identifiably worse; "
        "CONTENT_IRRELEVANT_AT_MATCHED_NORM only if the paired interval is "
        "inside +-EQUIVALENCE_MARGIN under both references; UNRESOLVED "
        "otherwise. A wide interval containing zero is UNRESOLVED, not a null "
        "result."),
    "Q3_norm_law_without_leakage": (
        "Fitting E_abs = E0 + delta*N on RESCALED COPIES ONLY (each base arm's "
        "physics frozen, amplitude moved) and predicting the arms never used in "
        "the fit gives a median relative prediction error below 0.5, beating a "
        "zero-parameter null that assigns every test arm the MEAN TRAINING "
        "error. Arms without a wall-normal assembled source (the algebraic and "
        "integral families) are excluded and reported as excluded."),
    "Q4_phase_permutation_is_norm_preserving": (
        "A phase permutation that renormalises the displaced source back to the "
        "station's own assembled norm changes only the physical correspondence. "
        "Reported as an INTERACTION contrast, [E(shift modelled) - E(modelled)] "
        "minus [E(shift exact) - E(exact)], with its own paired interval, plus "
        "a two-one-sided-test verdict for each arm."),
    "Q6_shape_versus_amplitude_factorial": (
        "An exact two-by-two factorial crosses source SHAPE (the parametrised "
        "surrogate against the measured transport, each reduced to unit "
        "assembled norm) with source AMPLITUDE (the assembled norm of each). "
        "If the assembled norm were the controlling variable the two shape "
        "contrasts would be null and the two amplitude contrasts would carry the "
        "whole effect. Both main effects and the interaction are reported with "
        "paired intervals under both corrected references."),
    "Q5_architecture_is_not_a_source_label": (
        "Two arms with the same assembled norm but different operator "
        "architecture are expected to differ. Reported as the measured spread of "
        "error at fixed norm across architectures, against the error range the "
        "amplitude sweep spans within one architecture; the norm law is claimed "
        "ONLY within the shooting-operator family."),
}


# --------------------------------------------------------------------------- #
def arc_length_of(fields, phases):
    """Arc length along the wall at the requested phases, and the period."""
    x = fields.x
    ds_dx = np.sqrt(1.0 + fields.slope ** 2)
    s_full = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1])
                                              * np.diff(x))))
    period = float(s_full[-1] + 0.5 * (ds_dx[-1] + ds_dx[0]) * fields.dx)
    ph_full = np.mod((x - x.min()) / C.LX, 1.0)
    order = np.argsort(ph_full)
    s_of_phase = np.interp(np.mod(np.asarray(phases, float), 1.0),
                           ph_full[order], s_full[order])
    return s_of_phase, period


def prepass(fields, phases, y_m_of_phase, log=print):
    """Per-station geometry, model inputs and exact term profiles."""
    n_st = len(phases)
    xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
    keys = ("dpds", "conv", "dRtt", "visc")
    terms = {k: np.zeros((n_st, N_QUAD)) for k in keys}
    st = {k: np.zeros(n_st) for k in
          ("u_m", "dpds_wall", "tau0", "y_m", "G", "tau_at_ym", "index",
           "pressure_impulse", "arc")}
    D_all = np.zeros((n_st, N_QUAD))
    n_all = np.zeros((n_st, N_QUAD))
    pde_conv = np.zeros((n_st, N_PDE))
    pde_pres = np.zeros((n_st, N_PDE))
    pde_dyn = np.zeros((n_st, N_PDE))
    pde_n = np.zeros((n_st, N_PDE))
    x_targets = np.mod(phases, 1.0) * C.LX
    arc, period = arc_length_of(fields, phases)
    t0 = time.time()
    for p, (xt, y_m) in enumerate(zip(x_targets, y_m_of_phase)):
        i = int(np.argmin(np.abs(fields.x - xt)))
        u_m, _, _ = fields.station(i, y_m)
        tau0 = wm.spalding_wall_stress(u_m, y_m, C.NU) if abs(u_m) > 1e-12 else 0.0
        n_grid = y_m * xi
        for k in keys:
            terms[k][p] = np.asarray(fields.profile_of(k, i)(n_grid), float)
        D = fw.equilibrium_diffusivity(n_grid, tau0, C.NU)
        st["u_m"][p] = u_m
        st["dpds_wall"][p] = float(fields.dpds_total[i])
        st["tau0"][p] = tau0
        st["y_m"][p] = y_m
        st["index"][p] = i
        st["arc"][p] = arc[p]
        st["G"][p] = float(np.trapezoid(1.0 / D, n_grid))
        st["tau_at_ym"][p] = float(fields.profile_of("tau", i)([y_m])[0])
        st["pressure_impulse"][p] = float(np.trapezoid(terms["dpds"][p], n_grid))
        D_all[p] = D
        n_all[p] = n_grid
        # uniform grid for the wall-layer PDE, and its dynamic eddy-viscosity
        # correction built from the reference Reynolds stress and mean strain
        n_uniform = np.linspace(0.0, y_m, N_PDE)
        pde_n[p] = n_uniform
        pde_conv[p] = np.interp(n_uniform, n_grid, terms["conv"][p])
        pde_pres[p] = np.interp(n_uniform, n_grid, terms["dpds"][p])
        pts = fields._normal_points(i, n_uniform)
        Ux, Uy, Vx, Vy, Ruu, Rvv, Ruv = [f(pts) for f in fields._interp("tau")]
        divergence = 0.5 * (Ux + Vy)
        Sxx, Syy, Sxy = Ux - divergence, Vy - divergence, 0.5 * (Uy + Vx)
        numerator = Ruu * Sxx + Rvv * Syy + 2.0 * Ruv * Sxy
        denominator = 2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = np.where(denominator > TINY,
                                  numerator / np.maximum(denominator, TINY), 0.0)
        pde_dyn[p] = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"  prepass {time.time() - t0:.0f}s over {n_st} stations")
    return (st, terms, D_all, n_all, dict(n=pde_n, conv=pde_conv, pres=pde_pres,
                                          dyn=pde_dyn), period)


# --------------------------------------------------------------------------- #
def build_sources(st, terms, n_all, p, n_st, matched_scales, permutation):
    """Source VALUE ARRAYS on this station's quadrature grid, one per arm."""
    n_grid = n_all[p]
    dpds_const = np.full(N_QUAD, st["dpds_wall"][p])
    exact = {k: terms[k][p] for k in ("dpds", "conv", "dRtt", "visc")}
    s = {}
    s["M1_pressure_gradient"] = dpds_const
    s["M2_hickel"] = wm.hickel_source(n_grid, float(st["dpds_wall"][p]), C.NU)
    s["M2_hickel_Aplus26_variant"] = s["M2_hickel"]
    s["Xc_exact_convection"] = dpds_const + exact["conv"]
    s["Xcp_pressure_plus_convection"] = exact["dpds"] + exact["conv"]
    s["Xcpr_plus_normal_stress"] = exact["dpds"] + exact["conv"] + exact["dRtt"]
    s["Xall"] = (exact["dpds"] + exact["conv"] + exact["dRtt"] + exact["visc"])
    # decisive matched-norm contrast: exact content at the modelled norm and
    # modelled content at the exact norm, one global constant each
    s["CTL_exact_at_modelled_norm"] = matched_scales["exact_to_modelled"] * s["Xall"]
    s["CTL_modelled_at_exact_norm"] = matched_scales["modelled_to_exact"] * s["M2_hickel"]
    # amplitude sweep: the ONLY arms the norm law is fitted on
    for base in SCALE_BASES:
        for c in SCALE_FACTORS:
            s[f"FIT_scale_{base}_{c:g}"] = c * s[base]
    # norm-preserving phase permutation
    for frac in SHIFT_FRACTIONS:
        q = permutation[frac][p]
        for base in SHIFT_BASES:
            if base == "Xall":
                vals = (terms["dpds"][q] + terms["conv"][q]
                        + terms["dRtt"][q] + terms["visc"][q])
            else:
                vals = wm.hickel_source(n_grid, float(st["dpds_wall"][q]), C.NU)
            s[f"CTL_shift_{base}_{frac:g}"] = vals
    return s


NORMLESS_ARMS = ("M3_yang_integral", "M5_meneveau")
SHOOT_A17 = {"M2_hickel"}          # the published Hickel damping constant


def evaluate_surface(fields, phases, y_m_of_phase, c_norm_values, log=print):
    n_st = len(phases)
    st, terms, D_all, n_all, pde, period = prepass(fields, phases,
                                                   y_m_of_phase, log=log)

    def rms_norm(vals_of_p):
        acc = np.empty(n_st)
        for p in range(n_st):
            acc[p], _ = fw.assembled_source_norm(n_all[p], D_all[p],
                                                 st["G"][p], vals_of_p(p))
        return float(np.sqrt(np.mean(acc ** 2)))

    exact_norm = rms_norm(lambda p: (terms["dpds"][p] + terms["conv"][p]
                                     + terms["dRtt"][p] + terms["visc"][p]))
    modelled_norm = rms_norm(lambda p: wm.hickel_source(
        n_all[p], float(st["dpds_wall"][p]), C.NU))
    matched_scales = {
        "exact_to_modelled": float(modelled_norm / max(exact_norm, TINY)),
        "modelled_to_exact": float(exact_norm / max(modelled_norm, TINY)),
    }
    log(f"  exact norm {exact_norm:.4e}, modelled norm {modelled_norm:.4e}, "
        f"scale {matched_scales['exact_to_modelled']:.4f}")

    # norm-preserving permutation index: a rigid displacement in station index
    permutation = {frac: (np.arange(n_st) + int(round(frac * n_st))) % n_st
                   for frac in SHIFT_FRACTIONS}

    probe = build_sources(st, terms, n_all, 0, n_st, matched_scales, permutation)
    arm_names = list(probe.keys())
    candidate_names = ([f"NLWM_Xall_c{v:.3e}" for v in c_norm_values]
                       + [f"NLWM_M1_c{v:.3e}" for v in c_norm_values]
                       + [f"NLWH_Xall_c{v:.3e}" for v in c_norm_values]
                       + [f"NLWH_M1_c{v:.3e}" for v in c_norm_values])
    extra = ["M0_equilibrium", "M3_yang_integral", "M4_park_moin",
             "M5_meneveau", "ORACLE_closure_free",
             "FAC_exactshape_modelnorm", "FAC_modelshape_exactnorm"]
    every = arm_names + extra + candidate_names
    pred = {a: np.full(n_st, np.nan) for a in every}
    norm = {a: np.full(n_st, np.nan) for a in every}
    work = {a: np.full(n_st, np.nan) for a in every}
    limiter_active = {a: np.zeros(n_st) for a in candidate_names}
    horizon_fraction = {a: np.full(n_st, np.nan) for a in candidate_names
                        if a.startswith("NLWH_")}
    pm_iterations = np.zeros(n_st)
    pm_converged = np.zeros(n_st)
    m5_status: list[str] = []

    t0 = time.time()
    for p in range(n_st):
        srcs = build_sources(st, terms, n_all, p, n_st, matched_scales, permutation)
        n_grid, D, G = n_all[p], D_all[p], float(st["G"][p])
        u_m, y_m, tau0 = float(st["u_m"][p]), float(st["y_m"][p]), float(st["tau0"][p])
        pred["M0_equilibrium"][p] = tau0
        norm["M0_equilibrium"][p] = 0.0
        work["M0_equilibrium"][p] = 0.0
        exact_sum = (terms["dpds"][p] + terms["conv"][p]
                     + terms["dRtt"][p] + terms["visc"][p])
        impulse = float(np.trapezoid(exact_sum, n_grid))
        # ORACLE: the closure-free momentum identity.  Labelled as an oracle
        # control, NOT as a published model, because it receives the exact
        # matching-plane traction AND the exact source.
        pred["ORACLE_closure_free"][p] = float(st["tau_at_ym"][p]) - impulse
        norm["ORACLE_closure_free"][p], work["ORACLE_closure_free"][p] = \
            fw.assembled_source_norm(n_grid, D, G, exact_sum)

        # --- faithful non-equilibrium wall layer, converged or fail-closed ---
        pm = fw.park_moin_wall_stress(
            u_m, y_m, C.NU, pde["n"][p], pde["conv"][p], pde["pres"][p],
            dynamic_correction=pde["dyn"][p], tolerance=1.0e-10,
            max_iterations=20000)
        pred["M4_park_moin"][p] = pm.tau_w
        pm_iterations[p] = pm.iterations
        pm_converged[p] = float(pm.converged)
        norm["M4_park_moin"][p], work["M4_park_moin"][p] = \
            fw.assembled_source_norm(n_grid, D, G,
                                     terms["dpds"][p] + terms["conv"][p])

        # --- generalised-Moody algebraic family ------------------------------
        if abs(u_m) < 1e-12:
            pred["M5_meneveau"][p], status = 0.0, "zero_matching_velocity"
        else:
            re_delta = abs(u_m) * y_m / C.NU
            psi_p = np.sign(u_m) * float(st["dpds_wall"][p]) * y_m ** 3 / C.NU ** 2
            if not (0.0 < re_delta < 1.0e7) or not abs(psi_p) < 2.0e7:
                pred["M5_meneveau"][p], status = np.nan, "outside_published_fit_domain"
            else:
                re_tau = wm.meneveau_pressure_re_tau(re_delta, psi_p)
                u_tau = abs(u_m) * re_tau / re_delta
                pred["M5_meneveau"][p] = float(np.sign(u_m) * u_tau * u_tau)
                status = "ok"
        m5_status.append(status)

        # --- shooting-operator family ----------------------------------------
        for a, vals in srcs.items():
            norm[a][p], work[a][p] = fw.assembled_source_norm(n_grid, D, G, vals)
            src = (lambda v: (lambda y: np.interp(np.asarray(y, float), n_grid, v)))(vals)
            a_plus = wm.HICKEL_VAN_DRIEST_A if a in SHOOT_A17 else wm.VAN_DRIEST_A
            res = wm.shoot_wall_stress(u_m, y_m, C.NU, src, continuation_tau=tau0,
                                       n_points=N_SHOOT, a_plus=a_plus)
            pred[a][p] = res.tau_w
        # exact 2x2 factorial: SHAPE (modelled surrogate / measured transport)
        # crossed with AMPLITUDE (the assembled norm of each).  Because the norm
        # is homogeneous of degree one, rescaling a source by the ratio of norms
        # puts it at the other level's amplitude EXACTLY, at every station.
        n_model, n_exact = norm["M2_hickel"][p], norm["Xall"][p]
        if n_model > TINY and n_exact > TINY:
            for a, vals in (("FAC_exactshape_modelnorm",
                             exact_sum * (n_model / n_exact)),
                            ("FAC_modelshape_exactnorm",
                             srcs["M2_hickel"] * (n_exact / n_model))):
                norm[a][p], work[a][p] = fw.assembled_source_norm(n_grid, D, G, vals)
                src = (lambda v: (lambda y: np.interp(np.asarray(y, float),
                                                      n_grid, v)))(vals)
                a_plus = (wm.HICKEL_VAN_DRIEST_A if a.startswith("FAC_modelshape")
                          else wm.VAN_DRIEST_A)
                pred[a][p] = wm.shoot_wall_stress(
                    u_m, y_m, C.NU, src, continuation_tau=tau0,
                    n_points=N_SHOOT, a_plus=a_plus).tau_w

        # renormalise the permuted arms back to the station's own norm, so the
        # intervention moves ONLY the physical correspondence
        for frac in SHIFT_FRACTIONS:
            for base in SHIFT_BASES:
                a = f"CTL_shift_{base}_{frac:g}"
                target = norm[base][p]
                if norm[a][p] > TINY and target > TINY:
                    vals = srcs[a] * (target / norm[a][p])
                    norm[a][p], work[a][p] = fw.assembled_source_norm(
                        n_grid, D, G, vals)
                    src = (lambda v: (lambda y: np.interp(np.asarray(y, float),
                                                          n_grid, v)))(vals)
                    res = wm.shoot_wall_stress(u_m, y_m, C.NU, src,
                                               continuation_tau=tau0,
                                               n_points=N_SHOOT)
                    pred[a][p] = res.tau_w

        # --- the candidate, over the calibration grid ------------------------
        for value in c_norm_values:
            for tag, vals in (("Xall", exact_sum),
                              ("M1", np.full(N_QUAD, st["dpds_wall"][p]))):
                a = f"NLWM_{tag}_c{value:.3e}"
                out = fw.norm_limited_wall_stress(u_m, y_m, C.NU, n_grid, vals,
                                                  c_norm=value, n_points=N_SHOOT)
                pred[a][p] = out.tau_w
                norm[a][p] = out.norm_after
                limiter_active[a][p] = float(out.limiter_active)
                b = f"NLWH_{tag}_c{value:.3e}"
                hz = fw.norm_horizon_wall_stress(u_m, y_m, C.NU, n_grid, vals,
                                                 c_norm=value, n_points=N_SHOOT)
                pred[b][p] = hz.tau_w
                norm[b][p] = hz.norm_after
                limiter_active[b][p] = float(hz.limiter_active)
                horizon_fraction[b][p] = hz.horizon_fraction
        if p % 32 == 0:
            done = p + 1
            rate = (time.time() - t0) / done
            log(f"  station {done}/{n_st}  {rate * (n_st - done):.0f}s remaining")
    log(f"  station loop {time.time() - t0:.0f}s")

    # --- faithful integral model: one coupled solve over the whole wall ------
    t0 = time.time()
    yang = fw.yang_integral_wall_stress_field(
        st["u_m"], st["y_m"], C.NU, st["tau_at_ym"], st["pressure_impulse"],
        st["arc"], period, tolerance=1.0e-9, max_iterations=20000,
        relaxation=0.15)
    pred["M3_yang_integral"] = yang.tau_w
    log(f"  integral model {time.time() - t0:.0f}s status={yang.status} "
        f"iters={yang.iterations} residual={yang.residual:.2e}")

    diagnostics = {
        "park_moin": {
            "converged_stations": int(pm_converged.sum()),
            "stations": int(n_st),
            "median_iterations": float(np.median(pm_iterations)),
            "max_iterations": float(np.max(pm_iterations)),
            "all_converged": bool(pm_converged.all()),
        },
        "yang_integral": {
            "status": yang.status, "iterations": int(yang.iterations),
            "residual": float(yang.residual),
            "matching_residual": float(yang.matching_residual),
            "converged": bool(yang.converged),
            "transport_load_reached": float(yang.load_reached),
            "continuation_history": [list(h) for h in yang.history],
            "note": ("the coupled profile/moment system is loaded in "
                     "continuously from the equilibrium limit; a load below 1 "
                     "means the model has NO self-consistent solution at the "
                     "measured transport, and the scored arm is its last "
                     "self-consistent state -- the most favourable evaluation "
                     "available to that family"),
        },
        "meneveau_status_counts": {s: int(m5_status.count(s)) for s in set(m5_status)},
        "matched_norm_scales": matched_scales,
        "exact_source_norm_rms": exact_norm,
        "modelled_source_norm_rms": modelled_norm,
        "limiter_active_fraction": {a: float(np.mean(v))
                                    for a, v in limiter_active.items()},
        "horizon_fraction_median": {a: float(np.nanmedian(v))
                                    for a, v in horizon_fraction.items()},
    }
    return st, pred, norm, work, diagnostics


# --------------------------------------------------------------------------- #
def score(phases, pred, ref_phase, ref_tau, arms):
    dense = np.arange(C.DENSE_N) / C.DENSE_N
    truth = C.periodic_interp(ref_phase, ref_tau, dense)
    preds_dense, metrics = {}, {}
    for a in arms:
        v = pred[a]
        ok = np.isfinite(v)
        if ok.sum() < 8:
            continue
        p_d = C.periodic_interp(np.asarray(phases)[ok], v[ok], dense)
        preds_dense[a] = p_d
        err = p_d - truth
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        metrics[a] = {
            "relative_rms": float(np.sqrt(np.mean(err ** 2))
                                  / np.sqrt(np.mean(truth ** 2))),
            "absolute_rms": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
            "sign_accuracy": float(np.mean(np.sign(p_d) == np.sign(truth))),
            "finite_stations": int(ok.sum()),
        }
    return truth, preds_dense, metrics


def tost(interval: dict, margin: float) -> str:
    """Two-one-sided-test verdict from a paired interval."""
    if interval["low"] > 0.0 or interval["high"] < 0.0:
        return "DIFFERENT"
    if interval["low"] > -margin and interval["high"] < margin:
        return "EQUIVALENT_WITHIN_MARGIN"
    return "UNRESOLVED_UNDERPOWERED"


def affine_fit(N, E):
    N = np.asarray(N, float)
    E = np.asarray(E, float)
    A = np.vstack([np.ones_like(N), N]).T
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean((E - A @ coef) ** 2)))


def run_surface(fields, name, phases, y_m_of_phase, note, c_norm_values,
                refs, log=print):
    log(f"surface {name}: {len(phases)} stations, {note}")
    st, pred, norm, work, diagnostics = evaluate_surface(
        fields, phases, y_m_of_phase, c_norm_values, log=log)
    arms = [a for a in pred if np.isfinite(pred[a]).any()]
    scores, intervals, truths = {}, {}, {}
    for rname, (rp, rt) in refs.items():
        truth, preds_dense, metrics = score(phases, pred, rp, rt, arms)
        boots = C.block_bootstrap_relative_rms(truth, preds_dense)
        scores[rname] = {a: dict(metrics[a], interval=C.interval(boots[a]))
                         for a in metrics}
        intervals[rname] = boots
        truths[rname] = truth
    return dict(st=st, pred=pred, norm=norm, work=work, arms=arms,
                scores=scores, intervals=intervals, truths=truths,
                diagnostics=diagnostics, phases=np.asarray(phases, float),
                y_m=np.asarray(y_m_of_phase, float), note=note)


def paired(intervals, first, second):
    out = {}
    for rname in ("A_withdrawn_linear4", "B_mglet", "C_xiao_repaired_cubic6"):
        b = intervals[rname]
        if first not in b or second not in b:
            return None
        out[rname] = C.interval(b[first] - b[second])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stamp", default=STAMP)
    ap.add_argument("--c-points", type=int, default=len(C_NORM_GRID_FULL),
                    help="SMOKE TEST ONLY: subsample the calibration grid.")
    ap.add_argument("--stride", type=int, default=1,
                    help="SMOKE TEST ONLY; the production run uses stride 1 and "
                         "the output records the stride it used.")
    args = ap.parse_args()
    grid = (C_NORM_GRID_FULL if args.c_points >= len(C_NORM_GRID_FULL)
            else tuple(float(v) for v in np.geomspace(
                C_NORM_GRID_FULL[0], C_NORM_GRID_FULL[-1], int(args.c_points))))
    t_start = time.time()

    bench = fw.benchmarks(nu=C.NU)
    if not bench["all_pass"]:
        raise SystemExit("faithful operators failed their analytic benchmarks; "
                         "refusing to score")

    fields = C.DnsTangentFields()
    surf = CL.surfaces(fields)
    phase_A, tau_A = CL.reference_A(fields)
    phase_C, tau_C = CL.reference_C(fields)
    phase_B, tau_B, trailing = CL.reference_B()
    x_K, tau_K = CL.reference_K()
    refs = {"A_withdrawn_linear4": (phase_A, tau_A),
            "B_mglet": (phase_B, tau_B),
            "C_xiao_repaired_cubic6": (phase_C, tau_C)}

    stride = max(1, args.stride)
    surfaces = {}
    # 1. CALIBRATION surface -- the candidate's constant is chosen here, and
    #    only here.  The full grid is evaluated on this surface alone.
    ph, ym, note = surf[CALIBRATION_SURFACE]
    surfaces[CALIBRATION_SURFACE] = run_surface(
        fields, CALIBRATION_SURFACE, np.asarray(ph, float)[::stride],
        np.asarray(ym, float)[::stride], note, grid, refs)

    cal = surfaces[CALIBRATION_SURFACE]["scores"]["B_mglet"]
    grid_scores = {}
    for family in ("NLWM", "NLWH"):
        grid_scores[family] = {v: cal[f"{family}_Xall_c{v:.3e}"]["relative_rms"]
                               for v in grid if f"{family}_Xall_c{v:.3e}" in cal}
    c_star = {family: float(min(values, key=values.get))
              for family, values in grid_scores.items() if values}
    if set(c_star) != {"NLWM", "NLWH"}:
        raise SystemExit("the calibration surface scored no candidate arms; "
                         "this happens only when a smoke-test stride leaves "
                         "fewer stations than the scoring protocol requires")
    for family, value in c_star.items():
        print(f"calibrated {family} c* = {value:.4e} on {CALIBRATION_SURFACE} "
              f"(relative RMS {grid_scores[family][value]:.4f})")

    # 2. PRIMARY surface -- the frozen constant only.
    ph, ym, note = surf[PRIMARY_SURFACE]
    frozen = tuple(sorted(set(c_star.values())))
    surfaces[PRIMARY_SURFACE] = run_surface(
        fields, PRIMARY_SURFACE, np.asarray(ph, float)[::stride],
        np.asarray(ym, float)[::stride], note, frozen, refs)

    result = {
        "schema": "faithful_tournament_l0/1",
        "node": "development/nodes/node_000 (L0 attempt 1, norm-limited candidate)",
        "question": ("does capping the ASSEMBLED SOURCE NORM of a one-dimensional "
                     "wall model beat the strongest faithfully implemented "
                     "published families over a repeating curved wall?"),
        "preregistered_predictions": PREREGISTERED,
        "equivalence_margin_relative_rms": EQUIVALENCE_MARGIN,
        "operator_benchmarks": bench,
        "calibration": {
            "surface": CALIBRATION_SURFACE,
            "reference_used": "B_mglet",
            "grid": list(grid),
            "grid_relative_rms": {family: {f"{k:.3e}": v for k, v in values.items()}
                                  for family, values in grid_scores.items()},
            "c_star": c_star,
            "protocol": ("the single constant is chosen on the coupled-mesh "
                         "surface against the primary reference and then FROZEN; "
                         "the primary surface and the geometry holdout evaluate "
                         "that frozen value only"),
        },
        "references": {
            "A_withdrawn_linear4": "NEGATIVE_CONTROL_withdrawn_estimator_not_a_truth",
            "B_mglet": "PRIMARY_TRUTH",
            "C_xiao_repaired_cubic6": "SENSITIVITY_BRACKET_same_simulation",
            "K_krank_stations": "SPARSE_INDEPENDENT_CROSS_CHECK",
        },
        "inputs": {
            "dns_archive": {"path": str(C.DNS_FILE.relative_to(ROOT)),
                            "sha256": C.sha256(C.DNS_FILE)},
            "mglet_wall": {"path": str(CL.MGLET.relative_to(ROOT)),
                           "sha256": C.sha256(CL.MGLET)},
            "krank_stations": {"path": str(CL.KRANK.relative_to(ROOT)),
                               "sha256": C.sha256(CL.KRANK)},
            "faithful_models": {"path": "codes/models/faithful_wall_models_l0.py",
                                "sha256": C.sha256(ROOT / "codes/models/faithful_wall_models_l0.py")},
            "reference_models": {"path": "codes/models/source_faithful_wall_models.py",
                                 "sha256": C.sha256(ROOT / "codes/models/source_faithful_wall_models.py")},
        },
        "mglet_trailing_rows_stripped": np.asarray(trailing).tolist(),
        "bootstrap": {"block_points": C.BLOCK_POINTS, "dense_points": C.DENSE_N,
                      "draws": C.BOOTSTRAP_DRAWS, "seed": C.BOOTSTRAP_SEED,
                      "pairing": "identical resampled blocks across arms"},
        "station_stride": stride,
        "surfaces": {},
    }

    arrays = {}
    for sname, S in surfaces.items():
        candidate = f"NLWH_Xall_c{c_star['NLWH']:.3e}"
        candidate_m1 = f"NLWH_M1_c{c_star['NLWH']:.3e}"
        candidate_scale = f"NLWM_Xall_c{c_star['NLWM']:.3e}"
        candidate_scale_m1 = f"NLWM_M1_c{c_star['NLWM']:.3e}"
        published = [a for a in ("M0_equilibrium", "M1_pressure_gradient",
                                 "M2_hickel", "M2_hickel_Aplus26_variant",
                                 "M3_yang_integral", "M4_park_moin",
                                 "M5_meneveau") if a in S["scores"]["B_mglet"]]
        best_published = min(
            (a for a in published if a != "M2_hickel_Aplus26_variant"),
            key=lambda a: S["scores"]["B_mglet"][a]["relative_rms"])

        entry = {
            "note": S["note"],
            "stations": int(S["phases"].size),
            "y_m_over_H": {"min": float(S["y_m"].min()), "max": float(S["y_m"].max()),
                           "median": float(np.median(S["y_m"]))},
            "diagnostics": S["diagnostics"],
            "scores": S["scores"],
            "source_norm": {a: {
                "N_rms": float(np.sqrt(np.mean(S["norm"][a][np.isfinite(S["norm"][a])] ** 2)))
                if np.isfinite(S["norm"][a]).any() else None,
                "N_median": float(np.median(S["norm"][a][np.isfinite(S["norm"][a])]))
                if np.isfinite(S["norm"][a]).any() else None,
                "W_rms": float(np.sqrt(np.mean(S["work"][a][np.isfinite(S["work"][a])] ** 2)))
                if np.isfinite(S["work"][a]).any() else None,
            } for a in S["arms"]},
            "best_published_family": best_published,
            "contrasts": [],
        }

        def add(first, second, kind, question):
            d = paired(S["intervals"], first, second)
            if d is None:
                return
            entry["contrasts"].append({
                "kind": kind, "first": first, "second": second,
                "question": question, "delta": d,
                "identified": CL.identify(d["B_mglet"], d["C_xiao_repaired_cubic6"]),
                "tost_B": tost(d["B_mglet"], EQUIVALENCE_MARGIN),
                "tost_C": tost(d["C_xiao_repaired_cubic6"], EQUIVALENCE_MARGIN),
            })

        # Q1 -- the candidate against every faithful published family
        for a in published:
            add(candidate, a, "candidate_vs_published",
                f"norm-limited wall model against {a}")
        for arm in (candidate_scale, candidate_scale_m1):
            for other in published:
                add(arm, other, "candidate_vs_published",
                    f"uniformly rescaled source-norm limiter against {other}")
        add(candidate, "Xall", "candidate_vs_host",
            "does a wall-normal source horizon repair the exact-completion source?")
        add(candidate_m1, "M1_pressure_gradient", "candidate_vs_host",
            "does a wall-normal source horizon repair the pressure-gradient ODE?")
        add(candidate_scale, "Xall", "candidate_vs_host",
            "does uniform norm rescaling repair the exact-completion source?")
        add(candidate_scale_m1, "M1_pressure_gradient", "candidate_vs_host",
            "does uniform norm rescaling repair the pressure-gradient ODE?")
        add(candidate, candidate_scale, "candidate_vs_candidate",
            "horizon against uniform rescaling at equal calibration protocol")
        # Q2 -- matched norm
        add("CTL_exact_at_modelled_norm", "M2_hickel", "matched_norm",
            "exact content at the modelled norm against the modelled source")
        add("CTL_modelled_at_exact_norm", "Xall", "matched_norm",
            "modelled content at the exact norm against the exact source")
        # Q4 -- norm-preserving permutation
        for frac in SHIFT_FRACTIONS:
            for base in SHIFT_BASES:
                add(f"CTL_shift_{base}_{frac:g}", base, "phase_permutation",
                    f"norm-preserving permutation of the {base} source")
        # architecture at fixed norm
        add("M4_park_moin", "Xcp_pressure_plus_convection", "architecture",
            "same assembled source, wall-layer PDE against the shooting operator")
        # 2x2 factorial: shape and amplitude main effects, and the interaction
        add("FAC_exactshape_modelnorm", "M2_hickel", "factorial_shape",
            "shape effect at the modelled amplitude")
        add("Xall", "FAC_modelshape_exactnorm", "factorial_shape",
            "shape effect at the measured amplitude")
        add("FAC_modelshape_exactnorm", "M2_hickel", "factorial_amplitude",
            "amplitude effect at the modelled shape")
        add("Xall", "FAC_exactshape_modelnorm", "factorial_amplitude",
            "amplitude effect at the measured shape")
        add("ORACLE_closure_free", "M3_yang_integral", "architecture",
            "oracle momentum identity against the faithful integral model")

        # 2x2 factorial interaction
        factorial = {"delta": {}}
        cells = ("M2_hickel", "FAC_exactshape_modelnorm",
                 "FAC_modelshape_exactnorm", "Xall")
        for rname in ("A_withdrawn_linear4", "B_mglet", "C_xiao_repaired_cubic6"):
            b = S["intervals"][rname]
            if any(k not in b for k in cells):
                factorial = None
                break
            factorial["delta"][rname] = C.interval(
                (b["Xall"] - b["FAC_exactshape_modelnorm"])
                - (b["FAC_modelshape_exactnorm"] - b["M2_hickel"]))
        if factorial is not None:
            factorial["identified"] = CL.identify(
                factorial["delta"]["B_mglet"],
                factorial["delta"]["C_xiao_repaired_cubic6"])
            factorial["cells_relative_rms"] = {
                a: S["scores"]["B_mglet"][a]["relative_rms"] for a in cells
                if a in S["scores"]["B_mglet"]}
            factorial["cells_source_norm"] = {
                a: entry["source_norm"][a]["N_rms"] for a in cells
                if a in entry["source_norm"]}
        entry["shape_amplitude_factorial"] = factorial

        # Q4 interaction contrast
        interactions = []
        for frac in SHIFT_FRACTIONS:
            record = {"shift_fraction_of_period": frac, "delta": {}}
            ok = True
            for rname in ("A_withdrawn_linear4", "B_mglet", "C_xiao_repaired_cubic6"):
                b = S["intervals"][rname]
                need = [f"CTL_shift_M2_hickel_{frac:g}", "M2_hickel",
                        f"CTL_shift_Xall_{frac:g}", "Xall"]
                if any(k not in b for k in need):
                    ok = False
                    break
                record["delta"][rname] = C.interval(
                    (b[need[0]] - b[need[1]]) - (b[need[2]] - b[need[3]]))
            if ok:
                record["identified"] = CL.identify(
                    record["delta"]["B_mglet"],
                    record["delta"]["C_xiao_repaired_cubic6"])
                interactions.append(record)
        entry["phase_permutation_interaction"] = interactions

        # Q3 -- the norm law with no leakage
        law = {}
        for rname in ("B_mglet", "C_xiao_repaired_cubic6"):
            sc = S["scores"][rname]
            train = [a for a in sc if a.startswith("FIT_scale_")]
            # the test set excludes every arm that was fitted on, every
            # rescaled copy of a fitted base (same construction as a training
            # arm), the candidate itself, and the two architectures for which a
            # wall-normal assembled source is not defined.
            test = [a for a in sc
                    if (not a.startswith("FIT_scale_"))
                    and (not a.startswith("NLWM_"))
                    and (not a.startswith("CTL_"))
                    and a not in NORMLESS_ARMS
                    and entry["source_norm"].get(a, {}).get("N_rms") is not None]
            leakage = sorted(set(train) & set(test))
            Ns = [entry["source_norm"][a]["N_rms"] for a in train]
            Es = [sc[a]["absolute_rms"] for a in train]
            E0, delta, rms = affine_fit(Ns, Es)
            null = float(np.mean(Es))          # zero-parameter, TRAINING arms only
            rel = {a: abs(E0 + delta * entry["source_norm"][a]["N_rms"]
                          - sc[a]["absolute_rms"]) / max(sc[a]["absolute_rms"], TINY)
                   for a in test}
            rel_null = {a: abs(null - sc[a]["absolute_rms"])
                        / max(sc[a]["absolute_rms"], TINY) for a in test}
            # leave-one-base-out: drop a whole base arm's scaled copies from the fit
            lobo = {}
            for base in SCALE_BASES:
                keep = [a for a in train if not a.startswith(f"FIT_scale_{base}_")]
                if len(keep) < 2:
                    continue
                e0, dl, _ = affine_fit([entry["source_norm"][a]["N_rms"] for a in keep],
                                       [sc[a]["absolute_rms"] for a in keep])
                held = [a for a in test if a == base]
                lobo[base] = {
                    "E0": e0, "delta": dl,
                    "held_out_relative_error": (
                        abs(e0 + dl * entry["source_norm"][base]["N_rms"]
                            - sc[base]["absolute_rms"]) / max(sc[base]["absolute_rms"], TINY)
                        if held else None),
                }
            law[rname] = {
                "fitted_on": sorted(train), "tested_on": sorted(test),
                "training_test_overlap": leakage,
                "excluded_normless_arms": list(NORMLESS_ARMS),
                "E0": E0, "delta": delta, "fit_rms": rms,
                "held_out_median_relative_error": float(np.median(list(rel.values()))),
                "held_out_median_relative_error_shooting_family_only": float(np.median(
                    [rel[a] for a in test
                     if a not in ("M4_park_moin", "ORACLE_closure_free")])),
                "zero_parameter_null_median_relative_error":
                    float(np.median(list(rel_null.values()))),
                "zero_parameter_null_value": null,
                "per_arm_relative_error": rel,
                "leave_one_base_out": lobo,
            }
        entry["norm_law"] = law

        # Q5 -- architecture spread at fixed norm
        spread = {}
        for rname in ("B_mglet", "C_xiao_repaired_cubic6"):
            sc = S["scores"][rname]
            same_norm = [a for a in ("Xcp_pressure_plus_convection", "M4_park_moin")
                         if a in sc]
            if len(same_norm) == 2:
                Ns = [entry["source_norm"][a]["N_rms"] for a in same_norm]
                Es = [sc[a]["absolute_rms"] for a in same_norm]
                sweep = [sc[a]["absolute_rms"] for a in sc if a.startswith("FIT_scale_")]
                spread[rname] = {
                    "arms": same_norm, "norms": Ns, "absolute_rms": Es,
                    "norm_ratio": float(max(Ns) / max(min(Ns), TINY)),
                    "error_ratio": float(max(Es) / max(min(Es), TINY)),
                    "amplitude_sweep_range": float(max(sweep) - min(sweep)) if sweep else None,
                }
        entry["architecture_at_fixed_norm"] = spread

        # Krank sparse cross-check
        kr = {}
        ph_k = np.mod(np.asarray(x_K, float) / C.LX, 1.0)
        for a in S["arms"]:
            v = S["pred"][a]
            ok = np.isfinite(v)
            if ok.sum() < 8:
                continue
            pk = C.periodic_interp(S["phases"][ok], v[ok], ph_k)
            err = pk - np.asarray(tau_K, float)
            kr[a] = float(np.sqrt(np.mean(err ** 2))
                          / np.sqrt(np.mean(np.asarray(tau_K, float) ** 2)))
        entry["krank_station_relative_rms"] = kr

        result["surfaces"][sname] = entry
        arrays[f"{sname}__phase"] = S["phases"]
        arrays[f"{sname}__y_m"] = S["y_m"]
        for a in S["arms"]:
            arrays[f"{sname}__pred__{a}"] = S["pred"][a]
            arrays[f"{sname}__norm__{a}"] = S["norm"][a]
        for k, v in S["st"].items():
            arrays[f"{sname}__station__{k}"] = v
        for rname, t in S["truths"].items():
            arrays[f"{sname}__truth_dense__{rname}"] = t

    # ---------------- registered verdicts ------------------------------------
    prim = result["surfaces"][PRIMARY_SURFACE]
    best = prim["best_published_family"]
    verdicts = {}
    for family, arm in (("NLWH", f"NLWH_Xall_c{c_star['NLWH']:.3e}"),
                        ("NLWM", f"NLWM_Xall_c{c_star['NLWM']:.3e}")):
        q1 = next((c for c in prim["contrasts"]
                   if c["kind"] == "candidate_vs_published"
                   and c["first"] == arm and c["second"] == best), None)
        verdicts[f"Q1_{family}_beats_the_best_faithful_family"] = {
            "best_published_family": best,
            "candidate": arm,
            "candidate_relative_rms": prim["scores"]["B_mglet"][arm]["relative_rms"],
            "best_published_relative_rms": prim["scores"]["B_mglet"][best]["relative_rms"],
            "identified": q1["identified"] if q1 else None,
            "verdict": ("WIN" if q1 and q1["identified"] == "IDENTIFIED_FIRST_BETTER"
                        else "REFUTED" if q1 and q1["identified"] == "IDENTIFIED_SECOND_BETTER"
                        else "NO_ADVANTAGE_DEMONSTRATED"),
        }
    q2 = next((c for c in prim["contrasts"]
               if c["kind"] == "matched_norm"
               and c["first"] == "CTL_exact_at_modelled_norm"), None)
    if q2:
        if q2["identified"] == "IDENTIFIED_FIRST_BETTER":
            v = "CONTENT_HELPS_AT_MATCHED_NORM"
        elif q2["identified"] == "IDENTIFIED_SECOND_BETTER":
            v = "CONTENT_HURTS_AT_MATCHED_NORM"
        elif q2["tost_B"] == "EQUIVALENT_WITHIN_MARGIN" and q2["tost_C"] == "EQUIVALENT_WITHIN_MARGIN":
            v = "CONTENT_IRRELEVANT_AT_MATCHED_NORM"
        else:
            v = "UNRESOLVED"
        verdicts["Q2_matched_norm_separates_content_from_magnitude"] = {
            "delta": q2["delta"], "identified": q2["identified"],
            "tost_B": q2["tost_B"], "tost_C": q2["tost_C"], "verdict": v}
    law = prim["norm_law"]["B_mglet"]
    verdicts["Q3_norm_law_without_leakage"] = {
        "training_test_overlap": law["training_test_overlap"],
        "held_out_median_relative_error": law["held_out_median_relative_error"],
        "zero_parameter_null": law["zero_parameter_null_median_relative_error"],
        "verdict": ("SUPPORTED" if (not law["training_test_overlap"]
                                    and law["held_out_median_relative_error"] < 0.5
                                    and law["held_out_median_relative_error"]
                                    < law["zero_parameter_null_median_relative_error"])
                    else "REFUTED"),
    }
    verdicts["Q4_phase_permutation_is_norm_preserving"] = {
        "interactions": prim["phase_permutation_interaction"],
        "per_arm": [c for c in prim["contrasts"] if c["kind"] == "phase_permutation"],
    }
    verdicts["Q5_architecture_is_not_a_source_label"] = prim["architecture_at_fixed_norm"]
    verdicts["Q6_shape_versus_amplitude_factorial"] = prim["shape_amplitude_factorial"]
    result["registered_verdicts"] = verdicts
    result["c_star"] = c_star
    result["runtime_seconds"] = time.time() - t_start

    out_json = ROOT / "codes/results" / f"faithful_tournament_l0_{args.out_stamp}.json"
    out_npz = ROOT / "codes/results" / f"faithful_tournament_l0_{args.out_stamp}.npz"
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    np.savez_compressed(out_npz, **arrays)
    print("wrote", out_json.name, out_npz.name)
    for sname in (CALIBRATION_SURFACE, PRIMARY_SURFACE):
        sc = result["surfaces"][sname]["scores"]["B_mglet"]
        print(f"--- {sname} (B_mglet) ---")
        for a in sorted(sc, key=lambda k: sc[k]["relative_rms"]):
            frozen_tags = tuple(f"c{v:.3e}" for v in c_star.values())
            if a.startswith("FIT_scale_") or (
                    (a.startswith("NLWM_") or a.startswith("NLWH_"))
                    and not any(tag in a for tag in frozen_tags)):
                continue
            N = result["surfaces"][sname]["source_norm"][a]["N_rms"]
            print(f"   {a:38s} E={sc[a]['relative_rms']:9.3f} "
                  f"R2={sc[a]['r2']:10.3f} N={'n/a' if N is None else f'{N:.4e}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
