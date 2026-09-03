#!/usr/bin/env python3
"""L0 attempt-3 producer: IS THE SOURCE NORM, RATHER THAN THE SOURCE PHYSICS,
WHAT SETS WALL-MODEL SKILL OVER A REPEATING CURVED WALL?

Question under test
-------------------
Every wall model in use integrates a one-dimensional wall-normal momentum
balance driven by a source term s(n).  The families differ in WHAT they put in
s: nothing (equilibrium), the wall pressure gradient (Balaras), a parametrised
convective surrogate (Hickel), the resolved convective flux (Park & Moin), the
integrated momentum flux (Yang), or -- in the exact-completion limit -- every
omitted transport term measured from the reference itself.

The published expectation is monotone: a source that is closer to the true
balance should give a better wall traction.  On a repeating curved wall the
opposite is observed, and the mechanism proposed here is that the
one-dimensional reduction is NOT a closed balance on a curved wall, so it
carries an inconsistency of relative size delta that multiplies whatever
source magnitude the model assembles.  If that is right, the controlling
variable is the ASSEMBLED SOURCE NORM

    N = (1/G) * int_0^{y_m} [ int_0^n |s| dn' ] / D(n) dn ,
    G =        int_0^{y_m} dn / D(n) ,

with D = nu + nu_t the eddy diffusivity of the EQUILIBRIUM solution (so N is
the same weighting for every arm, and contains no wall-stress reference at
all), and the absolute error should obey  E_abs ~ E_0 + delta * N  with delta
a property of the GEOMETRY and not of which physical terms are present.

That is a falsifiable statement, and this producer tests it by intervention
rather than by correlation:

  (i)  PHASE-SHIFT CONTROL.  Give the model the source belonging to a DIFFERENT
       station of the same wall (a rigid shift of Lx/8, Lx/4, Lx/2).  The norm,
       the smoothness and the term composition are untouched; only the physical
       correspondence to the station is destroyed.  If the physics carried the
       skill, the shifted arm must be much worse.  If the norm carried it, the
       shifted arm scores the same.
  (ii) AMPLITUDE SWEEP.  Scale a source by c in {0.5, 2} with its physics
       untouched: this moves N alone.
  (iii) MATCHED-NORM SINGLE-TERM ARMS.  Four physically distinct exact terms,
       each rescaled by one global constant to a COMMON norm.  Equal error at
       equal norm is content-independence.
  (iv) INTERPOLATION FAMILY.  s_lambda = (1-lambda) s_M1 + lambda s_Xall,
       a continuous path from the modelled to the exact source.

and it does so alongside a TOURNAMENT of the published families M0--M5 of the
source-faithful registry, all on one surface, one truth protocol and one
uncertainty protocol.  M3 (Yang) is evaluated in its integral-equation form
with resolved inputs, which is the best case that family can have; the producer
checks that it then coincides with the closure-free reconstruction.

Truth protocol is the campaign's: A = withdrawn linear-4 estimator (NEGATIVE
CONTROL ONLY), B = Peller & Manhart MGLET full-wall DNS (PRIMARY),
C = repaired same-simulation cubic (BRACKET), K = Krank stations (sparse
cross-check).  A contrast counts only when its paired phase-block interval
excludes zero with the same sign under BOTH B and C.

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

STAMP = "20260825"
N_QUAD = 400            # wall-normal quadrature points per station
N_SHOOT = 200           # deposited-ladder shooting resolution
TINY = 1.0e-30

# --------------------------------------------------------------------------- #
# Interventions fixed BEFORE the run.
# --------------------------------------------------------------------------- #
SHIFT_FRACTIONS = (0.125, 0.25, 0.5)      # of the period, applied to the source
SHIFT_BASES = ("M2_hickel", "Xall")
SCALE_FACTORS = (0.5, 2.0)
SCALE_BASES = ("M1_pressure_gradient", "M2_hickel", "Xall")
LAMBDAS = (0.25, 0.5, 0.75)
MATCHED_TERMS = ("dpds", "conv", "dRtt", "visc")

# Registered predictions.  Written here, before the run, so that the verdict is
# a comparison against a stated expectation and not a description of whatever
# came out.
PREREGISTERED = {
    "P1_shift_neutrality": (
        "If the assembled source norm governs the error, a rigid phase shift of "
        "the EXACT source (Xall) leaves its relative-RMS error statistically "
        "unchanged: the paired 95% interval of E(shifted) - E(exact) contains "
        "zero under BOTH corrected references. SUPPORTED / REFUTED / UNRESOLVED "
        "by that interval."),
    "P2_modelled_source_is_informative": (
        "The same shift applied to the MODELLED source (Hickel) must degrade it "
        "-- a modelled source that carries no station information would be an "
        "indictment of the model, not of the reduction. Interval of "
        "E(shifted M2) - E(M2) strictly positive under both references."),
    "P3_content_independence_at_matched_norm": (
        "Four physically distinct exact terms, rescaled by one global constant "
        "each to a COMMON norm, give errors whose spread is small compared with "
        "the error range spanned by the norm sweep. Quantified as "
        "spread_at_matched_norm / range_over_norm_sweep < 0.5."),
    "P4_affine_norm_law": (
        "E_abs is affine in N with a positive slope: fitting E_abs = E_0 + "
        "delta*N over the AMPLITUDE-SWEEP arms only (physics fixed, norm moved) "
        "and then PREDICTING the published families M0--M5 out of sample gives a "
        "median relative prediction error below 0.5. Reported with a "
        "zero-parameter null (predict every arm by the arm-mean error) so that "
        "the law has to beat something."),
    "P5_geometry_carries_delta": (
        "delta fitted on the FLAT floor is at least three times smaller than "
        "delta fitted on the SLOPED wall of the same simulation, i.e. the "
        "inconsistency is a property of surface curvature."),
}


# --------------------------------------------------------------------------- #
# reference-free source geometry
# --------------------------------------------------------------------------- #
def equilibrium_diffusivity(n_grid: np.ndarray, tau0: float, nu: float) -> np.ndarray:
    """D = nu + nu_t of the EQUILIBRIUM (constant-stress) solution at tau0.

    Arm-independent by construction, so the norm below weights every candidate
    source identically.  Returns D >= nu everywhere.
    """
    if abs(tau0) < TINY:
        return np.full_like(n_grid, nu)
    length = wm.mixing_length(n_grid, tau0, nu)
    strain = wm.stable_strain_from_stress(np.full_like(n_grid, tau0), length, nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(np.abs(strain) > TINY, np.abs(tau0) / np.abs(strain), nu)
    return np.maximum(np.nan_to_num(D, nan=nu, posinf=nu), nu)


def norm_and_work(n_grid: np.ndarray, D: np.ndarray, G: float,
                  source_values: np.ndarray) -> tuple[float, float]:
    """(N, W): the assembled and the net traction-equivalent source magnitude.

    W is the amount the source actually removes from tau_w in the linearised
    balance tau_w * G = u_m - W; N is the same functional applied to |s|, i.e.
    the magnitude that an inconsistency of relative size delta can convert into
    an error.  Neither uses any wall-stress reference.
    """
    inner_abs = _cumtrap(np.abs(source_values), n_grid)
    inner_net = _cumtrap(source_values, n_grid)
    N = float(np.trapezoid(inner_abs / D, n_grid) / G)
    W = float(np.trapezoid(inner_net / D, n_grid) / G)
    return N, W


def _cumtrap(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


# --------------------------------------------------------------------------- #
# station pre-pass: everything that needs no shooting
# --------------------------------------------------------------------------- #
def prepass(fields, phases, y_m_of_phase):
    """Per-station geometry, inputs and exact term profiles on a normalised grid."""
    n_st = len(phases)
    xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
    terms = {k: np.zeros((n_st, N_QUAD)) for k in MATCHED_TERMS}
    st = {k: np.zeros(n_st) for k in
          ("u_m", "dpds", "tau0", "y_m", "G", "tau_at_ym", "truth_A", "index")}
    D_all = np.zeros((n_st, N_QUAD))
    n_all = np.zeros((n_st, N_QUAD))
    x_targets = np.mod(phases, 1.0) * C.LX
    for p, (xt, y_m) in enumerate(zip(x_targets, y_m_of_phase)):
        i = int(np.argmin(np.abs(fields.x - xt)))
        u_m, _, _ = fields.station(i, y_m)
        dpds = float(fields.dpds_total[i])
        tau0 = wm.spalding_wall_stress(u_m, y_m, C.NU) if abs(u_m) > 1e-12 else 0.0
        n_grid = y_m * xi
        for k in MATCHED_TERMS:
            terms[k][p] = np.asarray(fields.profile_of(k, i)(n_grid), float)
        D = equilibrium_diffusivity(n_grid, tau0, C.NU)
        st["u_m"][p] = u_m
        st["dpds"][p] = dpds
        st["tau0"][p] = tau0
        st["y_m"][p] = y_m
        st["index"][p] = i
        st["G"][p] = float(np.trapezoid(1.0 / D, n_grid))
        st["tau_at_ym"][p] = float(fields.profile_of("tau", i)([y_m])[0])
        st["truth_A"][p] = fields.tau_s_truth[i]
        D_all[p] = D
        n_all[p] = n_grid
    return st, terms, D_all, n_all


def park_moin_arm(fields, st, terms, n_all, p):
    """M4: steady Park & Moin non-equilibrium wall layer with the published
    DYNAMIC eddy viscosity built from the reference Reynolds stress and mean
    deviatoric strain, marched to steady state on a uniform grid.

    The unsteady term of Park & Moin (2014) vanishes for a statistically steady
    reference, so the steady limit is the faithful a-priori form; what remains
    distinct from the mixing-length rungs is their closure, which is supplied
    here from the same reference the model would receive in a simulation.
    """
    i = int(st["index"][p])
    y_m = float(st["y_m"][p])
    n_uniform = np.linspace(0.0, y_m, 65)
    comps = [f(fields._normal_points(i, n_uniform)) for f in fields._interp("tau")]
    Ux, Uy, Vx, Vy, Ruu, Rvv, Ruv = comps
    # 2-D mean deviatoric strain and resolved Reynolds stress in the fixed frame
    Sxx = Ux - 0.5 * (Ux + Vy)
    Syy = Vy - 0.5 * (Ux + Vy)
    Sxy = 0.5 * (Uy + Vx)
    Rxx, Ryy, Rxy = Ruu, Rvv, Ruv
    numerator = Rxx * Sxx + Ryy * Syy + 2.0 * Rxy * Sxy
    denominator = 2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_t = np.where(denominator > TINY, numerator / np.maximum(denominator, TINY), 0.0)
    nu_t = np.maximum(np.nan_to_num(nu_t, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    conv = np.interp(n_uniform, n_all[p], terms["conv"][p])
    pres = np.interp(n_uniform, n_all[p], terms["dpds"][p])
    state = np.linspace(0.0, float(st["u_m"][p]), n_uniform.size)
    dt = max(y_m * y_m / (4.0 * C.NU) * 1.0e-2, 1.0e-4)
    tau = 0.0
    for _ in range(60):
        step = wm.park_moin_wall_layer_step(
            state, n_uniform, dt, C.NU, nu_t, float(st["u_m"][p]),
            convective_term=conv, pressure_gradient=pres)
        if np.max(np.abs(step.velocity - state)) < 1.0e-12 * max(1.0, abs(st["u_m"][p])):
            state = step.velocity
            tau = step.tau_w
            break
        state = step.velocity
        tau = step.tau_w
    return float(tau), float(np.max(np.abs(nu_t)))


def meneveau_arm(st, p):
    """M5 smooth pressure-corrected branch, Meneveau (2020) eqs (40)-(43)."""
    u_m = float(st["u_m"][p])
    y_m = float(st["y_m"][p])
    dpds = float(st["dpds"][p])
    if abs(u_m) < 1e-12:
        return 0.0, "zero_matching_velocity"
    re_delta = abs(u_m) * y_m / C.NU
    psi_p = np.sign(u_m) * dpds * y_m ** 3 / (C.NU ** 2)
    if not (0.0 < re_delta < 1.0e7) or not abs(psi_p) < 2.0e7:
        return np.nan, "outside_published_fit_domain"
    re_tau = wm.meneveau_pressure_re_tau(re_delta, psi_p)
    u_tau = abs(u_m) * re_tau / re_delta
    return float(np.sign(u_m) * u_tau * u_tau), "ok"


# --------------------------------------------------------------------------- #
# arm construction
# --------------------------------------------------------------------------- #
def build_sources(st, terms, n_all, p, n_st, matched_scales):
    """Source VALUE ARRAYS on this station's quadrature grid, one per arm."""
    n_grid = n_all[p]
    dpds_const = np.full(N_QUAD, st["dpds"][p])
    s = {}
    s["M1_pressure_gradient"] = dpds_const
    s["M2_hickel"] = wm.hickel_source(n_grid, float(st["dpds"][p]), C.NU)
    s["M2b_hickel_Aplus17"] = s["M2_hickel"]
    exact = {k: terms[k][p] for k in MATCHED_TERMS}
    s["Xc_exact_convection"] = dpds_const + exact["conv"]
    s["Xcp_pressure_plus_convection"] = exact["dpds"] + exact["conv"]
    s["Xcpr_plus_normal_stress"] = exact["dpds"] + exact["conv"] + exact["dRtt"]
    s["Xall"] = exact["dpds"] + exact["conv"] + exact["dRtt"] + exact["visc"]
    # (i) phase-shift controls: the source of another station of the same wall
    for frac in SHIFT_FRACTIONS:
        j = int(round(frac * n_st)) % n_st
        for base in SHIFT_BASES:
            q = (p + j) % n_st
            if base == "Xall":
                vals = (terms["dpds"][q] + terms["conv"][q]
                        + terms["dRtt"][q] + terms["visc"][q])
            else:
                vals = wm.hickel_source(n_grid, float(st["dpds"][q]), C.NU)
            s[f"CTL_shift_{base}_{frac:g}"] = vals
    # (ii) amplitude sweep: physics fixed, norm moved
    for base in SCALE_BASES:
        for c in SCALE_FACTORS:
            s[f"CTL_scale_{base}_{c:g}"] = c * s[base]
    # (iii) matched-norm single-term arms
    for k in MATCHED_TERMS:
        s[f"CTL_term_{k}_matchedN"] = matched_scales[k] * exact[k]
    # (iv) interpolation family
    for lam in LAMBDAS:
        s[f"CTL_lambda_{lam:g}"] = (1.0 - lam) * s["M1_pressure_gradient"] + lam * s["Xall"]
    return s


SHOOT_A17 = {"M2b_hickel_Aplus17"}


def evaluate_surface(fields, phases, y_m_of_phase, log=print):
    n_st = len(phases)
    t0 = time.time()
    st, terms, D_all, n_all = prepass(fields, phases, y_m_of_phase)
    log(f"  prepass {time.time() - t0:.0f}s over {n_st} stations")

    # global scale factors that put each single exact term at a COMMON norm.
    # The target is the norm of the exact-completion arm Xall, so the matched
    # arms are compared at the norm that the completion limit actually reaches.
    def term_norm(vals_of_p):
        acc = np.empty(n_st)
        for p in range(n_st):
            acc[p], _ = norm_and_work(n_all[p], D_all[p], st["G"][p], vals_of_p(p))
        return float(np.sqrt(np.mean(acc ** 2)))

    target = term_norm(lambda p: (terms["dpds"][p] + terms["conv"][p]
                                  + terms["dRtt"][p] + terms["visc"][p]))
    matched_scales = {}
    for k in MATCHED_TERMS:
        base = term_norm(lambda p, k=k: terms[k][p])
        matched_scales[k] = float(target / max(base, TINY))
    log(f"  matched-norm target N*={target:.4e}; scales " +
        ", ".join(f"{k}:{v:.3f}" for k, v in matched_scales.items()))

    probe = build_sources(st, terms, n_all, 0, n_st, matched_scales)
    arm_names = list(probe.keys())
    pred = {a: np.full(n_st, np.nan) for a in arm_names}
    norm = {a: np.full(n_st, np.nan) for a in arm_names}
    work = {a: np.full(n_st, np.nan) for a in arm_names}
    for a in ("M0_equilibrium", "M3_yang_integral", "M4_park_moin", "M5_meneveau",
              "Xfull_closure_free"):
        pred[a] = np.full(n_st, np.nan)
        norm[a] = np.full(n_st, np.nan)
        work[a] = np.full(n_st, np.nan)
    roots_M1 = np.zeros(n_st)
    m5_status = []
    t0 = time.time()
    for p in range(n_st):
        srcs = build_sources(st, terms, n_all, p, n_st, matched_scales)
        n_grid, D, G = n_all[p], D_all[p], float(st["G"][p])
        u_m, y_m, tau0 = float(st["u_m"][p]), float(st["y_m"][p]), float(st["tau0"][p])
        pred["M0_equilibrium"][p] = tau0
        norm["M0_equilibrium"][p] = 0.0
        work["M0_equilibrium"][p] = 0.0
        # closure-free reconstruction == Yang integral equation with resolved inputs
        exact_sum = (terms["dpds"][p] + terms["conv"][p]
                     + terms["dRtt"][p] + terms["visc"][p])
        impulse = float(np.trapezoid(exact_sum, n_grid))
        pred["Xfull_closure_free"][p] = float(st["tau_at_ym"][p]) - impulse
        pred["M3_yang_integral"][p] = wm.yang_integral_wall_stress(
            float(st["tau_at_ym"][p]), 0.0, impulse)
        for a in ("Xfull_closure_free", "M3_yang_integral"):
            norm[a][p], work[a][p] = norm_and_work(n_grid, D, G, exact_sum)
        pm_tau, _ = park_moin_arm(fields, st, terms, n_all, p)
        pred["M4_park_moin"][p] = pm_tau
        norm["M4_park_moin"][p], work["M4_park_moin"][p] = norm_and_work(
            n_grid, D, G, terms["dpds"][p] + terms["conv"][p])
        m5, status = meneveau_arm(st, p)
        pred["M5_meneveau"][p] = m5
        norm["M5_meneveau"][p] = 0.0
        work["M5_meneveau"][p] = 0.0
        m5_status.append(status)
        for a, vals in srcs.items():
            norm[a][p], work[a][p] = norm_and_work(n_grid, D, G, vals)
            src = (lambda v: (lambda y: np.interp(np.asarray(y, float), n_grid, v)))(vals)
            a_plus = wm.HICKEL_VAN_DRIEST_A if a in SHOOT_A17 else wm.VAN_DRIEST_A
            res = wm.shoot_wall_stress(u_m, y_m, C.NU, src, continuation_tau=tau0,
                                       n_points=N_SHOOT, a_plus=a_plus)
            pred[a][p] = res.tau_w
            if a == "M1_pressure_gradient":
                roots_M1[p] = len(res.roots)
        if p % 64 == 0:
            done = p + 1
            rate = (time.time() - t0) / done
            log(f"  station {done}/{n_st}  {rate * (n_st - done):.0f}s remaining")
    log(f"  shooting {time.time() - t0:.0f}s")
    return st, pred, norm, work, roots_M1, matched_scales, target, m5_status


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(phases, pred, ref_phase, ref_tau, arms):
    dense = np.arange(C.DENSE_N) / C.DENSE_N
    t = C.periodic_interp(ref_phase, ref_tau, dense)
    preds_dense, metrics = {}, {}
    for a in arms:
        v = pred[a]
        ok = np.isfinite(v)
        if ok.sum() < 8:
            continue
        p_d = C.periodic_interp(np.asarray(phases)[ok], v[ok], dense)
        preds_dense[a] = p_d
        err = p_d - t
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        metrics[a] = {
            "relative_rms": float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(t ** 2))),
            "absolute_rms": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
            "sign_accuracy": float(np.mean(np.sign(p_d) == np.sign(t))),
            "finite_stations": int(ok.sum()),
        }
    return t, preds_dense, metrics


def affine_fit(N, E):
    """Least-squares E = E0 + delta*N with the fit diagnostics."""
    N = np.asarray(N, float)
    E = np.asarray(E, float)
    A = np.vstack([np.ones_like(N), N]).T
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    resid = E - A @ coef
    return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(resid ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="archive_index10")
    ap.add_argument("--out-stamp", default=STAMP)
    ap.add_argument("--stride", type=int, default=1,
                    help="SMOKE TEST ONLY: evaluate every k-th station. The "
                         "production run uses stride 1 and the output records it.")
    args = ap.parse_args()

    t_start = time.time()
    fields = C.DnsTangentFields()
    surf = CL.surfaces(fields)
    if args.surface not in surf:
        raise SystemExit(f"unknown surface {args.surface}; have {list(surf)}")
    phases, y_m_of_phase, surface_note = surf[args.surface]
    phases = np.asarray(phases, float)[::max(1, args.stride)]
    y_m_of_phase = np.asarray(y_m_of_phase, float)[::max(1, args.stride)]
    print(f"surface {args.surface}: {len(phases)} stations "
          f"(stride {args.stride}), {surface_note}")

    st, pred, norm, work, roots_M1, matched_scales, target, m5_status = \
        evaluate_surface(fields, phases, y_m_of_phase)
    arms = [a for a in pred if np.isfinite(pred[a]).any()]

    phase_A, tau_A = CL.reference_A(fields)
    phase_C, tau_C = CL.reference_C(fields)
    phase_B, tau_B, trailing = CL.reference_B()
    x_K, tau_K = CL.reference_K()
    refs = {"A_withdrawn_linear4": (phase_A, tau_A),
            "B_mglet": (phase_B, tau_B),
            "C_xiao_repaired_cubic6": (phase_C, tau_C)}

    result = {
        "schema": "source_budget_tournament_l0/1",
        "node": "development/nodes/node_002 (L0 attempt 3, source-norm budget)",
        "question": ("does the ASSEMBLED SOURCE NORM, rather than the physical "
                     "content of the source, set wall-model skill over a "
                     "repeating curved wall?"),
        "surface": {"name": args.surface, "note": surface_note,
                    "stations": int(len(phases)), "station_stride": int(args.stride),
                    "y_m_over_H": {"min": float(y_m_of_phase.min()),
                                   "max": float(y_m_of_phase.max()),
                                   "median": float(np.median(y_m_of_phase))}},
        "preregistered_predictions": PREREGISTERED,
        "interventions": {
            "phase_shift_fractions_of_period": list(SHIFT_FRACTIONS),
            "phase_shift_bases": list(SHIFT_BASES),
            "amplitude_scale_factors": list(SCALE_FACTORS),
            "amplitude_scale_bases": list(SCALE_BASES),
            "interpolation_lambdas": list(LAMBDAS),
            "matched_norm_target": target,
            "matched_norm_global_scales": matched_scales,
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
            "wall_model_module": {"path": "codes/models/source_faithful_wall_models.py",
                                  "sha256": C.sha256(ROOT / "codes/models/source_faithful_wall_models.py")},
            "ladder_common_module": {"path": "codes/analysis/r2m4_ladder_common.py",
                                     "sha256": C.sha256(ROOT / "codes/analysis/r2m4_ladder_common.py")},
        },
        "mglet_trailing_rows_stripped": np.asarray(trailing).tolist(),
        "m5_status_counts": {s: int(m5_status.count(s)) for s in set(m5_status)},
        "M1_multiple_root_stations": int(np.sum(roots_M1 > 1)),
        "scores": {},
        "source_norm": {},
        "bootstrap": {"block_points": C.BLOCK_POINTS, "dense_points": C.DENSE_N,
                      "draws": C.BOOTSTRAP_DRAWS, "seed": C.BOOTSTRAP_SEED,
                      "pairing": "identical resampled blocks across arms within a reference"},
    }
    for a in arms:
        n = norm[a][np.isfinite(norm[a])]
        w = work[a][np.isfinite(work[a])]
        result["source_norm"][a] = {
            "N_rms": float(np.sqrt(np.mean(n ** 2))) if n.size else float("nan"),
            "N_median": float(np.median(n)) if n.size else float("nan"),
            "W_rms": float(np.sqrt(np.mean(w ** 2))) if w.size else float("nan"),
        }

    arrays = {"phase": phases, "y_m": y_m_of_phase}
    for a in arms:
        arrays[f"pred__{a}"] = pred[a]
        arrays[f"norm__{a}"] = norm[a]
        arrays[f"work__{a}"] = work[a]
    for k, v in st.items():
        arrays[f"station__{k}"] = v

    intervals = {}
    for rname, (rp, rt) in refs.items():
        t_dense, preds_dense, metrics = score(phases, pred, rp, rt, arms)
        boots = C.block_bootstrap_relative_rms(t_dense, preds_dense)
        result["scores"][rname] = {
            a: dict(metrics[a], interval=C.interval(boots[a])) for a in metrics}
        intervals[rname] = boots
        arrays[f"truth_dense__{rname}"] = t_dense
    # Krank station cross-check (station metric only; no bootstrap claimed)
    kr = {}
    ph_k = np.mod(np.asarray(x_K, float) / C.LX, 1.0)
    for a in arms:
        v = pred[a]
        ok = np.isfinite(v)
        pk = C.periodic_interp(np.asarray(phases)[ok], v[ok], ph_k)
        err = pk - np.asarray(tau_K, float)
        kr[a] = {"relative_rms": float(np.sqrt(np.mean(err ** 2))
                                       / np.sqrt(np.mean(np.asarray(tau_K, float) ** 2))),
                 "stations": int(len(ph_k))}
    result["scores"]["K_krank_stations"] = kr

    # ---------------- interventions, as paired identifiable contrasts --------
    def paired(first, second):
        out = {}
        for rname in ("A_withdrawn_linear4", "B_mglet", "C_xiao_repaired_cubic6"):
            b = intervals[rname]
            if first not in b or second not in b:
                return None
            out[rname] = C.interval(b[first] - b[second])
        return out

    contrasts = []
    for frac in SHIFT_FRACTIONS:
        for base in SHIFT_BASES:
            a = f"CTL_shift_{base}_{frac:g}"
            b = "Xall" if base == "Xall" else "M2_hickel"
            d = paired(a, b)
            if d is None:
                continue
            contrasts.append({
                "kind": "phase_shift",
                "first": a, "second": b, "shift_fraction_of_period": frac,
                "norm_ratio": (result["source_norm"][a]["N_rms"]
                               / max(result["source_norm"][b]["N_rms"], TINY)),
                "delta": d,
                "identified": CL.identify(d["B_mglet"], d["C_xiao_repaired_cubic6"]),
            })
    for a, b, why in (
            ("M2_hickel", "M0_equilibrium", "modelled convection vs equilibrium"),
            ("M1_pressure_gradient", "M0_equilibrium", "pressure-gradient ODE vs equilibrium"),
            ("Xall", "M0_equilibrium", "exact completion vs equilibrium"),
            ("Xall", "M2_hickel", "completion vs regularisation"),
            ("Xfull_closure_free", "Xall", "removing the last closure"),
            ("M3_yang_integral", "M0_equilibrium", "integral family vs equilibrium"),
            ("M4_park_moin", "M0_equilibrium", "non-equilibrium PDE family vs equilibrium"),
            ("M4_park_moin", "M2_hickel", "non-equilibrium PDE vs modelled convection"),
            ("M5_meneveau", "M0_equilibrium", "pressure-corrected algebraic vs equilibrium"),
    ):
        d = paired(a, b)
        if d is None:
            continue
        contrasts.append({"kind": "tournament", "first": a, "second": b, "question": why,
                          "delta": d,
                          "identified": CL.identify(d["B_mglet"], d["C_xiao_repaired_cubic6"])})
    result["contrasts"] = contrasts
    result["identifiability_rule"] = (
        "a contrast is IDENTIFIED only if its paired 95% phase-block interval "
        "excludes zero with the same sign under BOTH B and C")

    # ---------------- the norm law, fitted where physics is held fixed -------
    verdicts = {}
    for rname in ("B_mglet", "C_xiao_repaired_cubic6"):
        sc = result["scores"][rname]
        sweep = [a for a in sc if a.startswith("CTL_scale_")] + \
                [a for a in sc if a in SCALE_BASES]
        Ns = [result["source_norm"][a]["N_rms"] for a in sweep]
        Es = [sc[a]["absolute_rms"] for a in sweep]
        E0, delta, rms = affine_fit(Ns, Es)
        families = [a for a in ("M0_equilibrium", "M1_pressure_gradient", "M2_hickel",
                                "M2b_hickel_Aplus17", "M3_yang_integral", "M4_park_moin",
                                "M5_meneveau", "Xc_exact_convection",
                                "Xcp_pressure_plus_convection", "Xcpr_plus_normal_stress",
                                "Xall", "Xfull_closure_free") if a in sc]
        preds_out = {a: E0 + delta * result["source_norm"][a]["N_rms"] for a in families}
        rel = {a: abs(preds_out[a] - sc[a]["absolute_rms"]) / max(sc[a]["absolute_rms"], TINY)
               for a in families}
        null = float(np.mean([sc[a]["absolute_rms"] for a in families]))
        rel_null = {a: abs(null - sc[a]["absolute_rms"]) / max(sc[a]["absolute_rms"], TINY)
                    for a in families}
        matched = [a for a in sc if a.startswith("CTL_term_")]
        m_err = [sc[a]["absolute_rms"] for a in matched]
        spread = (max(m_err) - min(m_err)) if m_err else float("nan")
        rng = (max(Es) - min(Es)) if Es else float("nan")
        verdicts[rname] = {
            "affine_norm_law": {"E0": E0, "delta": delta, "fit_rms": rms,
                                "fitted_on": sweep},
            "out_of_sample_median_relative_error": float(np.median(list(rel.values()))),
            "zero_parameter_null_median_relative_error": float(np.median(list(rel_null.values()))),
            "per_family_relative_error": rel,
            "matched_norm_arms": matched,
            "matched_norm_absolute_rms": {a: sc[a]["absolute_rms"] for a in matched},
            "matched_norm_spread": spread,
            "amplitude_sweep_range": rng,
            "P3_ratio_spread_over_range": float(spread / rng) if rng else float("nan"),
        }
    result["norm_law"] = verdicts

    out_json = ROOT / "codes/results" / f"source_budget_tournament_l0_{args.surface}_{args.out_stamp}.json"
    out_npz = ROOT / "codes/results" / f"source_budget_tournament_l0_{args.surface}_{args.out_stamp}.npz"
    result["runtime_seconds"] = time.time() - t_start
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True))
    np.savez_compressed(out_npz, **arrays)
    print("wrote", out_json.name, out_npz.name)
    for rname in ("B_mglet", "C_xiao_repaired_cubic6"):
        print(f"--- {rname} ---")
        for a in sorted(result["scores"][rname],
                        key=lambda k: result["scores"][rname][k]["relative_rms"]):
            m = result["scores"][rname][a]
            print(f"   {a:38s} E={m['relative_rms']:9.3f}  R2={m['r2']:10.3f} "
                  f"N={result['source_norm'][a]['N_rms']:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
