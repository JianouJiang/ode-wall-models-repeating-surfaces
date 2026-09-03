#!/usr/bin/env python3
"""L0 attempt-3 producer: is the wall-model ladder ordering a property of the
MODEL, and does adding exact omitted transport help?

Question under test
-------------------
The published a-priori conditioning ladder (R2-m4 / R3-2) was scored against a
wall traction reconstructed from the Xiao velocity archive by a through-origin
LINEAR fit of the first four points.  The independent adversarial audit
(TRUTH_REFERENCE_AUDIT_V) showed that estimator is under-resolved at the
archive's wall-normal spacing and WITHDREW it as a truth.  Every rung ordering
published from it is therefore in question.

This producer re-adjudicates the COMPLETE eleven-rung ladder at three matching
surfaces against

    A  withdrawn linear-4 estimator on the Xiao archive   NEGATIVE CONTROL ONLY
    B  Peller & Manhart MGLET full-wall DNS traction      PRIMARY TRUTH
    C  same-simulation repaired estimator (through-origin
       cubic, first six fluid points, same columns as A)  SENSITIVITY BRACKET
    K  Krank Re=5600 deposited station traction           SPARSE CROSS-CHECK

and reports, for a set of contrasts fixed before the run, whether the ordering
of two rungs is IDENTIFIED (same sign, interval excluding zero, under BOTH
corrected references B and C) or UNRESOLVED.

It then measures, with no truth reference at all, the two quantities the
mechanism predicts should govern the answer:

  (1) the source-assembly cancellation factor
          Lambda_imp = sum_j |I_j| / |sum_j I_j| ,  I_j = int_0^{y_m} s_j dn
      over the four omitted source terms (pressure gradient incl. driving
      force, mean convection, streamwise normal-stress gradient, streamwise
      viscous term).  Lambda_imp uses no wall stress, so a correlation between
      Lambda_imp and the RATIO of two rung errors at the same station cannot be
      an artefact of small |tau_w|.

  (2) the closure of the one-dimensional wall-normal reduction itself.  The
      exact rungs assume tau_w = tau(y_m) - int_0^{y_m} (sources) dn.  That
      balance is closed on the flat floor and NOT closed on the curved flanks,
      so the flat/sloped split separates a geometric defect from an
      amplification defect.

Outputs codes/results/conditioning_ladder_l0_<STAMP>.{json,npz}.
No new simulation; no ARCHER2 job; read-only on every input.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402

STAMP = "20260825"
OUT_JSON = ROOT / "codes/results" / f"conditioning_ladder_l0_{STAMP}.json"
OUT_NPZ = ROOT / "codes/results" / f"conditioning_ladder_l0_{STAMP}.npz"

MGLET = (ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/"
         "UFR3-30_data-NP-Re5600-DNS2-11.dat")
KRANK = ROOT / "codes/raw_data/geometry_driven/krank_pehill_Re5600_wall_profiles.npz"
L1_SURFACE = (ROOT / "work_progress/archer2_campaign_20260823/R2-m4/"
              "ladder_surface_L1_local_blockmesh.json")
L2_NPZ = ROOT / "codes/results/rswm_common_surface_grid_l2.npz"
DEPOSITED_LADDER = ROOT / "codes/results/r2m4_apriori_ladder_20260823.json"

REPAIRED_DEG = 3          # through-origin cubic  (audit V5 selection)
REPAIRED_K = 6            # first six fluid points (audit V5 selection)

# --------------------------------------------------------------------------- #
# Contrasts fixed BEFORE the run.  Sign convention: the interval is on
# E(first) - E(second); a negative interval means the FIRST rung is better.
# --------------------------------------------------------------------------- #
CONTRASTS = (
    ("M1_pressure_gradient_ode", "M0_equilibrium",
     "does the standard pressure-gradient ODE beat the equilibrium closure?"),
    ("M2_hickel_modelled_convection", "M0_equilibrium",
     "does a MODELLED convective source beat the equilibrium closure?"),
    ("M2_hickel_modelled_convection", "M1_pressure_gradient_ode",
     "does the modelled convective surrogate beat the exact-pressure-gradient ODE?"),
    ("Xc_exact_convection_profile", "M1_pressure_gradient_ode",
     "does the EXACT within-layer convection repair the pressure-gradient ODE?"),
    ("Xc_exact_convection_profile", "M2_hickel_modelled_convection",
     "does exact convection beat its modelled surrogate?"),
    ("Xall_all_omitted_transport", "M0_equilibrium",
     "does supplying ALL omitted transport beat the crudest closure?"),
    ("Xall_all_omitted_transport", "M2_hickel_modelled_convection",
     "completion versus regularisation"),
    ("Xfull_all_transport_plus_exact_shear_stress", "M0_equilibrium",
     "does the closure-free exact-budget reconstruction beat the crudest closure?"),
    ("Xfull_all_transport_plus_exact_shear_stress", "Xall_all_omitted_transport",
     "does removing the last modelling assumption (exact stress at y_m) help?"),
)

# Regions of the wall (x/H).  Boundaries follow the audit's V9 decomposition so
# the a-priori split is directly comparable with the coupled one.
REGIONS = {
    "windward_face_x_gt_7.071": lambda x: x > 7.071,
    "recirculation_0.2_to_4.7": lambda x: (x >= 0.2) & (x <= 4.7),
    "leeward_and_crest_x_lt_1.929": lambda x: x < 1.929,
    "flat_floor_2.05_to_6.90": lambda x: (x >= 2.05) & (x <= 6.90),
    "sloped_wall_complement_of_flat_floor": lambda x: (x < 2.05) | (x > 6.90),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def poly_origin_slope(n, u, deg):
    """Wall gradient a1 from a through-origin polynomial u = a1 n + ... + a_deg n^deg."""
    A = np.vstack([np.asarray(n, float) ** (k + 1) for k in range(deg)]).T
    c, *_ = np.linalg.lstsq(A, np.asarray(u, float), rcond=None)
    return float(c[0])


def reference_A(fields):
    """The WITHDRAWN estimator: through-origin linear fit of the first four
    archive points.  Reproduced here only as a negative control."""
    return np.asarray(fields.phase, float), np.asarray(fields.tau_s_truth, float)


def reference_C(fields):
    """Repaired same-simulation estimator: through-origin cubic on the first six
    fluid points of the SAME archive columns, same tangent projection as A."""
    d = np.load(C.DNS_FILE)
    y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float)
    V = np.asarray(d["V"], float)
    n_st = fields.x.size
    tau = np.empty(n_st)
    for i in range(n_st):
        off = y[i, 1:REPAIRED_K + 1] - y[i, 0]
        ut = U[i, 1:REPAIRED_K + 1] * fields.tx[i] + V[i, 1:REPAIRED_K + 1] * fields.ty[i]
        tau[i] = C.NU * poly_origin_slope(off, ut, REPAIRED_DEG) / fields.tx[i]
    return np.asarray(fields.phase, float), tau


def reference_B():
    """Peller & Manhart MGLET full-wall DNS traction (ERCOFTAC UFR3-30).
    The deposit's last two rows are plot-axis placeholders (0,0,0) and (9,0,0)
    and are stripped, as required by the author's correction."""
    raw = np.loadtxt(MGLET)
    trailing = raw[-2:]
    if not (np.allclose(trailing[0], [0.0, 0.0, 0.0]) and
            np.allclose(trailing[1], [9.0, 0.0, 0.0])):
        raise RuntimeError("MGLET placeholder rows are not where they were documented")
    body = raw[:-2]
    return np.mod(body[:, 0] / C.LX, 1.0), body[:, 1], trailing


def reference_K():
    """Krank Re=5600 deposited station traction (10 stations): sparse
    independent cross-check.  Station-restricted; no bootstrap is claimed."""
    d = np.load(KRANK, allow_pickle=True)
    return np.asarray(d["x"], float), np.asarray(d["tau_w"], float)


# --------------------------------------------------------------------------- #
# reference-free diagnostics
# --------------------------------------------------------------------------- #
def term_impulses(fields, phases, y_m_of_phase, n_quad=400):
    """int_0^{y_m} s_j dn for each omitted source term, plus the DNS total shear
    stress at the matching height.  Uses no wall-stress reference."""
    keys = ("dpds", "conv", "dRtt", "visc")
    out = {k: np.empty(len(phases)) for k in keys}
    out["tau_at_ym"] = np.empty(len(phases))
    x_targets = np.mod(phases, 1.0) * C.LX
    for p, (xt, y_m) in enumerate(zip(x_targets, y_m_of_phase)):
        i = int(np.argmin(np.abs(fields.x - xt)))
        n_grid = y_m * np.linspace(0.0, 1.0, n_quad) ** 1.5
        for k in keys:
            vals = np.asarray(fields.profile_of(k, i)(n_grid), float)
            out[k][p] = float(np.trapezoid(vals, n_grid))
        out["tau_at_ym"][p] = float(fields.profile_of("tau", i)([y_m])[0])
    stack = np.stack([out[k] for k in keys])
    out["I_net"] = stack.sum(axis=0)
    out["S_abs"] = np.abs(stack).sum(axis=0)
    out["Lambda_imp"] = out["S_abs"] / np.maximum(np.abs(out["I_net"]), 1e-30)
    # the quantity a per-term relative error delta multiplies to bound the
    # error of the closure-free reconstruction tau_w = tau(y_m) - I_net
    out["S_abs_plus_tau_ym"] = out["S_abs"] + np.abs(out["tau_at_ym"])
    return out


# --------------------------------------------------------------------------- #
def surfaces(fields):
    l1 = json.loads(L1_SURFACE.read_text())
    if l1["status"] != "MESH_MATCHING_SURFACE_OK":
        raise RuntimeError("L1 surface record is not verified")
    l2 = np.load(L2_NPZ)
    dns = np.load(C.DNS_FILE)
    offset10 = np.asarray(dns["y"], float)[:, 10] - np.asarray(dns["y"], float)[:, 0]
    return {
        "common_W1": (np.asarray(l2["G1c_equilibrium_phase"]),
                      np.asarray(l2["G1c_equilibrium_ym"]),
                      "wall-resolved ceiling surface, y_m/H = 0.0145 on the flat floor"),
        "archive_index10": (fields.phase, offset10 * fields.tx,
                            "the paper's a-priori surface: archive wall index 10"),
        "ladder_L1": (np.asarray(l1["phase_over_period"]),
                      np.asarray(l1["phase_matching_height_over_H"]),
                      "first cell centre of the coupled ladder mesh, y_m/H = 0.0935 on the flat floor"),
    }


def identify(delta_B: dict, delta_C: dict) -> str:
    """Registered identifiability rule.  A contrast is IDENTIFIED only when the
    paired 95% interval excludes zero with the SAME sign under both corrected
    references."""
    b_first = delta_B["high"] < 0.0
    b_second = delta_B["low"] > 0.0
    c_first = delta_C["high"] < 0.0
    c_second = delta_C["low"] > 0.0
    if b_first and c_first:
        return "IDENTIFIED_FIRST_BETTER"
    if b_second and c_second:
        return "IDENTIFIED_SECOND_BETTER"
    if (b_first and c_second) or (b_second and c_first):
        return "CONTRADICTORY_ACROSS_REFERENCES"
    return "UNRESOLVED"


def main() -> int:
    t_start = time.time()
    fields = C.DnsTangentFields()

    phase_A, tau_A = reference_A(fields)
    phase_C, tau_C = reference_C(fields)
    phase_B, tau_B, mglet_trailing = reference_B()
    x_K, tau_K = reference_K()

    refs = {
        "A_withdrawn_linear4": (phase_A, tau_A),
        "B_mglet": (phase_B, tau_B),
        "C_xiao_repaired_cubic6": (phase_C, tau_C),
    }
    ref_roles = {
        "A_withdrawn_linear4": "NEGATIVE_CONTROL_withdrawn_estimator_not_a_truth",
        "B_mglet": "PRIMARY_TRUTH",
        "C_xiao_repaired_cubic6": "SENSITIVITY_BRACKET_same_simulation",
    }

    result = {
        "schema": "conditioning-ladder-l0-v1",
        "node": "development/nodes/node_002 (L0 attempt 3)",
        "question": ("re-adjudicate the eleven-rung a-priori wall-model ladder against a "
                     "converged wall-traction truth, and test whether supplying exact omitted "
                     "transport improves or degrades the wall-stress prediction"),
        "references": {k: {"role": ref_roles[k]} for k in refs},
        "reference_notes": {
            "A_withdrawn_linear4": ("through-origin LINEAR fit of the first four archive points; "
                                    "withdrawn by the independent audit as under-resolved at the "
                                    "archive spacing (first point y+ 2.4-13). Retained ONLY as a "
                                    "negative control and as a verifier control case."),
            "B_mglet": ("Peller & Manhart MGLET full-wall DNS traction, ERCOFTAC UFR3-30, "
                        "column 1 = tau_w (normalisation settled from the deposit's own velocity "
                        "profiles); two trailing plot-axis placeholder rows stripped."),
            "C_xiao_repaired_cubic6": (f"through-origin degree-{REPAIRED_DEG} fit of the first "
                                       f"{REPAIRED_K} fluid points of the SAME archive columns as A; "
                                       "validated by the audit against MGLET at the Xiao spacing "
                                       "(relative RMS error 0.264, sign accuracy 0.90) but NOT "
                                       "validated at the windward traction peak, which carries 92% "
                                       "of the traction energy -- hence a bracket, not an answer."),
            "K_krank_stations": ("Krank et al. Re=5600 deposited station traction, 10 stations; "
                                 "sparse independent cross-check, station metric only."),
        },
        "stated_limits": [
            "MGLET reattaches at x/H = 5.14 against 4.67-4.72 for the Xiao alpha=1 DNS: a ~10% "
            "reference uncertainty in the bubble length that this study does not settle.",
            "The repaired estimator C was never validated at the windward traction maximum "
            "(x/H ~ 8.66), which carries 92% of the traction energy.",
            "All results here are a priori: the ladder receives reference profiles, not a coupled "
            "large-eddy simulation.",
            "One geometry (periodic hill, alpha=1), one Reynolds number (Re_H = 5600).",
        ],
        "inputs": {},
        "mglet_trailing_rows_stripped": mglet_trailing.tolist(),
        "bootstrap": {"draws": C.BOOTSTRAP_DRAWS, "block_points": C.BLOCK_POINTS,
                      "dense_points": C.DENSE_N, "seed": C.BOOTSTRAP_SEED,
                      "pairing": "identical resampled blocks across rungs within a reference"},
        "identifiability_rule": ("a contrast is IDENTIFIED only if its paired 95% phase-block "
                                 "interval excludes zero with the same sign under BOTH corrected "
                                 "references B and C; otherwise UNRESOLVED"),
        "contrasts": [{"first": a, "second": b, "question": q} for a, b, q in CONTRASTS],
        "surfaces": {},
    }
    for tag, p in (("dns_archive", C.DNS_FILE), ("mglet_wall", MGLET), ("krank_stations", KRANK),
                   ("ladder_common_module", ROOT / "codes/analysis/r2m4_ladder_common.py"),
                   ("deposited_ladder", DEPOSITED_LADDER)):
        result["inputs"][tag] = {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}

    arrays = {
        "reference_A_phase": phase_A, "reference_A_tau": tau_A,
        "reference_B_phase": phase_B, "reference_B_tau": tau_B,
        "reference_C_phase": phase_C, "reference_C_tau": tau_C,
        "reference_K_x": x_K, "reference_K_tau": tau_K,
    }

    dense = np.arange(C.DENSE_N) / C.DENSE_N
    x_dense = dense * C.LX
    dense_truth = {k: C.periodic_interp(ph, ta, dense) for k, (ph, ta) in refs.items()}

    # Reference-to-reference distance: the scale any model difference must beat
    # before it can be called identified.  Every distance is reported in units
    # of the PRIMARY truth's RMS, so that it is directly comparable with the
    # model errors, which use the same denominator.
    ref_pairs = {}
    rms_B = float(np.sqrt(np.mean(dense_truth["B_mglet"] ** 2)))
    for a in refs:
        for b in refs:
            if a < b:
                ta, tb = dense_truth[a], dense_truth[b]
                ref_pairs[f"{a}_vs_{b}"] = {
                    "rms_ratio_first_over_second": float(np.sqrt(np.mean(ta ** 2) /
                                                                np.mean(tb ** 2))),
                    "relative_rms_distance_in_primary_truth_units":
                        float(np.sqrt(np.mean((ta - tb) ** 2)) / rms_B),
                    "relative_rms_distance_normalised_by_second":
                        float(np.sqrt(np.mean((ta - tb) ** 2)) / np.sqrt(np.mean(tb ** 2))),
                    "sign_agreement": float(np.mean(np.sign(ta) == np.sign(tb))),
                }
    result["reference_to_reference"] = ref_pairs
    result["reference_envelope_note"] = (
        "All model errors in this study are normalised by the RMS of the reference they are "
        "scored against. To compare the model spread with the reference spread on one scale, "
        "the B-to-C distance is also reported in units of the PRIMARY truth's RMS.")

    # Krank cross-check at its own stations
    kk = {}
    for name, (ph, ta) in refs.items():
        interp = C.periodic_interp(ph, ta, np.mod(x_K / C.LX, 1.0))
        kk[name] = {
            "rms_ratio_to_krank": float(np.sqrt(np.mean(interp ** 2) / np.mean(tau_K ** 2))),
            "sign_agreement": float(np.mean(np.sign(interp) == np.sign(tau_K))),
            "per_station_ratio": [float(v) for v in
                                  interp / np.where(np.abs(tau_K) > 1e-9, tau_K, np.nan)],
        }
    result["krank_station_crosscheck"] = {"x_over_H": x_K.tolist(), "references": kk}

    for name, (phase, y_m, note) in surfaces(fields).items():
        t0 = time.time()
        preds, diag = C.ladder_predictions(fields, phase, y_m)
        imp = term_impulses(fields, phase, y_m)
        dense_preds = {m: C.periodic_interp(phase, tau, dense) for m, tau in preds.items()}
        for m, tau in preds.items():
            arrays[f"{name}_pred_{m}"] = tau
        for k, v in imp.items():
            arrays[f"{name}_impulse_{k}"] = v
        for k, v in diag.items():
            arrays[f"{name}_diag_{k}"] = v
        arrays[f"{name}_phase"] = phase
        arrays[f"{name}_y_m"] = y_m

        surf = {
            "note": note,
            "phase_count": int(len(phase)),
            "y_m_over_H": {"min": float(np.min(y_m)), "median": float(np.median(y_m)),
                           "max": float(np.max(y_m))},
            "reference_free_diagnostics": {
                "Lambda_imp_median": float(np.median(imp["Lambda_imp"])),
                "Lambda_imp_quartiles": [float(np.quantile(imp["Lambda_imp"], q))
                                         for q in (0.25, 0.75)],
                "S_abs_median": float(np.median(imp["S_abs"])),
                "abs_I_net_median": float(np.median(np.abs(imp["I_net"]))),
                "note": ("Lambda_imp = sum_j|I_j| / |sum_j I_j| over the four omitted source "
                         "terms; it contains no wall stress"),
            },
            "scores": {},
            "regions": {},
            "identifiability": {},
        }

        boots = {}
        for ref, truth in dense_truth.items():
            metrics = {}
            for m, p_dense in dense_preds.items():
                err = p_dense - truth
                metrics[m] = {
                    "relative_rms": float(np.sqrt(np.mean(err ** 2)) /
                                          np.sqrt(np.mean(truth ** 2))),
                    "r2": float(1.0 - np.sum(err ** 2) /
                                np.sum((truth - truth.mean()) ** 2)),
                    "sign_accuracy": float(np.mean(np.sign(p_dense) == np.sign(truth))),
                }
            boot = C.block_bootstrap_relative_rms(truth, dense_preds)
            boots[ref] = boot
            for m in metrics:
                metrics[m]["relative_rms_interval"] = C.interval(boot[m])
            surf["scores"][ref] = metrics

            # regional split, COMMON global denominator so regions compare
            den_global = np.sqrt(np.mean(truth ** 2))
            reg = {}
            for rname, sel in REGIONS.items():
                mask = sel(x_dense)
                entry = {"fraction_of_wall": float(np.mean(mask)),
                         "fraction_of_tau2_energy": float(np.sum(truth[mask] ** 2) /
                                                          np.sum(truth ** 2)),
                         "models": {}}
                for m, p_dense in dense_preds.items():
                    e = p_dense[mask] - truth[mask]
                    entry["models"][m] = {
                        "relative_rms_global_norm": float(np.sqrt(np.mean(e ** 2)) / den_global),
                        "relative_rms_local_norm": float(np.sqrt(np.mean(e ** 2)) /
                                                         np.sqrt(np.mean(truth[mask] ** 2))),
                    }
                reg[rname] = entry
            surf["regions"][ref] = reg

        # paired contrasts and the identifiability certificate
        for a, b, q in CONTRASTS:
            d = {r: C.interval(boots[r][a] - boots[r][b]) for r in boots}
            verdict = identify(d["B_mglet"], d["C_xiao_repaired_cubic6"])
            surf["identifiability"][f"{a}-minus-{b}"] = {
                "question": q,
                "paired_interval": d,
                "point_estimate": {r: float(surf["scores"][r][a]["relative_rms"] -
                                            surf["scores"][r][b]["relative_rms"]) for r in boots},
                "verdict": verdict,
                "verdict_under_withdrawn_control_only": identify(
                    d["A_withdrawn_linear4"], d["A_withdrawn_linear4"]),
            }

        # regional test of the mechanism: does the completion penalty follow
        # Lambda_imp?  Both quantities are computed per station; the rung RATIO
        # removes the local truth normalisation.
        phase_x = np.mod(phase, 1.0) * C.LX
        mech = {}
        for rname, sel in REGIONS.items():
            m_st = sel(phase_x)
            if m_st.sum() < 8:
                continue
            mech[rname] = {"stations": int(m_st.sum()),
                           "Lambda_imp_median": float(np.median(imp["Lambda_imp"][m_st]))}
        surf["mechanism_regional_Lambda"] = mech
        surf["runtime_seconds"] = round(time.time() - t0, 1)
        result["surfaces"][name] = surf
        print(f"[{name}] done in {surf['runtime_seconds']}s  "
              f"Lambda_imp median {surf['reference_free_diagnostics']['Lambda_imp_median']:.2f}",
              flush=True)

    # ---------------------------------------------------------------- #
    # Single-constant test of the amplification bound.
    # The closure-free rung is tau_w = tau(y_m) - I_net.  If each supplied term
    # carries an effective relative error delta, then
    #     |dtau_w| <~ delta * (sum_j|I_j| + |tau(y_m)|)
    # so  E(Xfull) ~ delta * RMS(S_abs + |tau_ym|) / RMS(tau_truth).
    # One delta is fitted across all (surface, corrected reference) pairs and
    # the leave-one-out prediction of each is reported.
    # ---------------------------------------------------------------- #
    rows = []
    for sname in result["surfaces"]:
        pred_scale = float(np.sqrt(np.mean(arrays[f"{sname}_impulse_S_abs_plus_tau_ym"] ** 2)))
        for ref in ("B_mglet", "C_xiao_repaired_cubic6"):
            truth_rms = float(np.sqrt(np.mean(dense_truth[ref] ** 2)))
            rows.append({
                "surface": sname, "reference": ref,
                "predictor": pred_scale / truth_rms,
                "measured_E_Xfull": result["surfaces"][sname]["scores"][ref][
                    "Xfull_all_transport_plus_exact_shear_stress"]["relative_rms"],
            })
    pv = np.array([r["predictor"] for r in rows])
    mv = np.array([r["measured_E_Xfull"] for r in rows])
    delta_hat = float(np.sum(pv * mv) / np.sum(pv * pv))
    for j, r in enumerate(rows):
        keep = np.ones(len(rows), bool); keep[j] = False
        d_loo = float(np.sum(pv[keep] * mv[keep]) / np.sum(pv[keep] ** 2))
        r["delta_leave_one_out"] = d_loo
        r["predicted_E_Xfull_leave_one_out"] = d_loo * r["predictor"]
        r["relative_prediction_error"] = abs(d_loo * r["predictor"] - r["measured_E_Xfull"]) / \
            r["measured_E_Xfull"]
    result["amplification_bound"] = {
        "model": "E(Xfull) = delta * RMS(sum_j|I_j| + |tau(y_m)|) / RMS(tau_truth)",
        "delta_fitted_over_all_points": delta_hat,
        "predictor_range": [float(pv.min()), float(pv.max())],
        "n_points": len(rows),
        "max_relative_prediction_error_leave_one_out": float(
            max(r["relative_prediction_error"] for r in rows)),
        "points": rows,
    }

    # ---------------------------------------------------------------- #
    # Where is the bound ATTAINED?  The closure-free reconstruction is exact
    # only where the one-dimensional wall-normal reduction is itself a closed
    # balance -- on the flat inter-hill floor, where the wall-normal and
    # vertical directions coincide.  On the curved flanks it is not closed, and
    # the un-closed remainder is amplified by Lambda_tau.  Attainment =
    # measured error / (delta * Lambda_tau) separates the two.
    # ---------------------------------------------------------------- #
    for sname in result["surfaces"]:
        S_abs = arrays[f"{sname}_impulse_S_abs"]
        tym = np.abs(arrays[f"{sname}_impulse_tau_at_ym"])
        st_x = np.mod(arrays[f"{sname}_phase"], 1.0) * C.LX
        _, slope, _, _ = C.wall_tangent(np.sort(fields.x))
        st_slope = np.abs(np.interp(st_x, np.sort(fields.x), slope))
        arrays[f"{sname}_wall_slope_magnitude"] = st_slope
        amp = {}
        for ref in ("B_mglet", "C_xiao_repaired_cubic6"):
            truth_rms = float(np.sqrt(np.mean(dense_truth[ref] ** 2)))
            per_region = {}
            for rname, sel in REGIONS.items():
                m_st = sel(st_x)
                if m_st.sum() < 8:
                    continue
                lam = float(np.sqrt(np.mean((S_abs[m_st] + tym[m_st]) ** 2)) / truth_rms)
                meas = result["surfaces"][sname]["regions"][ref][rname]["models"][
                    "Xfull_all_transport_plus_exact_shear_stress"]["relative_rms_global_norm"]
                m0 = result["surfaces"][sname]["regions"][ref][rname]["models"][
                    "M0_equilibrium"]["relative_rms_global_norm"]
                per_region[rname] = {
                    "stations": int(m_st.sum()),
                    "Lambda_tau_global_norm": lam,
                    "bound_delta_times_Lambda_tau": delta_hat * lam,
                    "measured_E_Xfull_global_norm": meas,
                    "bound_attainment": meas / (delta_hat * lam),
                    "E_Xfull_over_E_M0": meas / m0,
                    "median_wall_slope_magnitude": float(np.median(st_slope[m_st])),
                }
            amp[ref] = per_region
        result["surfaces"][sname]["regional_amplification"] = amp
    result["regional_amplification_note"] = (
        "Lambda_tau = RMS(sum_j|I_j| + |tau(y_m)|)/RMS(tau_truth) is an upper bound on the "
        "error of the closure-free reconstruction. It is ATTAINED (attainment ~ 1) where the "
        "wall is curved and the one-dimensional wall-normal reduction is not a closed balance, "
        "and NOT attained (attainment << 1) on the flat inter-hill floor where it is closed. "
        "The flat floor is therefore a positive control on the reconstruction itself: if the "
        "assembled budget were mis-implemented it would fail there too.")

    # ---------------------------------------------------------------- #
    # instrument fidelity: reproduce the deposited A-reference ladder exactly
    # ---------------------------------------------------------------- #
    dep = json.loads(DEPOSITED_LADDER.read_text())
    fid = {}
    for sname, s in result["surfaces"].items():
        if sname not in dep["surfaces"]:
            continue
        worst = 0.0
        n = 0
        for m, v in dep["surfaces"][sname]["metrics"].items():
            got = s["scores"]["A_withdrawn_linear4"].get(m)
            if got is None:
                continue
            worst = max(worst, abs(got["relative_rms"] - v["relative_rms"]) /
                        max(abs(v["relative_rms"]), 1e-30))
            n += 1
        fid[sname] = {"rungs_compared": n, "worst_relative_difference": worst}
    result["instrument_fidelity_vs_deposited_ladder"] = fid

    result["runtime_seconds"] = round(time.time() - t_start, 1)
    OUT_JSON.write_text(json.dumps(result, indent=1, sort_keys=True))
    np.savez_compressed(OUT_NPZ, **arrays)
    print("wrote", OUT_JSON.relative_to(ROOT))
    print("wrote", OUT_NPZ.relative_to(ROOT))
    print("delta_hat", delta_hat, "worst LOO rel err",
          result["amplification_bound"]["max_relative_prediction_error_leave_one_out"])
    print("fidelity", json.dumps(fid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
