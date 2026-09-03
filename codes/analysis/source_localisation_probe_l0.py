#!/usr/bin/env python3
"""Registered follow-up: is it LOCALISATION that makes the surviving source work?

The factorial in ``faithful_tournament_l0.py`` shows that at exactly matched
assembled norm the measured transport is a much worse source than the published
parametrised surrogate.  The surrogate differs from the measured source in two
ways at once: it is smaller, and it is confined to a thin near-wall layer of
height ``y_pg = 4 (nu^2/|dp/ds|)^(1/3)``, above which it is identically zero.
The factorial removes the first difference.  This producer removes the second.

Four arms, all on the same stations, the same truth protocol and the same
uncertainty protocol as the tournament:

  LOC_measured_below_ypg
      the MEASURED source, kept in full below the surrogate's own near-wall
      length and dropped above it;
  LOC_measured_below_ypg_at_modelled_norm
      the same, then rescaled to the surrogate's assembled norm, so that
      localisation is the ONLY remaining difference;
  LOC_modelled_spread_to_matching_height
      the reverse control: the surrogate's source de-localised -- spread
      uniformly over the whole wall layer at its own assembled norm;
  and the surrogate and the measured source themselves, recomputed here so that
  every comparison is within one run.

Registered before the run:

  If localisation is what makes the surrogate work, then
  LOC_measured_below_ypg_at_modelled_norm must be identifiably better than the
  measured source at the modelled norm, and the de-localised surrogate must be
  identifiably worse than the surrogate.  If instead the surrogate's specific
  profile is what matters, the localised measured source stays worse than the
  surrogate and the result is that localisation is necessary but not
  sufficient.  A paired interval that straddles zero is UNRESOLVED, not a null.

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
TINY = 1.0e-30
SURFACES = ("ladder_L1", "archive_index10")

PREREGISTERED = {
    "L1_localisation_repairs_the_measured_source": (
        "E(measured source localised to y_pg, at the modelled norm) is "
        "identifiably LOWER than E(measured source at the modelled norm) under "
        "both corrected references."),
    "L2_delocalisation_breaks_the_surrogate": (
        "E(surrogate spread uniformly over the wall layer, same norm) is "
        "identifiably HIGHER than E(surrogate) under both corrected references."),
    "L3_localisation_is_sufficient": (
        "E(measured source localised to y_pg, at the modelled norm) is not "
        "identifiably higher than E(surrogate). If it IS identifiably higher, "
        "localisation is necessary but not sufficient and the surrogate's "
        "specific near-wall profile carries the remainder."),
}


def evaluate(fields, phases, y_m_of_phase, log=print):
    n_st = len(phases)
    xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
    names = ("M0_equilibrium", "M2_hickel", "Xall",
             "FAC_exactshape_modelnorm",
             "LOC_measured_below_ypg",
             "LOC_measured_below_ypg_at_modelled_norm",
             "LOC_modelled_spread_to_matching_height")
    pred = {a: np.full(n_st, np.nan) for a in names}
    norm = {a: np.full(n_st, np.nan) for a in names}
    horizon_fraction = np.full(n_st, np.nan)
    x_targets = np.mod(np.asarray(phases, float), 1.0) * C.LX
    t0 = time.time()
    for p, (xt, y_m) in enumerate(zip(x_targets, y_m_of_phase)):
        i = int(np.argmin(np.abs(fields.x - xt)))
        y_m = float(y_m)
        u_m, _, _ = fields.station(i, y_m)
        dpds = float(fields.dpds_total[i])
        tau0 = wm.spalding_wall_stress(u_m, y_m, C.NU) if abs(u_m) > 1e-12 else 0.0
        n_grid = y_m * xi
        D = fw.equilibrium_diffusivity(n_grid, tau0, C.NU)
        G = float(np.trapezoid(1.0 / D, n_grid))
        measured = sum(np.asarray(fields.profile_of(k, i)(n_grid), float)
                       for k in ("dpds", "conv", "dRtt", "visc"))
        modelled = wm.hickel_source(n_grid, dpds, C.NU)
        # the surrogate's own near-wall length, from its published definition
        y_pg = (4.0 * (C.NU * C.NU / abs(dpds)) ** (1.0 / 3.0)
                if abs(dpds) > TINY else y_m)
        horizon_fraction[p] = min(y_pg / y_m, 1.0)
        localised = np.where(n_grid <= y_pg, measured, 0.0)
        n_model, _ = fw.assembled_source_norm(n_grid, D, G, modelled)
        n_measured, _ = fw.assembled_source_norm(n_grid, D, G, measured)
        n_localised, _ = fw.assembled_source_norm(n_grid, D, G, localised)
        arms = {
            "M2_hickel": (modelled, wm.HICKEL_VAN_DRIEST_A),
            "Xall": (measured, wm.VAN_DRIEST_A),
            "LOC_measured_below_ypg": (localised, wm.VAN_DRIEST_A),
        }
        if n_measured > TINY and n_model > TINY:
            arms["FAC_exactshape_modelnorm"] = (
                measured * (n_model / n_measured), wm.VAN_DRIEST_A)
        if n_localised > TINY and n_model > TINY:
            arms["LOC_measured_below_ypg_at_modelled_norm"] = (
                localised * (n_model / n_localised), wm.VAN_DRIEST_A)
        # de-localised surrogate: uniform over the layer at the surrogate's norm
        uniform = np.full(N_QUAD, np.sign(dpds) if dpds != 0.0 else 0.0)
        n_uniform, _ = fw.assembled_source_norm(n_grid, D, G, uniform)
        if n_uniform > TINY and n_model > TINY:
            arms["LOC_modelled_spread_to_matching_height"] = (
                uniform * (n_model / n_uniform), wm.HICKEL_VAN_DRIEST_A)
        pred["M0_equilibrium"][p] = tau0
        norm["M0_equilibrium"][p] = 0.0
        for a, (vals, a_plus) in arms.items():
            norm[a][p], _ = fw.assembled_source_norm(n_grid, D, G, vals)
            src = (lambda v: (lambda y: np.interp(np.asarray(y, float), n_grid, v)))(vals)
            pred[a][p] = wm.shoot_wall_stress(
                u_m, y_m, C.NU, src, continuation_tau=tau0, n_points=N_SHOOT,
                a_plus=a_plus).tau_w
        if p % 128 == 0:
            log(f"  station {p + 1}/{n_st}")
    log(f"  {time.time() - t0:.0f}s")
    return pred, norm, horizon_fraction


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stamp", default=STAMP)
    args = ap.parse_args()
    t_start = time.time()

    fields = C.DnsTangentFields()
    surf = CL.surfaces(fields)
    phase_A, tau_A = CL.reference_A(fields)
    phase_C, tau_C = CL.reference_C(fields)
    phase_B, tau_B, trailing = CL.reference_B()
    refs = {"A_withdrawn_linear4": (phase_A, tau_A),
            "B_mglet": (phase_B, tau_B),
            "C_xiao_repaired_cubic6": (phase_C, tau_C)}

    result = {
        "schema": "source_localisation_probe_l0/1",
        "question": ("is it the near-wall LOCALISATION of the surviving source, "
                     "rather than its magnitude or its physical content, that "
                     "makes the parametrised surrogate work over a repeating "
                     "curved wall?"),
        "preregistered_predictions": PREREGISTERED,
        "references": {"A_withdrawn_linear4": "NEGATIVE_CONTROL",
                       "B_mglet": "PRIMARY_TRUTH",
                       "C_xiao_repaired_cubic6": "SENSITIVITY_BRACKET"},
        "inputs": {
            "dns_archive": {"path": str(C.DNS_FILE.relative_to(ROOT)),
                            "sha256": C.sha256(C.DNS_FILE)},
            "faithful_models": {"path": "codes/models/faithful_wall_models_l0.py",
                                "sha256": C.sha256(ROOT / "codes/models/faithful_wall_models_l0.py")},
        },
        "mglet_trailing_rows_stripped": np.asarray(trailing).tolist(),
        "bootstrap": {"block_points": C.BLOCK_POINTS, "dense_points": C.DENSE_N,
                      "draws": C.BOOTSTRAP_DRAWS, "seed": C.BOOTSTRAP_SEED},
        "surfaces": {},
    }
    arrays = {}
    for sname in SURFACES:
        phases, y_m_of_phase, note = surf[sname]
        phases = np.asarray(phases, float)
        y_m_of_phase = np.asarray(y_m_of_phase, float)
        print(f"surface {sname}: {len(phases)} stations")
        pred, norm, horizon = evaluate(fields, phases, y_m_of_phase)
        arms = [a for a in pred if np.isfinite(pred[a]).any()]
        dense = np.arange(C.DENSE_N) / C.DENSE_N
        entry = {"note": note, "stations": int(len(phases)),
                 "surrogate_horizon_over_matching_height": {
                     "median": float(np.nanmedian(horizon)),
                     "min": float(np.nanmin(horizon)),
                     "max": float(np.nanmax(horizon))},
                 "scores": {}, "source_norm": {}, "contrasts": []}
        intervals = {}
        for rname, (rp, rt) in refs.items():
            truth = C.periodic_interp(rp, rt, dense)
            preds_dense = {}
            for a in arms:
                ok = np.isfinite(pred[a])
                preds_dense[a] = C.periodic_interp(phases[ok], pred[a][ok], dense)
            boots = C.block_bootstrap_relative_rms(truth, preds_dense)
            intervals[rname] = boots
            entry["scores"][rname] = {}
            for a in arms:
                err = preds_dense[a] - truth
                ss_tot = float(np.sum((truth - truth.mean()) ** 2))
                entry["scores"][rname][a] = {
                    "relative_rms": float(np.sqrt(np.mean(err ** 2))
                                          / np.sqrt(np.mean(truth ** 2))),
                    "absolute_rms": float(np.sqrt(np.mean(err ** 2))),
                    "r2": float(1.0 - np.sum(err ** 2) / ss_tot),
                    "interval": C.interval(boots[a]),
                }
        for a in arms:
            finite = norm[a][np.isfinite(norm[a])]
            entry["source_norm"][a] = (float(np.sqrt(np.mean(finite ** 2)))
                                       if finite.size else None)

        def add(first, second, question):
            delta = {}
            for rname in refs:
                b = intervals[rname]
                if first not in b or second not in b:
                    return
                delta[rname] = C.interval(b[first] - b[second])
            entry["contrasts"].append({
                "first": first, "second": second, "question": question,
                "delta": delta,
                "identified": CL.identify(delta["B_mglet"],
                                          delta["C_xiao_repaired_cubic6"]),
            })

        add("LOC_measured_below_ypg_at_modelled_norm", "FAC_exactshape_modelnorm",
            "does localisation repair the measured source at fixed norm?")
        add("LOC_modelled_spread_to_matching_height", "M2_hickel",
            "does de-localising the surrogate break it at fixed norm?")
        add("LOC_measured_below_ypg_at_modelled_norm", "M2_hickel",
            "is localisation sufficient to reach the surrogate?")
        add("LOC_measured_below_ypg", "Xall",
            "does localisation alone repair the measured source?")
        add("LOC_measured_below_ypg_at_modelled_norm", "M0_equilibrium",
            "is the localised measured source better than no source at all?")
        result["surfaces"][sname] = entry
        arrays[f"{sname}__phase"] = phases
        arrays[f"{sname}__horizon_over_ym"] = horizon
        for a in arms:
            arrays[f"{sname}__pred__{a}"] = pred[a]
            arrays[f"{sname}__norm__{a}"] = norm[a]

    prim = result["surfaces"]["archive_index10"]

    def verdict_of(first, second, want_lower):
        record = next((c for c in prim["contrasts"]
                       if c["first"] == first and c["second"] == second), None)
        if record is None:
            return {"verdict": "MISSING"}
        identified = record["identified"]
        if want_lower:
            outcome = ("SUPPORTED" if identified == "IDENTIFIED_FIRST_BETTER"
                       else "REFUTED" if identified == "IDENTIFIED_SECOND_BETTER"
                       else "UNRESOLVED")
        else:
            outcome = ("SUPPORTED" if identified == "IDENTIFIED_SECOND_BETTER"
                       else "REFUTED" if identified == "IDENTIFIED_FIRST_BETTER"
                       else "UNRESOLVED")
        return {"verdict": outcome, "identified": identified,
                "delta": record["delta"]}

    result["registered_verdicts"] = {
        "L1_localisation_repairs_the_measured_source": verdict_of(
            "LOC_measured_below_ypg_at_modelled_norm",
            "FAC_exactshape_modelnorm", True),
        "L2_delocalisation_breaks_the_surrogate": verdict_of(
            "LOC_modelled_spread_to_matching_height", "M2_hickel", False),
        "L3_localisation_is_sufficient": verdict_of(
            "LOC_measured_below_ypg_at_modelled_norm", "M2_hickel", True),
    }
    result["runtime_seconds"] = time.time() - t_start
    out_json = ROOT / "codes/results" / f"source_localisation_probe_l0_{args.out_stamp}.json"
    out_npz = ROOT / "codes/results" / f"source_localisation_probe_l0_{args.out_stamp}.npz"
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    np.savez_compressed(out_npz, **arrays)
    print("wrote", out_json.name)
    for sname, entry in result["surfaces"].items():
        sc = entry["scores"]["B_mglet"]
        print(f"--- {sname} (B_mglet) ---")
        for a in sorted(sc, key=lambda k: sc[k]["relative_rms"]):
            print(f"   {a:44s} E={sc[a]['relative_rms']:8.3f} R2={sc[a]['r2']:9.3f}")
    for key, record in result["registered_verdicts"].items():
        print(f"{key}: {record['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
