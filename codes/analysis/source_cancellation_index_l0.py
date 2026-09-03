#!/usr/bin/env python3
"""The source a wall model integrates is itself a near-cancelling assembly.

The tournament and the localisation probe together show that, at exactly matched
assembled norm, a smooth single-signed source -- the published parametrised
surrogate, or even a uniform one -- outperforms the measured transport, and that
near-wall localisation is not what makes the difference.  This producer measures
the property that does differ, with no wall-stress reference and no wall model:

    K = N / |W|,

the ratio of the assembled magnitude of a source to its net traction-equivalent
work, both already defined in the paper.  ``K = 1`` for a source that never
changes sign in the wall layer; ``K`` grows without bound as the source becomes
an assembly of large contributions that cancel.  It is the same cancellation
index the paper applies to the wall-momentum balance, applied one level down --
to the source term the one-dimensional model is asked to integrate.

Also reported, per station: the number of sign changes of the source across the
wall layer, and the same statistics for each measured term separately.

No new simulation, no remote job, read-only on every input.
"""
from __future__ import annotations

import argparse
import json
import sys
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
TINY = 1.0e-30
SURFACES = ("ladder_L1", "archive_index10")


def sign_changes(values: np.ndarray, floor: float) -> int:
    keep = values[np.abs(values) > floor]
    if keep.size < 2:
        return 0
    return int(np.sum(np.diff(np.sign(keep)) != 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stamp", default=STAMP)
    args = ap.parse_args()
    fields = C.DnsTangentFields()
    surf = CL.surfaces(fields)
    xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
    result = {
        "schema": "source_cancellation_index_l0/1",
        "question": ("is the source that a one-dimensional wall model is asked "
                     "to integrate itself a near-cancelling assembly, and does "
                     "that distinguish the sources that work from those that do "
                     "not?"),
        "definition": ("K = N/|W| with N the assembled source norm and W its net "
                       "traction-equivalent work; K = 1 for a source of one sign "
                       "and grows as the source becomes an assembly of cancelling "
                       "contributions. Neither N nor W uses a wall-stress "
                       "reference."),
        "inputs": {"dns_archive": {"path": str(C.DNS_FILE.relative_to(ROOT)),
                                   "sha256": C.sha256(C.DNS_FILE)}},
        "surfaces": {},
    }
    for sname in SURFACES:
        phases, y_m_of_phase, note = surf[sname]
        phases = np.asarray(phases, float)
        y_m_all = np.asarray(y_m_of_phase, float)
        n_st = phases.size
        arms = ("M1_pressure_gradient", "M2_hickel", "Xc_exact_convection",
                "Xcp_pressure_plus_convection", "Xall",
                "term_dpds", "term_conv", "term_dRtt", "term_visc")
        index = {a: np.full(n_st, np.nan) for a in arms}
        crossings = {a: np.zeros(n_st) for a in arms}
        for p, ph in enumerate(phases):
            i = int(np.argmin(np.abs(fields.x - ph * C.LX)))
            y_m = float(y_m_all[p])
            grid = y_m * xi
            u_m, _, _ = fields.station(i, y_m)
            tau0 = wm.spalding_wall_stress(u_m, y_m, C.NU) if abs(u_m) > 1e-12 else 0.0
            D = fw.equilibrium_diffusivity(grid, tau0, C.NU)
            G = float(np.trapezoid(1.0 / D, grid))
            dpds = float(fields.dpds_total[i])
            terms = {k: np.asarray(fields.profile_of(k, i)(grid), float)
                     for k in ("dpds", "conv", "dRtt", "visc")}
            candidates = {
                "M1_pressure_gradient": np.full(N_QUAD, dpds),
                "M2_hickel": wm.hickel_source(grid, dpds, C.NU),
                "Xc_exact_convection": np.full(N_QUAD, dpds) + terms["conv"],
                "Xcp_pressure_plus_convection": terms["dpds"] + terms["conv"],
                "Xall": sum(terms.values()),
                "term_dpds": terms["dpds"], "term_conv": terms["conv"],
                "term_dRtt": terms["dRtt"], "term_visc": terms["visc"],
            }
            for a, values in candidates.items():
                N, W = fw.assembled_source_norm(grid, D, G, values)
                index[a][p] = N / max(abs(W), TINY)
                crossings[a][p] = sign_changes(values,
                                               0.01 * float(np.max(np.abs(values))))
        entry = {"note": note, "stations": int(n_st), "arms": {}}
        for a in arms:
            finite = index[a][np.isfinite(index[a])]
            entry["arms"][a] = {
                "cancellation_index_median": float(np.median(finite)),
                "cancellation_index_p90": float(np.quantile(finite, 0.9)),
                "cancellation_index_max": float(np.max(finite)),
                "fraction_of_stations_above_2": float(np.mean(finite > 2.0)),
                "sign_changes_median": float(np.median(crossings[a])),
                "fraction_of_stations_with_a_sign_change":
                    float(np.mean(crossings[a] > 0)),
            }
        result["surfaces"][sname] = entry
        print(f"--- {sname} ({n_st} stations) ---")
        for a in arms:
            record = entry["arms"][a]
            print(f"   {a:32s} K median {record['cancellation_index_median']:8.3f} "
                  f"p90 {record['cancellation_index_p90']:9.3f}  "
                  f"sign-changing stations "
                  f"{100 * record['fraction_of_stations_with_a_sign_change']:5.1f}%")
    out = ROOT / "codes/results" / f"source_cancellation_index_l0_{args.out_stamp}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
