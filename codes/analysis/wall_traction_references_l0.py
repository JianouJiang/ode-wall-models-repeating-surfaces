#!/usr/bin/env python3
"""Publish the periodic-hill wall-traction REFERENCES as one hash-bound artifact.

Every rendered figure and table that needs a wall traction to score against must
read this file and nothing else.  Three judges independently found rendered
figures still drawing their orange truth curve through the withdrawn four-point
through-origin estimator, because each producer rebuilt its own reference from
the raw archive.  A single published artifact makes the reference of a figure a
decidable property of the file it opens.

Contents (all on the periodic phase x/L_x of the Re_H = 5,600 hill):

  A_withdrawn_linear4   through-origin LINEAR fit of the first four archive
                        points.  WITHDRAWN: at the archive's wall spacing
                        (fit points at y+ 2.4-44) that fit under-resolves the
                        wall gradient.  Retained ONLY as a negative control and
                        flagged as such in `roles`.
  B_mglet               Peller & Manhart full-wall DNS traction (ERCOFTAC
                        UFR3-30).  PRIMARY.  The deposit's last two rows are
                        plot-axis placeholders and are stripped.
  C_xiao_repaired_cubic through-origin cubic on the first six fluid points of
                        the SAME archive columns as A.  SENSITIVITY BRACKET.
  K_krank_stations      Krank et al. ten-station traction.  Sparse independent
                        cross-check; station metric only, no bootstrap.

The producer is a thin, verified wrapper: the four reference constructions are
imported from conditioning_ladder_l0, so this file cannot drift from the
audited definitions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402
import conditioning_ladder_l0 as CL  # noqa: E402

STAMP = "20260825"
OUT_NPZ = ROOT / "codes/results" / f"wall_traction_references_{STAMP}.npz"
OUT_JSON = ROOT / "codes/results" / f"wall_traction_references_{STAMP}_summary.json"

ROLES = {
    "A_withdrawn_linear4": "WITHDRAWN_NEGATIVE_CONTROL_ONLY",
    "B_mglet": "PRIMARY_TRUTH",
    "C_xiao_repaired_cubic6": "SENSITIVITY_BRACKET_same_simulation",
    "K_krank_stations": "SPARSE_INDEPENDENT_CROSS_CHECK",
}


def main() -> int:
    fields = C.DnsTangentFields()
    phase_A, tau_A = CL.reference_A(fields)
    phase_C, tau_C = CL.reference_C(fields)
    phase_B, tau_B, trailing = CL.reference_B()
    x_K, tau_K = CL.reference_K()
    phase_K = np.mod(np.asarray(x_K, float) / C.LX, 1.0)

    arrays = {
        "A_withdrawn_linear4__phase": np.asarray(phase_A, float),
        "A_withdrawn_linear4__tau": np.asarray(tau_A, float),
        "B_mglet__phase": np.asarray(phase_B, float),
        "B_mglet__tau": np.asarray(tau_B, float),
        "C_xiao_repaired_cubic6__phase": np.asarray(phase_C, float),
        "C_xiao_repaired_cubic6__tau": np.asarray(tau_C, float),
        "K_krank_stations__phase": phase_K,
        "K_krank_stations__tau": np.asarray(tau_K, float),
        "K_krank_stations__x_over_H": np.asarray(x_K, float),
    }
    np.savez_compressed(OUT_NPZ, **arrays)

    def rms(v):
        return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))

    summary = {
        "schema": "wall_traction_references/1",
        "roles": ROLES,
        "withdrawn_keys": ["A_withdrawn_linear4__tau"],
        "why_A_is_withdrawn": (
            "a through-origin linear fit of the first four archive points "
            "under-resolves the wall gradient at the archive's wall spacing; "
            "the archive's velocity data are not in question"),
        "inputs": {
            "dns_archive": {"path": str(C.DNS_FILE.relative_to(ROOT)),
                            "sha256": C.sha256(C.DNS_FILE)},
            "mglet_wall": {"path": str(CL.MGLET.relative_to(ROOT)),
                           "sha256": C.sha256(CL.MGLET)},
            "krank_stations": {"path": str(CL.KRANK.relative_to(ROOT)),
                               "sha256": C.sha256(CL.KRANK)},
            "definition_module": {"path": "codes/analysis/conditioning_ladder_l0.py",
                                  "sha256": C.sha256(ROOT / "codes/analysis/conditioning_ladder_l0.py")},
        },
        "mglet_trailing_rows_stripped": np.asarray(trailing).tolist(),
        "rms": {k.split("__")[0]: rms(arrays[k]) for k in arrays if k.endswith("__tau")},
        "stations": {k.split("__")[0]: int(arrays[k].size) for k in arrays if k.endswith("__tau")},
        "reference_to_reference_rms_ratio": {
            "A_over_B": rms(tau_A) / rms(tau_B),
            "C_over_B": rms(tau_C) / rms(tau_B),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary["rms"], indent=1))
    print("wrote", OUT_NPZ.name, OUT_JSON.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
