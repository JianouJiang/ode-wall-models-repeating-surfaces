#!/usr/bin/env python3
"""Is the integral wall model's system SOLVABLE over this wall? A solver-free test.

The tournament finds that the integral model's coupled profile/moment system
loses its solution branch under continuation.  That is a statement obtained with
a solver, and a referee is entitled to ask whether it is a property of the model
or of the iteration.  This producer answers without any iteration at all.

Evaluate the model's own momentum equation AT THE ANSWER.  Take the reference
wall traction, build the assumed profile that (i) has that friction velocity and
(ii) matches the measured velocity at the matching height -- the profile the
model would have to produce if it were right -- and measure the residual of the
vertically integrated momentum equation it is then required to satisfy:

    R = tau_w + d/ds Int(U^2) - U_m d/ds Int(U) + Int(dp/ds) - tau(y_m).

If ``|R|`` is small compared with the wall traction, the model can represent the
flow and any failure is the solver's.  If ``|R|`` is large compared with the
wall traction, no choice of the model's two parameters can satisfy both of its
equations, and the family is inapplicable here as a matter of algebra.

The same quantity is evaluated with the TRUE profile in place of the assumed
one, which must return zero to the accuracy of the differentiation: that is the
control which shows the residual measures the closure, not the numerics.

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
import faithful_wall_models_l0 as fw  # noqa: E402

STAMP = "20260825"
N_QUAD = 400
SURFACES = ("ladder_L1", "archive_index10")


def arc_of(fields, phases):
    ds_dx = np.sqrt(1.0 + fields.slope ** 2)
    s_full = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1])
                                              * np.diff(fields.x))))
    period = float(s_full[-1] + 0.5 * (ds_dx[-1] + ds_dx[0]) * fields.dx)
    ph = np.mod((fields.x - fields.x.min()) / C.LX, 1.0)
    order = np.argsort(ph)
    return np.interp(np.mod(np.asarray(phases, float), 1.0), ph[order],
                     s_full[order]), period


def ddx(values, arc, period):
    forward = np.roll(arc, -1).copy()
    forward[-1] += period
    backward = np.roll(arc, 1).copy()
    backward[0] -= period
    return (np.roll(values, -1) - np.roll(values, 1)) / (forward - backward)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stamp", default=STAMP)
    args = ap.parse_args()
    fields = C.DnsTangentFields()
    surf = CL.surfaces(fields)
    phase_B, tau_B, trailing = CL.reference_B()
    phase_C, tau_C = CL.reference_C(fields)

    result = {
        "schema": "integral_model_consistency_l0/1",
        "question": ("can the integral wall model's assumed profile satisfy its "
                     "own matching and momentum equations simultaneously over a "
                     "repeating curved wall?"),
        "method": ("the residual of the vertically integrated momentum equation "
                   "is evaluated at the REFERENCE wall traction with the assumed "
                   "profile matched to the measured velocity -- no iteration, no "
                   "solver, no initial guess"),
        "inputs": {"dns_archive": {"path": str(C.DNS_FILE.relative_to(ROOT)),
                                   "sha256": C.sha256(C.DNS_FILE)},
                   "mglet_wall": {"path": str(CL.MGLET.relative_to(ROOT)),
                                  "sha256": C.sha256(CL.MGLET)}},
        "mglet_trailing_rows_stripped": np.asarray(trailing).tolist(),
        "surfaces": {},
    }
    for sname in SURFACES:
        phases, y_m_of_phase, note = surf[sname]
        phases = np.asarray(phases, float)
        y_m = np.asarray(y_m_of_phase, float)
        n_st = phases.size
        arc, period = arc_of(fields, phases)
        xi = np.linspace(0.0, 1.0, N_QUAD) ** 1.5
        u_m = np.empty(n_st)
        tau_at = np.empty(n_st)
        pressure = np.empty(n_st)
        true_first = np.empty(n_st)
        true_second = np.empty(n_st)
        for p, ph in enumerate(phases):
            i = int(np.argmin(np.abs(fields.x - ph * C.LX)))
            height = float(y_m[p])
            grid = height * xi
            u_m[p], _, _ = fields.station(i, height)
            tau_at[p] = float(fields.profile_of("tau", i)([height])[0])
            pressure[p] = float(np.trapezoid(
                np.asarray(fields.profile_of("dpds", i)(grid), float), grid))
            points = fields._normal_points(i, grid)
            u_x, u_y = (f(points) for f in fields._interp("U"))
            profile = u_x * fields.tx[i] + u_y * fields.ty[i]
            true_first[p] = float(np.trapezoid(profile, grid))
            true_second[p] = float(np.trapezoid(profile * profile, grid))
        entry = {"note": note, "stations": int(n_st), "references": {}}
        for rname, (rp, rt) in (("B_mglet", (phase_B, tau_B)),
                                ("C_xiao_repaired_cubic6", (phase_C, tau_C))):
            truth = C.periodic_interp(rp, rt, phases)
            v = np.sign(truth) * np.sqrt(np.abs(truth))
            grids = np.outer(y_m, xi)
            u_plus = fw.spalding_composite_uplus(
                grids * (np.abs(v) / C.NU)[:, None])
            linear = u_m - v * fw.spalding_composite_uplus(
                y_m * np.abs(v) / C.NU)
            assumed = v[:, None] * u_plus + linear[:, None] * grids / y_m[:, None]
            first = np.trapezoid(assumed, grids, axis=1)
            second = np.trapezoid(assumed * assumed, grids, axis=1)
            residual = (truth + ddx(second, arc, period)
                        - u_m * ddx(first, arc, period) + pressure - tau_at)
            control = (truth + ddx(true_second, arc, period)
                       - u_m * ddx(true_first, arc, period) + pressure - tau_at)
            scale = float(np.sqrt(np.mean(truth ** 2)))
            entry["references"][rname] = {
                "wall_traction_rms": scale,
                "assumed_profile_residual_rms_over_traction_rms":
                    float(np.sqrt(np.mean(residual ** 2)) / scale),
                "assumed_profile_residual_median_over_traction_rms":
                    float(np.median(np.abs(residual)) / scale),
                "true_profile_residual_rms_over_traction_rms":
                    float(np.sqrt(np.mean(control ** 2)) / scale),
                "matching_error_rms_over_velocity_rms": float(
                    np.sqrt(np.mean((assumed[:, -1] - u_m) ** 2))
                    / np.sqrt(np.mean(u_m ** 2))),
                "first_moment_relative_error_median": float(np.median(
                    np.abs(first - true_first) / np.maximum(np.abs(true_first), 1e-30))),
                "second_moment_relative_error_median": float(np.median(
                    np.abs(second - true_second) / np.maximum(np.abs(true_second), 1e-30))),
            }
        result["surfaces"][sname] = entry
        for rname, record in entry["references"].items():
            print(f"{sname}/{rname}: assumed-profile momentum residual = "
                  f"{record['assumed_profile_residual_rms_over_traction_rms']:.2f} "
                  f"x the wall-traction RMS; the TRUE profile gives "
                  f"{record['true_profile_residual_rms_over_traction_rms']:.2f}; "
                  f"assumed second moment is off by "
                  f"{100 * record['second_moment_relative_error_median']:.1f}%")
    out = ROOT / "codes/results" / f"integral_model_consistency_l0_{args.out_stamp}.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
