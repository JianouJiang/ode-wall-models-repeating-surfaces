#!/usr/bin/env python3
"""Where does the source-norm penalty come from?  Flat floor against curved flank.

The tournament producer measures one number for the whole wall: the fraction
delta of an assembled source norm that comes back as wall-traction error.  The
mechanism claims that delta is a property of the SURFACE -- the one-dimensional
wall-normal reduction is a closed balance on a flat wall and is not closed on a
curved one -- and not of the wall model.  If that is right, refitting delta on
the flat inter-hill floor alone must give a much smaller number than refitting
it on the curved flanks of the same simulation, with the same arms, the same
reference and the same protocol.

That is a within-simulation positive control: the flat floor is a region where
the reduction the models assume is exactly the balance the flow obeys, so any
residual delta there is the numerical floor of the experiment rather than the
effect under test.

Reads the deposited tournament arrays; runs no wall model and no simulation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C  # noqa: E402
import source_budget_tournament_l0 as T  # noqa: E402

STAMP = "20260825"
RESULTS = ROOT / "codes" / "results"
REFERENCES = RESULTS / f"wall_traction_references_{STAMP}.npz"
SURFACES = ("archive_index10", "ladder_L1")
REFS = {"B_mglet": "B_mglet", "C_xiao_repaired_cubic6": "C_xiao_repaired_cubic6"}

# The flat inter-hill floor of the Xiao geometry, and its complement.  These are
# the campaign's registered region boundaries (conditioning_ladder_l0.REGIONS),
# reused verbatim so the split is not chosen after seeing the answer.
FLAT_FLOOR = (2.05, 6.90)


def reference_at(phase, name):
    refs = np.load(REFERENCES)
    ph = np.asarray(refs[f"{name}__phase"], float)
    tw = np.asarray(refs[f"{name}__tau"], float)
    o = np.argsort(ph)
    ph, tw = ph[o], tw[o]
    q = np.mod(np.asarray(phase, float), 1.0)
    return np.interp(q, np.r_[ph - 1.0, ph, ph + 1.0], np.r_[tw, tw, tw])


def main() -> int:
    out = {
        "schema": "source_budget_regions_l0/1",
        "question": ("is the fraction delta of an assembled source norm that "
                     "returns as wall-traction error a property of the surface "
                     "rather than of the wall model?"),
        "registered_prediction_P5": T.PREREGISTERED["P5_geometry_carries_delta"],
        "flat_floor_x_over_H": list(FLAT_FLOOR),
        "region_definition_source": "conditioning_ladder_l0.REGIONS (registered)",
        "surfaces": {},
    }
    for surface in SURFACES:
        npz = np.load(RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.npz")
        summary = json.loads(
            (RESULTS / f"source_budget_tournament_l0_{surface}_{STAMP}.json").read_text())
        phase = np.asarray(npz["phase"], float)
        x = np.mod(phase, 1.0) * C.LX
        flat = (x >= FLAT_FLOOR[0]) & (x <= FLAT_FLOOR[1])
        regions = {"flat_floor": flat, "sloped_wall": ~flat}
        arms = sorted(k[len("pred__"):] for k in npz.files if k.startswith("pred__"))
        sweep_arms = [a for a in arms
                      if a.startswith("CTL_scale_") or a in T.SCALE_BASES]
        entry = {"stations": {k: int(v.sum()) for k, v in regions.items()},
                 "arms_used_for_the_fit": sweep_arms, "references": {}}
        for rname in REFS:
            truth = reference_at(phase, REFS[rname])
            per_ref = {}
            for region, mask in regions.items():
                Ns, Es, table = [], [], {}
                for a in arms:
                    p = np.asarray(npz[f"pred__{a}"], float)
                    n = np.asarray(npz[f"norm__{a}"], float)
                    ok = mask & np.isfinite(p) & np.isfinite(n)
                    if ok.sum() < 8:
                        continue
                    e_abs = float(np.sqrt(np.mean((p[ok] - truth[ok]) ** 2)))
                    n_rms = float(np.sqrt(np.mean(n[ok] ** 2)))
                    table[a] = {"absolute_rms": e_abs, "N_rms": n_rms,
                                "relative_rms": e_abs / float(np.sqrt(np.mean(truth[ok] ** 2))),
                                "stations": int(ok.sum())}
                    if a in sweep_arms:
                        Ns.append(n_rms)
                        Es.append(e_abs)
                E0, delta, fit_rms = T.affine_fit(Ns, Es)
                per_ref[region] = {
                    "E0": E0, "delta": delta, "fit_rms": fit_rms,
                    "truth_rms": float(np.sqrt(np.mean(truth[mask] ** 2))),
                    "per_arm": table,
                }
            per_ref["delta_ratio_sloped_over_flat"] = (
                per_ref["sloped_wall"]["delta"] / per_ref["flat_floor"]["delta"]
                if per_ref["flat_floor"]["delta"] not in (0.0,) else float("nan"))
            per_ref["P5_verdict"] = (
                "SUPPORTED" if per_ref["delta_ratio_sloped_over_flat"] >= 3.0
                else "REFUTED")
            # --- POST-HOC, computed after P5 failed as registered ------------
            # The fitted SLOPE turns out to be region-independent, so the two
            # regions must differ somewhere else.  These two ratios are the
            # places they do, and they are labelled post-hoc because they were
            # not the quantity named in advance.
            flat, slope = per_ref["flat_floor"], per_ref["sloped_wall"]
            attain = {}
            for a, row in flat["per_arm"].items():
                if a not in slope["per_arm"] or row["N_rms"] <= 0.0:
                    continue
                s_row = slope["per_arm"][a]
                if s_row["N_rms"] <= 0.0:
                    continue
                attain[a] = {
                    "flat_floor": row["absolute_rms"] / row["N_rms"],
                    "sloped_wall": s_row["absolute_rms"] / s_row["N_rms"],
                    "ratio_sloped_over_flat": ((s_row["absolute_rms"] / s_row["N_rms"])
                                               / (row["absolute_rms"] / row["N_rms"])),
                }
            per_ref["post_hoc_not_registered"] = {
                "why": ("P5 named the fitted slope; it is region-independent. "
                        "These are the two quantities that are not, found after "
                        "that test returned REFUTED."),
                "intercept_E0_ratio_sloped_over_flat":
                    slope["E0"] / flat["E0"] if flat["E0"] else float("nan"),
                "attainment_E_abs_over_N_by_arm": attain,
                "closure_free_attainment_ratio":
                    attain.get("Xfull_closure_free", {}).get("ratio_sloped_over_flat",
                                                             float("nan")),
            }
            entry["references"][rname] = per_ref
        out["surfaces"][surface] = entry
        for r in REFS:
            e = entry["references"][r]
            ph = e["post_hoc_not_registered"]
            print(f"{surface} / {r}: P5 slope flat {e['flat_floor']['delta']:.3f} vs "
                  f"sloped {e['sloped_wall']['delta']:.3f} "
                  f"(x{e['delta_ratio_sloped_over_flat']:.1f}) -> {e['P5_verdict']}"
                  f" | post hoc: intercept x{ph['intercept_E0_ratio_sloped_over_flat']:.1f}, "
                  f"closure-free attainment x{ph['closure_free_attainment_ratio']:.1f}")

    verdicts = {s: {r: out["surfaces"][s]["references"][r]["P5_verdict"] for r in REFS}
                for s in SURFACES}
    out["P5_verdict_overall"] = (
        "SUPPORTED" if all(v == "SUPPORTED" for s in verdicts for v in verdicts[s].values())
        else "MIXED" if any(v == "SUPPORTED" for s in verdicts for v in verdicts[s].values())
        else "REFUTED")
    out["P5_verdict_by_surface_and_reference"] = verdicts
    path = RESULTS / f"source_budget_regions_l0_{STAMP}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("P5 overall:", out["P5_verdict_overall"], "->", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
