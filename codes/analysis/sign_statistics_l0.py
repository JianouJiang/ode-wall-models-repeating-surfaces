#!/usr/bin/env python3
"""Sign statistics of the a-priori periodic-hill wall traction, under each reference.

The paper reports how often the pressure-gradient ODE predicts reversed wall
shear and how often the reference actually is reversed.  Both halves of that
sentence were taken from the velocity archive's stored traction, i.e. from the
withdrawn four-point estimator, and the reversed fraction of a small
sign-changing quantity is exactly what a poorly resolved wall-gradient fit gets
wrong.  This producer recomputes the whole sentence -- reference reversed
fraction, model reversed fraction, overall and conditional sign accuracy --
against each of the three references, on the dense phase grid the paper's other
metrics use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
STAMP = "20260825"
SURFACE = "archive_index10"
DENSE_N = 4096
MODEL = "M1_pressure_gradient"
REFS = ("A_withdrawn_linear4", "B_mglet", "C_xiao_repaired_cubic6")


def periodic_interp(x, y, target):
    o = np.argsort(np.asarray(x, float))
    x = np.asarray(x, float)[o]
    y = np.asarray(y, float)[o]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0], np.r_[y, y, y])


def main() -> int:
    d = np.load(RESULTS / f"source_budget_tournament_l0_{SURFACE}_{STAMP}.npz")
    refs = np.load(RESULTS / f"wall_traction_references_{STAMP}.npz")
    dense = np.arange(DENSE_N) / DENSE_N
    phase = np.asarray(d["phase"], float)
    pred = periodic_interp(phase, np.asarray(d[f"pred__{MODEL}"], float), dense)

    out = {"schema": "sign_statistics_l0/1",
           "surface": SURFACE,
           "model": MODEL,
           "model_note": ("the pressure-gradient ODE at the paper's a-priori matching "
                          "surface, the arm the reported sentence refers to"),
           "dense_points": DENSE_N,
           "model_reversed_fraction": float(np.mean(pred < 0.0)),
           "references": {}}
    for name in REFS:
        truth = periodic_interp(refs[f"{name}__phase"], refs[f"{name}__tau"], dense)
        rev = truth < 0.0
        att = ~rev
        out["references"][name] = {
            "reference_reversed_fraction": float(np.mean(rev)),
            "sign_accuracy_overall": float(np.mean(np.sign(pred) == np.sign(truth))),
            "sign_accuracy_on_reversed_reference": float(
                np.mean(np.sign(pred[rev]) == np.sign(truth[rev]))) if rev.any() else float("nan"),
            "sign_accuracy_on_attached_reference": float(
                np.mean(np.sign(pred[att]) == np.sign(truth[att]))) if att.any() else float("nan"),
            "over_prediction_of_reversal_points": float(
                np.mean(pred < 0.0) - float(np.mean(rev))),
        }
    path = RESULTS / f"sign_statistics_l0_{STAMP}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
