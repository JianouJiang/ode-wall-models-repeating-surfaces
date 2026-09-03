#!/usr/bin/env python3
"""Fluid-state audit of the deposited d-type rib LES (defect D2 certificate).

The operator campaign (work_progress/archer2_campaign_20260823/R2-4_M20/REBASE.md)
found that the single-pitch `rib_les_dtype` deposit has a laminar upper half:
the smooth-wall side of the channel carries no resolved turbulence for the
entire averaging window, so the case is not a wall-resolved LES of a turbulent
rib channel and its resolved-shear fraction of 0.992 is degenerate.  This
script recomputes that evidence directly from the deposited OpenFOAM fields so
every number quoted in the manuscript traces to a codes/results certificate.

For every cumulative fieldAverage write (timeStart 40) it reports spanwise-band
column averages of UPrime2Mean and of the instantaneous subgrid viscosity:
  vv, ww on y in [1.4, 1.8]  (upper smooth-wall half; laminar if ~1e-8 U_b^2)
  vv, uv on y in [0.2, 0.6]  (rib shear layer; the only turbulent band)
  nut/nu on y in [1.0, 1.8]  (subgrid model inactive above the shear layer)

Output: codes/results/rib_dtype_upper_half_audit.npz (+ _summary.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "openfoam", "rib_les_dtype")
OUT_NPZ = os.path.join(ROOT, "results", "rib_dtype_upper_half_audit.npz")
OUT_JSON = os.path.join(ROOT, "results", "rib_dtype_upper_half_audit_summary.json")
NU = 1.0 / 4200.0
# Laminar-upper-half criterion: resolved wall-normal variance in the upper band
# below 1e-6 U_b^2 (three decades under the shear-layer level) at every window.
VV_LAMINAR_MAX = 1.0e-6

sys.path.insert(0, os.path.join(ROOT, "openfoam"))
from of_ascii_fast import read_internal  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def time_dirs():
    out = []
    for d in os.listdir(CASE):
        p = os.path.join(CASE, d)
        if os.path.isdir(p) and re.fullmatch(r"[0-9.eE+-]+", d):
            if os.path.exists(os.path.join(p, "UPrime2Mean")):
                out.append((float(d), d))
    return sorted(out)


def main() -> int:
    C, _ = read_internal(os.path.join(CASE, "0", "C"))
    key = np.round(C[:, :2], 7)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(float)
    y = uniq[:, 1]

    def span_avg(f):
        return np.bincount(inv, weights=f, minlength=n) / cnt

    bands = {
        "upper": (y >= 1.4) & (y <= 1.8),
        "shear": (y >= 0.2) & (y <= 0.6),
        "nut_band": (y >= 1.0) & (y <= 1.8),
    }
    tds = time_dirs()
    if not tds:
        print("RIB_AUDIT_NO_AVERAGED_FIELDS")
        return 2

    windows, rows = [], []
    for t, d in tds:
        R, _ = read_internal(os.path.join(CASE, d, "UPrime2Mean"))
        nut, _ = read_internal(os.path.join(CASE, d, "nut"))
        uu, uv, vv, ww = (span_avg(R[:, i]) for i in (0, 1, 3, 5))
        nut_c = span_avg(nut)
        rows.append(dict(
            window_end=t,
            vv_upper=float(np.mean(vv[bands["upper"]])),
            ww_upper=float(np.mean(ww[bands["upper"]])),
            uu_upper=float(np.mean(uu[bands["upper"]])),
            vv_shear=float(np.mean(vv[bands["shear"]])),
            uv_shear=float(np.mean(uv[bands["shear"]])),
            nut_over_nu=float(np.mean(nut_c[bands["nut_band"]]) / NU),
        ))
        windows.append(t)

    laminar = all(r["vv_upper"] < VV_LAMINAR_MAX for r in rows)
    summary = dict(
        schema="rib-dtype-upper-half-audit-v1",
        case="codes/openfoam/rib_les_dtype",
        cells=int(len(C)),
        nu=NU,
        averaging_time_start=40.0,
        windows=windows,
        rows=rows,
        criterion=dict(vv_upper_max_for_laminar=VV_LAMINAR_MAX,
                       band_upper=[1.4, 1.8], band_shear=[0.2, 0.6],
                       band_nut=[1.0, 1.8]),
        laminar_upper_half=bool(laminar),
        verdict=("DEPOSIT_NOT_TURBULENT_WRLES: upper half laminar over the "
                 "entire averaging window; resolved-fraction 0.992 degenerate"
                 if laminar else "UPPER_HALF_TURBULENT"),
        source_sha256={d: sha256(os.path.join(CASE, d, "UPrime2Mean"))
                       for _, d in tds},
        cross_reference="work_progress/archer2_campaign_20260823/R2-4_M20/REBASE.md",
    )
    np.savez_compressed(
        OUT_NPZ,
        windows=np.array(windows),
        vv_upper=np.array([r["vv_upper"] for r in rows]),
        ww_upper=np.array([r["ww_upper"] for r in rows]),
        uu_upper=np.array([r["uu_upper"] for r in rows]),
        vv_shear=np.array([r["vv_shear"] for r in rows]),
        uv_shear=np.array([r["uv_shear"] for r in rows]),
        nut_over_nu=np.array([r["nut_over_nu"] for r in rows]),
        laminar_upper_half=np.array(laminar),
        cells=np.array(len(C)),
        nu=np.array(NU),
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=1)
    for r in rows:
        print("window [40,%g]: vv_upper=%.3g ww_upper=%.3g vv_shear=%.3g "
              "uv_shear=%.3g nut/nu=%.3g"
              % (r["window_end"], r["vv_upper"], r["ww_upper"],
                 r["vv_shear"], r["uv_shear"], r["nut_over_nu"]))
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
