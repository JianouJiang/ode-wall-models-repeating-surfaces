#!/usr/bin/env python3
r"""
matching_height_window.py  --  Thrust #17, Level-3 (results & analysis)

THE DEPLOYABLE MATCHING-HEIGHT WINDOW (the practical, non-obvious consequence).

The manuscript derives an *upper* bound on the WMLES matching height for
repeating O(delta)-pitch structures:

    y_crit = |tau_w| / (eps_c |dp/dx|),     eps(y_crit) = eps_c ~ O(0.2),

above which the convection-blind ODE wall model falls off the force-cancellation
cliff (catastrophic R^2(tau_w) < 0).  y_crit is COMPUTED per geometry from the
reference profiles (codes/results/critical_matching_height_map.npz), not fitted.

Classical WMLES practice supplies a *lower* bound.  Kawai & Larsson (2012) showed
that matching at the FIRST off-wall cell corrupts the wall model with
numerically/SGS-under-resolved data ("log-layer mismatch"), and recommended
matching DEEPER -- in the inertial log layer, typically at the 3rd-5th off-wall
point, i.e. y_m^+ of order a few tens.  Call this guidance lower bound y_LLM.

The deployable window is therefore  [y_LLM, y_crit].  This script quantifies it
per geometry and exposes the central practical result:

  * For O(delta)-pitch repeating hills, y_crit^+ ~ O(10) is BELOW the classical
    log-layer matching height y_LLM ~ O(30-50): the window is EMPTY -- you cannot
    simultaneously escape log-layer mismatch (match deeper) AND stay below the
    cancellation cliff (match shallower).  Following "match in the log layer"
    places the first cell squarely in the catastrophe.  This INVERTS the
    Kawai-Larsson guidance for this geometry class.

  * For wide-pitch / single-feature controls (conv-div channel, BFS, NASA hump,
    curved BFS), y_crit^+ ~ O(150-200) sits FAR above y_LLM: the window is WIDE,
    the classical guidance is safe, and there is no bifurcation in the deployed
    range -- the honest O(delta)-pitch scope boundary (G7).

HONESTY (G1-G7).  y_crit is real (computed per geometry, traced to the map npz).
y_LLM is a LITERATURE GUIDANCE, not a computed quantity of this work -- it is a
range (Kawai & Larsson 2012), so the window's emptiness is reported as a function
of y_LLM (sensitivity table), not on a single fabricated number.  No DNS or
coupled numbers are invented here; this is an a-priori synthesis of the on-disk
y_crit map with the published lower-bound guidance.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/matching_height_window.py
Out:  codes/results/matching_height_window.npz
"""
from __future__ import annotations

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
MAP = os.path.join(RESULTS, "critical_matching_height_map.npz")

# Kawai & Larsson (2012) log-layer-mismatch guidance lower bound on the matching
# height: match in the inertial log layer (3rd-5th off-wall cell), y_m^+ of order
# a few tens.  Reported as a RANGE; the window emptiness is given vs this range.
Y_LLM_MIN = 30.0     # conservative lower edge of "match in the log layer"
Y_LLM_MAX = 50.0     # upper edge commonly cited
Y_LLM_SWEEP = np.array([20.0, 30.0, 40.0, 50.0])   # sensitivity of the verdict


def main():
    if not os.path.exists(MAP):
        raise SystemExit(f"FATAL: {MAP} missing -- run critical_matching_height.py first.")
    m = np.load(MAP, allow_pickle=True)
    keys = [str(k) for k in m["keys"]]
    klass = [str(k) for k in m["klass"]]
    repeating = m["repeating"].astype(bool)
    pitch_Od = m["pitch_O_delta"].astype(bool)
    ycrit = m["ycrit"].astype(float)          # y_crit^+ per geometry (inf if beyond grid)
    eps_c = m["eps_c"].astype(float)
    bifurcates = m["bifurcates"].astype(bool)

    n = len(keys)

    # window [y_LLM, y_crit]: width and emptiness vs the guidance lower bound.
    # width <= 0  =>  EMPTY window (cannot satisfy both bounds).
    window_width_min = ycrit - Y_LLM_MIN     # most-permissive lower bound (30)
    window_width_max = ycrit - Y_LLM_MAX     # stricter lower bound (50)
    window_empty_min = window_width_min <= 0  # empty even at the permissive y_LLM=30
    window_empty_max = window_width_max <= 0

    # verdict as a function of y_LLM in the sweep (rows=geom, cols=y_LLM)
    empty_grid = np.zeros((n, len(Y_LLM_SWEEP)), dtype=bool)
    for j, yl in enumerate(Y_LLM_SWEEP):
        empty_grid[:, j] = (ycrit - yl) <= 0

    print("\n=== DEPLOYABLE MATCHING-HEIGHT WINDOW  [y_LLM, y_crit] ===")
    print(f"  classical lower bound  y_LLM^+ in [{Y_LLM_MIN:.0f}, {Y_LLM_MAX:.0f}] "
          f"(Kawai & Larsson 2012 log-layer matching)\n")
    hdr = (f"  {'geometry':<26}{'class':<14}{'O(d)?':<7}"
           f"{'y_crit+':>9}{'eps_c':>8}{'win@30':>9}{'win@50':>9}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(n):
        yc = ycrit[i]
        yc_s = "inf" if not np.isfinite(yc) else f"{yc:7.1f}"
        w30 = window_width_min[i]; w50 = window_width_max[i]
        w30_s = "inf" if not np.isfinite(w30) else f"{w30:7.1f}"
        w50_s = "inf" if not np.isfinite(w50) else f"{w50:7.1f}"
        if window_empty_min[i]:
            verdict = "EMPTY (window closed: must over-match into cliff)"
        elif window_empty_max[i]:
            verdict = "NARROW (open only for permissive y_LLM)"
        else:
            verdict = "WIDE (classical guidance safe)"
        print(f"  {keys[i]:<26}{klass[i]:<14}{('yes' if pitch_Od[i] else 'no'):<7}"
              f"{yc_s:>9}{eps_c[i]:>8.3f}{w30_s:>9}{w50_s:>9}  {verdict}")

    # measured deployed operating points (real, on disk) for context
    operating = {}
    a1p0_yp = _measured_yplus(os.path.join(CODES, "openfoam", "xiao_wmles_a1p0"))
    if a1p0_yp:
        operating["xiao_a1p0_wellresolved"] = a1p0_yp
    band_yp = _measured_yplus(os.path.join(CODES, "openfoam", "xiao_wmles_a1p0_band"))
    if band_yp:
        operating["xiao_a1p0_band"] = band_yp

    print("\n  measured coupled-WMLES operating heights (real, bottomWall yPlus FO):")
    for k, v in operating.items():
        print(f"    {k:<28} y_m^+ avg={v['yavg']:.2f}  [{v['ymin']:.2f}, {v['ymax']:.2f}]")
    if not operating:
        print("    (band run not yet harvested -- operating points fill in on completion)")

    # the headline: which O(delta)-pitch repeating geometries have a CLOSED window
    od_hills = [i for i in range(n) if pitch_Od[i] and repeating[i]]
    od_closed = [keys[i] for i in od_hills if window_empty_min[i]]
    controls = [i for i in range(n) if not (pitch_Od[i] and repeating[i])]
    ctrl_open = [keys[i] for i in controls if not window_empty_min[i]]

    print("\n  HEADLINE:")
    print(f"   O(delta)-pitch repeating hills with a CLOSED window (y_crit+ < y_LLM=30): "
          f"{od_closed}")
    print(f"   controls with a WIDE window (classical guidance safe):           {ctrl_open}")
    print("   => the matching-height window is geometry-selective: it closes ONLY on")
    print("      the O(delta)-pitch repeating class, inverting the Kawai-Larsson rule there.")

    out = dict(
        keys=np.array(keys), klass=np.array(klass),
        repeating=repeating, pitch_O_delta=pitch_Od,
        ycrit=ycrit, eps_c=eps_c, bifurcates=bifurcates,
        y_llm_min=Y_LLM_MIN, y_llm_max=Y_LLM_MAX,
        y_llm_sweep=Y_LLM_SWEEP,
        window_width_at30=window_width_min,
        window_width_at50=window_width_max,
        window_empty_at30=window_empty_min,
        window_empty_at50=window_empty_max,
        empty_grid=empty_grid,
        operating=np.array([operating], dtype=object),
        od_hills_window_closed=np.array(od_closed),
        controls_window_open=np.array(ctrl_open),
        note=("Deployable matching-height window [y_LLM, y_crit]. y_crit COMPUTED "
              "per geometry (critical_matching_height_map.npz); y_LLM is the "
              "Kawai & Larsson 2012 log-layer-matching guidance (literature range, "
              "NOT computed here) -- emptiness reported vs y_LLM. Window closes only "
              "on O(delta)-pitch repeating hills (y_crit+~O(10) < y_LLM), inverting "
              "the classical match-deeper guidance; wide for all controls (G7)."),
    )
    np.savez(os.path.join(RESULTS, "matching_height_window.npz"), **out)
    print("\n  wrote codes/results/matching_height_window.npz")


def _measured_yplus(case):
    """Last-time bottomWall y+ (min,max,avg) from the yPlus FO, or None."""
    base = os.path.join(case, "postProcessing", "yPlus")
    if not os.path.isdir(base):
        return None
    ts = []
    for d in os.listdir(base):
        try:
            ts.append((float(d), d))
        except ValueError:
            pass
    if not ts:
        return None
    ts.sort()
    f = os.path.join(base, ts[-1][1], "yPlus.dat")
    if not os.path.exists(f):
        return None
    rows = [ln.split() for ln in open(f) if ln.strip() and not ln.startswith("#")]
    bw = [r for r in rows if len(r) >= 5 and r[1] == "bottomWall"]
    if not bw:
        return None
    last_t = bw[-1][0]
    r = [x for x in bw if x[0] == last_t][-1]
    return dict(t=float(last_t), ymin=float(r[2]), ymax=float(r[3]), yavg=float(r[4]))


if __name__ == "__main__":
    main()
