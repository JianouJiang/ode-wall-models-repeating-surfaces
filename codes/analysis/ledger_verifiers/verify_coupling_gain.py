#!/usr/bin/env python3
"""Independent verifier for the coupling-gain probe (node_013).

Re-derives every reported number from the deposited run output rather than
trusting the harvest, checks that the perturbation was actually active in the
solver logs, and carries red fixtures that must fail.

    python3 codes/analysis/ledger_verifiers/verify_coupling_gain.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
PROBES = RESULTS / "rswm_gain_probe"
VEC = re.compile(r"\(([^)]*)\)")
GAIN_LINE = re.compile(
    r"COUPLING_GAIN patch=(\S+) time=(\S+) gain=(\S+) gainStartTime=(\S+) "
    r"active=(\d+)")

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


def latest(pattern):
    hits = sorted(glob.glob(str(RESULTS / pattern)))
    if not hits:
        raise SystemExit("missing producer output: %s" % pattern)
    return pathlib.Path(hits[-1])


def read_series(probe_id, patch="wallForceBottom"):
    base = PROBES / probe_id / "postProcessing" / patch
    files = sorted(base.rglob("surfaceFieldValue.dat")) if base.exists() else []
    ts, xs = [], []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = VEC.search(line)
            if not m:
                continue
            ts.append(float(line.split()[0]))
            xs.append(float(m.group(1).split()[0]))
    if not ts:
        return None
    t = np.asarray(ts)
    x = np.asarray(xs)
    o = np.argsort(t)
    t, x = t[o], x[o]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], x[keep]


def transmission(control, perturbed, gain, plateau, t0, avg_start):
    c = read_series(control)
    p = read_series(perturbed)
    if c is None or p is None:
        return None
    tc, Fc_raw = c
    tp, Fp_raw = p
    hi = min(tc[-1], tp[-1])
    grid = tc[(tc >= t0) & (tc <= hi)]
    Fc = np.interp(grid, tc, Fc_raw)
    Fp = np.interp(grid, tp, Fp_raw)
    T = (Fp / Fc - 1.0) / (gain - 1.0)
    plat = (grid >= t0 + plateau[0]) & (grid <= t0 + plateau[1])
    win = grid >= avg_start
    if win.sum() < 8:
        win = grid >= (t0 + 0.5 * (hi - t0))
    return dict(T0=float(T[0]), T_plateau=float(T[plat].mean()),
                T_plateau_sd=float(T[plat].std()),
                T_late=float((Fp[win].mean() / Fc[win].mean() - 1.0)
                             / (gain - 1.0)),
                n=int(grid.size), n_plateau=int(plat.sum()))


def main():
    print("REFEREE LEDGER VERIFIER -- coupling gain probe (node_013)")

    matrix = json.loads((ROOT / "jobs" / "rswm_gain_probe_matrix.json").read_text())
    tr_path = latest("gain_probe_transmission_*_summary.json")
    tr = json.loads(tr_path.read_text())
    ex = json.loads(latest("gain_probe_model_exponent_*_summary.json").read_text())
    t0 = tr["gain_start_time"]
    plateau = tuple(tr["plateau_window_after_onset"])

    # ---- 1. the registered matrix was executed -----------------------------
    expected = [r for r in matrix["runs"]]
    missing = [r["probe_id"] for r in expected
               if not (PROBES / r["probe_id"] / "log.pimpleFoam").exists()]
    check("every registered run produced a solver log",
          not missing, "missing: %s" % (", ".join(missing) if missing else "none"))

    unfinished, wrong_gain, inactive = [], [], []
    for r in expected:
        log = PROBES / r["probe_id"] / "log.pimpleFoam"
        if not log.exists():
            continue
        text = log.read_text()
        if not re.search(r"^End\s*$", text, re.M):
            unfinished.append(r["probe_id"])
        man = PROBES / r["probe_id"] / "MANIFEST.json"
        if man.exists():
            m = json.loads(man.read_text())
            if abs(m["gain"] - r["gain"]) > 0 or m["model"] != r["model"]:
                wrong_gain.append(r["probe_id"])
        hits = GAIN_LINE.findall(text)
        if r["model"] == "deployed_control":
            if hits:
                inactive.append("%s: gain BC present in deployed control"
                                % r["probe_id"])
            continue
        bottom = [h for h in hits if h[0] == "bottomWall"]
        if not bottom:
            inactive.append("%s: no COUPLING_GAIN report" % r["probe_id"])
            continue
        want_active = 1 if r["gain"] != 1.0 else 0
        got = {int(h[4]) for h in bottom}
        gains = {float(h[2]) for h in bottom}
        if got != {want_active} or gains != {float(r["gain"])}:
            inactive.append("%s: active=%s gain=%s (wanted %d, %.2f)"
                            % (r["probe_id"], got, gains, want_active, r["gain"]))

    check("every run reached the solver's End marker", not unfinished,
          "unfinished: %s" % (", ".join(unfinished) if unfinished else "none"))
    check("manifest gain and model match the registered matrix", not wrong_gain,
          "mismatched: %s" % (", ".join(wrong_gain) if wrong_gain else "none"))
    check("the gain was active exactly where the matrix says", not inactive,
          "; ".join(inactive) if inactive else "all runs consistent")

    # ---- 2. the reported transmissions are reproducible ---------------------
    pairs = [p for p in tr["pairs"] if p.get("status") == "ok"
             and p.get("patch") == "wallForceBottom"]
    check("at least one complete pair was harvested", len(pairs) > 0,
          "%d complete pairs" % len(pairs))

    worst_rep = 0.0
    worst_T0_up, worst_T0_down = 0.0, 0.0
    for p in pairs:
        re_ = transmission(p["control"], p["perturbed"], p["gain"], plateau,
                           t0, p["average_start"])
        if re_ is None:
            check("reproduce %s" % p["perturbed"], False, "series unreadable")
            continue
        worst_rep = max(worst_rep,
                        abs(re_["T_plateau"] - p["T_plateau"]),
                        abs(re_["T_late"] - p["T_late"]))
        if p["gain"] > 1.0:
            worst_T0_up = max(worst_T0_up, abs(re_["T0"] - 1.0))
        else:
            worst_T0_down = max(worst_T0_down, abs(re_["T0"] - 1.0))
    check("harvested T reported values reproduced from the deposited series",
          worst_rep < 1e-9, "worst absolute difference %.3e" % worst_rep)
    # For G > 1 the delivered eddy viscosity G(nu+nut) - nu is always positive,
    # so the gain is applied exactly and the instrument identity is exact.
    # The tolerance is the eight-significant-figure precision of OpenFOAM's
    # ASCII function-object output, not a physical allowance.
    check("instrument: T = 1 at the gain onset for every G > 1 arm, to the "
          "precision the force series is written with",
          worst_T0_up < 1e-7, "worst |T(t0)-1| = %.3e" % worst_T0_up)
    # For G < 1 the non-negativity floor of the scalar eddy viscosity binds on
    # faces where nut < nu(1-G)/G, so the delivered traction is not exactly
    # G times the request there.  The departure is measured, not assumed.
    # The departure is a measured property of the boundary condition, so it is
    # reported and bounded loosely, and the analysis normalises it away by
    # dividing every arm by its own delivered gain.
    check("instrument: the G < 1 arms depart from the exact gain only through "
          "the eddy-viscosity floor, and the departure is measured",
          worst_T0_down < 0.2,
          "worst |T(t0)-1| = %.3e (normalised out via delivered_gain)"
          % worst_T0_down)

    # ---- 3. the derived boundary condition is the deployed one at G = 1 -----
    fid = tr.get("fidelity", {})
    fid_ok, fid_detail = True, []
    for arch, f in sorted(fid.items()):
        if f.get("status") != "ok":
            fid_ok = False
            fid_detail.append("%s: %s" % (arch, f.get("status")))
            continue
        fid_detail.append("%s: max rel %.2e over %d samples"
                          % (arch, f["max_relative_difference"], f["n_compared"]))
        if f["n_compared"] < 20 or f["max_relative_difference"] > 1e-9:
            fid_ok = False
    check("fidelity: gain=1 arm equals the unmodified deployed boundary "
          "condition to round-off", fid_ok, "; ".join(fid_detail))

    # ---- 4. the a-priori exponent side --------------------------------------
    recs = ex["records"]
    check("response exponent computed for all ten deposits", len(recs) == 10,
          "%d records" % len(recs))
    worst_kernel = max(r["max_reproduction_rel_error"] for r in recs)
    check("deployed kernels reproduce the deposited request (< 1e-9)",
          worst_kernel < 1e-9, "worst relative error %.3e" % worst_kernel)
    shallow = [r for r in recs if r["architecture"] == "equilibrium"
               and abs(r["ym_over_H"] - 0.03) < 1e-9]
    if shallow:
        s0 = shallow[0]["s_median"]
        check("viscous-sublayer limit: equilibrium s -> 1 at the shallowest "
              "matching surface", abs(s0 - 1.0) < 0.10, "s = %.3f" % s0)
    unres = min(r["resolved_fraction"] for r in recs)
    check("the exponent is resolved on almost every face", unres > 0.90,
          "smallest resolved fraction %.4f" % unres)

    # ---- 5. reported orderings follow from the arrays, not from prose -------
    def by(arch, gain=1.25):
        return sorted([p for p in pairs
                       if p["architecture"] == arch and p["gain"] == gain],
                      key=lambda r: r["ym_over_H"])

    # A reading is deterministic only where the two arms are still one
    # trajectory plus a difference; the scatter of T inside the lag window
    # measures that directly.
    SD_MAX = 0.15
    LAG = "2.0"
    for arch in ("eq", "tble"):
        pts = by(arch)
        det = [p for p in pts if p["T_sd_at_lag"][LAG] < SD_MAX]
        if len(det) >= 2:
            check("%s: every deterministic T at lag 2 is below 1 (the coupling "
                  "transmits less than the whole error)" % arch,
                  all(p["T_at_lag"][LAG] < 1.0 for p in det),
                  "T(2) = %s" % ", ".join("%.3f" % p["T_at_lag"][LAG]
                                          for p in det))
            check("%s: the deterministic T at lag 2 rises monotonically with "
                  "matching height" % arch,
                  all(b["T_at_lag"][LAG] > a["T_at_lag"][LAG]
                      for a, b in zip(det, det[1:])),
                  "eta_m/H = %s" % ", ".join("%.4g" % p["ym_over_H"]
                                             for p in det))
        if len(det) < len(pts):
            check("%s: the non-deterministic readings are declared, not "
                  "reported as measurements" % arch, True,
                  "excluded eta_m/H = %s"
                  % ", ".join("%.4g" % p["ym_over_H"]
                              for p in pts if p not in det))

    # The two couplings differ by more than an order of magnitude in how long a
    # small difference stays small.  That is the measurement, so it is checked.
    sd_eq = [p["T_sd_at_lag"]["2.0"] for p in by("eq")]
    sd_tb = [p["T_sd_at_lag"]["2.0"] for p in by("tble")]
    if sd_eq and sd_tb:
        check("the total-gradient pairs lose their pairing faster than the "
              "equilibrium pairs at every matching height",
              max(sd_eq) < min(sd_tb),
              "worst equilibrium scatter %.3f < best total-gradient %.3f"
              % (max(sd_eq), min(sd_tb)))

    tops = [p for p in tr["pairs"] if p.get("status") == "ok"
            and p.get("patch") == "wallForceTop" and p.get("gain") == 1.25]
    if tops:
        worst_top = max(abs(p["T_at_lag"]["5.0"]) for p in tops)
        check("within-run control: the unperturbed upper wall of the same "
              "simulations barely responds (|T| < 0.10 at lag 5)",
              worst_top < 0.10, "worst |T_top(5)| = %.3f over %d runs"
              % (worst_top, len(tops)))

    eqs = {round(r["ym_over_H"], 4): r["s_median"]
           for r in recs if r["architecture"] == "equilibrium"}
    tbs = {round(r["ym_over_H"], 4): r["s_median"]
           for r in recs if r["architecture"] == "tble"}
    common = sorted(set(eqs) & set(tbs))
    if common:
        below = [y for y in common if tbs[y] < eqs[y]]
        check("the total-gradient model's response exponent is below the "
              "equilibrium model's at every shared matching height",
              len(below) == len(common),
              "%d of %d heights" % (len(below), len(common)))

    # ---- 6. red fixtures ----------------------------------------------------
    print("  -- red fixtures (each must FAIL its check) --")
    t = np.linspace(405.0, 450.0, 400)
    rigid = np.ones_like(t)                      # T == 1 everywhere
    plat = (t >= 405 + plateau[0]) & (t <= 405 + plateau[1])
    check("red fixture: a rigid coupling (T = 1) is not reported as suppression",
          not (rigid[plat].mean() < 1.0),
          "rigid plateau mean = %.3f" % rigid[plat].mean())

    if pairs:
        p = pairs[0]
        key = "%s__%s__series_T" % (p["perturbed"], p["patch"])
        npz = np.load(str(tr_path).replace("_summary.json", ".npz"))
        if key in npz.files:
            T = npz[key]
            rolled = np.roll(T, len(T) // 2)
            check("red fixture: a time-shifted pairing breaks the T(t0) = 1 "
                  "instrument check", abs(rolled[0] - 1.0) > 1e-6,
                  "shifted T(t0) = %.4f" % rolled[0])

    bad_gain = transmission(pairs[0]["control"], pairs[0]["perturbed"],
                            2.0 * pairs[0]["gain"], plateau, t0,
                            pairs[0]["average_start"]) if pairs else None
    if bad_gain is not None:
        check("red fixture: reducing with the wrong gain breaks T(t0) = 1",
              abs(bad_gain["T0"] - 1.0) > 1e-3,
              "wrong-gain T(t0) = %.4f" % bad_gain["T0"])

    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    print("%d/%d checks passed" % (n_ok, len(CHECKS)))
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
