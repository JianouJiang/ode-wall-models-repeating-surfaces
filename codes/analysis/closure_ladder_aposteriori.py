#!/usr/bin/env python3
r"""
closure_ladder_aposteriori.py  --  Level-2 implementation/experiment module
===========================================================================

THE EXPERIMENT (a-posteriori, coupled WMLES on periodic hills, Re_H=10595)
--------------------------------------------------------------------------
The cancellation mechanism makes a sharp, falsifiable a-posteriori prediction
that a-priori error metrics cannot test on their own: the catastrophic failure
on a repeating (O(delta)-pitch) geometry comes from dropping the CONVECTIVE
term, NOT from dropping the pressure-gradient term.  So climbing the LOCAL
closure ladder -- from an equilibrium ODE wall model (no dp/dx, no convection)
up to a TBLE / non-equilibrium ODE wall model (keeps dp/dx, still drops
convection) -- should NOT rescue the geometry.  A better *local* closure cannot
restore information that is *non-local* (convective).

  rung 0  EQUILIBRIUM ODE   tau_w = f(U_m, y_m, nu)              [no dp/dx, no conv]
  rung 1  TBLE / nonEq ODE  integrate d/dy[(nu+nu_t)dU/dy] = dp/dx  [+dp/dx, no conv]
          ----------------------------------------------------------------
          neither carries the streamwise CONVECTIVE flux d/dx(UU)+d/dy(UV)

PREDICTION (pre-registered, cancellation mechanism):
  Adding dp/dx (rung 0 -> rung 1) does NOT close the reattachment / C_f gap on
  periodic hills.  If anything TBLE can be WORSE, because dp/dx is the LARGE
  term that convection is supposed to nearly cancel; reinstating it WITHOUT the
  cancelling convective flux leaves an even larger un-cancelled forcing on the
  near-wall column.  Contrast: on a localised-separation geometry (eps~O(1))
  TBLE is a genuine improvement -- there dp/dx is not being cancelled.

This module reads the COMBINED coupled-WMLES harvest
(codes/results/aposteriori_wmles_pehill.npz, written by the OpenFOAM
orchestration analyze_aposteriori_combined.py) and, for EACH rung that is
actually present on disk, computes the SAME a-posteriori skill metrics against
the DNS reference (Cf R^2 with the node_001/node_003 convention: rung C_f at
its own stations vs DNS C_f interpolated there).  Rungs that have not finished
solving are flagged present=False and NOT scored -- no fabrication.

OUTPUT
  codes/results/closure_ladder_aposteriori.npz   (table + per-rung arrays)
  manuscript/figures/fig_closure_ladder.pdf      (C_f ladder + skill bars)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
NPZ_IN = os.path.join(RES, "aposteriori_wmles_pehill.npz")
NPZ_OUT = os.path.join(RES, "closure_ladder_aposteriori.npz")


def r2(pred, ref):
    """Coefficient of determination of pred against ref (node_001 convention)."""
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[m], ref[m]
    ss_res = np.sum((pred - ref) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return 1.0 - ss_res / ss_tot, ss_res, ss_tot


def relrms(pred, ref):
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[m], ref[m]
    return np.sqrt(np.mean((pred - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))


def pearson(pred, ref):
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    m = np.isfinite(pred) & np.isfinite(ref)
    return float(np.corrcoef(pred[m], ref[m])[0, 1])


def score_rung(d, prefix, dns_x, dns_cf, dns_x_reatt):
    """Return a dict of a-posteriori metrics for one rung, or None if absent."""
    if not bool(d.get(f"{prefix}_present", False)):
        return None
    x = np.asarray(d[f"{prefix}_x"], float)
    cf = np.asarray(d[f"{prefix}_cf"], float)
    # DNS Cf interpolated onto this rung's own stations (same convention as
    # coupled_wallstress_transmission.npz / node_001).
    cf_dns_on = np.interp(x, dns_x, dns_cf)
    R2, ssr, sst = r2(cf, cf_dns_on)
    out = dict(
        present=True,
        n_stations=int(x.size),
        R2_cf=float(R2),
        relrms_cf=float(relrms(cf, cf_dns_on)),
        pearson_cf=float(pearson(cf, cf_dns_on)),
        x_reatt=float(d[f"{prefix}_x_reatt"]),
        reatt_abs_err=float(d[f"{prefix}_reattachment_abs_err"]),
        reatt_rel_err_pct=float(d[f"{prefix}_reattachment_rel_err_pct"]),
        eps_median=float(d.get(f"{prefix}_eps_median", np.nan)),
        frac_below_1=float(d.get(f"{prefix}_frac_below_1", np.nan)),
        profile_rms_mean=float(d.get(f"{prefix}_profile_rms_mean", np.nan)),
        x=x, cf=cf, cf_dns_on=cf_dns_on,
    )
    return out


def main():
    if not os.path.exists(NPZ_IN):
        raise SystemExit(f"[ERR] missing {NPZ_IN} -- run the WMLES harvest first")
    d = np.load(NPZ_IN, allow_pickle=True)
    d = {k: d[k] for k in d.files}

    dns_x = np.asarray(d["dns_x"], float)
    dns_cf = np.asarray(d["dns_cf"], float)
    dns_x_reatt = float(d["dns_x_reatt"])
    dns_x_sep = float(d["dns_x_sep"])
    apriori_eps = float(d.get("apriori_median_eps", np.nan))

    # The ladder.  'eq' = equilibrium ODE rung; 'tble' = non-equilibrium ODE rung.
    rungs = [
        ("eq", "equilibrium ODE (no dp/dx, no convection)"),
        ("tble", "TBLE / non-eq ODE (keeps dp/dx, drops convection)"),
    ]

    print("=" * 74)
    print("CLOSURE-LADDER A-POSTERIORI EXPERIMENT  --  periodic hills, Re_H=10595")
    print("=" * 74)
    print(f"DNS reference:  x_sep/H={dns_x_sep:.3f}  x_reatt/H={dns_x_reatt:.3f}")
    print(f"a-priori median eps (corrected, wall-surface protocol) = {apriori_eps:.4f}")
    print()
    print(f"{'rung':<34}{'n':>4}{'R2(Cf)':>10}{'relRMS':>9}"
          f"{'x_reatt':>9}{'relErr%':>9}{'eps_med':>9}{'URMS':>7}")
    print("-" * 91)

    out = dict(
        dns_x=dns_x, dns_cf=dns_cf, dns_x_reatt=dns_x_reatt, dns_x_sep=dns_x_sep,
        apriori_median_eps=apriori_eps,
    )
    present_rungs = []
    for prefix, label in rungs:
        s = score_rung(d, prefix, dns_x, dns_cf, dns_x_reatt)
        out[f"{prefix}_present"] = bool(s is not None)
        if s is None:
            print(f"{label:<34}{'--  (solve not yet complete: present=False)':>57}")
            continue
        present_rungs.append(prefix)
        for k, v in s.items():
            if k in ("present",):
                out[f"{prefix}_present"] = True
            else:
                out[f"{prefix}_{k}"] = v
        print(f"{label:<34}{s['n_stations']:>4}{s['R2_cf']:>10.3f}"
              f"{s['relrms_cf']:>9.3f}{s['x_reatt']:>9.3f}"
              f"{s['reatt_rel_err_pct']:>+9.1f}{s['eps_median']:>9.3f}"
              f"{s['profile_rms_mean']:>7.3f}")

    out["present_rungs"] = np.array(present_rungs, dtype=object)

    # -- the headline contrast: does the dp/dx upgrade rescue the geometry? -----
    if "eq" in present_rungs and "tble" in present_rungs:
        d_reatt = abs(out["tble_reatt_rel_err_pct"]) - abs(out["eq_reatt_rel_err_pct"])
        verdict = ("dp/dx upgrade does NOT rescue the geometry"
                   if abs(out["tble_reatt_rel_err_pct"]) > 5.0
                   else "dp/dx upgrade closes the gap (UNEXPECTED)")
        out["ladder_complete"] = True
        out["tble_minus_eq_abs_relerr_pp"] = float(d_reatt)
        out["headline_verdict"] = verdict
        print("-" * 91)
        print(f"VERDICT: {verdict}")
        print(f"  |relErr| change eq -> tble = {d_reatt:+.1f} percentage points")
        print(f"  (mechanism predicts ~no rescue: failure is convective, not dp/dx)")
    else:
        out["ladder_complete"] = False
        print("-" * 91)
        print("LADDER INCOMPLETE: TBLE rung still solving -- verdict deferred,")
        print("no number fabricated (tble_present=%s)." % out["tble_present"])

    np.savez(NPZ_OUT, **{k: v for k, v in out.items()})
    print(f"\nSaved -> {os.path.relpath(NPZ_OUT, ROOT)}")
    return out


if __name__ == "__main__":
    main()
