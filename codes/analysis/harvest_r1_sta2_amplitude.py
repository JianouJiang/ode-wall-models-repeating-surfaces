#!/usr/bin/env python3
"""harvest_r1_sta2_amplitude.py -- the R1-STA-2 wavy-wall AMPLITUDE LADDER.

Combines the deposited mild ladder (2a/lambda = 0.10, ARCHER2, three grids, already
certified in codes/results/r1_sta2_wavy_wrles_<date>.json) with the new steep runs
(2a/lambda = 0.20, Oxford ARC) into one artifact

    codes/results/r1_sta2_wavy_amplitude_<date>.{json,npz}

EVERY per-case number is produced by importing harvest_r1_sta2 and calling the SAME
functions (`analyse_window`, `block_windows`, `summarise_blocks`, `load_hudson`,
`load_maass`, `reference_cross_l2`, and the shared ODE instrument `rib_eps_ode.evaluate`),
so the two amplitudes are metric-for-metric comparable by construction: identical
matching heights, identical wall-origin recovery, identical six-block uncertainty
treatment, identical epsilon and balance definitions.

Validation status is asymmetric and is recorded as such, never blurred:
  * 2a/lambda = 0.10 -- validated against ERCOFTAC cases 76 (Hudson LDV) and 77
    (Maass-Schumann DNS), both hash-pinned.
  * 2a/lambda = 0.20 -- **NO public reference dataset exists at this steepness and
    Reynolds number** (ERCOFTAC cases 076-083 enumerated 2026-08-24: only the 0.10 pair
    is hosted; Buckles, Hanratty & Adrian 1984 measured 2a/lambda = 0.20 but at
    Re_b = 12000, unmatched, and deposited no machine-readable data).  Validation
    therefore TRANSFERS from the mild case through identical numerics, grid policy and
    harvest.  The artifact states this explicitly in `validation_status` and the
    verifier refuses to report the steep family as reference-validated.

Usage:  python3 codes/analysis/harvest_r1_sta2_amplitude.py [--date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import harvest_r1_sta2 as H  # noqa: E402  (single source of truth for every metric)

SCHEMA = "r1_sta2_wavy_amplitude_v1"
LEDGER_ROW = "R1-STA-2"
STEEP_RUNS = RESULTS / "r1_sta2_wavy_amplitude"
MILD_RUNS = RESULTS / "r1_sta2_wavy_wrles"
IDEA = ("Doubling the wave amplitude at fixed wavelength, Reynolds number, domain, solver, "
        "schemes, SGS model, matching-height convention and harvest isolates amplitude as the "
        "single variable that decides whether the pressure-gradient ODE wall model fails.")
# steepness references, consistency notes only -- never gates (see validation_status)
STEEP_LITERATURE = {
    "buckles_hanratty_adrian_1984": dict(
        two_a_over_lambda=0.20, Re_b=12000.0, technique="LDV",
        note="same steepness, UNMATCHED Reynolds number, no machine-readable data deposited"),
    "dns_A_over_P_0p4_2025": dict(
        two_a_over_lambda=0.40, Re_b=5600.0, x_detach=0.068, x_reattach=0.860,
        note="steeper wave, different lambda/delta; brackets the expected separation extent"),
}


def load_family(runs_dir: Path, prefix: str, grids):
    out = {}
    for g in grids:
        cands = sorted(runs_dir.glob("%s%s_v*" % (prefix, g)))
        loaded = None
        for c in cands[::-1]:
            loaded = H.load_grid(c)
            if loaded:
                break
        if loaded is not None:
            out[g] = loaded
    return out


def analyse_family(fam, hudson, maass, cross, expected_end):
    """Run the SHARED per-case analysis on every grid of one amplitude family."""
    res = {}
    for g, G in fam.items():
        geo, snaps, ts = G["geo"], G["snaps"], G["timeseries"]
        t0 = float(geo["avg_start"])
        final = snaps[-1]
        t_end = float(final["time"])

        def gradP_between(a, b):
            if ts is None:
                return float("nan")
            m = (ts["time"] > a) & (ts["time"] <= b)
            return float(np.mean(ts["gradP"][m])) if m.any() else float("nan")

        cum = dict(final)
        cum.update(t_start=t0, t_end=t_end, kind="cumulative")
        R = H.analyse_window(cum, geo, gradP_between(t0, t_end), hudson, maass)
        blocks = [b for b in H.block_windows(snaps, t0)
                  if b["t_end"] - b["t_start"] >= 0.5 * H.BLOCK]
        block_res = [H.analyse_window(b, geo, gradP_between(b["t_start"], b["t_end"]),
                                      hudson, maass) for b in blocks]
        ustar = R["ustar_wavy"]
        lv = float(geo["nu"]) / max(ustar, 1e-30)
        resol = dict(dx_plus=float(geo["dx"]) / lv, dz_plus=float(geo["dz"]) / lv,
                     # meshed-wall origin, as in harvest_r1_sta2 -- the analytic
                     # cosine is not the surface the solver computes traction on,
                     # and the two differ by more than a cell height at 2a/lambda=0.20
                     y1_plus=R["dy1_mesh_max"] / lv * 2.0,
                     y1_plus_analytic_origin=float(np.max(final["dy_first_cell"])) / lv * 2.0,
                     dy1_mesh_min_over_max=R["dy1_mesh_min"] / R["dy1_mesh_max"],
                     dy_mid_plus=float(geo["dy_mid"]) / lv,
                     nut_wall_max_over_nu=(float(final["nut_wall_bottom_max"]) / float(geo["nu"])
                                           if "nut_wall_bottom_max" in final else float("nan")),
                     dt_window_mean=(float(np.mean(ts["deltaT"][ts["time"] > t0]))
                                     if ts is not None else float("nan")),
                     CoMax_window_max=(float(np.max(ts["CoMax"][ts["time"] > t0]))
                                       if ts is not None else float("nan")),
                     averaging_window=[t0, t_end],
                     flow_throughs=(t_end - t0) / (float(geo["Lx"]) / float(geo["Ub"])),
                     eddy_turnovers=(t_end - t0) * ustar / float(geo["delta"]))
        unc = {k: H.summarise_blocks(block_res, k) for k in
               ("x_sep", "x_re", "ustar_wavy", "tau_A1", "tau_phi1", "form_fraction", "f_reversed",
                "hudson_U_l2_median", "maass_U_l2_median")}
        unc_ode = {str(e): {k: H.summarise_blocks(block_res, k, sub=e) for k in
                            ("standard_ml", "controlled_ml", "controlled_dns",
                             "controlled_dns_total", "eps_median", "frac_eps_lt1", "pi_median")}
                   for e in H.ETA_MATCH_TARGETS}
        r2s = [R["ode"][e]["standard_ml"] for e in H.ETA_MATCH_TARGETS]
        eps = [R["ode"][e]["eps_median"] for e in H.ETA_MATCH_TARGETS]
        dns = [R["ode"][e]["controlled_dns"] for e in H.ETA_MATCH_TARGETS]
        res[g] = dict(
            case_id=G["case_dir"].name, grid=g, cells=int(geo["n_cells"]),
            mesh=[geo["nx"], geo["ny"], geo["nz"]],
            two_a_over_lambda=float(geo.get("two_a_over_lambda", 0.10)),
            a_over_delta=float(geo.get("a_over_delta", 0.10)),
            S_star=float(geo.get("S_star", np.pi * float(geo["amplitude"]) / float(geo["lambda_"]))),
            Re_h=float(geo["Re_h"]), lambda_over_delta=float(geo["lambda_over_delta"]),
            converged=bool(abs(t_end - expected_end) < 1e-6), t_end=t_end,
            machine=G["manifest"].get("machine", "archer2"),
            slurm_job_id=G["manifest"].get("slurm_job_id"),
            nodes=G["manifest"].get("nodes"), ranks=G["manifest"].get("ranks"),
            resolution=resol, n_blocks=len(block_res),
            wall=dict(x_sep=R["x_sep"], x_re=R["x_re"], f_reversed=R["f_reversed"],
                      bubble_length=float((R["x_re"] - R["x_sep"]) % 1.0),
                      ustar_wavy=R["ustar_wavy"], Re_tau_wavy=R["Re_tau_wavy"],
                      form_fraction=R["form_fraction"], momentum_closure_rel=R["momentum_closure_rel"],
                      tau_A1=R["tau_A1"], tau_phi1_deg=float(np.degrees(R["tau_phi1"])),
                      pw_A1=R["pw_A1"], pw_phi1_deg=float(np.degrees(R["pw_phi1"])),
                      wall_origin_fit_residual_over_dy1=R["wall_origin_fit_residual_over_dy1"]),
            validation=dict(hudson_U_l2_median=R["hudson_U_l2_median"],
                            maass_U_l2_median=R["maass_U_l2_median"]),
            ode={str(e): {k: v for k, v in R["ode"][e].items()
                          if not isinstance(v, (np.ndarray, dict))} for e in H.ETA_MATCH_TARGETS},
            verdict=dict(standard_ml_r2_by_eta={str(e): v for e, v in zip(H.ETA_MATCH_TARGETS, r2s)},
                         eps_median_by_eta={str(e): v for e, v in zip(H.ETA_MATCH_TARGETS, eps)},
                         controlled_dns_r2_by_eta={str(e): v for e, v in zip(H.ETA_MATCH_TARGETS, dns)},
                         ode_fails_all_heights=bool(np.all(np.array(r2s) < 0)),
                         failure_instance=bool(np.all(np.array(r2s) < 0) and np.all(np.array(eps) < 1))),
            uncertainty=dict(block_windows=unc, block_windows_ode=unc_ode, n_blocks=len(block_res)),
            source_hashes=dict(
                reduced={s["_file"].name: H.sha256(s["_file"]) for s in snaps},
                geometry=H.sha256(G["case_dir"] / "GEOMETRY.json")),
            _R=R, _blocks=block_res,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=_dt.date.today().strftime("%Y%m%d"))
    args = ap.parse_args()
    hudson, maass = H.load_hudson(), H.load_maass()
    cross = H.reference_cross_l2(hudson, maass)

    mild = load_family(MILD_RUNS, "r1sta2_wavy_", ("G0", "G1", "G2"))
    mild_arc = load_family(STEEP_RUNS, "r1sta2_arcmild_", ("G0",))
    steep = load_family(STEEP_RUNS, "r1sta2_steep_", ("S0", "S1"))
    if not steep:
        raise SystemExit("no steep (2a/lambda=0.20) reduced data under %s -- nothing harvested "
                         "(no fabrication)" % STEEP_RUNS)

    fam = {}
    fam["mild_archer2"] = analyse_family(mild, hudson, maass, cross, 200.0)
    if mild_arc:
        fam["mild_arc"] = analyse_family(mild_arc, hudson, maass, cross, 200.0)
    fam["steep_arc"] = analyse_family(steep, hudson, maass, cross, 150.0)

    out = dict(schema=SCHEMA, ledger_row=LEDGER_ROW, idea=IDEA,
               generated=_dt.datetime.now().isoformat(timespec="seconds"),
               eta_m_targets=list(H.ETA_MATCH_TARGETS), tolerances=H.TOL,
               hudson_dns_reference_l2=cross, steep_literature=STEEP_LITERATURE,
               validation_status=dict(
                   mild=dict(reference_validated=True,
                             references=["ERCOFTAC case 76 (Hudson, Dykhno & Hanratty 1996)",
                                         "ERCOFTAC case 77 (Maass & Schumann 1996)"],
                             certificate="codes/results/r1_sta2_wavy_wrles_20260824.json"),
                   steep=dict(reference_validated=False,
                              reason="No public reference dataset exists at 2a/lambda=0.20 and "
                                     "Re_h=3460. ERCOFTAC cases 076-083 were enumerated on "
                                     "2026-08-24 and host only the 2a/lambda=0.10 pair. Buckles, "
                                     "Hanratty & Adrian (1984) measured this steepness but at "
                                     "Re_b=12000 (unmatched) and deposited no machine-readable "
                                     "data; matching the mild case's Reynolds number is required "
                                     "for the amplitude ladder and takes precedence.",
                              validation_transfer="identical solver, schemes, SGS model, PIMPLE "
                                                  "settings, forcing, dt/CFL policy, wall-normal "
                                                  "grading law, spline density, domain shape, "
                                                  "Reynolds number, averaging-window policy, "
                                                  "reduction and harvest as the reference-validated "
                                                  "2a/lambda=0.10 case",
                              consistency_notes_only=list(STEEP_LITERATURE))),
               families={}, npz_keys=[])
    npz = {}
    for fname, F in fam.items():
        out["families"][fname] = {g: {k: v for k, v in d.items() if not k.startswith("_")}
                                  for g, d in F.items()}
        for g, d in F.items():
            R = d["_R"]
            tag = "%s_%s" % (fname, g)
            npz[tag + "_phase"] = R["phase"]
            npz[tag + "_tau_t"] = R["tau_t"]
            npz[tag + "_p_wall"] = R["p_wall"]
            npz[tag + "_y_wall_mesh"] = R["y_wall_mesh"]
            for e in H.ETA_MATCH_TARGETS:
                npz["%s_eta%g_tau_ref" % (tag, e)] = R["ode"][e]["tau_ref"]
                npz["%s_eta%g_eps" % (tag, e)] = R["ode"][e]["eps"]
                for k in ("standard_ml", "controlled_dns"):
                    npz["%s_eta%g_pred_%s" % (tag, e, k)] = R["ode"][e]["pred"][k]
            npz[tag + "_block_r2"] = np.array(
                [[b["ode"][e]["standard_ml"] for e in H.ETA_MATCH_TARGETS] for b in d["_blocks"]])
    out["npz_keys"] = sorted(npz)

    # ---------------- the amplitude comparison itself ----------------
    def finest(fname):
        F = out["families"].get(fname, {})
        return F[sorted(F, key=lambda g: F[g]["cells"])[-1]] if F else None

    m, s = finest("mild_archer2"), finest("steep_arc")
    ma = finest("mild_arc")
    ladder = dict(
        mild=dict(family="mild_archer2", grid=m["grid"], cells=m["cells"],
                  two_a_over_lambda=m["two_a_over_lambda"], S_star=m["S_star"],
                  machine=m["machine"], **{k: m["wall"][k] for k in
                                           ("x_sep", "x_re", "bubble_length", "f_reversed",
                                            "ustar_wavy", "form_fraction")},
                  standard_ml_r2_by_eta=m["verdict"]["standard_ml_r2_by_eta"],
                  eps_median_by_eta=m["verdict"]["eps_median_by_eta"],
                  failure_instance=m["verdict"]["failure_instance"]),
        steep=dict(family="steep_arc", grid=s["grid"], cells=s["cells"],
                   two_a_over_lambda=s["two_a_over_lambda"], S_star=s["S_star"],
                   machine=s["machine"], **{k: s["wall"][k] for k in
                                            ("x_sep", "x_re", "bubble_length", "f_reversed",
                                             "ustar_wavy", "form_fraction")},
                   standard_ml_r2_by_eta=s["verdict"]["standard_ml_r2_by_eta"],
                   eps_median_by_eta=s["verdict"]["eps_median_by_eta"],
                   failure_instance=s["verdict"]["failure_instance"]),
        amplitude_ratio=s["two_a_over_lambda"] / m["two_a_over_lambda"],
        matched=dict(Re_h=(abs(s["Re_h"] - m["Re_h"]) < 1e-9),
                     lambda_over_delta=(abs(s["lambda_over_delta"] - m["lambda_over_delta"]) < 1e-9)),
        same_cluster_tiepoint=(ma is not None))
    if ma is not None:
        ladder["mild_arc_tiepoint"] = dict(
            grid=ma["grid"], cells=ma["cells"], machine=ma["machine"],
            x_sep=ma["wall"]["x_sep"], x_re=ma["wall"]["x_re"], ustar_wavy=ma["wall"]["ustar_wavy"],
            standard_ml_r2_by_eta=ma["verdict"]["standard_ml_r2_by_eta"])
        m0 = out["families"]["mild_archer2"].get("G0")
        if m0:
            ladder["cross_cluster_check"] = dict(
                archer2_G0=dict(x_sep=m0["wall"]["x_sep"], x_re=m0["wall"]["x_re"],
                                ustar_wavy=m0["wall"]["ustar_wavy"],
                                r2_eta0p1=m0["verdict"]["standard_ml_r2_by_eta"]["0.1"]),
                arc_G0=dict(x_sep=ma["wall"]["x_sep"], x_re=ma["wall"]["x_re"],
                            ustar_wavy=ma["wall"]["ustar_wavy"],
                            r2_eta0p1=ma["verdict"]["standard_ml_r2_by_eta"]["0.1"]),
                d_x_sep=abs(ma["wall"]["x_sep"] - m0["wall"]["x_sep"]),
                d_ustar_rel=abs(ma["wall"]["ustar_wavy"] - m0["wall"]["ustar_wavy"]) / m0["wall"]["ustar_wavy"],
                d_r2_eta0p1=abs(ma["verdict"]["standard_ml_r2_by_eta"]["0.1"] -
                                m0["verdict"]["standard_ml_r2_by_eta"]["0.1"]),
                note="same case, same generator/driver/harvest, two clusters and two OpenFOAM "
                     "builds; bounds any cross-cluster contribution to the amplitude comparison")
    # verdict invariance within the steep family
    S = out["families"]["steep_arc"]
    ladder["steep_grid_invariant"] = bool(
        len({S[g]["verdict"]["failure_instance"] for g in S}) == 1 and len(S) >= 2)
    ladder["steep_n_grids"] = len(S)
    turns = (m["verdict"]["failure_instance"] != s["verdict"]["failure_instance"])
    ladder["amplitude_turns_the_verdict"] = bool(turns)
    ladder["statement"] = (
        "At 2a/lambda=%.2f (S*=%.3f) the pressure-gradient ODE %s (R2 by eta_m/delta = %s); at "
        "2a/lambda=%.2f (S*=%.3f), with wavelength, Reynolds number, domain, numerics, SGS model "
        "and harvest held identical, it %s (R2 = %s). Amplitude %s the verdict; the steep result is "
        "%s across %d grids."
        % (m["two_a_over_lambda"], m["S_star"],
           "FAILS" if m["verdict"]["failure_instance"] else "does NOT fail",
           {k: round(v, 3) for k, v in m["verdict"]["standard_ml_r2_by_eta"].items()},
           s["two_a_over_lambda"], s["S_star"],
           "FAILS" if s["verdict"]["failure_instance"] else "does NOT fail",
           {k: round(v, 3) for k, v in s["verdict"]["standard_ml_r2_by_eta"].items()},
           "TURNS" if turns else "does NOT turn",
           "invariant" if ladder["steep_grid_invariant"] else "NOT invariant", len(S)))
    out["amplitude_ladder"] = ladder
    complete = (len(S) >= 2 and all(S[g]["converged"] for g in S))
    out["status"] = ("R1_STA2_AMPLITUDE_OK" if (complete and ladder["steep_grid_invariant"])
                     else ("R1_STA2_AMPLITUDE_GRID_DISAGREE" if complete
                           else "R1_STA2_AMPLITUDE_PARTIAL_%d_OF_2_STEEP_GRIDS" % len(S)))

    tag = "r1_sta2_wavy_amplitude_%s" % args.date
    np.savez(RESULTS / (tag + ".npz"), **npz)
    (RESULTS / (tag + ".json")).write_text(json.dumps(H.jsonable(out), indent=2))
    print("status:", out["status"])
    for fname, F in out["families"].items():
        for g, d in sorted(F.items(), key=lambda kv: kv[1]["cells"]):
            print("%-13s %-3s 2a/l=%.2f cells=%7d %-7s | x_sep=%.3f x_re=%.3f L_bub=%.3f u*=%.4f "
                  "form=%.2f | y1+=%.2f dx+=%.1f FT=%.0f | R2(eta.1)=%+.3f eps=%.3f | fail=%s"
                  % (fname, g, d["two_a_over_lambda"], d["cells"], d["machine"],
                     d["wall"]["x_sep"], d["wall"]["x_re"], d["wall"]["bubble_length"],
                     d["wall"]["ustar_wavy"], d["wall"]["form_fraction"],
                     d["resolution"]["y1_plus"], d["resolution"]["dx_plus"],
                     d["resolution"]["flow_throughs"],
                     d["verdict"]["standard_ml_r2_by_eta"]["0.1"],
                     d["verdict"]["eps_median_by_eta"]["0.1"], d["verdict"]["failure_instance"]))
    print("\n" + ladder["statement"])
    print("saved ->", RESULTS / (tag + ".json"))


if __name__ == "__main__":
    main()
