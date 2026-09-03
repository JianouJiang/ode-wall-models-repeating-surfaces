#!/usr/bin/env python3
r"""
thrust27_collapse_l2.py  --  Task 27, Level 2 (Implementation & experiments)
===============================================================================
THE CROSS-GEOMETRY TRANSMISSION-COLLAPSE EXPERIMENT.

This is the load-bearing L2 deliverable the L0/L1 Judge flagged: land the
five-point cross-geometry collapse that the L1 methodology only specified.  It

  (Stage A)  reproduces the three on-disk Task 17 y_crit anchors
             (krank=15.93, periodic_hills_1p0=1.0, conv_div=207.9) by re-running
             the SAME variance-balance sweep (critical_matching_height.py
             functions, production ODE) -> a protocol-fidelity GATE;

  (Stage B)  extends y_crit to the two missing Xiao coupled cases (alpha=0.8,
             1.2) AND recomputes alpha=1.0, all from the *29-case* Xiao DNS
             (alph075 / alph10-9 / alph125) -- the SAME reference fields whose
             a-priori eps (0.098 / 0.149 / 0.205) populate
             aposteriori_dose_response.npz (verified to machine precision).
             [The 5-case case_0p8/case_1p2 ASCII files carry a broken
             ~zero pressure column -> unusable; see implementation.md Sec.3.]
             It tests the steepness monotonicity y_crit(0.75) > (1.0) > (1.25);

  (Stage C)  runs the collapse on the five real coupled a-posteriori points:
             deployed matching height y_m+ from each WMLES yPlus function object
             (time-mean over the stationary window), the transmission ratio
             r = y_m+/y_crit+, and the cross-geometry Spearman correlations
             rho(eps,e) [F1], rho(r,e) [F3 candidate], rho(y_m+,e) and
             rho(y_crit+,e) [the n=5 degeneracy diagnostic, L1 Sec.3.4], with a
             leave-one-out stability scan.

HONESTY (G1/G2/G4):
  * Every number traces to read-only DNS/LES or to a committed coupled WMLES run.
  * y_crit uses PER-GEOMETRY eps_c (the #17 protocol), never a fixed 0.22.
  * a-priori (reference fields) for y_crit; a-posteriori (deployed yPlus) for y_m+.
  * n=5 is reported as a mechanism-consistency check, never a powered regression;
    the degeneracy r-vs-y_m+ is reported explicitly, pass or fail.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/thrust27_collapse_l2.py
Out:  codes/results/thrust27_collapse.npz
"""
import os
import sys
import glob
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
ROOT = os.path.dirname(CODES)
RESULTS = os.path.join(CODES, "results")
NDD = os.path.join(CODES, "new_data_download")
GEOM = os.path.join(NDD, "geometry_driven")
B29 = os.path.join(GEOM, "xiao_pehill_parameterized", "pehill-29-cases-DNS")
OF = os.path.join(CODES, "openfoam")

sys.path.insert(0, os.path.join(HERE))
sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
# reuse the EXACT Task 17 sweep + crossing (production ODE inside)
import critical_matching_height as cmh           # noqa: E402
from dose_response_xiao import read_case, evaluate_case  # noqa: E402

RE_B = 5600.0
NU_X = 1.0 / RE_B


# ---------------------------------------------------------------------------
# deployed matching height y_m+ from each coupled WMLES yPlus function object
# ---------------------------------------------------------------------------
def deployed_ym_plus(case_dir, patch="bottomWall"):
    """Time-mean of the patch-average y+ over the stationary window (t>0).

    The yPlus FO reports the wall-adjacent cell-centre y+ that the wall model
    reads -- i.e. the deployed matching height in wall units.  t=0 is the
    initial-condition value (excluded); the remaining dumps are statistically
    stationary (see implementation.md)."""
    files = sorted(glob.glob(os.path.join(OF, case_dir,
                                          "postProcessing", "yPlus", "*", "yPlus.dat")))
    rows = {}
    for f in files:
        for line in open(f):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 5 or p[1] != patch:
                continue
            try:
                rows[float(p[0])] = float(p[4])    # col4 = patch-average y+
            except ValueError:
                pass
    t = np.array(sorted(rows))
    v = np.array([rows[x] for x in t])
    win = t > 0.0
    return float(np.mean(v[win])), float(np.std(v[win])), int(win.sum())


# ---------------------------------------------------------------------------
# y_crit from the production variance-balance sweep
# ---------------------------------------------------------------------------
def ycrit_from_file(path):
    """Stage-A protocol on an on-disk *_wall_profiles.npz (krank/conv_div/1p0)."""
    profs, tau, dpdx = cmh.load_geom(path)
    relrms, r2arr, epsmed, nval = cmh.sweep_geometry(profs, tau, dpdx)
    ycrit, eps_c, bif, ycrit_pl, p_pl = cmh.crossing(cmh.YMP_GRID, relrms, epsmed)
    return ycrit, eps_c, relrms, r2arr, epsmed


def ycrit_from_xiao29(case_name):
    """Stage-B: build per-station profiles from a 29-case dir (the F1 reference)
    and run the IDENTICAL sweep+crossing."""
    case = read_case(os.path.join(B29, case_name))
    profs = [(np.asarray(case["y"][i], float), np.asarray(case["U"][i], float), NU_X)
             for i in range(len(case["x"]))]
    tau = np.asarray(case["tau_w"], float)
    dpdx = np.asarray(case["dp_dx"], float)
    relrms, r2arr, epsmed, nval = cmh.sweep_geometry(profs, tau, dpdx)
    ycrit, eps_c, bif, ycrit_pl, p_pl = cmh.crossing(cmh.YMP_GRID, relrms, epsmed)
    ev = evaluate_case(case)                       # for the Y_IDX eps_med gate
    return ycrit, eps_c, ev["eps_median"], relrms, epsmed


def main():
    print("=" * 96)
    print("THRUST #27 L2 -- cross-geometry transmission collapse  (real DNS + coupled WMLES)")
    print("=" * 96)

    # ----- Stage A: reproduce the three #17 anchors (protocol-fidelity gate) ----
    print("\n[Stage A] reproduce the on-disk Task 17 y_crit anchors "
          "(variance-balance, production ODE)")
    m17 = np.load(os.path.join(RESULTS, "critical_matching_height_map.npz"),
                  allow_pickle=True)
    keys17 = [str(k) for k in m17["keys"]]
    anchor_paths = {
        "krank_pehill_Re10595":
            os.path.join(GEOM, "krank_pehill_Re10595_wall_profiles.npz"),
        "periodic_hills_1p0":
            os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
        "conv_div_channel":
            os.path.join(NDD, "conv_div_channel_Re12600_wall_profiles.npz"),
    }
    gateA = True
    anchor_yc = {}
    print(f"  {'geometry':22s} {'y_crit+ (recomp)':>16s} {'#17 map':>9s} "
          f"{'eps_c':>7s} {'#17 eps_c':>9s} {'gate':>6s}")
    for g, path in anchor_paths.items():
        yc, ec, *_ = ycrit_from_file(path)
        i = keys17.index(g)
        yc17 = float(m17["ycrit"][i]); ec17 = float(m17["eps_c"][i])
        # tolerance: 5% on finite y_crit, and eps_c within 5%
        ok = (np.isclose(yc, yc17, rtol=0.05, atol=0.2) and
              (np.isnan(ec) and np.isnan(ec17) or np.isclose(ec, ec17, rtol=0.05, atol=0.02)))
        gateA = gateA and ok
        anchor_yc[g] = yc
        print(f"  {g:22s} {yc:16.2f} {yc17:9.2f} {ec:7.3f} {ec17:9.3f} "
              f"{'OK' if ok else 'FAIL':>6s}")
    print(f"  Stage-A gate: {'PASS' if gateA else 'FAIL'} "
          f"(instrument reproduces the #17 protocol)")

    # ----- Stage B: y_crit for the three Xiao coupled cases (29-case reference) --
    print("\n[Stage B] y_crit for the Xiao coupled cases from the 29-case DNS "
          "(same reference as the a-priori eps in aposteriori_dose_response.npz)")
    xiao_map = {                                   # coupled label -> 29-case dir, eps on disk
        "xiao_a0p8": ("alph075-80355-3036", 0.75, 0.09824),
        "xiao_a1p0": ("alph10-9-3036",      1.00, 0.14909),
        "xiao_a1p2": ("alph125-99645-3036", 1.25, 0.20491),
    }
    xiao_yc = {}
    gateB = True
    print(f"  {'case':12s} {'29-case dir':22s} {'alpha':>5s} {'y_crit+':>8s} "
          f"{'eps_c':>7s} {'eps@Yidx':>9s} {'eps on-disk':>11s} {'gate':>5s}")
    for lab, (d, al, eps_disk) in xiao_map.items():
        yc, ec, eps_yidx, *_ = ycrit_from_xiao29(d)
        ok = np.isclose(eps_yidx, eps_disk, rtol=1e-3)   # reference-fidelity gate
        gateB = gateB and ok
        xiao_yc[lab] = (yc, ec, al)
        print(f"  {lab:12s} {d:22s} {al:5.2f} {yc:8.2f} {ec:7.3f} "
              f"{eps_yidx:9.5f} {eps_disk:11.5f} {'OK' if ok else 'FAIL':>5s}")
    print(f"  Stage-B reference-fidelity gate: {'PASS' if gateB else 'FAIL'}")

    # steepness monotonicity (L1 Sec.3.2): y_crit(0.75) > (1.0) > (1.25)
    yc075 = xiao_yc["xiao_a0p8"][0]
    yc100 = xiao_yc["xiao_a1p0"][0]
    yc125 = xiao_yc["xiao_a1p2"][0]
    mono = (yc075 > yc100 > yc125)
    print(f"  steepness monotonicity y_crit(0.75)={yc075:.2f} > (1.0)={yc100:.2f} "
          f"> (1.25)={yc125:.2f}  -> {'HOLDS' if mono else 'FALSIFIED'}")

    # ----- Stage C: the five-point cross-geometry collapse ----------------------
    print("\n[Stage C] five-point cross-geometry collapse "
          "(deployed y_m+ from coupled WMLES yPlus FO)")
    dr = np.load(os.path.join(RESULTS, "aposteriori_dose_response.npz"),
                 allow_pickle=True)
    labels = [str(x) for x in dr["labels"]]
    e_reatt = np.asarray(dr["e_reatt"], float)
    eps_med = np.asarray(dr["eps_med"], float)

    # case order MUST match aposteriori_dose_response.npz labels
    case_dirs = ["xiao_wmles_a0p8", "xiao_wmles_a1p0", "xiao_wmles_a1p2",
                 "pehill_wmles", "convdiv_wmles"]
    ycrit5 = [xiao_yc["xiao_a0p8"][0], xiao_yc["xiao_a1p0"][0], xiao_yc["xiao_a1p2"][0],
              anchor_yc["krank_pehill_Re10595"], anchor_yc["conv_div_channel"]]
    ycrit5 = np.array(ycrit5, float)

    ymp, ymp_sd, ymp_n = [], [], []
    for cd in case_dirs:
        m, s, n = deployed_ym_plus(cd)
        ymp.append(m); ymp_sd.append(s); ymp_n.append(n)
    ymp = np.array(ymp, float)
    r = ymp / ycrit5

    print(f"  {'case':36s} {'eps_med':>8s} {'y_crit+':>8s} {'y_m+ (dep)':>11s} "
          f"{'r=ym/yc':>8s} {'e_reatt':>8s}")
    for i, lab in enumerate(labels):
        print(f"  {lab:36s} {eps_med[i]:8.3f} {ycrit5[i]:8.2f} "
              f"{ymp[i]:6.2f}+-{ymp_sd[i]:4.2f} {r[i]:8.3f} {e_reatt[i]:8.3f}")

    # cross-geometry Spearman correlations
    rho_eps, p_eps = spearmanr(eps_med, e_reatt)        # F1 (on disk ~ +0.10)
    rho_r, p_r = spearmanr(r, e_reatt)                  # F3 candidate
    rho_ym, p_ym = spearmanr(ymp, e_reatt)              # degeneracy diag
    rho_yc, p_yc = spearmanr(ycrit5, e_reatt)           # y_crit alone
    print("\n  cross-geometry Spearman rho with coupled e_reatt (n=5):")
    print(f"    F1  rho(eps_med, e)   = {rho_eps:+.3f}  (p={p_eps:.3f})   "
          f"[on-disk anchor +0.10]")
    print(f"    F3  rho(r=ym/yc, e)   = {rho_r:+.3f}  (p={p_r:.3f})   "
          f"<- candidate transmission law")
    print(f"        rho(y_m+, e)      = {rho_ym:+.3f}  (p={p_ym:.3f})   "
          f"<- DEGENERACY check (L1 Sec.3.4)")
    print(f"        rho(y_crit+, e)   = {rho_yc:+.3f}  (p={p_yc:.3f})")
    degenerate = abs(rho_r - rho_ym) < 1e-9
    print(f"    r vs y_m+ rank-degenerate?  {degenerate}  "
          f"(|drho|={abs(rho_r-rho_ym):.3f})")

    # leave-one-out stability of rho(r,e)
    loo = []
    for k in range(5):
        idx = [j for j in range(5) if j != k]
        loo.append(spearmanr(r[idx], e_reatt[idx])[0])
    loo = np.array(loo)
    print(f"    LOO rho(r,e): {np.array2string(loo, precision=3)}  "
          f"min={loo.min():+.3f} max={loo.max():+.3f}")

    # the headline ordering test: does r RANK the errors correctly?
    order_r = np.argsort(r)
    order_e = np.argsort(e_reatt)
    print(f"\n  rank by r   (low->high): {[labels[i].split('(')[0].strip() for i in order_r]}")
    print(f"  rank by e   (low->high): {[labels[i].split('(')[0].strip() for i in order_e]}")

    # ----- save ----------------------------------------------------------------
    out = os.path.join(RESULTS, "thrust27_collapse.npz")
    np.savez(
        out,
        labels=np.array(labels, object),
        case_dirs=np.array(case_dirs, object),
        e_reatt=e_reatt, eps_med=eps_med,
        y_crit_plus=ycrit5, y_m_plus=ymp, y_m_plus_sd=np.array(ymp_sd),
        y_m_plus_nwin=np.array(ymp_n), ratio=r,
        rho_eps=float(rho_eps), rho_ratio=float(rho_r),
        rho_ym=float(rho_ym), rho_ycrit=float(rho_yc),
        p_eps=float(p_eps), p_ratio=float(p_r), p_ym=float(p_ym), p_ycrit=float(p_yc),
        loo_rho_ratio=loo, degenerate=bool(degenerate),
        ycrit_xiao=np.array([yc075, yc100, yc125]),
        ycrit_alpha=np.array([0.75, 1.00, 1.25]),
        steepness_monotone=bool(mono),
        stageA_gate=bool(gateA), stageB_gate=bool(gateB),
        note=("Task 27 cross-geometry transmission collapse. y_crit per "
              "geometry via #17 variance-balance sweep (production ODE). Xiao "
              "0.8/1.0/1.2 from 29-case DNS (alph075/alph10-9/alph125), the same "
              "reference as aposteriori_dose_response.npz eps. y_m+ = time-mean "
              "bottomWall yPlus FO over the stationary window. a-priori y_crit, "
              "a-posteriori y_m+. n=5 = mechanism-consistency check, not a "
              "powered regression."),
    )
    print(f"\nSaved -> results/{os.path.basename(out)}")
    print(f"\nGATES: Stage-A {'PASS' if gateA else 'FAIL'}  "
          f"Stage-B {'PASS' if gateB else 'FAIL'}  "
          f"monotonicity {'HOLDS' if mono else 'FALSIFIED'}")


if __name__ == "__main__":
    main()
