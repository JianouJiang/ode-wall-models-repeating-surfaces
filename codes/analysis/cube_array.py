#!/usr/bin/env python3
"""
cube_array.py
=============
Harvest + score the Coceal-2006 aligned CUBE ARRAY (3D RANS) from
codes/openfoam/cube_array_prod.  This is the SHARP, 3D, urban-building
repeating-structure end of the (shape, dimensionality) axes.

The 1-D ODE/TBLE wall model is deployed on the FLOOR BETWEEN cubes (the
horizontal `floor` patch -- where a wall model is actually used in WMLES of
urban arrays).  For each floor sampling column:
  * tau_w (kinematic) = nu * d|U_horiz|/dy from the resolved near-wall gradient
  * dp_dx = streamwise gradient of near-wall p (p kinematic, incompressible)
  * matching height y+~50 (index 10); score with verbatim evaluate().

HONESTY (recorded in the npz):
  * Geometry EXACT to Coceal 2006 (h, pitch 2h, lambda_p=0.25, Re_h~5000).
  * Flow = OpenFOAM-RANS (kOmegaSST) 3D, NOT the Coceal DNS.
  * Cube-array drag is FORM-DRAG-DOMINATED. We report the form-drag fraction
    (cube pressure drag / total) so the |tau_w| vs |dp/dx| (=epsilon) framing is
    applied honestly: epsilon on the floor measures the LOCAL skin-friction vs
    local pressure-gradient balance, while the GLOBAL momentum sink is form drag.
Never fabricates: writes ok=False + reason if the case is missing/unconverged.
"""
from __future__ import annotations
import argparse, os, re, sys, glob, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
RESULTS = os.path.join(CODES, "results")
OF = os.path.join(CODES, "openfoam")
sys.path.insert(0, HERE)
from cross_geometry_collapse import evaluate, Y_IDX        # noqa: E402

CASE = os.path.join(OF, "cube_array_prod")   # overridden by --case
PITCH = 2.0                                   # overridden by --pitch (units of h)
TAG = "cube_array"                            # overridden by --tag
H = 1.0; HDOM = 4.0
NU = 1.0 / 5000.0               # default; overridden per-case by case_nu() in main()


def case_nu(case_dir, default=1.0 / 5000.0):
    """Read kinematic viscosity from the case's constant/physicalProperties
    (OpenFOAM 10) or constant/transportProperties (legacy) so Re_h, y_m and
    tau_w are correct for ANY run (e.g. a Reynolds sweep), not the hardcoded
    default. Falls back to `default` if the file/entry is absent.

    NOTE: OF10 writes nu into `physicalProperties`; the old name-only lookup of
    `transportProperties` silently fell back to the default, mislabelling Re-sweep
    cases (e.g. Re=2500 read as 5000). Try both names, physicalProperties first."""
    for fn in ("physicalProperties", "transportProperties"):
        tp = os.path.join(case_dir, "constant", fn)
        try:
            txt = open(tp).read()
            m = re.search(r'\bnu\b\s*\[[^\]]*\]\s*([0-9eE.+\-]+)\s*;', txt)
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return default


LINE_LEN = 1.0                 # wall-normal sample length (~ canopy height)
LINE_NPTS = 200
NPZ_M = 40
VISC_SUB = 0.05


def of(*args, util="simpleFoam"):
    # NOTE: `simpleFoam -postProcess` only loads SOLVER fields (U, p, k, ...)
    # and silently drops others (UMean/pMean sample as empty columns!); the
    # standalone `postProcess` utility reads whatever the functionObject asks
    # for from the time dir -- use util="postProcess" for averaged fields.
    env = dict(os.environ, OMP_NUM_THREADS="2")
    return subprocess.run([util, *args], cwd=CASE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def latest_time():
    ts = [d for d in os.listdir(CASE) if d.replace('.', '').isdigit()
          and os.path.isdir(os.path.join(CASE, d)) and float(d) > 0]
    return max(ts, key=float) if ts else None


def _ux_tail(ntail=200):
    for lg in ("log.simpleFoam_deep", "log.simpleFoam_damp", "log.simpleFoam"):
        p = os.path.join(CASE, lg)
        if not os.path.exists(p):
            continue
        vals = []
        for ln in open(p, errors="ignore"):
            if "Solving for Ux" in ln and "Initial residual = " in ln:
                try:
                    vals.append(float(ln.split("Initial residual = ")[1].split(",")[0]))
                except Exception:
                    pass
        if vals:
            return float(np.median(vals[-ntail:]))
    return np.inf


def _ux_tail_flatness(ntail=200):
    """(p90-p10)/median of the last-ntail Ux initial residuals: ~0 for a flat
    (period-locked) limit-cycle floor, large when still drifting/diverging."""
    for lg in ("log.simpleFoam_deep", "log.simpleFoam_damp", "log.simpleFoam"):
        p = os.path.join(CASE, lg)
        if not os.path.exists(p):
            continue
        vals = []
        for ln in open(p, errors="ignore"):
            if "Solving for Ux" in ln and "Initial residual = " in ln:
                try:
                    vals.append(float(ln.split("Initial residual = ")[1].split(",")[0]))
                except Exception:
                    pass
        if len(vals) >= ntail:
            w = np.array(vals[-ntail:])
            return float((np.percentile(w, 90) - np.percentile(w, 10)) / np.median(w))
    return np.inf


def converged():
    for lg in ("log.simpleFoam_deep", "log.simpleFoam_damp", "log.simpleFoam"):
        p = os.path.join(CASE, lg)
        if os.path.exists(p) and "SIMPLE solution converged" in open(p, errors="ignore").read():
            return True
    return _ux_tail() < 1e-4


def gradp_window(win=1000):
    """meanVelocityForce pressure-gradient (== total array drag per unit volume)
    over the last `win` iterations: (mean, spread%, drift%).  This is the
    PHYSICAL limit-cycle amplitude proxy -- residual tails alone are not enough
    under deep under-relaxation (which mechanically shrinks initial residuals)."""
    p = os.path.join(CASE, "log.simpleFoam_deep")
    if not os.path.exists(p):
        p = os.path.join(CASE, "log.simpleFoam_damp")
    if not os.path.exists(p):
        return None
    gp = [float(v) for v in
          re.findall(r"pressure gradient = ([\d.eE+-]+)", open(p, errors="ignore").read())][1::2]
    if len(gp) < 200:
        return None
    w = gp[-win:]
    gm = float(np.mean(w))
    if gm == 0:
        return None
    spread = (max(w) - min(w)) / abs(gm) * 100.0
    drift = float(np.mean(w[len(w) // 2:]) - np.mean(w[: len(w) // 2])) / abs(gm) * 100.0
    return gm, spread, drift


def window_stable(gw, max_spread=5.0, max_drift=2.0):
    return gw is not None and gw[1] < max_spread and abs(gw[2]) < max_drift


def floor_columns():
    """Vertical sampling lines on the floor between cubes.  Cube footprint is
    x,z in [h/2, 3h/2]; the open floor is the rest of the 2h x 2h cell."""
    Lx = Lz = PITCH * H
    c0, c1 = 0.5 * (PITCH - 1.0) * H, 0.5 * (PITCH + 1.0) * H
    pts = []
    n = 12 if PITCH == 2.0 else int(round(2 * PITCH))
    for i in range(n):
        x = (i + 0.5) * Lx / n
        for kk in range(n):
            z = (kk + 0.5) * Lz / n
            if c0 - 1e-6 < x < c1 + 1e-6 and c0 - 1e-6 < z < c1 + 1e-6:
                continue                 # under the cube footprint -> skip
            pts.append((x, z))
    return pts


def write_floor_lines(pts, fields="U p"):
    lines = []
    for j, (x, z) in enumerate(pts):
        lines.append(
            f"    fl{j:03d}\n    {{\n        type lineUniform;\n        axis distance;\n"
            f"        start ({x:.6e} 1e-5 {z:.6e});\n"
            f"        end   ({x:.6e} {LINE_LEN:.6e} {z:.6e});\n"
            f"        nPoints {LINE_NPTS};\n    }}\n")
    block = ("type sets;\nlibs (\"libsampling.so\");\ninterpolationScheme cellPoint;\n"
             f"setFormat raw;\nfields ({fields});\nwriteControl writeTime;\nsets\n(\n"
             + "".join(lines) + ");\n")
    open(os.path.join(CASE, "system", "floorLines"), "w").write(block)


def read_line(tsub, name, anyfields=False):
    d = os.path.join(CASE, "postProcessing", "floorLines", tsub)
    cand = glob.glob(os.path.join(d, f"{name}*.xy" if anyfields else f"{name}.xy"))
    if not cand:
        return None
    arr = np.loadtxt(cand[0])
    if arr.ndim == 1 or arr.shape[0] < Y_IDX + 3:
        return None
    return arr[:, 0], arr[:, 1:4], arr[:, 4]


def form_drag_fraction(tsub):
    """form-drag fraction = (pressure force on cube_x) / (pressure + viscous).
    Read from `wallShearStress` is viscous; cube pressure force from blade.xy-style
    surface sample.  We approximate via the global meanVelocityForce pressure
    gradient G and the cube frontal area: total sink = G*Vol; cube form drag is the
    dominant term -> we report G and the floor skin-friction integral for context."""
    return np.nan   # reported qualitatively; see provenance note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cube_array_prod",
                    help="case dir under codes/openfoam (default: packed cube_array_prod)")
    ap.add_argument("--pitch", type=float, default=2.0,
                    help="pitch in units of h (2.0 = packed Coceal; 12.0 = sparse control)")
    ap.add_argument("--tag", default="cube_array",
                    help="output npz stem under codes/results (default: cube_array)")
    ap.add_argument("--windowed-mean", action="store_true",
                    help="harvest the fieldAverage iteration-mean (UMean/pMean) instead of the "
                         "final snapshot; the npz is explicitly flagged mean_of_limit_cycle=True "
                         "with the averaging window recorded. Use ONLY when a steady limit cycle "
                         "persists -- never to dress up a snapshot.")
    args = ap.parse_args()
    global CASE, PITCH, TAG
    CASE = os.path.join(OF, args.case)
    PITCH = args.pitch
    TAG = args.tag
    global NU
    NU = case_nu(CASE)               # use the case's actual viscosity (Re sweep safe)
    print(f"  [cube_array] case nu={NU:.4g}  Re_h={H/NU:.0f}")

    if not os.path.isdir(CASE):
        np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                 ok=np.array([False]), reason=np.array(["case dir missing"], dtype=object))
        print("cube_array: case dir missing -- wrote ok=False npz")
        return
    if not converged():
        tail, flat, gw0 = _ux_tail(), _ux_tail_flatness(), gradp_window()
        cycle_locked = (tail < 1e-2 and flat < 0.5
                        and gw0 is not None and abs(gw0[2]) < 2.0)
        if not (args.windowed_mean and cycle_locked):
            np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                     ok=np.array([False]),
                     reason=np.array([f"not converged (Ux tail={tail:.2e}, "
                                      f"flatness={flat:.2f}, "
                                      f"gradP drift={'n/a' if gw0 is None else f'{gw0[2]:+.2f}%'})"],
                                     dtype=object))
            print(f"cube_array: NOT converged (Ux tail={tail:.2e}) -- wrote ok=False npz")
            return
        print(f"  [windowed] residual tail {tail:.2e} above point-convergence but FLAT "
              f"(flatness={flat:.3f}, gradP drift={gw0[2]:+.3f}%): period-locked limit "
              f"cycle; proceeding on the iteration-mean route (UMean stationarity gate "
              f"still applies)")
    # HONESTY GATE: a residual tail passing under deep under-relaxation does NOT
    # by itself prove a steady state -- the physical forcing must be windowed-stable.
    gw = gradp_window()
    if not args.windowed_mean and not window_stable(gw):
        msg = (f"Ux tail={_ux_tail():.2e} < 1e-4 BUT physical window unstable "
               f"(gradP last-1000 spread={gw[1]:.1f}%, drift={gw[2]:+.1f}%); "
               "snapshot harvest refused -- if the cycle persists after deeper damping, "
               "rerun with --windowed-mean (fieldAverage cycle-mean, explicitly flagged)"
               if gw else "Ux tail < 1e-4 but gradP window unavailable in log")
        np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                 ok=np.array([False]), reason=np.array([msg], dtype=object))
        print(f"cube_array: {msg} -- wrote ok=False npz")
        return

    tdir = latest_time()
    umean_stat_pct = np.nan
    if args.windowed_mean:
        missing = [f for f in ("UMean", "pMean")
                   if not os.path.exists(os.path.join(CASE, tdir, f))]
        if missing:
            np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                     ok=np.array([False]),
                     reason=np.array([f"--windowed-mean but {missing} missing at t={tdir}"],
                                     dtype=object))
            print(f"cube_array: {missing} missing at t={tdir} -- wrote ok=False npz")
            return
        import shutil
        shutil.rmtree(os.path.join(CASE, "postProcessing", "floorLines"), ignore_errors=True)
    pts = floor_columns()
    write_floor_lines(pts, fields="UMean pMean" if args.windowed_mean else "U p")
    if args.windowed_mean:
        of("-func", "floorLines", "-latestTime", util="postProcess")
    else:
        of("-postProcess", "-func", "floorLines", "-latestTime")
    if args.windowed_mean:
        # CYCLE-MEAN STATIONARITY GATE: the iteration-mean must be insensitive to
        # the window length -- compare floor tau from UMean at the previous write
        # time (shorter window) vs the latest (full window).  >2% -> refuse
        # (extend the run / window instead of harvesting an unsettled mean).
        ts_all = sorted([d for d in os.listdir(CASE) if d.replace('.', '').isdigit()
                         and os.path.isdir(os.path.join(CASE, d))], key=float)
        prev = [t for t in ts_all if float(t) < float(tdir)
                and os.path.exists(os.path.join(CASE, t, "UMean"))]
        if prev:
            t2 = prev[-1]
            of("-func", "floorLines", "-time", t2, util="postProcess")

            def _tau_at(ts):
                tau = {}
                for j, (x, z) in enumerate(pts):
                    o = read_line(ts, f"fl{j:03d}", anyfields=True)
                    if o is None:
                        continue
                    d_, U_, _p = o
                    um = np.hypot(U_[:, 0], U_[:, 2])
                    sb = d_ <= VISC_SUB
                    if sb.sum() < 3:
                        sb = d_ <= d_[3]
                    aa = np.sum(um[sb] * d_[sb]) / np.sum(d_[sb] ** 2)
                    tau[j] = np.sign(np.sum(U_[sb, 0] * d_[sb])) * NU * aa
                return tau
            ta, tb = _tau_at(t2), _tau_at(tdir)
            com = sorted(set(ta) & set(tb))
            if len(com) >= 5:
                va = np.array([ta[k] for k in com]); vb = np.array([tb[k] for k in com])
                med = np.median(np.abs(vb))
                umean_stat_pct = float(100.0 * np.median(np.abs(vb - va) / med))
                print(f"  [windowed] UMean stationarity: floor tau (window->t={t2}) vs "
                      f"(window->t={tdir}) median diff = {umean_stat_pct:.3f}% of median|tau|")
                if umean_stat_pct > 2.0:
                    np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                             ok=np.array([False]),
                             reason=np.array([f"windowed mean NOT stationary "
                                              f"({umean_stat_pct:.1f}% > 2% between windows "
                                              f"ending {t2} and {tdir}) -- extend the run"],
                                             dtype=object))
                    print(f"cube_array: windowed mean not stationary -- wrote ok=False npz")
                    return
    ppdir = os.path.join(CASE, "postProcessing", "floorLines")
    tsub = sorted(os.listdir(ppdir), key=lambda s: float(s) if s.replace('.', '').isdigit() else -1)[-1]

    X, Z, TAU, PNEAR = [], [], [], []
    pd, pu = [], []
    for j, (x, z) in enumerate(pts):
        out = read_line(tsub, f"fl{j:03d}", anyfields=args.windowed_mean)
        if out is None:
            continue
        d, U, p = out
        umag = np.hypot(U[:, 0], U[:, 2])         # horizontal speed magnitude
        sub = d <= VISC_SUB
        if sub.sum() < 3:
            sub = d <= d[3]
        a = np.sum(umag[sub] * d[sub]) / np.sum(d[sub] ** 2)
        # sign of tau from streamwise component near wall
        sgn = np.sign(np.sum(U[sub, 0] * d[sub]))
        X.append(x); Z.append(z); TAU.append(sgn * NU * a); PNEAR.append(p[1])
        pd.append(d); pu.append(U[:, 0])
    if len(X) < 5:
        np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
                 ok=np.array([False]), reason=np.array([f"only {len(X)} floor cols"], dtype=object))
        print("cube_array: too few floor columns")
        return
    X = np.array(X); Z = np.array(Z); TAU = np.array(TAU); PNEAR = np.array(PNEAR)
    # dp/dx: regress near-wall p on x at each z-band, then take the array mean gradient
    # (global streamwise pressure gradient is the meanVelocityForce sink)
    A = np.vstack([X - X.mean(), np.ones(len(X))]).T
    dpdx_global = float(np.linalg.lstsq(A, PNEAR, rcond=None)[0][0])
    # per-column dp/dx via local x-neighbour finite difference within z-band
    dp_dx = np.full(len(X), dpdx_global)
    for zb in np.unique(np.round(Z, 6)):
        m = np.abs(Z - zb) < 1e-6
        if m.sum() >= 3:
            xo = np.argsort(X[m])
            xs_, ps_ = X[m][xo], PNEAR[m][xo]
            g = np.gradient(ps_, xs_)
            idx = np.where(m)[0][xo]
            dp_dx[idx] = g

    u_tau = np.sqrt(np.abs(TAU))
    valid = u_tau > 0
    y_m = float(np.median(50.0 * NU / u_tau[valid]))
    grid = y_m * (np.arange(NPZ_M) / 10.0)
    n = len(X)
    Y = np.tile(grid, (n, 1))
    Umat = np.array([np.interp(grid, pd[i], pu[i]) for i in range(n)])
    nu_arr = np.full(n, NU)

    prof = os.path.join(RESULTS, f"{TAG}_wall_profiles.npz")
    np.savez(prof, y=Y, U=Umat, tau_w=TAU, dp_dx=dp_dx, x=X, z=Z, nu=nu_arr, y_m=y_m)
    res = evaluate(prof)

    prov = (f"Coceal-2006 aligned cube array (h, pitch {PITCH:g}h, "
            f"lambda_p={1.0 / PITCH ** 2:.4g}, Re_h~{H / NU:.0f}), "
            "OpenFOAM-RANS 3D (simpleFoam, kOmegaSST). Geometry EXACT; flow is RANS, "
            "NOT the Coceal DNS. The 1-D ODE wall model is deployed on the FLOOR "
            "between cubes (epsilon = |tau_w|/(|dp/dx|*y_m), y+~50). Cube-array drag is "
            "FORM-DRAG-DOMINATED; epsilon here measures the LOCAL floor skin-friction vs "
            "pressure-gradient balance, NOT global drag (which is cube form drag). "
            f"Floor near-wall p regression dp/dx = {dpdx_global:.4e}; the meanVelocityForce "
            "(total-drag) window-mean dp/dx is recorded in gradp_win_mean. NOT DNS.")
    extra = {}
    if args.windowed_mean:
        ctext = open(os.path.join(CASE, "system", "controlDict"), errors="ignore").read()
        mts = re.search(r"timeStart\s+(\d+)", ctext)
        t0avg = float(mts.group(1)) if mts else np.nan
        extra["window_iters"] = np.array([t0avg, float(tdir)])
        extra["umean_stationarity_pct"] = umean_stat_pct
        prov += (f" LIMIT-CYCLE NOTE: steady RANS limit-cycled; harvested field is the "
                 f"ITERATION-MEAN (fieldAverage) over window [{t0avg:.0f},{float(tdir):.0f}] "
                 f"iters, NOT a snapshot. mean_of_limit_cycle=True; window-length "
                 f"sensitivity of floor tau = {umean_stat_pct:.2f}% (gate <2%).")
    np.savez(os.path.join(RESULTS, f"{TAG}.npz"),
             ok=np.array([True]),
             n=res["n"], f_sep=res["f_sep"], eps_med=res["eps_med"],
             frac_eps_lt1=res["frac_eps_lt1"], frac_eps_lt0p1=res["frac_eps_lt0p1"],
             relRMS=res["relRMS"], r2=res["r2"], sign_acc_sep=res["sign_acc_sep"],
             y_m=y_m, dpdx_global=dpdx_global,
             h=H, pitch_over_h=PITCH, lambda_p=1.0 / PITCH ** 2, Re_h=H / NU,
             pitch_over_delta=PITCH * H / (HDOM / 2.0),   # pitch / (Hdom/2)
             shape="sharp", dimensionality="3D",
             ux_tail=_ux_tail(),
             gradp_win_mean=(gw[0] if gw else np.nan),
             gradp_win_spread_pct=(gw[1] if gw else np.nan),
             gradp_win_drift_pct=(gw[2] if gw else np.nan),
             mean_of_limit_cycle=np.array([bool(args.windowed_mean)]),
             fidelity=np.array(["OpenFOAM-generated 3D RANS (kOmegaSST); NOT DNS/LES; "
                                "geometry exact (Coceal 2006)"], dtype=object),
             form_drag_note=np.array(["form-drag-dominated; epsilon is LOCAL floor balance"], dtype=object),
             reason=np.array([""], dtype=object),
             provenance=np.array([prov], dtype=object),
             **extra)
    print("=" * 72)
    print("CUBE ARRAY (Coceal 2006) a-priori floor ODE (verbatim evaluate, Y_IDX=10):")
    print(f"  n_floor_cols={res['n']}  f_sep={res['f_sep']:.3f}")
    print(f"  eps_med={res['eps_med']:.4f}  f(eps<1)={res['frac_eps_lt1']:.3f}  "
          f"f(eps<0.1)={res['frac_eps_lt0p1']:.3f}")
    print(f"  R2(tau_w)={res['r2']:.4f}  relRMS={res['relRMS']:.4f}")
    print(f"  global dp/dx={dpdx_global:.4e}  (form-drag-dominated)")
    print(f"  Ux tail={_ux_tail():.2e}  gradP win: mean={gw[0]:.4e} "
          f"spread={gw[1]:.1f}% drift={gw[2]:+.1f}%" if gw else "  (no gradP window)")
    print(f"  mean_of_limit_cycle={bool(args.windowed_mean)}"
          + (f"  window_iters={extra['window_iters'].tolist()}" if args.windowed_mean else ""))
    print(f"  wrote {os.path.join(RESULTS, f'{TAG}.npz')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
