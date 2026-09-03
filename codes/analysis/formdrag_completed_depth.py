#!/usr/bin/env python3
r"""
formdrag_completed_depth.py  --  L1 (Core methodology) for the iteration
"Form-Drag-Completed Universal Collapse".

PRE-REGISTERED HYPOTHESIS (research_direction.md, node_000):
    The champion paper's dimensionless collapse "breaks at sharp edges": the
    skin-friction cancellation depth  eps = |tau_w| / (|dp/dx| * y_m)  under-counts
    the residual on bluff/sharp repeating elements, where the wall removes
    streamwise momentum through TWO channels -- skin friction AND form drag.
    Candidate repair: a form-drag-completed depth
        eps_star = eps * (1 - phi_FD),     phi_FD = |F_p,x| / (|F_p,x| + |F_v,x|)
    where phi_FD is the streamwise FORM-DRAG FRACTION of the wall force (the
    pressure-force share the 1-D wall-normal ODE is structurally blind to).
    H1/H3:  eps_star < 1  <=>  R2(tau_w) < 0 (catastrophic), 0 misclassifications,
    reducing to the validated eps on smooth walls (phi_FD -> 0).

THIS SCRIPT LOCKS THE METHOD AND PUTS H1/H3 TO ITS PRE-REGISTERED FALSIFIER.
It is a-priori, computes only, fabricates nothing.  Every phi_FD comes from a real
OpenFOAM surface-force integral (the `forces` function object: pressure part =
form drag, viscous part = skin friction); every eps / R2 comes from the locked
production ODE via the SAME `evaluate` used for every hill number in the paper.

  output:  codes/results/formdrag_partition.npz
  guards:  hill R2 = -47.68617, rib_les_dtype R2 = -0.94317 (bit-exact, B-L0-5)

Body-fitted vs Cartesian (B-L0-1).  phi_FD is defined relative to the ODE's
operating frame.  The hill/conv-div DNS are in body-fitted (wall-following)
coordinates: y[:,0] = 0 at every station, tau_w is the tangential wall stress.
In that frame a SMOOTH wall presents no surface-normal face, so the entire
ODE-representable wall force IS tau_w and phi_FD = 0 exactly -- NOT the ~0.6-0.8
"form drag" a Cartesian projection of the same smooth hill would report.  phi_FD
is therefore the fraction of the wall force carried by surfaces the ODE cannot
represent (bluff/normal faces of sharp repeating elements).
"""
import os
import re
import sys
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RESULTS = os.path.join(ROOT, "results")
OF = os.path.join(ROOT, "openfoam")

sys.path.insert(0, HERE)
# locked production ODE evaluator -- IMPORTED, never re-implemented (anti-circularity)
from cross_geometry_collapse import evaluate, Y_IDX            # noqa: E402

# ---------------------------------------------------------------------------
# 0.  REGRESSION GUARDS (B-L0-5) -- bit-exact before any new number is trusted
# ---------------------------------------------------------------------------
GUARDS = {
    "periodic_hills_1p0": (
        os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
        -47.68617),
    "rib_les_dtype": (
        os.path.join(RESULTS, "rib_les_dtype_wall_profiles.npz"),
        -0.94317),
}


def check_guards():
    out = {}
    for key, (path, anchor) in GUARDS.items():
        r2 = float(evaluate(path)["r2"])
        drift = abs(r2 - anchor)
        ok = drift < 1e-4
        out[key] = (r2, anchor, drift, ok)
        flag = "OK" if ok else "*** DRIFT ***"
        print(f"  guard {key:22s} R2={r2:+.5f} (anchor {anchor:+.5f}, drift {drift:.2e}) {flag}")
        if not ok:
            raise SystemExit(f"REGRESSION GUARD FAILED for {key}: {r2} vs {anchor}")
    return out


# ---------------------------------------------------------------------------
# 1.  Form-drag fraction phi_FD from a real OpenFOAM `forces` integral
# ---------------------------------------------------------------------------
def parse_forces_dat(path):
    """Return (F_pressure_x, F_viscous_x) from the LAST row of an OpenFOAM
    forces.dat. Layout: time ((px py pz)(vx vy vz)) ((mom...)).  px=f[0], vx=f[3]."""
    last = None
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            last = line
    if last is None:
        raise ValueError(f"no data rows in {path}")
    toks = last.split(None, 1)                 # split off the time column
    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", toks[1])
    f = [float(x) for x in floats]
    return f[0], f[3]                          # pressure_x, viscous_x


def phi_FD_from_forces(path):
    fp_x, fv_x = parse_forces_dat(path)
    phi = abs(fp_x) / (abs(fp_x) + abs(fv_x))
    return phi, fp_x, fv_x


# ---------------------------------------------------------------------------
# 2.  The swept family: (eps, R2) from the locked instrument; phi_FD from fields
# ---------------------------------------------------------------------------
def latest_forces_dat(case_dir, func="forcesFD"):
    base = os.path.join(OF, case_dir, "postProcessing", func)
    times = [t for t in os.listdir(base) if t[0].isdigit()]
    t = max(times, key=lambda s: float(s))
    return os.path.join(base, t, "forces.dat")


def build_family():
    """Assemble each case: eps, R2 (locked evaluate / on-disk npz), phi_FD (fields).
    fidelity + provenance labelled; nothing fabricated."""
    rows = []

    # --- smooth body-fitted DNS/LES: phi_FD = 0 by construction (B-L0-1) ---
    for key, prof, eps, r2, fam in [
        ("periodic_hills_1p0",
         os.path.join(RESULTS, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
         None, None, "smooth_hill_DNS"),
    ]:
        rr = evaluate(prof)
        rows.append(dict(key=key, family=fam, fidelity="DNS/LES",
                         eps=float(rr["eps_med"]), r2=float(rr["r2"]),
                         phi_FD=0.0, fp_x=0.0, fv_x=float("nan"),
                         phi_source="body-fitted smooth wall (y_wall=0): no ODE-visible normal face",
                         shape="smooth"))

    # smooth controls read from the locked cross-geometry table (phi_FD = 0)
    tm = np.load(os.path.join(RESULTS, "transition_map_l2.npz"), allow_pickle=True)
    tkeys = list(tm["keys"])
    for key, fam in [("krank_pehill_Re10595", "smooth_hill_DNS"),
                     ("conv_div_channel", "conv_div_DNS")]:
        i = tkeys.index(key)
        rows.append(dict(key=key, family=fam, fidelity="DNS/LES",
                         eps=float(tm["eps"][i]), r2=float(tm["r2"][i]),
                         phi_FD=0.0, fp_x=0.0, fv_x=float("nan"),
                         phi_source="body-fitted smooth wall: phi_FD=0",
                         shape="smooth"))

    # --- sharp ribs: eps/R2 from locked evaluate; phi_FD from real OF forces ---
    # d-type rib LES uses the solver-live forcesRib output (the validated anchor).
    les_dat = os.path.join(OF, "rib_les_dtype", "postProcessing",
                           "forcesRib", "139.99984127", "forces.dat")
    sharp_specs = [
        ("rib_les_dtype", "rib_dtype_LES", "LES",
         os.path.join(RESULTS, "rib_les_dtype_wall_profiles.npz"), les_dat),
        ("rib_rans_dtype", "rib_dtype_RANS", "RANS",
         os.path.join(RESULTS, "rib_rans_dtype_wall_profiles.npz"),
         latest_forces_dat("rib_rans_dtype")),
        ("rib_rans_ktype", "rib_ktype_RANS", "RANS",
         os.path.join(RESULTS, "rib_rans_ktype_wall_profiles.npz"),
         latest_forces_dat("rib_rans_ktype")),
    ]
    for key, fam, fid, prof, dat in sharp_specs:
        rr = evaluate(prof)
        phi, fp, fv = phi_FD_from_forces(dat)
        rows.append(dict(key=key, family=fam, fidelity=fid,
                         eps=float(rr["eps_med"]), r2=float(rr["r2"]),
                         phi_FD=float(phi), fp_x=float(fp), fv_x=float(fv),
                         phi_source=os.path.relpath(dat, ROOT), shape="sharp"))

    # sharpness ladder rungs: eps/R2 from sharpness_ladder.npz; phi_FD from fields
    sl = np.load(os.path.join(RESULTS, "sharpness_ladder.npz"), allow_pickle=True)
    ladder_dirs = {0.0: "sharp_ladder_rk00", 0.25: "sharp_ladder_rk025",
                   0.5: "sharp_ladder_rk05"}
    for rk, cdir in ladder_dirs.items():
        j = int(np.argmin(np.abs(sl["rk"] - rk)))
        phi, fp, fv = phi_FD_from_forces(latest_forces_dat(cdir))
        rows.append(dict(key=f"ladder_rk{rk:g}", family="sharpness_ladder_RANS",
                         fidelity="RANS",
                         eps=float(sl["eps_med"][j]), r2=float(sl["r2"][j]),
                         phi_FD=float(phi), fp_x=float(fp), fv_x=float(fv),
                         phi_source=os.path.relpath(latest_forces_dat(cdir), ROOT),
                         shape="sharp"))

    # SPLEEN blade: eps/R2 from spleen_cascade_incompressible.npz; phi_FD from fields
    sp = np.load(os.path.join(RESULTS, "spleen_cascade_incompressible.npz"),
                 allow_pickle=True)
    phi, fp, fv = phi_FD_from_forces(latest_forces_dat("spleen_cascade_incomp_fine"))
    rows.append(dict(key="spleen_blade", family="blade_cascade_RANS",
                     fidelity="RANS",
                     eps=float(sp["eps_med"]), r2=float(sp["r2"]),
                     phi_FD=float(phi), fp_x=float(fp), fv_x=float(fv),
                     phi_source=os.path.relpath(
                         latest_forces_dat("spleen_cascade_incomp_fine"), ROOT),
                     shape="sharp"))
    return rows


# ---------------------------------------------------------------------------
# 3.  Spearman rho (no SciPy) -- tie-aware rank correlation + two-sided t-p
# ---------------------------------------------------------------------------
def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    # average ties
    out = r.copy()
    i = 0
    s = a[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            out[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rank(x), _rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n > 2 and abs(rho) < 1.0:
        t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    else:
        t = float("inf") * np.sign(rho)
    return rho, float(t), n


# ---------------------------------------------------------------------------
# 4.  Run: lock method, fire the falsifier, write npz
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("L1  form-drag-completed cancellation depth  eps* = eps*(1 - phi_FD)")
    print("    (a-priori; phi_FD from real OpenFOAM surface-force integrals)")
    print("=" * 74)
    print("\n[0] regression guards (B-L0-5):")
    guards = check_guards()

    print("\n[1] family (eps from locked ODE; phi_FD from fields):")
    rows = build_family()
    keys = [r["key"] for r in rows]
    eps = np.array([r["eps"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    phi = np.array([r["phi_FD"] for r in rows])
    eps_star = eps * (1.0 - phi)
    actual_fail = (r2 < 0.0).astype(int)              # catastrophic = R2 < 0
    pred_fail = (eps_star < 1.0).astype(int)          # H1 predictor
    eps_fail = (eps < 1.0).astype(int)                # the champion's plain-eps predictor
    misclass = (pred_fail != actual_fail).astype(int)

    hdr = (f"  {'case':18s} {'fid':5s} {'eps':>7s} {'phi_FD':>7s} "
           f"{'eps*':>7s} {'R2':>9s} {'actual':>7s} {'pred':>5s}")
    print(hdr)
    for i, r in enumerate(rows):
        a = "FAIL" if actual_fail[i] else "pass"
        p = "FAIL" if pred_fail[i] else "pass"
        m = "  <-- MISCLASS" if misclass[i] else ""
        print(f"  {r['key']:18s} {r['fidelity']:5s} {eps[i]:7.3f} {phi[i]:7.3f} "
              f"{eps_star[i]:7.3f} {r2[i]:+9.2f} {a:>7s} {p:>5s}{m}")

    n_mis = int(misclass.sum())
    print(f"\n[2] H1/H3 pre-registered falsifier:  eps* < 1  <=>  R2 < 0")
    print(f"    misclassifications by eps*  : {n_mis} / {len(rows)}")
    print(f"    misclassifications by plain eps: {int((eps_fail != actual_fail).sum())} / {len(rows)}")

    # negative-control statistics: is phi_FD related to the ODE outcome at all?
    rho_r2, t_r2, n = spearman(phi, r2)
    rho_fl, t_fl, _ = spearman(phi, actual_fail)
    # restrict to sharp cases (where phi_FD actually varies)
    sharp = np.array([r["shape"] == "sharp" for r in rows])
    rho_r2_sharp, _, n_sharp = spearman(phi[sharp], r2[sharp])
    print(f"\n[3] negative control -- phi_FD vs ODE outcome:")
    print(f"    Spearman(phi_FD, R2)        rho={rho_r2:+.3f}  (n={n})")
    print(f"    Spearman(phi_FD, fail)      rho={rho_fl:+.3f}  (n={n})")
    print(f"    Spearman(phi_FD, R2) sharp  rho={rho_r2_sharp:+.3f}  (n={n_sharp})")

    verdict = ("FALSIFIED" if n_mis > 0 else "SURVIVES")
    print(f"\n[4] VERDICT on eps* = eps*(1-phi_FD):  {verdict}")
    if n_mis > 0:
        bad = [keys[i] for i in range(len(rows)) if misclass[i]]
        print(f"    form-drag correction misclassifies: {bad}")
        print(f"    => the drag partition is NOT the order parameter; the ODE")
        print(f"       failure is form-drag/roughness-INDEPENDENT (negative control).")

    out = os.path.join(RESULTS, "formdrag_partition.npz")
    np.savez(
        out,
        keys=np.array(keys),
        family=np.array([r["family"] for r in rows]),
        fidelity=np.array([r["fidelity"] for r in rows]),
        shape=np.array([r["shape"] for r in rows]),
        phi_source=np.array([r["phi_source"] for r in rows]),
        eps=eps, r2=r2, phi_FD=phi, eps_star=eps_star,
        F_pressure_x=np.array([r["fp_x"] for r in rows]),
        F_viscous_x=np.array([r["fv_x"] for r in rows]),
        actual_fail=actual_fail, pred_fail_epsstar=pred_fail, pred_fail_eps=eps_fail,
        misclass_epsstar=misclass,
        n_misclass_epsstar=n_mis,
        n_misclass_eps=int((eps_fail != actual_fail).sum()),
        spearman_phi_r2=rho_r2, spearman_phi_r2_t=t_r2,
        spearman_phi_fail=rho_fl,
        spearman_phi_r2_sharp=rho_r2_sharp, n_sharp=n_sharp,
        guard_hill_r2=guards["periodic_hills_1p0"][0],
        guard_rib_r2=guards["rib_les_dtype"][0],
        verdict=verdict,
        protocol_y_idx=Y_IDX,
        note=("L1: pre-registered test of form-drag-completed eps*=eps*(1-phi_FD). "
              "phi_FD = streamwise form-drag fraction from real OpenFOAM `forces` "
              "integrals (pressure part = form drag, viscous part = skin friction); "
              "smooth body-fitted walls phi_FD=0 by construction. eps/R2 from the "
              "locked production ODE (evaluate, Y_IDX=10). a-priori; no fabrication."),
    )
    md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    print(f"\n[5] wrote {os.path.relpath(out, ROOT)}  md5={md5}")
    return out


if __name__ == "__main__":
    main()
