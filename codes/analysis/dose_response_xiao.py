"""
dose_response_xiao.py
=====================
Turn the cancellation diagnostic from a binary *label* into a predictive
*dose-response law* across a continuously parameterised family of repeating
geometries: the Xiao et al. (2020) 29-case parameterised periodic-hill DNS
database (Re_b = 5600), spanning hill steepness alpha in {0.5,0.75,1.0,1.25,1.5}
crossed with streamwise period L_x and channel height L_y.

For each case we run the *same* validated a-priori ODE/TBLE wall model used
throughout the manuscript and measure:
  - R^2(tau_w)                      : ODE wall-stress skill,
  - median cancellation parameter   : eps_tilde = median |tau_w| / (|dp/dx| y_m),
  - f(eps<0.1)                      : domain fraction in the residual regime,
  - f_sep                           : reversed-shear fraction,
  - f_rec = L_sep / ell_p           : recirculation coverage of one period,
against candidate *geometric* repeat parameters that an engineer can read off
the CAD geometry + a precursor estimate of delta:
  - alpha            (steepness),
  - h / L_x          (hill height over streamwise period; inverse pitch),
  - ell_p / delta    (repeat pitch over outer scale, delta ~ L_y/2),
  - L_sep / delta    (recirculation length over outer scale).

The Level-0 review required (concern #3) that we DO NOT pre-commit to a single
x-axis: we report the Spearman rank correlation of each candidate against the
failure measures and let the data say which (if any) collapses the family.

HONESTY (G1): every data point is real periodic-hill DNS. The law motivates the
broad repeating-structure class via the *parameter*; it never claims blades /
cities / exchangers were tested. (G2): every number is written to
codes/results/dose_response_xiao.npz by this committed script; per-case wall
profiles are written to codes/results/xiao29/. (G4): a-priori protocol
(reference profile + dp/dx in, not coupled WMLES) -- identical Y_IDX matching
height as the rest of the paper.

DATA COLUMN ORDER (verified empirically, 2026-05-29): the 29-case mean_files.dat
is  x  y  u  v  w  p  (NOT x y p u v w as the legacy 5-case reader assumed).
At mid-channel col2 ~ 0.87 (= U), col3 ~ 0 (= V), col5 = pressure. The hill is a
solid block of identically-zero field from y=0 up to the hill surface, so the
fluid wall is located per-station as the first point with non-zero velocity.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/dose_response_xiao.py            # full 29-case family
      OMP_NUM_THREADS=2 python3 codes/analysis/dose_response_xiao.py --subset   # feasibility (steep + shallow)
"""

import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RESULTS = os.path.join(ROOT, "results")
XIAO = os.path.join(ROOT, "new_data_download", "geometry_driven",
                    "xiao_pehill_parameterized", "pehill-29-cases-DNS")
OUTDIR = os.path.join(RESULTS, "xiao29")

# The validated ODE/TBLE wall model used throughout the manuscript.
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w               # noqa: E402

RE_B = 5600.0
NU = 1.0 / RE_B
Y_IDX = 10          # matching-height index, identical to the a-priori protocol
VEL_TOL = 1e-6      # |velocity| threshold separating solid hill from fluid

# steepness from the directory-name prefix alphNN (NN = alpha*10, zero-padded)
ALPHA_FROM_PREFIX = {"05": 0.5, "075": 0.75, "10": 1.0, "125": 1.25, "15": 1.5}


def parse_alpha(case_name):
    """alpha from the 'alphNN-...' directory prefix."""
    prefix = case_name.split("-")[0].replace("alph", "")
    return ALPHA_FROM_PREFIX.get(prefix, np.nan)


def largest_contiguous_span(x, mask):
    """Streamwise extent of the largest contiguous reversed-shear run
    (= per-element recirculation length L_sep), in the units of x."""
    if not mask.any():
        return 0.0, (np.nan, np.nan)
    best_len, best_span = 0.0, (np.nan, np.nan)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            span = x[j] - x[i]
            if span > best_len:
                best_len, best_span = span, (x[i], x[j])
            i = j + 1
        else:
            i += 1
    return best_len, best_span


def read_case(case_dir):
    """Read a single Xiao 29-case mean_files.dat into per-station wall profiles.

    Returns a dict with, per x-station: distance-from-wall profiles (y, U, V, P),
    the wall-shear stress tau_w (nu du/dy at the hill surface), the wall pressure
    gradient dp/dx (central difference of near-wall p), plus the geometric scales
    ell_p (streamwise period), L_y (channel height) and h (= 1, hill height).
    """
    mean_file = os.path.join(case_dir, "mean_files.dat")
    d = np.loadtxt(mean_file)
    x, y, u, v, p = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 5]

    xu = np.unique(np.round(x, 6))
    ell_p = float(x.max() - x.min())     # streamwise repeat pitch (period)
    L_y = float(y.max() - y.min())       # channel height (the L_y knob)

    x_st, tau_w, p_wall = [], [], []
    y_prof, U_prof, V_prof, P_prof = [], [], [], []
    for xv in xu:
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp = y[m], u[m], v[m], p[m]
        o = np.argsort(yy)
        yy, uu, vv, pp = yy[o], uu[o], vv[o], pp[o]
        # locate the fluid wall: first point above the solid hill block where the
        # velocity is non-zero (the hill interior is stored as identically zero).
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < Y_IDX + 2:
            continue
        k = max(fluid[0], 1)             # first fluid point above the wall
        yw = yy[k - 1]                   # wall (last solid / no-slip node)
        ywall = yy[k - 1:] - yw          # distance from wall (fluid column)
        ufl = uu[k - 1:]
        vfl = vv[k - 1:]
        pfl = pp[k - 1:]
        # tau_w = nu du/dy at the wall, from a least-squares fit of the first
        # few fluid points through the no-slip origin (u = (du/dy) y). This is
        # more robust than a single two-point difference and recovers the sign
        # of the wall shear (reversed shear -> separated).
        nfit = min(4, len(ywall) - 1)
        yf, uf = ywall[1:1 + nfit], ufl[1:1 + nfit]
        dudy = float(np.sum(yf * uf) / np.sum(yf * yf)) if np.sum(yf * yf) > 0 else 0.0
        tw = NU * dudy
        x_st.append(xv)
        tau_w.append(tw)
        p_wall.append(pfl[1])            # near-wall pressure for dp/dx
        y_prof.append(ywall)
        U_prof.append(ufl)
        V_prof.append(vfl)
        P_prof.append(pfl)

    x_st = np.asarray(x_st)
    tau_w = np.asarray(tau_w)
    p_wall = np.asarray(p_wall)

    # dp/dx: central difference of the near-wall pressure along the stations.
    dp_dx = np.gradient(p_wall, x_st)

    return dict(x=x_st, tau_w=tau_w, dp_dx=dp_dx, y=y_prof, U=U_prof, V=V_prof,
                P=P_prof, ell_p=ell_p, L_y=L_y, h=1.0)


def evaluate_case(case):
    """Run the a-priori ODE wall model on every station; return failure measures
    and geometric repeat parameters."""
    x = case["x"]
    tau_true = case["tau_w"]
    dp_dx = case["dp_dx"]
    n = len(x)

    tau_pred = np.full(n, np.nan)
    eps = np.full(n, np.nan)
    for i in range(n):
        yi = case["y"][i]
        Ui = case["U"][i]
        if Y_IDX >= len(yi):
            continue
        y_m, U_m = yi[Y_IDX], Ui[Y_IDX]
        if y_m <= 0 or np.isnan(U_m):
            continue
        tau_pred[i] = predict_tau_w(U_m, y_m, dp_dx[i], NU)
        denom = abs(dp_dx[i]) * y_m
        if denom > 1e-30:
            eps[i] = abs(tau_true[i]) / denom

    valid = ~np.isnan(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = np.sum((tw_t - tw_p) ** 2)
    ss_tot = np.sum((tw_t - tw_t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # scale-free relative L2 error of the wall stress
    rel_err = np.sqrt(ss_res) / (np.sqrt(np.sum(tw_t ** 2)) + 1e-30)

    sep = tau_true < 0
    L_sep, span = largest_contiguous_span(x, sep)
    ell_p, L_y, h = case["ell_p"], case["L_y"], case["h"]
    delta = 0.5 * L_y                       # outer-scale proxy (channel half-height)
    eps_valid = eps[~np.isnan(eps)]

    return dict(
        r2=float(r2),
        rel_err=float(rel_err),
        eps_median=float(np.nanmedian(eps)),
        frac_eps_lt_0p1=float(np.mean(eps_valid < 0.1)) if eps_valid.size else np.nan,
        f_sep=float(sep.mean()),
        L_sep=float(L_sep),
        f_rec=float(L_sep / ell_p) if ell_p > 0 else np.nan,
        ell_p=float(ell_p),
        L_y=float(L_y),
        delta=float(delta),
        h=float(h),
        # candidate collapse variables (geometric, a-priori-readable)
        cv_alpha=np.nan,                    # filled by caller (from name)
        cv_h_over_Lx=float(h / ell_p),
        cv_ellp_over_delta=float(ell_p / delta),
        cv_Lsep_over_delta=float(L_sep / delta),
        n_stations=int(n),
        n_sep=int(sep.sum()),
    )


def spearman(a, b):
    """Spearman rank correlation, NaN-safe, no SciPy dependency."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_bootstrap_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the Spearman rho of (a, b).

    Deterministic (fixed seed) so the reported confidence interval is itself
    reproducible. Returns (rho_point, lo, hi). The CI quantifies whether a
    candidate is a statistically-resolved dominant control (CI excludes 0)
    rather than resting on a single point estimate.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
    a, b = a[m], b[m]
    n = a.size
    if n < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[k] = spearman(a[idx], b[idx])
    boot = boot[~np.isnan(boot)]
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return spearman(a, b), lo, hi


def main():
    subset = "--subset" in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    cases = sorted(d for d in os.listdir(XIAO)
                   if os.path.isdir(os.path.join(XIAO, d)) and d.startswith("alph"))
    if subset:
        # feasibility: one shallow + one steep hill at the same channel height
        cases = ["alph05-7071-3036", "alph15-7929-3036"]

    rows = []
    print(f"Dose-response across {len(cases)} Xiao periodic-hill cases "
          f"({'feasibility subset' if subset else 'full family'}), Re_b={RE_B:.0f}")
    print("-" * 92)
    print(f"{'case':24s} {'alpha':>5s} {'ell_p':>6s} {'L_y':>5s} "
          f"{'f_sep':>5s} {'f_rec':>5s} {'eps~':>8s} {'R2':>10s}")
    for name in cases:
        case = read_case(os.path.join(XIAO, name))
        ev = evaluate_case(case)
        ev["case"] = name
        ev["cv_alpha"] = parse_alpha(name)
        rows.append(ev)
        # persist per-case wall profiles (writable results dir, never the symlink)
        np.savez(os.path.join(OUTDIR, f"{name}_wall_profiles.npz"),
                 x=case["x"], tau_w=case["tau_w"], dp_dx=case["dp_dx"],
                 ell_p=case["ell_p"], L_y=case["L_y"], alpha=ev["cv_alpha"])
        print(f"{name:24s} {ev['cv_alpha']:5.2f} {ev['ell_p']:6.2f} {ev['L_y']:5.2f} "
              f"{ev['f_sep']:5.2f} {ev['f_rec']:5.2f} {ev['eps_median']:8.2e} {ev['r2']:10.3g}")

    # ---- aggregate + candidate collapse analysis ---------------------------
    keys = ["case", "cv_alpha", "ell_p", "L_y", "delta", "f_sep", "f_rec",
            "L_sep", "eps_median", "frac_eps_lt_0p1", "r2", "rel_err",
            "cv_h_over_Lx", "cv_ellp_over_delta", "cv_Lsep_over_delta"]
    agg = {k: np.array([r[k] for r in rows]) for k in keys}

    candidates = {
        "alpha": agg["cv_alpha"],
        "h_over_Lx": agg["cv_h_over_Lx"],
        "ellp_over_delta": agg["cv_ellp_over_delta"],
        "Lsep_over_delta": agg["cv_Lsep_over_delta"],
        "f_rec": agg["f_rec"],
    }
    collapse = {}
    if not subset:
        print("\nCandidate collapse (Spearman rank correlation vs failure measures)")
        print("-" * 72)
        print(f"{'candidate':18s} {'rho(eps_med)':>13s} {'rho(f_eps<0.1)':>15s} "
              f"{'rho(R2)':>10s} {'rho(rel_err)':>13s}")
        for cname, cv in candidates.items():
            r_eps = spearman(cv, agg["eps_median"])
            r_fe = spearman(cv, agg["frac_eps_lt_0p1"])
            r_r2 = spearman(cv, agg["r2"])
            # rel_err is the SCALE-FREE relative L2 wall-stress error
            # ||tau_pred-tau_true||_2 / ||tau_true||_2. Unlike R^2 it carries no
            # geometry-dependent variance normalisation, so it is the cleaner
            # *effect* observable for separating a true control of the
            # cancellation from a confounded proxy of the R^2 normalisation.
            r_re = spearman(cv, agg["rel_err"])
            # 95% bootstrap CI on rho(R2) — the headline failure measure —
            # and on rho(rel_err), the scale-free effect measure.
            _, r2_lo, r2_hi = spearman_bootstrap_ci(cv, agg["r2"])
            _, re_lo, re_hi = spearman_bootstrap_ci(cv, agg["rel_err"])
            collapse[cname] = dict(rho_eps_median=r_eps, rho_frac_eps=r_fe,
                                   rho_r2=r_r2, rho_r2_ci_lo=r2_lo, rho_r2_ci_hi=r2_hi,
                                   rho_rel_err=r_re, rho_rel_err_ci_lo=re_lo,
                                   rho_rel_err_ci_hi=re_hi)
            print(f"{cname:18s} {r_eps:13.3f} {r_fe:15.3f} {r_r2:10.3f} "
                  f"  [{r2_lo:+.2f}, {r2_hi:+.2f}] {r_re:13.3f}")

    out = {f"agg_{k}": v for k, v in agg.items()}
    out["subset"] = subset
    out["Y_IDX"] = Y_IDX
    out["Re_b"] = RE_B
    for cname, c in collapse.items():
        for mk, mv in c.items():
            out[f"collapse_{cname}_{mk}"] = mv
    np.savez(os.path.join(RESULTS, "dose_response_xiao.npz"), **out)
    print(f"\nSaved -> {os.path.relpath(os.path.join(RESULTS, 'dose_response_xiao.npz'), ROOT)}")
    print(f"All periodic-hill cases reversed-shear-bearing: "
          f"{int(np.sum(agg['f_sep'] > 0))}/{len(rows)}")


if __name__ == "__main__":
    main()
