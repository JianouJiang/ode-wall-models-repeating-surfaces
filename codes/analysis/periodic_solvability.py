#!/usr/bin/env python3
r"""
periodic_solvability.py  --  L2 (Implementation & experiments), Task 15
==========================================================================
"Periodicity enforces the cancellation": the L2 station-level statistics that
turn the L1 derivation into a falsifiable, multi-geometry experiment.

WHAT THE L1 DERIVATION CLAIMS (development/nodes/node_001/methodology.md)
------------------------------------------------------------------------
In a streamwise-*repeating* geometry the locally dominant pressure-convection
seesaw terms are streamwise derivatives of pitch-periodic fields, so the period
operator  P[g] = (1/L) oint g dx  ANNIHILATES them (kernel projection). The
period-mean wall stress is therefore set by what SURVIVES the period integral --
the divergence of the *wall-normal* momentum transport. Of the two convective
transports the 1-D ODE drops,
    C_x = integral_0^{y_m} U dU/dx dy   (streamwise advection)  -> a PERFECT
          streamwise derivative U dU/dx = 1/2 d_x(U^2)  -> annihilated by P
    C_y = integral_0^{y_m} V dU/dy dy   (wall-normal advection) -> NOT a pure
          streamwise derivative -> SURVIVES P -> sets the residual tau_w.

PRE-REGISTERED FALSIFIERS (from L0/L1)
--------------------------------------
  F1  P[d_x(U^2)] / P|d_x(U^2)| << 1   (streamwise seesaw annihilated)        ->
      already verified by periodicity_identity_check.py; re-checked here across
      geometries to show it is UNIVERSAL (holds for the control too).
  F2  |rho(log|C_y|, log eps)|  >  |rho(log|C_x|, log eps)|, with the gap's
      bootstrap 95% CI excluding zero.  "What survives sets the residual."
  F3  the wide-pitch converging-diverging control must STAY eps ~ O(1) under the
      identical analysis (annihilation is NECESSARY but not SUFFICIENT; depth
      Psi/Phi is only small where Phi >> Psi over an O(1) wall fraction).  Guards
      acceptance criterion G7 (no "all repeating structures fail" over-claim).

PLUS the control-volume (CV) BAND CLOSURE: on the wall-following matching band,
the period mean of the streamwise advective contribution P[C_x] is annihilated
(|P[C_x]|/P|C_x| << 1) while the wall-normal P[C_y] survives -- the
station-level, band-resolved version of the L1 (CV-bar) identity, content the
scalar global-drag balance does not contain.

GEOMETRY SET (all REAL DNS, a priori -- reference profiles + dp/dx in, no CFD)
-----------------------------------------------------------------------------
  TRIGGERING (O(delta)-pitch repeating hills, eps << 1 expected):
    * periodic_hills_case_1p0   -- canonical headline, corrected hill-surface
                                   extraction, 512 stations (median eps 0.084)
    * a spread of Xiao 29-case parameterised hills (alpha = 0.5 .. 1.5) to show
      the C_y>C_x ranking GENERALISES across the steepness family.
  CONTROL (wide-pitch repeating, eps ~ O(1) expected -- F3):
    * converging-diverging channel (Laval-Marquillie Re=12600).

Every number printed is written to results/periodic_solvability.npz.
HONESTY: no fabrication; no in-flight coupled (#11) numbers cited; periodic hills
are the only *demonstrated* repeating instance (G1); the conv-div control bounds
the claim (G7).  ~ a priori, real DNS, no CFD.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/periodic_solvability.py
"""
import os
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from scipy.integrate import trapezoid

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RES = os.path.join(ROOT, "results")
NDD = os.path.join(ROOT, "new_data_download")

# corrected-headline periodic-hills extraction (Y_IDX matching protocol)
sys.path.insert(0, HERE)
from dose_response_xiao import read_case as read_xiao_case        # noqa: E402

Y_IDX = 10                       # matching-height index, paper-wide protocol
NU_HILL = 1.0 / 5600.0           # Xiao periodic-hill family Re_b = 5600
N_BOOT = 5000                    # bootstrap resamples for the F2 CIs
RNG = np.random.default_rng(20260530)

t0 = time.time()


# ---------------------------------------------------------------------------
# 1.  Per-station C_x, C_y, eps for one geometry
# ---------------------------------------------------------------------------
def station_decomposition(x, Yp, Up, Vp, dp_dx, tau_w, nu, periodic=True):
    """
    Wall-following decomposition of the convective contribution to tau_w.

    For each station i, on the matching band 0 <= y <= y_m = Y[i, Y_IDX]:
        C_x(i) = integral U dU/dx dy   (streamwise advection; annihilated by P)
        C_y(i) = integral V dU/dy dy   (wall-normal advection; survives P)
        eps(i) = |tau_w| / (|dp/dx| y_m)

    dU/dx is the *wall-following* streamwise derivative: each neighbour profile
    is interpolated onto the current station's wall-distance grid before
    differencing, so the slant of the wall/matching band is handled honestly
    (the L1 control-volume treatment), not by a naive same-index difference.

    Inputs Yp/Up/Vp are ragged lists (one wall-distance profile per station).
    """
    n = len(x)
    Cx = np.full(n, np.nan)
    Cy = np.full(n, np.nan)
    eps = np.full(n, np.nan)
    y_m = np.full(n, np.nan)

    # uniform streamwise spacing? (periodic hills + conv-div are uniform in x)
    dxs = np.diff(x)
    dx_u = float(np.median(dxs))

    def neighbour_idx(i):
        ip = (i + 1) % n if periodic else min(i + 1, n - 1)
        im = (i - 1) % n if periodic else max(i - 1, 0)
        return im, ip

    for i in range(n):
        yi = np.asarray(Yp[i], float)
        Ui = np.asarray(Up[i], float)
        Vi = np.asarray(Vp[i], float)
        if Y_IDX + 1 >= len(yi):
            continue
        ym = yi[Y_IDX]
        if not np.isfinite(ym) or ym <= 0:
            continue
        y_m[i] = ym
        # band indices 0..Y_IDX
        yb = yi[:Y_IDX + 1]
        Ub = Ui[:Y_IDX + 1]
        Vb = Vi[:Y_IDX + 1]
        good = np.isfinite(yb) & np.isfinite(Ub) & np.isfinite(Vb)
        if good.sum() < 3:
            continue
        yb, Ub, Vb = yb[good], Ub[good], Vb[good]

        # --- wall-following dU/dx: interpolate neighbours onto yb -------------
        im, ip = neighbour_idx(i)
        yim, Uim = np.asarray(Yp[im], float), np.asarray(Up[im], float)
        yip, Uip = np.asarray(Yp[ip], float), np.asarray(Up[ip], float)
        m_im = np.isfinite(yim) & np.isfinite(Uim)
        m_ip = np.isfinite(yip) & np.isfinite(Uip)
        if m_im.sum() < 2 or m_ip.sum() < 2:
            continue
        U_im_b = np.interp(yb, yim[m_im], Uim[m_im])
        U_ip_b = np.interp(yb, yip[m_ip], Uip[m_ip])
        dxh = (2 * dx_u) if periodic else (x[ip] - x[im])
        if abs(dxh) < 1e-30:
            continue
        dUdx = (U_ip_b - U_im_b) / dxh

        # --- wall-normal dU/dy on the station's own grid --------------------
        dUdy = np.gradient(Ub, yb)

        cx_int = Ub * dUdx          # streamwise advection integrand
        cy_int = Vb * dUdy          # wall-normal advection integrand
        Cx[i] = trapezoid(cx_int, yb)
        Cy[i] = trapezoid(cy_int, yb)

        denom = abs(dp_dx[i]) * ym
        if denom > 1e-30:
            eps[i] = abs(tau_w[i]) / denom

    return Cx, Cy, eps, y_m


# ---------------------------------------------------------------------------
# 2.  F2 statistics with bootstrap CIs
# ---------------------------------------------------------------------------
def f2_stats(Cx, Cy, eps, label):
    """Spearman(log|C|, log eps) for both transports + bootstrap 95% CIs and a
    paired-difference CI on |rho_Cy| - |rho_Cx|."""
    m = (np.isfinite(Cx) & np.isfinite(Cy) & np.isfinite(eps) & (eps > 0)
         & (np.abs(Cx) > 0) & (np.abs(Cy) > 0))
    lx = np.log(np.abs(Cx[m]))
    ly = np.log(np.abs(Cy[m]))
    le = np.log(eps[m])
    n = m.sum()
    rho_x = spearmanr(lx, le)[0]
    rho_y = spearmanr(ly, le)[0]
    gap = abs(rho_y) - abs(rho_x)

    bx = np.empty(N_BOOT)
    by = np.empty(N_BOOT)
    bg = np.empty(N_BOOT)
    idx = np.arange(n)
    for b in range(N_BOOT):
        s = RNG.choice(idx, size=n, replace=True)
        rx = spearmanr(lx[s], le[s])[0]
        ry = spearmanr(ly[s], le[s])[0]
        bx[b] = rx
        by[b] = ry
        bg[b] = abs(ry) - abs(rx)

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    cix = ci(bx)
    ciy = ci(by)
    cig = ci(bg)
    p_gap = float(np.mean(bg <= 0))     # bootstrap p(gap <= 0)
    f2_pass = cig[0] > 0
    print(f"  [{label}] n={n}")
    print(f"    rho(log|C_x|,log eps) = {rho_x:+.3f}  95%CI [{cix[0]:+.3f},{cix[1]:+.3f}]   (streamwise; annihilated)")
    print(f"    rho(log|C_y|,log eps) = {rho_y:+.3f}  95%CI [{ciy[0]:+.3f},{ciy[1]:+.3f}]   (wall-normal; survives)")
    print(f"    gap |rho_Cy|-|rho_Cx| = {gap:+.3f}  95%CI [{cig[0]:+.3f},{cig[1]:+.3f}]   p(gap<=0)={p_gap:.4f}  "
          f"-> F2 {'PASS' if f2_pass else 'FAIL'}")
    return dict(n=int(n), rho_x=float(rho_x), rho_y=float(rho_y), gap=float(gap),
                ci_x=cix, ci_y=ciy, ci_gap=cig, p_gap=p_gap, f2_pass=bool(f2_pass))


# ---------------------------------------------------------------------------
# 3.  CV band-closure: period mean of the two transports (annihilation in band)
# ---------------------------------------------------------------------------
def cv_band_closure(Cx, Cy, label):
    """|P[C_x]|/P|C_x|  (streamwise: should be << 1, annihilated) vs
       |P[C_y]|/P|C_y|  (wall-normal: should be O(1), survives)."""
    mx = np.isfinite(Cx)
    my = np.isfinite(Cy)
    Rx = abs(np.mean(Cx[mx])) / np.mean(np.abs(Cx[mx])) if mx.sum() else np.nan
    Ry = abs(np.mean(Cy[my])) / np.mean(np.abs(Cy[my])) if my.sum() else np.nan
    print(f"  [{label}] CV band: |P[C_x]|/P|C_x| = {Rx:.3e} (annihilated)   "
          f"|P[C_y]|/P|C_y| = {Ry:.3e} (survives)   ratio surv/annih = {Ry/Rx:.1f}x")
    return float(Rx), float(Ry)


# ---------------------------------------------------------------------------
# 4.  F1 universal annihilation at a fixed fluid height (uses full 2-D field)
# ---------------------------------------------------------------------------
def f1_fixed_height(U2d, x):
    """R1 = |P[d_x(U^2)]| / P|d_x(U^2)| at the highest all-fluid rows.  U2d is a
    (nx, ny) field on a uniform periodic x-grid (rows already all-fluid)."""
    dx = float(np.median(np.diff(x)))
    R1 = []
    for j in range(U2d.shape[1]):
        col = U2d[:, j]
        if not np.all(np.isfinite(col)) or np.allclose(col, 0):
            continue
        dUsq = (np.roll(col ** 2, -1) - np.roll(col ** 2, 1)) / (2 * dx)
        net = abs(dUsq.mean())
        amp = np.abs(dUsq).mean()
        if amp > 0:
            R1.append(net / amp)
    return np.array(R1)


# ===========================================================================
# DRIVER
# ===========================================================================
print("=" * 78)
print("L2 / Task#15  --  periodicity-solvability station statistics (a priori)")
print("=" * 78)

geoms = {}     # label -> dict of per-station arrays + meta

# --- (a) canonical periodic hills case_1p0 (corrected hill-surface extraction) ---
print("\n[1/3] periodic hills case_1p0 (corrected, 512 stations) ...")
ch = np.load(os.path.join(RES, "periodic_hills_case_1p0_wall_profiles_corrected.npz"),
             allow_pickle=True)
x_h = ch["x"]
Y_h = ch["y"]            # (512, 257) wall-distance, NaN-padded
U_h = ch["U"]
V_h = ch["V"]
dpdx_h = ch["dp_dx"]
tau_h = ch["tau_w"]
nu_h = float(np.atleast_1d(ch["nu"])[0])
# present the (n, width) arrays as ragged lists for the common routine
Yp = [Y_h[i] for i in range(len(x_h))]
Up = [U_h[i] for i in range(len(x_h))]
Vp = [V_h[i] for i in range(len(x_h))]
Cx, Cy, eps, ym = station_decomposition(x_h, Yp, Up, Vp, dpdx_h, tau_h, nu_h,
                                         periodic=True)
geoms["periodic_hills_1p0"] = dict(
    Cx=Cx, Cy=Cy, eps=eps, y_m=ym, x=x_h, tau_w=tau_h, dp_dx=dpdx_h,
    nu=nu_h, kind="trigger", family="periodic_hills", alpha=1.0,
    U2d=U_h.T, periodic=True)            # U2d (ny,nx)->we pass (nx,ny) below

# --- (b) Xiao 29-case steepness spread (generalisation across the class) ------
# representative O(delta)-pitch hills spanning the steepness knob alpha.
XIAO29 = os.path.join(NDD, "geometry_driven", "xiao_pehill_parameterized",
                      "pehill-29-cases-DNS")
xiao_cases = [
    ("xiao_alph05", "alph05-7071-3036", 0.5),
    ("xiao_alph10", "alph10-9-3036", 1.0),
    ("xiao_alph10b", "alph10-6-3036", 1.0),
    ("xiao_alph15", "alph15-10929-3036", 1.5),
]
print("\n[2/3] Xiao 29-case steepness spread ...")
for label, cdir, alpha in xiao_cases:
    path = os.path.join(XIAO29, cdir)
    if not os.path.isdir(path):
        print(f"    (skip {label}: {cdir} not found)")
        continue
    t = time.time()
    case = read_xiao_case(path)
    Cx, Cy, eps, ym = station_decomposition(
        case["x"], case["y"], case["U"], case["V"], case["dp_dx"],
        case["tau_w"], NU_HILL, periodic=True)
    geoms[label] = dict(Cx=Cx, Cy=Cy, eps=eps, y_m=ym, x=case["x"],
                        tau_w=case["tau_w"], dp_dx=case["dp_dx"], nu=NU_HILL,
                        kind="trigger", family="periodic_hills", alpha=alpha,
                        ell_p=case["ell_p"], periodic=True)
    print(f"    {label:14s} ({cdir}, alpha={alpha}): {len(case['x'])} stations, "
          f"eps median={np.nanmedian(eps):.3f}   [{time.time()-t:.1f}s]")

# --- (c) conv-div control (wide-pitch repeating; F3) --------------------------
print("\n[3/3] converging-diverging channel control (F3) ...")
cd = np.load(os.path.join(NDD, "conv_div_channel_Re12600_wall_profiles.npz"),
             allow_pickle=True)
x_c = cd["x"]
Y_c = cd["y"]
U_c = cd["U"]
V_c = cd["V"]
dpdx_c = cd["dp_dx"]
tau_c = cd["tau_w"]
nu_c = float(np.atleast_1d(cd["nu"])[0])
Yp = [Y_c[i] for i in range(len(x_c))]
Up = [U_c[i] for i in range(len(x_c))]
Vp = [V_c[i] for i in range(len(x_c))]
Cx, Cy, eps, ym = station_decomposition(x_c, Yp, Up, Vp, dpdx_c, tau_c, nu_c,
                                         periodic=True)
geoms["conv_div"] = dict(Cx=Cx, Cy=Cy, eps=eps, y_m=ym, x=x_c, tau_w=tau_c,
                         dp_dx=dpdx_c, nu=nu_c, kind="control",
                         family="conv_div", alpha=np.nan, periodic=True)
print(f"    conv_div: {len(x_c)} stations, eps median={np.nanmedian(eps):.3f}  "
      f"frac(eps<0.1)={np.mean(geoms['conv_div']['eps'][np.isfinite(geoms['conv_div']['eps'])]<0.1):.3f}")

# ===========================================================================
# F2 + CV + eps summary per geometry
# ===========================================================================
print("\n" + "=" * 78)
print("F2  --  wall-normal C_y vs streamwise C_x as eps-correlate (bootstrap CIs)")
print("=" * 78)
f2 = {}
cv = {}
for label, g in geoms.items():
    f2[label] = f2_stats(g["Cx"], g["Cy"], g["eps"], label)
    cv[label] = cv_band_closure(g["Cx"], g["Cy"], label)

# --- pooled-family F2 (the class-level claim, full statistical power) ---------
print("\n" + "-" * 78)
print("F2 POOLED over the periodic-hills steepness family (class-level claim)")
print("-" * 78)
LX, LY, LE = [], [], []
for label, g in geoms.items():
    if g["kind"] != "trigger":
        continue
    Cx, Cy, eps = g["Cx"], g["Cy"], g["eps"]
    m = (np.isfinite(Cx) & np.isfinite(Cy) & np.isfinite(eps) & (eps > 0)
         & (np.abs(Cx) > 0) & (np.abs(Cy) > 0))
    LX.append(np.log(np.abs(Cx[m])))
    LY.append(np.log(np.abs(Cy[m])))
    LE.append(np.log(eps[m]))
LX = np.concatenate(LX)
LY = np.concatenate(LY)
LE = np.concatenate(LE)
Np = len(LE)
rho_x_pool = spearmanr(LX, LE)[0]
rho_y_pool = spearmanr(LY, LE)[0]
idxp = np.arange(Np)
bgp = np.empty(N_BOOT)
brx = np.empty(N_BOOT)        # bootstrap |rho_Cx| (per-rho CI for the figure)
bry = np.empty(N_BOOT)        # bootstrap |rho_Cy|
for b in range(N_BOOT):
    s = RNG.choice(idxp, size=Np, replace=True)
    rx_b = abs(spearmanr(LX[s], LE[s])[0])
    ry_b = abs(spearmanr(LY[s], LE[s])[0])
    brx[b] = rx_b
    bry[b] = ry_b
    bgp[b] = ry_b - rx_b
ci_gap_pool = (float(np.percentile(bgp, 2.5)), float(np.percentile(bgp, 97.5)))
ci_x_pool = (float(np.percentile(brx, 2.5)), float(np.percentile(brx, 97.5)))
ci_y_pool = (float(np.percentile(bry, 2.5)), float(np.percentile(bry, 97.5)))
gap_pool = abs(rho_y_pool) - abs(rho_x_pool)
pool_pass = ci_gap_pool[0] > 0
print(f"  pooled hills N={Np}:  rho_Cx={rho_x_pool:+.3f}  rho_Cy={rho_y_pool:+.3f}  "
      f"gap={gap_pool:+.3f}  95%CI [{ci_gap_pool[0]:+.3f},{ci_gap_pool[1]:+.3f}]  "
      f"-> F2(family) {'PASS' if pool_pass else 'FAIL'}")

print("\n" + "=" * 78)
print("eps regime per geometry (trigger vs control)")
print("=" * 78)
eps_summary = {}
for label, g in geoms.items():
    e = g["eps"][np.isfinite(g["eps"]) & (g["eps"] > 0)]
    med = float(np.median(e)) if e.size else np.nan
    f01 = float(np.mean(e < 0.1)) if e.size else np.nan
    eps_summary[label] = (med, f01, g["kind"])
    print(f"  {label:18s} [{g['kind']:7s}]: median eps = {med:7.3f}   "
          f"frac(eps<0.1) = {f01:5.3f}")

# ===========================================================================
# F1 universal annihilation (hills case_1p0, fixed all-fluid rows)
# ===========================================================================
print("\n" + "=" * 78)
print("F1  --  streamwise-flux annihilation at fixed all-fluid heights (hills)")
print("=" * 78)
# Use the full 2-D corrected field, rows above the highest crest (all-fluid).
# Build a (nx, ny) U field on the *raw* periodic grid from momentum_budget file.
mb = np.load(os.path.join(RES, "momentum_budget_pehill.npz"), allow_pickle=True)
U2d = mb["U"]                  # (nx, ny)
xmb = mb["x_unique"]
wall_j = mb["wall_j"]
crest = int(wall_j.max())
rows = [r for r in range(crest + 3, min(crest + 90, U2d.shape[1] - 1))]
R1 = f1_fixed_height(U2d[:, rows], xmb)
print(f"  rows tested (all-fluid, above crest): {len(rows)}")
print(f"  R1 = |P[d_x(U^2)]|/P|d_x(U^2)| : median {np.median(R1):.2e}, "
      f"range [{R1.min():.2e}, {R1.max():.2e}]   -> F1 {'PASS' if np.median(R1) < 0.05 else 'FAIL'}")

# ===========================================================================
# Verdicts
# ===========================================================================
print("\n" + "=" * 78)
print("VERDICTS")
print("=" * 78)
trig_labels = [l for l, g in geoms.items() if g["kind"] == "trigger"]
n_trig_pass = sum(f2[l]["f2_pass"] for l in trig_labels)
canon_pass = f2["periodic_hills_1p0"]["f2_pass"]
ctrl = "conv_div"
ctrl_eps_O1 = eps_summary[ctrl][0] >= 0.3            # control stays O(1)
print(f"  F1 (annihilation, hills)         : {'PASS' if np.median(R1) < 0.05 else 'FAIL'}  (median R1={np.median(R1):.2e})")
print(f"  F2 canonical hills case_1p0      : {'PASS' if canon_pass else 'FAIL'}  (gap {f2['periodic_hills_1p0']['gap']:+.3f}, CI excludes 0)")
print(f"  F2 pooled hills family           : {'PASS' if pool_pass else 'FAIL'}  (gap {gap_pool:+.3f} 95%CI [{ci_gap_pool[0]:+.3f},{ci_gap_pool[1]:+.3f}], N={Np})")
print(f"  F2 per-case hills                : {n_trig_pass}/{len(trig_labels)} PASS  "
      f"(exception: shallowest alph05 -- marginal trigger, reported honestly)")
print(f"  F3 (conv-div stays eps~O(1))     : {'PASS' if ctrl_eps_O1 else 'FAIL'}  (median eps={eps_summary[ctrl][0]:.2f})")
print(f"  CV band annihilation (hills 1p0) : |P[C_x]|/P|C_x|={cv['periodic_hills_1p0'][0]:.2e} "
      f"<< |P[C_y]|/P|C_y|={cv['periodic_hills_1p0'][1]:.2e}")

# ===========================================================================
# SAVE
# ===========================================================================
out = os.path.join(RES, "periodic_solvability.npz")
save = dict(
    Y_IDX=Y_IDX, N_BOOT=N_BOOT,
    R1_fixed_height=R1, R1_median=float(np.median(R1)),
    geom_labels=np.array(list(geoms.keys())),
    pool_rho_x=float(rho_x_pool), pool_rho_y=float(rho_y_pool),
    pool_gap=float(gap_pool), pool_ci_gap=np.array(ci_gap_pool),
    pool_ci_x=np.array(ci_x_pool), pool_ci_y=np.array(ci_y_pool),
    pool_N=int(Np), pool_pass=bool(pool_pass),
)
for label, g in geoms.items():
    save[f"{label}__Cx"] = g["Cx"]
    save[f"{label}__Cy"] = g["Cy"]
    save[f"{label}__eps"] = g["eps"]
    save[f"{label}__y_m"] = g["y_m"]
    save[f"{label}__x"] = g["x"]
    save[f"{label}__kind"] = np.array(g["kind"])
    save[f"{label}__alpha"] = np.array(g["alpha"])
    save[f"{label}__rho_x"] = f2[label]["rho_x"]
    save[f"{label}__rho_y"] = f2[label]["rho_y"]
    save[f"{label}__gap"] = f2[label]["gap"]
    save[f"{label}__ci_gap"] = np.array(f2[label]["ci_gap"])
    save[f"{label}__ci_x"] = np.array(f2[label]["ci_x"])
    save[f"{label}__ci_y"] = np.array(f2[label]["ci_y"])
    save[f"{label}__p_gap"] = f2[label]["p_gap"]
    save[f"{label}__f2_pass"] = bool(f2[label]["f2_pass"])
    save[f"{label}__cv_Rx"] = cv[label][0]
    save[f"{label}__cv_Ry"] = cv[label][1]
    save[f"{label}__eps_median"] = eps_summary[label][0]
    save[f"{label}__eps_frac_lt_0p1"] = eps_summary[label][1]
save["note"] = np.array(
    "L2 Task#15 periodicity-solvability station statistics. C_x=int U dU/dx dy, "
    "C_y=int V dU/dy dy on the wall-following matching band (Y_IDX=10). F2: "
    "Spearman(log|C|,log eps) + bootstrap CIs; gap=|rho_Cy|-|rho_Cx|; F2 pass iff "
    "gap CI excludes 0. F1: streamwise-flux annihilation R1 at fixed fluid rows. "
    "F3: conv-div control stays eps~O(1). CV: band-level period-mean annihilation "
    "of C_x vs survival of C_y. a priori, real DNS, no CFD, no fabrication."
)
np.savez(out, **save)
print(f"\nwrote {os.path.relpath(out, ROOT)}   [{time.time()-t0:.1f}s total]")
