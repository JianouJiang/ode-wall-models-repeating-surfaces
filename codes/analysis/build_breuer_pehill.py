#!/usr/bin/env python3
"""
Thrust #19 L2 — process the canonical Breuer/Rapp/Manhart periodic-hill DNS
(ERCOFTAC UFR 3-30; Breuer et al. 2009, Comp. Fluids 38, 433-457) at three
GENUINE additional Reynolds numbers Re_b = 700, 1400, 2800 into the project's
wall-profile npz format, to add the THIRD..FIFTH Reynolds point the L1 Judge made
the single most important L2 deliverable.

The public Breuer files give wall-normal profiles (y/h, U/Ub, V/Ub, uu, vv, uv, k)
at the 10 standard streamwise stations x/h = {0.05,0.5,1,2,3,4,5,6,7,8}; there is
NO pressure column.  We therefore recover, from the VELOCITY profiles alone:
  * tau_w(x)  = nu * dU/dy|_wall      (near-wall linear fit; validated vs Krank)
  * the convective cancellation force the ODE drops,
        C(x)  = INT_0^ym ( U dU/dx + V dU/dy ) dy        (matched y_m = 0.10 H)
    split into C_x (needs streamwise FD across stations) and C_y (single profile).
We do NOT fabricate a pressure gradient; the multi-Re trend is carried by the
measurable residual tau_w (=> C_f(Re)) and the measurable dropped force C
(=> the closure-independent cancellation eps_C = |tau_w|/|C|).

Geometry: y is the GLOBAL vertical coordinate (0 at the bottom datum, 3.035 at the
top wall); the lower (hill) wall is the profile minimum y_wall(x).  Near-wall
analysis uses the wall-following coordinate y_loc = y - y_wall.

Pure read of downloaded ASCII (in codes/new_data_download/geometry_driven/
breuer_pehill_raw/); writes breuer_pehill_Re{Re}_wallonly.npz next to the Krank
files.  No CFD, no fabrication.

Run: OMP_NUM_THREADS=2 python3 codes/analysis/build_breuer_pehill.py
"""
import os
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
# GEO (symlink into the source project) is READ-ONLY; write our processed Breuer
# data to a project-LOCAL directory instead.
GEO  = os.path.join(ROOT, "codes", "data")
RAW  = os.path.join(GEO, "breuer_pehill_raw")

XSTATIONS = np.array([0.05, 0.5, 1., 2., 3., 4., 5., 6., 7., 8.])
YM = 0.10            # matched OUTER matching height (local wall-following), H=1
NEARWALL_NPTS = 3    # points used for the near-wall tau_w slope


def parse_station(fn):
    """Return y_global, U, V, uv (ascending in y_global)."""
    rows = []
    with open(fn) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 6:
                continue
            try:
                rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[5])])
            except ValueError:
                continue
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]            # ascending y
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def tau_w_nearwall(yloc, U, nu, k=NEARWALL_NPTS):
    """tau_w = nu dU/dy|_wall from the LOCAL near-wall slope (linear fit of the k
    nearest-wall points; NOT forced through a presumed wall location, so it needs
    no exact curvilinear wall position).  Sign preserved (negative if reversed)."""
    o = np.argsort(yloc)
    yy, uu = yloc[o][:k], U[o][:k]
    a = float(np.polyfit(yy, uu, 1)[0])     # dU/dy near the wall
    return nu * a, a


def convective_force(profiles, ix, ym=YM):
    """C = INT_0^ym (U dU/dx + V dU/dy) dy at station ix, wall-following.
    dU/dx by central FD across neighbouring stations at matched y_loc;
    dU/dy from the local profile.  Returns (C_total, C_x, C_y)."""
    yloc, U, V, _ = profiles[ix]
    # restrict to the matching layer
    m = yloc <= ym
    if m.sum() < 3:
        m = yloc <= np.sort(yloc)[3]
    yl, Ul, Vl = yloc[m], U[m], V[m]
    dUdy = np.gradient(Ul, yl)
    # streamwise gradient via neighbours on a common y_loc grid
    def Uinterp(jx):
        yj, Uj, _, _ = profiles[jx]
        return np.interp(yl, yj, Uj, left=Uj[0], right=Uj[-1])
    if 0 < ix < len(profiles) - 1:
        dx = XSTATIONS[ix + 1] - XSTATIONS[ix - 1]
        dUdx = (Uinterp(ix + 1) - Uinterp(ix - 1)) / dx
    elif ix == 0:
        dx = XSTATIONS[1] - XSTATIONS[0]
        dUdx = (Uinterp(1) - Ul) / dx
    else:
        dx = XSTATIONS[-1] - XSTATIONS[-2]
        dUdx = (Ul - Uinterp(ix - 1)) / dx
    Cx = np.trapz(Ul * dUdx, yl)
    Cy = np.trapz(Vl * dUdy, yl)
    return Cx + Cy, Cx, Cy


def process_Re(Re):
    nu = 1.0 / Re
    files = [os.path.join(RAW, f"UFR3-30_C_{Re}_data_MB-{s:03d}.dat") for s in range(1, 11)]
    profiles = []     # per station: (y_loc, U, V, uv)
    for fn in files:
        yg, U, V, uv = parse_station(fn)
        ywall = yg.min()
        yloc = yg - ywall
        profiles.append((yloc, U, V, uv))
    nst = len(profiles)
    tau_w = np.zeros(nst); u_tau = np.zeros(nst); is_sep = np.zeros(nst, bool)
    C = np.zeros(nst); Cx = np.zeros(nst); Cy = np.zeros(nst)
    for i in range(nst):
        yloc, U, V, uv = profiles[i]
        tw, _ = tau_w_nearwall(yloc, U, nu)
        tau_w[i] = tw
        u_tau[i] = np.sqrt(abs(tw))
        # separated if near-wall streamwise velocity is reversed
        is_sep[i] = np.mean(U[:NEARWALL_NPTS]) < 0
        C[i], Cx[i], Cy[i] = convective_force(profiles, i)
    out = dict(
        x=XSTATIONS, tau_w=tau_w, u_tau=u_tau, is_separated=is_sep,
        C_conv=C, C_x=Cx, C_y=Cy, nu=np.full(nst, nu), Re=np.full(nst, Re),
        dp_dx=np.full(nst, np.nan),    # no pressure column in Breuer public data
        geometry="periodic_hills",
        metadata_json=json.dumps({
            "source": "Breuer, Peller, Rapp & Manhart (2009)",
            "doi": "10.1016/j.compfluid.2008.05.002",
            "database": "ERCOFTAC UFR 3-30 (LESOCC curvilinear)",
            "Re_H": Re, "type": "DNS", "normalization": "h=1, U_b=1, rho=1",
            "notes": ("tau_w & convective force recovered from velocity profiles; "
                      "no pressure column; y_wall = profile min; 10 stations")}))
    fn_out = os.path.join(GEO, f"breuer_pehill_Re{Re}_wallonly.npz")
    np.savez(fn_out, **out)
    sep = is_sep
    print(f"Re={Re:5d}  nu={nu:.3e}  sep stations x={list(XSTATIONS[sep])}")
    print(f"   tau_w(sep)={np.round(tau_w[sep],5)}")
    print(f"   C_f=2<u_tau^2>(sep)={2*np.mean(u_tau[sep]**2):.3e}")
    eps_C = np.abs(tau_w[sep]) / np.abs(C[sep])
    print(f"   eps_C=|tau_w|/|C| (sep) median={np.median(eps_C):.3f}  "
          f"[C_y/C ratio median={np.median(np.abs(Cy[sep])/np.abs(C[sep])):.2f}]")
    print(f"   saved -> {os.path.basename(fn_out)}")
    return out


def main():
    print("=" * 78)
    print("BUILD Breuer periodic-hill DNS (Re=700,1400,2800) -> wallonly npz")
    print("=" * 78)
    for Re in (700, 1400, 2800):
        process_Re(Re)
    print("=" * 78)


if __name__ == "__main__":
    main()
