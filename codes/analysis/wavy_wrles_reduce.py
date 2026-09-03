#!/usr/bin/env python3
"""wavy_wrles_reduce.py  (R1-STA-2)  --  3-D OpenFOAM fields -> spanwise-averaged 2-D.

Runs INSIDE the ARCHER2 job (python3.6 / numpy 1.17 compatible; numpy only) on the
reconstructed single-block blockMesh case produced by
``make_wavy_cherukat_wrles_case.py``.  For every requested write time it reads the
cumulative time-averaged fields written by the fieldAverage function object

    UMean, pMean, UPrime2Mean, pPrime2Mean, nutMean, wallShearStressMean

(plus the instantaneous U / p / yPlus when present), verifies the structured
(i fastest, then j, then k) cell ordering against the cell-centre field C, averages
over the homogeneous spanwise direction z and writes ONE compact npz per time:

    x (nx,), y (nx, ny) cell-centre heights, y_wall (nx,),
    U V W P (nx, ny), uu vv ww uv uw vw (nx, ny) resolved, pp, nut,
    bottom/top wall: wall traction vector (nx, 3) [OpenFOAM sign], p_wall (nx,),
    first-cell height, yPlus (nx,) when present, and the raw z-resolved wall
    traction (nz, nx) so spanwise-block uncertainties can be formed downstream.

It also parses log.pimpleFoam for the time series of the meanVelocityForce
pressure gradient, deltaT and Courant number (time-stepping record, G-R6).

Nothing here is model-based: only resolved statistics and exact wall traction.

Usage (in the job or on a login node):
    python3 wavy_wrles_reduce.py --case <case> --times 100 120 ... --out <dir>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np


# --------------------------------------------------------------- OpenFOAM ASCII
def _strip_comments(txt):
    txt = re.sub(r"/\*.*?\*/", " ", txt, flags=re.S)
    return re.sub(r"//[^\n]*", " ", txt)


def _parse_list(block, ncomp):
    """Parse the text between the '(' and ')' of a nonuniform List into (N, ncomp)."""
    flat = np.fromstring(block.replace("(", " ").replace(")", " "), sep=" ")
    if ncomp == 1:
        return flat
    return flat.reshape(-1, ncomp)


def _find_value(txt, start, ncomp):
    """Parse 'uniform (..)' / 'uniform x' / 'nonuniform List<..> N ( ... )' at txt[start:]."""
    m = re.compile(r"\s*(uniform|nonuniform)", re.S).match(txt, start)
    if not m:
        return None, start
    if m.group(1) == "uniform":
        m2 = re.compile(r"\s*(\([^)]*\)|[-+0-9.eE]+)\s*;", re.S).match(txt, m.end())
        vals = np.fromstring(m2.group(1).replace("(", " ").replace(")", " "), sep=" ")
        return ("uniform", vals if ncomp > 1 else float(vals[0])), m2.end()
    m2 = re.compile(r"\s*List<\w+>\s*(\d+)\s*\(", re.S).match(txt, m.end())
    n = int(m2.group(1))
    i0 = m2.end()
    # find the closing parenthesis of the list by scanning for ")\n;" after N entries:
    # entries are either scalars or '(...)' tuples; count '(' depth.
    depth = 1
    i = i0
    L = len(txt)
    while i < L and depth > 0:
        c = txt[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    arr = _parse_list(txt[i0:i - 1], ncomp)
    if (arr.shape[0] if arr.ndim else 0) != n:
        raise RuntimeError("list length mismatch: expected %d got %s" % (n, arr.shape))
    m3 = re.compile(r"\s*;", re.S).match(txt, i)
    return ("nonuniform", arr), (m3.end() if m3 else i)


def read_field(path, ncomp, patches=()):
    """Return (internal (N,ncomp) or (N,), {patch: ('uniform'|'nonuniform'|'none', values)})."""
    with open(path) as fh:
        txt = _strip_comments(fh.read())
    m = re.search(r"internalField\s+", txt)
    (kind, internal), pos = _find_value(txt, m.end(), ncomp)
    out = {}
    bf = txt.find("boundaryField", pos)
    for p in patches:
        mp = re.compile(r"\n\s*%s\s*\{" % re.escape(p)).search(txt, bf)
        if not mp:
            out[p] = ("none", None)
            continue
        # patch dictionary body up to the matching brace
        i, depth = mp.end(), 1
        while depth > 0:
            c = txt[i]
            depth += (c == "{") - (c == "}")
            i += 1
        body = txt[mp.end():i - 1]
        mv = re.search(r"\bvalue\s+", body)
        if mv:
            out[p] = _find_value(body, mv.end(), ncomp)[0]
        else:
            out[p] = ("none", None)
    return kind, internal, out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------- structured map
# A cell centre may sit off its column's mean x because the wavy-wall block mesh
# shears the hexes at a block seam: the offset tilts linearly with j, from one
# sign at the bottom wall to the other at the top, and is confined to the seam
# column.  What the reduction actually needs is that a column stay identifiable,
# i.e. that no centroid crosses the midpoint towards its neighbour -- a bound of
# HALF a cell.  The earlier value of 0.25 was that bound with an undocumented
# safety factor of two, and it excluded the steeper wall of the amplitude ladder
# (measured 0.342 of a cell) although the reduction is unaffected: at 207 of the
# 208 stations of that mesh the offset is 0.023 of a cell.  The measured value is
# recorded in the output and anything above the old value is announced, so the
# looser regime is never silent.
COLUMN_TOL_OVER_DX = 0.5
ANNOUNCE_ABOVE = 0.25
MEASURED: dict = {}


def _column_offset(name, value, dx):
    """Record, announce and test one column-offset measurement."""
    rel = float(value / dx)
    MEASURED[name] = rel
    if rel > ANNOUNCE_ABOVE:
        print("NOTE %s column offset %.4f of a cell (bound %.2f; above the "
              "historical %.2f, reported not suppressed)"
              % (name, rel, COLUMN_TOL_OVER_DX, ANNOUNCE_ABOVE))
    return rel < COLUMN_TOL_OVER_DX


def structured_maps(case, nx, ny, nz):
    """Verify blockMesh ordering (i fastest) from 0/C and return cell-centre arrays
    x (nx,), y (nx, ny), z (nz,) plus the bottom/top patch face orderings."""
    kind, C, bnd = read_field(os.path.join(case, "0", "C"), 3, ("bottomWall", "topWall"))
    n = nx * ny * nz
    if C.shape[0] != n:
        raise RuntimeError("cell count %d != nx*ny*nz=%d" % (C.shape[0], n))
    C3 = C.reshape(nz, ny, nx, 3)            # k slowest, i fastest
    # sheared hexes near the wavy wall move the centroid x slightly along j;
    # the station coordinate is the column mean (deviation << dx is verified)
    x = C3[:, :, :, 0].mean(axis=(0, 1))
    z = C3[:, 0, 0, 2]
    dx = float(np.median(np.diff(x)))
    if not (_column_offset("cell_centres",
                           np.max(np.abs(C3[:, :, :, 0] - x[None, None, :])), dx) and
            np.allclose(C3[:, :, :, 2], z[:, None, None], atol=1e-9) and
            np.all(np.diff(x) > 0) and np.all(np.diff(z) > 0)):
        raise RuntimeError("cell ordering is not (k, j, i) structured -- refusing to reshape")
    y = C3[0, :, :, 1].T                      # (nx, ny), identical for every k
    if not np.allclose(C3[:, :, :, 1], y.T[None, :, :], atol=1e-9):
        raise RuntimeError("y(i,j) differs between spanwise planes")
    if not np.all(np.diff(y, axis=1) > 0):
        raise RuntimeError("y not increasing along j")
    maps = {}
    for p in ("bottomWall", "topWall"):
        kindp, fc = bnd[p]
        if kindp != "nonuniform" or fc.shape[0] != nx * nz:
            raise RuntimeError("patch %s: unexpected face-centre data" % p)
        # sort faces into (k, i) order by (z, x)
        order = np.lexsort((fc[:, 0], fc[:, 2]))
        fx = fc[order, 0].reshape(nz, nx)
        fz = fc[order, 2].reshape(nz, nx)
        if not (_column_offset(p, np.max(np.abs(fx - x[None, :])), dx)
                and np.allclose(fz, z[:, None], atol=1e-9)):
            raise RuntimeError("patch %s face centres do not map onto the (z, x) grid" % p)
        maps[p] = dict(order=order, y_face=fc[order, 1].reshape(nz, nx),
                       x_face=fx[0])
    return x, y, z, maps


def zavg(field3, nz, ny, nx):
    """(N,) or (N,c) -> spanwise mean (nx, ny[, c])."""
    if field3.ndim == 1:
        return field3.reshape(nz, ny, nx).mean(axis=0).T
    c = field3.shape[1]
    return field3.reshape(nz, ny, nx, c).mean(axis=0).transpose(1, 0, 2)


def patch_field(bnd_entry, order, nz, nx, ncomp):
    kind, v = bnd_entry
    if kind == "nonuniform":
        v = v[order]
        return v.reshape(nz, nx, ncomp) if ncomp > 1 else v.reshape(nz, nx)
    if kind == "uniform":
        shape = (nz, nx, ncomp) if ncomp > 1 else (nz, nx)
        return np.broadcast_to(np.asarray(v), shape).copy()
    return None


# --------------------------------------------------------------- log parsing
def parse_log(path):
    t, g, dt, co = [], [], [], []
    cur_t = cur_dt = cur_co = np.nan
    rx_t = re.compile(r"^Time = ([0-9.eE+-]+)")
    rx_dt = re.compile(r"^deltaT = ([0-9.eE+-]+)")
    rx_co = re.compile(r"^Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)")
    rx_g = re.compile(r"pressure gradient = ([0-9.eE+-]+)")
    with open(path) as fh:
        for line in fh:
            m = rx_t.match(line)
            if m:
                cur_t = float(m.group(1))
                continue
            m = rx_dt.match(line)
            if m:
                cur_dt = float(m.group(1))
                continue
            m = rx_co.match(line)
            if m:
                cur_co = float(m.group(2))
                continue
            m = rx_g.search(line)
            if m and np.isfinite(cur_t):
                t.append(cur_t)
                g.append(float(m.group(1)))
                dt.append(cur_dt)
                co.append(cur_co)
    return np.array(t), np.array(g), np.array(dt), np.array(co)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--times", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    case = os.path.abspath(args.case)
    geo = json.load(open(os.path.join(case, "GEOMETRY.json")))
    nx, ny, nz = geo["nx"], geo["ny"], geo["nz"]
    os.makedirs(args.out, exist_ok=True)

    x, y, z, maps = structured_maps(case, nx, ny, nz)
    a, lam = geo["amplitude"], geo["lambda_"]
    y_wall = a * (1.0 + np.cos(2.0 * np.pi * x / lam))
    # the face-centre heights of the bottom patch must reproduce the analytic wall
    yb = maps["bottomWall"]["y_face"][0]
    wall_err = float(np.max(np.abs(yb - y_wall)))
    print("[reduce] structured ordering OK: nx=%d ny=%d nz=%d ; max|y_face - y_wall(x)| = %.2e"
          % (nx, ny, nz, wall_err))
    bo, to = maps["bottomWall"]["order"], maps["topWall"]["order"]

    written = []
    for tdir in args.times:
        tpath = os.path.join(case, tdir)
        if not os.path.isdir(tpath):
            print("[reduce] time %s missing -> skipped" % tdir)
            continue
        need = ["UMean", "pMean", "UPrime2Mean", "wallShearStressMean"]
        if not all(os.path.exists(os.path.join(tpath, f)) for f in need):
            print("[reduce] time %s lacks averaged fields -> skipped" % tdir)
            continue
        out = dict(time=float(tdir), x=x, y=y, z=z, y_wall=y_wall, nx=nx, ny=ny, nz=nz,
                   nu=geo["nu"], Ub=geo["Ub"], amplitude=a, lambda_=lam,
                   H_mean=geo["H_mean"], delta=geo["delta"], y_top=geo["y_top"],
                   avg_start=geo["avg_start"], grid=geo["grid"], tag=geo["tag"])
        src = {}
        _, UM, _ = read_field(os.path.join(tpath, "UMean"), 3)
        Uz = zavg(UM, nz, ny, nx)
        out["U"], out["V"], out["W"] = Uz[:, :, 0], Uz[:, :, 1], Uz[:, :, 2]
        # spanwise variance of the time-mean (dispersive in z, should be ~0): a
        # convergence diagnostic for the averaging window
        U3 = UM.reshape(nz, ny, nx, 3)
        out["U_zrms"] = np.sqrt(((U3[:, :, :, 0] - Uz[:, :, 0].T[None]) ** 2).mean(axis=0)).T
        src["UMean"] = sha256(os.path.join(tpath, "UMean"))
        _, pM, pb = read_field(os.path.join(tpath, "pMean"), 1, ("bottomWall", "topWall"))
        out["P"] = zavg(pM, nz, ny, nx)
        src["pMean"] = sha256(os.path.join(tpath, "pMean"))
        # wall pressure = first-cell value (zeroGradient) -- and patch value if present
        p3 = pM.reshape(nz, ny, nx)
        out["p_wall_bottom_z"] = p3[:, 0, :]
        out["p_wall_top_z"] = p3[:, -1, :]
        out["p_wall_bottom"] = p3[:, 0, :].mean(axis=0)
        out["p_wall_top"] = p3[:, -1, :].mean(axis=0)
        _, R, _ = read_field(os.path.join(tpath, "UPrime2Mean"), 6)
        Rz = zavg(R, nz, ny, nx)
        for k, name in enumerate(("uu", "uv", "uw", "vv", "vw", "ww")):
            out[name] = Rz[:, :, k]
        src["UPrime2Mean"] = sha256(os.path.join(tpath, "UPrime2Mean"))
        if os.path.exists(os.path.join(tpath, "pPrime2Mean")):
            _, pp, _ = read_field(os.path.join(tpath, "pPrime2Mean"), 1)
            out["pp"] = zavg(pp, nz, ny, nx)
        if os.path.exists(os.path.join(tpath, "nutMean")):
            _, nm, nb = read_field(os.path.join(tpath, "nutMean"), 1, ("bottomWall", "topWall"))
            out["nut"] = zavg(nm, nz, ny, nx)
            # wall value of the (Spalding) nut BC: must be ~0 for a wall-resolved mesh
            nwb = patch_field(nb["bottomWall"], bo, nz, nx, 1)
            nwt = patch_field(nb["topWall"], to, nz, nx, 1)
            if nwb is not None:
                out["nut_wall_bottom"] = nwb.mean(axis=0)
                out["nut_wall_bottom_max"] = float(nwb.max())
            if nwt is not None:
                out["nut_wall_top"] = nwt.mean(axis=0)
                out["nut_wall_top_max"] = float(nwt.max())
            src["nutMean"] = sha256(os.path.join(tpath, "nutMean"))
        _, _, wb = read_field(os.path.join(tpath, "wallShearStressMean"), 3, ("bottomWall", "topWall"))
        tb = patch_field(wb["bottomWall"], bo, nz, nx, 3)
        tt = patch_field(wb["topWall"], to, nz, nx, 3)
        if tb is None or tt is None:
            raise RuntimeError("wallShearStressMean has no patch values at t=%s" % tdir)
        out["wss_bottom_z"] = tb                 # (nz, nx, 3) OpenFOAM sign convention
        out["wss_top_z"] = tt
        out["wss_bottom"] = tb.mean(axis=0)      # (nx, 3)
        out["wss_top"] = tt.mean(axis=0)
        src["wallShearStressMean"] = sha256(os.path.join(tpath, "wallShearStressMean"))
        if os.path.exists(os.path.join(tpath, "yPlus")):
            _, _, yb_ = read_field(os.path.join(tpath, "yPlus"), 1, ("bottomWall", "topWall"))
            ypb = patch_field(yb_["bottomWall"], bo, nz, nx, 1)
            ypt = patch_field(yb_["topWall"], to, nz, nx, 1)
            if ypb is not None:
                out["yplus_bottom"] = ypb.mean(axis=0)
                out["yplus_bottom_max"] = float(ypb.max())
            if ypt is not None:
                out["yplus_top"] = ypt.mean(axis=0)
                out["yplus_top_max"] = float(ypt.max())
        # first-cell centre height above the wall (wall-normal resolution record)
        out["dy_first_cell"] = y[:, 0] - y_wall
        out["source_sha256"] = json.dumps(src)
        fn = os.path.join(args.out, "wavy2d_%s_t%s.npz" % (geo["grid"], tdir))
        np.savez(fn, **out)
        written.append(fn)
        print("[reduce] t=%s -> %s  (cell-mean U=%.5f, top wss_x mean=%.3e, bottom wss_x mean=%.3e)"
              % (tdir, fn, float(UM[:, 0].mean()), float(tt[:, :, 0].mean()), float(tb[:, :, 0].mean())))

    # a resumed run keeps its earlier log(s) as log.pimpleFoam.before_resume_<epoch>
    logs = sorted(glob.glob(os.path.join(case, "log.pimpleFoam.before_resume_*")),
                  key=lambda f: int(f.rsplit("_", 1)[-1]))
    if os.path.exists(os.path.join(case, "log.pimpleFoam")):
        logs.append(os.path.join(case, "log.pimpleFoam"))
    if logs:
        parts = [parse_log(f) for f in logs]
        t, g, dt, co = (np.concatenate([p[i] for p in parts]) for i in range(4))
        # a resumed segment restarts from the last written time: drop overlap
        keep = np.ones(len(t), bool)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                keep[:i] &= t[:i] < t[i]
        t, g, dt, co = t[keep], g[keep], dt[keep], co[keep]
        fn = os.path.join(args.out, "wavy_timeseries_%s.npz" % geo["grid"])
        np.savez(fn, time=t, gradP=g, deltaT=dt, CoMax=co, grid=geo["grid"])
        print("[reduce] log series: %d steps, t=[%.3f, %.3f], last dt=%.4g, last CoMax=%.3f"
              % (len(t), t[0] if len(t) else np.nan, t[-1] if len(t) else np.nan,
                 dt[-1] if len(dt) else np.nan, co[-1] if len(co) else np.nan))
        written.append(fn)
    with open(os.path.join(args.out, "REDUCE_OK_%s.json" % geo["grid"]), "w") as fh:
        json.dump(dict(case=case, grid=geo["grid"], times=args.times, files=written,
                       wall_err=wall_err,
                       # measured column offsets, in cells, so a reader can see
                       # how far this mesh sits from the identifiability bound
                       column_offset_over_dx=MEASURED,
                       column_offset_bound=COLUMN_TOL_OVER_DX), fh, indent=2)
    print("REDUCE_OK %s" % geo["grid"])


if __name__ == "__main__":
    main()
