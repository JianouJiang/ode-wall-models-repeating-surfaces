#!/usr/bin/env python3
"""Agent V addendum 2b -- the 29-case Xiao family epsilon sweep, recomputed with a
corrected wall-traction estimator.

The deposited sweep (codes/analysis/dose_response_xiao.py) builds tau_w for EVERY one of
the 29 members with the withdrawn estimator: `nu * du/dy` from a 4-point through-origin
LINEAR fit of the STREAMWISE u against the VERTICAL offset, with no tangent correction.
That single quantity feeds
    eps~ (numerator), R^2(tau_w) (truth), rel_err (truth),
    and f_sep / L_sep / f_rec (through the SIGN of tau_w),
i.e. the whole causal chain of the amplitude-pitch section, not just the printed range.

Here every member is recomputed with:
  legacy   -- the deposited estimator, to reproduce the published numbers
  repaired -- through-origin CUBIC on the first 6 fluid points of the tangential velocity,
              divided by t_x, with t_x from a Fourier-filtered derivative of the extracted
              wall height (the family is parameterised, so no analytic hill exists for 28
              of the 29 members)
  mglet    -- ONLY for alph10-9-3036, the one member whose geometry (alpha=1, Lx=9H,
              Ly=3.036H) is the standard ERCOFTAC hill and therefore has an independent
              published DNS wall traction.  The other 28 members have NO independent
              wall-traction reference in the literature; for them the repaired estimator
              on Xiao's own field is the only option and that limitation is reported.

Out: work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/xiao29_epsilon_sweep.json
Local analysis only; no cluster jobs; read-only on all inputs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/xiao29_epsilon_sweep.json"
FAM = ROOT / "codes/raw_data/geometry_driven/xiao_pehill_parameterized/pehill-29-cases-DNS"
MGLET = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
sys.path.insert(0, str(ROOT / "codes/vendor/universal_wall_function/codes/analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

NU, Y_IDX, VEL_TOL = 1.0 / 5600.0, 10, 1.0e-6
FIT_DEG, FIT_K = 3, 6
MGLET_MEMBER = "alph10-9-3036"
DRAWS, SEED = 20000, 20260825


def member_geometry(name: str):
    m = re.match(r"alph(\d+)-(\d+)-(\d+)$", name)
    a, lx, ly = m.groups()
    alpha = float(a[0] + "." + a[1:])
    return alpha, float(lx[0] + "." + lx[1:]), float(ly[0] + "." + ly[1:])


def fourier_slope(h, x, keep=24):
    """Periodic derivative of the extracted wall height with a low-pass filter.
    The wall is staircased on the Cartesian grid, so a raw difference is noise."""
    n = h.size
    L = (x[-1] - x[0]) + (x[1] - x[0])
    k = np.fft.rfftfreq(n, d=L / n) * 2.0 * np.pi
    H = np.fft.rfft(h)
    H[keep + 1:] = 0.0
    return np.fft.irfft(1j * k * H, n=n)


def poly_origin_slope(nn, u, deg):
    A = np.vstack([np.asarray(nn, float) ** (j + 1) for j in range(deg)]).T
    return float(np.linalg.lstsq(A, np.asarray(u, float), rcond=None)[0][0])


def wrap_interp(xp, yp, t):
    o = np.argsort(np.mod(np.asarray(xp, float), 1.0))
    a = np.mod(np.asarray(xp, float), 1.0)[o]
    b = np.asarray(yp, float)[o]
    return np.interp(np.mod(np.asarray(t, float), 1.0), np.r_[a-1, a, a+1], np.r_[b, b, b])


def block_median_ci(v, draws=DRAWS, seed=SEED):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    block = max(2, n // 8)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    med = np.empty(draws)
    off = np.arange(block)
    for a in range(0, draws, 1000):
        b = min(a + 1000, draws)
        st = rng.integers(0, n, size=(b - a, nb))
        idx = ((st[:, :, None] + off[None, None, :]) % n).reshape(b - a, -1)[:, :n]
        med[a:b] = np.median(v[idx], axis=1)
    return [float(np.quantile(med, 0.05)), float(np.quantile(med, 0.95))]


def largest_negative_span(x, neg):
    best, i = 0.0, 0
    n = len(neg)
    while i < n:
        if not neg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and neg[j + 1]:
            j += 1
        best = max(best, x[j] - x[i])
        i = j + 1
    return best


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def read_member(d: Path):
    tab = pd.read_csv(d / "mean_files.dat", sep=r"\s+", header=None).values
    x, y, u, v, p = tab[:, 0], tab[:, 1], tab[:, 2], tab[:, 3], tab[:, 5]
    xu = np.unique(np.round(x, 6))
    ell_p = float(x.max() - x.min())
    L_y = float(y.max() - y.min())
    order = np.lexsort((y, np.round(x, 6)))
    x, y, u, v, p = x[order], y[order], u[order], v[order], p[order]
    ny = np.count_nonzero(np.abs(x - xu[0]) < 1e-6)
    Y = y.reshape(-1, ny); U = u.reshape(-1, ny); V = v.reshape(-1, ny); P = p.reshape(-1, ny)
    xs, hw, offs, Uc, Vc, pw = [], [], [], [], [], []
    for i in range(len(xu)):
        yy, uu, vv, pp = Y[i], U[i], V[i], P[i]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < Y_IDX + 2:
            continue
        k = max(fluid[0], 1)
        xs.append(xu[i]); hw.append(yy[k - 1])
        offs.append(yy[k - 1:] - yy[k - 1]); Uc.append(uu[k - 1:]); Vc.append(vv[k - 1:])
        pw.append(pp[k - 1:][1])
    return (np.asarray(xs), np.asarray(hw), offs, Uc, Vc, np.asarray(pw), ell_p, L_y)


def evaluate(name: str, mglet_tau=None):
    xs, hw, offs, Uc, Vc, pw, ell_p, L_y = read_member(FAM / name)
    slope = fourier_slope(hw, xs)
    tx = 1.0 / np.sqrt(1.0 + slope ** 2); ty = slope * tx
    n = xs.size
    tau_leg = np.empty(n); tau_rep = np.empty(n); pred = np.full(n, np.nan)
    y_m = np.empty(n)
    for i in range(n):
        nn, uu, vv = offs[i], Uc[i], Vc[i]
        tau_leg[i] = NU * float(np.sum(nn[1:5] * uu[1:5]) / np.sum(nn[1:5] ** 2))
        ut = uu[1:FIT_K + 1] * tx[i] + vv[1:FIT_K + 1] * ty[i]
        tau_rep[i] = NU * poly_origin_slope(nn[1:FIT_K + 1], ut, FIT_DEG) / tx[i]
        y_m[i] = nn[Y_IDX]
    dp_dx = np.gradient(pw, xs)
    for i in range(n):
        if y_m[i] > 0 and np.isfinite(Uc[i][Y_IDX]):
            pred[i] = predict_tau_w(Uc[i][Y_IDX], y_m[i], dp_dx[i], NU)
    den = np.abs(dp_dx) * y_m
    ok = den > 1e-30

    out = {"member": name, "alpha": member_geometry(name)[0],
           "ell_p": ell_p, "L_y": L_y, "n_stations": int(n),
           "median_dy_first": float(np.median([o[1] for o in offs])),
           "has_independent_reference": name == MGLET_MEMBER}
    truths = {"legacy": tau_leg, "repaired": tau_rep}
    if mglet_tau is not None:
        truths["mglet"] = mglet_tau
    for tag, t in truths.items():
        eps = np.full(n, np.nan)
        eps[ok] = np.abs(t[ok]) / den[ok]
        fin = np.isfinite(eps) & (eps > 0)
        v = pred[np.isfinite(pred)]; tt = t[np.isfinite(pred)]
        ss_res = float(np.sum((tt - v) ** 2)); ss_tot = float(np.sum((tt - tt.mean()) ** 2))
        neg = t < 0
        out[tag] = {
            "eps_median": float(np.nanmedian(eps[fin])),
            "eps_median_ci": block_median_ci(eps[fin]),
            "frac_eps_lt_0p1": float(np.mean(eps[fin] < 0.1)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rel_err": float(np.sqrt(ss_res) / (np.sqrt(np.sum(tt ** 2)) + 1e-30)),
            "f_sep": float(neg.mean()),
            "L_sep": float(largest_negative_span(xs, neg)),
            "median_abs_tau": float(np.median(np.abs(t))),
        }
        out[tag]["f_rec"] = out[tag]["L_sep"] / ell_p if ell_p > 0 else float("nan")
    out["repaired_over_legacy_eps"] = out["repaired"]["eps_median"] / out["legacy"]["eps_median"]
    return out


def main() -> int:
    members = sorted(p.name for p in FAM.iterdir() if p.is_dir() and p.name.startswith("alph"))
    mg = np.loadtxt(MGLET)[:-2]
    rows = []
    for name in members:
        mtau = None
        if name == MGLET_MEMBER:
            xs = read_member(FAM / name)[0]
            mtau = wrap_interp(mg[:, 0] / 9.0, mg[:, 1], np.mod((xs - xs.min()) / 9.0, 1.0))
        r = evaluate(name, mtau)
        rows.append(r)
        print(f"{name:20s} a={r['alpha']:.2f} dy={r['median_dy_first']:.5f} "
              f"eps~ {r['legacy']['eps_median']:.4f} -> {r['repaired']['eps_median']:.4f} "
              f"(x{r['repaired_over_legacy_eps']:.2f})  R2 {r['legacy']['r2']:+8.2f} -> {r['repaired']['r2']:+8.2f}"
              + (f"   [MGLET eps~ {r['mglet']['eps_median']:.4f} R2 {r['mglet']['r2']:+.2f}]" if mtau is not None else ""))

    def col(tag, key):
        return np.array([r[tag][key] for r in rows])
    alpha = np.array([r["alpha"] for r in rows])
    delta = np.array([0.5 * r["L_y"] for r in rows])
    res = {"schema": "xiao29-epsilon-sweep-v1", "agent": "V",
           "n_members": len(rows),
           "members_with_independent_wall_traction_reference": [MGLET_MEMBER],
           "members_without_any_independent_reference": [r["member"] for r in rows
                                                         if not r["has_independent_reference"]],
           "reference_limitation": (
               "28 of 29 members are parameterised geometries (alpha != 1 and/or Lx != 9H and/or "
               "Ly != 3.036H) for which no independent published DNS wall traction exists.  For "
               "them the repaired estimator applied to Xiao's own field is the ONLY option; it is "
               "validated at exactly one point, the alpha=1/Lx=9/Ly=3.036 member, against MGLET."),
           "per_member": rows}
    for tag in ("legacy", "repaired"):
        e = col(tag, "eps_median")
        res[f"{tag}_eps_range"] = [float(e.min()), float(e.max())]
        res[f"{tag}_eps_range_members"] = [rows[int(np.argmin(e))]["member"], rows[int(np.argmax(e))]["member"]]
        res[f"{tag}_eps_range_ci"] = [rows[int(np.argmin(e))][tag]["eps_median_ci"],
                                      rows[int(np.argmax(e))][tag]["eps_median_ci"]]
        res[f"{tag}_all_members_fail_r2_below_0"] = bool(np.all(col(tag, "r2") < 0))
        res[f"{tag}_max_r2"] = float(np.max(col(tag, "r2")))
        Ls = col(tag, "L_sep") / delta
        res[f"{tag}_spearman"] = {
            "rho(L_sep_over_delta, eps_median)": spearman(Ls, e),
            "rho(L_sep_over_delta, r2)": spearman(Ls, col(tag, "r2")),
            "rho(alpha, eps_median)": spearman(alpha, e),
            "rho(alpha, r2)": spearman(alpha, col(tag, "r2")),
            "rho(eps_median, r2)": spearman(e, col(tag, "r2")),
            "rho(eps_median, rel_err)": spearman(e, col(tag, "rel_err")),
        }
    res["contamination_correlates_with"] = {
        "rho(alpha, repaired_over_legacy)": spearman(alpha, np.array([r["repaired_over_legacy_eps"] for r in rows])),
        "rho(median_dy_first, repaired_over_legacy)": spearman(
            np.array([r["median_dy_first"] for r in rows]),
            np.array([r["repaired_over_legacy_eps"] for r in rows])),
        "rho(L_sep_over_delta_legacy, repaired_over_legacy)": spearman(
            col("legacy", "L_sep") / delta, np.array([r["repaired_over_legacy_eps"] for r in rows])),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nwritten -> {OUT.relative_to(ROOT)}")
    print("legacy  eps~ range", [round(v, 4) for v in res["legacy_eps_range"]], res["legacy_eps_range_members"])
    print("repaired eps~ range", [round(v, 4) for v in res["repaired_eps_range"]], res["repaired_eps_range_members"])
    for tag in ("legacy", "repaired"):
        print(f"{tag:9s} all R2<0: {res[f'{tag}_all_members_fail_r2_below_0']}  max R2 {res[f'{tag}_max_r2']:+.3f}")
        for k, v in res[f"{tag}_spearman"].items():
            print(f"    {k:42s} {v:+.3f}")
    print("contamination correlates:", {k: round(v, 3) for k, v in res["contamination_correlates_with"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
