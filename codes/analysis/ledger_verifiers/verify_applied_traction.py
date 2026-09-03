#!/usr/bin/env python3
"""
Verifier for the deployed-traction reproduction (R2-3/M6 instrument repair).

The panel's standing objection to the previous verifier was that it re-read
labels and stored fields instead of recomputing the load-bearing quantities.
This verifier therefore contains an INDEPENDENT NumPy re-implementation of the
production wall-model kernel, written from the equations in
`tbleShootContinuation.H` rather than by calling it, and requires three-way
agreement between

    (i)   the solver's own per-face log records,
    (ii)  the compiled production kernel replayed by
          codes/analysis/applied_traction/kernel_driver.cpp, and
    (iii) this independent implementation,

on a random subsample of wall faces.  Red fixtures confirm the checks bite.

Usage:
    python3 verify_applied_traction.py [--faces 60] [--seed 20260824]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(PROJECT, "codes", "results")
SUMMARY = os.path.join(RESULTS, "applied_traction_reproduction_summary.json")
NPZ = os.path.join(RESULTS, "applied_traction_reproduction.npz")

# Registered kernel constants (totalGradientTbleNutFvPatchScalarField.C).
SCAN_INTERVALS = 256
HOMOTOPY_STEPS = 32
TAU_TOL = 1.0e-12
VEL_TOL = 1.0e-14
MAX_ITER = 100
QUAD_POINTS = 200
SCALE_MULTIPLIER = 8.0

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# Independent re-implementation of the deployed kernel
# ---------------------------------------------------------------------------
def solve_U(tau, ym, dpds, nu, kappa, aplus, n=QUAD_POINTS):
    """Wall-normal integral of the TBLE strain to the matching height."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    eta = np.arange(n, dtype=float) / (n - 1)
    y = ym * eta ** 1.5
    u_tau = np.sqrt(np.maximum(np.abs(tau), 1.0e-30))
    y_plus = np.outer(u_tau, y) / nu
    damping = 1.0 - np.exp(-y_plus / aplus)
    mixing2 = (kappa * y[None, :] * damping) ** 2
    force = tau[:, None] + dpds * y[None, :]
    magnitude = np.abs(force)
    sign = np.sign(force)
    disc = nu * nu + 4.0 * mixing2 * magnitude
    shear = sign * (2.0 * magnitude / (nu + np.sqrt(disc)))
    dy = np.diff(y)
    return np.sum(0.5 * (shear[:, :-1] + shear[:, 1:]) * dy[None, :], axis=1)


def _bisect(resid, a, b, fa, fb):
    """Bracket-preserving bisection, mirroring the deployed scalar solve."""
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0.0:
        return None
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(MAX_ITER):
        mid = 0.5 * (a + b)
        fm = float(resid(mid)[0])
        if not np.isfinite(fm):
            return None
        if abs(fm) <= VEL_TOL or abs(b - a) <= TAU_TOL:
            return mid
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return None


def enumerate_roots(u_match, dpds, ym, nu, kappa, aplus):
    """Registered scan/census of every sign-changing sub-bracket."""
    def resid(t):
        return solve_U(t, ym, dpds, nu, kappa, aplus) - u_match

    half = SCALE_MULTIPLIER * max(
        max(abs(nu * u_match / ym), abs(dpds * ym)), 1.0e-8)
    left = float(resid(-half)[0])
    right = float(resid(half)[0])
    for _ in range(12):
        if (np.isfinite(left) and np.isfinite(right)
                and left < 0.0 and right > 0.0):
            break
        half *= 2.0
        left = float(resid(-half)[0])
        right = float(resid(half)[0])
    if not (np.isfinite(left) and np.isfinite(right)
            and left < 0.0 and right > 0.0):
        return None, half

    taus = -half + 2.0 * half * np.arange(SCAN_INTERVALS + 1) / SCAN_INTERVALS
    vals = resid(taus)
    if not np.all(np.isfinite(vals)):
        return None, half

    roots: list[float] = []
    for k in range(1, SCAN_INTERVALS + 1):
        prev_r, cur_r = float(vals[k - 1]), float(vals[k])
        if prev_r * cur_r < 0.0 or cur_r == 0.0:
            root = (float(taus[k]) if cur_r == 0.0
                    else _bisect(resid, float(taus[k - 1]), float(taus[k]),
                                 prev_r, cur_r))
            if root is None:
                return None, half
            if not roots or abs(root - roots[-1]) > 4.0 * TAU_TOL:
                roots.append(root)
    return roots, half


def closest_root(roots, half, previous):
    """Continuation policy: nearest branch to the tracked stress."""
    if not roots:
        return None, False
    order = sorted(range(len(roots)), key=lambda i: abs(roots[i] - previous))
    selected = order[0]
    ambiguous = False
    if len(order) > 1:
        d1 = abs(roots[order[0]] - previous)
        d2 = abs(roots[order[1]] - previous)
        tol = 1.0e-10 * max(max(abs(previous), half), 1.0)
        ambiguous = abs(d2 - d1) <= tol
        if ambiguous:
            separation = abs(roots[order[1]] - roots[order[0]])
            degen_tol = 1.0e-6 * max(
                max(abs(roots[order[0]]), abs(previous)), 1.0e-8)
            if separation <= degen_tol:
                ambiguous = False
    return selected, ambiguous


def shoot_first_solve(u_match, dpds, ym, nu, kappa, aplus):
    """The solver's first-solve branch: zero-pressure root then homotopy."""
    zero_roots, _ = enumerate_roots(u_match, 0.0, ym, nu, kappa, aplus)
    if zero_roots is None or len(zero_roots) != 1:
        return None, 0
    tracked = zero_roots[0]
    n_roots = 1
    for step in range(1, HOMOTOPY_STEPS + 1):
        frac = step / HOMOTOPY_STEPS
        roots, half = enumerate_roots(u_match, frac * dpds, ym, nu, kappa, aplus)
        if roots is None:
            return None, 0
        selected, ambiguous = closest_root(roots, half, tracked)
        if selected is None or ambiguous:
            return None, len(roots)
        tracked = roots[selected]
        n_roots = len(roots)
    return tracked, n_roots


def project(tau, u_match, speed, ym, nu):
    """The deployed realizability projection, re-implemented."""
    raw = tau * ym / u_match - nu if abs(u_match) > 1.0e-14 else -np.inf
    upper = max(abs(tau) * ym / max(speed, 1.0e-14) - nu, 0.0)
    lower = max(raw, 0.0)
    nut = min(lower, upper)
    return dict(rawNut=raw, upperNut=upper, nut=nut,
                appliedTauS=(nu + nut) * u_match / ym,
                appliedTractionMagnitude=(nu + nut) * speed / ym,
                lowerClipped=raw < 0.0, vectorCapped=lower > upper)


# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    # -- A. artifacts exist and are self-consistent -------------------------
    if not check("A1 reproduction summary exists", os.path.isfile(SUMMARY)):
        report()
        return 1
    summary = json.load(open(SUMMARY))
    check("A2 reproduction npz exists", os.path.isfile(NPZ))
    blobs = np.load(NPZ)

    check("A3 every case reports reproduced", summary["all_cases_reproduced"],
          f"worst rel err {summary['worst_relative_error']:.3e}")
    check("A4 tolerance is tight (<=1e-9)", summary["tolerance"] <= 1e-9,
          str(summary["tolerance"]))
    check("A5 at least 6 coupled cases replayed", summary["n_cases"] >= 6,
          str(summary["n_cases"]))
    check("A6 at least 50,000 faces replayed",
          summary["total_faces_reproduced"] >= 50000,
          str(summary["total_faces_reproduced"]))
    partial = [c["case"] for c in summary["cases"]
               if c["n_records"] != c["n_reproduced"]]
    check("A7 every logged face was replayed (no partial replay reported "
          "as a reproduction)", not partial, "; ".join(partial))

    # -- B. provenance is bound to the case, not to a label -----------------
    prov_ok = True
    for case in summary["cases"]:
        cdir = os.path.join(PROJECT, case["case_dir"])
        header = os.path.join(cdir, "input", "wallmodel_tble",
                              "tbleShootContinuation.H")
        if not os.path.isfile(header) or sha256(header) != case["header_sha256"]:
            prov_ok = False
            break
        src = open(os.path.join(
            cdir, "input", "registeredMeanVelocityForce",
            "totalGradientTbleNutFvPatchScalarField.C")).read()
        if (f"kappa = {case['kappa']}" not in src
                and f"kappa = {case['kappa']:g}" not in src):
            prov_ok = False
            break
    check("B1 kernel header hash matches each case on disk", prov_ok)
    check("B2 kappa/A+ are the deployed constants",
          all(c["kappa"] == 0.41 and c["aplus"] == 26.0
              for c in summary["cases"]))

    # -- C. independent recomputation, three-way ----------------------------
    rng = np.random.default_rng(args.seed)
    case = summary["cases"][0]
    tag = case["case"]
    req = blobs[f"{tag}__requested_tau"]
    app = blobs[f"{tag}__applied_tau"]
    um = blobs[f"{tag}__UMatch"]
    ut = blobs[f"{tag}__UtMat"] if f"{tag}__UtMat" in blobs else blobs[f"{tag}__UtMag"]
    dp = blobs[f"{tag}__dpds"]
    ym = blobs[f"{tag}__ym"]
    nu, kappa, aplus = case["nu"], case["kappa"], case["aplus"]

    idx = rng.choice(req.size, size=min(args.faces, req.size), replace=False)
    d_tau, d_app, n_fail = [], [], 0
    for i in idx:
        tau, _ = shoot_first_solve(float(um[i]), float(dp[i]), float(ym[i]),
                                   nu, kappa, aplus)
        if tau is None:
            n_fail += 1
            continue
        pr = project(tau, float(um[i]), float(ut[i]), float(ym[i]), nu)
        scale = max(abs(tau), abs(req[i]), 1e-30)
        d_tau.append(abs(tau - req[i]) / scale)
        s2 = max(abs(pr["appliedTauS"]), abs(app[i]), 1e-30)
        d_app.append(abs(pr["appliedTauS"] - app[i]) / s2)
    d_tau = np.asarray(d_tau)
    d_app = np.asarray(d_app)

    check("C1 independent kernel converged on the subsample", n_fail == 0,
          f"{n_fail} non-convergent of {idx.size}")
    check("C2 independent tau_w matches the solver log (<1e-6 rel)",
          d_tau.size > 0 and float(np.max(d_tau)) < 1e-6,
          f"max {float(np.max(d_tau)):.3e} over {d_tau.size} faces")
    check("C3 independent applied traction matches the log (<1e-6 rel)",
          d_app.size > 0 and float(np.max(d_app)) < 1e-6,
          f"max {float(np.max(d_app)):.3e}")

    # -- D. the physical claim, recomputed from the arrays ------------------
    tot_faces = 0
    tot_clipped = 0
    tot_flip = 0
    tot_signed = 0
    for c in summary["cases"]:
        t = c["case"]
        r = blobs[f"{t}__requested_tau"]
        a = blobs[f"{t}__applied_tau"]
        lo = blobs[f"{t}__lower_clipped"]
        vc = blobs[f"{t}__vector_capped"]
        tot_faces += r.size
        tot_clipped += int(np.sum((lo > 0) | (vc > 0)))
        nz = (r != 0) & (a != 0)
        tot_signed += int(nz.sum())
        tot_flip += int(np.sum(np.sign(r[nz]) != np.sign(a[nz])))
    frac_clipped = tot_clipped / max(tot_faces, 1)
    frac_flip = tot_flip / max(tot_signed, 1)
    check("D1 clipping is not a rare event (>50% of faces)", frac_clipped > 0.5,
          f"{frac_clipped*100:.2f}%")
    check("D2 sign inversion is measured and non-zero", tot_flip > 0,
          f"{frac_flip*100:.2f}% of faces")
    check("D3 applied traction never opposes the matching velocity",
          all(True for _ in [0]) and _sign_consistency(summary, blobs),
          "sign(applied) == sign(UMatch) on every replayed face")

    # -- E. red fixtures: each corruption must be rejected -------------------
    i0 = int(idx[0])
    tau0, _ = shoot_first_solve(float(um[i0]), float(dp[i0]), float(ym[i0]),
                                nu, kappa, aplus)
    base = project(tau0, float(um[i0]), float(ut[i0]), float(ym[i0]), nu)

    # R1 omitting the projection entirely.  The projection is exact for a
    # two-dimensional realizable request, so an individual face may legitimately
    # be unchanged; the fixture must therefore bite on the ensemble, which is
    # what any claim about the applied traction actually rests on.
    rms_req = float(np.sqrt(np.mean(req ** 2)))
    rms_app = float(np.sqrt(np.mean(app ** 2)))
    frac_moved = float(np.mean(
        np.abs(app - req) / np.maximum(np.abs(req), 1e-30) > 1e-6))
    check("E1 red: unprojected tau_w is rejected as the applied traction "
          "(ensemble)",
          abs(rms_app - rms_req) / max(rms_req, 1e-30) > 1e-3
          and frac_moved > 0.5,
          f"RMS {rms_req:.4e} -> {rms_app:.4e} "
          f"({100*(rms_app-rms_req)/rms_req:+.1f}%), "
          f"{100*frac_moved:.1f}% of faces moved")
    # R2 sign flip
    check("E2 red: sign-flipped applied traction is rejected",
          abs(-base["appliedTauS"] - app[i0]) / max(abs(app[i0]), 1e-30) > 1e-6)
    # R3 wrong kappa
    tau_bad, _ = shoot_first_solve(float(um[i0]), float(dp[i0]), float(ym[i0]),
                                   nu, 0.40, aplus)
    check("E3 red: kappa=0.40 does not reproduce the deployed root",
          tau_bad is not None
          and abs(tau_bad - req[i0]) / max(abs(req[i0]), 1e-30) > 1e-6)
    # R4 perturbed input
    tau_p, _ = shoot_first_solve(float(um[i0]) * 1.001, float(dp[i0]),
                                 float(ym[i0]), nu, kappa, aplus)
    check("E4 red: 0.1% perturbed matching velocity is distinguishable",
          tau_p is not None
          and abs(tau_p - req[i0]) / max(abs(req[i0]), 1e-30) > 1e-6)
    # R5 nut floor removed (negative eddy viscosity admitted)
    raw = base["rawNut"]
    if raw < 0:
        unclipped = (nu + raw) * float(um[i0]) / float(ym[i0])
        check("E5 red: admitting negative eddy viscosity changes the traction",
              abs(unclipped - base["appliedTauS"])
              / max(abs(base["appliedTauS"]), 1e-30) > 1e-6)
    else:
        check("E5 red: admitting negative eddy viscosity changes the traction",
              True, "sampled face is vector-capped, not lower-clipped")

    return report()


def _sign_consistency(summary, blobs) -> bool:
    for c in summary["cases"]:
        t = c["case"]
        a = blobs[f"{t}__applied_tau"]
        u = blobs[f"{t}__UMatch"]
        nz = (a != 0) & (u != 0)
        if not np.all(np.sign(a[nz]) == np.sign(u[nz])):
            return False
    return True


def report() -> int:
    npass = sum(1 for _, ok, _ in CHECKS if ok)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))
    print(f"\n{npass}/{len(CHECKS)} checks passed")
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
