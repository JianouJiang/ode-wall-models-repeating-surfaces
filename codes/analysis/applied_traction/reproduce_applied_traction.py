#!/usr/bin/env python3
"""
Independent reproduction of the DEPLOYED wall-model traction, face by face.

Motivation
----------
Every a-priori score published for this paper so far has been computed on the
TBLE shooting target tau_w -- the stress the ODE *requests*.  The coupled
solver does not apply that stress.  It converts the request into a scalar eddy
viscosity through `tbleVectorRealizableNut`, which is clipped at zero from
below and capped from above so that the complete tangential traction stays
bounded by |tau_w|.  The solver's own logs record `clipped=<all faces>` on the
periodic hill, so the traction the large-eddy simulation actually feels is
never the requested one.  This script measures the difference using the
production kernel itself.

Method
------
For each coupled TBLE case:
  1. parse the per-face `TOTAL_GRADIENT_TBLE_FACE` records the solver wrote,
  2. compile `kernel_driver.cpp` against THAT case's own copy of
     `wallmodel_tble/tbleShootContinuation.H` (sha256 recorded),
  3. replay every logged face through the kernel and the projection,
  4. compare the reproduced state with the logged state face by face, and
  5. report requested-vs-applied traction statistics.

Only faces the solver solved with `homotopySteps=33` (its first-solve branch,
`hasPrevious=false`) are replayed, because only those are reproducible without
the solver's per-face continuation state.  The script asserts that this
selection covers the logged records rather than assuming it.

Outputs
-------
codes/results/applied_traction_reproduction.npz
codes/results/applied_traction_reproduction_summary.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(PROJECT, "codes", "results")

# Constants hard-coded in the deployed boundary condition
# (totalGradientTbleNutFvPatchScalarField.C:135-136).  Read back from the
# source at run time so a silent divergence cannot go unnoticed.
KAPPA_EXPECTED = 0.41
APLUS_EXPECTED = 26.0

FACE_TAG = "TOTAL_GRADIENT_TBLE_FACE"

_SCALAR_KEYS = (
    "UMatch", "UtMag", "ym", "phaseDpds", "driveGradient", "driveProjection",
    "dpds", "tauW", "rawNut", "upperNut", "nut", "appliedTau",
    "appliedTractionMag", "lowerClipped", "vectorCapped", "clipped",
    "roots", "selected", "homotopySteps", "degenerate",
)
_KV = re.compile(r"(\w+)=(-?[0-9.eE+-]+|\S+)")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_face_records(log_path: str) -> dict:
    """Extract every per-face wall-model record from a solver log."""
    rows = {k: [] for k in _SCALAR_KEYS}
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if FACE_TAG not in line:
                continue
            # `centre=(x y z)` contains spaces; drop it before key/value scan.
            line = re.sub(r"centre=\([^)]*\)", "", line)
            kv = dict(_KV.findall(line))
            try:
                for key in _SCALAR_KEYS:
                    rows[key].append(float(kv[key]))
            except KeyError:
                continue
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def read_nu(case_dir: str) -> float:
    path = os.path.join(case_dir, "input", "physicalProperties")
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*nu\s+([0-9.eE+-]+)\s*;", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"nu not found in {path}")


def read_bc_constants(case_dir: str) -> tuple:
    """Read kappa and A+ back out of the deployed BC source."""
    path = os.path.join(
        case_dir, "input", "registeredMeanVelocityForce",
        "totalGradientTbleNutFvPatchScalarField.C",
    )
    src = open(path).read()
    kappa = float(re.search(r"const scalar kappa\s*=\s*([0-9.eE+-]+)", src).group(1))
    aplus = float(re.search(r"const scalar Aplus\s*=\s*([0-9.eE+-]+)", src).group(1))
    return kappa, aplus, path


def build_driver(header_path: str, workdir: str) -> str:
    """Compile the driver against the case's own kernel header."""
    exe = os.path.join(workdir, "kernel_driver")
    cmd = [
        "g++", "-O2", "-std=c++14",
        f'-DRSWM_KERNEL_HEADER="{header_path}"',
        os.path.join(HERE, "kernel_driver.cpp"),
        "-o", exe,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"driver build failed:\n{proc.stderr}")
    return exe


def run_driver(exe: str, rec: dict, nu: float, kappa: float, aplus: float,
               idx: np.ndarray) -> dict:
    def g(x):  # round-trip exact, and parseable by std::cin
        return format(float(x), ".17g")

    lines = []
    for i in idx:
        lines.append(" ".join(g(v) for v in (
            rec["UMatch"][i], rec["dpds"][i], rec["ym"][i],
            nu, kappa, aplus, rec["UtMag"][i],
        )))
    proc = subprocess.run(
        [exe], input="\n".join(lines) + "\n",
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"driver run failed:\n{proc.stderr[:2000]}")
    if not proc.stdout.strip():
        raise RuntimeError(
            f"driver produced no output for {len(lines)} input records; "
            f"stderr:\n{proc.stderr[:2000]}"
        )
    cols = np.array(
        [[float(x) for x in ln.split()] for ln in proc.stdout.strip().split("\n")],
        dtype=float,
    )
    names = ("tauW", "rootCount", "selectedRoot", "homotopySteps", "converged",
             "branchLoss", "ambiguous", "truncated", "finite", "rawNut",
             "upperNut", "nut", "appliedTauS", "appliedTractionMag",
             "lowerClipped", "vectorCapped", "projFinite")
    return {n: cols[:, j] for j, n in enumerate(names)}


def relerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.where(scale > 0, scale, 1.0)
    return np.abs(a - b) / scale


def analyse_case(case_dir: str, tol: float, verbose: bool = True) -> dict | None:
    log_path = os.path.join(case_dir, "log.pimpleFoam")
    header = os.path.join(case_dir, "input", "wallmodel_tble",
                          "tbleShootContinuation.H")
    if not (os.path.isfile(log_path) and os.path.isfile(header)):
        return None

    rec = parse_face_records(log_path)
    if rec["UMatch"].size == 0:
        return None

    nu = read_nu(case_dir)
    kappa, aplus, bc_src = read_bc_constants(case_dir)
    if abs(kappa - KAPPA_EXPECTED) > 1e-12 or abs(aplus - APLUS_EXPECTED) > 1e-12:
        raise RuntimeError(
            f"{case_dir}: BC constants {kappa}/{aplus} differ from the "
            f"documented {KAPPA_EXPECTED}/{APLUS_EXPECTED}"
        )

    # Only first-solve faces are state-free and therefore reproducible.  The
    # claim that this covers the logged records is checked, not assumed: a case
    # carrying any continued face would be replayed only in part, and a partial
    # replay must not be reported as a reproduction of the deployed state.
    fresh = rec["homotopySteps"] == 33
    n_total = int(fresh.size)
    n_fresh = int(fresh.sum())
    if n_fresh != n_total:
        raise RuntimeError(
            f"{os.path.basename(case_dir)}: {n_total - n_fresh} of {n_total} "
            f"logged faces were solved from continuation state and cannot be "
            f"replayed from the log alone; a partial replay is not a "
            f"reproduction of the deployed state"
        )
    idx = np.flatnonzero(fresh)

    with tempfile.TemporaryDirectory() as workdir:
        exe = build_driver(header, workdir)
        out = run_driver(exe, rec, nu, kappa, aplus, idx)

    # ---- face-by-face reproduction of the deployed state -------------------
    log_tau = rec["tauW"][idx]
    log_nut = rec["nut"][idx]
    log_applied = rec["appliedTau"][idx]
    log_mag = rec["appliedTractionMag"][idx]
    log_lower = rec["lowerClipped"][idx]
    log_vector = rec["vectorCapped"][idx]

    e_tau = relerr(out["tauW"], log_tau)
    e_nut = relerr(out["nut"], log_nut)
    e_applied = relerr(out["appliedTauS"], log_applied)
    e_mag = relerr(out["appliedTractionMag"], log_mag)
    flag_match = int(np.sum((out["lowerClipped"] == log_lower)
                            & (out["vectorCapped"] == log_vector)))

    reproduced = bool(
        np.max(e_tau) <= tol and np.max(e_nut) <= tol
        and np.max(e_applied) <= tol and np.max(e_mag) <= tol
        and flag_match == idx.size
    )

    # ---- requested versus applied traction --------------------------------
    # The projection forces nut >= 0, so the applied traction is always
    # aligned with the matching-point velocity.  Wherever the model requests a
    # stress opposing that velocity the applied traction has the opposite sign.
    req = log_tau
    app = log_applied
    nonzero = (req != 0.0) & (app != 0.0)
    sign_flip = int(np.sum(np.sign(req[nonzero]) != np.sign(app[nonzero])))
    ratio = np.divide(app, req, out=np.full_like(app, np.nan), where=req != 0.0)
    finite_ratio = ratio[np.isfinite(ratio)]

    n_clipped = int(np.sum((log_lower > 0) | (log_vector > 0)))
    res = dict(
        case=os.path.basename(case_dir),
        case_dir=os.path.relpath(case_dir, PROJECT),
        header_sha256=sha256(header),
        bc_source_sha256=sha256(bc_src),
        nu=nu, kappa=kappa, aplus=aplus,
        ym_median=float(np.median(rec["ym"])),
        n_records=n_total, n_reproduced=n_fresh,
        max_relerr_tauW=float(np.max(e_tau)),
        max_relerr_nut=float(np.max(e_nut)),
        max_relerr_appliedTau=float(np.max(e_applied)),
        max_relerr_appliedTractionMag=float(np.max(e_mag)),
        clip_flags_matched=flag_match,
        reproduced=reproduced,
        frac_clipped=float(n_clipped / max(idx.size, 1)),
        frac_lower_clipped=float(np.sum(log_lower > 0) / max(idx.size, 1)),
        frac_vector_capped=float(np.sum(log_vector > 0) / max(idx.size, 1)),
        frac_sign_inverted=float(sign_flip / max(int(nonzero.sum()), 1)),
        n_sign_inverted=sign_flip,
        median_applied_over_requested=float(np.median(finite_ratio)),
        mean_abs_applied_over_requested=float(np.mean(np.abs(finite_ratio))),
        L1_mismatch=float(np.sum(np.abs(app - req))),
        max_abs_mismatch=float(np.max(np.abs(app - req))),
        rms_requested=float(np.sqrt(np.mean(req ** 2))),
        rms_applied=float(np.sqrt(np.mean(app ** 2))),
        n_multiroot=int(np.sum(rec["roots"][idx] > 1)),
    )
    if verbose:
        status = "REPRODUCED" if reproduced else "MISMATCH"
        print(
            f"  {res['case']:<46s} n={n_fresh:6d} {status:<11s} "
            f"max|drel|={max(res['max_relerr_tauW'], res['max_relerr_appliedTau']):.2e} "
            f"clipped={res['frac_clipped']*100:5.1f}% "
            f"sign-inverted={res['frac_sign_inverted']*100:5.1f}%", flush=True
        )
    arrays = dict(
        requested_tau=req, applied_tau=app,
        UMatch=rec["UMatch"][idx], UtMag=rec["UtMag"][idx],
        dpds=rec["dpds"][idx], ym=rec["ym"][idx],
        nut=log_nut, lower_clipped=log_lower, vector_capped=log_vector,
        repro_tau=out["tauW"], repro_applied=out["appliedTauS"],
        repro_nut=out["nut"],
    )
    return res, arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(
        RESULTS, "rswm_r23m6_ym_campaign_final"))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(
        RESULTS, "applied_traction_reproduction"))
    args = ap.parse_args()

    case_dirs = []
    for root, dirs, files in os.walk(args.bundle):
        if "log.pimpleFoam" in files:
            case_dirs.append(root)
    case_dirs.sort()
    if not case_dirs:
        print(f"no cases with logs under {args.bundle}", file=sys.stderr)
        return 2

    print(f"Reproducing deployed wall-model traction over {len(case_dirs)} case(s)",
          flush=True)
    summaries, blobs = [], {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyse_case, d, args.tol): d for d in case_dirs}
        for fut in cf.as_completed(futures):
            got = fut.result()
            if got is None:
                continue
            res, arrays = got
            summaries.append(res)
            for k, v in arrays.items():
                blobs[f"{res['case']}__{k}"] = v
    summaries.sort(key=lambda s: s["case"])

    if not summaries:
        print("no TBLE cases carried per-face records", file=sys.stderr)
        return 2

    all_repro = all(s["reproduced"] for s in summaries)
    total_faces = sum(s["n_reproduced"] for s in summaries)
    worst = max(max(s["max_relerr_tauW"], s["max_relerr_appliedTau"])
                for s in summaries)

    summary = dict(
        bundle=os.path.relpath(args.bundle, PROJECT),
        tolerance=args.tol,
        n_cases=len(summaries),
        total_faces_reproduced=total_faces,
        all_cases_reproduced=all_repro,
        worst_relative_error=worst,
        cases=summaries,
    )
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out + ".npz", **blobs)

    print(f"\n{'ALL CASES REPRODUCED' if all_repro else 'REPRODUCTION FAILED'}: "
          f"{total_faces} faces, worst relative error {worst:.3e}")
    print(f"wrote {args.out}.npz and {args.out}_summary.json")
    return 0 if all_repro else 1


if __name__ == "__main__":
    raise SystemExit(main())
