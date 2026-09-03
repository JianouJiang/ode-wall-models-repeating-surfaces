#!/usr/bin/env python3
"""as_deployed_evaluation.py -- score the wall model as the solver applies it.

WHAT THIS MEASURES
------------------
Every a-priori assessment in this paper (and, as far as we can tell, in the
wall-model literature) scores ``S o M``: the stress the wall model *requests*
from matching-plane data.  The coupled assessment scores what the LES actually
experienced, which is ``P o S o M``: the stress the boundary condition was able
to *deliver*.  This producer computes both on the SAME developed mean field of
the SAME simulation, so the two are finally commensurable, and splits the
a-priori/coupled discrepancy into named, separately measurable operators:

    tau_measured - tau_request(U_DNS)
        = [tau_request(U_LES) - tau_request(U_DNS)]      (I) input transfer
        + [tau_deliver(U_LES)  - tau_request(U_LES)]     (D) delivery deficiency
        + [tau_measured        - tau_deliver(U_LES)]     (N) averaging residual

(I) is the classical "the LES feeds the model different data than the DNS
would" term.  (D) is the operator this node isolates.  (N) is what is left:
the non-commutation of time-averaging with the nonlinear delivery map, plus
any modelling of the delivery map that is imperfect.  (N) is a measured
residual, not an assumption -- if the transcription of P were wrong, (N) would
be large, so it doubles as the falsification test of the whole construction.

INPUTS (all already on disk; this script launches no simulation)
    codes/results/rswm_r23m6_ym_campaign_final/<rung>/<case>/            local
        input/polyMesh, input/C      -> exact face tangents, y_m, areas
        log.pimpleFoam               -> drive gradient history
        postProcessing_sampleBottomWall/<t>/bottomWall.xy  -> measured traction
    codes/results/deployed_operator_samples/<case>/                      remote
        deployedSample/<t>/{bottomWall,topWall}Internal.xy  -> UMean, grad(pMean)
        deployedFace/<t>/{bottomWall,topWall}Face.xy        -> wallShearStressMean

OUTPUT
    codes/results/as_deployed_evaluation_<date>.npz
    codes/results/as_deployed_evaluation_<date>_summary.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(HERE))

import harvest_m13_highre as HM                      # noqa: E402
import deployed_operator as DO                       # noqa: E402

L2 = HM.load_module(HM.L2_REDUCER, "rswm_l2_locked_node012")
L3 = HM.load_module(HM.L3_ANALYSER, "rswm_l3_locked_node012")

from verify_common_matching_surface import (          # noqa: E402
    newell_normal, read_faces, read_labels, read_patch, read_points,
    read_vector_field,
)

CAMPAIGN = ROOT / "codes" / "results" / "rswm_r23m6_ym_campaign_final"
SAMPLES = ROOT / "codes" / "results" / "deployed_operator_samples"
KERNEL_DRIVER = ROOT / "codes" / "analysis" / "applied_traction" / "kernel_driver.cpp"
VENDOR_SPALDING = (HERE / "vendor_openfoam" / "nutUSpaldingWallFunction"
                   / "nutUSpaldingWallFunctionFvPatchScalarField.C")
LX = 9.0
AVERAGING_START = 135.0
KAPPA, APLUS = 0.41, 26.0
DENSE_N = 4096
BLOCK_POINTS = 512          # Lx/8, the deposited primary phase block
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260824


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def mesh_patch(case: Path, patch: str) -> dict[str, np.ndarray]:
    """Face centres, wall tangent, matching distance and area for one patch.

    The tangent is constructed exactly as the deployed boundary condition
    constructs it: ``t = (-n_y, n_x, 0)`` normalised and oriented so that
    ``t_x >= 0``.  ``y_m`` is ``1/deltaCoeffs``, i.e. the normal projection of
    the face-centre-to-cell-centre vector, which is what the boundary condition
    passes to the wall model.
    """
    mesh = case / "input" / "polyMesh"
    centres = read_vector_field(case / "input" / "C")
    points = read_points(mesh / "points")
    faces = read_faces(mesh / "faces")
    owners = read_labels(mesh / "owner")
    start, count = read_patch(mesh / "boundary", patch)
    xyz, tangent, normal_out, distance, area = [], [], [], [], []
    for face_index in range(start, start + count):
        vertices = points[faces[face_index]]
        face_centre = vertices.mean(axis=0)
        normal = newell_normal(vertices)
        t_s = np.array([-normal[1], normal[0], 0.0])
        if t_s[0] < 0.0:
            t_s = -t_s
        t_s /= np.linalg.norm(t_s)
        xyz.append(face_centre)
        tangent.append(t_s)
        normal_out.append(normal)
        distance.append(abs(float(np.dot(centres[owners[face_index]] - face_centre,
                                         normal))))
        area.append(L2.face_area(vertices))
    out = {"xyz": np.asarray(xyz), "tangent": np.asarray(tangent),
           "normal": np.asarray(normal_out), "ym": np.asarray(distance),
           "area": np.asarray(area)}
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError(f"non-finite mesh extraction {case}:{patch}")
    return out


def align_rows(mesh: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Reorder sampled rows into mesh face order (same rule as the L2 reducer)."""
    mesh_order = np.lexsort((mesh["xyz"][:, 2], mesh["xyz"][:, 0]))
    sample_order = np.lexsort((rows[:, 2], rows[:, 0]))
    if len(mesh_order) != len(sample_order):
        raise ValueError("mesh/sample face-count mismatch")
    mismatch = float(np.max(np.linalg.norm(
        mesh["xyz"][mesh_order] - rows[sample_order, :3], axis=1)))
    if mismatch > 2.0e-6:
        raise ValueError(f"mesh/sample centre mismatch {mismatch:.3e}")
    aligned = np.empty_like(rows)
    aligned[mesh_order] = rows[sample_order]
    return aligned


def read_xy(path: Path) -> np.ndarray:
    rows = [[float(t) for t in line.split()]
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    values = np.asarray(rows, float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid sample file {path}")
    return values


def spanwise_curve(mesh: dict[str, np.ndarray], values: np.ndarray) -> dict[str, Any]:
    """Area-weighted spanwise average of a face scalar onto the phase curve."""
    rounded = np.round(mesh["xyz"][:, 0], 9)
    x_unique, inverse = np.unique(rounded, return_inverse=True)
    z_unique = np.unique(np.round(mesh["xyz"][:, 2], 9))
    dz = float(np.median(np.diff(z_unique)))
    span = float(np.ptp(z_unique) + dz)
    out, wall_ds, tangent_x = [], [], []
    for index in range(len(x_unique)):
        chosen = inverse == index
        weights = mesh["area"][chosen]
        out.append(np.average(values[chosen], weights=weights))
        wall_ds.append(float(np.sum(weights) / span))
        mean_t = np.average(mesh["tangent"][chosen], axis=0, weights=weights)
        tangent_x.append(float(mean_t[0] / np.linalg.norm(mean_t)))
    return {"phase": np.asarray(x_unique) / LX, "x": np.asarray(x_unique),
            "tau_s": np.asarray(out), "wall_ds": np.asarray(wall_ds),
            "tangent_x": np.asarray(tangent_x),
            "signed_tangent_force_per_span":
                float(np.sum(values * mesh["area"]) / span),
            "signed_x_force_per_span":
                float(np.sum(values * mesh["tangent"][:, 0] * mesh["area"]) / span),
            "tau_x": np.asarray(out) * np.asarray(tangent_x)}


# ---------------------------------------------------------------------------
# drive gradient
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"^Time = ([0-9.eE+-]+)s?\s*$")
_GRAD_RE = re.compile(r"pressure gradient = (-?[0-9.eE+-]+)")


def averaging_window(case: Path, t_end: float) -> tuple[float, float]:
    """The field-averaging window this deposit actually accumulated over.

    ``checkpoints/<t>/fieldAverageProperties`` records ``totalTime``; the
    window is therefore ``[t - totalTime, t]``.  The continuation deposits
    reset the accumulator, so the same physical run supplies two DISJOINT
    windows, which is used here as an independent averaging-window control.
    """
    prop = case / "checkpoints" / f"{t_end:g}" / "fieldAverageProperties"
    if not prop.exists():
        return AVERAGING_START, t_end
    m = re.search(r"totalTime\s+([0-9.eE+-]+)\s*;", prop.read_text())
    if not m:
        return AVERAGING_START, t_end
    return t_end - float(m.group(1)), t_end


def drive_gradient_mean(case: Path, t_end: float,
                        t_start: float | None = None) -> dict[str, float]:
    """Time-mean of the registered drive gradient over the averaging window.

    The boundary condition removes ``g * e_x`` from ``grad p`` before passing
    the streamwise pressure gradient to the wall model.  Because that removal
    is linear, the correct value to remove from ``grad(pMean)`` is the mean of
    ``g`` over exactly the field-averaging window of the deposit being scored.
    """
    if t_start is None:
        t_start = AVERAGING_START
    # The controller prints once per PISO corrector; the value that advances the
    # step is the last one printed within that step, so one value per time step
    # is collected and those are averaged over the field-averaging window.
    per_step: list[float] = []
    all_prints: list[float] = []
    now, current = None, None
    with open(case / "log.pimpleFoam", "r", errors="ignore") as fh:
        for line in fh:
            m = _TIME_RE.match(line.strip())
            if m:
                if current is not None and t_start < now <= t_end + 1e-9:
                    per_step.append(current)
                now, current = float(m.group(1)), None
                continue
            g = _GRAD_RE.search(line)
            if g and now is not None:
                current = float(g.group(1))
                if t_start < now <= t_end + 1e-9:
                    all_prints.append(current)
    if current is not None and now is not None and t_start < now <= t_end + 1e-9:
        per_step.append(current)
    if not per_step:
        raise ValueError(f"no drive-gradient records in window for {case}")
    arr = np.asarray(per_step, float)
    every = np.asarray(all_prints, float)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "n_steps": int(arr.size), "last": float(arr[-1]),
            "mean_all_correctors": float(every.mean()),
            "window": [t_start, t_end]}


# ---------------------------------------------------------------------------
# the deployed TBLE kernel, compiled from the case's own header
# ---------------------------------------------------------------------------
def build_tble_driver(case: Path, workdir: Path) -> Path:
    header = case / "input" / "wallmodel_tble" / "tbleShootContinuation.H"
    if not header.exists():
        raise FileNotFoundError(header)
    exe = workdir / "kernel_driver"
    cmd = ["g++", "-O2", "-std=c++14",
           f'-DRSWM_KERNEL_HEADER="{header}"',
           str(KERNEL_DRIVER), "-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"driver build failed:\n{proc.stderr[:2000]}")
    return exe


def run_tble_driver(exe: Path, u_m, dpds, y_m, nu, q,
                    workers: int = 2) -> dict[str, np.ndarray]:
    def g(v):
        return f"{float(v):.17g}"

    records = [" ".join((g(u_m[i]), g(dpds[i]), g(y_m[i]), g(nu), g(KAPPA),
                         g(APLUS), g(q[i])))
               for i in range(len(u_m))]
    # The kernel is a 32-stage pressure homotopy over a 256-interval root scan,
    # so it costs ~0.07 s per face.  Faces are independent: shard them over
    # `workers` identical driver processes and reassemble in order.
    shards = [records[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]
    with tempfile.TemporaryDirectory(prefix="tbleshard_") as tmp:
        tmp = Path(tmp)
        procs = []
        for k, shard in enumerate(shards):
            src, dst = tmp / f"in{k}", tmp / f"out{k}"
            src.write_text("\n".join(shard) + "\n")
            with open(src) as fin, open(dst, "w") as fout:
                procs.append(subprocess.Popen([str(exe)], stdin=fin,
                                              stdout=fout,
                                              stderr=subprocess.DEVNULL))
        for proc in procs:
            if proc.wait() != 0:
                raise RuntimeError("driver run failed")
        parsed = [np.asarray([[float(v) for v in ln.split()]
                              for ln in (tmp / f"out{k}").read_text()
                              .strip().splitlines()], float)
                  for k in range(len(shards))]
    out = np.empty((len(records), parsed[0].shape[1]), float)
    for k, block in enumerate(parsed):
        if block.shape[0] != len(shards[k]):
            raise RuntimeError(
                f"driver shard {k} returned {block.shape[0]} of {len(shards[k])}")
        out[k::workers] = block
    if out.shape[0] != len(u_m):
        raise RuntimeError(f"driver returned {out.shape[0]} of {len(u_m)} records")
    return {"tau": out[:, 0], "roots": out[:, 1], "converged": out[:, 4],
            "branch_loss": out[:, 5], "ambiguous": out[:, 6],
            "truncated": out[:, 7], "finite": out[:, 8],
            "nut": out[:, 11], "applied": out[:, 12],
            "lower_clipped": out[:, 14], "vector_capped": out[:, 15]}


# ---------------------------------------------------------------------------
# metric protocol (identical to the pinned reducers)
# ---------------------------------------------------------------------------
def curve_metrics(curve, truth_phase, truth_tau) -> dict[str, float]:
    return L2.metrics(curve, truth_phase, truth_tau)


def phase_block_interval(truth_dense, pred_dense, draws=BOOTSTRAP_DRAWS,
                         seed=BOOTSTRAP_SEED) -> dict[str, Any]:
    samples = L3.circular_block_bootstrap(
        truth_dense, {("c", "m"): pred_dense},
        block_points=BLOCK_POINTS, draws=draws, seed=seed)
    values = np.asarray(samples[("c", "m")], float)
    return {"median": float(np.median(values)),
            "lo": float(np.quantile(values, 0.025)),
            "hi": float(np.quantile(values, 0.975))}


# ---------------------------------------------------------------------------
def case_model(case: Path) -> str:
    text = (case / "input" / "nut").read_text(errors="ignore")
    if "totalGradientTbleNut" in text:
        return "total_gradient_tble"
    if "nutUSpaldingWallFunction" in text:
        return "equilibrium"
    raise ValueError(f"cannot identify the deployed wall model of {case}")


def nu_of(case: Path) -> float:
    text = (case / "input" / "physicalProperties").read_text(errors="ignore")
    m = re.search(r"^\s*nu\s+([0-9.eE+-]+)\s*;", text, re.M)
    if not m:
        raise ValueError(f"nu not found for {case}")
    return float(m.group(1))


def evaluate_patch(case: Path, patch: str, time: str, model: str, nu: float,
                   drive: float, exe: Path | None,
                   workers: int = 2) -> dict[str, Any]:
    mesh = mesh_patch(case, patch)
    tag = "bottomWall" if patch == "bottomWall" else "topWall"
    internal = read_xy(SAMPLES / case.name / "deployedSample" / time /
                       f"{tag}Internal.xy")
    face = read_xy(SAMPLES / case.name / "deployedFace" / time / f"{tag}Face.xy")
    internal = align_rows(mesh, internal)
    face = align_rows(mesh, face)

    u_vec = internal[:, 3:6]
    grad_p = internal[:, 6:9]
    t_s = mesh["tangent"]
    n_f = mesh["normal"]
    y_m = mesh["ym"]

    u_m = np.einsum("ij,ij->i", u_vec, t_s)
    u_n = np.einsum("ij,ij->i", u_vec, n_f)
    u_t = u_vec - u_n[:, None] * n_f
    q = np.linalg.norm(u_t, axis=1)                 # TBLE arm speed  |U_t|
    s = np.linalg.norm(u_vec, axis=1)               # Spalding arm speed |U_c|
    dpds = np.einsum("ij,ij->i", grad_p - drive * np.array([1.0, 0.0, 0.0]), t_s)

    if model == "total_gradient_tble":
        res = run_tble_driver(exe, u_m, dpds, y_m, nu, q, workers=workers)
        tau_r = res["tau"]
        # For this architecture the stress handed to the delivery map IS the
        # scored request, so internal and scored request coincide.
        tau_internal = tau_r
        deliv = DO.project_tble(tau_r, u_m, q, y_m, nu)
        kernel_ok = dict(
            converged=float(np.mean(res["converged"])),
            finite=float(np.mean(res["finite"])),
            multiroot=int(np.count_nonzero(res["roots"] > 1)),
            max_driver_vs_python=float(np.max(np.abs(
                res["applied"] - deliv.tau_delivered))))
        u_tau_iter = None
    else:
        u_tau = DO.spalding_utau_converged(s, y_m, nu)
        deliv = DO.project_spalding(u_tau, u_m, s, y_m, nu)
        # the model AS PUBLISHED and AS SCORED a priori: signed, streamwise
        tau_r = np.sign(u_m) * DO.spalding_utau_converged(
            np.abs(u_m), y_m, nu) ** 2
        # what the deployed wall function actually asks the delivery map for:
        # Spalding evaluated at the FULL relative speed s = |U_c| (proposition 2)
        tau_internal = np.sign(u_m) * u_tau ** 2
        u_tau_dep, iters = DO.spalding_utau_deployed(s, y_m, nu,
                                                     np.zeros_like(s))
        u_tau_iter = dict(
            max_rel_dev_deployed_vs_converged=float(np.max(
                np.abs(u_tau_dep - u_tau) / np.maximum(u_tau, 1e-30))),
            median_iterations=float(np.median(iters)))
        kernel_ok = dict(converged=1.0, finite=float(np.mean(np.isfinite(tau_r))),
                         multiroot=0, max_driver_vs_python=0.0)

    tau_d = deliv.tau_delivered
    # measured: the mean traction the LES actually carried on this patch
    tau_meas_face = np.einsum("ij,ij->i", -face[:, 3:6], t_s)

    curves = {
        "request": spanwise_curve(mesh, tau_r),
        "deliver": spanwise_curve(mesh, tau_d),
        "measured": spanwise_curve(mesh, tau_meas_face),
    }
    speed = q if model == "total_gradient_tble" else s
    # Corollary 1.2 is a statement about the stress the delivery map RECEIVES.
    bound = DO.contraction_bound(tau_internal, u_m, speed, y_m, nu)
    # The same bound evaluated against the a-priori-SCORED request is a
    # different question, and for the equilibrium arm a different answer:
    # proposition 2 says the deployed wall function evaluates the wall law at
    # the larger speed |U_c|, so it can deliver more than its scored request.
    bound_scored = DO.contraction_bound(tau_r, u_m, speed, y_m, nu)
    faithful = DO.faithful_mask(tau_internal, u_m, speed, y_m, nu)
    scale = max(float(np.sqrt(np.mean(tau_meas_face ** 2))), 1e-30)
    return {
        "patch": patch, "n_faces": int(len(tau_r)),
        "ym_median": float(np.median(y_m)),
        "ym_min": float(np.min(y_m)), "ym_max": float(np.max(y_m)),
        "curves": curves,
        "faces": {"u_m": u_m, "q": q, "s": s, "y_m": y_m, "dpds": dpds,
                  "tau_request": tau_r, "tau_internal": tau_internal,
                  "tau_deliver": tau_d,
                  "tau_measured": tau_meas_face, "regime": deliv.regime,
                  "nut": deliv.nut},
        "regime_fraction": {
            "faithful": float(np.mean(deliv.regime == 0)),
            "alignment_capped": float(np.mean(deliv.regime == 1)),
            "sign_refused": float(np.mean(deliv.regime == 2))},
        "identity_faithful_fraction": float(np.mean(faithful)),
        "sign_disagreement_request_vs_deliver":
            float(np.mean(np.sign(tau_r) != np.sign(tau_d))),
        "rms_request": float(np.sqrt(np.mean(tau_r ** 2))),
        "rms_deliver": float(np.sqrt(np.mean(tau_d ** 2))),
        "rms_measured": float(np.sqrt(np.mean(tau_meas_face ** 2))),
        "delivery_attenuation": float(1.0 - np.sqrt(np.mean(tau_d ** 2))
                                      / max(np.sqrt(np.mean(tau_r ** 2)), 1e-30)),
        "sign_opposed_fraction": float(np.mean(tau_r * u_m < 0.0)),
        "below_laminar_floor_fraction": float(np.mean(
            (deliv.regime == 2) & (tau_r * u_m >= 0.0))),
        "contraction_bound_violations": int(np.count_nonzero(
            np.abs(tau_d) > bound * (1.0 + 1e-9) + 1e-30)),
        "exceeds_bound_of_scored_request_fraction": float(np.mean(
            np.abs(tau_d) > bound_scored * (1.0 + 1e-9) + 1e-30)),
        "median_delivered_over_scored_request": float(np.median(
            np.abs(tau_d) / np.maximum(np.abs(tau_r), 1e-30))),
        "reconstruction_residual_rms_over_measured":
            float(np.sqrt(np.mean((tau_d - tau_meas_face) ** 2)) / scale),
        "kernel": kernel_ok,
        "spalding_iteration": u_tau_iter,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--times", default="405,360,315")
    ap.add_argument("--patches", default="bottomWall,topWall")
    ap.add_argument("--only", default="", help="comma-separated case filter")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    times = [t.strip() for t in args.times.split(",") if t.strip()]
    patches = [p.strip() for p in args.patches.split(",") if p.strip()]

    dns = np.load(HM.DNS_5600)
    truth_tau, truth_audit = L2.dns_tangent_reference(dns)
    truth_phase = np.mod((np.asarray(dns["x"], float)
                          - float(np.min(np.asarray(dns["x"], float)))) / LX, 1.0)
    dense = np.arange(DENSE_N, dtype=float) / DENSE_N
    truth_dense = L2.periodic_interp(truth_phase, truth_tau, dense)

    # Every deposit is a distinct logical record.  Two deposits may share a
    # case name (an original and its continuation) because the continuation
    # ran in the same directory; they are distinguished by their rung and by
    # the disjoint averaging windows they carry.
    unique = sorted([p for rung in CAMPAIGN.iterdir() if rung.is_dir()
                     for p in rung.iterdir() if p.is_dir()],
                    key=lambda p: (p.parent.name, p.name))
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        unique = [c for c in unique
                  if c.name in wanted or f"{c.parent.name}/{c.name}" in wanted]

    blobs: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    workdir = Path(tempfile.mkdtemp(prefix="asdeployed_"))

    for case in unique:
        if not (SAMPLES / case.name).exists():
            print(f"SKIP (no developed-state sample): {case.name}", flush=True)
            continue
        model = case_model(case)
        nu = nu_of(case)
        exe = build_tble_driver(case, workdir) if model == "total_gradient_tble" else None
        available = sorted((p.name for p in (case / "checkpoints").iterdir()
                            if p.is_dir()), key=float) \
            if (case / "checkpoints").is_dir() else []
        for time in [t for t in times if t in available] or available:
            if not (SAMPLES / case.name / "deployedSample" / time).exists():
                continue
            t_start, t_end = averaging_window(case, float(time))
            drive = drive_gradient_mean(case, t_end, t_start)
            for patch in patches:
                try:
                    ev = evaluate_patch(case, patch, time, model, nu,
                                        drive["mean"], exe,
                                        workers=args.workers)
                except FileNotFoundError:
                    continue
                rec: dict[str, Any] = {
                    "case": case.name, "deposit": case.parent.name,
                    "averaging_window": [t_start, t_end],
                    "model": model, "time": float(time),
                    "nu": nu, "drive_gradient": drive,
                    **{k: v for k, v in ev.items()
                       if k not in ("curves", "faces")},
                }
                if patch == "bottomWall":
                    for name, curve in ev["curves"].items():
                        m = curve_metrics(curve, truth_phase, truth_tau)
                        dense_pred = L2.periodic_interp(
                            curve["phase"], curve["tau_s"], dense)
                        rec[f"metrics_{name}"] = {
                            "r2": m["r2"], "relative_rms": m["relative_rms"],
                            "sign_accuracy": m["sign_accuracy"],
                            "separation_x_over_H": m["separation_x_over_H"],
                            "reattachment_x_over_H": m["reattachment_x_over_H"],
                            "interval_relative_rms": phase_block_interval(
                                truth_dense, dense_pred, draws=args.draws)}
                        blobs[f"{case.name}__{time}__{name}__dense"] = dense_pred
                    # the three-term bridge, on the dense phase grid
                    req = blobs[f"{case.name}__{time}__request__dense"]
                    dlv = blobs[f"{case.name}__{time}__deliver__dense"]
                    mea = blobs[f"{case.name}__{time}__measured__dense"]
                    denom = max(float(np.sqrt(np.mean(mea ** 2))), 1e-30)
                    rec["bridge"] = {
                        "delivery_deficiency_rms_over_measured":
                            float(np.sqrt(np.mean((dlv - req) ** 2)) / denom),
                        "averaging_residual_rms_over_measured":
                            float(np.sqrt(np.mean((mea - dlv) ** 2)) / denom),
                        "total_request_to_measured_rms_over_measured":
                            float(np.sqrt(np.mean((mea - req) ** 2)) / denom),
                        "delivery_share_of_total": float(
                            np.sqrt(np.mean((dlv - req) ** 2))
                            / max(float(np.sqrt(np.mean((mea - req) ** 2))), 1e-30)),
                    }
                for key, value in ev["faces"].items():
                    blobs[f"{case.name}__{time}__{patch}__{key}"] = value
                for name, curve in ev["curves"].items():
                    blobs[f"{case.name}__{time}__{patch}__{name}__phase"] = curve["phase"]
                    blobs[f"{case.name}__{time}__{patch}__{name}__tau"] = curve["tau_s"]
                records.append(rec)
                print(f"OK {case.name} t={time} {patch} model={model} "
                      f"faithful={ev['regime_fraction']['faithful']:.4f} "
                      f"refused={ev['regime_fraction']['sign_refused']:.4f} "
                      f"recon={ev['reconstruction_residual_rms_over_measured']:.4f}",
                      flush=True)

    stamp = _dt.datetime.now().strftime("%Y%m%d") + (f"_{args.tag}" if args.tag else "")
    summary: dict[str, Any] = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_records": len(records),
        "records": HM.json_ready(records) if hasattr(HM, "json_ready")
        else L2.json_ready(records),
        "truth": {"file": str(HM.DNS_5600.relative_to(ROOT)),
                  "sha256": HM.sha256(HM.DNS_5600),
                  "audit": L2.json_ready(truth_audit)},
        "provenance": {
            "tble_header_sha256": {
                c.name: HM.sha256(c / "input" / "wallmodel_tble"
                                  / "tbleShootContinuation.H")
                for c in unique
                if (c / "input" / "wallmodel_tble"
                    / "tbleShootContinuation.H").exists()},
            "kernel_driver_sha256": HM.sha256(KERNEL_DRIVER),
            "spalding_source_sha256": HM.sha256(VENDOR_SPALDING),
            "operator_module_sha256": HM.sha256(HERE / "deployed_operator.py"),
            "l2_reducer_sha256": HM.sha256(HM.L2_REDUCER),
            "l3_analyser_sha256": HM.sha256(HM.L3_ANALYSER),
        },
        "protocol": {
            "developed_state": ("OpenFOAM fieldAverage over [135, t]; the wall "
                                "model is evaluated on that mean field"),
            "kernel_branch": ("first-solve branch (hasPrevious=false): homotopy "
                              "from the zero-pressure-gradient root, i.e. the "
                              "same cold evaluation the a-priori ladder uses"),
            "metric": "pinned rswm_common_surface_grid_l2.metrics",
            "bootstrap": {"block_points": BLOCK_POINTS, "draws": args.draws,
                          "seed": BOOTSTRAP_SEED},
        },
    }
    out_npz = ROOT / "codes" / "results" / f"as_deployed_evaluation_{stamp}.npz"
    out_json = ROOT / "codes" / "results" / f"as_deployed_evaluation_{stamp}_summary.json"
    np.savez_compressed(out_npz, **blobs)
    out_json.write_text(json.dumps(summary, indent=1))
    print(f"WROTE {out_npz.relative_to(ROOT)} ({len(blobs)} arrays)")
    print(f"WROTE {out_json.relative_to(ROOT)} ({len(records)} records)")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
