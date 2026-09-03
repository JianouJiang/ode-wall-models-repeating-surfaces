#!/usr/bin/env python3
"""Run table and Slurm batch generator for the mode-resolved wall-traction
sensitivity experiment (L0 node_002).

The experiment branches coupled WMLES continuations from a single completed
Re = 5,600 periodic-hill baseline at t = 405 and changes exactly one thing: the
traction the momentum equation receives on the hill wall carries a prescribed
perturbation in one streamwise Fourier mode, injected at matched area-weighted
RMS across modes.

Three families of run:

  null    -- no perturbation, only a round-off-level one-shot kick.  These
             measure the CHAOTIC NOISE FLOOR of every reported quantity: the
             spread among statistically identical realisations of the same
             simulation.  No perturbation response is interpretable below it.
  nutmult -- PRIMARY.  The eddy viscosity is perturbed multiplicatively, which
             is non-negative by construction, so the realizability clip can
             never bind and the delivered perturbation is exactly the requested
             one.  Its delivered field is not a pure Fourier mode (it is the
             mode times the turbulent traction), which is why every response is
             normalised by the MEASURED delivered norm rather than the nominal
             amplitude.
  add     -- additive pure modes.  Exactly orthogonal on the wall as requested,
             but on a separating wall the realizability clip binds on a large
             fraction of faces, rectifies the perturbation and injects net force
             the requested field does not carry.  Retained, with the clipped
             fraction and the delivered net force measured and reported.
  mult    -- multiplicative on the total traction: an intermediate arm whose
             clip binds only where the eddy viscosity is below nu/3.

The k = 0 mode carries net force; the k >= 1 modes carry none in the continuum.
On the discrete curved wall the residual is measured, not assumed -- and the
flat top wall of the same run, where face areas are uniform, is the control
that shows what "zero" looks like.
"""

import argparse
import json
import math
import pathlib

# Baseline delivered wall traction on the hill wall of the branch point,
# measured from the deposited sample at t = 405 (area-weighted over 4,800
# faces): RMS|tau| = 3.38441e-03 in units of u_b^2 with u_b = 0.721045.
TAU_RMS_BASELINE = 3.38441e-03

A1 = 0.25 * TAU_RMS_BASELINE     # 8.4610e-04, a 25 % RMS traction perturbation
A2 = 0.50 * TAU_RMS_BASELINE     # 1.6922e-03, 50 %, for the linearity test

# Clip-free arm: RMS of the RELATIVE perturbation applied to the eddy
# viscosity.  For k >= 1 the mode RMS is 1/sqrt(2), so the peak relative
# perturbation is rms*sqrt(2); both values below keep |A g| < 1, which is the
# condition under which nut(1 + A g) >= 0 and the clip can never bind.
NUT_A1 = 0.30
NUT_A2 = 0.15

START_TIME = 405.0
END_TIME = 525.0                 # 120 time units ~ 9.6 flow-throughs
AVG_START = 425.0                # discard 20 units of adjustment
PERIOD = 9.0                     # Lx/H of the meshed Xiao hill


def run_table():
    runs = []

    # -- chaotic noise floor -------------------------------------------------
    # N0 is the deterministic continuation of the baseline; N1..N5 differ from
    # it only by a one-shot uniform traction gain of 1 + eps with eps at the
    # 1e-10 level, far below any physical effect and far above round-off.
    for i, eps in enumerate([0.0, 1e-10, 2e-10, 4e-10, 8e-10, 1.6e-9]):
        runs.append(dict(
            run_id=f"ms_null_{i}", family="null", kind="none",
            k=0, phase=0.0, rms=0.0, seed=eps,
        ))

    # -- PRIMARY: clip-free modal sweep --------------------------------------
    for tag, k, ph in [("k0", 0, 0.0), ("k1p00", 1, 0.0),
                       ("k1p90", 1, math.pi/2), ("k2", 2, 0.0),
                       ("k4", 4, 0.0), ("k8", 8, 0.0)]:
        runs.append(dict(
            run_id=f"ms_nutA_{tag}", family="nutmult", kind="nutMultiplicative",
            k=k, phase=ph, rms=NUT_A1, seed=0.0,
        ))

    # -- clip-free at half amplitude: is the response linear in the input? ---
    for tag, k, ph in [("k0", 0, 0.0), ("k1p00", 1, 0.0), ("k2", 2, 0.0)]:
        runs.append(dict(
            run_id=f"ms_nutB_{tag}", family="nutmult", kind="nutMultiplicative",
            k=k, phase=ph, rms=NUT_A2, seed=0.0,
        ))

    # -- additive pure modes: exactly orthogonal, but clipped ----------------
    for tag, k, ph in [("k0", 0, 0.0), ("k1p00", 1, 0.0),
                       ("k2", 2, 0.0), ("k4", 4, 0.0)]:
        runs.append(dict(
            run_id=f"ms_addA_{tag}", family="add", kind="additive",
            k=k, phase=ph, rms=A1, seed=0.0,
        ))

    # -- intermediate arm: multiplicative on the total traction -------------
    for tag, k in [("k0", 0), ("k1p00", 1)]:
        runs.append(dict(
            run_id=f"ms_mult_{tag}", family="mult", kind="multiplicative",
            k=k, phase=0.0, rms=0.25, seed=0.0,
        ))

    for r in runs:
        r.update(start_time=START_TIME, end_time=END_TIME,
                 avg_start=AVG_START, period_length=PERIOD)
    return runs


SLURM_HEADER = """#!/bin/bash -l
#SBATCH --job-name=msens{batch}
#SBATCH --account=engs-cfd-combusterflow
#SBATCH --clusters=arc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time={walltime}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
# Mode-resolved wall-traction sensitivity, batch {batch} of {nbatch}.
# Self-gating: the instrument's exactness test runs FIRST and the batch aborts
# if the null configuration is not bit-for-bit the deployed wall function.
echo "JOB_START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
set -e
bash "$SLURM_SUBMIT_DIR/jobs/rswm_modal_identity_driver.sh" 406 "b{batch}_$SLURM_JOB_ID"
echo "IDENTITY_GATE_PASSED"
set +e
rc_all=0
"""

SLURM_CASE = """
echo "=== CASE {run_id} ==="
bash "$SLURM_SUBMIT_DIR/jobs/rswm_modal_sensitivity_driver.sh" \\
    "{run_id}" "{kind}" {k} {phase!r} {rms!r} {seed!r} {end_time!r} {avg_start!r}
rc=$?; echo "CASE_RC {run_id} $rc"; [ $rc -eq 0 ] || rc_all=1
"""

SLURM_FOOTER = """
echo "BATCH_RC $rc_all"
[ $rc_all -eq 0 ] && echo "JOB_DONE_OK $(date -Is)" || echo "JOB_DONE_PARTIAL $(date -Is)"
exit $rc_all
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="jobs")
    ap.add_argument("--nbatch", type=int, default=4)
    ap.add_argument("--walltime", default="04:00:00")
    args = ap.parse_args()

    runs = run_table()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "modal_sensitivity_runs.json").write_text(
        json.dumps(dict(
            tau_rms_baseline=TAU_RMS_BASELINE,
            amplitude_A1=A1, amplitude_A2=A2,
            nut_amplitude_A1=NUT_A1, nut_amplitude_A2=NUT_A2,
            start_time=START_TIME, end_time=END_TIME, avg_start=AVG_START,
            period_length=PERIOD, runs=runs,
        ), indent=2)
    )

    # Interleave families across batches so a lost batch never removes a whole
    # family, and every batch carries at least one null member.
    batches = [[] for _ in range(args.nbatch)]
    for i, r in enumerate(runs):
        batches[i % args.nbatch].append(r)

    names = []
    for b, group in enumerate(batches):
        s = SLURM_HEADER.format(batch=b, nbatch=args.nbatch,
                                walltime=args.walltime)
        for r in group:
            s += SLURM_CASE.format(**r)
        s += SLURM_FOOTER
        f = out / f"modal_sensitivity_b{b}.slurm"
        f.write_text(s)
        names.append(str(f))
        print(f"{f}  ({len(group)} cases: "
              f"{', '.join(x['run_id'] for x in group)})")

    print(f"\n{len(runs)} runs in {args.nbatch} batches")
    return names


if __name__ == "__main__":
    main()
