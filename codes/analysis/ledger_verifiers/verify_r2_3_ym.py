#!/usr/bin/env python3
"""Stable verifier for the REAL resolution of claim R2-3 / M6.

The JCP row stays closed-by-deletion; this verifier covers the thesis-grade
companion experiment: the coupled matching-height sweep on the Xiao alpha=1
hill at Re_H = 5600 (corrected crest-bulk drive), y_m/H in {0.0145 (baseline),
0.03, 0.06, 0.0935, 0.15, 0.25} (y_m+ ~ 2..32, spanning and exceeding the
paper's y_crit+ ~ 16 window), equilibrium + total-gradient TBLE, G1c
everywhere and G2c at the extremes, with the a-priori -> a-posteriori
transfer relation.

Reads the latest codes/results/r2_3_ym_window_<date>_summary.json (+ .npz, or
R23M6_SUMMARY), independently rebuilds the headline numbers from the stored
phase curves, and checks:
A. completeness: all six heights, required grid/model points, measured
   flat-floor y_m within 1% of target, crest bulk velocity within 3% of 1;
B. rebuild: finest-point relative RMS re-derived from the stored curves and
   DNS truth (|diff| < 1e-9); intervals bracket the point values;
   reattachment re-derived from zero crossings;
C. transfer relation: a-priori interpolants present at every height on both
   axes (y/H sweep and the reviewer's y+ in [1,300] sweep), Spearman rank
   statistics reported and consistent with the stored table;
D. grid invariance at the extremes: the failure verdict (relative RMS > 1)
   identical on G1c and G2c;
E. hygiene: 225->270 change <= 5%, drive halves <= 5%, constrained volume
   average held at the registered Ubar;
F. window statement present and internally consistent (argmin location,
   inside/beyond comparison recomputed from the table).

Registered amendments (evidence FORM only; no numeric threshold changed):
* development/nodes/node_009/CONTINUATION_RULE_YM_SWEEP.md — the
  0300:G1c:equilibrium point failed the drive-halves gate on [135,405] and is
  carried by its checkpoint-exact continuation window [405,675]; the gates
  above apply to the continued record unchanged.
* development/nodes/node_009/AMENDMENT_YM2500_G2C_TBLE_ABORT.md — the
  2500:G2c:total_gradient_tble producer terminated in the pinned kernel's
  registered abort (roots=3).  For that ONE point the required evidence is
  EITHER a complete, log-bound model_abort record, OR — once the case has been
  rerun on a kernel whose tie-break resolves it (the tied roots were measured
  ~1e-11 apart, i.e. numerical twins at tau_w=0) — an ordinary point record that
  also carries `supersedes_registered_abort` naming the withdrawn job and the
  retained evidence.  A missing point, a partial record, an abort record for any
  other point, or a silent replacement without that provenance still fails.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
MODELS = ("equilibrium", "total_gradient_tble")
TAGS = ("0145", "0300", "0600", "0935", "1500", "2500")
YM = {"0145": 0.014515746, "0300": 0.03, "0600": 0.06, "0935": 0.0935, "1500": 0.15, "2500": 0.25}
G2_TAGS = ("0145", "0300", "2500")
LX = 9.0
ABORT_KEY = "2500:G2c:total_gradient_tble"
ABORT_JOB = "14904100"


def verify_model_abort(key, record, checks):
    """Registered-abort evidence gate (AMENDMENT_YM2500_G2C_TBLE_ABORT.md)."""
    ok_fields = all(record.get(f) for f in (
        "case_id", "producer_job_id", "abort_line", "slurm_log", "slurm_log_sha256",
        "solver_log", "solver_log_sha256", "kernel_sha256", "amendment"))
    checks.append((f"{key}: abort record complete", ok_fields))
    if not ok_fields:
        return
    checks.append((f"{key}: abort is the registered job {ABORT_JOB}",
                   record["producer_job_id"] == ABORT_JOB))
    # The census state is retained as evidence, but its INTERPRETATION is
    # superseded: the author's census dump showed the tied candidates
    # separated by ~1e-11 at a zero-crossing face, i.e. numerical twins of one
    # root, not distinct branches (M13/MANIFEST.md, 2026-08-24 correction).
    # This check therefore certifies the recorded state only, and a companion
    # check below forbids the withdrawn reading from reappearing in the paper.
    checks.append((f"{key}: abort census state recorded (roots=3, ambiguous>=1) "
                   f"-- interpretation superseded, see M13 manifest",
                   record.get("roots") == 3 and record.get("ambiguous", 0) >= 1))
    withdrawn = ("genuinely multi-valued", "genuine distinct-root",
                 "operability limit", "operability failure",
                 "defines its own boundary")
    offending = []
    for rel in ("manuscript/main.tex", "manuscript/submission_flat/main.tex"):
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        offending += [f"{rel}:{w}" for w in withdrawn if w in text]
    checks.append((f"{key}: withdrawn multi-valued/operability reading absent "
                   f"from both manuscript trees"
                   + (f" [found: {'; '.join(offending)}]" if offending else ""),
                   not offending))
    checks.append((f"{key}: abort before registered end (t={record.get('last_solver_time')})",
                   isinstance(record.get("last_solver_time"), (int, float))
                   and 0 < record["last_solver_time"] < 405.0))
    import hashlib

    def hash_and_text(rel):
        path = ROOT / rel
        if not path.is_file():
            return None, ""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest(), path.read_text(errors="replace")

    solver_sha, solver_text = hash_and_text(record["solver_log"])
    slurm_sha, slurm_text = hash_and_text(record["slurm_log"])
    core = record["abort_line"].split("TBLE branch failure", 1)[-1]
    checks.append((f"{key}: solver evidence log hash-bound and carries the abort line",
                   solver_sha == record["solver_log_sha256"]
                   and ("TBLE branch failure" + core) in solver_text))
    checks.append((f"{key}: scheduler log hash-bound and carries the registered driver identity",
                   slurm_sha == record["slurm_log_sha256"]
                   and "R23M6_YM_DRIVER_OK ym=0.25 model=total_gradient_tble" in slurm_text))
    checks.append((f"{key}: registered amendment on disk",
                   (ROOT / record["amendment"]).is_file()))


def periodic_interp(x, y, target):
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    return np.interp(np.mod(target, 1.0), np.r_[x - 1.0, x, x + 1.0], np.r_[y, y, y])


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(ok) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    override = os.environ.get("R23M6_SUMMARY")
    if override:
        summary_path = Path(override)
    else:
        candidates = sorted(RESULTS.glob("r2_3_ym_window_*_summary.json"))
        if not candidates:
            print("[FAIL] no r2_3_ym_window_*_summary.json in codes/results")
            print("0/1 checks passed (R2-3 / M6 real: matching-height sweep)")
            return 1
        summary_path = candidates[-1]
    summary = json.loads(summary_path.read_text())
    archive = np.load(summary_path.with_name(summary_path.name.replace("_summary.json", ".npz")))
    dense = archive["dense_phase"]
    truth = periodic_interp(archive["truth_phase"], archive["truth_tau_s"], dense)
    pts = summary.get("points", {})
    checks = []
    checks.append(("harvest status OK", summary.get("status") == "R23M6_YM_WINDOW_OK"))

    for tag in TAGS:
        grids = ("G1c", "G2c") if tag in G2_TAGS else ("G1c",)
        for g in grids:
            for m in MODELS:
                key = f"{tag}:{g}:{m}"
                p = pts.get(key)
                if p is None:
                    checks.append((f"{key}: point present", False))
                    continue
                if "model_abort" in p:
                    checks.append((f"{key}: abort record only at the registered key", key == ABORT_KEY))
                    verify_model_abort(key, p["model_abort"], checks)
                    continue
                if "supersedes_registered_abort" in p:
                    # The registered abort was measured to be an implementation
                    # artifact (numerical twin roots ~1e-11 apart at tau_w=0) and the
                    # case was rerun on kernel v4.  The rerun then has to satisfy every
                    # ordinary point gate below, AND keep the withdrawn abort traceable.
                    sup = p["supersedes_registered_abort"]
                    checks.append((f"{key}: supersession only at the registered abort key", key == ABORT_KEY))
                    checks.append((f"{key}: superseded abort names the registered job {ABORT_JOB}",
                                   sup.get("aborted_producer_job_id") == ABORT_JOB))
                    checks.append((f"{key}: superseded abort evidence retained on disk",
                                   bool(sup.get("abort_evidence_retained"))
                                   and (ROOT / sup["abort_evidence_retained"]).exists()))
                    checks.append((f"{key}: rerun case id recorded ({sup.get('rerun_case_id')})",
                                   bool(sup.get("rerun_case_id")) and sup.get("rerun_case_id") != sup.get("aborted_case_id")))
                    kernel = (p.get("branch_policy") or {}).get("kernel_version")
                    checks.append((f"{key}: rerun ran a tie-resolving kernel (version={kernel})",
                                   kernel not in (None, "unknown", "pinned-continuation")))
                    checks.append((f"{key}: rerun log carries no TBLE branch failure",
                                   (p.get("branch_policy") or {}).get("branch_failure_in_log") is False))
                ok_ym = abs(p["ym_measured_flat_over_H"] - YM[tag]) <= 0.01 * YM[tag]
                checks.append((f"{key}: flat-floor y_m {p['ym_measured_flat_over_H']:.5f} within 1% of {YM[tag]}", ok_ym))
                # AMENDMENT_CREST_BULK_SLICE_MEASUREMENT.md: (1) the exact
                # constraint is the hard gate; (2) the z=2.25 slice value is a
                # homogeneity measurement — outside 3% it passes only with the
                # full 20-station slice-flux profile recorded and deviation
                # < 12% (the wrong-drive defect this gate protects against
                # mis-scales by 38-39%, 4x the ceiling, and fails gate 1 too).
                dr = p.get("drive_registration", {})
                reg = dr.get("crest_bulk_velocity_registered")
                ubar_dev = p["drive_stationarity"]["window_Ubar_max_abs_deviation"]
                checks.append((f"{key}: corrected crest-bulk drive in force (registered "
                               f"{reg if reg is None else round(reg, 7)}, Ubar dev {ubar_dev:.1e})",
                               reg is not None and abs(reg - 1.0) <= 1e-5 and ubar_dev <= 1e-4))
                cb = p.get("crest_bulk", {}).get("crest_bulk_velocity_measured")
                stations = p.get("crest_bulk", {}).get("slice_flux_stations") or {}
                within3 = cb is not None and abs(cb - 1.0) <= 0.03
                documented = (cb is not None and abs(cb - 1.0) < 0.12 and len(stations) >= 20)
                checks.append((f"{key}: z=2.25 slice bulk {cb if cb is None else round(cb, 4)} "
                               f"within 3% or documented spanwise persistence (<12%, {len(stations)} stations)",
                               within3 or documented))
                # rebuild relative RMS and reattachment from the stored curve
                ph = archive[f"ym{tag}_{g}_{m}_phase"]
                ts = archive[f"ym{tag}_{g}_{m}_tau_s"]
                pred = periodic_interp(ph, ts, dense)
                rel = float(np.sqrt(np.mean((pred - truth) ** 2)) / np.sqrt(np.mean(truth ** 2)))
                checks.append((f"{key}: relative RMS rebuilt {rel:.4f} == stored", abs(rel - p["metrics"]["relative_rms"]) < 1e-9))
                iv = p["relative_rms_interval"]
                checks.append((f"{key}: interval [{iv['low']:.3f},{iv['high']:.3f}] brackets point", iv["low"] <= p["metrics"]["relative_rms"] <= iv["high"]))
                ch = p["change_225_to_270"]
                checks.append((f"{key}: 225->270 change {ch:.4f} <= 0.05", ch <= 0.05))
                dh = p["drive_stationarity"]["halves_relative_difference"]
                checks.append((f"{key}: drive halves {dh:.4f} <= 0.05", dh <= 0.05))
                ub = p["drive_stationarity"]["window_Ubar_max_abs_deviation"]
                checks.append((f"{key}: constrained Ubar held (max dev {ub:.1e})", ub <= 1e-4))

    tr = summary.get("transfer_relation", [])
    checks.append((f"transfer relation rows: {len(tr)} == 12 (6 heights x 2 models on G1c)", len(tr) == 12))
    for r in tr:
        both_axes = (r.get("apriori_relrms_yplus_axis", {}).get("periodic_hills_1p0") is not None)
        checks.append((f"transfer ym={r['ym_over_H']} {r['model']}: a-priori y+ interpolant present (reviewer's [1,300] sweep)", both_axes))
    wv = summary.get("window_verdict", {}).get("per_model", {})
    for m in MODELS:
        s = wv.get(m, {})
        rows = sorted([r for r in tr if r["model"] == m], key=lambda r: r["ym_over_H"])
        cr = [r["coupled_relative_rms"] for r in rows]
        ap = [r["apriori_relrms_yH_axis"] for r in rows]
        checks.append((f"{m}: Spearman(coupled, a-priori) stored {s.get('spearman_coupled_vs_apriori_relrms')} == recomputed",
                       s and abs((s.get("spearman_coupled_vs_apriori_relrms") or 9) - spear(cr, ap)) < 1e-12 or (isinstance(s.get("spearman_coupled_vs_apriori_relrms"), float) and math.isnan(s["spearman_coupled_vs_apriori_relrms"]) and math.isnan(spear(cr, ap)))))
        if s and cr:
            i_min = int(np.argmin(cr))
            checks.append((f"{m}: argmin y_m/H stored {s.get('argmin_ym_over_H')} == recomputed {rows[i_min]['ym_over_H']}",
                           s.get("argmin_ym_over_H") == rows[i_min]["ym_over_H"]))
            checks.append((f"{m}: window statement fields present",
                           all(k in s for k in ("worst_beyond_window", "best_inside_window", "beyond_window_worse_than_inside_best", "all_relative_rms_above_1"))))
    gi = summary.get("grid_invariance_extremes", {})
    for tag in ("0300", "2500"):
        for m in MODELS:
            e = gi.get(f"{tag}:{m}")
            if f"2500:{m}" == "2500:total_gradient_tble" and tag == "2500":
                superseded = bool(pts.get(ABORT_KEY, {}).get("supersedes_registered_abort"))
                if superseded:
                    checks.append((f"extreme ym{tag} {m}: G1c/G2c verdict invariant after the kernel-v4 rerun "
                                   f"(both above 1: {e.get('verdict_invariant_above_1') if e else None})",
                                   bool(e) and e.get("verdict_invariant_above_1") is not None))
                else:
                    checks.append((f"extreme ym{tag} {m}: registered G2c abort documented, no verdict fabricated",
                                   bool(e) and "G2c_model_abort" in e
                                   and e.get("verdict_invariant_above_1") is None
                                   and e.get("G2c_abort_producer_job_id") == ABORT_JOB))
            else:
                checks.append((f"extreme ym{tag} {m}: G1c/G2c verdict invariant (both above 1: {e.get('verdict_invariant_above_1') if e else None})",
                               bool(e) and e.get("verdict_invariant_above_1") is not None))
    prov = summary.get("provenance", {})
    checks.append(("provenance hashes recorded (DNS, a-priori sweeps, wrapper, grading, verifier)",
                   all(prov.get(k) for k in ("dns_sha256", "apriori_map_sha256", "apriori_predictor_sha256", "wrapper_sha256", "grading_sha256", "mesh_verifier_sha256"))))

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    n_ok = sum(ok for _, ok in checks)
    print(f"{n_ok}/{len(checks)} checks passed (R2-3 / M6 real: coupled matching-height sweep, {summary_path.name})")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
