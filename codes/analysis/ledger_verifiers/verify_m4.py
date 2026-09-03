#!/usr/bin/env python3
"""Independent stable verifier for referee-ledger row M4 (WM-SGS interaction).

Reads the newest ``codes/results/m4_sgs_sensitivity_<date>.{json,npz}``
certificate produced by ``codes/analysis/harvest_m4_sgs.py`` and checks,
without calling the harvest:

* identity of the locked inputs (WALE Level-2 archive, Level-3 certificate,
  campaign manifest, DNS reference) against the hashes the certificate carries;
* an independent rebuild of every headline E_tau from the raw sampled wall
  traction in the campaign bundle, using the ANALYTIC Xiao hill tangent
  (not the mesh normals the harvest used) and plain spanwise averaging;
* the exact sign-flip p-values, Holm adjustment, bootstrap quantiles and the
  A1/A2 acceptance logic recomputed from the stored samples;
* that the SGS swap was real: every bundle's momentumTransport names the
  alternative model, the solver selected it, the matching surface is the WALE
  one, and the wall-adjacent SGS viscosity differs from WALE;
* red/green fixtures on the criterion code in BOTH verdict directions.

It is outcome-neutral by construction: it certifies that the measurement is complete,
independently reproducible and correctly labelled, and prints the measured per-closure
verdict.  It passes on an honest non-invariance result and fails on a certificate that
claims an invariance its own numbers do not support (HANDOVER_20260825 section 5.1:
"gates that encode an outcome are bugs").
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
CAMPAIGN = RESULTS / "rswm_m4_sgs_re_campaign_final"
WALE_ROOT = RESULTS / "rswm_xiao_highre_campaign_m13_final" / "re5600"   # Agent B corrected baseline
EXPECTED_UBAR = 0.721045
MGLET_WALL = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
CREST_HEIGHT = 2.036
# Q/2.036 from a 300-point sampled mid-channel profile is a solution-dependent DIAGNOSTIC,
# not the constraint itself: meanVelocityForce enforces the volume average exactly (checked
# separately against fvConstraints at 2e-3), while how much of that flux passes the crest
# section depends on the closure, and the line integral carries its own quadrature error.
# Measured spread over the 14 corrected bundles is 0.891-1.055.  The failure mode this
# check exists to catch - the withdrawn volume-average drive - sits at 1.35-1.40, so a
# 0.15 band separates the two cleanly with a 0.20 margin.
CREST_TOLERANCE = 0.15
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
from make_xiao_dns_wmles_case import xiao_profile  # noqa: E402

MODELS = ("equilibrium", "total_gradient_tble")
DENSE_N = 4096
BLOCK = 512
LX = 9.0
RATIO_LOW, RATIO_HIGH, MARGIN_FRACTION, ALPHA = 0.75, 1.3333333333, 0.5, 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_interp(x, y, target):
    order = np.argsort(np.asarray(x, float))
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    return np.interp(np.mod(target, 1.0), np.r_[xs - 1.0, xs, xs + 1.0], np.r_[ys, ys, ys])


def exact_sign_p(values, two_sided=False):
    values = np.asarray(values, float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    observed = float(np.mean(values))
    return float(np.mean(np.abs(null) >= abs(observed))) if two_sided else float(np.mean(null >= observed))


def holm(p):
    ordered = sorted(p, key=p.get)
    out, running, n = {}, 0.0, len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * p[key]))
        out[key] = running
    return out


def bundle_ubar(bundle: Path, manifest: dict) -> float:
    """The constrained volume-average velocity of a finalized bundle.

    Two finalizer generations wrote two manifest schemas (the M4 v2 finalizer hoists the
    key, Agent B's M13 finalizer keeps the deposited schema), so fall back to the
    dictionary the solver actually read.  The guard is never skipped: a bundle whose
    drive cannot be established fails.  Proving Ubar = 0.721045 is what proves the
    crest-bulk correction - the reason the whole Re-5600 matrix was re-run - is in force.
    """
    for key in ("volume_average_Ubar", "fvConstraints_Ubar"):
        if key in manifest:
            return float(manifest[key])
    driver = manifest.get("driver_manifest") or {}
    for key in ("volume_average_Ubar", "fvConstraints_Ubar"):
        if key in driver:
            return float(driver[key])
    match = re.search(r"Ubar\s+\(\s*([0-9.eE+-]+)\s+0\s+0\s*\)",
                      (bundle / "input" / "fvConstraints").read_text())
    if match is None:
        raise AssertionError(f"{bundle.name}: no streamwise Ubar in input/fvConstraints")
    return float(match.group(1))


def relative_rms(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / np.sqrt(np.mean(truth ** 2)))


def raw_tangent_traction(bundle: Path, checkpoint: str):
    """Rebuild tau_s(phase) from the raw sampler file with the analytic tangent."""
    rows = np.loadtxt(bundle / "postProcessing_sampleBottomWall" / checkpoint / "bottomWall.xy")
    x, tau_vec = rows[:, 0], -rows[:, 3:6]
    h = 1.0e-6
    slope = np.asarray([(xiao_profile(min(xi, LX - xi) + h) - xiao_profile(min(xi, LX - xi) - h)) / (2 * h)
                        for xi in x])
    slope = np.where(x > LX / 2, -slope, slope)          # upslope mirrors the downslope
    tx, ty = 1.0 / np.sqrt(1 + slope ** 2), slope / np.sqrt(1 + slope ** 2)
    tau_s = tau_vec[:, 0] * tx + tau_vec[:, 1] * ty
    xs, inverse = np.unique(np.round(x, 9), return_inverse=True)
    mean = np.asarray([tau_s[inverse == i].mean() for i in range(len(xs))])
    return xs / LX, mean


def verdict_side(point, low, high, threshold=1.0):
    """Independent re-implementation of the producer's classification (not imported)."""
    if low > threshold:
        return "above"
    if high < threshold:
        return "below"
    return "straddles"


def criterion(e_sgs, low_sgs, high_sgs, p_sgs, e_wale, low_wale, high_wale, p_wale):
    """Outcome-neutral invariance test: does the SGS model CHANGE the verdict?

    Deliberately does not assert which side of the DNS-RMS threshold the runs land on.
    The 2026-08-25 truth-reference correction moved the Re-5600 coupled verdict from
    'above' to 'below'; a criterion hard-coding either would be a gate encoding an
    outcome (HANDOVER_20260825 section 5.1).
    """
    return {
        "A1_point_estimate_same_side_as_wale": (e_sgs > 1.0) == (e_wale > 1.0),
        "A1_interval_classification_matches_wale":
            verdict_side(e_sgs, low_sgs, high_sgs) == verdict_side(e_wale, low_wale, high_wale),
        "A1_exact_test_conclusion_matches_wale": (p_sgs <= ALPHA) == (p_wale <= ALPHA),
        "A2_ratio_within_class": RATIO_LOW <= e_sgs / e_wale <= RATIO_HIGH,
        "A2_change_below_half_threshold_margin":
            abs(e_sgs - e_wale) < MARGIN_FRACTION * abs(e_wale - 1.0),
    }


def main() -> int:
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    certificates = sorted(RESULTS.glob("m4_sgs_sensitivity_*.json"))
    check("certificate present", bool(certificates))
    cert_json = certificates[-1]
    cert_npz = cert_json.with_suffix(".npz")
    check("npz companion present", cert_npz.is_file())
    cert = json.loads(cert_json.read_text())
    archive = np.load(cert_npz)
    # Outcome-neutral status check.  This verifier certifies that the M4 MEASUREMENT is
    # complete, reproducible and correctly labelled - not that the answer came out one
    # way.  A certificate that claims invariance it does not have fails here; a
    # certificate that honestly reports non-invariance passes and the operator reads the
    # measured verdict off the summary line.
    table_preview = cert["table"]
    measured_all_invariant = all(r["verdict_invariant"] for r in table_preview.values())
    expected_status = ("M4_SGS_SENSITIVITY_OK" if cert["complete"] and measured_all_invariant
                       else "M4_SGS_SENSITIVITY_INCOMPLETE" if not cert["complete"]
                       else "M4_SGS_SENSITIVITY_VERDICT_NOT_INVARIANT")
    check(f"status label matches the measured table ({cert['status']})",
          cert["status"] == expected_status)
    check("certificate does not claim invariance it lacks",
          not (cert["status"] == "M4_SGS_SENSITIVITY_OK" and not measured_all_invariant))
    check("matrix complete", cert["complete"] and not cert["missing_cases"])
    check("row identity", cert["row"] == "M4")

    for relative, expected in cert["source_hashes"].items():
        path = ROOT / relative
        if not path.is_file() and "development/nodes/" in relative:
            # the pipeline rotates development/nodes/ into development/exhausted_*/;
            # accept any byte-identical copy of the same file name
            name = Path(relative).name
            for candidate in [ROOT / "codes" / "figures" / "node_generators" / name,
                              *sorted(ROOT.glob(f"development/exhausted_*/nodes/*/{name}"), reverse=True)]:
                if candidate.is_file():
                    path = candidate
                    break
        check(f"locked input {relative}", path.is_file() and sha256(path) == expected)
    check("corrected WALE baseline present", (WALE_ROOT / "CAMPAIGN_MANIFEST.json").is_file())
    check("bulk-velocity convention recorded", "crest" in cert.get("bulk_velocity_convention", ""))

    truth = np.asarray(archive["truth_tau_s_dense"], float)
    dense = np.arange(DENSE_N) / DENSE_N
    check("dense phase grid", np.allclose(archive["dense_phase"], dense) and truth.shape == (DENSE_N,))
    truth_rebuilt = periodic_interp(archive["truth_phase"], archive["truth_tau_s"], dense)
    check("dense truth rebuilt from the stored reference stations", np.allclose(truth_rebuilt, truth))
    # The wall-traction reference must be the MGLET deposit, NOT the withdrawn 4-point
    # through-origin fit on the Xiao velocity archive.  Reload it from the raw ERCOFTAC
    # file here, independently of the producer's module.
    mglet = np.loadtxt(MGLET_WALL)
    while len(mglet) and np.all(mglet[-1, 1:] == 0.0):
        mglet = mglet[:-1]
    mglet_dense = periodic_interp(np.mod(mglet[:, 0] / 9.0, 1.0), mglet[:, 1], dense)
    check("primary reference is the MGLET DNS deposit, rebuilt from the raw file",
          np.allclose(mglet_dense, truth, rtol=0, atol=1e-12))
    check("withdrawn linear-4 Xiao estimator is not the primary reference",
          cert["dns_tangent_reconstruction_audit"]["primary_reference"] == "B_mglet_deposited")
    check("reference sensitivity covers primary, bracket and superseded",
          {b["role"] for b in cert["reference_sensitivity"].values()} == {"primary", "bracket", "superseded"})

    table = cert["table"]
    alternative = sorted({row["sgs"] for row in table.values()})
    check(">= 2 alternative SGS closures", len(alternative) >= 2 and "WALE" not in alternative)
    for sgs in alternative:
        for grid in ("G1c", "G2c"):
            for model in MODELS:
                check(f"case present {sgs}:{grid}:{model}", f"{sgs}:{grid}:{model}" in table)

    raw_p = {}
    for label, row in table.items():
        sgs, grid, model = label.split(":")
        prefix = f"{sgs}_{grid}_{model}"
        pred = np.asarray(archive[f"{prefix}_tau_s_dense"], float)
        pred_rebuilt = periodic_interp(archive[f"{prefix}_phase"], archive[f"{prefix}_tau_s"], dense)
        check(f"{label} dense prediction rebuilt", np.allclose(pred, pred_rebuilt))
        e = relative_rms(pred, truth)
        check(f"{label} E_tau recomputed", math.isclose(e, row["relative_rms"], rel_tol=1e-9))
        # WALE reference: the archived corrected-baseline curve and its raw bundle
        wale_curve = np.asarray(archive[f"WALE_{grid}_{model}_tau_s_dense"], float)
        e_wale = relative_rms(wale_curve, truth)
        check(f"{label} WALE reference E_tau recomputed", math.isclose(e_wale, row["wale_relative_rms"], rel_tol=1e-9))
        wale_row = cert["wale_reference_table"][f"WALE:{grid}:{model}"]
        wale_dir = None
        for p in WALE_ROOT.iterdir():
            if (p / "MANIFEST.json").is_file():
                m = json.loads((p / "MANIFEST.json").read_text())
                if (m["grid"], m["model"]) == (grid, model):
                    wale_dir, wale_manifest = p, m
        check(f"{label} corrected WALE bundle present", wale_dir is not None)
        check(f"{label} WALE bundle is crest-bulk corrected at Re 5600",
              int(wale_manifest["Re_H"]) == 5600
              and abs(bundle_ubar(wale_dir, wale_manifest) - EXPECTED_UBAR) < 2e-3
              and wale_manifest["producer_job_id"] == wale_row["producer_job_id"])
        wale_checkpoint = (wale_dir / "checkpoint_times_l2.txt").read_text().split()[-1]
        wp, wt = raw_tangent_traction(wale_dir, wale_checkpoint)
        e_wale_raw = relative_rms(periodic_interp(wp, wt, dense), truth)
        check(f"{label} WALE raw analytic-tangent rebuild within 2% ({e_wale_raw:.4f} vs {e_wale:.4f})",
              abs(e_wale_raw - e_wale) / e_wale < 0.02)
        check(f"{label} WALE crest bulk velocity ~1 ({wale_row['crest_bulk_velocity']:.4f})",
              abs(wale_row["crest_bulk_velocity"] - 1.0) < CREST_TOLERANCE)
        # Outcome-neutral: record WHICH side the corrected WALE baseline lands on and
        # require the certificate to report the same classification.  Do not require a
        # failure: the corrected reference moved the Re-5600 verdict, and M4 certifies
        # invariance across SGS, not the direction of the verdict.
        w_iv = row["wale_relative_rms_interval_95"]
        check(f"{label} WALE verdict side recomputed ({verdict_side(e_wale, w_iv['low'], w_iv['high'])})",
              verdict_side(e_wale, w_iv["low"], w_iv["high"]) == row["wale_verdict_side"])
        # exact block test
        diff = (pred - truth) ** 2 - truth ** 2
        blocks = [np.mean(diff[i * BLOCK:(i + 1) * BLOCK]) for i in range(DENSE_N // BLOCK)]
        p1 = exact_sign_p(blocks)
        check(f"{label} exact sign-flip p", math.isclose(p1, row["p_one_sided_failure"], abs_tol=1e-12))
        raw_p[label] = p1
        # bootstrap quantiles from the stored samples
        samples = np.asarray(archive[f"{prefix}_bootstrap_relative_rms"], float)
        lo, med, hi = np.quantile(samples, (0.025, 0.5, 0.975))
        iv = row["relative_rms_interval_95"]
        check(f"{label} interval quantiles", all(math.isclose(a, b, rel_tol=1e-9) for a, b in
                                                 ((lo, iv["low"]), (med, iv["median"]), (hi, iv["high"]))))
        check(f"{label} bootstrap draws >= 20000", samples.shape[0] >= 20000)
        wale_samples = np.asarray(archive[f"WALE_{grid}_{model}_bootstrap_relative_rms"], float)
        dlo, dmed, dhi = np.quantile(samples - wale_samples, (0.025, 0.5, 0.975))
        div = row["sgs_minus_wale_interval_95"]
        check(f"{label} paired SGS-WALE interval", math.isclose(dlo, div["low"], rel_tol=1e-9)
              and math.isclose(dhi, div["high"], rel_tol=1e-9))
        # raw rebuild with the analytic tangent from the campaign bundle
        manifest_dir = None
        for p in CAMPAIGN.iterdir():
            if (p / "MANIFEST.json").is_file():
                m = json.loads((p / "MANIFEST.json").read_text())
                if (m["sgs_model"], m["grid"], m["model"]) == (sgs, grid, model):
                    manifest_dir, manifest = p, m
        check(f"{label} bundle present", manifest_dir is not None)
        checkpoint = (manifest_dir / "checkpoint_times_l2.txt").read_text().split()[-1]
        phase_raw, tau_raw = raw_tangent_traction(manifest_dir, checkpoint)
        e_raw = relative_rms(periodic_interp(phase_raw, tau_raw, dense), truth)
        check(f"{label} raw analytic-tangent rebuild within 2% ({e_raw:.4f} vs {e:.4f})",
              abs(e_raw - e) / e < 0.02)
        # the swap was real and the numerics were otherwise the WALE ones
        check(f"{label} momentumTransport names {sgs}",
              re.search(r"\bmodel\s+" + sgs + r"\s*;", (manifest_dir / "input" / "momentumTransport").read_text()) is not None)
        check(f"{label} deposited WALE dictionary was the start point",
              manifest.get("deposited_wale_momentumTransport_sha256") == "0147bd493928af1713ccd974703c3ea3378c24dc41e4fda69d44636c293b62b8")
        log = (manifest_dir / "log.pimpleFoam").read_text(errors="replace")
        check(f"{label} solver selected {sgs}", f"Selecting LES turbulence model {sgs}" in log)
        check(f"{label} cubeRootVol filter width retained", "Selecting LES delta type cubeRootVol" in log)
        check(f"{label} averaging window 135-405", manifest["average_start"] == 135.0 and manifest["latest_time"] == 405.0)
        check(f"{label} crest-bulk corrected Re 5600",
              int(manifest["Re_H"]) == 5600 and abs(float(manifest["nu"]) - 1.0 / 5600.0) < 1e-9
              and abs(bundle_ubar(manifest_dir, manifest) - EXPECTED_UBAR) < 2e-3)
        sc = cert["matching_surface_check"][label]
        check(f"{label} crest bulk velocity ~1 ({sc['crest_bulk_velocity']:.4f})",
              abs(sc["crest_bulk_velocity"] - 1.0) < CREST_TOLERANCE)
        # Ubar is written to 6 significant figures from each mesh's own checkMesh volume
        # (114.35901 vs 114.35908 -> 0.721045 vs 0.721046), so compare at 5e-6, three
        # orders tighter than the 2e-3 physical guard above.
        check(f"{label} same nu and Ubar as the WALE reference",
              abs(sc["volume_average_Ubar"] - bundle_ubar(wale_dir, wale_manifest)) < 5e-6
              and abs(sc["nu"] - float(wale_manifest["nu"])) < 1e-12)
        ubar_match = re.search(r"Ubar\s+\(\s*([0-9.eE+-]+)\s+0\s+0\s*\)", (manifest_dir / "input" / "fvConstraints").read_text())
        check(f"{label} fvConstraints carries the crest-bulk Ubar",
              ubar_match is not None and abs(float(ubar_match.group(1)) - EXPECTED_UBAR) < 2e-3)
        check(f"{label} Courant bound", manifest["maximum_courant"] <= 0.56)
        check(f"{label} matching surface and mesh are the WALE ones",
              cert["matching_surface_check"][label]["max_relative_ym_mismatch_vs_wale"] < 1e-9
              and cert["matching_surface_check"][label]["polyMesh_points_sha256_equals_wale_case"] is True)
        check(f"{label} fvSchemes convection for U unchanged",
              "div(phi,U)      Gauss LUST grad(U);" in (manifest_dir / "input" / "fvSchemes").read_text())
        check(f"{label} controlDict maxCo 0.5",
              re.search(r"maxCo\s+0\.5\s*;", (manifest_dir / "input" / "controlDict").read_text()) is not None)

    # Holm per (sgs, grid) family of two models, as in the WALE certificate
    for sgs in alternative:
        for grid in ("G1c", "G2c"):
            family = {f"{sgs}:{grid}:{m}": raw_p[f"{sgs}:{grid}:{m}"] for m in MODELS}
            adjusted = holm(family)
            for label, value in adjusted.items():
                check(f"{label} Holm(2 models)", math.isclose(value, table[label]["p_one_sided_failure_holm"], abs_tol=1e-12))
    # acceptance recomputed
    for label, row in table.items():
        iv, wiv = row["relative_rms_interval_95"], row["wale_relative_rms_interval_95"]
        recomputed = criterion(row["relative_rms"], iv["low"], iv["high"], row["p_one_sided_failure_holm"],
                               row["wale_relative_rms"], wiv["low"], wiv["high"], row["wale_p_one_sided_failure"])
        check(f"{label} criterion recomputed", recomputed == row["criterion"])
        check(f"{label} invariance flag matches its own criterion "
              f"({'invariant' if row['verdict_invariant'] else 'NOT invariant'})",
              row["verdict_invariant"] == all(recomputed.values()))

    # power: the SGS swap changed the wall-adjacent eddy viscosity
    power = cert["sgs_power_check_first_cell_nut"]
    for sgs in alternative:
        ratios = [abs(v["median_ratio_to_wale"] - 1.0) for k, v in power.items() if k.startswith(sgs + ":")]
        check(f"{sgs} changes first-cell nut by > 10% somewhere (max {max(ratios):.2f})", max(ratios) > 0.10)

    # The per-closure answer must be the same on all three wall-traction references, so
    # neither verdict can be an artefact of the 2026-08-25 reference change.
    per_closure = {}
    for sgs in alternative:
        rows = [r for k, r in table.items() if k.startswith(sgs + ":")]
        per_closure[sgs] = all(r["verdict_invariant"] for r in rows)
        for reference_name, block in cert["reference_sensitivity"].items():
            same = {rr["verdict_invariant"] for k, rr in block["rows"].items() if k.startswith(sgs + ":")}
            check(f"{sgs}: verdict on {reference_name} is unanimous across the four cases", len(same) == 1)
            check(f"{sgs}: {reference_name} agrees with the primary reference "
                  f"({'invariant' if per_closure[sgs] else 'NOT invariant'})",
                  same == {per_closure[sgs]})

    # red/green fixtures on the criterion code, in both verdict directions
    flipped = criterion(1.4, 1.2, 1.6, 0.004, 0.6, 0.4, 0.8, 0.9)
    check("red fixture: an SGS that flips the side of the threshold is not invariant",
          not flipped["A1_point_estimate_same_side_as_wale"]
          and not flipped["A1_interval_classification_matches_wale"]
          and not flipped["A1_exact_test_conclusion_matches_wale"])
    far = criterion(6.0, 5.0, 7.0, 0.004, 3.0, 2.0, 4.0, 0.004)
    check("red fixture: doubled error leaves the magnitude class", not far["A2_ratio_within_class"])
    decisive = criterion(0.95, 0.7, 1.1, 0.5, 0.6, 0.4, 0.8, 0.9)
    check("red fixture: an SGS change larger than half the threshold margin is not invariant",
          not decisive["A2_change_below_half_threshold_margin"])
    below = criterion(0.62, 0.45, 0.80, 0.9, 0.60, 0.43, 0.78, 0.9)
    check("green fixture: matched pair below the threshold is invariant", all(below.values()))
    above = criterion(3.05, 1.7, 3.3, 0.0078, 3.0, 1.65, 3.28, 0.0078)
    check("green fixture: matched pair above the threshold is invariant", all(above.values()))

    verdict = ", ".join(f"{s}={'INVARIANT' if v else 'NOT-INVARIANT'}" for s, v in sorted(per_closure.items()))
    print(f"M4_VERIFY_OK checks={len(checks)} certificate={cert_json.name} "
          f"cases={len(table)} reference={cert['dns_tangent_reconstruction_audit']['primary_reference']} "
          f"verdict[{verdict}] status={cert['status']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"M4_VERIFY_FAIL {error}")
        sys.exit(1)
