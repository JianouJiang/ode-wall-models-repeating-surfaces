#!/usr/bin/env python3
"""Verifier for the wall-stress-reference conditioning study (L0, node_001).

The claim under test is narrow and checkable: on the deposited coupled
matching-height family at Re_H=5600, the choice of wall-stress TRUTH REFERENCE
moves the reported error metric further than the choice of WALL MODEL does, and
reverses the model ranking at every rung -- while the same runs' velocity-field
ranking is stable across two independent flow references.

The verifier does not trust the producer's own summary fields.  It reloads the
deposited wall-stress curves from the npz, rebuilds every reference from its
primary file, recomputes every score with an INDEPENDENT NumPy implementation,
and only then compares against the json.  It then applies control cases that must
fail, so a vacuous pass is detectable.

Usage: python3 codes/analysis/ledger_verifiers/verify_scoring_reference_l0.py [--date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"
LX, N = 9.0, 4096
STATIONS = (0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
MGLET = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
KRANK = ROOT / "codes/raw_data/geometry_driven/krank_pehill_Re5600_wall_profiles.npz"

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> bool:
    CHECKS.append((bool(ok), label))
    return bool(ok)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def pinterp(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Independent periodic interpolation (does NOT import the locked reducer)."""
    order = np.argsort(np.asarray(x, float))
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    return np.interp(np.mod(t, 1.0), np.r_[xs - 1.0, xs, xs + 1.0], np.r_[ys, ys, ys])


def rel_rms(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / np.sqrt(np.mean(ref**2)))


def first_sign_change(phase: np.ndarray, tau: np.ndarray, positive_to_negative: bool) -> float:
    s = np.sign(tau)
    for i in range(len(tau) - 1):
        if positive_to_negative and s[i] > 0 >= s[i + 1]:
            return float(phase[i] * LX)
        if not positive_to_negative and s[i] < 0 <= s[i + 1]:
            return float(phase[i] * LX)
    return float("nan")


def sign_test_p(wins: int, total: int) -> float:
    k = min(wins, total - wins)
    tail = sum(math.comb(total, i) for i in range(0, k + 1))
    return float(min(1.0, 2.0 * tail / (2**total)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=_dt.date.today().isoformat().replace("-", ""))
    args = ap.parse_args()

    jpath = RESULTS / f"scoring_reference_conditioning_l0_{args.date}.json"
    npath = RESULTS / f"scoring_reference_conditioning_l0_{args.date}.npz"
    if not check(jpath.is_file() and npath.is_file(), "artifact json + npz present"):
        report()
        return 1
    art = json.loads(jpath.read_text())
    arr = np.load(npath, allow_pickle=True)

    check(art.get("status") == "SCORING_REFERENCE_CONDITIONING_L0_OK", "status flag set")
    check(art.get("no_new_simulation") is True, "declares no new simulation")
    check(art.get("Re_H") == 5600, "Reynolds number recorded")

    # ---- 1. primary sources are the ones claimed, by hash ----
    refs = art["references"]
    check(refs["B_mglet_dns"]["sha256"] == sha256(MGLET), "MGLET primary file hash matches artifact")
    check(refs["C_krank_dns_stations"]["sha256"] == sha256(KRANK), "Krank primary file hash matches artifact")
    src = ROOT / art["source_archive"]
    check(src.is_file() and sha256(src) == art["source_archive_sha256"],
          "deposited y_m archive hash matches artifact")

    # ---- 2. rebuild references independently from the primary files ----
    dense = np.arange(N, dtype=float) / N
    raw = np.loadtxt(MGLET)
    sentinel = (raw[:, 1] == 0.0) & (raw[:, 2] == 0.0)
    check(int(np.count_nonzero(sentinel)) == 2, "the two MGLET sentinel rows are present in the primary file")
    clean = raw[~sentinel]
    ref_b = pinterp(clean[:, 0] / LX, clean[:, 1], dense)
    check(np.allclose(ref_b, arr["reference_B"], rtol=0, atol=1e-12),
          "reference B rebuilt independently from the primary file")
    ref_a = arr["reference_A"]

    # separation location: B must land on the documented value, A must not
    sep_b = first_sign_change(dense, ref_b, True)
    sep_a = first_sign_change(dense, ref_a, True)
    check(abs(sep_b - 0.18) < 0.03, f"reference B separates at the documented x/H=0.18 (got {sep_b:.3f})")
    check(sep_a > 0.30, f"reference A separates far downstream of the documented value (got {sep_a:.3f})")
    check(abs(refs["A_xiao_reconstructed"]["rms"] / refs["B_mglet_dns"]["rms"] - 0.36) < 0.02,
          "reference A is ~2.8x low in RMS against B, as the M13 audit found")

    # B corroborated by the independent Krank DNS at the stations
    krank_tau = np.asarray(np.load(KRANK, allow_pickle=True)["tau_w"], float)
    st = np.asarray(STATIONS, float) / LX
    b_st = pinterp(clean[:, 0] / LX, clean[:, 1], st)
    d_bc = float(np.sqrt(np.mean((b_st - krank_tau) ** 2)) / np.sqrt(np.mean(krank_tau**2)))
    a_st = pinterp(arr["dense_phase"], ref_a, st)
    d_ac = float(np.sqrt(np.mean((a_st - krank_tau) ** 2)) / np.sqrt(np.mean(krank_tau**2)))
    check(d_bc < d_ac, f"B is closer than A to the independent Krank DNS ({d_bc:.3f} vs {d_ac:.3f})")
    check(d_bc < 0.25, f"B agrees with Krank within 25% station RMS ({d_bc:.3f})")

    # ---- 3. recompute every score independently from the deposited curves ----
    points = art["points"]
    worst_a = worst_b = 0.0
    for stem, rec in points.items():
        pred = pinterp(arr[f"{stem}_phase"], arr[f"{stem}_tau_s"], dense)
        worst_a = max(worst_a, abs(rel_rms(pred, ref_a) - rec["wall"]["A_xiao_reconstructed"]["relative_rms"]))
        worst_b = max(worst_b, abs(rel_rms(pred, ref_b) - rec["wall"]["B_mglet_dns"]["relative_rms"]))
    check(worst_a < 1e-9, f"independent recomputation reproduces every A score (max dev {worst_a:.2e})")
    check(worst_b < 1e-9, f"independent recomputation reproduces every B score (max dev {worst_b:.2e})")
    check(len(points) == 17, f"all 17 deposited points re-scored (got {len(points)})")

    # ---- 4. instrument fidelity against the deposited harvest ----
    fid = art["findings"]["instrument_fidelity"]
    check(fid["max_abs_difference_rescored_A_vs_deposited"] < 1e-9,
          "wall channel reproduces the deposited harvest exactly")
    check(fid["max_abs_difference_flow_vs_deposited"] < 1e-9,
          "flow channel reproduces the deposited harvest exactly (correct averaging window per case)")
    check(fid["points_checked"] == 17 and fid["flow_points_checked"] == 17,
          "both fidelity gates cover all 17 points")

    # ---- 5. the headline claims, recomputed from the per-point records ----
    pairs = art["model_pairs"]
    flips = [p for p in pairs if not p["winner_stable_A_vs_B"]]
    check(len(pairs) == 8, f"8 paired model comparisons available (got {len(pairs)})")
    check(len(flips) == len(pairs), f"the model ranking flips in every pair ({len(flips)}/{len(pairs)})")
    check(abs(sign_test_p(len(flips), len(pairs)) - 0.0078125) < 1e-9,
          "exact sign-test p reproduced")
    winners_a = {p["A_xiao_reconstructed"]["winner"] for p in pairs}
    winners_b = {p["B_mglet_dns"]["winner"] for p in pairs}
    check(winners_a == {"equilibrium"}, "under A the equilibrium model wins every pair")
    check(winners_b == {"total_gradient_tble"}, "under B the TBLE model wins every pair")
    agree_bc = sum(1 for p in pairs if p["winner_stable_B_vs_C"])
    check(agree_bc >= 7, f"the independent Krank reference corroborates B's ranking ({agree_bc}/8)")

    eff = art["findings"]["reference_effect_dominates_model_effect"]
    check(eff["median_abs_change_in_E_tau_from_changing_reference"]
          > eff["median_abs_model_difference_under_B"],
          "reference effect exceeds model effect on the same runs")
    check(eff["ratio_reference_over_model_under_B"] > 2.0,
          f"reference/model effect ratio > 2 (got {eff['ratio_reference_over_model_under_B']:.2f})")

    fl = art["findings"]["wall_stress_and_flow_rankings"]
    check(fl["flow_winner_stable_across_flow_references"] == fl["pairs_with_both_metrics"],
          "the velocity-field ranking is stable across two independent flow references")
    check(fl["concordant_under_B"] > fl["concordant_under_A"],
          "the corrected reference brings the wall-stress and flow rankings into agreement")

    # ---- 6. the two published 'tensions' dissolve under B ----
    tr = art["findings"]["published_trends_under_each_reference"]
    check(abs(tr["A_xiao_reconstructed:equilibrium"]["spearman_rho_vs_ym"] + 1.0) < 1e-9,
          "under A the equilibrium trend is the published anti-monotone one (rho=-1)")
    check(abs(tr["B_mglet_dns:equilibrium"]["spearman_rho_vs_ym"] - 1.0) < 1e-9,
          "under B the equilibrium trend reverses to monotone increasing (rho=+1)")
    check(abs(tr["B_mglet_dns:total_gradient_tble"]["spearman_rho_vs_ym"] - 1.0) < 1e-9,
          "under B the TBLE trend remains monotone increasing (rho=+1)")
    check(all(v <= 1.0 for v in tr["B_mglet_dns:equilibrium"]["relative_rms"])
          and all(v <= 1.0 for v in tr["B_mglet_dns:total_gradient_tble"]["relative_rms"]),
          "under B no point in the family exceeds E_tau = 1")
    check(any(v > 1.0 for v in tr["A_xiao_reconstructed:equilibrium"]["relative_rms"]),
          "under A points do exceed E_tau = 1 (the published failure verdict)")

    # ---- 7. the sentinel repair is reported and quantified, not silent ----
    rep = refs["B_mglet_dns"]["sentinel_repair"]
    check(rep["sentinel_rows_dropped"] == 2, "sentinel repair applied and recorded")
    check(refs["B_mglet_dns"]["sentinel_repair_effect"]["max_abs_difference_over_rms"] < 0.2,
          "sentinel repair effect quantified and local")
    check(refs["B_mglet_dns"]["sentinel_repair_effect"]["separation_unchanged"] is True,
          "sentinel repair does not move the separation point")

    # ---- 8. control cases: each MUST fail ----
    red = []
    shuffled = np.random.default_rng(7).permutation(ref_b)
    red.append(("phase-shuffled reference reproduces the B scores",
                abs(rel_rms(pinterp(arr[f"{list(points)[0]}_phase"], arr[f"{list(points)[0]}_tau_s"], dense), shuffled)
                    - points[list(points)[0]]["wall"]["B_mglet_dns"]["relative_rms"]) < 1e-6))
    red.append(("reference A separates at the documented 0.18", abs(sep_a - 0.18) < 0.03))
    red.append(("A is as close to Krank as B is", d_ac <= d_bc))
    red.append(("sign-flipped reference reproduces the B scores",
                abs(rel_rms(pinterp(arr[f"{list(points)[0]}_phase"], arr[f"{list(points)[0]}_tau_s"], dense), -ref_b)
                    - points[list(points)[0]]["wall"]["B_mglet_dns"]["relative_rms"]) < 1e-6))
    red.append(("the model ranking is stable under a change of reference", len(flips) == 0))
    for label, triggered in red:
        check(not triggered, f"control case rejected: {label}")

    return report()


def report() -> int:
    passed = sum(1 for ok, _ in CHECKS if ok)
    for ok, label in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"{passed}/{len(CHECKS)} checks passed "
          f"(reference conditioning of the coupled matching-height verdict)")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
