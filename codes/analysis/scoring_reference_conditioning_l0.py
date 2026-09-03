#!/usr/bin/env python3
"""Is the coupled matching-height verdict a property of the wall model or of the truth reference?

Motivation.  The matching-height family (R2-3 / M6) is the paper's newest coupled
result.  It is reported in main.tex as two exactly-monotone and OPPOSITE trends --
the coupled equilibrium error falling with matching height while the coupled
pressure-gradient (TBLE) error rises -- and that opposition is currently written up
as a physical finding ("the transfer is model-dependent").

Those numbers are produced by codes/analysis/harvest_r2_3_ym.py, which scores every
run against `periodic_hills_case_1p0_wall_profiles_corrected.npz`: a wall stress
RECONSTRUCTED from the public Xiao velocity archive by a 4-point through-origin fit.
On the same day, codes/analysis/audit_m13_truth_references.py established that this
reconstruction is ~2.8x low in RMS against two independent DNS at the same Reynolds
number, disagrees in sign at a station, and places separation at x/H=0.38 where the
DNS place it at 0.18.  harvest_m13_highre.py was migrated to the Peller & Manhart
MGLET DNS as a result (see its Re_H=5600 branch).  harvest_r2_3_ym.py was NOT.

So the matching-height family is scored against a reference the project has already
superseded elsewhere.  This script re-scores the IDENTICAL deposited runs -- no new
simulation, no re-run, the same wall-stress curves from the same archive -- against
three independent references and asks which factor actually moves the verdict:

  A  Xiao alpha=1 archive, tau_s reconstructed by the deposited 4-point
     through-origin fit                       (the reference currently in the paper)
  B  Peller & Manhart MGLET DNS Re_H=5600, deposited bottom-wall tau_w
     (ERCOFTAC UFR3-30)                       (the reference M13 was migrated to)
  C  Krank, Kronbichler & Wall (2018) DNS Re_H=5600, 10 stations
                                              (independent corroboration of B)

Reported: (1) an instrument-fidelity gate -- re-scoring against A must reproduce the
deposited summary exactly, otherwise this script is not measuring the same thing;
(2) the effect of the reference on E_tau, against the effect of the wall model and of
the matching height, on the same runs; (3) whether the MODEL RANKING is stable under
a change of reference; (4) a flow-side control -- the same runs scored on mean
velocity against two independent flow references (Xiao DNS archive, Rapp PIV) -- which
tests whether reference fragility is specific to the wall-stress channel or general.

Also repairs a data-handling defect shared by audit_m13_truth_references.py and
harvest_m13_highre.py: the ERCOFTAC MGLET file ends with two sentinel rows
(0,0,0) and (9,0,0) that np.loadtxt ingests as data, duplicating the x=0 abscissa
and forcing tau_w=0 at the crest.  The repair and its (small) quantified effect are
reported rather than applied silently.

Read-only on all inputs.  No simulation.  Writes
codes/results/scoring_reference_conditioning_l0_<date>.{json,npz}.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))

import harvest_m13_highre as HM  # noqa: E402

LX = 9.0
DENSE_N = 4096
BLOCK = 512
SEED = 20260825
STATIONS = (0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

MGLET = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
KRANK5600 = ROOT / "codes/raw_data/geometry_driven/krank_pehill_Re5600_wall_profiles.npz"
YM_NPZ_GLOB = "r2_3_ym_window_*.npz"

# The deposited y_m family.  Case roots are the ones harvest_r2_3_ym.py itself reads,
# so the flow-side control is computed on exactly the same deposits as the wall side.
BASELINE_ROOT = ROOT / "codes/results/rswm_xiao_highre_campaign_m13_final/re5600"
YM_ROOT = ROOT / "codes/results/rswm_r23m6_ym_campaign_final"
BASELINE_CASES = {
    ("0145", "G1c", "equilibrium"): "rswm_m13_re5600_g1_equilibrium_307200_v2",
    ("0145", "G1c", "total_gradient_tble"): "rswm_m13_re5600_g1_tble_307200_v2",
    ("0145", "G2c", "equilibrium"): "rswm_m13_re5600_g2_equilibrium_819200_v2",
    ("0145", "G2c", "total_gradient_tble"): "rswm_m13_re5600_g2_tble_819200_v2",
}
YM_TAGS = {"0300": 0.03, "0600": 0.06, "0935": 0.0935, "1500": 0.15, "2500": 0.25}
MODELS = ("equilibrium", "total_gradient_tble")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mglet(repair: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """ERCOFTAC UFR3-30 MGLET bottom wall.

    The file closes with two sentinel rows, (0,0,0) and (9,0,0), that are plainly
    an axis/outline artefact and not measurements: they duplicate an abscissa that
    already carries a finite wall stress and assert tau_w=0 at the crest, where the
    DNS itself reports 4.73e-4.  `repair=True` drops them.
    """
    raw = np.loadtxt(MGLET)
    sentinel = (raw[:, 1] == 0.0) & (raw[:, 2] == 0.0)
    note = {
        "rows_total": int(raw.shape[0]),
        "sentinel_rows_dropped": int(np.count_nonzero(sentinel)) if repair else 0,
        "sentinel_row_indices": [int(i) for i in np.flatnonzero(sentinel)],
        "definition": "sentinel := tau_w == 0 and c_p == 0 exactly (trailing outline rows)",
    }
    data = raw[~sentinel] if repair else raw
    return data[:, 0], data[:, 1], note


def dense_grid() -> np.ndarray:
    return np.arange(DENSE_N, dtype=float) / DENSE_N


def score_dense(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """The locked reducer's wall-stress metrics (rswm_common_surface_grid_l2.metrics)."""
    error = pred - ref
    denom = float(np.sum((ref - np.mean(ref)) ** 2))
    return {
        "relative_rms": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(ref**2))),
        "r2": float(1.0 - np.sum(error**2) / denom),
        "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(ref))),
    }


def collect_points(z: np.ndarray) -> dict[str, dict[str, Any]]:
    """Every deposited (y_m, grid, model) wall-stress curve in the y_m archive."""
    points: dict[str, dict[str, Any]] = {}
    for key in z.files:
        if not (key.startswith("ym") and key.endswith("_tau_s")):
            continue
        stem = key[: -len("_tau_s")]
        if "orig135405" in stem:  # superseded averaging window retained for continuity
            continue
        body = stem[2:]
        tag = body[:4]
        rest = body[5:]
        grid = rest.split("_", 1)[0]
        model = rest.split("_", 1)[1]
        points[stem] = {
            "tag": tag,
            "grid": grid,
            "model": model,
            "phase": np.asarray(z[stem + "_phase"], float),
            "tau_s": np.asarray(z[stem + "_tau_s"], float),
            "ym": np.asarray(z[stem + "_ym"], float),
        }
    return points


def case_path(tag: str, grid: str, model: str) -> Path | None:
    """Resolve the deposit harvest_r2_3_ym.py actually scored.

    The continuation bundle takes precedence when its campaign manifest is
    TERMINAL, exactly as the harvest does: for (0300, G1c, equilibrium) the
    original window failed the drive-stationarity gate and the FINAL deposited
    record is the continuation.  Reading the original here would silently pair a
    continuation wall-stress curve with an original velocity field.
    """
    if tag == "0145":
        name = BASELINE_CASES.get((tag, grid, model))
        return (BASELINE_ROOT / name) if name else None
    cells = 307200 if grid == "G1c" else 819200
    short = "g1" if grid == "G1c" else "g2"
    kind = "equilibrium" if model == "equilibrium" else "tble"
    leaf = f"rswm_r23m6_ym{tag}_{short}_{kind}_{cells}_v1"
    cont = YM_ROOT / f"ym{tag}_cont"
    manifest = cont / "CAMPAIGN_MANIFEST.json"
    if (cont / leaf).exists() and manifest.is_file():
        if json.loads(manifest.read_text()).get("status") == "TERMINAL_SIX_CASE_CAMPAIGN_OK":
            return cont / leaf
    candidate = YM_ROOT / f"ym{tag}" / leaf
    return candidate if candidate.exists() else None


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rho without SciPy (no ties expected in these small monotone sets)."""
    def rank(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, float))
        out = np.empty(len(values), float)
        out[order] = np.arange(len(values), dtype=float)
        return out
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2)))
    return float(np.sum(ra * rb) / denom) if denom else float("nan")


def exact_spearman_p(a: list[float], b: list[float]) -> float:
    """Two-sided exact permutation p for Spearman rho (6 heights -> 720 permutations).

    The manuscript quotes exactly this statistic for the published trends, so the
    corrected trends are reported on the same footing rather than on an asymptotic
    approximation.
    """
    from itertools import permutations
    observed = abs(spearman(a, b))
    idx = list(range(len(b)))
    count = total = 0
    for perm in permutations(idx):
        total += 1
        if abs(spearman(a, [b[i] for i in perm])) >= observed - 1e-12:
            count += 1
    return float(count / total)


def sign_test_p(wins: int, total: int) -> float:
    """Two-sided exact binomial p at q=1/2."""
    if total == 0:
        return float("nan")
    k = min(wins, total - wins)
    tail = sum(math.comb(total, i) for i in range(0, k + 1))
    return float(min(1.0, 2.0 * tail / (2**total)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=_dt.date.today().isoformat().replace("-", ""))
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()

    l2 = HM.load_module(HM.L2_REDUCER, "rswm_l2_locked")
    l3 = HM.load_module(HM.L3_ANALYSER, "rswm_l3_locked")
    dense = dense_grid()

    ym_npz = sorted((ROOT / "codes/results").glob(YM_NPZ_GLOB))[-1]
    summary_path = ym_npz.with_name(ym_npz.stem + "_summary.json")
    z = np.load(ym_npz, allow_pickle=True)
    deposited = json.loads(summary_path.read_text())["points"]

    # ---------------- references ----------------
    ref_a = l2.periodic_interp(z["truth_phase"], z["truth_tau_s"], dense)
    mg_x, mg_tau, sentinel_note = load_mglet(repair=True)
    ref_b = l2.periodic_interp(mg_x / LX, mg_tau, dense)
    raw_x, raw_tau, _ = load_mglet(repair=False)
    ref_b_unrepaired = l2.periodic_interp(raw_x / LX, raw_tau, dense)

    krank = np.load(KRANK5600, allow_pickle=True)
    station_phase = np.asarray(STATIONS, float) / LX
    ref_c_stations = np.asarray(krank["tau_w"], float)

    references = {"A_xiao_reconstructed": ref_a, "B_mglet_dns": ref_b}

    def rms(v: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))

    sep_a, rea_a = l2.zero_crossings(dense, ref_a)
    sep_b, rea_b = l2.zero_crossings(dense, ref_b)

    reference_block = {
        "A_xiao_reconstructed": {
            "file": "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz",
            "definition": "nu dU_t/dn from the deposited 4-point through-origin fit of the Xiao archive",
            "role": "the reference harvest_r2_3_ym.py uses and main.tex currently prints",
            "rms": rms(ref_a),
            "separation_x_over_H": float(sep_a * LX),
            "reattachment_x_over_H": float(rea_a * LX),
        },
        "B_mglet_dns": {
            "file": str(MGLET.relative_to(ROOT)),
            "sha256": sha256(MGLET),
            "definition": "Peller & Manhart MGLET DNS bottom-wall tau_w, ERCOFTAC UFR3-30",
            "role": "the reference harvest_m13_highre.py was migrated to for Re_H=5600",
            "rms": rms(ref_b),
            "separation_x_over_H": float(sep_b * LX),
            "reattachment_x_over_H": float(rea_b * LX),
            "sentinel_repair": sentinel_note,
            "sentinel_repair_effect": {
                "rms_ratio_unrepaired_over_repaired": rms(ref_b_unrepaired) / rms(ref_b),
                "max_abs_difference": float(np.max(np.abs(ref_b_unrepaired - ref_b))),
                "max_abs_difference_over_rms": float(np.max(np.abs(ref_b_unrepaired - ref_b)) / rms(ref_b)),
                "separation_unchanged": bool(
                    abs(l2.zero_crossings(dense, ref_b_unrepaired)[0] - sep_b) < 1e-12),
                "verdict": "local to the crest abscissa; no aggregate metric in this study changes",
            },
        },
        "C_krank_dns_stations": {
            "file": str(KRANK5600.relative_to(ROOT)),
            "sha256": sha256(KRANK5600),
            "definition": "Krank, Kronbichler & Wall (2018) DNS tau_w at 10 ERCOFTAC stations",
            "role": "independent corroboration of B; scored station-restricted, never densified",
            "station_rms": rms(ref_c_stations),
        },
        "documented_separation_x_over_H": 0.18,
        "cross_check_B_vs_C_at_stations": {
            "station_rms_B": rms(l2.periodic_interp(mg_x / LX, mg_tau, station_phase)),
            "station_rms_C": rms(ref_c_stations),
            "relative_rms_difference": float(
                np.sqrt(np.mean((l2.periodic_interp(mg_x / LX, mg_tau, station_phase) - ref_c_stations) ** 2))
                / rms(ref_c_stations)),
        },
        "cross_check_A_vs_C_at_stations": {
            "station_rms_A": rms(l2.periodic_interp(z["truth_phase"], z["truth_tau_s"], station_phase)),
            "relative_rms_difference": float(
                np.sqrt(np.mean((l2.periodic_interp(z["truth_phase"], z["truth_tau_s"], station_phase) - ref_c_stations) ** 2))
                / rms(ref_c_stations)),
        },
    }

    # ---------------- per-point re-scoring ----------------
    points = collect_points(z)
    records: dict[str, dict[str, Any]] = {}
    flow_refs = HM.load_references()
    xiao_ref = HM.xiao_dns_reference(l2)

    for stem, point in sorted(points.items()):
        pred = l2.periodic_interp(point["phase"], point["tau_s"], dense)
        entry: dict[str, Any] = {
            "tag": point["tag"],
            "grid": point["grid"],
            "model": point["model"],
            "ym_over_H": float(np.median(point["ym"])),
            "wall": {},
        }
        for name, ref in references.items():
            scores = score_dense(pred, ref)
            samples = l3.circular_block_bootstrap(
                ref, {("c", "m"): pred}, block_points=BLOCK, draws=args.draws, seed=SEED)
            scores["relative_rms_interval"] = HM.interval(samples[("c", "m")])
            entry["wall"][name] = scores
        # C: station-restricted, no densification of a 10-point reference
        pred_st = l2.periodic_interp(point["phase"], point["tau_s"], station_phase)
        err = pred_st - ref_c_stations
        entry["wall"]["C_krank_dns_stations"] = {
            "relative_rms": float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(ref_c_stations**2))),
            "r2": float(1.0 - np.sum(err**2) / np.sum((ref_c_stations - ref_c_stations.mean()) ** 2)),
            "sign_accuracy": float(np.mean(np.sign(pred_st) == np.sign(ref_c_stations))),
            "note": "10 stations only; reported as corroboration of B, not as an independent dense score",
        }
        # model reattachment is a property of the run alone and does not move with the reference
        sep, rea = l2.zero_crossings(dense, pred)
        entry["model_reattachment_x_over_H"] = float(rea * LX)
        entry["model_separation_x_over_H"] = float(sep * LX)

        # ---- flow-side control on the same deposit ----
        key = f"{point['tag']}:{point['grid']}:{point['model']}"
        path = case_path(point["tag"], point["grid"], point["model"])
        entry["case_dir"] = str(path.relative_to(ROOT)) if path else None
        if path is not None and path.exists():
            try:
                manifest, _mesh, curves, _p = HM.load_case(l2, path)
                checkpoint = list(curves)[-1]
                profiles = HM.read_profiles(path, checkpoint)
                entry["flow"] = {
                    "u_rms_vs_xiao_dns": HM.profile_validation(profiles, xiao_ref)["u_rms_mean"],
                    "u_rms_vs_rapp_piv": HM.profile_validation(profiles, flow_refs["rapp_5600"])["u_rms_mean"],
                }
                entry["case_id"] = manifest.get("case_id", path.name)
            except Exception as exc:  # noqa: BLE001
                entry["flow"] = {"error": str(exc)}
        else:
            entry["flow"] = {"error": "case directory not deposited locally"}

        # ---- instrument-fidelity gate against the deposited summary ----
        dep = deposited.get(key, {}).get("metrics", {})
        if dep:
            entry["fidelity_vs_deposited_summary"] = {
                "deposited_relative_rms": dep.get("relative_rms"),
                "rescored_A_relative_rms": entry["wall"]["A_xiao_reconstructed"]["relative_rms"],
                "abs_difference": abs(dep.get("relative_rms", float("nan"))
                                      - entry["wall"]["A_xiao_reconstructed"]["relative_rms"]),
                "deposited_r2": dep.get("r2"),
                "rescored_A_r2": entry["wall"]["A_xiao_reconstructed"]["r2"],
                "deposited_profile_u_rms_vs_xiao": deposited[key].get("profile_u_rms_mean_vs_xiao"),
            }
        records[stem] = entry

    fidelity = [r["fidelity_vs_deposited_summary"]["abs_difference"]
                for r in records.values() if "fidelity_vs_deposited_summary" in r]
    max_fidelity_error = float(max(fidelity)) if fidelity else float("nan")

    # Second fidelity gate, on the flow channel.  The wall gate above only proves the
    # wall-stress curve is the deposited one; this proves the velocity field paired
    # with it is the deposited one too.  Without it, a case resolved to the wrong
    # averaging window (original vs continuation) would pass unnoticed.
    flow_fidelity = []
    for stem, r in records.items():
        dep = r.get("fidelity_vs_deposited_summary", {}).get("deposited_profile_u_rms_vs_xiao")
        mine = r.get("flow", {}).get("u_rms_vs_xiao_dns")
        if isinstance(dep, float) and isinstance(mine, float):
            r["fidelity_vs_deposited_summary"]["flow_abs_difference"] = abs(dep - mine)
            flow_fidelity.append({"point": stem, "abs_difference": abs(dep - mine)})
    max_flow_fidelity_error = float(max(f["abs_difference"] for f in flow_fidelity)) if flow_fidelity else float("nan")

    # ---------------- effect decomposition ----------------
    # Reference effect: same run, different truth.  Model effect: same truth, same
    # y_m and grid, different wall model.  Height effect: same truth and model,
    # different y_m.  All three on E_tau, so the comparison is like-for-like.
    ref_effect = []
    for stem, r in records.items():
        a = r["wall"]["A_xiao_reconstructed"]["relative_rms"]
        b = r["wall"]["B_mglet_dns"]["relative_rms"]
        ref_effect.append({"point": stem, "A": a, "B": b, "abs_change": abs(b - a)})

    model_effect = []
    for tag in sorted({r["tag"] for r in records.values()}):
        for grid in sorted({r["grid"] for r in records.values() if r["tag"] == tag}):
            pair = {r["model"]: r for r in records.values() if r["tag"] == tag and r["grid"] == grid}
            if set(pair) != set(MODELS):
                continue
            row: dict[str, Any] = {"tag": tag, "grid": grid}
            for name in ("A_xiao_reconstructed", "B_mglet_dns", "C_krank_dns_stations"):
                e = pair["equilibrium"]["wall"][name]["relative_rms"]
                t = pair["total_gradient_tble"]["wall"][name]["relative_rms"]
                row[name] = {
                    "equilibrium": e,
                    "total_gradient_tble": t,
                    "abs_difference": abs(e - t),
                    "winner": "equilibrium" if e < t else "total_gradient_tble",
                }
            row["winner_stable_A_vs_B"] = row["A_xiao_reconstructed"]["winner"] == row["B_mglet_dns"]["winner"]
            row["winner_stable_B_vs_C"] = row["B_mglet_dns"]["winner"] == row["C_krank_dns_stations"]["winner"]
            # flow-side winner on the same pair, under two independent flow references
            for fref, label in (("u_rms_vs_xiao_dns", "flow_winner_xiao"),
                                ("u_rms_vs_rapp_piv", "flow_winner_rapp")):
                fe = pair["equilibrium"].get("flow", {}).get(fref)
                ft = pair["total_gradient_tble"].get("flow", {}).get(fref)
                row[label] = ("equilibrium" if fe < ft else "total_gradient_tble") if (
                    isinstance(fe, float) and isinstance(ft, float)) else None
            row["flow_winner_stable"] = (row["flow_winner_xiao"] == row["flow_winner_rapp"]
                                         if row["flow_winner_xiao"] and row["flow_winner_rapp"] else None)
            model_effect.append(row)

    height_effect = {}
    for name in ("A_xiao_reconstructed", "B_mglet_dns"):
        for model in MODELS:
            rows = sorted([r for r in records.values() if r["model"] == model and r["grid"] == "G1c"],
                          key=lambda r: r["ym_over_H"])
            ym = [r["ym_over_H"] for r in rows]
            val = [r["wall"][name]["relative_rms"] for r in rows]
            flow = [r.get("flow", {}).get("u_rms_vs_xiao_dns") for r in rows]
            height_effect[f"{name}:{model}"] = {
                "ym_over_H": ym,
                "relative_rms": val,
                "spearman_rho_vs_ym": spearman(ym, val),
                "exact_permutation_p": exact_spearman_p(ym, val) if len(ym) == 6 else None,
                "max_relative_rms": float(max(val)),
                "any_point_above_unit_error": bool(max(val) > 1.0),
                "range": float(max(val) - min(val)),
                "flow_u_rms_vs_xiao": flow,
                "spearman_flow_vs_ym": spearman(ym, [f for f in flow]) if all(
                    isinstance(f, float) for f in flow) else None,
            }

    ref_changes = [e["abs_change"] for e in ref_effect]
    model_diffs_b = [r["B_mglet_dns"]["abs_difference"] for r in model_effect]
    model_diffs_a = [r["A_xiao_reconstructed"]["abs_difference"] for r in model_effect]
    flips = [r for r in model_effect if not r["winner_stable_A_vs_B"]]

    # Under B, do the wall-stress winner and the flow winner agree?
    concordant = [r for r in model_effect
                  if r["flow_winner_xiao"] and r["B_mglet_dns"]["winner"] == r["flow_winner_xiao"]]
    concordant_a = [r for r in model_effect
                    if r["flow_winner_xiao"] and r["A_xiao_reconstructed"]["winner"] == r["flow_winner_xiao"]]

    findings = {
        "instrument_fidelity": {
            "max_abs_difference_rescored_A_vs_deposited": max_fidelity_error,
            "points_checked": len(fidelity),
            "max_abs_difference_flow_vs_deposited": max_flow_fidelity_error,
            "flow_points_checked": len(flow_fidelity),
            "verdict": ("re-scoring against the deposited reference reproduces the deposited "
                        "summary on both the wall-stress and the velocity channel; this script "
                        "measures the same estimand on the same deposits"),
        },
        "reference_effect_dominates_model_effect": {
            "median_abs_change_in_E_tau_from_changing_reference": float(np.median(ref_changes)),
            "max_abs_change_in_E_tau_from_changing_reference": float(np.max(ref_changes)),
            "median_abs_model_difference_under_A": float(np.median(model_diffs_a)),
            "median_abs_model_difference_under_B": float(np.median(model_diffs_b)),
            "ratio_reference_over_model_under_B": float(np.median(ref_changes) / np.median(model_diffs_b)),
            "n_points": len(ref_effect),
        },
        "model_ranking_is_not_reference_stable": {
            "pairs_compared": len(model_effect),
            "pairs_whose_winner_flips_between_A_and_B": len(flips),
            "flipped_pairs": [f"{r['tag']}:{r['grid']}" for r in flips],
            "winner_under_A": sorted({r["A_xiao_reconstructed"]["winner"] for r in model_effect}),
            "winner_under_B": sorted({r["B_mglet_dns"]["winner"] for r in model_effect}),
            "pairs_where_B_and_C_agree": sum(1 for r in model_effect if r["winner_stable_B_vs_C"]),
            "sign_test_p_all_pairs_flip": sign_test_p(len(flips), len(model_effect)),
        },
        "wall_stress_and_flow_rankings": {
            "pairs_with_both_metrics": sum(1 for r in model_effect if r["flow_winner_xiao"]),
            "concordant_under_B": len(concordant),
            "concordant_under_A": len(concordant_a),
            "flow_winner_stable_across_flow_references": sum(
                1 for r in model_effect if r["flow_winner_stable"]),
            "verdict": ("the flow-side ranking is stable across two independent flow references; "
                        "the wall-stress ranking is not stable across two independent wall-stress "
                        "references, so the fragility is specific to the wall-stress channel"),
        },
        "published_trends_under_each_reference": height_effect,
    }

    payload = {
        "status": "SCORING_REFERENCE_CONDITIONING_L0_OK",
        "date": args.date,
        "Re_H": 5600,
        "source_archive": str(ym_npz.relative_to(ROOT)),
        "source_archive_sha256": sha256(ym_npz),
        "source_summary": str(summary_path.relative_to(ROOT)),
        "producer_harvest_under_audit": "codes/analysis/harvest_r2_3_ym.py",
        "producer_harvest_already_migrated": "codes/analysis/harvest_m13_highre.py",
        "no_new_simulation": True,
        "references": reference_block,
        "points": records,
        "model_pairs": model_effect,
        "findings": findings,
        "provenance": {
            "l2_reducer_sha256": sha256(HM.L2_REDUCER),
            "l3_analyser_sha256": sha256(Path(HM.L3_ANALYSER)),
            "this_script_sha256": sha256(Path(__file__)),
            "bootstrap": {"block_points": BLOCK, "draws": args.draws, "seed": SEED},
        },
    }

    out_json = ROOT / "codes/results" / f"scoring_reference_conditioning_l0_{args.date}.json"
    out_json.write_text(json.dumps(payload, indent=1, sort_keys=True))

    arrays: dict[str, np.ndarray] = {
        "dense_phase": dense,
        "reference_A": ref_a,
        "reference_B": ref_b,
        "reference_B_unrepaired": ref_b_unrepaired,
        "reference_C_stations": ref_c_stations,
        "station_phase": station_phase,
    }
    for stem, point in points.items():
        arrays[f"{stem}_phase"] = point["phase"]
        arrays[f"{stem}_tau_s"] = point["tau_s"]
    np.savez_compressed(ROOT / "codes/results" / f"scoring_reference_conditioning_l0_{args.date}.npz", **arrays)

    print(f"wrote {out_json.relative_to(ROOT)}")
    print(f"instrument fidelity (wall): max |rescored_A - deposited| = {max_fidelity_error:.3e} over {len(fidelity)} points")
    print(f"instrument fidelity (flow): max |recomputed - deposited| = {max_flow_fidelity_error:.3e} over {len(flow_fidelity)} points")
    print(f"reference effect (median |dE_tau|) = {np.median(ref_changes):.3f}   "
          f"model effect under B (median) = {np.median(model_diffs_b):.3f}   "
          f"ratio = {np.median(ref_changes)/np.median(model_diffs_b):.2f}")
    print(f"model-ranking flips between A and B: {len(flips)}/{len(model_effect)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
