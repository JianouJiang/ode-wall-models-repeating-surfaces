#!/usr/bin/env python3
r"""
fig_verdict_map_real.py -- M17 (real): the verdict map rebuilt from real runs ONLY.
===================================================================================

The deleted Figs 5-6 class maps carried 18 of 28 verdict badges from a
hard-coded literal table (stage_classmaps_3d.py:197-201) rendered to look like
simulation output; referee point M17 called this the paper's second most
AI-suspicious artefact.  This generator rebuilds the map with the opposite
contract:

  * a cell carries a verdict badge ONLY if a high-fidelity run of that exact
    case exists in the campaign's deposited archives, and the cell records its
    provenance (case id, source artifact, sha256, metric, interval);
  * a run whose verdict interval straddles a bin boundary is badged UNRESOLVED
    (that is a result about the run, not a decoration);
  * every cell of the old maps with no run -- including every geometry the old
    figures badged without simulating -- is drawn explicitly EMPTY/UNTESTED.

Verdict bins and interval logic are IMPORTED from the R2-2 (real) evaluator
(epsilon_predictor_outoffamily), so both thesis-annex artifacts score every
case on identical rules.

Outputs
  work_progress/archer2_campaign_20260823/M17_real/fig_verdict_map_real.{pdf,png}
  codes/results/verdict_map_real_<date>.json   (per-cell provenance cert)

Usage:  python3 codes/figures/fig_verdict_map_real.py [--date 20260824]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
OUTDIR = ROOT / "work_progress/archer2_campaign_20260823/M17_real"
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
from epsilon_predictor_outoffamily import (r2_bin, interval_bin, latest,   # noqa: E402
                                           sha256, classify_truth,
                                           _m13_candidates)

COL = {"FAIL": "#c23b22", "MARGINAL": "#e0a800", "TOLERATED": "#2e7d32",
       "UNRESOLVED": "#5c6bc0", "UNTESTED": "#9e9e9e"}


def cell(cell_id, family, label, verdict=None, metric=None, provenance=None,
         evidence=None, note=None):
    tested = verdict is not None
    return dict(cell_id=cell_id, family=family, case_label=label,
                status="TESTED" if tested else "UNTESTED",
                verdict=verdict if tested else None,
                metric=metric, provenance=provenance, evidence=evidence, note=note)


# --------------------------------------------------------------------------
def cells_xiao():
    src = RESULTS / "dose_response_xiao.npz"
    if not src.exists():
        return []
    d = np.load(src, allow_pickle=True)
    h = sha256(src)
    out = []
    for i, case in enumerate(d["agg_case"]):
        r2 = float(d["agg_r2"][i])
        out.append(cell(
            f"hill_apriori::{case}", "smooth periodic hill -- Xiao DNS a-priori family",
            str(case), verdict=r2_bin(r2),
            metric=dict(name="a-priori standard_ml R2 (Y_IDX=10)", value=r2,
                        interval=None,
                        resolution_basis="point value; |R2|>1 margin (no archived interval)",
                        eps_median=float(d["agg_eps_median"][i])),
            provenance=dict(source=str(src.relative_to(ROOT)), sha256=h,
                            case_id=str(case)),
            evidence="a-priori on Xiao et al. DNS reference fields (archived instrument)"))
    return out


def cells_wavy():
    jsrc = latest("r1_sta2_wavy_wrles_*.json")
    fam = "smooth wavy wall (Hudson) -- WRLES, own truth"
    out = []
    if jsrc is None:
        return [cell(f"wavy::{g}", fam, f"lambda=2*delta, grid {g}",
                     note="wavy WRLES harvest not deposited") for g in ("G0", "G1", "G2")]
    art = json.loads(jsrc.read_text())
    npz = np.load(jsrc.with_suffix(".npz"), allow_pickle=True)
    h = sha256(jsrc)
    for g in ("G0", "G1", "G2"):
        label = f"a=0.05*lambda, lambda=2*delta, grid {g}"
        if g not in art.get("grids", {}):
            out.append(cell(f"wavy::{g}", fam, label,
                            note=f"grid {g} still running (jobs 14899155/14899157)"))
            continue
        od = art["grids"][g]["ode_diagnostic"]["0.1"]
        r2v = float(od["standard_ml"])
        bk = f"{g}_block_r2_standard_ml"
        if bk in npz.files:
            blocks = np.asarray(npz[bk], float)[:, 1]
            sem = float(blocks.std(ddof=1) / math.sqrt(len(blocks)))
            iv = [r2v - 2 * sem, r2v + 2 * sem]
            basis = f"block-window replicates n={len(blocks)}, +/-2*SEM"
        else:
            iv, basis = None, "point value"
        v = interval_bin(*iv) if iv else r2_bin(r2v)
        out.append(cell(
            f"wavy::{g}", fam, label, verdict=(v if v else "UNRESOLVED"),
            metric=dict(name="a-priori standard_ml R2 at eta_m=0.1*delta", value=r2v,
                        interval=iv, resolution_basis=basis,
                        eps_median=float(od["eps_median"])),
            provenance=dict(source=str(jsrc.relative_to(ROOT)), sha256=h,
                            case_id=art["grids"][g].get("case_id"),
                            slurm_job_id=art["grids"][g].get("slurm_job_id")),
            evidence="wall-resolved LES validated vs Hudson/Maass-Schumann"))
    return out


def cells_r24():
    jsrc = latest("r2_4_m20_les_*.json")
    out = []
    want = [("rib::dtype_p3", "sharp square rib -- matched WRLES pair",
             "d-type p/k=3 (pitch 0.6*delta)", "r24_rib_dtype_p3_G1"),
            ("rib::ktype_p8", "sharp square rib -- matched WRLES pair",
             "k-type p/k=8 (pitch 1.6*delta)", "r24_rib_ktype_p8_G1"),
            ("cube::aligned", "cube canopy (3-D) -- WRLES",
             "aligned lambda_p=0.25", "r24_cube_aligned_G1"),
            ("cube::staggered", "cube canopy (3-D) -- WRLES",
             "staggered lambda_p=0.25", "r24_cube_staggered_G1"),
            ("cube::sparse", "cube canopy (3-D) -- WRLES",
             "sparse lambda_p=0.0625", "r24_cube_sparse_G1")]
    art = json.loads(jsrc.read_text()) if jsrc else None
    h = sha256(jsrc) if jsrc else None
    for cid, fam, label, case in want:
        c = (art or {}).get("cases", {}).get(case)
        if c is None or c.get("status") != "OK":
            out.append(cell(cid, fam, label,
                            note=f"{case}: run still landing (jobs 14899302/03/06/07)"
                            if cid.startswith("cube") else f"{case}: not in harvest"))
            continue
        cums = [k for k in c["windows"] if k.startswith("cum_")]
        if not cums:
            out.append(cell(cid, fam, label, note=f"{case}: no cumulative window yet"))
            continue
        wname = max(cums, key=lambda k: float(k.split("_")[1]))
        w = c["windows"][wname] if c["kind"] == "rib" else c["windows"][wname]["floor"]
        ci = [float(v) for v in w["station_block_bootstrap"]["r2_ci95"]]
        r2v = float(w["standard_ml_r2"])
        v = interval_bin(*ci)
        out.append(cell(
            cid, fam, label, verdict=(v if v else "UNRESOLVED"),
            metric=dict(name=f"a-priori standard_ml R2, grid G1, window {wname}",
                        value=r2v, interval=ci,
                        resolution_basis="station moving-block bootstrap 95% CI",
                        eps_median=float(w["eps_median"])),
            provenance=dict(source=str(jsrc.relative_to(ROOT)), sha256=h, case_id=case,
                            deposit_manifest_sha256=c.get("source_manifest_sha256")),
            evidence="wall-resolved LES (WALE), Leonardi-box geometry, matched numerics"))
    return out


def cells_m13():
    jsrc = latest("m13_highre_coupled_*_summary.json")
    fam = "coupled a-posteriori WMLES -- canonical hill Re ladder"
    out = []
    art = json.loads(jsrc.read_text()) if jsrc else None
    h = sha256(jsrc) if jsrc else None
    for re_h in (5600, 10595, 19000, 37000):
        cid, label = f"hill_coupled::re{re_h}", f"Re_H={re_h}, equilibrium ODE, G2c"
        # amendment AMD-01: a withdrawn truth reference may never carry a badge
        cands = [t for t in _m13_candidates(re_h) if t[3]["status"] == "VALID"]
        if not cands:
            withdrawn = [t for t in _m13_candidates(re_h) if t[3]["status"] == "WITHDRAWN"]
            out.append(cell(cid, fam, label,
                            note=("truth reference WITHDRAWN (amendment AMD-01); "
                                  "no valid-reference score exists"
                                  if withdrawn else
                                  f"Re_H={re_h} bundle still running")))
            continue
        jsrc, art, c, truth_class = cands[0]
        h = sha256(jsrc)
        met = (c or {}).get("metrics", {}).get("G2c:equilibrium")
        if met is None:
            out.append(cell(cid, fam, label,
                            note=f"Re_H={re_h} bundle still running"
                            if re_h >= 19000 else "not in m13 harvest"))
            continue
        if "G2c:equilibrium" not in c.get("averaging", {}) or \
                "G2c:equilibrium" not in c.get("phase_bootstrap_primary_intervals", {}):
            out.append(cell(cid, fam, label,
                            note="bundle supplies event/profile truth, not the wall-traction reference"))
            continue
        wins = c["averaging"]["G2c:equilibrium"]
        wv = np.array([wins[t]["r2"] for t in ("180", "225", "270")], float)
        sem = float(wv.std(ddof=1) / math.sqrt(len(wv)))
        iv = [float(wv.mean() - 2 * sem), float(wv.mean() + 2 * sem)]
        pb = c["phase_bootstrap_primary_intervals"]["G2c:equilibrium"]
        wbin = interval_bin(*iv)
        rel_side = ("FAIL_side" if pb["low"] > 1.0 else
                    "TOLERATED_side" if pb["high"] < 1.0 else "straddles_1")
        consistent = ((wbin == "FAIL" and rel_side == "FAIL_side") or
                      (wbin in ("MARGINAL", "TOLERATED") and rel_side == "TOLERATED_side"))
        v = wbin if (wbin and consistent) else "UNRESOLVED"
        out.append(cell(
            cid, fam, label, verdict=v,
            metric=dict(name="coupled wall-traction R2 vs registered reference, longest window",
                        value=float(met["r2"]), interval=iv,
                        resolution_basis=("window replicates +/-2*SEM; relRMS bootstrap "
                                          f"[{pb['low']:.3f},{pb['high']:.3f}] {rel_side}"
                                          + ("" if consistent else " -> CONFLICTED")),
                        eps_c_median_separated=c["eps_c"]["G2c:equilibrium"]
                        ["eps_c_median_separated"]),
            provenance=dict(source=str(jsrc.relative_to(ROOT)), sha256=h,
                            case_id=c["cases"].get("G2c:equilibrium"),
                            truth_reference=truth_class["reference"],
                            truth_status=truth_class["status"]),
            evidence="coupled WMLES, corrected crest-bulk drive, matched numerics; "
                     f"truth = {truth_class['reference']}"))
    return out


def cells_never_run():
    """The old maps' geometries that have never been run at matched high fidelity.
    These stay EMPTY -- that is the whole point of M17."""
    fam = "old-map geometries with NO matched high-fidelity run"
    old = [("legacy::rounded_rib_all", "rounded rib (entire old-map row)",
            "old Fig 5 row 2: all four cells were never simulated"),
           ("legacy::gaussian_bump", "Gaussian bump (non-repeating)",
            "old Fig 5(d)/6(h): badge was hard-coded"),
           ("legacy::bfs_matched", "backward-facing step at matched WRLES fidelity",
            "old Fig 5(l) HOLD badge came from reference-data a-priori, not a matched run"),
           ("legacy::convdiv", "converging-diverging channel",
            "old Fig 6(k) HOLD badge was hard-coded"),
           ("legacy::krank_hill", "Krank hill at matched WRLES fidelity",
            "old Fig 5(c)/6(g) HOLD badges were hard-coded")]
    return [cell(cid, fam, label, note=note) for cid, label, note in old]


# --------------------------------------------------------------------------
def draw(cert, fig_pdf, fig_png):
    fams = []
    for c in cert["cells"]:
        if c["family"] not in fams:
            fams.append(c["family"])
    lane_h, pad = 1.0, 0.12
    fig_h = 1.8 + lane_h * len(fams)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(fams) * lane_h + 0.6)
    ax.axis("off")
    ax.set_title("Wall-model verdict map -- REAL RUNS ONLY "
                 f"({cert['status']}, {cert['date']})\n"
                 "every badge binds to a deposited dataset (sha256 in the cert); "
                 "unrun cells are explicitly UNTESTED",
                 fontsize=13, pad=14)
    import textwrap
    for fi, fam in enumerate(fams):
        y0 = (len(fams) - 1 - fi) * lane_h + 0.3
        cells_f = [c for c in cert["cells"] if c["family"] == fam]
        ax.text(0.002, y0 + lane_h * 0.44, "\n".join(textwrap.wrap(fam, 30)),
                fontsize=10, fontweight="bold", va="center", ha="left")
        n = len(cells_f)
        compact = n > 12                      # the 29-member hill family
        x0, x1 = 0.30, 0.995
        wcell = (x1 - x0) / n
        for ci_, c in enumerate(cells_f):
            xa = x0 + ci_ * wcell
            v = c["verdict"] if c["status"] == "TESTED" else "UNTESTED"
            fc = COL[v]
            style = dict(boxstyle="round,pad=0.006",
                         fc=(fc if c["status"] == "TESTED" else "white"),
                         ec=fc, lw=1.2,
                         ls="-" if c["status"] == "TESTED" else (0, (3, 2)))
            ax.add_patch(FancyBboxPatch((xa + 0.06 * wcell, y0 + 0.06),
                                        wcell * 0.88, lane_h * 0.75,
                                        mutation_scale=1, **style))
            tcol = "white" if c["status"] == "TESTED" else COL["UNTESTED"]
            if compact:
                ax.text(xa + 0.5 * wcell, y0 + 0.06 + lane_h * 0.375,
                        c["case_label"].replace("alph", "").replace("-", "\n"),
                        fontsize=3.6, ha="center", va="center", color=tcol)
            else:
                m = c.get("metric") or {}
                iv = m.get("interval")
                lines = [c["case_label"], v]
                if m.get("value") is not None:
                    lines.append(f"R2={m['value']:+.2f}")
                    if iv:
                        lines.append(f"[{iv[0]:+.2f},{iv[1]:+.2f}]")
                if c["status"] == "TESTED":
                    lines.append(c["provenance"]["sha256"][:8])
                else:
                    lines.append("no run")
                ax.text(xa + 0.5 * wcell, y0 + 0.06 + lane_h * 0.375, "\n".join(lines),
                        fontsize=6.4 if n > 4 else 7.6, ha="center", va="center",
                        color=tcol, linespacing=1.25)
        if compact:
            tested = [c for c in cells_f if c["status"] == "TESTED"]
            r2s = [c["metric"]["value"] for c in tested]
            ax.text(x0, y0 + 0.88,
                    f"{len(tested)}/{n} run, all {tested[0]['verdict']}: "
                    f"R2 in [{min(r2s):.1f}, {max(r2s):.1f}]  "
                    f"(source sha {tested[0]['provenance']['sha256'][:8]})",
                    fontsize=7.5, ha="left", va="bottom", color="#333")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, mfc=COL[k],
                          mec=COL[k], label=k) for k in
               ("FAIL", "MARGINAL", "TOLERATED", "UNRESOLVED")]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=11, mfc="white",
                              mec=COL["UNTESTED"], label="UNTESTED (no run -- empty by design)"))
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(fig_pdf); fig.savefig(fig_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cells = (cells_wavy() + cells_xiao() + cells_r24() + cells_m13()
             + cells_never_run())
    n_tested = sum(1 for c in cells if c["status"] == "TESTED")
    n_untested = len(cells) - n_tested
    pending = [c["cell_id"] for c in cells if c["status"] == "UNTESTED"
               and not c["cell_id"].startswith("legacy::")]
    status = "M17_REAL_PARTIAL" if pending else "M17_REAL_COMPLETE"
    cert = dict(
        schema="verdict_map_real_v1",
        ledger_row="M17 (real)",
        idea=("Class/verdict map rebuilt exclusively from deposited high-fidelity runs; "
              "every badge carries dataset provenance; unrun cells are explicitly "
              "UNTESTED (the old Figs 5-6 hard-coded 18 of 28 badges)."),
        date=a.date, generated=datetime.datetime.now().isoformat(timespec="seconds"),
        status=status,
        verdict_rule=dict(bins={"FAIL": "R2 < 0", "MARGINAL": "0 <= R2 < 0.5",
                                "TOLERATED": "R2 >= 0.5"},
                          unresolved="verdict interval straddles a bin boundary",
                          shared_with="codes/analysis/epsilon_predictor_outoffamily.py"),
        n_cells=len(cells), n_tested=n_tested, n_untested=n_untested,
        pending_cells=pending, cells=cells,
        figure=dict(pdf=str((OUTDIR / "fig_verdict_map_real.pdf").relative_to(ROOT)),
                    png=str((OUTDIR / "fig_verdict_map_real.png").relative_to(ROOT))),
        rerun_command="python3 codes/figures/fig_verdict_map_real.py --date <YYYYMMDD>",
    )
    fig_pdf = OUTDIR / "fig_verdict_map_real.pdf"
    fig_png = OUTDIR / "fig_verdict_map_real.png"
    draw(cert, fig_pdf, fig_png)
    cert["figure"]["pdf_sha256"] = sha256(fig_pdf)
    cert["figure"]["png_sha256"] = sha256(fig_png)
    out = RESULTS / f"verdict_map_real_{a.date}.json"
    out.write_text(json.dumps(cert, indent=1, default=float) + "\n")
    print(f"{status}: {n_tested} tested cells, {n_untested} untested "
          f"(pending real runs: {len(pending)})")
    for c in cells:
        if c["status"] == "TESTED":
            m = c["metric"]
            print(f"  [{c['verdict']:>10s}] {c['cell_id']:34s} R2={m['value']:+8.2f} "
                  f"src={c['provenance']['sha256'][:8]}")
    print("saved ->", out, "and", fig_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
