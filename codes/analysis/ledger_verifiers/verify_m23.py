#!/usr/bin/env python3
"""Stable verifier for ledger row M23: figure asset drift.

Ledger: "Every figure's .pdf/.png regenerated from its script."  For every
figure included by the ACTIVE build this verifier binds the asset to its
generating script and input data:

  * the figure file resolves (.pdf for pdflatex) and a .png twin exists;
  * the figure is in the provenance registry below (script + inputs found by
    reading each script's load statements);
  * mtime(figure) >= mtime(script) and >= mtime(every input) -- a figure older
    than its generator or its data is drift;
  * a figure generated elsewhere (codes/figures/, node directory) is
    byte-identical to the manuscript copy;
  * the flat submission tree copy (manuscript/submission_flat/) is
    byte-identical to the manuscript copy;
  * sha256 of every active figure is recorded in
    codes/results/figure_provenance_m23.json (rewritten on every run).
Exit 1 listing each stale / unregistered / mismatched asset.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _active_build as ab  # noqa: E402

ROOT = ab.ROOT
MAN = ab.MAN
FLAT = MAN / "submission_flat"
OUT = ROOT / "codes" / "results" / "figure_provenance_m23.json"
VENDOR = ROOT / "codes" / "vendor" / "universal_wall_function" / "codes" / "results"
RESULTS = ROOT / "codes" / "results"
UTILS = ROOT / "codes" / "utils"
XIAO5 = (ROOT / "codes" / "new_data_download" / "geometry_driven" /
         "xiao_pehill_parameterized" / "pehill-5-cases-DNS" / "case_1p0" / "dns-data")

# figure stem -> generating script, inputs it reads, and any twin copy it also writes
REGISTRY: dict[str, dict] = {
# --- BLUEPRINT FIGURES, 2026-08-26 -------------------------------------------
# These nine are a LAYOUT MOCKUP, not final artwork.  Every numeric value in them
# was transcribed by hand from the manuscript's own tables (5, 2, 6, 7, 8, 9, 10)
# and from the text of section 5.4, so their upstream input is the manuscript,
# not a results file -- which is why "inputs" is empty here and why the
# no-placeholder check is FAILING on purpose while they stand.
# PHASE 2 OBLIGATION: when each figure is redrawn from data, repoint its "inputs"
# at the codes/results file the corresponding table cites, and delete this note.
    # Drawn for real 2026-08-26: no longer a blueprint.  Its values are still
    # transcribed from the manuscript's own results sections, so "inputs" stays
    # empty until the phase-2 repointing described below.
    # Panel (c)'s forces bound 2026-08-27; panels (a) and (b) quote prose.
    "fig_regime_map": {
        "script": ROOT / "codes" / "figures" / "fig_regime_map.py",
        "inputs": [RESULTS / "r2_4_m20_les_20260823.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_regime_map.pdf"],
    },
    # Drawn for real 2026-08-27: no longer a blueprint.
    "fig_class_map": {
        "script": ROOT / "codes" / "figures" / "fig_class_map.py",
        "inputs": [ROOT / "codes" / "figures" / "fig_regime_map.py"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_class_map.pdf"],
    },
    # Bound to the archive 2026-08-27: seventeen of the eighteen cases are read
    # straight out of the frozen error-operator output and reproduce every
    # printed digit.  The periodic hill's corrected score comes from the
    # reference-rebase output, because the operator file scores that one row
    # against the WITHDRAWN estimator; its station-error columns remain stated
    # literals, and that gap is recorded in work_progress.
    "fig_error_hierarchy": {
        "script": ROOT / "codes" / "figures" / "fig_blueprint_20260826.py",
        "inputs": [RESULTS / "signed_wall_error_metrics_m2.summary.json",
                   RESULTS / "reference_rebase_headlines_l0_20260825.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_error_hierarchy.pdf"],
    },
    # Bound 2026-08-27; the repeat identifies itself from the machine field.
    "fig_wavy_amplitude": {
        "script": ROOT / "codes" / "figures" / "fig_blueprint_20260826.py",
        "inputs": [RESULTS / "r1_sta2_wavy_amplitude_20260825.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_wavy_amplitude.pdf"],
    },
    # Bound to the archive 2026-08-27: every printed digit of both ladders --
    # a priori at the mesh-recorded surface and coupled on the two grids -- is
    # read from the rescored artifact.  46 values, all reproduced exactly.
    "fig_ladder_apriori_coupled": {
        "script": ROOT / "codes" / "figures" / "fig_blueprint_20260826.py",
        "inputs": [RESULTS / "r2m4_ladder_rescored_20260825.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_ladder_apriori_coupled.pdf"],
    },
    # Bound 2026-08-27 to the COMPLETE campaign (20260823), which is what the
    # tables were built from and what verify_r2_4_m20 defaults to.  The later
    # 20260825 re-run agrees on point estimates but not on bootstrap intervals
    # and is missing the sparse cube.
    "fig_sharp_walls": {
        "script": ROOT / "codes" / "figures" / "fig_blueprint_20260826.py",
        "inputs": [RESULTS / "r2_4_m20_les_20260823.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_sharp_walls.pdf"],
    },
    # Bound to the archive 2026-08-27.  The sweep is read from the RE-SCORED
    # conditioning output, not from r2_3_ym_window_* / metric_station_results_*,
    # which hold the scoring against the withdrawn reconstruction.
    "fig_coupled_score": {
        "script": ROOT / "codes" / "figures" / "fig_blueprint_20260826.py",
        "inputs": [RESULTS / "m13_highre_coupled_20260825_summary.json",
                   RESULTS / "scoring_reference_conditioning_l0_20260825.json"],
        "twins": [ROOT / "manuscript" / "submission_flat" / "fig_coupled_score.pdf"],
    },
    "fig_coupling_gain": {
        "script": ROOT / "codes" / "figures" / "fig_coupling_gain.py",
        "inputs": [RESULTS / "gain_probe_transmission_20260824.npz",
                   RESULTS / "gain_probe_transmission_20260824_summary.json",
                   RESULTS / "gain_probe_model_exponent_20260824.npz",
                   RESULTS / "gain_probe_model_exponent_20260824_summary.json"],
        "twins": [ROOT / "manuscript" / "submission_flat"
                       / "fig_coupling_gain.pdf",
                  ROOT / "manuscript" / "submission_flat"
                       / "fig_coupling_gain.png"],
    },
    "fig_as_deployed_operator": {
        "script": ROOT / "codes" / "figures" / "fig_as_deployed_operator.py",
        "inputs": [RESULTS / "as_deployed_evaluation_20260824.npz",
                   RESULTS / "as_deployed_evaluation_20260824_summary.json",
                   RESULTS / "as_deployed_bridge_ladder.json",
                   RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz",
                   ROOT / "codes" / "analysis" / "deployed_operator"
                        / "deployed_operator.py"],
        "twins": [ROOT / "manuscript" / "submission_flat"
                       / "fig_as_deployed_operator.pdf",
                  ROOT / "manuscript" / "submission_flat"
                       / "fig_as_deployed_operator.png"],
    },
    "fig_velocity_profiles": {
        "script": ROOT / "codes" / "figures" / "fig_velocity_profiles.py",
        "inputs": [VENDOR / "bfs_Re13700_wall_profiles.npz",
                   VENDOR / "nasa_hump_Re936000_wall_profiles.npz",
                   RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz",
                   UTILS / "plotting_utils.py", UTILS / "data_paths.py"],
        "twins": [],
    },
    "fig_faithful_tournament_l0": {
        # replaces fig_source_budget_l0, whose tournament ranked two published
        # families through substitutes; that figure is out of the active build.
        "script": ROOT / "codes" / "figures" / "fig_faithful_tournament_l0.py",
        "inputs": [RESULTS / "faithful_tournament_l0_20260825.json",
                   RESULTS / "wavy_geometry_holdout_l0_20260825.json"],
        "twins": [ROOT / "manuscript" / "submission_flat"
                       / "fig_faithful_tournament_l0.pdf",
                  ROOT / "manuscript" / "submission_flat"
                       / "fig_faithful_tournament_l0.png"],
    },
    "fig_amplitude_pitch_sweep": {
        "script": ROOT / "codes" / "figures" / "plot_dose_response.py",
        "inputs": [RESULTS / "dose_response_xiao.npz",
                   RESULTS / "repeating_structure_contrast.npz",
                   UTILS / "plotting_utils.py"],
        "twins": [],
    },
    "cancellation_parameter_corrected": {
        "script": ROOT / "codes" / "figures" / "plot_cancellation_parameter_corrected.py",
        "inputs": [VENDOR / "bfs_Re13700_wall_profiles.npz",
                   VENDOR / "nasa_hump_Re936000_wall_profiles.npz",
                   RESULTS / "wall_extraction_artifact.npz",
                   RESULTS / "dose_response_xiao.npz",
                   XIAO5 / "mean_files.dat", UTILS / "plotting_utils.py"],
        "twins": [],
    },
    "fig_common_surface_grid_l3": {
        # re-homed 2026-08-25 out of the rotating node tree (development/nodes/
        # node_004 rotated to development/exhausted_*); the generator keeps its
        # three-deep position so its own ``parents[3]`` still resolves to ROOT.
        "script": ROOT / "codes" / "figures" / "node_generators" / "analyze_grid_results_l3.py",
        "inputs": [RESULTS / "rswm_common_surface_grid_l2.npz",
                   RESULTS / "rswm_common_surface_grid_l2_summary.json",
                   RESULTS / "rswm_xiao_dns_grid_campaign_final_l2" / "CAMPAIGN_MANIFEST.json",
                   RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"],
        "twins": [ROOT / "codes" / "figures" / "fig_common_surface_grid_l3.pdf",
                  ROOT / "codes" / "figures" / "fig_common_surface_grid_l3.png"],
    },
    "fig_m13_re19000_validation": {
        "script": ROOT / "codes" / "figures" / "plot_m13_re19000_validation.py",
        "inputs": [RESULTS / "m13_highre_coupled_20260824.npz",
                   RESULTS / "m13_highre_coupled_20260824_summary.json"],
        "twins": [],
    },
    "fig_applied_traction": {
        # node_011: face-by-face replay of the deployed wall-model kernel and
        # the realizability projection it applies.  See
        # development/nodes/node_011/results.md.
        "script": ROOT / "codes" / "figures" / "fig_applied_traction.py",
        "inputs": [RESULTS / "applied_traction_reproduction.npz",
                   RESULTS / "applied_traction_reproduction_summary.json",
                   RESULTS / "clipping_timeseries.npz",
                   RESULTS / "clipping_timeseries_summary.json"],
        "twins": [],
    },
    "fig_metric_station_l3": {
        # regenerated by node_011: node_010's generator drew the
        # pressure-gradient-ODE critical height across panels carrying both
        # architectures.  See development/nodes/node_011/results.md.
        # re-homed 2026-08-25 out of the rotating node tree (see above).
        "script": ROOT / "codes" / "figures" / "node_generators" / "fig_metric_station_l3.py",
        "inputs": [RESULTS / "metric_station_results_l3_summary.json",
                   RESULTS / "metric_station_results_l3.npz",
                   RESULTS / "model_matched_transfer_l2_20260824_summary.json",
                   RESULTS / "model_matched_transfer_l2_20260824.npz"],
        "twins": [ROOT / "codes" / "figures" / "fig_metric_station_l3.pdf",
                  ROOT / "codes" / "figures" / "fig_metric_station_l3.png"],
    },
}

checks: list[tuple[str, bool]] = []
notes: list[str] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    active = ab.active_source()
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", ab.body(active))
    stems = []
    for inc in includes:
        stem = Path(inc).name
        stem = re.sub(r"\.(pdf|png|eps|jpg)$", "", stem)
        stems.append(stem)
    check(f"active build includes {len(stems)} figure(s): {', '.join(stems)}", bool(stems))

    provenance: dict[str, dict] = {}
    unregistered = [s for s in stems if s not in REGISTRY]
    check("every active figure has a registered generating script", not unregistered)
    notes.extend(f"unregistered figure: {s}" for s in unregistered)

    for stem in stems:
        pdf = MAN / "figures" / f"{stem}.pdf"
        png = MAN / "figures" / f"{stem}.png"
        check(f"{stem}: .pdf and .png assets exist", pdf.exists() and png.exists())
        if not pdf.exists():
            continue
        entry = {"pdf_sha256": sha256(pdf), "pdf_mtime": stamp(pdf)}
        if png.exists():
            entry.update(png_sha256=sha256(png), png_mtime=stamp(png))
        if stem in REGISTRY:
            reg = REGISTRY[stem]
            script = reg["script"]
            entry["script"] = rel(script) if script.exists() else f"MISSING {rel(script)}"
            entry["inputs"] = {}
            missing_inputs = [p for p in reg["inputs"] if not p.exists()]
            check(f"{stem}: script and all {len(reg['inputs'])} inputs exist",
                  script.exists() and not missing_inputs)
            notes.extend(f"{stem}: missing input {rel(p)}" for p in missing_inputs)
            deps = [script] + [p for p in reg["inputs"] if p.exists()]
            stale = []
            for asset in (pdf, png):
                if not asset.exists():
                    continue
                for dep in deps:
                    if dep.exists() and asset.stat().st_mtime < dep.stat().st_mtime:
                        stale.append(f"{asset.name} ({stamp(asset)}) older than "
                                     f"{rel(dep)} ({stamp(dep)})")
            check(f"{stem}: assets newer than generating script and every input", not stale)
            notes.extend(f"{stem}: STALE {s}" for s in stale)
            for p in reg["inputs"]:
                if p.exists():
                    entry["inputs"][rel(p)] = {"mtime": stamp(p), "sha256": sha256(p)}
            twins_ok = True
            for twin in reg["twins"]:
                target = pdf if twin.suffix == ".pdf" else png
                same = twin.exists() and target.exists() and sha256(twin) == sha256(target)
                twins_ok &= same
                if not same:
                    notes.append(f"{stem}: twin {rel(twin)} differs from manuscript copy")
            if reg["twins"]:
                check(f"{stem}: generator-side copy is byte-identical to the manuscript copy",
                      twins_ok)
        flat = FLAT / f"{stem}.pdf"
        if FLAT.exists():
            same = flat.exists() and sha256(flat) == entry["pdf_sha256"]
            check(f"{stem}: submission_flat copy is byte-identical to manuscript/figures",
                  same)
        provenance[stem] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "figure-provenance-m23-v1",
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active_figures": provenance,
    }, indent=2, sort_keys=True) + "\n")
    print(f"  provenance written: {rel(OUT)}")
    for n in notes:
        print("  " + n)
    failed = [n for n, ok in checks if not ok]
    print(f"M23: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
