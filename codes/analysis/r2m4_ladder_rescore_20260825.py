#!/usr/bin/env python3
"""Re-score the R2-m4 / R3-2 ladder (a priori AND coupled) against the corrected
wall-traction references.  NEW producer, 2026-08-25 — the 2026-08-23 artifacts
are read unmodified and retained verbatim as superseded.

Primary reference: the MGLET DNS deposited bottom-wall tau_w (B).
Bracket: the same Xiao archive re-estimated with a curvature-aware through-origin
cubic (C).  The withdrawn 4-point linear reconstruction (A) is re-computed here
too, so this artifact reproduces the superseded numbers and the corrected ones
side by side under one metric.

Nothing about the ladder's PREDICTIONS changes: the wall-model inputs (u_m,
dp/ds, the convection profile at y_m) come from well-resolved interior data and
the predictions were never a function of the truth.  Only the score changes.

Output: codes/results/r2m4_ladder_rescored_20260825.{json,npz}
        codes/figures/fig_r2m4_ladder_rescored.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
import r2m4_ladder_common as C          # noqa: E402  (metric + bootstrap protocol)
import r2m4_truth_references as T       # noqa: E402

STAMP = "20260825"
APRIORI = ROOT / "codes/results/r2m4_apriori_ladder_20260823"
COUPLED = ROOT / "codes/results/r2m4_ladder_coupled_20260823"
OUT_JSON = ROOT / "codes/results" / f"r2m4_ladder_rescored_{STAMP}.json"
OUT_NPZ = ROOT / "codes/results" / f"r2m4_ladder_rescored_{STAMP}.npz"
FIG = ROOT / "codes/figures/fig_r2m4_ladder_rescored"
SURFACES = ("ladder_L1", "archive_index10", "common_W1")
COUPLED_GRIDS = ("L1", "L2", "W1")
COUPLED_MODELS = ("equilibrium", "totalGradient", "hickel",
                  "resolvedConvectionLinear", "resolvedConvectionConstant")
DENSE = np.arange(C.DENSE_N) / C.DENSE_N


def score_block(truth_dense, predictions_dense, seed_offset):
    """Metrics + paired phase-block intervals + paired differences for one family."""
    metrics = {}
    for name, p in predictions_dense.items():
        e = p - truth_dense
        metrics[name] = {
            "relative_rms": float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(truth_dense ** 2))),
            "r2": float(1.0 - np.sum(e ** 2) / np.sum((truth_dense - truth_dense.mean()) ** 2)),
            "sign_accuracy": float(np.mean(np.sign(p) == np.sign(truth_dense))),
            "signed_force_ratio": float(np.sum(p) / np.sum(truth_dense)),
            "correlation": float(np.corrcoef(p, truth_dense)[0, 1]),
        }
    boot = C.block_bootstrap_relative_rms(truth_dense, predictions_dense,
                                          seed=C.BOOTSTRAP_SEED + seed_offset)
    for name in metrics:
        metrics[name]["relative_rms_interval"] = C.interval(boot[name])
    return metrics, boot


def verdict_from(metrics, boot, restored, baseline):
    """Pre-registered rule (r2m4_ladder_common.side_verdict) on any reference."""
    if restored not in boot or baseline not in boot:
        return None, None
    difference = C.interval(boot[restored] - boot[baseline])
    return C.side_verdict(metrics[restored]["relative_rms_interval"], difference), difference


def main() -> int:
    refs = T.references()
    ref_dense = {k: C.periodic_interp(p, t, DENSE) for k, (p, t, _) in refs.items()}
    apz = np.load(str(APRIORI) + ".npz")
    ap_json = json.loads((Path(str(APRIORI) + ".json")).read_text())
    cpz = np.load(str(COUPLED) + ".npz")
    cp_json = json.loads((Path(str(COUPLED) + ".json")).read_text())

    result = {
        "schema": "r2m4-ladder-rescored-v1",
        "row": "R2-m4 / R3-2",
        "stamp": STAMP,
        "supersedes": {"apriori": str(Path(str(APRIORI) + ".json").relative_to(ROOT)),
                       "coupled": str(Path(str(COUPLED) + ".json").relative_to(ROOT)),
                       "note": ("both retained verbatim; their scores are reference A, which the "
                                "operator withdrew as a scoring reference on 2026-08-25")},
        "references": {k: {"label": v, "rms_over_primary": float(
            np.sqrt(np.mean(ref_dense[k] ** 2)) / np.sqrt(np.mean(ref_dense[T.PRIMARY] ** 2)))}
            for k, v in T.LABELS.items()},
        "primary_reference": T.PRIMARY,
        "bracket_reference": T.BRACKET,
        "withdrawn_reference": T.SUPERSEDED,
        "reference_defect": ("estimator, not data: the Xiao archive's wall spacing (0.0093-0.0136 H, fit "
                             "points y+ 2.4-44) makes a 4-point through-origin linear fit unconverged; the "
                             "same estimator on MGLET's own profiles resampled to that spacing reproduces "
                             "the RMS deficit and the sign flips at x/H = 5 and 7"),
        "metric": "RMS-normalised signed physical-tangent traction error on 4096 phase points",
        "bootstrap": {"draws": C.BOOTSTRAP_DRAWS, "block_points": C.BLOCK_POINTS,
                      "seed": C.BOOTSTRAP_SEED, "paired_within_reference": True},
        "apriori": {}, "coupled": {}, "verdicts": {}, "sources": {
            "apriori_npz_sha256": C.sha256(Path(str(APRIORI) + ".npz")),
            "coupled_npz_sha256": C.sha256(Path(str(COUPLED) + ".npz")),
            "mglet_sha256": C.sha256(T.MGLET_WALL),
            "xiao_archive_sha256": C.sha256(T.XIAO_ARCHIVE)},
    }
    arrays = {f"reference_{k}": v for k, v in ref_dense.items()}
    arrays["dense_phase"] = DENSE

    # ---------------------------------------------------------------- a priori
    for s_index, surface in enumerate(SURFACES):
        phase = apz[f"{surface}_phase"]
        preds = {m: C.periodic_interp(phase, apz[f"{surface}_{m}"], DENSE) for m in C.LADDER}
        for m in C.LADDER:
            arrays[f"apriori_{surface}_{m}_dense"] = preds[m]
        entry = {}
        for r_index, (ref, truth) in enumerate(ref_dense.items()):
            metrics, boot = score_block(truth, preds, 100 * s_index + r_index)
            pairs = {}
            for a, b in (("M1_pressure_gradient_ode", "M0_equilibrium"),
                         ("M2_hickel_modelled_convection", "M1_pressure_gradient_ode"),
                         ("Xc_resolved_convection_linear", "M1_pressure_gradient_ode"),
                         ("Xc_exact_convection_profile", "M1_pressure_gradient_ode"),
                         ("Xall_all_omitted_transport", "M0_equilibrium"),
                         ("Xfull_all_transport_plus_exact_shear_stress", "M0_equilibrium")):
                pairs[f"{a}-minus-{b}"] = C.interval(boot[a] - boot[b])
            v_linear, _ = verdict_from(metrics, boot, "Xc_resolved_convection_linear",
                                       "M1_pressure_gradient_ode")
            v_exact, _ = verdict_from(metrics, boot, "Xc_exact_convection_profile",
                                      "M1_pressure_gradient_ode")
            y_m = apz[f"{surface}_y_m"]
            dpds = apz[f"{surface}_diag_dpds"]
            tau_ref = C.periodic_interp(refs[ref][0], refs[ref][1], np.mod(phase, 1.0))
            entry[ref] = {
                "metrics": metrics,
                "paired_relative_rms_differences": pairs,
                "verdict_resolved_convection": v_linear,
                "verdict_exact_convection": v_exact,
                "epsilon_median": float(np.median(np.abs(tau_ref) /
                                                  np.maximum(np.abs(dpds) * y_m, 1e-30))),
                "best_rung": min(metrics, key=lambda k: metrics[k]["relative_rms"]),
            }
        result["apriori"][surface] = entry

    # ----------------------------------------------------------------- coupled
    for g_index, grid in enumerate(COUPLED_GRIDS):
        preds = {m: C.periodic_interp(cpz[f"{grid}_{m}_phase"], cpz[f"{grid}_{m}_tau_s"], DENSE)
                 for m in COUPLED_MODELS if f"{grid}_{m}_tau_s" in cpz.files}
        if not preds:
            continue
        for m, p in preds.items():
            arrays[f"coupled_{grid}_{m}_dense"] = p
        entry = {}
        for r_index, (ref, truth) in enumerate(ref_dense.items()):
            metrics, boot = score_block(truth, preds, 500 + 100 * g_index + r_index)
            pairs = {}
            for a, b in (("totalGradient", "equilibrium"), ("hickel", "totalGradient"),
                         ("resolvedConvectionLinear", "totalGradient"),
                         ("resolvedConvectionConstant", "totalGradient")):
                if a in boot and b in boot:
                    pairs[f"{a}-minus-{b}"] = C.interval(boot[a] - boot[b])
            v, _ = verdict_from(metrics, boot, "resolvedConvectionLinear", "totalGradient")
            entry[ref] = {"metrics": metrics, "paired_relative_rms_differences": pairs,
                          "verdict_resolved_convection": v,
                          "best_rung": min(metrics, key=lambda k: metrics[k]["relative_rms"])}
        result["coupled"][grid] = entry

    # ---------------------------------------------------------------- verdicts
    for ref in ref_dense:
        ap = result["apriori"]["ladder_L1"][ref]["verdict_resolved_convection"]
        co = result["coupled"]["L1"][ref]["verdict_resolved_convection"]
        result["verdicts"][ref] = {
            "apriori_ladder_L1": ap, "coupled_L1": co,
            "coupled_L2": result["coupled"]["L2"][ref]["verdict_resolved_convection"],
            "row": C.row_verdict(ap, co),
        }
    row_primary = result["verdicts"][T.PRIMARY]["row"]
    row_bracket = result["verdicts"][T.BRACKET]["row"]
    result["row_verdict_primary"] = row_primary
    result["row_verdict_bracket"] = row_bracket
    result["row_verdict_reference_robust"] = (
        row_primary if row_primary == row_bracket else
        f"REFERENCE_DEPENDENT_{row_primary}_vs_{row_bracket}")
    result["status"] = "R2M4_LADDER_RESCORED_OK"
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez(OUT_NPZ, **arrays)

    for surface in ("ladder_L1",):
        print(f"== a priori, surface {surface}")
        print("   %-44s %-22s %-22s %-22s" % ("rung", "A Xiao linear4 (WITHDRAWN)", "B MGLET (primary)", "C Xiao cubic6 (bracket)"))
        for m in C.LADDER:
            row = []
            for ref in ("A_xiao_linear4_deposited", "B_mglet_deposited", "C_xiao_cubic6_repaired"):
                e = result["apriori"][surface][ref]["metrics"][m]
                row.append("%.3f [%.2f,%.2f]" % (e["relative_rms"], e["relative_rms_interval"]["low"],
                                                 e["relative_rms_interval"]["high"]))
            print("   %-44s %-22s %-22s %-22s" % (m, *row))
    for grid in COUPLED_GRIDS:
        if grid not in result["coupled"]:
            continue
        print(f"== coupled, grid {grid}")
        for m in COUPLED_MODELS:
            if m not in result["coupled"][grid][T.PRIMARY]["metrics"]:
                continue
            row = []
            for ref in ("A_xiao_linear4_deposited", "B_mglet_deposited", "C_xiao_cubic6_repaired"):
                e = result["coupled"][grid][ref]["metrics"][m]
                row.append("%.3f [%.2f,%.2f]" % (e["relative_rms"], e["relative_rms_interval"]["low"],
                                                 e["relative_rms_interval"]["high"]))
            print("   %-30s %-22s %-22s %-22s" % (m, *row))
    print("verdicts:", json.dumps(result["verdicts"], sort_keys=True))
    print("row (primary B):", row_primary, "| row (bracket C):", row_bracket,
          "| reference-robust:", result["row_verdict_reference_robust"])
    make_figure(result, arrays)
    print("wrote", OUT_JSON.relative_to(ROOT), OUT_NPZ.relative_to(ROOT))
    return 0


def make_figure(result, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ("M0_equilibrium", "M1_pressure_gradient_ode", "M2_hickel_modelled_convection",
             "Xc_resolved_convection_linear", "Xc_exact_convection_profile",
             "Xall_all_omitted_transport", "Xfull_all_transport_plus_exact_shear_stress")
    labels = ["M0\nSpalding", "M1\nPG-ODE", "M2\nHickel", "Xc\nconv.", "Xc*\nexact conv.",
              "Xall\nall transport", "Xfull\n+exact stress"]
    colors = {"A_xiao_linear4_deposited": "#9e9e9e", "B_mglet_deposited": "#d62728",
              "C_xiao_cubic6_repaired": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    xs = np.arange(len(order))
    for ref, shift in (("A_xiao_linear4_deposited", -0.22), ("B_mglet_deposited", 0.0),
                       ("C_xiao_cubic6_repaired", 0.22)):
        m = result["apriori"]["ladder_L1"][ref]["metrics"]
        v = [m[k]["relative_rms"] for k in order]
        lo = [m[k]["relative_rms_interval"]["low"] for k in order]
        hi = [m[k]["relative_rms_interval"]["high"] for k in order]
        ax.errorbar(xs + shift, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o", ms=4,
                    color=colors[ref], capsize=2,
                    label=("A Xiao linear-4 (withdrawn)" if ref.startswith("A") else
                           "B MGLET DNS (primary)" if ref.startswith("B") else "C Xiao cubic-6 (bracket)"))
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"RMS$(\tau_s-\tau_s^{ref})$ / RMS$(\tau_s^{ref})$")
    ax.set_title("a priori ladder, $y_m/H=0.094$"); ax.legend(fontsize=7)
    ax = axes[1]
    corder = ("equilibrium", "totalGradient", "hickel", "resolvedConvectionLinear", "resolvedConvectionConstant")
    clabels = ["M0", "M1", "M2", "Xc lin.", "Xc const."]
    xs = np.arange(len(corder))
    for ref, shift in (("A_xiao_linear4_deposited", -0.22), ("B_mglet_deposited", 0.0),
                       ("C_xiao_cubic6_repaired", 0.22)):
        m = result["coupled"]["L1"][ref]["metrics"]
        pts = [(i, m[k]) for i, k in enumerate(corder) if k in m]
        v = [p[1]["relative_rms"] for p in pts]
        lo = [p[1]["relative_rms_interval"]["low"] for p in pts]
        hi = [p[1]["relative_rms_interval"]["high"] for p in pts]
        ax.errorbar([p[0] + shift for p in pts], v, yerr=[np.subtract(v, lo), np.subtract(hi, v)],
                    fmt="s", ms=4, color=colors[ref], capsize=2)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(clabels, fontsize=8)
    ax.set_title("coupled WMLES ladder, L1")
    ax = axes[2]
    for ref, style in (("A_xiao_linear4_deposited", "-"), ("B_mglet_deposited", "-"),
                       ("C_xiao_cubic6_repaired", "--")):
        ax.plot(arrays["dense_phase"], 2 * arrays[f"reference_{ref}"], style, color=colors[ref], lw=1.4,
                label=ref.split("_")[0])
    ax.set_xlabel("phase $x/L_x$"); ax.set_ylabel("$C_f$"); ax.set_title("the three references")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(FIG) + ".pdf"); fig.savefig(str(FIG) + ".png", dpi=160)


if __name__ == "__main__":
    raise SystemExit(main())
