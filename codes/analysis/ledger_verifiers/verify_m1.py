#!/usr/bin/env python3
"""Independent stable verifier for referee-ledger item M1."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes" / "results"
SEED = 20260821
N_BOOT = 5000

sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "vendor" / "universal_wall_function" /
                       "codes" / "analysis"))
from dose_response_xiao import NU as XIAO_NU  # noqa: E402
from dose_response_xiao import XIAO, Y_IDX, read_case  # noqa: E402
from ode_wall_model import predict_tau_w  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_tex(text: str) -> str:
    """Active build: nested ``\\iffalse`` archive blocks and comments removed."""
    kept: list[str] = []
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(r"\iffalse"):
            depth += 1
            continue
        if stripped.startswith(r"\fi") and depth:
            depth -= 1
            continue
        if depth:
            continue
        kept.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(kept)


def score(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, float)
    prediction = np.asarray(prediction, float)
    return 1.0 - np.sum((reference - prediction) ** 2) / np.sum(
        (reference - np.mean(reference)) ** 2
    )


def rng_for(label: str) -> np.random.Generator:
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
    return np.random.default_rng(SEED + offset)


def bootstrap(reference: np.ndarray, prediction: np.ndarray, label: str) -> np.ndarray:
    reference = np.asarray(reference, float)
    prediction = np.asarray(prediction, float)
    valid = np.isfinite(reference) & np.isfinite(prediction)
    reference, prediction = reference[valid], prediction[valid]
    block = max(2, int(math.ceil(math.sqrt(len(reference)))))
    rng = rng_for(label)
    out = np.empty(N_BOOT)
    count = int(math.ceil(len(reference) / block))
    for b in range(N_BOOT):
        starts = rng.integers(0, len(reference), size=count)
        index = np.concatenate(
            [(start + np.arange(block)) % len(reference) for start in starts]
        )[:len(reference)]
        out[b] = score(reference[index], prediction[index])
    return out


def trace(metrics: np.lib.npyio.NpzFile, name: str) -> tuple[np.ndarray, np.ndarray]:
    index = [str(v) for v in metrics["names"]].index(name)
    n = int(metrics["n_stations"][index])
    return metrics["station_tau_ref"][index, :n], metrics["station_tau_pred"][index, :n]


def raw_xiao(case_name: str) -> tuple[np.ndarray, np.ndarray]:
    case = read_case(str(Path(XIAO) / case_name))
    prediction = np.full(len(case["tau_w"]), np.nan)
    for i in range(len(prediction)):
        y, u = case["y"][i], case["U"][i]
        if Y_IDX < len(y) and y[Y_IDX] > 0 and np.isfinite(u[Y_IDX]):
            prediction[i] = predict_tau_w(float(u[Y_IDX]), float(y[Y_IDX]),
                                           float(case["dp_dx"][i]), XIAO_NU)
    return np.asarray(case["tau_w"], float), prediction


def close(a, b, atol=1e-12) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=False))


def window_two_sem(campaign: dict, model: str) -> list[float]:
    values = np.asarray([
        campaign["averaging"][f"G2c:{model}"][str(window)]["r2"]
        for window in (180, 225, 270)
    ], dtype=float)
    half_width = 2.0 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    centre = float(np.mean(values))
    return [centre - half_width, centre + half_width]


def main() -> None:
    certificate = json.loads((RESULTS / "headline_uncertainty_m1.json").read_text())
    samples = np.load(RESULTS / "headline_uncertainty_m1.npz", allow_pickle=False)
    active = json.loads(
        (RESULTS / "active_manuscript_number_reconciliation_m15.json").read_text()
    )
    checks: list[tuple[str, bool]] = []
    claims = certificate["claims"]
    active_names = set(active["claims"])
    checks.append(("schema and PASS status",
                   certificate["schema"] == "headline-uncertainty-m1-v1" and
                   certificate["status"] == "PASS"))
    checks.append(("exact active-claim coverage",
                   set(claims) == active_names and
                   certificate["active_claim_count"] == len(active_names)))
    checks.append(("stored sample inventory", set(map(str, samples["claim_names"])) == active_names))
    checks.append(("fixed seed and replicate count",
                   int(samples["bootstrap_seed"]) == SEED and
                   int(samples["bootstrap_replicates"]) == N_BOOT))

    # Binding policy (2026-08-23): raw data and result artifacts are bound by
    # bytes (changing them requires re-running the producer).  The M15
    # reconciliation certificate is a DERIVED inventory that is legitimately
    # re-emitted after every manuscript edit (its tex/pdf hashes change while
    # its claim content does not), so it is bound by content: the claim names
    # and the registered point values must agree with this certificate.
    derived = {"codes/results/active_manuscript_number_reconciliation_m15.json"}
    stale = [path for path, digest in certificate["source_sha256"].items()
             if path not in derived
             and not ((ROOT / path).is_file() and sha256(ROOT / path) == digest)]
    checks.append(("all byte-addressed data/result sources unchanged", not stale))
    if stale:
        print("  stale sources: " + ", ".join(stale))
    drifted = [name for name, claim in active["claims"].items()
               if name not in claims
               or not close(np.atleast_1d(np.asarray(claim["value"], float)),
                            np.atleast_1d(np.asarray(claims[name]["point"], float)),
                            atol=1e-10)]
    checks.append(("M15 inventory bound by content: every active claim value equals "
                   "the certified point", not drifted))
    if drifted:
        print("  drifted claims: " + ", ".join(drifted))

    # 2026-08-25: the hill traction reference was replaced.  It is rebuilt here
    # straight from the raw ERCOFTAC deposit -- not read from the producer's
    # output -- so that this verifier does not inherit the producer's choice.
    def corrected_hill_reference():
        raw = np.loadtxt(ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/"
                                "UFR3-30_data-NP-Re5600-DNS2-11.dat")
        if not (np.allclose(raw[-2], [0.0, 0.0, 0.0])
                and np.allclose(raw[-1], [9.0, 0.0, 0.0])):
            raise RuntimeError("ERCOFTAC placeholder rows moved")
        body = raw[:-2]
        src_phase = np.mod(body[:, 0] / 9.0, 1.0)
        order = np.argsort(src_phase)
        ph, tau = src_phase[order], body[order, 1]
        archive = np.load(RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz")
        tgt = np.mod(np.asarray(archive["x"], float) / 9.0, 1.0)
        return np.interp(tgt, np.concatenate([ph - 1, ph, ph + 1]),
                         np.concatenate([tau, tau, tau]))

    metrics = np.load(RESULTS / "signed_wall_error_metrics_m2.npz", allow_pickle=True)
    hill_reference = corrected_hill_reference()
    for label, name in (("canonical_hill_r2", "periodic_hills_case_1p0"),
                        ("curved_bfs_r2", "curved_bfs_Re13700_DNS")):
        reference, prediction = trace(metrics, name)
        if label == "canonical_hill_r2":
            reference = hill_reference
        draws = bootstrap(reference, prediction, label)
        checks.append((f"{label} raw point and 95% interval rebuilt",
                       close(score(reference, prediction), claims[label]["point"]) and
                       close(np.percentile(draws, [2.5, 97.5]), claims[label]["interval"]) and
                       close(draws, samples[label])))
    withdrawn_ref, hill_pred = trace(metrics, "periodic_hills_case_1p0")
    checks.append(("the certified hill score is the corrected one, not the withdrawn one",
                   abs(score(withdrawn_ref, hill_pred)
                       - claims["canonical_hill_r2"]["point"]) > 10.0))

    diagnostic = np.load(RESULTS / "diagnostic_test_corrected.npz", allow_pickle=False)
    mask = diagnostic["controlled_dns_valid"].astype(bool)
    ref, pred = hill_reference[mask], diagnostic["controlled_dns"][mask]
    draws = bootstrap(ref, pred, "corrected_exact_stress_r2")
    checks.append(("exact-stress control interval rebuilt",
                   close(score(ref, pred), claims["corrected_exact_stress_r2"]["point"]) and
                   close(np.percentile(draws, [2.5, 97.5]),
                         claims["corrected_exact_stress_r2"]["interval"]) and
                   close(draws, samples["corrected_exact_stress_r2"])))

    # The certified family range is now the measured range over all 29 members
    # under one corrected estimator; the previous object bootstrapped two
    # boundary members against a reference that has since been withdrawn.
    sweep = json.loads(
        (ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/"
         "xiao29_epsilon_sweep.json").read_text())
    corrected_r2 = [m["repaired"]["r2"] for m in sweep["per_member"]]
    xiao_ok = close([min(corrected_r2), max(corrected_r2)],
                    claims["xiao_family_r2_range"]["point"])
    xiao_ok &= len(corrected_r2) == 29 and max(corrected_r2) < 0.0
    xiao_ok &= claims["xiao_family_r2_range"]["members_with_independent_reference"] == \
        sweep["members_with_independent_wall_traction_reference"]
    xiao_ok &= "boundary_cases" not in claims["xiao_family_r2_range"]
    checks.append(("corrected 29-member family range rebuilt with its stated "
                   "reference limitation", xiao_ok))

    grid = json.loads((RESULTS / "rswm_grid_results_l3_summary.json").read_text())
    grid_npz = np.load(RESULTS / "rswm_grid_results_l3.npz", allow_pickle=False)
    corrected_campaigns = json.loads(
        (RESULTS / "m13_highre_coupled_20260825_summary.json").read_text())["campaigns"]
    coupled5600 = corrected_campaigns["5600"]
    models = ("equilibrium", "total_gradient_tble")
    finest = [coupled5600["metrics"][f"G2c:{model}"] for model in models]
    primary = coupled5600["phase_bootstrap_primary_intervals"]
    coupled_ok = close(
        [entry["relative_rms"] for entry in finest],
        claims["coupled_finest_traction_errors"]["point"])
    coupled_ok &= close(
        [[primary[f"G2c:{model}"]["low"], primary[f"G2c:{model}"]["high"]]
         for model in models],
        claims["coupled_finest_traction_errors"]["intervals"])
    coupled_ok &= close(
        np.vstack([grid_npz[f"G2c_{model}_primary_bootstrap_relative_rms"]
                   for model in models]),
        samples["coupled_finest_traction_errors"])
    coupled_ok &= close(
        [entry["r2"] for entry in finest], claims["coupled_finest_r2"]["point"])
    coupled_ok &= close(
        [[min(coupled5600["grid_path_convergence"][f"{model}:r2"]["values"]),
          max(coupled5600["grid_path_convergence"][f"{model}:r2"]["values"])]
         for model in models],
        claims["coupled_finest_r2"]["intervals"])
    holm = coupled5600["failure_significance_tests"]
    coupled_ok &= close(
        [holm[model]["p_one_sided_holm_two_models"] for model in models],
        claims["coupled_holm_p"]["point"])
    comparison = coupled5600["model_comparison"]
    coupled_ok &= close(
        [comparison["finest_relative_rms_tble_minus_equilibrium"],
         comparison["paired_exact_block_test"]["p_two_sided"]],
        claims["coupled_model_difference_p"]["point"])
    coupled_ok &= close(
        [comparison["primary_bootstrap_delta_interval"]["low"],
         comparison["primary_bootstrap_delta_interval"]["high"]],
        claims["coupled_model_difference_p"]["interval"])
    coupled_ok &= close(
        grid_npz["block512_G2c_delta_tble_minus_equilibrium"],
        samples["coupled_model_difference_p"])
    checks.append(("coupled Level-3 phase intervals, grid envelopes and exact tests traced",
                   coupled_ok))

    high_re_summary = json.loads(
        (RESULTS / "m13_highre_coupled_20260825_summary.json").read_text()
    )
    re5600 = high_re_summary["campaigns"]["5600"]
    high_re = high_re_summary["campaigns"]["10595"]
    re19000 = high_re_summary["campaigns"]["19000"]
    high_re_npz = np.load(RESULTS / "m13_highre_coupled_20260825.npz",
                          allow_pickle=False)
    high_re_finest = [high_re["metrics"][f"G2c:{model}"] for model in models]
    high_re_intervals = high_re["phase_bootstrap_primary_intervals"]
    high_re_ok = close(
        [entry["relative_rms"] for entry in high_re_finest],
        claims["re10595_finest_traction_errors"]["point"])
    high_re_ok &= close(
        [[high_re_intervals[f"G2c:{model}"]["low"],
          high_re_intervals[f"G2c:{model}"]["high"]] for model in models],
        claims["re10595_finest_traction_errors"]["intervals"])
    high_re_ok &= close(
        np.vstack([high_re_npz[
            f"re10595_G2c_{model}_primary_bootstrap_relative_rms"]
            for model in models]), samples["re10595_finest_traction_errors"])
    high_re_ok &= close(
        [entry["r2"] for entry in high_re_finest],
        claims["re10595_finest_r2"]["point"])
    high_re_ok &= close(
        [[min(high_re["grid_path_convergence"][f"{model}:r2"]["values"]),
          max(high_re["grid_path_convergence"][f"{model}:r2"]["values"])]
         for model in models], claims["re10595_finest_r2"]["intervals"])
    high_re_ok &= close(
        [window_two_sem(high_re, model) for model in models],
        claims["re10595_finest_r2"]["window_intervals"])
    truth_event = high_re["truth"]["events"]
    high_re_ok &= close(
        [entry["reattachment_x_over_H"] for entry in high_re_finest]
        + [truth_event["reattachment_x_over_H"]],
        claims["re10595_finest_reattachment"]["point"])
    high_re_ok &= close(
        [truth_event["documented_reattachment_x_over_H"]
         - truth_event["documented_reattachment_uncertainty_over_H"],
         truth_event["documented_reattachment_x_over_H"]
         + truth_event["documented_reattachment_uncertainty_over_H"]],
        claims["re10595_finest_reattachment"]["interval"])
    checks.append(("terminal Re10595 phase intervals, grid envelopes and event uncertainty traced",
                   high_re_ok))
    checks.append(("Re5600 averaging-window R2 intervals independently rebuilt",
                   close([window_two_sem(re5600, model) for model in models],
                         claims["coupled_finest_r2"]["window_intervals"])))

    re19000_ok = close(
        [re19000["profiles"][f"G2c:{model}"]["rapp_19000"]["u_rms_mean"]
         for model in models], claims["re19000_profile_mean_rms"]["point"])
    re19000_ok &= close(
        [[min(re19000["profiles"][f"{grid}:{model}"]["rapp_19000"]["u_rms_mean"]
              for grid in ("G1c", "G2c")),
          max(re19000["profiles"][f"{grid}:{model}"]["rapp_19000"]["u_rms_mean"]
              for grid in ("G1c", "G2c"))] for model in models],
        claims["re19000_profile_mean_rms"]["intervals"])
    exp19 = re19000["experimental_reattachment"]
    re19000_ok &= close(
        [re19000["metrics"][f"G2c:{model}"]["reattachment_x_over_H"]
         for model in models] + [exp19["estimate_x_over_H"]],
        claims["re19000_reattachment"]["point"])
    re19000_ok &= close(exp19["bracket_x_over_H"],
                        claims["re19000_reattachment"]["experimental_bracket"])
    re19000_ok &= all(
        re19000["grid_path_convergence"][f"{model}:reattachment_x_over_H"]["status"]
        == "two_grid_sensitivity_only" for model in models)
    re19000_ok &= close(
        [re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated"]
         for model in models], claims["re19000_cancellation_medians"]["point"])
    re19000_ok &= close(
        [[re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated_interval"]["low"],
          re19000["eps_c"][f"G2c:{model}"]["eps_c_median_separated_interval"]["high"]]
         for model in models], claims["re19000_cancellation_medians"]["intervals"])
    checks.append(("terminal Re19000 PIV, event, grid and cancellation uncertainty traced",
                   re19000_ok))

    uq = np.load(RESULTS / "uncertainty_certificate_l1.npz", allow_pickle=False)
    hill = [str(v) for v in uq["names"]].index("periodic_hills_case_1p0")
    checks.append(("three-term simultaneous interval traced",
                   close(uq["statistic_centre"][hill, 0], claims["three_term_hill_ratio"]["point"]) and
                   close([uq["statistic_simultaneous_ci_low"][hill, 0],
                          uq["statistic_simultaneous_ci_high"][hill, 0]],
                         claims["three_term_hill_ratio"]["interval"])))

    thin = np.load(RESULTS / "full_rans_thin_layer_audit.npz", allow_pickle=False)
    checks.append(("thin-layer operator-height envelopes traced",
                   close([np.min(thin["reynolds_streamwise_fraction"]),
                          np.max(thin["reynolds_streamwise_fraction"])],
                         claims["thin_layer_reynolds_range"]["interval"]) and
                   close([np.min(thin["viscous_streamwise_to_normal_ratio"]),
                          np.max(thin["viscous_streamwise_to_normal_ratio"])],
                         claims["thin_layer_viscous_range"]["interval"])))

    phase = np.load(RESULTS / "independent_phase_balance_r1_sta3.npz",
                    allow_pickle=False)
    primary = int(np.flatnonzero(phase["phase_counts"] == 48)[0])
    phase_range = [float(np.min(phase["phase_closure_operator_min"])),
                   float(np.max(phase["phase_closure_operator_max"]))]
    wave_range = [float(np.min(phase["wavelength_closure_operator_min"])),
                  float(np.max(phase["wavelength_closure_operator_max"]))]
    checks.append(("independent phase and wavelength envelopes traced",
                   close(phase["phase_closure_central"][primary],
                         claims["independent_phase_closure"]["point"]) and
                   close(phase_range,
                         claims["independent_phase_operator_envelope"]["interval"]) and
                   close(phase["wavelength_closure_central"][primary],
                         claims["independent_wavelength_closure"]["point"]) and
                   close(wave_range,
                         claims["independent_wavelength_closure"]["interval"])))

    kinds = [claim["uncertainty_kind"] for claim in claims.values()]
    checks.append(("uncertainty semantics are explicit and non-pooled",
                   all("uncertainty_kind" in claim for claim in claims.values()) and
                   any("deterministic" in kind for kind in kinds) and
                   any("95pct_interval" in kind for kind in kinds) and
                   not any("deterministic_confidence" in kind for kind in kinds)))

    tex = active_tex((ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8"))
    checks.append(("active manuscript build carries the complete certificate table",
                   r"\label{tab:headline_uncertainty}" in tex and
                   "5000" in tex and "deterministic envelope" in tex))

    # 2026-08-23 (writing node_006): the PRINTED certificate-table values are
    # bound to this frozen artifact.  Convention: every decimal in a bound row
    # equals the certified value rounded to three decimals (percent rows are
    # scaled by 100 first; the Hausmann VF-WMLES row uses its artifact's four
    # decimals).  Rows are anchored by unique label text; the decimals on each
    # anchored line are extracted in print order and compared exactly.  Both
    # the active tree and the flat submission tree are checked.
    def fmt(value, places=3):
        return float(f"{float(value):.{places}f}")

    hausmann = json.loads(
        (RESULTS / "hausmann_vfwmles_reproduction_l1.json").read_text())["summaries"]
    expected_rows: list[tuple[str, list[float]]] = [
        ("Independent phase closure",
         [fmt(100 * claims["independent_phase_closure"]["point"])]
         + [fmt(100 * v) for v in claims["independent_phase_closure"]["interval"]]),
        ("Independent wavelength closure",
         [fmt(100 * claims["independent_wavelength_closure"]["point"])]
         + [fmt(100 * v) for v in claims["independent_wavelength_closure"]["interval"]]),
        ("Public VF-WMLES Vreman",
         [fmt(hausmann[case]["LESVRE"]["median_relative_l2_u"], 4)
          for case in ("0035", "0070")]
         + [fmt(hausmann[case]["LESVRE"]["max_relative_l2_u"], 4)
            for case in ("0035", "0070")]),
        ("Streamwise Reynolds-stress fraction",
         [fmt(v) for v in claims["thin_layer_reynolds_range"]["interval"]]),
        ("Streamwise/normal viscous ratio",
         [fmt(v) for v in claims["thin_layer_viscous_range"]["interval"]]),
        (r"$|\tau_m|/\Phi$, canonical hill",
         [fmt(claims["three_term_hill_ratio"]["point"])]
         + [fmt(v) for v in claims["three_term_hill_ratio"]["interval"]]),
        (r"$\Rtwo(\tau_w)$, canonical hill",
         [fmt(claims["canonical_hill_r2"]["point"])]
         + [fmt(v) for v in claims["canonical_hill_r2"]["interval"]]),
        (r"$\Rtwo(\tau_w)$, curved step",
         [fmt(claims["curved_bfs_r2"]["point"])]
         + [fmt(v) for v in claims["curved_bfs_r2"]["interval"]]),
        (r"$\Rtwo(\tau_w)$, exact-stress control",
         [fmt(claims["corrected_exact_stress_r2"]["point"])]
         + [fmt(v) for v in claims["corrected_exact_stress_r2"]["interval"]]),
        (r"$\Rtwo(\tau_w)$, 29-hill range",
         [fmt(v) for v in claims["xiao_family_r2_range"]["point"]]),
        (r"Coupled finest-grid equilibrium $E_\tau$",
         [fmt(claims["coupled_finest_traction_errors"]["point"][0])]
         + [fmt(v) for v in claims["coupled_finest_traction_errors"]["intervals"][0]]),
        (r"Coupled finest-grid total-gradient $E_\tau$",
         [fmt(claims["coupled_finest_traction_errors"]["point"][1])]
         + [fmt(v) for v in claims["coupled_finest_traction_errors"]["intervals"][1]]),
        (r"Coupled finest-grid $\Rtwo(\tau_s)$",
         [fmt(v) for v in claims["coupled_finest_r2"]["point"]]
         + [fmt(v) for pair in claims["coupled_finest_r2"]["intervals"] for v in pair]
         + [fmt(v) for pair in claims["coupled_finest_r2"]["window_intervals"] for v in pair]),
        ("Coupled Holm-adjusted failure",
         [fmt(v) for v in claims["coupled_holm_p"]["point"]]),
        (r"Coupled TBLE--equilibrium $\Delta E_\tau$",
         [fmt(claims["coupled_model_difference_p"]["point"][0])]
         + [fmt(v) for v in claims["coupled_model_difference_p"]["interval"]]
         + [float(f"{claims['coupled_model_difference_p']['point'][1]:.4f}")]),
        (r"$Re_H=10595$ coupled finest-grid $E_\tau$",
         [fmt(v) for v in claims["re10595_finest_traction_errors"]["point"]]
         + [fmt(v) for pair in claims["re10595_finest_traction_errors"]["intervals"]
            for v in pair]),
        (r"$Re_H=10595$ coupled finest-grid $R^2(\tau_s)$",
         [fmt(v) for v in claims["re10595_finest_r2"]["point"]]
         + [fmt(v) for pair in claims["re10595_finest_r2"]["intervals"] for v in pair]
         + [fmt(v) for pair in claims["re10595_finest_r2"]["window_intervals"] for v in pair]),
        (r"$Re_H=10595$ coupled $x_r/H$",
         [fmt(v) for v in claims["re10595_finest_reattachment"]["point"]]
         + [fmt((claims["re10595_finest_reattachment"]["interval"][1]
                - claims["re10595_finest_reattachment"]["interval"][0]) / 2.0)]),
    ]

    for tree_label, tree_path in (
            ("active tree", ROOT / "manuscript" / "main.tex"),
            ("flat submission tree",
             ROOT / "manuscript" / "submission_flat" / "main.tex")):
        tree_tex = active_tex(tree_path.read_text(encoding="utf-8"))
        start = tree_tex.find(r"\label{tab:headline_uncertainty}")
        # Operator 2026-08-26: the certificate table is a tabularx, so it has no
        # \end{tabular} of its own.  Terminating on that token silently extended
        # the slice into whatever table came next, and once the neighbouring
        # tables were converted to figures the token vanished and every row read
        # as missing.  The slice now ends where the table environment ends.
        end = tree_tex.find(r"\end{table}", max(start, 0))
        table = tree_tex[start:end] if start >= 0 and end > start else ""
        bad: list[str] = []
        for anchor, expected in expected_rows:
            rows = [line for line in table.splitlines() if anchor in line]
            if len(rows) != 1:
                bad.append(f"{anchor} (found {len(rows)} rows)")
                continue
            printed = [float(tok) for tok in re.findall(r"-?\d*\.\d+", rows[0])]
            if len(printed) != len(expected) or not close(printed, expected, atol=1e-9):
                bad.append(f"{anchor} printed={printed} expected={expected}")
        checks.append((f"printed certificate-table values equal the frozen artifact "
                       f"({tree_label}, {len(expected_rows)} rows parsed)", not bad))
        if bad:
            for entry in bad:
                print("  stale row: " + entry)

    # Red fixture: perturbing one printed digit must be detected.
    red_row = next(line for line in table.splitlines()
                   if r"$\Rtwo(\tau_w)$, canonical hill" in line)
    # The perturbed digit tracks whatever value is currently certified, so the
    # fixture keeps working across a reference change instead of silently
    # becoming a no-op against a stale literal.
    _certified = f"{claims['canonical_hill_r2']['point']:.3f}"
    _perturbed = f"{claims['canonical_hill_r2']['point'] + 0.001:.3f}"
    if _certified not in red_row:
        raise RuntimeError("red fixture cannot find the certified value in the table row")
    red_numbers = [float(tok) for tok in
                   re.findall(r"-?\d*\.\d+", red_row.replace(_certified, _perturbed, 1))]
    canonical_expected = ([fmt(claims["canonical_hill_r2"]["point"])]
                          + [fmt(v) for v in claims["canonical_hill_r2"]["interval"]])
    checks.append(("red fixture: a one-digit table perturbation is detected",
                   not close(red_numbers, canonical_expected, atol=1e-9)))

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in checks)
    print(f"M1 verifier: {passed}/{len(checks)} PASS")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
