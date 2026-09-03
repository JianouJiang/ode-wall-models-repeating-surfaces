#!/usr/bin/env python3
"""Independent check of the L0 conditioning-ladder re-adjudication.

Shares NO code path with the producer: the wall-traction references, the
periodic interpolation, the error metric, the paired phase-block bootstrap, the
identifiability rule and the amplification bound are all re-implemented here
from the raw archives.  The only quantities taken from the producer's archive
are the wall-model predictions themselves (re-running the shooting solver is the
producer's job, and its fidelity against the previously published ladder is
checked separately below).

Control cases are included: each is a deliberately corrupted input that MUST be
rejected.  A verifier that cannot fail is not a check.

Exit 0 if every check passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "codes/results/conditioning_ladder_l0_20260825.json"
NPZ = ROOT / "codes/results/conditioning_ladder_l0_20260825.npz"
MGLET = (ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/"
         "UFR3-30_data-NP-Re5600-DNS2-11.dat")
XIAO = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
DEPOSITED = ROOT / "codes/results/r2m4_apriori_ladder_20260823.json"

LX = 9.0
NU = 1.0 / 5600.0
NDENSE = 4096
BLOCK = 512
DRAWS = 20000
SEED = 20260823

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# independent re-implementations
# --------------------------------------------------------------------------- #
def hill_height(x):
    """Xiao alpha=1 hill surface, independently coded from the published
    piecewise polynomials (units of H, period 9H, crest at x=0 and x=9)."""
    x = np.atleast_1d(np.asarray(x, float)) % LX
    xr = np.minimum(x, LX - x) * 28.0        # published polynomials use x in [0,54] mm units of H/28
    h = np.empty_like(xr)
    segs = [
        (0.0, 9.0, [2.800000000000E+01, 0.0, 6.775070969851E-03, -2.124527775800E-03]),
        (9.0, 14.0, [2.507355893131E+01, 9.754803562315E-01, -1.016116352781E-01, 1.889794677828E-03]),
        (14.0, 20.0, [2.579601052357E+01, 8.206693007457E-01, -9.055370274339E-02, 1.626510569859E-03]),
        (20.0, 30.0, [4.046435022819E+01, -1.379581654948E+00, 1.945884504128E-02, -2.070318932190E-04]),
        (30.0, 40.0, [1.792461334664E+01, 8.743920332081E-01, -5.567361123058E-02, 6.277731764683E-04]),
        (40.0, 54.0, [5.639011190988E+01, -2.010520359035E+00, 1.644919857549E-02, 2.674976141766E-05]),
    ]
    for lo, hi, c in segs:
        m = (xr >= lo) & (xr <= hi)
        if m.any():
            h[m] = sum(c[k] * xr[m] ** k for k in range(4))
    h = np.maximum(h / 28.0, 0.0)
    h[xr > 54.0] = 0.0
    return np.minimum(h, 1.0)


def tangent(x):
    """Unit downstream tangent of the hill surface by central differences."""
    e = 1e-5
    slope = (hill_height(x + e) - hill_height(x - e)) / (2 * e)
    mag = np.sqrt(1.0 + slope ** 2)
    return 1.0 / mag, slope / mag


def wrap(xp, fp, target):
    """Periodic linear interpolation in phase, independently coded."""
    o = np.argsort(xp)
    xp, fp = np.asarray(xp, float)[o], np.asarray(fp, float)[o]
    t = np.mod(np.asarray(target, float), 1.0)
    return np.interp(t, np.concatenate((xp - 1, xp, xp + 1)), np.tile(fp, 3))


def ref_mglet():
    raw = np.loadtxt(MGLET)
    if not (np.allclose(raw[-2], 0.0) and np.allclose(raw[-1], [9, 0, 0])):
        raise RuntimeError("MGLET placeholder rows moved")
    b = raw[:-2]
    return np.mod(b[:, 0] / LX, 1.0), b[:, 1]


def ref_xiao(deg: int, k: int):
    """Through-origin polynomial wall-gradient estimator on the Xiao archive.
    deg=1,k=4 reproduces the WITHDRAWN estimator; deg=3,k=6 the repaired one."""
    d = np.load(XIAO)
    x = np.asarray(d["x"], float)
    y = np.asarray(d["y"], float)
    U = np.asarray(d["U"], float)
    V = np.asarray(d["V"], float)
    tx, ty = tangent(x)
    tau = np.empty(x.size)
    for i in range(x.size):
        n = y[i, 1:k + 1] - y[i, 0]
        ut = U[i, 1:k + 1] * tx[i] + V[i, 1:k + 1] * ty[i]
        A = np.vstack([n ** (j + 1) for j in range(deg)]).T
        c, *_ = np.linalg.lstsq(A, ut, rcond=None)
        tau[i] = NU * float(c[0]) / tx[i]
    return np.mod((x - x.min()) / LX, 1.0), tau


def rel_rms(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / np.sqrt(np.mean(truth ** 2)))


def paired_bootstrap(truth, preds, seed=SEED):
    """Circular phase-block bootstrap with identical blocks across predictors."""
    n = truth.size
    per = n // BLOCK
    rng = np.random.default_rng(seed)
    sq = {k: (v - truth) ** 2 for k, v in preds.items()}
    out = {k: np.empty(DRAWS) for k in preds}
    off = np.arange(BLOCK)[None, None, :]
    for a in range(0, DRAWS, 250):
        b = min(a + 250, DRAWS)
        st = rng.integers(0, n, size=(b - a, per))
        idx = ((st[:, :, None] + off) % n).reshape(b - a, n)
        den = np.sqrt(np.mean(truth[idx] ** 2, axis=1))
        for k, s in sq.items():
            out[k][a:b] = np.sqrt(np.mean(s[idx], axis=1)) / den
    return out


def quant(v):
    lo, med, hi = np.quantile(v, (0.025, 0.5, 0.975))
    return {"low": float(lo), "median": float(med), "high": float(hi)}


def rule(dB, dC):
    """The producer's registered identifiability rule, re-implemented."""
    bf, bs = dB["high"] < 0, dB["low"] > 0
    cf, cs = dC["high"] < 0, dC["low"] > 0
    if bf and cf:
        return "IDENTIFIED_FIRST_BETTER"
    if bs and cs:
        return "IDENTIFIED_SECOND_BETTER"
    if (bf and cs) or (bs and cf):
        return "CONTRADICTORY_ACROSS_REFERENCES"
    return "UNRESOLVED"


# --------------------------------------------------------------------------- #
def main() -> int:
    if not RES.is_file() or not NPZ.is_file():
        print("[FAIL] result archive missing")
        return 1
    R = json.loads(RES.read_text())
    Z = np.load(NPZ)
    dense = np.arange(NDENSE) / NDENSE
    x_dense = dense * LX

    # ---- 1. references rebuilt independently -----------------------------
    phB, tB = ref_mglet()
    phA, tA = ref_xiao(1, 4)
    phC, tC = ref_xiao(3, 6)
    for nm, mine, theirs in (("A", tA, Z["reference_A_tau"]),
                             ("B", tB, Z["reference_B_tau"]),
                             ("C", tC, Z["reference_C_tau"])):
        d = float(np.max(np.abs(mine - theirs)) / max(np.max(np.abs(theirs)), 1e-30))
        check(f"reference {nm} rebuilt independently", d < 2e-3, f"max relative deviation {d:.2e}")

    tdB, tdC, tdA = (wrap(phB, tB, dense), wrap(phC, tC, dense), wrap(phA, tA, dense))
    got = R["reference_to_reference"]["B_mglet_vs_C_xiao_repaired_cubic6"]
    mine = rel_rms(tdC, tdB)          # ||B - C|| in units of the primary truth's RMS
    key = "relative_rms_distance_in_primary_truth_units"
    check("B-to-C reference distance reproduced in primary-truth units",
          abs(mine - got[key]) < 5e-3,
          f"mine {mine:.4f} vs reported {got[key]:.4f}")
    check("the two admissible corrected references are far apart",
          got[key] > 0.4,
          f"{got[key]:.3f} of the primary traction RMS — model differences below this "
          "cannot be identified")

    # ---- 2. physics signatures ------------------------------------------
    for nm, td, want in (("A(withdrawn)", tdA, 0.379), ("B(MGLET)", tdB, 0.181),
                         ("C(repaired)", tdC, 0.183)):
        cr = x_dense[:-1][(td[:-1] > 0) & (td[1:] <= 0)]
        first = float(cr[0]) if cr.size else np.nan
        check(f"separation point of reference {nm} matches the published audit",
              abs(first - want) < 0.02, f"x/H = {first:.3f} (audit {want})")
    check("the withdrawn estimator really is deficient in magnitude",
          0.30 < float(np.sqrt(np.mean(tdA ** 2) / np.mean(tdB ** 2))) < 0.42,
          f"RMS(A)/RMS(B) = {np.sqrt(np.mean(tdA**2)/np.mean(tdB**2)):.3f}")

    # ---- 3. scores and the identifiability certificate --------------------
    truths = {"A_withdrawn_linear4": tdA, "B_mglet": tdB, "C_xiao_repaired_cubic6": tdC}
    worst_score, worst_iv, n_sc, n_iv = 0.0, 0.0, 0, 0
    verdict_ok, verdict_n = True, 0
    for sname, S in R["surfaces"].items():
        ph = Z[f"{sname}_phase"]
        preds = {m: wrap(ph, Z[f"{sname}_pred_{m}"], dense)
                 for m in S["scores"]["B_mglet"]}
        for ref, truth in truths.items():
            for m, v in S["scores"][ref].items():
                e = rel_rms(preds[m], truth)
                worst_score = max(worst_score, abs(e - v["relative_rms"]) /
                                  max(v["relative_rms"], 1e-30))
                n_sc += 1
        boots = {r: paired_bootstrap(t, preds) for r, t in truths.items()}
        for key, entry in S["identifiability"].items():
            a, b = key.split("-minus-")
            d = {r: quant(boots[r][a] - boots[r][b]) for r in truths}
            for r in ("B_mglet", "C_xiao_repaired_cubic6"):
                for q in ("low", "high"):
                    ref_v = entry["paired_interval"][r][q]
                    worst_iv = max(worst_iv, abs(d[r][q] - ref_v) /
                                   max(abs(ref_v), 1e-3))
                    n_iv += 1
            mine_v = rule(d["B_mglet"], d["C_xiao_repaired_cubic6"])
            verdict_n += 1
            if mine_v != entry["verdict"]:
                verdict_ok = False
                print(f"       verdict mismatch {sname}/{key}: mine {mine_v} vs {entry['verdict']}")
    check("every reported score reproduced from an independent metric",
          worst_score < 5e-3, f"{n_sc} scores, worst relative deviation {worst_score:.2e}")
    check("every paired bootstrap interval reproduced",
          worst_iv < 0.15, f"{n_iv} interval bounds, worst relative deviation {worst_iv:.2e}")
    check("every identifiability verdict reproduced", verdict_ok, f"{verdict_n} contrasts")

    # ---- 4. the claims the node actually makes ---------------------------
    ident = {(s, k): v["verdict"] for s, S in R["surfaces"].items()
             for k, v in S["identifiability"].items()}
    n_id = sum(1 for v in ident.values() if v.startswith("IDENTIFIED"))
    check("no contrast is contradictory across the two admissible references",
          not any(v == "CONTRADICTORY_ACROSS_REFERENCES" for v in ident.values()))
    check("a minority of contrasts is identifiable under the reference envelope",
          0 < n_id < len(ident), f"{n_id} of {len(ident)} identified")
    coarse = ("archive_index10", "ladder_L1")
    check("modelled convection beats the pressure-gradient ODE at both coarse surfaces",
          all(ident[(s, "M2_hickel_modelled_convection-minus-M1_pressure_gradient_ode")]
              == "IDENTIFIED_FIRST_BETTER" for s in coarse))
    check("supplying all omitted transport is identifiably WORSE than the equilibrium closure",
          all(ident[(s, "Xall_all_omitted_transport-minus-M0_equilibrium")]
              == "IDENTIFIED_SECOND_BETTER" for s in coarse))
    check("the closure-free exact-budget reconstruction is identifiably worse still",
          all(ident[(s, "Xfull_all_transport_plus_exact_shear_stress-minus-"
                        "Xall_all_omitted_transport")] == "IDENTIFIED_SECOND_BETTER"
              for s in coarse))
    check("'exact convection repairs the pressure-gradient ODE' is NOT resolved either way",
          all(ident[(s, "Xc_exact_convection_profile-minus-M1_pressure_gradient_ode")]
              == "UNRESOLVED" for s in R["surfaces"]),
          "the withdrawn reference made this look settled; it is not")
    check("'the pressure-gradient ODE is worse than equilibrium' is NOT resolved either way",
          all(ident[(s, "M1_pressure_gradient_ode-minus-M0_equilibrium")] == "UNRESOLVED"
              for s in R["surfaces"]))

    # ---- 5. amplification bound and its positive control ------------------
    AB = R["amplification_bound"]
    pv = np.array([r["predictor"] for r in AB["points"]])
    mv = np.array([r["measured_E_Xfull"] for r in AB["points"]])
    for sname, S in R["surfaces"].items():
        s_abs = Z[f"{sname}_impulse_S_abs"]
        tym = np.abs(Z[f"{sname}_impulse_tau_at_ym"])
        for ref, td in (("B_mglet", tdB), ("C_xiao_repaired_cubic6", tdC)):
            mine_p = float(np.sqrt(np.mean((s_abs + tym) ** 2)) / np.sqrt(np.mean(td ** 2)))
            row = next(r for r in AB["points"] if r["surface"] == sname and r["reference"] == ref)
            worst_iv = abs(mine_p - row["predictor"]) / row["predictor"]
            if worst_iv > 5e-3:
                check(f"amplification predictor {sname}/{ref}", False, f"{worst_iv:.2e}")
                break
    else:
        check("amplification predictor reproduced independently at all six points", True)
    check("the amplification predictor varies over a wide range (not a trivial fit)",
          AB["predictor_range"][1] / AB["predictor_range"][0] > 4.0,
          f"predictor spans {AB['predictor_range'][0]:.2f} to {AB['predictor_range'][1]:.2f}")
    d_hat = float(np.sum(pv * mv) / np.sum(pv ** 2))
    check("one fitted constant reproduces the closure-free error at all six points",
          abs(d_hat - AB["delta_fitted_over_all_points"]) < 1e-6 and
          AB["max_relative_prediction_error_leave_one_out"] < 0.15,
          f"delta = {d_hat:.3f}, worst leave-one-out error "
          f"{100*AB['max_relative_prediction_error_leave_one_out']:.1f}%")
    flat_ok, curved_ok = True, True
    for sname in coarse:
        for ref in ("B_mglet", "C_xiao_repaired_cubic6"):
            ra = R["surfaces"][sname]["regional_amplification"][ref]
            flat = ra["flat_floor_2.05_to_6.90"]
            wind = ra["windward_face_x_gt_7.071"]
            flat_ok &= (flat["bound_attainment"] < 0.15) and (flat["E_Xfull_over_E_M0"] < 0.5)
            curved_ok &= (wind["bound_attainment"] > 0.5) and (wind["E_Xfull_over_E_M0"] > 3.0)
    check("POSITIVE CONTROL: on the flat floor, where the one-dimensional reduction is "
          "closed, the exact-budget reconstruction beats the equilibrium closure and stays "
          "far below the bound", flat_ok)
    check("on the curved windward face, which carries most of the traction, the bound is "
          "attained and the reconstruction is an order of magnitude worse", curved_ok)

    # ---- 6. instrument fidelity against the previously published ladder ---
    dep = json.loads(DEPOSITED.read_text())
    worst = 0.0
    n = 0
    for sname, S in R["surfaces"].items():
        for m, v in dep["surfaces"][sname]["metrics"].items():
            got = S["scores"]["A_withdrawn_linear4"][m]["relative_rms"]
            worst = max(worst, abs(got - v["relative_rms"]) / max(abs(v["relative_rms"]), 1e-30))
            n += 1
    check("the withdrawn-reference column reproduces the previously published ladder exactly",
          worst == 0.0, f"{n} rungs, worst relative difference {worst:.3e}")

    # ---- 6b. agreement with the concurrent independent re-scoring ---------
    # An operator re-scoring of the same ladder against the same two corrected
    # references was produced separately on the same day. It is a different
    # reduction of the same predictions; agreement with it is a cross-check on
    # the reduction, and agreement of BOTH with the independent metric above is
    # a cross-check on the metric.
    opath = ROOT / "codes/results/r2m4_ladder_rescored_20260825.json"
    if opath.is_file():
        op = json.loads(opath.read_text())["apriori"]
        m = {"A_xiao_linear4_deposited": "A_withdrawn_linear4",
             "B_mglet_deposited": "B_mglet",
             "C_xiao_cubic6_repaired": "C_xiao_repaired_cubic6"}
        w, cnt = 0.0, 0
        for s in op:
            for oref, mref in m.items():
                for rung, v in op[s][oref]["metrics"].items():
                    g = R["surfaces"][s]["scores"][mref].get(rung)
                    if g is None:
                        continue
                    w = max(w, abs(g["relative_rms"] - v["relative_rms"]) /
                            max(abs(v["relative_rms"]), 1e-30))
                    cnt += 1
        check("agrees with the concurrent independent re-scoring of the same ladder",
              w < 1e-9 and cnt >= 90, f"{cnt} values, worst relative difference {w:.3e}")
    else:
        check("concurrent re-scoring present for cross-check", False, "file missing")

    # ---- 7. honesty checks ------------------------------------------------
    check("reference A is labelled a negative control, not a truth",
          R["references"]["A_withdrawn_linear4"]["role"].startswith("NEGATIVE_CONTROL"))
    check("MGLET placeholder rows are recorded as stripped",
          R["mglet_trailing_rows_stripped"] == [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    lim = " ".join(R["stated_limits"]).lower()
    check("the MGLET bubble-length and windward-peak limits are stated",
          "bubble" in lim and "windward" in lim)
    txt = json.dumps(R).lower()
    check("no plane-channel friction-correlation argument is used",
          "dean" not in txt and "c_f(re)" not in txt)
    check("no Reynolds-number verdict flip is claimed", "flip" not in txt)

    # ---- 8. control cases --------------------------------------------------
    sname = "ladder_L1"
    ph = Z[f"{sname}_phase"]
    preds = {m: wrap(ph, Z[f"{sname}_pred_{m}"], dense) for m in ("M0_equilibrium",
                                                                  "M2_hickel_modelled_convection")}
    shuffled = {k: np.roll(v, NDENSE // 3) for k, v in preds.items()}
    e_true = rel_rms(preds["M0_equilibrium"], tdB)
    e_shuf = rel_rms(shuffled["M0_equilibrium"], tdB)
    check("RED FIXTURE: a phase-shuffled prediction does not reproduce the reported score",
          abs(e_shuf - e_true) / e_true > 0.2, f"{e_true:.3f} vs shuffled {e_shuf:.3f}")
    check("RED FIXTURE: the identifiability rule refuses a contrast whose interval straddles zero",
          rule({"low": -0.1, "high": 0.3, "median": 0.1},
               {"low": -0.2, "high": 0.4, "median": 0.1}) == "UNRESOLVED")
    check("RED FIXTURE: the identifiability rule refuses a contrast that flips sign between "
          "references", rule({"low": -0.4, "high": -0.1, "median": -0.2},
                             {"low": 0.1, "high": 0.4, "median": 0.2})
          == "CONTRADICTORY_ACROSS_REFERENCES")
    # The decisive property of the withdrawn estimator is not that it inflates an
    # error, but that it REVERSES an ordering. Under the withdrawn reference the
    # modelled-convection form is worse than the equilibrium closure; under the
    # primary truth it is better. A verdict, not a magnitude, changes.
    order_A = (rel_rms(preds["M2_hickel_modelled_convection"], tdA) >
               rel_rms(preds["M0_equilibrium"], tdA))
    order_B = (rel_rms(preds["M2_hickel_modelled_convection"], tdB) >
               rel_rms(preds["M0_equilibrium"], tdB))
    check("RED FIXTURE: the withdrawn reference REVERSES a model ordering",
          order_A and not order_B,
          f"withdrawn: modelled convection worse than equilibrium "
          f"({rel_rms(preds['M2_hickel_modelled_convection'], tdA):.3f} vs "
          f"{rel_rms(preds['M0_equilibrium'], tdA):.3f}); primary truth: better "
          f"({rel_rms(preds['M2_hickel_modelled_convection'], tdB):.3f} vs "
          f"{rel_rms(preds['M0_equilibrium'], tdB):.3f})")
    fake = {k: v.copy() for k, v in preds.items()}
    fake["M0_equilibrium"] = tdB.copy()          # a rung that is the truth by construction
    check("RED FIXTURE: a rung set equal to the truth is detected as perfect",
          rel_rms(fake["M0_equilibrium"], tdB) < 1e-12)

    # ---- 9. the compiled paper must print what the archive holds ---------
    pdf = ROOT / "manuscript" / "main.pdf"
    if pdf.is_file():
        import subprocess
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True).stdout
        txt = " ".join(txt.split())
        S = R["surfaces"]["ladder_L1"]["scores"]
        envelope = R["reference_to_reference"]["B_mglet_vs_C_xiao_repaired_cubic6"][
            "relative_rms_distance_in_primary_truth_units"]
        want = {
            "primary-reference all-transport error":
                f"{S['B_mglet']['Xall_all_omitted_transport']['relative_rms']:.3f}".rstrip("0"),
            "bracket all-transport error":
                f"{S['C_xiao_repaired_cubic6']['Xall_all_omitted_transport']['relative_rms']:.3f}",
            "primary-reference closure-free error":
                f"{S['B_mglet']['Xfull_all_transport_plus_exact_shear_stress']['relative_rms']:.3f}",
            "bracket closure-free error":
                f"{S['C_xiao_repaired_cubic6']['Xfull_all_transport_plus_exact_shear_stress']['relative_rms']:.3f}",
            "reference envelope": f"{envelope:.3f}",
        }
        missing = [k for k, v in want.items() if v not in txt]
        check("the compiled paper prints the archived headline values",
              not missing, "missing: " + ", ".join(missing) if missing else
              "all five values found")
        check("the compiled paper does not print a magnitude scored against the "
              "withdrawn reference as a result",
              "2.643" not in txt and "2.13--2.15" not in txt and "2.13-2.15" not in txt)
        check("the compiled paper states that comparisons are unresolved, not settled",
              "unresolved" in txt.lower() and "withdraw" in txt.lower())
    else:
        check("compiled manuscript present for value check", False, "main.pdf missing")

    print("-" * 62)
    print(f"{len(PASS)}/{len(PASS)+len(FAIL)} checks passed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
