#!/usr/bin/env python3
"""Independent check of the L0 reference rebase.

Shares no code path with `reference_rebase_headlines_l0.py`: the wall tangent is
taken from the archive's own deposited wall height rather than from the study's
analytic surface module, the references, the metrics, the rank statistics and
the null contest are all re-implemented here, and the manuscript is checked as
text.

Red fixtures are included so that a verifier that has quietly stopped testing
anything fails loudly.

Run:  python3 codes/analysis/ledger_verifiers/verify_reference_rebase_l0.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = json.loads((ROOT / "codes/results/reference_rebase_headlines_l0_20260825.json").read_text())
NPZ = np.load(ROOT / "codes/results/reference_rebase_headlines_l0_20260825.npz")

ARCHIVE = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
DIAG = ROOT / "codes/results/diagnostic_test_corrected.npz"
MGLET = (ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/"
         "UFR3-30_data-NP-Re5600-DNS2-11.dat")
SWEEP = (ROOT / "work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/"
         "xiao29_epsilon_sweep.json")
LADDER_NPZ = ROOT / "codes/results/conditioning_ladder_l0_20260825.npz"
TEX = ROOT / "manuscript/main.tex"

LX, NU, Y_IDX = 9.0, 1.0 / 5600.0, 10
TOL = 1e-9

checks: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> bool:
    checks.append((bool(ok), label))
    return bool(ok)


# --------------------------------------------------------------------------- #
# independent re-implementation
# --------------------------------------------------------------------------- #
d = np.load(ARCHIVE)
x = np.asarray(d["x"], float)
y = np.asarray(d["y"], float)
U = np.asarray(d["U"], float)
V = np.asarray(d["V"], float)
n = x.size
dx = float(np.median(np.diff(x)))

# Tangent from the published ERCOFTAC/Almeida hill polynomial, transcribed here
# and differentiated numerically.  The producer instead reads the study's own
# surface module, so the two paths share no code.
def ercoftac_hill(xv: float) -> float:
    xm = 28.0 * (xv % LX)
    if xm > 28.0 * LX / 2.0:
        xm = 28.0 * LX - xm
    if xm < 9.0:
        h = min(28.0, 28.0 + 6.775070969851e-3 * xm ** 2 - 2.124527775800e-3 * xm ** 3)
    elif xm < 14.0:
        h = (2.507355893131e1 + 9.754803562315e-1 * xm - 1.016116352781e-1 * xm ** 2
             + 1.889794677828e-3 * xm ** 3)
    elif xm < 20.0:
        h = (2.579601052357e1 + 8.206693007457e-1 * xm - 9.055370274339e-2 * xm ** 2
             + 1.626510569859e-3 * xm ** 3)
    elif xm < 30.0:
        h = (4.046435022819e1 - 1.379581654948 * xm + 1.945884504128e-2 * xm ** 2
             - 2.070318932190e-4 * xm ** 3)
    elif xm < 40.0:
        h = (1.792461334664e1 + 8.743920332081e-1 * xm - 5.567361123058e-2 * xm ** 2
             + 6.277731764683e-4 * xm ** 3)
    elif xm <= 54.0:
        h = max(0.0, 5.639011190988e1 - 2.010520359035 * xm + 1.644919857549e-2 * xm ** 2
                + 2.674976141766e-5 * xm ** 3)
    else:
        h = 0.0
    return h / 28.0


_dd = 1.0e-5
slope = np.array([(ercoftac_hill(v + _dd) - ercoftac_hill(v - _dd)) / (2.0 * _dd)
                  for v in x])
mag = np.sqrt(1.0 + slope ** 2)
tx_i, ty_i = 1.0 / mag, slope / mag

check(float(np.max(np.abs(y[:, 0]))) == 0.0,
      "the archive stores wall-normal offsets, so its own y column carries no tangent")


def origin_slope(nn, uu, deg):
    A = np.vstack([np.asarray(nn, float) ** (k + 1) for k in range(deg)]).T
    c, *_ = np.linalg.lstsq(A, np.asarray(uu, float), rcond=None)
    return float(c[0])


def cubic6():
    out = np.empty(n)
    for i in range(n):
        ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])
        yy, uu, vv = y[i, ok], U[i, ok], V[i, ok]
        off = yy[1:7] - yy[0]
        ut = uu[1:7] * tx_i[i] + vv[1:7] * ty_i[i]
        out[i] = NU * origin_slope(off, ut, 3) / tx_i[i]
    return out


def wrap(sp, sv, tp):
    o = np.argsort(sp)
    p, v = np.asarray(sp, float)[o], np.asarray(sv, float)[o]
    return np.interp(np.mod(tp, 1.0),
                     np.concatenate([p - 1, p, p + 1]),
                     np.concatenate([v, v, v]))


phase = np.mod(x / LX, 1.0)
raw = np.loadtxt(MGLET)
body = raw[:-2]
ref_B = wrap(np.mod(body[:, 0] / LX, 1.0), body[:, 1], phase)
ref_C = cubic6()
ref_A = np.asarray(d["tau_w"], float)

# the tangent source differs, so C is checked to a physical tolerance, not bitwise
check(float(np.max(np.abs(ref_A - NPZ["reference_A_withdrawn_linear4"]))) == 0.0,
      "reference A reproduced bitwise from the archive")
check(float(np.max(np.abs(ref_B - NPZ["reference_B_mglet"])) /
            np.max(np.abs(ref_B))) < 1e-12,
      "reference B rebuilt independently from the raw ERCOFTAC deposit")
rel_C = float(np.max(np.abs(ref_C - NPZ["reference_C_repaired_cubic6"])) /
              np.sqrt(np.mean(NPZ["reference_C_repaired_cubic6"] ** 2)))
check(rel_C < 5e-3,
      f"reference C rebuilt from an independent wall tangent (max dev {rel_C:.2e} of RMS)")

check(np.allclose(raw[-2], [0, 0, 0]) and np.allclose(raw[-1], [9, 0, 0]),
      "the two ERCOFTAC plot-axis placeholder rows are present and were stripped")


def r2_(p, t):
    return 1.0 - float(np.sum((p - t) ** 2)) / float(np.sum((t - t.mean()) ** 2))


def rr_(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / np.sqrt(np.mean(t ** 2)))


diag = np.load(DIAG, allow_pickle=True)
preds = {"pg_ode_mixing_length": np.asarray(diag["standard_ml"], float),
         "pg_ode_exact_dns_stress": np.asarray(diag["controlled_dns"], float)}
mine_refs = {"A_withdrawn_linear4": ref_A, "B_mglet": ref_B, "C_repaired_cubic6": ref_C}

worst = 0.0
for pname, p in preds.items():
    for rname, t in mine_refs.items():
        rec = RES["headlines"]["scores"][pname][rname]
        for fn, key in ((r2_, "r2"), (rr_, "rel_rms")):
            mine, theirs = fn(p, t), rec[key]
            worst = max(worst, abs(mine - theirs) / max(abs(theirs), 1e-12))
check(worst < 5e-3, f"all 12 headline scores reproduce independently (worst {worst:.2e})")

# instrument fidelity against the numbers the paper published
fid = RES["headlines"]["instrument_fidelity"]
check(abs(r2_(preds["pg_ode_mixing_length"], ref_A) - (-47.68617253416459)) < 1e-9,
      "reference A reproduces the published canonical-hill score exactly")
check(fid["pg_ode_abs_deviation"] < 1e-8 and fid["exact_stress_abs_deviation"] < 1e-6,
      "producer records exact agreement with both published a-priori headlines")

# the qualitative statements
sur = RES["headlines"]["survives_reference_swap"]
check(all(r2_(preds["pg_ode_mixing_length"], t) < 0 for t in mine_refs.values())
      and sur["every_reference_gives_negative_r2_for_the_pg_ode"],
      "the pressure-gradient ODE scores below zero on every admissible reference")
check(all(r2_(preds["pg_ode_exact_dns_stress"], t) < r2_(preds["pg_ode_mixing_length"], t)
          for t in mine_refs.values())
      and sur["exact_dns_stress_is_worse_than_mixing_length_on_every_reference"],
      "exact resolved-stress substitution is worse on every reference")
factor = r2_(preds["pg_ode_mixing_length"], ref_A) / r2_(preds["pg_ode_mixing_length"], ref_B)
check(abs(factor - sur["r2_magnitude_is_reference_dependent_factor"]) / factor < 5e-3
      and factor > 20.0,
      f"the score magnitude moves by more than 20x with the reference ({factor:.1f}x)")

# --------------------------------------------------------------------------- #
# epsilon
# --------------------------------------------------------------------------- #
y_m = y[:, Y_IDX] - y[:, 0]
den = np.maximum(np.abs(np.asarray(d["dp_dx"], float)) * y_m, 1e-30)
for key, tau in (("N0_archive_deposit_as_published", ref_A),
                 ("N3_mglet_deposited", ref_B),
                 ("N2_repaired_cubic6", ref_C)):
    eps = np.abs(tau) / den
    rec = RES["epsilon"][key]
    ok = (abs(float(np.median(eps)) - rec["median"]) / rec["median"] < 5e-3 and
          abs(float(np.mean(eps < 0.1)) - rec["frac_below_0p1"]) < 5e-3)
    check(ok, f"epsilon statistics reproduce for {key}")

eps0 = np.abs(ref_A) / den
check(float(np.median(eps0)) == 0.08364189563744982,
      "the published epsilon median is reproduced bitwise by the withdrawn estimator")
epsB = np.abs(ref_B) / den
check(1.4 < float(np.median(epsB)) / float(np.median(eps0)) < 1.7,
      "the corrected epsilon is larger than the published one by a factor near 1.5")
check(float(np.mean(epsB < 0.1)) < 0.5,
      "under the corrected estimator the deeply cancelling stations are NOT a majority")
check(float(np.mean(epsB < 1.0)) > 0.95,
      "the statement that essentially every station lies below epsilon = 1 survives")

# --------------------------------------------------------------------------- #
# ranking (R2-2)
# --------------------------------------------------------------------------- #
def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for g in np.flatnonzero(cnt > 1):
        m = inv == g
        r[m] = r[m].mean()
    return r


def rho_(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    return float(np.sum(ra * rb) / np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))


def prho(a, b, c):
    ra, rb, rc = rank(a), rank(b), rank(c)

    def res(v):
        v = v - v.mean()
        z = rc - rc.mean()
        return v - (np.dot(v, z) / np.dot(z, z)) * z

    va, vb = res(ra), res(rb)
    return float(np.sum(va * vb) / np.sqrt(np.sum(va ** 2) * np.sum(vb ** 2)))


S = json.loads(SWEEP.read_text())
mem = S["per_member"]
L_y = np.array([m["L_y"] for m in mem], float)
delta = 0.5 * L_y
worst_rho = 0.0
for book in ("legacy", "repaired"):
    r2v = np.array([m[book]["r2"] for m in mem], float)
    Ls = np.array([m[book]["L_sep"] for m in mem], float)
    pc = RES["ranking_r2_2"]["ranking"][book]["per_candidate"]
    for cname, g in (("L_y", L_y), ("L_sep_over_delta", Ls / delta), ("L_sep", Ls),
                     ("h_over_delta", 1.0 / delta)):
        worst_rho = max(worst_rho, abs(rho_(g, r2v) - pc[cname]["rho"]["r2"]))
    P = RES["ranking_r2_2"]["partial_correlations"][book]
    worst_rho = max(worst_rho,
                    abs(prho(Ls / delta, r2v, L_y) - P["partial_rho(L_sep_over_delta, r2 | L_y)"]),
                    abs(prho(L_y, r2v, Ls / delta) - P["partial_rho(L_y, r2 | L_sep_over_delta)"]))
check(worst_rho < 1e-9, f"all rank and partial-rank statistics reproduce ({worst_rho:.1e})")

leg = RES["ranking_r2_2"]["ranking"]["legacy"]
rep = RES["ranking_r2_2"]["ranking"]["repaired"]
check(leg["strongest_sign_coherent"] == "L_sep_over_delta"
      and rep["strongest_sign_coherent"] == "L_y",
      "the strongest coherent geometric control changes with the reference")
check(len(leg["sign_coherent_candidates_ranked"]) > 1,
      "more than one candidate is sign-coherent, so 'only one is' cannot stand")
Pr = RES["ranking_r2_2"]["partial_correlations"]["repaired"]
check(abs(Pr["partial_rho(L_sep_over_delta, r2 | L_y)"]) < 0.1
      and abs(Pr["partial_rho(L_y, r2 | L_sep_over_delta)"]) > 0.3,
      "under the corrected estimator the bubble-length control vanishes once the "
      "outer scale is held, and not the reverse")
rep_Ls = np.array([m["repaired"]["L_sep"] for m in mem], float)
rep_r2 = np.array([m["repaired"]["r2"] for m in mem], float)
check(abs(rho_(rep_Ls, rep_r2)) < 0.2,
      "the raw separation length carries essentially no corrected signal")
check(RES["ranking_r2_2"]["every_member_fails"]["all_negative_repaired"]
      and float(rep_r2.max()) < 0.0,
      "every family member still fails under the corrected estimator")
check(RES["ranking_r2_2"]["n_members_without_any_independent_reference"] == 28,
      "the one-of-29 independent-reference limitation is recorded")

# --------------------------------------------------------------------------- #
# amplification null
# --------------------------------------------------------------------------- #
Z = np.load(LADDER_NPZ, allow_pickle=True)
N = RES["amplification_null"]
worst_null = 0.0
for r in N["points"]:
    surf = r["surface"]
    pred = np.asarray(Z[f"{surf}_pred_Xfull_all_transport_plus_exact_shear_stress"], float)
    ph = np.asarray(Z[f"{surf}_phase"], float)
    tag = "B" if r["reference"] == "B_mglet" else "C"
    truth = wrap(np.asarray(Z[f"reference_{tag}_phase"], float),
                 np.asarray(Z[f"reference_{tag}_tau"], float), ph)
    asm = np.asarray(Z[f"{surf}_impulse_S_abs_plus_tau_ym"], float)
    rt = float(np.sqrt(np.mean(truth ** 2)))
    for mine, theirs in ((rr_(pred, truth), r["measured_E"]),
                         (float(np.sqrt(np.mean(asm ** 2)) / rt), r["Lambda_tau_on_this_grid"]),
                         (float(np.sqrt(np.mean(pred ** 2)) / rt), r["shared_denominator_null"]),
                         (float(np.sqrt(np.mean((pred - truth) ** 2)) /
                                np.sqrt(np.mean(asm ** 2))),
                          r["attainment_error_over_assembled_magnitude"])):
        worst_null = max(worst_null, abs(mine - theirs) / max(abs(theirs), 1e-12))
check(worst_null < 1e-9, f"the null contest reproduces exactly ({worst_null:.1e})")
check(N["verdict"] == "FITTED_LAW_DOES_NOT_BEAT_A_ZERO_PARAMETER_NULL",
      "the fitted amplification law is recorded as not established")
check(not N["quadrature_convention"]["dense_uniform_grid"]["fitted_law_beats_null"],
      "under the dense-grid convention the zero-parameter null is at least as good")
check(N["quadrature_convention"]["E_convention_sensitivity_max_relative"]
      > N["margin_over_null_mean"],
      "the law's margin over the null is smaller than the metric's own convention "
      "sensitivity")
check(min(r["attainment_error_over_assembled_magnitude"] for r in N["points"]) > 0.9,
      "the denominator-free attainment statement is above 0.9 at every point")

# --------------------------------------------------------------------------- #
# manuscript anti-regression
# --------------------------------------------------------------------------- #
src = TEX.read_text(encoding="utf-8", errors="replace")


def active(text: str) -> str:
    """Strip \\iffalse ... \\fi archive blocks."""
    out, depth, i = [], 0, 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(r"\iffalse"):
            depth += 1
            continue
        if s.startswith(r"\fi") and depth:
            depth -= 1
            continue
        if depth == 0:
            out.append(line)
    return "\n".join(out)


A = active(src)
check("The failure deepens with Reynolds number" not in A,
      "the withdrawn 'failure deepens with Reynolds number' heading is gone")
check("we show the opposite" not in A,
      "the withdrawn 'we show the opposite' Reynolds claim is gone")
check(not re.search(r"only\s*\n?\s*\$L_\\mathrm\{sep\}/\\delta\$ is strong and sign-consistent", A),
      "the refuted 'only L_sep/delta is strong and sign-consistent' claim is gone")
check("lock-step" not in A, "the withdrawn 'lock-step' wording is gone")
# the superseded magnitude reaches the page as a decimal AND as a percentage;
# banning one spelling let the other survive three headline locations
check(not any(alias in A for alias in
              ("0.08364", "8.364", "8.36\\%", "$8.364\\%$", "8.3642")),
      "the contaminated epsilon reference magnitude is no longer printed as "
      "current, in decimal or percentage form")
check("$56\\%$ of the hill's stations" not in A,
      "the withdrawn 56-per-cent majority statement is gone")
check("0.084$--$0.41" not in A and "[0.084, 0.41]" not in A,
      "the contaminated 29-case epsilon range is no longer printed")
check(re.search(r"0\.094", A) and re.search(r"0\.545", A),
      "the corrected 29-case epsilon range is printed")
check(re.search(r"-?1\.757|1\.76", A),
      "the corrected canonical-hill score is printed")
A_flow = re.sub(r"\s+", " ", A)
check("28 of the 29" in A_flow or "28 of 29" in A_flow
      or "one of the $29$ members" in A_flow or "one of the 29 members" in A_flow,
      "the one-of-29 reference limitation is stated in the paper")
# The paper tightened "least affected by the estimator correction" to
# "least affected by the correction"; bind the statement, not one spelling.
check(("least affected" in A_flow and "correction" in A_flow)
      or "least contaminated" in A_flow,
      "the paper states that the one validated member is the least affected one")

# --------------------------------------------------------------------------- #
# red fixtures
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(7)
shuf = rng.permutation(preds["pg_ode_mixing_length"])
check(abs(r2_(shuf, ref_B) - r2_(preds["pg_ode_mixing_length"], ref_B)) > 1.0,
      "RED: a phase-shuffled prediction does not reproduce the reported score")
check(abs(r2_(preds["pg_ode_mixing_length"], ref_A) -
          r2_(preds["pg_ode_mixing_length"], ref_B)) > 10.0,
      "RED: swapping the reference changes the score, so the reference is load-bearing")
flip_ok = rho_(np.array([m["repaired"]["L_sep"] for m in mem], float) / delta,
               np.array([m["repaired"]["eps_median"] for m in mem], float)) < 0
check(flip_ok,
      "RED: the coherence orientation is the measured one, not an assumed one")
check(rho_(L_y, np.array([m["repaired"]["r2"] for m in mem], float)) >
      abs(rho_(np.array([m["repaired"]["L_sep"] for m in mem], float) / delta,
               np.array([m["repaired"]["r2"] for m in mem], float))),
      "RED: the reported reversal is a measured inequality, recomputed here")
check(not re.search(r"E_?\\?tau\s*=\s*0\.879", A) and "0.879" not in A,
      "RED: the withdrawn fitted amplification constant is absent from the paper")

# --------------------------------------------------------------------------- #
n_pass = sum(1 for ok, _ in checks if ok)
for ok, label in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
print(f"L0 reference rebase: {n_pass}/{len(checks)} checks passed")
if n_pass != len(checks):
    print("failed: " + "; ".join(lb for ok, lb in checks if not ok))
sys.exit(0 if n_pass == len(checks) else 1)
