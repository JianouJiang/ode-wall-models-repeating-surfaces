#!/usr/bin/env python3
"""Independent verifier for the input-sufficiency bracket (node_014, Level 0).

Checks, in order:
  A. the deposited summary exists, carries the registered protocol, and claims
     nothing coupled (killer gate G4);
  B. the certified Lipschitz floor is arithmetically a valid lower bound, on
     synthetic sets whose answer is known in closed form;
  C. the empirical transfer is recomputed independently for one held-out case
     and agrees with the deposited value;
  D. the deployed-model baseline is recomputed independently at the same
     stations and agrees with the deposited value;
  E. red fixtures that MUST fail: label shuffling, input duplication, and a
     leakage protocol that does not actually hold the test case out;
  F. anti-regression on the manuscript: the operator-mandated pair of
     matching-height transfer relations is printed and the withdrawn
     "artefact / superseded" reading has not returned.

Exit status 0 only if every check passes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "codes" / "analysis"))

SUMMARY = ROOT / "codes/results/input_sufficiency_bracket_summary.json"
NPZ = ROOT / "codes/results/input_sufficiency_bracket.npz"
MAIN_TEX = ROOT / "manuscript/main.tex"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(f"{name}{(': ' + detail) if detail else ''}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    import input_sufficiency_bracket as isb

    # ---------------------------------------------------------------- A
    print("A. deposit and registered protocol")
    check("summary present", SUMMARY.exists(), str(SUMMARY.name))
    check("npz present", NPZ.exists(), str(NPZ.name))
    if not SUMMARY.exists():
        return 1
    S = json.loads(SUMMARY.read_text())
    check("schema", S.get("schema") == "input_sufficiency_bracket_v1",
          str(S.get("schema")))
    reg = S["registered"]
    check("registered k matches producer", reg["k_neighbours"] == isb.K_NEIGHBOURS)
    check("registered heights match producer",
          tuple(reg["eta_fractions"]) == tuple(isb.ETA_FRACTIONS))
    check("registered nu matches the wavy case file",
          abs(reg["nu"]["wavy"] - 1.0 / 3460.0) < 1e-12)
    blob = json.dumps(S).lower()
    check("G4: no coupled claim in the deposit",
          "coupled" not in blob, "a-priori only")

    # ---------------------------------------------------------------- B
    print("B. the certified floor is a valid lower bound")
    rng = np.random.default_rng(7)
    # B1 exact smooth function: floor must vanish at the true Lipschitz constant
    d = rng.uniform(-2, 2, size=(200, 2))
    t = 0.7 * d[:, 0]
    f0, _ = isb.certified_floor(d, t, L=0.7)
    check("B1 exact linear map -> zero floor at its own L", f0 <= 1e-9, f"{f0:.2e}")
    # B2 duplicated inputs with different outputs: floor must be positive for all L
    d2 = np.repeat(rng.uniform(-1, 1, size=(60, 2)), 2, axis=0)
    t2 = np.tile([1.0, -1.0], 60)
    vals = [isb.certified_floor(d2, t2, L=L)[0] for L in (0.0, 1.0, 10.0, 1e3)]
    check("B2 duplicated inputs -> positive floor at every L",
          all(v > 0.5 for v in vals), " ".join(f"{v:.3f}" for v in vals))
    # B3 monotone in L
    d3 = rng.uniform(-2, 2, size=(150, 2))
    t3 = np.sin(d3[:, 0]) + 0.3 * rng.normal(size=150)
    seq = [isb.certified_floor(d3, t3, L=L)[0] for L in (0.0, 0.5, 1.0, 2.0)]
    check("B3 floor is non-increasing in L",
          all(seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1)),
          " ".join(f"{v:.3f}" for v in seq))

    # ---------------------------------------------------------------- C/D
    print("C/D. independent recomputation at one height")
    frac = isb.ETA_FRACTIONS[0]
    tag = f"eta{frac:.2f}"
    hills, wavy, convdiv = isb.build_cases(frac)
    canon = next((c for c in hills if c["name"] == "alph10-6-3036"), hills[0])
    rec = S["heights"][tag]
    check("case count reproduced", len(hills) == rec["n_hill_cases"],
          f"{len(hills)} vs {rec['n_hill_cases']}")
    check("station count reproduced",
          sum(c["n_station"] for c in hills) == rec["n_hill_stations"],
          f"{sum(c['n_station'] for c in hills)}")

    train = [c for c in hills if c["group"] != canon["group"]]
    pred, _, _ = isb.knn_transfer(train, canon, use_b=True)
    r2_emp = isb.r2_score(pred, canon["tau"])
    check("C empirical transfer reproduced",
          abs(r2_emp - rec["canonical"]["r2_empirical"]) < 1e-9,
          f"{r2_emp:.6f} vs {rec['canonical']['r2_empirical']:.6f}")

    # independent Spalding implementation (not the module's) at the same stations
    def spalding_independent(u_m, y_m, nu, kappa=0.41, B=5.0):
        out = np.empty(len(u_m))
        for i, u in enumerate(u_m):
            if u == 0.0:
                out[i] = 0.0
                continue
            speed = abs(float(u))

            def res(ut):
                yp = y_m[i] * ut / nu
                up = speed / ut
                k = kappa * up
                if k > 700.0:
                    return -np.inf
                return yp - (up + np.exp(-kappa * B) * (
                    np.exp(k) - 1.0 - k - k**2 / 2.0 - k**3 / 6.0))
            lo, hi = 1e-12, max(speed * 10.0, nu / y_m[i])
            while res(hi) <= 0:
                hi *= 10.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if res(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            ut = 0.5 * (lo + hi)
            out[i] = np.sign(u) * ut * ut
        return out

    m0_ind = spalding_independent(canon["u_m"], canon["y_m"], canon["nu"])
    r2_m0 = isb.r2_score(m0_ind, canon["tau"])
    dep = rec["canonical"]["r2_equilibrium"]
    check("D equilibrium baseline reproduced by an independent solver",
          abs(r2_m0 - dep) <= 1e-3 * max(1.0, abs(dep)),
          f"{r2_m0:.4f} vs {dep:.4f}")
    # NOTE (node_014): an earlier form of this check asserted that the empirical
    # function beats *both* deployed models by a wide margin.  The measurement
    # refuted that: the equilibrium law scores +0.86 on this hill.  The check is
    # rewritten to test the claim the node actually makes, which is about the
    # pressure-gradient arm and about the information ordering.
    r2_emp_dep = rec["canonical"]["r2_empirical"]
    r2_tble_dep = rec["canonical"]["r2_tble"]
    check("D1 same-input function beats the pressure-gradient ODE by a wide margin",
          r2_emp_dep - r2_tble_dep > 1.0,
          f"empirical {r2_emp_dep:.3f} vs TBLE {r2_tble_dep:.3f}")
    check("D2 the model reading FEWER inputs beats the one reading more "
          "(the information ordering is violated)",
          dep > r2_tble_dep + 1.0,
          f"equilibrium(a) {dep:.3f} > TBLE(a,b) {r2_tble_dep:.3f}")
    fam = rec["deployed"]
    hills_only = [v for k, v in fam.items() if k.startswith("alph")]
    eq_med = float(np.median([h["r2_equilibrium"] for h in hills_only]))
    tb_med = float(np.median([h["r2_tble"] for h in hills_only]))
    check("D3 the information ordering is violated across the whole family",
          eq_med > tb_med + 1.0 and len(hills_only) == 29,
          f"median over {len(hills_only)} hills: equilibrium {eq_med:.3f} "
          f"vs TBLE {tb_med:.3f}")
    check("D4 no deployed-model solver failures anywhere in the family",
          all(h["n_model_failures"] == 0 for h in fam.values()),
          f"{sum(h['n_model_failures'] for h in fam.values())} failures")

    # ---------------------------------------------------------------- E
    print("E. red fixtures (each MUST fail to reproduce the claim)")
    shuffle_rng = np.random.default_rng(11)
    pred_s, _, _ = isb.knn_transfer(train, canon, use_b=True, shuffle_rng=shuffle_rng)
    r2_s = isb.r2_score(pred_s, canon["tau"])
    check("E1 label-shuffled training destroys the transfer",
          r2_s < 0.0 < r2_emp, f"shuffled {r2_s:.3f} vs honest {r2_emp:.3f}")
    dep_shuffled = S["heights"][tag]["transfer"]["a_and_b"]["label_shuffled_control"]
    check("E1b deposited shuffled control is also negative",
          dep_shuffled["median"] < 0.0, f"median {dep_shuffled['median']:.3f}")

    leak_pred, _, _ = isb.knn_transfer(train + [canon], canon, use_b=True)
    r2_leak = isb.r2_score(leak_pred, canon["tau"])
    check("E2 leakage protocol scores higher, so the hold-out is real",
          r2_leak > r2_emp, f"leaked {r2_leak:.4f} > held-out {r2_emp:.4f}")

    grp = rec["transfer"]["a_and_b"]["leave_one_group_out"]
    cas = rec["transfer"]["a_and_b"]["leave_one_case_out"]
    check("E3 the strict grouping is not looser than the case hold-out",
          grp["median"] <= cas["median"] + 1e-9,
          f"group {grp['median']:.3f} <= case {cas['median']:.3f}")

    xf = rec["transfer"]["a_and_b"]["cross_family"]
    check("E4 class locality is stated, not hidden",
          min(xf.values()) < grp["median"] - 0.3,
          f"cross-family {json.dumps(xf)} vs within {grp['median']:.3f}")

    # ---------------------------------------------------------------- H
    print("H. the probes, including the results that go against the thesis")
    wg_path = ROOT / "codes/results/input_sufficiency_wavy_grids.json"
    check("H0 wavy-grid probe deposited", wg_path.exists())
    if wg_path.exists():
        W = json.loads(wg_path.read_text())["heights"]["eta0.05"]
        emp = [W[g]["a_and_b"]["r2_hills_to_wavy"] for g in ("G0", "G1", "G2")]
        tbl = [W[g]["r2_tble"] for g in ("G0", "G1", "G2")]
        check("H1 the transfer failure on the wavy wall is grid-robust",
              all(v < 0 for v in emp) and max(emp) - min(emp) < 0.1,
              " ".join(f"{v:.3f}" for v in emp))
        check("H2 the reversal is recorded: the deployed ODE wins there",
              all(v > 0 for v in tbl) and min(tbl) > max(emp),
              " ".join(f"{v:.3f}" for v in tbl))

    pf_path = ROOT / "codes/results/input_sufficiency_pooled_floor.json"
    check("H3 corrected pooled-floor probe deposited", pf_path.exists())
    if pf_path.exists():
        P = json.loads(pf_path.read_text())["heights"]
        check("H4 P3b is recorded as NOT supported at both heights",
              all(not P[h]["P3b_supported"] for h in P),
              str({h: P[h]["P3b_supported"] for h in P}))
        check("H5 the P3b test was not vacuous: the classes overlap in input space",
              all(P[h]["overlap"]["wavy_in_hill_cloud"] > 0.2
                  and P[h]["overlap"]["convdiv_in_hill_cloud"] > 0.2 for h in P),
              str({h: round(P[h]["overlap"]["wavy_in_hill_cloud"], 3) for h in P}))

    rp_path = ROOT / "codes/results/input_sufficiency_reynolds_probe.json"
    check("H6 Reynolds probe deposited", rp_path.exists())
    if rp_path.exists():
        R = json.loads(rp_path.read_text())["heights"]["eta0.05"]["krank_pehill_Re5600"]
        check("H7 the adverse Reynolds row is present, not dropped",
              R["r2_tble"] > R["a_and_b"]["r2_empirical"],
              f"TBLE {R['r2_tble']:.3f} beats empirical "
              f"{R['a_and_b']['r2_empirical']:.3f} there")

    # ---------------------------------------------------------------- G
    print("G. internal consistency: the achieved error cannot beat the certificate")
    dcan = np.column_stack((isb.slog(canon["a"]), isb.slog(canon["b"])))
    dcan = (dcan - dcan.mean(0)) / (dcan.std(0) + 1e-30)
    scale = float(np.sqrt(np.mean(canon["tau"] ** 2)))
    l_hat = isb.measured_lipschitz(dcan, pred / scale)
    tn = canon["tau"] / scale
    idx = (np.arange(len(tn)) if len(tn) <= isb.FLOOR_SUBSAMPLE
           else np.linspace(0, len(tn) - 1, isb.FLOOR_SUBSAMPLE).astype(int))
    floor_at_lhat, _ = isb.certified_floor(dcan[idx], tn[idx], L=l_hat)
    achieved = float(np.sqrt(np.mean((pred - canon["tau"]) ** 2)) / scale)
    check("G1 achieved error is above its own certified floor",
          achieved >= floor_at_lhat - 1e-9,
          f"achieved {achieved:.4f} >= floor({l_hat:.3f}) = {floor_at_lhat:.4f}")

    # ---------------------------------------------------------------- F
    print("F. manuscript anti-regression (operator instruction 2026-08-24 20:10)")
    tex = MAIN_TEX.read_text()
    active = re.sub(r"\\iffalse.*?\\fi", "", tex, flags=re.S)
    window = active[active.find("single"):] if "single" in active else ""
    has_pair = (r"\rho=+1$ for the total-gradient TBLE and exactly $\rho=-1$"
                in active)
    check("F1 both transfer relations printed side by side", has_pair)
    check("F2 the withdrawn 'artefact ... superseded' reading has not returned",
          "artefact of the mismatched model" not in active)
    check("F3 the equilibrium anti-ranking is named as such",
          "anti-ranks the equilibrium model" in active)

    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
