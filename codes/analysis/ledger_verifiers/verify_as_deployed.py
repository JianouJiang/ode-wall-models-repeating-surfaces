#!/usr/bin/env python3
"""verify_as_deployed.py -- independent audit of the as-deployed evaluation.

Nothing here imports the producer's own delivery map.  Theorem 1 of
`development/nodes/node_012/methodology.md` is re-implemented from its closed
form, and every stored face is re-derived from the stored *inputs*
(u_m, q, s, y_m, tau_request) and compared with the stored outputs.  Five red
fixtures corrupt the reconstruction in ways the method must notice; each is
required to FAIL, and the verifier fails if a fixture passes.

    python3 codes/analysis/ledger_verifiers/verify_as_deployed.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# Theorem 1, re-implemented from the closed form (NOT from the producer)
# ---------------------------------------------------------------------------
def delivered_closed_form(tau_r, u_m, q, y_m, nu):
    """tau_d = u_m * max( |tau_r|/q * 1[tau_r u_m > 0], nu/y_m )."""
    tau_r = np.asarray(tau_r, float)
    u_m = np.asarray(u_m, float)
    q = np.maximum(np.asarray(q, float), 1.0e-14)
    y_m = np.asarray(y_m, float)
    aligned = np.where(tau_r * u_m > 0.0, np.abs(tau_r) / q, 0.0)
    return u_m * np.maximum(aligned, nu / y_m)


def spalding_utau(speed, y, nu, kappa=0.41, e=9.8):
    """Converged Spalding root by bisection; independent of the producer."""
    b = math.log(e) / kappa
    speed = np.asarray(speed, float)
    y = np.asarray(y, float) * np.ones_like(speed)
    out = np.zeros_like(speed)
    for i in range(speed.size):
        sp = float(speed.ravel()[i])
        yi = float(y.ravel()[i])
        if sp <= 0.0:
            continue

        def resid(ut):
            up = sp / ut
            ku = min(kappa * up, 50.0)
            return yi * ut / nu - (up + math.exp(-kappa * b) *
                                   (math.exp(ku) - 1.0 - ku - 0.5 * ku * ku
                                    - ku ** 3 / 6.0))

        lo, hi = 1e-16 * max(sp, 1.0), max(sp, nu / yi) * 10.0
        n = 0
        while resid(hi) <= 0.0 and n < 300:
            hi *= 2.0
            n += 1
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        out.ravel()[i] = 0.5 * (lo + hi)
    return out


# ---------------------------------------------------------------------------
def latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(ROOT / "codes" / "results" / pattern)))
    hits = [h for h in hits if "pilot" not in h]
    return Path(hits[-1]) if hits else None


def main() -> int:
    npz_path = latest("as_deployed_evaluation_*[0-9].npz")
    json_path = latest("as_deployed_evaluation_*_summary.json")
    if npz_path is None or json_path is None:
        check("A0 producer output exists", False, "no as_deployed_evaluation_* found")
        return report()
    check("A0 producer output exists", True, npz_path.name)

    blobs = np.load(npz_path)
    summary = json.loads(json_path.read_text())
    records = summary["records"]
    check("A1 records present", len(records) > 0, f"{len(records)} records")

    # ---- coverage -------------------------------------------------------
    cases = {r["case"] for r in records}
    models = {r["model"] for r in records}
    patches = {r["patch"] for r in records}
    times = {r["time"] for r in records}
    check("A2 both deployed architectures are measured",
          models == {"total_gradient_tble", "equilibrium"}, str(sorted(models)))
    check("A3 both wall patches are kept separate",
          patches == {"bottomWall", "topWall"}, str(sorted(patches)))
    check("A4 the whole deposited matching-surface campaign is covered",
          len(cases) >= 13, f"{len(cases)} cases")
    check("A5 more than one averaging window is scored",
          len(times) >= 2, str(sorted(times)))

    # patch identity really is preserved: the two walls must not share y_m
    hill_ym = {round(r["ym_median"], 4) for r in records if r["patch"] == "bottomWall"}
    top_ym = {round(r["ym_median"], 4) for r in records if r["patch"] == "topWall"}
    check("A6 hill and top wall have distinct matching heights (no pooling)",
          not (hill_ym & top_ym),
          f"hill {sorted(hill_ym)} top {sorted(top_ym)}")

    # ---- Theorem 1, face by face, on every record -----------------------
    worst_closed = 0.0
    worst_bound = 0.0
    n_faces_checked = 0
    n_identity_faces = 0
    n_regime_identity = 0
    sign_lock_violations = 0
    for r in records:
        key = f"{r['case']}__{r['time']:.0f}__{r['patch']}__"
        try:
            u_m = blobs[key + "u_m"]
            q = blobs[key + "q"]
            s = blobs[key + "s"]
            y_m = blobs[key + "y_m"]
            tau_r = blobs[key + "tau_request"]
            tau_i = blobs[key + "tau_internal"]
            tau_d = blobs[key + "tau_deliver"]
            regime = blobs[key + "regime"]
        except KeyError:
            check(f"B0 face arrays present for {key}", False)
            continue
        nu = r["nu"]
        n_faces_checked += u_m.size
        if r["model"] == "total_gradient_tble":
            rebuilt = delivered_closed_form(tau_r, u_m, q, y_m, nu)
        else:
            u_tau = spalding_utau(s, y_m, nu)
            rebuilt = u_m * np.maximum(u_tau ** 2 / np.maximum(s, 1e-14), nu / y_m)
        scale = max(float(np.sqrt(np.mean(tau_d ** 2))), 1e-30)
        worst_closed = max(worst_closed,
                           float(np.max(np.abs(rebuilt - tau_d))) / scale)
        speed = q if r["model"] == "total_gradient_tble" else s
        # Corollary 1.2 concerns the stress the delivery map RECEIVES, which
        # for the equilibrium arm is Spalding evaluated at the full speed, not
        # the a-priori-scored request (proposition 2).
        bound = np.maximum(np.abs(tau_i) * np.abs(u_m) / np.maximum(speed, 1e-14),
                           nu * np.abs(u_m) / y_m)
        worst_bound = max(worst_bound,
                          float(np.max(np.abs(tau_d) - bound * (1 + 1e-9))) / scale)
        sign_lock_violations += int(np.count_nonzero(
            (np.sign(tau_d) != np.sign(u_m)) & (np.abs(u_m) > 1e-14)))
        n_identity_faces += int(round(r["identity_faithful_fraction"] * u_m.size))
        n_regime_identity += int(np.count_nonzero(regime == 0))

    check("B1 Theorem 1 closed form reproduces every delivered stress",
          worst_closed < 1e-10,
          f"worst relative deviation {worst_closed:.3e} over "
          f"{n_faces_checked} faces")
    check("B2 Corollary 1.2 contraction bound is never violated",
          worst_bound <= 0.0, f"worst excess {worst_bound:.3e}")
    check("B3 Corollary 1.1 sign lock holds on every face",
          sign_lock_violations == 0, f"{sign_lock_violations} violations")
    check("B4 Corollary 1.3: the identity set of the delivery map is empty "
          "under the comparison the boundary condition itself makes",
          n_regime_identity == 0,
          f"{n_regime_identity} of {n_faces_checked} faces")
    check("B4b Corollary 1.3: it stays empty to within a 1e-12 relative "
          "alignment tolerance (measured, not asserted)",
          n_identity_faces / max(n_faces_checked, 1) < 1e-5,
          f"{n_identity_faces} of {n_faces_checked} faces "
          f"({n_identity_faces/max(n_faces_checked,1):.2e})")

    # ---- the falsification residual -------------------------------------
    hill = [r for r in records if r["patch"] == "bottomWall"]
    resid = [r["reconstruction_residual_rms_over_measured"] for r in hill] or [9e9]
    check("C1 reconstructed delivered traction tracks the solver's own "
          "wallShearStressMean",
          max(resid) < 0.5,
          f"max {max(resid):.4f}, median {float(np.median(resid)):.4f} "
          f"of the measured RMS over {len(resid)} hill records")
    def shares_of(model):
        return [r["bridge"]["delivery_share_of_total"] for r in hill
                if "bridge" in r and r["model"] == model]

    st, se = shares_of("total_gradient_tble"), shares_of("equilibrium")
    check("C2 for the TBLE arm the delivery deficiency dominates the "
          "request-to-measured gap",
          bool(st) and min(st) > 0.5,
          f"range [{min(st):.3f}, {max(st):.3f}] over {len(st)} records")
    check("C2b for the equilibrium arm it does NOT dominate, and the paper "
          "must not claim it does",
          bool(se) and max(se) < 0.5,
          f"range [{min(se):.3f}, {max(se):.3f}] over {len(se)} records")

    # the coupled curve reduced here must be the paper's own coupled curve
    ladder_path = ROOT / "codes" / "results" / "as_deployed_bridge_ladder.json"
    if ladder_path.exists():
        ladder = json.loads(ladder_path.read_text())["rows"]
        dev = [row["pinned_reducer_max_deviation_over_rms"] for row in ladder]
        finite = [d for d in dev if d == d]
        check("C3 the coupled curve is bit-identical to the one the pinned "
              "reducer already publishes",
              bool(finite) and max(finite) < 1e-12,
              f"max deviation {max(finite):.3e} over {len(finite)} ladder rows")
        eq_rows = [row for row in ladder if row["model"] == "equilibrium"]
        eq_i = [row["gaps_over_measured_rms"]["I_input_transfer"]
                for row in eq_rows]
        eq_d = [row["gaps_over_measured_rms"]["D_delivery_deficiency"]
                for row in eq_rows]
        # The ordering I > D is the claim the paper makes for this arm. It holds
        # record by record. The stronger "order of magnitude at every station"
        # form was written against a partial harvest and is FALSE on the full
        # campaign: the margin collapses with matching height, so it is checked
        # as a trend instead of as a fixed factor.
        n_ordered = sum(1 for i, d in zip(eq_i, eq_d) if i > d)
        check("C4 for the equilibrium arm the delivery term is below the "
              "input-transfer term in every record",
              bool(eq_i) and n_ordered == len(eq_i),
              f"{n_ordered}/{len(eq_i)} records ordered; "
              f"I in [{min(eq_i):.3f},{max(eq_i):.3f}] vs "
              f"D in [{min(eq_d):.4f},{max(eq_d):.4f}]")

        by_height: dict[float, list[float]] = {}
        for row in eq_rows:
            g = row["gaps_over_measured_rms"]
            by_height.setdefault(round(row["ym_median"], 4), []).append(
                g["I_input_transfer"] / g["D_delivery_deficiency"])
        heights = sorted(by_height)
        ratios = [sum(v) / len(v) for _, v in sorted(by_height.items())]
        shallow, deep = ratios[0], ratios[-1]
        check("C4b the equilibrium margin is a matching-height trend, not a "
              "fixed separation of scales: I/D falls monotonically and drops "
              "below 10 at the deepest surface",
              len(ratios) >= 4
              and all(a > b for a, b in zip(ratios, ratios[1:]))
              and shallow > 100 > 10 > deep,
              "I/D " + " -> ".join(f"{h:.4f}:{r:.0f}"
                                   for h, r in zip(heights, ratios)))

        # red fixture: the ordering check must be able to fail
        fake_i, fake_d = list(eq_i), list(eq_d)
        fake_i[0], fake_d[0] = fake_d[0], fake_i[0]
        check("R6 red: a record in which delivery exceeds input transfer is "
              "detected by C4",
              sum(1 for i, d in zip(fake_i, fake_d) if i > d) != len(fake_i),
              f"swapped record would give {sum(1 for i, d in zip(fake_i, fake_d) if i > d)}"
              f"/{len(fake_i)}")
    else:
        check("C3 bridge ladder present", False, "as_deployed_bridge_ladder.json")

    # proposition 2 as a measurement: the deployed equilibrium arm delivers
    # MORE than its own a-priori-scored request, because it uses |U_c| >= |U_m|
    eq_amp = [r["exceeds_bound_of_scored_request_fraction"]
              for r in records if r["model"] == "equilibrium"]
    tb_amp = [r["exceeds_bound_of_scored_request_fraction"]
              for r in records if r["model"] == "total_gradient_tble"]
    check("C5 Proposition 2 measured: the equilibrium delivery exceeds the "
          "bound of its a-priori-scored request, the TBLE delivery never does",
          bool(eq_amp) and min(eq_amp) > 0.5 and max(tb_amp) == 0.0,
          f"equilibrium [{min(eq_amp):.3f},{max(eq_amp):.3f}] "
          f"TBLE max {max(tb_amp):.3e}")

    # ---- the equilibrium arm is measured, not asserted -------------------
    eq = [r for r in records if r["model"] == "equilibrium"]
    check("D1 the equilibrium arm has its own per-face census",
          len(eq) >= 6 and all(r["regime_fraction"] is not None for r in eq),
          f"{len(eq)} equilibrium records")
    check("D2 Proposition 2: the equilibrium arm never refuses a sign",
          bool(eq) and max(r["regime_fraction"]["sign_refused"]
                           for r in eq) < 1e-12,
          f"max {max((r['regime_fraction']['sign_refused'] for r in eq), default=float('nan')):.3e}")
    tble = [r for r in records if r["model"] == "total_gradient_tble"
            and r["patch"] == "bottomWall"]
    check("D3 the TBLE arm does refuse signs on the hill (architectures differ "
          "by measurement, not by assumption)",
          bool(tble) and min(r["regime_fraction"]["sign_refused"]
                             for r in tble) > 0.05,
          "sign-refused fraction range ["
          f"{min((r['regime_fraction']['sign_refused'] for r in tble), default=float('nan')):.3f}, "
          f"{max((r['regime_fraction']['sign_refused'] for r in tble), default=float('nan')):.3f}]")

    # ---- provenance ------------------------------------------------------
    prov = summary["provenance"]
    headers = set(prov["tble_header_sha256"].values())
    check("E1 every TBLE case is bound to a hashed kernel header",
          len(prov["tble_header_sha256"]) >= 6 and len(headers) == 1,
          f"{len(prov['tble_header_sha256'])} cases, {len(headers)} distinct hash")
    check("E2 the stock Spalding source is vendored and hashed",
          len(prov.get("spalding_source_sha256", "")) == 64)
    check("E3 the pinned L2/L3 reducers are the ones the paper uses",
          len(prov["l2_reducer_sha256"]) == 64
          and len(prov["l3_analyser_sha256"]) == 64)

    # ---- RED FIXTURES: each must be detected ----------------------------
    r0 = next(r for r in records
              if r["model"] == "total_gradient_tble" and r["patch"] == "bottomWall")
    key = f"{r0['case']}__{r0['time']:.0f}__{r0['patch']}__"
    u_m, q, y_m = blobs[key + "u_m"], blobs[key + "q"], blobs[key + "y_m"]
    tau_r, tau_d = blobs[key + "tau_request"], blobs[key + "tau_deliver"]
    nu = r0["nu"]
    scale = float(np.sqrt(np.mean(tau_d ** 2)))

    bad = delivered_closed_form(tau_r, u_m, np.abs(u_m), y_m, nu)
    check("R1 red: using |u_m| in place of the tangential speed q is detected",
          float(np.max(np.abs(bad - tau_d))) / scale > 1e-3,
          f"deviation {float(np.max(np.abs(bad - tau_d)))/scale:.3e}")

    bad = np.abs(tau_r) * 0 + delivered_closed_form(-tau_r, u_m, q, y_m, nu)
    check("R2 red: flipping the sign of the request changes the delivered stress",
          float(np.max(np.abs(bad - tau_d))) / scale > 1e-3,
          f"deviation {float(np.max(np.abs(bad - tau_d)))/scale:.3e}")

    bad = u_m * np.abs(tau_r) / np.maximum(q, 1e-14)      # no sign gate
    check("R3 red: dropping the sign gate (Corollary 1.1) is detected",
          float(np.max(np.abs(bad - tau_d))) / scale > 1e-3,
          f"deviation {float(np.max(np.abs(bad - tau_d)))/scale:.3e}")

    bad = np.where(tau_r * u_m > 0.0, u_m * np.abs(tau_r) / np.maximum(q, 1e-14),
                   0.0)                                   # no laminar floor
    check("R4 red: dropping the laminar Couette floor is detected",
          float(np.max(np.abs(bad - tau_d))) / scale > 1e-3,
          f"deviation {float(np.max(np.abs(bad - tau_d)))/scale:.3e}")

    # the identity claim must actually bite: force q == |u_m| and the
    # faithful set must become non-empty
    forced = np.count_nonzero(
        (tau_r * u_m > 0.0) & (np.abs(tau_r) * y_m >= nu * np.abs(u_m)))
    check("R5 red: with the alignment condition forced, the faithful set is "
          "non-empty (so B4 is a real constraint, not a vacuous one)",
          forced > 0, f"{forced} of {u_m.size} faces would be faithful")

    # ---- manufactured boundary cases for the two corrected corollaries ----
    # R6: the envelope is NOT a contraction.  A weak, sign-consistent request
    # below the molecular floor is delivered AT the floor, above the request.
    d_amp = float(delivered_closed_form(0.1, 2.0, 3.0, 1.0, 1.0))
    check("R6 red: a weak same-sign request is delivered above itself, so "
          "'never amplifies' is false and must not be claimed",
          abs(d_amp) > 0.1 * (1 + 1e-12),
          f"tau_r = 0.1 delivered as {d_amp:.6g}")

    # R7: the identity set has a second branch.  On the molecular floor the
    # delivered stress equals the request even though q != |u_m|.
    d_id = float(delivered_closed_form(2.0, 2.0, 3.0, 1.0, 1.0))
    on_floor = bool(np.isclose(abs(2.0), 1.0 * abs(2.0) / 1.0, rtol=1e-12))
    check("R7 red: the molecular-floor branch of the identity set is "
          "recognised (tau_d == tau_r with q != |u_m|)",
          abs(d_id - 2.0) < 1e-12 and on_floor,
          f"tau_d = {d_id:.6g} with q = 3 != |u_m| = 2, on the floor")

    # R8: and the first branch still bites on its own terms.
    d_proj = float(delivered_closed_form(5.0, 2.0, 2.0, 1.0, 1.0))
    check("R8 red: an aligned, above-floor request is delivered unchanged",
          abs(d_proj - 5.0) < 1e-12,
          f"tau_d = {d_proj:.6g}")

    return report()


def report() -> int:
    failed = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
