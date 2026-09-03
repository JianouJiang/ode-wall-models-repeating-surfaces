#!/usr/bin/env python3
"""
feedback_loop_consistency.py  --  Thrust #16, Level-1 (methodology) check.

PURPOSE (L1, NOT L2): verify that the closed-loop derivation in
development/nodes/node_001/methodology.md is *internally consistent* with the
already-harvested, real on-disk anchors -- it makes NO new empirical claim and
fabricates NOTHING. It only:

  (1) reads coupled_norm_transfer.npz (real DNS + harvested coupled WMLES),
  (2) forms the open-loop vs closed-loop relative stress-error ratio (the
      "suppression factor" the methodology predicts to be ~ loop gain L_g ~ 1/eps),
  (3) compares it to 1/eps_median, confirming agreement to within an O(1) factor
      WITHOUT any fit,
  (4) confirms the healing is closure-independent (equilibrium vs TBLE C_f), as
      feedback (loop gain set by kappa, not the closure) predicts.

The quantitative measurement of the displacement identity (F2), the one-way F1
run, and the conv-div F3 budget are L2 work (codes/analysis/feedback_displacement.py).

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/feedback_loop_consistency.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
NPZ = os.path.join(RES, "coupled_norm_transfer.npz")


def main():
    d = np.load(NPZ, allow_pickle=True)

    # --- anchors (all real, traceable; see metrics.json / node_000) ----------
    d_open = float(d["apriori_relrms_tauw"])      # open-loop relative stress err
    dt_closed = float(d["coupled_eq_relrms_cf"])  # closed-loop relative C_f err
    eps_med = float(d["apriori_eps_median"])      # median cancellation depth
    r2_eq = float(d["coupled_eq_R2_cf"])          # coupled equilibrium-closure
    r2_tble = float(d["coupled_tble_R2_cf"])      # coupled TBLE-closure
    r2_open = float(d["apriori_R2_tauw"])         # open-loop a-priori catastrophe

    print("=" * 70)
    print("Thrust #16 L1 -- closed-loop methodology consistency check")
    print("source:", os.path.relpath(NPZ))
    print("=" * 70)

    # --- (1) suppression factor vs predicted loop gain -----------------------
    supp = d_open / dt_closed          # measured open->closed suppression
    Lg_pred = 1.0 / eps_med            # methodology: L_g ~ 1/eps (lambda ~ O(1))
    ratio = supp / Lg_pred             # should be O(1) if the partition holds
    print("\n[1] Loop-gain suppression of the wall-stress error")
    print(f"    open-loop relrms (a-priori)   d        = {d_open:8.3f}")
    print(f"    closed-loop relrms (coupled)  dtau/tau = {dt_closed:8.3f}")
    print(f"    MEASURED suppression  d/(dtau/tau)      = {supp:8.2f}")
    print(f"    PREDICTED loop gain   L_g ~ 1/eps        = {Lg_pred:8.2f}")
    print(f"    ratio measured/predicted (want O(1))     = {ratio:8.2f}")
    assert 0.3 < ratio < 3.0, "suppression not within an O(1) factor of 1/eps"
    print("    -> PASS: suppression ~ 1/eps to within an O(1) factor (no fit).")

    # --- (2) the suppression is real: open-loop is catastrophic --------------
    print("\n[2] The suppressed error was genuinely catastrophic open-loop")
    print(f"    R2(tau_w) a-priori (U_m frozen at truth) = {r2_open:8.2f}")
    print(f"    R2(C_f)   coupled  (U_m self-consistent) = {r2_eq:8.3f}")
    assert r2_open < -10 and r2_eq > 0.5
    print("    -> PASS: open-loop catastrophe (R2<<0) heals to R2>0 closed-loop.")

    # --- (3) closure-independence (feedback sets gain via kappa, not closure)-
    gap_pp = abs(r2_eq - r2_tble) * 100.0
    print("\n[3] Healing is closure-independent (feedback, not a better closure)")
    print(f"    R2(C_f) equilibrium closure              = {r2_eq:8.3f}")
    print(f"    R2(C_f) TBLE closure                     = {r2_tble:8.3f}")
    print(f"    gap                                      = {gap_pp:8.2f} pp")
    assert gap_pp < 5.0
    print("    -> PASS: <5pp gap; loop gain set by kappa~1/eps, not the closure.")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASS. Methodology is consistent with on-disk anchors.")
    print("F1 (one-way), F2 (displacement identity), F3 (conv-div) are L2 work.")
    print("=" * 70)


if __name__ == "__main__":
    main()
