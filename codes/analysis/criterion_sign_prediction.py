#!/usr/bin/env python3
"""criterion_sign_prediction.py  --  freeze the coverage-criterion sign rule.

Core-methodology (Level 1, iter #8) deliverable for the coupled NEGATIVE-CONTROL
design.  The paper claims that the recirculation COVERAGE fraction f_rec predicts the
SIGN of the coupled a-posteriori outcome of an ODE/equilibrium wall model across
repeating geometries:

    f_rec = L_sep / ell_p = (streamwise extent under separation) / (repeat pitch)

      f_rec > FAIL_THR  -> domain-wide cancellation  -> predict COUPLED FAILURE
      f_rec < PASS_THR  -> localised cancellation     -> predict COUPLED SUCCESS
      otherwise                                       -> MARGINAL

This script LOCKS that rule (thresholds fixed here, BEFORE any coupled conv-div run is
harvested) against the on-disk a-priori contrast, and records the predicted sign per
geometry.  The coupled-outcome column is 'pending' for conv-div -- filled at L3 from
the harvested coupled npz, NEVER fabricated here.

Reads : codes/results/repeating_structure_contrast.npz   (a-priori, real DNS;
        FLAT keys: <geom>_f_rec, <geom>_r2, <geom>_pitch_over_Lsep, <geom>_Lsep_over_delta)
Writes: codes/results/criterion_sign_prediction.npz

NOTE on hills R^2: the contrast file stores the *uncorrected* pehill_r2 (~ -4.7e6,
the y=0 wall-pinning extraction; see memory note). The manuscript reports the
hill-surface-aware corrected range R^2 in [-84,-18] (dose_response_xiao.npz). Either
way R^2 << 0 -> the a-priori sign is 'fail', which is all the consistency check uses.

All numbers trace to the input npz (killer gate G2). No fabrication (G1/G7).
Run:  OMP_NUM_THREADS=2 python3 codes/analysis/criterion_sign_prediction.py
"""
import os
import numpy as np

# --- pre-registered decision thresholds (FROZEN at L1, before coupled harvest) -----
PASS_THR = 0.10   # f_rec below this -> predict coupled SUCCESS
FAIL_THR = 0.20   # f_rec above this -> predict coupled FAILURE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SRC = os.path.join(ROOT, "codes", "results", "repeating_structure_contrast.npz")
OUT = os.path.join(ROOT, "codes", "results", "criterion_sign_prediction.npz")

# coupled outcome already harvested on disk (equilibrium-hills rung, iter #6,
# aposteriori_wmles_pehill.npz). Recorded here only to cross-check the prediction.
# conv-div is intentionally absent -> stays 'pending' until its L2 run is harvested.
COUPLED_HARVESTED = {
    "pehill": dict(coupled_xr=3.5804, dns_xr=4.5112, rel_err=-0.2063),
}


def predict_sign(f_rec):
    if f_rec > FAIL_THR:
        return "fail"
    if f_rec < PASS_THR:
        return "succeed"
    return "marginal"


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing input (run repeating_structure_contrast.py first): {SRC}")
    d = np.load(SRC, allow_pickle=True)
    # auto-discover geometries from flat keys "<geom>_f_rec"
    geoms = sorted({k[:-len("_f_rec")] for k in d.files if k.endswith("_f_rec")})
    if not geoms:
        raise SystemExit(f"no '<geom>_f_rec' keys in {SRC}; keys={list(d.files)}")

    rows = []
    print(f"{'geometry':10s} {'f_rec':>7s} {'lp/Lsep':>9s} {'Lsep/d':>7s} "
          f"{'apri R2':>12s} {'predict':>9s} {'coupled':>16s} {'consistent':>11s}")
    print("-" * 88)
    geoms_out, frec_out, pls_out, lsd_out, r2_out = [], [], [], [], []
    pred_out, coup_out, cons_out = [], [], []
    for g in geoms:
        f_rec = float(d[f"{g}_f_rec"])
        r2 = float(d[f"{g}_r2"])
        pls = float(d[f"{g}_pitch_over_Lsep"])
        lsd = float(d[f"{g}_Lsep_over_delta"])
        p = predict_sign(f_rec)
        apri_sign = "fail" if r2 < 0.5 else "succeed"   # a-priori R2 sign
        if g in COUPLED_HARVESTED:
            h = COUPLED_HARVESTED[g]
            cs = "fail" if abs(h["rel_err"]) > 0.05 else "succeed"  # >5% x_r err = fail
            coupled_str = f"{cs}({h['rel_err']*100:+.1f}%)"
        else:
            cs = "pending"
            coupled_str = "pending"
        cons = (p == apri_sign)
        # compact R2 print (handle huge artifact magnitude)
        r2s = f"{r2:.3f}" if abs(r2) < 1e4 else f"{r2:.2e}"
        print(f"{g:10s} {f_rec:7.3f} {pls:9.2f} {lsd:7.3f} {r2s:>12s} "
              f"{p:>9s} {coupled_str:>16s} {str(cons):>11s}")
        geoms_out.append(g); frec_out.append(f_rec); pls_out.append(pls)
        lsd_out.append(lsd); r2_out.append(r2)
        pred_out.append(p); coup_out.append(cs); cons_out.append(cons)

    print("-" * 88)
    print(f"thresholds (frozen): predict-fail if f_rec>{FAIL_THR}, "
          f"predict-succeed if f_rec<{PASS_THR}")
    n_consistent = int(np.sum(cons_out))
    print(f"a-priori sign consistency: {n_consistent}/{len(geoms)} "
          f"(predicted sign matches a-priori R2 sign)")
    print("coupled negative-control: pehill harvested (fail, -20.6%); "
          "convdiv PENDING (L2) -- predicted SUCCEED, not yet measured.")

    np.savez(
        OUT,
        geometries=np.array(geoms_out),
        f_rec=np.array(frec_out),
        pitch_over_Lsep=np.array(pls_out),
        L_sep_over_delta=np.array(lsd_out),
        apriori_R2=np.array(r2_out),
        predicted_sign=np.array(pred_out),
        coupled_sign=np.array(coup_out),
        apriori_consistent=np.array(cons_out),
        PASS_THR=PASS_THR,
        FAIL_THR=FAIL_THR,
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
