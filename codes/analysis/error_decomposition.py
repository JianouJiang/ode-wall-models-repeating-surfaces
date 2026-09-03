#!/usr/bin/env python3
r"""
error_decomposition.py
======================
Thrust #12 (L1/L2) -- quantify the two-term decomposition of the ODE
wall-stress error already stated in the manuscript (Eq. eq:error_exact),

    Delta tau_w  =  E_closure  +  E_struct ,
    E_closure    =  tau_hat(y_m) - tau(y_m)            (closure error)
    E_struct     =  C  =  int_0^{y_m} (U dU/dx + V dU/dy) dy   (structural / dropped convection)

and the COMPENSATION OBSERVATION that, in the cancellation regime, the
empirical mixing-length closure error E_closure is anti-correlated with the
structural error E_struct: the closure that ignores convection produces a
near-wall stress bias that absorbs the missing convective transport, so the two
errors partially cancel in the wall-stress prediction. Supplying the *exact*
DNS stress sets E_closure -> 0, removes the compensation, and exposes the full
structural error -- which is why exact stresses make the prediction WORSE.

This module is L1 scope: it formalises + quantifies the decomposition on the
canonical periodic hill (case_1p0, Re_b = 5600), reusing the PRODUCTION ODE /
shooting solver and the hill-surface-aware extraction UNCHANGED (imported from
diagnostic_test_corrected). It emits `results/error_decomposition.npz` so every
number that enters the manuscript's revised Sec. 3.7 traces to disk (G2).

Operational identification of the two terms (a-priori, real DNS):
  The exact-DNS-stress ODE has the exact closure, so E_closure = 0 and its total
  wall-stress error IS the structural error:
        E_struct(x)  =  tau_w^{DNS-stress ODE}(x) - tau_w^{true}(x)
  The mixing-length ODE carries both terms, so the closure error is the
  difference of the two ODE predictions (the +/- tau_w^true and the shared y_m,
  dp/dx cancel identically):
        E_closure(x) =  tau_w^{ML ODE}(x) - tau_w^{DNS-stress ODE}(x)
  C(x) is computed INDEPENDENTLY from the 2-D momentum budget
  (momentum_budget_pehill.npz) by trapezoidal integration of the convective term
  from the wall to the matching height y_m -- it is NOT derived from the ODE, so
  corr(E_struct, C) is a genuine cross-check that the exact-stress ODE error is
  the dropped convection (prediction P2), not a tautology.

P1 (compensation, hills): corr(E_closure, E_struct) < 0 and the exact-stress
   error exceeds the mixing-length error (mean|E_dns| > mean|E_ml|).
P2 (structural error = convective integral): corr(E_struct, C) > 0.

Cross-geometry control (P3: the compensation is regime-specific, vanishing on
eps ~ O(1) success geometries) and the convection-driven-closure test
(P4: corr(E_closure, local convection)) are the L2 extension; this L1 module
emits the hooks (writes nan placeholders + a TODO flag) so L2 only has to fill
the success-geometry rows.

A-priori only. No new CFD. <= 2 cores. No fabrication: every output is computed
from real DNS on disk.

Run:  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 codes/analysis/error_decomposition.py
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # codes/
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
import diagnostic_test_corrected as D        # noqa: E402  (production solver, UNCHANGED)

Y_IDX = D.Y_IDX                              # = 10, the manuscript matching index


def convective_integral_to_ym(budget_npz, y_idx=Y_IDX):
    """C(x) = int_0^{y_m} (U dU/dx + V dU/dy) dy from the 2-D momentum budget.

    Integrated by the trapezoidal rule from the wall (wall_j) up to the
    (y_idx)-th fluid point above the wall -- the same matching index the ODE
    uses.  Returns an array aligned 1:1 by station index with the diagnostic
    extraction (both are sorted on the unique DNS x-stations of case_1p0)."""
    b = np.load(budget_npz, allow_pickle=True)
    conv, yu, wall_j, fluid = b["convection"], b["y_unique"], b["wall_j"], b["fluid"]
    n = conv.shape[0]
    C = np.full(n, np.nan)
    for i in range(n):
        j0 = int(wall_j[i])
        col = np.where(fluid[i])[0]
        col = col[col >= j0]
        if len(col) < y_idx + 2:
            continue
        sel = col[: y_idx + 1]
        y = yu[sel] - yu[j0]                  # distance from wall
        C[i] = np.trapezoid(conv[i, sel], y)
    return C, b["x_unique"]


def main():
    diag = os.path.join(RESULTS, "diagnostic_test_corrected.npz")
    budget = os.path.join(RESULTS, "momentum_budget_pehill.npz")
    d = np.load(diag, allow_pickle=True)

    tau = d["tau_w_dns"]                       # true wall stress
    ml = d["standard_ml"]                      # mixing-length ODE prediction
    dns = d["controlled_dns"]                  # exact-DNS-stress ODE prediction
    sep = d["is_separated"]
    ok = sep & d["standard_ml_valid"] & d["controlled_dns_valid"] & d["valid_mask"]

    # --- the two-term decomposition, station by station -------------------
    E_total_ml = ml - tau                      # total error of the standard ODE
    E_struct = dns - tau                       # = E_struct (closure exact -> E_closure=0)
    E_closure = ml - dns                       # = E_total_ml - E_struct

    # --- independent convective integral (NOT from the ODE) ---------------
    C, x_budget = convective_integral_to_ym(budget)
    assert C.shape[0] == tau.shape[0], "budget / diagnostic station mismatch"

    okC = ok & np.isfinite(C)

    # --- P1: compensation on the hills cancellation stations --------------
    mean_abs_ml = float(np.abs(E_total_ml[ok]).mean())
    mean_abs_dns = float(np.abs(E_struct[ok]).mean())
    comp_ratio = mean_abs_dns / mean_abs_ml
    frac_ml_beats = float((np.abs(E_total_ml[ok]) < np.abs(E_struct[ok])).mean())
    r_comp, p_comp = pearsonr(E_closure[ok], E_struct[ok])

    # --- P2: structural error tracks the directly integrated convection ---
    r_struct_C, p_struct_C = pearsonr(E_struct[okC], C[okC])

    # --- aggregate R^2 paradox (already in manuscript, re-confirmed) -------
    r2_ml = float(d["standard_ml_r2"])
    r2_dns = float(d["controlled_dns_r2"])

    print("=" * 64)
    print("Error decomposition on the canonical periodic hill (case_1p0)")
    print(f"  separated stations used (P1): {int(ok.sum())}")
    print(f"  stations with finite C (P2):  {int(okC.sum())}")
    print("-" * 64)
    print(f"  P1  mean|E_total,ML|   = {mean_abs_ml:.4e}")
    print(f"  P1  mean|E_struct,DNS| = {mean_abs_dns:.4e}  ({comp_ratio:.2f}x worse)")
    print(f"  P1  frac ML beats DNS  = {frac_ml_beats:.3f}")
    print(f"  P1  corr(E_closure, E_struct) = {r_comp:+.3f}  (p={p_comp:.1e})")
    print(f"  P2  corr(E_struct,   C)       = {r_struct_C:+.3f}  (p={p_struct_C:.1e})")
    print(f"  R2(tau_w):  ML = {r2_ml:.2f}   DNS-stress = {r2_dns:.2f}")
    print("=" * 64)

    out = os.path.join(RESULTS, "error_decomposition.npz")
    np.savez(
        out,
        # per-station fields (512; use the masks below)
        tau_w_dns=tau,
        E_total_ml=E_total_ml,
        E_closure=E_closure,
        E_struct=E_struct,
        C=C,
        sep_mask=ok,
        sep_mask_with_C=okC,
        Y_IDX=Y_IDX,
        # P1 scalars
        n_sep=int(ok.sum()),
        mean_abs_E_ml=mean_abs_ml,
        mean_abs_E_dns=mean_abs_dns,
        compensation_ratio=comp_ratio,
        frac_ml_beats_dns=frac_ml_beats,
        corr_closure_struct=float(r_comp),
        p_closure_struct=float(p_comp),
        # P2 scalars
        n_sep_with_C=int(okC.sum()),
        corr_struct_C=float(r_struct_C),
        p_struct_C=float(p_struct_C),
        # aggregate R^2 paradox
        r2_ml=r2_ml,
        r2_dns=r2_dns,
        # L2 hooks (cross-geometry control P3 + convection-driven closure P4):
        # filled by the L2 extension; nan here flags "not yet computed".
        p3_success_compensation_ratio=np.nan,   # ratio on BFS/NASA-hump (expect ~1)
        p4_corr_closure_convection=np.nan,       # corr(E_closure, local convection)
        l2_todo=np.array(["P3_cross_geometry_control", "P4_convection_driven_closure"]),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
