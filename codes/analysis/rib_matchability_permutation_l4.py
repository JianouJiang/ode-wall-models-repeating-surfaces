#!/usr/bin/env python3
"""
L4 writing-pass support: exact-permutation significance of the matchability
correlation Spearman(relRMS, R^2) over the six-pitch RANS sweep.

Discharges B-L4-1: the L3 manuscript text called rho = -0.60 "the operative
correlation"; the L4 Judge bind requires the exact-permutation p-value at n=6 be
reported alongside, so a referee can judge the strength themselves.

This script ADDS NO new simulation. It loads the already-validated arrays from
codes/results/rib_fidelity_consistency_l3.npz (md5 c1d16f4b...) and computes the
exact two-sided / one-sided permutation p-value over all 6! = 720 orderings.
Every number it emits therefore traces to the L3 results file (G2).
"""
import os
import numpy as np
from itertools import permutations
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
L3 = os.path.join(RESULTS, "rib_fidelity_consistency_l3.npz")


def exact_permutation_p(x, y):
    """Exact two-sided and one-sided (<= obs) Spearman permutation p over n!."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs = spearmanr(x, y).statistic
    n = len(y)
    ge = le = tot = 0
    for perm in permutations(range(n)):
        r = spearmanr(x, y[list(perm)]).statistic
        tot += 1
        if abs(r) >= abs(obs) - 1e-12:
            ge += 1
        if r <= obs + 1e-12:
            le += 1
    return obs, ge / tot, le / tot, ge, le, tot


def main():
    d = np.load(L3, allow_pickle=True)
    relRMS = np.asarray(d["sweep_relRMS"], float)
    r2 = np.asarray(d["sweep_r2"], float)
    pk = np.asarray(d["sweep_pk"], float)

    rho, p2, p1, ge, le, tot = exact_permutation_p(relRMS, r2)
    rho_pk = spearmanr(pk, r2).statistic

    # qualitative leg: 5/6 matchable references certify; the sole non-matchable
    # (p/k=2, relRMS ~ 1.0) does not.
    matchable = relRMS < 0.7          # ODE-reproducible reference tau_w
    certifies = r2 >= 0.0
    n_match = int(matchable.sum())
    n_match_cert = int((matchable & certifies).sum())
    n_nonmatch = int((~matchable).sum())
    n_nonmatch_cert = int((~matchable & certifies).sum())

    print(f"n = {len(r2)} pitches")
    print(f"Spearman(relRMS, R^2) = {rho:+.4f}")
    print(f"  exact two-sided p = {p2:.4f} ({ge}/{tot})")
    print(f"  exact one-sided p = {p1:.4f} ({le}/{tot})")
    print(f"Spearman(p/k,   R^2) = {rho_pk:+.4f}")
    print(f"matchable refs certifying ODE: {n_match_cert}/{n_match}")
    print(f"non-matchable refs certifying: {n_nonmatch_cert}/{n_nonmatch} "
          f"(the sole non-matchable, p/k=2 relRMS={relRMS[np.argmax(relRMS)]:.3f})")

    out = os.path.join(RESULTS, "rib_matchability_permutation_l4.npz")
    np.savez(
        out,
        sweep_pk=pk, sweep_r2=r2, sweep_relRMS=relRMS,
        rho_relRMS_r2=rho, p_two_sided=p2, p_one_sided=p1,
        n_perm_ge=ge, n_perm_le=le, n_perm_tot=tot,
        rho_pk_r2=rho_pk,
        n_matchable=n_match, n_matchable_certifies=n_match_cert,
        n_nonmatchable=n_nonmatch, n_nonmatchable_certifies=n_nonmatch_cert,
        source="rib_fidelity_consistency_l3.npz",
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
