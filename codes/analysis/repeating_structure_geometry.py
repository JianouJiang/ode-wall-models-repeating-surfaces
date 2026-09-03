"""
repeating_structure_geometry.py
================================
Compute the *a-priori geometric* quantities that underpin the
repeating-structure failure criterion (Sec. theory, "repeating structures"):

  - the streamwise repeat pitch         ell_p   (period of the geometry),
  - the per-element recirculation length L_sep   (streamwise extent of
    reversed wall shear within one period),
  - the recirculation coverage fraction f_rec = L_sep / ell_p,
  - the structure height / outer thickness scale used to non-dimensionalise.

The point of the repeating-structure framing is that two geometric facts,
both readable from the DNS fields (and, for an engineering design, from the
CAD geometry + a precursor estimate of delta), control whether the
cancellation mechanism is *domain-wide* (ODE fails) or *localised*
(ODE tolerates):

  (i)  the per-element recirculation collapses the streamwise length scale to
       L_sep ~ O(h) ~ O(delta), so convection and pressure gradient become the
       same order inside each recirculation (Pi ~ O(1));
  (ii) the structure repeats at a pitch ell_p that is *not large* compared to
       L_sep, so the recirculation (and hence eps << 1) covers an O(1)
       fraction of every period -- i.e. domain-wide, not localised.

A single isolated obstacle satisfies (i) but not (ii): the same recirculation
is followed by a long attached recovery, so f_rec -> 0 globally and the ODE
survives. Periodic hills are the canonical minimal instance that satisfies
both.

All numbers are computed from the real periodic-hills DNS fields in
codes/results/momentum_budget_pehill.npz (derived from Xiao et al. 2020,
Re_b = 5600). NO fabricated data. Output -> codes/results/repeating_structure_geometry.npz.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/repeating_structure_geometry.py
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # codes/
RESULTS = os.path.join(ROOT, "results")


def contiguous_separated_span(x, sep_mask):
    """Streamwise extent of the largest contiguous reversed-shear (separated)
    run, in the same units as x. This is the per-element recirculation length
    L_sep; using the largest contiguous run (rather than the raw count)
    measures the physical bubble extent, not scattered sign flips."""
    if not sep_mask.any():
        return 0.0, (np.nan, np.nan)
    best_len, best_span = 0.0, (np.nan, np.nan)
    i = 0
    n = len(sep_mask)
    while i < n:
        if sep_mask[i]:
            j = i
            while j + 1 < n and sep_mask[j + 1]:
                j += 1
            span = x[j] - x[i]
            if span > best_len:
                best_len, best_span = span, (x[i], x[j])
            i = j + 1
        else:
            i += 1
    return best_len, best_span


def main():
    src = os.path.join(RESULTS, "momentum_budget_pehill.npz")
    d = np.load(src, allow_pickle=True)
    x = d["x_unique"]            # streamwise coordinate, normalised by hill height h
    y = d["y_unique"]            # wall-normal coordinate, normalised by h
    sep = d["sep_mask"]          # bool[Nx]: reversed wall shear (tau_w < 0)
    tau = d["tau_w_dns"]

    # --- geometric scales (units of hill height h; the DNS uses h = 1) -------
    ell_p = float(x.max() - x.min())          # streamwise repeat pitch (period)
    H_channel = float(y.max() - y.min())      # channel height
    # Outer / shear-layer thickness scale. In the confined periodic-hills
    # geometry the separated shear layer and recovering boundary layers fill
    # the channel, so delta ~ O(half-height). We report the half-height as a
    # conservative O(delta) proxy; the qualitative ordering does not depend on
    # the exact prefactor.
    delta_proxy = 0.5 * H_channel

    # --- per-element recirculation length and coverage ----------------------
    f_rev = float(sep.mean())                              # reversed-shear fraction
    L_sep, span = contiguous_separated_span(x, sep)        # largest bubble extent
    f_rec = L_sep / ell_p                                  # coverage of one period

    # dimensionless ratios that define the criterion
    pitch_over_delta = ell_p / delta_proxy
    Lsep_over_delta = L_sep / delta_proxy
    pitch_over_Lsep = ell_p / L_sep if L_sep > 0 else np.inf

    out = dict(
        source=os.path.relpath(src, ROOT),
        geometry="periodic_hills_Xiao_Re5600",
        hill_height_h=1.0,
        repeat_pitch_ell_p=ell_p,
        channel_height_H=H_channel,
        delta_proxy_halfheight=delta_proxy,
        reversed_shear_fraction=f_rev,
        recirc_length_L_sep=L_sep,
        recirc_span_x=np.asarray(span, dtype=float),
        recirc_coverage_fraction_f_rec=f_rec,
        pitch_over_delta=pitch_over_delta,
        Lsep_over_delta=Lsep_over_delta,
        pitch_over_Lsep=pitch_over_Lsep,
    )
    np.savez(os.path.join(RESULTS, "repeating_structure_geometry.npz"), **out)

    print("Repeating-structure geometric metrics (periodic hills, Xiao Re_b=5600)")
    print("-" * 68)
    print(f"  repeat pitch        ell_p        = {ell_p:6.3f} h")
    print(f"  channel height      H            = {H_channel:6.3f} h")
    print(f"  outer scale (proxy) delta ~ H/2  = {delta_proxy:6.3f} h")
    print(f"  recirc length       L_sep        = {L_sep:6.3f} h  (span x in [{span[0]:.2f}, {span[1]:.2f}])")
    print(f"  reversed-shear frac f_rev        = {f_rev:6.3f}")
    print(f"  coverage of period  f_rec        = {f_rec:6.3f}")
    print("-" * 68)
    print(f"  L_sep / delta                    = {Lsep_over_delta:6.2f}   (O(1)  -> Pi ~ O(1) inside recirc)")
    print(f"  ell_p / L_sep                    = {pitch_over_Lsep:6.2f}   (O(1)  -> recirculation is domain-wide)")
    print(f"  ell_p / delta                    = {pitch_over_delta:6.2f}")
    print("-" * 68)
    print("Interpretation: both (i) L_sep ~ O(delta) and (ii) pitch ~ O(L_sep)")
    print("hold, so eps << 1 covers ~%.0f%% of every period -> domain-wide failure." % (100 * f_rec))
    print("Saved -> results/repeating_structure_geometry.npz")


if __name__ == "__main__":
    main()
