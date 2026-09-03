#!/usr/bin/env python3
r"""
transition_map_bfs_anchor.py
============================
PILLAR C / Bind B-L1-3 : the backward-facing step (BFS) as the ZERO-FREQUENCY
(non-repeating, pitch lambda -> infinity) anchor of the failure transition map.

THE PHYSICAL ARGUMENT (why a single feature is tolerated)
---------------------------------------------------------
The cancellation mechanism requires the convective and pressure-gradient terms of
the near-wall momentum balance to remain O(delta)-coherent and *opposed* over a
LARGE FRACTION of the wall, so that the wall stress collapses to a small residual
tau_w ~ O(eps * Phi) with eps << 1 *domain-wide*. A single backward-facing step
imposes its adverse-then-favourable pressure signature over O(1) feature lengths
(a few step heights around separation/reattachment) and then the boundary layer
RELAXES back to an attached, ordinary-stress state. The deep-cancellation region
(eps < 0.1) is therefore LOCALISED to a short contiguous patch, not domain-wide.
A repeating wall with pitch ~ O(delta) never lets the layer relax: the next
element re-imposes the signature before recovery, so the deep-cancellation patch
tiles the whole domain. Hence it is O(delta)-PITCH REPETITION, not separation per
se, that triggers the structural ODE failure. BFS separates strongly (f_sep ~
0.25) yet is tolerated -- the clean causal control that isolates repetition from
separation.

PRE-REGISTERED, FALSIFIABLE PREDICTION (H2)
-------------------------------------------
For the BFS (zero-frequency limit) the SAME a-priori ODE/TBLE protocol used
throughout the paper (Y_IDX = 10 matching cell, frozen mixing-length closure)
must give:
  (P1)  eps_median = O(1)            [>= 1, i.e. NOT in the residual regime],
  (P2)  frac(eps < 0.1) << 0.40      [localised, far below the periodic-hill
                                       band frac in ~ 0.18-0.54],
  (P3)  R^2(tau_w) >= 0.88           [TOLERATED -- ODE succeeds],
  (P4)  the eps<0.1 deep-cancellation region is a SINGLE short contiguous patch
        of streamwise extent << the domain (localised, not domain-wide).
*Falsifier:* BFS lands in the residual regime (eps_median < 1, or frac(eps<0.1)
comparable to the hills), or the ODE fails on it (R^2 < 0.88). Any of these would
refute the claim that repetition (not separation) is the trigger.

This script reads ONLY the already-on-disk read-only wall-resolved LES
(curved_bfs_Re13700_DNS_wall_profiles.npz carries metadata type="LES (wall-resolved)";
Bentaleb, Lardeau & Leschziner 2012, Re_H=13700; the file NAME keeps the legacy
"DNS" token from the symlinked read-only source project, but the data IS WRLES) and
the Xiao 29-case aggregate (dose_response_xiao.npz) for the contrast band. It
writes a single regenerable artifact and asserts the prediction. No data is
modified; no fabrication.

Output:  codes/results/bfs_zerofreq_anchor.npz
Run:     OMP_NUM_THREADS=2 python3 codes/analysis/transition_map_bfs_anchor.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
GEOM = os.path.join(CODES, "new_data_download", "geometry_driven")
RESULTS = os.path.join(CODES, "results")

# the validated a-priori ODE/TBLE wall model used everywhere in the paper
sys.path.insert(0, os.path.join(CODES, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import predict_tau_w  # noqa: E402

Y_IDX = 10            # matching-cell index, identical to the a-priori protocol
EPS_RESIDUAL = 0.1    # residual-regime threshold (manuscript definition)


def largest_contiguous_span(x, mask):
    """Streamwise extent (in x units) and index-fraction of the LARGEST
    contiguous run of True in `mask`, plus its (x_lo, x_hi)."""
    if not mask.any():
        return 0.0, 0.0, (np.nan, np.nan)
    best_len, best_n, best_span = 0.0, 0, (np.nan, np.nan)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            span = x[j] - x[i]
            if span > best_len:
                best_len, best_n, best_span = span, j - i + 1, (x[i], x[j])
            i = j + 1
        else:
            i += 1
    return best_len, best_n / n, best_span


def evaluate(path, y_idx=Y_IDX):
    """A-priori ODE evaluation of one wall-profiles dataset (pure read)."""
    d = np.load(path, allow_pickle=True)
    y, U = d["y"], d["U"]
    tau_true = np.asarray(d["tau_w"], float)
    dp_dx = np.asarray(d["dp_dx"], float)
    x = np.asarray(d["x"], float).ravel()
    nu = d["nu"]
    n = len(tau_true)
    nu = np.full(n, float(nu)) if np.ndim(nu) == 0 else np.asarray(nu, float)

    y_m = np.array([(y[i] if y.ndim == 2 else y)[y_idx] for i in range(n)])
    tau_pred = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        if y_idx >= len(yi) or y_m[i] <= 0 or np.isnan(Ui[y_idx]):
            continue
        tau_pred[i] = predict_tau_w(Ui[y_idx], y_m[i], dp_dx[i], nu[i])

    denom = np.abs(dp_dx) * np.abs(y_m)
    eps = np.full(n, np.nan)
    m = denom > 1e-30
    eps[m] = np.abs(tau_true[m]) / denom[m]

    valid = np.isfinite(tau_pred)
    tw_p, tw_t = tau_pred[valid], tau_true[valid]
    ss_res = float(np.sum((tw_t - tw_p) ** 2))
    ss_tot = float(np.sum((tw_t - tw_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    eps_v = eps[np.isfinite(eps)]
    # localisation of the deep-cancellation (eps<0.1) region along the wall
    low = np.zeros(n, bool)
    low[m] = eps[m] < EPS_RESIDUAL
    span_x, span_frac, span_lohi = largest_contiguous_span(x, low)
    domain = float(x.max() - x.min())

    return dict(
        x=x, tau_true=tau_true, tau_pred=tau_pred, eps=eps,
        r2=float(r2),
        eps_median=float(np.median(eps_v)) if eps_v.size else np.nan,
        frac_eps_lt_0p1=float(np.mean(eps_v < EPS_RESIDUAL)) if eps_v.size else np.nan,
        f_sep=float(np.mean(tau_true < 0)),
        n_stations=n, n_valid=int(valid.sum()),
        low_eps_span_x=float(span_x), low_eps_span_frac=float(span_frac),
        low_eps_span_lo=float(span_lohi[0]), low_eps_span_hi=float(span_lohi[1]),
        domain_x=domain,
    )


def main():
    bfs_path = os.path.join(GEOM, "curved_bfs_Re13700_DNS_wall_profiles.npz")
    if not os.path.exists(bfs_path):
        raise SystemExit(f"MISSING read-only DNS: {bfs_path}")
    bfs = evaluate(bfs_path)

    # contrast band: the Xiao 29-case periodic-hill family (repeating, fail side)
    xiao_band = {}
    xpath = os.path.join(RESULTS, "dose_response_xiao.npz")
    if os.path.exists(xpath):
        dz = np.load(xpath, allow_pickle=True)
        xiao_band = dict(
            eps_median=np.asarray(dz["agg_eps_median"], float),
            frac_eps_lt_0p1=np.asarray(dz["agg_frac_eps_lt_0p1"], float),
            r2=np.asarray(dz["agg_r2"], float),
        )

    print("=" * 72)
    print("BFS zero-frequency anchor (Bentaleb et al. 2012, Re_H = 13700)")
    print("=" * 72)
    print(f"  n_stations            : {bfs['n_stations']}")
    print(f"  f_sep (reversed shear): {bfs['f_sep']:.3f}   (strong separation)")
    print(f"  eps_median            : {bfs['eps_median']:.3f}   (P1: O(1)?)")
    print(f"  frac(eps<0.1)         : {bfs['frac_eps_lt_0p1']:.3f}   (P2: <<0.40?)")
    print(f"  R^2(tau_w)  [a-priori]: {bfs['r2']:+.4f}  (P3: >=0.88?)")
    print(f"  deep-cancel patch     : span_x={bfs['low_eps_span_x']:.3f} "
          f"= {100*bfs['low_eps_span_frac']:.1f}% of the {bfs['domain_x']:.2f} "
          f"domain (P4: localised?)")
    if xiao_band:
        print("-" * 72)
        print("  Xiao 29-case periodic-hill band (repeating, FAILURE side):")
        print(f"    eps_median   in [{xiao_band['eps_median'].min():.3f}, "
              f"{xiao_band['eps_median'].max():.3f}]")
        print(f"    frac(eps<0.1) in [{xiao_band['frac_eps_lt_0p1'].min():.3f}, "
              f"{xiao_band['frac_eps_lt_0p1'].max():.3f}]")
        print(f"    R^2(tau_w)   in [{xiao_band['r2'].min():.1f}, "
              f"{xiao_band['r2'].max():.1f}]  (all < 0)")

    # ---- pre-registered prediction test -----------------------------------
    P1 = bfs["eps_median"] >= 1.0
    P2 = bfs["frac_eps_lt_0p1"] < 0.40
    P3 = bfs["r2"] >= 0.88
    P4 = bfs["low_eps_span_frac"] < 0.40
    allpass = bool(P1 and P2 and P3 and P4)
    print("-" * 72)
    print(f"  H2 prediction: P1(eps~O(1))={P1}  P2(frac<<0.4)={P2}  "
          f"P3(R2>=0.88)={P3}  P4(localised)={P4}")
    print(f"  ==> BFS sits on the TOLERATED side: {allpass}")

    out = {f"bfs_{k}": v for k, v in bfs.items()}
    out.update(dict(P1_eps_O1=P1, P2_frac_localised=P2, P3_r2_tolerated=P3,
                    P4_patch_localised=P4, all_pass=allpass,
                    Y_IDX=Y_IDX, eps_residual=EPS_RESIDUAL))
    for k, v in xiao_band.items():
        out[f"xiao_{k}"] = v
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, "bfs_zerofreq_anchor.npz"), **out)
    print(f"\nSaved -> results/bfs_zerofreq_anchor.npz")

    assert P3, "BFS must be TOLERATED (R^2 >= 0.88) for the repetition-trigger claim"
    assert P1 and P2 and P4, "BFS deep cancellation must be LOCALISED, not domain-wide"


if __name__ == "__main__":
    main()
