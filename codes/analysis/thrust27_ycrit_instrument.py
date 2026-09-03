#!/usr/bin/env python3
"""
thrust27_ycrit_instrument.py  --  Thrust #27, Level 1 (Core methodology).

THE TRANSMISSION-CRITERION INSTRUMENT (specification + runnable scaffold).

Computes / collects, for the five coupled repeating-geometry cases:
  - y_crit+(geometry)  -- a-priori critical matching height, Eq (1)
  - y_m+(deployed)     -- coupled WMLES first-cell height
  - r = y_m+/y_crit+   -- the transmission ratio (Eq 2)
and the cross-geometry collapse statistics (F1 vs F3) + the n=5 honesty
diagnostics (rho of r vs of bare y_m+).

DESIGN PRINCIPLES (honesty, after the L0 Judge's four fixes):
  * y_crit uses the Thrust #17 definition with PER-GEOMETRY eps_c (NOT a fixed
    0.22): the on-disk map critical_matching_height_map.npz stores eps_c per
    geometry (krank 0.223, periodic_hills_1p0 1.241, conv_div 0.128).  This
    instrument REPRODUCES those three anchors by reading the #17 map, and only
    then extends to the two Xiao cases (alpha=0.8, 1.2) that the map lacks.
  * The Xiao extension needs a-priori reference fields with the SAME sweep
    protocol.  Those reference files may not yet exist on disk -- this is the
    load-bearing L2 task the L0 Judge flagged.  The instrument checks at runtime
    and, if absent, reports the case as DEFERRED rather than fabricating a value.
  * Nothing is hardcoded; every number is read from disk or computed and printed
    with its source.  Run at L2; do NOT quote its output before then.

Usage:  OMP_NUM_THREADS=2 python3 codes/analysis/thrust27_ycrit_instrument.py
"""
import os, json, glob
import numpy as np
try:
    from scipy.stats import spearmanr
except Exception:                       # spearman = pearson of ranks (no scipy)
    def spearmanr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return (np.corrcoef(ra, rb)[0, 1], None)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "codes", "results")

def load(name):
    p = os.path.join(RES, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None

# ---- the five coupled cases, in the order of aposteriori_dose_response.npz ----
# (name in #17 map if present, coupled file candidates, alpha)
CASES = [
    {"key": "xiao_a0p8", "alpha": 0.8, "in_map": None,
     "ref_candidates": ["cancellation_parameter_xiao_a0p8.npz",
                         "cancellation_parameter_xiao_0p8.npz"],
     "cpl_candidates": ["aposteriori_dose_response_xiao_0p8.npz",
                        "aposteriori_dose_response_xiao_a0p8.npz"]},
    {"key": "xiao_a1p0", "alpha": 1.0, "in_map": "periodic_hills_1p0",
     "ref_candidates": ["cancellation_parameter_xiao_a1p0.npz",
                         "cancellation_parameter_xiao_1p0.npz",
                         "cancellation_parameter.npz"],
     "cpl_candidates": ["aposteriori_wmles_pehill.npz",
                        "aposteriori_dose_response_xiao_1p0.npz"]},
    {"key": "xiao_a1p2", "alpha": 1.2, "in_map": None,
     "ref_candidates": ["cancellation_parameter_xiao_a1p2.npz",
                         "cancellation_parameter_xiao_1p2.npz"],
     "cpl_candidates": ["aposteriori_dose_response_xiao_1p2.npz",
                        "aposteriori_dose_response_xiao_a1p2.npz"]},
    {"key": "krank_pehill_Re10595", "alpha": 1.0, "in_map": "krank_pehill_Re10595",
     "ref_candidates": ["cancellation_parameter.npz"],
     "cpl_candidates": ["aposteriori_wmles_krank.npz", "aposteriori_wmles_pehill.npz"]},
    {"key": "conv_div_channel", "alpha": np.nan, "in_map": "conv_div_channel",
     "ref_candidates": ["cancellation_parameter_conv_div.npz"],
     "cpl_candidates": ["aposteriori_wmles_convdiv.npz"]},
]

def first_existing(cands):
    for c in cands:
        if os.path.exists(os.path.join(RES, c)):
            return c
    return None

def ycrit_from_map(map_npz, geom_name):
    """Read the verified #17 y_crit+ and eps_c for a named geometry."""
    keys = [str(k) for k in map_npz["keys"]]
    if geom_name not in keys:
        return None, None
    i = keys.index(geom_name)
    return float(map_npz["ycrit"][i]), float(map_npz["eps_c"][i])

def ycrit_compute(ref, eps_c):
    """Eq (1) from a-priori reference fields with a GIVEN eps_c."""
    for k in ("tau_w", "dpdx", "u_tau", "nu"):
        if k not in ref.keys():
            return None
    tau = float(np.nanmedian(np.abs(ref["tau_w"])))
    dp  = float(np.nanmedian(np.abs(ref["dpdx"])))
    return tau / (eps_c * dp) * float(ref["u_tau"]) / float(ref["nu"])

def deployed_ym_plus(cpl):
    for k in ("y_m_plus", "ym_plus", "y_m+", "deployed_ym_plus"):
        if cpl is not None and k in cpl.keys():
            v = cpl[k]
            return float(np.nanmedian(v)) if getattr(v, "size", 1) > 1 else float(v)
    return None

def main():
    dr  = load("aposteriori_dose_response.npz")
    m17 = load("critical_matching_height_map.npz")
    assert dr is not None,  "aposteriori_dose_response.npz missing"
    assert m17 is not None, "critical_matching_height_map.npz (#17 anchors) missing"
    e_reatt = np.asarray(dr["e_reatt"], float)
    eps_med = np.asarray(dr["eps_med"], float)
    labels  = [str(x) for x in dr["labels"]]

    print("=" * 92)
    print("Thrust #27 L1 transmission-criterion instrument  (run at L2)")
    print("=" * 92)
    print(f"{'case':22s} {'alpha':>5s} {'eps_c':>7s} {'ycrit+':>9s} {'src':>9s} "
          f"{'ym+':>7s} {'r':>7s} {'eps_med':>7s} {'e_reatt':>8s}")

    yc, ym, status = [], [], []
    for c, lab in zip(CASES, labels):
        eps_c = None; ycp = None; src = "-"
        # (a) prefer the verified #17 anchor
        if c["in_map"]:
            ycp, eps_c = ycrit_from_map(m17, c["in_map"])
            if ycp is not None: src = "#17map"
        # (b) else compute from reference fields -- needs eps_c from a sweep (L2)
        if ycp is None:
            ref_f = first_existing(c["ref_candidates"])
            if ref_f is not None:
                # eps_c MUST come from the per-geometry sweep (Stage A protocol).
                # Until that sweep is wired for this case, do NOT invent eps_c.
                src = f"REF:{ref_f} (eps_c sweep PENDING-L2)"
            else:
                src = "DEFERRED (no reference field on disk)"
        cpl_f = first_existing(c["cpl_candidates"])
        ymp = deployed_ym_plus(load(cpl_f)) if cpl_f else None
        yc.append(ycp); ym.append(ymp)
        st = "OK" if (ycp is not None and ymp is not None) else "PENDING"
        status.append(st)
        r = (ymp / ycp) if (ycp and ymp) else np.nan
        eps_c_s = f"{eps_c:7.3f}" if eps_c is not None else "   --  "
        ycp_s   = f"{ycp:9.2f}" if ycp is not None else "   PEND  "
        ymp_s   = f"{ymp:7.2f}" if ymp is not None else "  PEND "
        print(f"{c['key']:22s} {c['alpha']:5.1f} {eps_c_s} {ycp_s} {src:>9s} "
              f"{ymp_s} {r:7.3f} {eps_med[CASES.index(c)]:7.3f} {e_reatt[CASES.index(c)]:8.3f}")

    yc = np.array(yc, float); ym = np.array(ym, float); r = ym / yc
    ok = np.array([s == "OK" for s in status])
    print("-" * 92)
    # F1 is always computable (on disk)
    rho_eps = spearmanr(eps_med, e_reatt)[0]
    print(f"F1  Spearman(eps_med, e_reatt)            = {rho_eps:+.3f}   (expect ~+0.10)")
    if ok.sum() >= 4:
        rho_ratio = spearmanr(r[ok], e_reatt[ok])[0]
        rho_ym    = spearmanr(ym[ok], e_reatt[ok])[0]
        rho_yc    = spearmanr(yc[ok], e_reatt[ok])[0]
        print(f"F3  Spearman(r=ym/ycrit, e_reatt)         = {rho_ratio:+.3f}  (n={ok.sum()})")
        print(f"    Spearman(ym+ alone, e_reatt)          = {rho_ym:+.3f}   <- degeneracy check")
        print(f"    Spearman(ycrit+ alone, e_reatt)       = {rho_yc:+.3f}")
        print(f"    [HONESTY] r vs ym+ rank-degenerate?    -> {abs(rho_ratio-rho_ym) < 1e-9}")
        np.savez(os.path.join(RES, "thrust27_collapse.npz"),
                 labels=np.array(labels, object), e_reatt=e_reatt, eps_med=eps_med,
                 y_m_plus=ym, y_crit_plus=yc, ratio=r, status=np.array(status, object),
                 rho_eps=float(rho_eps), rho_ratio=float(rho_ratio),
                 rho_ym=float(rho_ym), rho_ycrit=float(rho_yc),
                 note="Thrust #27 collapse; per-geometry eps_c from #17 map; "
                      "Xiao a0p8/a1p2 require L2 sweep before inclusion.")
        print("wrote thrust27_collapse.npz")
    else:
        print(f"F3  DEFERRED: only {ok.sum()}/5 cases have both y_crit+ and y_m+.")
        print("    -> L2 must (i) run the per-geometry eps_c sweep for xiao a0p8/a1p2,")
        print("       (ii) confirm their a-priori reference fields exist, (iii) re-run this.")
    print("-" * 92)
    print("NOTE: this is the L1 instrument; numbers are produced only when RUN at L2.")

if __name__ == "__main__":
    main()
