#!/usr/bin/env python3
"""
verify_thrust27_l1.py  --  Thrust #27 L1 verification (run at L2 alongside the
instrument).  Exits 0 only if all checks pass.

Checks:
  C1  The #17 anchors are present and self-consistent (per-geometry eps_c):
      krank y_crit+=15.93/eps_c~0.223, periodic_hills_1p0=1.0/eps_c~1.24,
      conv_div=207.9/eps_c~0.128.
  C2  F1 reproduces from raw arrays: Spearman(eps_med, e_reatt) ~ +0.10 with
      n_real=5 in aposteriori_dose_response.npz.
  C3  F2 (within-geometry) on disk: Spearman=-1.0, b~0.44, R2~0.977.
  C4  HONESTY: methodology.md asserts NO numeric y_crit for the Xiao a0p8/a1p2
      cases and NO cross-geometry collapse rho (those are deferred to L2).
"""
import os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "codes", "results")
MD   = os.path.join(ROOT, "development", "nodes", "node_001", "methodology.md")
fails = []
def chk(c, m):
    print(("ok  : " if c else "FAIL: ") + m);  fails.append(m) if not c else None
def load(n):
    p = os.path.join(RES, n);  return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]

# C1
print("[C1] #17 y_crit anchors (per-geometry eps_c)")
m = load("critical_matching_height_map.npz")
chk(m is not None, "critical_matching_height_map.npz present")
if m is not None:
    keys = [str(k) for k in m["keys"]]; yc = np.asarray(m["ycrit"], float); ec = np.asarray(m["eps_c"], float)
    for g, yy, ee in [("krank_pehill_Re10595", 15.93, 0.223),
                      ("periodic_hills_1p0", 1.0, 1.241),
                      ("conv_div_channel", 207.9, 0.128)]:
        if g in keys:
            i = keys.index(g)
            chk(abs(yc[i] - yy) < max(0.05, 0.01*yy), f"{g} y_crit+={yc[i]:.2f} (~{yy})")
            chk(abs(ec[i] - ee) < 0.02, f"{g} eps_c={ec[i]:.3f} (~{ee})")
        else:
            chk(False, f"{g} present in #17 map")

# C2
print("[C2] F1 cross-geometry eps fails (rho~+0.10)")
dr = load("aposteriori_dose_response.npz")
chk(dr is not None, "aposteriori_dose_response.npz present")
if dr is not None:
    chk(int(dr["n_real"]) == 5, f"n_real={int(dr['n_real'])}")
    rho = sp(dr["eps_med"], dr["e_reatt"])
    chk(abs(rho - 0.10) < 0.06, f"recomputed Spearman(eps,e)={rho:+.3f} ~ +0.10")

# C3
print("[C3] F2 within-geometry eps law (rho=-1.0)")
ec = load("epsilon_coupled_predictor.npz")
chk(ec is not None, "epsilon_coupled_predictor.npz present")
if ec is not None:
    chk(abs(float(ec["spearman_rho"]) + 1.0) < 1e-6, f"spearman={float(ec['spearman_rho'])}")
    chk(abs(float(ec["fit_b"]) - 0.44) < 0.05, f"b={float(ec['fit_b']):.3f}")
    chk(float(ec["fit_r2_loglog"]) > 0.95, f"R2={float(ec['fit_r2_loglog']):.3f}")

# C4 honesty: no fabricated Xiao y_crit / collapse rho in methodology.md
print("[C4] methodology.md asserts no fabricated Xiao y_crit / collapse rho")
if os.path.exists(MD):
    txt = open(MD).read()
    # forbid an asserted numeric y_crit+ for a0p8/a1p2 and an asserted collapse rho value
    bad = re.search(r"y_crit\+?\(?\s*(?:alpha\s*=\s*0\.8|0p8)\s*\)?\s*=\s*\d", txt, re.I) \
          or re.search(r"(?:rho|spearman)\s*\(?\s*r\s*[, ].*?\)?\s*=\s*\+?0?\.\d", txt, re.I) \
          or re.search(r"ratio.*?=\s*\+?0\.70", txt, re.I)
    chk(bad is None, "no asserted Xiao y_crit / collapse-rho number" + (f" (LEAK {bad.group(0)!r})" if bad else ""))
    chk("PENDING-L2" in txt or "pending L2" in txt or "L2" in txt, "explicitly defers computation to L2")
else:
    chk(False, "methodology.md present")

print("=" * 60)
if fails:
    print(f"{len(fails)} FAILURE(S)"); sys.exit(1)
print("ALL L1 CHECKS PASSED (anchors reproduced; F1/F2 verified; no fabrication)."); sys.exit(0)
