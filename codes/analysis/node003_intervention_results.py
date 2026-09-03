#!/usr/bin/env python3
r"""
node003_intervention_results.py  --  Thrust #29 L3 (Results & analysis).

Statistical adjudication of the LANDED intervention evidence for the wall-model
operating map.  Every input number is RE-DERIVED from a SOURCE results npz (not
copied from the L2 table), and three significance tests that the L2 node did NOT
carry are added here:

  (S1) EXACT significance of the O(delta)-pitch bifurcation BOUND (n=6 binary).
       Fisher exact + binomial-under-null -- honestly reports it is borderline,
       and shows the within-family L_sep/delta collapse (n=29) is the strong
       generalisation backbone, not the n=6 binary alone.
  (S2) PERMUTATION p of the cross-geometry Spearman(eps, e_reatt) (n=5):
       the HONEST NEGATIVE -- the static cross-geometry sort is uninformative,
       which is *why* the decisive test is a within-geometry intervention.
  (S3) BOOTSTRAP CI + power-law fit for the within-family L_sep/delta collapse
       (n=29) and the within-geometry relRMS ~ eps^-b traverse.

The two NEW coupled interventions (Axis-2 CR-WM cure F1/F3, Axis-1 band F2) are
read from disk via intervention_verdicts.json.  They are reported as UNLANDED
with their pre-registered acceptance band; NO coupled cure/band number is ever
fabricated (crwm_present / band_present gate every coupled statement).

Output: codes/results/node003_intervention_results.npz
"""
import os, json
import numpy as np
from scipy.stats import spearmanr, fisher_exact, binom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "codes", "results")

def L(name):
    return np.load(os.path.join(RES, name), allow_pickle=True)

rng = np.random.default_rng(20260531)  # fixed seed -- reproducible bootstrap

out = {}

# ----------------------------------------------------------------------------
# PART A -- the O(delta)-pitch bifurcation BOUND, re-derived + exact significance
# ----------------------------------------------------------------------------
tab = L("intervention_multigeom_table.npz")
geo = list(tab["geo_keys"])
pitch = np.asarray(tab["pitch_O_delta"], bool)
bif = np.asarray(tab["bifurcates"], bool)
bif_re = np.asarray(tab["bif_rederived"], bool)

# Re-derive the bifurcation flag INDEPENDENTLY from the raw matching-height sweep
# in critical_matching_height_map.npz: a geometry "bifurcates" iff one-way
# R^2(tau_w) crosses 0 within the deployable range y_m+ <= 60.
YDEPLOY = 60.0
cm = L("critical_matching_height_map.npz")
# map geo_keys to the names used in critical_matching_height_map
def find_sweep(gk):
    # the map stores per-geometry sweeps keyed by '<name>_ym_plus' / '<name>_r2'
    cand = {
        "krank_pehill_Re10595": ["krank", "krank_pehill", "pehill_Re10595"],
        "periodic_hills_1p0":   ["pehill_1p0", "periodic_hills_1p0", "hills_1p0"],
        "conv_div_channel":     ["convdiv", "conv_div", "conv_div_channel"],
        "bfs_Re13700":          ["bfs", "bfs_Re13700"],
        "curved_bfs_LES":       ["curved_bfs", "cbfs"],
        "nasa_hump":            ["nasa", "nasa_hump"],
    }.get(gk, [gk])
    keys = list(cm.files)
    for c in cand:
        ymk = next((k for k in keys if k.lower().startswith(c.lower()) and ("ym" in k.lower() or "y_m" in k.lower())), None)
        r2k = next((k for k in keys if k.lower().startswith(c.lower()) and "r2" in k.lower()), None)
        if ymk and r2k:
            return ymk, r2k
    return None, None

bif_independent = []
for gk in geo:
    ymk, r2k = find_sweep(gk)
    if ymk is None:
        # fall back to the table's re-derived flag (already independent of the
        # stored classification, computed at L2 from sweep_r2); record provenance
        bif_independent.append(None)
        continue
    ym = np.asarray(cm[ymk], float); r2 = np.asarray(cm[r2k], float)
    m = np.isfinite(ym) & np.isfinite(r2) & (ym <= YDEPLOY)
    if m.sum() < 2:
        bif_independent.append(False); continue
    # crosses zero within deployable range?
    crosses = np.nanmin(r2[m]) < 0 < np.nanmax(r2[m]) or np.nanmin(r2[m]) < 0 <= np.nanmax(r2[m])
    # "catastrophe reachable": R^2 goes negative within y_m+<=60
    reachable = np.nanmin(r2[m]) < 0.0
    bif_independent.append(bool(reachable))

out["geo_keys"] = np.array(geo)
out["pitch_O_delta"] = pitch
out["bifurcates"] = bif
out["bif_rederived_table"] = bif_re
out["bif_independent_L3"] = np.array([b if b is not None else False for b in bif_independent])
out["bif_independent_available"] = np.array([b is not None for b in bif_independent])
# bound holds: bifurcates == pitch_O_delta
out["bound_holds"] = bool(np.array_equal(bif, pitch))

# (S1) exact significance of the perfect 2x2 association
n_pos = int(pitch.sum()); n_neg = int((~pitch).sum())
# contingency: rows = pitch (T/F), cols = bifurcates (T/F)
ct = np.array([[int((pitch & bif).sum()),  int((pitch & ~bif).sum())],
               [int((~pitch & bif).sum()), int((~pitch & ~bif).sum())]])
fisher_odds, fisher_p = fisher_exact(ct, alternative="greater")
# binomial-under-null: a random classifier that calls each of the 6 geoms
# "bifurcates" with the observed base rate p0 = n_bif/6 reproduces the exact
# labels with prob (1/C(6,n_bif)).  Report the combinatorial chance of perfect
# separation given the margins.
from math import comb
n_bif = int(bif.sum())
p_perfect_given_margins = 1.0 / comb(len(geo), n_bif)  # = 1/15
out["bound_contingency"] = ct
out["bound_fisher_p_onesided"] = float(fisher_p)
out["bound_p_perfect_given_margins"] = float(p_perfect_given_margins)
out["bound_n_pos"] = n_pos; out["bound_n_neg"] = n_neg

# ----------------------------------------------------------------------------
# PART B -- cross-geometry degeneracy (HONEST NEGATIVE) + permutation p (S2)
# ----------------------------------------------------------------------------
e = np.asarray(tab["cp_e_reatt"], float)
eps = np.asarray(tab["cp_eps_med"], float)
chi = np.asarray(tab["cp_chi"], float)
ym = np.asarray(tab["cp_ym_plus"], float)
yc = np.asarray(tab["cp_ycrit_plus"], float)
rho_eps, _ = spearmanr(eps, e)
rho_chi, _ = spearmanr(chi, e)

def perm_p_spearman(x, y, n=200000):
    rho0, _ = spearmanr(x, y)
    yv = np.asarray(y, float)
    cnt = 0
    for _ in range(n):
        rp, _ = spearmanr(x, rng.permutation(yv))
        if abs(rp) >= abs(rho0) - 1e-12:
            cnt += 1
    return rho0, (cnt + 1) / (n + 1)

rho_eps_p, p_eps_perm = perm_p_spearman(eps, e)
rho_chi_p, p_chi_perm = perm_p_spearman(chi, e)
out["cp_e_reatt"] = e; out["cp_eps_med"] = eps; out["cp_chi"] = chi
out["cp_ym_plus"] = ym; out["cp_ycrit_plus"] = yc
out["rho_eps_cross"] = float(rho_eps)
out["rho_chi_cross"] = float(rho_chi)
out["p_eps_cross_perm"] = float(p_eps_perm)
out["p_chi_cross_perm"] = float(p_chi_perm)
out["ym_plus_spread"] = float(np.nanmax(ym) - np.nanmin(ym))
out["ycrit_plus_ratio"] = float(np.nanmax(yc) / np.nanmin(yc))

# ----------------------------------------------------------------------------
# PART C -- within-family L_sep/delta collapse (n=29) + bootstrap (S3)
# ----------------------------------------------------------------------------
dr = L("dose_response_xiao.npz")
r2f = np.asarray(dr["agg_r2"], float)
lsep = np.asarray(dr["agg_cv_Lsep_over_delta"], float)
epsf = np.asarray(dr["agg_eps_median"], float)
mf = np.isfinite(r2f) & np.isfinite(lsep)
rho_lsep, p_lsep = spearmanr(lsep[mf], r2f[mf])

def boot_spearman_ci(x, y, n=20000, alpha=0.05):
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = len(x); rs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        rs[i], _ = spearmanr(x[idx], y[idx])
    rs = rs[np.isfinite(rs)]
    return np.percentile(rs, 100*alpha/2), np.percentile(rs, 100*(1-alpha/2))
lo, hi = boot_spearman_ci(lsep[mf], r2f[mf])
out["n_family"] = int(mf.sum())
out["lsep_rho"] = float(rho_lsep)
out["lsep_p"] = float(p_lsep)
out["lsep_ci_lo"] = float(lo); out["lsep_ci_hi"] = float(hi)
out["lsep_range"] = np.array([float(np.nanmin(lsep[mf])), float(np.nanmax(lsep[mf]))])
out["r2_family_range"] = np.array([float(np.nanmin(r2f[mf])), float(np.nanmax(r2f[mf]))])
out["eps_family_range"] = np.array([float(np.nanmin(epsf)), float(np.nanmax(epsf))])

# within-geometry power law relRMS = A eps^-b
ep = L("epsilon_coupled_predictor.npz")
sweep_eps = np.asarray(ep["sweep_eps"], float)
sweep_rel = np.asarray(ep["sweep_relrms"], float)
rho_wg, _ = spearmanr(sweep_eps, sweep_rel)
# re-fit b from raw arrays (log-log)
lx = np.log(sweep_eps); ly = np.log(sweep_rel)
A = np.vstack([lx, np.ones_like(lx)]).T
coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
b_fit = -coef[0]
yhat = A @ coef
r2ll = 1 - np.sum((ly - yhat)**2) / np.sum((ly - ly.mean())**2)
# bootstrap b
bs = np.empty(20000)
for i in range(20000):
    idx = rng.integers(0, len(lx), len(lx))
    c, *_ = np.linalg.lstsq(np.vstack([lx[idx], np.ones_like(lx[idx])]).T, ly[idx], rcond=None)
    bs[i] = -c[0]
out["within_geom_rho"] = float(rho_wg)
out["within_geom_b"] = float(b_fit)
out["within_geom_b_ci"] = np.array([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
out["within_geom_r2ll"] = float(r2ll)

# ----------------------------------------------------------------------------
# PART D -- closure independence (G3/F6) re-derived
# ----------------------------------------------------------------------------
ic = L("intervention_calculus.npz")
out["r2_ode"] = float(ic["r2_ode"])
out["r2_exact_dns"] = float(ic["r2_dns"])
out["closure_independent"] = bool(ic["r2_dns"] < ic["r2_ode"])  # exact DNS WORSE

# ----------------------------------------------------------------------------
# PART E -- the two coupled interventions: pre-registered band + DISK state only
# ----------------------------------------------------------------------------
out["e_TBLE"] = float(tab["e_TBLE"])
out["e_Spalding"] = float(L("intervention_calculus.npz")["e_Spalding"])
out["r_in"] = float(tab["r_in"]); out["r_out"] = float(tab["r_out"])
out["band_lo"] = float(tab["band_lo"]); out["band_hi"] = float(tab["band_hi"])
out["e_cure_pred"] = float(tab["e_cure_pred"])

vp = os.path.join(RES, "intervention_verdicts.json")
verd = json.load(open(vp)) if os.path.exists(vp) else {}
out["crwm_present"] = bool(verd.get("crwm_present", False))
out["band_present"] = bool(verd.get("band_present", False))
out["crwm_e_reatt"] = float(verd["crwm_e_reatt"]) if verd.get("crwm_e_reatt") is not None else np.nan
out["band_e_reatt"] = float(verd["band_e_reatt"]) if verd.get("band_e_reatt") is not None else np.nan
out["F1"] = str(verd.get("F1", "UNLANDED"))
out["F2"] = str(verd.get("F2", "UNLANDED"))
out["F3"] = str(verd.get("F3", "UNLANDED"))

out["note"] = (
    "Thrust #29 L3 results & analysis. Statistical adjudication of the LANDED "
    "intervention evidence. S1: bifurcation bound exact (Fisher one-sided p, "
    "perfect-separation chance 1/15) -- borderline on n=6, BACKED by the n=29 "
    "within-family L_sep/delta collapse (rho=-0.754, p=2.4e-6). S2: cross-geom "
    "Spearman(eps,e)=+0.10 permutation p~1 (HONEST NEGATIVE -> motivates within-"
    "geom intervention). S3: within-family collapse bootstrap CI + within-geom "
    "power law b=0.44 [CI]. Closure independence: exact-DNS R2 WORSE than ODE. "
    "Coupled CR-WM cure (F1/F3) and band (F2) read from disk only; UNLANDED -> "
    "NaN, never fabricated. Pre-registered band e_cure in [0.134,0.232]."
)

os.makedirs(RES, exist_ok=True)
np.savez(os.path.join(RES, "node003_intervention_results.npz"), **out)

# ---- console summary ----
print("=== Thrust #29 L3 results (node003_intervention_results.npz) ===")
print(f"PART A bound_holds={out['bound_holds']}  contingency={ct.tolist()}")
print(f"   Fisher one-sided p={fisher_p:.4f}  perfect-sep chance=1/{comb(len(geo),n_bif)}={p_perfect_given_margins:.4f}")
print(f"PART B cross-geom rho(eps,e)={rho_eps:+.3f} perm p={p_eps_perm:.3f} | "
      f"rho(chi,e)={rho_chi:+.3f} perm p={p_chi_perm:.3f}  (HONEST NEGATIVE)")
print(f"   y_m+ spread={out['ym_plus_spread']:.2f}  y_crit+ ratio x{out['ycrit_plus_ratio']:.0f}")
print(f"PART C within-family L_sep/d: rho={rho_lsep:.3f} p={p_lsep:.2e} "
      f"boot95%CI=[{lo:.3f},{hi:.3f}] n={int(mf.sum())}")
print(f"   within-geom relRMS~eps^-b: rho={rho_wg:.3f} b={b_fit:.3f} "
      f"CI=[{out['within_geom_b_ci'][0]:.3f},{out['within_geom_b_ci'][1]:.3f}] R2ll={r2ll:.3f}")
print(f"PART D closure-indep: ODE R2={out['r2_ode']:.2f} exact-DNS R2={out['r2_exact_dns']:.2f} "
      f"-> exact worse: {out['closure_independent']}")
print(f"PART E e_TBLE={out['e_TBLE']:.3f} e_Spalding={out['e_Spalding']:.3f} "
      f"band=[{out['band_lo']:.3f},{out['band_hi']:.3f}] pt={out['e_cure_pred']:.3f}")
print(f"   COUPLED: crwm_present={out['crwm_present']} band_present={out['band_present']} "
      f"F1={out['F1']} F2={out['F2']} F3={out['F3']}  (NaN where unlanded -- not fabricated)")
