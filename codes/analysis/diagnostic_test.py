#!/usr/bin/env python3
"""
THE DIAGNOSTIC TEST: Does the ODE fail because of missing convective terms or bad turbulence model?

Test 1: Standard ODE with mixing-length (baseline)
  d/dy[(nu + nu_t) dU/dy] = dp/dx
  - Uses the exact solver from the main manuscript (ode_wall_model.py) for comparison.

Test 2: Controlled Mixing-Length (numerical consistency check)
  - Same physics as Test 1, but reimplemented with the exact same grid/numerics as Test 3.

Test 3: Controlled DNS Reynolds stress (closure-bypass control)
  d/dy[nu dU/dy - <u'v'>_DNS] = dp/dx
  - Uses the SAME grid (500 pts, wall-clustered) and integration (trapezoidal) as Test 2.
  - If this still fails, the mixing-length closure is not the primary issue.

All tests on periodic hills case_1p0.

Outputs:
  - codes/results/diagnostic_test_results.npz
"""
import numpy as np
from scipy.optimize import brentq
import os, sys, time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'utils')))
from data_paths import get_uwf_analysis_dir, get_uwf_results_dir

_UWF_RESULTS = str(get_uwf_results_dir())
RAW_DIR = os.path.join(_UWF_RESULTS, 'pehill_raw', 'case_1p0')
PREPROCESSED_PATH = os.path.join(_UWF_RESULTS, 'periodic_hills_case_1p0_wall_profiles.npz')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RE_B = 5600.0
NU = 1.0 / RE_B
# Control parameters
KAPPA = 0.41
A_PLUS = 26.0
N_GRID = 500  # Fine grid for controlled tests

print("=" * 80)
print("DIAGNOSTIC TEST: Why does the ODE fail on periodic hills?")
print("=" * 80)

# Load pre-processed dataset
print("\nLoading pre-processed dataset...")
pp = np.load(PREPROCESSED_PATH, allow_pickle=True)
pp_tau_w = pp['tau_w']       # ground-truth tau_w (512 profiles)
pp_dp_dx = pp['dp_dx']       # pressure gradient (512 profiles)
pp_y = pp['y']               # wall-normal coordinates
pp_U = pp['U']               # mean velocity
n_profiles_pp = len(pp_tau_w)

# FIX: Define separation consistently based on tau_w < 0
# Ignore pp['is_separated'] as it may use a different definition
is_separated = pp_tau_w < 0
n_sep = np.sum(is_separated)
print(f"Profiles: {n_profiles_pp}, Separated (tau_w < 0): {n_sep} ({n_sep/n_profiles_pp*100:.1f}%)")

# Load raw DNS data for Reynolds stress profiles
print("Loading raw DNS data for Reynolds stress profiles...")
mean_data = np.loadtxt(os.path.join(RAW_DIR, 'mean_files.dat'))
rms_data = np.loadtxt(os.path.join(RAW_DIR, 'rms_files1.dat'))

x_all = mean_data[:, 0]
y_all = mean_data[:, 1]
U_all = mean_data[:, 2]
V_all = mean_data[:, 3]
uv_all = rms_data[:, 5]  # Reynolds shear stress <u'v'>

x_unique = np.unique(x_all)

# Build profiles
profiles = []
for idx in range(n_profiles_pp):
    xi = float(pp['x'][idx]) if pp['x'].ndim == 1 else float(pp['x'][idx, 0])
    mask = np.abs(x_all - xi) < 1e-6
    if mask.sum() == 0:
        nearest = np.argmin(np.abs(x_unique - xi))
        xi_raw = x_unique[nearest]
        mask = np.abs(x_all - xi_raw) < 1e-8

    y_raw = y_all[mask]
    U_raw = U_all[mask]
    V_raw = V_all[mask]
    uv_raw = uv_all[mask]

    sort_idx = np.argsort(y_raw)
    y_raw = y_raw[sort_idx]
    U_raw = U_raw[sort_idx]
    V_raw = V_raw[sort_idx]
    uv_raw = uv_raw[sort_idx]

    wall_mask = y_raw < 1.5
    if wall_mask.sum() < 5:
        continue

    profiles.append({
        'x': xi,
        'y': y_raw[wall_mask],
        'U': U_raw[wall_mask],
        'uv': uv_raw[wall_mask],
        'tau_w_dns': float(pp_tau_w[idx]),
        'dpdx': float(pp_dp_dx[idx]),
        'is_sep': bool(is_separated[idx]),
    })

print(f"Organized {len(profiles)} wall-normal profiles")

# ====================================================================
# SOLVER 1: Standard ODE Solver (imported)
# ====================================================================
sys.path.insert(0, str(get_uwf_analysis_dir()))
from ode_wall_model import predict_tau_w as _main_predict_tau_w

# ====================================================================
# SOLVER 2: Controlled Solver (Internal)
# Uses identical grid and integration for ML and DNS-stress modes
# ====================================================================
def solve_controlled(tau_w_guess, y_match, dp_dx, mode, prof_data):
    """
    Integrate ODE from 0 to y_match using N_GRID points.
    Returns U(y_match).
    """
    u_tau = np.sqrt(max(abs(tau_w_guess), 1e-30))
    
    # 1. Consistent Grid: Clustered near wall
    eta = np.linspace(0, 1, N_GRID)
    y_grid = y_match * eta**1.5
    
    # 2. Physics Terms
    if mode == 'mixing_length':
        # Reimplement exact mixing length physics on this grid
        y_plus = y_grid * u_tau / NU
        D = 1.0 - np.exp(-y_plus / A_PLUS)
        l_m = KAPPA * y_grid * D
        l_m2 = l_m * l_m
        
        F = tau_w_guess + dp_dx * y_grid
        absF = np.abs(F)
        signF = np.sign(F)
        
        disc = NU * NU + 4.0 * l_m2 * absF
        viscous = l_m2 < 1e-30
        l_m2_safe = np.where(viscous, 1.0, l_m2)
        absS = (-NU + np.sqrt(disc)) / (2.0 * l_m2_safe)
        absS = np.where(viscous, absF / NU, absS)
        dUdy = signF * absS
        
    elif mode == 'dns_stress':
        # Interpolate DNS stress onto this grid
        uv_interp = np.interp(y_grid, prof_data['y'], prof_data['uv'])
        
        # nu * dU/dy - uv = tau_w + dp/dx * y
        # dU/dy = (tau_w + dp/dx * y + uv) / nu
        rhs = tau_w_guess + dp_dx * y_grid + uv_interp
        dUdy = rhs / NU
        
    else:
        return np.nan

    # 3. Integrate (Trapezoidal)
    dy = np.diff(y_grid)
    U_pred = np.sum(0.5 * (dUdy[:-1] + dUdy[1:]) * dy)
    
    return U_pred

def predict_tau_w_controlled(U_match, y_match, dp_dx, mode, prof_data):
    if abs(U_match) < 1e-15: return 0.0
    
    def residual(tw):
        return solve_controlled(tw, y_match, dp_dx, mode, prof_data) - U_match
        
    try:
        # Wide search range for separated flows
        return brentq(residual, -10.0, 10.0, maxiter=100, xtol=1e-8)
    except:
        return np.nan

# ====================================================================
# RUN TESTS
# ====================================================================
Y_IDX = 10  # matching height index

modes = [
    ('standard_ml', 'Test 1: Standard Mixing Length (Baseline)'),
    ('controlled_ml', 'Test 2: Controlled Mixing Length (Method Check)'),
    ('controlled_dns', 'Test 3: Controlled DNS Stress (Physics Check)'),
]

results = {m[0]: [] for m in modes}
ground_truth = []
valid_mask = []
is_sep_mask = []

t0 = time.time()
print(f"\nRunning tests on {len(profiles)} profiles...")

for i, prof in enumerate(profiles):
    if Y_IDX >= len(prof['y']):
        valid_mask.append(False)
        is_sep_mask.append(False)
        ground_truth.append(np.nan)
        for m in modes: results[m[0]].append(np.nan)
        continue

    y_match = prof['y'][Y_IDX]
    U_match = prof['U'][Y_IDX]
    dpdx = prof['dpdx']
    
    # 1. Standard
    tw1 = _main_predict_tau_w(U_match, y_match, dpdx, NU)
    
    # 2. Controlled ML
    tw2 = predict_tau_w_controlled(U_match, y_match, dpdx, 'mixing_length', prof)
    
    # 3. Controlled DNS
    tw3 = predict_tau_w_controlled(U_match, y_match, dpdx, 'dns_stress', prof)
    
    results['standard_ml'].append(tw1)
    results['controlled_ml'].append(tw2)
    results['controlled_dns'].append(tw3)
    ground_truth.append(prof['tau_w_dns'])
    valid_mask.append(True)
    is_sep_mask.append(prof['is_sep'])
    
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(profiles)} done ({time.time()-t0:.1f}s)")

# Convert to arrays
for k in results: results[k] = np.array(results[k])
ground_truth = np.array(ground_truth)
valid_mask = np.array(valid_mask)
is_sep_mask = np.array(is_sep_mask)

# ====================================================================
# ANALYZE & PRINT
# ====================================================================
print(f"\n{'='*80}")
print(f"{'Method':<25s} | {'R²':>8s} | {'RMSE':>8s} | {'SA (Sep)':>8s} | {'Valid':>5s}")
print(f"{'-'*80}")

final_data = {}

for mode_key, mode_name in modes:
    pred = results[mode_key]
    
    # Filtering
    mask = valid_mask & np.isfinite(pred)
    if mask.sum() < 10:
        print(f"{mode_key:<25s} | {'FAIL':>8s} | ...")
        continue
        
    p = pred[mask]
    t = ground_truth[mask]
    s = is_sep_mask[mask]
    
    ss_res = np.sum((p - t)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.mean((p - t)**2))
    
    # Sign accuracy on separated profiles ONLY
    # (Consistent with Judge's request for clean comparison)
    if s.sum() > 0:
        sa_sep = np.mean(np.sign(p[s]) == np.sign(t[s]))
    else:
        sa_sep = np.nan
        
    print(f"{mode_key:<25s} | {r2:8.4f} | {rmse:8.4f} | {sa_sep:8.4f} | {mask.sum():5d}")
    
    final_data[mode_key] = {
        'r2': r2, 'rmse': rmse, 'sa_sep': sa_sep, 
        'pred': pred, 'valid': mask
    }

print(f"{'-'*80}")
print(f"Separated profiles (tau_w < 0): {is_sep_mask.sum()}")
print(f"Method consistency check (Std vs Ctrl ML):")
diff = results['standard_ml'][valid_mask] - results['controlled_ml'][valid_mask]
mae = np.mean(np.abs(diff))
print(f"  MAE between Standard and Controlled ML: {mae:.6f}")
if mae < 1e-4:
    print("  -> PASSED: Solvers are numerically consistent.")
else:
    print("  -> WARNING: Numerical discrepancy detected.")

# Save — include raw predictions, masks, AND derived metrics for provenance
table4_mask = valid_mask & is_sep_mask
save_dict = dict(
    tau_w_dns=ground_truth,
    is_separated=is_sep_mask,
    valid_mask=valid_mask,
    table4_mask=table4_mask,
    n_profiles=len(profiles),
    n_separated=int(is_sep_mask.sum()),
)
for k, v in final_data.items():
    save_dict[k] = v['pred']
    save_dict[f'{k}_valid'] = v['valid']
    save_dict[f'{k}_r2'] = v['r2']
    save_dict[f'{k}_rmse'] = v['rmse']
    save_dict[f'{k}_sa_sep'] = v['sa_sep']

np.savez(os.path.join(RESULTS_DIR, 'diagnostic_test_results.npz'), **save_dict)
print(f"\nSaved to {os.path.join(RESULTS_DIR, 'diagnostic_test_results.npz')}")
