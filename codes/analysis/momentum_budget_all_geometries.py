#!/usr/bin/env python3
"""
Momentum budget analysis for the core benchmark (9 geometries, 31 datasets;
the manuscript-Table-1 set minus the 3 periodic-hills variants with missing
Reynolds shear stress). The dataset list comes from codes/manifest.py.

For each wall-normal profile at each x-station, compute:
  - Convection: U·dU/dx + V·dU/dy
  - Pressure gradient: dp/dx (from data)
  - Viscous diffusion: nu·d²U/dy²
  - Reynolds stress: -d<uv>/dy
  - Ratio: |Convection| / (|Viscous| + |Reynolds|)

dU/dx is estimated via central differences between neighboring x-stations.
For single-station geometries (channels, pipes, Couette), convection = 0.

Outputs: results/momentum_budget_all_geometries.npz
"""
import numpy as np
import os, glob
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'utils')))
from data_paths import get_uwf_results_dir

# Single source of truth: which datasets belong to the core momentum budget.
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..')))
import manifest

WALL_DIR = str(get_uwf_results_dir())

print("=" * 80)
print("MOMENTUM BUDGET ANALYSIS — CORE BENCHMARK (manifest-driven)")
print("=" * 80)

# The core momentum budget is computed for EXACTLY the core benchmark declared
# in codes/manifest.py. We do NOT glob the data directory blindly: it is a
# read-only symlink that accumulated extended/exploratory datasets, and silently
# pulling those in is what made the downstream criterion_evaluation crash. Any
# *_wall_profiles.npz present but not in the manifest core is reported as IGNORED.
CORE = manifest.core_names()

available = {os.path.basename(f).replace('_wall_profiles.npz', ''): f
             for f in sorted(glob.glob(os.path.join(WALL_DIR, '*_wall_profiles.npz')))}

profile_files = []
missing = []
for name in CORE:
    if name in available:
        profile_files.append(available[name])
    else:
        missing.append(name)
if missing:
    raise FileNotFoundError(
        "Core datasets missing from the data directory: " + ", ".join(missing))

ignored = sorted(set(available) - set(CORE))
print(f"Data directory holds {len(available)} wall-profile datasets.")
print(f"Analysing {len(profile_files)} core datasets (from manifest).")
if ignored:
    print(f"IGNORED {len(ignored)} non-core datasets (extended/exploratory/"
          f"incomplete): {', '.join(ignored)}")
print()

# For near-wall analysis, use points from y_plus ~ 5 to y_plus ~ 50
# (inner layer where wall model operates)
Y_PLUS_MIN = 5.0
Y_PLUS_MAX = 50.0

results = {}

for fpath in profile_files:
    fname = os.path.basename(fpath).replace('_wall_profiles.npz', '')
    d = np.load(fpath, allow_pickle=True)

    x = d['x']
    y = d['y']          # (n_stations, n_y)
    y_plus = d['y_plus']
    U = d['U']          # (n_stations, n_y)
    V = d['V']
    uv = d['uv']
    nu_arr = d['nu']
    dp_dx = d['dp_dx']
    tau_w = d['tau_w']
    u_tau = d['u_tau']
    # FIX: Recalculate separation from tau_w to ensure consistency across paper
    # (The flag in the file might be based on a different definition)
    is_sep = tau_w < 0
    geom = str(d['geometry'])
    n_stations = len(x)

    print(f"--- {fname} ({geom}): {n_stations} stations ---")

    # Per-station budget analysis
    conv_rms_stations = np.zeros(n_stations)
    visc_rms_stations = np.zeros(n_stations)
    reyn_rms_stations = np.zeros(n_stations)
    pres_rms_stations = np.zeros(n_stations)
    ratio_stations = np.zeros(n_stations)
    n_nw_pts = np.zeros(n_stations, dtype=int)

    for i in range(n_stations):
        yi = y[i]
        ypi = y_plus[i]
        Ui = U[i]
        Vi = V[i]
        uvi = uv[i]
        nu = nu_arr[i]

        # Find near-wall region
        nw_mask = (ypi >= Y_PLUS_MIN) & (ypi <= Y_PLUS_MAX) & (yi > 0)
        nw_idx = np.where(nw_mask)[0]
        if len(nw_idx) < 3:
            continue
        n_nw_pts[i] = len(nw_idx)

        # dU/dy and d2U/dy2 via finite differences (non-uniform y)
        ny = len(yi)
        dUdy = np.zeros(ny)
        d2Udy2 = np.zeros(ny)
        duvdy = np.zeros(ny)

        for j in range(1, ny - 1):
            hp = yi[j+1] - yi[j]
            hm = yi[j] - yi[j-1]
            if hp > 0 and hm > 0:
                dUdy[j] = (Ui[j+1]*hm/(hp*(hp+hm)) +
                           Ui[j]*(hp-hm)/(hp*hm) -
                           Ui[j-1]*hp/(hm*(hp+hm)))
                d2Udy2[j] = 2*(Ui[j+1]/(hp*(hp+hm)) -
                               Ui[j]/(hp*hm) +
                               Ui[j-1]/(hm*(hp+hm)))
                duvdy[j] = (uvi[j+1]*hm/(hp*(hp+hm)) +
                            uvi[j]*(hp-hm)/(hp*hm) -
                            uvi[j-1]*hp/(hm*(hp+hm)))

        # Boundary: forward difference at j=0
        if ny >= 2 and yi[1] > yi[0]:
            dy0 = yi[1] - yi[0]
            dUdy[0] = (Ui[1] - Ui[0]) / dy0
            duvdy[0] = (uvi[1] - uvi[0]) / dy0
        # Backward at j=ny-1
        if ny >= 2 and yi[-1] > yi[-2]:
            dyn = yi[-1] - yi[-2]
            dUdy[-1] = (Ui[-1] - Ui[-2]) / dyn
            duvdy[-1] = (uvi[-1] - uvi[-2]) / dyn

        # dU/dx from neighboring stations (central difference)
        dUdx = np.zeros(ny)
        if n_stations > 1:
            if i > 0 and i < n_stations - 1:
                dx = x[i+1] - x[i-1]
                if abs(dx) > 1e-15:
                    # Interpolate neighbor U onto current y grid
                    n_pts = min(ny, len(U[i-1]), len(U[i+1]))
                    # Use common indices (same n_points)
                    U_prev = np.interp(yi[:n_pts], y[i-1][:len(U[i-1])], U[i-1])
                    U_next = np.interp(yi[:n_pts], y[i+1][:len(U[i+1])], U[i+1])
                    dUdx[:n_pts] = (U_next - U_prev) / dx
            elif i == 0 and n_stations > 1:
                dx = x[1] - x[0]
                if abs(dx) > 1e-15:
                    U_next = np.interp(yi, y[1][:len(U[1])], U[1])
                    dUdx = (U_next - Ui) / dx
            elif i == n_stations - 1 and n_stations > 1:
                dx = x[-1] - x[-2]
                if abs(dx) > 1e-15:
                    U_prev = np.interp(yi, y[-2][:len(U[-2])], U[-2])
                    dUdx = (Ui - U_prev) / dx

        # Budget terms
        convection = Ui * dUdx + Vi * dUdy
        viscous = nu * d2Udy2
        reynolds = -duvdy
        pressure = np.full(ny, dp_dx[i])

        # Near-wall statistics
        conv_nw = convection[nw_idx]
        visc_nw = viscous[nw_idx]
        reyn_nw = reynolds[nw_idx]
        pres_nw = pressure[nw_idx]

        conv_rms_stations[i] = np.sqrt(np.mean(conv_nw**2))
        visc_rms_stations[i] = np.sqrt(np.mean(visc_nw**2))
        reyn_rms_stations[i] = np.sqrt(np.mean(reyn_nw**2))
        pres_rms_stations[i] = np.abs(dp_dx[i])

        diff_rms = visc_rms_stations[i] + reyn_rms_stations[i]
        if diff_rms > 1e-15:
            ratio_stations[i] = conv_rms_stations[i] / diff_rms

    # Summary
    valid = n_nw_pts > 0
    if valid.sum() == 0:
        print(f"  No valid near-wall points!")
        continue

    mean_ratio = np.mean(ratio_stations[valid])
    median_ratio = np.median(ratio_stations[valid])
    max_ratio = np.max(ratio_stations[valid])
    p90_ratio = np.percentile(ratio_stations[valid], 90)

    mean_conv = np.mean(conv_rms_stations[valid])
    mean_visc = np.mean(visc_rms_stations[valid])
    mean_reyn = np.mean(reyn_rms_stations[valid])
    mean_pres = np.mean(pres_rms_stations[valid])

    print(f"  Near-wall ({Y_PLUS_MIN}<y+<{Y_PLUS_MAX}): {valid.sum()} stations, "
          f"avg {np.mean(n_nw_pts[valid]):.0f} pts/station")
    print(f"  Conv RMS: {mean_conv:.4e}  Visc RMS: {mean_visc:.4e}  "
          f"Reyn RMS: {mean_reyn:.4e}  Pres RMS: {mean_pres:.4e}")
    print(f"  |Conv|/|Diff| ratio: mean={mean_ratio:.4f}, median={median_ratio:.4f}, "
          f"p90={p90_ratio:.4f}, max={max_ratio:.4f}")
    if is_sep.sum() > 0:
        sep_ratio = ratio_stations[valid & is_sep]
        att_ratio = ratio_stations[valid & ~is_sep]
        if len(sep_ratio) > 0:
            print(f"  Separated stations: mean ratio={sep_ratio.mean():.4f}")
        if len(att_ratio) > 0:
            print(f"  Attached stations:  mean ratio={att_ratio.mean():.4f}")

    results[fname] = {
        'geometry': geom,
        'n_stations': n_stations,
        'n_separated': int(is_sep.sum()),
        'x': x,
        'tau_w': tau_w,
        'is_separated': is_sep,
        'conv_rms': conv_rms_stations,
        'visc_rms': visc_rms_stations,
        'reyn_rms': reyn_rms_stations,
        'pres_rms': pres_rms_stations,
        'ratio': ratio_stations,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'max_ratio': max_ratio,
        'p90_ratio': p90_ratio,
        'mean_conv': mean_conv,
        'mean_visc': mean_visc,
        'mean_reyn': mean_reyn,
        'mean_pres': mean_pres,
    }

# =====================================================================
# Save results
# =====================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Flatten into arrays for easy plotting
geom_names = []
mean_ratios = []
median_ratios = []
max_ratios = []
p90_ratios = []
mean_convs = []
mean_viscs = []
mean_reyns = []
mean_press = []
n_stations_arr = []
n_sep_arr = []
geom_types = []

for name, r in sorted(results.items()):
    geom_names.append(name)
    mean_ratios.append(r['mean_ratio'])
    median_ratios.append(r['median_ratio'])
    max_ratios.append(r['max_ratio'])
    p90_ratios.append(r['p90_ratio'])
    mean_convs.append(r['mean_conv'])
    mean_viscs.append(r['mean_visc'])
    mean_reyns.append(r['mean_reyn'])
    mean_press.append(r['mean_pres'])
    n_stations_arr.append(r['n_stations'])
    n_sep_arr.append(r['n_separated'])
    geom_types.append(r['geometry'])

save_dict = {
    'geom_names': np.array(geom_names),
    'geom_types': np.array(geom_types),
    'mean_ratio': np.array(mean_ratios),
    'median_ratio': np.array(median_ratios),
    'max_ratio': np.array(max_ratios),
    'p90_ratio': np.array(p90_ratios),
    'mean_conv': np.array(mean_convs),
    'mean_visc': np.array(mean_viscs),
    'mean_reyn': np.array(mean_reyns),
    'mean_pres': np.array(mean_press),
    'n_stations': np.array(n_stations_arr),
    'n_separated': np.array(n_sep_arr),
}

# Also save per-station data for BFS, NASA hump, periodic hills
for key in ['bfs_Re13700', 'nasa_hump_Re936000',
            'periodic_hills_case_1p0']:
    if key in results:
        r = results[key]
        save_dict[f'{key}_x'] = r['x']
        save_dict[f'{key}_ratio'] = r['ratio']
        save_dict[f'{key}_conv_rms'] = r['conv_rms']
        save_dict[f'{key}_visc_rms'] = r['visc_rms']
        save_dict[f'{key}_reyn_rms'] = r['reyn_rms']
        save_dict[f'{key}_is_sep'] = r['is_separated']
        save_dict[f'{key}_tau_w'] = r['tau_w']

outpath = os.path.join(RESULTS_DIR, 'momentum_budget_all_geometries.npz')
np.savez(outpath, **save_dict)
print(f"Saved: {outpath}")

# =====================================================================
# Summary table
# =====================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE: CONVECTION/DIFFUSION RATIO BY GEOMETRY")
print("=" * 80)
print(f"{'Dataset':<40s} {'Geom':>12s} {'N_sta':>5s} {'N_sep':>5s} "
      f"{'Mean':>7s} {'Median':>7s} {'P90':>7s} {'Max':>7s}")
print("-" * 100)
for name, r in sorted(results.items()):
    print(f"{name:<40s} {r['geometry']:>12s} {r['n_stations']:>5d} {r['n_separated']:>5d} "
          f"{r['mean_ratio']:>7.4f} {r['median_ratio']:>7.4f} "
          f"{r['p90_ratio']:>7.4f} {r['max_ratio']:>7.4f}")

# Key result for manuscript
print("\n" + "=" * 80)
print("KEY RESULT FOR MANUSCRIPT")
print("=" * 80)
print("Geometries where ODE works (thin-BL holds):")
for name, r in sorted(results.items()):
    if 'periodic_hills' not in name:
        print(f"  {name}: |Conv|/|Diff| = {r['median_ratio']:.4f} (median)")
print("\nGeometries where ODE fails (thin-BL violated):")
for name, r in sorted(results.items()):
    if 'periodic_hills' in name:
        print(f"  {name}: |Conv|/|Diff| = {r['median_ratio']:.4f} (median)")

print("\nDONE")
