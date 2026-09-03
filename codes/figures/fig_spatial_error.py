#!/usr/bin/env python3
"""
Figure: Spatial distribution of station-level relative error |Δτ_w|/|τ_w|.

Compares four representative geometries to show that errors are localised
near separation/reattachment in successful cases but domain-wide on
periodic hills.

Two panels:
  (a) Streamwise |Δτ_w|/|τ_w| for APG TBL, BFS, JAXA Re600, periodic hills
  (b) CDF of station-level error for the same four
"""

import sys, os
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_paths import get_uwf_results_dir
from plotting_utils import setup_style, save_figure
import matplotlib.pyplot as plt

setup_style(font_size=9, use_tex=True)
UWF_DIR = get_uwf_results_dir()
NEW_DIR = os.path.join(os.path.dirname(__file__), '..', 'new_data_download')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

# ── ODE solver ────────────────────────────────────────────────────
KAPPA, A_PLUS, N_GRID = 0.41, 26.0, 500

def solve_ode_for_U(tau_w, y_match, dp_dx, nu, n_grid=N_GRID):
    u_tau = np.sqrt(max(abs(tau_w), 1e-30))
    eta = np.linspace(0, 1, n_grid)
    y = y_match * eta ** 1.5
    y_plus = y * u_tau / nu
    D = 1.0 - np.exp(-y_plus / A_PLUS)
    l_m = KAPPA * y * D
    l_m2 = l_m * l_m
    F = tau_w + dp_dx * y
    absF, signF = np.abs(F), np.sign(F)
    disc = nu ** 2 + 4.0 * l_m2 * absF
    viscous = l_m2 < 1e-30
    l_m2_safe = np.where(viscous, 1.0, l_m2)
    absS = (-nu + np.sqrt(disc)) / (2.0 * l_m2_safe)
    absS = np.where(viscous, absF / nu, absS)
    S = signF * absS
    dy = np.diff(y)
    return np.sum(0.5 * (S[:-1] + S[1:]) * dy)


def predict_tau_w(U_match, y_match, dp_dx, nu):
    if abs(U_match) < 1e-15 and abs(dp_dx) < 1e-15:
        return 0.0
    def residual(tw):
        return solve_ode_for_U(tw, y_match, dp_dx, nu) - U_match
    visc_est = abs(nu * U_match / y_match) if y_match > 0 else 1e-6
    dp_est = abs(dp_dx * y_match)
    scale = max(visc_est, dp_est, 1e-8) * 5.0
    lo, hi = -scale, scale
    try:
        f_lo, f_hi = residual(lo), residual(hi)
    except Exception:
        return np.nan
    for _ in range(12):
        if f_lo * f_hi < 0:
            break
        lo *= 3.0; hi *= 3.0
        try:
            f_lo, f_hi = residual(lo), residual(hi)
        except Exception:
            return np.nan
    if f_lo * f_hi >= 0:
        taus = np.linspace(lo, hi, 30)
        fs = np.array([residual(t) for t in taus])
        sc = np.where(np.diff(np.sign(fs)))[0]
        if len(sc) > 0:
            lo, hi = taus[sc[0]], taus[sc[0] + 1]
        else:
            return np.nan
    try:
        return brentq(residual, lo, hi, xtol=1e-12, rtol=1e-8, maxiter=100)
    except Exception:
        return np.nan


Y_IDX = 10

def evaluate_geometry(filepath, label_key):
    """Evaluate ODE on a dataset and return station-level errors."""
    d = np.load(filepath, allow_pickle=True)
    x = d['x'].ravel()
    y = d['y']
    U = d['U']
    tau_w_dns = d['tau_w'].ravel()
    dp_dx = d['dp_dx'].ravel()
    nu_arr = d['nu'].ravel()
    is_sep = d['is_separated'].ravel()
    n = len(x)

    tau_pred = np.full(n, np.nan)
    for i in range(n):
        yi = y[i] if y.ndim == 2 else y
        Ui = U[i] if U.ndim == 2 else U
        if Y_IDX >= len(yi):
            continue
        y_m, U_m = yi[Y_IDX], Ui[Y_IDX]
        if y_m <= 0 or np.isnan(U_m):
            continue
        tau_pred[i] = predict_tau_w(U_m, y_m, float(dp_dx[i]), float(nu_arr[i]))

    valid = ~np.isnan(tau_pred) & (np.abs(tau_w_dns) > 1e-30)
    rel_err = np.full(n, np.nan)
    rel_err[valid] = np.abs(tau_pred[valid] - tau_w_dns[valid]) / np.abs(tau_w_dns[valid])

    print(f"  {label_key}: {valid.sum()}/{n} valid, "
          f"median rel.err = {np.nanmedian(rel_err[valid]):.3f}, "
          f"n_sep = {is_sep.sum()}")

    return {'x': x, 'rel_err': rel_err, 'is_sep': is_sep, 'valid': valid,
            'tau_dns': tau_w_dns, 'tau_pred': tau_pred}


# ── Evaluate geometries ──────────────────────────────────────────
print("Evaluating ODE on selected geometries...")

geometries = [
    {
        'key': 'apg_tbl',
        'file': os.path.join(UWF_DIR, 'apg_tbl_kth_m18n_wall_profiles.npz'),
        'label': r'APG TBL (m18n)',
        'xlabel': r'station',
        'color': '#d62728',  # red
        'marker': 'v',
        'use_index': True,  # plot vs station index (no meaningful x-coord)
    },
    {
        'key': 'bfs',
        'file': os.path.join(UWF_DIR, 'bfs_Re13700_wall_profiles.npz'),
        'label': r'BFS ($Re_H = 13\,700$)',
        'xlabel': r'$x/H$',
        'color': '#1f77b4',  # blue
        'marker': 's',
        'use_index': False,
    },
    {
        'key': 'jaxa_Re600',
        'file': os.path.join(NEW_DIR, 'jaxa_sep_bubble_Re600_wall_profiles.npz'),
        'label': r'JAXA sep.\ bubble ($Re_{\theta_0} = 600$)',
        'xlabel': r'$x/\theta_0$',
        'color': '#ff7f0e',  # orange
        'marker': '^',
        'use_index': False,
    },
    {
        'key': 'periodic_hills',
        'file': os.path.join(RESULTS_DIR, 'periodic_hills_case_1p0_wall_profiles_corrected.npz'),
        'label': r'Periodic hills ($Re_b = 5600$)',
        'xlabel': r'$x/h$',
        'color': '#d62728',  # red
        'marker': 'p',
        'use_index': False,
    },
]

results = {}
for geom in geometries:
    results[geom['key']] = evaluate_geometry(geom['file'], geom['key'])

# ── Figure: 2-panel layout ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Panel (a): Streamwise relative error for BFS, JAXA, periodic hills
# (skip APG TBL — not streamwise-meaningful, save for CDF panel)
streamwise_geoms = ['bfs', 'jaxa_Re600', 'periodic_hills']
colors_map = {g['key']: g['color'] for g in geometries}
label_map = {g['key']: g['label'] for g in geometries}

from scipy.ndimage import gaussian_filter1d

for gkey in streamwise_geoms:
    r = results[gkey]
    x = r['x']
    rel_err = r['rel_err']
    valid = r['valid']
    is_sep = r['is_sep']
    color = colors_map[gkey]

    # Clip for plotting
    rel_err_plot = np.clip(rel_err, 1e-4, 1e6)

    if gkey == 'periodic_hills':
        # Smooth in log space for dense data
        log_err = np.log10(np.where(valid, rel_err_plot, np.nan))
        finite = np.isfinite(log_err)
        if finite.sum() > 50:
            log_smooth = log_err.copy()
            log_smooth[finite] = gaussian_filter1d(log_err[finite], sigma=5)
            rel_err_plot = 10 ** log_smooth
        ax1.semilogy(x[valid], rel_err_plot[valid], '-', color=color, lw=0.8,
                     alpha=0.7, label=label_map[gkey], rasterized=True)
        # Shade separation zones
        for i in range(len(x) - 1):
            if is_sep[i]:
                ax1.axvspan(x[i], x[i + 1], alpha=0.05, color='gray', lw=0)
    else:
        ax1.semilogy(x[valid], rel_err_plot[valid], 'o-', color=color,
                     lw=1.0, ms=3, mfc='none', mew=0.6,
                     label=label_map[gkey])
        # Shade separation
        for i in range(len(x) - 1):
            if is_sep[i]:
                ax1.axvspan(x[i], x[i + 1], alpha=0.05, color='gray', lw=0)

# Reference lines
ax1.axhline(y=1.0, color='k', ls='--', lw=0.5, alpha=0.3)
ax1.set_xlabel(r'$x$ (geometry-specific normalisation)')
ax1.set_ylabel(r'$|\Delta\tau_w| / |\tau_w|$')
ax1.set_ylim(1e-4, 1e5)
ax1.legend(fontsize=6.5, loc='upper left')
ax1.text(-0.14, 1.05, r'(a)', transform=ax1.transAxes,
         fontsize=10, fontweight='bold')

# Panel (b): CDF of station-level error
cdf_geoms = ['apg_tbl', 'bfs', 'jaxa_Re600', 'periodic_hills']
cdf_colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#d62728']
cdf_labels = [
    r'APG TBL',
    r'BFS',
    r'JAXA sep.\ bubble',
    r'Periodic hills',
]
cdf_styles = ['-', '--', '-.', '-']

for gkey, color, label, ls in zip(cdf_geoms, cdf_colors, cdf_labels, cdf_styles):
    r = results[gkey]
    rel_err = r['rel_err']
    valid = r['valid']
    err_valid = rel_err[valid]
    err_valid = err_valid[np.isfinite(err_valid)]
    if len(err_valid) == 0:
        continue
    sorted_err = np.sort(err_valid)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax2.semilogx(sorted_err, cdf, ls, color=color, lw=1.2, label=label)

ax2.axvline(x=1.0, color='k', ls='--', lw=0.5, alpha=0.3)
ax2.set_xlabel(r'$|\Delta\tau_w| / |\tau_w|$')
ax2.set_ylabel('CDF')
ax2.set_xlim(1e-4, 1e5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=6.5, loc='upper left')
ax2.text(-0.14, 1.05, r'(b)', transform=ax2.transAxes,
         fontsize=10, fontweight='bold')

# Annotation in panel (b)
ax2.annotate(r'$O(1)$ error boundary', xy=(1.0, 0.5),
             xytext=(5, 0.55), fontsize=6, color='0.4',
             arrowprops=dict(arrowstyle='->', color='0.4', lw=0.6))

save_figure(fig, 'fig_spatial_error', fig_dir=FIG_DIR)
print(f'\nSaved to {FIG_DIR}/fig_spatial_error.pdf')
