#!/usr/bin/env python3
"""
SUPERSEDED diagnostic for the wall-pinned extraction artifact.

Shows that ε ≪ 1 (domain-wide cancellation) on periodic hills
but ε ~ O(1) at most stations on BFS and NASA hump.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_paths import get_uwf_results_dir
from plotting_utils import setup_style, save_figure

setup_style(font_size=9)
UWF_DIR = get_uwf_results_dir()
QUARANTINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results',
                              '_superseded_wall_pinned')
os.makedirs(QUARANTINE_DIR, exist_ok=True)

Y_IDX = 10  # matching height index, consistent with paper evaluation

# ── Load and compute ε(x) for each geometry ──────────────────────

geometries = [
    {
        'key': 'bfs',
        'file': 'bfs_Re13700_wall_profiles.npz',
        'label': r'BFS ($Re_H = 13\,700$)',
        'xlabel': r'$x/H$',
        'color': '#1f77b4',
    },
    {
        'key': 'nasa_hump',
        'file': 'nasa_hump_Re936000_wall_profiles.npz',
        'label': r'NASA hump ($Re_c = 936\,000$)',
        'xlabel': r'$x/c$',
        'color': '#2ca02c',
    },
    {
        'key': 'periodic_hills',
        'file': 'periodic_hills_case_1p0_wall_profiles.npz',
        'label': r'Periodic hills ($Re_b = 5600$)',
        'xlabel': r'$x/h$',
        'color': '#d62728',
    },
]

results = {}
for geom in geometries:
    wp = np.load(os.path.join(UWF_DIR, geom['file']), allow_pickle=True)
    x = wp['x'].ravel()
    tau_w = wp['tau_w'].ravel()
    dp_dx = wp['dp_dx'].ravel()
    y = wp['y']
    n_sta = len(x)

    # Matching height at each station (y_idx-th point from wall)
    y_m = np.array([y[i, min(Y_IDX, y.shape[1] - 1)] for i in range(n_sta)])

    # ε(x) = |τ_w| / (|dp/dx| · y_m).
    # ε is only defined where there is a pressure gradient to balance; stations
    # with |dp/dx| ≈ 0 are MASKED (set NaN), not floored. Flooring the
    # denominator would turn such a station into a spurious huge ε that biases
    # the median upward. Masking matches the manuscript definition of
    # \tilde{ε} ("median over stations with |dp/dx| > 0").
    denom = np.abs(dp_dx) * np.abs(y_m)
    valid = denom > 1e-30  # truly valid stations
    epsilon = np.full(n_sta, np.nan)
    epsilon[valid] = np.abs(tau_w[valid]) / denom[valid]

    is_sep = tau_w < 0

    # Summary statistics
    finite = np.isfinite(epsilon) & (epsilon > 0)
    frac_below_01 = np.sum(finite & (epsilon < 0.1)) / max(np.sum(finite), 1)
    frac_below_1 = np.sum(finite & (epsilon < 1.0)) / max(np.sum(finite), 1)
    median_eps = np.nanmedian(epsilon[finite]) if np.any(finite) else np.nan

    results[geom['key']] = {
        'x': x, 'epsilon': epsilon, 'tau_w': tau_w, 'dp_dx': dp_dx,
        'y_m': y_m, 'is_sep': is_sep, 'valid': finite,
        'frac_below_01': frac_below_01, 'frac_below_1': frac_below_1,
        'median_eps': median_eps,
    }
    print(f"{geom['key']:20s}: median ε = {median_eps:.2e}, "
          f"frac(ε<0.1) = {frac_below_01:.1%}, "
          f"frac(ε<1) = {frac_below_1:.1%}, "
          f"n_valid = {np.sum(finite)}/{n_sta}")

# ── Create figure: 1×3 layout ────────────────────────────────────

from scipy.ndimage import gaussian_filter1d

def smooth_log(arr, sigma=5):
    """Gaussian-smooth in log space with periodic wrapping (for periodic hills)."""
    if len(arr) < 50:
        return arr
    log_arr = np.log10(np.clip(arr, 1e-15, None))
    pad = 3 * sigma
    padded = np.concatenate([log_arr[-pad:], log_arr, log_arr[:pad]])
    smoothed = gaussian_filter1d(padded, sigma=sigma)
    return 10**smoothed[pad:-pad]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

for ax, geom in zip(axes, geometries):
    r = results[geom['key']]
    x = r['x']
    eps = r['epsilon']
    is_sep = r['is_sep']
    valid = r['valid']

    # Shade separation zones
    for i in range(len(x) - 1):
        if is_sep[i]:
            ax.axvspan(x[i], x[i + 1], alpha=0.10, color='gray', lw=0)

    # Smooth periodic hills (512 stations); leave BFS/hump raw
    eps_clipped = np.clip(eps, 1e-6, 1e6)
    if geom['key'] == 'periodic_hills':
        eps_plot = smooth_log(eps_clipped)
        ax.semilogy(x, eps_plot, '-', color=geom['color'], lw=1.0,
                    rasterized=True)
    else:
        ax.semilogy(x, eps_clipped, '-', color=geom['color'], lw=1.0)

    # Reference lines
    ax.axhline(y=1.0, color='k', ls='--', lw=0.6, alpha=0.5)
    ax.axhline(y=0.1, color='k', ls=':', lw=0.5, alpha=0.3)

    ax.text(0.97, 0.03, r'$\varepsilon = 1$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, color='0.4')

    ax.set_xlabel(geom['xlabel'])
    ax.set_title(geom['label'], fontsize=8, loc='left')
    ax.set_ylim(1e-5, 1e5)

    # Annotation: median ε
    med = r['median_eps']
    if geom['key'] == 'periodic_hills':
        ax.text(0.5, 0.25, r'median $\varepsilon = %.0e$' % med
                + '\n' + r'\textbf{domain-wide cancellation}',
                transform=ax.transAxes, fontsize=6.5, ha='center',
                color='#b00000',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#b00000', alpha=0.85))
    else:
        ax.text(0.5, 0.25, r'median $\varepsilon = %.1f$' % med
                + '\n' + r'\textbf{dominant balance}',
                transform=ax.transAxes, fontsize=6.5, ha='center',
                color='#2ca02c',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#2ca02c', alpha=0.85))

axes[0].set_ylabel(r'$\varepsilon(x)$')

# Panel labels
for i, ax in enumerate(axes):
    ax.text(-0.12, 1.05, f'({chr(97 + i)})', transform=ax.transAxes,
            fontsize=10, fontweight='bold')

fig.suptitle('SUPERSEDED WALL-PINNED EXTRACTION — DO NOT USE',
             color='crimson', fontsize=9)
save_figure(fig, 'cancellation_parameter_wall_pinned_superseded',
            fig_dir=QUARANTINE_DIR)

# ── Save numerical results ───────────────────────────────────────

np.savez(
    os.path.join(QUARANTINE_DIR,
                 'cancellation_parameter_wall_pinned_superseded.npz'),
    **{f'{g["key"]}_x': results[g['key']]['x'] for g in geometries},
    **{f'{g["key"]}_epsilon': results[g['key']]['epsilon'] for g in geometries},
    **{f'{g["key"]}_is_sep': results[g['key']]['is_sep'] for g in geometries},
    **{f'{g["key"]}_median_eps': results[g['key']]['median_eps'] for g in geometries},
    **{f'{g["key"]}_frac_below_01': results[g['key']]['frac_below_01'] for g in geometries},
    y_idx=Y_IDX,
    protocol=np.array('legacy_wall_pinned_historical_only'),
    manuscript_status=np.array('superseded'),
    corrected_artifact=np.array('cancellation_parameter_corrected.npz'),
)
print('Results saved.')
