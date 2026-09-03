#!/usr/bin/env python3
"""
Extended cancellation parameter figure: original 3 geometries + new geometries.

Top row (a-c): original BFS, NASA hump, periodic hills
Bottom row (d-f): curved BFS, conv-div channel, Gaussian bump Re=2M

Also generates a summary scatter plot: ε_median vs R²(τ_w) for all geometries.
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
NEW_DIR = os.path.join(os.path.dirname(__file__), '..', 'new_data_download')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

Y_IDX = 10


def compute_epsilon(wp):
    """Compute ε(x) = |τ_w| / (|dp/dx| · y_m) for a wall-profiles dataset."""
    x = wp['x'].ravel()
    tau_w = wp['tau_w'].ravel()
    dp_dx = wp['dp_dx'].ravel()
    y = wp['y']
    n_sta = len(x)

    y_m = np.array([y[i, min(Y_IDX, y.shape[1] - 1)] for i in range(n_sta)])
    denom = np.abs(dp_dx) * np.abs(y_m)
    # ε(x) is only defined where there is a pressure gradient to balance.
    # Stations with |dp/dx| ~ 0 are masked out (NOT floored): flooring the
    # denominator turns such a station into a spurious huge ε that biases the
    # median upward. Masking matches the manuscript definition of \tilde{ε}
    # ("median over stations with |dp/dx| > 0") and the evaluation scripts.
    valid_dp = denom > 1e-30
    epsilon = np.full(n_sta, np.nan)
    epsilon[valid_dp] = np.abs(tau_w[valid_dp]) / denom[valid_dp]
    is_sep = tau_w < 0

    finite = np.isfinite(epsilon) & (epsilon > 0)
    median_eps = np.nanmedian(epsilon[finite]) if np.any(finite) else np.nan
    frac_below_01 = np.sum(finite & (epsilon < 0.1)) / max(np.sum(finite), 1)

    return {
        'x': x, 'epsilon': epsilon, 'is_sep': is_sep, 'valid': finite,
        'median_eps': median_eps, 'frac_below_01': frac_below_01,
    }


from scipy.ndimage import gaussian_filter1d

def smooth_log(arr, sigma=5):
    if len(arr) < 50:
        return arr
    log_arr = np.log10(np.clip(arr, 1e-15, None))
    pad = 3 * sigma
    padded = np.concatenate([log_arr[-pad:], log_arr, log_arr[:pad]])
    smoothed = gaussian_filter1d(padded, sigma=sigma)
    return 10**smoothed[pad:-pad]


# ── Load all geometries ────────────────────────────────────────────

# Original 3
orig_geometries = [
    {'key': 'bfs', 'file': os.path.join(str(UWF_DIR), 'bfs_Re13700_wall_profiles.npz'),
     'label': r'BFS ($Re_H\!=\!13\,700$)', 'xlabel': r'$x/H$', 'color': '#1f77b4', 'smooth': False},
    {'key': 'nasa_hump', 'file': os.path.join(str(UWF_DIR), 'nasa_hump_Re936000_wall_profiles.npz'),
     'label': r'NASA hump ($Re_c\!=\!936\,000$)', 'xlabel': r'$x/c$', 'color': '#2ca02c', 'smooth': False},
    {'key': 'periodic_hills', 'file': os.path.join(str(RESULTS_DIR), 'periodic_hills_case_1p0_wall_profiles_corrected.npz'),
     'label': r'Periodic hills ($Re_b\!=\!5600$)', 'xlabel': r'$x/h$', 'color': '#d62728', 'smooth': True},
]

# New geometries (best candidates for reviewer response)
new_geometries = [
    {'key': 'curved_bfs', 'file': os.path.join(NEW_DIR, 'geometry_driven', 'curved_bfs_Re13700_DNS_wall_profiles.npz'),
     'label': r'Curved BFS ($Re_H\!=\!13\,700$)', 'xlabel': r'$x/H$', 'color': '#8c564b', 'smooth': False},
    {'key': 'conv_div', 'file': os.path.join(NEW_DIR, 'conv_div_channel_Re12600_wall_profiles.npz'),
     'label': r'Conv-div channel ($Re\!=\!12\,600$)', 'xlabel': r'$x/h$', 'color': '#e377c2', 'smooth': False},
    {'key': 'gauss_bump', 'file': os.path.join(NEW_DIR, 'gaussian_bump_Re2M_wall_profiles.npz'),
     'label': r'Gaussian bump ($Re_L\!=\!2\!\times\!10^6$)', 'xlabel': r'$x/L$', 'color': '#17becf', 'smooth': False},
]

all_geom = orig_geometries + new_geometries
results = {}
for geom in all_geom:
    wp = np.load(geom['file'], allow_pickle=True)
    results[geom['key']] = compute_epsilon(wp)
    r = results[geom['key']]
    print(f"{geom['key']:20s}: median ε = {r['median_eps']:.2e}, "
          f"frac(ε<0.1) = {r['frac_below_01']:.1%}, n = {len(r['x'])}")


# ── Figure 1: 2×3 panel layout ─────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))

for row_idx, geom_list in enumerate([orig_geometries, new_geometries]):
    for col_idx, geom in enumerate(geom_list):
        ax = axes[row_idx, col_idx]
        r = results[geom['key']]
        x = r['x']
        eps = r['epsilon']
        is_sep = r['is_sep']

        # Shade separation zones
        for i in range(len(x) - 1):
            if is_sep[i]:
                ax.axvspan(x[i], x[i + 1], alpha=0.10, color='gray', lw=0)

        # Plot ε(x)
        eps_clipped = np.clip(eps, 1e-6, 1e6)
        if geom.get('smooth', False):
            eps_plot = smooth_log(eps_clipped)
            ax.semilogy(x, eps_plot, '-', color=geom['color'], lw=1.0, rasterized=True)
        else:
            ax.semilogy(x, eps_clipped, '-', color=geom['color'], lw=1.0)

        # Reference lines
        ax.axhline(y=1.0, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.axhline(y=0.1, color='k', ls=':', lw=0.5, alpha=0.3)

        ax.set_xlabel(geom['xlabel'])
        ax.set_title(geom['label'], fontsize=7.5, loc='left')
        ax.set_ylim(1e-5, 1e5)

        # Annotation
        med = r['median_eps']
        is_catastrophe = r['frac_below_01'] > 0.5

        if is_catastrophe:
            med_str = (r'$\tilde{\varepsilon} = %.2f$' % med if med >= 0.01
                       else r'$\tilde{\varepsilon} = %.0e$' % med)
            ax.text(0.5, 0.25,
                    med_str + '\n'
                    + r'\textbf{cancellation}',
                    transform=ax.transAxes, fontsize=6, ha='center',
                    color='#b00000',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#b00000', alpha=0.85))
        else:
            if med > 100:
                med_str = r'$\tilde{\varepsilon} = %.0f$' % med
            elif med > 10:
                med_str = r'$\tilde{\varepsilon} = %.0f$' % med
            else:
                med_str = r'$\tilde{\varepsilon} = %.1f$' % med
            ax.text(0.5, 0.25,
                    med_str + '\n' + r'\textbf{dominant balance}',
                    transform=ax.transAxes, fontsize=6, ha='center',
                    color='#006600',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#006600', alpha=0.85))

        # Panel label
        panel_idx = row_idx * 3 + col_idx
        ax.text(-0.15, 1.05, f'({chr(97 + panel_idx)})',
                transform=ax.transAxes, fontsize=10, fontweight='bold')

# y-axis labels
axes[0, 0].set_ylabel(r'$\varepsilon(x)$')
axes[1, 0].set_ylabel(r'$\varepsilon(x)$')

fig.tight_layout(h_pad=1.5, w_pad=0.8)
save_figure(fig, 'cancellation_parameter_extended', fig_dir=FIG_DIR)


# ── Figure 2: Scatter — ε_median vs R² across ALL datasets ─────────

# Load ODE evaluation results for the new datasets
new_results_file = os.path.join(NEW_DIR, 'all_new_dataset_results.npz')
nd = np.load(new_results_file, allow_pickle=True)
dataset_names = list(nd['dataset_names'])

fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 3.5))

# Geometry type classification for marker/color
geo_classes = {
    'periodic_hills': {'marker': 's', 'color': '#d62728', 'label': 'Periodic hills'},
    'channel': {'marker': 'D', 'color': '#1f77b4', 'label': 'Channel/pipe'},
    'bump': {'marker': '^', 'color': '#17becf', 'label': 'Bump/hump'},
    'bfs': {'marker': 'o', 'color': '#8c564b', 'label': 'Backward-facing step'},
    'sep_bubble': {'marker': 'v', 'color': '#ff7f0e', 'label': 'Separation bubble'},
    'other': {'marker': 'p', 'color': '#9467bd', 'label': 'Other'},
}

def classify_geometry(name):
    nl = name.lower()
    if 'pehill' in nl or 'periodic_hills' in nl or 'xiao' in nl or 'krank' in nl:
        return 'periodic_hills'
    elif 'bfs' in nl or 'curved_bfs' in nl:
        return 'bfs'
    elif 'bump' in nl or 'hump' in nl or 'gaussian' in nl:
        return 'bump'
    elif 'channel' in nl or 'conv_div' in nl or 'diffuser' in nl or 'pipe' in nl:
        return 'channel'
    elif 'sep' in nl or 'bubble' in nl or 'jaxa' in nl or 'kawai' in nl or 'swept' in nl:
        return 'sep_bubble'
    return 'other'

plotted_classes = set()

for name in dataset_names:
    r2_key = f'{name}_r2'
    eps_key = f'{name}_eps_median'
    fsep_key = f'{name}_f_sep'

    if r2_key not in nd.files or eps_key not in nd.files:
        continue

    r2 = float(nd[r2_key])
    eps_med = float(nd[eps_key])
    f_sep = float(nd[fsep_key])

    if np.isnan(eps_med) or eps_med <= 0:
        continue

    gc = classify_geometry(name)
    style = geo_classes[gc]

    label = style['label'] if gc not in plotted_classes else None
    plotted_classes.add(gc)

    # Size proportional to f_sep
    sz = max(30, f_sep * 300)

    ax2.scatter(eps_med, r2, s=sz, marker=style['marker'], c=style['color'],
                edgecolors='k', linewidths=0.3, label=label, zorder=5, alpha=0.85)

# Reference lines
ax2.axhline(y=0, color='k', ls=':', lw=0.5, alpha=0.3)
ax2.axvline(x=0.1, color='#b00000', ls='--', lw=0.8, alpha=0.5, label=r'$\varepsilon = 0.1$')

ax2.set_xscale('log')
ax2.set_xlabel(r'Median cancellation parameter $\tilde{\varepsilon}$')
ax2.set_ylabel(r'$R^2(\tau_w)$')
ax2.set_xlim(1e-4, 1e4)
ax2.set_ylim(-1.5, 1.05)
ax2.legend(fontsize=6.5, loc='lower right', framealpha=0.9)

# Shade the catastrophe zone
ax2.axvspan(1e-4, 0.1, alpha=0.06, color='red', zorder=0)
ax2.text(0.02, 0.95, r'\textbf{cancellation zone}', transform=ax2.transAxes,
         fontsize=6.5, color='#b00000', va='top')
ax2.text(0.55, 0.95, r'\textbf{dominant balance}', transform=ax2.transAxes,
         fontsize=6.5, color='#006600', va='top')

fig2.tight_layout()
save_figure(fig2, 'epsilon_vs_r2_scatter', fig_dir=FIG_DIR)

# ── Save extended cancellation results ──────────────────────────────

save_dict = {}
for geom in all_geom:
    k = geom['key']
    r = results[k]
    save_dict[f'{k}_x'] = r['x']
    save_dict[f'{k}_epsilon'] = r['epsilon']
    save_dict[f'{k}_is_sep'] = r['is_sep']
    save_dict[f'{k}_median_eps'] = r['median_eps']
    save_dict[f'{k}_frac_below_01'] = r['frac_below_01']

save_dict['y_idx'] = Y_IDX

np.savez(os.path.join(os.path.dirname(__file__), '..', 'results',
                       'cancellation_parameter_extended.npz'), **save_dict)

print("\nExtended results saved.")
print("\nDone — figures saved to manuscript/figures/")
