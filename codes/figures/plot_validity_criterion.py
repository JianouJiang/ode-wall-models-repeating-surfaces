#!/usr/bin/env python3
"""
Figures for Steps 2 and 3 of the paper:

Figure 1 (Step 2): Streamwise budget comparison — BFS, NASA hump, separation bubble
  vs periodic hills. Shows convective terms are small where ODE works.

Figure 2 (Step 3): Validity criterion — 2-panel figure:
  (a) Primary metric: fraction of separated stations vs ODE R²
  (b) Secondary metric: composite convective intensity vs ODE R²

Figure 3: Bar chart — budget terms by geometry type (Step 2 summary)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from plotting_utils import setup_style, save_figure

setup_style(font_size=9)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

budget = np.load(os.path.join(RESULTS_DIR, 'momentum_budget_all_geometries.npz'),
                 allow_pickle=True)
criterion = np.load(os.path.join(RESULTS_DIR, 'criterion_evaluation.npz'),
                    allow_pickle=True)
budget_geom_names = list(budget['geom_names'])
budget_geom_types = list(budget['geom_types'])

# =====================================================================
# Figure 1: Streamwise budget comparison (Step 2)
# =====================================================================
fig1, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

# Helper to plot streamwise budget for a geometry
def _smooth(arr, sigma=5):
    """Gaussian smooth with periodic wrapping (only for long arrays)."""
    if len(arr) < 50:
        return arr  # BFS/hump have ~50 stations, no smoothing needed
    from scipy.ndimage import gaussian_filter1d
    pad = 3 * sigma
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    return gaussian_filter1d(padded, sigma=sigma)[pad:-pad]

def plot_budget_streamwise(ax, key, title, xlabel=r'$x/L$', show_legend=False):
    """Plot near-wall budget term RMS vs x for a given geometry."""
    xb = budget[f'{key}_x']
    conv = _smooth(budget[f'{key}_conv_rms'])
    visc = _smooth(budget[f'{key}_visc_rms'])
    reyn = _smooth(budget[f'{key}_reyn_rms'])
    tau_w = budget[f'{key}_tau_w']
    sep = tau_w < 0

    valid = (conv > 0) | (visc > 0)

    ax.semilogy(xb[valid], conv[valid], '-', color='#d62728', lw=1.5, label='Convection')
    ax.semilogy(xb[valid], reyn[valid], '--', color='#2ca02c', lw=1.3, label='Reynolds stress')
    ax.semilogy(xb[valid], visc[valid], '-.', color='#ff7f0e', lw=1.3, label='Viscous')

    # Shade separation
    if sep.any():
        x_sep = xb[sep]
        if len(x_sep) > 0:
            ax.axvspan(x_sep.min(), x_sep.max(), alpha=0.1, color='gray')

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_ylabel('Near-wall RMS')
    ax.set_xlabel(xlabel)
    if show_legend:
        ax.legend(fontsize=6.5, loc='upper right')

# BFS
plot_budget_streamwise(axes[0, 0], 'bfs_Re13700',
    r'(a) BFS, $Re_H=13\,700$', xlabel=r'$x/H$', show_legend=True)
axes[0, 0].text(2, 2e-4, r'Convection \textbf{localised} ($f_\mathrm{sep}=0.24$)'
    + '\n' + r'$\Rightarrow$ ODE works ($R^2=0.89$)',
    fontsize=6.5, va='bottom', ha='center', color='#2ca02c',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2ca02c', alpha=0.85))

# NASA hump
plot_budget_streamwise(axes[0, 1], 'nasa_hump_Re936000',
    r'(b) NASA hump, $Re_c=936\,000$', xlabel=r'$x/c$')
axes[0, 1].text(0.5, 2e-2, r'Convection \textbf{localised} ($f_\mathrm{sep}=0.14$)'
    + '\n' + r'$\Rightarrow$ ODE works ($R^2=0.96$)',
    fontsize=6.5, va='bottom', ha='center', color='#2ca02c',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2ca02c', alpha=0.85))

# Periodic hills
plot_budget_streamwise(axes[1, 0], 'periodic_hills_case_1p0',
    r'(c) Periodic hills, $Re_b=5\,600$', xlabel=r'$x/h$')
axes[1, 0].text(4.5, 1e-6, r'Convection \textbf{domain-wide} ($f_\mathrm{sep}=0.40$)'
    + '\n' + r'$\Rightarrow$ ODE fails ($R^2 \approx -48$)',
    fontsize=6.5, va='bottom', ha='center', color='#b00000',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#b00000', alpha=0.85))

# Panel (d): Cancellation parameter ε CDF — the correct metric.
# Use the hill-surface-aware corrected ε for periodic hills (the legacy
# cancellation_parameter.npz pins the wall at y=0, giving a spurious ε~1e-4);
# BFS and NASA hump are carried over unchanged (their walls sit at the grid
# floor). See analysis/build_corrected_pehill_profiles.py and
# figures/plot_cancellation_parameter_corrected.py.
eps_data = np.load(os.path.join(RESULTS_DIR, 'cancellation_parameter_corrected.npz'),
                   allow_pickle=True)
ax = axes[1, 1]

thresholds = np.logspace(-6, 5, 300)
for key, label, color in [
    ('bfs', 'BFS', '#1f77b4'),
    ('nasa_hump', 'NASA hump', '#ff7f0e'),
    ('periodic_hills', 'Periodic hills', '#d62728'),
]:
    eps = eps_data[f'{key}_epsilon']
    valid = np.isfinite(eps) & (eps > 0)
    eps_valid = eps[valid]
    frac = np.array([np.mean(eps_valid < t) for t in thresholds])
    ax.semilogx(thresholds, frac, '-', color=color, lw=1.3, label=label)

ax.axvline(x=1.0, color='k', ls='--', lw=0.6, alpha=0.5)
ax.set_xlabel(r'Threshold $\varepsilon_0$')
ax.set_ylabel(r'Fraction with $\varepsilon < \varepsilon_0$')
ax.set_xlim(1e-5, 1e5)
ax.set_ylim(0, 1.05)
ax.set_title(r'(d) Cancellation parameter $\varepsilon$ CDF', fontsize=8, pad=3)
ax.legend(fontsize=6.5, loc='lower left')
ax.text(3e-3, 0.8, r'\textbf{ODE fails}' + '\n' + r'$\varepsilon \ll 1$',
        fontsize=6.5, ha='center', color='#b00000',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#b00000', alpha=0.85))
ax.text(1e3, 0.8, r'\textbf{ODE works}' + '\n' + r'$\varepsilon \gg 1$',
        fontsize=6.5, ha='center', color='#2ca02c',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#2ca02c', alpha=0.85))

fig1.tight_layout()
save_figure(fig1, 'budget_comparison_geometries', fig_dir=FIG_DIR)
print("Figure 1 (budget comparison) saved.")

# =====================================================================
# Figure 2: Validity criterion — single panel with threshold
# =====================================================================

# Validity figure uses the 9 multi-station datasets from criterion evaluation.
# (22 single-station equilibrium datasets omitted: R² is degenerate for n=1.)
geom_names = list(criterion['geom_names'])
geom_types = list(criterion['geom_types'])
frac_sep = criterion['frac_sep']
conv_intensity = criterion['conv_intensity']
ode_r2 = criterion['ode_r2']
fsep_threshold = float(criterion['fsep_threshold'])

# Geometry type markers and colors
GEOM_STYLE = {
    'channel': ('o', '#4C78A8', 'Channel'),
    'pipe': ('s', '#72B7B2', 'Pipe'),
    'couette': ('D', '#B279A2', 'Couette'),
    'tbl': ('^', '#54A24B', 'TBL (ZPG)'),
    'apg_tbl': ('v', '#E45756', 'TBL (APG)'),
    'bfs': ('P', '#17BECF', 'BFS'),
    'nasa_hump': ('h', '#F58518', 'NASA hump'),
    'separation_bubble': ('X', '#BCBD22', 'Sep. bubble'),
    'periodic_hills': ('H', '#D62728', 'Periodic hills'),
    # New geometry types for extended validation
    'gaussian_bump': ('p', '#9467BD', 'Gaussian bump'),
    'conv_div_channel': ('d', '#8C564B', 'Conv-div channel'),
    'swept_separation': ('*', '#E377C2', 'Swept sep. bubble'),
    'jaxa_separation': ('+', '#7F7F7F', r'JAXA sep. bubble'),
}

# Separated-flow geometries get larger markers
SEPARATED_TYPES = {'bfs', 'nasa_hump', 'separation_bubble', 'periodic_hills',
                   'gaussian_bump', 'conv_div_channel', 'swept_separation',
                   'jaxa_separation'}

# --- New datasets from extended validation ---
# Trustworthy: swept sep, Gaussian bump Re=2M, conv-div channel
# Preliminary (dagger): JAXA sep bubbles (Bernoulli dp/dx estimate)
NEW_DATASETS = [
    # (name, geom_type, f_sep, R², is_preliminary)
    ('swept_sep_caseC35',        'swept_separation',  0.100, +0.9974, False),
    ('conv_div_channel_Re12600', 'conv_div_channel',  0.098, +0.9342, False),
    ('gaussian_bump_Re2M',       'gaussian_bump',     0.131, +0.9747, False),
    ('jaxa_sep_Re300',           'jaxa_separation',   0.340, +0.9973, False),
    ('jaxa_sep_Re600',           'jaxa_separation',   0.340, +0.9964, False),
    ('jaxa_sep_Re900',           'jaxa_separation',   0.360, +0.9962, False),
]

fig2, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

# Shaded valid/invalid regions
ax.axvspan(-0.02, fsep_threshold, alpha=0.07, color='#2ca02c', zorder=0)
ax.axvspan(fsep_threshold, 0.55, alpha=0.07, color='#d62728', zorder=0)
ax.axvline(fsep_threshold, color='0.25', ls='--', lw=1.0, zorder=1)

# Region labels — positioned to avoid data
ax.text(0.14, 0.55, 'ODE succeeds\non present sample', ha='center',
        fontsize=7, color='#2E7D32', alpha=0.8)
ax.text(0.46, 0.55, 'ODE fails on\nperiodic hills', ha='center',
        fontsize=7, color='#C62828', alpha=0.8)

# Threshold label — positioned clearly
ax.text(fsep_threshold + 0.012, 1.05, r'$f_{\mathrm{sep}} = 0.3$',
        fontsize=7, color='0.3', va='top')

# Plot ORIGINAL data points
plotted_labels = set()
for i, name in enumerate(geom_names):
    gtype = geom_types[i]
    r2_val = ode_r2[i]
    r2_display = max(r2_val, -0.55)
    fs = frac_sep[i]
    marker, color, label = GEOM_STYLE.get(gtype, ('o', '0.5', gtype))
    show_label = label not in plotted_labels
    plotted_labels.add(label)

    ms = 10 if gtype in SEPARATED_TYPES else 7
    zord = 8 if gtype == 'periodic_hills' else 5

    ax.plot(fs, r2_display, marker, color=color, ms=ms,
            label=label if show_label else None, mec='k', mew=0.35,
            alpha=0.9, zorder=zord)

# Plot NEW data points
for dname, gtype, fs, r2_val, is_prelim in NEW_DATASETS:
    r2_display = max(r2_val, -0.55)
    marker, color, label = GEOM_STYLE.get(gtype, ('o', '0.5', gtype))
    show_label = label not in plotted_labels
    plotted_labels.add(label)

    ms = 9
    alpha = 0.9
    mew = 0.35
    mec = 'k'
    zord = 6

    ax.plot(fs, r2_display, marker, color=color, ms=ms,
            label=label if show_label else None, mec=mec, mew=mew,
            alpha=alpha, zorder=zord)

# Annotate periodic hills — bottom right
for i, name in enumerate(geom_names):
    if geom_types[i] == 'periodic_hills':
        ax.annotate('Periodic hills',
                    xy=(frac_sep[i], max(ode_r2[i], -0.55)),
                    xytext=(0.46, -0.30), fontsize=7,
                    arrowprops=dict(arrowstyle='->', color='0.35', lw=0.8),
                    color='0.25')

# Annotate JAXA cluster — clear space above
ax.annotate(r'JAXA sep. bubbles' + '\n' +
            r'($f_\mathrm{sep} > 0.3$,' + '\n' + r'ODE succeeds)',
            xy=(0.355, 0.995), xytext=(0.44, 0.78), fontsize=6.5,
            arrowprops=dict(arrowstyle='->', color='0.5', lw=0.7),
            color='0.4')

# Annotate the APG TBL cluster
apg_idx = [i for i, t in enumerate(geom_types) if t == 'apg_tbl']
if len(apg_idx) > 1:
    apg_r2_mean = np.mean([ode_r2[i] for i in apg_idx])
    ax.annotate(f'{len(apg_idx)} APG TBL',
                xy=(0.0, apg_r2_mean), xytext=(0.06, 0.82),
                fontsize=6.5, color='0.35',
                arrowprops=dict(arrowstyle='->', color='0.5', lw=0.6))

ax.set_xlabel(r'Reversed-shear fraction $f_{\mathrm{sep}}$')
ax.set_ylabel(r'ODE $R^2(\tau_w)$')
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(-0.6, 1.12)
ax.axhline(0, color='0.75', ls='-', lw=0.5)

# Legend — 2 columns, larger font, placed in lower-left with enough room
leg = ax.legend(fontsize=6, ncol=2, loc='lower left', frameon=True,
                framealpha=0.95, edgecolor='0.8',
                handletextpad=0.4, columnspacing=1.0, borderpad=0.5,
                bbox_to_anchor=(0.0, 0.0))
leg.set_zorder(20)

save_figure(fig2, 'validity_criterion', fig_dir=FIG_DIR)
print("Figure 2 (validity criterion) saved.")

# =====================================================================
# Figure 3: Normalized heatmap — budget terms by geometry (Step 2 summary)
# =====================================================================
from matplotlib.colors import LogNorm

# Group by geometry type, compute mean of means for each budget term
type_conv = defaultdict(list)
type_visc = defaultdict(list)
type_reyn = defaultdict(list)
type_pres = defaultdict(list)

for i, name in enumerate(budget_geom_names):
    gtype = budget_geom_types[i]
    type_conv[gtype].append(budget['mean_conv'][i])
    type_visc[gtype].append(budget['mean_visc'][i])
    type_reyn[gtype].append(budget['mean_reyn'][i])
    type_pres[gtype].append(budget['mean_pres'][i])

# Order: simple → complex
order = ['channel', 'pipe', 'couette', 'tbl', 'apg_tbl',
         'separation_bubble', 'nasa_hump', 'bfs', 'periodic_hills']
geom_labels = ['Channel', 'Pipe', 'Couette', 'TBL (ZPG)', 'TBL (APG)',
               'Sep. bubble', 'NASA hump', 'BFS', 'Periodic hills']
term_labels = ['Convection', 'Pressure', 'Viscous', 'Reynolds']

valid_types = [t for t in order if t in type_conv]
valid_labels = [geom_labels[order.index(t)] for t in valid_types]
n_geom = len(valid_types)

# Build matrix: each row normalized by the dominant diffusion term (max of visc, reyn)
conv_means = np.array([np.mean(type_conv[t]) for t in valid_types])
pres_means = np.array([np.mean(type_pres[t]) for t in valid_types])
visc_means = np.array([np.mean(type_visc[t]) for t in valid_types])
reyn_means = np.array([np.mean(type_reyn[t]) for t in valid_types])

# Dominant diffusion = max(visc, reyn) per geometry
dominant = np.maximum(visc_means, reyn_means)
dominant[dominant == 0] = 1e-10  # avoid division by zero

data_matrix = np.column_stack([
    conv_means / dominant,
    pres_means / dominant,
    visc_means / dominant,
    reyn_means / dominant,
])
# Floor tiny values for log colormap
data_matrix = np.clip(data_matrix, 1e-4, None)

# ODE R² for the right-hand panel (one per geometry type)
criterion_names = list(criterion['geom_names'])
criterion_types = list(criterion['geom_types'])
criterion_r2 = criterion['ode_r2']
type_r2 = {}
for i, t in enumerate(criterion_types):
    type_r2.setdefault(t, []).append(criterion_r2[i])
r2_per_type = []
for t in valid_types:
    if t in type_r2:
        r2_per_type.append(np.mean(type_r2[t]))
    else:
        r2_per_type.append(np.nan)
r2_per_type = np.array(r2_per_type)

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5),
                                 gridspec_kw={'width_ratios': [3, 1.2],
                                              'wspace': 0.05})

# Panel (a): Heatmap
im = ax1.imshow(data_matrix, cmap='YlOrRd', aspect='auto',
                norm=LogNorm(vmin=1e-3, vmax=10))
ax1.set_xticks(range(4))
ax1.set_xticklabels(term_labels, fontsize=8)
ax1.set_yticks(range(n_geom))
ax1.set_yticklabels(valid_labels, fontsize=8)

# Annotate each cell with the numeric value
for i in range(n_geom):
    for j in range(4):
        val = data_matrix[i, j]
        txt = f'{val:.2f}' if val >= 0.01 else f'{val:.0e}'
        text_color = 'white' if val > 1.0 else 'black'
        ax1.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                 color=text_color)

cb = fig3.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cb.set_label('Term / dominant diffusion', fontsize=7)
cb.ax.tick_params(labelsize=6)
ax1.set_title('(a) Relative budget term magnitudes', fontsize=9)

# Horizontal separator between "ODE works" and "ODE fails"
# Periodic hills is the last row (index n_geom-1); separator above it
sep_line = n_geom - 1.5
ax1.axhline(sep_line, color='k', ls='--', lw=0.8)

# Panel (b): ODE R² horizontal bars
# Fill NaN (trivial single-station) with 1.0 shown as light grey
r2_display = r2_per_type.copy()
for i in range(n_geom):
    if np.isnan(r2_per_type[i]):
        r2_display[i] = 1.0  # trivially correct

# Panel (b): ODE R^2 as an aligned dot plot (no bars) -- one marker per
# heatmap row; colour = green (works), red (fails), grey (trivial). Geometry
# names live on the panel-(a) heatmap y-axis, so they are not repeated here.
ax2.axvline(0, color='0.5', ls=':', lw=0.6)
ax2.axvline(0.85, color='#2ca02c', ls='--', lw=0.7, alpha=0.5)
for i, r2 in enumerate(r2_per_type):
    if np.isnan(r2):
        col = '0.7'
    elif r2 > 0.85:
        col = '#2ca02c'
    else:
        col = '#d62728'
    xpos = float(np.clip(r2_display[i], -0.5, 1.1))
    ax2.plot([0, xpos], [i, i], '-', color=col, lw=0.8, alpha=0.35, zorder=2)
    ax2.plot(xpos, i, 'o', color=col, ms=7, zorder=3)
ax2.set_yticks([])
ax2.set_xlabel(r'ODE $R^2(\tau_w)$', fontsize=8)
ax2.set_xlim(-0.6, 1.4)
ax2.set_ylim(n_geom - 0.5, -0.5)  # invert y to match imshow (row 0 at top)
ax2.set_title(r'(b) ODE $R^2$', fontsize=9)
ax2.axhline(sep_line, color='k', ls='--', lw=0.8)

# R² value annotations
for i, r2 in enumerate(r2_per_type):
    if np.isnan(r2):
        continue
    if r2 > 0:
        ax2.text(max(r2, 0) + 0.03, i, f'{r2:.2f}',
                 va='center', fontsize=5, color='0.3')
    else:
        ax2.text(0.03, i, r'$< -0.5$',
                 va='center', fontsize=5, color='0.3')

# Big brackets on the right side for the three groups
# Group 1: Trivial (rows 0-3)
ax2.annotate('', xy=(1.25, -0.4), xytext=(1.25, 3.4),
             arrowprops=dict(arrowstyle='-', color='0.5', lw=1.5))
ax2.plot([1.22, 1.25], [-0.4, -0.4], '-', color='0.5', lw=1.5, clip_on=False)
ax2.plot([1.22, 1.25], [3.4, 3.4], '-', color='0.5', lw=1.5, clip_on=False)
ax2.text(1.28, 1.5, r'\textbf{Trivial}', fontsize=6, va='center',
         rotation=270, color='0.5', clip_on=False)

# Group 2: Works (rows 4-7)
ax2.annotate('', xy=(1.25, 3.6), xytext=(1.25, 7.4),
             arrowprops=dict(arrowstyle='-', color='#2ca02c', lw=1.5))
ax2.plot([1.22, 1.25], [3.6, 3.6], '-', color='#2ca02c', lw=1.5, clip_on=False)
ax2.plot([1.22, 1.25], [7.4, 7.4], '-', color='#2ca02c', lw=1.5, clip_on=False)
ax2.text(1.28, 5.5, r'\textbf{Works}', fontsize=6, va='center',
         rotation=270, color='#2ca02c', clip_on=False)

# Group 3: Fails (row 8)
ax2.annotate('', xy=(1.25, 7.6), xytext=(1.25, 8.4),
             arrowprops=dict(arrowstyle='-', color='#d62728', lw=1.5))
ax2.plot([1.22, 1.25], [7.6, 7.6], '-', color='#d62728', lw=1.5, clip_on=False)
ax2.plot([1.22, 1.25], [8.4, 8.4], '-', color='#d62728', lw=1.5, clip_on=False)
ax2.text(1.28, 8.0, r'\textbf{Fails}', fontsize=6, va='center',
         rotation=270, color='#d62728', clip_on=False)

save_figure(fig3, 'budget_bar_chart', fig_dir=FIG_DIR)
print("Figure 3 (budget heatmap) saved.")

print("\nAll figures generated.")
