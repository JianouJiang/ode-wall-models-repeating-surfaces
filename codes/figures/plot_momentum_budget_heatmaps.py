#!/usr/bin/env python3
"""
Figure: 2D momentum budget heatmaps for periodic hills.

Multi-panel figure showing:
  (a) Convection: U dU/dx + V dU/dy
  (b) Pressure gradient: -dp/dx
  (c) Viscous diffusion: nu d2U/dy2
  (d) Reynolds stress: -d<uv>/dy
  (e) Residual
  (f) |Convection| / |Diffusion| ratio

The key message: convection is O(10^-1), comparable to pressure,
while viscous terms are O(10^-2). ODE wall models drop the O(10^-1) terms.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from plotting_utils import setup_style, save_figure

setup_style(font_size=9)

# Load computed budget
data = np.load(os.path.join(os.path.dirname(__file__), '..', 'results', 'momentum_budget_pehill.npz'))

x = data['x_unique']
y = data['y_unique']
nx, ny = len(x), len(y)

convection = data['convection']
pressure = data['pressure_grad']
viscous_y = data['viscous_y']
reynolds_y = data['reynolds_y']
residual = data['residual']
fluid = data['fluid']
wall_j = data['wall_j']

# Create meshgrid for pcolormesh
X, Y = np.meshgrid(x, y, indexing='ij')

# Mask solid region
def mask_solid(field):
    masked = field.copy()
    masked[~fluid] = np.nan
    return masked

# Also mask first 2 cells above wall (noisy derivatives)
clean_fluid = fluid.copy()
for i in range(nx):
    j0 = wall_j[i]
    clean_fluid[i, :min(j0 + 2, ny)] = False

def mask_clean(field):
    masked = field.copy()
    masked[~clean_fluid] = np.nan
    return masked

# Clip y for display (focus on near-wall region y < 1.5)
y_max = 1.5
jmax = np.searchsorted(y, y_max)

# =====================================================================
# Figure: 6-panel heatmaps
# =====================================================================
fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.5), sharex=True, sharey=True)

# Symmetric colormap limits
vmax_main = 0.3  # main terms
vmax_visc = 0.05
vmax_resid = 0.15

panels = [
    (axes[0, 0], mask_clean(convection)[:, :jmax], r'(a) Convection: $U\frac{\partial U}{\partial x} + V\frac{\partial U}{\partial y}$', vmax_main),
    (axes[0, 1], mask_clean(pressure)[:, :jmax], r'(b) Pressure: $-\frac{\partial p}{\partial x}$', vmax_main),
    (axes[1, 0], mask_clean(viscous_y)[:, :jmax], r'(c) Viscous: $\nu\frac{\partial^2 U}{\partial y^2}$', vmax_visc),
    (axes[1, 1], mask_clean(reynolds_y)[:, :jmax], r'(d) Reynolds: $-\frac{\partial \langle u^\prime v^\prime \rangle}{\partial y}$', vmax_main),
    (axes[2, 0], mask_clean(residual)[:, :jmax], r'(e) Residual', vmax_resid),
]

# |Convection|/|Diffusion| ratio
diff_total = np.abs(viscous_y) + np.abs(reynolds_y)
ratio = np.where(diff_total > 1e-10, np.abs(convection) / diff_total, np.nan)
ratio_masked = mask_clean(ratio)[:, :jmax]

for ax, field, title, vmax in panels:
    pcm = ax.pcolormesh(X[:, :jmax], Y[:, :jmax], field,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        shading='auto', rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, 9)

# Ratio panel with different colormap
ax = axes[2, 1]
pcm = ax.pcolormesh(X[:, :jmax], Y[:, :jmax], ratio_masked,
                    cmap='hot_r', vmin=0, vmax=3.0,
                    shading='auto', rasterized=True)
cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
ax.set_title(r'(f) $|\mathrm{Convection}|$ / $|\mathrm{Diffusion}|$', fontsize=8, pad=3)
ax.set_ylim(0, y_max)
ax.set_xlim(0, 9)

# Draw hill profile on all panels
# Approximate hill from wall_j
y_wall = np.array([y[wall_j[i]] if wall_j[i] < ny else 0 for i in range(nx)])
for axrow in axes:
    for ax in axrow:
        ax.fill_between(x, 0, y_wall, color='0.7', zorder=5)
        ax.plot(x, y_wall, 'k-', lw=0.5, zorder=6)

# Labels
for ax in axes[:, 0]:
    ax.set_ylabel(r'$y/h$')
for ax in axes[-1, :]:
    ax.set_xlabel(r'$x/h$')

# ── Annotations: tell the story ──────────────────────────────────
from matplotlib.patches import Ellipse

# Two intense regions: left hill slope (x/h ≈ 0.5-2) and right hill slope (x/h ≈ 7-9)
# These are where deep red/blue appear in panels (a) and (b)
left_cx, left_cy = 1.0, 0.80
left_w, left_h = 1.8, 0.65
right_cx, right_cy = 8.0, 0.80
right_w, right_h = 1.8, 0.65

def add_two_ellipses(ax, ec, lw=1.3):
    """Add dashed ellipses on both hill slopes with arrows from label."""
    ell_L = Ellipse((left_cx, left_cy), left_w, left_h,
                    fill=False, ec=ec, ls='--', lw=lw, zorder=8)
    ell_R = Ellipse((right_cx, right_cy), right_w, right_h,
                    fill=False, ec=ec, ls='--', lw=lw, zorder=8)
    ax.add_patch(ell_L)
    ax.add_patch(ell_R)

def add_label_with_arrows(ax, text, ec, text_x=4.5, text_y=1.35, fontsize=7):
    """Add text label at top-center with arrows pointing to both circles."""
    ax.text(text_x, text_y, text,
            fontsize=fontsize, ha='center', va='center', color=ec,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=ec, alpha=0.85))
    # Arrow to left circle
    ax.annotate('', xy=(left_cx + left_w/2, left_cy + left_h/2 - 0.05),
                xytext=(text_x - 0.8, text_y - 0.08),
                arrowprops=dict(arrowstyle='->', color=ec, lw=0.8))
    # Arrow to right circle
    ax.annotate('', xy=(right_cx - right_w/2, right_cy + right_h/2 - 0.05),
                xytext=(text_x + 0.8, text_y - 0.08),
                arrowprops=dict(arrowstyle='->', color=ec, lw=0.8))

# Panel (a): Convection — circle BOTH intense hill-slope regions
ax_a = axes[0, 0]
add_two_ellipses(ax_a, '#b00000')
add_label_with_arrows(ax_a, r'\textbf{ODE drops this} --- $O(10^{-1})$', '#b00000')

# Panel (b): Pressure — same two regions, equally intense
ax_b = axes[0, 1]
add_two_ellipses(ax_b, '#0055b0')
add_label_with_arrows(ax_b, r'\textbf{ODE keeps this} --- $O(10^{-1})$', '#0055b0')

# Panel (c): Viscous — same two regions are pale/weak
ax_c = axes[1, 0]
add_two_ellipses(ax_c, '#666666')
add_label_with_arrows(ax_c, r'$O(10^{-2})$: \textbf{10$\times$ smaller}', '#666666')

# Panel (d): Reynolds stress — ODE retains this via closure, also O(10⁻¹)
ax_d = axes[1, 1]
add_two_ellipses(ax_d, '#2ca02c')
add_label_with_arrows(ax_d, r'\textbf{ODE retains (via $\nu_t$)} --- $O(10^{-1})$', '#2ca02c')

# Panel (e): Residual — budget closes
ax_e = axes[2, 0]
add_two_ellipses(ax_e, '#555555')
add_label_with_arrows(ax_e, r'Residual small: \textbf{budget closes}', '#555555')

# Panel (f): Ratio — circle the bright Π > 1 region spanning the trough
ax_f = axes[2, 1]
add_two_ellipses(ax_f, '#b00000')
add_label_with_arrows(ax_f, r'$\Pi > 1$: \textbf{ODE invalid}', '#b00000')

# Key message between rows: the cancellation punchline
fig.text(0.50, 0.675,
         r'$\tau_w = O(10^{-5})$: tiny residual of (a) $-$ (b)',
         ha='center', va='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='#fff3cd', ec='#856404',
                   alpha=0.9), zorder=10)

fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')
save_figure(fig, 'momentum_budget_heatmaps', fig_dir=fig_dir)
print("Figure saved.")

# =====================================================================
# Figure 2: Streamwise variation of near-wall budget terms
# =====================================================================
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)

conv_rms = data['conv_rms_x']
pres_rms = data['pres_rms_x']
visc_rms = data['visc_rms_x']
reyn_rms = data['reyn_rms_x']
tau_w = data['tau_w_dns']
sep = data['sep_mask']

# Light Gaussian smoothing to suppress grid-level noise from the
# narrow near-wall averaging band (7 cells).  Sigma = 5 grid points
# ≈ 0.09h — small enough to preserve physical features, large enough
# to remove finite-difference artifacts near the hill crests.
from scipy.ndimage import gaussian_filter1d
_sigma = 5

def smooth(arr):
    """Gaussian-smooth with periodic wrapping (periodic hills geometry)."""
    # Pad periodically, smooth, un-pad
    padded = np.concatenate([arr[-3*_sigma:], arr, arr[:3*_sigma]])
    smoothed = gaussian_filter1d(padded, sigma=_sigma)
    return smoothed[3*_sigma:-3*_sigma]

conv_rms_s = smooth(conv_rms)
pres_rms_s = smooth(pres_rms)
visc_rms_s = smooth(visc_rms)
reyn_rms_s = smooth(reyn_rms)

# Panel (a): budget term RMS along x (smoothed)
ax1.plot(x, conv_rms_s, '-', color='#d62728', lw=1.5, label='Convection')
ax1.plot(x, pres_rms_s, '-', color='#1f77b4', lw=1.5, label='Pressure')
ax1.plot(x, reyn_rms_s, '--', color='#2ca02c', lw=1.3, label=r'Reynolds $-\partial\langle u^\prime v^\prime\rangle/\partial y$')
ax1.plot(x, visc_rms_s, '-.', color='#ff7f0e', lw=1.3, label=r'Viscous $\nu\partial^2 U/\partial y^2$')

# Shade separation region
if sep.any():
    x_sep_start = x[sep][0]
    x_sep_end = x[sep][-1]
    ax1.axvspan(x_sep_start, x_sep_end, alpha=0.1, color='gray',
                label='Separation zone')

ax1.set_ylabel('Near-wall RMS magnitude')
ax1.set_yscale('log')
ax1.set_ylim(1e-4, 1)
ax1.legend(fontsize=7, ncol=2, loc='lower right')
ax1.set_title(r'(a) Near-wall momentum budget terms (3--10 cells above wall, smoothed)', fontsize=8)

# Annotations for panel (a): all pointing inside the separation zone
# Arrow targets at x/h ≈ 2.5 (well inside grey zone)
_xt = 2.5
_ix = np.argmin(np.abs(x - _xt))

# Point arrow at x/h ≈ 1.0 where convection is clearly high (near separation)
_ix_conv = np.argmin(np.abs(x - 1.0))
ax1.annotate(
    r'\textbf{ODE drops}: $O(10^{-1})$',
    xy=(1.0, conv_rms_s[_ix_conv]),
    xytext=(1.5, 0.55),
    fontsize=7, ha='center', color='#d62728',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8),
    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#d62728', alpha=0.85))

ax1.annotate(
    r'\textbf{ODE keeps}: $O(10^{-1})$',
    xy=(3.5, pres_rms_s[np.argmin(np.abs(x - 3.5))]),
    xytext=(3.5, 0.55),
    fontsize=7, ha='center', color='#1f77b4',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8),
    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#1f77b4', alpha=0.85))

ax1.annotate(
    r'$O(10^{-2})$: \textbf{10$\times$ smaller}',
    xy=(_xt, visc_rms_s[_ix]),
    xytext=(1.5, 2e-4),
    fontsize=7, ha='center', color='#ff7f0e',
    arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=0.8),
    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#ff7f0e', alpha=0.85))

# Panel (b): convection/diffusion ratio and tau_w (smoothed)
ratio_s = smooth(data['ratio_x'])
ax2.plot(x, ratio_s, '-', color='#d62728', lw=1.5)
ax2.axhline(1.0, color='0.5', ls='--', lw=0.5)
ax2.set_ylabel(r'$|\mathrm{Conv}|$ / $|\mathrm{Diff}|$', color='#d62728')
ax2.set_ylim(0, 5)

if sep.any():
    ax2.axvspan(x_sep_start, x_sep_end, alpha=0.1, color='gray')

# Secondary axis: tau_w (smoothed)
tau_w_s = smooth(tau_w)
ax2b = ax2.twinx()
ax2b.plot(x, tau_w_s * 1e3, '-', color='#1f77b4', lw=1.0, alpha=0.7)
ax2b.axhline(0, color='#1f77b4', ls='-', lw=0.3)
ax2b.set_ylabel(r'$\tau_w \times 10^3$', color='#1f77b4')

ax2.set_xlabel(r'$x/h$')
ax2.set_xlim(0, 9)
ax2.set_title('(b) Convection/diffusion ratio and wall shear stress', fontsize=8)

# Key insight 1: R > 1 AND τ_w ≈ 0 in the separation zone
# Two arrows from text to both red-curve peaks inside sep zone
# Find the two R peaks: near separation (~x/h 0.5) and near reattachment (~x/h 4)
_sep_mask_left = sep & (x < 2.5)
_sep_mask_right = sep & (x > 2.5)
_peak1_i = np.argmax(ratio_s * _sep_mask_left)
_peak2_i = np.argmax(ratio_s * _sep_mask_right)

ax2.text(2.0, 4.5,
         r'$\mathcal{R} > 1$ and $\tau_w \approx 0$'
         '\n'
         r'$\Rightarrow\; \varepsilon \ll 1$: cancellation',
         fontsize=6.5, ha='center', color='#b00000',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#b00000', alpha=0.85))
# Arrow to left peak (separation)
ax2.annotate('', xy=(x[_peak1_i], min(ratio_s[_peak1_i], 4.8)),
             xytext=(1.3, 4.2),
             arrowprops=dict(arrowstyle='->', color='#b00000', lw=0.8))
# Arrow to right peak (reattachment)
ax2.annotate('', xy=(x[_peak2_i], min(ratio_s[_peak2_i], 4.8)),
             xytext=(2.7, 4.2),
             arrowprops=dict(arrowstyle='->', color='#b00000', lw=0.8))

# Key insight 2: R > 1 but τ_w large at the hill crest
# Arrow points to the BLUE τ_w curve at x/h ≈ 8.3 (right axis)
_ix_83 = np.argmin(np.abs(x - 8.3))
_tau_at_83 = tau_w_s[_ix_83] * 1e3  # in the τ_w × 10³ units
# Convert τ_w right-axis value to left-axis coordinates for annotation
# Use ax2b (right axis) for the arrow target
ax2.text(7.0, 3.0,
         r'$\mathcal{R} > 1$ but $\tau_w$ large'
         '\n'
         r'$\Rightarrow\; \varepsilon \sim O(1)$: ODE survives',
         fontsize=6.5, ha='center', color='#0055b0',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#0055b0', alpha=0.85))
ax2.annotate('', xy=(8.4, 3.0), xytext=(8.0, 3.0),
             arrowprops=dict(arrowstyle='->', color='#0055b0', lw=1.2))

save_figure(fig2, 'momentum_budget_streamwise', fig_dir=fig_dir)
print("Streamwise figure saved.")

# =====================================================================
# Print summary for manuscript
# =====================================================================
print("\n=== SUMMARY FOR MANUSCRIPT ===")
print(f"Separation region: x/h in [{x[sep].min():.2f}, {x[sep].max():.2f}]")
print(f"Near-wall conv RMS (overall): {conv_rms.mean():.4e}")
print(f"Near-wall pres RMS (overall): {pres_rms.mean():.4e}")
print(f"Near-wall visc RMS (overall): {visc_rms.mean():.4e}")
print(f"Near-wall reyn RMS (overall): {reyn_rms.mean():.4e}")
valid_ratio = data['ratio_x'][data['ratio_x'] > 0]
print(f"Conv/Diff ratio: median={np.median(valid_ratio):.2f}, "
      f"mean={valid_ratio.mean():.2f}, "
      f"p90={np.percentile(valid_ratio, 90):.2f}")
