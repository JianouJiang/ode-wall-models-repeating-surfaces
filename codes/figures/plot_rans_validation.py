#!/usr/bin/env python3
"""
Figure: RANS k-omega SST wall shear stress distributions for periodic hills
and backward-facing step, with an inset table summarising RANS vs DNS f_sep.

Sign convention: OpenFOAM wallShearStress has tau_x > 0 for reversed
(separated) flow and tau_x < 0 for attached forward flow on bottom walls.
Shading marks regions with tau_x > 0 (separated).
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from plotting_utils import setup_style, save_figure

setup_style(font_size=9)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

data = np.load(os.path.join(RESULTS_DIR, 'rans_validation.npz'), allow_pickle=True)

# === Data ===
# Periodic hills
ph_cx = data['pehill_cx']
ph_tau = data['pehill_tau_x']
ph_fsep = float(data['pehill_fsep'])
ph_fsep_dns = float(data['pehill_fsep_dns'])

# BFS — downstream only
bfs_cx = data['bfs_ds_cx']
bfs_tau = data['bfs_ds_tau_x']
bfs_fsep = float(data['bfs_fsep_downstream'])
bfs_fsep_dns = float(data['bfs_fsep_dns'])
bfs_reattach = float(data['bfs_reattachment_x'])

# Sort by x
ph_sort = np.argsort(ph_cx)
ph_cx = ph_cx[ph_sort]
ph_tau = ph_tau[ph_sort]

bfs_sort = np.argsort(bfs_cx)
bfs_cx = bfs_cx[bfs_sort]
bfs_tau = bfs_tau[bfs_sort]

# === Figure: 2 panels + inset table ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.0), height_ratios=[1, 1])

# --- Panel (a): Periodic hills ---
ax1.plot(ph_cx, ph_tau, '-', color='#1f77b4', lw=1.5)
ax1.axhline(0, color='0.5', ls='-', lw=0.3)
ax1.fill_between(ph_cx, ph_tau, 0, where=ph_tau > 0,
                 color='#4C78A8', alpha=0.25, label='Separated',
                 hatch='///', edgecolor='#4C78A8', linewidth=0)
ax1.fill_between(ph_cx, ph_tau, 0, where=ph_tau <= 0,
                 color='#E8E8E8', alpha=0.5, label='Attached')

ax1.set_xlabel(r'$x/H$')
ax1.set_ylabel(r'$\tau_{w,x}$')
ax1.set_title(r'(a) Periodic hills, $Re_b = 5600$ (k-$\omega$ SST)', fontsize=9, pad=3)
ax1.legend(fontsize=7, loc='lower left')
# Sign convention note
ax1.text(0.97, 0.05, r'$\tau_{w,x} > 0$: reversed flow',
         transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=6, color='0.45', style='italic')

ax1.text(0.97, 0.95,
         rf'$f_{{\mathrm{{sep}}}}^{{\mathrm{{RANS}}}} = {ph_fsep:.2f}$'
         '\n'
         rf'$f_{{\mathrm{{sep}}}}^{{\mathrm{{DNS}}}} = {ph_fsep_dns:.2f}$',
         transform=ax1.transAxes, ha='right', va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))

# --- Panel (b): BFS ---
ax2.plot(bfs_cx, bfs_tau, '-', color='#1f77b4', lw=1.5)
ax2.axhline(0, color='0.5', ls='-', lw=0.3)

ax2.fill_between(bfs_cx, bfs_tau, 0, where=bfs_tau > 0,
                 color='#4C78A8', alpha=0.25, label='Separated',
                 hatch='///', edgecolor='#4C78A8', linewidth=0)
ax2.fill_between(bfs_cx, bfs_tau, 0, where=bfs_tau <= 0,
                 color='#E8E8E8', alpha=0.5, label='Attached')

# Reattachment markers (RANS)
ax2.axvline(bfs_reattach, color='#d62728', ls='--', lw=0.8, alpha=0.7)
ax2.annotate(rf'$x_r^{{\mathrm{{RANS}}}} = {bfs_reattach:.1f}H$',
             xy=(bfs_reattach, 0),
             xytext=(bfs_reattach + 1.5, max(bfs_tau)*0.6), fontsize=6.5,
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.6),
             color='#d62728')

# DNS reattachment reference
ax2.axvline(6.28, color='0.4', ls=':', lw=0.8, alpha=0.7)
ax2.annotate(r'$x_r^{\mathrm{DNS}} = 6.3H$',
             xy=(6.28, 0), xytext=(3.5, max(bfs_tau)*0.8), fontsize=6.5,
             arrowprops=dict(arrowstyle='->', color='0.4', lw=0.6),
             color='0.4')

ax2.set_xlabel(r'$x/H$')
ax2.set_ylabel(r'$\tau_{w,x}$')
ax2.set_title(r'(b) Backward-facing step, $Re_H = 13\,700$ (k-$\omega$ SST)',
              fontsize=9, pad=3)
ax2.legend(fontsize=6.5, loc='upper right', ncol=2)

ax2.text(0.03, 0.95,
         rf'$f_{{\mathrm{{sep}}}}^{{\mathrm{{RANS}}}} = {bfs_fsep:.2f}$'
         '\n'
         rf'$f_{{\mathrm{{sep}}}}^{{\mathrm{{DNS}}}} = {bfs_fsep_dns:.2f}$',
         transform=ax2.transAxes, ha='left', va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))

# --- Inset table in panel (b) showing RANS vs DNS summary ---
threshold = float(data['fsep_threshold'])
table_data = [
    ['', r'$f_{\mathrm{sep}}^{\mathrm{DNS}}$',
     r'$f_{\mathrm{sep}}^{\mathrm{RANS}}$', r'$\Delta$'],
    ['P. hills', f'{ph_fsep_dns:.2f}', f'{ph_fsep:.2f}',
     f'+{ph_fsep - ph_fsep_dns:.2f}'],
    ['BFS', f'{bfs_fsep_dns:.2f}', f'{bfs_fsep:.2f}',
     f'+{bfs_fsep - bfs_fsep_dns:.2f}'],
]

# Place a matplotlib table in lower-right of panel (b)
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='lower right',
                bbox=[0.55, 0.03, 0.43, 0.38])
tbl.auto_set_font_size(False)
tbl.set_fontsize(6.5)
for key, cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
    cell.set_edgecolor('0.6')
    if key[0] == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('white')

fig.tight_layout()
save_figure(fig, 'rans_validation', fig_dir=FIG_DIR)
print("RANS validation figure saved.")
