#!/usr/bin/env python3
"""
Figure 1: Paper contribution overview — two-row flowchart.

Top row: BEFORE — engineer's workflow without ε diagnostic
  Flow with separation → "ODE fails in separation" → two bad outcomes

Bottom row: AFTER — with ε diagnostic
  Flow with separation → compute ε → informed decision → correct outcome
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from plotting_utils import setup_style, save_figure

setup_style(font_size=9, use_tex=True)
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 3.2),
                                      gridspec_kw={'hspace': 0.5})

GREEN = '#2ca02c'
RED = '#d62728'
BLUE = '#1f77b4'
ORANGE = '#e67e00'

def add_box(ax, x, y, w, h, text, fc, ec, fontsize=6.5, text_color='black',
            bold_text=None, bold_color=None):
    """Add a rounded box with text."""
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle='round,pad=0.12', fc=fc, ec=ec, lw=1.0))
    if bold_text:
        ax.text(x, y + 0.08, text, ha='center', va='bottom',
                fontsize=fontsize, color=text_color)
        ax.text(x, y - 0.08, bold_text, ha='center', va='top',
                fontsize=fontsize, color=bold_color or text_color,
                fontweight='bold')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=text_color)

def add_arrow(ax, x1, y1, x2, y2, color='0.3'):
    """Add arrow between boxes."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3))

# =====================================================================
# TOP ROW: Before (without ε)
# =====================================================================
ax = ax_top
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.8, 1.0)
ax.axis('off')

# Row label
ax.text(-0.3, 0.95, r'\textbf{(a) Before}', fontsize=9, va='top', color='0.3')

# Box 1: Starting point
add_box(ax, 1.0, 0.4, 1.8, 0.7,
        r'Flow with' + '\n' + r'separation',
        fc='#f0f0f0', ec='0.5', fontsize=7)

# Arrow →
add_arrow(ax, 1.95, 0.4, 2.55, 0.4)

# Box 2: Conventional wisdom
add_box(ax, 3.7, 0.4, 2.0, 0.7,
        r'\textit{``ODE fails in}' + '\n' + r'\textit{separation''}',
        fc='#ffe0e0', ec=RED, fontsize=7, text_color=RED)

# Arrow → splits into two outcomes
add_arrow(ax, 4.75, 0.65, 6.2, 0.65)
add_arrow(ax, 4.75, 0.15, 6.2, 0.15)

# Box 3a: Outcome 1 — waste
add_box(ax, 7.5, 0.65, 2.3, 0.55,
        r'Use expensive PDE/ML model',
        fc='#fff3cd', ec=ORANGE, fontsize=6.5,
        bold_text=r'Waste: ODE would work', bold_color=ORANGE)

# Box 3b: Outcome 2 — disaster
add_box(ax, 7.5, 0.1, 2.3, 0.55,
        r'Trust ODE (validated on BFS)',
        fc='#f8d7da', ec=RED, fontsize=6.5,
        bold_text=r'Disaster: wrong by $10^4\times$', bold_color=RED)

# Labels on arrows
ax.text(5.5, 0.78, r'easy flow', fontsize=5.5, ha='center',
        color='0.4', style='italic')
ax.text(5.5, 0.0, r'hard flow', fontsize=5.5, ha='center',
        color='0.4', style='italic')

# Red X — no way to tell
ax.text(10.2, 0.4, r'\textbf{?}', fontsize=14, ha='center', va='center',
        color=RED)

# =====================================================================
# BOTTOM ROW: After (with ε)
# =====================================================================
ax = ax_bot
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.8, 1.0)
ax.axis('off')

# Row label
ax.text(-0.3, 0.95, r'\textbf{(b) This paper}', fontsize=9, va='top', color=BLUE)

# Box 1: Same starting point
add_box(ax, 1.0, 0.4, 1.8, 0.7,
        r'Flow with' + '\n' + r'separation',
        fc='#f0f0f0', ec='0.5', fontsize=7)

# Arrow →
add_arrow(ax, 1.95, 0.4, 2.55, 0.4, color=BLUE)

# Box 2: NEW — identify separation type
add_box(ax, 3.7, 0.4, 2.0, 0.7,
        r'Identify separation' + '\n' +
        r'mechanism',
        fc='#e8f4fd', ec=BLUE, fontsize=7, text_color=BLUE)

# Arrow → splits by separation type
add_arrow(ax, 4.75, 0.65, 6.2, 0.65, color=GREEN)
add_arrow(ax, 4.75, 0.15, 6.2, 0.15, color=RED)

# Box 3a: thin-BL / pressure-induced → ODE OK
add_box(ax, 7.5, 0.65, 2.3, 0.55,
        r'Thin-BL / pressure-induced:',
        fc='#d4edda', ec=GREEN, fontsize=6.5,
        bold_text=r'ODE works ($\varepsilon \sim O(1)$)', bold_color=GREEN)

# Box 3b: geometry-driven recirculation → use PDE
add_box(ax, 7.5, 0.1, 2.3, 0.55,
        r'Geometry-driven recirc.:',
        fc='#f8d7da', ec=RED, fontsize=6.5,
        bold_text=r'ODE fails ($\varepsilon \ll 1$)', bold_color=RED)

# Labels on arrows
ax.text(5.5, 0.78, r'$L_\mathrm{sep} \gg \delta$', fontsize=6, ha='center',
        color=GREEN, fontweight='bold')
ax.text(5.5, 0.0, r'$L_\mathrm{sep} \sim \delta$', fontsize=6, ha='center',
        color=RED, fontweight='bold')

# Green check
ax.text(10.2, 0.4, r'$\surd$', fontsize=18, ha='center', va='center',
        color=GREEN)

save_figure(fig, 'fig_contribution_overview', fig_dir=FIG_DIR)
print("Figure 1 saved.")
