#!/usr/bin/env python3
"""earlier submission graphical abstract: a single, clean pseudo-3-D repeating-structure
geometry -- subplot (j) of figure 6, i.e. smooth_hill(n=4, a/h=0.80) -- with NO
text, tag, badge, caption or border. Reuses the exact geometry generator and
Lambert shading from stage_classmaps_3d so it is visually identical to the panel.

Output: manuscript/figures/graphical_abstract.jpg
  - aspect ratio 1.2 : 1 (landscape), as earlier submission requires (6 cm x 5 cm)
  - 300 dpi high-resolution JPEG, white background, single panel, no border
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage_classmaps_3d as S          # exact geometry + shading, reused

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "manuscript", "figures"))

# --- subplot (j): a/h = 0.80 (amp 0.78), pitch l_p/h = 2.50 -> 4 repeats -------
x, y = S.smooth_hill(4, 0.78)

DX, DY, BASE = S.DX, S.DY, S.BASE
NS = 200
xs = np.linspace(0, 1, NS)
ys = np.interp(xs, x, y)

# Figure: single axes filling the whole canvas (no margins/ticks/spines).
# figsize 6:5 = 1.2:1; xrange/yrange is set to 1.2 as well -> isotropic mapping
# (hills are not stretched) while the frame fills exactly 1.2:1.
fig = plt.figure(figsize=(6.0, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

# (0) ground shadow plane at the solid base
ax.fill([0, 1, 1 + DX, DX], [BASE, BASE, BASE + DY, BASE + DY],
        color=S.GROUND, ec="none", zorder=0)
# (i) back cross-section + left end face (occluders)
ax.fill(np.r_[DX, xs + DX, 1 + DX], np.r_[BASE + DY, ys + DY, BASE + DY],
        color=S.DEPTH_COLOR, ec="none", zorder=1)
ax.fill([0, DX, DX, 0], [BASE, BASE + DY, ys[0] + DY, ys[0]],
        color=S.DEPTH_COLOR, ec="none", zorder=1)
# (ii) swept, Lambert-shaded skin (one quad per profile segment)
for i in range(NS - 1):
    dy = ys[i + 1] - ys[i]
    dx = xs[i + 1] - xs[i]
    c = S.skin_color(dx, dy)
    ax.fill([xs[i], xs[i + 1], xs[i + 1] + DX, xs[i] + DX],
            [ys[i], ys[i + 1], ys[i + 1] + DY, ys[i] + DY],
            color=c, ec=c, lw=0.5, zorder=2)
# (iii) right end face (in shadow)
rc = S.skin_color(0.0, -1.0)
ax.fill([1, 1, 1 + DX, 1 + DX], [BASE, ys[-1], ys[-1] + DY, BASE + DY],
        color=rc, ec=rc, lw=0.5, zorder=2)
# (iv) near front cross-section face + crisp edges
ax.fill(np.r_[0.0, x, 1.0], np.r_[BASE, y, BASE], color=S.FRONT_COLOR,
        ec="none", zorder=5)
ax.plot(xs + DX, ys + DY, color=S.EDGE, lw=0.6, alpha=0.5, zorder=4)
ax.plot(x, y, color=S.EDGE, lw=1.4, zorder=6)
ax.plot([0, 1], [BASE, BASE], color=S.EDGE, lw=1.4, zorder=6)
ax.plot([0, 0], [BASE, y[0]], color=S.EDGE, lw=1.4, zorder=6)
ax.plot([1, 1], [BASE, y[-1]], color=S.EDGE, lw=1.4, zorder=6)

# frame in a 1.2:1 window centred on the geometry (xrange/yrange = 1.2)
cx, cy, Rx = 0.58, 0.40, 1.44
Ry = Rx / 1.2
ax.set_xlim(cx - Rx / 2, cx + Rx / 2)
ax.set_ylim(cy - Ry / 2, cy + Ry / 2)

out = os.path.join(OUT, "graphical_abstract.jpg")
fig.savefig(out, dpi=300, facecolor="white", format="jpg",
            pil_kwargs={"quality": 95})
plt.close(fig)
w, h = fig.get_size_inches() * 300
print(f"wrote {out}  ({int(w)}x{int(h)} px, aspect {w/h:.3f}:1)")
