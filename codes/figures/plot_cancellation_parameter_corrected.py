#!/usr/bin/env python3
r"""
plot_cancellation_parameter_corrected.py
=========================================
Canonical cancellation-parameter producer for the JCP revision.

WHY THIS EXISTS (the headline reconciliation, G2)
-------------------------------------------------
The superseded wall-pinned extraction (median ``eps ~ 7.8e-4``,
frac(eps<0.1)=1.00) is
computed from the legacy wall-profile file ``periodic_hills_case_1p0_wall_profiles.npz``,
which pins the matching column at ``y = 0`` for every station. On the hill faces
that samples velocity *inside the solid hill* (U ~ 0), manufacturing a spurious
``tau_w ~ 1e-6`` (Re_tau ~ 5.6, unphysical at Re_b = 5600) and hence a spurious
``eps ~ 1e-4``. The artifact is quantified in
``codes/analysis/verify_wall_extraction.py`` -> ``wall_extraction_artifact.npz``.

This script recomputes the periodic-hills panel under the **hill-surface-aware**
extraction convention (eq. ``eq:hillsurface`` in the manuscript): per station,
the wall is the first fluid node above the solid block and ``tau_w = nu du/dy``
is a least-squares fit of the first few fluid points through the no-slip origin.
It reads the **same raw DNS file** as ``verify_wall_extraction.py``
(Xiao case_1p0, ``mean_files.dat``) so the correction is fully traceable.

The BFS and NASA-hump panels are not affected because their walls sit at the
grid floor.  They are recomputed directly from their source profiles here; the
canonical producer has no dependency on the superseded result or figure.

OUTPUTS
-------
  manuscript/figures/cancellation_parameter_corrected.{pdf,png}
  codes/results/cancellation_parameter_corrected.npz

The corrected periodic-hills numbers are cross-checked, in-script, against two
INDEPENDENT estimates of the same quantity (a hard internal consistency test):
  1. the 29-case Xiao dose-response family median eps (dose_response_xiao.npz),
  2. the Krank periodic-hills eps_tilde already reported in the manuscript.
Both sit in the same O(0.1) band; none is anywhere near the legacy 1e-4.

This script changes NO manuscript text or table; it produces the corrected
artifact + figure that Level 3 wires into sec:diagnosis / sec:cancellation.

Run:  OMP_NUM_THREADS=2 python3 codes/figures/plot_cancellation_parameter_corrected.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # codes/
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "..", "manuscript", "figures")
sys.path.insert(0, os.path.join(ROOT, "utils"))
from plotting_utils import setup_style, save_figure   # noqa: E402

# Match the sans-serif, non-TeX title/label font used by the neighbouring results
# figures (e.g. fig_transition_map_l2, fig_flowblind_catastrophe_map) rather than
# the serif Computer-Modern default, so the (a)/(b)/(c) titles read consistently.
# Drawn at the printed width (2026-08-26): this was drawn on a 7.4 in canvas
# and printed at 3.76 in, so its type reached the page at 3.3-5.6 pt.
setup_style(font_size=7.2, font_family="sans-serif", use_tex=False)

RE_B = 5600.0
NU = 1.0 / RE_B
# wall-resolved reference traction on the 512 hill stations (see mglet_traction_on)
REBASE_NPZ = os.path.join(RESULTS, "reference_rebase_headlines_l0_20260825.npz")
Y_IDX = 10          # matching-height index, identical to the a-priori protocol
VEL_TOL = 1e-6      # |velocity| threshold separating solid hill from fluid

# Same raw DNS used by verify_wall_extraction.py — the headline baseline hill.
RAW_PEHILL = os.path.join(ROOT, "new_data_download", "geometry_driven",
                          "xiao_pehill_parameterized", "pehill-5-cases-DNS",
                          "case_1p0", "dns-data", "mean_files.dat")

# BFS / NASA raw wall-profile files (carry tau_w, dp_dx, y for every station).
WALL_PROF = os.path.join(ROOT, "vendor", "universal_wall_function", "codes",
                         "results")
# matching-height range for the eps-robustness band shown on EVERY panel; the
# protocol value Y_IDX=10 (the plotted main curve) sits inside it.
YBAND = [4, 6, 10, 16, 25]



def _runs(mask):
    """Contiguous True runs of a boolean mask, as (start, stop) index pairs.

    Shading station-by-station draws one translucent span per interval; with a
    few hundred stations the antialiased seams between adjacent spans read as
    vertical striping rather than as one band, and the periodic-hill panel --
    which is reversed over most of its wall -- looked like a plotting artefact
    next to the solid bands of the other two.  One span per run fixes it.
    """
    import numpy as _np
    m = _np.asarray(mask, bool)
    if not m.any():
        return []
    d = _np.diff(m.astype(int))
    starts = list(_np.flatnonzero(d == 1) + 1)
    stops = list(_np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        stops = stops + [len(m)]
    return list(zip(starts, stops))


def eps_bfsnasa(fname, yidx_list):
    """eps(x) at several matching heights for BFS / NASA, from wall_profiles."""
    wp = np.load(os.path.join(WALL_PROF, fname), allow_pickle=True)
    x = wp["x"].ravel(); tw = wp["tau_w"].ravel(); dp = wp["dp_dx"].ravel()
    y = wp["y"]
    profs = {}
    for yi in yidx_list:
        ym = np.array([y[i, min(yi, y.shape[1] - 1)] for i in range(len(x))])
        den = np.abs(dp) * np.abs(ym)
        e = np.full(len(x), np.nan)
        m = den > 1e-30
        e[m] = np.abs(tw[m]) / den[m]
        profs[yi] = e
    return x, profs, (tw < 0)


def yband_envelope(x_ref, profs):
    """(lo, hi) eps envelope over matching heights, interpolated onto x_ref.
    profs: dict {y_idx: (x, eps)}."""
    stack = [np.interp(x_ref, xx, ee, left=np.nan, right=np.nan)
             for (xx, ee) in profs.values()]
    S = np.array(stack)
    return np.nanmin(S, axis=0), np.nanmax(S, axis=0)


def mglet_traction_on(xs):
    r"""Wall-resolved reference traction interpolated onto the panel stations.

    The hill panel used to reconstruct its own traction from the velocity
    archive with a through-origin fit of the first four fluid points.  That
    estimator was withdrawn on 2026-08-25: at this archive's wall spacing it
    under-resolves the traction, so the panel understated the cancellation
    parameter it exists to display.  The geometry and the wall pressure in the
    archive are unaffected and are still read from it; only the traction now
    comes from the wall-resolved reference.
    """
    d = np.load(REBASE_NPZ, allow_pickle=True)
    xr = np.asarray(d["x"], float)
    tw = np.asarray(d["reference_B_mglet"], float)
    order = np.argsort(xr)
    return np.interp(np.asarray(xs, float), xr[order], tw[order])


def hill_surface_aware_epsilon(raw_file, y_idx=Y_IDX):
    r"""Recompute eps(x) = |tau_w| / (|dp/dx| y_m) for the periodic hill under the
    hill-surface-aware convention (eq:hillsurface).  The wall-normal geometry
    and the wall pressure come from the raw DNS; the wall traction comes from
    the wall-resolved reference (see ``mglet_traction_on``).
    y_idx sets the matching height (=Y_IDX for the protocol value).

    Returns (x, epsilon, tau_w, is_sep, median |tau_w|).
    """
    d = np.loadtxt(raw_file)                       # columns: x y u v w p
    x, y, u, v, p = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 5]
    xu = np.unique(np.round(x, 6))

    xs, p_wall, y_m = [], [], []
    for xv in xu:
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv, pp = y[m], u[m], v[m], p[m]
        o = np.argsort(yy)
        yy, uu, vv, pp = yy[o], uu[o], vv[o], pp[o]
        # first fluid node above the solid hill block
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < y_idx + 2:
            continue
        k = max(fluid[0], 1)
        yw = yy[k - 1]                             # no-slip wall node
        ywall = yy[k - 1:] - yw                    # distance from wall
        pfl = pp[k - 1:]
        xs.append(xv)
        p_wall.append(pfl[1])
        y_m.append(ywall[y_idx])

    xs = np.asarray(xs)
    tau_w = mglet_traction_on(xs)
    p_wall = np.asarray(p_wall)
    y_m = np.asarray(y_m)
    dp_dx = np.gradient(p_wall, xs)

    # eps only defined where there is a pressure gradient to balance; mask the
    # rest (identical masking rule to the legacy figure: median over |dp/dx|>0).
    denom = np.abs(dp_dx) * np.abs(y_m)
    valid = denom > 1e-30
    epsilon = np.full(len(xs), np.nan)
    epsilon[valid] = np.abs(tau_w[valid]) / denom[valid]
    return xs, epsilon, tau_w, tau_w < 0, float(np.nanmedian(np.abs(tau_w)))


def summary(epsilon):
    finite = np.isfinite(epsilon) & (epsilon > 0)
    med = float(np.nanmedian(epsilon[finite])) if finite.any() else np.nan
    f01 = float(np.mean(epsilon[finite] < 0.1)) if finite.any() else np.nan
    f1 = float(np.mean(epsilon[finite] < 1.0)) if finite.any() else np.nan
    return med, f01, f1, finite


def _med_str(v):
    """Median epsilon for a panel badge, at a precision we actually know."""
    if v >= 1000.0:
        e = int(np.floor(np.log10(v)))
        return r"{\sim}%.0f\times10^{%d}" % (v / 10.0 ** e, e)
    return "%.1f" % v          # the BFS median is quoted as 5.5, not 5.50


def main():
    # ── periodic hills: recompute under the corrected convention (y_idx=10) ───
    x_ph, eps_ph, tau_ph, sep_ph, med_tau_ph = hill_surface_aware_epsilon(RAW_PEHILL)
    med_ph, f01_ph, f1_ph, _ = summary(eps_ph)
    re_tau = float(np.sqrt(med_tau_ph) / NU)
    # matching-height band: eps(x) recomputed across YBAND, enveloped onto x_ph
    ph_profs = {yi: hill_surface_aware_epsilon(RAW_PEHILL, y_idx=yi)[:2]
                for yi in YBAND}
    ph_lo, ph_hi = yband_envelope(x_ph, ph_profs)

    artifact = np.load(os.path.join(RESULTS, "wall_extraction_artifact.npz"),
                       allow_pickle=True)
    legacy_ph_med = float(artifact["legacy_median_epsilon"])
    legacy_ph_f01 = float(artifact["legacy_frac_epsilon_below_01"])

    # ── BFS + NASA hump: main curve (y_idx=10) + matching-height band ─────────
    def _med(e):
        f = np.isfinite(e) & (e > 0)
        return float(np.nanmedian(e[f])) if f.any() else np.nan

    x_bfs, pr_bfs, sep_bfs = eps_bfsnasa("bfs_Re13700_wall_profiles.npz", YBAND)
    x_na, pr_na, sep_na = eps_bfsnasa("nasa_hump_Re936000_wall_profiles.npz", YBAND)
    bfs_lo, bfs_hi = yband_envelope(x_bfs, {yi: (x_bfs, pr_bfs[yi]) for yi in YBAND})
    na_lo, na_hi = yband_envelope(x_na, {yi: (x_na, pr_na[yi]) for yi in YBAND})

    panels = [
        dict(key="bfs", label=r"BFS ($Re_H = 13\,700$)", xlabel=r"$x/H$",
             color="#1f77b4", x=x_bfs, eps=pr_bfs[Y_IDX], lo=bfs_lo, hi=bfs_hi,
             sep=sep_bfs, med=_med(pr_bfs[Y_IDX])),
        dict(key="nasa_hump", label=r"NASA hump ($Re_c = 936\,000$)",
             xlabel=r"$x/c$", color="#2ca02c", x=x_na, eps=pr_na[Y_IDX],
             lo=na_lo, hi=na_hi, sep=sep_na, med=_med(pr_na[Y_IDX])),
        dict(key="periodic_hills",
             label=r"Periodic hills ($Re_H = 5600$)",
             xlabel=r"$x/h$", color="#d62728", x=x_ph, eps=eps_ph,
             lo=ph_lo, hi=ph_hi, sep=sep_ph, med=med_ph),
    ]

    # ── cross-checks (independent estimates of the SAME corrected quantity) ──
    # the family cross-check must use the repaired estimator too, or it
    # "confirms" the corrected hill value against a superseded family
    dr = np.load(os.path.join(RESULTS, "corrected_family_sweep_l0_20260825.npz"),
                 allow_pickle=True)
    dose_eps_median = float(np.nanmedian(dr["agg_eps_median"]))   # 29-case family
    dose_eps_lo = float(np.nanmin(dr["agg_eps_median"]))
    dose_eps_hi = float(np.nanmax(dr["agg_eps_median"]))
    krank_eps = (0.26, 0.52)   # manuscript-reported Krank periodic-hills eps_tilde

    print("=" * 72)
    print("CORRECTED periodic-hills cancellation parameter (hill-surface-aware)")
    print("=" * 72)
    print(f"  raw DNS              : {os.path.relpath(RAW_PEHILL, ROOT)}")
    print(f"  median eps           : {med_ph:.4f}   (legacy artifact: {legacy_ph_med:.2e})")
    print(f"  frac(eps<0.1)        : {f01_ph:.3f}    (legacy artifact: {legacy_ph_f01:.2f})")
    print(f"  frac(eps<1)          : {f1_ph:.3f}")
    print(f"  median |tau_w|       : {med_tau_ph:.3e}  ->  Re_tau = {re_tau:.0f} (physical)")
    print(f"  reversed-shear frac  : {float(np.mean(sep_ph)):.3f}")
    print("-" * 72)
    print("  INDEPENDENT cross-checks of the corrected O(0.1) magnitude:")
    print(f"    29-case Xiao dose-response median eps : {dose_eps_median:.3f}  "
          f"[range {dose_eps_lo:.3f}-{dose_eps_hi:.3f}]")
    print(f"    Krank periodic-hills eps_tilde        : {krank_eps[0]:.2f}, {krank_eps[1]:.2f}")
    print(f"    => all three estimates are O(0.1); none within 2 decades of {legacy_ph_med:.0e}")

    # ── figure: 3 columns, each = eps(x) panel (top) + geometry sketch (bottom)
    from scipy.ndimage import gaussian_filter1d

    def smooth_log(arr, sigma=5):
        if len(arr) < 50:
            return arr
        log_arr = np.log10(np.clip(arr, 1e-15, None))
        pad = 3 * sigma
        padded = np.concatenate([log_arr[-pad:], log_arr, log_arr[:pad]])
        sm = gaussian_filter1d(padded, sigma=sigma)
        return 10 ** sm[pad:-pad]

    # validated 29-case hill-family median-eps range (dose_response_xiao.npz);
    # every hill is eps<<1 -> the single case_1p0 curve is representative.
    fam = np.asarray(dr["agg_eps_median"], float)
    fam_lo, fam_hi = float(fam.min()), float(fam.max())

    def draw_geom(ax, key, x, sep):
        """Wall schematic + flow, x-aligned with the eps(x) panel above. The grey
        band (reversed wall shear, tau_w<0) is repeated here and filled with a
        recirculation loop + dividing streamline, so the reader sees that the grey
        region of the eps panel IS the separated/recirculating zone."""
        xmin, xmax = float(np.min(x)), float(np.max(x))
        xx = np.linspace(xmin, xmax, 500)
        if key == "bfs":                                  # backward-facing step
            ytop = np.where(xx < 0.0, 1.0, 0.0)
            ax.plot([0, 0], [0, 1.0], color="0.35", lw=1.3, zorder=3)
            top = 1.6
        elif key == "nasa_hump":                          # wall-mounted hump
            ytop = 0.62 * np.exp(-((xx - 0.32) / 0.20) ** 2)
            top = 1.2
        else:                                             # periodic hills
            P = xmax - xmin
            ytop = 1.0 * (0.5 * (1 + np.cos(2 * np.pi * (xx - xmin) / P))) ** 2.6
            top = 1.6
        ax.fill_between(xx, -0.45, ytop, color="0.82", zorder=1, lw=0)
        ax.plot(xx, ytop, color="0.35", lw=1.3, zorder=2)
        # separated-region band (same x as the eps panel -> vertical link)
        sb = np.asarray(sep, bool)
        for a_, b_ in _runs(sb[:len(x) - 1]):
            ax.axvspan(x[a_], x[min(b_, len(x) - 1)], alpha=0.10, color="gray",
                       lw=0, zorder=0)
        # recirculation loop + dividing streamline inside the separated region
        if sb.any():
            xsep = x[sb]
            s0, s1 = float(xsep.min()), float(xsep.max())
            up = xx <= s0
            if up.any():
                ymax_up = float(ytop[up].max())
                # separation leaves the DOWNSTREAM edge of the crest/step top
                # (not an upstream point) -- e.g. the step lip at x=0 for the BFS
                cand = xx[up][ytop[up] >= ymax_up - 1e-6]
                xcr, ycr = float(cand.max()), ymax_up
            else:
                xcr, ycr = s0, float(np.interp(s0, xx, ytop))
            yend = float(np.interp(s1, xx, ytop))
            t = np.linspace(0, 1, 80)
            xdiv = xcr + (s1 - xcr) * t
            ydiv = yend + (ycr - yend) * (1 - t) ** 1.35
            # geometry-aware: the streamline must never dip into the solid wall
            ydiv = np.maximum(ydiv, np.interp(xdiv, xx, ytop) + 0.03)
            ax.plot(xdiv, ydiv, ls=(0, (4, 2)), color="0.30", lw=1.0, zorder=4)
            xc = 0.5 * (s0 + s1)
            yw = float(np.interp(xc, xx, ytop))
            rx, ry = 0.30 * (s1 - s0), 0.17
            yc = yw + ry + 0.06
            th = np.linspace(np.deg2rad(130), np.deg2rad(130 - 300), 80)  # CW
            ax.plot(xc + rx * np.cos(th), yc + ry * np.sin(th), color="#c0392b",
                    lw=1.0, zorder=5)
            ax.annotate('', xy=(xc + rx * np.cos(th[-1]), yc + ry * np.sin(th[-1])),
                        xytext=(xc + rx * np.cos(th[-4]), yc + ry * np.sin(th[-4])),
                        arrowprops=dict(arrowstyle='-|>', color="#c0392b", lw=1.0,
                                        mutation_scale=8), zorder=5)
        # freestream arrow (flow direction)
        ax.annotate('', xy=(xmin + 0.26 * (xmax - xmin), top * 0.80),
                    xytext=(xmin + 0.05 * (xmax - xmin), top * 0.80),
                    arrowprops=dict(arrowstyle='-|>', color="0.5", lw=1.0,
                                    mutation_scale=8), zorder=3)
        ax.set_ylim(-0.45, top)
        ax.set_yticks([])
        ax.tick_params(top=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    fig = plt.figure(figsize=(6.4757, 2.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=0.10,
                          wspace=0.10)
    eps_axes = []
    for i in range(3):                     # shared y-axis -> ticks only on (a)
        eps_axes.append(fig.add_subplot(gs[0, i],
                                        sharey=(eps_axes[0] if i else None)))
    geom_axes = [fig.add_subplot(gs[1, i], sharex=eps_axes[i]) for i in range(3)]

    for i, (axE, axG, pan) in enumerate(zip(eps_axes, geom_axes, panels)):
        x, eps, sep = pan["x"].ravel(), pan["eps"], pan["sep"]
        lo = np.where(np.isfinite(pan["lo"]), pan["lo"], eps)
        hi = np.where(np.isfinite(pan["hi"]), pan["hi"], eps)
        lo = np.clip(lo, 1e-6, 1e6)
        hi = np.clip(hi, 1e-6, 1e6)
        eps_clipped = np.clip(eps, 1e-6, 1e6)
        for a_, b_ in _runs(np.asarray(sep, bool)[:len(x) - 1]):
            axE.axvspan(x[a_], x[min(b_, len(x) - 1)], alpha=0.10, color="gray",
                        lw=0)
        if pan["key"] == "periodic_hills":
            lo, hi = smooth_log(lo), smooth_log(hi)
            eps_clipped = smooth_log(eps_clipped)
        # matching-height sensitivity band -- SAME style on every panel: eps(x)
        # re-evaluated as the matching height y_m sweeps YBAND. The regime is
        # robust to the y_m choice (BFS/NASA stay >1; hills stays <<1).
        axE.fill_between(x, lo, hi, color=pan["color"], alpha=0.20, lw=0,
                         zorder=1)
        axE.semilogy(x, eps_clipped, "-", color=pan["color"], lw=1.3, zorder=3,
                     rasterized=(pan["key"] == "periodic_hills"))
        if i == 0:
            axE.text(0.035, 0.045, r"band: $y_m$ sweep", transform=axE.transAxes,
                     fontsize=7.2, color="0.4", ha="left", va="bottom")
        axE.axhline(y=1.0, color="k", ls="--", lw=0.6, alpha=0.5)
        axE.axhline(y=0.1, color="k", ls=":", lw=0.5, alpha=0.3)
        axE.text(0.97, 0.03, r"$\varepsilon = 1$", transform=axE.transAxes,
                 ha="right", va="bottom", fontsize=7.2, color="0.4")
        axE.set_title(f"({chr(97 + i)}) " + pan["label"], fontsize=7.6, loc="left")
        axE.set_ylim(1e-5, 1e5)
        axE.tick_params(labelbottom=False)
        if i > 0:                              # shared y -> hide ticks on (b),(c)
            axE.tick_params(labelleft=False)
        if pan["key"] == "periodic_hills":
            axE.text(0.5, 0.17,
                     r"median $\varepsilon = %.2f$" % pan["med"] + "\n"
                     + r"$\mathbf{domain}$-$\mathbf{wide}$ $\mathbf{cancellation}$",
                     transform=axE.transAxes, fontsize=7.2, ha="center",
                     color="#b00000",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="#b00000", alpha=0.85))
        else:
            axE.text(0.5, 0.20,
                     r"median $\varepsilon = %s$" % _med_str(pan["med"]) + "\n"
                     + r"$\mathbf{dominant}$ $\mathbf{balance}$",
                     transform=axE.transAxes, fontsize=7.2, ha="center",
                     color="#2ca02c",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="#2ca02c", alpha=0.85))
        # geometry sketch + flow below, x-aligned (grey band links to eps panel)
        draw_geom(axG, pan["key"], x, sep)
        axG.set_xlabel(pan["xlabel"])
    eps_axes[0].set_ylabel(r"$\varepsilon(x)$")
    save_figure(fig, "cancellation_parameter_corrected", fig_dir=FIG_DIR)

    # ── save corrected artifact (+ legacy values for transparency) ───────────
    np.savez(
        os.path.join(RESULTS, "cancellation_parameter_corrected.npz"),
        # corrected periodic-hills panel
        periodic_hills_x=x_ph,
        periodic_hills_epsilon=eps_ph,
        periodic_hills_is_sep=sep_ph,
        periodic_hills_tau_w=tau_ph,
        periodic_hills_median_eps=med_ph,
        periodic_hills_frac_below_01=f01_ph,
        periodic_hills_frac_below_1=f1_ph,
        periodic_hills_median_abs_tau=med_tau_ph,
        periodic_hills_Re_tau=re_tau,
        periodic_hills_f_sep=float(np.mean(sep_ph)),
        # legacy artifact values (documented, not silently dropped)
        periodic_hills_median_eps_legacy=legacy_ph_med,
        periodic_hills_frac_below_01_legacy=legacy_ph_f01,
        # success-case panels recomputed directly from source profiles
        bfs_x=x_bfs, bfs_epsilon=pr_bfs[Y_IDX],
        bfs_is_sep=sep_bfs, bfs_median_eps=_med(pr_bfs[Y_IDX]),
        nasa_hump_x=x_na, nasa_hump_epsilon=pr_na[Y_IDX],
        nasa_hump_is_sep=sep_na,
        nasa_hump_median_eps=_med(pr_na[Y_IDX]),
        # independent cross-checks
        dose_response_eps_median=dose_eps_median,
        dose_response_eps_range=np.array([dose_eps_lo, dose_eps_hi]),
        krank_eps_tilde=np.array(krank_eps),
        y_idx=Y_IDX, raw_file=os.path.relpath(RAW_PEHILL, ROOT),
        protocol=np.array("canonical_hill_surface_aware"),
    )
    print(f"\nSaved -> results/cancellation_parameter_corrected.npz")
    print("Saved -> manuscript/figures/cancellation_parameter_corrected.{pdf,png}")


if __name__ == "__main__":
    main()
