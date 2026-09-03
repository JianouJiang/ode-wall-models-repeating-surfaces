#!/usr/bin/env python3
"""
Figure: Velocity profile comparisons — DNS vs ODE reconstruction.

Shows U/U_ref vs y/y_m at selected stations for three geometries:
  (a–c) BFS: upstream attached, separation onset, mid-recirculation
  (d–f) NASA hump: upstream attached, separation, recovery
  (g–i) Periodic hills: attached, separation, mid-recirculation

Demonstrates that the ODE reconstructs profiles well when ε ~ O(1)
but catastrophically fails when ε ≪ 1.
"""

import sys, os
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_paths import get_uwf_results_dir
from plotting_utils import setup_style, save_figure
import matplotlib.pyplot as plt

# The figure prints at 0.836\textwidth = 5.40 in.  Draw the canvas at that
# size so every point size on it is the point size that reaches the page:
# the previous 7.0 in canvas was scaled to 0.50 by \includegraphics, which
# halved every label and made the legend illegible.
setup_style(font_size=8, use_tex=True)
UWF_DIR = get_uwf_results_dir()
NEW_DIR = os.path.join(os.path.dirname(__file__), '..', 'new_data_download')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

# ── ODE solver (copied from fig_parity.py for self-containedness) ──────────
KAPPA, A_PLUS, N_GRID = 0.41, 26.0, 500

def solve_ode_profile(tau_w, y_arr, dp_dx, nu):
    """
    Given tau_w, compute the ODE velocity profile at wall-normal locations y_arr.
    Returns U(y) array.
    """
    u_tau = np.sqrt(max(abs(tau_w), 1e-30))
    n = len(y_arr)
    U = np.zeros(n)
    for j in range(1, n):
        # Integrate from y[j-1] to y[j]
        y_mid = 0.5 * (y_arr[j - 1] + y_arr[j])
        dy = y_arr[j] - y_arr[j - 1]
        y_plus = y_mid * u_tau / nu
        D = 1.0 - np.exp(-y_plus / A_PLUS)
        l_m = KAPPA * y_mid * D
        l_m2 = l_m ** 2
        F = tau_w + dp_dx * y_mid
        absF = abs(F)
        signF = np.sign(F)
        if l_m2 < 1e-30:
            absS = absF / nu
        else:
            disc = nu ** 2 + 4.0 * l_m2 * absF
            absS = (-nu + np.sqrt(disc)) / (2.0 * l_m2)
        S = signF * absS
        U[j] = U[j - 1] + S * dy
    return U


def solve_ode_for_U(tau_w, y_match, dp_dx, nu, n_grid=N_GRID):
    """Integrate ODE from y=0 to y=y_match. Returns U(y_match)."""
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
    """Root-find tau_w."""
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


def get_ode_profile(y_arr, y_match_idx, dp_dx, nu, U_dns):
    """Compute ODE-predicted profile for a given DNS station."""
    y_m = y_arr[y_match_idx]
    U_m = U_dns[y_match_idx]
    tau_w_pred = predict_tau_w(U_m, y_m, dp_dx, nu)
    if np.isnan(tau_w_pred):
        return None, np.nan
    # Build fine ODE grid up to y_match
    n_fine = 200
    eta_fine = np.linspace(0, 1, n_fine)
    y_fine = y_m * eta_fine ** 1.5
    U_ode = solve_ode_profile(tau_w_pred, y_fine, dp_dx, nu)
    return (y_fine, U_ode), tau_w_pred


# ── Data loading ──────────────────────────────────────────────────
Y_IDX = 10

print("Loading datasets...")

# BFS
d_bfs = np.load(os.path.join(UWF_DIR, 'bfs_Re13700_wall_profiles.npz'),
                allow_pickle=True)
# NASA hump
d_hump = np.load(os.path.join(UWF_DIR, 'nasa_hump_Re936000_wall_profiles.npz'),
                 allow_pickle=True)
# Periodic hills — hill-surface-aware extraction (eq:hillsurface); the legacy
# y=0-pinned file samples velocity inside the solid hill on the hill faces.
d_pehill = np.load(os.path.join(RESULTS_DIR,
                   'periodic_hills_case_1p0_wall_profiles_corrected.npz'),
                   allow_pickle=True)

# ── Station selection ─────────────────────────────────────────────
# BFS: 46 stations, x in [-7.3, 15.3], step at x=0
#   - upstream attached (x ~ -2), separation onset (x ~ 1), mid-recirculation (x ~ 3)
bfs_x = d_bfs['x']
bfs_sep = d_bfs['is_separated']
bfs_stations = []
# Find closest indices
for target_x in [-2.0, 1.0, 3.0]:
    idx = np.argmin(np.abs(bfs_x - target_x))
    bfs_stations.append(idx)

# NASA hump: 51 stations, x in [-2.0, 1.6], hump at x~0.5-1.2
#   - upstream (x ~ 0.0), separation (x ~ 0.75), recovery (x ~ 1.3)
hump_x = d_hump['x']
hump_sep = d_hump['is_separated']
for target_x in [0.0, 0.75, 1.3]:
    idx = np.argmin(np.abs(hump_x - target_x))

hump_stations = []
for target_x in [0.0, 0.75, 1.3]:
    idx = np.argmin(np.abs(hump_x - target_x))
    hump_stations.append(idx)

# Periodic hills: 512 stations, x in [0, 9], hill at x ~ 0-1 and x ~ 8-9.
# Both admissible references separate at x/H ~ 0.18 and reattach at 5.14 (B,
# primary) / 4.72 (C, bracket), so ALL THREE stations below are inside the
# bubble; the row once called x ~ 4.5 "attached", which its own red badge, both
# references and the quoted reattachment spread (4.47-5.14) all contradict.
#   - early recirculation (x ~ 1.5), mid (x ~ 3), near reattachment (x ~ 4.5)
pehill_x = d_pehill['x']
pehill_stations = []
for target_x in [1.5, 3.0, 4.5]:
    idx = np.argmin(np.abs(pehill_x - target_x))
    pehill_stations.append(idx)

# The periodic-hill row is scored against the PUBLISHED reference artifact, not
# against the traction stored in the velocity archive.  That stored traction --
# and the ``is_separated`` flag defined from its sign -- come from a
# through-origin linear fit of the archive's first four points, which the
# independent audit withdrew as under-resolved at that wall spacing.  The
# annotation therefore uses the full-wall DNS traction (primary) and reports
# the same-simulation cubic as a bracket; the archive's own velocity data,
# which is what the panels actually plot, are untouched and not in question.
_refs = np.load(os.path.join(RESULTS_DIR, 'wall_traction_references_20260825.npz'))


def _pehill_reference(x_query, name):
    """Reference wall traction at x/H, interpolated periodically over 0..9."""
    ph = np.asarray(_refs[f'{name}__phase'], float)
    tw = np.asarray(_refs[f'{name}__tau'], float)
    o = np.argsort(ph)
    ph, tw = ph[o], tw[o]
    q = np.mod(np.asarray(x_query, float) / 9.0, 1.0)
    return np.interp(q, np.r_[ph - 1.0, ph, ph + 1.0], np.r_[tw, tw, tw])


PEHILL_TAU_B = _pehill_reference(pehill_x, 'B_mglet')
PEHILL_TAU_C = _pehill_reference(pehill_x, 'C_xiao_repaired_cubic6')

# ── Compute ODE profiles ─────────────────────────────────────────
print("Computing ODE profiles...")

datasets = [
    ('BFS', d_bfs, bfs_stations, bfs_x,
     [r'Attached ($x/H = %.1f$)', r'Near separation ($x/H = %.1f$)',
      r'Recirculation ($x/H = %.1f$)']),
    ('NASA hump', d_hump, hump_stations, hump_x,
     [r'Attached ($x/c = %.2f$)', r'Separation ($x/c = %.2f$)',
      r'Recovery ($x/c = %.2f$)']),
    ('Periodic hills', d_pehill, pehill_stations, pehill_x,
     [r'Recirculation ($x/h = %.1f$)', r'Recirculation ($x/h = %.1f$)',
      r'Near reattachment ($x/h = %.1f$)']),
]

all_profiles = []
for name, data, stations, x_arr, labels_fmt in datasets:
    geom_profiles = []
    for j, idx in enumerate(stations):
        y_dns = data['y'][idx]
        U_dns = data['U'][idx]
        dp_dx_i = float(data['dp_dx'][idx])
        nu_i = float(data['nu'][idx])
        if name == 'Periodic hills':
            tau_w_dns = float(PEHILL_TAU_B[idx])
            tau_w_bracket = float(PEHILL_TAU_C[idx])
        else:
            tau_w_dns = float(data['tau_w'][idx])
            tau_w_bracket = None
        is_sep_i = bool(tau_w_dns < 0.0)

        result, tau_w_pred = get_ode_profile(y_dns, Y_IDX, dp_dx_i, nu_i, U_dns)
        x_val = x_arr[idx]
        label_i = labels_fmt[j] % x_val

        geom_profiles.append({
            'y_dns': y_dns, 'U_dns': U_dns,
            'ode_result': result, 'tau_w_dns': tau_w_dns,
            'tau_w_bracket': tau_w_bracket,
            'tau_w_pred': tau_w_pred, 'is_sep': is_sep_i,
            'label': label_i, 'x_val': x_val,
            'dp_dx': dp_dx_i, 'nu': nu_i,
        })
        status = "SEP" if is_sep_i else "ATT"
        tw_err = abs(tau_w_pred - tau_w_dns) / max(abs(tau_w_dns), 1e-30)
        print(f"  {name} station {idx} (x={x_val:.2f}, {status}): "
              f"tau_w reference={tau_w_dns:.2e}, ODE={tau_w_pred:.2e}, "
              f"rel.err={tw_err:.1f}"
              + ("" if tau_w_bracket is None
                 else f" [bracket reference {tau_w_bracket:.2e}, "
                      f"rel.err={abs(tau_w_pred - tau_w_bracket) / max(abs(tau_w_bracket), 1e-30):.1f}]"))
    all_profiles.append(geom_profiles)


# ── Plot: 3×3 layout ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(5.4, 4.15))

# Model/reference colours follow the manuscript-wide comparison convention.
# Geometry is encoded by rows and titles rather than by colour, so the same
# visual channel always means the same thing across figures.
geom_labels = [
    r'BFS ($Re_H = 13\,700$)',
    r'NASA hump ($Re_c = 936\,000$)',
    r'Periodic hills ($Re_H = 5600$)',
]
C_TRUTH = '#e08214'
C_ODE = '#111111'

for row, (geom_profiles, geom_name) in enumerate(zip(all_profiles, geom_labels)):
    for col, prof in enumerate(geom_profiles):
        ax = axes[row, col]

        y_dns = prof['y_dns']
        U_dns = prof['U_dns']
        ode = prof['ode_result']

        # Normalize: y by y_m, U by |U(y_m)| for shape comparison
        y_m = y_dns[Y_IDX]
        U_ref = abs(U_dns[Y_IDX]) if abs(U_dns[Y_IDX]) > 1e-15 else 1.0

        # Plot DNS profile up to ~2 y_m
        mask = y_dns <= 2.5 * y_m
        ax.plot(U_dns[mask] / U_ref, y_dns[mask] / y_m,
                '-', color=C_TRUTH, lw=1.2,
                label='reference DNS/LES' if row == 0 and col == 0 else None)

        # Plot ODE profile
        if ode is not None:
            y_ode, U_ode = ode
            ax.plot(U_ode / U_ref, y_ode / y_m,
                    '--', color=C_ODE, lw=1.1,
                    label='pressure-gradient ODE' if row == 0 and col == 0 else None)

        # Matching point marker
        ax.plot(U_dns[Y_IDX] / U_ref, 1.0, 'o', color='0.5',
                ms=3, zorder=5, mfc='none', mew=0.7,
                label=r'$y_m$' if row == 0 and col == 0 else None)

        # Title with station info (panel tag merged in, one aligned line)
        tag = chr(97 + 3 * row + col)
        ax.set_title(f"({tag}) " + prof['label'], fontsize=7.2, loc='left')

        # Separation indicator
        if prof['is_sep']:
            ax.text(0.955, 0.93, r'$\tau_w < 0$', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.2, color='#d62728',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#fff0f0',
                              ec='#d62728', alpha=0.85, lw=0.5))

        # τ_w comparison
        tw_d = prof['tau_w_dns']
        tw_p = prof['tau_w_pred']
        tw_b = prof.get('tau_w_bracket')
        if abs(tw_d) > 1e-30:
            def _fmt(v):
                return ('%.0f' % v) if v > 10 else (('%.1f' % v) if v > 1 else ('%.2f' % v))
            rel_err = abs(tw_p - tw_d) / abs(tw_d)
            err_str = r'$|\Delta\tau_w|/|\tau_w| = %s$' % _fmt(rel_err)
            ax.text(0.97, 0.03, err_str, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=7.2, color='k',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.8))

        ax.set_ylim(0, 2.2)
        # Panels are 1.6 in wide in print; the default locator puts five
        # labels such as -1.00 -0.75 -0.50 -0.25 0.00 on that width and they
        # touch.  Cap the count instead of rotating or shrinking them.
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)

        if col == 0:
            ax.set_ylabel(r'$y / y_m$')
        if row == 2:
            ax.set_xlabel(r'$U / U(y_m)$')

    # Row label on the left; geometry is encoded textually, not by line colour.
    axes[row, 0].text(-0.32, 0.5, geom_name, transform=axes[row, 0].transAxes,
                      fontsize=7.2, rotation=90, va='center', ha='center',
                      fontweight='bold')


# Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
# 'outside lower center' is managed by constrained_layout, which RESERVES a
# band for the legend.  The previous bbox_to_anchor=(0.5, -0.02) placed it
# outside that management, so it printed on top of the bottom row's
# $U/U(y_m)$ axis labels.
fig.legend(handles, labels, loc='outside lower center', ncol=3,
           fontsize=7.2, frameon=False)

save_figure(fig, 'fig_velocity_profiles', fig_dir=FIG_DIR)
print(f'\nSaved to {FIG_DIR}/fig_velocity_profiles.pdf')
