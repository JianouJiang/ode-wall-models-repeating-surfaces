#!/usr/bin/env python3
"""
Figure: Parity plot — predicted tau_w vs DNS tau_w for ODE model and Spalding.
Each dataset normalized by its own scale (mean |tau_w|) so different geometries
are comparable. Two panels: (a) ODE model, (b) Signed Spalding.

Local copy from universal_wall_function with alpha reduced to 0.5 for better
marker visibility in dense clusters.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import numpy as np
from scipy.optimize import brentq
from plotting_utils import setup_style, save_figure
import matplotlib.pyplot as plt

setup_style(font_size=10, use_tex=True)

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'vendor',
                       'universal_wall_function', 'codes', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')

# ── ODE model ────────────────────────────────────────────────────────────────
KAPPA, A_PLUS, B_SPALD, N_GRID = 0.41, 26.0, 5.0, 500

def solve_ode_for_U(tau_w, y_match, dp_dx, nu, n_grid=N_GRID):
    u_tau = np.sqrt(max(abs(tau_w), 1e-30))
    eta = np.linspace(0, 1, n_grid)
    y = y_match * eta**1.5
    y_plus = y * u_tau / nu
    D = 1.0 - np.exp(-y_plus / A_PLUS)
    l_m = KAPPA * y * D
    l_m2 = l_m * l_m
    F = tau_w + dp_dx * y
    absF, signF = np.abs(F), np.sign(F)
    disc = nu**2 + 4.0 * l_m2 * absF
    viscous = l_m2 < 1e-30
    l_m2_safe = np.where(viscous, 1.0, l_m2)
    absS = (-nu + np.sqrt(disc)) / (2.0 * l_m2_safe)
    absS = np.where(viscous, absF / nu, absS)
    S = signF * absS
    dy = np.diff(y)
    return np.sum(0.5 * (S[:-1] + S[1:]) * dy)

def predict_tau_w_ode(U_match, y_match, dp_dx, nu):
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

def spalding_yplus(u_plus):
    ku = KAPPA * u_plus
    return u_plus + np.exp(-KAPPA * B_SPALD) * (np.exp(ku) - 1 - ku - ku**2/2 - ku**3/6)

def predict_tau_w_spalding_signed(U_match, y_match, nu):
    if abs(U_match) < 1e-15 or abs(y_match) < 1e-15:
        return 0.0
    def residual(u_tau):
        if u_tau < 1e-15:
            return -1.0
        return spalding_yplus(abs(U_match) / u_tau) - y_match * u_tau / nu
    try:
        u_tau = brentq(residual, 1e-10, abs(U_match), maxiter=200)
        return np.sign(U_match) * u_tau**2
    except Exception:
        return 0.0

# ── Dataset definitions ──────────────────────────────────────────────────────
GEOMETRY_GROUPS = {
    'Channel': ['channel_Re180', 'channel_Re395', 'channel_Re550',
                'channel_Re1000', 'channel_Re2000', 'channel_Re5200'],
    'Pipe': ['pipe_Re180', 'pipe_Re360', 'pipe_Re550', 'pipe_Re1000'],
    'TBL (ZPG)': ['tbl_Re670', 'tbl_Re1000', 'tbl_Re1410', 'tbl_Re2000',
                  'tbl_Re2540', 'tbl_Re3030', 'tbl_Re3270', 'tbl_Re3630', 'tbl_Re4060'],
    'TBL (APG)': ['apg_tbl_kth_b1n', 'apg_tbl_kth_b2n',
                  'apg_tbl_kth_m13n', 'apg_tbl_kth_m16n', 'apg_tbl_kth_m18n'],
    'Couette': ['couette_Re93', 'couette_Re220', 'couette_Re500'],
    'Sep.\\ bubble': ['separation_bubble_caseC'],
    'BFS': ['bfs_Re13700'],
    'NASA hump': ['nasa_hump_Re936000'],
}

GROUP_COLORS = {
    'Channel': '#1f77b4', 'Pipe': '#ff7f0e', 'TBL (ZPG)': '#2ca02c',
    'TBL (APG)': '#d62728', 'Couette': '#9467bd', 'Sep.\\ bubble': '#8c564b',
    'BFS': '#e377c2', 'NASA hump': '#17becf',
}
GROUP_MARKERS = {
    'Channel': 'o', 'Pipe': 's', 'TBL (ZPG)': '^', 'TBL (APG)': 'v',
    'Couette': 'D', 'Sep.\\ bubble': 'p', 'BFS': '*', 'NASA hump': 'h',
}

Y_IDX = 10

# ── Compute predictions ─────────────────────────────────────────────────────
results = {}
for group, datasets in GEOMETRY_GROUPS.items():
    tau_dns_all, tau_ode_all, tau_spal_all = [], [], []
    for ds_name in datasets:
        fpath = os.path.join(RESULTS, f'{ds_name}_wall_profiles.npz')
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        y, U, tau_w = d['y'], d['U'], d['tau_w']
        dp_dx, nu = d['dp_dx'], d['nu']
        n_prof = len(tau_w)
        scale = np.mean(np.abs(tau_w))
        if scale < 1e-30:
            scale = 1.0

        for i in range(n_prof):
            yi = y[i] if y.ndim == 2 else y
            Ui = U[i] if U.ndim == 2 else U
            if Y_IDX >= len(yi):
                continue
            y_m, U_m, nu_i = yi[Y_IDX], Ui[Y_IDX], float(nu[i])
            if y_m <= 0 or np.isnan(U_m):
                continue
            dp_i = float(dp_dx[i])
            tw_ode = predict_tau_w_ode(U_m, y_m, dp_i, nu_i)
            tw_spal = predict_tau_w_spalding_signed(U_m, y_m, nu_i)
            if np.isnan(tw_ode):
                continue
            tau_dns_all.append(float(tau_w[i]) / scale)
            tau_ode_all.append(tw_ode / scale)
            tau_spal_all.append(tw_spal / scale)

    if tau_dns_all:
        results[group] = (
            np.array(tau_dns_all), np.array(tau_ode_all),
            np.array(tau_spal_all),
        )
        print(f'{group}: {len(tau_dns_all)} pts')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

for ax_idx, (ax, model_name, col_idx) in enumerate([
    (axes[0], 'ODE model', 1),
    (axes[1], 'Signed Spalding', 2),
]):
    all_dns = np.concatenate([v[0] for v in results.values()])
    all_pred = np.concatenate([v[col_idx] for v in results.values()])
    vmin = min(all_dns.min(), all_pred.min())
    vmax = max(all_dns.max(), all_pred.max())
    pad = 0.05 * (vmax - vmin)
    lims = (vmin - pad, vmax + pad)

    ax.plot(lims, lims, 'k-', lw=0.8, zorder=0)
    ax.axhline(0, color='0.7', lw=0.5, ls='--', zorder=0)
    ax.axvline(0, color='0.7', lw=0.5, ls='--', zorder=0)

    for group in GEOMETRY_GROUPS:
        if group not in results:
            continue
        tau_dns, tau_ode, tau_spal = results[group]
        tau_pred = tau_ode if col_idx == 1 else tau_spal
        ms = 35 if group in ('BFS', 'NASA hump') else 18
        ax.scatter(tau_dns, tau_pred,
                   c=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                   s=ms, alpha=0.5, edgecolors='none',
                   label=f'{group}' if ax_idx == 0 else None, zorder=2)

    ax.set_xlabel(r'$\tau_{w,\mathrm{DNS}} / \langle|\tau_w|\rangle$')
    ax.set_ylabel(r'$\tau_{w,\mathrm{pred}} / \langle|\tau_w|\rangle$')
    ax.set_title(f'({chr(97 + ax_idx)}) {model_name}', fontsize=10)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))

fig.tight_layout(rect=[0, 0.08, 1, 1])
save_figure(fig, 'fig_parity', fig_dir=FIG_DIR)
print(f'\nSaved to {FIG_DIR}/fig_parity.pdf')
