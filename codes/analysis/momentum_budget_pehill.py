#!/usr/bin/env python3
"""
Full 2D x-momentum budget for periodic hills DNS (Xiao et al. 2020).

Computes every term in the RANS x-momentum equation:
  U dU/dx + V dU/dy = -dp/dx + nu(d2U/dx2 + d2U/dy2) - d<uu>/dx - d<uv>/dy

Handles the hill geometry by masking solid regions and using safe
derivatives that only use fluid-side stencils.
"""
import numpy as np
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'utils')))
# The materialised vendor subset predates the shear-stress file.  Use the
# project-owned Xiao raw source repaired by WP0, which contains rms_files2.dat.
RAW_DIR = os.path.normpath(os.path.join(
    _SCRIPT_DIR, '..', 'raw_data', 'geometry_driven',
    'xiao_pehill_parameterized', 'pehill-5-cases-DNS',
    'case_1p0', 'dns-data'))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RE_B = 5600.0
NU = 1.0 / RE_B

print("=" * 80)
print("FULL 2D MOMENTUM BUDGET — Periodic Hills DNS")
print("=" * 80)

# =====================================================================
# 1. Load and reshape DNS data
# =====================================================================
print("\nLoading DNS data...")
mean_data = np.loadtxt(os.path.join(RAW_DIR, 'mean_files.dat'))
rms1_data = np.loadtxt(os.path.join(RAW_DIR, 'rms_files1.dat'))
rms2_data = np.loadtxt(os.path.join(RAW_DIR, 'rms_files2.dat'))

# Correct column mapping: x, y, U, V, W, p
x_flat = mean_data[:, 0]
y_flat = mean_data[:, 1]
U_flat = mean_data[:, 2]
V_flat = mean_data[:, 3]
p_flat = mean_data[:, 5]

# Documented Xiao schema: normal stresses in rms_files1 and shear stresses in
# rms_files2.  The previous rms_files1[:,5] read was a pressure statistic.
uu_flat = rms1_data[:, 2]
uv_flat = rms2_data[:, 2]

x_unique = np.unique(x_flat)
y_unique = np.unique(y_flat)
nx, ny = len(x_unique), len(y_unique)
print(f"Grid: {nx} x {ny}")

# Data is y-major: for each y, all x values are listed
U = U_flat.reshape(ny, nx).T    # shape (nx, ny)
V = V_flat.reshape(ny, nx).T
p = p_flat.reshape(ny, nx).T
uu = uu_flat.reshape(ny, nx).T
uv = uv_flat.reshape(ny, nx).T

# Grid spacings (uniform in x, stretched in y)
dx_val = x_unique[1] - x_unique[0]  # uniform
dy_arr = np.diff(y_unique)

print(f"U range: [{U.min():.4f}, {U.max():.4f}]")
print(f"V range: [{V.min():.4f}, {V.max():.4f}]")
print(f"p range: [{p.min():.4f}, {p.max():.4f}]")
print(f"uv range: [{uv.min():.6f}, {uv.max():.6f}]")

# =====================================================================
# 2. Build fluid mask: identify wall and solid
# =====================================================================
speed = np.sqrt(U**2 + V**2)
fluid = speed > 1e-10  # True for fluid points

# Find wall index at each x (first fluid point in y-direction)
wall_j = np.full(nx, ny, dtype=int)  # default: no wall found
for i in range(nx):
    nz = np.where(fluid[i, :])[0]
    if len(nz) > 0:
        wall_j[i] = nz[0]

# Interior fluid mask: exclude 2 grid cells from the wall surface
# to avoid finite-difference artifacts at the solid-fluid boundary
interior = np.zeros((nx, ny), dtype=bool)
for i in range(nx):
    if wall_j[i] + 3 < ny:
        interior[i, wall_j[i] + 3:] = True
# Also need fluid neighbors in x for periodic derivatives
# Exclude points where either x-neighbor is solid
for i in range(nx):
    ip = (i + 1) % nx
    im = (i - 1) % nx
    for j in range(ny):
        if interior[i, j] and not (fluid[ip, j] and fluid[im, j]):
            interior[i, j] = False

n_int = interior.sum()
print(f"Interior fluid points: {n_int}/{nx*ny} ({100*n_int/(nx*ny):.1f}%)")

# =====================================================================
# 3. Compute derivatives with periodic x, non-periodic y
# =====================================================================
print("\nComputing derivatives...")

def ddx_p(f):
    """Central difference in x, periodic."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx_val)

def d2dx2_p(f):
    """Second derivative in x, periodic."""
    return (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dx_val**2)

def ddy_nonuniform(f):
    """Central difference in y with non-uniform spacing."""
    dfdy = np.zeros_like(f)
    for j in range(1, ny-1):
        hp = y_unique[j+1] - y_unique[j]
        hm = y_unique[j] - y_unique[j-1]
        dfdy[:, j] = (f[:, j+1]*hm/(hp*(hp+hm)) +
                       f[:, j]*(hp-hm)/(hp*hm) -
                       f[:, j-1]*hp/(hm*(hp+hm)))
    # Boundaries: one-sided
    dfdy[:, 0] = (f[:, 1] - f[:, 0]) / dy_arr[0]
    dfdy[:, -1] = (f[:, -1] - f[:, -2]) / dy_arr[-1]
    return dfdy

def d2dy2_nonuniform(f):
    """Second derivative in y with non-uniform spacing."""
    d2f = np.zeros_like(f)
    for j in range(1, ny-1):
        hp = y_unique[j+1] - y_unique[j]
        hm = y_unique[j] - y_unique[j-1]
        d2f[:, j] = 2*(f[:, j+1]/(hp*(hp+hm)) - f[:, j]/(hp*hm) + f[:, j-1]/(hm*(hp+hm)))
    return d2f

dUdx = ddx_p(U)
dUdy = ddy_nonuniform(U)
d2Udx2 = d2dx2_p(U)
d2Udy2 = d2dy2_nonuniform(U)
dpdx = ddx_p(p)
duudx = ddx_p(uu)
duvdy = ddy_nonuniform(uv)

# =====================================================================
# 4. Momentum budget terms
# =====================================================================
print("Computing budget terms...")

convection = U * dUdx + V * dUdy
pressure_grad = -dpdx
viscous_y = NU * d2Udy2
viscous_x = NU * d2Udx2
reynolds_y = -duvdy
reynolds_x = -duudx
residual = convection - pressure_grad - viscous_y - viscous_x - reynolds_y - reynolds_x

# Statistics on interior fluid points
def stats(field, name, mask):
    vals = field[mask]
    if len(vals) == 0:
        return
    print(f"  {name:30s}: RMS={np.sqrt((vals**2).mean()):.4e}, "
          f"range=[{vals.min():.4e}, {vals.max():.4e}]")

print("\nBudget terms (interior fluid, excluding near-wall 3 cells):")
stats(convection, "Convection (UdU/dx + VdU/dy)", interior)
stats(pressure_grad, "Pressure (-dp/dx)", interior)
stats(viscous_y, "Viscous y (nu d2U/dy2)", interior)
stats(viscous_x, "Viscous x (nu d2U/dx2)", interior)
stats(reynolds_y, "Reynolds y (-d<uv>/dy)", interior)
stats(reynolds_x, "Reynolds x (-d<uu>/dx)", interior)
stats(residual, "RESIDUAL", interior)

rms_res = np.sqrt((residual[interior]**2).mean())
rms_max = max(
    np.sqrt((convection[interior]**2).mean()),
    np.sqrt((pressure_grad[interior]**2).mean()),
    np.sqrt((reynolds_y[interior]**2).mean()),
)
print(f"\nBudget closure: RMS(residual)/RMS(largest) = {rms_res/rms_max:.4f}")

# =====================================================================
# 5. Near-wall analysis with careful masking
# =====================================================================
print("\n" + "=" * 80)
print("NEAR-WALL ANALYSIS (wall-adjacent fluid, 3-10 cells above wall)")
print("=" * 80)

# For near-wall analysis, use points 3-10 cells above the wall
# (skip first 2 cells to avoid interface artifacts)
near_wall = np.zeros((nx, ny), dtype=bool)
for i in range(nx):
    j0 = wall_j[i] + 3
    j1 = wall_j[i] + 10
    if j0 < ny and j1 < ny:
        near_wall[i, j0:j1] = True
        # Also need x-neighbors to be fluid
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(j0, j1):
            if not (fluid[ip, j] and fluid[im, j]):
                near_wall[i, j] = False

n_nw = near_wall.sum()
print(f"Near-wall analysis points: {n_nw}")

print("\nNear-wall budget terms:")
stats(convection, "Convection", near_wall)
stats(pressure_grad, "Pressure", near_wall)
stats(viscous_y, "Viscous y", near_wall)
stats(viscous_x, "Viscous x", near_wall)
stats(reynolds_y, "Reynolds y", near_wall)
stats(reynolds_x, "Reynolds x", near_wall)
stats(residual, "RESIDUAL", near_wall)

# =====================================================================
# 6. Streamwise variation of budget terms (near-wall averaged)
# =====================================================================
print("\n" + "=" * 80)
print("STREAMWISE VARIATION OF NEAR-WALL BUDGET")
print("=" * 80)

conv_rms_x = np.zeros(nx)
pres_rms_x = np.zeros(nx)
visc_rms_x = np.zeros(nx)
reyn_rms_x = np.zeros(nx)
resid_rms_x = np.zeros(nx)

for i in range(nx):
    nw = near_wall[i, :]
    if nw.sum() < 2:
        continue
    conv_rms_x[i] = np.sqrt((convection[i, nw]**2).mean())
    pres_rms_x[i] = np.sqrt((pressure_grad[i, nw]**2).mean())
    visc_rms_x[i] = np.sqrt((viscous_y[i, nw]**2).mean())
    reyn_rms_x[i] = np.sqrt((reynolds_y[i, nw]**2).mean())
    resid_rms_x[i] = np.sqrt((residual[i, nw]**2).mean())

# Ratio: |convection| / (|viscous_y| + |reynolds_y|)
denom = visc_rms_x + reyn_rms_x
ratio_x = np.where(denom > 1e-15, conv_rms_x / denom, 0)

# DNS wall shear stress
tau_w_dns = np.zeros(nx)
for i in range(nx):
    j0 = wall_j[i]
    if j0 + 1 < ny:
        dy0 = y_unique[j0 + 1] - y_unique[j0]
        if dy0 > 1e-15:
            tau_w_dns[i] = NU * (U[i, j0 + 1] - U[i, j0]) / dy0

sep_mask = tau_w_dns < 0
n_sep = sep_mask.sum()
if n_sep > 0:
    print(f"Separation: x in [{x_unique[sep_mask].min():.3f}, {x_unique[sep_mask].max():.3f}], {n_sep}/{nx} stations")

att_valid = (~sep_mask) & (conv_rms_x > 0) & (denom > 0)
sep_valid = sep_mask & (conv_rms_x > 0) & (denom > 0)
if att_valid.sum() > 0:
    print(f"Attached region — Conv/Diff ratio: mean={ratio_x[att_valid].mean():.4f}, max={ratio_x[att_valid].max():.4f}")
if sep_valid.sum() > 0:
    print(f"Separated region — Conv/Diff ratio: mean={ratio_x[sep_valid].mean():.4f}, max={ratio_x[sep_valid].max():.4f}")

# Key result: what determines tau_w at the wall?
# ODE wall model: tau_w = f(dp/dx, U_match) — ignores convection
# Full RANS: tau_w is determined by ALL terms including convection
print("\n--- KEY INSIGHT ---")
print(f"Near-wall convection RMS: {np.sqrt((convection[near_wall]**2).mean()):.4e}")
print(f"Near-wall pressure RMS:   {np.sqrt((pressure_grad[near_wall]**2).mean()):.4e}")
print(f"Near-wall viscous_y RMS:  {np.sqrt((viscous_y[near_wall]**2).mean()):.4e}")
print(f"Near-wall reynolds_y RMS: {np.sqrt((reynolds_y[near_wall]**2).mean()):.4e}")

# =====================================================================
# 7. Save results
# =====================================================================
print("\nSaving results...")
np.savez(os.path.join(RESULTS_DIR, 'momentum_budget_pehill.npz'),
    x_unique=x_unique, y_unique=y_unique,
    U=U, V=V, p=p, uu=uu, uv=uv,
    convection=convection, pressure_grad=pressure_grad,
    viscous_y=viscous_y, viscous_x=viscous_x,
    reynolds_y=reynolds_y, reynolds_x=reynolds_x,
    residual=residual,
    fluid=fluid, interior=interior, near_wall=near_wall,
    wall_j=wall_j, tau_w_dns=tau_w_dns,
    conv_rms_x=conv_rms_x, pres_rms_x=pres_rms_x,
    visc_rms_x=visc_rms_x, reyn_rms_x=reyn_rms_x,
    resid_rms_x=resid_rms_x, ratio_x=ratio_x,
    sep_mask=sep_mask,
)
outpath = os.path.join(RESULTS_DIR, 'momentum_budget_pehill.npz')
print(f"Saved: {outpath}")
print("DONE")
