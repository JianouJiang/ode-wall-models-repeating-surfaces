#!/usr/bin/env python3
"""The three wall-traction references used to score the R2-m4 / R3-2 ladder.

Written 2026-08-25 after the operator withdrew the deposited Xiao wall-gradient
reconstruction as a SCORING reference (independent audit:
work_progress/archer2_campaign_20260823/TRUTH_REFERENCE_AUDIT_V/REPORT.md).

What is wrong is the ESTIMATOR, not the archive.  The Xiao 512x257 velocity
archive is the same flow on the same geometry (crest bulk 0.999, U_max within
0.41% of MGLET at all ten stations), but its wall spacing is 0.0093-0.0136 H
(fit points at y+ 2.4-44) against the MGLET DNS deposit's 0.0010-0.0015 H, so a
four-point through-origin LINEAR fit is unconverged and biased low, hardest
where u_tau is largest.  Running the same estimator on MGLET's own profiles
resampled to the archive spacing reproduces both the RMS deficit and the sign
flips at x/H = 5 and 7.

    A  xiao_linear4_deposited  the superseded reference (kept for provenance and
                               for reproducing the 2026-08-23 numbers verbatim)
    B  mglet_deposited         PRIMARY: Peller & Manhart MGLET DNS bottom-wall
                               tau_w, ERCOFTAC UFR3-30, trailing plot-axis
                               placeholder rows stripped
    C  xiao_cubic6_repaired    SENSITIVITY BRACKET: curvature-aware through-origin
                               cubic on the first six fluid points of the same
                               Xiao archive columns (validated against MGLET's own
                               tau at the Xiao spacing: relative RMS error 0.264)

B and C still differ by ~37% in RMS; both are reported, B is primary.
The ladder's INPUTS (u_m, dp/ds, the convection profile at y_m = 0.0935 H) come
from well-resolved interior data and are unaffected by any of this.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis"))
sys.path.insert(0, str(ROOT / "codes" / "openfoam"))
from harvest_m13_highre import _strip_mglet_placeholders  # noqa: E402  (operator instruction)
from make_xiao_dns_wmles_case import HALF_WIDTH, xiao_profile  # noqa: E402

LX = 9.0
NU = 1.0 / 5600.0
CUBIC_DEGREE = 3
CUBIC_POINTS = 6
MGLET_WALL = ROOT / "codes/raw_data/periodic_hill_ufr3_30/ercoftac_ufr3_30/UFR3-30_data-NP-Re5600-DNS2-11.dat"
XIAO_ARCHIVE = ROOT / "codes/results/periodic_hills_case_1p0_wall_profiles_corrected.npz"
LABELS = {
    "A_xiao_linear4_deposited": "Xiao archive, deposited 4-point through-origin linear fit (WITHDRAWN as a scoring reference)",
    "B_mglet_deposited": "Peller & Manhart MGLET DNS Re=5600, deposited bottom-wall tau_w (ERCOFTAC UFR3-30) [PRIMARY]",
    "C_xiao_cubic6_repaired": "Xiao archive, curvature-aware through-origin cubic on the first six fluid points [BRACKET]",
}


def wall_tangent(x):
    h = np.asarray([xiao_profile(v) if v <= HALF_WIDTH else
                    xiao_profile(LX - v) if v >= LX - HALF_WIDTH else 0.0 for v in x])
    dx = float(np.median(np.diff(x)))
    slope = (8.0 * (np.roll(h, -1) - np.roll(h, 1)) - (np.roll(h, -2) - np.roll(h, 2))) / (12.0 * dx)
    mag = np.sqrt(1.0 + slope ** 2)
    return h, slope, 1.0 / mag, slope / mag


def _through_origin_slope(n, u, degree=CUBIC_DEGREE):
    """du/dn at the wall from u = a1 n + a2 n^2 + ... (no constant term)."""
    design = np.vstack([np.asarray(n, float) ** (k + 1) for k in range(degree)]).T
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(u, float), rcond=None)
    return float(coefficients[0])


def xiao_station_tractions():
    """(x, tau_linear4, tau_cubic6) from the Xiao archive's own columns."""
    d = np.load(XIAO_ARCHIVE)
    x, y, U, V = (np.asarray(d[k], float) for k in ("x", "y", "U", "V"))
    _, _, tx, ty = wall_tangent(x)
    linear = np.empty(x.size)
    cubic = np.empty(x.size)
    for i in range(x.size):
        ok = np.isfinite(y[i]) & np.isfinite(U[i]) & np.isfinite(V[i])
        n = y[i, ok] - y[i, ok][0]
        ut = U[i, ok] * tx[i] + V[i, ok] * ty[i]
        four = slice(1, 5)
        linear[i] = NU * float(np.sum(n[four] * ut[four]) / np.sum(n[four] ** 2)) / tx[i]
        six = slice(1, CUBIC_POINTS + 1)
        cubic[i] = NU * _through_origin_slope(n[six], ut[six]) / tx[i]
    return x, linear, cubic


def references():
    """{key: (phase, tau_s, label)} for the three references."""
    x, linear, cubic = xiao_station_tractions()
    phase = np.mod((x - x.min()) / LX, 1.0)
    mglet = _strip_mglet_placeholders(np.loadtxt(MGLET_WALL))
    return {
        "A_xiao_linear4_deposited": (phase, linear, LABELS["A_xiao_linear4_deposited"]),
        "B_mglet_deposited": (np.mod(mglet[:, 0] / LX, 1.0), mglet[:, 1], LABELS["B_mglet_deposited"]),
        "C_xiao_cubic6_repaired": (phase, cubic, LABELS["C_xiao_cubic6_repaired"]),
    }


PRIMARY = "B_mglet_deposited"
BRACKET = "C_xiao_cubic6_repaired"
SUPERSEDED = "A_xiao_linear4_deposited"
