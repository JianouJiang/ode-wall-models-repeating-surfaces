#!/usr/bin/env python3
"""
Figure: a-posteriori coupled WMLES of the periodic hill vs the Krank, Kronbichler
& Wall (2018) DNS reference (Re_H = 10595).

The a-priori study (rest of the paper) hands the wall model pristine DNS inputs.
Here the *equilibrium* wall model (nutUSpaldingWallFunction — a convection- and
pressure-gradient-dropping wall model, the broadest member of the class the paper
indicts) is two-way coupled inside a WALE LES and must supply tau_w from the
solver's own (imperfect) matching-cell velocity.  The question: does the
cancellation failure survive coupling?

Panels:
  (a) Bottom-wall C_f(x): DNS (orange) vs coupled WMLES (green).  Reattachment =
      last C_f sign change; the gap is the a-posteriori reattachment error.
  (b) Mean streamwise velocity profiles U/u_b at DNS stations: DNS (orange
      circles) vs coupled WMLES (green).

Colour scheme (paper convention): orange = ground truth (DNS),
green = equilibrium/Spalding wall model.  No fabrication: the script reads only
real DNS files and the OpenFOAM-produced samples; if the coupled run has not yet
produced mean fields it prints a notice and draws the DNS-only reference.
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "codes", "results")
CASE = os.path.join(ROOT, "codes", "openfoam", "pehill_wmles")
DNS_DIR = os.path.join(
    ROOT, "codes", "new_data_download", "geometry_driven",
    "krank_pehill_Re10595", "m1415670", "KKW_DNS_Periodic_Hill_Re10595")
FIG_DIR = os.path.join(ROOT, "manuscript", "figures")

C_DNS = "#ff7f0e"     # orange — ground truth
C_WM = "#2ca02c"      # green  — equilibrium (Spalding) wall model
U_B = 1.0

DNS_STATIONS = {  # filename suffix -> x/H
    "Station01": 0.05, "Station02": 0.50, "Station03": 1.00, "Station04": 2.00,
    "Station05": 3.00, "Station06": 4.00, "Station07": 5.00, "Station08": 6.00,
    "Station09": 7.00, "Station10": 8.00,
}
PROFILE_X = [0.50, 2.00, 4.00, 6.00]   # stations shown in panel (b)
PROFILE_SET = {0.50: "x0p50", 2.00: "x2p00", 4.00: "x4p00", 6.00: "x6p00"}


def load_dns_cf():
    f = os.path.join(DNS_DIR, "KKW_DNS_Periodic_Hill_Re10595_cf_cp_bottom.dat")
    rows = []
    with open(f) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            p = [t for t in s.replace(",", " ").split() if t]
            if len(p) >= 2:
                rows.append([float(p[0]), float(p[1])])
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def reattach(x, cf, lo=0.1, hi=7.0):
    m = (x >= lo) & (x <= hi)
    xm, cfm = x[m], cf[m]
    up = []
    for i in range(len(cfm) - 1):
        if cfm[i] < 0 and cfm[i + 1] >= 0:
            up.append(xm[i] - cfm[i] * (xm[i + 1] - xm[i]) / (cfm[i + 1] - cfm[i]))
    return up[-1] if up else np.nan


def load_dns_profile(suffix):
    f = os.path.join(DNS_DIR, f"KKW_DNS_Periodic_Hill_Re10595_{suffix}.dat")
    rows = []
    with open(f) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            p = [t for t in s.replace(",", " ").split() if t]
            if len(p) >= 2:
                rows.append([float(p[0]), float(p[1])])
    a = np.array(rows)
    return a[:, 0], a[:, 1]   # y/H, u/u_b


def load_wmles_profile(setname):
    """sets raw output: postProcessing/sampleProfiles/<t>/<set>_UMean.xy
    columns: y Ux Uy Uz."""
    base = os.path.join(CASE, "postProcessing", "sampleProfiles")
    if not os.path.isdir(base):
        return None
    times = sorted((float(d) for d in os.listdir(base)
                    if d.replace(".", "").isdigit()))
    if not times:
        return None
    t = times[-1]
    # OF10 raw setFormat writes "<set>.xy" (single field) or "<set>_<field>.xy"
    # (multiple fields); match either so the WMLES profiles are not silently
    # dropped (the figure would otherwise show DNS-only).
    sdir = os.path.join(base, f"{t:g}")
    cand = (glob.glob(os.path.join(sdir, f"{setname}_*.xy"))
            + glob.glob(os.path.join(sdir, f"{setname}.xy")))
    if not cand:
        return None
    a = np.loadtxt(cand[0])
    return a[:, 0], a[:, 1]   # y, Ux


def main():
    apz = os.path.join(RESULTS, "aposteriori_wmles_pehill.npz")
    d = np.load(apz, allow_pickle=True)
    present = bool(d["wmles_present"])
    dns_x, dns_cf = load_dns_cf()
    dns_xr = float(d["dns_x_reatt"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # ── panel (a): C_f(x) ───────────────────────────────────────────────
    ax1.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax1.plot(dns_x, dns_cf, color=C_DNS, lw=1.6, label="DNS (Krank 2018)")
    ax1.axvline(dns_xr, color=C_DNS, ls=":", lw=1.0)
    if present and "wmles_x" in d.files:
        wx, wcf, wxr = d["wmles_x"], d["wmles_cf"], float(d["wmles_x_reatt"])
        ax1.plot(wx, wcf, color=C_WM, lw=1.6, label="WMLES (equilibrium)")
        ax1.axvline(wxr, color=C_WM, ls=":", lw=1.0)
        ax1.annotate(
            rf"$x_r^{{\rm DNS}}={dns_xr:.2f}$" + "\n" +
            rf"$x_r^{{\rm WMLES}}={wxr:.2f}$",
            xy=(0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.5))
    else:
        ax1.text(0.5, 0.9, "coupled run not yet averaged",
                 transform=ax1.transAxes, ha="center", fontsize=8, color="0.4")
    ax1.set_xlabel(r"$x/H$")
    ax1.set_ylabel(r"$C_f = 2\tau_w/u_b^2$")
    ax1.set_xlim(0, 9)
    ax1.legend(fontsize=7, loc="upper left", frameon=False)
    ax1.set_title("(a) bottom-wall skin friction", fontsize=9)

    # ── panel (b): mean U profiles ──────────────────────────────────────
    dx = 1.2   # horizontal offset per station, in u_b units
    for k, xs in enumerate(PROFILE_X):
        suf = [s for s, xv in DNS_STATIONS.items() if abs(xv - xs) < 1e-6][0]
        yd, ud = load_dns_profile(suf)
        ax2.plot(ud + k * dx, yd, color=C_DNS, lw=0.0, marker="o", ms=2.2,
                 mfc="none", mew=0.7,
                 label="DNS" if k == 0 else None)
        wp = load_wmles_profile(PROFILE_SET[xs])
        if wp is not None:
            yw, uw = wp
            good = np.isfinite(uw)
            ax2.plot(uw[good] + k * dx, yw[good], color=C_WM, lw=1.3,
                     label="WMLES" if k == 0 else None)
        ax2.axvline(k * dx, color="0.85", lw=0.5, zorder=0)
        ax2.text(k * dx, 3.15, rf"$x/H={xs:g}$", ha="center", fontsize=7)
    ax2.set_xlabel(r"$U/u_b$ (offset by station)")
    ax2.set_ylabel(r"$y/H$")
    ax2.set_ylim(0, 3.4)
    ax2.legend(fontsize=7, loc="lower right", frameon=False)
    ax2.set_title("(b) mean velocity profiles", fontsize=9)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"fig_aposteriori.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)
    print(f"[fig] wmles_present={present}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
