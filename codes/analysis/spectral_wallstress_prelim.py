"""Thrust #21 L0 PRELIMINARY — streamwise-wavenumber spectrum of the ODE
wall-stress error on the canonical periodic hill (alpha=1.0, Re_b=5600).

A-priori, no new CFD, no fabrication. Reads only real on-disk DNS-derived
arrays:
  - momentum_budget_pehill.npz : full periodic-domain fields (512 stations / pitch)
  - diagnostic_test_corrected.npz : production ODE tau_w (standard_ml) + tau_w_dns

Thesis (Thrust #21): the ODE is the k_x=0 streamwise-wavenumber truncation; the
error it makes is concentrated in the NONZERO modes (pitch fundamental k_0),
NOT broadband and NOT a k_x=0 (DC) offset. This script verifies F1 on alpha=1.0.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/spectral_wallstress_prelim.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))


def ac_frac(P, lo, hi):
    """Fraction of nonzero-wavenumber (AC) energy in modes [lo, hi]."""
    return float(P[lo:hi + 1].sum() / P[1:].sum())


def main():
    mb = np.load(os.path.join(RES, "momentum_budget_pehill.npz"), allow_pickle=True)
    dg = np.load(os.path.join(RES, "diagnostic_test_corrected.npz"), allow_pickle=True)

    x = mb["x_unique"]
    tau_dns = dg["tau_w_dns"]
    tau_ode = dg["standard_ml"]
    conv = mb["conv_rms_x"]
    valid = dg["standard_ml_valid"] & np.isfinite(tau_dns) & np.isfinite(tau_ode)

    n = len(x)
    Lx = float(x.max() - x.min())          # one pitch (periodic domain)
    dx = np.diff(x)
    assert dx.std() / dx.mean() < 1e-5, "x not uniform; FFT assumption broken"

    err = np.where(valid, tau_ode - tau_dns, 0.0)
    win = np.hanning(n)

    def spec(sig):
        return np.abs(np.fft.rfft(sig * win)) ** 2

    Perr = spec(err)
    Pdns = spec(tau_dns)
    Pconv = spec(conv - conv.mean())

    out = dict(
        geometry="xiao_pehill_1p0", alpha=1.0, Re_b=5600.0, Y_IDX=int(dg["Y_IDX"]),
        n_stations=n, L_x_pitch=Lx,
        err_mode0_frac_total=float(Perr[0] / Perr.sum()),
        err_AC_fundamental=ac_frac(Perr, 1, 1),
        err_AC_harm_1_3=ac_frac(Perr, 1, 3),
        err_AC_harm_1_6=ac_frac(Perr, 1, 6),
        dns_AC_harm_1_3=ac_frac(Pdns, 1, 3),
        conv_AC_harm_1_3=ac_frac(Pconv, 1, 3),
        note=("F1 preliminary: ODE wall-stress error spectrum concentrated at the "
              "pitch fundamental (k_0). a-priori; reads only real DNS-derived npz."),
    )
    np.savez(os.path.join(RES, "spectral_wallstress_prelim.npz"), **out)

    print("=== Thrust #21 L0 preliminary (alpha=1.0 periodic hill) ===")
    for k, v in out.items():
        print(f"  {k:24s} = {v}")
    print("\nF1 signature: error AC energy in fundamental =",
          f"{out['err_AC_fundamental']*100:.1f}% (broadband would be ~1/Nmodes).")


if __name__ == "__main__":
    main()
