"""
verify_wall_extraction.py
=========================
Document and quantify a wall-extraction artifact in the legacy periodic-hills
wall-profile file, and demonstrate the hill-surface-aware extraction the
dose-response analysis uses instead.

In the Xiao periodic-hill DNS the lower wall is a hill whose surface y_w(x)
rises well above the channel floor y = 0; the solid hill interior is stored as
an identically-zero field. The legacy file
  vendor/.../periodic_hills_case_1p0_wall_profiles.npz
places the matching column at y = 0 for EVERY station, so on the hill faces it
samples velocities *inside the solid* (U ~ 0). That yields a spuriously tiny
wall shear (median |tau_w| ~ 1e-6, Re_tau ~ 5.6 -- unphysical for turbulent
periodic hills at Re_b = 5600) and hence an artificially small cancellation
parameter on those stations.

The hill-surface-aware extraction in dose_response_xiao.py locates, per station,
the first fluid node above the solid block and measures the wall shear there,
recovering a physical wall shear (median |tau_w| ~ 1e-3, du/dy ~ O(5)).

This script prints the side-by-side comparison at hill-face stations and writes
codes/results/wall_extraction_artifact.npz. It changes NO manuscript number; it
is a diagnostic that flags the legacy file for the results stage.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/verify_wall_extraction.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "utils"))
from data_paths import get_uwf_results_dir            # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "vendor", "universal_wall_function",
                                "codes", "analysis"))
from ode_wall_model import evaluate_on_dataset        # noqa: E402

RAW = os.path.join(ROOT, "new_data_download", "geometry_driven",
                   "xiao_pehill_parameterized", "pehill-5-cases-DNS",
                   "case_1p0", "dns-data", "mean_files.dat")
NU = 1.0 / 5600.0
VEL_TOL = 1e-6


def main():
    legacy = np.load(os.path.join(get_uwf_results_dir(),
                                  "periodic_hills_case_1p0_wall_profiles.npz"),
                     allow_pickle=True)
    xc = legacy["x"].ravel()
    yc, Uc = legacy["y"], legacy["U"]
    tau_legacy = legacy["tau_w"].ravel()
    dp_legacy = legacy["dp_dx"].ravel()
    y_legacy = np.asarray(legacy["y"], dtype=float)

    d = np.loadtxt(RAW)
    x, y, u, v = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    xu = np.unique(np.round(x, 6))

    def wall_y_and_tau(xv):
        m = np.abs(x - xv) < 1e-6
        yy, uu, vv = y[m], u[m], v[m]
        o = np.argsort(yy)
        yy, uu, vv = yy[o], uu[o], vv[o]
        fluid = np.where((np.abs(uu) > VEL_TOL) | (np.abs(vv) > VEL_TOL))[0]
        if len(fluid) < 5:
            return np.nan, np.nan
        k = max(fluid[0], 1)
        yw = yy[k - 1]
        yloc = yy[k:k + 3] - yw
        uloc = uu[k:k + 3]
        dudy = float(np.sum(yloc * uloc) / np.sum(yloc * yloc))
        return yw, NU * dudy

    print("Wall-extraction comparison at hill-face stations (Xiao case_1p0)")
    print("-" * 76)
    print(f"{'x':>6s} {'wall_y(true)':>12s} {'tau_legacy':>12s} {'tau_corrected':>14s}")
    rows = []
    for xt in [7.0, 7.5, 8.0, 8.5]:
        xv = xu[np.argmin(np.abs(xu - xt))]
        ic = np.argmin(np.abs(xc - xt))
        yw, tw = wall_y_and_tau(xv)
        rows.append((xv, yw, tau_legacy[ic], tw))
        print(f"{xv:6.2f} {yw:12.4f} {tau_legacy[ic]:12.3e} {tw:14.3e}")

    med_legacy = float(np.nanmedian(np.abs(tau_legacy)))
    # Preserve the obsolete epsilon value only as named provenance.  The live
    # corrected producer reads these explicitly-labelled fields instead of
    # regenerating the old manuscript figure/result.
    y_idx = 10
    y_m_legacy = np.array([
        row[min(y_idx, len(row) - 1)] for row in y_legacy
    ])
    den_legacy = np.abs(dp_legacy) * np.abs(y_m_legacy)
    eps_legacy = np.full_like(tau_legacy, np.nan, dtype=float)
    valid_legacy = den_legacy > 1e-30
    eps_legacy[valid_legacy] = (
        np.abs(tau_legacy[valid_legacy]) / den_legacy[valid_legacy]
    )
    finite_legacy = np.isfinite(eps_legacy) & (eps_legacy > 0)
    legacy_median_eps = float(np.nanmedian(eps_legacy[finite_legacy]))
    legacy_frac_below_01 = float(np.mean(eps_legacy[finite_legacy] < 0.1))
    legacy_eval = evaluate_on_dataset(
        os.path.join(get_uwf_results_dir(),
                     "periodic_hills_case_1p0_wall_profiles.npz"),
        y_indices=[y_idx])[y_idx]
    legacy_r2 = float(legacy_eval["r2_tau"])
    print("-" * 76)
    print(f"legacy median |tau_w| = {med_legacy:.3e}  (Re_tau ~ "
          f"{float(np.nanmedian(legacy['Re_tau'])):.1f}, unphysically low)")
    print("On the hill faces the legacy file samples U ~ 0 inside the solid hill "
          "(wall pinned at y=0),\nartificially deflating tau_w and hence the "
          "cancellation parameter there.")

    np.savez(os.path.join(RESULTS, "wall_extraction_artifact.npz"),
             x=np.array([r[0] for r in rows]),
             wall_y_true=np.array([r[1] for r in rows]),
             tau_legacy=np.array([r[2] for r in rows]),
             tau_corrected=np.array([r[3] for r in rows]),
             legacy_median_abs_tau=med_legacy,
             legacy_median_Re_tau=float(np.nanmedian(legacy["Re_tau"])),
             legacy_median_epsilon=legacy_median_eps,
             legacy_frac_epsilon_below_01=legacy_frac_below_01,
             legacy_wall_pinned_r2_tau=legacy_r2,
             legacy_y_idx=y_idx,
             status=np.array("superseded_wall_pinned_provenance_only"))
    print(f"\nSaved -> {os.path.relpath(os.path.join(RESULTS, 'wall_extraction_artifact.npz'), ROOT)}")


if __name__ == "__main__":
    main()
