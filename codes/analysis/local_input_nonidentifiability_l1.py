#!/usr/bin/env python3
r"""Matched-input non-identifiability theorem for local ODE wall models.

This program constructs the numerical witness used to verify ledger item M9.
It does not use the wall-following wall-force reconstruction and it does not
fit a new diagnostic.  Instead it asks whether the information supplied to the
standard pressure-gradient ODE is sufficient to identify wall traction.

For a deterministic, similarity-equivariant, single-sample local model with
fixed dimensionless closure constants, dimensional analysis gives

    tau_hat_M = F(a, b),
    a = U_m y_m / nu,
    b = (dp/dx) y_m^3 / nu^2,
    tau_hat = tau_w y_m^2 / nu^2.

If two reference states have identical (a,b) but different true tau_hat, the
model must return the same value at both states.  The triangle inequality then
gives the exact minimax bound

    max_i |m-t_i|/|t_i| >= |t_1-t_2|/(|t_1|+|t_2|).

When the two true tractions have opposite signs, the right-hand side is one:
every model in the stated class has at least 100% relative error, and the sign
is wrong (or zero) at one of the two states.

The primary witness is an intersection of the (a,b) trajectories from the
corrected Xiao periodic-hill DNS and the Bentaleb rounded-backward-facing-step
LES.  A second witness uses a wall-resolved square-rib LES against the same
step.  The intersections are solved independently with piecewise-linear,
PCHIP and cubic interpolation and repeated over matching indices 8--15.  No
RANS field, modelled stress, eddy viscosity or wall-model prediction enters.

Outputs
-------
codes/results/local_input_nonidentifiability_l1.npz
codes/results/local_input_nonidentifiability_l1_summary.json
codes/results/fig_local_input_nonidentifiability_l1.{pdf,png}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "codes" / "results"
HILL_PATH = RESULTS / "periodic_hills_case_1p0_wall_profiles_corrected.npz"
BFS_PATH = (
    ROOT
    / "codes"
    / "vendor"
    / "universal_wall_function"
    / "codes"
    / "results"
    / "bfs_Re13700_wall_profiles.npz"
)
RIB_PATH = RESULTS / "rib_les_dtype_wall_profiles.npz"
Y_INDEX = 10
HEIGHT_INDICES = tuple(range(8, 16))


@dataclass(frozen=True)
class Family:
    name: str
    source: Path
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    traction: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def station_value(array: np.ndarray, station: int, index: int) -> float:
    return float(array[station, index] if array.ndim == 2 else array[index])


def load_family(name: str, path: Path, index: int = Y_INDEX) -> Family:
    data = np.load(path, allow_pickle=True)
    n_station = len(data["tau_w"])
    if index < 0:
        index = int(data["Y_IDX"]) if "Y_IDX" in data.files else Y_INDEX
    x = np.asarray(data["x"], dtype=float)
    nu_values = np.asarray(data["nu"], dtype=float).reshape(-1)
    valid_x: list[float] = []
    a: list[float] = []
    b: list[float] = []
    traction: list[float] = []

    for i in range(n_station):
        y_m = station_value(np.asarray(data["y"]), i, index)
        u_m = station_value(np.asarray(data["U"]), i, index)
        nu = float(nu_values[i] if nu_values.size == n_station else nu_values[0])
        pressure_gradient = float(data["dp_dx"][i])
        tau_w = float(data["tau_w"][i])
        if not np.all(np.isfinite((x[i], y_m, u_m, nu, pressure_gradient, tau_w))):
            continue
        if y_m <= 0.0 or nu <= 0.0:
            continue
        valid_x.append(float(x[i]))
        a.append(u_m * y_m / nu)
        b.append(pressure_gradient * y_m**3 / nu**2)
        traction.append(tau_w * y_m**2 / nu**2)

    x_array = np.asarray(valid_x)
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    traction_array = np.asarray(traction)
    order = np.argsort(x_array)
    return Family(
        name,
        path,
        x_array[order],
        a_array[order],
        b_array[order],
        traction_array[order],
    )


def cross2(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def linear_intersections(first: Family, second: Family) -> list[dict[str, float]]:
    """Return every exact intersection of the two piecewise-linear input paths."""
    p_path = np.column_stack((first.a, first.b))
    q_path = np.column_stack((second.a, second.b))
    hits: list[dict[str, float]] = []
    for i in range(len(p_path) - 1):
        p0 = p_path[i]
        p_dir = p_path[i + 1] - p0
        for j in range(len(q_path) - 1):
            q0 = q_path[j]
            q_dir = q_path[j + 1] - q0
            denominator = cross2(p_dir, q_dir)
            if abs(denominator) < 1.0e-14:
                continue
            frac_first = cross2(q0 - p0, q_dir) / denominator
            frac_second = cross2(q0 - p0, p_dir) / denominator
            if not (-1.0e-12 <= frac_first <= 1.0 + 1.0e-12):
                continue
            if not (-1.0e-12 <= frac_second <= 1.0 + 1.0e-12):
                continue
            point = p0 + frac_first * p_dir
            t_first = first.traction[i] + frac_first * (
                first.traction[i + 1] - first.traction[i]
            )
            t_second = second.traction[j] + frac_second * (
                second.traction[j + 1] - second.traction[j]
            )
            hits.append(
                {
                    "segment_first": i,
                    "segment_second": j,
                    "fraction_first": float(frac_first),
                    "fraction_second": float(frac_second),
                    "x_first": float(
                        first.x[i] + frac_first * (first.x[i + 1] - first.x[i])
                    ),
                    "x_second": float(
                        second.x[j] + frac_second * (second.x[j + 1] - second.x[j])
                    ),
                    "a": float(point[0]),
                    "b": float(point[1]),
                    "traction_first": float(t_first),
                    "traction_second": float(t_second),
                    "gap": float(abs(t_first - t_second)),
                    "relative_floor": float(
                        abs(t_first - t_second)
                        / (abs(t_first) + abs(t_second) + np.finfo(float).tiny)
                    ),
                }
            )
    return hits


def interpolators(family: Family, method: str):
    constructor = PchipInterpolator if method == "pchip" else CubicSpline
    return tuple(
        constructor(family.x, field)
        for field in (family.a, family.b, family.traction)
    )


def refine_hit(
    first: Family,
    second: Family,
    seed: dict[str, float],
    method: str,
    pad: int = 2,
) -> dict[str, float]:
    f = interpolators(first, method)
    g = interpolators(second, method)
    i = int(seed["segment_first"])
    j = int(seed["segment_second"])
    lower = np.array(
        [first.x[max(i - pad, 0)], second.x[max(j - pad, 0)]], dtype=float
    )
    upper = np.array(
        [
            first.x[min(i + pad + 1, len(first.x) - 1)],
            second.x[min(j + pad + 1, len(second.x) - 1)],
        ],
        dtype=float,
    )

    def residual(position: np.ndarray) -> np.ndarray:
        return np.array(
            [
                f[0](position[0]) - g[0](position[1]),
                f[1](position[0]) - g[1](position[1]),
            ],
            dtype=float,
        )

    fit = least_squares(
        residual,
        np.array([seed["x_first"], seed["x_second"]]),
        bounds=(lower, upper),
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=1000,
    )
    q_first = np.array([f[0](fit.x[0]), f[1](fit.x[0])], dtype=float)
    q_second = np.array([g[0](fit.x[1]), g[1](fit.x[1])], dtype=float)
    t_first = float(f[2](fit.x[0]))
    t_second = float(g[2](fit.x[1]))
    return {
        "x_first": float(fit.x[0]),
        "x_second": float(fit.x[1]),
        "a": float(0.5 * (q_first[0] + q_second[0])),
        "b": float(0.5 * (q_first[1] + q_second[1])),
        "input_residual": float(np.linalg.norm(q_first - q_second)),
        "traction_first": t_first,
        "traction_second": t_second,
        "gap": float(abs(t_first - t_second)),
        "relative_floor": float(
            abs(t_first - t_second)
            / (abs(t_first) + abs(t_second) + np.finfo(float).tiny)
        ),
        "opposite_sign": bool(t_first * t_second < 0.0),
        "success": bool(fit.success),
    }


def matching_height_sensitivity(
    first_name: str, first_path: Path, second_name: str, second_path: Path
) -> list[dict[str, float]]:
    """Repeat the collision search at eight adjacent physical samples.

    The result is not eight independent datasets.  It is a registered
    observation-operator sensitivity test: the impossibility witness must not
    depend on choosing index 10.
    """
    records: list[dict[str, float]] = []
    for index in HEIGHT_INDICES:
        first = load_family(first_name, first_path, index=index)
        second = load_family(second_name, second_path, index=index)
        hits = linear_intersections(first, second)
        opposite = [
            hit
            for hit in hits
            if hit["traction_first"] * hit["traction_second"] < 0.0
        ]
        if not opposite:
            records.append(
                {
                    "index": index,
                    "n_intersections": len(hits),
                    "n_opposite": 0,
                    "opposite_sign": False,
                }
            )
            continue
        seed = max(opposite, key=lambda hit: hit["gap"])
        pchip = refine_hit(first, second, seed, "pchip", pad=3)
        cubic = refine_hit(first, second, seed, "cubic", pad=3)
        records.append(
            {
                "index": index,
                "n_intersections": len(hits),
                "n_opposite": len(opposite),
                "linear_gap": float(seed["gap"]),
                "linear_floor": float(seed["relative_floor"]),
                "pchip_input_residual": float(pchip["input_residual"]),
                "cubic_input_residual": float(cubic["input_residual"]),
                "pchip_traction_first": float(pchip["traction_first"]),
                "pchip_traction_second": float(pchip["traction_second"]),
                "cubic_traction_first": float(cubic["traction_first"]),
                "cubic_traction_second": float(cubic["traction_second"]),
                "opposite_sign": bool(
                    pchip["opposite_sign"] and cubic["opposite_sign"]
                ),
            }
        )
    return records


def make_figure(
    hill: Family,
    bfs: Family,
    primary: dict[str, float],
    out_pdf: Path,
    out_png: Path,
) -> None:
    orange = "#D55E00"
    blue_grey = "#607D8B"
    black = "#111111"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))

    ax = axes[0]
    window_h = np.abs(hill.x - primary["x_first"]) < 0.30
    window_c = np.abs(bfs.x - primary["x_second"]) < 1.0
    ax.plot(hill.a[window_h], hill.b[window_h], color=orange, lw=2.0, label="periodic hill DNS")
    ax.plot(bfs.a[window_c], bfs.b[window_c], color=blue_grey, lw=2.0, label="rounded-step LES")
    ax.scatter(primary["a"], primary["b"], s=48, color=black, zorder=5, label="matched local input")
    ax.set_xlabel(r"$a=U_m y_m/\nu$")
    ax.set_ylabel(r"$b=(\partial_x p)y_m^3/\nu^2$")
    ax.set_title("(a) Local-input trajectories intersect", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[1]
    t1 = primary["traction_first"]
    t2 = primary["traction_second"]
    model_value = np.linspace(1.25 * min(t1, t2), 1.25 * max(t1, t2), 800)
    worst = np.maximum(np.abs(model_value - t1) / abs(t1), np.abs(model_value - t2) / abs(t2))
    ax.plot(model_value, worst, color=black, lw=2.0)
    ax.axhline(1.0, color=orange, ls="--", lw=1.4, label="theorem floor")
    ax.scatter([0.0], [1.0], color=orange, s=38, zorder=5)
    ax.set_xlabel(r"common local-model output $\widehat\tau_M$")
    ax.set_ylabel("worst relative error across the pair")
    ax.set_ylim(0.0, min(6.0, float(np.nanpercentile(worst, 80))))
    ax.set_title("(b) Opposite truth signs force 100% error", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    hill = load_family("periodic_hill_dns", HILL_PATH)
    bfs = load_family("rounded_bfs_les", BFS_PATH)
    rib = load_family("square_rib_les", RIB_PATH)

    linear_hits = linear_intersections(hill, bfs)
    opposite_linear = [h for h in linear_hits if h["traction_first"] * h["traction_second"] < 0.0]
    if not opposite_linear:
        raise RuntimeError("no opposite-sign matched-input witness")
    seed = max(opposite_linear, key=lambda h: h["gap"])
    pchip = refine_hit(hill, bfs, seed, "pchip", pad=3)
    cubic = refine_hit(hill, bfs, seed, "cubic", pad=3)
    hill_height = matching_height_sensitivity(
        "periodic_hill_dns", HILL_PATH, "rounded_bfs_les", BFS_PATH
    )
    rib_height = matching_height_sensitivity(
        "square_rib_les", RIB_PATH, "rounded_bfs_les", BFS_PATH
    )

    primary = dict(pchip)
    primary.update(
        segment_first=int(seed["segment_first"]),
        segment_second=int(seed["segment_second"]),
        linear_a=float(seed["a"]),
        linear_b=float(seed["b"]),
        linear_traction_first=float(seed["traction_first"]),
        linear_traction_second=float(seed["traction_second"]),
    )

    npz_path = RESULTS / "local_input_nonidentifiability_l1.npz"
    summary_path = RESULTS / "local_input_nonidentifiability_l1_summary.json"
    figure_pdf = RESULTS / "fig_local_input_nonidentifiability_l1.pdf"
    figure_png = RESULTS / "fig_local_input_nonidentifiability_l1.png"

    np.savez(
        npz_path,
        hill_source=np.array(str(HILL_PATH.relative_to(ROOT))),
        bfs_source=np.array(str(BFS_PATH.relative_to(ROOT))),
        rib_source=np.array(str(RIB_PATH.relative_to(ROOT))),
        hill_sha256=np.array(sha256(HILL_PATH)),
        bfs_sha256=np.array(sha256(BFS_PATH)),
        rib_sha256=np.array(sha256(RIB_PATH)),
        y_index=np.array(Y_INDEX),
        n_linear_intersections=np.array(len(linear_hits)),
        n_opposite_linear=np.array(len(opposite_linear)),
        primary=np.array(primary, dtype=object),
        pchip=np.array(pchip, dtype=object),
        cubic=np.array(cubic, dtype=object),
        hill_height_sensitivity=np.array(hill_height, dtype=object),
        rib_height_sensitivity=np.array(rib_height, dtype=object),
        all_linear=np.array(linear_hits, dtype=object),
    )

    summary = {
        "approach": "matched-input non-identifiability theorem",
        "model_class": (
            "deterministic similarity-equivariant single-sample local pressure-gradient "
            "ODEs with fixed dimensionless closure constants and inputs (U_m,y_m,dpdx,nu)"
        ),
        "sources": {
            "periodic_hill_dns": {
                "path": str(HILL_PATH.relative_to(ROOT)),
                "sha256": sha256(HILL_PATH),
                "stations": int(len(hill.x)),
            },
            "rounded_bfs_les": {
                "path": str(BFS_PATH.relative_to(ROOT)),
                "sha256": sha256(BFS_PATH),
                "stations": int(len(bfs.x)),
            },
            "square_rib_les": {
                "path": str(RIB_PATH.relative_to(ROOT)),
                "sha256": sha256(RIB_PATH),
                "stations": int(len(rib.x)),
            },
        },
        "linear_intersections": len(linear_hits),
        "opposite_sign_linear_intersections": len(opposite_linear),
        "primary_pchip": primary,
        "primary_cubic": cubic,
        "matching_height_indices": list(HEIGHT_INDICES),
        "hill_height_sensitivity": hill_height,
        "rib_height_sensitivity": rib_height,
        "height_sensitivity_all_opposite_sign": bool(
            all(row["opposite_sign"] for row in hill_height + rib_height)
        ),
        "height_sensitivity_max_input_residual": float(
            max(
                max(row.get("pchip_input_residual", np.inf), row.get("cubic_input_residual", np.inf))
                for row in hill_height + rib_height
            )
        ),
        "theorem_relative_error_floor": float(primary["relative_floor"]),
        "theorem_proof_check": bool(
            primary["opposite_sign"]
            and primary["input_residual"] < 1.0e-8
            and abs(primary["relative_floor"] - 1.0) < 1.0e-12
        ),
        "complexity": "O(N_h N_b) segment census; O(N_h+N_b) storage",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_figure(hill, bfs, primary, figure_pdf, figure_png)

    # Outputs are deliberately written before assertions (anti-empty-node guard).
    assert summary["theorem_proof_check"]
    assert summary["height_sensitivity_all_opposite_sign"]
    assert summary["height_sensitivity_max_input_residual"] < 1.0e-7
    assert cubic["opposite_sign"] and cubic["input_residual"] < 1.0e-8

    print("MATCHED-INPUT NON-IDENTIFIABILITY THEOREM")
    print(f"  linear intersections: {len(linear_hits)}")
    print(f"  opposite-sign intersections: {len(opposite_linear)}")
    print(
        "  primary PCHIP input: "
        f"a={primary['a']:.9g}, b={primary['b']:.9g}, "
        f"residual={primary['input_residual']:.3e}"
    )
    print(
        "  reference tractions: "
        f"hill={primary['traction_first']:.9g}, "
        f"rounded-BFS={primary['traction_second']:.9g}"
    )
    print(f"  theorem minimax relative-error floor: {primary['relative_floor']:.6f}")
    print(
        "  interpolation/height robustness: "
        f"{len(hill_height) + len(rib_height)}/{len(hill_height) + len(rib_height)} "
        f"opposite sign, max input residual "
        f"{summary['height_sensitivity_max_input_residual']:.3e}"
    )
    print(f"  wrote {npz_path.relative_to(ROOT)}")
    print(f"  wrote {summary_path.relative_to(ROOT)}")
    print(f"  wrote {figure_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
