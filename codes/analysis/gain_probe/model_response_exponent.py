#!/usr/bin/env python3
"""Wall-model response exponent s = d ln|tau_w| / d ln u_m, from the deployed kernels.

For a scalar-eddy-viscosity wall boundary condition the traction handed to the
momentum equation is tau_d = rho (nu + nut_w) u_m / y_m, so the slope the
boundary condition presents to the momentum equation is k_BC = rho (nu+nut_w)/y_m
= tau_d/u_m.  The slope the *model* actually has is Sigma = d tau_w / d u_m, and

    s = Sigma / k_BC = d ln|tau_w| / d ln u_m

is the local power-law exponent of tau_w ~ u_m^s.  The equilibrium logarithmic
law gives s slightly below 2; a wall model whose stress is set by an imposed
pressure-gradient impulse rather than by the matching velocity has s -> 0.

s is computed here by re-solving the production kernels at u_m(1 +/- h):
  * total-gradient TBLE - the compiled campaign kernel, through
    codes/analysis/gain_probe/tble_response_probe;
  * equilibrium - the deployed nutUSpaldingWallFunction Newton iteration
    transcribed in codes/analysis/deployed_operator/deployed_operator.py.

No CFD is run: every face is a deposited coupled state at t = 405.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis" / "deployed_operator"))

import deployed_operator as dop  # noqa: E402

DEPOSIT_RE = re.compile(
    r"^rswm_r23m6_ym(?P<ym>\d{4})_g1_(?P<arch>tble|equilibrium)_307200_v1$")

# The deposited flat-floor matching heights, y_m/H.
YM_OF_TAG = {"0300": 0.03, "0600": 0.06, "0935": 0.0935,
             "1500": 0.15, "2500": 0.25}
NU = 1.785714e-04          # Re_H = 5600, constant/physicalProperties
H_REL = 0.02               # relative central-difference step in u_m


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def deposits(npz):
    found = {}
    for key in npz.files:
        if "__405__bottomWall__u_m" not in key:
            continue
        case = key.split("__405__")[0]
        m = DEPOSIT_RE.match(case)
        if m:
            found[case] = m.groupdict()
    return found


def tble_exponent(u_m, dpds, y_m, probe, workers=8):
    """Central-difference response of the compiled kernel, sharded over cores.

    Each shard is a contiguous slice of the face list, solved by its own copy of
    the same executable; the shards are concatenated in order, so the result is
    independent of the number of workers.
    """
    lines = ["%.17g %.17g %.17g %.17g 0\n" % (u, d, y, NU)
             for u, d, y in zip(u_m, dpds, y_m)]
    n = len(lines)
    workers = max(1, min(workers, n))
    edges = [round(i * n / workers) for i in range(workers + 1)]
    procs = []
    for a, b in zip(edges[:-1], edges[1:]):
        procs.append(subprocess.Popen(
            [str(probe), "%.17g" % H_REL], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True))
        procs[-1].stdin.write("".join(lines[a:b]))
        procs[-1].stdin.close()
    chunks = []
    for pr in procs:
        chunks.append(pr.stdout.read())
        pr.wait()
        if pr.returncode != 0:
            raise RuntimeError("kernel probe shard failed (rc=%d)" % pr.returncode)
    rows = [r.split() for r in "".join(chunks).strip().splitlines()]
    if len(rows) != len(u_m):
        raise RuntimeError("probe returned %d of %d rows"
                           % (len(rows), len(u_m)))
    ok = np.array([int(r[0]) for r in rows], bool)
    t0 = np.array([float(r[1]) for r in rows])
    tm = np.array([float(r[2]) for r in rows])
    tp = np.array([float(r[3]) for r in rows])
    roots = np.array([int(r[4]) for r in rows])
    return ok, t0, tm, tp, roots


def spalding_exponent(speed, y_m, nut_seed):
    def tau(sp):
        u_tau, _ = dop.spalding_utau_deployed(sp, y_m, NU, nut_seed)
        return u_tau ** 2
    t0 = tau(speed)
    tm = tau(speed * (1.0 - H_REL))
    tp = tau(speed * (1.0 + H_REL))
    ok = np.isfinite(t0) & np.isfinite(tm) & np.isfinite(tp) & (t0 > 0)
    return ok, t0, tm, tp, np.ones_like(t0, dtype=int)


def log_slope(tm, tp, ok):
    s = np.full(tm.shape, np.nan)
    good = ok & (np.abs(tm) > 0) & (np.abs(tp) > 0) & (np.sign(tm) == np.sign(tp))
    s[good] = (np.log(np.abs(tp[good])) - np.log(np.abs(tm[good]))) \
        / (np.log(1.0 + H_REL) - np.log(1.0 - H_REL))
    return s, good


def bootstrap_median(values, draws, rng):
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(draws, values.size))
    meds = np.median(values[idx], axis=1)
    return (float(np.median(values)),
            float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    default="codes/results/as_deployed_evaluation_20260824.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--draws", type=int, default=20000)
    args = ap.parse_args()

    src = ROOT / args.source
    probe = HERE / "tble_response_probe"
    if not probe.exists():
        raise SystemExit("build the kernel probe first: g++ -O3 -std=c++14 "
                         "-I jobs codes/analysis/gain_probe/"
                         "tble_response_probe.cpp -o %s" % probe)

    npz = np.load(src, allow_pickle=True)
    rng = np.random.default_rng(20260824)
    records = []
    arrays = {}

    for case, meta in sorted(deposits(npz).items()):
        pre = "%s__405__bottomWall__" % case
        u_m = np.asarray(npz[pre + "u_m"], float)
        q = np.asarray(npz[pre + "q"], float)
        y_m = np.asarray(npz[pre + "y_m"], float)
        dpds = np.asarray(npz[pre + "dpds"], float)
        tau_req = np.asarray(npz[pre + "tau_request"], float)
        tau_int = np.asarray(npz[pre + "tau_internal"], float)
        speed = np.asarray(npz[pre + "s"], float)
        nut = np.asarray(npz[pre + "nut"], float)

        if meta["arch"] == "tble":
            # The deployed TBLE kernel takes the signed tangential matching
            # velocity, and its request is also the scored request.
            ok, t0, tm, tp, roots = tble_exponent(u_m, dpds, y_m, probe)
            velocity = u_m
            reference = np.abs(tau_req)
        else:
            # The deployed nutUSpaldingWallFunction evaluates the wall law at
            # the full relative speed |U_c| and seeds its Newton iteration from
            # the patch eddy viscosity, so both are used here.  Its reference is
            # tau_internal, the stress that function actually asks for; the
            # scored request tau_request is evaluated at the tangential
            # component instead and is a different quantity.
            ok, t0, tm, tp, roots = spalding_exponent(speed, y_m, nut)
            velocity = speed
            reference = np.abs(tau_int)

        s, good = log_slope(tm, tp, ok)
        # Reproduction check: the kernel re-solved at the deposited u_m must
        # return the deposited request.
        rel = np.full(t0.shape, np.nan)
        nz = np.abs(reference) > 0
        rel[nz] = np.abs(np.abs(t0[nz]) - reference[nz]) / reference[nz]

        med, lo, hi = bootstrap_median(s[good], args.draws, rng)
        rec = dict(
            case=case,
            architecture=meta["arch"],
            ym_tag=meta["ym"],
            ym_over_H=YM_OF_TAG[meta["ym"]],
            n_faces=int(s.size),
            n_resolved=int(good.sum()),
            resolved_fraction=float(good.mean()),
            s_median=med, s_ci_lo=lo, s_ci_hi=hi,
            s_mean=float(np.nanmean(s[good])) if good.any() else float("nan"),
            s_p10=float(np.percentile(s[good], 10)) if good.any() else float("nan"),
            s_p90=float(np.percentile(s[good], 90)) if good.any() else float("nan"),
            fraction_s_below_1=float(np.mean(s[good] < 1.0)) if good.any() else float("nan"),
            fraction_s_nonpositive=float(np.mean(s[good] <= 0.0)) if good.any() else float("nan"),
            fraction_multiroot=float(np.mean(roots > 1)),
            max_reproduction_rel_error=float(np.nanmax(rel)) if np.isfinite(rel).any() else float("nan"),
            median_reproduction_rel_error=float(np.nanmedian(rel)) if np.isfinite(rel).any() else float("nan"),
            velocity_argument="u_m (tangential)" if meta["arch"] == "tble"
            else "q (speed, as the deployed Spalding function uses)",
        )
        records.append(rec)
        arrays["%s__s" % case] = s
        arrays["%s__tau0" % case] = t0
        arrays["%s__u" % case] = velocity
        arrays["%s__y_m" % case] = y_m
        arrays["%s__dpds" % case] = dpds
        arrays["%s__resolved" % case] = good
        print("%-46s ym/H=%.4f  s=%.3f [%.3f,%.3f]  resolved %.3f  "
              "reproduce max %.2e"
              % (case, rec["ym_over_H"], med, lo, hi,
                 rec["resolved_fraction"], rec["max_reproduction_rel_error"]))

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d")
    out = pathlib.Path(args.out) if args.out else \
        ROOT / ("codes/results/gain_probe_model_exponent_%s.npz" % stamp)
    np.savez_compressed(out, **arrays)
    summary = dict(
        generated=datetime.datetime.utcnow().isoformat() + "Z",
        source=str(args.source),
        source_sha256=sha256(src),
        probe_source_sha256=sha256(HERE / "tble_response_probe.cpp"),
        kernel_sha256=sha256(ROOT / "jobs" / "rswm_m13_tbleShoot_degenerate.H"),
        h_relative=H_REL,
        nu=NU,
        definition="s = dln|tau_w|/dln u_m = (dtau_w/du_m)/(tau_d/u_m); "
                   "tau_d/u_m = rho(nu+nut_w)/y_m is the slope the scalar "
                   "eddy-viscosity boundary condition presents to the "
                   "momentum equation",
        records=records,
    )
    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print("wrote %s (+ summary)" % out)


if __name__ == "__main__":
    main()
