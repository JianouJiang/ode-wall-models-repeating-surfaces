#!/usr/bin/env python3
"""Harvest and reduce the mode-resolved wall-traction sensitivity experiment
(L0 node_002).

Reads the per-run artifacts produced on Oxford ARC by
`jobs/rswm_modal_sensitivity_driver.sh` and produces one artifact holding

  * the DELIVERED perturbation actually received by the momentum equation,
    measured face by face inside the boundary condition and reported by it --
    never the requested perturbation, and never assumed equal to it;
  * the chaotic noise floor of every reported quantity, measured from an
    ensemble of statistically identical unperturbed continuations;
  * the response of each quantity of interest to each injected mode, expressed
    both in physical units and in units of that floor.

Nothing here is scored against an external truth reference.  Every number is a
difference between two runs of the same simulation that differ only in the
perturbation, so the reference-conditioning problem that governs the rest of
this paper does not arise.

Usage:
    python3 codes/analysis/modal_sensitivity_harvest.py \
        --root codes/results/modal_sensitivity \
        --out  codes/results/modal_sensitivity_l0_20260825
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

MODALPERT = re.compile(
    r"^MODALPERT\s+patch=(?P<patch>\S+)\s+time=(?P<time>\S+)\s+index=(?P<index>\S+)\s+"
    r"kind=(?P<kind>\S+)\s+k=(?P<k>\S+)\s+phase=(?P<phase>\S+)\s+"
    r"modeActive=(?P<modeActive>\S+)\s+seedActive=(?P<seedActive>\S+)\s+"
    r"faces=(?P<faces>\S+)\s+area=(?P<area>\S+)\s+rmsTarget=(?P<rmsTarget>\S+)\s+"
    r"rmsReq=(?P<rmsReq>\S+)\s+rmsDel=(?P<rmsDel>\S+)\s+netReq=(?P<netReq>\S+)\s+"
    r"netDel=(?P<netDel>\S+)\s+tauReqInt=(?P<tauReqInt>\S+)\s+"
    r"tauDelInt=(?P<tauDelInt>\S+)\s+clipped=(?P<clipped>\S+)\s*$"
)

# meanVelocityForce prints the instantaneous pressure-gradient source needed to
# hold the prescribed bulk velocity.  It is the total streamwise force per unit
# volume the simulation must supply, so it is a reference-free global QoI.
DRIVE = re.compile(r"pressure gradient\s*=\s*([-\dEe.+]+)")
TIMELINE = re.compile(r"^Time = (\S+)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_modalpert(path, patch="bottomWall"):
    """Time series of the boundary condition's own delivery diagnostics."""
    rows = []
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = MODALPERT.match(line.strip())
        if m and m.group("patch") == patch:
            rows.append({k: m.group(k) for k in m.groupdict()})
    if not rows:
        return None
    out = {}
    for key in ("time", "rmsTarget", "rmsReq", "rmsDel", "netReq", "netDel",
                "tauReqInt", "tauDelInt", "area", "phase"):
        out[key] = np.array([float(r[key]) for r in rows])
    for key in ("index", "faces", "clipped", "k", "modeActive", "seedActive"):
        out[key] = np.array([int(float(r[key])) for r in rows])
    return out


def read_drive(path, timeline_path):
    """Pair each printed drive gradient with the time step it belongs to."""
    if not path.exists():
        return None, None
    g = np.array([float(m.group(1))
                  for m in (DRIVE.search(l) for l in path.read_text().splitlines())
                  if m])
    t = None
    if timeline_path.exists():
        t = np.array([float(m.group(1))
                      for m in (TIMELINE.match(l)
                                for l in timeline_path.read_text().splitlines())
                      if m])
    if t is not None and len(t) and len(g):
        n = min(len(t), len(g))
        return t[:n], g[:n]
    return None, g


def read_wall_sample(run_dir):
    """Spanwise-averaged mean wall traction on the hill wall, x-sorted."""
    cands = sorted((run_dir / "postProcessing" / "sampleBottomWall").glob("*/bottomWall.xy")) \
        if (run_dir / "postProcessing" / "sampleBottomWall").is_dir() else []
    if not cands:
        return None
    d = np.loadtxt(cands[-1])
    if d.ndim != 2 or d.shape[1] < 7:
        return None
    x = np.round(d[:, 0], 9)
    xs = np.unique(x)
    tau_x = np.array([d[x == v, 3].mean() for v in xs])
    tau_mag = np.array([np.linalg.norm(d[x == v, 3:6], axis=1).mean() for v in xs])
    p = np.array([d[x == v, 6].mean() for v in xs])
    return dict(x=xs, tau_x=tau_x, tau_mag=tau_mag, p=p,
                time=float(cands[-1].parent.name))


def crossings(x, f):
    """Sign changes of f(x), linearly interpolated."""
    out = []
    s = np.sign(f)
    for i in range(len(f) - 1):
        if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
            t = f[i] / (f[i] - f[i + 1])
            out.append(float(x[i] + t * (x[i + 1] - x[i])))
    return out


def window_mean(t, y, t0, t1):
    if t is None or y is None:
        return None, None, None
    m = (t >= t0) & (t <= t1)
    if m.sum() < 8:
        return None, None, None
    return float(y[m].mean()), float(y[m].std(ddof=1)), int(m.sum())


def harvest_run(run_dir):
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())

    rec = dict(cfg)
    rec["run_dir"] = str(run_dir)

    t0, t1 = cfg["avg_start"], cfg["end_time"]

    # --- delivered perturbation (measured inside the boundary condition) ----
    mp = read_modalpert(run_dir / "modalpert.log", "bottomWall")
    mp_top = read_modalpert(run_dir / "modalpert.log", "topWall")
    if mp is not None:
        w = (mp["time"] >= t0) & (mp["time"] <= t1)
        if w.sum() >= 4:
            rec["delivered"] = dict(
                rms_requested=float(mp["rmsReq"][w].mean()),
                rms_delivered=float(mp["rmsDel"][w].mean()),
                rms_target=float(mp["rmsTarget"][w].mean()),
                net_requested=float(mp["netReq"][w].mean()),
                net_delivered=float(mp["netDel"][w].mean()),
                tau_req_int=float(mp["tauReqInt"][w].mean()),
                tau_del_int=float(mp["tauDelInt"][w].mean()),
                clipped_faces=float(mp["clipped"][w].mean()),
                n_faces=int(mp["faces"][w][0]),
                wall_area=float(mp["area"][w][0]),
                n_samples=int(w.sum()),
            )
            rec["delivered"]["clipped_fraction"] = (
                rec["delivered"]["clipped_faces"] / max(rec["delivered"]["n_faces"], 1)
            )
            rec["delivered"]["delivery_efficiency"] = (
                rec["delivered"]["rms_delivered"]
                / max(rec["delivered"]["rms_requested"], 1e-30)
            )
    if mp_top is not None:
        w = (mp_top["time"] >= t0) & (mp_top["time"] <= t1)
        if w.sum() >= 4:
            # The flat top wall is never perturbed: this is the within-run
            # control that the instrument touches only the surface under test.
            rec["control_top_wall"] = dict(
                rms_delivered=float(mp_top["rmsDel"][w].mean()),
                net_delivered=float(mp_top["netDel"][w].mean()),
                clipped_faces=float(mp_top["clipped"][w].mean()),
            )

    # --- global QoI: the drive the simulation needs -------------------------
    t, g = read_drive(run_dir / "drive.log", run_dir / "timeline.log")
    gm, gs, gn = window_mean(t, g, t0, t1)
    rec["drive"] = dict(mean=gm, std=gs, n=gn,
                        series_len=int(len(g)) if g is not None else 0)

    # --- wall QoI: separation and reattachment ------------------------------
    ws = read_wall_sample(run_dir)
    if ws is not None:
        cr = crossings(ws["x"], ws["tau_x"])
        rec["wall"] = dict(
            sample_time=ws["time"],
            tau_x_rms=float(np.sqrt((ws["tau_x"] ** 2).mean())),
            tau_mag_mean=float(ws["tau_mag"].mean()),
            tau_x_int=float(np.trapz(ws["tau_x"], ws["x"])),
            crossings=cr,
            n_stations=int(len(ws["x"])),
        )
        rec["_wall_profile"] = dict(x=ws["x"].tolist(),
                                    tau_x=ws["tau_x"].tolist())
    return rec


def reduce_campaign(records):
    """Null floor, then every perturbed run expressed in units of that floor."""
    nulls = [r for r in records if r["kind"] == "none"]
    perts = [r for r in records if r["kind"] != "none"]

    def collect(rs, path):
        out = []
        for r in rs:
            v = r
            for p in path:
                v = (v or {}).get(p) if isinstance(v, dict) else None
            if v is not None:
                out.append(float(v))
        return np.array(out)

    floor = {}
    for name, path in [("drive_mean", ("drive", "mean")),
                       ("tau_x_rms", ("wall", "tau_x_rms")),
                       ("tau_x_int", ("wall", "tau_x_int")),
                       ("tau_mag_mean", ("wall", "tau_mag_mean"))]:
        v = collect(nulls, path)
        floor[name] = dict(
            n=int(len(v)),
            mean=float(v.mean()) if len(v) else None,
            sd=float(v.std(ddof=1)) if len(v) > 1 else None,
            spread=float(v.max() - v.min()) if len(v) else None,
            values=v.tolist(),
        )

    # Reattachment: first crossing common to the null members.
    xr = []
    for r in nulls:
        c = (r.get("wall") or {}).get("crossings") or []
        if c:
            xr.append(c[0] if len(c) == 1 else c[-1])
    floor["reattachment"] = dict(
        n=len(xr),
        mean=float(np.mean(xr)) if xr else None,
        sd=float(np.std(xr, ddof=1)) if len(xr) > 1 else None,
        spread=float(max(xr) - min(xr)) if xr else None,
        values=xr,
    )

    responses = []
    for r in perts:
        d = r.get("delivered") or {}
        rms_del = d.get("rms_delivered")
        item = dict(run_id=r["run_id"], kind=r["kind"], k=r["wave_number"],
                    phase=r["phase"], rms_target=r.get("rms_amplitude"),
                    rms_delivered=rms_del,
                    net_delivered=d.get("net_delivered"),
                    clipped_fraction=d.get("clipped_fraction"),
                    delivery_efficiency=d.get("delivery_efficiency"))
        for name, path in [("drive_mean", ("drive", "mean")),
                           ("tau_x_rms", ("wall", "tau_x_rms")),
                           ("tau_x_int", ("wall", "tau_x_int")),
                           ("tau_mag_mean", ("wall", "tau_mag_mean"))]:
            v = r
            for p in path:
                v = (v or {}).get(p) if isinstance(v, dict) else None
            base = floor[name]
            if v is None or base["mean"] is None:
                item[name] = None
                continue
            delta = float(v) - base["mean"]
            item[name] = dict(
                value=float(v), delta=delta,
                in_floor_sd=(delta / base["sd"]) if base["sd"] else None,
                sensitivity=(delta / rms_del) if rms_del else None,
            )
        c = (r.get("wall") or {}).get("crossings") or []
        if c and floor["reattachment"]["mean"] is not None:
            x = c[0] if len(c) == 1 else c[-1]
            delta = x - floor["reattachment"]["mean"]
            item["reattachment"] = dict(
                value=x, delta=delta,
                in_floor_sd=(delta / floor["reattachment"]["sd"])
                if floor["reattachment"]["sd"] else None,
                sensitivity=(delta / rms_del) if rms_del else None,
            )
        responses.append(item)

    return dict(floor=floor, responses=responses,
                n_null=len(nulls), n_perturbed=len(perts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="codes/results/modal_sensitivity")
    ap.add_argument("--out", default="codes/results/modal_sensitivity_l0_20260825")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_dir():
        print(f"no results yet at {root}", file=sys.stderr)
        return 2

    records, provenance = [], {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name.startswith("_"):
            continue
        rec = harvest_run(run_dir)
        if rec is None:
            print(f"  incomplete: {run_dir.name}", file=sys.stderr)
            continue
        records.append(rec)
        for f in ("run_config.json", "modalpert.log", "drive.log"):
            p = run_dir / f
            if p.exists():
                provenance[f"{run_dir.name}/{f}"] = sha256(p)

    if not records:
        print("no complete runs found", file=sys.stderr)
        return 3

    summary = reduce_campaign(records)

    # Identity tests deposited by the self-gating batches.
    identity = []
    for p in sorted(root.glob("_identity*/identity_test.json")):
        identity.append(json.loads(p.read_text()))

    payload = dict(
        experiment="mode-resolved wall-traction sensitivity of coupled WMLES",
        case="Xiao periodic hill, Re_H=5600, G1 307200 cells, equilibrium arm",
        branch_point_time=405.0,
        n_runs=len(records),
        identity_tests=identity,
        summary=summary,
        runs=records,
        provenance_sha256=provenance,
    )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    npz = {}
    for r in records:
        prof = r.pop("_wall_profile", None)
        if prof:
            npz[f"{r['run_id']}__x"] = np.array(prof["x"])
            npz[f"{r['run_id']}__tau_x"] = np.array(prof["tau_x"])
    if npz:
        np.savez(out.with_suffix(".npz"), **npz)

    print(json.dumps(summary["floor"], indent=2))
    print(f"\n{len(records)} runs harvested "
          f"({summary['n_null']} null, {summary['n_perturbed']} perturbed)")
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
