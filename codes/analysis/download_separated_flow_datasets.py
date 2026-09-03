#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULTS_DIR = Path(
    "/home/jianoujiang/Desktop/paper-factory/projects/pde_wall_model/"
    "codes/vendor/universal_wall_function/codes/results"
)
RAW_DIR = RESULTS_DIR / "raw_public_downloads"

FIELD_KEYS = ["y", "y_plus", "U", "U_plus", "V", "P", "uu", "vv", "ww", "uv"]
SCALAR_KEYS = ["x", "dp_dx", "tau_w", "u_tau", "Re_tau", "Re", "nu", "is_separated"]


@dataclass
class SummaryRow:
    geometry: str
    n_stations: int
    f_sep: float
    re_value: str
    source: str


def fetch_url(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    print(f"Downloading {dest.name} ...")
    data = fetch_url(url, timeout=timeout)
    dest.write_bytes(data)
    return dest


def page_link(page_url: str, needle: str) -> str:
    text = fetch_url(page_url, timeout=60).decode("utf-8", "ignore")
    match = re.search(r'href="([^"]*%s[^"]*)"' % re.escape(needle), text, re.I)
    if not match:
        raise RuntimeError(f"Could not find link containing {needle!r} on {page_url}")
    return urllib.request.urljoin(page_url, match.group(1))


def unzip_first_member(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = zf.namelist()[0]
        out_path = dest_dir / Path(member).name
        if not out_path.exists():
            print(f"Extracting {member} ...")
            out_path.write_bytes(zf.read(member))
        return out_path


def separated_flag(tau_w: float) -> bool:
    return bool((not np.isfinite(tau_w)) or tau_w < 0.0 or abs(tau_w) < 1e-6)


def load_ascii_numeric(path: Path, min_cols: int = 2) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("TITLE", "VARIABLES", "ZONE", "DT=", "#", "%")):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < min_cols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No numeric rows parsed from {path}")
    return np.asarray(rows, dtype=float)


def parse_tecplot_ascii(path: Path) -> dict[str, Any]:
    var_names: list[str] = []
    ni = nj = None
    data_offset = None

    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            text = line.decode("utf-8", "ignore").strip()
            if re.match(r"^variables\s*=", text, re.I) or text.startswith('"'):
                var_names.extend(re.findall(r'"([^"]+)"', text))
            if " I=" in f" {text}" or " J=" in f" {text}":
                mi = re.search(r"\bI\s*=\s*(\d+)", text, re.I)
                mj = re.search(r"\bJ\s*=\s*(\d+)", text, re.I)
                if mi:
                    ni = int(mi.group(1))
                if mj:
                    nj = int(mj.group(1))
            if text.startswith("DT="):
                data_offset = handle.tell()
                break

    if ni is None or nj is None or data_offset is None:
        raise RuntimeError(f"Failed to parse Tecplot header from {path}")

    nvars = len(var_names)
    total = ni * nj
    with path.open("rb") as handle:
        handle.seek(data_offset)
        values = np.fromfile(handle, dtype=float, sep=" ")
    expected = nvars * total
    if values.size < expected:
        raise RuntimeError(f"{path}: parsed {values.size} values, expected at least {expected}")

    values = values[:expected]
    data: dict[str, Any] = {"ni": ni, "nj": nj, "var_names": var_names}
    for iv, name in enumerate(var_names):
        block = values[iv * total : (iv + 1) * total]
        data[name] = block.reshape((nj, ni))
    return data


def pad_profiles(profiles: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    max_n = max(len(np.asarray(p["y"])) for p in profiles)
    for key in FIELD_KEYS:
        arr = np.full((len(profiles), max_n), np.nan, dtype=float)
        for i, prof in enumerate(profiles):
            vals = np.asarray(prof[key], dtype=float)
            arr[i, : len(vals)] = vals
        out[key] = arr
    for key in SCALAR_KEYS:
        arr = np.zeros(len(profiles), dtype=bool if key == "is_separated" else float)
        if key != "is_separated":
            arr[:] = np.nan
        for i, prof in enumerate(profiles):
            arr[i] = prof[key]
        out[key] = arr
    out["n_points"] = np.asarray([len(np.asarray(p["y"])) for p in profiles], dtype=int)
    return out


def save_profiles_npz(
    geometry: str,
    re_identifier: str,
    profiles: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> SummaryRow:
    payload = pad_profiles(profiles)
    payload["geometry"] = np.array(geometry)
    payload["re_identifier"] = np.array(re_identifier)
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    out_path = RESULTS_DIR / f"{geometry}_{re_identifier}_wall_profiles.npz"
    np.savez(out_path, **payload)
    f_sep = float(np.mean(payload["tau_w"] < 0.0))
    print(f"Saved {out_path.name}: {len(profiles)} stations, f_sep = {f_sep:.3f}")
    return SummaryRow(
        geometry=f"{geometry}_{re_identifier}",
        n_stations=len(profiles),
        f_sep=f_sep,
        re_value=str(metadata.get("Re", re_identifier)),
        source=metadata.get("source_page", metadata.get("source_dataset", "")),
    )


def finalize_profile(
    *,
    x: float,
    y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray | None,
    P: np.ndarray | None,
    uu: np.ndarray | None,
    vv: np.ndarray | None,
    ww: np.ndarray | None,
    uv: np.ndarray | None,
    nu: float,
    re_value: float,
    tau_w: float | None = None,
    dp_dx: float | None = None,
    y_plus_override: np.ndarray | None = None,
    u_plus_override: np.ndarray | None = None,
    re_tau_override: float | None = None,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    U = np.asarray(U, dtype=float)
    order = np.argsort(y)
    y = y[order]
    U = U[order]
    y = y - y[0]

    def ordered_or_nan(arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.full_like(y, np.nan)
        arr = np.asarray(arr, dtype=float)[order]
        n = min(len(arr), len(y))
        out = np.full_like(y, np.nan)
        out[:n] = arr[:n]
        return out

    V_arr = ordered_or_nan(V)
    P_arr = ordered_or_nan(P)
    uu_arr = ordered_or_nan(uu)
    vv_arr = ordered_or_nan(vv)
    ww_arr = ordered_or_nan(ww)
    uv_arr = ordered_or_nan(uv)

    if tau_w is None:
        if len(y) >= 2 and y[1] > y[0]:
            tau_w = float(nu * (U[1] - U[0]) / (y[1] - y[0]))
        else:
            tau_w = float("nan")
    tau_w = float(tau_w)
    u_tau = math.copysign(math.sqrt(abs(tau_w)), tau_w) if np.isfinite(tau_w) else np.nan

    if y_plus_override is not None:
        y_plus = np.asarray(y_plus_override, dtype=float)[order]
    elif np.isfinite(u_tau) and abs(u_tau) > 1e-14:
        y_plus = y * abs(u_tau) / nu
    else:
        y_plus = np.full_like(y, np.nan)

    if u_plus_override is not None:
        U_plus = np.asarray(u_plus_override, dtype=float)[order]
    elif np.isfinite(u_tau) and abs(u_tau) > 1e-14:
        U_plus = U / u_tau
    else:
        U_plus = np.full_like(y, np.nan)

    if re_tau_override is not None:
        re_tau = float(re_tau_override)
    elif np.isfinite(u_tau):
        re_tau = float(abs(u_tau) / nu)
    else:
        re_tau = np.nan

    return {
        "y": y,
        "y_plus": y_plus,
        "U": U,
        "U_plus": U_plus,
        "V": V_arr,
        "P": P_arr,
        "uu": uu_arr,
        "vv": vv_arr,
        "ww": ww_arr,
        "uv": uv_arr,
        "x": float(x),
        "dp_dx": float(dp_dx) if dp_dx is not None else np.nan,
        "tau_w": tau_w,
        "u_tau": u_tau,
        "Re_tau": re_tau,
        "Re": float(re_value),
        "nu": float(nu),
        "is_separated": separated_flag(tau_w),
    }


def structured_profiles_from_columns(
    data: dict[str, Any],
    *,
    x_name: str,
    y_name: str,
    u_name: str,
    v_name: str | None,
    p_name: str | None,
    uu_name: str | None,
    vv_name: str | None,
    ww_name: str | None,
    uv_name: str | None,
    nu: float,
    re_value: float,
    target_stations: int = 60,
    x_min: float | None = None,
    x_max: float | None = None,
    surface_x: np.ndarray | None = None,
    surface_tau_w: np.ndarray | None = None,
    surface_dpdx: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    x2d = np.asarray(data[x_name], dtype=float)
    y2d = np.asarray(data[y_name], dtype=float)
    stream_axis = 1 if len(np.unique(np.round(x2d[0, :], 12))) >= len(np.unique(np.round(x2d[:, 0], 12))) else 0
    n_stream = x2d.shape[stream_axis]
    stride = max(1, n_stream // target_stations)
    profiles: list[dict[str, Any]] = []

    for idx in range(0, n_stream, stride):
        if stream_axis == 1:
            x_line = x2d[:, idx]
            y_line = y2d[:, idx]
            getv = lambda name: None if name is None else np.asarray(data[name], dtype=float)[:, idx]
        else:
            x_line = x2d[idx, :]
            y_line = y2d[idx, :]
            getv = lambda name: None if name is None else np.asarray(data[name], dtype=float)[idx, :]

        x_station = float(np.nanmedian(x_line))
        if x_min is not None and x_station < x_min:
            continue
        if x_max is not None and x_station > x_max:
            continue

        tau_w = None
        dp_dx = None
        if surface_x is not None and surface_tau_w is not None:
            tau_w = float(np.interp(x_station, surface_x, surface_tau_w))
        if surface_x is not None and surface_dpdx is not None:
            dp_dx = float(np.interp(x_station, surface_x, surface_dpdx))

        profiles.append(
            finalize_profile(
                x=x_station,
                y=np.asarray(y_line, dtype=float),
                U=np.asarray(getv(u_name), dtype=float),
                V=getv(v_name),
                P=getv(p_name),
                uu=getv(uu_name),
                vv=getv(vv_name),
                ww=getv(ww_name),
                uv=getv(uv_name),
                nu=nu,
                re_value=re_value,
                tau_w=tau_w,
                dp_dx=dp_dx,
            )
        )
    return profiles


def build_gaussian_speed_bump() -> list[SummaryRow]:
    page = "https://tmbwg.github.io/turbmodels/Other_DNS_Data/gaussianbump_nasadns.html"
    out: list[SummaryRow] = []
    cases = [
        ("Re1M", 1.0e6, "SpeedBump-ReL-1M", "speedbump-rel-1m-statistics-dat.zip"),
        ("Re2M", 2.0e6, "SpeedBump-ReL-2M", "speedbump-rel-2m-statistics-dat.zip"),
    ]
    for re_id, re_value, stem, zip_name in cases:
        cf_path = download_file(
            f"https://tmbwg.github.io/turbmodels/Other_DNS_Data/Gaussianbump_nasadns/{stem}-Cf.dat",
            RAW_DIR / "gaussian_bump" / f"{stem}-Cf.dat",
        )
        cp_path = download_file(
            f"https://tmbwg.github.io/turbmodels/Other_DNS_Data/Gaussianbump_nasadns/{stem}-Cp.dat",
            RAW_DIR / "gaussian_bump" / f"{stem}-Cp.dat",
        )
        zip_path = download_file(
            f"https://www.nasa.gov/wp-content/uploads/2025/11/{zip_name}",
            RAW_DIR / "gaussian_bump" / zip_name,
            timeout=1200,
        )
        dat_path = unzip_first_member(zip_path, RAW_DIR / "gaussian_bump")

        print(f"Parsing Gaussian speed bump {re_id} field ...")
        data = parse_tecplot_ascii(dat_path)
        cf = load_ascii_numeric(cf_path, min_cols=2)
        cp = load_ascii_numeric(cp_path, min_cols=2)
        sx = cf[:, 0]
        tau_w = 0.5 * cf[:, 1]
        dp_dx = np.gradient(0.5 * cp[:, 1], cp[:, 0])

        profiles = structured_profiles_from_columns(
            data,
            x_name="x/L",
            y_name="y/L",
            u_name="U/U_inf",
            v_name="V/U_inf",
            p_name="P/p_inf" if "P/p_inf" in data else "p/p_inf",
            uu_name="uu/U_inf^2",
            vv_name="vv/U_inf^2",
            ww_name="ww/U_inf^2",
            uv_name="uv/U_inf^2",
            nu=1.0 / re_value,
            re_value=re_value,
            target_stations=64,
            surface_x=sx,
            surface_tau_w=tau_w,
            surface_dpdx=dp_dx,
        )
        f_sep = float(np.mean([p["tau_w"] < 0.0 for p in profiles]))
        print(f"Found {len(profiles)} stations, f_sep = {f_sep:.3f}")
        meta = {
            "source_page": page,
            "source_type": "downloaded_tecplot_zip_plus_surface_ascii",
            "Re": re_value,
            "geometry": "gaussian_speed_bump",
            "normalization": "x/L, y/L, U/U_inf, p/p_inf, rho=1, nu=1/Re_L",
            "surface_cf_file": str(cf_path),
            "surface_cp_file": str(cp_path),
            "field_file": str(dat_path),
        }
        out.append(save_profiles_npz("gaussian_speed_bump", re_id, profiles, meta))
        del data, profiles
    return out


def build_conv_div_channel() -> list[SummaryRow]:
    page = "https://tmbwg.github.io/turbmodels/Other_DNS_Data/conv-div-channel12600.html"
    zip_path = download_file(
        "https://www.nasa.gov/wp-content/uploads/2025/11/conv-div-mean-dat.zip",
        RAW_DIR / "conv_div" / "conv-div-mean-dat.zip",
        timeout=1200,
    )
    dat_path = unzip_first_member(zip_path, RAW_DIR / "conv_div")
    surf_path = download_file(
        "https://tmbwg.github.io/turbmodels/Other_DNS_Data/Conv-div-channel/statistics_streamwise.dat",
        RAW_DIR / "conv_div" / "statistics_streamwise.dat",
    )
    print("Parsing converging-diverging channel field ...")
    data = parse_tecplot_ascii(dat_path)
    surf = load_ascii_numeric(surf_path, min_cols=8)
    sx = surf[:, 0]
    cp0 = surf[:, 4]
    cf0 = surf[:, 6]
    tau_w = 0.5 * cf0
    dp_dx = np.gradient(0.5 * cp0, sx)

    profiles = structured_profiles_from_columns(
        data,
        x_name="X",
        y_name="Y",
        u_name="mean_u",
        v_name="mean_v",
        p_name=None,
        uu_name="reynolds_stress_uu",
        vv_name="reynolds_stress_vv",
        ww_name="reynolds_stress_ww",
        uv_name="reynolds_stress_uv",
        nu=1.0 / 12600.0,
        re_value=12600.0,
        target_stations=64,
        x_min=float(np.min(sx)),
        x_max=float(np.max(sx)),
        surface_x=sx,
        surface_tau_w=tau_w,
        surface_dpdx=dp_dx,
    )
    f_sep = float(np.mean([p["tau_w"] < 0.0 for p in profiles]))
    print(f"Found {len(profiles)} stations, f_sep = {f_sep:.3f}")
    meta = {
        "source_page": page,
        "source_type": "downloaded_tecplot_zip_plus_surface_ascii",
        "Re": 12600.0,
        "geometry": "conv_div_channel",
        "normalization": "channel half-height and U0 normalized, nu=1/Re",
        "field_file": str(dat_path),
        "surface_file": str(surf_path),
    }
    return [save_profiles_npz("conv_div_channel", "Re12600", profiles, meta)]


def build_separation_bubble_family() -> list[SummaryRow]:
    out: list[SummaryRow] = []
    page = "https://tmbwg.github.io/turbmodels/Other_DNS_Data/separation_bubble_2d.html"
    cases = [
        ("caseA", "Qofx_CaseA_xavg.dat", "qofxy-casea-xavg-dat.zip", "Qofxy_CaseA_xavg.dat"),
        ("caseB", "Qofx_CaseB_xavg.dat", "qofxy-caseb-xavg-dat.zip", "Qofxy_CaseB_xavg.dat"),
        ("caseE", "Qofx_CaseE.dat", "qofxy-casee-dat.zip", "Qofxy_CaseE.dat"),
    ]
    for case_id, qx_name, zip_name, _ in cases:
        qx_path = download_file(
            f"https://tmbwg.github.io/turbmodels/Other_DNS_Data/Separation_bubble_2d/{qx_name}",
            RAW_DIR / "separation_bubble" / qx_name,
        )
        zip_path = download_file(
            f"https://www.nasa.gov/wp-content/uploads/2025/11/{zip_name}",
            RAW_DIR / "separation_bubble" / zip_name,
            timeout=1200,
        )
        dat_path = unzip_first_member(zip_path, RAW_DIR / "separation_bubble")
        print(f"Parsing separation bubble {case_id} field ...")
        data = parse_tecplot_ascii(dat_path)
        qx = load_ascii_numeric(qx_path, min_cols=9)
        sx = qx[:, 0]
        cp = qx[:, 1]
        theta = qx[:, 5]
        rtheta = qx[:, 6]
        cf = qx[:, 8]
        finite = np.isfinite(theta) & np.isfinite(rtheta) & (theta > 0)
        nu = float(np.median(theta[finite] / rtheta[finite]))
        re_ref = float(np.median(rtheta[finite]))
        tau_w = 0.5 * cf
        dp_dx = np.gradient(0.5 * cp, sx)
        names = data["var_names"]
        has_pressure = any("P/" in name or name == "P" for name in names)
        p_name = next((n for n in names if "P/" in n or n == "P"), None)
        uv_name = next((n for n in names if "u'v'" in n), None)
        if uv_name is None:
            uv_name = next((n for n in names if "uv" in n.lower()), None)
        profiles = structured_profiles_from_columns(
            data,
            x_name="x",
            y_name="y",
            u_name="U",
            v_name="V",
            p_name=p_name if has_pressure else None,
            uu_name=next((n for n in names if "u'u'" in n), None),
            vv_name=next((n for n in names if "v'v'" in n), None),
            ww_name=next((n for n in names if "w'w'" in n), None),
            uv_name=uv_name,
            nu=nu,
            re_value=re_ref,
            target_stations=56,
            surface_x=sx,
            surface_tau_w=tau_w,
            surface_dpdx=dp_dx,
        )
        if uv_name and uv_name.startswith("-"):
            for prof in profiles:
                prof["uv"] = -prof["uv"]
        f_sep = float(np.mean([p["tau_w"] < 0.0 for p in profiles]))
        print(f"Found {len(profiles)} stations, f_sep = {f_sep:.3f}")
        meta = {
            "source_page": page,
            "source_type": "downloaded_tecplot_zip_plus_surface_ascii",
            "geometry": "separation_bubble",
            "Re": re_ref,
            "case": case_id,
            "field_file": str(dat_path),
            "surface_file": str(qx_path),
            "normalization": "all variables normalized by U_inf and theta0-scale Y per TMR README; nu inferred from theta/Rtheta",
        }
        out.append(save_profiles_npz("separation_bubble", case_id, profiles, meta))
    return out


def build_swept_bubble() -> list[SummaryRow]:
    page = "https://tmbwg.github.io/turbmodels/Other_DNS_Data/separation_bubble_swept.html"
    qx_path = download_file(
        "https://tmbwg.github.io/turbmodels/Other_DNS_Data/Separation_bubble_swept/Qofx_CaseC35_xavg.dat",
        RAW_DIR / "separation_bubble_swept" / "Qofx_CaseC35_xavg.dat",
    )
    zip_path = download_file(
        "https://www.nasa.gov/wp-content/uploads/2025/11/qofxy-casec35-xavg-dat.zip",
        RAW_DIR / "separation_bubble_swept" / "qofxy-casec35-xavg-dat.zip",
        timeout=1200,
    )
    dat_path = unzip_first_member(zip_path, RAW_DIR / "separation_bubble_swept")
    print("Parsing swept separation bubble field ...")
    data = parse_tecplot_ascii(dat_path)
    qx = load_ascii_numeric(qx_path, min_cols=13)
    sx = qx[:, 0]
    cp = qx[:, 1]
    theta = qx[:, 4]
    rtheta = qx[:, 9]
    cf = qx[:, 11]
    finite = np.isfinite(theta) & np.isfinite(rtheta) & (theta > 0)
    nu = float(np.median(theta[finite] / rtheta[finite]))
    re_ref = float(np.median(rtheta[finite]))
    tau_w = 0.5 * cf
    dp_dx = np.gradient(0.5 * cp, sx)
    names = data["var_names"]
    profiles = structured_profiles_from_columns(
        data,
        x_name="x",
        y_name="y",
        u_name="U",
        v_name="V",
        p_name=None,
        uu_name=next((n for n in names if "u'u'" in n), None),
        vv_name=next((n for n in names if "v'v'" in n), None),
        ww_name=next((n for n in names if "w'w'" in n), None),
        uv_name=next((n for n in names if "u'v'" in n), None),
        nu=nu,
        re_value=re_ref,
        target_stations=56,
        surface_x=sx,
        surface_tau_w=tau_w,
        surface_dpdx=dp_dx,
    )
    uv_name = next((n for n in names if "u'v'" in n), None)
    if uv_name and uv_name.startswith("-"):
        for prof in profiles:
            prof["uv"] = -prof["uv"]
    f_sep = float(np.mean([p["tau_w"] < 0.0 for p in profiles]))
    print(f"Found {len(profiles)} stations, f_sep = {f_sep:.3f}")
    meta = {
        "source_page": page,
        "source_type": "downloaded_tecplot_zip_plus_surface_ascii",
        "geometry": "separation_bubble_swept",
        "Re": re_ref,
        "field_file": str(dat_path),
        "surface_file": str(qx_path),
        "normalization": "all variables normalized by U_inf and theta0-scale Y per TMR README; nu inferred from theta/Rtheta",
    }
    return [save_profiles_npz("separation_bubble_swept", "caseC35", profiles, meta)]


def read_jaxa_tables(path: Path) -> list[str]:
    return path.read_text(errors="ignore").splitlines()


def parse_jaxa_streamwise(lines: list[str], start_header: str) -> np.ndarray:
    start = None
    for i, line in enumerate(lines):
        if start_header in line:
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"Could not locate streamwise section {start_header!r}")
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = stripped.split()
        try:
            vals = [float(x.replace("------------", "nan")) for x in parts]
        except ValueError:
            continue
        rows.append(vals)
    return np.asarray(rows, dtype=float)


def parse_jaxa_station_tables(lines: list[str]) -> dict[float, dict[str, np.ndarray]]:
    stations: dict[float, dict[str, np.ndarray]] = {}
    current_x = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*\d+\)\s*x/theta_0\s*=\s*([0-9.]+)", line)
        if m:
            current_x = float(m.group(1))
            stations.setdefault(current_x, {})
            i += 1
            continue
        if current_x is None:
            i += 1
            continue
        hdr = line.strip()
        if hdr.startswith("j"):
            rows: list[list[float]] = []
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if not stripped:
                    break
                parts = stripped.split()
                try:
                    vals = [float(x.replace("------------", "nan")) for x in parts[1:]]
                except ValueError:
                    break
                rows.append(vals)
                i += 1
            stations[current_x][hdr] = np.asarray(rows, dtype=float)
        i += 1
    return stations


def build_jaxa_bubbles() -> list[SummaryRow]:
    page = "https://jaxa-dns-database.jaxa.jp/separation.html"
    out: list[SummaryRow] = []
    for re_theta0 in [300, 600, 900]:
        path = download_file(
            f"https://jaxa-dns-database.jaxa.jp/separation/sep{re_theta0}LB.dat",
            RAW_DIR / "jaxa_separation" / f"sep{re_theta0}LB.dat",
        )
        print(f"Parsing JAXA sep{re_theta0}LB.dat ...")
        lines = read_jaxa_tables(path)
        mean_stream = parse_jaxa_streamwise(lines, "x/theta_0         C_f")
        cp_stream = parse_jaxa_streamwise(lines, "x/theta_0         C_p")
        stations = parse_jaxa_station_tables(lines)
        nu = 1.0 / re_theta0
        profiles: list[dict[str, Any]] = []
        sx_cf = mean_stream[:, 0]
        cf = mean_stream[:, 1]
        sx_cp = cp_stream[:, 0]
        cp = cp_stream[:, 1]
        theta_ratio = mean_stream[:, 2]

        for x_station, blocks in sorted(stations.items()):
            mean_hdr = next((k for k in blocks if "u_mean/U_0" in k and "y/theta_0" in k), None)
            vv_hdr = next((k for k in blocks if "vv/U_0^2" in k and "y/theta_0" in k), None)
            plus_hdr = next((k for k in blocks if "y+" in k and "u_mean+" in k), None)
            ydelta_hdr = next((k for k in blocks if "y/delta99" in k and "umean/U_e" in k), None)
            if mean_hdr is None:
                continue
            mean = blocks[mean_hdr]
            y = mean[:, 0]
            u = mean[:, 1]
            uu = mean[:, 2]
            ww = mean[:, 3]
            k = mean[:, 4]
            vv = np.full_like(y, np.nan)
            uv = np.full_like(y, np.nan)
            if vv_hdr is not None:
                vvuv = blocks[vv_hdr]
                n = min(len(vv), len(vvuv))
                vv[:n] = vvuv[:n, 1]
                uv[:n] = vvuv[:n, 2]
            if np.all(np.isnan(vv)):
                vv = 2.0 * k - uu - ww

            tau_w = 0.5 * float(np.interp(x_station, sx_cf, cf))
            dp_dx = float(np.interp(x_station, sx_cp, np.gradient(0.5 * cp, sx_cp)))
            u_tau = math.copysign(math.sqrt(abs(tau_w)), tau_w)

            y_plus = None
            u_plus = None
            re_tau = None
            if plus_hdr is not None:
                plus = blocks[plus_hdr]
                y_plus = plus[:, 0]
                u_plus = plus[:, 1]
                finite = np.isfinite(y_plus) & np.isfinite(y) & (y > 0)
                if np.any(finite):
                    wall_factor = float(np.median(y_plus[finite] / y[finite]))
                    delta99 = wall_factor * nu / abs(u_tau) if abs(u_tau) > 1e-14 else np.nan
                    re_tau = delta99 * abs(u_tau) / nu if np.isfinite(delta99) else np.nan
            if re_tau is None and ydelta_hdr is not None:
                yd = blocks[ydelta_hdr]
                finite = np.isfinite(yd[:, 0]) & np.isfinite(y) & (yd[:, 0] > 0)
                if np.any(finite):
                    delta99 = float(np.median(y[finite] / yd[finite, 0]))
                    re_tau = delta99 * abs(u_tau) / nu
            if re_tau is None:
                theta_local = float(np.interp(x_station, sx_cf, theta_ratio))
                re_tau = theta_local * abs(u_tau) / nu

            profiles.append(
                finalize_profile(
                    x=x_station,
                    y=y,
                    U=u,
                    V=np.zeros_like(y),
                    P=np.full_like(y, np.nan),
                    uu=uu,
                    vv=vv,
                    ww=ww,
                    uv=uv,
                    nu=nu,
                    re_value=float(re_theta0),
                    tau_w=tau_w,
                    dp_dx=dp_dx,
                    y_plus_override=y_plus,
                    u_plus_override=u_plus,
                    re_tau_override=re_tau,
                )
            )

        f_sep = float(np.mean([p["tau_w"] < 0.0 for p in profiles]))
        print(f"Found {len(profiles)} stations, f_sep = {f_sep:.3f}")
        meta = {
            "source_page": page,
            "source_type": "downloaded_ascii_report",
            "geometry": "jaxa_separation",
            "Re": float(re_theta0),
            "source_dataset": str(path),
            "normalization": "x/theta0, y/theta0, U/U0, nu=theta0*U0/Re_theta0 with theta0=U0=1",
        }
        out.append(save_profiles_npz("jaxa_separation", f"ReTheta{re_theta0}", profiles, meta))
    return out


def print_summary(rows: list[SummaryRow]) -> None:
    print("\nSummary")
    print("geometry | n_stations | f_sep | Re | source")
    for row in rows:
        print(f"{row.geometry} | {row.n_stations} | {row.f_sep:.3f} | {row.re_value} | {row.source}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[SummaryRow] = []
    summary.extend(build_gaussian_speed_bump())
    summary.extend(build_conv_div_channel())
    summary.extend(build_separation_bubble_family())
    summary.extend(build_swept_bubble())
    summary.extend(build_jaxa_bubbles())

    print("Skipping flattened speed bump: only public field file is Tecplot binary `.plt`, no ASCII/public parser implemented.")
    print("Skipping separation_bubble caseD: TMR page exposes only surface data, no public 2-D field file.")
    print("Skipping McConkey Kaggle DOI: no anonymous direct machine-readable download URL was available.")
    print_summary(summary)


if __name__ == "__main__":
    main()
