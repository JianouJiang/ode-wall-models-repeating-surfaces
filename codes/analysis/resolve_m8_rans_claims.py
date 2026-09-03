#!/usr/bin/env python3
"""Record the removal of load-bearing steady-RANS claims for claim M8."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
TEX = ROOT / "manuscript/main.tex"
PDF = ROOT / "manuscript/main.pdf"
RIB = ROOT / "codes/results/rib_les_dtype_apriori.npz"
STATIONARITY = ROOT / "codes/results/rib_les_stationarity.json"
OUTPUT = ROOT / "codes/results/rans_claim_removal_m8.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_pdf_text(path: pathlib.Path) -> str:
    """Bind a PDF by its extracted text: an unchanged-source rebuild alters
    only metadata (CreationDate, /ID), which must not break the binding."""
    text = subprocess.run(["pdftotext", str(path), "-"],
                          capture_output=True, text=True, check=True).stdout
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def active_tex(text: str) -> str:
    kept: list[str] = []
    depth = 0
    for line in text.splitlines():
        if "\\iffalse" in line:
            depth += 1
            continue
        if depth and re.search(r"\\fi(?:\s|$)", line):
            depth -= 1
            continue
        if depth == 0:
            kept.append(line)
    if depth:
        raise RuntimeError("unclosed \\iffalse block")
    return "\n".join(kept)


def main() -> None:
    source = active_tex(TEX.read_text())
    pdf_text = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages = int(
        re.search(
            r"^Pages:\s+(\d+)$",
            subprocess.run(
                ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
            ).stdout,
            re.MULTILINE,
        ).group(1)
    )
    with np.load(RIB, allow_pickle=False) as data:
        retained = {
            "case": str(data["case"]),
            "fidelity": str(data["fidelity"]),
            "n_stations": int(data["n_profiles"]),
            "resolved_shear_fraction_median": float(data["f_res_band_median"]),
            "r2_modelled_stress": float(data["standard_ml_r2"]),
            "relrms_modelled_stress": float(data["standard_ml_relRMS"]),
            "r2_resolved_stress": float(data["controlled_dns_r2"]),
            "relrms_resolved_stress": float(data["controlled_dns_relRMS"]),
            "epsilon_median": float(data["eps_median"]),
        }
    stationarity = json.loads(STATIONARITY.read_text())
    retained["window_r2"] = [
        float(row["standard_ml_r2"]) for row in stationarity["windows"]
    ]
    retained["window_epsilon_median"] = [
        float(row["eps_median"]) for row in stationarity["windows"]
    ]

    contexts = [line.strip() for line in pdf_text.splitlines() if "RANS" in line]
    forbidden = [
        "fig_cube_array",
        "fig_transition_map_l2",
        "cube-array RANS",
        "k-type RANS",
        "wavy wall (RANS)",
        "fillet-radius series",
        "mechanism survives three-dimensionality",
    ]
    for token in forbidden:
        if token in source or token in pdf_text:
            raise RuntimeError(f"load-bearing RANS token remains active: {token}")
    # Wrap-robust context policy (2026-08-23): every compiled RANS mention must
    # be an exclusion statement, the related-work literature mention, or a
    # reference-list title -- never a performance claim.
    allowed_tokens = (
        "Unmatched RANS comparisons do not enter the active",
        "No steady-RANS result is used to infer transfer",
        "Reynolds-averaged Navier",
        "(RANS) wall-function tradition",
        "Compound wall treatment for RANS",
    )
    bad = [line for line in contexts
           if not any(tok in line for tok in allowed_tokens)]
    if bad or not any("Unmatched RANS comparisons" in c for c in contexts) \
           or not any("No steady-RANS result" in c for c in contexts):
        raise RuntimeError(f"unexpected compiled RANS contexts: {bad!r}")
    if pages > 20:
        raise RuntimeError(f"body is {pages} pages")

    summary = {
        "schema": "rans-claim-removal-m8-v1",
        "ledger_row": "M8",
        "resolution": "deleted_not_demoted",
        "removed_from_active_evidence": [
            "steady-RANS cube array and pitch/Reynolds sweeps",
            "steady-RANS fillet-radius sweep",
            "steady-RANS wavy-wall transition point",
            "unmatched steady-RANS k-type rib",
        ],
        "retained_reference_case": retained,
        "compiled_rans_contexts": contexts,
        "body_pages": pages,
        "sha256": {
            "manuscript_main_tex": sha256(TEX),
            "manuscript_main_pdf": sha256_pdf_text(PDF),
            "rib_les_dtype_apriori_npz": sha256(RIB),
            "rib_les_stationarity_json": sha256(STATIONARITY),
        },
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"M8 active evidence: WRLES rib only; {pages}-page body")


if __name__ == "__main__":
    main()
