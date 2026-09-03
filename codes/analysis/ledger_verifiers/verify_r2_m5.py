#!/usr/bin/env python3
"""Stable guard for Reviewer-2 minor point 5 (exact-stress interpretation).

The DNS-stress substitution is an intentionally inconsistent intervention: the
stress comes from the complete DNS balance but is inserted into an equation that
still omits convection.  It can show that a better local stress does not restore
discarded transport; it cannot, by itself, rank models that add that transport.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
TEX = ROOT / "manuscript" / "main.tex"
PDF = ROOT / "manuscript" / "main.pdf"
DATA = ROOT / "codes" / "results" / "diagnostic_test_corrected.npz"


def active_tex(text: str) -> str:
    """Remove TeX blocks disabled by literal ``\\iffalse ... \\fi`` guards."""
    out: list[str] = []
    depth = 0
    for line in text.splitlines():
        if re.search(r"\\iffalse\b", line):
            depth += 1
            continue
        if depth and re.search(r"\\fi\b", line):
            depth -= 1
            continue
        if depth == 0:
            out.append(line)
    if depth:
        raise AssertionError("unbalanced \\iffalse/\\fi guards")
    return "\n".join(out)


def require(ok: bool, label: str, checks: list[str]) -> None:
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main() -> int:
    checks: list[str] = []
    require(TEX.exists() and PDF.exists() and DATA.exists(), "required artifacts exist", checks)

    source = active_tex(TEX.read_text())
    low = re.sub(r"\s+", " ", source).lower()
    require("worsening is expected" in low,
            "active source says the inconsistent substitution is expected to worsen", checks)
    require("strictly narrower" in low and "local} stress" in low,
            "active source narrows the inference to local stress", checks)
    require("cannot restore the transport discarded by that equation" in low,
            "active source states the permitted conclusion", checks)
    require("does not bear on models that \\emph{add}" in low,
            "active source excludes convection-adding models from the inference", checks)
    require("changing only the local stress closure is insufficient in this test" in low,
            "active summary carries the narrow wording", checks)
    require(not re.search(r"exact dns stress.{0,180}(proves|demonstrates|shows).{0,80}closure.independent", low),
            "active source makes no exact-stress closure-independence claim", checks)

    with np.load(DATA, allow_pickle=False) as z:
        require(abs(float(z["standard_ml_r2"]) + 47.68617253416459) < 1e-10,
                "baseline R2 is rebuilt from the corrected surface-aware artifact", checks)
        require(abs(float(z["controlled_dns_r2"]) + 482.97683570594313) < 1e-9,
                "DNS-stress R2 is rebuilt from the same corrected protocol", checks)

    require(PDF.stat().st_mtime_ns >= TEX.stat().st_mtime_ns,
            "compiled PDF is not older than the source", checks)
    proc = subprocess.run(["pdftotext", str(PDF), "-"], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    pdf = re.sub(r"\s+", " ", proc.stdout).lower()
    require("worsening is expected" in pdf,
            "compiled PDF carries the expected-worsening statement", checks)
    require("strictly narrower" in pdf and
            "cannot restore the transport discarded by that equation" in pdf,
            "compiled PDF carries the narrow inference", checks)
    require("does not bear on models that add convective terms" in pdf,
            "compiled PDF excludes convection-adding models", checks)
    require("−927" not in proc.stdout and "-927" not in proc.stdout,
            "superseded/excluded exact-stress protocol is absent from the active PDF", checks)

    print(f"R2-m5: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R2-m5: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
