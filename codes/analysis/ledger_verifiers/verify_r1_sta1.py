#!/usr/bin/env python3
"""Stable wrapper for the R1-STA-1 uncited-claims disposition audit."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "codes" / "analysis" / "audit_uncited_claims_r1sta1.py"

if __name__ == "__main__":
    result = subprocess.run([sys.executable, str(AUDIT)], cwd=ROOT)
    sys.exit(result.returncode)
