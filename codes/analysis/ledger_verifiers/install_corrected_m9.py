#!/usr/bin/env python3
"""Install the verified remote M9 rebuild without losing the superseded case.

The corrected raw fields stay on ARCHER2.  ``down`` transfers the compact
``codes/results/m9_corrected`` bundle, whose receipt binds the producing Slurm
job, all raw source hashes and both result files.  This installer first stages
and checks those bytes, preserves the one-pitch/laminar-upper-half certificate,
and only then atomically replaces the canonical JSON/NPZ pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "codes/results"
BUNDLE = RESULTS / "m9_corrected"
LEGACY = RESULTS / "legacy_laminar_upper_half_rib_20260823"
NAMES = ("direct_force_adequacy_certificate_l1.json",
         "direct_force_adequacy_certificate_l1.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config_path = BUNDLE / "rebuild_config.json"
    receipt = BUNDLE / "REMOTE_REBUILD_COMPLETE"
    if not config_path.is_file() or not receipt.is_file():
        raise SystemExit("corrected M9 remote bundle is incomplete")
    config = json.loads(config_path.read_text())
    if config.get("status") != "M9_CORRECTED_RIB_REBUILD_OK":
        raise SystemExit("corrected M9 producer did not pass")
    if str(config.get("slurm_job_id", "")) not in receipt.read_text():
        raise SystemExit("Slurm receipt does not bind the recorded producer")
    for name in NAMES:
        path = BUNDLE / name
        expected = config.get("result_sha256", {}).get(name)
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"bundle hash mismatch: {name}")
    summary = json.loads((BUNDLE / NAMES[0]).read_text())
    if (summary.get("status") != "PASS" or
            summary.get("mesh", {}).get("cells", 0) <= 94976 or
            summary.get("source_hashes") != config.get("source_hashes")):
        raise SystemExit("corrected summary fails schema/substrate checks")

    LEGACY.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        current = RESULTS / name
        preserved = LEGACY / name
        if current.exists() and not preserved.exists():
            shutil.copy2(current, preserved)

    with tempfile.TemporaryDirectory(prefix="m9-install-", dir=RESULTS) as tmp:
        stage = Path(tmp)
        for name in NAMES:
            shutil.copy2(BUNDLE / name, stage / name)
            if sha256(stage / name) != config["result_sha256"][name]:
                raise SystemExit(f"staging hash mismatch: {name}")
        for name in NAMES:
            os.replace(stage / name, RESULTS / name)
    print("M9_CORRECTED_INSTALL_OK job={} cells={} phases={}".format(
        config["slurm_job_id"], config["cells"],
        config["phase_control_volumes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
