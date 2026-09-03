#!/usr/bin/env python3
"""Stable guard on the class maps.

Ledger row M17 was opened because the class maps carried UNRUN verdict badges
from a hard-coded literal table, and was resolved by deleting both maps under
the supervisor's "cut, do not defend" instruction.

That is still true of those two maps, and the first block of checks below keeps
them out.  But this file used to test the row by FILENAME alone, and a class
map was later reinstated as figure 2 under a new name (`fig_class_map`), so the
row went on passing 7/7 while the object it was about was back in the build --
with its badges once again typed in as literals.  Two of them had drifted by
2026-08-27: the Gaussian bump printed +0.98 against the archive's 0.9747 and
the steep sinusoid +0.81 against its finest grid's 0.798.

So the second block tests the PROPERTY the row is actually about: any class map
in the active build must take every badge from an artifact, and the compiled
figure must print what those artifacts say.  A future map under a third name is
caught by the same check.
"""
from __future__ import annotations

import subprocess
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT = ROOT / "manuscript"
MAIN = MANUSCRIPT / "main.tex"
SUPPLEMENT = MANUSCRIPT / "supplementary.tex"
PDF = MANUSCRIPT / "main.pdf"
FLS = MANUSCRIPT / "main.fls"
SUMMARY = ROOT / "codes/results/classmap_provenance_m17.json"
LEDGER = ROOT / "REFEREE_POINT_LEDGER.md"

FORBIDDEN = (
    "fig_repeating_class_morphology",
    "fig_repeating_class_amplitude",
    "fig:class_morphology",
    "fig:class_amplitude",
)

# The class map that IS in the build, and the generator that must derive rather
# than declare its badges.
LIVE_MAP_PDF = MANUSCRIPT / "figures" / "fig_class_map.pdf"
LIVE_MAP_SRC = ROOT / "codes" / "figures" / "fig_class_map.py"


def live_map_badges():
    """Recompute figure 2's badges from the artifacts and read what it printed.

    Returns (expected, printed, bound) -- bound is False if the generator
    declares any score as a literal instead of reading it.
    """
    import re
    sys.path.insert(0, str(ROOT / "codes" / "figures"))
    import importlib
    mod = importlib.import_module("fig_class_map")
    importlib.reload(mod)
    expected = sorted(mod._badge(k)[1] for k in mod.S)
    printed = subprocess.run(["pdftotext", "-layout", str(LIVE_MAP_PDF), "-"],
                             check=True, capture_output=True, text=True).stdout
    printed = re.sub(r"\s+", " ", printed)
    src = LIVE_MAP_SRC.read_text(encoding="utf-8")
    cells = src[src.index("SMOOTH = ["):src.index("def geometry(")]
    # a typed score in the cell tables is the defect this row exists for
    bound = not re.search(r'"[+-]\d+\.\d+"', cells)
    return expected, printed, bound



def body_pages(pdf_path) -> int:
    """Pages of the body: front matter through conclusions, references excluded.

    The operative length target (user directive, 2026-08-25 16:10) is stated in
    BODY pages, not total pages, and it replaced the earlier supervisor figure
    of 20.  Measuring the wrong quantity against a superseded number is how a
    stale target keeps propagating, so the count is taken here from the
    compiled text.
    """
    import re
    import subprocess as _sp
    txt = _sp.run(["pdftotext", "-layout", str(pdf_path), "-"],
                  capture_output=True, text=True).stdout
    for i, page in enumerate(txt.split("\f"), 1):
        if re.search(r"^\s*REFERENCES\s*$|^\s*References\s*$", page, re.M):
            return i - 1
    return len(txt.split("\f"))


def main() -> int:
    source = MAIN.read_text(encoding="utf-8")
    supplementary = SUPPLEMENT.read_text(encoding="utf-8")
    fls = FLS.read_text(encoding="utf-8") if FLS.exists() else ""
    pdf_text = subprocess.run(
        ["pdftotext", str(PDF), "-"], check=True, capture_output=True, text=True
    ).stdout
    info = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
    ).stdout
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pages = int(next(line.split(":", 1)[1] for line in info.splitlines()
                     if line.startswith("Pages:")))
    suspension_is_live = (
        "--closed-row-replay" in sys.argv
        and "OPEN — gate suspended by user directive 2026-08-23 15:45"
        in LEDGER.read_text()
    )

    checks = [
        ("active source has no class-map include/reference/label",
         all(token not in source for token in FORBIDDEN)),
        ("supplement has no class-map include/reference/label",
         all(token not in supplementary for token in FORBIDDEN)),
        ("LaTeX recorder loaded neither class-map asset",
         all(token not in fls for token in FORBIDDEN[:2])),
        ("compiled paper contains neither deleted map caption",
         "morphology map" not in pdf_text and
         "Available smooth-wall cases placed" not in pdf_text),
        ("compiled PDF is newer than edited source",
         PDF.exists() and PDF.stat().st_mtime >= MAIN.stat().st_mtime),
        (f"operative body-page target (<= 25) met or explicitly suspended during "
         f"closed-row replay (measured body {body_pages(PDF)}, total {pages})",
         body_pages(PDF) <= 25 or suspension_is_live),
        ("the class map in the build derives its badges, none typed",
         live_map_badges()[2]),
        ("every badge it prints is the artifact value",
         all(v.replace("-", "\u2212") in live_map_badges()[1]
             or v in live_map_badges()[1] for v in live_map_badges()[0])),
        ("resolution artifact records deletion",
         summary["status"] == "CLOSED_BY_DELETION" and
         summary["active_manuscript_assets"] == [] and
         summary["expected_checks"] in (7, 9)),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
