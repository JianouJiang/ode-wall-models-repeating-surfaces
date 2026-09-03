#!/usr/bin/env python3
"""Inventory every numeric literal in the ACTIVE manuscript source.

Compression is only safe if it removes words and not evidence.  This tool takes
the numeric fingerprint of the active build (archive blocks excluded) so that a
before/after comparison shows exactly which numbers a rewrite dropped, kept or
introduced.  A dropped number is not automatically an error --- it may have
moved, with its reasoning, to the thesis chapter --- but it must be a decision,
not an accident.

Usage:
    number_inventory.py dump  <main.tex> <out.json>
    number_inventory.py diff  <before.json> <after.json>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def active_source(tex: Path) -> str:
    """Return the source with \\iffalse ... \\fi archive blocks removed."""
    out, skip = [], False
    for line in tex.read_text(errors="replace").split("\n"):
        if line.startswith("\\iffalse"):
            skip = True
            continue
        if line.startswith("\\fi"):
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out)


# A numeric literal: optional sign, digits, optional decimals, allowing the
# LaTeX thin-space thousands separator and scientific notation.
NUM = re.compile(r"(?<![A-Za-z0-9])"
                 r"-?\d[\d,]*(?:\\,\d{3})*(?:\.\d+)?"
                 r"(?:\s*\\times\s*10\^\{?-?\d+\}?)?")


def normalise(tok: str) -> str:
    return re.sub(r"[\s,]|\\,", "", tok)


def inventory(tex: Path) -> Counter:
    src = active_source(tex)
    # Strip LaTeX comments, but NOT the escaped percent sign.  Treating "\%" as
    # a comment silently deleted the rest of every line carrying a percentage,
    # which is most of the lines that carry results, and made the tool report
    # numbers as dropped while they were sitting in the paper.
    src = re.sub(r"(?<!\\)%.*", "", src)
    # Citation years, label/ref arguments and length units are typography, not
    # evidence; excluding them keeps the fingerprint on reported quantities.
    src = re.sub(r"\\cite[a-z]*\*?(\[[^\]]*\])*\{[^}]*\}", " ", src)
    src = re.sub(r"\\(label|ref|eqref|includegraphics|setlength|tabcolsep|"
                 r"resizebox|columnwidth|textwidth|vspace|hspace)"
                 r"(\[[^\]]*\])*\{[^}]*\}", " ", src)
    src = re.sub(r"\d+(\.\d+)?\s*(pt|em|ex|cm|mm|in)\b", " ", src)
    return Counter(normalise(m.group(0)) for m in NUM.finditer(src))


def main() -> int:
    mode = sys.argv[1]
    if mode == "dump":
        inv = inventory(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(dict(inv), indent=0, sort_keys=True))
        print(f"{len(inv)} distinct numeric literals, {sum(inv.values())} occurrences")
        return 0
    if mode == "diff":
        a = Counter(json.loads(Path(sys.argv[2]).read_text()))
        b = Counter(json.loads(Path(sys.argv[3]).read_text()))
        dropped = {k: a[k] for k in a if k not in b}
        added = {k: b[k] for k in b if k not in a}
        fewer = {k: (a[k], b[k]) for k in a if k in b and b[k] < a[k]}
        print(f"before: {len(a)} distinct / {sum(a.values())} occurrences")
        print(f"after : {len(b)} distinct / {sum(b.values())} occurrences")
        print(f"\nDROPPED ENTIRELY ({len(dropped)}):")
        for k in sorted(dropped, key=lambda s: (-len(s), s)):
            print(f"  {k}  (was x{dropped[k]})")
        print(f"\nNEWLY INTRODUCED ({len(added)}):")
        for k in sorted(added, key=lambda s: (-len(s), s)):
            print(f"  {k}  (x{added[k]})")
        print(f"\nFEWER OCCURRENCES ({len(fewer)}):")
        for k in sorted(fewer):
            print(f"  {k}  {fewer[k][0]} -> {fewer[k][1]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
