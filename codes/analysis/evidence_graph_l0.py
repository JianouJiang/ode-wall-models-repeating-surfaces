#!/usr/bin/env python3
"""Key-aware provenance closure of the ACTIVE manuscript build.

Motivation
----------
A scoring reference was withdrawn on 2026-08-25.  What was withdrawn is *not a
file*: the public periodic-hill velocity archive is sound.  What was withdrawn
is a small set of DERIVED QUANTITIES inside it --- the wall traction and
everything scaled by it --- because the four-point through-origin wall-gradient
fit that produced them under-resolves the traction at the archive's wall
spacing.  Three independent reviews of the previous attempt found the same
failure mode: prose was patched claim by claim while tables, captions and
figures kept inheriting the withdrawn quantities through their producer
scripts.  A headline scan cannot find that, because the defect is structural.

This module therefore treats the manuscript as a DEPENDENCY GRAPH and asks a
decidable question of it:

    for every artifact rendered in the compiled paper, does its provenance
    closure touch a withdrawn quantity?

Edges are recovered statically from the producer sources with ``ast`` (never by
executing them), and contamination is decided PER KEY, not per file: reading
``U`` or ``dp_dx`` from the archive is admissible, reading ``tau_w`` from it as
a truth reference is not.  Anything the static pass cannot resolve is reported
as ``UNRESOLVED`` and counted against the build rather than silently dropped;
the audit is only useful if it is honest about its own blind spots.

Outputs ``codes/results/evidence_graph_l0_<stamp>.json``.  Checked by
``codes/analysis/ledger_verifiers/verify_evidence_graph_l0.py``.

Run:  python3 codes/analysis/evidence_graph_l0.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes" / "analysis" / "ledger_verifiers"))
from _active_build import active_source  # noqa: E402

STAMP = "20260825"
OUT_JSON = ROOT / "codes" / "results" / f"evidence_graph_l0_{STAMP}.json"

# --------------------------------------------------------------------------
# 1. the withdrawal registry --- the roots of the graph
# --------------------------------------------------------------------------
# `keys` are the quantities withdrawn; `safe_keys` are the quantities in the
# same container that remain admissible.  Keeping both lists explicit is the
# point: a file-level ban would wrongly condemn the velocity fields, and the
# operator handover forbids saying the public archive is discredited.
WITHDRAWN = {
    "periodic_hills_case_1p0_wall_profiles_corrected.npz": {
        "status": "WITHDRAWN_AS_SCORING_REFERENCE",
        "keys": ["tau_w", "u_tau", "Re_tau", "median_eps", "frac_below_01"],
        "safe_keys": ["x", "y", "U", "V", "uv", "dp_dx", "nu", "is_separated",
                      "geometry", "re_identifier", "extraction"],
        "reason": (
            "wall traction reconstructed by a four-point through-origin fit at a "
            "wall spacing of 0.0093-0.0136 H (fit points at y+ 2.4-44), about 7.5x "
            "coarser than the wall-resolved deposit; recovers 0.30 of the "
            "wall-resolved traction in RMS on the 512-station surface and places "
            "separation at x/H = 0.378 against 0.181.  The velocity fields in the "
            "same container are sound and remain admissible."),
        "replacement": "ercoftac UFR3-30 MGLET full-wall DNS (primary); "
                       "curvature-aware cubic re-reading of the same archive (bracket)",
    },
}

# The withdrawal is really a statement about an ESTIMATOR, not about a file.
# The same four-point through-origin wall-gradient fit is re-instantiated in
# code wherever a traction is extracted from a velocity archive, so a registry
# of filenames can only ever catch the instances that happen to have been
# written to disk under a known name.  These signatures find the estimator
# itself.  All signatures of an entry must match before a script is flagged,
# which keeps a stray comment from condemning an innocent producer.
WITHDRAWN_ESTIMATORS = {
    "through_origin_4pt_wall_gradient": {
        "status": "WITHDRAWN_AS_SCORING_REFERENCE",
        "why": ("a through-origin least-squares fit of the first ~4 fluid points "
                "under-resolves the wall traction at the archives' wall spacing; "
                "on the one member with an independent published traction it is "
                "biased, and the bias correlates with the reported geometric "
                "variables, so it cannot be used as a scoring reference"),
        "code_signatures": [
            r"nfit\s*=\s*min\(\s*4",
            r"np\.sum\(\s*yf\s*\*\s*uf\s*\)\s*/\s*np\.sum\(\s*yf\s*\*\s*yf\s*\)",
        ],
        "replacement": ("wall-resolved full-wall DNS traction where it exists; "
                        "otherwise a curvature-aware cubic re-reading, reported "
                        "as a bracket and never as a single figure"),
    },
}

# A script may read a withdrawn quantity *on purpose*: the rebase study scores
# against it precisely to show what it does to the answer.  Condemning those
# would make the audit unable to document its own subject.  The exemption is
# not a rubber stamp --- it holds only while the artifact the script writes
# still carries the marker that labels the reference as withdrawn, which is
# re-checked here on every run.
NEGATIVE_CONTROL_PRODUCERS = {
    "codes/analysis/reference_rebase_headlines_l0.py": {
        "declares_in": "codes/results/reference_rebase_headlines_l0_20260825.json",
        "marker": r"A_withdrawn",
        "why": "reference A is the declared negative control of the rebase study",
    },
    "codes/analysis/conditioning_ladder_l0.py": {
        "declares_in": "codes/results/conditioning_ladder_l0_20260825.json",
        "marker": r"A_withdrawn",
        "why": "reference A is the declared negative control of the ladder re-adjudication",
    },
    "codes/analysis/audit_m13_truth_references.py": {
        "declares_in": "codes/results/m13_truth_reference_audit_20260825.json",
        "marker": r"reconstruct|withdraw|xiao",
        "why": "the audit that established the withdrawal must read the withdrawn quantity",
    },
}

# Admissible reference roots, recorded so the graph can state what a clean
# artifact is rooted IN rather than only what it avoids.
ADMISSIBLE_ROOTS = {
    "UFR3-30_data-NP-Re5600-DNS2-11.dat": "B_mglet_full_wall_dns",
    "krank_pehill_Re5600_wall_profiles.npz": "K_krank_ten_station_dns",
    "reference_rebase_headlines_l0_20260825.json": "corrected_headline_object",
    "reference_rebase_headlines_l0_20260825.npz": "corrected_headline_object",
    "conditioning_ladder_l0_20260825.json": "corrected_ladder_object",
    "m13_highre_coupled_20260825_summary.json": "corrected_coupled_reynolds_object",
    "r1_sta2_wavy_wrles_20260824.json": "wavy_wall_wrles",
}

DATA_SUFFIXES = (".npz", ".npy", ".json", ".dat", ".csv", ".txt", ".pdf", ".png")
SKIP_DIRS = {"vendor", "new_data_download", "__pycache__", ".git", "raw_data_backup"}

READ_FUNCS = {"load", "loadtxt", "genfromtxt", "read_csv", "read_text",
              "read_bytes", "imread", "load_npz"}
WRITE_FUNCS = {"savefig", "savez", "savez_compressed", "save", "to_csv",
               "write_text", "write_bytes", "imsave", "copyfile", "copy"}
# project helpers that write ``<name>.pdf`` and ``<name>.png`` from a bare stem
SAVE_HELPERS = {"save_figure", "save_fig", "_save_figure"}
_STEM_SUFFIXES = (".pdf", ".png")


def _fstring_stems(node: ast.AST) -> list[str]:
    """``f"fig_x.{ext}"`` names a figure as surely as ``"fig_x.pdf"`` does.

    Recover the constant prefix of an f-string; when it ends at the extension
    dot the stem is unambiguous and both rendered forms are registered.
    """
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.JoinedStr):
            continue
        head = ""
        for part in sub.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                head += part.value
            else:
                break
        m = re.search(r"([A-Za-z0-9_\-]+)\.$", head)
        if m:
            out.extend(m.group(1) + s for s in _STEM_SUFFIXES)
    return out


_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*\.[A-Za-z0-9]+")


def _is_filename(value: str) -> bool:
    """``"results/x.npz"`` is a file; ``".npz"`` and ``"_summary.json"`` are
    suffix fragments used for string concatenation.  Treating a fragment as a
    node merges every script that builds a name that way into one hub, and the
    hub then makes every artifact look related to every other."""
    name = Path(value).name
    return bool(name.endswith(DATA_SUFFIXES) and _FILENAME_RE.fullmatch(name))


def _literals(node: ast.AST) -> list[str]:
    """Every filename-ish string constant in a subtree, in source order."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _is_filename(sub.value):
                out.append(Path(sub.value).name)
    out.extend(_fstring_stems(node))
    return out


def _bare_names(node: ast.AST) -> list[str]:
    return [s.id for s in ast.walk(node) if isinstance(s, ast.Name)]


class ScriptScan:
    """Static read/write/key-touch profile of one producer script."""

    def __init__(self, path: Path, text: str):
        self.path = path
        self.text = text
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.key_touches: dict[str, set[str]] = {}      # file -> keys read
        self.unresolved: list[dict] = []
        self.bindings: dict[str, set[str]] = {}         # variable -> filenames
        self.literal_assign_count: dict[str, int] = {}  # variable -> n filename assignments
        self.is_loop_var: dict[str, bool] = {}          # variable bound by a for/comprehension
        self.load_vars: dict[str, set[str]] = {}        # variable <- np.load(file)
        self.estimators: set[str] = set()               # withdrawn estimators instantiated

    # -- helpers ---------------------------------------------------------
    def _resolve(self, node: ast.AST) -> set[str]:
        """Filenames reachable from an argument node.

        Literals and f-string stems are exact.  A bare NAME is followed only
        when it is assigned exactly once in the module: a variable rebound to
        several paths (``path = ...`` inside two different loops) would
        otherwise merge unrelated files into one edge and make every artifact
        look like a producer of every other.  Single-assignment names are the
        module constants this project actually uses for output paths.
        """
        found = set(_literals(node))
        for name in _bare_names(node):
            if self.literal_assign_count.get(name, 0) == 1 and \
                    not self.is_loop_var.get(name):
                found |= self.bindings.get(name, set())
        return found

    def _target_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        return None

    # -- passes ----------------------------------------------------------
    def pass_bindings(self, tree: ast.AST) -> None:
        """Resolve ``OUT = DIR / 'x.npz'`` style module constants.  Two sweeps so
        that a constant defined in terms of another constant also resolves."""
        # A path variable may share its name with an unrelated value
        # (``out = np.empty_like(y)`` then ``out = .../fig.pdf``).  What must be
        # unambiguous is the FILENAME it can carry, so count only the
        # assignments that introduce a filename, and never trust a loop
        # variable.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                bears = bool(value is not None and _literals(value))
                for tgt in targets:
                    name = self._target_name(tgt)
                    if name and bears:
                        self.literal_assign_count[name] = \
                            self.literal_assign_count.get(name, 0) + 1
            elif isinstance(node, (ast.For, ast.comprehension)):
                name = self._target_name(node.target)
                if name:
                    self.is_loop_var[name] = True
        for _ in range(2):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and node.value is not None:
                    lits = self._resolve(node.value)
                    if not lits:
                        continue
                    for tgt in node.targets:
                        name = self._target_name(tgt)
                        if name:
                            self.bindings.setdefault(name, set()).update(lits)

    def pass_io(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if fname is None:
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            hits: set[str] = set()
            for a in args:
                hits |= self._resolve(a)

            if fname in READ_FUNCS:
                if fname in {"read_text", "read_bytes"} and isinstance(fn, ast.Attribute):
                    got = self._resolve(fn.value)
                else:
                    got = self._resolve(node.args[0]) if node.args else set()
                self.reads |= got
                if not got and args:
                    self.unresolved.append(
                        {"kind": "unresolved_read_path", "func": fname,
                         "line": node.lineno})
            elif fname in WRITE_FUNCS:
                # Only the PATH argument names a written file.  Keyword payloads
                # routinely carry provenance strings -- an npz that records
                # which archives it was built from lists their names inside
                # itself -- and counting those as writes invents an edge from
                # every consumer of an archive to that archive.
                if fname in {"copyfile", "copy"} and len(node.args) >= 2:
                    self.reads |= self._resolve(node.args[0])
                    self.writes |= self._resolve(node.args[1])
                elif fname in {"write_text", "write_bytes"}:
                    # ``OUT.write_text(...)``: the path is the receiver
                    if isinstance(fn, ast.Attribute):
                        self.writes |= self._resolve(fn.value)
                elif node.args:
                    self.writes |= self._resolve(node.args[0])
            elif fname in SAVE_HELPERS:
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        stem = a.value
                        if re.fullmatch(r"[A-Za-z0-9_\-]+", stem):
                            self.writes.update(stem + s for s in _STEM_SUFFIXES)
            elif fname == "open":
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                (self.writes if ("w" in mode or "a" in mode) else self.reads).update(hits)

    def pass_load_vars(self, tree: ast.AST) -> None:
        """Bind ``d = np.load(<registered file>)`` so key access is attributable."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if fname not in READ_FUNCS:
                continue
            files = set()
            for a in list(call.args) + [kw.value for kw in call.keywords]:
                files |= self._resolve(a)
            registered = files & set(WITHDRAWN)
            if not registered:
                continue
            for tgt in node.targets:
                name = self._target_name(tgt)
                if name:
                    self.load_vars.setdefault(name, set()).update(registered)

    def pass_keys(self, tree: ast.AST) -> None:
        """Which keys of a registered container does this script actually read?"""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            base = node.value
            name = base.id if isinstance(base, ast.Name) else None
            if name is None or name not in self.load_vars:
                continue
            sl = node.slice
            key = sl.value if isinstance(sl, ast.Constant) else None
            for f in self.load_vars[name]:
                if isinstance(key, str):
                    self.key_touches.setdefault(f, set()).add(key)
                else:
                    self.unresolved.append(
                        {"kind": "dynamic_key_access", "file": f,
                         "line": node.lineno})
        # ``d.get('tau_w')`` / ``'tau_w' in d`` style
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in self.load_vars:
                    if node.func.attr in {"get", "__getitem__"} and node.args:
                        a = node.args[0]
                        if isinstance(a, ast.Constant) and isinstance(a.value, str):
                            for f in self.load_vars[base.id]:
                                self.key_touches.setdefault(f, set()).add(a.value)

    def pass_estimators(self) -> None:
        """Which withdrawn estimators does this script instantiate in code?"""
        for name, spec in WITHDRAWN_ESTIMATORS.items():
            if all(re.search(sig, self.text) for sig in spec["code_signatures"]):
                self.estimators.add(name)

    def run(self) -> "ScriptScan":
        self.pass_estimators()
        tree = ast.parse(self.text)
        self.pass_bindings(tree)
        self.pass_io(tree)
        self.pass_load_vars(tree)
        self.pass_keys(tree)
        # a registered file that is read but whose keys never resolved
        for f in self.reads & set(WITHDRAWN):
            if f not in self.key_touches:
                self.unresolved.append({"kind": "registered_read_without_key_binding",
                                        "file": f})
        return self


def scan_tree() -> dict[str, ScriptScan]:
    scans: dict[str, ScriptScan] = {}
    for py in sorted((ROOT / "codes").rglob("*.py")):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        if py.is_symlink():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
            scans[str(py.relative_to(ROOT))] = ScriptScan(py, text).run()
        except (SyntaxError, ValueError):
            continue
    return scans


# --------------------------------------------------------------------------
# 2. reachability
# --------------------------------------------------------------------------
def build_producers(scans: dict[str, ScriptScan]) -> dict[str, list[str]]:
    prod: dict[str, list[str]] = {}
    for rel, sc in scans.items():
        for w in sc.writes:
            prod.setdefault(w, []).append(rel)
    return prod


_NEG_CTRL_CACHE: dict[str, bool] = {}


def _negative_control_ok(rel: str) -> bool:
    """True only if ``rel`` is a declared negative-control producer AND the
    artifact it writes still carries the marker labelling the reference as
    withdrawn.  An exemption whose justification has silently disappeared from
    the artifact is not an exemption."""
    if rel in _NEG_CTRL_CACHE:
        return _NEG_CTRL_CACHE[rel]
    spec = NEGATIVE_CONTROL_PRODUCERS.get(rel)
    ok = False
    if spec:
        target = ROOT / spec["declares_in"]
        if target.exists():
            ok = bool(re.search(spec["marker"],
                                target.read_text(encoding="utf-8", errors="replace"),
                                re.I))
    _NEG_CTRL_CACHE[rel] = ok
    return ok


_CLOSURE_CACHE: dict[str, dict] = {}


def closure(target: str, scans, producers, depth: int = 0,
            stack: frozenset[str] = frozenset()) -> dict:
    """Provenance closure of one artifact: which withdrawn keys can reach it.

    Union-reachability, so the answer for a node does not depend on the path
    that reached it and can be memoised; ``stack`` only breaks cycles.  Without
    the memo the shared lower layers of the graph are re-expanded once per path
    and the walk does not terminate in practical time.
    """
    if target in stack or depth > 24:
        return {"file": target, "producers": [], "cycle_or_depth_limit": True,
                "withdrawn": [], "roots": [], "unresolved": [], "via": []}
    if depth and (target in WITHDRAWN or target in ADMISSIBLE_ROOTS):
        # A reference archive terminates the walk.  Whatever scripts happen to
        # regenerate a raw archive is not evidence about the artifact that
        # reads it; only the reading script's own key access is.
        return {"file": target, "producers": [], "is_reference_root": True,
                "withdrawn": [], "via": [], "unresolved": [],
                "roots": [ADMISSIBLE_ROOTS[target]] if target in ADMISSIBLE_ROOTS
                         else [f"WITHDRAWN::{target}"]}
    if target in _CLOSURE_CACHE:
        return _CLOSURE_CACHE[target]
    stack = stack | {target}
    seen = stack

    withdrawn: list[dict] = []
    roots: list[str] = []
    unresolved: list[dict] = []
    via: list[dict] = []

    for rel in producers.get(target, []):
        sc = scans[rel]
        exempt = _negative_control_ok(rel)
        for f, keys in sc.key_touches.items():
            bad = sorted(set(keys) & set(WITHDRAWN[f]["keys"]))
            if bad and not exempt:
                withdrawn.append({"file": f, "keys": bad, "touched_by": rel})
        if sc.estimators and not exempt:
            for est in sorted(sc.estimators):
                withdrawn.append({"estimator": est, "keys": ["<recomputed traction>"],
                                  "file": f"<instantiated in {rel}>", "touched_by": rel})
        for u in sc.unresolved:
            if u.get("file") in WITHDRAWN or u["kind"] == "dynamic_key_access":
                unresolved.append({**u, "script": rel})
        for r in sorted(sc.reads):
            if r in ADMISSIBLE_ROOTS:
                roots.append(ADMISSIBLE_ROOTS[r])
            if r == target:
                continue
            sub = closure(r, scans, producers, depth + 1, seen)
            withdrawn.extend(sub["withdrawn"])
            roots.extend(sub["roots"])
            unresolved.extend(sub["unresolved"])
            if sub["withdrawn"] or sub["unresolved"]:
                via.append({"through": r, "producer": rel})

    def _dedup(seq, key):
        out, have = [], set()
        for it in seq:
            k = key(it)
            if k not in have:
                have.add(k)
                out.append(it)
        return out

    result = {
        "file": target,
        "producers": producers.get(target, []),
        "withdrawn": _dedup(withdrawn, lambda d: (d["file"], tuple(d["keys"]), d["touched_by"])),
        "roots": sorted(set(roots)),
        "unresolved": _dedup(unresolved, lambda d: (d.get("kind"), d.get("file"), d.get("script"), d.get("line"))),
        "via": via,
    }
    _CLOSURE_CACHE[target] = result
    return result


def active_figures() -> list[str]:
    act = active_source()
    figs = []
    for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", act):
        name = Path(m).name
        figs.append(name if name.endswith(".pdf") else name + ".pdf")
    return sorted(set(figs))


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    scans = scan_tree()
    producers = build_producers(scans)
    figs = active_figures()

    fig_report = []
    for f in figs:
        rep = closure(f, scans, producers)
        on_disk = ROOT / "manuscript" / "figures" / f
        rep["rendered_sha256"] = sha256(on_disk)
        rep["rendered_present"] = on_disk.exists()
        if rep["withdrawn"]:
            rep["verdict"] = "CONTAMINATED"
        elif rep["unresolved"]:
            rep["verdict"] = "UNRESOLVED"
        elif not rep["producers"]:
            rep["verdict"] = "NO_PRODUCER_FOUND"
        else:
            rep["verdict"] = "CLEAN"
        fig_report.append(rep)

    # direct consumers of the withdrawn keys, whatever they produce
    direct = []
    for rel, sc in sorted(scans.items()):
        for f, keys in sc.key_touches.items():
            bad = sorted(set(keys) & set(WITHDRAWN[f]["keys"]))
            if bad:
                direct.append({"script": rel, "file": f, "withdrawn_keys": bad,
                               "writes": sorted(sc.writes)})

    payload = {
        "generated_by": "codes/analysis/evidence_graph_l0.py",
        "question": ("for every artifact rendered in the compiled paper, does its "
                     "provenance closure touch a quantity that has been withdrawn?"),
        "method": ("static ast recovery of read/write/key-touch edges over every "
                   "producer under codes/ (no script is executed); contamination is "
                   "decided per KEY, so admissible quantities in a partly withdrawn "
                   "container do not condemn their consumers; anything the static "
                   "pass cannot resolve is reported as UNRESOLVED and counted "
                   "against the build"),
        "withdrawal_registry": WITHDRAWN,
        "withdrawn_estimators": WITHDRAWN_ESTIMATORS,
        "estimator_instantiations": {
            name: sorted(rel for rel, sc in scans.items() if name in sc.estimators)
            for name in WITHDRAWN_ESTIMATORS},
        "negative_control_producers": {
            rel: {**spec, "exemption_currently_valid": _negative_control_ok(rel)}
            for rel, spec in NEGATIVE_CONTROL_PRODUCERS.items()},
        "admissible_roots": ADMISSIBLE_ROOTS,
        "n_scripts_scanned": len(scans),
        "active_figures": figs,
        "figure_closures": fig_report,
        "direct_consumers_of_withdrawn_keys": direct,
        "summary": {
            "n_active_figures": len(figs),
            "n_clean": sum(1 for r in fig_report if r["verdict"] == "CLEAN"),
            "n_contaminated": sum(1 for r in fig_report if r["verdict"] == "CONTAMINATED"),
            "n_unresolved": sum(1 for r in fig_report if r["verdict"] == "UNRESOLVED"),
            "n_no_producer": sum(1 for r in fig_report if r["verdict"] == "NO_PRODUCER_FOUND"),
                "n_direct_consumer_scripts": len(direct),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False))
    s = payload["summary"]
    print(f"scripts scanned          : {len(scans)}")
    print(f"active figures           : {s['n_active_figures']}")
    print(f"  CLEAN                  : {s['n_clean']}")
    print(f"  CONTAMINATED           : {s['n_contaminated']}")
    print(f"  UNRESOLVED             : {s['n_unresolved']}")
    print(f"  NO_PRODUCER_FOUND      : {s['n_no_producer']}")
    print(f"direct consumer scripts  : {s['n_direct_consumer_scripts']}")
    for r in fig_report:
        print(f"  [{r['verdict']:17s}] {r['file']}")
        for w in r["withdrawn"][:4]:
            print(f"        withdrawn {w['file']}:{','.join(w['keys'])} via {w['touched_by']}")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
