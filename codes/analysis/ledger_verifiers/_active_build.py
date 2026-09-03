#!/usr/bin/env python3
"""Shared ACTIVE-build extractor for the prose ledger verifiers.

The manuscript keeps retired material inside ``\\iffalse ... \\fi`` blocks
(provenance, not part of the compiled paper).  Every prose verifier must
therefore reason about the active build only: comments stripped, ``\\input``
files expanded, nested ``\\iffalse`` regions removed.  This module is the one
place that logic lives; the number verifiers (verify_m15.py / verify_m1.py)
carry an equivalent line-based extractor.

Also provides: section slicing, a LaTeX-aware sentence splitter and a math
symbol enumerator.  No verifier semantics live here.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAN = ROOT / "manuscript"
TEX = MAN / "main.tex"
PDF = MAN / "main.pdf"


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def expand_inputs(text: str, base: Path) -> str:
    def replace(match: re.Match) -> str:
        target = base / match.group(1)
        if target.suffix == "":
            target = target.with_suffix(".tex")
        if target.exists():
            return expand_inputs(strip_comments(target.read_text(encoding="utf-8")),
                                 target.parent)
        return match.group(0)
    return re.sub(r"\\(?:input|include)\{([^}]*)\}", replace, text)


def drop_iffalse(text: str) -> str:
    """Remove nested ``\\iffalse`` ... ``\\fi`` regions (token based, so an
    inline ``\\iffalse`` mid-line is handled too)."""
    out: list[str] = []
    depth = 0
    pos = 0
    for match in re.finditer(r"\\(iffalse|fi)\b", text):
        if match.group(1) == "iffalse":
            if depth == 0:
                out.append(text[pos:match.start()])
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                pos = match.end()
    if depth:
        raise RuntimeError("unclosed \\iffalse block in the manuscript source")
    out.append(text[pos:])
    return "".join(out)


def active_source(path: Path = TEX) -> str:
    text = strip_comments(path.read_text(encoding="utf-8"))
    text = expand_inputs(text, path.parent)
    return drop_iffalse(text)


def body(active: str) -> str:
    m = re.search(r"\\begin\{document\}", active)
    return active[m.end():] if m else active


def preamble(active: str) -> str:
    m = re.search(r"\\begin\{document\}", active)
    return active[:m.start()] if m else ""


def newcommands(active: str) -> dict[str, str]:
    """``\\newcommand{\\foo}{...}`` / ``\\renewcommand`` definitions (no args)."""
    out: dict[str, str] = {}
    for m in re.finditer(r"\\(?:re)?newcommand\{?\\([A-Za-z]+)\}?\s*\{", active):
        depth, i = 1, m.end()
        while i < len(active) and depth:
            depth += {"{": 1, "}": -1}.get(active[i], 0)
            i += 1
        out[m.group(1)] = active[m.end():i - 1]
    return out


def pdf_text(path: Path = PDF, layout: bool = False) -> str:
    cmd = ["pdftotext"] + (["-layout"] if layout else []) + [str(path), "-"]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def abstract(active: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", active, re.S)
    return m.group(1).strip() if m else ""


def sections(active: str) -> list[dict]:
    """Top-level ``\\section`` slices of the active body, in source order."""
    text = body(active)
    heads = list(re.finditer(r"\\section\*?\{([^}]*)\}", text))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append({"title": h.group(1), "start": h.start(), "end": end,
                    "text": text[h.end():end]})
    return out


def section_by_title(active: str, pattern: str) -> str:
    for sec in sections(active):
        if re.search(pattern, sec["title"], re.I):
            return sec["text"]
    raise KeyError(f"no section matching /{pattern}/ in the active build")


# --------------------------------------------------------------------------
# sentences
# --------------------------------------------------------------------------
_ABBREV = r"(?:e\.g|i\.e|cf|vs|et al|Fig|Figs|Eq|Eqs|Sec|Ref|Refs|No|approx|viz|ca|resp)"
_MATH_TOKEN = "\uE000"


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Replace $...$, \\(...\\), \\[...\\] and display environments by
    placeholders so that periods inside math never split a sentence."""
    store: list[str] = []

    def keep(m: re.Match) -> str:
        store.append(m.group(0))
        return f"{_MATH_TOKEN}{len(store) - 1}{_MATH_TOKEN}"
    text = re.sub(r"\\begin\{(equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?)\}"
                  r".*?\\end\{\1\}", keep, text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", keep, text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", keep, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(?:\\.|[^$\\])*\$", keep, text)
    text = re.sub(r"\\\((?:\\.|[^\\])*?\\\)", keep, text)
    return text, store


def _restore_math(text: str, store: list[str]) -> str:
    return re.sub(f"{_MATH_TOKEN}(\\d+){_MATH_TOKEN}", lambda m: store[int(m.group(1))], text)


def prose_blocks(text: str, keep_tables: bool = False) -> list[str]:
    """Split a section into paragraph-ish blocks, dropping tables/figures
    (their captions are kept) and list/label scaffolding.  With
    ``keep_tables`` every tabular row becomes its own block (so a symbol
    first used in a column header counts as a use)."""
    if keep_tables:
        text = re.sub(r"\\begin\{tabular\*?\}\{[^}]*\}", "\n\n", text)
        text = re.sub(r"\\end\{tabular\*?\}", "\n\n", text)
        text = re.sub(r"\\\\\s*", "\n\n", text)
    else:
        text = re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{(figure|table)\*?\}|\\end\{(figure|table)\*?\}", "\n\n", text)
    text = re.sub(r"\\(?:centering|small|scriptsize|footnotesize|toprule|midrule|"
                  r"bottomrule|noindent|clearpage|newpage|hline)\b", " ", text)
    text = re.sub(r"\\(?:label|includegraphics|setlength|vspace|hspace)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", text)
    text = re.sub(r"\\begin\{(enumerate|itemize|description|keywords)\}|"
                  r"\\end\{(enumerate|itemize|description|keywords)\}", "\n\n", text)
    text = re.sub(r"\\item\b", "\n\n", text)
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def sentences(text: str, keep_tables: bool = False) -> list[str]:
    """LaTeX-aware sentence split.  Math, citations and ``\\S\\,\\ref`` never
    split; abbreviations are protected; a sentence ends at [.!?] followed by
    whitespace and an upper-case / math / macro start, or at block end."""
    out: list[str] = []
    for block in prose_blocks(text, keep_tables=keep_tables):
        protected, store = _protect_math(block)
        protected = re.sub(r"\s+", " ", protected)
        protected = re.sub(rf"\b{_ABBREV}\.", lambda m: m.group(0).replace(".", "\uE001"), protected)
        protected = protected.replace("\\S\\,", "\\S\uE002")
        protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\uE001" + m.group(2), protected)
        parts = re.split(r"(?<=[.!?])(?:\)|'')?\s+(?=[A-Z\\(`\"\uE000]|\(\w)", protected)
        for part in parts:
            part = part.replace("\uE001", ".").replace("\uE002", "\\,")
            part = _restore_math(part, store).strip()
            if part:
                out.append(part)
    return out


def display_math_blocks(text: str) -> list[str]:
    return re.findall(r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}.*?"
                      r"\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}", text, re.S)


def math_spans(text: str) -> list[str]:
    """All math fragments (inline and display) in source order."""
    _, store = _protect_math(text)
    spans = []
    for s in store:
        s = re.sub(r"^\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}$", "", s.strip())
        s = re.sub(r"^\$\$?|\$\$?$", "", s.strip())
        s = re.sub(r"^\\\(|\\\)$|^\\\[|\\\]$", "", s.strip())
        spans.append(s)
    return spans


# --------------------------------------------------------------------------
# symbols
# --------------------------------------------------------------------------
# macros that are structure/operators, never "symbols that need defining"
_OPERATOR_MACROS = {
    "frac", "tfrac", "dfrac", "sqrt", "left", "right", "big", "Big", "bigl", "bigr",
    "times", "cdot", "cdots", "ldots", "dots", "int", "sum", "prod", "lim", "partial",
    "nabla", "infty", "pm", "mp", "le", "leq", "ge", "geq", "ne", "neq", "approx", "sim",
    "simeq", "equiv", "propto", "in", "notin", "subset", "cup", "cap", "to", "rightarrow",
    "leftarrow", "Rightarrow", "mapsto", "ll", "gg", "langle", "rangle", "lVert", "rVert",
    "lvert", "rvert", "vert", "Vert", "lceil", "rceil", "lfloor", "rfloor", "quad", "qquad",
    ",", ";", ":", "!", " ", "mathrm", "mathit", "mathbf", "mathcal", "boldsymbol", "bm",
    "text", "textrm", "textit", "mbox", "operatorname", "max", "min", "exp", "log", "ln",
    "sin", "cos", "tan", "sign", "mathop", "limits", "nolimits", "displaystyle",
    "textstyle", "scriptstyle", "hat", "bar", "tilde", "overline", "underline", "vec",
    "dot", "ddot", "prime", "dd", "rm", "it", "bf", "cal", "triangle", "setminus", "mid",
    "colon", "%", "&", "\\", "{", "}", "_", "^", "label", "tag", "nonumber", "notag",
    "ref", "eqref", "cite", "citep", "citet", "emph", "ensuremath", "hphantom",
    "vphantom", "phantom", "space", "star", "circ", "bullet", "ast", "cdotp",
}
# operator-like words inside \mathrm{...} that are not symbols
_OPERATOR_WORDS = {"sign", "max", "min", "exp", "log", "ln", "sin", "cos", "tan", "e",
                   "d", "i", "const", "and", "or", "for", "if", "else", "with", "sgn",
                   "median", "RMS", "rms", "mean", "diag", "tr", "det"}
# commonly understood, not expected to be defined
CONVENTIONAL = {"x", "y", "z", "t", "i", "j", "k", "n", "N", "e", "Re", "\\Rey", "\\pi",
                "\\infty", "p"}  # p only as a p-value with '=' cue; handled by caller


def math_symbols(math: str) -> list[str]:
    """Enumerate symbol keys in one math fragment, in order of appearance.

    A key is a macro (``\\eta``, ``\\tauw``) or a single Latin/Greek letter,
    joined with its immediate subscript (``y_m``, ``R_{ij}``, ``\\tau_w``) and a
    decorating accent (``\\tilde\\varepsilon``, ``\\bar\\phi``, ``U''``).
    Numbers, operators, delimiters and \\mathrm operator words are skipped.
    """
    s = math
    s = re.sub(r"\\(?:label|tag)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:text|mbox|textrm|textit)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigl|bigr)\b", " ", s)
    out: list[str] = []
    group = r"(?:\{[^{}]*\}|\\[A-Za-z]+(?:\{[^{}]*\})?|[A-Za-z0-9])"
    token = re.compile(
        r"(?P<accent>\\(?:tilde|bar|hat|overline|vec|dot|ddot|widetilde|widehat)\s*)?"
        r"(?P<base>\\[A-Za-z]+|\\mathrm\{[^}]*\}|\\mathcal\{[^}]*\}|\\mathit\{[^}]*\}|"
        r"\{\\(?:cal|rm|bf|it)\s+[A-Za-z]+\}|[A-Za-z])"
        r"(?P<prime>'{1,3})?"
        rf"(?:\^{group})?"
        rf"(?P<sub>_{group})?"
        rf"(?:\^{group})?"
    )
    for m in token.finditer(s):
        base = m.group("base")
        accent = (m.group("accent") or "").strip()
        if base.startswith("\\"):
            name = base[1:]
            inner = re.match(r"(mathrm|mathcal|mathit)\{([^}]*)\}", name)
            if inner:
                word = inner.group(2).strip()
                if not word or word in _OPERATOR_WORDS or not re.match(r"[A-Za-z]", word):
                    continue
                key = word if inner.group(1) == "mathrm" else f"\\{inner.group(1)}{{{word}}}"
            elif name in _OPERATOR_MACROS:
                continue
            else:
                key = base
        elif base.startswith("{\\"):
            key = base
        else:
            key = base
        if accent:
            key = f"{accent}{key}"
        if m.group("prime"):
            key += m.group("prime")
        sub = m.group("sub")
        if sub:
            key += sub
        out.append(key)
    return out


def symbol_family(key: str) -> str:
    """``y_m`` -> ``y``; ``\\tilde\\varepsilon`` -> ``\\varepsilon``; ``U''`` -> ``U``."""
    fam = re.sub(r"^\\(?:tilde|bar|hat|overline|vec|dot|ddot|widetilde|widehat)", "", key)
    fam = re.sub(r"'+", "", fam)
    return re.sub(r"_.*$", "", fam)


DEFINITION_CUES = re.compile(
    r"\b(denotes?|denoted|defined?|definition|is the|are the|be the|where|let|called|"
    r"namely|stands? for|represents?|introduce|writes?|written|so that|we write|"
    r"is|are|gives?|given by)\b", re.I)
# "the wall-following coordinate $\eta=...$": the noun phrase defines the HEAD
# symbol of the math span that immediately follows it (its LHS, else its first
# symbol), nothing else in the sentence.
APPOSITION = re.compile(
    r"\b(?:the|a|an) (?:[\w-]+ ){0,3}(?:coordinate|height|velocity|length|stress|error|"
    r"parameter|ratio|fraction|number|scale|operator|average|fluctuation|deviation|"
    r"thickness|pitch|amplitude|set|function|flux|force|traction|residual|norm|"
    r"score|interval|envelope|count|index|threshold|tolerance)\s*"
    r"(?P<math>\$(?:\\.|[^$\\])*\$)", re.I)


def apposition_heads(sentence: str) -> set[str]:
    heads: set[str] = set()
    for m in APPOSITION.finditer(sentence):
        span = m.group("math").strip("$")
        lhs = re.match(r"\s*(.*?)\s*(?:=|\\equiv)", span)
        keys = math_symbols(lhs.group(1) if lhs else span)
        if keys:
            heads.add(keys[0] if not lhs else keys[0])
            if lhs:
                heads.update(keys)
    return heads


def has_definition_cue(sentence: str) -> bool:
    return (bool(DEFINITION_CUES.search(sentence)) or "=" in sentence
            or "\\equiv" in sentence or bool(re.search(r"\$\s*:", sentence)))


def lhs_defined(sentence: str, key: str) -> bool:
    """True if ``key`` sits immediately left of '=' or '\\equiv' in some math
    span of the sentence (``\\eta=y-h(x)`` defines eta, not y or h)."""
    for span in math_spans(sentence):
        for part in re.split(r",|\\qquad|\\quad|;", span):
            m = re.match(r"\s*(.*?)\s*(?:=|\\equiv)", part)
            if m and key in math_symbols(m.group(1)):
                return True
    return False


if __name__ == "__main__":
    act = active_source()
    print(f"active build: {len(act.splitlines())} lines; sections:")
    for sec in sections(act):
        print(f"  {sec['title']}")
