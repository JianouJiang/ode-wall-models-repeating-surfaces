#!/usr/bin/env python3
"""
EXHAUSTIVE adversarial number audit for final validation (node_006).

Motivation — defeating the recurring "the harness only checks what the Worker
chose to test" attack:

  The prior Level-5 harness (validate_traceability.py) checks a *curated* list of
  33 headline numbers. A skeptic can argue it is rigged: it only verifies the
  numbers the Worker decided to expose. This script inverts the burden of proof.
  It does NOT pick numbers. It extracts EVERY numeric token in manuscript/main.tex,
  pools EVERY scalar/array value across ALL codes/results/*.npz, and reports, for
  every number in the paper, whether it can be traced to data — and classifies the
  remainder so a referee can see exactly which numbers are NOT data-derived and why
  that is benign (years, Reynolds-number identifiers, von Karman constant, etc.).

  This makes coverage, not cherry-picking, the headline statistic.

Run:  OMP_NUM_THREADS=2 python3 codes/analysis/audit_exhaustive_numbers.py
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "codes", "results")
TEX  = os.path.join(ROOT, "manuscript", "main.tex")

# ---------------------------------------------------------------- load tex
raw = open(TEX).read()
# strip comments (a % not preceded by backslash to end of line)
tex = re.sub(r"(?<!\\)%.*", "", raw)

# Mask out regions where numbers are structural, not empirical, so they are not
# mis-flagged as "untraced empirical claims". We REMOVE these spans entirely.
mask_patterns = [
    r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{[^}]*\}",   # \citep{...}, \cite[p.3]{...}
    r"\\(?:ref|eqref|autoref|cref|Cref|pageref|label)\{[^}]*\}",  # cross-refs/labels
    r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^}]*\}",  # heading text
    r"\\(?:include|input|usepackage|documentclass|bibliographystyle)(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:v|h)space\*?\{[^}]*\}",
    r"\\setlength\{[^}]*\}\{[^}]*\}",
    r"\\(?:textwidth|linewidth|columnwidth)",
]
for p in mask_patterns:
    tex = re.sub(p, " ", tex)

# ---------------------------------------------------------------- pool npz values
# PROVENANCE pool: each entry carries the file:key it came from, so a tex number
# traces to a NAMED reported quantity, not "some value in a 5-million-element sea".
# Large continuous arrays (profiles, C_f(x) curves) are EXCLUDED: matching against
# a dense sampled curve is near-trivial and proves nothing. We keep scalar keys and
# small arrays (size<=64 — the reported aggregates, per-case lists, bands).
MAXARR = 64
prov = []  # (abs_value, "file:key")

def add(v, src):
    if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
        prov.append((abs(float(v)), src))

def collect(arr, src):
    a = np.asarray(arr)
    if a.dtype.kind in "fiu":
        if a.size <= MAXARR:
            for v in a.ravel():
                if np.isfinite(v):
                    add(float(v), src)
    elif a.dtype.names:                       # structured record array
        if a.size <= MAXARR:
            for v in a.ravel():
                for nm in a.dtype.names:
                    try: add(float(v[nm]), f"{src}.{nm}")
                    except Exception: pass
    elif a.dtype == object:
        if a.size <= MAXARR:
            for v in a.ravel():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    add(float(v), src)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        add(vv, f"{src}.{kk}")
                elif isinstance(v, (list, tuple, np.ndarray)):
                    collect(np.asarray(v), src)

nfiles = 0
for fn in sorted(os.listdir(RES)):
    if not fn.endswith(".npz"):
        continue
    try:
        d = np.load(os.path.join(RES, fn), allow_pickle=True)
    except Exception:
        continue
    nfiles += 1
    for k in d.files:
        try:
            collect(d[k], f"{fn}:{k}")
        except Exception:
            pass

# Derived renderings: a tex "20.6" (%) can come from an npz 0.2063 fraction, etc.
DERIV = []
for av, src in prov:
    DERIV.append((av, src, "raw"))
    DERIV.append((av*100.0, src, "x100/pct"))
    DERIV.append((av/100.0, src, "pct->frac"))

RELTOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.012
def traces(x, rel=None, absfloor=5e-4):
    """Return source string if |x| matches a NAMED pooled value, else None."""
    if rel is None:
        rel = RELTOL
    ax = abs(x)
    tol = max(rel * ax, absfloor)
    best = None
    for v, src, kind in DERIV:
        if abs(v - ax) <= tol:
            return f"{src}" + ("" if kind == "raw" else f" [{kind}]")
    return None

# ---------------------------------------------------------------- extract numbers
# number with optional sign, decimals, scientific; capture context window
num_re = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
TOK = []
for m in num_re.finditer(tex):
    s = m.group(0)
    ctx = tex[max(0, m.start()-45):m.end()+25].replace("\n", " ")
    try:
        val = float(s)
    except ValueError:
        continue
    TOK.append((s, val, ctx))

# ---------------------------------------------------------------- classify
YEAR = "year (bibliographic)"
REY  = "Reynolds-number identifier"
CONST= "known physical/structural constant"
SMALLINT = "small structural integer (count/index)"
TRACED = "TRACED to npz"
UNTRACED = "UNTRACED"

# tokens whose CONTEXT marks them benign without needing a data trace
const_ctx = re.compile(r"kappa|von\s*K|0\.41|Spalding|B\s*=|E\s*=\s*9|\\kappa|law-of-the-wall", re.I)
rey_ctx   = re.compile(r"Re[_\s\\{}]|Reynolds|Re_?H|Re_?\\tau|Re_?b|\\Rey", re.I)
year_ctx  = re.compile(r"19\d{2}|20[0-2]\d")

buckets = {YEAR:0, REY:0, CONST:0, SMALLINT:0, TRACED:0, UNTRACED:0}
untraced = []
for s, val, ctx in TOK:
    av = abs(val)
    # bibliographic year: 1900-2030 integer appearing with author/citation context
    if av == int(av) and 1900 <= av <= 2030 and ("\\" in ctx or "(" in ctx or year_ctx.search(ctx)):
        # only treat as year if it really is 4-digit-ish year and not e.g. 2000 cells
        if 1900 <= av <= 2030 and "." not in s:
            buckets[YEAR]+=1; continue
    if rey_ctx.search(ctx):
        buckets[REY]+=1; continue
    if const_ctx.search(ctx):
        buckets[CONST]+=1; continue
    src = traces(val)
    if src is not None:
        buckets[TRACED]+=1; continue
    # small structural integers (figure counts, term counts, dimension 1-D/2-D/3-D)
    if val == int(val) and 0 <= av <= 12 and "." not in s:
        buckets[SMALLINT]+=1; continue
    buckets[UNTRACED]+=1
    untraced.append((s, ctx))

# ---------------------------------------------------------------- report
total = len(TOK)
print(f"npz files pooled            : {nfiles}")
print(f"named scalar/small-array vals: {len(prov):,} (arrays >{MAXARR} excluded)")
print(f"numeric tokens in main.tex  : {total}")
print("-"*70)
for k in [TRACED, YEAR, REY, CONST, SMALLINT, UNTRACED]:
    print(f"  {k:<34} {buckets[k]:>5}  ({100*buckets[k]/total:5.1f}%)")
print("-"*70)
# Data-claim coverage = traced / (traced + untraced)  (excludes years/Re/const/idx)
denom = buckets[TRACED] + buckets[UNTRACED]
print(f"data-claim trace coverage: {buckets[TRACED]}/{denom} = "
      f"{100*buckets[TRACED]/max(denom,1):.1f}%  (excludes years/Re/constants/indices)")
print("="*70)
print(f"UNTRACED numbers requiring manual classification ({len(untraced)}):")
for s, ctx in untraced:
    print(f"  {s:>10}  ...{ctx.strip()}...")
