# Replaying the analysis

The verification programs read **reduced records**, not raw simulation output. Two ways to run:

## 1. Records only (fast, no large download)

The JSON records in `../records/` already contain the reported values, their intervals and their
provenance. Programs that only re-derive numbers from records run directly:

```bash
for v in ../codes/analysis/ledger_verifiers/verify_*.py; do python3 "$v"; done
```

Checks that require bulk arrays report the missing input rather than silently passing.

## 2. Full replay from the archived deposit

Fetch the deposit (doi:10.5281/zenodo.22069527, open access), unpack it, and stage its
`codes/results/` and `codes/openfoam/` alongside this tree so that paths resolve as they do in
the article:

```
<repo>/codes/results/      <- from the deposit
<repo>/codes/openfoam/     <- from the deposit (cases, wall-model implementation, solver logs)
<repo>/codes/raw_data/     <- third-party reference data, fetched from the sources cited in the article
```

Then re-run the same loop. Each program prints one line per check and exits non-zero on the
first failure, so a reader can see exactly which claim a change breaks.

## Reading the output

Programs print `[PASS]` / `[FAIL]` per check with the quantity, the measured value and the
condition. A `[FAIL]` is informative, not fatal to the study: several checks are deliberately
sensitive to the choice of reference data, and where a reference was withdrawn during the study
both the superseded and the corrected values are retained in the record.

`../SHA256SUMS.txt` lists the checksum of every record; `sha256sum -c ../SHA256SUMS.txt` run from
this directory's parent confirms that the records are the ones the article was built from.
