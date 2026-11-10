<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y4_regions

Screens regions of gene space (by expression, specificity, identity and annotation) for where Δα and the count evidence agree with independent data, tuned on odd and scored on even autosomes.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.Y4_regions.annotations
    python -m analysis.tasks.Y4_regions.build
    python -m analysis.tasks.Y4_regions.screen coverage
    python -m analysis.tasks.Y4_regions.screen discovery
    python -m analysis.tasks.Y4_regions.carry
    python -m analysis.tasks.Y4_regions.replicate
    python -m analysis.tasks.Y4_regions.posthoc
    python -m analysis.tasks.Y4_regions.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y4/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `X1_testis_discovery`, `X4_where_differs`
- `data/analysis/ensembl110/`
- `data/analysis/functional/`
- `data/analysis/x3_annotations/`
- `data/processed/`

**Notes.** Labeled exploratory in the paper (near-identical proteins; level in moderately expressed cell-type-specific liver genes). Definitions in rules.py. replicate.py opens the replication half only after it finds a marker in the authors' working notes, which are not in the repository; the archived result tables are the record.
