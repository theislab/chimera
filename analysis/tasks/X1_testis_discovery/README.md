<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X1_testis_discovery

Testis: genes grouped by the stage in which they peak, their Δα and count evidence against matched genes, across cohorts and seeds.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Supplementary Section C.21.12 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X1_testis_discovery.build
    python -m analysis.tasks.X1_testis_discovery.seeds --runs <run ...>
    python -m analysis.tasks.X1_testis_discovery.leads
    python -m analysis.tasks.X1_testis_discovery.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X1/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T2_5_celltype_margin`, `T4_2_gene_baselines`
- `data/analysis/ensembl110/`
- `data/processed/`

**Notes.** Labeled exploratory in the paper (spermatocyte-peak genes).
