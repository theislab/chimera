<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_5_celltype_margin

Splits the count evidence S_g of every gene into contributions of species and cell types by forward passes through the archived models.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Figure 6c (`fig:validation`); Supplementary Figure S12a (`fig:supp-dalpha-sources`); Supplementary Figure S22 (`fig:supp-celltype-closure`); Supplementary Figure S23 (`fig:supp-celltype-seeds`); Supplementary Section C.21.5 (`sssec:rev-decomposition`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_5_celltype_margin.run [--runs all] [--threads 4] [--n-boot 200]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_5/`: bundle `analysis_results`

**Reads** (as named in the code)

- `training/runs/`

**Notes.** margin.py holds the decomposition; it is reused unchanged by the cohort analyses (T11_5, T12_3, P13, Y1, E3).
