<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X4_where_differs

Where Δα and the count evidence differ from sequence constraint and from direct baselines: a likelihood margin computed without a model, split into pattern gain and level cost; relations among all scores; named genes.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Section 3.2 (`sec:axis`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Figure S16 (`fig:supp-relations-heatmap`); Supplementary Figure S17 (`fig:supp-relations-density`); Supplementary Section C.21.11 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X4_where_differs.build
    python -m analysis.tasks.X4_where_differs.atlas
    python -m analysis.tasks.X4_where_differs.direct_margin
    python -m analysis.tasks.X4_where_differs.analyses
    python -m analysis.tasks.X4_where_differs.verdicts
    python -m analysis.tasks.X4_where_differs.relations
    python -m analysis.tasks.X4_where_differs.figures
    python -m analysis.tasks.X4_where_differs.verify

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X4/`: bundle `analysis_results`

**Notes.** Labeled exploratory in the paper. Definitions in rules.py; verify.py recomputes headline numbers with independent code.
