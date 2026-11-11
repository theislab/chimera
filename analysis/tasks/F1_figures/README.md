<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# F1_figures

Redraws, at print size and from the cached enrichment tables, the Hallmark panels that notebook 06 first drew.

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | infrastructure |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Figure S21a, b, c (`fig:supp-dalpha-hallmark`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.F1_figures.notebook_panels

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `figures/panels/supp_hallmark/`: tracked in git

**Reads** (as named in the code)

- `training/runs/`

**Notes.** notebook_panels.py runs the setup cells of notebook 06 and draws the panels from the cached GSEA tables. It saves through T8_4_figures/common.py under the names of figures/panel_names.tsv; the figures are composed by figures/compose.py (figures/README.md). The gene-category panels of testis Human+Macaque are drawn by figures/redraw/categories.py, with those of four-species liver. compose.py of this folder only raises an ImportError that points to figures/compose.py; the spec files of three exploratory tasks (D1_sparse_genes, Y6_uses, Z2_embedding) still import a layout tool from here and are not used by the build.
