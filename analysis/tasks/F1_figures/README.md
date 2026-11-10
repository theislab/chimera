<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# F1_figures

Redraws, at print size and from the same data, panels that notebooks and task scripts first drew: the gene-category panels of testis Human+Macaque and the Hallmark panels (notebooks 04 and 06), the closure panel of the cell-type decomposition and the ten-method integration view.

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | infrastructure |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Figure S20a, b (`fig:supp-dalpha-testis-detailed`); Supplementary Figure S21a, b, c (`fig:supp-dalpha-hallmark`); Supplementary Figure S22 (`fig:supp-celltype-closure`); Supplementary Figure S9 (`fig:supp-integration-purity`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.F1_figures.notebook_panels
    python -m analysis.tasks.F1_figures.regen_supp

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `figures/panels/supp_categories_testis/`: tracked in git
- `figures/panels/supp_hallmark/`: tracked in git
- `figures/panels/supp_celltype_closure/`: tracked in git
- `notebooks/figures/baselines_master_ext/`: tracked in git

**Reads** (as named in the code)

- `notebooks/cache/`
- `training/runs/`

**Notes.** notebook_panels.py runs the setup cells of notebooks 04 and 06 and draws the panels from the same gene tables and cached GSEA tables. regen_supp.py calls the plotting functions of T5_5_benchmark/views.py and of T2_5_celltype_margin/run.py. Both save through T8_4_figures/common.py under the names of figures/panel_names.tsv; the figures are composed by figures/compose.py (figures/README.md). compose.py of this folder only raises an ImportError that points there; a few spec files in task folders still import a layout tool from here and are not used by the build.
