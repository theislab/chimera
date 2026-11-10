<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T8_4_figures

Panel scripts of main and supplementary figures that draw from cached result tables (integration, cell-type sources and tissue comparison, validation, control trainings, seed reliability), and generated supplementary tables.

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Figure 3 (`fig:integration`); Figure 4 (`fig:dalpha-biology`); Figure 5 (`fig:validation`); Figure 8 (`fig:gin-esm2`); Supplementary Figure S26 (`fig:supp-seed-reliability`); Supplementary Figure S11 (`fig:supp-integration-ext`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T8_4_figures.make_all (single figures: fig3_integration, fig3_integration_ext, fig4_additions, fig5_controls, fig7_validation, supp_figures; tables: tables)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `figures/panels/`: tracked in git
- `results/analysis/T8_4/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `B2_benchmark`, `T2_3_tissue`, `T2_5_celltype_margin`
- `notebooks/cache/`
- `training/runs/`

**Notes.** Script names keep earlier figure numbers (fig7_validation draws fig:validation, fig5_controls the control panel of fig:gin-esm2). Which panel file belongs to which figure: figures/INDEX.md.
