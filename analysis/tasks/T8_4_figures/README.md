<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T8_4_figures

Panel scripts of main and supplementary figures that draw from cached result tables (integration, cell-type sources and tissue comparison, validation, control trainings, seed reliability).

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Figure 3 (`fig:integration`); Figure 4c (`fig:dalpha-biology`); Figure 6c (`fig:validation`); Supplementary Figure S12 (`fig:supp-dalpha-sources`); Supplementary Figure S26 (`fig:supp-validation-tests`); Supplementary Figure S62c (`fig:supp-main-gin-esm2`); Supplementary Figure S29 (`fig:supp-seed-reliability`); Supplementary Figure S10 (`fig:supp-integration-ext`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T8_4_figures.make_all (single figures: fig3_integration, fig3_integration_ext, fig4_additions, fig5_controls, fig7_validation, supp_figures)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `figures/panels/`: tracked in git
- `results/analysis/T8_4/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `B2_benchmark`, `H1_harmonise`, `T2_3_tissue`, `T2_5_celltype_margin`, `T3_4_destroy_features`, `T3_5_count_perm`, `T3_6_matched`, `T4_3_scoreboard`, `T5_6_gene_level`, `X2_liver_biology`, `X5_sequence_axis`, `X6_adds_beyond`
- `notebooks/cache/`
- `training/runs/`

**Notes.** Script names keep earlier figure numbers: fig3_integration_ext draws the composite strip of fig:integration and the heat maps of fig:supp-integration-ext; fig4_additions the gene plane of fig:dalpha-biology, the cell-type panel of fig:validation and panels of fig:supp-dalpha-sources; fig7_validation the panels of fig:supp-validation-tests; fig5_controls the control panel of fig:supp-main-gin-esm2. Single panels: <script> --panel <family/name>. Which panel file belongs to which figure: figures/INDEX.md.
