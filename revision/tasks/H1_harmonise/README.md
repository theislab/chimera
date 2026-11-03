# H1 One reading across tissues

Definitions: `rules.py` (registry, thresholds, naming rule, sets, lineage panels). Exploratory throughout.

Run from `chimera/`, in `chimera_env`:

    bash revision/tasks/H1_harmonise/run_all.sh

| Step | Script | Output in `results/revision/H1/` |
|---|---|---|
| Per-gene tables, run health, identity checks, audit, cross-cohort table | `tables.py` | `genes/<cohort>.{parquet,tsv.gz}`, `all_genes.parquet`, `runs.tsv`, `identity_checks.csv`, `identity_cross_cohort.csv`, `within_tissue_models.csv`, `audit_cohorts.csv`, `cohort_table.csv` |
| Set z, recurrence, tissue-specific sets, separate axis | `recurrence.py` | `set_z.csv`, `recurrence.csv`, `tissue_specific.csv`, `lineage_z.csv`, `set_axis.csv`, `axis.csv` |
| Hallmark GSEA as in the first submission | `gsea.py` | `gsea/<cohort>.csv` |
| Archetypes as in the first submission | `archetypes.py` | `archetypes/` |
| Cell-type contributions from existing forward passes | `drivers.py` | `drivers/` |
| Panels, one PDF each | `figures.py` | `panels/` (not in the manuscript) |
| Composed figures | `F1_figures/compose.py --spec specs_h1.py --all` | `chimera-manuscript/figures/revision/<name>_composed.pdf`, previews in `results/revision/F1/` |
| Text | `tex.py` | `chimera-manuscript/supplementary/harmonised_cohorts.tex` |

## Cohorts and flags

26 cohorts: the 16 of the first passes and the ten C1 cohorts (`results/revision/C1/kept_runs.tsv`, joined by `rules.load_c1`; `H1_C1=0` leaves them out). The reading runs on eight tissues (`rules.TISSUE_ORDER`); the six-tissue thread is recomputed beside it (`six_*` readouts); muscle enters as a sensitivity (`D_with_muscle`). Names need `nameable`: strict support, no zero or quiet species, no mapping flag, not in a DLPFC lower tail (`reference_artefact`).

`rules.py` holds the registry; Van Zyl 5 species is grid 2 (`rev-vz-grid2-s42` to `s44`, T11_2c); grid 1 (`full_g1`) is not a row. H+Pig and H+M. fascicularis come from `extra_runs.tsv` (`cohort`, `seed`, `run`). `tables.py` stops if D or the saturated flag differ from the T11 tidy table.
Per-gene tables carry O1's flags as columns (`o1_T2`, `o1_M_wrong_row`, `o1_M_false_zero`, `o1_zero_any`); no gene is dropped. `nameable` is what figures and captions may name.

After a change of cohorts, also refresh the panels that read H1 (F1 ledger): `python -m revision.tasks.T8_4_figures.fig4_additions`, `python -m revision.tasks.T8_4_figures.fig7_validation`, `python -m revision.tasks.F1_figures.compose fig4_cont`. Then `chimera-private/.../drafts/numbers_H1.py`.
