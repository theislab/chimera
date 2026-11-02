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
| Figures | `figures.py` | `chimera-manuscript/figures/revision/`, previews in `preview/` |
| Text | `tex.py` | `chimera-manuscript/supplementary/harmonised_cohorts.tex` |

## Adding Van Zyl H+Pig and H+M. fascicularis

When `ledgers/Z1_retune.md` lists their seeds, write `revision/tasks/H1_harmonise/extra_runs.tsv` (tab-separated: `cohort`, `seed`, `run`; cohort `vz_hpig` or `vz_hmfas`, one row per run of the selected recipe, failing runs included) and re-run `run_all.sh`. The health rule decides which enter. If the Van Zyl tasks have made `results/revision/T11/readout/genes_<hpig|hmfas>.parquet` and `T11/drivers/fw_<ds>_s<seed>.npz`, B1 and cell-type contributions join; otherwise those columns stay empty and the cohort is reported without them. Then re-run `chimera-private/.../drafts/numbers_H1.py`.
