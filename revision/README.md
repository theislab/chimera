# Revision analyses

Analyses added for the Cell Systems revision. Nothing here changes the `chimera` package.

| Path | Holds |
|---|---|
| `common/gene_ids.py` | The only gene-identity helper. `ortho_N` IDs are table-local, so lookups need a cohort |
| `common/cohorts.py` | Canonical runs, `final_state.npz` readout, raw prior, per-gene covariates |
| `common/cds.py` | Human CDS length and GC from the Ensembl 110 GTF and CDS FASTA |
| `common/outputs.py` | Atomic task outputs (`*.tmp` then rename), provenance, `DONE` last |
| `envs/` | Extra environment definitions and the `chimera_env` lock file |
| `slurm/` | CPU job templates, single and array |
| `tasks/T<major>_<minor>_<slug>/` | One folder per task; outputs go to `results/revision/T<major>_<minor>/` |

## Task map

| Task | Script | Output |
|---|---|---|
| T2.3 | `tasks/T2_3_tissue/run.py` | `results/revision/T2_3/` tissue comparison, program means, reproduction check |
| T2.4 | `tasks/T2_4_covariates/run.py` | `results/revision/T2_4/` covariate table, CV R², projection, masking, figure |
| T2.5 | `tasks/T2_5_celltype_margin/run.py` (needs `chimera_env`) | `results/revision/T2_5/<run>__{S_gsk,C_gk,genes}.parquet`, acceptance, figures |
| T2.6 | `tasks/T2_6_matched/run.py` | `results/revision/T2_6/category_matched.csv`, random sets, mtDNA detection check |
| T2.7 | `tasks/T2_7_consistency/run.py` | `results/revision/T2_7/consistency_values.csv` |
| T3.5 | `tasks/T3_5_count_perm/make_control_h5ad.py`, `offline_null.py`, `analyze.py`; runs in `manifest.tsv` | `data/revision/controls/<control>.{h5ad,json,perm.npz}`; `results/revision/T3_5/null/`, `per_run.csv`, `arm_summary.csv`, `floor.csv` |
| T3.6 | `tasks/T3_6_matched/make_matched_h5ad.py`, `s_bal.py`, `analyze.py`; run in `manifest.tsv` | `data/revision/controls/liv_4sp_matchA.{h5ad,json}`; `results/revision/T3_6/s_bal/`, `matched_vs_canonical.csv`, `landmarks.csv` |
| T4.2 | `tasks/T4_2_gene_baselines/run.py` (needs `chimera_env`) | `results/revision/T4_2/baselines__<cohort>.csv` (B1b, B1c, B1z, B2b, B3a, B3b per ENSG), pseudobulk units, summaries, marker check |
| T8.1 | `tasks/T8_1_residue/run.py` (needs `chimera_env`; reads `notebooks/cache/gin_attribution/`) | `results/revision/T8_1/residue_specificity.csv` (gene-swap and random-direction nulls), `r2_marginal_split.csv`, `class_crosstab.csv`, `residue-specificity.tex`, `summary.md` |
| X1 (exploratory) | `tasks/X1_testis_discovery/{build,seeds,leads,figures}.py` (needs `chimera_env`) | `results/revision/X1/`: per-gene tables (`genes/`), stage excess, attribution, X, depth, mtDNA, tissue tables, seed robustness, `figures/` |
| X3 (exploratory) | `tasks/X3_cross_cohort/{annotations,build,analyses,summarize,figures}.py` (needs `chimera_env`; annotation downloads in `data/revision/x3_annotations/`, sha256 in `MANIFEST.tsv`) | `results/revision/X3/`: per-gene consensus tables (`genes/`, `seeds/`), depth, core, class, ESM and agreement tables, `summary_*.csv`, `figures/` |
