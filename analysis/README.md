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
| `tasks/T<major>_<minor>_<slug>/` | One folder per task; outputs go to `results/analysis/T<major>_<minor>/` |

## Task map

| Task | Script | Output |
|---|---|---|
| T2.3 | `tasks/T2_3_tissue/run.py` | `results/analysis/T2_3/` tissue comparison, program means, reproduction check |
| T2.4 | `tasks/T2_4_covariates/run.py` | `results/analysis/T2_4/` covariate table, CV R², projection, masking, figure |
| T2.5 | `tasks/T2_5_celltype_margin/run.py` (needs `chimera_env`) | `results/analysis/T2_5/<run>__{S_gsk,C_gk,genes}.parquet`, acceptance, figures |
| T2.6 | `tasks/T2_6_matched/run.py` | `results/analysis/T2_6/category_matched.csv`, random sets, mtDNA detection check |
| T2.7 | `tasks/T2_7_consistency/run.py` | `results/analysis/T2_7/consistency_values.csv` |
| T3.5 | `tasks/T3_5_count_perm/make_control_h5ad.py`, `offline_null.py`, `analyze.py`; runs in `manifest.tsv` | `data/analysis/controls/<control>.{h5ad,json,perm.npz}`; `results/analysis/T3_5/null/`, `per_run.csv`, `arm_summary.csv`, `floor.csv` |
| T3.6 | `tasks/T3_6_matched/make_matched_h5ad.py`, `s_bal.py`, `analyze.py`; run in `manifest.tsv` | `data/analysis/controls/liv_4sp_matchA.{h5ad,json}`; `results/analysis/T3_6/s_bal/`, `matched_vs_canonical.csv`, `landmarks.csv` |
| T4.2 | `tasks/T4_2_gene_baselines/run.py` (needs `chimera_env`) | `results/analysis/T4_2/baselines__<cohort>.csv` (B1b, B1c, B1z, B2b, B3a, B3b per ENSG), pseudobulk units, summaries, marker check |
| T4.3 | `tasks/T4_3_scoreboard/pipeline.py` reads `MANIFEST.tsv` and runs `golds.py`, `baseline_splits.py`, the T4.2 B3b rerun on the consensus D and `run.py --mode real` (which calls `decide.py` and `figures.py`); `null_check.py` and `run.py --mode null-D/null-G` are the mechanics checks; `verify_primary.py`, `calibration_dominant.py` and the exploratory `dominance_exploratory.py` run after it (needs `chimera_env`) | `results/analysis/T4_3/`: `scoreboard.csv`, `head_to_head.csv`, `sets.csv`, `family3.csv`, `calibration_sets.csv`, `row.json`, `figures/`, `verify_primary.csv`, `calibration_dominant.csv`, `exploratory_dominance.csv`, `rho_decomposition_exploratory.csv` |
| T7.3, T7.8 | `tasks/T7_3_mash/{de,guilliams,hd4,post,hd1,t78_second_contrast}.py`, `run.sbatch` (needs `revision_de_env`) | `results/analysis/T7_3/`: `de/`, `h1.tsv`, `h1b.tsv`, `h4.tsv`, sensitivity, controls and calibration tables; `results/analysis/T7_8/`: `h1_gse136103.tsv`, `hagai.tsv`, `de/`, `gse136103/` |
| T8.1 | `tasks/T8_1_residue/run.py` (needs `chimera_env`; reads `notebooks/cache/gin_attribution/`) | `results/analysis/T8_1/residue_specificity.csv` (gene-swap and random-direction nulls), `r2_marginal_split.csv`, `class_crosstab.csv`, `residue-specificity.tex`, `summary.md` |
| X1 (exploratory) | `tasks/X1_testis_discovery/{build,seeds,leads,figures}.py` (needs `chimera_env`) | `results/analysis/X1/`: per-gene tables (`genes/`), stage excess, attribution, X, depth, mtDNA, tissue tables, seed robustness, `figures/` |
| X2 (exploratory) | `tasks/X2_liver_biology/{sets,forward,build,zonation,analyses,figures}.py` (needs `chimera_env`) | `results/analysis/X2/`: zone-resolved forward sums (`forward/`), per-gene liver tables (`genes/`), H+Mouse 10-seed consensus (`consensus/`), zonation (`zonation/`), drivers, named genes, mtDNA, divergent-conserved genes, presence, calibration, `figures/` |
| X3 (exploratory) | `tasks/X3_cross_cohort/{annotations,build,analyses,summarize,figures}.py` (needs `chimera_env`; annotation downloads in `data/analysis/x3_annotations/`, sha256 in `MANIFEST.tsv`) | `results/analysis/X3/`: per-gene consensus tables (`genes/`, `seeds/`), depth, core, class, ESM and agreement tables, `summary_*.csv`, `figures/` |
| X4 (exploratory) | `tasks/X4_where_differs/{rules,common,build,atlas,direct_margin,analyses,verdicts,figures,relations}.py` (needs `chimera_env`; definitions in `rules.py`) | `results/analysis/X4/`: per-gene tables (`genes/`), conservation map, strata and class contrasts against golds, direct-baseline disagreement and head-to-head, named genes, vignettes, re-reads of X1-X3, `verdicts.csv`, drivers, incremental value (`adds_*`), direct margin and its pattern and level terms (`cache/Sdir__*`, `mechanism_*`), `relations.csv`, `figures/` |
| Y2 (exploratory) | `tasks/Y2_functional/{rules,fetch.sh,annot_pheno,annot_reg,bw,build,analyses,literature,followup,figures,qc}` (needs `chimera_env`; definitions in `rules.py`; downloads in `data/analysis/Y2/` with `MANIFEST.tsv`, read with `python -I`) | `results/analysis/Y2/`: `annot/`, `genes_*.parquet`, `pheno_*`, `viability`, `regulatory*`, `discordant*`, `literature_*`, `F*` explanation tests, `figures/fig_y2_functional` |
| B1 | `tasks/B1_ground_truth/` (README there; needs `chimera_env`, tools in their envs) | `results/analysis/B1/`: `B1_tidy.csv`, `B1_overview.csv`, `B1_trackD_tidy.csv`, `qc/`, `scores/`, `truth/`, `metrics/`, `placebo/`, `track_d/`, `figures/` |
