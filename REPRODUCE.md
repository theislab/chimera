# Reproducing the Chimera results

This is the one guide from the paper to the code and data behind it. It covers what can be
reproduced and at what cost, the environment, the archived deposit, a map from every figure,
table and section of the paper to the code, the names used in files and folders, and the two
tests of a release. `[TAG]` and `[DOI]` stand for the release tag and the Zenodo DOI; they are
written in when a release is archived ([`release/README.md`](release/README.md)).

For the model itself and for training on your own cohort, see [`README.md`](README.md). For
where things are in the repository, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 1. What can be reproduced, and at what cost

| Level | What you do | Needs | Cost |
|---|---|---|---|
| Read the results | Open the archived tables: per-gene tables, result tables of every analysis | bundles `gene_tables`, `analysis_results` | minutes; no code |
| Check the release | Run the smoke test: the test suite, the per-gene tables recomputed from the archived model states, one model reloaded | the environment; the data and model bundles | about 5 minutes on 4 CPU threads |
| Recompose the figures | Build every figure from its panel files | the repository only | minutes ([`figures/README.md`](figures/README.md)) |
| Re-run an analysis | Run a task of `analysis/tasks/` on the archived models and tables | the environment; the bundles the task reads (its README) | seconds to hours per task; a few tasks need a cluster |
| Re-run a notebook | Execute a workflow notebook on the archived runs and caches | the environment; `models_canonical`, `notebook_caches` | minutes to an hour per notebook |
| Retrain a model | Train a recipe again | the environment; a data bundle | 20 to 55 minutes per canonical cohort on 16 CPU threads |

No number of the paper needs a model to be trained again: every analysis reads the archived
runs.

**What is exact and what is not.** The archived tables are recomputed exactly from the archived
model states (differences below 1e-9; the smoke test checks this). A retrained model is not
expected to repeat an archived one value by value:

- The canonical models were trained on Apple MPS, all other models on CPU. A fit repeats exactly
  only with the same seed on one CPU thread on one machine (`results/analysis/G1/seedfix_proof.csv`).
  With several threads the same seed can end in a different training state, and Δα follows that
  state.
- How much this matters depends on the cohort. Between seeds, Δα agrees at Spearman 0.93 to 0.97
  on four-species liver and the testis cohorts, but only about 0.3 on liver Human+Mouse and
  Human+Monkey (`results/gene_tables/reliability.csv`), where runs fall into an integrated and a
  poorly mixed state. Two repeats of liver Human+Mouse with unchanged seed gave 0.10 and 0.63
  (`results/analysis/T6_8b_repeat.csv`).
- The paper therefore reads seed-consensus tables, and the archived runs are the reference.
  Before comparing a retrained run with an archived one, check its training state in
  `summary.json` (`adv_acc_final`, `failed_expectations`).
- Re-rendered panels carry the same values as the archived ones but are not pixel-identical on
  another platform (fonts, antialiasing).
- Three notebooks first read live services (Enrichr gene-set libraries, Ensembl REST, AlphaFold
  DB). Their responses are frozen in the `notebook_caches` bundle; without it a run fetches
  current versions, which may differ.

## 2. Code and environment

```bash
git clone --branch [TAG] https://github.com/theislab/chimera.git
cd chimera
bash analysis/envs/build_env_from_lock.sh chimera_env   # Linux x86-64, CPU: every package at its locked version
pytest -q                                                # inside the environment
```

`build_env_from_lock.sh` creates the environment with Python 3.12.16, installs every pin of
[`analysis/envs/chimera_env.linux-64.lock.txt`](analysis/envs/chimera_env.linux-64.lock.txt),
installs this checkout in editable mode and runs `pip check`. On other platforms use
[`notebooks/00_environment.ipynb`](notebooks/00_environment.ipynb) or the manual steps of the
README. Every command below runs inside the environment, from the repository root. Scripts find
the repository from their own location; `CHIMERA_ROOT=/path/to/chimera` overrides it.

A few analyses need a second environment: differential expression (`revision_de_env`,
[`analysis/envs/revision_de_env.yml`](analysis/envs/revision_de_env.yml)), the compared tools
(`saturn_env`, `camex_env`, `samap_env`, `genespectra_env`, `cacimar_env`; files in
[`analysis/envs/`](analysis/envs/), tools and pinned commits in
[`analysis/external_tools/README.md`](analysis/external_tools/README.md)), and the figure build
(`chimera_figures_env`, [`figures/environment.yml`](figures/environment.yml)). Each task README
names its environment.

## 3. The deposit

Data, models and results are not in git (`data/`, `training/runs/`, `notebooks/cache/` and
`results/` are ignored). They are deposited at Zenodo as bundles, one record each; `[DOI]`
lists them all. A record holds `chimera_<bundle>.tar.gz`, `MANIFEST.tsv` (every member with its
size and sha256), `SHA256SUMS`, `README.md` and `LICENSES.md`. Member paths are relative to the
repository root, so every bundle is unpacked there:

```bash
sha256sum -c SHA256SUMS
tar -xzf chimera_<bundle>.tar.gz -C /path/to/chimera
```

| Bundle | Unpacks to | Content |
|---|---|---|
| `data_liver_testis` | `data/processed/` | The seven processed AnnData objects of the canonical cohorts (liver from the CAMEX bundle of Guilliams et al. 2022 and Wang et al. 2023; testis from Murat et al. 2023), with their build configurations, the Ensembl release 110 one-to-one ortholog tables for liver and testis, and the GIN feature scalers. camex_pancreas_unified.h5ad is the liver 4-species cohort (historical file name). |
| `data_other_cohorts` | `data/analysis/` | Processed objects built from public data: the eye cohort (Van Zyl aqueous-humor outflow tissues, GEO) with its five species subsets and the class-filter sensitivity build; bone marrow human, mouse lemur and mouse (Tabula Sapiens, Tabula Microcebus, Tabula Muris Senis via CELLxGENE); kidney (KPMP, CELLxGENE) and islet (CELLxGENE) healthy training objects and disease pseudobulks; cortex M1 and DLPFC, lung, skeletal muscle and tumor myeloid training objects (CELLxGENE, GEO) with their ortholog tables and feature scalers; liver MASH pseudobulks (GEO GSE202379, GSE192742) and cell annotations; the count and feature control objects; and the download manifests (URL, date, sha256) of every third-party input that is linked rather than redistributed. |
| `data_identifier_joined` | `data/analysis/O1/repair_*/`, `training/runs/o2-*/` | Liver (Human+Mouse, Human+Monkey, 4-species) and testis (Human+Mouse) objects in which every count column is taken from the raw row that carries the Ensembl gene ID of the ortholog table (repair_log.csv per tissue lists each column that differs from the symbol-joined object; build reports and ortholog tables are beside the objects), and the Chimera models trained on them with the canonical recipes (seeds 42-44; liver Human+Monkey seeds 42-51 with a matched control on the symbol-joined object). These are the recommended objects for new work on these cohorts. Liver macaque is identifier-joined only where its matrix carries Ensembl IDs; zebrafish carries none and is unchanged. |
| `models_canonical` | `training/runs/<run>/` (the runs of `configs/canonical_recipes.tsv`) | The 11 canonical Chimera runs (liver Human+Monkey seeds 42-44, Human+Mouse seeds 42-44, Human+Zebrafish, 4-species; testis Human+Macaque, Human+Mouse, 3-species), complete run folders: model/model.pt, final_state.npz (alpha, prior, correction, S_g, latent means), config.json, summary.json, per-epoch alpha snapshots, training metrics, tables and figure caches. Trained on Apple MPS; the recipe of each run is in configs/canonical_recipes.tsv of the code release. |
| `models_seeds_controls_cohorts` | `training/runs/rev-*/`, `training/runs/g1-*/` | Protein-feature-gate runs beyond the 11 canonical ones: further seeds of the canonical recipes (liver 4-species seeds 43-46, liver Human+Mouse seeds 45-51, testis 3-species seeds 43-46, testis Human+Mouse seeds 43-51), CPU platform and same-seed repeat runs, the control runs (feature destruction, count permutation, matched cell types, prior swap, testis without labels), the recipe grids and seeds of every further cohort (bone marrow, eye and its species subsets, islet, kidney, cortex M1 and DLPFC, lung, skeletal muscle, tumor myeloid cells), and the runs with the latent guard --clamp-z-logvar (rev-g1-*). Per run: model/, final_state.npz, config.json, summary.json, run_summary.json, report.md, provenance and the training metrics. For the latent-guard runs on thinned objects (g1-*): configuration, summaries and final state. CPU-trained. |
| `models_per_gene_gate` | `training/runs/z3-*/`, `z4-*/`, `rev-ctl-liv_hmouse-m-*/` | Fits with a per-gene gate (--config M: the canonical model with one free correction per gene in place of the protein-feature network, recipe otherwise unchanged): every fit of liver Human+Mouse, liver 4-species and testis Human+Mouse, and the seeds kept by the health rule on each of the 26 harmonized cohorts (results/analysis/Z4/kept.json). Per run: model/, final_state.npz, config.json, summary.json, run_summary.json, report.md, provenance and the training metrics. CPU-trained. |
| `gene_tables` | `results/gene_tables/` | One CSV per canonical cohort with the human Ensembl ID, symbol, prior, raw prior, alpha, correction, delta-alpha, the seed-consensus readout D, S_g and sign consistency, pooled over seeds; seed reliability and seed-pair agreement. Per-gene tables of the 26 harmonized cohorts in one schema, for the protein-feature gate and for the per-gene gate, with count-support, strict-support and naming flags. README.md inside describes every column. |
| `notebook_caches` | `notebooks/cache/`, `training/runs/baselines_*` | Caches read by the figure notebooks (including frozen responses of live services: Enrichr libraries, Ensembl REST sequences, AlphaFold DB structures, GIN attributions) and the integration baseline runs (scVI, scANVI, sysVI, Harmony, PCA, SATURN, CAMEX). |
| `analysis_results` | `results/analysis/<task>/` | Outputs of the analyses of the paper, one folder per task (results/analysis/<task>/): reliability and consensus, controls, gold standards and baselines, scoreboard, functional layer, disease cases, further cohorts, residue checks, the harmonized per-gene tables of 26 cohorts for both gates (H1, Z4) and the gate comparisons (Z3, Z4), the identifier-joined count comparison (O1), the latent-guard checks (G1), the gate gap (X8), integration with the per-gene gate (I1), the exploratory analyses (program atlas, protein and gene architecture, cell types and lineages, translational, regulatory architecture, literature species differences: E1-E6) and the figure sources. Most task folders are deposited as tables, text and figures without binary intermediates; permutation and leave-one-out differential-expression draws are not included. Per-gene EVEE values (license unstated) are not included; they are rebuilt from the archived EVEE download. |
| `tool_runs_saturn_camex` | `results/analysis/T5_*/` | SATURN and CAMEX runs (environment smoke test, pilot, cell-level runs on the seven canonical cohorts, gene-level runs with three seeds): outputs, embeddings, gene-level scalars and logs. SATURN macrogene-centroid initializations and pre-training snapshots are not included. |
| `gene_level_benchmark` | `results/analysis/B1/`, `B2/`, `training/runs/b1*-*/`, `b2-*/` | The gene-level benchmark: Track A (ground-truth spike-in of pattern, level and loss changes, and null alterations, on liver, testis and kidney Human+Mouse), Track B (robustness to perturbed cells, labels and depth), Track C (independent data across tissues), Track D (run time and memory) and the overview, for Chimera (both gates) and the compared tools: scored tables, truth sets, health records, logs, overview tables and figures, and configuration, summary and final state of every scored Chimera fit. Altered objects and the models of the altered fits are rebuilt by the task code. |
| `analysis_plan` | `results/analysis/_release/` | The pre-specified analysis plan of the study as frozen before any association between a Chimera readout and an independent gold standard was computed, with the sections added afterwards and the dated amendment log. |

The table is printed by `python release/zenodo/stage.py --bundle-table`; the declaration of the
deposit is [`release/zenodo/bundles.yaml`](release/zenodo/bundles.yaml).

**Which bundles for which purpose**

| Purpose | Bundles |
|---|---|
| Smoke test, per-gene values of the canonical cohorts | `data_liver_testis`, `models_canonical` |
| Smoke test with all per-gene table checks | add `gene_tables`, `analysis_results`, `models_seeds_controls_cohorts`, `models_per_gene_gate` |
| Workflow notebooks | `data_liver_testis`, `models_canonical`, `notebook_caches` |
| An analysis task | the bundles named in the task's README (section 4) |
| New work on liver or testis | `data_identifier_joined` |

**Not redistributed.** Raw data stay with their sources (accessions in the paper and in the
`MANIFEST.tsv` files under `data/analysis/`, which give URL, date and sha256 of every download).
The SATURN ESM-2 embeddings are not redistributed; the gene features derived from them are
inside the AnnData objects. EVEE evolutionary statistics carry no stated license, so per-gene
EVEE columns are removed from the deposited tables and are rebuilt from the EVEE download listed
in `data/analysis/gold/G1/MANIFEST.tsv`. Altered objects of the benchmark and the models fitted
on them are rebuilt by the task code. Licenses of third-party sources:
[`release/zenodo/LICENSES.md`](release/zenodo/LICENSES.md).

## 4. From the paper to the code and data

Three steps lead from an item of the paper to its source.

1. **Find the item in the map below** (by figure, table or section number; the LaTeX label is
   the stable name). The map names the task folders of [`analysis/tasks/`](analysis/README.md)
   and the notebooks behind it.
2. **Open the task folder's `README.md`**: what the task does, the commands in run order, its
   environment, what it reads, where it writes, and the deposit bundle of each output.
3. **For a figure, open [`figures/INDEX.md`](figures/INDEX.md)**: page in
   [`figures/figures.pdf`](figures/figures.pdf), the panel files the figure is composed from, and
   its spec. `python -m figures.compose <key>` composes one figure; `bash figures/build.sh`
   builds all ([`figures/README.md`](figures/README.md)).

A number in the running text comes from the result folders of the tasks listed for its section;
the tables are CSV or parquet files with one row per test or gene.

### 4.1 Map

Figure, table and section numbers are those of the current manuscript build; labels do not
change when numbers do. Panel letters are given in parentheses where a task feeds single panels.
"Other code" names notebooks and scripts outside `analysis/tasks/`.

<!-- paper-map:begin (generated by analysis/make_index.py; do not edit by hand) -->
| In the paper | Label | What | Task folders behind it | Other code behind it |
|---|---|---|---|---|
| Figure 1 | `fig:overview`; figure key `main_overview` | Overview: what the model takes, does and returns | - | `figures/redraw/` (drawn, not computed) |
| Section 2 | `sec:methods` | Methods | - | `chimera/` (the model) |
| Figure 2 | `fig:alpha-intro`; figure key `main_alpha_vs_prior` | The gate value α against the protein prior | - | `notebooks/03_alpha_master.ipynb` |
| Section 3.1 | `sec:model` | One fit aligns the cells and reads each gene against its protein | `T5_3_runs`, `T5_5_benchmark`, `Z2_embedding`, `B2_benchmark`, `C1_cohorts` | `notebooks/07_baselines_master.ipynb`; `notebooks/08_chimera_umap.ipynb` |
| Section 3.2 | `sec:axis` | Deployment is an axis of its own, beside protein sequence | `T2_3_tissue`, `X5_sequence_axis`, `X4_where_differs` | - |
| Figure 3 | `fig:integration`; figure key `main_integration` | Integration benchmark on the seven liver and testis cohorts | `T5_3_runs`, `T5_5_benchmark`, `T8_4_figures` | `notebooks/02_baselines.ipynb`; `notebooks/07_baselines_master.ipynb` |
| Section 3.3 | `sec:findings` | The axis reveals less conserved proteins used alike and conserved proteins used differently | `T2_5_celltype_margin`, `T2_6_matched`, `X1_testis_discovery`, `X2_liver_biology`, `X4_where_differs`, `Y3_findings`, `Y4_regions`, `Y5_aspects`, `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble`, `T3_6_matched`, `C1_cohorts` | - |
| Figure 4 | `fig:dalpha-biology`; figure key `main_delta_alpha_liver` | Δα by gene category and across the protein prior, four-species liver | `T2_3_tissue` (f), `X5_sequence_axis` (g), `T2_5_celltype_margin` (d), `T2_4_covariates` (e), `T8_4_figures` | `notebooks/04_delta_alpha_biology.ipynb`; `notebooks/04_delta_alpha_biology_simpler.ipynb`; `notebooks/06_pathway_analysis.ipynb` |
| Section 3.4 | `sec:validation` | Independent data back the count evidence on cell-state pattern; Δα passes on liver pattern | `T4_2_gene_baselines`, `T4_3_scoreboard`, `X6_adds_beyond` | - |
| Table 1 | `tab:validation` | Pre-specified validation of Δα against independent data | `T4_2_gene_baselines`, `T4_3_scoreboard` | - |
| Figure 5 | `fig:validation`; figure key `main_validation` | Pre-specified tests against independent data and comparison with direct baselines | `T4_2_gene_baselines` (c), `T4_3_scoreboard` (a, b, c), `Z3_free_gate` (e, f, h, i), `F2_per_gene` (e, f, g, h, i), `T5_6_gene_level` (c), `X6_adds_beyond` (d), `T8_4_figures` | - |
| Section 3.4 | `sec:reliability` | Reliability and feature dependence (paragraph) | `T3_3_consensus`, `T2_4_covariates`, `T3_4_destroy_features`, `T3_5_count_perm`, `D1_sparse_genes` | - |
| Section 3.5 | `sec:per-gene` | Read gene by gene from counts and evaluated afterward, Δα agrees with independent data in liver and testis | `Z3_free_gate`, `F2_per_gene`, `I1_free_gate_integration`, `X8_gate_divergence` | - |
| Section 3.6 | `sec:adds` | The count evidence adds to direct baselines on pattern and needs no labels in liver | `T5_6_gene_level`, `X6_adds_beyond`, `B1_ground_truth`, `B2_benchmark`, `E6_known_differences` | - |
| Section 3.7 | `sec:tissues` | One reading across eight tissues | `T11_1_vanzyl`, `T12_1_bone_marrow`, `T11_4_vz_integration`, `Z4_free_gate_all`, `H1_harmonise`, `C1_cohorts`, `Y1_gene_tissue`, `E1_atlas`, `E3_cells`, `E2_protein`, `E5_regulatory`, `T11_5_vz_genes`, `T11_6_vz_glaucoma`, `T11_7_vz_figure`, `T12_3_tests` | - |
| Figure 6 | `fig:one-reading`; figure key `main_tissues` | Δα against protein identity and dN/dS, and named genes, in eight tissues | `H1_harmonise` (a, b, c), `Y1_gene_tissue` (d, e, f, g, h, i), `E3_cells` (j), `E2_protein` (k), `E5_regulatory` (l) | - |
| Section 3.8 | `sec:disease` | Program transfer shows which human disease responses a mouse model reproduces | `Z4_free_gate_all`, `C1_cohorts`, `T7_3_mash`, `T7_9_islet`, `P13_kidney`, `E4_translational` | - |
| Figure 7 | `fig:disease`; figure key `main_disease` | Transfer of disease programs between human and mouse | `T7_3_mash`, `T7_9_islet`, `P13_kidney` (a, b, c), `T7_10_fig8` | - |
| Section 3.9 | `sec:protein` | The protein-feature gate maps deployment over protein space | `X3_cross_cohort`, `T8_1_residue` | `notebooks/09_gin_attention_overall.ipynb`; `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb`; `notebooks/10_gin_attribution_protein_structure.ipynb` |
| Figure 8 | `fig:gin-esm2`; figure key `main_protein` | Inputs of the gate network and control trainings | `T3_4_destroy_features` (c), `T3_5_count_perm` (c), `T3_6_matched` (c), `T8_4_figures` | `notebooks/09_gin_attention_overall.ipynb`; `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Table 2 | `tab:when-to-use` | When to use which readout or method | `X6_adds_beyond`, `B1_ground_truth`, `B2_benchmark` | - |
| Supplementary Table S1 | `tab:tool-inclusion` | Tools considered for comparison and why each was included or not | - | text of the manuscript (no computed values) |
| Supplementary Table S2 | `tab:related-work-comparison` | Comparison with related methods | - | text of the manuscript (no computed values) |
| Supplementary Table S3 | `tab:assumptions-summary` | Summary of assumptions and limitations | `T2_7_consistency` | text of the manuscript (no computed values) |
| Supplementary Section C | `sec:extended-methods` | Extended Methods | `O1_orthologs` | `chimera/`, `training/` |
| Supplementary Table S4 | `tab:notation` | Notation | - | text of the manuscript (no computed values) |
| Supplementary Table S5 | `tab:notation-continued` | Notation (continued) | - | text of the manuscript (no computed values) |
| Supplementary Section C.8.4 | `sssec:per-gene-gate` | The per-gene gate | `Z3_free_gate` | - |
| Supplementary Section C.12.2 | `par:latent-guard` | Latent guard (paragraph) | `G1_guard`, `T12_2b_debug` | - |
| Supplementary Section C.21 | `ssec:revision-analyses` | Pre-specified Analysis Plan and Departures | - | - |
| Supplementary Section C.21.1 | `sssec:rev-plan` | Pre-specified analysis plan | `Z1_retune`, `T11_2_recipe`, `T11_2b_vz_subsets`, `T11_2c_count_rule`, `T12_2_recipe`, `T12_2b_recipe`, `T7_9_recipe`, `T7_9_recipe_b`, `P13_recipe` | - |
| Supplementary Section C.21.2 | `sssec:rev-seeds` | Seeds, seed pooling and training states | `T3_1_platform`, `T3_2_seeds`, `T3_2b_extra_seeds`, `T3_1b_integration_gate`, `T6_8b_repeat`, `T3_3_consensus` | - |
| Supplementary Section C.21.3 | `sssec:rev-identity` | Gene identity and tissue comparison | `T2_2_regen`, `T2_3_tissue`, `T1_3_smoke` | - |
| Supplementary Section C.21.4 | `sssec:rev-covariates` | Covariate audit, masking and margin support | `T2_4_covariates` | - |
| Supplementary Section C.21.5 | `sssec:rev-decomposition` | Cell-type decomposition of the count-evidence margin | `T2_5_celltype_margin` | - |
| Supplementary Section C.21.6 | `sssec:rev-categories` | Category tests under matching | `T2_6_matched` | - |
| Supplementary Section C.21.7 | `sssec:rev-controls` | Retraining controls | `T3_4_destroy_features`, `T3_5_count_perm`, `T3_6_matched` | - |
| Supplementary Section C.21.8 | `sssec:rev-validation` | Independent validation and direct baselines | `T4_1_gold`, `T4_2_gene_baselines`, `T4_3_scoreboard` | - |
| Supplementary Section C.21.9 | `sssec:rev-benchmark` | Integration benchmark with SATURN and CAMEX, and the testis label ablation | `T5_1_envs`, `T5_2_adapters`, `T5_3_runs`, `T5_5_benchmark`, `T5_4b_label_ablation` | - |
| Supplementary Section C.21.10 | `sssec:rev-gene-level` | Gene-level comparison with SATURN and CAMEX | `T5_6_gene_level` | - |
| Supplementary Section C.21.11 | `sssec:rev-functional` | Functional layer and prior swap | `T6_1_functional`, `T6_8b_repeat`, `T6_common`, `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble`, `T6_8_priorswap` | - |
| Supplementary Section C.21.12 | `sssec:rev-exploratory` | Exploratory analyses | `X1_testis_discovery`, `X2_liver_biology`, `X4_where_differs`, `Y3_findings`, `X6_adds_beyond`, `X7_architecture_outputs` | - |
| Supplementary Section C.21.13 | `sssec:rev-residue` | Residue specificity control | `T8_1_residue` | - |
| Supplementary Section C.21.14 | `sssec:rev-disease` | Disease case studies, bone marrow and eye | `T7_1_holdouts`, `T7_2_disease_objects`, `T11_1_vanzyl`, `T12_1_bone_marrow`, `T11_3_vz_seeds`, `T11_8_vz_class_filter`, `T11_5_vz_genes`, `T11_6_vz_glaucoma`, `T12_3_tests`, `T7_3_mash`, `T7_9_islet`, `P13_kidney` | - |
| Supplementary Section C.21.15 | `sssec:rev-dataprep` | Data preparation of the cohorts beyond liver and testis | `C1_cohorts` | - |
| Supplementary Section C.21.16 | `sssec:rev-gene-benchmark` | Gene-level benchmark | `B1_ground_truth`, `B2_benchmark` | - |
| Supplementary Section C.22 | `ssec:reproducibility` | Reproducibility | - | `REPRODUCE.md`, `release/` |
| Supplementary Table S6 | `tab:failure-modes` | Failure modes and the loss term that closes each | - | text of the manuscript (no computed values) |
| Supplementary Table S7 | `tab:covariate-flow` | Which covariate enters which module | - | text of the manuscript (no computed values) |
| Supplementary Table S8 | `tab:scvi-contrast` | Contrast with scVI | - | text of the manuscript (no computed values) |
| Supplementary Table S9 | `tab:three-means` | The three negative-binomial means | - | text of the manuscript (no computed values) |
| Supplementary Table S10 | `tab:gradient-routing-matrix` | Gradient routing | - | text of the manuscript (no computed values) |
| Supplementary Table S11 | `tab:loss-components` | Loss components | - | text of the manuscript (no computed values) |
| Supplementary Table S12 | `tab:hyperparameters` | Hyperparameters | - | text of the manuscript (no computed values) |
| Supplementary Table S13 | `tab:k-scaling` | Scaling of recipe weights with the number of species | - | text of the manuscript (no computed values) |
| Supplementary Table S14 | `tab:recipe-matrix` | Recipe matrix | - | text of the manuscript (no computed values) |
| Supplementary Table S15 | `tab:per-dataset-recipes` | Recipe and outcome per canonical cohort | `T2_7_consistency` | - |
| Supplementary Table S16 | `tab:evaluation-gates` | Training expectations (evaluation gates) | - | text of the manuscript (no computed values) |
| Supplementary Table S17 | `tab:evaluation-gates-e8` | Training expectations: comparison with scVI | - | text of the manuscript (no computed values) |
| Supplementary Section D.1 | `ssec:er-model` | Chimera learns a protein-relative gene readout on an integrated cell-state substrate | `T2_2_regen`, `T5_5_benchmark`, `I1_free_gate_integration` | `notebooks/03_alpha_master.ipynb`; `notebooks/07_baselines_master.ipynb` |
| Supplementary Section D.2 | `ssec:er-axis` | Chimera's Δα places deployment pattern relative to protein conservation | `T2_2_regen`, `T2_3_tissue`, `X5_sequence_axis`, `T2_6_matched`, `X2_liver_biology`, `X4_where_differs`, `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble` | - |
| Supplementary Section D.3 | `ssec:er-validation` | Seed reliability, retraining controls and independent data show what Δα reads from counts and from features | `T4_3_scoreboard`, `T3_3_consensus`, `T3_4_destroy_features`, `T3_5_count_perm`, `T3_6_matched`, `T5_6_gene_level` | - |
| Supplementary Section D.4 | `sec:supp-gate-gap` | The two readouts differ reproducibly where a gene departs from its protein family | `X8_gate_divergence` | - |
| Supplementary Section D.5 | `ssec:er-e6` | Published species differences in expression level sit low on the count evidence | `E6_known_differences` | - |
| Supplementary Section D.6 | `ssec:er-b1` | In a ground-truth benchmark, S_g detects level and pattern changes from one fit | `B1_ground_truth` | - |
| Supplementary Section D.7 | `ssec:er-e1` | Deployment programs recur across eight tissues and cut across protein conservation | `E1_atlas` | - |
| Supplementary Section D.8 | `ssec:er-e3` | Cell-type programs sit above their protein prior while their levels move | `E3_cells` | - |
| Supplementary Section D.9 | `ssec:er-e2` | Protein architecture and gene history predict where deployment sits relative to protein conservation | `E2_protein` | - |
| Supplementary Section D.10 | `ssec:er-e5` | Regulatory input predicts where deployment sits relative to protein conservation | `E5_regulatory` | - |
| Supplementary Section D.11 | `ssec:er-disease` | Mouse ischemic injury reproduces human kidney-injury programs by stage | `T7_3_mash`, `T7_9_islet`, `P13_kidney` | - |
| Supplementary Section D.12 | `ssec:er-e4` | Drug targets and disease genes sit above their protein prior in their own tissue | `E4_translational` | - |
| Supplementary Section D.13 | `ssec:er-gin` | Chimera's GIN projects count evidence onto protein-language and expression features | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Section D.14 | `ssec:er-atlas` | Chimera learns a protein-space atlas of sequence-relative expression deployment | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Section D.15 | `ssec:er-residue` | Exploratory projection of Δα attribution onto protein structure | `T2_2_regen`, `T8_1_residue` | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Table S18 | `tab:data-preparation` | Data preparation of every cohort | `T7_2_disease_objects`, `T11_1_vanzyl`, `T12_1_bone_marrow`, `O1_orthologs`, `X2_liver_biology`, `C1_cohorts`, `T7_9_islet`, `P13_kidney` | `notebooks/01_data_preparation.ipynb` (liver and testis) |
| Supplementary Table S19 | `tab:harmonised-cohorts` | The 26 cohorts: cells, genes, recipe, runs and training health | `Z1_retune`, `T11_2_recipe`, `T11_2b_vz_subsets`, `T11_2c_count_rule`, `T12_2_recipe`, `T12_2b_recipe`, `T7_9_recipe`, `T7_9_recipe_b`, `P13_recipe`, `H1_harmonise`, `C1_cohorts` | - |
| Supplementary Table S20 | `tab:harmonised-rules` | Rules of the per-gene tables and of gene naming | `H1_harmonise` | - |
| Supplementary Table S21 | `tab:harmonised-rules-c1` | Rules for the cohorts of cortex, lung, muscle and tumor | `H1_harmonise` | - |
| Supplementary Figure S1 | `fig:supp-main-alpha-intro` | Extended description of Figure 2 (text only) | - | `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S2 | `fig:supp-alpha-training-dynamics`; figure key `supp_alpha_training` | Training dynamics of α | - | `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S3 | `fig:supp-alpha-robustness`; figure key `supp_alpha_robustness` | Program-level Δα across seeds, species sets and tissues | `T2_2_regen` (g, h, i, j, k) | `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S4 | `fig:supp-alpha-decorrelation`; figure key `supp_alpha_decorrelation` | Δα against expression magnitude statistics | - | `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S5 | `fig:supp-main-integration`; figure key `supp_integration_seven_methods` | Integration scores, seven-method panel | - | `notebooks/02_baselines.ipynb`; `notebooks/07_baselines_master.ipynb` |
| Supplementary Figure S6 | `fig:supp-integration-metrics`; figure key `supp_integration_metrics` | Raw integration metrics, seven-method panel | - | `notebooks/07_baselines_master.ipynb` |
| Supplementary Figure S7 | `fig:supp-integration-testis-umap`; figure key `supp_integration_umap_testis` | UMAP by method, three-species testis | - | `notebooks/08_chimera_umap.ipynb` |
| Supplementary Figure S8 | `fig:supp-integration-liver-umap`; figure key `supp_integration_umap_liver` | UMAP by method, liver Human+Monkey | - | `notebooks/08_chimera_umap.ipynb` |
| Supplementary Figure S9 | `fig:supp-integration-purity`; figure key `supp_integration_ten_methods` | Integration: mixing and conservation, ten-method panel | `T5_5_benchmark` | - |
| Supplementary Figure S10 | `fig:supp-integration-raw`; figure key `supp_integration_raw_metrics` | Raw integration metrics, ten-method panel | `T5_5_benchmark` | - |
| Supplementary Figure S11 | `fig:supp-integration-ext`; figure key `supp_integration_eleven_cohorts` | Integration benchmark on eleven cohorts | `T11_4_vz_integration`, `B2_benchmark`, `T8_4_figures` | - |
| Supplementary Table S22 | `tab:benchmark-label-use` | Use of cell-type labels by each integration method | `T5_4b_label_ablation` | text of the manuscript (no computed values) |
| Supplementary Table S23 | `tab:integration-per-gene-gate` | Integration scores of the per-gene gate, seven canonical cohorts | `I1_free_gate_integration` | - |
| Supplementary Table S24 | `tab:integration-per-gene-gate-harmonised` | Integration scores of the per-gene gate, all cohorts | `I1_free_gate_integration` | - |
| Supplementary Figure S12 | `fig:supp-main-dalpha-biology` | Extended description of Figure 4 (text only) | - | `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S13 | `fig:supp-sequence-axis`; figure key `supp_sequence_axis` | Protein identity, dN/dS and Δα gene by gene | `X5_sequence_axis` | - |
| Supplementary Figure S14 | `fig:supp-known-measures`; figure key `supp_known_measures` | Δα and count evidence among known gene measures | `X5_sequence_axis` | - |
| Supplementary Figure S15 | `fig:supp-relations-heatmap`; figure key `supp_relations_heatmap` | Relations of Δα and count evidence to sequence, expression and direct baselines | `X4_where_differs` | - |
| Supplementary Figure S16 | `fig:supp-relations-density`; figure key `supp_relations_density` | The same relations gene by gene | `X4_where_differs` | - |
| Supplementary Figure S17 | `fig:supp-worked-genes`; figure key `supp_worked_genes` | Two worked genes in the counts | `Y3_findings` | - |
| Supplementary Figure S18 | `fig:supp-dalpha-broad-categories`; figure key `supp_categories_broad` | Δα by broad gene category, liver and testis | `T2_2_regen` (d) | `notebooks/04_delta_alpha_biology_simpler.ipynb` |
| Supplementary Figure S19 | `fig:supp-dalpha-liver-detailed`; figure key `supp_categories_liver` | Δα by gene category, four-species liver, all 38 categories | - | `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S20 | `fig:supp-dalpha-testis-detailed`; figure key `supp_categories_testis` | Δα by gene category, testis Human+Macaque | `T2_2_regen`, `F1_figures` (a, b) | `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S21 | `fig:supp-dalpha-hallmark`; figure key `supp_hallmark` | Hallmark pathway enrichment of Δα | `T2_2_regen` (c), `F1_figures` (a, b, c) | `notebooks/06_pathway_analysis.ipynb` |
| Supplementary Figure S22 | `fig:supp-celltype-closure`; figure key `supp_celltype_closure` | Closure of the cell-type decomposition of the count evidence | `T2_5_celltype_margin` | - |
| Supplementary Figure S23 | `fig:supp-celltype-seeds`; figure key `supp_celltype_seeds` | Seed stability of cell-type contributions to the count evidence | `T2_5_celltype_margin` | - |
| Supplementary Figure S24 | `fig:supp-functional-layer`; figure key `supp_functional_layer` | Pathway and network coherence and conservation | `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble` | - |
| Supplementary Figure S25 | `fig:supp-covariate-audit`; figure key `supp_covariate_audit` | Covariate audit of Δα in seven cohorts | `T2_4_covariates` | - |
| Supplementary Figure S26 | `fig:supp-seed-reliability`; figure key `supp_seed_reliability` | Agreement between training seeds | `T3_3_consensus`, `T8_4_figures` | - |
| Supplementary Figure S27 | `fig:supp-gate-gap`; figure key `supp_gate_gap` | Where the two readouts agree and differ | `X8_gate_divergence` | - |
| Supplementary Figure S28 | `fig:supp-gate-gap-cont`; figure key `supp_gate_gap_classes` | The difference between the readouts by tissue, gene class and independent reference | `X8_gate_divergence` | - |
| Supplementary Figure S29 | `fig:supp-gate-gap-named`; figure key `supp_gate_gap_named` | Named genes where the two readouts differ | `X8_gate_divergence` | - |
| Supplementary Figure S30 | `fig:supp-e6`; figure key `supp_published_differences` | Published species differences on the count evidence | `E6_known_differences` | - |
| Supplementary Figure S31 | `fig:supp-b1`; figure key `supp_ground_truth` | Ground-truth benchmark: level alterations and protein identity | `B1_ground_truth` | - |
| Supplementary Figure S32 | `fig:supp-b2-overview`; figure key `supp_benchmark_overview` | Gene-level benchmark: overview | `B2_benchmark` | - |
| Supplementary Figure S33 | `fig:supp-b2-tasks`; figure key `supp_benchmark_tasks` | Gene-level benchmark: every task | `B2_benchmark` | - |
| Supplementary Figure S34 | `fig:supp-h1-recurrence-all`; figure key `supp_tissues_recurrence` | All gene sets that recur across tissues | `H1_harmonise` | - |
| Supplementary Figure S35 | `fig:supp-e1`; figure key `supp_programs_recurrent` | Gene programs that recur across eight tissues | `E1_atlas` | - |
| Supplementary Figure S36 | `fig:supp-e1-cont`; figure key `supp_programs_by_tissue` | Tissue programs in each tissue | `E1_atlas` | - |
| Supplementary Figure S37 | `fig:supp-e1-genes`; figure key `supp_programs_named_genes` | Named genes that recur across tissues | `E1_atlas` | - |
| Supplementary Figure S38 | `fig:supp-h1-bone-marrow`; figure key `supp_tissue_bone_marrow` | The reading in bone marrow | `H1_harmonise` | - |
| Supplementary Figure S39 | `fig:supp-h1-kidney`; figure key `supp_tissue_kidney` | The reading in kidney | `H1_harmonise` | - |
| Supplementary Figure S40 | `fig:supp-h1-islet`; figure key `supp_tissue_islet` | The reading in pancreatic islet | `H1_harmonise` | - |
| Supplementary Figure S41 | `fig:supp-h1-eye`; figure key `supp_tissue_eye` | The reading in the eye | `H1_harmonise` | - |
| Supplementary Figure S42 | `fig:supp-h1-cortex`; figure key `supp_tissue_cortex` | The reading in cerebral cortex | `H1_harmonise` | - |
| Supplementary Figure S43 | `fig:supp-h1-lung`; figure key `supp_tissue_lung` | The reading in lung | `H1_harmonise` | - |
| Supplementary Figure S44 | `fig:supp-h1-muscle`; figure key `supp_tissue_muscle` | The reading in skeletal muscle | `H1_harmonise` | - |
| Supplementary Figure S45 | `fig:supp-h1-tumour`; figure key `supp_tissue_tumor_myeloid` | The reading in lung-tumor myeloid cells | `H1_harmonise` | - |
| Supplementary Figure S46 | `fig:supp-h1-lineage`; figure key `supp_tissues_lineage` | Lineage marker sets across tissues | `H1_harmonise` | - |
| Supplementary Figure S47 | `fig:supp-bone-marrow`; figure key `supp_bone_marrow_reference` | Bone marrow against sorted-lineage conservation of expression | `T12_3_tests` | - |
| Supplementary Figure S48 | `fig:supp-e3-compartments`; figure key `supp_cell_compartments` | Home compartments of genes against Δα | `E3_cells` | - |
| Supplementary Figure S49 | `fig:supp-e3-lineages`; figure key `supp_cell_lineages` | Cell types that carry shared deployment | `E3_cells` | - |
| Supplementary Figure S50 | `fig:supp-e3-primates`; figure key `supp_primate_branches` | Species-specific departures along the primate phylogeny | `E3_cells` | - |
| Supplementary Figure S51 | `fig:supp-e2`; figure key `supp_protein_architecture` | Protein architecture and gene history against Δα | `E2_protein` | - |
| Supplementary Figure S52 | `fig:supp-e5`; figure key `supp_regulatory_architecture` | Regulatory and gene-architecture features against Δα | `E5_regulatory` | - |
| Supplementary Figure S53 | `fig:supp-e4`; figure key `supp_disease_genes` | Disease genes and drug targets by organ system | `E4_translational` | - |
| Supplementary Figure S54 | `fig:supp-e4-targets`; figure key `supp_drug_targets` | Approved drug targets against the protein prior | `E4_translational` | - |
| Supplementary Figure S55 | `fig:supp-main-gin-esm2` | Extended description of Figure 8a-c (text only) | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S56 | `fig:supp-gin-esm2-linear-baseline`; figure key `supp_gin_linear_baseline` | Linear surrogate and scalar-correlation controls of the gate correction | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S57 | `fig:supp-gin-esm2-attribution-suite`; figure key `supp_gin_attribution_suite` | Attribution of the gate correction to input groups, all methods | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S58 | `fig:supp-gin-esm2-method-consensus`; figure key `supp_gin_method_consensus` | Consensus of the attribution methods | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S59 | `fig:supp-gin-esm2-orthogonality`; figure key `supp_gin_orthogonality` | ESM-2 and expression inputs: orthogonality and ablation | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S60 | `fig:supp-gin-esm2-per-pc`; figure key `supp_gin_per_pc` | Contribution of single ESM-2 principal components | - | `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S61 | `fig:supp-main-esm2-umap` | Extended description of Figure 8d-n (text only) | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Figure S62 | `fig:supp-esm2-umap-spatial-autocorr`; figure key `supp_atlas_spatial_autocorrelation` | Neighborhood coherence of α and Δα on the ESM-2 graph | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Figure S63 | `fig:supp-esm2-umap-archetype-trajectory`; figure key `supp_atlas_archetype_trajectory` | Δα of the archetypes during training | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Figure S64 | `fig:supp-esm2-umap-archetype-enrichment`; figure key `supp_atlas_archetype_enrichment` | Gene categories enriched in the archetypes | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Table S25 | `tab:residue-specificity` | Specificity of the residue-level contrasts | `T8_1_residue` | - |
| Supplementary Figure S65 | `fig:supp-main-gin-protein`; figure key `supp_residue_audit` | Residue-level attribution audit | `T2_2_regen` (e) | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb`; `notebooks/08_ortholog_qc.ipynb` |
| Supplementary Figure S66 | `fig:supp-gin-protein-residue-beta`; figure key `supp_residue_mixed_effects` | Residue-level mixed-effects regression | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Figure S67 | `fig:supp-gin-protein-cliffs-delta`; figure key `supp_residue_cliffs_delta` | Protein-replicated effect sizes of residue features | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Figure S68 | `fig:supp-gin-protein-attention-by-feature`; figure key `supp_residue_attention_by_feature` | Per-residue attribution by structural feature | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Figure S69 | `fig:supp-gin-protein-etv1`; figure key `supp_residue_card_etv1` | Per-gene card: ETV1 | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Figure S70 | `fig:supp-gin-protein-hprt1`; figure key `supp_residue_card_hprt1` | Per-gene card: HPRT1 | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Figure S71 | `fig:supp-gin-protein-runx1`; figure key `supp_residue_card_runx1` | Per-gene card: RUNX1 | - | `notebooks/10_gin_attribution_protein_structure.ipynb`; `notebooks/10b_gin_attribution_figures.ipynb` |
<!-- paper-map:end -->

### 4.2 Workflow notebooks

The notebooks render the panels of the model's own figures into `notebooks/figures/<name>/`
(tracked) from the canonical runs. Notebooks 03 to 11 are generated by their builders
(`notebooks/_*_builder.py`).
Run one headless with

```bash
papermill notebooks/<notebook>.ipynb <output folder>/<notebook>.ipynb --cwd notebooks -k python3
```

(`-k python3` is the kernel of the active environment; `00_environment.ipynb` also registers it
as `chimera_env`). Writing the output back onto the notebook instead replaces the tracked copy.

| Notebook | Does | Feeds |
|---|---|---|
| `00_environment` | Builds `chimera_env` and its Jupyter kernel (run interactively) | - |
| `01_data_preparation` | Builds the liver and testis objects from the CAMEX bundle and the ESM-2 embeddings | `data/processed/`; Supplementary Table of data preparation |
| `02_baselines` | Trains the integration baselines (scVI, scANVI, sysVI, Harmony, PCA, SATURN, CAMEX) | `training/runs/baselines_*`; Figure 3 |
| `03_alpha_master` | α against the protein prior, training dynamics, robustness | Figure 2 and its supplementary figures |
| `04_delta_alpha_biology`, `04_delta_alpha_biology_simpler` | Δα by gene category (detailed and broad categories) | Figure 4 and its supplementary figures |
| `05_esm_alpha_geometry` | Δα over the protein-language space | Figure 8 (atlas panels) and supplementary figures |
| `06_pathway_analysis` | Pre-ranked gene-set enrichment of Δα | Figure 4; Hallmark supplementary figure |
| `07_baselines_master` | Integration scores of the seven-method panel | Figure 3 and its supplementary figures |
| `08_chimera_umap` | UMAPs by method | supplementary UMAP figures |
| `08_ortholog_qc` | Ortholog pairs whose Δα disagrees with their identity | residue-audit supplementary figure |
| `08_alpha_integration_confidence` | α as per-gene integration confidence by cell type | no figure of the paper was traced to it |
| `09_gin_attention_overall` | Attribution of the gate network to its inputs | Figure 8a, b and supplementary figures |
| `09_protein_structure_visualization` | Selects genes and caches structures for notebook 10 | `notebooks/cache/` |
| `10_gin_attribution_protein_structure`, `10b_gin_attribution_figures` | Residue-level attribution and protein cards | residue supplementary figures |
| `11_gene_archetypes` | Gene archetypes from the gate network's hidden state | Figure 8 (archetype panels) and supplementary figures |

### 4.3 Per-gene values

- **One run.** `training/runs/<run>/final_state.npz` (load with `allow_pickle=False`) holds
  `alpha`, `prior`, `correction`, `S_g_ema` per gene; `delta_alpha = alpha - prior`. The file has
  no gene names: take `var_names` from the cohort object and map them with the cohort's own
  ortholog table ([`analysis/common/gene_ids.py`](analysis/common/gene_ids.py)). Testis runs were
  trained with the prior clamped at 0.85 (`prior` is the clamped one); the unclamped prior is
  `conservation_prior_raw.npy` in the run folder.
- **Seed consensus of the seven canonical cohorts.** `results/gene_tables/<cohort>__consensus.csv`:
  means over the seeds listed in `seed_pooling.json` (`used`).
- **All 26 cohorts, both gates, one schema.** `results/analysis/H1/genes/<cohort>.parquet`
  (protein-feature gate; seeds kept in `results/analysis/H1/runs.tsv`) and
  `results/analysis/Z4/h1/genes/<cohort>.parquet` (per-gene gate; seeds in
  `results/analysis/Z4/h1/kept_used.json`); `.tsv.gz` copies in `results/gene_tables/harmonised/`.
  Columns are defined in [`analysis/tasks/H1_harmonise/rules.py`](analysis/tasks/H1_harmonise/rules.py).
  Both are recomputed from the archived states by

  ```bash
  python analysis/check_gene_tables.py     # exit 0 when every table of both gates matches its runs
  ```

- **Loading a model.** `ChimeraModel.load(run_dir + "/model", adata=adata, accelerator="cpu")`
  for a protein-feature-gate run. A per-gene-gate run (`z3-*`, `z4-*`) carries one extra
  parameter per gene, which `ChimeraModel.load` refuses; use
  `analysis.common.load_run.load_run(run_dir, adata)`, which loads both kinds.

### 4.4 Retraining

[`configs/canonical_recipes.tsv`](configs/canonical_recipes.tsv) lists, per canonical cohort,
the object, ortholog table, epochs, options, seeds and run folder. All use `--config A` (the
protein-feature gate) and no latent guard.

```bash
PRINT_ONLY=1 bash analysis/retrain_from_recipe.sh liv_hmouse 42      # print the command only
bash analysis/retrain_from_recipe.sh liv_hmouse 42 retrain-liv_hmouse-s42 cpu
CONFIG=M bash analysis/retrain_from_recipe.sh test_hmouse 43         # the per-gene gate with the same recipe
```

A run goes to `training/runs/<run name>/` and is never written over. On 16 CPU threads liver
Human+Mouse takes 20 to 45 minutes, and four-species liver, testis Human+Mouse and three-species
testis about 50 to 55 minutes each. The arguments of every other fit are in the manifest of the
task that launched it (`analysis/tasks/<task>/manifest.tsv`; section 5 names the task of each
run family) and in the run's own `config.json`. Read section 1 before comparing a retrained run
with an archived one.

## 5. Names

- **`panc_*` and `pancreas_*` mean liver.** The liver cohorts come from the CAMEX liver bundle
  and carry `pancreas` in file names (`camex_pancreas_unified.h5ad` is four-species liver). The
  names were kept so that code, caches and run folders stay consistent. The paper says "liver".
- **Gene names.** `var_names` are `ortho_N`: row N of the ortholog table that built the cohort.
  The numbering is local to each table, so the same `ortho_N` names different genes in different
  tissues. Look genes up with `analysis/common/gene_ids.py`, which takes the cohort, and join
  across tissues on `hsapiens_gene_id` only.
- **The two gates.** The protein-feature gate (the default; `--config A`; "canonical" in code and
  file names) and the per-gene gate (`--config M`; "free gate" in code and file names).
  `results/analysis/H1/` holds the first, `results/analysis/Z4/h1/` the second.
- **Readouts in tables.** `delta_alpha` = `alpha - prior`; `D` = Δα centered on the cohort
  median, averaged over seeds; `S_ema` or `S_g_ema` = the count evidence S_g; `pi_raw`,
  `pi_clamped` = the protein prior before and after the clamp of a recipe. Direct baselines keep
  short codes in tables: `B1b` pseudobulk profile correlation over cell types, `B1c` rank-matched
  level per cell type, `B1z` profile correlation over clusters of Chimera's latent, `B2b`
  specificity similarity, `B3a` and `B3b` predictions from gene features alone (all defined in
  the docstring of `analysis/tasks/T4_2_gene_baselines/run.py`); `Sdir` a likelihood margin
  computed without a model and `Lfc` fold change (`analysis/tasks/X4_where_differs/rules.py`).
- **Cohort keys.** Canonical: `liv_hmonkey`, `liv_hmouse`, `liv_hzeb`, `liv_4sp`, `test_hmac`,
  `test_hmouse`, `test_3sp`. Others: `bone_marrow`, `kidney`, `islet`, eye `vz_full` (five
  species), `vz_hmouse`, `vz_hmmul`, `vz_primates`, `vz_hpig`, `vz_hmfas`; cortex `brain_dlpfc`
  (four primates), `brain_dlpfc_hchimp`, `brain_dlpfc_hmac`, `brain_dlpfc_hmac_own`,
  `brain_dlpfc_hmarm`, `brain_m1`, `brain_m1_hmouse`; `lung_healthy`, `muscle_ageing`,
  `tumour_zilionis`. File and folder names keep their spelling (`harmonised/`, `tumour_zilionis`).
- **Result folders.** `results/analysis/<name>/` belongs to the task folder whose name starts
  with `<name>`; the exceptions are listed in [`analysis/README.md`](analysis/README.md).
- **Run folders** (`training/runs/`). The prefix says which family a run belongs to:

| Run name | Family | Launched by | Bundle |
|---|---|---|---|
| names in `configs/canonical_recipes.tsv` (`iter90-150ep-s<seed>`, `xds-*`, `testis-*`) | the 11 canonical runs, trained on Apple MPS | `training/run_experiment.py` | `models_canonical` |
| `rev-<cohort>-s<seed>` | further seeds of the canonical recipes, on CPU | `T3_2_seeds`, `T3_2b_extra_seeds` | `models_seeds_controls_cohorts` |
| `rev-cpu-*`, `rev-rep-*`, `rev-calib-*` | platform repeat, same-seed repeat, timing runs | `T3_1_platform`, `T6_8b_repeat`, `T1_3_smoke` | same |
| `rev-ctl-<cohort>-<arm>-s<seed>` | retraining controls: `rand`, `zesm`, `zexpr` (features), `permA/B/C` (counts), `matchA` (matched cell types), `priorswap`, `nolab` (no labels), `m` (no feature network) | `T3_4`, `T3_5`, `T3_6`, `T6_8`, `T5_4b` | same (`m` runs: `models_per_gene_gate`) |
| `rev-vz-*`, `rev-z1-*` | eye cohort: recipe grids, species subsets, seeds, class-filter build | `T11_2`, `T11_2b`, `T11_2c`, `T11_3`, `T11_8`, `Z1_retune` | same |
| `rev-bm-*`, `rev-bm2-*` | bone marrow: first and second recipe grid, seeds, diagnostics (`rev-bm-dbg-*`) | `T12_2_recipe`, `T12_2b_recipe`, `T12_2b_debug` | same |
| `rev-isl-*`, `rev-kid-*` | healthy islet and kidney models: grids and seeds | `T7_9_recipe`, `T7_9_recipe_b`, `P13_recipe` | same |
| `rev-c1-<cohort>-*` | cortex, lung, skeletal muscle, tumor myeloid cells: grids and seeds | `C1_cohorts` | same |
| `rev-g1-*`, `rev-g1u-*`, `g1-*` | fits with the latent guard | `G1_guard` | same |
| `z3-<cohort>-m-s<seed>`, `z4-<cohort>-m-s<seed>` | per-gene gate | `Z3_free_gate`, `Z4_free_gate_all` | `models_per_gene_gate` (kept seeds) |
| `o2-*` | models on the identifier-joined objects, with controls | `O1_orthologs` | `data_identifier_joined` |
| `b1-*`, `b1free-*`, `b1hybrid-*`, `b1trackd-*`, `b2-*` | benchmark fits on altered or perturbed objects (`free`: per-gene gate; `hybrid`: feature network plus a per-gene bias) | `B1_ground_truth`, `B2_benchmark` | `gene_level_benchmark` (no models) |
| `baselines_<method>/` | latents of the integration baselines | `notebooks/02_baselines.ipynb` | `notebook_caches` |
| `d1-*`, `z2-*`, `o1-*` | fits on thinned or held-out objects, embedding variants, a build check | `D1_sparse_genes`, `Z2_embedding`, `O1_orthologs` | not deposited; their tables are |
| anything else | runs from the development of the model | - | not deposited |

  The prefix `rev-` marks runs trained on CPU from a manifest of `analysis/tasks/`.

## 6. Smoke test

```bash
bash analysis/reproduce_smoke.sh                     # inputs, tests, Δα tables, gene identity, per-gene tables
RUN_CHECKPOINT=1 bash analysis/reproduce_smoke.sh    # also reload one model on CPU
RUN_FIGURE=1 bash analysis/reproduce_smoke.sh        # also execute notebook 03_alpha_master
```

The script exits 0 only if every check passes, and says which check failed otherwise. It needs
`data_liver_testis` and `models_canonical`. With `gene_tables` unpacked it recomputes every
column of the seven consensus tables from the archived states; with `analysis_results` it
recomputes the per-gene tables of all 26 cohorts for both gates (these two steps also need
`models_seeds_controls_cohorts` and `models_per_gene_gate`). `RUN_FIGURE=1` executes the notebook
with the `python3` kernel of the active environment (`KERNEL=<name>` to change it) and writes its
panels into `notebooks/figures/alpha_master/`, so `git status` shows which panels were
re-rendered.

## 7. Release test

A release is tested from a clean clone: a fresh `git clone`, an environment built from the lock
file and the staged bundles are put together in an empty folder, and the smoke test and further
checks run there while the authors' working tree is hidden from them.

```bash
python release/zenodo/stage.py --check          # is the staged deposit current?
bash release/clean_clone_test.sh --fresh all    # exit 0: the release reproduces from clone and deposit alone
```

Steps, options and how to read the optional retrain comparison:
[`release/clean_clone/README.md`](release/clean_clone/README.md). How the deposit is built:
[`release/zenodo/README.md`](release/zenodo/README.md).

## 8. From raw data to the processed objects

- **Liver and testis.** [`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)
  builds the objects with the functions of `chimera/_data_utils.py` from the CAMEX bundle
  (figshare doi:10.6084/m9.figshare.31131808.v1) and the SATURN ESM-2 embeddings. Use the
  archived ortholog tables instead of querying BioMart again: they are from Ensembl release 110,
  and if the release-110 archive does not answer, the query falls back to the current release
  without an error.
- **Liver and testis with counts joined on the Ensembl gene ID** (recommended for new work):
  `python -m analysis.tasks.O1_orthologs.repair --tissue liver --mode repair` writes
  `data/analysis/O1/repair_liver/`, with `repair_log.csv` listing each changed column. The models
  and tables of the paper were trained on the objects in `data/processed/`; the models trained on
  the identifier-joined objects are `training/runs/o2-*`.
- **Other cohorts.** Each has a build script in its task folder (`T11_1_vanzyl`,
  `T12_1_bone_marrow`, `P13_kidney`, `T7_9_islet`, `T7_2_disease_objects`, `C1_cohorts`); every
  download is recorded in a `MANIFEST.tsv` under `data/analysis/`. Check a new object before
  training with `python -m analysis.common.check_cohort <h5ad>`.
