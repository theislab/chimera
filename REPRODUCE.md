# Reproducing the Chimera results

This is the one guide from the paper to the code and data behind it. It covers what can be
reproduced and at what cost, the environment, the archived deposit, a map from every figure,
table and section of the paper to the code, the names used in files and folders, and the two
tests of a release. It describes release `v0.1.0`: the code at that tag is archived at
[10.5281/zenodo.24137767](https://doi.org/10.5281/zenodo.24137767), and section 3 gives the DOI
of every record of the deposit ([`release/README.md`](release/README.md) tells how a release is
made).

For the model itself and for training on your own cohort, see [`README.md`](README.md). For
where things are in the repository, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 1. What can be reproduced, and at what cost

| Level | What you do | Needs | Cost |
|---|---|---|---|
| Read the results | Open the archived tables: per-gene tables, result tables of every analysis | bundles `gene_tables`, `analysis_results` | minutes; no code |
| Check the release | Run the smoke test: the test suite, the per-gene tables recomputed from the archived model states, one model reloaded | the environment; the data and model bundles | about 5 minutes on 4 CPU threads |
| Recompose the figures | Build every figure from its panel files | the repository only (the panel files are tracked; [`figures/INDEX.md`](figures/INDEX.md) lists them per figure) | minutes ([`figures/README.md`](figures/README.md)) |
| Redraw a panel | Run the script that draws it ([`figures/producers.tsv`](figures/producers.tsv): panel file, script, command) | the environment; the files its docstring names (section 3 gives the bundle of each path; the map of section 4.1 names inputs and bundles of the main figures) | seconds to minutes |
| Re-run an analysis | Run a task of `analysis/tasks/` on the archived models and tables | the environment; the bundles the task reads (its README) | seconds to hours per task; a few tasks need a cluster |
| Re-run a notebook | Execute a workflow notebook on the archived runs and caches | the environment; `data_liver_testis`, `models_canonical`, `notebook_caches` | minutes to an hour per notebook ([`notebooks/README.md`](notebooks/README.md)) |
| Retrain a model | Train a recipe again | the environment; a data bundle | 20 to 55 minutes per canonical cohort on 16 CPU threads |
| Rebuild a cohort object | Download the raw data and run the cohort's build (section 8) | the environment; network; up to about 20 GB of downloads per tissue | minutes to hours per tissue |

No number of the paper needs a model to be trained again: every analysis reads the archived
runs.

**The chain, step by step.** The deposit holds the outputs of steps 3 to 6, so the chain can be
entered at any step; raw data (step 2) stay with their sources.

| Step | Where it is described | Entry command | Reads | Writes |
|---|---|---|---|---|
| 1. Environments | section 2 | `bash analysis/envs/build_env_from_lock.sh chimera_env` | lock file | `chimera_env`; the other environments as section 2 says |
| 2. Raw data | section 8 | `notebooks/01_data_preparation.ipynb` (liver, testis, protein embeddings); `bash analysis/tasks/<task>/fetch*.sh $PWD` | public sources | `data/raw/` (notebook `01`); `data/analysis/<cohort>/raw/`, with a `MANIFEST.tsv` per folder |
| 3. Cohort objects | section 8 | `notebooks/01_data_preparation.ipynb`; `python -m analysis.tasks.<task>.build*` | step 2 | `data/processed/*.h5ad` (liver, testis), `data/analysis/<cohort>/*.h5ad` (all others), ortholog tables |
| 4. Models | section 4.4, section 5 (run families) | `bash analysis/retrain_from_recipe.sh <cohort> <seed>`; a `manifest.tsv` of a task through `analysis/slurm/array.sbatch` | step 3 | `training/runs/<run>/` |
| 5. Per-gene tables | section 4.3 | `python -m analysis.tasks.T3_3_consensus.run`; `bash analysis/tasks/H1_harmonise/run_all.sh`; `python -m analysis.tasks.Z4_free_gate_all.readouts` | step 4 | `results/analysis/T3_3/`, `H1/genes/`, `Z4/h1/genes/` (in the deposit also as `results/gene_tables/`) |
| 6. Analyses | section 4.1, then [`analysis/README.md`](analysis/README.md) and the task README | the task's run steps | steps 3 to 5 | `results/analysis/<task>/` |
| 7. Panels and figures | [`figures/README.md`](figures/README.md), [`figures/INDEX.md`](figures/INDEX.md), [`figures/producers.tsv`](figures/producers.tsv) | the command of the panel in `figures/producers.tsv` (a script of a task folder or of `figures/redraw/`), then `bash figures/build.sh` | steps 4 to 6 | `figures/panels/`, `figures/figures.pdf` |
| 8. Deposit and release test | section 7, [`release/README.md`](release/README.md) | `python release/zenodo/stage.py`; `bash release/clean_clone_test.sh --fresh all` | steps 3 to 6 | the staged bundles; a pass or fail |

The manuscript is not in this repository; it includes `figures/figures.pdf` page by page
([`figures/README.md`](figures/README.md), "The paper").

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
  poorly mixed state. Two repeats of liver Human+Mouse with unchanged seed agreed with their
  originals at 0.11 and 0.63 on Δα, and at 0.10 and 0.65 on the correction
  (`results/analysis/T6_8b_repeat.csv`).
- The paper therefore reads seed-consensus tables, and the archived runs are the reference.
  Before comparing a retrained run with an archived one, check its training state in
  `summary.json` (`adv_acc_final`, `failed_expectations`).
- Re-rendered panels carry the same values as the archived ones but are not pixel-identical on
  another platform (fonts, antialiasing).
- Three notebooks read live services. The sequences and structures that notebooks 09 and 10
  fetched (Ensembl REST, UniProt, AlphaFold DB) are frozen in the `notebook_caches` bundle. The
  enrichment tables of notebook 06 are frozen in the run folders of `models_canonical`; its two
  Enrichr gene-set libraries are fetched into the gseapy cache of the home folder when they are
  absent. Without the frozen files a run fetches current versions, which may differ.

## 2. Code and environment

```bash
git clone --branch v0.1.0 https://github.com/theislab/chimera.git
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

A few analyses need a second environment. Each task README names its environment.

| Environment | For | Built by |
|---|---|---|
| `revision_de_env` | pseudobulk differential expression (`pydeseq2`) | `mamba env create -f analysis/envs/revision_de_env.yml` |
| `saturn_env`, `camex_env` | SATURN and CAMEX | `bash analysis/tasks/T5_1_envs/build_envs.sh` |
| `samap_env`, `genespectra_env`, `cacimar_env` | the other compared tools | `bash analysis/tasks/B2_benchmark/tools/build_envs.sh` |
| `chimera_figures_env` | composing the figures | `mamba env create -f figures/environment.yml` |

The environment files are in [`analysis/envs/`](analysis/envs/); the tools, their pinned commits
and where they are cloned are in
[`analysis/external_tools/README.md`](analysis/external_tools/README.md).

## 3. The deposit

Data, models and results are not in git (`data/`, `training/runs/`, `notebooks/cache/` and
`results/` are ignored). They are deposited at Zenodo as bundles, one record each; the code
archive [10.5281/zenodo.24137767](https://doi.org/10.5281/zenodo.24137767) links every record,
and the table below gives each record's DOI. A record holds `chimera_<bundle>.tar.gz`,
`MANIFEST.tsv` (every member with its size and sha256), `SHA256SUMS`, `README.md` and
`LICENSES.md`. Member paths are relative to the repository root, so every bundle is unpacked
there:

```bash
sha256sum -c SHA256SUMS
tar -xzf chimera_<bundle>.tar.gz -C /path/to/chimera
```

| Bundle | DOI | Unpacks to | Content |
|---|---|---|---|
| `data_liver_testis` | [10.5281/zenodo.24137733](https://doi.org/10.5281/zenodo.24137733) | `data/processed/`, `data/raw/MANIFEST.tsv` | The seven processed AnnData objects of the canonical cohorts (liver from the CAMEX bundle of Guilliams et al. 2022 and Wang et al. 2023; testis from Murat et al. 2023), with their build configurations, the Ensembl release 110 one-to-one ortholog tables for liver and testis, and the GIN feature scalers. camex_pancreas_unified.h5ad is the liver 4-species cohort (historical file name). data/raw/MANIFEST.tsv lists the raw files that the data preparation downloads (the CAMEX files, the SATURN protein embeddings, the eye count matrices) with address, date, size, MD5 and sha256; the raw files stay with their sources. |
| `data_other_cohorts` | [10.5281/zenodo.24137739](https://doi.org/10.5281/zenodo.24137739) | `data/analysis/`, `data/processed/ortholog_table_van_zyl.csv` | Processed objects built from public data: the eye cohort (Van Zyl aqueous-humor outflow tissues, GEO) with its five species subsets, the class-filter sensitivity build and the five-species Ensembl release 110 ortholog table that its build reads; bone marrow human, mouse lemur and mouse (Tabula Sapiens, Tabula Microcebus, Tabula Muris Senis via CELLxGENE); kidney (KPMP, CELLxGENE) and islet (CELLxGENE) healthy training objects and disease pseudobulks; cortex M1 and DLPFC, lung, skeletal muscle and tumor myeloid training objects (CELLxGENE, GEO) with their ortholog tables and feature scalers; liver MASH pseudobulks (GEO GSE202379, GSE192742) and cell annotations; the count and feature control objects; and the download manifests (URL, date, sha256) of every third-party input that is linked rather than redistributed. |
| `data_identifier_joined` | [10.5281/zenodo.24137741](https://doi.org/10.5281/zenodo.24137741) | `data/analysis/O1/repair_*/`, `training/runs/o2-*/` | Liver (Human+Mouse, Human+Monkey, 4-species) and testis (Human+Mouse) objects in which every count column is taken from the raw row that carries the Ensembl gene ID of the ortholog table (repair_log.csv per tissue lists each column that differs from the symbol-joined object; build reports and ortholog tables are beside the objects), and the Chimera models trained on them with the canonical recipes (seeds 42-44; liver Human+Monkey seeds 42-51 with a matched control on the symbol-joined object). These are the recommended objects for new work on these cohorts. Liver macaque is identifier-joined only where its matrix carries Ensembl IDs; zebrafish carries none and is unchanged. |
| `models_canonical` | [10.5281/zenodo.24137745](https://doi.org/10.5281/zenodo.24137745) | `training/runs/<run>/` (the runs of `configs/canonical_recipes.tsv`) | The 11 canonical Chimera runs (liver Human+Monkey seeds 42-44, Human+Mouse seeds 42-44, Human+Zebrafish, 4-species; testis Human+Macaque, Human+Mouse, 3-species), complete run folders: model/model.pt, final_state.npz (alpha, prior, correction, S_g, latent means), config.json, summary.json, per-epoch alpha snapshots, training metrics, tables and figure caches. Trained on Apple MPS; the recipe of each run is in configs/canonical_recipes.tsv of the code release. |
| `models_seeds_controls_cohorts` | [10.5281/zenodo.24137747](https://doi.org/10.5281/zenodo.24137747) | `training/runs/rev-*/`, `training/runs/g1-*/` | Protein-feature-gate runs beyond the 11 canonical ones: further seeds of the canonical recipes (liver 4-species seeds 43-46, liver Human+Mouse seeds 45-51, testis 3-species seeds 43-46, testis Human+Mouse seeds 43-51), CPU platform and same-seed repeat runs, the control runs (feature destruction, count permutation, matched cell types, prior swap, testis without labels), the recipe grids and seeds of every further cohort (bone marrow, eye and its species subsets, islet, kidney, cortex M1 and DLPFC, lung, skeletal muscle, tumor myeloid cells), and the runs with the latent guard --clamp-z-logvar (rev-g1-*). Per run: model/, final_state.npz, config.json, summary.json, run_summary.json, report.md, provenance and the training metrics. For the latent-guard runs on thinned objects (g1-*): configuration, summaries and final state. CPU-trained. |
| `models_per_gene_gate` | [10.5281/zenodo.24137749](https://doi.org/10.5281/zenodo.24137749) | `training/runs/z3-*/`, `z4-*/`, `rev-ctl-liv_hmouse-m-*/` | Fits with a per-gene gate (--config M: the canonical model with one free correction per gene in place of the protein-feature network, recipe otherwise unchanged): every fit of liver Human+Mouse, liver 4-species and testis Human+Mouse, and the seeds kept by the health rule on each of the 26 harmonized cohorts (results/analysis/Z4/kept.json). Per run: model/, final_state.npz, config.json, summary.json, run_summary.json, report.md, provenance and the training metrics. CPU-trained. |
| `gene_tables` | [10.5281/zenodo.24137751](https://doi.org/10.5281/zenodo.24137751) | `results/gene_tables/` | One CSV per canonical cohort with the human Ensembl ID, symbol, prior, raw prior, alpha, correction, delta-alpha, the seed-consensus readout D, S_g and sign consistency, pooled over seeds; seed reliability and seed-pair agreement. Per-gene tables of the 26 harmonized cohorts in one schema, for the protein-feature gate and for the per-gene gate, with count-support, strict-support and naming flags. README.md inside describes every column. |
| `notebook_caches` | [10.5281/zenodo.24137755](https://doi.org/10.5281/zenodo.24137755) | `notebooks/cache/`, `training/runs/baselines_*` | Caches of the workflow notebooks, read by the notebooks and the figure code: integration score tables of the compared methods, UMAP coordinates, integration-confidence tables, the ortholog quality-control summary, residue attributions of the gate network with their ESM-2 embeddings, and the stored responses of live services (Ensembl REST sequences, UniProt accessions, AlphaFold DB structures); and the integration baseline runs (scVI, scANVI, sysVI, Harmony, PCA, SATURN, CAMEX). |
| `analysis_results` | [10.5281/zenodo.24137757](https://doi.org/10.5281/zenodo.24137757) | `results/analysis/<task>/` | Outputs of the analyses of the paper, one folder per task under results/analysis/: reliability and consensus, controls, gold standards and baselines, scoreboard, functional layer, disease cases, further cohorts, residue checks, the harmonized per-gene tables of 26 cohorts for both gates (H1, Z4) and the gate comparisons (Z3, Z4), the identifier-joined count comparison (O1), the latent-guard checks (G1), the gate gap (X8), integration with the per-gene gate (I1), the exploratory analyses (program atlas, protein and gene architecture, cell types and lineages, translational, regulatory architecture, literature species differences: E1-E6) and the figure sources. Most task folders are deposited as tables, text and figures without binary intermediates; permutation and leave-one-out differential-expression draws are not included. Per-gene EVEE values (license unstated) are not included; they are rebuilt from the archived EVEE download. |
| `tool_runs_saturn_camex` | [10.5281/zenodo.24137761](https://doi.org/10.5281/zenodo.24137761) | `results/analysis/T5_*/` | SATURN and CAMEX runs (environment smoke test, pilot, cell-level runs on the seven canonical cohorts, gene-level runs with three seeds): outputs, embeddings, gene-level scalars and logs. SATURN macrogene-centroid initializations and pre-training snapshots are not included. |
| `gene_level_benchmark` | [10.5281/zenodo.24137763](https://doi.org/10.5281/zenodo.24137763) | `results/analysis/B1/`, `B2/`, `training/runs/b1*-*/`, `b2-*/` | The gene-level benchmark: Track A (ground-truth spike-in of pattern, level and loss changes, and null alterations, on liver, testis and kidney Human+Mouse), Track B (robustness to perturbed cells, labels and depth), Track C (independent data across tissues), Track D (run time and memory) and the overview, for Chimera (both gates) and the compared tools: scored tables, truth sets, health records, logs, overview tables and figures, and configuration, summary and final state of every scored Chimera fit. Altered objects and the models of the altered fits are rebuilt by the task code. |
| `analysis_plan` | [10.5281/zenodo.24137729](https://doi.org/10.5281/zenodo.24137729) | `results/analysis/_release/` | The pre-specified analysis plan of the study as frozen on 1 November 2026, before any association between a Chimera readout and an independent gold standard was computed, with the sections added afterwards and the dated amendment log. |

The table is printed by `python release/zenodo/stage.py --bundle-table`; the declaration of the
deposit, with the release tag and the DOIs, is
[`release/zenodo/bundles.yaml`](release/zenodo/bundles.yaml).

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
on them are rebuilt by the task code. Our own files in the records are released under
CC BY 4.0; third-party sources keep their terms, listed in
[`release/zenodo/LICENSES.md`](release/zenodo/LICENSES.md).

## 4. From the paper to the code and data

Three steps lead from an item of the paper to its source.

1. **Find the item in the map below** (by figure, table or section number; the LaTeX label is
   the stable name). The map names the task folders of [`analysis/tasks/`](analysis/README.md)
   and the notebooks behind it.
2. **Open the task folder's `README.md`**: what the task does, the commands in run order, its
   environment, what it reads, where it writes, and the deposit bundle of each output.
3. **For a figure, open [`figures/INDEX.md`](figures/INDEX.md)**: page in
   [`figures/figures.pdf`](figures/figures.pdf), the panel files the figure is composed from, the
   script that draws each panel (also in [`figures/producers.tsv`](figures/producers.tsv), with
   its command), and its spec. `python -m figures.compose <key>` composes one figure;
   `bash figures/build.sh` builds all ([`figures/README.md`](figures/README.md)).

A number in the running text comes from the result folders of the tasks listed for its section;
the tables are CSV or parquet files with one row per test or gene.

### 4.1 Map

Figure, table and section numbers are those of the current manuscript build; labels do not
change when numbers do. Panel letters are given in parentheses where a task feeds single panels.
"Other code" names notebooks and scripts outside `analysis/tasks/`.

<!-- paper-map:begin (generated by analysis/make_index.py; do not edit by hand) -->
| In the paper | Label | What | Task folders behind it | Other code behind it |
|---|---|---|---|---|
| Figure 1 | `fig:overview`; figure key `main_overview` | Overview: what the model takes, does and returns | - | `figures/redraw/overview/overview.py` (drawn, not computed); inputs: its tracked `assets/`, made by `build_assets.py` from the liver Human+Monkey run (bundle `models_canonical`), its cached UMAP (bundle `notebook_caches`) and `results/analysis/H1/axis.csv` (bundle `analysis_results`) |
| Section 2 | `sec:methods` | Methods | - | `chimera/` (the model) |
| Section 3.1 | `sec:model` | One fit aligns the cells and reads each gene against its protein | `T5_3_runs`, `T5_5_benchmark`, `Z2_embedding`, `B2_benchmark`, `C1_cohorts` | `notebooks/07_baselines_master.ipynb`; `notebooks/08_chimera_umap.ipynb` |
| Figure 2 | `fig:alpha-intro`; figure key `main_alpha_vs_prior` | The gate value α against the protein prior | - | panels: `figures/redraw/main_figures.py` (`scatter`, `shift`, `distribution`, `training`); inputs: `final_state.npz` and, for panel d, `alpha_snapshots/alpha_epoch_*.npz` of the liver Human+Monkey run (bundle `models_canonical`); first analysis: `notebooks/03_alpha_master.ipynb` |
| Figure 3 | `fig:integration`; figure key `main_integration` | Composite integration score of ten methods on eleven cohorts | `T5_3_runs`, `T5_5_benchmark`, `B2_benchmark`, `T8_4_figures` | panel: `analysis/tasks/T8_4_figures/fig3_integration_ext.py` (`composite_strip`); input: `results/analysis/B2/integration/master_table_extended.csv` (bundle `gene_level_benchmark`); the scores of seven of the ten methods come from `notebooks/02_baselines.ipynb` (latents) and `notebooks/07_baselines_master.ipynb` (scores) (bundle `notebook_caches`) |
| Section 3.2 | `sec:axis` | Deployment is an axis of its own, beside protein sequence | `T2_3_tissue`, `X5_sequence_axis`, `X4_where_differs`, `H1_harmonise` | - |
| Figure 4 | `fig:dalpha-biology`; figure key `main_delta_alpha_liver` | Δα by gene category and across the protein prior, four-species liver | `T8_4_figures` (c) | panels a, b: `figures/redraw/main_figures.py` (`category_boxes`, `category_effect_sizes`); panel c: `analysis/tasks/T8_4_figures/fig4_additions.py` (`panel_landscape`); inputs: `final_state.npz` of the four-species liver run (bundle `models_canonical`), the gene names of its data object (bundle `data_liver_testis`), `results/analysis/T2_5/liv_4sp__genes.parquet` and, for the three genes named in bold, `results/analysis/X2/named_genes.csv` (bundle `analysis_results`); first analysis: `notebooks/04_delta_alpha_biology.ipynb`, `notebooks/05_esm_alpha_geometry.ipynb` |
| Section 3.3 | `sec:findings` | The axis reveals less conserved proteins used alike and conserved proteins used differently | `T2_5_celltype_margin`, `T2_6_matched`, `X1_testis_discovery`, `X2_liver_biology`, `X4_where_differs`, `Y3_findings`, `Y4_regions`, `Y5_aspects`, `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble`, `T3_6_matched`, `C1_cohorts` | - |
| Figure 5 | `fig:axis-tissues`; figure key `main_axis_tissues` | Protein identity, dN/dS, count evidence and Δα in eight tissues | `H1_harmonise` (a, b) | panels: `analysis/tasks/H1_harmonise/figures.py` (`main_axis_summary`, `main_axis_clouds`); inputs: `results/analysis/H1/axis.csv`, `results/analysis/H1/genes/<cohort>.parquet` (bundle `analysis_results`) |
| Section 3.4 | `sec:validation` | Independent data back the count evidence on cell-state pattern; Δα passes on liver pattern | `T4_2_gene_baselines`, `T4_3_scoreboard`, `X6_adds_beyond` | - |
| Section 3.4 | `sec:per-gene` | Read gene by gene from counts and evaluated afterward, Δα agrees with independent data in liver and testis | `Z3_free_gate`, `F2_per_gene`, `I1_free_gate_integration`, `X8_gate_divergence` | - |
| Figure 6 | `fig:validation`; figure key `main_validation` | Two worked genes, the cell types behind the count evidence, and the readouts against independent data | `T2_5_celltype_margin` (c), `Y3_findings` (a, b), `Z3_free_gate` (d, e), `F2_per_gene` (d, e), `T8_4_figures` (c) | panels a, b: `analysis/tasks/Y3_findings/figures.py`; c: `analysis/tasks/T8_4_figures/fig4_additions.py` (`panel_drivers_enzymes`); d, e: `analysis/tasks/F2_per_gene/panels.py` (`panel_readouts`, `panel_named_compact`); inputs: `results/analysis/Y3/worked_pseudobulk.csv`, `worked_readouts.csv`; `results/analysis/T2_5/liv_4sp__C_gk.parquet`; `results/analysis/Z3/compare_hyp.csv`, `compare_y3.csv` (bundle `analysis_results`) |
| Section 3.4 | `sec:adds` | The count evidence adds to direct baselines on pattern and needs no labels in liver | `T5_6_gene_level`, `X6_adds_beyond`, `B1_ground_truth`, `B2_benchmark`, `E6_known_differences` | - |
| Section 3.4 | `sec:reliability` | Reliability and feature dependence (paragraph) | `T3_3_consensus`, `T2_4_covariates`, `T3_4_destroy_features`, `T3_5_count_perm`, `D1_sparse_genes` | - |
| Section 3.5 | `sec:tissues` | One reading across eight tissues | `T11_1_vanzyl`, `T12_1_bone_marrow`, `T11_4_vz_integration`, `Z4_free_gate_all`, `H1_harmonise`, `C1_cohorts`, `Y1_gene_tissue`, `E1_atlas`, `E3_cells`, `E2_protein`, `E5_regulatory`, `T11_5_vz_genes`, `T11_6_vz_glaucoma`, `T11_7_vz_figure`, `T12_3_tests`, `E4_translational` | - |
| Figure 7 | `fig:one-reading`; figure key `main_tissues` | Gene programs and classes across eight tissues | `H1_harmonise` | panel: `analysis/tasks/H1_harmonise/figures.py` (`main_recurrence_compact`); inputs: `results/analysis/H1/recurrence.csv`, `tissue_specific.csv` (bundle `analysis_results`) |
| Figure 8 | `fig:above-prior`; figure key `main_specialized_cells` | Home compartments of genes: Δα under both gates, profile correlation and level similarity | `E3_cells` | panels: `analysis/tasks/E3_cells/figure.py` (`main_compartments`, `main_pattern_vs_level`); input: `results/analysis/E3/excess.csv` (bundle `analysis_results`) |
| Section 3.6 | `sec:protein` | The protein-feature gate maps deployment over protein space | `X3_cross_cohort`, `T8_1_residue` | `notebooks/09_gin_attention_overall.ipynb`; `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb`; `notebooks/10_gin_attribution_protein_structure.ipynb` |
| Figure 9 | `fig:drug-targets`; figure key `main_drug_targets` | Approved drug targets against the protein prior | `E4_translational` | panel: `analysis/tasks/E4_translational/figures.py` (`p_targets`); inputs: `results/analysis/E4/tables/cohorts.parquet`, `results/analysis/E4/annot/targets.parquet` and, for the printed counts, `results/analysis/E4/q23_callshare.csv` and `callshare_perm.csv` (bundle `analysis_results`) |
| Figure 10 | `fig:gin-esm2`; figure key `main_protein` | Protein-space atlas | - | panels: `figures/redraw/main_figures.py` (`atlas_alpha`, `atlas_delta_alpha`, `atlas_correction`, `atlas_categories`, `atlas_categories_key`, `atlas_clusters`); inputs: `final_state.npz` of the four liver runs and `cache/esm2_umap.npy` of the four-species liver run (bundle `models_canonical`), the ESM-2 components stored with the data objects (bundle `data_liver_testis`); first analysis: `notebooks/05_esm_alpha_geometry.ipynb` |
| Table 1 | `tab:when-to-use` | When to use which readout or method | `X6_adds_beyond`, `B1_ground_truth`, `B2_benchmark` | - |
| Supplementary Table S1 | `tab:tool-inclusion` | Tools considered for comparison and why each was included or not | - | text of the manuscript (no computed values) |
| Supplementary Table S2 | `tab:related-work-comparison` | Comparison with related methods | - | text of the manuscript (no computed values) |
| Supplementary Table S3 | `tab:assumptions-summary` | Summary of assumptions and limitations | `T2_7_consistency` | text of the manuscript (no computed values) |
| Supplementary Section C | `sec:extended-methods` | Extended Methods | `O1_orthologs` | `chimera/`, `training/` |
| Supplementary Table S4 | `tab:notation` | Notation | - | text of the manuscript (no computed values) |
| Supplementary Table S5 | `tab:notation-continued` | Notation (continued) | - | text of the manuscript (no computed values) |
| Supplementary Section C.8.4 | `sssec:per-gene-gate` | The per-gene gate | `Z3_free_gate` | - |
| Supplementary Section C.12.2 | `par:latent-guard` | Latent guard (paragraph) | `G1_guard`, `T12_2b_debug` | - |
| Supplementary Section C.15.8 | `sssec:rev-residue` | Residue specificity control | `T8_1_residue` | - |
| Supplementary Section C.19.2 | `sssec:rev-benchmark` | Integration benchmark with SATURN and CAMEX, and the testis label ablation | `T5_1_envs`, `T5_2_adapters`, `T5_3_runs`, `T5_5_benchmark`, `T5_4b_label_ablation` | - |
| Supplementary Section C.21 | `ssec:revision-analyses` | Pre-specified Analysis Plan and Departures | - | - |
| Supplementary Section C.21.1 | `sssec:rev-plan` | Pre-specified analysis plan | `Z1_retune`, `T11_2_recipe`, `T11_2b_vz_subsets`, `T11_2c_count_rule`, `T12_2_recipe`, `T12_2b_recipe`, `T7_9_recipe`, `T7_9_recipe_b`, `P13_recipe` | - |
| Supplementary Section C.21.2 | `sssec:rev-seeds` | Seeds, seed pooling and training states | `T3_1_platform`, `T3_2_seeds`, `T3_2b_extra_seeds`, `T3_1b_integration_gate`, `T6_8b_repeat`, `T3_3_consensus` | - |
| Supplementary Section C.21.3 | `sssec:rev-identity` | Gene identity and tissue comparison | `T2_2_regen`, `T2_3_tissue`, `T1_3_smoke` | - |
| Supplementary Section C.21.4 | `sssec:rev-covariates` | Covariate audit, masking and margin support | `T2_4_covariates` | - |
| Supplementary Section C.21.5 | `sssec:rev-decomposition` | Cell-type decomposition of the count-evidence margin | `T2_5_celltype_margin` | - |
| Supplementary Section C.21.6 | `sssec:rev-categories` | Category tests under matching | `T2_6_matched` | - |
| Supplementary Section C.21.7 | `sssec:rev-controls` | Retraining controls | `T3_4_destroy_features`, `T3_5_count_perm`, `T3_6_matched` | - |
| Supplementary Section C.21.8 | `sssec:rev-validation` | Independent validation and direct baselines | `T4_1_gold`, `T4_2_gene_baselines`, `T4_3_scoreboard` | - |
| Supplementary Section C.21.9 | `sssec:rev-gene-level` | Gene-level comparison with SATURN and CAMEX | `T5_6_gene_level` | - |
| Supplementary Section C.21.10 | `sssec:rev-functional` | Functional layer and prior swap | `T6_1_functional`, `T6_8b_repeat`, `T6_common`, `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble`, `T6_8_priorswap` | - |
| Supplementary Section C.21.11 | `sssec:rev-exploratory` | Exploratory analyses | `X1_testis_discovery`, `X2_liver_biology`, `X4_where_differs`, `Y3_findings`, `X6_adds_beyond`, `X7_architecture_outputs` | - |
| Supplementary Section C.21.12 | `sssec:rev-dataprep` | Data preparation of the cohorts beyond liver and testis | `C1_cohorts` | - |
| Supplementary Section C.21.13 | `sssec:rev-disease` | Disease case studies, bone marrow and eye | `T7_1_holdouts`, `T7_2_disease_objects`, `T11_1_vanzyl`, `T12_1_bone_marrow`, `T11_3_vz_seeds`, `T11_8_vz_class_filter`, `T11_5_vz_genes`, `T11_6_vz_glaucoma`, `T12_3_tests`, `T7_3_mash`, `T7_9_islet`, `P13_kidney` | - |
| Supplementary Section C.21.14 | `sssec:rev-gene-benchmark` | Gene-level benchmark | `B1_ground_truth`, `B2_benchmark` | - |
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
| Supplementary Section D.8 | `ssec:er-tissues` | Each tissue adds its own genes to the reading | - | - |
| Supplementary Section D.9 | `ssec:er-e3` | Cell-type programs sit above their protein prior while their levels move | `E3_cells` | - |
| Supplementary Section D.10 | `ssec:er-e2` | Protein architecture and gene history predict where deployment sits relative to protein conservation | `E2_protein` | - |
| Supplementary Section D.11 | `ssec:er-e5` | Regulatory input predicts where deployment sits relative to protein conservation | `E5_regulatory` | - |
| Supplementary Section D.12 | `ssec:er-disease` | Mouse ischemic injury reproduces human kidney-injury programs by stage | `T7_3_mash`, `T7_9_islet`, `P13_kidney` | - |
| Supplementary Section D.12 | `sec:disease` | Program transfer shows which human disease responses a mouse model reproduces | `Z4_free_gate_all`, `C1_cohorts`, `T7_3_mash`, `T7_9_islet`, `P13_kidney` | - |
| Supplementary Section D.13 | `ssec:er-e4` | Drug targets and disease genes sit above their protein prior in their own tissue | `E4_translational` | - |
| Supplementary Section D.14 | `ssec:er-gin` | Chimera's GIN projects count evidence onto protein-language and expression features | - | `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Section D.15 | `ssec:er-atlas` | Chimera learns a protein-space atlas of sequence-relative expression deployment | - | `figures/redraw/atlas.py`; first analysis: `notebooks/05_esm_alpha_geometry.ipynb`, `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Section D.16 | `ssec:er-residue` | Exploratory projection of Δα attribution onto protein structure | `T2_2_regen`, `T8_1_residue` | `figures/redraw/residue.py`, `figures/redraw/residue_audit.py`; first analysis: `notebooks/10_gin_attribution_protein_structure.ipynb`, `notebooks/10b_gin_attribution_figures.ipynb` |
| Supplementary Table S18 | `tab:data-preparation` | Data preparation of every cohort | `T7_2_disease_objects`, `T11_1_vanzyl`, `T12_1_bone_marrow`, `O1_orthologs`, `X2_liver_biology`, `C1_cohorts`, `T7_9_islet`, `P13_kidney` | `notebooks/01_data_preparation.ipynb` (liver and testis) |
| Supplementary Table S19 | `tab:harmonised-cohorts` | The 26 cohorts: cells, genes, recipe, runs and training health | `Z1_retune`, `T11_2_recipe`, `T11_2b_vz_subsets`, `T11_2c_count_rule`, `T12_2_recipe`, `T12_2b_recipe`, `T7_9_recipe`, `T7_9_recipe_b`, `P13_recipe`, `H1_harmonise`, `C1_cohorts` | - |
| Supplementary Table S20 | `tab:harmonised-rules` | Rules of the per-gene tables and of gene naming | `H1_harmonise` | - |
| Supplementary Table S21 | `tab:harmonised-rules-c1` | Rules for the cohorts of cortex, lung, muscle and tumor | `H1_harmonise` | - |
| Supplementary Figure S1 | `fig:supp-main-overview` | Extended caption of Figure 1 (text only) | - | - |
| Supplementary Figure S2 | `fig:supp-main-alpha-intro` | Extended caption of Figure 2 (text only) | - | `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S3 | `fig:supp-alpha-training-dynamics`; figure key `supp_alpha_training` | Training dynamics of α | - | panels: `figures/redraw/alpha.py`; first analysis: `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S4 | `fig:supp-alpha-robustness`; figure key `supp_alpha_robustness` | Program-level Δα across seeds and species sets | `T2_2_regen` (g, h, i, j, k) | panels: `figures/redraw/robustness.py`; first analysis: `notebooks/03_alpha_master.ipynb` |
| Supplementary Figure S5 | `fig:supp-main-integration` | Extended caption of Figure 3 (text only) | - | - |
| Supplementary Figure S6 | `fig:supp-integration-testis-umap`; figure key `supp_integration_umap_testis` | UMAP by method, three-species testis | - | panels: `figures/redraw/integration_umap.py`; stored coordinates: `notebooks/08_chimera_umap.ipynb` |
| Supplementary Figure S7 | `fig:supp-integration-liver-umap`; figure key `supp_integration_umap_liver` | UMAP by method, liver Human+Monkey | - | panels: `figures/redraw/integration_umap.py`; stored coordinates: `notebooks/08_chimera_umap.ipynb` |
| Supplementary Figure S8 | `fig:supp-integration-purity`; figure key `supp_integration_ten_methods` | Integration: mixing and conservation, ten-method panel | `T5_5_benchmark` | panels: `figures/redraw/integration_ten_methods.py` |
| Supplementary Figure S9 | `fig:supp-integration-raw`; figure key `supp_integration_raw_metrics` | Raw integration metrics, ten-method panel | `T5_5_benchmark` | panels: `figures/redraw/integration_ten_methods.py` |
| Supplementary Figure S10 | `fig:supp-integration-ext`; figure key `supp_integration_eleven_cohorts` | Integration benchmark on eleven cohorts | `T11_4_vz_integration`, `B2_benchmark`, `T8_4_figures` | panel script: `analysis/tasks/T8_4_figures/fig3_integration_ext.py` |
| Supplementary Table S22 | `tab:benchmark-label-use` | Use of cell-type labels by each integration method | `T5_4b_label_ablation` | text of the manuscript (no computed values) |
| Supplementary Table S23 | `tab:integration-per-gene-gate` | Integration scores of the per-gene gate, seven canonical cohorts | `I1_free_gate_integration` | - |
| Supplementary Table S24 | `tab:integration-per-gene-gate-harmonised` | Integration scores of the per-gene gate, all cohorts | `I1_free_gate_integration` | - |
| Supplementary Figure S11 | `fig:supp-main-dalpha-biology` | Extended caption of Figure 4 (text only) | - | `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S12 | `fig:supp-dalpha-sources`; figure key `supp_delta_alpha_sources` | Cell-type sources of the count evidence, margin support, sequence measures and tissue comparison | `T2_3_tissue` (d), `X5_sequence_axis` (c), `T2_5_celltype_margin` (a), `T2_4_covariates` (b), `T8_4_figures` | panels a, b, d: `analysis/tasks/T8_4_figures/fig4_additions.py`; panel c: `analysis/tasks/T8_4_figures/fig7_validation.py`; inputs: `results/analysis/T2_5/`, `results/analysis/X2/named_genes.csv` (the three genes in bold of panel a), `results/analysis/T2_3/tissue_comparison.csv`, `results/analysis/H1/`, `results/analysis/X5/` (bundle `analysis_results`), `final_state.npz` of the first kept run of the cohorts beyond liver and testis (bundle `models_seeds_controls_cohorts`) |
| Supplementary Figure S13 | `fig:supp-main-axis-tissues` | Extended caption of Figure 5 (text only) | - | - |
| Supplementary Figure S14 | `fig:supp-sequence-axis`; figure key `supp_sequence_axis` | Protein identity, dN/dS and Δα gene by gene | `X5_sequence_axis` | panel script: `analysis/tasks/X5_sequence_axis/run.py` |
| Supplementary Figure S15 | `fig:supp-known-measures`; figure key `supp_known_measures` | Δα and count evidence among known gene measures | `X5_sequence_axis` | panel script: `analysis/tasks/X5_sequence_axis/known.py` |
| Supplementary Figure S16 | `fig:supp-relations-heatmap`; figure key `supp_relations_heatmap` | Relations of Δα and count evidence to sequence, expression and direct baselines | `X4_where_differs` | panel script: `analysis/tasks/X4_where_differs/relations.py` |
| Supplementary Figure S17 | `fig:supp-relations-density`; figure key `supp_relations_density` | The same relations gene by gene | `X4_where_differs` | panel script: `analysis/tasks/X4_where_differs/relations.py` |
| Supplementary Figure S18 | `fig:supp-dalpha-broad-categories`; figure key `supp_categories_broad` | Δα by broad gene category, liver and testis | `T2_2_regen` (d) | panels: `figures/redraw/categories_broad.py`; first analysis: `notebooks/04_delta_alpha_biology_simpler.ipynb` |
| Supplementary Figure S19 | `fig:supp-dalpha-liver-detailed`; figure key `supp_categories_liver` | Δα by gene category, four-species liver, all 38 categories | - | panels: `figures/redraw/categories.py`; first analysis: `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S20 | `fig:supp-dalpha-testis-detailed`; figure key `supp_categories_testis` | Δα by gene category, testis Human+Macaque | `T2_2_regen` | panels: `figures/redraw/categories.py`; first analysis: `notebooks/04_delta_alpha_biology.ipynb` |
| Supplementary Figure S21 | `fig:supp-dalpha-hallmark`; figure key `supp_hallmark` | Hallmark pathway enrichment of Δα | `T2_2_regen` (c), `F1_figures` (a, b, c) | panels: `analysis/tasks/F1_figures/notebook_panels.py`; enrichment tables: `notebooks/06_pathway_analysis.ipynb` |
| Supplementary Figure S22 | `fig:supp-celltype-closure`; figure key `supp_celltype_closure` | Closure of the cell-type decomposition of the count evidence | `T2_5_celltype_margin` | panel script: `analysis/tasks/T2_5_celltype_margin/run.py` |
| Supplementary Figure S23 | `fig:supp-celltype-seeds`; figure key `supp_celltype_seeds` | Seed stability of cell-type contributions to the count evidence | `T2_5_celltype_margin` | panel script: `analysis/tasks/T8_4_figures/supp_figures.py` |
| Supplementary Figure S24 | `fig:supp-functional-layer`; figure key `supp_functional_layer` | Pathway and network coherence and conservation | `T6_3_pathway`, `T6_4_network`, `T6_6_regulatory`, `T6_9_assemble` | panel script: `analysis/tasks/T6_9_assemble/assemble.py` |
| Supplementary Figure S25 | `fig:supp-worked-genes` | Extended caption of Figure 6, with the two worked genes (text only; the float also carries fig:supp-main-validation) | `Y3_findings` | - |
| Supplementary Table S25 | `tab:validation` | Pre-specified validation of Δα against independent data | `T4_2_gene_baselines`, `T4_3_scoreboard` | - |
| Supplementary Figure S26 | `fig:supp-validation-tests`; figure key `supp_validation_tests` | Pre-specified tests against independent data and comparison with direct baselines | `T4_2_gene_baselines` (c), `T4_3_scoreboard` (a, b, c), `T5_6_gene_level` (c), `X6_adds_beyond` (d), `T8_4_figures` | panels: `analysis/tasks/T8_4_figures/fig7_validation.py`; inputs: `results/analysis/T4_3/`, `results/analysis/X6/` (bundle `analysis_results`), `results/analysis/T5_6/` (bundle `tool_runs_saturn_camex`) |
| Supplementary Figure S27 | `fig:supp-gates`; figure key `supp_gates_readouts` | The tests and two named findings under the protein-feature gate and the per-gene gate | `Z3_free_gate` (a, b, d, e), `F2_per_gene` (a, b, c, d, e) | panels: `analysis/tasks/F2_per_gene/panels.py`; inputs: `results/analysis/Z3/` (`compare_hyp.csv`, `compare_pairs.csv`, `compare_reversal.csv`, `compare_y3.csv`, `compare_x5.csv`), `results/analysis/F2/` (bundle `analysis_results`) |
| Supplementary Figure S28 | `fig:supp-covariate-audit`; figure key `supp_covariate_audit` | Covariate audit of Δα in seven cohorts | `T2_4_covariates` | panel script: `analysis/tasks/T2_4_covariates/run.py` |
| Supplementary Figure S29 | `fig:supp-seed-reliability`; figure key `supp_seed_reliability` | Agreement between training seeds | `T3_3_consensus`, `T8_4_figures` | panel script: `analysis/tasks/T8_4_figures/supp_figures.py` |
| Supplementary Figure S30 | `fig:supp-gate-gap`; figure key `supp_gate_gap` | Where the two readouts agree and differ | `X8_gate_divergence` | panel script: `analysis/tasks/X8_gate_divergence/figures.py` |
| Supplementary Figure S31 | `fig:supp-gate-gap-cont`; figure key `supp_gate_gap_classes` | The difference between the readouts by tissue, gene class and independent reference | `X8_gate_divergence` | panel script: `analysis/tasks/X8_gate_divergence/figures.py` |
| Supplementary Figure S32 | `fig:supp-gate-gap-named`; figure key `supp_gate_gap_named` | Named genes where the two readouts differ | `X8_gate_divergence` | panel script: `analysis/tasks/X8_gate_divergence/figures.py` |
| Supplementary Figure S33 | `fig:supp-e6`; figure key `supp_published_differences` | Published species differences on the count evidence | `E6_known_differences` | panel script: `analysis/tasks/E6_known_differences/figures.py` |
| Supplementary Figure S34 | `fig:supp-b1`; figure key `supp_ground_truth` | Ground-truth benchmark: level alterations and protein identity | `B1_ground_truth` | panel script: `analysis/tasks/B1_ground_truth/figures.py` |
| Supplementary Figure S35 | `fig:supp-b2-overview`; figure key `supp_benchmark_overview` | Gene-level benchmark: overview | `B2_benchmark` | panel script: `analysis/tasks/B2_benchmark/overview.py` |
| Supplementary Figure S36 | `fig:supp-b2-tasks`; figure key `supp_benchmark_tasks` | Gene-level benchmark: every task | `B2_benchmark` | panel script: `analysis/tasks/B2_benchmark/overview.py` |
| Supplementary Figure S37 | `fig:supp-main-one-reading` | Extended caption of Figure 7 (text only) | - | - |
| Supplementary Figure S38 | `fig:supp-h1-recurrence-all`; figure key `supp_tissues_recurrence` | All gene sets that recur across tissues | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S39 | `fig:supp-e1`; figure key `supp_programs_recurrent` | Gene programs that recur across eight tissues | `E1_atlas` | panel script: `analysis/tasks/E1_atlas/figures.py` |
| Supplementary Figure S40 | `fig:supp-e1-cont`; figure key `supp_programs_by_tissue` | Tissue programs in each tissue | `E1_atlas` | panel script: `analysis/tasks/E1_atlas/figures.py` |
| Supplementary Figure S41 | `fig:supp-e1-genes`; figure key `supp_programs_named_genes` | Named genes that recur across tissues | `E1_atlas` | panel script: `analysis/tasks/E1_atlas/figures.py` |
| Supplementary Figure S42 | `fig:supp-gene-by-tissue`; figure key `supp_gene_by_tissue` | Gene and tissue components of the readouts | `Y1_gene_tissue` (a, b, c, d, e, f) | panels: `analysis/tasks/Y1_gene_tissue/figure.py`; inputs: `results/analysis/Y1/` (`variance_split.csv`, `contrast_rho.csv`, `contrast_delta.csv` and the tables beside them; bundle `analysis_results`) |
| Supplementary Figure S43 | `fig:supp-tissue-genes`; figure key `supp_tissues_named_genes` | Δα against protein identity and dN/dS, and named genes, in eight tissues | `H1_harmonise` (a, b) | panels: `analysis/tasks/H1_harmonise/figures.py` (`main_axis`, `main_genes`); inputs: `results/analysis/H1/genes/<cohort>.parquet`, `results/analysis/H1/axis.csv` (bundle `analysis_results`) |
| Supplementary Figure S44 | `fig:supp-h1-bone-marrow`; figure key `supp_tissue_bone_marrow` | The reading in bone marrow | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S45 | `fig:supp-h1-kidney`; figure key `supp_tissue_kidney` | The reading in kidney | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S46 | `fig:supp-h1-islet`; figure key `supp_tissue_islet` | The reading in pancreatic islet | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S47 | `fig:supp-h1-eye`; figure key `supp_tissue_eye` | The reading in the eye | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S48 | `fig:supp-h1-cortex`; figure key `supp_tissue_cortex` | The reading in cerebral cortex | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S49 | `fig:supp-h1-lung`; figure key `supp_tissue_lung` | The reading in lung | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S50 | `fig:supp-h1-muscle`; figure key `supp_tissue_muscle` | The reading in skeletal muscle | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S51 | `fig:supp-h1-tumour`; figure key `supp_tissue_tumor_myeloid` | The reading in lung-tumor myeloid cells | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S52 | `fig:supp-h1-lineage`; figure key `supp_tissues_lineage` | Lineage marker sets across tissues | `H1_harmonise` | panel script: `analysis/tasks/H1_harmonise/figures.py` |
| Supplementary Figure S53 | `fig:supp-bone-marrow`; figure key `supp_bone_marrow_reference` | Bone marrow against sorted-lineage conservation of expression | `T12_3_tests` | panel script: `analysis/tasks/T12_3_tests/figure.py` |
| Supplementary Figure S54 | `fig:supp-e3-compartments`; figure key `supp_cell_compartments` | Home compartments of genes against Δα | `E3_cells` | panel script: `analysis/tasks/E3_cells/figure.py` |
| Supplementary Figure S55 | `fig:supp-e3-lineages`; figure key `supp_cell_lineages` | Cell types that carry shared deployment | `E3_cells` | panel script: `analysis/tasks/E3_cells/figure.py` |
| Supplementary Figure S56 | `fig:supp-e3-primates`; figure key `supp_primate_branches` | Species-specific departures along the primate phylogeny | `E3_cells` | panel script: `analysis/tasks/E3_cells/figure.py` |
| Supplementary Figure S57 | `fig:supp-e2`; figure key `supp_protein_architecture` | Protein architecture and gene history against Δα | `E2_protein` | panel script: `analysis/tasks/E2_protein/figures.py` |
| Supplementary Figure S58 | `fig:supp-e5`; figure key `supp_regulatory_architecture` | Regulatory and gene-architecture features against Δα | `E5_regulatory` | panel script: `analysis/tasks/E5_regulatory/figures.py` |
| Supplementary Figure S59 | `fig:disease`; figure key `supp_disease_transfer` | Transfer of disease programs between human and mouse | `T7_3_mash`, `T7_9_islet`, `P13_kidney` (a, b, c), `T7_10_fig8` | panels: `analysis/tasks/T7_10_fig8/fig8_disease.py`; inputs: `results/analysis/P13/`, `results/analysis/T7_3/`, `results/analysis/T7_9/`, `results/analysis/T7_10/` (bundle `analysis_results`) |
| Supplementary Figure S60 | `fig:supp-e4-targets` | Extended caption of Figure 9 (text only) | `E4_translational` | - |
| Supplementary Figure S61 | `fig:supp-e4`; figure key `supp_disease_genes` | Disease genes and drug targets by organ system | `E4_translational` | panel script: `analysis/tasks/E4_translational/figures.py` |
| Supplementary Figure S62 | `fig:supp-main-gin-esm2`; figure key `supp_gin_inputs_controls` | Inputs of the gate network and control trainings | `T3_4_destroy_features` (c), `T3_5_count_perm` (c), `T3_6_matched` (c), `T8_4_figures` (c) | panel a: `figures/redraw/main_figures.py` (`attribution_shares`, values stored by `notebooks/09_gin_attention_overall.ipynb`); panel b: `figures/redraw/gate_network.py` (`inputs_contributions`; it evaluates the stored gate network of the four-species liver run, bundle `models_canonical`); panel c: `analysis/tasks/T8_4_figures/fig5_controls.py`; inputs: `results/analysis/T3_4/`, `results/analysis/T3_5/`, `results/analysis/T3_6/` (bundle `analysis_results`) |
| Supplementary Figure S63 | `fig:supp-gin-esm2-linear-baseline`; figure key `supp_gin_linear_baseline` | Linear surrogate and scalar-correlation controls of the gate correction | - | panels: `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S64 | `fig:supp-gin-esm2-attribution-suite`; figure key `supp_gin_attribution_suite` | Attribution of the gate correction to input groups, all methods | - | panels: `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S65 | `fig:supp-gin-esm2-method-consensus`; figure key `supp_gin_method_consensus` | Consensus of the attribution methods | - | panels: `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S66 | `fig:supp-gin-esm2-orthogonality`; figure key `supp_gin_orthogonality` | ESM-2 and expression inputs: orthogonality and ablation | - | panels: `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S67 | `fig:supp-gin-esm2-per-pc`; figure key `supp_gin_per_pc` | Contribution of single ESM-2 principal components | - | panels: `figures/redraw/gate_network.py`; first analysis: `notebooks/09_gin_attention_overall.ipynb` |
| Supplementary Figure S68 | `fig:supp-main-esm2-umap` | Extended caption of Figure 10, protein-space atlas (text only) | - | `notebooks/05_esm_alpha_geometry.ipynb`; `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Figure S69 | `fig:supp-archetypes`; figure key `supp_atlas_archetypes` | Gene archetypes of the gate network | - | panels a, c to f: `figures/redraw/main_figures.py` (`archetype_number`, `archetype_delta_alpha`, `archetype_protein_map`, `archetype_neighbors`, `archetype_prediction`); panel b: stored by `notebooks/11_gene_archetypes.ipynb` (needs the trained gate network); inputs: `tables/gene_archetypes.csv` and `cache/esm2_umap.npy` of the four-species liver run (bundle `models_canonical`) |
| Supplementary Figure S70 | `fig:supp-esm2-umap-spatial-autocorr`; figure key `supp_atlas_spatial_autocorrelation` | Neighborhood coherence of α and Δα on the ESM-2 graph | - | panels: `figures/redraw/atlas.py`; first analysis: `notebooks/05_esm_alpha_geometry.ipynb` |
| Supplementary Figure S71 | `fig:supp-esm2-umap-archetype-trajectory`; figure key `supp_atlas_archetype_trajectory` | Δα of the archetypes during training | - | panels: `figures/redraw/atlas.py`; archetype table: `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Figure S72 | `fig:supp-esm2-umap-archetype-enrichment`; figure key `supp_atlas_archetype_enrichment` | Gene categories enriched in the archetypes | - | panels: `figures/redraw/atlas.py`; archetype table: `notebooks/11_gene_archetypes.ipynb` |
| Supplementary Table S26 | `tab:residue-specificity` | Specificity of the residue-level contrasts | `T8_1_residue` | - |
| Supplementary Figure S73 | `fig:supp-main-gin-protein`; figure key `supp_residue_audit` | Residue-level attribution audit | `T2_2_regen` (e) | panels: `figures/redraw/residue_audit.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb` |
| Supplementary Figure S74 | `fig:supp-gin-protein-residue-beta`; figure key `supp_residue_mixed_effects` | Residue-level mixed-effects regression | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
| Supplementary Figure S75 | `fig:supp-gin-protein-cliffs-delta`; figure key `supp_residue_cliffs_delta` | Protein-replicated effect sizes of residue features | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
| Supplementary Figure S76 | `fig:supp-gin-protein-attention-by-feature`; figure key `supp_residue_attention_by_feature` | Per-residue attribution by structural feature | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
| Supplementary Figure S77 | `fig:supp-gin-protein-etv1`; figure key `supp_residue_card_etv1` | Per-gene card: ETV1 | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
| Supplementary Figure S78 | `fig:supp-gin-protein-hprt1`; figure key `supp_residue_card_hprt1` | Per-gene card: HPRT1 | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
| Supplementary Figure S79 | `fig:supp-gin-protein-runx1`; figure key `supp_residue_card_runx1` | Per-gene card: RUNX1 | - | panels: `figures/redraw/residue.py`; residue tables: `notebooks/10_gin_attribution_protein_structure.ipynb`; sequences: `notebooks/09_protein_structure_visualization.ipynb` |
<!-- paper-map:end -->

**Retired from the paper.** Five displays of earlier analyses are not in the paper, because the
paper shows a later analysis of the same question on the same data. Their code, panel files and
caches stay in the repository, so each can still be drawn.

| Not in the paper | The paper shows instead | Code and files of the retired display |
|---|---|---|
| Integration scorecard among seven methods (composite score, mean score, cohorts won) | the benchmark of ten methods on the same cohorts and metrics: task `T5_5_benchmark`, `fig:integration`, `fig:supp-integration-purity` | `notebooks/07_baselines_master.ipynb`; `figures/redraw/integration_scores.py`; `figures/panels/supp_integration_seven_methods/` |
| Raw integration metrics among seven methods | the same six metrics for ten methods: `fig:supp-integration-raw` | the same notebook and script; `figures/panels/supp_integration_metrics/` |
| Δα against detection rate, mean count and count variance on liver Human+Monkey | the covariate audit of the seven canonical cohorts: task `T2_4_covariates`, `fig:supp-covariate-audit` | `notebooks/03_alpha_master.ipynb`; `figures/redraw/alpha.py`; `figures/panels/supp_alpha_decorrelation/` |
| Program means of Δα, liver against testis, against the clamped prior (two panels) | the tissue comparison against the raw prior, centered, genes matched on the human Ensembl gene ID: task `T2_3_tissue`, `fig:supp-dalpha-sources` | `figures/redraw/robustness.py`; `figures/panels/supp_alpha_robustness/tissues_*.pdf` |
| Sign classes of Δα, liver against testis, against the clamped prior (one panel) | the same comparison of `T2_3_tissue` | `notebooks/08_ortholog_qc.ipynb`; `notebooks/figures/ortholog_qc/i3c_cross_tissue_disagreement_portrait_grayscale.pdf` |

The layouts of the retired figures are kept with the figure specs
([`figures/README.md`](figures/README.md), "Retired figures"). Tasks that a later task replaced
have the status "retired, replaced by" in [`analysis/README.md`](analysis/README.md)
(`T12_2_recipe`, `T7_9_recipe`); stand-alone scripts of the first version of the model, which no
analysis of the paper uses, are in [`scripts/`](scripts/README.md).

### 4.2 Workflow notebooks

[`notebooks/README.md`](notebooks/README.md) is the guide to the notebooks: one row per notebook
(what it does, what it needs and writes, which scripts read its outputs today), how the builders
(`notebooks/_*_builder.py`) relate to them, the numbering, and the pitfalls. In short:

- `00` builds the environment, `01` the liver and testis objects, `02` the baseline latents of
  the seven liver and testis cohorts.
- `03` to `11` hold the first analyses of the gate, of gene categories and pathways, of the
  protein space, of the gate network and of residue-level attribution, with their executed
  outputs. Several of them write tables and caches that figure scripts and tasks read.
- The panels of the paper are not drawn by executing a notebook. They are drawn at print size
  by `figures/redraw/` and by task scripts from the stored runs and caches, following the cells
  of the notebooks ([`figures/producers.tsv`](figures/producers.tsv) names the script and the
  command of every panel). The figures take from `notebooks/figures/` only the few files that a
  figure spec names with `NB(...)`; [`figures/INDEX.md`](figures/INDEX.md) lists them.

| Notebook | Status |
|---|---|
| `00_environment` | producer: the environment |
| `01_data_preparation` | producer: the liver and testis objects, the protein embeddings and the raw matrices that other builds read; its own eye object is replaced by `T11_1_vanzyl` |
| `02_baselines` | producer: baseline latents of the seven liver and testis cohorts |
| `03_alpha_master` | record of the first rendering; its panels of Δα against expression statistics and of liver against testis are retired (the paper shows `T2_4_covariates`, `T2_3_tissue`) |
| `04_delta_alpha_biology`, `04_delta_alpha_biology_simpler` | record of the first rendering |
| `05_esm_alpha_geometry` | producer: the stored protein map; otherwise record |
| `06_pathway_analysis` | producer: the enrichment tables |
| `07_baselines_master` | producer: it scores the seven general-purpose methods, and their raw metrics enter the ten-method benchmark of `T5_5_benchmark`; its own scorecard panels are retired |
| `08_chimera_umap` | producer: the stored UMAP coordinates |
| `08_ortholog_qc` | record; its liver-against-testis portrait is retired (the paper shows `T2_3_tissue`) |
| `08_alpha_integration_confidence` | record; no figure of the paper |
| `09_gin_attention_overall` | record; no panel of a figure comes from it (`figures/redraw/gate_network.py` repeats its computation on the stored gate network) |
| `09_protein_structure_visualization` | producer: frozen sequences and structures |
| `10_gin_attribution_protein_structure` | producer: the residue tables; do not execute it to redraw a figure |
| `10b_gin_attribution_figures` | record of the first rendering |
| `11_gene_archetypes` | producer: the archetype table; one panel still placed (panel b of the supplementary figure `supp_atlas_archetypes`) |

"Producer": scripts of the paper read its outputs. "Record": its executed outputs document the
first analysis, and the panels of the paper are drawn by the scripts that follow its cells. No
notebook is deleted or renamed.

Run one headless, from the repository root, with

```bash
papermill notebooks/<notebook>.ipynb <output folder>/<notebook>.ipynb --cwd notebooks -k python3
```

(`-k python3` is the kernel of the active environment; `00_environment.ipynb` also registers it
as `chimera_env`). Writing the output back onto the notebook instead replaces the tracked copy.
Do not execute notebook `10` to redraw a figure (see the pitfalls of the notebook guide).

### 4.3 Per-gene values

- **One run.** `training/runs/<run>/final_state.npz` (load with `allow_pickle=False`) holds
  `alpha`, `prior`, `correction`, `S_g_ema` per gene; `delta_alpha = alpha - prior`. The file has
  no gene names: take `var_names` from the cohort object and map them with the cohort's own
  ortholog table ([`analysis/common/gene_ids.py`](analysis/common/gene_ids.py)). Testis runs were
  trained with the prior clamped at 0.85 (`prior` is the clamped one); the unclamped prior is
  `conservation_prior_raw.npy` in the run folder.
- **Seed consensus of the seven canonical cohorts.** `results/gene_tables/<cohort>__consensus.csv`:
  means over the seeds listed in `seed_pooling.json` (`used`). `results/gene_tables/` exists only
  in the deposit; `T3_3_consensus` writes the same files into `results/analysis/T3_3/`.
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
- **Codes that look alike.** `H1` to `H6` in validation tables are the pre-specified hypotheses,
  not the task `H1_harmonise`. `G1` to `G10` under `data/analysis/gold/` are independent
  references, not the task `G1_guard`. `B1b`, `B1c`, `B1z`, `B2b`, `B3a`, `B3b` are direct
  baselines, not the benchmark tasks `B1_ground_truth` and `B2_benchmark`. `E1` to `E9` are the
  training expectations of `training/check_expectations.py`, not the tasks `E1` to `E6`. `F1`,
  `F2`, `F3` in the column `family` of scoreboard tables are families of tests, not the tasks
  `F1_figures` and `F2_per_gene`.
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
| `baselines_<method>/` | latents of the integration baselines | `notebooks/02_baselines.ipynb` (PCA, Harmony, scVI, scANVI, sysVI on the seven liver and testis cohorts); `T5_3_runs`, `T5_6_gene_level` (SATURN, CAMEX); `T11_4_vz_integration`, `B2_benchmark` (the further cohorts) | `notebook_caches` |
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

All 26 cohorts in ten tissues are built from public data by code of this repository. Run
[`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) first: besides the
liver and testis objects it downloads the SATURN ESM-2 protein embeddings into
`data/raw/saturn_esm2/`, which every other build reads, and the count matrices of the eye cohort.
The functions are those of `chimera/_data_utils.py` in every build.

| Tissue | Cohort keys | Raw data | Fetched by | Built by | Object |
|---|---|---|---|---|---|
| Liver | `liv_hmonkey`, `liv_hmouse`, `liv_hzeb`, `liv_4sp` | the CAMEX bundle (archived at figshare doi:10.6084/m9.figshare.31131808.v1) | notebook `01` (`download_camex_pancreas`: the CAMEX authors' Google Drive folder, with `gdown`) | notebook `01` | `data/processed/camex_human_monkey.h5ad`, `camex_human_mouse.h5ad`, `camex_human_zebrafish.h5ad`, `camex_pancreas_unified.h5ad` |
| Testis | `test_hmac`, `test_hmouse`, `test_3sp` | the CAMEX bundle | notebook `01` (`download_camex_testis`) | notebook `01` | `data/processed/camex_testis_human_macaque.h5ad`, `camex_testis_human_mouse.h5ad`, `camex_testis_unified.h5ad` |
| Eye | `vz_full`, `vz_hmouse`, `vz_hmmul`, `vz_primates`, `vz_hpig`, `vz_hmfas` | five GEO count matrices (GSE146186, GSE146187, GSE148371, GSE148373, GSE148374) | notebook `01`, section 11 (`download_van_zyl_raw`), which also writes the source ortholog table `data/processed/ortholog_table_van_zyl.csv` | `python -m analysis.tasks.T11_1_vanzyl.build_vanzyl`, then `python -m analysis.tasks.T11_2b_vz_subsets.build_subsets` | `data/analysis/van_zyl/van_zyl_unified.h5ad` (five species), `van_zyl_<subset>.h5ad` |
| Bone marrow | `bone_marrow` | three CELLxGENE files | `bash analysis/tasks/T12_1_bone_marrow/fetch_bm.sh $PWD` | `python -m analysis.tasks.T12_1_bone_marrow.build_orthologs`, then `python -m analysis.tasks.T12_1_bone_marrow.build_bm` | `data/analysis/bone_marrow/v2/bone_marrow_unified.h5ad` |
| Kidney | `kidney` | two CELLxGENE files | `bash analysis/tasks/P13_kidney/fetch_kidney.sh $PWD` | `python -I analysis/tasks/P13_kidney/build_orthologs.py $PWD`, then `python -m analysis.tasks.P13_kidney.build` | `data/analysis/disease_kidney/kidney_healthy_train.h5ad` |
| Pancreatic islet | `islet` | two CELLxGENE files | `bash analysis/tasks/C1_cohorts/fetch_manifest.sh data/analysis/disease_islet/raw/MANIFEST.tsv` | `python -m analysis.tasks.T7_9_islet.build` | `data/analysis/disease_islet/islet_healthy_train.h5ad` |
| Cerebral cortex, lung, skeletal muscle, tumor myeloid cells | `brain_dlpfc*`, `brain_m1*`, `lung_healthy`, `muscle_ageing`, `tumour_zilionis` | CELLxGENE and GEO files | `bash analysis/tasks/C1_cohorts/fetch_manifest.sh data/analysis/new_usecases/MANIFEST_SHA256.tsv` | `python -m analysis.tasks.C1_cohorts.orthologs`, then `python -m analysis.tasks.C1_cohorts.build <cohort>` | `data/analysis/new_usecases/c1_build/objects/<cohort>.h5ad` |

The accessions and dataset versions are in the paper (Data and code availability); the README of
each task lists every step, and `analysis/common/gene_ids.py` (`h5ad_path`, `load_table`) knows
the object and the ortholog table of every cohort.

- **Downloads.** The fetch scripts of the cohorts take the repository root as their argument and
  write one row per file into the `MANIFEST.tsv` of the folder: file, URL, date, size and sha256
  (through `analysis/tasks/T4_1_gold/fetch_item.sh`, which resumes a broken download). The files
  of the islet cohort and of cortex, lung, skeletal muscle and tumor myeloid cells are fetched
  from their manifests, which come with bundle `data_other_cohorts`:
  `analysis/tasks/C1_cohorts/fetch_manifest.sh <manifest>` downloads each row to the path the
  build reads, checks size and sha256, keeps a file that is present and correct, and with
  `--dry-run` only lists what it would do. `data/analysis/new_usecases/MANIFEST_SHA256.tsv` gives
  URL, date, size and sha256 of 44 files of that folder (22 GB): the 39 CELLxGENE and GEO files
  of the two download logs beside it (`MANIFEST.tsv`, `MANIFEST_C1.tsv`) and five Ensembl files
  that the builds read (four annotation files of earlier releases in `ensembl_old/`, the human
  protein file of release 110). The other Ensembl release 110 files of that folder have their
  own manifest and are fetched by `analysis/tasks/C1_cohorts/fetch_ensembl.sbatch`.
- **Downloads of notebook `01`.** They go to `data/raw/`. The notebook fetches the CAMEX
  authors' Google Drive folder with `gdown` (folder id in `chimera/_data_utils.py`,
  `download_camex_pancreas` and `download_camex_testis`), the SATURN archives
  `protein_embeddings.tar.gz` and `ah_atlas_export.tar.gz` from snap.stanford.edu (no cohort
  build reads the second), and the five eye matrices from GEO with their cell annotation table.
  `data/raw/MANIFEST.tsv` (bundle `data_liver_testis`) lists these files with address, date,
  size, MD5 and sha256. For the 39 CAMEX files the address is that of the same file in the
  CAMEX record at figshare (doi:10.6084/m9.figshare.31131808.v1), to which the downloaded file
  is byte-identical; so the files can be fetched from either place and checked against the
  manifest. One command fetches and checks all 47 files of the manifest:
  `bash analysis/tasks/C1_cohorts/fetch_manifest.sh data/raw/MANIFEST.tsv`; the notebook then
  finds them and downloads nothing. The sha256 of the eye matrices and of the ESM-2 files that the
  eye build reads are also in `data/analysis/van_zyl/build_report.json` (bundle
  `data_other_cohorts`). The processed liver and testis objects are in `data_liver_testis`,
  whose `MANIFEST.tsv` gives their sha256.
- **Ortholog tables.** Use the archived tables instead of querying BioMart again: they are from
  Ensembl release 110, and if the release-110 archive does not answer, the query of notebook `01`
  falls back to the current release without an error. The tables of liver and testis are in
  `data_liver_testis`; the source table of the eye build
  (`data/processed/ortholog_table_van_zyl.csv`) is in `data_other_cohorts`.
- **Check.** `python -m analysis.common.check_cohort <h5ad> --batch-key <batch column>` before any
  training (without `--batch-key` the check reads the column `species`).
- **Liver and testis with counts joined on the Ensembl gene ID** (recommended for new work):
  `python -m analysis.tasks.O1_orthologs.repair --tissue liver --mode repair` writes
  `data/analysis/O1/repair_liver/`, with `repair_log.csv` listing each changed column. The models
  and tables of the paper were trained on the objects in `data/processed/`; the models trained on
  the identifier-joined objects are `training/runs/o2-*`.
- **Disease objects and references.** The disease case studies build their own objects
  (`T7_1_holdouts` and `T7_2_disease_objects` for liver, the later steps of `T7_9_islet` and
  `P13_kidney`, `C1_cohorts`); the independent references and annotations are downloaded by
  `T4_1_gold`, `T6_1_functional` and the `fetch.sh` of `E1_atlas`, `E2_protein`,
  `E4_translational` and `E5_regulatory`.
- **What was replaced.** The eye object that notebook `01` itself writes
  (`data/processed/van_zyl_unified.h5ad`) and the first build of the bone-marrow object
  (`data/analysis/bone_marrow/bone_marrow_unified.h5ad`) are not the cohorts of the paper and are
  not deposited; the objects in the table are.
