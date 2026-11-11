# Notebooks

The workflow of the model on the seven liver and testis cohorts: the environment, the data
preparation, the integration baselines, and the first analyses of the gate (α, Δα), of gene
categories and pathways, of the protein space, of the gate network and of residue-level
attribution. The analyses of the further cohorts, the validation, the controls and the
benchmarks are not here: they are in [`../analysis/`](../analysis/README.md).

What the notebooks are today:

- **Producers.** `00` builds the environment, `01` the liver and testis objects, `02` the
  baseline latents. Several analysis notebooks write tables and caches that the figure scripts
  and the analysis tasks read (table below, column "Read today by").
- **No notebook trains a Chimera model.** The models are trained by `training/run_experiment.py`
  ([`../training/README.md`](../training/README.md); the canonical recipes are in
  `configs/canonical_recipes.tsv`); `01` only constructs a model to check that an object is ready,
  `02` trains the baselines, and `03` to `11` read the stored runs under `training/runs/`.
- **The record of the first analyses.** Each analysis notebook keeps its executed outputs:
  numbers, tables and the first rendering of its panels in `notebooks/figures/<name>/`.
- **Not the place where the figures of the paper are drawn.** The panels of the paper are drawn
  at print size by [`../figures/redraw/`](../figures/redraw/) and by task scripts, from the stored
  runs, the processed objects and the caches named below; these scripts follow the cells of the
  notebooks and name them in their docstrings
  ([`../figures/producers.tsv`](../figures/producers.tsv) names the script of every panel). The
  figures of the paper take from `notebooks/figures/` only the few files that a figure spec names
  with `NB(...)`; [`../figures/INDEX.md`](../figures/INDEX.md) lists them under the figure that
  uses them (search it for `notebooks/figures/`), with the page and the number of every figure.

## What is in this folder

| Path | What it is |
|---|---|
| `00_*.ipynb` to `11_*.ipynb` | the notebooks, with their executed outputs (table below) |
| `_*_builder.py` | one Python source per analysis notebook: it holds every cell as text and writes the `.ipynb` without outputs (table below, column "Builder") |
| `_figure_design.py` | type, colors, axis ranges and bin edges shared by all notebook panels; `set_figure_dir` and the save function |
| `_paper_utils.py` | loaders shared by the notebooks, the figure scripts and several tasks: gene names of a cohort through its own ortholog table (`analysis/common/gene_ids.py`) |
| `_gene_categories.py`, `_gene_categories_broad.py` | the gene-category vocabularies (narrow categories, and ten broad ones); read by notebooks, `figures/redraw/` and tasks |
| `_summary_figures.py` | the drawing functions of notebooks `10` and `10b` |
| `_pipeline_cache.py` | the cache of notebook `10`: a step is skipped when the content hash of its inputs matches the stored one |
| `_gin_attribution_precompute.py` | the two memory-heavy steps of notebook `10` as a script (`--help`); it fills `cache/gin_attribution/` so the notebook only loads |
| `_scrub_output_paths.py` | removes machine paths from stored outputs (`--check` verifies; see `CONTRIBUTING.md`) |
| `figures/<name>/` | the panels as the notebooks rendered them (the first drawing of most analyses; PDF and PNG, and under `gin_attribution/3d/` the interactive structure views that notebook `10` writes as HTML), tracked in git; `<name>` is given in the table below. The figures of the paper take from it only the few files that a figure spec names with `NB(...)`; every other panel is drawn at print size by a script of `../figures/producers.tsv`. `figures/baselines_master_ext/` is written by the task `T5_5_benchmark`, not by a notebook. Not to be confused with [`../figures/`](../figures/README.md), where the figures of the paper are built |
| `cache/` | tables, coordinates and frozen downloads written by the notebooks and by `T5_5_benchmark`; not in git, part of the deposit (bundle `notebook_caches`) |

## The notebooks

Paths in this table and below are relative to the repository root. Every analysis notebook
reads the canonical runs (`training/runs/<run>/`, bundle `models_canonical`) and the processed
objects (`data/processed/`, bundle `data_liver_testis`).
"Run tables" and "run caches" are `training/runs/<run>/tables/` and `training/runs/<run>/cache/`
of the canonical runs; they travel with `models_canonical`.

| Notebook | Builder | Does | Also needs | Writes | Read today by | Status |
|---|---|---|---|---|---|---|
| `00_environment` | none | builds `chimera_env` from `pyproject.toml` (three install stages, editable install) and registers its Jupyter kernel | `mamba`; network | the environment | everything | producer; run interactively only |
| `01_data_preparation` | none | downloads the CAMEX bundle (liver, testis), the SATURN ESM-2 embeddings and the eye matrices; queries the ortholog tables (Ensembl 110); computes the protein prior and the 72 gene features; builds the unified objects and the species subsets | network; the `data` extra | `data/raw/`; `data/processed/`: seven `camex_*.h5ad` with their `_config.json`, `ortholog_table.csv`, `ortholog_table_testis.csv`, `gin_scaler*.pkl`; and a first eye object with `ortholog_table_van_zyl.csv` | every canonical model and every analysis; the cohort builders of `analysis/tasks/` read `data/raw/saturn_esm2/`; `T11_1_vanzyl` reads `ortholog_table_van_zyl.csv` | producer of the liver and testis objects. Its eye object (`data/processed/van_zyl_unified.h5ad`) is not the eye cohort of the paper, which `analysis/tasks/T11_1_vanzyl` rebuilds from the GEO matrices into `data/analysis/van_zyl/` |
| `02_baselines` | none | trains PCA, Harmony, scVI (species batch and donor batch), scANVI and sysVI on the seven cohorts through `training/run_all_baselines.py` | CPU, about 100 epochs per model | `training/runs/baselines_{pca,harmony,scvi,scanvi,sysvi}/` | notebooks `07` and `08_chimera_umap`; `T5_5_benchmark` | producer of the seven-cohort baseline latents. SATURN and CAMEX are not run here (`analysis/tasks/T5_3_runs`), nor are the baselines of the further cohorts (`T11_4_vz_integration`, `B2_benchmark/integration`) |
| `03_alpha_master` | `_alpha_master_builder.py` | α against the prior, α during training, Δα across seeds, species sets and tissues, Δα against expression statistics | - | `notebooks/figures/alpha_master/` | `figures/redraw/main_figures.py`, `alpha.py` and `robustness.py` follow its cells; the smoke test executes it with `RUN_FIGURE=1` | record of the first rendering. Its panels of Δα against expression statistics and of liver against testis are retired from the paper: the paper shows the covariate audit of seven cohorts (`analysis/tasks/T2_4_covariates`) and the tissue comparison against the raw prior (`analysis/tasks/T2_3_tissue`) |
| `04_delta_alpha_biology` | `_alpha_biology_builder.py` | Δα by gene category (narrow categories), enrichment in the tails, top and bottom genes, per cohort and across cohorts | - | `notebooks/figures/alpha_biology/`; run tables `delta_alpha_top30.csv`, `delta_alpha_bot30.csv` | `figures/redraw/categories.py` (four-species liver and testis Human+Macaque) and `main_figures.py` follow its cells | record of the first rendering |
| `04_delta_alpha_biology_simpler` | `_alpha_biology_simpler_builder.py` | the same analysis with the ten broad categories | as `04` | `notebooks/figures/alpha_biology_simpler/`; the same two run tables | `figures/redraw/categories_broad.py` follows its cells | record of the first rendering |
| `05_esm_alpha_geometry` | `_esm_geometry_builder.py` (not in step, see Pitfalls) | α, Δα and the correction on a UMAP of the 64 ESM-2 components; neighbor coherence and Moran's I with two nulls; protein clusters; Δα of liver against testis on shared genes | `umap-learn` | `notebooks/figures/alpha_geometry/`; run caches `esm2_umap.npy` (the stored protein map), `esm2_umap_shared_panc4sp_test3sp.npy`; run table `cross_tissue_delta_alpha_flip.csv` | `figures/redraw/main_figures.py` reads the stored map; `figures/redraw/atlas.py` repeats the neighbor statistics | producer of the stored protein map; otherwise record |
| `06_pathway_analysis` | `_pathway_builder.py` (not in step) | pre-ranked gene-set enrichment of Δα and of the prior (Hallmark, Reactome), per cohort and for liver against testis | the two Enrichr libraries in `~/.cache/gseapy/` (fetched when absent); `gseapy` | `notebooks/figures/pathway/`; run caches `gsea__<library>__<metric>.csv`, `gsea_crosstissue__<library>.csv`; run tables `gsea_*.csv` | `analysis/tasks/F1_figures/notebook_panels.py` (Hallmark panels) and `T2_2_regen/values.py` read the cached enrichment tables | producer of the enrichment tables |
| `07_baselines_master` | `_baselines_master_builder.py` | six integration metrics and the composite score for seven methods on seven cohorts (`training/compare_vs_scvi.py::compute_integration_metrics`) | the latents of `02` | `notebooks/figures/baselines_master/`; `notebooks/cache/baselines_master/<cohort>__<method>.json`, `master_table.csv` | `T5_5_benchmark` adds SATURN, SATURN-Leiden and CAMEX and writes the ten-method table `notebooks/cache/baselines_master_ext/master_table_ext.csv`; `figures/redraw/integration_scores.py`; `I1_free_gate_integration` | producer: it scores the seven general-purpose methods, and their raw metrics enter the ten-method benchmark of `T5_5_benchmark`, which is the benchmark of the paper. Its own scorecard panels (composite and raw metrics among seven methods) are retired from the paper |
| `08_chimera_umap` | `_chimera_umap_builder.py` | UMAP of every method's latent on every cohort, colored by species, cell type and donor | the latents of `02` | `notebooks/figures/chimera_umap/`; `notebooks/cache/chimera_umap/<cohort>__<method>__umap.npy` | `figures/redraw/integration_umap.py`, `figures/redraw/overview/build_assets.py` read the stored coordinates | producer of the stored UMAP coordinates |
| `08_ortholog_qc` | `_ortholog_qc_builder.py` | genes whose α departs strongly from their protein prior, by direction, by species and across tissues | - | `notebooks/figures/ortholog_qc/`; run tables `ortho_qc_*.csv`; `notebooks/cache/ortholog_qc_summary.csv` | `T2_2_regen/values.py` reads the summary | record. Its liver-against-testis portrait is retired from the paper (the paper shows the comparison of `analysis/tasks/T2_3_tissue`); no panel of a figure comes from it |
| `08_alpha_integration_confidence` | `_alpha_confidence_builder.py` | α as a three-tier integration confidence per gene, by cell type, with marker checks | `notebooks/cache/baselines_master/` | `notebooks/figures/alpha_confidence/`; `notebooks/cache/alpha_confidence/` | `T2_2_regen/values.py` reads two of its tables | record; no figure of the paper |
| `09_gin_attention_overall` | `_gin_attention_overall_builder.py` | what the gate network reads: linear baselines and four attribution methods on the four-species liver model | the saved model (`model/model.pt`); `captum` | `notebooks/figures/gin_attention_overall/` | `figures/redraw/gate_network.py` repeats the computation; `figures/redraw/main_figures.py` copies the attribution shares that one cell printed; one panel is placed from `notebooks/figures/gin_attention_overall/` | record; no panel of a figure comes from it |
| `09_protein_structure_visualization` | `_protein_structure_builder.py` | selects genes; fetches ortholog protein sequences (Ensembl REST), accessions (UniProt) and structures (AlphaFold DB); per-residue identity tracks | network on a first run | `notebooks/figures/protein_structure/`; `notebooks/cache/protein_structure/` (`sequences.json`, `uniprot.json`, `pdbs/`) | notebook `10`; `figures/redraw/residue.py` | producer of the frozen sequences and structures |
| `10_gin_attribution_protein_structure` | `_gin_attribution_builder.py` (not in step) | residue-level attribution: integrated gradients on the gate network, projected through the PCA basis onto residues of 70 proteins; structure features; pooled and per-gene tests; mixed model | `notebooks/cache/protein_structure/`; SATURN ESM-2 embeddings; ESM-2 35M weights; `captum` | `notebooks/figures/gin_attribution/`; `notebooks/cache/gin_attribution/` (attributions, per-residue signals, `L_analysis/*.pkl`) | `figures/redraw/residue.py`, `residue_audit.py`; `analysis/tasks/T8_1_residue` | producer of the residue tables. Do not execute it on another machine (Pitfalls) |
| `10b_gin_attribution_figures` | `_gin_attribution_figures_builder.py` (not in step) | redraws the residue figures from `notebooks/cache/gin_attribution/L_analysis/` only (seconds) | the cache of `10` | `notebooks/figures/gin_attribution/` | `figures/redraw/residue.py` and `residue_audit.py` draw the same tables at print size | record of the first rendering |
| `11_gene_archetypes` | `_gene_archetypes_builder.py` | clusters genes by the hidden state of the gate network (archetypes), profiles them, tests category enrichment | the saved model; `captum` | `notebooks/figures/gene_archetypes/`; run tables `gene_archetypes.csv`, `gene_archetypes_top_genes.csv` | `figures/redraw/atlas.py` and `main_figures.py` read the archetype table, and `main_figures.py` copies three tables that its cells printed; `analysis/tasks/T2_6_matched`, `X2_liver_biology`; one panel is placed from `notebooks/figures/gene_archetypes/` | producer of the archetype table; one panel still placed (panel b of the supplementary figure `supp_atlas_archetypes`) |

No notebook is deleted or renamed. What the paper no longer shows, and what it shows instead, is
said in the last column (the eye object of `01`; panels of `03`, `07` and `08_ortholog_qc`) and,
in one table, in [`../REPRODUCE.md`](../REPRODUCE.md), section 4.1 ("Retired from the paper").

## Numbering

The number is the order in which a notebook can be run, not a unique key: a notebook needs only
notebooks with a lower number. Notebooks that share a number are independent of each other.

- `04` and `04_..._simpler` are twins: the same cells with the narrow and with the ten broad
  categories.
- `08_chimera_umap`, `08_ortholog_qc`, `08_alpha_integration_confidence`: three independent
  analyses that need `07` or earlier.
- `09_gin_attention_overall` (gate network) and `09_protein_structure_visualization`
  (sequences and structures) are independent; `10` needs the second.
- `10b` draws from the cache of `10`.

File names, the folders under `figures/` and `notebooks/cache/`, and the builders are keyed to each other
by these names. Do not rename a notebook.

In code and file names `panc_*` and `pancreas_*` mean the **liver** cohorts
([`../REPRODUCE.md`](../REPRODUCE.md), section 5).

## Running a notebook

Inputs: unpack `data_liver_testis`, `models_canonical` and `notebook_caches`
([`../REPRODUCE.md`](../REPRODUCE.md), section 3). From the repository root, inside `chimera_env`:

```bash
OUT=$(mktemp -d)
papermill notebooks/03_alpha_master.ipynb $OUT/03_alpha_master.ipynb --cwd notebooks -k python3
```

- `-k python3` is the kernel of the active environment. The analysis notebooks name the kernel
  `chimera_env`, which `00_environment` registers; `-k python3` works with and without it.
- Write the executed copy outside the repository, as above. Writing it back onto the notebook
  replaces the tracked copy and its stored outputs.
- An executed notebook writes its panels into `notebooks/figures/<name>/` and its tables into
  `notebooks/cache/` and the run folders, so `git status` shows what changed.
- With warm caches: `03` about 2 minutes, `04` 3, `05` 10, `06` 1 to 2, `07`, `09`, `10b` under
  1, `08_chimera_umap` 1, `11` 3. `01` takes about 30 minutes and 10 GB; `02` trains every
  baseline on every cohort.
- `00_environment` is run interactively (it creates the environment it would run in). On Linux
  the environment is built without it: `bash analysis/envs/build_env_from_lock.sh chimera_env`.

To draw the panels of the paper, do not run a notebook: run the script named in the table
(`python -m figures.redraw.<script>`, see [`../figures/README.md`](../figures/README.md)).

## Builders

`python notebooks/<builder>.py` rewrites its notebook from the cells in the builder, **without
outputs**; executing the notebook afterwards restores them. Edit an analysis in the builder, not
in the notebook. Ten notebooks equal what their builder writes, cell for cell: `03`, `04`,
`04_..._simpler`, `07`, the three `08_*`, the two `09_*` and `11`. Four do not, and must not be
regenerated before their cells are carried into the builder:

| Notebook | What only the notebook holds |
|---|---|
| `05_esm_alpha_geometry` | the liver and testis variants of the coherence and Moran panels, the twelve-category panels and the effect-size panel (15 cells differ) |
| `06_pathway_analysis` | the plot options of the liver-against-testis volcano (2 code cells differ) |
| `10_gin_attribution_protein_structure` | the wording of 19 text cells; code cells are equal |
| `10b_gin_attribution_figures` | one more figure variant and a closing summary (3 cells), and the wording of 14 text cells |

`00`, `01` and `02` have no builder.

## Pitfalls

- **Stored outputs come from two environments.** `03`, `04`, `04_..._simpler`, `05`, `06`,
  `08_alpha_integration_confidence` and `08_ortholog_qc` hold the outputs of a headless run on
  Linux with the lock-file environment. `00`, `01`, `02`, `07`, `08_chimera_umap`, `09_*`, `10`,
  `10b` and `11` hold the outputs of their interactive runs on the Apple machine that trained the
  canonical models. A re-run on another machine repeats the values, not every last digit (for
  example the completeness error of integrated gradients in `09`).
- **Do not execute `10` to redraw a figure.** Its cache is keyed by content hashes of its inputs;
  on another machine a hash can differ, and the notebook then recomputes embeddings and
  attributions, downloads again, and writes over `notebooks/cache/gin_attribution/`, which the figure
  scripts and `T8_1_residue` read. Use `figures/redraw/residue.py` and `residue_audit.py`, which
  only read that cache.
- **Live services.** `09_protein_structure_visualization` and `10` read Ensembl REST, UniProt and
  AlphaFold DB; their responses are frozen in `notebooks/cache/protein_structure/`. `06` fetches the two
  Enrichr libraries into `~/.cache/gseapy/` when they are absent; its enrichment tables are
  frozen in the run caches. `01` queries Ensembl BioMart: use the archived ortholog tables
  instead of querying again ([`../REPRODUCE.md`](../REPRODUCE.md), section 8).
- **`04` and its twin write the same run tables** (`delta_alpha_top30.csv`,
  `delta_alpha_bot30.csv`): the genes and values are equal, and the category column is that of
  the notebook that ran last.
- **The prose of the notebooks is the working text of the time they were written.** It cites
  iteration numbers of the tuning record (`training/iteration_log.md`) and a few design notes
  that are not part of the repository. The paper, not the notebook prose, states the claims.
