# Reproducing the Chimera results

> Draft. `[TAG]` and `[DOI]` are filled in when the release is archived.

This guide regenerates the analyses of the paper from the archived artefacts of release
`[TAG]` (archive `[DOI]`). Training the canonical models again is possible from the recipe
table, but is not needed to reproduce any number: every analysis reads the archived runs.
The composite figures of the paper were assembled by hand from the rendered panels.

## 1. Code and environment

```bash
git clone --branch [TAG] https://github.com/theislab/chimera.git
cd chimera
```

Create `chimera_env` with [`notebooks/00_environment.ipynb`](notebooks/00_environment.ipynb)
(or by hand, see the README), then check the installation with `pytest -q`. Every command
below runs inside `chimera_env` from the repository root. Notebooks find the repository from
their own location; set `CHIMERA_ROOT=/path/to/chimera` to override.

## 2. Archived artefacts

Download the bundles from `[DOI]`, verify them against `SHA256SUMS`, and unpack them in the
repository root:

| Bundle | Unpacks to | Content |
|---|---|---|
| `processed.tar` | `data/processed/` | 7 cohort AnnData files, ortholog tables, GIN feature scalers |
| `checkpoints.tar` | `training/runs/<run>/` | Canonical runs: `model/`, `final_state.npz`, alpha snapshots, metrics, config, tables, caches |
| `gene_tables.tar` | `results/gene_tables/` | One CSV per cohort: Ensembl gene ID, symbol, prior, raw prior, α, Δα, correction, S_g |
| `caches.tar` | `notebooks/cache/`, `training/runs/baselines_*` | Baseline runs and notebook caches, including frozen external lookups |
| `revision.tar` | `results/revision/` | Outputs of the analyses added during revision |

`data/`, `training/runs/`, `notebooks/cache/` and `results/` are not tracked by git.

## 3. Naming

- **`panc_*` and `pancreas_*` mean liver.** The liver cohorts come from the CAMEX liver
  bundle and were named `pancreas` early in the project; the names were kept so that code,
  caches and run directories stay consistent. `camex_pancreas_unified.h5ad` is the liver
  4-species cohort. Figure labels and the paper say "liver".
- Cohort keys used in `revision/` and `configs/canonical_recipes.tsv`: `liv_hmonkey`,
  `liv_hmouse`, `liv_hzeb`, `liv_4sp`, `test_hmac`, `test_hmouse`, `test_3sp`.
- **Gene identifiers.** `var_names` are `ortho_N`, the row number in the ortholog table that
  built the cohort: `ortholog_table.csv` for liver, `ortholog_table_testis.csv` for testis,
  `ortholog_table_van_zyl.csv` for van Zyl. The numbering is local to each table, so the same
  `ortho_N` names different genes in different tissues. Look genes up with
  [`revision/common/gene_ids.py`](revision/common/gene_ids.py), which takes the cohort
  explicitly, and join across tissues on `hsapiens_gene_id` only.

## 4. Canonical runs and Δα tables

[`configs/canonical_recipes.tsv`](configs/canonical_recipes.tsv) lists, per cohort, the AnnData
file, ortholog table, epochs, overrides, seeds and run directory. All runs use `--config A`.
To train one again:

```bash
python training/run_experiment.py --run-name <name> --config A --max-epochs <N> \
    --adata data/processed/<h5ad> --seed <seed> --accelerator cpu <overrides>
```

The checkpoints were trained on Apple MPS; retraining on CPU or CUDA gives close but not
identical values, and the archived runs remain the reference.

Per-gene values come from `training/runs/<run>/final_state.npz` (load with
`allow_pickle=False`): `delta_alpha = alpha - prior`. The file holds no gene names; take
`var_names` from the cohort AnnData and map them with the cohort's own table. Testis runs
were trained with the prior clamped at 0.85 (`prior` in `final_state.npz` is clamped); the
unclamped prior is `conservation_prior_raw.npy` in the same run directory.

## 5. Figures

Each notebook renders its panels into `notebooks/figures/<notebook>/`. Run a notebook
headless with

```bash
papermill notebooks/<notebook>.ipynb notebooks/<notebook>.ipynb --cwd notebooks -k chimera_env
```

| Figure | Notebooks |
|---|---|
| 2 and supplement | `03_alpha_master` |
| 3 and supplement | `07_baselines_master`, `08_chimera_umap` |
| 4 and supplement | `04_delta_alpha_biology`, `04_delta_alpha_biology_simpler`, `06_pathway_analysis` |
| 5 and supplement | `09_gin_attention_overall` |
| 6 and supplement | `05_esm_alpha_geometry`, `11_gene_archetypes` |
| 7 and supplement | `10_gin_attribution_protein_structure`, `10b_gin_attribution_figures`, `08_ortholog_qc` |

Notebooks `06`, `09_protein_structure_visualization` and `10` used live services when first
run (Enrichr gene-set libraries, Ensembl REST, AlphaFold DB). Their responses are frozen in
the archived caches and are read from there; a run without the caches fetches current
versions, which may differ.

## 6. Smoke test

```bash
bash revision/reproduce_smoke.sh                     # inputs, tests, Δα tables, gene identity
RUN_CHECKPOINT=1 bash revision/reproduce_smoke.sh    # also reload one checkpoint on CPU
RUN_FIGURE=1 bash revision/reproduce_smoke.sh        # also execute 03_alpha_master
```

The script exits 0 only if every check passes.

## 7. From raw data to the processed files

[`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) builds the
processed AnnData files with the functions in `chimera/_data_utils.py`, from the CAMEX bundle
(figshare doi:10.6084/m9.figshare.31131808.v1) and the SATURN ESM-2 embeddings. Use the
archived ortholog tables rather than querying BioMart again: the tables are from Ensembl
release 110, and if the release-110 archive does not answer, the query falls back to the
current Ensembl release without an error.

## 8. Revision analyses

Analyses added during the revision live in [`revision/`](revision/README.md), one folder per
task, with outputs in `results/revision/<task>/`.
