# Chimera

**Cross-species Harmonization via Integrated Mixing of Evolutionary Residual Alphas**

When a protein is conserved across species, is the gene also used the same way in cells?
Chimera is a variational autoencoder, built on [scvi-tools](https://scvi-tools.org/), that
aligns single-cell RNA-seq data across species and returns, for every one-to-one ortholog, two
gene-level readouts from the same fit:

- **the count evidence S_g**: how much better one cell-state pattern shared by all species
  explains the gene's counts than a separate average per species. It rewards a shared pattern
  and is lowered by a gap in expression level between species.
- **Δα**: where the gene's deployment sits relative to a protein-identity prior. Δα above zero:
  the gene is used more alike across species than its protein identity suggests; below zero:
  less alike.

| You want | Go to |
|---|---|
| to train Chimera on your own cohort | [Quick start](#quick-start-your-own-cohort) |
| the per-gene tables of the paper | [`REPRODUCE.md`](REPRODUCE.md), section 3 (bundle `gene_tables`) |
| the code and data behind a figure, table or number of the paper | [`REPRODUCE.md`](REPRODUCE.md), section 4 |
| the whole chain, from the raw data to the figures | [`REPRODUCE.md`](REPRODUCE.md), section 1 (the chain) and section 8 (raw data to objects) |
| to find your way in the repository | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## What the model does

Cells of all species are encoded into one latent space. The decoder predicts each gene's
expression through two heads that are always combined:

```
log rate = shared head (sees no species label) + (1 - α_g) × residual head (species-aware)
```

- The **shared head** must explain a gene in every species with one cell-state pattern.
- The **residual head** adds a species-specific correction, scaled per gene by `1 - α_g`.
- The **gate** `α_g = sigmoid(logit(π_g) + c_g)` starts at the **protein prior** `π_g`, the
  protein identity of the gene's orthologs, and moves only under count evidence: its target is
  S_g, the log-likelihood gain of the shared head over a per-species average. `Δα_g = α_g - π_g`.

The correction `c_g` is read in two ways:

| Gate | Option | What sets `c_g` | Use |
|---|---|---|---|
| **protein-feature gate** (default) | `--config A` | a small network, the Gene Integration Network (GIN), on 72 gene features: 64 principal components of ESM-2 protein embeddings, ESM-2 variance between species, protein identity and its spread, three expression statistics | places every gene through its protein neighborhood, genes with too few counts included (their values are imputed from features) |
| **per-gene gate** (companion) | `--config M` | one free offset per gene | follows each gene's own counts; no protein-feature network |

The loss adds four structural terms to the usual reconstruction and KL terms, each closing one
way in which a gated decoder fails silently: a sparsity penalty on the residual head (otherwise
it fits everything and α falls to zero), a reconstruction loss of the shared head alone
(otherwise the shared head receives no gradient), an anchor of α to the prior (otherwise α of
divergent genes collapses), and a fit-quality reward driven by S_g (otherwise α can only move
down). Species information is removed from the latent space by an adversarial classifier and a
maximum-mean-discrepancy term. The paper's Methods and Supplement derive each term.

What to keep in mind when reading the outputs (the paper's Results and Discussion give the
evidence):

- Genes with too few counts have S_g = 0 ("masked"). With the protein-feature gate their Δα is
  imputed from protein and expression features; read it as such.
- Δα is read within a cohort (centered on the cohort median, averaged over seeds). How well it
  repeats across seeds differs between cohorts; check the training state of every fit.
- S_g and Δα reward shared pattern. Where cell-type labels are rich, direct comparisons of
  pseudobulk profiles match or beat them; S_g needs no labels.

## Installation

Python 3.12 was used for all results.

```bash
git clone https://github.com/theislab/chimera.git && cd chimera
mamba create -n chimera_env python=3.12 pip
mamba activate chimera_env
pip install torch==2.11.0
pip install -e ".[dev,graph,attributions,protein,evolution,data,baselines]"
pytest -q
```

Two other routes build the same environment: the notebook
[`notebooks/00_environment.ipynb`](notebooks/00_environment.ipynb) (any platform; picks the
PyTorch build for CPU, CUDA or Apple MPS and registers a Jupyter kernel), and, on Linux x86-64,
`bash analysis/envs/build_env_from_lock.sh chimera_env`, which installs every package at the
exact version of the lock file. Core dependencies are pinned in
[`pyproject.toml`](pyproject.toml) to the versions the archived models were trained with.

## Quick start: your own cohort

**1. Build the cohort object.** Chimera reads one AnnData file with all species:

| Field | Content |
|---|---|
| `X` | raw integer counts, cells × one-to-one ortholog groups |
| `obs['species']` | species of each cell |
| `obs['batch']` | data set or chemistry within a species (see below) |
| `var['conservation_prior']` | protein prior π_g in (0, 1) |
| `varm['gene_features']` | the 72 standardized gene features |
| `uns['p_null']` | per-species average expression proportions |

The functions that build it are in [`chimera/_data_utils.py`](chimera/_data_utils.py)
(`query_ortholog_table`, `compute_conservation_prior`, `pca_esm2`,
`compute_esm2_interspecies_variance`, `compute_expression_stats`, `assemble_gin_features`,
`construct_unified_adata`, `precompute_p_null`). Two worked examples:
[`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) (liver and testis)
and [`analysis/tasks/C1_cohorts/build.py`](analysis/tasks/C1_cohorts/build.py) (cohorts from
CELLxGENE and GEO, one configuration per cohort). Then check the object:

```bash
python -m analysis.common.check_cohort my_cohort.h5ad --batch-key batch
```

**Batch key.** Use a batch column with few levels per species (the source data set or the
chemistry). Sample- or donor-level batches made training fail in our cohorts.

**2. Train.** Start from the canonical recipe closest to your cohort
([`configs/canonical_recipes.tsv`](configs/canonical_recipes.tsv); two species at moderate
divergence: the liver Human+Mouse row):

```bash
python training/run_experiment.py --run-name my-cohort-s42 --adata my_cohort.h5ad \
    --config A --max-epochs 150 --adv-weight 200 --free-bits 1.0 --seed 42 --accelerator cpu
```

`--accelerator` is `cpu`, `gpu` or `mps` (the default). `--config M` trains the per-gene gate
with the same recipe. `python training/run_experiment.py --help` lists every option. The run is
written to `training/runs/my-cohort-s42/`:

| File | Content |
|---|---|
| `final_state.npz` | per gene: `alpha`, `prior`, `correction`, `S_g_ema` (the count evidence); per cell: `z_mean` (the latent) |
| `summary.json`, `report.md` | training health: which expectations passed (`failed_expectations`), key metrics |
| `model/` | the saved model |
| `config.json` | every setting of the run |

**3. Check the fit** before reading it: `failed_expectations` in `summary.json`, the final
adversary accuracy (near chance when species are mixed), and that Δα is not flat (near zero for
almost every gene). Train at least three seeds; the paper reads the mean over seeds.
`--clamp-z-logvar` bounds the log-variance of the sampled latent; it is off by default and in
every canonical model. Use it only if a fit becomes non-finite. It keeps such fits finite but
does not cure a flat gate.

**4. Read the gene table.**

```python
import anndata as ad, numpy as np, pandas as pd

adata = ad.read_h5ad("my_cohort.h5ad", backed="r")
fs = np.load("training/runs/my-cohort-s42/final_state.npz", allow_pickle=False)
genes = pd.DataFrame(
    {"prior": fs["prior"], "alpha": fs["alpha"], "delta_alpha": fs["alpha"] - fs["prior"], "S_g": fs["S_g_ema"]},
    index=adata.var_names,
)
genes["count_evidence"] = np.abs(genes["S_g"]) >= 1e-6     # False: masked, Δα imputed from features
genes["D"] = genes["delta_alpha"] - genes["delta_alpha"].median()   # centered within the cohort
```

A saved model is reloaded with
`ChimeraModel.load("training/runs/my-cohort-s42/model", adata=adata, accelerator="cpu")`
(per-gene-gate runs: `analysis.common.load_run.load_run(run_dir, adata)`); `get_alpha()`,
`get_delta_alpha()`, `get_S_g()` and `get_latent_representation()` return the readouts and the
latent.

## Where everything is

| Path | Holds |
|---|---|
| [`chimera/`](chimera/) | the package: model (`_model.py`), module and loss (`_module.py`), gate network and other components (`_components.py`), training plan, data preparation (`_data_utils.py`), diagnostics, constants |
| [`configs/`](configs/) | `canonical_recipes.tsv` (the recipe of every canonical model), `optuna_search_space.json` |
| [`training/`](training/README.md) | `run_experiment.py` (train, evaluate, summarize), baselines, diagnostics; `runs/` holds trained models (not in git) |
| [`notebooks/`](notebooks/README.md) | the workflow on the liver and testis cohorts: environment, data preparation, baselines and the first analyses of the gate (notebooks 00 to 11), with their builders, stored outputs and caches; the guide says what each still produces |
| [`analysis/`](analysis/README.md) | the analyses of the paper, one folder per analysis, with an index; shared helpers; environments; job templates |
| [`figures/`](figures/README.md) | every figure of the paper: panels, specs, the build, one PDF with one figure per page, an index |
| [`release/`](release/README.md) | how a release is staged for Zenodo and tested from a clean clone |
| [`scripts/`](scripts/README.md) | stand-alone scripts of the first version of the model, kept as starting points (no analysis of the paper uses them; none was run again for the release): ablations, hyperparameter search, robustness, demonstrations |
| [`tests/`](tests/) | the test suite (`pytest -q`) |
| `data/`, `results/`, `training/runs/`, `notebooks/cache/` | data, results, models and caches; not in git, filled from the deposit ([`REPRODUCE.md`](REPRODUCE.md), section 3) |

## Data

All analyzed data are public; no new data were generated. The canonical cohorts:

| Data | Source |
|---|---|
| Liver: human, macaque, mouse | Guilliams et al. 2022, *Cell* 185:379, GEO GSE192742 |
| Liver: zebrafish | Wang et al. 2023, *Nucleic Acids Res.*, doi:10.1093/nar/gkac633, GEO GSE198832 |
| Testis: human, macaque, mouse | Murat et al. 2023, *Nature*, doi:10.1038/s41586-022-05547-7, E-MTAB-11063, E-MTAB-11068, E-MTAB-11071 |
| Both, as processed counts | CAMEX bundle, Guo, Huang and Zhang 2026, *Nat. Commun.* 17:3017, archived at figshare doi:10.6084/m9.figshare.31131808.v1; the data preparation downloads it from the CAMEX authors' Google Drive folder ([`REPRODUCE.md`](REPRODUCE.md), section 8) |
| ESM-2 protein embeddings (`esm2_t48_15B_UR50D`) | SATURN, Rosen et al. 2024, *Nat. Methods* (archive `protein_embeddings.tar.gz` at snap.stanford.edu) |
| Ortholog tables | Ensembl BioMart, release 110 |

The sources of the other cohorts (bone marrow, kidney, pancreatic islet, eye, cerebral cortex,
lung, skeletal muscle, tumor myeloid cells) and of every reference are listed in the paper and
in the `MANIFEST.tsv` files of the deposit; how each object is built from them is in
[`REPRODUCE.md`](REPRODUCE.md), section 8. Processed objects, ortholog tables, trained models
and per-gene tables are archived at Zenodo, one record per bundle:
[`REPRODUCE.md`](REPRODUCE.md), section 3, gives the DOI of each, and the code archive of
release `v0.1.0`, [10.5281/zenodo.24137767](https://doi.org/10.5281/zenodo.24137767), links them
all. In file names, `panc_*` and `pancreas_*` mean the **liver** cohorts.

## Reproducing the paper

[`REPRODUCE.md`](REPRODUCE.md) is the guide: what can be reproduced at which cost, the deposit,
a map from every figure, table and section to the code behind it, the names used in files, and
the tests of a release (`bash analysis/reproduce_smoke.sh`).

## Citation

If you use Chimera, please cite:

Inecik K, Sevinc ES, Erken E, Theis FJ. Chimera: Cross-Species Single-Cell Generative Modeling of
Expression Deployment Relative to Protein Conservation. *bioRxiv* 2026.11.11.766029 (2026).
[doi:10.64898/2026.11.11.766029](https://doi.org/10.64898/2026.11.11.766029)

```bibtex
@article{inecik2026chimera,
  title   = {Chimera: Cross-Species Single-Cell Generative Modeling of Expression Deployment Relative to Protein Conservation},
  author  = {Inecik, Kemal and Sevinc, Eylul Sevinc and Erken, Erkmen and Theis, Fabian J.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.11.11.766029},
  url     = {https://www.biorxiv.org/content/10.64898/2026.11.11.766029v1}
}
```

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International); see [LICENSE](LICENSE). The records
of the deposit are released under the same license; third-party sources keep their own terms
([`release/zenodo/LICENSES.md`](release/zenodo/LICENSES.md)). Chimera follows the conventions of
[scvi-tools](https://scvi-tools.org/) for model registration, AnnData handling and training
plans.
