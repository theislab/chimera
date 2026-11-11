# Changelog

All notable changes of the `chimera` package and of this repository, for users. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## 0.1.0 (first release; the date is set when the release is tagged)

The version that produced the results of the paper.

### Package

- `ChimeraModel` (scvi-tools model): `setup_anndata`, `train`, `save`, `load`, `from_config`;
  readouts `get_alpha`, `get_delta_alpha`, `get_correction`, `get_S_g`,
  `get_latent_representation`, `get_expression`, `get_shared_only_expression`;
  `evaluate_holdout`.
- Two gates, chosen by configuration: the protein-feature gate (`A`, the default) and the
  per-gene gate (`M`, one free offset per gene). The other named configurations of
  `chimera/_constants.py` are ablations.
- Data preparation in `chimera/_data_utils.py`: ortholog table, protein prior, ESM-2 features,
  expression statistics, the unified multi-species object, `p_null`.
- Diagnostics in `chimera/_diagnostics.py`.

### Training

- `training/run_experiment.py` trains, evaluates the training expectations and writes
  `summary.json` and `final_state.npz` per run.
- `--seed` fixes the train/validation split as well as the model seed, so a fit repeats exactly
  with the same seed on one CPU thread.
- `--clamp-z-logvar` bounds the log-variance of the sampled latent. Off by default; no archived
  canonical model uses it.
- Canonical recipes in `configs/canonical_recipes.tsv`; `analysis/retrain_from_recipe.sh`
  builds the training command from them.

### Version numbers

- "v2.4" in comments and docstrings of the package, and in the default name of the search study
  (`chimera_v2.4_hpo`), is the number of the design that this release implements. The package
  version is 0.1.0.

### Known limitations

- `ChimeraModel.load` does not load models of the per-gene gate (they carry one extra parameter
  per gene). Use `analysis.common.load_run.load_run(run_dir, adata)`, which loads both kinds.
- Training on several CPU threads or on another platform is not deterministic at a fixed seed
  (`REPRODUCE.md`, section 1).

### Retired from the paper

Kept in the repository, not shown in the paper, because the paper shows a later analysis of the
same question on the same data (`REPRODUCE.md`, section 4.1, lists the code of each):

- The integration scorecard among seven methods; the paper shows the benchmark of ten methods
  (`analysis/tasks/T5_5_benchmark`).
- The raw integration metrics among seven methods; the paper shows them for ten methods.
- Δα against detection rate, mean count and count variance on liver Human+Monkey; the paper
  shows the covariate audit of the seven canonical cohorts (`analysis/tasks/T2_4_covariates`).
- Program means of Δα, liver against testis, against the clamped prior; the paper shows the
  tissue comparison against the raw prior (`analysis/tasks/T2_3_tissue`).
- Sign classes of Δα, liver against testis, against the clamped prior; the paper shows the same
  comparison of `T2_3_tissue`.

Retired in the repository: the first recipe grids of bone marrow and islet
(`analysis/tasks/T12_2_recipe`, `T7_9_recipe`; status "retired, replaced by" in
`analysis/README.md`). The stand-alone scripts of `scripts/` are kept as starting points; no
analysis of the paper uses them.

### Repository

- `analysis/`: the analyses of the paper, one folder per analysis, with an index
  (`analysis/README.md`) generated from `analysis/tasks/INDEX.tsv`.
- `figures/`: every figure of the paper, composed from panel files into one PDF.
- `release/`: staging of the Zenodo deposit and the clean-clone test.
- `REPRODUCE.md`: the guide from the paper to code and data. `CONTRIBUTING.md`: where things
  are and the rules of the repository.
- Dependencies pinned in `pyproject.toml` to the versions the archived models were trained
  with; exact Linux environment in `analysis/envs/chimera_env.linux-64.lock.txt`.
