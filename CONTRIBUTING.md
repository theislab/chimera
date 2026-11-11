# Working in this repository

A map for new contributors and for anyone who has to find something quickly: where things are,
which rules must not be broken, and how to check that nothing broke. For what Chimera is, see
[`README.md`](README.md); for reproducing the paper, [`REPRODUCE.md`](REPRODUCE.md).

## Directory map

| Path | One line |
|---|---|
| `chimera/` | The installable package. `_model.py` the scvi-tools model class, `_module.py` the network and the loss, `_components.py` the gate network and other parts, `_training_plan.py` the adversarial training plan, `_data_utils.py` data preparation, `_diagnostics.py` checks, `_constants.py` defaults and named configurations. |
| `configs/` | `canonical_recipes.tsv`: the recipe of each canonical model. `optuna_search_space.json`: the default search space. |
| `training/` | `run_experiment.py` and the scripts it calls; baselines; diagnostics; the record of the model's tuning. [`training/README.md`](training/README.md). `training/runs/` holds trained models (not in git). |
| `notebooks/` | Workflow notebooks `00` to `11` on the liver and testis cohorts: environment, data preparation, baselines, the first analyses of the gate. `_*_builder.py` hold their cells; `_figure_design.py`, `_paper_utils.py`, `_gene_categories*.py`, `_pipeline_cache.py` are shared helpers, also read by tasks and figure scripts; `figures/` the panels as first rendered (tracked); `cache/` (not in git). [`notebooks/README.md`](notebooks/README.md) says what each notebook does and which of its outputs are read today. |
| `analysis/` | The analyses of the paper. `tasks/<folder>/` one analysis each; `tasks/INDEX.tsv` the index; `common/` shared helpers; `envs/` environments; `slurm/` job templates; `external_tools/` third-party tools. [`analysis/README.md`](analysis/README.md). |
| `figures/` | Every figure of the paper: `panels/`, `specs/`, `compose.py`, `build.sh`, `figures.pdf`, `INDEX.md`. [`figures/README.md`](figures/README.md). |
| `release/` | Staging the Zenodo deposit and the clean-clone test. [`release/README.md`](release/README.md). |
| `scripts/` | Stand-alone scripts of the first version of the model, kept as starting points; no analysis of the paper uses them. [`scripts/README.md`](scripts/README.md). |
| `tests/` | The test suite. |
| `data/`, `results/`, `training/runs/`, `notebooks/cache/` | Storage. Ignored by git; filled from the deposit or by running code. Never commit anything from them. |

## Where do I find ...

| I am looking for | It is in |
|---|---|
| the model class (setup, train, save, load, readouts) | `chimera/_model.py`, class `ChimeraModel` |
| the forward pass: shared head, residual head, `(1 - α)` | `chimera/_module.py`, `ChimeraModule.forward` |
| the loss and its terms | `chimera/_module.py`, `ChimeraModule.loss`; weights in `chimera/_constants.py` |
| the count evidence S_g | `chimera/_module.py`, `_compute_S_g`, `_update_S_g_ema`; read with `ChimeraModel.get_S_g()` |
| the gate: protein-feature gate | `chimera/_components.py`, `GeneIntegrationNetwork` |
| the gate: per-gene gate | configuration `M` in `chimera/_constants.py`; the parameter `per_gene_correction` of the module |
| the adversary and the training loop | `chimera/_training_plan.py`, `ChimeraTrainingPlan` |
| the training entry point | `training/run_experiment.py` (`--help` lists every option) |
| the recipes | `configs/canonical_recipes.tsv` (canonical); the `manifest.tsv` of the task that launched a run (all others); `config.json` in every run folder |
| how a cohort object is built from raw data | the table of `REPRODUCE.md`, section 8: `notebooks/01_data_preparation.ipynb` (liver, testis); `analysis/tasks/C1_cohorts/build.py` and the `build*.py` of `T11_1_vanzyl`, `T12_1_bone_marrow`, `P13_kidney`, `T7_9_islet` |
| a cohort's object and ortholog table | `analysis/common/gene_ids.py`: `h5ad_path(cohort)`, `load_table(cohort)`; liver and testis in `data/processed/`, the others under `data/analysis/` |
| a run's outputs | `training/runs/<run>/` (`final_state.npz`, `summary.json`, `config.json`, `model/`); run-name families in `REPRODUCE.md`, section 5 |
| per-gene tables | `results/gene_tables/` (seven canonical cohorts, seed consensus); `results/analysis/H1/genes/` (26 cohorts, protein-feature gate); `results/analysis/Z4/h1/genes/` (per-gene gate); columns in `analysis/tasks/H1_harmonise/rules.py` |
| the analysis behind a figure, table or section | the map in `REPRODUCE.md`, section 4.1, then the task folder's `README.md` |
| what a task folder does | `analysis/README.md` (all tasks by question) or `analysis/tasks/<folder>/README.md` |
| a figure's panels and layout | `figures/INDEX.md` (panel files per figure), `figures/specs/*.py` (layout), `figures/panels/<family>/` |
| the script that draws a panel | `figures/producers.tsv` (panel file, script, command) and, under each figure, `figures/INDEX.md`. The script is in the analysis task of the panel's topic (`analysis/tasks/*/fig*.py`, `figures.py`, `panels.py`) or in `figures/redraw/`; a notebook builder only for the few panels a figure places from `notebooks/figures/` |
| what a notebook does, and whether its outputs are still read | `notebooks/README.md` |
| what a later analysis replaced | status `retired, replaced by` in `analysis/README.md`; "Retired from the paper" in `REPRODUCE.md`, section 4.1; the last column of the table in `notebooks/README.md`; "Record of the model's development" in `training/README.md`; `scripts/README.md` |
| the paper's numbers | result tables under `results/analysis/<task>/` (CSV, parquet), named in each task README; per-gene values as above |
| the statistics used throughout (partial rank correlation, block bootstrap) | `analysis/tasks/T4_3_scoreboard/stats.py` |
| the deposit: what is in which bundle | `REPRODUCE.md`, section 3; `release/zenodo/bundles.yaml` |
| third-party tools and their versions | `analysis/external_tools/README.md`, `analysis/envs/` |

## Rules that must not be broken

1. **Gene identity.** Gene names in the objects are `ortho_N`, local to one ortholog table.
   Never join two cohorts of different tissues on `ortho_N` or on `var_names`. Look genes up
   through `analysis/common/gene_ids.py` with the cohort named, and join across tissues on the
   human Ensembl gene ID (`hsapiens_gene_id`). `tests/test_gene_identity.py` guards this.
2. **Batch key.** The batch of a cohort is the source data set (or chemistry) within a species,
   never the donor or sample; at most three batches per species; every batch belongs to one
   species. `python -m analysis.common.check_cohort <h5ad> --batch-key <batch column>` checks a new
   object (without `--batch-key` it reads the column `species`): raw integer
   counts in `X`, no layers, 72 feature columns, `p_null`. Run it before training.
3. **Archived things keep their names.** Task folders, result folders, run names and file names
   such as `camex_pancreas_unified.h5ad` (liver) or `harmonised/` are identifiers that archived
   configurations, tables and manifests carry. Do not rename them; explain them (`REPRODUCE.md`,
   section 5).
4. **Runs are never written over.** A training run goes to a new folder under `training/runs/`.
   Analyses read archived runs and tables; they do not retrain canonical models.
5. **Environment.** All Python runs in `chimera_env`. A dependency changes in
   [`pyproject.toml`](pyproject.toml), then the environment is rebuilt and the lock file
   `analysis/envs/chimera_env.linux-64.lock.txt` regenerated; no ad-hoc `pip install`. Tools
   with conflicting dependencies get their own environment file in `analysis/envs/`.
6. **Model code and saved models.** A change in `chimera/` must leave the archived models
   loadable and their stored readouts reproducible (`RUN_CHECKPOINT=1 bash analysis/reproduce_smoke.sh`).
7. **No machine paths in tracked files.** Scripts find the repository from their own location or
   from `CHIMERA_ROOT`; job templates take `mamba` from `PATH`. Notebook outputs are cleaned
   with `python notebooks/_scrub_output_paths.py`.
8. **Generated files are not edited by hand.** The task tables of `analysis/README.md`, the map
   in `REPRODUCE.md` and the task READMEs come from `analysis/tasks/INDEX.tsv`
   (`python analysis/make_index.py`); `figures/INDEX.md`, the manifest and `figures.pdf` from
   `bash figures/build.sh`; the notebooks `03` to `11` from their builders (four of them hold
   cells that their builder lacks and must not be regenerated yet: `notebooks/README.md`); the bundle table of
   `REPRODUCE.md` from `python release/zenodo/stage.py --bundle-table`.
9. **Pre-specified rules stay as written.** Files that fix a rule before results were read
   (`rules.py`, `PREDICTIONS.md`, the recipe READMEs, `grid.tsv`) are a record; add an amendment
   rather than rewriting them.

## Checks

Run from the repository root, inside `chimera_env`.

| Check | Command | Passes when |
|---|---|---|
| Unit and integration tests | `pytest -q` | all tests pass (about 2 minutes) |
| Style of changed Python files | `ruff check <files>` | no finding (settings in `pyproject.toml`) |
| Index of the analyses | `python analysis/make_index.py --check` | every task folder has a row, every named script exists, generated files are current |
| Per-gene tables against the archived models | `python analysis/check_gene_tables.py` | exit 0 (needs the deposit) |
| Smoke test of a checkout with the deposit | `bash analysis/reproduce_smoke.sh` | exit 0 (`RUN_CHECKPOINT=1` also reloads a model) |
| Figures current | `bash figures/build.sh --check` | exit 0 (needs `chimera_figures_env`) |
| Notebook outputs free of machine paths | `python notebooks/_scrub_output_paths.py --check` | exit 0 |
| Staged deposit current | `python release/zenodo/stage.py --check` | exit 0 |
| Release reproduces from a clean clone | `bash release/clean_clone_test.sh --fresh all` | exit 0 |

## Adding things

- **An analysis.** A new folder `analysis/tasks/<code>_<slug>/`; outputs under
  `results/analysis/<code>/` (use `analysis/common/outputs.py` for atomic outputs with a
  provenance record); one row in `analysis/tasks/INDEX.tsv`; `python analysis/make_index.py`.
- **A cohort.** Build the object (the table of `REPRODUCE.md`, section 8, names the existing
  builders), run `check_cohort` with `--batch-key`, register its ortholog table and object in
  `analysis/common/gene_ids.py`, choose the recipe by a rule written down before training
  (see `analysis/tasks/T11_2_recipe/README.md` for the form). To bring it into the tables of the
  paper: add it to the registry `COHORTS` of `analysis/tasks/H1_harmonise/rules.py` and run
  `bash analysis/tasks/H1_harmonise/run_all.sh`; write the manifest of its per-gene-gate fits
  with `python -m analysis.tasks.Z4_free_gate_all.make_manifest` and train them; run
  `python analysis/check_gene_tables.py`; add the cohort key to `REPRODUCE.md`, sections 5 and
  8; assign the new data folder and run family in `release/zenodo/bundles.yaml` and run
  `python release/zenodo/stage.py --check`.
- **A training run on a cluster.** One row in a `manifest.tsv` (`run name<TAB>arguments`), then
  `sbatch --array=<rows> analysis/slurm/array.sbatch <manifest.tsv>`.
- **A figure or a panel.** [`figures/README.md`](figures/README.md), "Changing a figure".
- **Something for the deposit.** One line in `release/zenodo/bundles.yaml`
  ([`release/zenodo/README.md`](release/zenodo/README.md)).

## Common changes, step by step

- **Redraw a panel with other inputs** (another gene list, another cohort). Find the panel in
  [`figures/producers.tsv`](figures/producers.tsv): panel file, script, command. The lists and
  constants of a panel stand at the top of its script or function (for example `ENZYMES` in
  `analysis/tasks/T8_4_figures/fig4_additions.py`). Edit, run the command of that row inside
  `chimera_env` with `PYTHONPATH=$PWD`, then `bash figures/build.sh`. If the panel now shows
  something else, change the `shows=` text of its figure in `figures/specs/` and the caption in
  the paper. Checks: `bash figures/build.sh --check`, `ruff check <file>`.
- **Add a cohort.** The steps under "Adding things".
- **Rerun the gene-level benchmark.** `analysis/tasks/B1_ground_truth/README.md` and
  `analysis/tasks/B2_benchmark/README.md` list the steps, the environments, the inputs and the
  bundles; the compared tools and their environments are in
  `analysis/external_tools/README.md`. The fits are submitted by the feeder of each task on a
  SLURM cluster: pass your partition and QOS (`analysis/README.md`, "Running a task"). The
  altered objects are rebuilt by the task code (about 95 GB). `results/analysis/B1/completion.csv`
  and `results/analysis/B2/completion.csv` count every run that was tried and finished.
- **Rebuild the deposit after a result changed.** `python release/zenodo/stage.py --check` names
  what is stale; `python release/zenodo/stage.py` rebuilds only that; a new storage folder or run
  family gets one line in `release/zenodo/bundles.yaml`. Before an upload:
  `python release/zenodo/stage.py --check --deep` and
  `bash release/clean_clone_test.sh --fresh all` ([`release/README.md`](release/README.md)).
- **Change the model code** (`chimera/`). Rule 6 above; then `pytest -q`,
  `RUN_CHECKPOINT=1 bash analysis/reproduce_smoke.sh` and `python analysis/check_gene_tables.py`.

## Names

Cohort keys, run-name families, readout columns and the names that differ between code and
paper (`panc_*` means liver; "canonical" is the protein-feature gate, "free" the per-gene gate)
are listed once, in [`REPRODUCE.md`](REPRODUCE.md), section 5. Folder-name prefixes of the
analyses are explained in [`analysis/README.md`](analysis/README.md).
