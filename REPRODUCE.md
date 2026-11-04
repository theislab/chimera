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

Create `chimera_env` either with [`notebooks/00_environment.ipynb`](notebooks/00_environment.ipynb)
or, on Linux x86-64, from the exact package versions of the lock file:

```bash
bash revision/envs/build_env_from_lock.sh chimera_env   # CPU torch, pinned versions, editable install
```

The script creates the environment with Python 3.12.16, installs every pin of
[`revision/envs/chimera_env.linux-64.lock.txt`](revision/envs/chimera_env.linux-64.lock.txt),
installs this checkout in editable mode and runs `pip check`. Check the installation with
`pytest -q`. Every command below runs inside the environment from the repository root.
Notebooks find the repository from their own location; set `CHIMERA_ROOT=/path/to/chimera`
to override.

## 2. Archived artefacts

Each bundle is one archive record (`[DOI]` lists them all) holding
`chimera_<bundle>.tar.gz`, `MANIFEST.tsv` (every member with its size and sha256) and
`SHA256SUMS`. Verify and unpack each bundle into the repository root; member paths are
relative to it:

```bash
sha256sum -c SHA256SUMS
tar -xzf chimera_<bundle>.tar.gz -C /path/to/chimera
```

| Bundle | Unpacks to | Content |
|---|---|---|
| `processed_data` | `data/processed/` | 7 cohort AnnData files, liver and testis ortholog tables (Ensembl 110), GIN feature scalers, build configs |
| `checkpoints_canonical` | `training/runs/<run>/` | The 11 canonical runs: `model/`, `final_state.npz`, alpha snapshots, metrics, config, summary, tables, caches |
| `checkpoints_revision` | `training/runs/rev-*/`, `training/runs/g1-*/` | Revision runs (added seeds, CPU platform and repeat runs, controls, recipe grids and seeds of every added cohort, latent-guard runs `rev-g1-*`): `model/`, `final_state.npz`, config, summary, metrics; for the latent-guard arm on thinned objects (`g1-*`) config, summary and final state |
| `checkpoints_free_gate` | `training/runs/z3-*/`, `z4-*/`, `rev-ctl-liv_hmouse-m-*/` | Per-gene (free) gate fits (`--config M`): every fit of liver Human+Mouse, liver 4-species and testis Human+Mouse, and the kept seeds of each harmonised cohort (`results/revision/Z4/kept.json`); same files as `checkpoints_revision` |
| `repaired_data` | `data/revision/O1/repair_*/`, `training/runs/o2-*/` | **Recommended for reuse:** liver and testis objects with the counts joined by Ensembl gene ID (build logs and repair lists beside them), and the models trained on them |
| `gene_tables` | `results/gene_tables/` | One CSV per canonical cohort (seed consensus): Ensembl gene ID, symbol, prior, raw prior, α, correction, Δα, D, S_g, sign consistency; seed reliability; per-gene tables of the 26 harmonised cohorts for both gates (`harmonised/`); column description in its `README.md` |
| `caches` | `notebooks/cache/`, `training/runs/baselines_*` | Notebook caches, including frozen external lookups, and the baseline runs |
| `processed_revision` | `data/revision/` | Objects of the cohorts added during revision (Van Zyl and its subsets, bone marrow, kidney, islet, cortex, lung, skeletal muscle, tumour myeloid cells, liver MASH pseudobulks and cell annotations), control objects, and the download manifests of every third-party input |
| `revision_results` | `results/revision/` | Outputs of the analyses added during revision, including the harmonised per-gene tables of both gates (`H1/genes/`, `Z4/h1/genes/`) and the exploratory result tables |
| `revision_competitors` | `results/revision/T5_*/` | SATURN and CAMEX runs of the benchmark |
| `benchmark` | `results/revision/B1/`, `B2/`, `training/runs/b1*-*/`, `b2-*/` | Gene-level benchmark (ground-truth spike-in, robustness, independent data, practical cost): scored tables, overview, figures; config, summary and final state of every scored fit (altered objects are rebuilt by the task code) |
| `plan_and_corrections` | `results/revision/_release/` | Pre-specified analysis plan with its amendment log; `before_after.csv` (numbers changed by the ortholog-identifier correction) |

`data/`, `training/runs/`, `notebooks/cache/` and `results/` are not tracked by git.

**Not redistributed.** Raw data stay with their sources (accessions in the paper and in the
`MANIFEST.tsv` files under `data/revision/`, which give URL, date and sha256 of every
download). The SATURN ESM-2 embeddings are not redistributed; the gene features derived from
them are inside the AnnData files. EVEE evolutionary statistics (Chen et al. 2019) carry no
stated licence, so per-gene EVEE columns are removed from the deposited tables; the H1/H2 gold
standards are rebuilt from the archived EVEE download listed in
`data/revision/gold/G1/MANIFEST.tsv`. SATURN macrogene-centroid initialisations and pre-training
snapshots are left out of the competitor bundle.

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
Canonical recipes run without the latent guard `--clamp-z-logvar` (off by default; see the README, "Training on a new cohort").
To train one again (the command is printed before it runs; `PRINT_ONLY=1` prints only):

```bash
bash revision/retrain_from_recipe.sh liv_hmouse 42 retrain-liv_hmouse-s42 cpu
# = python training/run_experiment.py --run-name <name> --config A --max-epochs <N> \
#       --adata data/processed/<h5ad> --seed <seed> --accelerator cpu <overrides>
```

On 16 CPU threads liver Human+Mouse takes 20-45 minutes depending on the node, and liver
4-species, testis Human+Mouse and testis 3-species about 50-55 minutes each. The canonical checkpoints were trained on Apple MPS and the revision seeds on
CPU. A retrained run does not reproduce an archived run value by value: on CPU the same seed
and configuration can end in a different training state (species mixing, adversary accuracy),
and Δα follows that state. How much this matters depends on the cohort: seed-pair Spearman of Δα
is 0.93-0.97 on liver 4-species and the testis cohorts but about 0.3 on liver Human+Mouse and
Human+Monkey (`results/gene_tables/reliability.csv`), and two same-seed CPU repeats on liver
Human+Mouse gave 0.10 and 0.63 (`results/revision/T6_8b_repeat.csv`). On liver Human+Mouse the
runs fall into two states: integrated (final adversary accuracy about 0.5) and poorly mixed
(adversary accuracy 0.74-0.93, expectations E6.1 and E6.2 failing, narrow Δα); Δα agrees within a
state far better than between states. Check `summary.json` (`adv_acc_final`,
`failed_expectations`) before comparing a retrained run with an archived one. The readouts of the paper
are therefore seed consensus tables (`results/gene_tables/`), and the archived runs remain the
reference.

The job templates in `revision/slurm/` were written for the authors' cluster: they default to
the authors' paths and to an environment named `chimera_env`, so set `CHIMERA_ROOT` to the
checkout before using them.

Per-gene values come from `training/runs/<run>/final_state.npz` (load with
`allow_pickle=False`): `delta_alpha = alpha - prior`. The seed consensus tables in
`results/gene_tables/` are means over the seeds listed in its `seed_pooling.json` (`used`);
seeds that are not canonical runs are `training/runs/rev-<cohort>-s<seed>/`. The file holds no gene names; take
`var_names` from the cohort AnnData and map them with the cohort's own table. Testis runs
were trained with the prior clamped at 0.85 (`prior` in `final_state.npz` is clamped); the
unclamped prior is `conservation_prior_raw.npy` in the same run directory.

**Harmonised cohorts, both gates.** The 26 harmonised cohorts have one per-gene table per gate,
in one schema: `results/revision/H1/genes/<cohort>.parquet` (canonical gate, the seeds kept in
`results/revision/H1/runs.tsv`) and `results/revision/Z4/h1/genes/<cohort>.parquet` (per-gene gate,
`--config M`, the seeds in `results/revision/Z4/h1/kept_used.json`); `.tsv.gz` copies are in
`results/gene_tables/harmonised/`. Columns are defined in
[`revision/tasks/H1_harmonise/rules.py`](revision/tasks/H1_harmonise/rules.py). Both are recomputed
from the archived states, without the AnnData files, by

```bash
python revision/check_gene_tables.py     # exit 0 when every table of both gates matches its runs
```

A per-gene gate fit uses the cohort's recipe with `--config A` replaced by `--config M`
(`CONFIG=M bash revision/retrain_from_recipe.sh <cohort> <seed> <run> cpu` for the seven canonical
cohorts; the argv of every fit is in `revision/tasks/Z3_free_gate/manifest.tsv` and
`revision/tasks/Z4_free_gate_all/manifest.tsv`). Load a saved per-gene gate model with
`revision.common.load_run.load_run(run_dir, adata)`: `ChimeraModel.load` rebuilds the canonical gate
only and refuses the extra `per_gene_correction` of these fits (the helper loads canonical runs too). The latent guard `--clamp-z-logvar` is a
per-cohort recipe flag for cohorts whose fits go non-finite; the runs that used it are named
`rev-g1-*` and `g1-*` and record it in `config.json` (`clamp_z_logvar`).

**Repaired objects.** `data/revision/O1/repair_liver/` and `repair_testis/` hold the liver and
testis cohorts with every count column taken from the raw row that carries the ortholog table's
Ensembl ID (`repair_log.csv` lists each changed column). They are the recommended objects for new
work on these cohorts; liver macaque is repaired only where its matrix carries Ensembl IDs, and
zebrafish not at all. The canonical models and tables of the paper were trained on the
unrepaired objects (`processed_data`); the models trained on the repaired objects are
`training/runs/o2-*`.

## 5. Figures

Each notebook renders its panels into `notebooks/figures/<notebook>/`. Run a notebook
headless with

```bash
papermill notebooks/<notebook>.ipynb notebooks/<notebook>.ipynb --cwd notebooks -k python3
```

`-k python3` is the kernel of the active environment; `00_environment.ipynb` also registers it
as `chimera_env`, the lock-file script does not.

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
bash revision/reproduce_smoke.sh                     # inputs, tests, Δα tables, gene identity, gene tables
RUN_CHECKPOINT=1 bash revision/reproduce_smoke.sh    # also reload one checkpoint on CPU
RUN_FIGURE=1 bash revision/reproduce_smoke.sh        # also execute 03_alpha_master
```

The script exits 0 only if every check passes. With `results/gene_tables/` present it
recomputes every column of the seven consensus tables from the archived `final_state.npz` files
(this needs the `checkpoints_revision` bundle too). `RUN_FIGURE=1` executes the notebook with
the `python3` kernel of the active environment (`KERNEL=<name>` to change it) and writes its
panels into `notebooks/figures/alpha_master/` of the checkout, so `git status` shows which
panels were re-rendered. The re-rendered panels carry the same values as the archived ones but are not
pixel-identical: the archived panels were rendered on macOS, and fonts and antialiasing differ
between platforms.

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
