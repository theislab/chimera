# Analyses

The analyses of the paper that read the trained models: validation against independent data,
controls, benchmarks, the cohorts beyond liver and testis, and the disease case studies. One
folder per analysis under [`tasks/`](tasks/). Nothing here changes the `chimera` package. The
figure workflow of the model itself is in [`../notebooks/`](../notebooks/); figures are composed
in [`../figures/`](../figures/README.md).

To reproduce a result, start from [`../REPRODUCE.md`](../REPRODUCE.md): it maps every figure,
table and section of the paper to the folders listed here.

## What is in this folder

| Path | Holds |
|---|---|
| [`tasks/`](tasks/) | One folder per analysis (93). Each has a `README.md`. |
| [`tasks/INDEX.tsv`](tasks/INDEX.tsv) | The index: one row per task folder. The tables below and the folder READMEs are generated from it. |
| [`tasks/PAPER.tsv`](tasks/PAPER.tsv) | Figures, tables and sections of the paper by LaTeX label, with their printed number and title. |
| [`make_index.py`](make_index.py) | Writes the tables below, the paper map of `REPRODUCE.md` and the folder READMEs from the two tables; `--check` verifies them. |
| [`common/`](common/) | Shared helpers (table below). |
| [`check_gene_tables.py`](check_gene_tables.py) | Recomputes the per-gene tables of the 26 cohorts, both gates, from the archived model states. |
| [`reproduce_smoke.sh`](reproduce_smoke.sh) | The smoke test of a checkout with the deposit unpacked (see `REPRODUCE.md`). |
| [`retrain_from_recipe.sh`](retrain_from_recipe.sh) | Trains one canonical recipe again from `configs/canonical_recipes.tsv`. |
| [`envs/`](envs/) | The lock file of `chimera_env`, `build_env_from_lock.sh`, and the environment files of the tools that are compared. |
| [`external_tools/`](external_tools/README.md) | Which third-party tools the analyses run, at which commit, and where they are fetched. |
| [`slurm/`](slurm/) | Job templates for a SLURM cluster: `cpu.sbatch` (one command), `array.sbatch` (one training run per row of a manifest), `_body.sh` (shared). |

### `common/`

| Module | Use |
|---|---|
| `gene_ids.py` | Gene identity. `ortho_N` names are local to an ortholog table, so every lookup names its cohort; tables are joined across tissues on the human Ensembl gene ID. Also `REPO_ROOT` and the path of every cohort object (`h5ad_path`). |
| `cohorts.py` | The canonical runs, reading `final_state.npz`, the raw prior, per-gene covariates. |
| `check_cohort.py` | Checks a new cohort object before training (counts, batch key, cell types per species): `python -m analysis.common.check_cohort <h5ad>`. |
| `load_run.py` | Loads a saved model of either gate (`load_run(run_dir, adata)`), and the state of a run without the model. |
| `controls.py` | Shared statistics of the retraining controls: seed consensus, covariate residuals, seed floor. |
| `cds.py` | Human coding-sequence length and GC content from the Ensembl 110 files. |
| `depth_anchor.py` | Anchor weight as a function of sequencing depth (used by the depth experiments). |
| `outputs.py` | Output folder of a task: files are written under a temporary name and renamed, with a provenance record and `DONE` last. |

## Running a task

- Run from the repository root, inside the environment named in the folder README (`chimera_env`
  unless stated).
- `python -m analysis.tasks.<folder>.<script>` works as it is. A command that names a file
  (`python analysis/tasks/<folder>/<script>.py`) needs the repository root on the module path:
  `export PYTHONPATH=$PWD` once.
- Tasks read archived models and cached tables; they do not retrain. Outputs go to
  `results/analysis/<name>/`, which git ignores. Scripts that write through `common/outputs.py`
  skip work whose output exists (`DONE`) and take `--force` and `--outdir`.
- Rows that train models are manifests: one run per line, `run name<TAB>arguments of
  training/run_experiment.py`. On a SLURM cluster:
  `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch <manifest.tsv>`. Without a cluster, one
  row at a time: `SLURM_ARRAY_TASK_ID=<row> bash analysis/slurm/array.sbatch <manifest.tsv>`
  (add `PRINT_ONLY=1` to print the training command only).
- One file in `tasks/` is not a folder: [`tasks/run_T2_3_to_T2_7.sh`](tasks/run_T2_3_to_T2_7.sh)
  runs the five tasks `T2_3` to `T2_7` in sequence.

## Folder names

Folder names are identifiers, not descriptions. They are kept because archived run
configurations, result tables and manifests carry them. The letter says which family a task
belongs to, the digits its place in that family.

| Prefix | Family |
|---|---|
| `T1` to `T8` | The analyses of the analysis plan, numbered by stage: `T1` environment, `T2` gene identity, tissue comparison, covariates and cell types, `T3` seeds and retraining controls, `T4` independent data and direct baselines, `T5` SATURN and CAMEX, `T6` functional layer, `T7` disease case studies in liver and islet, `T8` residue control and figure panels. |
| `T11`, `T12`, `P13` | Cohorts beyond liver and testis: eye (`T11`), bone marrow (`T12`, results in `P12`), kidney (`P13`). |
| `C1` | The cohorts of cortex, lung, skeletal muscle and tumor myeloid cells. |
| `X`, `Y` | Analyses chosen after the pre-specified results were known. They are labeled exploratory wherever the paper uses them. |
| `Z` | The per-gene gate (`Z3`, `Z4`), recipes of remaining cohorts (`Z1`) and the embedding (`Z2`). |
| `H1`, `I1` | One schema for all 26 cohorts (`H1`); integration with the per-gene gate (`I1`). |
| `E1` to `E6` | Analyses across all cohorts: gene programs, protein architecture, cell types, drug targets and disease genes, regulatory input, published species differences. |
| `B1`, `B2` | The gene-level benchmark: ground truth (`B1`); robustness, independent data and overview (`B2`). |
| `D1`, `G1`, `O1` | Genes with few counts (`D1`), the latent guard (`G1`), the ortholog audit and the identifier-joined objects (`O1`). |
| `F1`, `F2`, `S1` | Figure helpers. |

A result folder usually has the name of its task without the description
(`tasks/T4_3_scoreboard` writes `results/analysis/T4_3/`). The exceptions: `T12_3_tests` writes
`results/analysis/P12/`; `T11_4` to `T11_7`, `T11_2c` and part of `T11_8` write
`results/analysis/T11/`; `P13_kidney` and `P13_recipe` share `results/analysis/P13/`;
`T7_9_islet`, `T7_9_recipe` and `T7_9_recipe_b` share `results/analysis/T7_9/`; `T7_3_mash` also
writes `results/analysis/T7_8/`; `T5_5_benchmark` writes to `notebooks/cache/`.

## The tasks, by the question they answer

In the order of the paper. **Status**: *in the paper* (numbers or panels of the paper come from
the task), *supporting* (builds inputs, launches runs, or is a check that the paper reports as a
method detail), *exploratory, not in the paper*, *recipe search* (the recipe grid of a cohort and
its selection rule), *infrastructure*. **Run** shows the first step; the folder README lists all
steps, the environment, what the task reads, and the deposit bundle of each output. The commands
were checked to exist and to start (`--help` or import); the analyses were not run again for
this index.

<!-- index:begin (generated by analysis/make_index.py; do not edit by hand) -->
### Data

Which objects, ortholog tables and independent references do the analyses read?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T4_1_gold`](tasks/T4_1_gold/README.md) | Downloads the independent references of the pre-specified tests and the Ensembl annotation, homology and dN/dS files, and checks identifiers, scales and coverage. No Chimera readout is read. | `bash analysis/tasks/T4_1_gold/fetch_gold.sh` (+4 more steps) | `data/analysis/gold/`<br>`data/analysis/ensembl110/`<br>`results/analysis/T4_1/` | Supplementary Section C.21.8 | supporting |
| [`T6_1_functional`](tasks/T6_1_functional/README.md) | Downloads the sources of the functional layer (gene sets, protein interaction networks, transcription-factor binding) and reports identifier coverage. Counts only. | `bash analysis/tasks/T6_1_functional/fetch_functional.sh` (+2 more steps) | `data/analysis/functional/`<br>`results/analysis/T6_1/` | Supplementary Section C.21.11 | supporting |
| [`T7_1_holdouts`](tasks/T7_1_holdouts/README.md) | Downloads the liver disease data sets, checks them against the file lists of GEO, and fixes the mouse hold-out samples before any expression is read. | `bash analysis/tasks/T7_1_holdouts/fetch_disease.sh` (+5 more steps) | `data/analysis/disease/raw/`<br>`results/analysis/T7_1/` | Supplementary Section C.21.14 | supporting |
| [`T7_2_disease_objects`](tasks/T7_2_disease_objects/README.md) | Builds the annotated objects of the liver MASH case study (human and mouse, healthy and disease) from the downloads of T7_1_holdouts. | `python -m analysis.tasks.T7_2_disease_objects.build [--force]` | `data/analysis/disease/`<br>`results/analysis/T7_2/` | Supplementary Table S18; Supplementary Section C.21.14 | in the paper |
| [`T11_1_vanzyl`](tasks/T11_1_vanzyl/README.md) | Rebuilds the eye cohort (aqueous-humor outflow tissue of five species, Van Zyl et al. 2020) from the GEO count matrices, verifies it gene by gene against the raw matrices, and fixes the glaucoma gene list. | `python -m analysis.tasks.T11_1_vanzyl.build_vanzyl [--out DIR] [--no-compare]` (+3 more steps) | `data/analysis/van_zyl/`<br>`data/analysis/glaucoma/`<br>`results/analysis/T11_1/` | Section 3.7; Supplementary Section C.21.14; Supplementary Table S18 | supporting |
| [`T12_1_bone_marrow`](tasks/T12_1_bone_marrow/README.md) | Builds the bone-marrow cohort (human, mouse lemur, mouse) from the CELLxGENE files with its one-to-one ortholog table, and recomputes the conservation of expression of Shay et al. 2013 used as its reference. | `bash analysis/tasks/T12_1_bone_marrow/fetch_bm.sh` (+5 more steps) | `data/analysis/bone_marrow/`<br>`results/analysis/T12_1/` | Section 3.7; Supplementary Section C.21.14; Supplementary Table S18 | supporting |
| [`O1_orthologs`](tasks/O1_orthologs/README.md) | Audits every ortholog pair of the cohorts against Ensembl Compara, flags pairs and count columns, repeats the validation without flagged genes, and builds the liver and testis objects with counts joined on the Ensembl gene ID together with the models trained on them. | `python -m analysis.tasks.O1_orthologs.audit_pairs` (+9 more steps) | `results/analysis/O1/`<br>`data/analysis/O1/repair_liver/`<br>`data/analysis/O1/repair_testis/`<br>`training/runs/o1-*`<br>`training/runs/o2-*` | Supplementary Section C; Supplementary Table S18 | supporting |
| [`T2_2_regen`](tasks/T2_2_regen/README.md) | Computes, with each cohort's own ortholog table and genes matched between tissues on the human Ensembl gene ID, the statistics that depend on gene identity in testis and on the match of genes between tissues: program-level agreement of Δα within testis and between testis and liver, sign quadrants of the tissue means, category effects on testis, and the gene-set enrichment tables; one row per quantity. | `python -m analysis.tasks.T2_2_regen.values` | `results/analysis/T2_2/values.csv` | Supplementary Section C.21.3; Supplementary Section D.1; Supplementary Section D.2; Supplementary Section D.15; Supplementary Figure S3g, h, i, j, k; Supplementary Figure S18d; Supplementary Figure S20; Supplementary Figure S21c; Supplementary Figure S65e | in the paper |

### Model fits beyond the canonical runs

Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T3_1_platform`](tasks/T3_1_platform/README.md) | Trains two canonical recipes again on CPU with the canonical seed, to compare the CPU platform with the Apple MPS runs. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_1_platform/manifest.tsv` | `training/runs/rev-cpu-<cohort>-s42/`<br>`results/analysis/T3_1/` | Supplementary Section C.21.2 | supporting |
| [`T3_2_seeds`](tasks/T3_2_seeds/README.md) | Trains further seeds of the canonical recipes of liver Human+Mouse, four-species liver, testis Human+Mouse and three-species testis on CPU. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_2_seeds/manifest.tsv` | `training/runs/rev-<cohort>-s<seed>/`<br>`results/analysis/T3_2/` | Supplementary Section C.21.2 | supporting |
| [`T3_2b_extra_seeds`](tasks/T3_2b_extra_seeds/README.md) | Trains seeds 47 to 51 of liver Human+Mouse and testis Human+Mouse, which complete the ten-seed consensus of the two primary cohorts. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_2b_extra_seeds/manifest.tsv` | `training/runs/rev-<cohort>-s<seed>/`<br>`results/analysis/T3_2b/` | Supplementary Section C.21.2 | supporting |
| [`T3_1b_integration_gate`](tasks/T3_1b_integration_gate/README.md) | Scores the integration checks E8.1 to E8.6 of training/compare_vs_scvi.py on every seed run, which separates integrated from poorly mixed training states. | `python -m analysis.tasks.T3_1b_integration_gate.run [--outdir results/analysis/T3_1b]` | `results/analysis/T3_1b/` | Supplementary Section C.21.2 | in the paper |
| [`T6_8b_repeat`](tasks/T6_8b_repeat/README.md) | Trains liver Human+Mouse seeds 46 and 47 a second time with unchanged arguments and compares the repeat, the original and the prior-swap run of each seed, to measure how much two runs with the same seed differ on CPU. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T6_8b_repeat/manifest.tsv` (+1 more steps) | `training/runs/rev-rep-liv_hmouse-s<seed>/`<br>`results/analysis/T6_8b/`<br>`results/analysis/T6_8b_repeat.csv` | Supplementary Section C.21.2; Supplementary Section C.21.11 | in the paper |
| [`G1_guard`](tasks/G1_guard/README.md) | Tests the latent guard (--clamp-z-logvar): scans the posterior log-variance of archived runs, shows that the option leaves guarded-off fits bit-identical, and counts finished and non-finite fits with and without it. | `python -m analysis.tasks.G1_guard.step1_scan --runs analysis/tasks/G1_guard/step1_runs.tsv --out results/analysis/G1` (+5 more steps) | `results/analysis/G1/`<br>`training/runs/rev-g1-*`<br>`training/runs/rev-g1u-*`<br>`training/runs/g1-*` | Supplementary Section C.12.2 | in the paper |
| [`Z1_retune`](tasks/Z1_retune/README.md) | Measures count dependence and training health of a run, and writes the recipes of the two eye pairs (Human+Pig, Human+M. fascicularis) that the grid of T11_2b_vz_subsets did not settle. | `python -m analysis.tasks.Z1_retune.runs_reference` (+3 more steps) | `results/analysis/Z1/`<br>`training/runs/rev-z1-*` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T11_2_recipe`](tasks/T11_2_recipe/README.md) | Recipe grid of the five-species eye cohort and its selection rule, fixed before training: training health first, then the integration composite. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T11_2_recipe/manifest.tsv` (+1 more steps) | `training/runs/rev-vz-grid<k>-s42/`<br>`results/analysis/T11_2/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T11_2b_vz_subsets`](tasks/T11_2b_vz_subsets/README.md) | Builds the species subsets of the eye cohort (Human+Mouse, Human+M. mulatta, primates, Human+Pig, Human+M. fascicularis) and selects a recipe for each from a grid fixed before training. | `python -m analysis.tasks.T11_2b_vz_subsets.build_subsets` (+3 more steps) | `data/analysis/van_zyl/`<br>`training/runs/rev-vz-<subset>-*`<br>`results/analysis/T11_2b/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T11_2c_count_rule`](tasks/T11_2c_count_rule/README.md) | Chooses the recipe of the five-species eye cohort by count dependence, a rule that reads alpha, the prior and the count evidence only. | `python -m analysis.tasks.T11_2c_count_rule.select_count` (+1 more steps) | `results/analysis/T11/count_rule/`<br>`training/runs/rev-vz-grid2-s<seed>/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T11_3_vz_seeds`](tasks/T11_3_vz_seeds/README.md) | Seed runs of the selected eye recipe. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T11_3_vz_seeds/manifest.tsv` | `training/runs/rev-vz-s<seed>/`<br>`results/analysis/T11_3/` | Supplementary Section C.21.14 | supporting |
| [`T11_8_vz_class_filter`](tasks/T11_8_vz_class_filter/README.md) | Sensitivity build of the eye cohort without cell types that have fewer than 50 cells in a species, with models trained on it and a comparison against the models of the paper by a rule fixed in advance. | `python -m analysis.tasks.T11_8_vz_class_filter.build_hmouse` (+2 more steps) | `data/analysis/van_zyl_f50/`<br>`training/runs/rev-vz-f50-*`<br>`results/analysis/T11_8/`<br>`results/analysis/T11/class_filter/` | Supplementary Section C.21.14 | supporting |
| [`T12_2_recipe`](tasks/T12_2_recipe/README.md) | First recipe grid of the bone-marrow cohort and its selection rule, fixed before training. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T12_2_recipe/manifest.tsv` (+1 more steps) | `training/runs/rev-bm-grid<k>-s42/`<br>`results/analysis/T12_2/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T12_2b_recipe`](tasks/T12_2b_recipe/README.md) | Second recipe grid of the bone-marrow cohort, on the build without a mitochondrial filter and with a cap per cell type; same selection rule; the seed runs of the selected recipe. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T12_2b_recipe/manifest.tsv` (+2 more steps) | `training/runs/rev-bm2-*`<br>`results/analysis/T12_2b/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T12_2b_debug`](tasks/T12_2b_debug/README.md) | Diagnostic objects and a probe that locates the first non-finite tensor in a bone-marrow training run. Training health only; no readout. | `python -m analysis.tasks.T12_2b_debug.make_debug_objects --rule prop --n 25000` (+1 more steps) | `data/analysis/bone_marrow/debug/`<br>`training/runs/rev-bm-dbg-*` | Supplementary Section C.12.2 | supporting |
| [`T7_9_recipe`](tasks/T7_9_recipe/README.md) | First recipe grid of the healthy islet model and its selection rule, fixed before training. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T7_9_recipe/manifest.tsv` (+2 more steps) | `training/runs/rev-isl-grid<k>-s42/`<br>`results/analysis/T7_9/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`T7_9_recipe_b`](tasks/T7_9_recipe_b/README.md) | Second recipe grid of the healthy islet model with the unchanged selection rule, and the seed runs of the selected recipe. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T7_9_recipe_b/manifest.tsv` (+2 more steps) | `training/runs/rev-isl-*`<br>`results/analysis/T7_9/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |
| [`P13_recipe`](tasks/P13_recipe/README.md) | Recipe grid of the healthy kidney model, its selection rule fixed before training, and the seed runs of the selected recipe. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/P13_recipe/manifest.tsv` (+3 more steps) | `training/runs/rev-kid-*`<br>`results/analysis/P13/` | Supplementary Table S19; Supplementary Section C.21.1 | recipe search |

### Cell-level integration

Does one fit align the cells across species, and how does it compare with other tools?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T5_1_envs`](tasks/T5_1_envs/README.md) | Builds the environments of SATURN and CAMEX and fetches the two tools at their pinned commits. | `bash analysis/tasks/T5_1_envs/build_envs.sh` | `results/analysis/T5_1/` | Supplementary Section C.21.9 | infrastructure |
| [`T5_2_adapters`](tasks/T5_2_adapters/README.md) | Adapters that run SATURN (its native genes, and the cohort's one-to-one orthologs) and CAMEX on the Chimera cohorts, with a pilot that checks the tools against their published scores. | `sbatch analysis/tasks/T5_2_adapters/tool.sbatch <saturn|camex> <stem> <labels|-> <seed> <work> <dest>` (+1 more steps) | `data/analysis/T5_inputs/`<br>`results/analysis/T5_2/` | Supplementary Section C.21.9 | supporting |
| [`T5_3_runs`](tasks/T5_3_runs/README.md) | Cell-level runs of SATURN, SATURN with Leiden clusters in place of labels, and CAMEX on the seven canonical cohorts, with a check of every latent. | `sbatch --array=0-<rows-1> analysis/tasks/T5_3_runs/tool_array.sbatch analysis/tasks/T5_3_runs/saturn.tsv (likewise saturn_leiden.tsv and camex.tsv)` (+1 more steps) | `results/analysis/T5_3/` | Section 3.1; Figure 3; Supplementary Section C.21.9 | supporting |
| [`T5_5_benchmark`](tasks/T5_5_benchmark/README.md) | Scores SATURN, SATURN-Leiden and CAMEX with the metrics of the seven-method panel, giving the ten-method integration scorecard of the seven canonical cohorts. | `python -m analysis.tasks.T5_5_benchmark.score_external` (+1 more steps) | `notebooks/cache/baselines_master_ext/`<br>`notebooks/figures/baselines_master_ext/` | Section 3.1; Figure 3; Supplementary Figure S9; Supplementary Figure S10; Supplementary Section D.1; Supplementary Section C.21.9 | in the paper |
| [`T5_4b_label_ablation`](tasks/T5_4b_label_ablation/README.md) | Trains the testis recipes without cell-type labels (no cell-type-conditional MMD) and compares integration, Δα and its validity with the canonical testis models. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T5_4b_label_ablation/manifest.tsv` (+1 more steps) | `training/runs/rev-ctl-<cohort>-nolab-s<seed>/`<br>`results/analysis/T5_4b/` | Supplementary Section C.21.9; Supplementary Table S22 | in the paper |
| [`T11_4_vz_integration`](tasks/T11_4_vz_integration/README.md) | Integration scorecard of the eye data sets, reported as rows of their own beside the seven canonical cohorts. | `sbatch --array=0-<rows-1> analysis/tasks/T11_4_vz_integration/cmd_array.sbatch analysis/tasks/T11_4_vz_integration/baselines.tsv` (+1 more steps) | `results/analysis/T11/integration/` | Section 3.7; Supplementary Figure S11 | in the paper |
| [`Z2_embedding`](tasks/Z2_embedding/README.md) | Asks how well Chimera's latent resolves cell states inside a cell-type label, beside scVI on the same cells, and what sets it (depth, capacity, MMD conditioning), with a factorial of training variants. | `python -m analysis.tasks.Z2_embedding.measure prep <cohort>` (+6 more steps) | `results/analysis/Z2/`<br>`training/runs/z2-*` | Section 3.1 | in the paper |

### Δα as an axis of its own

Is deployment relative to the protein prior separate from protein sequence?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T2_3_tissue`](tasks/T2_3_tissue/README.md) | Compares Δα between liver and testis for the matched species pairs, with genes joined on the human Ensembl ID, against the clamped and the raw prior. | `python -m analysis.tasks.T2_3_tissue.run [--force] (or bash analysis/tasks/run_T2_3_to_T2_7.sh for T2_3 to T2_7)` | `results/analysis/T2_3/` | Section 3.2; Figure 4f; Supplementary Section C.21.3; Supplementary Section D.2 | in the paper |
| [`X5_sequence_axis`](tasks/X5_sequence_axis/README.md) | Sets protein identity, dN/dS and other known gene measures beside α, Δα and the count evidence on the same genes, cohort by cohort. | `python -m analysis.tasks.X5_sequence_axis.run` (+1 more steps) | `results/analysis/X5/` | Section 3.2; Figure 4g; Supplementary Figure S13; Supplementary Figure S14; Supplementary Section D.2 | in the paper |

### What the readouts find in liver and testis

Which genes, cell types, pathways and gene classes sit above or below the protein prior?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T2_5_celltype_margin`](tasks/T2_5_celltype_margin/README.md) | Splits the count evidence S_g of every gene into contributions of species and cell types by forward passes through the archived models. | `python -m analysis.tasks.T2_5_celltype_margin.run [--runs all] [--threads 4] [--n-boot 200]` | `results/analysis/T2_5/` | Section 3.3; Figure 4d; Supplementary Figure S22; Supplementary Figure S23; Supplementary Section C.21.5 | in the paper |
| [`T2_6_matched`](tasks/T2_6_matched/README.md) | Tests the gene-category and landmark statements against genes matched on expression and coding length, and checks the mitochondrial genes under detection matching. | `python -m analysis.tasks.T2_6_matched.run [--n-null 2000]` | `results/analysis/T2_6/` | Section 3.3; Supplementary Section D.2; Supplementary Section C.21.6 | in the paper |
| [`X1_testis_discovery`](tasks/X1_testis_discovery/README.md) | Testis: genes grouped by the stage in which they peak, their Δα and count evidence against matched genes, across cohorts and seeds. | `python -m analysis.tasks.X1_testis_discovery.build` (+3 more steps) | `results/analysis/X1/` | Section 3.3; Supplementary Section C.21.12 | in the paper |
| [`X2_liver_biology`](tasks/X2_liver_biology/README.md) | Liver: gene sets, hepatocyte zonation and the hepatocyte catabolic genes with conserved proteins and Δα below the prior, on per-gene liver tables and zone-resolved forward passes. | `python -m analysis.tasks.X2_liver_biology.sets` (+5 more steps) | `results/analysis/X2/` | Section 3.3; Supplementary Section D.2; Supplementary Table S18; Supplementary Section C.21.12 | in the paper |
| [`X3_cross_cohort`](tasks/X3_cross_cohort/README.md) | Gene classes (transcription factors, receptors, enzymes and others) and protein-space neighborhood coherence of the readouts over the seven canonical cohorts. | `python -m analysis.tasks.X3_cross_cohort.annotations` (+4 more steps) | `results/analysis/X3/`<br>`data/analysis/x3_annotations/` | Section 3.9 | in the paper |
| [`X4_where_differs`](tasks/X4_where_differs/README.md) | Where Δα and the count evidence differ from sequence constraint and from direct baselines: a likelihood margin computed without a model, split into pattern gain and level cost; relations among all scores; named genes. | `python -m analysis.tasks.X4_where_differs.build` (+7 more steps) | `results/analysis/X4/` | Section 3.3; Section 3.2; Supplementary Section D.2; Supplementary Figure S15; Supplementary Figure S16; Supplementary Section C.21.12 | in the paper |
| [`Y3_findings`](tasks/Y3_findings/README.md) | Worked examples in the counts (spermatocyte-peak and hepatocyte catabolic genes), the same profile-correlation recipe applied to the latents of other tools, and controls of the kidney count evidence. | `python -m analysis.tasks.Y3_findings.worked_genes` (+3 more steps) | `results/analysis/Y3/` | Section 3.3; Supplementary Figure S17; Supplementary Section C.21.12 | in the paper |
| [`Y4_regions`](tasks/Y4_regions/README.md) | Screens regions of gene space (by expression, specificity, identity and annotation) for where Δα and the count evidence agree with independent data, tuned on odd and scored on even autosomes. | `python -m analysis.tasks.Y4_regions.annotations` (+7 more steps) | `results/analysis/Y4/` | Section 3.3 | in the paper |
| [`Y5_aspects`](tasks/Y5_aspects/README.md) | Further aspects of the readouts on the canonical cohorts: protein properties, phylogenetic depth, cell states inside a label, gene modules, species by cell-type margins and X linkage in testis. | `python -m analysis.tasks.Y5_aspects.build annot` (+10 more steps) | `results/analysis/Y5/`<br>`data/analysis/y5_annotations/` | Section 3.3 | in the paper |
| [`Y2_functional`](tasks/Y2_functional/README.md) | Asks whether Δα or the count evidence relate to data outside expression: mouse phenotypes, viability, regulatory sequence conservation and literature cases. | `bash analysis/tasks/Y2_functional/fetch.sh` (+7 more steps) | `results/analysis/Y2/`<br>`data/analysis/Y2/` | not in the paper | exploratory, not in the paper |
| [`T6_common`](tasks/T6_common/README.md) | Shared inputs of the functional-layer tests: residual readouts, strata and gene identifiers. | `(imported by T6_3_pathway, T6_4_network, T6_6_regulatory, T6_9_assemble)` | `results/analysis/T6_3/cache/` | Supplementary Section C.21.11 | supporting |
| [`T6_3_pathway`](tasks/T6_3_pathway/README.md) | Functional layer: reliability of gene-set statistics across seeds, and coherence of Δα within pathways. | `python -m analysis.tasks.T6_3_pathway.run [--n-perm 10000]` (+1 more steps) | `results/analysis/T6_3/` | Section 3.3; Supplementary Figure S24; Supplementary Section D.2; Supplementary Section C.21.11 | in the paper |
| [`T6_4_network`](tasks/T6_4_network/README.md) | Functional layer: Moran's I of Δα on protein-interaction graphs and retention of interologs. | `python -m analysis.tasks.T6_4_network.run [--only F2|F3] [--procs 6]` | `results/analysis/T6_4/` | Section 3.3; Supplementary Figure S24; Supplementary Section D.2; Supplementary Section C.21.11 | in the paper |
| [`T6_6_regulatory`](tasks/T6_6_regulatory/README.md) | Functional layer: conserved binding of liver transcription factors (ChIP-Atlas) against the residual readout on liver Human+Mouse, with testis as tissue control. | `python -m analysis.tasks.T6_6_regulatory.fix_panel` (+1 more steps) | `results/analysis/T6_6/` | Section 3.3; Supplementary Figure S24; Supplementary Section D.2; Supplementary Section C.21.11 | in the paper |
| [`T6_9_assemble`](tasks/T6_9_assemble/README.md) | Applies the multiple-testing family of the functional layer over its seven tests and draws the supplementary functional-layer figure. | `python -m analysis.tasks.T6_9_assemble.assemble <output pdf>` | `results/analysis/T6_9/` | Section 3.3; Supplementary Figure S24; Supplementary Section D.2; Supplementary Section C.21.11 | in the paper |

### Pre-specified validation

Do independent data back Δα and the count evidence S_g, and how do direct baselines compare?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T4_2_gene_baselines`](tasks/T4_2_gene_baselines/README.md) | Computes the direct baselines for every cohort from the same counts: pseudobulk profile correlation, rank-matched level per cell type, profile correlation over clusters of Chimera's latent, specificity similarity, and predictions from gene features alone. | `python -m analysis.tasks.T4_2_gene_baselines.run [--cohorts all] [--delta-alpha canonical]` | `results/analysis/T4_2/` | Section 3.4; Table 1; Figure 5c; Supplementary Section C.21.8 | in the paper |
| [`T4_3_scoreboard`](tasks/T4_3_scoreboard/README.md) | The pre-specified validation: partial rank correlation of Δα with four independent references given covariates, with block-bootstrap intervals, gene-set tests, head-to-head against the direct baselines, and the mechanical decision procedure. | `python -m analysis.tasks.T4_3_scoreboard.pipeline (runs golds, baseline_splits, run --mode real, decide, figures from MANIFEST.tsv)` (+3 more steps) | `results/analysis/T4_3/` | Section 3.4; Table 1; Figure 5a, b, c; Supplementary Section C.21.8; Supplementary Section D.3 | in the paper |

### Reliability and controls

How stable are the readouts across seeds, and what do they read from counts and from features?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T3_3_consensus`](tasks/T3_3_consensus/README.md) | Applies the seed-pooling rule and builds the seed-consensus per-gene tables of the seven canonical cohorts with their seed reliability. | `python -m analysis.tasks.T3_3_consensus.run [--cohorts ...]` | `results/analysis/T3_3/` | Section 3.4; Supplementary Figure S26; Supplementary Section D.3; Supplementary Section C.21.2 | in the paper |
| [`T2_4_covariates`](tasks/T2_4_covariates/README.md) | Covariate audit of Δα in the seven canonical cohorts: expression level, detection, specificity, coding length and GC, how much features alone predict, and which genes carry no count evidence (masked). | `python -m analysis.tasks.T2_4_covariates.run [--n-perm 1000]` | `results/analysis/T2_4/` | Section 3.4; Supplementary Figure S25; Figure 4e; Supplementary Section C.21.4 | in the paper |
| [`T3_4_destroy_features`](tasks/T3_4_destroy_features/README.md) | Retraining control that keeps the counts and removes, randomizes or zeroes the gene features. | `sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_4_destroy_features/manifest.tsv` (+1 more steps) | `training/runs/rev-ctl-<cohort>-<arm>-s<seed>/`<br>`results/analysis/T3_4/` | Section 3.4; Figure 8c; Supplementary Section D.3; Supplementary Section C.21.7 | in the paper |
| [`T3_5_count_perm`](tasks/T3_5_count_perm/README.md) | Retraining control that keeps the features and shuffles the counts across genes, with a projection null that needs no training. | `python -m analysis.tasks.T3_5_count_perm.make_control_h5ad` (+3 more steps) | `data/analysis/controls/`<br>`training/runs/rev-ctl-<cohort>-perm<A|B|C>-s<seed>/`<br>`results/analysis/T3_5/` | Section 3.4; Figure 8c; Supplementary Section D.3; Supplementary Section C.21.7 | in the paper |
| [`T3_6_matched`](tasks/T3_6_matched/README.md) | Retraining control on four-species liver restricted to the two cell types that all species share, beside a composition-balanced count evidence computed from the canonical model. | `python -m analysis.tasks.T3_6_matched.make_matched_h5ad` (+3 more steps) | `data/analysis/controls/`<br>`training/runs/rev-ctl-liv_4sp-matchA-s<seed>/`<br>`results/analysis/T3_6/` | Section 3.3; Figure 8c; Supplementary Section D.3; Supplementary Section C.21.7 | in the paper |
| [`T6_8_priorswap`](tasks/T6_8_priorswap/README.md) | Retraining control with a smoothed protein prior (half the prior, half its average over protein-interaction partners), compared with same-seed runs and with the seed floor. | `python -m analysis.tasks.T6_8_priorswap.make_control_h5ad` (+2 more steps) | `data/analysis/controls/`<br>`training/runs/rev-ctl-liv_hmouse-priorswap-s<seed>/`<br>`results/analysis/T6_8/` | Supplementary Section C.21.11 | in the paper |
| [`D1_sparse_genes`](tasks/D1_sparse_genes/README.md) | Asks whether the protein-feature gate gives a usable reading for genes with too few counts: masked genes against independent data, thinned-depth fits, held-out chromosomes, and gates fed selected feature blocks. | `python -m analysis.tasks.D1_sparse_genes.fwd --cohort <cohort>` (+8 more steps) | `results/analysis/D1/`<br>`data/analysis/D1/`<br>`training/runs/d1-*` | Section 3.4 | in the paper |

### The per-gene gate

What does Δα show when it is read gene by gene from counts, without the protein-feature network?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`Z3_free_gate`](tasks/Z3_free_gate/README.md) | Trains the per-gene gate (--config M: one free offset per gene in place of the protein-feature network) on liver Human+Mouse, four-species liver and testis Human+Mouse, and scores its Δα on the pre-specified tests beside the protein-feature gate on the same seeds. | `sbatch --array=<rows> analysis/tasks/Z3_free_gate/train.sbatch` (+3 more steps) | `training/runs/z3-<cohort>-m-s<seed>/`<br>`results/analysis/Z3/` | Section 3.5; Figure 5e, f, h, i; Supplementary Section C.8.4 | in the paper |
| [`Z4_free_gate_all`](tasks/Z4_free_gate_all/README.md) | Trains the per-gene gate on every one of the 26 cohorts with the cohort's own recipe, keeps seeds by a health rule, and builds the per-gene tables and reference comparisons in the schema of H1_harmonise. | `python -m analysis.tasks.Z4_free_gate_all.make_manifest` (+5 more steps) | `training/runs/z4-<cohort>-m-s<seed>/`<br>`results/analysis/Z4/` | Section 3.7; Section 3.8 | in the paper |
| [`F2_per_gene`](tasks/F2_per_gene/README.md) | Builds the per-gene tables behind the per-gene-gate panels of the validation figure and draws the panels. | `python -m analysis.tasks.F2_per_gene.data` (+1 more steps) | `results/analysis/F2/`<br>`figures/panels/` | Figure 5e, f, g, h, i; Section 3.5 | in the paper |
| [`I1_free_gate_integration`](tasks/I1_free_gate_integration/README.md) | Compares cell-level integration of the per-gene gate and the protein-feature gate with the metrics of the integration benchmark, on every cohort. | `python -m analysis.tasks.I1_free_gate_integration.score check` (+4 more steps) | `results/analysis/I1/` | Section 3.5; Supplementary Table S23; Supplementary Table S24; Supplementary Section D.1 | in the paper |
| [`X8_gate_divergence`](tasks/X8_gate_divergence/README.md) | Where the two gates place a gene differently (the gate gap): its reproducibility, its recurrence across tissues, gene classes and protein neighborhoods, its relation to independent references, and named examples. | `python -m analysis.tasks.X8_gate_divergence.gap` (+6 more steps) | `results/analysis/X8/` | Supplementary Section D.4; Section 3.5; Supplementary Figure S27; Supplementary Figure S28; Supplementary Figure S29 | in the paper |

### What the count evidence adds

What do the readouts add to direct baselines and to other tools, on real and on ground-truth data?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T5_6_gene_level`](tasks/T5_6_gene_level/README.md) | Derives per-ortholog scores from SATURN (macrogene weights) and CAMEX (gene embeddings), measures their seed reliability and relation to Δα, and scores them on the pre-specified tests beside Δα. | `python -m analysis.tasks.T5_6_gene_level.scalars` (+2 more steps) | `results/analysis/T5_6/` | Section 3.6; Figure 5c; Supplementary Section D.3; Supplementary Section C.21.10 | in the paper |
| [`X6_adds_beyond`](tasks/X6_adds_beyond/README.md) | What each score adds once all simpler scores are known (incremental information), and how label-dependent direct scores degrade when cell-type labels are noisy, coarse or replaced by clusters. | `python -m analysis.tasks.X6_adds_beyond.degrade` (+2 more steps) | `results/analysis/X6/` | Section 3.6; Section 3.4; Figure 5d; Table 2; Supplementary Section C.21.12 | in the paper |
| [`X7_architecture_outputs`](tasks/X7_architecture_outputs/README.md) | Tests outputs that only this architecture produces (per-cell residual share, spread of the residual across species, species-resolved margins in a joint fit) against independent per-type divergence and against baselines from the same counts. | `python -m analysis.tasks.X7_architecture_outputs.forward --runs <labels>` (+5 more steps) | `results/analysis/X7/` | Supplementary Section C.21.12 | in the paper |
| [`B1_ground_truth`](tasks/B1_ground_truth/README.md) | Gene-level benchmark with known truth: pattern, level or loss changes are written into the counts of one species of real cohorts, every method is scored on recovering the altered genes, with null alterations for type I error; and a run-time and memory track. | `python -m analysis.tasks.B1_ground_truth.base_tables` (+9 more steps) | `results/analysis/B1/`<br>`data/analysis/B1/`<br>`training/runs/b1-*`<br>`training/runs/b1free-*`<br>`training/runs/b1hybrid-*`<br>`training/runs/b1trackd-*` | Section 3.6; Supplementary Section D.6; Supplementary Figure S31; Table 2; Supplementary Section C.21.16 | in the paper |
| [`B2_benchmark`](tasks/B2_benchmark/README.md) | Gene-level benchmark on real data: robustness of every method to fewer cells, damaged labels and lower depth; agreement with independent data across tissues; the adapters of SAMap, GeneSpectra and CACIMAR; the integration scores of the further cohorts; and the overview of all tracks. | `python -m analysis.tasks.B2_benchmark.make_grid` (+7 more steps) | `results/analysis/B2/`<br>`data/analysis/B2/`<br>`training/runs/b2-*` | Section 3.6; Section 3.1; Table 2; Supplementary Figure S32; Supplementary Figure S33; Supplementary Figure S11; Supplementary Section C.21.16 | in the paper |
| [`E6_known_differences`](tasks/E6_known_differences/README.md) | Tests whether the readouts recover species differences in cell-type gene expression that the literature documents, list by list. | `python -m analysis.tasks.E6_known_differences.known` (+4 more steps) | `results/analysis/E6/`<br>`data/analysis/E6/` | Section 3.6; Supplementary Section D.5; Supplementary Figure S30 | in the paper |
| [`Y6_uses`](tasks/Y6_uses/README.md) | Tries four further uses of existing fits: checking genome annotation, choosing a model species, drug targets, and cis-regulatory evidence by proxy. | `bash analysis/tasks/Y6_uses/fetch.sh` (+10 more steps) | `results/analysis/Y6/`<br>`data/analysis/Y6/` | not in the paper | exploratory, not in the paper |

### One reading across tissues

Does the reading hold across tissues, and what goes with deployment above the prior?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`H1_harmonise`](tasks/H1_harmonise/README.md) | One reading for all 26 cohorts in ten tissues: per-gene tables in one schema with support and naming flags, run health, gene-set recurrence across tissues, the separate-axis statistics, per-tissue panels and the cohort tables of the Supplement. | `bash analysis/tasks/H1_harmonise/run_all.sh` | `results/analysis/H1/` | Section 3.7; Figure 6a, b, c; Supplementary Table S19; Supplementary Table S20; Supplementary Table S21; Supplementary Figure S34; Supplementary Figure S38; Supplementary Figure S39; Supplementary Figure S40; Supplementary Figure S41; Supplementary Figure S42; Supplementary Figure S43; Supplementary Figure S44; Supplementary Figure S45; Supplementary Figure S46 | in the paper |
| [`C1_cohorts`](tasks/C1_cohorts/README.md) | The cohorts of cerebral cortex (dorsolateral prefrontal and primary motor), lung, skeletal muscle and tumor myeloid cells: one build path, recipe grid and selection, readouts in the schema of H1_harmonise, tests against held-out data, and their disease cases. | `python -m analysis.tasks.C1_cohorts.orthologs` (+10 more steps) | `data/analysis/new_usecases/`<br>`training/runs/rev-c1-<cohort>-*`<br>`results/analysis/C1/` | Section 3.7; Section 3.3; Section 3.8; Section 3.1; Supplementary Table S19; Supplementary Table S18; Supplementary Section C.21.15 | in the paper |
| [`Y1_gene_tissue`](tasks/Y1_gene_tissue/README.md) | Gene by tissue: how much of each readout belongs to the gene and how much to the gene in a tissue, whether tissue differences within a gene follow independent atlas differences, and worked genes. | `python -I analysis/tasks/Y1_gene_tissue/launch.py analysis.tasks.Y1_gene_tissue.<module> [args], in the order atlas_gold, forward, build, prep, describe, variance, contrast, classes, worked, figure` | `results/analysis/Y1/`<br>`data/analysis/Y1/` | Section 3.7; Figure 6d, e, f, g, h, i | in the paper |
| [`E1_atlas`](tasks/E1_atlas/README.md) | Atlas of gene programs across tissues: set statistics per cohort and readout for both gates, recurrence across eight tissues, dependence on evolutionary distance, and recurrent named genes. | `bash analysis/tasks/E1_atlas/run_all.sh` | `results/analysis/E1/`<br>`data/analysis/E1/` | Section 3.7; Supplementary Section D.7; Supplementary Figure S35; Supplementary Figure S36; Supplementary Figure S37 | in the paper |
| [`E3_cells`](tasks/E3_cells/README.md) | Cell types and lineages: genes grouped by their home compartment, their excess over matched genes on both gates and on direct baselines; species-specific departures in fits with three or more species; replication in independent primate data. | `python -m analysis.tasks.E3_cells.build` (+8 more steps) | `results/analysis/E3/` | Section 3.7; Figure 6j; Supplementary Section D.8; Supplementary Figure S48; Supplementary Figure S49; Supplementary Figure S50 | in the paper |
| [`E2_protein`](tasks/E2_protein/README.md) | Protein architecture and gene history: Pfam domains, protein families, complexes, gene age and duplication class against Δα on 26 cohorts and both gates. | `bash analysis/tasks/E2_protein/fetch.sh` (+5 more steps) | `results/analysis/E2/`<br>`data/analysis/E2/` | Section 3.7; Figure 6k; Supplementary Section D.9; Supplementary Figure S51 | in the paper |
| [`E5_regulatory`](tasks/E5_regulatory/README.md) | Regulatory input and gene architecture: enhancer number, promoter class, regulons and neighborhood against Δα on 26 cohorts and both gates. | `bash analysis/tasks/E5_regulatory/fetch.sh` (+6 more steps) | `results/analysis/E5/`<br>`data/analysis/E5/` | Section 3.7; Figure 6l; Supplementary Section D.10; Supplementary Figure S52 | in the paper |
| [`T11_5_vz_genes`](tasks/T11_5_vz_genes/README.md) | Eye cohort: per-gene tables, covariate audit, direct baselines and cell-type contributions to the count evidence for the six eye data sets. | `python -m analysis.tasks.T11_5_vz_genes.forward --missing` (+1 more steps) | `results/analysis/T11/readout/`<br>`results/analysis/T11/genes/`<br>`results/analysis/T11/drivers/` | Section 3.7; Supplementary Section C.21.14 | in the paper |
| [`T11_6_vz_glaucoma`](tasks/T11_6_vz_glaucoma/README.md) | Eye cohort: where glaucoma genes sit on Δα and the count evidence, and the per-species residual by cell class. | `python -m analysis.tasks.T11_6_vz_glaucoma.glaucoma` (+1 more steps) | `results/analysis/T11/glaucoma/`<br>`results/analysis/T11/residuals/` | Section 3.7; Supplementary Section C.21.14 | in the paper |
| [`T11_7_vz_figure`](tasks/T11_7_vz_figure/README.md) | Eye cohort: the tidy summary table of all eye data sets and a figure of the eye results. | `bash analysis/tasks/T11_7_vz_figure/run_all.sh` | `results/analysis/T11/summary.csv`<br>`results/analysis/T11/figures/` | Section 3.7 | supporting |
| [`T12_3_tests`](tasks/T12_3_tests/README.md) | Bone marrow: seed health, per-gene table, the tests of Δα and the count evidence against the conservation of expression of Shay et al. 2013, cell-type contributions, and the supplementary figure. | `python -m analysis.tasks.T12_3_tests.seed_health` (+5 more steps) | `results/analysis/P12/` | Section 3.7; Supplementary Figure S47; Supplementary Section C.21.14 | in the paper |

### Disease

Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T7_3_mash`](tasks/T7_3_mash/README.md) | Liver MASH: hepatocyte pseudobulk differential expression in human and mouse, the gene-level test of healthy Δα against human-mouse agreement of the disease response, program transfer, and a second contrast with an innate-immune response list. | `sbatch analysis/tasks/T7_3_mash/run.sbatch fits` (+2 more steps) | `results/analysis/T7_3/`<br>`results/analysis/T7_8/` | Section 3.8; Figure 7; Supplementary Section D.11; Supplementary Section C.21.14 | in the paper |
| [`T7_9_islet`](tasks/T7_9_islet/README.md) | Pancreatic islet, type 2 diabetes: objects of healthy and diabetic islets, pseudobulk differential expression, the gene-level test of healthy Δα, and program transfer between human and mouse with label-permutation nulls. | `python -m analysis.tasks.T7_9_islet.build` (+8 more steps) | `data/analysis/disease_islet/`<br>`results/analysis/T7_9/` | Section 3.8; Figure 7; Supplementary Table S18; Supplementary Section D.11; Supplementary Section C.21.14 | in the paper |
| [`P13_kidney`](tasks/P13_kidney/README.md) | Kidney, human acute kidney injury against mouse ischemia-reperfusion injury: objects and ortholog table, pseudobulk differential expression with permutation and leave-one-out fits, the gene-level test of healthy Δα, program transfer by injury stage, and a published list of injury genes. | `bash analysis/tasks/P13_kidney/fetch_kidney.sh` (+12 more steps) | `data/analysis/disease_kidney/`<br>`results/analysis/P13/` | Section 3.8; Figure 7a, b, c; Supplementary Section D.11; Supplementary Section C.21.14; Supplementary Table S18 | in the paper |
| [`T7_10_fig8`](tasks/T7_10_fig8/README.md) | Draws the panels of the disease figure from the result tables of the kidney, islet and liver case studies. | `python -m analysis.tasks.T7_10_fig8.fig8_disease` | `results/analysis/T7_10/`<br>`figures/panels/` | Figure 7 | in the paper |
| [`E4_translational`](tasks/E4_translational/README.md) | Drug targets and disease genes: where they sit relative to the protein prior in their own tissue, on both gates, and whether placement relates to how well the mouse model of the gene matches the human disease. | `bash analysis/tasks/E4_translational/run_all.sh` | `results/analysis/E4/`<br>`data/analysis/E4/` | Section 3.8; Supplementary Section D.12; Supplementary Figure S53; Supplementary Figure S54 | in the paper |

### Protein space

Is the residue-level attribution specific?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T8_1_residue`](tasks/T8_1_residue/README.md) | Specificity control of the residue-level attribution: the same contrasts recomputed with genes swapped and with random directions, on the cached attribution inputs. | `python -m analysis.tasks.T8_1_residue.run` | `results/analysis/T8_1/` | Section 3.9; Supplementary Table S25; Supplementary Section D.15; Supplementary Section C.21.13 | in the paper |

### Figures and tables

Which scripts draw panels or measure table values from cached results?

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T8_4_figures`](tasks/T8_4_figures/README.md) | Panel scripts of main and supplementary figures that draw from cached result tables (integration, cell-type sources and tissue comparison, validation, control trainings, seed reliability), and generated supplementary tables. | `python -m analysis.tasks.T8_4_figures.make_all (single figures: fig3_integration, fig3_integration_ext, fig4_additions, fig5_controls, fig7_validation, supp_figures; tables: tables)` | `figures/panels/`<br>`results/analysis/T8_4/` | Figure 3; Figure 4; Figure 5; Figure 8; Supplementary Figure S26; Supplementary Figure S11 | in the paper |
| [`F1_figures`](tasks/F1_figures/README.md) | Redraws, at print size and from the same data, panels that were first made by the notebooks 04 and 06 and by task scripts. | `python -m analysis.tasks.F1_figures.notebook_panels` (+1 more steps) | `figures/panels/`<br>`results/analysis/F1/` | Figure 4; Supplementary Figure S21 | infrastructure |
| [`S1_figures`](tasks/S1_figures/README.md) | Helper that removes the panel letters from one page of an assembled figure page, so that the layout tool can draw them. | `python -m analysis.tasks.S1_figures.strip_page_labels` | `results/analysis/S1/` | not in the paper | infrastructure |
| [`T2_7_consistency`](tasks/T2_7_consistency/README.md) | Measures, from the archived runs, the values that the recipe and assumption tables of the Supplement print (gene counts, shared cell types, training outcomes per cohort). | `python -m analysis.tasks.T2_7_consistency.run` | `results/analysis/T2_7/` | Supplementary Table S15; Supplementary Table S3 | in the paper |

### Infrastructure

Environment and checkpoint checks.

| Task folder | What it does | Run (first step) | Writes | Paper | Status |
|---|---|---|---|---|---|
| [`T1_3_smoke`](tasks/T1_3_smoke/README.md) | Environment check of the authors' cluster: freezes the environment into the lock file, imports the dependencies, runs the tests and reloads the 11 canonical checkpoints on CPU against their stored α. | `bash analysis/tasks/T1_3_smoke/smoke.sh (rewrites analysis/envs/chimera_env.linux-64.lock.txt)` (+1 more steps) | `results/analysis/T1_3/`<br>`training/runs/rev-calib-*` | Supplementary Section C.21.3 | infrastructure |
<!-- index:end -->

## Keeping the index current

A new task folder needs one row in `tasks/INDEX.tsv` (tab-separated; items inside a cell are
separated by ` ; `; `paper` holds labels of `tasks/PAPER.tsv`, with a panel letter after a space
where known). Then:

```bash
python analysis/make_index.py            # rewrite this file's tables, the map in REPRODUCE.md and the folder READMEs
python analysis/make_index.py --check    # exit 1 if a row is wrong or a generated file is stale
```
