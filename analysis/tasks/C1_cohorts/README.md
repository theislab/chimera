<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# C1_cohorts

The cohorts of cerebral cortex (dorsolateral prefrontal and primary motor), lung, skeletal muscle and tumor myeloid cells: one build path, recipe grid and selection, readouts in the schema of H1_harmonise, tests against held-out data, and their disease cases.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env`, revision_de_env for de_case.py |
| Used in the paper | Section 3.7 (`sec:tissues`); Section 3.3 (`sec:findings`); Section 3.8 (`sec:disease`); Section 3.1 (`sec:model`); Supplementary Table S19 (`tab:harmonised-cohorts`); Supplementary Table S18 (`tab:data-preparation`); Supplementary Section C.21.15 (`sssec:rev-dataprep`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.C1_cohorts.orthologs
    python -m analysis.tasks.C1_cohorts.build <cohort>
    python -m analysis.tasks.C1_cohorts.zero_check
    python -m analysis.tasks.C1_cohorts.grid manifest <cohort>
    sbatch --array=0-<rows-1> analysis/tasks/C1_cohorts/array.sbatch <manifest.tsv>
    python -m analysis.tasks.C1_cohorts.grid report <cohort>
    python -m analysis.tasks.C1_cohorts.h1_c1 tables
    python -m analysis.tasks.C1_cohorts.golds
    python -m analysis.tasks.C1_cohorts.tests_c1
    python -m analysis.tasks.C1_cohorts.de_case --case <case>
    python -m analysis.tasks.C1_cohorts.hd4_case

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/new_usecases/`: bundle `data_other_cohorts`
- `training/runs/rev-c1-<cohort>-*`: bundle `models_seeds_controls_cohorts`
- `results/analysis/C1/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Y6_uses`
- `data/analysis/bone_marrow/`
- `data/analysis/ensembl110/`
- `data/analysis/new_usecases/`
- `data/processed/`
- `data/raw/`
- `training/runs/`

**Notes.** configs.py holds one configuration per cohort; grids/ the recipe grids; references.tsv the downloads. esm_embed.py computes ESM-2 gene embeddings for species outside the SATURN export. Labeled exploratory in the paper.
