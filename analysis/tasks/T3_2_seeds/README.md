<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_2_seeds

Trains further seeds of the canonical recipes of liver Human+Mouse, four-species liver, testis Human+Mouse and three-species testis on CPU.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.2 (`sssec:rev-seeds`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_2_seeds/manifest.tsv

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-<cohort>-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T3_2/`: bundle `analysis_results`

**Notes.** Manifest only (14 runs).
