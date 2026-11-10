<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_1_platform

Trains two canonical recipes again on CPU with the canonical seed, to compare the CPU platform with the Apple MPS runs.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.2 (`sssec:rev-seeds`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_1_platform/manifest.tsv

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-cpu-<cohort>-s42/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T3_1/`: bundle `analysis_results`

**Notes.** Manifest only; the comparison and the seed-pooling rule are applied in T3_3_consensus.
