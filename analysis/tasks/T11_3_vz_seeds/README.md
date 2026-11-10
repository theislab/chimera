<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_3_vz_seeds

Seed runs of the selected eye recipe.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T11_3_vz_seeds/manifest.tsv

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-vz-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T11_3/`: bundle `analysis_results`

**Notes.** Manifest and seed map only; written by T11_2_recipe/select_recipe.py --write-t113-manifest.
