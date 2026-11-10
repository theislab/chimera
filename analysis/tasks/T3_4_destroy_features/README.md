<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_4_destroy_features

Retraining control that keeps the counts and removes, randomizes or zeroes the gene features.

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:reliability`); Figure 8c (`fig:gin-esm2`); Supplementary Section D.3 (`ssec:er-validation`); Supplementary Section C.21.7 (`sssec:rev-controls`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_4_destroy_features/manifest.tsv
    python -m analysis.tasks.T3_4_destroy_features.analyze

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-ctl-<cohort>-<arm>-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T3_4/`: bundle `analysis_results`

**Reads** (as named in the code)

- `training/runs/`

**Notes.** Arms: rand (random features), zesm and zexpr (ESM-2 or expression features zeroed), m (no feature network: the per-gene gate). The arms are training options on the canonical objects; the shared statistics are in analysis/common/controls.py.
