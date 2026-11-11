<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_5_count_perm

Retraining control that keeps the features and shuffles the counts across genes, with a projection null that needs no training.

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:reliability`); Supplementary Figure S62c (`fig:supp-main-gin-esm2`); Supplementary Section D.3 (`ssec:er-validation`); Supplementary Section C.21.7 (`sssec:rev-controls`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T3_5_count_perm.make_control_h5ad
    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T3_5_count_perm/manifest.tsv
    python -m analysis.tasks.T3_5_count_perm.offline_null
    python -m analysis.tasks.T3_5_count_perm.analyze

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/controls/`: bundle `data_other_cohorts`
- `training/runs/rev-ctl-<cohort>-perm<A|B|C>-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T3_5/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/controls/`

**Notes.** controls.tsv lists the control objects.
