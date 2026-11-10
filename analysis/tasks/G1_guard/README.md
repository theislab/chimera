<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# G1_guard

Tests the latent guard (--clamp-z-logvar): scans the posterior log-variance of archived runs, shows that the option leaves guarded-off fits bit-identical, and counts finished and non-finite fits with and without it.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.12.2 (`par:latent-guard`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.G1_guard.step1_scan --runs analysis/tasks/G1_guard/step1_runs.tsv --out results/analysis/G1
    python -m analysis.tasks.G1_guard.step1_table
    sbatch analysis/tasks/G1_guard/bitproof.sbatch <tree before> <tree after>
    sbatch --array=<rows> analysis/tasks/G1_guard/step3.sbatch
    python -m analysis.tasks.G1_guard.nan_rate
    python -m analysis.tasks.G1_guard.step3_dcompare

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/G1/`: bundle `analysis_results`
- `training/runs/rev-g1-*`: bundle `models_seeds_controls_cohorts`
- `training/runs/rev-g1u-*`: bundle `models_seeds_controls_cohorts`
- `training/runs/g1-*`: bundle `models_seeds_controls_cohorts`

**Reads** (as named in the code)

- results of `B2_benchmark`, `D1_sparse_genes`, `H1_harmonise`
- `data/analysis/G1/`
- `data/processed/`
- `training/runs/`
- code of `T12_2b_recipe`

**Notes.** The guard is off by default and in every canonical model. seedfix.sbatch shows that --seed also fixes the train/validation split; upper_only.py and check_upper_equiv.py test a one-sided variant that was not adopted; nan_watch.py is a cluster helper.
