<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_8_priorswap

Retraining control with a smoothed protein prior (half the prior, half its average over protein-interaction partners), compared with same-seed runs and with the seed floor.

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.11 (`sssec:rev-functional`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T6_8_priorswap.make_control_h5ad
    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T6_8_priorswap/manifest.tsv
    python -m analysis.tasks.T6_8_priorswap.analyze

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/controls/`: bundle `data_other_cohorts`
- `training/runs/rev-ctl-liv_hmouse-priorswap-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T6_8/`: bundle `analysis_results`

**Notes.** Stated in the Discussion together with the same-seed repeats of T6_8b_repeat.
