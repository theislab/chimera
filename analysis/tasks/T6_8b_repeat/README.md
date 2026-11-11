<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_8b_repeat

Trains liver Human+Mouse seeds 46 and 47 a second time with unchanged arguments and compares the repeat, the original and the prior-swap run of each seed, to measure how much two runs with the same seed differ on CPU.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.2 (`sssec:rev-seeds`); Supplementary Section C.21.10 (`sssec:rev-functional`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T6_8b_repeat/manifest.tsv
    python -m analysis.tasks.T6_8b_repeat.compare

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-rep-liv_hmouse-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T6_8b/`: bundle `analysis_results`
- `results/analysis/T6_8b_repeat.csv`: bundle `analysis_results`

**Notes.** compare.py writes results/analysis/T6_8b_repeat.csv from the three archived runs of each seed: Spearman correlation across genes of the correction, of Δα and of S_g between repeat, original and prior-swap run, and of the change of the correction against the change of the prior. It needs the prior-swap runs of T6_8_priorswap. With --check it recomputes the table and compares it with the stored one without writing. results/analysis/T6_8b/ holds the job logs of the two repeat runs. The result is also stated in the Discussion.
