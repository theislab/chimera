<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_8b_repeat

Trains liver Human+Mouse seeds 46 and 47 a second time with unchanged arguments, to measure how much two runs with the same seed differ on CPU.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.2 (`sssec:rev-seeds`); Supplementary Section C.21.11 (`sssec:rev-functional`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/T6_8b_repeat/manifest.tsv

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/rev-rep-liv_hmouse-s<seed>/`: bundle `models_seeds_controls_cohorts`
- `results/analysis/T6_8b/`: bundle `analysis_results`
- `results/analysis/T6_8b_repeat.csv`: bundle `analysis_results`

**Notes.** Manifest only. The result is also stated in the Discussion. No tracked script writes the comparison table results/analysis/T6_8b_repeat.csv (not established which command produced it); it holds the Spearman correlations of repeat against original run.
