<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_7_vz_figure

Eye cohort: the tidy summary table of all eye data sets and a figure of the eye results.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T11_7_vz_figure/run_all.sh

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T11/summary.csv`: bundle `analysis_results`
- `results/analysis/T11/figures/`: bundle `analysis_results`

**Notes.** Writes under results/analysis/T11/. The summary table is read by H1_harmonise; the figure is not in the paper (the eye figure of the paper is drawn by H1_harmonise).
