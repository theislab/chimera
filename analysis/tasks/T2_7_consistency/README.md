<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_7_consistency

Measures, from the archived runs, the values that the recipe and assumption tables of the Supplement print (gene counts, shared cell types, training outcomes per cohort).

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Table S15 (`tab:per-dataset-recipes`); Supplementary Table S3 (`tab:assumptions-summary`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_7_consistency.run

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_7/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/processed/`
- `data/raw/`
- `training/runs/`
