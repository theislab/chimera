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
    python -m analysis.tasks.T2_7_consistency.run --values-only

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_7/values.csv`: bundle `analysis_results`
- `results/analysis/T2_7/attribution_candidates_vs_rule.csv`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/processed/`
- `data/raw/`
- `training/runs/`

**Notes.** values.csv has one row per measured value: item, measured_value, source (the object, run file or code line it is read from). It is the deposited table and the source of the numbers in the two tables of the Supplement. The task writes it together with consistency_values.csv, a working table that sets each measured value beside the value stated in the tables when the task was run, with the place of the statement; that table is not deposited. The second command writes values.csv from a stored consistency_values.csv and measures nothing again. attribution_candidates_vs_rule.csv lists the attribution candidates with the flag that the ortholog rule gives each.
