<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T5_5_benchmark

Scores SATURN, SATURN-Leiden and CAMEX with the metrics of the seven-method panel, giving the ten-method integration scorecard of the seven canonical cohorts.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.1 (`sec:model`); Figure 3 (`fig:integration`); Supplementary Figure S8 (`fig:supp-integration-purity`); Supplementary Figure S9 (`fig:supp-integration-raw`); Supplementary Section D.1 (`ssec:er-model`); Supplementary Section C.19.2 (`sssec:rev-benchmark`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T5_5_benchmark.score_external
    python -m analysis.tasks.T5_5_benchmark.views

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `notebooks/cache/baselines_master_ext/`: bundle `notebook_caches`
- `notebooks/figures/baselines_master_ext/`: tracked in git

**Reads** (as named in the code)

- `data/processed/`
- `notebooks/cache/`
- `training/runs/`

**Notes.** Writes to the notebook cache, not to results/analysis/. master_table_ext.csv is the integration benchmark of the paper: the seven methods of notebooks/07_baselines_master.ipynb with their raw metrics unchanged (checked against master_table.csv at every run) and the three tools added, with the scaled scores and the composite recomputed over the ten methods. The seven-method composite of the notebook is therefore not the composite of the ten-method table.
