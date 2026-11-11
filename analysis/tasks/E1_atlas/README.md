<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E1_atlas

Atlas of gene programs across tissues: set statistics per cohort and readout for both gates, recurrence across eight tissues, dependence on evolutionary distance, and recurrent named genes.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:tissues`); Supplementary Section D.7 (`ssec:er-e1`); Supplementary Figure S39 (`fig:supp-e1`); Supplementary Figure S40 (`fig:supp-e1-cont`); Supplementary Figure S41 (`fig:supp-e1-genes`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/E1_atlas/run_all.sh

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E1/`: bundle `analysis_results`
- `data/analysis/E1/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Z4_free_gate_all`
- `data/analysis/E1/`
- code of `H1_harmonise`

**Notes.** Labeled exploratory in the paper. Definitions and predictions in rules.py.
