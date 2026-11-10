<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# S1_figures

Helper that removes the panel letters from one page of an assembled figure page, so that the layout tool can draw them.

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | infrastructure |
| Environment | `chimera_env` |
| Used in the paper | no |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.S1_figures.strip_page_labels

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/S1/`: bundle `analysis_results`

**Notes.** Used only for pages that are still placed whole (figures/sources/).
