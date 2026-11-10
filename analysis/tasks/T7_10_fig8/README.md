<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T7_10_fig8

Draws the panels of the disease figure from the result tables of the kidney, islet and liver case studies.

| | |
|---|---|
| Question | Disease: Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Figure 7 (`fig:disease`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T7_10_fig8.fig8_disease

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T7_10/`: bundle `analysis_results`
- `figures/panels/`: tracked in git

**Notes.** The folder name keeps an earlier figure number; the figure is fig:disease.
