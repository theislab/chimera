<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# F1_figures

Redraws, at print size and from the same data, panels that were first made by the notebooks 04 and 06 and by task scripts.

| | |
|---|---|
| Question | Figures and tables: Which scripts draw panels or measure table values from cached results? |
| Status | infrastructure |
| Environment | `chimera_env` |
| Used in the paper | Figure 4 (`fig:dalpha-biology`); Supplementary Figure S21 (`fig:supp-dalpha-hallmark`) |
| How we know | a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.F1_figures.notebook_panels
    python -m analysis.tasks.F1_figures.regen_supp

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `figures/panels/`: tracked in git
- `results/analysis/F1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `notebooks/cache/`
- `training/runs/`

**Notes.** compose.py is the former layout tool; figures are now composed by figures/compose.py (see figures/README.md).
