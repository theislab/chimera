<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E3_cells

Cell types and lineages: genes grouped by their home compartment, their excess over matched genes on both gates and on direct baselines; species-specific departures in fits with three or more species; replication in independent primate data.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Figure 6j (`fig:one-reading`); Supplementary Section D.8 (`ssec:er-e3`); Supplementary Figure S48 (`fig:supp-e3-compartments`); Supplementary Figure S49 (`fig:supp-e3-lineages`); Supplementary Figure S50 (`fig:supp-e3-primates`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.E3_cells.build
    python -m analysis.tasks.E3_cells.forward_free manifest
    sbatch --array=<rows> analysis/tasks/E3_cells/forward_free.sbatch
    python -m analysis.tasks.E3_cells.compartments
    python -m analysis.tasks.E3_cells.departures
    python -m analysis.tasks.E3_cells.evo
    python -m analysis.tasks.E3_cells.named
    python -m analysis.tasks.E3_cells.robust
    python -m analysis.tasks.E3_cells.figure

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E3/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Z4_free_gate_all`

**Notes.** Labeled exploratory in the paper. Definitions in rules.py.
