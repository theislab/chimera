<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_5_vz_genes

Eye cohort: per-gene tables, covariate audit, direct baselines and cell-type contributions to the count evidence for the six eye data sets.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:tissues`); Supplementary Section C.21.13 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T11_5_vz_genes.forward --missing
    python -m analysis.tasks.T11_5_vz_genes.readout [--datasets full,hmouse]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T11/readout/`: bundle `analysis_results`
- `results/analysis/T11/genes/`: bundle `analysis_results`
- `results/analysis/T11/drivers/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T4_3_scoreboard`
- `training/runs/`

**Notes.** Writes under results/analysis/T11/. vz.py is the data-set registry of T11_4 to T11_7.
