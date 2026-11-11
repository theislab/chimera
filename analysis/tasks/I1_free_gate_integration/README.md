<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# I1_free_gate_integration

Compares cell-level integration of the per-gene gate and the protein-feature gate with the metrics of the integration benchmark, on every cohort.

| | |
|---|---|
| Question | The per-gene gate: What does Δα show when it is read gene by gene from counts, without the protein-feature network? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:per-gene`); Supplementary Table S23 (`tab:integration-per-gene-gate`); Supplementary Table S24 (`tab:integration-per-gene-gate-harmonised`); Supplementary Section D.1 (`ssec:er-model`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.I1_free_gate_integration.score check
    python -m analysis.tasks.I1_free_gate_integration.score unit <cohort> <run>
    python -m analysis.tasks.I1_free_gate_integration.summarise
    python -m analysis.tasks.I1_free_gate_integration.tables
    python -m analysis.tasks.I1_free_gate_integration.replication test

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/I1/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `Z3_free_gate`
- `notebooks/cache/`
- `training/runs/`

**Notes.** tables.py writes the two tables as CSV and as LaTeX into results/analysis/I1/. mechanism.py is labeled exploratory. Definitions in rules.py.
