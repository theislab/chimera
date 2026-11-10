<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_4_network

Functional layer: Moran's I of Δα on protein-interaction graphs and retention of interologs.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Supplementary Figure S24 (`fig:supp-functional-layer`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Section C.21.11 (`sssec:rev-functional`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T6_4_network.run [--only F2|F3] [--procs 6]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T6_4/`: bundle `analysis_results`

**Reads** (as named in the code)

- code of `T6_1_functional`
