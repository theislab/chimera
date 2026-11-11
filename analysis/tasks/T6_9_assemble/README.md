<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_9_assemble

Applies the multiple-testing family of the functional layer over its seven tests and draws the supplementary functional-layer figure.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Supplementary Figure S24 (`fig:supp-functional-layer`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Section C.21.10 (`sssec:rev-functional`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T6_9_assemble.assemble <output pdf>

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T6_9/`: bundle `analysis_results`
