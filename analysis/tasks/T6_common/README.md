<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_common

Shared inputs of the functional-layer tests: residual readouts, strata and gene identifiers.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.11 (`sssec:rev-functional`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    (imported by T6_3_pathway, T6_4_network, T6_6_regulatory, T6_9_assemble)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T6_3/cache/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T3_3_consensus`
- `data/analysis/ensembl110/`
- `data/analysis/gold/`
- code of `T6_1_functional`

**Notes.** A module, not a task: functional.py.
