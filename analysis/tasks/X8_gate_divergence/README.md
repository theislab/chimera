<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X8_gate_divergence

Where the two gates place a gene differently (the gate gap): its reproducibility, its recurrence across tissues, gene classes and protein neighborhoods, its relation to independent references, and named examples.

| | |
|---|---|
| Question | The per-gene gate: What does Δα show when it is read gene by gene from counts, without the protein-feature network? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section D.4 (`sec:supp-gate-gap`); Section 3.5 (`sec:per-gene`); Supplementary Figure S27 (`fig:supp-gate-gap`); Supplementary Figure S28 (`fig:supp-gate-gap-cont`); Supplementary Figure S29 (`fig:supp-gate-gap-named`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X8_gate_divergence.gap
    python -m analysis.tasks.X8_gate_divergence.recur
    python -m analysis.tasks.X8_gate_divergence.sets
    python -m analysis.tasks.X8_gate_divergence.nbr
    python -m analysis.tasks.X8_gate_divergence.golds trackc|cortex
    python -m analysis.tasks.X8_gate_divergence.named
    python -m analysis.tasks.X8_gate_divergence.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X8/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Z4_free_gate_all`
- code of `H1_harmonise`

**Notes.** Definitions in rules.py.
