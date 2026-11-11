<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# F2_per_gene

Builds the per-gene tables behind the per-gene-gate panels and draws them: the readouts against the four independent references and the named findings of the main validation figure, and the five panels of the supplementary figure on the two gates.

| | |
|---|---|
| Question | The per-gene gate: What does Δα show when it is read gene by gene from counts, without the protein-feature network? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Figure 6d, e (`fig:validation`); Supplementary Figure S27a, b, c, d, e (`fig:supp-gates`); Section 3.4 (`sec:per-gene`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.F2_per_gene.data
    python -m analysis.tasks.F2_per_gene.panels

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/F2/`: bundle `analysis_results`
- `figures/panels/`: tracked in git

**Reads** (as named in the code)

- results of `Z3_free_gate`

**Notes.** Uses the statistics of Z3_free_gate; no new statistic. Single panels of the main figure: panels.py --panel <family/name>.
