<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X5_sequence_axis

Sets protein identity, dN/dS and other known gene measures beside α, Δα and the count evidence on the same genes, cohort by cohort.

| | |
|---|---|
| Question | Δα as an axis of its own: Is deployment relative to the protein prior separate from protein sequence? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.2 (`sec:axis`); Figure 4g (`fig:dalpha-biology`); Supplementary Figure S13 (`fig:supp-sequence-axis`); Supplementary Figure S14 (`fig:supp-known-measures`); Supplementary Section D.2 (`ssec:er-axis`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X5_sequence_axis.run
    python -m analysis.tasks.X5_sequence_axis.known

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X5/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/x3_annotations/`
