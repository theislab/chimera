<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X7_architecture_outputs

Tests outputs that only this architecture produces (per-cell residual share, spread of the residual across species, species-resolved margins in a joint fit) against independent per-type divergence and against baselines from the same counts.

| | |
|---|---|
| Question | What the count evidence adds: What do the readouts add to direct baselines and to other tools, on real and on ground-truth data? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.11 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X7_architecture_outputs.forward --runs <labels>
    python -m analysis.tasks.X7_architecture_outputs.golds
    python -m analysis.tasks.X7_architecture_outputs.baselines
    python -m analysis.tasks.X7_architecture_outputs.analyses
    python -m analysis.tasks.X7_architecture_outputs.posthoc
    python -m analysis.tasks.X7_architecture_outputs.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X7/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T2_5_celltype_margin`, `T4_3_scoreboard`
- `data/raw/`
- `training/runs/`
- code of `X4_where_differs`

**Notes.** Labeled exploratory in the paper; the outcome is stated in the Discussion. Definitions in rules.py.
