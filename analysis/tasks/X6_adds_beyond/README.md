<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X6_adds_beyond

What each score adds once all simpler scores are known (incremental information), and how label-dependent direct scores degrade when cell-type labels are noisy, coarse or replaced by clusters.

| | |
|---|---|
| Question | What the count evidence adds: What do the readouts add to direct baselines and to other tools, on real and on ground-truth data? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:adds`); Section 3.4 (`sec:validation`); Supplementary Figure S26d (`fig:supp-validation-tests`); Table 1 (`tab:when-to-use`); Supplementary Section C.21.11 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X6_adds_beyond.degrade
    python -m analysis.tasks.X6_adds_beyond.analyses
    python -m analysis.tasks.X6_adds_beyond.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X6/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T4_3_scoreboard`, `X4_where_differs`
- code of `X4_where_differs`

**Notes.** Labeled exploratory in the paper. Definitions in rules.py.
