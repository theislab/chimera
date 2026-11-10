<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_3_consensus

Applies the seed-pooling rule and builds the seed-consensus per-gene tables of the seven canonical cohorts with their seed reliability.

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:reliability`); Supplementary Figure S26 (`fig:supp-seed-reliability`); Supplementary Section D.3 (`ssec:er-validation`); Supplementary Section C.21.2 (`sssec:rev-seeds`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T3_3_consensus.run [--cohorts ...]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T3_3/`: bundle `analysis_results`

**Notes.** The consensus tables of the deposit (results/gene_tables/) are built from this folder's outputs by the release tool.
