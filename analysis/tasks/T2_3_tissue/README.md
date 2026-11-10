<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_3_tissue

Compares Δα between liver and testis for the matched species pairs, with genes joined on the human Ensembl ID, against the clamped and the raw prior.

| | |
|---|---|
| Question | Δα as an axis of its own: Is deployment relative to the protein prior separate from protein sequence? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.2 (`sec:axis`); Figure 4f (`fig:dalpha-biology`); Supplementary Section C.21.3 (`sssec:rev-identity`); Supplementary Section D.2 (`ssec:er-axis`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_3_tissue.run [--force] (or bash analysis/tasks/run_T2_3_to_T2_7.sh for T2_3 to T2_7)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_3/`: bundle `analysis_results`
