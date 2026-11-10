<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T4_2_gene_baselines

Computes the direct baselines for every cohort from the same counts: pseudobulk profile correlation, rank-matched level per cell type, profile correlation over clusters of Chimera's latent, specificity similarity, and predictions from gene features alone.

| | |
|---|---|
| Question | Pre-specified validation: Do independent data back Δα and the count evidence S_g, and how do direct baselines compare? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:validation`); Table 1 (`tab:validation`); Figure 5c (`fig:validation`); Supplementary Section C.21.8 (`sssec:rev-validation`) |
| How we know | read by tasks whose results are in the paper |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T4_2_gene_baselines.run [--cohorts all] [--delta-alpha canonical]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T4_2/`: bundle `analysis_results`

**Notes.** Read by T4_3_scoreboard and by most later analyses. Column names (B1b, B1c, B1z, B2b, B3a, B3b) are explained in the docstring of run.py.
