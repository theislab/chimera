<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_4_covariates

Covariate audit of Δα in the seven canonical cohorts: expression level, detection, specificity, coding length and GC, how much features alone predict, and which genes carry no count evidence (masked).

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:reliability`); Supplementary Figure S25 (`fig:supp-covariate-audit`); Figure 4e (`fig:dalpha-biology`); Supplementary Section C.21.4 (`sssec:rev-covariates`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_4_covariates.run [--n-perm 1000]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_4/`: bundle `analysis_results`
