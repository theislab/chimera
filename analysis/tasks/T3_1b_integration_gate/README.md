<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T3_1b_integration_gate

Scores the integration checks E8.1 to E8.6 of training/compare_vs_scvi.py on every seed run, which separates integrated from poorly mixed training states.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.2 (`sssec:rev-seeds`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T3_1b_integration_gate.run [--outdir results/analysis/T3_1b]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T3_1b/`: bundle `analysis_results`

**Reads** (as named in the code)

- `training/runs/`
