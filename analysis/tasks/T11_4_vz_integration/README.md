<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_4_vz_integration

Integration scorecard of the eye data sets, reported as rows of their own beside the seven canonical cohorts.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:tissues`); Supplementary Figure S10 (`fig:supp-integration-ext`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/tasks/T11_4_vz_integration/cmd_array.sbatch analysis/tasks/T11_4_vz_integration/baselines.tsv
    python -m analysis.tasks.T11_4_vz_integration.score

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T11/integration/`: bundle `analysis_results`
- `training/runs/baselines_<method>/van_zyl_*`: bundle `notebook_caches`

**Reads** (as named in the code)

- `training/runs/`

**Notes.** Writes under results/analysis/T11/. baselines.tsv lists the baseline fits (PCA, Harmony, scVI, scANVI, sysVI through training/baseline_runner.py).
