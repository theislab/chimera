<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_6_vz_glaucoma

Eye cohort: where glaucoma genes sit on Δα and the count evidence, and the per-species residual by cell class.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:tissues`); Supplementary Section C.21.13 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T11_6_vz_glaucoma.glaucoma
    python -m analysis.tasks.T11_6_vz_glaucoma.residuals [--threads 12]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T11/glaucoma/`: bundle `analysis_results`
- `results/analysis/T11/residuals/`: bundle `analysis_results`

**Reads** (as named in the code)

- `training/runs/`

**Notes.** Writes under results/analysis/T11/.
