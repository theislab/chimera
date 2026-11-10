<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T12_3_tests

Bone marrow: seed health, per-gene table, the tests of Δα and the count evidence against the conservation of expression of Shay et al. 2013, cell-type contributions, and the supplementary figure.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Supplementary Figure S47 (`fig:supp-bone-marrow`); Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T12_3_tests.seed_health
    python -m analysis.tasks.T12_3_tests.readout
    python -m analysis.tasks.T12_3_tests.tests
    python -m analysis.tasks.T12_3_tests.drivers --forward 42
    python -m analysis.tasks.T12_3_tests.drivers --combine
    python -m analysis.tasks.T12_3_tests.figure

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/P12/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/bone_marrow/`
- `training/runs/`

**Notes.** Writes to results/analysis/P12/, not T12_3. explore_e2.py is labeled exploratory.
