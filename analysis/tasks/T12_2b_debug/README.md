<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T12_2b_debug

Diagnostic objects and a probe that locates the first non-finite tensor in a bone-marrow training run. Training health only; no readout.

| | |
|---|---|
| Question | Model fits beyond the canonical runs: Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.12.2 (`par:latent-guard`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T12_2b_debug.make_debug_objects --rule prop --n 25000
    python -m analysis.tasks.T12_2b_debug.nan_probe <run_experiment.py arguments>

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/bone_marrow/debug/`: bundle `data_other_cohorts`
- `training/runs/rev-bm-dbg-*`: bundle `models_seeds_controls_cohorts`

**Reads** (as named in the code)

- `data/analysis/bone_marrow/`
- `training/runs/`

**Notes.** The finding (runaway of the posterior log-variance) led to the latent guard tested in G1_guard.
