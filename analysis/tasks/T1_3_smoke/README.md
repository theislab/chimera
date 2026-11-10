<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T1_3_smoke

Environment check of the authors' cluster: freezes the environment into the lock file, imports the dependencies, runs the tests and reloads the 11 canonical checkpoints on CPU against their stored α.

| | |
|---|---|
| Question | Infrastructure: Environment and checkpoint checks. |
| Status | infrastructure |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.3 (`sssec:rev-identity`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T1_3_smoke/smoke.sh (rewrites analysis/envs/chimera_env.linux-64.lock.txt)
    python -m analysis.tasks.T1_3_smoke.checkpoints <out.tsv>

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T1_3/`: bundle `analysis_results`
- `training/runs/rev-calib-*`: bundle `models_seeds_controls_cohorts`

**Reads** (as named in the code)

- `data/processed/`
- `training/runs/`

**Notes.** A maintainer tool: smoke.sh overwrites the lock file with the packages of the active environment. To check a checkout use analysis/reproduce_smoke.sh instead. calib_manifest.tsv holds two timing runs.
