<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T5_2_adapters

Adapters that run SATURN (its native genes, and the cohort's one-to-one orthologs) and CAMEX on the Chimera cohorts, with a pilot that checks the tools against their published scores.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `saturn_env`, `camex_env`, `chimera_env` |
| Used in the paper | Supplementary Section C.21.9 (`sssec:rev-benchmark`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch analysis/tasks/T5_2_adapters/tool.sbatch <saturn|camex> <stem> <labels|-> <seed> <work> <dest>
    python -m analysis.tasks.T5_2_adapters.pilot_metrics

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/T5_inputs/`: not deposited (rebuilt by the task code)
- `results/analysis/T5_2/`: bundle `tool_runs_saturn_camex`

**Reads** (as named in the code)

- `data/analysis/T5_inputs/`
- `data/analysis/disease_kidney/`
- `data/processed/`
- `data/raw/`
- `training/runs/`

**Notes.** cohorts.py is the cohort table of the adapters; prep_*.py build tool inputs, run_*.py run the tools; _tool_body.sh is the shared job body.
