<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T5_3_runs

Cell-level runs of SATURN, SATURN with Leiden clusters in place of labels, and CAMEX on the seven canonical cohorts, with a check of every latent.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `saturn_env`, `camex_env`, revision_de_env for check_outputs.py |
| Used in the paper | Section 3.1 (`sec:model`); Figure 3 (`fig:integration`); Supplementary Section C.21.9 (`sssec:rev-benchmark`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=0-<rows-1> analysis/tasks/T5_3_runs/tool_array.sbatch analysis/tasks/T5_3_runs/saturn.tsv (likewise saturn_leiden.tsv and camex.tsv)
    python analysis/tasks/T5_3_runs/check_outputs.py

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T5_3/`: bundle `tool_runs_saturn_camex`

**Reads** (as named in the code)

- `data/processed/`
