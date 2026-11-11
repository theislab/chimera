<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Z4_free_gate_all

Trains the per-gene gate on every one of the 26 cohorts with the cohort's own recipe, keeps seeds by a health rule, and builds the per-gene tables and reference comparisons in the schema of H1_harmonise.

| | |
|---|---|
| Question | The per-gene gate: What does Δα show when it is read gene by gene from counts, without the protein-feature network? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:tissues`); Supplementary Section D.12 (`sec:disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.Z4_free_gate_all.make_manifest
    sbatch --array=<rows> analysis/tasks/Z4_free_gate_all/train.sbatch
    python -m analysis.tasks.Z4_free_gate_all.health
    python -m analysis.tasks.Z4_free_gate_all.readouts
    python -m analysis.tasks.Z4_free_gate_all.references trackc|lists|cortex|cases|summary
    python -m analysis.tasks.Z4_free_gate_all.summary

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/z4-<cohort>-m-s<seed>/`: bundle `models_per_gene_gate`
- `results/analysis/Z4/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `H1_harmonise`
- `data/processed/`
- `training/runs/`

**Notes.** results/analysis/Z4/h1/genes/<cohort>.parquet are the per-gene tables of the per-gene gate, checked by analysis/check_gene_tables.py; kept.json lists the kept seeds.
