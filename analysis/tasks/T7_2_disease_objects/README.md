<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T7_2_disease_objects

Builds the annotated objects of the liver MASH case study (human and mouse, healthy and disease) from the downloads of T7_1_holdouts.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Table S18 (`tab:data-preparation`); Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T7_2_disease_objects.build [--force]

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/disease/`: bundle `data_other_cohorts`
- `results/analysis/T7_2/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/disease/`
- `data/analysis/ensembl110/`
- `data/processed/`
- `data/raw/`
