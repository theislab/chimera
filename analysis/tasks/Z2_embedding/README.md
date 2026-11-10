<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Z2_embedding

Asks how well Chimera's latent resolves cell states inside a cell-type label, beside scVI on the same cells, and what sets it (depth, capacity, MMD conditioning), with a factorial of training variants.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.1 (`sec:model`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.Z2_embedding.measure prep <cohort>
    python -m analysis.tasks.Z2_embedding.measure cohort <cohort>
    python -m analysis.tasks.Z2_embedding.make_factorial
    sbatch --array=<rows> analysis/tasks/Z2_embedding/train.sbatch
    python -m analysis.tasks.Z2_embedding.post_factorial
    python -m analysis.tasks.Z2_embedding.tables
    python -m analysis.tasks.Z2_embedding.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Z2/`: bundle `analysis_results`
- `training/runs/z2-*`: not deposited (the tables computed from these runs are)

**Reads** (as named in the code)

- results of `B2_benchmark`, `H1_harmonise`, `T4_3_scoreboard`, `T5_6_gene_level`, `X2_liver_biology`, `Y5_aspects`
- `data/analysis/bone_marrow/`
- `data/analysis/disease_islet/`
- `data/analysis/disease_kidney/`
- `data/analysis/gold/`
- `data/analysis/van_zyl/`
- `data/processed/`
- `training/runs/`
- code of `F1_figures`

**Notes.** The paper states the outcome in one sentence of Results (within-label resolution against scVI) and in the Discussion. Definitions in rules.py. The z2-* runs are not deposited; their tables are.
