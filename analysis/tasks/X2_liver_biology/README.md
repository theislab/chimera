<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X2_liver_biology

Liver: gene sets, hepatocyte zonation and the hepatocyte catabolic genes with conserved proteins and Δα below the prior, on per-gene liver tables and zone-resolved forward passes.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Table S18 (`tab:data-preparation`); Supplementary Section C.21.11 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X2_liver_biology.sets
    python -m analysis.tasks.X2_liver_biology.forward --label liv_4sp
    python -m analysis.tasks.X2_liver_biology.build
    python -m analysis.tasks.X2_liver_biology.zonation
    python -m analysis.tasks.X2_liver_biology.analyses
    python -m analysis.tasks.X2_liver_biology.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X2/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T3_3_consensus`
- `data/processed/`
- `data/raw/`
- `training/runs/`

**Notes.** Labeled exploratory in the paper.
