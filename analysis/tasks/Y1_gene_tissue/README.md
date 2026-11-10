<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y1_gene_tissue

Gene by tissue: how much of each readout belongs to the gene and how much to the gene in a tissue, whether tissue differences within a gene follow independent atlas differences, and worked genes.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Figure 6d, e, f, g, h, i (`fig:one-reading`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -I analysis/tasks/Y1_gene_tissue/launch.py analysis.tasks.Y1_gene_tissue.<module> [args], in the order atlas_gold, forward, build, prep, describe, variance, contrast, classes, worked, figure

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y1/`: bundle `analysis_results`
- `data/analysis/Y1/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- `data/analysis/disease_islet/`
- `data/analysis/disease_kidney/`
- `data/analysis/gold/`
- `data/analysis/van_zyl/`
- `data/analysis/x3_annotations/`
- `data/processed/`
- `training/runs/`

**Notes.** Labeled exploratory in the paper. launch.py runs a module in isolated mode because downloaded objects are read. Definitions in rules.py; verify.py recomputes headline numbers with independent code.
