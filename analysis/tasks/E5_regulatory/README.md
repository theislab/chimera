<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E5_regulatory

Regulatory input and gene architecture: enhancer number, promoter class, regulons and neighborhood against Δα on 26 cohorts and both gates.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Figure 6l (`fig:one-reading`); Supplementary Section D.10 (`ssec:er-e5`); Supplementary Figure S52 (`fig:supp-e5`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/E5_regulatory/fetch.sh
    python -I analysis/tasks/E5_regulatory/annotate.py
    python -m analysis.tasks.E5_regulatory.analyse --workers <n>
    python -m analysis.tasks.E5_regulatory.summarise
    python -m analysis.tasks.E5_regulatory.neighbour --workers <n>
    python -m analysis.tasks.E5_regulatory.extras --workers <n>
    python -m analysis.tasks.E5_regulatory.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E5/`: bundle `analysis_results`
- `data/analysis/E5/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `E2_protein`, `Y2_functional`
- `data/analysis/E5/`
- `data/analysis/Y2/`
- `data/analysis/ensembl110/`
- `data/analysis/x3_annotations/`
- code of `E2_protein`

**Notes.** Labeled exploratory in the paper. Uses the loader and scan statistic of E2_protein. Definitions in rules.py.
