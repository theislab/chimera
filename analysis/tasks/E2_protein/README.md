<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E2_protein

Protein architecture and gene history: Pfam domains, protein families, complexes, gene age and duplication class against Δα on 26 cohorts and both gates.

| | |
|---|---|
| Question | One reading across tissues: Does the reading hold across tissues, and what goes with deployment above the prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Figure 6k (`fig:one-reading`); Supplementary Section D.9 (`ssec:er-e2`); Supplementary Figure S51 (`fig:supp-e2`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/E2_protein/fetch.sh
    python -I analysis/tasks/E2_protein/annotate.py
    python -m analysis.tasks.E2_protein.analyse --workers <n>
    python -m analysis.tasks.E2_protein.summarise
    python -m analysis.tasks.E2_protein.extras
    python -m analysis.tasks.E2_protein.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E2/`: bundle `analysis_results`
- `data/analysis/E2/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Y5_aspects`, `Z4_free_gate_all`
- `data/analysis/E2/`
- `data/analysis/Y2/`
- `data/analysis/ensembl110/`
- `data/analysis/functional/`
- `data/analysis/gold/`
- `data/analysis/x3_annotations/`
- code of `F1_figures`

**Notes.** Labeled exploratory in the paper. Definitions in rules.py.
