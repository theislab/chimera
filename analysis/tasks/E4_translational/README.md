<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E4_translational

Drug targets and disease genes: where they sit relative to the protein prior in their own tissue, on both gates, and whether placement relates to how well the mouse model of the gene matches the human disease.

| | |
|---|---|
| Question | Disease: Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.8 (`sec:disease`); Supplementary Section D.12 (`ssec:er-e4`); Supplementary Figure S53 (`fig:supp-e4`); Supplementary Figure S54 (`fig:supp-e4-targets`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/E4_translational/run_all.sh

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E4/`: bundle `analysis_results`
- `data/analysis/E4/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `F1_figures`
- `data/analysis/E4/`
- `data/analysis/Y2/`
- code of `E1_atlas`, `F1_figures`, `H1_harmonise`

**Notes.** Labeled exploratory in the paper. Definitions and predictions in rules.py; q1_followup.py is post hoc and labeled so.
