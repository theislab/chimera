<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# E6_known_differences

Tests whether the readouts recover species differences in cell-type gene expression that the literature documents, list by list.

| | |
|---|---|
| Question | What the count evidence adds: What do the readouts add to direct baselines and to other tools, on real and on ground-truth data? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.6 (`sec:adds`); Supplementary Section D.5 (`ssec:er-e6`); Supplementary Figure S30 (`fig:supp-e6`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.E6_known_differences.known
    python -m analysis.tasks.E6_known_differences.summarize
    python -m analysis.tasks.E6_known_differences.named
    python -m analysis.tasks.E6_known_differences.posthoc
    python -m analysis.tasks.E6_known_differences.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/E6/`: bundle `analysis_results`
- `data/analysis/E6/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `H1_harmonise`, `Z4_free_gate_all`
- `data/analysis/E6/`
- `data/analysis/bone_marrow/`
- `data/analysis/disease_kidney/`
- `data/analysis/gold/`
- `data/analysis/new_usecases/`
- `data/analysis/van_zyl/`
- `data/processed/`

**Notes.** Labeled exploratory in the paper. lists/ holds the gene lists derived from the supplementary tables of each publication, derive/ the script and note per publication; assign.tsv maps lists to cohorts. Definitions in rules.py.
